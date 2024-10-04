import Combinatorics
import Data.Rat.Basic
import Mathlib
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Group.Pi
import Mathlib.Algebra.Order.AbsoluteValue
import Mathlib.Algebra.Polynomial
import Mathlib.Algebra.Ring.Basic
import Mathlib.Algebra.Trigonometry
import Mathlib.Analysis.Calculus
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Calculus.Fderiv.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Finset.Card
import Mathlib.Data.Int.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Polynomial.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Sqrt
import Mathlib.Data.Set.Basic
import Mathlib.Data.Zmod.Basic
import Mathlib.Geometry.Euclidean.Circle.Basic
import Mathlib.MeasureTheory.Integral.SetIntegral
import Mathlib.NumberTheory.Basic
import Mathlib.Probability.Basic
import Mathlib.ProbabilityTheory.Bernoulli
import Mathlib.ProbabilityTheory.Independence
import Mathlib.ProbabilityTheory.Poisson
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring
import ProbabilityTheory

namespace probability_even_sum_l158_158267

-- Definition of the prime numbers set and the selection scenario
def firstTenPrimes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Condition stating that we select two distinct numbers at random from the first ten primes
def randomSelection (s : List ℕ) := {x : ℕ × ℕ // x.1 ≠ x.2 ∧ x.1 ∈ s ∧ x.2 ∈ s}

-- Define the event that their sum is even
def evenSum (p : ℕ × ℕ) := (p.1 + p.2) % 2 = 0

-- Define the probability calculation
def probabilityEvenSum := 
  let totalPairs := (firstTenPrimes.length.choose 2) -- Calculate the number of ways to choose 2 numbers from the 10 primes
  let evenPairs := (randomSelection firstTenPrimes).count (λ p => evenSum p.val)
  evenPairs.toFloat / totalPairs.toFloat

-- Theorem statement that the probability of their sum being even is 1/5
theorem probability_even_sum : probabilityEvenSum = 1/5 := by
  sorry -- The actual proof is to be provided

end probability_even_sum_l158_158267


namespace find_k_of_perpendicular_vec_l158_158606

variable (k : ℝ)

noncomputable def a : ℝ × ℝ := (1, 2)
noncomputable def b : ℝ × ℝ := (2, k)
noncomputable def sum_vec : ℝ × ℝ := (2 * a.1 + b.1, 2 * a.2 + b.2)

theorem find_k_of_perpendicular_vec
  (h : (sum_vec k) • a = 0) : k = -6 := by
sorry

end find_k_of_perpendicular_vec_l158_158606


namespace f_2005_eq_11_l158_158613

noncomputable def digit_sum (n : ℕ) : ℕ :=
  (to_string n).to_list.map (λ c, c.to_nat - '0'.to_nat).sum

noncomputable def f (n : ℕ) : ℕ :=
  digit_sum (n^2 + 1)

noncomputable def f_k (k : ℕ) (n : ℕ) : ℕ :=
  (λ f n => (λ n => iterate f k n) f n) f n

theorem f_2005_eq_11 : f_k 2005 8 = 11 := by
  sorry

end f_2005_eq_11_l158_158613


namespace log_order_preservation_l158_158532

theorem log_order_preservation {a b : ℝ} (ha : a > 0) (hb : b > 0) : 
  (Real.log a > Real.log b) → (a > b) :=
by
  sorry

end log_order_preservation_l158_158532


namespace even_value_when_n_is_101_l158_158796

theorem even_value_when_n_is_101 :
  let n := 101 in
  (¬ ∃ k, 3 * n = 2 * k) ∧
  (¬ ∃ k, n + 2 = 2 * k) ∧
  (¬ ∃ k, n - 12 = 2 * k) ∧
  (∃ k, 2 * n - 2 = 2 * k) ∧
  (¬ ∃ k, 3 * n + 2 = 2 * k) :=
by
  sorry

end even_value_when_n_is_101_l158_158796


namespace pow_2m_n_eq_12_l158_158164

variable (a m n : ℝ)
variable (log_a_2 : log a 2 = m)
variable (log_a_3 : log a 3 = n)

theorem pow_2m_n_eq_12 : a^(2*m + n) = 12 :=
by
  sorry

end pow_2m_n_eq_12_l158_158164


namespace paths_from_A_to_B_avoiding_C_l158_158520

theorem paths_from_A_to_B_avoiding_C (n : ℕ) :
  n = 66 ↔ ( (choose 9 4) - (choose 4 2 * choose 5 2) ) = n := by
  sorry

end paths_from_A_to_B_avoiding_C_l158_158520


namespace value_of_expression_l158_158436

theorem value_of_expression : 7^3 - 4 * 7^2 + 4 * 7 - 1 = 174 :=
by
  sorry

end value_of_expression_l158_158436


namespace math_problem_proof_l158_158183

variables {x y a b: ℝ}
variables (P Q : ℝ × ℝ)
variables (T R : ℝ × ℝ)

noncomputable def ellipse : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

def focus_on_line (P : ℝ × ℝ) : Prop := P.1 = 1

noncomputable def eccentricity_eq_half : Prop :=
  (sqrt (a^2 - b^2) / a) = 1 / 2

def ellipse_equation : Prop := 
  x^2 / 4 + y^2 / 3 = 1

noncomputable def fixed_point_R : Prop :=
  ∃ R : ℝ × ℝ, R = (1 / 4, 0) ∧ ∀ P Q : ℝ × ℝ,
  (P.1, P.2) ≠ (Q.1, Q.2) ∧
  (P.1 + Q.1) / 2 = 1 ∧ 
  |R.1 - P.1| = |R.1 - Q.1| ∧ 
  |R.2 - P.2| = |R.2 - Q.2|

noncomputable def isosceles_right_triangle_not_possible (P Q : ℝ × ℝ) (R : ℝ × ℝ) : Prop :=
  ¬((P.1 - R.1) * (Q.1 - R.1) + P.2 * Q.2 = 0)

theorem math_problem_proof :
  ellipse ∧ focus_on_line (1, 0) ∧ eccentricity_eq_half →
  ellipse_equation ∧ fixed_point_R ∧ isosceles_right_triangle_not_possible :=
by
  sorry

end math_problem_proof_l158_158183


namespace fg_eval_at_3_l158_158684

def f (x : ℝ) : ℝ := 4 * x - 1
def g (x : ℝ) : ℝ := (x + 2) ^ 2

theorem fg_eval_at_3 : f (g 3) = 99 := by
  sorry

end fg_eval_at_3_l158_158684


namespace missing_digit_B_divisible_by_3_l158_158539

theorem missing_digit_B_divisible_by_3 (B : ℕ) (h1 : (2 * 10 + 8 + B) % 3 = 0) :
  B = 2 :=
sorry

end missing_digit_B_divisible_by_3_l158_158539


namespace pool_full_capacity_is_2000_l158_158669

-- Definitions based on the conditions given
def water_loss_per_jump : ℕ := 400 -- in ml
def jumps_before_cleaning : ℕ := 1000
def cleaning_threshold : ℚ := 0.80 -- 80%
def total_water_loss : ℕ := water_loss_per_jump * jumps_before_cleaning -- in ml
def water_loss_liters : ℚ := total_water_loss / 1000 -- converting ml to liters
def cleaning_loss_fraction : ℚ := 1 - cleaning_threshold -- 20% loss

-- The actual proof statement
theorem pool_full_capacity_is_2000 :
  (water_loss_liters : ℚ) / cleaning_loss_fraction = 2000 :=
by
  sorry

end pool_full_capacity_is_2000_l158_158669


namespace greatest_integer_1000x_l158_158498

def cube_edge_length : ℝ := 2
def shadow_area_excluding_cube : ℝ := 288
def total_shadow_area : ℝ := 292

noncomputable def shadow_side_length : ℝ := real.sqrt total_shadow_area
noncomputable def reduced_shadow_side_length : ℝ := shadow_side_length - cube_edge_length

axiom similar_triangles_relation (x : ℝ) : x / cube_edge_length = cube_edge_length / reduced_shadow_side_length

noncomputable def x_value : ℝ := 2 * (2 / (shadow_side_length - cube_edge_length))

noncomputable def target_value : ℝ := 1000 * x_value
noncomputable def greatest_integer_le (z : ℝ) : ℕ := int.to_nat (int.floor z)

theorem greatest_integer_1000x : greatest_integer_le target_value = 265 :=
by sorry

end greatest_integer_1000x_l158_158498


namespace problem_correct_calculation_l158_158442

theorem problem_correct_calculation (a b : ℕ) : 
  (4 * a - 2 * a ≠ 2) ∧ 
  (a^8 / a^4 ≠ a^2) ∧ 
  (a^2 * a^3 = a^5) ∧ 
  ((b^2)^3 ≠ b^5) :=
by {
  sorry
}

end problem_correct_calculation_l158_158442


namespace arccos_proof_l158_158957

noncomputable def arccos_identity : Prop := 
  ∃ x : ℝ, x = 1 / Real.sqrt 2 ∧ Real.arccos x = Real.pi / 4

theorem arccos_proof : arccos_identity :=
by
  sorry

end arccos_proof_l158_158957


namespace arccos_one_over_sqrt_two_l158_158978

theorem arccos_one_over_sqrt_two :
  Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_l158_158978


namespace distance_from_P_to_origin_l158_158640

namespace CartesianCoordinate

def distance_to_origin (x y : ℝ) : ℝ := real.sqrt (x^2 + y^2)

theorem distance_from_P_to_origin : distance_to_origin 5 12 = 13 :=
by
  sorry

end CartesianCoordinate

end distance_from_P_to_origin_l158_158640


namespace each_integer_appears_exactly_once_l158_158340

noncomputable def sequence : ℕ → ℤ := sorry

axiom seq_inf_pos_neg :
  (∀ n, ∃ i > n, 0 < sequence i) ∧ (∀ n, ∃ i > n, sequence i < 0)

axiom seq_distinct_remainders (n : ℕ) (hn : 0 < n):
  ∀ i j < n, sequence i % n ≠ sequence j % n → i = j

theorem each_integer_appears_exactly_once :
  ∀ x : ℤ, ∃! n, sequence n = x :=
sorry

end each_integer_appears_exactly_once_l158_158340


namespace solve_for_x_l158_158155

theorem solve_for_x (x : ℝ) : 2^(3 * x) * 8^(x - 1) = 32^(2 * x) → x = -3 / 4 :=
by
  intro h
  sorry

end solve_for_x_l158_158155


namespace quadratic_has_real_roots_l158_158626

theorem quadratic_has_real_roots (k : ℝ) : (∃ x : ℝ, k * x^2 - 6 * x + 9 = 0) ↔ (k ≤ 1 ∧ k ≠ 0) :=
by
  sorry

end quadratic_has_real_roots_l158_158626


namespace probability_divisibility_by_10_l158_158746

theorem probability_divisibility_by_10 (a b c d e: ℕ) (digits : Finset ℕ)
  (h_digits : digits = {0, a, b, c, d, e})
  (h_unique : Multiset.card digits.val = 6) :
  (1 : ℝ) = 1 :=
by
  -- Assume the conditions
  have h_contains_0 : 0 ∈ digits := by 
    -- Use the condition that 'digits' is the set of {0, a, b, c, d, e}
    rw [h_digits]
    simp
  have h_not_multiple_of_10_possible : ∀ n ∈ digits.val.erase 0, n ≠ 0 := by
    intro n hn
    simp at hn
    cases hn
    case inl => contradiction
    case inr => 
      intro h
      rw [h] at hn 
      contradiction
  have h_arrangements : multiset.permutations (digits.val.erase 0) = unordered_multiset.permutations (unordered_multiset.val.erase 0) := by sorry
  have h_num_arrangements : multiset.card (multiset.permutations {a, b, c, 5, 9}.val) = 1 := by sorry

  -- Conclusion:
  have h_prob : (1 : ℝ) = 1 := rfl
  exact h_prob

end probability_divisibility_by_10_l158_158746


namespace find_d_squared_l158_158983

noncomputable def g (c d : ℝ) (z : ℂ) : ℂ := (c + d * complex.I) * z

theorem find_d_squared (c d : ℝ) (z : ℂ) (h1 : |c + d * complex.I| = 10)
  (h2 : ∀ (z : ℂ), |g c d z - z| = |g c d z|) :
  d^2 = 399 / 4 := 
sorry

end find_d_squared_l158_158983


namespace correct_calculation_l158_158440

variable {a : ℝ}

theorem correct_calculation : a^2 * a^3 = a^5 :=
by sorry

end correct_calculation_l158_158440


namespace cubic_polynomial_evaluation_l158_158614

theorem cubic_polynomial_evaluation
  (f : ℚ → ℚ)
  (cubic_f : ∃ a b c d : ℚ, ∀ x, f x = a*x^3 + b*x^2 + c*x + d)
  (h1 : f (-2) = -4)
  (h2 : f 3 = -9)
  (h3 : f (-4) = -16) :
  f 1 = -23 :=
sorry

end cubic_polynomial_evaluation_l158_158614


namespace certain_number_is_14_l158_158455

theorem certain_number_is_14 
  (a b n : ℕ) 
  (h1 : ∃ k1, a = k1 * n) 
  (h2 : ∃ k2, b = k2 * n) 
  (h3 : b = a + 11 * n) 
  (h4 : b = a + 22 * 7) : n = 14 := 
by 
  sorry

end certain_number_is_14_l158_158455


namespace range_of_a_l158_158168

variable {a : ℝ}

def f (x : ℝ) : ℝ := (x + a / x) * Real.exp x

def f' (x : ℝ) : ℝ := ((x^3 + x^2 + a * x - a) / x^2) * Real.exp x

theorem range_of_a (h_extremum : ∃! x ∈ Ioo (0 : ℝ) 1, f' x = 0) : a > 0 := 
sorry

end range_of_a_l158_158168


namespace min_value_expression_l158_158545

theorem min_value_expression (x y z : ℝ) (h1 : x > 1) (h2 : y > 1) (h3 : z > 1) : ∃ C, C = 12 ∧
  ∀ (x y z : ℝ), x > 1 → y > 1 → z > 1 → (x^2 / (y - 1) + y^2 / (z - 1) + z^2 / (x - 1)) ≥ C := by
  sorry

end min_value_expression_l158_158545


namespace circle_center_on_line_l158_158395

theorem circle_center_on_line (α : ℝ) :
  let center := (cos α, -sin α) in
  (cos α) * (fst center) - (sin α) * (snd center) = 1 :=
by
  let center := (cos α, -sin α)
  sorry

end circle_center_on_line_l158_158395


namespace number_of_players_taking_mathematics_l158_158105

def total_players : ℕ := 25
def players_taking_physics : ℕ := 12
def players_taking_both : ℕ := 5

theorem number_of_players_taking_mathematics :
  total_players - players_taking_physics + players_taking_both = 18 :=
by
  sorry

end number_of_players_taking_mathematics_l158_158105


namespace probability_sum_even_l158_158291

def primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def sum_is_even (a b : ℕ) : Prop :=
  (a + b) % 2 = 0

theorem probability_sum_even :
  (nat.choose 10 2) = 45 →
  (∀ a b : ℕ, a ∈ primes → b ∈ primes → a ≠ b →
    (sum_is_even a b ↔ a ≠ 2 ∧ b ≠ 2)) →
  ((45 - 9) / 45 : ℚ) = 4 / 5 :=
by sorry

end probability_sum_even_l158_158291


namespace range_of_f_on_nonneg_reals_l158_158584

theorem range_of_f_on_nonneg_reals (k : ℕ) (h_even : k % 2 = 0) (h_pos : 0 < k) :
    ∀ y : ℝ, 0 ≤ y ↔ ∃ x : ℝ, 0 ≤ x ∧ x^k = y :=
by
  sorry

end range_of_f_on_nonneg_reals_l158_158584


namespace arccos_of_one_over_sqrt_two_l158_158891

theorem arccos_of_one_over_sqrt_two :
  Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 := 
sorry

end arccos_of_one_over_sqrt_two_l158_158891


namespace arccos_one_over_sqrt_two_eq_pi_over_four_l158_158923

theorem arccos_one_over_sqrt_two_eq_pi_over_four : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_eq_pi_over_four_l158_158923


namespace current_speed_is_2_l158_158487

-- The rowing speed in still water
def still_water_speed : ℝ := 8

-- The distance to the place (one way)
def distance : ℝ := 7.5

-- The total time for the round trip
def total_time : ℝ := 2

-- Define the effective speed with and against the current
def with_current_speed (v : ℝ) : ℝ := still_water_speed + v
def against_current_speed (v : ℝ) : ℝ := still_water_speed - v

-- Define the time taken to row to the place and back
def time_to_place (v : ℝ) : ℝ := distance / with_current_speed v
def time_back (v : ℝ) : ℝ := distance / against_current_speed v

-- Prove that for a given total time of 2 hours, the current speed v is 2 kmph
theorem current_speed_is_2 (v : ℝ) (h_v : time_to_place v + time_back v = total_time) : v = 2 :=
by
  sorry

end current_speed_is_2_l158_158487


namespace probability_even_sum_l158_158281

def first_ten_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def distinct_pairs (l : List ℕ) : List (ℕ × ℕ) :=
  (l.product l).filter (λ p, p.1 < p.2)

theorem probability_even_sum : 
  (first_ten_primes.length = 10) →
  (∀ a b : ℕ, a ∈ first_ten_primes → b ∈ first_ten_primes → a ≠ b → 
    ((a + b) % 2 = 0 ↔ 2 ∉ [a, b])) →
  (∑ pair in distinct_pairs first_ten_primes, ((pair.1 + pair.2) % 2 = 0).toNat) 
    = 36 → 
  ((∑ pair in distinct_pairs first_ten_primes, (1:ℚ)) ⁻¹ * 
   (∑ pair in distinct_pairs first_ten_primes, ((pair.1 + pair.2) % 2 = 0).toNat))
    = 4 / 5 := by
  sorry

end probability_even_sum_l158_158281


namespace arccos_one_over_sqrt_two_l158_158977

theorem arccos_one_over_sqrt_two :
  Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_l158_158977


namespace find_p_l158_158213

theorem find_p (p x_0 : ℝ) (h1 : 0 < p) (h2 : (2 * p * x_0)^2 = (2 * √2)^2)
(h3 : 5 + x_0^2 = (x_0 + (1 / 2) * p)^2) : p = 2 :=
sorry

end find_p_l158_158213


namespace simple_interest_years_l158_158095

variables (T R : ℝ)

def principal : ℝ := 1000
def additional_interest : ℝ := 90

theorem simple_interest_years
  (H: principal * (R + 3) * T / 100 - principal * R * T / 100 = additional_interest) :
  T = 3 :=
by sorry

end simple_interest_years_l158_158095


namespace sum_of_coordinates_of_reflected_midpoint_l158_158359

def midpoint (P Q : ℝ × ℝ) : ℝ × ℝ :=
  ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

def reflect_over_x_axis (point : ℝ × ℝ) : ℝ × ℝ :=
  (point.1, -point.2)

theorem sum_of_coordinates_of_reflected_midpoint :
  let P     : ℝ × ℝ := (2, 1)
  let R     : ℝ × ℝ := (12, 15)
  let M     : ℝ × ℝ := midpoint P R
  let M_ref : ℝ × ℝ := reflect_over_x_axis M
  M_ref.1 + M_ref.2 = -1 :=
by
  -- Definitions are enough to state the problem,
  -- and the proof is omitted as per instructions.
  sorry

end sum_of_coordinates_of_reflected_midpoint_l158_158359


namespace find_VW_l158_158312

-- Define the problem conditions
variables (X Y Z W M U V N S: Type)
variables [Inhabited X] [Inhabited Y] [Inhabited Z] [Inhabited W] [Inhabited M] [Inhabited U] [Inhabited V] [Inhabited N] [Inhabited S]

-- Hypotheses corresponding to the problem conditions
variables (h1 : ∠XMW = 90)
variables (h2 : perpendicular U V Y Z)
variables (h3 : YM = MV)
variables (h4 : MW ∩ UV = N)
variables (h5 : S ∈ ZW)
variables (h6 : S = line_through SX N)
variables (h7 : MX = 24)
variables (h8 : XN = 32)
variables (h9 : MN = 40)

-- The statement to be proven
theorem find_VW : VW = 38.4 :=
by
  sorry

end find_VW_l158_158312


namespace quadratic_equation_real_roots_l158_158630

theorem quadratic_equation_real_roots (k : ℝ) : 
  (∃ x : ℝ, k * x^2 - 6 * x + 9 = 0) ↔ (k ≤ 1 ∧ k ≠ 0) :=
by
  sorry

end quadratic_equation_real_roots_l158_158630


namespace solve_for_x_l158_158560

def α(x : ℚ) : ℚ := 4 * x + 9
def β(x : ℚ) : ℚ := 9 * x + 6

theorem solve_for_x (x : ℚ) (h : α(β(x)) = 8) : x = -25 / 36 :=
by
  sorry

end solve_for_x_l158_158560


namespace super_square_count_largest_super_square_number_l158_158839

-- Definitions representing the two-digit perfect squares
def two_digit_squares : List Nat := [16, 25, 36, 49, 64, 81]

-- Predicate to determine if a number is a "super square"
def is_super_square (N : Nat) : Prop :=
  let digits := N.digits
  digits.length ≥ 2 ∧
  ∀ i, i < digits.length - 1 → (digits.nth_le i 0 * 10 + digits.nth_le (i + 1) 0) ∈ two_digit_squares

-- Part (a): Prove that there are 14 "super square" numbers
theorem super_square_count : 
  (∃ ns : List Nat, (∀ n ∈ ns, is_super_square n) ∧ List.length ns = 14) :=
sorry

-- Part (b): Prove that the largest "super square" number is 81649
theorem largest_super_square_number :
  ∃ n : Nat, is_super_square n ∧ n = 81649 :=
sorry

end super_square_count_largest_super_square_number_l158_158839


namespace solve_cross_number_l158_158132

def regular_polygon_internal_angle (exterior_angle : ℕ) : ℕ :=
  180 - exterior_angle

def is_multiple_of (n k : ℕ) : Prop :=
  ∃ m, n = k * m

def is_proper_factor (a b : ℕ) : Prop :=
  a ≠ b ∧ ∃ k, b = a * k

def is_valid_square (n : ℕ) : Prop :=
  ∃ m, n = m * m

theorem solve_cross_number :
∃ (acr1 acr5 dn1 dn2 dn4 acr3 : ℕ),
  is_multiple_of acr1 7 ∧ 
  acr5 > 10 ∧
  (is_multiple_of dn1 9 ∨ is_multiple_of dn1 25 ∨ is_multiple_of dn1 49) ∧
  ¬ is_valid_square dn1 ∧ ¬ is_multiple_of dn1 (λ m, m * m * m) ∧
  (dn2 = regular_polygon_internal_angle 12 ∨ dn2 = regular_polygon_internal_angle 15 ∨ dn2 = regular_polygon_internal_angle 18) ∧
  is_proper_factor dn4 acr5 ∧ ¬ is_proper_factor dn4 dn1 ∧
  acr3 = 961 :=
sorry

end solve_cross_number_l158_158132


namespace find_p_l158_158590

noncomputable def hyperbola_asymptotes : ℝ × ℝ → Prop :=
λ (x y : ℝ), y = 2 * x ∨ y = -2 * x

noncomputable def parabola_directrix (p : ℝ) (h_pos : 0 < p) : ℝ → Prop :=
λ x, x = -p / 2

noncomputable def triangle_area (p : ℝ) : ℝ :=
1 / 2 * (p / 2) * (2 * p)

theorem find_p (p : ℝ) (h_pos : 0 < p) :
  (∃ A B : ℝ × ℝ, hyperbola_asymptotes A.1 A.2 ∧ 
                 hyperbola_asymptotes B.1 B.2 ∧ 
                 parabola_directrix p h_pos A.1 ∧ 
                 parabola_directrix p h_pos B.1 ∧ 
                 triangle_area p = 1) → 
  p = Real.sqrt 2 :=
by
  sorry

end find_p_l158_158590


namespace two_hundredth_digit_of_5_over_13_l158_158421

theorem two_hundredth_digit_of_5_over_13 :
  (∀ n : ℕ, n ≥ 0 →
    let d := "384615".charAt ( (n % 6) - 1 ) in
    if n % 6 = 0 then '5' else d  = 8) :=
sorry

end two_hundredth_digit_of_5_over_13_l158_158421


namespace radius_increase_l158_158751

variable (C₁ C₂ : ℝ)
hypothesis (h₁ : C₁ = 40)
hypothesis (h₂ : C₂ = 50)
theorem radius_increase : (C₂ / (2 * Real.pi) - C₁ / (2 * Real.pi)) = 5 / Real.pi := by
  sorry

end radius_increase_l158_158751


namespace staircase_sum_of_digits_l158_158705

def Ollie_jumps (n : ℕ) : ℕ := (n + 2) / 3
def Dana_jumps (n : ℕ) : ℕ := (n + 3) / 4

theorem staircase_sum_of_digits :
  let possible_steps := {n : ℕ | Ollie_jumps n - Dana_jumps n = 10}
  let s := possible_steps.to_list.sum
  Nat.digits 10 s |>.sum = 3 := by
  sorry

end staircase_sum_of_digits_l158_158705


namespace colby_mango_sales_l158_158120

theorem colby_mango_sales
  (total_kg : ℕ)
  (mangoes_per_kg : ℕ)
  (remaining_mangoes : ℕ)
  (half_sold_to_market : ℕ) :
  total_kg = 60 →
  mangoes_per_kg = 8 →
  remaining_mangoes = 160 →
  half_sold_to_market = 20 := by
    sorry

end colby_mango_sales_l158_158120


namespace phosphorus_atoms_l158_158481

theorem phosphorus_atoms (x : ℝ) : 122 = 26.98 + 30.97 * x + 64 → x = 1 := by
sorry

end phosphorus_atoms_l158_158481


namespace number_of_white_balls_l158_158308

theorem number_of_white_balls (x : ℕ) (h : (5 : ℚ) / (5 + x) = 1 / 4) : x = 15 :=
by 
  sorry

end number_of_white_balls_l158_158308


namespace arccos_sqrt_half_l158_158971

theorem arccos_sqrt_half : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 := by
  sorry

end arccos_sqrt_half_l158_158971


namespace pencil_boxes_count_l158_158094

theorem pencil_boxes_count (x : ℕ) (h1 : ∃ y, y = 80 * x)
  (h2 : 320 * x + 5 * (2 * (80 * x) + 300) = 18300) : x = 15 :=
by
  -- Core assumptions
  let num_pencils := 80 * x
  let cost_pencils := 4 * num_pencils
  let num_pens := 2 * num_pencils + 300
  let cost_pens := 5 * num_pens
  have total_cost := cost_pencils + cost_pens
  -- Checking the total cost condition
  total_cost = 18300
  sorry

end pencil_boxes_count_l158_158094


namespace sum_of_repeating_digit_numbers_l158_158434

/-- Let S be the sum of all 8-digit numbers consisting only of the digits {1, 2, 3, 4, 5, 6, 7} where each digit appears at least once and exactly one digit repeats. -/
theorem sum_of_repeating_digit_numbers :
  let N := 8!
  let repeated_digit_ways := 7
  let factor := N / 2
  let digit_sum := 1 + 2 + 3 + 4 + 5 + 6 + 7
  let positional_sum := (10^8 - 1) / 9
  (repeated_digit_ways * factor * digit_sum * positional_sum = 8! * 14 * (10^8 - 1) / 9) :=
by
  sorry

end sum_of_repeating_digit_numbers_l158_158434


namespace teachers_per_grade_correct_l158_158370

def fifth_graders : ℕ := 109
def sixth_graders : ℕ := 115
def seventh_graders : ℕ := 118
def parents_per_grade : ℕ := 2
def number_of_grades : ℕ := 3
def buses : ℕ := 5
def seats_per_bus : ℕ := 72

-- Total number of students
def total_students : ℕ := fifth_graders + sixth_graders + seventh_graders

-- Total number of parents
def total_parents : ℕ := parents_per_grade * number_of_grades

-- Total number of seats available on the buses
def total_seats : ℕ := buses * seats_per_bus

-- Seats left for teachers
def seats_for_teachers : ℕ := total_seats - total_students - total_parents

-- Teachers per grade
def teachers_per_grade : ℕ := seats_for_teachers / number_of_grades

theorem teachers_per_grade_correct : teachers_per_grade = 4 := sorry

end teachers_per_grade_correct_l158_158370


namespace arrange_triangle_sum_l158_158115

/-- 
  Prove that there exists a permutation of the numbers 2016, 2017, ..., 2024 
  such that arranged on the positions of a triangle, 
  the sum of the numbers on each side of the triangle is equal.
-/
theorem arrange_triangle_sum (numbers : Fin 9 → ℕ) 
  (h_numbers : ∀ i, numbers i ∈ {2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024}) :
  ∃ (perm : Permutation (Fin 9)), 
    (numbers (perm 0) + numbers (perm 1) + numbers (perm 2) = 
    numbers (perm 3) + numbers (perm 4) + numbers (perm 5)) ∧ 
    (numbers (perm 3) + numbers (perm 4) + numbers (perm 5) = 
    numbers (perm 6) + numbers (perm 7) + numbers (perm 8)) ∧ 
    (numbers (perm 6) + numbers (perm 7) + numbers (perm 8) = 
    numbers (perm 0) + numbers (perm 1) + numbers (perm 2)) :=
sorry

end arrange_triangle_sum_l158_158115


namespace largest_tangent_circle_l158_158314

-- Define the line equation and the circle center
def line_equation (m : ℝ) (x y : ℝ) : Prop := 2 * m * x - y - 4 * m + 1 = 0
def circle_center : ℝ × ℝ := (1, 0)

-- Define the largest circle tangent to the line
noncomputable def largest_tangent_circle_radius : ℝ := sqrt 2
noncomputable def largest_tangent_circle_equation (x y : ℝ) : Prop := 
  (x - 1) ^ 2 + y ^ 2 = largest_tangent_circle_radius ^ 2

-- Proof statement
theorem largest_tangent_circle (m : ℝ) :
  ∃ (r : ℝ), r = largest_tangent_circle_radius ∧
  ∀ x y : ℝ, largest_tangent_circle_equation x y ↔ (x - 1)^2 + y^2 = 2 :=
by
  sorry

end largest_tangent_circle_l158_158314


namespace difference_in_balances_l158_158864

/-- Define the parameters for Angela's and Bob's accounts --/
def P_A : ℕ := 5000  -- Angela's principal
def r_A : ℚ := 0.05  -- Angela's annual interest rate
def n_A : ℕ := 2  -- Compounding frequency for Angela
def t : ℕ := 15  -- Time in years

def P_B : ℕ := 7000  -- Bob's principal
def r_B : ℚ := 0.04  -- Bob's annual interest rate

/-- Computing the final amounts for Angela and Bob after 15 years --/
noncomputable def A_A : ℚ := P_A * ((1 + (r_A / n_A)) ^ (n_A * t))  -- Angela's final amount
noncomputable def A_B : ℚ := P_B * (1 + r_B * t)  -- Bob's final amount

/-- Proof statement: The difference in account balances to the nearest dollar --/
theorem difference_in_balances : abs (A_A - A_B) = 726 := by
  sorry

end difference_in_balances_l158_158864


namespace symmetry_conditions_l158_158393

--Definitions based on conditions
def is_symmetric {α β : Type*} [Add α] [Neg β] (f : α → β) (a : α) (b : β) : Prop :=
  ∀ x : α, f (x + a) - b = -(f (-x + a) - b)

def f1 (x : ℝ) (a b : ℝ) := a * x + b

def f2 (x : ℝ) := (2 * x + 1) / (x + 1)

def f3 (x : ℝ) := x^3 - 2*x^2

def f4 (x : ℝ) : ℝ :=
  if x > 0 ∧ x ≤ 3 then x^2 - 2*x - 3
  else 0 -- assuming 0 for values outside the given range for simplicity

-- Statement of the theorem proving questions == correct answers given conditions
theorem symmetry_conditions :
  ¬ (∀ x : ℝ, is_symmetric (f1 x 1 0) 0 0) ∧
  is_symmetric f2 (-1) 2 ∧
  ¬ (∀ x : ℝ, is_symmetric (f3 x) (4/3) 0) ∧
  (is_symmetric f4 0 (-1) → (0, 3) = (-4, 2)) :=
by sorry

end symmetry_conditions_l158_158393


namespace first_player_wins_with_optimal_play_l158_158041

-- Define the game and the grid size
structure GridConfig where
  height : ℕ
  width : ℕ
  condition : height = 19 ∧ width = 94

-- Define the player and their move strategy
inductive Player
| first
| second

-- Define a move on the grid
structure Move where
  player : Player
  squareSize : ℕ -- side length of the square
  positionX : ℕ -- x coordinate of the top left corner
  positionY : ℕ -- y coordinate of the top left corner

-- Define the game state
structure GameState where
  currentGrid : GridConfig
  markedCells : list (Move)

noncomputable def optimalPlay : Prop :=
  ∀ (gameState : GameState), 
  -- Assuming optimal moves from both players,
  -- the first player will always win.
  Player.first = win gameState

theorem first_player_wins_with_optimal_play :
  optimalPlay :=
sorry

end first_player_wins_with_optimal_play_l158_158041


namespace arrangement_count_of_students_l158_158071

theorem arrangement_count_of_students
  (m f : Finset ℕ)
  (hm : m.card = 3)
  (hf : f.card = 3)
  (h_total : (m ∪ f).card = 6)
  (h_adjacent_females : ∃ (a b : ℕ) (hf1 hf2 : a ∈ f) (hf3 : b ∈ f) (hf4 : a ≠ b), 
    (m ∪ f).list ~ [a] ++ [b])
  : ∃ (n : ℕ), n = 432 := 
sorry

end arrangement_count_of_students_l158_158071


namespace probability_even_sum_l158_158271

-- Definition of the prime numbers set and the selection scenario
def firstTenPrimes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Condition stating that we select two distinct numbers at random from the first ten primes
def randomSelection (s : List ℕ) := {x : ℕ × ℕ // x.1 ≠ x.2 ∧ x.1 ∈ s ∧ x.2 ∈ s}

-- Define the event that their sum is even
def evenSum (p : ℕ × ℕ) := (p.1 + p.2) % 2 = 0

-- Define the probability calculation
def probabilityEvenSum := 
  let totalPairs := (firstTenPrimes.length.choose 2) -- Calculate the number of ways to choose 2 numbers from the 10 primes
  let evenPairs := (randomSelection firstTenPrimes).count (λ p => evenSum p.val)
  evenPairs.toFloat / totalPairs.toFloat

-- Theorem statement that the probability of their sum being even is 1/5
theorem probability_even_sum : probabilityEvenSum = 1/5 := by
  sorry -- The actual proof is to be provided

end probability_even_sum_l158_158271


namespace exists_polynomial_same_diff_vals_l158_158534

theorem exists_polynomial_same_diff_vals :
  ∃ f : ℚ[X],  (∀ x y : ℝ, x ≠ y → x ∈ {0, real.sqrt 2, -real.sqrt 2} → f.eval x = 0 → f.eval y = 0) ∧
              (∀ x y : ℚ, x ≠ y → f.eval x ≠ f.eval y) :=
by {
  use polynomial.Cx (x^3 - 2 * x),
  sorry
}

end exists_polynomial_same_diff_vals_l158_158534


namespace investor_difference_l158_158862

/-
Scheme A yields 30% of the capital within a year.
Scheme B yields 50% of the capital within a year.
Investor invested $300 in scheme A.
Investor invested $200 in scheme B.
We need to prove that the difference in total money between scheme A and scheme B after a year is $90.
-/

def schemeA_yield_rate : ℝ := 0.30
def schemeB_yield_rate : ℝ := 0.50
def schemeA_investment : ℝ := 300
def schemeB_investment : ℝ := 200

def total_after_year (investment : ℝ) (yield_rate : ℝ) : ℝ :=
  investment * (1 + yield_rate)

theorem investor_difference :
  total_after_year schemeA_investment schemeA_yield_rate - total_after_year schemeB_investment schemeB_yield_rate = 90 := by
  sorry

end investor_difference_l158_158862


namespace probability_even_sum_l158_158253

def first_ten_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def is_odd (n : ℕ) : Prop := n % 2 = 1

def count_odd_primes (primes : List ℕ) : ℕ :=
  primes.countp is_odd

def binom (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

open Nat

theorem probability_even_sum :
  let primes := first_ten_primes in
  let odd_primes := count_odd_primes primes in
  let num_ways_even_sum := binom odd_primes 2 in
  let total_ways := binom primes.length 2 in
  (num_ways_even_sum : ℚ) / (total_ways : ℚ) = 4 / 5 :=
by
  sorry

end probability_even_sum_l158_158253


namespace probability_even_sum_l158_158258

def first_ten_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def is_odd (n : ℕ) : Prop := n % 2 = 1

def count_odd_primes (primes : List ℕ) : ℕ :=
  primes.countp is_odd

def binom (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

open Nat

theorem probability_even_sum :
  let primes := first_ten_primes in
  let odd_primes := count_odd_primes primes in
  let num_ways_even_sum := binom odd_primes 2 in
  let total_ways := binom primes.length 2 in
  (num_ways_even_sum : ℚ) / (total_ways : ℚ) = 4 / 5 :=
by
  sorry

end probability_even_sum_l158_158258


namespace largest_whole_number_l158_158017

theorem largest_whole_number (x : ℕ) (h1 : 9 * x < 150) : x ≤ 16 :=
by sorry

end largest_whole_number_l158_158017


namespace sequence_evaluation_l158_158318

noncomputable def a (n : ℕ) : ℝ :=
  if n = 0 then 1 / 3 else 
    let an := a (n-1);
    1 / (3 / (an * (an + 3)))

noncomputable def b (n : ℕ) : ℝ :=
  1 / (3 + a n)

noncomputable def P (n : ℕ) : ℝ :=
  (List.range (n + 1)).map b.prod

noncomputable def S (n : ℕ) : ℝ :=
  (List.range (n + 1)).map b.sum

theorem sequence_evaluation (n : ℕ) : 3^(n+1) * P n + S n = 3 :=
by
  sorry

end sequence_evaluation_l158_158318


namespace fractional_inequality_solution_l158_158401

theorem fractional_inequality_solution
  (a b x : ℝ)
  (h1 : ∀ x, ax + b > 0 → x < 1)
  (h2 : a + b = 0)
  (h3 : a < 0) :
  (∀ x, (bx - a)/(x + 2) > 0 ↔ x ∈ (Iio (-2) ∪ Ioi (-1))) :=
by
  sorry

end fractional_inequality_solution_l158_158401


namespace length_of_first_train_l158_158496

theorem length_of_first_train (speed_first_train_kmph : ℝ) (speed_second_train_kmph : ℝ) 
    (crossing_time_seconds : ℝ) (length_second_train_m : ℝ) : 
    speed_first_train_kmph = 120 → 
    speed_second_train_kmph = 80 → 
    crossing_time_seconds = 9 → 
    length_second_train_m = 270.04 → 
    let relative_speed_mps := (speed_first_train_kmph + speed_second_train_kmph) * 1000 / 3600 in
    let combined_length_m := relative_speed_mps * crossing_time_seconds in
    let length_first_train_m := combined_length_m - length_second_train_m in
    length_first_train_m = 230 :=
by {
  intros sft ssrtc sfs lsm rel_speed_mps comb_len len_first;
  sorry
}

end length_of_first_train_l158_158496


namespace min_value_of_expression_l158_158578

theorem min_value_of_expression (x y : ℝ) (hx : x > y) (hy : y > 0) (hxy : x + y ≤ 2) :
  ∃ m : ℝ, m = (2 / (x + 3 * y) + 1 / (x - y)) ∧ m = (3 + 2 * Real.sqrt 2) / 4 :=
by
  sorry

end min_value_of_expression_l158_158578


namespace Vlad_score_l158_158377

-- Defining the initial conditions of the problem
def total_rounds : ℕ := 30
def points_per_win : ℕ := 5
def total_points : ℕ := total_rounds * points_per_win

-- Taro's score as described in the problem
def Taros_score := (3 * total_points / 5) - 4

-- Prove that Vlad's score is 64 points
theorem Vlad_score : total_points - Taros_score = 64 := by
  sorry

end Vlad_score_l158_158377


namespace arccos_sqrt_half_l158_158968

theorem arccos_sqrt_half : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 := by
  sorry

end arccos_sqrt_half_l158_158968


namespace upper_side_length_l158_158241

variable (L U h : ℝ)

-- Given conditions
def condition1 : Prop := U = L - 6
def condition2 : Prop := 72 = (1 / 2) * (L + U) * 8
def condition3 : Prop := h = 8

-- The length of the upper side of the trapezoid
theorem upper_side_length (h : h = 8) (c1 : U = L - 6) (c2 : 72 = (1 / 2) * (L + U) * 8) : U = 6 := 
by
  sorry

end upper_side_length_l158_158241


namespace number_of_valid_subsets_l158_158572

open Set

def odd_numbers := {3, 5}
def universal_set := {3, 4, 5}

theorem number_of_valid_subsets : 
  ∃ (A : Finset ℕ), A ⊆ universal_set ∧ (∃ x ∈ A, x ∈ odd_numbers) ∧
  Finset.card (Finset.filter (λ B, B ⊆ universal_set ∧ ∃ x ∈ B, x ∈ odd_numbers) (Finset.powerset universal_set)) = 6 :=
sorry

end number_of_valid_subsets_l158_158572


namespace paint_cans_needed_l158_158508

theorem paint_cans_needed
    (num_bedrooms : ℕ)
    (num_other_rooms : ℕ)
    (total_rooms : ℕ)
    (gallons_per_room : ℕ)
    (color_paint_cans_per_gallon : ℕ)
    (white_paint_cans_per_gallon : ℕ)
    (total_paint_needed : ℕ)
    (color_paint_cans_needed : ℕ)
    (white_paint_cans_needed : ℕ)
    (total_paint_cans : ℕ)
    (h1 : num_bedrooms = 3)
    (h2 : num_other_rooms = 2 * num_bedrooms)
    (h3 : total_rooms = num_bedrooms + num_other_rooms)
    (h4 : gallons_per_room = 2)
    (h5 : total_paint_needed = total_rooms * gallons_per_room)
    (h6 : color_paint_cans_per_gallon = 1)
    (h7 : white_paint_cans_per_gallon = 3)
    (h8 : color_paint_cans_needed = num_bedrooms * gallons_per_room * color_paint_cans_per_gallon)
    (h9 : white_paint_cans_needed = (num_other_rooms * gallons_per_room) / white_paint_cans_per_gallon)
    (h10 : total_paint_cans = color_paint_cans_needed + white_paint_cans_needed) :
    total_paint_cans = 10 :=
by sorry

end paint_cans_needed_l158_158508


namespace original_savings_l158_158460

-- Given conditions:
def total_savings (s : ℝ) : Prop :=
  1 / 4 * s = 230

-- Theorem statement: 
theorem original_savings (s : ℝ) (h : total_savings s) : s = 920 :=
sorry

end original_savings_l158_158460


namespace belle_biscuits_l158_158107

-- Define the conditions
def cost_per_rawhide_bone : ℕ := 1
def num_rawhide_bones_per_evening : ℕ := 2
def cost_per_biscuit : ℚ := 0.25
def total_weekly_cost : ℚ := 21
def days_in_week : ℕ := 7

-- Define the number of biscuits Belle eats every evening
def num_biscuits_per_evening : ℚ := 4

-- Define the statement that encapsulates the problem
theorem belle_biscuits :
  (total_weekly_cost = days_in_week * (num_rawhide_bones_per_evening * cost_per_rawhide_bone + num_biscuits_per_evening * cost_per_biscuit)) :=
sorry

end belle_biscuits_l158_158107


namespace scientist_birth_day_is_wednesday_l158_158380

noncomputable def calculate_birth_day : String :=
  let years := 150
  let leap_years := 36
  let regular_years := years - leap_years
  let total_days_backward := regular_years + 2 * leap_years -- days to move back
  let days_mod := total_days_backward % 7
  let day_of_birth := (5 + 7 - days_mod) % 7 -- 5 is for backward days from Monday
  match day_of_birth with
  | 0 => "Monday"
  | 1 => "Sunday"
  | 2 => "Saturday"
  | 3 => "Friday"
  | 4 => "Thursday"
  | 5 => "Wednesday"
  | 6 => "Tuesday"
  | _ => "Error"

theorem scientist_birth_day_is_wednesday :
  calculate_birth_day = "Wednesday" :=
  by
    sorry

end scientist_birth_day_is_wednesday_l158_158380


namespace arithmetic_sqrt_of_25_l158_158002

theorem arithmetic_sqrt_of_25 : ∃ (x : ℝ), x^2 = 25 ∧ x = 5 :=
by 
  sorry

end arithmetic_sqrt_of_25_l158_158002


namespace patsy_pigs_in_a_blanket_l158_158711

theorem patsy_pigs_in_a_blanket :
  ∀ (guests : ℕ) (appetizers_per_guest : ℕ) (dozen_deviled_eggs : ℕ) (dozen_kebabs : ℕ) (extra_dozen : ℕ),
  guests = 30 →
  appetizers_per_guest = 6 →
  dozen_deviled_eggs = 3 →
  dozen_kebabs = 2 →
  extra_dozen = 8 →
  let total_appetizers := guests * appetizers_per_guest,
      appetizers_so_far := dozen_deviled_eggs * 12 + dozen_kebabs * 12,
      appetizers_needed := total_appetizers - appetizers_so_far,
      appetizers_extra := extra_dozen * 12,
      appetizers_to_be_made := appetizers_needed - appetizers_extra in
  appetizers_to_be_made / 12 = 2 :=
by
  intros guests appetizers_per_guest dozen_deviled_eggs dozen_kebabs extra_dozen H1 H2 H3 H4 H5
  let total_appetizers := guests * appetizers_per_guest
  let appetizers_so_far := dozen_deviled_eggs * 12 + dozen_kebabs * 12
  let appetizers_needed := total_appetizers - appetizers_so_far
  let appetizers_extra := extra_dozen * 12
  let appetizers_to_be_made := appetizers_needed - appetizers_extra
  exact eq.trans (Nat.div_eq_of_eq_mul_left (by decide) rfl) (by sympathetic)
  sorry

end patsy_pigs_in_a_blanket_l158_158711


namespace trigonometric_identity_l158_158563

theorem trigonometric_identity (x : ℝ) (h : sin (π / 3 - x) = 3 / 5) : cos (5 * π / 6 - x) = -3 / 5 :=
by sorry

end trigonometric_identity_l158_158563


namespace smallest_a_for_shift_l158_158373

def periodic (f : ℝ → ℝ) (p : ℝ) := ∀ x, f (x - p) = f x

theorem smallest_a_for_shift (f : ℝ → ℝ) (h : periodic f 15) :
  ∃ a : ℝ, a > 0 ∧ (∀ x, f (x / 3 - a / 3) = f (x / 3)) ∧ (∀ b, b > 0 ∧ (∀ x, f (x / 3 - b / 3) = f (x / 3)) → a ≤ b) :=
begin 
  use 45,
  split,
  { norm_num, },
  split,
  { intros x,
    rw [← h (x / 3), ← sub_eq_add_neg],
    congr' 1,
    field_simp, },
  { intros b hb_positive hb_condition,
    have : b / 3 = 15, {
      specialize hb_condition 0,
      rw [zero_sub, neg_div] at hb_condition,
      exact eq_of_periodic_of_periodic _ zero_ne_three ⁻¹ hb_condition },
    linarith, }
end

end smallest_a_for_shift_l158_158373


namespace find_a3_l158_158770

def sequence (a : ℕ → ℝ) (λ : ℝ) : Prop :=
a 1 = 1 ∧
a 2 = 3 ∧
∀ n, a (n + 1) = (2 * n - λ) * a n

theorem find_a3 {a : ℕ → ℝ} (λ : ℝ) (h : sequence a λ) (hλ : λ = -1) : 
  a 3 = 15 :=
by
  sorry

end find_a3_l158_158770


namespace smallest_k_l158_158181

noncomputable def Δ (f : ℕ → ℤ) (n : ℕ) : ℤ := f (n+1) - f n

def un (n : ℕ) : ℤ := n^3 + 2*n^2 + n

def Δ1_un (n : ℕ) : ℤ := Δ un n

def Δ2_un (n : ℕ) : ℤ := Δ Δ1_un n

def Δ3_un (n : ℕ) : ℤ := Δ Δ2_un n

def Δ4_un (n : ℕ) : ℤ := Δ Δ3_un n

theorem smallest_k (n : ℕ) : Δ4_un n = 0 :=
by {
  sorry
}

end smallest_k_l158_158181


namespace casey_pumping_time_l158_158880

structure PlantRow :=
  (rows : ℕ) (plants_per_row : ℕ) (water_per_plant : ℚ)

structure Animal :=
  (count : ℕ) (water_per_animal : ℚ)

def morning_pump_rate := 3 -- gallons per minute
def afternoon_pump_rate := 5 -- gallons per minute

def corn := PlantRow.mk 4 15 0.5
def pumpkin := PlantRow.mk 3 10 0.8
def pigs := Animal.mk 10 4
def ducks := Animal.mk 20 0.25
def cows := Animal.mk 5 8

def total_water_needed_for_plants (corn pumpkin : PlantRow) : ℚ :=
  (corn.rows * corn.plants_per_row * corn.water_per_plant) +
  (pumpkin.rows * pumpkin.plants_per_row * pumpkin.water_per_plant)

def total_water_needed_for_animals (pigs ducks cows : Animal) : ℚ :=
  (pigs.count * pigs.water_per_animal) +
  (ducks.count * ducks.water_per_animal) +
  (cows.count * cows.water_per_animal)

def time_to_pump (total_water pump_rate : ℚ) : ℚ :=
  total_water / pump_rate

theorem casey_pumping_time :
  let total_water_plants := total_water_needed_for_plants corn pumpkin
  let total_water_animals := total_water_needed_for_animals pigs ducks cows
  let time_morning := time_to_pump total_water_plants morning_pump_rate
  let time_afternoon := time_to_pump total_water_animals afternoon_pump_rate
  time_morning + time_afternoon = 35 := by
sorry

end casey_pumping_time_l158_158880


namespace sum_of_integer_values_a_l158_158245

-- Define the conditions for the variable \(x\)
def eq_x(a : ℤ) : Prop := 
  ∃ x : ℝ, (ax + 3) / 2 - (2x - 1) / 3 = 1 ∧ x > 0 ∧ x = 5 / (4 - 3a)

-- Define the conditions for the system of inequalities with \(y\)
def system_y(a : ℤ) : Prop :=
  ∃ y1 y2 : ℤ, y1 + 3 > 1 ∧ 3y1 - a < 1 ∧ y2 + 3 > 1 ∧ 3y2 - a < 1 ∧ y1 ≠ y2 ∧ (-2 < y1 ∧ y1 < (a + 1) / 3) ∧ (-2 < y2 ∧ y2 < (a + 1) / 3)

-- Define the property we want to prove
theorem sum_of_integer_values_a : ∑ (a : ℤ) in {a : ℤ | eq_x(a) ∧ system_y(a)}, a = 1 := 
  sorry

end sum_of_integer_values_a_l158_158245


namespace largest_natural_number_n_l158_158993

theorem largest_natural_number_n (n : ℕ) :
  (4^995 + 4^1500 + 4^n).natAbs ∈ {x : ℕ | ∃ k : ℕ, x = k^2} ↔ n = 2004 := 
sorry

end largest_natural_number_n_l158_158993


namespace students_participated_in_both_l158_158492

theorem students_participated_in_both (total_students volleyball track field no_participation both: ℕ) 
  (h1 : total_students = 45) 
  (h2 : volleyball = 12) 
  (h3 : track = 20) 
  (h4 : no_participation = 19) 
  (h5 : both = volleyball + track - (total_students - no_participation)) 
  : both = 6 :=
by
  sorry

end students_participated_in_both_l158_158492


namespace sum_of_real_roots_eq_12_l158_158530

open Real

def polynomial1 (x : ℝ) : ℝ := x^2 - 6 * x + 5
def polynomial2 (x : ℝ) : ℝ := x^2 - 6 * x + 3

theorem sum_of_real_roots_eq_12 :
  (∑ x in (Finset.filter (λ x => (polynomial1 x) ^ polynomial2 x = 1) Finset.univ), x) = 12 :=
by
  sorry

end sum_of_real_roots_eq_12_l158_158530


namespace tan_roots_sum_correct_l158_158772

noncomputable def tan_roots_sum : ℝ :=
  let roots := {x | 0 < x ∧ x < (π / 2) ∧ tan(15 * x) = 15 * tan(x)} in
  ∑ r in roots, 1 / (tan r)^2

theorem tan_roots_sum_correct (h : ∃ s : Finset ℝ, s = {r | r ∈ {x | 0 < x ∧ x < (π / 2) ∧ tan(15 * x) = 15 * tan(x)} } ∧ s.card = 6) :
  tan_roots_sum = 78 / 5 :=
sorry

end tan_roots_sum_correct_l158_158772


namespace arccos_one_over_sqrt_two_l158_158947

theorem arccos_one_over_sqrt_two : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_l158_158947


namespace sum_of_swapped_numbers_l158_158416

-- Define the original 4x4 grid
def grid : array 4 (array 4 ℕ) :=
  array.from_list [
    array.from_list [9, 6, 3, 16],
    array.from_list [4, 13, 10, 5],
    array.from_list [14, 1, 8, 11],
    array.from_list [7, 12, 15, 2]
  ]

-- Define a property which checks if a given grid is a magic square
def isMagicSquare (g : array 4 (array 4 ℕ)) : Prop :=
  let rows := [0, 1, 2, 3].map (λ i => g[i].to_list.sum) in
  let cols := [0, 1, 2, 3].map (λ i => [0, 1, 2, 3].map (λ j => g[j][i]).sum) in
  let diag1 := [0, 1, 2, 3].map (λ i => g[i][i]).sum in
  let diag2 := [0, 1, 2, 3].map (λ i => g[i][3 - i]).sum in
  rows.all (λ s => s = 34) ∧ cols.all (λ s => s = 34) ∧ diag1 = 34 ∧ diag2 = 34

-- Helper function to swap two numbers in the grid
def swap (g : array 4 (array 4 ℕ)) (pos1 pos2 : (ℕ × ℕ)) : array 4 (array 4 ℕ) :=
  let ((r1, c1), (r2, c2)) := pos1, pos2 in
  let val1 := g[r1][c1] in
  let val2 := g[r2][c2] in
  g.modify r1 (λ row => row.modify c1 (λ _ => val2))
   .modify r2 (λ row => row.modify c2 (λ _ => val1))

-- Defining the positions to swap
def pos1 : (ℕ × ℕ) := (1, 1) -- corresponds to 13 at (1,1)
def pos2 : (ℕ × ℕ) := (3, 2) -- corresponds to 15 at (3,2)

-- Swapped grid
def swappedGrid := swap grid pos1 pos2

theorem sum_of_swapped_numbers : 
  grid[1][1] + grid[3][2] = 28 ∧ isMagicSquare swappedGrid :=
by
  sorry

end sum_of_swapped_numbers_l158_158416


namespace probability_even_sum_l158_158251

def firstTenPrimes : Set ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

def isOdd (n : ℕ) : Prop := n % 2 = 1

def distinctPairs (s : Set ℕ) : Set (ℕ × ℕ) :=
  { (x, y) | x ∈ s ∧ y ∈ s ∧ x ≠ y }

def eventEvenSum (s : Set ℕ) : Set (ℕ × ℕ) :=
  { (x, y) | x ∈ s ∧ y ∈ s ∧ x ≠ y ∧ (x + y) % 2 = 0 }

theorem probability_even_sum :
  (Set.card (eventEvenSum firstTenPrimes)).toRat / (Set.card (distinctPairs firstTenPrimes)).toRat = 4 / 5 :=
  sorry

end probability_even_sum_l158_158251


namespace find_cost_price_l158_158837

theorem find_cost_price (C S : ℝ) (h1 : S = 1.35 * C) (h2 : S - 25 = 0.98 * C) : C = 25 / 0.37 :=
by
  sorry

end find_cost_price_l158_158837


namespace digit_200_of_5_div_13_is_8_l158_158428

/-- Prove that the 200th digit beyond the decimal point in the decimal representation
    of 5/13 is 8 --/
theorem digit_200_of_5_div_13_is_8 :
  let repeating_sequence := "384615" in
  let digit_200 := repeating_sequence[200 % 6 -1] in
  digit_200 = '8' :=
by
  sorry

end digit_200_of_5_div_13_is_8_l158_158428


namespace csc_squared_product_l158_158448

theorem csc_squared_product :
  let m := 2
  let n := 29
  let prod_csc := ∏ k in Finset.range 30, (1 / (Real.sin (3 * (k + 1)) * π / 180))^2
  m^n = prod_csc ∧ m + n = 31 :=
by
  have h1 : m = 2 := rfl
  have h2 : n = 29 := rfl
  have h3 : prod_csc = (2^29) := sorry
  rw [h1, h2, h3]
  exact ⟨rfl, rfl⟩

end csc_squared_product_l158_158448


namespace gcd_45_81_63_l158_158228

theorem gcd_45_81_63 : Nat.gcd 45 (Nat.gcd 81 63) = 9 := 
sorry

end gcd_45_81_63_l158_158228


namespace sum_real_solutions_abs_eq_l158_158404

theorem sum_real_solutions_abs_eq :
  (∑ x in {x : ℝ | |x - 1| = 3 * |x + 3|}, x) = -7 :=
sorry

end sum_real_solutions_abs_eq_l158_158404


namespace find_x_set_eq_l158_158685

noncomputable def f : ℝ → ℝ :=
sorry -- The actual definition of f according to its properties is omitted

lemma odd_function (x : ℝ) : f (-x) = -f x :=
sorry

lemma periodic_function (x : ℝ) : f (x + 2) = -f x :=
sorry

lemma f_definition (x : ℝ) (h : 0 ≤ x ∧ x ≤ 1) : f x = 1 / 2 * x :=
sorry

theorem find_x_set_eq (x : ℝ) : (f x = -1 / 2) ↔ (∃ k : ℤ, x = 4 * k - 1) :=
sorry

end find_x_set_eq_l158_158685


namespace curve_is_hyperbola_l158_158153

theorem curve_is_hyperbola (u : ℝ) (x y : ℝ) 
  (h1 : x = Real.cos u ^ 2)
  (h2 : y = Real.sin u ^ 4) : 
  ∃ (a b : ℝ), a ≠ 0 ∧  b ≠ 0 ∧ x / a ^ 2 - y / b ^ 2 = 1 := 
sorry

end curve_is_hyperbola_l158_158153


namespace quadratic_real_roots_l158_158634

theorem quadratic_real_roots (k : ℝ) : (∃ x : ℝ, k * x^2 - 6 * x + 9 = 0) ↔ (k ≤ 1 ∧ k ≠ 0) :=
  sorry

end quadratic_real_roots_l158_158634


namespace quadratic_two_distinct_real_roots_l158_158243

theorem quadratic_two_distinct_real_roots (m : ℝ) (h : -4 * m > 0) : m = -1 :=
sorry

end quadratic_two_distinct_real_roots_l158_158243


namespace tangent_line_correct_l158_158542

noncomputable def f (x : ℝ) : ℝ := exp (-5 * x) + 2
def point : ℝ × ℝ := (0, 3)
def tangent_line (x : ℝ) : ℝ := -5 * x + 3

theorem tangent_line_correct : 
    ∀ x y, (y = f x) → x = 0 → y = 3 → (∀ t, tangent_line t = -5 * t + 3) := 
by
  sorry

end tangent_line_correct_l158_158542


namespace last_ball_is_red_l158_158063

-- Define the initial conditions.
def initial_blue_balls : ℕ := 1001
def initial_red_balls : ℕ := 1000
def initial_green_balls : ℕ := 1000

-- Define the operation rules as inductive definitions.
inductive Operation
| Rule1 : Operation  -- Take out one blue and one green, put back one red.
| Rule2 : Operation  -- Take out one red and one green, put back one red.
| Rule3 : Operation  -- Take out two red, put back two blue.
| Rule4 : Operation  -- All other cases, put back one green.

-- Define the state of balls in the bottle.
structure State :=
(blue : ℕ)
(red : ℕ)
(green : ℕ)

-- Define a function that applies an operation to a state.
def apply_operation (s : State) (op : Operation) : State :=
match op with
| Operation.Rule1 => { s with blue := s.blue - 1, green := s.green - 1, red := s.red + 1 }
| Operation.Rule2 => { s with red := s.red + 0, green := s.green - 1, red := s.red - 0 }
| Operation.Rule3 => { s with red := s.red - 2, blue := s.blue + 2 }
| Operation.Rule4 => { s with green := s.green + 0 }  -- Cases like blue & blue or green & green.

-- Define the main theorem to prove.
theorem last_ball_is_red
  (s : State := { blue := initial_blue_balls, red := initial_red_balls, green := initial_green_balls })
  : ∀ (h : s.blue + s.red + s.green = 1), s.red = 1 :=
sorry

end last_ball_is_red_l158_158063


namespace proof_problem_l158_158162

variable (α β : ℝ) (a b : ℝ × ℝ) (m : ℝ)
variable (hα : 0 < α ∧ α < Real.pi / 4)
variable (hβ : β = Real.pi)
variable (ha_def : a = (Real.tan (α + β / 4) - 1, 0))
variable (hb : b = (Real.cos α, 2))
variable (ha_dot : a.1 * b.1 + a.2 * b.2 = m)

-- Proof statement
theorem proof_problem :
  (0 < α ∧ α < Real.pi / 4) ∧
  β = Real.pi ∧
  a = (Real.tan (α + β / 4) - 1, 0) ∧
  b = (Real.cos α, 2) ∧
  (a.1 * b.1 + a.2 * b.2 = m) →
  (2 * Real.cos α * Real.cos α + Real.sin (2 * (α + β))) / (Real.cos α - Real.sin β) = 2 * (m + 2) := by
  sorry

end proof_problem_l158_158162


namespace mean_home_runs_is_correct_l158_158757

variable (n₅ n₆ n₇ n₈ n₁₁ : ℕ)
variable (mean : ℝ)

def total_home_runs : ℕ :=
  (5 * n₅) + (6 * n₆) + (7 * n₇) + (8 * n₈) + (11 * n₁₁)

def total_players : ℕ :=
  n₅ + n₆ + n₇ + n₈ + n₁₁

theorem mean_home_runs_is_correct :
  n₅ = 4 → n₆ = 3 → n₇ = 2 → n₈ = 1 → n₁₁ = 1 →
  mean = (71.0 / 11.0) →
  total_home_runs n₅ n₆ n₇ n₈ n₁₁ = 71 ∧
  total_players n₅ n₆ n₇ n₈ n₁₁ = 11 ∧
  mean = 6.45 :=
begin
  intros h₅ h₆ h₇ h₈ h₁₁ hmean,
  simp [total_home_runs, total_players, *],
  rw [h₅, h₆, h₇, h₈, h₁₁],
  norm_num,
  split,
  { exact calc (5 * 4) + (6 * 3) + (7 * 2) + 8 + 11 = 71 },
  split,
  { exact calc 4 + 3 + 2 + 1 + 1 = 11 },
  { exact hmean }
end

end mean_home_runs_is_correct_l158_158757


namespace cost_per_box_per_month_l158_158473

theorem cost_per_box_per_month 
  (length width height : ℕ)
  (total_volume : ℕ)
  (total_cost_per_month : ℕ)
  (total_boxes_volume_eq: total_volume = 15 * 12 * 10)
  (total_cost_eq : total_cost_per_month = 300)
  (volume_eq : 15 * 12 * 10 = 1800)
  (total_volume_eq: total_volume = 1080000):
  ∀ boxes, boxes = total_volume / (15 * 12 * 10) 
  → total_cost_per_month.to_rat / boxes.to_rat = 0.5 :=
by
  intros
  sorry

end cost_per_box_per_month_l158_158473


namespace two_hundredth_digit_of_5_over_13_l158_158422

theorem two_hundredth_digit_of_5_over_13 :
  (∀ n : ℕ, n ≥ 0 →
    let d := "384615".charAt ( (n % 6) - 1 ) in
    if n % 6 = 0 then '5' else d  = 8) :=
sorry

end two_hundredth_digit_of_5_over_13_l158_158422


namespace element_in_set_l158_158347

-- Define the set M
def M : Set ℝ := {x | x ≤ 3 * Real.sqrt 3}

-- Define the element a
def a : ℝ := 2 * Real.sqrt 6

-- Prove the statement
theorem element_in_set : a ∈ M := by
  sorry

end element_in_set_l158_158347


namespace two_hundredth_digit_div_five_thirteen_l158_158424

theorem two_hundredth_digit_div_five_thirteen : 
  (let n := 200 
       d := 13 
       repeating_cycle := "384615".to_list -- Convert repeating cycle to list of digits
       cycle_length := repeating_cycle.length
       division := n / cycle_length
       remainder := n % cycle_length in
   repeating_cycle.nth remainder = some '8') :=
by sorry

end two_hundredth_digit_div_five_thirteen_l158_158424


namespace incorrect_statement_c_l158_158657

-- We define the problem context and specify the statements
def survey_population : Prop := 800  -- Defining the entire population
def survey_sample : Prop := 200  -- Defining the size of the sample

def is_sample_size_correct : Prop := survey_sample = 200
def is_individual_opinion : Prop := ∀ student : ℕ, student ∈ survey_sample → True
def is_population : Prop := survey_sample = survey_population
def is_sample_of_population : Prop := survey_sample < survey_population

-- Prove that the statement "The preference level of 200 students is the population" is incorrect
theorem incorrect_statement_c : ¬ is_population :=
by
  intro h
  have contra := (survey_sample < survey_population)
  sorry  -- Proof steps go here

end incorrect_statement_c_l158_158657


namespace roots_of_cubic_eq_l158_158229

theorem roots_of_cubic_eq (p q r : ℝ) 
    (h : polynomial.roots (polynomial.C (-r^3) + 3 * polynomial.C q^2 * polynomial.X + -3 * polynomial.C p * polynomial.X^2 + polynomial.X^3) = {p, q, r}) : 
    p = q ∧ q = r := 
sorry

end roots_of_cubic_eq_l158_158229


namespace simplify_expression_l158_158043

theorem simplify_expression :
  (4 + 2 + 6) / 3 - (2 + 1) / 3 = 3 := by
  sorry

end simplify_expression_l158_158043


namespace friends_meeting_distance_l158_158461

theorem friends_meeting_distance (R_q : ℝ) (t : ℝ) (D_p D_q trail_length : ℝ) :
  trail_length = 36 ∧ D_p = 1.25 * R_q * t ∧ D_q = R_q * t ∧ D_p + D_q = trail_length → D_p = 20 := by
  sorry

end friends_meeting_distance_l158_158461


namespace total_number_of_boys_in_camp_l158_158641

theorem total_number_of_boys_in_camp (T : ℕ)
  (hA1 : ∃ (boysA : ℕ), boysA = 20 * T / 100)
  (hA2 : ∀ (boysS : ℕ) (boysM : ℕ), boysS = 30 * boysA / 100 ∧ boysM = 40 * boysA / 100)
  (hB1 : ∃ (boysB : ℕ), boysB = 30 * T / 100)
  (hB2 : ∀ (boysS : ℕ) (boysM : ℕ), boysS = 25 * boysB / 100 ∧ boysM = 35 * boysB / 100)
  (hC1 : ∃ (boysC : ℕ), boysC = 50 * T / 100)
  (hC2 : ∀ (boysS : ℕ) (boysM : ℕ), boysS = 15 * boysC / 100 ∧ boysM = 45 * boysC / 100)
  (hA_no_SM : 77 = 70 * boysA / 100)
  (hB_no_SM : 72 = 60 * boysB / 100)
  (hC_no_SM : 98 = 60 * boysC / 100) :
  T = 535 :=
by
  sorry

end total_number_of_boys_in_camp_l158_158641


namespace symmetric_point_x_axis_l158_158383

/-- The coordinates of the point symmetric to P(3, -5) with respect to the x-axis are (3, 5). -/
theorem symmetric_point_x_axis (x y : ℤ) (h : x = 3 ∧ y = -5) :
  ∃ x' y', x' = 3 ∧ y' = -y ∧ x' = 3 ∧ y' = 5 :=
by
  use [3, 5]
  sorry

end symmetric_point_x_axis_l158_158383


namespace smallest_x_value_l158_158372

theorem smallest_x_value : ∃ x : ℤ, ∃ y : ℤ, (xy + 7 * x + 6 * y = -8) ∧ x = -40 :=
by
  sorry

end smallest_x_value_l158_158372


namespace cube_root_367_cube_root_neg_0_003670_l158_158611

-- Definitions based on given conditions
def cube_root_0_3670_eq_0_7160 : Prop := (real.cbrt 0.3670 = 0.7160)
def cube_root_3_670_eq_1_542 : Prop := (real.cbrt 3.670 = 1.542)

-- Main propositions to prove based on conditions
theorem cube_root_367 (h1 : cube_root_0_3670_eq_0_7160) : real.cbrt 367 = 7.160 := 
  by sorry

theorem cube_root_neg_0_003670 (h2 : cube_root_3_670_eq_1_542) : real.cbrt (-0.003670) = -0.1542 := 
  by sorry 

end cube_root_367_cube_root_neg_0_003670_l158_158611


namespace arccos_sqrt2_l158_158941

def arccos_eq (x : ℝ) := arccos x
def range_arccos := ∀ (x : ℝ), 0 ≤ arccos x ∧ arccos x ≤ π
def cos_pi_div_four : Real.cos (π / 4) = 1 / Real.sqrt 2 := by sorry

theorem arccos_sqrt2 :
  arccos_eq (1 / Real.sqrt 2) = π / 4 :=
by
  have h1 : Real.cos (π / 4) = 1 / Real.sqrt 2, from cos_pi_div_four
  sorry

end arccos_sqrt2_l158_158941


namespace overtaking_points_l158_158160

-- defining constants for the given conditions
def L : ℕ := 55
def x : ℝ := 1 -- unit time constant, can be defined as literal 1 for simplicity
def vp : ℝ := 100 * x
def vc : ℝ := 155 * x

-- main statement: the pedestrian is overtaken at exactly 11 different points
theorem overtaking_points : 
  (∀ (L vp vc : ℕ) (x : ℝ), L = 55 ∧ vp = 100 * x ∧ vc = 155 * x → ∃ n : ℕ, n = 11) :=
by
  intro L vp vc x h
  cases h with hL hV
  cases hV with hVp hVc
  use 11
  sorry

end overtaking_points_l158_158160


namespace value_of_x_plus_y_l158_158616

noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem value_of_x_plus_y
  (x y : ℝ)
  (h1 : x ≥ 1)
  (h2 : y ≥ 1)
  (h3 : x * y = 10)
  (h4 : x^(lg x) * y^(lg y) ≥ 10) :
  x + y = 11 :=
  sorry

end value_of_x_plus_y_l158_158616


namespace wam_gm_inequality_l158_158683

variable (k b : ℝ)
variable (hk : 0 < k ∧ k < 1) (hb : b > 0)

def a := k * b
def c := (a + b) / 2
def WAM := (2 * a + 3 * b + 4 * c) / 9
def GM := Real.cbrt (a * b * c)
def diff := WAM - GM
def rhs := ((b - a) ^ 2) / (8 * a)

theorem wam_gm_inequality : diff < rhs := sorry

end wam_gm_inequality_l158_158683


namespace termite_ridden_not_collapsing_fraction_l158_158704

theorem termite_ridden_not_collapsing_fraction (h1 : (5 : ℚ) / 8) (h2 : (11 : ℚ) / 16) :
  (5 : ℚ) / 8 - ((5 : ℚ) / 8) * ((11 : ℚ) / 16) = 25 / 128 :=
by
  sorry

end termite_ridden_not_collapsing_fraction_l158_158704


namespace smarandache_16_smarandache_2016_smarandache_max_7_smarandache_infinitely_many_composites_l158_158000

-- Define the Smarandache function S
def smarandache_function (n : ℕ) : ℕ := 
  Nat.find (λ m, n ∣ Nat.factorial m)

-- Prove S(16) = 6
theorem smarandache_16 : smarandache_function 16 = 6 :=
  sorry

-- Prove S(2016) = 8
theorem smarandache_2016 : smarandache_function 2016 = 8 :=
  sorry

-- Prove the maximum n for S(n) = 7 is 7!
theorem smarandache_max_7 : ∃ n, smarandache_function n = 7 ∧ n = Nat.factorial 7 :=
  sorry

-- Prove there are infinitely many composite numbers n with S(n) = p where p is the largest prime factor of n
theorem smarandache_infinitely_many_composites (p : ℕ) [hp : Fact (Nat.prime p)] :
  ∃ᶠ n in at_top, (∃ m, n = 2 * m) ∧ (smarandache_function n = p ∧ n ≠ p) :=
  sorry

end smarandache_16_smarandache_2016_smarandache_max_7_smarandache_infinitely_many_composites_l158_158000


namespace solve_diophantine_l158_158369

theorem solve_diophantine :
  {xy : ℤ × ℤ | 5 * (xy.1 ^ 2) + 5 * xy.1 * xy.2 + 5 * (xy.2 ^ 2) = 7 * xy.1 + 14 * xy.2} = {(-1, 3), (0, 0), (1, 2)} :=
by sorry

end solve_diophantine_l158_158369


namespace distance_between_centers_of_internally_tangent_circles_l158_158522

theorem distance_between_centers_of_internally_tangent_circles 
  (O₁ O₂ A : Point) (r₁ r₂ : ℝ) 
  (h₁ : dist O₁ A = r₁)
  (h₂ : dist O₂ A = r₂)
  (h₃ : dist O₁ O₂ = r₁ + r₂) :
  dist O₁ O₂ = r₁ + r₂ := 
sorry

end distance_between_centers_of_internally_tangent_circles_l158_158522


namespace y_intercept_of_line_l158_158999

theorem y_intercept_of_line (x y : ℝ) (h : 5 * x - 3 * y = 15) : (0, -5) = (0, (-5 : ℝ)) :=
by
  sorry

end y_intercept_of_line_l158_158999


namespace length_MD_measure_angle_KMD_l158_158177

-- Definitions of given conditions
variables {A B C D K M : Type}
variable [EuclideanGeometry A B C D K M]

-- Parallelogram ABCD
axiom is_parallelogram : parallelogram A B C D

-- M is midpoint of BC
axiom M_midpoint_BC : midpoint M B C

-- K on AD such that BK = BM
axiom K_on_AD : ∃ K ∈ segment A D, dist B K = dist B M

-- KBMD is a cyclic quadrilateral
axiom KBMD_cyclic : cyclic_quad K B M D

-- Given lengths and angles
axiom AD_is_19 : dist A D = 19
axiom angle_BAD_is_44 : angle B A D = 44

-- Prove the length of MD equals 9.5
theorem length_MD : dist M D = 9.5 := sorry

-- Prove the measure of angle KMD equals 44 degrees
theorem measure_angle_KMD : angle K M D = 44 := sorry

end length_MD_measure_angle_KMD_l158_158177


namespace sqrt_equivalent_is_x_7_4_l158_158104

noncomputable def sqrt_equivalent (x : ℝ) (hx : 0 < x) : ℝ :=
  sqrt (x^3 * sqrt x)

theorem sqrt_equivalent_is_x_7_4 (x : ℝ) (hx : 0 < x) :
  sqrt_equivalent x hx = x^(7/4) :=
by
  sorry

end sqrt_equivalent_is_x_7_4_l158_158104


namespace arccos_of_one_over_sqrt_two_l158_158886

theorem arccos_of_one_over_sqrt_two :
  Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 := 
sorry

end arccos_of_one_over_sqrt_two_l158_158886


namespace train_length_correct_l158_158495

-- Define the conditions
def speed_km_per_hr : ℝ := 144
def time_sec : ℝ := 0.49996000319974404

-- Convert speed from km/hr to m/s
def speed_m_per_s : ℝ := speed_km_per_hr * (1000 / 3600)

-- Calculate the length of the train
def length_of_train : ℝ := speed_m_per_s * time_sec

-- Statement for the proof
theorem train_length_correct :
  length_of_train = 19.998400127989762 :=
by
  sorry

end train_length_correct_l158_158495


namespace find_d_l158_158233

theorem find_d (a d : ℝ) (h : ∀ x : ℝ, (x + 3) * (x + a) = x^2 + d * x + 12) :
  d = 7 :=
sorry

end find_d_l158_158233


namespace acute_triangle_values_l158_158097

-- Definitions and conditions
def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def is_acute (a b c : ℕ) : Prop :=
  a^2 + b^2 > c^2 ∧ a^2 + c^2 > b^2 ∧ b^2 + c^2 > a^2

-- Main statement we want to prove
theorem acute_triangle_values :
  ∃ (k : ℕ), k ∈ {16, 17, 18} ∧
             is_triangle 8 17 k ∧
             is_acute 8 17 k :=
begin
  sorry
end

end acute_triangle_values_l158_158097


namespace number_of_blue_balls_l158_158807

theorem number_of_blue_balls (T : ℕ) (h1 : (1 / 4) * T = green) (h2 : (1 / 8) * T = blue)
    (h3 : (1 / 12) * T = yellow) (h4 : 26 = white) (h5 : green + blue + yellow + white = T) :
    blue = 6 :=
by
  sorry

end number_of_blue_balls_l158_158807


namespace valid_c_values_count_l158_158526

theorem valid_c_values_count :
  {c : ℕ | 0 ≤ c ∧ c ≤ 2000 ∧ ∃ x : ℝ, 9 * (floor x) + 3 * (ceil x) = c}.card = 333 :=
sorry

end valid_c_values_count_l158_158526


namespace arccos_one_over_sqrt_two_eq_pi_over_four_l158_158902

theorem arccos_one_over_sqrt_two_eq_pi_over_four : 
  Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_eq_pi_over_four_l158_158902


namespace arccos_one_over_sqrt_two_l158_158949

theorem arccos_one_over_sqrt_two : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_l158_158949


namespace square_area_l158_158353

-- Define the problem conditions
variables {A B C D E F : Point}
variable {ABCD : square A B C D}
variable {E_on_AD : (E ∈ segment A D)}
variable {F_on_BC : (F ∈ segment B C)}
variable {BE : length B E = 20}
variable {EF : length E F = 40}
variable {FD : length F D = 20}

-- The goal statement to prove
theorem square_area (ABCD : square A B C D)
    (E_on_AD : E ∈ segment A D)
    (F_on_BC : F ∈ segment B C)
    (BE : length B E = 20)
    (EF : length E F = 40)
    (FD : length F D = 20) :
    (area ABCD = 6400) :=
by
  sorry

end square_area_l158_158353


namespace volume_comparison_l158_158569

-- Define the volumes based on the given conditions.
noncomputable def volume_cube(S : ℝ) : ℝ :=
  let a := (S / 6)^(1/2) in 
  a^3

noncomputable def volume_cylinder(S : ℝ) : ℝ :=
  let R := (S / (6 * Real.pi))^(1/2) in 
  2 * Real.pi * R^3

noncomputable def volume_sphere(S : ℝ) : ℝ :=
  let R := (S / (4 * Real.pi))^(1/2) in 
  (4 / 3) * Real.pi * R^3

-- The proof statement
theorem volume_comparison (S : ℝ) (hS : 0 < S) :
  volume_cube(S) < volume_cylinder(S) ∧ volume_cylinder(S) < volume_sphere(S) := by
  sorry

end volume_comparison_l158_158569


namespace quadratic_transformation_l158_158847

theorem quadratic_transformation (a b c : ℝ) (h : a * (x - 1)^2 + b * (x - 1) + c = 2 * x^2 - 3 * x - 1) : 
  a = 2 ∧ b = 1 ∧ c = -2 := by
sorry

end quadratic_transformation_l158_158847


namespace smallest_angle_of_trapezoid_l158_158382

theorem smallest_angle_of_trapezoid (a d : ℝ) (h1 : a + 3 * d = 120) (h2 : 4 * a + 6 * d = 360) :
  a = 60 := by
  sorry

end smallest_angle_of_trapezoid_l158_158382


namespace paint_cans_needed_l158_158516

-- Conditions as definitions
def bedrooms : ℕ := 3
def other_rooms : ℕ := 2 * bedrooms
def paint_per_room : ℕ := 2
def color_can_capacity : ℕ := 1
def white_can_capacity : ℕ := 3

-- Total gallons needed
def total_color_gallons_needed : ℕ := paint_per_room * bedrooms
def total_white_gallons_needed : ℕ := paint_per_room * other_rooms

-- Total cans needed
def total_color_cans_needed : ℕ := total_color_gallons_needed / color_can_capacity
def total_white_cans_needed : ℕ := total_white_gallons_needed / white_can_capacity
def total_cans_needed : ℕ := total_color_cans_needed + total_white_cans_needed

theorem paint_cans_needed : total_cans_needed = 10 := by
  -- Proof steps (skipped) to show total_cans_needed = 10
  sorry

end paint_cans_needed_l158_158516


namespace min_value_of_a_l158_158589

theorem min_value_of_a (a : ℝ) :
  (∀ x : ℝ, 0 < x ∧ x ≤ 1 / 2 → x^2 + 2 * a * x + 1 ≥ 0) → a ≥ -5 / 4 := 
sorry

end min_value_of_a_l158_158589


namespace quadratic_real_roots_l158_158633

theorem quadratic_real_roots (k : ℝ) : (∃ x : ℝ, k * x^2 - 6 * x + 9 = 0) ↔ (k ≤ 1 ∧ k ≠ 0) :=
  sorry

end quadratic_real_roots_l158_158633


namespace set_union_is_real_l158_158188

def A : Set ℝ := {x | x > -1}
def B : Set ℝ := {x | x^2 - x - 2 ≥ 0}

theorem set_union_is_real :
  A ∪ B = set.univ := by
  sorry

end set_union_is_real_l158_158188


namespace total_chocolate_bars_in_large_box_l158_158058

-- Define the given conditions
def small_boxes : ℕ := 16
def chocolate_bars_per_box : ℕ := 25

-- State the proof problem
theorem total_chocolate_bars_in_large_box :
  small_boxes * chocolate_bars_per_box = 400 :=
by
  -- The proof is omitted
  sorry

end total_chocolate_bars_in_large_box_l158_158058


namespace infinite_series_sum_l158_158032

-- Define the polynomial P(x)
def P (x : ℝ) : ℝ := 30 * x^3 - 50 * x^2 + 22 * x - 1

-- Define a set of conditions ensuring roots a, b, c are between 0 and 1
def roots_in_unit_interval (a b c : ℝ) :=
  0 < a ∧ a < 1 ∧ 0 < b ∧ b < 1 ∧ 0 < c ∧ c < 1

-- The main theorem to state the problem
theorem infinite_series_sum (a b c : ℝ) (h_roots : roots_in_unit_interval a b c)
  (h_root1 : P(a) = 0) (h_root2 : P(b) = 0) (h_root3 : P(c) = 0) :
  (∑' n : ℕ, a^n + b^n + c^n) = 12 :=
sorry

end infinite_series_sum_l158_158032


namespace cos_neg_52_over_3_pi_l158_158067

theorem cos_neg_52_over_3_pi : cos (- (52 / 3) * Real.pi) = -1 / 2 :=
by
  sorry

end cos_neg_52_over_3_pi_l158_158067


namespace overtaking_points_count_l158_158157

def track_length : ℝ := 55
def pedestrian_speed (x : ℝ) : ℝ := 100 * x
def cyclist_speed (x : ℝ) : ℝ := 155 * x

theorem overtaking_points_count (x : ℝ) > 0 : ∃ n : ℕ, n = 11 :=
by
  sorry

end overtaking_points_count_l158_158157


namespace cyclic_quadrilateral_AB_l158_158814

variables {A B C D X : Type}
variables (BC BX XD AC : ℝ)

-- Given conditions
axiom points_circle : ∃ (A B C D : Type), True
axiom intersects : ∃ X, True
axiom BC_val : BC = 6
axiom BX_val : BX = 4
axiom XD_val : XD = 5
axiom AC_val : AC = 11

-- Statement to be proved
theorem cyclic_quadrilateral_AB :
  ∃ (AB : ℝ), AB = 6 :=
by {
  sorry -- Proof omitted
}

end cyclic_quadrilateral_AB_l158_158814


namespace find_monic_quadratic_polynomial_l158_158145

theorem find_monic_quadratic_polynomial (x : ℝ) :
  ∀ (p : Polynomial ℝ), p.monic ∧ (p.eval (1 - 2 * I) = 0) → p = Polynomial.X^2 - 2 * Polynomial.X + 5 :=
by
  sorry

end find_monic_quadratic_polynomial_l158_158145


namespace point_on_graph_sum_l158_158747

theorem point_on_graph_sum {f : ℝ → ℝ} (h : (2, 3) ∈ set_of (λ p : ℝ × ℝ, p.2 = f(p.1) / 2)) :
  (∃ p : ℝ × ℝ, p = (6, f⁻¹ 6 / 2) ∧ p.1 + p.2 = 7) :=
by
  sorry

end point_on_graph_sum_l158_158747


namespace equilateral_triangle_perimeter_l158_158836

theorem equilateral_triangle_perimeter :
  ∃ (line_through_origin : ℝ → ℝ), ∀ x : ℝ, y : ℝ,
  (line_through_origin 0 = 0) ∧
  (line_through_origin 1 = y) ∧
  (y = -ℓ * x ∨ y = 1 + x / ℓ) ∧
  (∀ x1 y1 x2 y2, (x1 = 1 ∧ y1 = line_through_origin x1) ∧ 
                (x2 = 1 ∧ y2 = 1 + x2 / √3) →
                distance (x1, y1) (x2, y2) = _) →
  (3 * _) = 3 + 4 * √3 :=
sorry

end equilateral_triangle_perimeter_l158_158836


namespace sequence_condition_sum_remainder_mod_500_l158_158984
noncomputable def b : ℕ → ℕ
| 0 := 2
| 1 := 3
| 2 := 5
| (n+3) := b n + b (n+1) + b (n+2)

def b15 := 9857
def b16 := 18150
def b17 := 33407

theorem sequence_condition : b 15 = b15 ∧ b 16 = b16 ∧ b 17 = b17 :=
by {
  split,
  { unfold b, sorry }, -- Verification omitted
  split,
  { unfold b, sorry }, -- Verification omitted
  { unfold b, sorry }  -- Verification omitted
}

theorem sum_remainder_mod_500 
  (sum_15 := ∑ k in finset.range 15, b k) :
  sum_15 % 500 = x :=
sorry -- Proof omitted

end sequence_condition_sum_remainder_mod_500_l158_158984


namespace jame_profit_l158_158326

/-- Define the initial investment -/
def initial_investment : ℝ := 40000.0

/-- Define the initial number of cattle -/
def number_of_cattle : ℕ := 100

/-- Define the initial weight of each cattle -/
def initial_weight_per_cattle : ℝ := 1000.0

/-- Define the monthly weight increase rates -/
def weight_increase_rates : List ℝ := [0.01, 0.015, 0.005, 0.01, 0.02, 0.03]

/-- Define the monthly feeding cost per head multiplier -/
def feeding_cost_multiplier : ℝ := 0.2

/-- Define the selling price per pound in June -/
def selling_price_per_pound : ℝ := 2.2

/-- Define the tax rate on the selling price -/
def tax_rate : ℝ := 0.05

/-- Define the transportation cost per cattle -/
def transportation_cost_per_cattle : ℝ := 500.0

theorem jame_profit
  (initial_investment : ℝ)
  (number_of_cattle : ℕ)
  (initial_weight_per_cattle : ℝ)
  (weight_increase_rates : List ℝ)
  (feeding_cost_multiplier : ℝ)
  (selling_price_per_pound : ℝ)
  (tax_rate : ℝ)
  (transportation_cost_per_cattle : ℝ) :
  let final_weight := List.foldl (λ (weight : ℝ) (rate : ℝ), weight * (1 + rate)) initial_weight_per_cattle weight_increase_rates,
      total_selling_price := final_weight * selling_price_per_pound * number_of_cattle,
      monthly_feeding_cost := (initial_investment / number_of_cattle) * feeding_cost_multiplier,
      total_feeding_cost := monthly_feeding_cost * List.length weight_increase_rates * number_of_cattle,
      selling_tax := total_selling_price * tax_rate,
      total_transportation_cost := transportation_cost_per_cattle * number_of_cattle,
      profit := total_selling_price - initial_investment - total_feeding_cost - selling_tax - total_transportation_cost in
  profit = 90485.70 :=
by
  sorry

end jame_profit_l158_158326


namespace max_daily_sales_l158_158087

def f (t : ℕ) : ℝ := -2 * t + 200
def g (t : ℕ) : ℝ :=
  if t ≤ 30 then 12 * t + 30
  else 45

def S (t : ℕ) : ℝ := f t * g t

theorem max_daily_sales : ∃ t, 1 ≤ t ∧ t ≤ 50 ∧ S t = 54600 := 
  sorry

end max_daily_sales_l158_158087


namespace find_a2016_l158_158317

-- Define the sequence according to the conditions
def seq (a : ℕ → ℤ) : Prop :=
  a 1 = 1 ∧ a 2 = 5 ∧ ∀ n : ℕ, n ≥ 1 → a (n + 2) = a (n + 1) - a n

-- State the main theorem we want to prove
theorem find_a2016 :
  ∃ a : ℕ → ℤ, seq a ∧ a 2016 = -4 :=
by
  sorry

end find_a2016_l158_158317


namespace length_BA_correct_area_ABCDE_correct_l158_158320

variables {BE CD CE CA : ℝ}
axiom BE_eq : BE = 13
axiom CD_eq : CD = 3
axiom CE_eq : CE = 10
axiom CA_eq : CA = 10

noncomputable def length_BA : ℝ := 3
noncomputable def area_ABCDE : ℝ := 4098 / 61

theorem length_BA_correct (h1 : BE = 13) (h2 : CD = 3) (h3 : CE = 10) (h4 : CA = 10) :
  length_BA = 3 := 
by { sorry }

theorem area_ABCDE_correct (h1 : BE = 13) (h2 : CD = 3) (h3 : CE = 10) (h4 : CA = 10) :
  area_ABCDE = 4098 / 61 := 
by { sorry }

end length_BA_correct_area_ABCDE_correct_l158_158320


namespace fraction_vegan_soyfree_l158_158873

theorem fraction_vegan_soyfree :
  ∀ (total_dishes vegan_dishes soy_dishes : ℕ),
    vegan_dishes = 5 →
    total_dishes = vegan_dishes * 4 →
    soy_dishes = 4 →
    (vegan_dishes - soy_dishes) / total_dishes = 1 / 20 :=
by
  intros total_dishes vegan_dishes soy_dishes,
  sorry

end fraction_vegan_soyfree_l158_158873


namespace arccos_sqrt2_l158_158939

def arccos_eq (x : ℝ) := arccos x
def range_arccos := ∀ (x : ℝ), 0 ≤ arccos x ∧ arccos x ≤ π
def cos_pi_div_four : Real.cos (π / 4) = 1 / Real.sqrt 2 := by sorry

theorem arccos_sqrt2 :
  arccos_eq (1 / Real.sqrt 2) = π / 4 :=
by
  have h1 : Real.cos (π / 4) = 1 / Real.sqrt 2, from cos_pi_div_four
  sorry

end arccos_sqrt2_l158_158939


namespace arccos_identity_l158_158922

theorem arccos_identity : 
  ∀ x : ℝ, x = 1 / real.sqrt 2 → real.arccos x = real.pi / 4 :=
begin
  intros x hx,
  have h1 : real.cos (real.pi / 4) = 1 / real.sqrt 2 := by sorry,
  have h2 : real.arccos (1 / real.sqrt 2) = real.pi / 4,
  { rw ← h1,
    rw real.arccos_cos,
    exact_mod_cast hx },
  exact h2,
end

end arccos_identity_l158_158922


namespace translation_is_B_l158_158802

/-
 Four sets of distances moved by the legs of a dining table.
 We need to prove that the set B represents translation.
-/

def is_translation (distances : List ℝ) : Prop :=
  distances.all (λ d, d = distances.head!)

def distances_A : List ℝ := [10.8, 11.1, 11.1, 11.2]
def distances_B : List ℝ := [5.1, 5.1, 5.1, 5.1]
def distances_C : List ℝ := [3.1, 4.3, 5.5, 4.3]
def distances_D : List ℝ := [0, Real.pi / 2, Real.sqrt 2 * Real.pi / 2, Real.pi / 2]

theorem translation_is_B :
  is_translation distances_B ∧
  ¬ is_translation distances_A ∧
  ¬ is_translation distances_C ∧
  ¬ is_translation distances_D :=
by
  sorry

end translation_is_B_l158_158802


namespace find_lamp_cost_l158_158673

def lamp_and_bulb_costs (L B : ℝ) : Prop :=
  B = L - 4 ∧ 2 * L + 6 * B = 32

theorem find_lamp_cost : ∃ L : ℝ, ∃ B : ℝ, lamp_and_bulb_costs L B ∧ L = 7 :=
by
  sorry

end find_lamp_cost_l158_158673


namespace subset_m_values_l158_158604

theorem subset_m_values
  {A B : Set ℝ}
  (hA : A = { x | x^2 + x - 6 = 0 })
  (hB : ∃ m, B = { x | m * x + 1 = 0 })
  (h_subset : ∀ {x}, x ∈ B → x ∈ A) :
  (∃ m, m = -1/2 ∨ m = 0 ∨ m = 1/3) :=
sorry

end subset_m_values_l158_158604


namespace napoleon_jelly_beans_l158_158352

variable (N S M : ℕ)

def sedrich_jelly_beans (N : ℕ) : ℕ := N + 4

def mikey_jelly_beans : ℕ := 19

def twice_sum_jelly_beans (N S : ℕ) : ℕ := 2 * (N + S)

def four_times_mikey_jelly_beans (M : ℕ) : ℕ := 4 * M

theorem napoleon_jelly_beans :
  ∀ (N : ℕ), let S := sedrich_jelly_beans N in
  let M := mikey_jelly_beans in
  twice_sum_jelly_beans N S = four_times_mikey_jelly_beans M → N = 17 := 
by
  sorry

end napoleon_jelly_beans_l158_158352


namespace evaluate_statements_l158_158502

theorem evaluate_statements :
  (¬ (∀ l1 l2 : Line, are_parallel l1 l2 → ∀ c1 c2 : Angle, corresponding c1 c2 → c1 = c2)) ∧
  (∀ a1 a2 : Angle, a1 = a2 → (complementary a1) = (complementary a2)) ∧
  (¬ (∀ l1 l2 : Line, are_parallel l1 l2 → ∀ i1 i2 : Angle, interior_same_side l1 l2 i1 i2 → i1 = i2)) ∧
  (∀ p : Point, ∀ l : Line, non_mem p l → ∃! l' : Line, perpendicular l l' ∧ mem p l') ∧
  (∀ p : Point, ∀ l : Line, non_mem p l → ∃ d : ℝ, distance p l = d) ∧
  (∀ a1 a2 : Angle, vertical a1 a2 → ∃ l : Line, angle_bisector a1 l ∧ angle_bisector a2 l) :=
sorry

end evaluate_statements_l158_158502


namespace power_function_inequality_l158_158603

-- Define the conditions
def conditions (m : ℕ) (a : ℝ) : Prop :=
  0 < m ∧ (3 * m - 9 : ℤ) % 2 = 0 ∧ (m=1 ∨ m=2) ∧
  (∀ x > 0, x ^ (3 * m - 9) > 0) ∧
  (∀ x y, 0 < x ∧ x < y → (x ^ (3 * m - 9) > y ^ (3 * m - 9)))

-- Define the theorem to prove
theorem power_function_inequality (m : ℕ) (a : ℝ) (h₁ : conditions m a) :
  (a+1)^(m/3) < (3-2a)^(m/3) → a < 2/3 :=
sorry

end power_function_inequality_l158_158603


namespace simplified_expression_l158_158731

variable (m : ℝ) (h : m = Real.sqrt 3)

theorem simplified_expression : (m - (m + 9) / (m + 1)) / ((m^2 + 3 * m) / (m + 1)) = 1 - Real.sqrt 3 :=
by
  rw [h]
  sorry

end simplified_expression_l158_158731


namespace tomas_ate_1_5_pounds_of_chocolate_fudge_l158_158038

theorem tomas_ate_1_5_pounds_of_chocolate_fudge
  (katya_fudge_pounds : ℝ)
  (boris_fudge_pounds : ℝ)
  (total_fudge_ounces : ℝ)
  (katya_fudge_half_pound : katya_fudge_pounds = 0.5)
  (boris_fudge_two_pounds : boris_fudge_pounds = 2)
  (total_fudge_sixty_four_ounces : total_fudge_ounces = 64) :
  let tomas_fudge_ounces := total_fudge_ounces - (katya_fudge_pounds * 16 + boris_fudge_pounds * 16) in
  let tomas_fudge_pounds := tomas_fudge_ounces / 16 in
  tomas_fudge_pounds = 1.5 := by
  sorry

end tomas_ate_1_5_pounds_of_chocolate_fudge_l158_158038


namespace sum_first_21_terms_l158_158650

variable {a : ℕ → ℝ} -- Define the arithmetic sequence

def a_11_eq_20 : a 11 = 20 := 
sorry

theorem sum_first_21_terms (a_11_eq_20 : a 11 = 20) : (∑ i in Finset.range 21, a (i + 1)) = 420 :=
  sorry

end sum_first_21_terms_l158_158650


namespace optionB_is_opposites_l158_158054

-- Define the pairs of numbers
def optionA_pair : ℕ × ℕ := (3, abs (-3))
def optionB_pair : ℤ × ℤ := (-abs (-3), -(-3))
def optionC_pair : ℤ × ℚ := (-3, -1/3)
def optionD_pair : ℤ × ℚ := (-3, 1/3)

-- Define what it means for two numbers to be opposite
def are_opposites (a b : ℤ) : Prop := a = -b

-- Prove that the pairs in optionB are opposite
theorem optionB_is_opposites : are_opposites (optionB_pair.1) (optionB_pair.2) :=
by
  unfold optionB_pair are_opposites
  simp
  sorry

end optionB_is_opposites_l158_158054


namespace two_hundredth_digit_div_five_thirteen_l158_158423

theorem two_hundredth_digit_div_five_thirteen : 
  (let n := 200 
       d := 13 
       repeating_cycle := "384615".to_list -- Convert repeating cycle to list of digits
       cycle_length := repeating_cycle.length
       division := n / cycle_length
       remainder := n % cycle_length in
   repeating_cycle.nth remainder = some '8') :=
by sorry

end two_hundredth_digit_div_five_thirteen_l158_158423


namespace two_digit_factors_of_2_pow_24_minus_1_l158_158227

-- Given condition
def two_digit_factors (n : ℤ) : List ℤ :=
  (List.range' 10 90).filter (λ x, x ∣ n)

-- Prove the main statement
theorem two_digit_factors_of_2_pow_24_minus_1 : 
  two_digit_factors (2^24 - 1) = 12 :=
by
  sorry

end two_digit_factors_of_2_pow_24_minus_1_l158_158227


namespace probability_blue_is_4_over_13_l158_158826

def num_red : ℕ := 5
def num_green : ℕ := 6
def num_yellow : ℕ := 7
def num_blue : ℕ := 8
def total_jelly_beans : ℕ := num_red + num_green + num_yellow + num_blue

def probability_blue : ℚ := num_blue / total_jelly_beans

theorem probability_blue_is_4_over_13
  (h_num_red : num_red = 5)
  (h_num_green : num_green = 6)
  (h_num_yellow : num_yellow = 7)
  (h_num_blue : num_blue = 8) :
  probability_blue = 4 / 13 :=
by
  sorry

end probability_blue_is_4_over_13_l158_158826


namespace arccos_of_one_over_sqrt_two_l158_158887

theorem arccos_of_one_over_sqrt_two :
  Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 := 
sorry

end arccos_of_one_over_sqrt_two_l158_158887


namespace a2023_le_100000_l158_158783

/-- 
  A sequence \{a_n\} is defined with:
  - a₀ = 0
  - a₁ = 1
  - for n > 1, aₙ is the smallest natural number greater than aₙ₋₁ such that
    there are no three numbers among a₀, a₁, ..., aₙ forming a triplet.
  Prove that a₂₀₂₃ ≤ 100000.
-/
def sequence (n : Nat) : Nat := sorry

theorem a2023_le_100000 : sequence 2023 ≤ 100000 :=
sorry

end a2023_le_100000_l158_158783


namespace arccos_one_over_sqrt_two_l158_158974

theorem arccos_one_over_sqrt_two :
  Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_l158_158974


namespace number_of_points_P_l158_158349

-- Define the ellipse and line equations and their intersection points
def ellipse (x y : ℝ) : Prop := (x^2 / 16) + (y^2 / 9) = 1
def line (x y : ℝ) : Prop := (x / 4) + (y / 3) = 1

-- Define the number of points P such that the area of the triangle PAB = 3
theorem number_of_points_P (x y: ℝ) (P: ℝ × ℝ) :
  ellipse P.1 P.2 →
  let A := exists (a b: ℝ), ellipse a b ∧ line a b,
      B := exists (a b: ℝ), ellipse a b ∧ line a b
  ∃ P1 P2: ℝ × ℝ, P1 ≠ P2 ∧ 
  ∃ (a1 a2 : ℝ) (P1: ℝ × ℝ) (P2: ℝ × ℝ), ellipse P1.1 P1.2 ∧ ellipse P2.1 P2.2 ∧
  let triangle_area (x1 y1 x2 y2 x3 y3: ℝ) : ℝ := 
    (1 / 2) * abs ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)),
      area_P_AB := triangle_area A.1 A.2 B.1 B.2 P.1 P.2 in 
  area_P_AB = 3 :=
begin
  sorry
end

end number_of_points_P_l158_158349


namespace card_arrangements_valid_removal_l158_158727

open Finset

theorem card_arrangements_valid_removal :
  let cards := {1, 2, 3, 4, 5, 6, 7}
  ∃ n : ℕ, 
  (n = 7 ∧ 
  (∀ L : list ℕ, L ∈ (cards.powerset.filter (λ s, s.card = 6)) ∧ (sorted L ∨ sorted (L.reverse)) → n = 74 )) := sorry

end card_arrangements_valid_removal_l158_158727


namespace probability_of_all_co_captains_l158_158036

noncomputable def probability_all_co_captains := 
  have prob_team_6 := 1 / 20
  have prob_team_9 := 1 / 84
  have prob_team_10 := 1 / 120
  have total_prob := (1/3) * (prob_team_6 + prob_team_9 + prob_team_10)
  total_prob = 59 / 2520

theorem probability_of_all_co_captains :
  probability_all_co_captains = 59 / 2520 := 
by
  sorry

end probability_of_all_co_captains_l158_158036


namespace partition_ladies_club_l158_158405

-- Define the conditions given in the problem
def members : ℕ := 100
def board_members : ℕ := 50
def tea_partners : (ℕ → ℕ) := λ _ => 56

-- Predicate for having tea among the board members
def board_all_tea_with_each_other (i j : ℕ) : Prop := i < board_members ∧ j < board_members ∧ i ≠ j

-- Theorem statement proving the partitioning of the club
theorem partition_ladies_club :
  members = 100 →
  (∀ i, tea_partners i = 56) →
  (∀ i, i < board_members → ∀ j, j < board_members → i ≠ j → board_all_tea_with_each_other i j) →
  ∃ A B : finset ℕ, A.card = board_members ∧ B.card = (members - board_members) ∧
  (∀ a1 a2 ∈ A, a1 ≠ a2 → board_all_tea_with_each_other a1 a2) ∧
  (∀ b1 b2 ∈ B, b1 ≠ b2 → ¬board_all_tea_with_each_other b1 b2) :=
begin
  intros h_members h_tea_partners h_board_all_tea,
  -- sorry is used to skip the proof
  sorry,
end

end partition_ladies_club_l158_158405


namespace find_length_l158_158635

-- Define the perimeter and breadth as constants
def P : ℕ := 950
def B : ℕ := 100

-- State the theorem
theorem find_length (L : ℕ) (H : 2 * (L + B) = P) : L = 375 :=
by sorry

end find_length_l158_158635


namespace find_s_plus_u_l158_158453

variables (p r s u : ℂ) (q t : ℤ)

-- Define conditions
def q_condition : q = 4 := sorry
def t_condition : t = -p.re - r.re := sorry
def sum_condition : (p + 4 * Complex.I) + (r + s * Complex.I) + (Complex.mk (-p.re - r.re) (u)) = 3 * Complex.I := sorry

-- The main theorem to be proved
theorem find_s_plus_u (h1 : q_condition) (h2 : t_condition) (h3 : sum_condition) : s + u = -1 := sorry

end find_s_plus_u_l158_158453


namespace integral_sqrt_minus_x_l158_158136

theorem integral_sqrt_minus_x :
  ∫ (x : ℝ) in 0..1, (√(1-(x-1)^2) - x) = (Real.pi / 4 - 1 / 2) :=
by
  sorry

end integral_sqrt_minus_x_l158_158136


namespace x1_add_x2_eq_2016_l158_158585

noncomputable def x1_root (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : ℝ :=
  Classical.choose (Exists.some_spec (exists_root (λ x, log a x + x - 2016) (by sorry)))

noncomputable def x2_root (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : ℝ :=
  Classical.choose (Exists.some_spec (exists_root (λ x, a ^ x + x - 2016) (by sorry)))

theorem x1_add_x2_eq_2016 (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (x1_root a h1 h2) + (x2_root a h1 h2) = 2016 :=
by sorry

end x1_add_x2_eq_2016_l158_158585


namespace arccos_one_over_sqrt_two_eq_pi_over_four_l158_158895

theorem arccos_one_over_sqrt_two_eq_pi_over_four : 
  Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_eq_pi_over_four_l158_158895


namespace probability_no_two_adjacent_stand_l158_158379

theorem probability_no_two_adjacent_stand (n : ℕ) (h_n : n = 10) : 
  let total_outcomes := 2 ^ n in
  let favorable_outcomes := 123 in
  (favorable_outcomes : ℚ) / total_outcomes = 123 / 1024 :=
by {
  have h1 : total_outcomes = 1024 := by 
    rw [h_n],
    norm_num,
    sorry,
  have h2 : (favorable_outcomes : ℚ) = 123 := by 
    sorry,
  have h3 : total_outcomes = 2 ^ 10 := by 
    rw [h_n],
    norm_num,
  rw [h2, h1, ←div_eq_div_iff],
  simp,
  norm_num,
  sorry
}

end probability_no_two_adjacent_stand_l158_158379


namespace number_of_correct_conditions_l158_158860

open Set

def condition_1 : Prop := {0} ∈ ({0, 1, 2} : Set (Set ℕ))
def condition_2 : Prop := {0, 1, 2} ⊆ ({2, 1, 0} : Set ℕ)
def condition_3 : Prop := ∅ ⊆ ({0, 1, 2} : Set ℕ)
def condition_4 : Prop := ∅ = ({0} : Set ℕ)
def condition_5 : Prop := ({0, 1} : Set ℕ) = ({(0, 1)} : Set (ℕ × ℕ))
def condition_6 : Prop := (0 : ℕ) = ({0} : Set ℕ)

theorem number_of_correct_conditions :
  (if condition_1 then 1 else 0) +
  (if condition_2 then 1 else 0) +
  (if condition_3 then 1 else 0) +
  (if condition_4 then 1 else 0) +
  (if condition_5 then 1 else 0) +
  (if condition_6 then 1 else 0) = 2 :=
sorry

end number_of_correct_conditions_l158_158860


namespace g_has_two_zeros_l158_158593

def f (x : ℝ) : ℝ :=
  if x > 0 then x - 2 else -x^2 + (1:ℝ) / 2 * x + 1

def g (x : ℝ) : ℝ := f x + x

theorem g_has_two_zeros :
  f 0 = 1 →
  f 0 + 2 * f (-1) = 0 →
  ∃ z1 z2 : ℝ, z1 ≠ z2 ∧ g z1 = 0 ∧ g z2 = 0 :=
by
  intros
  sorry

end g_has_two_zeros_l158_158593


namespace liangliang_speed_l158_158756

theorem liangliang_speed (d_initial : ℝ) (t : ℝ) (d_final : ℝ) (v_mingming : ℝ) (v_liangliang : ℝ) :
  d_initial = 3000 →
  t = 20 →
  d_final = 2900 →
  v_mingming = 80 →
  (v_liangliang = 85 ∨ v_liangliang = 75) :=
by
  sorry

end liangliang_speed_l158_158756


namespace min_value_trig_l158_158210

theorem min_value_trig (x y z : ℝ) (hx : 0 ≤ x) (hx2 : x ≤ π/2)
                       (hy : 0 ≤ y) (hy2 : y ≤ π/2)
                       (hz : 0 ≤ z) (hz2 : z ≤ π/2) :
  ∃ A, A = cos (x - y) + cos (y - z) + cos (z - x) ∧ A ≥ 1 :=
sorry

end min_value_trig_l158_158210


namespace statement_A_statement_B_statement_C_statement_D_l158_158587

variable (a b : ℝ)

-- Given conditions
axiom positive_a : 0 < a
axiom positive_b : 0 < b
axiom condition : a + 2 * b = 2 * a * b

-- Prove the statements
theorem statement_A : a + 2 * b ≥ 4 := sorry
theorem statement_B : ¬ (a + b ≥ 4) := sorry
theorem statement_C : ¬ (a * b ≤ 2) := sorry
theorem statement_D : a^2 + 4 * b^2 ≥ 8 := sorry

end statement_A_statement_B_statement_C_statement_D_l158_158587


namespace arccos_one_over_sqrt_two_l158_158944

theorem arccos_one_over_sqrt_two : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_l158_158944


namespace cos_theta_simplified_l158_158581

theorem cos_theta_simplified (θ : ℝ) (h : sin θ / sin (θ / 2) = 5 / 3) : cos θ = 7 / 18 := 
sorry

end cos_theta_simplified_l158_158581


namespace flower_percentages_l158_158644

-- Definitions for the given conditions
def total_flowers : ℕ := 100
def red_flowers := total_flowers * (1 - 0.6)
def white_flowers := total_flowers * 0.6

def white_roses := (3 / 5) * white_flowers
def white_tulips := white_flowers - white_roses

def red_tulips := red_flowers / 2
def red_roses := red_flowers - red_tulips

-- Definitions for the total count and percentage of tulips and roses
def total_tulips := red_tulips + white_tulips
def total_roses := red_roses + white_roses

def tulip_percentage := (total_tulips / total_flowers) * 100
def rose_percentage := (total_roses / total_flowers) * 100

-- The theorem to prove
theorem flower_percentages :
  tulip_percentage = 44 ∧ rose_percentage = 56 :=
by
  sorry

end flower_percentages_l158_158644


namespace multiply_divide_repeating_decimals_l158_158792

theorem multiply_divide_repeating_decimals :
  (8 * (1 / 3) / 1) = 8 / 3 := by
  sorry

end multiply_divide_repeating_decimals_l158_158792


namespace digit_after_decimal_l158_158419

theorem digit_after_decimal (n : ℕ) : (n = 123) → (123 % 12 ≠ 0) → (123 % 12 = 3) → (∃ d : ℕ, d = 1 ∧ (43 / 740 : ℚ)^123 = 0 + d / 10^(123)) := 
by
    intros h₁ h₂ h₃
    sorry

end digit_after_decimal_l158_158419


namespace arccos_sqrt_half_l158_158967

theorem arccos_sqrt_half : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 := by
  sorry

end arccos_sqrt_half_l158_158967


namespace overtaking_points_l158_158159

-- defining constants for the given conditions
def L : ℕ := 55
def x : ℝ := 1 -- unit time constant, can be defined as literal 1 for simplicity
def vp : ℝ := 100 * x
def vc : ℝ := 155 * x

-- main statement: the pedestrian is overtaken at exactly 11 different points
theorem overtaking_points : 
  (∀ (L vp vc : ℕ) (x : ℝ), L = 55 ∧ vp = 100 * x ∧ vc = 155 * x → ∃ n : ℕ, n = 11) :=
by
  intro L vp vc x h
  cases h with hL hV
  cases hV with hVp hVc
  use 11
  sorry

end overtaking_points_l158_158159


namespace pencils_needed_l158_158648

theorem pencils_needed (pencilsA : ℕ) (pencilsB : ℕ) (classroomsA : ℕ) (classroomsB : ℕ) (total_shortage : ℕ)
  (hA : pencilsA = 480)
  (hB : pencilsB = 735)
  (hClassA : classroomsA = 6)
  (hClassB : classroomsB = 9)
  (hShortage : total_shortage = 85) 
  : 90 = 6 + 5 * ((total_shortage / (classroomsA + classroomsB)) + 1) * classroomsB :=
by {
  sorry
}

end pencils_needed_l158_158648


namespace find_k_in_quadratic_form_l158_158637

theorem find_k_in_quadratic_form (x : ℝ) : 
  ∃ k : ℝ, ∀ a h, a = 1 ∧ h = -3 → (x^2 + 6*x = a*(x - h)^2 + k) → k = -9 :=
begin
  -- The proof will go here.
  sorry,
end

end find_k_in_quadratic_form_l158_158637


namespace Jerry_walked_9_miles_l158_158668

theorem Jerry_walked_9_miles (x : ℕ) (h : 2 * x = 18) : x = 9 := 
by
  sorry

end Jerry_walked_9_miles_l158_158668


namespace rhombus_area_l158_158062

noncomputable def diagonal_length (p1 p2 : (ℝ × ℝ)) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem rhombus_area : 
  let A := (0, 5.5) 
  let B := (8, 0)
  let C := (0, -5.5)
  let D := (-8, 0)
  let d1 := diagonal_length A C -- (0,5.5) and (0,-5.5)
  let d2 := diagonal_length B D -- (8,0) and (-8,0)
  (d1 * d2) / 2 = 88 := 
by
  sorry

end rhombus_area_l158_158062


namespace probability_sum_even_l158_158275

-- Let's define the set of the first ten prime numbers.
def first_ten_primes : Finset ℕ := Finset.of_list [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define the proposition to prove:
theorem probability_sum_even :
  let n := first_ten_primes.card in 
  let total_pairs := Finset.card (Finset.powersetLen 2 first_ten_primes) in 
  let even_pairs := Finset.card (Finset.filter (λ s, s.sum % 2 = 0) (Finset.powersetLen 2 first_ten_primes)) in 
  (total_pairs = 45 ∧ even_pairs = 36) →
  (even_pairs / total_pairs = 4 / 5) :=
by
  sorry

end probability_sum_even_l158_158275


namespace probability_both_quit_same_tribe_l158_158025

noncomputable def total_contestants : ℕ := 18
noncomputable def tribe_size : ℕ := 9
noncomputable def immune_contestant : ℕ := 1
noncomputable def quitters : ℕ := 2

theorem probability_both_quit_same_tribe (total_contestants = 18) (tribe_size = 9) 
  (immune_contestant = 1) (quitters = 2) :
  let remaining_contestants := total_contestants - immune_contestant in
  let total_ways_to_pick_quitters := nat.choose remaining_contestants quitters in
  let ways_to_pick_quitters_from_one_tribe := nat.choose tribe_size 2 in
  let total_ways_to_pick_quitters_from_same_tribe := 2 * ways_to_pick_quitters_from_one_tribe in
  (total_ways_to_pick_quitters_from_same_tribe / total_ways_to_pick_quitters : ℚ) = 9 / 17 :=
by
  sorry

end probability_both_quit_same_tribe_l158_158025


namespace probability_of_even_sum_is_four_fifths_l158_158266

-- Define the first ten prime numbers
def first_ten_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- The function to calculate the probability that the sum of two distinct primes is even
def probability_even_sum (primes : List ℕ) : ℚ :=
  let pairs := primes.product primes
  let distinct_pairs := pairs.filter (λ p, p.1 ≠ p.2)
  let even_sum_pairs := distinct_pairs.filter (λ p, (p.1 + p.2) % 2 = 0)
  (even_sum_pairs.length : ℚ) / (distinct_pairs.length : ℚ)

-- Prove that the probability is 4/5
theorem probability_of_even_sum_is_four_fifths :
  probability_even_sum first_ten_primes = 4 / 5 := sorry

end probability_of_even_sum_is_four_fifths_l158_158266


namespace lines_parallel_l158_158198

-- Define the lines l1 and l2
def l1 (x y: ℝ) : Prop := 2 * x - 4 * y + 7 = 0
def l2 (x y: ℝ) : Prop := x - 2 * y + 5 = 0

-- Define the property to prove
theorem lines_parallel : 
  (∀ x y : ℝ, l1 x y ↔ 2 * x - 4 * y + 7 = 0) ∧ 
  (∀ x y : ℝ, l2 x y ↔ x - 2 * y + 5 = 0) → 
  (∀ c₁ c₂ m₁ m₂ y₁ y₂: ℝ, 
   (2 * c₁ - 4 * y₁ + 7 = 0) → 
   (c₂ - 2 * y₂ + 5 = 0) → 
   c₁ / 2 = c₂ / 1 / 2) → 
  (c₁ / 2 ≠ c₂ / 1 / 2) →
  Parallel l1 l2 :=
sorry

end lines_parallel_l158_158198


namespace number_of_nonempty_proper_subsets_of_B_l158_158622

def A := {x : ℕ | -1 < x ∧ x ≤ 2}
def B := {x : ℕ | ∃ a b ∈ A, x = a * b}

theorem number_of_nonempty_proper_subsets_of_B : (2 ^ (Finset.card B) - 2 = 14) :=
by
  -- Proof steps would go here
  sorry

end number_of_nonempty_proper_subsets_of_B_l158_158622


namespace tangent_chord_rectangle_perimeter_l158_158830

theorem tangent_chord_rectangle_perimeter (R x k : ℝ)
    (h1 : R > 0)
    (h2 : x ≥ 0)
    (h3 : 2 * R * (sqrt(5) + 1) = k):
  k := sorry

end tangent_chord_rectangle_perimeter_l158_158830


namespace equation_1_solutions_equation_2_solutions_l158_158738

-- Equation 1: Proving solutions for (x+8)(x+1) = -12
theorem equation_1_solutions (x : ℝ) :
  (x + 8) * (x + 1) = -12 ↔ x = -4 ∨ x = -5 :=
sorry

-- Equation 2: Proving solutions for (2x-3)^2 = 5(2x-3)
theorem equation_2_solutions (x : ℝ) :
  (2 * x - 3) ^ 2 = 5 * (2 * x - 3) ↔ x = 3 / 2 ∨ x = 4 :=
sorry

end equation_1_solutions_equation_2_solutions_l158_158738


namespace percent_unionized_men_is_70_l158_158642

-- Definitions based on the problem conditions
def total_employees : ℕ := 100
def percent_men : ℤ := 52
def percent_unionized : ℤ := 60
def percent_non_union_women : ℤ := 75

-- Derived definitions
def num_men : ℕ := (percent_men * total_employees) / 100
def num_unionized : ℕ := (percent_unionized * total_employees) / 100
def num_non_union : ℕ := total_employees - num_unionized
def num_non_union_men : ℕ := ((100 - percent_non_union_women) * num_non_union) / 100
def num_union_men : ℕ := num_men - num_non_union_men
def percent_union_men : ℤ := (num_union_men * 100) / num_unionized

-- The proof statement
theorem percent_unionized_men_is_70 : percent_union_men = 70 := by
  sorry

end percent_unionized_men_is_70_l158_158642


namespace circles_common_intersection_l158_158214

noncomputable def points_exist :
  ∃ (A B C D : Point), 
    (¬ collinear A B C) ∧ 
    (¬ collinear A B D) ∧ 
    (¬ collinear A C D) ∧ 
    (¬ collinear B C D) :=
sorry

noncomputable def intersections_exist (A B C D : Point) (h₁ : ¬ collinear A B C) (h₂ : ¬ collinear A B D) (h₃ : ¬ collinear A C D) (h₄ : ¬ collinear B C D):
  ∃ (E F : Point), 
    (intersect (Line A B) (Line C D) = E) ∧ 
    (intersect (Line B C) (Line D A) = F) := 
sorry

theorem circles_common_intersection (A B C D E F : Point)
  (h₁ : ¬ collinear A B C)
  (h₂ : ¬ collinear A B D)
  (h₃ : ¬ collinear A C D)
  (h₄ : ¬ collinear B C D)
  (hE : intersect (Line A B) (Line C D) = E)
  (hF : intersect (Line B C) (Line D A) = F) :
  (∃ P : Point, on_circle P (CircleDiameter A C) ∧ on_circle P (CircleDiameter B D) ∧ on_circle P (CircleDiameter E F)) ∨
  (∀ P Q : Point, (on_circle P (CircleDiameter A C) ∧ on_circle Q (CircleDiameter B D) → P ≠ Q) ∧
  (on_circle P (CircleDiameter B D) ∧ on_circle Q (CircleDiameter E F) → P ≠ Q) ∧
  (on_circle P (CircleDiameter A C) ∧ on_circle Q (CircleDiameter E F) → P ≠ Q)) :=
sorry

end circles_common_intersection_l158_158214


namespace well_depth_is_784_l158_158093

noncomputable def depth_of_well (t_total : ℝ) : ℝ :=
  let v_sound := 1120
  let h : ℝ → ℝ := λ d, d / v_sound
  let g : ℝ → ℝ := λ t, 16 * t^2
  let t_fall_eq := λ d, ( sqrt d ) / 4
  let t_sound_eq := λ d, d / v_sound
  let equation := λ d, t_fall_eq d + t_sound_eq d = t_total

theorem well_depth_is_784 (h1 : 7.7 = t_total) : depth_of_well 7.7 = 784 :=
by
  sorry

end well_depth_is_784_l158_158093


namespace number_of_valid_arrangements_l158_158724

theorem number_of_valid_arrangements : 
  (∃ (arrangements : Finset (List ℕ)), 
   arrangements.card = 4 ∧
   ∀ (L : List ℕ), L ∈ arrangements → 
     (∃ (k : ℕ), k ∈ List.range 7 ∧ 
      (List.remove_nth L k).sorted (≤) ∨ (List.remove_nth L k).sorted (≥))) := 
sorry

end number_of_valid_arrangements_l158_158724


namespace cyclic_hexagon_proportionality_l158_158753

variables {A B C D E F O P : Type}
variables [has_dist A B C D E F O P] -- Assuming 'has_dist' handles the distance and necessary geometric properties

noncomputable def cyclic_convex_hexagon (AB CD EF : Type) 
  (cyclic: is_cyclic [A, B, C, D, E, F])
  (convex: is_convex [A, B, C, D, E, F])
  (concurrent: is_concurrent [AD, BE, CF]) 
  (dist_eq: has_dist A B = has_dist C D = has_dist E F)
  (intersection: intersect (AD, CE) = P) : Prop := 
    has_dist C P / has_dist P E = (has_dist A C / has_dist C E) ^ 2

theorem cyclic_hexagon_proportionality 
  (cyclic : is_cyclic [A, B, C, D, E, F])
  (convex : is_convex [A, B, C, D, E, F])
  (side_eq : has_dist A B = has_dist C D ∧ has_dist C D = has_dist E F)
  (concurrent : is_concurrent [AD, BE, CF])
  (P_intersection : intersect (AD, CE) = P) : 
  ∃ (P : Type), has_dist C P / has_dist P E = (has_dist A C / has_dist C E) ^ 2 :=
by
  sorry

end cyclic_hexagon_proportionality_l158_158753


namespace outfit_combinations_l158_158056

theorem outfit_combinations (short_sleeve_shirts long_sleeve_shirts jeans trousers : ℕ) :
  short_sleeve_shirts = 5 → 
  long_sleeve_shirts = 3 → 
  jeans = 6 → 
  trousers = 2 →
  let total_shirts := short_sleeve_shirts + long_sleeve_shirts in
  let total_pants := jeans + trousers in
  total_shirts * total_pants = 64 :=
by
  intros h1 h2 h3 h4
  simp [h1, h2, h3, h4]
  sorry

end outfit_combinations_l158_158056


namespace log_equation_sqrt_algebra_l158_158518

theorem log_equation : log (2 * real.sqrt 2) / log (real.sqrt 2) + (log 3 / log 2) * (log 4 / log 3) = 5 :=
by {
  sorry,
}

theorem sqrt_algebra (a : ℝ) (ha : a < 0) : real.sqrt (a ^ 2) * 3 * (a ^ 3) * (a ^ (-1)) = -a :=
by {
  sorry,
}

end log_equation_sqrt_algebra_l158_158518


namespace domain_ln_x_minus_1_l158_158011

def domain_of_log_function (x : ℝ) : Prop := x > 1

theorem domain_ln_x_minus_1 (x : ℝ) : domain_of_log_function x ↔ x > 1 :=
by {
  sorry
}

end domain_ln_x_minus_1_l158_158011


namespace elliptical_billiard_distance_l158_158504

theorem elliptical_billiard_distance 
  (A B : ℝ × ℝ) 
  (major_axis focal_distance : ℝ)
  (h1 : 0 < major_axis)
  (h2 : 0 < focal_distance)
  (h3 : major_axis = 2)
  (h4 : focal_distance = 1)
  (optical_property : ∀ x, x ≠ A → x ≠ B → ray_from_focus_passes_through_other_focus (A, B) x) :
  ∃ d, d = 4 ∨ d = 3 ∨ d = 1 :=
by
  sorry

end elliptical_billiard_distance_l158_158504


namespace product_modulo_23_l158_158790

theorem product_modulo_23 :
  (3001 * 3002 * 3003 * 3004 * 3005) % 23 = 0 :=
by {
  have h1 : 3001 % 23 = 19 := rfl,
  have h2 : 3002 % 23 = 20 := rfl,
  have h3 : 3003 % 23 = 21 := rfl,
  have h4 : 3004 % 23 = 22 := rfl,
  have h5 : 3005 % 23 = 0 := rfl,
  sorry
}

end product_modulo_23_l158_158790


namespace exists_mk_Mk_sum_reciprocals_difference_bounds_l158_158179

variable (n : ℕ) (hn : n ≥ 3)
variable (C_n : set (Fin n → ℝ)) 
variable (σ_k : ∀ k, C_n → set ℝ)

-- Define m_k and M_k based on the conditions
def m_k (k : ℕ) : ℝ := max (λ a, Inf (σ_k k a))
def M_k (k : ℕ) : ℝ := min (λ a, Sup (σ_k k a))

-- Proof of existence of m_k and M_k for given k
theorem exists_mk_Mk (k : ℕ) (hk : 1 ≤ k ∧ k ≤ n - 1) :
  ∃ (m_k : ℝ) (M_k : ℝ), 
    m_k = max (λ a, Inf (σ_k k a)) ∧
    M_k = min (λ a, Sup (σ_k k a)) :=
sorry

-- Main theorem about the sum of reciprocals differences
theorem sum_reciprocals_difference_bounds :
  1 ≤ ∑ k in finset.range (n-1), (1 / M_k k - 1 / m_k k) ∧
  ∑ k in finset.range (n-1), (1 / M_k k - 1 / m_k k) ≤ n - 2 :=
sorry

end exists_mk_Mk_sum_reciprocals_difference_bounds_l158_158179


namespace still_water_speed_l158_158488

theorem still_water_speed (upstream_speed downstream_speed : ℝ) 
(H_upstream : upstream_speed = 25) (H_downstream : downstream_speed = 55) : 
(upstream_speed + downstream_speed) / 2 = 40 := 
by
  rw [H_upstream, H_downstream]
  norm_num
  sorry

end still_water_speed_l158_158488


namespace mass_of_measuring_stick_l158_158470

variables {g : ℝ} [fact (0 < g)]

noncomputable def torque_due_to_rock (g : ℝ) : ℝ :=
  2 * g * 0.80

noncomputable def torque_due_to_stick (m : ℝ) (g : ℝ) : ℝ :=
  m * g * 0.30

theorem mass_of_measuring_stick (g : ℝ) [fact (0 < g)] : 
  ∃ (m : ℝ), torque_due_to_rock g = torque_due_to_stick m g ∧ m = 16 / 3 :=
by
  use 16 / 3
  split
  · simp [torque_due_to_rock, torque_due_to_stick]
  · refl

end mass_of_measuring_stick_l158_158470


namespace number_of_TVs_in_shop_c_l158_158037

theorem number_of_TVs_in_shop_c 
  (a b d e : ℕ) 
  (avg : ℕ) 
  (num_shops : ℕ) 
  (total_TVs_in_other_shops : ℕ) 
  (total_TVs : ℕ) 
  (sum_shops : a + b + d + e = total_TVs_in_other_shops) 
  (avg_sets : avg = total_TVs / num_shops) 
  (number_shops : num_shops = 5)
  (avg_value : avg = 48)
  (T_a : a = 20) 
  (T_b : b = 30) 
  (T_d : d = 80) 
  (T_e : e = 50) 
  : (total_TVs - total_TVs_in_other_shops = 60) := 
by 
  sorry

end number_of_TVs_in_shop_c_l158_158037


namespace arccos_proof_l158_158959

noncomputable def arccos_identity : Prop := 
  ∃ x : ℝ, x = 1 / Real.sqrt 2 ∧ Real.arccos x = Real.pi / 4

theorem arccos_proof : arccos_identity :=
by
  sorry

end arccos_proof_l158_158959


namespace problem_statement_l158_158065

noncomputable def complex_mul : Prop :=
  (3 - 2 * complex.i) * (1 + 2 * complex.i) = 7 + 4 * complex.i

theorem problem_statement : complex_mul := 
by
  sorry

end problem_statement_l158_158065


namespace arccos_proof_l158_158962

noncomputable def arccos_identity : Prop := 
  ∃ x : ℝ, x = 1 / Real.sqrt 2 ∧ Real.arccos x = Real.pi / 4

theorem arccos_proof : arccos_identity :=
by
  sorry

end arccos_proof_l158_158962


namespace bug_paths_integer_coordinates_count_l158_158075

theorem bug_paths_integer_coordinates_count :
  let A : ℤ × ℤ := (-4, 3)
  let B : ℤ × ℤ := (4, -3)
  let manhattan_distance (p q : ℤ × ℤ) := |p.1 - q.1| + |p.2 - q.2|
  let valid_path_length := 24
  let is_valid_point (p : ℤ × ℤ) := |p.1 + 4| + |p.1 - 4| + |p.2 - 3| + |p.2 + 3| ≤ valid_path_length

  -- Given the conditions specified, the total number of valid points is 153
  (finset.card (finset.filter is_valid_point (finset.univ : finset (ℤ × ℤ)))) = 153 :=
begin
  let A := (-4, 3),
  let B := (4, -3),
  let manhattan_distance := λ p q : ℤ × ℤ, |p.1 - q.1| + |p.2 - q.2|,
  let valid_path_length := 24,
  let is_valid_point := λ p : ℤ × ℤ, |p.1 + 4| + |p.1 - 4| + |p.2 - 3| + |p.2 + 3| ≤ valid_path_length,

  have h1 : (manhattan_distance A B = 14), from sorry,
  have h2 : (finset.card (finset.filter is_valid_point (finset.univ : finset (ℤ × ℤ)))) = 153, from sorry,

  exact h2,
end

end bug_paths_integer_coordinates_count_l158_158075


namespace find_lambda_l158_158219

open Real

def vec_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 + v2.1, v1.2 + v2.2)

def vec_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 - v2.1, v1.2 - v2.2)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem find_lambda (λ : ℝ) 
  (m : ℝ × ℝ := (λ + 1, 1))
  (n : ℝ × ℝ := (λ + 2, 2)) 
  (h : dot_product (vec_add m n) (vec_sub m n) = 0) :
  λ = -3 :=
by
  sorry

end find_lambda_l158_158219


namespace g_range_l158_158996

noncomputable def g (x : ℝ) : ℝ := ⌊2 * x⌋ - x - 1

theorem g_range : set.Ioo (-2 : ℝ) 0 ∪ {0} = {y : ℝ | ∃ x : ℝ, g x = y} :=
begin
  sorry,
end

end g_range_l158_158996


namespace perpendicular_lines_cond_l158_158817

theorem perpendicular_lines_cond :
  ∀ m : ℝ, (m = 1/2) ↔ ((m+1)(4*m-2) = 0) :=
begin
  sorry
end

end perpendicular_lines_cond_l158_158817


namespace probability_even_sum_l158_158269

-- Definition of the prime numbers set and the selection scenario
def firstTenPrimes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Condition stating that we select two distinct numbers at random from the first ten primes
def randomSelection (s : List ℕ) := {x : ℕ × ℕ // x.1 ≠ x.2 ∧ x.1 ∈ s ∧ x.2 ∈ s}

-- Define the event that their sum is even
def evenSum (p : ℕ × ℕ) := (p.1 + p.2) % 2 = 0

-- Define the probability calculation
def probabilityEvenSum := 
  let totalPairs := (firstTenPrimes.length.choose 2) -- Calculate the number of ways to choose 2 numbers from the 10 primes
  let evenPairs := (randomSelection firstTenPrimes).count (λ p => evenSum p.val)
  evenPairs.toFloat / totalPairs.toFloat

-- Theorem statement that the probability of their sum being even is 1/5
theorem probability_even_sum : probabilityEvenSum = 1/5 := by
  sorry -- The actual proof is to be provided

end probability_even_sum_l158_158269


namespace positive_real_number_solution_l158_158548

theorem positive_real_number_solution (x : ℝ) (h1 : x > 0) (h2 : x ≠ 11) (h3 : (x - 6) / 11 = 6 / (x - 11)) : x = 17 :=
sorry

end positive_real_number_solution_l158_158548


namespace wrapping_paper_dimensions_l158_158091

theorem wrapping_paper_dimensions (w : ℝ) (h l : ℝ) 
  (h_w : h = w) (l_w : l = 2 * w) : 
  wrapping_paper_dimensions = (3 * w) * (4 * w) := by 
  -- Proof goes here
  sorry

end wrapping_paper_dimensions_l158_158091


namespace ratio_of_calls_processed_l158_158829

theorem ratio_of_calls_processed
  (A B : ℕ)
  (C : ℕ)
  (hA : A = 5 * B / 8)
  (hCallsB : real := 8 / 9)
  (hCallsA : real := 1 / 9)
  (hTotalCallsB : real := C * hCallsB)
  (hTotalCallsA : real := C * hCallsA)
  : (B / A) * (hCallsA / hCallsB) = 1 / 5 := by
  sorry

end ratio_of_calls_processed_l158_158829


namespace tan_identity_l158_158580

theorem tan_identity (a b : ℝ) (x : ℝ) (h1 : sin (2 * x) = a) (h2 : cos (2 * x) = b) (h3 : 0 < x ∧ x < π / 4) :
  tan (x + π / 4) = (a - b + 1) / (a + b - 1) :=
by
  sorry

end tan_identity_l158_158580


namespace f_2017_eq_4033_l158_158374

-- Define the function f that satisfies the conditions
variable (f : ℝ → ℝ)
variable f_property : ∀ (a b : ℝ), 3 * f ((a + 2 * b) / 3) = f (a) + 2 * f (b)
variable f_at_1 : f 1 = 1
variable f_at_4 : f 4 = 7

-- Define the theorem to prove f(2017) = 4033
theorem f_2017_eq_4033 : f 2017 = 4033 := sorry

end f_2017_eq_4033_l158_158374


namespace relationship_A_B_l158_158691

variable (x y : ℝ)

noncomputable def A : ℝ := (x + y) / (1 + x + y)

noncomputable def B : ℝ := (x / (1 + x)) + (y / (1 + y))

theorem relationship_A_B (hx : 0 < x) (hy : 0 < y) : A x y < B x y := sorry

end relationship_A_B_l158_158691


namespace overtaking_points_count_l158_158158

def track_length : ℝ := 55
def pedestrian_speed (x : ℝ) : ℝ := 100 * x
def cyclist_speed (x : ℝ) : ℝ := 155 * x

theorem overtaking_points_count (x : ℝ) > 0 : ∃ n : ℕ, n = 11 :=
by
  sorry

end overtaking_points_count_l158_158158


namespace neilInitialGames_l158_158222

-- Hypotheses based on the given conditions
variable (initialHenryGames : ℕ) (givenGamesToNeil : ℕ) (henryGamesAfterGiving : ℕ)
variable (henryHasFourTimesMoreGames : ℕ)

-- Given conditions as Lean variable assignments
def henryInitialGames := initialHenryGames = 33
def gamesGivenToNeil := givenGamesToNeil = 5
def henryGamesPostGiving := henryGamesAfterGiving = 33 - 5
def henryFourTimesMoreGames := henryHasFourTimesMoreGames = 4 * (henryGamesAfterGiving / 4)

-- Final statement to prove using the conditions
theorem neilInitialGames (H1 : henryInitialGames) (H2 : gamesGivenToNeil) (H3 : henryGamesPostGiving) (H4 : henryFourTimesMoreGames) : 
  henryGamesAfterGiving / 4 - 5 = 2 :=
by
  sorry

end neilInitialGames_l158_158222


namespace investments_interest_yielded_l158_158851

def total_investment : ℝ := 15000
def part_one_investment : ℝ := 8200
def rate_one : ℝ := 0.06
def rate_two : ℝ := 0.075

def part_two_investment : ℝ := total_investment - part_one_investment

def interest_one : ℝ := part_one_investment * rate_one * 1
def interest_two : ℝ := part_two_investment * rate_two * 1

def total_interest : ℝ := interest_one + interest_two

theorem investments_interest_yielded : total_interest = 1002 := by
  sorry

end investments_interest_yielded_l158_158851


namespace centroid_has_integer_coordinates_l158_158361

open Finset

noncomputable def centroid (p1 p2 p3 : ℤ × ℤ) : ℚ × ℚ :=
  ((p1.1 + p2.1 + p3.1) / 3, (p1.2 + p2.2 + p3.2) / 3)

theorem centroid_has_integer_coordinates (points : Finset (ℤ × ℤ))
  (h_points : points.card = 13)
  (h_no_three_collinear: ∀ (p1 p2 p3 : ℤ × ℤ), p1 ∈ points → p2 ∈ points → p3 ∈ points → 
    (p1.1 - p2.1) * (p1.2 - p3.2) ≠ (p1.1 - p3.1) * (p1.2 - p2.2)) :
  ∃ (p1 p2 p3 : ℤ × ℤ), p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧ 
    ∃ (c : ℤ × ℤ), centroid p1 p2 p3 = c := 
sorry

end centroid_has_integer_coordinates_l158_158361


namespace probability_even_sum_l158_158248

def firstTenPrimes : Set ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

def isOdd (n : ℕ) : Prop := n % 2 = 1

def distinctPairs (s : Set ℕ) : Set (ℕ × ℕ) :=
  { (x, y) | x ∈ s ∧ y ∈ s ∧ x ≠ y }

def eventEvenSum (s : Set ℕ) : Set (ℕ × ℕ) :=
  { (x, y) | x ∈ s ∧ y ∈ s ∧ x ≠ y ∧ (x + y) % 2 = 0 }

theorem probability_even_sum :
  (Set.card (eventEvenSum firstTenPrimes)).toRat / (Set.card (distinctPairs firstTenPrimes)).toRat = 4 / 5 :=
  sorry

end probability_even_sum_l158_158248


namespace arccos_one_over_sqrt_two_eq_pi_four_l158_158905

theorem arccos_one_over_sqrt_two_eq_pi_four : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 := 
by
  sorry

end arccos_one_over_sqrt_two_eq_pi_four_l158_158905


namespace math_club_team_selection_l158_158703

theorem math_club_team_selection :
  let boys := 9
  let girls := 10
  let experienced_boys := 4
  let total_team := 7
  let required_boys := 4
  let required_girls := 3
  (finset.card (finset.powerset (finset.range experienced_boys)).filter (λ s, s.card = 2) *
  finset.card (finset.powerset (finset.range (boys - experienced_boys))).filter (λ s, s.card = (required_boys - 2)) *
  finset.card (finset.powerset (finset.range girls)).filter (λ s, s.card = required_girls)) = 7200 :=
by
  sorry

end math_club_team_selection_l158_158703


namespace probability_even_sum_l158_158283

def first_ten_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def distinct_pairs (l : List ℕ) : List (ℕ × ℕ) :=
  (l.product l).filter (λ p, p.1 < p.2)

theorem probability_even_sum : 
  (first_ten_primes.length = 10) →
  (∀ a b : ℕ, a ∈ first_ten_primes → b ∈ first_ten_primes → a ≠ b → 
    ((a + b) % 2 = 0 ↔ 2 ∉ [a, b])) →
  (∑ pair in distinct_pairs first_ten_primes, ((pair.1 + pair.2) % 2 = 0).toNat) 
    = 36 → 
  ((∑ pair in distinct_pairs first_ten_primes, (1:ℚ)) ⁻¹ * 
   (∑ pair in distinct_pairs first_ten_primes, ((pair.1 + pair.2) % 2 = 0).toNat))
    = 4 / 5 := by
  sorry

end probability_even_sum_l158_158283


namespace are_opposites_l158_158051
noncomputable theory

def opposite (a b : Int) : Prop :=
  a = -b

theorem are_opposites : 
  opposite (-|(-3 : Int)|) (-( -3)) :=
by
  unfold opposite
  sorry

end are_opposites_l158_158051


namespace shortest_distance_l158_158026

noncomputable theory

open Real

def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 6 * x - 6 * y + 10 = 0
def line_eq (x y : ℝ) : Prop := x + y = 0

theorem shortest_distance :
  ∃ p : ℝ × ℝ,
    circle_eq p.1 p.2 ∧
    ∃ q : ℝ × ℝ,
      line_eq q.1 q.2 ∧
      (sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = sqrt 2) :=
sorry

end shortest_distance_l158_158026


namespace translate_triangle_vertex_l158_158195

theorem translate_triangle_vertex 
    (a b : ℤ) 
    (hA : (-3, a) = (-1, 2) + (-2, a - 2)) 
    (hB : (b, 3) = (1, -1) + (b - 1, 4)) :
    (2 + (-3 - (-1)), 1 + (3 - (-1))) = (0, 5) :=
by 
  -- proof is omitted as instructed
  sorry

end translate_triangle_vertex_l158_158195


namespace ab_gt_c_l158_158336

theorem ab_gt_c {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (h : 1 / a + 4 / b = 1) (hc : c < 9) : a + b > c :=
sorry

end ab_gt_c_l158_158336


namespace smallest_integer_sequence_reaches_l158_158521

def u (n : ℕ) : ℕ :=
  Nat.recOn n (2010^2010) (λ n' uf, if uf % 2 = 1 then uf + 7 else uf / 2)

theorem smallest_integer_sequence_reaches :
  ∃ n : ℕ, u n = 1 :=
sorry

end smallest_integer_sequence_reaches_l158_158521


namespace proof_am_dot_cd_l158_158307

noncomputable theory

open_locale big_operators

def side_length : ℝ := 4

def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (4, 0)
def C : ℝ × ℝ := (2, 2 * real.sqrt 3)

def D : ℝ × ℝ := (4/3, 0)

def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.fst + p2.fst) / 2, (p1.snd + p2.snd) / 2)

def M : ℝ × ℝ := midpoint B C

def vector (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  (p2.fst - p1.fst, p2.snd - p1.snd)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.fst * v2.fst + v1.snd * v2.snd

theorem proof_am_dot_cd :
  dot_product (vector A M) (vector C D) = -8 :=
by
  sorry

end proof_am_dot_cd_l158_158307


namespace train_speed_l158_158096

def train_length : ℝ := 90
def crossing_time_secs : ℝ := 4.499640028797696
def crossing_time_hrs : ℝ := crossing_time_secs / 3600
def distance_km : ℝ := train_length / 1000

theorem train_speed : (distance_km / crossing_time_hrs) = 72 := by
  sorry

end train_speed_l158_158096


namespace number_of_4_letter_words_with_vowel_l158_158608

def is_vowel (c : Char) : Bool :=
c = 'A' ∨ c = 'E'

def count_4letter_words_with_vowels : Nat :=
  let total_words := 5^4
  let words_without_vowels := 3^4
  total_words - words_without_vowels

theorem number_of_4_letter_words_with_vowel :
  count_4letter_words_with_vowels = 544 :=
by
  -- proof goes here
  sorry

end number_of_4_letter_words_with_vowel_l158_158608


namespace sum_modulo_remainder_l158_158045

theorem sum_modulo_remainder :
  (finset.range 156).sum (λ x, x + 1) % 7 = 3 :=
sorry

end sum_modulo_remainder_l158_158045


namespace g_at_5_l158_158686

def g (x : ℝ) : ℝ := 2 * x^4 - 17 * x^3 + 28 * x^2 - 20 * x - 80

theorem g_at_5 : g 5 = -5 := 
  by 
  -- Proof goes here
  sorry

end g_at_5_l158_158686


namespace collinear_points_l158_158321

variables {α : Type*} [metric_space α]

noncomputable def is_incenter (I A B C : α) : Prop :=
-- definition of the incenter goes here
sorry

noncomputable def incircle_touch_point (I A B C : α) (D : α) : Prop :=
-- D is where the incircle touches BC
sorry

noncomputable def ray_intersect (A I P Q : α) : Prop :=
-- definitions for AI and DI intersecting circumcircle at P and Q respectively
sorry

noncomputable def line_contains_points (I K L : α) : Prop :=
-- line containing points
sorry

noncomputable def midpoint (L D I B : α) : Prop :=
L = D + (B - I) / 2

theorem collinear_points
  (I A B C D P Q K L : α)
  (h_incenter : is_incenter I A B C)
  (h_incircle_touch : incircle_touch_point I A B C D)
  (h_ray_intersect_P : ray_intersect A I P)
  (h_ray_intersect_Q : ray_intersect D I Q)
  (h_pass_line : line_contains_points I K L)
  (h_midpoint : midpoint L D I B) :
  ∃ X Y Z : α, X ≠ Y ∧ Y ≠ Z ∧ X ≠ Z ∧ X ∈ {K, P, Q, L} ∧ Y ∈ {K, P, Q, L} ∧ Z ∈ {K, P, Q, L} ∧ collinear X Y Z :=
sorry

end collinear_points_l158_158321


namespace problem1_problem2_l158_158819

-- Problem 1
theorem problem1 (α : ℝ) (h : sin (π / 3 - α) = 1 / 2) : cos (π / 6 + α) = 1 / 2 := 
  sorry

-- Problem 2
theorem problem2 (α : ℝ) (h1 : cos (5 * π / 12 + α) = 1 / 3) (h2 : -π < α ∧ α < -π / 2) :
  cos (7 * π / 12 - α) + sin (α - 7 * π / 12) = -1 / 3 + 2 * sqrt 2 / 3 :=
  sorry

end problem1_problem2_l158_158819


namespace survival_rate_is_98_l158_158737

def total_flowers := 150
def unsurviving_flowers := 3
def surviving_flowers := total_flowers - unsurviving_flowers

theorem survival_rate_is_98 : (surviving_flowers : ℝ) / total_flowers * 100 = 98 := by
  sorry

end survival_rate_is_98_l158_158737


namespace cyclic_MNST_l158_158506

open EuclideanGeometry

theorem cyclic_MNST (A B C D P M N T S : Point) (h1: IsQuadrilateral A B C D) 
(h2: ⟂ A C B D ∧ Intersects A C B D = P) 
(h3: FootPerpendicular P A B = M ∧ FootPerpendicular P A D = N)
(h4: Intersection (LineThrough P M) C D = T ∧ Intersection (LineThrough P N) B C = S) :
  Cyclic M N S T :=
by sorry

end cyclic_MNST_l158_158506


namespace zeta_2k_bernolli_formula_l158_158459

theorem zeta_2k_bernolli_formula (k : ℕ) (h : k ≥ 1) :
  let ζ (s : ℕ) := sum (fun n => 1 / (n : ℝ) ^ s) (range 1 (infty)),
      B : ℕ → ℝ := bernoulli,
      π : ℝ := real.pi,
      cot : ℝ → ℝ := real.cot
  in ζ (2 * k) = (2 * π)^(2 * k) * B (2 * k) / (2 * (2 * k)!) 
:= sorry

end zeta_2k_bernolli_formula_l158_158459


namespace angles_sum_n_l158_158454

/-- Given that the sum of the measures in degrees of angles A, B, C, D, E, and F is 90 * n,
    we need to prove that n = 4. -/
theorem angles_sum_n (A B C D E F : ℝ) (n : ℕ) 
  (h : A + B + C + D + E + F = 90 * n) :
  n = 4 :=
sorry

end angles_sum_n_l158_158454


namespace find_f_of_2_l158_158170

noncomputable def f (x : ℝ) : ℝ := x^2 + 2

theorem find_f_of_2 (x : ℝ) (hx : f(x - 1/x) = x^2 + 1/x^2) : f 2 = 6 := by
  sorry

end find_f_of_2_l158_158170


namespace arccos_one_over_sqrt_two_eq_pi_four_l158_158910

theorem arccos_one_over_sqrt_two_eq_pi_four : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 := 
by
  sorry

end arccos_one_over_sqrt_two_eq_pi_four_l158_158910


namespace find_b_l158_158774

-- Definitions
def polynomial (a b : ℝ) := λ x : ℝ, x^3 - a * x^2 + b * x - a^2

def all_roots_real (f : ℝ → ℝ) : Prop :=
  ∀ root : ℝ, f root = 0

-- Given conditions
def a := 9
noncomputable def b := 27

-- Lean statement for the proof
theorem find_b : ∀ a b : ℝ, (all_roots_real (polynomial a b) → a = 9 → b = 27) :=
by {
  intros a b h ha,
  sorry -- Proof goes here
}

end find_b_l158_158774


namespace simplified_expression_l158_158733

variable (m : ℝ) (h : m = Real.sqrt 3)

theorem simplified_expression : (m - (m + 9) / (m + 1)) / ((m^2 + 3 * m) / (m + 1)) = 1 - Real.sqrt 3 :=
by
  rw [h]
  sorry

end simplified_expression_l158_158733


namespace probability_of_even_sum_is_four_fifths_l158_158261

-- Define the first ten prime numbers
def first_ten_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- The function to calculate the probability that the sum of two distinct primes is even
def probability_even_sum (primes : List ℕ) : ℚ :=
  let pairs := primes.product primes
  let distinct_pairs := pairs.filter (λ p, p.1 ≠ p.2)
  let even_sum_pairs := distinct_pairs.filter (λ p, (p.1 + p.2) % 2 = 0)
  (even_sum_pairs.length : ℚ) / (distinct_pairs.length : ℚ)

-- Prove that the probability is 4/5
theorem probability_of_even_sum_is_four_fifths :
  probability_even_sum first_ten_primes = 4 / 5 := sorry

end probability_of_even_sum_is_four_fifths_l158_158261


namespace arccos_sqrt_half_l158_158964

theorem arccos_sqrt_half : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 := by
  sorry

end arccos_sqrt_half_l158_158964


namespace find_k_m_n_l158_158579

theorem find_k_m_n (t m n k : ℝ) (h1 : (1 + real.sin t) * (1 + real.cos t) = 5 / 4) 
(h2 : (1 - real.sin t) * (1 - real.cos t) = m / n - real.sqrt k) 
(h3 : k > 0) (h4 : m > 0) (h5 : n > 0) 
(h6 : nat.coprime (int.ofNat m) (int.ofNat n)) : k + m + n = 27 := 
sorry

end find_k_m_n_l158_158579


namespace minimize_expected_cost_l158_158857

noncomputable theory

def ticket_cost : ℝ := 50
def fine : ℝ := 450
def q : ℝ := 0.1

def expected_cost (p : ℝ) : ℝ :=
  50 * p + 500 * (1 - p) * q

theorem minimize_expected_cost :
  ∀ p : ℝ, 0 ≤ p ∧ p ≤ 1 → expected_cost p = 50 :=
begin
  intros p hp,
  unfold expected_cost,
  have h : 500 * (1 - p) * q = 50 * (1 - p), by norm_num [q],
  rw h,
  ring,
end

end minimize_expected_cost_l158_158857


namespace proposition2_proposition3_l158_158570

variable (a : Line) (α β : Plane)

def parallel (l1 l2 : Line) : Prop := sorry -- Define parallel property as needed
def perpendicular (l1 l2 : Line) : Prop := sorry -- Define perpendicular property as needed
def is_in_plane (l : Line) (p : Plane) : Prop := sorry -- Define line being in plane property as needed

theorem proposition2 (h : perpendicular a α) :
  ∀ l, is_in_plane l α → perpendicular a l :=
sorry

theorem proposition3 (h : parallel α β) :
  ∀ m, is_in_plane m β → parallel α m :=
sorry

end proposition2_proposition3_l158_158570


namespace walk_neg_eight_meters_east_means_eight_meters_west_l158_158784

theorem walk_neg_eight_meters_east_means_eight_meters_west :
  ∀ (dist : ℤ), dist = -8 → (dist < 0 → ∀ (direction : String), direction = "east" → true) → 
  dist = 8 ∧ ∀ (direction : String), direction = "west" := 
by
  intros dist hdist hneg dir hdir
  sorry

end walk_neg_eight_meters_east_means_eight_meters_west_l158_158784


namespace probability_sum_even_l158_158274

-- Let's define the set of the first ten prime numbers.
def first_ten_primes : Finset ℕ := Finset.of_list [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define the proposition to prove:
theorem probability_sum_even :
  let n := first_ten_primes.card in 
  let total_pairs := Finset.card (Finset.powersetLen 2 first_ten_primes) in 
  let even_pairs := Finset.card (Finset.filter (λ s, s.sum % 2 = 0) (Finset.powersetLen 2 first_ten_primes)) in 
  (total_pairs = 45 ∧ even_pairs = 36) →
  (even_pairs / total_pairs = 4 / 5) :=
by
  sorry

end probability_sum_even_l158_158274


namespace find_a_tangent_parallel_l158_158148

noncomputable theory
open Real

def tangent_parallel_condition (a : ℝ) : Prop :=
  let f : ℝ → ℝ := λ x, a * x * cos x + 16
  let slope_tangent := deriv f (π / 2)
  slope_tangent = 1

theorem find_a_tangent_parallel :
  ∃ a : ℝ, tangent_parallel_condition a ∧ a = -2 / π :=
sorry

end find_a_tangent_parallel_l158_158148


namespace calculate_r_l158_158049

def a := 0.24 * 450
def b := 0.62 * 250
def c := 0.37 * 720
def d := 0.38 * 100
def sum_bc := b + c
def diff := sum_bc - a
def r := diff / d

theorem calculate_r : r = 8.25 := by
  sorry

end calculate_r_l158_158049


namespace sum_of_reciprocals_of_shifted_roots_l158_158681

theorem sum_of_reciprocals_of_shifted_roots (p q r : ℝ)
  (h1 : p^3 - 2 * p^2 - p + 3 = 0)
  (h2 : q^3 - 2 * q^2 - q + 3 = 0)
  (h3 : r^3 - 2 * r^2 - r + 3 = 0) :
  (1 / (p - 2)) + (1 / (q - 2)) + (1 / (r - 2)) = -3 :=
by
  sorry

end sum_of_reciprocals_of_shifted_roots_l158_158681


namespace dice_symmetric_sum_l158_158121

theorem dice_symmetric_sum (n : ℕ) (s1 s6 : ℕ) (k : ℕ) : 
  n = 9 → s1 = 1 → s6 = 6 → k = 15 → 
  2 * (n * (s1 + s6)) / 2 - k = 48 :=
by
  intros n_eq s1_eq s6_eq k_eq
  rw [n_eq, s1_eq, s6_eq, k_eq]
  calc
    2 * (9 * (1 + 6)) / 2 - 15 = 2 * 31.5 - 15 : by norm_num
                        ... = 48 : by norm_num

end dice_symmetric_sum_l158_158121


namespace derivative_function_evaluation_l158_158595

theorem derivative_function_evaluation (f : ℝ → ℝ) (h : f = λ x, (1/2) * (f' 1) * x^2 + log x + (f 1) / (3 * x)) :
  deriv f 2 = 33 / 4 := by
sorry

end derivative_function_evaluation_l158_158595


namespace digit_200_of_5_div_13_is_8_l158_158426

/-- Prove that the 200th digit beyond the decimal point in the decimal representation
    of 5/13 is 8 --/
theorem digit_200_of_5_div_13_is_8 :
  let repeating_sequence := "384615" in
  let digit_200 := repeating_sequence[200 % 6 -1] in
  digit_200 = '8' :=
by
  sorry

end digit_200_of_5_div_13_is_8_l158_158426


namespace B_work_days_l158_158078

-- Define work rates and conditions
def A_work_rate : ℚ := 1 / 18
def B_work_rate : ℚ := 1 / 15
def A_days_after_B_left : ℚ := 6
def total_work : ℚ := 1

-- Theorem statement
theorem B_work_days : ∃ x : ℚ, (x * B_work_rate + A_days_after_B_left * A_work_rate = total_work) → x = 10 := by
  sorry

end B_work_days_l158_158078


namespace maximum_area_of_garden_l158_158106

noncomputable def max_area (perimeter : ℕ) : ℕ :=
  let half_perimeter := perimeter / 2
  let x := half_perimeter / 2
  x * x

theorem maximum_area_of_garden :
  max_area 148 = 1369 :=
by
  sorry

end maximum_area_of_garden_l158_158106


namespace mutual_fund_share_increase_l158_158874

theorem mutual_fund_share_increase (P : ℝ) (h1 : (P * 1.20) = 1.20 * P) (h2 : (1.20 * P) * (1 / 3) = 0.40 * P) :
  ((1.60 * P) = (P * 1.60)) :=
by
  sorry

end mutual_fund_share_increase_l158_158874


namespace minnie_penny_time_difference_l158_158351

noncomputable def minnie_time_uphill (distance speed: ℝ) := distance / speed
noncomputable def minnie_time_downhill (distance speed: ℝ) := distance / speed
noncomputable def minnie_time_flat (distance speed: ℝ) := distance / speed
noncomputable def penny_time_flat (distance speed: ℝ) := distance / speed
noncomputable def penny_time_downhill (distance speed: ℝ) := distance / speed
noncomputable def penny_time_uphill (distance speed: ℝ) := distance / speed
noncomputable def break_time (minutes: ℝ) := minutes / 60

noncomputable def minnie_total_time :=
  minnie_time_uphill 12 6 + minnie_time_downhill 18 25 + minnie_time_flat 25 18

noncomputable def penny_total_time :=
  penny_time_flat 25 25 + penny_time_downhill 12 35 + 
  penny_time_uphill 18 12 + break_time 10

noncomputable def time_difference := (minnie_total_time - penny_total_time) * 60

theorem minnie_penny_time_difference :
  time_difference = 66 := by
  sorry

end minnie_penny_time_difference_l158_158351


namespace pure_imaginary_number_l158_158623

theorem pure_imaginary_number (a : ℝ) (ha : (1 + a) / (1 + a^2) = 0) : a = -1 :=
sorry

end pure_imaginary_number_l158_158623


namespace number_of_valid_arrangements_l158_158725

theorem number_of_valid_arrangements : 
  (∃ (arrangements : Finset (List ℕ)), 
   arrangements.card = 4 ∧
   ∀ (L : List ℕ), L ∈ arrangements → 
     (∃ (k : ℕ), k ∈ List.range 7 ∧ 
      (List.remove_nth L k).sorted (≤) ∨ (List.remove_nth L k).sorted (≥))) := 
sorry

end number_of_valid_arrangements_l158_158725


namespace planes_intersect_l158_158178

def Point : Type := sorry
def Plane : Type := sorry

variables (A B C : Point) (α : Plane)

-- Conditions
axiom A_on_α : A ∈ α
axiom B_not_on_α : B ∉ α
axiom C_not_on_α : C ∉ α

-- Define the plane ABC
def plane_ABC : Plane := sorry

-- Goal
theorem planes_intersect : ∃ l : Line, ∀ p : Point, (p ∈ l ↔ (p ∈ α ∧ p ∈ plane_ABC)) :=
sorry

end planes_intersect_l158_158178


namespace circle_equation_l158_158475

theorem circle_equation :
  ∃ (C : ℝ × ℝ) (r : ℝ), C = (3, 0) ∧ r = sqrt 2 ∧
    (∀ (x y : ℝ), (x - 3)^2 + y^2 = 2 ↔ (C.1 - x)^2 + (C.2 - y)^2 = r^2) ∧
    (∃ (A B : ℝ × ℝ), A = (4, 1) ∧ B = (2, 1) ∧ 
     ((∃ (m : ℝ), m ≠ 0 ∧ (x - y = 1)) ∧
     (∃ (m : ℝ), m ≠ 0 ∧ (x + y = 3)))) :=
begin
  sorry
end

end circle_equation_l158_158475


namespace arccos_of_one_over_sqrt_two_l158_158884

theorem arccos_of_one_over_sqrt_two :
  Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 := 
sorry

end arccos_of_one_over_sqrt_two_l158_158884


namespace logarithmic_and_inverse_proportion_l158_158208

theorem logarithmic_and_inverse_proportion :
  (∀ (a : ℝ), a > 0 ∧ a ≠ 1 → f(2) = log a 2 → f = log 4) ∧
  (∀ (k : ℝ), g(2) = k / 2 → k = 1 ∧ g = λ x, 1 / x) ∧
  (∀ (x : ℝ), (g(f(x)) < 2) ↔ (0 < x ∧ x < 1) ∨ (2 < x)) :=
by sorry

end logarithmic_and_inverse_proportion_l158_158208


namespace neilInitialGames_l158_158221

-- Hypotheses based on the given conditions
variable (initialHenryGames : ℕ) (givenGamesToNeil : ℕ) (henryGamesAfterGiving : ℕ)
variable (henryHasFourTimesMoreGames : ℕ)

-- Given conditions as Lean variable assignments
def henryInitialGames := initialHenryGames = 33
def gamesGivenToNeil := givenGamesToNeil = 5
def henryGamesPostGiving := henryGamesAfterGiving = 33 - 5
def henryFourTimesMoreGames := henryHasFourTimesMoreGames = 4 * (henryGamesAfterGiving / 4)

-- Final statement to prove using the conditions
theorem neilInitialGames (H1 : henryInitialGames) (H2 : gamesGivenToNeil) (H3 : henryGamesPostGiving) (H4 : henryFourTimesMoreGames) : 
  henryGamesAfterGiving / 4 - 5 = 2 :=
by
  sorry

end neilInitialGames_l158_158221


namespace arccos_sqrt2_l158_158938

def arccos_eq (x : ℝ) := arccos x
def range_arccos := ∀ (x : ℝ), 0 ≤ arccos x ∧ arccos x ≤ π
def cos_pi_div_four : Real.cos (π / 4) = 1 / Real.sqrt 2 := by sorry

theorem arccos_sqrt2 :
  arccos_eq (1 / Real.sqrt 2) = π / 4 :=
by
  have h1 : Real.cos (π / 4) = 1 / Real.sqrt 2, from cos_pi_div_four
  sorry

end arccos_sqrt2_l158_158938


namespace arccos_of_one_over_sqrt_two_l158_158883

theorem arccos_of_one_over_sqrt_two :
  Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 := 
sorry

end arccos_of_one_over_sqrt_two_l158_158883


namespace meter_to_skips_l158_158742

/-!
# Math Proof Problem
Suppose hops, skips and jumps are specific units of length. Given the following conditions:
1. \( b \) hops equals \( c \) skips.
2. \( d \) jumps equals \( e \) hops.
3. \( f \) jumps equals \( g \) meters.

Prove that one meter equals \( \frac{cef}{bdg} \) skips.
-/

theorem meter_to_skips (b c d e f g : ℝ) (h1 : b ≠ 0) (h2 : c ≠ 0) (h3 : d ≠ 0) (h4 : e ≠ 0) (h5 : f ≠ 0) (h6 : g ≠ 0) :
  (1 : ℝ) = (cef) / (bdg) :=
by
  -- skipping the proof
  sorry

end meter_to_skips_l158_158742


namespace arccos_one_over_sqrt_two_eq_pi_over_four_l158_158925

theorem arccos_one_over_sqrt_two_eq_pi_over_four : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_eq_pi_over_four_l158_158925


namespace arccos_one_over_sqrt_two_l158_158979

theorem arccos_one_over_sqrt_two :
  Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_l158_158979


namespace price_of_appliance_l158_158535

-- Definitions of the discount conditions
def discount (price : ℝ) : ℝ :=
  if price ≤ 200 then 0
  else if price ≤ 500 then (price - 200) * 0.1
  else (300 * 0.1) + (price - 500) * 0.2

-- Given condition: total savings is 330 yuan
def total_savings (savings : ℝ) := savings = 330

-- Proof statement to show the price of the appliance
theorem price_of_appliance (price : ℝ) (h : total_savings (discount price)) : price = 2000 :=
by sorry

end price_of_appliance_l158_158535


namespace triangle_cosine_sum_l158_158363

theorem triangle_cosine_sum (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C)
  (hsum : A + B + C = π) : 
  (Real.cos A + Real.cos B + Real.cos C > 1) :=
sorry

end triangle_cosine_sum_l158_158363


namespace cubes_sum_l158_158745

theorem cubes_sum (a b c : ℝ) (h1 : a + b + c = 1) (h2 : ab + ac + bc = -4) (h3 : abc = -6) :
  a^3 + b^3 + c^3 = -5 :=
by
  sorry

end cubes_sum_l158_158745


namespace additional_money_needed_l158_158116

def original_num_bales : ℕ := 10
def original_cost_per_bale : ℕ := 15
def new_cost_per_bale : ℕ := 18

theorem additional_money_needed :
  (2 * original_num_bales * new_cost_per_bale) - (original_num_bales * original_cost_per_bale) = 210 :=
by
  sorry

end additional_money_needed_l158_158116


namespace arccos_one_over_sqrt_two_eq_pi_four_l158_158903

theorem arccos_one_over_sqrt_two_eq_pi_four : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 := 
by
  sorry

end arccos_one_over_sqrt_two_eq_pi_four_l158_158903


namespace line_passes_through_fixed_point_minimum_area_triangle_l158_158174

theorem line_passes_through_fixed_point (k : ℝ) :
  ∃ P : ℝ × ℝ, P = (2, 1) ∧ ∀ k : ℝ, (k * 2 - 1 + 1 - 2 * k = 0) :=
sorry

theorem minimum_area_triangle (k : ℝ) :
  ∀ k: ℝ, k < 0 → 1/2 * (2 - 1/k) * (1 - 2*k) ≥ 4 ∧ 
           (1/2 * (2 - 1/k) * (1 - 2*k) = 4 ↔ k = -1/2) :=
sorry

end line_passes_through_fixed_point_minimum_area_triangle_l158_158174


namespace arccos_sqrt2_l158_158942

def arccos_eq (x : ℝ) := arccos x
def range_arccos := ∀ (x : ℝ), 0 ≤ arccos x ∧ arccos x ≤ π
def cos_pi_div_four : Real.cos (π / 4) = 1 / Real.sqrt 2 := by sorry

theorem arccos_sqrt2 :
  arccos_eq (1 / Real.sqrt 2) = π / 4 :=
by
  have h1 : Real.cos (π / 4) = 1 / Real.sqrt 2, from cos_pi_div_four
  sorry

end arccos_sqrt2_l158_158942


namespace modified_cube_edges_l158_158853

theorem modified_cube_edges : 
  let original_edges := 12
  let modifications := 4 * 3 + 4 * 3
  let resulting_edges := original_edges + modifications
  in resulting_edges = 48 :=
by
  let original_edges := 12
  let modifications := 4 * 3 + 4 * 3
  let resulting_edges := original_edges + modifications
  show resulting_edges = 48
  sorry

end modified_cube_edges_l158_158853


namespace arccos_of_one_over_sqrt_two_l158_158889

theorem arccos_of_one_over_sqrt_two :
  Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 := 
sorry

end arccos_of_one_over_sqrt_two_l158_158889


namespace find_max_min_l158_158143

noncomputable def given_function (x : ℝ) : ℝ :=
  (∫ t in -Real.pi / 4 .. x, Real.sin t) + -Real.sin (x - Real.pi / 3)

theorem find_max_min :
  ∃ x_min x_max f_min f_max,
    -Real.pi / 4 ≤ x_min ∧ x_min ≤ Real.pi / 4 ∧
    -Real.pi / 4 ≤ x_max ∧ x_max ≤ Real.pi / 4 ∧
    (∀ x, -Real.pi / 4 ≤ x ∧ x ≤ Real.pi / 4 → given_function x_min ≤ given_function x) ∧
    (∀ x, -Real.pi / 4 ≤ x ∧ x ≤ Real.pi / 4 → given_function x_max ≥ given_function x) :=
sorry

end find_max_min_l158_158143


namespace arnold_total_protein_l158_158866

-- Definitions and conditions
def collagen_protein_per_scoop : ℕ := 18 / 2
def protein_powder_protein_per_scoop : ℕ := 21
def steak_protein : ℕ := 56

def collagen_scoops : ℕ := 1
def protein_powder_scoops : ℕ := 1
def steaks : ℕ := 1

-- Statement of the theorem/problem
theorem arnold_total_protein : 
  (collagen_protein_per_scoop * collagen_scoops) + 
  (protein_powder_protein_per_scoop * protein_powder_scoops) + 
  (steak_protein * steaks) = 86 :=
by
  sorry

end arnold_total_protein_l158_158866


namespace bisectors_coincide_l158_158358

theorem bisectors_coincide
  (ABC : Triangle)
  (A1 B1 C1 : Point)
  (A1_midpoint : midpoint A1 ABC.B ABC.C)
  (B1_midpoint : midpoint B1 ABC.A ABC.C)
  (C1_midpoint : midpoint C1 ABC.A ABC.B)
  (E F : Point)
  (E_on_midline : on_midline E C1 B1)
  (F_on_midline : on_midline F A1 B1)
  (BE_angle_bisector : angle_bisector (BE E A1 B1))
  (BF_angle_bisector : angle_bisector (BF F C1 B1)) :
  angle_bisectors_coincide (angle_ABC ABC) (angle_FBE E B1 F) := 
sorry

end bisectors_coincide_l158_158358


namespace part1_part2_l158_158804

-- Definitions of conditions
def cost_price : ℝ := 5
def original_selling_price : ℝ := 8
def original_sales_volume : ℝ := 100
def sales_increase_per_0_1_decrease : ℝ := 10

-- Definition for part 1
def sales_volume_at_price (new_price : ℝ) : ℝ :=
  original_sales_volume + sales_increase_per_0_1_decrease * ((original_selling_price - new_price) / 0.1)

-- Part 1 statement
theorem part1 : sales_volume_at_price 7 = 200 := 
by sorry

-- Definitions for part 2
def profit_per_piece (decrease_in_price : ℝ) : ℝ :=
  original_selling_price - decrease_in_price - cost_price

def daily_sales_volume (decrease_in_price : ℝ) : ℝ :=
  original_sales_volume + sales_increase_per_0_1_decrease * (decrease_in_price / 0.1)

def daily_profit (decrease_in_price : ℝ) : ℝ :=
  (profit_per_piece decrease_in_price) * (daily_sales_volume decrease_in_price)

-- Part 2 statement
theorem part2 : daily_profit 1.5 = 375 := 
by sorry

end part1_part2_l158_158804


namespace arccos_one_over_sqrt_two_l158_158945

theorem arccos_one_over_sqrt_two : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_l158_158945


namespace max_lambda_l158_158544

theorem max_lambda (λ : ℝ) (h₁ : ∀ x y : ℝ, x > 0 → y > 0 → 2 * x - y = 2 * x^3 + y^3) 
  (h₂ : ∀ x y : ℝ, x > 0 → y > 0 → x^2 + λ * y^2 ≤ 1) :
  λ ≤ (Real.sqrt 5 + 1) / 2 :=
sorry

end max_lambda_l158_158544


namespace distance_between_foci_l158_158990

-- Define the foci points
def F1 : ℝ × ℝ := (4, 3)
def F2 : ℝ × ℝ := (-6, 9)

-- Define the distance function between two points
def dist (A B : ℝ × ℝ) : ℝ := sqrt ((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2)

-- Theorem stating the distance between F1 and F2 is 2 * sqrt 34
theorem distance_between_foci : dist F1 F2 = 2 * sqrt 34 := by
  sorry

end distance_between_foci_l158_158990


namespace boys_in_class_l158_158766

theorem boys_in_class (students : ℕ) (ratio_girls_boys : ℕ → Prop)
  (h1 : students = 56)
  (h2 : ratio_girls_boys 4 ∧ ratio_girls_boys 3) :
  ∃ k : ℕ, 4 * k + 3 * k = students ∧ 3 * k = 24 :=
by
  sorry

end boys_in_class_l158_158766


namespace maximal_set_of_integers_l158_158529

open Finset

/-- The maximal size of a set of positive integers with digits from {1,2,3,4,5,6},
    where no digit occurs more than once in the same integer, digits are in increasing order,
    any two integers have at least one digit in common, and no digit appears in all integers,
    is 32. -/
theorem maximal_set_of_integers (S : Finset (Finset ℕ)) (elems := {1, 2, 3, 4, 5, 6} : Finset ℕ) :
    (∀ x ∈ S, x ⊆ elems) ∧
    (∀ x ∈ S, (∀ {a b}, a ≠ b → a ∈ x → b ∈ x → a < b)) ∧
    (∀ {x y}, x ∈ S → y ∈ S → (x ≠ y → x ∩ y ≠ ∅)) ∧
    (∀ n, ∃ x ∈ S, ¬ n ∈ x) →
    S.card = 32 := 
sorry

end maximal_set_of_integers_l158_158529


namespace major_premise_is_wrong_l158_158845

-- Definitions of the conditions
def line_parallel_to_plane (l : Type) (p : Type) : Prop := sorry
def line_contained_in_plane (l : Type) (p : Type) : Prop := sorry

-- Stating the main problem: the major premise is wrong
theorem major_premise_is_wrong :
  ∀ (a b : Type) (α : Type), line_contained_in_plane a α → line_parallel_to_plane b α → ¬ (line_parallel_to_plane b a) := 
by 
  intros a b α h1 h2
  sorry

end major_premise_is_wrong_l158_158845


namespace pirate_ends_with_all_coins_l158_158086

noncomputable def pirate_theft (coins : ℕ) (rounds : ℕ) : ℕ := sorry

theorem pirate_ends_with_all_coins :
  ∀ (pirates : List ℕ) (total_coins : ℕ) (rounds : ℕ), 
    total_coins = 128 →
    rounds = 7 →
    (∀ i, i < rounds → even coins) →
    (∃ p, p ∈ pirates ∧ pirate_theft total_coins rounds = total_coins) :=
sorry

end pirate_ends_with_all_coins_l158_158086


namespace probability_even_sum_l158_158282

def first_ten_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def distinct_pairs (l : List ℕ) : List (ℕ × ℕ) :=
  (l.product l).filter (λ p, p.1 < p.2)

theorem probability_even_sum : 
  (first_ten_primes.length = 10) →
  (∀ a b : ℕ, a ∈ first_ten_primes → b ∈ first_ten_primes → a ≠ b → 
    ((a + b) % 2 = 0 ↔ 2 ∉ [a, b])) →
  (∑ pair in distinct_pairs first_ten_primes, ((pair.1 + pair.2) % 2 = 0).toNat) 
    = 36 → 
  ((∑ pair in distinct_pairs first_ten_primes, (1:ℚ)) ⁻¹ * 
   (∑ pair in distinct_pairs first_ten_primes, ((pair.1 + pair.2) % 2 = 0).toNat))
    = 4 / 5 := by
  sorry

end probability_even_sum_l158_158282


namespace abs_ineq_range_l158_158568

theorem abs_ineq_range (x : ℝ) : |x - 3| + |x + 1| ≥ 4 ↔ -1 ≤ x ∧ x ≤ 3 :=
sorry

end abs_ineq_range_l158_158568


namespace area_of_triangle_DEF_l158_158296

variables (DE EF DF : ℝ)
def area_of_triangle (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2 in
  real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem area_of_triangle_DEF :
  DE = 24 → EF = 24 → DF = 35 → area_of_triangle DE EF DF = 288 :=
by
  intros hDE hEF hDF
  rw [hDE, hEF, hDF]
  sorry

end area_of_triangle_DEF_l158_158296


namespace inequality_proof_l158_158721

theorem inequality_proof (a b : ℝ) (h : a + b ≠ 0) :
  (a + b) / (a^2 - a * b + b^2) ≤ 4 / |a + b| ∧
  ((a + b) / (a^2 - a * b + b^2) = 4 / |a + b| ↔ a = b) :=
by
  sorry

end inequality_proof_l158_158721


namespace arccos_sqrt_half_l158_158972

theorem arccos_sqrt_half : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 := by
  sorry

end arccos_sqrt_half_l158_158972


namespace arccos_one_over_sqrt_two_l158_158981

theorem arccos_one_over_sqrt_two :
  Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_l158_158981


namespace probability_even_sum_l158_158247

def firstTenPrimes : Set ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

def isOdd (n : ℕ) : Prop := n % 2 = 1

def distinctPairs (s : Set ℕ) : Set (ℕ × ℕ) :=
  { (x, y) | x ∈ s ∧ y ∈ s ∧ x ≠ y }

def eventEvenSum (s : Set ℕ) : Set (ℕ × ℕ) :=
  { (x, y) | x ∈ s ∧ y ∈ s ∧ x ≠ y ∧ (x + y) % 2 = 0 }

theorem probability_even_sum :
  (Set.card (eventEvenSum firstTenPrimes)).toRat / (Set.card (distinctPairs firstTenPrimes)).toRat = 4 / 5 :=
  sorry

end probability_even_sum_l158_158247


namespace express_as_scientific_notation_l158_158102

-- Define the question and condition
def trillion : ℝ := 1000000000000
def num := 6.13 * trillion

-- The main statement to be proven
theorem express_as_scientific_notation : num = 6.13 * 10^12 :=
by
  sorry

end express_as_scientific_notation_l158_158102


namespace arnold_total_protein_l158_158867

-- Definitions and conditions
def collagen_protein_per_scoop : ℕ := 18 / 2
def protein_powder_protein_per_scoop : ℕ := 21
def steak_protein : ℕ := 56

def collagen_scoops : ℕ := 1
def protein_powder_scoops : ℕ := 1
def steaks : ℕ := 1

-- Statement of the theorem/problem
theorem arnold_total_protein : 
  (collagen_protein_per_scoop * collagen_scoops) + 
  (protein_powder_protein_per_scoop * protein_powder_scoops) + 
  (steak_protein * steaks) = 86 :=
by
  sorry

end arnold_total_protein_l158_158867


namespace solve_hyperbola_problem_solve_ellipse_problem_l158_158069

noncomputable def hyperbola_condition1 (m : ℝ) : Prop := 
  ∀ (x y : ℝ), (x^2 / m) + (y^2 / 2) = 1 → 
  let e := (real.sqrt ((2 - m) / 2)) in
  e = 2 → m = -6

theorem solve_hyperbola_problem : hyperbola_condition1 m :=
by
  sorry

noncomputable def ellipse_condition1 (a^2 b^2 : ℝ) : Prop :=
  let ellipse1 := (x : ℝ) (y : ℝ), x^2 / a^2 + y^2 / b^2 = 1
  let ellipse2 := (x : ℝ) (y : ℝ), y^2 / a^2 + x^2 / b^2 = 1
  (2 * b^2 = a^2) → (ellipse1 (2, -6) ∨ ellipse2 (2, -6)) →
  (a^2 = 148 ∧ b^2 = 37) ∨ (a^2 = 52 ∧ b^2 = 13)

theorem solve_ellipse_problem : ellipse_condition1 a^2 b^2 :=
by
  sorry

end solve_hyperbola_problem_solve_ellipse_problem_l158_158069


namespace solve_equation_l158_158165

-- Define the given matrix
def matrix (x a : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![x^2 + 2 * a, a, 2 * x],
    ![a, x^2 + 3 * a, a],
    ![2 * x, a, x^2 + a]]

-- Define the condition
def condition (a : ℝ) : Prop := a ≠ 0

-- Define the solution
def solution (a : ℝ) : ℝ := -a^(2 / 3)

-- Statement to prove
theorem solve_equation (a : ℝ) (h : condition a) : 
  (det (matrix (solution a) a) = 0) :=
sorry

end solve_equation_l158_158165


namespace p_sufficient_not_necessary_for_q_l158_158567

variables {x : ℝ}
def p := x = 2
def q := 0 < x ∧ x < 3

theorem p_sufficient_not_necessary_for_q : (p → q) ∧ ¬(q → p) :=
by 
  sorry

end p_sufficient_not_necessary_for_q_l158_158567


namespace arccos_of_one_over_sqrt_two_l158_158892

theorem arccos_of_one_over_sqrt_two :
  Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 := 
sorry

end arccos_of_one_over_sqrt_two_l158_158892


namespace bill_needs_paint_cans_l158_158512

theorem bill_needs_paint_cans :
  let bedrooms := 3
  let other_rooms := 2 * bedrooms
  let gallons_per_room := 2
  let color_paint_cans := 6 -- (bedrooms * gallons_per_room) / 1-gallon per can
  let white_paint_cans := 4 -- (other_rooms * gallons_per_room) / 3-gallons per can
  (color_paint_cans + white_paint_cans) = 10 := sorry

end bill_needs_paint_cans_l158_158512


namespace probability_even_sum_l158_158246

def firstTenPrimes : Set ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

def isOdd (n : ℕ) : Prop := n % 2 = 1

def distinctPairs (s : Set ℕ) : Set (ℕ × ℕ) :=
  { (x, y) | x ∈ s ∧ y ∈ s ∧ x ≠ y }

def eventEvenSum (s : Set ℕ) : Set (ℕ × ℕ) :=
  { (x, y) | x ∈ s ∧ y ∈ s ∧ x ≠ y ∧ (x + y) % 2 = 0 }

theorem probability_even_sum :
  (Set.card (eventEvenSum firstTenPrimes)).toRat / (Set.card (distinctPairs firstTenPrimes)).toRat = 4 / 5 :=
  sorry

end probability_even_sum_l158_158246


namespace compare_abc_l158_158167

noncomputable def a : ℝ := (1 / 2) * Real.cos (4 * Real.pi / 180) - (Real.sqrt 3 / 2) * Real.sin (4 * Real.pi / 180)
noncomputable def b : ℝ := Real.cos (2 * 13 * Real.pi / 180)
noncomputable def c : ℝ := Real.tan (2 * 23 * Real.pi / 180)

theorem compare_abc : a < b ∧ b < c := by
  sorry

end compare_abc_l158_158167


namespace arccos_one_over_sqrt_two_l158_158973

theorem arccos_one_over_sqrt_two :
  Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_l158_158973


namespace starting_player_wins_l158_158776

-- Define the initial piles of matches
def initial_piles : list ℕ := [100, 200, 300]

-- Define the game condition and winning strategy
theorem starting_player_wins (piles : list ℕ) : 
  piles = initial_piles → 
  (∃ strategy : list ℕ → list ℕ, ∀ piles', (strategy piles').length = 2 ∧ (strategy (strategy piles')).length = 2 → 
  (∃ winning_strategy : list ℕ → list ℕ, ∀ p', p' = strategy (winning_strategy piles) → starting_player))
  ∧ (@winning_condition : list ℕ → Prop, winning_condition []) → 
  (starting_player) :=
by
  intros h_piles h_strategy h_winning_condition,
  have h_initial : piles = [100, 200, 300], from h_piles,
  -- We will assume the remaining parts to have their respective strategy function
  -- and the recursive property to demonstrate that the first player always wins
  sorry

end starting_player_wins_l158_158776


namespace range_g_l158_158986

theorem range_g (f : ℝ → ℝ) (g : ℝ → ℝ) (h_f : ∀ x, f x = x^2) (h_g : ∀ x, g x = f x - 2 * x) :
  set.range g = set.Icc (-1 : ℝ) 8 :=
by
  sorry

end range_g_l158_158986


namespace total_cows_l158_158099

theorem total_cows (Matthews Aaron Tyron Marovich : ℕ) 
  (h1 : Matthews = 60)
  (h2 : Aaron = 4 * Matthews)
  (h3 : Tyron = Matthews - 20)
  (h4 : Aaron + Matthews + Tyron = Marovich + 30) :
  Aaron + Matthews + Tyron + Marovich = 650 :=
by
  sorry

end total_cows_l158_158099


namespace total_chickens_l158_158367

theorem total_chickens (hens : ℕ) (roosters : ℕ) (h1 : hens = 52) (h2 : roosters = hens + 16) : hens + roosters = 120 :=
by
  rw [h1, h2]
  norm_num
  sorry

end total_chickens_l158_158367


namespace arccos_identity_l158_158921

theorem arccos_identity : 
  ∀ x : ℝ, x = 1 / real.sqrt 2 → real.arccos x = real.pi / 4 :=
begin
  intros x hx,
  have h1 : real.cos (real.pi / 4) = 1 / real.sqrt 2 := by sorry,
  have h2 : real.arccos (1 / real.sqrt 2) = real.pi / 4,
  { rw ← h1,
    rw real.arccos_cos,
    exact_mod_cast hx },
  exact h2,
end

end arccos_identity_l158_158921


namespace find_value_l158_158139

theorem find_value : (1 / 4 * (5 * 9 * 4) - 7) = 38 := 
by
  sorry

end find_value_l158_158139


namespace probability_even_sum_l158_158270

-- Definition of the prime numbers set and the selection scenario
def firstTenPrimes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Condition stating that we select two distinct numbers at random from the first ten primes
def randomSelection (s : List ℕ) := {x : ℕ × ℕ // x.1 ≠ x.2 ∧ x.1 ∈ s ∧ x.2 ∈ s}

-- Define the event that their sum is even
def evenSum (p : ℕ × ℕ) := (p.1 + p.2) % 2 = 0

-- Define the probability calculation
def probabilityEvenSum := 
  let totalPairs := (firstTenPrimes.length.choose 2) -- Calculate the number of ways to choose 2 numbers from the 10 primes
  let evenPairs := (randomSelection firstTenPrimes).count (λ p => evenSum p.val)
  evenPairs.toFloat / totalPairs.toFloat

-- Theorem statement that the probability of their sum being even is 1/5
theorem probability_even_sum : probabilityEvenSum = 1/5 := by
  sorry -- The actual proof is to be provided

end probability_even_sum_l158_158270


namespace binary_sum_of_350_and_1500_has_11_digits_l158_158438

theorem binary_sum_of_350_and_1500_has_11_digits :
  let sum := 350 + 1500
  in (Integer.log2 sum + 1) = 11 :=
by
  let sum := 350 + 1500
  have h : sum = 1850 := by norm_num
  rw [h]
  exact (Integer.log2 1850 + 1 = 11)

end binary_sum_of_350_and_1500_has_11_digits_l158_158438


namespace arccos_one_over_sqrt_two_eq_pi_four_l158_158906

theorem arccos_one_over_sqrt_two_eq_pi_four : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 := 
by
  sorry

end arccos_one_over_sqrt_two_eq_pi_four_l158_158906


namespace correct_statements_are_abd_l158_158643

theorem correct_statements_are_abd
  (initial_scores : list ℚ)
  (h_length : initial_scores.length = 9) :
  let valid_scores := (initial_scores.erase (initial_scores.maximum)).erase (initial_scores.minimum) in
  valid_scores.length = 7 ∧
  (list.median initial_scores = list.median valid_scores) ∧
  (list.lowerQuartile initial_scores = list.lowerQuartile valid_scores) ∧
  (list.variance initial_scores ≥ list.variance valid_scores) :=
by
  sorry

end correct_statements_are_abd_l158_158643


namespace team_A_wins_2_1_team_B_wins_l158_158779

theorem team_A_wins_2_1 (p_a p_b : ℝ)
  (h1 : p_a = 0.6)
  (h2 : p_b = 0.4)
  (h3 : ∀ {x y: ℝ}, x + y = 1)
  (h4 : ∃ n : ℕ, n = 3) : (2 * p_a * p_b) * p_a = 0.288 := by
  sorry

theorem team_B_wins (p_a p_b : ℝ)
  (h1 : p_a = 0.6)
  (h2 : p_b = 0.4)
  (h3 : ∀ {x y: ℝ}, x + y = 1)
  (h4 : ∃ n : ℕ, n = 3) : (p_b * p_b) + (2 * p_a * p_b * p_b) = 0.352 := by
  sorry

end team_A_wins_2_1_team_B_wins_l158_158779


namespace age_youngest_child_l158_158484

theorem age_youngest_child (a b c : ℕ) (ha : a ≤ b) (hb : b ≤ c) (h_abc_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_father_cost : 5 : ℝ) 
  (h_total_bill : (5 + 0.50 * (a + b + c)) = 11.50 : ℝ) : 
  a = 1 ∨ a = 2 ∨ a = 3 :=
by
  sorry

end age_youngest_child_l158_158484


namespace root_in_interval_l158_158396

open Function

def f (x : ℝ) : ℝ := 2^x - x^2 - 1 / 2

theorem root_in_interval : ∃ x ∈ Ioo (3 / 2) 2, f x = 0 := by
  sorry

end root_in_interval_l158_158396


namespace conclusion_2_conclusion_3_conclusion_5_proof_correct_l158_158554

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem conclusion_2 (x0 : ℝ) (h : f' x0 = 2) : x0 = Real.exp 1 := 
by
  unfold f'
  sorry

theorem conclusion_3 : ∀ x : ℝ, x ∈ (Set.Icc (1 / Real.exp 1) ∞) ↔ (Real.log x + 1 > 0) := 
by 
  unfold f
  sorry

theorem conclusion_5 : ∃ x : ℝ, f x = -1 / Real.exp 1 ∧ ∀ y : ℝ, y ≠ x → f y ≠ -1 / Real.exp 1 := 
by 
  unfold f
  sorry

theorem proof_correct : 
  (¬ ∀ x : ℝ, f x = f (-x)) ∧
  (∀ x0 : ℝ, f' x0 = 2 → x0 = Real.exp 1) ∧
  (∀ x : ℝ, x ∈ (Set.Icc (1 / Real.exp 1) ∞) ↔ (Real.log x + 1 > 0)) ∧
  (Set.range f = Set.Icc (-1 / Real.exp 1) ∞) ∧
  (∃ x : ℝ, f x = -1 / Real.exp 1 ∧ ∀ y : ℝ, y ≠ x → f y ≠ -1 / Real.exp 1) := 
by
  unfold f
  sorry

end conclusion_2_conclusion_3_conclusion_5_proof_correct_l158_158554


namespace soda_pyramid_cases_needed_l158_158698

theorem soda_pyramid_cases_needed : 
  let triangular_number (n : ℕ) := n * (n + 1) / 2 in
  (triangular_number 1) + 
  (triangular_number 2) + 
  (triangular_number 3) + 
  (triangular_number 4) + 
  (triangular_number 5) + 
  (triangular_number 6) = 56 := 
by
  sorry

end soda_pyramid_cases_needed_l158_158698


namespace vector_dot_product_l158_158231

variable {V : Type*} [InnerProductSpace ℝ V]

variables (p q r : V)

theorem vector_dot_product :
  ⟪q, 5 • r - 3 • p⟫ = -35 :=
by
  -- Given conditions
  have h1 : ⟪p, q⟫ = 5 := sorry,
  have h2 : ⟪p, r⟫ = -2 := sorry,
  have h3 : ⟪q, r⟫ = -4 := sorry,
  -- Prove the result
  sorry

end vector_dot_product_l158_158231


namespace carol_paints_180_square_feet_l158_158501

noncomputable def total_work : ℕ := 300

def ratio_alice : nat := 4
def ratio_carol : nat := 6
def total_parts := ratio_alice + ratio_carol
def work_per_part := total_work / total_parts
def work_carol := ratio_carol * work_per_part

theorem carol_paints_180_square_feet : work_carol = 180 := by
  sorry

end carol_paints_180_square_feet_l158_158501


namespace exists_distinct_complex_numbers_l158_158666

def z : ℕ → ℂ := sorry  -- z is a sequence of complex numbers

def condition1 (z : ℕ → ℂ) := (|z 1| = 1 ∧ |z 2| = 1)
def condition2 (z : ℕ → ℂ) := |z 2019| = Real.sqrt 1010
def condition3 (z : ℕ → ℂ) (i : ℕ) (h : i ∈ {k | k % 2 = 1 ∧ k ≤ 2017}) :=
  |2 * z (i+2) - (z (i+1) + z i)| = |z i - z (i+1)|
def condition4 (z : ℕ → ℂ) (i : ℕ) (h : i ∈ {k | k % 2 = 1 ∧ k ≤ 2017}) :=
  |2 * z (i+3) - (z (i+1) + z i)| = |z i - z (i+1)|

theorem exists_distinct_complex_numbers (z : ℕ → ℂ) :
  condition1 z ∧ 
  condition2 z ∧ 
  (∀ i : ℕ, i ∈ {k | k % 2 = 1 ∧ k ≤ 2017} → condition3 z i ∧ condition4 z i) → 
  ∃ z' : ℕ → ℂ, (z ≠ z' → false) :=
sorry

end exists_distinct_complex_numbers_l158_158666


namespace value_of_expression_l158_158437

theorem value_of_expression : 7^3 - 4 * 7^2 + 4 * 7 - 1 = 174 :=
by
  sorry

end value_of_expression_l158_158437


namespace arccos_one_over_sqrt_two_eq_pi_over_four_l158_158930

theorem arccos_one_over_sqrt_two_eq_pi_over_four : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_eq_pi_over_four_l158_158930


namespace infinitely_many_n_exist_l158_158550

-- Define the function f(p, n), denoting the largest integer k such that p^k divides n!
def f (p : ℕ) (n : ℕ) [Fact (Nat.Prime p)] : ℕ :=
  ∑ i in Nat.Icc 1 (Nat.log p n), n / p^i

-- Main theorem
theorem infinitely_many_n_exist (p : ℕ) (m : ℕ) (c : ℕ) [Fact (Nat.Prime p)] (hm : 0 < m) (hc : 0 < c) :
  ∃ᶠ n in at_top, (f p n) % m = c % m :=
sorry

end infinitely_many_n_exist_l158_158550


namespace complete_sets_of_real_numbers_l158_158695

def complete (A : set ℝ) : Prop :=
  A.nonempty ∧ ∀ a b : ℝ, (a + b ∈ A) → (a * b ∈ A)

theorem complete_sets_of_real_numbers (A : set ℝ) (h : complete A) : A = set.univ :=
sorry

end complete_sets_of_real_numbers_l158_158695


namespace seating_arrangements_for_conditions_l158_158072

-- Definition of the circular seating arrangement problem
def num_seating_arrangements (n : ℕ) (leader_index : Fin n) (deputy_leader_index : Fin n) (recorder_index : Fin n) : ℕ :=
  if n = 8 then
    let block := 3
    let remaining_people := n - block
    let base_permutations := (remaining_people - 1)!
    let valid_sequences := 2
    base_permutations * valid_sequences
  else
    0

-- The proof problem statement
theorem seating_arrangements_for_conditions :
  num_seating_arrangements 8 ⟨0, by decide⟩ ⟨1, by decide⟩ ⟨2, by decide⟩ = 240 :=
by {
  sorry
}

end seating_arrangements_for_conditions_l158_158072


namespace AF_less_BD_l158_158710

-- Define the geometric shapes and their properties.
def triangle_equilateral (A B E : Point) : Prop :=
  is_equilateral_triangle A B E

def rhombus_construction (B C D E : Point) : Prop :=
  is_rhombus B C D E

-- Define the intersection point of the segments AC and BD.
def intersect_at (A C B D F : Point) : Prop :=
  is_intersection_of_lines_on_planar A C B D F

-- Now write the Lean 4 statement to prove AF < BD under these conditions.

theorem AF_less_BD
  (A B C D E F : Point)
  (h_triangle : triangle_equilateral A B E)
  (h_rhombus : rhombus_construction B C D E)
  (h_intersect : intersect_at A C B D F) :
  length (segment A F) < length (segment B D) :=
sorry

end AF_less_BD_l158_158710


namespace mass_of_man_l158_158074

theorem mass_of_man (L B h ρ V m: ℝ) (boat_length: L = 3) (boat_breadth: B = 2) 
  (boat_sink_depth: h = 0.01) (water_density: ρ = 1000) 
  (displaced_volume: V = L * B * h) (displaced_mass: m = ρ * V): m = 60 := 
by 
  sorry

end mass_of_man_l158_158074


namespace x1_x2_square_less_than_two_l158_158596

def f (a x : ℝ) := Real.log x - a * x

theorem x1_x2_square_less_than_two (x1 x2 m : ℝ) (ha : 1 = 1) (h1 : x1 < x2) (h2 : ∀ x > 0, f 1 x = m → x = x1 ∨ x = x2) (hm : m < -2) : 
x1 * x2^2 < 2 := 
by
  sorry

end x1_x2_square_less_than_two_l158_158596


namespace milk_production_l158_158743

variable (a b c d e : ℝ)

theorem milk_production (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (he : 0 < e) :
  let rate_per_cow_per_day := b / (a * c)
  let production_per_day := d * rate_per_cow_per_day
  let total_production := production_per_day * e
  total_production = (b * d * e) / (a * c) :=
by
  sorry

end milk_production_l158_158743


namespace milk_production_l158_158744

variable (a b c d e : ℝ)

theorem milk_production (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (he : 0 < e) :
  let rate_per_cow_per_day := b / (a * c)
  let production_per_day := d * rate_per_cow_per_day
  let total_production := production_per_day * e
  total_production = (b * d * e) / (a * c) :=
by
  sorry

end milk_production_l158_158744


namespace ellipse_equation_line_intersects_ellipse_existence_of_point_E_l158_158196

-- Problem part (1)
theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
    (e : ℝ) (he : e = sqrt 3 / 2) (h3 : (1 / 2) * a * b = 1) :
    (1 : ℝ) = (a : ℝ)^2 - (b : ℝ)^2 - (sqrt 3 : ℝ)^2 :=
  sorry

-- Problem part (2)
theorem line_intersects_ellipse (h : P ≠ Q) (h1 : OP⊥OQ)
  : (P : ℝ × ℝ) × (Q : ℝ × ℝ) × (l : ℝ × ℝ → Prop) = (line_equation : 2 * x - y = ± 2) :=
  sorry

-- Problem part (3)
theorem existence_of_point_E (E : ℝ × ℝ)
  (h1 : E = (m, 0)) 
  (h2 : ∀ M N : ℝ × ℝ, line_through E M N → intersects_ellipse M N) :
  ∃ (ℝ : ℝ), (E = (± (2 * sqrt 15 / 5), 0)) ∧ (const_value = 5) :=
  sorry

end ellipse_equation_line_intersects_ellipse_existence_of_point_E_l158_158196


namespace at_least_two_same_books_l158_158156

def sum_of_digits (n : Nat) : Nat :=
  n.digits 10 |>.sum

def satisfied (n : Nat) : Prop :=
  n / sum_of_digits n = 13

theorem at_least_two_same_books (n1 n2 n3 n4 : Nat) (h1 : satisfied n1) (h2 : satisfied n2) (h3 : satisfied n3) (h4 : satisfied n4) :
  n1 = n2 ∨ n1 = n3 ∨ n1 = n4 ∨ n2 = n3 ∨ n2 = n4 ∨ n3 = n4 :=
sorry

end at_least_two_same_books_l158_158156


namespace inclination_angle_of_line_l158_158391

theorem inclination_angle_of_line : 
  ∃ θ ∈ set.Ico 0 180, tan (θ * (π / 180)) = -1 ∧ θ = 135 := 
by {
  use 135,
  split,
  { show 135 ∈ set.Ico 0 180, from ⟨le_refl _, by norm_num⟩ },
  split,
  { show tan (135 * (π / 180)) = -1, from by simp [real.tan_pi_div_four, mul_div_cancel_left' (by norm_num : 0 < 2)] },
  { show 135 = 135, from eq.refl 135 }
}

end inclination_angle_of_line_l158_158391


namespace probability_enemy_plane_hit_l158_158824

noncomputable def P_A : ℝ := 0.6
noncomputable def P_B : ℝ := 0.4

theorem probability_enemy_plane_hit : 1 - ((1 - P_A) * (1 - P_B)) = 0.76 :=
by
  sorry

end probability_enemy_plane_hit_l158_158824


namespace arccos_one_over_sqrt_two_eq_pi_over_four_l158_158927

theorem arccos_one_over_sqrt_two_eq_pi_over_four : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_eq_pi_over_four_l158_158927


namespace function_properties_l158_158449

noncomputable def f (x : Real) : Real :=
  sin x + sin (3 * x) / 3 + sin (5 * x) / 5 + sin (7 * x) / 7

theorem function_properties :
  (∀ x, f (π + x) = - f x) ∧
  (∀ x, f (-x) = - f x) ∧
  (∀ x, f' x ≤ 4) :=
by
  -- Proofs required for each part statements
  sorry

end function_properties_l158_158449


namespace log_expression_equals_eight_l158_158435

theorem log_expression_equals_eight :
  (Real.log 4 / Real.log 10) + 
  2 * (Real.log 5 / Real.log 10) + 
  3 * (Real.log 2 / Real.log 10) + 
  6 * (Real.log 5 / Real.log 10) + 
  (Real.log 8 / Real.log 10) = 8 := 
by 
  sorry

end log_expression_equals_eight_l158_158435


namespace small_leg_radius_ratio_l158_158849

theorem small_leg_radius_ratio 
  (ABC : Triangle)
  (angle_BAC : ABC.angle BAC.vertices = 60)
  (angle_ABC : ABC.angle ABC.vertices = 30)
  (hypotenuse_2 : ABC.hypotenuse = 2)
  (inscribed_circles : ∀ (r : ℝ), InscribedCirclesInTriangle ABC r)
  :
  let
    (AC, BC) = ABC.sides
  in
  AC / (radius ABC) = 2 + sqrt 3
:= sorry

end small_leg_radius_ratio_l158_158849


namespace width_increase_to_maintain_area_l158_158392

noncomputable def percentage_increase (W : ℝ) : ℝ :=
  let W1 := 1.20 * W in
  let W2 := W / 0.765 in
  ((W2 - W1) / W1) * 100

theorem width_increase_to_maintain_area 
  (L W : ℝ) 
  (hL : L > 0) 
  (hW : W > 0)
  : percentage_increase W ≈ 8.933 :=
by
  sorry

end width_increase_to_maintain_area_l158_158392


namespace cat_arrangements_l158_158310

theorem cat_arrangements :
  let cages := 5
  let golden_tabby_units := 1
  let silver_tabby_pairs := 2
  let ragdoll_units := 1
  let units := golden_tabby_units + silver_tabby_pairs + ragdoll_units
  let silver_pairs_arrangements := 3
  let total_arrangements := Nat.fact 5 / Nat.fact (5 - units) * silver_pairs_arrangements
  total_arrangements = 360 :=
by
  sorry

end cat_arrangements_l158_158310


namespace missing_match_l158_158494

def player := String

structure match :=
  (winner : player)
  (loser : player)

def matches : Set match :=
  { ⟨"Bella", "Ann"⟩, 
    ⟨"Celine", "Donna"⟩, 
    ⟨"Gina", "Holly"⟩, 
    ⟨"Gina", "Celine"⟩, 
    ⟨"Celine", "Bella"⟩, 
    ⟨"Emma", "Farah"⟩ }

theorem missing_match : match ∈ matches → match := sorry

end missing_match_l158_158494


namespace Pythagorean_triple_example_1_Pythagorean_triple_example_2_l158_158452

theorem Pythagorean_triple_example_1 : 3^2 + 4^2 = 5^2 := by
  sorry

theorem Pythagorean_triple_example_2 : 5^2 + 12^2 = 13^2 := by
  sorry

end Pythagorean_triple_example_1_Pythagorean_triple_example_2_l158_158452


namespace derivatives_when_neg_l158_158192

variables {R : Type*} [linear_ordered_field R] {f g : R → R}

-- Condition definitions
def is_odd (f : R → R) := ∀ x, f (-x) = -f x
def is_even (g : R → R) := ∀ x, g (-x) = g x
def increasing_on_pos (f' : R → R) := ∀ x, 0 < x → 0 < f' x
def g_increasing_on_pos (g' : R → R) := ∀ x, 0 < x → 0 < g' x

theorem derivatives_when_neg
  (h₁ : is_odd f)
  (h₂ : is_even g)
  (h₃ : increasing_on_pos (deriv f))
  (h₄ : g_increasing_on_pos (deriv g)) :
  (∀ x, x < 0 → 0 < deriv f x) ∧ (∀ x, x < 0 → deriv g x < 0) := 
sorry

end derivatives_when_neg_l158_158192


namespace shirts_left_l158_158856

theorem shirts_left (initial_shirts : ℕ) (sold_shirts : ℕ) : 
  initial_shirts = 49 → sold_shirts = 21 → initial_shirts - sold_shirts = 28 := 
by 
  intros h1 h2
  rw [h1, h2]
  exact rfl

end shirts_left_l158_158856


namespace same_color_points_distance_2004_l158_158022

noncomputable def exists_same_color_points_at_distance_2004 (color : ℝ × ℝ → ℕ) : Prop :=
  ∃ (p q : ℝ × ℝ), (p ≠ q) ∧ (color p = color q) ∧ (dist p q = 2004)

/-- The plane is colored in two colors. Prove that there exist two points of the same color at a distance of 2004 meters. -/
theorem same_color_points_distance_2004 {color : ℝ × ℝ → ℕ}
  (hcolor : ∀ p, color p = 1 ∨ color p = 2) :
  exists_same_color_points_at_distance_2004 color :=
sorry

end same_color_points_distance_2004_l158_158022


namespace sum_of_all_possible_y_l158_158797

noncomputable def list := [12, 3, 6, 3, 7, 3, y : ℝ]
def mean := (34 + y) / 7
def mode := 3
def median := 
  if y ≤ 3 then 3 
  else if 3 < y ∧ y < 6 then y 
  else 6

lemma arithmetic_progression_sum :
  (insert mean (insert mode (insert median {}))).to_finset.sum = 14 :=
sorry

theorem sum_of_all_possible_y :
  let y_values := {15, 43} in
  (y_values.sum : ℝ) = 58 :=
sorry

end sum_of_all_possible_y_l158_158797


namespace arccos_one_over_sqrt_two_eq_pi_four_l158_158909

theorem arccos_one_over_sqrt_two_eq_pi_four : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 := 
by
  sorry

end arccos_one_over_sqrt_two_eq_pi_four_l158_158909


namespace find_abc_l158_158343

theorem find_abc : 
  ∃ (a b c : ℕ), 
    let x := Real.sqrt ((Real.sqrt 73) / 2 + 5 / 2) in
    (x^60 = 3 * x^58 + 10 * x^56 + 12 * x^54 - x^30 + a * x^26 + b * x^24 + c * x^20) ∧ 
    (a + b + c = 359) :=
by
  existsi (29 : ℕ)
  existsi (150 : ℕ)
  existsi (180 : ℕ)
  have x_def : x = Real.sqrt ((Real.sqrt 73) / 2 + 5 / 2) := by sorry
  have power_relation : 
    ∀ k x, x^60 = 3 * x^58 + 10 * x^56 + 12 * x^54 - x^30 + 29 * x^26 + 150 * x^24 + 180 * x^20 :=
    by sorry
  use x_def
  use power_relation
  trivial

end find_abc_l158_158343


namespace group_8_extracted_number_is_72_l158_158821

-- Definitions related to the problem setup
def individ_to_group (n : ℕ) : ℕ := n / 10 + 1
def unit_digit (n : ℕ) : ℕ := n % 10
def extraction_rule (k m : ℕ) : ℕ := (k + m - 1) % 10

-- Given condition: total individuals split into sequential groups and m = 5
def total_individuals : ℕ := 100
def total_groups : ℕ := 10
def m : ℕ := 5
def k_8 : ℕ := 8

-- The final theorem statement
theorem group_8_extracted_number_is_72 : ∃ n : ℕ, individ_to_group n = k_8 ∧ unit_digit n = extraction_rule k_8 m := by
  sorry

end group_8_extracted_number_is_72_l158_158821


namespace part1_part2_l158_158649

noncomputable def sin (x : ℝ) : ℝ := sorry
noncomputable def cos (x : ℝ) : ℝ := sorry
def pi : ℝ := Real.pi

namespace Triangle

variables {A B C a b c : ℝ}

/-- Given conditions of the acute triangle and equation relating sines and sides --/
def given_equation := (sin A - sin B) / ((sqrt 3) * a - c) = sin C / (a + b)

-- Angle B in an acute triangle meeting the given conditions -/
theorem part1 (h₁ : given_equation) (h₂ : 0 < B ∧ B < pi / 2) :
  B = pi / 6 :=
sorry

-- Range of possible values for the perimeter when a = 2 -/
theorem part2 (h₁ : given_equation) (h₂ : a = 2) :
  3 + sqrt 3 < a + b + c ∧ a + b + c < 2 + 2 * sqrt 3 :=
sorry

end Triangle

end part1_part2_l158_158649


namespace number_of_surjections_l158_158235

def is_surjection {A B : Type} (f : A → B) :=
  ∀ b : B, ∃ a : A, f a = b

theorem number_of_surjections (A B : Type) [Fintype A] [Fintype B] (hA : Fintype.card A = 3) (hB : Fintype.card B = 2) :
  Fintype.card { f : A → B // is_surjection f } = 6 := 
by
  sorry

end number_of_surjections_l158_158235


namespace paint_cans_needed_l158_158515

-- Conditions as definitions
def bedrooms : ℕ := 3
def other_rooms : ℕ := 2 * bedrooms
def paint_per_room : ℕ := 2
def color_can_capacity : ℕ := 1
def white_can_capacity : ℕ := 3

-- Total gallons needed
def total_color_gallons_needed : ℕ := paint_per_room * bedrooms
def total_white_gallons_needed : ℕ := paint_per_room * other_rooms

-- Total cans needed
def total_color_cans_needed : ℕ := total_color_gallons_needed / color_can_capacity
def total_white_cans_needed : ℕ := total_white_gallons_needed / white_can_capacity
def total_cans_needed : ℕ := total_color_cans_needed + total_white_cans_needed

theorem paint_cans_needed : total_cans_needed = 10 := by
  -- Proof steps (skipped) to show total_cans_needed = 10
  sorry

end paint_cans_needed_l158_158515


namespace probability_sum_even_l158_158278

-- Let's define the set of the first ten prime numbers.
def first_ten_primes : Finset ℕ := Finset.of_list [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define the proposition to prove:
theorem probability_sum_even :
  let n := first_ten_primes.card in 
  let total_pairs := Finset.card (Finset.powersetLen 2 first_ten_primes) in 
  let even_pairs := Finset.card (Finset.filter (λ s, s.sum % 2 = 0) (Finset.powersetLen 2 first_ten_primes)) in 
  (total_pairs = 45 ∧ even_pairs = 36) →
  (even_pairs / total_pairs = 4 / 5) :=
by
  sorry

end probability_sum_even_l158_158278


namespace problem_statement_l158_158688

theorem problem_statement
  (y : Fin 50 → ℝ)
  (h1 : ∑ i, y i = 2)
  (h2 : ∑ i, y i / (1 - y i) = 2) :
  ∑ i, y i ^ 2 / (1 - y i) = 0 :=
by
  sorry

end problem_statement_l158_158688


namespace sufficient_but_not_necessary_condition_l158_158816

-- Define the predicate for a line intersecting with a circle
def line_intersects_circle (k : ℝ) : Prop :=
  let d := (|k|) / Real.sqrt 2 in
  d < 1

-- State the theorem for the proof problem
theorem sufficient_but_not_necessary_condition :
  (line_intersects_circle 1) ∧ ¬ ∀ k, (line_intersects_circle k) → k = 1 :=
by
  sorry -- Proof to be provided

end sufficient_but_not_necessary_condition_l158_158816


namespace negation_of_universal_proposition_l158_158761

theorem negation_of_universal_proposition :
  ¬(∀ x ∈ (Ioo 0 1), x^2 - x < 0) ↔ ∃ x0 ∈ (Ioo 0 1), x0^2 - x0 ≥ 0 :=
by
  sorry

end negation_of_universal_proposition_l158_158761


namespace circle_equation_l158_158477

theorem circle_equation
  (a b r : ℝ)
  (ha : (4 - a)^2 + (1 - b)^2 = r^2)
  (hb : (2 - a)^2 + (1 - b)^2 = r^2)
  (ht : (b - 1) / (a - 2) = -1) :
  (a = 3) ∧ (b = 0) ∧ (r = 2) :=
by {
  sorry
}

-- Given the above values for a, b, r
def circle_equation_verified : Prop :=
  (∀ (x y : ℝ), ((x - 3)^2 + y^2) = 4)

example : circle_equation_verified :=
by {
  sorry
}

end circle_equation_l158_158477


namespace root_conjugate_l158_158346

variables (f : ℂ → ℂ) (a b : ℝ) 

-- Assumptions
def is_polynomial_with_real_coeff (f : ℂ → ℂ) : Prop :=
∀ z : ℂ, f(z) = ∑ i in finset.range (degree(f) + 1), (coeff(f, i) * z^i)

def real_parts (f : ℂ → ℂ) : Prop := 
∀ n : ℕ, coeff(f, n) = conj(coeff(f, n))

-- Statement
theorem root_conjugate (hf : is_polynomial_with_real_coeff f) (ha : f(a + b * complex.I) = 0) : 
  f(a - b * complex.I) = 0 :=
sorry

end root_conjugate_l158_158346


namespace range_of_a_l158_158624

noncomputable section

open Real

def f (x a : ℝ) : ℝ := (1 / 3) * x^3 - (a + 1 / 2) * x^2 + (a^2 + a) * x - (1 / 2) * a^2 + 1 / 2

theorem range_of_a (a : ℝ) (h : ∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ f x a = 0 ∧ f y a = 0 ∧ f z a = 0) :
  -7 / 2 < a ∧ a < -1 :=
by
  sorry

end range_of_a_l158_158624


namespace arccos_one_over_sqrt_two_eq_pi_four_l158_158911

theorem arccos_one_over_sqrt_two_eq_pi_four : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 := 
by
  sorry

end arccos_one_over_sqrt_two_eq_pi_four_l158_158911


namespace non_intersecting_matching_l158_158303

-- Introduce the necessary variables and definitions
variables {Point : Type} [plane : EuclideanPlane Point]

-- Define the problem in Lean 4
noncomputable def non_intersecting_segments (n : ℕ) (red_points blue_points : fin n → Point) : Prop :=
  ∀ (p1 p2 : Point) (r1 r2 : Point), 
  set_of (λ (i : nat), i < n ∧ (p1 = red_points i ∨ p1 = blue_points i) ∧ (p2 = red_points i ∨ p2 = blue_points i)) = 
  ∅ → ¬ (line_segment p1 p2 ∩ line_segment r1 r2 ≠ ∅)

theorem non_intersecting_matching {n : ℕ} 
  (red_points blue_points : fin n → Point) 
  (no_three_collinear : ∀ (p1 p2 p3 : Point), p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 → 
                       ¬collinear {p1, p2, p3}) :
  ∃ (pairs : fin n → (Point × Point)), 
    (∀ (i : fin n), (pairs i).fst ∈ (set_of (red_points i)) ∧ (pairs i).snd ∈ (set_of (blue_points i))) ∧
    non_intersecting_segments n red_points blue_points :=
sorry

end non_intersecting_matching_l158_158303


namespace projection_of_vector_l158_158146

open Matrix

def projection (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let dot_product := u.1 * v.1 + u.2 * v.2 + u.3 * v.3
  let magnitude_squared := v.1 * v.1 + v.2 * v.2 + v.3 * v.3
  let scalar := dot_product / magnitude_squared
  (scalar * v.1, scalar * v.2, scalar * v.3)

theorem projection_of_vector :
  projection (3, 5, -2) (1, 3, -2) = (11 / 7, 33 / 7, -22 / 7) :=
by
  sorry

end projection_of_vector_l158_158146


namespace probability_sum_even_l158_158280

-- Let's define the set of the first ten prime numbers.
def first_ten_primes : Finset ℕ := Finset.of_list [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define the proposition to prove:
theorem probability_sum_even :
  let n := first_ten_primes.card in 
  let total_pairs := Finset.card (Finset.powersetLen 2 first_ten_primes) in 
  let even_pairs := Finset.card (Finset.filter (λ s, s.sum % 2 = 0) (Finset.powersetLen 2 first_ten_primes)) in 
  (total_pairs = 45 ∧ even_pairs = 36) →
  (even_pairs / total_pairs = 4 / 5) :=
by
  sorry

end probability_sum_even_l158_158280


namespace product_modulo_23_l158_158791

theorem product_modulo_23 :
  (3001 * 3002 * 3003 * 3004 * 3005) % 23 = 0 :=
by {
  have h1 : 3001 % 23 = 19 := rfl,
  have h2 : 3002 % 23 = 20 := rfl,
  have h3 : 3003 % 23 = 21 := rfl,
  have h4 : 3004 % 23 = 22 := rfl,
  have h5 : 3005 % 23 = 0 := rfl,
  sorry
}

end product_modulo_23_l158_158791


namespace part1_part2_l158_158203

noncomputable def f (a x : ℝ) : ℝ := (a * 2^x + a - 2) / (2^x + 1)

-- Statement 1: If f(x) is an odd function, then a = 1.
theorem part1 (a : ℝ) : (∀ x : ℝ, f a (-x) = - f a x) → a = 1 :=
sorry

-- Statement 2: If f(x) is defined on [-4, +∞), and for all x in the domain, 
-- f(cos(x) + b + 1/4) ≥ f(sin^2(x) - b - 3), then b ∈ [-1,1].
theorem part2 (a : ℝ) (b : ℝ) :
  (∀ x : ℝ, f a (Real.cos x + b + 1/4) ≥ f a (Real.sin x ^ 2 - b - 3)) ∧
  (∀ x : ℝ, -4 ≤ x) ∧ -4 ≤ a ∧ a = 1 → -1 ≤ b ∧ b ≤ 1 :=
sorry

end part1_part2_l158_158203


namespace z_location_l158_158193

noncomputable def z_exists : ℂ := (1 + 7 * complex.I) / (-1 + 3 * complex.I)

theorem z_location :
  let z : ℂ := z_exists in
  let coordinates := (z.re, z.im) in
  z * (-1 + 3 * complex.I) = 1 + 7 * complex.I →
  0 < z.re ∧ z.im < 0 :=
sorry

end z_location_l158_158193


namespace arccos_proof_l158_158955

noncomputable def arccos_identity : Prop := 
  ∃ x : ℝ, x = 1 / Real.sqrt 2 ∧ Real.arccos x = Real.pi / 4

theorem arccos_proof : arccos_identity :=
by
  sorry

end arccos_proof_l158_158955


namespace sequence_property_l158_158182

open Nat

theorem sequence_property (a : ℕ → ℕ)
  (h_pos : ∀ n, a n > 0)
  (h_gcd : ∀ i j, i ≠ j → gcd (a i) (a j) = gcd i j) :
  ∀ n, a n = n := 
by 
  sorry

end sequence_property_l158_158182


namespace arccos_proof_l158_158953

noncomputable def arccos_identity : Prop := 
  ∃ x : ℝ, x = 1 / Real.sqrt 2 ∧ Real.arccos x = Real.pi / 4

theorem arccos_proof : arccos_identity :=
by
  sorry

end arccos_proof_l158_158953


namespace rectangle_area_k_l158_158767

theorem rectangle_area_k (d : ℝ) (x : ℝ) (h_ratio : 5 * x > 0 ∧ 2 * x > 0) (h_diagonal : d^2 = (5 * x)^2 + (2 * x)^2) :
  ∃ k : ℝ, (∃ (h : k = 10 / 29), (5 * x) * (2 * x) = k * d^2) := by
  use 10 / 29
  sorry

end rectangle_area_k_l158_158767


namespace prove_a_eq_b_l158_158602

theorem prove_a_eq_b 
  (p q a b : ℝ) 
  (h1 : p + q = 1) 
  (h2 : p * q ≠ 0) 
  (h3 : p / a + q / b = 1 / (p * a + q * b)) : 
  a = b := 
sorry

end prove_a_eq_b_l158_158602


namespace tangent_line_correct_l158_158543

noncomputable def f (x : ℝ) : ℝ := exp (-5 * x) + 2
def point : ℝ × ℝ := (0, 3)
def tangent_line (x : ℝ) : ℝ := -5 * x + 3

theorem tangent_line_correct : 
    ∀ x y, (y = f x) → x = 0 → y = 3 → (∀ t, tangent_line t = -5 * t + 3) := 
by
  sorry

end tangent_line_correct_l158_158543


namespace original_price_l158_158083

-- Definitions based on the conditions
def selling_price : ℝ := 1080
def gain_percent : ℝ := 80

-- The proof problem: Prove that the cost price is Rs. 600
theorem original_price (CP : ℝ) (h_sp : CP + CP * (gain_percent / 100) = selling_price) : CP = 600 :=
by
  -- We skip the proof itself
  sorry

end original_price_l158_158083


namespace tangent_intersection_at_single_point_l158_158709

theorem tangent_intersection_at_single_point
  (A B C K1 K2 : Point)
  (hK1_on_BC : on_segment B C K1)
  (hK2_on_BC : on_segment B C K2)
  (tangent_ABK1_ACK2 : is_external_tangent (incircle (triangle A B K1)) (incircle (triangle A C K2)))
  (tangent_ABK2_ACK1 : is_external_tangent (incircle (triangle A B K2)) (incircle (triangle A C K1))) :
  ∃ O : Point, 
    (is_tangent_from O (incircle (triangle A B K1))) ∧ 
    (is_tangent_from O (incircle (triangle A C K2))) ∧ 
    (is_tangent_from O (incircle (triangle A B K2))) ∧ 
    (is_tangent_from O (incircle (triangle A C K1))) := 
sorry

end tangent_intersection_at_single_point_l158_158709


namespace no_consecutive_factorials_with_first_digits_1_to_9_l158_158365

theorem no_consecutive_factorials_with_first_digits_1_to_9 (N : ℕ) : 
  ¬ (∀ i : ℕ, 1 ≤ i ∧ i ≤ 9 →
      (∃ k : ℤ, i * 10^k ≤ (N + i)! ∧ (N + i)! < (i + 1) * 10^k)) :=
by 
  sorry

end no_consecutive_factorials_with_first_digits_1_to_9_l158_158365


namespace number_of_proper_subsets_l158_158394

theorem number_of_proper_subsets (A : Set ℕ) (h : A = {x | 0 < x ∧ x < 4}) :
  ∃ n, n = 7 ∧ ∃ B, B ⊆ A ∧ B ≠ A ∧ B ≠ ∅ :=
by
  use 7,
  sorry

end number_of_proper_subsets_l158_158394


namespace num_nat_satisfy_eq_l158_158333

open Nat

def floor (x : ℝ) : ℤ := ⌊x⌋

theorem num_nat_satisfy_eq :
  ∃! x : ℕ, floor (3.8 * x) = floor 3.8 * x + 1 :=
by
  sorry

end num_nat_satisfy_eq_l158_158333


namespace probability_even_sum_l158_158259

def first_ten_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def is_odd (n : ℕ) : Prop := n % 2 = 1

def count_odd_primes (primes : List ℕ) : ℕ :=
  primes.countp is_odd

def binom (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

open Nat

theorem probability_even_sum :
  let primes := first_ten_primes in
  let odd_primes := count_odd_primes primes in
  let num_ways_even_sum := binom odd_primes 2 in
  let total_ways := binom primes.length 2 in
  (num_ways_even_sum : ℚ) / (total_ways : ℚ) = 4 / 5 :=
by
  sorry

end probability_even_sum_l158_158259


namespace trajectory_of_center_of_moving_circle_l158_158015

noncomputable def center_trajectory (x y : ℝ) : Prop :=
  0 < y ∧ y ≤ 1 ∧ x^2 = 4 * (y - 1)

theorem trajectory_of_center_of_moving_circle (x y : ℝ) :
  0 ≤ y ∧ y ≤ 2 ∧ x^2 + y^2 = 4 ∧ 0 < y → center_trajectory x y :=
by
  sorry

end trajectory_of_center_of_moving_circle_l158_158015


namespace evaluate_expression_l158_158134

theorem evaluate_expression : 
  (Int.floor (Real.ceil ((15 / 8 : ℝ)^2) + 19 / 5)) = 7 := 
sorry

end evaluate_expression_l158_158134


namespace proof_problem_l158_158064

variables {n : ℕ} (a : Fin n → ℝ) (hpos : ∀ i, 0 < a i)
noncomputable def S := ∑ i, a i

theorem proof_problem :
  ∑ i in Finset.range (n - 1), 
    Real.sqrt (a i ^ 2 + a i * a (i + 1) + a (i + 1) ^ 2) 
  ≥ Real.sqrt ((S a - a 0) ^ 2 + (S a - a 0) * (S a - a (n - 1)) + (S a - a (n - 1)) ^ 2) :=
sorry

end proof_problem_l158_158064


namespace shooting_match_sequences_l158_158647

theorem shooting_match_sequences :
  let targets := [1, 2, 3, 4]
  let columns := (targets, targets)
  let sequences := { seq : List (Sum (List ℕ) (List ℕ)) // seq.length = 8 }
  ∃ (seq : sequences), seq.1.count (Sum.inl targets) = 4 ∧ seq.1.count (Sum.inr targets) = 4 → sequences.cardinality = 70 :=
sorry

end shooting_match_sequences_l158_158647


namespace probability_sum_even_l158_158293

def primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def sum_is_even (a b : ℕ) : Prop :=
  (a + b) % 2 = 0

theorem probability_sum_even :
  (nat.choose 10 2) = 45 →
  (∀ a b : ℕ, a ∈ primes → b ∈ primes → a ≠ b →
    (sum_is_even a b ↔ a ≠ 2 ∧ b ≠ 2)) →
  ((45 - 9) / 45 : ℚ) = 4 / 5 :=
by sorry

end probability_sum_even_l158_158293


namespace total_number_of_stickers_l158_158713

def sticker_count (sheets : ℕ) (stickers_per_sheet : ℕ) : ℕ := sheets * stickers_per_sheet

theorem total_number_of_stickers 
    (sheets_per_folder : ℕ)
    (red_folder_stickers_per_sheet : ℕ)
    (green_folder_stickers_per_sheet : ℕ)
    (blue_folder_stickers_per_sheet : ℕ) :
    sticker_count sheets_per_folder red_folder_stickers_per_sheet +
    sticker_count sheets_per_folder green_folder_stickers_per_sheet +
    sticker_count sheets_per_folder blue_folder_stickers_per_sheet = 60 := 
begin
    -- Given conditions
    let sheets := 10      -- Each folder contains 10 sheets of paper.
    let red := 3          -- Each sheet in the red folder gets 3 stickers.
    let green := 2        -- Each sheet in the green folder gets 2 stickers.
    let blue := 1         -- Each sheet in the blue folder gets 1 sticker.
    have h1 : sticker_count sheets red = 30, by sorry, -- Calculation omitted
    have h2 : sticker_count sheets green = 20, by sorry, -- Calculation omitted
    have h3 : sticker_count sheets blue = 10, by sorry, -- Calculation omitted

    -- Summing the stickers
    show h1 + h2 + h3 = 60, by sorry
end

end total_number_of_stickers_l158_158713


namespace measure_of_angle_G_l158_158652

-- Define the properties of the parallelogram and the given angle measurement.
variables (EFGH : Type) [parallelogram EFGH]
variable (angle_E : ℝ) (hE : angle_E = 125)

-- Define the angles in the parallelogram.
def angle_G (EFGH : Type) [parallelogram EFGH] : ℝ := angle_E

theorem measure_of_angle_G : angle_G EFGH = 125 := by
  -- Add the proof here
  sorry

end measure_of_angle_G_l158_158652


namespace domain_of_inverse_l158_158012

noncomputable def f (x : ℝ) : ℝ := 1 / (x + 1)

theorem domain_of_inverse :
  let inv_domain := {y : ℝ | y ≠ 0} in
  inv_domain = (Set.Iio 0 ∪ Set.Ioi 0) :=
by
  sorry

end domain_of_inverse_l158_158012


namespace proof_equivalent_problem_l158_158531

theorem proof_equivalent_problem (m n a b : ℝ) (h1 : m ≠ n) (h2 : m ≠ 0) (h3 : n ≠ 0) :
  ∃ x : ℝ, x = a * m + b * n ∧ (x + m)^3 - (x + n)^3 = (m - n)^3 ↔ (a ∧ b depend on m ∧ n) :=
by
  sorry

end proof_equivalent_problem_l158_158531


namespace eval_expression_l158_158137

theorem eval_expression (a : ℕ) (h : a = 2) : 
  8^3 + 4 * a * 8^2 + 6 * a^2 * 8 + a^3 = 1224 := 
by
  rw [h]
  sorry

end eval_expression_l158_158137


namespace sequence_integers_l158_158398

theorem sequence_integers :
  (∀ (n : ℕ), 1 ≤ n ∧ n ≤ 2000 →
    ∃ (a : ℕ), a = (list.sum (list.map (λ i, i) (list.range n)) * list.sum (list.map (λ i, i) (list.range n)))
  ) →
  (∀ (n : ℕ), 1 ≤ n ∧ n ≤ 2000 → (∃ (a : ℕ), a ∈ ℤ)) := 
sorry

end sequence_integers_l158_158398


namespace prove_PM_PN_distance_l158_158664

variable (ρ θ : ℝ)
def curve_C := ρ^2 = 2*ρ*cos θ - 4*ρ*sin θ + 4
def line_l1 := ρ*(cos θ - sin θ) = 3

variable (x y t t1 t2 t3 α : ℝ)
def rect_curve_C := (x - 1)^2 + (y + 2)^2 = 9
def rect_line_l1 := x - y - 3 = 0

def line_l2 (t : ℝ) := (x = -1 + t * cos α) ∧ (y = t * sin α)
def param_eq_curve_C (t : ℝ) := t^2 - 4*(cos α - sin α)*t - 1 = 0
def t1_t2_sum := t1 + t2 = 4*(cos α - sin α)

def PM_distance := |(t1 + t2) / 2|
def PN_distance := |4 / (cos α - sin α)|

theorem prove_PM_PN_distance :
  PM_distance * PN_distance = 8 :=
by
  sorry

end prove_PM_PN_distance_l158_158664


namespace additional_cost_l158_158119

theorem additional_cost (original_bales : ℕ) (previous_cost_per_bale : ℕ) (better_quality_cost_per_bale : ℕ) :
  let new_bales := original_bales * 2
  let original_cost := original_bales * previous_cost_per_bale
  let new_cost := new_bales * better_quality_cost_per_bale
  new_cost - original_cost = 210 :=
by sorry

end additional_cost_l158_158119


namespace find_monic_polynomial_with_scaled_roots_l158_158687

noncomputable def polynomial_with_scaled_roots (p1 p2 p3 : ℝ) (h : (polynomial.C 1 * polynomial.X^3 - 4 * polynomial.X^2 + 5 * polynomial.X - 3).is_root p1 ∧ 
                                                     (polynomial.C 1 * polynomial.X^3 - 4 * polynomial.X^2 + 5 * polynomial.X - 3).is_root p2 ∧ 
                                                     (polynomial.C 1 * polynomial.X^3 - 4 * polynomial.X^2 + 5 * polynomial.X - 3).is_root p3) 
                                                    : polynomial ℝ :=
  polynomial.C 1 * polynomial.X^3 - 12 * polynomial.X^2 + 45 * polynomial.X - 81

theorem find_monic_polynomial_with_scaled_roots (p1 p2 p3 : ℝ)  (h : (polynomial.C 1 * polynomial.X^3 - 4 * polynomial.X^2 + 5 * polynomial.X - 3).is_root p1 ∧ 
                                                    (polynomial.C 1 * polynomial.X^3 - 4 * polynomial.X^2 + 5 * polynomial.X - 3).is_root p2 ∧ 
                                                    (polynomial.C 1 * polynomial.X^3 - 4 * polynomial.X^2 + 5 * polynomial.X - 3).is_root p3) :
  polynomial_with_scaled_roots p1 p2 p3 h = polynomial.C 1 * polynomial.X^3 - 12 * polynomial.X^2 + 45 * polynomial.X - 81 :=
sorry

end find_monic_polynomial_with_scaled_roots_l158_158687


namespace probability_sum_even_l158_158290

def primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def sum_is_even (a b : ℕ) : Prop :=
  (a + b) % 2 = 0

theorem probability_sum_even :
  (nat.choose 10 2) = 45 →
  (∀ a b : ℕ, a ∈ primes → b ∈ primes → a ≠ b →
    (sum_is_even a b ↔ a ≠ 2 ∧ b ≠ 2)) →
  ((45 - 9) / 45 : ℚ) = 4 / 5 :=
by sorry

end probability_sum_even_l158_158290


namespace team_A_more_points_than_team_B_l158_158538

theorem team_A_more_points_than_team_B :
  let number_of_teams := 8
  let number_of_remaining_games := 6
  let win_probability_each_game := (1 : ℚ) / 2
  let team_A_beats_team_B_initial : Prop := True -- Corresponding to the condition team A wins the first game
  let probability_A_wins := 1087 / 2048
  team_A_beats_team_B_initial → win_probability_each_game = 1 / 2 → number_of_teams = 8 → 
    let A_more_points_than_B := team_A_beats_team_B_initial ∧ win_probability_each_game ^ number_of_remaining_games = probability_A_wins
    A_more_points_than_B :=
  sorry

end team_A_more_points_than_team_B_l158_158538


namespace eight_triangles_exists_l158_158717

theorem eight_triangles_exists (A B C D E : Point)
    (h_ab : collinear A B C) (h_de : collinear D E C)
    (h_int : ¬collinear A B D) (h_int2 : ¬collinear B C E) (h_int3 : ¬collinear A D E) 
    (h_int4 : ¬collinear A C E) (h_int5 : ¬collinear B D E) 
    (h_int6 : ¬collinear C D A) (h_int7 : ¬collinear C E B) : 
  ∃ (triangles : set (triangle Point)), triangles = 
    { {A, B, E}, {A, B, D}, {B, C, D}, {B, C, E}, {A, D, E}, {A, C, D}, {A, C, E}, {B, D, E} } 
    ∧ triangles.size = 8 := 
by
  sorry

end eight_triangles_exists_l158_158717


namespace election_winner_votes_diff_l158_158651

theorem election_winner_votes_diff
    (V : ℕ)  -- Total number of votes
    (P : ℕ)  -- Votes received by the winning candidate
    (H1 : 0.70 * V = P)  -- The winner received 70% of the votes
    (H2 : P = 490)  -- The winner received 490 votes
    : ∃ L : ℕ, (V = P + L) ∧ (P - L = 280) := 
  by
    sorry

end election_winner_votes_diff_l158_158651


namespace IH_length_eq_l158_158103

noncomputable def length_IH : ℝ :=
  let AB := 4
  let CE := 12
  let DF := real.sqrt (16^2 + 12^2)
  let r := 12 / 20
  let GJ := r * 12
  GJ

theorem IH_length_eq : length_IH = 36 / 5 := by
  -- Definitions
  let AB := 4
  let CE := 12
  let DE := AB + CE
  let EF := CE
  let DF := real.sqrt (DE^2 + EF^2)
  let r := EF / DF
  let GJ := r * EF

  -- Proof sketch (not required)
  -- 1. Calculate DF using Pythagorean theorem
  -- 2. Use similarity ratio to find GJ
  -- 3. Show that IH equals GJ and simplify to obtain the result

  rw [DE, EF, DF, GJ] -- Replace with actual values
  repeat { rw [pow_two] }
  repeat { rw [real.sqrt_eq_rpow] }
  norm_num -- Simplify the result
  exact_mod_cast sorry-- Exact proof of the final simplified result

end IH_length_eq_l158_158103


namespace ratio_snakes_to_lions_is_S_per_100_l158_158723

variables {S G : ℕ}

/-- Giraffe count in Safari National Park is 10 fewer than snakes -/
def safari_giraffes_minus_ten (S G : ℕ) : Prop := G = S - 10

/-- The number of lions in Safari National Park -/
def safari_lions : ℕ := 100

/-- The ratio of number of snakes to number of lions in Safari National Park -/
def ratio_snakes_to_lions (S : ℕ) : ℕ := S / safari_lions

/-- Prove the ratio of the number of snakes to the number of lions in Safari National Park -/
theorem ratio_snakes_to_lions_is_S_per_100 :
  ∀ S G, safari_giraffes_minus_ten S G → (ratio_snakes_to_lions S = S / 100) :=
by
  intros S G h
  sorry

end ratio_snakes_to_lions_is_S_per_100_l158_158723


namespace central_angle_of_cone_l158_158018

theorem central_angle_of_cone (A : ℝ) (l : ℝ) (r : ℝ) (θ : ℝ)
  (hA : A = (1 / 2) * 2 * Real.pi * r)
  (hl : l = 1)
  (ha : A = (3 / 8) * Real.pi) :
  θ = (3 / 4) * Real.pi :=
by
  sorry

end central_angle_of_cone_l158_158018


namespace arccos_of_one_over_sqrt_two_l158_158888

theorem arccos_of_one_over_sqrt_two :
  Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 := 
sorry

end arccos_of_one_over_sqrt_two_l158_158888


namespace number_of_sets_C_l158_158553

variable {A B : Set} {M N : ℕ}
variable (card_A : card A = M) (card_B : card B = N) (hMN : M < N)
variable (h_sub : A ⊆ B)

theorem number_of_sets_C (A B : Set) (M N : ℕ) (card_A : card A = M) (card_B : card B = N) (hMN : M < N) (h_sub : A ⊆ B) : 
  ∃ (C : Set), A ⊆ C ∧ C ⊆ B ∧ card {C : Set | A ⊆ C ∧ C ⊆ B} = 2 ^ (N - M) :=
sorry

end number_of_sets_C_l158_158553


namespace profit_relationship_max_profit_l158_158474

noncomputable def W (x : ℝ) : ℝ :=
if h : 0 ≤ x ∧ x ≤ 2 then 5 * (x^2 + 3)
else if h : 2 < x ∧ x ≤ 5 then 50 * x / (1 + x)
else 0

noncomputable def f (x : ℝ) : ℝ :=
15 * W x - 10 * x - 20 * x

theorem profit_relationship:
  (∀ x, 0 ≤ x ∧ x ≤ 2 → f x = 75 * x^2 - 30 * x + 225) ∧
  (∀ x, 2 < x ∧ x ≤ 5 → f x = (750 * x)/(1 + x) - 30 * x) :=
by
  -- to be proven
  sorry

theorem max_profit:
  ∃ x, 0 ≤ x ∧ x ≤ 5 ∧ f x = 480 ∧ 10 * x = 40 :=
by
  -- to be proven
  sorry

end profit_relationship_max_profit_l158_158474


namespace sin_alpha_correct_l158_158332

noncomputable def sin_alpha_between_line_and_plane 
  (d : ℝ × ℝ × ℝ) (n : ℝ × ℝ × ℝ) (alpha : ℝ) : ℝ :=
  let dot_product := (d.1 * n.1) + (d.2 * n.2) + (d.3 * n.3) in
  let d_magnitude := real.sqrt (d.1^2 + d.2^2 + d.3^2) in
  let n_magnitude := real.sqrt (n.1^2 + n.2^2 + n.3^2) in
  real.abs ((dot_product) / (d_magnitude * n_magnitude))

theorem sin_alpha_correct : 
  sin_alpha_between_line_and_plane (3, 4, 5) (8, 4, -9) = -5 / real.sqrt 8050 := 
by
  sorry

end sin_alpha_correct_l158_158332


namespace conditional_probability_B_given_A_is_three_eighths_l158_158237

def num_ways_event_A : ℕ := 90
def num_ways_event_B_given_A : ℕ := 240
def P_B_given_A := (num_ways_event_A : ℚ) / (num_ways_event_B_given_A : ℚ)

theorem conditional_probability_B_given_A_is_three_eighths :
  P_B_given_A = 3 / 8 :=
by {
  -- We define the constants
  let num_ways_event_A := 90,
  let num_ways_event_B_given_A := 240,
  
  -- Convert to rational numbers for precision
  let P_B_given_A : ℚ := num_ways_event_A / num_ways_event_B_given_A,
  
  -- Implement the proof
  have P : ℚ := 3 / 8,
  exact eq.refl P
}

end conditional_probability_B_given_A_is_three_eighths_l158_158237


namespace sum_of_six_terms_arithmetic_sequence_l158_158403

theorem sum_of_six_terms_arithmetic_sequence (S : ℕ → ℕ)
    (h1 : S 2 = 2)
    (h2 : S 4 = 10) :
    S 6 = 42 :=
by
  sorry

end sum_of_six_terms_arithmetic_sequence_l158_158403


namespace arccos_one_over_sqrt_two_l158_158975

theorem arccos_one_over_sqrt_two :
  Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_l158_158975


namespace num_students_only_math_l158_158762

def oakwood_ninth_grade_problem 
  (total_students: ℕ)
  (students_in_math: ℕ)
  (students_in_foreign_language: ℕ)
  (students_in_science: ℕ)
  (students_in_all_three: ℕ)
  (students_total_from_ie: ℕ) :=
  (total_students = 120) ∧
  (students_in_math = 85) ∧
  (students_in_foreign_language = 65) ∧
  (students_in_science = 75) ∧
  (students_in_all_three = 20) ∧
  total_students = students_in_math + students_in_foreign_language + students_in_science 
  - (students_total_from_ie) + students_in_all_three - (students_in_all_three)

theorem num_students_only_math 
  (total_students: ℕ := 120)
  (students_in_math: ℕ := 85)
  (students_in_foreign_language: ℕ := 65)
  (students_in_science: ℕ := 75)
  (students_in_all_three: ℕ := 20)
  (students_total_from_ie: ℕ := 45) :
  oakwood_ninth_grade_problem total_students students_in_math students_in_foreign_language students_in_science students_in_all_three students_total_from_ie →
  ∃ (students_only_math: ℕ), students_only_math = 75 :=
by
  sorry

end num_students_only_math_l158_158762


namespace fixed_point_of_log_function_l158_158201

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log a (x - 2016) + 1

theorem fixed_point_of_log_function (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) :
  ∃ P : ℝ × ℝ, P = (2017, 1) ∧ ∀ x, f a x = 1 → x = 2017 :=
by
  use (2017, 1)
  sorry

end fixed_point_of_log_function_l158_158201


namespace matrix_pow_minus_l158_158676

open Matrix

def B : Matrix (Fin 2) (Fin 2) ℤ := ![![3, 4], ![0, 2]]

theorem matrix_pow_minus : B ^ 20 - 3 * (B ^ 19) = ![![0, 4 * (2 ^ 19)], ![0, -(2 ^ 19)]] :=
by
  sorry

end matrix_pow_minus_l158_158676


namespace ratio_AB_AC_equals_FB_FC_l158_158682

theorem ratio_AB_AC_equals_FB_FC
  (A B C D E F: Point)
  (Γ: Circle)
  (h_scalene: scalene △ A B C)
  (h_circumcircle: \Gamma.circumscribes △ A B C)
  (h_bisector_A: internal_bisector A intersects [D, E])
  (h_diameter_DE: circle_with_diameter D E)
  (h_F_on_Γ: F ∈ \Gamma)
  (h_second_intersection: second_intersection F circle_with_diameter D E on_line [Γ]):
  ratio AB AC = ratio FB FC := 
begin
  sorry -- Proof goes here.
end

end ratio_AB_AC_equals_FB_FC_l158_158682


namespace route_C_is_quicker_l158_158702

/-
  Define the conditions based on the problem:
  - Route C: 8 miles at 40 mph.
  - Route D: 5 miles at 35 mph and 2 miles at 25 mph with an additional 3 minutes stop.
-/

def time_route_C : ℚ := (8 : ℚ) / (40 : ℚ) * 60  -- in minutes

def time_route_D : ℚ := ((5 : ℚ) / (35 : ℚ) * 60) + ((2 : ℚ) / (25 : ℚ) * 60) + 3  -- in minutes

def time_difference : ℚ := time_route_D - time_route_C  -- difference in minutes

theorem route_C_is_quicker : time_difference = 4.37 := 
by 
  sorry

end route_C_is_quicker_l158_158702


namespace func_eq_id_l158_158485

theorem func_eq_id (f : ℕ+ → ℕ+)
  (h : ∀ m n : ℕ+, (m^2 + n)^2 ∣ f^2 m + f n) :
  ∀ n : ℕ+, f n = n :=
sorry

end func_eq_id_l158_158485


namespace function_decrease_interval_l158_158528

-- Definition of the function f(x)
def f (x : ℝ) : ℝ := x^3 - 15*x^2 - 33*x + 6

-- Statement of the proof problem
theorem function_decrease_interval :
  ∀ x : ℝ, x ∈ Ioo (-1) 11 → f' x < 0 :=
by
  sorry

end function_decrease_interval_l158_158528


namespace general_term_series_value_l158_158571

-- Given a sequence {a_n} such that S_n + 1/2 * a_n = 1 for all n ∈ ℕ+
def sequence_condition (n : ℕ) (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  S n + (1/2)*a n = 1

-- Define the general term of the sequence {a_n}
noncomputable def a_n (n : ℕ) : ℝ := 2 / 3 ^ (n : ℕ)

-- Given b_n = log_{1/3}(1 - S_n)
noncomputable def S_n (n : ℕ) : ℝ := 1 - (1/3) ^ (n : ℕ)
noncomputable def b_n (n : ℕ) : ℝ := Math.logBase (1/3 : ℝ) (1 - S_n n) 

-- Prove that a_n = 2 / 3^n
theorem general_term (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h : ∀ n : ℕ, sequence_condition n S a) :
  ∀ n : ℕ, a n = 2 / 3 ^ (n : ℕ) :=
sorry

-- Prove that the value of the series is n / (n + 1)
theorem series_value (n : ℕ) (h : ∀ n : ℕ, sequence_condition n S_n a_n) :
  (∑ i in Finset.range n, 1 / (b_n i * b_n (i + 1))) = n / (n + 1) :=
sorry

end general_term_series_value_l158_158571


namespace decode_digits_with_5_queries_l158_158010

noncomputable def letter_digit_sum (letters : List Char) (digits : List Nat) : Nat :=
  letters.map (λ c => match c with
                 | 'A' => digits.get 0
                 | 'B' => digits.get 1
                 | 'C' => digits.get 2
                 | 'D' => digits.get 3
                 | 'E' => digits.get 4
                 | 'F' => digits.get 5
                 | 'G' => digits.get 6
                 | 'H' => digits.get 7
                 | 'I' => digits.get 8
                 | 'J' => digits.get 9
                 | _   => 0
               end).sum

theorem decode_digits_with_5_queries :
  ∃ (queries : List (List Char)), List.length queries = 5 ∧ 
  ∀ (digits : List Nat), List.length digits = 10 ∧ (List.range 10).sum = 45 →
  (∀ (answers : List Nat), List.length answers = 5 → 
   ∃ (mapping : Char → Nat), 
    ∀ c, c ∈ ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'] → 
         (letter_digit_sum ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'] digits = 45) →
         mapping c = digits.get (['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'].indexOf c)) := sorry

end decode_digits_with_5_queries_l158_158010


namespace solution_ratio_l158_158042

-- Describe the problem conditions
variable (a b : ℝ) -- amounts of solutions A and B

-- conditions
def proportion_A : ℝ := 0.20 -- Alcohol concentration in solution A
def proportion_B : ℝ := 0.60 -- Alcohol concentration in solution B
def final_proportion : ℝ := 0.40 -- Final alcohol concentration

-- Lean statement
theorem solution_ratio (h : 0.20 * a + 0.60 * b = 0.40 * (a + b)) : a = b := by
  sorry

end solution_ratio_l158_158042


namespace parallel_vectors_sum_l158_158218

variable (x y : ℝ)
variable (k : ℝ)

theorem parallel_vectors_sum :
  (k * 3 = 2) ∧ (k * x = 4) ∧ (k * y = 5) → x + y = 27 / 2 :=
by
  sorry

end parallel_vectors_sum_l158_158218


namespace find_side_b_l158_158638

theorem find_side_b (α : ℝ) (sin_a : ℝ) (sin_3a : ℝ) (sin_4a : ℝ)
  (ha : a = 27) (hc : c = 48) (h_C : angle_C = 3 * angle_A) 
  (h_sin3a : sin_3a = 3 * sin_a - 4 * sin_a^3)
  (h_sin4a : sin_4a = 2 * (2 * sin_a * sqrt(1 - sin_a^2)) * (1 - 2 * sin_a^2)) : b = 35 :=
by
  sorry

end find_side_b_l158_158638


namespace min_cost_11_l158_158300

def min_cost_for_prizes (x y : ℕ) : ℕ :=
  3 * x + 2 * y

theorem min_cost_11 :
  ∃ (x y : ℕ), x + y ≤ 10 ∧ |x - y| ≤ 2 ∧ x ≥ 3 ∧ min_cost_for_prizes x y = 11 :=
sorry

end min_cost_11_l158_158300


namespace only_function_l158_158465

def divides (a b : ℕ) : Prop := ∃ k, b = k * a

def satisfies_condition (f : ℕ → ℕ) : Prop :=
  ∀ m n : ℕ, divides (f m + f n) (m + n)

theorem only_function (f : ℕ → ℕ) (h : satisfies_condition f) : f = id :=
by
  -- Proof goes here.
  sorry

end only_function_l158_158465


namespace arccos_one_over_sqrt_two_l158_158950

theorem arccos_one_over_sqrt_two : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_l158_158950


namespace prove_f_2_eq_3_l158_158200

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x ≤ 1 then 3 * a ^ x else Real.log (2 * x + 4) / Real.log a

theorem prove_f_2_eq_3 (a : ℝ) (h1 : f 1 a = 6) : f 2 a = 3 :=
by
  -- Define the conditions
  have h1 : 3 * a = 6 := by simp [f] at h1; assumption
  -- Two subcases: x <= 1 and x > 1
  have : a = 2 := by linarith
  simp [f, this]
  sorry

end prove_f_2_eq_3_l158_158200


namespace first_discount_is_15_l158_158397

-- Conditions as definitions
def initial_price : ℝ := 400
def final_price : ℝ := 323
def first_discount (x : ℝ) : ℝ := initial_price - (x / 100) * initial_price
def second_discount (x : ℝ) : ℝ := first_discount x - 0.05 * first_discount x

-- Theorem statement (proof to be provided)
theorem first_discount_is_15 : ∃ x : ℝ, second_discount x = final_price ∧ x = 15 :=
by
  sorry

end first_discount_is_15_l158_158397


namespace fraction_simplest_form_l158_158411

theorem fraction_simplest_form (a b : ℕ) (h1 : (a + 2) * 7 = 4 * b) (h2 : a * 25 = 14 * (b - 2)) :
  a = 6 * (11 : ℕ ∙ a + b = 77) := by
  sorry

end fraction_simplest_form_l158_158411


namespace quadratic_equation_real_roots_l158_158629

theorem quadratic_equation_real_roots (k : ℝ) : 
  (∃ x : ℝ, k * x^2 - 6 * x + 9 = 0) ↔ (k ≤ 1 ∧ k ≠ 0) :=
by
  sorry

end quadratic_equation_real_roots_l158_158629


namespace band_percentage_proof_l158_158150

noncomputable def total_band_income (members: ℕ) (income_per_member: ℕ) : ℕ :=
  members * income_per_member

noncomputable def total_revenue (attendees: ℕ) (ticket_price: ℕ) : ℕ :=
  attendees * ticket_price

noncomputable def band_percentage_of_revenue (band_income: ℕ) (revenue: ℕ) : ℚ :=
  (band_income / revenue : ℚ) * 100

theorem band_percentage_proof :
  ∀ (attendees: ℕ) (ticket_price: ℕ) (members: ℕ) (income_per_member: ℕ),
    attendees = 500 →
    ticket_price = 30 →
    members = 4 →
    income_per_member = 2625 →
    band_percentage_of_revenue (total_band_income members income_per_member) (total_revenue attendees ticket_price) = 70 :=
begin
  intros attendees ticket_price members income_per_member,
  intros attendees_eq ticket_price_eq members_eq income_per_member_eq,
  sorry
end

end band_percentage_proof_l158_158150


namespace probability_x_gt_5y_l158_158718

def rectangle : set (ℝ × ℝ) := {(x, y) | 0 ≤ x ∧ x ≤ 3000 ∧ 0 ≤ y ∧ y ≤ 2000}

def line_inequality (x y : ℝ) : Prop := x > 5 * y

theorem probability_x_gt_5y : 
∃ (p : ℚ), p = 3 / 20 ∧ ∀ (x y : ℝ), (x, y) ∈ rectangle → ¬ line_inequality x y ↔ x <= 5 * y :=
sorry

end probability_x_gt_5y_l158_158718


namespace partition_into_k_plus_2_l158_158409

-- Defining the basic structures and hypotheses
def Vertex := Type
def Edge (V : Type) := V × V

variables (V : Vertex) (k : ℕ)
variable (G : fin k → set (Edge V))

-- Hypothesis: Each airline's flight connections share a common endpoint
axiom common_endpoint (i : fin k) :
  ∃ v : V, ∀ e ∈ G i, v = e.1 ∨ v = e.2

-- The theorem to prove
theorem partition_into_k_plus_2 (k : ℕ) :
  ∃ (P : fin (k+2) → set V), 
    (∀ i j, i ≠ j → P i ∩ P j = ∅) ∧ -- Partition into disjoint sets
    (∀ i : fin (k+2), ∀ v1 v2 ∈ P i, ∀ (n : fin k), 
      (v1, v2) ∉ G n) := sorry

end partition_into_k_plus_2_l158_158409


namespace arccos_one_over_sqrt_two_eq_pi_over_four_l158_158900

theorem arccos_one_over_sqrt_two_eq_pi_over_four : 
  Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_eq_pi_over_four_l158_158900


namespace units_digit_41_pow_3_plus_23_pow_3_l158_158794

def units_digit (n : ℕ) : ℕ := n % 10

def ends_in (n d : ℕ) : Prop := units_digit n = d

theorem units_digit_41_pow_3_plus_23_pow_3 :
  ends_in 41 1 ∧ ends_in 23 3 → units_digit (41^3 + 23^3) = 8 := 
by 
  intros h,
  sorry

end units_digit_41_pow_3_plus_23_pow_3_l158_158794


namespace distance_between_points_eq_l158_158319

theorem distance_between_points_eq (z : ℝ) :
  (real.sqrt ((5 - 2)^2 + (4 - 2)^2 + (z - 5)^2)) = 7 ↔ (z = 11 ∨ z = -1) :=
by
  sorry

end distance_between_points_eq_l158_158319


namespace find_a_max_and_min_values_l158_158202

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x * (x^2 + x + a)
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := Real.exp x * (x^2 + 3*x + a + 1)

theorem find_a (a : ℝ) : (f' a 0) = 2 → a = 1 :=
by {
  -- Proof omitted
  sorry
}

theorem max_and_min_values (a : ℝ) :
  (a = 1) →
  (Real.exp (-2) * (4 - 2 + 1) = (3 / Real.exp 2)) ∧
  (Real.exp (-1) * (1 - 1 + 1) = (1 / Real.exp 1)) ∧
  (Real.exp 2 * (4 + 2 + 1) = (7 * Real.exp 2)) :=
by {
  -- Proof omitted
  sorry
}

end find_a_max_and_min_values_l158_158202


namespace value_of_m_l158_158122

theorem value_of_m (a b c d: ℝ) (h_d_nonzero : d ≠ 0)
  (h_roots_opposite_sign : ∀ x, x ^ 2 - (b + 1) * x = (a - 1) * x - (c + d) → ∃ r, x = r ∨ x = -r) :
  m = 2 * (a - b - 2) / (a + b) :=
begin
  sorry,
end

end value_of_m_l158_158122


namespace arccos_identity_l158_158918

theorem arccos_identity : 
  ∀ x : ℝ, x = 1 / real.sqrt 2 → real.arccos x = real.pi / 4 :=
begin
  intros x hx,
  have h1 : real.cos (real.pi / 4) = 1 / real.sqrt 2 := by sorry,
  have h2 : real.arccos (1 / real.sqrt 2) = real.pi / 4,
  { rw ← h1,
    rw real.arccos_cos,
    exact_mod_cast hx },
  exact h2,
end

end arccos_identity_l158_158918


namespace card_arrangements_valid_removal_l158_158726

open Finset

theorem card_arrangements_valid_removal :
  let cards := {1, 2, 3, 4, 5, 6, 7}
  ∃ n : ℕ, 
  (n = 7 ∧ 
  (∀ L : list ℕ, L ∈ (cards.powerset.filter (λ s, s.card = 6)) ∧ (sorted L ∨ sorted (L.reverse)) → n = 74 )) := sorry

end card_arrangements_valid_removal_l158_158726


namespace Vlad_score_l158_158378

-- Defining the initial conditions of the problem
def total_rounds : ℕ := 30
def points_per_win : ℕ := 5
def total_points : ℕ := total_rounds * points_per_win

-- Taro's score as described in the problem
def Taros_score := (3 * total_points / 5) - 4

-- Prove that Vlad's score is 64 points
theorem Vlad_score : total_points - Taros_score = 64 := by
  sorry

end Vlad_score_l158_158378


namespace solve_quad_eq_l158_158027

theorem solve_quad_eq (x : ℝ) : x ^ 2 = 2 * x → x = 0 ∨ x = 2 :=
  sorry

end solve_quad_eq_l158_158027


namespace probability_quadratic_distinct_real_roots_l158_158371

/--
Suppose \( a \) is the number obtained by throwing a die. 
The probability that the quadratic equation \( x^2 + ax + 2 = 0 \) has two distinct real roots is \( \frac{2}{3} \).
-/
theorem probability_quadratic_distinct_real_roots :
  (∃ (a : ℕ), a ∈ {1, 2, 3, 4, 5, 6} ∧ (a^2 - 8 > 0)) = (4 / 6) := 
by
  sorry

end probability_quadratic_distinct_real_roots_l158_158371


namespace table_relationship_l158_158128

theorem table_relationship (x y : ℕ) (h : (x, y) ∈ [(1, 1), (2, 8), (3, 27), (4, 64), (5, 125)]) : y = x^3 :=
sorry

end table_relationship_l158_158128


namespace art_group_students_count_l158_158324

theorem art_group_students_count (x : ℕ) (h1 : x * (1 / 60) + 2 * (x + 15) * (1 / 60) = 1) : x = 10 :=
by {
  sorry
}

end art_group_students_count_l158_158324


namespace ellipse_equation_standard_form_l158_158503

theorem ellipse_equation_standard_form :
  ∃ (a b : ℝ) (h k : ℝ), 
    a = (Real.sqrt 146 + Real.sqrt 242) / 2 ∧ 
    b = Real.sqrt ((Real.sqrt 146 + Real.sqrt 242) / 2)^2 - 9 ∧ 
    h = 1 ∧ 
    k = 4 ∧ 
    (∀ x y : ℝ, (x, y) = (12, -4) → 
      ((x - h)^2 / a^2 + (y - k)^2 / b^2 = 1)) :=
  sorry

end ellipse_equation_standard_form_l158_158503


namespace asymptote_of_hyperbola_l158_158600

-- Define the given hyperbola and its conditions
structure Hyperbola where
  a : ℝ
  b : ℝ
  h : a > 0
  k : b > 0
  eq : (x y : ℝ) → (x^2 / a^2 - y^2 / b^2 = 1)

-- Define the given parabola
structure Parabola where
  eq : (x y : ℝ) → (y^2 = 16 * x)
  focus : (x y : ℝ) × (x = 4 ∧ y = 0)

-- Define the given eccentricity condition
def eccentricity (hyp: Hyperbola) := (hyp.a * 2 = 4)

-- State the goal as a theorem
theorem asymptote_of_hyperbola (hyp : Hyperbola) (par : Parabola) (e : eccentricity hyp) : 
  hyp.b = 2 * sqrt(3) ∧ (forall x y : ℝ, hyp.eq x y → (y = sqrt(3) * x ∨ y = -sqrt(3) * x)) :=
by
  sorry

end asymptote_of_hyperbola_l158_158600


namespace sum_of_digits_of_10_pow_30_minus_36_l158_158462

def sum_of_digits (n : ℕ) : ℕ := 
  (n.digits 10).sum

theorem sum_of_digits_of_10_pow_30_minus_36 : 
  sum_of_digits (10^30 - 36) = 11 := 
by 
  -- proof goes here
  sorry

end sum_of_digits_of_10_pow_30_minus_36_l158_158462


namespace probability_of_even_sum_is_four_fifths_l158_158265

-- Define the first ten prime numbers
def first_ten_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- The function to calculate the probability that the sum of two distinct primes is even
def probability_even_sum (primes : List ℕ) : ℚ :=
  let pairs := primes.product primes
  let distinct_pairs := pairs.filter (λ p, p.1 ≠ p.2)
  let even_sum_pairs := distinct_pairs.filter (λ p, (p.1 + p.2) % 2 = 0)
  (even_sum_pairs.length : ℚ) / (distinct_pairs.length : ℚ)

-- Prove that the probability is 4/5
theorem probability_of_even_sum_is_four_fifths :
  probability_even_sum first_ten_primes = 4 / 5 := sorry

end probability_of_even_sum_is_four_fifths_l158_158265


namespace product_4_7_25_l158_158787

theorem product_4_7_25 : 4 * 7 * 25 = 700 :=
by sorry

end product_4_7_25_l158_158787


namespace sum_of_digits_of_smallest_number_l158_158125

noncomputable def sum_of_digits (n : Nat) : Nat :=
  n.digits 10 |>.foldl (· + ·) 0

theorem sum_of_digits_of_smallest_number :
  (n : Nat) → (h1 : (Nat.ceil (n / 2) - Nat.ceil (n / 3) = 15)) → 
  sum_of_digits n = 9 :=
by
  sorry

end sum_of_digits_of_smallest_number_l158_158125


namespace poisson_theorem_estimate_l158_158364

noncomputable theory

variable (n : ℕ) (λ : ℝ)

/-- Given independent Poisson-distributed random variables ηᵢ with parameter λ / n,
and independent Bernoulli-distributed random variables ξᵢ defined as in the problem,
prove that the following estimate holds:
supₖ |Pₙ(k) - λᵏ e^(-λ) / k!| ≤ λ² / n.
-/
theorem poisson_theorem_estimate :
  ∀ (η : ℕ → measure_theory.measurable_space ℕ)
    (P_n : ℕ → ℝ)
  (hyp1 : ∀ i, measure_theory.probability_measure (η i))
  (hyp2 : ∀ i, measure_theory.prob (λ ω, η i ω = 0) = λ / n)
  (hyp3 : ∀ i j, i ≠ j → measure_theory.indep (η i) (η j))
  (hyp4 : ∀ k, P_n k = (λ ^ k) * real.exp (-λ) / k!) ,
  (∀ k, |P_n k - (λ ^ k * real.exp (-λ) / k!)| ≤ λ ^ 2 / n) := 
sorry

end poisson_theorem_estimate_l158_158364


namespace arccos_identity_l158_158914

theorem arccos_identity : 
  ∀ x : ℝ, x = 1 / real.sqrt 2 → real.arccos x = real.pi / 4 :=
begin
  intros x hx,
  have h1 : real.cos (real.pi / 4) = 1 / real.sqrt 2 := by sorry,
  have h2 : real.arccos (1 / real.sqrt 2) = real.pi / 4,
  { rw ← h1,
    rw real.arccos_cos,
    exact_mod_cast hx },
  exact h2,
end

end arccos_identity_l158_158914


namespace minimum_value_of_C_over_D_is_three_l158_158234

variable (x : ℝ) (C D : ℝ)
variables (hxC : x^3 + 1/(x^3) = C) (hxD : x - 1/(x) = D)

theorem minimum_value_of_C_over_D_is_three (hC : C = D^3 + 3 * D) :
  ∃ x : ℝ, x^3 + 1/(x^3) = C ∧ x - 1/(x) = D → C / D ≥ 3 :=
by
  sorry

end minimum_value_of_C_over_D_is_three_l158_158234


namespace proof_problem_l158_158068

universe u

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 3, 4}
def B : Set ℕ := {1, 3}
def complement (s : Set ℕ) : Set ℕ := {x | x ∉ s}

theorem proof_problem : ((complement A ∪ A) ∪ B) = U :=
by sorry

end proof_problem_l158_158068


namespace four_digit_number_divisible_by_18_l158_158154

theorem four_digit_number_divisible_by_18 : ∃ n : ℕ, (n % 2 = 0) ∧ (10 + n) % 9 = 0 ∧ n = 8 :=
by
  sorry

end four_digit_number_divisible_by_18_l158_158154


namespace largest_prime_factor_is_94_l158_158050

/-- 
Prove that the number with the largest prime factor among 
55, 63, 85, 94, and 133 is 94.
-/
theorem largest_prime_factor_is_94 :
  ∀ (n : ℕ),
    (n = 55 ∨ n = 63 ∨ n = 85 ∨ n = 94 ∨ n = 133) →
      (∀ p, prime p → p ∣ n → p ≤ 47) →
      n ≠ 94 → False :=
by
  intros n h hn heq
  cases h with h1 h
  cases h1 with h55 h
  cases h55 with h63 h
  cases h63 with h85 h94
  cases h94 with h94 _

  -- Proof is omitted
  sorry

end largest_prime_factor_is_94_l158_158050


namespace auntie_em_parking_probability_l158_158841

theorem auntie_em_parking_probability :
  let total_spaces := 20
  let cars := 15
  let empty_spaces := total_spaces - cars
  let possible_configurations := Nat.choose total_spaces cars
  let unfavourable_configurations := Nat.choose (empty_spaces - 8 + 5) (empty_spaces - 8)
  let favourable_probability := 1 - ((unfavourable_configurations : ℚ) / (possible_configurations : ℚ))
  (favourable_probability = 1839 / 1938) :=
by
  -- sorry to skip the actual proof
  sorry

end auntie_em_parking_probability_l158_158841


namespace arccos_identity_l158_158915

theorem arccos_identity : 
  ∀ x : ℝ, x = 1 / real.sqrt 2 → real.arccos x = real.pi / 4 :=
begin
  intros x hx,
  have h1 : real.cos (real.pi / 4) = 1 / real.sqrt 2 := by sorry,
  have h2 : real.arccos (1 / real.sqrt 2) = real.pi / 4,
  { rw ← h1,
    rw real.arccos_cos,
    exact_mod_cast hx },
  exact h2,
end

end arccos_identity_l158_158915


namespace length_of_side_of_base_of_pyramid_l158_158381

noncomputable def length_of_side (area_of_face : ℝ) (slant_height : ℝ) : ℝ :=
  (2 * area_of_face) / slant_height

theorem length_of_side_of_base_of_pyramid :
  length_of_side 75 30 = 5 :=
by
  rw [length_of_side]
  norm_num
  sorry

end length_of_side_of_base_of_pyramid_l158_158381


namespace additional_money_needed_l158_158117

def original_num_bales : ℕ := 10
def original_cost_per_bale : ℕ := 15
def new_cost_per_bale : ℕ := 18

theorem additional_money_needed :
  (2 * original_num_bales * new_cost_per_bale) - (original_num_bales * original_cost_per_bale) = 210 :=
by
  sorry

end additional_money_needed_l158_158117


namespace find_result_l158_158330

variable (B C B' C' A M H T : Point) (circumcircle : Circle)
variable [rectangle : Rectangle B C B' C']
variable [mid_M : Midpoint M B' C']
variable [on_circumcircle : OnCircumcircle A B C B' C' circumcircle]
variable [orthocenter : Orthocenter H A B C]
variable [foot_perpendicular : FootPerpendicular T H A M]
variable (AM_val : Real) [AM_val = 2]
variable (area_ABC : Real) [area_ABC = 2020]
variable (BC_val : Real) [BC_val = 10]

theorem find_result (AT : Rational) (m n : ℕ) (h_coprime : gcd m n = 1) 
    (h_AT : AT = m / n) : 100 * m + n = 2102 := 
sorry

end find_result_l158_158330


namespace probability_even_sum_l158_158286

def first_ten_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def distinct_pairs (l : List ℕ) : List (ℕ × ℕ) :=
  (l.product l).filter (λ p, p.1 < p.2)

theorem probability_even_sum : 
  (first_ten_primes.length = 10) →
  (∀ a b : ℕ, a ∈ first_ten_primes → b ∈ first_ten_primes → a ≠ b → 
    ((a + b) % 2 = 0 ↔ 2 ∉ [a, b])) →
  (∑ pair in distinct_pairs first_ten_primes, ((pair.1 + pair.2) % 2 = 0).toNat) 
    = 36 → 
  ((∑ pair in distinct_pairs first_ten_primes, (1:ℚ)) ⁻¹ * 
   (∑ pair in distinct_pairs first_ten_primes, ((pair.1 + pair.2) % 2 = 0).toNat))
    = 4 / 5 := by
  sorry

end probability_even_sum_l158_158286


namespace geometric_sequence_sum_l158_158659

theorem geometric_sequence_sum (a : ℕ → ℕ) (b : ℕ → ℕ)
  (h1 : a 2 = 3)
  (h2 : a 5 = 81)
  (h3 : ∀ n, a n = 3^(n-1))
  (h4 : ∀ n, b n = log 3 (a n) + 1) :
  ∑ i in Finset.range n, 1 / (b i * b (i + 1)) = n / (n + 1) := 
by
  sorry

end geometric_sequence_sum_l158_158659


namespace tangent_line_at_P_tangent_lines_through_Q_l158_158591

noncomputable def f (x : ℝ) : ℝ := x^3 - 2 * x^2 + 1

theorem tangent_line_at_P :
  ∀ x y, (∃ k, f' x = k ∧ y = k * (x - 1) + f 1) -> (x + y - 1 = 0) :=
by
  sorry

noncomputable def f' (x : ℝ) : ℝ := 3 * x^2 - 4 * x

theorem tangent_lines_through_Q :
  ∀ x y, (∃ k x₀, f' x₀ = k ∧ y = k * (x - x₀) + f x₀) ->
    (y = 1 ∨ 4 * x - y - 7 = 0) :=
by
  sorry

end tangent_line_at_P_tangent_lines_through_Q_l158_158591


namespace find_three_digit_number_l158_158138

noncomputable def is_valid_digit (d : Nat) : Prop :=
  d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 5 ∨ d = 6 ∨ d = 7 ∨ d = 8 ∨ d = 9

noncomputable def russian_starting_letter (d : Nat) : Char :=
  if d = 1 then 'о'
  else if d = 2 then 'д'
  else if d = 3 then 'т'
  else if d = 4 then 'ч'
  else if d = 5 then 'п'
  else if d = 6 then 'ш'
  else if d = 7 then 'с'
  else if d = 8 then 'в'
  else 'д' -- d = 9

theorem find_three_digit_number :
  ∃ (n : Nat), (100 ≤ n ∧ n < 1000) ∧ 
               (∀ d1 d2 d3 : Nat, d1 ≠ d2 ∧ d2 ≠ d3 ∧ d1 ≠ d3 →
                (d1 * 100 + d2 * 10 + d3 = n) → 
                (is_valid_digit d1 ∧ is_valid_digit d2 ∧ is_valid_digit d3) ∧
                (d1 < d2 ∧ d2 < d3) ∧
                (russian_starting_letter d1 = russian_starting_letter d2 ∧
                 russian_starting_letter d2 = russian_starting_letter d3)) :=
begin
  use 147,
  split,
  -- 147 is between 100 and 1000
  linarith,
  intros d1 d2 d3 h_unique h_n,
  split, 
  -- Valid digits
  { sorry },
  split,
  -- Ascending order
  { sorry },
  -- Starting letters in Russian
  { sorry },
end

end find_three_digit_number_l158_158138


namespace length_CD_l158_158323

noncomputable def IsoscelesTriangleABE : Type := sorry
noncomputable def condition_1 (T : IsoscelesTriangleABE) : Prop := isIsosceles T
noncomputable def condition_2 (T : IsoscelesTriangleABE) : Prop := area T = 200
noncomputable def condition_3 (T : IsoscelesTriangleABE) : Prop := 
  exists_trapezoid_and_smaller_triangle T ∧ 
  area_trapezoid T = 150
noncomputable def condition_4 (T : IsoscelesTriangleABE) : Prop := altitude_from_A T = 25

theorem length_CD (T : IsoscelesTriangleABE) 
  (h1 : condition_1 T) 
  (h2 : condition_2 T)
  (h3 : condition_3 T) 
  (h4 : condition_4 T) : 
  length_CD T = 8 := sorry

end length_CD_l158_158323


namespace arccos_sqrt_half_l158_158963

theorem arccos_sqrt_half : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 := by
  sorry

end arccos_sqrt_half_l158_158963


namespace simplify_and_evaluate_l158_158734

noncomputable def my_expression (m : ℝ) : ℝ :=
  (m - (m + 9) / (m + 1)) / ((m ^ 2 + 3 * m) / (m + 1))

theorem simplify_and_evaluate : my_expression (Real.sqrt 3) = 1 - Real.sqrt 3 :=
by
  sorry

end simplify_and_evaluate_l158_158734


namespace floor_sqrt_18_squared_l158_158135

theorem floor_sqrt_18_squared : ∃ x : ℕ, 4 < real.sqrt 18 ∧ real.sqrt 18 < 5 ∧ x = ⌊real.sqrt 18⌋ ∧ x^2 = 16 := by
  sorry

end floor_sqrt_18_squared_l158_158135


namespace probability_even_sum_l158_158252

def firstTenPrimes : Set ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

def isOdd (n : ℕ) : Prop := n % 2 = 1

def distinctPairs (s : Set ℕ) : Set (ℕ × ℕ) :=
  { (x, y) | x ∈ s ∧ y ∈ s ∧ x ≠ y }

def eventEvenSum (s : Set ℕ) : Set (ℕ × ℕ) :=
  { (x, y) | x ∈ s ∧ y ∈ s ∧ x ≠ y ∧ (x + y) % 2 = 0 }

theorem probability_even_sum :
  (Set.card (eventEvenSum firstTenPrimes)).toRat / (Set.card (distinctPairs firstTenPrimes)).toRat = 4 / 5 :=
  sorry

end probability_even_sum_l158_158252


namespace mode_of_reading_time_l158_158016

-- Define the reading times and their frequencies
def reading_times : List ℕ := [7, 8, 9, 10]
def frequencies : List ℕ := [4, 12, 13, 6]

-- Define a function to find the mode
def mode (times : List ℕ) (counts : List ℕ) : ℕ :=
  times[counts.indexOf (List.maximum counts).getD 0]

-- State the theorem
theorem mode_of_reading_time : mode reading_times frequencies = 9 :=
by
  -- This is a statement, the proof will be provided later
  sorry

end mode_of_reading_time_l158_158016


namespace smaller_investment_value_l158_158101

theorem smaller_investment_value :
  ∃ (x : ℝ), 0.07 * x + 0.27 * 1500 = 0.22 * (x + 1500) ∧ x = 500 :=
by
  sorry

end smaller_investment_value_l158_158101


namespace smallest_N_l158_158414

/--
This statement proves that given four distinct natural numbers with the pairwise greatest common divisors being 1, 2, 3, 4, 5, and N where N > 5, the smallest possible value of N is 14.
-/
theorem smallest_N (a b c d : ℕ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (gcds_1_2_3_4_5_N : (∃ N : ℕ, setOf (gcd a b) ∪ setOf (gcd a c) ∪ setOf (gcd a d)
      ∪ setOf (gcd b c) ∪ setOf (gcd b d) ∪ setOf (gcd c d) = {1, 2, 3, 4, 5, N} ∧ N > 5)) :
  ∃ N, N = 14 := sorry

end smallest_N_l158_158414


namespace sum_of_ages_is_22_l158_158001

noncomputable def Ashley_Age := 8
def Mary_Age (M : ℕ) := 7 * Ashley_Age = 4 * M

theorem sum_of_ages_is_22 (M : ℕ) (h : Mary_Age M):
  Ashley_Age + M = 22 :=
by
  -- skipping proof details
  sorry

end sum_of_ages_is_22_l158_158001


namespace lcm_144_132_eq_1584_l158_158142

theorem lcm_144_132_eq_1584 :
  Nat.lcm 144 132 = 1584 :=
by
  sorry

end lcm_144_132_eq_1584_l158_158142


namespace arccos_one_over_sqrt_two_eq_pi_over_four_l158_158932

theorem arccos_one_over_sqrt_two_eq_pi_over_four : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_eq_pi_over_four_l158_158932


namespace probability_even_sum_l158_158249

def firstTenPrimes : Set ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

def isOdd (n : ℕ) : Prop := n % 2 = 1

def distinctPairs (s : Set ℕ) : Set (ℕ × ℕ) :=
  { (x, y) | x ∈ s ∧ y ∈ s ∧ x ≠ y }

def eventEvenSum (s : Set ℕ) : Set (ℕ × ℕ) :=
  { (x, y) | x ∈ s ∧ y ∈ s ∧ x ≠ y ∧ (x + y) % 2 = 0 }

theorem probability_even_sum :
  (Set.card (eventEvenSum firstTenPrimes)).toRat / (Set.card (distinctPairs firstTenPrimes)).toRat = 4 / 5 :=
  sorry

end probability_even_sum_l158_158249


namespace xy_sum_eq_16_l158_158617

theorem xy_sum_eq_16 (x y : ℕ) (h1: x > 0) (h2: y > 0) (h3: x < 20) (h4: y < 20) (h5: x + y + x * y = 76) : x + y = 16 :=
  sorry

end xy_sum_eq_16_l158_158617


namespace incorrect_arrangements_of_good_l158_158625

theorem incorrect_arrangements_of_good : 
  let total_permutations := (factorial 4) / (factorial 2 * factorial 1 * factorial 1) in
  (total_permutations - 1 = 11) :=
by
  sorry

end incorrect_arrangements_of_good_l158_158625


namespace lauren_time_8_miles_l158_158329

-- Conditions
def time_alex_run_6_miles : ℕ := 36
def time_lauren_run_5_miles : ℕ := time_alex_run_6_miles / 3
def time_per_mile_lauren : ℚ := time_lauren_run_5_miles / 5

-- Proof statement
theorem lauren_time_8_miles : 8 * time_per_mile_lauren = 19.2 := by
  sorry

end lauren_time_8_miles_l158_158329


namespace proof_EF_sum_l158_158316

noncomputable def E : ℕ := 1
noncomputable def F : ℕ := 0

theorem proof_EF_sum : E + F = 1 :=
by {
  -- Given conditions
  have h1: E = 1 := by rfl,
  have h2: F = 0 := by rfl,
  -- Prove E + F = 1
  rw [h1, h2],
  exact rfl
}

end proof_EF_sum_l158_158316


namespace sum_fib_sum_fib_even_sum_fib_odd_l158_158112

/-- Define the Fibonacci sequence. -/
def fib : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fib (n+1) + fib n

/-- Prove that the sum of the first n Fibonacci numbers is equal to the (n+2)th Fibonacci number minus 1. -/
theorem sum_fib (n : ℕ) : (∑ i in Finset.range (n + 1), fib i) = fib (n + 2) - 1 := 
sorry

/-- Prove that the sum of the even-indexed Fibonacci numbers up to n is equal to the (2n+1)th Fibonacci number. -/
theorem sum_fib_even (n : ℕ) : (∑ i in Finset.range (n + 1), fib (2 * i)) = fib (2 * n + 1) := 
sorry

/-- Prove that the sum of the odd-indexed Fibonacci numbers up to n is equal to the (2n+2)th Fibonacci number. -/
theorem sum_fib_odd (n : ℕ) : (∑ i in Finset.range (n + 1), fib (2 * i + 1)) = fib (2 * n + 2) := 
sorry

end sum_fib_sum_fib_even_sum_fib_odd_l158_158112


namespace length_AD_l158_158653

variables {A B C D E : Type}
variables (d : D) (a b c e : E)
variables [Inhabited D] [Inhabited E]

def midpoint (A D : E) : E := sorry -- Define midpoint
def perpendicular (A C B E : E) : Prop := sorry -- Define perpendicular

axiom AB_eq_150 : ∀ {A B : E}, AB = 150
axiom E_mid_AD : midpoint A_D = E
axiom AC_BE_perpendicular : perpendicular A C B E

theorem length_AD : 
  ∀ {A B C D E : E}, (AB = 150) → 
  (midpoint A D = E) → 
  (perpendicular A C B E) → 
  AD = 150*sqrt(2) :=
by
  intro A B C D E AB_eq_150 E_mid_AD AC_BE_perpendicular
  sorry

end length_AD_l158_158653


namespace g_9_pow_4_l158_158740

theorem g_9_pow_4 (f g : ℝ → ℝ) (h1 : ∀ x ≥ 1, f (g x) = x^2) (h2 : ∀ x ≥ 1, g (f x) = x^4) (h3 : g 81 = 81) : (g 9)^4 = 81 :=
sorry

end g_9_pow_4_l158_158740


namespace convex_quad_inequality_l158_158173

variables {A B C D E : Type} [innocent : Geometry E]

-- Definitions: E is the midpoint of AB
def midpoint (E A B : E) : Prop :=
  ∃ M : E, E = M ∧ A = M + M ∧ B = M + M

-- Definitions: Quadrilateral is convex
def convex_quadrilateral (A B C D : E) : Prop :=
  ∃ P: E, triangle A B C ∧ same_side D A B

-- Definitions: Angle BCD = 90 degrees
def right_angle (B C D : E) : Prop :=
  ∠ B C D = 90

theorem convex_quad_inequality
  (hconvex : convex_quadrilateral A B C D)
  (hright : right_angle B C D)
  (hmiddle : midpoint E A B) : 
  2 * distance E C ≤ distance A D + distance B D :=
by
  sorry

end convex_quad_inequality_l158_158173


namespace minimum_value_of_A_l158_158212

open Real

noncomputable def A (x y z : ℝ) : ℝ := cos (x - y) + cos (y - z) + cos (z - x)

theorem minimum_value_of_A :
  ∃ x y z ∈ Icc 0 (π / 2), A x y z ≥ 1 ∧ (∀ x' y' z' ∈ Icc 0 (π / 2), A x' y' z' ≥ A x y z) := sorry

end minimum_value_of_A_l158_158212


namespace compute_f_zero_l158_158615

variable (f g : ℝ → ℝ)

theorem compute_f_zero
  (h1 : ∀ x, g x = 2 - x^3)
  (h2 : ∀ x, f (g x) = x^3 - 2 * x) :
  f 0 = 2 - 2 * real.cbrt 2 :=
sorry

end compute_f_zero_l158_158615


namespace ratio_of_B_to_C_l158_158030

theorem ratio_of_B_to_C (A B C : ℕ) (h1 : A + B + C = 98) (h2 : (A : ℚ) / B = 2 / 3) (h3 : B = 30) : ((B : ℚ) / C) = 5 / 8 :=
by
  sorry

end ratio_of_B_to_C_l158_158030


namespace algebraic_expression_evaluation_l158_158564

-- Given condition and goal statement
theorem algebraic_expression_evaluation (a b : ℝ) (h : a - 2 * b + 3 = 0) : 5 + 2 * b - a = 8 :=
by sorry

end algebraic_expression_evaluation_l158_158564


namespace probability_even_sum_l158_158268

-- Definition of the prime numbers set and the selection scenario
def firstTenPrimes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Condition stating that we select two distinct numbers at random from the first ten primes
def randomSelection (s : List ℕ) := {x : ℕ × ℕ // x.1 ≠ x.2 ∧ x.1 ∈ s ∧ x.2 ∈ s}

-- Define the event that their sum is even
def evenSum (p : ℕ × ℕ) := (p.1 + p.2) % 2 = 0

-- Define the probability calculation
def probabilityEvenSum := 
  let totalPairs := (firstTenPrimes.length.choose 2) -- Calculate the number of ways to choose 2 numbers from the 10 primes
  let evenPairs := (randomSelection firstTenPrimes).count (λ p => evenSum p.val)
  evenPairs.toFloat / totalPairs.toFloat

-- Theorem statement that the probability of their sum being even is 1/5
theorem probability_even_sum : probabilityEvenSum = 1/5 := by
  sorry -- The actual proof is to be provided

end probability_even_sum_l158_158268


namespace expectation_S_tau_eq_varliminf_ratio_S_tau_l158_158335

noncomputable def xi : ℕ → ℝ := sorry
noncomputable def tau : ℝ := sorry

-- Statement (a)
theorem expectation_S_tau_eq (ES_tau : ℝ := sorry) (E_tau : ℝ := sorry) (E_xi1 : ℝ := sorry) :
  ES_tau = E_tau * E_xi1 := sorry

-- Statement (b)
theorem varliminf_ratio_S_tau (liminf_val : ℝ := sorry) (E_tau : ℝ := sorry) :
  (liminf_val = E_tau) := sorry

end expectation_S_tau_eq_varliminf_ratio_S_tau_l158_158335


namespace arccos_identity_l158_158917

theorem arccos_identity : 
  ∀ x : ℝ, x = 1 / real.sqrt 2 → real.arccos x = real.pi / 4 :=
begin
  intros x hx,
  have h1 : real.cos (real.pi / 4) = 1 / real.sqrt 2 := by sorry,
  have h2 : real.arccos (1 / real.sqrt 2) = real.pi / 4,
  { rw ← h1,
    rw real.arccos_cos,
    exact_mod_cast hx },
  exact h2,
end

end arccos_identity_l158_158917


namespace express_f12_in_terms_of_a_l158_158239

variable {f : ℝ → ℝ}
variable {a : ℝ}
variable (f_add : ∀ x y : ℝ, f (x + y) = f x + f y)
variable (f_neg_three : f (-3) = a)

theorem express_f12_in_terms_of_a : f 12 = -4 * a := sorry

end express_f12_in_terms_of_a_l158_158239


namespace parabola_directrix_distance_l158_158840

noncomputable def parabola_properties : Prop :=
  let ellipse := ∀ (x y : ℝ), 4 * x^2 + y^2 = 1 in
  let focus_1 := (0, sqrt(3) / 2) in
  let focus_2 := (0, -sqrt(3) / 2) in
  let vertex := (0,0) in
  let distance_focus_directrix (focus vertex : ℝ × ℝ) : ℝ := 2 * abs(focus.2 - vertex.2) in
  let focus := focus_1 in 
  distance_focus_directrix focus vertex = sqrt(3)

theorem parabola_directrix_distance : parabola_properties := sorry

end parabola_directrix_distance_l158_158840


namespace simplify_and_evaluate_l158_158729

-- Define the given expression
noncomputable def given_expression (m : ℝ) : ℝ :=
  (m - (m + 9) / (m + 1)) / ((m^2) + 3 * m) / (m + 1)

-- Define the condition
def condition (m : ℝ) : Prop :=
  m = Real.sqrt 3

-- Define the correct answer
def correct_answer : ℝ :=
  1 - Real.sqrt 3

-- State the theorem
theorem simplify_and_evaluate 
  (m : ℝ) (h : condition m) : 
  given_expression m = correct_answer := by
  sorry

end simplify_and_evaluate_l158_158729


namespace annabelle_savings_l158_158865

theorem annabelle_savings:
  let allowance := 30 in
  let junk_food := 0.4 * allowance in
  let sweets := 8 in
  let spent := junk_food + sweets in
  let remaining_after_spent := allowance - spent in
  let donation := 0.05 * remaining_after_spent in
  let savings := remaining_after_spent - donation in
  savings = 9.50 :=
by
  let allowance := 30
  let junk_food := 0.4 * allowance
  let sweets := 8
  let spent := junk_food + sweets
  let remaining_after_spent := allowance - spent
  let donation := 0.05 * remaining_after_spent
  let savings := remaining_after_spent - donation
  have h1: junk_food = 12 := by sorry
  have h2: spent = 20 := by sorry
  have h3: remaining_after_spent = 10 := by sorry
  have h4: donation = 0.50 := by sorry
  have h5: savings = 9.50 := by sorry
  show savings = 9.50 from h5

end annabelle_savings_l158_158865


namespace arccos_one_over_sqrt_two_l158_158980

theorem arccos_one_over_sqrt_two :
  Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_l158_158980


namespace arccos_one_over_sqrt_two_l158_158952

theorem arccos_one_over_sqrt_two : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_l158_158952


namespace additional_cost_l158_158118

theorem additional_cost (original_bales : ℕ) (previous_cost_per_bale : ℕ) (better_quality_cost_per_bale : ℕ) :
  let new_bales := original_bales * 2
  let original_cost := original_bales * previous_cost_per_bale
  let new_cost := new_bales * better_quality_cost_per_bale
  new_cost - original_cost = 210 :=
by sorry

end additional_cost_l158_158118


namespace neil_initial_games_l158_158223

variable {N : ℕ}
variable h_games_initial : ℕ := 33
variable h_games_given : ℕ := 5
variable h_games_after : ℕ := h_games_initial - h_games_given
variable h_games_mul : ℕ := 4
variable neil_games_after : ℕ := N + h_games_given

theorem neil_initial_games (h : h_games_after = h_games_mul * neil_games_after) : N = 2 :=
sorry

end neil_initial_games_l158_158223


namespace jake_total_work_hours_l158_158667

def initial_debt_A := 150
def payment_A := 60
def hourly_rate_A := 15
def remaining_debt_A := initial_debt_A - payment_A
def hours_to_work_A := remaining_debt_A / hourly_rate_A

def initial_debt_B := 200
def payment_B := 80
def hourly_rate_B := 20
def remaining_debt_B := initial_debt_B - payment_B
def hours_to_work_B := remaining_debt_B / hourly_rate_B

def initial_debt_C := 250
def payment_C := 100
def hourly_rate_C := 25
def remaining_debt_C := initial_debt_C - payment_C
def hours_to_work_C := remaining_debt_C / hourly_rate_C

def total_hours_to_work := hours_to_work_A + hours_to_work_B + hours_to_work_C

theorem jake_total_work_hours :
  total_hours_to_work = 18 :=
sorry

end jake_total_work_hours_l158_158667


namespace find_value_of_n_l158_158850

theorem find_value_of_n (n : ℕ) 
  (total_students : ℕ) 
  (boys_to_girls_ratio : 6:5) 
  (sample_size : ℕ := total_students / 10) 
  (sample_contains_12_more_boys_than_girls : 
     ∃ m, m = 𝑛 ∧ (6/11 * sample_size) - (5/11 * sample_size) = 12) :
  n = 1320 :=
sorry

end find_value_of_n_l158_158850


namespace find_AB_dot_BC_l158_158299

variables {A B C : Type} [EuclideanSpace ℝ A] [EuclideanSpace ℝ B] [EuclideanSpace ℝ C]

noncomputable def triangle_ABC (a b c : ℝ) (cosB cosC : ℝ) : Prop :=
  (2 * a - c) * cosB = b * cosC

noncomputable def vector_dot_product_AB_BC (a b c : ℝ) (cosB : ℝ) : ℝ :=
  -(a * c * cosB)

theorem find_AB_dot_BC :
  ∀ (a b c : ℝ) (cosB cosC : ℝ),
    a = 2 →
    c = 3 →
    triangle_ABC a b c cosB cosC →
    vector_dot_product_AB_BC a b c cosB = -3 :=
by
  sorry

end find_AB_dot_BC_l158_158299


namespace find_m_l158_158206

noncomputable def f (x m : ℝ) : ℝ := x ^ 2 + m
noncomputable def g (x : ℝ) : ℝ := 6 * Real.log x - 4 * x

theorem find_m (m : ℝ) : 
  ∃ a b : ℝ, (0 < a) ∧ (f a m = b) ∧ (g a = b) ∧ (2 * a = (6 / a) - 4) → m = -5 := 
by
  sorry

end find_m_l158_158206


namespace label_edges_with_gcd_one_l158_158741

def Graph (V : Type*) := (V → V → Prop)

def is_connected {V : Type*} (G : Graph V) : Prop :=
  ∀ (v w : V), v ≠ w → ∃ (p : list V), p.head = some v ∧ p.last = some w ∧ ∀ (u v : V), (u, v) ∈ list.zip p (p.tail) → G u v

def num_edges {V : Type*} (G : Graph V) : ℕ := sorry -- definition of number of edges

def label_edges {V : Type*} (G : Graph V) (labels : list ℕ) : Prop :=
  ∀ (v : V), (∃ (e1 e2 : V), e1 ≠ e2 ∧ G v e1 ∧ G v e2 ∧ nat.gcd (labels.nth e1) (labels.nth e2) = 1) 

theorem label_edges_with_gcd_one {V : Type*} (G : Graph V) (k : ℕ) (h_conn : is_connected G) (h_num_edges : num_edges G = k) :
  ∃ (labels : list ℕ), label_edges G labels := 
begin
  sorry
end

end label_edges_with_gcd_one_l158_158741


namespace average_of_16_consecutive_odds_l158_158035

theorem average_of_16_consecutive_odds (a : ℕ) (x : ℕ) (h1 : a = 399)
  (h2 : 16 = x) : ((2 * (399 + 15) + 399) / 2) = 414 :=
by
  -- These hypotheses match the conditions about the sequence
  have L : ℕ := 399 + 2 * (16 - 1)
  have A : ℚ := (399 + L) / 2
  have H : L = 429 := by
    calc
      L = 399 + 2 * 15 : by rw [show 16 - 1 = 15, by norm_num]
        _ = 399 + 30 : by norm_num
        _ = 429 : by norm_num
  have H2 : A = 414 := by
    calc
      A = (399 + 429 : ℚ) / 2 : by sorry
        _ = 828 / 2 : by norm_num
        _ = 414 : by norm_num
  exact H2

end average_of_16_consecutive_odds_l158_158035


namespace sequence_inequality_l158_158185

-- Define the problem
theorem sequence_inequality (a : ℕ → ℕ) (h0 : ∀ n, 0 < a n) (h1 : a 1 > a 0) (h2 : ∀ n ≥ 2, a n = 3 * a (n-1) - 2 * a (n-2)) : a 100 > 2^99 :=
by
  sorry

end sequence_inequality_l158_158185


namespace arnolds_total_protein_l158_158868

theorem arnolds_total_protein (collagen_protein_per_two_scoops : ℕ) (protein_per_scoop : ℕ) 
    (steak_protein : ℕ) (scoops_of_collagen : ℕ) (scoops_of_protein : ℕ) :
    collagen_protein_per_two_scoops = 18 →
    protein_per_scoop = 21 →
    steak_protein = 56 →
    scoops_of_collagen = 1 →
    scoops_of_protein = 1 →
    (collagen_protein_per_two_scoops / 2 * scoops_of_collagen + protein_per_scoop * scoops_of_protein + steak_protein = 86) :=
by
  intros hc p s sc sp
  sorry

end arnolds_total_protein_l158_158868


namespace geo_sequence_sum_l158_158658

theorem geo_sequence_sum (a : ℕ → ℝ) (q : ℝ)
  (h1 : a 1 + a 2 = 2)
  (h2 : a 4 + a 5 = 4)
  (h_geo : ∀ n, a (n + 1) = q * a n) :
  a 10 + a 11 = 16 := by
  -- Insert proof here
  sorry  -- skipping the proof

end geo_sequence_sum_l158_158658


namespace non_zero_number_is_nine_l158_158750

theorem non_zero_number_is_nine {x : ℝ} (h1 : (x + x^2) / 2 = 5 * x) (h2 : x ≠ 0) : x = 9 :=
by
  sorry

end non_zero_number_is_nine_l158_158750


namespace operation_proof_l158_158525

def operation (x y : ℤ) : ℤ := x * y - 3 * x - 4 * y

theorem operation_proof : (operation 7 2) - (operation 2 7) = 5 :=
by
  sorry

end operation_proof_l158_158525


namespace solve_x_if_alpha_beta_eq_8_l158_158559

variable (x : ℝ)

def alpha (x : ℝ) := 4 * x + 9
def beta (x : ℝ) := 9 * x + 6

theorem solve_x_if_alpha_beta_eq_8 (hx : alpha (beta x) = 8) : x = (-25 / 36) :=
by
  sorry

end solve_x_if_alpha_beta_eq_8_l158_158559


namespace arccos_one_over_sqrt_two_l158_158976

theorem arccos_one_over_sqrt_two :
  Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_l158_158976


namespace length_EF_l158_158331

structure Parallelogram where
  A B C D E F : Type
  AB CD BC DA : ℝ
  AC BD : ℝ
  DF : ℝ
  AB_eq_CD : AB = 15
  BC_eq_DA : BC = 20
  AC_length : AC = 25
  diagonals_perpendicular : True
  E_mid_BD : True
  DF_length : DF = 5

theorem length_EF (p : Parallelogram) : sqrt ((15 - 5)^2 + (25 / 2)^2) = 16.closestIntApprox 1 :=
by sorry

end length_EF_l158_158331


namespace math_problem_proof_l158_158607

-- Definitions based on conditions
def is_square_root (a b c : ℚ) : Prop :=
  b ≠ a ∧ a^2 = c ∧ b^2 = c

def integer_part_of_sqrt_10 (a : ℚ) : Prop :=
  ∃ n : ℤ, n = Int.floor 3 ∧ a = n

-- Prove in the Lean statement
theorem math_problem_proof :
  ∃ x y m : ℚ, is_square_root (2*x - 1) (4*x + 3) m ∧
  integer_part_of_sqrt_10 (2*y + 2) ∧ 
  x = -1/3 ∧ y = 1/2 ∧ m = 25/9 ∧ 
  (1 + 4*y) = 3 ∧ (1 + 4*y)^(1/2) ∈ { √3, -√3 } :=
by
  sorry

end math_problem_proof_l158_158607


namespace problem_evaluation_l158_158152

theorem problem_evaluation (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a + b + c + d)⁻¹ * (a⁻¹ + b⁻¹ + c⁻¹ + d⁻¹) * (ab + bc + cd + da + ac + bd)⁻¹ * ((ab)⁻¹ + (bc)⁻¹ + (cd)⁻¹ + (da)⁻¹ + (ac)⁻¹ + (bd)⁻¹) = 
  (1 / (a * b * c * d)) * (1 / (a * b * c * d)) :=
by
  sorry

end problem_evaluation_l158_158152


namespace probability_even_sum_l158_158272

-- Definition of the prime numbers set and the selection scenario
def firstTenPrimes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Condition stating that we select two distinct numbers at random from the first ten primes
def randomSelection (s : List ℕ) := {x : ℕ × ℕ // x.1 ≠ x.2 ∧ x.1 ∈ s ∧ x.2 ∈ s}

-- Define the event that their sum is even
def evenSum (p : ℕ × ℕ) := (p.1 + p.2) % 2 = 0

-- Define the probability calculation
def probabilityEvenSum := 
  let totalPairs := (firstTenPrimes.length.choose 2) -- Calculate the number of ways to choose 2 numbers from the 10 primes
  let evenPairs := (randomSelection firstTenPrimes).count (λ p => evenSum p.val)
  evenPairs.toFloat / totalPairs.toFloat

-- Theorem statement that the probability of their sum being even is 1/5
theorem probability_even_sum : probabilityEvenSum = 1/5 := by
  sorry -- The actual proof is to be provided

end probability_even_sum_l158_158272


namespace second_solution_percentage_l158_158805

theorem second_solution_percentage (P : ℝ) : 
  (28 * 0.30 + 12 * P = 40 * 0.45) → P = 0.8 :=
by
  intros h
  sorry

end second_solution_percentage_l158_158805


namespace spider_paths_l158_158084

theorem spider_paths : 
  let up_steps := 5 in
  let right_steps := 6 in
  let total_steps := up_steps + right_steps in
  (Nat.choose total_steps up_steps = 462) :=
by
  let up_steps := 5
  let right_steps := 6
  let total_steps := up_steps + right_steps
  show Nat.choose total_steps up_steps = 462
  exact sorry

end spider_paths_l158_158084


namespace neither_sufficient_nor_necessary_condition_l158_158566

-- Given conditions
def p (a : ℝ) : Prop := ∃ (x y : ℝ), a * x + y + 1 = 0 ∧ a * x - y + 2 = 0
def q : Prop := ∃ (a : ℝ), a = 1

-- The proof problem
theorem neither_sufficient_nor_necessary_condition : 
  ¬ ((∀ a, p a → q) ∧ (∀ a, q → p a)) :=
sorry

end neither_sufficient_nor_necessary_condition_l158_158566


namespace find_fraction_l158_158619

theorem find_fraction
  (w x y F : ℝ)
  (h1 : 5 / w + F = 5 / y)
  (h2 : w * x = y)
  (h3 : (w + x) / 2 = 0.5) :
  F = 10 := 
sorry

end find_fraction_l158_158619


namespace extreme_values_a_quarter_range_of_a_l158_158597

noncomputable def f (x a : ℝ) : ℝ := log (x + 1) + a * x ^ 2 - x

-- Part 1: Prove the extreme values for a = 1/4
theorem extreme_values_a_quarter :
  ∀ x : ℝ, x = 0 → f x (1 / 4) = 0 ∧
           ∀ x : ℝ, x = 1 → f x (1 / 4) = Real.log 2 - 3 / 4 :=
by sorry

-- Part 2: Prove the range of values for a
theorem range_of_a :
  ∀ b : ℝ, b ∈ Ioo 1 2 →
    (∀ x ∈ Ioc (-1) b, f x a ≤ f b a) ↔ 1 - Real.log 2 ≤ a :=
by sorry

end extreme_values_a_quarter_range_of_a_l158_158597


namespace max_ab_bc_cd_da_l158_158337

theorem max_ab_bc_cd_da (a b c d : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : 0 ≤ d) (h₄ : a + b + c + d = 200) :
  ab + bc + cd + da ≤ 10000 :=
sorry

end max_ab_bc_cd_da_l158_158337


namespace pairs_count_l158_158674

theorem pairs_count (A B : Set ℕ) (h1 : A ∪ B = {1, 2, 3, 4, 5}) (h2 : 3 ∈ A ∩ B) : 
  Nat.card {p : Set ℕ × Set ℕ | p.1 ∪ p.2 = {1, 2, 3, 4, 5} ∧ 3 ∈ p.1 ∩ p.2} = 81 := by
  sorry

end pairs_count_l158_158674


namespace quadratic_equation_real_roots_l158_158631

theorem quadratic_equation_real_roots (k : ℝ) : 
  (∃ x : ℝ, k * x^2 - 6 * x + 9 = 0) ↔ (k ≤ 1 ∧ k ≠ 0) :=
by
  sorry

end quadratic_equation_real_roots_l158_158631


namespace license_plates_count_correct_l158_158110

/-- Calculate the number of five-character license plates. -/
def count_license_plates : Nat :=
  let num_consonants := 20
  let num_vowels := 6
  let num_digits := 10
  num_consonants^2 * num_vowels^2 * num_digits

theorem license_plates_count_correct :
  count_license_plates = 144000 :=
by
  sorry

end license_plates_count_correct_l158_158110


namespace find_unknown_rate_l158_158060

variable (x : ℕ)

theorem find_unknown_rate
    (c3 : ℕ := 3 * 100)
    (c5 : ℕ := 5 * 150)
    (n : ℕ := 10)
    (avg_price : ℕ := 160) 
    (h : c3 + c5 + 2 * x = avg_price * n) :
    x = 275 := 
by
  -- Proof goes here.
  sorry

end find_unknown_rate_l158_158060


namespace geometric_series_q_and_S6_l158_158660

theorem geometric_series_q_and_S6 (a : ℕ → ℝ) (q : ℝ) (S_6 : ℝ) 
  (ha_pos : ∀ n, a n > 0)
  (ha2 : a 2 = 3)
  (ha4 : a 4 = 27) :
  q = 3 ∧ S_6 = 364 :=
by
  sorry

end geometric_series_q_and_S6_l158_158660


namespace most_lines_of_symmetry_circle_l158_158799

-- Define the figures and their lines of symmetry
def regular_pentagon_lines_of_symmetry : ℕ := 5
def isosceles_triangle_lines_of_symmetry : ℕ := 1
def circle_lines_of_symmetry : ℕ := 0  -- Representing infinite lines of symmetry in Lean is unconventional; we'll use a special case.
def regular_hexagon_lines_of_symmetry : ℕ := 6
def ellipse_lines_of_symmetry : ℕ := 2

-- Define a predicate to check if one figure has more lines of symmetry than all others
def most_lines_of_symmetry {α : Type} [LinearOrder α] (f : α) (others : List α) : Prop :=
  ∀ x ∈ others, f ≥ x

-- Define the problem statement in Lean
theorem most_lines_of_symmetry_circle :
  most_lines_of_symmetry circle_lines_of_symmetry [
    regular_pentagon_lines_of_symmetry,
    isosceles_triangle_lines_of_symmetry,
    regular_hexagon_lines_of_symmetry,
    ellipse_lines_of_symmetry ] :=
by {
  -- To represent infinite lines, we consider 0 as a larger "dummy" number in this context,
  -- since in Lean we don't have a built-in representation for infinity in finite ordering.
  -- Replace with a suitable model if necessary.
  sorry
}

end most_lines_of_symmetry_circle_l158_158799


namespace older_friend_is_38_l158_158006

-- Define the conditions
def younger_friend_age (x : ℕ) : Prop := 
  ∃ (y : ℕ), (y = x + 2 ∧ x + y = 74)

-- Define the age of the older friend
def older_friend_age (x : ℕ) : ℕ := x + 2

-- State the theorem
theorem older_friend_is_38 : ∃ x, younger_friend_age x ∧ older_friend_age x = 38 :=
by
  sorry

end older_friend_is_38_l158_158006


namespace largest_root_range_l158_158123

-- Define the polynomial equation
def polynomial_eq (x b3 b2 b1 b0 : ℝ) : Prop :=
  x^4 + b3 * x^3 + b2 * x^2 + b1 * x + b0 = 0

-- Define the coefficient constraints
def valid_coefficient (b : ℝ) : Prop :=
  abs(b) < 3

-- Define the condition for all coefficients
def valid_coefficients (b3 b2 b1 b0 : ℝ) : Prop :=
  valid_coefficient b3 ∧ valid_coefficient b2 ∧ valid_coefficient b1 ∧ valid_coefficient b0

-- The main theorem to prove
theorem largest_root_range (b3 b2 b1 b0 : ℝ) (h : valid_coefficients b3 b2 b1 b0) :
  ∃ s : ℝ, (polynomial_eq s b3 b2 b1 b0) ∧ (3 < s) ∧ (s < 4) :=
sorry

end largest_root_range_l158_158123


namespace quadrilateral_is_rhombus_l158_158384

-- Definitions of basic geometric entities and properties
structure Point (α : Type) := 
  (x y : α)

structure Triangle (α : Type) :=
  (a b c : Point α)

structure Quadrilateral (α : Type) :=
  (a b c d : Point α)

def perimeter {α : Type} [field α] (t : Triangle α) : α :=
  (dist t.a t.b) + (dist t.b t.c) + (dist t.c t.a)

def equal_perimeter {α : Type} [field α] (q : Quadrilateral α) : Prop :=
  let tri1 := Triangle.mk q.a q.b q.d in
  let tri2 := Triangle.mk q.b q.c q.d in
  let tri3 := Triangle.mk q.a q.c q.d in
  let tri4 := Triangle.mk q.a q.b q.c in
  (perimeter tri1 = perimeter tri2) ∧ 
  (perimeter tri2 = perimeter tri3) ∧
  (perimeter tri3 = perimeter tri4)

-- Statement to be proved: A quadrilateral with equal perimeter triangles formed by its diagonals is a rhombus.
theorem quadrilateral_is_rhombus {α : Type} [field α] (q : Quadrilateral α) 
  (h : equal_perimeter q) : 
  is_rhombus q :=
sorry

end quadrilateral_is_rhombus_l158_158384


namespace cost_of_black_and_white_drawing_l158_158328

-- Given the cost of the color drawing is 1.5 times the cost of the black and white drawing
-- and John paid $240 for the color drawing, we need to prove the cost of the black and white drawing is $160.

theorem cost_of_black_and_white_drawing (C : ℝ) (h : 1.5 * C = 240) : C = 160 :=
by
  sorry

end cost_of_black_and_white_drawing_l158_158328


namespace min_value_expr_min_max_value_expr_max_l158_158232

noncomputable def min_value_expr (a b : ℝ) : ℝ := 
  1 / (a - b) + 4 / (b - 1)

noncomputable def max_value_expr (a b : ℝ) : ℝ :=
  a * b - b^2 - a + b

theorem min_value_expr_min (a b : ℝ) (h1 : a > b) (h2 : b > 1) (h3 : a + 3 * b = 5) : 
  min_value_expr a b = 25 :=
sorry

theorem max_value_expr_max (a b : ℝ) (h1 : a > b) (h2 : b > 1) (h3 : a + 3 * b = 5) :
  max_value_expr a b = 1 / 16 :=
sorry

end min_value_expr_min_max_value_expr_max_l158_158232


namespace stationary_train_length_l158_158858

noncomputable def speed_train_kmh : ℝ := 144
noncomputable def speed_train_ms : ℝ := (speed_train_kmh * 1000) / 3600
noncomputable def time_to_pass_pole : ℝ := 8
noncomputable def time_to_pass_stationary : ℝ := 18
noncomputable def length_moving_train : ℝ := speed_train_ms * time_to_pass_pole
noncomputable def total_distance : ℝ := speed_train_ms * time_to_pass_stationary
noncomputable def length_stationary_train : ℝ := total_distance - length_moving_train

theorem stationary_train_length :
  length_stationary_train = 400 := by
  sorry

end stationary_train_length_l158_158858


namespace remaining_pie_after_carlos_and_maria_l158_158879

theorem remaining_pie_after_carlos_and_maria (C M R : ℝ) (hC : C = 0.60) (hM : M = 0.25 * (1 - C)) : R = 1 - C - M → R = 0.30 :=
by
  intro hR
  simp only [hC, hM] at hR
  sorry

end remaining_pie_after_carlos_and_maria_l158_158879


namespace probability_of_even_sum_is_four_fifths_l158_158263

-- Define the first ten prime numbers
def first_ten_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- The function to calculate the probability that the sum of two distinct primes is even
def probability_even_sum (primes : List ℕ) : ℚ :=
  let pairs := primes.product primes
  let distinct_pairs := pairs.filter (λ p, p.1 ≠ p.2)
  let even_sum_pairs := distinct_pairs.filter (λ p, (p.1 + p.2) % 2 = 0)
  (even_sum_pairs.length : ℚ) / (distinct_pairs.length : ℚ)

-- Prove that the probability is 4/5
theorem probability_of_even_sum_is_four_fifths :
  probability_even_sum first_ten_primes = 4 / 5 := sorry

end probability_of_even_sum_is_four_fifths_l158_158263


namespace probability_red_ball_l158_158149

noncomputable def event_A : Type := "a red ball is drawn"

def boxType := {H1 | H2 | H3: Prop}

def prob_H1 : ℝ := 2/5
def prob_H2 : ℝ := 2/5
def prob_H3 : ℝ := 1/5

def prob_A_given_H1 : ℝ := 4/10
def prob_A_given_H2 : ℝ := 2/10
def prob_A_given_H3 : ℝ := 8/10

theorem probability_red_ball :
  let P (A : event_A) :=
    prob_H1 * prob_A_given_H1 +
    prob_H2 * prob_A_given_H2 +
    prob_H3 * prob_A_given_H3 in
  P(A) = 0.4 :=
  by
  -- Proof omitted
  sorry

end probability_red_ball_l158_158149


namespace club_members_l158_158480

variable (x : ℕ)

theorem club_members (h1 : 2 * x + 5 = x + 15) : x = 10 := by
  sorry

end club_members_l158_158480


namespace perp_bisector_slope_l158_158576

-- Define the points A and B
structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := 1, y := 2 }
def B : Point := { x := 3, y := 1 }

-- Define slope calculation for a line segment
def slope (P Q : Point) : ℝ :=
  (Q.y - P.y) / (Q.x - P.x)

-- Define the perpendicular slope calculation, the negative reciprocal
def perp_slope (m : ℝ) : ℝ := -1 / m

-- The theorem statement
theorem perp_bisector_slope :
  perp_slope (slope A B) = 2 := by
  sorry

end perp_bisector_slope_l158_158576


namespace problem_solution_l158_158519

theorem problem_solution :
  (2023 * Real.pi)^0 + (-1 / 2)^(-1:ℤ) + |1 - Real.sqrt 3| - 2 * Real.sin (Real.pi / 3) = -2 :=
by
  -- Mathematical details given in conditions can be expressed here
  have h1 : (2023 * Real.pi)^0 = 1, from pow_zero (2023 * Real.pi),
  have h2 : (-1 / 2)^(-1:ℤ) = -2, from rfl,
  have h3 : |1 - Real.sqrt 3| = Real.sqrt 3 - 1, from abs_of_neg (sub_neg_of_lt (Real.sqrt_lt_of_pos 3 (by norm_num))),
  have h4 : Real.sin (Real.pi / 3) = Real.sqrt 3 / 2, from Real.sin_pi_div_three,
  -- Brings it all together
  sorry

end problem_solution_l158_158519


namespace ab_value_l158_158806

theorem ab_value (a b : ℝ) (h1 : a - b = 6) (h2 : a^2 + b^2 = 48) : a * b = 6 :=
by 
  sorry

end ab_value_l158_158806


namespace arccos_sqrt2_l158_158936

def arccos_eq (x : ℝ) := arccos x
def range_arccos := ∀ (x : ℝ), 0 ≤ arccos x ∧ arccos x ≤ π
def cos_pi_div_four : Real.cos (π / 4) = 1 / Real.sqrt 2 := by sorry

theorem arccos_sqrt2 :
  arccos_eq (1 / Real.sqrt 2) = π / 4 :=
by
  have h1 : Real.cos (π / 4) = 1 / Real.sqrt 2, from cos_pi_div_four
  sorry

end arccos_sqrt2_l158_158936


namespace find_positive_integer_solution_l158_158540

theorem find_positive_integer_solution : ∃ n : ℕ, n > 0 ∧ (1 + 3 + 5 + ... + (2 * n - 3)) / (2 + 4 + 6 + ... + 2 * n) = 7 / 8 :=
by
  -- Sum of the first (n-1) odd numbers
  have numerator_sum_formula : ∀ (n : ℕ), 1 + 3 + 5 + ... + (2 * (n-1) - 1) = (n-1)^2 :=
  sorry,

  -- Sum of the first n even numbers
  have denominator_sum_formula : ∀ (n : ℕ), 2 + 4 + 6 + ...+ 2 * n = n * (n + 1) :=
  sorry,

  -- The given condition and equation transformation
  have equation_condition : ∀ (n : ℕ), (numerator_sum_formula n) / (denominator_sum_formula n) = 7 / 8 ↔ 8 * (n-1)^2 = 7 * n * (n + 1) :=
  sorry,
  
  -- Solving the resulting quadratic equation leads to n = 23
  sorry,
end

end find_positive_integer_solution_l158_158540


namespace find_a1_and_d_l158_158574

-- Given conditions
variables {a : ℕ → ℤ} 
variables {a1 d : ℤ}

def is_arithmetic_sequence (a : ℕ → ℤ) (a1 d : ℤ) : Prop :=
∀ n : ℕ, a n = a1 + n * d

theorem find_a1_and_d 
  (h1 : is_arithmetic_sequence a a1 d)
  (h2 : (a 3) * (a 7) = -16)
  (h3 : (a 4) + (a 6) = 0)
  : (a1 = -8 ∧ d = 2) ∨ (a1 = 8 ∧ d = -2) :=
sorry

end find_a1_and_d_l158_158574


namespace necessary_but_not_sufficient_condition_l158_158818

noncomputable def is_unique_intersection (k : ℝ) : Prop :=
  ∃ x y : ℝ, y = k * x - 1 ∧ x^2 - y^2 = 4 ∧
  (∀ x' y' : ℝ, y' = k * x' - 1 ∧ x'^2 - y'^2 = 4 → (x', y') = (x, y))

theorem necessary_but_not_sufficient_condition :
  (k = ±(Real.sqrt 5 / 2) → is_unique_intersection k) ∧
  ¬(∀ k', is_unique_intersection k' → k' = ±(Real.sqrt 5 / 2)) :=
sorry

end necessary_but_not_sufficient_condition_l158_158818


namespace range_of_m_l158_158187

theorem range_of_m (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : ∀ m : ℝ, (1 / a + 1 / b) * real.sqrt (a^2 + b^2) ≥ 2 * m - 4) : 
  ∀ m : ℝ, m ≤ 2 + real.sqrt 2 := 
sorry

end range_of_m_l158_158187


namespace building_units_total_l158_158076

def num_one_bedroom_units (total_cost : ℕ) (cost_one_bedroom : ℕ) (cost_two_bedroom : ℕ) (num_two_bedroom_units : ℕ) : ℕ :=
  (total_cost - (cost_two_bedroom * num_two_bedroom_units)) / cost_one_bedroom

def total_units (num_one_bedroom_units : ℕ) (num_two_bedroom_units : ℕ) : ℕ :=
  num_one_bedroom_units + num_two_bedroom_units

theorem building_units_total (y : ℕ) (total_cost : ℕ) (cost_one_bedroom : ℕ) (cost_two_bedroom : ℕ)
    (H_y : y = 7) (H_total_cost : total_cost = 4950) (H_cost_one_bedroom : cost_one_bedroom = 360)
    (H_cost_two_bedroom : cost_two_bedroom = 450) : total_units (num_one_bedroom_units total_cost cost_one_bedroom cost_two_bedroom y) y = 12 :=
  by
    have H_num_one_bedroom_units : num_one_bedroom_units total_cost cost_one_bedroom cost_two_bedroom y = 5 := by
      rw [H_y, H_total_cost, H_cost_one_bedroom, H_cost_two_bedroom]
      simp
      sorry
    rw [H_num_one_bedroom_units, H_y]
    simp

end building_units_total_l158_158076


namespace merchant_markup_percentage_l158_158489

-- Define the conditions
def cost_price := 100
def profit_after_discount (cost_price : ℝ) (profit_percentage : ℝ) :=
  cost_price * (1 + profit_percentage / 100)
def marked_price (selling_price : ℝ) (discount_percentage : ℝ) :=
  selling_price / (1 - discount_percentage / 100)
def markup_percentage (marked_price cost_price : ℝ) :=
  (marked_price - cost_price) / cost_price * 100

-- Define the problem statement
theorem merchant_markup_percentage : 
  ∀ (cost_price profit_percentage discount_percentage : ℝ),
    profit_percentage = 35 →
    discount_percentage = 25 →
    cost_price = 100 →
    markup_percentage (marked_price (profit_after_discount cost_price profit_percentage) discount_percentage) cost_price = 80 := 
by
  intros cost_price profit_percentage discount_percentage hp_profit hp_discount hc_cost
  sorry

end merchant_markup_percentage_l158_158489


namespace hyperbola_foci_coordinates_l158_158005

theorem hyperbola_foci_coordinates :
  ∀ x y : ℝ, (x^2 / 4) - (y^2 / 12) = 1 → (x, y) = (4, 0) ∨ (x, y) = (-4, 0) :=
by
  -- We assume the given equation of the hyperbola
  intro x y h
  -- sorry is used to skip the actual proof steps
  sorry

end hyperbola_foci_coordinates_l158_158005


namespace cube_root_fraction_l158_158130

theorem cube_root_fraction : (√[3] (18 / (27 / 4)) = (2 / (3 ^ (1 / 3)))) :=
by
  sorry

end cube_root_fraction_l158_158130


namespace seven_digit_numbers_with_digit_sum_2_l158_158450

theorem seven_digit_numbers_with_digit_sum_2 : 
  ∃ (l : List ℕ), l.length = 7 ∧ (∀ d ∈ l, 0 ≤ d ∧ d ≤ 9) 
                 ∧ (l.sum = 2) ∧ (l.head ≠ 0) ∧ (l.nodup) ∧ l.permutations.length = 7 := 
sorry

end seven_digit_numbers_with_digit_sum_2_l158_158450


namespace Emily_cleaning_time_in_second_room_l158_158696

/-
Lilly, Fiona, Jack, and Emily are cleaning 3 rooms.
For the first room: Lilly and Fiona together: 1/4 of the time, Jack: 1/3 of the time, Emily: the rest of the time.
In the second room: Jack: 25%, Emily: 25%, Lilly and Fiona: the remaining 50%.
In the third room: Emily: 40%, Lilly: 20%, Jack: 20%, Fiona: 20%.
Total time for all rooms: 12 hours.

Prove that the total time Emily spent cleaning in the second room is 60 minutes.
-/

theorem Emily_cleaning_time_in_second_room :
  let total_time := 12 -- total time in hours
  let time_per_room := total_time / 3 -- time per room in hours
  let time_per_room_minutes := time_per_room * 60 -- time per room in minutes
  let emily_cleaning_percentage := 0.25 -- Emily's cleaning percentage in the second room
  let emily_cleaning_time := emily_cleaning_percentage * time_per_room_minutes -- cleaning time in minutes
  emily_cleaning_time = 60 := by
  sorry

end Emily_cleaning_time_in_second_room_l158_158696


namespace probability_sum_even_l158_158279

-- Let's define the set of the first ten prime numbers.
def first_ten_primes : Finset ℕ := Finset.of_list [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define the proposition to prove:
theorem probability_sum_even :
  let n := first_ten_primes.card in 
  let total_pairs := Finset.card (Finset.powersetLen 2 first_ten_primes) in 
  let even_pairs := Finset.card (Finset.filter (λ s, s.sum % 2 = 0) (Finset.powersetLen 2 first_ten_primes)) in 
  (total_pairs = 45 ∧ even_pairs = 36) →
  (even_pairs / total_pairs = 4 / 5) :=
by
  sorry

end probability_sum_even_l158_158279


namespace trajectory_of_moving_point_P_l158_158838

theorem trajectory_of_moving_point_P (x y : ℝ) :
    (∃ A B : ℝ × ℝ, (x, y) ∉ {A, B} ∧ angle A (0, 0) B = 60) →
    x^2 + y^2 = 4 :=
by sorry

end trajectory_of_moving_point_P_l158_158838


namespace factorial_div_power_of_two_odd_l158_158798

theorem factorial_div_power_of_two_odd (n k : ℕ) (h₁ : k = (nat.binary_length n).succ - nat.count_ones n) (h₂ : nat.count_ones n = k) :
  odd (n! / 2^(n - k)) :=
begin
  sorry
end

end factorial_div_power_of_two_odd_l158_158798


namespace rectangle_angle_AMD_l158_158311

theorem rectangle_angle_AMD {A B C D M : Type} [MetricSpace A]
  (h_rect : MetricSpace.IsRectangle A B C D)
  (AB BC : ℝ)
  (h_AB : MetricSpace.dist A B = 10)
  (h_BC : MetricSpace.dist B C = 4)
  (h_AM_MD : MetricSpace.dist A M = 2 * MetricSpace.dist M D)
  : ∠A M D = Real.arctan (5/6) :=
by
  sorry

end rectangle_angle_AMD_l158_158311


namespace radius_of_circle_l158_158749

theorem radius_of_circle (r : ℝ) (h : π * r^2 = 81 * π) : r = 9 :=
by
  sorry

end radius_of_circle_l158_158749


namespace Jake_needs_to_lose_12_pounds_l158_158325

theorem Jake_needs_to_lose_12_pounds (J S : ℕ) (h1 : J + S = 156) (h2 : J = 108) : J - 2 * S = 12 := by
  sorry

end Jake_needs_to_lose_12_pounds_l158_158325


namespace spatial_vector_operations_l158_158444

theorem spatial_vector_operations :
  ¬ (∀ (V : Type) [Add V] [Sub V] [Mul ℝ V] [Div V], False) := 
begin
  sorry
end

end spatial_vector_operations_l158_158444


namespace arccos_one_over_sqrt_two_eq_pi_four_l158_158908

theorem arccos_one_over_sqrt_two_eq_pi_four : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 := 
by
  sorry

end arccos_one_over_sqrt_two_eq_pi_four_l158_158908


namespace remainder_3001_3005_mod_23_l158_158789

theorem remainder_3001_3005_mod_23 : 
  (3001 * 3002 * 3003 * 3004 * 3005) % 23 = 9 :=
by {
  sorry
}

end remainder_3001_3005_mod_23_l158_158789


namespace remove_56_blue_balls_l158_158863

theorem remove_56_blue_balls 
  (initial_total : ℕ := 120)
  (red_percentage : ℝ := 0.4)
  (desired_red_percentage : ℝ := 0.75) :
  (let red := (red_percentage * initial_total : ℕ) in
   let blue := initial_total - red in
   let remaining_total (x : ℕ) := initial_total - x in
   red / remaining_total 56 = desired_red_percentage) :=
by
  let red := (red_percentage * initial_total : ℕ)
  let blue := initial_total - red
  let remaining_total (x : ℕ) := initial_total - x

  have h : 56 = (42 / 0.75) := rfl -- derived from the solution
  sorry

end remove_56_blue_balls_l158_158863


namespace distance_to_asymptote_l158_158207

theorem distance_to_asymptote :
  let C := {p : ℝ × ℝ | p.1 ^ 2 - p.2 ^ 2 = 1}
  let point := (4, 0)
  let asymptotes := [{p : ℝ × ℝ | p.1 + p.2 = 0}, {p : ℝ × ℝ | p.1 - p.2 = 0}]
  ∃ d, d = 2 * real.sqrt 2 ∧
    ∀ A ∈ asymptotes, (distance_between point A = d) :=
by
  sorry

end distance_to_asymptote_l158_158207


namespace fraction_of_milk_in_cup1_l158_158671

variable (initial_coffee_cup1_serv1 : ℚ) (initial_milk_cup2_serv2 : ℚ)
variable (transferred_coffee_serv3 : ℚ) (transferred_mix_serv4 : ℚ)
variable (coffee_cup1_serv5 : ℚ) (milk_cup1_serv6 : ℚ)
variable (final_coffee_cup1_serv7 : ℚ) (final_milk_cup1_serv8 : ℚ)
variable (final_total_cup1_serv9 : ℚ)

-- Definitions based on conditions
def cup1_initial := initial_coffee_cup1_serv1 = 6
def cup2_initial := initial_milk_cup2_serv2 = 4
def coffee_transfer := transferred_coffee_serv3 = 3
def mix_transfer := transferred_mix_serv4 = 4
def cup1_coffee_remain := coffee_cup1_serv5 = 3
def cup2_mixed := 7
def mixture_ratio_coffee := (coffee_cup1_serv5 : ℚ) / cup2_mixed
def mixture_ratio_milk := (milk_cup1_serv6 : ℚ) / cup2_mixed
def coffee_back := (milk_cup1_serv6 : ℚ) := mixture_ratio_coffee * transferred_mix_serv4
def milk_back := (milk_cup1_serv6 : ℚ) := mixture_ratio_milk * transferred_mix_serv4
def final_coffee_in_cup1 := final_coffee_cup1_serv7 = coffee_cup1_serv5 + coffee_back
def final_milk_in_cup1 := final_milk_cup1_serv8 = milk_back
def final_total := final_total_cup1_serv9 = final_coffee_cup1_serv7 + final_milk_cup1_serv8

-- The statement to be proven in Lean 4
theorem fraction_of_milk_in_cup1 :
  cup1_initial ∧ cup2_initial ∧ coffee_transfer ∧ mix_transfer ∧ final_coffee_in_cup1 ∧ final_milk_in_cup1 ∧ final_total →
  (final_milk_cup1_serv8 / final_total_cup1_serv9) = 16 / 49 :=
by
  sorry

end fraction_of_milk_in_cup1_l158_158671


namespace sequence_properties_l158_158675

theorem sequence_properties (S : ℕ → ℝ) (a : ℕ → ℝ)
  (λ : ℝ) (hS : ∀ n, S (n + 1) = λ * S n + 1)
  (ha1 : a 1 = 1) (ha3 : a 3 = 4) (hλ_pos : 0 < λ) :
  (∀ n, a n = 2 ^ (n - 1)) ∧ (∀ n, ∑ i in finset.range n, i * a i = n * 2 ^ n - 2 ^ n + 1) :=
by
  -- Proof not required as per instructions
  sorry

end sequence_properties_l158_158675


namespace percentage_calculation_l158_158070

-- Definitions based on conditions
def x : ℕ := 5200
def p1 : ℚ := 0.50
def p2 : ℚ := 0.30
def p3 : ℚ := 0.15

-- The theorem stating the desired proof
theorem percentage_calculation : p3 * (p2 * (p1 * x)) = 117 := by
  sorry

end percentage_calculation_l158_158070


namespace arccos_one_over_sqrt_two_eq_pi_over_four_l158_158894

theorem arccos_one_over_sqrt_two_eq_pi_over_four : 
  Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_eq_pi_over_four_l158_158894


namespace width_of_room_l158_158019

-- Define the length of the room
def length_of_room : ℝ := 5.5

-- Define the total cost of paving
def total_cost_of_paving : ℝ := 12375.0

-- Define the cost per square meter
def cost_per_square_meter : ℝ := 600.0

-- Define the area of the floor
def area_of_floor : ℝ := total_cost_of_paving / cost_per_square_meter

-- Prove the width of the room
theorem width_of_room : area_of_floor / length_of_room = 3.75 :=
by -- Omit the proof
  sorry

end width_of_room_l158_158019


namespace probability_even_sum_l158_158254

def first_ten_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def is_odd (n : ℕ) : Prop := n % 2 = 1

def count_odd_primes (primes : List ℕ) : ℕ :=
  primes.countp is_odd

def binom (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

open Nat

theorem probability_even_sum :
  let primes := first_ten_primes in
  let odd_primes := count_odd_primes primes in
  let num_ways_even_sum := binom odd_primes 2 in
  let total_ways := binom primes.length 2 in
  (num_ways_even_sum : ℚ) / (total_ways : ℚ) = 4 / 5 :=
by
  sorry

end probability_even_sum_l158_158254


namespace ratio_MN_BC_eq_sqrt_expr_l158_158306

theorem ratio_MN_BC_eq_sqrt_expr (ABC : Triangle) (A B C M N : Point) (R r : ℝ)
  (H_M_on_AB : M ∈ LineSegment A B) 
  (H_N_on_AC : N ∈ LineSegment A C)
  (H_MB_eq_BC : distance M B = distance B C)
  (H_BC_eq_CN : distance B C = distance C N)
  (H_R_circumradius : circumradius ABC = R)
  (H_r_inradius : inradius ABC = r) :
  distance M N / distance B C = Real.sqrt (1 - 2 * r / R) :=
sorry

end ratio_MN_BC_eq_sqrt_expr_l158_158306


namespace probability_sum_even_l158_158288

def primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def sum_is_even (a b : ℕ) : Prop :=
  (a + b) % 2 = 0

theorem probability_sum_even :
  (nat.choose 10 2) = 45 →
  (∀ a b : ℕ, a ∈ primes → b ∈ primes → a ≠ b →
    (sum_is_even a b ↔ a ≠ 2 ∧ b ≠ 2)) →
  ((45 - 9) / 45 : ℚ) = 4 / 5 :=
by sorry

end probability_sum_even_l158_158288


namespace real_root_of_cubic_l158_158354

theorem real_root_of_cubic (c d: ℝ) (h1: cx^3 + 4x^2 + dx - 100 = 0) 
(h2: is_root (λ x: ℂ, c * x^3 + (4: ℝ) * x^2 + (d: ℝ) * x - 100) (-3 - 4i) ):
  ∃ r : ℝ, is_root (λ x : ℝ, c * x^3 + 4 * x^2 + d * x - 100) r ∧ r = 25 / 3 :=
sorry

end real_root_of_cubic_l158_158354


namespace total_number_of_stickers_l158_158712

def sticker_count (sheets : ℕ) (stickers_per_sheet : ℕ) : ℕ := sheets * stickers_per_sheet

theorem total_number_of_stickers 
    (sheets_per_folder : ℕ)
    (red_folder_stickers_per_sheet : ℕ)
    (green_folder_stickers_per_sheet : ℕ)
    (blue_folder_stickers_per_sheet : ℕ) :
    sticker_count sheets_per_folder red_folder_stickers_per_sheet +
    sticker_count sheets_per_folder green_folder_stickers_per_sheet +
    sticker_count sheets_per_folder blue_folder_stickers_per_sheet = 60 := 
begin
    -- Given conditions
    let sheets := 10      -- Each folder contains 10 sheets of paper.
    let red := 3          -- Each sheet in the red folder gets 3 stickers.
    let green := 2        -- Each sheet in the green folder gets 2 stickers.
    let blue := 1         -- Each sheet in the blue folder gets 1 sticker.
    have h1 : sticker_count sheets red = 30, by sorry, -- Calculation omitted
    have h2 : sticker_count sheets green = 20, by sorry, -- Calculation omitted
    have h3 : sticker_count sheets blue = 10, by sorry, -- Calculation omitted

    -- Summing the stickers
    show h1 + h2 + h3 = 60, by sorry
end

end total_number_of_stickers_l158_158712


namespace construct_triangle_l158_158124

variable {a b : ℝ}

-- main theorem statement
theorem construct_triangle (h1 : 0 < b) (h2 : b < a) (h3 : a < 2 * b) :
  exists (α β γ : ℝ), (α = 2 * β) ∧ (α + β + γ = 180) ∧ 
  (sin β = b / (2 * b)) ∧ (α > 0) ∧ (β > 0) ∧ (γ > 0) :=
sorry

end construct_triangle_l158_158124


namespace shortest_distance_is_to_E_l158_158500

-- Define the grid and points P, A, B, C, D, and E
structure Point :=
  (x : ℕ)
  (y : ℕ)

def P : Point := { x := 0, y := 0 }
def A : Point := { x := 5, y := 4 }
def B : Point := { x := 6, y := 2 }
def C : Point := { x := 3, y := 3 }
def D : Point := { x := 5, y := 1 }
def E : Point := { x := 1, y := 4 }

-- Define the distance function
def distance (p1 p2 : Point) : ℕ :=
  abs (p2.x - p1.x) + abs (p2.y - p1.y)

-- Define the shortest distance problem
theorem shortest_distance_is_to_E :
  distance P E = min (distance P A) (min (distance P B) (min (distance P C) (min (distance P D) (distance P E)))) :=
by
  sorry

end shortest_distance_is_to_E_l158_158500


namespace negation_proposition_l158_158760

open Real

theorem negation_proposition (h : ∀ x : ℝ, x^2 - 2*x - 1 > 0) :
  ¬ (∀ x : ℝ, x^2 - 2*x - 1 > 0) = ∃ x_0 : ℝ, x_0^2 - 2*x_0 - 1 ≤ 0 :=
by 
  sorry

end negation_proposition_l158_158760


namespace least_possible_value_l158_158044

noncomputable def min_value_expression : ℝ :=
  let f (x : ℝ) := (x + 1) * (x + 2) * (x + 4) * (x + 5) + 2164
  have : (∀ x, f x ≥ 2161.75) := sorry,
  infi f

theorem least_possible_value : ∃ x : ℝ, (x + 1) * (x + 2) * (x + 4) * (x + 5) + 2164 = 2161.75 ∧ 
  (∀ x : ℝ, (x + 1) * (x + 2) * (x + 4) * (x + 5) + 2164 ≥ 2161.75) :=
sorry

end least_possible_value_l158_158044


namespace range_of_f_l158_158199

noncomputable def f (x : ℝ) : ℝ := 2^(2*x) + 2^(x+1) + 3

theorem range_of_f : ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ y > 3 :=
by
  sorry

end range_of_f_l158_158199


namespace simplify_and_evaluate_l158_158728

-- Define the given expression
noncomputable def given_expression (m : ℝ) : ℝ :=
  (m - (m + 9) / (m + 1)) / ((m^2) + 3 * m) / (m + 1)

-- Define the condition
def condition (m : ℝ) : Prop :=
  m = Real.sqrt 3

-- Define the correct answer
def correct_answer : ℝ :=
  1 - Real.sqrt 3

-- State the theorem
theorem simplify_and_evaluate 
  (m : ℝ) (h : condition m) : 
  given_expression m = correct_answer := by
  sorry

end simplify_and_evaluate_l158_158728


namespace problem_statement_l158_158244

def oper (x : ℕ) (w : ℕ) := (2^x) / (2^w)

theorem problem_statement : ∃ n : ℕ, oper (oper 4 2) n = 2 ↔ n = 3 :=
by sorry

end problem_statement_l158_158244


namespace circle_equation_l158_158476

theorem circle_equation :
  ∃ (C : ℝ × ℝ) (r : ℝ), C = (3, 0) ∧ r = sqrt 2 ∧
    (∀ (x y : ℝ), (x - 3)^2 + y^2 = 2 ↔ (C.1 - x)^2 + (C.2 - y)^2 = r^2) ∧
    (∃ (A B : ℝ × ℝ), A = (4, 1) ∧ B = (2, 1) ∧ 
     ((∃ (m : ℝ), m ≠ 0 ∧ (x - y = 1)) ∧
     (∃ (m : ℝ), m ≠ 0 ∧ (x + y = 3)))) :=
begin
  sorry
end

end circle_equation_l158_158476


namespace distance_between_centers_of_tangent_circles_l158_158479

-- Definitions of the entities involved
variables (O P Q T S : Point)
variable (rO rP : ℝ)
variable (distance : Point → Point → ℝ)

-- Given conditions
def conditions : Prop :=
  rO = 12 ∧ rP = 5 ∧ distance O Q = rO ∧ distance P Q = rP ∧ -- radii
  (∀ T, distance O T = rO ∧ distance P S = rP → distance T S = TS) -- tangents

-- The statement to be proven
theorem distance_between_centers_of_tangent_circles (h : conditions O P Q T S rO rP distance) : distance O P = 17 :=
sorry -- proof omitted

end distance_between_centers_of_tangent_circles_l158_158479


namespace mom_apples_leftover_l158_158220

variable (G S M : ℕ) -- Greg's, Sarah's, and Mark's apples
variable (total_apples : ℕ) -- Total apples

-- Conditions
def greg_sarah_split := G + S = 18 ∧ G = S
def susan_apples := ∀ G : ℕ, ∃ Su: ℕ, Su = 2 * G
def mark_apples := ∀ Su : ℕ, ∃ M : ℕ, M = Su - 5
def mom_needs := 40

-- Total apples calculation
def calculate_total_apples (G S M : ℕ) :=
  total_apples = G + S + 2 * G + (2 * G - 5)

-- Theorem statement
theorem mom_apples_leftover
  (G S M total_apples : ℕ)
  (h_split : greg_sarah_split G S)
  (h_susan : susan_apples G)
  (h_mark : mark_apples (2 * G))
  (h_total : calculate_total_apples G S M)
  (h_pie : total_apples = 49)
  : total_apples - mom_needs = 9 :=
  sorry

end mom_apples_leftover_l158_158220


namespace bob_gave_terry_24_bushels_l158_158109

def bushels_given_to_terry (total_bushels : ℕ) (ears_per_bushel : ℕ) (ears_left : ℕ) : ℕ :=
    (total_bushels * ears_per_bushel - ears_left) / ears_per_bushel

theorem bob_gave_terry_24_bushels : bushels_given_to_terry 50 14 357 = 24 := by
    sorry

end bob_gave_terry_24_bushels_l158_158109


namespace set_intersection_l158_158216

open Set

theorem set_intersection :
  let M := {x : ℤ | x < 3}
  let N := {x : ℤ | 0 < x ∧ x < 6}
  M ∩ N = {1, 2} :=
by {
  let M := {x : ℤ | x < 3},
  let N := {x : ℤ | 0 < x ∧ x < 6},
  sorry
}

end set_intersection_l158_158216


namespace arccos_sqrt2_l158_158934

def arccos_eq (x : ℝ) := arccos x
def range_arccos := ∀ (x : ℝ), 0 ≤ arccos x ∧ arccos x ≤ π
def cos_pi_div_four : Real.cos (π / 4) = 1 / Real.sqrt 2 := by sorry

theorem arccos_sqrt2 :
  arccos_eq (1 / Real.sqrt 2) = π / 4 :=
by
  have h1 : Real.cos (π / 4) = 1 / Real.sqrt 2, from cos_pi_div_four
  sorry

end arccos_sqrt2_l158_158934


namespace trapezoid_bisect_base_l158_158823

-- Given a trapezoid ABCD
variables {A B C D E F G : Type}

-- Definitions of points and properties
variables {trapezoid : Trapezoid A B C D}
          {diag_inter : Intersect A C B D E}
          {nonpara_inter : Intersect A D B C F}

-- Midpoint condition of larger base
def midpoints := ∀ (A B G : Type), Midpoint A B G

-- The proof statement
theorem trapezoid_bisect_base
  (h1 : trapezoid)
  (h2 : diag_inter)
  (h3 : nonpara_inter)
: midpoints A B G := 
sorry

end trapezoid_bisect_base_l158_158823


namespace can_be_indices_of_roots_l158_158114

noncomputable def nth_root (a : ℝ) (n : ℝ) : ℝ := a ^ (1 / n)

theorem can_be_indices_of_roots (a : ℝ) :
  (nth_root a 1 = a) ∧
  (0 < a → Filter.Tendsto (λ n : ℝ, a ^ (1 / n)) Filter.atTop 0) ∧
  (1 < a → Filter.Tendsto (λ n : ℝ, a ^ (1 / n)) Filter.atTop Filter.atTop) ∧
  (a = 1 → Filter.Tendsto (λ n : ℝ, a ^ (1 / n)) Filter.atTop 1) :=
by
  sorry

end can_be_indices_of_roots_l158_158114


namespace reflection_line_slope_sum_l158_158758

-- Given conditions and definitions
def point := ℝ × ℝ

def reflection (p q : point) (m b : ℝ) : Prop :=
  let (x1, y1) := p in
  let (x2, y2) := q in
  (x1 + x2) / 2 = (y1 + y2 - b) / (2 * m) ∧
  m * x1 - y1 + m * x2 - y2 = b * m

-- Problem statement
theorem reflection_line_slope_sum (m b : ℝ) :
  reflection (2, -2) (8, 4) m b → m + b = 5 :=
by trivial

end reflection_line_slope_sum_l158_158758


namespace fifth_roll_six_probability_l158_158881
noncomputable def probability_fifth_roll_six : ℚ := sorry

theorem fifth_roll_six_probability :
  let fair_die_prob : ℚ := (1/6)^4
  let biased_die_6_prob : ℚ := (2/3)^3 * (1/15)
  let biased_die_3_prob : ℚ := (1/10)^3 * (1/2)
  let total_prob := (1/3) * fair_die_prob + (1/3) * biased_die_6_prob + (1/3) * biased_die_3_prob
  let normalized_biased_6_prob := (1/3) * biased_die_6_prob / total_prob
  let prob_of_fifth_six := normalized_biased_6_prob * (2/3)
  probability_fifth_roll_six = prob_of_fifth_six :=
sorry

end fifth_roll_six_probability_l158_158881


namespace evaporation_weight_l158_158413

theorem evaporation_weight
  (initial_total_weight : ℕ)
  (empty_glass_weight : ℕ)
  (initial_water_percentage : ℝ)
  (final_water_percentage : ℝ)
  (initial_total_weight = 500)
  (empty_glass_weight = 300)
  (initial_water_percentage = 0.99)
  (final_water_percentage = 0.98) :
  let initial_solution_weight := initial_total_weight - empty_glass_weight,
      initial_water_weight := initial_solution_weight * initial_water_percentage,
      solute_weight := initial_solution_weight - initial_water_weight,
      final_solution_weight := solute_weight / (1 - final_water_percentage),
      final_total_weight := final_solution_weight + empty_glass_weight in
  final_total_weight = 400 := by
  sorry

end evaporation_weight_l158_158413


namespace eq_triangle_no_same_branch_find_Q_R_coordinates_l158_158987

-- Problem 1
theorem eq_triangle_no_same_branch (P Q R : ℝ × ℝ) 
  (H_P_xy : P.1 * P.2 = 1) (H_Q_xy : Q.1 * Q.2 = 1) (H_R_xy : R.1 * R.2 = 1) 
  (H_equilateral : dist P Q = dist Q R ∧ dist Q R = dist R P) : 
  ¬ (P.1 > 0 ∧ Q.1 > 0 ∧ R.1 > 0 ∨ P.1 < 0 ∧ Q.1 < 0 ∧ R.1 < 0) := 
sorry

-- Problem 2
theorem find_Q_R_coordinates (Q R : ℝ × ℝ) 
  (H_P : (-1 : ℝ, -1) * (-1, -1) = (1 : ℝ)) 
  (H_Q_xy : Q.1 * Q.2 = 1) 
  (H_R_xy : R.1 * R.2 = 1) 
  (H_equilateral : dist (-1 : ℝ, -1) Q = dist Q R ∧ dist Q R = dist R (-1, -1)) : 
  Q = (2 - real.sqrt(3), 2 + real.sqrt(3)) ∧ R = (2 + real.sqrt(3), 2 - real.sqrt(3)) := 
sorry

end eq_triangle_no_same_branch_find_Q_R_coordinates_l158_158987


namespace triangle_construction_exists_l158_158523

variables {a b : ℝ} (h : a < b)

theorem triangle_construction_exists (h3angle : ∃ (A B C : Type) 
  (angle_ABC angle_BAC : ℝ), angle_ABC = 3 * angle_BAC ∧ 
  ∀ (side_AB side_BC : ℝ), side_AB = b ∧ side_BC = a) : 
  ∃ (A B C : Type), true :=
begin
  -- Proof will be inserted here
  sorry
end

end triangle_construction_exists_l158_158523


namespace mod_81256_eq_16_l158_158785

theorem mod_81256_eq_16 : ∃ n : ℤ, 0 ≤ n ∧ n < 31 ∧ 81256 % 31 = n := by
  use 16
  sorry

end mod_81256_eq_16_l158_158785


namespace probability_even_sum_l158_158257

def first_ten_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def is_odd (n : ℕ) : Prop := n % 2 = 1

def count_odd_primes (primes : List ℕ) : ℕ :=
  primes.countp is_odd

def binom (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

open Nat

theorem probability_even_sum :
  let primes := first_ten_primes in
  let odd_primes := count_odd_primes primes in
  let num_ways_even_sum := binom odd_primes 2 in
  let total_ways := binom primes.length 2 in
  (num_ways_even_sum : ℚ) / (total_ways : ℚ) = 4 / 5 :=
by
  sorry

end probability_even_sum_l158_158257


namespace problem_statement_l158_158191

variable {x y : ℝ}

theorem problem_statement 
  (h1 : y > x)
  (h2 : x > 0)
  (h3 : x + y = 1) :
  x < 2 * x * y ∧ 2 * x * y < (x + y) / 2 ∧ (x + y) / 2 < y := by
  sorry

end problem_statement_l158_158191


namespace arccos_one_over_sqrt_two_eq_pi_four_l158_158907

theorem arccos_one_over_sqrt_two_eq_pi_four : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 := 
by
  sorry

end arccos_one_over_sqrt_two_eq_pi_four_l158_158907


namespace find_m_and_operation_result_l158_158552

def custom_op (x y m : ℝ) : ℝ :=
  (4 * x * y) / (m * x + 3 * y)

theorem find_m_and_operation_result
  (m : ℝ)
  (h : custom_op 1 2 m = 1) :
  m = 2 ∧ custom_op 3 12 2 = 24 / 7 :=
by
  sorry

end find_m_and_operation_result_l158_158552


namespace solve_for_x_l158_158561

def α(x : ℚ) : ℚ := 4 * x + 9
def β(x : ℚ) : ℚ := 9 * x + 6

theorem solve_for_x (x : ℚ) (h : α(β(x)) = 8) : x = -25 / 36 :=
by
  sorry

end solve_for_x_l158_158561


namespace probability_even_sum_l158_158287

def first_ten_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def distinct_pairs (l : List ℕ) : List (ℕ × ℕ) :=
  (l.product l).filter (λ p, p.1 < p.2)

theorem probability_even_sum : 
  (first_ten_primes.length = 10) →
  (∀ a b : ℕ, a ∈ first_ten_primes → b ∈ first_ten_primes → a ≠ b → 
    ((a + b) % 2 = 0 ↔ 2 ∉ [a, b])) →
  (∑ pair in distinct_pairs first_ten_primes, ((pair.1 + pair.2) % 2 = 0).toNat) 
    = 36 → 
  ((∑ pair in distinct_pairs first_ten_primes, (1:ℚ)) ⁻¹ * 
   (∑ pair in distinct_pairs first_ten_primes, ((pair.1 + pair.2) % 2 = 0).toNat))
    = 4 / 5 := by
  sorry

end probability_even_sum_l158_158287


namespace cylinder_total_surface_area_eq_130π_l158_158483

open Real

-- Definitions from the problem conditions.
def cylinder_height : ℝ := 8
def cylinder_radius : ℝ := 5

-- Lean function to calculate the total surface area of a cylinder.
def total_surface_area (h r : ℝ) : ℝ := 
  let end_area := 2 * (π * r^2)
  let curved_surface_area := (2 * π * r) * h
  end_area + curved_surface_area

-- Proof statement that the total surface area of a cylinder with given height and radius is 130π.
theorem cylinder_total_surface_area_eq_130π : 
  total_surface_area cylinder_height cylinder_radius = 130 * π := sorry

end cylinder_total_surface_area_eq_130π_l158_158483


namespace neil_initial_games_l158_158224

variable {N : ℕ}
variable h_games_initial : ℕ := 33
variable h_games_given : ℕ := 5
variable h_games_after : ℕ := h_games_initial - h_games_given
variable h_games_mul : ℕ := 4
variable neil_games_after : ℕ := N + h_games_given

theorem neil_initial_games (h : h_games_after = h_games_mul * neil_games_after) : N = 2 :=
sorry

end neil_initial_games_l158_158224


namespace radius_of_semicircle_l158_158039

theorem radius_of_semicircle (r1 r2 : ℝ) (h_r1 : r1 = sqrt 19) (h_r2 : r2 = sqrt 76)
  (h_r2_eq_2r1 : r2 = 2 * r1) (R : ℝ) :
  (∃ (O O1 O2 : ℝ × ℝ),
    let O := (0, 0),
        O1 := (r1, sqrt ((R - r1) * (R - r1))),
        O2 := (2 * r1, sqrt ((R - 2 * r1) * (R - 2 * r1))) in 
    dist O1 O2 = r1 + r2 ∧ 
    dist O O1 = R - r1 ∧ 
    dist O O2 = R - 2 * r1
  ) → R = 4 * sqrt 19 :=
by sorry

end radius_of_semicircle_l158_158039


namespace general_term_of_sequence_l158_158573

-- Definition of arithmetic sequence with positive common difference
def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) := ∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions
variables {a : ℕ → ℤ} {d : ℤ}
axiom positive_common_difference : d > 0
axiom cond1 : a 3 * a 4 = 117
axiom cond2 : a 2 + a 5 = 22

-- Target statement to prove
theorem general_term_of_sequence : is_arithmetic_sequence a d → a n = 4 * n - 3 :=
by sorry

end general_term_of_sequence_l158_158573


namespace owen_work_hours_l158_158355

def total_hours := 24
def chores_hours := 7
def sleep_hours := 11

theorem owen_work_hours : total_hours - chores_hours - sleep_hours = 6 := by
  sorry

end owen_work_hours_l158_158355


namespace remainder_3001_3005_mod_23_l158_158788

theorem remainder_3001_3005_mod_23 : 
  (3001 * 3002 * 3003 * 3004 * 3005) % 23 = 9 :=
by {
  sorry
}

end remainder_3001_3005_mod_23_l158_158788


namespace age_ratio_contradiction_l158_158764

variables (L B : ℕ)

theorem age_ratio_contradiction : ¬ (L = 6 ∧ B = 6) → ((L + 3) / (B + 3) = 3 / 5) → false :=
by
  intros h1 h2
  have L_eq_6 := h1.1
  have B_eq_6 := h1.2
  rw [L_eq_6, B_eq_6] at h2
  simp at h2
  exact h2 sorry

#check age_ratio_contradiction

end age_ratio_contradiction_l158_158764


namespace solution_l158_158871

noncomputable def x1 : ℝ := sorry
noncomputable def x2 : ℝ := sorry
noncomputable def x3 : ℝ := sorry
noncomputable def x4 : ℝ := sorry
noncomputable def x5 : ℝ := sorry
noncomputable def x6 : ℝ := sorry
noncomputable def x7 : ℝ := sorry
noncomputable def x8 : ℝ := sorry

axiom cond1 : x1 + 4 * x2 + 9 * x3 + 16 * x4 + 25 * x5 + 36 * x6 + 49 * x7 + 64 * x8 = 10
axiom cond2 : 4 * x1 + 9 * x2 + 16 * x3 + 25 * x4 + 36 * x5 + 49 * x6 + 64 * x7 + 81 * x8 = 40
axiom cond3 : 9 * x1 + 16 * x2 + 25 * x3 + 36 * x4 + 49 * x5 + 64 * x6 + 81 * x7 + 100 * x8 = 170

theorem solution : 16 * x1 + 25 * x2 + 36 * x3 + 49 * x4 + 64 * x5 + 81 * x6 + 100 * x7 + 121 * x8 = 400 := 
by
  sorry

end solution_l158_158871


namespace odd_n_cube_minus_n_div_by_24_l158_158360

theorem odd_n_cube_minus_n_div_by_24 (n : ℤ) (h_odd : n % 2 = 1) : 24 ∣ (n^3 - n) :=
sorry

end odd_n_cube_minus_n_div_by_24_l158_158360


namespace log_sum_fractional_parts_l158_158677

def fractional_part (x : ℝ) : ℝ := x - real.floor x

theorem log_sum_fractional_parts : 
  (∑ n in finset.range 1991, fractional_part (real.log n / real.log 2)) = 19854 := 
by 
  sorry

end log_sum_fractional_parts_l158_158677


namespace arccos_sqrt2_l158_158935

def arccos_eq (x : ℝ) := arccos x
def range_arccos := ∀ (x : ℝ), 0 ≤ arccos x ∧ arccos x ≤ π
def cos_pi_div_four : Real.cos (π / 4) = 1 / Real.sqrt 2 := by sorry

theorem arccos_sqrt2 :
  arccos_eq (1 / Real.sqrt 2) = π / 4 :=
by
  have h1 : Real.cos (π / 4) = 1 / Real.sqrt 2, from cos_pi_div_four
  sorry

end arccos_sqrt2_l158_158935


namespace surface_area_ratio_l158_158366

/-- 
Given the side length of a cube is 4x and the side lengths of a rectangular prism are x, 3x, and 2x,
prove that the ratio of their total surface areas is 48:11 
-/
theorem surface_area_ratio (x : ℝ) (hx : 0 < x) :
  let sa_cube := 6 * (4 * x)^2,
      sa_prism := 2 * (x * 3 * x) + 2 * (x * 2 * x) + 2 * (3 * x * 2 * x) in
  (sa_cube : ℝ) / sa_prism = 48 / 11 := 
by 
  let sa_cube := 6 * (4 * x)^2
  let sa_prism := 2 * (x * 3 * x) + 2 * (x * 2 * x) + 2 * (3 * x * 2 * x)
  have h_sa_cube : sa_cube = 96 * x^2 := by sorry
  have h_sa_prism : sa_prism = 22 * x^2 := by sorry
  calc 
    (96 * x^2) / (22 * x^2) = 96 / 22 : by sorry
    ... = 48 / 11 : by sorry

end surface_area_ratio_l158_158366


namespace inverse_correctness_l158_158033

noncomputable def f : ℕ → ℕ
| 1 := 4
| 2 := 6
| 3 := 2
| 4 := 5
| 5 := 3
| 6 := 1
| _ := 0  -- we might assume the function is undefined for other inputs but we give a default

noncomputable def f_inv : ℕ → ℕ
| 4 := 1
| 6 := 2
| 2 := 3
| 5 := 4
| 3 := 5
| 1 := 6
| _ := 0  -- we might assume the function is undefined for other inputs but we give a default

theorem inverse_correctness : f_inv (f_inv (f_inv 4)) = 2 := by
  sorry

end inverse_correctness_l158_158033


namespace math_problem_circles_intersection_l158_158882

noncomputable def problem2 : ℕ :=
  let a : ℕ := 10
  let b : ℕ := 155
  let c : ℕ := 79
  a + b + c

theorem math_problem_circles_intersection :
  let n := (10 * n.sqrt(155)) / 79
  a + b + c = 244 :=
by
  sorry

end math_problem_circles_intersection_l158_158882


namespace coin_stack_l158_158656

def penny_thickness : ℝ := 1.55
def nickel_thickness : ℝ := 1.95
def dime_thickness : ℝ := 1.35
def quarter_thickness : ℝ := 1.75
def stack_height : ℝ := 14

theorem coin_stack (n_penny n_nickel n_dime n_quarter : ℕ) 
  (h : n_penny * penny_thickness + n_nickel * nickel_thickness + n_dime * dime_thickness + n_quarter * quarter_thickness = stack_height) :
  n_penny + n_nickel + n_dime + n_quarter = 8 :=
sorry

end coin_stack_l158_158656


namespace work_completion_l158_158809

noncomputable def efficiency (p q: ℕ) := q = 3 * p / 5

theorem work_completion (p q : ℕ) (h1 : efficiency p q) (h2: p * 24 = 100) :
  2400 / (p + q) = 15 :=
by 
  sorry

end work_completion_l158_158809


namespace parts_in_batch_l158_158699

-- Let x be the original number of days planned for the completion batch
variable (x : ℕ)

-- Let N be the total number of processed parts
variable (N : ℕ)

-- Let r be the original production rate
variable (r : ℕ)

-- Condition 1
def cond1 : Prop := 20 * (x - 1) = N

-- Condition 2
def cond2 : Prop := 4 * r + (r + 5) * (x - 7) = N - 4 * r

-- Proof statement
theorem parts_in_batch (h₁: cond1) (h₂: cond2) : N = 280 :=
by
  sorry

end parts_in_batch_l158_158699


namespace arccos_sqrt2_l158_158940

def arccos_eq (x : ℝ) := arccos x
def range_arccos := ∀ (x : ℝ), 0 ≤ arccos x ∧ arccos x ≤ π
def cos_pi_div_four : Real.cos (π / 4) = 1 / Real.sqrt 2 := by sorry

theorem arccos_sqrt2 :
  arccos_eq (1 / Real.sqrt 2) = π / 4 :=
by
  have h1 : Real.cos (π / 4) = 1 / Real.sqrt 2, from cos_pi_div_four
  sorry

end arccos_sqrt2_l158_158940


namespace empty_seats_in_theater_l158_158815

theorem empty_seats_in_theater :
  let total_seats := 750
  let occupied_seats := 532
  total_seats - occupied_seats = 218 :=
by
  sorry

end empty_seats_in_theater_l158_158815


namespace value_of_x_plus_y_l158_158061

-- Define the sum of integers from 50 to 60
def sum_integers_50_to_60 : ℤ := List.sum (List.range' 50 (60 - 50 + 1))

-- Calculate the number of even integers from 50 to 60
def count_even_integers_50_to_60 : ℤ := List.length (List.filter (λ n => n % 2 = 0) (List.range' 50 (60 - 50 + 1)))

-- Define x and y based on the given conditions
def x : ℤ := sum_integers_50_to_60
def y : ℤ := count_even_integers_50_to_60

-- The main theorem to prove
theorem value_of_x_plus_y : x + y = 611 := by
  -- Placeholder for the proof
  sorry

end value_of_x_plus_y_l158_158061


namespace largest_multiple_of_9_less_than_neg_70_l158_158430

theorem largest_multiple_of_9_less_than_neg_70 : 
  ∃ k : ℤ, 9 * k < -70 ∧ ∀ m : ℤ, 9 * m < -70 → 9 * m ≤ 9 * k :=
begin
  use -8,
  split,
  { norm_num, },
  { intros m hm,
    apply mul_le_mul_of_nonpos_left _ (by norm_num : 9 ≤ 0),
    linarith, },
end

end largest_multiple_of_9_less_than_neg_70_l158_158430


namespace arccos_one_over_sqrt_two_eq_pi_over_four_l158_158926

theorem arccos_one_over_sqrt_two_eq_pi_over_four : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_eq_pi_over_four_l158_158926


namespace ratio_E_divides_BC_l158_158639

variable {A B C F G H E : Type}
variables (P : A → B → C → Prop) (Q : A → F → Prop) (R : B → F → G → Prop)
variables (S : A → F → H → Prop) (T : G → H → E → Prop)

-- Conditions
axiom cond1 : ∃ k l : Nat, k = 2 ∧ l = 3 ∧ Q A F
axiom cond2 : ∃ m n : Nat, m = 2 ∧ n = 1 ∧ R B F G
axiom cond3 : A = midpoint A F → S A F H
axiom cond4 : P A B C ∧ S A F H ∧ T G H E

-- Proof Statement
theorem ratio_E_divides_BC (A B C F G H E : Type) [P A B C]
  [Q A F] [R B F G] [S A F H] [T G H E] :
  divides E B C (1 : 5) :=
sorry

end ratio_E_divides_BC_l158_158639


namespace simplify_and_evaluate_l158_158735

noncomputable def my_expression (m : ℝ) : ℝ :=
  (m - (m + 9) / (m + 1)) / ((m ^ 2 + 3 * m) / (m + 1))

theorem simplify_and_evaluate : my_expression (Real.sqrt 3) = 1 - Real.sqrt 3 :=
by
  sorry

end simplify_and_evaluate_l158_158735


namespace reflections_concur_point_l158_158362

-- Define the triangle, circumcenter, and reflections
variables {A B C : Point}
variables (O : Point) (h : Line)
variables (h_a h_b h_c : Line)
variables (O_a O_b O_c : Point)

-- Conditions
def is_circumcenter (O : Point) (ABC : Triangle) : Prop :=
  -- O is the circumcenter of triangle ABC
  O = circumcenter ABC

def reflects_over_side (h : Line) (ABC : Triangle) : Line :=
  -- h is reflected line over the sides of triangle ABC
  reflect_line h (side ABC)

def reflections (h : Line) (ABC : Triangle) (O_a O_b O_c : Point) : Prop :=
  -- Reflections of line passing through circumcenter with respect sides
  O_a = reflect O (BC ABC) ∧ O_b = reflect O (CA ABC) ∧ O_c = reflect O (AB ABC)

-- Goal
theorem reflections_concur_point (ABC : Triangle)
  (h h_a h_b h_c : Line) (circumcenter O : Point) :
  is_circumcenter O ABC →
  reflects_over_side h ABC = (h_a) →
  reflects_over_side h ABC = (h_b) →
  reflects_over_side h ABC = (h_c) →
  reflections h ABC (O_a O_b O_c) →
  (∃ P : Point, P ∈ h_a ∧ P ∈ h_b ∧ P ∈ h_c) :=
by
  sorry

end reflections_concur_point_l158_158362


namespace distance_to_larger_cross_section_l158_158040

theorem distance_to_larger_cross_section
    (A B : ℝ)
    (a b : ℝ)
    (d : ℝ)
    (h : ℝ)
    (h_eq : h = 30):
  A = 300 * Real.sqrt 2 → 
  B = 675 * Real.sqrt 2 → 
  a = Real.sqrt (A / B) → 
  b = d / (1 - a) → 
  d = 10 → 
  b = h :=
by
  sorry

end distance_to_larger_cross_section_l158_158040


namespace golf_money_l158_158621

-- Definitions based on conditions
def cost_per_round : ℤ := 80
def number_of_rounds : ℤ := 5

-- The theorem/problem statement
theorem golf_money : cost_per_round * number_of_rounds = 400 := 
by {
  -- Proof steps would go here, but to skip the proof, we use sorry
  sorry
}

end golf_money_l158_158621


namespace max_area_of_equilateral_triangle_in_square_l158_158344

theorem max_area_of_equilateral_triangle_in_square :
  ∀ (W X Y Z : ℝ) (a : ℝ),
    a = 8 →
    WXYZ_is_square W X Y Z a →
    ∃ (A : ℝ), 
      A = 16 * Real.sqrt 3 ∧
      fits_within W X Y Z a equilateral_triangle_with_area A :=
by
  sorry

end max_area_of_equilateral_triangle_in_square_l158_158344


namespace centroid_minimizes_sum_of_squares_distances_l158_158547

noncomputable def sum_of_squares_distances_to_vertices (M A B C : Point) : ℝ :=
  (dist M A)^2 + (dist M B)^2 + (dist M C)^2

theorem centroid_minimizes_sum_of_squares_distances (A B C : Point) :
  ∃ I : Point, is_centroid I A B C ∧
  ∀ M : Point, sum_of_squares_distances_to_vertices M A B C ≥ sum_of_squares_distances_to_vertices I A B C :=
sorry

end centroid_minimizes_sum_of_squares_distances_l158_158547


namespace friends_meeting_probability_l158_158781

def event_meeting {x y : ℝ} (hx : 0 ≤ x ∧ x ≤ 30) (hy : 0 ≤ y ∧ y ≤ 30) : Prop :=
  |x - y| ≤ 20

theorem friends_meeting_probability :
  let Ω := set.univ.prod set.univ
  let A := {p : ℝ × ℝ | event_meeting ⟨p.1, p.2⟩ ⟨p.1, p.2⟩}
  (measure_theory.measure_space.volume A) / (measure_theory.measure_space.volume Ω) = 8 / 9 :=
sorry

end friends_meeting_probability_l158_158781


namespace sunflower_seeds_more_than_half_l158_158706

/--
On the second day (Tuesday), just after Jessie has replenished the feeder,
more than half the seeds in the feeder are sunflower seeds.
-/
theorem sunflower_seeds_more_than_half :
  ∀ (n : ℕ), n = 2 → 
  let initial_sunflower := 0.4
  let daily_add := 0.4
  let daily_consume_factor := 0.6
  let total_sunflower (n : ℕ) := daily_add * (1 - daily_consume_factor ^ n) / (1 - daily_consume_factor)
  (total_sunflower n) / (initial_sunflower + n * 1) > 0.5 :=
begin
  sorry
end

end sunflower_seeds_more_than_half_l158_158706


namespace probability_sum_even_l158_158277

-- Let's define the set of the first ten prime numbers.
def first_ten_primes : Finset ℕ := Finset.of_list [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define the proposition to prove:
theorem probability_sum_even :
  let n := first_ten_primes.card in 
  let total_pairs := Finset.card (Finset.powersetLen 2 first_ten_primes) in 
  let even_pairs := Finset.card (Finset.filter (λ s, s.sum % 2 = 0) (Finset.powersetLen 2 first_ten_primes)) in 
  (total_pairs = 45 ∧ even_pairs = 36) →
  (even_pairs / total_pairs = 4 / 5) :=
by
  sorry

end probability_sum_even_l158_158277


namespace largest_n_for_square_sum_l158_158994

theorem largest_n_for_square_sum :
  ∃ n : ℕ, (4^995 + 4^1500 + 4^n).natAbs.is_square ∧ (∀ m : ℕ, (4^995 + 4^1500 + 4^m).natAbs.is_square → m ≤ 1490) :=
sorry

end largest_n_for_square_sum_l158_158994


namespace relationship_among_a_b_c_l158_158166

noncomputable def a : ℝ := (0.8 : ℝ)^(5.2 : ℝ)
noncomputable def b : ℝ := (0.8 : ℝ)^(5.5 : ℝ)
noncomputable def c : ℝ := (5.2 : ℝ)^(0.1 : ℝ)

theorem relationship_among_a_b_c : b < a ∧ a < c := by
  sorry

end relationship_among_a_b_c_l158_158166


namespace construct_triangle_condition_l158_158989

theorem construct_triangle_condition (m_a f_a s_a : ℝ) : 
  (m_a < f_a) ∧ (f_a < s_a) ↔ (exists A B C : Type, true) :=
sorry

end construct_triangle_condition_l158_158989


namespace older_friend_is_38_l158_158007

-- Define the conditions
def younger_friend_age (x : ℕ) : Prop := 
  ∃ (y : ℕ), (y = x + 2 ∧ x + y = 74)

-- Define the age of the older friend
def older_friend_age (x : ℕ) : ℕ := x + 2

-- State the theorem
theorem older_friend_is_38 : ∃ x, younger_friend_age x ∧ older_friend_age x = 38 :=
by
  sorry

end older_friend_is_38_l158_158007


namespace division_by_vertical_line_eq_3_l158_158315

-- Lean 4 statement
theorem division_by_vertical_line_eq_3
  (A B C : ℝ × ℝ)
  (hA : A = (0,0))
  (hB : B = (1,1))
  (hC : C = (9,1)) :
  ∃ (a : ℝ), a = 3 ∧ 
  let left_area := 1 / 2 * (A.1 - a) * (C.2 - A.2)
      right_area := 1 / 2 * (C.1 - a) * (C.2 - A.2)
  in left_area = 2 ∧ right_area = 2 :=
sorry

end division_by_vertical_line_eq_3_l158_158315


namespace train_speed_l158_158456

theorem train_speed (length_train length_bridge time_crossing speed : ℝ)
  (h1 : length_train = 100)
  (h2 : length_bridge = 300)
  (h3 : time_crossing = 24)
  (h4 : speed = (length_train + length_bridge) / time_crossing) :
  speed = 16.67 := 
sorry

end train_speed_l158_158456


namespace circle_equation_with_focus_center_and_tangent_directrix_l158_158387

theorem circle_equation_with_focus_center_and_tangent_directrix :
  ∃ (x y : ℝ), (∃ k : ℝ, y^2 = -8 * x ∧ k = 2 ∧ (x = -2 ∧ y = 0) ∧ (x + 2)^2 + y^2 = 16) :=
by
  sorry

end circle_equation_with_focus_center_and_tangent_directrix_l158_158387


namespace transform_to_A_plus_one_l158_158412

theorem transform_to_A_plus_one (A : ℕ) (hA : A > 0) : 
  ∃ n : ℕ, (∀ i : ℕ, (i ≤ n) → ((A + 9 * i) = A + 1 ∨ ∃ j : ℕ, (A + 9 * i) = (A + 1 + 10 * j))) :=
sorry

end transform_to_A_plus_one_l158_158412


namespace arccos_proof_l158_158960

noncomputable def arccos_identity : Prop := 
  ∃ x : ℝ, x = 1 / Real.sqrt 2 ∧ Real.arccos x = Real.pi / 4

theorem arccos_proof : arccos_identity :=
by
  sorry

end arccos_proof_l158_158960


namespace range_of_m_l158_158013

theorem range_of_m (x y : ℝ) (m : ℝ) (h1 : x^2 + y^2 = 9) (h2 : |x| + |y| ≥ m) :
    m ≤ 3 / 2 := 
sorry

end range_of_m_l158_158013


namespace f_1998_equals_zero_l158_158088

def is_perfect (n : ℕ) : Prop :=
  ∑ m in (range n).filter (λ m, n % m = 0), m = n

def f : ℕ → ℕ
| n := if is_perfect n then 0
       else if n % 10 = 4 then 0
       else 0 -- default placeholder definition

theorem f_1998_equals_zero :
  f(1998) = 0 := sorry

end f_1998_equals_zero_l158_158088


namespace probability_product_divisible_by_4_l158_158848

open ProbabilityTheory

theorem probability_product_divisible_by_4 :
  (1 - ((3/4)^4 + 4 * (1/4) * (3/4)^3) = 13/16) :=
by
  sorry

end probability_product_divisible_by_4_l158_158848


namespace problem_solution_l158_158800

-- Definitions and conditions for each option
def optionA : Prop := ∀ a : ℝ, ∃ x y : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ x / a + y / a = 1

def optionB : Prop := ∃ p1 p2 p3 : ℝ × ℝ, 
  (p1.1 ^ 2 + p1.2 ^ 2 = 4) ∧
  (p2.1 ^ 2 + p2.2 ^ 2 = 4) ∧
  (p3.1 ^ 2 + p3.2 ^ 2 = 4) ∧
  (dist (0, 0) (p1.1 - p1.2 + Real.sqrt 2) = 1) ∧ 
  (dist (0, 0) (p2.1 - p2.2 + Real.sqrt 2) = 1) ∧ 
  (dist (0, 0) (p3.1 - p3.2 + Real.sqrt 2) = 1)

def optionC : Prop := ∃ m : ℝ,
  (\forall x y : ℝ, x^2 + y^2 + 2 * x = 0 ↔
  (x + 1)^2 + y^2 = 1) ∧
  (\forall x y : ℝ, x^2 + y^2 - 4 * x - 8 * y + m = 0 ↔
  (x - 2)^2 + (y - 4)^2 = 20 - m) ∧ 
  (\|(-1 - 2, 0 - 4)\| = 1 + Real.sqrt(20 - m)) ∧
  (m = 4)

def optionD : Prop := 
  (\forall x y : ℝ, ∃ P, (x, y).fst() / 4 + (x, y).snd() / 2 = 1 →
  \load x^2 + y^2 - (x, y).fst() * x - (x, y).snd() * y = 0 →
  \load x * (x, y).fst() + y * (x, y).snd() = 1 →
  \load x = 1 / 4 ∧ y = 1 / 2)

-- The final theorem statement that needs to be proved
theorem problem_solution : ¬ optionA ∧ optionB ∧ optionC ∧ optionD := 
  by sorry

end problem_solution_l158_158800


namespace pyramid_cross_section_distance_l158_158415

theorem pyramid_cross_section_distance 
  (A1 A2 : ℝ) (d : ℝ) (h : ℝ) 
  (hA1 : A1 = 125 * Real.sqrt 3)
  (hA2 : A2 = 500 * Real.sqrt 3)
  (hd : d = 12) :
  h = 24 :=
by
  sorry

end pyramid_cross_section_distance_l158_158415


namespace arccos_sqrt_half_l158_158969

theorem arccos_sqrt_half : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 := by
  sorry

end arccos_sqrt_half_l158_158969


namespace least_months_to_exceed_triple_debt_l158_158700

/-- 
Given:
- Principal amount (P) = 1200,
- Monthly interest rate (r) = 0.06,
- Multiplier (1 + r) = 1.06,
- Target amount = 3600.

Prove:
- The least integer t such that 1200 * 1.06^t > 3600 is 17.
--/
theorem least_months_to_exceed_triple_debt 
  (P : ℚ := 1200) 
  (r : ℚ := 0.06) 
  (multiplier : ℚ := 1.06) 
  (target : ℚ := 3600) 
  (t : ℕ) :
  1200 * 1.06^t > 3600 ↔ t = 17 := 
sorry

end least_months_to_exceed_triple_debt_l158_158700


namespace parabola_circle_conditions_l158_158194

theorem parabola_circle_conditions 
  (x y : ℝ)
  (F : ℝ × ℝ := (0, 1))
  (C : x ^ 2 = 4 * y)
  (E : (x - 2) ^ 2 + y ^ 2 = 1)
  (l1_tangent_to_A : ∃ A : ℝ × ℝ, (fst A - 2) ^ 2 + (snd A) ^ 2 = 1)
  (l2_tangent_to_B : ∃ B : ℝ × ℝ, (fst B - 2) ^ 2 + (snd B) ^ 2 = 1)
  (l1_intersects_parabola : ∃ M N : ℝ × ℝ, fst M ^ 2 = 4 * snd M ∧ fst N ^ 2 = 4 * snd N ∧ (M = N → false))
  (l2_intersects_parabola : ∃ P Q : ℝ × ℝ, fst P ^ 2 = 4 * snd P ∧ fst Q ^ 2 = 4 * snd Q ∧ (P = Q → false))
: 
  (∀ F A B : ℝ × ℝ, (F = (0, 1)) ∧ (fst A - 2) ^ 2 + (snd A) ^ 2 = 1 ∧ (fst B - 2) ^ 2 + (snd B) ^ 2 = 1 → 
  (fst A) ^ 2 + (snd A) ^ 2 - 2 * (fst A) - (snd A) = 0) ∧
  (∀ A B : ℝ × ℝ, (fst A - 2) ^ 2 + (snd A) ^ 2 = 1 ∧ (fst B - 2) ^ 2 + (snd B) ^ 2 = 1 → dist A B = 4 * sqrt 5 / 5) ∧
  (∀ M N P Q : ℝ × ℝ, (fst M ^ 2 = 4 * snd M) ∧ (fst N ^ 2 = 4 * snd N) ∧ (fst P ^ 2 = 4 * snd P) ∧ (fst Q ^ 2 = 4 * snd Q) 
     → dist M N + dist P Q = 136 / 9)
:= sorry

end parabola_circle_conditions_l158_158194


namespace percentage_increase_after_decrease_l158_158765

variable (P : ℝ) (x : ℝ)

-- Conditions
def decreased_price : ℝ := 0.80 * P
def final_price_condition : Prop := 0.80 * P + (x / 100) * (0.80 * P) = 1.04 * P
def correct_answer : Prop := x = 30

-- The proof goal
theorem percentage_increase_after_decrease : final_price_condition P x → correct_answer x :=
by sorry

end percentage_increase_after_decrease_l158_158765


namespace circle_equation_l158_158478

theorem circle_equation
  (a b r : ℝ)
  (ha : (4 - a)^2 + (1 - b)^2 = r^2)
  (hb : (2 - a)^2 + (1 - b)^2 = r^2)
  (ht : (b - 1) / (a - 2) = -1) :
  (a = 3) ∧ (b = 0) ∧ (r = 2) :=
by {
  sorry
}

-- Given the above values for a, b, r
def circle_equation_verified : Prop :=
  (∀ (x y : ℝ), ((x - 3)^2 + y^2) = 4)

example : circle_equation_verified :=
by {
  sorry
}

end circle_equation_l158_158478


namespace problem1_problem2_l158_158467

-- Problem 1: Lean 4 Statement
theorem problem1 (n : ℕ) (hn : n > 0) : 20 ∣ (4 * 6^n + 5^(n + 1) - 9) :=
sorry

-- Problem 2: Lean 4 Statement
theorem problem2 : (3^100 % 7) = 4 :=
sorry

end problem1_problem2_l158_158467


namespace race_problem_l158_158876

theorem race_problem (x : ℕ) (A : Point) (B : Point)  
  (dist_AB : dist A B = x)
  (meet : dist A meet_point = 24)
  (speed_A_half : speed A' = (speed A) / 2)
  (catch_up : dist B finish = 48)
   (A_reaches_B_first : reaches_first A B)
  : dist A finish = 16 :=   
by
  -- Given conditions recognized and accepted by Lean as logical constructs
  sorry

end race_problem_l158_158876


namespace arccos_of_one_over_sqrt_two_l158_158890

theorem arccos_of_one_over_sqrt_two :
  Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 := 
sorry

end arccos_of_one_over_sqrt_two_l158_158890


namespace average_of_11_results_l158_158034

theorem average_of_11_results 
  (S1: ℝ) (S2: ℝ) (fifth_result: ℝ) -- Define the variables
  (h1: S1 / 5 = 49)                -- sum of the first 5 results
  (h2: S2 / 7 = 52)                -- sum of the last 7 results
  (h3: fifth_result = 147)         -- the fifth result 
  : (S1 + S2 - fifth_result) / 11 = 42 := -- statement of the problem
by
  sorry

end average_of_11_results_l158_158034


namespace meal_combinations_count_l158_158875

/-- Crystal can choose a meal combination from the following options:
1. 4 entrees: Pizza, Chicken Teriyaki, Corn Dog, Fish and Chips.
2. 3 drinks: Lemonade, Root Beer, Iced Tea.
3. 3 desserts: Frozen Yogurt, Chocolate Chip Cookie, Apple Pie.

There is a special condition that "Apple Pie" dessert is only available if "Iced Tea" is chosen as the drink.

Prove that there are 20 distinct possible meals Crystal can buy. -/
theorem meal_combinations_count : 
  let entrees := 4 in
  let drinks := 3 in
  let desserts := 3 in
  let special_condition := ∀ chosen_drink,
    if chosen_drink = "Iced Tea" then 1 else 2 in
  ∃ combinations : ℕ, 
    combinations = entrees * (if drinks = 3 then 1 * 1 + 2 * 2 else 0) ∧
    combinations = 20 :=
by
  let entrees := 4
  let drinks := 3
  let desserts := 3
  let special_condition := ∀ chosen_drink,
    if chosen_drink = "Iced Tea" then 1 else 2
  have : 20 = 4 * (1 * 1 + 2 * 2),
    by simp
  exact ⟨20, this⟩

sorry

end meal_combinations_count_l158_158875


namespace are_opposites_l158_158052
noncomputable theory

def opposite (a b : Int) : Prop :=
  a = -b

theorem are_opposites : 
  opposite (-|(-3 : Int)|) (-( -3)) :=
by
  unfold opposite
  sorry

end are_opposites_l158_158052


namespace minimum_perimeter_triangle_ABC_l158_158719

noncomputable def point := (ℝ × ℝ)

def y_axis (B : point) : Prop := B.1 = 0

def line_l (C : point) : Prop := C.1 - C.2 - 2 = 0

def A : point := (2, 3)

def distance (P Q : point) : ℝ := real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

def perimeter (A B C : point) : ℝ := distance A B + distance B C + distance C A

theorem minimum_perimeter_triangle_ABC :
  ∃ (B C : point), y_axis B ∧ line_l C ∧ perimeter A B C = 3 * real.sqrt 2 :=
sorry

end minimum_perimeter_triangle_ABC_l158_158719


namespace line_circle_no_intersection_l158_158610

theorem line_circle_no_intersection :
  ∀ (x y : ℝ), (3 * x + 4 * y = 12) ∧ (x^2 + y^2 = 4) → false :=
sorry

end line_circle_no_intersection_l158_158610


namespace correctStatement_l158_158801

def isValidInput : String → Bool
| "INPUT a, b, c;" => true
| "INPUT x=3;" => false
| _ => false

def isValidOutput : String → Bool
| "PRINT 20,3*2." => true
| "PRINT A=4;" => false
| _ => false

def isValidStatement : String → Bool
| stmt => (isValidInput stmt ∨ isValidOutput stmt)

theorem correctStatement : isValidStatement "PRINT 20,3*2." = true ∧ 
                           ¬(isValidStatement "INPUT a; b; c;" = true) ∧ 
                           ¬(isValidStatement "INPUT x=3;" = true) ∧ 
                           ¬(isValidStatement "PRINT A=4;" = true) := 
by sorry

end correctStatement_l158_158801


namespace product_of_sums_of_two_squares_l158_158295

theorem product_of_sums_of_two_squares
  (a b a1 b1 : ℤ) :
  ((a^2 + b^2) * (a1^2 + b1^2)) = ((a * a1 - b * b1)^2 + (a * b1 + b * a1)^2) := 
sorry

end product_of_sums_of_two_squares_l158_158295


namespace number_of_tangerines_l158_158407

theorem number_of_tangerines (total_fruits bananas apples pears : ℕ) 
  (h_total_fruits : total_fruits = 60)
  (h_bananas : bananas = 32)
  (h_apples : apples = 10)
  (h_pears : pears = 5) :
  ∃ tangerines, tangerines = total_fruits - (bananas + apples + pears) ∧ tangerines = 13 :=
by
  use total_fruits - (bananas + apples + pears)
  split
  · rfl
  · rw [h_total_fruits, h_bananas, h_apples, h_pears]
    norm_num

end number_of_tangerines_l158_158407


namespace line_passes_through_point_line_perpendicular_to_x_plus_y_plus_one_l158_158189

noncomputable def line (λ : ℝ) : ℝ × ℝ → Prop :=
  λ ⟨x, y⟩, 2 * x + y - 3 + λ * (x - 2 * y + 1) = 0

-- Prove that the line passes through the point (1,1) for any real number λ
theorem line_passes_through_point (λ : ℝ) : line λ (1, 1) :=
begin
  simp [line],
  ring,
end

noncomputable def slope (a b c : ℝ) : ℝ := -a / b

noncomputable def line_slope (λ : ℝ) : ℝ :=
  slope (2 + λ) (1 - 2 * λ) (-3 + λ)

-- Prove that the line is perpendicular to x + y + 1 = 0 if and only if λ = 3
theorem line_perpendicular_to_x_plus_y_plus_one (λ : ℝ) :
  λ = 3 ↔ ∃ x y, line λ (x, y) ∧ slope (2 + λ) (1 - 2 * λ) = -1 :=
begin
  split,
  {
    intro hλ,
    use [1, 1],
    simp [line, slope, hλ],
    ring,
  },
  {
    rintro ⟨x, y, hline, hslope⟩,
    simp [line, slope] at *,
    sorry
  }
end

end line_passes_through_point_line_perpendicular_to_x_plus_y_plus_one_l158_158189


namespace probability_even_sum_l158_158255

def first_ten_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def is_odd (n : ℕ) : Prop := n % 2 = 1

def count_odd_primes (primes : List ℕ) : ℕ :=
  primes.countp is_odd

def binom (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

open Nat

theorem probability_even_sum :
  let primes := first_ten_primes in
  let odd_primes := count_odd_primes primes in
  let num_ways_even_sum := binom odd_primes 2 in
  let total_ways := binom primes.length 2 in
  (num_ways_even_sum : ℚ) / (total_ways : ℚ) = 4 / 5 :=
by
  sorry

end probability_even_sum_l158_158255


namespace midpoint_locus_point_Z_locus_l158_158689

noncomputable theory

-- Definitions of Points based on given cubes A, B, C, D, A', B', C', D'
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

-- The points of the cuboid
def A : Point := ⟨0, 0, 0⟩
def B : Point := ⟨a, 0, 0⟩
def C : Point := ⟨a, b, 0⟩
def D : Point := ⟨0, b, 0⟩
def A' : Point := ⟨0, 0, c⟩
def B' : Point := ⟨a, 0, c⟩
def C' : Point := ⟨a, b, c⟩
def D' : Point := ⟨0, b, c⟩

-- X on AC
def X (k : ℝ) : Point := ⟨(k * a) / (k + 1), (k * b) / (k + 1), 0⟩
-- Y on B'D'
def Y (m : ℝ) : Point := ⟨0, (m * b) / (m + 1), c⟩

-- Midpoint of X and Y
def M (k m : ℝ) : Point :=
  ⟨((k * a) / (k + 1)) / 2, (((k * b) / (k + 1)) + (m * b) / (m + 1)) / 2, c / 2⟩

-- Point Z on XY where ZY = 2XZ
def Z (k m : ℝ) : Point :=
  ⟨(2 * (k * a) / (k + 1)) / 3, (2 * (k * b) / (k + 1) + (m * b) / (m + 1)) / 3, c / 3⟩

-- Problem statements

theorem midpoint_locus (a b c k m : ℝ): 
  ∃ x y z, M k m = ⟨x, y, z⟩ ∧ x = a / 2 ∧ z = c / 2 :=
sorry

theorem point_Z_locus (a b c k m : ℝ):
  ∃ x y z, Z k m = ⟨x, y, z⟩ ∧ Z k m = ⟨(2 * k * a) / (3 * (k + 1)), (k * b) / (k + 1), c / 3⟩ :=
sorry

end midpoint_locus_point_Z_locus_l158_158689


namespace length_of_AB_is_7_l158_158400

-- Define the triangles ABC and DEF with their given side lengths.
structure Triangle where
  A B C D : Point
  AB BC CA DE EF FD : ℝ
  angleBAC angleFDE : ℝ

-- Define the given information.
def triangleABCDEF : Triangle := {
  A := ⟨0, 0⟩,
  B := ⟨0, 0⟩,
  C := ⟨0, 0⟩,
  D := ⟨0, 0⟩,
  AB := 7,
  BC := 14,
  CA := 10,
  DE := 3,
  EF := 6,
  FD := 5,
  angleBAC := (2 / 3) * Real.pi,
  angleFDE := (2 / 3) * Real.pi
}

-- Define the similarity and proportionality of sides.
def similarTriangles (T1 T2 : Triangle) : Prop :=
  T1.angleBAC = T2.angleFDE ∧
  T1.AB / T2.DE = T1.BC / T2.EF

-- The target theorem to prove the length of AB is 7 cm.
theorem length_of_AB_is_7 (ABC DEF : Triangle) :
  similarTriangles ABC DEF →
  ABC.BC = 14 →
  DEF.EF = 6 →
  DEF.DE = 3 →
  ABC.AB = 7 :=
by
  sorry

end length_of_AB_is_7_l158_158400


namespace probability_sum_even_l158_158289

def primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def sum_is_even (a b : ℕ) : Prop :=
  (a + b) % 2 = 0

theorem probability_sum_even :
  (nat.choose 10 2) = 45 →
  (∀ a b : ℕ, a ∈ primes → b ∈ primes → a ≠ b →
    (sum_is_even a b ↔ a ≠ 2 ∧ b ≠ 2)) →
  ((45 - 9) / 45 : ℚ) = 4 / 5 :=
by sorry

end probability_sum_even_l158_158289


namespace arccos_one_over_sqrt_two_eq_pi_over_four_l158_158931

theorem arccos_one_over_sqrt_two_eq_pi_over_four : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_eq_pi_over_four_l158_158931


namespace range_of_a_l158_158601

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → |2^x - a| < |5 - 2^x|) → (3 < a ∧ a < 5) :=
by
  intro h,
  sorry

end range_of_a_l158_158601


namespace pipe_q_fills_in_9_hours_l158_158716

theorem pipe_q_fills_in_9_hours (x : ℝ) :
  (1 / 3 + 1 / x + 1 / 18 = 1 / 2) → x = 9 :=
by {
  sorry
}

end pipe_q_fills_in_9_hours_l158_158716


namespace squirrel_prob_l158_158305

noncomputable def P : ℕ → ℚ
| 0 := 0
| 14 := 1
| 7 := 1/2 * P 6 + 1/2 * P 8
| n := if 0 < n ∧ n < 14 then (n / 14) * P (n - 1) + (1 - n / 14) * P (n + 1) else 0

theorem squirrel_prob : P 3 = 243 / 512 := 
sorry

end squirrel_prob_l158_158305


namespace ratio_of_semicircles_to_main_circle_l158_158301

theorem ratio_of_semicircles_to_main_circle
    (r : ℝ) 
    (r_pos : 0 < r)
    (diameter_semicircles_eq_radius_main_circle : ∀ A B C D E F O, 
        (A ≠ B ∧ C ≠ D ∧ E ≠ F ∧ O ≠ A ∧ O ≠ B ∧ O ≠ C ∧ O ≠ D ∧ O ≠ E ∧ O ≠ F) → 
        let radius_semicircle := r / 2 in 
        let area_semicircle := π * (radius_semicircle)^2 / 2 in 
        let combined_area_semicircles := 2 * area_semicircle in 
        let area_main_circle := π * r^2 in
        combined_area_semicircles / area_main_circle = 1 / 4) 
    : ℝ :=
ratio_of_semicircles_to_main_circle r sorry -- We leave the proof as sorry

end ratio_of_semicircles_to_main_circle_l158_158301


namespace bicycle_distance_l158_158843

def distance : ℝ := 15

theorem bicycle_distance :
  ∀ (x y : ℝ),
  (x + 6) * (y - 5 / 60) = x * y →
  (x - 5) * (y + 6 / 60) = x * y →
  x * y = distance :=
by
  intros x y h1 h2
  sorry

end bicycle_distance_l158_158843


namespace find_p_l158_158813

noncomputable def p (x1 x2 x3 x4 n : ℝ) :=
  (x1 + x3) * (x2 + x3) + (x1 + x4) * (x2 + x4)

theorem find_p (x1 x2 x3 x4 n : ℝ) (h1 : x1 ≠ x2)
(h2 : (x1 + x3) * (x1 + x4) = n - 10)
(h3 : (x2 + x3) * (x2 + x4) = n - 10) :
  p x1 x2 x3 x4 n = n - 20 :=
sorry

end find_p_l158_158813


namespace class_duration_l158_158672

theorem class_duration (x : ℝ) (h : 3 * x = 6) : x = 2 :=
by
  sorry

end class_duration_l158_158672


namespace compare_routes_l158_158812

variables (P C Z : Type) 
variables (length_PZ length_CZ length_PC : ℝ)

-- Conditions
axiom cond1 : length_PC + length_CZ = 3 * length_PZ
axiom cond2 : length_PZ + length_CZ = 2 * length_CZ

theorem compare_routes : length_PC < length_PZ + length_CZ := 
begin
  sorry,
end

end compare_routes_l158_158812


namespace miley_total_cost_l158_158701

-- Define the cost per cellphone
def cost_per_cellphone : ℝ := 800

-- Define the number of cellphones
def number_of_cellphones : ℝ := 2

-- Define the discount rate
def discount_rate : ℝ := 0.05

-- Define the total cost without discount
def total_cost_without_discount : ℝ := cost_per_cellphone * number_of_cellphones

-- Define the discount amount
def discount_amount : ℝ := total_cost_without_discount * discount_rate

-- Define the total cost with discount
def total_cost_with_discount : ℝ := total_cost_without_discount - discount_amount

-- Prove that the total amount Miley paid is $1520
theorem miley_total_cost : total_cost_with_discount = 1520 := by
  sorry

end miley_total_cost_l158_158701


namespace bill_needs_paint_cans_l158_158511

theorem bill_needs_paint_cans :
  let bedrooms := 3
  let other_rooms := 2 * bedrooms
  let gallons_per_room := 2
  let color_paint_cans := 6 -- (bedrooms * gallons_per_room) / 1-gallon per can
  let white_paint_cans := 4 -- (other_rooms * gallons_per_room) / 3-gallons per can
  (color_paint_cans + white_paint_cans) = 10 := sorry

end bill_needs_paint_cans_l158_158511


namespace intersection_with_complement_l158_158771

-- Definitions for the universal set and set A
def U : Set ℝ := Set.univ

def A : Set ℝ := { -1, 0, 1 }

-- Definition for set B using the given condition
def B : Set ℝ := { x : ℝ | (x - 2) / (x + 1) > 0 }

-- Definition for the complement of B
def B_complement : Set ℝ := { x : ℝ | -1 <= x ∧ x <= 0 }

-- Theorem stating the intersection of A and the complement of B equals {-1, 0, 1}
theorem intersection_with_complement : 
  A ∩ B_complement = { -1, 0, 1 } :=
by
  sorry

end intersection_with_complement_l158_158771


namespace optionB_is_opposites_l158_158053

-- Define the pairs of numbers
def optionA_pair : ℕ × ℕ := (3, abs (-3))
def optionB_pair : ℤ × ℤ := (-abs (-3), -(-3))
def optionC_pair : ℤ × ℚ := (-3, -1/3)
def optionD_pair : ℤ × ℚ := (-3, 1/3)

-- Define what it means for two numbers to be opposite
def are_opposites (a b : ℤ) : Prop := a = -b

-- Prove that the pairs in optionB are opposite
theorem optionB_is_opposites : are_opposites (optionB_pair.1) (optionB_pair.2) :=
by
  unfold optionB_pair are_opposites
  simp
  sorry

end optionB_is_opposites_l158_158053


namespace arccos_one_over_sqrt_two_eq_pi_over_four_l158_158924

theorem arccos_one_over_sqrt_two_eq_pi_over_four : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_eq_pi_over_four_l158_158924


namespace john_paid_percentage_l158_158029

theorem john_paid_percentage (SRP WP : ℝ) (h1 : SRP = 1.40 * WP) (h2 : ∀ P, P = (1 / 3) * SRP) : ((1 / 3) * SRP / SRP * 100) = 33.33 :=
by
  sorry

end john_paid_percentage_l158_158029


namespace infinite_series_sum_l158_158988

theorem infinite_series_sum : 
  let S := λ (n : ℕ), (if n % 3 = 0 then (-1 : ℚ)^(n / 3) * 3^((n / 3) - 1)
                       else (if (n + 2) % 3 = 0 then (-1 : ℚ)^(n / 3 + 1) * 3^((n / 3) - 1) 
                       else 0)) 
  in ∑' n, S n = 15 / 26 :=
by
  sorry

end infinite_series_sum_l158_158988


namespace non_acute_angles_l158_158457

theorem non_acute_angles (n : ℕ) (O : Point) (vertices : Fin n → Point)
  (h_convex : convex_hull (range vertices) O) :
  ∃ (non_acutes : Finset (Fin n × Fin n)),
  non_acutes.card ≥ n - 1 ∧
  ∀ ⟨i, j⟩ ∈ non_acutes, i ≠ j ∧ ¬is_acute (angle O (vertices i) (vertices j)) :=
begin
  sorry -- Proof omitted
end

end non_acute_angles_l158_158457


namespace smallest_positive_integer_divisible_by_four_distinct_primes_l158_158433

theorem smallest_positive_integer_divisible_by_four_distinct_primes :
  ∀ n : ℕ, (∀ p ∈ {2, 3, 5, 7}, p ∣ n) → n ≥ 210 :=
by
  intro n
  intro h
  sorry

end smallest_positive_integer_divisible_by_four_distinct_primes_l158_158433


namespace children_sum_zero_l158_158549

theorem children_sum_zero (a b c d e : ℤ) :
  let final_a := a * (e + d - b - c),
      final_b := b * (a + e - c - d),
      final_c := c * (b + a - d - e),
      final_d := d * (c + b - e - a),
      final_e := e * (d + c - a - b) in
  final_a + final_b + final_c + final_d + final_e = 0 :=
by 
  sorry

end children_sum_zero_l158_158549


namespace arccos_one_over_sqrt_two_eq_pi_four_l158_158912

theorem arccos_one_over_sqrt_two_eq_pi_four : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 := 
by
  sorry

end arccos_one_over_sqrt_two_eq_pi_four_l158_158912


namespace probability_sum_even_l158_158294

def primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def sum_is_even (a b : ℕ) : Prop :=
  (a + b) % 2 = 0

theorem probability_sum_even :
  (nat.choose 10 2) = 45 →
  (∀ a b : ℕ, a ∈ primes → b ∈ primes → a ≠ b →
    (sum_is_even a b ↔ a ≠ 2 ∧ b ≠ 2)) →
  ((45 - 9) / 45 : ℚ) = 4 / 5 :=
by sorry

end probability_sum_even_l158_158294


namespace ratio_37m48s_2h13m15s_l158_158048

-- Define the total seconds for 37 minutes and 48 seconds
def t1 := 37 * 60 + 48

-- Define the total seconds for 2 hours, 13 minutes, and 15 seconds
def t2 := 2 * 3600 + 13 * 60 + 15

-- Prove the ratio t1 / t2 = 2268 / 7995
theorem ratio_37m48s_2h13m15s : t1 / t2 = 2268 / 7995 := 
by sorry

end ratio_37m48s_2h13m15s_l158_158048


namespace average_speed_l158_158028

theorem average_speed (d1 d2 : ℕ) (t1 t2 : ℕ) (h1 : d1 = 120) (h2 : d2 = 70) (ht1 : t1 = 1) (ht2 : t2 = 1) :
  (d1 + d2) / (t1 + t2) = 95 :=
by
  -- assume d1 = 120, d2 = 70, t1 = 1, t2 = 1
  rw [h1, h2, ht1, ht2]
  -- simplify the expression (120 + 70) / (1 + 1)
  sorry

end average_speed_l158_158028


namespace profit_23_percent_of_cost_price_l158_158842

-- Define the conditions
variable (C : ℝ) -- Cost price of the turtleneck sweaters
variable (C_nonneg : 0 ≤ C) -- Ensure cost price is non-negative

-- Definitions based on conditions
def SP1 (C : ℝ) : ℝ := 1.20 * C
def SP2 (SP1 : ℝ) : ℝ := 1.25 * SP1
def SPF (SP2 : ℝ) : ℝ := 0.82 * SP2

-- Define the profit calculation
def Profit (C : ℝ) : ℝ := (SPF (SP2 (SP1 C))) - C

-- Statement of the theorem
theorem profit_23_percent_of_cost_price (C : ℝ) (C_nonneg : 0 ≤ C):
  Profit C = 0.23 * C :=
by
  -- The actual proof would go here
  sorry

end profit_23_percent_of_cost_price_l158_158842


namespace percent_of_area_triangle_in_pentagon_l158_158491

-- Defining a structure for the problem statement
structure PentagonAndTriangle where
  s : ℝ -- side length of the equilateral triangle
  side_square : ℝ -- side of the square
  area_triangle : ℝ
  area_square : ℝ
  area_pentagon : ℝ

noncomputable def calculate_areas (s : ℝ) : PentagonAndTriangle :=
  let height_triangle := s * (Real.sqrt 3) / 2
  let area_triangle := Real.sqrt 3 / 4 * s^2
  let area_square := height_triangle^2
  let area_pentagon := area_square + area_triangle
  { s := s, side_square := height_triangle, area_triangle := area_triangle, area_square := area_square, area_pentagon := area_pentagon }

/--
Prove that the percentage of the pentagon's area that is the area of the equilateral triangle is (3 * (Real.sqrt 3 - 1)) / 6 * 100%.
-/
theorem percent_of_area_triangle_in_pentagon 
  (s : ℝ) 
  (pt : PentagonAndTriangle)
  (h₁ : pt = calculate_areas s)
  : pt.area_triangle / pt.area_pentagon = (3 * (Real.sqrt 3 - 1)) / 6 * 100 :=
by
  sorry

end percent_of_area_triangle_in_pentagon_l158_158491


namespace molecular_weight_correct_l158_158431

-- Define the atomic weights
def atomic_weight_K : ℝ := 39.10
def atomic_weight_Br : ℝ := 79.90
def atomic_weight_O : ℝ := 16.00
def atomic_weight_H : ℝ := 1.01
def atomic_weight_N : ℝ := 14.01

-- Define the number of atoms of each element
def num_atoms_K : ℕ := 2
def num_atoms_Br : ℕ := 2
def num_atoms_O : ℕ := 4
def num_atoms_H : ℕ := 3
def num_atoms_N : ℕ := 1

-- Calculate the molecular weight
def molecular_weight : ℝ :=
  num_atoms_K * atomic_weight_K +
  num_atoms_Br * atomic_weight_Br +
  num_atoms_O * atomic_weight_O +
  num_atoms_H * atomic_weight_H +
  num_atoms_N * atomic_weight_N

-- Define the expected molecular weight
def expected_molecular_weight : ℝ := 319.04

-- The theorem stating that the calculated molecular weight matches the expected molecular weight
theorem molecular_weight_correct : molecular_weight = expected_molecular_weight :=
  by
  sorry -- Proof is skipped

end molecular_weight_correct_l158_158431


namespace jefferson_speed_l158_158327

def speed_of_horse (distance : ℝ) (flat_fee : ℝ) (hourly_rate : ℝ) (total_payment : ℝ) : ℝ :=
  let hourly_charge := total_payment - flat_fee
  let time_spent := hourly_charge / hourly_rate
  distance / time_spent

theorem jefferson_speed :
  speed_of_horse 20 20 30 80 = 10 :=
by 
  sorry

end jefferson_speed_l158_158327


namespace sum_geom_series_eq_l158_158418

-- Definition that 1 + a + a^2 + ... + a^(n+2) equals (1 - a^(n+3)) / (1 - a) for a != 1 and n in ℕ*
theorem sum_geom_series_eq (a : ℝ) (h : a ≠ 1) (n : ℕ) (hn : n > 0) : 
  ∑ i in Finset.range (n + 3), a ^ i = (1 - a ^ (n + 3)) / (1 - a) :=
sorry

end sum_geom_series_eq_l158_158418


namespace hands_overlap_l158_158507

-- Definitions of various angles and movement rates
def angle3OC := 90 -- angle in degrees at 3:00
def minuteHandMovement := 6 -- degrees per minute
def hourHandMovement := 0.5 -- degrees per minute

-- Prove the time x when hands overlap after 3:00
theorem hands_overlap (x : ℕ) : 
  x * minuteHandMovement = angle3OC + x * hourHandMovement → x = 16 := 
  by
  sorry

end hands_overlap_l158_158507


namespace quadratic_real_roots_l158_158632

theorem quadratic_real_roots (k : ℝ) : (∃ x : ℝ, k * x^2 - 6 * x + 9 = 0) ↔ (k ≤ 1 ∧ k ≠ 0) :=
  sorry

end quadratic_real_roots_l158_158632


namespace vlad_score_l158_158375

theorem vlad_score :
  ∀ (rounds wins : ℕ) (totalPoints taroPoints vladPoints : ℕ),
    rounds = 30 →
    (wins = 5) →
    (totalPoints = rounds * wins) →
    (taroPoints = (3 * totalPoints) / 5 - 4) →
    (vladPoints = totalPoints - taroPoints) →
    vladPoints = 64 :=
by
  intros rounds wins totalPoints taroPoints vladPoints h1 h2 h3 h4 h5
  sorry

end vlad_score_l158_158375


namespace triangle_area_ratio_l158_158820

noncomputable def triangle (A B C : Type) : Type := sorry

def points_in_segment (A B : Type) (K : A) (N : B) : Prop := sorry

def proportional_segments (BK KA CN BN : ℕ) : Prop :=
  BK = 2 * KA ∧ CN = 2 * BN

def concurrent_points (A N : Type) (C K : Type) (Q : Type) : Prop := sorry

def area_ratio (ABC BCQ : Type) : ℚ := 7 / 4

theorem triangle_area_ratio (A B C K N Q : Type) (h1 : triangle A B C)
  (h2 : points_in_segment A B K N) (h3 : proportional_segments 2 1 2 1)
  (h4 : concurrent_points A N C K Q) :
  area_ratio (triangle A B C) (triangle B C Q) = 7 / 4 := 
  sorry

end triangle_area_ratio_l158_158820


namespace area_BEIH_l158_158057

noncomputable def B := (0 : ℝ, 0 : ℝ)
noncomputable def A := (0 : ℝ, 3 : ℝ)
noncomputable def D := (3 : ℝ, 3 : ℝ)
noncomputable def C := (3 : ℝ, 0 : ℝ)
noncomputable def E := (0 : ℝ, 1.5 : ℝ) -- midpoint of AB
noncomputable def F := (1 : ℝ, 0 : ℝ) -- one-third from B to C along BC
noncomputable def I := (3 / 5, 12 / 5) -- intersection point of DE and AF
noncomputable def H := (2 / 3, 2 / 3) -- intersection point of BD and AF

def shoelace_area (points : List (ℝ × ℝ)) : ℝ :=
  (1/2) * (List.sum (List.zipWith (λ (p q : ℝ × ℝ), p.1 * q.2) points (List.tail points ++ [points.head!])) -
           List.sum (List.zipWith (λ (p q : ℝ × ℝ), p.2 * q.1) points (List.tail points ++ [points.head!])))

theorem area_BEIH : shoelace_area [B, E, I, H] = 27 / 20 :=
by
  -- Coordinates are as follows:
  -- B = (0, 0), E = (0, 1.5), I = (3/5, 12/5), H = (2/3, 2/3)
  -- Area calculation using shoelace formula
  sorry

end area_BEIH_l158_158057


namespace simplify_and_evaluate_l158_158736

noncomputable def my_expression (m : ℝ) : ℝ :=
  (m - (m + 9) / (m + 1)) / ((m ^ 2 + 3 * m) / (m + 1))

theorem simplify_and_evaluate : my_expression (Real.sqrt 3) = 1 - Real.sqrt 3 :=
by
  sorry

end simplify_and_evaluate_l158_158736


namespace largest_average_is_28_l158_158055

-- Define the sequence of multiples within a range
def multiples_in_range (k n : ℕ) : List ℕ :=
  List.filter (λ x, x % k = 0) (List.range (n + 1))

-- Define the average of a list of natural numbers
def average (l : List ℕ) : ℝ :=
  l.sum / l.length

-- Define the specific sets of multiples between 1 and 51
def multiples_of_3 : List ℕ := multiples_in_range 3 51
def multiples_of_4 : List ℕ := multiples_in_range 4 51
def multiples_of_5 : List ℕ := multiples_in_range 5 51
def multiples_of_6 : List ℕ := multiples_in_range 6 51
def multiples_of_7 : List ℕ := multiples_in_range 7 51

-- Define the averages of these sets
def avg_3 : ℝ := average multiples_of_3
def avg_4 : ℝ := average multiples_of_4
def avg_5 : ℝ := average multiples_of_5
def avg_6 : ℝ := average multiples_of_6
def avg_7 : ℝ := average multiples_of_7

theorem largest_average_is_28 :
  max (max (max avg_3 avg_4) (max avg_5 avg_6)) avg_7 = avg_7 ∧ avg_7 = 28 :=
by
  sorry

end largest_average_is_28_l158_158055


namespace tangent_function_intersection_l158_158592

theorem tangent_function_intersection (ω : ℝ) (hω : ω > 0) (h_period : (π / ω) = 3 * π) :
  let f (x : ℝ) := Real.tan (ω * x + π / 3)
  f π = -Real.sqrt 3 :=
by
  sorry

end tangent_function_intersection_l158_158592


namespace arccos_identity_l158_158919

theorem arccos_identity : 
  ∀ x : ℝ, x = 1 / real.sqrt 2 → real.arccos x = real.pi / 4 :=
begin
  intros x hx,
  have h1 : real.cos (real.pi / 4) = 1 / real.sqrt 2 := by sorry,
  have h2 : real.arccos (1 / real.sqrt 2) = real.pi / 4,
  { rw ← h1,
    rw real.arccos_cos,
    exact_mod_cast hx },
  exact h2,
end

end arccos_identity_l158_158919


namespace beth_crayon_packs_l158_158108

theorem beth_crayon_packs (P : ℕ) (h1 : 10 * P + 6 = 46) : P = 4 :=
by
  sorry

end beth_crayon_packs_l158_158108


namespace expression_on_neg_infinity_l158_158184

noncomputable theory

def odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

theorem expression_on_neg_infinity (f : ℝ → ℝ) (h_odd : odd_function f) 
    (h_pos : ∀ x, 0 < x → f x = x * (1 - x)) : 
    ∀ x, x < 0 → f x = x * (1 + x) :=
by
  sorry

end expression_on_neg_infinity_l158_158184


namespace initial_weight_in_pounds_l158_158493

variable (initial_weight_kg : ℝ) (final_weight_kg : ℝ)
variable (conversion_factor : ℝ)

-- Provide all the conditions from the problem
def retains_after_first_stage (w : ℝ) : ℝ := (3/5) * w
def retains_after_second_stage (w : ℝ) : ℝ := (7/10) * w
def retains_after_third_stage (w : ℝ) : ℝ := (3/4) * w
def retains_after_fourth_stage (w : ℝ) : ℝ := (3/8) * w
def retains_after_fifth_stage (w : ℝ) : ℝ := (5/7) * w

theorem initial_weight_in_pounds (W : ℝ)
  (h1 : W * 3/5 * 7/10 * 3/4 * 3/8 * 5/7 = 68.2)
  (h2 : final_weight_kg = 68.2)
  (h3 : conversion_factor = 2.205) :
  (W * conversion_factor ≈ 1782.993) :=
by
  sorry

end initial_weight_in_pounds_l158_158493


namespace john_total_distance_l158_158670

theorem john_total_distance : 
  let daily_distance := 1700
  let days_run := 6
  daily_distance * days_run = 10200 :=
by
  sorry

end john_total_distance_l158_158670


namespace log_sum_l158_158151

variable {a : ℕ → ℝ}

-- Definition of a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- Conditions
axiom a_geometric : is_geometric_sequence a
axiom a_positive : ∀ n, 0 < a n 
axiom a3_a6_eight : a 3 * a 6 = 8

-- Theorem to prove
theorem log_sum : (∑ i in Finset.range 8, Real.logb 2 (a (i + 1))) = 12 :=
by
  -- The theorem statement
  sorry

end log_sum_l158_158151


namespace common_tangent_lines_count_l158_158004

def center (x₀ y₀ r : ℝ) : ℝ × ℝ := (x₀, y₀)
def radius (r : ℝ) : ℝ := r

noncomputable def number_of_common_tangents (c₁ c₂ : ℝ × ℝ) (r₁ r₂ : ℝ) : ℕ :=
  let d := dist c₁ c₂ in
  if d = r₁ + r₂ then 3 else sorry

theorem common_tangent_lines_count :
  let c₁ := center 4 0 3
  let c₂ := center 0 3 2
  let r₁ := 3
  let r₂ := 2
  number_of_common_tangents c₁ c₂ r₁ r₂ = 3 := 
sorry

end common_tangent_lines_count_l158_158004


namespace aang_caught_7_fish_l158_158859

theorem aang_caught_7_fish (A : ℕ) (h_avg : (A + 5 + 12) / 3 = 8) : A = 7 :=
by
  sorry

end aang_caught_7_fish_l158_158859


namespace digit_200_of_5_div_13_is_8_l158_158427

/-- Prove that the 200th digit beyond the decimal point in the decimal representation
    of 5/13 is 8 --/
theorem digit_200_of_5_div_13_is_8 :
  let repeating_sequence := "384615" in
  let digit_200 := repeating_sequence[200 % 6 -1] in
  digit_200 = '8' :=
by
  sorry

end digit_200_of_5_div_13_is_8_l158_158427


namespace cricket_run_rate_l158_158808

theorem cricket_run_rate
  (run_rate_first_10_overs : ℝ)
  (overs_first_10_overs : ℕ)
  (target_runs : ℕ)
  (remaining_overs : ℕ)
  (run_rate_required : ℝ) :
  run_rate_first_10_overs = 3.2 →
  overs_first_10_overs = 10 →
  target_runs = 242 →
  remaining_overs = 40 →
  run_rate_required = 5.25 →
  (target_runs - (run_rate_first_10_overs * overs_first_10_overs)) = 210 →
  (target_runs - (run_rate_first_10_overs * overs_first_10_overs)) / remaining_overs = run_rate_required :=
by
  sorry

end cricket_run_rate_l158_158808


namespace new_fraction_value_l158_158242

def original_fraction : ℚ := 1 / 12

def increased_numerator (n : ℚ) := n + 0.2 * n

def decreased_denominator (d : ℚ):= d - 0.25 * d

theorem new_fraction_value :
  let new_fraction := increased_numerator (1) / decreased_denominator (12)
  in new_fraction = 0.13333333333333333 := by
  sorry

end new_fraction_value_l158_158242


namespace paint_cans_needed_l158_158510

theorem paint_cans_needed
    (num_bedrooms : ℕ)
    (num_other_rooms : ℕ)
    (total_rooms : ℕ)
    (gallons_per_room : ℕ)
    (color_paint_cans_per_gallon : ℕ)
    (white_paint_cans_per_gallon : ℕ)
    (total_paint_needed : ℕ)
    (color_paint_cans_needed : ℕ)
    (white_paint_cans_needed : ℕ)
    (total_paint_cans : ℕ)
    (h1 : num_bedrooms = 3)
    (h2 : num_other_rooms = 2 * num_bedrooms)
    (h3 : total_rooms = num_bedrooms + num_other_rooms)
    (h4 : gallons_per_room = 2)
    (h5 : total_paint_needed = total_rooms * gallons_per_room)
    (h6 : color_paint_cans_per_gallon = 1)
    (h7 : white_paint_cans_per_gallon = 3)
    (h8 : color_paint_cans_needed = num_bedrooms * gallons_per_room * color_paint_cans_per_gallon)
    (h9 : white_paint_cans_needed = (num_other_rooms * gallons_per_room) / white_paint_cans_per_gallon)
    (h10 : total_paint_cans = color_paint_cans_needed + white_paint_cans_needed) :
    total_paint_cans = 10 :=
by sorry

end paint_cans_needed_l158_158510


namespace total_charts_16_l158_158872

def total_charts_brought (number_of_associate_professors : Int) (number_of_assistant_professors : Int) : Int :=
  number_of_associate_professors * 1 + number_of_assistant_professors * 2

theorem total_charts_16 (A B : Int)
  (h1 : 2 * A + B = 11)
  (h2 : A + B = 9) :
  total_charts_brought A B = 16 :=
by {
  -- the proof will go here
  sorry
}

end total_charts_16_l158_158872


namespace arccos_one_over_sqrt_two_eq_pi_over_four_l158_158897

theorem arccos_one_over_sqrt_two_eq_pi_over_four : 
  Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_eq_pi_over_four_l158_158897


namespace false_statement_d_l158_158447

-- Conditions
def isosceles_triangle_with_60_is_equilateral : Prop :=
∀ (T : Triangle), is_isosceles T → (angle T = 60°) → is_equilateral T

def three_angle_bisectors_intersect_at_incenter : Prop :=
∀ (T : Triangle), let I := incenter T in (is_intersection_of_three_bisectors I T) → (∀ (P : Point), on_bisectors_equal_dist I P T)

def obtuse_triangle_has_angle_gt_90 : Prop :=
∀ (T : Triangle), is_obtuse T → (∃ α, α ∈ angles T ∧ α > 90°)

def corresponding_angles_equal : Prop :=
-- Here we need to express that "corresponding angles are equal" without assuming parallel lines, which is false in general.
∀ (α β : Angle), (corresponding α β) → (α = β)

-- Theorem to prove
theorem false_statement_d : ¬ corresponding_angles_equal :=
sorry

end false_statement_d_l158_158447


namespace distance_from_focus_l158_158846

-- Define the parabola equation
def parabola_eq (M : ℝ × ℝ) : Prop := M.2^2 = 3 * M.1

-- Define the focus of the parabola y^2 = 3x
def focus : ℝ × ℝ := (3 / 4, 0)

-- Define a point M at a distance 1 from the y-axis
def distance_from_y_axis (M : ℝ × ℝ) : Prop := abs M.1 = 1

-- The theorem to prove
theorem distance_from_focus (M : ℝ × ℝ) (h_parabola_eq : parabola_eq M) (h_distance_y_axis : distance_from_y_axis M) :
  dist M focus = 7 / 4 :=
by {
  sorry
}

end distance_from_focus_l158_158846


namespace neg_p_is_true_neg_q_is_true_l158_158451

theorem neg_p_is_true : ∃ m : ℝ, ∀ x : ℝ, (x^2 + x - m = 0 → False) :=
sorry

theorem neg_q_is_true : ∀ x : ℝ, (x^2 + x + 1 > 0) :=
sorry

end neg_p_is_true_neg_q_is_true_l158_158451


namespace percentage_loss_is_14_96_l158_158085

-- Define the cost price C based on the problem's conditions
def cost_price (C : ℝ) : Prop := 1.05 * C = 12.35

-- Define the original selling price
def original_selling_price : ℝ := 10

-- Define the loss per kg
def loss_per_kg (C : ℝ) : ℝ := C - original_selling_price

-- Define the percentage of loss
def percentage_loss (C : ℝ) : ℝ := (loss_per_kg C / C) * 100

-- State the theorem
theorem percentage_loss_is_14_96
  (C : ℝ)
  (h : cost_price C) : percentage_loss C ≈ 14.96 :=
begin
  sorry
end

end percentage_loss_is_14_96_l158_158085


namespace geometric_sequence_problem_l158_158583

variable {α : Type*} [Field α]

def is_geometric_sequence (a b c : α) : Prop :=
  ∃ q : α, b = q * a ∧ c = q * b

theorem geometric_sequence_problem (a b c d : α)
  (h₁ : is_geometric_sequence a c d)
  (h₂ : ∃ q : α, c = q * a ∧ d = q * c) :
  (is_geometric_sequence ab b (c + d) ∨ is_geometric_sequence (abbc) d ∨ is_geometric_sequence ab (b - c) (-d)) ↔ (1 : α) := 
  sorry

end geometric_sequence_problem_l158_158583


namespace sequence_has_infinitely_many_squares_l158_158399

noncomputable def sequence (n : ℕ) : ℕ := (Real.floor (Real.sqrt 2 * n))

theorem sequence_has_infinitely_many_squares :
  ∃ᶠ n in at_top, ∃ m : ℕ, sequence n = m * m := 
begin
  sorry
end

end sequence_has_infinitely_many_squares_l158_158399


namespace sampling_method_systematic_l158_158486

variable (Classes : Type) (Students : Classes → Type)

-- Assuming each class contains exactly 56 students
variable [Finite (Students : ∀ C : Classes, {s : ℕ // s < 56})]

-- Numbering of students from 1 to 56 (0 to 55 in zero-indexed)
def StudentNumber (c : Classes) := Fin (56)

-- Each class has a student numbered 14
def Student14 (c : Classes) : Students c := (14 : Fin 56)

-- Define what it means for a sampling method to be systematic: choosing the 14th student
def systematic_sampling (c : Classes) : Students c := Student14 c

theorem sampling_method_systematic (h : ∀ c : Classes, systematic_sampling c = Student14 c) :
  h = λ c, systematic_sampling c := sorry

end sampling_method_systematic_l158_158486


namespace bernoulli_inequality_l158_158341

theorem bernoulli_inequality (n : ℕ) (x : ℝ) (h : x ≥ -1) : 1 + n * x ≤ (1 + x) ^ n :=
sorry

end bernoulli_inequality_l158_158341


namespace range_of_a_l158_158215

theorem range_of_a (a : ℝ) : (∀ x : ℝ, a * x ^ 2 - |x + 1| + 2 * a ≥ 0) ↔ a ∈ (Set.Ici ((Real.sqrt 3 + 1) / 4)) := by
  sorry

end range_of_a_l158_158215


namespace domain_R_increasing_function_l158_158598

def f (a x : ℝ) : ℝ := 2^(x + a)

theorem domain_R (a : ℝ) : ∀ x : ℝ, f a x ∈ ℝ :=
by
  intro x
  exact ⟨x, rfl⟩

theorem increasing_function (a : ℝ) : ∀ x₁ x₂ : ℝ, x₁ < x₂ → f a x₁ < f a x₂ :=
by
  intros x₁ x₂ h
  sorry

end domain_R_increasing_function_l158_158598


namespace problem_part1_problem_part2_l158_158562

noncomputable def dot_product (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

noncomputable def f (x : ℝ) : ℝ :=
  dot_product (Real.cos x, Real.cos x) (Real.sqrt 3 * Real.cos x, Real.sin x)

theorem problem_part1 :
  (∀ x : ℝ, f (x + π) = f x) ∧
  (∀ k : ℤ, ∀ x : ℝ, (x ∈ Set.Icc (k * π + π / 12) (k * π + 7 * π / 12)) → MonotoneOn f (Set.Icc (k * π + π / 12) (k * π + 7 * π / 12))) :=
sorry

theorem problem_part2 (A : ℝ) (a b c : ℝ) (area : ℝ) :
  f (A / 2 - π / 6) = Real.sqrt 3 ∧ 
  c = 2 ∧ 
  area = 2 * Real.sqrt 3 →
  a = 2 * Real.sqrt 3 ∨ a = 2 * Real.sqrt 7 :=
sorry

end problem_part1_problem_part2_l158_158562


namespace triangle_distance_EF_l158_158334

-- Definitions based on the given conditions in a)
def right_triangle (A B C : Type) [euclidean_geometry A B C] : Prop :=
∃ (x y : ℝ), x^2 + y^2 = (10 * real.sqrt 21)^2

def point_on_AB (A B E : Type) [euclidean_geometry A B E] : Prop :=
∃ (EA EB : ℝ), EA = 10 * real.sqrt 7 ∧ EB = 20 * real.sqrt 7

def foot_of_altitude (C F : Type) [euclidean_geometry C F] : Prop :=
-- definition of foot of altitude goes here

def distance_EF (EF : ℝ) : Prop :=
EF = (10 * (real.sqrt 21 - 3 * real.sqrt 7)) / 3

def m_n_sum (m n : ℕ) : Prop :=
m + n = 31 ∧ m * real.sqrt n = (10 * (real.sqrt 21 - 3 * real.sqrt 7)) / 3

-- Main theorem based on the above conditions
theorem triangle_distance_EF (A B C E F : Type) [euclidean_geometry A B C] [euclidean_geometry A B E] [euclidean_geometry C F] :
  right_triangle A B C →
  point_on_AB A B E →
  foot_of_altitude C F →
  ∃ (m n : ℕ), distance_EF (10 * (real.sqrt 21 - 3 * real.sqrt 7) / 3) ∧ m_n_sum m n :=
by
  intros hABC hABE hCF
  use [10, 21]
  split
  · exact distance_EF (10 * (real.sqrt 21 - 3 * real.sqrt 7) / 3)
  · exact m_n_sum 10 21
  sorry

end triangle_distance_EF_l158_158334


namespace paint_cans_needed_l158_158514

-- Conditions as definitions
def bedrooms : ℕ := 3
def other_rooms : ℕ := 2 * bedrooms
def paint_per_room : ℕ := 2
def color_can_capacity : ℕ := 1
def white_can_capacity : ℕ := 3

-- Total gallons needed
def total_color_gallons_needed : ℕ := paint_per_room * bedrooms
def total_white_gallons_needed : ℕ := paint_per_room * other_rooms

-- Total cans needed
def total_color_cans_needed : ℕ := total_color_gallons_needed / color_can_capacity
def total_white_cans_needed : ℕ := total_white_gallons_needed / white_can_capacity
def total_cans_needed : ℕ := total_color_cans_needed + total_white_cans_needed

theorem paint_cans_needed : total_cans_needed = 10 := by
  -- Proof steps (skipped) to show total_cans_needed = 10
  sorry

end paint_cans_needed_l158_158514


namespace solve_series_eq_l158_158368

noncomputable theory
open Real

/-- The main theorem to prove -/
theorem solve_series_eq (x : ℝ) (h : |x| < 0.5) :
  (1 + 2 * x + 4 * x^2 + ∑' (n : ℕ), (2 * x)^(n+2)) = 3.4 - 1.2 * x → x = 1 / 3 := 
sorry

end solve_series_eq_l158_158368


namespace min_value_g_squared_plus_f_l158_158217

def f (a b : ℝ) (x : ℝ) : ℝ := a * x + b
def g (a c : ℝ) (x : ℝ) : ℝ := a * x + c

theorem min_value_g_squared_plus_f (a b c : ℝ) (h : a ≠ 0) 
  (min_f_squared_plus_g : ∀ x : ℝ, (f a b x)^2 + g a c x ≥ 4)
  (exists_x_min : ∃ x : ℝ, (f a b x)^2 + g a c x = 4) :
  ∃ x : ℝ, (g a c x)^2 + f a b x = -9 / 2 :=
sorry

end min_value_g_squared_plus_f_l158_158217


namespace arccos_one_over_sqrt_two_l158_158982

theorem arccos_one_over_sqrt_two :
  Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_l158_158982


namespace smallest_number_divisible_l158_158432

/-- The smallest number which, when diminished by 20, is divisible by 15, 30, 45, and 60 --/
theorem smallest_number_divisible (n : ℕ) (h : ∀ k : ℕ, n - 20 = k * Int.lcm 15 (Int.lcm 30 (Int.lcm 45 60))) : n = 200 :=
sorry

end smallest_number_divisible_l158_158432


namespace number_of_students_more_than_pets_l158_158536

theorem number_of_students_more_than_pets 
  (students_per_classroom pets_per_classroom num_classrooms : ℕ)
  (h1 : students_per_classroom = 20)
  (h2 : pets_per_classroom = 3)
  (h3 : num_classrooms = 5) :
  (students_per_classroom * num_classrooms) - (pets_per_classroom * num_classrooms) = 85 := 
by
  sorry

end number_of_students_more_than_pets_l158_158536


namespace calculate_value_of_A6_3_minus_C6_3_l158_158878

def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

def permutations (n r : ℕ) : ℕ :=
  factorial n / factorial (n - r)

def combinations (n r : ℕ) : ℕ :=
  factorial n / (factorial r * factorial (n - r))

theorem calculate_value_of_A6_3_minus_C6_3 : permutations 6 3 - combinations 6 3 = 100 := by
  sorry

end calculate_value_of_A6_3_minus_C6_3_l158_158878


namespace probability_of_even_sum_is_four_fifths_l158_158262

-- Define the first ten prime numbers
def first_ten_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- The function to calculate the probability that the sum of two distinct primes is even
def probability_even_sum (primes : List ℕ) : ℚ :=
  let pairs := primes.product primes
  let distinct_pairs := pairs.filter (λ p, p.1 ≠ p.2)
  let even_sum_pairs := distinct_pairs.filter (λ p, (p.1 + p.2) % 2 = 0)
  (even_sum_pairs.length : ℚ) / (distinct_pairs.length : ℚ)

-- Prove that the probability is 4/5
theorem probability_of_even_sum_is_four_fifths :
  probability_even_sum first_ten_primes = 4 / 5 := sorry

end probability_of_even_sum_is_four_fifths_l158_158262


namespace arccos_proof_l158_158961

noncomputable def arccos_identity : Prop := 
  ∃ x : ℝ, x = 1 / Real.sqrt 2 ∧ Real.arccos x = Real.pi / 4

theorem arccos_proof : arccos_identity :=
by
  sorry

end arccos_proof_l158_158961


namespace leg_expression_l158_158180

-- Given definitions and conditions
variables {a b c : ℝ}
hypothesis (h1 : a^2 + b^2 = c^2) 
hypothesis (h2 : b^2 = a * c)

-- The theorem we need to prove
theorem leg_expression (h1 : a^2 + b^2 = c^2) (h2 : b^2 = a * c) :
  a = (c * (Real.sqrt 5 - 1)) / 2 :=
sorry

end leg_expression_l158_158180


namespace complement_of_domain_l158_158693

variable {x : ℝ}

def f (x : ℝ) : ℝ := ln ((1 + x) / (1 - x))

theorem complement_of_domain :
  {x : ℝ | ¬ (-1 < x ∧ x < 1)} = {x : ℝ | x ≤ -1 ∨ x ≥ 1} :=
by
  sorry

end complement_of_domain_l158_158693


namespace probability_sum_even_l158_158276

-- Let's define the set of the first ten prime numbers.
def first_ten_primes : Finset ℕ := Finset.of_list [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define the proposition to prove:
theorem probability_sum_even :
  let n := first_ten_primes.card in 
  let total_pairs := Finset.card (Finset.powersetLen 2 first_ten_primes) in 
  let even_pairs := Finset.card (Finset.filter (λ s, s.sum % 2 = 0) (Finset.powersetLen 2 first_ten_primes)) in 
  (total_pairs = 45 ∧ even_pairs = 36) →
  (even_pairs / total_pairs = 4 / 5) :=
by
  sorry

end probability_sum_even_l158_158276


namespace max_area_with_22_matches_l158_158782

-- Definitions based on the conditions
def perimeter := 22

def is_valid_length_width (l w : ℕ) : Prop := l + w = 11

def area (l w : ℕ) : ℕ := l * w

-- Statement of the proof problem
theorem max_area_with_22_matches : 
  ∃ (l w : ℕ), is_valid_length_width l w ∧ (∀ l' w', is_valid_length_width l' w' → area l w ≥ area l' w') ∧ area l w = 30 :=
  sorry

end max_area_with_22_matches_l158_158782


namespace cats_sold_correct_l158_158844

def initial_siames_cats : ℕ := 13
def initial_house_cats : ℕ := 5
def cats_left_after_sale : ℕ := 8

def total_initial_cats : ℕ := initial_siames_cats + initial_house_cats
def cats_sold : ℕ := total_initial_cats - cats_left_after_sale

theorem cats_sold_correct : cats_sold = 10 := by
  have h1 : total_initial_cats = 18 := by
    unfold total_initial_cats
    norm_num
  have h2 : cats_sold = 10 := by
    unfold cats_sold
    rw [h1]
    norm_num
  exact h2

end cats_sold_correct_l158_158844


namespace sergeant_proof_l158_158835

theorem sergeant_proof (recruits : List ℕ) (turn : ℕ → ℕ) :
  ∃ pos : ℕ, (∑ i in List.range pos, (if turn recruits[i] = 1 then 1 else 0)) 
            = (∑ i in List.range(pos+1, recruits.length), (if turn recruits[i] = 2 then 1 else 0)) :=
  sorry

end sergeant_proof_l158_158835


namespace arccos_sqrt_half_l158_158966

theorem arccos_sqrt_half : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 := by
  sorry

end arccos_sqrt_half_l158_158966


namespace arch_height_at_10_l158_158089

def parabolic_arch_height (x : ℝ) : ℝ :=
  let a := -4 / 125
  let k := 20
  a * x^2 + k

theorem arch_height_at_10
  (height : ℝ)
  (span : ℝ)
  (x₀ : ℝ)
  (expected_height : ℝ) :
  height = 20 →
  span = 50 →
  x₀ = 10 →
  expected_height = 16.8 →
  parabolic_arch_height x₀ = expected_height :=
by
  intros
  sorry

end arch_height_at_10_l158_158089


namespace ratio_of_surface_areas_l158_158755

theorem ratio_of_surface_areas (R r m : ℝ) (h1 : R > r)
  (h2 : m = sqrt (2 * R * r - r^2))
  (vol_cone : ℝ := (π * m / 3) * (R^2 + R * r + r^2))
  (vol_hemi : ℝ := 2 * π * m^3 / 3)
  (h_vol : 6 / 7 * vol_cone = vol_hemi) :
  let l := sqrt (R^2 + m^2) in
  let lateral_surface := π * (R + r) * l in
  let hemisphere_surface := 2 * π * m^2 in
  lateral_surface / hemisphere_surface = 2 * (20 / 21) :=
by sorry

end ratio_of_surface_areas_l158_158755


namespace solve_quadratic_eq1_solve_quadratic_eq2_l158_158739

-- Define the statement for the first problem
theorem solve_quadratic_eq1 (x : ℝ) : x^2 - 49 = 0 → x = 7 ∨ x = -7 :=
by
  sorry

-- Define the statement for the second problem
theorem solve_quadratic_eq2 (x : ℝ) : 2 * (x + 1)^2 - 49 = 1 → x = 4 ∨ x = -6 :=
by
  sorry

end solve_quadratic_eq1_solve_quadratic_eq2_l158_158739


namespace perpendicular_tangents_l158_158020

-- Defining the points and lines
variables {A B C D P K : Type}
variables (circle : Circle A B) (tangent_PC : TangentLine circle P C) (tangent_PD : TangentLine circle P D)
          (line_AC : Line A C) (line_BD : Line B D) (K : IntersectionPoint line_AC line_BD)

-- Stating the theorem 
theorem perpendicular_tangents (h : IsTangent tangent_PC circle C) (h' : IsTangent tangent_PD circle D) :
  IsPerpendicular (Line P K) (Line A B) :=
by
  sorry

end perpendicular_tangents_l158_158020


namespace first_player_wins_l158_158417

-- Define the game state and rules
structure GameGrid (n m : ℕ) :=
  (grid : Fin n × Fin m → Bool)

def initialGameState (n m : ℕ) : GameGrid n m :=
  { grid := λ _, false }

def validMove (g : GameGrid n m) (x : Fin n) (y : Fin m) : Prop :=
  ∀ i j, i ≥ x → j ≥ y → g.grid (i, j) = false

inductive turn : Type
| First : turn
| Second : turn

-- Main theorem that needs to be proved
theorem first_player_wins (n m : ℕ) (h1 : 1 < n ∨ 1 < m) :
  ∃ strategy : GameGrid n m → (Fin n × Fin m), 
    ∀ g, validMove g (strategy g).1 (strategy g).2 ∧ (
      ∃ g', g' = ⟨λ p, p ≥ (strategy g) ∨ g.grid p⟩ → 
        ∃ winner : turn, winner = turn.First
    ) :=
sorry

end first_player_wins_l158_158417


namespace sphere_surface_area_l158_158832

theorem sphere_surface_area (edge_length : ℝ) (diameter_eq_edge_length : (diameter : ℝ) = edge_length) :
  (edge_length = 2) → (diameter = 2) → (surface_area : ℝ) = 8 * Real.pi :=
by
  sorry

end sphere_surface_area_l158_158832


namespace count_squares_ending_in_4_l158_158609

theorem count_squares_ending_in_4 (n : ℕ) : 
  (∀ k : ℕ, (n^2 < 5000) → (n^2 % 10 = 4) → (k ≤ 70)) → 
  (∃ m : ℕ, m = 14) :=
by 
  sorry

end count_squares_ending_in_4_l158_158609


namespace arccos_identity_l158_158913

theorem arccos_identity : 
  ∀ x : ℝ, x = 1 / real.sqrt 2 → real.arccos x = real.pi / 4 :=
begin
  intros x hx,
  have h1 : real.cos (real.pi / 4) = 1 / real.sqrt 2 := by sorry,
  have h2 : real.arccos (1 / real.sqrt 2) = real.pi / 4,
  { rw ← h1,
    rw real.arccos_cos,
    exact_mod_cast hx },
  exact h2,
end

end arccos_identity_l158_158913


namespace quadratic_function_properties_l158_158556

def quadratic_function (x : ℝ) : ℝ :=
  - (3 / 4) * x ^ 2 + (9 / 2) * x - (15 / 4)

theorem quadratic_function_properties :
  (quadratic_function 1 = 0) ∧ (quadratic_function 5 = 0) ∧ (quadratic_function 3 = 3) :=
by
  -- Use the 'sorry' keyword to skip the proof steps
  sorry

end quadratic_function_properties_l158_158556


namespace perfect_square_selection_l158_158406

theorem perfect_square_selection :
  let k_range := {k | 1 ≤ k ∧ k ≤ 153}
  let cards := (k_range.map (λ k => 3^k)).union (k_range.map (λ k => 19^k))
  (cards.card = 306) →
  let even_count := 76
  let odd_count := 77
  (Nat.choose even_count 2 + Nat.choose odd_count 2) * 2 + even_count * even_count =
  17328 :=
by 
  intro k_range cards h
  unfold even_count
  unfold odd_count
  sorry

end perfect_square_selection_l158_158406


namespace largest_integer_satisfying_inequality_l158_158991

theorem largest_integer_satisfying_inequality :
  ∃ n : ℤ, n = 4 ∧ (1 / 4 + n / 8 < 7 / 8) ∧ ∀ m : ℤ, m > 4 → ¬(1 / 4 + m / 8 < 7 / 8) :=
by
  sorry

end largest_integer_satisfying_inequality_l158_158991


namespace distance_C_A_l158_158100

noncomputable def distance_from_A_to_C : ℂ := abs (1170 + 1560 * complex.I)

theorem distance_C_A :
  distance_from_A_to_C = 1950 :=
by
  -- Proof is omitted
  sorry

end distance_C_A_l158_158100


namespace extremum_point_range_l158_158205

noncomputable def f (x k : ℝ) : ℝ := (Real.exp x) / x + k * (Real.log x - x)

theorem extremum_point_range {k : ℝ} 
  (hf : ∀ x > 0, f x k = (Real.exp x) / x + k * (Real.log x - x))
  (h_extr : ∀ x > 0, deriv (λ x, (Real.exp x) / x + k * (Real.log x - x)) x = 0 → x = 1) :
  k ≤ Real.exp 1 := sorry

end extremum_point_range_l158_158205


namespace num_x_intercepts_l158_158226

theorem num_x_intercepts (f : ℝ → ℝ) : 
  f = (λ x, (x - 5) * (x + 3)^2) → (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (f x1 = 0 ∧ f x2 = 0) ∧ ¬∃ x3 : ℝ, f x3 = 0 ∧ x3 ≠ x1 ∧ x3 ≠ x2) := 
by
  sorry

end num_x_intercepts_l158_158226


namespace find_angle_between_unit_vectors_l158_158342

variables (a b : ℝ^3) -- Assuming vectors in 3D space for simplicity

-- Defining unit vectors
def unit_vector (v : ℝ^3) : Prop := ∥v∥ = 1

-- Orthogonality condition
def orthogonal (u v : ℝ^3) : Prop := dot_product u v = 0

-- Main proof problem
theorem find_angle_between_unit_vectors
  (ha : unit_vector a)
  (hb : unit_vector b)
  (h_ortho : orthogonal (a + 3 • b) (7 • a - 6 • b)) :
  real.angle a b = real.arccos (11 / 21) :=
sorry

end find_angle_between_unit_vectors_l158_158342


namespace twelve_gon_from_midpoints_l158_158645

open Complex

def square_vertices : ℂ × ℂ × ℂ × ℂ :=
(1, Complex.I, -1, -Complex.I)

def is_equilateral_triangle (a b c : ℂ) : Prop :=
∃ (r : ℂ), r * (b - a) = c - a ∧ abs (b - a) = abs (c - a)

def square_with_equilateral_triangles (a b c d k l m n : ℂ) : Prop :=
square_vertices = (a, b, c, d) ∧
is_equilateral_triangle a b k ∧
is_equilateral_triangle b c l ∧
is_equilateral_triangle c d m ∧
is_equilateral_triangle d a n

def regular_dodecagon_midpoints (a b c d k l m n : ℂ) (midpoints : list ℂ) : Prop :=
midpoints = [
  (k + l) / 2, (l + m) / 2, (m + n) / 2, (n + k) / 2,
  (a + k) / 2, (b + k) / 2, (b + l) / 2, (c + l) / 2,
  (c + m) / 2, (d + m) / 2, (d + n) / 2, (a + n) / 2
]

theorem twelve_gon_from_midpoints (a b c d k l m n : ℂ)
(h : square_with_equilateral_triangles a b c d k l m n) :
∃ (midpoints : list ℂ), regular_dodecagon_midpoints a b c d k l m n midpoints := 
sorry

end twelve_gon_from_midpoints_l158_158645


namespace arccos_proof_l158_158956

noncomputable def arccos_identity : Prop := 
  ∃ x : ℝ, x = 1 / Real.sqrt 2 ∧ Real.arccos x = Real.pi / 4

theorem arccos_proof : arccos_identity :=
by
  sorry

end arccos_proof_l158_158956


namespace tangent_line_parallel_to_4x_minus_1_l158_158527

theorem tangent_line_parallel_to_4x_minus_1 :
  ∃ (a b : ℝ), (a = 1 ∧ b = 0 ∨ a = -1 ∧ b = -4) ∧
  (∀ x, x^3 + x - 2 = (fderiv ℝ (λ x : ℝ, x^3 + x - 2) x).slope 4 ↔
    (∀ (x y : ℝ), y = 4 * (x + a) + b ∨ y = 4 * x → y = 4x - 4 ∨ y = 4x) ) := sorry

end tangent_line_parallel_to_4x_minus_1_l158_158527


namespace planting_schedule_time_l158_158350

variables (x : ℝ) (n : ℝ)
def original_plan : n := 20000
def efficiency_increase : ℝ := 1.25
def time_difference : ℝ := 5

theorem planting_schedule_time:
  (n / x - n / (x * efficiency_increase) = time_difference) :=
sorry

end planting_schedule_time_l158_158350


namespace difference_between_largest_and_second_largest_l158_158773

-- Define the set of numbers
def numbers : Finset ℕ := {9, 2, 4, 1, 5, 6}

-- Largest and second largest elements of the set
def max_element : ℕ := numbers.max' (by decide) -- maximum element in the set
def second_max_element : ℕ := numbers.erase max_element).max' (by sorry) -- second maximum element in the set

-- Formal statement to prove the required difference
theorem difference_between_largest_and_second_largest : max_element - second_max_element = 3 := by
  sorry

end difference_between_largest_and_second_largest_l158_158773


namespace smallest_possible_result_l158_158661

def multi_level_fraction (D P O B b : ℕ) : ℚ :=
  D + 1 / (P + 1 / (O + 1 / (B + 1 / b)))

theorem smallest_possible_result : ∃ (D P O B b : ℕ), 
  (D ≠ 0 ∧ P ≠ 0 ∧ O ≠ 0 ∧ B ≠ 0 ∧ b ≠ 0) ∧
  (D ≠ P ∧ D ≠ O ∧ D ≠ B ∧ D ≠ b ∧
   P ≠ O ∧ P ≠ B ∧ P ≠ b ∧ 
   O ≠ B ∧ O ≠ b ∧ 
   B ≠ b) ∧
  multi_level_fraction D P O B b ≈ (1 + 53 / 502) :=
by 
  sorry

end smallest_possible_result_l158_158661


namespace sample_center_on_regression_line_l158_158769

theorem sample_center_on_regression_line 
  (n : ℕ) 
  (x y : Fin n → ℝ) 
  (b a : ℝ) 
  (h : ∀ i, y i = b * x i + a) : 
  let x̄ := (∑ i, x i) / n 
  let ȳ := (∑ i, y i) / n 
  in ȳ = b * x̄ + a := 
by
  sorry

end sample_center_on_regression_line_l158_158769


namespace factorization_difference_l158_158163

theorem factorization_difference :
  ∃ (a b : ℕ), prime a ∧ prime b ∧ b > a ∧ 456456 = 8 * a * 7 * 11 * 13 * b ∧ b - a = 16 :=
by
  sorry

end factorization_difference_l158_158163


namespace true_propositions_l158_158575

-- Conditions as given in the problem
axiom condition_1 (α β γ : Plane) (h1 : α ≠ β) (h2 : α ≠ γ) :
  (α ∥ β) ∧ (α ∥ γ) → (β ∥ γ)

axiom condition_2 (α β : Plane) (a : Line) (h1 : α ≠ β) :
  (a ∥ α) ∧ (a ∥ β) → (α ∥ β) ∨ (α ∩ β ≠ ∅)

axiom condition_3 (α β γ : Plane) (h1 : α ≠ β) :
  (α ⟂ γ) ∧ (β ⟂ γ) → (α ∥ β) ∨ (α ∩ β ≠ ∅)

axiom condition_4 (α β : Plane) (a : Line) (h1 : α ≠ β) :
  (a ⟂ α) ∧ (a ⟂ β) → (α ∥ β)

-- Prove that propositions 1 and 4 are true, and propositions 2 and 3 are false
theorem true_propositions :
  (condition_1 α β γ h1αβ h1αγ) ∧
  ¬(condition_2 α β a h2) ∧
  ¬(condition_3 α β γ h3αβ) ∧
  (condition_4 α β a h4αβ) :=
sorry

end true_propositions_l158_158575


namespace bill_needs_paint_cans_l158_158513

theorem bill_needs_paint_cans :
  let bedrooms := 3
  let other_rooms := 2 * bedrooms
  let gallons_per_room := 2
  let color_paint_cans := 6 -- (bedrooms * gallons_per_room) / 1-gallon per can
  let white_paint_cans := 4 -- (other_rooms * gallons_per_room) / 3-gallons per can
  (color_paint_cans + white_paint_cans) = 10 := sorry

end bill_needs_paint_cans_l158_158513


namespace probability_even_sum_l158_158273

-- Definition of the prime numbers set and the selection scenario
def firstTenPrimes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Condition stating that we select two distinct numbers at random from the first ten primes
def randomSelection (s : List ℕ) := {x : ℕ × ℕ // x.1 ≠ x.2 ∧ x.1 ∈ s ∧ x.2 ∈ s}

-- Define the event that their sum is even
def evenSum (p : ℕ × ℕ) := (p.1 + p.2) % 2 = 0

-- Define the probability calculation
def probabilityEvenSum := 
  let totalPairs := (firstTenPrimes.length.choose 2) -- Calculate the number of ways to choose 2 numbers from the 10 primes
  let evenPairs := (randomSelection firstTenPrimes).count (λ p => evenSum p.val)
  evenPairs.toFloat / totalPairs.toFloat

-- Theorem statement that the probability of their sum being even is 1/5
theorem probability_even_sum : probabilityEvenSum = 1/5 := by
  sorry -- The actual proof is to be provided

end probability_even_sum_l158_158273


namespace arccos_proof_l158_158954

noncomputable def arccos_identity : Prop := 
  ∃ x : ℝ, x = 1 / Real.sqrt 2 ∧ Real.arccos x = Real.pi / 4

theorem arccos_proof : arccos_identity :=
by
  sorry

end arccos_proof_l158_158954


namespace arnolds_total_protein_l158_158869

theorem arnolds_total_protein (collagen_protein_per_two_scoops : ℕ) (protein_per_scoop : ℕ) 
    (steak_protein : ℕ) (scoops_of_collagen : ℕ) (scoops_of_protein : ℕ) :
    collagen_protein_per_two_scoops = 18 →
    protein_per_scoop = 21 →
    steak_protein = 56 →
    scoops_of_collagen = 1 →
    scoops_of_protein = 1 →
    (collagen_protein_per_two_scoops / 2 * scoops_of_collagen + protein_per_scoop * scoops_of_protein + steak_protein = 86) :=
by
  intros hc p s sc sp
  sorry

end arnolds_total_protein_l158_158869


namespace rate_per_meter_for_fencing_l158_158759

theorem rate_per_meter_for_fencing
  (w : ℕ) (length : ℕ) (perimeter : ℕ) (cost : ℕ)
  (h1 : length = w + 10)
  (h2 : perimeter = 2 * (length + w))
  (h3 : perimeter = 340)
  (h4 : cost = 2210) : (cost / perimeter : ℝ) = 6.5 := by
  sorry

end rate_per_meter_for_fencing_l158_158759


namespace conditional_probability_l158_158499

noncomputable def P (event : Type) : Prop := sorry

variables (A B : Type)
axiom strong_winds_prob : P A = 0.4
axiom rain_prob : P B = 0.5
axiom strong_winds_and_rain_prob : P (A ∩ B) = 0.3

theorem conditional_probability :
  P (B | A) = 3 / 4 :=
by sorry

end conditional_probability_l158_158499


namespace ratio_of_cylinder_volumes_l158_158073

open Real

noncomputable def volume_cylinder (r h : ℝ) : ℝ :=
  π * r^2 * h

theorem ratio_of_cylinder_volumes (h₁ : ℝ) (h₂ : ℝ) (c₁ : ℝ) (c₂ : ℝ) :
  h₁ = 9 → h₂ = 7 → c₁ = 7 → c₂ = 9 →
  let r₁ := c₁ / (2 * π) in
  let r₂ := c₂ / (2 * π) in
  let V₁ := volume_cylinder r₁ h₁ in
  let V₂ := volume_cylinder r₂ h₂ in
  (V₂ / V₁) = (1 / 7) :=
by
  intros
  let r₁ := c₁ / (2 * π)
  have := h₁ = 9
  have := h₂ = 7
  let r₂ := c₂ / (2 * π)
  have := c₁ = 7
  have := c₂ = 9
  let V₁ := volume_cylinder r₁ h₁
  let V₂ := volume_cylinder r₂ h₂
  sorry

end ratio_of_cylinder_volumes_l158_158073


namespace part_a_part_b_l158_158707

noncomputable def problem_conditions : Prop :=
  let m := 3
  let n := 57
  ∃ (table : ℕ × ℕ → ℕ), 
  (∀ i j, 1 ≤ table (i, j) ∧ table (i, j) ≤ m * n) ∧
  list.sort (finset.univ.image (λ (ij : ℕ × ℕ), table ij)).val = list.range (m * n + 1) ∧
  table (m, 1) = 1 ∧
  (∀ a, a ≥ 1 ∧ a < m * n → (∀ (i j i' j'), table (i, j) = a ∧ table (i', j') = a + 1 →
     abs (i' - i) + abs (j' - j) = 1))

theorem part_a : problem_conditions → 
  (∃ (i j : ℕ), abs ((3, 1).1 - i) + abs ((3, 1).2 - j) = 1 ∧ 
    (∀ m n, abs ((3, 1).1 - m) + abs ((3, 1).2 - n) = 1 → table (m, n) = 2) ∧
    table (i, j) = 170) := sorry

theorem part_b : problem_conditions → 
  (∃ (good_cells : finset (ℕ × ℕ)),
    good_cells.card = 85 ∧
    (∀ (i j), (i, j) ∈ good_cells → 
      ∃ (i' j'), abs (i - i') + abs (j - j') = 1 ∧ 
      table (i', j') = 170 ∧ table (i, j) = 171)) := sorry

end part_a_part_b_l158_158707


namespace greatest_possible_value_of_q_minus_r_l158_158763

theorem greatest_possible_value_of_q_minus_r :
  ∃ q r : ℕ, 0 < q ∧ 0 < r ∧ 852 = 21 * q + r ∧ q - r = 28 :=
by
  -- Proof goes here
  sorry

end greatest_possible_value_of_q_minus_r_l158_158763


namespace ashok_borrowed_l158_158458

theorem ashok_borrowed (P : ℝ) (h : 11400 = P * (6 / 100 * 2 + 9 / 100 * 3 + 14 / 100 * 4)) : P = 12000 :=
by
  sorry

end ashok_borrowed_l158_158458


namespace sum_100_necessary_9_removed_sum_100_necessary_8_removed_l158_158555

theorem sum_100_necessary_9_removed : ∃ S ⊆ {1..100}, S.card = 9 ∧ (∀ T ⊆ S, T.card = 9 → T.sum = 100) := sorry

theorem sum_100_necessary_8_removed : ∀ S ⊆ {1..100}, S.card = 8 → (∃ T ⊆ {1..100} \ S, T.card = 4 ∧ T.sum = 100) := sorry

end sum_100_necessary_9_removed_sum_100_necessary_8_removed_l158_158555


namespace math_proof_problem_general_l158_158588

open Real

variables (a b x y m : ℝ)
variables (A B F M N P : Point)

def ellipse_C1 := (a > b) ∧ (b > 0) ∧ (a = sqrt 2) ∧ (b = 1) ∧ (eccentricity = (sqrt 2) / 2) ∧ 
  (x^2 / a^2 + y^2 / b^2 = 1)

def parabola_C2 := (line_eq := y = x + b) ∧ (tangent_to_parabola := y^2 = 4*x)

noncomputable def equation_ellipse (a b : ℝ) : Prop := (a = sqrt 2) ∧ (b = 1) ∧ (x^2 / 2 + y^2 = 1)

noncomputable def angle_equality (A F M B N : Point) : Prop :=
  ∃ (P : Point), line (P, M) = ∅ ∧ line (P, N) = ∅ ∧ angle A F M = angle B F N

noncomputable def max_area_triangle (M F N : Point) : ℝ :=
  let S := (1 / 2) * |F P| * abs (y1 - y2) in
  sqrt 2 / 4

theorem math_proof_problem_general (a b : ℝ) (A B F M N P : Point) :
  (ellipse_C1 a b x y) ∧ (parabola_C2 y x b) →
  (equation_ellipse a b) ∧ (angle_equality A F M B N) ∧ (max_area_triangle M F N = sqrt 2 / 4) :=
begin
  sorry
end

end math_proof_problem_general_l158_158588


namespace earl_rate_l158_158537

theorem earl_rate :
  ∃ E : ℝ, 
    let L := (2 / 3) * E in
    L = (2 / 3) * E ∧ E + L = 60 ∧ E = 36 :=
by
  sorry

end earl_rate_l158_158537


namespace problem_1_problem_2_problem_3_l158_158204

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x
noncomputable def g (x : ℝ) : ℝ := f (x + Real.pi / 4) + f (x + 3 * Real.pi / 4)

theorem problem_1 : f (Real.pi / 2) = 1 := 
sorry

theorem problem_2 : (∃ p > 0, ∀ x, f (x + p) = f x) ∧ (∀ p, p > 0 ∧ (∀ x, f (x + p) = f x) → p ≥ 2 * Real.pi) := 
sorry

theorem problem_3 : ∃ x : ℝ, g x = -2 := 
sorry

end problem_1_problem_2_problem_3_l158_158204


namespace arccos_sqrt_half_l158_158965

theorem arccos_sqrt_half : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 := by
  sorry

end arccos_sqrt_half_l158_158965


namespace yes_answers_last_monday_september_l158_158309

-- Definitions based on conditions
constant city_N : Type
constant Resident : city_N → Prop
constant blonde : city_N → Prop
constant brunette : city_N → Prop
constant lies : blonde → Prop
constant tells_truth : brunette → Prop
constant dyes_hair_daily : Resident → Prop
constant october_birth : Resident → Prop
constant autumn_birth : Resident → Prop
constant residents : city_N → Prop
constant asked_on_monday : Resident → Prop
constant answers_yes : Resident → Prop
constant last_monday_september : Resident → Prop

-- Conditions as axioms
axiom only_blondes_and_brunettes (r : city_N) : residents r → (blonde r ∨ brunette r)
axiom blondes_lie (b : city_N) : blonde b → lies b
axiom brunettes_tell_truth (b : city_N) : brunette b → tells_truth b
axiom hair_dye_daily (r : city_N) : dyes_hair_daily r
axiom monday_asked (r : city_N) : autumn_birth r → asked_on_monday r → answers_yes r → 200
axiom friday_asked (r : city_N) : autumn_birth r → ¬asked_on_monday r → (answers_yes r → 50)
axiom exact_four_mondays_october : 4 = 4
axiom no_november_birth (r : city_N) : residents r → ¬november_birth r

-- Theorem stating the core question with conditions
theorem yes_answers_last_monday_september : 
  ∀ r : city_N, residents r → last_monday_september r → ¬answer_yes r :=
by
  sorry

end yes_answers_last_monday_september_l158_158309


namespace ratio_NO_NQ_l158_158854

noncomputable def side_length_of_square_s : ℝ := sorry
noncomputable def width_of_rectangle_w : ℝ := sorry
noncomputable def height_of_rectangle_h : ℝ := sorry

-- Given conditions:
def square_area := side_length_of_square_s ^ 2
def rectangle_area := width_of_rectangle_w * height_of_rectangle_h
def area_overlap := 0.4 * square_area

-- Condition 1: Square shares 40% of its area with rectangle:
axiom overlap_square_rectangle : area_overlap = 0.25 * rectangle_area

-- Condition 2: Ratio of length to width of rectangle is 4:1
axiom ratio_of_rectangle : width_of_rectangle_w = 4 * height_of_rectangle_h

-- Proving the desired ratio NO/NQ
theorem ratio_NO_NQ (h_area_overlap : overlap_square_rectangle) (h_ratio : ratio_of_rectangle) :
  (width_of_rectangle_w / height_of_rectangle_h) = 4 :=
by
  -- proof will go here
  sorry

end ratio_NO_NQ_l158_158854


namespace distance_traveled_l158_158620

-- Definitions based on the given conditions
variables (D T : ℝ)
-- D = 4 * T
def cond1 : Prop := D = 4 * T
-- D + 6 = 5 * T
def cond2 : Prop := D + 6 = 5 * T

-- Prove the question equals the answer given conditions
theorem distance_traveled : cond1 D T → cond2 D T → D = 24 :=
by intros h1 h2
sry -- The proof is skipped

end distance_traveled_l158_158620


namespace two_hundredth_digit_of_5_over_13_l158_158420

theorem two_hundredth_digit_of_5_over_13 :
  (∀ n : ℕ, n ≥ 0 →
    let d := "384615".charAt ( (n % 6) - 1 ) in
    if n % 6 = 0 then '5' else d  = 8) :=
sorry

end two_hundredth_digit_of_5_over_13_l158_158420


namespace arccos_one_over_sqrt_two_eq_pi_over_four_l158_158899

theorem arccos_one_over_sqrt_two_eq_pi_over_four : 
  Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_eq_pi_over_four_l158_158899


namespace solve_polar_and_distance_l158_158663

-- Defining the polar and parametric equations
def polar_eq_c1 (ρ θ r : ℝ) : Prop :=
  ρ * (ρ - 4 * Real.cos θ) = r^2 - 4

def param_eq_c2 (r θ x y : ℝ) : Prop :=
  x = 4 + Real.sqrt 3 * r * Real.cos θ ∧
  y = Real.sqrt 3 * r * Real.sin θ

-- Defining the transformation from polar to Cartesian for curves
def cartesian_eq_c1 (x y r : ℝ) : Prop :=
  x^2 + y^2 - 4 * x + 4 = r^2

def polar_eq_c2 (ρ θ r : ℝ) : Prop :=
  ρ^2 - 8 * ρ * Real.cos θ + 16 = 3 * r^2

-- Defining the line and intersection
def line_l (t : ℝ) : ℝ × ℝ :=
  (1/2 * t, Real.sqrt 3 / 2 * t)

def curve_c3 (x y : ℝ) : Prop :=
  x^2 + y^2 - 2 * x = 2

def OA_OB (A B : ℝ) : ℝ :=
  |A - 1| - |B - 1|

-- Main theorem encapsulation
theorem solve_polar_and_distance (ρ θ r x y t A B : ℝ) :
  polar_eq_c1 ρ θ r →
  cartesian_eq_c1 x y r →
  param_eq_c2 r θ x y →
  polar_eq_c2 ρ θ r →
  curve_c3 (fst (line_l t)) (snd (line_l t)) →
  OA_OB (fst (line_l A)) (snd (line_l B)) = 1 :=
sorry

end solve_polar_and_distance_l158_158663


namespace find_p_l158_158985

open Real

noncomputable def ellipse_eq (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 = 1

noncomputable def focus_point : ℝ × ℝ :=
  (sqrt 3, 0)

noncomputable def point_P (p : ℝ) : ℝ × ℝ :=
  (p, 0)

theorem find_p :
  ∃ p > 0, ∀ (A B : ℝ × ℝ), ellipse_eq A.1 A.2 ∧ ellipse_eq B.1 B.2 ∧
  ((A.1 - sqrt 3) * (B.1 - sqrt 3) + A.2 * B.2 = 0) →
  ∠point_P p focus_point A = ∠point_P p focus_point B →
  p = 2 + sqrt 3 :=
sorry

end find_p_l158_158985


namespace area_of_walkways_l158_158748

-- Define the dimensions of the individual flower bed
def flower_bed_width : ℕ := 8
def flower_bed_height : ℕ := 3

-- Define the number of rows and columns of flower beds
def rows_of_beds : ℕ := 4
def cols_of_beds : ℕ := 3

-- Define the width of the walkways
def walkway_width : ℕ := 2

-- Calculate the total width and height of the garden including walkways
def total_width : ℕ := (cols_of_beds * flower_bed_width) + (cols_of_beds + 1) * walkway_width
def total_height : ℕ := (rows_of_beds * flower_bed_height) + (rows_of_beds + 1) * walkway_width

-- Calculate the area of the garden including walkways
def total_area : ℕ := total_width * total_height

-- Calculate the total area of all the flower beds
def total_beds_area : ℕ := (rows_of_beds * cols_of_beds) * (flower_bed_width * flower_bed_height)

-- Prove the area of walkways
theorem area_of_walkways : total_area - total_beds_area = 416 := by
  sorry

end area_of_walkways_l158_158748


namespace find_a_l158_158023

theorem find_a (a : ℝ) : 
  let A := (a, 1)
  let B := (9, 0)
  let C := (-3, 4)
  (∀ A B C: ℝ × ℝ, ((A = (a, 1)) ∧ (B = (9, 0)) ∧ (C = (-3, 4)) → 
  ((C.2 - B.2) / (C.1 - B.1) = (B.2 - A.2) / (B.1 - A.1))) 
  → a = 6 :=
by
  sorry

end find_a_l158_158023


namespace time_for_C_to_complete_work_l158_158472

variable (A B C : ℕ) (R : ℚ)

def work_completion_in_days (days : ℕ) (portion : ℚ) :=
  portion = 1 / days

theorem time_for_C_to_complete_work :
  work_completion_in_days A 8 →
  work_completion_in_days B 12 →
  work_completion_in_days (A + B + C) 4 →
  C = 24 :=
by
  sorry

end time_for_C_to_complete_work_l158_158472


namespace triangle_inequality_l158_158322

theorem triangle_inequality 
  (A B C : ℝ) -- angle measures
  (a b c : ℝ) -- side lengths
  (h1 : a = b * (Real.cos C) + c * (Real.cos B)) 
  (cos_half_C_pos : 0 < Real.cos (C/2)) 
  (cos_half_C_lt_one : Real.cos (C/2) < 1)
  (cos_half_B_pos : 0 < Real.cos (B/2)) 
  (cos_half_B_lt_one : Real.cos (B/2) < 1) :
  2 * b * Real.cos (C / 2) + 2 * c * Real.cos (B / 2) > a + b + c :=
by
  sorry

end triangle_inequality_l158_158322


namespace latest_start_time_to_finish_turkey_by_6pm_l158_158697

noncomputable def calculate_total_cooking_time (weights : List ℕ) : ℕ :=
  weights.take 2 |>.sum * 15 + 2 * 20 + weights.drop 2 |>.sum * 15

def start_time_to_finish_by (finish_time : ℕ) (total_minutes : ℕ) : ℕ :=
  let total_hours := total_minutes / 60
  let remaining_minutes := total_minutes % 60
  finish_time - total_hours * 100 - remaining_minutes

theorem latest_start_time_to_finish_turkey_by_6pm :
  let weights := [16, 18, 20, 22]
  let total_time := calculate_total_cooking_time weights
  start_time_to_finish_by 1800 total_time = 2220 := by
sorry

end latest_start_time_to_finish_turkey_by_6pm_l158_158697


namespace minimum_value_l158_158144

noncomputable def f (x : ℝ) : ℝ := x^2 / (x - 10)

theorem minimum_value (x : ℝ) (h : x > 10) : (∃ y : ℝ, (∀ x' : ℝ, x' > 10 → f x' ≥ y) ∧ y = 40) := 
sorry

end minimum_value_l158_158144


namespace height_of_right_triangle_on_parabola_is_one_l158_158811

theorem height_of_right_triangle_on_parabola_is_one
    (x0 x1 x2 : ℝ)
    (h0 : x0 ≠ x1)
    (h1 : x1 ≠ x2)
    (h2 : x0 ≠ x2) :
    let y0 := x0^2
        y1 := x1^2
        y2 := x2^2
    in  
    let C := (x0, y0)
        A := (x1, y1)
        B := (x2, y2)
    in
    A.2 = B.2 → y0 = 1 := 
by
  intros
  sorry

end height_of_right_triangle_on_parabola_is_one_l158_158811


namespace PRINT_3_3_2_l158_158768

def PRINT (a b : Nat) : Nat × Nat := (a, b)

theorem PRINT_3_3_2 :
  PRINT 3 (3 + 2) = (3, 5) :=
by
  sorry

end PRINT_3_3_2_l158_158768


namespace concrete_volume_l158_158833

noncomputable def volume_of_concrete (width_ft length_ft thickness_in : ℝ) : ℝ :=
  let width_yds := width_ft / 3
  let length_yds := length_ft / 3
  let thickness_yds := thickness_in / 36
  width_yds * length_yds * thickness_yds

theorem concrete_volume :
  volume_of_concrete 4 80 4 |> ceil = 4 := 
by
  sorry

end concrete_volume_l158_158833


namespace lyudochka_cannot_complete_task_with_11_lost_cards_l158_158408

theorem lyudochka_cannot_complete_task_with_11_lost_cards :
  (∀ (A B C D : ℕ), A ∈ (Set.range 89).map (+12) 
     ∧ B ∈ (Set.range 89).map (+12) 
     ∧ C ∈ (Set.range 89).map (+12) 
     ∧ D ∈ (Set.range 89).map (+12) 
     → A + B + C + D ≠ 50) :=
by {
  sorry
}

end lyudochka_cannot_complete_task_with_11_lost_cards_l158_158408


namespace identify_counterfeit_13_coins_identify_and_determine_weight_14_coins_impossible_with_14_coins_l158_158066

-- Proving the identification of the counterfeit coin among 13 coins in 3 weighings
theorem identify_counterfeit_13_coins (coins : Fin 13 → Real) (is_counterfeit : ∃! i, coins i ≠ coins 0) :
  ∃ measure_count : ℕ, measure_count <= 3 ∧ 
    ∃ i, (coins i ≠ coins 0) :=
sorry

-- Proving counterfeit coin weight determination with an additional genuine coin using 3 weighings
theorem identify_and_determine_weight_14_coins (coins : Fin 14 → Real) (genuine : Real) (is_counterfeit : ∃! i, coins i ≠ genuine) :
  ∃ method_exists : Prop, 
    (method_exists ∧ ∃ measure_count : ℕ, measure_count <= 3 ∧ 
    ∃ (i : Fin 14), coins i ≠ genuine) :=
sorry

-- Proving the impossibility of identifying counterfeit coin among 14 coins using 3 weighings
theorem impossible_with_14_coins (coins : Fin 14 → Real) (is_counterfeit : ∃! i, coins i ≠ coins 0) :
  ¬ (∃ measure_count : ℕ, measure_count <= 3 ∧ 
    ∃ i, (coins i ≠ coins 0)) :=
sorry

end identify_counterfeit_13_coins_identify_and_determine_weight_14_coins_impossible_with_14_coins_l158_158066


namespace arccos_identity_l158_158916

theorem arccos_identity : 
  ∀ x : ℝ, x = 1 / real.sqrt 2 → real.arccos x = real.pi / 4 :=
begin
  intros x hx,
  have h1 : real.cos (real.pi / 4) = 1 / real.sqrt 2 := by sorry,
  have h2 : real.arccos (1 / real.sqrt 2) = real.pi / 4,
  { rw ← h1,
    rw real.arccos_cos,
    exact_mod_cast hx },
  exact h2,
end

end arccos_identity_l158_158916


namespace dasha_ate_one_bowl_l158_158021

-- Define the quantities for Masha, Dasha, Glasha, and Natasha
variables (M D G N : ℕ)

-- Given conditions
def conditions : Prop :=
  (M + D + G + N = 16) ∧
  (G + N = 9) ∧
  (M > D) ∧
  (M > G) ∧
  (M > N)

-- The problem statement rewritten in Lean: Prove that given the conditions, Dasha ate 1 bowl.
theorem dasha_ate_one_bowl (h : conditions M D G N) : D = 1 :=
sorry

end dasha_ate_one_bowl_l158_158021


namespace toothpicks_total_l158_158778

-- Definitions based on the conditions
def grid_length : ℕ := 50
def grid_width : ℕ := 40

-- Mathematical statement to prove
theorem toothpicks_total : (grid_length + 1) * grid_width + (grid_width + 1) * grid_length = 4090 := by
  sorry

end toothpicks_total_l158_158778


namespace sum_of_g_eq_zero_l158_158679

def g (x : ℝ) : ℝ := x^2 * (1 - x)^2

theorem sum_of_g_eq_zero :
  (∑ k in finset.range 2020, (if k % 2 = 0 then g (k / 2021 + 1) else - g (k / 2021 + 1))) = 0 :=
sorry

end sum_of_g_eq_zero_l158_158679


namespace equation_of_line_through_point_with_equal_intercepts_l158_158140

-- Define a structure for a 2D point
structure Point :=
(x : ℝ)
(y : ℝ)

-- Define the problem-specific points and conditions
def A : Point := {x := 4, y := -1}

-- Define the conditions and the theorem to be proven
theorem equation_of_line_through_point_with_equal_intercepts
  (p : Point)
  (h : p = A) : 
  ∃ (a : ℝ), a ≠ 0 → (∀ (a : ℝ), ((∀ (b : ℝ), b = a → b ≠ 0 → x + y - a = 0)) ∨ (x + 4 * y = 0)) :=
sorry

end equation_of_line_through_point_with_equal_intercepts_l158_158140


namespace probability_of_even_sum_is_four_fifths_l158_158264

-- Define the first ten prime numbers
def first_ten_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- The function to calculate the probability that the sum of two distinct primes is even
def probability_even_sum (primes : List ℕ) : ℚ :=
  let pairs := primes.product primes
  let distinct_pairs := pairs.filter (λ p, p.1 ≠ p.2)
  let even_sum_pairs := distinct_pairs.filter (λ p, (p.1 + p.2) % 2 = 0)
  (even_sum_pairs.length : ℚ) / (distinct_pairs.length : ℚ)

-- Prove that the probability is 4/5
theorem probability_of_even_sum_is_four_fifths :
  probability_even_sum first_ten_primes = 4 / 5 := sorry

end probability_of_even_sum_is_four_fifths_l158_158264


namespace bus_speed_kmph_l158_158077

theorem bus_speed_kmph (distance : ℝ) (time : ℝ) (conversion_factor : ℝ)
    (h_distance : distance = 300.024)
    (h_time : time = 30)
    (h_conversion_factor : conversion_factor = 3.6) :
    (distance / time) * conversion_factor = 36.003 :=
by
  simp [h_distance, h_time, h_conversion_factor]
  norm_num

-- The statement asserts with given h_distance, h_time,
-- and h_conversion_factor hypotheses that the computed speed results in 36.003 kmph.

end bus_speed_kmph_l158_158077


namespace faces_of_prism_with_24_edges_l158_158090

theorem faces_of_prism_with_24_edges (L : ℕ) (h1 : 3 * L = 24) : L + 2 = 10 := by
  sorry

end faces_of_prism_with_24_edges_l158_158090


namespace largest_natural_number_n_l158_158992

theorem largest_natural_number_n (n : ℕ) :
  (4^995 + 4^1500 + 4^n).natAbs ∈ {x : ℕ | ∃ k : ℕ, x = k^2} ↔ n = 2004 := 
sorry

end largest_natural_number_n_l158_158992


namespace arccos_identity_l158_158920

theorem arccos_identity : 
  ∀ x : ℝ, x = 1 / real.sqrt 2 → real.arccos x = real.pi / 4 :=
begin
  intros x hx,
  have h1 : real.cos (real.pi / 4) = 1 / real.sqrt 2 := by sorry,
  have h2 : real.arccos (1 / real.sqrt 2) = real.pi / 4,
  { rw ← h1,
    rw real.arccos_cos,
    exact_mod_cast hx },
  exact h2,
end

end arccos_identity_l158_158920


namespace MilkConsumption_l158_158302

theorem MilkConsumption :
  let regular_milk := 2.0
  let soy_milk := 1.5
  let almond_milk := 1.0
  let oat_milk := 0.8
  let total_milk := regular_milk + soy_milk + almond_milk + oat_milk
  total_milk = 5.3 :=
by
  let regular_milk := 2.0
  let soy_milk := 1.5
  let almond_milk := 1.0
  let oat_milk := 0.8
  let total_milk := regular_milk + soy_milk + almond_milk + oat_milk
  show total_milk = 5.3, from sorry

end MilkConsumption_l158_158302


namespace periods_needed_l158_158831

theorem periods_needed 
  {num_students : ℕ} 
  {period_length : ℕ} 
  {individual_presentation_time individual_QA_time num_group_presentations group_presentation_time : ℕ}
  (h1 : num_students = 32)
  (h2 : period_length = 40)
  (h3 : individual_presentation_time = 5)
  (h4 : individual_QA_time = 3)
  (h5 : num_group_presentations = 4)
  (h6 : group_presentation_time = 12)
  : 
  (⌈(
    (num_students - num_group_presentations) * (individual_presentation_time + individual_QA_time) + 
    num_group_presentations * group_presentation_time
  ) / period_length⌉) = 7 := 
sorry

end periods_needed_l158_158831


namespace statement_b_statement_e_l158_158533

-- Statement (B): ∀ x, if x^3 > 0 then x > 0.
theorem statement_b (x : ℝ) : x^3 > 0 → x > 0 := sorry

-- Statement (E): ∀ x, if x < 1 then x^3 < x.
theorem statement_e (x : ℝ) : x < 1 → x^3 < x := sorry

end statement_b_statement_e_l158_158533


namespace profit_without_discount_l158_158852

theorem profit_without_discount (CP SP MP : ℝ) (discountRate profitRate : ℝ)
  (h1 : CP = 100)
  (h2 : discountRate = 0.05)
  (h3 : profitRate = 0.235)
  (h4 : SP = CP * (1 + profitRate))
  (h5 : MP = SP / (1 - discountRate)) :
  (((MP - CP) / CP) * 100) = 30 := 
sorry

end profit_without_discount_l158_158852


namespace correct_operation_l158_158445

theorem correct_operation (a : ℝ) : (a^3)^3 = a^9 := 
sorry

end correct_operation_l158_158445


namespace product_even_negatives_sign_and_last_digit_l158_158111

theorem product_even_negatives_sign_and_last_digit :
  let L := list.range' 1 2020 in
  let evens := L.filter (λ n : ℕ, n % 2 = 0) in
  let negatives := evens.map (λ n, -n) in
  let product := list.prod negatives in
  product < 0 ∧ product % 10 = 0 :=
by
  -- Definitions needed for the problem
  let L := list.range' 1 2020
  let evens := L.filter (λ n, n % 2 = 0)
  let negatives := evens.map (λ n, -n)
  let product := list.prod negatives

  -- Proof outline omitted, but includes the steps to show product < 0 and product % 10 = 0
  sorry

end product_even_negatives_sign_and_last_digit_l158_158111


namespace complex_conjugate_product_l158_158171

theorem complex_conjugate_product (z : ℂ) (h : z = 3 + 2 * I) : z * conj z = 13 := by 
  sorry

end complex_conjugate_product_l158_158171


namespace probability_of_even_sum_is_four_fifths_l158_158260

-- Define the first ten prime numbers
def first_ten_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- The function to calculate the probability that the sum of two distinct primes is even
def probability_even_sum (primes : List ℕ) : ℚ :=
  let pairs := primes.product primes
  let distinct_pairs := pairs.filter (λ p, p.1 ≠ p.2)
  let even_sum_pairs := distinct_pairs.filter (λ p, (p.1 + p.2) % 2 = 0)
  (even_sum_pairs.length : ℚ) / (distinct_pairs.length : ℚ)

-- Prove that the probability is 4/5
theorem probability_of_even_sum_is_four_fifths :
  probability_even_sum first_ten_primes = 4 / 5 := sorry

end probability_of_even_sum_is_four_fifths_l158_158260


namespace probability_even_sum_l158_158250

def firstTenPrimes : Set ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

def isOdd (n : ℕ) : Prop := n % 2 = 1

def distinctPairs (s : Set ℕ) : Set (ℕ × ℕ) :=
  { (x, y) | x ∈ s ∧ y ∈ s ∧ x ≠ y }

def eventEvenSum (s : Set ℕ) : Set (ℕ × ℕ) :=
  { (x, y) | x ∈ s ∧ y ∈ s ∧ x ≠ y ∧ (x + y) % 2 = 0 }

theorem probability_even_sum :
  (Set.card (eventEvenSum firstTenPrimes)).toRat / (Set.card (distinctPairs firstTenPrimes)).toRat = 4 / 5 :=
  sorry

end probability_even_sum_l158_158250


namespace car_highway_miles_per_tankful_l158_158079

-- Defining conditions as per given problem
def city_miles_per_tank : ℕ := 336
def city_miles_per_gallon : ℕ := 8
def difference_miles_per_gallon : ℕ := 3
def highway_miles_per_gallon := city_miles_per_gallon + difference_miles_per_gallon
def tank_size := city_miles_per_tank / city_miles_per_gallon
def highway_miles_per_tank := highway_miles_per_gallon * tank_size

-- Theorem statement to prove
theorem car_highway_miles_per_tankful :
  highway_miles_per_tank = 462 :=
sorry

end car_highway_miles_per_tankful_l158_158079


namespace orthic_triangle_similarity_l158_158304

theorem orthic_triangle_similarity (n : ℕ) (h₀ : 2 ∣ n → False) :
  let Kₙ := regular_polygon n in
  let vertices := vertices_of Kₙ in
  ∃ (H₀ Hₖ : triangle), H₀ ∈ triangles_of vertices ∧
                         (∀ k : ℕ, Hₖ = orthic_triangle (H (k - 1)) ∧ 
                                   H (k - 1) ∈ triangles_of vertices) ∧
                         (∃ k : ℕ, similar H₀ (H k)) :=
begin
  sorry
end

end orthic_triangle_similarity_l158_158304


namespace trigonometric_identity_l158_158612

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = -3) :
  (Real.sin θ - 2 * Real.cos θ) / (Real.sin θ + Real.cos θ) = 5 / 2 :=
by
  sorry

end trigonometric_identity_l158_158612


namespace f_f_10_eq_2_l158_158594

def f (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 + 1 else Real.log x

theorem f_f_10_eq_2 : f (f 10) = 2 :=
by {
  sorry
}

end f_f_10_eq_2_l158_158594


namespace rhombus_perimeter_l158_158754

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 18) (h2 : d2 = 32) : 
  let half1 := d1 / 2, 
      half2 := d2 / 2, 
      side := Real.sqrt (half1^2 + half2^2) in
  4 * side = 4 * Real.sqrt 337 :=
by
  sorry

end rhombus_perimeter_l158_158754


namespace problem_proof_l158_158092

-- Define the sequence using the given initial term and recursive relation.
def x : ℕ → ℤ
| 0 := 2
| (n+1) := x n^2 / 32^(n+1)

-- State the main theorem to prove the correctness of the computed result.
theorem problem_proof : (x 4) / 128 = 1 / 2048 :=
by sorry

end problem_proof_l158_158092


namespace age_of_older_friend_l158_158009

theorem age_of_older_friend (a b : ℕ) (h1 : a - b = 2) (h2 : a + b = 74) : a = 38 :=
by
  sorry

end age_of_older_friend_l158_158009


namespace three_letter_words_at_least_two_as_l158_158225

theorem three_letter_words_at_least_two_as :
  let letters := ['A', 'B', 'C', 'D', 'E']
  let words_with_repeats: Finset (List Char) := Finset.univ.image (fun x : Fin 5 × Fin 5 × Fin 5 => [letters[nth x.fst.1], letters[nth x.fst.2], letters[nth x.snd]])
  let valid_words := words_with_repeats.filter (λ w, w.count ('A' == ∘ (λ c => c == 'A')) >= 2)
  valid_words.card = 13 :=
by sorry

end three_letter_words_at_least_two_as_l158_158225


namespace solve_x_if_alpha_beta_eq_8_l158_158558

variable (x : ℝ)

def alpha (x : ℝ) := 4 * x + 9
def beta (x : ℝ) := 9 * x + 6

theorem solve_x_if_alpha_beta_eq_8 (hx : alpha (beta x) = 8) : x = (-25 / 36) :=
by
  sorry

end solve_x_if_alpha_beta_eq_8_l158_158558


namespace exists_zero_in_continuous_interval_l158_158240

variables {X : Type*} [TopologicalSpace X] [LinearOrder X] [TopologicalLinearOrder X]
variables {Y : Type*} [TopologicalSpace Y]
variable {f : X → Y}
variables {a b : X} {c : X}

theorem exists_zero_in_continuous_interval (h_cont : ContinuousOn f (Icc a b)) (h_pos : f a * f b > 0) :
  (∃ c ∈ Icc a b, f c = 0) :=
sorry

end exists_zero_in_continuous_interval_l158_158240


namespace triangulation_graph_one_stroke_iff_three_divides_n_l158_158082

theorem triangulation_graph_one_stroke_iff_three_divides_n
    (n : ℕ) 
    (H₀ : n ≥ 3)
    (H₁ : ∃ tri_gas : set (fin n), is_triangulation_graph tri_gas)
    :
    (∃ p : path tri_gas, is_one_stroke_path p) ↔ (3 ∣ n) := 
sorry

end triangulation_graph_one_stroke_iff_three_divides_n_l158_158082


namespace increasing_function_on_interval_l158_158861

noncomputable def f_A (x : ℝ) : ℝ := 3 - x
noncomputable def f_B (x : ℝ) : ℝ := x^2 - 3 * x
noncomputable def f_C (x : ℝ) : ℝ := - (1 / (x + 1))
noncomputable def f_D (x : ℝ) : ℝ := -|x|

theorem increasing_function_on_interval (h0 : ∀ x : ℝ, x > 0):
  (∀ x y : ℝ, 0 < x -> x < y -> f_C x < f_C y) ∧ 
  (∀ (g : ℝ → ℝ), (g ≠ f_C) → (∀ x y : ℝ, 0 < x -> x < y -> g x ≥ g y)) :=
by sorry

end increasing_function_on_interval_l158_158861


namespace divide_into_n_rectangles_l158_158708

theorem divide_into_n_rectangles (n : ℕ) (marked_cells : set (ℕ × ℕ)) :
  (∃ (positions : list (ℕ × ℕ)), 
    positions.length = 2 * n ∧ 
    (∀ i j, i ≠ j → positions.nth i ≠ positions.nth j) ∧
    (∀ pos, pos ∈ marked_cells ↔ pos ∈ positions.to_finset) ∧
    (∀ (x₁ x₂ y₁ y₂ : ℕ), 
      (x₁, y₁) ∈ marked_cells ∧ (x₂, y₂) ∈ marked_cells → 
      (x₁ = x₂ ∨ y₁ = y₂))) →
  (∃ (rects : list (set (ℕ × ℕ))), 
    rects.length = n ∧ 
    (⋃ rect ∈ rects, rect) = marked_cells ∧
    (∀ rect, rect ∈ rects → 
      (∃ x₁ x₂ y₁ y₂, ∀ x y, (x, y) ∈ rect ↔ (x₁ ≤ x ∧ x ≤ x₂ ∧ y₁ ≤ y ∧ y ≤ y₂)))) :=
sorry

end divide_into_n_rectangles_l158_158708


namespace life_persistence_l158_158646

-- Define the parameters and properties of the grid.
def grid_alive_persists (m n : ℕ) : Prop :=
  ∃ config : Vector (Vector Bool m) n,
    ∀ t : ℕ, ∃ i j, config[i][j] = true

-- The main statement of our problem in Lean
theorem life_persistence (m n : ℕ) :
  (m ≠ 1 ∨ n ≠ 1) ∧ (m ≠ 1 ∨ n ≠ 3) ∧ (m ≠ 3 ∨ n ≠ 1) →
  grid_alive_persists m n :=
sorry

end life_persistence_l158_158646


namespace problem1_l158_158469

theorem problem1 (n : ℕ) (hn : 0 < n) : 20 ∣ (4 * 6^n + 5^(n+1) - 9) := 
  sorry

end problem1_l158_158469


namespace minimum_value_l158_158546

def quadratic_expression (x y : ℝ) : ℝ := x^2 - 2*x*y + y^2 + 2*y + 1

theorem minimum_value :
  ∃ (x y : ℝ), quadratic_expression x y = 0 :=
by
  use [(-1/2), (-1/2)]
  simp [quadratic_expression]
  sorry

end minimum_value_l158_158546


namespace arccos_one_over_sqrt_two_eq_pi_over_four_l158_158929

theorem arccos_one_over_sqrt_two_eq_pi_over_four : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_eq_pi_over_four_l158_158929


namespace modular_expression_l158_158517

theorem modular_expression (h₁ : 7 * 55 ≡ 1 [MOD 63]) (h₂ : 13 * 29 ≡ 1 [MOD 63]) :
  3 * (7⁻¹ : ℤ) + 9 * (13⁻¹ : ℤ) ≡ 48 [MOD 63] := 
by {
  -- define shorthand for inverses 
  let inv7 : ℤ := 55,
  let inv13 : ℤ := 29,
  -- confirm the multiplicative inverses
  have h₃ : (7⁻¹ : ℤ) = inv7 := by {
    norm_num,
    exact h₁,
  },
  have h₄ : (13⁻¹ : ℤ) = inv13 := by {
    norm_num,
    exact h₂,
  },
  -- substitute and simplify
  rw [h₃, h₄],
  calc
    3 * 55 + 9 * 29
      ≡ 165 + 261 [MOD 63] : by norm_num
  ... ≡ 426 [MOD 63] : by norm_num
  ... ≡ 48 [MOD 63] : by norm_num,
  sorry
}

end modular_expression_l158_158517


namespace problem1_problem2_l158_158466

-- Problem 1: Lean 4 Statement
theorem problem1 (n : ℕ) (hn : n > 0) : 20 ∣ (4 * 6^n + 5^(n + 1) - 9) :=
sorry

-- Problem 2: Lean 4 Statement
theorem problem2 : (3^100 % 7) = 4 :=
sorry

end problem1_problem2_l158_158466


namespace increase_product_types_related_to_model_store_selection_model_1_prediction_correct_model_2_prediction_correct_l158_158080

-- Define the given profit data
def profit_data : List (ℕ × ℝ) :=
  [(2014, 27.6), (2015, 42.0), (2016, 38.4), (2017, 48.0), (2018, 63.6),
   (2019, 63.7), (2020, 72.8), (2021, 80.1), (2023, 99.3)]

def recent_profit_data : List (ℕ × ℝ) :=
  [(2019, 63.7), (2020, 72.8), (2021, 80.1), (2023, 99.3)]

-- Define the chi-square critical value for a significance level
def chi_square_critical_value : ℝ := 3.841

-- Define the contingency table data
def contingency_table := 
  { selected_2000_types := 2,
    selected_3000_types := 5,
    not_selected_2000_types := 3,
    not_selected_3000_types := 0 }

-- Define the computed chi-square value for the given contingency table
def computed_chi_square_value : ℝ := 4.29

-- Proof Problem 1: Prove that the increase in product types is related to model store selection
theorem increase_product_types_related_to_model_store_selection :
  computed_chi_square_value > chi_square_critical_value :=
sorry

-- Linear regression coefficients for the first data period
def model_1_coeffs : (ℝ × ℝ) := (7.627, -15332.20)

-- Linear regression coefficients for the second data period
def model_2_coeffs : (ℝ × ℝ) := (5.89, -11828.41)

-- Prediction for 2024 using Model 1
def model_1_prediction_2024 : ℝ := 104.9

-- Prediction for 2024 using Model 2
def model_2_prediction_2024 : ℝ := 93.0

-- Proof Problem 2: Prove that the regression models correctly predict the profit for 2024
theorem model_1_prediction_correct :
  model_1_prediction_2024 ≈ 104.9 :=
sorry

theorem model_2_prediction_correct :
  model_2_prediction_2024 ≈ 93.0 :=
sorry

end increase_product_types_related_to_model_store_selection_model_1_prediction_correct_model_2_prediction_correct_l158_158080


namespace line_AB_passes_through_P_l158_158780

noncomputable def tangency_external (ω1 ω2 : Circle) (R : Point) : Prop := sorry
noncomputable def intersection_of_tangents (ω1 ω2 : Circle) (P : Point) : Prop := sorry
noncomputable def perp (A B R : Point) : Prop := sorry

theorem line_AB_passes_through_P 
  (ω1 ω2 : Circle) (R P A B : Point)
  (h_tangent : tangency_external ω1 ω2 R)
  (h_intersection : intersection_of_tangents ω1 ω2 P)
  (h_perp : perp A B R) 
  (hA : A ∈ ω1)
  (hB : B ∈ ω2) : 
  Line_through A B P :=
sorry

end line_AB_passes_through_P_l158_158780


namespace value_of_f_neg_a_l158_158565

noncomputable def f (x : ℝ) : ℝ := x^3 + Real.sin x + 1

theorem value_of_f_neg_a (a : ℝ) (h : f a = 2) : f (-a) = -2 := 
by 
  sorry

end value_of_f_neg_a_l158_158565


namespace non_zero_real_solution_of_equation_l158_158795

noncomputable def equation_solution : Prop :=
  ∀ (x : ℝ), x ≠ 0 ∧ (7 * x) ^ 14 = (14 * x) ^ 7 → x = 2 / 7

theorem non_zero_real_solution_of_equation : equation_solution := sorry

end non_zero_real_solution_of_equation_l158_158795


namespace value_of_f_at_zero_l158_158599

-- Define the functions
def g (x : ℝ) : ℝ := 1 - 3 * x
def f_comp (x : ℝ) : ℝ := (1 - x^2) / (1 + x^2)

-- State the theorem to be proved
theorem value_of_f_at_zero : f_comp (g (0)) = 4 / 5 := by
  sorry

end value_of_f_at_zero_l158_158599


namespace actual_distance_traveled_l158_158236

theorem actual_distance_traveled (D : ℝ) (T : ℝ) (h1 : D = 15 * T) (h2 : D + 35 = 25 * T) : D = 52.5 := 
by
  sorry

end actual_distance_traveled_l158_158236


namespace ratio_of_part_to_whole_l158_158490

theorem ratio_of_part_to_whole {N x : ℕ} 
    (h1 : x + 7 = N / 4 - 7) 
    (h2 : N = 280) : 
    x / N = 1 / 5 := 
by 
s

end ratio_of_part_to_whole_l158_158490


namespace probability_sum_even_l158_158292

def primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def sum_is_even (a b : ℕ) : Prop :=
  (a + b) % 2 = 0

theorem probability_sum_even :
  (nat.choose 10 2) = 45 →
  (∀ a b : ℕ, a ∈ primes → b ∈ primes → a ≠ b →
    (sum_is_even a b ↔ a ≠ 2 ∧ b ≠ 2)) →
  ((45 - 9) / 45 : ℚ) = 4 / 5 :=
by sorry

end probability_sum_even_l158_158292


namespace points_selection_l158_158870

noncomputable def num_valid_selections : ℕ := 18

theorem points_selection (L1 L2 : Line) (C1 C2 : Circle) (P : Fin 9 → Point) :
  (∀ i j k : Fin 9, i ≠ j → j ≠ k → k ≠ i → 
    ¬collinear (P i) (P j) (P k) ∧ ¬cocircular (P i) (P j) (P k)) →
  (∃ (S : Finset (Fin 9)), S.card = 4 ∧ 
    (∀ i j k : Fin 4, i ≠ j → j ≠ k → k ≠ i → 
      ¬collinear (P (S.toList.get i)) (P (S.toList.get j)) (P (S.toList.get k)) ∧ 
      ¬cocircular (P (S.toList.get i)) (P (S.toList.get j)) (P (S.toList.get k))) ∧
    S.card! = num_valid_selections) :=
sorry

end points_selection_l158_158870


namespace average_reciprocal_sequence_l158_158127

theorem average_reciprocal_sequence:
  (∑ i in Finset.range 10, (1 : ℚ) / ((2 * i + 1) * (2 * i + 3))) = 10 / 21 := by
  sorry

end average_reciprocal_sequence_l158_158127


namespace probability_even_sum_l158_158256

def first_ten_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def is_odd (n : ℕ) : Prop := n % 2 = 1

def count_odd_primes (primes : List ℕ) : ℕ :=
  primes.countp is_odd

def binom (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

open Nat

theorem probability_even_sum :
  let primes := first_ten_primes in
  let odd_primes := count_odd_primes primes in
  let num_ways_even_sum := binom odd_primes 2 in
  let total_ways := binom primes.length 2 in
  (num_ways_even_sum : ℚ) / (total_ways : ℚ) = 4 / 5 :=
by
  sorry

end probability_even_sum_l158_158256


namespace weight_difference_l158_158003

theorem weight_difference (W_A W_B W_C W_D W_E : ℝ)
  (h_avg_ABC : (W_A + W_B + W_C) / 3 = 80)
  (h_WA : W_A = 95)
  (h_avg_ABCD : (W_A + W_B + W_C + W_D) / 4 = 82)
  (h_avg_BCDE : (W_B + W_C + W_D + W_E) / 4 = 81) :
  W_E - W_D = 3 :=
by
  sorry

end weight_difference_l158_158003


namespace sum_of_angles_neg_2pi_over_3_l158_158190

theorem sum_of_angles_neg_2pi_over_3 (α β : ℝ) 
    (hαβ_in_range : α ∈ Ioo (-π/2) (π/2) ∧ β ∈ Ioo (-π/2) (π/2))
    (h_tan_roots : (α = (Real.arctan (roots.quadratic 1 (3 * Real.sqrt 3) 4)).fst ∧ 
                  β = (Real.arctan (roots.quadratic 1 (3 * Real.sqrt 3) 4)).snd) ∨ 
                  (α = (Real.arctan (roots.quadratic 1 (3 * Real.sqrt 3) 4)).snd ∧ 
                  β = (Real.arctan (roots.quadratic 1 (3 * Real.sqrt 3) 4)).fst)) : 
  α + β = -2 * π / 3 := 
sorry

end sum_of_angles_neg_2pi_over_3_l158_158190


namespace arccos_sqrt_half_l158_158970

theorem arccos_sqrt_half : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 := by
  sorry

end arccos_sqrt_half_l158_158970


namespace circles_intersect_l158_158047

noncomputable def circle1 := {p : ℝ × ℝ | p.1^2 + p.2^2 = 4}
noncomputable def circle2 := {p : ℝ × ℝ | (p.1 - 2)^2 + p.2^2 = 9}

theorem circles_intersect :
  ∃ p : ℝ × ℝ, p ∈ circle1 ∧ p ∈ circle2 :=
sorry

end circles_intersect_l158_158047


namespace log_value_sum_l158_158230

theorem log_value_sum (x y z : ℝ) 
  (h1 : log 2 (log 3 x) = 0) 
  (h2 : log 3 (log 4 y) = 0) 
  (h3 : log 4 (log 2 z) = 0) : 
  x + y + z = 9 :=
sorry

end log_value_sum_l158_158230


namespace data_point_frequency_l158_158777

theorem data_point_frequency 
  (data : Type) 
  (categories : data → Prop) 
  (group_counts : data → ℕ) :
  ∀ d, categories d → group_counts d = frequency := sorry

end data_point_frequency_l158_158777


namespace max_value_frac_inv_l158_158186

theorem max_value_frac_inv (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3: x^2 + y^2 = 1) : 
  (frac_inv_value : ℝ) ∃ frac_inv_value, frac_inv_value = (1/x + 1/y) ∧ frac_inv_value ≤ 2 * Real.sqrt 2 :=
sorry

end max_value_frac_inv_l158_158186


namespace arccos_one_over_sqrt_two_eq_pi_four_l158_158904

theorem arccos_one_over_sqrt_two_eq_pi_four : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 := 
by
  sorry

end arccos_one_over_sqrt_two_eq_pi_four_l158_158904


namespace customers_at_discounted_price_l158_158524

-- Definitions
def popularity (p : ℝ) : ℝ := p
def discounted_price (initial_price discount : ℝ) : ℝ :=
  initial_price * (1 - discount)

-- Inverse proportionality constant
def k : ℝ := 15 * 300

-- Prove the number of customers who buy the blender at a 25% discount
theorem customers_at_discounted_price :
  let original_price := 900
  let discount := 0.25
  let new_price := discounted_price original_price discount
  k / new_price = 6.67 :=
by
  let original_price := 900
  let discount := 0.25
  let new_price := discounted_price original_price discount
  have new_price_eq : new_price = 675 := by
    simp [discounted_price, original_price, discount]
  have popularity_eq : k / new_price = 6.67 := by
    simp [k, new_price_eq, new_price]
  exact popularity_eq

end customers_at_discounted_price_l158_158524


namespace ratio_of_diagonals_in_regular_octagon_l158_158385

-- Definition of a regular octagon
def regular_octagon (n : ℕ) : Prop :=
  n = 8

-- Condition for the diagonals having two distinct lengths
def has_two_distinct_diagonal_lengths (polygon : ℕ) : Prop :=
  regular_octagon polygon

-- Theorem statement: the ratio of the shorter diagonal to the longer diagonal in a regular octagon
theorem ratio_of_diagonals_in_regular_octagon (n : ℕ) (h : regular_octagon n) : 
  has_two_distinct_diagonal_lengths n → real.sqrt 2 * (real.sqrt 2 / 2) = 1 :=
by
  intros h_distinct_lengths
  sorry

end ratio_of_diagonals_in_regular_octagon_l158_158385


namespace equal_opposite_triangles_l158_158855

-- Define the lengths of sides and conditions given
variable (a : ℝ)  -- side length of the square table
variable (t1 t2 t3 t4 : ℝ)  -- heights of the triangular overhangs

-- Define the conditions provided
axiom h1 : t1 = t2  -- adjacent overhangs equality
axiom h2 : t3 = t4  -- the other pair of opposite overhangs
axiom h3 : no_overlap : ∀ i j, i ≠ j → tᵢ ∩ tⱼ = ∅  -- tablecloth does not overlap itself

-- Define the theorem statement
theorem equal_opposite_triangles : t1 = t3 :=
by sorry

end equal_opposite_triangles_l158_158855


namespace sum_fraction_sets_l158_158126

theorem sum_fraction_sets (m n : ℕ) (hm : 1 < m) :
  let a_1 := (Finset.sum (Finset.range m) (λ k, k / m))
      a_2 := (Finset.sum (Finset.range (m^2)) (λ k, k / m^2)) - a_1
      a_3 := (Finset.sum (Finset.range (m^3)) (λ k, k / m^3)) - (a_2 + a_1)
      -- continues similarly for a_4, ..., a_n
      sum_all := a_1 + a_2 + ... + finset_sum (Finset.range (m^n)) (λ k, k / m^n)
  sum_all = (m^n - 1) / 2 :=
sorry

end sum_fraction_sets_l158_158126


namespace arccos_sqrt2_l158_158937

def arccos_eq (x : ℝ) := arccos x
def range_arccos := ∀ (x : ℝ), 0 ≤ arccos x ∧ arccos x ≤ π
def cos_pi_div_four : Real.cos (π / 4) = 1 / Real.sqrt 2 := by sorry

theorem arccos_sqrt2 :
  arccos_eq (1 / Real.sqrt 2) = π / 4 :=
by
  have h1 : Real.cos (π / 4) = 1 / Real.sqrt 2, from cos_pi_div_four
  sorry

end arccos_sqrt2_l158_158937


namespace road_length_is_10km_l158_158505

noncomputable def project_completion
  (days_total : ℕ) (days_initial : ℕ) (initial_completion : ℕ)
  (men_initial : ℕ) (men_total : ℕ) : ℕ :=
initial_completion * (days_total / days_initial) * (men_total / men_initial)

theorem road_length_is_10km :
  ∀ (days_total days_initial initial_completion men_initial men_total : ℕ),
  days_total = 300 →
  days_initial = 100 →
  initial_completion = 2 →
  men_initial = 30 →
  men_total = 60 →
  project_completion days_total days_initial initial_completion men_initial men_total = 10 :=
begin
  intros,
  -- The proof will calculate project_completion and verify it equals 10
  sorry
end

end road_length_is_10km_l158_158505


namespace triangle_inequality_l158_158339

variable {ℝ : Type} [OrderedAddCommGroup ℝ] [OrderedSemiring ℝ]

theorem triangle_inequality 
    {a b c : ℝ} 
    (ha : 0 < a) 
    (hb : 0 < b) 
    (hc : 0 < c) 
    (triangle : (a + b > c) ∧ (b + c > a) ∧ (c + a > b)) :
    a^2 * c * (a - b) + b^2 * a * (b - c) + c^2 * b * (c - a) ≥ 0 :=
sorry

end triangle_inequality_l158_158339


namespace subset_pairwise_differences_l158_158722

theorem subset_pairwise_differences (n : ℕ) :
  ∃ (S : Finset ℕ), S ⊆ Finset.range (n + 1) ∧ S.card ≤ 2 * Nat.floor (Real.sqrt n) + 1 ∧
  (∀ d ∈ Finset.range n, ∃ x y ∈ S, d = Int.natAbs (x - y)) := 
sorry

end subset_pairwise_differences_l158_158722


namespace complex_in_first_quadrant_l158_158752

noncomputable def quadrant (z : ℂ) : String :=
if z.re > 0 ∧ z.im > 0 then "first"
else if z.re < 0 ∧ z.im > 0 then "second"
else if z.re < 0 ∧ z.im < 0 then "third"
else if z.re > 0 ∧ z.im < 0 then "fourth"
else "none"

theorem complex_in_first_quadrant : 
  let z := (i : ℂ) / (1 + i) 
  in quadrant z = "first" :=
by
  let z := (i : ℂ) / (1 + i)
  have hz : z = 1 / 2 + (1 / 2) * I := sorry  -- Simplification step
  rw hz
  simp [quadrant]
  let cond := (1 / 2 : ℝ) > 0 ∧ (1 / 2 : ℝ) > 0
  simp [cond]
  exact rfl

end complex_in_first_quadrant_l158_158752


namespace num_squares_intersecting_circle_l158_158081

def unit_square_intersects_circle (x y : ℤ) (r : ℝ) : Prop :=
  ∃ (cx cy : ℝ), |cx - x| < 1 ∧ |cy - y| < 1 ∧ cx^2 + cy^2 < r^2

theorem num_squares_intersecting_circle : 
  let r := 6 in 
  let n := 132 in 
  ∑ x in finset.range (2 * r + 1), ∑ y in finset.range (2 * r + 1), if unit_square_intersects_circle x y r then 1 else 0 = n :=
by
  sorry

end num_squares_intersecting_circle_l158_158081


namespace arccos_sqrt2_l158_158933

def arccos_eq (x : ℝ) := arccos x
def range_arccos := ∀ (x : ℝ), 0 ≤ arccos x ∧ arccos x ≤ π
def cos_pi_div_four : Real.cos (π / 4) = 1 / Real.sqrt 2 := by sorry

theorem arccos_sqrt2 :
  arccos_eq (1 / Real.sqrt 2) = π / 4 :=
by
  have h1 : Real.cos (π / 4) = 1 / Real.sqrt 2, from cos_pi_div_four
  sorry

end arccos_sqrt2_l158_158933


namespace arccos_one_over_sqrt_two_eq_pi_over_four_l158_158901

theorem arccos_one_over_sqrt_two_eq_pi_over_four : 
  Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_eq_pi_over_four_l158_158901


namespace population_after_decade_l158_158024

variables (InitialPopulation : ℝ) (rate_of_increase : ℝ) (years : ℕ)
#check InitialPopulation
#check rate_of_increase
#check years

noncomputable def final_population (InitialPopulation : ℝ) (rate_of_increase : ℝ) (years : ℕ) : ℝ :=
  InitialPopulation * (1 + rate_of_increase) ^ years

theorem population_after_decade :
  final_population 175000 0.05 10 ≈ 285056 :=
by sorry


end population_after_decade_l158_158024


namespace g_of_f_neg_5_l158_158338

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^2 - 8

-- Assume g(42) = 17
axiom g_f_5_eq_17 : ∀ (g : ℝ → ℝ), g (f 5) = 17

-- State the theorem to be proven
theorem g_of_f_neg_5 (g : ℝ → ℝ) : g (f (-5)) = 17 :=
by
  sorry

end g_of_f_neg_5_l158_158338


namespace parabola_directrix_eq_l158_158014

theorem parabola_directrix_eq (p : ℝ) (h : y^2 = 2 * x ∧ p = 1) : x = -p / 2 := by
  sorry

end parabola_directrix_eq_l158_158014


namespace arccos_one_over_sqrt_two_eq_pi_over_four_l158_158928

theorem arccos_one_over_sqrt_two_eq_pi_over_four : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_eq_pi_over_four_l158_158928


namespace solve_problem_l158_158618

theorem solve_problem (a b c d : ℤ) (h1 : a - b - c + d = 13) (h2 : a + b - c - d = 5) : (b - d) ^ 2 = 16 :=
by
  sorry

end solve_problem_l158_158618


namespace linear_function_quadrants_l158_158582

-- Lean 4 statement
theorem linear_function_quadrants (a b c : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0) (k : ℝ)
  (h₃ : (b + c - a) / a = k) (h₄ : (a + c - b) / b = k) (h₅ : (a + b - c) / c = k) :
  (∃ (k : ℝ), k = 2 ∨ k = -2) → ∀ x : ℝ, 
  let y := k * x - k in
  (if 1 ≤ x ∧ 0 ≤ y then (1, 4)
   else if 0 ≤ x ∧ 0 ≤ y then 1
   else if x ≤ 0 ∧ 0 ≤ y then 2
   else if x > 0 ∧ y < 0 then 4
   else 3) = 1 ∨ (if 1 ≤ x ∧ y ≤ 0 then 4 else 3) := 
sorry

end linear_function_quadrants_l158_158582


namespace train_travel_time_l158_158402

theorem train_travel_time 
  (speed : ℝ := 120) -- speed in kmph
  (distance : ℝ := 80) -- distance in km
  (minutes_in_hour : ℝ := 60) -- conversion factor
  : (distance / speed) * minutes_in_hour = 40 :=
by
  -- Sorry is used as a placeholder for the proof
  sorry

end train_travel_time_l158_158402


namespace find_explicit_formula_find_range_of_m_l158_158345

-- Conditions
def func (a b x : ℝ) : ℝ := a * x^2 + (b - 2) * x + 3

def diff_func (a b x : ℝ) : ℝ := func a b (x + 1) - func a b x

-- Explicit definition based on conditions
theorem find_explicit_formula (a b : ℝ) (h₁ : a ≠ 0) (h₂ : ∀ x, diff_func a b x = 2 * x - 1) : 
  func a b = λ x, x^2 - 2 * x + 3 :=
sorry

-- Definition of g(x)
def g (a b m x : ℝ) : ℝ := func a b x - m * x

-- Range for m based on the condition
theorem find_range_of_m (a b : ℝ) (h₁ : a ≠ 0) (h₂ : ∀ x, diff_func a b x = 2 * x - 1) (m : ℝ) :
  (∀ x1 x2 ∈ set.Icc (1:ℝ) 2, abs (g a b m x1 - g a b m x2) ≤ 2) ↔ m ∈ set.Icc (-1:ℝ) 3 :=
sorry

end find_explicit_formula_find_range_of_m_l158_158345


namespace arccos_one_over_sqrt_two_eq_pi_over_four_l158_158898

theorem arccos_one_over_sqrt_two_eq_pi_over_four : 
  Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_eq_pi_over_four_l158_158898


namespace arccos_one_over_sqrt_two_l158_158943

theorem arccos_one_over_sqrt_two : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_l158_158943


namespace borrowed_sheets_l158_158133

theorem borrowed_sheets (sheets borrowed: ℕ) (average_page : ℝ) 
  (total_pages : ℕ := 80) (pages_per_sheet : ℕ := 2) (total_sheets : ℕ := 40) 
  (h1 : borrowed ≤ total_sheets)
  (h2 : sheets = total_sheets - borrowed)
  (h3 : average_page = 26) : borrowed = 17 :=
sorry 

end borrowed_sheets_l158_158133


namespace probability_even_sum_l158_158284

def first_ten_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def distinct_pairs (l : List ℕ) : List (ℕ × ℕ) :=
  (l.product l).filter (λ p, p.1 < p.2)

theorem probability_even_sum : 
  (first_ten_primes.length = 10) →
  (∀ a b : ℕ, a ∈ first_ten_primes → b ∈ first_ten_primes → a ≠ b → 
    ((a + b) % 2 = 0 ↔ 2 ∉ [a, b])) →
  (∑ pair in distinct_pairs first_ten_primes, ((pair.1 + pair.2) % 2 = 0).toNat) 
    = 36 → 
  ((∑ pair in distinct_pairs first_ten_primes, (1:ℚ)) ⁻¹ * 
   (∑ pair in distinct_pairs first_ten_primes, ((pair.1 + pair.2) % 2 = 0).toNat))
    = 4 / 5 := by
  sorry

end probability_even_sum_l158_158284


namespace trainSpeed_l158_158059

/-- This represents the length of the train in meters. --/
def lengthOfTrain : ℝ := 130

/-- This represents the time in seconds that the train takes to cross the man. --/
def timeToCross : ℝ := 6

/-- This represents the speed of the man in km/h. --/
def speedOfMan_kmh : ℝ := 5

/-- This represents the equivalent speed of the man in m/s. --/
def speedOfMan_ms : ℝ := speedOfMan_kmh * (1000 / 3600)

/-- This represents the speed of the train we want to prove, converted from m/s to km/h. --/
def speedOfTrain_kmh (speedInMs : ℝ) : ℝ := speedInMs * (3600 / 1000)

/-- The speed of the train in m/s, derived from the given conditions. --/
def speedOfTrain_ms : ℝ := (lengthOfTrain / timeToCross) - speedOfMan_ms

/-- The ultimate theorem to prove that the speed of the train is approximately 73 km/h. --/
theorem trainSpeed (ε : ℝ) (hε : ε > 0) : abs (speedOfTrain_kmh speedOfTrain_ms - 73) < ε := 
by
  sorry

end trainSpeed_l158_158059


namespace cupcakes_frosted_l158_158348

def Cagney_rate := 1 / 25
def Lacey_rate := 1 / 35
def time_duration := 600
def combined_rate := Cagney_rate + Lacey_rate
def total_cupcakes := combined_rate * time_duration

theorem cupcakes_frosted (Cagney_rate Lacey_rate time_duration combined_rate total_cupcakes : ℝ) 
  (hC: Cagney_rate = 1 / 25)
  (hL: Lacey_rate = 1 / 35)
  (hT: time_duration = 600)
  (hCR: combined_rate = Cagney_rate + Lacey_rate)
  (hTC: total_cupcakes = combined_rate * time_duration) :
  total_cupcakes = 41 :=
sorry

end cupcakes_frosted_l158_158348


namespace find_smallest_positive_period_l158_158678

noncomputable def smallest_positive_period (ω φ : ℝ) (hω : ω > 0) (hφ : 0 < φ ∧ φ < π / 2) : ℝ :=
  if (∃ (f : ℝ → ℝ), f = (λ x, cos (ω * x + φ)) 
      ∧ (∀ x y : ℝ, -π / 24 ≤ x ∧ x ≤ y ∧ y ≤ 5 * π / 24 → f x ≤ f y) 
      ∧ (f (-π / 24) = -f (5 * π / 24))
      ∧ (f (-π / 24) = -f (11 * π / 24))) 
  then π 
  else 0

theorem find_smallest_positive_period :
  ∀ (ω φ : ℝ) (hω : ω > 0) (hφ : 0 < φ ∧ φ < π / 2)
  (h_monotonic : ∀ x y : ℝ, -π / 24 ≤ x ∧ x ≤ y ∧ y ≤ 5 * π / 24 → (cos(ω * x + φ)) ≤ (cos(ω * y + φ)))
  (h_symmetry : cos(ω * (-π / 24) + φ) = -cos(ω * (5 * π / 24) + φ) = -cos(ω * (11 * π / 24) + φ)), 
    smallest_positive_period ω φ hω hφ = π :=
sorry

end find_smallest_positive_period_l158_158678


namespace arccos_of_one_over_sqrt_two_l158_158885

theorem arccos_of_one_over_sqrt_two :
  Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 := 
sorry

end arccos_of_one_over_sqrt_two_l158_158885


namespace greatest_prime_factor_of_expression_l158_158694

def square (n : ℕ) : ℕ := n * n
def cube (n : ℕ) : ℕ := n * n * n

theorem greatest_prime_factor_of_expression :
  let a := square 42
  let b := cube 22
  let sum := a + b
  prime 3103 ∧ (∀ p : ℕ, prime p ∧ p ∣ sum → p ≤ 3103) :=
by
  let a := square 42
  let b := cube 22
  let sum := a + b
  sorry

end greatest_prime_factor_of_expression_l158_158694


namespace peg_stickers_total_l158_158714

def stickers_in_red_folder : ℕ := 10 * 3
def stickers_in_green_folder : ℕ := 10 * 2
def stickers_in_blue_folder : ℕ := 10 * 1

def total_stickers : ℕ := stickers_in_red_folder + stickers_in_green_folder + stickers_in_blue_folder

theorem peg_stickers_total : total_stickers = 60 := by
  sorry

end peg_stickers_total_l158_158714


namespace alternate_draws_probability_l158_158828

def draw_prob (white black red : ℕ) : ℚ := sorry

theorem alternate_draws_probability:
  (draw_prob 5 5 1 = 1 / 231) := 
begin
  sorry
end

end alternate_draws_probability_l158_158828


namespace quadratic_has_real_roots_l158_158627

theorem quadratic_has_real_roots (k : ℝ) : (∃ x : ℝ, k * x^2 - 6 * x + 9 = 0) ↔ (k ≤ 1 ∧ k ≠ 0) :=
by
  sorry

end quadratic_has_real_roots_l158_158627


namespace count_interesting_moments_l158_158390

def interesting_moments (X Y : ℕ) : Prop :=
  (0.5 * Y = 6 * X) ∧ (6 * Y = 0.5 * X)

theorem count_interesting_moments :
  ∃ n, n = 143 ∧ ∀ X, 1 ≤ X ∧ X ≤ 720 → ∃ Y, 1 ≤ Y ∧ Y ≤ 720 ∧ interesting_moments X Y → X = 12 * Y :=
by
  sorry

end count_interesting_moments_l158_158390


namespace fourth_vertex_of_square_l158_158775

theorem fourth_vertex_of_square :
  ∃ z₄ : ℂ, 
    (∃ z₁ z₂ z₃ : ℂ, 
      z₁ = (2 + 3 * complex.I) ∧ 
      z₂ = (-3 + 2 * complex.I) ∧ 
      z₃ = (-2 - 3 * complex.I) ∧ 
      (∃ x y : ℂ, (z₁ - z₂) * (z₁ - z₃) = (x - z₂) * (y - z₃)) ∧ 
      (∃ a b : ℂ, (z₂ - z₃) * (z₁ - b) = (a - z₃) * (z₁ - z₄))
    ) ∧ z₄ = (0.2 - 1.5 * complex.I) :=
sorry

end fourth_vertex_of_square_l158_158775


namespace largest_n_for_square_sum_l158_158995

theorem largest_n_for_square_sum :
  ∃ n : ℕ, (4^995 + 4^1500 + 4^n).natAbs.is_square ∧ (∀ m : ℕ, (4^995 + 4^1500 + 4^m).natAbs.is_square → m ≤ 1490) :=
sorry

end largest_n_for_square_sum_l158_158995


namespace circles_coplanar_or_sphere_l158_158356

noncomputable section

variables {P Q R : Point}
variable (C_P C_Q C_R : Circle) 

-- The circles defined with the given properties.
variables (h1 : C_P ∋ Q)
variables (h2 : C_P ∋ R)
variables (h3 : C_Q ∋ R)
variables (h4 : C_Q ∋ P)
variables (h5 : C_R ∋ P)
variables (h6 : C_R ∋ Q)

-- Coinciding tangents properties.
variables (t1 : TangentAt C_Q P = TangentAt C_R P)
variables (t2 : TangentAt C_R Q = TangentAt C_P Q)
variables (t3 : TangentAt C_P R = TangentAt C_Q R)

-- The end goal to be proven.
theorem circles_coplanar_or_sphere :
  Coplanar C_P C_Q C_R ∨ Spherical C_P C_Q C_R :=
sorry

end circles_coplanar_or_sphere_l158_158356


namespace number_of_truth_sayers_is_666_l158_158822

def person := ℕ
def number_of_people : ℕ := 2000
def is_truth_sayer (p : person) : Prop := ∃ n : ℕ, n ≤ 2000     -- We use a dummy definition here for is_truth_sayer

def statement (p : person) : Prop :=
  let p1 := (p + 1) % number_of_people in
  let p2 := (p + 2) % number_of_people in
  let p3 := (p + 3) % number_of_people in
  (is_truth_sayer p → (¬ is_truth_sayer p1 ∧ ¬ is_truth_sayer p2) ∨ (¬ is_truth_sayer p2 ∧ ¬ is_truth_sayer p3) ∨ (¬ is_truth_sayer p1 ∧ ¬ is_truth_sayer p3)) ∧ 
  (¬ is_truth_sayer p → (is_truth_sayer p1 ∨ is_truth_sayer p2) ∧ (is_truth_sayer p2 ∨ is_truth_sayer p3) ∧ (is_truth_sayer p1 ∨ is_truth_sayer p3))

-- Main theorem to be proven
theorem number_of_truth_sayers_is_666 : ∃ (n : ℕ), n = 666 ∧ ∀ p : person, statement p :=
sorry

end number_of_truth_sayers_is_666_l158_158822


namespace reflected_triangle_on_incircle_l158_158810

-- Define the problem statement.
theorem reflected_triangle_on_incircle
  (A1 A2 A3 K1 K2 K3 L1 L2 L3 : Point)
  (h1 : acute A1 A2 A3)
  (h2 : altitude_foot A1 A2 A3 K1 ∧ altitude_foot A2 A3 A1 K2 ∧ altitude_foot A3 A1 A2 K3)
  (h3 : incircle_touch_point A1 A2 A3 L1 ∧ incircle_touch_point A2 A3 A1 L2 ∧ incircle_touch_point A3 A1 A2 L3)
  (h4 : reflected_line K1 K2 L1 L2 K2' ∧ reflected_line K2 K3 L2 L3 K3' ∧ reflected_line K3 K1 L3 L1 K1') :
  exists M1 M2 M3 : Point, triangle M1 M2 M3 ∧ (incircle A1 A2 A3).contains M1 ∧ (incircle A1 A2 A3).contains M2 ∧ (incircle A1 A2 A3).contains M3 := sorry

end reflected_triangle_on_incircle_l158_158810


namespace f_m_minus_1_pos_l158_158692

variable {R : Type*} [LinearOrderedField R]

def quadratic_function (x a : R) : R :=
  x^2 - x + a

theorem f_m_minus_1_pos {a m : R} (h_pos : 0 < a) (h_fm : quadratic_function m a < 0) :
  quadratic_function (m - 1 : R) a > 0 :=
sorry

end f_m_minus_1_pos_l158_158692


namespace cube_root_sum_power_l158_158113

theorem cube_root_sum_power (a b : ℝ) : 
  (root 3 (4 * 5 ^ 7)) = (5 ^ (7 / 3)) * (root 3 4) := sorry

end cube_root_sum_power_l158_158113


namespace discounted_price_l158_158827

variable (marked_price : ℝ) (discount_rate : ℝ)
variable (marked_price_def : marked_price = 150)
variable (discount_rate_def : discount_rate = 20)

theorem discounted_price (hmp : marked_price = 150) (hdr : discount_rate = 20) : 
  marked_price - (discount_rate / 100) * marked_price = 120 := by
  rw [hmp, hdr]
  sorry

end discounted_price_l158_158827


namespace ellipse_triangle_perimeter_l158_158197

theorem ellipse_triangle_perimeter
  (a b : ℝ)
  (h1 : a > b)
  (h2 : b > 0)
  (F_1 F_2 A B : ℝ)
  (h_ellipse : ∀ {x y : ℝ}, (x, y) ∈ {(x, y) | x^2 / a^2 + y^2 / b^2 = 1})
  (h_foci : F_1 = a ∨ F_1 = -a)
  (h_foci : F_2 = a ∨ F_2 = -a)
  (h_intersect : True) := -- A line through F2 intersects at points A and B
|AF_1| + |AF_2| = 2 * a -> 
|BF_1| + |BF_2| = 2 * a -> 
|AF_1| + |AB| + |BF_1| = 4 * a := sorry

end ellipse_triangle_perimeter_l158_158197


namespace find_Q_l158_158586

variable {x P Q : ℝ}

theorem find_Q (h₁ : x + 1 / x = P) (h₂ : P = 1) : x^6 + 1 / x^6 = 2 :=
by
  sorry

end find_Q_l158_158586


namespace problem1_problem2_l158_158463

variable (α : ℝ) (n : ℕ) -- Define the variables

theorem problem1 :
  (∑ k in Finset.range n, Real.cos^2 (k.succ * α)) = 
  ((Real.sin ((n + 1) * α) * Real.cos (n * α) / (2 * Real.sin α)) + (n - 1) / 2) :=
sorry

theorem problem2 :
  (∑ k in Finset.range n, Real.sin^2 (k.succ * α)) = 
  ((n + 1) / 2 - (Real.sin ((n + 1) * α) * Real.cos (n * α) / (2 * Real.sin α))) :=
sorry

end problem1_problem2_l158_158463


namespace cube_surface_area_with_holes_l158_158497

/-- A theorem to calculate the entire surface area, including internal surface area, of a wooden cube with specific circular holes. -/
theorem cube_surface_area_with_holes :
  let s := 4 in -- side length of the cube
  let r := 1 in -- radius of the circular hole
  (6 * s^2) - (6 * π * r^2) + (6 * (2 * π * r * s)) = 96 + 42 * π := sorry

end cube_surface_area_with_holes_l158_158497


namespace minimum_value_expression_l158_158786

noncomputable def expr (x y : ℝ) : ℝ :=
  Real.sqrt (x^2 + y^2 - 2*x - 2*y + 2) + 
  Real.sqrt (x^2 + y^2 - 2*x + 4*y + 2*Real.sqrt 3*y + 8 + 4*Real.sqrt 3) +
  Real.sqrt (x^2 + y^2 + 8*x + 4*Real.sqrt 3*x - 4*y + 32 + 16*Real.sqrt 3)

theorem minimum_value_expression : (∃ x y : ℝ, expr x y = 3*Real.sqrt 6 + 4*Real.sqrt 2) :=
sorry

end minimum_value_expression_l158_158786


namespace curve_rect_eqn_theorem_line_intersect_value_l158_158662

-- Define the curve C in polar coordinates
def curve_polar_eqn (ρ θ : ℝ) : Prop :=
  ρ^2 = 4 * ρ * (Real.cos θ) + 5

-- Define the curve C in rectangular coordinates
def curve_rect_eqn (x y : ℝ) : Prop :=
  (x - 2)^2 + y^2 = 9

-- Define the parametric equations of the line l
def line_param_eqn (t : ℝ) : (ℝ × ℝ) :=
  (-2 + 2 * (Real.sqrt 5) / 5 * t, (Real.sqrt 5) / 5 * t)

-- Given condition for the line l intersection and point P condition 
def line_intersect_point_eqn (t : ℝ) : Prop :=
  ∃ A B : ℝ × ℝ, A = line_param_eqn t ∧ B = line_param_eqn t ∧ A ≠ B ∧ A.1 = -2 ∧ A.2 = 0

-- Theorem statement for problem part 1
theorem curve_rect_eqn_theorem (ρ θ x y : ℝ) (h : curve_polar_eqn ρ θ) :
  curve_rect_eqn x y :=
sorry

-- Theorem statement for problem part 2
theorem line_intersect_value (t : ℝ) (x y : ℝ) (h₁ : curve_rect_eqn x y) (h₂ : line_param_eqn t = (x, y)) :
  (1 / Real.abs (-2 - x)) + (1 / Real.abs (0 - y)) = (16 * Real.sqrt 5) / 35 :=
sorry

end curve_rect_eqn_theorem_line_intersect_value_l158_158662


namespace find_correct_function_l158_158141

noncomputable def option_A : ℝ → ℝ := λ x => Real.log (x + 1)
noncomputable def option_B : ℝ → ℝ := λ x => x * Real.sin x
noncomputable def option_C : ℝ → ℝ := λ x => x - x^3
noncomputable def option_D : ℝ → ℝ := λ x => 3 * x + Real.sin x

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def is_monotonically_increasing (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f x ≤ f y

theorem find_correct_function :
  (is_odd option_D) ∧ (is_monotonically_increasing option_D (-1) 1) ∧
  (¬(is_odd option_A) ∨ ¬(is_monotonically_increasing option_A (-1) 1)) ∧
  (¬(is_odd option_B) ∨ ¬(is_monotonically_increasing option_B (-1) 1)) ∧
  (¬(is_odd option_C) ∨ ¬(is_monotonically_increasing option_C (-1) 1)) :=
by
  sorry

end find_correct_function_l158_158141


namespace coordinates_of_B_l158_158357

theorem coordinates_of_B (A : ℝ × ℝ) (hA1 : A = (-3, -2)) : 
  let B := (A.1 - 2, A.2 + 3) in B = (-5, 1) :=
by 
  have hA2 : A.1 = -3 := by simp [hA1]
  have hA3 : A.2 = -2 := by simp [hA1]
  let B := (A.1 - 2, A.2 + 3)
  simp [hA2, hA3]
  sorry

end coordinates_of_B_l158_158357


namespace probability_even_sum_l158_158285

def first_ten_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def distinct_pairs (l : List ℕ) : List (ℕ × ℕ) :=
  (l.product l).filter (λ p, p.1 < p.2)

theorem probability_even_sum : 
  (first_ten_primes.length = 10) →
  (∀ a b : ℕ, a ∈ first_ten_primes → b ∈ first_ten_primes → a ≠ b → 
    ((a + b) % 2 = 0 ↔ 2 ∉ [a, b])) →
  (∑ pair in distinct_pairs first_ten_primes, ((pair.1 + pair.2) % 2 = 0).toNat) 
    = 36 → 
  ((∑ pair in distinct_pairs first_ten_primes, (1:ℚ)) ⁻¹ * 
   (∑ pair in distinct_pairs first_ten_primes, ((pair.1 + pair.2) % 2 = 0).toNat))
    = 4 / 5 := by
  sorry

end probability_even_sum_l158_158285


namespace problem_correct_calculation_l158_158443

theorem problem_correct_calculation (a b : ℕ) : 
  (4 * a - 2 * a ≠ 2) ∧ 
  (a^8 / a^4 ≠ a^2) ∧ 
  (a^2 * a^3 = a^5) ∧ 
  ((b^2)^3 ≠ b^5) :=
by {
  sorry
}

end problem_correct_calculation_l158_158443


namespace pizza_topping_slices_l158_158471

theorem pizza_topping_slices 
  (total_slices pepperoni_slices mushroom_slices olive_slices : ℕ)
  (pepperoni_slices_has_at_least_one_topping : pepperoni_slices = 8)
  (mushroom_slices_has_at_least_one_topping : mushroom_slices = 12)
  (olive_slices_has_at_least_one_topping : olive_slices = 14)
  (total_slices_has_one_topping : total_slices = 16)
  (slices_with_at_least_one_topping : 8 + 12 + 14 - 2 * x = 16) :
  x = 9 :=
by
  sorry

end pizza_topping_slices_l158_158471


namespace q_lt_t_l158_158551

-- Definitions based on the problem's conditions
variables {a b c : ℝ}

-- Given conditions
def triangle_with_area_circumradius (a b c : ℝ) (A : ℝ) (R : ℝ) : Prop :=
A = 1 / 4 ∧ R = 1 ∧ (a * b * c = 4 * A * R)

-- Definitions of q and t
def q (a b c : ℝ) : ℝ :=
√a + √b + √c

def t (a b c : ℝ) : ℝ :=
(1 / a) + (1 / b) + (1 / c)

-- Statement of the theorem
theorem q_lt_t (h : triangle_with_area_circumradius a b c (1 / 4) 1) : q a b c < t a b c :=
sorry

end q_lt_t_l158_158551


namespace vlad_score_l158_158376

theorem vlad_score :
  ∀ (rounds wins : ℕ) (totalPoints taroPoints vladPoints : ℕ),
    rounds = 30 →
    (wins = 5) →
    (totalPoints = rounds * wins) →
    (taroPoints = (3 * totalPoints) / 5 - 4) →
    (vladPoints = totalPoints - taroPoints) →
    vladPoints = 64 :=
by
  intros rounds wins totalPoints taroPoints vladPoints h1 h2 h3 h4 h5
  sorry

end vlad_score_l158_158376


namespace shortest_time_bake_l158_158410

/-- 
Given three cakes, each side of which needs to be baked for 1 minute,
and a pan that can bake two cakes at a time, prove that the shortest time
required to bake all three cakes is 3 minutes.
-/

theorem shortest_time_bake (cakes : ℕ) (bake_time : ℕ) (pan_capacity : ℕ) : 
  cakes = 3 → bake_time = 1 → pan_capacity = 2 → 
  ∃ t, t = 3 ∧ (∃ schedule : list (list ℕ), length schedule = t ∧ ∀ c, c < cakes → (∃ s, s ∈ schedule ∧ c ∈ s ∧ length s ≤ pan_capacity ∧ ∀ c' ∈ s, bake_time = 1) ) :=
by
  intro h_cakes h_bake_time h_pan_capacity
  use 3
  split
  · refl
  · sorry

end shortest_time_bake_l158_158410


namespace value_of_c_l158_158389

noncomputable def midpoint (p1 p2 : Point) : Point :=
  Point.mk ((p1.x + p2.x) / 2) ((p1.y + p2.y) / 2)

structure Point :=
  (x : ℝ)
  (y : ℝ)

def line_equation (line : ℝ × ℝ × ℝ) (p : Point) : Prop :=
  line.1 * p.x + line.2 * p.y = line.3

theorem value_of_c :
  let p1 := Point.mk 2 5 in
  let p2 := Point.mk 8 (-1) in
  let mid := midpoint p1 p2 in
  ∃ c : ℝ, (∀ point : Point, (line_equation (2, -1, c) point ↔ point = mid)) → c = 8 :=
by {
  intros p1 p2 mid,
  use 8,
  sorry  -- proof goes here
}

end value_of_c_l158_158389


namespace quadratic_square_binomial_l158_158131

theorem quadratic_square_binomial (a r s : ℚ) (h1 : a = r^2) (h2 : 2 * r * s = 26) (h3 : s^2 = 9) :
  a = 169/9 := sorry

end quadratic_square_binomial_l158_158131


namespace girls_select_same_color_l158_158825

def select_marbles (bag : ℕ) : Prop :=
  bag = 8 ∧ ∃ (white black : ℕ), white = 4 ∧ black = 4

def probability_same_color (prob : ℚ) : Prop :=
  prob = 1 / 35

theorem girls_select_same_color (bag : ℕ) (prob : ℚ) (h_bag : select_marbles bag) : 
  probability_same_color prob :=
sorry

end girls_select_same_color_l158_158825


namespace minimum_value_of_A_l158_158211

open Real

noncomputable def A (x y z : ℝ) : ℝ := cos (x - y) + cos (y - z) + cos (z - x)

theorem minimum_value_of_A :
  ∃ x y z ∈ Icc 0 (π / 2), A x y z ≥ 1 ∧ (∀ x' y' z' ∈ Icc 0 (π / 2), A x' y' z' ≥ A x y z) := sorry

end minimum_value_of_A_l158_158211


namespace simplified_expression_l158_158732

variable (m : ℝ) (h : m = Real.sqrt 3)

theorem simplified_expression : (m - (m + 9) / (m + 1)) / ((m^2 + 3 * m) / (m + 1)) = 1 - Real.sqrt 3 :=
by
  rw [h]
  sorry

end simplified_expression_l158_158732


namespace inscribed_quadrilateral_inradius_l158_158031

noncomputable def calculate_inradius (a b c d: ℝ) (A: ℝ) : ℝ := (A / ((a + c + b + d) / 2))

theorem inscribed_quadrilateral_inradius {a b c d: ℝ} (h1: a + c = 10) (h2: b + d = 10) (h3: a + b + c + d = 20) (hA: 12 = 12):
  calculate_inradius a b c d 12 = 6 / 5 :=
by
  sorry

end inscribed_quadrilateral_inradius_l158_158031


namespace arccos_one_over_sqrt_two_eq_pi_over_four_l158_158893

theorem arccos_one_over_sqrt_two_eq_pi_over_four : 
  Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_eq_pi_over_four_l158_158893


namespace sum_of_possible_k_l158_158997

theorem sum_of_possible_k : 
  let p1 := λ x : ℜ, x^2 - 4 * x + 3
  let p2 := λ (x k : ℜ), x^2 - 6 * x + k
  let k_values := {k | ∃ x : ℜ, p1 x = 0 ∧ p2 x k = 0}
  ∑ k in k_values, k = 14 :=
by
  sorry

end sum_of_possible_k_l158_158997


namespace area_of_circle_l158_158429

theorem area_of_circle (x y : ℝ) (h : x^2 + y^2 + 4 * x - 6 * y + 9 = 0) : real.pi * 2^2 = 4 * real.pi :=
by
  sorry

end area_of_circle_l158_158429


namespace sum_of_products_of_splits_l158_158464

theorem sum_of_products_of_splits (n : ℕ) (h : n = 25) :
  (∑ k in (finset.range n).powerset.filter (λ s, 1 ≤ s.card ∧ s.card < n),
    ((s.card) * (n - s.card))) =  300 :=
by
  sorry

end sum_of_products_of_splits_l158_158464


namespace difference_increased_decreased_l158_158386

theorem difference_increased_decreased (x : ℝ) (hx : x = 80) : 
  ((x * 1.125) - (x * 0.75)) = 30 := by
  have h1 : x * 1.125 = 90 := by rw [hx]; norm_num
  have h2 : x * 0.75 = 60 := by rw [hx]; norm_num
  rw [h1, h2]
  norm_num
  done

end difference_increased_decreased_l158_158386


namespace correct_calculation_l158_158441

variable {a : ℝ}

theorem correct_calculation : a^2 * a^3 = a^5 :=
by sorry

end correct_calculation_l158_158441


namespace ratio_of_new_time_to_previous_time_l158_158098

theorem ratio_of_new_time_to_previous_time:
  ∀ (original_distance original_time new_speed : ℝ),
  original_distance = 465 ∧ original_time = 5 ∧ new_speed = 62 →
  (original_distance / new_speed) / original_time = 1.5 :=
by
  intros original_distance original_time new_speed h,
  cases h with h1 h,
  cases h with h2 h3,
  sorry

end ratio_of_new_time_to_previous_time_l158_158098


namespace peg_stickers_total_l158_158715

def stickers_in_red_folder : ℕ := 10 * 3
def stickers_in_green_folder : ℕ := 10 * 2
def stickers_in_blue_folder : ℕ := 10 * 1

def total_stickers : ℕ := stickers_in_red_folder + stickers_in_green_folder + stickers_in_blue_folder

theorem peg_stickers_total : total_stickers = 60 := by
  sorry

end peg_stickers_total_l158_158715


namespace elements_map_to_4_l158_158175

def f (x : ℝ) : ℝ := x^2

theorem elements_map_to_4 :
  { x : ℝ | f x = 4 } = {2, -2} :=
by
  sorry

end elements_map_to_4_l158_158175


namespace quadratic_has_real_roots_l158_158628

theorem quadratic_has_real_roots (k : ℝ) : (∃ x : ℝ, k * x^2 - 6 * x + 9 = 0) ↔ (k ≤ 1 ∧ k ≠ 0) :=
by
  sorry

end quadratic_has_real_roots_l158_158628


namespace two_hundredth_digit_div_five_thirteen_l158_158425

theorem two_hundredth_digit_div_five_thirteen : 
  (let n := 200 
       d := 13 
       repeating_cycle := "384615".to_list -- Convert repeating cycle to list of digits
       cycle_length := repeating_cycle.length
       division := n / cycle_length
       remainder := n % cycle_length in
   repeating_cycle.nth remainder = some '8') :=
by sorry

end two_hundredth_digit_div_five_thirteen_l158_158425


namespace inequality_solution_l158_158541

theorem inequality_solution (x : ℝ) :
  (0 < x ∧ x ≤ 5 / 6 ∨ 2 < x) ↔ 
  ((2 * x) / (x - 2) + (x - 3) / (3 * x) ≥ 2) :=
by
  sorry

end inequality_solution_l158_158541


namespace intersection_of_M_and_N_l158_158605

-- Define sets M and N
def M : Set ℤ := {x | -2 ≤ x ∧ x ≤ 2}
def N : Set ℤ := {0, 1, 2}

-- The theorem to be proven: M ∩ N = {0, 1, 2}
theorem intersection_of_M_and_N : M ∩ N = {0, 1, 2} :=
by
  sorry

end intersection_of_M_and_N_l158_158605


namespace arccos_proof_l158_158958

noncomputable def arccos_identity : Prop := 
  ∃ x : ℝ, x = 1 / Real.sqrt 2 ∧ Real.arccos x = Real.pi / 4

theorem arccos_proof : arccos_identity :=
by
  sorry

end arccos_proof_l158_158958


namespace simplify_and_evaluate_l158_158730

-- Define the given expression
noncomputable def given_expression (m : ℝ) : ℝ :=
  (m - (m + 9) / (m + 1)) / ((m^2) + 3 * m) / (m + 1)

-- Define the condition
def condition (m : ℝ) : Prop :=
  m = Real.sqrt 3

-- Define the correct answer
def correct_answer : ℝ :=
  1 - Real.sqrt 3

-- State the theorem
theorem simplify_and_evaluate 
  (m : ℝ) (h : condition m) : 
  given_expression m = correct_answer := by
  sorry

end simplify_and_evaluate_l158_158730


namespace coupon1_greater_discount_l158_158482

theorem coupon1_greater_discount (x : ℝ) : 
  208.33 < x ∧ x < 300 →
  (0.12 * x > 25) ∧ (0.12 * x > 0.20 * x - 24) :=
by
intro h
cases h with h1 h2
split
{ linarith }
{ linarith }

end coupon1_greater_discount_l158_158482


namespace num_ways_to_place_pawns_l158_158172

theorem num_ways_to_place_pawns : 
  let chessboard := fin 5
  5! = 120 := by
  sorry

end num_ways_to_place_pawns_l158_158172


namespace range_of_c_div_a_plus_b_l158_158665

variables {A B C a b c : ℝ}
variables (triangle_ABC : a * b * sin C = sqrt 3 / 2 * (a^2 + b^2 - c^2))
variables (condition1 : (sin B * sin C) / (3 * sin A) = cos A / a + cos C / c)

noncomputable def range_of_c_over_a_plus_b : set ℝ :=
  {[1 / 2, 1)}

theorem range_of_c_div_a_plus_b :
  (range_of_c_over_a_plus_b = ∅ ∨ range_of_c_over_a_plus_b = {[1 / 2, 1)} ∨ ∃ l u, ¬(l < 1 / 2) ∧ (u < 1 ∨ l < 1)) :=
begin
  sorry
end

end range_of_c_div_a_plus_b_l158_158665


namespace paint_cans_needed_l158_158509

theorem paint_cans_needed
    (num_bedrooms : ℕ)
    (num_other_rooms : ℕ)
    (total_rooms : ℕ)
    (gallons_per_room : ℕ)
    (color_paint_cans_per_gallon : ℕ)
    (white_paint_cans_per_gallon : ℕ)
    (total_paint_needed : ℕ)
    (color_paint_cans_needed : ℕ)
    (white_paint_cans_needed : ℕ)
    (total_paint_cans : ℕ)
    (h1 : num_bedrooms = 3)
    (h2 : num_other_rooms = 2 * num_bedrooms)
    (h3 : total_rooms = num_bedrooms + num_other_rooms)
    (h4 : gallons_per_room = 2)
    (h5 : total_paint_needed = total_rooms * gallons_per_room)
    (h6 : color_paint_cans_per_gallon = 1)
    (h7 : white_paint_cans_per_gallon = 3)
    (h8 : color_paint_cans_needed = num_bedrooms * gallons_per_room * color_paint_cans_per_gallon)
    (h9 : white_paint_cans_needed = (num_other_rooms * gallons_per_room) / white_paint_cans_per_gallon)
    (h10 : total_paint_cans = color_paint_cans_needed + white_paint_cans_needed) :
    total_paint_cans = 10 :=
by sorry

end paint_cans_needed_l158_158509


namespace sum_fractions_to_approximation_l158_158793

theorem sum_fractions_to_approximation :
  let S := ∑ n in Finset.range 2010, (3 / ((n + 1) * ((n + 1) + 3))) in
  abs (S - 1.832) < 0.001 :=
by sorry

end sum_fractions_to_approximation_l158_158793


namespace age_of_older_friend_l158_158008

theorem age_of_older_friend (a b : ℕ) (h1 : a - b = 2) (h2 : a + b = 74) : a = 38 :=
by
  sorry

end age_of_older_friend_l158_158008


namespace intersection_M_N_l158_158636

open Real

def M := {x : ℝ | x^2 - 2 * x - 3 ≤ 0}
def N := {x : ℝ | 2 - abs x > 0}

theorem intersection_M_N :
  M ∩ N = {x : ℝ | -1 ≤ x ∧ x < 2} := by
sorry

end intersection_M_N_l158_158636


namespace area_ratio_proof_l158_158313

open Real

noncomputable def area_ratio (FE AF DE CD ABCE : ℝ) :=
  (AF = 3 * FE) ∧ (CD = 3 * DE) ∧ (ABCE = 16 * FE^2) →
  (10 * FE^2 / ABCE = (5 / 8))

theorem area_ratio_proof (FE AF DE CD ABCE : ℝ) :
  AF = 3 * FE → CD = 3 * DE → ABCE = 16 * FE^2 →
  10 * FE^2 / ABCE = 5 / 8 :=
by
  intro hAF hCD hABCE
  sorry

end area_ratio_proof_l158_158313


namespace ways_to_climb_four_steps_l158_158803

theorem ways_to_climb_four_steps (ways_to_climb : ℕ → ℕ) 
  (h1 : ways_to_climb 1 = 1) 
  (h2 : ways_to_climb 2 = 2) 
  (h3 : ways_to_climb 3 = 3) 
  (h_step : ∀ n, ways_to_climb n = ways_to_climb (n - 1) + ways_to_climb (n - 2)) : 
  ways_to_climb 4 = 5 := 
sorry

end ways_to_climb_four_steps_l158_158803


namespace min_value_trig_l158_158209

theorem min_value_trig (x y z : ℝ) (hx : 0 ≤ x) (hx2 : x ≤ π/2)
                       (hy : 0 ≤ y) (hy2 : y ≤ π/2)
                       (hz : 0 ≤ z) (hz2 : z ≤ π/2) :
  ∃ A, A = cos (x - y) + cos (y - z) + cos (z - x) ∧ A ≥ 1 :=
sorry

end min_value_trig_l158_158209


namespace prod_of_three_consec_ints_l158_158439

theorem prod_of_three_consec_ints (a : ℤ) (h : a + (a + 1) + (a + 2) = 27) :
  a * (a + 1) * (a + 2) = 720 :=
by
  sorry

end prod_of_three_consec_ints_l158_158439


namespace power_division_l158_158877

theorem power_division : (19^11 / 19^6 = 247609) := sorry

end power_division_l158_158877


namespace population_of_metropolitan_county_l158_158654

theorem population_of_metropolitan_county : 
  let average_population := 5500
  let two_populous_cities_population := 2 * average_population
  let remaining_cities := 25 - 2
  let remaining_population := remaining_cities * average_population
  let total_population := (2 * two_populous_cities_population) + remaining_population
  total_population = 148500 := by
sorry

end population_of_metropolitan_county_l158_158654


namespace arccos_one_over_sqrt_two_l158_158946

theorem arccos_one_over_sqrt_two : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_l158_158946


namespace period_tan_add_csc_l158_158046

theorem period_tan_add_csc (x : ℝ) : (∀ x, tan (x + π) = tan x) ∧ (∀ x, csc (x + 2 * π) = csc x) 
→ ∃ p > 0, ∀ x, (tan x + csc x) = (tan (x + p) + csc (x + p)) := 
by
  sorry

end period_tan_add_csc_l158_158046


namespace exponent_addition_l158_158557

theorem exponent_addition (m n : ℝ) (h1 : 3^m = 2) (h2 : 3^n = 3) : 3^(m + n) = 6 := by
  sorry

end exponent_addition_l158_158557


namespace cos_trig_identity_l158_158998

theorem cos_trig_identity :
  cos (35 * (Real.pi / 180)) * cos (25 * (Real.pi / 180)) - 
  sin (145 * (Real.pi / 180)) * cos (65 * (Real.pi / 180)) = 
  1 / 2 :=
by
  -- proof goes here
  sorry

end cos_trig_identity_l158_158998


namespace similarity_set_count_l158_158297

variable (AB A1B1 BC B1C1 AC A1C1 : ℝ)
variable (A A1 B B1 C C1 : ℝ)

def are_similar_conditions (cond1 cond2 : Prop) : Prop :=
  cond1 ∧ cond2 ∧ ((cond1 = (AB / A1B1 = BC / B1C1) ∧ cond2 = (BC / B1C1 = AC / A1C1)) ∨
                   (cond1 = (AB / A1B1 = BC / B1C1) ∧ cond2 = (B = B1)) ∨
                   (cond1 = (BC / B1C1 = AC / A1C1) ∧ cond2 = (C = C1)) ∨
                   (cond1 = (A = A1) ∧ (cond2 = (B = B1) ∨ cond2 = (C = C1))) ∨
                   (cond1 = (B = B1) ∧ cond2 = (C = C1)))

def count_similarity_sets : Nat :=
  if   are_similar_conditions (AB / A1B1 = BC / B1C1) (BC / B1C1 = AC / A1C1)
  then 6 -- 1 (SSS) + 2 (SAS) + 3 (AA)
  else 0

theorem similarity_set_count : count_similarity_sets = 6 :=
  sorry

end similarity_set_count_l158_158297


namespace floor_S_squared_l158_158690

noncomputable def S : ℝ := ∑ i in Finset.range 1003, real.sqrt (1 + 1/(i+1)^2 + 1/(i+2)^2)

theorem floor_S_squared : ⌊S^2⌋ = 1008014 := 
  by sorry

end floor_S_squared_l158_158690


namespace arccos_one_over_sqrt_two_l158_158951

theorem arccos_one_over_sqrt_two : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_l158_158951


namespace integer_valued_polynomial_representation_l158_158720

theorem integer_valued_polynomial_representation
  (P : ℤ → ℤ)
  (A : ℕ → ℤ)
  (n : ℕ)
  (hP : ∀ x : ℤ, P x = x^n + A 1 * x^(n-1) + A 2 * x^(n-2) + ... + A (n-1) * x + A n) :
  ∃ b : ℕ → ℤ,
    ∀ x : ℤ,
      P x = b 0 * 1 + b 1 * x + b 2 * (x * (x - 1) / 2) + ...
            + b n * (x * (x - 1) * ... * (x - n + 1) / n!) :=
sorry

end integer_valued_polynomial_representation_l158_158720


namespace increasing_on_interval_l158_158169

def f (x : ℝ) : ℝ := x / (x + 2)

theorem increasing_on_interval : ∀ (x₁ x₂ : ℝ), x₁ < x₂ ∧ x₂ < -2 → f x₁ < f x₂ :=
by
  intros x₁ x₂ h
  let hx₁ := h.1
  let hx₂ := h.2
  calc
    f x₁ = x₁ / (x₁ + 2)          : rfl
    ...    < x₂ / (x₂ + 2)          : sorry   -- Proof needs to be completed.

end increasing_on_interval_l158_158169


namespace function_specifications_l158_158388

noncomputable def f : ℝ → ℝ := λ x => 2 * sin (2 * x - π / 3)

def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

def is_symmetric_about (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (2 * a - x) = f x

theorem function_specifications :
  (is_periodic f π) ∧ (is_symmetric_about f (π / 3)) :=
by
  sorry

end function_specifications_l158_158388


namespace z_diff_modulus_eq_one_l158_158161

theorem z_diff_modulus_eq_one : 
  ∀ (z : ℕ → ℂ),
    (∀ n : ℕ, z n = (1 + complex.I) * (Π k in finset.range n, (1 + complex.I / real.sqrt (k + 1)))) →
    ∣ z 2014 - z 2015 ∣ = 1 := by
  sorry

end z_diff_modulus_eq_one_l158_158161


namespace irrational_sqrt_3_l158_158446

theorem irrational_sqrt_3 : irrational (real.sqrt 3) :=
sorry

end irrational_sqrt_3_l158_158446


namespace intersection_points_lie_on_incircle_l158_158176

open_locale classical

variables {A B C D M N P Q : Point} 
variables {BD : Line} {excircle_ABD : Circle}

-- Assume the existence of parallelogram ABCD
variable (parallelogram_ABCD : parallelogram A B C D)

-- Assume an excircle of triangle ABD touches the extensions of AD and AB at points M and N
variable (excircle_touches_AD_at_M : excircle_ABD.tangent AD M)
variable (excircle_touches_AB_at_N : excircle_ABD.tangent AB N)

-- Assume points P and Q are the intersections of MN with BC and CD respectively
variable (P_intersect_MN_BC : intersect_line_segments MN BC P)
variable (Q_intersect_MN_CD : intersect_line_segments MN CD Q)

-- Define our theorem
theorem intersection_points_lie_on_incircle 
  (incircle_BCD : Circle)
  (incircle_BCD_tangent_BC : incircle_BCD.tangent BC) 
  (incircle_BCD_tangent_CD : incircle_BCD.tangent CD) 
  (incircle_BCD_tangent_BD : incircle_BCD.tangent BD) :
  incircle_BCD.on_circle P ∧ incircle_BCD.on_circle Q :=
sorry

end intersection_points_lie_on_incircle_l158_158176


namespace prop_sufficient_not_necessary_l158_158680

-- Let p and q be simple propositions.
variables (p q : Prop)

-- Define the statement to be proved: 
-- "either p or q is false" is a sufficient but not necessary condition 
-- for "not p is true".
theorem prop_sufficient_not_necessary (hpq : ¬(p ∧ q)) : ¬ p :=
sorry

end prop_sufficient_not_necessary_l158_158680


namespace arccos_one_over_sqrt_two_l158_158948

theorem arccos_one_over_sqrt_two : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_l158_158948


namespace tangent_circles_t_value_l158_158238

theorem tangent_circles_t_value (t : ℝ) :
  (∀ (x y : ℝ), x^2 + y^2 = t^2 → x^2 + y^2 + 6 * x - 8 * y + 24 = 0 → dist (0, 0) (-3, 4) = t + 1) → t = 4 :=
by
  sorry

end tangent_circles_t_value_l158_158238


namespace power_of_7_l158_158147

theorem power_of_7 (k : Nat) : 
  ∃ k, (∏ (factors : Nat) in [4^11, 7^k, 11^2], prime_factors_count factors) = 29 → k = 5 :=
by
  -- Definitions to convert given conditions to Lean 4 format
  let expression := 4^11 * 7^k * 11^2
  have h1 : prime_factors_count (4^11) = 22 := sorry
  have h2 : prime_factors_count (11^2) = 2 := sorry
  have h3 : prime_factors_count (expression) = 29 := sorry
  -- Prove that k must be 5
  use 5
  sorry

end power_of_7_l158_158147


namespace largest_n_is_107_l158_158129

noncomputable def largest_n : ℕ :=
  classical.some (exists_unique (λ n : ℕ, ∀ (a1 a2 a3 a4 a5 a6 : ℕ), 
    a1 ≠ a2 ∧ a1 ≠ a3 ∧ a1 ≠ a4 ∧ a1 ≠ a5 ∧ a1 ≠ a6 ∧
    a2 ≠ a3 ∧ a2 ≠ a4 ∧ a2 ≠ a5 ∧ a2 ≠ a6 ∧
    a3 ≠ a4 ∧ a3 ≠ a5 ∧ a3 ≠ a6 ∧
    a4 ≠ a5 ∧ a4 ≠ a6 ∧
    a5 ≠ a6 ∧
    a1 ≤ n ∧ a2 ≤ n ∧ a3 ≤ n ∧ a4 ≤ n ∧ a5 ≤ n ∧ a6 ≤ n →
    ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ a * b > c ∧ b * c > a ∧ c * a > b))

theorem largest_n_is_107 : largest_n = 107 :=
sorry

end largest_n_is_107_l158_158129


namespace average_visitors_per_day_l158_158834

theorem average_visitors_per_day (avg_visitors_Sunday : ℕ) (avg_visitors_other_days : ℕ) (total_days : ℕ) (starts_on_Sunday : Bool) :
  avg_visitors_Sunday = 500 → 
  avg_visitors_other_days = 140 → 
  total_days = 30 → 
  starts_on_Sunday = true → 
  (4 * avg_visitors_Sunday + 26 * avg_visitors_other_days) / total_days = 188 :=
by
  intros h1 h2 h3 h4
  sorry

end average_visitors_per_day_l158_158834


namespace arccos_one_over_sqrt_two_eq_pi_over_four_l158_158896

theorem arccos_one_over_sqrt_two_eq_pi_over_four : 
  Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_eq_pi_over_four_l158_158896


namespace measure_angle_BED_l158_158298

/-- In triangle ABC, with ∠A = 45°, ∠C = 65°, points D and E are on sides 
AB and BC respectively. If DE = BC, then the measure of ∠BED is 35°. -/
theorem measure_angle_BED {A B C D E : Type} [angle A B C] [angle A = 45°] 
  [angle C = 65°] (hD : D ∈ line_segment A B) (hE : E ∈ line_segment B C) 
  (hDE : distance D E = distance B C) : angle D B E = 35° := 
by
  sorry

end measure_angle_BED_l158_158298


namespace lakers_win_sequences_count_l158_158655

-- Definitions and conditions
def best_of_seven_series_wins (games : List (Bool)) : Bool :=
  games.filter id.length = 4

def lakers_win_series (games : List (Bool)) : Bool :=
  games.filter id.length = 4 ∧ games.length ≤ 7

-- Statement of the theorem
theorem lakers_win_sequences_count : ∃ (count : Nat), 
  count = 30 ∧ 
  (∀ (games : List (Bool)), lakers_win_series games → best_of_seven_series_wins games) :=
sorry

end lakers_win_sequences_count_l158_158655


namespace problem1_l158_158468

theorem problem1 (n : ℕ) (hn : 0 < n) : 20 ∣ (4 * 6^n + 5^(n+1) - 9) := 
  sorry

end problem1_l158_158468


namespace min_value_expression_max_value_expression_l158_158577

variable {a b : ℝ} (ha : 0 < a) (hb : 0 < b) (hab : a < b)
variable (x1 x2 x3 x4 : ℝ) (hx1 : a ≤ x1) (hx1b : x1 ≤ b) 
  (hx2 : a ≤ x2) (hx2b : x2 ≤ b) 
  (hx3 : a ≤ x3) (hx3b : x3 ≤ b) 
  (hx4 : a ≤ x4) (hx4b : x4 ≤ b)

noncomputable def expression :=
  (x1^2 / x2 + x2^2 / x3 + x3^2 / x4 + x4^2 / x1) / (x1 + x2 + x3 + x4)
  
theorem min_value_expression : 
  ∃ x1 x2 x3 x4, expression ha hb hab x1 hx1 hx1b x2 hx2 hx2b x3 hx3 hx3b x4 hx4 hx4b = 1 :=
sorry

theorem max_value_expression : 
  ∃ x1 x2 x3 x4, expression ha hb hab x1 hx1 hx1b x2 hx2 hx2b x3 hx3 hx3b x4 hx4 hx4b = (b/a + a/b - 1) :=
sorry

end min_value_expression_max_value_expression_l158_158577
