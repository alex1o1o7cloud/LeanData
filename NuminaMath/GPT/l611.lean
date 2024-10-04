import Mathlib
import Mathlib.Algebra.Algebra.Basic
import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.BigOperators.Order
import Mathlib.Algebra.Factorial
import Mathlib.Algebra.Field
import Mathlib.Algebra.Group.Default
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Module.Basic
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.MeanInequalities
import Mathlib.Analysis.NormedSpace.Basic
import Mathlib.Analysis.Probability
import Mathlib.Analysis.SpecialFunctions.Exponential
import Mathlib.Analysis.SpecialFunctions.Integrals
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Lcm
import Mathlib.Data.Nat.Parity
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Logic.Basic
import Mathlib.Probability
import Mathlib.Probability.Independence
import Mathlib.Probability.Independent
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic

namespace evaluate_expression_l611_611955

theorem evaluate_expression : (4 - 3) * 2 = 2 := by
  sorry

end evaluate_expression_l611_611955


namespace find_am_2n_l611_611016

-- Definition of the conditions
variables {a : ℝ} {m n : ℝ}
axiom am_eq_5 : a ^ m = 5
axiom an_eq_2 : a ^ n = 2

-- The statement we want to prove
theorem find_am_2n : a ^ (m - 2 * n) = 5 / 4 :=
by {
  sorry
}

end find_am_2n_l611_611016


namespace find_m_range_l611_611685

def A : Set ℝ := { x | -3 ≤ x ∧ x ≤ 4 }
def B (m : ℝ) : Set ℝ := { x | -1 ≤ x ∧ x ≤ m + 1 }

theorem find_m_range (m : ℝ) : (B m ⊆ A) ↔ (-2 ≤ m ∧ m ≤ 3) := by
  sorry

end find_m_range_l611_611685


namespace composite_function_value_l611_611079

theorem composite_function_value 
  (f : ℝ → ℝ) (a b c d k : ℝ) 
  (g : ℝ → ℝ := λ x, a * x^3 + b * x^2 + c * x + d)
  (h : ℝ → ℝ := f ∘ g) 
  (H : h (2 * k) = 3 * k) : Prop :=
  h (2 * k) = 3 * k

end composite_function_value_l611_611079


namespace problem_1_problem_2_l611_611379

variables (a b : ℝ^3)
variables (theta : ℝ)

-- Conditions
def condition_1 := ‖a‖ = Real.sqrt 2
def condition_2 := ‖b‖ = 1
def angle_theta := θ = Real.pi / 4

-- Problem (1) statement
theorem problem_1 : condition_1 ∧ condition_2 ∧ angle_theta → ‖a + b‖ = Real.sqrt 5 :=
by
  sorry

-- Problem (2) statement
theorem problem_2 : condition_1 ∧ condition_2 ∧ (a - b) • b = 0 → theta = Real.pi / 4 :=
by
  sorry

end problem_1_problem_2_l611_611379


namespace fishing_tomorrow_l611_611826

theorem fishing_tomorrow (yesterday_fishers today_fishers : ℕ)
  (every_day_fishers every_other_day_fishers every_three_days_fishers : ℕ)
  (total_population : ℕ):
  yesterday_fishers = 12 → 
  today_fishers = 10 → 
  every_day_fishers = 7 → 
  every_other_day_fishers = 8 → 
  every_three_days_fishers = 3 → 
  total_population = yesterday_fishers + today_fishers + (total_population - (every_day_fishers + every_other_day_fishers + every_three_days_fishers)) →
  ∃ tomorrow_fishers : ℕ, tomorrow_fishers = 15 :=
by {
  -- This is a statement definition, the proof is not required and thus marked as "sorry:"
  sorry
}

end fishing_tomorrow_l611_611826


namespace grasshopper_lands_in_hole_l611_611838

/-- 
Prove that a grasshopper can land in a circular hole in a meadow with the shape of a square. The grasshopper jumps in the direction of a chosen vertex, with each jump length being half the distance to that vertex.
-/
theorem grasshopper_lands_in_hole
  (meadow : Type*) [fintype meadow]
  (cell_size : ℕ → ℝ)
  (hole : meadow)
  (jump : meadow → meadow → ℝ)
  (base_case : cell_size 0 = 1)
  (inductive_hypothesis : ∀ n, ∃ cell, cell_size n = (1 / 2)^n)
  (homothety : ∀ n cell vertex, jump cell vertex = (cell_size n) / 2) :
  ∃ (n : ℕ), True :=
by
  sorry

end grasshopper_lands_in_hole_l611_611838


namespace minimum_oranges_l611_611120

theorem minimum_oranges (n : ℕ) (m : ℕ → ℝ) (h : ∀ i j k : ℕ, i < n → j < n → k < n → i ≠ j → i ≠ k → j ≠ k → (m i + m j + m k) < 0.05 * ∑ l in Finset.range n \ {i, j, k}, m l) : n ≥ 64 := 
sorry

end minimum_oranges_l611_611120


namespace determine_value_of_x_l611_611463

theorem determine_value_of_x (x y z : ℤ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxy : x ≥ y) (hyz : y ≥ z)
  (h1 : x^2 - y^2 - z^2 + x * y = 4033) 
  (h2 : x^2 + 4 * y^2 + 4 * z^2 - 4 * x * y - 3 * x * z - 3 * y * z = -3995) : 
  x = 69 := sorry

end determine_value_of_x_l611_611463


namespace puppies_per_cage_l611_611243

theorem puppies_per_cage
  (initial_puppies : ℕ)
  (sold_puppies : ℕ)
  (remaining_puppies : ℕ)
  (cages : ℕ)
  (puppies_per_cage : ℕ)
  (h1 : initial_puppies = 78)
  (h2 : sold_puppies = 30)
  (h3 : remaining_puppies = initial_puppies - sold_puppies)
  (h4 : cages = 6)
  (h5 : puppies_per_cage = remaining_puppies / cages) :
  puppies_per_cage = 8 := by
  sorry

end puppies_per_cage_l611_611243


namespace smallest_square_value_l611_611472

theorem smallest_square_value (a b : ℕ) (ha : 0 < a) (hb : 0 < b)
  (r s : ℕ) (hr : 15 * a + 16 * b = r^2) (hs : 16 * a - 15 * b = s^2) :
  min (r^2) (s^2) = 481^2 :=
  sorry

end smallest_square_value_l611_611472


namespace nilpotent_matrix_squared_zero_l611_611068

variable {R : Type*} [Field R]
variable (A : Matrix (Fin 2) (Fin 2) R)

theorem nilpotent_matrix_squared_zero (h : A^4 = 0) : A^2 = 0 := 
sorry

end nilpotent_matrix_squared_zero_l611_611068


namespace min_value_inverse_sum_l611_611351

theorem min_value_inverse_sum (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + 2 * b = 2) :
  (1 / a + 2 / b) ≥ 9 / 2 :=
sorry

end min_value_inverse_sum_l611_611351


namespace incorrect_statement_l611_611398

noncomputable def prove_incorrect_statement (a b c : Line) (α β : Plane) : Prop :=
  (∀ c α β : Line, c ⊥ α → c ⊥ β → α ∥ β) ∧
  (∀ b α β : Line, b ⊂ α → b ⊥ β → α ⊥ β) ∧
  (∀ b a c α : Line, b ⊂ α → ¬(a ⊂ α) → c = projection_of α a → b ⊥ c → a ⊥ b) ∧
  ¬(∀ b c α : Line, b ⊂ α → ¬(c ⊂ α) → c ∥ α → b ∥ c)

theorem incorrect_statement :
  ∀ (a b c : Line) (α β : Plane), prove_incorrect_statement a b c α β :=
by
  intros a b c α β
  apply and.intro
  { intros c α β hcα hcβ,
    -- proof for statement A
    sorry } -- statement A is correct
  apply and.intro
  { intros b α β hbα hbβ,
    -- proof for statement B
    sorry } -- statement B is correct
  apply and.intro
  { intros b a c α hbα hnaα hpac hbc,
    -- proof for statement C
    sorry } -- statement C is correct
  { intros b c α hbα hncα hpcα,
    -- proof to show statement D is incorrect
    sorry } -- statement D is incorrect

end incorrect_statement_l611_611398


namespace fishing_tomorrow_l611_611828

-- Conditions
def every_day_fishers : Nat := 7
def every_other_day_fishers : Nat := 8
def every_three_days_fishers : Nat := 3
def yesterday_fishers : Nat := 12
def today_fishers : Nat := 10

-- Determine the number who will fish tomorrow
def fishers_tomorrow : Nat :=
  let every_day_tomorrow := every_day_fishers
  let every_three_day_tomorrow := every_three_days_fishers
  let every_other_day_yesterday := yesterday_fishers - every_day_fishers
  let every_other_day_tomorrow := every_other_day_fishers - every_other_day_yesterday
  every_day_tomorrow + every_three_day_tomorrow + every_other_day_tomorrow

theorem fishing_tomorrow : fishers_tomorrow = 15 :=
  by
    -- skipping the actual proof with sorry
    sorry

end fishing_tomorrow_l611_611828


namespace Hellys_theorem_all_n_convex_have_common_point_l611_611985

theorem Hellys_theorem_all_n_convex_have_common_point (n : ℕ) (M : fin n → set ℝ^2)
  (H_convex : ∀ i : fin n, convex (M i))
  (H_common_point : ∀ (i j k : fin n), i ≠ j → j ≠ k → i ≠ k → (M i ∩ M j ∩ M k).nonempty) :
  (⋂ i, M i).nonempty :=
sorry

end Hellys_theorem_all_n_convex_have_common_point_l611_611985


namespace fishing_tomorrow_l611_611817

theorem fishing_tomorrow (seven_every_day eight_every_other_day three_every_three_days twelve_yesterday ten_today : ℕ)
  (h1 : seven_every_day = 7)
  (h2 : eight_every_other_day = 8)
  (h3 : three_every_three_days = 3)
  (h4 : twelve_yesterday = 12)
  (h5 : ten_today = 10) :
  (seven_every_day + (eight_every_other_day - (twelve_yesterday - seven_every_day)) + three_every_three_days) = 15 :=
by
  sorry

end fishing_tomorrow_l611_611817


namespace eventual_495_or_0_l611_611545

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 100 + ((n / 10) % 10) * 10 + (n / 100)

def iterate_reverse_subtract (n : ℕ) : ℕ :=
  let r := reverse_digits n
  in if n >= r then n - r else r - n

theorem eventual_495_or_0 (n : ℕ) (h : 100 ≤ n ∧ n < 1000) : 
  ∃ k : ℕ, (iterate k iterate_reverse_subtract n = 495 ∨ iterate k iterate_reverse_subtract n = 0) :=
sorry

end eventual_495_or_0_l611_611545


namespace fraction_sum_eq_decimal_l611_611998

theorem fraction_sum_eq_decimal : (2 / 5) + (2 / 50) + (2 / 500) = 0.444 := by
  sorry

end fraction_sum_eq_decimal_l611_611998


namespace sin_product_ratio_l611_611587

theorem sin_product_ratio (a b : ℝ) :
  (a = (finset.range 89).prod (λ k, real.sin (k + 1) * real.sin (if k % 2 = 1 then k + 1 else 1)))
  ∧ (b = (finset.range 45).prod (λ k, real.sin (2 * k + 1))) → 
  a / b = (finset.range 44).prod (λ k, real.sin (2 * (k + 1))) :=
by
  intro h
  sorry

end sin_product_ratio_l611_611587


namespace min_value_complex_mod_one_l611_611712

/-- Given that the modulus of the complex number \( z \) is 1, prove that the minimum value of
    \( |z - 4|^2 + |z + 3 * Complex.I|^2 \) is \( 17 \). -/
theorem min_value_complex_mod_one (z : ℂ) (h : ‖z‖ = 1) : 
  ∃ α : ℝ, (‖z - 4‖^2 + ‖z + 3 * Complex.I‖^2) = 17 :=
sorry

end min_value_complex_mod_one_l611_611712


namespace eccentricity_of_C_l611_611855

theorem eccentricity_of_C 
  (C : Type) 
  (curve_C : C → ℝ × ℝ) 
  (phi : ℝ × ℝ → ℝ × ℝ)
  (phi_def : ∀ (x y : ℝ), phi (x, y) = (1/3 * x, 1/2 * y))
  (curve_transformation : ∀ (x y : ℝ), curve_C (x, y) -> ((1/3 * x)^2 + (1/2 * y)^2 = 1))
  : ∃ e : ℝ, e = (sqrt 5) / 3 :=
by
  sorry

end eccentricity_of_C_l611_611855


namespace fishers_tomorrow_l611_611793

-- Definitions based on conditions
def people_every_day : ℕ := 7
def people_every_other_day : ℕ := 8
def people_every_three_days : ℕ := 3
def people_yesterday : ℕ := 12
def people_today : ℕ := 10

-- Theorem to be proved
theorem fishers_tomorrow (people_every_day people_every_other_day people_every_three_days people_yesterday people_today : ℕ) : 
  people_every_day = 7 ∧ 
  people_every_other_day = 8 ∧ 
  people_every_three_days = 3 ∧
  people_yesterday = 12 ∧ 
  people_today = 10 →
  15 := 
by {
  sorry
}

end fishers_tomorrow_l611_611793


namespace find_a_l611_611343

-- Definitions of sets A and B based on the variable a
def A (a : ℝ) : Set ℝ := {a^2, a+1, -3}
def B (a : ℝ) : Set ℝ := {a-3, 2a + 1, a^2 + 3}

-- Given condition
theorem find_a (a : ℝ) (h : A a ∩ B a = {-3}) : a = -2 := by
  sorry

end find_a_l611_611343


namespace find_extrema_l611_611676

noncomputable def f (x : ℝ) : ℝ := -x^2 - 2 * x + 3

theorem find_extrema :
  ∃ (xmin xmax : ℝ) (xmin_pt xmax_pt : ℝ),
    xmin_pt ∈ Icc (-2 : ℝ) 1 ∧ xmax_pt ∈ Icc (-2 : ℝ) 1 ∧
    (f(xmin_pt) = 0) ∧ (f(xmax_pt) = 4) ∧
    (∀ x ∈ Icc (-2 : ℝ) 1, xmin ≤ f(x) ∧ f(x) ≤ xmax) :=
sorry

end find_extrema_l611_611676


namespace sum_of_quadratic_term_l611_611326

theorem sum_of_quadratic_term (x a_n : ℕ) (n : ℕ) (a : Fin (n + 1) → ℕ) :
  (x + 1) ^ n = ∑ i in Finset.range (n + 1), a i * (x - 1) ^ i →
  ∑ i in Finset.range (n + 1), a i = 243 →
  n = 5 →
  (let b : Fin 6 → ℕ := fun _ => (Nat.choose n 2) * (2 ^ (n - 2 : ℕ)) in 
   b 2 = 32) := sorry

end sum_of_quadratic_term_l611_611326


namespace problem_1_problem_2_l611_611714

def S (n : ℕ) : ℕ := n^2 + 2 * n

def a (n : ℕ) : ℕ := if n = 1 then 3 else S n - S (n - 1)
def b (n : ℕ) : ℕ := n
def c (n : ℕ) : ℕ := (a n + 1)^(n + 1) / (b n + 1)^n
def T (n : ℕ) : ℕ := n * 2^(n + 2)

theorem problem_1 (n : ℕ) : b n = n := by
  -- Proof required
  sorry

theorem problem_2 (n : ℕ) : ∑ k in Finset.range n, c k = T n := by
  -- Proof required
  sorry

end problem_1_problem_2_l611_611714


namespace inclination_angle_of_line_l611_611941

open Real

theorem inclination_angle_of_line (x y : ℝ) (h : x + y - 3 = 0) : 
  ∃ θ : ℝ, θ = 3 * π / 4 :=
by
  sorry

end inclination_angle_of_line_l611_611941


namespace probability_two_students_same_pair_l611_611947

theorem probability_two_students_same_pair (students_choose_two_out_of_three_events : 
  ℕ) : 
  let total_outcomes : ℕ := 3 * 3 * 3,
      favorable_outcomes : ℕ := 3 * 3 * 2 in
  students_choose_two_out_of_three_events = 3 →
  favorable_outcomes / total_outcomes = (2 / 3 : ℝ) :=
by
  sorry

end probability_two_students_same_pair_l611_611947


namespace sum_of_all_solutions_l611_611148

/-
Let x be a real number such that 2^(x^2 + 6x + 9) = 4^(x + 3).
Then the sum of all possible values of x equals -4.
-/

noncomputable def is_solution (x : ℝ) : Prop := 2^(x^2 + 6*x + 9) = 4^(x + 3)

theorem sum_of_all_solutions : (∑ x in {x : ℝ | is_solution x}, x) = -4 := by
  sorry

end sum_of_all_solutions_l611_611148


namespace cyclic_quad_side_CD_l611_611264

theorem cyclic_quad_side_CD (A B C D : Type)
  (h1 : Quadrilateral A B C D)
  (h2 : ∃ O : Point, Circle O 2 ∧ Cyclic A B C D)
  (h3 : Perpendicular (Line AC) (Line BD))
  (h4 : Distance A B = 3) :
  Distance C D = sqrt 7 := 
sorry

end cyclic_quad_side_CD_l611_611264


namespace f_odd_f_decreasing_on_01_l611_611936

noncomputable def f (x : ℝ) (hx : x ≠ 0) : ℝ := x + 1 / x

theorem f_odd (x : ℝ) (hx : x ≠ 0) : f (-x) hx = -f x hx :=
by sorry

theorem f_decreasing_on_01 : ∀ x ∈ Ioo (0 : ℝ) 1, deriv (λ x, f x (ne_of_gt (lt_of_lt_of_le zero_lt_one x))) x < 0 :=
by sorry

end f_odd_f_decreasing_on_01_l611_611936


namespace last_digits_divisible_by_8_l611_611092

theorem last_digits_divisible_by_8 : 
  ∃ S : Set ℕ, 
    (∀ n ∈ S, n < 10) ∧ 
    (∀ n ∈ S, ∃ x y z : ℕ, n = z ∧ x * 100 + y * 10 + z % 8 = 0) ∧ 
    S.card = 5 :=
  sorry

end last_digits_divisible_by_8_l611_611092


namespace range_of_a_not_monotonic_l611_611026

def f (a x : ℝ) : ℝ := x^2 + 2*(a - 1)*x + 2

theorem range_of_a_not_monotonic (a : ℝ) :
  ¬ (∀ x y ∈ (Ioo 1 4), (x < y → f a x ≤ f a y) ∨ (x < y → f a x ≥ f a y))
  ↔ (−3 < a ∧ a < 0) :=
sorry

end range_of_a_not_monotonic_l611_611026


namespace valid_numbers_l611_611236

def is_valid_100_digit_number (N N' : ℕ) (k m n : ℕ) (a : ℕ) : Prop :=
  0 ≤ a ∧ a < 100 ∧ 0 ≤ m ∧ m < 10^k ∧ 
  N = m + 10^k * a + 10^(k + 2) * n ∧ 
  N' = m + 10^k * n ∧
  N = 87 * N'

theorem valid_numbers : ∀ (N : ℕ), (∃ N' k m n a, is_valid_100_digit_number N N' k m n a) →
  N = 435 * 10^97 ∨ 
  N = 1305 * 10^96 ∨ 
  N = 2175 * 10^96 ∨ 
  N = 3045 * 10^96 :=
by
  sorry

end valid_numbers_l611_611236


namespace log_10_54883_l611_611540

noncomputable def log_predicate : Prop :=
  ∃ (c d: ℤ), (log 10 10000 = 4) ∧ (log 10 100000 = 5) ∧
              (∀ x y, x < y → log 10 x < log 10 y) ∧
              4 < log 10 54883 ∧ log 10 54883 < 5 ∧
              c + d = 9

theorem log_10_54883 :
  log_predicate :=
by
  sorry

end log_10_54883_l611_611540


namespace minimum_number_of_oranges_l611_611136

noncomputable def minimum_oranges_picked : ℕ :=
  let n : ℕ := 64 in n

theorem minimum_number_of_oranges (n : ℕ) : 
  (∀ (m_i m_j m_k : ℝ) (remaining_masses : Finset ℝ),
    remaining_masses.card = n - 3 →
    (m_i + m_j + m_k) < 0.05 * (∑ m in remaining_masses, m)) →
  n ≥ minimum_oranges_picked :=
by {
  sorry
}

end minimum_number_of_oranges_l611_611136


namespace find_numbers_l611_611674

theorem find_numbers :
  ∃ (x y z : ℕ), 
    x = 80 ∧ 
    y = 100 ∧ 
    z = 90 ∧ 
    x = 0.8 * y ∧ 
    (y : ℚ) / (z : ℚ) = (10 : ℚ) / (9 : ℚ) ∧ 
    x + z = y + 70 :=
by
  sorry

end find_numbers_l611_611674


namespace largest_angle_in_ratio_triangle_l611_611166

theorem largest_angle_in_ratio_triangle (x : ℝ) (h1 : 3 * x + 4 * x + 5 * x = 180) : 
  5 * (180 / (3 + 4 + 5)) = 75 := by
  sorry

end largest_angle_in_ratio_triangle_l611_611166


namespace simplify_exponents_l611_611918

theorem simplify_exponents (x : ℝ) : x^5 * x^3 = x^8 :=
by
  sorry

end simplify_exponents_l611_611918


namespace magnitude_of_conjugate_z_l611_611745

theorem magnitude_of_conjugate_z (a b : ℝ) (h₀: b ≠ 0) 
  (h₁ : (2 + a * Complex.I) / (3 - Complex.I) = b * Complex.I) :
  Complex.abs (Complex.conj (a + b * Complex.I)) = 2 * Real.sqrt 10 :=
by
  sorry

end magnitude_of_conjugate_z_l611_611745


namespace divisor_count_l611_611874

theorem divisor_count (m : ℕ) (h : m = 2^15 * 5^12) :
  let m_squared := m * m
  let num_divisors_m := (15 + 1) * (12 + 1)
  let num_divisors_m_squared := (30 + 1) * (24 + 1)
  let divisors_of_m_squared_less_than_m := (num_divisors_m_squared - 1) / 2
  num_divisors_m_squared - num_divisors_m = 179 :=
by
  subst h
  sorry

end divisor_count_l611_611874


namespace minimum_number_of_oranges_l611_611138

noncomputable def minimum_oranges_picked : ℕ :=
  let n : ℕ := 64 in n

theorem minimum_number_of_oranges (n : ℕ) : 
  (∀ (m_i m_j m_k : ℝ) (remaining_masses : Finset ℝ),
    remaining_masses.card = n - 3 →
    (m_i + m_j + m_k) < 0.05 * (∑ m in remaining_masses, m)) →
  n ≥ minimum_oranges_picked :=
by {
  sorry
}

end minimum_number_of_oranges_l611_611138


namespace centers_intersection_l611_611619

-- Preliminary definitions
def Rectangle := sorry  -- This would be defined in terms of geometry in Lean's library.
def Center (rect: Rectangle) : (ℝ × ℝ) := sorry  -- Given a rectangle, returns the center point.

-- The actual problem condition
def divided_rectangle (R : Rectangle) (R_div : list Rectangle) : Prop :=
  sorry  -- Definition of a rectangle divided into smaller rectangles.

-- The formal problem in a Lean theorem statement
theorem centers_intersection
  (R : Rectangle)
  (R_div : list Rectangle)
  (Hdiv : divided_rectangle R R_div) :
  ∀ r1 r2 ∈ R_div, r1 ≠ r2 → 
  ∃ r3 ∈ R_div, 
    r3 ≠ r1 ∧ r3 ≠ r2 ∧ (Center r1, Center r2) intersects r3 :=
sorry

end centers_intersection_l611_611619


namespace chinese_lunar_year_2032_is_ren_zi_l611_611153

-- Define the Heavenly Stems and Earthly Branches as lists
def heavenlyStems : List String := ["Jia", "Yi", "Bing", "Ding", "Wu", "Ji", "Geng", "Xin", "Ren", "Gui"]
def earthlyBranches : List String := ["Zi", "Chou", "Yin", "Mao", "Chen", "Si", "Wu", "Wei", "Shen", "You", "Xu", "Hai"]

-- Define the function to get the Chinese lunar year given a Gregorian year
def chineseLunarYear (gregorianYear : Nat) : String :=
  let startYear := 1984
  let yearDiff := gregorianYear - startYear
  let heavenlyStemIndex := yearDiff % 10
  let earthlyBranchIndex := yearDiff % 12
  heavenlyStems[heavenlyStemIndex] ++ " " ++ earthlyBranches[earthlyBranchIndex]

-- State the theorem
theorem chinese_lunar_year_2032_is_ren_zi : chineseLunarYear 2032 = "Ren Zi" :=
sorry

end chinese_lunar_year_2032_is_ren_zi_l611_611153


namespace smallest_d_l611_611491

-- Constants and conditions
variables (c d : ℝ)
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Conditions involving c and d
def conditions (c d : ℝ) : Prop :=
  2 < c ∧ c < d ∧ ¬triangle_inequality 2 c d ∧ ¬triangle_inequality (1/d) (1/c) 2

-- Goal statement: the smallest possible value of d
theorem smallest_d (c d : ℝ) (h : conditions c d) : d = 2 + Real.sqrt 2 :=
sorry

end smallest_d_l611_611491


namespace value_sq_dist_OP_OQ_l611_611345

-- Definitions from problem conditions
def origin : ℝ × ℝ := (0, 0)
def ellipse (x y : ℝ) : Prop := (x^2) / 4 + (y^2) / 3 = 1
def perpendicular (p q : ℝ × ℝ) : Prop := p.1 * q.1 + p.2 * q.2 = 0

-- The proof statement
theorem value_sq_dist_OP_OQ 
  (P Q : ℝ × ℝ) 
  (hP : ellipse P.1 P.2) 
  (hQ : ellipse Q.1 Q.2) 
  (h_perp : perpendicular P Q)
  : (P.1^2 + P.2^2) + (Q.1^2 + Q.2^2) = 48 / 7 := 
sorry

end value_sq_dist_OP_OQ_l611_611345


namespace fishing_problem_l611_611786

theorem fishing_problem
  (everyday : ℕ)
  (every_other_day : ℕ)
  (every_three_days : ℕ)
  (yesterday_fishing : ℕ)
  (today_fishing : ℕ)
  (h_everyday : everyday = 7)
  (h_every_other_day : every_other_day = 8)
  (h_every_three_days : every_three_days = 3)
  (h_yesterday_fishing : yesterday_fishing = 12)
  (h_today_fishing : today_fishing = 10) :
  (every_three_days + everyday + (every_other_day - (yesterday_fishing - everyday))) = 15 := by
  sorry

end fishing_problem_l611_611786


namespace number_subtracted_l611_611249

-- Define the variables x and y
variable (x y : ℝ)

-- Define the conditions
def condition1 := 6 * x - y = 102
def condition2 := x = 40

-- Define the theorem to prove
theorem number_subtracted (h1 : condition1 x y) (h2 : condition2 x) : y = 138 :=
sorry

end number_subtracted_l611_611249


namespace wire_cut_l611_611593

theorem wire_cut (x : ℝ) (h1 : x + (100 - x) = 100) (h2 : x = (7/13) * (100 - x)) : x = 35 :=
sorry

end wire_cut_l611_611593


namespace solution_set_of_inequality_l611_611952

theorem solution_set_of_inequality :
  {x : ℝ | 4 * x ^ 2 - 4 * x + 1 ≤ 0} = {1 / 2} :=
by
  sorry

end solution_set_of_inequality_l611_611952


namespace jack_distance_from_start_l611_611383

theorem jack_distance_from_start (race_length : ℕ) (harper_ran : race_length = 1000) 
  (harper_jack_distance_apart : ℕ) (dist_between : harper_jack_distance_apart = 848) : 
  ∃ jack_distance_from_start : ℕ, jack_distance_from_start = 152 :=
by
  use 1000 - 848
  simp
  sorry

end jack_distance_from_start_l611_611383


namespace count_valid_N_l611_611389

-- Defining the problem conditions in Lean
def is_valid_N (N : ℕ) : Prop :=
  3000 ≤ N ∧ N < 8000 ∧ N % 4 = 0 ∧
  let b := (N % 10000) / 1000 in
  let c := (N % 1000) / 100 in
  2 ≤ b ∧ b ≤ c ∧ c < 7

-- Claim: There are exactly 225 such numbers
theorem count_valid_N : {N : ℕ | is_valid_N N}.finite.toFinset.card = 225 := 
by
  sorry

end count_valid_N_l611_611389


namespace minimum_at_neg_one_l611_611360

noncomputable def f (x : Real) : Real := x * Real.exp x

theorem minimum_at_neg_one : 
  ∃ c : Real, c = -1 ∧ ∀ x : Real, f c ≤ f x := sorry

end minimum_at_neg_one_l611_611360


namespace diameter_eq_4_l611_611209

theorem diameter_eq_4 (π : ℝ) (d r : ℝ) (h1 : π * d = π * r^2) (h2 : d = 2 * r) : d = 4 :=
by
  sorry

end diameter_eq_4_l611_611209


namespace water_pumping_problem_l611_611508

theorem water_pumping_problem :
  let pumpA_rate := 300 -- gallons per hour
  let pumpB_rate := 500 -- gallons per hour
  let combined_rate := pumpA_rate + pumpB_rate -- Combined rate per hour
  let time_duration := 1 / 2 -- Time in hours (30 minutes)
  combined_rate * time_duration = 400 := -- Total volume in gallons
by
  -- Lean proof would go here
  sorry

end water_pumping_problem_l611_611508


namespace evaluate_expression_l611_611658

theorem evaluate_expression : (1 / 16 : ℝ) ^ (-1 / 2) = 4 := 
by 
  have h1 : (1 / (16 : ℝ)) ^ (-1 / 2) = (16 : ℝ) ^ (1 / 2) := by
    exact Real.rpow_neg_one_div h2
    sorry
  have h2 : (16 : ℝ) ^ (1 / 2) = Real.sqrt 16 := by
    exact Real.rpow_eq_sqrt
    sorry
  have h3 : Real.sqrt 16 = 4 := by
    exact Real.sqrt_eq_rpow
    exact Real.sqrt.quadratic_root
    sorry
  sorry

end evaluate_expression_l611_611658


namespace milkman_cows_l611_611599

theorem milkman_cows (x : ℕ) (c : ℕ) :
  (3 * x * c = 720) ∧ (3 * x * c + 50 * c + 140 * c + 63 * c = 3250) → x = 24 :=
by
  sorry

end milkman_cows_l611_611599


namespace ramesh_profit_percentage_l611_611490

noncomputable def purchase_price : ℝ := 16500
noncomputable def transport_cost : ℝ := 125
noncomputable def installation_cost : ℝ := 250
noncomputable def discount_rate : ℝ := 0.20
noncomputable def selling_price : ℝ := 23100

theorem ramesh_profit_percentage :
  let labelled_price := purchase_price / (1 - discount_rate) in
  let total_cost := purchase_price + transport_cost + installation_cost in
  let profit := selling_price - total_cost in
  let profit_percentage := (profit / total_cost) * 100 in
  profit_percentage = 36.89 :=
by
  sorry

end ramesh_profit_percentage_l611_611490


namespace monthly_growth_rate_l611_611032

theorem monthly_growth_rate
  (rev_feb : ℝ := 400)                  -- Revenue in February (in ten thousand yuan)
  (rev_increase_rate : ℝ := 0.1)        -- Revenue increase rate in March (10%)
  (rev_may : ℝ := 633.6)                -- Revenue in May (in ten thousand yuan)
  : ∃ x : ℝ, (1 + x)^2 = (rev_may / (rev_feb * (1 + rev_increase_rate))) :=
by
  have h_rev_march : ℝ := rev_feb * (1 + rev_increase_rate) -- March revenue calculation
  have h_equation : ℝ := rev_may / h_rev_march
  existsi (↑(1.2) - 1 : ℝ) -- The monthly growth rate x = 1.2 - 1 = 0.2 (20%)
  rw [pow_two, (rev_may / (rev_feb * (1 + rev_increase_rate)))],
  sorry

end monthly_growth_rate_l611_611032


namespace real_part_of_z_l611_611328

-- Define the complex number z
def z : ℂ := complex.I * (1 - 2*complex.I)

-- State the theorem
theorem real_part_of_z : z.re = 2 :=
  by {
    -- Sorry is a placeholder for the actual proof
    sorry
  }

end real_part_of_z_l611_611328


namespace sequence_a_n_property_l611_611335

theorem sequence_a_n_property {a : ℕ → ℕ} {S : ℕ → ℕ}
  (h₀ : ∀ n, S n = ∑ i in finset.range (n + 1), a i)
  (h₁ : ∀ n m, S n + S m = S (n + m))
  (h₂ : a 1 = 1) :
  a 100 = 1 :=
sorry

end sequence_a_n_property_l611_611335


namespace bonita_ate_29_peanuts_l611_611959

-- Define initial condition: total peanuts
def total_peanuts : ℕ := 148

-- Define the fraction of peanuts Brock ate
def brock_fraction : ℚ := 1 / 4

-- Calculate number of peanuts Brock ate
def brock_peanuts : ℕ := (total_peanuts : ℚ) * brock_fraction

-- Calculate the remaining peanuts after Brock ate
def remaining_after_brock : ℕ := total_peanuts - brock_peanuts

-- Define the number of peanuts remaining after Bonita ate
def remaining_after_bonita : ℕ := 82

-- Define the number of peanuts Bonita ate
def bonita_peanuts : ℕ := remaining_after_brock - remaining_after_bonita

-- Theorem statement
theorem bonita_ate_29_peanuts : bonita_peanuts = 29 := by
  sorry

end bonita_ate_29_peanuts_l611_611959


namespace line_passes_through_incenter_of_isosceles_triangle_l611_611898

-- Define the isosceles triangle ABC with AB = AC
variables {A B C D E I : Point}
variable [∀ {X Y Z : Point}, Triangle X Y Z]

/-- Define the isosceles triangle -/
def is_isosceles_triangle (A B C : Point) : Prop :=
  A ≠ B ∧ A ≠ C ∧ AB = AC

/-- Define the condition of points D and E on the sides of the triangle -/
def points_on_sides (A B C D E : Point) : Prop :=
  on_line D A B ∧ on_line E A C

/-- Define the incenter I of the triangle ABC -/
def incenter (A B C I : Point) : Prop :=
  is_incenter A B C I

/-- Define the line through points D and E passing through incenter -/
def line_through_incenter (D E I : Point) : Prop :=
  on_line I D E

/-- The theorem statement in Lean 4 -/
theorem line_passes_through_incenter_of_isosceles_triangle
  (A B C D E I : Point)
  (h1 : is_isosceles_triangle A B C)
  (h2 : points_on_sides A B C D E)
  (h3 : incenter A B C I) :
  line_through_incenter D E I :=
sorry

end line_passes_through_incenter_of_isosceles_triangle_l611_611898


namespace inequality_system_solution_l611_611923

theorem inequality_system_solution (x : ℝ) :
  (3 * x > x + 6) ∧ ((1 / 2) * x < -x + 5) ↔ (3 < x) ∧ (x < 10 / 3) :=
by
  sorry

end inequality_system_solution_l611_611923


namespace foci_distance_l611_611520

-- Define the variables and constants
variables (a b : ℝ)
def hyperbola := x^2 / a^2 - y^2 / b^2 = 1

-- The given hyperbola conditions are a = 3 and b = 2
def a_val := 3
def b_val := 2

-- The definition for the distance from the center to each focus
def focal_distance (a b : ℝ) : ℝ :=
  real.sqrt (a^2 + b^2)

-- The full distance between the foci
def distance_between_foci (a b : ℝ) : ℝ :=
  2 * focal_distance a b

-- The theorem that we need to prove
theorem foci_distance :
  distance_between_foci a_val b_val = 2 * real.sqrt 13 :=
sorry

end foci_distance_l611_611520


namespace find_c_l611_611278

def line (x : ℝ) (c : ℝ) : ℝ := 4 * x + c
def point_T (c : ℝ) : (ℝ × ℝ) := (0, c)
def point_U (c : ℝ) : (ℝ × ℝ) := (2, 8 + c)

noncomputable def area_ratio {c : ℝ} (h : c < 0) : ℝ :=
  let T := point_T c;
  let U := point_U c;
  let V := (0, c + 1);
  let W := (0, 4 + c);
  let VU := (8 + c) - (1 + c) in -- VU = 7
  let WT := (8 + c) - (4 + c) in -- WT = 4 + c
  (VU / WT) / (16 / 49) -- given ratio is 16:49

theorem find_c : ∃ (c : ℝ) (h : c < 0), area_ratio h = 1 :=
by
  sorry

end find_c_l611_611278


namespace smallest_A_l611_611539

theorem smallest_A (A B C D E : ℕ) 
  (hA_even : A % 2 = 0)
  (hB_even : B % 2 = 0)
  (hC_even : C % 2 = 0)
  (hD_even : D % 2 = 0)
  (hE_even : E % 2 = 0)
  (hA_three_digit : 100 ≤ A ∧ A < 1000)
  (hB_three_digit : 100 ≤ B ∧ B < 1000)
  (hC_three_digit : 100 ≤ C ∧ C < 1000)
  (hD_three_digit : 100 ≤ D ∧ D < 1000)
  (hE_three_digit : 100 ≤ E ∧ E < 1000)
  (h_sorted : A < B ∧ B < C ∧ C < D ∧ D < E)
  (h_sum : A + B + C + D + E = 4306) :
  A = 326 :=
sorry

end smallest_A_l611_611539


namespace percentage_failed_in_english_l611_611429

theorem percentage_failed_in_english (total_students : ℕ) (hindi_failed : ℕ) (both_failed : ℕ) (both_passed : ℕ) 
  (H1 : hindi_failed = total_students * 25 / 100)
  (H2 : both_failed = total_students * 25 / 100)
  (H3 : both_passed = total_students * 50 / 100)
  : (total_students * 50 / 100) = (total_students * 75 / 100) + (both_failed) - both_passed
:= sorry

end percentage_failed_in_english_l611_611429


namespace find_length_CF_l611_611434

-- Definitions and conditions given in the problem
def right_angled_triangle (A B C : Type) : Prop :=
  (∃ F : Type, right_angled A F C ∧ right_angled B F C)

def thirty_degree_angle (A B : Type) : Prop := angle A B 30

variables {A B C D F AF BF CF : Type} 

axiom AF_equals_30 : (AF = 30)

axiom sides_relationship_in_30deg_triangle : 
  ∀{X Y : Type} (h : thirty_degree_angle X Y), (opposite_side h) = (1/2) * (hypotenuse h)

-- The theorem we need to prove
theorem find_length_CF : CF = 7.5 :=
by sorry

end find_length_CF_l611_611434


namespace parallel_perpendicular_line_l611_611746

section ParallelAndPerpendicularLines

variables {m n : Line} {α β : Plane}

-- Given conditions
variable (h1 : m ≠ n)
variable (h2 : α ≠ β)

-- Equivalent Lean statement to prove:
theorem parallel_perpendicular_line 
  (h3 : m ∥ n) 
  (h4 : m ⟂ α) 
  : n ⟂ α := 
sorry

end ParallelAndPerpendicularLines

end parallel_perpendicular_line_l611_611746


namespace permutation_remainder_unique_l611_611295

open Nat

theorem permutation_remainder_unique (n : ℕ) :
  Exists (λ (a : Fin n → Fin n), ∀ k < n, ∀ l < n, k ≠ l → (a 1 * a 2 * ... * a k % n) ≠ (a 1 * a 2 * ... * a l % n)) ↔
  n = 1 ∨ n = 4 ∨ Nat.Prime n :=
sorry

end permutation_remainder_unique_l611_611295


namespace sqrt_sum_l611_611637

theorem sqrt_sum : (Real.sqrt 50) + (Real.sqrt 32) = 9 * (Real.sqrt 2) :=
by
  sorry

end sqrt_sum_l611_611637


namespace locus_of_centers_are_two_parabolas_l611_611690

noncomputable def tangent_circle_locus (O : EuclideanPlane.Point) (r : ℝ) (b : EuclideanPlane.Line) : Set (EuclideanPlane.Point) := sorry

theorem locus_of_centers_are_two_parabolas (O : EuclideanPlane.Point) (r : ℝ) (b : EuclideanPlane.Line) :
  ∃ (P Q : EuclideanPlane.Parabola), 
    P.focus = O ∧ 
    Q.focus = O ∧ 
    P.directrix.parallel b ∧ 
    Q.directrix.parallel b ∧ 
    tangent_circle_locus O r b = P.locus_of_vertices ∪ Q.locus_of_vertices := 
sorry

end locus_of_centers_are_two_parabolas_l611_611690


namespace exclude_invalid_three_digit_numbers_l611_611392

theorem exclude_invalid_three_digit_numbers : 
  let total_three_digit_numbers := 900
  let excluded_numbers := 9 * 9
  let valid_numbers := total_three_digit_numbers - excluded_numbers
  valid_numbers = 819 :=
begin
  let total_three_digit_numbers := 900,
  let excluded_numbers := 9 * 9,
  let valid_numbers := total_three_digit_numbers - excluded_numbers,
  show valid_numbers = 819,
  sorry
end

end exclude_invalid_three_digit_numbers_l611_611392


namespace find_x_l611_611572

theorem find_x (x : ℝ) (h : (2012 + x)^2 = x^2) : x = -1006 :=
by
  sorry

end find_x_l611_611572


namespace crops_planting_ways_l611_611605

-- Definitions of the crop types
inductive Crop
| corn | wheat | soybeans | potatoes

-- Definition of the 3x3 grid
def Grid := Array (Array Crop)

-- Conditions of the problem
def adjacent (x y: Nat) : Prop := (x = y + 1) ∨ (x = y - 1)

def valid_placement (field: Grid) : Prop :=
  -- Check no two adjacent squares have the same crop
  ∀ i j : Nat, i < 3 → j < 3 → (∀ di dj : Nat, adjacent i di → adjacent j dj → field[i] ≠ field[di] ∧ field[j] ≠ field[dj]) ∧
  -- Check corn not adjacent to wheat
  ∀ i j : Nat, i < 3 → j < 3 → (field[i][j] = Crop.corn → (∀ di dj : Nat, adjacent i di → adjacent j dj → field[di][dj] ≠ Crop.wheat))

-- The property we want to prove is there are exactly 16 proper plantings
theorem crops_planting_ways : ∃ field : Grid, valid_placement field ∧ (number_of_valid_placements field = 16) :=
begin
  sorry
end

end crops_planting_ways_l611_611605


namespace fishing_tomorrow_l611_611812

theorem fishing_tomorrow (seven_every_day eight_every_other_day three_every_three_days twelve_yesterday ten_today : ℕ)
  (h1 : seven_every_day = 7)
  (h2 : eight_every_other_day = 8)
  (h3 : three_every_three_days = 3)
  (h4 : twelve_yesterday = 12)
  (h5 : ten_today = 10) :
  (seven_every_day + (eight_every_other_day - (twelve_yesterday - seven_every_day)) + three_every_three_days) = 15 :=
by
  sorry

end fishing_tomorrow_l611_611812


namespace greatest_word_value_l611_611935

def letter_value (ch : Char) : ℕ :=
  match ch with
  | 'A' => 1
  | 'B' => 2
  | 'C' => 3
  | 'D' => 4
  | 'E' => 5
  | 'F' => 6
  | _ => 0

def word_value (word : String) : ℕ :=
  word.foldl (λ (sum : ℕ) (ch : Char) => sum + letter_value ch) 0

theorem greatest_word_value :
  ∀ words, words = ["BEEF", "FADE", "FEED", "FACE", "DEAF"] →
    list.maximum_by word_value words = some "FEED" :=
by
  sorry

end greatest_word_value_l611_611935


namespace verify_statements_l611_611515

-- Conditions: The table of prices for the first 5 months
def prices : List (ℕ × ℝ) := [(1, 0.5), (2, 0.8), (3, 1), (4, 1.2), (5, 1.5)]

-- Empirical regression equation
def regression_eq (x : ℕ) (a : ℝ) : ℝ := 0.28 * x + a

-- Summarizing the correct answer 
def correct_answer : List Char := ['B', 'C', 'D']

theorem verify_statements :
  let a := 0.16 in
  let mean_x := 3 in
  let mean_y := 1 in
  let r :=
    (∑ (p : ℕ × ℝ) in prices, (p.1 - mean_x) * (p.2 - mean_y)) /
    (Real.sqrt (∑ (p : ℕ × ℝ) in prices, (p.1 - mean_x) ^ 2) *
     Real.sqrt (∑ (p : ℕ × ℝ) in prices, (p.2 - mean_y) ^ 2))
  in
    (r > 0) ∧
    (a = 0.16) ∧
    (regression_eq 3 a = 1) ∧
    (regression_eq 6 a ≈ 1.84)
    → (['B', 'C', 'D' : Char] = correct_answer) :=
by
  intros a mean_x mean_y r;
  rw list.eq_to_iff;
  sorry

end verify_statements_l611_611515


namespace counting_lamps_l611_611188

theorem counting_lamps (n m : ℕ) (h1 : n = 2011) (h2 : m = 300) :
  (∃ (f : ℕ → bool), (∀ i, f i = true → f (i + 1) ≠ true) ∧
  f 0 = false ∧ f (n - 1) = false ∧
  ∑ i in finset.range n, if f i then 1 else 0 = m) →
  nat.choose (2011 - 300 - 1) 300 = nat.choose 1710 300 :=
by simp [h1, h2]

end counting_lamps_l611_611188


namespace campground_distance_l611_611505

-- Definitions based on conditions
def speed1 : ℕ := 60
def time1 : ℕ := 2
def speed2 : ℕ := 50
def time2 : ℕ := 3
def speed3 : ℕ := 55
def time3 : ℕ := 4

-- Compute distance using given conditions
def distance1 := speed1 * time1
def distance2 := speed2 * time2
def distance3 := speed3 * time3

def total_distance := distance1 + distance2 + distance3

-- The proof statement
theorem campground_distance : total_distance = 490 := by
  -- Provided estimation, should be replaced with actual proof
  have h1 : distance1 = 60 * 2 := by rfl
  have h2 : distance2 = 50 * 3 := by rfl
  have h3 : distance3 = 55 * 4 := by rfl
  have d1 : distance1 = 120 := by
    rw [h1]
    norm_num
  have d2 : distance2 = 150 := by
    rw [h2]
    norm_num
  have d3 : distance3 = 220 := by
    rw [h3]
    norm_num
  show total_distance = 490, from calc 
    total_distance = distance1 + distance2 + distance3 : by rfl
    ... = 120 + 150 + 220 : by rw [d1, d2, d3]
    ... = 490 : by norm_num

end campground_distance_l611_611505


namespace monotonicity_f_range_f_l611_611369

noncomputable def f (a b x : ℝ) : ℝ := x - a * Real.log x - b / x - 2

theorem monotonicity_f (a b : ℝ) (h_a_b : a - b = 1) (h_a : 1 < a) :
  (a = 2 → ∀ x > 0, differentiable_at ℝ (f a b) x ∧ deriv (f a b) x ≥ 0) ∧
  (a > 2 → ∀ x > 0, differentiable_at ℝ (f a b) x ∧ 
    ((x < 1 ∨ x > a - 1) → deriv (f a b) x > 0) ∧ 
    (1 < x ∧ x < a - 1 → deriv (f a b) x < 0)) ∧
  (1 < a ∧ a < 2 → ∀ x > 0, differentiable_at ℝ (f a b) x ∧ 
    ((x < a - 1 ∨ x > 1) → deriv (f a b) x > 0) ∧ 
    (a - 1 < x ∧ x < 1 → deriv (f a b) x < 0)) :=
sorry

theorem range_f (a : ℝ) (h_b : -1 = -1) (h_a : a ≤ 4) :
  (∀ x ∈ Icc 2 4, (f a (-1) x) < -3 / x) ↔ (2 / Real.log 2 < a ∧ a ≤ 4) :=
sorry

end monotonicity_f_range_f_l611_611369


namespace male_students_plant_trees_l611_611527

theorem male_students_plant_trees (total_avg : ℕ) (female_trees : ℕ) (male_trees : ℕ) 
  (h1 : total_avg = 6) 
  (h2 : female_trees = 15)
  (h3 : 1 / (male_trees : ℝ) + 1 / (female_trees : ℝ) = 1 / (total_avg : ℝ)) : 
  male_trees = 10 := 
sorry

end male_students_plant_trees_l611_611527


namespace fishing_tomorrow_l611_611797

theorem fishing_tomorrow (every_day_fishers every_other_day_fishers every_three_days_fishers fishers_yesterday fishers_today : ℕ) :
  (every_day_fishers = 7) →
  (every_other_day_fishers = 8) →
  (every_three_days_fishers = 3) →
  (fishers_yesterday = 12) →
  (fishers_today = 10) →
  (every_three_days_fishers + every_day_fishers + (every_other_day_fishers - (fishers_yesterday - every_day_fishers)) = 15) :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  simp
  sorry

end fishing_tomorrow_l611_611797


namespace problem_statement_l611_611399

theorem problem_statement (x : ℂ) (h : x + 1 / x = real.sqrt 5) : x ^ 10 = 1 := by
  sorry

end problem_statement_l611_611399


namespace coefficient_x3_in_expansion_is_correct_l611_611459

noncomputable def a : ℝ := ∫ x in 0..π, (Real.cos x - Real.sin x)

theorem coefficient_x3_in_expansion_is_correct :
  a = -2 →
  (∃ (c : ℝ), c = -160 ∧ ∃ (f : ℕ → ℝ), (f 3 = c ∧ (x^2 + a / x) ^ 6 = ∑ k in Finset.range 13, f k * x ^ (12 - 3 * k))) :=
begin
  intro ha,
  use -160,
  split,
  exact rfl,
  use (λ k, (-1)^k * 2^k * Nat.choose 6 k),
  split,
  exact rfl,
  sorry
end

end coefficient_x3_in_expansion_is_correct_l611_611459


namespace LN_parallel_AB_l611_611332

open EuclideanGeometry

noncomputable theory

variables {A B C D K L M N : Point}
variables (P : Set Point) [∀ x : Point, Point x]

-- Let ABCD be a parallelogram
variable (h_parallelogram : parallelogram A B C D)

-- Points K, L, and M are on sides AB, BC, and the extension of CD beyond D respectively
variable (h_K_on_AB : K ∈ line_through A B)
variable (h_L_on_BC : L ∈ line_through B C)
variable (h_M_on_CD_ext : M ∈ line_extension_through CD D)

-- Triangles KLM and BCA are congruent
variable (h_congruent : congruent (triangle K L M) (triangle B C A))

-- Segment KM intersects segment AD at point N
variable (h_intersect : intersects_pt (seg K M) (seg A D) N)

-- Conclusion: LN is parallel to AB
theorem LN_parallel_AB : is_parallel (line_through L N) (line_through A B) :=
sorry

end LN_parallel_AB_l611_611332


namespace people_per_table_l611_611601

def total_people_invited : ℕ := 68
def people_who_didn't_show_up : ℕ := 50
def number_of_tables_needed : ℕ := 6

theorem people_per_table (total_people_invited people_who_didn't_show_up number_of_tables_needed : ℕ) : 
  total_people_invited - people_who_didn't_show_up = 18 ∧
  (total_people_invited - people_who_didn't_show_up) / number_of_tables_needed = 3 :=
by
  sorry

end people_per_table_l611_611601


namespace number_of_possible_A2_eq_one_l611_611065

noncomputable def unique_possible_A2 (A : Matrix (Fin 2) (Fin 2) ℝ) : Prop :=
  (A^4 = 0) → (A^2 = 0)

theorem number_of_possible_A2_eq_one (A : Matrix (Fin 2) (Fin 2) ℝ) :
  unique_possible_A2 A :=
by 
  sorry

end number_of_possible_A2_eq_one_l611_611065


namespace eccentricity_of_hyperbola_is_2_l611_611408

noncomputable def hyperbola_eccentricity (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) : ℝ :=
let e := (1 + (b^2 / a^2)).sqrt in
if ((abs (2 * b) / (a^2 + b^2).sqrt = (3).sqrt) ∧ (a_pos) ∧ (b_pos)) then 2 else e

-- To state the theorem in Lean

theorem eccentricity_of_hyperbola_is_2 (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) :
  let e := hyperbola_eccentricity a b a_pos b_pos in e = 2 :=
by
  sorry

end eccentricity_of_hyperbola_is_2_l611_611408


namespace election_winner_margin_l611_611960

theorem election_winner_margin
  (total_votes : ℕ)
  (winner_votes : ℕ)
  (winner_percentage : ℝ) :
  winner_votes = 775 →
  winner_percentage = 0.62 →
  total_votes = winner_votes / winner_percentage →
  total_votes * (1 - winner_percentage) = total_votes - winner_votes →
  winner_votes - (total_votes - winner_votes) = 300 :=
begin
  intros,
  sorry
end

end election_winner_margin_l611_611960


namespace correct_calculation_l611_611213

-- Define the conditions as boolean values
def condition_A : Prop := (sqrt 2) ^ 0 = sqrt 2
def condition_B : Prop := 2 * sqrt 3 + 3 * sqrt 3 = 5 * sqrt 6
def condition_C : Prop := sqrt 8 = 4 * sqrt 2
def condition_D : Prop := sqrt 3 * (2 * sqrt 3 - 2) = 6 - 2 * sqrt 3

-- State the theorem to be proved
theorem correct_calculation : condition_D :=
by
  -- the proof is skipped with sorry
  sorry

end correct_calculation_l611_611213


namespace max_edge_length_cube_in_tetrahedron_l611_611329

theorem max_edge_length_cube_in_tetrahedron (a : ℝ) (r : ℝ) : 
    let edge_length_tetrahedron := 3 * real.sqrt 6 in
    let edge_length_cube := real.sqrt 3 in
    (∀ (cube_rotating_freely : Prop), 
        cube_rotating_freely → 
        r = 3 / 2 →
        a = edge_length_cube) := sorry

end max_edge_length_cube_in_tetrahedron_l611_611329


namespace bill_share_is_600_l611_611099

-- Define the given conditions
def share_ratio := 1 : 2 : 3
def bobs_share := 900
def one_part := bobs_share / 3
def bills_share := 2 * one_part

-- Theorem statement to prove that Bill's share is $600
theorem bill_share_is_600 (h1 : share_ratio = 1 : 2 : 3)
                          (h2 : bobs_share = 900) :
  bills_share = 600 := by
  -- Proof omitted
  sorry

end bill_share_is_600_l611_611099


namespace fundraiser_problem_l611_611307

theorem fundraiser_problem
  (students_brownies : ℕ)
  (students_cookies : ℕ)
  (students_donuts : ℕ)
  (brownies_per_student : ℕ)
  (cookies_per_student : ℕ)
  (donuts_per_student : ℕ)
  (price_per_item : ℕ)
  (total_raised : ℕ) :
  students_brownies = 30 →
  brownies_per_student = 12 →
  donuts_per_student = 12 →
  students_donuts = 15 →
  price_per_item = 2 →
  total_raised = 2040 →
  (30 * 12 * 2 + students_cookies * 24 * 2 + 15 * 12 * 2) = 2040 →
  students_cookies = 20 :=
by {
  intros hb hs hc hfd hp ht heq,
  have eq_brownies := (hb ▸ hs ▸ rfl) : 30 * 12 = 360,
  have eq_donuts := (hc ▸ hfd ▸ rfl) : 15 * 12 = 180,
  have eq_total := calc
    360 * 2 + (students_cookies * 24 * 2) + 180 * 2
      = 2040 : heq,
  sorry
}

end fundraiser_problem_l611_611307


namespace alt_rep_of_set_l611_611509

def NatPos (x : ℕ) := x > 0

theorem alt_rep_of_set : {x : ℕ | NatPos x ∧ x - 3 < 2} = {1, 2, 3, 4} := by
  sorry

end alt_rep_of_set_l611_611509


namespace susie_investment_l611_611926

theorem susie_investment :
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ 2000 ∧
  (x * 1.04 + (2000 - x) * 1.06 = 2120) → (x = 0) :=
by
  sorry

end susie_investment_l611_611926


namespace largest_angle_is_75_l611_611173

-- Let the measures of the angles be represented as 3x, 4x, and 5x for some value x
variable (x : ℝ)

-- Define the angles based on the given ratio
def angle1 := 3 * x
def angle2 := 4 * x
def angle3 := 5 * x

-- The sum of the angles in a triangle is 180 degrees
axiom angle_sum : angle1 + angle2 + angle3 = 180

-- Prove that the largest angle is 75 degrees
theorem largest_angle_is_75 : 5 * (180 / 12) = 75 :=
by
  -- Proof is not required as per the instructions
  sorry

end largest_angle_is_75_l611_611173


namespace ages_sum_13_and_product_72_l611_611318

theorem ages_sum_13_and_product_72 (g b s : ℕ) (h1 : b < g) (h2 : g < s) (h3 : b * g * s = 72) : b + g + s = 13 :=
sorry

end ages_sum_13_and_product_72_l611_611318


namespace train_cross_duration_l611_611047

noncomputable def train_length : ℝ := 250
noncomputable def train_speed_kmph : ℝ := 162
noncomputable def train_speed_mps : ℝ := train_speed_kmph * (1000 / 3600)
noncomputable def time_to_cross_pole : ℝ := train_length / train_speed_mps

theorem train_cross_duration :
  time_to_cross_pole = 250 / (162 * (1000 / 3600)) :=
by
  -- The detailed proof is omitted as per instructions
  sorry

end train_cross_duration_l611_611047


namespace find_common_ratio_l611_611298

-- Define the terms of the series
def a₀ := 7 / 8
def a₁ := -21 / 32
def a₂ := 63 / 128

-- Define the common ratio to be proved
noncomputable def common_ratio := -3 / 4

-- Define the infinite geometric series
def geometric_series (n : ℕ) : ℚ :=
  if n = 0 then a₀ else a₀ * (common_ratio ^ n)

theorem find_common_ratio :
  (a₁ / a₀ = common_ratio) ∧ (a₂ / a₁ = common_ratio) :=
by
  have h1 : a₁ / a₀ = -3 / 4 := by sorry
  have h2 : a₂ / a₁ = -3 / 4 := by sorry
  exact ⟨h1, h2⟩

end find_common_ratio_l611_611298


namespace decision_func_composition_l611_611906

def dem_func (x y z : Bool) : Bool :=
  x ∧ y ∨ y ∧ z ∨ z ∧ x

def is_dictatoric (f : Vector Bool n → Bool) : Prop :=
  ∃ i, ∀ (x : Vector Bool n), f x = x.nth i

def is_decision (f : Vector Bool n → Bool) : Prop :=
  (∀ (x y : Vector Bool n), f x ≠ f y → ∃ i, x.nth i ≠ y.nth i) ∧
  (∀ (x : Vector Bool n) (i : Fin n) (y : Bool),
   (f (x.update_nth i y) = f x) → (∀ (j : Fin n), f (x.update_nth j f x) = f x))

theorem decision_func_composition (f : Vector Bool n → Bool)
  (h : is_decision f) :
  ∃ g h (g1 g2 : Vector Bool n → Bool), 
    (g1 = dem_func ∧ is_dictatoric h ∧ f = h ∘ g1 ∘ g ∘ ... ∘ g2) ∨
    (∃ g', f = g' ∘ dem_func) :=
sorry

end decision_func_composition_l611_611906


namespace four_digit_numbers_with_4_and_5_l611_611390

theorem four_digit_numbers_with_4_and_5 : 
  (number_of_valid_numbers : ℕ) = 770 :=
by
  -- Define the range of four-digit numbers
  let four_digit_range := list.range (9999 - 1000 + 1).map (λ x => x + 1000)
  -- Define a function to check if a number contains at least one 4 and one 5
  let contains_4_and_5 (n : ℕ) : bool := 
    let digits := n.digits 10
    list.mem 4 digits ∧ list.mem 5 digits
  -- Filter four_digit_range to get the numbers containing at least one 4 and one 5
  let valid_numbers := list.filter contains_4_and_5 four_digit_range
  -- Assert the number of valid numbers is 770
  exact valid_numbers.length = 770
sorry

end four_digit_numbers_with_4_and_5_l611_611390


namespace increasing_function_on_positive_real_l611_611631

noncomputable def f_A (x : ℝ) := -1 / (x + 1)
noncomputable def f_B (x : ℝ) := x^2 - 3 * x
noncomputable def f_C (x : ℝ) := 3 - x
noncomputable def f_D (x : ℝ) := -|x|

theorem increasing_function_on_positive_real :
  ( ∀ (x : ℝ), 0 < x → f_A x ≤ f_A (x+ε) ∀ (ε : ℝ), 0 < ε) ∧
  ( ∃ x : ℝ, \ (0 < x → x < 3/2 → f_B x ≤ f_B (x+ε) ∀ (ε : ℝ), 0 < ε)) = false)reor ( ∀ (x : ℝ), 0 < x → f_C x ≤ f_C (x+ε) ∀ (ε : ℝ), 0 < ε) ∧
  ( ∀ (x : ℝ), 0 < x → f_D x ≤ f_D (x+ε) ∀ (ε : ℝ), 0 < ε) :=
by 
  sorry

end increasing_function_on_positive_real_l611_611631


namespace ratio_of_plums_to_peaches_is_three_l611_611683

theorem ratio_of_plums_to_peaches_is_three :
  ∃ (L P W : ℕ), W = 1 ∧ P = W + 12 ∧ L = 3 * P ∧ W + P + L = 53 ∧ (L / P) = 3 :=
by
  sorry

end ratio_of_plums_to_peaches_is_three_l611_611683


namespace problem_part1_problem_part2_l611_611082

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := (b * x / Real.log x) - (a * x)
noncomputable def f' (x : ℝ) (a b : ℝ) : ℝ :=
  (b * (Real.log x - 1) / (Real.log x)^2) - a

theorem problem_part1 (a b : ℝ) :
  (f' (Real.exp 2) a b = -(3/4)) ∧ (f (Real.exp 2) a b = -(1/2) * (Real.exp 2)) →
  a = 1 ∧ b = 1 :=
sorry

theorem problem_part2 (a : ℝ) :
  (∃ x1 x2, x1 ∈ Set.Icc (Real.exp 1) (Real.exp 2) ∧ x2 ∈ Set.Icc (Real.exp 1) (Real.exp 2) ∧ f x1 a 1 ≤ f' x2 a 1 + a) →
  a ≥ (1/2 - 1/(4 * Real.exp 2)) :=
sorry

end problem_part1_problem_part2_l611_611082


namespace matrix_det_problem_l611_611569

-- Define the determinant of a 2x2 matrix
def det (a b c d : ℤ) : ℤ := a * d - b * c

-- State the problem in Lean
theorem matrix_det_problem : 2 * det 5 7 2 3 = 2 := by
  sorry

end matrix_det_problem_l611_611569


namespace sum_seq_formula_l611_611538

-- Define the sequence
def seq (n : ℕ) : ℕ := 3 + 2^n

-- Define the sum of the first n terms of the sequence
def sum_seq (n : ℕ) : ℕ :=
  (finset.range n).sum (λ k, seq k)

-- Define the sum of the first n terms as derived in the solution
def sum_formula (n : ℕ) : ℕ :=
  2^(n + 1) + 3 * n - 2

-- The theorem to be proved
theorem sum_seq_formula (n : ℕ) : sum_seq n = sum_formula n := 
  sorry

end sum_seq_formula_l611_611538


namespace g_at_pi_over_4_l611_611725

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 3)
noncomputable def g (x : ℝ) : ℝ := f x + 1

theorem g_at_pi_over_4 : g (Real.pi / 4) = 3 / 2 :=
by 
  sorry

end g_at_pi_over_4_l611_611725


namespace regular_seven_gon_l611_611767

theorem regular_seven_gon 
    (A : Fin 7 → ℝ × ℝ)
    (cong_diagonals_1 : ∀ (i : Fin 7), dist (A i) (A ((i + 2) % 7)) = dist (A 0) (A 2))
    (cong_diagonals_2 : ∀ (i : Fin 7), dist (A i) (A ((i + 3) % 7)) = dist (A 0) (A 3))
    : ∀ (i j : Fin 7), dist (A i) (A ((i + 1) % 7)) = dist (A j) (A ((j + 1) % 7)) :=
by sorry

end regular_seven_gon_l611_611767


namespace age_difference_l611_611607

theorem age_difference (A B : ℕ) (h1 : B = 38) (h2 : A + 10 = 2 * (B - 10)) : A - B = 8 :=
by
  sorry

end age_difference_l611_611607


namespace problem_2012_square_eq_x_square_l611_611570

theorem problem_2012_square_eq_x_square : 
  ∃ x : ℤ, (2012 + x)^2 = x^2 ∧ x = -1006 :=
by {
  existsi (-1006 : ℤ),
  split,
  sorry,
  refl
}

end problem_2012_square_eq_x_square_l611_611570


namespace vectors_coplanar_l611_611263

theorem vectors_coplanar :
  let a := (4, 1, 2 : ℝ)
  let b := (9, 2, 5 : ℝ)
  let c := (1, 1, -1 : ℝ)
  let A := !![a.1, a.2, a.3; b.1, b.2, b.3; c.1, c.2, c.3]
  matrix.det A = 0 :=
by
  sorry

end vectors_coplanar_l611_611263


namespace donation_relationship_l611_611836

-- Definitions and conditions.
def total_students : ℕ := 45
def donation_per_girl : ℕ := 20
def donation_per_boy : ℕ := 25

-- Variables representing the number of girls and total donation.
variables (x y : ℕ)

-- The theorem asserting the relationship between y and x.
theorem donation_relationship (h1 : total_students = 45)
                             (h2 : donation_per_girl = 20)
                             (h3 : donation_per_boy = 25)
                             (h4 : x <= total_students)
                             (h5 : y = donation_per_girl * x + donation_per_boy * (total_students - x)) :
    y = -5 * x + 1125 := 
sorry

end donation_relationship_l611_611836


namespace x_pow_10_eq_correct_answer_l611_611401

noncomputable def x : ℝ := sorry

theorem x_pow_10_eq_correct_answer (h : x + (1 / x) = Real.sqrt 5) : 
  x^10 = (50 + 25 * Real.sqrt 5) / 2 := 
sorry

end x_pow_10_eq_correct_answer_l611_611401


namespace polynomial_remainder_l611_611353

theorem polynomial_remainder (a b : ℝ) (h : ∀ x : ℝ, (x^3 - 2*x^2 + a*x + b) % ((x - 1)*(x - 2)) = 2*x + 1) : 
  a = 1 ∧ b = 3 := 
sorry

end polynomial_remainder_l611_611353


namespace fishing_tomorrow_l611_611796

theorem fishing_tomorrow (every_day_fishers every_other_day_fishers every_three_days_fishers fishers_yesterday fishers_today : ℕ) :
  (every_day_fishers = 7) →
  (every_other_day_fishers = 8) →
  (every_three_days_fishers = 3) →
  (fishers_yesterday = 12) →
  (fishers_today = 10) →
  (every_three_days_fishers + every_day_fishers + (every_other_day_fishers - (fishers_yesterday - every_day_fishers)) = 15) :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  simp
  sorry

end fishing_tomorrow_l611_611796


namespace max_sin_A_sin_B_l611_611039

theorem max_sin_A_sin_B (A B : ℝ) (h : A + B = π / 2) : 
  ∃ (M : ℝ), M = 1 / 2 ∧ ∀ A B, sin A * sin B ≤ M := 
sorry

end max_sin_A_sin_B_l611_611039


namespace fishing_tomorrow_l611_611816

theorem fishing_tomorrow (seven_every_day eight_every_other_day three_every_three_days twelve_yesterday ten_today : ℕ)
  (h1 : seven_every_day = 7)
  (h2 : eight_every_other_day = 8)
  (h3 : three_every_three_days = 3)
  (h4 : twelve_yesterday = 12)
  (h5 : ten_today = 10) :
  (seven_every_day + (eight_every_other_day - (twelve_yesterday - seven_every_day)) + three_every_three_days) = 15 :=
by
  sorry

end fishing_tomorrow_l611_611816


namespace min_oranges_picked_l611_611124

noncomputable def minimum_oranges (n : ℕ) : Prop :=
  ∀ (m : ℕ → ℕ), (∀ i j k : ℕ, i ≠ j ∧ j ≠ k ∧ i ≠ k → (m i + m j + m k : ℝ) / (∑ l : ℕ in finset.univ \ {i, j, k}, m l) < 0.05) → n ≥ 64

theorem min_oranges_picked : ∃ n, minimum_oranges n := by
  use 64
  sorry

end min_oranges_picked_l611_611124


namespace quadratic_expression_l611_611462

noncomputable def roots_of_quadratic (a b c : ℝ) : set ℝ :=
{s | a * s^2 + b * s + c = 0}

theorem quadratic_expression (m n : ℝ) (hm : m ∈ roots_of_quadratic 1 5 (-2023))
(hn : n ∈ roots_of_quadratic 1 5 (-2023)) :
  m^2 + 7 * m + 2 * n = 2013 :=
sorry

end quadratic_expression_l611_611462


namespace extreme_point_max_value_range_of_m_l611_611723

noncomputable def f (x m : ℝ) := log x - m * x ^ 2 - x
noncomputable def g (x m : ℝ) := (log x / x) - m * x - 1 + x
noncomputable def h (x : ℝ) := (1 - log x) / x^2

theorem extreme_point_max_value 
  (m : ℝ) 
  (h_extr : f (1/2:ℝ) m = 0)
  : f (1/2:ℝ) m = - log 2 - 3 / 4 :=
sorry

theorem range_of_m 
  (m x1 x2 : ℝ) 
  (h1 : 1/e ≤ x1) 
  (h2 : x1 ≤ e) 
  (h3 : 1/e ≤ x2) 
  (h4 : x2 ≤ e)
  (h5 : x1 ≠ x2)
  : x2 * f x1 m - x1 * f x2 m > x1 * x2 * (x2 - x1) →
    (-∞ < m ∧ m ≤ 1) ∨ (2 * real.exp (2) + 1 ≤ m ∧ m < ∞) :=
sorry

end extreme_point_max_value_range_of_m_l611_611723


namespace jacket_cost_is_30_l611_611105

-- Let's define the given conditions
def num_dresses := 5
def cost_per_dress := 20 -- dollars
def num_pants := 3
def cost_per_pant := 12 -- dollars
def num_jackets := 4
def transport_cost := 5 -- dollars
def initial_amount := 400 -- dollars
def remaining_amount := 139 -- dollars

-- Define the cost per jacket
def cost_per_jacket := 30 -- dollars

-- Final theorem statement to be proved
theorem jacket_cost_is_30:
  num_dresses * cost_per_dress + num_pants * cost_per_pant + num_jackets * cost_per_jacket + transport_cost = initial_amount - remaining_amount :=
sorry

end jacket_cost_is_30_l611_611105


namespace count_integers_with_property_l611_611937

noncomputable def F : ℕ → ℕ := sorry

lemma F_properties (n : ℕ) :
  F(4 * n) = F(2 * n) + F(n) ∧
  F(4 * n + 2) = F(4 * n) + 1 ∧
  F(2 * n + 1) = F(2 * n) + 1 :=
sorry

theorem count_integers_with_property (m : ℕ) (hm : 0 < m) :
  (finset.card {n | 0 ≤ n ∧ n < 2^m ∧ F(4 * n) = F(3 * n)}) = F(2^(m+1)) :=
sorry

end count_integers_with_property_l611_611937


namespace table_results_equal_l611_611430

theorem table_results_equal (M : Matrix (Fin 4) (Fin 4) (Fin 2 → 1 → 1 → 2)) :
  ∃ (i₁ i₂ : Fin 4), i₁ ≠ i₂ ∧ (∑ j, M i₁ j) = ∑ j, M i² j ∨
  ∃ (j₁ j₂ : Fin 4), j₁ ≠ j₂ ∧ (∏ i, M i j₁) = ∏ i, M i³ j₂ :=
sorry

end table_results_equal_l611_611430


namespace angle_between_DC_AM_l611_611840

noncomputable def angle_between_vectors (a : ℝ) : ℝ :=
  let m := (2 * a / 3)^2
  let n := (2 * a / 3)^2
  let p := (a / 3)^2
  let dot_product := 5 * a^2 / 6
  let magnitude_dc := a
  let magnitude_am := a * (√13) / 3
  let cos_angle := dot_product / (magnitude_dc * magnitude_am)
  arccos cos_angle

theorem angle_between_DC_AM {a : ℝ} (ha : 0 < a) :
  angle_between_vectors a = arccos (5 / (2 * √13)) :=
by sorry

end angle_between_DC_AM_l611_611840


namespace fishing_tomorrow_l611_611821

theorem fishing_tomorrow (yesterday_fishers today_fishers : ℕ)
  (every_day_fishers every_other_day_fishers every_three_days_fishers : ℕ)
  (total_population : ℕ):
  yesterday_fishers = 12 → 
  today_fishers = 10 → 
  every_day_fishers = 7 → 
  every_other_day_fishers = 8 → 
  every_three_days_fishers = 3 → 
  total_population = yesterday_fishers + today_fishers + (total_population - (every_day_fishers + every_other_day_fishers + every_three_days_fishers)) →
  ∃ tomorrow_fishers : ℕ, tomorrow_fishers = 15 :=
by {
  -- This is a statement definition, the proof is not required and thus marked as "sorry:"
  sorry
}

end fishing_tomorrow_l611_611821


namespace eval_expression_cos_30_l611_611499

noncomputable def evaluated_expression (x : ℝ) : ℝ := (x - (2 * x - 1) / x) / (x / (x - 1))

theorem eval_expression_cos_30 : 
    let x := real.cos (real.pi / 6) 
    in evaluated_expression x = real.cos (real.pi / 6) - 1 :=
by
  let x := real.cos (real.pi / 6)
  show evaluated_expression x = x - 1
  sorry

end eval_expression_cos_30_l611_611499


namespace minimum_oranges_l611_611115

theorem minimum_oranges (n : ℕ) (m : ℕ → ℝ) (h : ∀ i j k : ℕ, i < n → j < n → k < n → i ≠ j → i ≠ k → j ≠ k → (m i + m j + m k) < 0.05 * ∑ l in Finset.range n \ {i, j, k}, m l) : n ≥ 64 := 
sorry

end minimum_oranges_l611_611115


namespace a4_equals_zero_l611_611729

-- Define the general term of the sequence
def a (n : ℕ) (h : n > 0) : ℤ := n^2 - 3 * n - 4

-- The theorem statement to prove a_4 = 0
theorem a4_equals_zero : a 4 (by norm_num) = 0 :=
sorry

end a4_equals_zero_l611_611729


namespace find_x_perpendicular_vectors_l611_611684

theorem find_x_perpendicular_vectors : 
    ∀ (x : ℝ), (1 : ℝ, 2 : ℝ) ⬝ (2 : ℝ, x) = 0 → x = -1 :=
by {
    intro x,
    intro h,
    sorry
}

end find_x_perpendicular_vectors_l611_611684


namespace projection_AC_on_AB_l611_611321

/-- Define points in 3D space -/
structure Point3D :=
(x : ℝ)
(y : ℝ)
(z : ℝ)

/-- Calculate vector from two points -/
def vector (P Q : Point3D) : Point3D :=
{ x := Q.x - P.x, y := Q.y - P.y, z := Q.z - P.z }

/-- Calculate dot product of two vectors -/
def dot_product (u v : Point3D) : ℝ :=
u.x * v.x + u.y * v.y + u.z * v.z

/-- Calculate the magnitude of a vector -/
def magnitude (v : Point3D) : ℝ :=
Real.sqrt (v.x * v.x + v.y * v.y + v.z * v.z)

/-- Define the projection of vector u on vector v -/
def projection (u v : Point3D) : Point3D :=
let scalar_proj := (dot_product u v) / (magnitude v) in
{ x := scalar_proj * v.x / (magnitude v), y := scalar_proj * v.y / (magnitude v), z := scalar_proj * v.z / (magnitude v) }

noncomputable def A : Point3D := { x := 1, y := -2, z := 1 }
noncomputable def B : Point3D := { x := 1, y := -5, z := 4 }
noncomputable def C : Point3D := { x := 2, y := 3, z := 4 }

noncomputable def AB : Point3D := vector A B
noncomputable def AC : Point3D := vector A C

theorem projection_AC_on_AB :
  projection AC AB = { x := 0, y := 1, z := -1 } :=
sorry

end projection_AC_on_AB_l611_611321


namespace number_of_ways_to_choose_4_cards_with_conditions_l611_611741

-- Defines a standard deck of 52 cards.
constant deck : finset (ℕ × string)
-- Assume a standard set of 52 cards
axiom deck_cards : deck.cardinality = 52
-- Define suits and face cards.
constant suits : finset string := {"hearts", "diamonds", "clubs", "spades"}
constant face_cards : finset ℕ := {11, 12, 13} -- Face cards are considered as Jack(11), Queen(12), King(13)

-- The problem is to show that the number of ways to choose 4 cards from a standard deck 
-- such that all 4 cards are of different suits and at least one card is a face card is 26364.
theorem number_of_ways_to_choose_4_cards_with_conditions :
  ∃ (cards : finset (ℕ × string)), 
    cards.cardinality = 4 ∧ 
    (∀ card ∈ cards, card.2 ∈ suits) ∧ 
    (∀ suit ∈ suits, ∃! card ∈ cards, card.2 = suit) ∧ 
    (∃ card ∈ cards, card.1 ∈ face_cards) ∧
    (fintype.card (finset (choose_4_cards deck)) = 26364) :=
sorry

end number_of_ways_to_choose_4_cards_with_conditions_l611_611741


namespace trapezoid_DBCE_area_66_l611_611851

-- Declare the relevant variables
variables {A B C D E : Type} [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E]

-- Declare all the necessary conditions
def similar_isosceles_triangles (ABC : Triangle A B C) : Prop :=
  ABC.isosceles ∧ ∀ t, similar t ABC

def six_small_triangles_of_area_2 (triangles : List (Triangle A B C)) : Prop :=
  (∀ t ∈ triangles, t.area = 2) ∧ triangles.length = 6

def triangle_ABC_area_72 (ABC : Triangle A B C) : Prop :=
  ABC.area = 72

-- Define the problem
theorem trapezoid_DBCE_area_66 (ABC : Triangle A B C) (triangles : List (Triangle A B C)) (DBCE : Trapezoid D B C E) :
  similar_isosceles_triangles ABC →
  six_small_triangles_of_area_2 triangles →
  triangle_ABC_area_72 ABC →
  DBCE.area = 66 :=
begin
  -- Use the three conditions provided to prove the area of the trapezoid is 66
  intros,
  sorry
end

end trapezoid_DBCE_area_66_l611_611851


namespace correct_assertions_l611_611651

-- Definitions for the respective assertions
def assertion1 (x : ℝ) : Prop := 
  (f : ℝ → ℝ) (f x = |x| / x) ≠ 
  (λ x, if x < 0 then -1 else 1)

def assertion2 : Prop := 
  (∀ x t : ℝ, (2 * x - 1) = (2 * t - 1) ↔ x = t)

def assertion3 (f : ℝ → ℝ) : Prop := 
  (∀ y : ℝ, ∃! x : ℝ, y = f x)

def assertion4 : Prop := 
  ¬ (∀ x : ℝ, y = 1)

-- The final statement proving that only assertions 2 and 3 are correct
theorem correct_assertions : assertion2 ∧ assertion3 := by
  sorry

end correct_assertions_l611_611651


namespace ways_to_select_cards_l611_611957

-- Define the conditions
def is_valid_selection (total_cards : List (List ℕ)) (selected_cards : List ℕ) : Prop :=
  ∑ c in selected_cards, if c = 1 then 1 else 0 ≤ 1 ∧
  selected_cards.length = 3 ∧
  ∃ c1 c2, c1 ≠ c2 ∧ (List.count selected_cards c1 > 0) ∧ (List.count selected_cards c2 > 0)

-- Main theorem to prove the number of valid ways to select cards given constraints
theorem ways_to_select_cards 
  (total_cards : List (List ℕ))
  (valid_selection_condition : total_cards.length = 4 ∧ 
                              (∀ color_cards, color_cards ∈ total_cards → color_cards.length = 4)) :
  Σ' (selection : List ℕ), is_valid_selection total_cards selection = 472 :=
  sorry

end ways_to_select_cards_l611_611957


namespace max_a_plus_b_l611_611182

noncomputable def max_temperature_diff (a b : ℝ) (t : ℝ) : ℝ :=
  a * Real.sin t + b * Real.cos t

theorem max_a_plus_b 
  (a b : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (h : ∃ T : ℝ → ℝ, ∀ t : ℝ, T t = max_temperature_diff a b t)
  (temp_diff : ∀ T : ℝ → ℝ, ∃ Tmax Tmin : ℝ, Tmax - Tmin = 10) :
  a + b ≤ 5 * Real.sqrt 2 :=
sorry

end max_a_plus_b_l611_611182


namespace final_price_relative_l611_611596

-- Definitions of the conditions
variable (x : ℝ)
#check x * 1.30  -- original price increased by 30%
#check x * 1.30 * 0.85  -- after 15% discount on increased price
#check x * 1.30 * 0.85 * 1.05  -- after applying 5% tax on discounted price

-- Theorem to prove the final price relative to the original price
theorem final_price_relative (x : ℝ) : 
  (x * 1.30 * 0.85 * 1.05) = (1.16025 * x) :=
by
  sorry

end final_price_relative_l611_611596


namespace smallest_y_square_factor_l611_611763

theorem smallest_y_square_factor (y n : ℕ) (h₀ : y = 10) 
  (h₁ : ∀ m : ℕ, ∃ k : ℕ, k * k = m * y)
  (h₂ : ∀ (y' : ℕ), (∀ m : ℕ, ∃ k : ℕ, k * k = m * y') → y ≤ y') : 
  n = 10 :=
by sorry

end smallest_y_square_factor_l611_611763


namespace one_third_percent_of_150_l611_611969

theorem one_third_percent_of_150 : (1/3) * (150 / 100) = 0.5 := by
  sorry

end one_third_percent_of_150_l611_611969


namespace equilateral_triangle_area_l611_611510

theorem equilateral_triangle_area (h : ∀ (a : ℝ), a = 2 * Real.sqrt 3) : 
  ∃ (a : ℝ), a = 4 * Real.sqrt 3 := 
sorry

end equilateral_triangle_area_l611_611510


namespace possible_numbers_erased_one_digit_reduce_sixfold_l611_611613

theorem possible_numbers_erased_one_digit_reduce_sixfold (N : ℕ) :
  (∃ N' : ℕ, N = 6 * N' ∧ N % 10 ≠ 0 ∧ ¬N = N') ↔
  N = 12 ∨ N = 24 ∨ N = 36 ∨ N = 48 ∨ N = 108 :=
by {
  sorry
}

end possible_numbers_erased_one_digit_reduce_sixfold_l611_611613


namespace selling_price_of_book_l611_611610

theorem selling_price_of_book (cost_price : ℕ) (profit_rate : ℕ) (profit : ℕ) (selling_price : ℕ) :
  cost_price = 50 → profit_rate = 80 → profit = (profit_rate * cost_price) / 100 → selling_price = cost_price + profit → selling_price = 90 :=
by
  intros h_cost_price h_profit_rate h_profit h_selling_price
  rw [h_cost_price, h_profit_rate] at h_profit
  simp at h_profit
  rw [h_cost_price, h_profit] at h_selling_price
  exact h_selling_price

end selling_price_of_book_l611_611610


namespace smallest_value_l611_611014

theorem smallest_value (a b c : ℕ) (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0)
    (h1 : a = 2 * b) (h2 : b = 2 * c) (h3 : 4 * c = a) :
    (Int.floor ((a + b : ℚ) / c) + Int.floor ((b + c : ℚ) / a) + Int.floor ((c + a : ℚ) / b)) = 8 := 
sorry

end smallest_value_l611_611014


namespace speed_of_train_is_78_kmph_l611_611253

variable (length_of_train length_of_tunnel : ℕ) (time_to_cross_tunnel : ℕ)

def convert_distance_to_kilometers (distance_in_meters : ℕ) : ℝ := distance_in_meters / 1000.0
def convert_time_to_hours (time_in_minutes : ℕ) : ℝ := time_in_minutes / 60.0

def total_distance_covered (length_of_train length_of_tunnel : ℕ) : ℕ := length_of_train + length_of_tunnel

def speed_of_train (length_of_train length_of_tunnel time_to_cross_tunnel : ℕ) : ℝ :=
  let distance_km := convert_distance_to_kilometers (total_distance_covered length_of_train length_of_tunnel)
  let time_hr := convert_time_to_hours time_to_cross_tunnel
  distance_km / time_hr

theorem speed_of_train_is_78_kmph
  (h1 : length_of_train = 800)
  (h2 : length_of_tunnel = 500)
  (h3 : time_to_cross_tunnel = 1) :
  speed_of_train length_of_train length_of_tunnel time_to_cross_tunnel = 78 :=
by
  sorry

end speed_of_train_is_78_kmph_l611_611253


namespace horner_value_v1_l611_611201

/-- Given the polynomial f(x) = x^6 - 5x^5 + 6x^4 + x^2 + 0.3x + 2 and x = -2, 
prove the value of v₁ using Horner's method is -7. -/
theorem horner_value_v1 (x : ℝ) 
  (f : ℝ → ℝ) 
  (Hf : f = λ x, x^6 - 5 * x^5 + 6 * x^4 + x^2 + 0.3 * x + 2) 
  (H_x : x = -2) : 
  let v₀ := 1,
      v₁ := v₀ * x - 5
  in v₁ = -7 :=
by {
  sorry
}

end horner_value_v1_l611_611201


namespace tan_arithmetic_sequence_product_sum_l611_611762

-- Angles forming the arithmetic sequence
variable (α β γ : ℝ)
variable (h_arith_seq : ∃ d, α = β - d ∧ γ = β + d ∧ d = π / 3)

-- The statement we want to prove
theorem tan_arithmetic_sequence_product_sum :
  (α = β - (π / 3)) → (γ = β + (π / 3)) →
  (tan α * tan β + tan β * tan γ + tan γ * tan α = -3) :=
by
  intro hα hγ
  have h1 : α = β - (π / 3) := hα
  have h2 : γ = β + (π / 3) := hγ
  sorry

end tan_arithmetic_sequence_product_sum_l611_611762


namespace actual_distance_l611_611896

-- Define the constants and conditions
def scaleFactor_cm_per_km := 0.6
def scaleDistance_km := 6.6
def mapDistance_cm := 80.5
def actualDistance_km := 887.17

-- State the theorem to be proved
theorem actual_distance {
  scale_cm_per_km : ℝ,
  scale_km : ℝ,
  map_cm : ℝ,
  actual_km : ℝ
} 
(h_scale : scale_cm_per_km = scaleFactor_cm_per_km) 
(h_scaleDist : scale_km = scaleDistance_km) 
(h_mapDist : map_cm = mapDistance_cm) 
(h_actualDist : actual_km = (mapDistance_cm * scaleDistance_km) / scaleFactor_cm_per_km):
actual_km = actualDistance_km :=
by 
  -- Extended the automatic prover should be able to prove it with proper tactics
  sorry

end actual_distance_l611_611896


namespace num_of_terms_in_arith_seq_l611_611391

-- Definitions of the conditions
def a : Int := -5 -- Start of the arithmetic sequence
def l : Int := 85 -- End of the arithmetic sequence
def d : Nat := 5  -- Common difference

-- The theorem that needs to be proved
theorem num_of_terms_in_arith_seq : (l - a) / d + 1 = 19 := sorry

end num_of_terms_in_arith_seq_l611_611391


namespace one_third_percent_of_150_l611_611970

theorem one_third_percent_of_150 : (1/3) * (150 / 100) = 0.5 := by
  sorry

end one_third_percent_of_150_l611_611970


namespace concatenated_room_probability_l611_611647

theorem concatenated_room_probability:
  let rooms := Finset.range 61 \ {0} -- Rooms are from 1 to 60
  let concat (a b : ℕ) : ℕ := a * 10^(Nat.log10 (b + 1)) + b -- Concatenates a and b
  let valid_rooms := Finset.range 361 -- Valid room numbers are from 1 to 360
  let total_combinations := rooms.card * (rooms.card - 1) -- Total possible room combinations
  let valid_combinations := rooms.sum (λ a, rooms.sum (λ b, if concat a b ∈ valid_rooms then 1 else 0))
  in (valid_combinations : ℚ) / total_combinations = 153 / 1180 := by
sorry

end concatenated_room_probability_l611_611647


namespace find_integer_l611_611577

theorem find_integer (x : ℕ) (h1 : (4 * x)^2 + 2 * x = 3528) : x = 14 := by
  sorry

end find_integer_l611_611577


namespace max_area_triangle_OPQ_l611_611678

open Real

variable (k : ℝ) (X Y P Q : ℝ × ℝ) (O : ℝ × ℝ := (0, 0))

-- Conditions
def condition_k := k > 1
def condition_X := X = (1, X.2) ∧ 0 < X.2
def condition_Y := Y = (1, Y.2) ∧ 0 < Y.2
def condition_perpendicular := (1, 0)
def condition_AY_AX := Y.2 = k * X.2
def condition_circle := dist O P = 1 ∧ dist O Q = 1
def condition_intersections := dist O X = dist O P ∧ dist O Y = dist O Q

-- Theorem statement
theorem max_area_triangle_OPQ :
  condition_k k →
  condition_X X →
  condition_Y Y →
  condition_AY_AX k X Y →
  condition_circle O P Q →
  condition_intersections X Y P Q →
  let area (A B C : ℝ × ℝ) : ℝ := 0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) in
  let max_area : ℝ := 0.5 * abs ((k - 1) / (k + 1)) in
  ∃ α β : ℝ, 
  area O P Q ≤ max_area :=
  by 
    intros;
    sorry

end max_area_triangle_OPQ_l611_611678


namespace taizhou_mock_test_2010_l611_611997

/-- The total amount raised (in yuan) expressed in scientific notation. -/
def amount_in_scientific_notation (amount : ℝ) : Prop :=
  ∃ (a : ℝ) (n : ℤ), 1 ≤ |a| ∧ |a| < 10 ∧ amount = a * 10^n

theorem taizhou_mock_test_2010 :
  amount_in_scientific_notation (2.175 * 10^9) :=
by
  use 2.175
  use 9
  simp
  split
  · norm_num
  · norm_num
  · norm_num

end taizhou_mock_test_2010_l611_611997


namespace number_of_proper_subsets_l611_611375
open Set

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {1, 2, 5}
def B : Set ℕ := {1, 3, 4}

theorem number_of_proper_subsets :
  card (powerset ((U \ A) ∩ B)) - 1 = 3 :=
by
  sorry

end number_of_proper_subsets_l611_611375


namespace pure_imaginary_number_solution_l611_611024

theorem pure_imaginary_number_solution (a : ℝ) 
  (h : ∃ x : ℝ, (a^2 - 3 * a + 2) + (a - 1) * complex.I = 0 + x * complex.I) : 
  a = 2 :=
sorry

end pure_imaginary_number_solution_l611_611024


namespace prove_midpoint_trajectory_eq_l611_611484

noncomputable def midpoint_trajectory_eq {x y : ℝ} (h : ∃ (x_P y_P : ℝ), (x_P^2 - y_P^2 = 1) ∧ (x = x_P / 2) ∧ (y = y_P / 2)) : Prop :=
  4*x^2 - 4*y^2 = 1

theorem prove_midpoint_trajectory_eq (x y : ℝ) (h : ∃ (x_P y_P : ℝ), (x_P^2 - y_P^2 = 1) ∧ (x = x_P / 2) ∧ (y = y_P / 2)) :
  midpoint_trajectory_eq h :=
sorry

end prove_midpoint_trajectory_eq_l611_611484


namespace negation_of_exists_l611_611945

theorem negation_of_exists {x : ℝ} :
  (¬ ∃ x : ℝ, x^2 - 2 ≤ 0) ↔ ∀ x : ℝ, x^2 - 2 > 0 :=
by
  sorry

end negation_of_exists_l611_611945


namespace candidate_a_votes_l611_611988

theorem candidate_a_votes (total_votes : ℕ) (invalid_percent : ℝ) (candidate_a_percent : ℝ)
  (Htotal_votes : total_votes = 560000)
  (Hinvalid_percent : invalid_percent = 0.15)
  (Hcandidate_a_percent : candidate_a_percent = 0.75) :
  (candidate_a_percent * (1 - invalid_percent) * total_votes = 357000) :=
by
  simp only [Htotal_votes, Hinvalid_percent, Hcandidate_a_percent]
  ring
  sorry

end candidate_a_votes_l611_611988


namespace smallest_sum_of_digits_l611_611183

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem smallest_sum_of_digits (N : ℕ) (hN_pos : 0 < N) 
  (h : sum_of_digits N = 3 * sum_of_digits (N + 1)) :
  sum_of_digits N = 12 :=
by {
  sorry
}

end smallest_sum_of_digits_l611_611183


namespace remainder_when_divided_by_x_minus_2_l611_611206

noncomputable def f (x : ℝ) : ℝ := x^4 - 3 * x^3 + 2 * x^2 + 11 * x - 6

theorem remainder_when_divided_by_x_minus_2 :
  (f 2) = 16 := by
  sorry

end remainder_when_divided_by_x_minus_2_l611_611206


namespace part1_part2_l611_611700

-- Part 1:
theorem part1 (a b : ℝ) (h1 : a > b > 0) (e : ℝ) (h2 : e = sqrt 2 / 2) (h3 : ∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 → (x - sqrt 2)^2 + y^2 ≤ (sqrt 2 + 2)^2):
  (∃ x y : ℝ, x^2 / 4 + y^2 / 2 = 1) :=
sorry

-- Part 2:
theorem part2 (a b : ℝ) (h1 : a > b > 0) (e : ℝ) (h2 : e = sqrt 2 / 2) (h3 : ∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 → (x - sqrt 2)^2 + y^2 ≤ (sqrt 2 + 2)^2) (h4 : ∀ x y : ℝ, x^2 + y^2 = 4 / 3) :
  (∀ A B : ℝ × ℝ, tangent_line_of_circle (x^2 + y^2 = 4 / 3) intersects_ellipse (x^2 / 4 + y^2 / 2 = 1) at A B → |(A.1 - B.1, A.2 - B.2)| ≤ 6) :=
sorry

end part1_part2_l611_611700


namespace circus_tent_capacity_l611_611961

theorem circus_tent_capacity (num_sections : ℕ) (people_per_section : ℕ) 
  (h1 : num_sections = 4) (h2 : people_per_section = 246) :
  num_sections * people_per_section = 984 :=
by
  sorry

end circus_tent_capacity_l611_611961


namespace fishing_tomorrow_l611_611774

theorem fishing_tomorrow 
  (P1 : ∀ day : ℕ, 7 ∈ {p | p goes fishing on day})
  (P2 : ∀ day : ℕ, day % 2 = 0 → 8 ∈ {p | p goes fishing on day})
  (P3 : ∀ day : ℕ, day % 3 = 0 → 3 ∈ {p | p goes fishing on day})
  (P4 : ℕ)
  (yesterday : ℕ)
  (today : ℕ)
  (tomorrow : ℕ)
  (hyesterday : yesterday = 12)
  (htoday : today = 10)
  : tomorrow = 15 := by
  sorry

end fishing_tomorrow_l611_611774


namespace people_believing_mostly_purple_l611_611590

theorem people_believing_mostly_purple :
  ∀ (total : ℕ) (mostly_pink : ℕ) (both_mostly_pink_purple : ℕ) (neither : ℕ),
  total = 150 →
  mostly_pink = 80 →
  both_mostly_pink_purple = 40 →
  neither = 25 →
  (total - neither + both_mostly_pink_purple - mostly_pink) = 85 :=
by
  intros total mostly_pink both_mostly_pink_purple neither h_total h_mostly_pink h_both h_neither
  have people_identified_without_mostly_purple : ℕ := mostly_pink + both_mostly_pink_purple - mostly_pink + neither
  have leftover_people : ℕ := total - people_identified_without_mostly_purple
  have people_mostly_purple := both_mostly_pink_purple + leftover_people
  suffices people_mostly_purple = 85 by sorry
  sorry

end people_believing_mostly_purple_l611_611590


namespace homes_with_panels_installed_l611_611290

-- Let's define the conditions as constants
def total_homes : ℕ := 20
def panels_per_home : ℕ := 10
def panels_less_delivered : ℕ := 50

-- State the problem as a theorem
theorem homes_with_panels_installed :
  let total_panels_required := total_homes * panels_per_home,
      panels_delivered := total_panels_required - panels_less_delivered,
      homes_installed := panels_delivered / panels_per_home in
  homes_installed = 15 :=
by
  let total_panels_required := total_homes * panels_per_home
  let panels_delivered := total_panels_required - panels_less_delivered
  let homes_installed := panels_delivered / panels_per_home
  show homes_installed = 15
  sorry

end homes_with_panels_installed_l611_611290


namespace cube_surface_area_and_diagonal_l611_611672

theorem cube_surface_area_and_diagonal (V : ℝ) (hV : V = 343) :
  ∃ s SA d : ℝ, s = ∛343 ∧ SA = 6 * s^2 ∧ d = s * √3 ∧ SA = 294 ∧ d ≈ 12.124 :=
by
  use ∛343 -- side length
  have s_def : ∛343 = 7 := by norm_num
  simp [s_def] at *

  use 6 * (7 ^ 2) -- surface area
  have SA_def : 6 * (7 ^ 2) = 294 := by norm_num
  simp [SA_def] at *

  use 7 * √3 -- space diagonal
  have d_def : 7 * √3 ≈ 12.124 := by norm_num -- Note: approximation handled with norm_num
  simp [d_def] at *

  exact ⟨s_def, SA_def, d_def⟩

end cube_surface_area_and_diagonal_l611_611672


namespace mike_games_l611_611891

theorem mike_games (init_money spent_money game_cost : ℕ) (h1 : init_money = 42) (h2 : spent_money = 10) (h3 : game_cost = 8) :
  (init_money - spent_money) / game_cost = 4 :=
by
  sorry

end mike_games_l611_611891


namespace new_energy_vehicle_profit_and_cost_effectiveness_l611_611093

/-- New energy vehicle problem -/
theorem new_energy_vehicle_profit_and_cost_effectiveness:
  (∀ n : ℕ, 1 ≤ n ∧ n ≤ 8 → 5.25 * n - (0.25 * n ^ 2 + 0.25 * n) - 9 > 0 → n = 3) ∧
  (∀ n : ℕ, 1 ≤ n ∧ n ≤ 8 → 
    let avg_profit := -0.25 * (n + (36 / n) - 20) in 
    avg_profit = 2 → n = 6) :=
sorry

end new_energy_vehicle_profit_and_cost_effectiveness_l611_611093


namespace least_positive_integer_divisors_l611_611942

theorem least_positive_integer_divisors (n m k : ℕ) (h₁ : n = m * 6^k)
  (h₂ : 6 ∣ m → False)
  (h₃ : ∀ p : ℕ, p.prime → ∏ (d in (n.divisors.filter prime), (p + 1)) = 2021) :
  m + k = 58 := 
sorry

end least_positive_integer_divisors_l611_611942


namespace minimum_lambda_l611_611368

noncomputable def f (a x : ℝ) := a * Real.log x + (1 / 2) * x^2 - a * x

theorem minimum_lambda (a : ℝ) (h : a > 4) (x₁ x₂ : ℝ) (hx₁ : x₁ = (a - Real.sqrt(a^2 - 4 * a)) / 2) (hx₂ : x₂ = (a + Real.sqrt(a^2 - 4 * a)) / 2) :
  (f a x₁ + f a x₂) / (x₁ + x₂) < Real.log 4 - 3 :=
sorry

end minimum_lambda_l611_611368


namespace studios_total_l611_611547

section

variable (s1 s2 s3 : ℕ)

theorem studios_total (h1 : s1 = 110) (h2 : s2 = 135) (h3 : s3 = 131) : s1 + s2 + s3 = 376 :=
by
  sorry

end

end studios_total_l611_611547


namespace people_left_second_hour_l611_611241

theorem people_left_second_hour
  (came_in_first_hour : ℕ)
  (left_first_hour : ℕ)
  (came_in_second_hour : ℕ)
  (total_after_two_hours : ℕ)
  (num_first_hour : came_in_first_hour - left_first_hour = 67)
  (num_second_hour : 67 + came_in_second_hour = 85)
  (total_people : 85 - total_after_two_hours = 9) :
  came_in_first_hour = 94 ∧ left_first_hour = 27 ∧ came_in_second_hour = 18 ∧ total_after_two_hours = 76 :=
begin
  sorry
end

end people_left_second_hour_l611_611241


namespace obtuse_triangle_acute_triangle_if_dot_product_circumcenter_statement_vector_ratio_l611_611860

variables (A B C : ℝ) (a b c : ℝ)
variables (vector_a vector_b : ℝ × ℝ)
variable (O : Type) [Inhabited O]
variable (OA OB OC AO BC : ℝ)
variable (O_circumcenter : O)

-- Define the conditions
def triangle_sides : Prop := a = opposite_angle_A ∧ b = opposite_angle_B ∧ c = opposite_angle_C
def vector_a_def : Prop := vector_a = (tan A + tan B, tan C)
def vector_b_def : Prop := vector_b = (1, 1)
def dot_product_condition : Prop := vector_a.1 * vector_b.1 + vector_a.2 * vector_b.2 > 0
def circumcenter_O : Prop := is_circumcenter O_circumcenter
def condition_vec_AO_BC : Prop := (AO * BC = (1 / 2) * (b^2 - c^2))
def condition_sin_squared : Prop := sin A ^ 2 + sin B ^ 2 = sin C ^ 2
def condition_vec_sum_zero : Prop := OA + OB + OC = 0

-- Proof problem statements
theorem obtuse_triangle (h1 : cos B * cos C > sin B * sin C) : triangle_obtuse A B C :=
sorry

theorem acute_triangle_if_dot_product
  (h2 : dot_product_condition vector_a vector_b) : triangle_acute A B C :=
sorry

theorem circumcenter_statement (h3 : circumcenter_O O_circumcenter) : AO * BC = (1 / 2) * (b^2 - c^2) :=
sorry

theorem vector_ratio (h4 : condition_sin_squared A B C)
  (h5 : condition_vec_sum_zero OA OB OC) : (|OA| ^ 2 + |OB| ^ 2) / |OC| ^ 2 = 5 :=
sorry

end obtuse_triangle_acute_triangle_if_dot_product_circumcenter_statement_vector_ratio_l611_611860


namespace payments_option1_option2_option1_more_effective_combined_option_cost_l611_611600

variable {x : ℕ}

-- Condition 1: Prices and discount options
def badminton_rackets_price : ℕ := 40
def shuttlecocks_price : ℕ := 10
def discount_option1_free_shuttlecocks (pairs : ℕ): ℕ := pairs
def discount_option2_price (price : ℕ) : ℕ := price * 9 / 10

-- Condition 2: Buying requirements
def pairs_needed : ℕ := 10
def shuttlecocks_needed (n : ℕ) : ℕ := n
axiom x_gt_10 : x > 10

-- Proof Problem 1: Payment calculations
theorem payments_option1_option2 (x : ℕ) (h : x > 10) :
  (shuttlecocks_price * (shuttlecocks_needed x - discount_option1_free_shuttlecocks pairs_needed) + badminton_rackets_price * pairs_needed =
    10 * x + 300) ∧
  (discount_option2_price (shuttlecocks_price * shuttlecocks_needed x + badminton_rackets_price * pairs_needed) =
    9 * x + 360) :=
sorry

-- Proof Problem 2: More cost-effective option when x=30
theorem option1_more_effective (x : ℕ) (h : x = 30) :
  (10 * x + 300 < 9 * x + 360) :=
sorry

-- Proof Problem 3: Another cost-effective method when x=30
theorem combined_option_cost (x : ℕ) (h : x = 30) :
  (badminton_rackets_price * pairs_needed + discount_option2_price (shuttlecocks_price * (shuttlecocks_needed x - 10)) = 580) :=
sorry

end payments_option1_option2_option1_more_effective_combined_option_cost_l611_611600


namespace extreme_points_inequality_l611_611364

def f (x a : ℝ) := 1 / x - x + a * log x

theorem extreme_points_inequality (a x1 x2 : ℝ) (ha : a > 2) (hx1 : x1 > 0) (hx2 : x2 > 0) 
  (extreme : (-x1^2 + a * x1 - 1 = 0) ∧ (-x2^2 + a * x2 - 1 = 0)) 
  (hx1x2 : x1 * x2 = 1) :
  (f x1 a - f x2 a) / (x1 - x2) < a - 2 := 
sorry

end extreme_points_inequality_l611_611364


namespace expression_value_l611_611207

-- Proving the value of the expression using the factorial and sum formulas
theorem expression_value :
  (Nat.factorial 10) / (10 * 11 / 2) = 66069 := 
sorry

end expression_value_l611_611207


namespace simplify_expression_l611_611500

noncomputable def verify_expression : Prop :=
  let deg_to_rad := (fun (deg : ℝ) => deg * (real.pi / 180))
  let sin := real.sin ∘ deg_to_rad
  let cos := real.cos ∘ deg_to_rad
  (sin 10 > 0) ∧ (cos 10 > sin 10) → 
  (sqrt (1 - 2 * sin 10 * cos 10) / (cos 10 - sqrt (1 - cos 170 * cos 170))) = real.tan (deg_to_rad 10)
  
theorem simplify_expression : verify_expression :=
  by sorry

end simplify_expression_l611_611500


namespace tourists_count_l611_611240

theorem tourists_count (n k : ℕ) (h1 : 2 * n % k = 1) (h2 : 3 * n % k = 13) : k = 23 := 
sorry

end tourists_count_l611_611240


namespace cost_price_percentage_respect_to_selling_price_l611_611156

-- Define the given constants and variables
def SP : ℝ := sorry -- Selling price, can be any positive real number
def CP : ℝ := sorry -- Cost price, to be determined in terms of SP

-- Given condition: profit percentage
def profit_percentage : ℝ := 5.263157894736842 / 100

-- Relation between selling price, cost price, and profit
def profit : ℝ := SP * profit_percentage
def profit_equation := SP - CP = profit

-- The percentage of cost price with respect to selling price
def percentage_CP_SP : ℝ := (CP / SP) * 100 

-- The goal statement in Lean
theorem cost_price_percentage_respect_to_selling_price :
  percentage_CP_SP = 94.73684210526316 := by
  sorry

end cost_price_percentage_respect_to_selling_price_l611_611156


namespace line_EF_passes_through_midpoint_K_l611_611071

/-- Let \( H \) be the orthocenter of \( \triangle ABC \), and let \( P \) be a point on the circumcircle of \( \triangle ABC \).
Perpendiculars are drawn from \( P \) to the extensions of \( AB \) and \( AC \) with feet \( E \) and \( F \) respectively.
Prove that the line \( EF \) passes through the midpoint \( K \) of the segment \( PH \). -/
theorem line_EF_passes_through_midpoint_K
  (A B C H P E F K : Point)
  (orthocenter_H : is_orthocenter A B C H)
  (on_circumcircle_P : is_on_circumcircle A B C P)
  (perpendicular_PE : is_perpendicular P E (line AB).extension)
  (perpendicular_PF : is_perpendicular P F (line AC).extension)
  (midpoint_K : is_midpoint K P H) :
  passes_through (line EF) K := 
begin
  sorry
end

end line_EF_passes_through_midpoint_K_l611_611071


namespace brownies_left_l611_611193

def brownies_initial := 24.0
def tina_lunch_days := 5
def tina_lunch_per_day := 1.5
def tina_dinner_days := 5
def tina_dinner_per_day := 0.5
def husband_days := 5
def husband_per_day := 0.75
def guests_days := 2
def guests_per_day := 2.5

theorem brownies_left :
  let tina_total_lunch := tina_lunch_days * tina_lunch_per_day in
  let tina_total_dinner := tina_dinner_days * tina_dinner_per_day in
  let tina_total := tina_total_lunch + tina_total_dinner in
  let husband_total := husband_days * husband_per_day in
  let guests_total := guests_days * guests_per_day in
  let total_consumed := tina_total + husband_total + guests_total in
  brownies_initial - total_consumed = 5.25 :=
by
  sorry

end brownies_left_l611_611193


namespace midpoint_locus_l611_611548

noncomputable def midpoint_of_segment (A A' B B' C C' : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let M := (A.1 / 3, B.1 / 3, C.1 / 3)
  let M' := (A'.1 / 3, B'.1 / 3, C'.1 / 3)
  let S := ((M.1 + M'.1) / 2, (M.2 + M'.2) / 2, (M.3 + M'.3) / 2)
  S

theorem midpoint_locus (A A' B B' C C' : ℝ × ℝ × ℝ)
  (hA : A = (a, 0, 0)) (hA' : A' = (a', 0, 0)) 
  (hB : B = (0, b, 0)) (hB' : B' = (0, b', 0)) 
  (hC : C = (0, 0, c)) (hC' : C' = (0, 0, c')):
  midpoint_of_segment A A' B B' C C' = (x / 3, y / 3, z / 3) :=
by
  sorry

end midpoint_locus_l611_611548


namespace largest_digit_B_divisible_by_3_l611_611857

theorem largest_digit_B_divisible_by_3 :
  ∃ (B : ℕ), B ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ (4 + B + 6 + 8 + 2 + 5 + 1) % 3 = 0 ∧
    (∀ B' ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
      (4 + B' + 6 + 8 + 2 + 5 + 1) % 3 = 0 → B' ≤ B) ∧
    B = 7 :=
begin
  sorry,
end

end largest_digit_B_divisible_by_3_l611_611857


namespace problem_2012_square_eq_x_square_l611_611571

theorem problem_2012_square_eq_x_square : 
  ∃ x : ℤ, (2012 + x)^2 = x^2 ∧ x = -1006 :=
by {
  existsi (-1006 : ℤ),
  split,
  sorry,
  refl
}

end problem_2012_square_eq_x_square_l611_611571


namespace functional_relationship_w_x_selling_price_for_200_profit_maximize_daily_profit_l611_611239

-- Define the cost price constant
def cost_price : ℝ := 30

-- Define the daily sales quantity function
def daily_sales (x : ℝ) : ℝ := -x + 60

-- Define the profit function
def profit_per_unit (x : ℝ) : ℝ := x - cost_price

-- Define the daily profit function
def daily_profit (x : ℝ) : ℝ := profit_per_unit x * daily_sales x

-- Prove the functional relationship between w and x
theorem functional_relationship_w_x :
  ∀ x : ℝ, daily_profit x = -x^2 + 90 * x - 1800 :=
begin
  assume x,
  unfold daily_profit,
  unfold profit_per_unit,
  unfold daily_sales,
  calc (x - 30) * (-x + 60) = -x^2 + 60 * x - 30 * x + 1800 : by ring
end

-- Prove the selling price for given conditions
theorem selling_price_for_200_profit :
  ∃ x : ℝ, daily_profit x = 200 ∧ x ≤ 48 ∧ x = 40 :=
begin
  use 40,
  split,
  { 
    unfold daily_profit,
    unfold profit_per_unit,
    unfold daily_sales,
    calc (40 - 30) * (-40 + 60) = 200 : by norm_num 
  },
  {
    split,
    { linarith },
    { refl }
  }
end

-- Prove the selling price to maximize profit and the maximum profit
theorem maximize_daily_profit :
  ∃ x w : ℝ, daily_profit x = w ∧ x = 45 ∧ w = 225 :=
begin
  use 45,
  use 225,
  split,
  {
    unfold daily_profit,
    unfold profit_per_unit,
    unfold daily_sales,
    calc (45 - 30) * (-45 + 60) = 225 : by norm_num 
  },
  {
    split,
    { refl },
    { refl }
  }
end

end functional_relationship_w_x_selling_price_for_200_profit_maximize_daily_profit_l611_611239


namespace karsyn_total_payment_l611_611055

def initial_price : ℝ := 600
def discount_rate : ℝ := 0.20
def phone_case_cost : ℝ := 25
def screen_protector_cost : ℝ := 15
def store_discount_rate : ℝ := 0.05
def sales_tax_rate : ℝ := 0.035

noncomputable def total_payment : ℝ :=
  let discounted_price := discount_rate * initial_price
  let total_cost := discounted_price + phone_case_cost + screen_protector_cost
  let store_discount := store_discount_rate * total_cost
  let discounted_total := total_cost - store_discount
  let tax := sales_tax_rate * discounted_total
  discounted_total + tax

theorem karsyn_total_payment : total_payment = 157.32 := by
  sorry

end karsyn_total_payment_l611_611055


namespace fishing_tomorrow_l611_611798

theorem fishing_tomorrow (every_day_fishers every_other_day_fishers every_three_days_fishers fishers_yesterday fishers_today : ℕ) :
  (every_day_fishers = 7) →
  (every_other_day_fishers = 8) →
  (every_three_days_fishers = 3) →
  (fishers_yesterday = 12) →
  (fishers_today = 10) →
  (every_three_days_fishers + every_day_fishers + (every_other_day_fishers - (fishers_yesterday - every_day_fishers)) = 15) :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  simp
  sorry

end fishing_tomorrow_l611_611798


namespace parabola_equation_l611_611371

variable (p : ℝ) (h_pos : p > 0)
variable (F : ℝ × ℝ := (0, p / 2))
variable (M : ℝ × ℝ)
variable (h_M_on_C : M = (sqrt (6 * p), p / 4))
variable (h_distance_OM : sqrt ((M.1)^2 + (M.2)^2) = 3)
variable (h_distance_MF : sqrt ((M.1 - 0)^2 + (M.2 - (p / 2))^2) = 3)

theorem parabola_equation : (2 * p = 8) := 
by {
  sorry
}

end parabola_equation_l611_611371


namespace ram_distance_approx_l611_611479

def distance_between_mountains_map : ℝ := 312 -- inches
def distance_between_mountains_actual : ℝ := 136 -- kilometers
def ram_location_map : ℝ := 28 -- inches

def map_scale : ℝ := distance_between_mountains_actual / distance_between_mountains_map

noncomputable def ram_actual_distance := ram_location_map * map_scale

theorem ram_distance_approx :
  abs (ram_actual_distance - 12.205) < 0.001 :=
by
  sorry

end ram_distance_approx_l611_611479


namespace number_of_envelopes_l611_611260

theorem number_of_envelopes (total_weight_grams : ℕ) (weight_per_envelope_grams : ℕ) (n : ℕ) :
  total_weight_grams = 7480 ∧ weight_per_envelope_grams = 8500 ∧ n = 880 → total_weight_grams = n * weight_per_envelope_grams := 
sorry

end number_of_envelopes_l611_611260


namespace minimum_area_right_triangle_l611_611881

theorem minimum_area_right_triangle (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  (∃ a b : ℝ, (a > 0 ∧ b > 0) ∧ A = \frac{1}{2} * a * b ∧ (x/a) + (y/b) = 1) -> A = \frac{1}{2} * x * y :=
sorry

end minimum_area_right_triangle_l611_611881


namespace tree_initial_height_l611_611575

noncomputable def initial_tree_height (H : ℝ) (growth_rate : ℝ) (years : ℕ) : ℝ := 
  H + growth_rate * years

theorem tree_initial_height :
  ∀ (H : ℝ), 
  (∀ (years : ℕ), ∃ h : ℝ, h = initial_tree_height H 0.5 years) →
  initial_tree_height H 0.5 6 = initial_tree_height H 0.5 4 * (7 / 6) →
  H = 4 :=
by
  intro H height_increase condition
  sorry

end tree_initial_height_l611_611575


namespace value_of_k_for_square_of_binomial_l611_611574

theorem value_of_k_for_square_of_binomial (a k : ℝ) : (x : ℝ) → x^2 - 14 * x + k = (x - a)^2 → k = 49 :=
by
  intro x h
  sorry

end value_of_k_for_square_of_binomial_l611_611574


namespace general_eq_C1_cartesian_eq_C2_range_m_l611_611031

section Problem1

variable (θ : ℝ) (x y ρ m : ℝ)

-- Conditions for curve C1
def C1_parametric_eqs (θ : ℝ) (x y : ℝ) : Prop :=
  x = 2 + 2 * Real.cos θ ∧ y = 2 * Real.sin θ ∧ θ ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2)

-- Condition for curve C2 in polar coordinates
def C2_polar_eq (θ : ℝ) (ρ : ℝ) : Prop :=
  θ = Real.pi / 4

-- Translated curve C2
def C2_translated_eq (m : ℝ) (x y : ℝ) : Prop :=
  y = x - m

-- Proving the general equation of C1
theorem general_eq_C1 (C1_parametric_eqs θ x y) : (x - 2)^2 + y^2 = 4 :=
sorry

-- Proving the Cartesian equation of C2
theorem cartesian_eq_C2 (C2_polar_eq θ ρ) : y = x :=
sorry

-- Proving the range of m for exactly two common points with C1
theorem range_m (C1_parametric_eqs θ x y) (C2_translated_eq m x y) : 4 ≤ m ∧ m < 2 + 2 * Real.sqrt 2 :=
sorry

end Problem1

end general_eq_C1_cartesian_eq_C2_range_m_l611_611031


namespace determine_omega_l611_611367

noncomputable def function_zero_and_symmetry (ω ϕ : ℝ) : Prop :=
  f : ℝ → ℝ := λ x, Real.sin (ω * x + ϕ),
  ∀ x, f (-(π / 4)) = 0 ∧ (f (π / 3) = f (- (π / 3)))

theorem determine_omega (ω ϕ : ℝ) (hω : 0 < ω ∧ ω < 3) (hϕ : 0 < ϕ ∧ ϕ < π)
  (hzero : function_zero_and_symmetry ω ϕ)
  : ω = 6 / 7 :=
sorry

end determine_omega_l611_611367


namespace remainder_of_product_mod_5_l611_611568

theorem remainder_of_product_mod_5 :
  (2685 * 4932 * 91406) % 5 = 0 :=
by
  sorry

end remainder_of_product_mod_5_l611_611568


namespace bounded_function_inequality_equality_case_l611_611465

variable {D : Type*} [inhabited D]
variable {f : D → ℝ}
variable {M : ℝ}
variable {n : ℕ}
variable {x : Fin n → D}

def bounded_function_on_domain (M : ℝ) (f : D → ℝ) (x : D) : Prop :=
  |f x| ≤ M

theorem bounded_function_inequality (h_bound : ∀ x ∈ (Finset.univ : Finset (Fin n)), bounded_function_on_domain M f (x x)) :
  (n - 1) * M ^ n + ∏ i : Fin n, f (x i) ≥ M ^ (n - 1) * ∑ i : Fin n, f (x i) :=
sorry

theorem equality_case (h_bound : ∀ x ∈ (Finset.univ : Finset (Fin n)), bounded_function_on_domain M f (x x)) :
  (∀ i : Fin n, f (x i) = M) ↔ (n - 1) * M ^ n + ∏ i : Fin n, f (x i) = M ^ (n - 1) * ∑ i : Fin n, f (x i) :=
sorry

end bounded_function_inequality_equality_case_l611_611465


namespace arithmetic_square_root_of_solution_sum_l611_611982

variables (m n t : ℤ)

theorem arithmetic_square_root_of_solution_sum :
  (m * (7 / 2 : ℚ) + (-2) = 5) →
  ((2 * (7 / 2 : ℚ)) - n * (-2) = 13) →
  (3 * m + (-7) = 5) →
  ((t ≤ m + 1) ∧ (t > n)) →
  (√(4 + 5) = 3) :=
by
  sorry

end arithmetic_square_root_of_solution_sum_l611_611982


namespace usual_time_to_catch_bus_l611_611220

variable {S T T' D : ℝ}

theorem usual_time_to_catch_bus (h1 : D = S * T)
  (h2 : D = (4 / 5) * S * T')
  (h3 : T' = T + 4) : T = 16 := by
  sorry

end usual_time_to_catch_bus_l611_611220


namespace largest_angle_in_ratio_3_4_5_l611_611168

theorem largest_angle_in_ratio_3_4_5 (x : ℝ) (h1 : 3 * x + 4 * x + 5 * x = 180) : 5 * x = 75 :=
by
  sorry

end largest_angle_in_ratio_3_4_5_l611_611168


namespace sequences_proposition_correctness_l611_611050

variable {a_n : ℕ → ℝ} (n : ℕ)

-- Conditions of the problem
def S (n : ℕ) : ℝ :=
  (n * (a_n 1 + a_n n)) / 2

-- Given inequalities
variable (h₁ : S 5 > S 6) (h₂ : S 6 > S 4)

theorem sequences_proposition_correctness :
  (a_n 6 - a_n 5 < 0) ∧ (S 10 > 0) ∧ (S 11 < 0) :=
by
  sorry

end sequences_proposition_correctness_l611_611050


namespace total_distance_12_hours_l611_611989

-- Define the initial conditions for the speed and distance calculation
def speed_increase : ℕ → ℕ
  | 0 => 50
  | n + 1 => speed_increase n + 2

def distance_in_hour (n : ℕ) : ℕ := speed_increase n

-- Define the total distance traveled in 12 hours
def total_distance (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | n + 1 => total_distance n + distance_in_hour n

theorem total_distance_12_hours :
  total_distance 12 = 732 := by
  sorry

end total_distance_12_hours_l611_611989


namespace incorrect_monotonicity_l611_611160

noncomputable def f (x φ : ℝ) : ℝ := 2 * (Real.sin (2 * x + (Real.pi / 3) + φ))
def symmetric_about_y_axis (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem incorrect_monotonicity {φ : ℝ} (hφ : |φ| < Real.pi / 2) :
  (symmetric_about_y_axis (λ x, f x φ)) →
    (∃ x, ((-3 * Real.pi / 4) < x ∧ x < (-Real.pi / 4)) → ¬MonotoneOn (λ x, f x φ) (-3 * Real.pi / 4, -Real.pi / 4)) :=
by sorry

#eval incorrect_monotonicity

end incorrect_monotonicity_l611_611160


namespace max_value_of_y_l611_611726

noncomputable def max_value_expression
  (a b e : ℝ × ℝ)
  (h_a_norm : ∥a∥ = 1)
  (h_b_norm : ∥b∥ = 2)
  (h_a_b_dot : a.1 * b.1 + a.2 * b.2 = 1)
  (h_e_unit : ∥e∥ = 1) :
  real :=
  a.1 * e.1 + a.2 * e.2 + b.1 * e.1 + b.2 * e.2

theorem max_value_of_y
  (a b e : ℝ × ℝ)
  (h_a_norm : ∥a∥ = 1)
  (h_b_norm : ∥b∥ = 2)
  (h_a_b_dot : a.1 * b.1 + a.2 * b.2 = 1)
  (h_e_unit : ∥e∥ = 1) :
  max_value_expression a b e h_a_norm h_b_norm h_a_b_dot h_e_unit ≤ real.sqrt 7 :=
sorry

end max_value_of_y_l611_611726


namespace sufficient_but_not_necessary_condition_l611_611993

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x > 2 → x^2 + 2 * x - 8 > 0) ∧ (¬(x > 2) → ¬(x^2 + 2 * x - 8 > 0)) → false :=
by 
  sorry

end sufficient_but_not_necessary_condition_l611_611993


namespace find_x_l611_611573

theorem find_x (x : ℝ) (h : (2012 + x)^2 = x^2) : x = -1006 :=
by
  sorry

end find_x_l611_611573


namespace sum_and_gap_l611_611759

-- Define the gap condition
def gap_condition (x : ℝ) : Prop :=
  |5.46 - x| = 3.97

-- Define the main theorem to be proved 
theorem sum_and_gap :
  ∀ (x : ℝ), gap_condition x → x < 5.46 → x + 5.46 = 6.95 := 
by 
  intros x hx hlt
  sorry

end sum_and_gap_l611_611759


namespace jessica_withdraw_fraq_l611_611862

theorem jessica_withdraw_fraq {B : ℝ} (h : B - 200 + (1 / 2) * (B - 200) = 450) :
  (200 / B) = 2 / 5 := by
  sorry

end jessica_withdraw_fraq_l611_611862


namespace n_squared_plus_d_not_perfect_square_l611_611880

theorem n_squared_plus_d_not_perfect_square (n d : ℕ) (h1 : n > 0)
  (h2 : d > 0) (h3 : d ∣ 2 * n^2) : ¬ ∃ x : ℕ, n^2 + d = x^2 := 
sorry

end n_squared_plus_d_not_perfect_square_l611_611880


namespace fishing_tomorrow_l611_611820

theorem fishing_tomorrow (yesterday_fishers today_fishers : ℕ)
  (every_day_fishers every_other_day_fishers every_three_days_fishers : ℕ)
  (total_population : ℕ):
  yesterday_fishers = 12 → 
  today_fishers = 10 → 
  every_day_fishers = 7 → 
  every_other_day_fishers = 8 → 
  every_three_days_fishers = 3 → 
  total_population = yesterday_fishers + today_fishers + (total_population - (every_day_fishers + every_other_day_fishers + every_three_days_fishers)) →
  ∃ tomorrow_fishers : ℕ, tomorrow_fishers = 15 :=
by {
  -- This is a statement definition, the proof is not required and thus marked as "sorry:"
  sorry
}

end fishing_tomorrow_l611_611820


namespace num_divisors_gcd_90_100_l611_611152

noncomputable def num_divisors (n : ℕ) : ℕ := (List.range (n + 1)).filter (λ d => n % d = 0).length

theorem num_divisors_gcd_90_100 : num_divisors (Nat.gcd 90 100) = 3 :=
by 
  have h_gcd := Nat.gcd_eq_right (by decide : 90 % 100 = 90)
  rw [h_gcd]
  have h_fact_90 : Nat.factors 90 = [2, 3, 3, 5] := by decide
  have h_fact_100 : Nat.factors 100 = [2, 2, 5, 5] := by decide
  rw [Nat.factors_gcd, h_fact_90, h_fact_100]
  apply List.length_eq_of_forall_mem _
  repeat { intro x, rintro rfl, simp }

end num_divisors_gcd_90_100_l611_611152


namespace sum_of_three_numbers_l611_611943

theorem sum_of_three_numbers (a b c : ℕ) (mean_least difference greatest_diff : ℕ)
  (h1 : a ≤ b) (h2 : b ≤ c)
  (h3 : mean_least = 8) (h4 : greatest_diff = 25)
  (h5 : c - a = 26)
  (h6 : (a + b + c) / 3 = a + mean_least) 
  (h7 : (a + b + c) / 3 = c - greatest_diff) : 
a + b + c = 81 := 
sorry

end sum_of_three_numbers_l611_611943


namespace length_AE_calculation_l611_611036

noncomputable def prove_length_AE (AF CE ED : ℝ) (Area : ℝ) : ℝ :=
  let height := (AF + CE) / 2
  let y := ED
  let equation := Area = (1 / 2) * (x + y) * height
  let x_solution := (Area / (1 / 2 * height)) - y
  x_solution

theorem length_AE_calculation : prove_length_AE 30 40 50 7200 ≈ 361.43 := 
by
  sorry

end length_AE_calculation_l611_611036


namespace lcm_two_numbers_l611_611411

theorem lcm_two_numbers (a b : ℕ) (h1 : a * b = 17820) (h2 : Nat.gcd a b = 12) : Nat.lcm a b = 1485 := 
by
  sorry

end lcm_two_numbers_l611_611411


namespace white_cells_even_l611_611847

theorem white_cells_even (board : Fin 1998 → Fin 2002 → Bool)
  (h_row : ∀ i : Fin 1998, odd (board i).count (λ j, board i j = true))
  (h_col : ∀ j : Fin 2002, odd (board j).count (λ i, board i j = true)) :
  even ((board.enum.filter (λ p, is_white p.1.1 p.1.2 ∧ p.2)).length) := sorry

def is_white (i j : Nat) : Bool :=
  (i % 2 = j % 2)

end white_cells_even_l611_611847


namespace fishers_tomorrow_l611_611803

-- Define the groups of fishers and their fishing pattern
def everyday_fishers := 7
def every_other_day_fishers := 8
def every_three_days_fishers := 3

-- Given counts for yesterday and today
def fishers_yesterday := 12
def fishers_today := 10

-- The problem to prove: 15 people will fish tomorrow
theorem fishers_tomorrow : 
  (fishers_yesterday = 12) →
  (fishers_today = 10) →
  (everyday_fishers = 7) →
  (every_other_day_fishers = 8) →
  (every_three_days_fishers = 3) →
  15 = everyday_fishers + every_other_day_fishers / 2 + every_three_days_fishers / 3 :=
begin
  sorry
end

end fishers_tomorrow_l611_611803


namespace number_of_rectangles_l611_611021

theorem number_of_rectangles (m n : ℕ) (h1 : m = 8) (h2 : n = 10) : (m - 1) * (n - 1) = 63 := by
  sorry

end number_of_rectangles_l611_611021


namespace question1_question2_l611_611736

variables (a b c : EuclideanSpace ℝ (Fin 2))

def a : EuclideanSpace ℝ (Fin 2) := ![1, 0]
def b : EuclideanSpace ℝ (Fin 2) := ![-1, 2]

-- For question (1)
def c : EuclideanSpace ℝ (Fin 2) := ![c1, c2]
def d := a - b

theorem question1 (hc : ∥c∥ = 1) (hc_parallel : ∃ k : ℝ, c = k • d) : 
  c = ![sqrt(2)/2, -sqrt(2)/2] ∨ c = ![-sqrt(2)/2, sqrt(2)/2] :=
sorry

-- For question (2)
variables (t : ℝ)

def v1 : EuclideanSpace ℝ (Fin 2) := 2 * t • a - b
def v2 : EuclideanSpace ℝ (Fin 2) := 3 • a + t • b

theorem question2 (h_perp : inner v1 v2 = 0) :
  t = -1 ∨ t = 3 / 2 :=
sorry

end question1_question2_l611_611736


namespace transmission_time_estimation_l611_611287

noncomputable def number_of_blocks := 80
noncomputable def chunks_per_block := 640
noncomputable def transmission_rate := 160 -- chunks per second
noncomputable def seconds_per_minute := 60
noncomputable def total_chunks := number_of_blocks * chunks_per_block
noncomputable def total_time_seconds := total_chunks / transmission_rate
noncomputable def total_time_minutes := total_time_seconds / seconds_per_minute

theorem transmission_time_estimation : total_time_minutes = 5 := 
  sorry

end transmission_time_estimation_l611_611287


namespace original_distance_cycled_l611_611614

theorem original_distance_cycled
  (x t d : ℝ)
  (h1 : d = x * t)
  (h2 : d = (x + 1/4) * (3/4 * t))
  (h3 : d = (x - 1/4) * (t + 3)) :
  d = 4.5 := 
sorry

end original_distance_cycled_l611_611614


namespace complex_number_problem_l611_611753

-- Define the complex number z
def z : ℂ := complex.mk 0 (sqrt 2)

-- State the conjecture
theorem complex_number_problem : z^4 = 4 :=
by {
  sorry
}

end complex_number_problem_l611_611753


namespace minimum_number_of_oranges_l611_611135

noncomputable def minimum_oranges_picked : ℕ :=
  let n : ℕ := 64 in n

theorem minimum_number_of_oranges (n : ℕ) : 
  (∀ (m_i m_j m_k : ℝ) (remaining_masses : Finset ℝ),
    remaining_masses.card = n - 3 →
    (m_i + m_j + m_k) < 0.05 * (∑ m in remaining_masses, m)) →
  n ≥ minimum_oranges_picked :=
by {
  sorry
}

end minimum_number_of_oranges_l611_611135


namespace degrees_conversion_l611_611995

theorem degrees_conversion : (54.12 : ℝ) = 54 + 7/60 + 12/3600 := 
by 
  let degrees := 54
  let minutes := (0.12 * 60).toNat
  let seconds := (0.12 * 60 - minutes) * 60
  have h1 : (54.12 : ℝ) = degrees + (minutes : ℝ) / 60 + seconds / 3600 := by sorry
  rw [←h1]
  sorry

end degrees_conversion_l611_611995


namespace min_value_problem_l611_611464

noncomputable def minValueOfExpression (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x * y * z = 1) : ℝ :=
  (x + 2 * y) * (y + 2 * z) * (x * z + 1)

theorem min_value_problem (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x * y * z = 1) :
  minValueOfExpression x y z hx hy hz hxyz = 16 :=
  sorry

end min_value_problem_l611_611464


namespace problem_statement_l611_611892

-- The conditions of the problem
variables (x : Real)

-- Define the conditions as hypotheses
def condition1 : Prop := (Real.sin (3 * x) * Real.sin (4 * x)) = (Real.cos (3 * x) * Real.cos (4 * x))
def condition2 : Prop := Real.sin (7 * x) = 0

-- The theorem we need to prove
theorem problem_statement (h1 : condition1 x) (h2 : condition2 x) : x = Real.pi / 7 :=
by sorry

end problem_statement_l611_611892


namespace fishers_tomorrow_l611_611810

-- Define the groups of fishers and their fishing pattern
def everyday_fishers := 7
def every_other_day_fishers := 8
def every_three_days_fishers := 3

-- Given counts for yesterday and today
def fishers_yesterday := 12
def fishers_today := 10

-- The problem to prove: 15 people will fish tomorrow
theorem fishers_tomorrow : 
  (fishers_yesterday = 12) →
  (fishers_today = 10) →
  (everyday_fishers = 7) →
  (every_other_day_fishers = 8) →
  (every_three_days_fishers = 3) →
  15 = everyday_fishers + every_other_day_fishers / 2 + every_three_days_fishers / 3 :=
begin
  sorry
end

end fishers_tomorrow_l611_611810


namespace minimum_moves_2_chips_l611_611478

def board := Array (Array (Option Nat)) -- Representing board as 2D Array with optional chips

-- Predicate that defines a valid configuration on the board where each row and each column contains exactly 2 chips.
def valid_configuration (b : board) : Prop :=
  Array.all (fun row => Array.count (fun cell => cell.isSome) row = 2) b ∧
  Array.all (fun col_idx => Array.count (fun row => row[col_idx].isSome) b = 2) (Array.range b.size)

-- Predicate that defines the movement of a chip within the rules.
def valid_move (b1 b2 : board) : Prop :=
  ∃ pos1 pos2 : (Nat × Nat),
  b1[pos1.1][pos1.2].isSome ∧ b2[pos2.1][pos2.2].isNone ∧
  (pos2.1 = pos1.1 ∨ pos2.2 = pos1.2 ∨ (pos2.1.mod 3 = pos1.1.mod 3 ∧ pos2.2.mod 3 = pos1.2.mod 3)) ∧ -- Adjacent logic (vertical, horizontal, diagonal)
  Array.forallidx (fun idx row => row = if idx = pos1.1 then b1[idx]
                                  else if idx = pos2.1 then b2[idx]
                                  else row)

-- Function to represent 2 moves to valid configuration logic.
noncomputable def min_moves_to_valid_config (b1 b2 b3: board) : Prop :=
  valid_move b1 b2 ∧ valid_move b2 b3 ∧ valid_configuration b3

-- Problem statement
theorem minimum_moves_2_chips : ∃ b1 b2 b3 : board, min_moves_to_valid_config b1 b2 b3 := 
sorry -- Proof is omitted

end minimum_moves_2_chips_l611_611478


namespace min_n_pairwise_coprime_l611_611108

def S : Set ℕ := {n | 1 ≤ n ∧ n ≤ 2005}

def pairwise_coprime (s : Set ℕ) : Prop :=
  ∀ (a b : ℕ), a ∈ s → b ∈ s → a ≠ b → Nat.gcd a b = 1

def exists_prime (s : Set ℕ) : Prop :=
  ∃ p ∈ s, Nat.Prime p

theorem min_n_pairwise_coprime (n : ℕ) :
  (∀ s ⊆ S, pairwise_coprime s ∧ s.card = n → exists_prime s) ↔ n = 15 :=
sorry

end min_n_pairwise_coprime_l611_611108


namespace number_of_factors_of_polynomial_l611_611208

theorem number_of_factors_of_polynomial (P : ℤ[X]) (h : P = X^11 - X) : 
  ∃ (factors : List (ℤ[X])), (∀ f ∈ factors, irreducible f ∨ f = X ∨ f = -X) ∧ (factors.length = 6) :=
sorry

end number_of_factors_of_polynomial_l611_611208


namespace range_of_k_l611_611924

theorem range_of_k (x y k : ℝ) (h1 : x + y ≤ 4) (h2 : x - 2 * y ≤ -2) (h3 : k * x - y ≥ -1) (hk : k > 1/2) (h_min_z : ∀ x y, x + y ≤ 4 → x - 2 * y ≤ -2 → k * x - y ≥ -1 → x - y > -3) : 1/2 < k ∧ k < 5 :=
begin
  sorry
end

end range_of_k_l611_611924


namespace solution_to_first_equation_solution_to_second_equation_l611_611149

theorem solution_to_first_equation (x : ℝ) : 
  x^2 - 6 * x + 1 = 0 ↔ x = 3 + 2 * Real.sqrt 2 ∨ x = 3 - 2 * Real.sqrt 2 :=
by sorry

theorem solution_to_second_equation (x : ℝ) : 
  (2 * x - 3)^2 = 5 * (2 * x - 3) ↔ x = 3 / 2 ∨ x = 4 :=
by sorry

end solution_to_first_equation_solution_to_second_equation_l611_611149


namespace three_x_plus_y_eq_zero_l611_611586

theorem three_x_plus_y_eq_zero (x y : ℝ) (h : (2 * x + y) ^ 3 + x ^ 3 + 3 * x + y = 0) : 3 * x + y = 0 :=
sorry

end three_x_plus_y_eq_zero_l611_611586


namespace range_of_a_l611_611354

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₀ d : ℝ), ∀ n, a n = a₀ + n * d

theorem range_of_a (a : ℝ) (a_seq : ℕ → ℝ) (b : ℕ → ℝ)
  (h1 : is_arithmetic_sequence a_seq) 
  (h2 : a_seq 0 = a)
  (h3 : ∀ n, b n = (1 + a_seq n) / a_seq n)
  (h4 : ∀ n : ℕ, 0 < n → b n ≥ b 8) :
  -8 < a ∧ a < -7 :=
sorry

end range_of_a_l611_611354


namespace range_of_ab_l611_611028

theorem range_of_ab (a b : ℝ) :
  (∃ x y : ℝ, a * x - b * y + 1 = 0 ∧ (x + 1)^2 + (y - 2)^2 = 4) →
  ∃ k : set.Icc (-∞ : ℝ) (1 / 8) (λ ab, ab = a * b) :=
by
  sorry

end range_of_ab_l611_611028


namespace intersection_points_l611_611514

-- Define the line equation
def line (x : ℝ) : ℝ := 2 * x - 1

-- Problem statement to be proven
theorem intersection_points :
  (line 0.5 = 0) ∧ (line 0 = -1) :=
by 
  sorry

end intersection_points_l611_611514


namespace smaller_angle_at_8_oclock_l611_611388

theorem smaller_angle_at_8_oclock : let angle_per_hour := 30
  let hour_at_8 := 8
  let total_degrees := 360
  let hour_angle := hour_at_8 * angle_per_hour
  let larger_angle := hour_angle
  let smaller_angle := if larger_angle > 180 then total_degrees - larger_angle else larger_angle
  in smaller_angle = 120 :=
by
  sorry

end smaller_angle_at_8_oclock_l611_611388


namespace math_solution_l611_611996

noncomputable def math_problem (a b : ℝ) : Prop :=
  {x : ℝ | x^2 + a*x + b = 0} ⊆ {1, 2} ∧ ((a = -2 ∧ b = 1) ∨ (a = -4 ∧ b = 4) ∨ (a = -3 ∧ b = 2))

theorem math_solution : ∃ a b : ℝ, math_problem a b := 
begin
  use [-2, 1],
  split,
  { intros x hx,
    simp at hx,
    linarith,
  },
  left,
  exact ⟨rfl, rfl⟩,
  sorry,
  sorry
end

end math_solution_l611_611996


namespace find_real_numbers_l611_611344

theorem find_real_numbers (x y : ℝ) (h : (2 * x - y - 2) + (y - 2) * complex.I = 0) :
  x = 2 ∧ y = 2 :=
sorry

end find_real_numbers_l611_611344


namespace tan_of_sinx_sub_2cosx_eq_sqrt5_l611_611706

theorem tan_of_sinx_sub_2cosx_eq_sqrt5 (x : ℝ) (h : sin x - 2 * cos x = real.sqrt 5) :
  tan x = -1 / 2 :=
sorry

end tan_of_sinx_sub_2cosx_eq_sqrt5_l611_611706


namespace employees_salaries_l611_611843

theorem employees_salaries (M N P : ℝ)
  (hM : M = 1.20 * N)
  (hN_median : N = N) -- Indicates N is the median
  (hP : P = 0.65 * M)
  (h_total : N + M + P = 3200) :
  M = 1288.58 ∧ N = 1073.82 ∧ P = 837.38 :=
by
  sorry

end employees_salaries_l611_611843


namespace transformational_thinking_l611_611979

theorem transformational_thinking:
  ∀ (x y: ℝ), (2 * x + y = 10) ∧ (x = 2 * y) →
  (euqlas(5 * y, 10) ↔ transformatinal_thought) :=
  sorry

end transformational_thinking_l611_611979


namespace fishing_tomorrow_l611_611814

theorem fishing_tomorrow (seven_every_day eight_every_other_day three_every_three_days twelve_yesterday ten_today : ℕ)
  (h1 : seven_every_day = 7)
  (h2 : eight_every_other_day = 8)
  (h3 : three_every_three_days = 3)
  (h4 : twelve_yesterday = 12)
  (h5 : ten_today = 10) :
  (seven_every_day + (eight_every_other_day - (twelve_yesterday - seven_every_day)) + three_every_three_days) = 15 :=
by
  sorry

end fishing_tomorrow_l611_611814


namespace sin_plus_cos_over_sin_minus_cos_l611_611711

-- Define α so that its terminal side falls on the line x + 2y = 0
def α_terminal_on_line_x_plus_2y_eq_0 (α : ℝ) : Prop :=
  ∃ (x y : ℝ), x + 2 * y = 0 ∧
  tan α = y / x

-- Define the problem statement
theorem sin_plus_cos_over_sin_minus_cos (α : ℝ) 
  (hα : α_terminal_on_line_x_plus_2y_eq_0 α) :
  (sin α + cos α) / (sin α - cos α) = -1/3 := by
  sorry

end sin_plus_cos_over_sin_minus_cos_l611_611711


namespace fishing_problem_l611_611780

theorem fishing_problem
  (everyday : ℕ)
  (every_other_day : ℕ)
  (every_three_days : ℕ)
  (yesterday_fishing : ℕ)
  (today_fishing : ℕ)
  (h_everyday : everyday = 7)
  (h_every_other_day : every_other_day = 8)
  (h_every_three_days : every_three_days = 3)
  (h_yesterday_fishing : yesterday_fishing = 12)
  (h_today_fishing : today_fishing = 10) :
  (every_three_days + everyday + (every_other_day - (yesterday_fishing - everyday))) = 15 := by
  sorry

end fishing_problem_l611_611780


namespace problem_statement_l611_611466

structure Point := (x : ℝ) (y : ℝ)

noncomputable def exists_valid_triangle (A B : set Point) : Prop :=
  ∃ t : set Point, 
    (t ⊆ A ∨ t ⊆ B) ∧ 
    t.card = 3 ∧
    (∀ p ∈ interior_convex_hull t, p ∉ (if t ⊆ A then B else A))

theorem problem_statement (A B : set Point) 
  (hA : finite A) 
  (hB : finite B)
  (h_disjoint : A ∩ B = ∅)
  (h_non_collinear : ∀ p q r ∈ (A ∪ B), ¬ collinear {p, q, r})
  (h_size : A.card ≥ 5 ∨ B.card ≥ 5) :
  exists_valid_triangle A B :=
by
  sorry

end problem_statement_l611_611466


namespace part1_part2_l611_611361

noncomputable def f (x : ℝ) : ℝ := |x| + |x + 1|

theorem part1 (x : ℝ) : f x > 3 ↔ x > 1 ∨ x < -2 :=
by
  sorry

theorem part2 (m : ℝ) (hx : ∀ x : ℝ, m^2 + 3 * m + 2 * f x ≥ 0) : m ≤ -2 ∨ m ≥ -1 :=
by
  sorry

end part1_part2_l611_611361


namespace base9_to_base10_l611_611604

def num_base9 : ℕ := 521 -- Represents 521_9
def base : ℕ := 9

theorem base9_to_base10 : 
  (1 * base^0 + 2 * base^1 + 5 * base^2) = 424 := 
by
  -- Sorry allows us to skip the proof.
  sorry

end base9_to_base10_l611_611604


namespace part1_part2_l611_611456

-- Define the vectors a and b
def vector_a : ℝ × ℝ := (2, 0)
def vector_b : ℝ × ℝ := (1, real.sqrt 3)

-- Define the sum of vectors a and b
def vector_sum : ℝ × ℝ := (vector_a.1 + vector_b.1, vector_a.2 + vector_b.2)

-- Define the conditions for unit vectors perpendicular to vector_sum
def is_unit_perpendicular (v : ℝ × ℝ) : Prop :=
  (v.1 * vector_sum.1 + v.2 * vector_sum.2 = 0) ∧ (v.1 ^ 2 + v.2 ^ 2 = 1)

-- Unit vectors perpendicular to a + b:
theorem part1 : 
  is_unit_perpendicular (-1 / 2, real.sqrt 3 / 2) ∧ 
  is_unit_perpendicular (1 / 2, -real.sqrt 3 / 2) :=
sorry

-- Define the vectors involved in part (2)
def t_vector_a_plus_b (t : ℝ) : ℝ × ℝ := (2 * t + 1, real.sqrt 3)
def vector_a_plus_t_vector_b (t : ℝ) : ℝ × ℝ := (2 + t, real.sqrt 3 * t)

-- Define the condition for the angle to be obtuse (dot product < 0)
def is_obtuse (t : ℝ) : Prop :=
  let dot_product := (t_vector_a_plus_b t).1 * (vector_a_plus_t_vector_b t).1 + 
                     (t_vector_a_plus_b t).2 * (vector_a_plus_t_vector_b t).2 in
  dot_product < 0

-- Define the range for t
def t_range (t : ℝ) : Prop :=
  -2 - real.sqrt 3 < t ∧ t < -2 + real.sqrt 3 ∧ t ≠ -1

-- Prove the range for t
theorem part2 : ∀ t : ℝ, is_obtuse(t) ↔ t_range(t) :=
sorry

end part1_part2_l611_611456


namespace find_denominator_l611_611624

theorem find_denominator (d : ℤ) : (9 / (d + 7) = 1 / 3) → d = 20 := by
  intro h
  have h1 : 9 = (d + 7) / 3 := by
    rw [eq_div_iff_mul_eq _ _ (by norm_num : (3 : ℝ) ≠ 0)]
    exact eq.symm h
  linarith

end find_denominator_l611_611624


namespace maria_bottles_proof_l611_611889

theorem maria_bottles_proof 
    (initial_bottles : ℕ)
    (drank_bottles : ℕ)
    (current_bottles : ℕ)
    (bought_bottles : ℕ) 
    (h1 : initial_bottles = 14)
    (h2 : drank_bottles = 8)
    (h3 : current_bottles = 51)
    (h4 : current_bottles = initial_bottles - drank_bottles + bought_bottles) :
  bought_bottles = 45 :=
by
  sorry

end maria_bottles_proof_l611_611889


namespace area_of_highest_points_curve_l611_611245

def projectile_motion_curve_area (v g h : ℝ) : ℝ :=
  π / 8 * (v ^ 4) / (g ^ 2)

theorem area_of_highest_points_curve (v g h : ℝ) (θ : ℝ)
  (hv : v > 0) (hg : g > 0) (hθ : 0 ≤ θ ∧ θ ≤ π / 2) :
  ∃ area, area = projectile_motion_curve_area v g h :=
begin
  use (π / 8) * (v ^ 4) / (g ^ 2),
  refl,
end

end area_of_highest_points_curve_l611_611245


namespace fishing_tomorrow_l611_611771

theorem fishing_tomorrow 
  (P1 : ∀ day : ℕ, 7 ∈ {p | p goes fishing on day})
  (P2 : ∀ day : ℕ, day % 2 = 0 → 8 ∈ {p | p goes fishing on day})
  (P3 : ∀ day : ℕ, day % 3 = 0 → 3 ∈ {p | p goes fishing on day})
  (P4 : ℕ)
  (yesterday : ℕ)
  (today : ℕ)
  (tomorrow : ℕ)
  (hyesterday : yesterday = 12)
  (htoday : today = 10)
  : tomorrow = 15 := by
  sorry

end fishing_tomorrow_l611_611771


namespace math_problem_equivalent_proof_l611_611699

noncomputable def problem_statement
    (A B C : Point)
    (Γ : Circle)
    (ABC_acute : IsAcuteTriangle A B C)
    (AB_gt_AC : A.distance B > A.distance C)
    (M : Point)
    (IsMidpointOfMinorArc : Γ.isMidpointOfArc B C M)
    (D : Point)
    (IsIntersection : LineThrough A C ∩ LineThrough B M = {D})
    (E : Point)
    (AngleBisectorIntersection : AngleBisector A C B ∩ Circumcircle B D C = {E})
    (E_in_ABC : E ∈ Triangle A B C)
    (N : Point)
    (DE_intersects_Γ : LineThrough D E ∩ Γ = {N})
    (E_midpoint_DN : E = Midpoint D N)
    (I_B I_C : Point)
    (I_B_excenter : IsExcenterOppositeToAngle ABC I_B)
    (I_C_excenter : IsExcenterOppositeToAngle ACB I_C)
    : Prop :=
  IsMidpoint N I_B I_C

-- example use of "problem_statement" to declare the theorem (without proving it)
theorem math_problem_equivalent_proof :
  ∀ (A B C : Point) (Γ : Circle),
    IsAcuteTriangle A B C →
    A.distance B > A.distance C →
    ∃ (M D E N I_B I_C : Point),
      Γ.isMidpointOfArc B C M ∧
      (LineThrough A C ∩ LineThrough B M = {D}) ∧
      (AngleBisector A C B ∩ Circumcircle B D C = {E}) ∧
      (E ∈ Triangle A B C) ∧
      (LineThrough D E ∩ Γ = {N}) ∧
      (E = Midpoint D N) ∧
      IsExcenterOppositeToAngle ABC I_B ∧
      IsExcenterOppositeToAngle ACB I_C → IsMidpoint N I_B I_C := by
  intros A B C Γ h₁ h₂
  repeat { use sorry }
  all_goals { sorry }

end math_problem_equivalent_proof_l611_611699


namespace mean_noon_temperature_l611_611176

theorem mean_noon_temperature :
  let temps := [81, 78, 82, 84, 86, 88, 85, 87, 89, 83] in
  (temps.sum / temps.length) = 84 :=
by
  sorry

end mean_noon_temperature_l611_611176


namespace Morgan_first_SAT_score_l611_611090

variable (S : ℝ) -- Morgan's first SAT score
variable (improved_score : ℝ := 1100) -- Improved score on second attempt
variable (improvement_rate : ℝ := 0.10) -- Improvement rate

theorem Morgan_first_SAT_score:
  improved_score = S * (1 + improvement_rate) → S = 1000 := 
by 
  sorry

end Morgan_first_SAT_score_l611_611090


namespace divide_into_equal_parts_l611_611005

def vessel_capacity := (v1 : ℕ) (v2 : ℕ) (v3 : ℕ) := v1 = 3 ∧ v2 = 5 ∧ v3 = 8

def initial_state := (v1 : ℕ) (v2 : ℕ) (v3 : ℕ) := v1 = 0 ∧ v2 = 0 ∧ v3 = 8

theorem divide_into_equal_parts :
  ∃ (v1 v2 v3 : ℕ), vessel_capacity v1 v2 v3 ∧ initial_state 0 0 8 ∧ (v2 = 4 ∧ v3 = 4) :=
by
  sorry

end divide_into_equal_parts_l611_611005


namespace percent_increase_in_area_l611_611097

-- Define the conditions
def diameter1 := 8
def diameter2 := 14

def radius (d : ℕ) : ℕ := d / 2

def area (r : ℕ) : ℕ := (r * r)

-- Prove the desired statement
theorem percent_increase_in_area : 
  let r1 := radius diameter1 in
  let r2 := radius diameter2 in
  let area1 := area r1 in
  let area2 := area r2 in
  (area2 - area1) * 100 / area1 = 206.25 := 
by
  sorry

end percent_increase_in_area_l611_611097


namespace num_points_P_l611_611524

def line : (ℝ × ℝ) → Prop := λ (p : ℝ × ℝ), p.1 / 4 + p.2 / 3 = 1

def ellipse : (ℝ × ℝ) → Prop := λ (p : ℝ × ℝ), p.1^2 / 16 + p.2^2 / 9 = 1

def intersects (A B : ℝ × ℝ) : Prop :=
  line A ∧ ellipse A ∧ line B ∧ ellipse B

noncomputable def area (P A B: ℝ × ℝ) : ℝ :=
  1/2 * abs (P.1*(A.2-B.2) + A.1*(B.2-P.2) + B.1*(P.2-A.2))

theorem num_points_P (A B : ℝ × ℝ) (h1 : intersects A B) :
  (∃ P : (ℝ × ℝ), ellipse P ∧ area P A B = 3) → ∃! P₁ P₂ : ℝ × ℝ, 
  P₁ ≠ P₂ ∧ ellipse P₁ ∧ ellipse P₂ ∧ area P₁ A B = 3 ∧ area P₂ A B = 3 :=
sorry

end num_points_P_l611_611524


namespace geometric_series_sum_l611_611288

theorem geometric_series_sum :
  let a := 2 / 3
  let r := 1 / 3
  a / (1 - r) = 1 :=
by
  sorry

end geometric_series_sum_l611_611288


namespace original_triangle_area_l611_611939

theorem original_triangle_area (a h : ℝ) (A1 A2 : ℝ) (h_eq : h = 2) (A_inc : 12 = (A2 - A1)) (h_inc: A2 = (1 / 2) * a * (h + 6)) (A1_eq : A1 = (1 / 2) * a * h) : A1 = 4 :=
by
    have ha : a = 4,
    {
        sorry
    },
    rw [ha] at A1_eq,
    exact A1_eq

end original_triangle_area_l611_611939


namespace count_even_factors_is_correct_l611_611640

def prime_factors_444_533_72 := (2^8 * 5^3 * 7^2)

def range_a := {a : ℕ | 0 ≤ a ∧ a ≤ 8}
def range_b := {b : ℕ | 0 ≤ b ∧ b ≤ 3}
def range_c := {c : ℕ | 0 ≤ c ∧ c ≤ 2}

def even_factors_count : ℕ :=
  (8 - 1 + 1) * (3 - 0 + 1) * (2 - 0 + 1)

theorem count_even_factors_is_correct :
  even_factors_count = 96 := by
  sorry

end count_even_factors_is_correct_l611_611640


namespace sandy_original_money_l611_611494

theorem sandy_original_money 
  (X : ℝ)
  (spending_clothing : ℝ := 0.25 * X)
  (spending_electronics : ℝ := 0.15 * X)
  (spending_food : ℝ := 0.10 * X)
  (total_spent : ℝ := spending_clothing + spending_electronics + spending_food)
  (sales_tax : ℝ := 0.08 * total_spent)
  (amount_remaining : ℝ := 140) :
  X - (total_spent + sales_tax) = amount_remaining → 
  X = 304.35 :=
by 
  intros h,
  sorry

end sandy_original_money_l611_611494


namespace arithmetic_progression_integers_l611_611442

theorem arithmetic_progression_integers {a b c d : ℤ} (h1 : a + d = b) (h2 : a + 2 * d = c) 
(h3 : a^2 ∈ {a, b, c}) (h4 : b^2 ∈ {a, b, c}) (h5 : c^2 ∈ {a, b, c}):
  ∀ n : ℕ, ∃ m : ℤ, a + n * d = m :=
by 
   sorry

end arithmetic_progression_integers_l611_611442


namespace career_circle_degrees_l611_611033

theorem career_circle_degrees 
  (total_students : ℕ)
  (male_ratio : ℕ)
  (female_ratio : ℕ)
  (career_males : ℕ)
  (career_females : ℕ) :
  total_students = 30 →
  male_ratio = 2 →
  female_ratio = 3 →
  career_males = 2 →
  career_females = 3 →
  (career_males + career_females) / total_students * 360 = 60 :=
begin
  sorry
end

end career_circle_degrees_l611_611033


namespace largest_angle_in_ratio_3_4_5_l611_611170

theorem largest_angle_in_ratio_3_4_5 (x : ℝ) (h1 : 3 * x + 4 * x + 5 * x = 180) : 5 * x = 75 :=
by
  sorry

end largest_angle_in_ratio_3_4_5_l611_611170


namespace favor_both_proposals_l611_611635

variables (T C D : Finset ℕ)
variables (total_students num_C num_D num_Cc_and_Dc num_C_and_D : ℕ)

theorem favor_both_proposals :
  total_students = 230 →
  num_C = 171 →
  num_D = 137 →
  num_Cc_and_Dc = 37 →
  num_C_and_D = num_C + num_D - (total_students - num_Cc_and_Dc) →
  num_C_and_D = 115 :=
by {
  intros hT hC hD hCCc_and_Dc hC_and_D,
  rw [hT, hC, hD, hCCc_and_Dc] at hC_and_D, 
  exact hC_and_D,
  sorry -- you can fill in the actual steps here
}

end favor_both_proposals_l611_611635


namespace mr_c_gain_1000_l611_611091

-- Define the initial conditions
def initial_mr_c_cash := 15000
def initial_mr_c_house := 12000
def initial_mrs_d_cash := 16000

-- Define the changes in the house value
def house_value_appreciated := 13000
def house_value_depreciated := 11000

-- Define the cash changes after transactions
def mr_c_cash_after_first_sale := initial_mr_c_cash + house_value_appreciated
def mrs_d_cash_after_first_sale := initial_mrs_d_cash - house_value_appreciated
def mrs_d_cash_after_second_sale := mrs_d_cash_after_first_sale + house_value_depreciated
def mr_c_cash_after_second_sale := mr_c_cash_after_first_sale - house_value_depreciated

-- Define the final net worth for Mr. C
def final_mr_c_cash := mr_c_cash_after_second_sale
def final_mr_c_house := house_value_depreciated
def final_mr_c_net_worth := final_mr_c_cash + final_mr_c_house
def initial_mr_c_net_worth := initial_mr_c_cash + initial_mr_c_house

-- Statement to prove
theorem mr_c_gain_1000 : final_mr_c_net_worth = initial_mr_c_net_worth + 1000 := by
  sorry

end mr_c_gain_1000_l611_611091


namespace contractor_male_workers_l611_611238

noncomputable def number_of_male_workers (M : ℕ) : Prop :=
  let female_wages : ℕ := 15 * 20
  let child_wages : ℕ := 5 * 8
  let total_wages : ℕ := 35 * M + female_wages + child_wages
  let total_workers : ℕ := M + 15 + 5
  (total_wages / total_workers) = 26

theorem contractor_male_workers : ∃ M : ℕ, number_of_male_workers M ∧ M = 20 :=
by
  use 20
  sorry

end contractor_male_workers_l611_611238


namespace monotonic_intervals_range_of_f_on_0_2_range_of_k_bounded_difference_l611_611359

-- Given function definition
def f (x : ℝ) : ℝ := x^3 - 3 * x

-- 1. Monotonicity intervals
theorem monotonic_intervals :
  (∀ x, x < -1 → monotone_inc (f x)) ∧
  (∀ x, x > 1 → monotone_inc (f x)) ∧
  (∀ x, -1 < x ∧ x < 1 → monotone_dec (f x)) := sorry

-- 2. Range of f(x) on [0, 2]
theorem range_of_f_on_0_2 :
  {y | ∃ x ∈ set.Icc (0 : ℝ) (2 : ℝ), f x = y} = set.Icc (-2 : ℝ) (2 : ℝ) := sorry

-- 3. Range of k for inequality
theorem range_of_k (k : ℝ) :
  (∀ x, 0 ≤ x ∧ x ≤ 2 → f x - k ≥ 0) → k ≤ -2 := sorry

-- 4. Bounded difference between f(x1) and f(x2)
theorem bounded_difference (x₁ x₂ : ℝ) :
  (x₁ ∈ set.Icc (-1 : ℝ) (1 : ℝ)) ∧ (x₂ ∈ set.Icc (-1 : ℝ) (1 : ℝ)) → 
  |f x₁ - f x₂| ≤ 4 := sorry

end monotonic_intervals_range_of_f_on_0_2_range_of_k_bounded_difference_l611_611359


namespace odd_polygon_triangulation_l611_611185

theorem odd_polygon_triangulation (n : ℕ) (h_odd : odd n) 
    (P : Polygon n) (h_convex : convex P) 
    (h_coloring : ∀ (i j : ℕ), i < j → j < i + 2 
      (vertex_color P i ≠ vertex_color P j)) :
  ∃ (triangulation : Triangulation P), 
    ∀ (d : Diagonal triangulation), vertex_color P (d.1) ≠ vertex_color P (d.2) :=
  sorry

end odd_polygon_triangulation_l611_611185


namespace simplify_and_evaluate_expression_l611_611921

theorem simplify_and_evaluate_expression (a : ℝ) (h : a = (Real.sqrt 2) + 1) : 
  (1 - (1 / a)) / ((a ^ 2 - 2 * a + 1) / a) = (Real.sqrt 2) / 2 :=
by
  sorry

end simplify_and_evaluate_expression_l611_611921


namespace find_h_plus_k_l611_611662

theorem find_h_plus_k (h k : ℝ) :
  (∀ (x y : ℝ),
    (x - 3) ^ 2 + (y + 4) ^ 2 = 49) → 
  h = 3 ∧ k = -4 → 
  h + k = -1 :=
by
  sorry

end find_h_plus_k_l611_611662


namespace hour_hand_rotation_minute_hand_rotation_l611_611940

def degrees_per_hour := 30

theorem hour_hand_rotation :
  let rotation := degrees_per_hour * 2 in rotation = 60 :=
by
  sorry

def degrees_per_minute := 6

theorem minute_hand_rotation :
  let start_time := 9 * 60 + 15 in
  let rotation := 90 in
  let major_divisions := rotation / degrees_per_minute in
  let end_time := start_time + major_divisions * 5 in
  end_time = 9 * 60 + 30 :=
by
  sorry

end hour_hand_rotation_minute_hand_rotation_l611_611940


namespace cardboard_plastic_coincide_l611_611095

-- Definition of the conditions
structure Square (α : Type*) :=
(vertices : set α)

noncomputable def coincides (c p : Square ℝ) : Prop :=
c.vertices = p.vertices

theorem cardboard_plastic_coincide (n : ℕ) (cardboard_squares plastic_squares : fin n → Square ℝ) :
  (∀ i j, i ≠ j → disjoint cardboard_squares i.vertices (cardboard_squares j).vertices) →
  (∀ i j, i ≠ j → disjoint plastic_squares i.vertices (plastic_squares j).vertices) →
  (set.univ ⋃ i, (cardboard_squares i).vertices = set.univ ⋃ i, (plastic_squares i).vertices) →
  ∀ i, ∃ j, coincides (cardboard_squares i) (plastic_squares j) :=
sorry

end cardboard_plastic_coincide_l611_611095


namespace least_number_divisible_31_and_9_remainder_3_l611_611582

theorem least_number_divisible_31_and_9_remainder_3 : 
  ∃ n : ℕ, (n % 31 = 3) ∧ (n % 9 = 3) ∧ ∀ m : ℕ, ((m % 31 = 3) ∧ (m % 9 = 3)) → (m ≥ n) := 
begin
  use 282,
  split,
  { exact nat.mod_eq_of_lt (by norm_num)},
  split,
  { exact nat.mod_eq_of_lt (by norm_num)},
  { intros m h,
    cases h with h31 h9,
    have : m ≡ 3 [MOD (31 * 9)] := int.modeq.and (nat.modeq.symm h31) (nat.modeq.symm h9) using 3,
    exact int.modeq.least modeq modes at.
  }
end

end least_number_divisible_31_and_9_remainder_3_l611_611582


namespace problem_statement_l611_611400

theorem problem_statement (x : ℂ) (h : x + 1 / x = real.sqrt 5) : x ^ 10 = 1 := by
  sorry

end problem_statement_l611_611400


namespace polar_to_rectangular_and_PA_PB_sum_l611_611854

theorem polar_to_rectangular_and_PA_PB_sum :
  ∀ (C : ℝ → ℝ) (l : ℝ → ℝ) (P : ℝ × ℝ),
    (∀ θ, C (θ) = (Real.sin θ / Real.cos θ ^ 2) * (Real.cos θ ^ 2)) →
    (P = (0, 2)) →
    (∀ t, l (t) = (t * Real.sqrt 2 / 2, 2 + t * Real.sqrt 2 / 2)) →
    let A := (t1, l t1) in
    let B := (t2, l t2) in
    (∀ t1 t2, t1 ^ 2 - t1 * Real.sqrt 2 - 4 = 0 ∧ t2 ^ 2 - t2 * Real.sqrt 2 - 4 = 0) →
    (|PA| + |PB| / |PA| * |PB| = 3 * Real.sqrt 2 / 4) := by
  sorry

end polar_to_rectangular_and_PA_PB_sum_l611_611854


namespace ratio_of_parallel_lines_l611_611265

theorem ratio_of_parallel_lines (DE BC : ℝ) (h_parallel : DE ∥ BC) 
  (area_ADE : Real := 2) (area_trapezoid_DBCE : Real := 6) :
  DE / BC = 1 / 2 := by
  sorry

end ratio_of_parallel_lines_l611_611265


namespace investment_duration_is_2_years_l611_611299

noncomputable def compound_interest_years 
  (P : ℝ) (r : ℝ) (n : ℕ) (CI : ℝ) : ℝ :=
let A := P + CI in
let t := (Real.log (A / P)) / (n * Real.log (1 + r / n)) in
t

theorem investment_duration_is_2_years :
  compound_interest_years 50000 0.04 2 4121.608 ≈ 2 := by
  -- we will prove this by calculating the value explicitly or showing that 
  -- compound_interest_years approximates to 2 with given conditions
  sorry

end investment_duration_is_2_years_l611_611299


namespace travel_impossible_l611_611503

-- Definitions based on problem conditions
structure Island (V : Type) (E : Type) [DecidableEq V] :=
  (vertices : Finset V)
  (edges : Finset E)
  (incidence : E → V × V)
  (connected : ∀ x y : V, x ∈ vertices → y ∈ vertices → 
    ∃ p : List V, p.head = some x ∧ p.last = some y ∧ ∀ v ∈ p, v ∈ vertices ∧ ∀ v w : V, (v, w) ∈ (p.zip p.tail).toFinset → (∃ e ∈ edges, incidence e = (v, w) ∨ incidence e = (w, v)))
  (remaining_connected : ∀ x y z u : V, x ∈ vertices → y ∈ vertices → z ∈ vertices → u ∈ vertices → x ≠ y → z ≠ u → 
    (∀ v ∈ vertices \ {x, y}, v ∈ vertices → 
    ∃ p : List V, p.head = some z ∧ p.last = some u ∧ ∀ v' ∈ p, v' ∈ vertices \ {x, y} ∧ ∀ v' w: V, (v', w') ∈ (p.zip p.tail).toFinset → (∃ e ∈ edges, incidence e = (v', w') ∨ incidence e = (w', v')).toFinset))

-- Statement of the problem
theorem travel_impossible {V E : Type} [DecidableEq V] (island : Island V E) (x y z : E) 
  (distinct : island.incidence x ≠ island.incidence y ∧ island.incidence y ≠ island.incidence z ∧ island.incidence z ≠ island.incidence x) :
  ¬ ∃ (start : V) (cycle : List E), cycle.head = some x ∧ cycle.last = some start ∧ ∀ e ∈ cycle, e ∈ {x, y, z} ∧ ∀ v w : V, (v, w) ∈ (cycle.zip cycle.tail).toFinset → (island.incidence e = (v, w) ∨ island.incidence e = (w, v)) :=
sorry

end travel_impossible_l611_611503


namespace fishers_tomorrow_l611_611805

-- Define the groups of fishers and their fishing pattern
def everyday_fishers := 7
def every_other_day_fishers := 8
def every_three_days_fishers := 3

-- Given counts for yesterday and today
def fishers_yesterday := 12
def fishers_today := 10

-- The problem to prove: 15 people will fish tomorrow
theorem fishers_tomorrow : 
  (fishers_yesterday = 12) →
  (fishers_today = 10) →
  (everyday_fishers = 7) →
  (every_other_day_fishers = 8) →
  (every_three_days_fishers = 3) →
  15 = everyday_fishers + every_other_day_fishers / 2 + every_three_days_fishers / 3 :=
begin
  sorry
end

end fishers_tomorrow_l611_611805


namespace three_pow_neg_x_l611_611749

theorem three_pow_neg_x (x : ℝ) (h : 64^7 = 32^x) : 3^(-x) = (1 / 3)^((42 : ℝ) / 5) := by
  sorry

end three_pow_neg_x_l611_611749


namespace max_traffic_flow_at_40_range_for_traffic_flow_exceeds_10_l611_611630

noncomputable def traffic_flow (v : ℝ) : ℝ :=
  920 * v / (v^2 + 3 * v + 1600)

theorem max_traffic_flow_at_40 :
  is_max_value (traffic_flow 40) (λ v, v > 0 → traffic_flow v) :=
by
  sorry

theorem range_for_traffic_flow_exceeds_10 :
  ∀ v : ℝ, 25 < v ∧ v < 64 → traffic_flow v > 10 :=
by
  sorry

end max_traffic_flow_at_40_range_for_traffic_flow_exceeds_10_l611_611630


namespace c_share_l611_611227

theorem c_share (A B C D : ℝ) 
    (h1 : A = 1/2 * B) 
    (h2 : B = 1/2 * C) 
    (h3 : D = 1/4 * 392) 
    (h4 : A + B + C + D = 392) : 
    C = 168 := 
by 
    sorry

end c_share_l611_611227


namespace range_sin_cos_two_x_is_minus2_to_9_over_8_l611_611180

noncomputable def range_of_function : Set ℝ :=
  { y : ℝ | ∃ x : ℝ, y = Real.sin x + Real.cos (2 * x) }

theorem range_sin_cos_two_x_is_minus2_to_9_over_8 :
  range_of_function = Set.Icc (-2) (9 / 8) := 
by
  sorry

end range_sin_cos_two_x_is_minus2_to_9_over_8_l611_611180


namespace evaporation_period_l611_611606

theorem evaporation_period (initial_amount: ℝ) (daily_evaporation: ℝ) (percentage: ℝ) (total_days: ℝ) 
  (h_initial: initial_amount = 10) 
  (h_daily: daily_evaporation = 0.01) 
  (h_percentage: percentage = 2 / 100) 
  (h_evaporated: percentage * initial_amount = 0.2) : 
  total_days = 0.2 / 0.01 :=
begin
  sorry
end

end evaporation_period_l611_611606


namespace compare_a_b_c_l611_611686

noncomputable def a := (2/3 : ℝ)^(-1/3)
noncomputable def b := (5/3 : ℝ)^(-2/3)
noncomputable def c := (3/2 : ℝ)^(2/3)

theorem compare_a_b_c : b < a ∧ a < c :=
by
  sorry

end compare_a_b_c_l611_611686


namespace enclosed_area_locus_of_P_l611_611060

noncomputable def dist_to_line (P : ℝ × ℝ) (a : ℝ) : ℝ :=
  let h := P.1
  let k := P.2
  abs(h + k) / real.sqrt 2

noncomputable def dist_to_point (P : ℝ × ℝ) : ℝ :=
  let h := P.1
  let k := P.2
  real.sqrt (h^2 + (k - 1)^2)

theorem enclosed_area_locus_of_P
  (a : ℝ)
  (C : ℝ × ℝ → Prop)
  (touch_line : ∀ P, C P → dist_to_line P a = dist_to_point P)
  (pass_through_point : ∀ P, C P → dist_to_point P = dist_to_point (0, 1))
  (P : ℝ × ℝ)
  (center_C : C P) :
  ∫ y in 0..1, 2 * real.sqrt (4 * y - 2) = 2 * real.sqrt 2 / 3 :=
by
  sorry

end enclosed_area_locus_of_P_l611_611060


namespace largest_angle_in_ratio_3_4_5_l611_611169

theorem largest_angle_in_ratio_3_4_5 (x : ℝ) (h1 : 3 * x + 4 * x + 5 * x = 180) : 5 * x = 75 :=
by
  sorry

end largest_angle_in_ratio_3_4_5_l611_611169


namespace most_likely_passes_l611_611285

theorem most_likely_passes (n : ℕ) (p : ℝ) (k₀ : ℕ) (h_n : n = 15) (h_p : p = 0.9) :
  k₀ = 14 :=
by
  rw [h_n, h_p]
  sorry

end most_likely_passes_l611_611285


namespace isosceles_triangle_base_length_l611_611338

theorem isosceles_triangle_base_length
  (a b : ℝ) (h₁ : a = 4) (h₂ : b = 8) (h₃ : a ≠ b)
  (triangle_inequality : ∀ x y z : ℝ, x + y > z) :
  ∃ base : ℝ, base = 8 := by
  sorry

end isosceles_triangle_base_length_l611_611338


namespace total_players_on_ground_l611_611987

theorem total_players_on_ground 
  (cricket_players : ℕ) (hockey_players : ℕ) (football_players : ℕ) (softball_players : ℕ)
  (hcricket : cricket_players = 16) (hhokey : hockey_players = 12) 
  (hfootball : football_players = 18) (hsoftball : softball_players = 13) :
  cricket_players + hockey_players + football_players + softball_players = 59 :=
by
  sorry

end total_players_on_ground_l611_611987


namespace sum_of_base3_representation_l611_611294

theorem sum_of_base3_representation :
  ∃ (n : Fin 8 → ℕ), (∀ k : ℤ, −1985 ≤ k → k ≤ 1985 → 
  ∃ (a : Fin 8 → ℤ), (∀ i, a i ∈ {-1, 0, 1}) ∧ k = ∑ i, a i * n i) :=
sorry

end sum_of_base3_representation_l611_611294


namespace angle_VUT_20_l611_611435

-- Define the geometrical setup with points and lines
variables (m n : Line) 
variables (T U V : Point)

-- Define the conditions given in the problem
axiom parallel_lines : m ∥ n
axiom T_on_m : LiesOn T m
axiom U_on_n : LiesOn U n
axiom V_on_UV : LiesOn V (Line.mk U V)
axiom perpendicular_UV_n : Perpendicular (Line.mk U V) n
axiom angle_UTV : ∠ U T V = 110

-- Define what needs to be proven
theorem angle_VUT_20 :
  ∠ V U T = 20 :=
sorry

end angle_VUT_20_l611_611435


namespace supremum_expression_l611_611309

theorem supremum_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
  Sup {M : ℝ | ∃ (a b : ℝ), 0 < a ∧ 0 < b ∧ a + b = 1 ∧ M = - (1 / (2 * a) + 2 / b)} = - 9 / 2 :=
sorry

end supremum_expression_l611_611309


namespace parabola_b_value_l611_611178

variable {q : ℝ}

theorem parabola_b_value (a b c : ℝ) (h_a : a = -3 / q)
  (h_eq : ∀ x : ℝ, (a * x^2 + b * x + c) = a * (x - q)^2 + q)
  (h_intercept : (a * 0^2 + b * 0 + c) = -2 * q)
  (h_q_nonzero : q ≠ 0) :
  b = 6 / q := 
sorry

end parabola_b_value_l611_611178


namespace exist_student_all_olympiads_l611_611432

variable {Student : Type}

theorem exist_student_all_olympiads 
  (A : Fin 50 → Finset Student)
  (h1 : ∀ i, A i.card = 30)
  (h2 : ∀ i j, i ≠ j → A i ≠ A j)
  (h3 : ∀ S : Finset (Fin 50), S.card = 30 → ∃ x, ∀ i ∈ S, x ∈ A i) : 
  ∃ x, ∀ i, x ∈ A i :=
sorry

end exist_student_all_olympiads_l611_611432


namespace distinct_elements_no_perfect_square_l611_611468

def is_perfect_square (x : ℤ) : Prop :=
  ∃ n : ℤ, n * n = x

theorem distinct_elements_no_perfect_square (d : ℤ) (h : d > 0 ∧ d ≠ 2 ∧ d ≠ 5 ∧ d ≠ 13) :
  ∃ a b ∈ ({2, 5, 13, d} : set ℤ), a ≠ b ∧ ¬ is_perfect_square (a * b - 1) :=
sorry

end distinct_elements_no_perfect_square_l611_611468


namespace real_expression_implies_a_eq_2_l611_611346

theorem real_expression_implies_a_eq_2 (a : ℝ) (h: ∃ r : ℝ, (a:ℂ) / (1 + complex.I) + (1 + complex.I) = r) : a = 2 :=
by
  sorry

end real_expression_implies_a_eq_2_l611_611346


namespace integer_solutions_l611_611522

theorem integer_solutions (n : ℕ) : (finite { x : ℤ | x*x < n }).toFinset.card = n ↔ n = 0 ∨ n = 1 ∨ n = 3 ∨ n = 5 := by
  sorry

end integer_solutions_l611_611522


namespace smallest_number_divisible_l611_611219

theorem smallest_number_divisible (n : ℕ) :
  (n + 2) % 12 = 0 ∧ 
  (n + 2) % 30 = 0 ∧ 
  (n + 2) % 48 = 0 ∧ 
  (n + 2) % 74 = 0 ∧ 
  (n + 2) % 100 = 0 ↔ 
  n = 44398 :=
by sorry

end smallest_number_divisible_l611_611219


namespace circle_chord_integer_lengths_l611_611098

theorem circle_chord_integer_lengths (Q : Point) (O : Point) (r : ℝ) (d_OQ : dist O Q = 5) (radius : r = 13) :
  ∃! (n : ℕ), n = 3 := sorry

end circle_chord_integer_lengths_l611_611098


namespace solve_for_a_l611_611887

variable (a : ℝ)

/-- Problem Statement: Given the sets U and A defined as below, and the complement of A in U being {5},
find the values of a. -/
theorem solve_for_a (h1 : {2, 3, a^2 + 2 * a - 3} = {2, 3, a^2 + 2 * a - 3}) 
                    (h2 : {2, |a + 1|} = {2, |a + 1|})
                    (h3 : ({5} : Set ℝ) = ({x | x ∈ {2, 3, a^2 + 2 * a - 3} ∧ x ∉ {2, |a + 1|}} : Set ℝ)) :
 a = 2 ∨ a = -4 := 
by
  sorry

end solve_for_a_l611_611887


namespace f_2011_eq_neg2_l611_611069

def f : ℝ → ℝ := sorry -- The function definition will be provided in the proof

axiom f_symmetric : ∀ x : ℝ, f (1 - x) = - f (x - 1)
axiom f_periodic : ∀ x : ℝ, f ((3/4) - x) = f ((3/4) + x)
axiom f_piecewise : ∀ x : ℝ, x < - (3/4) ∧ x ≥ - (3/2) → f x = Real.log2 (-3 * x + 1)

theorem f_2011_eq_neg2 : f 2011 = -2 := by
  sorry -- The proof will be provided here

end f_2011_eq_neg2_l611_611069


namespace largest_angle_in_ratio_triangle_l611_611167

theorem largest_angle_in_ratio_triangle (x : ℝ) (h1 : 3 * x + 4 * x + 5 * x = 180) : 
  5 * (180 / (3 + 4 + 5)) = 75 := by
  sorry

end largest_angle_in_ratio_triangle_l611_611167


namespace area_of_smaller_base_frustam_l611_611929

theorem area_of_smaller_base_frustam
  (circumference_ratio : ℝ)
  (slant_height : ℝ)
  (lateral_area : ℝ) :
  circumference_ratio = 3 →
  slant_height = 3 →
  lateral_area = 84 * Real.pi →
  ∃ (r : ℝ), 
  let smaller_base_area := Real.pi * r^2 in
  smaller_base_area = 49 * Real.pi :=
by {
  intro h1 h2 h3,
  use 7,
  rw [h1, h2, h3],
  sorry
}

end area_of_smaller_base_frustam_l611_611929


namespace identify_A_and_B_l611_611255

-- Definitions for A and B's statements
def A_statement : Prop := (∀ x, x = "A" ∨ x = "B" → "x is a monkey")
def B_statement : Prop := (∀ x, x = "A" ∨ x = "B" → "x is a liar")

-- Definitions for being a monkey and knight/liar
def is_monkey (x : String) : Prop := sorry
def is_knight (x : String) : Prop := sorry
def is_liar (x : String) : Prop := sorry

-- Define A and B
noncomputable def A : String := "A"
noncomputable def B : String := "B"

-- Conditions based on their statements
axiom A_condition : A_statement
axiom B_condition : B_statement

-- Proof goal
theorem identify_A_and_B : 
  (is_monkey A ∧ is_knight A) ∧ (is_monkey B ∧ is_liar B) := sorry

end identify_A_and_B_l611_611255


namespace smallest_lambda_two_l611_611677

variable {x : ℕ → ℝ} {y : ℕ → ℝ} {λ : ℝ} {m : ℕ}

-- Define the sequences
noncomputable def y_seq (x : ℕ → ℝ) : ℕ → ℝ
| 1      => x 1
| (n + 1) => x (n + 1) - (Finset.range n).sum (λ i, x (i + 1)^2) ^ (1 / 2)

-- The inequality to be proven
theorem smallest_lambda_two (hλ : 0 < λ) : 
  (∀ x : ℕ → ℝ, ∀ m : ℕ, 0 < m → 
    1 / (m : ℝ) * (Finset.range m).sum (λ i, x (i + 1)^2) ≤ 
    (Finset.range m).sum (λ i, λ ^ (m - i) * (y_seq x (i + 1))^2)) ↔ λ = 2 := 
sorry

end smallest_lambda_two_l611_611677


namespace at_most_1_plus_sqrt_2p_integers_l611_611453

theorem at_most_1_plus_sqrt_2p_integers (p : ℕ) [hp : Fact (Nat.Prime p)] 
  (hq : Nat.Prime ((p - 1) / 2))
  (a b c : ℤ) (ha : ¬ p ∣ a) (hb : ¬ p ∣ b) (hc : ¬ p ∣ c) :
  ∃ S : Finset ℕ, S.card ≤ 1 + Nat.sqrt (2 * p) ∧ ∀ n ∈ S, n < p ∧ p ∣ (a ^ n + b ^ n + c ^ n) :=
by
  sorry

end at_most_1_plus_sqrt_2p_integers_l611_611453


namespace conditional_probability_law_bayes_theorem_l611_611867

variables {Ω : Type*} [ProbabilitySpace Ω]
variables (M N : Event Ω)

theorem conditional_probability_law :
  P(M ∩ N) = P(M) * P(N | M) :=
sorry

theorem bayes_theorem :
  P(M | N) = (P(N | M) * P(M)) / P(N) :=
sorry

end conditional_probability_law_bayes_theorem_l611_611867


namespace horner_evaluation_l611_611561

-- Definition of the polynomial f(x)
def f (x : ℝ) : ℝ := 2 * x^6 + 3 * x^5 + 5 * x^3 + 6 * x^2 + 7 * x + 1

-- Evaluating the polynomial using Horner's method
noncomputable def evaluate_horner (x : ℝ) : ℝ :=
    ((2 * x + 3) * x^2 + 5) * x^3 + (6 * x^2 + 7 * x + 1)

-- The proof problem statement
theorem horner_evaluation :
  ∀ (x : ℝ), x = 0.5 → 
    -- Given this evaluation requires 6 multiplications and 6 additions
    (∃ (mult_operations : ℕ) (add_operations : ℕ),
       mult_operations = 6 ∧ add_operations = 6) :=
by
  -- Definitions of the number of operations
  let mult_operations := 6
  let add_operations := 6
  intro x hx
  have : x = 0.5 := hx
  use mult_operations, add_operations
  split
  · exact rfl
  · exact rfl
  sorry

end horner_evaluation_l611_611561


namespace find_m_l611_611454

def U := {0, 1, 2, 3}
def A (m : ℝ) := {x ∈ U | x * (x + m) = 0}
def CU_A (m : ℝ) := U \ A m

theorem find_m (m : ℝ) : CU_A m = {1, 2} → m = -3 := by
  intro h
  -- Proof goes here
  sorry

end find_m_l611_611454


namespace space_between_trees_l611_611035

theorem space_between_trees (n_trees : ℕ) (tree_space : ℕ) (total_length : ℕ) (spaces_between_trees : ℕ) (result_space : ℕ) 
  (h1 : n_trees = 8)
  (h2 : tree_space = 1)
  (h3 : total_length = 148)
  (h4 : spaces_between_trees = n_trees - 1)
  (h5 : result_space = (total_length - n_trees * tree_space) / spaces_between_trees) : 
  result_space = 20 := 
by sorry

end space_between_trees_l611_611035


namespace range_of_a_quadratic_root_conditions_l611_611373

theorem range_of_a_quadratic_root_conditions (a : ℝ) :
  ((∃ x₁ x₂ : ℝ, x₁ > 2 ∧ x₂ < 2 ∧ (ax^2 - 2*(a+1)*x + a-1 = 0)) ↔ (0 < a ∧ a < 5)) :=
by
  sorry

end range_of_a_quadratic_root_conditions_l611_611373


namespace gcd_360_504_l611_611161

theorem gcd_360_504 : Int.gcd 360 504 = 72 := by
  sorry

end gcd_360_504_l611_611161


namespace find_angle_l611_611655

theorem find_angle :
  ∃ (m : ℤ), (-180 ≤ m ∧ m ≤ 180) ∧ Real.sin (m * (Float.pi / 180)) = Real.sin (945 * (Float.pi / 180)) ∧ m = -135 := 
sorry

end find_angle_l611_611655


namespace trains_pass_time_correct_l611_611557

noncomputable def train_time_to_pass (len1 len2 : ℕ) (speed1_kmh speed2_kmh : ℚ) (angle_deg : ℚ) : ℚ :=
  let speed1 := speed1_kmh * (1000 / 3600) in
  let speed2 := speed2_kmh * (1000 / 3600) in
  let rad := π / 180 in
  let angle_rad := angle_deg * rad in
  let component1 := speed1 * real.cos angle_rad in
  let component2 := speed2 * real.cos angle_rad in
  let relative_speed := component1 + component2 in
  let total_length := len1 + len2 in
  total_length / relative_speed

theorem trains_pass_time_correct :
  train_time_to_pass 140 160 57.3 38.7 30 ≈ 12.99 := sorry

end trains_pass_time_correct_l611_611557


namespace voting_system_of_25_stabilizes_l611_611591

-- Define the context and conditions
def voting_system_stabilizes (n : ℕ) (votes : ℕ → fin n → bool) : Prop :=
  ∀ round : ℕ, ∃ T : ℕ, ∀ t ≥ T, ∀ i : fin n,
  votes (t+1) i = votes t i

-- Specifying the main theorem
theorem voting_system_of_25_stabilizes :
  voting_system_stabilizes 25 := 
sorry

end voting_system_of_25_stabilizes_l611_611591


namespace opp_sign_sum_eq_three_l611_611404

theorem opp_sign_sum_eq_three (x y : ℝ) :
  abs (x^2 - 4 * x + 4) + sqrt (2 * x - y - 3) = 0 →
  x + y = 3 := 
by
  sorry

end opp_sign_sum_eq_three_l611_611404


namespace spadesuit_evaluation_l611_611314

-- Define the operation
def spadesuit (a b : ℝ) : ℝ := (a + b) * (a - b)

-- The theorem to prove
theorem spadesuit_evaluation : spadesuit 4 (spadesuit 5 (-2)) = -425 :=
by
  sorry

end spadesuit_evaluation_l611_611314


namespace binomial_expansion_properties_l611_611413

theorem binomial_expansion_properties:
  ∀ n : ℕ, (2^n = 64)
  → (n = 6 ∧ 
      (∀ x, x ≠ 0 → 
          let term := (x - (2 / x^2))^6 in 
          (term.coeff 3 = -12) ∧ 
          (term.biggest_coeff = -160 / x^3) ∧ 
          (term.coeff_sum = 1)
      )
  ) :=
by {
  sorry
}

end binomial_expansion_properties_l611_611413


namespace largest_x_value_l611_611501

-- Definition of the equation
def equation (x : ℚ) : Prop := 3 * (9 * x^2 + 10 * x + 11) = x * (9 * x - 45)

-- The problem to prove is that the largest value of x satisfying the equation is -1/2
theorem largest_x_value : ∃ x : ℚ, equation x ∧ ∀ y : ℚ, equation y → y ≤ -1/2 := by
  sorry

end largest_x_value_l611_611501


namespace parallel_lines_slope_l611_611030

theorem parallel_lines_slope (a : ℝ) : 
  let m1 := - (a / 2)
  let m2 := 3
  ax + 2 * y + 2 = 0 ∧ 3 * x - y - 2 = 0 → m1 = m2 → a = -6 := 
by
  intros
  sorry

end parallel_lines_slope_l611_611030


namespace fishing_tomorrow_l611_611778

theorem fishing_tomorrow 
  (P1 : ∀ day : ℕ, 7 ∈ {p | p goes fishing on day})
  (P2 : ∀ day : ℕ, day % 2 = 0 → 8 ∈ {p | p goes fishing on day})
  (P3 : ∀ day : ℕ, day % 3 = 0 → 3 ∈ {p | p goes fishing on day})
  (P4 : ℕ)
  (yesterday : ℕ)
  (today : ℕ)
  (tomorrow : ℕ)
  (hyesterday : yesterday = 12)
  (htoday : today = 10)
  : tomorrow = 15 := by
  sorry

end fishing_tomorrow_l611_611778


namespace ratio_rocks_eaten_to_collected_l611_611282

def rocks_collected : ℕ := 10
def rocks_left : ℕ := 7
def rocks_spit_out : ℕ := 2

theorem ratio_rocks_eaten_to_collected : 
  (rocks_collected - rocks_left + rocks_spit_out) * 2 = rocks_collected := 
by 
  sorry

end ratio_rocks_eaten_to_collected_l611_611282


namespace fishers_tomorrow_l611_611789

-- Definitions based on conditions
def people_every_day : ℕ := 7
def people_every_other_day : ℕ := 8
def people_every_three_days : ℕ := 3
def people_yesterday : ℕ := 12
def people_today : ℕ := 10

-- Theorem to be proved
theorem fishers_tomorrow (people_every_day people_every_other_day people_every_three_days people_yesterday people_today : ℕ) : 
  people_every_day = 7 ∧ 
  people_every_other_day = 8 ∧ 
  people_every_three_days = 3 ∧
  people_yesterday = 12 ∧ 
  people_today = 10 →
  15 := 
by {
  sorry
}

end fishers_tomorrow_l611_611789


namespace smallest_nonneg_sum_of_squares_l611_611768

theorem smallest_nonneg_sum_of_squares :
  ∃ f : ℕ → ℤ, (∀ n ∈ { n : ℕ | 1 ≤ n ∧ n ≤ 2005 }, f n = n^2 ∨ f n = -(n^2)) ∧
  ∃ s : ℤ, (s = ∑ n in (Finset.Icc 1 2005), f n) ∧ s ≥ 0 ∧ s = 1 :=
sorry

end smallest_nonneg_sum_of_squares_l611_611768


namespace unoccupied_volume_l611_611553

noncomputable def cylinder_volume (r h : ℝ) : ℝ := π * r^2 * h
noncomputable def cone_volume (r h : ℝ) : ℝ := (1 / 3) * π * r^2 * h
noncomputable def sphere_volume (r : ℝ) : ℝ := (4 / 3) * π * r^3

theorem unoccupied_volume :
  let V_cylinder := cylinder_volume 10 35
  let V_cones := 2 * (cone_volume 10 15)
  let V_sphere := sphere_volume 5
  V_cylinder - V_cones - V_sphere = (7000 * π / 3) :=
by
  -- This is where the proof would go, but it's not needed for this task.
  sorry

end unoccupied_volume_l611_611553


namespace find_a_l611_611324

noncomputable def f (x : ℝ) : ℝ := 3 * ((x - 1) / 2) - 2

theorem find_a (x a : ℝ) (hx : f a = 4) (ha : a = 2 * x + 1) : a = 5 :=
by
  sorry

end find_a_l611_611324


namespace correct_calculation_l611_611212

-- Define the conditions as boolean values
def condition_A : Prop := (sqrt 2) ^ 0 = sqrt 2
def condition_B : Prop := 2 * sqrt 3 + 3 * sqrt 3 = 5 * sqrt 6
def condition_C : Prop := sqrt 8 = 4 * sqrt 2
def condition_D : Prop := sqrt 3 * (2 * sqrt 3 - 2) = 6 - 2 * sqrt 3

-- State the theorem to be proved
theorem correct_calculation : condition_D :=
by
  -- the proof is skipped with sorry
  sorry

end correct_calculation_l611_611212


namespace minimum_value_S_l611_611339

variable (x y z : ℝ)
variable (hx : x > 0)
variable (hy : y > 0)
variable (hz : z > 0)
variable (h : x^2 + y^2 + z^2 = 1)

theorem minimum_value_S : 
  (∃ S, S = (xy : ℝ) / z + (yz : ℝ) / x + (zx : ℝ) / y) ∧ S = sqrt(3) :=
by
  sorry

end minimum_value_S_l611_611339


namespace perpendicular_line_plane_l611_611471

variable {α β : Type} [binary_relation α β]

theorem perpendicular_line_plane (m n : α) (α : β) : 
  (m ⊥ α) → (n ∦ α) → (m ⊥ n) :=
by 
  intros 
  sorry

end perpendicular_line_plane_l611_611471


namespace range_of_m_l611_611722

theorem range_of_m (f g : ℝ → ℝ) (m : ℝ) (h_f : ∀ x, f x = (3 - m * 3^x) / 3^x)
  (h_g : ∀ x, g x = log 2 (x^2 + x + 2))
  (h_cond : ∀ x1 ∈ set.Icc (-1 : ℝ) 2, ∃ x2 ∈ set.Icc (0 : ℝ) 3, f x1 ≥ g x2) :
  m ≤ -2/3 := 
sorry

end range_of_m_l611_611722


namespace pyramid_volume_is_one_l611_611349

-- Define a regular square pyramid with base edge length and height
structure SquarePyramid where
  base_edge_length : ℝ
  height : ℝ

-- Define the formula for the volume of a square pyramid
def volume (pyramid : SquarePyramid) : ℝ :=
  (1 / 3) * (pyramid.base_edge_length * pyramid.base_edge_length) * pyramid.height

-- Define the given conditions
def pyramid_conditions : SquarePyramid :=
{ base_edge_length := 1,
  height := 3 }

-- State the proof goal
theorem pyramid_volume_is_one : volume pyramid_conditions = 1 := by
  sorry

end pyramid_volume_is_one_l611_611349


namespace minimal_cost_is_five_l611_611897

def is_valid_marking_strategy (S : set ℕ) : Prop :=
  (2 ∈ S ∧ 3 ∈ S ∧ 4 ∈ S ∧ 5 ∈ S ∧ 6 ∈ S ∧ 7 ∈ S ∧ 8 ∈ S ∧ 9 ∈ S ∧ 10 ∈ S ∧ 11 ∈ S ∧
   12 ∈ S ∧ 13 ∈ S ∧ 14 ∈ S ∧ 15 ∈ S ∧ 16 ∈ S ∧ 17 ∈ S ∧ 18 ∈ S ∧ 19 ∈ S ∧ 20 ∈ S ∧
   21 ∈ S ∧ 22 ∈ S ∧ 23 ∈ S ∧ 24 ∈ S ∧ 25 ∈ S ∧ 26 ∈ S ∧ 27 ∈ S ∧ 28 ∈ S ∧ 29 ∈ S ∧
   30 ∈ S) ∧
  (∀ n ∈ S, ∀ d, d ∣ n → d ∈ S) ∧
  (∀ n ∈ S, ∀ m, n ∣ m → m ≤ 30 → m ∈ S)

def min_rubles_cost (S : set ℕ) : ℕ :=
  if is_valid_marking_strategy S then
    S.card
  else
    0

theorem minimal_cost_is_five :
  ∃ S : set ℕ, is_valid_marking_strategy S ∧ min_rubles_cost S = 5 := sorry

end minimal_cost_is_five_l611_611897


namespace range_of_a_l611_611378

open Set

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Icc (1 : ℝ) (2 : ℝ), x^2 - a ≥ 0)
  ∧ (∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0) → a ≤ -2 ∨ a = 1 :=
begin
  intro h,
  sorry  -- Skipping the proof.
end

end range_of_a_l611_611378


namespace minimum_oranges_condition_l611_611128

theorem minimum_oranges_condition (n : ℕ) :
  (∀ i j k : ℕ, i ≠ j → j ≠ k → i ≠ k → i < n → j < n → k < n → 
  let m := 1 in (3 * m / ((n-3) * m) < 0.05) → n ≥ 64) :=
begin
  intros h,
  sorry
end

end minimum_oranges_condition_l611_611128


namespace lines_parallel_iff_l611_611377

theorem lines_parallel_iff (a : ℝ) : (∀ x y : ℝ, x + 2*a*y - 1 = 0 ∧ (2*a - 1)*x - a*y - 1 = 0 → x = 1 ∧ x = -1 ∨ ∃ (slope : ℝ), slope = - (1 / (2 * a)) ∧ slope = (2 * a - 1) / a) ↔ (a = 0 ∨ a = 1/4) :=
by
  sorry

end lines_parallel_iff_l611_611377


namespace two_vertical_asymptotes_l611_611681

def g (x k : ℝ) := (x^3 - 2*x + k) / (x^3 - 7*x^2 + 14*x - 8)

theorem two_vertical_asymptotes (k : ℝ) :
  (∃ a₁ a₂ : ℝ, a₁ ≠ a₂ ∧ 
    (∀ x : ℝ, g x k = (x - a₁) * (x - a₂)) ∧ 
    (∀ b : ℝ, g b k = 0 → b = a₁ ∨ b = a₂)) ↔ 
  (k = 1 ∨ k = -4) :=
sorry

end two_vertical_asymptotes_l611_611681


namespace find_x_coordinate_l611_611433

-- Definition of the point and conditions
def point_dist_eq (x y : ℝ) : Prop :=
  (abs y = abs x) ∧ (abs y = abs (x + y - 4) / real.sqrt 2)

-- Main theorem stating the required x-coordinate
theorem find_x_coordinate (x y : ℝ) (h : point_dist_eq x y) : x = 2 :=
by 
-- Placeholder for the proof
sorry

end find_x_coordinate_l611_611433


namespace diane_allison_age_ratio_l611_611978

theorem diane_allison_age_ratio 
( diane_age_now : ℕ )
( alex_age_now : ℕ )
( allison_age_now : ℕ )
( diane_age_when_30 : ℕ )
( alex_age_when_30 : ℕ )
( allison_age_when_30 : ℕ )
( sum_ages_now : ℕ ) :
diane_age_now = 16 →
sum_ages_now = 47 →
diane_age_when_30 = 30 →
alex_age_when_30 = 60 →
alex_age_now + allison_age_now = sum_ages_now →
alex_age_now = alex_age_when_30 - (diane_age_when_30 - diane_age_now) →
allison_age_now = sum_ages_now - alex_age_now →
allison_age_when_30 = allison_age_now + (diane_age_when_30 - diane_age_now) →
(diane_age_when_30 / allison_age_when_30) = 2 :=
begin
  intros h1 h2 h3 h4 h5 h6 h7 h8,
  sorry
end

end diane_allison_age_ratio_l611_611978


namespace max_b_value_l611_611083

-- Definitions of the functions f and g
def f (x a : ℝ) := (3 / 2) * x ^ 2 - 2 * a * x
def g (x a b : ℝ) := a ^ 2 * Real.log x + b

-- Theorem statement
theorem max_b_value (a : ℝ) (b : ℝ) (x₀ : ℝ) :
  a > 0 ∧ f x₀ a = g x₀ a b ∧ (3 * x₀ - 2 * a) = (a ^ 2 / x₀) →
  b ≤ 1 / (2 * Real.exp 2) :=
sorry

end max_b_value_l611_611083


namespace min_oranges_picked_l611_611121

noncomputable def minimum_oranges (n : ℕ) : Prop :=
  ∀ (m : ℕ → ℕ), (∀ i j k : ℕ, i ≠ j ∧ j ≠ k ∧ i ≠ k → (m i + m j + m k : ℝ) / (∑ l : ℕ in finset.univ \ {i, j, k}, m l) < 0.05) → n ≥ 64

theorem min_oranges_picked : ∃ n, minimum_oranges n := by
  use 64
  sorry

end min_oranges_picked_l611_611121


namespace z_in_third_quadrant_l611_611325

open Complex

noncomputable def i : ℂ := Complex.I

def z : ℂ := (1 + 2 * i)⁻¹ * (-i)

theorem z_in_third_quadrant :
  (z.re < 0) ∧ (z.im < 0) := by
sorry

end z_in_third_quadrant_l611_611325


namespace total_words_read_l611_611009

/-- Proof Problem Statement:
  Given the following conditions:
  - Henri has 8 hours to watch movies and read.
  - He watches one movie for 3.5 hours.
  - He watches another movie for 1.5 hours.
  - He watches two more movies with durations of 1.25 hours and 0.75 hours, respectively.
  - He reads for the remaining time after watching movies.
  - For the first 30 minutes of reading, he reads at a speed of 12 words per minute.
  - For the following 20 minutes, his reading speed decreases to 8 words per minute.
  - In the last remaining minutes, his reading speed increases to 15 words per minute.
  Prove that the total number of words Henri reads during his free time is 670.
--/
theorem total_words_read : 8 * 60 - (7 * 60) = 60 ∧
  (30 * 12) + (20 * 8) + ((60 - 30 - 20) * 15) = 670 :=
by
  sorry

end total_words_read_l611_611009


namespace sum_of_squares_iff_even_difference_l611_611057

theorem sum_of_squares_iff_even_difference (m n : ℕ) (hm : m > 0) (hn : n > 0) : 
  (∃ a b c d : ℕ, 5^n + 5^m = a^2 + b^2 = c^2 + d^2) ↔ (n - m) % 2 = 0 := sorry

end sum_of_squares_iff_even_difference_l611_611057


namespace volume_ratio_of_cubes_l611_611974

theorem volume_ratio_of_cubes:
  (let edgeL1 : ℝ := 10
   let edgeL2_feet : ℝ := 5
   let edgeL2_inches : ℝ := edgeL2_feet * 12
   let ratio : ℝ := (edgeL1 / edgeL2_inches) ^ 3
   in ratio
  ) = 1 / 216 := sorry

end volume_ratio_of_cubes_l611_611974


namespace number_of_bad_arrangements_l611_611529

-- Definitions based on the problem statement
def is_consecutive_sum (circle : List ℕ) (s : ℕ) : Prop :=
  ∃ (start len : ℕ), 0 ≤ start ∧ start < circle.length ∧ 1 ≤ len ∧ len ≤ circle.length ∧
  s = (List.take len (List.drop start (circle ++ circle))).sum

def is_bad_arrangement (circle : List ℕ) : Prop :=
  ∃ (n ∈ (List.range 17).tail), ¬ is_consecutive_sum circle n

-- The proof problem statement translating the solution into Lean
theorem number_of_bad_arrangements : 
  (∃ circles : List (List ℕ), circles.length = 3 ∧ 
    ∀ circle ∈ circles, is_bad_arrangement circle ∧ 
    (circle.to_finset = {1, 2, 3, 4, 6}.to_finset) ∧ 
    ∀ circle1 circle2 ∈ circles, circle1 ≠ circle2 → 
    ¬ (circle1.is_rotational_variation_of circle2 ∨ 
       circle1.is_reflection of circle2))
sorry

end number_of_bad_arrangements_l611_611529


namespace fishing_tomorrow_l611_611827

-- Conditions
def every_day_fishers : Nat := 7
def every_other_day_fishers : Nat := 8
def every_three_days_fishers : Nat := 3
def yesterday_fishers : Nat := 12
def today_fishers : Nat := 10

-- Determine the number who will fish tomorrow
def fishers_tomorrow : Nat :=
  let every_day_tomorrow := every_day_fishers
  let every_three_day_tomorrow := every_three_days_fishers
  let every_other_day_yesterday := yesterday_fishers - every_day_fishers
  let every_other_day_tomorrow := every_other_day_fishers - every_other_day_yesterday
  every_day_tomorrow + every_three_day_tomorrow + every_other_day_tomorrow

theorem fishing_tomorrow : fishers_tomorrow = 15 :=
  by
    -- skipping the actual proof with sorry
    sorry

end fishing_tomorrow_l611_611827


namespace question_proof_l611_611461

-- Given initial conditions
def a_0 : ℝ := -1
def b_0 : ℝ := 2

-- Recursive definitions
def a (n : ℕ) : ℝ :=
  match n with
  | 0 => a_0
  | n+1 => a n * b n + sqrt ((a n)^2 + (b n)^2)

def b (n : ℕ) : ℝ :=
  match n with
  | 0 => b_0
  | n+1 => a n * b n - sqrt ((a n)^2 + (b n)^2)

-- The main proof problem
theorem question_proof : (1 / a 100) + (1 / b 100) = -0.5 := 
  sorry

end question_proof_l611_611461


namespace part_one_part_two_l611_611396

theorem part_one (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℤ) (m : ℤ)
  (h : (1 + m * x) ^ 8 = a₀ + a₁ * x + a₂ * x ^ 2 + a₃ * x ^ 3 + a₄ * x ^ 4 + a₅ * x ^ 5 + a₆ * x ^ 6 + a₇ * x ^ 7 + a₈ * x ^ 8)
  (h₁ : a₃ = -56) :
  m = -1 := sorry

theorem part_two (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℤ)
  (h₀ : (1 - x) ^ 8 = a₀ + a₁ * x + a₂ * x ^ 2 + a₃ * x ^ 3 + a₄ * x ^ 4 + a₅ * x ^ 5 + a₆ * x ^ 6 + a₇ * x ^ 7 + a₈ * x ^ 8)
  (h₁ : a₃ = -56)
  (hm : (m = -1))
  (h₂ : ∑ i in (Finset.range 9), a i = 0) :
  (a₀ + a₂ + a₄ + a₆ + a₈) ^ 2 - (a₁ + a₃ + a₅ + a₇) ^ 2 = 0 := sorry

end part_one_part_two_l611_611396


namespace open_locks_indices_l611_611187

def is_multiple (k m : ℕ) : Prop := m % k = 0

def toggle (state : Bool) : Bool := !state

def apply_operations (n : ℕ) (initial_state : List Bool) : List Bool :=
  List.foldl (λ state k =>
    List.mapWithIndex (λ m lock =>
      if is_multiple k (m + 1) then toggle lock else lock) state)
  initial_state (List.range n).map (λ x => x + 1)

def perfect_squares (n : ℕ) : List ℕ :=
  List.filter (λ m => ∃ k : ℕ, k * k = m) (List.range n).map (λ x => x + 1)

theorem open_locks_indices (n : ℕ) :
  { i // i < n ∧ (apply_operations n (List.repeat true n)).nth i = some false } =
  { i // i < n ∧ i ∈ perfect_squares n } :=
sorry

end open_locks_indices_l611_611187


namespace a_n_divisible_by_2013_a_n_minus_207_is_cube_l611_611311

theorem a_n_divisible_by_2013 (n : ℕ) (h : n ≥ 1) : 2013 ∣ (4 ^ (6 ^ n) + 1943) :=
by sorry

theorem a_n_minus_207_is_cube (n : ℕ) : (∃ k : ℕ, 4 ^ (6 ^ n) + 1736 = k^3) ↔ (n = 1) :=
by sorry

end a_n_divisible_by_2013_a_n_minus_207_is_cube_l611_611311


namespace area_DEF_eq_2_l611_611427

open EuclideanGeometry

-- Definitions and conditions based on the problem description
def A : Point := ⟨0, 4⟩
def B : Point := ⟨0, 0⟩
def C : Point := ⟨6, 0⟩

def midpoint (p1 p2 : Point) : Point := 
  ⟨(p1.x + p2.x) / 2, (p1.y + p2.y) / 2⟩

def D := midpoint A B
def E := midpoint A C
def F := midpoint B C

-- Calculate the area of triangle using vertices D, E, F
def triangle_area (p1 p2 p3 : Point) : ℝ := 
  0.5 * abs ((p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y)))

theorem area_DEF_eq_2 : 
  triangle_area D E F = 2 := 
by 
  -- Definitions for points should align with the provided Lean statement,
  -- But here we skip the proof details following the procedures.
  sorry

end area_DEF_eq_2_l611_611427


namespace inheritance_division_l611_611914

variables {M P Q R : ℝ} {p q r : ℕ}

theorem inheritance_division (hP : P < 99 * (p : ℝ))
                             (hR : R > 10000 * (r : ℝ))
                             (hM : M = P + Q + R)
                             (hRichPoor : R ≥ P) : 
                             R ≥ 100 * P := 
sorry

end inheritance_division_l611_611914


namespace tod_drive_time_l611_611966

def total_distance (dist_north dist_west: ℝ) : ℝ :=
  dist_north + dist_west

def travel_time (distance speed: ℝ) : ℝ :=
  distance / speed

theorem tod_drive_time (dist_north dist_west speed: ℝ) (h1: dist_north = 55) 
  (h2: dist_west = 95) (h3: speed = 25) : 
  travel_time (total_distance dist_north dist_west) speed = 6 := by
  sorry

end tod_drive_time_l611_611966


namespace passes_to_center_more_than_left_l611_611616

-- Given conditions
variables {L R C X : ℕ}
axiom L_def : L = 12
axiom R_def : R = 2 * L
axiom total_passes : L + R + C = 50
axiom C_def : C = L + X

-- Proof statement
theorem passes_to_center_more_than_left : X = 2 :=
by
  -- Using the conditions to establish the proof
  rw [L_def, R_def, C_def] at total_passes
  sorry

end passes_to_center_more_than_left_l611_611616


namespace exists_subset_T_l611_611452

theorem exists_subset_T (m n : ℕ) (a : ℕ → ℕ) (h_m : 1 ≤ m) (h_n : 1 ≤ n) (h_sorted : ∀ i j, i < j → a i < a j) :
  ∃ T : set ℕ, T.finite ∧ T.card ≤ 1 + (a n - a 1) / (2 * n + 1) ∧
  ∀ i, 1 ≤ i ∧ i ≤ m → ∃ t ∈ T, ∃ s ∈ {s | -↑n ≤ s ∧ s ≤ ↑n}, a i = t + s := sorry

end exists_subset_T_l611_611452


namespace asafa_and_florence_times_l611_611222

theorem asafa_and_florence_times (speed_asafa : ℕ) (PQ QR RS : ℕ) (speed_asafa_val : speed_asafa = 21)
  (distance_PQ : PQ = 8) (distance_QR : QR = 15) (distance_RS : RS = 7) :
  let T_asafa := (PQ + QR + RS) / speed_asafa in
  let PR := Real.sqrt (QR^2 + PQ^2) in
  let speed_florence := (PR + RS) / T_asafa in
  let time_asafa_RS := RS / speed_asafa in 
  let time_florence_RS := RS / speed_florence in
  (time_florence_RS - time_asafa_RS) * 60 = 5 := 
sorry

end asafa_and_florence_times_l611_611222


namespace find_x_l611_611592

theorem find_x (x : ℝ) : 
  0.65 * x = 0.20 * 682.50 → x = 210 := 
by 
  sorry

end find_x_l611_611592


namespace proof_20_exp_l611_611394

-- Definitions based on conditions
variables (a b : ℝ)
axiom ha : 100^a = 7
axiom hb : 100^b = 11

-- Proof statement
theorem proof_20_exp : 20^((1 - a - b) / (2 * (1 - b))) = 100 / 77 :=
by sorry

end proof_20_exp_l611_611394


namespace perfect_squares_with_odd_digits_l611_611846

theorem perfect_squares_with_odd_digits : 
  ∃ n : ℕ, n = 2 ∧ (∀ d ∈ (digits 10 n), d % 2 = 1) := 
sorry

end perfect_squares_with_odd_digits_l611_611846


namespace probability_event_l611_611348

def uniformProbability (a b : ℝ) (h : a < b) (P : set ℝ → Prop) : ℝ :=
  (∫ x in a..b, if P x then 1 else 0) / (b - a)

theorem probability_event : 
  uniformProbability 0 2 (by linarith : 0 < 2) (λ x, 3 * x - 2 ≥ 0) = 2 / 3 :=
by
  sorry

end probability_event_l611_611348


namespace equivalent_expression_for_35_mn_l611_611506

def P (m : ℤ) : ℤ := 5 ^ m
def Q (n : ℤ) : ℤ := 7 ^ n

theorem equivalent_expression_for_35_mn (m n : ℤ) : 
  (35 ^ (m * n)) = (P m ^ n) * (Q n ^ m) :=
by sorry

end equivalent_expression_for_35_mn_l611_611506


namespace power_function_condition_l611_611981

theorem power_function_condition (m α : ℝ) 
  (h: m * (1/2)^α = (real.sqrt 2) / 2) : m + α = 3 / 2 :=
sorry

end power_function_condition_l611_611981


namespace range_of_k_l611_611680

theorem range_of_k (k : ℤ) (x : ℤ) 
  (h1 : -4 * x - k ≤ 0) 
  (h2 : x = -1 ∨ x = -2) : 
  8 ≤ k ∧ k < 12 :=
sorry

end range_of_k_l611_611680


namespace cosine_dihedral_angle_l611_611046

-- Define the tetrahedron with the given angles
structure Tetrahedron :=
  (O A B C : Point)
  (angle_AOB : ℝ)
  (angle_AOC : ℝ)
  (angle_BOC : ℝ)

-- Define the angles in the tetrahedron
def given_tetrahedron : Tetrahedron :=
{ O := ⟨0, 0, 0⟩,
  A := ⟨1, 0, 0⟩,
  B := ⟨0, 1, 0⟩,
  C := ⟨0, 0, 1⟩,
  angle_AOB := 45,
  angle_AOC := 30,
  angle_BOC := 30 }

-- Statement to prove the cosine value of the dihedral angle
theorem cosine_dihedral_angle (T : Tetrahedron) (α : ℝ) : 
  T = given_tetrahedron → cos α = 2 * sqrt 2 - 3 :=
by 
  sorry

end cosine_dihedral_angle_l611_611046


namespace smaller_angle_at_8_oclock_l611_611386

def degrees_per_hour : ℝ := 360 / 12

def angle_at_8_oclock : ℝ := 8 * degrees_per_hour

def full_circle : ℝ := 360

theorem smaller_angle_at_8_oclock :
  ∀ (time : ℝ), time = 8 → min angle_at_8_oclock (full_circle - angle_at_8_oclock) = 120 := 
by 
  sorry

end smaller_angle_at_8_oclock_l611_611386


namespace portion_apples_weight_fraction_l611_611235

-- Given conditions
def total_apples : ℕ := 28
def total_weight_kg : ℕ := 3
def number_of_portions : ℕ := 7

-- Proof statement
theorem portion_apples_weight_fraction :
  (1 / number_of_portions = 1 / 7) ∧ (3 / number_of_portions = 3 / 7) :=
by
  -- Proof goes here
  sorry

end portion_apples_weight_fraction_l611_611235


namespace prob_vw_condition_is_zero_l611_611882

noncomputable def prob_vw_condition : Prop :=
  let roots := (λ k : ℕ, complex.exp (2 * real.pi * complex.I * k / 2021)) in
  ∀ v w : complex, v ≠ w → v ∈ set.range roots → w ∈ set.range roots → (complex.abs (v + w) ≥ real.sqrt (2 + real.sqrt 5)) = false

theorem prob_vw_condition_is_zero :
  prob_vw_condition := sorry

end prob_vw_condition_is_zero_l611_611882


namespace oranges_min_count_l611_611144

theorem oranges_min_count (n : ℕ) (m : ℕ → ℝ) 
  (h : ∀ i j k, m i + m j + m k < 0.05 * ∑ l in (finset.univ \ {i, j, k}), m l) : 
  64 ≤ n :=
sorry

end oranges_min_count_l611_611144


namespace work_completing_days_l611_611595

/-
Problem Statement:
A and B can complete a work in 15 days and 10 days respectively.
They started doing the work together but after some days B had to leave
and A alone completed the remaining work. The whole work was completed in 12 days.
Prove that A and B worked together for 2 days before B left.
-/

theorem work_completing_days {W : ℝ} (A B : ℝ) (x : ℝ) (H1 : A = W / 15) (H2 : B = W / 10) (H3 : (12 - x) * (W / 15) + x * (W / 6) = W) : x = 2 :=
begin
  -- The proof would go here.
  sorry
end

end work_completing_days_l611_611595


namespace find_second_divisor_l611_611244

theorem find_second_divisor (k : ℕ) (d : ℕ) 
  (h1 : k % 5 = 2)
  (h2 : k < 42)
  (h3 : k % 7 = 3)
  (h4 : k % d = 5) : d = 12 := 
sorry

end find_second_divisor_l611_611244


namespace area_of_triangle_ABC_is_zero_l611_611192

-- Definitions
def point := (ℝ × ℝ)
def A : point := (1, 0)
def B : point := (2, 1)
def C : point := (2, 1)

-- The formula for the area of a triangle with points (x1, y1), (x2, y2), and (x3, y3)
def triangle_area (P Q R : point) : ℝ :=
  let (x1, y1) := P in
  let (x2, y2) := Q in
  let (x3, y3) := R in
  (1 / 2) * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

-- Proof goal
theorem area_of_triangle_ABC_is_zero : triangle_area A B C = 0 :=
by
  sorry

end area_of_triangle_ABC_is_zero_l611_611192


namespace power_inequality_l611_611100

theorem power_inequality (n : ℕ) (x : ℝ) (hn : n ≥ 2) (hx : abs x < 1) : 
  2^n > (1 - x)^n + (1 + x)^n := 
sorry

end power_inequality_l611_611100


namespace combinations_count_l611_611081

theorem combinations_count:
  let valid_a (a: ℕ) := a < 1000 ∧ a % 29 = 7
  let valid_b (b: ℕ) := b < 1000 ∧ b % 47 = 22
  let valid_c (c: ℕ) (a b: ℕ) := c < 1000 ∧ c = (a + b) % 23 
  ∃ (a b c: ℕ), valid_a a ∧ valid_b b ∧ valid_c c a b :=
  sorry

end combinations_count_l611_611081


namespace angle_phi_l611_611184

variables (u v w : ℝ³)
variables (φ : ℝ)

-- Conditions
def cond_norm_u : ∥u∥ = 2 := sorry
def cond_norm_v : ∥v∥ = 2 := sorry
def cond_norm_w : ∥w∥ = 3 := sorry
def cond_relation : u × (u × w) + 2 • v = 0 := sorry

-- Goal
theorem angle_phi (u v w : ℝ³) (φ : ℝ)
  (h1 : ∥u∥ = 2)
  (h2 : ∥v∥ = 2)
  (h3 : ∥w∥ = 3)
  (h4 : u × (u × w) + 2 • v = 0) :
  φ = real.arccos (√5 / 3) ∨ φ = real.arccos (-√5 / 3) :=
sorry

end angle_phi_l611_611184


namespace min_oranges_picked_l611_611123

noncomputable def minimum_oranges (n : ℕ) : Prop :=
  ∀ (m : ℕ → ℕ), (∀ i j k : ℕ, i ≠ j ∧ j ≠ k ∧ i ≠ k → (m i + m j + m k : ℝ) / (∑ l : ℕ in finset.univ \ {i, j, k}, m l) < 0.05) → n ≥ 64

theorem min_oranges_picked : ∃ n, minimum_oranges n := by
  use 64
  sorry

end min_oranges_picked_l611_611123


namespace simplify_exponents_l611_611917

theorem simplify_exponents (x : ℝ) : x^5 * x^3 = x^8 :=
by
  sorry

end simplify_exponents_l611_611917


namespace fishers_tomorrow_l611_611787

-- Definitions based on conditions
def people_every_day : ℕ := 7
def people_every_other_day : ℕ := 8
def people_every_three_days : ℕ := 3
def people_yesterday : ℕ := 12
def people_today : ℕ := 10

-- Theorem to be proved
theorem fishers_tomorrow (people_every_day people_every_other_day people_every_three_days people_yesterday people_today : ℕ) : 
  people_every_day = 7 ∧ 
  people_every_other_day = 8 ∧ 
  people_every_three_days = 3 ∧
  people_yesterday = 12 ∧ 
  people_today = 10 →
  15 := 
by {
  sorry
}

end fishers_tomorrow_l611_611787


namespace max_underwear_pairs_l611_611196

-- Define the weights of different clothing items
def weight_socks : ℕ := 2
def weight_underwear : ℕ := 4
def weight_shirt : ℕ := 5
def weight_shorts : ℕ := 8
def weight_pants : ℕ := 10

-- Define the washing machine limit
def max_weight : ℕ := 50

-- Define the current load of clothes Tony plans to wash
def current_load : ℕ :=
  1 * weight_pants +
  2 * weight_shirt +
  1 * weight_shorts +
  3 * weight_socks

-- State the theorem regarding the maximum number of additional pairs of underwear
theorem max_underwear_pairs : 
  current_load ≤ max_weight →
  (max_weight - current_load) / weight_underwear = 4 :=
by
  sorry

end max_underwear_pairs_l611_611196


namespace fishers_tomorrow_l611_611791

-- Definitions based on conditions
def people_every_day : ℕ := 7
def people_every_other_day : ℕ := 8
def people_every_three_days : ℕ := 3
def people_yesterday : ℕ := 12
def people_today : ℕ := 10

-- Theorem to be proved
theorem fishers_tomorrow (people_every_day people_every_other_day people_every_three_days people_yesterday people_today : ℕ) : 
  people_every_day = 7 ∧ 
  people_every_other_day = 8 ∧ 
  people_every_three_days = 3 ∧
  people_yesterday = 12 ∧ 
  people_today = 10 →
  15 := 
by {
  sorry
}

end fishers_tomorrow_l611_611791


namespace snakes_not_hiding_l611_611636

theorem snakes_not_hiding (total_snakes hiding_snakes not_hiding: ℕ)
  (h1 : total_snakes = 95)
  (h2 : hiding_snakes = 64) :
  not_hiding = total_snakes - hiding_snakes :=
begin
  sorry
end

-- Apply specific values to verify the specific proof
example : ∃ not_hiding, snakes_not_hiding 95 64 not_hiding :=
begin
  use 31,
  intros,
  exact eq.refl 31,
end

end snakes_not_hiding_l611_611636


namespace faster_train_length_is_150_l611_611585

def speed_faster_train_kmph : ℝ := 72
def speed_slower_train_kmph : ℝ := 36
def time_seconds : ℝ := 15

noncomputable def length_faster_train : ℝ :=
  let relative_speed_kmph := speed_faster_train_kmph - speed_slower_train_kmph
  let relative_speed_mps := relative_speed_kmph * 1000 / 3600
  relative_speed_mps * time_seconds

theorem faster_train_length_is_150 :
  length_faster_train = 150 := by
sorry

end faster_train_length_is_150_l611_611585


namespace remaining_oranges_l611_611888

theorem remaining_oranges (initial_oranges : ℕ) (sold_to_mary : ℕ) (sold_to_joe : ℕ) (donated_to_charity : ℕ) (gifted_to_neighbor : ℕ) :
    initial_oranges = 150 →
    sold_to_mary = (initial_oranges * 20) / 100 →
    let remaining_after_mary := initial_oranges - sold_to_mary in
    sold_to_joe = (remaining_after_mary * 30) / 100 →
    let remaining_after_joe := remaining_after_mary - sold_to_joe in
    donated_to_charity = (remaining_after_joe * 10) / 100 →
    let remaining_after_charity := remaining_after_joe - donated_to_charity in
    gifted_to_neighbor = 1 →
    let remaining_final := remaining_after_charity - gifted_to_neighbor in
    remaining_final = 75 :=
begin
  intros h_initial h_sold_to_mary h_sold_to_joe h_donated_to_charity h_gifted_to_neighbor,
  rw [h_sold_to_mary, h_sold_to_joe, h_donated_to_charity, h_gifted_to_neighbor],
  dsimp,
  norm_num,
end

end remaining_oranges_l611_611888


namespace crop_fraction_brought_to_AD_l611_611627

-- Definitions based only on the conditions
def trapezoid_A_lengths (AB: ℝ) (BC: ℝ) (CD: ℝ) (DA: ℝ) :=
AB = 120 ∧ BC = 80 ∧ CD = 120 ∧ DA = 160

def angle_ABC_45 (∠ABC: ℝ) := ∠ABC = 45

-- Main theorem statement
theorem crop_fraction_brought_to_AD (AB BC CD DA ∠ABC: ℝ)
  (h_lengths: trapezoid_A_lengths AB BC CD DA)
  (h_angle: angle_ABC_45 ∠ABC):
  1 / 2 = 1 / 2 :=
by
  sorry

end crop_fraction_brought_to_AD_l611_611627


namespace sector_area_correct_l611_611225

def sector_area (arc_length diameter : ℝ) : ℝ :=
  let radius := diameter / 2
  (1 / 2) * arc_length * radius

theorem sector_area_correct :
  sector_area 30 16 = 120 :=
by
  sorry

end sector_area_correct_l611_611225


namespace common_external_tangent_length_l611_611588

theorem common_external_tangent_length (R r : ℝ) (hR : 0 < R) (hr : 0 < r) :
  let d := R + r in 
  let T := 2 * Real.sqrt (R * r) in
  ∀ (O₁ O₂ : ℝ), O₁ ≠ O₂ → ∃ t, Real.dist O₁ t = R ∧ Real.dist O₂ t = r ∧ Real.dist O₁ O₂ = d → 
  t = T := 
by
  sorry

end common_external_tangent_length_l611_611588


namespace geometric_progression_condition_l611_611652

theorem geometric_progression_condition
  (a b k : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : k > 0)
  (a_seq : ℕ → ℝ) 
  (h_def : ∀ n, a_seq (n+2) = k * a_seq n * a_seq (n+1)) :
  (a_seq 1 = a ∧ a_seq 2 = b) ↔ a_seq 1 = a_seq 2 :=
by
  sorry

end geometric_progression_condition_l611_611652


namespace distinct_sets_fat_pair_l611_611451

-- Let m, n, and k be positive integers greater than 1.
variables {m n k : ℕ}
variable (S : ℕ → ℕ → ℕ) -- S(i, j) denotes the number of elements in S strictly between i and j.

-- Define a fat pair condition for sets X and Y.
def fat_pair (X Y : finset ℕ) : Prop :=
  ∀ i j ∈ X ∩ Y, S i j % n = S i j % n

-- Assuming m sets of k positive integers where no two form a fat pair.
variable (sets : finset (finset ℕ))
variable (card_sets : sets.card = m) -- There are m distinct sets.
variable (card_elements : ∀ x ∈ sets, x.card = k) -- Each set is of size k.
variable (no_fat_pairs : ∀ X Y ∈ sets, X ≠ Y → ¬fat_pair X Y)

-- Prove that m < n^(k-1).
theorem distinct_sets_fat_pair : m < n ^ (k - 1) :=
sorry

end distinct_sets_fat_pair_l611_611451


namespace prob_same_color_seven_red_and_five_green_l611_611743

noncomputable def probability_same_color (red_plat : ℕ) (green_plat : ℕ) : ℚ :=
  let total_plates := red_plat + green_plat
  let total_pairs := (total_plates.choose 2) -- total ways to select 2 plates
  let red_pairs := (red_plat.choose 2) -- ways to select 2 red plates
  let green_pairs := (green_plat.choose 2) -- ways to select 2 green plates
  (red_pairs + green_pairs) / total_pairs

theorem prob_same_color_seven_red_and_five_green :
  probability_same_color 7 5 = 31 / 66 :=
by
  sorry

end prob_same_color_seven_red_and_five_green_l611_611743


namespace fishing_tomorrow_l611_611773

theorem fishing_tomorrow 
  (P1 : ∀ day : ℕ, 7 ∈ {p | p goes fishing on day})
  (P2 : ∀ day : ℕ, day % 2 = 0 → 8 ∈ {p | p goes fishing on day})
  (P3 : ∀ day : ℕ, day % 3 = 0 → 3 ∈ {p | p goes fishing on day})
  (P4 : ℕ)
  (yesterday : ℕ)
  (today : ℕ)
  (tomorrow : ℕ)
  (hyesterday : yesterday = 12)
  (htoday : today = 10)
  : tomorrow = 15 := by
  sorry

end fishing_tomorrow_l611_611773


namespace polynomial_solution_l611_611304

noncomputable def q (x : ℝ) : ℝ := x^4 - 18

theorem polynomial_solution (q : ℝ → ℝ) 
  (hq : ∀ x : ℝ, q(x^4) - q(x^4 - 3) = q(x)^3 + 18) : 
  q = (λ x, x^4 - 18) := 
begin
  sorry
end

end polynomial_solution_l611_611304


namespace only_option_A_is_monotonic_l611_611283

def is_monotonically_increasing_on_R (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

theorem only_option_A_is_monotonic :
  ∀ (f : ℝ → ℝ),
    (f = (λ x : ℝ, 2^x) ∨
     f = (λ x : ℝ, real.log x / real.log 2) ∨
     f = (λ x : ℝ, x^2) ∨
     f = (λ x : ℝ, -x^2))
    → (is_monotonically_increasing_on_R f ↔ f = (λ x : ℝ, 2^x)) :=
by sorry

end only_option_A_is_monotonic_l611_611283


namespace min_translation_a_l611_611720

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := sin(ω * x) ^ 2 - 1 / 2

theorem min_translation_a (ω a : ℝ) (hω : ω > 0) (ha : a > 0) 
  (h_period : ∀ x, f ω x = f ω (x + π / 2)) 
  (h_symmetry : ∀ x, f ω (x + a) = f ω (-x - a)) : 
  a = π / 8 :=
  sorry

end min_translation_a_l611_611720


namespace f_derivative_at_1_l611_611362

noncomputable def choose (n k : ℕ) : ℕ := Nat.choose n k

def f (n : ℕ) (x : ℝ) : ℝ :=
  ∑ r in Finset.range (n + 1), choose n r * (-1)^r * x^(2 * n - 1 + r)

theorem f_derivative_at_1 (n : ℕ) (hn : 0 < n) : (deriv (f n)) 1 = 0 := by
  sorry

end f_derivative_at_1_l611_611362


namespace prove_b_zero_l611_611865

variables {a b c : ℕ}

theorem prove_b_zero (h1 : ∃ (a b c : ℕ), a^5 + 4 * b^5 = c^5 ∧ c % 2 = 0) : b = 0 :=
sorry

end prove_b_zero_l611_611865


namespace training_weeks_l611_611054

variable (adoption_fee training_per_week cert_cost insurance_coverage out_of_pocket : ℕ)
variable (x : ℕ)

def adoption_fee_value : ℕ := 150
def training_per_week_cost : ℕ := 250
def certification_cost_value : ℕ := 3000
def insurance_coverage_percentage : ℕ := 90
def total_out_of_pocket : ℕ := 3450

theorem training_weeks :
  adoption_fee = adoption_fee_value →
  training_per_week = training_per_week_cost →
  cert_cost = certification_cost_value →
  insurance_coverage = insurance_coverage_percentage →
  out_of_pocket = total_out_of_pocket →
  (out_of_pocket = adoption_fee + training_per_week * x + (cert_cost * (100 - insurance_coverage)) / 100) →
  x = 12 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5] at h6
  sorry

end training_weeks_l611_611054


namespace alarm_sound_probability_l611_611555

open MeasureTheory

variable (Ω : Type) [MeasurableSpace Ω] (μ : Measure Ω)

-- Defining events A1 and A2 and their probabilities
variable (A1 A2 : Set Ω)
variable (hA1 : μ[A1] = 0.4)
variable (hA2 : μ[A2] = 0.4)
variable (indep : μ[A1 ∩ A2] = μ[A1] * μ[A2])

-- Definition and statement of the theorem
theorem alarm_sound_probability :
  (∀ A1 A2 : Set Ω, μ[A1] = 0.4 ∧ μ[A2] = 0.4 ∧ μ[A1 ∩ A2] = μ[A1] * μ[A2] → 
    μ[A1 ∪ A2] = 0.64) :=
by
  intros A1 A2 h
  let prob_neither := (1 - μ[A1]) * (1 - μ[A2])
  have h_neither : prob_neither = 0.36 :=
    by
      simp [hA1, hA2]
      norm_num

  let prob_at_least_one := 1 - prob_neither
  have h_at_least_one : prob_at_least_one = 0.64 :=
    by
      simp [h_neither]
      norm_num

  exact h_at_least_one

end alarm_sound_probability_l611_611555


namespace common_chord_bisects_BH_l611_611059

variables {Point : Type}

structure Quadrilateral (A B C D : Point) :=
  (non_cyclic_convex: Prop)

structure Orthocenter (A B D H: Point) :=
  (orthocenter: Prop)

structure PerpendicularFoot (A B C D P Q R X Y Z: Point) :=
  (P_on_BC: Prop)
  (Q_on_BD: Prop)
  (R_on_CD_extended: Prop)
  (X_on_AC: Prop)
  (Y_on_BC: Prop)
  (Z_on_BA_extended: Prop)

structure Circumcircle (Δ : Type) :=
  (common_chord: Prop)

variables (A B C D P Q R X Y Z H : Point)

theorem common_chord_bisects_BH
  (quad : Quadrilateral A B C D)
  (foot : PerpendicularFoot A B C D P Q R X Y Z)
  (ortho : Orthocenter A B D H) :
  (Circumcircle (Δ₁ : Triangle P Q R)) → 
  (Circumcircle (Δ₂ : Triangle X Y Z)) → 
  (common_chord Δ₁ Δ₂ (segment B H).midpoint) := 
begin
  sorry
end

end common_chord_bisects_BH_l611_611059


namespace odd_numbers_identifiable_l611_611912

def isIdentifiable (S : Set ℕ) := ∃ P : List (ℕ × ℕ × ℕ), ∀ initialState : Fin 101 → Bool, 
    (∀ pos = 1 → 
         (numberOfBalls initialState ∈ S ↔ initialState 1 = true)) 
  where
    numberOfBalls (state : Fin 101 → Bool) : ℕ :=
      Finset.card {i | state i}

theorem odd_numbers_identifiable : isIdentifiable {n | n % 2 = 1} :=
sorry

end odd_numbers_identifiable_l611_611912


namespace evaluate_at_two_l611_611872

def f (x : ℚ) : ℚ := (4 * x^2 + 6 * x + 10) / (x^2 - x + 5)
def g (x : ℚ) : ℚ := x - 2

theorem evaluate_at_two : f (g 2) + g (f 2) = 38 / 7 := by
  sorry

end evaluate_at_two_l611_611872


namespace simplify_div_expr_l611_611146

theorem simplify_div_expr (x : ℝ) (h : x = Real.sqrt 3) :
  ((x - 1) / x - (x - 2) / (x + 1)) / ((2 * x - 1) / (x^2 + 2 * x + 1)) = 1 + Real.sqrt 3 / 3 := by
sorry

end simplify_div_expr_l611_611146


namespace find_common_ratio_l611_611355

variable (a₁ : ℝ) (q : ℝ)

def S₁ (a₁ : ℝ) : ℝ := a₁
def S₃ (a₁ q : ℝ) : ℝ := a₁ + a₁ * q + a₁ * q ^ 2
def a₃ (a₁ q : ℝ) : ℝ := a₁ * q ^ 2

theorem find_common_ratio (h : 2 * S₃ a₁ q = S₁ a₁ + 2 * a₃ a₁ q) : q = -1 / 2 :=
by
  sorry

end find_common_ratio_l611_611355


namespace P_iff_nonQ_l611_611689

-- Given conditions
def P (x y : ℝ) : Prop := x^2 + y^2 = 0
def Q (x y : ℝ) : Prop := x ≠ 0 ∨ y ≠ 0
def nonQ (x y : ℝ) : Prop := x = 0 ∧ y = 0

-- Main statement
theorem P_iff_nonQ (x y : ℝ) : P x y ↔ nonQ x y :=
sorry

end P_iff_nonQ_l611_611689


namespace mn_bisects_pq_l611_611007

-- Declare all necessary geometric entities and properties
variables (circle1 circle2 : Circle) (A : Point) (M N P Q : Point)

-- Provided conditions
axiom circles_intersect_at_A : ∃ (B : Point), B ≠ A ∧ (on_circle circle1 A) ∧ (on_circle circle2 A)
axiom M_tangent_to_circle1 : tangent (line_through M A) circle1 M
axiom N_tangent_to_circle1 : tangent (line_through N A) circle1 N
axiom P_on_circle2 : ∃ (l : Line), (on_line l P) ∧ (on_line l A) ∧ (on_circle circle2 P) ∧ (l ≠ (line_through M A))
axiom Q_on_circle2 : ∃ (l : Line), (on_line l Q) ∧ (on_line l A) ∧ (on_circle circle2 Q) ∧ (l ≠ (line_through N A))

-- The proposition to be proved
theorem mn_bisects_pq : is_perpendicular_bisector (line_through M N) (segment P Q) :=
sorry

end mn_bisects_pq_l611_611007


namespace distance_between_towns_proof_l611_611933

noncomputable def distance_between_towns : ℕ :=
  let distance := 300
  let time_after_departure := 2
  let remaining_distance := 40
  let speed_difference := 10
  let total_distance_covered := distance - remaining_distance
  let speed_slower_train := 60
  let speed_faster_train := speed_slower_train + speed_difference
  let relative_speed := speed_slower_train + speed_faster_train
  distance

theorem distance_between_towns_proof 
  (distance : ℕ) 
  (time_after_departure : ℕ) 
  (remaining_distance : ℕ) 
  (speed_difference : ℕ) 
  (h1 : distance = 300) 
  (h2 : time_after_departure = 2) 
  (h3 : remaining_distance = 40) 
  (h4 : speed_difference = 10) 
  (speed_slower_train speed_faster_train relative_speed : ℕ)
  (h_speed_faster : speed_faster_train = speed_slower_train + speed_difference)
  (h_relative_speed : relative_speed = speed_slower_train + speed_faster_train) :
  distance = 300 :=
by {
  sorry
}

end distance_between_towns_proof_l611_611933


namespace intersection_A_B_l611_611374

def A : Set ℝ := {x | abs x <= 1}

def B : Set ℝ := {y | ∃ x : ℝ, y = x^2}

theorem intersection_A_B :
  (A ∩ B) = {x | 0 ≤ x ∧ x ≤ 1} := sorry

end intersection_A_B_l611_611374


namespace projection_theorem_l611_611179

noncomputable def projection_problem : Prop :=
  let w := (2 : ℝ, -1)
  let v1 := (2 : ℝ, 5)
  let proj_v1_on_w := (2/5 : ℝ, -1/5)
  let v2 := (3 : ℝ, 2)
  let proj_v2_on_w := (8/5 : ℝ, -4/5) in
  (proj_v1_on_w = let scale_factor := (v1.1 * w.1 + v1.2 * w.2) / ((w.1)^2 + (w.2)^2) in (scale_factor * w.1, scale_factor * w.2))
  → (proj_v2_on_w = let scale_factor := (v2.1 * w.1 + v2.2 * w.2) / ((w.1)^2 + (w.2)^2) in (scale_factor * w.1, scale_factor * w.2))

theorem projection_theorem : projection_problem :=
  sorry

end projection_theorem_l611_611179


namespace total_weight_in_pounds_l611_611911

/-- Definition of ounces per container and containers. -/
def ounces_per_container := 25
def number_of_containers := 4

/-- Definition of conversion rate from ounces to pounds. -/
def ounces_per_pound := 16

/-- The total weight of rice in pounds is calculated and stated here. -/
theorem total_weight_in_pounds 
  (h1 : ounces_per_container = 25) 
  (h2 : number_of_containers = 4) 
  (h3 : ounces_per_pound = 16) : 
  (ounces_per_container * number_of_containers) / ounces_per_pound = 6.25 :=
sorry

end total_weight_in_pounds_l611_611911


namespace max_value_of_y_l611_611565

theorem max_value_of_y : 
  ∃ y, 3 * y^2 + 30 * y - 90 = y * (y + 18) ∧ (∀ y', 3 * y'^2 + 30 * y' - 90 = y' * (y' + 18) → y' ≤ 5) := 
begin
  sorry
end

end max_value_of_y_l611_611565


namespace area_of_triangle_AGE_l611_611481

def Point := (ℝ × ℝ)

structure Square where
  A B C D : Point
  s : ℝ
  eq_AB_CD : ∥B - A∥ = s ∧ ∥C - B∥ = s ∧ ∥D - C∥ = s ∧ ∥A - D∥ = s
  eq_diag : ∥C - A∥ = s * sqrt 2 ∧ ∥B - D∥ = s * sqrt 2

structure Triangle where
  A B E : Point

structure Circumcircle where
  center : Point
  radius : ℝ
  circum_eq : ∀ (P : Point), P ∈ {A, B, E} → ∥P - center∥ = radius

noncomputable def area (P1 P2 P3 : Point) : ℝ :=
  (1/2) * abs (P1.1 * (P2.2 - P3.2) + P2.1 * (P3.2 - P1.2) + P3.1 * (P1.2 - P2.2))

theorem area_of_triangle_AGE : 
  ∀ (s : ℝ) (E G : Point),
  s = 5 → 
  ∥E - (5,0)∥ = 2 ∧ ∥(5,2) - E∥ = 3 →
  ∥G - (5,0)∥ <= 5 * sqrt 2 → -- G on the diagonal (BD)
  area (0,0) G (5,2) = 67.25 :=
by
  intros
  sorry

end area_of_triangle_AGE_l611_611481


namespace a_11_eq_l611_611730

variable {a : ℕ → ℝ}
def a_1 : a 1 = 7 := rfl
def a_3 : a 3 = 1 := rfl
def geom_seq : ∀ n, (1 / (a n + 1)) = (1 / (a 1 + 1)) * (4 : ℝ)^(n - 1) := sorry

theorem a_11_eq :
  a 11 = -127 / 128 := by
  sorry

end a_11_eq_l611_611730


namespace tom_books_read_in_may_l611_611967

def books_read_in_june := 6
def books_read_in_july := 10
def total_books_read := 18

theorem tom_books_read_in_may : total_books_read - (books_read_in_june + books_read_in_july) = 2 :=
by sorry

end tom_books_read_in_may_l611_611967


namespace max_projection_value_l611_611372

variables {V : Type*} [inner_product_space ℝ V]
variables (e₁ e₂ : V)
variables (h₁ : ∥e₁∥ = 2) (h₂ : ∥3 • e₁ + e₂∥ = 2)

theorem max_projection_value :
  ∃ (c : ℝ), c = -(4 * real.sqrt 2 / 3) ∧
             ∀ (v : V), (v ≠ 0) → (∃ (v' : V), e₁ = v + v' ∧ ∃ (k : ℝ), (projection v e₁).norm = c) :=
sorry

end max_projection_value_l611_611372


namespace part_a_part_b_minimal_value_29_l611_611450

-- Conditions from the problem
def α_pos_root : Real := (sqrt 21 - 1) / 2
def P (n : ℕ) (c : Fin (n+1) → ℕ) (x : Real) : Real := ∑ i in Finset.range (n+1), c i * x^i

-- Part (a): Prove that c_0 + c_1 + c_2 + ... + c_n ≡ 2 (mod 3)
theorem part_a (n : ℕ) (c : Fin (n+1) → ℕ) 
  (h_sum : P n c α_pos_root = 2015) : 
  (∑ i in Finset.range (n+1), c i) % 3 = 2 := 
sorry

-- Part (b): Prove that 29 is the minimum sum c_0 + c_1 + c_2 + ... + c_n
theorem part_b (n : ℕ) (c : Fin (n+1) → ℕ) 
  (h_sum : P n c α_pos_root = 2015) : 
  ∑ i in Finset.range (n+1), c i ≥ 29 := 
sorry

theorem minimal_value_29 (n : ℕ) (c : Fin (n+1) → ℕ) 
  (h_sum : P n c α_pos_root = 2015) : 
  ∑ i in Finset.range (n+1), c i = 29 := 
sorry

end part_a_part_b_minimal_value_29_l611_611450


namespace triangle_angle_B_l611_611421

theorem triangle_angle_B 
  (a b c : ℝ) (A B C : ℝ) 
  (h1 : 0 < A ∧ A < π)
  (h2 : 0 < B ∧ B < π)
  (h3 : 0 < C ∧ C < π)
  (h4 : a * sin B = b * sin A)
  (h5 : c * sin B = b * sin C)
  (h6 : 2 * b * cos B = a * cos C + c * cos A) : 
  B = π / 3 :=
by { sorry }

end triangle_angle_B_l611_611421


namespace sequence_first_five_terms_distinct_first_three_constant_sequence_exists_l611_611949

theorem sequence_first_five_terms (m : ℕ) (h : m = 5) (a : ℕ → ℕ) 
  (h1 : a 1 = m) 
  (h2 : ∀ n ≥ 2, a n ≤ n - 1) 
  (h3 : ∀ n ≥ 1, n ∣ (finset.range (n + 1)).sum a) : 
  a 1 = 5 ∧ a 2 = 1 ∧ a 3 = 0 ∧ a 4 = 2 ∧ a 5 = 2 := 
sorry

theorem distinct_first_three (m : ℕ) (a : ℕ → ℕ)
  (h1 : a 1 = m) 
  (h2 : ∀ n ≥ 2, a n ≤ n - 1) 
  (h3 : ∀ n ≥ 1, n ∣ (finset.range (n + 1)).sum a)
  (h4 : ∀ n ≥ 3, a n = a 3) 
  (h5 : a 1 ≠ a 2 ∧ a 1 ≠ a 3 ∧ a 2 ≠ a 3) :
  m = 2 ∨ m = 3 ∨ m = 4 :=
sorry

theorem constant_sequence_exists (m : ℕ) (a : ℕ → ℕ)
  (h1 : a 1 = m) 
  (h2 : ∀ n ≥ 2, a n ≤ n - 1) 
  (h3 : ∀ n ≥ 1, n ∣ (finset.range (n + 1)).sum a) :
  ∃ M : ℕ, ∀ n ≥ M, a n = a M :=
sorry

end sequence_first_five_terms_distinct_first_three_constant_sequence_exists_l611_611949


namespace tangent_line_eq_sum_of_zeros_positive_l611_611365

-- Definitions from the conditions
def f (a x : ℝ) : ℝ := (a * (x + 1)) / (Real.exp x) + (1 / 2) * x ^ 2

-- Part (1): Prove the equation of the tangent line when a = 1
theorem tangent_line_eq (x : ℝ) (h : x = -1) (f_a : ∀ x, f 1 x = (x + 1) / (Real.exp x) + (1 / 2) * x ^ 2) :
  let y := f 1 x in
  ∃ m b, y = m * (-1) + b ∧ ∀ x, y = (Real.exp 1 - 1) * x + (Real.exp 1 - 1 / 2) := 
sorry

-- Part (2): Prove that x₁ + x₂ > 0 when a < 0 and f has two distinct zeros
theorem sum_of_zeros_positive (a : ℝ) (ha : a < 0)
  (hx1 x2 : ℝ) (hf_zero : f a x1 = 0 ∧ f a x2 = 0) (hx1_ne_x2 : x1 ≠ x2) :
  x1 + x2 > 0 :=
sorry

end tangent_line_eq_sum_of_zeros_positive_l611_611365


namespace remove_500th_digit_larger_l611_611447

theorem remove_500th_digit_larger (n : ℕ) (h : n = 500) : 
  let frac := (3 : ℚ) / 7
  let dec := string.to_list (frac.to_string)
  let dec_head := dec.take (h - 1)
  let dec_tail := dec.drop h
  let new_num_str := dec_head ++ dec_tail
-- Assuming a function that converts the entire decimal string back to a rational number (for the purpose of this illustration):
  ∀ dec_num, dec_num.to_rat new_num_str > frac := 
begin
  sorry
end

end remove_500th_digit_larger_l611_611447


namespace distance_from_P_to_AD_is_correct_l611_611504

noncomputable def P_distance_to_AD : ℝ :=
  let A : ℝ × ℝ := (0, 6)
  let D : ℝ × ℝ := (0, 0)
  let C : ℝ × ℝ := (6, 0)
  let M : ℝ × ℝ := (3, 0)
  let radius1 : ℝ := 5
  let radius2 : ℝ := 6
  let circle1_eq := fun (x y : ℝ) => (x - 3)^2 + y^2 = 25
  let circle2_eq := fun (x y : ℝ) => x^2 + (y - 6)^2 = 36
  let P := (24/5, 18/5)
  let AD := fun x y : ℝ => x = 0
  abs ((P.fst : ℝ) - 0)

theorem distance_from_P_to_AD_is_correct :
  P_distance_to_AD = 24 / 5 := by
  sorry

end distance_from_P_to_AD_is_correct_l611_611504


namespace find_natural_numbers_l611_611665

theorem find_natural_numbers (n : ℕ) (x : ℕ) (y : ℕ) (hx : n = 10 * x + y) (hy : 10 * x + y = 14 * x) : n = 14 ∨ n = 28 :=
by
  sorry

end find_natural_numbers_l611_611665


namespace monotonous_positive_integers_1_to_11_l611_611048

theorem monotonous_positive_integers_1_to_11 :
  (9 + 2 * ∑ n in Finset.range 10 \ {0, 1}, Nat.choose 11 n) = 4081 := 
by
  sorry

end monotonous_positive_integers_1_to_11_l611_611048


namespace exists_k_inequality_l611_611073

theorem exists_k_inequality {n : ℕ} (a b : Fin n → ℂ) :
  ∃ k : Fin n, (∑ i, Complex.abs (a i - a k) ^ 2) ≤ (∑ i, Complex.abs (b i - a k) ^ 2) ∨
               (∑ i, Complex.abs (b i - b k) ^ 2) ≤ (∑ i, Complex.abs (a i - b k) ^ 2) := 
by
  sorry

end exists_k_inequality_l611_611073


namespace unique_value_of_W_l611_611439

theorem unique_value_of_W (T O W F U R : ℕ) (h1 : T = 8) (h2 : O % 2 = 0) (h3 : ∀ x y, x ≠ y → x = O → y = T → x ≠ O) :
  (T + T) * 10^2 + (W + W) * 10 + (O + O) = F * 10^3 + O * 10^2 + U * 10 + R → W = 3 :=
by
  sorry

end unique_value_of_W_l611_611439


namespace ellipse_equation_l611_611704

theorem ellipse_equation :
  ∃ (a b : ℝ), a = 2 ∧ b = sqrt 3 ∧ 
  (∀ x y : ℝ, (1 ≤ a ∧ a > b ∧ b > 0) → 
  ((x, y) = (1, 3 / 2) ∧ |sqrt(x^2 + y^2) - 1| + |sqrt(x^2 + y^2) + 1| = 4) → 
  (x^2 / 4 + y^2 / 3 = 1)) := sorry

end ellipse_equation_l611_611704


namespace oranges_min_number_l611_611113

theorem oranges_min_number (n : ℕ) 
  (h : ∀ (m : ℕ → ℝ), (∀ i j k : ℕ, i < n → j < n → k < n →
    i ≠ j → j ≠ k → i ≠ k → m i + m j + m k < 0.05 * (∑ l in (finset.range n \ {i,j,k}), m l))) : 
  n ≥ 64 :=
sorry

end oranges_min_number_l611_611113


namespace mean_temperature_is_0_5_l611_611507

def temperatures : List ℝ := [-3.5, -2.25, 0, 3.75, 4.5]

theorem mean_temperature_is_0_5 :
  (temperatures.sum / temperatures.length) = 0.5 :=
by
  sorry

end mean_temperature_is_0_5_l611_611507


namespace centroid_of_triangle_l611_611853

open Classical

variable {A B C G A' B' C' : Type} [OrderedCommRing G]

-- Define properties for points A', B', C', and G
variable (onBC : line[BC] A')
variable (onAC : line[AC] B')
variable (onAB : line[AB] C')
variable (concurrent : CoConcurrent AA' BB' CC' G)
variable (ratio_condition : AG / GA' = BG / GB' ∧ BG / GB' = CG / GC')

-- State the theorem
theorem centroid_of_triangle 
  (h1 : onBC)
  (h2 : onAC)
  (h3 : onAB)
  (h4 : concurrent)
  (h5 : ratio_condition) :
  G = centroid ABC := 
sorry

end centroid_of_triangle_l611_611853


namespace number_of_whole_numbers_between_sqrt10_and_3pi_l611_611393

noncomputable def sqrt_10 : ℝ := Real.sqrt 10
noncomputable def three_pi : ℝ := 3 * Real.pi

theorem number_of_whole_numbers_between_sqrt10_and_3pi : 
  let lower_bound := Real.ceil sqrt_10,
      upper_bound := Real.floor three_pi in
  (upper_bound - lower_bound + 1) = 6 :=
by
  sorry

end number_of_whole_numbers_between_sqrt10_and_3pi_l611_611393


namespace solve_inequality_l611_611666

open Set Real

def condition1 (x : ℝ) : Prop := 6 * x + 2 < (x + 2) ^ 2
def condition2 (x : ℝ) : Prop := (x + 2) ^ 2 < 8 * x + 4

theorem solve_inequality (x : ℝ) : condition1 x ∧ condition2 x ↔ x ∈ Ioo (2 + Real.sqrt 2) 4 := by
  sorry

end solve_inequality_l611_611666


namespace percentage_increase_in_sales_l611_611576

open Real

theorem percentage_increase_in_sales
  (P S : ℝ) (h₁ : P > 0) (h₂ : S > 0)
  (reduction : ℝ := 0.6)
  (net_effect : ℝ := 1.08) :
  let new_price := P * reduction,
      new_sales := S * (1 + 80 / 100) in
  new_price * new_sales = net_effect * P * S :=
by
  sorry

end percentage_increase_in_sales_l611_611576


namespace oranges_min_count_l611_611142

theorem oranges_min_count (n : ℕ) (m : ℕ → ℝ) 
  (h : ∀ i j k, m i + m j + m k < 0.05 * ∑ l in (finset.univ \ {i, j, k}), m l) : 
  64 ≤ n :=
sorry

end oranges_min_count_l611_611142


namespace arrangement_of_ketchup_mustard_l611_611038

theorem arrangement_of_ketchup_mustard :
  let total_bottles := 10
  let K := 3
  let M := 7
  num_valid_arrangements (total_bottles, K, M) = 22 := 
begin
  sorry
end

end arrangement_of_ketchup_mustard_l611_611038


namespace nate_total_time_l611_611893

/-- Definitions for the conditions -/
def sectionG : ℕ := 18 * 12
def sectionH : ℕ := 25 * 10
def sectionI : ℕ := 17 * 11
def sectionJ : ℕ := 20 * 9
def sectionK : ℕ := 15 * 13

def speedGH : ℕ := 8
def speedIJ : ℕ := 10
def speedK : ℕ := 6

/-- Compute the time spent in each section, rounding up where necessary -/
def timeG : ℕ := (sectionG + speedGH - 1) / speedGH
def timeH : ℕ := (sectionH + speedGH - 1) / speedGH
def timeI : ℕ := (sectionI + speedIJ - 1) / speedIJ
def timeJ : ℕ := (sectionJ + speedIJ - 1) / speedIJ
def timeK : ℕ := (sectionK + speedK - 1) / speedK

/-- Compute the total time spent -/
def totalTime : ℕ := timeG + timeH + timeI + timeJ + timeK

/-- The proof statement -/
theorem nate_total_time : totalTime = 129 := by
  -- the proof goes here
  sorry

end nate_total_time_l611_611893


namespace power_quotient_example_l611_611648

theorem power_quotient_example (a : ℕ) (m n : ℕ) (h : 23^11 / 23^8 = 23^(11 - 8)) : 23^3 = 12167 := by
  sorry

end power_quotient_example_l611_611648


namespace minimize_distances_l611_611701

-- Definitions for points on a line
def isOnLine (P : ℝ → Prop) (points : list ℝ) : Prop :=
  ∀ p ∈ points, P p

-- Definition to represent sum of specific distances being minimized
def sumOfDistances (P : ℝ) (distances : list ℝ) : ℝ :=
  distances.sum (λ x, abs (P - x))

theorem minimize_distances (P : ℝ) (P1 P2 P3 P4 P5 P6 P7 P8 P9 : ℝ) 
  (h_order : P1 < P2 ∧ P2 < P3 ∧ P3 < P4 ∧ P4 < P5 ∧ P5 < P6 ∧ P6 < P7 ∧ P7 < P8 ∧ P8 < P9) :
  P = P5 :=
sorry

end minimize_distances_l611_611701


namespace perpendicular_vectors_l611_611380

theorem perpendicular_vectors (x : ℝ) : 
  let a := (x, -3)
  let b := (2, -2)
  (a.1 * b.1 + a.2 * b.2 = 0) → x = -3 :=
by
  intro h
  change x * 2 + (-3) * (-2) with (a.1 * b.1 + a.2 * b.2) at h
  rw [mul_comm (-3) (-2), mul_neg_eq_neg_mul_symm] at h
  rw [mul_comm] at h
  sorry

end perpendicular_vectors_l611_611380


namespace proof_shortest_side_l611_611900

-- Definitions based on problem conditions
def side_divided (a b : ℕ) : Prop := a + b = 20

def radius (r : ℕ) : Prop := r = 5

noncomputable def shortest_side (a b c : ℕ) : ℕ :=
  if a ≤ b ∧ a ≤ c then a
  else if b ≤ a ∧ b ≤ c then b
  else c

-- Proof problem statement
theorem proof_shortest_side {a b c : ℕ} (h1 : side_divided 9 11) (h2 : radius 5) :
  shortest_side 15 (11 + 9) (2 * 6 + 9) = 14 :=
sorry

end proof_shortest_side_l611_611900


namespace tiling_remainder_l611_611259

-- Define the conditions
def valid_tiling (board_length : ℕ) (tile_colors : ℕ → ℕ) (colors : set ℕ) : Prop :=
  ∃ (partitions : list (list ℕ)),
    sum partitions.length = board_length ∧
    (∀ seg ∈ partitions, (2 ≤ seg.length)) ∧
    (∀ seg, tile_colors seg ∈ colors) ∧
    (∀ color ∈ colors, 2 ≤ sum (tile_colors '' {seg | seg ∈ partitions ∧ tile_colors seg = color}))

-- Define the problem
theorem tiling_remainder :
  let board_length := 8
  let colors := {0, 1, 2} -- Assuming 0=red, 1=blue, 2=green
  let N : ℕ := {partitions | valid_tiling board_length (partition_colors partitions) colors}.count
  in N % 1000 = 90 := by sorry

end tiling_remainder_l611_611259


namespace probability_at_least_one_solves_l611_611006

theorem probability_at_least_one_solves :
  ∀ (P : Type → Prop) [ProbabilityTheory P],
  let A := 0.6
  let B := 0.7 in
  independent_events A B →
  1 - ((1 - A) * (1 - B)) = 0.88 :=
by
  intros P _ A B independence
  sorry

end probability_at_least_one_solves_l611_611006


namespace ann_age_l611_611634

variable (A T : ℕ)

-- Condition 1: Tom is currently two times older than Ann
def tom_older : Prop := T = 2 * A

-- Condition 2: The sum of their ages 10 years later will be 38
def age_sum_later : Prop := (A + 10) + (T + 10) = 38

-- Theorem: Ann's current age
theorem ann_age (h1 : tom_older A T) (h2 : age_sum_later A T) : A = 6 :=
by
  sorry

end ann_age_l611_611634


namespace quadratic_max_value_4_at_2_l611_611938

theorem quadratic_max_value_4_at_2 (a b c : ℝ) (m : ℝ)
  (h1 : ∀ x : ℝ, x ≠ 2 → (a * 2^2 + b * 2 + c) = 4)
  (h2 : a * 0^2 + b * 0 + c = -20)
  (h3 : a * 5^2 + b * 5 + c = m) :
  m = -50 :=
sorry

end quadratic_max_value_4_at_2_l611_611938


namespace total_balloons_sam_and_dan_l611_611106

noncomputable def sam_initial_balloons : ℝ := 46.0
noncomputable def balloons_given_to_fred : ℝ := 10.0
noncomputable def dan_balloons : ℝ := 16.0

theorem total_balloons_sam_and_dan :
  (sam_initial_balloons - balloons_given_to_fred) + dan_balloons = 52.0 := 
by 
  sorry

end total_balloons_sam_and_dan_l611_611106


namespace peanut_butter_last_days_l611_611901

-- Definitions for the problem conditions
def daily_consumption : ℕ := 2
def servings_per_jar : ℕ := 15
def num_jars : ℕ := 4

-- The statement to prove
theorem peanut_butter_last_days : 
  (num_jars * servings_per_jar) / daily_consumption = 30 :=
by
  sorry

end peanut_butter_last_days_l611_611901


namespace max_sum_of_products_l611_611841

/--
Given 1999 distinct numbers arranged in a circle, proving that a specific arrangement yields 
the maximum sum of products of any 10 consecutive numbers.
-/
theorem max_sum_of_products (a : Fin 1999 → ℝ) (h : ∀ i j : Fin 1999, i ≠ j → a i ≠ a j) :
  ∃ arrangement, 
  (∀ n, sum_of_products_of_10_consecutive (rearrange a arrangement) n 
    = maximum_sum_of_products_of_any_10_consecutive a) := 
sorry

end max_sum_of_products_l611_611841


namespace min_sum_of_areas_max_product_of_areas_l611_611280

theorem min_sum_of_areas (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (h : x + y = 10) : 
  (x^2 / 16 + y^2 / 24) ≥ 5 / 2 :=
begin
  sorry
end

theorem max_product_of_areas (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (h : x + y = 10) : 
  (x^2 / 16 * y^2 / 24) ≤ 625 / 384 :=
begin
  sorry
end

end min_sum_of_areas_max_product_of_areas_l611_611280


namespace quadratic_completing_square_l611_611534

theorem quadratic_completing_square:
  ∃ (b c : ℝ), (∀ x : ℝ, x^2 + 900 * x + 1800 = (x + b)^2 + c) ∧ (c / b = -446.22222) :=
by
  -- We'll skip the proof steps here
  sorry

end quadratic_completing_square_l611_611534


namespace find_theta_l611_611718

variable (x : ℝ) (θ : ℝ) (k : ℤ)

def condition := (3 - 3^(-|x - 3|))^2 = 3 - Real.cos θ

theorem find_theta (h : condition x θ) : ∃ k : ℤ, θ = (2 * k + 1) * Real.pi :=
by
  sorry

end find_theta_l611_611718


namespace convert_23_to_binary_l611_611653

theorem convert_23_to_binary :
  Nat.toBinary 23 = "10111" :=
sorry

end convert_23_to_binary_l611_611653


namespace probability_symmetric_interval_l611_611617

noncomputable def X : Type := sorry -- Define the type of the random variable X

variable [MeasureTheory.ProbabilityMeasure X]
variable (a : ℝ) (h_mean : a = 10)
variable (h_interval1 : ∀ (X : ℝ), (10 < X ∧ X < 20) → MeasureTheory.ProbabilityMeasure X = 0.3)

theorem probability_symmetric_interval :
  MeasureTheory.Probability (Set.Ioo 0 10) = 0.3 :=
by
  sorry -- The actual proof would go here, but is omitted for this example.

end probability_symmetric_interval_l611_611617


namespace f_9_plus_f_0_eq_3_l611_611719

def f (x : ℝ) : ℝ :=
  if x > 0 then real.log x / real.log 3 else 2 ^ x

theorem f_9_plus_f_0_eq_3 : f 9 + f 0 = 3 := by
  have h1 : f 9 = real.log 9 / real.log 3 := by simp [f, gt]
  have h2 : f 0 = 1 := by simp [f, le]
  rw [h1, h2]
  sorry

end f_9_plus_f_0_eq_3_l611_611719


namespace average_percentage_l611_611250

theorem average_percentage (x : ℝ) : (60 + x + 80) / 3 = 70 → x = 70 :=
by
  intro h
  sorry

end average_percentage_l611_611250


namespace smallest_n_for_5_pairwise_rel_prime_l611_611072

-- Define the set S
def S : Set Nat := {i | i ≤ 280}

-- Definition to check relative primality (coprimeness)
def pairwise_rel_prime (lst : List Nat) : Prop :=
  ∀ (x y : Nat), x ∈ lst → y ∈ lst → x ≠ y → Nat.gcd x y = 1

-- Definition to check every subset of a given size n contains 5 pairwise relatively prime elements
def every_subset_contains_5_pairwise_rel_prime (n : Nat) : Prop :=
  ∀ (subset : Finset Nat), subset ⊆ S → subset.card = n → ∃ (lst : List Nat), lst ⊆ subset.val ∧ lst.length = 5 ∧ pairwise_rel_prime lst

-- The theorem to be proven
theorem smallest_n_for_5_pairwise_rel_prime : (∀ n, every_subset_contains_5_pairwise_rel_prime n) → 217 :=
by
  sorry

end smallest_n_for_5_pairwise_rel_prime_l611_611072


namespace diagonals_in_dodecagon_l611_611603

theorem diagonals_in_dodecagon : 
  let n := 12 in
  n = 12 → (n * (n - 3)) / 2 = 54 :=
by
  intros n h
  sorry

end diagonals_in_dodecagon_l611_611603


namespace point_Q_locus_l611_611358

-- Definitions for the given problem
def C1 (x y : ℝ) : Prop := (x ^ 2) / 3 + (y ^ 2) / 4 = 1

def C2 (x y : ℝ) : Prop := x + y = 1

def P_on_C2 (P : ℝ × ℝ) : Prop := C2 P.1 P.2

def OP (O P : ℝ × ℝ) (t : ℝ) : ℝ × ℝ := (t * P.1, t * P.2)

def OP_intersects_C1 (O P : ℝ × ℝ) (R : ℝ × ℝ) (t : ℝ) : Prop := 
  OP O P t = R ∧ C1 R.1 R.2

def OR (O R : ℝ × ℝ) : ℝ := real.sqrt ((R.1 - O.1) ^ 2 + (R.2 - O.2) ^ 2)

def OQ_condition (O Q P R : ℝ × ℝ) : Prop := 
  ((real.sqrt (Q.1^2 + Q.2^2)) * (real.sqrt (P.1^2 + P.2^2))) = (OR O R) ^ 2

-- The theorem to be proved
theorem point_Q_locus : 
  (∀ P : ℝ × ℝ, P_on_C2 P → ∃ Q : ℝ × ℝ, (∃ R : ℝ × ℝ, ∃ t : ℝ, OP_intersects_C1 (0, 0) P R t ∧ OQ_condition (0, 0) Q P R) → 
  (C1 Q.1 Q.2 → Q.1 - 3/2) ^ 2 / (21 / 4) + (Q.2 - 2) ^ 2 / 7 = 1) := 
sorry

end point_Q_locus_l611_611358


namespace natural_numbers_sum_of_divisors_l611_611296

def greatest_divisor_excluding_self (n : ℕ) : ℕ :=
  if n < 2 then 0 else n.divisors.finset.filter (λ d, d < n).max

theorem natural_numbers_sum_of_divisors (n : ℕ) :
  (n = 58 ∨ n = 66) ↔
  (n = greatest_divisor_excluding_self (n - 1) +
  greatest_divisor_excluding_self (n - 2) +
  greatest_divisor_excluding_self (n - 3)) := by
  sorry

end natural_numbers_sum_of_divisors_l611_611296


namespace integer_solutions_abs_inequality_l611_611738

theorem integer_solutions_abs_inequality :
  ∃ (n : ℕ), n = 11 ∧ ∀ (x : ℤ), |x - 2| ≤ 5.6 ↔ x ∈ (...the set of integers satisfying the inequality...) :=
sorry

end integer_solutions_abs_inequality_l611_611738


namespace domain_intersection_l611_611232

theorem domain_intersection (A B : Set ℝ) 
    (h1 : A = {x | x < 1})
    (h2 : B = {y | y ≥ 0}) : A ∩ B = {z | 0 ≤ z ∧ z < 1} := 
by
  sorry

end domain_intersection_l611_611232


namespace measure_of_angle_d_l611_611475

-- Define all the angles and the relations given in the problem
variables {p q r s : ℝ}
variables {a b c d : ℝ}

-- Conditions
def p_parallel_q : Prop := true -- p ∥ q
def r_intersects_pq : Prop := true -- r intersects both p and q
def a_b_on_same_side_r : Prop := true -- a on p, b on q
def angle_a_is_one_sixth_of_b : a = (1/6) * b := sorry
def s_intersects_c_d : Prop := true -- s intersects q forming c and d
def c_is_one_third_of_b : c = (1/3) * b := sorry
def d_c_straight_line : d + c = 180 := sorry

-- Proposition to prove
theorem measure_of_angle_d (a b c d : ℝ) 
  (a_is_one_sixth_b : a = (1/6) * b)
  (c_is_one_third_b : c = (1/3) * b)
  (d_plus_c_straight : d + c = 180) :
  d = 60 :=
by 
  let x := a
  have bx : 6 * x = b := sorry
  have dx : d = x := sorry
  have cx : c = 2 * x := sorry
  have sum_c_d : 2 * x + x = 180 := sorry
  have three_x : 3 * x = 180 := sum_c_d
  sorry

end measure_of_angle_d_l611_611475


namespace angle_in_second_quadrant_l611_611948

-- Define the conditions and the problem
def angle_quadrant (θ : ℤ) : ℕ :=
  (θ % 360 + 360) % 360

theorem angle_in_second_quadrant (θ : ℤ) :
  θ = -1320 → angle_quadrant θ = 120 ∧ 120 > 90 ∧ 120 < 180 :=
by
  intros h
  rw [h]
  sorry

end angle_in_second_quadrant_l611_611948


namespace part_a_part_b_l611_611660

-- Define variables and conditions
variables (classrooms : ℕ) (students : ℕ)

-- Part (a): Prove that at least one classroom has at least 3 students
theorem part_a (h_classrooms : classrooms = 9) (h_students : students = 19) : 
  ∃ c : ℕ, (c ≤ classrooms) ∧ (students / classrooms + if students % classrooms > 0 then 1 else 0 >= 3) :=
by 
  -- Proof goes here
  sorry

-- Part (b): Prove that it's not necessarily true that there exists a classroom with exactly 3 students
theorem part_b (h_classrooms : classrooms = 9) (h_students : students = 19) : 
  ¬ ∀ c : ℕ, (c ≤ classrooms) → (students / classrooms = 3) :=
by
  -- Proof goes here
  sorry

end part_a_part_b_l611_611660


namespace expected_unpaired_socks_l611_611493

noncomputable def E (n : ℕ) : ℚ :=
if n = 1 then 2 else (2 * n / (2 * n - 1)) * E (n - 1)

-- Statement: For 2024 pairs of socks, the expected number of unpaired socks drawn is
theorem expected_unpaired_socks : 
  E 2024 - 2 = (4^2024 : ℚ) / ((nat.fac 4048) / (nat.fac 2024 * nat.fac 2024)) - 2 :=
sorry

end expected_unpaired_socks_l611_611493


namespace log2_b6b8_l611_611844

variable {a : ℕ → ℝ} -- Arithmetic sequence
variable {b : ℕ → ℝ} -- Geometric sequence

-- First, we define the arithmetic sequence conditions
def is_arithmetic_seq (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) - a n = d

-- Next, we state the given equation in the problem
def arithmetic_condition (a : ℕ → ℝ) : Prop := 
  2 * a 3 - a 7 + 2 * a 11 = 0

-- Define the geometric sequence with the given relation b_7 = a_7
def is_geometric_seq (b a : ℕ → ℝ) (r : ℝ) : Prop :=
  b 7 = a 7 ∧ ∀ n : ℕ, b (n + 1) = b n * r

-- Finally, we state the target to prove
theorem log2_b6b8 (a b : ℕ → ℝ) (d r : ℝ) 
  (ar_seq : is_arithmetic_seq a d)
  (geo_seq : is_geometric_seq b a r)
  (cond : arithmetic_condition a) :
  log 2 (b 6 * b 8) = 4 := 
sorry

end log2_b6b8_l611_611844


namespace minimum_oranges_l611_611119

theorem minimum_oranges (n : ℕ) (m : ℕ → ℝ) (h : ∀ i j k : ℕ, i < n → j < n → k < n → i ≠ j → i ≠ k → j ≠ k → (m i + m j + m k) < 0.05 * ∑ l in Finset.range n \ {i, j, k}, m l) : n ≥ 64 := 
sorry

end minimum_oranges_l611_611119


namespace keith_bought_cards_l611_611315

theorem keith_bought_cards (orig : ℕ) (now : ℕ) (bought : ℕ) 
  (h1 : orig = 40) (h2 : now = 18) (h3 : bought = orig - now) : bought = 22 := by
  sorry

end keith_bought_cards_l611_611315


namespace area_of_triangle_l611_611420

variable {α : Type*} [LinearOrder α] [LinearOrderedField α] [RealLinearOrderedField α]

variable {a b c : α} (angle_A : α)

-- Conditions
def h1 : b^2 + c^2 = a^2 + b * c := sorry
def h2 : b * c * cos angle_A = 4 := sorry

-- Theorem
theorem area_of_triangle (h1 : b^2 + c^2 = a^2 + b * c) (h2 : b * c * cos angle_A = 4) (angle_A : α) :
  (1 / 2) * b * c * sin angle_A = 2 * sqrt 3 := sorry

end area_of_triangle_l611_611420


namespace equal_sides_of_equiangular_convex_ngon_l611_611905

theorem equal_sides_of_equiangular_convex_ngon (n : ℕ) (a : Fin n → ℝ) 
  (h_angle_eq : ∀ (i j : Fin n), angle i j = 2 * π / n) 
  (h_non_inc : ∀ (i : Fin (n-1)), a i ≥ a (i+1)) : 
  (∀ (i j : Fin n), a i = a j) :=
by
  sorry

end equal_sides_of_equiangular_convex_ngon_l611_611905


namespace fishing_tomorrow_l611_611819

theorem fishing_tomorrow (yesterday_fishers today_fishers : ℕ)
  (every_day_fishers every_other_day_fishers every_three_days_fishers : ℕ)
  (total_population : ℕ):
  yesterday_fishers = 12 → 
  today_fishers = 10 → 
  every_day_fishers = 7 → 
  every_other_day_fishers = 8 → 
  every_three_days_fishers = 3 → 
  total_population = yesterday_fishers + today_fishers + (total_population - (every_day_fishers + every_other_day_fishers + every_three_days_fishers)) →
  ∃ tomorrow_fishers : ℕ, tomorrow_fishers = 15 :=
by {
  -- This is a statement definition, the proof is not required and thus marked as "sorry:"
  sorry
}

end fishing_tomorrow_l611_611819


namespace problem_inequality_l611_611877

theorem problem_inequality (a b c m n p : ℝ) (h1 : a + b + c = 1) (h2 : m + n + p = 1) :
  -1 ≤ a * m + b * n + c * p ∧ a * m + b * n + c * p ≤ 1 := by
  sorry

end problem_inequality_l611_611877


namespace tangent_length_l611_611350

noncomputable def circle : set (ℝ × ℝ) := {p | (p.1 - 2)^2 + (p.2 - 1)^2 = 4}
def line (a : ℝ) : set (ℝ × ℝ) := {p | p.1 + a * p.2 - 1 = 0}

theorem tangent_length (a : ℝ) 
  (A : ℝ × ℝ) 
  (hA : A = (-4, -1)) 
  (h1 : line a (2, 1))
  (h2 : a = -1) 
  (h3 : (2, 1) ∈ circle) : 
  ∃ B : ℝ × ℝ, (B ∈ circle ∧ dist A B = 6 ∧ segment_length A B = 6) := 
sorry

end tangent_length_l611_611350


namespace fishing_tomorrow_l611_611834

-- Conditions
def every_day_fishers : Nat := 7
def every_other_day_fishers : Nat := 8
def every_three_days_fishers : Nat := 3
def yesterday_fishers : Nat := 12
def today_fishers : Nat := 10

-- Determine the number who will fish tomorrow
def fishers_tomorrow : Nat :=
  let every_day_tomorrow := every_day_fishers
  let every_three_day_tomorrow := every_three_days_fishers
  let every_other_day_yesterday := yesterday_fishers - every_day_fishers
  let every_other_day_tomorrow := every_other_day_fishers - every_other_day_yesterday
  every_day_tomorrow + every_three_day_tomorrow + every_other_day_tomorrow

theorem fishing_tomorrow : fishers_tomorrow = 15 :=
  by
    -- skipping the actual proof with sorry
    sorry

end fishing_tomorrow_l611_611834


namespace find_starting_number_l611_611541

-- Definitions based on the conditions
def numbers_without_digit_1 (start end : ℕ) : ℕ := 
  -- Placeholder for the actual function that calculates how many numbers in a range don't contain the digit 1
  sorry

theorem find_starting_number :
  ∃ start : ℕ, numbers_without_digit_1 start 1000 = 728 → start = 271 :=
begin
  -- This is a placeholder statement, the actual proof is omitted
  sorry
end

end find_starting_number_l611_611541


namespace find_k_l611_611001

noncomputable def f (k : ℤ) (x : ℝ) := (k^2 + k - 1) * x^(k^2 - 3 * k)

-- The conditions in the problem
variables (k : ℤ) (x : ℝ)
axiom sym_y_axis : ∀ (x : ℝ), f k (-x) = f k x
axiom decreasing_on_positive : ∀ x1 x2, 0 < x1 → x1 < x2 → f k x1 > f k x2

-- The proof problem statement
theorem find_k : k = 1 :=
sorry

end find_k_l611_611001


namespace find_interest_rate_per_annum_l611_611300

noncomputable def interest_rate_per_annum (P : ℝ) (t : ℝ) (d : ℝ) : ℝ :=
r where
  SI := P * r * t / 100
  CI := P * (1 + r / 100) ^ t - P
  h : CI - SI = d

theorem find_interest_rate_per_annum :
  interest_rate_per_annum 2000 2 3.20 = 4 := 
sorry

end find_interest_rate_per_annum_l611_611300


namespace abigail_monthly_saving_l611_611629

-- Definitions based on the conditions
def total_saving := 48000
def months_in_year := 12

-- The statement to be proved
theorem abigail_monthly_saving : total_saving / months_in_year = 4000 :=
by sorry

end abigail_monthly_saving_l611_611629


namespace one_step_polynomial_transformation_l611_611102

theorem one_step_polynomial_transformation (a b c x y z : ℤ) :
    (∃ seq : ℕ → ℤ → ℤ, seq 0 = id ∧ (∀ n, seq (n + 1) = (λ t, t * (seq n t))) ∧
    (seq (n+1) a = x ∧ seq (n+1) b = y ∧ seq (n+1) c = z)) →
    (∃ P : ℤ → ℤ, P a = x ∧ P b = y ∧ P c = z) :=
begin
  sorry
end

end one_step_polynomial_transformation_l611_611102


namespace find_radius_of_circumcircle_l611_611041

noncomputable def radius_of_circumcircle (triangle_ABC : Type)
  [triangle triangle_ABC]
  (A B C : triangle_ABC)
  (circumcenter O : triangle_ABC)
  (orthocenter H : triangle_ABC)
  (AD : segment A A)
  (h_angle_bisector : is_angle_bisector AD)
  (h_bisects_AO : bisects AD (segmentSegment O H))
  (h_AC_eq_2 : AC = 2)
  (h_AD_eq_sqrt3_sqrt2_minus1 : AD = sqrt 3 + sqrt 2 - 1)
  : ℝ :=
  ∃ (R : ℝ), R = (sqrt 6 - sqrt 2 + 2) / sqrt (2 + sqrt 2)

theorem find_radius_of_circumcircle
  (triangle_ABC : Type)
  [triangle triangle_ABC]
  (A B C : triangle_ABC)
  (circumcenter O : triangle_ABC)
  (orthocenter H : triangle_ABC)
  (AD : segment A A)
  (h_angle_bisector : is_angle_bisector AD)
  (h_bisects_AO : bisects AD (segmentSegment O H))
  (h_AC_eq_2 : AC = 2)
  (h_AD_eq_sqrt3_sqrt2_minus1 : AD = sqrt 3 + sqrt 2 - 1) :
  radius_of_circumcircle A B C O H AD h_angle_bisector h_bisects_AO h_AC_eq_2 h_AD_eq_sqrt3_sqrt2_minus1
  =
  (sqrt 6 - sqrt 2 + 2) / sqrt (2 + sqrt 2)
:= sorry

end find_radius_of_circumcircle_l611_611041


namespace sophia_book_problem_l611_611150

/-
Prove that the total length of the book P is 270 pages, and verify the number of pages read by Sophia
on the 4th and 5th days (50 and 40 pages respectively), given the following conditions:
1. Sophia finished 2/3 of the book in the first three days.
2. She calculated that she finished 90 more pages than she has yet to read.
3. She plans to finish the entire book within 5 days.
4. She will read 10 fewer pages each day from the 4th day until she finishes.
-/

theorem sophia_book_problem
  (P : ℕ)
  (h1 : (2/3 : ℝ) * P = P - (90 + (1/3 : ℝ) * P))
  (h2 : P = 3 * 90)
  (remaining_pages : ℕ := P / 3)
  (h3 : remaining_pages = 90)
  (pages_day4 : ℕ)
  (pages_day5 : ℕ := pages_day4 - 10)
  (h4 : pages_day4 + pages_day4 - 10 = 90)
  (h5 : 2 * pages_day4 - 10 = 90)
  (h6 : 2 * pages_day4 = 100)
  (h7 : pages_day4 = 50) :
  P = 270 ∧ pages_day4 = 50 ∧ pages_day5 = 40 := 
by {
  sorry -- Proof is skipped
}

end sophia_book_problem_l611_611150


namespace remainder_3a_plus_b_l611_611087

theorem remainder_3a_plus_b (p q : ℤ) (a b : ℤ)
  (h1 : a = 98 * p + 92)
  (h2 : b = 147 * q + 135) :
  ((3 * a + b) % 49) = 19 := by
sorry

end remainder_3a_plus_b_l611_611087


namespace range_of_x_l611_611158

-- Definitions based on the conditions provided
variable {ℝ : Type*} [linear_ordered_field ℝ]

-- Define a function f
variable (f : ℝ → ℝ)

-- Define the conditions
def monotonic_decreasing_on (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ x y ∈ I, x ≤ y → f y ≤ f x

def symmetric_about_y_axis (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- Given conditions
variables (h1 : monotonic_decreasing_on f (set.Ici 0))
variables (h2 : symmetric_about_y_axis f)
variable (h3 : f (-2) = 1)

-- Main theorem to prove
theorem range_of_x (f : ℝ → ℝ) (x : ℝ) :
  monotonic_decreasing_on f (set.Ici 0) →
  symmetric_about_y_axis f →
  (f (-2) = 1) →
  (f (x - 2) ≥ 1 ↔ 0 ≤ x ∧ x ≤ 4) :=
by
  intros
  sorry

end range_of_x_l611_611158


namespace inequality_x_y_l611_611485

theorem inequality_x_y 
  (x y : ℝ) 
  (hx : 0 < x) 
  (hy : 0 < y) : 
  (x / (x + 5 * y)) + (y / (y + 5 * x)) ≤ 1 := 
by 
  sorry

end inequality_x_y_l611_611485


namespace correct_calculation_l611_611211

theorem correct_calculation : 
  ¬((sqrt 2)^0 = sqrt 2) ∧ 
  ¬(2 * sqrt 3 + 3 * sqrt 3 = 5 * sqrt 6) ∧ 
  ¬(sqrt 8 = 4 * sqrt 2) ∧ 
  (sqrt 3 * (2 * sqrt 3 - 2) = 6 - 2 * sqrt 3) := by
  sorry

end correct_calculation_l611_611211


namespace parametric_line_segment_computation_l611_611525

theorem parametric_line_segment_computation :
  ∃ (a b c d : ℝ), 
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 
   (-3, 10) = (a * t + b, c * t + d) ∧
   (4, 16) = (a * 1 + b, c * 1 + d)) ∧
  (b = -3) ∧ (d = 10) ∧ 
  (a + b = 4) ∧ (c + d = 16) ∧ 
  (a^2 + b^2 + c^2 + d^2 = 194) :=
sorry

end parametric_line_segment_computation_l611_611525


namespace part1_part2_l611_611724

noncomputable def f (x t : ℝ) := (Real.log x / Real.log 2) + t

def g (x a : ℝ) : ℝ :=
if h : x ≤ 1 then
  f x 2
else
  x^2 - 2 * a * x

theorem part1 (t : ℝ) (ht : 0 < t) (hft : f t t = 3) : f t 2 = f x 2 :=
by
  sorry

theorem part2 (a : ℝ) : (-1/2 ≤ a) ∧ (1 < a) ∨ (-1/2 ≤ a) ∧ (a ≤ 1) :=
by
  sorry

end part1_part2_l611_611724


namespace min_pencils_for_each_color_max_pencils_remaining_each_color_max_red_pencils_to_ensure_five_remaining_l611_611384

-- Condition Definitions
def blue := 5
def red := 9
def green := 6
def yellow := 4

-- Theorem Statements
theorem min_pencils_for_each_color :
  ∀ B R G Y : ℕ, blue = 5 ∧ red = 9 ∧ green = 6 ∧ yellow = 4 →
  ∃ min_pencils : ℕ, min_pencils = 21 := by
  sorry

theorem max_pencils_remaining_each_color :
  ∀ B R G Y : ℕ, blue = 5 ∧ red = 9 ∧ green = 6 ∧ yellow = 4 →
  ∃ max_pencils : ℕ, max_pencils = 3 := by
  sorry

theorem max_red_pencils_to_ensure_five_remaining :
  ∀ B R G Y : ℕ, blue = 5 ∧ red = 9 ∧ green = 6 ∧ yellow = 4 →
  ∃ max_red_pencils : ℕ, max_red_pencils = 4 := by
  sorry

end min_pencils_for_each_color_max_pencils_remaining_each_color_max_red_pencils_to_ensure_five_remaining_l611_611384


namespace num_students_is_92_l611_611544

noncomputable def total_students (S : ℕ) : Prop :=
  let remaining := S - 20
  let biking := (5/8 : ℚ) * remaining
  let walking := (3/8 : ℚ) * remaining
  walking = 27

theorem num_students_is_92 : total_students 92 :=
by
  let remaining := 92 - 20
  let biking := (5/8 : ℚ) * remaining
  let walking := (3/8 : ℚ) * remaining
  have walk_eq : walking = 27 := by sorry
  exact walk_eq

end num_students_is_92_l611_611544


namespace fishing_tomorrow_l611_611825

theorem fishing_tomorrow (yesterday_fishers today_fishers : ℕ)
  (every_day_fishers every_other_day_fishers every_three_days_fishers : ℕ)
  (total_population : ℕ):
  yesterday_fishers = 12 → 
  today_fishers = 10 → 
  every_day_fishers = 7 → 
  every_other_day_fishers = 8 → 
  every_three_days_fishers = 3 → 
  total_population = yesterday_fishers + today_fishers + (total_population - (every_day_fishers + every_other_day_fishers + every_three_days_fishers)) →
  ∃ tomorrow_fishers : ℕ, tomorrow_fishers = 15 :=
by {
  -- This is a statement definition, the proof is not required and thus marked as "sorry:"
  sorry
}

end fishing_tomorrow_l611_611825


namespace abs_ineq_solution_set_l611_611671

theorem abs_ineq_solution_set (x : ℝ) : |x + 1| - |x - 5| < 4 ↔ x < 4 :=
sorry

end abs_ineq_solution_set_l611_611671


namespace find_p_l611_611761

-- Definitions
noncomputable def p : ℝ := sorry
def ξ : Type := sorry
def B : Type := sorry
def n : ℝ := sorry

-- Conditions
axiom xi_binomial : ξ = B(n, p)
axiom expected_value_condition : (n * p) = 10
axiom variance_condition : (n * p * (1 - p)) = 8

-- Theorem statement
theorem find_p : p = 1 / 5 := sorry

end find_p_l611_611761


namespace linear_function_value_l611_611873

theorem linear_function_value (g : ℝ → ℝ) (h_linear : ∀ x y, g (x + y) = g x + g y)
  (h_scale : ∀ c x, g (c * x) = c * g x) (h : g 10 - g 0 = 20) : g 20 - g 0 = 40 :=
by
  sorry

end linear_function_value_l611_611873


namespace calories_in_250g_of_lemonade_l611_611646

theorem calories_in_250g_of_lemonade (
  lemon_juice_weight : ℕ,
  sugar_weight : ℕ,
  water_weight : ℕ,
  lemon_calories_per_100g : ℕ,
  sugar_calories_per_100g : ℕ,

  lemon_juice_weight = 150,
  sugar_weight = 150,
  water_weight = 400,
  lemon_calories_per_100g = 25,
  sugar_calories_per_100g = 386
) :
  ∃ (x : ℕ), x = 220 :=
by sorry

end calories_in_250g_of_lemonade_l611_611646


namespace thomas_annual_insurance_cost_l611_611963

theorem thomas_annual_insurance_cost (total_cost : ℕ) (number_of_years : ℕ) 
  (h1 : total_cost = 40000) (h2 : number_of_years = 10) : 
  total_cost / number_of_years = 4000 := 
by 
  sorry

end thomas_annual_insurance_cost_l611_611963


namespace fishing_tomorrow_l611_611795

theorem fishing_tomorrow (every_day_fishers every_other_day_fishers every_three_days_fishers fishers_yesterday fishers_today : ℕ) :
  (every_day_fishers = 7) →
  (every_other_day_fishers = 8) →
  (every_three_days_fishers = 3) →
  (fishers_yesterday = 12) →
  (fishers_today = 10) →
  (every_three_days_fishers + every_day_fishers + (every_other_day_fishers - (fishers_yesterday - every_day_fishers)) = 15) :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  simp
  sorry

end fishing_tomorrow_l611_611795


namespace fishing_tomorrow_l611_611830

-- Conditions
def every_day_fishers : Nat := 7
def every_other_day_fishers : Nat := 8
def every_three_days_fishers : Nat := 3
def yesterday_fishers : Nat := 12
def today_fishers : Nat := 10

-- Determine the number who will fish tomorrow
def fishers_tomorrow : Nat :=
  let every_day_tomorrow := every_day_fishers
  let every_three_day_tomorrow := every_three_days_fishers
  let every_other_day_yesterday := yesterday_fishers - every_day_fishers
  let every_other_day_tomorrow := every_other_day_fishers - every_other_day_yesterday
  every_day_tomorrow + every_three_day_tomorrow + every_other_day_tomorrow

theorem fishing_tomorrow : fishers_tomorrow = 15 :=
  by
    -- skipping the actual proof with sorry
    sorry

end fishing_tomorrow_l611_611830


namespace smallest_n_sqrt_difference_l611_611975

theorem smallest_n_sqrt_difference : ∃ (n : ℕ), (0 < n) ∧ (sqrt n - sqrt (n - 1 : ℝ) < 0.05) ∧ (n = 101) :=
by {
  sorry
}

end smallest_n_sqrt_difference_l611_611975


namespace solve_equation_l611_611502

theorem solve_equation :
  ∀ x : ℝ, (1 + 2 * x ^ (1/2) - x ^ (1/3) - 2 * x ^ (1/6) = 0) ↔ (x = 1 ∨ x = 1 / 64) :=
by
  sorry

end solve_equation_l611_611502


namespace probability_third_smallest_seven_l611_611317

theorem probability_third_smallest_seven :
  let s := Finset.Icc 1 20 in
  let total_ways := s.card.choose 8 in
  let favorable_ways := (Finset.Icc 1 6).card.choose 2 * (Finset.Icc 8 20).card.choose 5 in
  (favorable_ways : ℚ) / (total_ways : ℚ) = 645 / 4199 :=
by
  let s := Finset.Icc 1 20
  let total_ways := s.card.choose 8
  let favorable_ways := (Finset.Icc 1 6).card.choose 2 * (Finset.Icc 8 20).card.choose 5
  have h : (s.card = 20) := rfl
  have h1 : ((Finset.Icc 1 6).card = 6) := rfl
  have h2 : ((Finset.Icc 8 20).card = 13) := rfl
  unfold Finset.card
  rw [h, h1, h2]
  have h_total : total_ways = 125970 := rfl
  have h_favorable : favorable_ways = 19305 := rfl
  field_simp [h_total, h_favorable]
  norm_num
  sorry

end probability_third_smallest_seven_l611_611317


namespace fishing_tomorrow_l611_611799

theorem fishing_tomorrow (every_day_fishers every_other_day_fishers every_three_days_fishers fishers_yesterday fishers_today : ℕ) :
  (every_day_fishers = 7) →
  (every_other_day_fishers = 8) →
  (every_three_days_fishers = 3) →
  (fishers_yesterday = 12) →
  (fishers_today = 10) →
  (every_three_days_fishers + every_day_fishers + (every_other_day_fishers - (fishers_yesterday - every_day_fishers)) = 15) :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  simp
  sorry

end fishing_tomorrow_l611_611799


namespace distance_from_focus_to_asymptote_correct_l611_611330

noncomputable def distance_from_focus_to_asymptote (a b : ℝ) (F P : ℝ × ℝ) (p : ℝ) : ℝ :=
  let (Fx, Fy) := F in
  let (Px, Py) := P in
  abs ((√3) * Fx - Fy) / sqrt ((√3) ^ 2 + 1 ^ 2)

theorem distance_from_focus_to_asymptote_correct :
  let a := 1
  let b := sqrt 3
  let F := (2, 0)
  let P := (3, 2 * sqrt 6)
  let p := 5 in
  distance_from_focus_to_asymptote a b F P p = sqrt 3 := by
  sorry

end distance_from_focus_to_asymptote_correct_l611_611330


namespace calculation_result_l611_611203

theorem calculation_result :
  15 * (1 / 3 + 1 / 4 + 1 / 6)⁻¹ = 20 := by
  sorry

end calculation_result_l611_611203


namespace nilpotent_matrix_squared_zero_l611_611067

variable {R : Type*} [Field R]
variable (A : Matrix (Fin 2) (Fin 2) R)

theorem nilpotent_matrix_squared_zero (h : A^4 = 0) : A^2 = 0 := 
sorry

end nilpotent_matrix_squared_zero_l611_611067


namespace harold_tips_fraction_l611_611382

theorem harold_tips_fraction (A : ℝ) :
  let tip_other_months := 6 * A,
      tip_august := 6 * A,
      total_tips := tip_other_months + tip_august
  in tip_august / total_tips = (1 / 2) :=
by
  sorry

end harold_tips_fraction_l611_611382


namespace find_x_l611_611412

theorem find_x (k : ℚ) :
  (∀ x y : ℚ, 5 * x - 3 = k * (2 * y + 10)) ∧ (3 * 5 - 3 = k * (2 * 2 + 10)) →
  ∃ x : ℚ, y = 5 → x = 47 / 5 :=
by 
  intro h1 h2
  use 47 / 5
  sorry

end find_x_l611_611412


namespace dice_sum_to_11_l611_611202

/-- Define the conditions for the outcomes of the dice rolls -/
def valid_outcomes (x : Fin 5 → ℕ) : Prop :=
  (∀ i, 1 ≤ x i ∧ x i ≤ 6) ∧ (x 0 + x 1 + x 2 + x 3 + x 4 = 11)

/-- Prove that there are exactly 205 ways to achieve a sum of 11 with five different colored dice -/
theorem dice_sum_to_11 : 
  (∃ (s : Finset (Fin 5 → ℕ)), (∀ x ∈ s, valid_outcomes x) ∧ s.card = 205) :=
  by
    sorry

end dice_sum_to_11_l611_611202


namespace roots_of_quadratic_l611_611020

theorem roots_of_quadratic (a b : ℝ) (h : a ≠ 0) (h1 : a + b = 0) :
  ∀ x, (a * x^2 + b * x = 0) → (x = 0 ∨ x = 1) := 
by
  sorry

end roots_of_quadratic_l611_611020


namespace infinite_geometric_series_sum_l611_611642

theorem infinite_geometric_series_sum :
  let a := 2
  let r := (2 / 5)
  ∃ S : ℚ, S = 10 / 3 ∧ S = a / (1 - r) :=
by
  let a := 2
  let r := 2 / 5
  existsi (10 / 3)
  split
  · rfl
  · sorry

end infinite_geometric_series_sum_l611_611642


namespace inequality_solution_l611_611181

theorem inequality_solution (x : ℝ) : 
  (x-20) / (x+16) ≤ 0 ↔ -16 < x ∧ x ≤ 20 := by
  sorry

end inequality_solution_l611_611181


namespace problem_statement_l611_611043

noncomputable def length_MH (ABC : Triangle) (ABD : Triangle) (areaABC areaABD : ℝ)
                           (AB : ℝ) (M : Point) (H : Point) (CD : Line) : ℝ := sorry

theorem problem_statement (ABC ABD : Triangle) (areaABC : 100 = 100) (areaABD : 72 = 72)
                           (AB : 20 = 20) (M : Point) (H : Point)
                           (midpoint : M = (D + C) / 2) (right_angle : ⦃angle MHB = 90°⦄) :
  length_MH ABC ABD 100 72 20 M H sorry = 8.6 := sorry

end problem_statement_l611_611043


namespace jane_test_scores_l611_611861

theorem jane_test_scores
  (s1 s2 s3 : ℕ)
  (avg_scores : ℕ)
  (test_scores : list ℕ)
  (total_score : s1 + s2 + s3 + 252 = 510)
  (unique_scores : list.nodup test_scores)
  (score_limit : ∀ (x : ℕ), x ∈ test_scores → x ≤ 95)
  (all_scores_diff : test_scores = [s1, s2, s3, 95, 83, 74]) :
  list.reverse test_scores = [95, 94, 86, 83, 78, 74] :=
by 
  sorry

end jane_test_scores_l611_611861


namespace line_eq_x_1_parallel_y_axis_l611_611157

theorem line_eq_x_1_parallel_y_axis (P : ℝ × ℝ) (hP : P = (1, 0)) (h_parallel : ∀ y : ℝ, (1, y) = P ∨ P = (1, y)) :
  ∃ x : ℝ, (∀ y : ℝ, P = (x, y)) → x = 1 := 
by 
  sorry

end line_eq_x_1_parallel_y_axis_l611_611157


namespace B_interval_l611_611650

noncomputable def f : ℕ → ℝ
| 3       := real.log 3
| (n + 1) := real.log (n + 1 + f n)

theorem B_interval :
  let B := f 2024 in 
  real.log 2027 < B ∧ B < real.log 2028 :=
sorry

end B_interval_l611_611650


namespace simplify_exponent_multiplication_l611_611920

theorem simplify_exponent_multiplication (x : ℝ) : x^5 * x^3 = x^8 :=
by sorry

end simplify_exponent_multiplication_l611_611920


namespace calculate_expression_l611_611269

theorem calculate_expression (b : ℝ) (hb : b ≠ 0) : 
  (1 / 8 * b ^ 0 + (1 / (8 * b)) ^ 0) - (32 ^ (-1 / 5) + (-64) ^ (-1 / 2)) = 3 / 4 := 
by 
  sorry

end calculate_expression_l611_611269


namespace minimum_oranges_l611_611116

theorem minimum_oranges (n : ℕ) (m : ℕ → ℝ) (h : ∀ i j k : ℕ, i < n → j < n → k < n → i ≠ j → i ≠ k → j ≠ k → (m i + m j + m k) < 0.05 * ∑ l in Finset.range n \ {i, j, k}, m l) : n ≥ 64 := 
sorry

end minimum_oranges_l611_611116


namespace minimum_value_of_4a_plus_b_l611_611702

noncomputable def minimum_value (a b : ℝ) :=
  if a > 0 ∧ b > 0 ∧ a^2 + a*b - 3 = 0 then 4*a + b else 0

theorem minimum_value_of_4a_plus_b :
  ∀ (a b : ℝ), a > 0 → b > 0 → a^2 + a*b - 3 = 0 → 4*a + b ≥ 6 :=
by
  intros a b ha hb hab
  sorry

end minimum_value_of_4a_plus_b_l611_611702


namespace centroid_ineq_l611_611904

theorem centroid_ineq
  (A B C G A' B' C' : Type*)
  [AddGroup G] [Module ℝ G] [MetricSpace G] [NormedGroup G] [NormedSpace ℝ G]
  (M : Basis (Fin 3) ℝ G)
  (AD BE CF : G) 
  (circ_ABCCircumcircle : Circumcircles)
  (AG : ℝ)
  (BG : ℝ)
  (CG : ℝ)
  (A'G : ℝ)
  (B'G : ℝ)
  (C'G : ℝ)
  (hG : IsCentroidOf A B C G)
  (hA'median : IsMedian A D A')
  (hB'median : IsMedian B E B')
  (hC'median : IsMedian C F C')
  (hA'circ : OnCircumcircleOfTriangle circ_ABCCircumcircle A')
  (hB'circ : OnCircumcircleOfTriangle circ_ABCCircumcircle B')
  (hC'circ : OnCircumcircleOfTriangle circ_ABCCircumcircle C') :
  AG / A'G * BG / B'G * CG / C'G ≤ 1 :=
sorry

end centroid_ineq_l611_611904


namespace find_repair_cost_l611_611913

variable (P : ℕ) (T : ℕ) (SP : ℕ) (PP : ℕ) (R : ℕ)

-- Define the conditions as hypotheses
def conditions : Prop :=
  P = 11000 ∧
  T = 1000 ∧
  SP = 25500 ∧
  PP = 50

-- State the goal to prove
def goal : Prop :=
  SP = 1.5 * (P + T + R) → R = 5000

-- Combine them into a theorem
theorem find_repair_cost (h : conditions) : goal := by
  sorry

end find_repair_cost_l611_611913


namespace minimum_number_of_oranges_l611_611133

noncomputable def minimum_oranges_picked : ℕ :=
  let n : ℕ := 64 in n

theorem minimum_number_of_oranges (n : ℕ) : 
  (∀ (m_i m_j m_k : ℝ) (remaining_masses : Finset ℝ),
    remaining_masses.card = n - 3 →
    (m_i + m_j + m_k) < 0.05 * (∑ m in remaining_masses, m)) →
  n ≥ minimum_oranges_picked :=
by {
  sorry
}

end minimum_number_of_oranges_l611_611133


namespace dot_product_eq_l611_611428

-- Define data for an equilateral triangle with side length 1 and its centroid.
variable {A B C G : Point}
variable (h1 : equilateral_triangle A B C)
variable (h2 : centroid G A B C)
variable (h3 : vector_length (vector A B) = 1)
variable (h4 : vector_length (vector A G) = Real.sqrt 3 / 3)
variable (h5 : angle (vector A B) (vector A G) = π / 6)

-- Goal: Prove that the dot product of vector AB and vector AG is 1/2.
theorem dot_product_eq :
  (vector A B) ⬝ (vector A G) = 1 / 2 :=
sorry

end dot_product_eq_l611_611428


namespace range_of_a_monotonically_decreasing_l611_611757

noncomputable def h (a x : ℝ) : ℝ := log x - (1/2) * a * x^2 - 2 * x
noncomputable def h' (a x : ℝ) : ℝ := 1/x - a * x - 2

theorem range_of_a_monotonically_decreasing :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 4 → h' a x < 0) → a ∈ Ioi (-1) :=
by
  intro h'_decreasing
  -- Proof to be filled in
  sorry

end range_of_a_monotonically_decreasing_l611_611757


namespace speed_in_kmh_l611_611612

def distance : ℝ := 550.044
def time : ℝ := 30
def conversion_factor : ℝ := 3.6

theorem speed_in_kmh : (distance / time) * conversion_factor = 66.00528 := 
by
  sorry

end speed_in_kmh_l611_611612


namespace x_pow_10_eq_correct_answer_l611_611402

noncomputable def x : ℝ := sorry

theorem x_pow_10_eq_correct_answer (h : x + (1 / x) = Real.sqrt 5) : 
  x^10 = (50 + 25 * Real.sqrt 5) / 2 := 
sorry

end x_pow_10_eq_correct_answer_l611_611402


namespace find_number_l611_611221

theorem find_number (x : ℝ) (h : (x / 6) * 12 = 13) : x = 6.5 :=
by
  sorry

end find_number_l611_611221


namespace total_food_items_donated_l611_611675

def FosterFarmsDonation : ℕ := 45
def AmericanSummitsDonation : ℕ := 2 * FosterFarmsDonation
def HormelDonation : ℕ := 3 * FosterFarmsDonation
def BoudinButchersDonation : ℕ := HormelDonation / 3
def DelMonteFoodsDonation : ℕ := AmericanSummitsDonation - 30

theorem total_food_items_donated :
  FosterFarmsDonation + AmericanSummitsDonation + HormelDonation + BoudinButchersDonation + DelMonteFoodsDonation = 375 :=
by
  sorry

end total_food_items_donated_l611_611675


namespace Skew_Lines_l611_611080

-- Definitions for the problem
variable (A B : Point)
variable (α β : Plane)
variable (l : Line)

-- Conditions
axiom A_on_α : A ∈ α
axiom B_on_β : B ∈ β
axiom α_int_β_eq_l : α ∩ β = l
axiom A_not_on_l : A ∉ l
axiom B_not_on_l : B ∉ l

-- Problem statement: Given the conditions, prove that l and AB are skew lines
theorem Skew_Lines (A B α β l : Type) [Point A] [Point B] [Plane α] [Plane β] [Line l] 
  (A_on_α : A ∈ α)
  (B_on_β : B ∈ β)
  (α_int_β_eq_l : α ∩ β = l)
  (A_not_on_l : A ∉ l)
  (B_not_on_l : B ∉ l) : 
  are_skew_lines l (line_through A B) :=
by
  sorry

end Skew_Lines_l611_611080


namespace ns_value_l611_611062

namespace Proof

open Set

-- Define the set S as a nonzero reals
def S := {x : ℝ | x ≠ 0}

-- Define the function f with the given properties
def f (x : S) : S := sorry

-- The first property of function f
axiom f_property1 : ∀ x : S, f (1 / ↑x) = 2 * ↑x * f x

-- The second property of function f
axiom f_property2 : ∀ x y : S, x + y ∈ S → f (1 / ↑x) + f (1 / ↑y) = 2 + f (1 / (↑x + ↑y))

-- Define f(1)
def f1 := f ⟨1, by norm_num⟩

-- Define n as the count of possible values of f(1)
def n := 1

-- Define s as the sum of all possible values of f(1)
def s := (1 / (2 * 1 : ℝ) + 1)

-- Statement to prove
theorem ns_value : n * s = 3 / 2 :=
by 
    simp [n, s]
    norm_num

end Proof

end ns_value_l611_611062


namespace compare_sqrt1_compare_sqrt2_l611_611274

open Real

theorem compare_sqrt1 : 2 * sqrt 7 < 4 * sqrt 2 := 
by {
  calc 2 * sqrt 7 < 4 * sqrt 2 : sorry
}

theorem compare_sqrt2 : (sqrt 5 - 1) / 2 > 0.5 := 
by {
  calc (sqrt 5 - 1) / 2 > 0.5 : sorry
}


end compare_sqrt1_compare_sqrt2_l611_611274


namespace range_of_a_l611_611004

theorem range_of_a (a : ℝ) : 
  (∀ x, x ∈ {x | x^2 ≤ 1} ∪ {a} ↔ x ∈ {x | x^2 ≤ 1}) → (-1 ≤ a ∧ a ≤ 1) :=
by
  intro h
  sorry

end range_of_a_l611_611004


namespace intersection_points_count_l611_611758

theorem intersection_points_count :
  (∀ x : ℝ, f (x + 2) = f x) ∧
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f x = 1 - x^2) ∧
  (∀ x : ℝ, g x = Real.log (|x|)) →
  ∃ n : ℕ, n = 8 ∧
    ∀ I : Set ℝ, I = Set.Icc (-5 : ℝ) (5 : ℝ) →
      (SetOf (λ x, f x = g x) ∩ I).card = 8 :=
by
  sorry

end intersection_points_count_l611_611758


namespace smaller_angle_at_8_oclock_l611_611387

theorem smaller_angle_at_8_oclock : let angle_per_hour := 30
  let hour_at_8 := 8
  let total_degrees := 360
  let hour_angle := hour_at_8 * angle_per_hour
  let larger_angle := hour_angle
  let smaller_angle := if larger_angle > 180 then total_degrees - larger_angle else larger_angle
  in smaller_angle = 120 :=
by
  sorry

end smaller_angle_at_8_oclock_l611_611387


namespace problem_statement_l611_611697

-- Define the sequence satisfying the equation and the sum of its terms
def seq_an (n : ℕ) (h_pos : n > 0) : ℝ := 1 / (n * (n + 1))

def sum_sn (n : ℕ) (h_pos : n > 0) : ℝ := (Finset.range n).sum (λ i, seq_an (i + 1) (Nat.succ_pos i))

-- The goal is to prove that 2019 times the sum of the first 2018 terms is equal to 2018
theorem problem_statement : 2019 * sum_sn 2018 (Nat.succ_pos 2017) = 2018 := 
by sorry

end problem_statement_l611_611697


namespace minimum_oranges_condition_l611_611130

theorem minimum_oranges_condition (n : ℕ) :
  (∀ i j k : ℕ, i ≠ j → j ≠ k → i ≠ k → i < n → j < n → k < n → 
  let m := 1 in (3 * m / ((n-3) * m) < 0.05) → n ≥ 64) :=
begin
  intros h,
  sorry
end

end minimum_oranges_condition_l611_611130


namespace even_function_is_D_l611_611516

def A (x : ℝ) : ℝ := x + 1 / x
def B (x : ℝ) : ℝ := x^3
def C (x : ℝ) : ℝ := sqrt x
def D (x : ℝ) : ℝ := |x| + 1

theorem even_function_is_D : 
  ∀ (x : ℝ), (∀ (f : ℝ → ℝ), (f = A ∨ f = B ∨ f = C ∨ f = D) → (∀ x, f x = f (-x)) ↔ f = D) :=
sorry

end even_function_is_D_l611_611516


namespace odd_numbers_in_pascals_triangle_row_l611_611669

noncomputable def count_ones_in_binary (n : ℕ) : ℕ :=
  nat.popcount n

theorem odd_numbers_in_pascals_triangle_row (n : ℕ) : 
  ∃ k, k = count_ones_in_binary n ∧ (∃ m, m = 2^k ∧ (row_odd_count n = m)) :=
sorry

end odd_numbers_in_pascals_triangle_row_l611_611669


namespace Tyler_saltwater_animals_l611_611560

theorem Tyler_saltwater_animals :
  let total_freshwater_aquariums := 52
  let freshwater_aquariums_with_64_animals := 38
  let animals_per_64_aquarium := 64
  let total_freshwater_animals := 6310

  let total_saltwater_aquariums := 28
  let saltwater_aquariums_with_52_animals := 18
  let animals_per_52_aquarium := 52
  let min_saltwater_animals_per_aquarium := 20

  let animals_64 := freshwater_aquariums_with_64_animals * animals_per_64_aquarium
  let remaining_freshwater_animals := total_freshwater_animals - animals_64
  let animals_52 := saltwater_aquariums_with_52_animals * animals_per_52_aquarium
  let min_remaining_saltwater_animals := (total_saltwater_aquariums - saltwater_aquariums_with_52_animals) * min_saltwater_animals_per_aquarium
  let total_saltwater_animals := animals_52 + min_remaining_saltwater_animals

  total_saltwater_animals ≥ 1136 :=
by
  let total_freshwater_aquariums := 52
  let freshwater_aquariums_with_64_animals := 38
  let animals_per_64_aquarium := 64
  let total_freshwater_animals := 6310

  let total_saltwater_aquariums := 28
  let saltwater_aquariums_with_52_animals := 18
  let animals_per_52_aquarium := 52
  let min_saltwater_animals_per_aquarium := 20

  let animals_64 := freshwater_aquariums_with_64_animals * animals_per_64_aquarium
  let remaining_freshwater_animals := total_freshwater_animals - animals_64
  let animals_52 := saltwater_aquariums_with_52_animals * animals_per_52_aquarium
  let min_remaining_saltwater_animals := (total_saltwater_aquariums - saltwater_aquariums_with_52_animals) * min_saltwater_animals_per_aquarium
  let total_saltwater_animals := animals_52 + min_remaining_saltwater_animals

  show total_saltwater_animals ≥ 1136 from sorry

end Tyler_saltwater_animals_l611_611560


namespace circle_cartesian_eq_and_slope_l611_611000

-- Defining the polar equation of the circle
def circle_polar (θ : ℝ) : ℝ :=
  4 * Real.cos θ - 6 * Real.sin θ

-- Parametric equations of the line
def line_parametric (t θ : ℝ) : ℝ × ℝ :=
  (4 + t * Real.cos θ, t * Real.sin θ)

-- Proposition to be proved
theorem circle_cartesian_eq_and_slope (θ : ℝ) (t : ℝ) (P Q : ℝ × ℝ) :
  (circle_polar θ) = (4 * Real.cos θ - 6 * Real.sin θ) →
  (line_parametric t θ) = (4 + t * Real.cos θ, t * Real.sin θ) →
  let C_cartesian := (λ x y : ℝ, x^2 + y^2 - 4 * x + 6 * y = 0) in
  let center := (2, -3) in
  let radius := Real.sqrt 13 in
  let chord_length := 4 in
  -- Cartesian coordinates and properties
  (∀ x y, (circle_polar θ = 4 * Real.cos θ - 6 * Real.sin θ) →
    C_cartesian x y = (x-2)^2 + (y+3)^2 = 13)
  ∧ (center = (2, -3))
  ∧ (radius = Real.sqrt 13)
  -- Slope of the line
  ∧ (∀ k, (k = 0 ∨ k = -12/5)
  → chord_length = 4)
:=
sorry

end circle_cartesian_eq_and_slope_l611_611000


namespace largest_angle_is_75_l611_611171

-- Let the measures of the angles be represented as 3x, 4x, and 5x for some value x
variable (x : ℝ)

-- Define the angles based on the given ratio
def angle1 := 3 * x
def angle2 := 4 * x
def angle3 := 5 * x

-- The sum of the angles in a triangle is 180 degrees
axiom angle_sum : angle1 + angle2 + angle3 = 180

-- Prove that the largest angle is 75 degrees
theorem largest_angle_is_75 : 5 * (180 / 12) = 75 :=
by
  -- Proof is not required as per the instructions
  sorry

end largest_angle_is_75_l611_611171


namespace sector_area_l611_611713

theorem sector_area (r l : ℝ) (h1 : l + 2 * r = 8) (h2 : l = 2 * r) : 
  (1 / 2) * l * r = 4 := 
by sorry

end sector_area_l611_611713


namespace rectangle_in_semicircle_l611_611618

theorem rectangle_in_semicircle
  (h : ℝ)
  (EG : ℝ)
  (EQ EP QF : ℝ)
  (rect_inscribed : ∃ E F G H, ∃ O, ∃ r, ∃ semicircle_eq : ∀ P, EF = 2 * r)
  (area_ratio : ∃ P Q R, P ∈ line m ∧ Q ∈ line m ∧ R ∈ line m ∧ divides_area_ratio (region_semicircle_sub rect_insemicircle) 3 1)
  (E_Q_cond : EQ = 120)
  (E_P_cond : EP = 180)
  (Q_F_cond : QF = 240) :
  EG = 180 * sqrt 2 := sorry

end rectangle_in_semicircle_l611_611618


namespace find_a5_l611_611334

noncomputable def sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ ∀ n, a (n + 1) = 2 * a n + 3 * n

theorem find_a5 (a : ℕ → ℕ) (h : sequence a) : a 5 = 94 :=
sorry

end find_a5_l611_611334


namespace num_integer_solutions_abs_count_integer_solutions_l611_611010

theorem num_integer_solutions_abs (x y : ℤ) :
  (|x| + 1) * (|y| - 3) = 7 ↔ x = 0 ∧ (y = 10 ∨ y = -10) ∨ (x = 6 ∨ x = -6) ∧ (y = 4 ∨ y = -4) :=
sorry

theorem count_integer_solutions : (card {p : ℤ × ℤ | (|p.1| + 1) * (|p.2| - 3) = 7}) = 6 :=
sorry

end num_integer_solutions_abs_count_integer_solutions_l611_611010


namespace problem_1_part1_problem_1_part2_problem_2_l611_611234

theorem problem_1_part1 (x : ℝ) (h : x + x⁻¹ = 3) : x^(1/2) + x^(-1/2) = Real.sqrt 5 := 
by
  sorry

theorem problem_1_part2 (x : ℝ) (h : x + x⁻¹ = 3) : x^2 + x^(-2) = 7 := 
by
  sorry

theorem problem_2 : (Real.log 2)^2 + (Real.log 2) * (Real.log 50) + (Real.log 25) = 2 :=
by
  sorry

end problem_1_part1_problem_1_part2_problem_2_l611_611234


namespace orange_ribbons_count_l611_611426

def total_ribbons (yellow_fraction purple_fraction orange_fraction : ℚ) (silver_ribbons : ℕ) : ℕ :=
  let total_fraction_non_silver := yellow_fraction + purple_fraction + orange_fraction
  in silver_ribbons * 4 / (1 - total_fraction_non_silver.toReal) |_|

theorem orange_ribbons_count (yellow_fraction : ℚ) (purple_fraction : ℚ) (orange_fraction : ℚ) (silver_ribbons : ℕ) :
  yellow_fraction = 1/4 → purple_fraction = 1/3 → orange_fraction = 1/6 → silver_ribbons = 40 →
  let total := total_ribbons yellow_fraction purple_fraction orange_fraction silver_ribbons in
  total * orange_fraction = 27 :=
begin
  intros h1 h2 h3 h4,
  unfold total_ribbons,
  sorry
end

end orange_ribbons_count_l611_611426


namespace find_z_l611_611407

noncomputable def complex_solution (z : ℂ) : Prop :=
  arg (z^2 - 4) = 5 * Real.pi / 6 ∧ arg (z^2 + 4) = Real.pi / 3

theorem find_z (z : ℂ) (h : complex_solution z) : z = 1 + Complex.I * Real.sqrt 3 ∨ z = -1 - Complex.I * Real.sqrt 3 :=
by
  sorry

end find_z_l611_611407


namespace find_OH_l611_611858

theorem find_OH 
  (ABC : Triangle)
  (I O H : Point)
  (AI AO AH : ℝ)
  (H1 : incenter I ABC)
  (H2 : circumcenter O ABC)
  (H3 : orthocenter H ABC)
  (H4 : dist A I = 11)
  (H5 : dist A O = 13)
  (H6 : dist A H = 13) :
  dist O H = 10 :=
sorry

end find_OH_l611_611858


namespace fishing_problem_l611_611782

theorem fishing_problem
  (everyday : ℕ)
  (every_other_day : ℕ)
  (every_three_days : ℕ)
  (yesterday_fishing : ℕ)
  (today_fishing : ℕ)
  (h_everyday : everyday = 7)
  (h_every_other_day : every_other_day = 8)
  (h_every_three_days : every_three_days = 3)
  (h_yesterday_fishing : yesterday_fishing = 12)
  (h_today_fishing : today_fishing = 10) :
  (every_three_days + everyday + (every_other_day - (yesterday_fishing - everyday))) = 15 := by
  sorry

end fishing_problem_l611_611782


namespace question1_question2_l611_611735

variables (a b c : EuclideanSpace ℝ (Fin 2))

def a : EuclideanSpace ℝ (Fin 2) := ![1, 0]
def b : EuclideanSpace ℝ (Fin 2) := ![-1, 2]

-- For question (1)
def c : EuclideanSpace ℝ (Fin 2) := ![c1, c2]
def d := a - b

theorem question1 (hc : ∥c∥ = 1) (hc_parallel : ∃ k : ℝ, c = k • d) : 
  c = ![sqrt(2)/2, -sqrt(2)/2] ∨ c = ![-sqrt(2)/2, sqrt(2)/2] :=
sorry

-- For question (2)
variables (t : ℝ)

def v1 : EuclideanSpace ℝ (Fin 2) := 2 * t • a - b
def v2 : EuclideanSpace ℝ (Fin 2) := 3 • a + t • b

theorem question2 (h_perp : inner v1 v2 = 0) :
  t = -1 ∨ t = 3 / 2 :=
sorry

end question1_question2_l611_611735


namespace count_integer_points_l611_611177

-- Define the conditions: the parabola P with focus at (0,0) and passing through (6,4) and (-6,-4)
def parabola (P : ℝ × ℝ → Prop) : Prop :=
  ∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ 
  (∀ x y : ℝ, P (x, y) ↔ y = a*x^2 + b) ∧ 
  P (6, 4) ∧ P (-6, -4)

-- Define the main theorem to be proved: the count of integer points satisfying the inequality
theorem count_integer_points (P : ℝ × ℝ → Prop) (hP : parabola P) :
  ∃ n : ℕ, n = 45 ∧ ∀ (x y : ℤ), P (x, y) → |6 * x + 4 * y| ≤ 1200 :=
sorry

end count_integer_points_l611_611177


namespace smaller_angle_at_8_oclock_l611_611385

def degrees_per_hour : ℝ := 360 / 12

def angle_at_8_oclock : ℝ := 8 * degrees_per_hour

def full_circle : ℝ := 360

theorem smaller_angle_at_8_oclock :
  ∀ (time : ℝ), time = 8 → min angle_at_8_oclock (full_circle - angle_at_8_oclock) = 120 := 
by 
  sorry

end smaller_angle_at_8_oclock_l611_611385


namespace area_of_EPHQ_l611_611909

-- Definitions based on conditions:
def rectangle_area (length : ℝ) (width : ℝ) : ℝ :=
  length * width

def midpoint_segment_length (length : ℝ) : ℝ :=
  length / 2

def area_triangle (base : ℝ) (height : ℝ) : ℝ :=
  0.5 * base * height

theorem area_of_EPHQ (length width : ℝ) (P_midpoint : Prop) (Q_midpoint : Prop) :
  (length = 10) → (width = 5) → P_midpoint → Q_midpoint → rectangle_area length width - (area_triangle length (midpoint_segment_length width) + area_triangle width (midpoint_segment_length length)) = 25 :=
by
  intros h1 h2 h3 h4
  simp [rectangle_area, area_triangle, midpoint_segment_length, h1, h2]
  sorry

end area_of_EPHQ_l611_611909


namespace possible_CD_lengths_l611_611276

-- Define the problem conditions
def tetrahedron_edges (A B C D : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D]
  (AB AC AD BC BD CD : ℝ)
  (h_AB : AB = 4) (h_AC : AC = 6) (h_BC : BC = 6) (h_AD : AD = 7) (h_BD : BD = 7) : Prop :=
∃ R : ℝ, -- the radius of the cylinder
∀ (h_radius : AB = 2 * R),
-- Define the possible lengths of CD
CD = real.sqrt 41 + 2 * real.sqrt 7 ∨
CD = real.sqrt 41 - 2 * real.sqrt 7

-- State the theorem
theorem possible_CD_lengths (A B C D : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D]
  (AB AC AD BC BD CD : ℝ)
  (h_AB : AB = 4) (h_AC : AC = 6) (h_BC : BC = 6) (h_AD : AD = 7) (h_BD : BD = 7) :
  tetrahedron_edges A B C D AB AC AD BC BD CD h_AB h_AC h_BC h_AD h_BD :=
begin
  sorry, -- the proof is omitted
end

end possible_CD_lengths_l611_611276


namespace tangent_line_eq_sum_of_zeros_positive_l611_611366

-- Definitions from the conditions
def f (a x : ℝ) : ℝ := (a * (x + 1)) / (Real.exp x) + (1 / 2) * x ^ 2

-- Part (1): Prove the equation of the tangent line when a = 1
theorem tangent_line_eq (x : ℝ) (h : x = -1) (f_a : ∀ x, f 1 x = (x + 1) / (Real.exp x) + (1 / 2) * x ^ 2) :
  let y := f 1 x in
  ∃ m b, y = m * (-1) + b ∧ ∀ x, y = (Real.exp 1 - 1) * x + (Real.exp 1 - 1 / 2) := 
sorry

-- Part (2): Prove that x₁ + x₂ > 0 when a < 0 and f has two distinct zeros
theorem sum_of_zeros_positive (a : ℝ) (ha : a < 0)
  (hx1 x2 : ℝ) (hf_zero : f a x1 = 0 ∧ f a x2 = 0) (hx1_ne_x2 : x1 ≠ x2) :
  x1 + x2 > 0 :=
sorry

end tangent_line_eq_sum_of_zeros_positive_l611_611366


namespace power_quotient_example_l611_611649

theorem power_quotient_example (a : ℕ) (m n : ℕ) (h : 23^11 / 23^8 = 23^(11 - 8)) : 23^3 = 12167 := by
  sorry

end power_quotient_example_l611_611649


namespace dice_event_random_lower_probability_l611_611980

def throw_three_dice_event (d1 d2 d3 : ℕ) : Prop :=
  d1 = 6 ∧ d2 = 6 ∧ d3 = 6
  
def probability_three_six : ℝ :=
  1 / 216
  
theorem dice_event_random_lower_probability :
  ∃ e : Prop, (throw_three_dice_event 6 6 6) = e ∧ e ∧ probability_three_six = 1 / 216 :=
sorry

end dice_event_random_lower_probability_l611_611980


namespace angle_C_measure_l611_611633

theorem angle_C_measure
  (D C : ℝ)
  (h1 : C + D = 90)
  (h2 : C = 3 * D) :
  C = 67.5 :=
by
  sorry

end angle_C_measure_l611_611633


namespace more_pups_than_adult_dogs_l611_611443

def number_of_huskies := 5
def number_of_pitbulls := 2
def number_of_golden_retrievers := 4
def pups_per_husky := 3
def pups_per_pitbull := 3
def additional_pups_per_golden_retriever := 2
def pups_per_golden_retriever := pups_per_husky + additional_pups_per_golden_retriever

def total_pups := (number_of_huskies * pups_per_husky) + (number_of_pitbulls * pups_per_pitbull) + (number_of_golden_retrievers * pups_per_golden_retriever)
def total_adult_dogs := number_of_huskies + number_of_pitbulls + number_of_golden_retrievers

theorem more_pups_than_adult_dogs : (total_pups - total_adult_dogs) = 30 :=
by
  -- proof steps, which we will skip
  sorry

end more_pups_than_adult_dogs_l611_611443


namespace math_city_intersections_l611_611089

theorem math_city_intersections (n : ℕ) (h : n = 10)
  (streets : Fin n → (ℝ × ℝ)) (h_parallel : ∀ i j : Fin n, i ≠ j → streets i ≠ streets j) :
  ∑ i in Finset.range n, i = 45 :=
by
  rw h
  have sum_formula : ∀ k : ℕ, ∑ i in Finset.range k, i = k * (k - 1) / 2
  sorry
  exact sum_formula 10

end math_city_intersections_l611_611089


namespace area_of_region_l611_611667

open Set Real

-- Definitions of the given conditions
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 1
def x_axis (y : ℝ) : Prop := y = 0
def line (x y : ℝ) : Prop := 3 * x = 4 * y

-- Statement to prove the area is as specified
theorem area_of_region :
  let α := arctan (3 / 4) in
  ∫ θ in 0..α, 0.5 * (1 / (cos (2 * θ))) dθ = (log 7) / 4 :=
by
  sorry

end area_of_region_l611_611667


namespace magnitude_of_conjugate_z_l611_611744

theorem magnitude_of_conjugate_z (a b : ℝ) (h₀: b ≠ 0) 
  (h₁ : (2 + a * Complex.I) / (3 - Complex.I) = b * Complex.I) :
  Complex.abs (Complex.conj (a + b * Complex.I)) = 2 * Real.sqrt 10 :=
by
  sorry

end magnitude_of_conjugate_z_l611_611744


namespace standard_equation_of_ellipse_sum_of_slopes_constant_l611_611717

-- Given the ellipse C: x^2/a^2 + y^2/b^2 = 1 with a > b > 0 and an eccentricity of sqrt(6)/3,
-- and a circle centered at M(1,0) with the ellipse's minor axis as its radius tangent to the line x - y + sqrt(2) - 1 = 0,

-- Question 1: Prove that the standard equation of the ellipse C is x^2/3 + y^2 = 1.
theorem standard_equation_of_ellipse (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a > b) 
    (eccentricity_eq : real.sqrt 6 / 3 = real.sqrt (1 - b^2 / a^2)) : 
    ∃ a, b, (a = real.sqrt 3) ∧ (b = 1) ∧ (∀ x y : ℝ, (x^2 / 3 + y^2 = 1)) :=
by
  sorry

-- Question 2: Given point N(3,2), let line l pass through point M and intersect ellipse C at points A and B.
-- If the slopes of lines AN and BN are k1 and k2 respectively, prove that k1 + k2 is a constant value 2.
theorem sum_of_slopes_constant (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a > b) (M N : ℝ × ℝ) (hM : M = (1, 0)) (hN : N = (3, 2))
    (ellipse_eq : ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1) (l : ℝ → ℝ) (h_line_through_M : ∀ x, l x = (x - M.fst)) :
    ∀ A B : ℝ × ℝ, (A = (1, real.sqrt 6 / 3) ∨ A = (1, -real.sqrt 6 / 3)) ∧ (B = (1, real.sqrt 6 / 3) ∨ B = (1, -real.sqrt 6 / 3)) →
    ∃ k_1 k_2 : ℝ, (k_1 = (real.sqrt 6 / 3 - 2) / (1 - 3) ∨ k_1 = (-real.sqrt 6 / 3 - 2) / (1 - 3))
                ∧ (k_2 = (real.sqrt 6 / 3 - 2) / (1 - 3) ∨ k_2 = (-real.sqrt 6 / 3 - 2) / (1 - 3)) ∧ (k_1 + k_2 = 2) :=
by
  sorry

end standard_equation_of_ellipse_sum_of_slopes_constant_l611_611717


namespace oranges_min_count_l611_611143

theorem oranges_min_count (n : ℕ) (m : ℕ → ℝ) 
  (h : ∀ i j k, m i + m j + m k < 0.05 * ∑ l in (finset.univ \ {i, j, k}), m l) : 
  64 ≤ n :=
sorry

end oranges_min_count_l611_611143


namespace laurent_at_least_twice_chloe_l611_611272

noncomputable def probability_laurent_at_least_twice_chloe : ℝ :=
  let chloe_pdf : ℝ → ℝ := λ x, if 0 ≤ x ∧ x ≤ 1000 then 1 / 1000 else 0
  let laurent_pdf : ℝ → ℝ := λ y, if 0 ≤ y ∧ y ≤ 2000 then 1 / 2000 else 0  
  (∫ x in 0..1000, ∫ y in (2*x)..2000, laurent_pdf y * chloe_pdf x) / 
  (∫ x in 0..1000, ∫ y in 0..2000, laurent_pdf y * chloe_pdf x)

theorem laurent_at_least_twice_chloe : 
  probability_laurent_at_least_twice_chloe = 1 / 4 := 
by 
  sorry

end laurent_at_least_twice_chloe_l611_611272


namespace problem_statement_l611_611709

theorem problem_statement (a : Fin 2018 → ℝ) :
  (∑ i : Fin 2018, (i : ℕ + 1) * (if i % 2 = 0 then -1 else 1) * a i) = -4034 :=
by
  sorry

end problem_statement_l611_611709


namespace exponentiation_correct_l611_611578

theorem exponentiation_correct (a : ℝ) : (a ^ 2) ^ 3 = a ^ 6 :=
sorry

end exponentiation_correct_l611_611578


namespace domain_f_x_squared_minus_one_l611_611754

-- Define the function f and its domain
def f (x : ℝ) : ℝ := sorry  -- The specific function is not important for the domain analysis

-- Predicate representing the domain of a function
def in_domain (f : ℝ → ℝ) (d : set ℝ) : Prop :=
  ∀ x, x ∈ d → ∃ y, f y = x

-- Given 
axiom domain_f : in_domain f (set.Icc (-1 : ℝ) (1 : ℝ))

-- To Prove
theorem domain_f_x_squared_minus_one : 
  in_domain f (set.Icc (-(real.sqrt 2 : ℝ)) (real.sqrt 2 : ℝ)) :=
sorry

end domain_f_x_squared_minus_one_l611_611754


namespace john_saves_water_in_june_l611_611053

def dailyFlushes : Nat := 15
def oldFlushGallons : Nat := 5
def waterReductionRate : Rat := 0.8
def numDaysJune : Nat := 30

theorem john_saves_water_in_june :
  let oldDailyUsage := dailyFlushes * oldFlushGallons
  let oldMonthlyUsage := oldDailyUsage * numDaysJune
  let savingsPerFlush := oldFlushGallons * waterReductionRate
  let newFlushGallons := oldFlushGallons - savingsPerFlush.to_nat
  let newDailyUsage := dailyFlushes * newFlushGallons
  let newMonthlyUsage := newDailyUsage * numDaysJune
  oldMonthlyUsage - newMonthlyUsage = 1800 :=
by
  sorry

end john_saves_water_in_june_l611_611053


namespace identify_at_least_13_tricksters_l611_611189

theorem identify_at_least_13_tricksters
  (inhabitants : Finset ℕ)
  (H_inhabitants : inhabitants.card = 217)
  (knights tricksters : Finset ℕ)
  (H_knights : knights.card = 17)
  (H_tricksters : tricksters.card = 200)
  (lists : inhabitants → Finset ℕ)
  (H_knights_lists : ∀ k ∈ knights, ∀ t ∈ lists k, t ∈ tricksters)
  (H_tricksters_lists : ∀ t ∈ tricksters, lists t ⊆ inhabitants) 
  : ∃ (s : Finset ℕ), s.card ≥ 13 ∧ ∀ t ∈ s, t ∈ tricksters :=
sorry

end identify_at_least_13_tricksters_l611_611189


namespace median_remains_unchanged_l611_611842

theorem median_remains_unchanged (scores : List ℕ) (h_len : scores.length = 10) (h_sorted : scores = scores.sorted) :
  let scores' := scores.drop(1).take(8)
  scores.median = scores'.median :=
by
  sorry

end median_remains_unchanged_l611_611842


namespace angle_PQE_or_PQF_l611_611331

-- Definitions of the elements described in the problem
variables {A B C D E F M Q P I : Type}
variable [EuclideanGeometry A B C D E F M Q P I]

-- Point I is the incenter of the triangle ABC
variables (incircle : Incenter A B C I)
-- D, E, and F are the points where the incircle touches sides BC, CA, and AB respectively.
variables (D E F : PointOnICircle incircle)

variables (M : MidpointOfLineSegment B C)
variables (Q : PointOnICircle incircle)
variables (angle_AQD_90 : InnerAngle A Q D = 90)
variables (P : PointOnLine A I)
variables (MD_eq_MP : EuclideanDistance M D = EuclideanDistance M P)

-- The mathematical statement to prove
theorem angle_PQE_or_PQF :
  ∀ (angle_PQE angle_PQF : Angle),
    (angle_PQE = 90 ∨ angle_PQF = 90) :=
by
  sorry

end angle_PQE_or_PQF_l611_611331


namespace functional_inequality_solution_l611_611306

theorem functional_inequality_solution (f : ℝ → ℝ) (a b : ℝ) (differentiable : Differentiable ℝ f) :
  (∀ x y z : ℝ, f(x + y + z) + f(x) ≥ f(x + y) + f(x + z)) → (∀ x : ℝ, f x = a * x + b) :=
sorry

end functional_inequality_solution_l611_611306


namespace initial_distance_b_l611_611594

-- Define the conditions
variables (A B : Type) [AddGroup A] [AddGroup B]
          (v1 v2 : ℝ) -- speeds of A and B respectively
          (d : ℝ) -- initial distance between A and B
          (t1 t2 : ℝ) (L : ℝ)
          (hA : 0 ≤ t1) -- time of first overtake
          (hB : 0 ≤ t2) -- time of second overtake
          (hC : A = 0 ∨ B = 0) -- starting time

-- Hypotheses based on conditions
axiom speed_constant_A : ∀ t, v1 * t
axiom speed_constant_B : ∀ t, v2 * t
axiom first_overtake : v1 * t1 = v2 * t1 + L
axiom second_overtake : v1 * t2 = v2 * t2 + 2 * L

-- Theorem statement: Prove that initial distance is 180 meters
theorem initial_distance_b : d = 180 :=
by {
  sorry
}

end initial_distance_b_l611_611594


namespace find_equation_of_line_l611_611301

theorem find_equation_of_line 
  (l : ℝ → ℝ → Prop)
  (h_intersect : ∃ x y : ℝ, 2 * x + y - 8 = 0 ∧ x - 2 * y + 1 = 0 ∧ l x y)
  (h_parallel : ∀ x y : ℝ, l x y → 4 * x - 3 * y - 6 = 0) :
  ∀ x y : ℝ, l x y ↔ 4 * x - 3 * y - 6 = 0 :=
by
  sorry

end find_equation_of_line_l611_611301


namespace closed_broken_line_1996_possible_closed_broken_line_1997_impossible_l611_611645

theorem closed_broken_line_1996_possible :
  ∃ (s : Set (Set Point)), 
    (∀ t ∈ s, ∃! u ∈ s, intersects t u) ∧ 
    (∀ u v w ∈ s, ¬ collinear u v w) ∧
    |s| = 1996 :=
sorry

theorem closed_broken_line_1997_impossible :
  ¬ ∃ (s : Set (Set Point)),
    (∀ t ∈ s, ∃! u ∈ s, intersects t u) ∧ 
    (∀ u v w ∈ s, ¬ collinear u v w) ∧
    |s| = 1997 :=
sorry

end closed_broken_line_1996_possible_closed_broken_line_1997_impossible_l611_611645


namespace agents_monitoring_cyclic_odd_l611_611856

theorem agents_monitoring_cyclic_odd (n : ℕ) (h : ∀ i : ℕ, i > 0 ∧ i ≤ n → (i % n) + 1) : n % 2 = 1 :=
sorry

end agents_monitoring_cyclic_odd_l611_611856


namespace fishing_tomorrow_l611_611802

theorem fishing_tomorrow (every_day_fishers every_other_day_fishers every_three_days_fishers fishers_yesterday fishers_today : ℕ) :
  (every_day_fishers = 7) →
  (every_other_day_fishers = 8) →
  (every_three_days_fishers = 3) →
  (fishers_yesterday = 12) →
  (fishers_today = 10) →
  (every_three_days_fishers + every_day_fishers + (every_other_day_fishers - (fishers_yesterday - every_day_fishers)) = 15) :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  simp
  sorry

end fishing_tomorrow_l611_611802


namespace final_number_is_odd_l611_611523

theorem final_number_is_odd :
  ∀ (board : List ℕ),
  (∀ x ∈ board, 1 ≤ x ∧ x ≤ 2021) →
  (∀ x ∈ board, board.erase x ≠ []) →
  (∀ a b : ℕ, a ∈ board → b ∈ board → a ≠ b → board = (board.erase a).erase b ++ [Int.natAbs (a - b)]) →
  ∃ n : ℕ, board = [n] ∧ Nat.odd n := 
by
  intro board hcond1 hcond2 hcond3
  sorry

end final_number_is_odd_l611_611523


namespace fishing_tomorrow_l611_611831

-- Conditions
def every_day_fishers : Nat := 7
def every_other_day_fishers : Nat := 8
def every_three_days_fishers : Nat := 3
def yesterday_fishers : Nat := 12
def today_fishers : Nat := 10

-- Determine the number who will fish tomorrow
def fishers_tomorrow : Nat :=
  let every_day_tomorrow := every_day_fishers
  let every_three_day_tomorrow := every_three_days_fishers
  let every_other_day_yesterday := yesterday_fishers - every_day_fishers
  let every_other_day_tomorrow := every_other_day_fishers - every_other_day_yesterday
  every_day_tomorrow + every_three_day_tomorrow + every_other_day_tomorrow

theorem fishing_tomorrow : fishers_tomorrow = 15 :=
  by
    -- skipping the actual proof with sorry
    sorry

end fishing_tomorrow_l611_611831


namespace find_multiple_sales_l611_611864

theorem find_multiple_sales 
  (A : ℝ) 
  (M : ℝ)
  (h : M * A = 0.35294117647058826 * (11 * A + M * A)) 
  : M = 6 :=
sorry

end find_multiple_sales_l611_611864


namespace Sweetwater_discount_l611_611145

theorem Sweetwater_discount :
  ∀ (retail_price : ℝ) (GC_discount : ℝ) (GC_shipping : ℝ) (Sweetwater_shipping : ℝ) (saving : ℝ),
    retail_price = 1000 →
    GC_discount = 0.15 →
    GC_shipping = 100 →
    Sweetwater_shipping = 0 →
    saving = 50 →
    (retail_price - ((retail_price * GC_discount) + GC_shipping) - saving) / retail_price * 100 = 10 :=
by
  intros retail_price GC_discount GC_shipping Sweetwater_shipping saving
  intros h_retail_price h_GC_discount h_GC_shipping h_Sweetwater_shipping h_saving
  rw [h_retail_price, h_GC_discount, h_GC_shipping, h_Sweetwater_shipping, h_saving]
  norm_num
  sorry

end Sweetwater_discount_l611_611145


namespace sin_2x_eq_7_div_25_l611_611708

theorem sin_2x_eq_7_div_25 (x : ℝ) (h : Real.sin (Real.pi / 4 - x) = 3 / 5) :
    Real.sin (2 * x) = 7 / 25 := by
  sorry

end sin_2x_eq_7_div_25_l611_611708


namespace min_vertices_in_graph_l611_611205

/-- A graph representation with degree and cycle length conditions -/
structure Graph :=
(V : Type)
(E : V → V → Prop)
(degree : ∀ v : V, (finset.filter (E v) finset.univ).card = 3)
(no_cycle_lt_6 : ∀ (p : list V), p.nodup ∧ (∀ i < p.length - 1, E (p.nth i) (p.nth (i + 1))) → p.length ≥ 6)

noncomputable def minimum_vertices (G : Graph) : ℕ :=
14

-- The proof that any graph satisfying the given conditions has at least 14 vertices
theorem min_vertices_in_graph (G : Graph) : (G.V → Prop) ∃ v : finset V, finset.card v ≥ 14 :=
sorry

end min_vertices_in_graph_l611_611205


namespace fishers_tomorrow_l611_611794

-- Definitions based on conditions
def people_every_day : ℕ := 7
def people_every_other_day : ℕ := 8
def people_every_three_days : ℕ := 3
def people_yesterday : ℕ := 12
def people_today : ℕ := 10

-- Theorem to be proved
theorem fishers_tomorrow (people_every_day people_every_other_day people_every_three_days people_yesterday people_today : ℕ) : 
  people_every_day = 7 ∧ 
  people_every_other_day = 8 ∧ 
  people_every_three_days = 3 ∧
  people_yesterday = 12 ∧ 
  people_today = 10 →
  15 := 
by {
  sorry
}

end fishers_tomorrow_l611_611794


namespace true_proposition_given_conditions_l611_611341

theorem true_proposition_given_conditions
  (p : Prop) (q : Prop)
  (hp : p)
  (hq : ¬q) :
  (¬p ∨ ¬q) := 
by {
  show (¬p ∨ ¬q),
  from or.inr hq
}

end true_proposition_given_conditions_l611_611341


namespace sixty_percent_of_fifty_minus_forty_percent_of_thirty_l611_611742

theorem sixty_percent_of_fifty_minus_forty_percent_of_thirty : 
  (0.6 * 50) - (0.4 * 30) = 18 :=
by
  sorry

end sixty_percent_of_fifty_minus_forty_percent_of_thirty_l611_611742


namespace find_divisor_l611_611242

theorem find_divisor
  (n : ℕ) (add_least : ℕ) (divisor : ℕ)
  (h1 : n = 821562)
  (h2 : add_least = 6)
  (h3 : divisor = 6) :
  (n + add_least) % divisor = 0 :=
by
  rw [h1, h2, h3]
  norm_num

end find_divisor_l611_611242


namespace area_of_EPHQ_l611_611910

-- Definitions based on conditions:
def rectangle_area (length : ℝ) (width : ℝ) : ℝ :=
  length * width

def midpoint_segment_length (length : ℝ) : ℝ :=
  length / 2

def area_triangle (base : ℝ) (height : ℝ) : ℝ :=
  0.5 * base * height

theorem area_of_EPHQ (length width : ℝ) (P_midpoint : Prop) (Q_midpoint : Prop) :
  (length = 10) → (width = 5) → P_midpoint → Q_midpoint → rectangle_area length width - (area_triangle length (midpoint_segment_length width) + area_triangle width (midpoint_segment_length length)) = 25 :=
by
  intros h1 h2 h3 h4
  simp [rectangle_area, area_triangle, midpoint_segment_length, h1, h2]
  sorry

end area_of_EPHQ_l611_611910


namespace tiles_needed_correct_l611_611620

noncomputable def tiles_needed (floor_length : ℝ) (floor_width : ℝ) (tile_length_inch : ℝ) (tile_width_inch : ℝ) (border_width : ℝ) : ℝ :=
  let tile_length := tile_length_inch / 12
  let tile_width := tile_width_inch / 12
  let main_length := floor_length - 2 * border_width
  let main_width := floor_width - 2 * border_width
  let main_area := main_length * main_width
  let tile_area := tile_length * tile_width
  main_area / tile_area

theorem tiles_needed_correct :
  tiles_needed 15 20 3 9 1 = 1248 := 
by 
  sorry -- Proof skipped.

end tiles_needed_correct_l611_611620


namespace area_EPHQ_l611_611907

variable (EF GH : ℝ)
variable (EP EH FP HQ : ℝ)

-- Given conditions
axiom rect_EFGH : EF = 10 ∧ GH = 5
axiom midpoint_P : FP = GH / 2 ∧ EP = EF
axiom midpoint_Q : HQ = GH / 2 ∧ EH = GH

-- Proof of the area of region EPHQ
theorem area_EPHQ : EP * EH / 2 + FP * HQ / 2 = 25 :=
by
  have EF_value : EF = 10 := rect_EFGH.1
  have GH_value : GH = 5 := rect_EFGH.2
  sorry

end area_EPHQ_l611_611907


namespace positive_integers_b_log_b_243_l611_611740

theorem positive_integers_b_log_b_243 :
  {b : ℕ | ∃ n : ℕ, 0 < n ∧ b^n = 243}.card = 2 :=
by
  sorry

end positive_integers_b_log_b_243_l611_611740


namespace proof_problem_l611_611270

noncomputable def problem_statement : Prop :=
  ( (4 - (5 / 8))^(- 1 / 3) * (- 7 / 6)^0 + (1 / 3)^Real.log (1 / 2) + (1 / 2) * Real.log (25) / Real.log 10 + Real.log 2 / Real.log 10 ) = 11 / 3

theorem proof_problem : problem_statement :=
  by
  sorry

end proof_problem_l611_611270


namespace inequality_solution_l611_611922

theorem inequality_solution (x : ℝ) : 
  (4 ≤ 64 ^ ((log 4 x) ^ 2) - 15 * x ^ (log 4 x)) → 
  (0 < x ∧ x ≤ 1/4) ∨ (4 ≤ x) := 
sorry

end inequality_solution_l611_611922


namespace min_area_VPBC_l611_611876

def parabola (P : Point) := P.y^2 = 2 * P.x
def on_y_axis (B C : Point) := B.x = 0 ∧ C.x = 0
def inscribed_circle (V P B C : Point) := (V.x - 1)^2 + V.y^2 = 1

theorem min_area_VPBC {P B C V : Point} 
  (hP : parabola P)
  (hBC : on_y_axis B C)
  (hI : inscribed_circle V P B C) :
  area_VPBC V P B C ≥ 8 :=
sorry

end min_area_VPBC_l611_611876


namespace certain_number_value_l611_611022

theorem certain_number_value
  (t b c x : ℝ)
  (h1 : (t + b + c + x + 15) / 5 = 12)
  (h2 : (t + b + c + 29) / 4 = 15) :
  x = 14 :=
by 
  sorry

end certain_number_value_l611_611022


namespace sine_shift_left_l611_611549

theorem sine_shift_left (x : ℝ) : sin (x + 1) = sin x :=
by sorry

end sine_shift_left_l611_611549


namespace exists_k_for_blocks_of_2022_l611_611487

theorem exists_k_for_blocks_of_2022 (n : ℕ) (hn : 0 < n) :
  ∃ k : ℕ, (0 < k) ∧ (∀ i : ℕ, (1 ≤ i ∧ i ≤ n) → (∃ j, 
  k^i / 10^j % 10^4 = 2022)) :=
sorry

end exists_k_for_blocks_of_2022_l611_611487


namespace general_formula_an_Tn_inequality_l611_611695

-- Define the monotonically increasing sequence a_n and the sum S_n of the first n terms
variable {a : ℕ → ℕ}
variable (S : ℕ → ℕ)
variable (b : ℕ → ℚ)
variable (T : ℕ → ℚ)

-- Define the conditions given in the problem
axiom monotone_a : ∀ n, a n ≤ a (n + 1)
axiom Sn_def : ∀ n, S n = ∑ i in finset.range (n + 1), a i
axiom main_eq : ∀ n, 2 * S n = a n ^ 2 + n
axiom bn_def : ∀ n, b n = a (n + 2) / (2 ^ (n + 1) * a n * a (n + 1))
axiom Tn_def : ∀ n, T n = ∑ i in finset.range n, b i

-- Statement to prove the general formula for a_n and the inequality for T_n
theorem general_formula_an : ∀ n, a n = n :=
sorry

theorem Tn_inequality : ∀ n, T n < 1 / 2 :=
sorry

end general_formula_an_Tn_inequality_l611_611695


namespace initial_investment_l611_611305

theorem initial_investment :
  ∃ x : ℝ, P = 705.03 ∧ r = 0.12 ∧ n = 5 ∧ P = x * (1 + r)^n ∧ x = 400 :=
by
  let P := 705.03
  let r := 0.12
  let n := 5
  use 400
  simp [P, r, n]
  sorry

end initial_investment_l611_611305


namespace volume_expansion_rate_l611_611623

theorem volume_expansion_rate (R m : ℝ) (h1 : R = 1) (h2 : (4 * π * (m^3 - 1) / 3) / (m - 1) = 28 * π / 3) : m = 2 :=
sorry

end volume_expansion_rate_l611_611623


namespace seating_arrangement_l611_611431

theorem seating_arrangement :
  let total_arrangements := Nat.factorial 8
  let adjacent_arrangements := Nat.factorial 7 * 2
  total_arrangements - adjacent_arrangements = 30240 :=
by
  sorry

end seating_arrangement_l611_611431


namespace exp_values_l611_611017

variable {a x y : ℝ}

theorem exp_values (hx : a^x = 3) (hy : a^y = 2) :
  a^(x - y) = 3 / 2 ∧ a^(2 * x + y) = 18 :=
by
  sorry

end exp_values_l611_611017


namespace even_card_sum_probability_l611_611316

theorem even_card_sum_probability :
  let cards := Finset.range 10 \ 0, -- Cards from 1 to 9
      even_cards := cards.filter (λ n, n % 2 = 0), -- Filter even cards
      odd_cards := cards.filter (λ n, n % 2 = 1), -- Filter odd cards
      
      n := cards.card * (cards.card - 1) / 2, -- Total number of ways to pick 2 cards out of 9
      
      even_pairs := Finset.card (even_cards.pairCombinations), 
      odd_pairs := Finset.card (odd_cards.pairCombinations),
      
      m := even_pairs + odd_pairs, -- Total number of successful outcomes
      p := (m : ℚ) / n in -- Probability of successful outcome

      p = 4 / 9 := by {
  -- Card definition
  let cards := Finset.range 10 \ 0,
  -- Card counts
  have card_count := Finset.card cards,
  -- Even and odd cards
  let even_cards := cards.filter (λ n, n % 2 = 0),
  let odd_cards := cards.filter (λ n, n % 2 = 1),
  -- Number of ways to choose 2 cards out of 9
  have n := Finset.card (cards.pairCombinations),
  -- Even pairs count
  have even_pairs := Finset.card (even_cards.pairCombinations),
  -- Odd pairs count
  have odd_pairs := Finset.card (odd_cards.pairCombinations),
  -- Total success outcomes
  have m := even_pairs + odd_pairs,
  -- Probability computation
  let p := (m : ℚ) / n,
  -- Calculation
  have n_calc : n = 36 := sorry,
  have m_calc : m = 16 := sorry,
  -- Check
  rw [n_calc, m_calc] at p,
  norm_num at p,
  exact p,
}

end even_card_sum_probability_l611_611316


namespace intersection_of_sets_l611_611885

open Set -- This opens the Set namespace, which includes operations on sets

theorem intersection_of_sets :
  let P := {1, 2, 3, 4} : Set ℝ
  let Q := { x | |x| ≤ 3 } : Set ℝ
  P ∩ Q = {1, 2, 3} := by
sorry

end intersection_of_sets_l611_611885


namespace max_abs_diff_seven_up_eight_down_pairs_l611_611281

def is_upward_number (m : ℕ) := 
  ∃ (a : ℕ), m = 100 * a + 10 * (a + 1) + (a + 2) ∧ 1 ≤ a ∧ a ≤ 7

def is_downward_number (n : ℕ) := 
  ∃ (b : ℕ), n = 100 * b + 10 * (b - 1) + (b - 2) ∧ 3 ≤ b ∧ b ≤ 9

def F (m : ℕ) : ℕ := 7 * m

def G (n : ℕ) : ℕ := 8 * n

def valid_pair (m n : ℕ) := 
  is_upward_number m ∧ is_downward_number n ∧ (F(m) + G(n)) % 18 = 0

theorem max_abs_diff_seven_up_eight_down_pairs : 
  ∀ m n, valid_pair m n → |m - n| ≤ 531 :=
by sorry

end max_abs_diff_seven_up_eight_down_pairs_l611_611281


namespace trigonometric_identity_l611_611356

-- Definition of the given angle theta
def tan_theta : ℝ := 2
def theta : ℝ := Real.arctan tan_theta

-- Using trigonometric identities
theorem trigonometric_identity :
  (sin (3 * Real.pi / 2 + theta) + cos (Real.pi - theta)) /
  (sin (Real.pi / 2 - theta) - sin (Real.pi - theta)) = 2 :=
by
  sorry

end trigonometric_identity_l611_611356


namespace fishers_tomorrow_l611_611806

-- Define the groups of fishers and their fishing pattern
def everyday_fishers := 7
def every_other_day_fishers := 8
def every_three_days_fishers := 3

-- Given counts for yesterday and today
def fishers_yesterday := 12
def fishers_today := 10

-- The problem to prove: 15 people will fish tomorrow
theorem fishers_tomorrow : 
  (fishers_yesterday = 12) →
  (fishers_today = 10) →
  (everyday_fishers = 7) →
  (every_other_day_fishers = 8) →
  (every_three_days_fishers = 3) →
  15 = everyday_fishers + every_other_day_fishers / 2 + every_three_days_fishers / 3 :=
begin
  sorry
end

end fishers_tomorrow_l611_611806


namespace new_ratio_after_adding_6_l611_611556

theorem new_ratio_after_adding_6
  (x : ℤ) 
  (hx : 4 * x = 24)
  (new_smaller : ℤ := x + 6) :
  new_ratio : ℤ × ℤ := (new_smaller, 24) :=
  sorry

end new_ratio_after_adding_6_l611_611556


namespace complex_abs_squared_l611_611155

variable {z : ℂ} (hz : z * complex.abs z = 3 + 12 * complex.I)

theorem complex_abs_squared (hz : z * |z| = 3 + 12 * complex.I) : |z|^2 = 3 :=
sorry

end complex_abs_squared_l611_611155


namespace probability_difference_multiple_of_six_l611_611656

theorem probability_difference_multiple_of_six (S : Finset ℤ) (h_card : S.card = 8) 
  (h_range : ∀ x ∈ S, 1 ≤ x ∧ x ≤ 2010) (h_even : ∀ x ∈ S, x % 2 = 0) :
  ∃ x y ∈ S, x ≠ y ∧ (x - y) % 6 = 0 :=
by
  sorry

end probability_difference_multiple_of_six_l611_611656


namespace range_of_t_l611_611732

theorem range_of_t 
  (A : ℝ) (t : ℝ)
  (hA1 : 0 < A) 
  (hA2 : A < Real.pi / 2)
  (h_eq : (sin A) * (sqrt 3 * cos A) + (cos A) * (-cos A) = t - 1 / 2) : 
  (-1 / 2 < t ∧ t ≤ 1 / 2) ∨ (t = 1) :=
sorry

end range_of_t_l611_611732


namespace sequence_not_necessarily_periodic_l611_611694

theorem sequence_not_necessarily_periodic
  (a : ℕ → ℝ)
  (h_finite : {n : ℕ | a n ∈ set.range a}.finite)
  (h_periodic_subseq : ∀ k > 1, ∃ T, ∀ n, a (k * (n + T)) = a (k * n)) :
  ¬ (∃ T, ∀ n, a (n + T) = a n) :=
sorry

end sequence_not_necessarily_periodic_l611_611694


namespace quadrilateral_perimeter_l611_611567

/--
Given a quadrilateral \(EFGH\) with the following properties:
- \( \overline{EF} \perp \overline{FG} \)
- \( \overline{HG} \perp \overline{FG} \)
- \( EF = 12 \text{ cm} \)
- \( HG = 3 \text{ cm} \)
- \( FG = 16 \text{ cm} \)

Prove that the perimeter of \( EFGH \) is \( 31 + \sqrt{337} \text{ cm} \).
-/
theorem quadrilateral_perimeter (EF FG HG : ℝ)
  (h0 : EF = 12) (h1 : FG = 16) (h2 : HG = 3)
  (h3 : ⟪(0, EF)⟫ ⊥ ⟪(FG, 0)⟫) (h4 : ⟪(0, HG)⟫ ⊥ ⟪(FG, 0)⟫) :
  EF + FG + HG + Real.sqrt (EF^2 + (FG - HG)^2) = 31 + Real.sqrt 337 := 
by {
  -- This theorem states the perimeter calculation based on the given conditions
  sorry
}

end quadrilateral_perimeter_l611_611567


namespace fishing_tomorrow_l611_611772

theorem fishing_tomorrow 
  (P1 : ∀ day : ℕ, 7 ∈ {p | p goes fishing on day})
  (P2 : ∀ day : ℕ, day % 2 = 0 → 8 ∈ {p | p goes fishing on day})
  (P3 : ∀ day : ℕ, day % 3 = 0 → 3 ∈ {p | p goes fishing on day})
  (P4 : ℕ)
  (yesterday : ℕ)
  (today : ℕ)
  (tomorrow : ℕ)
  (hyesterday : yesterday = 12)
  (htoday : today = 10)
  : tomorrow = 15 := by
  sorry

end fishing_tomorrow_l611_611772


namespace simplify_evaluate_expression_l611_611147

theorem simplify_evaluate_expression (a b : ℤ) (h1 : a = -1) (h2 : b = 2) :
  3 * (a^2 * b + a * b^2) - 2 * (a^2 * b - 1) - 2 * (a * b^2) - 2 = -2 :=
by
  sorry

end simplify_evaluate_expression_l611_611147


namespace sequence_general_formula_summation_b_n_l611_611337

theorem sequence_general_formula (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h1 : a 5 = 9) (h2 : S 5 = 25)
  (h3 : ∀ n, S n = n * (2 * a 1 + (n - 1) * ((a 2 - a 1) / 1)) / 2): 
  (∀ n, a n = 2 * n - 1) :=
sorry

theorem summation_b_n (a : ℕ → ℤ) (b : ℕ → ℚ)
  (h1 : ∀ n, a n = 2 * n - 1)
  (h2 : ∀ n, b n = 1 / (a n * a (n + 1))):
  (∑ i in Finset.range 100, b (i + 1) = 100 / 201) :=
sorry

end sequence_general_formula_summation_b_n_l611_611337


namespace find_principal_l611_611991

theorem find_principal :
  ∃ P r : ℝ, (8820 = P * (1 + r) ^ 2) ∧ (9261 = P * (1 + r) ^ 3) → (P = 8000) :=
by
  sorry

end find_principal_l611_611991


namespace minimum_number_of_oranges_l611_611137

noncomputable def minimum_oranges_picked : ℕ :=
  let n : ℕ := 64 in n

theorem minimum_number_of_oranges (n : ℕ) : 
  (∀ (m_i m_j m_k : ℝ) (remaining_masses : Finset ℝ),
    remaining_masses.card = n - 3 →
    (m_i + m_j + m_k) < 0.05 * (∑ m in remaining_masses, m)) →
  n ≥ minimum_oranges_picked :=
by {
  sorry
}

end minimum_number_of_oranges_l611_611137


namespace distinct_real_roots_transformation_l611_611103

theorem distinct_real_roots_transformation (a b c : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x3 ≠ x1 ∧
                   polynomial.eval x1 (polynomial.X^3 + a * polynomial.X^2 + b * polynomial.X + polynomial.C c) = 0 ∧
                   polynomial.eval x2 (polynomial.X^3 + a * polynomial.X^2 + b * polynomial.X + polynomial.C c) = 0 ∧
                   polynomial.eval x3 (polynomial.X^3 + a * polynomial.X^2 + b * polynomial.X + polynomial.C c) = 0) →
  (∃ y1 y2 y3 : ℝ, y1 ≠ y2 ∧ y2 ≠ y3 ∧ y3 ≠ y1 ∧
                   polynomial.eval y1 (polynomial.X^3 + a * polynomial.X^2 + polynomial.C (1/4 * (a^2 + b)) * polynomial.X + polynomial.C (1/8 * (ab - c))) = 0 ∧
                   polynomial.eval y2 (polynomial.X^3 + a * polynomial.X^2 + polynomial.C (1/4 * (a^2 + b)) * polynomial.X + polynomial.C (1/8 * (ab - c))) = 0 ∧
                   polynomial.eval y3 (polynomial.X^3 + a * polynomial.X^2 + polynomial.C (1/4 * (a^2 + b)) * polynomial.X + polynomial.C (1/8 * (ab - c))) = 0) :=
sorry

end distinct_real_roots_transformation_l611_611103


namespace factor_polynomial_l611_611755

theorem factor_polynomial (x y z : ℝ) :
  x^3 * (y^2 - z^2) + y^3 * (z^2 - x^2) + z^3 * (x^2 - y^2) = 
  (x - y) * (y - z) * (z - x) * (x * y + x * z + y * z) := by
  sorry

end factor_polynomial_l611_611755


namespace tan_b_plus_tan_c_minus_two_tan_b_tan_c_l611_611417

def area_of_triangle (a b c S : ℝ) := S = 1/2 * (b^2 + c^2 - a^2)

theorem tan_b_plus_tan_c_minus_two_tan_b_tan_c
  (a b c B C S : ℝ)
  (area_expr : area_of_triangle a b c S) :
  tan B + tan C - 2 * tan B * tan C = -2 :=
sorry

end tan_b_plus_tan_c_minus_two_tan_b_tan_c_l611_611417


namespace no_ratio_p_squared_l611_611075

theorem no_ratio_p_squared {p : ℕ} (hp : Nat.Prime p) :
  ∀ l n m : ℕ, 1 ≤ l → (∃ k : ℕ, k = p^l) → ((2 * (n*(n+1)) = (m*(m+1))*p^(2*l)) → false) := 
sorry

end no_ratio_p_squared_l611_611075


namespace maximum_sides_for_convex_polygon_l611_611034

-- Define the convex polygon with n sides
structure ConvexPolygon (n : ℕ) :=
  (interior_angles_sum : ℝ)
  (obtuse_angles : Finset ℝ) -- Use finite set of ℝ for obtuse angles
  (acute_angles : Finset ℝ)  -- Use finite set of ℝ for acute angles

-- Definition of conditions as used in the problem
def satisfies_conditions (polygon : ConvexPolygon n) : Prop :=
  polygon.interior_angles_sum = 180 * (n - 2) ∧
  polygon.obtuse_angles.card = 4 ∧
  polygon.acute_angles.card = n - 4 ∧
  ∀ θ ∈ polygon.obtuse_angles, 90 < θ ∧
  ∀ θ ∈ polygon.acute_angles, θ < 90

-- Main theorem to be proved
theorem maximum_sides_for_convex_polygon : ∀ (n : ℕ), satisfies_conditions (ConvexPolygon n) → n ≤ 7 := 
sorry

end maximum_sides_for_convex_polygon_l611_611034


namespace remainder_of_f_100_div_100_l611_611277

def pascal_triangle_row_sum (n : ℕ) : ℕ :=
  2^n - 2

theorem remainder_of_f_100_div_100 : 
  (pascal_triangle_row_sum 100) % 100 = 74 :=
by
  sorry

end remainder_of_f_100_div_100_l611_611277


namespace line_through_parabola_vertex_unique_value_l611_611679

theorem line_through_parabola_vertex_unique_value :
  ∃! a : ℝ, ∃ y : ℝ, y = x + a ∧ y = x^2 - 2*a*x + a^2 :=
sorry

end line_through_parabola_vertex_unique_value_l611_611679


namespace tangent_to_both_circles_l611_611928

noncomputable def circle : Type := sorry
noncomputable def tangent (A B : circle) (P : Type) : Prop := sorry
noncomputable def point_on_circle (P : Type) (C : circle) : Prop := sorry
noncomputable def second_intersection_point (L : Type) (C : circle) : Type := sorry
noncomputable def line : Type := sorry
noncomputable def is_tangent (L : line) (C : circle) : Prop := sorry

variables (S S₁ S₂ : circle) (A₁ A₂ B : Type) (K₁ K₂ : Type)

-- Conditions
axiom h1 : tangent S S₁ A₁
axiom h2 : tangent S S₂ A₂
axiom h3 : point_on_circle B S
axiom h4 : K₁ = second_intersection_point (line A₁ B) S₁
axiom h5 : K₂ = second_intersection_point (line A₂ B) S₂
axiom h6 : is_tangent (line K₁ K₂) S₁

-- Proof statement
theorem tangent_to_both_circles : is_tangent (line K₁ K₂) S₂ :=
sorry

end tangent_to_both_circles_l611_611928


namespace fishing_tomorrow_l611_611822

theorem fishing_tomorrow (yesterday_fishers today_fishers : ℕ)
  (every_day_fishers every_other_day_fishers every_three_days_fishers : ℕ)
  (total_population : ℕ):
  yesterday_fishers = 12 → 
  today_fishers = 10 → 
  every_day_fishers = 7 → 
  every_other_day_fishers = 8 → 
  every_three_days_fishers = 3 → 
  total_population = yesterday_fishers + today_fishers + (total_population - (every_day_fishers + every_other_day_fishers + every_three_days_fishers)) →
  ∃ tomorrow_fishers : ℕ, tomorrow_fishers = 15 :=
by {
  -- This is a statement definition, the proof is not required and thus marked as "sorry:"
  sorry
}

end fishing_tomorrow_l611_611822


namespace abs_alpha_eq_sqrt_10_l611_611064

theorem abs_alpha_eq_sqrt_10 (α β : ℂ) (h₁ : α.re = β.re ∧ α.im = -β.im) 
  (h₂ : (α^2 / β^3).im = 0) (h₃ : abs (α - β) = 2 * Real.sqrt 5) : abs α = Real.sqrt 10 := 
by
  sorry

end abs_alpha_eq_sqrt_10_l611_611064


namespace bricks_required_l611_611602

-- Definitions
def courtyard_length : ℝ := 25
def courtyard_width : ℝ := 18
def brick_length : ℝ := 20 / 100  -- Converting cm to meters for consistency
def brick_width : ℝ := 10 / 100   -- Converting cm to meters for consistency

-- Theorem statement
theorem bricks_required : (25 * 18 * 10000) / (20 * 10) = 22500 := by
  sorry

end bricks_required_l611_611602


namespace pure_imaginary_solution_l611_611023

theorem pure_imaginary_solution (a : ℝ) (i : ℂ) (h : i = complex.I) (h_imag : complex.re (↑a + 6 * i) / (3 - i) = 0) : a = 2 := by
  sorry

end pure_imaginary_solution_l611_611023


namespace feb1_is_wednesday_l611_611747

-- Define the days of the week as a data type
inductive DayOfWeek
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

open DayOfWeek

-- Define a function that models the backward count for days of the week from a given day
def days_backward (start : DayOfWeek) (days : Nat) : DayOfWeek :=
  match days % 7 with
  | 0 => start
  | 1 => match start with
         | Sunday => Saturday
         | Monday => Sunday
         | Tuesday => Monday
         | Wednesday => Tuesday
         | Thursday => Wednesday
         | Friday => Thursday
         | Saturday => Friday
  | 2 => match start with
         | Sunday => Friday
         | Monday => Saturday
         | Tuesday => Sunday
         | Wednesday => Monday
         | Thursday => Tuesday
         | Friday => Wednesday
         | Saturday => Thursday
  | 3 => match start with
         | Sunday => Thursday
         | Monday => Friday
         | Tuesday => Saturday
         | Wednesday => Sunday
         | Thursday => Monday
         | Friday => Tuesday
         | Saturday => Wednesday
  | 4 => match start with
         | Sunday => Wednesday
         | Monday => Thursday
         | Tuesday => Friday
         | Wednesday => Saturday
         | Thursday => Sunday
         | Friday => Monday
         | Saturday => Tuesday
  | 5 => match start with
         | Sunday => Tuesday
         | Monday => Wednesday
         | Tuesday => Thursday
         | Wednesday => Friday
         | Thursday => Saturday
         | Friday => Sunday
         | Saturday => Monday
  | 6 => match start with
         | Sunday => Monday
         | Monday => Tuesday
         | Tuesday => Wednesday
         | Wednesday => Thursday
         | Thursday => Friday
         | Friday => Saturday
         | Saturday => Sunday
  | _ => start  -- This case is unreachable because days % 7 is always between 0 and 6

-- Proof statement: given February 28 is a Tuesday, prove that February 1 is a Wednesday
theorem feb1_is_wednesday (h : days_backward Tuesday 27 = Wednesday) : True :=
by
  sorry

end feb1_is_wednesday_l611_611747


namespace oblique_asymptote_l611_611564

theorem oblique_asymptote : 
  (tendsto (λ x, (3*x^2 + 4*x + 8) / (x + 5) - (3*x - 11)) at_top (nhds 0)) ∧ 
  (tendsto (λ x, (3*x^2 + 4*x + 8) / (x + 5) - (3*x - 11)) at_bot (nhds 0)) := 
sorry

end oblique_asymptote_l611_611564


namespace projection_AC_on_AB_l611_611323

-- Define the points
structure Point3D where
  x : ℝ 
  y : ℝ 
  z : ℝ 

def A : Point3D := ⟨1, -2, 1⟩
def B : Point3D := ⟨1, -5, 4⟩
def C : Point3D := ⟨2, 3, 4⟩

-- Define vector subtraction
def vectorSub (p1 p2 : Point3D) : Point3D :=
  ⟨p1.x - p2.x, p1.y - p2.y, p1.z - p2.z⟩

-- Define dot product
def dotProduct (v1 v2 : Point3D) : ℝ :=
  v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

-- Define magnitude
def magnitude (v : Point3D) : ℝ :=
  Real.sqrt (v.x^2 + v.y^2 + v.z^2)

-- Define scalar multiplication
def scalarMul (k : ℝ) (v : Point3D) : Point3D :=
  ⟨k * v.x, k * v.y, k * v.z⟩

-- Define projection calculation
def projection (v1 v2 : Point3D) : Point3D :=
  let scaleFactor : ℝ := (dotProduct v1 v2) / (magnitude v2) / (magnitude v2)
  scalarMul scaleFactor v2

-- The statement we want to prove
theorem projection_AC_on_AB : 
  projection (vectorSub C A) (vectorSub B A) = ⟨0, 1, -1⟩ := 
by 
  -- placeholder for the proof steps
  sorry

end projection_AC_on_AB_l611_611323


namespace B_joins_after_nine_months_l611_611248

theorem B_joins_after_nine_months :
  ∃ x : ℕ, x ≈ 9 ∧ 
    (4500 * 12) / (16200 * (12 - x)) = 2 / 3 :=
by
  sorry

end B_joins_after_nine_months_l611_611248


namespace initial_budget_calculation_l611_611608

variable (flaskCost testTubeCost safetyGearCost totalExpenses remainingAmount initialBudget : ℕ)

theorem initial_budget_calculation (h1 : flaskCost = 150)
                               (h2 : testTubeCost = 2 * flaskCost / 3)
                               (h3 : safetyGearCost = testTubeCost / 2)
                               (h4 : totalExpenses = flaskCost + testTubeCost + safetyGearCost)
                               (h5 : remainingAmount = 25)
                               (h6 : initialBudget = totalExpenses + remainingAmount) :
                               initialBudget = 325 := by
  sorry

end initial_budget_calculation_l611_611608


namespace triangle_inscribed_dot_product_eq_minus_one_fifth_l611_611229

variables {ℝ : Type*} [inner_product_space ℝ (fin 3)]

open inner_product_space

def triangle_inscribed_circle (A B C O : ℝ (fin 3)) (r : ℝ)
  (hO_radius : ∥O - A∥ = 1 ∧ ∥O - B∥ = 1 ∧ ∥O - C∥ = 1)
  (h_vectors : 3 • (O - A) + 4 • (O - B) + 5 • (O - C) = 0) : Prop :=
  (O - C) ⬝ (B - A) = -1/5

-- Lean 4 statement theorem
theorem triangle_inscribed_dot_product_eq_minus_one_fifth 
  {A B C O : ℝ (fin 3)} : 
  ∥O - A∥ = 1 ∧ ∥O - B∥ = 1 ∧ ∥O - C∥ = 1 ∧ 
  3 • (O - A) + 4 • (O - B) + 5 • (O - C) = 0 → 
  (O - C) ⬝ (B - A) = -1/5 :=
begin
  sorry
end

end triangle_inscribed_dot_product_eq_minus_one_fifth_l611_611229


namespace pascal_sum_diff_l611_611342

theorem pascal_sum_diff :
  let a_i (i : ℕ) := Nat.choose 2015 i
  let b_i (i : ℕ) := Nat.choose 2016 i
  let c_i (i : ℕ) := Nat.choose 2017 i in
  (∑ i in Finset.range (2016 + 1), (b_i i) / (c_i i)) - 
  (∑ i in Finset.range (2015 + 1), (a_i i) / (b_i i)) = 1 / 2 := 
by
  sorry

end pascal_sum_diff_l611_611342


namespace chess_tournament_games_l611_611751

theorem chess_tournament_games (n : ℕ) (h : n * (n - 1) / 2 = 190) : n = 20 :=
sorry

end chess_tournament_games_l611_611751


namespace existence_of_points_for_equal_areas_l611_611333

variable {Point : Type} [add_group Point] [affine_space Point]
variable {Rectangle Point : Type} [measurable_space (Rectangle Point)] [has_area (Rectangle Point)]
variables {A B C D E F G H L M R S : Point}
variable {EF GH : segment Point} -- EF and GH as segments

/- The conditions given in the problem -/
variables (rect : Rectangle Point)
variables (E_on_AB : E ∈ segment A B) (F_on_CD : F ∈ segment C D) (G_on_DA : G ∈ segment D A)
variables (H_on_EF : H ∈ segment E F)
variables (EF_divides : divides EF rect) (GH_divides : divides GH rect)

/- The areas we've to match -/
variable [decidable_eq Point]

-- Prove existence of such points L, M, R, S
theorem existence_of_points_for_equal_areas :
  ∃ (L M : Point), L ∈ segment A B ∧ M ∈ segment C D ∧ area (rectangle B C M L) = area (trapezoid B C F E) ∧
  ∃ (R S : Point), R ∈ segment D A ∧ S ∈ segment L M ∧ area (rectangle A L S R) = area (quadrilateral A E H G) :=
sorry

end existence_of_points_for_equal_areas_l611_611333


namespace volume_of_regular_tetrahedron_l611_611511

noncomputable def volume_of_tetrahedron (a : ℝ) : ℝ :=
  (a ^ 3 * Real.sqrt 2) / 12

theorem volume_of_regular_tetrahedron (a : ℝ) : 
  volume_of_tetrahedron a = (a ^ 3 * Real.sqrt 2) / 12 := 
by
  sorry

end volume_of_regular_tetrahedron_l611_611511


namespace projection_AC_on_AB_l611_611320

/-- Define points in 3D space -/
structure Point3D :=
(x : ℝ)
(y : ℝ)
(z : ℝ)

/-- Calculate vector from two points -/
def vector (P Q : Point3D) : Point3D :=
{ x := Q.x - P.x, y := Q.y - P.y, z := Q.z - P.z }

/-- Calculate dot product of two vectors -/
def dot_product (u v : Point3D) : ℝ :=
u.x * v.x + u.y * v.y + u.z * v.z

/-- Calculate the magnitude of a vector -/
def magnitude (v : Point3D) : ℝ :=
Real.sqrt (v.x * v.x + v.y * v.y + v.z * v.z)

/-- Define the projection of vector u on vector v -/
def projection (u v : Point3D) : Point3D :=
let scalar_proj := (dot_product u v) / (magnitude v) in
{ x := scalar_proj * v.x / (magnitude v), y := scalar_proj * v.y / (magnitude v), z := scalar_proj * v.z / (magnitude v) }

noncomputable def A : Point3D := { x := 1, y := -2, z := 1 }
noncomputable def B : Point3D := { x := 1, y := -5, z := 4 }
noncomputable def C : Point3D := { x := 2, y := 3, z := 4 }

noncomputable def AB : Point3D := vector A B
noncomputable def AC : Point3D := vector A C

theorem projection_AC_on_AB :
  projection AC AB = { x := 0, y := 1, z := -1 } :=
sorry

end projection_AC_on_AB_l611_611320


namespace monotonic_decreasing_interval_l611_611526

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * x^2 - Real.log x

def decreasing_interval (a b : ℝ) := 
  ∀ x : ℝ, a < x ∧ x < b → deriv f x < 0

theorem monotonic_decreasing_interval : decreasing_interval 0 1 :=
sorry

end monotonic_decreasing_interval_l611_611526


namespace only_A_can_form_triangle_l611_611214

/--
Prove that from the given sets of lengths, only the set {5cm, 8cm, 12cm} can form a valid triangle.

Given:
- A: 5 cm, 8 cm, 12 cm
- B: 2 cm, 3 cm, 6 cm
- C: 3 cm, 3 cm, 6 cm
- D: 4 cm, 7 cm, 11 cm

We need to show that only Set A satisfies the triangle inequality theorem.
-/
theorem only_A_can_form_triangle :
  (∀ (a b c : ℕ), a = 5 ∧ b = 8 ∧ c = 12 → a + b > c ∧ a + c > b ∧ b + c > a) ∧
  (∀ (a b c : ℕ), a = 2 ∧ b = 3 ∧ c = 6 → ¬(a + b > c ∧ a + c > b ∧ b + c > a)) ∧
  (∀ (a b c : ℕ), a = 3 ∧ b = 3 ∧ c = 6 → ¬(a + b > c ∧ a + c > b ∧ b + c > a)) ∧
  (∀ (a b c : ℕ), a = 4 ∧ b = 7 ∧ c = 11 → ¬(a + b > c ∧ a + c > b ∧ b + c > a)) :=
by
  sorry -- Proof to be provided

end only_A_can_form_triangle_l611_611214


namespace triangle_def_dm_simplified_sum_l611_611551

theorem triangle_def_dm_simplified_sum 
  (DE EF FD : ℕ) 
  (DE_len : DE = 9) 
  (EF_len : EF = 10) 
  (FD_len : FD = 11)
  (cos_angle_DEF : (DE^2 + EF^2 - FD^2) / (2 * DE * EF) = 1/3)
  (omega3 : ω3.passes_through E ∧ ω3.is_tangent DF D)
  (omega4 : ω4.passes_through F ∧ ω4.is_tangent DE D)
  (intersection_property : ∃ M, M ≠ D ∧ M ∈ ω3 ∧ M ∈ ω4) :
  let DM := 99/16 in
  (∃ p q : ℕ, DM = p / q ∧ gcd p q = 1 ∧ (p + q) = 115) :=
sorry

end triangle_def_dm_simplified_sum_l611_611551


namespace product_remainder_l611_611076

open Complex

noncomputable def z : ℂ := exp ((2 * π * I) / 101)
noncomputable def omega : ℂ := exp ((2 * π * I) / 10)

theorem product_remainder :
  (∏ a in Finset.range 10, ∏ b in Finset.range 101, ∏ c in Finset.range 101, 
  (omega^a + z^b + z^c) : ℤ) % 101 = 13 :=
sorry

end product_remainder_l611_611076


namespace positive_solution_x_l611_611731

theorem positive_solution_x (x y z : ℝ) (h1 : x * y = 10 - 3 * x - 2 * y) 
(h2 : y * z = 10 - 5 * y - 3 * z) 
(h3 : x * z = 40 - 5 * x - 2 * z) 
(h_pos : x > 0) : 
  x = 8 :=
sorry

end positive_solution_x_l611_611731


namespace molecular_weight_of_one_mole_l611_611973

theorem molecular_weight_of_one_mole (total_weight : ℝ) (number_of_moles : ℕ) 
    (h : total_weight = 204) (n : number_of_moles = 3) : 
    (total_weight / number_of_moles) = 68 :=
by
  have h_weight : total_weight = 204 := h
  have h_moles : number_of_moles = 3 := n
  rw [h_weight, h_moles]
  norm_num

end molecular_weight_of_one_mole_l611_611973


namespace bottle_caps_problem_l611_611051

def number_of_boxes (total_caps caps_per_box : ℕ) : ℕ :=
  total_caps / caps_per_box

theorem bottle_caps_problem : 
  ∀ (total_caps caps_per_box : ℕ), 
    total_caps = 316 ∧ caps_per_box = 4 → 
    number_of_boxes total_caps caps_per_box = 79 :=
by
  intros total_caps caps_per_box h
  cases h with h1 h2
  rw [h1, h2]
  exact Nat.div_eq_of_eq_mul_left (by norm_num) (by norm_num)

end bottle_caps_problem_l611_611051


namespace find_local_value_of_7_in_difference_l611_611204

-- Define the local value of 3 in the number 28943712.
def local_value_of_3_in_28943712 : Nat := 30000

-- Define the property that the local value of 7 in a number Y is 7000.
def local_value_of_7 (Y : Nat) : Prop := (Y / 1000 % 10) = 7

-- Define the unknown number X and its difference with local value of 3 in 28943712.
variable (X : Nat)

-- Assumption: The difference between X and local_value_of_3_in_28943712 results in a number whose local value of 7 is 7000.
axiom difference_condition : local_value_of_7 (X - local_value_of_3_in_28943712)

-- The proof problem statement to be solved.
theorem find_local_value_of_7_in_difference : local_value_of_7 (X - local_value_of_3_in_28943712) = true :=
by
  -- Proof is omitted.
  sorry

end find_local_value_of_7_in_difference_l611_611204


namespace sum_remainder_l611_611018

theorem sum_remainder (m : ℤ) : ((9 - m) + (m + 5)) % 8 = 6 :=
by
  sorry

end sum_remainder_l611_611018


namespace factorial_trailing_zeros_50_l611_611406

theorem factorial_trailing_zeros_50 : 
  (nat.trailing_zeros (nat.factorial 50) = 12) :=
sorry

end factorial_trailing_zeros_50_l611_611406


namespace smallest_y_square_l611_611766

theorem smallest_y_square (y n : ℕ) (h1 : y = 10) (h2 : ∀ m : ℕ, (∃ z : ℕ, m * y = z^2) ↔ (m = n)) : n = 10 :=
sorry

end smallest_y_square_l611_611766


namespace min_oranges_picked_l611_611126

noncomputable def minimum_oranges (n : ℕ) : Prop :=
  ∀ (m : ℕ → ℕ), (∀ i j k : ℕ, i ≠ j ∧ j ≠ k ∧ i ≠ k → (m i + m j + m k : ℝ) / (∑ l : ℕ in finset.univ \ {i, j, k}, m l) < 0.05) → n ≥ 64

theorem min_oranges_picked : ∃ n, minimum_oranges n := by
  use 64
  sorry

end min_oranges_picked_l611_611126


namespace geometric_series_sum_l611_611641

theorem geometric_series_sum :
  let a := 3
  let r := 3
  let n := 9
  let last_term := a * r^(n - 1)
  last_term = 19683 →
  let S := a * (r^n - 1) / (r - 1)
  S = 29523 :=
by
  intros
  sorry

end geometric_series_sum_l611_611641


namespace range_of_a_monotonically_decreasing_l611_611756

noncomputable def f (a x : ℝ) : ℝ := a * x^3 - 3 * x

theorem range_of_a_monotonically_decreasing (a : ℝ) :
  (∀ x : ℝ, -1 < x ∧ x < 1 → deriv (f a) x ≤ 0) ↔ 0 < a ∧ a ≤ 1 :=
by
  sorry

end range_of_a_monotonically_decreasing_l611_611756


namespace smallest_y_square_factor_l611_611764

theorem smallest_y_square_factor (y n : ℕ) (h₀ : y = 10) 
  (h₁ : ∀ m : ℕ, ∃ k : ℕ, k * k = m * y)
  (h₂ : ∀ (y' : ℕ), (∀ m : ℕ, ∃ k : ℕ, k * k = m * y') → y ≤ y') : 
  n = 10 :=
by sorry

end smallest_y_square_factor_l611_611764


namespace cos_x_plus_2y_eq_one_l611_611707

theorem cos_x_plus_2y_eq_one 
  (x y a : ℝ)
  (hx : x ∈ set.Icc (-π / 4) (π / 4))
  (hy : y ∈ set.Icc (-π / 4) (π / 4))
  (h1 : x^3 + Real.sin x - 2 * a = 0)
  (h2 : 4 * y^3 + Real.sin y * Real.cos y + a = 0) : 
  Real.cos (x + 2 * y) = 1 :=
sorry

end cos_x_plus_2y_eq_one_l611_611707


namespace fishing_problem_l611_611783

theorem fishing_problem
  (everyday : ℕ)
  (every_other_day : ℕ)
  (every_three_days : ℕ)
  (yesterday_fishing : ℕ)
  (today_fishing : ℕ)
  (h_everyday : everyday = 7)
  (h_every_other_day : every_other_day = 8)
  (h_every_three_days : every_three_days = 3)
  (h_yesterday_fishing : yesterday_fishing = 12)
  (h_today_fishing : today_fishing = 10) :
  (every_three_days + everyday + (every_other_day - (yesterday_fishing - everyday))) = 15 := by
  sorry

end fishing_problem_l611_611783


namespace inequality_amgm_l611_611486

theorem inequality_amgm (a b c d : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) :
  Real.sqrt ((a ^ 2 + b ^ 2 + c ^ 2 + d ^ 2) / 4) ≥ Real.cbrt ((a * b * c + a * b * d + a * c * d + b * c * d) / 4) := 
by
  sorry

end inequality_amgm_l611_611486


namespace natural_number_sequences_l611_611094

def identical_lines_eventually (a : Fin 2018 → ℕ) : Prop :=
  ∃ n : ℕ, ∀ m ≥ n, 
    let seq := fun l : ℕ => (fun k : Fin 2018 => (Finset.univ.filter (λ i, a (Fin 2018 i k) = i)).card)
    in seq n = seq m

theorem natural_number_sequences (a : Fin 2018 → ℕ) : identical_lines_eventually a :=
sorry

end natural_number_sequences_l611_611094


namespace fishing_tomorrow_l611_611800

theorem fishing_tomorrow (every_day_fishers every_other_day_fishers every_three_days_fishers fishers_yesterday fishers_today : ℕ) :
  (every_day_fishers = 7) →
  (every_other_day_fishers = 8) →
  (every_three_days_fishers = 3) →
  (fishers_yesterday = 12) →
  (fishers_today = 10) →
  (every_three_days_fishers + every_day_fishers + (every_other_day_fishers - (fishers_yesterday - every_day_fishers)) = 15) :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  simp
  sorry

end fishing_tomorrow_l611_611800


namespace volume_of_tetrahedron_correct_l611_611927

noncomputable def volume_of_tetrahedron (PQ PR PS QR QS RS : ℝ)
  (h_PQ : PQ = 6)
  (h_PR : PR = 5)
  (h_PS : PS = 5)
  (h_QR : QR = 5)
  (h_QS : QS = 5)
  (h_RS : RS = 4)
  (h_PQR : PR = QR ∧ PQ = QR) : ℝ :=
  let V := (72 * Real.sqrt 3) / 5 in
  V

theorem volume_of_tetrahedron_correct :
  ∀ PQ PR PS QR QS RS : ℝ,
    PQ = 6 →
    PR = 5 →
    PS = 5 →
    QR = 5 →
    QS = 5 →
    RS = 4 →
    (PR = QR ∧ PQ = QR) →
    volume_of_tetrahedron PQ PR PS QR QS RS PQ PR PS QR QS RS h_PQ h_PR h_PS h_QR h_QS h_RS h_PQR = (72 * Real.sqrt 3) / 5 :=
by
  intros PQ PR PS QR QS RS h_PQ h_PR h_PS h_QR h_QS h_RS h_PQR
  sorry

end volume_of_tetrahedron_correct_l611_611927


namespace average_value_of_T_l611_611870

noncomputable def avg_value (T : Finset ℕ) : ℝ :=
  (↑(T.sum id) : ℝ) / (T.card : ℝ)

noncomputable def avg_without (T : Finset ℕ) (i : ℕ) : ℝ :=
  (↑((T.erase i).sum id) : ℝ) / ((T.card - 1 : ℕ) : ℝ)

noncomputable def avg_without_two (T : Finset ℕ) (i j : ℕ) : ℝ :=
  (↑((T.erase i).erase j.sum id) : ℝ) / ((T.card - 2 : ℕ) : ℝ)

theorem average_value_of_T (T : Finset ℕ) (b₁ bₘ : ℕ) (h₁ : b₁ ∈ T) (h₂ : bₘ ∈ T) (h₁₂ : b₁ < bₘ)
  (h₁T : avg_without (T.erase bₘ) b₁ = 50)
  (h₂T : avg_without bₘ b₁ = 55)
  (h₃T : avg_without T bₘ = 45)
  (h₄T : bₘ = b₁ + 90) :
  avg_value T = 49.5 :=
by
  sorry

end average_value_of_T_l611_611870


namespace hall_width_l611_611837

theorem hall_width
  (L H E C : ℝ)
  (hL : L = 20)
  (hH : H = 5)
  (hE : E = 57000)
  (hC : C = 60) :
  ∃ w : ℝ, (w * 50 + 100) * C = E ∧ w = 17 :=
by
  use 17
  simp [hL, hH, hE, hC]
  sorry

end hall_width_l611_611837


namespace fishing_problem_l611_611779

theorem fishing_problem
  (everyday : ℕ)
  (every_other_day : ℕ)
  (every_three_days : ℕ)
  (yesterday_fishing : ℕ)
  (today_fishing : ℕ)
  (h_everyday : everyday = 7)
  (h_every_other_day : every_other_day = 8)
  (h_every_three_days : every_three_days = 3)
  (h_yesterday_fishing : yesterday_fishing = 12)
  (h_today_fishing : today_fishing = 10) :
  (every_three_days + everyday + (every_other_day - (yesterday_fishing - everyday))) = 15 := by
  sorry

end fishing_problem_l611_611779


namespace ratio_of_container_volumes_l611_611258

-- Define the volumes of the first and second containers.
variables (A B : ℝ )

-- Hypotheses based on the problem conditions
-- First container is 4/5 full
variable (h1 : A * 4 / 5 = B * 2 / 3)

-- The statement to prove
theorem ratio_of_container_volumes : A / B = 5 / 6 :=
by
  sorry

end ratio_of_container_volumes_l611_611258


namespace oranges_min_number_l611_611111

theorem oranges_min_number (n : ℕ) 
  (h : ∀ (m : ℕ → ℝ), (∀ i j k : ℕ, i < n → j < n → k < n →
    i ≠ j → j ≠ k → i ≠ k → m i + m j + m k < 0.05 * (∑ l in (finset.range n \ {i,j,k}), m l))) : 
  n ≥ 64 :=
sorry

end oranges_min_number_l611_611111


namespace fishing_tomorrow_l611_611829

-- Conditions
def every_day_fishers : Nat := 7
def every_other_day_fishers : Nat := 8
def every_three_days_fishers : Nat := 3
def yesterday_fishers : Nat := 12
def today_fishers : Nat := 10

-- Determine the number who will fish tomorrow
def fishers_tomorrow : Nat :=
  let every_day_tomorrow := every_day_fishers
  let every_three_day_tomorrow := every_three_days_fishers
  let every_other_day_yesterday := yesterday_fishers - every_day_fishers
  let every_other_day_tomorrow := every_other_day_fishers - every_other_day_yesterday
  every_day_tomorrow + every_three_day_tomorrow + every_other_day_tomorrow

theorem fishing_tomorrow : fishers_tomorrow = 15 :=
  by
    -- skipping the actual proof with sorry
    sorry

end fishing_tomorrow_l611_611829


namespace slower_train_speed_is_36_l611_611558

noncomputable def speed_of_slower_train
  (length_train : ℝ)
  (faster_speed : ℝ)
  (passing_time : ℝ)
  (relative_distance : ℝ) : ℝ :=
  let relative_speed := (relative_distance / passing_time) * 18 / 5
  faster_speed - relative_speed

theorem slower_train_speed_is_36 :
  ∀ (length_train : ℝ) (faster_speed : ℝ) (passing_time : ℝ)
    (relative_distance : ℝ), 
    length_train = 65 → 
    faster_speed = 49 →
    passing_time = 36 →
    relative_distance = 2 * length_train →
    speed_of_slower_train length_train faster_speed passing_time relative_distance = 36 :=
by
  intros
  rw [h, ←show (2 * 65 : ℝ) = 130, by norm_num] at h3
  calc
    speed_of_slower_train 65 49 36 130
        = 49 - ((130 / 36) * 18 / 5) : rfl
    ... = 36 : by norm_num

end slower_train_speed_is_36_l611_611558


namespace max_edges_k_clique_free_is_k_minus_1_partite_turaan_theorem_k_minus_1_partite_max_clique_cardinality_bound_l611_611986

open Classical

noncomputable theory

-- Problem a
theorem max_edges_k_clique_free_is_k_minus_1_partite (G : Graph) (k : ℕ) (hG : ¬ ∃ (S : Finset G.V), S.card = k ∧ ∀ (u v ∈ S), G.adj u v) (hMax : ∀ (H : Graph), (¬ ∃ (S : Finset H.V), S.card = k ∧ ∀ (u v ∈ S), H.adj u v) → H.edges ≤ G.edges) :
  ∃ (part : Finset (Finset G.V)), part.card = k - 1 ∧ (∀ (u v ∈ G.V), ¬ G.adj u v → ∃ P ∈ part, u ∈ P ∧ v ∈ P) := sorry

-- Problem b
theorem turaan_theorem_k_minus_1_partite (n k : ℕ) (h : 2 ≤ k) :
  ∀ (G : Graph), G.is_k_minus_1_partite n (k - 1) → G.edges ≤ (n ^ 2 / 2) * (1 - 1 / (k - 1)) := sorry

-- Problem c
theorem max_clique_cardinality_bound (G : Graph) (n : ℕ) (degrees : Finset ℕ) (forall d ∈ degrees, ∃ v : G.V, G.degree v = d) :
  ∃ T : ℕ, T ≥ ∑ i in degrees, 1 / (n - i) := sorry

end max_edges_k_clique_free_is_k_minus_1_partite_turaan_theorem_k_minus_1_partite_max_clique_cardinality_bound_l611_611986


namespace fishing_tomorrow_l611_611823

theorem fishing_tomorrow (yesterday_fishers today_fishers : ℕ)
  (every_day_fishers every_other_day_fishers every_three_days_fishers : ℕ)
  (total_population : ℕ):
  yesterday_fishers = 12 → 
  today_fishers = 10 → 
  every_day_fishers = 7 → 
  every_other_day_fishers = 8 → 
  every_three_days_fishers = 3 → 
  total_population = yesterday_fishers + today_fishers + (total_population - (every_day_fishers + every_other_day_fishers + every_three_days_fishers)) →
  ∃ tomorrow_fishers : ℕ, tomorrow_fishers = 15 :=
by {
  -- This is a statement definition, the proof is not required and thus marked as "sorry:"
  sorry
}

end fishing_tomorrow_l611_611823


namespace oranges_min_number_l611_611114

theorem oranges_min_number (n : ℕ) 
  (h : ∀ (m : ℕ → ℝ), (∀ i j k : ℕ, i < n → j < n → k < n →
    i ≠ j → j ≠ k → i ≠ k → m i + m j + m k < 0.05 * (∑ l in (finset.range n \ {i,j,k}), m l))) : 
  n ≥ 64 :=
sorry

end oranges_min_number_l611_611114


namespace min_value_expression_l611_611457

theorem min_value_expression (θ φ : ℝ) :
  (3 * cos θ + 4 * sin φ - 10)^2 + (3 * sin θ + 4 * cos φ - 20)^2 ≥ 235.97 :=
by
  sorry

end min_value_expression_l611_611457


namespace total_interest_l611_611536

variables (P : ℝ) (R₁ R₂ R₃ : ℝ) (T₁ T₂ T₃ : ℝ) (SI₁ SI₂ SI₃ : ℝ)

-- Condition: The simple interest on a sum of money is Rs. 400 after 5 years at an annual interest rate of 5%.
axiom h1 : SI₁ = P * R₁ * T₁ / 100
axiom h2 : SI₁ = 400
axiom h3 : R₁ = 5
axiom h4 : T₁ = 5

-- Condition: The principal is trebled after the first 5 years.
noncomputable def P' := 3 * P

-- Condition: The annual interest rate is increased to 7% for the next 3 years.
axiom h5 : R₂ = 7
axiom h6 : T₂ = 3
axiom h7 : SI₂ = P' * R₂ * T₂ / 100

-- Condition: The annual interest rate is reduced to 4% for the remaining 2 years.
axiom h8 : R₃ = 4
axiom h9 : T₃ = 2
axiom h10 : SI₃ = P' * R₃ * T₃ / 100

-- Prove: What will be the total interest at the end of the tenth year?
theorem total_interest (h1 h2 h3 h4 h5 h6 h7 h8 h9 h10) :
  SI₁ + SI₂ + SI₃ = 539.2 :=
sorry

end total_interest_l611_611536


namespace fishing_tomorrow_l611_611833

-- Conditions
def every_day_fishers : Nat := 7
def every_other_day_fishers : Nat := 8
def every_three_days_fishers : Nat := 3
def yesterday_fishers : Nat := 12
def today_fishers : Nat := 10

-- Determine the number who will fish tomorrow
def fishers_tomorrow : Nat :=
  let every_day_tomorrow := every_day_fishers
  let every_three_day_tomorrow := every_three_days_fishers
  let every_other_day_yesterday := yesterday_fishers - every_day_fishers
  let every_other_day_tomorrow := every_other_day_fishers - every_other_day_yesterday
  every_day_tomorrow + every_three_day_tomorrow + every_other_day_tomorrow

theorem fishing_tomorrow : fishers_tomorrow = 15 :=
  by
    -- skipping the actual proof with sorry
    sorry

end fishing_tomorrow_l611_611833


namespace oranges_min_number_l611_611112

theorem oranges_min_number (n : ℕ) 
  (h : ∀ (m : ℕ → ℝ), (∀ i j k : ℕ, i < n → j < n → k < n →
    i ≠ j → j ≠ k → i ≠ k → m i + m j + m k < 0.05 * (∑ l in (finset.range n \ {i,j,k}), m l))) : 
  n ≥ 64 :=
sorry

end oranges_min_number_l611_611112


namespace set_of_points_union_of_lines_and_ellipses_l611_611104

theorem set_of_points_union_of_lines_and_ellipses :
  {p : ℝ × ℝ × ℝ | let (x, y, z) := p in x^4 - x^2 = y^4 - y^2 ∧ y^4 - y^2 = z^4 - z^2} =
  (⋃ (i j : ℝ), {p : ℝ × ℝ × ℝ | let (x, y, z) := p in x = i * y ∧ y = j * z ∨ x = i * y ∧ y^2 + z^2 = 1 ∨ x^2 + y^2 = 1 ∧ y = j * z ∨ x^2 + y^2 = 1 ∧ y^2 + z^2 = 1}) := 
sorry

end set_of_points_union_of_lines_and_ellipses_l611_611104


namespace can_transform_w1_to_w2_l611_611086

-- Define the operations allowed on words
structure Word where
  letters : List Char

def remove_first (w : Word) : Word :=
  { letters := w.letters.tail! }

def remove_last (w : Word) : Word :=
  { letters := w.letters.dropLast 1 }

def double (w : Word) : Word :=
  { letters := w.letters ++ w.letters }

-- Initial and target words
def w1 : Word := { letters := ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 
                               'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 
                               'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'] }

def w2 : Word := { letters := ['Z', 'Y', 'X', 'W', 'V', 'U', 'T', 'S', 'R', 
                               'Q', 'P', 'O', 'N', 'M', 'L', 'K', 'J', 'I', 
                               'H', 'G', 'F', 'E', 'D', 'C', 'B', 'A'] }

-- Statement that we want to prove
theorem can_transform_w1_to_w2 :
  ∃ (steps : List (Word → Word)), 
    (steps.foldl (λ w f => f w) w1 = w2) :=
by
  sorry

end can_transform_w1_to_w2_l611_611086


namespace oranges_min_count_l611_611141

theorem oranges_min_count (n : ℕ) (m : ℕ → ℝ) 
  (h : ∀ i j k, m i + m j + m k < 0.05 * ∑ l in (finset.univ \ {i, j, k}), m l) : 
  64 ≤ n :=
sorry

end oranges_min_count_l611_611141


namespace p_is_sufficient_but_not_necessary_for_q_l611_611883

def p (x : ℝ) : Prop := 1 < x ∧ x < 2
def q (x : ℝ) : Prop := 2^x > 1

theorem p_is_sufficient_but_not_necessary_for_q (x : ℝ) : 
  ((p x → q x) ∧ ¬(q x → p x)) :=
by
  sorry

end p_is_sufficient_but_not_necessary_for_q_l611_611883


namespace compare_x_y_l611_611347

variable (a b : ℝ)
variable (a_pos : 0 < a)
variable (b_pos : 0 < b)
variable (a_ne_b : a ≠ b)

noncomputable def x : ℝ := (Real.sqrt a + Real.sqrt b) / Real.sqrt 2
noncomputable def y : ℝ := Real.sqrt (a + b)

theorem compare_x_y : y a b > x a b := sorry

end compare_x_y_l611_611347


namespace payment_per_minor_character_l611_611444

noncomputable def M : ℝ := 285000 / 19 

theorem payment_per_minor_character
    (num_main_characters : ℕ := 5)
    (num_minor_characters : ℕ := 4)
    (total_payment : ℝ := 285000)
    (payment_ratio : ℝ := 3)
    (eq1 : 5 * 3 * M + 4 * M = total_payment) :
    M = 15000 :=
by
  sorry

end payment_per_minor_character_l611_611444


namespace rectangle_perimeter_of_right_triangle_l611_611254

-- Define the conditions for the triangle and the rectangle
def rightTriangleArea (a b c : ℕ) (h : a^2 + b^2 = c^2) : ℕ :=
  (1 / 2) * a * b

def rectanglePerimeter (width area : ℕ) : ℕ :=
  2 * ((area / width) + width)

theorem rectangle_perimeter_of_right_triangle :
  ∀ (a b c width : ℕ) (h_a : a = 5) (h_b : b = 12) (h_c : c = 13)
    (h_pyth : a^2 + b^2 = c^2) (h_width : width = 5)
    (h_area_eq : rightTriangleArea a b c h_pyth = width * (rightTriangleArea a b c h_pyth / width)),
  rectanglePerimeter width (rightTriangleArea a b c h_pyth) = 22 :=
by
  intros
  sorry

end rectangle_perimeter_of_right_triangle_l611_611254


namespace correct_calculation_l611_611210

theorem correct_calculation : 
  ¬((sqrt 2)^0 = sqrt 2) ∧ 
  ¬(2 * sqrt 3 + 3 * sqrt 3 = 5 * sqrt 6) ∧ 
  ¬(sqrt 8 = 4 * sqrt 2) ∧ 
  (sqrt 3 * (2 * sqrt 3 - 2) = 6 - 2 * sqrt 3) := by
  sorry

end correct_calculation_l611_611210


namespace no_tiling_with_sphinx_l611_611271

def triangle_sphinx_tiling_impossible : Prop :=
  ∀ (n : ℕ), n = 6 → (∃ (numTriangles: ℤ), numTriangles = 36 ∧ 
  (∃ (grisTriangles blancTriangles : ℤ), grisTriangles = 15 ∧ blancTriangles = 21) ∧ 
  (∀ (sphinxGris sphinxBlanc : ℤ), sphinxGris = 4 ∧ sphinxBlanc = 2 → 
  (grisTriangles % 2 = 1 ∧ blancTriangles % 2 = 1) → 
  ∀ (k : ℤ), k * sphinxGris ≤ grisTriangles ∧ k * sphinxBlanc ≤ blancTriangles 
  → false))

theorem no_tiling_with_sphinx : triangle_sphinx_tiling_impossible := 
by 
  intro n h₁
  use 36
  split
  . assumption
  use 15
  use 21
  split
  . assumption
  split
  . assumption
  intro sphinxGris sphinxBlanc hsphinx k htiling
  sorry

end no_tiling_with_sphinx_l611_611271


namespace roots_polynomial_expression_l611_611077

-- Let a, b, c be the roots of the polynomial x^3 - 3x - 2 = 0.
noncomputable def is_root (p : Polynomial ℚ) (x : ℚ) : Prop := p.eval x = 0

-- Define the polynomial
def poly : Polynomial ℚ := Polynomial.C 1 * X^3 - Polynomial.C 3 * X - Polynomial.C 2

-- Prove the main statement
theorem roots_polynomial_expression :
  ∀ a b c : ℚ, is_root poly a → is_root poly b → is_root poly c →
  a + b + c = 0 → ab + ac + bc = -3 → abc = -2 →
  a * (b + c)^2 + b * (c + a)^2 + c * (a + b)^2 = 6 :=
by
  intros a b c ha hb hc habc sumabcsum avbivbc
  sorry

end roots_polynomial_expression_l611_611077


namespace odd_perfect_square_of_sigma_eq_2n_add_1_l611_611879

theorem odd_perfect_square_of_sigma_eq_2n_add_1 (n : ℕ) (hn : 0 < n) (hσ : ∑ d in divisors n, d = 2 * n + 1) : ∃ k : ℕ, n = k * k ∧ odd n :=
by
  sorry

end odd_perfect_square_of_sigma_eq_2n_add_1_l611_611879


namespace part1_part2_l611_611734

namespace ProofProblems

-- Problem Part (1)
def vector_a : ℝ × ℝ := (1, 0)
def vector_b : ℝ × ℝ := (-1, 2)
def vector_diff : ℝ × ℝ := (vector_a.1 - vector_b.1, vector_a.2 - vector_b.2)

def is_unit_vector (v : ℝ × ℝ) : Prop := v.1^2 + v.2^2 = 1
def is_parallel (v1 v2 : ℝ × ℝ) : Prop := ∃ (k : ℝ), v1 = (k * v2.1, k * v2.2)

theorem part1 (c : ℝ × ℝ) (h1 : is_unit_vector c) (h2 : is_parallel c vector_diff) :
  c = (real.sqrt 2 / 2, -real.sqrt 2 / 2) ∨ c = (-real.sqrt 2 / 2, real.sqrt 2 / 2) :=
sorry

-- Problem Part (2)
def perpendicular (v1 v2 : ℝ × ℝ) : Prop := v1.1 * v2.1 + v1.2 * v2.2 = 0

def vector_expr1 (t : ℝ) : ℝ × ℝ := (2 * t + 1, -2)
def vector_expr2 (t : ℝ) : ℝ × ℝ := (3 - t, 2 * t)

theorem part2 (t : ℝ) (h : perpendicular (vector_expr1 t) (vector_expr2 t)) :
  t = -1 ∨ t = 3 / 2 :=
sorry

end ProofProblems

end part1_part2_l611_611734


namespace fishers_tomorrow_l611_611808

-- Define the groups of fishers and their fishing pattern
def everyday_fishers := 7
def every_other_day_fishers := 8
def every_three_days_fishers := 3

-- Given counts for yesterday and today
def fishers_yesterday := 12
def fishers_today := 10

-- The problem to prove: 15 people will fish tomorrow
theorem fishers_tomorrow : 
  (fishers_yesterday = 12) →
  (fishers_today = 10) →
  (everyday_fishers = 7) →
  (every_other_day_fishers = 8) →
  (every_three_days_fishers = 3) →
  15 = everyday_fishers + every_other_day_fishers / 2 + every_three_days_fishers / 3 :=
begin
  sorry
end

end fishers_tomorrow_l611_611808


namespace zoo_ticket_problem_l611_611956

def students_6A (total_cost_6A : ℕ) (saved_tickets_6A : ℕ) (ticket_price : ℕ) : ℕ :=
  let paid_tickets := (total_cost_6A / ticket_price)
  (paid_tickets + saved_tickets_6A)

def students_6B (total_cost_6B : ℕ) (total_students_6A : ℕ) (ticket_price : ℕ) : ℕ :=
  let paid_tickets := (total_cost_6B / ticket_price)
  let total_students := paid_tickets + (paid_tickets / 4)
  (total_students - total_students_6A)

theorem zoo_ticket_problem :
  (students_6A 1995 4 105 = 23) ∧
  (students_6B 4410 23 105 = 29) :=
by {
  -- The proof will follow the steps to confirm the calculations and final result
  sorry
}

end zoo_ticket_problem_l611_611956


namespace boat_fuel_cost_per_hour_l611_611483

variable (earnings_per_photo : ℕ)
variable (shark_frequency_minutes : ℕ)
variable (hunting_hours : ℕ)
variable (expected_profit : ℕ)

def cost_of_fuel_per_hour (earnings_per_photo shark_frequency_minutes hunting_hours expected_profit : ℕ) : ℕ :=
  sorry

theorem boat_fuel_cost_per_hour
  (h₁ : earnings_per_photo = 15)
  (h₂ : shark_frequency_minutes = 10)
  (h₃ : hunting_hours = 5)
  (h₄ : expected_profit = 200) :
  cost_of_fuel_per_hour earnings_per_photo shark_frequency_minutes hunting_hours expected_profit = 50 :=
  sorry

end boat_fuel_cost_per_hour_l611_611483


namespace arithmetic_progression_y_value_l611_611518

theorem arithmetic_progression_y_value (x y : ℚ) 
  (h1 : x = 2)
  (h2 : 2 * y - x = (y + x + 3) - (2 * y - x))
  (h3 : (3 * y + x) - (y + x + 3) = (y + x + 3) - (2 * y - x)) : 
  y = 10 / 3 :=
by
  sorry

end arithmetic_progression_y_value_l611_611518


namespace triangle_side_length_l611_611550

theorem triangle_side_length 
  (AB BC : ℝ) 
  (h1 : AB = 9) 
  (h2 : BC = 4) 
  (similar : ∀ hA hB hC : ℝ, (hA = (2 * S / AB)) ∧ (hB = (2 * S / BC)) ∧ (hC = (2 * S / AC)) → 
                            (triangle ABC similar to triangle formed by altitudes))
  
  : (AC = 6) :=
sorry

end triangle_side_length_l611_611550


namespace exists_perpendicular_line_in_plane_l611_611312

variable {Point : Type}
variable [EuclideanGeometry Point]

-- Define line l and plane α
variable (l : Line Point)
variable (α : Plane Point)

-- Prove that there exists a line m in plane α such that m is perpendicular to l
theorem exists_perpendicular_line_in_plane (l : Line Point) (α : Plane Point) : 
  ∃ m : Line Point, m ∈ α ∧ m ⟂ l :=
by
  sorry

end exists_perpendicular_line_in_plane_l611_611312


namespace reasoning_type_l611_611226

def AllMetalsConductElectricity : Prop := ∀ (m : Type) [Metal m], ConductsElectricity m
def IronIsMetal : Iron : Type → Metal Iron

theorem reasoning_type : (AllMetalsConductElectricity ∧ IronIsMetal) → DeductiveReasoning := 
by
  intro conditions
  have h1 : ∀ (m : Type) [Metal m], ConductsElectricity m := conditions.left
  have h2 : Metal Iron := conditions.right
  exact DeductiveReasoning

end reasoning_type_l611_611226


namespace eccentricity_of_ellipse_l611_611261

theorem eccentricity_of_ellipse (a c e : ℝ) 
  (h1 : c = (sqrt 3 / 2) * a) 
  (h2 : e = c / a) :
  e = sqrt 3 / 2 := 
sorry

end eccentricity_of_ellipse_l611_611261


namespace number_of_special_square_numbers_l611_611845

-- Define a predicate for checking whether a given number is a "special square number"
def isSpecialSquareNumber (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 1000000 ∧ -- Six-digit number
  (∀ d ∈ List.ofDigits n.to_digits 10, d ≠ 0) ∧ -- None of its digits are zero
  ∃ k : ℕ, k * k = n ∧ -- It is a perfect square
  (let d := n.to_digits 10 in 
     ∀ i ∈ [0, 2, 4], 
     let m := 10 * d.get_or_else i 0 + d.get_or_else (i+1) 0 in
       ∃ k : ℕ, k * k = m) -- The first two, middle two, and last two digits are perfect squares

-- Theorem statement: There exist exactly 2 such special square numbers
theorem number_of_special_square_numbers : 
  (Finset.filter isSpecialSquareNumber (Finset.range 1000000)).card = 2 :=
sorry

end number_of_special_square_numbers_l611_611845


namespace fishing_tomorrow_l611_611777

theorem fishing_tomorrow 
  (P1 : ∀ day : ℕ, 7 ∈ {p | p goes fishing on day})
  (P2 : ∀ day : ℕ, day % 2 = 0 → 8 ∈ {p | p goes fishing on day})
  (P3 : ∀ day : ℕ, day % 3 = 0 → 3 ∈ {p | p goes fishing on day})
  (P4 : ℕ)
  (yesterday : ℕ)
  (today : ℕ)
  (tomorrow : ℕ)
  (hyesterday : yesterday = 12)
  (htoday : today = 10)
  : tomorrow = 15 := by
  sorry

end fishing_tomorrow_l611_611777


namespace fishing_tomorrow_l611_611776

theorem fishing_tomorrow 
  (P1 : ∀ day : ℕ, 7 ∈ {p | p goes fishing on day})
  (P2 : ∀ day : ℕ, day % 2 = 0 → 8 ∈ {p | p goes fishing on day})
  (P3 : ∀ day : ℕ, day % 3 = 0 → 3 ∈ {p | p goes fishing on day})
  (P4 : ℕ)
  (yesterday : ℕ)
  (today : ℕ)
  (tomorrow : ℕ)
  (hyesterday : yesterday = 12)
  (htoday : today = 10)
  : tomorrow = 15 := by
  sorry

end fishing_tomorrow_l611_611776


namespace oranges_min_count_l611_611140

theorem oranges_min_count (n : ℕ) (m : ℕ → ℝ) 
  (h : ∀ i j k, m i + m j + m k < 0.05 * ∑ l in (finset.univ \ {i, j, k}), m l) : 
  64 ≤ n :=
sorry

end oranges_min_count_l611_611140


namespace checkerboard_tiling_impossible_l611_611644

theorem checkerboard_tiling_impossible :
  ¬(∃ tiling : (∑ (i j : Fin 10), bool), 
    ∀ i j n, (tiling i j = tt → i < 10 ∧ j < 10) ∧
              (tiling i j = tt → ∃ k, n = 4 * k) ∧
              (∀ p q m n, tiling p q = tt ∧ tiling m n = tt → p ≠ m ∨ q ≠ n)) :=
sorry

end checkerboard_tiling_impossible_l611_611644


namespace solve_for_x_l611_611011

theorem solve_for_x (x y : ℝ) (h1 : 2 * x - 3 * y = 18) (h2 : x + 2 * y = 8) : x = 60 / 7 := sorry

end solve_for_x_l611_611011


namespace inradius_inequality_part1_inradius_inequality_part2_l611_611045

-- Part 1: Prove relation between inradii of similar triangles given the conditions.
theorem inradius_inequality_part1 (AB CD AP BC Q r1 r2 r3: ℝ) 
  (hSquare: AB = CD) 
  (hSimilar: similar_triangles_general ABQ PDA PCQ)
  (hInradii: r1 = inradius(ABQ) ∧ r2 = inradius(PAD) ∧ r3 = inradius(PCQ)) 
  (hAPQ: extend(AP) ∩ extend(BC) = Q) :
  r1^2 ≥ 4 * r2 * r3 := 
sorry

-- Part 2: Prove the inequality involving inradii sums and ensure side length condition is considered
theorem inradius_inequality_part2 (AB CD AP BC Q r1 r2 r3: ℝ) 
  (hSquare: AB = CD) 
  (hSimilar: similar_triangles_general ABQ PDA PCQ)
  (hInradii: r1 = inradius(ABQ) ∧ r2 = inradius(PAD) ∧ r3 = inradius(PCQ))
  (hAPQ: extend(AP) ∩ extend(BC) = Q) 
  (hAB1: AB = 1) : 
  3 - 2 * Real.sqrt 2 < r1^2 + r2^2 + r3^2 ∧ r1^2 + r2^2 + r3^2 < 1/2 :=
sorry

end inradius_inequality_part1_inradius_inequality_part2_l611_611045


namespace quadratic_function_m_eq_neg_1_l611_611027

noncomputable def y (m x : ℝ) := (m - 1) * x ^ (m ^ 2 + 1) + 3 * x

theorem quadratic_function_m_eq_neg_1 (m : ℝ) (h : ∀ x : ℝ, y m x = (m - 1) * x ^ (m ^ 2 + 1) + 3 * x) :
  (m^2 + 1 = 2) → (m - 1 ≠ 0) → m = -1 :=
by
  intros h1 h2
  have : m^2 = 1 := by linarith
  cases eq_or_neg_eq.1 (eq_of_square_eq_self _ this)
  · contradiction
  · assumption

end quadratic_function_m_eq_neg_1_l611_611027


namespace range_of_m_l611_611008

theorem range_of_m (x : ℝ) (m : ℝ) (H : ∃ a b c ∈ Ioo (π/6) (5*π/6), 
    (sin x * -sin x) + ((m + 1) * sin x) = 0) : 
    1/2 < m ∧ m < 1 :=
by
  sorry

end range_of_m_l611_611008


namespace A_intersect_B_l611_611535

def A : Set ℝ := { x | abs x < 2 }
def B : Set ℝ := { x | x^2 - 5 * x - 6 < 0 }

theorem A_intersect_B : A ∩ B = { x | -1 < x ∧ x < 2 } := by
  sorry

end A_intersect_B_l611_611535


namespace coefficients_of_quadratic_function_l611_611930

-- Define the quadratic function.
def quadratic_function (x : ℝ) : ℝ :=
  2 * (x - 3) ^ 2 + 2

-- Define the expected expanded form.
def expanded_form (x : ℝ) : ℝ :=
  2 * x ^ 2 - 12 * x + 20

-- State the proof problem.
theorem coefficients_of_quadratic_function :
  ∀ (x : ℝ), quadratic_function x = expanded_form x := by
  sorry

end coefficients_of_quadratic_function_l611_611930


namespace original_rope_length_l611_611621

-- Define the conditions
noncomputable def π : ℝ := Real.pi
def additional_area : ℝ := 565.7142857142857
def new_length : ℝ := 18

-- The given condition in the problem
def grazed_area_difference (r : ℝ) : ℝ := π * (new_length^2) - π * (r^2)

-- The main theorem we need to prove
theorem original_rope_length (r : ℝ) (h : grazed_area_difference r = additional_area) : r = 12 := by
  sorry

end original_rope_length_l611_611621


namespace oranges_min_count_l611_611139

theorem oranges_min_count (n : ℕ) (m : ℕ → ℝ) 
  (h : ∀ i j k, m i + m j + m k < 0.05 * ∑ l in (finset.univ \ {i, j, k}), m l) : 
  64 ≤ n :=
sorry

end oranges_min_count_l611_611139


namespace correct_statements_l611_611216

-- Define the statements
def StatementA : Prop := ∀ (a b c : ℝ), a + b + c = 180 ∧ a < 90 ∧ b < 90 → a < 90 ∨ b < 90 ∨ c < 90
def StatementB : Prop := ∀ (r : ℝ) (m : ℝ), 0 < r ∧ 0 < m → 
  ∃ (x y : ℝ), ((x - r)^2 + (y - r)^2 = r^2) ∧ ((x - r)^2 + y^2 = m^2) ∧ ((x + r)^2 + (y - r)^2 = r^2) → x = y
def StatementC : Prop := ∀ (α β : ℝ), α + β = 90 → (180 - α) + (180 - β) = 90
def StatementD : Prop := ∀ (l₁ l₂ m : Line), intersects m l₁ ∧ intersects m l₂ → ∃ θ₁ θ₂, corresponding l₁ m θ₁ ∧ corresponding l₂ m θ₂ ∧ θ₁ = θ₂

-- Theorem to prove Statements A and B are correct
theorem correct_statements : StatementA ∧ StatementB := by
  split
  sorry
  sorry

end correct_statements_l611_611216


namespace correct_statement_l611_611215

variable (P Q : Prop)
variable (hP : P)
variable (hQ : Q)

theorem correct_statement :
  (P ∧ Q) :=
by
  exact ⟨hP, hQ⟩

end correct_statement_l611_611215


namespace modulus_of_complex_number_l611_611327

noncomputable def z := Complex

theorem modulus_of_complex_number (z : Complex) (h : z * (1 + Complex.I) = 2) :
  Complex.abs z = Real.sqrt 2 :=
sorry

end modulus_of_complex_number_l611_611327


namespace problem_1_problem_2_l611_611002

open Set

variable {x : ℝ}

def A : Set ℝ := { x | x - 2 ≥ 0 }
def B : Set ℝ := { x | x < 5 }

theorem problem_1 :
  A ∪ B = univ :=
by sorry

theorem problem_2 :
  (complement A) ∩ B = { x | x < 2 } :=
by sorry

end problem_1_problem_2_l611_611002


namespace average_speed_of_train_l611_611984

-- Given conditions
def distance1 : ℝ := 240
def time1 : ℝ := 3
def distance2 : ℝ := 450
def time2 : ℝ := 5

-- Total distance and time
def total_distance : ℝ := distance1 + distance2
def total_time : ℝ := time1 + time2

-- Average speed
def average_speed : ℝ := total_distance / total_time

-- Proof statement
theorem average_speed_of_train :
  average_speed = 86.25 := 
by 
  -- This sorry acts as a placeholder for the actual proof
  sorry

end average_speed_of_train_l611_611984


namespace perimeter_triangle_CEF_l611_611899

theorem perimeter_triangle_CEF
  (A B C D E F : Point)
  (s : ℝ)
  (h_square : square A B C D s)
  (s_eq_one : s = 1)
  (h_E_on_BC : on_segment E B C)
  (h_F_on_CD : on_segment F C D)
  (h_angle_EAF : ∠EAF = 45) :
  perimeter_triangle C E F = 2 :=
by sorry

end perimeter_triangle_CEF_l611_611899


namespace max_underwear_pairs_l611_611197

-- Define the weights of different clothing items
def weight_socks : ℕ := 2
def weight_underwear : ℕ := 4
def weight_shirt : ℕ := 5
def weight_shorts : ℕ := 8
def weight_pants : ℕ := 10

-- Define the washing machine limit
def max_weight : ℕ := 50

-- Define the current load of clothes Tony plans to wash
def current_load : ℕ :=
  1 * weight_pants +
  2 * weight_shirt +
  1 * weight_shorts +
  3 * weight_socks

-- State the theorem regarding the maximum number of additional pairs of underwear
theorem max_underwear_pairs : 
  current_load ≤ max_weight →
  (max_weight - current_load) / weight_underwear = 4 :=
by
  sorry

end max_underwear_pairs_l611_611197


namespace parallel_lines_distance_l611_611409

theorem parallel_lines_distance (m n : ℝ) (h1 : m > 0) (h2 : 2 = -2 / n)
    (h3 : (|m + 3| / sqrt (1 + 4)) = sqrt 5) : m + n = -2 :=
by
  sorry

end parallel_lines_distance_l611_611409


namespace infinite_integer_triples_solution_l611_611916

theorem infinite_integer_triples_solution (a b c : ℤ) : 
  ∃ (a b c : ℤ), ∀ n : ℤ, a^2 + b^2 = c^2 + 3 :=
sorry

end infinite_integer_triples_solution_l611_611916


namespace outer_radius_correct_l611_611162

noncomputable def outerCircleRadius (C_inner : ℕ) (w : ℕ) : ℝ :=
  let r_inner := C_inner / (2 * Real.pi)
  r_inner + w

theorem outer_radius_correct : outerCircleRadius 880 18 ≈ 158.01 :=
by
  -- Add proof here
  sorry

end outer_radius_correct_l611_611162


namespace find_x_l611_611693

-- Definitions based on conditions
def parabola_eq (y x p : ℝ) : Prop := y^2 = 2 * p * x
def point_on_parabola (p : ℝ) : Prop := ∃ y x, parabola_eq y x p ∧ (x = 1) ∧ (y = 2)
def valid_p (p : ℝ) : Prop := p > 0
def dist_to_focus (x : ℝ) : ℝ := 1
def dist_to_line (x : ℝ) : ℝ := abs (x + 1)

-- Main statement to be proven
theorem find_x (p : ℝ) (h1 : point_on_parabola p) (h2 : valid_p p) :
  ∃ x, dist_to_focus x = dist_to_line x ∧ x = 1 :=
sorry

end find_x_l611_611693


namespace number_of_cows_l611_611425

-- Define conditions
def total_bags_consumed_by_some_cows := 45
def bags_consumed_by_one_cow := 1

-- State the theorem to prove the number of cows
theorem number_of_cows (h1 : total_bags_consumed_by_some_cows = 45) (h2 : bags_consumed_by_one_cow = 1) : 
  total_bags_consumed_by_some_cows / bags_consumed_by_one_cow = 45 :=
by
  -- Proof goes here
  sorry

end number_of_cows_l611_611425


namespace ribbon_per_gift_l611_611863

-- Definitions for the conditions in the problem
def total_ribbon_used : ℚ := 4/15
def num_gifts: ℕ := 5

-- Statement to prove
theorem ribbon_per_gift : total_ribbon_used / num_gifts = 4 / 75 :=
by
  sorry

end ribbon_per_gift_l611_611863


namespace general_formula_l611_611414

def sum_of_terms (a : ℕ → ℕ) (n : ℕ) : ℕ := 3 / 2 * a n - 3

def sequence_term (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  if n = 0 then 6 
  else a (n - 1) * 3

theorem general_formula (a : ℕ → ℕ) (n : ℕ) :
  (∀ n, sum_of_terms a n = 3 / 2 * a n - 3) →
  (∀ n, n = 0 → a n = 6) →
  (∀ n, n > 0 → a n = a (n - 1) * 3) →
  a n = 2 * 3^n := by
  sorry

end general_formula_l611_611414


namespace min_oranges_picked_l611_611125

noncomputable def minimum_oranges (n : ℕ) : Prop :=
  ∀ (m : ℕ → ℕ), (∀ i j k : ℕ, i ≠ j ∧ j ≠ k ∧ i ≠ k → (m i + m j + m k : ℝ) / (∑ l : ℕ in finset.univ \ {i, j, k}, m l) < 0.05) → n ≥ 64

theorem min_oranges_picked : ∃ n, minimum_oranges n := by
  use 64
  sorry

end min_oranges_picked_l611_611125


namespace quadrilateral_area_l611_611849

theorem quadrilateral_area {AB BC : ℝ} (hAB : AB = 4) (hBC : BC = 8) :
  ∃ area : ℝ, area = 16 := by
  sorry

end quadrilateral_area_l611_611849


namespace f_value_at_5pi_over_12_l611_611370

noncomputable def f (x : ℝ) : ℝ := (real.sqrt 3) * real.sin x - real.cos x

theorem f_value_at_5pi_over_12 : f (5 * real.pi / 12) = real.sqrt 2 :=
by 
  -- This is the point where the proof would go
  sorry

end f_value_at_5pi_over_12_l611_611370


namespace shape_is_cone_l611_611308

def shape_of_phi_const_in_spherical (ρ θ φ c : ℝ) (h_c : 0 ≤ c ∧ c ≤ π) : Prop :=
  φ = c → ∃ (ρ θ : ℝ), 0 ≤ ρ ∧ 0 ≤ θ ∧ θ ≤ 2 * π ∧ (ρ, θ, c) formsCone

-- A function to define when the points in spherical coordinates form a cone:
def formsCone (ρ θ phi : ℝ) : Prop := sorry

theorem shape_is_cone (c : ℝ) (h_c : 0 ≤ c ∧ c ≤ π) :
  shape_of_phi_const_in_spherical ρ θ φ c h_c = true :=
sorry

end shape_is_cone_l611_611308


namespace hypotenuse_length_l611_611562

-- Definition of the right triangle with the given leg lengths
structure RightTriangle :=
  (BC AC AB : ℕ)
  (right : BC^2 + AC^2 = AB^2)

-- The theorem we need to prove
theorem hypotenuse_length (T : RightTriangle) (h1 : T.BC = 5) (h2 : T.AC = 12) :
  T.AB = 13 :=
by
  sorry

end hypotenuse_length_l611_611562


namespace min_oranges_picked_l611_611122

noncomputable def minimum_oranges (n : ℕ) : Prop :=
  ∀ (m : ℕ → ℕ), (∀ i j k : ℕ, i ≠ j ∧ j ≠ k ∧ i ≠ k → (m i + m j + m k : ℝ) / (∑ l : ℕ in finset.univ \ {i, j, k}, m l) < 0.05) → n ≥ 64

theorem min_oranges_picked : ∃ n, minimum_oranges n := by
  use 64
  sorry

end min_oranges_picked_l611_611122


namespace fishers_tomorrow_l611_611792

-- Definitions based on conditions
def people_every_day : ℕ := 7
def people_every_other_day : ℕ := 8
def people_every_three_days : ℕ := 3
def people_yesterday : ℕ := 12
def people_today : ℕ := 10

-- Theorem to be proved
theorem fishers_tomorrow (people_every_day people_every_other_day people_every_three_days people_yesterday people_today : ℕ) : 
  people_every_day = 7 ∧ 
  people_every_other_day = 8 ∧ 
  people_every_three_days = 3 ∧
  people_yesterday = 12 ∧ 
  people_today = 10 →
  15 := 
by {
  sorry
}

end fishers_tomorrow_l611_611792


namespace race_winner_and_liar_l611_611682

def Alyosha_statement (pos : ℕ → Prop) : Prop := ¬ pos 1 ∧ ¬ pos 4
def Borya_statement (pos : ℕ → Prop) : Prop := ¬ pos 4
def Vanya_statement (pos : ℕ → Prop) : Prop := pos 1
def Grisha_statement (pos : ℕ → Prop) : Prop := pos 4

def three_true_one_false (s1 s2 s3 s4 : Prop) : Prop := 
  (s1 ∧ s2 ∧ s3 ∧ ¬ s4) ∨
  (s1 ∧ s2 ∧ ¬ s3 ∧ s4) ∨
  (s1 ∧ ¬ s2 ∧ s3 ∧ s4) ∨
  (¬ s1 ∧ s2 ∧ s3 ∧ s4)

def race_result (pos : ℕ → Prop) : Prop :=
  Vanya_statement pos ∧
  three_true_one_false (Alyosha_statement pos) (Borya_statement pos) (Vanya_statement pos) (Grisha_statement pos) ∧
  Borya_statement pos = false

theorem race_winner_and_liar:
  ∃ (pos : ℕ → Prop), race_result pos :=
sorry

end race_winner_and_liar_l611_611682


namespace remaining_bag_weight_l611_611159

theorem remaining_bag_weight :
  ∃ (w1 w2 w_remaining : ℝ),
  let
    weights := [15, 16, 18, 19, 20, 31],
    total_weight := 119
  in
    w1 = 2 * w2 ∧
    w1 + w2 = total_weight - w_remaining ∧
    ∀ (bag : ℝ), bag ∈ weights → w1 + w2 + bag = total_weight →
                 w_remaining = bag → w_remaining = 20 :=
sorry

end remaining_bag_weight_l611_611159


namespace equilateral_triangle_dot_product_sum_l611_611037

theorem equilateral_triangle_dot_product_sum (a b c : ℝ³) (ha : a.norm = 1) (hb : b.norm = 1) (hc : c.norm = 1) 
    (angle_ab : angle_between a b = 2 * Real.pi / 3) 
    (angle_bc : angle_between b c = 2 * Real.pi / 3) 
    (angle_ca : angle_between c a = 2 * Real.pi / 3) :
  a • b + b • c + c • a = -3 / 2 := 
sorry

end equilateral_triangle_dot_product_sum_l611_611037


namespace measure_of_angle_F_l611_611440

-- Define measures of the angles
def D : ℝ := 75
def F : ℝ := 30
def E : ℝ := 2 * F + 15

-- Angle sum property for triangle
def angle_sum_property : Prop := D + E + F = 180

theorem measure_of_angle_F : F = 30 :=
by
  have h_D : D = 75 := rfl
  have h_F : F = 30 := rfl
  have h_E : E = 2 * F + 15 := rfl
  have h_sum : angle_sum_property := by
    rw [h_D, h_E, h_F]
    norm_num
  exact h_F

end measure_of_angle_F_l611_611440


namespace Rick_received_amount_l611_611492

theorem Rick_received_amount :
  let total_promised := 400
  let sally_owes := 35
  let amy_owes := 30
  let derek_owes := amy_owes / 2
  let carl_owes := 35
  let total_owed := sally_owes + amy_owes + derek_owes + carl_owes
  total_promised - total_owed = 285 :=
by
  sorry

end Rick_received_amount_l611_611492


namespace sum_tens_units_digit_l611_611977

theorem sum_tens_units_digit (n : ℕ) (hne : n = 2004) : 
  let decimal_repr := 9^n in
  let tens_digit := (decimal_repr / 10) % 10 in
  let units_digit := decimal_repr % 10 in
  tens_digit + units_digit = 7 := sorry

end sum_tens_units_digit_l611_611977


namespace product_is_approximately_9603_l611_611533

noncomputable def smaller_number : ℝ := 97.49871794028884
noncomputable def successive_number : ℝ := smaller_number + 1
noncomputable def product_of_numbers : ℝ := smaller_number * successive_number

theorem product_is_approximately_9603 : abs (product_of_numbers - 9603) < 10e-3 := 
sorry

end product_is_approximately_9603_l611_611533


namespace segment_DF_bisects_EG_l611_611449

-- Define the points and assumptions
variable (A B C E F G D : Type*)
variable [is_triangle A B C]
variable [BC_lt_AB : B.toPoint.dist(C.toPoint) < A.toPoint.dist(B.toPoint)]
variable [BE_angle_bisector : BE.bisects (angle B A C)]
variable [line_l : is_line (through C) ∧ perpendicular line_l BE]
variable [l_intersects_BE_at_F : F ∈ line_l ∧ F ∈ BE]
variable [l_intersects_BD_at_G : G ∈ line_l ∧ G ∈ BD]
variable [D_midpoint_AC : D ∈ midpoint A C]

-- Desired conclusion: DF bisects EG
theorem segment_DF_bisects_EG : bisects_segment DF EG := 
sorry

end segment_DF_bisects_EG_l611_611449


namespace ratio_AE_EC_eq_half_l611_611769

noncomputable def ratio_AE_EC (A B C D E : Point)
  (h1 : AB = 8)
  (h2 : BC = 10)
  (h3 : AC = 12)
  (h4 : on (D, AC))
  (h5 : BD = 8)
  (h6 : on (E, AB))
  (h7 : BE = AE) : Real :=
  AE / EC

theorem ratio_AE_EC_eq_half
  (A B C D E : Point)
  (h1 : AB = 8)
  (h2 : BC = 10)
  (h3 : AC = 12)
  (h4 : on (D, AC))
  (h5 : BD = 8)
  (h6 : on (E, AB))
  (h7 : BE = AE) : 
  ratio_AE_EC A B C D E h1 h2 h3 h4 h5 h6 h7 = 1 / 2 :=
sorry

end ratio_AE_EC_eq_half_l611_611769


namespace largest_angle_in_ratio_triangle_l611_611165

theorem largest_angle_in_ratio_triangle (x : ℝ) (h1 : 3 * x + 4 * x + 5 * x = 180) : 
  5 * (180 / (3 + 4 + 5)) = 75 := by
  sorry

end largest_angle_in_ratio_triangle_l611_611165


namespace johns_total_payment_l611_611446

theorem johns_total_payment
  (cost_candy_bar : ℝ)
  (cost_gum_per_stick : ℝ)
  (cost_bag_chips : ℝ)
  (total_gum_cost : ℝ)
  (total_candy_cost : ℝ)
  (total_chips_cost: ℝ)
  (discount_total_purchase: ℝ)
  (discount_candy : ℝ)
  (total_discount: ℝ)
  (discounted_total : ℝ)
  (sales_tax: ℝ)
  (final_amount : ℝ) :
  cost_candy_bar = 1.50 →
  cost_gum_per_stick = cost_candy_bar / 2 →
  cost_bag_chips = cost_candy_bar * 2 →
  total_gum_cost = 5 * cost_gum_per_stick →
  total_candy_cost = 4 * cost_candy_bar →
  total_chips_cost = 2 * cost_bag_chips →
  let total_cost_before_discounts := total_gum_cost + total_candy_cost + total_chips_cost in
  discount_total_purchase = total_cost_before_discounts * 0.10 →
  discount_candy = total_candy_cost * 0.075 →
  total_discount = discount_total_purchase + discount_candy →
  discounted_total = total_cost_before_discounts - total_discount →
  sales_tax = discounted_total * 0.05 →
  final_amount = discounted_total + sales_tax →
  final_amount = 14.41 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12
  sorry

end johns_total_payment_l611_611446


namespace value_division_l611_611415

theorem value_division (x y : ℝ) (h1 : y ≠ 0) (h2 : 2 * x - y = 1.75 * x) 
                       (h3 : x / y = n) : n = 4 := 
by 
sorry

end value_division_l611_611415


namespace sum_of_roots_l611_611532

theorem sum_of_roots (p : ℝ) (h : (4 - p) / 2 = 9) : (p / 2 = 7) :=
by 
  sorry

end sum_of_roots_l611_611532


namespace polynomial_roots_equation_l611_611470

theorem polynomial_roots_equation (a b c d : ℂ)
  (h₀ : (polynomial.X ^ 4 - 15 * polynomial.X ^ 3 + 35 * polynomial.X ^ 2 - 21 * polynomial.X + 6).roots = {a, b, c, d}) :
  (a - b) ^ 2 + (b - c) ^ 2 + (c - d) ^ 2 + (d - a) ^ 2 = 336 :=
sorry

end polynomial_roots_equation_l611_611470


namespace trailing_zeros_50_factorial_l611_611638

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def count_factors (n : ℕ) (p : ℕ) : ℕ :=
  if n = 0 then 0 else (n / p) + count_factors (n / p) p

theorem trailing_zeros_50_factorial : count_factors 50 5 = 12 := 
sorry

end trailing_zeros_50_factorial_l611_611638


namespace complement_M_in_U_l611_611474

open Finset

-- Definitions of the universal set and subset
def U := {1, 2, 3, 4, 5, 6} : Finset ℕ
def M := {1, 2, 4} : Finset ℕ

-- The statement that needs to be proved
theorem complement_M_in_U : U \ M = {3, 5, 6} := by
  sorry

end complement_M_in_U_l611_611474


namespace min_intersection_cardinality_l611_611866

variables {A B C : Type}
def number_of_subsets (S : Type) := 2 ^ (fintype.card S)

theorem min_intersection_cardinality 
  (h1 : fintype.card A = 120) 
  (h2 : fintype.card B = 120) 
  (h3 : fintype.card C = 130) 
  (h4 : number_of_subsets A + number_of_subsets B + number_of_subsets C = number_of_subsets (A ⊔ B ⊔ C)) : 
  ∃ n, n = 110 ∧ fintype.card (A ∩ B ∩ C) = n :=
by sorry

end min_intersection_cardinality_l611_611866


namespace inclination_of_x_eq_1_l611_611352

theorem inclination_of_x_eq_1 : (let α := 90 in α = 90) :=
by
  let α := 90
  exact rfl

end inclination_of_x_eq_1_l611_611352


namespace triangle_ABC_inequality_l611_611992

variable (A B C A₁ B₁ C₁ : Type)
variable [metric_space A] [metric_space B] [metric_space C]
variables [metric_space A₁] [metric_space B₁] [metric_space C₁]
variables [has_distance A] [has_distance B] [has_distance C]
variables [has_distance A₁] [has_distance B₁] [has_distance C₁]

noncomputable def length (x y : Type) [metric_space x] [metric_space y] [has_distance x] [has_distance y] : ℝ := 
dist x y

variables (BC CA AB : ℝ)
variables (P P₁ : ℝ)
variable (λ : ℝ)
hypothesis h1 : length B A₁ = λ * BC
hypothesis h2 : length C B₁ = λ * CA
hypothesis h3 : length A C₁ = λ * AB
hypothesis h4 : 1/2 < λ ∧ λ < 1

theorem triangle_ABC_inequality :
    (2 * λ - 1) * (AB + BC + CA) < (length A₁ B₁ + length B₁ C₁ + length C₁ A₁) ∧ 
    (length A₁ B₁ + length B₁ C₁ + length C₁ A₁) < λ * (AB + BC + CA) := 
    sorry

end triangle_ABC_inequality_l611_611992


namespace contains_k_disjoint_cycles_l611_611074

-- Define appropriate terms required for the proof.

-- k is a natural number.
variable (k : ℕ)

-- Define H as a cubic multigraph.
structure CubicMultigraph where
  vertices : Type
  edges : vertices → vertices → Prop
  cubic : ∀ v, (∑ w, edges v w) = 3

-- Define the size of a graph.
def size {V} (H : CubicMultigraph) : ℕ :=
  nat.card H.vertices

-- Define s_k based on the problem context (assuming s : ℕ → ℕ as some function).
constant s : ℕ → ℕ

-- The main theorem statement.
theorem contains_k_disjoint_cycles (k : ℕ) (H : CubicMultigraph) (hk : size H ≥ s k) :
  ∃ (C : list (CubicMultigraph)), (list.length C = k ∧ disjoint_cycles C) :=
    sorry

-- Define property for disjoint cycles (assuming disjointness means vertex disjoint).
def disjoint_cycles (C : list (CubicMultigraph)) : Prop :=
  ∀ (C1 C2 ∈ C), C1 ≠ C2 → ∀ v, v ∈ C1.vertices → v ∉ C2.vertices

-- Explanation for required properties
-- - ∃ (C : list (CubicMultigraph)), represents the existence of k disjoint cycles
-- - list.length C = k represents the k disjoint cycles requirement
-- - disjoint_cycles C represents the property that these cycles are disjoint

end contains_k_disjoint_cycles_l611_611074


namespace inverse_function_point_l611_611691

-- Definitions and conditions
def f (x : ℝ) : ℝ := sorry  -- Placeholder for the function f
axiom f_0_1 : f 0 = 1

-- Target proof statement
theorem inverse_function_point :
  ∃ g : ℝ → ℝ, (∀ x, g (f x) = x) → (1, 4) ∈ (λ y, (4 - f⁻¹ y))⁻¹ :=
sorry

end inverse_function_point_l611_611691


namespace replaced_person_is_65_l611_611513

-- Define the conditions of the problem context
variable (W : ℝ)
variable (avg_increase : ℝ := 3.5)
variable (num_persons : ℕ := 8)
variable (new_person_weight : ℝ := 93)

-- Express the given condition in the problem: 
-- The total increase in weight is given by the number of persons multiplied by the average increase in weight
def total_increase : ℝ := num_persons * avg_increase

-- Express the relationship between the new person's weight and the person who was replaced
def replaced_person_weight (W : ℝ) : ℝ := new_person_weight - total_increase

-- Stating the theorem to be proved
theorem replaced_person_is_65 : replaced_person_weight W = 65 := by
  sorry

end replaced_person_is_65_l611_611513


namespace part1_part2_l611_611733

namespace ProofProblems

-- Problem Part (1)
def vector_a : ℝ × ℝ := (1, 0)
def vector_b : ℝ × ℝ := (-1, 2)
def vector_diff : ℝ × ℝ := (vector_a.1 - vector_b.1, vector_a.2 - vector_b.2)

def is_unit_vector (v : ℝ × ℝ) : Prop := v.1^2 + v.2^2 = 1
def is_parallel (v1 v2 : ℝ × ℝ) : Prop := ∃ (k : ℝ), v1 = (k * v2.1, k * v2.2)

theorem part1 (c : ℝ × ℝ) (h1 : is_unit_vector c) (h2 : is_parallel c vector_diff) :
  c = (real.sqrt 2 / 2, -real.sqrt 2 / 2) ∨ c = (-real.sqrt 2 / 2, real.sqrt 2 / 2) :=
sorry

-- Problem Part (2)
def perpendicular (v1 v2 : ℝ × ℝ) : Prop := v1.1 * v2.1 + v1.2 * v2.2 = 0

def vector_expr1 (t : ℝ) : ℝ × ℝ := (2 * t + 1, -2)
def vector_expr2 (t : ℝ) : ℝ × ℝ := (3 - t, 2 * t)

theorem part2 (t : ℝ) (h : perpendicular (vector_expr1 t) (vector_expr2 t)) :
  t = -1 ∨ t = 3 / 2 :=
sorry

end ProofProblems

end part1_part2_l611_611733


namespace total_amount_divided_l611_611597

theorem total_amount_divided (A B C : ℝ) (h1 : A = (2/3) * (B + C)) (h2 : B = (2/3) * (A + C)) (h3 : A = 200) :
  A + B + C = 500 :=
by
  sorry

end total_amount_divided_l611_611597


namespace sum_remainder_l611_611019

theorem sum_remainder (m : ℤ) : ((9 - m) + (m + 5)) % 8 = 6 :=
by
  sorry

end sum_remainder_l611_611019


namespace temp_at_9pm_is_3_l611_611266

-- Define the times and temperatures as mappings.
def temperatures : List (Nat × ℤ) := [
  (14, 3), -- 2 p.m. (14:00)
  (15, _), -- 3 p.m.
  (16, _), -- 4 p.m.
  (17, _), -- 5 p.m.
  (18, _), -- 6 p.m.
  (19, _), -- 7 p.m.
  (20, _), -- 8 p.m.
  (21, _), -- 9 p.m. (21:00)
  (22, _)]-- 10 p.m. (22:00)

-- Hypothesis: The temperature at 2 p.m. was 3°C.
axiom temp_at_2pm : temperatures.head?.2 = 3

-- Hypothesis: Temperatures were measured every hour from 14:00 to 22:00.
axiom measurements_every_hour (n : Nat) : n ∈ List.range 14 9 → (∃ t, (14 + n, t) ∈ temperatures)

-- The goal is to prove that the temperature at 9 p.m. (21:00) is also 3°C.
theorem temp_at_9pm_is_3 :
  (21, 3) ∈ temperatures :=
sorry

end temp_at_9pm_is_3_l611_611266


namespace oranges_min_number_l611_611110

theorem oranges_min_number (n : ℕ) 
  (h : ∀ (m : ℕ → ℝ), (∀ i j k : ℕ, i < n → j < n → k < n →
    i ≠ j → j ≠ k → i ≠ k → m i + m j + m k < 0.05 * (∑ l in (finset.range n \ {i,j,k}), m l))) : 
  n ≥ 64 :=
sorry

end oranges_min_number_l611_611110


namespace four_digit_number_l611_611542

-- Defining the cards and their holders
def cards : List ℕ := [2, 0, 1, 5]
def A : ℕ := 5
def B : ℕ := 1
def C : ℕ := 2
def D : ℕ := 0

-- Conditions based on statements
def A_statement (a b c d : ℕ) : Prop := 
  ¬ ((b = a + 1) ∨ (b = a - 1) ∨ (c = a + 1) ∨ (c = a - 1) ∨ (d = a + 1) ∨ (d = a - 1))

def B_statement (a b c d : ℕ) : Prop := 
  (b = a + 1) ∨ (b = a - 1) ∨ (c = a + 1) ∨ (c = a - 1) ∨ (d = a + 1) ∨ (d = a - 1)

def C_statement (c : ℕ) : Prop := ¬ (c = 1 ∨ c = 2 ∨ c = 5)
def D_statement (d : ℕ) : Prop := d ≠ 0

-- Truth conditions
def is_even (n : ℕ) : Prop := n % 2 = 0
def is_odd (n : ℕ) : Prop := ¬ is_even n

def tells_truth (n : ℕ) : Prop := is_odd n
def lies (n : ℕ) : Prop := is_even n

-- Proof statement
theorem four_digit_number (a b c d : ℕ) 
  (ha : a ∈ cards) (hb : b ∈ cards) (hc : c ∈ cards) (hd : d ∈ cards) 
  (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (truth_A : tells_truth a → A_statement a b c d)
  (lie_A : lies a → ¬ A_statement a b c d)
  (truth_B : tells_truth b → B_statement a b c d)
  (lie_B : lies b → ¬ B_statement a b c d)
  (truth_C : tells_truth c → C_statement c)
  (lie_C : lies c → ¬ C_statement c)
  (truth_D : tells_truth d → D_statement d)
  (lie_D : lies d → ¬ D_statement d) :
  a * 1000 + b * 100 + c * 10 + d = 5120 := 
  by
    sorry

end four_digit_number_l611_611542


namespace series_sum_correct_l611_611659

open Classical

noncomputable def series_sum : ℝ :=
  ∑' k : ℕ, (2 * (k+1)) / 4^(k+1)

theorem series_sum_correct :
  series_sum = 8 / 9 :=
by
  sorry

end series_sum_correct_l611_611659


namespace gdp_scientific_notation_l611_611198

theorem gdp_scientific_notation :
  ∃ a n : ℝ, 1 ≤ |a| ∧ |a| < 10 ∧ n ∈ ℤ ∧ (7_298_000_000_000 : ℝ) = a * 10 ^ n ∧ a = 7.298 ∧ n = 12 :=
by
  sorry

end gdp_scientific_notation_l611_611198


namespace math_proof_problem_l611_611003

def A : Set ℝ := {x | x^2 - x - 2 ≤ 0}
def B : Set ℝ := {x | 1 < x ∧ x ≤ 3}
def complement_R (s : Set ℝ) := {x : ℝ | x ∉ s}

theorem math_proof_problem :
  (complement_R A ∩ B) = {x | 2 < x ∧ x ≤ 3} :=
sorry

end math_proof_problem_l611_611003


namespace inner_square_area_l611_611151

theorem inner_square_area (a b c d : ℝ) (h1: a = 10) (h2: b = 3) (h3: c = 30) : 
  (let s := (8 / (Real.sqrt 3)) in let area := s^2 in area = 64 / 3) :=
by
  sorry

end inner_square_area_l611_611151


namespace area_EPHQ_l611_611908

variable (EF GH : ℝ)
variable (EP EH FP HQ : ℝ)

-- Given conditions
axiom rect_EFGH : EF = 10 ∧ GH = 5
axiom midpoint_P : FP = GH / 2 ∧ EP = EF
axiom midpoint_Q : HQ = GH / 2 ∧ EH = GH

-- Proof of the area of region EPHQ
theorem area_EPHQ : EP * EH / 2 + FP * HQ / 2 = 25 :=
by
  have EF_value : EF = 10 := rect_EFGH.1
  have GH_value : GH = 5 := rect_EFGH.2
  sorry

end area_EPHQ_l611_611908


namespace geometric_sequence_product_l611_611715

/-- Given a geometric sequence with positive terms where a_3 = 3 and a_6 = 1/9,
    prove that a_4 * a_5 = 1/3. -/
theorem geometric_sequence_product (a : ℕ → ℝ) (q : ℝ) (h_pos : ∀ n, a n > 0)
    (h_geometric : ∀ n, a (n + 1) = a n * q) (ha3 : a 3 = 3) (ha6 : a 6 = 1 / 9) :
  a 4 * a 5 = 1 / 3 := 
by
  sorry

end geometric_sequence_product_l611_611715


namespace quadratic_equation_solution_l611_611025

-- Define the problem statement and the conditions: the equation being quadratic.
theorem quadratic_equation_solution (m : ℤ) :
  (∃ (a : ℤ), a ≠ 0 ∧ (a*x^2 - x - 2 = 0)) →
  m = -1 :=
by
  sorry

end quadratic_equation_solution_l611_611025


namespace part1_part2_l611_611336

open BigOperators

noncomputable def seq (a : ℕ → ℝ) : Prop :=
  (∀ n : ℕ, n ≠ 0 → a n > 0) ∧
  (a 1 = 2) ∧
  (∀ n : ℕ, n ≠ 0 → (n + 1) * (a (n + 1)) ^ 2 = n * (a n) ^ 2 + a n)

theorem part1 (a : ℕ → ℝ) (h : seq a)
  (n : ℕ) (hn : n ≠ 0) 
  : 1 < a (n+1) ∧ a (n+1) < a n :=
sorry

theorem part2 (a : ℕ → ℝ) (h : seq a)
  : ∑ k in Finset.range 2022 \ {0}, (a (k+1))^2 / (k+1)^2 < 2 :=
sorry

end part1_part2_l611_611336


namespace smallest_n_for_pythagorean_triplets_l611_611084

def is_pythagorean_triplet (x y z : ℕ) : Prop :=
  x^2 + y^2 = z^2

def S : set ℕ := {1, 2, 3, ..., 50}

def contains_pythagorean_triplet (subset : set ℕ) : Prop :=
  ∃ x y z ∈ subset, is_pythagorean_triplet x y z

theorem smallest_n_for_pythagorean_triplets :
  ∃ (n : ℕ), (∀ subset ⊆ S, subset.card = n → contains_pythagorean_triplet subset) ∧
              (∀ m < n, ∃ subset ⊆ S, subset.card = m ∧ ¬ contains_pythagorean_triplet subset) :=
sorry

end smallest_n_for_pythagorean_triplets_l611_611084


namespace two_x_minus_four_y_eq_zero_l611_611886

-- Define points A and B
def A : ℝ × ℝ := (20, 10)
def B : ℝ × ℝ := (10, 5)

-- Define the midpoint function
def midpoint (P Q : ℝ × ℝ) : ℝ × ℝ :=
  ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- Define the point C as the midpoint of A and B
def C : ℝ × ℝ := midpoint A B

-- Define the goal statement
theorem two_x_minus_four_y_eq_zero :
  let x := C.1
  let y := C.2
  2 * x - 4 * y = 0 :=
by
  sorry

end two_x_minus_four_y_eq_zero_l611_611886


namespace normal_vector_of_line_l611_611710

open Real

theorem normal_vector_of_line (θ : ℝ) (hθ : θ = π / 3) :
  ∃ (a : ℝ × ℝ), a = (1, - sqrt 3 / 3) :=
by
  use (1, - sqrt 3 / 3)
  sorry

end normal_vector_of_line_l611_611710


namespace negation_of_exists_rational_number_l611_611175

theorem negation_of_exists_rational_number :
  (¬ ∃ x : ℚ, x^2 = 2) ↔ (∀ x : ℚ, x^2 ≠ 2) :=
begin
  sorry
end

end negation_of_exists_rational_number_l611_611175


namespace geometric_series_sum_l611_611643

theorem geometric_series_sum :
  let a := (1 / 4 : ℚ)
  let r := (1 / 4 : ℚ)
  let n := 5
  let S := a * (1 - r^n) / (1 - r)
  S = 341 / 1024 :=
by
  let a := (1 / 4 : ℚ)
  let r := (1 / 4 : ℚ)
  let n := 5
  let S := a * (1 - r^n) / (1 - r)
  show S = 341 / 1024
  sorry

end geometric_series_sum_l611_611643


namespace probability_of_ge_four_is_one_eighth_l611_611200

noncomputable def probability_ge_four : ℝ :=
sorry

theorem probability_of_ge_four_is_one_eighth :
  ∀ (x y : ℝ), (0 ≤ x ∧ x ≤ 2) ∧ (0 ≤ y ∧ y ≤ 2) →
  (probability_ge_four = 1 / 8) :=
sorry

end probability_of_ge_four_is_one_eighth_l611_611200


namespace uniform_motion_direct_proportion_l611_611632

-- Definitions based on conditions
variable (a b h t v S : ℝ)

-- The conditions given a constant area of a rectangle, area of a square, constant area of a triangle, and uniform motion
def constant_area_rectangle := (S = a * b)
def area_square := (S = a^2)
def constant_area_triangle := (S = 1 / 2 * b * h)
def uniform_motion_fixed_speed (v : ℝ) := (S = v * t)

-- The final statement to prove
theorem uniform_motion_direct_proportion {v : ℝ} (hv : v ≠ 0) (t : ℝ) : uniform_motion_fixed_speed a b := 
have h1 : uniform_motion_fixed_speed v := (S = v * t),
sorry

end uniform_motion_direct_proportion_l611_611632


namespace mail_cars_in_train_l611_611251

theorem mail_cars_in_train (n : ℕ) (hn : n % 2 = 0) (hfront : 1 ≤ n ∧ n ≤ 20)
  (hclose : ∀ i, 1 ≤ i ∧ i < n → (∃ j, i < j ∧ j ≤ 20))
  (hlast : 4 * n ≤ 20)
  (hconn : ∀ k, (k = 4 ∨ k = 5 ∨ k = 15 ∨ k = 16) → 
                  (∃ j, j = k + 1 ∨ j = k - 1)) :
  ∃ (i : ℕ) (j : ℕ), i = 4 ∧ j = 16 :=
by
  sorry

end mail_cars_in_train_l611_611251


namespace length_DP_l611_611467

theorem length_DP (A B C D P : Type) [MetricSpace A] [MetricSpace B]
  (AB CD : A) (AP BP CP DP : ℝ)
  (h1 : AB ⟂ CD) (h2 : P ∈ AB) (h3 : P ∈ CD) 
  (h4 : AP = 2) (h5 : BP = 3) (h6 : CP = 1)
  (h_circle : Set.mem A {B, C, D}) :
  DP = 6 :=
sorry

end length_DP_l611_611467


namespace a4_in_factorial_base_945_l611_611946

open Nat

def factorial_base_representation (n : Nat) : List Nat :=
  let rec aux (n k : Nat) (acc : List Nat) : List Nat :=
    if k = 0 then acc.reverse
    else
      let coeff := n / (factorial k)
      let rem := n % (factorial k)
      aux rem (k - 1) (coeff :: acc)
  aux n n []

theorem a4_in_factorial_base_945 : (factorial_base_representation 945).nth 4 = some 4 :=
by
  sorry

end a4_in_factorial_base_945_l611_611946


namespace find_x_l611_611990

theorem find_x : ∃ x : ℕ, 320 * 2 * 3 = x ∧ x = 1920 :=
by
  use 1920
  constructor
  · exact rfl
  · exact rfl

end find_x_l611_611990


namespace rectangle_area_computation_l611_611469

variable (PA PB PC PD : ℝ)
variable (a b c : ℕ)

theorem rectangle_area_computation
  (h1 : PA * PB = 2)
  (h2 : PC * PD = 18)
  (h3 : PB * PC = 9)
  (area_proof : ∃ (A B : ℕ) (C : ℝ), A = 208 ∧ B = 85 ∧ C = 17 ∧ area_rectangle = A * sqrt(C) / B) :
  100 * 208 + 10 * 17 + 85 = 21055 := by
sorry

end rectangle_area_computation_l611_611469


namespace relationship_between_abc_l611_611015

noncomputable def a : ℝ := 1.1 * Real.log 1.1
noncomputable def b : ℝ := 0.1 * Real.exp 0.1
def c : ℝ := 1 / 9

theorem relationship_between_abc : a < b ∧ b < c := by 
  sorry

end relationship_between_abc_l611_611015


namespace solve_arctan_equation_l611_611663

theorem solve_arctan_equation :
  ∃ x : ℝ, 3 * arctan (1 / 4) + arctan (1 / 20) + arctan (1 / x) = π / 4 ∧ x = 1985 :=
by
  sorry

end solve_arctan_equation_l611_611663


namespace homes_with_panels_installed_l611_611291

-- Let's define the conditions as constants
def total_homes : ℕ := 20
def panels_per_home : ℕ := 10
def panels_less_delivered : ℕ := 50

-- State the problem as a theorem
theorem homes_with_panels_installed :
  let total_panels_required := total_homes * panels_per_home,
      panels_delivered := total_panels_required - panels_less_delivered,
      homes_installed := panels_delivered / panels_per_home in
  homes_installed = 15 :=
by
  let total_panels_required := total_homes * panels_per_home
  let panels_delivered := total_panels_required - panels_less_delivered
  let homes_installed := panels_delivered / panels_per_home
  show homes_installed = 15
  sorry

end homes_with_panels_installed_l611_611291


namespace cos_a3_value_l611_611042

theorem cos_a3_value (a : ℕ → ℝ) (h : ∀ n, a (n + 1) - a n = a 1 - a 0) 
  (h_sum : a 1 + a 3 + a 5 = Real.pi) : 
  Real.cos (a 3) = 1/2 := 
by 
  sorry

end cos_a3_value_l611_611042


namespace part_1_part_2_part_3_l611_611438

section SequenceProofs

def a : ℕ → ℚ
| 0       := 1
| (n + 1) := 2 * a n / (2 + a n)

theorem part_1 : a 1 = 2 / 3 ∧ a 2 = 2 / 4 ∧ a 3 = 2 / 5 := 
by {
  have h1 : a 1 = 2 / 3 := by sorry,
  have h2 : a 2 = 2 / 4 := by sorry,
  have h3 : a 3 = 2 / 5 := by sorry,
  exact ⟨h1, h2, h3⟩
}

theorem part_2 : ∀ n : ℕ, a n = 2 / (n + 1) :=
by {
  intro n,
  sorry
}

def b (n : ℕ) : ℚ := a n / n

def S (n : ℕ) : ℚ := ∑ i in finset.range n, b (i + 1)

theorem part_3 : ∀ n : ℕ, S n = 2 * n / (n + 1) :=
by {
  intro n,
  sorry
}

end SequenceProofs

end part_1_part_2_part_3_l611_611438


namespace identical_circles_fully_inside_not_touching_no_common_tangents_l611_611554

def identical_circles (c1 c2 : circle) : Prop :=
  c1.radius = c2.radius

def fully_inside (c1 c2 : circle) : Prop :=
  c1.center.distance c2.center < c2.radius - c1.radius

def not_touching (c1 c2 : circle) : Prop :=
  c1.center.distance c2.center < abs (c1.radius - c2.radius)

noncomputable def common_tangents_count : ℕ :=
  0

theorem identical_circles_fully_inside_not_touching_no_common_tangents (c1 c2 : circle)
  (h1 : identical_circles c1 c2)
  (h2 : fully_inside c1 c2)
  (h3 : not_touching c1 c2) :
  common_tangents_count = 0 :=
sorry

end identical_circles_fully_inside_not_touching_no_common_tangents_l611_611554


namespace range_of_d_l611_611040

section proof_problem

variables {k m : ℝ}

/-- Definitions of the ellipse, foci, and line l --/
def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1
def F1 : ℝ × ℝ := (-1, 0)
def F2 : ℝ × ℝ := (1, 0)
def line_l (x y : ℝ) : Prop := y = k * x + m

/-- Conditions for intersection and slope constraints --/
def intersects_ellipse (A B : ℝ × ℝ) : Prop :=
  ∃ x1 y1 x2 y2, A = (x1, y1) ∧ B = (x2, y2) ∧ ellipse x1 y1 ∧ ellipse x2 y2 ∧
  line_l x1 y1 ∧ line_l x2 y2

def slopes_arithmetic_sequence (A B : ℝ × ℝ) : Prop :=
  ∃ x1 y1 x2 y2, A = (x1, y1) ∧ B = (x2, y2) ∧
  let m1 := (y1 / (x1 + 1)) in
  let m2 := k in
  let m3 := (y2 / (x2 + 1)) in
  (m1 + m3) = 2 * m2

/-- Distance from F2 to line l --/
def distance_to_line (k m : ℝ) : ℝ :=
  (abs (k + k + 1 / (2 * k)) / real.sqrt (1 + k^2))

/-- Main theorem to prove the range of d --/
theorem range_of_d (A B : ℝ × ℝ) (h1 : intersects_ellipse A B) (h2 : slopes_arithmetic_sequence A B) :
  ∃ d : ℝ, √3 < d ∧ d < 2 ∧ d = distance_to_line k m :=
sorry

end proof_problem

end range_of_d_l611_611040


namespace det_reflection_dilation_matrix_l611_611455

theorem det_reflection_dilation_matrix :
  let R := (matrix.of ![![5, 0], ![0, 5]]) * (matrix.of ![![1, 0], ![0, -1]]) in
  matrix.det R = -25 := by
  sorry

end det_reflection_dilation_matrix_l611_611455


namespace perpendicular_line_to_plane_l611_611994

theorem perpendicular_line_to_plane {l : ℝ → ℝ → ℝ} {α : set (ℝ → ℝ)} (h1 : ∀ x y, x ∈ α → y ∈ α → x ≠ y → (l ⊥ x ∧ l ⊥ y)) : 
  (∀ x y, x ∈ α → y ∈ α → x ≠ y → l ⊥ x → l ⊥ y) ∧ ¬(∀ x y, x ∈ α → y ∈ α → x ≠ y → l ⊥ α) :=
sorry

end perpendicular_line_to_plane_l611_611994


namespace find_yield_of_1_tree_l611_611835

variables (yield_3_trees yield_2_trees yield_1_tree average_yield : ℕ)
variables (total_trees total_yield : ℕ)

def problem_conditions := 
  yield_3_trees = 3 * 60 ∧
  yield_2_trees = 2 * 120 ∧
  average_yield = 100 ∧
  total_trees = 3 + 2 + 1 ∧
  total_yield = total_trees * average_yield

theorem find_yield_of_1_tree
  (yield_3_trees yield_2_trees yield_1_tree average_yield total_trees total_yield : ℕ)
  (h : problem_conditions yield_3_trees yield_2_trees yield_1_tree average_yield total_trees total_yield) :
  yield_1_tree = 180 :=
by
  sorry

end find_yield_of_1_tree_l611_611835


namespace coronavirus_diameter_in_scientific_notation_l611_611477

noncomputable def nanometers_to_millimeters (nm : ℕ) : ℝ := nm * 0.000001

theorem coronavirus_diameter_in_scientific_notation :
  nanometers_to_millimeters 80 = 8 * 10^(-5) :=
by
  sorry

end coronavirus_diameter_in_scientific_notation_l611_611477


namespace simplify_sqrt_sin_double_angle_l611_611405

noncomputable def theta : ℝ := sorry

theorem simplify_sqrt_sin_double_angle (hθ : theta ∈ Set.Icc (5/4 * Real.pi) (3/2 * Real.pi)) :
  sqrt (1 - Real.sin (2 * theta)) - sqrt (1 + Real.sin (2 * theta)) = 2 * Real.cos theta :=
 
  sorry

end simplify_sqrt_sin_double_angle_l611_611405


namespace biscuit_dimensions_l611_611968

theorem biscuit_dimensions (sheet_length : ℝ) (sheet_width : ℝ) (num_biscuits : ℕ) 
  (h₁ : sheet_length = 12) (h₂ : sheet_width = 12) (h₃ : num_biscuits = 16) :
  ∃ biscuit_length : ℝ, biscuit_length = 3 :=
by
  sorry

end biscuit_dimensions_l611_611968


namespace sum_of_prime_factors_l611_611976

theorem sum_of_prime_factors (n : ℕ) (h : n = 257040) : 
  (2 + 5 + 3 + 107 = 117) :=
by sorry

end sum_of_prime_factors_l611_611976


namespace total_sum_lent_l611_611626

noncomputable def interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem total_sum_lent 
  (x y : ℝ)
  (h1 : interest x (3 / 100) 5 = interest y (5 / 100) 3) 
  (h2 : y = 1332.5) : 
  x + y = 2665 :=
by
  -- We would continue the proof steps here.
  sorry

end total_sum_lent_l611_611626


namespace fishers_tomorrow_l611_611807

-- Define the groups of fishers and their fishing pattern
def everyday_fishers := 7
def every_other_day_fishers := 8
def every_three_days_fishers := 3

-- Given counts for yesterday and today
def fishers_yesterday := 12
def fishers_today := 10

-- The problem to prove: 15 people will fish tomorrow
theorem fishers_tomorrow : 
  (fishers_yesterday = 12) →
  (fishers_today = 10) →
  (everyday_fishers = 7) →
  (every_other_day_fishers = 8) →
  (every_three_days_fishers = 3) →
  15 = everyday_fishers + every_other_day_fishers / 2 + every_three_days_fishers / 3 :=
begin
  sorry
end

end fishers_tomorrow_l611_611807


namespace negation_of_p_is_correct_l611_611727

variable (c : ℝ)

-- Proposition p defined as: there exists c > 0 such that x^2 - x + c = 0 has a solution
def proposition_p : Prop :=
  ∃ c > 0, ∃ x : ℝ, x^2 - x + c = 0

-- Negation of proposition p
def neg_proposition_p : Prop :=
  ∀ c > 0, ¬ ∃ x : ℝ, x^2 - x + c = 0

-- The Lean statement to prove
theorem negation_of_p_is_correct :
  neg_proposition_p ↔ (∀ c > 0, ¬ ∃ x : ℝ, x^2 - x + c = 0) :=
by
  sorry

end negation_of_p_is_correct_l611_611727


namespace fishers_tomorrow_l611_611809

-- Define the groups of fishers and their fishing pattern
def everyday_fishers := 7
def every_other_day_fishers := 8
def every_three_days_fishers := 3

-- Given counts for yesterday and today
def fishers_yesterday := 12
def fishers_today := 10

-- The problem to prove: 15 people will fish tomorrow
theorem fishers_tomorrow : 
  (fishers_yesterday = 12) →
  (fishers_today = 10) →
  (everyday_fishers = 7) →
  (every_other_day_fishers = 8) →
  (every_three_days_fishers = 3) →
  15 = everyday_fishers + every_other_day_fishers / 2 + every_three_days_fishers / 3 :=
begin
  sorry
end

end fishers_tomorrow_l611_611809


namespace difficult_more_than_easy_l611_611218

-- Define the condition problem
variables (x y z : ℕ)

-- State the conditions
def condition_1 := x + y + z = 100
def condition_2 := x + 2y + 3z = 180

-- Prove that the number of difficult problems is more than the number of easy problems by 20
theorem difficult_more_than_easy (h1 : condition_1 x y z) (h2 : condition_2 x y z) : x - z = 20 :=
by {
  sorry
}

end difficult_more_than_easy_l611_611218


namespace abs_diff_eq_two_l611_611289

noncomputable def a1 := 17
noncomputable def b1 := 15

theorem abs_diff_eq_two :
  ∀ (a_1 a_2 ... a_m : ℕ) (b_1 b_2 ... b_n : ℕ),
    (a_1 ≥ a_2) → (b_1 ≥ b_2) → 
    (a_1 ≥ ... ≥ a_m) → (b_1 ≥ ... ≥ b_n) →
    (a_1 + b_1) = 32 →
    (1870 = (nat.factorial a_1) * (nat.factorial a_2) * ... * (nat.factorial a_m) / 
     ((nat.factorial b_1) * (nat.factorial b_2) * ... * (nat.factorial b_n))) →
    abs (a_1 - b_1) = 2 :=
begin
  sorry
end

end abs_diff_eq_two_l611_611289


namespace percentage_of_green_ducks_l611_611839

theorem percentage_of_green_ducks (ducks_small_pond ducks_large_pond : ℕ) 
  (green_fraction_small_pond green_fraction_large_pond : ℚ) 
  (h1 : ducks_small_pond = 20) 
  (h2 : ducks_large_pond = 80) 
  (h3 : green_fraction_small_pond = 0.20) 
  (h4 : green_fraction_large_pond = 0.15) :
  let total_ducks := ducks_small_pond + ducks_large_pond
  let green_ducks := (green_fraction_small_pond * ducks_small_pond) + 
                     (green_fraction_large_pond * ducks_large_pond)
  (green_ducks / total_ducks) * 100 = 16 := 
by 
  sorry

end percentage_of_green_ducks_l611_611839


namespace find_number_l611_611589

noncomputable def x : Real := 1.375  -- This sets the intended solution

theorem find_number (x : Real) : (0.6667 * x + 0.75 = 1.6667) → (x = 1.375) :=
by
correct_answer := 1.375
sorry

end find_number_l611_611589


namespace part_a_l611_611448

-- Defining the variables and conditions
variable {A B C K L M N P Q S T : Point}

-- Let ABC be a right-angled triangle with ∠C = 90°
axiom H1 : ∠ A C B = 90

-- K, L, M be the midpoints of sides AB, BC, and CA respectively
axiom H2 : midpoint A B K
axiom H3 : midpoint B C L
axiom H4 : midpoint C A M

-- N is a point on side AB
axiom H5 : on_segment A B N

-- Line CN meets KM and KL at points P and Q respectively
axiom H6 : meet C N K M P
axiom H7 : meet C N K L Q

-- Points S and T are on AC and BC such that APQS and BPQT are cyclic quadrilaterals
axiom H8 : on_segment A C S
axiom H9 : on_segment B C T
axiom H10 : cyclic_quad A P Q S
axiom H11 : cyclic_quad B P Q T

-- Prove the concurrency result
theorem part_a : is_bisector C N → concur C N M L S T := 
by sorry

end part_a_l611_611448


namespace train_length_l611_611252

theorem train_length (time : ℝ) (speed_kmh : ℝ) (speed_ms : ℝ) (length : ℝ) : 
  time = 3.499720022398208 ∧ 
  speed_kmh = 144 ∧ 
  speed_ms = 40 ∧ 
  length = speed_ms * time → 
  length = 139.98880089592832 :=
by sorry

end train_length_l611_611252


namespace top_face_even_dots_probability_l611_611480

noncomputable def probability_top_face_even_dots : ℚ :=
let prob := 1/6 * (1 + 2/420 + (1 - 6/420) + 12/420 + (1 - 20/420) + 30/420) in
167 / 630

theorem top_face_even_dots_probability :
  probability_top_face_even_dots = 167 / 630 := by
-- Include the actual mathematical proof steps here
sorry

end top_face_even_dots_probability_l611_611480


namespace solve_linear_system_l611_611397

theorem solve_linear_system (x y : ℝ) (h1 : 2 * x + 3 * y = 5) (h2 : 3 * x + 2 * y = 10) : x + y = 3 := 
by
  sorry

end solve_linear_system_l611_611397


namespace fishing_tomorrow_l611_611775

theorem fishing_tomorrow 
  (P1 : ∀ day : ℕ, 7 ∈ {p | p goes fishing on day})
  (P2 : ∀ day : ℕ, day % 2 = 0 → 8 ∈ {p | p goes fishing on day})
  (P3 : ∀ day : ℕ, day % 3 = 0 → 3 ∈ {p | p goes fishing on day})
  (P4 : ℕ)
  (yesterday : ℕ)
  (today : ℕ)
  (tomorrow : ℕ)
  (hyesterday : yesterday = 12)
  (htoday : today = 10)
  : tomorrow = 15 := by
  sorry

end fishing_tomorrow_l611_611775


namespace not_more_than_half_million_moves_l611_611894

theorem not_more_than_half_million_moves :
  ∀ (cards : ℕ → Prop) (covered : ℕ → Prop),
  (∀ n, 1 ≤ n ∧ n ≤ 1000 → cards n) →
  (∀ n, covered n → 1 ≤ n ∧ n ≤ 1000) →
  (∀ n, (∃ m, m = n + 1 ∧ covered m → covered n) → ∃ m, cards m ∧ cards (m + 1)) →
  ∃ max_moves : ℕ, 
    (max_moves = (999 * 1000) / 2 ∧ max_moves ≤ 500000) :=
begin
  sorry
end

end not_more_than_half_million_moves_l611_611894


namespace oranges_per_box_l611_611495

theorem oranges_per_box (h_oranges : 56 = 56) (h_boxes : 8 = 8) : 56 / 8 = 7 :=
by
  -- Placeholder for the proof
  sorry

end oranges_per_box_l611_611495


namespace mona_game_group_size_l611_611476

theorem mona_game_group_size 
  (x : ℕ)
  (h_conditions: 9 * (x - 1) - 3 = 33) : x = 5 := 
by 
  sorry

end mona_game_group_size_l611_611476


namespace fishers_tomorrow_l611_611788

-- Definitions based on conditions
def people_every_day : ℕ := 7
def people_every_other_day : ℕ := 8
def people_every_three_days : ℕ := 3
def people_yesterday : ℕ := 12
def people_today : ℕ := 10

-- Theorem to be proved
theorem fishers_tomorrow (people_every_day people_every_other_day people_every_three_days people_yesterday people_today : ℕ) : 
  people_every_day = 7 ∧ 
  people_every_other_day = 8 ∧ 
  people_every_three_days = 3 ∧
  people_yesterday = 12 ∧ 
  people_today = 10 →
  15 := 
by {
  sorry
}

end fishers_tomorrow_l611_611788


namespace oranges_min_number_l611_611109

theorem oranges_min_number (n : ℕ) 
  (h : ∀ (m : ℕ → ℝ), (∀ i j k : ℕ, i < n → j < n → k < n →
    i ≠ j → j ≠ k → i ≠ k → m i + m j + m k < 0.05 * (∑ l in (finset.range n \ {i,j,k}), m l))) : 
  n ≥ 64 :=
sorry

end oranges_min_number_l611_611109


namespace imaginaria_8_letter_words_mod_1000_l611_611422

def a (n : ℕ) : ℕ := if n = 2 then 4 else 2 * (a (n-1) + b (n-1) + c (n-1))
def b (n : ℕ) : ℕ := if n = 2 then 2 else 2 * c (n-1)
def c (n : ℕ) : ℕ := if n = 2 then 4 else 2 * a (n-1)

theorem imaginaria_8_letter_words_mod_1000 : 
  (a 8 + b 8 + c 8) % 1000 = 56 :=
by
  -- This would be the point where the proof steps to establish the correctness of
  -- a 8, b 8, and c 8 would be included.
  sorry

end imaginaria_8_letter_words_mod_1000_l611_611422


namespace at_least_one_mistake_l611_611895

def chessboard := fin 8 × fin 8
def adjacent_by_side (p q : chessboard) : Prop :=
  (p.1 = q.1 ∧ |p.2 - q.2| = 1) ∨ (p.2 = q.2 ∧ |p.1 - q.1| = 1)

variables {n : ℕ} (arr : chessboard → ℕ)

-- Sum of products of adjacent pairs written by Fedir
def Fedir_value : ℕ :=
  ∑ p q, if adjacent_by_side p q then arr p * arr q else 0

-- Product of sums of adjacent pairs written by Oleksiy
def Oleksiy_value : ℕ :=
  ∏ p q, if adjacent_by_side p q then arr p + arr q else 1

theorem at_least_one_mistake (arr : chessboard → ℕ) :
  last_digit (Fedir_value arr) = 1 → last_digit (Oleksiy_value arr) = 1 → false :=
by
  sorry

end at_least_one_mistake_l611_611895


namespace deriv_y1_deriv_y2_deriv_y3_l611_611639

variable (x : ℝ)

-- Prove the derivative of y = 3x^3 - 4x is 9x^2 - 4
theorem deriv_y1 : deriv (λ x => 3 * x^3 - 4 * x) x = 9 * x^2 - 4 := by
sorry

-- Prove the derivative of y = (2x - 1)(3x + 2) is 12x + 1
theorem deriv_y2 : deriv (λ x => (2 * x - 1) * (3 * x + 2)) x = 12 * x + 1 := by
sorry

-- Prove the derivative of y = x^2 (x^3 - 4) is 5x^4 - 8x
theorem deriv_y3 : deriv (λ x => x^2 * (x^3 - 4)) x = 5 * x^4 - 8 * x := by
sorry


end deriv_y1_deriv_y2_deriv_y3_l611_611639


namespace trajectory_of_M_is_parabola_l611_611944

variable (x y : ℝ)

theorem trajectory_of_M_is_parabola (h : 5 * Real.sqrt (x^2 + y^2) = abs (3 * x + 4 * y - 12)) : 
  ∃ (F : Point Real) (D : Line Real), ∀ (M : Point Real), M.focal_point_relation F D :=
sorry

end trajectory_of_M_is_parabola_l611_611944


namespace y_odd_and_period_pi_l611_611230

def y (x : ℝ) : ℝ := (sin x - cos x) ^ 2 - 1

theorem y_odd_and_period_pi :
  (∀ x : ℝ, y (-x) = -y x) ∧ (∃ T : ℝ, T = π ∧ ∀ x : ℝ, y (x + T) = y x) :=
sorry

end y_odd_and_period_pi_l611_611230


namespace integral_value_l611_611687

-- Given conditions
variable (a : ℝ) (h₁ : a > 0)
variable (h₂ : (15 : ℤ) = (Nat.choose 6 2 * a^4 : ℤ))

-- Question and result to prove
theorem integral_value : 
  ∫ x in -a..a, (x^2 + x + real.sqrt (4 - x^2)) = (2/3) + (2 * real.pi / 3) + real.sqrt 3 :=
sorry

end integral_value_l611_611687


namespace sin_x1_x2_eq_l611_611363

def f : ℝ → ℝ := λ x => 2 * sin (2 * x + π / 4) + 2 * cos (x + π / 8) ^ 2 - 1

def g (x : ℝ) : ℝ := f (x - π / 8)

noncomputable def value_of_m : ℝ := sorry -- This will be the value of m derived in the problem

def x1 (x : ℝ) : Prop := g x - value_of_m = 0
def x2 (x : ℝ) : Prop := g x - value_of_m = 0

theorem sin_x1_x2_eq : ∀ (x1 x2 : ℝ), 
  (0 ≤ x1 ∧ x1 ≤ π / 2) ∧ (0 ≤ x2 ∧ x2 ≤ π / 2) ∧ 
  g x1 - value_of_m = 0 ∧ g x2 - value_of_m = 0 → 
  sin (x1 + x2) = (2 * Real.sqrt 5) / 5 := by
  sorry

end sin_x1_x2_eq_l611_611363


namespace part1_part2_part3_l611_611721

def f (x : ℝ) : ℝ := x^2 * exp (x - 1) - x^3 - x^2
def g (x : ℝ) : ℝ := x^3 - x^2

theorem part1 (h_ext : ∀ x, x = -2 ∨ x = 1 → deriv f x = 0) : 
  f = λ x, x^2 * exp (x - 1) - x^3 - x^2 :=
sorry

theorem part2 (h_ext : ∀ x, x = -2 ∨ x = 1 → deriv f x = 0) : 
  (∀ x, x ∈ Ioo (-∞ : ℝ) (-2)    → deriv f x < 0) ∧
  (∀ x, x ∈ Ioo (-2 : ℝ) 0       → deriv f x > 0) ∧
  (∀ x, x ∈ Ioo (0 : ℝ) 1        → deriv f x < 0) ∧
  (∀ x, x ∈ Ioo (1 : ℝ) (∞ : ℝ)  → deriv f x > 0) :=
sorry

theorem part3 : ∀ x, f x ≥ g x :=
sorry

end part1_part2_part3_l611_611721


namespace crayons_per_box_l611_611890

axiom Michelle_boxes : ℕ := 7
axiom Michelle_total_crayons : ℕ := 35

theorem crayons_per_box : (Michelle_total_crayons / Michelle_boxes) = 5 := by
  sorry

end crayons_per_box_l611_611890


namespace pizza_volume_piece_l611_611622

theorem pizza_volume_piece (r h : ℝ) (n : ℕ) (pizza_is_cylinder : true) (pizza_thickness : h = 1 / 3) 
    (pizza_diameter : 2 * r = 12) (congruent_pieces : n = 12) : 
    (π * r^2 * h) / n = π :=
by
  have r_value : r = 6 := by
    linarith
  have h_value : h = 1 / 3 := pizza_thickness
  have n_value : n = 12 := congruent_pieces

  calc
    (π * r^2 * h) / n = (π * (6)^2 * (1 / 3)) / 12 := by
      rw [r_value, h_value, n_value]
    ... = (π * 36 * 1 / 3) / 12 := by
      norm_num
    ... = (π * 12 ) / 12 := by
      norm_num
    ... = π := by
      norm_num

end pizza_volume_piece_l611_611622


namespace problem1_problem2_l611_611233

-- Problem (I) conditions and conclusion
theorem problem1 (x : ℝ) : 4^x - 2^(x+2) - 12 = 0 → x = 1 := 
sorry

-- Problem (II) conditions
theorem problem2 (x t : ℝ) (ht : log 3 x = 1 + |t|) : ∀ x, log 3 x = 1 + |t| → log 2 (x^2 - 4*x + 5) ∈ Set.Icc 1 real_top :=
sorry

end problem1_problem2_l611_611233


namespace surface_area_ratio_l611_611186

theorem surface_area_ratio (V1 V2 S1 S2 : ℝ) 
  (h1 : V1 / V2 = 8 / 27) 
  (h2 : V1 = (4/3) * π * (r1^3)) 
  (h3 : V2 = (4/3) * π * (r2^3)) 
  (h4 : S1 = 4 * π * (r1^2)) 
  (h5 : S2 = 4 * π * (r2^2)) : 
  S1 / S2 = 4 / 9 :=
by
  -- Convert the volume ratio condition to radii ratio
  have hr: (r1 / r2)^3 = 8 / 27, from sorry,
  -- Convert radii ratio to the surface area ratio
  have hs : (r1 / r2)^2 = 4 / 9, from sorry,
  sorry

end surface_area_ratio_l611_611186


namespace bird_problem_l611_611190

theorem bird_problem (B : ℕ) (h : (2 / 15) * B = 60) : B = 450 ∧ (2 / 15) * B = 60 :=
by
  sorry

end bird_problem_l611_611190


namespace complex_fraction_evaluation_l611_611579

theorem complex_fraction_evaluation :
  ( 
    ((3 + 1/3) / 10 + 0.175 / 0.35) / 
    (1.75 - (1 + 11/17) * (51/56)) - 
    ((11/18 - 1/15) / 1.4) / 
    ((0.5 - 1/9) * 3)
  ) = 1/2 := 
sorry

end complex_fraction_evaluation_l611_611579


namespace exists_solution_negation_correct_l611_611340

theorem exists_solution_negation_correct :
  (∃ x : ℝ, x^2 - x = 0) ↔ (∃ x : ℝ, True) ∧ (∀ x : ℝ, ¬ (x^2 - x = 0)) :=
by
  sorry

end exists_solution_negation_correct_l611_611340


namespace not_possible_6x6_table_l611_611049

open Function

theorem not_possible_6x6_table :
  ¬(∃ f : Fin 6 × Fin 6 → ℝ,
    (∀ i : Fin 6, (∏ j, f (i, j)) = 2) ∧
    (∀ j : Fin 6, (∏ i, f (i, j)) = 2) ∧
    (∀ i j : Fin 5, (∏ di dj, f (⟨i.val + di.val % 2, by linarith⟩, ⟨j.val + dj.val % 2, by linarith⟩)) = 2)) :=
by
  sorry

end not_possible_6x6_table_l611_611049


namespace number_of_correct_propositions_l611_611528

-- Definitions of conditions based on the problem
def proposition1 (α β : Plane) : Prop := 
  ∃ p : Point, p ∈ α ∧ p ∈ β ∧ finite ({p' : Point | p' ∈ α ∧ p' ∈ β}.to_set)

def proposition2 (l : Line) (α : Plane) : Prop :=
  ∃ (S : Set Point) (h : infinite S), (∀ p ∈ S, p ∉ α) → l ∥ α

def proposition3 (l₁ l₂ l₃ : Line) : Prop :=
  equal_angle l₁ l₃ l₂ l₃ → l₁ ∥ l₂

def proposition4 (α β : Plane) (a b : Line) : Prop :=
  (a ⊆ α ∧ a ∥ β) → (b ⊆ β ∧ b ∥ α) → α ∥ β

-- Theorem stating the number of correct propositions
theorem number_of_correct_propositions (α β : Plane) (l l₁ l₂ l₃ : Line) (a b : Line) : 
  (¬ proposition1 α β) ∧ (¬ proposition2 l α) ∧ (¬ proposition3 l₁ l₂ l₃) ∧ proposition4 α β a b → 1 =
suffices from sorry

end number_of_correct_propositions_l611_611528


namespace number_of_elements_of_complement_l611_611884

open Set

def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {3, 4, 5}
def U : Set ℕ := A ∪ B

theorem number_of_elements_of_complement :
  (U \ (A ∩ B)).card = 3 :=
by
  sorry

end number_of_elements_of_complement_l611_611884


namespace solar_panels_l611_611292

theorem solar_panels (total_homes panels_per_home total_shortfall : ℕ) (h_total_homes : total_homes = 20) (h_panels_per_home : panels_per_home = 10) (h_total_shortfall : total_shortfall = 50) :
  (total_homes * panels_per_home - total_shortfall) / panels_per_home = 15 :=
by
  rw [h_total_homes, h_panels_per_home, h_total_shortfall]
  norm_num
  sorry

end solar_panels_l611_611292


namespace quotient_of_2213_div_13_in_base4_is_53_l611_611661

-- Definitions of the numbers in base 4
def n₁ : ℕ := 2 * 4^3 + 2 * 4^2 + 1 * 4^1 + 3 * 4^0  -- 2213_4 in base 10
def n₂ : ℕ := 1 * 4^1 + 3 * 4^0  -- 13_4 in base 10

-- The correct quotient in base 4 (converted from quotient in base 10)
def expected_quotient : ℕ := 5 * 4^1 + 3 * 4^0  -- 53_4 in base 10

-- The proposition we want to prove
theorem quotient_of_2213_div_13_in_base4_is_53 : n₁ / n₂ = expected_quotient := by
  sorry

end quotient_of_2213_div_13_in_base4_is_53_l611_611661


namespace servant_service_duration_l611_611611

theorem servant_service_duration
    (comp_full_year : ℕ)
    (uniform_value : ℕ)
    (received_salary : ℕ)
    (received_uniform_value : ℕ)
    (total_months : ℕ)
    (received_total : ℕ) :
    (comp_full_year + uniform_value) * total_months = received_total * 12 → 
    total_months = 9 :=
by
  intros h_eq
  rw [mul_comm 12 total_months, ← mul_assoc] at h_eq
  have h := congr_arg (λ x, x / 800) h_eq
  rwa [Nat.mul_div_cancel_left _ (Nat.pos_of_ne_zero zero_ne_800)] at h
  
-- Define the parameters based on the conditions given
def comp_full_year := 600
def uniform_value := 200
def received_salary := 400
def received_uniform_value := 200
def total_months := 9
def received_total := received_salary + received_uniform_value

end servant_service_duration_l611_611611


namespace hexagon_angle_arith_prog_l611_611512

theorem hexagon_angle_arith_prog (x d : ℝ) (hx : x > 0) (hd : d > 0) 
  (h_eq : 6 * x + 15 * d = 720) : x = 120 :=
by
  sorry

end hexagon_angle_arith_prog_l611_611512


namespace supplement_complement_60_eq_150_l611_611971

theorem supplement_complement_60_eq_150 :
  ∀ (θ : ℝ), θ = 60 → (180 - (90 - θ)) = 150 :=
by
  intro θ
  intro hθ
  rw hθ
  linarith

end supplement_complement_60_eq_150_l611_611971


namespace option_A_correct_option_B_correct_option_C_incorrect_option_D_correct_l611_611085

-- Definitions and conditions
def seq_a (S_n : ℕ → ℚ) (n : ℕ) : ℚ := if n = 1 then S_n 1 else S_n n - S_n (n - 1)
def mean_seq_b (S_n : ℕ → ℚ) (n : ℕ) : ℚ := S_n n / n
def S_n (n : ℕ) : ℚ := n * (n + 1) / 2^n
def b_n (n : ℕ) : ℚ := (n + 1) / 2^n
def T_n (n : ℕ) : ℚ := ∑ i in finset.range n, b_n (i + 1)

-- Theorem statements to prove the conclusions
theorem option_A_correct : seq_a S_n 5 = -5 / 16 := sorry
theorem option_B_correct : T_n 5 = 11 / 4 := sorry
theorem option_C_incorrect : ¬(∀ (n : ℕ), S_n (n + 1) < S_n n) := sorry
theorem option_D_correct (n : ℕ) (m : ℚ) (h : 4 * m^2 - m ≥ S_n n) : m ≤ -1/2 ∨ m ≥ 3/4 := sorry

end option_A_correct_option_B_correct_option_C_incorrect_option_D_correct_l611_611085


namespace bernoullis_misplacement_problem_l611_611517

/-
  Define the derangement function recursively.
-/
def derangement : ℕ → ℕ
| 0     := 1
| 1     := 0
| (n+1) := n * (derangement n + derangement (n-1))

/-
  Define factorial function.
-/
def fact : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * fact n

open nat

theorem bernoullis_misplacement_problem (n : ℕ) (n_ge_two : n ≥ 2) :
  derangement 6 = 265 ∧
  fact 6 = 720 ∧
  ((derangement 3 = 2 ∧ fact 3 = 6) ∨ (derangement 6 / fact 6 = 53 / 144)) ∧
  (derangement 4 = 9 ∧ derangement 5 = 44 ∧ derangement 6 / fact 6 ≠ 11 / 36) ∧
  (derangement 4 = 9 ∧ derangement 6 / fact 6 = 7 / 90) :=
begin
  sorry
end

end bernoullis_misplacement_problem_l611_611517


namespace seonho_original_money_l611_611107

variable (X : ℝ)
variable (spent_snacks : ℝ := (1/4) * X)
variable (remaining_after_snacks : ℝ := X - spent_snacks)
variable (spent_food : ℝ := (2/3) * remaining_after_snacks)
variable (final_remaining : ℝ := remaining_after_snacks - spent_food)

theorem seonho_original_money :
  final_remaining = 2500 -> X = 10000 := by
  -- Proof goes here
  sorry

end seonho_original_money_l611_611107


namespace problem1_problem2_l611_611357

-- Definitions and Conditions
def vector_a : ℝ × ℝ := (-1, 2)
def vector_c (m : ℝ) : ℝ × ℝ := (m - 1, 3 * m)
def vector_b (b1 b2 : ℝ) : ℝ × ℝ := (b1, b2)

-- Problem 1: Prove m = 2/5 given that vector_c is parallel to vector_a
theorem problem1 (m : ℝ) (h : 2 * (m - 1) + 3 * m = 0) : m = 2 / 5 :=
sorry

-- Problem 2: Prove the angle between vector_a - vector_b and vector_b is 3π/4 
theorem problem2 {b1 b2 : ℝ} 
  (h1 : real.sqrt ((-1 - b1) ^ 2 + (2 - b2) ^ 2) = 3)
  (h2 : (-1 + 2 * b1) * (2 * -1 - b1) + 2 * b2 * (-b2) = 0) :
  real.angle (vector_a - vector_b b1 b2) (vector_b b1 b2) = 3 * real.pi / 4 :=
sorry

end problem1_problem2_l611_611357


namespace sum_first_n_geometric_terms_l611_611063

theorem sum_first_n_geometric_terms (S : ℕ → ℝ) (a : ℕ → ℝ)
  (h1 : S 2 = 2) (h2 : S 6 = 4) :
  S 4 = 1 + Real.sqrt 5 :=
by
  sorry

end sum_first_n_geometric_terms_l611_611063


namespace smallest_positive_period_and_max_value_of_f_value_of_a_in_triangle_ABC_l611_611737

-- Conditions
def vec_m (x : ℝ) : ℝ × ℝ := (sqrt 3 * sin (2 * x) + 2, cos x)
def vec_n (x : ℝ) : ℝ × ℝ := (1, 2 * cos x)
def f (x : ℝ) : ℝ := (vec_m x).1 * (vec_n x).1 + (vec_m x).2 * (vec_n x).2

-- Main statements
theorem smallest_positive_period_and_max_value_of_f :
  (∀ x : ℝ, f (x + π) = f x) ∧ (∀ x : ℝ, f x ≤ 5) ∧ (∃ x : ℝ, f x = 5) :=
sorry

theorem value_of_a_in_triangle_ABC (A b area : ℝ) (b_pos : b = 1) (area_pos : area = sqrt 3 / 2)
  (fA : f A = 4) : ∃ a : ℝ, a = sqrt 3 :=
sorry

end smallest_positive_period_and_max_value_of_f_value_of_a_in_triangle_ABC_l611_611737


namespace largest_angle_is_75_l611_611172

-- Let the measures of the angles be represented as 3x, 4x, and 5x for some value x
variable (x : ℝ)

-- Define the angles based on the given ratio
def angle1 := 3 * x
def angle2 := 4 * x
def angle3 := 5 * x

-- The sum of the angles in a triangle is 180 degrees
axiom angle_sum : angle1 + angle2 + angle3 = 180

-- Prove that the largest angle is 75 degrees
theorem largest_angle_is_75 : 5 * (180 / 12) = 75 :=
by
  -- Proof is not required as per the instructions
  sorry

end largest_angle_is_75_l611_611172


namespace cost_for_sugar_substitutes_l611_611273

def packets_per_cup : ℕ := 1
def cups_per_day : ℕ := 2
def days : ℕ := 90
def packets_per_box : ℕ := 30
def price_per_box : ℕ := 4

theorem cost_for_sugar_substitutes : 
  (packets_per_cup * cups_per_day * days / packets_per_box) * price_per_box = 24 := by
  sorry

end cost_for_sugar_substitutes_l611_611273


namespace altitudes_intersect_at_one_point_l611_611581

theorem altitudes_intersect_at_one_point {A B C : Point} : 
  ∃ H : Point, is_orthocenter H A B C :=
sorry

end altitudes_intersect_at_one_point_l611_611581


namespace euler_formula_l611_611101

theorem euler_formula (x : ℝ) : complex.exp (complex.I * x) = complex.cos x + complex.sin x * complex.I := 
sorry

end euler_formula_l611_611101


namespace shaded_region_area_l611_611496

theorem shaded_region_area (diameter : ℝ) (r : ℝ) (length_ft : ℝ) :
  diameter = 4 → r = 2 → length_ft = 2 → 
  let length_in : ℝ := length_ft * 12  -- converting feet to inches
  let num_semicircles : ℝ := length_in / diameter
  let num_circles : ℝ := num_semicircles / 2
  let area_one_circle : ℝ := π * r^2
  let total_area : ℝ := num_circles * area_one_circle
  total_area = 12 * π :=
begin
  intros hdiam hr hlength,
  -- Conversion from feet to inches
  let length_in := length_ft * 12, 
  -- Number of semicircles in the pattern
  let num_semicircles := length_in / diameter,
  -- Number of full circles
  let num_circles := num_semicircles / 2,
  -- Area of one full circle
  let area_one_circle := π * r^2,
  -- Total area of the shaded region
  let total_area := num_circles * area_one_circle,
  -- Proof that the total_area equals 12π
  rw [hdiam, hr, hlength],
  sorry
end

end shaded_region_area_l611_611496


namespace arithmetic_sequence_a4_l611_611869

theorem arithmetic_sequence_a4 (S : ℕ → ℚ) (a : ℕ → ℚ) (a1 : ℚ) (d : ℚ) :
  S 8 = 30 → S 4 = 7 → 
      (∀ n, S n = n * a1 + (n * (n - 1) / 2) * d) → 
      a 4 = a1 + 3 * d → 
      a 4 = 13 / 4 := by
  intros hS8 hS4 hS_formula ha4_formula
  -- Formal proof to be filled in
  sorry

end arithmetic_sequence_a4_l611_611869


namespace fishers_tomorrow_l611_611804

-- Define the groups of fishers and their fishing pattern
def everyday_fishers := 7
def every_other_day_fishers := 8
def every_three_days_fishers := 3

-- Given counts for yesterday and today
def fishers_yesterday := 12
def fishers_today := 10

-- The problem to prove: 15 people will fish tomorrow
theorem fishers_tomorrow : 
  (fishers_yesterday = 12) →
  (fishers_today = 10) →
  (everyday_fishers = 7) →
  (every_other_day_fishers = 8) →
  (every_three_days_fishers = 3) →
  15 = everyday_fishers + every_other_day_fishers / 2 + every_three_days_fishers / 3 :=
begin
  sorry
end

end fishers_tomorrow_l611_611804


namespace Tony_can_add_4_pairs_of_underwear_l611_611194

-- Define relevant variables and conditions
def max_weight : ℕ := 50
def w_socks : ℕ := 2
def w_underwear : ℕ := 4
def w_shirt : ℕ := 5
def w_shorts : ℕ := 8
def w_pants : ℕ := 10

def pants : ℕ := 1
def shirts : ℕ := 2
def shorts : ℕ := 1
def socks : ℕ := 3

def total_weight (pants shirts shorts socks : ℕ) : ℕ :=
  pants * w_pants + shirts * w_shirt + shorts * w_shorts + socks * w_socks

def remaining_weight : ℕ :=
  max_weight - total_weight pants shirts shorts socks

def additional_pairs_of_underwear_cannot_exceed : ℕ :=
  remaining_weight / w_underwear

-- Problem statement in Lean
theorem Tony_can_add_4_pairs_of_underwear :
  additional_pairs_of_underwear_cannot_exceed = 4 :=
  sorry

end Tony_can_add_4_pairs_of_underwear_l611_611194


namespace choose_six_with_consecutives_l611_611489

theorem choose_six_with_consecutives (h : ∀ t : Finset ℕ, t.card = 6 → t ⊆ Finset.range 1 50 → 
  ¬(∀ i ∈ t, ∀ j ∈ t, i ≠ j + 1)) : 
  finset.card ((finset.filter (λ s : finset ℕ, ∃ i ∈ s, ∃ j ∈ s, i = j + 1) 
  (finset.powerset_length 6 (finset.range 1 50)) ) = nat.choose 49 6 - nat.choose 44 6 :=
sorry

end choose_six_with_consecutives_l611_611489


namespace count_perfect_square_factors_other_than_one_l611_611739

-- Define the set S as {1, 2, 3, ..., 50}
def S : Finset ℕ := Finset.range 51

-- Define the predicate for having a perfect square factor other than one
def has_perfect_square_factor_other_than_one (n : ℕ) : Prop :=
  ∃ m : ℕ, m > 1 ∧ m * m ∣ n

-- Define the main theorem
theorem count_perfect_square_factors_other_than_one :
  (S.filter has_perfect_square_factor_other_than_one).card = 19 :=
by
  sorry

end count_perfect_square_factors_other_than_one_l611_611739


namespace possible_median_values_l611_611061

def S_pre : set ℤ := {5, 6, 7, 10, 12, 15, 20}

def is_odd (n : ℤ) : Prop := n % 2 ≠ 0

theorem possible_median_values :
  ∀ (additional : set ℤ), 
    (∀ x ∈ additional, is_odd x) →
    S_pre ∪ additional = S →
    S_pre.count + additional.count = 11 →
    ∃ possible_medians, possible_medians.count = 4 := 
sorry

end possible_median_values_l611_611061


namespace length_of_OT_l611_611423

-- Definitions for points and lengths
variables {P M O S T : Type*} [Point P] [Point M] [Point O] [Point S] [Point T]
noncomputable def P_MO := λ (PM PO : ℝ), PM = 6 * real.sqrt 3 ∧ PO = 12 * real.sqrt 3 

-- Conditions
variables (P M O S T : Point) (PM PO MO PS : Line)
variables (H_PM : length PM = 6 * real.sqrt 3)
variables (H_PO : length PO = 12 * real.sqrt 3)
variables (H_PS_angle_bisector : is_angle_bisector (angle MPO) PS)
variables (H_S_reflection : reflection PM S T)
variables (H_PO_parallel_MT : parallel PO (line_segment M T))

-- Length of OT to be proven
theorem length_of_OT (H : P_MO 6 * real.sqrt 3 12 * real.sqrt 3) : length (line_segment O T) = 2 * real.sqrt 183 :=
  sorry

end length_of_OT_l611_611423


namespace exists_sum_of_three_l611_611903

theorem exists_sum_of_three (
  S : Finset ℕ
) (hS₁ : S.card = 69) (hS₂ : ∀ x ∈ S, x ≤ 100) :
  ∃ a b c ∈ S, a + b + c ∈ S :=
sorry

end exists_sum_of_three_l611_611903


namespace fishers_tomorrow_l611_611790

-- Definitions based on conditions
def people_every_day : ℕ := 7
def people_every_other_day : ℕ := 8
def people_every_three_days : ℕ := 3
def people_yesterday : ℕ := 12
def people_today : ℕ := 10

-- Theorem to be proved
theorem fishers_tomorrow (people_every_day people_every_other_day people_every_three_days people_yesterday people_today : ℕ) : 
  people_every_day = 7 ∧ 
  people_every_other_day = 8 ∧ 
  people_every_three_days = 3 ∧
  people_yesterday = 12 ∧ 
  people_today = 10 →
  15 := 
by {
  sorry
}

end fishers_tomorrow_l611_611790


namespace apples_total_l611_611056

theorem apples_total (initial_apples : ℕ) (additional_apples : ℕ) (total_apples : ℕ) : 
  initial_apples = 56 → 
  additional_apples = 49 → 
  total_apples = initial_apples + additional_apples → 
  total_apples = 105 :=
by 
  intros h_initial h_additional h_total 
  rw [h_initial, h_additional] at h_total 
  exact h_total

end apples_total_l611_611056


namespace quadratic_root_in_interval_l611_611154

variable (a b c : ℝ)

theorem quadratic_root_in_interval 
  (h : 2 * a + 3 * b + 6 * c = 0) :
  ∃ x : ℝ, 0 < x ∧ x < 1 ∧ a * x^2 + b * x + c = 0 :=
by
  sorry

end quadratic_root_in_interval_l611_611154


namespace two_m_leq_n_plus_k_l611_611878

noncomputable theory

-- Definitions of f, g, two functions defined on integers with |x| ≤ 100
def f : ℤ → ℤ := sorry 
def g : ℤ → ℤ := sorry

-- Definition of the terms m, n, k
def m : ℕ := (∑ x in ((Finset.Icc (-100 : ℤ) 100) : Finset ℤ), ∑ y in ((Finset.Icc (-100 : ℤ) 100) : Finset ℤ), if f x = g y then 1 else 0)
def n : ℕ := (∑ x in ((Finset.Icc (-100 : ℤ) 100) : Finset ℤ), ∑ y in ((Finset.Icc (-100 : ℤ) 100) : Finset ℤ), if f x = f y then 1 else 0)
def k : ℕ := (∑ x in ((Finset.Icc (-100 : ℤ) 100) : Finset ℤ), ∑ y in ((Finset.Icc (-100 : ℤ) 100) : Finset ℤ), if g x = g y then 1 else 0)

theorem two_m_leq_n_plus_k : 2 * m ≤ n + k :=
by sorry

end two_m_leq_n_plus_k_l611_611878


namespace smallest_y_square_l611_611765

theorem smallest_y_square (y n : ℕ) (h1 : y = 10) (h2 : ∀ m : ℕ, (∃ z : ℕ, m * y = z^2) ↔ (m = n)) : n = 10 :=
sorry

end smallest_y_square_l611_611765


namespace unique_solution_eq_l611_611284

theorem unique_solution_eq (x : ℝ) : 
  (x ≠ 0 ∧ x ≠ 5) ∧ (∀ x, (3 * x^3 - 15 * x^2) / (x^2 - 5 * x) = x - 2) 
  → ∃! (x : ℝ), (3 * x ^ 3 - 15 * x ^ 2) / (x^2 - 5 * x) = x - 2 := 
by sorry

end unique_solution_eq_l611_611284


namespace determine_a_l611_611460

theorem determine_a (a : ℝ) (h : {2, 9} = {1 - a, 9}) : a = -1 :=
sorry

end determine_a_l611_611460


namespace minimum_oranges_condition_l611_611131

theorem minimum_oranges_condition (n : ℕ) :
  (∀ i j k : ℕ, i ≠ j → j ≠ k → i ≠ k → i < n → j < n → k < n → 
  let m := 1 in (3 * m / ((n-3) * m) < 0.05) → n ≥ 64) :=
begin
  intros h,
  sorry
end

end minimum_oranges_condition_l611_611131


namespace solve_equation_l611_611228

theorem solve_equation (x : ℝ) : 
  (4 * (1 - x)^2 = 25) ↔ (x = -3 / 2 ∨ x = 7 / 2) := 
by 
  sorry

end solve_equation_l611_611228


namespace impossible_to_relocate_top_left_impossible_to_relocate_top_right_l611_611850

-- Definitions
def chessboard_size := 8
def initial_pos := { x : Fin chessboard_size | x < 3 }
def allowed_jump (start end : (Fin chessboard_size × Fin chessboard_size)) : Prop :=
  abs (start.fst - end.fst) = 2 ∧ start.snd = end.snd ∨
  abs (start.snd - end.snd) = 2 ∧ start.fst = end.fst ∨
  abs ((start.fst - end.fst) * (start.snd - end.snd)) = 1

-- Invariant: Coloring
def is_white (pos : Fin chessboard_size × Fin chessboard_size) : Prop :=
  (pos.fst.val + pos.snd.val) % 2 = 0
def is_black (pos : Fin chessboard_size × Fin chessboard_size) : Prop :=
  (pos.fst.val + pos.snd.val) % 2 = 1

-- Initial color distribution
def initial_white_positions : Fin chessboard_size × Fin chessboard_size → Prop :=
  λ pos, initial_pos pos.fst.val ∧ initial_pos pos.snd.val ∧ is_white pos
def initial_black_positions : Fin chessboard_size × Fin chessboard_size → Prop :=
  λ pos, initial_pos pos.fst.val ∧ initial_pos pos.snd.val ∧ is_black pos

-- Final positions
def final_pos_top_left :=
  { x : Fin chessboard_size | x < 3 }

def final_pos_top_right :=
  { x : Fin chessboard_size | x ≥ chessboard_size - 3 }

-- Final color distribution
def final_white_positions_top_left : Fin chessboard_size × Fin chessboard_size → Prop :=
  λ pos, final_pos_top_left pos.fst.val ∧ final_pos_top_left pos.snd.val ∧ is_white pos
def final_black_positions_top_left : Fin chessboard_size × Fin chessboard_size → Prop :=
  λ pos, final_pos_top_left pos.fst.val ∧ final_pos_top_left pos.snd.val ∧ is_black pos

def final_white_positions_top_right : Fin chessboard_size × Fin chessboard_size → Prop :=
  λ pos, final_pos_top_right pos.fst.val ∧ final_pos_top_left pos.snd.val ∧ is_white pos
def final_black_positions_top_right : Fin chessboard_size × Fin chessboard_size → Prop :=
  λ pos, final_pos_top_right pos.fst.val ∧ final_pos_top_left pos.snd.val ∧ is_black pos

theorem impossible_to_relocate_top_left :
  ¬ ∃ pos : (Fin chessboard_size × Fin chessboard_size) → Prop,
  (∀ p, initial_white_positions p ↔ pos p) ∧
  (∀ p, initial_black_positions p ↔ pos p) ∧
  (∀ p, allowed_jump p (pos p)) ∧
  (∀ p, pos p = final_white_positions_top_left p ∨ pos p = final_black_positions_top_left p) := 
sorry

theorem impossible_to_relocate_top_right :
  ¬ ∃ pos : (Fin chessboard_size × Fin chessboard_size) → Prop,
  (∀ p, initial_white_positions p ↔ pos p) ∧
  (∀ p, initial_black_positions p ↔ pos p) ∧
  (∀ p, allowed_jump p (pos p)) ∧
  (∀ p, pos p = final_white_positions_top_right p ∨ pos p = final_black_positions_top_right p) := 
sorry

end impossible_to_relocate_top_left_impossible_to_relocate_top_right_l611_611850


namespace Marla_bottle_caps_per_day_l611_611437

noncomputable def lizard_to_bottle_caps := 8
noncomputable def lizards_to_gallons_of_water := 5 / 3
noncomputable def horse_to_gallons_of_water := 80
noncomputable def daily_expense := 4
noncomputable def days := 24
noncomputable def horse_to_bottle_caps := (horse_to_gallons_of_water / (lizards_to_gallons_of_water)) * lizard_to_bottle_caps

def total_needed_bottle_caps := horse_to_bottle_caps + daily_expense * days
def caps_per_day := total_needed_bottle_caps / days

theorem Marla_bottle_caps_per_day : caps_per_day = 20 := by
  sorry

end Marla_bottle_caps_per_day_l611_611437


namespace time_against_walkway_l611_611615

/-- The time it takes for a person to walk 100 meters against the direction of a moving walkway
    is approximately 150 seconds, given the following conditions:
    1. The walkway is 100 meters long.
    2. Assisted by the walkway, it takes the person 25 seconds to walk from one end to the other.
    3. If the walkway stops moving, it takes 42.857142857142854 seconds to walk 100 meters.
    4. The person's walking speed is constant.
    5. The walkway's speed is constant. -/
theorem time_against_walkway (walkway_length : ℕ)
    (t_with : ℕ)
    (t_without : real)
    (v_p : real)
    (v_w : real) :
    walkway_length = 100 →
    t_with = 25 →
    t_without = 42.857142857142854 →
    v_p = 100 / t_without →
    v_p + v_w = 4 →
    (v_p - v_w) > 0 →
    100 / (v_p - v_w) ≈ 150 := 
by
  intros h₁ h₂ h₃ h₄ h₅ h₆
  have h₇ : v_w = 4 - v_p, from calc
    v_w = 4 - v_p : by exactly h₅
  have h₈ : v_p = 100 / 42.857142857142854, from calc
    v_p = 100 / 42.857142857142854 : by exactly h₄
  have h₉ : v_p ≈ 2.333333333333333, from sorry
  have h₁₀ : v_w ≈ 1.666666666666667, from sorry
  have h₁₁ : v_p - v_w ≈ 0.666666666666666, from sorry
  have t_against : real := 100 / (v_p - v_w)
  show t_against ≈ 150 from sorry

end time_against_walkway_l611_611615


namespace find_three_numbers_l611_611673

theorem find_three_numbers (x y z : ℝ)
  (h1 : x - y = (1 / 3) * z)
  (h2 : y - z = (1 / 3) * x)
  (h3 : z - 10 = (1 / 3) * y) :
  x = 45 ∧ y = 37.5 ∧ z = 22.5 :=
by
  sorry

end find_three_numbers_l611_611673


namespace find_vector_exists_l611_611279

def vector_parallel_to_line (t : ℝ) (x y : ℝ) : Prop :=
  x = 5 * t + 1 ∧ y = 2 * t + 3

theorem find_vector_exists :
  ∃ (a b : ℝ), 
    (∃ t : ℝ, vector_parallel_to_line t (-39) (-13)) ∧ 
    (a, b) = (-39, -13) ∧ 
    ∃ k : ℝ, (a, b) = (3 * k, k) :=
begin
  sorry
end

end find_vector_exists_l611_611279


namespace find_angle_C_l611_611418

variable (A B C : ℝ)
variable (m n : ℝ × ℝ)
variable (π : ℝ)
variable [RealAngle A] [RealAngle B] [RealAngle C]

-- Conditions
def triangle_interior_angles := A + B + C = π
def vector_m := m = (sqrt 3 * sin A, sin B)
def vector_n := n = (cos B, sqrt 3 * cos A)
def dot_product_m_n := m.1 * n.1 + m.2 * n.2 = 1 + cos (A + B)

-- Theorem statement
theorem find_angle_C (h1 : triangle_interior_angles) 
                     (h2 : vector_m)
                     (h3 : vector_n)
                     (h4 : dot_product_m_n) : 
  C = 2 * π / 3 := sorry

end find_angle_C_l611_611418


namespace square_perimeter_l611_611247

theorem square_perimeter : ∀ (s : ℕ), s = 12 → 4 * s = 48 :=
by
  intro s h
  rw h
  norm_num

end square_perimeter_l611_611247


namespace f_2011_eq_neg1_l611_611654

-- Assume we have an odd function f : ℝ → ℝ defined on the reals
axiom f : ℝ → ℝ

-- Condition that f is odd: ∀ x ∈ ℝ, f(-x) = -f(x)
axiom f_odd : ∀ x : ℝ, f(-x) = -f(x)

-- Condition that f satisfies: ∀ x ∈ [-1, 1], f(1 + x) = f(1 - x)
axiom f_symmetry : ∀ x : ℝ, x ∈ set.Icc (-1 : ℝ) 1 → f (1 + x) = f (1 - x)

-- Condition specifying the value of f for x in [-1, 1]
axiom f_defined : ∀ x : ℝ, x ∈ set.Icc (-1 : ℝ) 1 → f x = x^3

-- The theorem stating what we need to prove
theorem f_2011_eq_neg1 : f 2011 = -1 :=
by
  sorry

end f_2011_eq_neg1_l611_611654


namespace imaginary_part_conjugate_l611_611716

theorem imaginary_part_conjugate :
  ∃ Z : ℂ, (Z = (-2 + λIm) / (i ^ 2018)) →
    let conj_Z := conj(Z) in
    Im conj_Z = 1 :=
by
  sorry

end imaginary_part_conjugate_l611_611716


namespace particular_solution_of_differential_eq_l611_611670

theorem particular_solution_of_differential_eq :
  ∃ y : ℝ → ℝ, (∀ x, 2 * y x * (1 / (x + 1)) = y' x) ∧ y 1 = 4 ∧ ∀ x, y x = (1 + x)^2 :=
sorry

end particular_solution_of_differential_eq_l611_611670


namespace at_most_2012_dots_without_arrows_l611_611199

theorem at_most_2012_dots_without_arrows
  (dots : ℕ → Prop)
  (arrows : ℕ → ℕ)
  (injective_arrows : Function.Injective arrows)
  (arrow_distance : ∀ n, |arrows n - n| ≤ 1006) :
  ∃ (N : ℕ), N ≤ 2012 ∧ ∀ m, dots m → ∃ n, arrows n = m → n ∨ ∃ D, dots D ∧ D = m - N := 
begin
  sorry
end

end at_most_2012_dots_without_arrows_l611_611199


namespace tan_subtraction_inequality_l611_611488

theorem tan_subtraction_inequality (x y : ℝ) 
  (hx : 0 < x ∧ x < (π / 2)) 
  (hy : 0 < y ∧ y < (π / 2)) 
  (h : Real.tan x = 3 * Real.tan y) : 
  x - y ≤ π / 6 ∧ (x - y = π / 6 ↔ (x = π / 3 ∧ y = π / 6)) := 
sorry

end tan_subtraction_inequality_l611_611488


namespace num_distinct_log_values_is_21_l611_611268

open Nat

noncomputable def num_distinct_log_values : ℕ :=
  let numbers := {1, 2, 3, 4, 9, 18}
  let possible_pairs := { (a, b) | a ∈ numbers ∧ b ∈ numbers ∧ a ≠ b }
  let log_values := { log b a : ℝ | (a, b) ∈ possible_pairs ∧ b > 0 ∧ b ≠ 1 }
  log_values.to_finset.card

theorem num_distinct_log_values_is_21 :
  num_distinct_log_values = 21 := by
  sorry

end num_distinct_log_values_is_21_l611_611268


namespace oranges_cost_l611_611962

def cost_for_multiple_dozens (price_per_dozen: ℝ) (dozens: ℝ) : ℝ := 
    price_per_dozen * dozens

theorem oranges_cost (price_for_4_dozens: ℝ) (price_for_5_dozens: ℝ) :
  price_for_4_dozens = 28.80 →
  price_for_5_dozens = cost_for_multiple_dozens (28.80 / 4) 5 →
  price_for_5_dozens = 36 :=
by
  intros h1 h2
  sorry

end oranges_cost_l611_611962


namespace hank_donates_90_percent_l611_611381

theorem hank_donates_90_percent (x : ℝ) : 
  (100 * x + 0.75 * 80 + 50 = 200) → (x = 0.9) :=
by
  intro h
  sorry

end hank_donates_90_percent_l611_611381


namespace add_and_round_l611_611256

theorem add_and_round :
  Float.round (123.456 + 78.9102) 2 = 202.37 := 
sorry

end add_and_round_l611_611256


namespace minimum_oranges_condition_l611_611127

theorem minimum_oranges_condition (n : ℕ) :
  (∀ i j k : ℕ, i ≠ j → j ≠ k → i ≠ k → i < n → j < n → k < n → 
  let m := 1 in (3 * m / ((n-3) * m) < 0.05) → n ≥ 64) :=
begin
  intros h,
  sorry
end

end minimum_oranges_condition_l611_611127


namespace Tim_driving_hours_l611_611965

theorem Tim_driving_hours (D T : ℕ) (h1 : T = 2 * D) (h2 : D + T = 15) : D = 5 :=
by
  sorry

end Tim_driving_hours_l611_611965


namespace find_number_l611_611552

theorem find_number (x : ℤ) (h : 2 * x + 5 = 17) : x = 6 := 
by
  sorry

end find_number_l611_611552


namespace simplify_tangent_expression_l611_611498

theorem simplify_tangent_expression :
  (1 + Real.tan (15 * Real.pi / 180)) / (1 - Real.tan (15 * Real.pi / 180)) = Real.sqrt 3 :=
by
  sorry

end simplify_tangent_expression_l611_611498


namespace stack_of_pipes_height_l611_611546

theorem stack_of_pipes_height (d : ℝ) (r : ℝ) (h : ℝ) :
    d = 8 → r = d / 2 → h = r + r * sqrt(3) + r →
    h = 8 + 4 * sqrt(3) :=
by
  intro d_eq_8 r_eq_half_d h_eq_sum
  rw [d_eq_8, r_eq_half_d] at *
  norm_num at *
  exact h_eq_sum

end stack_of_pipes_height_l611_611546


namespace triangle_longest_side_l611_611164

theorem triangle_longest_side 
  (x : ℝ)
  (h1 : 7 + (x + 4) + (2 * x + 1) = 36) :
  2 * x + 1 = 17 := by
  sorry

end triangle_longest_side_l611_611164


namespace solve_for_x_l611_611403

theorem solve_for_x (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 7 * x^2 + 14 * x * y = x^3 + 3 * x^2 * y) : x = 7 :=
by
  sorry

end solve_for_x_l611_611403


namespace asymptote_of_hyperbola_l611_611868

theorem asymptote_of_hyperbola (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) (P : ℝ × ℝ)
  (H : P.1^2 / a^2 - P.2^2 / b^2 = 1) (angle_condition : angle F1 P F2 = 60)
  (distance_condition : |OP| = √7 * a) :
  equation_of_asymptote = √2 * x ± y = 0 := 
sorry

end asymptote_of_hyperbola_l611_611868


namespace parking_cost_l611_611932

def total_cost (x : ℝ) : ℝ := 12 + 7 * x

def average_cost (total : ℝ) (hours : ℝ) : ℝ := total / hours

theorem parking_cost (x : ℝ) (h_avg : average_cost (total_cost x) 9 = 2.6944444444444446) : x = 1.75 :=
by
  sorry

end parking_cost_l611_611932


namespace fishing_problem_l611_611781

theorem fishing_problem
  (everyday : ℕ)
  (every_other_day : ℕ)
  (every_three_days : ℕ)
  (yesterday_fishing : ℕ)
  (today_fishing : ℕ)
  (h_everyday : everyday = 7)
  (h_every_other_day : every_other_day = 8)
  (h_every_three_days : every_three_days = 3)
  (h_yesterday_fishing : yesterday_fishing = 12)
  (h_today_fishing : today_fishing = 10) :
  (every_three_days + everyday + (every_other_day - (yesterday_fishing - everyday))) = 15 := by
  sorry

end fishing_problem_l611_611781


namespace fishing_tomorrow_l611_611815

theorem fishing_tomorrow (seven_every_day eight_every_other_day three_every_three_days twelve_yesterday ten_today : ℕ)
  (h1 : seven_every_day = 7)
  (h2 : eight_every_other_day = 8)
  (h3 : three_every_three_days = 3)
  (h4 : twelve_yesterday = 12)
  (h5 : ten_today = 10) :
  (seven_every_day + (eight_every_other_day - (twelve_yesterday - seven_every_day)) + three_every_three_days) = 15 :=
by
  sorry

end fishing_tomorrow_l611_611815


namespace arc_length_of_circle_l611_611424

theorem arc_length_of_circle 
  (diameter : ℝ) 
  (angle_deg : ℝ) 
  (r : ℝ := diameter / 2) 
  (angle_rad : ℝ := angle_deg * Real.pi / 180) : 
  diameter = 4 ∧ angle_deg = 72 → 
  let l := r * angle_rad in
  l = (4 * Real.pi) / 5 :=
by
  intros h
  cases h with h1 h2
  have r_def : r = 2 := by rw [h1]; norm_num
  have angle_rad_def : angle_rad = (2 * Real.pi) / 5 := by rw [h2]; norm_num; field_simp
  rw [r_def, angle_rad_def]
  norm_num
  sorry

end arc_length_of_circle_l611_611424


namespace min_value_expr_l611_611458

theorem min_value_expr (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ s > 0, s = a + b + c ∧ (a^2 + b^2 + c^2 + 1 / (a + b + c)^3) = (2 * (3 : ℝ)^(2/5)) / 3 + (3 : ℝ)^(1/3) :=
by {
  let s := a + b + c,
  use s,
  split,
  {
    sorry,  -- proof that s > 0
  },
  split,
  {
    sorry,  -- proof that s = a + b + c
  },
  {
    sorry,  -- proof that a^2 + b^2 + c^2 + 1 / (a + b + c)^3 = (2 * (3 : ℝ)^(2/5)) / 3 + (3 : ℝ)^(1/3)
  }
}

end min_value_expr_l611_611458


namespace find_functional_relationship_price_for_profit_1000_maximum_profit_l611_611598

variables 
  (x y w : ℝ)
  (purchase_price : ℝ := 20)
  (data1 : x = 25 → y = 110)
  (data2 : x = 30 → y = 100)
  (k b : ℝ)
  (profit : ℝ := (x - purchase_price) * (y))

-- Define the linear relationship function
def linear_relation : Prop := y = k * x + b

-- 1. Prove the functional relationship
theorem find_functional_relationship (h1 : linear_relation) : k = -2 ∧ b = 160 :=
  sorry

-- 2. Prove the price to achieve a profit of 1000 yuan
theorem price_for_profit_1000 
  (h1 : linear_relation) 
  (hyp : profit = 1000) : x = 30 :=
  sorry

-- 3. Prove the maximum profit and the price at which it is achieved
theorem maximum_profit 
  (h1 : linear_relation) 
  (h2 : 20 ≤ x ∧ x ≤ 40)
  : x = 40 ∧ profit = 1600 :=
  sorry

end find_functional_relationship_price_for_profit_1000_maximum_profit_l611_611598


namespace solar_panels_l611_611293

theorem solar_panels (total_homes panels_per_home total_shortfall : ℕ) (h_total_homes : total_homes = 20) (h_panels_per_home : panels_per_home = 10) (h_total_shortfall : total_shortfall = 50) :
  (total_homes * panels_per_home - total_shortfall) / panels_per_home = 15 :=
by
  rw [h_total_homes, h_panels_per_home, h_total_shortfall]
  norm_num
  sorry

end solar_panels_l611_611293


namespace fishing_tomorrow_l611_611811

theorem fishing_tomorrow (seven_every_day eight_every_other_day three_every_three_days twelve_yesterday ten_today : ℕ)
  (h1 : seven_every_day = 7)
  (h2 : eight_every_other_day = 8)
  (h3 : three_every_three_days = 3)
  (h4 : twelve_yesterday = 12)
  (h5 : ten_today = 10) :
  (seven_every_day + (eight_every_other_day - (twelve_yesterday - seven_every_day)) + three_every_three_days) = 15 :=
by
  sorry

end fishing_tomorrow_l611_611811


namespace sum_F_n_eq_4087523_l611_611313

def F (n : ℕ) : ℕ :=
  if n > 1 then
    let sols := (λ x : ℝ, sin x = sin (n * x));
    let I := set.Icc 0 (2 * real.pi);
    finset.card {x ∈ I | sols x}
  else
    0

theorem sum_F_n_eq_4087523 : (finset.range 2022).sum (λ i, F (i + 2)) = 4087523 :=
by sorry

end sum_F_n_eq_4087523_l611_611313


namespace eval_definite_integral_l611_611657

noncomputable def definite_integral : ℝ :=
  ∫ x in 0..1, sqrt (1 - (x - 1) ^ 2) - x

theorem eval_definite_integral : definite_integral = (π - 2) / 4 :=
by
  sorry

end eval_definite_integral_l611_611657


namespace complex_number_problem_l611_611752

-- Define the complex number z
def z : ℂ := complex.mk 0 (sqrt 2)

-- State the conjecture
theorem complex_number_problem : z^4 = 4 :=
by {
  sorry
}

end complex_number_problem_l611_611752


namespace joan_gave_away_kittens_l611_611445

-- Definitions based on conditions in the problem
def original_kittens : ℕ := 8
def kittens_left : ℕ := 6

-- Mathematical statement to be proved
theorem joan_gave_away_kittens : original_kittens - kittens_left = 2 :=
by
  sorry

end joan_gave_away_kittens_l611_611445


namespace minimum_oranges_condition_l611_611129

theorem minimum_oranges_condition (n : ℕ) :
  (∀ i j k : ℕ, i ≠ j → j ≠ k → i ≠ k → i < n → j < n → k < n → 
  let m := 1 in (3 * m / ((n-3) * m) < 0.05) → n ≥ 64) :=
begin
  intros h,
  sorry
end

end minimum_oranges_condition_l611_611129


namespace eventually_periodic_sequence_l611_611058

theorem eventually_periodic_sequence (A B C : Point) (C_on_perp_bisector_AB : is_perp_bisector A B C) :
  (∃ k p : ℕ, ∀ n ≥ k, C_n (A B C) (n + p) = C_n (A B C) n) ↔ (∃ q : ℚ, x_1 (A B C) ∈ q) :=
sorry

end eventually_periodic_sequence_l611_611058


namespace positive_integer_solution_l611_611297

theorem positive_integer_solution (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x ≤ y ∧ y ≤ z) (h_eq : 5 * (x * y + y * z + z * x) = 4 * x * y * z) :
  (x = 2 ∧ y = 5 ∧ z = 10) ∨ (x = 2 ∧ y = 4 ∧ z = 20) :=
sorry

end positive_integer_solution_l611_611297


namespace simplify_exponent_multiplication_l611_611919

theorem simplify_exponent_multiplication (x : ℝ) : x^5 * x^3 = x^8 :=
by sorry

end simplify_exponent_multiplication_l611_611919


namespace fishing_tomorrow_l611_611801

theorem fishing_tomorrow (every_day_fishers every_other_day_fishers every_three_days_fishers fishers_yesterday fishers_today : ℕ) :
  (every_day_fishers = 7) →
  (every_other_day_fishers = 8) →
  (every_three_days_fishers = 3) →
  (fishers_yesterday = 12) →
  (fishers_today = 10) →
  (every_three_days_fishers + every_day_fishers + (every_other_day_fishers - (fishers_yesterday - every_day_fishers)) = 15) :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  simp
  sorry

end fishing_tomorrow_l611_611801


namespace AB_parallel_EF_l611_611436

-- Define the essential geometrical setup
variables (O A B M C D E F : Point)
variables (circle : Circle O)
variables (tangent : Tangent A circle B)
variables (secant : Secant M circle C D)
variables (midpoint : Midpoint A B M)
variables (intersection1 : IntersectLineCircle A C circle E)
variables (intersection2 : IntersectLineCircle A D circle F)

-- Define the proof problem as a theorem
theorem AB_parallel_EF :
  is_parallel (line_seg A B) (line_seg E F) := by
  sorry

end AB_parallel_EF_l611_611436


namespace change_of_b_l611_611931

variable {t b1 b2 C C_new : ℝ}

theorem change_of_b (hC : C = t * b1^4) 
                   (hC_new : C_new = 16 * C) 
                   (hC_new_eq : C_new = t * b2^4) : 
                   b2 = 2 * b1 :=
by
  sorry

end change_of_b_l611_611931


namespace cost_per_pound_beef_l611_611088

-- Definitions based on the conditions
def total_grocery_cost : ℝ := 16
def total_chicken_cost : ℝ := 3
def oil_cost : ℝ := 1
def pounds_beef : ℝ := 3

-- Mathematically equivalent proof problem
theorem cost_per_pound_beef :
  let total_remaining_cost := total_grocery_cost - total_chicken_cost,
      total_beef_cost := total_remaining_cost - oil_cost,
      cost_per_pound := total_beef_cost / pounds_beef
  in cost_per_pound = 4 := 
by 
  -- This proof is left as an exercise to the reader.
  sorry

end cost_per_pound_beef_l611_611088


namespace no_such_natural_number_l611_611664

theorem no_such_natural_number (m p q r : ℕ) (hp : p.prime) (hq : q.prime) (hr : r.prime) :
  (p-1) ∣ m ∧ (q*r-1) ∣ m ∧ ¬(q-1) ∣ m ∧ ¬(r-1) ∣ m ∧ ¬3 ∣ (q + r) ∧ m = p * q * r → false :=
sorry

end no_such_natural_number_l611_611664


namespace least_positive_integer_reducible_fraction_l611_611303

theorem least_positive_integer_reducible_fraction :
  ∃ n : ℕ, n > 0 ∧ (∀ m : ℕ, m > 0 ∧ (¬ ∃ k : ℕ, n = k * m ∧ m > 1) → m = n) ∧ (int.gcd (n - 17) (7 * n + 5) > 1) ∧ n = 48 :=
sorry

end least_positive_integer_reducible_fraction_l611_611303


namespace total_amount_l611_611983

theorem total_amount (x y z : ℕ) (xy : ℕ) (zy : ℕ) (x_share y_share z_share total : ℕ) : 
  (for all x : ℕ, y_share = 45 * x / 100 <-> true) →
  (for all x : ℕ, z_share = 50 * x / 100 <-> true) →
  y_share = 1800 →
  total = x_share + y_share + z_share →
  total = 7800 :=
by
  sorry

end total_amount_l611_611983


namespace projection_AC_on_AB_l611_611322

-- Define the points
structure Point3D where
  x : ℝ 
  y : ℝ 
  z : ℝ 

def A : Point3D := ⟨1, -2, 1⟩
def B : Point3D := ⟨1, -5, 4⟩
def C : Point3D := ⟨2, 3, 4⟩

-- Define vector subtraction
def vectorSub (p1 p2 : Point3D) : Point3D :=
  ⟨p1.x - p2.x, p1.y - p2.y, p1.z - p2.z⟩

-- Define dot product
def dotProduct (v1 v2 : Point3D) : ℝ :=
  v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

-- Define magnitude
def magnitude (v : Point3D) : ℝ :=
  Real.sqrt (v.x^2 + v.y^2 + v.z^2)

-- Define scalar multiplication
def scalarMul (k : ℝ) (v : Point3D) : Point3D :=
  ⟨k * v.x, k * v.y, k * v.z⟩

-- Define projection calculation
def projection (v1 v2 : Point3D) : Point3D :=
  let scaleFactor : ℝ := (dotProduct v1 v2) / (magnitude v2) / (magnitude v2)
  scalarMul scaleFactor v2

-- The statement we want to prove
theorem projection_AC_on_AB : 
  projection (vectorSub C A) (vectorSub B A) = ⟨0, 1, -1⟩ := 
by 
  -- placeholder for the proof steps
  sorry

end projection_AC_on_AB_l611_611322


namespace percentage_decrease_l611_611848

theorem percentage_decrease 
  (P0 : ℕ) (P2 : ℕ) (H0 : P0 = 10000) (H2 : P2 = 9600) 
  (P1 : ℕ) (H1 : P1 = P0 + (20 * P0) / 100) :
  ∃ (D : ℕ), P2 = P1 - (D * P1) / 100 ∧ D = 20 :=
by
  sorry

end percentage_decrease_l611_611848


namespace tiling_properties_l611_611563

theorem tiling_properties (x y z : ℕ) (hx : 3 < x) (hy : 3 < y) (hz : 3 < z) :
  (x - 2) * y * z + (y - 2) * x * z + (z - 2) * x * y = 2 * x * y * z → 
  (1.toFloat / x.toFloat + 1.toFloat / y.toFloat + 1.toFloat / z.toFloat = 0.5) :=
by
  sorry

end tiling_properties_l611_611563


namespace parallel_lines_slope_l611_611029

theorem parallel_lines_slope (a : ℝ) : 
  let m1 := - (a / 2)
  let m2 := 3
  ax + 2 * y + 2 = 0 ∧ 3 * x - y - 2 = 0 → m1 = m2 → a = -6 := 
by
  intros
  sorry

end parallel_lines_slope_l611_611029


namespace fishing_problem_l611_611784

theorem fishing_problem
  (everyday : ℕ)
  (every_other_day : ℕ)
  (every_three_days : ℕ)
  (yesterday_fishing : ℕ)
  (today_fishing : ℕ)
  (h_everyday : everyday = 7)
  (h_every_other_day : every_other_day = 8)
  (h_every_three_days : every_three_days = 3)
  (h_yesterday_fishing : yesterday_fishing = 12)
  (h_today_fishing : today_fishing = 10) :
  (every_three_days + everyday + (every_other_day - (yesterday_fishing - everyday))) = 15 := by
  sorry

end fishing_problem_l611_611784


namespace find_second_type_material_l611_611163

section CherylMaterials

noncomputable def first_type_material : ℚ := 2 / 9
noncomputable def material_leftover : ℚ := 2 / 9
noncomputable def material_used : ℚ := 125 / 1000 -- Converting 0.125 to rational

theorem find_second_type_material :
  ∃ (second_type_material : ℚ), 
    second_type_material = material_used :=
begin
  use material_used,
  sorry
end

end CherylMaterials

end find_second_type_material_l611_611163


namespace valid_numbers_l611_611237

def is_valid_100_digit_number (N N' : ℕ) (k m n : ℕ) (a : ℕ) : Prop :=
  0 ≤ a ∧ a < 100 ∧ 0 ≤ m ∧ m < 10^k ∧ 
  N = m + 10^k * a + 10^(k + 2) * n ∧ 
  N' = m + 10^k * n ∧
  N = 87 * N'

theorem valid_numbers : ∀ (N : ℕ), (∃ N' k m n a, is_valid_100_digit_number N N' k m n a) →
  N = 435 * 10^97 ∨ 
  N = 1305 * 10^96 ∨ 
  N = 2175 * 10^96 ∨ 
  N = 3045 * 10^96 :=
by
  sorry

end valid_numbers_l611_611237


namespace candy_necklaces_per_pack_l611_611286

theorem candy_necklaces_per_pack (packs_total packs_opened packs_left candies_left necklaces_per_pack : ℕ) 
  (h_total : packs_total = 9) 
  (h_opened : packs_opened = 4) 
  (h_left : packs_left = packs_total - packs_opened) 
  (h_candies_left : candies_left = 40) 
  (h_necklaces_per_pack : candies_left = packs_left * necklaces_per_pack) :
  necklaces_per_pack = 8 :=
by
  -- Proof goes here
  sorry

end candy_necklaces_per_pack_l611_611286


namespace geometric_sequence_tan_sum_l611_611419

theorem geometric_sequence_tan_sum
  (A B C : ℝ)
  (a b c : ℝ)
  (h1 : b^2 = a * c)
  (h2 : Real.tan B = 3/4):
  1 / Real.tan A + 1 / Real.tan C = 5 / 3 := 
by
  sorry

end geometric_sequence_tan_sum_l611_611419


namespace find_angle_l611_611705

variables {A B C G : Type} -- points
variables {a b c : ℝ} -- lengths of sides opposite their respective angles
variables [inner_product_space ℝ V] (GA GB GC : V) -- position vectors

-- Condition: G is the centroid of triangle ABC.
def is_centroid (G A B C : V) : Prop :=
  (GA + GB + GC = 0)

-- Condition: given vector equation.
def given_vector_eq (a b c : ℝ) (GA GB GC : V) : Prop :=
  a • GA + b • GB + (sqrt 3 / 3) * c • GC = 0

-- Theorem statement
theorem find_angle (h_centroid : is_centroid G A B C)
                  (h_vector_eq : given_vector_eq a b c GA GB GC) :
  angle A = π / 6 :=
  sorry

end find_angle_l611_611705


namespace fishing_tomorrow_l611_611824

theorem fishing_tomorrow (yesterday_fishers today_fishers : ℕ)
  (every_day_fishers every_other_day_fishers every_three_days_fishers : ℕ)
  (total_population : ℕ):
  yesterday_fishers = 12 → 
  today_fishers = 10 → 
  every_day_fishers = 7 → 
  every_other_day_fishers = 8 → 
  every_three_days_fishers = 3 → 
  total_population = yesterday_fishers + today_fishers + (total_population - (every_day_fishers + every_other_day_fishers + every_three_days_fishers)) →
  ∃ tomorrow_fishers : ℕ, tomorrow_fishers = 15 :=
by {
  -- This is a statement definition, the proof is not required and thus marked as "sorry:"
  sorry
}

end fishing_tomorrow_l611_611824


namespace correct_choices_l611_611559

variable (atom molecule cation anion : Type)
variable (num_protons num_electrons : Type → ℕ)
variable (has_same_protons_and_electrons : Type → Type → Prop) 

-- Conditions: Particles should have the same number of protons and electrons
axiom same_protons_electrons : ∀ x y, has_same_protons_and_electrons x y ↔ num_protons x = num_protons y ∧ num_electrons x = num_electrons y

-- Prove that under the given conditions, the following particles have the same number of protons and electrons
theorem correct_choices :
  (has_same_protons_and_electrons atom (Type)) ∧ -- ① Two different atoms
  (has_same_protons_and_electrons atom molecule) ∧ -- ③ One atom and one molecule
  (has_same_protons_and_electrons (Type) (Type)) ∧ -- ⑤ Two different molecules
  (has_same_protons_and_electrons cation cation) ∧ -- ⑦ Two different cations
  (has_same_protons_and_electrons anion anion) -- ⑧ Two different anions
  :=
  sorry

end correct_choices_l611_611559


namespace min_value_of_f_l611_611566

noncomputable def f (x : ℝ) : ℝ := 7 * x^2 - 28 * x + 1425

theorem min_value_of_f : ∃ (x : ℝ), f x = 1397 :=
by
  sorry

end min_value_of_f_l611_611566


namespace trigonometric_identity_l611_611012

theorem trigonometric_identity (α : ℝ) (h : (1 + Real.tan α) / (1 - Real.tan α) = 2012) : 
  (1 / Real.cos (2 * α)) + Real.tan (2 * α) = 2012 := 
by
  -- This will be the proof body which we omit with sorry
  sorry

end trigonometric_identity_l611_611012


namespace simplify_expression_l611_611497

theorem simplify_expression (y : ℝ) : (3 * y + 4 * y + 5 * y + 7) = (12 * y + 7) :=
by
  sorry

end simplify_expression_l611_611497


namespace number_of_possible_A2_eq_one_l611_611066

noncomputable def unique_possible_A2 (A : Matrix (Fin 2) (Fin 2) ℝ) : Prop :=
  (A^4 = 0) → (A^2 = 0)

theorem number_of_possible_A2_eq_one (A : Matrix (Fin 2) (Fin 2) ℝ) :
  unique_possible_A2 A :=
by 
  sorry

end number_of_possible_A2_eq_one_l611_611066


namespace sphere_surface_area_relationship_l611_611728

variables (R1 R2 R3 S1 S2 S3 : ℝ)
variables (h1 : R1 + 2 * R2 = 3 * R3)
variables (hS1 : S1 = 4 * Real.pi * R1^2)
variables (hS2 : S2 = 4 * Real.pi * R2^2)
variables (hS3 : S3 = 4 * Real.pi * R3^2)

theorem sphere_surface_area_relationship 
  (h1 : R1 + 2 * R2 = 3 * R3) 
  (hS1 : S1 = 4 * Real.pi * R1^2) 
  (hS2 : S2 = 4 * Real.pi * R2^2) 
  (hS3 : S3 = 4 * Real.pi * R3^2) : 
  Real.sqrt S1 + 2 * Real.sqrt S2 = 3 * Real.sqrt S3 := 
begin
  sorry
end

end sphere_surface_area_relationship_l611_611728


namespace correct_average_l611_611584

theorem correct_average (n : Nat) (incorrect_avg correct_mark incorrect_mark : ℝ) 
  (h1 : n = 30) (h2 : incorrect_avg = 60) (h3 : correct_mark = 15) (h4 : incorrect_mark = 90) :
  (incorrect_avg * n - incorrect_mark + correct_mark) / n = 57.5 :=
by
  sorry

end correct_average_l611_611584


namespace center_of_hyperbola_l611_611668

theorem center_of_hyperbola :
  (∃ h k : ℝ, ∀ x y : ℝ, (3*y + 3)^2 / 49 - (2*x - 5)^2 / 9 = 1 ↔ x = h ∧ y = k) → 
  h = 5 / 2 ∧ k = -1 :=
by
  sorry

end center_of_hyperbola_l611_611668


namespace minimum_number_of_oranges_l611_611134

noncomputable def minimum_oranges_picked : ℕ :=
  let n : ℕ := 64 in n

theorem minimum_number_of_oranges (n : ℕ) : 
  (∀ (m_i m_j m_k : ℝ) (remaining_masses : Finset ℝ),
    remaining_masses.card = n - 3 →
    (m_i + m_j + m_k) < 0.05 * (∑ m in remaining_masses, m)) →
  n ≥ minimum_oranges_picked :=
by {
  sorry
}

end minimum_number_of_oranges_l611_611134


namespace find_A_from_equation_l611_611958

variable (A B C D : ℕ)
variable (diff_numbers : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
variable (eq1 : A * 1000 + B * 100 + 82 - 900 + C * 10 + 9 = 4000 + 900 + 30 + D)

theorem find_A_from_equation (A B C D : ℕ) (diff_numbers : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) 
  (eq1 : A * 1000 + B * 100 + 82 - (900 + C * 10 + 9) = 4000 + 900 + 30 + D) : A = 5 :=
by sorry

end find_A_from_equation_l611_611958


namespace fraction_power_minus_one_l611_611267

theorem fraction_power_minus_one :
  (5 / 3) ^ 4 - 1 = 544 / 81 := 
by
  sorry

end fraction_power_minus_one_l611_611267


namespace total_votes_l611_611580

theorem total_votes (V : ℝ) (h1 : 0.32 * V = 0.32 * V) (h2 : 0.32 * V + 1908 = 0.68 * V) : V = 5300 :=
by
  sorry

end total_votes_l611_611580


namespace length_of_BC_in_triangle_l611_611859

theorem length_of_BC_in_triangle (A B : Real) (BC AB : Real) 
  (h1 : cos (3 * A - B) + sin (A + B) = 2)
  (h2 : AB = 4) : 
  BC = 2 * sqrt (2 - sqrt 2) := 
sorry

end length_of_BC_in_triangle_l611_611859


namespace compare_logarithms_l611_611688

theorem compare_logarithms (a b c : ℝ) (h1 : a = Real.log 2 / Real.log 3) 
                           (h2 : b = (Real.log 2 / Real.log 3)^2) 
                           (h3 : c = Real.log (2/3) / Real.log 4) : c < b ∧ b < a :=
by
  sorry

end compare_logarithms_l611_611688


namespace no_valid_filling_exists_l611_611441

theorem no_valid_filling_exists :
  ¬ ∃ (filling : Fin₁⟦21⟧ → ℕ),
    (∀ (r c : ℕ) (a b : ℕ), 
      r > 1 → c = r - 1 → filling ⟨r, c⟩ = |filling ⟨r - 1, c - 1⟩ - filling ⟨r - 1, c⟩|) →
    (∀ x, 1 ≤ filling x ∧ filling x ≤ 21) →
    ∑ i in Finset.range 21, filling i = 231 :=
sorry

end no_valid_filling_exists_l611_611441


namespace slope_of_line_inclination_l611_611044

theorem slope_of_line_inclination (α : ℝ) (h1 : 0 ≤ α) (h2 : α < 180) 
  (h3 : Real.tan (α * Real.pi / 180) = Real.sqrt 3 / 3) : α = 30 :=
by
  sorry

end slope_of_line_inclination_l611_611044


namespace men_women_arrangement_l611_611999

theorem men_women_arrangement :
  let men := 2
  let women := 4
  let slots := 5
  (Nat.choose slots women) * women.factorial * men.factorial = 240 :=
by
  sorry

end men_women_arrangement_l611_611999


namespace fishing_tomorrow_l611_611813

theorem fishing_tomorrow (seven_every_day eight_every_other_day three_every_three_days twelve_yesterday ten_today : ℕ)
  (h1 : seven_every_day = 7)
  (h2 : eight_every_other_day = 8)
  (h3 : three_every_three_days = 3)
  (h4 : twelve_yesterday = 12)
  (h5 : ten_today = 10) :
  (seven_every_day + (eight_every_other_day - (twelve_yesterday - seven_every_day)) + three_every_three_days) = 15 :=
by
  sorry

end fishing_tomorrow_l611_611813


namespace pure_imaginary_solution_l611_611078

theorem pure_imaginary_solution (a : ℝ) (ha : a + 5 * Complex.I / (1 - 2 * Complex.I) = a + (1 : ℂ) * Complex.I) :
  a = 2 :=
by
  sorry

end pure_imaginary_solution_l611_611078


namespace inequality_true_l611_611013

-- Define the conditions
variables (a b : ℝ) (h : a < b) (hb_neg : b < 0)

-- State the theorem to be proved
theorem inequality_true (ha : a < b) (hb : b < 0) : (|a| / |b| > 1) :=
sorry

end inequality_true_l611_611013


namespace max_regions_with_five_spheres_l611_611972

def b : ℕ → ℕ
| 0       := 1
| (n + 1) := b n + 1 + (n + 1) + (n * (n + 1)) / 2

theorem max_regions_with_five_spheres : b 5 = 47 := 
by {
  have h_base : b 0 = 1 := rfl,
  have h_step : ∀ n, b (n + 1) = b n + 1 + (n + 1) + (n * (n + 1)) / 2,
  -- The proof is omitted here
  sorry
}

end max_regions_with_five_spheres_l611_611972


namespace num_of_valid_numbers_l611_611953

def is_valid_number (n : ℕ) : Prop :=
  let a := n / 10
  let b := n % 10
  a >= 1 ∧ a <= 9 ∧ b >= 0 ∧ b <= 9 ∧ (9 * a) % 10 = 4

theorem num_of_valid_numbers : ∃ n, n = 10 :=
by {
  sorry
}

end num_of_valid_numbers_l611_611953


namespace zoo_peacocks_l611_611543

theorem zoo_peacocks (R P : ℕ) (h1 : R + P = 60) (h2 : 4 * R + 2 * P = 192) : P = 24 :=
by
  sorry

end zoo_peacocks_l611_611543


namespace gold_copper_alloy_ratio_l611_611583

def ratio_gold_copper (G C : ℝ) : ℝ := G / C

theorem gold_copper_alloy_ratio :
  ∀ (G C : ℝ),
    (11 * G + 5 * C) / (G + C) = 8 → 
    ratio_gold_copper G C = 1 :=
by
  intros G C h
  -- Proof is omitted
  sorry

end gold_copper_alloy_ratio_l611_611583


namespace matchstick_problem_l611_611482

-- Defining the figure with 24 matches and the removal of matches to achieve the desired pattern.
structure Figure :=
  (matches : Finset ℕ) -- Representing matchsticks as a finite set of integers.

def initial_figure : Figure := {matches := Finset.range 24}

def is_large_square_and_3_small_squares (f : Figure) : Prop := sorry

-- Two different sets of removals leading to the desired pattern.
def removal_set1 : Finset ℕ := {0, 1, 22, 18}
def removal_set2 : Finset ℕ := {0, 18, 22, 23}

-- The proposition we need to prove.
theorem matchstick_problem :
  ∃ (fig1 fig2 : Figure),
    (fig1.matches = initial_figure.matches \ removal_set1 ∧ fig2.matches = initial_figure.matches \ removal_set2) ∧
    is_large_square_and_3_small_squares fig1 ∧
    is_large_square_and_3_small_squares fig2 :=
sorry

end matchstick_problem_l611_611482


namespace problem_statement_l611_611875

-- Definitions of the conditions
variables (x y z w : ℕ)

-- The proof problem
theorem problem_statement
  (hx : x^3 = y^2)
  (hz : z^4 = w^3)
  (hzx : z - x = 17)
  (hx_pos : x > 0)
  (hy_pos : y > 0)
  (hz_pos : z > 0)
  (hw_pos : w > 0) :
  w - y = 229 :=
sorry

end problem_statement_l611_611875


namespace sum_series_eq_l611_611223

theorem sum_series_eq (n : ℕ) : 
  (∑ k in finset.range n, (k + 1 : ℕ) * 5 ^ 2 * 2 ^ k * 3 ^ k) = (5 * n - 1) * 6 ^ n + 1 :=
sorry

end sum_series_eq_l611_611223


namespace find_dot_product_ad_l611_611871

open Real

noncomputable def vec_a := sorry -- Placeholder for the vector a
noncomputable def vec_b := sorry -- Placeholder for the vector b
noncomputable def vec_c := sorry -- Placeholder for the vector c
noncomputable def vec_d := sorry -- Placeholder for the vector d
noncomputable def vec_e := sorry -- Placeholder for the vector e

-- Define unit vector property
axiom unit_vectors : 
  ∥vec_a∥ = 1 ∧ ∥vec_b∥ = 1 ∧ ∥vec_c∥ = 1 ∧ ∥vec_d∥ = 1 ∧ ∥vec_e∥ = 1

-- Define the dot product conditions
axiom dot_product_conditions :
  vec_a ⬝ vec_b = -1 / 5 ∧
  vec_a ⬝ vec_c = -1 / 5 ∧
  vec_b ⬝ vec_c = -1 / 5 ∧
  vec_b ⬝ vec_d = -1 / 7 ∧
  vec_c ⬝ vec_d = -1 / 7 ∧
  vec_a ⬝ vec_e = -1 / 7

theorem find_dot_product_ad :
  vec_a ⬝ vec_d = -2414 / 2450 :=
begin
  sorry
end

end find_dot_product_ad_l611_611871


namespace fishing_tomorrow_l611_611832

-- Conditions
def every_day_fishers : Nat := 7
def every_other_day_fishers : Nat := 8
def every_three_days_fishers : Nat := 3
def yesterday_fishers : Nat := 12
def today_fishers : Nat := 10

-- Determine the number who will fish tomorrow
def fishers_tomorrow : Nat :=
  let every_day_tomorrow := every_day_fishers
  let every_three_day_tomorrow := every_three_days_fishers
  let every_other_day_yesterday := yesterday_fishers - every_day_fishers
  let every_other_day_tomorrow := every_other_day_fishers - every_other_day_yesterday
  every_day_tomorrow + every_three_day_tomorrow + every_other_day_tomorrow

theorem fishing_tomorrow : fishers_tomorrow = 15 :=
  by
    -- skipping the actual proof with sorry
    sorry

end fishing_tomorrow_l611_611832


namespace find_p_l611_611096

noncomputable def ellipse : set (ℝ × ℝ) :=
  {p | (p.1)^2 / 4 + (p.2)^2 = 1}

def is_focus (F : ℝ × ℝ) : Prop :=
  F = (1, 0)

def point_condition (p : ℝ) : Prop :=
  p > 0

def angle_condition (p A B F: ℝ × ℝ) : Prop :=
  ∀ A B : ℝ × ℝ, A ∈ ellipse → B ∈ ellipse → (A.2 - B.2) ≠ 0 → 
  let AP_slope := (A.2 - F.2) / (A.1 - p.1) in
  let BP_slope := (B.2 - F.2) / (B.1 - p.1) in
  AP_slope = -BP_slope

theorem find_p : ∃ p > 0, 
  (∀ (A B F: ℝ × ℝ), A ∈ ellipse → B ∈ ellipse → (A.2 - B.2) ≠ 0 → 
  let AP_slope := (A.2 - F.2) / (A.1 - p.1) in
  let BP_slope := (B.2 - F.2) / (B.1 - p.1) in
  AP_slope = -BP_slope) :=
sorry

end find_p_l611_611096


namespace find_50th_term_of_sequence_l611_611950

-- Define the sequence
def is_power_of_4_or_sum_of_distinct_powers_of_4 (n : ℕ) : Prop :=
  ∃ (k : ℕ) (a : Fin k → ℕ), (∀ i, a i = 4 ^ i) ∧ (n = ∑ i, a i)

-- Statement of the problem
theorem find_50th_term_of_sequence : 
  ∃ n, (∃ k a, (∀ i, a i = 4 ^ i) ∧ (n = ∑ i in Fin k, a i)) ∧ n = 1284
  ∧ sorted_nth (λ n,  ∃ k a, (∀ i, a i = 4 ^ i) ∧ (n = ∑ i in Fin k, a i)) 50 = 1284 :=
by
  sorry

end find_50th_term_of_sequence_l611_611950


namespace joan_balloons_l611_611052

theorem joan_balloons {B L: ℕ} (hB : B = 9) (hL : L = 2) : B - L = 7 := 
by
  rw [hB, hL]
  exact rfl

end joan_balloons_l611_611052


namespace find_number_division_l611_611231

theorem find_number_division :
  let result := (208 / 100) * 1265 in
  ∃ (x : ℝ), result / x ≈ 438.53333333333336 ∧ x ≈ result / 438.53333333333336 :=
by
  let result := (208 / 100) * 1265
  use result / 438.53333333333336
  split
  sorry
  sorry

end find_number_division_l611_611231


namespace b_seq_formula_c_seq_arithmetic_c_seq_formula_a_seq_formula_sum_S_5_l611_611696

-- Define the sequences a_n, b_n, and c_n along with their properties

-- Definitions
def a_seq (n : ℕ) : ℕ := sorry            -- Define a_n

def S_seq (n : ℕ) : ℕ := sorry            -- Define S_n

def b_seq (n : ℕ) : ℕ := a_seq (n+1) - 2 * a_seq n

def c_seq (n : ℕ) : ℕ := a_seq n / 2^n

-- Conditions
axiom S_n_condition (n : ℕ) : S_seq (n+1) = 4 * a_seq n + 2
axiom a_1_condition : a_seq 1 = 1

-- Goals
theorem b_seq_formula (n : ℕ) : b_seq n = 3 * 2^(n-1) := sorry

theorem c_seq_arithmetic (n : ℕ) : c_seq (n+1) - c_seq n = 3 / 4 := sorry

theorem c_seq_formula (n : ℕ) : c_seq n = (3 * n - 1) / 4 := sorry

theorem a_seq_formula (n : ℕ) : a_seq n = (3 * n - 1) * 2^(n-2) := sorry

theorem sum_S_5 : S_seq 5 = 178 := sorry

end b_seq_formula_c_seq_arithmetic_c_seq_formula_a_seq_formula_sum_S_5_l611_611696


namespace division_of_powers_of_ten_l611_611275

theorem division_of_powers_of_ten : 10^8 / (2 * 10^6) = 50 := by 
  sorry

end division_of_powers_of_ten_l611_611275


namespace sufficient_not_necessary_l611_611703

theorem sufficient_not_necessary (a b : ℝ) (h : b > a ∧ a > 0) : (1 / a > 1 / b) :=
by {
  sorry -- the proof steps are intentionally omitted
}

end sufficient_not_necessary_l611_611703


namespace domain_of_f_l611_611934

def f (x : ℝ) : ℝ := real.sqrt (x - 1) + real.sqrt (1 - x)

theorem domain_of_f : {x : ℝ | 0 ≤ x - 1 ∧ 0 ≤ 1 - x} = {1} :=
by sorry

end domain_of_f_l611_611934


namespace triangle_equilateral_of_constraints_l611_611698

theorem triangle_equilateral_of_constraints {a b c : ℝ}
  (h1 : a^4 = b^4 + c^4 - b^2 * c^2)
  (h2 : b^4 = c^4 + a^4 - a^2 * c^2) : 
  a = b ∧ b = c :=
by 
  sorry

end triangle_equilateral_of_constraints_l611_611698


namespace Tony_can_add_4_pairs_of_underwear_l611_611195

-- Define relevant variables and conditions
def max_weight : ℕ := 50
def w_socks : ℕ := 2
def w_underwear : ℕ := 4
def w_shirt : ℕ := 5
def w_shorts : ℕ := 8
def w_pants : ℕ := 10

def pants : ℕ := 1
def shirts : ℕ := 2
def shorts : ℕ := 1
def socks : ℕ := 3

def total_weight (pants shirts shorts socks : ℕ) : ℕ :=
  pants * w_pants + shirts * w_shirt + shorts * w_shorts + socks * w_socks

def remaining_weight : ℕ :=
  max_weight - total_weight pants shirts shorts socks

def additional_pairs_of_underwear_cannot_exceed : ℕ :=
  remaining_weight / w_underwear

-- Problem statement in Lean
theorem Tony_can_add_4_pairs_of_underwear :
  additional_pairs_of_underwear_cannot_exceed = 4 :=
  sorry

end Tony_can_add_4_pairs_of_underwear_l611_611195


namespace minimum_oranges_l611_611118

theorem minimum_oranges (n : ℕ) (m : ℕ → ℝ) (h : ∀ i j k : ℕ, i < n → j < n → k < n → i ≠ j → i ≠ k → j ≠ k → (m i + m j + m k) < 0.05 * ∑ l in Finset.range n \ {i, j, k}, m l) : n ≥ 64 := 
sorry

end minimum_oranges_l611_611118


namespace prove_equation_l611_611748

theorem prove_equation (x : ℚ) (h : 5 * x - 3 = 15 * x + 21) : 3 * (2 * x + 5) = 3 / 5 :=
by
  sorry

end prove_equation_l611_611748


namespace slowly_increasing_interval_l611_611410

def f (x : ℝ) : ℝ := (1/2) * x^2 - x + (3/2)

noncomputable def g (x : ℝ) : ℝ := f x / x 

theorem slowly_increasing_interval (I : Set ℝ) : 
  I = Set.Icc 1 (Real.sqrt 3) →
  (∀ x ∈ I, 0 ≤ hasDerivAt f x) ∧ (∀ x ∈ I, 0 ≤ hasDerivAt g x) :=
by
  intro hI
  sorry

end slowly_increasing_interval_l611_611410


namespace compute_expression_l611_611310

theorem compute_expression (x z : ℝ) (h1 : x ≠ 0) (h2 : z ≠ 0) (h3 : x = 1 / z^2) : 
  (x - 1 / x) * (z^2 + 1 / z^2) = x^2 - z^4 :=
by
  sorry

end compute_expression_l611_611310


namespace max_area_equilateral_triangle_l611_611951

theorem max_area_equilateral_triangle (p q r : ℕ) (hp : 313 = p) (hq : q = 3) (hr : r = 468) :
  p + q + r = 784 :=
by
  have hpq : p = 313 := hp
  have hqq : q = 3 := hq
  have hrr : r = 468 := hr
  rw [hpq, hqq, hrr]
  exact rfl

end max_area_equilateral_triangle_l611_611951


namespace average_marks_physics_chemistry_marks_of_other_subject_average_marks_3_subjects_l611_611625

noncomputable def marks :=
  let P := 80 in
  let C := 60 in
  let M := 100 in
  (P, C, M)

theorem average_marks_physics_chemistry (P C : ℕ) (h1 : P = 80) (h2 : (P + C) / 2 = 70) : 
  C = 60 :=
  by
  sorry

theorem marks_of_other_subject (P M: ℕ) (h1 : P = 80) (h2 : (P + M) / 2 = 90) :
  M = 100 :=
  by
  sorry

theorem average_marks_3_subjects (P C M: ℕ) (h1 : P = 80) (h2 : C = 60) (h3 : M = 100) (h4 : (P + C + M) / 3 = 80):
  (P, C, M) = marks :=
  by
  sorry

end average_marks_physics_chemistry_marks_of_other_subject_average_marks_3_subjects_l611_611625


namespace domino_perfect_play_winner_l611_611770

theorem domino_perfect_play_winner (complete_set : Set Domino) (initial_condition : domino_game_condition) :
  Player I wins :=
sorry

end domino_perfect_play_winner_l611_611770


namespace inscribed_cone_volume_l611_611521

theorem inscribed_cone_volume
  (H : ℝ) 
  (α : ℝ)
  (h_pos : 0 < H)
  (α_pos : 0 < α ∧ α < π / 2) :
  (1 / 12) * π * H ^ 3 * (Real.sin α) ^ 2 * (Real.sin (2 * α)) ^ 2 = 
  (1 / 3) * π * ((H * Real.sin α * Real.cos α / 2) ^ 2) * (H * (Real.sin α) ^ 2) :=
by sorry

end inscribed_cone_volume_l611_611521


namespace bushes_needed_l611_611609

-- Define the radius of the pond
def radius : ℝ := 8

-- Define the spacing between bushes
def spacing : ℝ := 0.5

-- Define the approximate value of π
def pi_approx : ℝ := 3.14

-- Calculate the number of bushes needed to surround the pond
def number_of_bushes (r : ℝ) (s : ℝ) (pi : ℝ) : ℝ :=
  let circumference := 2 * pi * r
  (circumference / s).round

-- State the theorem to be proven
theorem bushes_needed : number_of_bushes radius spacing pi_approx = 100 := by
  sorry

end bushes_needed_l611_611609


namespace fishing_tomorrow_l611_611818

theorem fishing_tomorrow (seven_every_day eight_every_other_day three_every_three_days twelve_yesterday ten_today : ℕ)
  (h1 : seven_every_day = 7)
  (h2 : eight_every_other_day = 8)
  (h3 : three_every_three_days = 3)
  (h4 : twelve_yesterday = 12)
  (h5 : ten_today = 10) :
  (seven_every_day + (eight_every_other_day - (twelve_yesterday - seven_every_day)) + three_every_three_days) = 15 :=
by
  sorry

end fishing_tomorrow_l611_611818


namespace isosceles_triangle_area_in_ellipse_l611_611262

theorem isosceles_triangle_area_in_ellipse :
  ∀ (x y : ℝ),
    2*x^2 + 3*y^2 = 6 ∧ -- Condition for the ellipse
    (1, real.sqrt 2) = (x, y) ∧ -- Vertex condition
    ∃ v1 v2 v3 : (ℝ × ℝ), -- There exists three vertices of the triangle (v1, v2, v3)
      v1 = (1, real.sqrt 2) ∧ -- v1 is (1, sqrt 2)
      v2 = (-1, real.sqrt 2) ∧ -- By symmetry v2 is (-1, sqrt 2)
      v3 = (0, 0) -- The third vertex C on the x-axis
    → 
      ∃ (area : ℝ), -- exists a real number area
        area = real.sqrt 2 -- such that the area of the triangle is sqrt(2)
:= sorry

end isosceles_triangle_area_in_ellipse_l611_611262


namespace cos_double_angle_l611_611395

theorem cos_double_angle (x y : ℝ) (h : cos x * cos y + sin x * sin y = 1 / 3) : cos (2 * x - 2 * y) = -7 / 9 := by
  sorry

end cos_double_angle_l611_611395


namespace max_value_of_ratio_max_value_of_ratio_is_9_l611_611070

noncomputable def max_value_ratio (x y z : ℝ) := (x + y + z)^3 / (x^3 + y^3 + z^3)

theorem max_value_of_ratio (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  max_value_ratio x y z ≤ 9 := by
  sorry

theorem max_value_of_ratio_is_9 (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  ∃ x y z > 0, max_value_ratio x y z = 9 := by
  use [1, 1, 1]
  simp [max_value_ratio]
  norm_num
  sorry

end max_value_of_ratio_max_value_of_ratio_is_9_l611_611070


namespace fishing_problem_l611_611785

theorem fishing_problem
  (everyday : ℕ)
  (every_other_day : ℕ)
  (every_three_days : ℕ)
  (yesterday_fishing : ℕ)
  (today_fishing : ℕ)
  (h_everyday : everyday = 7)
  (h_every_other_day : every_other_day = 8)
  (h_every_three_days : every_three_days = 3)
  (h_yesterday_fishing : yesterday_fishing = 12)
  (h_today_fishing : today_fishing = 10) :
  (every_three_days + everyday + (every_other_day - (yesterday_fishing - everyday))) = 15 := by
  sorry

end fishing_problem_l611_611785


namespace circumcircle_radius_FKO_l611_611902

theorem circumcircle_radius_FKO (A K L M F O : Point)
  (h1 : A ∈ segment L M)
  (h2 : ∠ K = 120 °)
  (h3 : incenter AKL = F)
  (h4 : incenter AKM = O)
  (h5 : dist A F = 3)
  (h6 : dist A O = 6) :
  circumradius F K O = sqrt 15 := 
sorry

end circumcircle_radius_FKO_l611_611902


namespace grazing_area_shape_rope_length_condition_l611_611246

theorem grazing_area_shape (l1 l2 : ℝ) (r : ℝ) (h_r : r = 20) :
  (abs (l1 - l2) < r) ∨ (abs (l1 - l2) ≥ r) :=
begin
  by_cases h : abs (l1 - l2) < r,
  { left, exact h },
  { right, exact not_lt.mp h }
end

theorem rope_length_condition (l1 l2 : ℝ) (r : ℝ) (h_r : r = 20) :
  min l1 l2 < r / 2 :=
begin
  have hr2 : r / 2 = 10, from (show 20 / 2 = 10, by norm_num),
  by_cases h : l1 < l2,
  { simp [min, h], linarith, },
  { simp [min, h], linarith, }
end

end grazing_area_shape_rope_length_condition_l611_611246


namespace a7_of_expansion_x10_l611_611531

theorem a7_of_expansion_x10 : 
  (∃ (a : ℕ) (a1 : ℕ) (a2 : ℕ) (a3 : ℕ) 
     (a4 : ℕ) (a5 : ℕ) (a6 : ℕ) 
     (a8 : ℕ) (a9 : ℕ) (a10 : ℕ),
     ((x : ℕ) → x^10 = a + a1*(x-1) + a2*(x-1)^2 + a3*(x-1)^3 + 
                      a4*(x-1)^4 + a5*(x-1)^5 + a6*(x-1)^6 + 
                      120*(x-1)^7 + a8*(x-1)^8 + a9*(x-1)^9 + a10*(x-1)^10)) :=
  sorry

end a7_of_expansion_x10_l611_611531


namespace minimum_oranges_condition_l611_611132

theorem minimum_oranges_condition (n : ℕ) :
  (∀ i j k : ℕ, i ≠ j → j ≠ k → i ≠ k → i < n → j < n → k < n → 
  let m := 1 in (3 * m / ((n-3) * m) < 0.05) → n ≥ 64) :=
begin
  intros h,
  sorry
end

end minimum_oranges_condition_l611_611132


namespace standard_equation_of_hyperbola_distance_from_focus_to_asymptote_l611_611692

noncomputable def hyperbola_center_origin : Type :=
{ C : ℝ × ℝ → ℝ // C (0, 0) = 0 }

noncomputable def right_focus : Type :=
{ F : ℝ × ℝ // F = (2, 0) }

noncomputable def real_axis_length : ℝ :=
2 * Real.sqrt 3

theorem standard_equation_of_hyperbola (C : hyperbola_center_origin) (F : right_focus) (a b : ℝ) (h_a : a = Real.sqrt 3) (h_c : 2 = 2) (h_eq : a ^ 2 + b ^ 2 = 4) :
  ∃ c : ℝ, c = 2 ∧ ∃ b : ℝ, b = 1 ∧ by
    calc
      (a ^ 2 + b ^ 2) = 4 : by sorry
    exact (a = Real.sqrt 3) ∧ (b = 1) :=
sorry

theorem distance_from_focus_to_asymptote (F : right_focus) (a b : ℝ) (h_a : a = Real.sqrt 3) (h_b : b = 1) :
  ∃ d : ℝ, d = 1 ∧ (λ (x y : ℝ), y = x / a ∨ y = -x / a) :=
sorry

end standard_equation_of_hyperbola_distance_from_focus_to_asymptote_l611_611692


namespace minjeong_marbles_l611_611954

theorem minjeong_marbles : ∃ (M : ℕ), 
  (let y : ℕ := M + 5 in y + M = 43) ∧ M = 19 :=
by
  use 19
  split
  { sorry }
  { refl }

end minjeong_marbles_l611_611954


namespace irrational_number_line_representation_l611_611217

theorem irrational_number_line_representation :
  ∀ (x : ℝ), ¬ (∃ r s : ℚ, x = r / s ∧ r ≠ 0 ∧ s ≠ 0) → ∃ p : ℝ, x = p := 
by
  sorry

end irrational_number_line_representation_l611_611217


namespace find_percent_defective_l611_611852

def percent_defective (D : ℝ) : Prop :=
  (0.04 * D = 0.32)

theorem find_percent_defective : ∃ D, percent_defective D ∧ D = 8 := by
  sorry

end find_percent_defective_l611_611852


namespace triangle_third_side_proof_l611_611628

noncomputable def triangle_third_side (AB : ℝ) (BC : ℝ) (AF : ℝ) (FB : ℝ) (r : ℝ) : ℝ := do
  let s := (AB + BC + (AB + r)) / 2
  let x := 12 -- This is derived from the condition given and solution steps.
  36

-- Given conditions
theorem triangle_third_side_proof : triangle_third_side 24 30 10 14 5 = 36 := by
  unfold triangle_third_side
  sorry

end triangle_third_side_proof_l611_611628


namespace triangle_side_count_l611_611760

theorem triangle_side_count :
  {b c : ℕ} → b ≤ 5 → 5 ≤ c → c - b < 5 → ∃ t : ℕ, t = 15 :=
by
  sorry

end triangle_side_count_l611_611760


namespace negation_universal_proposition_l611_611174

theorem negation_universal_proposition :
  ¬(∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ ∃ x : ℝ, x^3 - x^2 + 1 > 0 :=
by sorry

end negation_universal_proposition_l611_611174


namespace concert_students_l611_611191

theorem concert_students (num_buses : ℕ) (students_per_bus : ℕ) (extra_students : ℕ) 
  (num_buses_eq : num_buses = 12)
  (students_per_bus_eq : students_per_bus = 38)
  (extra_students_eq : extra_students = 5) :
  num_buses * students_per_bus + extra_students = 461 := 
by {
  rw [num_buses_eq, students_per_bus_eq, extra_students_eq],
  norm_num,
  sorry
}

end concert_students_l611_611191


namespace yardage_difference_is_minus_ten_l611_611537

noncomputable def total_yardage_A : ℕ := 150
noncomputable def passing_yardage_A : ℕ := 60
noncomputable def total_yardage_B : ℕ := 180
noncomputable def passing_yardage_B : ℕ := 80

def running_yardage_A := total_yardage_A - passing_yardage_A
def running_yardage_B := total_yardage_B - passing_yardage_B
def yardage_difference_A_B := running_yardage_A - running_yardage_B

theorem yardage_difference_is_minus_ten : yardage_difference_A_B = -10 := by
  sorry

end yardage_difference_is_minus_ten_l611_611537


namespace harmonic_arithmetic_max_and_count_l611_611925

noncomputable section

open Real Set

def P (a b c : ℤ) := (1 / (a : ℝ) + 1 / (b : ℝ) = 2 / (c : ℝ)) ∧ (a + c = 2 * b) ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a

def M := {x : ℤ | |x| ≤ 2014}

theorem harmonic_arithmetic_max_and_count :
  ∃ a b c : ℤ, (P a b c) ∧ ({a, b, c} ⊆ M)
  ∧ (∀ x ∈ {a, b, c}, x ≤ 2012)
  ∧ (∀ {a' b' c' : ℤ}, (P a' b' c') ∧ ({a', b', c'} ⊆ M) → {a', b', c'} ⊆ Iic (2012 : ℤ))
  ∧ (∃! (P : Finset ℤ), (P.card = 3) ∧ (∀ x ∈ P, x ∈ M) ∧ P.fst ≤ 503 ∧ P.snd ≤ 1006)
:= sorry

end harmonic_arithmetic_max_and_count_l611_611925


namespace proof_problem_l611_611915

variables {AO' AO_1 AB AC t s s_1 s_2 s_3 : ℝ}
variables {alpha : ℝ}

-- Conditions
def condition1 : Prop := AO' * Real.sin (alpha / 2) = t / s
def condition2 : Prop := AO_1 * Real.sin (alpha / 2) = t / s_1
def condition3 : Prop := AO' * AO_1 = t^2 / (s * s_1 * (Real.sin (alpha / 2))^2)
def condition4 : Prop := (Real.sin (alpha / 2))^2 = (s_2 * s_3) / (AB * AC)

-- Statement to prove
theorem proof_problem (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : condition4) : 
  AO' * AO_1 = AB * AC :=
by
  sorry

end proof_problem_l611_611915


namespace value_of_x_squared_plus_reciprocal_squared_l611_611750

theorem value_of_x_squared_plus_reciprocal_squared (x : ℝ) (h : 45 = x^4 + 1 / x^4) : 
  x^2 + 1 / x^2 = Real.sqrt 47 :=
by
  sorry

end value_of_x_squared_plus_reciprocal_squared_l611_611750


namespace minimum_oranges_l611_611117

theorem minimum_oranges (n : ℕ) (m : ℕ → ℝ) (h : ∀ i j k : ℕ, i < n → j < n → k < n → i ≠ j → i ≠ k → j ≠ k → (m i + m j + m k) < 0.05 * ∑ l in Finset.range n \ {i, j, k}, m l) : n ≥ 64 := 
sorry

end minimum_oranges_l611_611117


namespace tan_alpha_plus_beta_mul_tan_alpha_l611_611319

theorem tan_alpha_plus_beta_mul_tan_alpha (α β : ℝ) (h : 2 * Real.cos (2 * α + β) + 3 * Real.cos β = 0) :
  Real.tan (α + β) * Real.tan α = -5 := 
by
  sorry

end tan_alpha_plus_beta_mul_tan_alpha_l611_611319


namespace trigonometric_identity_proof_l611_611224

theorem trigonometric_identity_proof : 
  (sin (10 * Real.pi / 180) + sin (20 * Real.pi / 180)) / (cos (10 * Real.pi / 180) + cos (20 * Real.pi / 180)) 
  = tan (15 * Real.pi / 180) := 
by 
  sorry

end trigonometric_identity_proof_l611_611224


namespace measure_B_is_pi_div_3_maximum_area_l611_611416

variables (a b c : ℝ) (A B C : ℝ)
variables (triangle_ABC : b = 4)
variables (cos_ratio: cos B / cos C = 4 / (2 * a - c))

theorem measure_B_is_pi_div_3 :
  B = π / 3 := by
  sorry

theorem maximum_area :
  let S := (1 / 2) * a * c * sin B in
  S ≤ 4 * sqrt 3 := by
  sorry

end measure_B_is_pi_div_3_maximum_area_l611_611416


namespace jellybeans_problem_l611_611257

theorem jellybeans_problem (n : ℕ) (h : n ≥ 100) (h_mod : n % 13 = 11) : n = 102 :=
sorry

end jellybeans_problem_l611_611257


namespace angle_of_inclination_l611_611530

-- Define the parametric equations.
def parametric_x (t : ℝ) : ℝ := -1 + t * Real.cos (50 * Real.pi / 180)
def parametric_y (t : ℝ) : ℝ := -t * Real.sin (50 * Real.pi / 180)

-- The main theorem stating the angle of inclination is 130 degrees.
theorem angle_of_inclination : ∃ θ : ℝ, θ = 130 * Real.pi / 180 :=
by
  -- Skipping the proof steps as per instructions
  sorry

end angle_of_inclination_l611_611530


namespace AM_GM_HM_inequality_l611_611376

variables {a b : ℝ} (ha : 0 < a) (hb : 0 < b) (hab : a < b)

def AM : ℝ := (a + b) / 2
def GM : ℝ := real.sqrt (a * b)
def HM : ℝ := 2 * a * b / (a + b)

theorem AM_GM_HM_inequality : (AM ha hb hab - GM ha hb hab) / (AM ha hb hab - HM ha hb hab) < 1 :=
by
  sorry

end AM_GM_HM_inequality_l611_611376


namespace tiles_touching_walls_of_room_l611_611519

theorem tiles_touching_walls_of_room (length width : Nat) 
    (hl : length = 10) (hw : width = 5) : 
    2 * length + 2 * width - 4 = 26 := by
  sorry

end tiles_touching_walls_of_room_l611_611519


namespace problem_conditions_maximum_value_of_m_a1_value_cn_arithmetic_seq_l611_611473

noncomputable def a_n (n : ℕ) := 2^n * (n + 1)
noncomputable def S_n (n : ℕ) := ∑ i in finset.range n.succ, a_n i

def b_n (n : ℕ) := log (2,a_n n) - log (2,(↑n + 1))

def B_n (n : ℕ) := ∑ i in finset.range n.succ, 1 / b_n i

theorem problem_conditions (n : ℕ) (hn : n > 0) :
    (S_n n / 2 = a_n n - 2^n) ∧
  (∀ n ≥ 2, ((B_n (3 * n) - B_n n) > (18 / 20))) :=
begin
  sorry
end

theorem maximum_value_of_m (n : ℕ) :
    n ≥ 2 → (B_n (3 * n) - B_n n) > (18 / 20) :=
begin
  sorry
end

theorem a1_value : a_n 1 = 4 :=
begin
  sorry
end

theorem cn_arithmetic_seq (n : ℕ) : 
    ∀ n ≥ 2, let c_n := a_n n / 2^n in 
    c_n = (c_n - 1) + 1 :=
begin
  sorry
end

end problem_conditions_maximum_value_of_m_a1_value_cn_arithmetic_seq_l611_611473


namespace part1_l611_611302

def f (x : ℝ) : ℝ := x^2 + 2 * x

theorem part1 (x : ℝ) : f(2 * x + 1) = 4 * x^2 + 8 * x + 3 :=
by
  sorry

end part1_l611_611302


namespace Tian_walk_distance_l611_611964

theorem Tian_walk_distance (steps_distance : ℕ → ℕ → ℕ) :
  steps_distance 625 500 * 10000 / 625 = 8000 / 1000 :=
by {
  -- Provide a definition for steps_distance based on the conditions
  let steps_distance := λ n m, m / n,
  sorry
}

end Tian_walk_distance_l611_611964
