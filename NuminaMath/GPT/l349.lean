import Mathlib
import Mathlib.Algebra.Basic
import Mathlib.Algebra.Cubic.Root
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Floor
import Mathlib.Algebra.Order.Ring
import Mathlib.Algebra.Polynomial
import Mathlib.Algebra.Polynomial.Basic
import Mathlib.Algebra.Quadratic
import Mathlib.Algebra.Quadratic.Discriminant
import Mathlib.Algebra.Ring.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Complex
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Totient
import Mathlib.Data.Polynomial.Basic
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Probability.ProbabilityMassFunc
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Finite
import Mathlib.LinearAlgebra.Matrix
import Mathlib.Logic.Basic
import Mathlib.NumberTheory.ArithmeticFunctions
import Mathlib.NumberTheory.Prime
import Mathlib.Probability.Basic
import Mathlib.Probability.ProbabilityMassFunc
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import data.matrix.basic

namespace find_original_number_l349_349264

theorem find_original_number (x : ℝ)
  (h : (((x + 3) * 3 - 3) / 3) = 10) : x = 8 :=
sorry

end find_original_number_l349_349264


namespace delegates_sitting_probability_l349_349975

theorem delegates_sitting_probability :
  ∃ (m n : ℕ), (Nat.coprime m n) ∧
  (∏ i in Finset.range 12, (12 - i) / (4! * 4! * 4!)) * ((4! * 12 - 3 * 36 + 12) / 12!) = m / n ∧ m + n = 37757 :=
by
  sorry

end delegates_sitting_probability_l349_349975


namespace velocity_ratio_proof_l349_349226

noncomputable def velocity_ratio (V U : ℝ) : ℝ := V / U

-- The conditions:
-- 1. A smooth horizontal surface.
-- 2. The speed of the ball is perpendicular to the face of the block.
-- 3. The mass of the ball is much smaller than the mass of the block.
-- 4. The collision is elastic.
-- 5. After the collision, the ball’s speed is halved and it moves in the opposite direction.

def ball_block_collision 
    (V U U_final : ℝ) 
    (smooth_surface : Prop) 
    (perpendicular_impact : Prop) 
    (ball_much_smaller : Prop) 
    (elastic_collision : Prop) 
    (speed_halved : Prop) : Prop :=
  U_final = U ∧ V / U = 4

theorem velocity_ratio_proof : 
  ∀ (V U U_final : ℝ)
    (smooth_surface : Prop)
    (perpendicular_impact : Prop)
    (ball_much_smaller : Prop)
    (elastic_collision : Prop)
    (speed_halved : Prop),
    ball_block_collision V U U_final smooth_surface perpendicular_impact ball_much_smaller elastic_collision speed_halved := 
sorry

end velocity_ratio_proof_l349_349226


namespace count_primes_between_30_and_50_l349_349429

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between_30_and_50 : List ℕ :=
  [31, 37, 41, 43, 47]

theorem count_primes_between_30_and_50 : 
  (primes_between_30_and_50.filter is_prime).length = 5 :=
by
  sorry

end count_primes_between_30_and_50_l349_349429


namespace total_clothes_count_l349_349008

theorem total_clothes_count (shirts_per_pants : ℕ) (pants : ℕ) (shirts : ℕ) : shirts_per_pants = 6 → pants = 40 → shirts = shirts_per_pants * pants → shirts + pants = 280 := by
  intro h1 h2 h3
  rw [h1, h2] at h3
  rw [h3]
  sorry

end total_clothes_count_l349_349008


namespace regression_change_l349_349763

theorem regression_change (x : ℝ) (y : ℝ) (h : y = 3 - 5 * x) : 
  let y' := 3 - 5 * (x + 1) in y - y' = 5 :=
by
  sorry

end regression_change_l349_349763


namespace quadratic_has_single_real_root_l349_349448

theorem quadratic_has_single_real_root (n : ℝ) (h : (6 * n) ^ 2 - 4 * 1 * (2 * n) = 0) : n = 2 / 9 :=
by
  sorry

end quadratic_has_single_real_root_l349_349448


namespace symmetric_point_y_axis_l349_349144

theorem symmetric_point_y_axis (A : ℝ × ℝ) (B : ℝ × ℝ) : 
  A = (4, -8) → B = (-4, -8) → B = (-A.1, A.2) :=
by 
  intros hA hB
  rw [hA, hB]
  simp
  sorry

end symmetric_point_y_axis_l349_349144


namespace find_a_for_local_minimum_l349_349947

theorem find_a_for_local_minimum :
  ∃(a : ℝ), (∃ f : ℝ → ℝ, (∀ x, f x = a * x^3 + 2 * x^2 - a^2 * x) ∧ 
  (∃ df : ℝ → ℝ, (∀ x, df x = 3 * a * x^2 + 4 * x - a^2) ∧ df 1 = 0) ∧ df 1 > 0) ∧ a = 4 :=
sorry

end find_a_for_local_minimum_l349_349947


namespace largestOddDivisor_inequality_l349_349924

-- Define g(k) as the largest odd divisor of k
def largestOddDivisor (k : ℕ) : ℕ :=
  let rec oddDiv (n : ℕ) : ℕ := 
    if n % 2 = 1 then n
    else oddDiv (n / 2)
  oddDiv k

-- State the given problem in Lean
theorem largestOddDivisor_inequality (n : ℕ) (hn : n > 0) :
  0 < (∑ k in Finset.range (n + 1).filter (≠ 0), (largestOddDivisor k : ℚ) / k) - (2 * n : ℚ) / 3 ∧
  (∑ k in Finset.range (n + 1).filter (≠ 0), (largestOddDivisor k : ℚ) / k) - (2 * n : ℚ) / 3 < 2 / 3 :=
sorry

end largestOddDivisor_inequality_l349_349924


namespace min_slope_at_a_half_l349_349356

theorem min_slope_at_a_half (a : ℝ) (h : 0 < a) :
  (∀ b : ℝ, 0 < b → 4 * b + 1 / b ≥ 4) → (4 * a + 1 / a = 4) → a = 1 / 2 :=
by
  sorry

end min_slope_at_a_half_l349_349356


namespace angle_measure_l349_349554

theorem angle_measure (x y : ℝ) 
  (h1 : y = 3 * x + 10) 
  (h2 : x + y = 180) : x = 42.5 :=
by
  -- Proof goes here
  sorry

end angle_measure_l349_349554


namespace isosceles_triangle_properties_l349_349027

noncomputable def point := (ℝ × ℝ)
def line (a b c : ℝ) : set point := {p : point | a * p.1 + b * p.2 + c = 0}

variables (A B C : point) (l : line 1 (-1) 1)
-- A is the intersection of l with the y-axis
def A : point := (0, 1)
-- B is given
def B : point := (1, 3)
-- C must be solved
def C : point := (2, 2)

-- isosceles triangle conditions
def isosceles_triangle (A B C : point) : Prop := 
  dist A B = dist A C

-- questions
def equation_of_line_BC (B C : point) : line 1 1 (-4) := 
{x : point | x.1 + x.2 - 4 = 0}

def area_triangle (A B C : point) : ℝ := 
  1/2 * |det (matrix.row_vec 3 [
    [A.1, A.2, 1], 
    [B.1, B.2, 1], 
    [C.1, C.2, 1]
  ])|

theorem isosceles_triangle_properties :
  isosceles_triangle A B C →
  equation_of_line_BC B C = λ x, x.1 + x.2 - 4 ∧ 
  area_triangle A B C = 3/2 :=
begin
  sorry
end

end isosceles_triangle_properties_l349_349027


namespace digits_of_primorial_843301_l349_349291

-- Define the primorial function
def primorial (n : ℕ) : ℕ :=
  (finset.filter nat.prime (finset.range (n+1))).prod id

-- Define the number of digits
def num_digits (n : ℕ) : ℕ :=
  if n = 0 then 1 else nat.floor (real.log10 (n : ℝ)) + 1

-- State the theorem about the number of digits in the primorial of 843301
theorem digits_of_primorial_843301 : num_digits (primorial 843301) = 366263 := 
  sorry

end digits_of_primorial_843301_l349_349291


namespace square_field_area_l349_349209

-- Define the given conditions
def diagonal_length : ℝ := 20
def calculate_area (d : ℝ) : ℝ := (d^2) / 2

-- State the theorem
theorem square_field_area : calculate_area diagonal_length = 200 := by
  sorry

end square_field_area_l349_349209


namespace obtuse_triangle_acute_angles_l349_349815

theorem obtuse_triangle_acute_angles (A B C : ℝ) (h : A + B + C = 180)
  (hA : A > 90) : (B < 90) ∧ (C < 90) :=
sorry

end obtuse_triangle_acute_angles_l349_349815


namespace binomial_10_2_l349_349693

noncomputable def binom (n k : ℕ) : ℕ := n! / (k! * (n - k)!)

theorem binomial_10_2 : binom 10 2 = 45 := by
  sorry

end binomial_10_2_l349_349693


namespace binom_10_2_eq_45_l349_349697

-- Definitions used in the conditions
def binom (n k : ℕ) := n.choose k

-- The statement that needs to be proven
theorem binom_10_2_eq_45 : binom 10 2 = 45 :=
by
  sorry

end binom_10_2_eq_45_l349_349697


namespace num_unique_four_digit_numbers_l349_349819

theorem num_unique_four_digit_numbers : 
  let digits := [3, 0, 3, 3] in
  let valid_numbers := {n : ℕ | n / 1000 = 3 ∧ ∃ (h t u d : ℕ), n = h * 1000 + t * 100 + u * 10 + d ∧ h = 3 ∧ {t, u, d} = {0, 3, 3}} in
  ∃ (n : ℕ), n ∈ valid_numbers ∧ valid_numbers.finite ∧ valid_numbers.card = 1 :=
by 
  sorry

end num_unique_four_digit_numbers_l349_349819


namespace average_stamps_per_day_l349_349912

theorem average_stamps_per_day :
  (∀ n : ℕ, n < 5 → ((8 * (n + 1) * n) / 2) + 8 * (n + 1) = 8 * (n + 1) := sorry
  (8 + 16 + 24 + 32 + 40) / 5 = 24 :=
suffices : ((8 + (8+8) + (8+16) + (8+24) + (8+32)) / 5) = 24,
from this
sorry

end average_stamps_per_day_l349_349912


namespace number_of_primes_between_30_and_50_l349_349431

-- Define what it means to be a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the interval condition
def in_interval (n : ℕ) : Prop :=
  30 ≤ n ∧ n ≤ 50

-- Define the count of prime numbers in the interval
def prime_count_in_interval : ℕ :=
  (List.range' 30 21).countp (λ n, is_prime n)

-- We state that the above count is equal to 5
theorem number_of_primes_between_30_and_50 : prime_count_in_interval = 5 :=
  sorry

end number_of_primes_between_30_and_50_l349_349431


namespace smallest_integer_k_l349_349324

theorem smallest_integer_k (k : ℕ) : 
  (k > 1 ∧ 
   k % 13 = 1 ∧ 
   k % 7 = 1 ∧ 
   k % 5 = 1 ∧ 
   k % 3 = 1) ↔ k = 1366 := 
sorry

end smallest_integer_k_l349_349324


namespace product_of_m_l349_349551

theorem product_of_m (m y x : ℤ) (h1 : m - 3/2 * x ≥ -1 - x) (h2 : 4 - x ≥ 0)
  (h3 : (my : ℚ) / (y - 2) + 1 = -(3 * y) / (y - 2)) (int_solution : ∃ y_val: ℤ, (my : ℚ) / (y_val - 2) + 1 = -(3 * y_val) / (y_val - 2)) : 
  ∀ (m_vals : list ℤ), (m_vals = [0, 1, 4] ∧ 0 ∈ m_vals ∧ 1 ∈ m_vals ∧ 4 ∈ m_vals) → 
    list.prod m_vals = 0 :=
by sorry

end product_of_m_l349_349551


namespace probability_exactly_two_boys_l349_349548

theorem probability_exactly_two_boys 
   (total_members : ℕ)
   (boys : ℕ) 
   (girls : ℕ) 
   (committee_size : ℕ) 
   (two_boys : ℕ) 
   (three_girls : ℕ)
   (binom : ℕ → ℕ → ℕ) 
   (probability : ℚ) : 
   total_members = 30 →
   boys = 12 →
   girls = 18 →
   committee_size = 5 →
   two_boys = binom boys 2 →
   three_girls = binom girls 3 →
   probability = two_boys * three_girls / binom total_members committee_size →
   probability = 26928 / 71253 :=
by
  intros h_total h_boys h_girls h_committee h_two h_three h_prob
  rw [h_total, h_boys, h_girls, h_committee, h_two, h_three] at h_prob
  sorry

end probability_exactly_two_boys_l349_349548


namespace last_four_digits_of_7_pow_5000_l349_349595

theorem last_four_digits_of_7_pow_5000 (h : 7 ^ 250 ≡ 1 [MOD 1250]) : 7 ^ 5000 ≡ 1 [MOD 1250] :=
by
  -- Proof (will be omitted)
  sorry

end last_four_digits_of_7_pow_5000_l349_349595


namespace complex_number_conditions_l349_349333

open Complex Real

noncomputable def is_real (a : ℝ) : Prop :=
a ^ 2 - 2 * a - 15 = 0

noncomputable def is_imaginary (a : ℝ) : Prop :=
a ^ 2 - 2 * a - 15 ≠ 0

noncomputable def is_purely_imaginary (a : ℝ) : Prop :=
a ^ 2 - 9 = 0 ∧ a ^ 2 - 2 * a - 15 ≠ 0

theorem complex_number_conditions (a : ℝ) :
  (is_real a ↔ (a = 5 ∨ a = -3))
  ∧ (is_imaginary a ↔ (a ≠ 5 ∧ a ≠ -3))
  ∧ (¬(∃ a : ℝ, is_purely_imaginary a)) :=
by
  sorry

end complex_number_conditions_l349_349333


namespace binomial_10_2_l349_349695

noncomputable def binom (n k : ℕ) : ℕ := n! / (k! * (n - k)!)

theorem binomial_10_2 : binom 10 2 = 45 := by
  sorry

end binomial_10_2_l349_349695


namespace probability_diff_suits_l349_349023

theorem probability_diff_suits (n : ℕ) (h₁ : n = 65) (suits : ℕ) (h₂ : suits = 5) (cards_per_suit : ℕ) (h₃ : cards_per_suit = n / suits) : 
  (52 : ℚ) / (64 : ℚ) = (13 : ℚ) / (16 : ℚ) := 
by 
  sorry

end probability_diff_suits_l349_349023


namespace power_fun_f3_l349_349833

noncomputable def f (x : ℝ) : ℝ := x ^ (1 : ℝ) -- based on the solution, α = 1

theorem power_fun_f3 (h : f 4 = 2 * f 2) : f 3 = 3 := by
  unfold f at h -- unfold the definition of f at the hypothesis
  rw [←pow_add] at h -- using properties of exponents
  have ha : (4:ℝ) = 2^2 := by norm_num -- rewriting 4 as 2^2
  rw [ha] at h
  norm_num at h -- simplify the numeral expressions
  norm_num -- simplify to reveal α = 1
  rw [h] -- replace f with the correct alpha
  exact eq.refl 3 -- which gives us the required result

end power_fun_f3_l349_349833


namespace fraction_clerical_staff_reduced_l349_349232

theorem fraction_clerical_staff_reduced
  (total_employees : ℕ)
  (clerical_fraction : ℚ)
  (remaining_clerical_percentage : ℚ) :
  total_employees = 3600 →
  clerical_fraction = 1/6 →
  remaining_clerical_percentage = 0.13043478260869565 →
  let clerical_initial := clerical_fraction * total_employees,
      clerical_reduced_fraction := (clerical_initial - clerical_initial * x) / (total_employees - clerical_initial * x)
  in clerical_reduced_fraction = remaining_clerical_percentage → x = 1/4 :=
begin
  intros h1 h2 h3 h4,
  have clerical_initial : ℚ := clerical_fraction * total_employees,
  have total_employees_after_reduction : ℚ := total_employees - clerical_initial * x,
  have clerical_remaining : ℚ := clerical_initial - clerical_initial * x,
  replace h4 := (clerical_remaining / total_employees_after_reduction) = remaining_clerical_percentage,
  sorry
end

end fraction_clerical_staff_reduced_l349_349232


namespace count_primes_between_30_and_50_l349_349412

-- Define the range of numbers from 30 to 50
def range_30_to_50 := Set.of_list (List.range' 30 (51 - 30))

-- Define a predicate to check if a number is prime
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

-- Extract all prime numbers in the specified range
def primes_between_30_and_50 : List ℕ :=
  List.filter is_prime (Set.toList range_30_to_50)

theorem count_primes_between_30_and_50 : primes_between_30_and_50.length = 5 :=
by
  -- The proof goes here
  sorry

end count_primes_between_30_and_50_l349_349412


namespace polar_to_cartesian_2_pi_over_6_l349_349285

theorem polar_to_cartesian_2_pi_over_6 :
  let r : ℝ := 2
  let θ : ℝ := (Real.pi / 6)
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  (x, y) = (Real.sqrt 3, 1) := by
    -- Initialize the constants and their values
    let r := 2
    let θ := Real.pi / 6
    let x := r * Real.cos θ
    let y := r * Real.sin θ
    -- Placeholder for the actual proof
    sorry

end polar_to_cartesian_2_pi_over_6_l349_349285


namespace fraction_females_prefer_career_l349_349163

variable (x : ℝ) (f : ℝ)

-- The ratio of males to females in the class is 2:3
-- The number of males is 2x and the number of females is 3x
def males := 2 * x
def females := 3 * x

-- 144 degrees of the circle, which is 2/5 of the circle, is used to represent a career
-- that is preferred by one-fourth of the males
def fraction_of_circle := 2 / 5
def males_prefer_career := (1 / 4) * males

-- Let f be the fraction of females who prefer this career
def females_prefer_career := f * females

-- The total number of students preferring this career is the sum of males and females preferring it
def total_prefer_career := males_prefer_career + females_prefer_career

-- Prove that f = 1/2 given total student preference fraction
theorem fraction_females_prefer_career : 
  total_prefer_career = fraction_of_circle * (males + females) → f = 1 / 2 :=
by
  sorry

end fraction_females_prefer_career_l349_349163


namespace final_ratio_of_milk_to_water_l349_349846

-- Initial conditions definitions
def initial_milk_ratio : ℚ := 5 / 8
def initial_water_ratio : ℚ := 3 / 8
def additional_milk : ℚ := 8
def total_capacity : ℚ := 72

-- Final ratio statement
theorem final_ratio_of_milk_to_water :
  (initial_milk_ratio * (total_capacity - additional_milk) + additional_milk) / (initial_water_ratio * (total_capacity - additional_milk)) = 2 := by
  sorry

end final_ratio_of_milk_to_water_l349_349846


namespace total_students_class_l349_349457

theorem total_students_class (S R : ℕ) 
  (h1 : 2 + 12 + 10 + R = S)
  (h2 : (0 * 2) + (1 * 12) + (2 * 10) + (3 * R) = 2 * S) :
  S = 40 := by
  sorry

end total_students_class_l349_349457


namespace exponentiation_problem_l349_349670

theorem exponentiation_problem : (8^8 / 8^5) * 2^10 * 2^3 = 2^22 := by
  sorry

end exponentiation_problem_l349_349670


namespace number_of_men_in_second_group_l349_349615

variable (n m : ℕ)

theorem number_of_men_in_second_group 
  (h1 : 42 * 18 = n)
  (h2 : n = m * 28) : 
  m = 27 := by
  sorry

end number_of_men_in_second_group_l349_349615


namespace range_of_values_of_k_l349_349829

theorem range_of_values_of_k (k : ℝ) : 
  (∃ (A B C : Type) [geometry_tri ABC], 
    ∠ABC = 60 ∧ length AC = 12 ∧ length BC = k) → 
  (0 < k ∧ k ≤ 12) ∨ k = 8 * sqrt 3 := 
by sorry

end range_of_values_of_k_l349_349829


namespace triangle_AC_length_l349_349451

theorem triangle_AC_length (A B C : Type) [metric_space A] [metric_space B] [metric_space C]
  (angle_B : real) (AB : real) (S : real) (AC : real)
  (h_angle_B : angle_B = 30)
  (h_AB : AB = 2 * sqrt 3)
  (h_S : S = sqrt 3) :
  AC = 4 * sqrt 3 :=
by
  sorry

end triangle_AC_length_l349_349451


namespace max_value_f_range_a_sum_inequality_l349_349798

-- Definitions for the conditions
def f (x : ℝ) : ℝ := log x - x + 1
def g (x : ℝ) (a : ℝ) : ℝ := x^3 - a * x

-- Problem 1: Maximum value of f(x)
theorem max_value_f : ∀ x : ℝ, 0 < x → f(x) ≤ 0 := by
  -- proof steps would go here
  sorry

-- Problem 2: Range of a given the condition
theorem range_a (a : ℝ) : (∀ x1 : ℝ, 0 < x1 → ∃ x2 : ℝ, 1 ≤ x2 ∧ x2 ≤ 2 ∧ f(x1) ≤ g(x2, a)) → a ≤ 4 := by
  -- proof steps would go here
  sorry

-- Problem 3: Prove the inequality
theorem sum_inequality (n : ℕ) (h : 0 < n) : (∑ k in Finset.range n, (k.succ / n) ^ n) < (Real.exp 1) / (Real.exp 1 - 1) := by
  -- proof steps would go here
  sorry

end max_value_f_range_a_sum_inequality_l349_349798


namespace whole_numbers_between_cubics_l349_349826

theorem whole_numbers_between_cubics : 
  let a := real.cbrt 10
  let b := real.cbrt 200
  2 < a ∧ a < 3 → 5 < b ∧ b < 6 → (∃ n : ℕ, n = 3) := 
by
  intros a b h₁ h₂
  let whole_numbers := {n : ℕ | a < n ∧ n < b}
  have : whole_numbers = {3, 4, 5} := sorry
  have count := finset.card ⟨whole_numbers.1, sorry⟩
  exact sorry

end whole_numbers_between_cubics_l349_349826


namespace equation_one_solution_equation_two_solution_l349_349743

theorem equation_one_solution (x : ℕ) : 8 * (x + 1)^3 = 64 ↔ x = 1 := by 
  sorry

theorem equation_two_solution (x : ℤ) : (x + 1)^2 = 100 ↔ x = 9 ∨ x = -11 := by 
  sorry

end equation_one_solution_equation_two_solution_l349_349743


namespace parallel_lines_condition_l349_349544

theorem parallel_lines_condition (a : ℝ) :
  let line1 := λ x y : ℝ, a*x + 2*y + 1 = 0,
      line2 := λ x y : ℝ, 3*x + (a-1)*y + 1 = 0 in
  ((∀ x y, line1 x y = 0 ↔ line2 x y ≠ 0) → a = -2) :=
begin
  sorry
end

end parallel_lines_condition_l349_349544


namespace distance_from_A_to_plane_alpha_l349_349751

-- Definitions of the points and vector
def pointA := (1, 0, 1 : ℝ × ℝ × ℝ)
def pointB := (-1, 2, 2 : ℝ × ℝ × ℝ)
def normalVec := (1, 0, 1 : ℝ × ℝ × ℝ)

-- Definition of the distance from a point to a plane
noncomputable def distance_from_point_to_plane (pointA pointB normalVec : ℝ × ℝ × ℝ) : ℝ :=
  let AB := ((pointB.1 − pointA.1), (pointB.2 − pointA.2), (pointB.3 − pointA.3)) in
  let dotProduct := AB.1 * normalVec.1 + AB.2 * normalVec.2 + AB.3 * normalVec.3 in
  let normNormalVec := real.sqrt (normalVec.1 ^ 2 + normalVec.2 ^ 2 + normalVec.3 ^ 2) in
  (real.abs dotProduct) / normNormalVec

-- The theorem statement
theorem distance_from_A_to_plane_alpha :
  distance_from_point_to_plane pointA pointB normalVec = real.sqrt 2 / 2 := by
  sorry

end distance_from_A_to_plane_alpha_l349_349751


namespace frosting_cupcakes_l349_349513

theorem frosting_cupcakes (time_mark : ℕ) (time_julia : ℕ) (total_time : ℕ) (cupcakes_frosted : ℕ) 
  (h1 : time_mark = 15) (h2 : time_julia = 40) (h3 : total_time = 600) :
  cupcakes_frosted = total_time * (1 / time_mark + 1 / time_julia) := by
  have h5 : (1 / 15 : ℚ) = (1 / 15) := sorry
  have h6 : (1 / 40 : ℚ) = (1 / 40) := sorry
  have h7 : (1 / 15 + 1 / 40 : ℚ) = 11 / 120 := sorry
  have h8 : (600 * (11 / 120) : ℚ) = 55 := sorry
  exact h8

end frosting_cupcakes_l349_349513


namespace soccer_team_games_played_l349_349641

theorem soccer_team_games_played (t : ℝ) (h1 : 0.40 * t = 63.2) : t = 158 :=
sorry

end soccer_team_games_played_l349_349641


namespace a_eq_b_pow_n_l349_349772

theorem a_eq_b_pow_n (a b n : ℕ) (h : ∀ k : ℕ, k ≠ b → (a - k^n) % (b - k) = 0) : a = b^n :=
sorry

end a_eq_b_pow_n_l349_349772


namespace angle_NML_is_90_degrees_l349_349487

theorem angle_NML_is_90_degrees
  (A B C L M N : Point)
  (h1 : AngleBisector (Angle A B C) (Line B L) (Line L C))
  (h2 : OnSeg M A C)
  (h3 : OnSeg N A B)
  (h4 : Concurrent (Line A L) (Line B M) (Line C N))
  (h5 : ∠AMN = ∠ALB) :
  ∠NML = 90 := 
sorry

end angle_NML_is_90_degrees_l349_349487


namespace right_triangle_345_l349_349996

theorem right_triangle_345 : 
  (∃ a b c : ℕ, a = 3 ∧ b = 4 ∧ c = 5 ∧ a^2 + b^2 = c^2) ∧ 
  ¬(∃ a b c : ℕ, a = 2 ∧ b = 3 ∧ c = 4 ∧ a^2 + b^2 = c^2) ∧ 
  ¬(∃ a b c : ℕ, a = 4 ∧ b = 5 ∧ c = 6 ∧ a^2 + b^2 = c^2) ∧ 
  ¬(∃ a b c : ℕ, a = 6 ∧ b = 8 ∧ c = 9 ∧ a^2 + b^2 = c^2) :=
by {
  sorry
}

end right_triangle_345_l349_349996


namespace ag_eq_ig_l349_349661

noncomputable def Triangle := sorry

structure Point (α : Type*) :=
(x : α)
(y : α)

structure Circle (α : Type*) :=
(center : Point α)
(radius : α)

variables {α : Type*} [LinearOrderedField α]

def is_perpendicular (l1 l2 : Line α) : Prop := sorry
def lies_on (P : Point α) (l : Line α) : Prop := sorry
def intersects (l1 l2 : Line α) (P : Point α) : Prop := sorry
def circumcenter (t : Triangle α) : Point α := sorry
def incenter (t : Triangle α) : Point α := sorry
def excenter (t : Triangle α) : Point α := sorry

axiom given_conditions (t : Triangle α) 
  (O I J : Point α) 
  (OJ : Line α) 
  (H G : Point α)
  (IH : Line α):
  circumcenter t = O ∧
  incenter t = I ∧
  excenter t = J ∧
  is_perpendicular IH OJ ∧
  intersects IH (line_through I O) H ∧
  intersects IH (line_through I (foot t B C)) G

theorem ag_eq_ig 
  (t : Triangle α) 
  (A B C O I J H G : Point α) 
  (OJ : Line α) 
  (IH : Line α) :
  given_conditions t O I J OJ H G IH →
  dist A G = dist I G :=
by sorry

end ag_eq_ig_l349_349661


namespace emily_earnings_l349_349295

theorem emily_earnings :
  let monday_hours := 1
  let wednesday_hours := (2 + 40/60 : ℝ)
  let thursday_hours := 0.5
  let saturday_hours := 0.5
  let total_hours := monday_hours + wednesday_hours + thursday_hours + saturday_hours
  let hourly_rate := 4
  total_hours * hourly_rate = 18.68 :=
by
  let monday_hours := 1
  let wednesday_hours := (2 + 40/60 : ℝ)
  let thursday_hours := 0.5
  let saturday_hours := 0.5
  let total_hours := monday_hours + wednesday_hours + thursday_hours + saturday_hours
  let hourly_rate := 4
  suffices total_hours * hourly_rate = 18.68 from this
  sorry

end emily_earnings_l349_349295


namespace express_delivery_avg_growth_rate_l349_349914

noncomputable def monthly_avg_growth_rate (D_o D_d : ℕ) : ℝ :=
  (D_d.toReal / D_o.toReal)^(1/2) - 1

theorem express_delivery_avg_growth_rate :
  monthly_avg_growth_rate 100000 121000 = 0.1 :=
by
  rw [monthly_avg_growth_rate]
  sorry

end express_delivery_avg_growth_rate_l349_349914


namespace largest_4_digit_divisible_by_12_l349_349185

theorem largest_4_digit_divisible_by_12 : ∃ n : ℕ, (1000 ≤ n ∧ n ≤ 9999) ∧ 12 ∣ n ∧ ∀ m : ℕ, (1000 ≤ m ∧ m ≤ 9999) ∧ 12 ∣ m → m ≤ n :=
sorry

end largest_4_digit_divisible_by_12_l349_349185


namespace either_x_or_y_is_even_l349_349928

theorem either_x_or_y_is_even (x y z : ℤ) (h : x^2 + y^2 = z^2) : (2 ∣ x) ∨ (2 ∣ y) :=
by
  sorry

end either_x_or_y_is_even_l349_349928


namespace minimum_value_f_maximum_value_a_l349_349511

noncomputable def f (x k : ℝ) : ℝ := (x - k) * real.exp x

theorem minimum_value_f (k : ℝ) :
  (k ≤ 1 → ∃ x ∈ set.Icc 0 1, f x k = -k) ∧
  (1 < k ∧ k < 2 → ∃ x ∈ set.Icc 0 1, f x k = - real.exp (k - 1) - 1) ∧
  (k ≥ 2 → ∃ x ∈ set.Icc 0 1, f x k = (1 - k) * real.exp 1) :=
by sorry

theorem maximum_value_a : 
  (∀ x > 0, f x 1 - a * real.exp x + a > 0) → (a = 3) :=
by assume a ha,
   sorry

end minimum_value_f_maximum_value_a_l349_349511


namespace isosceles_triangle_problem_l349_349852

noncomputable def isosceles_triangle_area_projection (A B C M K O : Point) (AB BC : ℝ) (areaBOM areaCOM : ℝ) : ℝ × ℝ :=
  let triangleABC_area := 176
  let segment_projection := 2 * sqrt(3 / 11)
  (triangleABC_area, segment_projection)

theorem isosceles_triangle_problem (A B C M K O : Point) (H1 : isosceles_triangle A B C AB BC)
  (H2 : angle_bisector_intersects_at A B C M K O) (H3 : triangle_area B O M = 25)
  (H4 : triangle_area C O M = 30) : 
  isosceles_triangle_area_projection A B C M K O AB BC = (176, 2 * sqrt(3 / 11)) :=
  by
    sorry

end isosceles_triangle_problem_l349_349852


namespace negation_proof_l349_349774

theorem negation_proof : 
  (¬(∀ x : ℝ, x < 2^x) ↔ ∃ x : ℝ, x ≥ 2^x) :=
by
  sorry

end negation_proof_l349_349774


namespace total_students_class_l349_349458

theorem total_students_class (S R : ℕ) 
  (h1 : 2 + 12 + 10 + R = S)
  (h2 : (0 * 2) + (1 * 12) + (2 * 10) + (3 * R) = 2 * S) :
  S = 40 := by
  sorry

end total_students_class_l349_349458


namespace abs_neg_six_l349_349217

theorem abs_neg_six : abs (-6) = 6 := by
  sorry

end abs_neg_six_l349_349217


namespace initial_candies_proof_l349_349240

noncomputable def initial_candies (n : ℕ) := 
  ∃ c1 c2 c3 c4 c5 : ℕ, 
    c5 = 1 ∧
    c5 = n * 1 / 6 ∧
    c4 = n * 5 / 6 ∧
    c3 = n * 4 / 5 ∧
    c2 = n * 3 / 4 ∧
    c1 = n * 2 / 3 ∧
    n = 2 * c1

theorem initial_candies_proof (n : ℕ) : initial_candies n → n = 720 :=
  by
    sorry

end initial_candies_proof_l349_349240


namespace total_profit_correct_l349_349600

-- Definitions for the conditions
def investment_a := 24000
def investment_b := 32000
def investment_c := 36000

def profit_share_c := 36000

-- Definition for the total profit
def total_profit := 92000

-- Lean theorem statement
theorem total_profit_correct :
  let total_investment := investment_a + investment_b + investment_c in
  (investment_c / total_investment) * total_profit = profit_share_c :=
by
  -- Lean will check this as a part of the proof process, but we have skipped the full proof.
  sorry

end total_profit_correct_l349_349600


namespace die_roll_probability_l349_349643

theorem die_roll_probability :
  let rolls := 8,
      die_faces := 6,
      prime_odd_rolls := {3, 5},
      favorable_cases := prime_odd_rolls.size,
      probability_single_roll := favorable_cases / die_faces,
      probability_eight_rolls := probability_single_roll ^ rolls
  in probability_eight_rolls = 1 / 6561 :=
by
  let rolls := 8
  let die_faces := 6
  let prime_odd_rolls := {3, 5}
  let favorable_cases := prime_odd_rolls.size
  let probability_single_roll := favorable_cases / die_faces
  let probability_eight_rolls := probability_single_roll ^ rolls
  show probability_eight_rolls = 1 / 6561
  sorry

end die_roll_probability_l349_349643


namespace binom_10_2_eq_45_l349_349709

theorem binom_10_2_eq_45 :
  binom 10 2 = 45 := by
  sorry

end binom_10_2_eq_45_l349_349709


namespace cos_sum_equals_fraction_sqrt_13_minus_1_div_4_l349_349930

noncomputable def cos_sum : ℝ :=
  (Real.cos (2 * Real.pi / 17) +
   Real.cos (6 * Real.pi / 17) +
   Real.cos (8 * Real.pi / 17))

theorem cos_sum_equals_fraction_sqrt_13_minus_1_div_4 :
  cos_sum = (Real.sqrt 13 - 1) / 4 := 
sorry

end cos_sum_equals_fraction_sqrt_13_minus_1_div_4_l349_349930


namespace positive_difference_l349_349584

def a : ℕ := (8^2 - 8^2) / 8
def b : ℕ := (8^2 * 8^2) / 8

theorem positive_difference : |b - a| = 512 :=
by
  sorry

end positive_difference_l349_349584


namespace unique_tiling_of_bounded_below_set_l349_349535

-- Declare the main theorem with given conditions
theorem unique_tiling_of_bounded_below_set (A S : Set ℕ) (hAB : ∃ b ∈ A, ∀ a ∈ A, b ≤ a)
  (hTiling : ∀ a ∈ A, ∃ s ∈ S, a ∈ s) : 
  ∀ a1 a2 ∈ A, a1 < a2 → (∃ s ∈ S, a1 ∈ s ∧ a2 ∈ s) → False :=
sorry

end unique_tiling_of_bounded_below_set_l349_349535


namespace soup_cost_l349_349329

def muffin_cost := 2
def coffee_cost := 4
def salad_cost := 5.25
def lemonade_cost := 0.75
def lunch_more_than_breakfast := 3

def breakfast_cost := muffin_cost + coffee_cost
def lunch_cost := breakfast_cost + lunch_more_than_breakfast
def salad_and_lemonade_cost := salad_cost + lemonade_cost

theorem soup_cost :
  lunch_cost - salad_and_lemonade_cost = 3 :=
  sorry

end soup_cost_l349_349329


namespace decrement_value_each_observation_l349_349154

theorem decrement_value_each_observation 
  (n : ℕ) 
  (original_mean updated_mean : ℝ) 
  (n_pos : n = 50) 
  (original_mean_value : original_mean = 200)
  (updated_mean_value : updated_mean = 153) :
  (original_mean * n - updated_mean * n) / n = 47 :=
by
  sorry

end decrement_value_each_observation_l349_349154


namespace range_of_b_l349_349017

theorem range_of_b (b : ℝ) : (¬ ∃ a < 0, a + 1/a > b) → b ≥ -2 := 
by {
  sorry
}

end range_of_b_l349_349017


namespace random_event_is_crane_among_chickens_l349_349265

-- Definitions of the idioms as events
def coveringTheSkyWithOneHand : Prop := false
def fumingFromAllSevenOrifices : Prop := false
def stridingLikeAMeteor : Prop := false
def standingOutLikeACraneAmongChickens : Prop := ¬false

-- The theorem stating that Standing out like a crane among chickens is a random event
theorem random_event_is_crane_among_chickens :
  ¬coveringTheSkyWithOneHand ∧ ¬fumingFromAllSevenOrifices ∧ ¬stridingLikeAMeteor → standingOutLikeACraneAmongChickens :=
by 
  sorry

end random_event_is_crane_among_chickens_l349_349265


namespace cube_probability_problem_l349_349233

-- Definitions based on the problem's conditions
def total_faces : ℕ := 6
def faces_labeled_1 : ℕ := 1
def faces_labeled_2 : ℕ := 2
def faces_labeled_3 : ℕ := 3

-- Proposition (1): Probability of rolling a 2
def prob_rolling_2 : ℚ := faces_labeled_2 / total_faces

-- Proposition (2): Number with the highest probability of facing up
def highest_prob_number : ℕ := if (faces_labeled_1 > faces_labeled_2) ∧ (faces_labeled_1 > faces_labeled_3) then 1
  else if (faces_labeled_2 > faces_labeled_1) ∧ (faces_labeled_2 > faces_labeled_3) then 2
  else 3

-- Proposition (3): Chances of Winning
def player_A_wins_faces : ℕ := faces_labeled_1 + faces_labeled_2
def player_B_wins_faces : ℕ := faces_labeled_3
def chances_equal : Bool := player_A_wins_faces = player_B_wins_faces

-- The statement to be proved in Lean 4
theorem cube_probability_problem :
  prob_rolling_2 = 1/3 ∧
  highest_prob_number = 3 ∧
  chances_equal = true :=
sorry

end cube_probability_problem_l349_349233


namespace tank_overflow_time_l349_349207

noncomputable def pipeARate : ℚ := 1 / 32
noncomputable def pipeBRate : ℚ := 3 * pipeARate
noncomputable def combinedRate (rateA rateB : ℚ) : ℚ := rateA + rateB

theorem tank_overflow_time : 
  combinedRate pipeARate pipeBRate = 1 / 8 ∧ (1 / combinedRate pipeARate pipeBRate = 8) :=
by
  sorry

end tank_overflow_time_l349_349207


namespace primes_between_30_and_50_l349_349392

theorem primes_between_30_and_50 : (Finset.card (Finset.filter Nat.Prime (Finset.Ico 30 51))) = 5 :=
by
  sorry

end primes_between_30_and_50_l349_349392


namespace vertical_line_meets_parabola_once_l349_349967

theorem vertical_line_meets_parabola_once (m : ℝ) : 
  (∃ y : ℝ, -4 * y^2 - 6 * y + 10 = m) ↔ m = 49 / 4 :=
begin
  sorry
end

end vertical_line_meets_parabola_once_l349_349967


namespace number_of_divisors_that_are_multiples_of_2_l349_349004

-- Define the prime factorization of 540
def prime_factorization_540 : ℕ × ℕ × ℕ := (2, 3, 5)

-- Define the constraints for a divisor to be a multiple of 2
def valid_divisor_form (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 2 ∧ 0 ≤ b ∧ b ≤ 3 ∧ 0 ≤ c ∧ c ≤ 1

noncomputable def count_divisors (prime_info : ℕ × ℕ × ℕ) : ℕ :=
  let (p1, p2, p3) := prime_info
  2 * 4 * 2 -- Correspond to choices for \( a \), \( b \), and \( c \)

theorem number_of_divisors_that_are_multiples_of_2 (p1 p2 p3 : ℕ) (h : prime_factorization_540 = (p1, p2, p3)) :
  ∃ (count : ℕ), count = 16 :=
by
  use count_divisors (2, 3, 5)
  sorry

end number_of_divisors_that_are_multiples_of_2_l349_349004


namespace P_eight_value_l349_349097

def quadratic_polynomial (P : ℤ → ℤ) : Prop :=
  ∃ a b c : ℤ, ∀ x : ℤ, P(x) = a * x^2 + b * x + c

theorem P_eight_value (P : ℤ → ℤ)
  (H : ∀ x : ℤ, P(P(x)) = x^4 - 2 * x^3 + 4 * x^2 - 3 * x + 4)
  (H_quad : quadratic_polynomial P) :
  P 8 = 58 :=
sorry

end P_eight_value_l349_349097


namespace jia_steps_when_meet_l349_349876

-- Define the conditions
variable (v : ℝ) -- Jia's walking speed in feet per unit time
variable (d : ℝ := 10560) -- Distance between houses in feet
variable (s : ℝ := 2.5) -- Jia's step length in feet

-- Define Yi's speed as 5 times Jia's speed
noncomputable def yi_speed : ℝ := 5 * v

-- Define the theorem to prove the number of steps Jia takes when they meet
theorem jia_steps_when_meet : 
  let t := d / (v + yi_speed) in 
  let steps := (v * t) / s in 
  steps = 704 :=
by
  sorry

end jia_steps_when_meet_l349_349876


namespace prime_count_30_to_50_l349_349404

def is_prime (n: ℕ) : Prop :=
  n > 1 ∧ ∀ m: ℕ, m ∣ n → m = 1 ∨ m = n

def primes_in_range (a b: ℕ) : list ℕ :=
  list.filter is_prime (list.range' a (b - a + 1))

theorem prime_count_30_to_50 : (primes_in_range 30 50).length = 5 :=
by sorry

end prime_count_30_to_50_l349_349404


namespace transformed_sum_l349_349639

theorem transformed_sum (n : ℕ) (x : Fin n → ℝ) (s : ℝ)
  (h_sum : (∑ i, x i) = s) :
  (∑ i, (3 * (x i + 10) - 10)) = 3 * s + 20 * n := by
  sorry

end transformed_sum_l349_349639


namespace expression_divisible_by_x_minus_1_squared_l349_349524

theorem expression_divisible_by_x_minus_1_squared :
  ∀ (n : ℕ) (x : ℝ), x ≠ 1 →
  (n * x^(n + 1) * (1 - 1 / x) - x^n * (1 - 1 / x^n)) / (x - 1)^2 = 
  (n * x^(n + 1) - n * x^n - x^n + 1) / (x - 1)^2 :=
by
  intro n x hx_ne_1
  sorry

end expression_divisible_by_x_minus_1_squared_l349_349524


namespace abs_sum_ge_sqrt_three_over_two_l349_349734

open Real

theorem abs_sum_ge_sqrt_three_over_two
  (a b : ℝ) : (|a| + |b| ≥ 2 / sqrt 3) ∧ (∀ x, |a * sin x + b * sin (2 * x)| ≤ 1) ↔
  (a, b) = (4 / (3 * sqrt 3), 2 / (3 * sqrt 3)) ∨ 
  (a, b) = (-4 / (3 * sqrt 3), -2 / (3 * sqrt 3)) ∨
  (a, b) = (4 / (3 * sqrt 3), -2 / (3 * sqrt 3)) ∨
  (a, b) = (-4 / (3 * sqrt 3), 2 / (3 * sqrt 3)) := 
sorry

end abs_sum_ge_sqrt_three_over_two_l349_349734


namespace prime_count_30_to_50_l349_349405

def is_prime (n: ℕ) : Prop :=
  n > 1 ∧ ∀ m: ℕ, m ∣ n → m = 1 ∨ m = n

def primes_in_range (a b: ℕ) : list ℕ :=
  list.filter is_prime (list.range' a (b - a + 1))

theorem prime_count_30_to_50 : (primes_in_range 30 50).length = 5 :=
by sorry

end prime_count_30_to_50_l349_349405


namespace sum_stepstool_numbers_up_to_300_l349_349250

-- Define the number of divisors of a number
def num_divisors (n : ℕ) : ℕ :=
  (list.range (n + 1)).filter (λ x, n % x = 0).length

-- Define what it means to be a stepstool number
def is_stepstool_number (n : ℕ) : Prop :=
  num_divisors(n) = num_divisors(n + 1) - 1

-- Define a set containing all stepstool numbers less than 300
def stepstool_numbers_up_to_300 : finset ℕ :=
  (finset.range 300).filter is_stepstool_number

-- Sum of all stepstool numbers less than 300
theorem sum_stepstool_numbers_up_to_300 : 
  (stepstool_numbers_up_to_300.sum id) = 687 :=
sorry

end sum_stepstool_numbers_up_to_300_l349_349250


namespace line_BC_l349_349766

noncomputable def Point := (ℝ × ℝ)
def A : Point := (-1, -4)
def l₁ := { p : Point | p.2 + 1 = 0 }
def l₂ := { p : Point | p.1 + p.2 + 1 = 0 }
def A' : Point := (-1, 2)
def A'' : Point := (3, 0)

theorem line_BC :
  ∃ (c₁ c₂ c₃ : ℝ), c₁ ≠ 0 ∨ c₂ ≠ 0 ∧
  ∀ (p : Point), (c₁ * p.1 + c₂ * p.2 + c₃ = 0) ↔ p ∈ { x | x = A ∨ x = A'' } :=
by sorry

end line_BC_l349_349766


namespace average_number_of_fish_is_75_l349_349517

-- Define the conditions
def BoastPool_fish := 75
def OnumLake_fish := BoastPool_fish + 25
def RiddlePond_fish := OnumLake_fish / 2

-- Prove the average number of fish
theorem average_number_of_fish_is_75 :
  (BoastPool_fish + OnumLake_fish + RiddlePond_fish) / 3 = 75 :=
by
  sorry

end average_number_of_fish_is_75_l349_349517


namespace quadratic_polynomial_P_l349_349114

noncomputable def P : Polynomial := {
  to_fun := λ x : ℝ, x^2 - x + 2,
  degree := 2
}

theorem quadratic_polynomial_P (P : ℝ → ℝ) 
  (h : ∀ x, P (P x) = x^4 - 2 * x^3 + 4 * x^2 - 3 * x + 4) : 
  P 8 = 58 := 
by
  sorry

end quadratic_polynomial_P_l349_349114


namespace find_k_l349_349870

variables (A B C M : Point)

-- Define the points and distances
def AB := dist A B = 5
def BC := dist B C = 12
def AC := dist A C = 13

-- Define M as the midpoint of AC
def midpoint_def (M A C : Point) := 2 * dist M A = dist A C ∧ 2 * dist M C = dist A C

-- Given that BM = k * sqrt 2
variables (k : ℝ)
def BM := dist B M = k * Real.sqrt 2

-- The goal is to find k
theorem find_k : 
  (AB ∧ BC ∧ AC ∧ midpoint_def M A C ∧ BM) → 
  k = (13 * Real.sqrt 2) / 4 :=
by
  sorry

end find_k_l349_349870


namespace domain_of_f_l349_349574

noncomputable def f (x : ℝ) := 1 / ((x - 3) + (x - 6))

theorem domain_of_f :
  (∀ x : ℝ, x ≠ 9/2 → ∃ y : ℝ, f x = y) ∧ (∀ x : ℝ, x = 9/2 → ¬ (∃ y : ℝ, f x = y)) :=
by
  sorry

end domain_of_f_l349_349574


namespace projection_matrix_is_correct_l349_349321

open matrix
open_locale matrix

def u : vector ℚ 2 := ![3, 1]

def proj_matrix (u : vector ℚ 2) : matrix (fin 2) (fin 2) ℚ :=
  let u_dot_u := (u ⬝ u) 0 0 in
  (1 / u_dot_u) • (u ⬝ u.transpose)

theorem projection_matrix_is_correct :
  proj_matrix u = ![![9/10, 3/10], ![3/10, 1/10]] :=
by {
  sorry
}

end projection_matrix_is_correct_l349_349321


namespace number_of_primes_between_30_and_50_l349_349414

/-- 
  Prove that there are exactly 5 prime numbers in the range from 30 to 50. 
  These primes are 31, 37, 41, 43, and 47.
-/
theorem number_of_primes_between_30_and_50 : 
  (Finset.filter Nat.Prime (Finset.range 51)).card - 
  (Finset.filter Nat.Prime (Finset.range 31)).card = 5 := 
by 
  sorry

end number_of_primes_between_30_and_50_l349_349414


namespace find_Y_l349_349744

theorem find_Y (Y : ℝ) (h : (100 + Y / 90) * 90 = 9020) : Y = 20 := 
by 
  sorry

end find_Y_l349_349744


namespace count_primes_between_30_and_50_l349_349409

-- Define the range of numbers from 30 to 50
def range_30_to_50 := Set.of_list (List.range' 30 (51 - 30))

-- Define a predicate to check if a number is prime
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

-- Extract all prime numbers in the specified range
def primes_between_30_and_50 : List ℕ :=
  List.filter is_prime (Set.toList range_30_to_50)

theorem count_primes_between_30_and_50 : primes_between_30_and_50.length = 5 :=
by
  -- The proof goes here
  sorry

end count_primes_between_30_and_50_l349_349409


namespace unique_zero_point_of_quadratic_l349_349445

theorem unique_zero_point_of_quadratic (a : ℝ) :
  (∀ x : ℝ, (a * x^2 - x - 1 = 0 → x = -1)) ↔ (a = 0 ∨ a = -1 / 4) :=
by
  sorry

end unique_zero_point_of_quadratic_l349_349445


namespace negation_equiv_l349_349381

variable (f : ℝ → ℝ)

theorem negation_equiv :
  ¬ (∀ (x₁ x₂ : ℝ), (f x₂ - f x₁) * (x₂ - x₁) ≥ 0) ↔
  ∃ (x₁ x₂ : ℝ), (f x₂ - f x₁) * (x₂ - x₁) < 0 := by
sorry

end negation_equiv_l349_349381


namespace number_of_subsets_of_P_l349_349382

-- Define the original set M
def M : Set ℕ := {1, 2, 3, 4}

-- Define the set P based on the given conditions
def P : Set ℕ := {x ∈ M | 2 * x ∉ M}

-- Provide the theorem statement proving the number of subsets of P
theorem number_of_subsets_of_P : (P.to_finset.powerset.card = 4) :=
sorry

end number_of_subsets_of_P_l349_349382


namespace simplify_and_evaluate_expression_l349_349529

theorem simplify_and_evaluate_expression 
  (a b : ℚ) 
  (ha : a = 2) 
  (hb : b = 1 / 3) : 
  (a / (a - b)) * ((1 / b) - (1 / a)) + ((a - 1) / b) = 6 := 
by
  -- Place the steps verifying this here. For now:
  sorry

end simplify_and_evaluate_expression_l349_349529


namespace sector_central_angle_l349_349789

theorem sector_central_angle (r θ : ℝ) (h1 : 2 * r + r * θ = 6) (h2 : 0.5 * r * r * θ = 2) : θ = 1 ∨ θ = 4 :=
sorry

end sector_central_angle_l349_349789


namespace trigonometric_identity_l349_349523

open Real

theorem trigonometric_identity (α φ : ℝ) :
  cos α ^ 2 + cos φ ^ 2 + cos (α + φ) ^ 2 - 2 * cos α * cos φ * cos (α + φ) = 1 :=
sorry

end trigonometric_identity_l349_349523


namespace quarters_value_percentage_l349_349599

theorem quarters_value_percentage (dimes_count quarters_count dimes_value quarters_value : ℕ) (h1 : dimes_count = 75)
    (h2 : quarters_count = 30) (h3 : dimes_value = 10) (h4 : quarters_value = 25) :
    (quarters_count * quarters_value * 100) / (dimes_count * dimes_value + quarters_count * quarters_value) = 50 := 
by
    sorry

end quarters_value_percentage_l349_349599


namespace select_female_athletes_l349_349253

theorem select_female_athletes (males females sample_size total_size : ℕ)
    (h1 : males = 56) (h2 : females = 42) (h3 : sample_size = 28)
    (h4 : total_size = males + females) : 
    (females * sample_size / total_size = 12) := 
by
  sorry

end select_female_athletes_l349_349253


namespace part1_part2_l349_349799

-- Definitions for the problem
def f (x a : ℝ) : ℝ := |x - a| + 3 * x

-- Part (1)
theorem part1 (x : ℝ) (h : f x 1 ≥ 3 * x + 2) : x ≥ 3 ∨ x ≤ -1 :=
sorry

-- Part (2)
theorem part2 (h : ∀ x, f x a ≤ 0 → x ≤ -1) : a = 2 :=
sorry

end part1_part2_l349_349799


namespace scientific_notation_of_8450_l349_349867

theorem scientific_notation_of_8450 :
  ∃ (a : ℝ) (n : ℤ), (1 ≤ |a| ∧ |a| < 10) ∧ (8450 : ℝ) = a * 10^n ∧ (a = 8.45) ∧ (n = 3) :=
sorry

end scientific_notation_of_8450_l349_349867


namespace area_of_rectangle_l349_349270

theorem area_of_rectangle (l w : ℝ) (h : l + w = 15) : 
  let s := 7 in 
  l = 8 → 
  w = s → 
  l * w = 56 := 
by 
  intro l w h s hs hs' 
  have l_eq_8 : l = 8 := hs 
  have w_eq_7 : w = s := hs' 
  rw [l_eq_8, w_eq_7]
  exact mul_eq_of_eq l_eq_8 w_eq_7

end area_of_rectangle_l349_349270


namespace min_value_func_l349_349156

def func (x : ℝ) : ℝ := x + 1 / (x + 1)

theorem min_value_func : ∀ x : ℝ, 0 ≤ x → 1 ≤ func x := by
  sorry

end min_value_func_l349_349156


namespace part1_part2_l349_349369

open Real

noncomputable def f (x : ℝ) : ℝ :=
  (x - 1) * log (x - 1)

theorem part1 {λ : ℝ} (h : ∀ x ∈ Ioi 1, f(x) ≥ λ * (x - 2)) : λ = 1 :=
sorry

theorem part2 {a x1 x2 : ℝ} (h : f(x1 + 1) = a ∧ f(x2 + 1) = a) :
  abs (x1 - x2) < (3/2) * a + 1 + (1 / (2 * exp 3)) :=
sorry

end part1_part2_l349_349369


namespace regular_octagon_diagonals_l349_349236

theorem regular_octagon_diagonals : 
  let n := 8 in (n * (n - 3)) / 2 = 20 :=
by
  let n := 8
  calc
    (n * (n - 3)) / 2 = (8 * (8 - 3)) / 2  : by rw n
                   ... = (8 * 5) / 2         : by norm_num
                   ... = 40 / 2              : by norm_num
                   ... = 20                  : by norm_num

end regular_octagon_diagonals_l349_349236


namespace option_A_option_B_option_D_l349_349041

-- Definitions to represent the problem conditions
variables {a b c : ℝ} {A B C : ℝ}
variables {ΔABC : Prop} -- Assume ΔABC implies angles are A, B, C and sides are a, b, c
variables (acute_triangle : ΔABC → A < π / 2 ∧ B < π / 2 ∧ C < π / 2)

-- Proof goal 1: If a > b, then sin A > sin B
theorem option_A (h : ΔABC) (h1 : a > b) : sin A > sin B := 
sorry

-- Proof goal 2: If sin A > sin B, then A > B
theorem option_B (h : ΔABC) (h1 : sin A > sin B) : A > B := 
sorry

-- Proof goal 3: If triangle ABC is acute, then sin A > cos B
theorem option_D (h : ΔABC) (h1 : acute_triangle h) : sin A > cos B := 
sorry

end option_A_option_B_option_D_l349_349041


namespace number_of_primes_between_30_and_50_l349_349436

-- Define what it means to be a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the interval condition
def in_interval (n : ℕ) : Prop :=
  30 ≤ n ∧ n ≤ 50

-- Define the count of prime numbers in the interval
def prime_count_in_interval : ℕ :=
  (List.range' 30 21).countp (λ n, is_prime n)

-- We state that the above count is equal to 5
theorem number_of_primes_between_30_and_50 : prime_count_in_interval = 5 :=
  sorry

end number_of_primes_between_30_and_50_l349_349436


namespace cos_vertex_angle_of_isosceles_triangle_area_of_right_triangle_l349_349363

-- Problem 1: Isosceles triangle
theorem cos_vertex_angle_of_isosceles_triangle (A B C : ℝ) (a b c : ℝ)
  (hB : sin B^2 = 2 * sin A * sin C)
  (h_iso : a = b) :
  cos C = 7 / 8 :=
sorry

-- Problem 2: Right triangle with given side length
theorem area_of_right_triangle (A B C : ℝ) (a b c : ℝ)
  (hB : sin B^2 = 2 * sin A * sin C)
  (h_right : B = π / 2)
  (h_bc : c = sqrt 2) :
  (1 / 2) * a * c = 1 :=
sorry

end cos_vertex_angle_of_isosceles_triangle_area_of_right_triangle_l349_349363


namespace integer_part_of_sum_is_2_l349_349549

-- Define the sequence a_n
def a : ℕ → ℝ
| 0 => 1/3
| (n + 1) => a n * (a n + 1)

-- Define the sum S
def S (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, 1 / (a (i + 1) + 1)

-- The theorem to prove
theorem integer_part_of_sum_is_2 :
  ⌊ S 2016 ⌋ = 2 :=
by
  sorry

end integer_part_of_sum_is_2_l349_349549


namespace candles_lit_time_correct_l349_349172

noncomputable def candle_time : String :=
  let initial_length := 1 -- Since the length is uniform, we use 1
  let rateA := initial_length / (6 * 60) -- Rate at which Candle A burns out
  let rateB := initial_length / (8 * 60) -- Rate at which Candle B burns out
  let t := 320 -- The time in minutes that satisfy the condition
  let time_lit := (16 * 60 - t) / 60 -- Convert minutes to hours
  if time_lit = 10 + 40 / 60 then "10:40 AM" else "Unknown"

theorem candles_lit_time_correct :
  candle_time = "10:40 AM" := 
by
  sorry

end candles_lit_time_correct_l349_349172


namespace primes_between_30_and_50_l349_349396

theorem primes_between_30_and_50 : (Finset.card (Finset.filter Nat.Prime (Finset.Ico 30 51))) = 5 :=
by
  sorry

end primes_between_30_and_50_l349_349396


namespace probability_B_given_A_l349_349326

-- Definitions of events A and B in a sample space of coin flips.
def coin := {true, false}  -- Assuming true represents heads, false represents tails
def eventA (flips: list bool) := flips.head? = some false  -- Tails on the first flip
def eventB (flips: list bool) := flips.tail.head? = some true  -- Heads on the second flip

-- Known probabilities
def P (event: set (list bool)) := (1 : ℝ) / 2  -- Probability of heads or tails in one coin flip

-- Assuming independence, calculate joint probability of A and B
def P_A_and_B : ℝ := P {flips | eventA flips ∧ eventB flips}

-- Probability of A
def P_A : ℝ := P {flips | eventA flips}

-- Conditional probability P(B|A)
def conditional_probability : ℝ := P_A_and_B / P_A

-- Statement to prove
theorem probability_B_given_A : conditional_probability = (1 : ℝ) / 2 := by
  sorry

end probability_B_given_A_l349_349326


namespace remainder_4059_div_32_l349_349990

theorem remainder_4059_div_32 : 4059 % 32 = 27 := by
  sorry

end remainder_4059_div_32_l349_349990


namespace trig_identity_solution_l349_349336

theorem trig_identity_solution 
  (alpha : ℝ) 
  (h : Real.sin (π / 6 - alpha) = 1 / 3) : 
  2 * Real.cos^2 (π / 6 + alpha / 2) - 1 = 1 / 3 := 
by 
  sorry

end trig_identity_solution_l349_349336


namespace sum_possible_remainders_l349_349713

theorem sum_possible_remainders : 
  let m (k : ℕ) := 11111 * k + 43210 in
  (∑ k in Finset.range 6, (m k % 13)) = 51 :=
by
  let m := λ k : ℕ, 11111 * k + 43210
  have mod_identity : ∀ (k : ℕ), m k % 13 = (k + 6) % 13 := sorry
  calc 
    ∑ k in Finset.range 6, (m k % 13) = ∑ k in Finset.range 6, (k + 6 % 13) : by congr; funext; apply mod_identity
                            ... = ∑ k in Finset.range 6, (k + 6) : sorry
                            ... = (∑ k in Finset.range 6, k) + 6 * 6 : sorry
                            ... = 15 + 36 : by norm_num
                            ... = 51 : by norm_num

end sum_possible_remainders_l349_349713


namespace min_value_xyz_l349_349499

-- Definition of the problem
theorem min_value_xyz (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x * y * z = 108):
  x^2 + 9 * x * y + 9 * y^2 + 3 * z^2 ≥ 324 :=
sorry

end min_value_xyz_l349_349499


namespace count_primes_between_30_and_50_l349_349424

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between_30_and_50 : List ℕ :=
  [31, 37, 41, 43, 47]

theorem count_primes_between_30_and_50 : 
  (primes_between_30_and_50.filter is_prime).length = 5 :=
by
  sorry

end count_primes_between_30_and_50_l349_349424


namespace gridPolygon_side_longer_than_one_l349_349520

-- Define the structure of a grid polygon
structure GridPolygon where
  area : ℕ  -- Area of the grid polygon
  perimeter : ℕ  -- Perimeter of the grid polygon
  no_holes : Prop  -- Polyon does not contain holes

-- Definition of a grid polygon with specific properties
def specificGridPolygon : GridPolygon :=
  { area := 300, perimeter := 300, no_holes := true }

-- The theorem we want to prove that ensures at least one side is longer than 1
theorem gridPolygon_side_longer_than_one (P : GridPolygon) (h_area : P.area = 300) (h_perimeter : P.perimeter = 300) (h_no_holes : P.no_holes) : ∃ side_length : ℝ, side_length > 1 :=
  by
  sorry

end gridPolygon_side_longer_than_one_l349_349520


namespace main_theorem_l349_349888

variables {A B C D E F H : Type*}
variables [acute_angled_triangle: a b c : ℝ]
variables {AD BE CF : ℝ}
variables {AH BH CH : ℝ}

-- Given conditions in a)
variable (is_acute_angled_triangle_ABC : ∀ (A B C : Type*) (a b c : ℝ), true)
variable (are_altitudes : ∀ (AD BE CF : ℝ), true)
variable (is_orthocenter_H : ∀ (H : Type*) (A B C : Type*) (AD BE CF : ℝ), true)

-- Main theorem statement from c)
theorem main_theorem (h_acute : is_acute_angled_triangle_ABC A B C a b c)
                     (h_altitudes : are_altitudes AD BE CF)
                     (h_orthocenter : is_orthocenter_H H A B C AD BE CF) :
  (AB * AC + BC * CA + CA * CB) ≤ 2 * (AH * AD + BH * BE + CH * CF) :=
sorry

end main_theorem_l349_349888


namespace intersection_points_l349_349737

noncomputable def circle1 (x y : ℝ) : Prop := (x - 2)^2 + (y - 10)^2 = 50
noncomputable def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2 * (x - y) - 18 = 0

theorem intersection_points : 
  (circle1 3 3 ∧ circle2 3 3) ∧ (circle1 (-3) 5 ∧ circle2 (-3) 5) :=
by sorry

end intersection_points_l349_349737


namespace correct_average_is_50_l349_349605

-- Declare the assumptions
variable (n : ℕ) (incorrect_avg : ℝ) (incorrect_number correct_number : ℝ)

-- Assume n=10, incorrect_avg=46, incorrect_number=25, correct_number=65
axiom H1 : n = 10
axiom H2 : incorrect_avg = 46
axiom H3 : incorrect_number = 25
axiom H4 : correct_number = 65

-- Define the correct average
def correct_avg := (incorrect_avg * n - incorrect_number + correct_number) / n

-- State the theorem
theorem correct_average_is_50 : correct_avg n incorrect_avg incorrect_number correct_number = 50 :=
by
  rw [H1, H2, H3, H4]
  sorry

end correct_average_is_50_l349_349605


namespace question_evaluation_l349_349866

def f (x : ℝ) : ℝ := (x^3 + 2*x^2 - 3*x - 6) / (x^4 - x^3 - 6*x^2)

def number_of_holes (f : ℝ → ℝ) : ℕ := 0        -- Assumes the function does not have cancelable factors
def number_of_vertical_asymptotes (f : ℝ → ℝ) : ℕ := 3
def number_of_horizontal_asymptotes (f : ℝ → ℝ) : ℕ := 1
def number_of_oblique_asymptotes (f : ℝ → ℝ) : ℕ := 0

theorem question_evaluation :
  let a := number_of_holes f
  let b := number_of_vertical_asymptotes f
  let c := number_of_horizontal_asymptotes f
  let d := number_of_oblique_asymptotes f
  a + 2 * b + 3 * c + 4 * d = 9 :=
by
  -- Proof placeholder
  sorry

end question_evaluation_l349_349866


namespace triangle_inequality_positive_difference_l349_349868

theorem triangle_inequality_positive_difference (x : ℕ) (h1 : 3 < x) (h2 : x < 17) : 
  (Nat.greatest (4 <= x)) - (Nat.least (x <= 16)) = 12 :=
by
  sorry

end triangle_inequality_positive_difference_l349_349868


namespace sum_of_series_eq_one_fourth_l349_349729

theorem sum_of_series_eq_one_fourth :
  (∑' n : ℕ, n ≠ 0 → (3^n / (1 + 3^n + 3^(n + 1) + 3^(2*n + 1)))) = 1 / 4 :=
by
  sorry

end sum_of_series_eq_one_fourth_l349_349729


namespace num_three_digit_primes_end_with_3_l349_349005

theorem num_three_digit_primes_end_with_3 : 
  (Finset.filter (λ n : ℕ, n.digits.length = 3 ∧ n % 10 = 3 ∧ n.Prime) (Finset.range 1000)).card = 70 := by
sorry

end num_three_digit_primes_end_with_3_l349_349005


namespace value_at_11_l349_349274

def quadratic (p : ℝ → ℝ) (x0 : ℝ) := ∀ x, p (2 * x0 - x) = p x

theorem value_at_11 (p : ℝ → ℝ) (a : ℝ):
  quadratic p 7.5 ->
  p 4 = -4 ->
  p 11 = -4 :=
by
  intros h_symmetry h_p4
  have h_symmetry_at_11 : p 11 = p 4 := h_symmetry 11
  rw h_p4 at h_symmetry_at_11
  exact h_symmetry_at_11

end value_at_11_l349_349274


namespace binom_10_2_eq_45_l349_349702

-- Definitions used in the conditions
def binom (n k : ℕ) := n.choose k

-- The statement that needs to be proven
theorem binom_10_2_eq_45 : binom 10 2 = 45 :=
by
  sorry

end binom_10_2_eq_45_l349_349702


namespace trajectory_proof_max_area_proof_l349_349762

noncomputable def trajectory_equation (x y : ℝ) : Prop :=
  (x^2 / 2) + y^2 = 1

noncomputable def max_area_of_triangle (x1 y1 x2 y2 : ℕ) (k b : ℝ) (O A B : ℝ × ℝ) (S : ℝ) : Prop :=
  let A := (x1, y1)
  let B := (x2, y2)
  let D := (0, 1/2)
  let O := (0, 0)
  let triangle_area := S = (sqrt 2 / 2) in
  (y1 - y2 = k * (x1 - x2)) ∧ 
  (y1 + y2 = k * (x1 + x2) + 2 * b) ∧ 
  ((x1 - x2) * (x1 + x2) + (y1 - y2) * (y1 + y2 - 1) = 0) ∧
  S = max_triangle_area A B O

theorem trajectory_proof (x y : ℝ) (R : ℝ) (N : ℝ × ℝ) (M : ℝ × ℝ) :
  (x + 1) ^ 2 + y ^ 2 = 8 ∧ N = (1, 0) ∧ M = (-1, 0) → trajectory_equation x y := by
  sorry

theorem max_area_proof (x1 y1 x2 y2 : ℕ) (k b : ℝ) (O A B : ℝ × ℝ) :
  let midpoint_condition := (x1^2 + (y1 - 1/2)^2 = x2^2 + (y2 - 1/2)^2) in
  (midpoint_condition) → max_area_of_triangle x1 y1 x2 y2 k b O A B := by
  sorry

end trajectory_proof_max_area_proof_l349_349762


namespace primes_between_30_and_50_l349_349394

theorem primes_between_30_and_50 : (Finset.card (Finset.filter Nat.Prime (Finset.Ico 30 51))) = 5 :=
by
  sorry

end primes_between_30_and_50_l349_349394


namespace interval_of_monotonic_increase_l349_349951

-- Define the function f
def f (x : ℝ) : ℝ := log (1 / 2) (x^2 - 6 * x + 11)

-- Define the theorem to prove the interval of monotonic increase
theorem interval_of_monotonic_increase :
  {x : ℝ | ∃ I : set ℝ, I = {t : ℝ | t ≤ 3} ∧ ∀ x ∈ I, f x = log (1 / 2) (x^2 - 6 * x + 11)} = {x : ℝ | x ≤ 3} :=
sorry

end interval_of_monotonic_increase_l349_349951


namespace rocky_miles_total_l349_349856

-- Defining the conditions
def m1 : ℕ := 4
def m2 : ℕ := 2 * m1
def m3 : ℕ := 3 * m2

-- The statement to be proven
theorem rocky_miles_total : m1 + m2 + m3 = 36 := by
  sorry

end rocky_miles_total_l349_349856


namespace parking_space_area_l349_349205

theorem parking_space_area (L W : ℕ) (h1 : L = 9) (h2 : L + 2 * W = 37) : L * W = 126 := by
  sorry

end parking_space_area_l349_349205


namespace return_amount_is_25_l349_349241

def workers_return_amount (x : ℝ) :=
  let days_worked := 6
  let earnings := 600
  let days_not_worked := 24
  let return_amount := 24 * x
  earnings = return_amount

theorem return_amount_is_25 : workers_return_amount 25 :=
by
  -- Conditions
  let days_worked := 6
  let earnings := 600
  let days_not_worked := 24
  let return_amount := 24 * 25
  -- Proof
  show earnings = return_amount
  sorry

end return_amount_is_25_l349_349241


namespace can_invent_1001_sad_stories_l349_349134

-- Definitions
def is_natural (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 17

def is_sad_story (a b c : ℕ) : Prop :=
  ∀ x y : ℤ, a * x + b * y ≠ c

-- The Statement
theorem can_invent_1001_sad_stories :
  ∃ stories : Finset (ℕ × ℕ × ℕ),
    (∀ (a b c : ℕ), (a, b, c) ∈ stories → is_natural a ∧ is_natural b ∧ is_natural c ∧ is_sad_story a b c) ∧
    stories.card ≥ 1001 :=
by
  sorry

end can_invent_1001_sad_stories_l349_349134


namespace vector_at_t1_l349_349629

theorem vector_at_t1 {α : Type*} [Add α] [Mul α] [Neg α]
  (a5 : α) (b5 : α) (a6 : α) (b6 : α) (result_x : α) (result_y : α) :
  let p5 := (2, 1)   -- vector at t = 5
  let p6 := (5, -7)  -- vector at t = 6
  let pt := (result_x, result_y)  -- vector at t = 1
  p5 = (a5, b5) → p6 = (a6, b6) →
  pt = (-40, 113) :=
by
  intros h1 h2
  sorry

end vector_at_t1_l349_349629


namespace game_is_fair_l349_349202

theorem game_is_fair :
  -- Conditions: 
  -- 1. 2 dice are rolled by Xénia and Yvonne respectively.
  -- 2. The game is fair if each player has an equal number of winning outcomes.
  -- Prove the game is fair.
  ∀ (die1 die2 die3 die4 : ℤ), 
    (1 ≤ die1 ∧ die1 ≤ 6) ∧ (1 ≤ die2 ∧ die2 ≤ 6) ∧ (1 ≤ die3 ∧ die3 ≤ 6) ∧ (1 ≤ die4 ∧ die4 ≤ 6)
    → 
    let sum := die1 + die2 + die3 + die4 in 
    (∃ (outcome : bool), (outcome = true ↔ sum % 2 = 1) ∧ (outcome = false ↔ sum % 2 = 0)) :=
  sorry

end game_is_fair_l349_349202


namespace price_per_working_game_eq_six_l349_349052

-- Define the total number of video games
def total_games : Nat := 10

-- Define the number of non-working video games
def non_working_games : Nat := 8

-- Define the total income from selling working games
def total_earning : Nat := 12

-- Calculate the number of working video games
def working_games : Nat := total_games - non_working_games

-- Define the expected price per working game
def expected_price_per_game : Nat := 6

-- Theorem statement: Prove that the price per working game is $6
theorem price_per_working_game_eq_six :
  total_earning / working_games = expected_price_per_game :=
by sorry

end price_per_working_game_eq_six_l349_349052


namespace ball_bounces_17_times_to_reach_below_2_feet_l349_349225

theorem ball_bounces_17_times_to_reach_below_2_feet:
  ∃ k: ℕ, (∀ n, n < k → (800 * ((2: ℝ) / 3) ^ n) ≥ 2) ∧ (800 * ((2: ℝ) / 3) ^ k < 2) ∧ k = 17 :=
by
  sorry

end ball_bounces_17_times_to_reach_below_2_feet_l349_349225


namespace find_S10_l349_349901

noncomputable def S (n : ℕ) : ℤ := 2 * (-2 ^ (n - 1)) + 1

theorem find_S10 : S 10 = -1023 :=
by
  sorry

end find_S10_l349_349901


namespace find_k_Uk_l349_349637

def is_prime (n : ℕ) : Prop :=
  n ≥ 2 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def next_unassigned (sequence : List ℕ) : ℕ :=
  (Nat.find (λ n, n > 0 ∧ n ∉ sequence)).val

def next_prime_unassigned (sequence : List ℕ) : ℕ :=
  (Nat.find (λ n, is_prime n ∧ n ∉ sequence)).val

noncomputable def U : ℕ → ℕ
| 1     := 2
| (n+1) := if is_prime (U n) then next_unassigned (List.range (n+1)).map U else next_prime_unassigned (List.range (n+1)).map U

theorem find_k_Uk (k : ℕ) (U : ℕ → ℕ) (hU : ∀ n, U (n+1) = if is_prime (U n) then next_unassigned (List.range (n+1)).map U else next_prime_unassigned (List.range (n+1)).map U)
 (hk : U (k+1) - U k > 10) : k = 18 ∧ U 18 = 15 ∧ k * U k = 270 := sorry

end find_k_Uk_l349_349637


namespace banana_permutations_l349_349003

theorem banana_permutations :
  let n := 6
  let n_a := 3
  let n_n := 2
  let n_b := 1
  (nat.factorial n) / ((nat.factorial n_a) * (nat.factorial n_n) * (nat.factorial n_b)) = 60 :=
by
  -- Defining the variables
  let n := 6
  let n_a := 3
  let n_n := 2
  let n_b := 1
  -- The statement we need to prove
  have equation := (nat.factorial n) / ((nat.factorial n_a) * (nat.factorial n_n) * (nat.factorial n_b)) = 60
  -- Sorry means the proof is omitted.
  sorry

end banana_permutations_l349_349003


namespace least_children_to_guarantee_all_colors_l349_349165

theorem least_children_to_guarantee_all_colors :
  ∀ (pencils : ℕ) (colors : ℕ) (children : ℕ) (pencils_per_child : ℕ) (pencils_per_color : ℕ),
  pencils = 24 → colors = 4 → children = 6 → pencils_per_child = 4 → pencils_per_color = 6 →
  (∃ n : ℕ, n ≤ children ∧ (∀ chosen_children : Finset ℕ, chosen_children.card = n → 
    ∃ (color_set : Finset ℕ), color_set.card = colors ∧ ∀ c ∈ color_set, ∃ k ∈ chosen_children, pencil_color k == c) →
    n = 5) :=
by 
  sorry

end least_children_to_guarantee_all_colors_l349_349165


namespace diameter_of_circle_l349_349987

theorem diameter_of_circle (A : ℝ) (hA : A = 225 * Real.pi) : ∃ d : ℝ, d = 30 := 
by
  have h1 : 225 * Real.pi = Real.pi * (15 ^ 2) := by rw [Real.pi_mul, Real.norm_eq_of_mem_real]; norm_num
  have h2 : A = Real.pi * (15 ^ 2) := by rw [hA, h1]
  let r : ℝ := 15
  have hr : r = 15 := rfl
  sorry

end diameter_of_circle_l349_349987


namespace proof_tangent_normal_lines_l349_349973

def parametric_curve_tangent_normal (t0 : ℝ) :=
  let x := (λ t : ℝ, 2 * t - t ^ 2)
  let y := (λ t : ℝ, 3 * t - t ^ 3)
  let pt := (x t0, y t0)
  let xt' := (λ t : ℝ, 2 - 2 * t)
  let yt' := (λ t : ℝ, 3 - 3 * t ^ 2)
  let dy_dx := λ t : ℝ, yt' t / xt' t
  let slope := dy_dx t0
  let tangent_eq := λ x y : ℝ, y - pt.2 = slope * (x - pt.1)
  let normal_eq := λ x y : ℝ, - (1 / slope) * (x - pt.1) = y - pt.2
  (tangent_eq, normal_eq)

theorem proof_tangent_normal_lines:
  parametric_curve_tangent_normal 1 = (λ x y : ℝ, y = 3 * x - 1, λ x y : ℝ, y = -1/3 * x + 7/3) :=
by 
  -- We state that a proof is required here
  sorry

end proof_tangent_normal_lines_l349_349973


namespace perimeter_last_triangle_l349_349492

noncomputable def T₁_side_lengths : ℕ × ℕ × ℕ := (1005, 1006, 1007)

structure Triangle (α : Type) :=
  (a b c : α)

def next_triangle (T : Triangle ℕ) : Triangle ℚ :=
  let x := (T.a + T.b - T.c) / 2
  let y := (T.b + T.c - T.a) / 2
  let z := (T.c + T.a - T.b) / 2
  ⟨x, y, z⟩

theorem perimeter_last_triangle : ∃ n ≥ 1, 
  ∀ T : Triangle ℕ, 
  let T_next := Iterated next_triangle T n in
  T_next.a + T_next.b + T_next.c = 1509 / 64
:=
sorry

end perimeter_last_triangle_l349_349492


namespace task2_probability_l349_349199

variable (P_task1_on_time P_task2_on_time : ℝ)

theorem task2_probability 
  (h1 : P_task1_on_time = 5 / 8)
  (h2 : (P_task1_on_time * (1 - P_task2_on_time)) = 0.25) :
  P_task2_on_time = 3 / 5 := by
  sorry

end task2_probability_l349_349199


namespace larger_integer_of_two_integers_diff_8_prod_120_l349_349978

noncomputable def larger_integer (a b : ℕ) : ℕ :=
if a > b then a else b

theorem larger_integer_of_two_integers_diff_8_prod_120 (a b : ℕ) 
  (h_diff : a - b = 8) 
  (h_product : a * b = 120) 
  (h_positive_a : 0 < a) 
  (h_positive_b : 0 < b) : larger_integer a b = 20 := by
  sorry

end larger_integer_of_two_integers_diff_8_prod_120_l349_349978


namespace ellipse_equation_exists_line_equation_exists_l349_349656

-- Definitions of the given conditions

def ellipse_eq (x y a b : ℝ) : Prop :=
  (x^2 / a^2 + y^2 / b^2 = 1) ∧ (a > b) ∧ (b > 0)

def passes_through_A (x y : ℝ) (a b : ℝ) : Prop :=
  ellipse_eq x y a b ∧ (x = 1) ∧ (y = 3/2)

def eccentricity (a b e : ℝ) : Prop :=
  (e = 1/2) ∧ (b^2 / a^2 = 3/4)

-- Theorems to be proved
theorem ellipse_equation_exists (a b : ℝ) :
  (passes_through_A 1 (3/2) a b) ∧ (eccentricity a b (1/2)) → (ellipse_eq 1 (3/2) 2 (√3)) :=
sorry

theorem line_equation_exists (a b : ℝ) (k : ℝ):
  (passes_through_A 1 (3/2) a b) ∧ (eccentricity a b (1/2)) ∧ ((12 * Real.sqrt 1 + k^2) / (4*k^2 + 3) = 12*Real.sqrt 2/7) → (k = 1 ∨ k = -1) :=
sorry

end ellipse_equation_exists_line_equation_exists_l349_349656


namespace inequality_f_geq_neg_2_range_m_for_inequality_l349_349802

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := 3 * x^2 - 2 * m * x - 1
noncomputable def g (x : ℝ) : ℝ := |x| - (7 / 4)

theorem inequality_f_geq_neg_2 (x m : ℝ) : 
  (f x m >= -2) = 
  if -real.sqrt 3 <= m ∧ m <= real.sqrt 3 
  then true 
  else (x < (m - real.sqrt (m^2 - 3)) / 3 ∨ x > (m + real.sqrt (m^2 - 3)) / 3) := 
sorry

theorem range_m_for_inequality (m : ℝ) : 
  (∀ (x : ℝ), 0 <= x ∧ x <= 2 → f x m >= g x) ↔ m <= 1 := 
sorry

end inequality_f_geq_neg_2_range_m_for_inequality_l349_349802


namespace max_inspections_240_poles_l349_349627

theorem max_inspections_240_poles : 
  ∀ (poles : ℕ), poles = 240 → 
  let segments := poles - 1 in
  (segments / 3) + 1 = 5 :=
by 
  intros poles h
  let segments := poles - 1
  have h_segments : segments = 239, from nat.sub_add_cancel (nat.succ_le_succ (nat.zero_le 239))
  rw h_segments
  have h_div3 : (239 / 3) = 79, by sorry
  have h_div3_plus_1 : (239 / 3) + 1 = 80, by rw [h_div3, nat.succ_eq_add_one]
  sorry

end max_inspections_240_poles_l349_349627


namespace number_of_n_with_fn_eq_n_l349_349289

def f : ℕ → ℕ 
| 1 := 1
| 3 := 3
| (2 * n) := f n
| (4 * n + 1) := 2 * f (2 * n + 1) - f n
| (4 * n + 3) := 3 * f (2 * n + 1) - 2 * f n

theorem number_of_n_with_fn_eq_n :
  ∃! k, k = 92 ∧ ∀ n, (1 ≤ n ∧ n ≤ 1988) → (f n = n ↔ k = 92) :=
sorry

end number_of_n_with_fn_eq_n_l349_349289


namespace matrix_swap_transformation_l349_349344

theorem matrix_swap_transformation (n : ℕ) (h : n ≥ 2)
  (A B : matrix (fin n) (fin n) ℕ)
  (hA : ∀ x : ℕ, x ∈ finset.univ.bUnion (λ i, finset.univ.image (λ j, A i j)) ↔ x ∈ finset.range (n * n))
  (hB : ∀ x : ℕ, x ∈ finset.univ.bUnion (λ i, finset.univ.image (λ j, B i j)) ↔ x ∈ finset.range (n * n)) :
  ∃ m : ℕ, m = 2 * n * (n - 1) ∧ ∀ (A B : matrix (fin n) (fin n) ℕ), 
    ∃ m' : ℕ, m' ≤ m ∧ (∀ (i j : fin n), A i j = B i j → A i j = B i j) :=
  by
  sorry

end matrix_swap_transformation_l349_349344


namespace color_of_85th_bead_l349_349244

/-- Definition for the repeating pattern of beads -/
def pattern : List String := ["red", "orange", "yellow", "yellow", "yellow", "green", "blue", "blue"]

/-- Definition for finding the color of the n-th bead -/
def bead_color (n : Nat) : Option String :=
  let index := (n - 1) % pattern.length
  pattern.get? index

theorem color_of_85th_bead : bead_color 85 = some "yellow" := by
  sorry

end color_of_85th_bead_l349_349244


namespace binomial_10_2_l349_349692

noncomputable def binom (n k : ℕ) : ℕ := n! / (k! * (n - k)!)

theorem binomial_10_2 : binom 10 2 = 45 := by
  sorry

end binomial_10_2_l349_349692


namespace rebus_decrypt_correct_l349_349288

-- Definitions
def is_digit (d : ℕ) : Prop := 0 ≤ d ∧ d ≤ 9
def is_odd (d : ℕ) : Prop := is_digit d ∧ d % 2 = 1
def is_even (d : ℕ) : Prop := is_digit d ∧ d % 2 = 0

-- Variables representing ċharacters H, Ч (C), A, D, Y, E, F, B, K
variables (H C A D Y E F B K : ℕ)

-- Conditions
axiom H_odd : is_odd H
axiom C_even : is_even C
axiom A_even : is_even A
axiom D_odd : is_odd D
axiom Y_even : is_even Y
axiom E_even : is_even E
axiom F_odd : is_odd F
axiom B_digit : is_digit B
axiom K_odd : is_odd K

-- Correct answers
def H_val : ℕ := 5
def C_val : ℕ := 3
def A_val : ℕ := 2
def D_val : ℕ := 9
def Y_val : ℕ := 8
def E_val : ℕ := 8
def F_val : ℕ := 5
def B_any : ℕ := B
def K_val : ℕ := 5

-- Proof statement
theorem rebus_decrypt_correct : 
  H = H_val ∧
  C = C_val ∧
  A = A_val ∧
  D = D_val ∧
  Y = Y_val ∧
  E = E_val ∧
  F = F_val ∧
  K = K_val :=
sorry

end rebus_decrypt_correct_l349_349288


namespace find_annual_interest_rate_l349_349557

def face_value : ℝ := 2240
def true_discount : ℝ := 240
def period : ℝ := 0.75
def present_value (FV TD : ℝ) : ℝ := FV - TD
def annual_interest_rate (PV TD T : ℝ) : ℝ := (TD * 100) / (PV * T)

theorem find_annual_interest_rate :
  annual_interest_rate (present_value face_value true_discount) true_discount period = 16 :=
by
  -- Calculation that shows the annual interest rate is 16.
  -- We are not providing the proof; inserting sorry.
  sorry

end find_annual_interest_rate_l349_349557


namespace rhombus_angle_EFGH_l349_349466

-- Define the basic properties and definitions for the problem
variables {α : Type*} [add_group α] [module ℝ α]

-- Define what it means for a quadrilateral to be a rhombus
structure rhombus (E F G H : α) : Prop :=
  (equal_sides : ∀ E F G H, E = F ∧ F = G ∧ G = H)
  (opposite_angles : ∀ α β, α = β)

-- Given a rhombus EFGH with ∠F = 130°, we need to prove that ∠H = 130°
theorem rhombus_angle_EFGH (E F G H : α) (h : rhombus E F G H) 
  (angle_F : ∠ F = 130) : ∠ H = 130 :=
sorry

end rhombus_angle_EFGH_l349_349466


namespace square_perimeter_not_necessarily_integer_l349_349642

theorem square_perimeter_not_necessarily_integer (s : ℝ) :
  (∃ (rectangles : list (ℝ × ℝ)), (∀ r ∈ rectangles, ∃ p : ℕ, 2 * (r.1 + r.2) = p) ∧ (4 * s) ∉ ℕ) :=
by
  sorry

end square_perimeter_not_necessarily_integer_l349_349642


namespace number_of_primes_between_30_and_50_l349_349434

-- Define what it means to be a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the interval condition
def in_interval (n : ℕ) : Prop :=
  30 ≤ n ∧ n ≤ 50

-- Define the count of prime numbers in the interval
def prime_count_in_interval : ℕ :=
  (List.range' 30 21).countp (λ n, is_prime n)

-- We state that the above count is equal to 5
theorem number_of_primes_between_30_and_50 : prime_count_in_interval = 5 :=
  sorry

end number_of_primes_between_30_and_50_l349_349434


namespace max_area_curved_shape_l349_349660

-- Definitions
def length_seg (x y : ℝ) := x + y = 1
def radius_semi (x : ℝ) := r = x / π
def height_rect (x r : ℝ) := h = (1 - x - 2 * r) / 2
def area_semi (r : ℝ) := (π * r^2) / 2
def area_rect (r h : ℝ) := 2 * r * h

-- Theorem Statement
theorem max_area_curved_shape
  (x y r h : ℝ)
  (hxysum : length_seg x y)
  (hradius : radius_semi x)
  (hheight : height_rect x r)
  (semi_area : ∀ r, area_semi r)
  (rect_area : ∀ r h, area_rect r h) :
  ∃ S : ℝ, S = 1 / (2 * (π + 4)) :=
by
  sorry

end max_area_curved_shape_l349_349660


namespace part_one_part_two_l349_349811

variables (a b : ℝ → ℝ → ℝ) [inner_product_space ℝ ℝ]
noncomputable def lambda_value (λ : ℝ) : Prop :=
  let a := 1 in 
  let b := 2 in
  let angle_ab := (real.pi / 3) in
  let dot_prod := a * b * real.cos(angle_ab) in
  (real.sqrt (a * a) - λ * real.sqrt (b * b) + (λ - 1) * dot_prod) = 0

theorem part_one (λ : ℝ) (cond : lambda_value λ) : λ = 0 := 
sorry

variables (a b : ℝ → ℝ → ℝ) [inner_product_space ℝ ℝ]

theorem part_two 
  (H1: ∥a∥ = 1) 
  (H2: ∥b∥ = 2)
  (H3: angle a b = real.pi / 3) :
  ∠a (2 • a - b) = real.pi / 3 :=
sorry

end part_one_part_two_l349_349811


namespace largest_possible_last_digit_l349_349626

/-!
# Problem Statement
- A digit string consists of 2023 digits.
- The first digit of this string is 1.
- Any two-digit number formed by consecutive digits within this string is divisible by either 19, 29, or 31.
- Prove that the largest possible last digit of this string is 8.
-/

def is_digit (d : ℕ) : Prop := d < 10

def valid_two_digit_num (n : ℕ) : Prop :=
  (n % 19 = 0 ∨ n % 29 = 0 ∨ n % 31 = 0) ∧ n / 10 < 10

def valid_string (s : list ℕ) : Prop :=
  s.length = 2023 ∧
  list.head s = some 1 ∧
  (∀ (i : ℕ), i < s.length - 1 → valid_two_digit_num (10 * (s.nth_le i sorry) + (s.nth_le (i + 1) sorry))) ∧
  is_digit (list.head s) 

theorem largest_possible_last_digit : ∃ (s : list ℕ), valid_string s ∧ s.ilast sorry = 8 := sorry

end largest_possible_last_digit_l349_349626


namespace alice_operations_terminate_l349_349325

theorem alice_operations_terminate (a : List ℕ) (h_pos : ∀ x ∈ a, x > 0) : 
(∀ x y z, (x, y) = (y + 1, x) ∨ (x, y) = (x - 1, x) → ∃ n, (x :: y :: z).sum ≤ n) :=
by sorry

end alice_operations_terminate_l349_349325


namespace balance_squares_circles_l349_349937

theorem balance_squares_circles (x y z : ℕ) (h1 : 5 * x + 2 * y = 21 * z) (h2 : 2 * x = y + 3 * z) : 
  3 * y = 9 * z :=
by 
  sorry

end balance_squares_circles_l349_349937


namespace grass_cut_cost_each_time_l349_349477

def grass_initial_height := 2 -- initial height of the grass in inches
def grass_final_height := 4 -- height at which the grass is cut back in inches
def grass_growth_rate := 0.5 -- grass growth rate in inches per month
def yearly_cost := 300 -- total yearly cost in dollars

noncomputable def cost_per_cut : ℕ → ℝ
| 0 := yearly_cost
| n := yearly_cost / n

theorem grass_cut_cost_each_time :
  let months_to_grow := grass_final_height - grass_initial_height / grass_growth_rate
  let cuts_per_year := 12 / months_to_grow
  300 / cuts_per_year = 100 :=
by
  let months_to_grow := grass_final_height - grass_initial_height / grass_growth_rate
  let cuts_per_year := 12 / months_to_grow
  have h1 : months_to_grow = (4 - 2) / 0.5
  { sorry }
  have h2 : cuts_per_year = 12 / 4
  { sorry }
  have h3 : 300 / 3 = 100
  { sorry }
  exact h3

end grass_cut_cost_each_time_l349_349477


namespace find_x_l349_349585

theorem find_x : ∃ x : ℕ, x + 5 * 12 / (180 / 3) = 66 :=
by
  let x := 65
  use x
  have h : x + 1 = 66 := by sorry
  exact h

end find_x_l349_349585


namespace real_solutions_count_l349_349438

theorem real_solutions_count :
  ∀ f g : ℝ → ℝ, (∀ x, f x = (1 / 3) * x^4) → (∀ x, g x = 5 * |x|) →
  (number_of_solutions (λ x, f x + g x) 7) = 2 :=
by
  sorry

end real_solutions_count_l349_349438


namespace teams_face_each_other_l349_349847

theorem teams_face_each_other (teams games : ℕ) (h_teams : teams = 14) (h_games : games = 455) :
  (games = teams * (teams - 1) / 2 * 5) :=
by
  rw [h_teams, h_games]
  sorry

end teams_face_each_other_l349_349847


namespace length_of_DE_l349_349874

theorem length_of_DE 
  (area_ABC : ℝ) 
  (area_trapezoid : ℝ) 
  (altitude_ABC : ℝ) 
  (h1 : area_ABC = 144) 
  (h2 : area_trapezoid = 96)
  (h3 : altitude_ABC = 24) :
  ∃ (DE_length : ℝ), DE_length = 2 * Real.sqrt 3 := 
sorry

end length_of_DE_l349_349874


namespace necessary_and_sufficient_condition_l349_349570

variable {A B : Prop}

theorem necessary_and_sufficient_condition (h1 : A → B) (h2 : B → A) : A ↔ B := 
by 
  sorry

end necessary_and_sufficient_condition_l349_349570


namespace quadratic_polynomial_P_l349_349117

noncomputable def P : Polynomial := {
  to_fun := λ x : ℝ, x^2 - x + 2,
  degree := 2
}

theorem quadratic_polynomial_P (P : ℝ → ℝ) 
  (h : ∀ x, P (P x) = x^4 - 2 * x^3 + 4 * x^2 - 3 * x + 4) : 
  P 8 = 58 := 
by
  sorry

end quadratic_polynomial_P_l349_349117


namespace slope_length_l349_349970

theorem slope_length (BC : ℝ) (slope_ratio : ℝ) (hBC : BC = 30) (h_slope : slope_ratio = 2) : 
  ∃ AB : ℝ, AB = 30 * real.sqrt 5 :=
by
  let AC := slope_ratio * BC
  have h_AC : AC = 60 := sorry
  let AB_sq := AC^2 + BC^2
  have h_AB_sq : AB_sq = 4500 := sorry
  let AB := real.sqrt AB_sq
  exact ⟨AB, sorry⟩

end slope_length_l349_349970


namespace airplane_altitude_l349_349653

theorem airplane_altitude (h : ℝ) (AB AC BC : ℝ) (angle_Alice angle_Bob : ℝ) 
  (h_45_Alice : angle_Alice = 45) 
  (h_45_Bob : angle_Bob = 45) 
  (h_AB : AB = 15) 
  (h_AC_BC : AC = BC) 
  (h_tan_45 : ∀ x, tan x = 1 ↔ x = 45) : 
  h = 15 :=
by 
  sorry

end airplane_altitude_l349_349653


namespace share_ratio_l349_349230

theorem share_ratio (A B C x : ℝ)
  (h1 : A = 280)
  (h2 : A + B + C = 700)
  (h3 : A = x * (B + C))
  (h4 : B = (6 / 9) * (A + C)) :
  A / (B + C) = 2 / 3 :=
by
  sorry

end share_ratio_l349_349230


namespace difference_in_square_side_lengths_l349_349606

theorem difference_in_square_side_lengths (h : ℝ) (r : ℝ) (α β : ℝ) 
  (h_condition : β = sqrt (1 / 5)) 
  (sin_condition : sin α = sqrt (4 / 5)) 
  (cos_condition : cos α = sqrt (1 / 5)) 
  : 
  2 * h * cos β * sin α = 8 * h / 5 :=
by 
  sorry

end difference_in_square_side_lengths_l349_349606


namespace find_a_b_z_l349_349900

noncomputable def a_b_imaginary_values (a b : ℝ) : Prop :=
a + b * (complex.I) = ((1 - complex.I) ^ 2 + 2 * (5 + complex.I)) / (3 + complex.I)

noncomputable def z_imaginary_bisector (z : ℂ) (a b : ℝ) : Prop :=
let w := a + b * complex.I in 
let complex_number := w * z in
complex_number.re = complex_number.im

theorem find_a_b_z :
  (∃ a b : ℝ, a_b_imaginary_values a b ∧ a = 3 ∧ b = -1) ∧ 
  (∃ y : ℝ, let z := -1 + y * (complex.I) in z_imaginary_bisector z 3 (-1) ∧ z = -1 - 2 * complex.I) :=
by {
  have h_a_b : ∃ a b : ℝ, a_b_imaginary_values a b, 
  { use [3, -1], 
    sorry 
  }, 
  have h_z : ∃ y : ℝ, let z := -1 + y * (complex.I) in z_imaginary_bisector z 3 (-1),
  { use -2,
    sorry 
  }, 
  split,
  { use [3, -1], 
    split; 
    assumption 
  },
  { use -2, 
    split;
    assumption 
  }
}

end find_a_b_z_l349_349900


namespace volume_ratio_of_solids_l349_349628

-- Define the surface area of a cube and a regular octahedron
def cube_surface_area (a : ℝ) : ℝ := 6 * a^2
def octahedron_surface_area (b : ℝ) : ℝ := 2 * real.sqrt(3) * b^2

-- Define the volume of a cube and a regular octahedron
def cube_volume (a : ℝ) : ℝ := a^3
def octahedron_volume (b : ℝ) : ℝ := (real.sqrt(2) / 3) * b^3

-- Define the conditions
variable (a b : ℝ)
variable h1 : cube_surface_area a = octahedron_surface_area b

-- The proof problem
theorem volume_ratio_of_solids
  (h1 : cube_surface_area a = octahedron_surface_area b)
  (h2 : b = a * real.sqrt(3)^(1 / 4)) :
  cube_volume a / octahedron_volume b = 3 / real.sqrt(6 * real.sqrt(3)) :=
sorry

end volume_ratio_of_solids_l349_349628


namespace percent_value_in_quarters_l349_349596

theorem percent_value_in_quarters (num_dimes num_quarters : ℕ) 
  (value_dime value_quarter total_value value_in_quarters : ℕ) 
  (h1 : num_dimes = 75)
  (h2 : num_quarters = 30)
  (h3 : value_dime = num_dimes * 10)
  (h4 : value_quarter = num_quarters * 25)
  (h5 : total_value = value_dime + value_quarter)
  (h6 : value_in_quarters = num_quarters * 25) :
  (value_in_quarters / total_value) * 100 = 50 :=
by
  sorry

end percent_value_in_quarters_l349_349596


namespace jolene_babysitting_families_l349_349051

theorem jolene_babysitting_families (F : ℕ) 
  (h1 : Jolene earns $30 for each family) 
  (h2 : Jolene washes 5 neighbors' cars for $12 each = $60) 
  (h3 : Jolene raised $180 total):
  30 * F + 60 = 180 → F = 4 :=
by {
  sorry
}

end jolene_babysitting_families_l349_349051


namespace henry_wins_l349_349001

-- Definitions of conditions
def total_games : ℕ := 14
def losses : ℕ := 2
def draws : ℕ := 10

-- Statement of the theorem
theorem henry_wins : (total_games - losses - draws) = 2 :=
by
  -- Proof goes here
  sorry

end henry_wins_l349_349001


namespace ordered_pairs_subsets_l349_349823

theorem ordered_pairs_subsets (n : ℕ) : 
  ∃ (A B : finset (fin n)), A ⊆ B → (∃! count : ℕ, count = 3^n) :=
by
  sorry

end ordered_pairs_subsets_l349_349823


namespace right_triangle_partition_l349_349893

-- Definitions for the vertices and points on the equilateral triangle
variables {A B C D E F : Point}

-- Conditions for the positions of points D, E, and F
axiom DC_2BD : dist C D = 2 * dist B D
axiom AE_2EC : dist A E = 2 * dist E C
axiom BF_2FA : dist B F = 2 * dist F A

-- Definition of sets S, S1, and S2
def S : set Point := { p | on_triangle_side p A B C }  -- Definition of S as points on sides of triangle
def S1 S2 : set Point := sorry  -- Assume arbitrary partition

theorem right_triangle_partition :
  ∃ T : set Point, (T = S1 ∨ T = S2) ∧ contains_right_triangle T :=
  sorry

-- Helper definitions
def on_triangle_side (p : Point) (A B C : Point) : Prop := 
  -- p is on the sides of the triangle ABC
  (on_line p A B ∨ on_line p B C ∨ on_line p C A)

def contains_right_triangle (T : set Point) : Prop := 
  ∃ (X Y Z ∈ T), is_right_triangle X Y Z


end right_triangle_partition_l349_349893


namespace check_line_properties_l349_349794

-- Define the conditions
def line_equation (x y : ℝ) : Prop := y + 7 = -x - 3

-- Define the point and slope
def point_and_slope (x y : ℝ) (m : ℝ) : Prop := (x, y) = (-3, -7) ∧ m = -1

-- State the theorem to prove
theorem check_line_properties :
  ∃ x y m, line_equation x y ∧ point_and_slope x y m :=
sorry

end check_line_properties_l349_349794


namespace positive_difference_l349_349581

noncomputable def calculate_diff : ℕ :=
  let first_term := (8^2 - 8^2) / 8
  let second_term := (8^2 * 8^2) / 8
  second_term - first_term

theorem positive_difference : calculate_diff = 512 := by
  sorry

end positive_difference_l349_349581


namespace BD_over_CD_eq_expr_l349_349919

-- Define the points and lines
variable (A B C P P1 Q Q1 D : Point)
variable (PQ P1Q1 BC : Line)

-- Conditions
axiom P_on_AB : OnLine P (Line.mk A B)
axiom P1_on_AB : OnLine P1 (Line.mk A B)
axiom Q_on_AC : OnLine Q (Line.mk A C)
axiom Q1_on_AC : OnLine Q1 (Line.mk A C)

-- Intersection conditions
axiom I : Point
axiom PQ_intersect_P1Q1 : Intersect PQ P1Q1 = I
axiom A_to_I_line_D : IntersectsAt (Line.mk A I) BC D

-- Required result
theorem BD_over_CD_eq_expr :
  (BD / CD) = ((BP / PA) - (BP1 / P1A)) / ((CQ / QA) - (CQ1 / Q1A)) :=
sorry

end BD_over_CD_eq_expr_l349_349919


namespace hyperbola_eccentricity_l349_349739

noncomputable def eccentricity (a b c e : ℝ) : Prop :=
  (∀ x y, x ± 2 * y = 0 → ∃ b, b = a / 2) ∧
  (b^2 = c^2 - a^2) ∧
  (e^2 = c^2 / a^2) ∧
  (e = Real.sqrt 5 / 2)

theorem hyperbola_eccentricity (a b c e : ℝ) 
  (h1 : (∀ x y, x ± 2 * y = 0 → ∃ b, b = a / 2)) 
  (h2 : b^2 = c^2 - a^2) 
  (h3 : e^2 = c^2 / a^2) : 
  e = Real.sqrt 5 / 2 :=
by
  sorry

end hyperbola_eccentricity_l349_349739


namespace regular_polygon_sides_l349_349024

theorem regular_polygon_sides (n : ℕ) (h : (n - 2) * 180 = 144 * n) : n = 10 := 
by 
  sorry

end regular_polygon_sides_l349_349024


namespace cc_l349_349484

variables {A B C A1 B1 A0 B0 C' : Type*}
variables [point A] [point B] [point C] [point A1] [point B1] [point A0] [point B0] [point C']
variables (ABC : triangle A B C)

-- Define the conditions
def is_altitude (p : point) (a : point) (b : point) (alt : point) : Prop :=
  line_through p alt ∧ perpendicular (line_through p alt) (line_through a b)

def is_midpoint (m : point) (a : point) (b : point) : Prop :=
  betweenness a m b ∧ distance a m = distance m b

def intersects (l1 : line) (l2 : line) (p : point) : Prop :=
  lies_on p l1 ∧ lies_on p l2

variable (h1 : is_altitude A A B A1)
variable (h2 : is_altitude B B C B1)
variable (h3 : is_midpoint A0 B C)
variable (h4 : is_midpoint B0 C A)
variable (h5 : intersects (line_through A1 B1) (line_through A0 B0) C')

-- Define the Euler line
def euler_line (t : triangle) : line :=
  let h := orthocenter t in
  let o := circumcenter t in
  line_through h o

-- Proof Statement
theorem cc'_perpendicular_to_euler (t : triangle A B C) :
  perpendicular (line_through C C') (euler_line t) :=
sorry

end cc_l349_349484


namespace min_value_of_expr_l349_349354

theorem min_value_of_expr (a b c : ℝ) (h1 : 0 < a ∧ a ≤ b ∧ b ≤ c) (h2 : a * b * c = 1) :
    (1 / (a ^ 3 * (b + c)) + 1 / (b ^ 3 * (a + c)) + 1 / (c ^ 3 * (a + b))) ≥ 3 / 2 := 
by
  sorry

end min_value_of_expr_l349_349354


namespace min_positive_period_of_f_l349_349155

def f (x : ℝ) : ℝ := Real.sin (x + Real.pi / 3) * Real.sin (x + Real.pi / 2)

theorem min_positive_period_of_f : ∀ x: ℝ, f(x + Real.pi) = f x := by
  sorry

end min_positive_period_of_f_l349_349155


namespace quadratic_polynomial_P8_l349_349132

theorem quadratic_polynomial_P8 :
  ∃ (a b c : ℝ), 
  (∀ x : ℝ, P x = a * x^2 + b * x + c) ∧ 
  (∀ x : ℝ, P (P x) = x^4 - 2 * x^3 + 4 * x^2 - 3 * x + 4) ∧ 
  P 8 = 58 :=
begin
  sorry
end

end quadratic_polynomial_P8_l349_349132


namespace benjamin_speed_l349_349834

-- Define the problem conditions
def distance : ℕ := 800 -- Distance in kilometers
def time : ℕ := 10 -- Time in hours

-- Define the main statement
theorem benjamin_speed : distance / time = 80 := by
  sorry

end benjamin_speed_l349_349834


namespace length_CM_correct_l349_349138

-- Definitions based on conditions
def side_length : ℝ := 4
def square_area : ℝ := side_length * side_length
def part_area : ℝ := square_area / 4

-- We are given segments CM and CN that divide the square into four equal parts
-- Define the point M to be at the midpoint of side AB (in a coordinate system centered at A)
def M : ℝ × ℝ := (2, 0)
-- Define the point C to be at (4, 0) initially and then translated
def C : ℝ × ℝ := (4, 4)

-- Define the hypotenuse of the right triangle CBM
def length_CM : ℝ := Real.sqrt ((C.1 - 0)^2 + (C.2 - M.2)^2)
-- Prove that the length of CM is 2√5
theorem length_CM_correct : length_CM = 2 * Real.sqrt 5 :=
by
  -- Lean proof would go here
  sorry

end length_CM_correct_l349_349138


namespace count_primes_between_30_and_50_l349_349408

-- Define the range of numbers from 30 to 50
def range_30_to_50 := Set.of_list (List.range' 30 (51 - 30))

-- Define a predicate to check if a number is prime
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

-- Extract all prime numbers in the specified range
def primes_between_30_and_50 : List ℕ :=
  List.filter is_prime (Set.toList range_30_to_50)

theorem count_primes_between_30_and_50 : primes_between_30_and_50.length = 5 :=
by
  -- The proof goes here
  sorry

end count_primes_between_30_and_50_l349_349408


namespace find_common_ratio_l349_349792

variable {a_n : ℕ → ℕ} -- an arithmetic geometric sequence
variable {S_n : ℕ → ℕ} -- sum of the first n terms of the sequence
variable {q : ℕ} -- common ratio

-- Given conditions
axiom H1 : S_n 2 = 3
axiom H2 : S_n 4 = 15

theorem find_common_ratio (H : ∀ n, S_n (2*n) - S_n n = q^n * S_n n) :
  q = 2 :=
by
  have Hq2 : q^2 = (S_n 4 - S_n 2) / S_n 2,
  { calc
      q^2 = (S_n 4 - S_n 2) / S_n 2 : by sorry
           ... = 4 : by sorry },
  sorry

end find_common_ratio_l349_349792


namespace min_distance_from_P_to_origin_l349_349352

noncomputable def distance_to_origin : ℝ := 8 / 5

theorem min_distance_from_P_to_origin
  (P : ℝ × ℝ)
  (hA : P.1^2 + P.2^2 = 1)
  (hB : (P.1 - 3)^2 + (P.2 + 4)^2 = 10)
  (h_tangent : PE = PD) :
  dist P (0, 0) = distance_to_origin := 
sorry

end min_distance_from_P_to_origin_l349_349352


namespace number_of_ways_to_place_pawns_l349_349007

noncomputable def numberOfWaysToPlacePawns : ℕ :=
  5! * 5!

theorem number_of_ways_to_place_pawns : numberOfWaysToPlacePawns = 14400 :=
sorry

end number_of_ways_to_place_pawns_l349_349007


namespace close_connection_is_4_over_9_l349_349048

noncomputable def close_connection_probability : ℚ :=
  let possible_values := {1, 2, 3, 4, 5, 6}
  let pairs := (possible_values ×ˢ possible_values).to_finset
  let favorable_pairs := pairs.filter (λ p, (p.1 - p.2).natAbs ≤ 1)
  let total_possible_outcomes := (possible_values.to_finset.card) * (possible_values.to_finset.card)
  let total_favorable_outcomes := favorable_pairs.card
  (total_favorable_outcomes : ℚ) / (total_possible_outcomes : ℚ)

theorem close_connection_is_4_over_9 :
  close_connection_probability = 4 / 9 := sorry

end close_connection_is_4_over_9_l349_349048


namespace possible_increasing_geometric_progressions_l349_349607

variable (a p q : ℤ)
variable (b : ℤ := a * p * q)

def increasing_gp_sum_57 (a p q : ℤ) : Prop :=
  a * (p^2 + p * q + q^2) = 57 ∧ (∀ x y, x ≠ y → (x, y) ∈ {(a * q^2, a * p * q), (a * p * q, a * p^2)})

theorem possible_increasing_geometric_progressions :
  ∃ (a p q : ℤ), increasing_gp_sum_57 a p q ∧
    (a * q^2, a * p * q, a * p^2) = (1, 7, 49) ∨
    (a * q^2, a * p * q, a * p^2) = (12, 18, 27) :=
sorry

end possible_increasing_geometric_progressions_l349_349607


namespace count_primes_between_30_and_50_l349_349413

-- Define the range of numbers from 30 to 50
def range_30_to_50 := Set.of_list (List.range' 30 (51 - 30))

-- Define a predicate to check if a number is prime
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

-- Extract all prime numbers in the specified range
def primes_between_30_and_50 : List ℕ :=
  List.filter is_prime (Set.toList range_30_to_50)

theorem count_primes_between_30_and_50 : primes_between_30_and_50.length = 5 :=
by
  -- The proof goes here
  sorry

end count_primes_between_30_and_50_l349_349413


namespace binom_10_2_eq_45_l349_349703

-- Definitions used in the conditions
def binom (n k : ℕ) := n.choose k

-- The statement that needs to be proven
theorem binom_10_2_eq_45 : binom 10 2 = 45 :=
by
  sorry

end binom_10_2_eq_45_l349_349703


namespace area_of_triangle_DEF_l349_349030

-- Define the variables and conditions
variables (DE : ℝ) (height : ℝ)
def area_of_triangle (base height : ℝ) : ℝ := (1/2) * base * height

-- Given conditions
axiom DE_is_12 : DE = 12
axiom height_is_7 : height = 7

-- Prove the area of triangle DEF
theorem area_of_triangle_DEF : area_of_triangle DE height = 42 :=
by {
  rw [DE_is_12, height_is_7],
  -- The expected result
  sorry
}

end area_of_triangle_DEF_l349_349030


namespace perpendicular_bisector_theorem_l349_349362

noncomputable def is_inscribed (O : Type) (A B C D E F P : O) : Prop :=
  ∃ (circle : Set O), circle.is_circle ∧
    A ∈ circle ∧ B ∈ circle ∧ C ∈ circle ∧ D ∈ circle ∧ E ∈ circle ∧ F ∈ circle ∧ P ∈ circle

noncomputable def intersects (A B O : Type) : Prop :=
  ∃ X ∈ (A ∩ B), X ∈ O

noncomputable def extension_intersects (C F O P : Type) : Prop :=
  ∃ (circle : Set O), circle.is_circle ∧
    P ∈ (line_through C F) ∩ circle

noncomputable def diagonals_intersect (A D B E F : Type) : Prop :=
  ∃ (F : Type), F ∈ (line_through A D) ∧ F ∈ (line_through B E)

noncomputable def perpendicular (O A E P : Type) : Prop :=
  ∃ (circle : Set O), circle.is_circle ∧ (O P ⊥ A E)

theorem perpendicular_bisector_theorem (O A B C D E F P : Type) 
  (h1: is_inscribed O A B C D E F P)
  (h2: diagonals_intersect A D B E F)
  (h3: extension_intersects C F O P)
  (h4: AB * CD = BC * ED) : perpendicular O A E P :=
begin
  sorry,
end

end perpendicular_bisector_theorem_l349_349362


namespace largest_int_less_than_100_rem_5_by_7_l349_349317

theorem largest_int_less_than_100_rem_5_by_7 :
  ∃ k : ℤ, (7 * k + 5 = 96) ∧ ∀ n : ℤ, (7 * n + 5 < 100) → (n ≤ k) :=
sorry

end largest_int_less_than_100_rem_5_by_7_l349_349317


namespace vertex_of_parabola_l349_349292

-- Define the parabolic function
def parabola (x : ℝ) : ℝ := 2 * x^2 + 16 * x + 50

-- Define the vertex coordinates we need to prove
def vertex_x : ℝ := -4
def vertex_y : ℝ := 18

-- Prove that the vertex of the parabola is at (-4, 18)
theorem vertex_of_parabola : ∀ x : ℝ, 
  (vertex_x, vertex_y) = (-4, 2 * ((x + 4)^2 - 4^2) + 50) := 
by 
  sorry

end vertex_of_parabola_l349_349292


namespace min_inverse_sum_l349_349357
open Classical

theorem min_inverse_sum (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_sum : a + b + c = 1) :
  (∑ x in [a, b, c], 1/x) ≥ 9 :=
by
  -- Placeholder for the actual proof
  sorry

end min_inverse_sum_l349_349357


namespace dedalo_traps_minotaur_l349_349719

theorem dedalo_traps_minotaur (c : ℕ) (h : c = 75) : 
  ∃ (S : list (list bool)), 
    (∀ (s : list bool), s ∈ S → is_binary_string s) ∧ 
    (∑ s in S, (1 / (2 : ℝ) ^ (s.length)) ≤ (c / 100)) ∧ 
    ∀ (inf_seq : ℕ → bool), ∃ s ∈ S, is_sublist s inf_seq :=
sorry

end dedalo_traps_minotaur_l349_349719


namespace sally_pens_miscalculation_l349_349526

theorem sally_pens_miscalculation :
  let pens_taken := 8135
  let pens_first_class := 54 * 95
  let pens_second_class := 77 * 62
  let total_pens_given := pens_first_class + pens_second_class
  total_pens_given > pens_taken :=
by
  let pens_taken := 8135
  let pens_first_class := 54 * 95
  let pens_second_class := 77 * 62
  let total_pens_given := pens_first_class + pens_second_class
  have h1 : pens_first_class = 5130 := by norm_num
  have h2 : pens_second_class = 4774 := by norm_num
  have h3 : total_pens_given = pens_first_class + pens_second_class := rfl
  have h4 : total_pens_given = 5130 + 4774 := by rw [h1, h2]
  have h5 : total_pens_given = 9904 := by norm_num
  have h6 : 9904 > 8135 := by norm_num
  exact h6

end sally_pens_miscalculation_l349_349526


namespace quadratic_polynomial_P_l349_349115

noncomputable def P : Polynomial := {
  to_fun := λ x : ℝ, x^2 - x + 2,
  degree := 2
}

theorem quadratic_polynomial_P (P : ℝ → ℝ) 
  (h : ∀ x, P (P x) = x^4 - 2 * x^3 + 4 * x^2 - 3 * x + 4) : 
  P 8 = 58 := 
by
  sorry

end quadratic_polynomial_P_l349_349115


namespace binom_10_2_eq_45_l349_349707

theorem binom_10_2_eq_45 :
  binom 10 2 = 45 := by
  sorry

end binom_10_2_eq_45_l349_349707


namespace total_people_going_to_zoo_l349_349221

def cars : ℝ := 3.0
def people_per_car : ℝ := 63.0

theorem total_people_going_to_zoo : cars * people_per_car = 189.0 :=
by 
  sorry

end total_people_going_to_zoo_l349_349221


namespace tangent_line_at_P_l349_349555

-- Define the parabola and the parallel condition
def parabola (x : ℝ) : ℝ := x^2

def line (x y : ℝ) : Prop := 2 * x - y + 4 = 0

def tangent_parallel_line (x y : ℝ) : Prop := 2 * x - y + 4 = 0

theorem tangent_line_at_P (P : ℝ × ℝ) (tangent_line : ℝ → ℝ) 
  (h1 : P = (1, 1))
  (h2 : tangent_line = λ x, 2 * x - 1) :
  (tangent_line 1 = 1) ∧ (parabola 1 = 1) ∧ line 1 1 := 
sorry

end tangent_line_at_P_l349_349555


namespace sin_double_angle_l349_349337

theorem sin_double_angle (x : ℝ) (h : sin x - cos x = 1/2) : sin (2 * x) = 3 / 4 :=
by
  sorry

end sin_double_angle_l349_349337


namespace problem_proof_l349_349345

open Nat

noncomputable def seq (n : ℕ) : ℕ := (n + 1) * 2^(n - 1)

noncomputable def Sn (n : ℕ) : ℕ := Σ k in range (n + 1), seq k

noncomputable def Pn (n : ℕ) : ℝ := (Sn n) / (2 * seq n)

noncomputable def Tn (n : ℕ) : ℝ := Real.sqrt ((1 - Pn n) / (1 + Pn n))

theorem problem_proof (n : ℕ) (hn : 0 < n) :
    (Pn 1 * Pn 3 * Pn 5 * ... * Pn (2 * n - 1)) < Tn n ∧ Tn n < (Real.sqrt 2) * Real.sin (Tn n) :=
sorry


end problem_proof_l349_349345


namespace pure_imaginary_complex_number_l349_349786

theorem pure_imaginary_complex_number (a : ℝ) 
  (z : ℂ := (a^2 - 1) + (a - 2) * complex.I) 
  (h1 : z.im = (a - 2))
  (h2 : z = (a^2 - 1) + (a - 2) * complex.I)
  (h3 : z.re = 0) : a = 1 ∨ a = -1 :=
by {
  have h4: a^2 - 1 = 0, { sorry },
  have h5: a - 2 ≠ 0, { sorry },
  sorry
}

end pure_imaginary_complex_number_l349_349786


namespace directrix_parabola_l349_349540

theorem directrix_parabola (p : ℝ) (h : 4 * p = 2) : 
  ∃ d : ℝ, d = -p / 2 ∧ d = -1/2 :=
by
  sorry

end directrix_parabola_l349_349540


namespace quadratic_polynomial_value_l349_349104

theorem quadratic_polynomial_value (P : ℝ → ℝ) (hP : ∀ x, P(P(x)) = x^4 - 2*x^3 + 4*x^2 - 3*x + 4) :
  P 8 = 58 :=
sorry

end quadratic_polynomial_value_l349_349104


namespace num_unique_four_digit_numbers_l349_349817

theorem num_unique_four_digit_numbers (d1 d2 d3 d4 : ℕ) (h1 : d1 = 3) (h2 : d2 = 0) (h3 : d3 = 3) (h4 : d4 = 3) :
  {x // (x = [d1, d2, d3, d4].perm.filter (λ l, l.head ≠ 0))}.card = 3 :=
sorry

end num_unique_four_digit_numbers_l349_349817


namespace min_value_of_f_l349_349322

noncomputable def f (x θ : ℝ) : ℝ := cos (x + 2 * θ) + 2 * sin θ * sin (x + θ)

theorem min_value_of_f (θ : ℝ) : ∃ x : ℝ, f x θ = -1 := 
by sorry

end min_value_of_f_l349_349322


namespace xy_yx_eq_zy_yz_eq_xz_zx_l349_349522

theorem xy_yx_eq_zy_yz_eq_xz_zx 
  (x y z : ℝ) 
  (h : x * (y + z - x) / x = y * (z + x - y) / y ∧ y * (z + x - y) / y = z * (x + y - z) / z): 
  x ^ y * y ^ x = z ^ y * y ^ z ∧ z ^ y * y ^ z = x ^ z * z ^ x :=
by
  sorry

end xy_yx_eq_zy_yz_eq_xz_zx_l349_349522


namespace no_term_in_sequence_is_3_alpha_5_beta_l349_349759

theorem no_term_in_sequence_is_3_alpha_5_beta :
  ∀ (v : ℕ → ℕ),
    v 0 = 0 →
    v 1 = 1 →
    (∀ n, 1 ≤ n → v (n + 1) = 8 * v n * v (n - 1)) →
    ∀ n, ∀ (α β : ℕ), α > 0 → β > 0 → v n ≠ 3^α * 5^β := by
  intros v h0 h1 recurrence n α β hα hβ
  sorry

end no_term_in_sequence_is_3_alpha_5_beta_l349_349759


namespace log_subtraction_property_l349_349297

theorem log_subtraction_property : log 2 12 - log 2 3 = 2 := 
sorry

end log_subtraction_property_l349_349297


namespace major_axis_length_of_ellipse_l349_349803

-- Definition of the conditions
def line (x y : ℝ) : Prop := 2 * x - y + 4 = 0
def ellipse (x y : ℝ) (m : ℝ) : Prop := (x^2) / m + (y^2) / 2 = 1
def is_focus (x y m : ℝ) : Prop := line x y ∧ ellipse x y m

theorem major_axis_length_of_ellipse (m : ℝ) (h₀ : m > 0) :
  (∃ (x y : ℝ), is_focus x y m) → 2 * Real.sqrt 6 = 2 * Real.sqrt m :=
sorry

end major_axis_length_of_ellipse_l349_349803


namespace radius_of_shorter_cylinder_l349_349565

theorem radius_of_shorter_cylinder (h r : ℝ) (V_s V_t : ℝ) (π : ℝ) : 
  V_s = 500 → 
  V_t = 500 → 
  V_t = π * 5^2 * 4 * h → 
  V_s = π * r^2 * h → 
  r = 10 :=
by 
  sorry

end radius_of_shorter_cylinder_l349_349565


namespace binomial_10_2_equals_45_l349_349679

open Nat

theorem binomial_10_2_equals_45 : Nat.choose 10 2 = 45 := 
by
  sorry

end binomial_10_2_equals_45_l349_349679


namespace inradius_relation_l349_349887

-- Define the necessary entities and conditions
variables {A B C D : Type} [Real A] [Real B] [Real C] [Real D]

def is_right_triangle (A B C : Type) (angle_B_eq_90 : ∀ {x : A}, x = 90) : Prop := sorry

def on_line_segment (D : Type) (AC : Type) : Prop := sorry

def equal_inradii (r' r : ℝ) : Prop := sorry

def inradius (triangle : Type) (r : ℝ) : Prop := sorry

-- The main theorem to be proved
theorem inradius_relation
  {ABC : Type} {ABD : Type} {CBD : Type} {r r' BD : ℝ}
  (h1 : is_right_triangle ABC (λ x, x = 90))
  (h2 : on_line_segment D (AC))
  (h3 : equal_inradii r' (inradius ABD r) (inradius CBD r))
  (h4 : inradius ABC r):
  1 / r' = 1 / r + 1 / BD := sorry

end inradius_relation_l349_349887


namespace minimum_m_plus_n_l349_349534

theorem minimum_m_plus_n (m n : ℕ) (h1 : 98 * m = n ^ 3) (h2 : 0 < m) (h3 : 0 < n) : m + n = 42 :=
sorry

end minimum_m_plus_n_l349_349534


namespace general_formula_a_n_l349_349755

-- Definition of the sequence and the function
noncomputable def f (x : ℝ) : ℝ := (2 * x) / (2 + x)

-- Definition of the sequence a_n
def a_seq : ℕ+ → ℝ 
| 1 := 1
| (n+1) := f (a_seq n)

-- Statement to prove: a_n = 2 / (n + 1)
theorem general_formula_a_n (n : ℕ+) : a_seq n = 2 / (n + 1) := 
by
  sorry

end general_formula_a_n_l349_349755


namespace race_position_problem_l349_349452

theorem race_position_problem 
  (Cara Bruno Emily David Fiona Alan: ℕ)
  (participants : Finset ℕ)
  (participants_card : participants.card = 12)
  (hCara_Bruno : Cara = Bruno - 3)
  (hEmily_David : Emily = David + 1)
  (hAlan_Bruno : Alan = Bruno + 4)
  (hDavid_Fiona : David = Fiona + 3)
  (hFiona_Cara : Fiona = Cara - 2)
  (hBruno : Bruno = 9)
  (Cara_in_participants : Cara ∈ participants)
  (Bruno_in_participants : Bruno ∈ participants)
  (Emily_in_participants : Emily ∈ participants)
  (David_in_participants : David ∈ participants)
  (Fiona_in_participants : Fiona ∈ participants)
  (Alan_in_participants : Alan ∈ participants)
  : David = 7 := 
sorry

end race_position_problem_l349_349452


namespace count_primes_between_30_and_50_l349_349425

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between_30_and_50 : List ℕ :=
  [31, 37, 41, 43, 47]

theorem count_primes_between_30_and_50 : 
  (primes_between_30_and_50.filter is_prime).length = 5 :=
by
  sorry

end count_primes_between_30_and_50_l349_349425


namespace quadratic_polynomial_value_l349_349099

theorem quadratic_polynomial_value (P : ℝ → ℝ) (hP : ∀ x, P(P(x)) = x^4 - 2*x^3 + 4*x^2 - 3*x + 4) :
  P 8 = 58 :=
sorry

end quadratic_polynomial_value_l349_349099


namespace isosceles_triangle_base_l349_349464

theorem isosceles_triangle_base (a b r : ℝ)
  (ht : isosceles_triangle a b)
  (h_perpendiculars : kites_divided_by_perpendiculars a b r)
  (h_equal_areas : ∑₁ A1 = A2 * 2)
  : b = (2/3) * a := 
sorry

# Definitions needed for the theorem
def isosceles_triangle (a b : ℝ) : Prop := ∀ P Q R (h : PQR.is_isosceles a b), true

def kites_divided_by_perpendiculars (a b r : ℝ) : Prop := 
  ∀ I (hI : ∀ k, I.is_intersection_of_angle_bisectors a b r), 
    (side_areas k = ∑ k : kite, kite.area)

def kites (a b r : ℝ) := (to_kites.rectangle_of_isosceles a b r).area

end isosceles_triangle_base_l349_349464


namespace B_value_M_range_l349_349021

namespace TriangleProblem

-- Condition (1): In triangle ABC, given a*cos(B) + b*cos(A) = 2c*cos(B)
variables (A B C a b c : ℝ)
variable (h₁ : a * Real.cos B + b * Real.cos A = 2 * c * Real.cos B)

-- Question (1): Prove that B = π / 3
theorem B_value : B = Real.pi / 3 := sorry

-- Question (2): Prove the range of M = sin(A) * (sqrt(3) * cos(A) - sin(A)) is (-3/2, 1/2]
def M (A : ℝ) : ℝ :=
  Real.sin A * (Real.sqrt 3 * Real.cos A - Real.sin A)

theorem M_range : ∀ (A : ℝ), 0 < A ∧ A < 2 * Real.pi / 3 → -3 / 2 < M A ∧ M A ≤ 1 / 2 := sorry

end TriangleProblem

end B_value_M_range_l349_349021


namespace intersection_is_correct_l349_349383

def A : set ℝ := {x | x^2 - x + 1 ≥ 0}
def B : set ℝ := {x | x^2 - 5x + 4 ≥ 0}
def intersection_A_B : set ℝ := {x | x ≤ 1 ∨ x ≥ 4}

theorem intersection_is_correct : A ∩ B = intersection_A_B := by
  sorry

end intersection_is_correct_l349_349383


namespace polar_to_cartesian_l349_349287

theorem polar_to_cartesian (r θ : ℝ) (h_r : r = 2) (h_θ : θ = π / 6) :
  (r * Real.cos θ, r * Real.sin θ) = (Real.sqrt 3, 1) :=
by
  rw [h_r, h_θ]
  have h_cos : Real.cos (π / 6) = Real.sqrt 3 / 2 := sorry -- This identity can be used from trigonometric property.
  have h_sin : Real.sin (π / 6) = 1 / 2 := sorry -- This identity can be used from trigonometric property.
  rw [h_cos, h_sin]
  -- some algebraic steps to simplifiy left sides to (Real.sqrt 3, 1) should follow here. using multiplication and commmutaivity properties mainly.
  sorry

end polar_to_cartesian_l349_349287


namespace distance_AB_l349_349149

-- Definition of points in 3D space
structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

-- Distance formula in 3D space
def dist (A B : Point3D) : ℝ :=
  real.sqrt ((B.x - A.x)^2 + (B.y - A.y)^2 + (B.z - A.z)^2)

-- Points A(1,2,-2) and B(-1,0,-1)
def A : Point3D := { x := 1, y := 2, z := -2 }
def B : Point3D := { x := -1, y := 0, z := -1 }

-- Theorem: the distance between points A and B is 3
theorem distance_AB : dist A B = 3 :=
by 
  sorry

end distance_AB_l349_349149


namespace max_min_S_values_l349_349359

theorem max_min_S_values (x y : ℝ) (S : ℝ) 
  (h₁ : (x-1)^2 + (y+2)^2 = 4) : 
  (S = 3x - y → 5 - 2 * Real.sqrt 10 ≤ S ∧ S ≤ 5 + 2 * Real.sqrt 10) :=
by
  intro h₂
  have h₃ : (|S - 5| ≤ 2 * Real.sqrt 10), from sorry
  have h₄ : (5 - 2 * Real.sqrt 10 ≤ S ∧ S ≤ 5 + 2 * Real.sqrt 10), from sorry
  exact h₄

end max_min_S_values_l349_349359


namespace quadratic_P_value_l349_349124

noncomputable def P (x : ℝ) : ℝ :=
  x^2 - x + 2

theorem quadratic_P_value :
  P (P 8) = 8^4 - 2 * 8^3 + 4 * 8^2 - 3 * 8 + 4 ∧ P 8 = 58 := 
by
  have h1 : P(P(8)) = 8^4 - 2 * 8^3 + 4 * 8^2 - 3 * 8 + 4 := sorry
  have h2 : P(8) = 58 := sorry
  exact ⟨h1, h2⟩  

end quadratic_P_value_l349_349124


namespace interval_monotonic_increase_log_l349_349952

theorem interval_monotonic_increase_log :
  (∀ x : ℝ, x^2 - 4 * x + 3 > 0 → ∃ D : Set ℝ, D = (Iio 1)) → 
  ∀ y = log (1/3) (x^2 - 4 * x + 3), x ∈ D → monotonic_increasing y :=
sorry

end interval_monotonic_increase_log_l349_349952


namespace volume_between_spheres_l349_349170

theorem volume_between_spheres (r_small r_large : ℝ) (h_small : r_small = 4) (h_large : r_large = 8) : 
  (4 / 3) * Real.pi * (r_large ^ 3) - (4 / 3) * Real.pi * (r_small ^ 3) = (1792 / 3) * Real.pi := 
by
  rw [h_small, h_large]
  sorry

end volume_between_spheres_l349_349170


namespace smallest_denominator_fraction_l349_349740

theorem smallest_denominator_fraction 
  (p q : ℕ) (hp : 0 < p) (hq : 0 < q) 
  (h1 : 99 / 100 < p / q) 
  (h2 : p / q < 100 / 101) :
  p = 199 ∧ q = 201 := 
by 
  sorry

end smallest_denominator_fraction_l349_349740


namespace triangle_MNQ_is_equilateral_l349_349850

-- Define the conditions of the problem
variables {A B C M N Q : Type} [Point A] [Point B] [Point C]
  [is_acute_triangle : AcuteTriangle A B C]
  [angle_B_is_60 : ∠B A C = 60]
  [altitude_AM : Altitude A M B C]
  [altitude_CN : Altitude C N B A]
  [midpoint_Q : Midpoint Q A C]

-- The theorem to prove
theorem triangle_MNQ_is_equilateral :
  EquilateralTriangle M N Q :=
sorry

end triangle_MNQ_is_equilateral_l349_349850


namespace zainab_work_hours_l349_349997

theorem zainab_work_hours:
  ∀ (h: ℕ),
  (Zainab earns $2 per hour ∧ 
   Zainab works 3 days a week ∧ 
   Zainab's total earnings after 4 weeks are $96) ↔ 
  (h = 2) :=
by
  sorry

end zainab_work_hours_l349_349997


namespace min_number_of_circles_required_for_2011_gon_l349_349180

noncomputable def min_circles (sides : ℕ) : ℕ :=
  if sides = 2011 then 504 else 0

theorem min_number_of_circles_required_for_2011_gon :
  min_circles 2011 = 504 :=
begin
  -- placeholder for the actual proof
  sorry
end

end min_number_of_circles_required_for_2011_gon_l349_349180


namespace length_of_each_train_l349_349177

-- Definitions and conditions
def train_speed_faster := 46 -- in km/hr
def train_speed_slower := 36 -- in km/hr
def catch_time := 45 -- in seconds
def relative_speed_kmh := train_speed_faster - train_speed_slower  -- in km/hr
def relative_speed_ms := (relative_speed_kmh * 1000) / 3600 -- in m/s
def distance_covered := relative_speed_ms * catch_time -- in meters

-- The question to prove: length of each train
theorem length_of_each_train :
  distance_covered / 2 = 62.5 :=
sorry

end length_of_each_train_l349_349177


namespace count_primes_between_30_and_50_l349_349406

-- Define the range of numbers from 30 to 50
def range_30_to_50 := Set.of_list (List.range' 30 (51 - 30))

-- Define a predicate to check if a number is prime
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

-- Extract all prime numbers in the specified range
def primes_between_30_and_50 : List ℕ :=
  List.filter is_prime (Set.toList range_30_to_50)

theorem count_primes_between_30_and_50 : primes_between_30_and_50.length = 5 :=
by
  -- The proof goes here
  sorry

end count_primes_between_30_and_50_l349_349406


namespace eval_composition_l349_349064

def f (x : ℝ) : ℝ := 5 - 2 * x
def g (x : ℝ) : ℝ := x^3 - 2

theorem eval_composition : f (g 2) = -7 := 
by {
  sorry
}

end eval_composition_l349_349064


namespace cannot_form_blue_loop_with_13_tiles_l349_349034

-- Define the tile and tiling properties for the Tantrix Solitaire problem
structure Tile :=
  (id : Nat)
  (rotations : List Tile) -- A tile can be rotated to any of its rotations

-- Define the problem conditions
def blue_lines_form_loop (tiles : List Tile) : Prop := sorry
def no_holes (tiles : List Tile) : Prop := sorry
def all_tiles (tiles : List Tile) : Prop := tiles.length = 14
def thirteen_tiles (tiles : List Tile) : Prop := tiles.length = 13

-- Main theorem: it is impossible to form a blue loop with 13 tiles without holes
theorem cannot_form_blue_loop_with_13_tiles (tiles : List Tile) :
  all_tiles tiles →
  blue_lines_form_loop tiles →
  no_holes tiles →
  ¬ (thirteen_tiles tiles → blue_lines_form_loop tiles ∧ no_holes tiles) := 
sorry

end cannot_form_blue_loop_with_13_tiles_l349_349034


namespace sum_of_distances_l349_349758

open Complex

theorem sum_of_distances (n : ℕ) (hn : 0 < n) (O : Complex) (A : ℕ → Complex) (M : Complex)
  (hA : ∀ k : ℕ, k < n → ∃ θ, A k = Complex.exp (2 * Real.pi * Complex.I * θ / n))
  (hM : ∃ r : ℝ, 1 < r ∧ M = r) :
  (∑ k in Finset.range n, 1 / Complex.abs (M - A k)) ≥ n / Complex.abs M :=
by
  sorry

end sum_of_distances_l349_349758


namespace prime_exists_solution_l349_349927

theorem prime_exists_solution (p : ℕ) [hp : Fact p.Prime] :
  ∃ n : ℕ, (6 * n^2 + 5 * n + 1) % p = 0 :=
by
  sorry

end prime_exists_solution_l349_349927


namespace product_mnp_l349_349151

theorem product_mnp (a x y z c : ℕ) (m n p : ℕ) :
  (a ^ 8 * x * y * z - a ^ 7 * y * z - a ^ 6 * x * z = a ^ 5 * (c ^ 5 - 1) ∧
   (a ^ m * x * z - a ^ n) * (a ^ p * y * z - a ^ 3) = a ^ 5 * c ^ 5) →
  m = 5 ∧ n = 4 ∧ p = 3 ∧ m * n * p = 60 :=
by
  sorry

end product_mnp_l349_349151


namespace P_P_eq_P_eight_equals_58_l349_349086

open Polynomial

noncomputable def P(x : ℚ) : ℚ := x^2 - x + 2

theorem P_P_eq :
  (P ∘ P)(x) = x^4 - 2 * x^3 + 4 * x^2 - 3 * x + 4 := sorry

theorem P_eight_equals_58 :
  P 8 = 58 := sorry

end P_P_eq_P_eight_equals_58_l349_349086


namespace area_of_R_l349_349746

open Set

/-- Define the region R_alpha(alpha) as a convex pentagon in R^2 with given vertices. -/
def R_alpha (alpha : ℝ) (halpha : 0 < alpha ∧ alpha < 1) : Set (ℝ × ℝ) :=
  { p | p.1 ∈ Icc 0 1 ∧ p.2 ∈ Icc (0 : ℝ) (1 : ℝ) ∧
        (p.1 = 0 → p.2 = 1 - alpha) ∧
        (p.1 = alpha → p.2 = 0) ∧
        (p.1 = 1 → p.2 = 1) }

/-- Define the region R as the intersection of all R_alpha(alpha) for 0 < alpha < 1. -/
def R : Set (ℝ × ℝ) :=
  ⋂ (alpha : ℝ) (halpha : 0 < alpha ∧ alpha < 1), R_alpha alpha halpha

/-- Theorem stating the area of the region R as 2/3. -/
theorem area_of_R : measure_theory.measure.univ (R) = 2 / 3 := sorry

end area_of_R_l349_349746


namespace abba_divisible_by_11_l349_349444

-- Given any two-digit number with digits a and b
def is_divisible_by_11 (a b : ℕ) : Prop :=
  (1001 * a + 110 * b) % 11 = 0

theorem abba_divisible_by_11 (a b : ℕ) (ha : a < 10) (hb : b < 10) : is_divisible_by_11 a b :=
  sorry

end abba_divisible_by_11_l349_349444


namespace count_valid_m_eq_50_l349_349223

noncomputable def discriminant (b c : ℤ) : ℤ := b * b - 4 * c

def is_perfect_square (n : ℤ) : Prop :=
  ∃ k : ℤ, k * k = n

def valid_m (m : ℤ) : Prop :=
  is_perfect_square (discriminant 202200 (2022 * m))

theorem count_valid_m_eq_50 : (finset.filter valid_m (finset.range 51)).card = 50 :=
sorry

end count_valid_m_eq_50_l349_349223


namespace scientific_notation_conversion_l349_349609

theorem scientific_notation_conversion : 450000000 = 4.5 * 10^8 :=
by
  sorry

end scientific_notation_conversion_l349_349609


namespace quadratic_P_value_l349_349125

noncomputable def P (x : ℝ) : ℝ :=
  x^2 - x + 2

theorem quadratic_P_value :
  P (P 8) = 8^4 - 2 * 8^3 + 4 * 8^2 - 3 * 8 + 4 ∧ P 8 = 58 := 
by
  have h1 : P(P(8)) = 8^4 - 2 * 8^3 + 4 * 8^2 - 3 * 8 + 4 := sorry
  have h2 : P(8) = 58 := sorry
  exact ⟨h1, h2⟩  

end quadratic_P_value_l349_349125


namespace xyz_distinct_real_squares_l349_349164

theorem xyz_distinct_real_squares (x y z : ℝ) 
  (h1 : x^2 = 2 + y)
  (h2 : y^2 = 2 + z)
  (h3 : z^2 = 2 + x) 
  (h4 : x ≠ y ∧ y ≠ z ∧ z ≠ x) :
  x^2 + y^2 + z^2 = 5 ∨ x^2 + y^2 + z^2 = 6 ∨ x^2 + y^2 + z^2 = 9 :=
by 
  sorry

end xyz_distinct_real_squares_l349_349164


namespace jane_receives_an_A_jane_receives_an_A_version_l349_349916

variables (A : Prop) (Q : Prop) (E : Prop)

-- Conditions: 
-- A: Jane received an A
-- Q: Jane answered at least 90% of the multiple choice questions correctly
-- E: Jane completed the extra credit question
axiom condition : A ↔ (Q ∧ E)

-- The proof problem
theorem jane_receives_an_A : A → (Q ∧ E) :=
begin
  assume hA : A,
  exact condition.mp hA,
end

theorem jane_receives_an_A_version : A → (Q ∧ E) := 
by {
  assume hA,
  exact condition.mp hA,
}

end jane_receives_an_A_jane_receives_an_A_version_l349_349916


namespace find_p_and_q_l349_349472

theorem find_p_and_q (ABC : Triangle)
  (AB BC CA : ℝ) (M N : Point) (O H : Point)
  (E F P Q K X Y Z R S T : Point)
  (conditions : 
    AB = 3 * sqrt 30 - sqrt 10 ∧ 
    BC = 12 ∧ 
    CA = 3 * sqrt 30 + sqrt 10 ∧ 
    midpoint M A B ∧ 
    midpoint N A C ∧ 
    circumcenter O ABC ∧ 
    orthocenter H ABC ∧ 
    passes_through l O H ∧ 
    foot E B l ∧ 
    foot F C l ∧ 
    reflection l l' BC ∧ 
    intersects l' AE P ∧ 
    intersects l' AF Q ∧ 
    intersects BP CQ K ∧ 
    reflection K X BC.bisector ∧ 
    reflection K Y CA.bisector ∧ 
    reflection K Z AB.bisector ∧ 
    midpoint R X Y ∧ 
    midpoint S X Z ∧ 
    intersects MR NS T) :
  let OT := distance O T 
  ∃ p q : ℕ, nat.coprime p q ∧ OT = p / q ∧ 100 * p + q = ? :=
sorry

end find_p_and_q_l349_349472


namespace binomial_10_2_equals_45_l349_349680

open Nat

theorem binomial_10_2_equals_45 : Nat.choose 10 2 = 45 := 
by
  sorry

end binomial_10_2_equals_45_l349_349680


namespace num_ways_to_place_pawns_l349_349760

theorem num_ways_to_place_pawns :
  let n := 5 in
  let pawns := Finset.range n in
  let positions := Finset.pi Finset.univ (λ _, pawns) in
  let valid_placements := {placement | ∀ i j, i ≠ j → placement i ≠ placement j} in
  positions.card = 120 := by
  -- Number of permutations of 5 distinct elements
  sorry

end num_ways_to_place_pawns_l349_349760


namespace seventh_graders_trip_count_l349_349936

theorem seventh_graders_trip_count (fifth_graders sixth_graders teachers_per_grade parents_per_grade grades buses seats_per_bus : ℕ) 
  (hf : fifth_graders = 109) 
  (hs : sixth_graders = 115)
  (ht : teachers_per_grade = 4) 
  (hp : parents_per_grade = 2) 
  (hg : grades = 3) 
  (hb : buses = 5)
  (hsb : seats_per_bus = 72) : 
  ∃ seventh_graders : ℕ, seventh_graders = 118 := 
by
  sorry

end seventh_graders_trip_count_l349_349936


namespace andy_loss_more_likely_than_win_l349_349268

def prob_win_first := 0.30
def prob_lose_first := 0.70

def prob_win_second := 0.50
def prob_lose_second := 0.50

def prob_win_both := prob_win_first * prob_win_second
def prob_lose_both := prob_lose_first * prob_lose_second
def diff_probability := prob_lose_both - prob_win_both
def percentage_more_likely := (diff_probability / prob_win_both) * 100

theorem andy_loss_more_likely_than_win :
  percentage_more_likely = 133.33 := sorry

end andy_loss_more_likely_than_win_l349_349268


namespace compare_extreme_and_maximum_l349_349364

-- Define the functions f and h
def f (x : ℝ) := Real.exp x + (x^2) / 2 - Real.log x
def h (x : ℝ) := Real.log x / (2 * x)

-- Define the extreme point x1 and maximum point x2
variable (x1 x2 : ℝ)
hypothesis (H1 : x1 ∈ Set.Ioo (1/4) (1/2)) -- x1 is in the open interval (1/4, 1/2)
hypothesis (H2 : x2 = Real.exp 1) -- x2 is equal to e

-- The statement to be proved
theorem compare_extreme_and_maximum : x1 > x2 := sorry

end compare_extreme_and_maximum_l349_349364


namespace positive_difference_of_expressions_l349_349576

theorem positive_difference_of_expressions :
  let a := 8
  let expr1 := (a^2 - a^2) / a
  let expr2 := (a^2 * a^2) / a
  expr1 = 0 → expr2 = 512 → 512 - 0 = 512 := 
by
  introv h_expr1 h_expr2
  rw [h_expr1, h_expr2]
  norm_num
  exact rfl

end positive_difference_of_expressions_l349_349576


namespace larger_of_two_numbers_l349_349208

theorem larger_of_two_numbers (A B : ℕ) (HCF : ℕ) (factor1 factor2 : ℕ) (h_hcf : HCF = 23) (h_factor1 : factor1 = 13) (h_factor2 : factor2 = 14)
(hA : A = HCF * factor1) (hB : B = HCF * factor2) :
  max A B = 322 :=
by
  sorry

end larger_of_two_numbers_l349_349208


namespace could_be_simple_random_sampling_l349_349459

-- Conditions
def boys : Nat := 20
def girls : Nat := 30
def total_students : Nat := boys + girls
def sample_size : Nat := 10
def boys_in_sample : Nat := 4
def girls_in_sample : Nat := 6

-- Theorem Statement
theorem could_be_simple_random_sampling :
  boys = 20 ∧ girls = 30 ∧ sample_size = 10 ∧ boys_in_sample = 4 ∧ girls_in_sample = 6 →
  (∃ (sample_method : String), sample_method = "simple random sampling"):=
by 
  sorry

end could_be_simple_random_sampling_l349_349459


namespace triangle_BD_zero_l349_349516

theorem triangle_BD_zero
  (ABC ABD : Triangle)
  (right_triangle_ABC : ABC.isRightTriangle)
  (right_triangle_ABD : ABD.isRightTriangle)
  (hypotenuse_AB_ABC : ABC.hypotenuse = AB)
  (hypotenuse_AB_ABD : ABD.hypotenuse = AB)
  (BC : ℝ) (AC : ℝ) (AD : ℝ)
  (BC_eq_3 : BC = 3)
  (AC_eq_4 : AC = 4)
  (AD_eq_5 : AD = 5)
  : (∃ BD : ℝ, BD = 0) :=
by
  sorry

end triangle_BD_zero_l349_349516


namespace polynomial_value_at_8_l349_349108

noncomputable def P : ℝ → ℝ := λ x, x^2 - x + 2

theorem polynomial_value_at_8 :
  (P (P x) = x^4 - 2*x^3 + 4*x^2 - 3*x + 4) ∧ (P = λ x, x^2 - x + 2) → P 8 = 58 :=
by
  sorry

end polynomial_value_at_8_l349_349108


namespace Gibbs_triangular_diagram_l349_349717

structure Point := (x y : ℝ)

variables A B C : Point

def P1 (C A : Point) : Point :=
{ x := C.x + 0.6 * (A.x - C.x), y := C.y + 0.6 * (A.y - C.y) }

def P2 (A B : Point) : Point :=
{ x := A.x + 0.3 * (B.x - A.x), y := A.y + 0.3 * (B.y - A.y) }

noncomputable def line (P Q : Point) : ℝ → Point :=
λ t, { x := P.x + t * (Q.x - P.x), y := P.y + t * (Q.y - P.y) }

noncomputable def intersection (line1 line2 : ℝ → Point) : Point :=
sorry -- Details of intersection computation are omitted

theorem Gibbs_triangular_diagram (A B C : Point) :
    ∃ K : Point,
    let P1 := P1 C A,
    let P2 := P2 A B,
    let line1 := line P1 (λ t, { x := t, y := 0 }), -- line parallel to BC
    let line2 := line P2 (λ t, { x := 0, y := t }) -- line parallel to AC
    in K = intersection line1 line2 :=
sorry

end Gibbs_triangular_diagram_l349_349717


namespace matrix_determinant_l349_349296

theorem matrix_determinant : ∀ (y : ℝ), 
  det ![
    ![y + 2, y - 1, y + 1],
    ![y + 1, y + 2, y - 1],
    ![y - 1, y + 1, y + 2]] = 6 * y^2 + 23 * y + 14 :=
by
  intros
  sorry

end matrix_determinant_l349_349296


namespace origin_movement_by_dilation_l349_349237

/-- Given a dilation of the plane that maps a circle with radius 4 centered at (3,3) 
to a circle of radius 6 centered at (7,9), calculate the distance the origin (0,0)
moves under this transformation to be 0.5 * sqrt(10). -/
theorem origin_movement_by_dilation :
  let B := (3, 3)
  let B' := (7, 9)
  let radius_B := 4
  let radius_B' := 6
  let dilation_factor := radius_B' / radius_B
  let center_of_dilation := (-1, -3)
  let initial_distance := Real.sqrt ((-1)^2 + (-3)^2) 
  let moved_distance := dilation_factor * initial_distance
  moved_distance - initial_distance = 0.5 * Real.sqrt (10) := 
by
  sorry

end origin_movement_by_dilation_l349_349237


namespace minimum_integer_discount_l349_349623

noncomputable def effective_discount (r : ℝ) (d1 : ℝ) (d2 : ℝ) (d3 : ℝ) : ℝ :=
let x := 1 in -- original price normalized to 1
  max (1 - (1 - d1) * (1 - d2))
      (max (1 - (1 - r)^4)
           (1 - (1 - d2) * (1 - d3)))

theorem minimum_integer_discount : 
  ∀ n : ℕ, n ≥ 38 →
  ∃ n_min : ℕ, n_min = n ∧
  effective_discount 0.08 0.2 0.1 0.3 < (1 - (n / 100 : ℝ)) :=
begin
  sorry
end

end minimum_integer_discount_l349_349623


namespace coefficient_x_in_expansion_l349_349469

theorem coefficient_x_in_expansion :
  let p := (1 + X)^3 + (1 + X)^3 + (1 + X)^3 in
  polynomial.coeff p 1 = 9 :=
by {
  sorry
}

end coefficient_x_in_expansion_l349_349469


namespace at_least_one_misses_l349_349453

-- Definitions for the given conditions
variables {p q : Prop}

-- Lean 4 statement proving the equivalence
theorem at_least_one_misses (hp : p → false) (hq : q → false) : (¬p ∨ ¬q) :=
by sorry

end at_least_one_misses_l349_349453


namespace remainder_modulo_9_l349_349167

open Int

theorem remainder_modulo_9 (a b c d : ℕ) (ha : a < 9) (hb : b < 9) (hc : c < 9) (hd : d < 9)
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (hinv : ∀ x ∈ {a, b, c, d}, Nat.gcd x 9 = 1) :
  ((a * b * c + a * b * d + a * c * d + b * c * d) * (a * b * c * d)⁻¹) % 9 = 6 :=
sorry

end remainder_modulo_9_l349_349167


namespace quadratic_P_value_l349_349123

noncomputable def P (x : ℝ) : ℝ :=
  x^2 - x + 2

theorem quadratic_P_value :
  P (P 8) = 8^4 - 2 * 8^3 + 4 * 8^2 - 3 * 8 + 4 ∧ P 8 = 58 := 
by
  have h1 : P(P(8)) = 8^4 - 2 * 8^3 + 4 * 8^2 - 3 * 8 + 4 := sorry
  have h2 : P(8) = 58 := sorry
  exact ⟨h1, h2⟩  

end quadratic_P_value_l349_349123


namespace P_P_eq_P_eight_equals_58_l349_349091

open Polynomial

noncomputable def P(x : ℚ) : ℚ := x^2 - x + 2

theorem P_P_eq :
  (P ∘ P)(x) = x^4 - 2 * x^3 + 4 * x^2 - 3 * x + 4 := sorry

theorem P_eight_equals_58 :
  P 8 = 58 := sorry

end P_P_eq_P_eight_equals_58_l349_349091


namespace above_parabola_probability_l349_349531

theorem above_parabola_probability :
  let S := {p : ℕ × ℕ | 1 ≤ p.1 ∧ p.1 ≤ 9 ∧ 1 ≤ p.2 ∧ p.2 ≤ 9} in
  let count_above_parabola : ℕ := S.to_finset.filter (λ p, p.2 > p.1^2 + p.1*p.2).card in
  ∃ S : finset (ℕ × ℕ), (∀ p ∈ S, 1 ≤ p.1 ∧ p.1 ≤ 9 ∧ 1 ≤ p.2 ∧ p.2 ≤ 9) ∧
  (count_above_parabola.to_float / (S.card.to_float)) = 7 / 81 := 
begin
  let S : finset (ℕ × ℕ) := finset.univ.filter (λ p, 1 ≤ p.1 ∧ p.1 ≤ 9 ∧ 1 ≤ p.2 ∧ p.2 ≤ 9),
  let count_above_parabola : ℕ := S.filter (λ p, p.2 > p.1^2 + p.1*p.2).card,
  use S,
  split,
  { intros p hp, simp only [finset.mem_filter, finset.mem_univ] at hp, exact hp.1, },
  { sorry }
end

end above_parabola_probability_l349_349531


namespace number_of_primes_between_30_and_50_l349_349417

/-- 
  Prove that there are exactly 5 prime numbers in the range from 30 to 50. 
  These primes are 31, 37, 41, 43, and 47.
-/
theorem number_of_primes_between_30_and_50 : 
  (Finset.filter Nat.Prime (Finset.range 51)).card - 
  (Finset.filter Nat.Prime (Finset.range 31)).card = 5 := 
by 
  sorry

end number_of_primes_between_30_and_50_l349_349417


namespace point_geq_incenter_l349_349485

variables {A B C I P : Type} [IsTriangle A B C] [IsIncenter A B C I] [Interior A B C P]

open Classical Real

/-- If P is a point inside triangle ABC such that the sum of certain angles involving P, B, and C satisfies 
  the given condition, then the inequality AP ≥ AI holds with equality iff P = I. -/
theorem point_geq_incenter (h₁ : ∠ P B A + ∠ P C A = ∠ P B C + ∠ P C B) :
  dist A P ≥ dist A I ∧ (dist A P = dist A I ↔ P = I) :=
by
  sorry

end point_geq_incenter_l349_349485


namespace geo_seq_first_term_and_terms_find_four_numbers_l349_349613

-- Problem 1
theorem geo_seq_first_term_and_terms (a1 : ℤ) (n : ℕ) (a5 : ℤ) (Sn : ℤ) (q : ℤ) :
  a5 = a1 * q^4 ∧ Sn = a1 * (q - 1) * (q^n - 1) / (q - 1) → a1 = 2 ∧ n = 5 :=
by {
  sorry
}

-- Problem 2
theorem find_four_numbers (a q x y z w : ℤ) :
  a^3 = 216 ∧ a + aq + 2aq - a = 36 ∧ x = a/q ∧ y = a ∧ z = aq ∧ w = 2aq - a →
  x = 3 ∧ y = 6 ∧ z = 12 ∧ w = 18 :=
by {
  sorry
}

end geo_seq_first_term_and_terms_find_four_numbers_l349_349613


namespace range_of_omega_l349_349373

theorem range_of_omega 
    (f : ℝ → ℝ) 
    (ω : ℝ) 
    (h1 : ∀ x, f x = 2 * Real.sin (ω * x + Real.pi / 4)) 
    (h2 : ω > 0)
    (h3 : ∃ x1 x2 x3 : ℝ, 0 ≤ x1 ∧ x1 ≤ 1 ∧ 0 ≤ x2 ∧ x2 ≤ 1 ∧ 0 ≤ x3 ∧ x3 ≤ 1 ∧ 
           (∀ x ∈ Set.Icc 0 1, f x ≤ f x1 ∧ f x ≤ f x2 ∧ f x ≤ f x3)) :
  ∀ x, (Real.pi / 4 + ω * 0 ≤ x ∧ x ≤ Real.pi / 4 + ω * 1 ∧ 
  (17 * Real.pi / 4 ≤ ω ∧ ω < 25 * Real.pi / 4)) :=
begin
  sorry
end

end range_of_omega_l349_349373


namespace num_four_digit_numbers_l349_349387

theorem num_four_digit_numbers : 
  let N := (λ (a b c d : ℕ), 1000 * a + 100 * b + 10 * c + d) in
  ∃ (count : ℕ), 
    (∀ (a b c d : ℕ), 
      5000 ≤ N a b c d ∧ N a b c d < 7000 ∧
      d % 2 = 0 ∧ 
      2 ≤ b ∧ b ≤ c ∧ c ≤ 7 ∧ b + c = 9 →
      true) ∧
    count = 60 :=
by
  sorry

end num_four_digit_numbers_l349_349387


namespace number_of_triangles_l349_349271

theorem number_of_triangles (points : List ℝ) (h₀ : points.length = 12)
  (h₁ : ∀ p ∈ points, p ≠ A ∧ p ≠ B ∧ p ≠ C ∧ p ≠ D): 
  (∃ triangles : ℕ, triangles = 216) :=
  sorry

end number_of_triangles_l349_349271


namespace binom_10_2_eq_45_l349_349705

theorem binom_10_2_eq_45 :
  binom 10 2 = 45 := by
  sorry

end binom_10_2_eq_45_l349_349705


namespace divisibility_by_5_l349_349923

theorem divisibility_by_5 (n : ℕ) (h : 0 < n) : (1^n + 2^n + 3^n + 4^n) % 5 = 0 ↔ n % 4 ≠ 0 := 
sorry

end divisibility_by_5_l349_349923


namespace tangent_line_at_3_when_a_2_monotonicity_and_extreme_values_of_g_l349_349797

noncomputable def f (x : ℝ) (a : ℝ) := (1/3) * x^3 - (1/2) * a * x^2

theorem tangent_line_at_3_when_a_2 :
  let a := 2 in
  let f := λ x, (1/3) * x^3 - x^2 in
  let y := f(3) in
  let k := (λ x, x^2 - 2*x) 3 in
  ∀ x y, y = f 3 → k = 3 → y = 0 →
  3 * x - y - 9 = 0 := sorry

noncomputable def g (x : ℝ) (a : ℝ) := f x a + (x - a) * Real.cos x - Real.sin x

theorem monotonicity_and_extreme_values_of_g (a : ℝ):
  ∀ x, let g := λ x, f x a + (x - a) * Real.cos x - Real.sin x in
  g'(x) = x^2 - a * x + (x - a) * Real.sin x →
  (if a > 0 then 
    ((∀ x < 0, g'(x) > 0) ∧ (∀ x > a, g'(x) > 0) ∧ (∀ 0 < x < a, g'(x) < 0) ∧
    (g a = - (1/6) * a^3 - Real.sin a) ∧ (g 0 = -a))
  else if a < 0 then 
    ((∀ x > 0, g'(x) > 0) ∧ (∀ x < a, g'(x) > 0) ∧ (∀ a < x < 0, g'(x) < 0) ∧ 
    (g a = - (1/6) * a^3 - Real.sin a) ∧ (g 0 = -a))
  else 
    ((∀ x ≠ 0, g'(x) > 0) ∧ (∀ x, g x = (1/3) * x^3 + x * Real.cos x - Real.sin x ∧ g' x = x * (x + Real.sin x)))) := sorry

end tangent_line_at_3_when_a_2_monotonicity_and_extreme_values_of_g_l349_349797


namespace derivative_evaluation_l349_349754

noncomputable def f (x : ℝ) : ℝ := Real.exp x
def f_prime (x : ℝ) : ℝ := Real.exp x

theorem derivative_evaluation :
  f_prime (-2) = Real.exp (-2) := by
  sorry

end derivative_evaluation_l349_349754


namespace total_cost_of_shoes_before_discount_l349_349875

theorem total_cost_of_shoes_before_discount (S J H : ℝ) (D : ℝ) (shoes jerseys hats : ℝ) :
  jerseys = 1/4 * shoes ∧
  hats = 2 * jerseys ∧
  D = 0.9 * (6 * shoes + 4 * jerseys + 3 * hats) ∧
  D = 620 →
  6 * shoes = 486.30 := by
  sorry

end total_cost_of_shoes_before_discount_l349_349875


namespace right_triangle_third_side_length_l349_349788

theorem right_triangle_third_side_length {x y : ℝ} (h : abs (x^2 - 4) + sqrt (y^2 - 5 * y + 6) = 0) :
  (∃ c : ℝ, c = sqrt (x^2 + y^2) ∧ (c = 2 * sqrt 2 ∨ c = sqrt 13 ∨ c = sqrt 5)) :=
begin
  sorry -- Proof goes here
end

end right_triangle_third_side_length_l349_349788


namespace sum_of_consecutive_multiples_of_4_l349_349991

theorem sum_of_consecutive_multiples_of_4 (n : ℝ) (h : 4 * n + (4 * n + 8) = 140) :
  4 * n + (4 * n + 4) + (4 * n + 8) = 210 :=
sorry

end sum_of_consecutive_multiples_of_4_l349_349991


namespace arithmetic_sequence_sum_formula_l349_349982

variable {α : Type*} [LinearOrderedRing α]

def arithmetic_sum (a_1 d : α) (S : ℕ → α) : Prop :=
  ∀ n, S n = n * a_1 + (n * (n - 1)) / 2 * d

theorem arithmetic_sequence_sum_formula (a_1 d : α) (S : ℕ → α)
  (h1 : ∀ n, arithmetic_sum a_1 d S n) :
  ∀ k, S k = k * a_1 + (k * (k - 1)) / 2 * d :=
by
  intro k
  apply h1 k

end arithmetic_sequence_sum_formula_l349_349982


namespace min_value_of_expression_l349_349776

-- Define the conditions: A is an acute angle.
def is_acute_angle (A : ℝ) : Prop :=
  0 < A ∧ A < π / 2

-- Define the expression to evaluate.
def expression (A : ℝ) : ℝ :=
  real.sqrt (real.sin A ^ 4 + 1) + real.sqrt (real.cos A ^ 4 + 4)

-- State the theorem to be proven.
theorem min_value_of_expression (A : ℝ) (hA : is_acute_angle A) :
  ∃ m, m = expression A ∧ (∀ x, is_acute_angle x → expression x ≥ m) ∧ m = real.sqrt 10 :=
sorry

end min_value_of_expression_l349_349776


namespace solution_proof_l349_349820

def count_multiples (n : ℕ) (m : ℕ) (limit : ℕ) : ℕ :=
  (limit - 1) / m + 1

def problem_statement : Prop :=
  let multiples_of_10 := count_multiples 1 10 300
  let multiples_of_10_and_6 := count_multiples 1 30 300
  let multiples_of_10_and_11 := count_multiples 1 110 300
  let unwanted_multiples := multiples_of_10_and_6 + multiples_of_10_and_11
  multiples_of_10 - unwanted_multiples = 20

theorem solution_proof : problem_statement :=
  by {
    sorry
  }

end solution_proof_l349_349820


namespace number_of_routes_l349_349278

variable {City : Type}
variable (A B C D E : City)
variable (AB_N AB_S AD AE BC BD CD DE : City → City → Prop)
  
theorem number_of_routes 
  (hAB_N : AB_N A B) (hAB_S : AB_S A B)
  (hAD : AD A D) (hAE : AE A E)
  (hBC : BC B C) (hBD : BD B D)
  (hCD : CD C D) (hDE : DE D E) :
  ∃ r : ℕ, r = 16 := 
sorry

end number_of_routes_l349_349278


namespace inequality_and_equality_condition_l349_349508

variable (n : ℕ) 
variable (x : Fin (n + 1) → ℝ) 
variable (x_nonneg : ∀ i, 0 ≤ x i)
variable (a : ℝ)
variable (a_is_min : ∀ i, a ≤ x i)

theorem inequality_and_equality_condition :
  (∑ j : Fin n, (1 + x j) / (1 + x ⟨(j.val + 1) % n, sorry⟩)) ≤ n + (1 / (1 + a) ^ 2) * (∑ j : Fin n, (x j - a) ^ 2) ∧
  ( (∑ j : Fin n, (1 + x j) / (1 + x ⟨(j.val + 1) % n, sorry⟩)) = n + (1 / (1+ a) ^ 2) * (∑ j : Fin n, (x j - a) ^ 2) 
    ↔ ∀ i, x i = a ) := sorry

end inequality_and_equality_condition_l349_349508


namespace perimeter_triangle_inequality_l349_349851

theorem perimeter_triangle_inequality
  (ABC : Triangle) (h_acute : ABC.isAcute) 
  (H : Altitudes ABC AA_1 BB_1 CC_1)
  (P_ABC : ℝ) (P_A1B1C1 : ℝ) (r R : ℝ)
  (h_ratio : P_A1B1C1 / P_ABC = r / R)
  (h_ineq : r ≤ R / 2) :
  P_A1B1C1 ≤ (1 / 2) * P_ABC := by
  sorry

end perimeter_triangle_inequality_l349_349851


namespace altitude_ratio_cos_l349_349845

variable {α : Type*} [LinearOrderedField α]

theorem altitude_ratio_cos 
  {A B C D E : α}
  (h_angle_A : ∠A = a)
  (h_CD_altitude : is_altitude CD AB)
  (h_BE_altitude : is_altitude BE AC) :
  DE / BC = |cos a| :=
sorry

end altitude_ratio_cos_l349_349845


namespace basketball_team_wins_l349_349618

theorem basketball_team_wins (f : ℚ) (h1 : 40 + 40 * f + (40 + 40 * f) = 130) : f = 5 / 8 :=
by
  sorry

end basketball_team_wins_l349_349618


namespace resultant_force_magnitude_distance_A_B_approx_l349_349171

open Complex

def f1 : ℂ := sqrt 2 * (cos (π / 4) + sin (π / 4) * I)
def f2 : ℂ := 2 * (cos (-π / 6) + sin (-π / 6) * I)

theorem resultant_force_magnitude :
  abs (f1 + f2) = sqrt 3 + 1 :=
  sorry

def A : ℂ := f1
def B : ℂ := f2

theorem distance_A_B_approx :
  abs (A - B) ≈ 2.1 :=
  sorry

-- Note: The ≈ symbol here is used to indicate approximation, which typically would be defined 
-- more formally depending on context.

end resultant_force_magnitude_distance_A_B_approx_l349_349171


namespace crossing_time_l349_349178

namespace TrainProblem

def LengthTrain : ℝ := 120
def TimeToPostFirstTrain : ℝ := 12
def TimeToPostSecondTrain : ℝ := 20

def SpeedFirstTrain : ℝ := LengthTrain / TimeToPostFirstTrain
def SpeedSecondTrain : ℝ := LengthTrain / TimeToPostSecondTrain

def RelativeSpeed : ℝ := SpeedFirstTrain + SpeedSecondTrain
def TotalDistance : ℝ := 2 * LengthTrain

theorem crossing_time: (TotalDistance / RelativeSpeed) = 15 := by
  sorry

end TrainProblem

end crossing_time_l349_349178


namespace problem_1_problem_2_l349_349805

noncomputable def a_n (n : ℕ) : ℕ :=
  if h : n = 0 then 0 else n * 2^(n-1) - 2^n + 1

noncomputable def S_n (n : ℕ) : ℕ :=
  ∑ k in (Finset.range n).map (Finset.nth_le ((λ i, i + 1) <$> Finset.range n) n (by linarith)),
    a_n k

theorem problem_1 (n : ℕ) (h : n ≠ 0) : a_n n = n * 2^(n-1) - 2^n + 1 := sorry

theorem problem_2 (n : ℕ) : S_n n = (n - 3) * 2^n + n + 3 := sorry

end problem_1_problem_2_l349_349805


namespace square_area_of_inscribed_in_parabola_l349_349254

theorem square_area_of_inscribed_in_parabola : 
    ∀ (s : ℝ), -2s = (3 + s)^2 - 6*(3 + s) + 7 → (2*s)^2 = 16 - 8*Real.sqrt 3 :=
by
  intro s hs
  sorry

end square_area_of_inscribed_in_parabola_l349_349254


namespace factor_polynomial_l349_349750

theorem factor_polynomial {x : ℝ} : 4 * x^3 - 16 * x = 4 * x * (x + 2) * (x - 2) := 
sorry

end factor_polynomial_l349_349750


namespace count_primes_between_30_and_50_l349_349407

-- Define the range of numbers from 30 to 50
def range_30_to_50 := Set.of_list (List.range' 30 (51 - 30))

-- Define a predicate to check if a number is prime
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

-- Extract all prime numbers in the specified range
def primes_between_30_and_50 : List ℕ :=
  List.filter is_prime (Set.toList range_30_to_50)

theorem count_primes_between_30_and_50 : primes_between_30_and_50.length = 5 :=
by
  -- The proof goes here
  sorry

end count_primes_between_30_and_50_l349_349407


namespace count_primes_between_30_and_50_l349_349411

-- Define the range of numbers from 30 to 50
def range_30_to_50 := Set.of_list (List.range' 30 (51 - 30))

-- Define a predicate to check if a number is prime
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

-- Extract all prime numbers in the specified range
def primes_between_30_and_50 : List ℕ :=
  List.filter is_prime (Set.toList range_30_to_50)

theorem count_primes_between_30_and_50 : primes_between_30_and_50.length = 5 :=
by
  -- The proof goes here
  sorry

end count_primes_between_30_and_50_l349_349411


namespace problem_A1_rational_condition_l349_349732

theorem problem_A1_rational_condition (n : ℕ) (h_n : 3 ≤ n) :
  (∀ (a : ℕ → ℝ), 
    (∀ i : ℕ, 0 < i ∧ i ≤ n → a i = a 1 + (i-1)*(a 2 - a 1)) → 
    (∃ k : ℕ, 1 ≤ k ∧ k ≤ n ∧ a k ∈ ℚ) → 
    (∃ k : ℕ, 1 ≤ k ∧ k ≤ n ∧ a k ∈ ℚ)) ↔ 
  (n % 3 = 1) :=
sorry

end problem_A1_rational_condition_l349_349732


namespace child_sum_announcements_l349_349219

theorem child_sum_announcements (a : ℕ → ℕ)
    (h1 : ∀ i, 1 ≤ i ∧ i ≤ 9 -> a i + a (i + 9) = 10 - i)
    (h_total : ∑ i in range 9, 10 - i + S_10 = 2 * ∑ i in range 10, a i) :
    S_10 = 1 :=
begin
  sorry
end

end child_sum_announcements_l349_349219


namespace f_g_2_minus_g_f_2_eq_20_l349_349504

def f (x : ℝ) : ℝ := x^2 + 2
def g (x : ℝ) : ℝ := x / 2 + 4

theorem f_g_2_minus_g_f_2_eq_20 : f (g 2) - g (f 2) = 20 := by
  sorry

end f_g_2_minus_g_f_2_eq_20_l349_349504


namespace distance_ab_l349_349175

noncomputable def distance (speed_a speed_b time_first_meet) := speed_a * time_first_meet + speed_b * time_first_meet

theorem distance_ab (speed_a speed_b : ℕ) (time_first_meet : ℕ) :
  let time_to_meet_again := (2 * speed_b * time_first_meet) / speed_a - (2 * speed_a * time_first_meet) / speed_b
  in time_to_meet_again = 14 → distance speed_a speed_b time_first_meet = 1680 :=
by
  sorry

end distance_ab_l349_349175


namespace calculate_expression_l349_349668

theorem calculate_expression : 
  (-7 : ℤ)^7 / (7 : ℤ)^4 + 2^6 - 8^2 = -343 :=
by
  sorry

end calculate_expression_l349_349668


namespace sum_x_y_l349_349019

theorem sum_x_y (x y : ℤ) (h1 : x - y = 40) (h2 : x = 32) : x + y = 24 := by
  sorry

end sum_x_y_l349_349019


namespace luncheon_cost_l349_349143

theorem luncheon_cost
  (s c p : ℝ)
  (h1 : 3 * s + 7 * c + p = 3.15)
  (h2 : 4 * s + 10 * c + p = 4.20) :
  s + c + p = 1.05 :=
by sorry

end luncheon_cost_l349_349143


namespace polynomial_not_factorable_l349_349380

theorem polynomial_not_factorable (b c d : Int) (h₁ : (b * d + c * d) % 2 = 1) : 
  ¬ ∃ p q r : Int, (x + p) * (x^2 + q * x + r) = x^3 + b * x^2 + c * x + d :=
by 
  sorry

end polynomial_not_factorable_l349_349380


namespace sale_in_first_month_l349_349239

theorem sale_in_first_month
  (m2 : ℕ := 7927) 
  (m3 : ℕ := 7855) 
  (m4 : ℕ := 8230) 
  (m5 : ℕ := 7562) 
  (m6 : ℕ := 5991) 
  (avg : ℕ := 7500)
  (n_months : ℕ := 6) :
  (let total_sale := avg * n_months in 
   let sum_known_sales := m2 + m3 + m4 + m5 + m6 in
   total_sale - sum_known_sales = 7435) :=
by 
  sorry

end sale_in_first_month_l349_349239


namespace find_a_l349_349450

theorem find_a (a : ℝ) :
  (∃ b : ℝ, 4 * b + 3 = 7 ∧ 5 * (-b) - 1 = 2 * (-b) + a) → a = -4 :=
by
  sorry

end find_a_l349_349450


namespace region_area_is_four_l349_349306

-- Definitions matching the conditions
def r_sec_theta (θ : ℝ) : ℝ := 2 / (Real.cos θ)
def r_csc_theta (θ : ℝ) : ℝ := 2 / (Real.sin θ)

-- The region bounded by the graphs and the axes form a square from origin to (2, 2)

-- Lean statement for proving the area of the described region is 4
theorem region_area_is_four : 
  let r_sec_theta (θ : ℝ) := 2 / (Real.cos θ),
      r_csc_theta (θ : ℝ) := 2 / (Real.sin θ) in
  let vertices := [(0, 0), (2, 0), (2, 2), (0, 2)] in
  let area := (2 - 0) * (2 - 0) in
  area = 4 :=
by sorry

end region_area_is_four_l349_349306


namespace greatest_b_value_l349_349150

def equation_has_integer_solutions (b : ℕ) : Prop :=
  ∃ (x : ℤ), x * (x + b) = -20

theorem greatest_b_value : ∃ (b : ℕ), b = 21 ∧ equation_has_integer_solutions b :=
by
  sorry

end greatest_b_value_l349_349150


namespace regular_14_gon_inequality_l349_349632

noncomputable def side_length_of_regular_14_gon : ℝ := 2 * Real.sin (Real.pi / 14)

theorem regular_14_gon_inequality (a : ℝ) (h : a = side_length_of_regular_14_gon) :
  (2 - a) / (2 * a) > Real.sqrt (3 * Real.cos (Real.pi / 7)) :=
by
  sorry

end regular_14_gon_inequality_l349_349632


namespace unique_pair_a_b_l349_349323

open Complex

theorem unique_pair_a_b :
  ∃! (a b : ℂ), a^4 * b^3 = 1 ∧ a^6 * b^7 = 1 := by
  sorry

end unique_pair_a_b_l349_349323


namespace binom_10_2_eq_45_l349_349689

theorem binom_10_2_eq_45 : Nat.binomial 10 2 = 45 := by
  sorry

end binom_10_2_eq_45_l349_349689


namespace problem_ab_cd_l349_349497

def land (a b : ℝ) : ℝ := if a ≤ b then a else b
def lor (a b : ℝ) : ℝ := if a ≤ b then b else a

theorem problem_ab_cd (a b c d : ℝ) (h_ab : 0 < a) (h_ab' : 0 < b) (h1 : a * b ≤ 4) (h_cd : 0 < c) (h_cd' : 0 < d) (h2 : c + d ≥ 4) :
       (land a b) ≤ 2 ∧ (lor c d) ≥ 2 :=
sorry

end problem_ab_cd_l349_349497


namespace binom_10_2_eq_45_l349_349698

-- Definitions used in the conditions
def binom (n k : ℕ) := n.choose k

-- The statement that needs to be proven
theorem binom_10_2_eq_45 : binom 10 2 = 45 :=
by
  sorry

end binom_10_2_eq_45_l349_349698


namespace determination_of_C_and_D_l349_349725

theorem determination_of_C_and_D :
  (∃ C D : ℚ, (∀ x ≠ 11, x ≠ -5, 7 * x - 4 = C * (x + 5) + D * (x - 11)) ∧ 
               C = 73 / 16 ∧ 
               D = 39 / 16) :=
by {
  sorry
}

end determination_of_C_and_D_l349_349725


namespace product_of_all_eight_rolls_is_odd_prime_l349_349646

noncomputable def probability_odd_prime_product : ℚ :=
  let outcomes := {1, 2, 3, 4, 5, 6}
  let odd_primes := {3, 5}
  let prob_single_roll_odd_prime := 2 / 6 -- Probability of rolling a 3 or 5 on a single roll
  prob_single_roll_odd_prime ^ 8 -- Probability of rolling a 3 or 5 on all eight rolls

theorem product_of_all_eight_rolls_is_odd_prime :
  probability_odd_prime_product = 1 / 6561 :=
by
  sorry -- Proof to be filled in

end product_of_all_eight_rolls_is_odd_prime_l349_349646


namespace egg_rolls_total_l349_349918

def total_egg_rolls (omar_rolls : ℕ) (karen_rolls : ℕ) : ℕ :=
  omar_rolls + karen_rolls

theorem egg_rolls_total :
  total_egg_rolls 219 229 = 448 :=
by
  sorry

end egg_rolls_total_l349_349918


namespace joan_games_l349_349049

theorem joan_games (games_this_year games_total games_last_year : ℕ) 
  (h1 : games_this_year = 4) 
  (h2 : games_total = 9) 
  (h3 : games_total = games_this_year + games_last_year) :
  games_last_year = 5 :=
by {
  -- The proof goes here
  sorry
}

end joan_games_l349_349049


namespace sequence_no_division_l349_349721

theorem sequence_no_division (m n : ℤ) (p : ℕ) (hp_prime : Nat.prime p) (hp_gt_5 : p > 5) (hp_mod_4 : p % 4 = 1) :
  ∃ m n : ℤ, ∀ k : ℕ, p ∣ (sequence m n k) → False :=
by
  sorry

-- Definition of the sequence based on the recurrence relation
def sequence (m n : ℤ) : ℕ → ℤ
| 0     := m
| 1     := n
| (k+2) := 4 * (sequence m n (k+1)) - 5 * (sequence m n k)

end sequence_no_division_l349_349721


namespace sqrt_of_three_a_plus_b_l349_349441

theorem sqrt_of_three_a_plus_b (a b : ℤ) (h1 : a + 7 = 9) (h2 : 2b + 2 = 8) : (Nat.sqrt (3 * a + b) = 3 ∨ Nat.sqrt (3 * a + b) = -3) :=
by
  sorry

end sqrt_of_three_a_plus_b_l349_349441


namespace quadratic_polynomial_P_l349_349113

noncomputable def P : Polynomial := {
  to_fun := λ x : ℝ, x^2 - x + 2,
  degree := 2
}

theorem quadratic_polynomial_P (P : ℝ → ℝ) 
  (h : ∀ x, P (P x) = x^4 - 2 * x^3 + 4 * x^2 - 3 * x + 4) : 
  P 8 = 58 := 
by
  sorry

end quadratic_polynomial_P_l349_349113


namespace prime_count_30_to_50_l349_349403

def is_prime (n: ℕ) : Prop :=
  n > 1 ∧ ∀ m: ℕ, m ∣ n → m = 1 ∨ m = n

def primes_in_range (a b: ℕ) : list ℕ :=
  list.filter is_prime (list.range' a (b - a + 1))

theorem prime_count_30_to_50 : (primes_in_range 30 50).length = 5 :=
by sorry

end prime_count_30_to_50_l349_349403


namespace xiaopangs_score_is_16_l349_349593

-- Define the father's score
def fathers_score : ℕ := 48

-- Define Xiaopang's score in terms of father's score
def xiaopangs_score (fathers_score : ℕ) : ℕ := fathers_score / 2 - 8

-- The theorem to prove that Xiaopang's score is 16
theorem xiaopangs_score_is_16 : xiaopangs_score fathers_score = 16 := 
by
  sorry

end xiaopangs_score_is_16_l349_349593


namespace probability_zero_l349_349072

def s : Set ℕ := {2, 3, 4, 5, 9, 12, 18}
def b : Set ℕ := {4, 5, 6, 7, 8, 11, 14, 19}

def is_congruent (n m a : ℕ) : Prop := n % m = a

def odd (n : ℕ) : Prop := n % 2 = 1
def even (n : ℕ) : Prop := n % 2 = 0

theorem probability_zero :
  (∀ (x ∈ s) (y ∈ b), (is_congruent x 3 2) ∧ (is_congruent y 4 1) ∧
    ((odd x ∧ even y) ∨ (even x ∧ odd y)) ∧ (x + y = 20) → False) :=
by {
  sorry
}

end probability_zero_l349_349072


namespace quadratic_polynomial_value_l349_349100

theorem quadratic_polynomial_value (P : ℝ → ℝ) (hP : ∀ x, P(P(x)) = x^4 - 2*x^3 + 4*x^2 - 3*x + 4) :
  P 8 = 58 :=
sorry

end quadratic_polynomial_value_l349_349100


namespace problem_proof_l349_349068

def f (x : ℝ) : ℝ := 3 * x - 4
def g (x : ℝ) : ℝ := x^2 + 2

theorem problem_proof : f (1 + g 3) = 32 := by
  sorry

end problem_proof_l349_349068


namespace P_P_eq_P_eight_equals_58_l349_349089

open Polynomial

noncomputable def P(x : ℚ) : ℚ := x^2 - x + 2

theorem P_P_eq :
  (P ∘ P)(x) = x^4 - 2 * x^3 + 4 * x^2 - 3 * x + 4 := sorry

theorem P_eight_equals_58 :
  P 8 = 58 := sorry

end P_P_eq_P_eight_equals_58_l349_349089


namespace A_correct_B_correct_complement_intersection_l349_349384

open Set

variable (A : Set ℝ)
variable (B : Set ℝ)

noncomputable def A_def : Set ℝ := {x | (1 / 2 : ℝ) ≤ 2^x ∧ 2^x < 8}
noncomputable def B_def : Set ℝ := {x | (5 : ℝ) / (x + 2) ≥ 1}

theorem A_correct : A_def = {x | -1 ≤ x ∧ x < 3} :=
by sorry

theorem B_correct : B_def = {x | -2 < x ∧ x ≤ 3} :=
by sorry

theorem complement_intersection :
  (compl {x | -1 ≤ x ∧ x < 3} ∩ {x | -2 < x ∧ x ≤ 3}) = {x | -2 < x ∧ x < -1} ∪ {x | x = 3} :=
by sorry

end A_correct_B_correct_complement_intersection_l349_349384


namespace find_sixth_number_l349_349966

theorem find_sixth_number (x : ℝ) 
  (h : 3.6 * 0.48 * 2.50 / (0.12 * 0.09 * x) = 800.0000000000001) : 
  x ≈ 1.25 :=
sorry

end find_sixth_number_l349_349966


namespace rocky_run_miles_l349_349858

theorem rocky_run_miles : 
  let day1 := 4 in
  let day2 := 2 * day1 in
  let day3 := 3 * day2 in
  day1 + day2 + day3 = 36 :=
by
  sorry

end rocky_run_miles_l349_349858


namespace count_complex_numbers_l349_349897

-- Define the function f
def f (z : ℂ) : ℂ := z^2 + complex.I * z + 1

-- Define the conditions
def is_integer_within_bounds (c : ℂ) : Prop :=
  (c.re.abs ≤ 10) ∧ (c.im.abs ≤ 10) ∧ (c.re.denominator = 1) ∧ (c.im.denominator = 1)

-- Define the main theorem
theorem count_complex_numbers :
  (finset.univ.filter (λ z : ℂ, (z.im > 0) ∧ is_integer_within_bounds (f z))).card = 399 :=
begin
  sorry
end

end count_complex_numbers_l349_349897


namespace required_run_rate_is_five_l349_349604

-- Define the conditions.
def initial_overs : ℕ := 10
def initial_run_rate : ℝ := 3.2
def target_runs : ℕ := 282
def remaining_overs : ℕ := 50

-- Calculate the runs scored in the initial overs.
def initial_runs_scored : ℝ := initial_overs * initial_run_rate

-- Calculate the remaining runs needed to reach the target.
def remaining_runs_needed : ℝ := target_runs - initial_runs_scored

-- State the theorem that needs to be proved.
theorem required_run_rate_is_five : 
  remaining_runs_needed / remaining_overs = 5 :=
by 
  skip
-- sorry: The actual proof would go here

end required_run_rate_is_five_l349_349604


namespace max_value_frac_l349_349791

theorem max_value_frac (x y : ℝ) (h1 : 3 ≤ x * y^2 ∧ x * y^2 ≤ 8) (h2 : 4 ≤ x^2 / y ∧ x^2 / y ≤ 9) :
  ∃ k, ∀ z, z = x^3 / y^4 → z ≤ 27 ∧ z = 27 :=
by
  sorry

end max_value_frac_l349_349791


namespace sum_of_coefficients_l349_349066

noncomputable def integral_sin_minus_cos : ℝ :=
  ∫ x in 0..Real.pi, Real.sin x - Real.cos x

theorem sum_of_coefficients (x : ℝ) (a : Fin 9 → ℝ) (h_integral : integral_sin_minus_cos = 2)
  (h_expansion : (1 - integral_sin_minus_cos * x) ^ 8 = ∑ i, a i * x ^ i) :
  (∑ i in Finset.range 8, a i.succ) = 0 :=
  sorry

end sum_of_coefficients_l349_349066


namespace area_of_region_l349_349182

theorem area_of_region :
  let region_eq := ∀ x y : ℝ, x^2 + y^2 + 10*x - 6*y = 14
  in (∃ (r : ℝ), region_eq) → (∃ (area : ℝ), area = 48 * real.pi) := 
sorry

end area_of_region_l349_349182


namespace angle_B_determination_l349_349037

def angle_in_range (B : ℝ) : Prop := 0 < B ∧ B < 180

theorem angle_B_determination (a b : ℝ) (A : ℝ) (h_a : a = 2) (h_b : b = 2 * Real.sqrt 3) (h_A : A = 30) :
  ∃ B : ℝ, (sin (B / 180 * pi) = (b * sin (A / 180 * pi)) / a) ∧ (angle_in_range B) :=
by
  sorry

end angle_B_determination_l349_349037


namespace range_of_a_l349_349500

theorem range_of_a (x y : ℝ) (a : ℝ) :
  (0 < x ∧ x ≤ 2) ∧ (0 < y ∧ y ≤ 2) ∧ (x * y = 2) ∧ (6 - 2 * x - y ≥ a * (2 - x) * (4 - y)) →
  a ≤ 1 :=
by sorry

end range_of_a_l349_349500


namespace length_of_real_axis_l349_349942

-- Given conditions
def is_equilateral_hyperbola (h : ℝ → ℝ → Prop) : Prop :=
  ∃ λ : ℝ, ∀ x y : ℝ, h x y ↔ x^2 - y^2 = λ

def parabola_directrix (d : ℝ → ℝ → Prop) : Prop :=
  ∀ x y : ℝ, d x y ↔ y = -1

variables {λ : ℝ} {x y : ℝ}

-- The equilateral hyperbola and the parabola's directrix
def Σ (x y : ℝ) : Prop := x^2 - y^2 = λ
def parabola_d (x y : ℝ) : Prop := y = -1

-- Intersection points and distance condition
def intersects_with_directrix (h : ℝ → ℝ → Prop) (d : ℝ → ℝ → Prop) : Prop :=
  ∃ x : ℝ, h x (-1) ∧ h (-x) (-1)

def distance_condition (x : ℝ) : Prop :=
  2 * x = 4

-- The final theorem statement
theorem length_of_real_axis (h : ℝ → ℝ → Prop) (d : ℝ → ℝ → Prop)
  (H_hyperbola : is_equilateral_hyperbola h)
  (H_directrix : parabola_directrix d)
  (H_intersection : intersects_with_directrix h d)
  (H_distance : distance_condition x) :
  ∃ λ : ℝ, (h (2 : ℝ) (-1) ∧ h (-2) (-1)) ∧ λ = 3 →
  ∀ x y : ℝ, h x y ↔ x^2 - y^2 = 3 :=
sorry

end length_of_real_axis_l349_349942


namespace problem1_problem2_l349_349214

-- Problem 1: Calculate the given expression
theorem problem1 : (sqrt 16 - (1/3)^(-2) + (π - 3.14)^0 = -4) :=
by
  sorry

-- Problem 2: Solve the inequality
theorem problem2 (x : ℝ) : (x ≤ 5/4) → (x + 1) / 2 ≤ (2 - x) / 6 + 1 :=
by
  sorry

end problem1_problem2_l349_349214


namespace general_term_of_sequence_inequality_on_t_n_l349_349365

variable {n : ℕ}
variable (a : ℕ → ℝ) (S : ℕ → ℝ) (T : ℕ → ℝ)

-- Conditions given in the problem
def S_n (n : ℕ) := 2 - 1 / 2^(n-1)

-- General term of the sequence
theorem general_term_of_sequence :
  (∀ n, S n = S_n n) →
  (∀ n, a n = if n = 1 then S 1 else S n - S (n-1)) →
  (∀ n, a n = 1 / 2^(n-1)) :=
sorry

-- Second part: prove the inequality
theorem inequality_on_t_n :
  (∀ n, S n = S_n n) →
  (∀ n, a n = if n = 1 then S 1 else S n - S (n-1)) →
  (T n = ∑ i in range n, log 2 (a i)) →
  (∀ n, n ≥ 2 → ∑ i in range (n-1), 1 / (T (i + 2)) > -2) :=
sorry

end general_term_of_sequence_inequality_on_t_n_l349_349365


namespace second_candidate_percentage_l349_349619

variable (T P X : ℝ)
variable (condition1 : 0.30 * T = P - 30)
variable (condition2 : (X / 100) * T = P + 15)
variable (P_est : P = 120)

theorem second_candidate_percentage :
  ∃ (X : ℝ), X = 45 :=
by
  use 45
  sorry

end second_candidate_percentage_l349_349619


namespace quadratic_polynomial_P8_l349_349131

theorem quadratic_polynomial_P8 :
  ∃ (a b c : ℝ), 
  (∀ x : ℝ, P x = a * x^2 + b * x + c) ∧ 
  (∀ x : ℝ, P (P x) = x^4 - 2 * x^3 + 4 * x^2 - 3 * x + 4) ∧ 
  P 8 = 58 :=
begin
  sorry
end

end quadratic_polynomial_P8_l349_349131


namespace negation_of_implication_l349_349608

theorem negation_of_implication (x : ℝ) :
  (¬ (x = 0 ∨ x = 1) → x^2 - x ≠ 0) ↔ (x ≠ 0 ∧ x ≠ 1 → x^2 - x ≠ 0) :=
by sorry

end negation_of_implication_l349_349608


namespace quadratic_roots_relation_l349_349994

variable (a b c X1 X2 : ℝ)

theorem quadratic_roots_relation (h : a ≠ 0) : 
  (X1 + X2 = -b / a) ∧ (X1 * X2 = c / a) :=
sorry

end quadratic_roots_relation_l349_349994


namespace hexagon_perimeter_l349_349251

theorem hexagon_perimeter (s : ℝ) (h_area : s ^ 2 * (3 * Real.sqrt 3 / 2) = 54 * Real.sqrt 3) :
  6 * s = 36 :=
by
  sorry

end hexagon_perimeter_l349_349251


namespace perfect_family_cardinality_bound_l349_349533

variable (U : Finset α) (𝓕 : Finset (Finset α))

-- Define the condition for a family of sets to be "perfect".
def is_perfect_family (𝓕 : Finset (Finset α)) : Prop :=
  ∀ X₁ X₂ X₃ ∈ 𝓕, (X₁ \ X₂).nonempty → (X₃ \ (X₂ ∩ X₁)).empty

-- The main theorem statement.
theorem perfect_family_cardinality_bound {α : Type*} [DecidableEq α] (U : Finset α) (𝓕 : Finset (Finset α))
  (h : is_perfect_family 𝓕) : 𝓕.card ≤ U.card + 1 :=
sorry

end perfect_family_cardinality_bound_l349_349533


namespace maximum_value_expression_l349_349507

theorem maximum_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x + y + z)^2 / (x^2 + y^2 + z^2) ≤ 3 :=
by
  sorry

end maximum_value_expression_l349_349507


namespace region_area_is_four_l349_349307

-- Definitions matching the conditions
def r_sec_theta (θ : ℝ) : ℝ := 2 / (Real.cos θ)
def r_csc_theta (θ : ℝ) : ℝ := 2 / (Real.sin θ)

-- The region bounded by the graphs and the axes form a square from origin to (2, 2)

-- Lean statement for proving the area of the described region is 4
theorem region_area_is_four : 
  let r_sec_theta (θ : ℝ) := 2 / (Real.cos θ),
      r_csc_theta (θ : ℝ) := 2 / (Real.sin θ) in
  let vertices := [(0, 0), (2, 0), (2, 2), (0, 2)] in
  let area := (2 - 0) * (2 - 0) in
  area = 4 :=
by sorry

end region_area_is_four_l349_349307


namespace BacteriaUreaPhenolRed_l349_349196

-- Define the conditions as predicates
def phenol_red_indicator_added : Prop := true
def urea_as_only_nitrogen_source : Prop := true
def bacteria_cultured : Prop := true
def decomposes_urea (bacteria : Prop) : Prop := bacteria → urease_formed
def urease_formed : Prop := true
def ammonia_formed (urease_formed : Prop) : Prop := urease_formed → alkaline_environment
def alkaline_environment : Prop := true
def phenol_red_turns_red (alkaline_environment : Prop) : Prop := alkaline_environment → color_red

-- Question: The color of the indicator
def color_of_indicator : Prop := color_red

-- Proof statement to be confirmed
theorem BacteriaUreaPhenolRed (h1 : phenol_red_indicator_added) 
                               (h2 : urea_as_only_nitrogen_source) 
                               (h3 : bacteria_cultured) : 
                               decomposes_urea bacteria_cultured → 
                               ammonia_formed urease_formed → 
                               phenol_red_turns_red alkaline_environment → 
                               color_of_indicator :=
by 
  -- skips the proof
  sorry


end BacteriaUreaPhenolRed_l349_349196


namespace region_area_l349_349308

theorem region_area :
  let side_length := 2 in
  side_length * side_length = 4 :=
by
  -- The proof can be filled in here
  sorry

end region_area_l349_349308


namespace binom_10_2_eq_45_l349_349708

theorem binom_10_2_eq_45 :
  binom 10 2 = 45 := by
  sorry

end binom_10_2_eq_45_l349_349708


namespace prime_count_30_to_50_l349_349398

def is_prime (n: ℕ) : Prop :=
  n > 1 ∧ ∀ m: ℕ, m ∣ n → m = 1 ∨ m = n

def primes_in_range (a b: ℕ) : list ℕ :=
  list.filter is_prime (list.range' a (b - a + 1))

theorem prime_count_30_to_50 : (primes_in_range 30 50).length = 5 :=
by sorry

end prime_count_30_to_50_l349_349398


namespace compare_bases_l349_349675

theorem compare_bases :
  let n1 := 8 * 9 + 5,
      n2 := 2 * 6^2 + 1 * 6,
      n3 := 1 * 4^3,
      n4 := 1 * 2^6 - 1
  in n2 > n1 ∧ n1 > n3 ∧ n3 > n4 :=
by
  let n1 := 8 * 9 + 5
  let n2 := 2 * 6^2 + 1 * 6
  let n3 := 1 * 4^3
  let n4 := 1 * 2^6 - 1
  sorry

end compare_bases_l349_349675


namespace percentage_subtracted_l349_349530

theorem percentage_subtracted (a : ℝ) (p : ℝ) (h : (1 - p / 100) * a = 0.97 * a) : p = 3 :=
by
  sorry

end percentage_subtracted_l349_349530


namespace bisector_position_varies_l349_349215

theorem bisector_position_varies 
(AB : Line) (O : Point) (circ : Circle) (C D E : Point)
(h1 : diameter AB circ)
(h2 : center O circ)
(h3 : on_circle C circ)
(h4 : on_circle D circ)
(h5 : on_circle E circ)
(h6 : diametrically_opposite A E circ)
(h7 : ∠ C D E = 60°):
   ∃ P : Point, on_circle P circ ∧ varies (λ C, bisector_intersects P (∠ O C D)) :=
sorry

end bisector_position_varies_l349_349215


namespace num_different_products_l349_349527

theorem num_different_products :
  let nums := {0, 2, 3, 4, 6, 12}
  let products := {x * y | x in nums, y in nums, x ≠ y}
  |products| = 9 :=
by
  sorry

end num_different_products_l349_349527


namespace problem_statement_l349_349832

   def f (a : ℤ) : ℤ := a - 2
   def F (a b : ℤ) : ℤ := b^2 + a

   theorem problem_statement : F 3 (f 4) = 7 := by
     sorry
   
end problem_statement_l349_349832


namespace infimum_of_f_eq_neg_one_l349_349748

noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 1 then -3 * x ^ 2 + 2
  else if -1 ≤ x ∧ x < 0 then -3 * (-x) ^ 2 + 2
  else -3 * (x % 2) ^ 2 + 2  -- handling the periodicity

theorem infimum_of_f_eq_neg_one :
  ∀ M, (∀ x, f(x) ≥ M) → M ≤ -1 :=
begin
  intro M,
  intro hM,
  have h1 : f(0) = 2,
  { simp [f], -- Calculate f(0) directly. },
  have h2 : f(-1) = -1,
  { simp [f], -- Calculate f(-1) directly. },
  have h3 : f(1) = -1,
  { simp [f], -- Calculate f(1) directly. },
  -- Infimum check
  have h : M ≤ -1,
  { by_contra h,
    obtain ⟨eps, heps⟩ := not_le.mp h,
    specialize hM 1,
    linarith [hM]
  },
  exact h,
sorry -- end the proof here
end

end infimum_of_f_eq_neg_one_l349_349748


namespace max_subsets_five_elements_l349_349988

open Finset

noncomputable def max_subsets (A : Finset ℕ) (k : ℕ) (n : ℕ) : ℕ :=
  let subsets := (powersetLen k A).filter (λ s, 
    ∀ (s1 s2 : Finset ℕ), s1 ∈ subsets → s2 ∈ subsets → s1 ≠ s2 → (s1 ∩ s2).card = 1) 
  subsets.card

theorem max_subsets_five_elements :
  let A := finset.range 21 in
  max_subsets A 5 20 = 16 :=
  sorry

end max_subsets_five_elements_l349_349988


namespace quadrilateral_area_points_l349_349761

theorem quadrilateral_area_points (A B C D : Point) (hConvex : ConvexQuadrilateral A B C D)
(hArea : Area (Quadrilateral A B C D) = 1) :
  ∃ P Q R S : Point, 
    (OnSideOrInside P (Quadrilateral A B C D) ∧ OnSideOrInside Q (Quadrilateral A B C D) ∧ 
     OnSideOrInside R (Quadrilateral A B C D) ∧ OnSideOrInside S (Quadrilateral A B C D)) ∧ 
    (Area (Triangle P Q R) > 1/4) ∧ 
    (Area (Triangle P Q S) > 1/4) ∧ 
    (Area (Triangle P R S) > 1/4) ∧ 
    (Area (Triangle Q R S) > 1/4) :=
sorry

end quadrilateral_area_points_l349_349761


namespace function_is_odd_and_monotonically_increasing_on_pos_l349_349655

-- Define odd function
def odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f (x)

-- Define monotonically increasing on (0, +∞)
def monotonically_increasing_on_pos (f : ℝ → ℝ) := ∀ x y : ℝ, (0 < x ∧ x < y) → f (x) < f (y)

-- Define the function in question
def f (x : ℝ) := x * |x|

-- Prove the function is odd and monotonically increasing on (0, +∞)
theorem function_is_odd_and_monotonically_increasing_on_pos :
  odd_function f ∧ monotonically_increasing_on_pos f :=
by
  sorry

end function_is_odd_and_monotonically_increasing_on_pos_l349_349655


namespace sum_fourth_and_sixth_l349_349462

theorem sum_fourth_and_sixth :
  let a₁ := 1 in
  let a := λ n, if n = 1 then a₁ else (λ (k:ℕ), (a₁ *  
      (∏ i in Finset.range (k-1) + 1, (λ j, (1 + (1 / ↑j))^2))) ((n-1) + 1)) 
  in a 4 + a 6 = 724 / 225 :=
by
  sorry

end sum_fourth_and_sixth_l349_349462


namespace james_weekly_hours_l349_349475

def james_meditation_total : ℕ :=
  let weekly_minutes := (30 * 2 * 6) + (30 * 2 * 2) -- 1 hour/day for 6 days + 2 hours on Sunday
  weekly_minutes / 60

def james_yoga_total : ℕ :=
  let weekly_minutes := (45 * 2) -- 45 minutes on Monday and Friday
  weekly_minutes / 60

def james_bikeride_total : ℕ :=
  let weekly_minutes := 90
  weekly_minutes / 60

def james_dance_total : ℕ :=
  2 -- 2 hours on Saturday

def james_total_activity_hours : ℕ :=
  james_meditation_total + james_yoga_total + james_bikeride_total + james_dance_total

theorem james_weekly_hours : james_total_activity_hours = 13 := by
  sorry

end james_weekly_hours_l349_349475


namespace JulioHasMoreSoda_l349_349880

-- Define the number of bottles each person has
def JulioOrangeBottles : ℕ := 4
def JulioGrapeBottles : ℕ := 7
def MateoOrangeBottles : ℕ := 1
def MateoGrapeBottles : ℕ := 3

-- Define the volume of each bottle in liters
def BottleVolume : ℕ := 2

-- Define the total liters of soda each person has
def JulioTotalLiters : ℕ := JulioOrangeBottles * BottleVolume + JulioGrapeBottles * BottleVolume
def MateoTotalLiters : ℕ := MateoOrangeBottles * BottleVolume + MateoGrapeBottles * BottleVolume

-- Prove the difference in total liters of soda between Julio and Mateo
theorem JulioHasMoreSoda : JulioTotalLiters - MateoTotalLiters = 14 := by
  sorry

end JulioHasMoreSoda_l349_349880


namespace min_value_4x5_5x_neg4_l349_349498

def min_value (x : ℝ) : ℝ := 4 * x^5 + 5 * x^(-4)

theorem min_value_4x5_5x_neg4 (hx : 0 < x) : ∃ (x_min : ℝ), min_value x ≥ 9 ∧ (min_value x = 9 ↔ x = 1) :=
sorry

end min_value_4x5_5x_neg4_l349_349498


namespace shelves_needed_l349_349665

-- Definitions from conditions
def total_pots : ℕ := 60
def pots_per_set : ℕ := 5
def sets_per_shelf : ℕ := 3

-- Theorem statement
theorem shelves_needed :
  (total_pots / (pots_per_set * sets_per_shelf)) = 4 :=
by
  calc total_pots / (pots_per_set * sets_per_shelf) = 60 / 15 : by sorry
  ... = 4 : by sorry

end shelves_needed_l349_349665


namespace divide_inequality_by_negative_l349_349293

theorem divide_inequality_by_negative {x : ℝ} (h : -6 * x > 2) : x < -1 / 3 :=
by sorry

end divide_inequality_by_negative_l349_349293


namespace apple_multiple_l349_349886

theorem apple_multiple (K Ka : ℕ) (M : ℕ) 
  (h1 : K + Ka = 340)
  (h2 : Ka = M * K + 10)
  (h3 : Ka = 274) : 
  M = 4 := 
by
  sorry

end apple_multiple_l349_349886


namespace prob_all_three_co_captains_are_selected_l349_349968

theorem prob_all_three_co_captains_are_selected :
  let P6 := 1 / (Nat.choose 6 3),
      P8 := 1 / (Nat.choose 8 3),
      P9 := 1 / (Nat.choose 9 3),
      P10 := 1 / (Nat.choose 10 3),
      P_total := 1 / 4 * (P6 + P8 + P9 + P10)
  in P_total = 53 / 3360 := by
  sorry

end prob_all_three_co_captains_are_selected_l349_349968


namespace binom_10_2_eq_45_l349_349704

theorem binom_10_2_eq_45 :
  binom 10 2 = 45 := by
  sorry

end binom_10_2_eq_45_l349_349704


namespace andrew_bought_6_kg_of_grapes_l349_349657

def rate_grapes := 74
def rate_mangoes := 59
def kg_mangoes := 9
def total_paid := 975

noncomputable def number_of_kg_grapes := 6

theorem andrew_bought_6_kg_of_grapes :
  ∃ G : ℕ, (rate_grapes * G + rate_mangoes * kg_mangoes = total_paid) ∧ G = number_of_kg_grapes := 
by
  sorry

end andrew_bought_6_kg_of_grapes_l349_349657


namespace JulioHasMoreSoda_l349_349881

-- Define the number of bottles each person has
def JulioOrangeBottles : ℕ := 4
def JulioGrapeBottles : ℕ := 7
def MateoOrangeBottles : ℕ := 1
def MateoGrapeBottles : ℕ := 3

-- Define the volume of each bottle in liters
def BottleVolume : ℕ := 2

-- Define the total liters of soda each person has
def JulioTotalLiters : ℕ := JulioOrangeBottles * BottleVolume + JulioGrapeBottles * BottleVolume
def MateoTotalLiters : ℕ := MateoOrangeBottles * BottleVolume + MateoGrapeBottles * BottleVolume

-- Prove the difference in total liters of soda between Julio and Mateo
theorem JulioHasMoreSoda : JulioTotalLiters - MateoTotalLiters = 14 := by
  sorry

end JulioHasMoreSoda_l349_349881


namespace bridge_length_l349_349210

theorem bridge_length
  (train_length : ℕ) (train_speed_kph : ℕ) (time_seconds : ℕ)
  (h_train_length : train_length = 140)
  (h_train_speed : train_speed_kph = 45)
  (h_time_seconds : time_seconds = 30) :
  let train_speed_mps : ℝ := (train_speed_kph * 1000) / 3600
  let total_distance : ℝ := train_speed_mps * time_seconds
  let bridge_length : ℝ := total_distance - train_length
  in bridge_length = 235 :=
by
  sorry

end bridge_length_l349_349210


namespace custom_operator_example_l349_349830

def custom_operator (a b : ℕ) : ℕ := a^2 - 2*a*b + b^2

theorem custom_operator_example : custom_operator 5 3 = 4 := by
  sorry

end custom_operator_example_l349_349830


namespace min_max_expression_l349_349063

variable (a b c d e : ℝ)

def expression (a b c d e : ℝ) : ℝ :=
  5 * (a^3 + b^3 + c^3 + d^3 + e^3) - (a^4 + b^4 + c^4 + d^4 + e^4)

theorem min_max_expression :
  a + b + c + d + e = 10 →
  a^2 + b^2 + c^2 + d^2 + e^2 = 20 →
  expression a b c d e = 120 := by
  sorry

end min_max_expression_l349_349063


namespace john_total_payment_l349_349878

-- Definitions of the conditions
def yearly_cost_first_8_years : ℕ := 10000
def yearly_cost_9_to_18_years : ℕ := 2 * yearly_cost_first_8_years
def university_tuition : ℕ := 250000
def total_cost := (8 * yearly_cost_first_8_years) + (10 * yearly_cost_9_to_18_years) + university_tuition

-- John pays half of the total cost
def johns_total_cost := total_cost / 2

-- Theorem stating the total cost John pays
theorem john_total_payment : johns_total_cost = 265000 := by
  sorry

end john_total_payment_l349_349878


namespace P_eight_value_l349_349092

def quadratic_polynomial (P : ℤ → ℤ) : Prop :=
  ∃ a b c : ℤ, ∀ x : ℤ, P(x) = a * x^2 + b * x + c

theorem P_eight_value (P : ℤ → ℤ)
  (H : ∀ x : ℤ, P(P(x)) = x^4 - 2 * x^3 + 4 * x^2 - 3 * x + 4)
  (H_quad : quadratic_polynomial P) :
  P 8 = 58 :=
sorry

end P_eight_value_l349_349092


namespace number_of_proper_subsets_l349_349778

-- Define the set A
def A : Set ℤ := {x | abs (x - 3) < Real.pi}

-- Define the set B
def B : Set ℤ := {x | x^2 - 11*x + 5 < 0}

-- Define the set C
def C : Set ℤ := {x | 2*x^2 - 11*x + 10 ≥ abs (3*x - 2)}

-- Complement of C
def C_complement : Set ℤ := {x | ¬ (2*x^2 - 11*x + 10 ≥ abs (3*x - 2))}

-- Define the final set we are interested in
def final_set : Set ℤ := {x | x ∈ A ∧ x ∈ B ∧ x ∈ C_complement}

-- Proof statement in Lean
theorem number_of_proper_subsets :
  Finset.card (Set.toFinset (final_set)) = 3 → Finset.card (Set.toFinset (setOf (λ s, s ⊆ final_set ∧ s ≠ final_set)) = 7) :=
sorry

end number_of_proper_subsets_l349_349778


namespace jordan_7_miles_time_l349_349479

noncomputable section

-- Given conditions
def steve_time_5_miles : ℝ := 40
def jordan_time_3_miles := (2/3) * steve_time_5_miles

/-- 
  Prove that the time it takes Jordan to run 7 miles is 185/3 minutes 
  given Jordan ran 3 miles in two-thirds the time it took Steve to run 
  5 miles and it took Steve 40 minutes to run 5 miles.
-/
theorem jordan_7_miles_time : 
  let jordan_pace := jordan_time_3_miles / 3
  in 7 * jordan_pace = 185 / 3 := 
by 
  let jordan_pace := jordan_time_3_miles / 3
  calc 7 * jordan_pace =
        7 * ((2/3) * steve_time_5_miles / 3) : by sorry
      ... = 185 / 3 : by sorry


end jordan_7_miles_time_l349_349479


namespace binom_10_2_eq_45_l349_349688

theorem binom_10_2_eq_45 : Nat.binomial 10 2 = 45 := by
  sorry

end binom_10_2_eq_45_l349_349688


namespace prove_sum_and_count_sequences_l349_349489

def x_seq := ℕ → ℕ

def conditions (x : x_seq) : Prop :=
  (∀ n, x n > 0) ∧ 
  (∀ n, n > 0 → x n ≥ x (n - 1)) ∧ 
  (x 0 = 1) ∧ 
  (({i : ℕ | i > 0 ∧ i ≤ 2017}).card = 25)

theorem prove_sum_and_count_sequences (x : x_seq) 
  (h : conditions x) : 
  ∑ i in finset.range 2016, (x (i + 2) - x i) * x (i + 2) ≥ 623 
  ∧ 
  finset.card {s : finset (fin 2018) 
               | conditions (λ i, if i ∈ s then x i else 0) 
               ∧ ∑ i in s, (x (i + 2) - x i) * x (i + 2) = 623 }
  = nat.choose 1992 23 := 
sorry

end prove_sum_and_count_sequences_l349_349489


namespace simplify_expression_l349_349181

theorem simplify_expression : (3^4 + 3^4) / (3^(-4) + 3^(-4)) = 6561 := by
  sorry

end simplify_expression_l349_349181


namespace loss_percent_l349_349603

theorem loss_percent (CP SP : ℝ) (h₁ : CP = 600) (h₂ : SP = 300) : 
  (CP - SP) / CP * 100 = 50 :=
by
  rw [h₁, h₂]
  norm_num

end loss_percent_l349_349603


namespace slope_OM_line_l_exists_l349_349770

-- Define the ellipse G
def ellipse_G (x y : ℝ) := (x^2 / 2) + y^2 = 1

-- Define the left focus F_1
def left_focus := (-sqrt 2, 0)

-- Define the line passing through F_1 with a slope of 1
def line_l (x y : ℝ) := y = x + sqrt 2

-- Midpoint of A and B when line_l intersects ellipse_G
def midpoint_AB : ℝ × ℝ := (-2 / 3, sqrt 2 / 3)

-- Prove the slope of line OM when M is midpoint_AB
theorem slope_OM : (midpoint_AB.snd / midpoint_AB.fst) = sqrt 2 / 2 := sorry

-- Define existence of line l such that |AM|^2 = |CM| * |DM|
def line_l_condition : Prop := ∃ (k : ℝ ≠ 0), 
  (∀ x y, y = k * (x + sqrt 2) → 
   let midpoint_AB := (-2 * k^2 / (2 * k^2 + 1), k * sqrt 2 / (2 * k^2 + 1)),
       CD_slope := -1 / (2 * k),
       |AM| := sqrt ((midpoint_AB.1 - 0)^2 + (midpoint_AB.2 - sqrt 2)^2),
       line_CD (x : ℝ) := -1 / (2 * k) * x in
   |AM|^2 = ((cd : ℝ × ℝ) in line_CD).2 * cd.1)

theorem line_l_exists : line_l_condition := sorry

end slope_OM_line_l_exists_l349_349770


namespace line_equation_M_l349_349839

theorem line_equation_M (x y : ℝ) : 
  (∃ c1 m1 : ℝ, m1 = 2 / 3 ∧ c1 = 4 ∧ 
  (∃ m2 c2 : ℝ, m2 = 2 * m1 ∧ c2 = (1 / 2) * c1 ∧ y = m2 * x + c2)) → 
  y = (4 / 3) * x + 2 := 
sorry

end line_equation_M_l349_349839


namespace sum_of_roots_ln_abs_l349_349999

theorem sum_of_roots_ln_abs (m : ℝ) (x1 x2 : ℝ) 
  (h1 : ln |x1 - 2| = m) (h2 : ln |x2 - 2| = m) : x1 + x2 = 4 := 
sorry

end sum_of_roots_ln_abs_l349_349999


namespace approx_equal_e_l349_349572
noncomputable def a : ℝ := 69.28
noncomputable def b : ℝ := 0.004
noncomputable def c : ℝ := 0.03
noncomputable def d : ℝ := a * b
noncomputable def e : ℝ := d / c

theorem approx_equal_e : abs (e - 9.24) < 0.01 :=
by
  sorry

end approx_equal_e_l349_349572


namespace henry_total_books_l349_349814

variable (initial_books novels science_books cookbooks philosophy_books history_books self_help_books borrowed lent returned_books : ℕ)
variable (donated_novels donated_science_books donated_cookbooks donated_philosophy_books donated_history_books donated_self_help_books : ℕ)
variable (borrowed_remaining lent_remaining recycled_books new_books1 new_books2 new_books3 : ℕ)

theorem henry_total_books :
  initial_books = 250 →
  novels = 75 →
  science_books = 55 →
  cookbooks = 40 →
  philosophy_books = 35 →
  history_books = 25 →
  self_help_books = 20 →
  borrowed = 8 →
  lent = 6 →
  returned_books = 3 →
  donated_novels = 48 →
  donated_science_books = 31 →
  donated_cookbooks = 20 →
  donated_philosophy_books = 12 →
  donated_history_books = 6 →
  donated_self_help_books = 19 →
  recycled_books = 5 →
  new_books1 = 8 →
  new_books2 = 14 →
  new_books3 = 10 →
  borrowed_remaining = borrowed - 4 →
  lent_remaining = lent - returned_books →
  let books_after_donating_and_recycling := initial_books - (donated_novels + donated_science_books + donated_cookbooks + donated_philosophy_books + donated_history_books + donated_self_help_books) - recycled_books in
  let books_after_acquiring := books_after_donating_and_recycling + new_books1 + new_books2 + new_books3 in
  let total_books := books_after_acquiring + borrowed_remaining - lent_remaining in
  total_books = 152 :=
sorry

end henry_total_books_l349_349814


namespace sum_of_g1_l349_349065

-- Define the main conditions
variable {g : ℝ → ℝ}
variable (h_nonconst : ∀ a b : ℝ, a ≠ b → g a ≠ g b)
axiom main_condition : ∀ x : ℝ, x ≠ 0 → g (x - 1) + g x + g (x + 1) = (g x) ^ 2 / (2025 * x)

-- Define the goal
theorem sum_of_g1 :
  g 1 = 6075 :=
sorry

end sum_of_g1_l349_349065


namespace james_potatoes_l349_349476

-- Define the given conditions
def cost_per_bag := 5 -- dollars
def weight_per_bag := 20 -- pounds
def total_cost := 15 -- dollars
def weight_per_person := 1.5 -- pounds

-- Define the problem statement
theorem james_potatoes : 
  (total_cost / cost_per_bag) * weight_per_bag / weight_per_person = 40 := 
by
  -- The proof would be filled in here
  sorry

end james_potatoes_l349_349476


namespace sequence_pattern_l349_349074

theorem sequence_pattern (n : ℕ) (h : 0 < n) :
  1 + 5 + 15 + ... + (1 / 24) * n * (n + 1) * (n + 2) * (n + 3) = 
  (1 / 120) * n * (n + 1) * (n + 2) * (n + 3) * (n + 4) :=
sorry

end sequence_pattern_l349_349074


namespace probability_three_heads_in_a_row_l349_349993

theorem probability_three_heads_in_a_row (h : ℝ) (p_head : h = 1/2) (ind_flips : ∀ (n : ℕ), true) : 
  (1/2 * 1/2 * 1/2 = 1/8) :=
by
  sorry

end probability_three_heads_in_a_row_l349_349993


namespace biggest_number_in_ratio_is_l349_349614

noncomputable def ratio_problem := 
  let A := 2 * x
  let B := 3 * x
  let C := 4 * x
  let D := 5 * x
  let sum := A + B + C + D
  let total := 1344
  let biggest := 480
  (2 * x + 3 * x + 4 * x + 5 * x = 1344) → 5 * (1344 / 14) = 480

theorem biggest_number_in_ratio_is
  (x : ℕ)
  (A B C D : ℕ)
  (sum : ℕ := A + B + C + D)
  (ratio_condition : A = 2 * x ∧ B = 3 * x ∧ C = 4 * x ∧ D = 5 * x)
  (sum_condition : sum = 1344) :
  D = 480 := 
  by
    rw [ratio_condition.1, ratio_condition.2.1, ratio_condition.2.2.1, ratio_condition.2.2.2]
    rw sum_condition
    have : x = 1344 / 14 := by sorry
    show 5 * x = 480 from by sorry

end biggest_number_in_ratio_is_l349_349614


namespace units_digit_specified_expression_l349_349193

theorem units_digit_specified_expression :
  let numerator := (1 * 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9 * 10 * 11)
  let denominator := 8000
  let product := numerator * 20
  (∃ d, product / denominator = d ∧ (d % 10 = 6)) :=
by
  sorry

end units_digit_specified_expression_l349_349193


namespace division_decimal_l349_349985

theorem division_decimal (x : ℝ) (h : x = 0.3333): 12 / x = 36 :=
  by
    sorry

end division_decimal_l349_349985


namespace polynomial_value_at_8_l349_349112

noncomputable def P : ℝ → ℝ := λ x, x^2 - x + 2

theorem polynomial_value_at_8 :
  (P (P x) = x^4 - 2*x^3 + 4*x^2 - 3*x + 4) ∧ (P = λ x, x^2 - x + 2) → P 8 = 58 :=
by
  sorry

end polynomial_value_at_8_l349_349112


namespace comic_story_books_proportion_l349_349036

theorem comic_story_books_proportion (x : ℕ) :
  let initial_comic_books := 140
  let initial_story_books := 100
  let borrowed_books_per_day := 4
  let comic_books_after_x_days := initial_comic_books - borrowed_books_per_day * x
  let story_books_after_x_days := initial_story_books - borrowed_books_per_day * x
  (comic_books_after_x_days = 3 * story_books_after_x_days) -> x = 20 :=
by
  sorry

end comic_story_books_proportion_l349_349036


namespace y2_over_x2_plus_x2_over_y2_eq_9_over_4_l349_349532

theorem y2_over_x2_plus_x2_over_y2_eq_9_over_4 (x y : ℝ) 
  (h : (1 / x) - (1 / (2 * y)) = (1 / (2 * x + y))) : 
  (y^2 / x^2) + (x^2 / y^2) = 9 / 4 := 
by 
  sorry

end y2_over_x2_plus_x2_over_y2_eq_9_over_4_l349_349532


namespace max_bc_value_l349_349909

noncomputable def complex_inequality (a b c : ℂ) : Prop :=
  ∀ z : ℂ, abs z ≤ 1 → abs (a * z^2 + b * z + c) ≤ 1

theorem max_bc_value (a b c : ℂ) (h : complex_inequality a b c) : abs (b * c) ≤ (3 * real.sqrt 3) / 16 :=
sorry

end max_bc_value_l349_349909


namespace betty_total_oranges_l349_349664

-- Definitions for the given conditions
def boxes : ℝ := 3.0
def oranges_per_box : ℝ := 24

-- Theorem statement to prove the correct answer to the problem
theorem betty_total_oranges : boxes * oranges_per_box = 72 := by
  sorry

end betty_total_oranges_l349_349664


namespace inversely_proportional_y_value_l349_349158

theorem inversely_proportional_y_value (x y k : ℝ)
  (h1 : ∀ x y : ℝ, x * y = k)
  (h2 : ∃ y : ℝ, x = 3 * y ∧ x + y = 36 ∧ x * y = k)
  (h3 : x = -9) : y = -27 := 
by
  sorry

end inversely_proportional_y_value_l349_349158


namespace binom_10_2_eq_45_l349_349685

theorem binom_10_2_eq_45 : Nat.binomial 10 2 = 45 := by
  sorry

end binom_10_2_eq_45_l349_349685


namespace expression_range_l349_349490

theorem expression_range (a b c d e : ℝ) (h₁ : 0 ≤ a) (h₂ : a ≤ 1) (h₃ : 0 ≤ b) (h₄ : b ≤ 1) 
    (h₅ : 0 ≤ c) (h₆ : c ≤ 1) (h₇ : 0 ≤ d) (h₈ : d ≤ 1) (h₉ : 0 ≤ e) (h₁₀ : e ≤ 1) :
    ∃ x : ℝ, 
      x = sqrt (a^2 + (1 - b)^2) + sqrt (b^2 + (1 - c)^2) + sqrt (c^2 + (1 - d)^2) + 
             sqrt (d^2 + (1 - e)^2) + sqrt (e^2 + (1 - a)^2) ∧
      x ∈ set.Icc (↑(5) / real.sqrt (2)) 5 :=
begin
  -- Proof here
  sorry
end

end expression_range_l349_349490


namespace uncle_wang_withdraw_amount_l349_349566

noncomputable def total_amount (principal : ℕ) (rate : ℚ) (time : ℕ) : ℚ :=
  principal + principal * rate * time

theorem uncle_wang_withdraw_amount :
  total_amount 100000 (315/10000) 2 = 106300 := by
  sorry

end uncle_wang_withdraw_amount_l349_349566


namespace smallest_positive_period_solve_for_a_l349_349812

-- Vectors defined based on given conditions
def m (a x : ℝ) : ℝ × ℝ := (a + 1, Real.sin x)
def n (x : ℝ) : ℝ × ℝ := (1, 4 * Real.cos (x + Real.pi / 6))

-- Function g(x) defined as the dot product of m and n
def g (a x : ℝ) : ℝ := (m a x).1 * (n x).1 + (m a x).2 * (n x).2

-- The problem statements to be proved
theorem smallest_positive_period (a x : ℝ) : ∀ x : ℝ, g a (x + Real.pi) = g a x :=
by
  sorry

theorem solve_for_a (a : ℝ) (h : ∀ x : ℝ, x ∈ Ico 0 (Real.pi / 3) → a + 2 * Real.sin (2 * x + Real.pi / 6) ≤ a + 2 ∧ a + 2 * Real.sin (2 * x + Real.pi / 6) ≥ a + 1 ∧ a + 3 = 7) : a = 2 :=
by
  sorry

end smallest_positive_period_solve_for_a_l349_349812


namespace total_number_of_students_is_40_l349_349455

variables (S R : ℕ)

-- Conditions
def students_not_borrowed_any_books := 2
def students_borrowed_1_book := 12
def students_borrowed_2_books := 10
def average_books_per_student := 2

-- Definition of total books borrowed
def total_books_borrowed := (0 * students_not_borrowed_any_books) + (1 * students_borrowed_1_book) + (2 * students_borrowed_2_books) + (3 * R)

-- Expression for total number of students
def total_students := students_not_borrowed_any_books + students_borrowed_1_book + students_borrowed_2_books + R

-- Mathematical statement to prove
theorem total_number_of_students_is_40 (h : total_books_borrowed R / total_students R = average_books_per_student) : total_students R = 40 :=
sorry

end total_number_of_students_is_40_l349_349455


namespace count_primes_between_30_and_50_l349_349423

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between_30_and_50 : List ℕ :=
  [31, 37, 41, 43, 47]

theorem count_primes_between_30_and_50 : 
  (primes_between_30_and_50.filter is_prime).length = 5 :=
by
  sorry

end count_primes_between_30_and_50_l349_349423


namespace expand_and_simplify_l349_349302

theorem expand_and_simplify (x : ℝ) (hx : x ≠ 0) :
  (3 / 7) * (14 / x^3 + 15 * x - 6 * x^5) = (6 / x^3) + (45 * x / 7) - (18 * x^5 / 7) :=
by
  sorry

end expand_and_simplify_l349_349302


namespace only_identity_satisfies_condition_l349_349315

theorem only_identity_satisfies_condition (f : ℕ → ℕ) (h : ∀ n : ℕ, f(f(n)) < f(n+1)) : ∀ n : ℕ, f(n) = n := 
by {
  sorry
}

end only_identity_satisfies_condition_l349_349315


namespace erased_odd_number_sum_l349_349201

theorem erased_odd_number_sum {n : ℕ} (H1 : (∑ i in finset.range n, (2 * i + 1)) - k = 2008) (H2 :  n*n - 2008 = k) : 
   k = 17 :=
by
  sorry

end erased_odd_number_sum_l349_349201


namespace problem_solution_l349_349169

def is_invertible_modulo_9 (a : ℕ) : Prop := Int.gcd a 9 = 1

theorem problem_solution (a b c d : ℕ) 
  (h1 : a < 9) (h2 : b < 9) (h3 : c < 9) (h4 : d < 9)
  (h5 : a ≠ b) (h6 : a ≠ c) (h7 : a ≠ d)
  (h8 : b ≠ c) (h9 : b ≠ d) (h10 : c ≠ d)
  (h11 : is_invertible_modulo_9 a)
  (h12 : is_invertible_modulo_9 b)
  (h13 : is_invertible_modulo_9 c)
  (h14 : is_invertible_modulo_9 d) :
  ((a * b * c + a * b * d + a * c * d + b * c * d) *
   Nat.gcd_a (a * b * c * d) 9) % 9 = 6 :=
by sorry

end problem_solution_l349_349169


namespace directrix_of_given_parabola_l349_349311

-- Define the equation of the parabola
def parabola (x : ℝ) : ℝ := -1 / 4 * x ^ 2

-- Define what it means for a point (x, y) to be on the parabola
def on_parabola (x y : ℝ) : Prop := y = parabola x

-- Define what it means to be the directrix of the parabola
def is_directrix (d : ℝ) : Prop :=
  ∀ (x : ℝ), (parabola x - (d * √((x - 0) ^ 2 + (parabola x - (-d)) ^ 2))) = 0

theorem directrix_of_given_parabola : is_directrix 1 :=
sorry

end directrix_of_given_parabola_l349_349311


namespace find_p_l349_349810

theorem find_p (x y : ℝ) (h : y = 1.15 * x * (1 - p / 100)) : p = 15 :=
sorry

end find_p_l349_349810


namespace smallest_positive_multiple_l349_349190

theorem smallest_positive_multiple (a : ℕ) (h : 17 * a % 53 = 7) : 17 * a = 544 :=
sorry

end smallest_positive_multiple_l349_349190


namespace quarters_value_percentage_l349_349598

theorem quarters_value_percentage (dimes_count quarters_count dimes_value quarters_value : ℕ) (h1 : dimes_count = 75)
    (h2 : quarters_count = 30) (h3 : dimes_value = 10) (h4 : quarters_value = 25) :
    (quarters_count * quarters_value * 100) / (dimes_count * dimes_value + quarters_count * quarters_value) = 50 := 
by
    sorry

end quarters_value_percentage_l349_349598


namespace three_lines_intersect_at_one_point_l349_349347

theorem three_lines_intersect_at_one_point (sq : Square) (lines : Fin 9 → Line) 
  (area_ratio : ∀ i, divides_square_with_area_ratio lines[i] 2 3) : 
  ∃ p : Point, ∃ I J K: Fin 9, I ≠ J ∧ J ≠ K ∧ I ≠ K ∧ 
  (lines I).contains p ∧ (lines J).contains p ∧ (lines K).contains p :=
sorry

end three_lines_intersect_at_one_point_l349_349347


namespace polynomial_value_at_8_l349_349106

noncomputable def P : ℝ → ℝ := λ x, x^2 - x + 2

theorem polynomial_value_at_8 :
  (P (P x) = x^4 - 2*x^3 + 4*x^2 - 3*x + 4) ∧ (P = λ x, x^2 - x + 2) → P 8 = 58 :=
by
  sorry

end polynomial_value_at_8_l349_349106


namespace lower_limit_of_b_l349_349443

theorem lower_limit_of_b (a : ℤ) (b : ℤ) (h₁ : 8 < a ∧ a < 15) (h₂ : ∃ x, x < b ∧ b < 21) (h₃ : (14 : ℚ) / b - (9 : ℚ) / b = 1.55) : b = 4 :=
by
  sorry

end lower_limit_of_b_l349_349443


namespace acute_angles_in_first_quadrant_l349_349197

theorem acute_angles_in_first_quadrant:
  ∀ θ : ℝ, 0 < θ ∧ θ < (real.pi / 2) ↔ 0 < θ ∧ θ < (real.pi / 2) :=
by
  sorry

end acute_angles_in_first_quadrant_l349_349197


namespace evaluate_expression_l349_349728

theorem evaluate_expression : 
  -3 * 5 - (-4 * -2) + (-15 * -3) / 3 = -8 :=
by
  sorry

end evaluate_expression_l349_349728


namespace number_of_primes_between_30_and_50_l349_349420

/-- 
  Prove that there are exactly 5 prime numbers in the range from 30 to 50. 
  These primes are 31, 37, 41, 43, and 47.
-/
theorem number_of_primes_between_30_and_50 : 
  (Finset.filter Nat.Prime (Finset.range 51)).card - 
  (Finset.filter Nat.Prime (Finset.range 31)).card = 5 := 
by 
  sorry

end number_of_primes_between_30_and_50_l349_349420


namespace number_of_primes_between_30_and_50_l349_349435

-- Define what it means to be a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the interval condition
def in_interval (n : ℕ) : Prop :=
  30 ≤ n ∧ n ≤ 50

-- Define the count of prime numbers in the interval
def prime_count_in_interval : ℕ :=
  (List.range' 30 21).countp (λ n, is_prime n)

-- We state that the above count is equal to 5
theorem number_of_primes_between_30_and_50 : prime_count_in_interval = 5 :=
  sorry

end number_of_primes_between_30_and_50_l349_349435


namespace remainder_modulo_9_l349_349166

open Int

theorem remainder_modulo_9 (a b c d : ℕ) (ha : a < 9) (hb : b < 9) (hc : c < 9) (hd : d < 9)
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (hinv : ∀ x ∈ {a, b, c, d}, Nat.gcd x 9 = 1) :
  ((a * b * c + a * b * d + a * c * d + b * c * d) * (a * b * c * d)⁻¹) % 9 = 6 :=
sorry

end remainder_modulo_9_l349_349166


namespace car_owners_without_motorcycles_l349_349454

theorem car_owners_without_motorcycles 
    (total_adults : ℕ) 
    (car_owners : ℕ) 
    (motorcycle_owners : ℕ) 
    (total_with_vehicles : total_adults = 500) 
    (total_car_owners : car_owners = 480) 
    (total_motorcycle_owners : motorcycle_owners = 120) : 
    car_owners - (car_owners + motorcycle_owners - total_adults) = 380 := 
by
    sorry

end car_owners_without_motorcycles_l349_349454


namespace shaded_area_ECODF_l349_349174

theorem shaded_area_ECODF :
  let r := 3
  let OA := 3 * Real.sqrt 3
  let AB := 2 * OA
  let AB_area := r * AB
  let triangle_area := 1 / 2 * r * r
  let sector_area := 1 / 6 * π * r ^ 2
  ECODF_area = AB_area - 2 * triangle_area - 2 * sector_area := 
by
  let AB := 6 * Real.sqrt 3
  let ABFE_area := 18 * Real.sqrt 3
  let triangles_area := 9
  let sectors_area := 9 * π
  let ECODF_area := 18 * Real.sqrt 3 - 9 - 9 * π
  rfl

end shaded_area_ECODF_l349_349174


namespace total_revenue_correct_l349_349634

def pies_sold_each_day : list (ℕ × ℕ) := [
  (1, 9),    -- Monday: 9 pies
  (2, 12),   -- Tuesday: 12 pies
  (3, 18),   -- Wednesday: 18 pies (with 10% discount)
  (4, 14),   -- Thursday: 14 pies
  (5, 16),   -- Friday: 16 pies (with 5% discount)
  (6, 20),   -- Saturday: 20 pies
  (7, 11)    -- Sunday: 11 pies
]

def pie_cost : ℕ := 5

def discount (day: ℕ) : ℕ → ℚ
| 3 := 0.9 -- 10% discount on Wednesday
| 5 := 0.95 -- 5% discount on Friday
| _ := 1   -- No discount on other days

def revenue_per_day : ℕ × ℕ → ℚ :=
  λ (day_pies : ℕ × ℕ), (day_pies.snd * pie_cost : ℕ) * discount day_pies.fst

def total_revenue : ℚ :=
  pies_sold_each_day.map revenue_per_day |>.sum

theorem total_revenue_correct : total_revenue = 487 := by
  -- Proof goes here
  sorry

end total_revenue_correct_l349_349634


namespace find_sum_of_angles_l349_349470

-- Given conditions
def angleP := 34
def angleQ := 76
def angleR := 28

-- Proposition to prove
theorem find_sum_of_angles (x z : ℝ) (h1 : x + z = 138) : x + z = 138 :=
by
  have angleP := 34
  have angleQ := 76
  have angleR := 28
  exact h1

end find_sum_of_angles_l349_349470


namespace min_gb_for_plan_y_to_be_cheaper_l349_349654

theorem min_gb_for_plan_y_to_be_cheaper (g : ℕ) : 20 * g > 3000 + 10 * g → g ≥ 301 := by
  sorry

end min_gb_for_plan_y_to_be_cheaper_l349_349654


namespace volume_of_solid_l349_349028

noncomputable def volume_proof : Real :=
∫ x in -Real.sqrt 3..Real.sqrt 3,
∫ y in -Real.sqrt 3..Real.sqrt 3,
∫ z in 0..1 - (x^2 + y^2) / 4,
1

theorem volume_of_solid :
  volume_proof = 6 :=
sorry

end volume_of_solid_l349_349028


namespace sum_AiK_lt_S_l349_349718

-- Define the problem conditions and the goal to be proved
variable (n : ℕ)
variable (P Q : Point)
variable (A : Fin n → Point)
variable (K : Point)
variable (S : ℝ)
variable (A_iK : Fin n → ℝ)

-- Conditions
def midpoint (P Q K : Point) : Prop := dist P K = dist K Q

def extended (A K B_i : Point) : Prop := dist A K = dist K B_i

def bisection (A_i K P Q : Point) : Prop := 
  ∃ B_i : Point, midpoint P Q K ∧ extended A_i K B_i ∧ 
  ∃ QX : Line, ∃ B_iX : Line, liesOn A_i QX ∧ liesOn B_i B_iX ∧ meet QX B_iK K P Q

-- Goal
theorem sum_AiK_lt_S 
  (A : Fin n → Point) (P Q K : Point) (A_iK : Fin n → ℝ) (S : ℝ)
  (midpoint_PQ : midpoint P Q K)
  (extended_AK : ∀ i, extended (A i) K (B i))
  (bisection_AiPBQ : ∀ i, bisection (A i) K P Q) : (
  ∑ i, dist (A i) K < S
) :=
sorry

end sum_AiK_lt_S_l349_349718


namespace min_val_of_expr_on_line_l349_349587

theorem min_val_of_expr_on_line (x y : ℝ) (h : x + y = 3) : 2^x + 2^y ≥ 4 * real.sqrt 2 :=
by sorry

end min_val_of_expr_on_line_l349_349587


namespace primes_between_30_and_50_l349_349397

theorem primes_between_30_and_50 : (Finset.card (Finset.filter Nat.Prime (Finset.Ico 30 51))) = 5 :=
by
  sorry

end primes_between_30_and_50_l349_349397


namespace available_codes_count_l349_349514

-- Definitions for conditions
def valid_digit_set : Set ℕ := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
def given_code : Fin 10 × Fin 10 × Fin 10 := (0, 2, 3)
def switch_positions (code : Fin 10 × Fin 10 × Fin 10) : Set (Fin 10 × Fin 10 × Fin 10) :=
  { (code.2.2, code.2.1, code.1), (code.2.1, code.1, code.2.2), (code.2.2, code.1, code.2.1) }

-- Condition that restricts codes to those that do not match two or more digits with the given code
def differs_in_two_or_more_digits (c1 c2 : Fin 10 × Fin 10 × Fin 10) : Bool :=
  (c1.1 ≠ c2.1 + c1.2.1 ≠ c2.2.1 + c1.2.2 ≠ c2.2.2) >= 2

-- Condition defining valid codes for Reckha
def is_valid_code (code : Fin 10 × Fin 10 × Fin 10) : Bool :=
  code ≠ given_code &&
  ¬(differs_in_two_or_more_digits code given_code) &&
  code ∉ switch_positions given_code

-- Main statement to prove
theorem available_codes_count : 
  (Fin 10 × Fin 10 × Fin 10).univ.filter is_valid_code).card = 969 :=
by
  sorry

end available_codes_count_l349_349514


namespace binom_10_2_eq_45_l349_349699

-- Definitions used in the conditions
def binom (n k : ℕ) := n.choose k

-- The statement that needs to be proven
theorem binom_10_2_eq_45 : binom 10 2 = 45 :=
by
  sorry

end binom_10_2_eq_45_l349_349699


namespace vinces_bus_ride_length_l349_349983

theorem vinces_bus_ride_length (zachary_ride : ℝ) (vince_extra : ℝ) (vince_ride : ℝ) :
  zachary_ride = 0.5 →
  vince_extra = 0.13 →
  vince_ride = zachary_ride + vince_extra →
  vince_ride = 0.63 :=
by
  intros hz hv he
  -- proof steps here
  sorry

end vinces_bus_ride_length_l349_349983


namespace total_pieces_of_clothes_l349_349010

theorem total_pieces_of_clothes (shirts_per_pant pants : ℕ) (h1 : shirts_per_pant = 6) (h2 : pants = 40) : 
  shirts_per_pant * pants + pants = 280 :=
by
  rw [h1, h2]
  sorry

end total_pieces_of_clothes_l349_349010


namespace tetrahedron_volume_l349_349633

theorem tetrahedron_volume (R : ℝ) : 
  let volume := (8 * R^3 * Real.sqrt 2) / 27 in
  ∃ ρ : ℝ, 
  (ρ = R) ∧ 
  ∃ V : ℝ, 
  (V = volume) ∧ 
  V = (8 * ρ^3 * Real.sqrt 2) / 27 :=
by
  sorry

end tetrahedron_volume_l349_349633


namespace product_of_all_eight_rolls_is_odd_prime_l349_349645

noncomputable def probability_odd_prime_product : ℚ :=
  let outcomes := {1, 2, 3, 4, 5, 6}
  let odd_primes := {3, 5}
  let prob_single_roll_odd_prime := 2 / 6 -- Probability of rolling a 3 or 5 on a single roll
  prob_single_roll_odd_prime ^ 8 -- Probability of rolling a 3 or 5 on all eight rolls

theorem product_of_all_eight_rolls_is_odd_prime :
  probability_odd_prime_product = 1 / 6561 :=
by
  sorry -- Proof to be filled in

end product_of_all_eight_rolls_is_odd_prime_l349_349645


namespace least_possible_n_l349_349625

variables (n d : ℕ) (d_pos : d > 0)

def total_cost : ℕ := d
def donated_cost : ℕ := 3 * d / (2 * n)
def remaining_radios : ℕ := n - 3
def sell_price_per_radio : ℕ := d / n + 10
def total_income : ℕ := donated_cost + (remaining_radios * sell_price_per_radio)
def profit : ℕ := total_income - total_cost

theorem least_possible_n : n = 11 :=
  sorry

end least_possible_n_l349_349625


namespace percent_value_in_quarters_l349_349597

theorem percent_value_in_quarters (num_dimes num_quarters : ℕ) 
  (value_dime value_quarter total_value value_in_quarters : ℕ) 
  (h1 : num_dimes = 75)
  (h2 : num_quarters = 30)
  (h3 : value_dime = num_dimes * 10)
  (h4 : value_quarter = num_quarters * 25)
  (h5 : total_value = value_dime + value_quarter)
  (h6 : value_in_quarters = num_quarters * 25) :
  (value_in_quarters / total_value) * 100 = 50 :=
by
  sorry

end percent_value_in_quarters_l349_349597


namespace binomial_coefficient_of_second_term_l349_349468

theorem binomial_coefficient_of_second_term :
  let f := λ (x : ℝ), (x^2 - 1/x)^5,
      binom_coef := λ (n k : ℕ), nat.choose n k in 
  ∃ r : ℕ, r = 1 ∧ 
  coefficient (λ (x : ℝ), binom_coef 5 r * (-1)^r * x^(10 - 3 * r)) = -5 :=
by 
  sorry

end binomial_coefficient_of_second_term_l349_349468


namespace quadratic_polynomial_P_l349_349116

noncomputable def P : Polynomial := {
  to_fun := λ x : ℝ, x^2 - x + 2,
  degree := 2
}

theorem quadratic_polynomial_P (P : ℝ → ℝ) 
  (h : ∀ x, P (P x) = x^4 - 2 * x^3 + 4 * x^2 - 3 * x + 4) : 
  P 8 = 58 := 
by
  sorry

end quadratic_polynomial_P_l349_349116


namespace digit_in_1999th_position_l349_349935

theorem digit_in_1999th_position : 
  let sequence := List.join $ List.map (fun n => n.toString.toList) (List.range (1999 + 1)) -- Concatenate the range of numbers as strings
  sequence[1999 - 1] = '7' := 
by
  sorry

end digit_in_1999th_position_l349_349935


namespace orthocenter_and_circumradius_l349_349972

theorem orthocenter_and_circumradius (R : ℝ)
  (Ω₁ Ω₂ Ω₃ : Circle)
  (h₁ : Ω₁.radius = R)
  (h₂ : Ω₂.radius = R)
  (h₃ : Ω₃.radius = R)
  (M A B C : Point)
  (hMA : M ∈ Ω₁ ∧ A ∈ Ω₁ ∧ A ≠ M ∧ M ∈ Ω₂ ∧ A ∈ Ω₂)
  (hMB : M ∈ Ω₂ ∧ B ∈ Ω₂ ∧ B ≠ M ∧ M ∈ Ω₃ ∧ B ∈ Ω₃)
  (hMC : M ∈ Ω₃ ∧ C ∈ Ω₃ ∧ C ≠ M ∧ M ∈ Ω₁ ∧ C ∈ Ω₁)
  : M.is_orthocenter (Triangle.mk A B C) ∧ (circumradius (Triangle.mk A B C) = R) :=
by
  sorry

end orthocenter_and_circumradius_l349_349972


namespace minimum_games_pasha_wins_l349_349078

noncomputable def pasha_initial_money : Nat := 9 -- Pasha has a single-digit amount
noncomputable def igor_initial_money : Nat := 1000 -- Igor has a four-digit amount
noncomputable def pasha_final_money : Nat := 100 -- Pasha has a three-digit amount
noncomputable def igor_final_money : Nat := 99 -- Igor has a two-digit amount

theorem minimum_games_pasha_wins :
  ∃ (games_won_by_pasha : Nat), 
    (games_won_by_pasha >= 7) ∧
    (games_won_by_pasha <= 7) := sorry

end minimum_games_pasha_wins_l349_349078


namespace find_pairs_l349_349731

def is_solution_pair (m n : ℕ) : Prop :=
  Nat.lcm m n = 3 * m + 2 * n + 1

theorem find_pairs :
  { pairs : List (ℕ × ℕ) // ∀ (m n : ℕ), (m, n) ∈ pairs ↔ is_solution_pair m n } :=
by
  let pairs := [(3,10), (4,9)]
  have key : ∀ (m n : ℕ), (m, n) ∈ pairs ↔ is_solution_pair m n := sorry
  exact ⟨pairs, key⟩

end find_pairs_l349_349731


namespace square_side_length_l349_349545

variables (s : ℝ) (π : ℝ)
  
theorem square_side_length (h : 4 * s = π * s^2 / 2) : s = 8 / π :=
by sorry

end square_side_length_l349_349545


namespace area_of_triangle_DEF_l349_349029

-- Define the variables and conditions
variables (DE : ℝ) (height : ℝ)
def area_of_triangle (base height : ℝ) : ℝ := (1/2) * base * height

-- Given conditions
axiom DE_is_12 : DE = 12
axiom height_is_7 : height = 7

-- Prove the area of triangle DEF
theorem area_of_triangle_DEF : area_of_triangle DE height = 42 :=
by {
  rw [DE_is_12, height_is_7],
  -- The expected result
  sorry
}

end area_of_triangle_DEF_l349_349029


namespace juice_bottles_left_l349_349561

-- Define the conditions and theorem to prove
theorem juice_bottles_left 
  (total_crates : ℕ)
  (bottles_per_crate : ℕ)
  (broken_crates : ℕ)
  (total_bottles_before : total_crates * bottles_per_crate = 42)
  (broken_bottles : broken_crates * bottles_per_crate = 18) :
  total_crates = 7 →
  bottles_per_crate = 6 →
  broken_crates = 3 →
  total_crates * bottles_per_crate - broken_crates * bottles_per_crate = 24 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3] at total_bottles_before broken_bottles
  rw [total_bottles_before, broken_bottles]
  sorry

end juice_bottles_left_l349_349561


namespace balloons_total_l349_349203

-- Statement translating the problem to Lean 4
theorem balloons_total (initial_balloons : ℝ) (given_balloons : ℝ) :
  initial_balloons = 7.0 →
  given_balloons = 5.0 →
  (initial_balloons + given_balloons) = 12.0 :=
by
  intros h_initial h_given
  rw [h_initial, h_given]
  exact rfl


end balloons_total_l349_349203


namespace metal_weights_l349_349624

def weights {G C S F : ℝ} : Prop :=
  G + C + S + F = 60 ∧
  G + C = 40 ∧
  G + S = 45 ∧
  G + F = 36

theorem metal_weights {G C S F : ℝ} (h : weights) : 
  G = 30.5 ∧ C = 9.5 ∧ S = 14.5 ∧ F = 5.5 :=
begin
  sorry,
end

end metal_weights_l349_349624


namespace P_P_eq_P_eight_equals_58_l349_349085

open Polynomial

noncomputable def P(x : ℚ) : ℚ := x^2 - x + 2

theorem P_P_eq :
  (P ∘ P)(x) = x^4 - 2 * x^3 + 4 * x^2 - 3 * x + 4 := sorry

theorem P_eight_equals_58 :
  P 8 = 58 := sorry

end P_P_eq_P_eight_equals_58_l349_349085


namespace union_result_l349_349510

variable (a b : ℝ)

-- Define the sets A and B based on given conditions
def A := {5, Real.log (a + 3) / Real.log 2}
def B := {a, b}

-- State the condition that A ∩ B = {2}
def condition : Prop := ({5, Real.log (a + 3) / Real.log 2} ∩ {a, b} = {2})

-- State the goal that A ∪ B = {1, 2, 5}
theorem union_result (h : condition a b) : {5, Real.log (a + 3) / Real.log 2} ∪ {a, b} = {1, 2, 5} := sorry

end union_result_l349_349510


namespace soda_difference_l349_349882

theorem soda_difference :
  let Julio_orange_bottles := 4
  let Julio_grape_bottles := 7
  let Mateo_orange_bottles := 1
  let Mateo_grape_bottles := 3
  let liters_per_bottle := 2
  let Julio_total_liters := Julio_orange_bottles * liters_per_bottle + Julio_grape_bottles * liters_per_bottle
  let Mateo_total_liters := Mateo_orange_bottles * liters_per_bottle + Mateo_grape_bottles * liters_per_bottle
  Julio_total_liters - Mateo_total_liters = 14 := by
    sorry

end soda_difference_l349_349882


namespace min_max_sum_squares_l349_349560

noncomputable def minSumSquares (n : ℕ) : ℕ :=
  n^2 * (2 * n + 1)

noncomputable def maxSumSquares (n : ℕ) : ℚ :=
  (n * (2 * n + 1) * (4 * n + 1)) / 3

theorem min_max_sum_squares (n : ℕ) (w : Fin (2 * n + 1) → ℕ) 
  (h : (Finset.univ : Finset (Fin (2 * n + 1))).sum w = n * (2 * n + 1)) :
  minSumSquares n ≤ (Finset.univ : Finset (Fin (2 * n + 1))).sum (λ i, (w i)^2) ∧
  (Finset.univ : Finset (Fin (2 * n + 1))).sum (λ i, (w i)^2) ≤ maxSumSquares n :=
by
  sorry

end min_max_sum_squares_l349_349560


namespace demand_decrease_l349_349622

theorem demand_decrease (original_price_increase effective_price_increase demand_decrease : ℝ)
  (h1 : original_price_increase = 0.2)
  (h2 : effective_price_increase = original_price_increase / 2)
  (h3 : new_price = original_price * (1 + effective_price_increase))
  (h4 : 1 / new_price = original_demand)
  : demand_decrease = 0.0909 := sorry

end demand_decrease_l349_349622


namespace smallest_possible_positive_value_l349_349294

noncomputable def sum_pairs (a : Fin 100 → ℤ) : ℤ :=
  (Finset.sum (Finset.Ico 0 100) (λ i, Finset.sum (Finset.Ico (i + 1) 100) (λ j, a i * a j)))

theorem smallest_possible_positive_value (a : Fin 100 → ℤ) (h : ∀ i, a i = 1 ∨ a i = -1) :
  ∃ S > 0, sum_pairs a = S ∧ S = 22 :=
by
  sorry

end smallest_possible_positive_value_l349_349294


namespace rocky_run_miles_l349_349857

theorem rocky_run_miles : 
  let day1 := 4 in
  let day2 := 2 * day1 in
  let day3 := 3 * day2 in
  day1 + day2 + day3 = 36 :=
by
  sorry

end rocky_run_miles_l349_349857


namespace range_of_m_l349_349059

noncomputable def G (x : ℝ) (m : ℝ) : ℝ := (8 * x ^ 2 + 24 * x + 5 * m) / 8

theorem range_of_m (x : ℝ) (m : ℝ) : 
  (∃ c : ℝ, G x m = (x + c) ^ 2 ∧ c ^ 2 = 3) → 4 ≤ m ∧ m ≤ 5 := 
by
  sorry

end range_of_m_l349_349059


namespace smallest_multiple_17_7_more_53_l349_349191

theorem smallest_multiple_17_7_more_53 : 
  ∃ a : ℕ, (17 * a ≡ 7 [MOD 53]) ∧ 17 * a = 187 := 
by
  have h : 17* 11 = 187 := by norm_num
  use 11
  constructor
  · norm_num
  · exact h
  sorry

end smallest_multiple_17_7_more_53_l349_349191


namespace max_n_valid_rearrangement_l349_349076

-- Definitions of points and distances
def circle_length := 2013
def num_points := 2013

-- Function to calculate the shorter arc distance
def shorter_arc_distance(p₁ p₂ : ℕ) : ℕ := 
  min (abs (p₁ - p₂)) (circle_length - abs (p₁ - p₂))

-- Function to return if a rearrangement is valid
def valid_rearrangement (original new : ℕ → ℕ) (n : ℕ) : Prop :=
  ∀ i j, (shorter_arc_distance i j ≤ n → shorter_arc_distance (new i) (new j) ≥ shorter_arc_distance i j)

-- The statement of the problem:
theorem max_n_valid_rearrangement : Exists (n : ℕ), n = 670 ∧ 
  (∀ chips, ∃ chips', (∃ n, valid_rearrangement chips chips' n) → n ≤ 670) := 
sorry

end max_n_valid_rearrangement_l349_349076


namespace total_number_of_students_is_40_l349_349456

variables (S R : ℕ)

-- Conditions
def students_not_borrowed_any_books := 2
def students_borrowed_1_book := 12
def students_borrowed_2_books := 10
def average_books_per_student := 2

-- Definition of total books borrowed
def total_books_borrowed := (0 * students_not_borrowed_any_books) + (1 * students_borrowed_1_book) + (2 * students_borrowed_2_books) + (3 * R)

-- Expression for total number of students
def total_students := students_not_borrowed_any_books + students_borrowed_1_book + students_borrowed_2_books + R

-- Mathematical statement to prove
theorem total_number_of_students_is_40 (h : total_books_borrowed R / total_students R = average_books_per_student) : total_students R = 40 :=
sorry

end total_number_of_students_is_40_l349_349456


namespace exists_m_in_interval_l349_349722

noncomputable def sequence (x : ℕ → ℝ) : (∀ n, x (n + 1) = (x n ^ 2 + 6 * x n + 9) / (x n + 7)) :=
sorry

theorem exists_m_in_interval :
  ∃ m : ℕ, 243 ≤ m ∧ m ≤ 728 ∧ x m ≤ 5 + 1 / 3^15 :=
begin
  let x : ℕ → ℝ := λ n, nat.rec_on n 7 (λ n y, (y ^ 2 + 6 * y + 9) / (y + 7)),
  have hx : ∀ n, x (n + 1) = (x n ^ 2 + 6 * x n + 9) / (x n + 7),
  { intro n, simp [x, nat.rec_on] },
  sorry
end

end exists_m_in_interval_l349_349722


namespace quadratic_P_value_l349_349120

noncomputable def P (x : ℝ) : ℝ :=
  x^2 - x + 2

theorem quadratic_P_value :
  P (P 8) = 8^4 - 2 * 8^3 + 4 * 8^2 - 3 * 8 + 4 ∧ P 8 = 58 := 
by
  have h1 : P(P(8)) = 8^4 - 2 * 8^3 + 4 * 8^2 - 3 * 8 + 4 := sorry
  have h2 : P(8) = 58 := sorry
  exact ⟨h1, h2⟩  

end quadratic_P_value_l349_349120


namespace induction_inequality_l349_349057

noncomputable theory

variable {n : ℕ} {a b : ℕ → ℝ}

-- Conditions given
def condition1 (a b : ℕ → ℝ) (n : ℕ) : Prop :=
  ∀ i, 1 ≤ i → i ≤ n → a i ≤ b i

def condition2 (a b : ℕ → ℝ) : Prop :=
  ∃ r, a r = b r

def initial_case (a b : ℕ → ℝ) : Prop :=
  a 1 > b 1

-- Transformation definitions
def transform_a (a : ℕ → ℝ) : ℕ → ℝ
| 1 := a 1 + a 2
| 2 := a 3
| i := a (i + 1) + F (i - 2) * a 1

def transform_b (b : ℕ → ℝ) : ℕ → ℝ
| 1 := b 1 + b 2
| 2 := b 3
| i := b (i + 1) + F (i - 2) * b 1

-- Main theorem
theorem induction_inequality {n : ℕ} {a b : ℕ → ℝ} (h1 : condition1 a b n)
  (h2 : condition2 a b) (h3 : initial_case a b) (h4 : n ≥ 3) :
  ∀ i, 1 ≤ i → i ≤ n → a i ≤ b i :=
begin
  sorry
end

end induction_inequality_l349_349057


namespace minimum_b_minus_a_l349_349376

noncomputable def f (x : ℝ) : ℝ := 1 + x - x^2 / 2 + x^3 / 3
noncomputable def g (x : ℝ) : ℝ := 1 - x + x^2 / 2 - x^3 / 3
noncomputable def F (x : ℝ) : ℝ := f (x - 4) * g (x + 3)

theorem minimum_b_minus_a : 
  (∃ a b : ℤ, a < b ∧ ∀ x : ℝ, F x = 0 → x ∈ set.Icc (a : ℝ) (b : ℝ) ) → 6 = 6 :=
by
  sorry

end minimum_b_minus_a_l349_349376


namespace tank_capacity_is_correct_l349_349649

theorem tank_capacity_is_correct :
  ∃ (C : ℝ), 
    (let leak_rate := C / 6 in
     let pipe_A_rate := 3.5 * 60 in
     let pipe_B_rate := 4.5 * 60 in
     let net_rate_with_both_pipes_open := pipe_A_rate + pipe_B_rate - leak_rate in
     let net_rate_with_pipe_A_open := pipe_A_rate - leak_rate in
     let total_water_in_out :=
       (net_rate_with_pipe_A_open * 1) + (net_rate_with_both_pipes_open * 7) - (leak_rate * 8) in
     total_water_in_out = 0) ∧ C = 1338.75 :=
begin
  -- The proof will go here
  sorry
end

end tank_capacity_is_correct_l349_349649


namespace smallest_positive_natural_number_l349_349568

theorem smallest_positive_natural_number (a b c d e : ℕ) 
    (h1 : a = 3) (h2 : b = 5) (h3 : c = 6) (h4 : d = 18) (h5 : e = 23) :
    ∃ (x y : ℕ), x = (e - a) / b - d / c ∨ x = e - d + b - c - a ∧ x = 1 := by
  sorry

end smallest_positive_natural_number_l349_349568


namespace rate_percent_l349_349989

noncomputable def calculate_rate (P: ℝ) : ℝ :=
  let I : ℝ := 320
  let t : ℝ := 2
  I * 100 / (P * t)

theorem rate_percent (P: ℝ) (hP: P > 0) : calculate_rate P = 4 := 
by
  sorry

end rate_percent_l349_349989


namespace train_length_l349_349650

-- Define the given speeds and time
def train_speed_km_per_h := 25
def man_speed_km_per_h := 2
def crossing_time_sec := 36

-- Convert speeds to m/s
def km_per_h_to_m_per_s (v : ℕ) : ℕ := (v * 1000) / 3600
def train_speed_m_per_s := km_per_h_to_m_per_s train_speed_km_per_h
def man_speed_m_per_s := km_per_h_to_m_per_s man_speed_km_per_h

-- Define the relative speed in m/s
def relative_speed_m_per_s := train_speed_m_per_s + man_speed_m_per_s

-- Theorem to prove the length of the train
theorem train_length : (relative_speed_m_per_s * crossing_time_sec) = 270 :=
by
  -- sorry is used to skip the proof
  sorry

end train_length_l349_349650


namespace triangle_areas_sum_l349_349020

theorem triangle_areas_sum
  (AC : ℝ)
  (H_AC : AC = 1)
  (angle_BAC angle_ABC angle_ACB : ℝ)
  (H_angle_BAC : angle_BAC = 30)
  (H_angle_ABC : angle_ABC = 80)
  (H_angle_ACB : angle_ACB = 70)
  (angle_DEC : ℝ)
  (H_angle_DEC : angle_DEC = 40)
  (BC : ℝ) 
  (H_BC : BC = AC * sin (angle_ACB))
  (E_midpoint : BC / 2)
  : 
  (0.5 * BC * sin (angle_BAC) + 2 * (0.5 * 0.5 * BC / 2 * sin (angle_DEC))) = 
  ((sin 70) / (sin 80)) * (1 / 4 + sin 40 / 4) := 
sorry

end triangle_areas_sum_l349_349020


namespace max_angle_with_projection_l349_349012

theorem max_angle_with_projection
  (l : Line) (π : Plane) (θ : ℝ)
  (hp : θ = 72)
  (hperp : Perpendicular l.project_to_plane π) :
  ∃ k, k ∈ π ∧ passes_through k l.foot_of_perpendicular ∧ angle_between l k = 90 :=
sorry

end max_angle_with_projection_l349_349012


namespace shaded_percentage_correct_l349_349195

def total_squares : ℕ := 6 * 6
def shaded_squares : ℕ := 18
def percentage_shaded (total shaded : ℕ) : ℕ := (shaded * 100) / total

theorem shaded_percentage_correct : percentage_shaded total_squares shaded_squares = 50 := by
  sorry

end shaded_percentage_correct_l349_349195


namespace min_amount_lottery_l349_349621

theorem min_amount_lottery (selected_from_01_to_10 : ℕ)
                          (selected_from_11_to_20 : ℕ)
                          (selected_from_21_to_30 : ℕ)
                          (selected_from_31_to_36 : ℕ)
                          (cost_per_bet : ℕ) : ℕ :=
  let ways_01_to_10 := 8
  let ways_11_to_20 := 10
  let ways_21_to_30 := 10
  let ways_31_to_36 := 6
  let total_combinations := ways_01_to_10 * ways_11_to_20 * ways_21_to_30 * ways_31_to_36
  let total_cost := total_combinations * cost_per_bet
  total_cost
  sorry

example : min_amount_lottery 3 2 1 1 2 = 8640 :=
by sorry

end min_amount_lottery_l349_349621


namespace twenty_solutions_n_values_l349_349070

theorem twenty_solutions_n_values (n : ℕ) : 
  (∃ (x y z : ℕ), (3 * x + 3 * y + 2 * z = n) ∧ (0 < x ∧ 0 < y ∧ 0 < z)) ∧ 
  (card { (x, y, z) : ℕ × ℕ × ℕ | 3 * x + 3 * y + 2 * z = n ∧ 0 < x ∧ 0 < y ∧ 0 < z } = 20) →
  (n = 20 ∨ n = 23) :=
by
  sorry

end twenty_solutions_n_values_l349_349070


namespace illuminate_plane_with_lights_l349_349969

open Set

theorem illuminate_plane_with_lights :
  ∃ (points : Fin 6 → ℝ × ℝ) (angles : Fin 6 → ℝ),
  let illuminate (p : ℝ × ℝ) (angle : ℝ) := { q | ∃ θ ∈ Icc (angle - π/6) (angle + π/6), q = (p.1 + cos θ, p.2 + sin θ) }
  -- Orienting the light sources appropriately
  (∃ a : Icc (0 : ℝ) (2 * π), (⋃ i, illuminate (points i) (angles i + a)) = univ) :=
sorry

end illuminate_plane_with_lights_l349_349969


namespace quadratic_polynomial_P8_l349_349129

theorem quadratic_polynomial_P8 :
  ∃ (a b c : ℝ), 
  (∀ x : ℝ, P x = a * x^2 + b * x + c) ∧ 
  (∀ x : ℝ, P (P x) = x^4 - 2 * x^3 + 4 * x^2 - 3 * x + 4) ∧ 
  P 8 = 58 :=
begin
  sorry
end

end quadratic_polynomial_P8_l349_349129


namespace problem_statement_l349_349943

-- The structure definitions would include the points, circle, chords, and centroid.

variables {Point : Type*} [MetricSpace Point]

-- Definitions based on the conditions
def is_circle (O : Point) (R : ℝ) : Prop := sorry  -- Defining a circle with center O and radius R

def is_chord (O : Point) (R : ℝ) (A A1 : Point) : Prop := sorry -- Chord AA1 of the circle

def centroid (A B C : Point) : Point := sorry -- Centroid of the triangle ABC

def diameter_circle (O M : Point) : Set Point := sorry -- Circle with diameter OM

-- Main theorem statement
theorem problem_statement (O M X : Point) {A A1 B B1 C C1 : Point} {R : ℝ}
  (h_circle : is_circle O R)
  (h_chord_A : is_chord O R A A1)
  (h_chord_B : is_chord O R B B1)
  (h_chord_C : is_chord O R C C1)
  (h_centroid : M = centroid A B C)
  (h_diameter_circle : ∀ X, X ∈ diameter_circle O M ↔ OX^2 + MX^2 = OM^2) :
  (\frac{AX}{XA1} + \frac{BX}{XB1} + \frac{CX}{XC1} = 3) ↔ X ∈ diameter_circle O M :=
sorry

end problem_statement_l349_349943


namespace largest_int_less_than_100_with_remainder_5_l349_349318

theorem largest_int_less_than_100_with_remainder_5 (x : ℤ) (n : ℤ) (h₁ : x = 7 * n + 5) (h₂ : x < 100) : 
  x = 96 := by
  sorry

end largest_int_less_than_100_with_remainder_5_l349_349318


namespace proof_problem_l349_349378

-- Definitions given in the problem
def line (λ : ℝ) : ℝ × ℝ → Prop :=
  λ p, p.1 + λ * p.2 - 2 * λ - 1 = 0

def circle (p : ℝ × ℝ) : Prop :=
  p.1 ^ 2 + p.2 ^ 2 = 1

-- Statements based on the conditions
def statement1 (λ : ℝ) : Prop :=
  let l_eqn := 3 * (0 : ℝ) - 4 * (1 : ℝ) + 5 = 0 in 
  ∃ p, line λ p ∧ circle p ∧ l_eqn

def statement2 (λ : ℝ) : Prop :=
  let max_distance := sqrt (5 : ℝ) in
  ∃ p, line λ p ∧ p = (max_distance, 0)

def statement3 (λ : ℝ) : Prop :=
  λ = -(1/2 : ℝ)

def statement4 (λ : ℝ) : Prop :=
  let area_max := 0.5 in  -- approximated triangular area
  (λ = -1/7 ∨ λ = -1) ∧ area_max = 0.5

-- Proof problem statement
theorem proof_problem : ∃ n : ℕ, n = 3 ∧ 
  (statement1 0.5 = false) ∧ 
  (statement2 1 = true) ∧ 
  (statement3 (-1/2) = true) ∧ 
  (statement4 (-1/7) = true) := 
sorry

end proof_problem_l349_349378


namespace marble_counts_l349_349971

theorem marble_counts (A B C : ℕ) : 
  (∃ x : ℕ, 
    A = 165 ∧ 
    B = 57 ∧ 
    C = 21 ∧ 
    (A = 55 * x / 27) ∧ 
    (B = 19 * x / 27) ∧ 
    (C = 7 * x / 27) ∧ 
    (7 * x / 9 = x / 9 + 54) ∧ 
    (A + B + C) = 3 * x
  ) :=
sorry

end marble_counts_l349_349971


namespace N_cannot_be_sum_of_three_squares_l349_349486

theorem N_cannot_be_sum_of_three_squares (K : ℕ) (L : ℕ) (N : ℕ) (h1 : N = 4^K * L) (h2 : L % 8 = 7) : ¬ ∃ (a b c : ℕ), N = a^2 + b^2 + c^2 := 
sorry

end N_cannot_be_sum_of_three_squares_l349_349486


namespace increasing_interval_ln_div_x_l349_349723

theorem increasing_interval_ln_div_x : 
  ∀ x : ℝ, 0 < x ∧ x < Real.exp 1 → (∀ y : ℝ, 0 < y ∧ y < Real.exp 1 → y > x → (ln y) / y > (ln x) / x) :=
by
  intros x hx y hy hxy
  sorry

end increasing_interval_ln_div_x_l349_349723


namespace smallest_number_from_digits_l349_349662

theorem smallest_number_from_digits : 
  ∀ (d1 d2 d3 d4 : ℕ), (d1 = 2) → (d2 = 0) → (d3 = 1) → (d4 = 6) →
  ∃ n : ℕ, (n = 1026) ∧ 
  ((n = d1 * 1000 + d2 * 100 + d3 * 10 + d4) ∨ 
   (n = d1 * 1000 + d2 * 100 + d4 * 10 + d3) ∨ 
   (n = d1 * 1000 + d3 * 100 + d2 * 10 + d4) ∨ 
   (n = d1 * 1000 + d3 * 100 + d4 * 10 + d2) ∨ 
   (n = d1 * 1000 + d4 * 100 + d2 * 10 + d3) ∨ 
   (n = d1 * 1000 + d4 * 100 + d3 * 10 + d2) ∨ 
   (n = d2 * 1000 + d1 * 100 + d3 * 10 + d4) ∨ 
   (n = d2 * 1000 + d1 * 100 + d4 * 10 + d3) ∨ 
   (n = d2 * 1000 + d3 * 100 + d1 * 10 + d4) ∨ 
   (n = d2 * 1000 + d3 * 100 + d4 * 10 + d1) ∨ 
   (n = d2 * 1000 + d4 * 100 + d1 * 10 + d3) ∨ 
   (n = d2 * 1000 + d4 * 100 + d3 * 10 + d1) ∨ 
   (n = d3 * 1000 + d1 * 100 + d2 * 10 + d4) ∨ 
   (n = d3 * 1000 + d1 * 100 + d4 * 10 + d2) ∨ 
   (n = d3 * 1000 + d2 * 100 + d1 * 10 + d4) ∨ 
   (n = d3 * 1000 + d2 * 100 + d4 * 10 + d1) ∨ 
   (n = d3 * 1000 + d4 * 100 + d1 * 10 + d2) ∨ 
   (n = d3 * 1000 + d4 * 100 + d2 * 10 + d1) ∨ 
   (n = d4 * 1000 + d1 * 100 + d2 * 10 + d3) ∨ 
   (n = d4 * 1000 + d1 * 100 + d3 * 10 + d2) ∨ 
   (n = d4 * 1000 + d2 * 100 + d1 * 10 + d3) ∨ 
   (n = d4 * 1000 + d2 * 100 + d3 * 10 + d1) ∨ 
   (n = d4 * 1000 + d3 * 100 + d1 * 10 + d2) ∨ 
   (n = d4 * 1000 + d3 * 100 + d2 * 10 + d1)) := sorry

end smallest_number_from_digits_l349_349662


namespace proof_problem_l349_349040

-- Definitions and conditions
variables {A B C : ℝ} {a b c : ℝ}

-- This condition describes a triangle with sides opposite to angles A, B, and C
axiom triangle_ABC (h₁ : 0 < A) (h₂ : 0 < B) (h₃ : 0 < C) 
                   (h₁₂ : A + B + C = π) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
                   (h_sine_rule : ∀ a b c A B C, a / sin A = b / sin B = c / sin C) :

-- Option A
def option_A (h : a > b) : Prop :=
  sin A > sin B

-- Option B
def option_B (h : sin A > sin B) : Prop :=
  A > B

-- Option C
def option_C (h : a * cos A = b * cos B) : Prop :=
  a = b ∧ A = B

-- Option D
def triangle_is_acute (h : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2) : Prop :=
  A + B + C = π

def option_D (h : triangle_is_acute)
            (h_sine_cosine : ∀ A B, sin A = cos B) : Prop :=
  sin A > cos B

-- The theorem that encapsulates all conditions and options
theorem proof_problem :
  (triangle_ABC h₁ h₂ h₃ h₁₂ ha hb hc h_sine_rule) →
  (∀ h, option_A h) →
  (∀ h, option_B h) →
  (∀ h, ¬ option_C h) →
  (∀ hₐ, option_D hₐ h_sine_cosine) :=
sorry

end proof_problem_l349_349040


namespace dividend_is_686_l349_349460

theorem dividend_is_686 (divisor quotient remainder : ℕ) (h1 : divisor = 36) (h2 : quotient = 19) (h3 : remainder = 2) :
  divisor * quotient + remainder = 686 :=
by
  sorry

end dividend_is_686_l349_349460


namespace value_of_PA_PB_is_3_l349_349860

-- Define the parametric equation of the line l
def parametric_line (t : ℝ) : ℝ × ℝ := ( (real.sqrt 2 / 2) * t, 3 + (real.sqrt 2 / 2) * t)

-- Define the polar equation of the curve C
def polar_curve (theta : ℝ) : ℝ := 4 * real.sin theta - 2 * real.cos theta

-- Define the standard (Cartesian) equation of line l
def cartesian_line (x y : ℝ) : Prop := x - y + 3 = 0

-- Define the Cartesian coordinate equation of curve C
def cartesian_curve (x y : ℝ) : Prop := x^2 + y^2 + 2 * x - 4 * y = 0

-- Define the intersection points P, A, and B
def point_p : ℝ × ℝ := (0, 3)

def intersection_points (x y : ℝ) : Prop :=
  cartesian_line x y ∧ cartesian_curve x y

-- Prove the value of |PA||PB| is 3
theorem value_of_PA_PB_is_3 :
  (∀ (x₁ y₁ x₂ y₂ : ℝ), 
    intersection_points x₁ y₁ ∧ intersection_points x₂ y₂ ∧ (x₁ ≠ 0 ∨ y₁ ≠ 3) ∧ (x₂ ≠ 0 ∨ y₂ ≠ 3) →
    real.sqrt ((x₁ - 0)^2 + (y₁ - 3)^2) * real.sqrt ((x₂ - 0)^2 + (y₂ - 3)^2) = 3)
:=
by
  sorry

end value_of_PA_PB_is_3_l349_349860


namespace binomial_10_2_l349_349691

noncomputable def binom (n k : ℕ) : ℕ := n! / (k! * (n - k)!)

theorem binomial_10_2 : binom 10 2 = 45 := by
  sorry

end binomial_10_2_l349_349691


namespace area_of_enclosed_region_l349_349940

noncomputable def area_enclosed_by_curves : ℝ :=
  2 * (∫ x in 0..1, x^2 + 1 - ∫ x in 1..2, (1 / 4) * x^2)

theorem area_of_enclosed_region :
  area_enclosed_by_curves = 4 / 3 :=
by
  sorry

end area_of_enclosed_region_l349_349940


namespace die_prob_tangent_die_prob_isosceles_l349_349261

-- Defining the problem context
def die_faces := {1, 2, 3, 4, 5, 6}

-- Defining the probability calculation
def prob_tangent (a b : ℕ) : ℝ :=
  if a^2 + b^2 = 25 then 1 else 0

def valid_isosceles (a b : ℕ) :=
  (a = b ∨ a = 5 ∨ b = 5)

-- Main theorem statements
theorem die_prob_tangent :
  let outcomes := ∑ a in die_faces, ∑ b in die_faces, 1 in
  let valid_tangents := ∑ a in die_faces, ∑ b in die_faces, prob_tangent a b in
  valid_tangents / outcomes = (1 : ℝ) / 18 :=
sorry

theorem die_prob_isosceles :
  let outcomes := ∑ a in die_faces, ∑ b in die_faces, 1 in
  let valid_isosceles := ∑ a in die_faces, ∑ b in die_faces, if valid_isosceles a b then 1 else 0 in
  valid_isosceles / outcomes = (7 : ℝ) / 18 :=
sorry

end die_prob_tangent_die_prob_isosceles_l349_349261


namespace baker_work_alone_time_l349_349224

theorem baker_work_alone_time 
  (rate_baker_alone : ℕ) 
  (rate_baker_with_helper : ℕ) 
  (total_time : ℕ) 
  (total_flour : ℕ)
  (time_with_helper : ℕ)
  (flour_used_baker_alone_time : ℕ)
  (flour_used_with_helper_time : ℕ)
  (total_flour_used : ℕ) 
  (h1 : rate_baker_alone = total_flour / 6) 
  (h2 : rate_baker_with_helper = total_flour / 2) 
  (h3 : total_time = 150)
  (h4 : flour_used_baker_alone_time = total_flour * flour_used_baker_alone_time / 6)
  (h5 : flour_used_with_helper_time = total_flour * (total_time - flour_used_baker_alone_time) / 2)
  (h6 : total_flour_used = total_flour) :
  flour_used_baker_alone_time = 45 :=
by
  sorry

end baker_work_alone_time_l349_349224


namespace two_pos_real_nums_are_one_and_five_l349_349176

noncomputable def find_pos_nums : ℝ × ℝ :=
  let a := 1
  let b := 5
  (a, b)

theorem two_pos_real_nums_are_one_and_five :
  ∃ (a b : ℝ), (0 < a) ∧ (0 < b) ∧ (a * b = 5) ∧ (2 * (a * b) / (a + b) = 5 / 3) ∧ (a = 1 ∧ b = 5 ∨ a = 5 ∧ b = 1) :=
by {
  -- Starting the proof, introducing variables a and b
  let a := (find_pos_nums).fst
  let b := (find_pos_nums).snd,

  -- Addressing conditions
  use [a, b],

  -- Verifying all conditions
  split, 
    apply zero_lt_one,
  split,
    apply zero_lt_five,
  split,
    apply find_pos_nums,
  split,
    by calc
      2 * (1 * 5) / (1 + 5) = 2 * 5 / 6 : by simp
                         ...  = 10 / 6 : by simp
                         ...  = 5 / 3 : by norm_num,
  
  -- Finally concluding the theorem
  right, split,
    refl,
    refl
}

end two_pos_real_nums_are_one_and_five_l349_349176


namespace binomial_10_2_equals_45_l349_349682

open Nat

theorem binomial_10_2_equals_45 : Nat.choose 10 2 = 45 := 
by
  sorry

end binomial_10_2_equals_45_l349_349682


namespace jills_present_age_l349_349962

-- Define the problem parameters and conditions
variables (H J : ℕ)
axiom cond1 : H + J = 43
axiom cond2 : H - 5 = 2 * (J - 5)

-- State the goal
theorem jills_present_age : J = 16 :=
sorry

end jills_present_age_l349_349962


namespace tetrahedron_edge_equality_l349_349242

/-- A tetrahedron has vertices A, B, C, and D. A line passes through the centroid and the circumcenter
of the tetrahedron and intersects edges AB and CD. We need to prove that AC = BD and AD = BC. -/
theorem tetrahedron_edge_equality
    (A B C D : Point)
    (line_through_centroid_circumcenter : Line)
    (intersects_AB : intersects_line_edge line_through_centroid_circumcenter A B)
    (intersects_CD : intersects_line_edge line_through_centroid_circumcenter C D)
    (centroid : centroid_of_tetrahedron {A, B, C, D})
    (circumcenter : circumcenter_of_tetrahedron {A, B, C, D})
    (line_through_centroid_and_circumcenter : passes_through line_through_centroid_circumcenter centroid ∧ passes_through line_through_centroid_circumcenter circumcenter)
    : AC = BD ∧ AD = BC :=
sorry

end tetrahedron_edge_equality_l349_349242


namespace exists_set_J_for_matrix_l349_349053

theorem exists_set_J_for_matrix 
  (n : ℕ) 
  (A : Matrix (Fin n) (Fin n) ℝ) 
  (h_diag : ∀ i : Fin n, A i i = 0) : 
  ∃ (J : Finset (Fin n)), 
    ∑ i in J, ∑ j in Jᶜ, A i j + ∑ i in Jᶜ, ∑ j in J, A i j ≥ 
    (1 / 2) * ∑ i in Finset.univ, ∑ j in Finset.univ, A i j := 
sorry

end exists_set_J_for_matrix_l349_349053


namespace equal_segments_l349_349248

-- Definitions of points and lines in the geometric setting
variables {A B C K L M N : Type} [LineGeo A B C K L M N]

def Circle (center : Type) (diameter : Type) := sorry

-- Define the centers of the circles and their properties
variables {O O1 O2 : Type}

-- Circles with diameters AB, AC, and BC respectively
variables (circle_AB : Circle O AB)
variables (circle_AC : Circle O1 AC)
variables (circle_BC : Circle O2 BC)

-- Line passing through C intersects the circles
variables (line_C : LineGeo C K L M N)

-- Typeclass instance representing the required segment length properties and congruence
class LineGeo (A B C K L M N : Type) where
  length_AC : ℝ
  length_BC : ℝ
  length_AB : ℝ
  angle_KO1O : ℝ
  angle_O2OL : ℝ
  congruent_tris : (segment K O1) ≅ (segment O2 L)

-- Theorem that proves the main goal
theorem equal_segments (h : LineGeo A B C K L M N) :
  segment KM = segment LN :=
sorry

end equal_segments_l349_349248


namespace rectangle_new_area_l349_349537

theorem rectangle_new_area (l w : ℝ) (h_area : l * w = 540) : 
  (1.15 * l) * (0.8 * w) = 497 :=
by
  sorry

end rectangle_new_area_l349_349537


namespace number_of_primes_between_30_and_50_l349_349432

-- Define what it means to be a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the interval condition
def in_interval (n : ℕ) : Prop :=
  30 ≤ n ∧ n ≤ 50

-- Define the count of prime numbers in the interval
def prime_count_in_interval : ℕ :=
  (List.range' 30 21).countp (λ n, is_prime n)

-- We state that the above count is equal to 5
theorem number_of_primes_between_30_and_50 : prime_count_in_interval = 5 :=
  sorry

end number_of_primes_between_30_and_50_l349_349432


namespace binomial_10_2_equals_45_l349_349681

open Nat

theorem binomial_10_2_equals_45 : Nat.choose 10 2 = 45 := 
by
  sorry

end binomial_10_2_equals_45_l349_349681


namespace problem_inequality_l349_349067

theorem problem_inequality (a b c : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_c_pos : 0 < c) (h_abc : a * b * c = 1) :
  (a - 1 + (1 / b)) * (b - 1 + (1 / c)) * (c - 1 + (1 / a)) ≤ 1 :=
sorry

end problem_inequality_l349_349067


namespace upstream_downstream_ratio_l349_349963

theorem upstream_downstream_ratio
  (V_b : ℝ) (V_s : ℝ) (h1 : V_b = 51) (h2 : V_s = 17) :
  let V_up := V_b - V_s,
      V_down := V_b + V_s,
      T_up := (D : ℝ) → D / V_up,
      T_down := (D : ℝ) → D / V_down
  in ∀ (D : ℝ), V_down / V_up = 2 :=
by
  sorry

end upstream_downstream_ratio_l349_349963


namespace number_of_comedies_rented_l349_349854

noncomputable def comedies_rented (r : ℕ) (a : ℕ) : ℕ := 3 * a

theorem number_of_comedies_rented (a : ℕ) (h : a = 5) : comedies_rented 3 a = 15 := by
  rw [h]
  exact rfl

end number_of_comedies_rented_l349_349854


namespace value_of_r_l349_349061

noncomputable theory

-- Define the conditions
variables {a b c r : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hr : 0 < r) (hne1 : r ≠ 1)
variables (h_geom_b : b = a * r) (h_geom_c : c = a * r^2)
variables (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)

-- Define the arithmetic sequence of logarithms
def logs_form_arith_seq := 
  let x := Real.log a in
  let y := Real.log b in
  let z := Real.log c in
  let log_a_r := Real.log r in
  let log_c_a := -x / (2 * x)
  let log_b_c := 1 + (Real.log (a * r^2) / Real.log (a * r))
  let log_a_b := 1 + log_a_r in
  log_b_c - log_c_a = log_a_b - log_b_c

-- The theorem statement
theorem value_of_r (h_logs_arith : logs_form_arith_seq a b c r) : 
  (2 + (Real.log a r)) / (1 + (Real.log a r)) - (-1) = 3 / 2 :=
sorry

end value_of_r_l349_349061


namespace area_enclosed_by_graph_eq_4_l349_349986

theorem area_enclosed_by_graph_eq_4 :
  (∫ x in -1..1, ∫ y in -1..1, indicator (λ z : ℝ × ℝ, x^4 + y^4 = abs x^3 + abs y^3) (x,y)) = 4 :=
by
  sorry

end area_enclosed_by_graph_eq_4_l349_349986


namespace students_like_both_l349_349000

-- Definitions based on given conditions
def total_students : ℕ := 500
def students_like_mountains : ℕ := 289
def students_like_sea : ℕ := 337
def students_like_neither : ℕ := 56

-- Statement to prove
theorem students_like_both : 
  students_like_mountains + students_like_sea - 182 + students_like_neither = total_students := 
by
  sorry

end students_like_both_l349_349000


namespace tangent_line_eq_monotonic_intervals_l349_349375

noncomputable def f (x : ℝ) (a : ℝ) := x - a * Real.log x
noncomputable def f' (x : ℝ) (a : ℝ) := 1 - (a / x)

theorem tangent_line_eq (x y : ℝ) (h : x = 1 ∧ a = 2) :
  y = f 1 2 → (x - 1) + (y - 1) - 2 * ((x - 1) + (y - 1)) = 0 := by sorry

theorem monotonic_intervals (a : ℝ) :
  (a ≤ 0 → ∀ x > 0, f' x a > 0) ∧
  (a > 0 → ∀ x > 0, (x < a → f' x a < 0) ∧ (x > a → f' x a > 0)) := by sorry

end tangent_line_eq_monotonic_intervals_l349_349375


namespace binom_10_2_eq_45_l349_349684

theorem binom_10_2_eq_45 : Nat.binomial 10 2 = 45 := by
  sorry

end binom_10_2_eq_45_l349_349684


namespace least_possible_value_of_k_l349_349055

theorem least_possible_value_of_k (n : ℕ) (h : n > 1) :
  ∃ k, (∀ (s : Finset ℕ), s.card = k → ∃ x y ∈ s, x^2 ∣ y) ∧ k = n^2 - n + 1 :=
sorry

end least_possible_value_of_k_l349_349055


namespace quadratic_polynomial_value_l349_349103

theorem quadratic_polynomial_value (P : ℝ → ℝ) (hP : ∀ x, P(P(x)) = x^4 - 2*x^3 + 4*x^2 - 3*x + 4) :
  P 8 = 58 :=
sorry

end quadratic_polynomial_value_l349_349103


namespace MQ_parallel_NP_l349_349269

open Real Complex

-- Define the rhombus and tangents structure.
structure Rhombus (A B C D O E F G H M N P Q : ℝ) :=
  (incircle : ∀ t ∈ {A, B, C, D}, abs t = 1)
  (tangent_points : ∀ v ∈ {E, F, G, H}, abs v = 1)
  (tangents_intersections : ∀ e ∈ {M, N, P, Q}, abs e = 1)

-- Proving MQ parallel to NP given the above structure
theorem MQ_parallel_NP 
  {A B C D O E F G H M N P Q : ℝ}
  (rhomb : Rhombus A B C D O E F G H M N P Q) :
  (MQ ∥ NP) := sorry

end MQ_parallel_NP_l349_349269


namespace trains_clear_time_l349_349980

-- Defining the lengths of the trains
def length_train1 : ℝ := 100 -- in meters
def length_train2 : ℝ := 280 -- in meters

-- Defining the speeds of the trains
def speed_train1 : ℝ := 42 -- in kmph
def speed_train2 : ℝ := 30 -- in kmph

-- Conversion factor from kmph to m/s
def kmph_to_mps : ℝ := 1000 / 3600

-- Converting speeds from kmph to m/s
def speed_train1_mps : ℝ := speed_train1 * kmph_to_mps
def speed_train2_mps : ℝ := speed_train2 * kmph_to_mps

-- Relative speed in m/s
def relative_speed : ℝ := speed_train1_mps + speed_train2_mps

-- Total length of both trains in meters
def total_length : ℝ := length_train1 + length_train2

-- Time in seconds for the trains to clear each other
def time_to_clear : ℝ := total_length / relative_speed

-- Proof statement
theorem trains_clear_time : time_to_clear = 19 :=
by 
  -- to be proved 
  sorry

end trains_clear_time_l349_349980


namespace cannot_form_set_l349_349589

-- Definitions of conditions
inductive GroupOfObjects
| AllRightAngledTriangles
| AllPointsOnCircle
| StudentsFarFromSchool (year : ℕ)
| HomeroomTeachers (year : ℕ)

-- The group of students in the first year of high school whose homes are far from the school cannot form a set
theorem cannot_form_set : 
  (GroupOfObjects.StudentsFarFromSchool 1) cannot form a set :=
sorry

end cannot_form_set_l349_349589


namespace ferris_wheel_time_l349_349616

noncomputable def ferris_wheel_time_computation : ℝ :=
  let A := 30
  let D := 30
  let B := real.pi / 45
  let desired_height := 45
  (fractional_part (real.arccos ((desired_height - D) / A) / B)).to_real

theorem ferris_wheel_time : ferris_wheel_time_computation = 15 :=
by
  sorry

end ferris_wheel_time_l349_349616


namespace bob_improvement_l349_349275

theorem bob_improvement :
  let bob_time := 10 * 60 + 40 -- 640 seconds
  let sister_time := 9 * 60 + 42 -- 582 seconds
  let time_difference := bob_time - sister_time -- 58 seconds
  (time_difference : ℚ) / bob_time * 100 ≈ 9.0625 :=
by
  sorry

end bob_improvement_l349_349275


namespace range_of_x_when_y_lt_0_l349_349035

variable (a b c n m : ℝ)

-- The definition of the quadratic function
def quadratic_function (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Conditions given in the problem
axiom value_at_neg1 : quadratic_function a b c (-1) = 4
axiom value_at_0 : quadratic_function a b c 0 = 0
axiom value_at_1 : quadratic_function a b c 1 = n
axiom value_at_2 : quadratic_function a b c 2 = m
axiom value_at_3 : quadratic_function a b c 3 = 4

-- Proof statement
theorem range_of_x_when_y_lt_0 : ∀ (x : ℝ), quadratic_function a b c x < 0 ↔ 0 < x ∧ x < 2 :=
sorry

end range_of_x_when_y_lt_0_l349_349035


namespace number_of_sides_of_convex_polygon_arithmetic_sequence_l349_349022

theorem number_of_sides_of_convex_polygon_arithmetic_sequence 
  (n : ℕ) 
  (n ≥ 3)
  (angles : Fin n → ℝ) 
  (is_arithmetic_sequence : ∃ a d, ∀ k, angles ⟨k, k < n⟩ = a + k * d) 
  (smallest_angle : angles 0 = 100) 
  (largest_angle : angles (n-1) = 140) 
  (sum_of_interior_angles : ∑ i in Finset.range n, angles i = (n - 2) * 180) : 
  n = 6 := 
  sorry

end number_of_sides_of_convex_polygon_arithmetic_sequence_l349_349022


namespace approx_equal_e_l349_349573
noncomputable def a : ℝ := 69.28
noncomputable def b : ℝ := 0.004
noncomputable def c : ℝ := 0.03
noncomputable def d : ℝ := a * b
noncomputable def e : ℝ := d / c

theorem approx_equal_e : abs (e - 9.24) < 0.01 :=
by
  sorry

end approx_equal_e_l349_349573


namespace monotonicity_of_f_l349_349372

def f (x a : ℝ) : ℝ := (x - 2) * Real.exp x + a * (x - 1) ^ 2

theorem monotonicity_of_f (a : ℝ) :
  (a ≥ 0 → (∀ x < 1, differentiable_at ℝ (f x a) x ∧ (∀ x, f' x a < 0)) ∧ (∀ x > 1, differentiable_at ℝ (f x a) x ∧ (∀ x, f' x a > 0))) ∧
  (a = -Real.exp 0.5 → (∀ x, differentiable_at ℝ (f x a) x ∧ (∀ x, f' x a ≥ 0))) ∧
  (a < -Real.exp 0.5 → (∀ x < 1, differentiable_at ℝ (f x a) x ∧ (∀ x, f' x a > 0)) ∧
    (∀ x > Real.log (-2 * a), differentiable_at ℝ (f x a) x ∧ (∀ x, f' x a > 0)) ∧
    (∀ x ∈ (1, Real.log (-2 * a)), differentiable_at ℝ (f x a) x ∧ (∀ x, f' x a < 0))) ∧
  (-Real.exp 0.5 < a ∧ a < 0 → (∀ x < Real.log (-2 * a), differentiable_at ℝ (f x a) x ∧ (∀ x, f' x a > 0)) ∧
    (∀ x > 1, differentiable_at ℝ (f x a) x ∧ (∀ x, f' x a > 0)) ∧
    (∀ x ∈ (Real.log (-2 * a), 1), differentiable_at ℝ (f x a) x ∧ (∀ x, f' x a < 0))) :=
sorry

end monotonicity_of_f_l349_349372


namespace cars_meet_time_l349_349976

theorem cars_meet_time :
  ∃ t : ℚ, 
    let d1 := 65 * t,
        d2 := 75 * t,
        total_distance := 600 in
    d1 + d2 = total_distance ∧ t = 30 / 7 :=
begin
  -- proof goes here
  sorry
end

end cars_meet_time_l349_349976


namespace Isabella_speed_is_correct_l349_349044

-- Definitions based on conditions
def distance_km : ℝ := 17.138
def time_s : ℝ := 38

-- Conversion factor
def conversion_factor : ℝ := 1000

-- Distance in meters
def distance_m : ℝ := distance_km * conversion_factor

-- Correct answer (speed in m/s)
def correct_speed : ℝ := 451

-- Statement to prove
theorem Isabella_speed_is_correct : distance_m / time_s = correct_speed :=
by
  sorry

end Isabella_speed_is_correct_l349_349044


namespace tan_A_tan_B_l349_349161

theorem tan_A_tan_B (A B C : ℝ) (R : ℝ) (H F : ℝ)
  (HF : H + F = 26) (h1 : 2 * R * Real.cos A * Real.cos B = 8)
  (h2 : 2 * R * Real.sin A * Real.sin B = 26) :
  Real.tan A * Real.tan B = 13 / 4 :=
by
  sorry

end tan_A_tan_B_l349_349161


namespace soda_difference_l349_349883

theorem soda_difference :
  let Julio_orange_bottles := 4
  let Julio_grape_bottles := 7
  let Mateo_orange_bottles := 1
  let Mateo_grape_bottles := 3
  let liters_per_bottle := 2
  let Julio_total_liters := Julio_orange_bottles * liters_per_bottle + Julio_grape_bottles * liters_per_bottle
  let Mateo_total_liters := Mateo_orange_bottles * liters_per_bottle + Mateo_grape_bottles * liters_per_bottle
  Julio_total_liters - Mateo_total_liters = 14 := by
    sorry

end soda_difference_l349_349883


namespace derivative_correct1_derivative_correct2_l349_349738

noncomputable def derivative1 (x : ℝ) : ℝ :=
3 * x^2 * log x + x^2

noncomputable def derivative2 (x : ℝ) : ℝ :=
(x^2 - 2 * x - 1) / exp x

theorem derivative_correct1 (x : ℝ) (hx : x > 0) :
  has_deriv_at (λ x, x^3 * log x) (derivative1 x) x :=
by sorry

theorem derivative_correct2 (x : ℝ) (hx : x > 0) :
  has_deriv_at (λ x, (1 - x^2) / exp x) (derivative2 x) x :=
by sorry

end derivative_correct1_derivative_correct2_l349_349738


namespace john_alice_total_dollars_l349_349877

theorem john_alice_total_dollars :
  let john_dollars := (5 / 8 : ℚ)
  let alice_dollars := (7 / 20 : ℚ)
  (john_dollars + alice_dollars) = (39 / 40 : ℚ) :=
by
  let john_dollars := (5 / 8 : ℚ)
  let alice_dollars := (7 / 20 : ℚ)
  have h1 : john_dollars + alice_dollars = (5 / 8) + (7 / 20) := by rfl
  have h2 : (5 / 8 : ℚ) + (7 / 20 : ℚ) = (100 + 35) / 160 := by norm_num
  have h3 : (135 / 160 : ℚ) = (27 / 32) := by norm_num
  have h4 : (27 / 32 : ℚ) = (39 / 40 : ℚ) := by split_ifs; norm_num
  exact h4

end john_alice_total_dollars_l349_349877


namespace agnes_hourly_wage_l349_349913

theorem agnes_hourly_wage (A : ℝ) 
  (Mila_hourly_wage : ℝ := 10) 
  (Agnes_weekly_hours : ℝ := 8) 
  (weeks_per_month : ℝ := 4) 
  (Mila_work_hours : ℝ := 48) 
  (Mila_monthly_earnings_eq_Agnes : Mila_hourly_wage * Mila_work_hours = Agnes_weekly_hours * weeks_per_month * A) :
  A = 15 :=
by
  have h : 32 * A = 480 := by rwa [Mila_monthly_earnings_eq_Agnes, show Agnes_weekly_hours * weeks_per_month = 32, by norm_num,
                                    show Mila_hourly_wage * Mila_work_hours = 480, by norm_num] 
  linarith

end agnes_hourly_wage_l349_349913


namespace quadratic_polynomial_value_l349_349102

theorem quadratic_polynomial_value (P : ℝ → ℝ) (hP : ∀ x, P(P(x)) = x^4 - 2*x^3 + 4*x^2 - 3*x + 4) :
  P 8 = 58 :=
sorry

end quadratic_polynomial_value_l349_349102


namespace hearty_beads_count_l349_349385

theorem hearty_beads_count :
  let blue_packages := 3
  let red_packages := 5
  let beads_per_package := 40
  let total_beads := blue_packages * beads_per_package + red_packages * beads_per_package
  total_beads = 320 :=
by
  let blue_packages := 3
  let red_packages := 5
  let beads_per_package := 40
  let total_beads := blue_packages * beads_per_package + red_packages * beads_per_package
  show total_beads = 320
  sorry

end hearty_beads_count_l349_349385


namespace solve_cubic_equation_l349_349932

theorem solve_cubic_equation :
  ∃ x : ℝ, x = - (1 / (1 + real.cbrt 2)) ∧ (x^3 + x^2 + x + 1/3 = 0) :=
by
  use - (1 / (1 + real.cbrt 2))
  split
  . rfl
  . sorry

end solve_cubic_equation_l349_349932


namespace tangent_length_eq_three_l349_349726

theorem tangent_length_eq_three (P : ℝ × ℝ) (hP : P = (-1, 4))
  (h_circle : ∀ x y : ℝ, x^2 + y^2 - 4 * x - 6 * y + 12 = 0 → (x - 2)^2 + (y - 3)^2 = 1) :
  ∃ l : ℝ, l = 3 ∧ (length_of_tangent_line P h_circle) = l :=
begin
  sorry
end

noncomputable def length_of_tangent_line (P : ℝ × ℝ) (h_circle : ∀ x y : ℝ, x^2 + y^2 - 4 * x - 6 * y + 12 = 0 → (x - 2)^2 + (y - 3)^2 = 1) : ℝ :=
  let d := real.sqrt ((2 - P.1)^2 + (3 - P.2)^2) in
  let r := 1 in
  real.sqrt (d^2 - r^2)

end tangent_length_eq_three_l349_349726


namespace largest_int_less_than_100_with_remainder_5_l349_349319

theorem largest_int_less_than_100_with_remainder_5 (x : ℤ) (n : ℤ) (h₁ : x = 7 * n + 5) (h₂ : x < 100) : 
  x = 96 := by
  sorry

end largest_int_less_than_100_with_remainder_5_l349_349319


namespace circle_equation_midpoint_trajectory_l349_349341

-- Define the given constants and conditions
def radius : ℝ := 2 * real.sqrt 5
def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (1, 10)
def Q : ℝ × ℝ := (-3, -6)

-- Statement for part (1): Equation of the circle
theorem circle_equation :
  ∃ Cx Cy : ℝ, (Cx > 0) ∧ (Cy > 0) ∧ (Cy = 6) ∧ ((Cx - 3)^2 + (Cy - 6)^2 = 20) ∧
  ((1 - Cx)^2 + (2 - Cy)^2 = 20) ∧ ((1 - Cx)^2 + (10 - Cy)^2 = 20) :=
sorry

-- Statement for part (2): Trajectory equation of midpoint M
theorem midpoint_trajectory :
  ∃ (M : ℝ × ℝ), ∀ (P : ℝ × ℝ), ((P.1 - 3)^2 + (P.2 - 6)^2 = 20) →
  let Mx := (P.1 - 3) / 2,
      My := (P.2 - 6) / 2 in
  (Mx^2 + My^2 = 5) :=
sorry

end circle_equation_midpoint_trajectory_l349_349341


namespace abs_distance_diff_eq_2_sqrt_2_l349_349953

theorem abs_distance_diff_eq_2_sqrt_2 :
  let line (x y : ℝ) := y - 2 * x - 1,          -- Line equation
      parabola (x y : ℝ) := y^2 = 4 * x + 1,     -- Parabola equation
      Q := (2 : ℝ, 0) in                       -- Point Q
  ∀ C D : ℝ × ℝ,                               -- Points C and D (intersections)
    parabola C.1 C.2 ∧ line C.1 C.2 = 0 ∧ parabola D.1 D.2 ∧ line D.1 D.2 = 0 →
    |(C.snd - Q.snd) ^ 2 + (C.fst - Q.fst) ^ 2 - ((D.snd - Q.snd) ^ 2 + (D.fst - Q.fst) ^ 2) | = 2 * real.sqrt 2 := 
begin
  sorry
end

end abs_distance_diff_eq_2_sqrt_2_l349_349953


namespace find_loss_percentage_l349_349640

-- Assume the total worth of stock is 22500
def W : ℝ := 22500

-- Overall loss is given as 450
def OL : ℝ := 450

-- Profit from 20% of the stock sold at 10% profit
def profit_from_20_percent := 0.1 * 0.2 * W

-- Loss from 80% of the stock sold at L% loss
def loss_from_80_percent (L : ℝ) := (L / 100) * 0.8 * W

-- Overall loss is calculated as the difference between loss from remaining stock and profit from 20% stock
theorem find_loss_percentage (L : ℝ) (h : OL = loss_from_80_percent L - profit_from_20_percent) : L = 5 := by
  sorry

end find_loss_percentage_l349_349640


namespace john_spends_6_dollars_l349_349050

-- Let treats_per_day, cost_per_treat, and days_in_month be defined by the conditions of the problem.
def treats_per_day : ℕ := 2
def cost_per_treat : ℝ := 0.1
def days_in_month : ℕ := 30

-- The total expenditure should be defined as the number of treats multiplied by their cost.
def total_number_of_treats := treats_per_day * days_in_month
def total_expenditure := total_number_of_treats * cost_per_treat

-- The statement to be proven: John spends $6 on the treats.
theorem john_spends_6_dollars :
  total_expenditure = 6 :=
sorry

end john_spends_6_dollars_l349_349050


namespace initial_average_weight_l349_349141

theorem initial_average_weight 
    (W : ℝ)
    (a b c d e : ℝ)
    (h1 : (a + b + c) / 3 = W)
    (h2 : (a + b + c + d) / 4 = W)
    (h3 : (b + c + d + (d + 3)) / 4 = 68)
    (h4 : a = 81) :
    W = 70 := 
sorry

end initial_average_weight_l349_349141


namespace equation_p_adic_solutions_l349_349083

noncomputable def p_adic_solution := sorry

theorem equation_p_adic_solutions :
  ∃ x1 x2 x3 x4 : ℕ, (x1^2 = x1 ∧ x2^2 = x2 ∧ x3^2 = x3 ∧ x4^2 = x4) ∧
  (x1 = 0) ∧ (x2 = 1) ∧ (x3 = p_adic_solution) ∧ (x4 = p_adic_solution) :=
sorry

end equation_p_adic_solutions_l349_349083


namespace sum_of_leading_digits_of_roots_l349_349281

def leading_digit (n : ℕ) : ℕ :=
  let str_n := n.toString
  str_n[0].toNat - '0'.toNat

def g (M : ℕ) (r : ℕ) : ℕ :=
  leading_digit (Nat.root r M)

theorem sum_of_leading_digits_of_roots :
  let M := Nat.pow 10 150 - 1 in -- M is the 150-digit number with all digits 9.
  g M 2 + g M 3 + g M 4 + g M 5 + g M 6 = 8 :=
by
  sorry

end sum_of_leading_digits_of_roots_l349_349281


namespace number_of_primes_between_30_and_50_l349_349416

/-- 
  Prove that there are exactly 5 prime numbers in the range from 30 to 50. 
  These primes are 31, 37, 41, 43, and 47.
-/
theorem number_of_primes_between_30_and_50 : 
  (Finset.filter Nat.Prime (Finset.range 51)).card - 
  (Finset.filter Nat.Prime (Finset.range 31)).card = 5 := 
by 
  sorry

end number_of_primes_between_30_and_50_l349_349416


namespace quadratic_P_value_l349_349126

noncomputable def P (x : ℝ) : ℝ :=
  x^2 - x + 2

theorem quadratic_P_value :
  P (P 8) = 8^4 - 2 * 8^3 + 4 * 8^2 - 3 * 8 + 4 ∧ P 8 = 58 := 
by
  have h1 : P(P(8)) = 8^4 - 2 * 8^3 + 4 * 8^2 - 3 * 8 + 4 := sorry
  have h2 : P(8) = 58 := sorry
  exact ⟨h1, h2⟩  

end quadratic_P_value_l349_349126


namespace det_matrix_A_l349_349895

noncomputable def matrix_A (a b c d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![a, b], ![c, d]]

noncomputable def matrix_A_inv (a b c d : ℝ) (h : a * d - b * c ≠ 0) : Matrix (Fin 2) (Fin 2) ℝ :=
  (a * d - b * c)⁻¹ • ![![d, -b], ![-c, a]]

theorem det_matrix_A (a b c d : ℝ) (h : matrix_A a b c d + 2 • matrix_A_inv a b c d (by linarith) = 0) :
  Matrix.det (matrix_A a b c d) = 2 :=
begin
  sorry,
end

end det_matrix_A_l349_349895


namespace range_of_k_l349_349749

-- Define the sequence and its "good number"
def A_n (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (∑ i in finset.range n, 2 ^ i * a (i + 1)) / n

-- Define the sum of the first n terms of the modified sequence
def S_n (a : ℕ → ℝ) (k : ℝ) (n : ℕ) : ℝ :=
  ∑ i in finset.range n, a (i + 1) - k * (i + 1)

-- State the main theorem
theorem range_of_k (a : ℕ → ℝ) (k : ℝ) :
  (∀ n, A_n a n = 2 ^ (n + 1)) →
  (∀ n, S_n a k n ≤ S_n a k 6) →
  k ∈ set.Icc (16 / 7 : ℝ) (7 / 3 : ℝ) := sorry

end range_of_k_l349_349749


namespace extended_morse_code_symbols_l349_349463

def symbol_count (n : ℕ) : ℕ :=
  if n = 1 then 2
  else if n = 2 then 2
  else if n = 3 then 1
  else if n = 4 then 1 + 4 + 1
  else if n = 5 then 1 + 8
  else 0

theorem extended_morse_code_symbols : 
  (symbol_count 1 + symbol_count 2 + symbol_count 3 + symbol_count 4 + symbol_count 5) = 20 :=
by sorry

end extended_morse_code_symbols_l349_349463


namespace quadratic_root_2020_l349_349840

theorem quadratic_root_2020 (a b : ℝ) (h₀ : a ≠ 0) (h₁ : a * 2019^2 + b * 2019 - 1 = 0) :
    ∃ x : ℝ, (a * (x - 1)^2 + b * (x - 1) = 1) ∧ x = 2020 :=
by
  sorry

end quadratic_root_2020_l349_349840


namespace part_a_part_b_l349_349502

-- Part (a)
theorem part_a (a b : ℕ) (h : Nat.lcm a (a + 5) = Nat.lcm b (b + 5)) : a = b :=
sorry

-- Part (b)
theorem part_b (a b c : ℕ) (gcd_abc : Nat.gcd a (Nat.gcd b c) = 1) :
  Nat.lcm a b = Nat.lcm (a + c) (b + c) → False :=
sorry

end part_a_part_b_l349_349502


namespace sum_of_digits_of_A15B94_multiple_of_99_l349_349843

theorem sum_of_digits_of_A15B94_multiple_of_99 (A B : ℕ) 
  (hA : A < 10) (hB : B < 10)
  (h_mult_99 : ∃ n : ℕ, (100000 * A + 10000 + 5000 + 100 * B + 90 + 4) = 99 * n) :
  A + B = 8 := 
by
  sorry

end sum_of_digits_of_A15B94_multiple_of_99_l349_349843


namespace P_eight_value_l349_349093

def quadratic_polynomial (P : ℤ → ℤ) : Prop :=
  ∃ a b c : ℤ, ∀ x : ℤ, P(x) = a * x^2 + b * x + c

theorem P_eight_value (P : ℤ → ℤ)
  (H : ∀ x : ℤ, P(P(x)) = x^4 - 2 * x^3 + 4 * x^2 - 3 * x + 4)
  (H_quad : quadratic_polynomial P) :
  P 8 = 58 :=
sorry

end P_eight_value_l349_349093


namespace coefficient_x9_l349_349310

theorem coefficient_x9 (p : Polynomial ℚ) : 
  p = (1 + 3 * Polynomial.X - Polynomial.X^2)^5 →
  Polynomial.coeff p 9 = 15 := 
by
  intro h
  rw [h]
  -- additional lean tactics to prove the statement would go here
  sorry

end coefficient_x9_l349_349310


namespace primes_between_30_and_50_l349_349395

theorem primes_between_30_and_50 : (Finset.card (Finset.filter Nat.Prime (Finset.Ico 30 51))) = 5 :=
by
  sorry

end primes_between_30_and_50_l349_349395


namespace polygon_interior_sum_polygon_angle_ratio_l349_349611

-- Part 1: Number of sides based on the sum of interior angles
theorem polygon_interior_sum (n: ℕ) (h: (n - 2) * 180 = 2340) : n = 15 :=
  sorry

-- Part 2: Number of sides based on the ratio of interior to exterior angles
theorem polygon_angle_ratio (n: ℕ) (exterior_angle: ℕ) (ratio: 13 * exterior_angle + 2 * exterior_angle = 180) : n = 15 :=
  sorry

end polygon_interior_sum_polygon_angle_ratio_l349_349611


namespace parents_attended_meeting_l349_349231

variable (S B R N : ℕ)
variable (S_eq : S = 25) (B_eq : B = 11) (R_eq : R = 42) (R_eq_1_5N : 1.5 * N = R)

theorem parents_attended_meeting : S + R - B + N = 95 :=
by
  -- Using the given conditions
  rw [S_eq, B_eq, R_eq, ←R_eq_1_5N]  -- We use ← to rewrite in the reverse direction
  -- Sorry to skip the actual proof steps
  sorry

end parents_attended_meeting_l349_349231


namespace parallelogram_diagonals_intersect_l349_349245

variables {A B C O N' P N P' : Type*}

theorem parallelogram_diagonals_intersect
  (points_distinct: A ≠ B ∧ B ≠ C ∧ C ≠ A)
  (parallelogram_side: Segment A B)
  (O_condition: segment_intersects_point A C N' O ∧ segment_intersects_point B C P O)
  (parallel_condition: line_parallel AB N'P) :
  exists (parallelogram : Parallelogram), parallelogram_diagonals_intersect_at parallelogram O :=
begin
  sorry -- Proof
end

end parallelogram_diagonals_intersect_l349_349245


namespace cube_root_expression_l349_349298

theorem cube_root_expression (N : ℝ) (h : 1 < N) : (√[3] (N * (√ (N * (√[3] N))))) = N ^ (5/9) :=
by 
sorry

end cube_root_expression_l349_349298


namespace num_unique_four_digit_numbers_l349_349816

theorem num_unique_four_digit_numbers (d1 d2 d3 d4 : ℕ) (h1 : d1 = 3) (h2 : d2 = 0) (h3 : d3 = 3) (h4 : d4 = 3) :
  {x // (x = [d1, d2, d3, d4].perm.filter (λ l, l.head ≠ 0))}.card = 3 :=
sorry

end num_unique_four_digit_numbers_l349_349816


namespace sector_area_l349_349790

noncomputable def area_of_sector (C : ℝ) (theta : ℝ) : ℝ := 
  let r := C / (theta + 2)
  in (1/2) * r * r * theta

-- Given conditions
variables (C : ℝ) (theta : ℝ)
-- We are given that C = 6 and theta = 1
def C := 6
def theta := 1

theorem sector_area :
  area_of_sector C theta = 2 :=
by
  unfold area_of_sector
  rw [C, theta]
  sorry

end sector_area_l349_349790


namespace binom_P_X_4_eq_3_times_0_4_4_l349_349449

noncomputable def P_X_equals_4 (n : ℕ) (X : ℕ → ℝ) : ℝ := 
  if (E (pmf_of_fn (λ k, if k = 4 then n * 0.4 else 0)) = 2) then 
    (pmf_of_fn (λ k, if k = 4 then (n choose 4) * (0.4) ^ 4 * (0.6) ^ (n - 4) else 0)) 4
  else 0

theorem binom_P_X_4_eq_3_times_0_4_4 : 
  (X : ℕ → ℝ) (hX : E (pmf_of_fn (λ k, if k = 4 then n * 0.4 else 0)) = 2) :
  P_X_equals_4 5 X = 3 * (0.4) ^ 4 :=
by sorry

end binom_P_X_4_eq_3_times_0_4_4_l349_349449


namespace rotation_earth_certain_event_l349_349588

def Event (description : String) : Type := description

def certain_event : Prop := ∀ (e : Event "The rotation of the Earth"), e = e

theorem rotation_earth_certain_event : certain_event :=
by sorry

end rotation_earth_certain_event_l349_349588


namespace quadratic_polynomial_value_l349_349101

theorem quadratic_polynomial_value (P : ℝ → ℝ) (hP : ∀ x, P(P(x)) = x^4 - 2*x^3 + 4*x^2 - 3*x + 4) :
  P 8 = 58 :=
sorry

end quadratic_polynomial_value_l349_349101


namespace reciprocal_of_lcm_24_221_l349_349536

theorem reciprocal_of_lcm_24_221 : (1 / Nat.lcm 24 221) = (1 / 5304) :=
by 
  sorry

end reciprocal_of_lcm_24_221_l349_349536


namespace total_tomatoes_l349_349885

noncomputable def first_plant : ℕ := 24

noncomputable def second_plant : ℕ := (first_plant / 2) + 5

noncomputable def third_plant : ℕ := second_plant + 2

theorem total_tomatoes : first_plant + second_plant + third_plant = 60 :=
by
  rw [first_plant, second_plant, third_plant]
  -- after expanding we get: 24 + ((24 / 2) + 5) + (((24 / 2) + 5) + 2)
  calc 24 + ((24 / 2) + 5) + (((24 / 2) + 5) + 2) = 24 + (12 + 5) + (17 + 2) : by norm_num
  ... = 24 + 17 + 19 : by norm_num
  ... = 60 : by norm_num

end total_tomatoes_l349_349885


namespace swap_equality_l349_349521

theorem swap_equality {a1 b1 a2 b2 : ℝ} 
  (h1 : a1^2 + b1^2 = 1)
  (h2 : a2^2 + b2^2 = 1)
  (h3 : a1 * a2 + b1 * b2 = 0) :
  b1 = a2 ∨ b1 = -a2 :=
by sorry

end swap_equality_l349_349521


namespace find_c_l349_349905

variable (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)

theorem find_c (h1 : x = 2.5 * y) (h2 : 2 * y = (c / 100) * x) : c = 80 :=
sorry

end find_c_l349_349905


namespace evaluate_expression_l349_349672

theorem evaluate_expression : 
  (-1 : ℝ) ^ 2023 - ((-1 / 4) ^ 0 : ℝ) + 2 * real.cos (real.pi / 3) = -1 := by
  have h1 : (-1 : ℝ) ^ 2023 = -1 := by sorry
  have h2 : ((-1 / 4) ^ 0 : ℝ) = 1 := by sorry
  have h3 : real.cos (real.pi / 3) = 1 / 2 := by sorry
  calc (-1 : ℝ) ^ 2023 - ((-1 / 4) ^ 0 : ℝ) + 2 * real.cos (real.pi / 3)
      = -1 - 1 + 1 : by rw [h1, h2, h3]
  ... = -1   : by sorry

end evaluate_expression_l349_349672


namespace acute_triangle_half_perimeter_l349_349026

theorem acute_triangle_half_perimeter 
  {A B C D E F : Type} 
  [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E] [metric_space F]
  (hA : acute_triangle A B C)
  (hAD : altitude A D)
  (hBE : altitude B E)
  (hCF : altitude C F) :
  perimeter (D, E, F) ≤ (1/2) * perimeter (A, B, C) := 
sorry

end acute_triangle_half_perimeter_l349_349026


namespace smallest_number_of_people_l349_349586

theorem smallest_number_of_people : 
  ∃ x, (∀ y, (y % 14 = 0 ∧ y % 21 = 0 ∧ y % 35 = 0) ⇒ y ≥ x) ∧ (14 ∣ x ∧ 21 ∣ x ∧ 35 ∣ x ∧ x = 210) :=
by
  -- proof outline (add sorry to skip the actual proof)
  sorry

end smallest_number_of_people_l349_349586


namespace base_eight_to_base_ten_l349_349571

theorem base_eight_to_base_ten : 
  let base_eight_number := 145 in 
  let base_ten_answer := 101 in 
  (1 * 8^2 + 4 * 8^1 + 5 * 8^0 = base_ten_answer) → 
  (natrepr base_eight_number 8 = base_ten_answer) := 
by 
  intros base_eight_number base_ten_answer h
  exact h
  -- 1 * 64 + 4 * 8 + 5 * 1 = 101

-- Definitions and details to assist the theorem could be:
def natrepr (n: ℕ) (base: ℕ): ℕ :=
  let d₀ := n % 10 in
  let d₁ := (n / 10) % 10 in
  let d₂ := (n / 100) % 10 in
  d₀ * base^0 + d₁ * base^1 + d₂ * base^2

#eval natrepr 145 8 -- Evaluates to 101, verifying the function.

end base_eight_to_base_ten_l349_349571


namespace symmetric_difference_A_B_l349_349745

open Set

variable (R : Type) [LinearOrderedField R]

def A : Set R := {y | ∃ x : R, y = x ^ 2}
def B : Set R := {y | -2 ≤ y ∧ y ≤ 2}

theorem symmetric_difference_A_B : 
  A ∆ B = {y | y > 2} ∪ {y | -2 ≤ y ∧ y < 0} := 
by 
  sorry

end symmetric_difference_A_B_l349_349745


namespace perimeter_of_midpoint_triangle_l349_349842

theorem perimeter_of_midpoint_triangle (a : ℝ) (h : a = 3) : 
  ∃ p, p = (3 * (a / 2)) ∧ p = 9 / 2 :=
by
  use (3 * (a / 2)),
  split,
  { rw h, },
  { rw h, }

end perimeter_of_midpoint_triangle_l349_349842


namespace problem_tan_alpha_beta_cos_sin_l349_349355

theorem problem_tan_alpha_beta_cos_sin (α β : ℝ) (h₀ : ∀ x : ℝ, x^2 - 4 * x - 2 = 0 → x = tan α ∨ x = tan β) :
  cos (α + β)^2 + 2 * sin (α + β) * cos (α + β) - 2 * sin (α + β)^2 = 1 / 25 :=
by sorry

end problem_tan_alpha_beta_cos_sin_l349_349355


namespace unique_zero_identity_l349_349525

theorem unique_zero_identity (n : ℤ) : (∀ z : ℤ, z + n = z ∧ z * n = 0) → n = 0 :=
by
  intro h
  have h1 : ∀ z : ℤ, z + n = z := fun z => (h z).left
  have h2 : ∀ z : ℤ, z * n = 0 := fun z => (h z).right
  sorry

end unique_zero_identity_l349_349525


namespace count_primes_between_30_and_50_l349_349428

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between_30_and_50 : List ℕ :=
  [31, 37, 41, 43, 47]

theorem count_primes_between_30_and_50 : 
  (primes_between_30_and_50.filter is_prime).length = 5 :=
by
  sorry

end count_primes_between_30_and_50_l349_349428


namespace cheese_cookie_packs_l349_349229

def packs_per_box (P : ℕ) : Prop :=
  let cartons := 12
  let boxes_per_carton := 12
  let total_boxes := cartons * boxes_per_carton
  let total_cost := 1440
  let box_cost := total_cost / total_boxes
  let pack_cost := 1
  P = box_cost / pack_cost

theorem cheese_cookie_packs : packs_per_box 10 := by
  sorry

end cheese_cookie_packs_l349_349229


namespace range_of_a_l349_349796

variable {R : Type} [LinearOrderedField R]

def f (x a : R) : R := |x - 1| + |x - 2| - a

theorem range_of_a (h : ∀ x : R, f x a > 0) : a < 1 :=
by
  sorry

end range_of_a_l349_349796


namespace cos_product_identity_l349_349610

theorem cos_product_identity :
  3.422 * (Real.cos (π / 15)) * (Real.cos (2 * π / 15)) * (Real.cos (3 * π / 15)) *
  (Real.cos (4 * π / 15)) * (Real.cos (5 * π / 15)) * (Real.cos (6 * π / 15)) * (Real.cos (7 * π / 15)) =
  (1 / 2^7) :=
sorry

end cos_product_identity_l349_349610


namespace complex_quadratic_problem_l349_349896

theorem complex_quadratic_problem (ω : ℂ) (hω : ω^5 = 1) (hω1 : ω ≠ 1) :
  let α := ω + ω^2
  let β := ω^3 + ω^4
  ∃ a b : ℝ, α = -(β) ∧ α * β = b ∧ (a = 0 ∧ b = 2) := 
by {
  let α := ω + ω^2,
  let β := ω^3 + ω^4,
  use [0, 2],
  split,
  sorry, -- Proof that α = -β
  split,
  sorry, -- Proof that α * β = 2
  split,
  refl,
  refl
}

end complex_quadratic_problem_l349_349896


namespace supermarket_sold_54_pints_l349_349648

theorem supermarket_sold_54_pints (x s : ℝ) 
  (h1 : x * s = 216)
  (h2 : x * (s + 2) = 324) : 
  x = 54 := 
by 
  sorry

end supermarket_sold_54_pints_l349_349648


namespace george_total_fees_l349_349334

variables {borrow_amount : ℝ} {initial_fee_percentage : ℝ} {time_weeks : ℕ}

/-- Define the conditions: borrow amount, initial fee percentage and duration in weeks -/
def george_loan_conditions (borrow_amount : ℝ) (initial_fee_percentage : ℝ) (time_weeks : ℕ) : Prop :=
  borrow_amount = 100 ∧ initial_fee_percentage = 0.05 ∧ time_weeks = 2

/-- Define the calculation of the total fee -/
def calculate_total_fees (borrow_amount : ℝ) (initial_fee_percentage : ℝ) (time_weeks : ℕ) : ℝ :=
  let first_week_fee := borrow_amount * initial_fee_percentage in
  let second_week_fee := borrow_amount * (initial_fee_percentage * 2) in
  first_week_fee + second_week_fee

/-- Proof problem statement: Prove that the total fee is $15 -/
theorem george_total_fees (h : george_loan_conditions borrow_amount initial_fee_percentage time_weeks) :
  calculate_total_fees borrow_amount initial_fee_percentage time_weeks = 15 :=
by sorry

end george_total_fees_l349_349334


namespace sequence_product_l349_349638

theorem sequence_product (a : ℕ → ℚ) (h0 : a 0 = 1/3) (h : ∀ n, a (n + 1) = 1 + (a n - 1)^3) :
  (∀ n, a n ≠ 0) → (∀ n, a n > 0) → (a 0 * a 1 * a 2 * a 3 * ...) = 3/5 :=
by
  sorry

end sequence_product_l349_349638


namespace adam_lessons_on_monday_l349_349262

theorem adam_lessons_on_monday :
  (∃ (time_monday time_tuesday time_wednesday : ℝ) (n_monday_lessons : ℕ),
    time_tuesday = 3 ∧
    time_wednesday = 2 * time_tuesday ∧
    time_monday + time_tuesday + time_wednesday = 12 ∧
    n_monday_lessons = time_monday / 0.5 ∧
    n_monday_lessons = 6) :=
by
  sorry

end adam_lessons_on_monday_l349_349262


namespace rainy_day_probability_l349_349379

theorem rainy_day_probability 
  (P_A : ℝ) (P_B : ℝ) (P_A_and_B : ℝ) 
  (h1 : P_A = 0.20) 
  (h2 : P_B = 0.18) 
  (h3 : P_A_and_B = 0.12) :
  P_A_and_B / P_A = 0.60 :=
sorry

end rainy_day_probability_l349_349379


namespace polynomial_value_at_8_l349_349111

noncomputable def P : ℝ → ℝ := λ x, x^2 - x + 2

theorem polynomial_value_at_8 :
  (P (P x) = x^4 - 2*x^3 + 4*x^2 - 3*x + 4) ∧ (P = λ x, x^2 - x + 2) → P 8 = 58 :=
by
  sorry

end polynomial_value_at_8_l349_349111


namespace pentagon_operation_terminates_l349_349727

theorem pentagon_operation_terminates :
  ∀ (a b c d e : ℤ),  
  a + b + c + d + e > 0 → 
  (∃ x y z, (x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e) ∧ 
            y < 0 ∧ 
            z = x ∧ z ≠ y ∧ (z = a ∨ z = b ∨ z = c ∨ z = d ∨ z = e)) →
  ∃ n, ∀ i, i > n → 
    let nums := [a, b, c, d, e] in
    let op (x y z: ℤ) := (x + y, -y, z + y) in
    ∀ (u v w x y z: ℤ), 
    ((u, v, w) = op (nums.nth i, nums.nth (i+1) % 5, nums.nth (i+2) % 5)) →
    u ≥ 0 ∧ v ≥ 0 ∧ w ≥ 0 :=
by
  sorry

end pentagon_operation_terminates_l349_349727


namespace range_of_g_l349_349483

noncomputable def g (x : ℝ) : ℝ := (Real.arccos x)^2 + (Real.arcsin x)^2

theorem range_of_g :
  (∀ x ∈ (-1:ℝ) .. 1, g x ∈ Set.Icc (π^2 / 8) (π^2 / 2)) ∧
  (∀ y ∈ Set.Icc (π^2 / 8) (π^2 / 2), ∃ x ∈ (-1:ℝ) .. 1, g x = y) :=
sorry

end range_of_g_l349_349483


namespace each_family_member_uses_mask_for_4_days_l349_349658

-- Define the conditions
def masks_in_package := 100
def days_to_finish_package := 80
def family_members := 5

-- Define the question as a proof statement
theorem each_family_member_uses_mask_for_4_days :
  (masks_in_package / family_members / days_to_finish_package)⁻¹ = 4 :=
by
  sorry

end each_family_member_uses_mask_for_4_days_l349_349658


namespace simplify_tan_sum_l349_349931

theorem simplify_tan_sum (x : ℝ) (h : ∀ θ : ℝ, cot θ - 2 * cot (2 * θ) = tan θ) : 
  tan x + 2 * tan (2 * x) + 4 * tan (4 * x) + 8 * tan (8 * x) = cot x :=
sorry

end simplify_tan_sum_l349_349931


namespace ellipse_eccentricity_l349_349769

-- Given the conditions
variable (a b : ℝ) (h_a_gt_b : a > b) (h_b_gt_0 : b > 0)

-- Converting the slope condition to corresponding condition on ellipse
variable (M : ℝ × ℝ) (M_on_ellipse : M.1^2 / a^2 + M.2^2 / b^2 = 1)
variable (A1 A2 : ℝ × ℝ)
variable (h_A1 : A1 = (-a, 0))
variable (h_A2 : A2 = (a, 0))
variable (h_slope : (M.2 / (M.1 + a)) * (M.2 / (M.1 - a)) = - (1 / 2))

-- Proven goal
theorem ellipse_eccentricity : 
  let b_sq := (a^2 / 2)
      e_sq := 1 - (b^2 / a^2)
  in a > b ∧ b > 0 ∧ M.1^2 / a^2 + M.2^2 / b_sq = 1 ∧ (M.2 / (M.1 + a)) * (M.2 / (M.1 - a)) = - 1 / 2 
     → sqrt e_sq = sqrt 2 / 2 :=
begin
  sorry
end

end ellipse_eccentricity_l349_349769


namespace chocolate_candy_pieces_l349_349212

-- Define the initial number of boxes and the boxes given away
def initial_boxes : Nat := 12
def boxes_given : Nat := 7

-- Define the number of remaining boxes
def remaining_boxes := initial_boxes - boxes_given

-- Define the number of pieces per box
def pieces_per_box : Nat := 6

-- Calculate the total pieces Tom still has
def total_pieces := remaining_boxes * pieces_per_box

-- State the theorem
theorem chocolate_candy_pieces : total_pieces = 30 :=
by
  -- proof steps would go here
  sorry

end chocolate_candy_pieces_l349_349212


namespace num_paths_7x6_avoid_3_3_l349_349824

theorem num_paths_7x6_avoid_3_3 : 
  let total_paths := Nat.choose 11 5,
      paths_to_33 := Nat.choose 6 3 * Nat.choose 5 2,
      valid_paths := total_paths - paths_to_33
  valid_paths = 262 :=
by
  let total_paths := Nat.choose 11 5
  let paths_to_33 := Nat.choose 6 3 * Nat.choose 5 2
  let valid_paths := total_paths - paths_to_33
  have h_total : total_paths = 462 := by simp
  have h_invalid : paths_to_33 = 200 := by simp
  rw [h_total, h_invalid]
  norm_num

end num_paths_7x6_avoid_3_3_l349_349824


namespace ellipse_ratio_squared_l349_349715

theorem ellipse_ratio_squared (a b c : ℝ) 
    (h1 : b / a = a / c) 
    (h2 : c^2 = a^2 - b^2) : (b / a)^2 = 1 / 2 :=
by
  sorry

end ellipse_ratio_squared_l349_349715


namespace quadratic_polynomial_value_l349_349105

theorem quadratic_polynomial_value (P : ℝ → ℝ) (hP : ∀ x, P(P(x)) = x^4 - 2*x^3 + 4*x^2 - 3*x + 4) :
  P 8 = 58 :=
sorry

end quadratic_polynomial_value_l349_349105


namespace number_of_primes_between_30_and_50_l349_349415

/-- 
  Prove that there are exactly 5 prime numbers in the range from 30 to 50. 
  These primes are 31, 37, 41, 43, and 47.
-/
theorem number_of_primes_between_30_and_50 : 
  (Finset.filter Nat.Prime (Finset.range 51)).card - 
  (Finset.filter Nat.Prime (Finset.range 31)).card = 5 := 
by 
  sorry

end number_of_primes_between_30_and_50_l349_349415


namespace expressions_for_a_b_T_n_expression_l349_349346

-- Defining the sequences and their conditions
def S : ℕ → ℝ := λ n, 2 * (n:ℝ) ^ 2 + n
def a (n : ℕ) : ℝ := if n = 1 then 3 else S n - S (n - 1)
def b (n : ℕ) : ℝ := 2 ^ n

-- Theorem stating the expressions for a_n and b_n
theorem expressions_for_a_b (n : ℕ) (hn : n > 0) :
  a n = 4 * n - 1 ∧ b n = 2 ^ n := by
  sorry

-- Defining the sum of the first n terms of the given sequences
def a_n_b_n_sum (n : ℕ) : ℝ :=
  ∑ k in finset.range n, a (k + 1) * b (k + 1)

-- Theorem stating the result for the sum T_n
theorem T_n_expression (n : ℕ) (hn : n > 0) :
  a_n_b_n_sum n = (4 * n - 5) * 2 ^ (n + 1) + 10 := by
  sorry

end expressions_for_a_b_T_n_expression_l349_349346


namespace solve_quartic_equation_l349_349137

theorem solve_quartic_equation :
  (∀ x : ℝ, -x^2 = (5 * x - 2) / (x - 2) - (x + 4) / (x + 2) → 
    (x = 3 ∨ x = -1 ∨ x = -1 + real.sqrt 5 ∨ x = -1 - real.sqrt 5)) := by 
  sorry

end solve_quartic_equation_l349_349137


namespace prime_count_30_to_50_l349_349399

def is_prime (n: ℕ) : Prop :=
  n > 1 ∧ ∀ m: ℕ, m ∣ n → m = 1 ∨ m = n

def primes_in_range (a b: ℕ) : list ℕ :=
  list.filter is_prime (list.range' a (b - a + 1))

theorem prime_count_30_to_50 : (primes_in_range 30 50).length = 5 :=
by sorry

end prime_count_30_to_50_l349_349399


namespace find_m_l349_349780

variable {a b c m : ℝ}

theorem find_m (h1 : a + b = 4)
               (h2 : a * b = m)
               (h3 : b + c = 8)
               (h4 : b * c = 5 * m) : m = 0 ∨ m = 3 :=
by {
  sorry
}

end find_m_l349_349780


namespace number_of_solutions_eq_9241_l349_349906

def floor (x : ℝ) : ℤ := Int.floor x
def frac (x : ℝ) : ℝ := x - floor x

theorem number_of_solutions_eq_9241 : 
  (∃ n : ℕ, ∑ x in (1..20), 3 * (n^2 + n)) + 1 = 9241 :=
sorry

end number_of_solutions_eq_9241_l349_349906


namespace shaded_area_equals_circle_area_l349_349926

noncomputable def rectangle_area (a b : ℝ) : ℝ :=
a * b

noncomputable def circle_area_diameter (d : ℝ) : ℝ :=
π * (d / 2) ^ 2

theorem shaded_area_equals_circle_area (a b : ℝ) :
  rectangle_area a b =
  circle_area_diameter (Real.sqrt (a ^ 2 + b ^ 2)) :=
by
  sorry

end shaded_area_equals_circle_area_l349_349926


namespace maximize_sum_of_sequence_l349_349957

-- Define the sequence term (general term)
noncomputable def sequence_term (k : Nat) : Real :=
  Real.log10 (1000 * (Real.cos (Real.pi / 3)) ^ k)

-- Define the sum of the first n terms of the sequence
noncomputable def sum_of_sequence (n : Nat) : Real :=
  (List.range n).sum (fun k => sequence_term k)

-- Prove n = 10 maximizes the sum of the first n terms
theorem maximize_sum_of_sequence : (∃ n : Nat, sum_of_sequence n = sum_of_sequence 10) :=
sorry

end maximize_sum_of_sequence_l349_349957


namespace largest_even_digit_multiple_of_five_l349_349575

theorem largest_even_digit_multiple_of_five : ∃ n : ℕ, n = 8860 ∧ n < 10000 ∧ (∀ digit ∈ (n.digits 10), digit % 2 = 0) ∧ n % 5 = 0 :=
by
  sorry

end largest_even_digit_multiple_of_five_l349_349575


namespace probability_odd_product_greater_than_15_l349_349135

/-
 Seven balls are numbered 1 through 7 and placed in a bowl. 
 Josh will randomly choose a ball from the bowl, look at its number, and then put it back into the bowl. 
 Then Josh will again randomly choose a ball from the bowl and look at its number. 
 Prove that the probability that the product of the two numbers will be odd and greater than 15 is 6/49.
-/
theorem probability_odd_product_greater_than_15 : 
  let balls := {1, 2, 3, 4, 5, 6, 7}
  let total_pairs := 49
  let valid_pairs := [(3,7), (5,5), (5,7), (7,3), (7,5), (7,7)]
  |valid_pairs| = 6 →
  let prob := (|valid_pairs| : ℚ) / (total_pairs : ℚ)
  in prob = 6/49 :=
by
  sorry

end probability_odd_product_greater_than_15_l349_349135


namespace true_propositions_l349_349590

open Set

theorem true_propositions (M N : Set ℕ) (a b m : ℕ) (h1 : M ⊆ N) 
  (h2 : a > b) (h3 : b > 0) (h4 : m > 0) (p : ∀ x : ℝ, x > 0) :
  (M ⊆ M ∪ N) ∧ ((b + m) / (a + m) > b / a) ∧ 
  ¬(∀ (a b c : ℝ), a = b ↔ a * c ^ 2 = b * c ^ 2) ∧ 
  ¬(∃ x₀ : ℝ, x₀ ≤ 0) := sorry

end true_propositions_l349_349590


namespace base_7_to_base_10_equiv_l349_349252

theorem base_7_to_base_10_equiv : 
  ∀ (d2 d1 d0 : ℕ), 
      d2 = 3 → d1 = 4 → d0 = 6 → 
      (d2 * 7^2 + d1 * 7^1 + d0 * 7^0) = 181 := 
by 
  sorry

end base_7_to_base_10_equiv_l349_349252


namespace right_triangle_AC_length_l349_349859

-- Define the right triangle and given conditions
variable (A B C : Type) [metric_space A] [metric_space B] [metric_space C]
variable (dist : A → B → ℝ) [non_empty A] [non_empty B] [non_empty C]

-- Hypotheses
variables (AB AC BC : ℝ)
variables (hC : ∠ C = 90) (hAB : AB = 10) (hBC : BC = 8)

-- Define the main theorem to be proven
theorem right_triangle_AC_length (h : AB^2 = AC^2 + BC^2) : AC = 6 :=
by 
  -- Use the Pythagorean theorem
  have h1 : AB^2 = 10^2 := by rw hAB
  have h2 : BC^2 = 8^2 := by rw hBC
  have h3 : AC^2 + 64 = 100 := by {rw[h1, h2], assumption}
  have h4 : AC^2 = 36 := by linarith
  have h5 : AC = 6 := by linarith
  exact sorry

end right_triangle_AC_length_l349_349859


namespace smallest_positive_period_and_symmetry_l349_349374

noncomputable def f (x : ℝ) : ℝ := 
  Real.sin (x + (7 * Real.pi / 4)) + 
  Real.cos (x - (3 * Real.pi / 4))

theorem smallest_positive_period_and_symmetry :
  (∃ T > 0, T = 2 * Real.pi ∧ ∀ x, f (x + T) = f x) ∧ 
  (∃ a, a = - (Real.pi / 4) ∧ ∀ x, f (2 * a - x) = f x) :=
by
  sorry

end smallest_positive_period_and_symmetry_l349_349374


namespace number_of_people_in_house_l349_349198

theorem number_of_people_in_house :
  let bedroom := 2 + (1 + 4) in
  let living_room := 8 in
  let kitchen := living_room + 3 in
  let garage := kitchen / 2 in
  let patio := garage * 2 in
  bedroom + living_room + kitchen + garage + patio = 41 :=
by
  sorry

end number_of_people_in_house_l349_349198


namespace find_coordinates_of_vector_OA_l349_349784

noncomputable def vector_OA (O A : ℝ × ℝ) (h : O = (0, 0)) 
  (H1 : |(A.1, A.2)| = 2) 
  (H2 : 150 * π / 180 * 2)
  (H3 : A.1 ≤ 0) 
  (H4 : A.2 ≥ 0) : ℝ × ℝ :=
(A.1, A.2)

theorem find_coordinates_of_vector_OA (O A : ℝ × ℝ) (h : O = (0, 0))
  (H1 : |(A.1, A.2)| = 2) 
  (H2 : 150 * π / 180 * 2)
  (H3 : A.1 ≤ 0) 
  (H4 : A.2 ≥ 0) :
  vector_OA O A h H1 H2 H3 H4 = (-√3, 1) :=
sorry

end find_coordinates_of_vector_OA_l349_349784


namespace log_function_fixed_point_l349_349361

theorem log_function_fixed_point :
  ∃ a : ℝ, ∀ x : ℝ, (f(x) = log a (x-1) + 1) → (f(2) = 1) :=
by
  sorry

end log_function_fixed_point_l349_349361


namespace circle_center_and_radius_l349_349804

-- Definition of the polar equation as a condition
def polar_equation (θ ρ : ℝ) : Prop := 
  ρ = 2 * cos θ

-- The main theorem which states the Cartesian coordinates of the circle's center and its radius
theorem circle_center_and_radius :
  (∀ θ ρ, polar_equation θ ρ → ∃ x y r, ρ^2 = x^2 + y^2 ∧ x^2 + y^2 = 2 * x ∧ (x - 1)^2 + y^2 = r^2) →
  ∃ x y r, (x = 1 ∧ y = 0 ∧ r = 1) :=
by
  assume h,
  use [1, 0, 1],
  sorry

end circle_center_and_radius_l349_349804


namespace number_of_primes_between_30_and_50_l349_349418

/-- 
  Prove that there are exactly 5 prime numbers in the range from 30 to 50. 
  These primes are 31, 37, 41, 43, and 47.
-/
theorem number_of_primes_between_30_and_50 : 
  (Finset.filter Nat.Prime (Finset.range 51)).card - 
  (Finset.filter Nat.Prime (Finset.range 31)).card = 5 := 
by 
  sorry

end number_of_primes_between_30_and_50_l349_349418


namespace cannot_obtain_sums_l349_349950

def cards : List ℕ := [2, 2, 5, 5, 8, 9]

theorem cannot_obtain_sums : (finset.range 32).filter (λ s, ¬ ∃ (vals : multiset ℕ), vals ⊆ cards ∧ vals.sum = s).card = 6 :=
by
  sorry

end cannot_obtain_sums_l349_349950


namespace marcus_goal_points_value_l349_349911

-- Definitions based on conditions
def marcus_goals_first_type := 5
def marcus_goals_second_type := 10
def second_type_goal_points := 2
def team_total_points := 70
def marcus_percentage_points := 50

-- Theorem statement
theorem marcus_goal_points_value : 
  ∃ (x : ℕ), 5 * x + 10 * 2 = 35 ∧ 35 = 50 * team_total_points / 100 := 
sorry

end marcus_goal_points_value_l349_349911


namespace rate_percent_is_10_l349_349188

theorem rate_percent_is_10
  (SI : ℕ) (P : ℕ) (T : ℕ) (R : ℕ) 
  (h1 : SI = 2500) (h2 : P = 5000) (h3 : T = 5) :
  R = 10 :=
by
  sorry

end rate_percent_is_10_l349_349188


namespace systematic_sampling_first_group_l349_349636

theorem systematic_sampling_first_group
  (total_groups : ℕ)
  (interval : ℕ)
  (number_17th_group : ℕ)
  (h1 : total_groups = 20)
  (h2 : interval = 7)
  (h3 : number_17th_group = 117) :
  let x := 5 in x + interval * (17 - 1) = number_17th_group :=
by
  intros
  simp [h2, h3]
  sorry

end systematic_sampling_first_group_l349_349636


namespace binomial_10_2_l349_349690

noncomputable def binom (n k : ℕ) : ℕ := n! / (k! * (n - k)!)

theorem binomial_10_2 : binom 10 2 = 45 := by
  sorry

end binomial_10_2_l349_349690


namespace grid_blue_probability_l349_349282

-- Define the problem in Lean
theorem grid_blue_probability :
  let n := 4
  let p_tile_blue := 1 / 2
  let invariant_prob := (p_tile_blue ^ (n / 2))
  let pair_prob := (p_tile_blue * p_tile_blue)
  let total_pairs := (n * n / 2 - n / 2)
  let final_prob := (invariant_prob ^ 2) * (pair_prob ^ total_pairs)
  final_prob = 1 / 65536 := by
  sorry

end grid_blue_probability_l349_349282


namespace exists_large_independent_set_l349_349902

-- Definitions:
variable (G : Type) [Fintype G] [DecidableEq G] (adj : G → G → Prop) [Symmetric adj] [Irreflexive adj] 
variable (d : G → ℕ) -- degree function

-- Define a vertex being part of an independent set
def isIndependent (A : Finset G) : Prop :=
  ∀ {x y : G}, x ∈ A → y ∈ A → adj x y → False 

-- Define the sum of 1/(1 + degree) for all vertices in the graph
def sumInvDegreePlusOne : ℚ :=
  ∑ x in (Finset.univ : Finset G), (1 : ℚ) / (1 + d x)

-- The theorem
theorem exists_large_independent_set :
  ∃ (A : Finset G), isIndependent G adj A ∧ (A.card : ℚ) ≥ sumInvDegreePlusOne G d := 
by 
  sorry -- Proof not required here

end exists_large_independent_set_l349_349902


namespace polar_to_cartesian_l349_349286

theorem polar_to_cartesian (r θ : ℝ) (h_r : r = 2) (h_θ : θ = π / 6) :
  (r * Real.cos θ, r * Real.sin θ) = (Real.sqrt 3, 1) :=
by
  rw [h_r, h_θ]
  have h_cos : Real.cos (π / 6) = Real.sqrt 3 / 2 := sorry -- This identity can be used from trigonometric property.
  have h_sin : Real.sin (π / 6) = 1 / 2 := sorry -- This identity can be used from trigonometric property.
  rw [h_cos, h_sin]
  -- some algebraic steps to simplifiy left sides to (Real.sqrt 3, 1) should follow here. using multiplication and commmutaivity properties mainly.
  sorry

end polar_to_cartesian_l349_349286


namespace smallest_m_condition_l349_349328

-- Define the function f_9
def f_9 (n : ℕ) : ℕ := (List.range' 1 9).count (λ d, d ∣ n)

-- The main theorem statement
theorem smallest_m_condition (m : ℕ) (b : ℕ → ℝ) (h : ∀ n > m, f_9 n = ∑ j in Finset.range(m), b j * f_9 (n - j)) : m = 28 :=
sorry

end smallest_m_condition_l349_349328


namespace area_difference_l349_349827

noncomputable def radius_larger_circle : ℝ := 10
noncomputable def diameter_smaller_circle : ℝ := 10
noncomputable def radius_smaller_circle : ℝ := diameter_smaller_circle / 2

def area_circle (r : ℝ) : ℝ := π * r^2

theorem area_difference :
  area_circle radius_larger_circle - area_circle radius_smaller_circle = 75 * π :=
by
  sorry

end area_difference_l349_349827


namespace equation_pattern_l349_349075

theorem equation_pattern (n : ℕ) (h : n = 999999) : n^2 = (n + 1) * (n - 1) + 1 :=
by
  sorry

end equation_pattern_l349_349075


namespace total_candies_l349_349910

theorem total_candies (Linda_candies Chloe_candies Michael_candies : ℕ) (h1 : Linda_candies = 340) (h2 : Chloe_candies = 280) (h3 : Michael_candies = 450) :
  Linda_candies + Chloe_candies + Michael_candies = 1070 :=
by
  subst h1
  subst h2
  subst h3
  simp
  sorry

end total_candies_l349_349910


namespace range_of_a_l349_349446

theorem range_of_a (a : ℝ) :
  (-1 < x ∧ x < 0 → (x^2 - a * x + 2 * a) > 0) ∧
  (0 < x → (x^2 - a * x + 2 * a) < 0) ↔ -1 / 3 < a ∧ a < 0 :=
sorry

end range_of_a_l349_349446


namespace range_of_a_three_log_x1_plus_log_x2_l349_349795

-- Definition of the function f(x)
def f (x : ℝ) (a : ℝ) : ℝ := x * Real.log x - a * x^3 - x

-- Derivative of the function f(x)
def f_prime (x a : ℝ) : ℝ := Real.log x - 3 * a * x^2

-- Condition: a must be < 1/(6e) for the function to have a monotonically increasing interval
theorem range_of_a (a : ℝ) : a < 1 / (6 * Real.exp 1) → ∃ x, f_prime x a > 0 := by
  sorry

-- Given f'(x₁) = 0, f'(x₂) = 0 and x₁ < x₂
variables {a x₁ x₂ : ℝ}
hypothesis h1 : f_prime x₁ a = 0
hypothesis h2 : f_prime x₂ a = 0
hypothesis h3 : x₁ < x₂

-- Prove that 3 * ln x₁ + ln x₂ > 1
theorem three_log_x1_plus_log_x2 (x₁ x₂ a : ℝ) (h1 : f_prime x₁ a = 0) (h2 : f_prime x₂ a = 0) (h3 : x₁ < x₂) : 
  3 * Real.log x₁ + Real.log x₂ > 1 := by
  sorry

end range_of_a_three_log_x1_plus_log_x2_l349_349795


namespace proof_problem_l349_349801

noncomputable def f (x : ℝ) : ℝ := Real.exp x + 1 / x

noncomputable def f'' (x : ℝ) : ℝ := Real.exp x + 2 / x^3

theorem proof_problem {x0 m n : ℝ} (hx0_pos : 0 < x0)
  (H : f'' x0 = 0) (hm : 0 < m) (hmx0 : m < x0) (hn : x0 < n) :
  f'' m < 0 ∧ f'' n > 0 := sorry

end proof_problem_l349_349801


namespace number_of_primes_between_30_and_50_l349_349430

-- Define what it means to be a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the interval condition
def in_interval (n : ℕ) : Prop :=
  30 ≤ n ∧ n ≤ 50

-- Define the count of prime numbers in the interval
def prime_count_in_interval : ℕ :=
  (List.range' 30 21).countp (λ n, is_prime n)

-- We state that the above count is equal to 5
theorem number_of_primes_between_30_and_50 : prime_count_in_interval = 5 :=
  sorry

end number_of_primes_between_30_and_50_l349_349430


namespace long_side_length_l349_349714

variable {a b d : ℝ}

theorem long_side_length (h1 : a / b = 2 * (b / d)) (h2 : a = 4) (hd : d = Real.sqrt (a^2 + b^2)) :
  b = Real.sqrt (2 + 4 * Real.sqrt 17) :=
sorry

end long_side_length_l349_349714


namespace die_roll_probability_l349_349644

theorem die_roll_probability :
  let rolls := 8,
      die_faces := 6,
      prime_odd_rolls := {3, 5},
      favorable_cases := prime_odd_rolls.size,
      probability_single_roll := favorable_cases / die_faces,
      probability_eight_rolls := probability_single_roll ^ rolls
  in probability_eight_rolls = 1 / 6561 :=
by
  let rolls := 8
  let die_faces := 6
  let prime_odd_rolls := {3, 5}
  let favorable_cases := prime_odd_rolls.size
  let probability_single_roll := favorable_cases / die_faces
  let probability_eight_rolls := probability_single_roll ^ rolls
  show probability_eight_rolls = 1 / 6561
  sorry

end die_roll_probability_l349_349644


namespace first_term_geometric_sequence_l349_349556

theorem first_term_geometric_sequence (a r : ℚ) 
    (h1 : a * r^2 = 8) 
    (h2 : a * r^4 = 27 / 4) : 
    a = 256 / 27 :=
by sorry

end first_term_geometric_sequence_l349_349556


namespace area_of_hexagon_l349_349984

theorem area_of_hexagon (c d : ℝ) (a b : ℝ)
  (h₁ : a^2 + b^2 = c^2)
  (h₂ : a + b = d) : 
  (c^2 + d^2 = c^2 + a^2 + b^2 + 2*a*b) :=
by
  sorry

end area_of_hexagon_l349_349984


namespace positive_difference_l349_349582

def a : ℕ := (8^2 - 8^2) / 8
def b : ℕ := (8^2 * 8^2) / 8

theorem positive_difference : |b - a| = 512 :=
by
  sorry

end positive_difference_l349_349582


namespace num_positive_is_one_l349_349908

theorem num_positive_is_one (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a * b * c = 1) : 
  ( (0 < a) + (0 < b) + (0 < c) = 1) :=
sorry

end num_positive_is_one_l349_349908


namespace breath_holding_factor_l349_349276

theorem breath_holding_factor :
  (∃ F : ℝ, 30 * F = 60) → ∃ F : ℝ, F = 2 :=
by
  intro h
  cases h with F hF
  use 2
  linarith

end breath_holding_factor_l349_349276


namespace exists_unique_element_in_sequence_l349_349889

theorem exists_unique_element_in_sequence (a : ℕ → ℝ) (h_inc : ∀ n, a n < a (n + 1)) (h_interval : ∀ n, 0 < a n ∧ a n < 1) :
  ∃ x, x ∈ (λ i, a i / i : ℕ → ℝ) '' (Set.univ) ∧ ∀ j k, j ≠ k → (a j / j) ≠ (a k / k) :=
by
  sorry

end exists_unique_element_in_sequence_l349_349889


namespace binomial_10_2_equals_45_l349_349676

open Nat

theorem binomial_10_2_equals_45 : Nat.choose 10 2 = 45 := 
by
  sorry

end binomial_10_2_equals_45_l349_349676


namespace count_primes_between_30_and_50_l349_349427

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between_30_and_50 : List ℕ :=
  [31, 37, 41, 43, 47]

theorem count_primes_between_30_and_50 : 
  (primes_between_30_and_50.filter is_prime).length = 5 :=
by
  sorry

end count_primes_between_30_and_50_l349_349427


namespace amount_each_student_should_pay_l349_349546

noncomputable def total_rental_fee_per_book_per_half_hour : ℕ := 4000 
noncomputable def total_books : ℕ := 4
noncomputable def total_students : ℕ := 6
noncomputable def total_hours : ℕ := 3
noncomputable def total_half_hours : ℕ := total_hours * 2

noncomputable def total_fee_one_book : ℕ := total_rental_fee_per_book_per_half_hour * total_half_hours
noncomputable def total_fee_all_books : ℕ := total_fee_one_book * total_books

theorem amount_each_student_should_pay : total_fee_all_books / total_students = 16000 := by
  sorry

end amount_each_student_should_pay_l349_349546


namespace logs_in_stack_l349_349255

theorem logs_in_stack (a l : ℕ) (h1 : a = 15) (h2 : l = 4) (h3 : 4 ≤ 15) :
  let n := a - l + 1 in
  let avg := (a + l) / 2 in
  let sum := avg * n in
  sum = 114 := by
  sorry

end logs_in_stack_l349_349255


namespace find_B_calculate_area_l349_349873

noncomputable theory

variables (a b c : ℝ)
variables (A B C : ℝ)

def satisfies_equation : Prop :=
  (b * Real.cos C + c * Real.cos B) / 2 = (Real.sqrt 3 / 3) * a * Real.cos B

def correct_angle (B : ℝ) : Prop :=
  B = Real.pi / 6

def correct_area (b c : ℝ) (a : ℝ) : Prop :=
  let A := Real.pi / 3 in
  1 / 2 * b * c * Real.sin A = (3 * Real.sqrt 7) / 2

theorem find_B (h1 : satisfies_equation a b c A B C) :
  correct_angle B := sorry

theorem calculate_area (h1 : satisfies_equation a b c A B C) (hb : b = Real.sqrt 7)
  (hc : c = 2 * Real.sqrt 3) (ha : a > b) :
  ∃ (a : ℝ), correct_area b c a := sorry

end find_B_calculate_area_l349_349873


namespace positive_difference_of_expressions_l349_349577

theorem positive_difference_of_expressions :
  let a := 8
  let expr1 := (a^2 - a^2) / a
  let expr2 := (a^2 * a^2) / a
  expr1 = 0 → expr2 = 512 → 512 - 0 = 512 := 
by
  introv h_expr1 h_expr2
  rw [h_expr1, h_expr2]
  norm_num
  exact rfl

end positive_difference_of_expressions_l349_349577


namespace total_distance_correct_l349_349273

def day1_distance : ℕ := (5 * 4) + (3 * 2) + (4 * 3)
def day2_distance : ℕ := (6 * 3) + (2 * 1) + (6 * 3) + (3 * 4)
def day3_distance : ℕ := (4 * 2) + (2 * 1) + (7 * 3) + (5 * 2)

def total_distance : ℕ := day1_distance + day2_distance + day3_distance

theorem total_distance_correct :
  total_distance = 129 := by
  sorry

end total_distance_correct_l349_349273


namespace kerosene_cost_l349_349849

theorem kerosene_cost (A B C : ℝ)
  (h1 : A = B)
  (h2 : C = A / 2)
  (h3 : C * 2 = 24 / 100) :
  24 = 24 := 
sorry

end kerosene_cost_l349_349849


namespace odd_function_extremes_l349_349159

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x

theorem odd_function_extremes (b c d m : ℝ) (h_odd : ∀ x : ℝ, f (-x) = -f x) 
  (h_extreme : f'(-1) = 0 ∧ f'(1) = 0)
  (h_interval : -1 < m):
  (∀ x : ℝ, f x = x^3 - 3 * x) ∧ 
  ((-1 < m ∧ m ≤ 1) → (∃ f_min f_max, f_min = m^3 - 3 * m ∧ f_max = 2)) ∧
  ((1 < m ∧ m ≤ 2) → (∃ f_min f_max, f_min = -2 ∧ f_max = 2)) ∧
  ((m > 2) → (∃ f_min f_max, f_min = -2 ∧ f_max = m^3 - 3 * m)) := 
by 
  sorry

end odd_function_extremes_l349_349159


namespace problem_ACD_correct_l349_349058

variables {Ω : Type*} [probability_space Ω]
variables {A B : set Ω}

def mutually_exclusive (A B : set Ω) : Prop := disjoint A B
def complementary (A B : set Ω) : Prop := A = Bᶜ

theorem problem_ACD_correct 
  (A B : set Ω) 
  [measure_space Ω]
  (P : measure_theory.measure Ω)
  (hA_measurable : measurable_set A)
  (hB_measurable : measurable_set B) 
  :
  (mutually_exclusive A B → P (A ∪ B) = P A + P B) ∧
  (complementary A B → P (A ∪ B) = P A + P B) ∧
  (complementary A B → P A + P B = 1) := 
by 
  sorry

#check problem_ACD_correct

end problem_ACD_correct_l349_349058


namespace number_properties_l349_349553

theorem number_properties (a b x : ℝ) 
  (h1 : a + b = 40) 
  (h2 : a * b = 375) 
  (h3 : a - b = x) : 
  (a = 25 ∧ b = 15 ∧ x = 10) ∨ (a = 15 ∧ b = 25 ∧ x = 10) :=
by
  sorry

end number_properties_l349_349553


namespace P_P_eq_P_eight_equals_58_l349_349090

open Polynomial

noncomputable def P(x : ℚ) : ℚ := x^2 - x + 2

theorem P_P_eq :
  (P ∘ P)(x) = x^4 - 2 * x^3 + 4 * x^2 - 3 * x + 4 := sorry

theorem P_eight_equals_58 :
  P 8 = 58 := sorry

end P_P_eq_P_eight_equals_58_l349_349090


namespace downstream_rate_l349_349243

theorem downstream_rate (rate_in_still_water : ℝ) (rate_of_current : ℝ) : 
  rate_in_still_water = 34 → 
  rate_of_current = 11 → 
  rate_in_still_water + rate_of_current = 45 :=
by
  intros h1 h2
  rw [h1, h2]
  rfl

end downstream_rate_l349_349243


namespace sweet_numbers_count_l349_349998

def triple_or_subtract (n : ℕ) : ℕ :=
if n <= 25 then 3 * n else n - 15

def sweet_number (F : ℕ) : Prop :=
¬ ∃ n, (triple_or_subtract^[n] F = 18)

def sweet_numbers (range : ℕ) : Finset ℕ :=
(Finset.range range).filter sweet_number

theorem sweet_numbers_count : (sweet_numbers 50).card = 7 := 
sorry

end sweet_numbers_count_l349_349998


namespace students_in_both_clubs_l349_349272

variables (Total Students RoboticClub ScienceClub EitherClub BothClubs : ℕ)

theorem students_in_both_clubs
  (h1 : Total = 300)
  (h2 : RoboticClub = 80)
  (h3 : ScienceClub = 130)
  (h4 : EitherClub = 190)
  (h5 : EitherClub = RoboticClub + ScienceClub - BothClubs) :
  BothClubs = 20 :=
by
  sorry

end students_in_both_clubs_l349_349272


namespace comb_divisible_by_p_cube_l349_349081

theorem comb_divisible_by_p_cube (p : ℕ) (hp_prime : p.prime) (hp_gt_3 : p > 3) :
  (Nat.choose (2 * p - 1) (p - 1) - 1) % (p^3) = 0 := 
sorry

end comb_divisible_by_p_cube_l349_349081


namespace hearty_beads_count_l349_349386

theorem hearty_beads_count :
  let blue_packages := 3
  let red_packages := 5
  let beads_per_package := 40
  let total_beads := blue_packages * beads_per_package + red_packages * beads_per_package
  total_beads = 320 :=
by
  let blue_packages := 3
  let red_packages := 5
  let beads_per_package := 40
  let total_beads := blue_packages * beads_per_package + red_packages * beads_per_package
  show total_beads = 320
  sorry

end hearty_beads_count_l349_349386


namespace correct_operation_l349_349995

theorem correct_operation :
  sqrt 3 * real.sin (60 * real.pi / 180) = 3 / 2 :=
by
  sorry

end correct_operation_l349_349995


namespace total_clothes_count_l349_349009

theorem total_clothes_count (shirts_per_pants : ℕ) (pants : ℕ) (shirts : ℕ) : shirts_per_pants = 6 → pants = 40 → shirts = shirts_per_pants * pants → shirts + pants = 280 := by
  intro h1 h2 h3
  rw [h1, h2] at h3
  rw [h3]
  sorry

end total_clothes_count_l349_349009


namespace coefficient_of_x4_in_expansion_l349_349184

theorem coefficient_of_x4_in_expansion : 
  let x := 0 : ℝ;
  let a := x + 3 * real.sqrt 2;
  9.choose 5 * (3 * real.sqrt 2)^5 = 122472 * real.sqrt 2 :=
by
  sorry

end coefficient_of_x4_in_expansion_l349_349184


namespace vendor_calculations_vendor_profit_l349_349651

-- The conditions for the problem
variables (x y : ℝ)
variable h1 : x + y = 40
variable h2 : 3 * x + 2.4 * y = 114

-- The proof problems
theorem vendor_calculations : x = 30 ∧ y = 10 :=
by
  sorry

theorem vendor_profit : (5 - 3) * 30 + (4 - 2.4) * 10 = 76 :=
by
  sorry

end vendor_calculations_vendor_profit_l349_349651


namespace min_value_of_expression_l349_349496

theorem min_value_of_expression (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_abc : a * b * c = 4) :
  (3 * a + b) * (2 * b + 3 * c) * (a * c + 4) ≥ 384 := 
by sorry

end min_value_of_expression_l349_349496


namespace translate_parabola_l349_349564

theorem translate_parabola :
  (∀ x : ℝ, (y : ℝ) = 6 * x^2 -> y = 6 * (x + 2)^2 + 3) :=
by
  sorry

end translate_parabola_l349_349564


namespace total_pieces_of_clothes_l349_349011

theorem total_pieces_of_clothes (shirts_per_pant pants : ℕ) (h1 : shirts_per_pant = 6) (h2 : pants = 40) : 
  shirts_per_pant * pants + pants = 280 :=
by
  rw [h1, h2]
  sorry

end total_pieces_of_clothes_l349_349011


namespace ellipse_equation_and_sum_of_xcoords_is_constant_l349_349349

open Real

noncomputable def ellipse := {x : ℝ × ℝ | ∃ (a b: ℝ) (h1: a > 0) (h2: b > 0), a > b ∧
                             x.2 = sqrt 3 ∧
                             (a^2 - b^2 = b^2 * (ecc / (1 - ecc^2))) ∧
                             ecc = 1 / 2 ∧
                             ((x.1 ^ 2) / (a^2) + (x.2 ^ 2) / (b^2) = 1)}

theorem ellipse_equation_and_sum_of_xcoords_is_constant
  {P: ℝ × ℝ} {M N: ℝ × ℝ}
  (hP: P ∈ ellipse)
  (hM: M ∈ ellipse)
  (hN: N ∈ ellipse)
  (h_slope_product: (M.2 - P.2) / (M.1 - P.1) * (N.2 - P.2) / (N.1 - P.1) = -3 / 4) :
  (∃ (a b: ℝ) (h1: a > 0) (h2: b > 0), a > b ∧
   b = sqrt 3 ∧ a = 2 ∧ ecc = (1 / 2) ∧
   ((x.1 ^ 2) / (a^2) + (x.2 ^ 2) / (b^2) = 1)) ∧
  M.1 + N.1 = 0 := 
sorry

end ellipse_equation_and_sum_of_xcoords_is_constant_l349_349349


namespace f_at_2014_l349_349071

variables {a b c x : ℝ}

-- Given conditions
def f (x : ℝ) : ℝ :=
  (c * (x - a) * (x - b)) / ((c - a) * (c - b)) +
  (a * (x - b) * (x - c)) / ((a - b) * (a - c)) +
  (b * (x - c) * (x - a)) / ((b - c) * (b - a))

-- Main theorem to be proved
theorem f_at_2014 (h1 : a < b) (h2 : b < c) :
  f(2014) = 2014 :=
sorry

end f_at_2014_l349_349071


namespace monotonic_condition_l349_349541

noncomputable def f (a x : ℝ) := x^2 - 2 * a * x

theorem monotonic_condition (a : ℝ) (h : a ≤ -1) : 
  ∀ x y ∈ Icc (-1 : ℝ) (1 : ℝ), (f a x) ≤ (f a y) ∨ (f a y) ≤ (f a x) :=
sorry

end monotonic_condition_l349_349541


namespace Taimour_paint_time_l349_349046

theorem Taimour_paint_time (T : ℝ) (H1 : ∀ t : ℝ, t = 2 / T → t ≠ 0) (H2 : (1 / T + 2 / T) = 1 / 3) : T = 9 :=
by
  sorry

end Taimour_paint_time_l349_349046


namespace quadratic_P_value_l349_349121

noncomputable def P (x : ℝ) : ℝ :=
  x^2 - x + 2

theorem quadratic_P_value :
  P (P 8) = 8^4 - 2 * 8^3 + 4 * 8^2 - 3 * 8 + 4 ∧ P 8 = 58 := 
by
  have h1 : P(P(8)) = 8^4 - 2 * 8^3 + 4 * 8^2 - 3 * 8 + 4 := sorry
  have h2 : P(8) = 58 := sorry
  exact ⟨h1, h2⟩  

end quadratic_P_value_l349_349121


namespace modulus_of_complex_l349_349368

open Complex

theorem modulus_of_complex (z : ℂ) (h : (1 + z) / (1 - z) = ⟨0, 1⟩) : abs z = 1 := 
sorry

end modulus_of_complex_l349_349368


namespace prime_count_30_to_50_l349_349400

def is_prime (n: ℕ) : Prop :=
  n > 1 ∧ ∀ m: ℕ, m ∣ n → m = 1 ∨ m = n

def primes_in_range (a b: ℕ) : list ℕ :=
  list.filter is_prime (list.range' a (b - a + 1))

theorem prime_count_30_to_50 : (primes_in_range 30 50).length = 5 :=
by sorry

end prime_count_30_to_50_l349_349400


namespace problem_prob_abc_ab_a_div_by_4_l349_349922

theorem problem_prob_abc_ab_a_div_by_4 :
  ∃ (a b c : ℕ), (1 ≤ a ∧ a ≤ 5000) ∧ (1 ≤ b ∧ b ≤ 5000) ∧ (1 ≤ c ∧ c ≤ 5000) →
  (P : ℚ) = 25 / 64 :=
begin
  sorry
end

end problem_prob_abc_ab_a_div_by_4_l349_349922


namespace num_unique_four_digit_numbers_l349_349818

theorem num_unique_four_digit_numbers : 
  let digits := [3, 0, 3, 3] in
  let valid_numbers := {n : ℕ | n / 1000 = 3 ∧ ∃ (h t u d : ℕ), n = h * 1000 + t * 100 + u * 10 + d ∧ h = 3 ∧ {t, u, d} = {0, 3, 3}} in
  ∃ (n : ℕ), n ∈ valid_numbers ∧ valid_numbers.finite ∧ valid_numbers.card = 1 :=
by 
  sorry

end num_unique_four_digit_numbers_l349_349818


namespace problem_given_conditions_intersection_eq_union_eq_complement_eq_difference_AB_eq_difference_A_A_minus_B_eq_l349_349807

open Set

variable (U : Set ℝ) (A : Set ℝ) (B : Set ℝ)
theorem problem_given_conditions :
  (U = univ) ∧ 
  (A = {x | 4 < x}) ∧
  (B = {x | -6 < x ∧ x < 6}) :=
by
  tauto

theorem intersection_eq : A ∩ B = {x | 4 < x ∧ x < 6} :=
sorry

theorem union_eq : A ∪ B = {x | 4 <  x ∨ (-6 < x ∧ x < 6)} :=
sorry

theorem complement_eq : U \ B = {x | x ≤ -6 ∨ x ≥ 6} :=
sorry

def A_minus_B := {x | x ∈ A ∧ x ∉ B}
theorem difference_AB_eq : A_minus_B = {x | x ≥ 6} :=
sorry

def A_minus_A_minus_B := {x | x ∈ A ∧ x ∉ A_minus_B}
theorem difference_A_A_minus_B_eq : A_minus_A_minus_B = {x | 4 < x ∧ x < 6} :=
sorry

end problem_given_conditions_intersection_eq_union_eq_complement_eq_difference_AB_eq_difference_A_A_minus_B_eq_l349_349807


namespace minimum_value_exists_l349_349290

-- Definitions of the components
noncomputable def quadratic_expression (k x y : ℝ) : ℝ := 
  9 * x^2 - 12 * k * x * y + (4 * k^2 + 3) * y^2 - 6 * x - 9 * y + 12

theorem minimum_value_exists (k : ℝ) :
  (∃ x y : ℝ, quadratic_expression k x y = 0) ↔ k = 2 := 
sorry

end minimum_value_exists_l349_349290


namespace right_triangle_of_given_area_condition_l349_349925

noncomputable def semiPerimeter (a b c : ℝ) : ℝ := (a + b + c) / 2
noncomputable def area (s a b c : ℝ) : ℝ := sqrt (s * (s - a) * (s - b) * (s - c))

theorem right_triangle_of_given_area_condition (a b c : ℝ)
    (h_area_eq : (semiPerimeter a b c - a) * (semiPerimeter a b c - c) = area (semiPerimeter a b c) a b c) : 
    b^2 = a^2 + c^2 := 
sorry

end right_triangle_of_given_area_condition_l349_349925


namespace man_l349_349630

theorem man's_rate_in_still_water (
  with_stream: ℝ,
  against_stream: ℝ
) (h1: with_stream = 24) (h2: against_stream = 10) : ∃ (rate: ℝ), rate = 17 :=
by
  sorry

end man_l349_349630


namespace solution_correct_l349_349733

noncomputable def satisfies_conditions (f : ℤ → ℝ) : Prop :=
  (f 1 = 5 / 2) ∧ (f 0 ≠ 0) ∧ (∀ m n : ℤ, f m * f n = f (m + n) + f (m - n))

theorem solution_correct (f : ℤ → ℝ) :
  satisfies_conditions f → ∀ n : ℤ, f n = 2^n + (1/2)^n :=
by sorry

end solution_correct_l349_349733


namespace infinite_power_tower_eq_four_l349_349539

-- Define the infinite power tower
noncomputable def infinite_power_tower (x : ℝ) : ℝ :=
  real.pow x (infinite_power_tower x)

-- Given conditions
noncomputable def y : ℝ := infinite_power_tower (real.sqrt 2)
noncomputable def x : ℝ := real.sqrt 2

-- Theorem to be proven
theorem infinite_power_tower_eq_four : infinite_power_tower x = 4 :=
sorry

end infinite_power_tower_eq_four_l349_349539


namespace right_triangles_with_specific_area_and_perimeter_l349_349822

theorem right_triangles_with_specific_area_and_perimeter :
  ∃ (count : ℕ),
    count = 7 ∧
    ∀ (a b : ℕ), 
      (a > 0 ∧ b > 0 ∧ (a ≠ b) ∧ (a^2 + b^2 = c^2) ∧ (a * b / 2 = 5 * (a + b + c))) → 
      count = 7 :=
by
  sorry

end right_triangles_with_specific_area_and_perimeter_l349_349822


namespace length_of_DB_l349_349033

-- Definitions of the conditions in the problem.
variable (AC AD : ℝ)
variable (angle_ABC_right angle_ADB_right : Prop)

-- Given values from the problem.
def AC_value : ℝ := 21
def AD_value : ℝ := 7

-- Definition of the length of segment DC.
def DC := AC_value - AD_value

-- Similarity ratio setup.
def ratio := DC / AD_value

-- Statement to prove: The length of DB is 7 * sqrt 2
def prove_DB : ℝ := sqrt (AD_value * DC)

theorem length_of_DB (h₁ : AC = AC_value) (h₂ : AD = AD_value) (h₃ : angle_ABC_right) (h₄ : angle_ADB_right) :
  prove_DB = 7 * sqrt 2 :=
sorry

end length_of_DB_l349_349033


namespace subtraction_and_addition_l349_349139

theorem subtraction_and_addition
  (a b c : ℝ)
  (h₁ : a = 1234.56)
  (h₂ : b = 567.89)
  (h₃ : c = 300.30) :
  ((a - b) + c).round = (966.97 : ℝ).round :=
by
  sorry

end subtraction_and_addition_l349_349139


namespace Q_mul_P_plus_Q_eq_one_l349_349949

noncomputable def sqrt5_plus_2_pow (n : ℕ) :=
  (Real.sqrt 5 + 2)^(2 * n + 1)

noncomputable def P (n : ℕ) :=
  Int.floor (sqrt5_plus_2_pow n)

noncomputable def Q (n : ℕ) :=
  sqrt5_plus_2_pow n - P n

theorem Q_mul_P_plus_Q_eq_one (n : ℕ) : Q n * (P n + Q n) = 1 := by
  sorry

end Q_mul_P_plus_Q_eq_one_l349_349949


namespace ratio_eq_one_third_l349_349353

noncomputable def quadratic_discriminant (a b c : ℚ) : ℚ :=
  b * b - 4 * a * c

noncomputable def root_difference (a b : ℚ) (d : ℚ) : ℚ :=
  (d / a).sqrt

def f1 (a : ℚ) : Polynomial ℚ := Polynomial.C 3 - Polynomial.C (2 * a) * Polynomial.X + Polynomial.X ^ 2
def f2 (b : ℚ) : Polynomial ℚ := Polynomial.C b + Polynomial.C 1 * Polynomial.X + Polynomial.X ^ 2
def f3 (a b : ℚ) : Polynomial ℚ := Polynomial.C (6 + b) - Polynomial.C (4 * a - 1) * Polynomial.X + Polynomial.C 3 * Polynomial.X ^ 2
def f4 (a b : ℚ) : Polynomial ℚ := Polynomial.C (3 + 2 * b) - Polynomial.C (2 * a - 2) * Polynomial.X + Polynomial.C 3 * Polynomial.X ^ 2

noncomputable def A (a : ℚ) : ℚ :=
  root_difference 1 (-2 * a) (quadratic_discriminant 1 (-2 * a) 3)

noncomputable def B (b : ℚ) : ℚ :=
  root_difference 1 1 (quadratic_discriminant 1 1 b)

noncomputable def C (a b : ℚ) : ℚ :=
  root_difference 3 (1 - 4 * a) (quadratic_discriminant 3 (1 - 4 * a) (6 + b))

noncomputable def D (a b : ℚ) : ℚ :=
  root_difference 3 (2 - 2 * a) (quadratic_discriminant 3 (2 - 2 * a) (3 + 2 * b))

theorem ratio_eq_one_third (a b : ℚ) (h : |A a| ≠ |B b|) : 
  (C a b)^2 - (D a b)^2 = (1 / 3) * ((A a)^2 - (B b)^2) := 
by sorry

end ratio_eq_one_third_l349_349353


namespace problem_solution_l349_349168

def is_invertible_modulo_9 (a : ℕ) : Prop := Int.gcd a 9 = 1

theorem problem_solution (a b c d : ℕ) 
  (h1 : a < 9) (h2 : b < 9) (h3 : c < 9) (h4 : d < 9)
  (h5 : a ≠ b) (h6 : a ≠ c) (h7 : a ≠ d)
  (h8 : b ≠ c) (h9 : b ≠ d) (h10 : c ≠ d)
  (h11 : is_invertible_modulo_9 a)
  (h12 : is_invertible_modulo_9 b)
  (h13 : is_invertible_modulo_9 c)
  (h14 : is_invertible_modulo_9 d) :
  ((a * b * c + a * b * d + a * c * d + b * c * d) *
   Nat.gcd_a (a * b * c * d) 9) % 9 = 6 :=
by sorry

end problem_solution_l349_349168


namespace scientific_notation_of_graphene_l349_349813

theorem scientific_notation_of_graphene :
  0.00000000034 = 3.4 * 10^(-10) :=
sorry

end scientific_notation_of_graphene_l349_349813


namespace count_primes_between_30_and_50_l349_349422

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between_30_and_50 : List ℕ :=
  [31, 37, 41, 43, 47]

theorem count_primes_between_30_and_50 : 
  (primes_between_30_and_50.filter is_prime).length = 5 :=
by
  sorry

end count_primes_between_30_and_50_l349_349422


namespace shortest_trip_on_cube_surface_l349_349267

variable (A B C D E F G H : Point)
variable (AB AC AD AE AF AG AH BC BD BE BG BH CD CE CF CG CH DE DF DG DH EF EG EH FG FH GH : Line)
variables (cube : Cube) (edge_length : ℝ)

def edge_length_of_cube := edge_length = 2

def midpoint_of_AB := midpoint AB
def midpoint_of_EF := midpoint EF

theorem shortest_trip_on_cube_surface :
  edge_length_of_cube →
  shortest_path_length_on_surface cube (midpoint_of_AB) (midpoint_of_EF) = 4 := 
sorry

end shortest_trip_on_cube_surface_l349_349267


namespace andy_harry_difference_l349_349659

theorem andy_harry_difference :
  let pi := 3.14
  let Andy_final_area := 5 * pi
  let Harry_final_area := 5
  abs (Andy_final_area - Harry_final_area) = 11 :=
by
  sorry

end andy_harry_difference_l349_349659


namespace sum_tan_squared_l349_349060

def T := {x : ℝ | 0 < x ∧ x < π ∧ 
  (∃ a b c : ℝ, ({a, b, c}, {sin x, cos x, tan x, csc x}).snd ⊆ {a, b, c} ∧ 
  (a^2 + b^2 = c^2 ∨ a^2 = b^2 + c^2 ∨ b^2 = a^2 + c^2))}

theorem sum_tan_squared : (∑ x in T, (tan x)^2) = 1 := by
  sorry

end sum_tan_squared_l349_349060


namespace identify_problem_l349_349471

open EuclideanGeometry

section TriangleProblem

variables (A B C D E P G F : Point)
variables (hABC : triangle A B C)
variables (hD : midpoint D A B)
variables (hE : midpoint E A C)
variables (hP : collinear D P E)
variables (hG : extends B P G ∧ ∈same_line G P C)
variables (hF : extends C P F ∧ ∈same_line F P B)

theorem identify_problem 
    (h_condition1 : dist A B = 1)
    (h_condition2 : dist A C = 1)
    : (1 / dist B F) + (1 / dist C G) = 3 :=
sorry

end TriangleProblem

end identify_problem_l349_349471


namespace xiaoying_school_trip_l349_349594

theorem xiaoying_school_trip :
  ∃ (x y : ℝ), 
    (1200 / 1000) = (3 / 60) * x + (5 / 60) * y ∧ 
    x + y = 16 :=
by
  sorry

end xiaoying_school_trip_l349_349594


namespace remaining_number_is_one_l349_349592

/--
After performing operations A and B on the set of natural numbers {1, ..., 1988}
repeatedly until only one number remains, the last remaining number on the blackboard
is 1.
-/
theorem remaining_number_is_one :
  ∃ d : ℕ → ℕ, (∀ k, 1 ≤ k ∧ k ≤ 1987 → d k = 1) →
  let original_sum := ∑ k in range 1988, k + 1 in
  let new_sum := original_sum - ∑ k in range 1987, (1989 - k) * d k in
  new_sum = 1 :=
begin
  sorry
end

end remaining_number_is_one_l349_349592


namespace boys_next_to_each_other_arrangement_l349_349848

theorem boys_next_to_each_other_arrangement :
  ∃ (ways : ℕ), (let students := 5 in
                 let boys := 2 in
                 let girls := 3 in
                 let boys_next_to_each_other := true in
                 ways = 48) :=
sorry

end boys_next_to_each_other_arrangement_l349_349848


namespace find_m_n_find_a_l349_349775

-- Part 1: Prove m = 4 and n = 5 given 2-i is a root of the equation
theorem find_m_n (m n : ℝ) (h : (2 - 1*I : ℂ)⁅root (X^2 - m*X + n)) :
  m = 4 ∧ n = 5 :=
begin
  sorry
end

-- Part 2: Given m = 4 and n = 5, and z is purely imaginary, prove a = 1
theorem find_a (a m n : ℝ) (h_m : m = 4) (h_n : n = 5)
  (h : let z := (a^2 - n*a + m + (a - m)*I : ℂ) in z.im ≠ 0 ∧ z.re = 0) :
  a = 1 :=
begin
  sorry
end

end find_m_n_find_a_l349_349775


namespace sum_of_coefficients_of_terms_with_rational_coefficients_l349_349552

theorem sum_of_coefficients_of_terms_with_rational_coefficients :
  let f := (λ x, (sqrt (2 ^ (1/3)) - 2 / x)^7)
  -- Extract the coefficients as rational terms
  let r_1_coeff := - (nat.choose 7 1) * (2 ^ 2)
  let r_7_coeff := - (nat.choose 7 7) * (2 ^ 7)
  -- Sum the coefficients
  let sum_coeff := r_1_coeff + r_7_coeff
  sum_coeff = -156 := by
  sorry

end sum_of_coefficients_of_terms_with_rational_coefficients_l349_349552


namespace problem_area_of_triangle_ABC_l349_349146

noncomputable def area_of_triangle_ABC : ℚ :=
  let r := 3
  let AB := 2 * r
  let BD := 4
  let AD := AB + BD
  let DE := 6
  let EA := Real.sqrt (AD^2 + DE^2)
  let R := 3
  let EC := 76 / EA
  let AC := EA - EC
  let BC := Real.sqrt (R^2 - AC^2)
  (BC * AC) / 2

theorem problem_area_of_triangle_ABC :
  area_of_triangle_ABC = 270 / 34 :=
sorry

end problem_area_of_triangle_ABC_l349_349146


namespace theresa_hours_l349_349563

theorem theresa_hours (h : avg_hours : ℝ) (w1 w2 w3 w4 w5 : ℝ) (total_weeks : ℕ) (target_avg : ℝ)
  (H1 : total_weeks = 6) (H2 : target_avg = 10) 
  (H3 : w1 = 8) (H4 : w2 = 11) (H5 : w3 = 7) (H6 : w4 = 12) (H7 : w5 = 10) :
  ∃ x, (w1 + w2 + w3 + w4 + w5 + x) / total_weeks = target_avg := 
begin
  use 12,
  simp [H1, H2, H3, H4, H5, H6, H7],
  norm_num
end

end theresa_hours_l349_349563


namespace smallest_positive_multiple_l349_349189

theorem smallest_positive_multiple (a : ℕ) (h : 17 * a % 53 = 7) : 17 * a = 544 :=
sorry

end smallest_positive_multiple_l349_349189


namespace other_x_intercept_l349_349331

theorem other_x_intercept (a b c : ℝ) 
  (h₁ : ∀ x, a * x^2 + b * x + c = 0 → x = 3 → 0) 
  (h₂ : ∃ y, ∀ x, a * x^2 + b * x + c = y → x = 2 ∧ y = 9) :
  ∃ x, x = 1 ∧ a * x^2 + b * x + c = 0 :=
by
  sorry

end other_x_intercept_l349_349331


namespace evaluate_five_applications_of_f_l349_349509

def f (x : ℤ) : ℤ :=
  if x ≥ 0 then x + 5 else -x^2 - 3

theorem evaluate_five_applications_of_f :
  f (f (f (f (f (-1))))) = -17554795004 :=
by
  sorry

end evaluate_five_applications_of_f_l349_349509


namespace angle_bisector_length_l349_349038

open Real

-- Given conditions for triangle XYZ
def XY : ℝ := 4
def XZ : ℝ := 5
def cosAngleX : ℝ := 3 / 10

-- Prove that the length of the angle bisector XD is approximately 2.9
theorem angle_bisector_length :
  let YZ := sqrt (XY^2 + XZ^2 - 2 * XY * XZ * cosAngleX) in
  let YD := 4 * (sqrt 17 / 9) in
  let XD := sqrt (16 + (YD)^2 - 2 * XY * YD * (sqrt ((1 + cosAngleX * 2) / 2) / 10)) in
  abs (XD - 2.9) < 0.01 :=
by
  sorry

end angle_bisector_length_l349_349038


namespace functional_equation_holds_l349_349054

def f (p q : ℕ) : ℝ :=
  if p = 0 ∨ q = 0 then 0 else (p * q : ℝ)

theorem functional_equation_holds (p q : ℕ) : 
  f p q = 
    if p = 0 ∨ q = 0 then 0 
    else 1 + (1 / 2) * f (p + 1) (q - 1) + (1 / 2) * f (p - 1) (q + 1) :=
  by 
    sorry

end functional_equation_holds_l349_349054


namespace orchids_count_l349_349562

-- Define the number of roses now in the vase
def roses_now := 3

-- Define the difference between the number of orchids and roses
def orchid_diff := 10

-- Define the number of orchids now in the vase
def orchids := roses_now + orchid_diff

-- Theorem stating that the number of orchids in the vase now is 13
theorem orchids_count : orchids = 13 :=
by
  dsimp [orchids, roses_now, orchid_diff]
  rfl


end orchids_count_l349_349562


namespace binom_10_2_eq_45_l349_349706

theorem binom_10_2_eq_45 :
  binom 10 2 = 45 := by
  sorry

end binom_10_2_eq_45_l349_349706


namespace domain_shift_l349_349015

theorem domain_shift (f : ℝ → ℝ) :
  (∀ x, x ∈ set.Icc (-3 : ℝ) 1 → ∃ y, y ∈ set.Icc (-4) 0 ∧ f (x + 1) = g y) ↔
  (∀ y, y ∈ set.Icc (-4 : ℝ) 0 → ∃ x, x ∈ set.Icc (-3) 1 ∧ g y = f (x + 1)) := by
sorry

end domain_shift_l349_349015


namespace determine_phi_l349_349152

theorem determine_phi (phi : ℝ) (h : 0 < phi ∧ phi < π) :
  (∃ k : ℤ, phi = 2*k*π + (3*π/4)) :=
by
  sorry

end determine_phi_l349_349152


namespace binom_10_2_eq_45_l349_349683

theorem binom_10_2_eq_45 : Nat.binomial 10 2 = 45 := by
  sorry

end binom_10_2_eq_45_l349_349683


namespace number_of_primes_between_30_and_50_l349_349433

-- Define what it means to be a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the interval condition
def in_interval (n : ℕ) : Prop :=
  30 ≤ n ∧ n ≤ 50

-- Define the count of prime numbers in the interval
def prime_count_in_interval : ℕ :=
  (List.range' 30 21).countp (λ n, is_prime n)

-- We state that the above count is equal to 5
theorem number_of_primes_between_30_and_50 : prime_count_in_interval = 5 :=
  sorry

end number_of_primes_between_30_and_50_l349_349433


namespace p_necessary_not_sufficient_q_l349_349756

-- Define the conditions p and q
def p (a : ℝ) : Prop := a < 1
def q (a : ℝ) : Prop := 0 < a ∧ a < 1

-- State the necessary but not sufficient condition theorem
theorem p_necessary_not_sufficient_q (a : ℝ) : p a → q a → p a ∧ ¬∀ (a : ℝ), p a → q a :=
by
  sorry

end p_necessary_not_sufficient_q_l349_349756


namespace min_distance_sum_of_n_gon_center_l349_349084

noncomputable def distance (X Y : ℝ × ℝ) : ℝ :=
Real.sqrt ((X.1 - Y.1)^2 + (X.2 - Y.2)^2)

theorem min_distance_sum_of_n_gon_center (n : ℕ) (n_pos : 0 < n) (A : Fin n → ℝ × ℝ) (X O: ℝ × ℝ) :
  let sum_distance (P : ℝ × ℝ) := ∑ i in Finset.univ, distance P (A i) in
  sum_distance X ≥ sum_distance O :=
sorry

end min_distance_sum_of_n_gon_center_l349_349084


namespace height_increase_rate_l349_349206

noncomputable def rate_of_height_increase
  (A : ℝ) (dVdt : ℝ) : ℝ :=
dVdt / A

theorem height_increase_rate :
  let base_area := 100 in
  let fill_rate := 1000 in
  rate_of_height_increase base_area fill_rate = 10 :=
by
  let A := 100
  let dVdt := 1000
  show rate_of_height_increase A dVdt = 10
  sorry

end height_increase_rate_l349_349206


namespace average_cost_proof_l349_349647

noncomputable def original_price (discounted_price : ℝ) (discount : ℝ) : ℝ :=
  discounted_price / (1 - discount)

noncomputable def total_cost (original_price : ℝ) (shipping_cost : ℝ) : ℝ :=
  original_price + shipping_cost

noncomputable def total_cost_cents (total_cost : ℝ) : ℕ :=
  (total_cost * 100).toNat

noncomputable def average_cost_per_pencil (total_cost_cents : ℕ) (pencil_count : ℕ) : ℕ :=
  (total_cost_cents / pencil_count)

theorem average_cost_proof 
  (pencil_count : ℕ)
  (discounted_price : ℝ)
  (discount : ℝ)
  (shipping_cost : ℝ)
  (total_cost_cents : ℕ)
  (average_cost : ℕ)
  (h_total_cost_cents : total_cost_cents = total_cost_cents (total_cost (original_price discounted_price discount) shipping_cost))
  (h_average_cost : average_cost = average_cost_per_pencil total_cost_cents pencil_count) :
  average_cost = 15 :=
by
  -- Sorry, the proof is omitted for this task
  sorry

#eval average_cost_proof 250 25.5 0.15 8.1 3810 15 

end average_cost_proof_l349_349647


namespace P_P_eq_P_eight_equals_58_l349_349087

open Polynomial

noncomputable def P(x : ℚ) : ℚ := x^2 - x + 2

theorem P_P_eq :
  (P ∘ P)(x) = x^4 - 2 * x^3 + 4 * x^2 - 3 * x + 4 := sorry

theorem P_eight_equals_58 :
  P 8 = 58 := sorry

end P_P_eq_P_eight_equals_58_l349_349087


namespace difference_perimeter_square_l349_349954

theorem difference_perimeter_square (s : ℕ) (h : 4 * s ≤ 35) :
  35 - 4 * s = 3 :=
by
  have : s ≤ 8 :=
    by linarith
  have s_eq_8 : s = 8 := by linarith
  rw [s_eq_8, mul_comm, ← nat.cast_mul s 4, nat.cast_succ, ← add_sub_assoc, add_comm, add_comm 2, nat.cast_35]
  sorry

end difference_perimeter_square_l349_349954


namespace uncle_mane_calculator_l349_349981

def no_zero_digit (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, d ≠ 0

def possible_multiplication_results (a b : ℕ) : Prop :=
  no_zero_digit (a * b) ∧ (String.toNat (String.filter (≠ '0') (a * b).repr) = 11)

theorem uncle_mane_calculator :
  ∀ a b : ℕ, 10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 →
  possible_multiplication_results a b →
  (a = 11 ∧ b = 91) ∨ (a = 13 ∧ b = 77) ∨ (a = 25 ∧ b = 44) :=
sorry

end uncle_mane_calculator_l349_349981


namespace integral_value_l349_349864

-- Define the binomial coefficient
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Define the conditions of the problem
def a : ℝ := 2 -- This is derived from the problem condition

-- The main theorem statement
theorem integral_value :
  (∫ x in (0 : ℝ)..a, (Real.exp x + 2 * x)) = Real.exp 2 + 3 := by
  sorry

end integral_value_l349_349864


namespace triangle_side_AC_l349_349872

noncomputable def cosRule (a b c : ℝ) : ℝ := 
  (b^2 + c^2 - a^2) / (2 * b * c)

theorem triangle_side_AC 
  (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
  (AB_length BC_length : ℝ) (cos_A : ℝ) (AC_length : ℝ) :
  AB_length = 1 → BC_length = sqrt(5) → cos_A = 5/6 → (cosRule BC_length AC_length AB_length = 5/6) → AC_length = 3 :=
by
  sorry

end triangle_side_AC_l349_349872


namespace max_superior_cells_l349_349343

def is_superior (n : ℕ) (matrix : matrix (fin n) (fin n) ℕ) (i j : fin n) : Prop :=
  ∀ k : fin n, matrix i j ≥ matrix i k ∧ matrix i j ≥ matrix k j

theorem max_superior_cells (n : ℕ) (h : n = 2004) (matrix : matrix (fin n) (fin n) ℕ) :
  (∑ i : fin n, ∑ j : fin n, if is_superior n matrix i j then 1 else 0) = 2004 :=
sorry

end max_superior_cells_l349_349343


namespace find_num_lineups_l349_349140

noncomputable def numStartingLineups (totalPlayers : ℕ) (lineupSize : ℕ) (excludedPlayers : Finset ℕ) : ℕ :=
  let players := Finset.range totalPlayers
  let remainingPlayers := players \ excludedPlayers
  let binom := Finset.card remainingPlayers
  let cases := (remainingPlayers.choose (lineupSize - 1)).card * 3
              + (remainingPlayers.choose lineupSize).card
  cases

theorem find_num_lineups : numStartingLineups 15 5 {0, 1, 2} = 2277 := by
  sorry

end find_num_lineups_l349_349140


namespace candies_on_second_day_l349_349915

noncomputable def total_candies := 45
noncomputable def days := 5
noncomputable def difference := 3

def arithmetic_sum (n : ℕ) (a₁ d : ℕ) :=
  (n * (2 * a₁ + (n - 1) * d)) / 2

theorem candies_on_second_day (a : ℕ) (h : arithmetic_sum days a difference = total_candies) :
  a + difference = 6 := by
  sorry

end candies_on_second_day_l349_349915


namespace prime_count_30_to_50_l349_349401

def is_prime (n: ℕ) : Prop :=
  n > 1 ∧ ∀ m: ℕ, m ∣ n → m = 1 ∨ m = n

def primes_in_range (a b: ℕ) : list ℕ :=
  list.filter is_prime (list.range' a (b - a + 1))

theorem prime_count_30_to_50 : (primes_in_range 30 50).length = 5 :=
by sorry

end prime_count_30_to_50_l349_349401


namespace theta_solution_count_l349_349305

theorem theta_solution_count : 
  let interval := set.Ioc 0 (2 * Real.pi)
      f (θ : ℝ) : ℝ := 3 + 2 * Real.cos(2 * θ) - 4 * Real.sin(4 * θ)
  in set.finite {θ ∈ interval | f θ = 0} ∧ 
     set.card {θ ∈ interval | f θ = 0} = 16 :=
by 
  let interval := set.Ioc 0 (2 * Real.pi)
  let f (θ : ℝ) : ℝ := 3 + 2 * Real.cos(2 * θ) - 4 * Real.sin(4 * θ)
  sorry

end theta_solution_count_l349_349305


namespace cylinder_surface_area_and_volume_l349_349234

noncomputable def cylinder_total_surface_area (r h : ℝ) : ℝ :=
  2 * Real.pi * r^2 + 2 * Real.pi * r * h

noncomputable def cylinder_volume (r h : ℝ) : ℝ :=
  Real.pi * r^2 * h

theorem cylinder_surface_area_and_volume (r h : ℝ) (hr : r = 5) (hh : h = 15) :
  cylinder_total_surface_area r h = 200 * Real.pi ∧ cylinder_volume r h = 375 * Real.pi :=
by
  sorry -- Proof omitted

end cylinder_surface_area_and_volume_l349_349234


namespace g_neither_even_nor_odd_l349_349043

def g (x : ℝ) : ℝ := Real.floor x + 1/3

theorem g_neither_even_nor_odd : 
  ¬ (∀ x : ℝ, g x = g (-x)) ∧ ¬ (∀ x : ℝ, g (-x) = - g x) :=
by
  sorry

end g_neither_even_nor_odd_l349_349043


namespace range_of_a_exists_x_ax2_ax_1_lt_0_l349_349519

theorem range_of_a_exists_x_ax2_ax_1_lt_0 :
  {a : ℝ | ∃ x : ℝ, a * x^2 + a * x + 1 < 0} = {a : ℝ | a < 0 ∨ a > 4} :=
sorry

end range_of_a_exists_x_ax2_ax_1_lt_0_l349_349519


namespace avg_age_10_students_l349_349941

-- Defining the given conditions
def avg_age_15_students : ℕ := 15
def total_students : ℕ := 15
def avg_age_4_students : ℕ := 14
def num_4_students : ℕ := 4
def age_15th_student : ℕ := 9

-- Calculating the total age based on given conditions
def total_age_15_students : ℕ := avg_age_15_students * total_students
def total_age_4_students : ℕ := avg_age_4_students * num_4_students
def total_age_10_students : ℕ := total_age_15_students - total_age_4_students - age_15th_student

-- Problem to be proved
theorem avg_age_10_students : total_age_10_students / 10 = 16 := 
by sorry

end avg_age_10_students_l349_349941


namespace calculation_proof_l349_349671

theorem calculation_proof :
  5^(Real.log 9 / Real.log 5) + (1 / 2) * (Real.log 32 / Real.log 2) - Real.log (Real.log 8 / Real.log 2) / Real.log 3 = 21 / 2 := 
  sorry

end calculation_proof_l349_349671


namespace find_b_l349_349899

noncomputable def g (b : ℝ) (x : ℝ) : ℝ := b * x ^ 3 + b * x - 3

theorem find_b (b : ℝ) (h : g b (g b 1) = 1) : b = 1 / 2 :=
by
  sorry

end find_b_l349_349899


namespace catherine_initial_pens_l349_349674

-- Defining the conditions
def equal_initial_pencils_and_pens (P : ℕ) : Prop := true
def pens_given_away_per_friend : ℕ := 8
def pencils_given_away_per_friend : ℕ := 6
def number_of_friends : ℕ := 7
def remaining_pens_and_pencils : ℕ := 22

-- The total number of items given away
def total_pens_given_away : ℕ := pens_given_away_per_friend * number_of_friends
def total_pencils_given_away : ℕ := pencils_given_away_per_friend * number_of_friends

-- The problem statement in Lean 4
theorem catherine_initial_pens (P : ℕ) 
  (h1 : equal_initial_pencils_and_pens P)
  (h2 : P - total_pens_given_away + P - total_pencils_given_away = remaining_pens_and_pencils) : 
  P = 60 :=
sorry

end catherine_initial_pens_l349_349674


namespace articles_bought_l349_349014

theorem articles_bought (C : ℝ) (N : ℝ) (h1 : (N * C) = (30 * ((5 / 3) * C))) : N = 50 :=
by
  sorry

end articles_bought_l349_349014


namespace proof_of_geometric_identity_l349_349283

open EuclideanGeometry

/-- Given parallelogram ABCD with an acute angle at A, where AC and BD intersect at E. 
Circumscribed circle of triangle ACD intersects AB, BC, and BD at K, L, and P respectively. 
The circumscribed circle of triangle CEL intersects BD at M. 
Prove that KD * KM = KL * PC. -/
theorem proof_of_geometric_identity
  (A B C D E K L P M : Point)
  (h_parallelogram : parallelogram A B C D)
  (h_acute : acute_angle_on_A : angle A B C < π / 2)
  (h_intersect_AC_BD_E : intersect AC BD E)
  (h_circumscribed_ACD : circumscribed_circle_triangle A C D)
  (h_intersects_K : intersect_circle_line h_circumscribed_ACD AB K)
  (h_intersects_L : intersect_circle_line h_circumscribed_ACD BC L)
  (h_intersects_P : intersect_circle_line h_circumscribed_ACD BD P)
  (h_circumscribed_CEL : circumscribed_circle_triangle C E L)
  (h_intersects_M : intersect_circle_line h_circumscribed_CEL BD M)
  : length (KD) * length (KM) = length (KL) * length (PC)
  :=
sorry

end proof_of_geometric_identity_l349_349283


namespace correct_propositions_l349_349327

-- Define the conditions of the problem
variables {A B C : ℝ} -- Angles of the triangle
variables {a b c : ℝ} -- Sides of the triangle

-- Propositions
def prop1 : Prop := (sin (2 * A) = sin (2 * B)) → (A = B)
def prop2 : Prop := (sin B = cos A) → (A + B = π / 2)
def prop3 : Prop := (sin A ^ 2 + sin B ^ 2 > sin C ^ 2) → (A + B + C < π)
def prop4 : Prop := (a / cos A = b / cos B ∧ b / cos B = c / cos C) → (A = B ∧ B = C)

-- The main theorem stating that only fourth proposition is correct
theorem correct_propositions :
  ¬prop1 ∧ ¬prop2 ∧ ¬prop3 ∧ prop4 :=
by
  sorry

end correct_propositions_l349_349327


namespace value_of_m_l349_349783

theorem value_of_m
  (α : ℝ)
  (m : ℝ)
  (h1 : α > π ∧ α < 3*π / 2) -- α is in the third quadrant
  (h2 : sin α + cos α = 2 * m)
  (h3 : sin (2 * α) = m ^ 2) :
  m = - (sqrt (3) / 3) :=
sorry

end value_of_m_l349_349783


namespace methane_required_l349_349821

def mole_of_methane (moles_of_oxygen : ℕ) : ℕ := 
  if moles_of_oxygen = 2 then 1 else 0

theorem methane_required (moles_of_oxygen : ℕ) : 
  moles_of_oxygen = 2 → mole_of_methane moles_of_oxygen = 1 := 
by 
  intros h
  simp [mole_of_methane, h]

end methane_required_l349_349821


namespace volume_sphere_kaili_method_l349_349861

theorem volume_sphere_kaili_method (V : ℝ) : (4 * Real.pi * (1/3)^2 = 4 * Real.pi / 9) → 
  (2 * (1/3) = real.cbrt(16 * V / 9)) → V = 1 / 6 :=
by
  intros h₁ h₂
  sorry

end volume_sphere_kaili_method_l349_349861


namespace polynomial_value_at_8_l349_349110

noncomputable def P : ℝ → ℝ := λ x, x^2 - x + 2

theorem polynomial_value_at_8 :
  (P (P x) = x^4 - 2*x^3 + 4*x^2 - 3*x + 4) ∧ (P = λ x, x^2 - x + 2) → P 8 = 58 :=
by
  sorry

end polynomial_value_at_8_l349_349110


namespace angle_CED_60_degrees_l349_349891

theorem angle_CED_60_degrees
  (circle : Type*)
  [metric_space circle]
  [normed_add_comm_group circle]
  [normed_space ℝ circle]
  {O A B E C D F : circle}
  (h1 : is_diameter O A B) 
  (h2 : point_on_circle E O)
  (h3 : tangent_point B E C D)
  (h4 : tangent_point B E C A)
  (h5 : perpendicular (line_through C F) (line_through A B))
  (h6 : angle (line_through B A) (line_through A E) = 30)
  : angle (line_through C E) (line_through E D) = 60 :=
sorry

end angle_CED_60_degrees_l349_349891


namespace solve_for_x_l349_349006

theorem solve_for_x (x : ℝ) (h : 5 + 7 / x = 6 - 5 / x) : x = 12 := 
sorry

end solve_for_x_l349_349006


namespace smallest_n_bound_condition_l349_349903

def harmonic_series (n : ℕ) : ℝ := ∑ k in finset.range (n + 1), 1 / (k + 1)

theorem smallest_n_bound_condition : ∃ (n : ℕ), harmonic_series n > 10 ∧ (∀ m < n, harmonic_series m ≤ 10) :=
begin
  use 12320,
  split,
  { 
    sorry -- We need to prove harmonic_series 12320 > 10
  },
  {
    intro m,
    intro hmn,
    sorry -- We need to prove harmonic_series m ≤ 10 for all m < 12320
  }
end

end smallest_n_bound_condition_l349_349903


namespace equivalent_problem_l349_349062

noncomputable def problem_statement : Prop :=
  ∀ (a b c d : ℝ), a ≠ -1 → b ≠ -1 → c ≠ -1 → d ≠ -1 →
  ∀ (ω : ℂ), ω^4 = 1 → ω ≠ 1 →
  (1 / (a + ω) + 1 / (b + ω) + 1 / (c + ω) + 1 / (d + ω) = 4 / (1 + ω)) →
  (1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) + 1 / (d + 1) = 2)

#check problem_statement

-- Expected output for type checking without providing the proof
theorem equivalent_problem : problem_statement :=
  sorry

end equivalent_problem_l349_349062


namespace positive_difference_l349_349579

noncomputable def calculate_diff : ℕ :=
  let first_term := (8^2 - 8^2) / 8
  let second_term := (8^2 * 8^2) / 8
  second_term - first_term

theorem positive_difference : calculate_diff = 512 := by
  sorry

end positive_difference_l349_349579


namespace tan_A_tan_B_l349_349160

theorem tan_A_tan_B (A B C : ℝ) (R : ℝ) (H F : ℝ)
  (HF : H + F = 26) (h1 : 2 * R * Real.cos A * Real.cos B = 8)
  (h2 : 2 * R * Real.sin A * Real.sin B = 26) :
  Real.tan A * Real.tan B = 13 / 4 :=
by
  sorry

end tan_A_tan_B_l349_349160


namespace log_product_solution_l349_349591

theorem log_product_solution (x : ℝ) (hx : 0 < x) : 
  (Real.log x / Real.log 2) * (Real.log x / Real.log 5) = Real.log 10 / Real.log 2 ↔ 
  x = 2 ^ Real.sqrt (6 * Real.log 2) :=
sorry

end log_product_solution_l349_349591


namespace find_a_of_perpendicular_lines_l349_349016

theorem find_a_of_perpendicular_lines (a : ℝ) :
  let line1 : ℝ := a * x + y - 1
  let line2 : ℝ := 4 * x + (a - 3) * y - 2
  (∀ x y : ℝ, (line1 = 0 → line2 ≠ 0 → line1 * line2 = -1)) → a = 3 / 5 :=
by
  sorry

end find_a_of_perpendicular_lines_l349_349016


namespace infinite_geometric_series_sum_l349_349351

theorem infinite_geometric_series_sum (a : ℕ → ℝ) (a1 : a 1 = 1) (r : ℝ) (h : r = 1 / 3) (S : ℝ) (H : S = a 1 / (1 - r)) : S = 3 / 2 :=
by
  sorry

end infinite_geometric_series_sum_l349_349351


namespace mode_of_gold_medals_is_8_l349_349515

def countries : List String := ["Norway", "Germany", "China", "USA", "Sweden", "Netherlands", "Austria"]

def gold_medals : List Nat := [16, 12, 9, 8, 8, 8, 7]

def mode (lst : List Nat) : Nat :=
  lst.foldr
    (fun (x : Nat) acc =>
      if lst.count x > lst.count acc then x else acc)
    lst.head!

theorem mode_of_gold_medals_is_8 :
  mode gold_medals = 8 :=
by sorry

end mode_of_gold_medals_is_8_l349_349515


namespace proof_eq1_proof_eq2_l349_349933

variable (x : ℝ)

-- Proof problem for Equation (1)
theorem proof_eq1 (h : (1 - x) / 3 - 2 = x / 6) : x = -10 / 3 := sorry

-- Proof problem for Equation (2)
theorem proof_eq2 (h : (x + 1) / 0.25 - (x - 2) / 0.5 = 5) : x = -3 / 2 := sorry

end proof_eq1_proof_eq2_l349_349933


namespace min_value_expression_l349_349787

theorem min_value_expression (a b : ℝ) (h1 : 2^a * 2^b = 16) (h2 : a ≥ 0) (h3 : b ≥ 0) :
  ∃ x, (x = min (λ x, (4 / (2 * a + b)) + (1 / (a + 2 * b))) x) ∧ x = 3 / 4 :=
by sorry

end min_value_expression_l349_349787


namespace first_car_speed_l349_349173

-- Definitions based on problem conditions
def highway_length : ℕ := 105
def second_car_speed : ℕ := 20
def meeting_time : ℕ := 3

-- The question is to prove the speed of the first car
theorem first_car_speed : ∃ v : ℕ, 3 * v + 3 * second_car_speed = highway_length ∧ v = 15 :=
by
  use 15
  split
  { calc
      3 * 15 + 3 * 20 = highway_length : by norm_num
  }
  { refl }

end first_car_speed_l349_349173


namespace probability_sum_six_is_one_fifth_l349_349559

/-- There are 5 balls in a pocket, all of the same material and size, numbered 1, 2, 3, 4, 5.
Players A and B play a game where A draws a ball first, records the number, and then B draws a ball.
If the sum of the two numbers is 6, calculate the probability of this event. -/
noncomputable def probability_sum_six : ℚ :=
  let possible_outcomes := [(1,2), (1,3), (1,4), (1,5), (2,1), (2,3), (2,4), (2,5), (3,1),
  			     (3,2), (3,4), (3,5), (4,1), (4,2), (4,3), (4,5), (5,1), (5,2), (5,3) in
  let event_A := [(1,5), (2,4), (3,3), (4,2), (5,1)] in
  let P_A := (event_A.length : ℚ) / (possible_outcomes.length : ℚ) in
  P_A

-- The theorem to prove the probability is rational value 1/5
theorem probability_sum_six_is_one_fifth : probability_sum_six = 1 / 5 :=
sorry

end probability_sum_six_is_one_fifth_l349_349559


namespace fisherman_more_fish_than_pelican_l349_349246

variable (pelican_fish : ℕ)
variable (kingfisher_fish : ℕ)
variable (fisherman_fish : ℕ)

def pelican_caught_fish := (pelican_fish = 13)
def kingfisher_caught_fish := (kingfisher_fish = pelican_fish + 7)
def fisherman_caught_fish := (fisherman_fish = 3 * (pelican_fish + kingfisher_fish))

theorem fisherman_more_fish_than_pelican (h1 : pelican_caught_fish) (h2 : kingfisher_caught_fish) (h3 : fisherman_caught_fish) :
  (fisherman_fish - pelican_fish) = 86 :=
by
  cases h1
  cases h2
  cases h3
  sorry

end fisherman_more_fish_than_pelican_l349_349246


namespace walking_east_of_neg_west_l349_349018

-- Define the representation of directions
def is_walking_west (d : ℕ) (x : ℤ) : Prop := x = d
def is_walking_east (d : ℕ) (x : ℤ) : Prop := x = -d

-- Given the condition and states the relationship is the proposition to prove.
theorem walking_east_of_neg_west (d : ℕ) (x : ℤ) (h : is_walking_west 2 2) : is_walking_east 5 (-5) :=
by
  sorry

end walking_east_of_neg_west_l349_349018


namespace right_triangle_products_eq_one_l349_349491

noncomputable def S : set (ℕ × ℕ) :=
  {p | p.1 ∈ {0, 1, 2, 3} ∧ p.2 ∈ {0, 1, 2, 3, 4} ∧ p ≠ (0, 4)}

structure triangle :=
  (A B C : ℕ × ℕ)
  (right_angle_B : (B ∈ S) ∧ ((A,B,C) forms a right triangle))

noncomputable def T : set triangle :=
  {t : triangle | t.A ∈ S ∧ t.B ∈ S ∧ t.C ∈ S ∧ t.right_angle_B}

def f (t : triangle) : ℝ :=
  tan (angle t.A t.C t.B)

theorem right_triangle_products_eq_one : 
  ∏ t in T, f t = 1 :=
sorry

end right_triangle_products_eq_one_l349_349491


namespace polar_to_cartesian_2_pi_over_6_l349_349284

theorem polar_to_cartesian_2_pi_over_6 :
  let r : ℝ := 2
  let θ : ℝ := (Real.pi / 6)
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  (x, y) = (Real.sqrt 3, 1) := by
    -- Initialize the constants and their values
    let r := 2
    let θ := Real.pi / 6
    let x := r * Real.cos θ
    let y := r * Real.sin θ
    -- Placeholder for the actual proof
    sorry

end polar_to_cartesian_2_pi_over_6_l349_349284


namespace exists_infinite_diff_but_not_sum_of_kth_powers_l349_349069

theorem exists_infinite_diff_but_not_sum_of_kth_powers (k : ℕ) (hk : k > 1) :
  ∃ (infinitely_many x : ℕ), (∃ (a b : ℕ), x = a^k - b^k) ∧ ¬ (∃ (c d : ℕ), x = c^k + d^k) :=
  sorry

end exists_infinite_diff_but_not_sum_of_kth_powers_l349_349069


namespace probability_of_diamond_or_ace_at_least_one_l349_349228

noncomputable def prob_at_least_one_diamond_or_ace : ℚ := 
  1 - (9 / 13) ^ 2

theorem probability_of_diamond_or_ace_at_least_one :
  prob_at_least_one_diamond_or_ace = 88 / 169 := 
by
  sorry

end probability_of_diamond_or_ace_at_least_one_l349_349228


namespace tangent_normal_lines_equations_l349_349279

noncomputable def curve (x : ℝ) : ℝ := x * Real.log x

noncomputable def derivative_curve (x : ℝ) : ℝ := Real.log x + 1

def point := (Real.exp 1, Real.exp 1)

def tangent_equation (x : ℝ) : ℝ := 2 * x - Real.exp 1

def normal_equation (x : ℝ) : ℝ := -0.5 * x + 1.5 * Real.exp 1

theorem tangent_normal_lines_equations :
  (∀ x, x = Real.exp 1 → curve x = Real.exp 1) ∧
  (∀ x, x = Real.exp 1 → derivative_curve x = 2) ∧
  (∀ x, tangent_equation x = 2 * x - Real.exp 1) ∧
  (∀ x, normal_equation x = -0.5 * x + 1.5 * Real.exp 1) :=
by
  -- Proof omitted
sorry

end tangent_normal_lines_equations_l349_349279


namespace quadratic_polynomial_P8_l349_349133

theorem quadratic_polynomial_P8 :
  ∃ (a b c : ℝ), 
  (∀ x : ℝ, P x = a * x^2 + b * x + c) ∧ 
  (∀ x : ℝ, P (P x) = x^4 - 2 * x^3 + 4 * x^2 - 3 * x + 4) ∧ 
  P 8 = 58 :=
begin
  sorry
end

end quadratic_polynomial_P8_l349_349133


namespace average_and_median_of_sales_l349_349213

theorem average_and_median_of_sales :
  let sales := [35, 47, 50, 48, 42, 60, 68] in
  let average := 50 in
  let median := 48 in
  (1 / 7 * (35 + 47 + 50 + 48 + 42 + 60 + 68) = average) ∧
  (list.median sales = median) :=
by
  -- Calculation for average
  let sales := [35, 47, 50, 48, 42, 60, 68]
  let average := 50
  let median := 48
  have h_avg : (1 / 7) * (35 + 47 + 50 + 48 + 42 + 60 + 68) = average := sorry
  have h_median : list.median sales = median := sorry
  exact ⟨h_avg, h_median⟩


end average_and_median_of_sales_l349_349213


namespace part1_part2_l349_349800

section Part1
variable (a : ℝ)
def f (x : ℝ) := a * (2^x + x^2) + 2^(-x)

theorem part1 : (∀ x : ℝ, f a x = f a (-x)) → a = 1 := sorry
end Part1

section Part2
def f_part2 (x : ℝ) := 2^x + x^2 + 2^(-x)
variable (m : ℝ)

theorem part2 : (∀ x : ℝ, 0 < x → m * f_part2 x ≤ 2^(-x) + m * x^2 + m - 1) → m ≤ -1/3 := sorry
end Part2

end part1_part2_l349_349800


namespace sin_double_angle_l349_349752

theorem sin_double_angle (x : ℝ) (h : Real.sin (Real.pi / 4 - x) = 3 / 5) : Real.sin (2 * x) = 7 / 25 := by
  sorry

end sin_double_angle_l349_349752


namespace contrapositive_example_contrapositive_proof_l349_349944

theorem contrapositive_example (x : ℝ) (h : x > 1) : x^2 > 1 := 
sorry

theorem contrapositive_proof (x : ℝ) (h : x^2 ≤ 1) : x ≤ 1 :=
sorry

end contrapositive_example_contrapositive_proof_l349_349944


namespace quadratic_polynomial_P8_l349_349128

theorem quadratic_polynomial_P8 :
  ∃ (a b c : ℝ), 
  (∀ x : ℝ, P x = a * x^2 + b * x + c) ∧ 
  (∀ x : ℝ, P (P x) = x^4 - 2 * x^3 + 4 * x^2 - 3 * x + 4) ∧ 
  P 8 = 58 :=
begin
  sorry
end

end quadratic_polynomial_P8_l349_349128


namespace P_parity_Q_div_by_3_l349_349332

-- Define polynomial P(x)
def P (x p q : ℤ) : ℤ := x*x + p*x + q

-- Define polynomial Q(x)
def Q (x p q : ℤ) : ℤ := x*x*x + p*x + q

-- Part (a) proof statement
theorem P_parity (p q : ℤ) (h1 : Odd p) (h2 : Even q ∨ Odd q) :
  (∀ x : ℤ, Even (P x p q)) ∨ (∀ x : ℤ, Odd (P x p q)) :=
sorry

-- Part (b) proof statement
theorem Q_div_by_3 (p q : ℤ) (h1 : q % 3 = 0) (h2 : p % 3 = 2) :
  ∀ x : ℤ, Q x p q % 3 = 0 :=
sorry

end P_parity_Q_div_by_3_l349_349332


namespace min_value_modulus_l349_349501

-- Given condition: |z - 2i| + |z - 5| = 7
-- We need to prove that the minimum value of |z| is sqrt(100 / 29)

theorem min_value_modulus 
  (z : ℂ) 
  (h : complex.abs (z - complex.I * 2) + complex.abs (z - 5) = 7) : 
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ complex.abs (5 * (1 - t) + 2 * t * complex.I) = real.sqrt (100 / 29) := 
sorry

end min_value_modulus_l349_349501


namespace dogs_return_same_time_l349_349977

theorem dogs_return_same_time
  (L : ℝ)  -- the distance between the two people
  (u v V : ℝ)  -- speeds of the slow person, fast person, and the dogs respectively
  (h1 : u > 0)  -- speed of the slow person is positive
  (h2 : v > 0)  -- speed of the fast person is positive
  (h3 : V > 0)  -- speed of the dogs is positive
  (h4 : L > 0)  -- initial distance is positive
  : (2 * (L / (u + v + V))) = (2 * (L / (u + v + V))) := -- the equation asserts that the round trip time for both dogs are equal
begin 
  -- skipping proof as instructed
  sorry
end

end dogs_return_same_time_l349_349977


namespace area_of_centroid_triangle_l349_349892

noncomputable def area_G1G2G3 (P A B C G1 G2 G3: Type) [metric_space P] [metric_space A] [metric_space B] [metric_space C] [metric_space G1] [metric_space G2] [metric_space G3] : ℝ :=
  if h : (P ∈ triangle ABC) ∧ (G1G2G3_is_centroid_of_PBC_PCA_PAB G1 G2 G3 P A B C) ∧
            (area_of_triangle_ABC A B C = 24) ∧
            (similarity_ratio_of_centroid_scaling G1 G2 G3 P A B C = 1/4)
  then
    24 * (1/4)^2
  else
    0

theorem area_of_centroid_triangle :
  ∀ (P A B C G1 G2 G3 : Type) [metric_space P] [metric_space A] [metric_space B] [metric_space C] [metric_space G1] [metric_space G2] [metric_space G3],
  (P ∈ triangle ABC) →
  G1G2G3_is_centroid_of_PBC_PCA_PAB G1 G2 G3 P A B C →
  area_of_triangle_ABC A B C = 24 →
  similarity_ratio_of_centroid_scaling G1 G2 G3 P A B C = 1/4 →
  area_G1G2G3 P A B C G1 G2 G3 = 1.5 :=
by
  intros P A B C G1 G2 G3 _ _ _ _ _ h1 h2 h3 h4
  -- The proof goes here
  sorry

end area_of_centroid_triangle_l349_349892


namespace unique_solution_l349_349304

theorem unique_solution (x y a : ℝ) :
  (x^2 + y^2 = 2 * a ∧ x + Real.log (y^2 + 1) / Real.log 2 = a) ↔ a = 0 ∧ x = 0 ∧ y = 0 :=
by
  sorry

end unique_solution_l349_349304


namespace sum_of_remainders_l349_349631

theorem sum_of_remainders :
  ∑ n in ({0, 1, 2, 3, 4, 5} : Finset ℕ), ((18 * n + 25) % 31) = 110 := by
    sorry

end sum_of_remainders_l349_349631


namespace sum_c_n_l349_349958

noncomputable def sequence (n : ℕ) : ℝ :=
  if n = 0 then 2 else if n = 1 then 1/6 else if n % 2 = 1 then (2 : ℝ) / (3 ^ (n / 2)) else (1/6 : ℝ) / (3 ^ ((n-1)/2)) 

def c_n (n : ℕ) : ℝ := sequence n + sequence (n + 1)

theorem sum_c_n : (∑' n : ℕ, c_n n) = 9 / 2 := by
  sorry

end sum_c_n_l349_349958


namespace distance_inequality_l349_349567

open Real

theorem distance_inequality 
  (A B C D : ℝ × ℝ × ℝ) : 
  let (x1, y1, z1) := A 
  let (x2, y2, z2) := B 
  let (x3, y3, z3) := C 
  let (x4, y4, z4) := D in
  (let AC_sq := (x1 - x3)^2 + (y1 - y3)^2 + (z1 - z3)^2 in
   let BD_sq := (x2 - x4)^2 + (y2 - y4)^2 + (z2 - z4)^2 in
   let AD_sq := (x1 - x4)^2 + (y1 - y4)^2 + (z1 - z4)^2 in
   let BC_sq := (x2 - x3)^2 + (y2 - y3)^2 + (z2 - z3)^2 in
   let AB_sq := (x1 - x2)^2 + (y1 - y2)^2 + (z1 - z2)^2 in
   let CD_sq := (x3 - x4)^2 + (y3 - y4)^2 + (z3 - z4)^2 in
   AC_sq + BD_sq + AD_sq + BC_sq ≥ AB_sq + CD_sq) := by
  sorry

end distance_inequality_l349_349567


namespace ellipse_equation_l349_349771

noncomputable def ellipse_center_origin_eccentricity (x y : ℝ) : Prop :=
  let e := (sqrt 3) / 2 in
  let c := sqrt 3 in
  let a := 2 in
  let b := sqrt (a^2 - c^2) in
  (b = 1) ∧ (a > b ∧ b > 0) ∧ 
  (x^2 + (y^2 / 4) = 1) ↔
  (x^2 + (y^2 / 4) = 1)

theorem ellipse_equation : 
  ∀ (x y : ℝ), ellipse_center_origin_eccentricity x y → (x^2 + (y^2 / 4) = 1) :=
sorry

end ellipse_equation_l349_349771


namespace river_flow_rate_l349_349635

variables (depth width volume_per_minute : ℝ)

def cross_sectional_area (depth width : ℝ) : ℝ :=
  depth * width

def flow_rate (volume_per_minute : ℝ) : ℝ :=
  volume_per_minute / 60

def speed_m_per_s (flow_rate area : ℝ) : ℝ :=
  flow_rate / area

def speed_kmph (speed_m_per_s : ℝ) : ℝ :=
  speed_m_per_s * 3.6

theorem river_flow_rate :
  depth = 3 →
  width = 55 →
  volume_per_minute = 2750 →
  speed_kmph (speed_m_per_s (flow_rate volume_per_minute) (cross_sectional_area depth width)) = 1 :=
by
  intros h_depth h_width h_volume
  -- Proof will be inserted here
  sorry

end river_flow_rate_l349_349635


namespace quadratic_polynomial_P8_l349_349130

theorem quadratic_polynomial_P8 :
  ∃ (a b c : ℝ), 
  (∀ x : ℝ, P x = a * x^2 + b * x + c) ∧ 
  (∀ x : ℝ, P (P x) = x^4 - 2 * x^3 + 4 * x^2 - 3 * x + 4) ∧ 
  P 8 = 58 :=
begin
  sorry
end

end quadratic_polynomial_P8_l349_349130


namespace median_sixth_time_l349_349478

theorem median_sixth_time (times: list ℝ) (ht: times = [12.5, 11.8, 11.2, 12.0, 12.2])
                         (new_times : list ℝ) (hnew: new_times = insert 12.0 times)
                         (median: ℝ) (hmed: median = 11.9) :
  ∃ x, x = 12.0 ∧ median_of_list (insert x times) = 11.9 :=
by 
  let six_times := insert 12.0 times,
  have h: sorted (six_times) := sorry,
  have hm : median_of_list(six_times) = 11.9 := sorry,
  use 12.0,
  simp [six_times, ht, auto_param_eq]

end median_sixth_time_l349_349478


namespace positive_difference_l349_349583

def a : ℕ := (8^2 - 8^2) / 8
def b : ℕ := (8^2 * 8^2) / 8

theorem positive_difference : |b - a| = 512 :=
by
  sorry

end positive_difference_l349_349583


namespace find_initial_investment_l349_349442

-- Define the necessary parameters for the problem
variables (P r : ℝ)

-- Given conditions
def condition1 : Prop := P * (1 + r * 3) = 240
def condition2 : Prop := 150 * (1 + r * 6) = 210

-- The statement to be proved
theorem find_initial_investment (h1 : condition1 P r) (h2 : condition2 r) : P = 200 :=
sorry

end find_initial_investment_l349_349442


namespace max_distance_S_origin_l349_349956

-- The complex numbers z, (1-i)z, and 3conjugate_z representing points P, Q, and R
-- Conditions: |z| = 1, P, Q, R are not collinear.

theorem max_distance_S_origin 
  (z : ℂ) 
  (hz : abs z = 1)
  (h_collinear : ¬ collinear ({z, (1 - complex.I) * z, 3 * conj z} : set ℂ)) :
  ∃ S : ℂ, distance S 0 = sqrt 14 :=
begin
  use (3 * conj z - complex.I * z),
  calc distance (3 * conj z - complex.I * z) 0
    = abs (3 * conj z - complex.I * z) : by rw [distance_eq_abs]
    ... = sqrt 14 : sorry
end

end max_distance_S_origin_l349_349956


namespace part1_part2_part3_1_part3_2_l349_349330

variables {x y a t : ℚ} -- Define the rational variables

definition beautiful_association (x y a t : ℚ) : Prop :=
  |x - a| + |y - a| = t

theorem part1 : beautiful_association (-1) 5 2 6 :=
sorry

theorem part2 {x: ℚ} : beautiful_association x 5 3 4 → (x = 1 ∨ x = 5) :=
sorry

theorem part3_1 {x0 x1 : ℚ} : beautiful_association x0 x1 1 1 → x0 + x1 = 1 :=
sorry

theorem part3_2 (h : ∀ n : ℕ, n < 2000 → beautiful_association (x n) (x (n+1)) (n+1) 1)
               : (∑ i in finset.range 2001, x i) = 2001000 :=
sorry

end part1_part2_part3_1_part3_2_l349_349330


namespace rate_per_kg_mangoes_is_55_l349_349667

def total_amount : ℕ := 1125
def rate_per_kg_grapes : ℕ := 70
def weight_grapes : ℕ := 9
def weight_mangoes : ℕ := 9

def cost_grapes := rate_per_kg_grapes * weight_grapes
def cost_mangoes := total_amount - cost_grapes

theorem rate_per_kg_mangoes_is_55 (rate_per_kg_mangoes : ℕ) (h : rate_per_kg_mangoes = cost_mangoes / weight_mangoes) : rate_per_kg_mangoes = 55 :=
by
  -- proof construction
  sorry

end rate_per_kg_mangoes_is_55_l349_349667


namespace trig_identity_l349_349280

theorem trig_identity :
  sin (47 * real.pi / 180) * cos (17 * real.pi / 180) - cos (47 * real.pi / 180) * cos (73 * real.pi / 180) = 1 / 2 := 
sorry

end trig_identity_l349_349280


namespace a_3_eq_5_l349_349767

variable (a : ℕ → ℕ) -- Defines the arithmetic sequence
variable (S : ℕ → ℕ) -- The sum of the first n terms of the sequence

-- Condition: S_5 = 25
axiom S_5_eq_25 : S 5 = 25

-- Define what it means for S to be the sum of the first n terms of the arithmetic sequence
axiom sum_arith_seq : ∀ n, S n = n * (a 1 + a n) / 2

theorem a_3_eq_5 : a 3 = 5 :=
by
  -- Proof is skipped using sorry
  sorry

end a_3_eq_5_l349_349767


namespace area_of_triangle_DEF_is_42_l349_349031

-- Given conditions
def DE : ℝ := 12
def height_from_D_to_EF : ℝ := 7

-- Definition of the area of triangle DEF
def area_triangle_DEF (DE : ℝ) (height : ℝ) : ℝ :=
  1/2 * DE * height

-- The proposition to prove
theorem area_of_triangle_DEF_is_42 :
  area_triangle_DEF DE height_from_D_to_EF = 42 :=
by
  sorry

end area_of_triangle_DEF_is_42_l349_349031


namespace vectors_perpendicular_of_equal_magnitudes_l349_349809

variables {α : Type} [InnerProductSpace ℝ α]
variables (a b : α)

theorem vectors_perpendicular_of_equal_magnitudes (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h : ∥a + b∥ = ∥a - b∥) : ⟪a, b⟫ = 0 :=
sorry

end vectors_perpendicular_of_equal_magnitudes_l349_349809


namespace difference_in_circumferences_l349_349716

theorem difference_in_circumferences 
  (d : ℝ) (track_width : ℝ) (h : track_width = 12) :
  let outer_diameter := d + 2 * track_width
  in π * (outer_diameter - d) = 24 * π := by
  -- Conditions
  have h1 : outer_diameter = d + 24 := by simp [outer_diameter, h, mul_comm, mul_assoc, add_comm], sorry
  -- Required to prove the statement
  calc
    π * (outer_diameter - d)
      = π * (d + 24 - d) : by rw h1
  ... = π * 24 : by simp
  ... = 24 * π : by rw mul_comm

end difference_in_circumferences_l349_349716


namespace arithmetic_sequence_condition_l349_349862

theorem arithmetic_sequence_condition (a : ℕ → ℝ) (h : 2 * (a 1 + a 3 + a 5) + 3 * (a 8 + a 10) = 36) : 
a 6 = 3 := 
by 
  sorry

end arithmetic_sequence_condition_l349_349862


namespace circumcircle_intersections_l349_349652

/--
Given an acute-angled triangle ABC with circumcenter O and circumradius R, and points A', B', and C' defined on the circumcircles of triangles BOC, COA, and AOB respectively by the intersections of the lines AO, BO, and CO with their respective circumcircles, prove that 
\[ OA' \cdot OB' \cdot OC' \geq 8R^3 \]. 
Show that equality occurs if and only if ΔABC is equilateral.
-/
theorem circumcircle_intersections (ABC : Type) [EuclideanGeometry ABC] 
  (O : PointABC) (R : length) 
  (A B C : PointABC)
  (A' B' C' : PointABC)
  (h_triangle : acute_angle (AngleABC an A B C))
  (h_O : ∀ {X Y Z : PointABC}, Circle O (Radius X) → lies_on_circumcircle O (Triangle X Y Z)) :
  (length O A') * (length O B') * (length O C') ≥ 8 * (R ^ 3) ∧ 
  ((length O A') * (length O B') * (length O C') = 8 * R^3 ↔ equilateral_triangle A B C) := sorry

end circumcircle_intersections_l349_349652


namespace mean_mode_median_relationship_l349_349712

def dataset_2020_leap_year :=
  (dates: finset ℕ) (occs: ℕ → ℕ)
  (h1 : ∀ d, 1 ≤ d ∧ d ≤ 29 → occs d = 12)
  (h2 : occs 30 = 11)
  (h3 : occs 31 = 7)

noncomputable def mean (s : dataset_2020_leap_year) : ℝ :=
  let numer := ∑ d in s.dates, (s.occs d) * d
  let denom := ∑ d in s.dates, s.occs d
  numer / denom

noncomputable def median (s : dataset_2020_leap_year) : ℕ :=
  let sorted_dates := multiset.sort (≤) (s.dates.val.flat_map (λ d, multiset.repeat d (s.occs d)))
  sorted_dates[(multiset.card sorted_dates) / 2]

noncomputable def mode_median (s : dataset_2020_leap_year) : ℝ :=
  let mode_values := multiset.sort (≤) (multiset.repeat 1 12 ++ multiset.repeat 2 12 ++ ... ++ multiset.repeat 29 12)
  (mode_values[14] + mode_values[15]) / 2

theorem mean_mode_median_relationship (s : dataset_2020_leap_year) :
  let μ := mean s
  let M := median s
  let d := mode_median s
  d < μ ∧ μ < M :=
sorry

end mean_mode_median_relationship_l349_349712


namespace area_of_triangle_ABC_l349_349147

-- Define the setup for the problem
noncomputable def circle_radius : ℝ := 3
noncomputable def BD : ℝ := 4
noncomputable def ED : ℝ := 6

-- Define points A, B, D, E, and C
structure Point :=
(x : ℝ)
(y : ℝ)

def A : Point := ⟨0, 0⟩
def B : Point := ⟨2 * circle_radius, 0⟩
def D : Point := ⟨2 * circle_radius + BD, 0⟩
def E : Point := ⟨2 * circle_radius + BD, ED⟩

-- Define intersection point C
axiom C_between_AE (C : Point) : Prop
axiom AE_intersects_circle_at_C (C : Point) : Prop

-- Define the conditions in the problem
axiom perpendicular_ED_AD : E.y - D.y = 6

-- Define the theorem we want to prove
theorem area_of_triangle_ABC : 
  ∃ (C : Point), C_between_AE C ∧ AE_intersects_circle_at_C C →
  (1 / 2) * abs ((D.x - A.x) * (E.y - A.y) - (D.y - A.y) * (E.x - A.x)) = 36 * sqrt 30.24 / 25 := 
sorry

end area_of_triangle_ABC_l349_349147


namespace symmetric_rw_max_snk_n_sm_eq_prob_diff_l349_349894

theorem symmetric_rw_max_snk_n_sm_eq_prob_diff 
  (n N m : ℕ) 
  (hn : m ≤ N)
  (hprob : p + q = 1) 
  (p q : ℝ)
  (h : 0 ≤ p ∧ 0 ≤ q) 
  (h_symm : p = 1/2 ∧ q = 1/2) :
  (∃ S : ℕ → ℝ, 
    S 0 = 0 ∧
    (∀ k ≥ 1, 
      S k = (finset.range k).sum (λ i, if (decide (rand() ≥ p)) then 1 else -1))) →
  ennreal.to_real (measure_theory.measure_space.prob 
    {ω | (finset.range n).max S ≥ N ∧ S n = m}) = 
  ennreal.to_real (measure_theory.measure_space.prob {ω | S n = 2 * N - m}) -
  ennreal.to_real (measure_theory.measure_space.prob {ω | S n = 2 * N - m + 2}) :=
sorry

end symmetric_rw_max_snk_n_sm_eq_prob_diff_l349_349894


namespace height_of_spruce_tree_l349_349480

theorem height_of_spruce_tree (t : ℚ) (h1 : t = 25 / 64) :
  (∃ s : ℚ, s = 3 / (1 - t) ∧ s = 64 / 13) :=
by
  sorry

end height_of_spruce_tree_l349_349480


namespace JulioHasMoreSoda_l349_349879

-- Define the number of bottles each person has
def JulioOrangeBottles : ℕ := 4
def JulioGrapeBottles : ℕ := 7
def MateoOrangeBottles : ℕ := 1
def MateoGrapeBottles : ℕ := 3

-- Define the volume of each bottle in liters
def BottleVolume : ℕ := 2

-- Define the total liters of soda each person has
def JulioTotalLiters : ℕ := JulioOrangeBottles * BottleVolume + JulioGrapeBottles * BottleVolume
def MateoTotalLiters : ℕ := MateoOrangeBottles * BottleVolume + MateoGrapeBottles * BottleVolume

-- Prove the difference in total liters of soda between Julio and Mateo
theorem JulioHasMoreSoda : JulioTotalLiters - MateoTotalLiters = 14 := by
  sorry

end JulioHasMoreSoda_l349_349879


namespace primer_cost_before_discount_l349_349666

theorem primer_cost_before_discount (primer_cost_after_discount : ℝ) (paint_cost : ℝ) (total_cost : ℝ) 
  (rooms : ℕ) (primer_discount : ℝ) (paint_cost_per_gallon : ℝ) :
  (primer_cost_after_discount = total_cost - (rooms * paint_cost_per_gallon)) →
  (rooms * (primer_cost - primer_discount * primer_cost) = primer_cost_after_discount) →
  primer_cost = 30 := by
  sorry

end primer_cost_before_discount_l349_349666


namespace jars_needed_l349_349222

theorem jars_needed (stars_per_jar : ℕ) (already_made : ℕ) (to_make : ℕ) (n : ℕ) :
  stars_per_jar = 85 → already_made = 33 → to_make = 307 → n = (already_made + to_make) / stars_per_jar → n = 4 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  rw [nat.add_comm] at h4
  exact h4

#eval jars_needed 85 33 307 4 sorry sorry sorry sorry

end jars_needed_l349_349222


namespace probability_distribution_6_balls_4_boxes_l349_349187

theorem probability_distribution_6_balls_4_boxes :
  let n := 4^6
  let m := (Nat.choose 6 3) * (Nat.choose 3 2) * (Nat.choose 1 1) * Finset.univ.perm_count 4
  m = 1440 ∧ n = 4096 → (m / n : ℚ) = 45 / 128 :=
by sorry

end probability_distribution_6_balls_4_boxes_l349_349187


namespace determine_angle_A_l349_349025

noncomputable section

open Real

-- Definition of an acute triangle and its sides
variables {A B : ℝ} {a b : ℝ}

-- Additional conditions that are given before providing the theorem
variables (h1 : 0 < A) (h2 : A < π / 2) (h3 : 0 < B) (h4 : B < π / 2)
          (h5 : 2 * a * sin B = sqrt 3 * b)

-- Theorem statement
theorem determine_angle_A (h1 : 0 < A) (h2 : A < π / 2) (h3 : 0 < B) (h4 : B < π / 2)
  (h5 : 2 * a * sin B = sqrt 3 * b) : A = π / 3 :=
sorry

end determine_angle_A_l349_349025


namespace prime_count_30_to_50_l349_349402

def is_prime (n: ℕ) : Prop :=
  n > 1 ∧ ∀ m: ℕ, m ∣ n → m = 1 ∨ m = n

def primes_in_range (a b: ℕ) : list ℕ :=
  list.filter is_prime (list.range' a (b - a + 1))

theorem prime_count_30_to_50 : (primes_in_range 30 50).length = 5 :=
by sorry

end prime_count_30_to_50_l349_349402


namespace parabola_directrix_l349_349314

theorem parabola_directrix (a : ℝ) (x y : ℝ) : 
  y = -(1/4) * x^2 → ∃ d : ℝ, d = 1/16 ∧ directrix_eq y d :=
by
  sorry

-- Definition of the directrix for a given parabola equation
def directrix_eq (y : ℝ) (d : ℝ) : Prop := y = d

end parabola_directrix_l349_349314


namespace binom_10_2_eq_45_l349_349700

-- Definitions used in the conditions
def binom (n k : ℕ) := n.choose k

-- The statement that needs to be proven
theorem binom_10_2_eq_45 : binom 10 2 = 45 :=
by
  sorry

end binom_10_2_eq_45_l349_349700


namespace probability_of_selecting_articles_l349_349439

theorem probability_of_selecting_articles : 
  let total_clothing := 22 in
  let ways_to_choose_three := Nat.choose total_clothing 3 in
  let ways_to_choose_specific := 6 * 7 * 3 in
  let probability := ways_to_choose_specific / ways_to_choose_three in
  probability = 63 / 770 := 
by
  sorry

end probability_of_selecting_articles_l349_349439


namespace find_ellipse_equation_l349_349350

-- Conditions
variables (a b c : ℝ)
variables (eccentricity perimeter : ℝ)
variable (points_on_ellipse : set (ℝ × ℝ))

-- Given conditions
def is_ellipse (a b : ℝ) := ∀ (x y : ℝ), (x, y) ∈ points_on_ellipse → (x^2 / (a^2) + y^2 / (b^2) = 1)
def is_foci (a c : ℝ) := c = real.sqrt (a^2 - b^2)
def eccentricity_eq (a c : ℝ) := c / a = real.sqrt (3) / 3
def perimeter_eq (a : ℝ) := 4 * a = 4 * real.sqrt 6

-- Proof problem
theorem find_ellipse_equation (a b c : ℝ)
  (h₁ : is_ellipse a b)
  (h₂ : is_foci a c)
  (h₃ : eccentricity_eq a c)
  (h₄ : perimeter_eq a) :
  (a = real.sqrt 6) → (b = 2) → (∀ (x y : ℝ), ((x^2 / 6) + (y^2 / 4) = 1)) :=
by sorry

end find_ellipse_equation_l349_349350


namespace rationalized_value_l349_349218

open Real

theorem rationalized_value :
  let A := 2
  let B := 4
  let C := 3
  A + B + C = 9 :=
by
  -- By rationalizing the denominator of 4 / (3 * (8 ^ (1/3))), it can be shown that:
  -- rationalized form is (2 * (4 ^ (1/3))) / 3
  -- thus A = 2, B = 4, C = 3 and 2 + 4 + 3 = 9
  sorry

end rationalized_value_l349_349218


namespace mushroom_problem_l349_349612

theorem mushroom_problem (n : ℕ) (total_mushrooms : ℕ) 
  (h_total_mushrooms : total_mushrooms = 338) :
  (∀ (set : Finset ℕ), set.card = n → ∑ x in set, x ≤ total_mushrooms → n ≤ 26) ∨
  (∀ (set : Finset ℕ), set.card = n → ∑ x in set, x ≤ total_mushrooms → ∃ (i j : ℕ), i ≠ j ∧ set.to_list.nth i = set.to_list.nth j) := 
sorry

end mushroom_problem_l349_349612


namespace solve_proof_problem_l349_349921

noncomputable def proof_problem (a b c : ℝ) (n : ℕ) (x y z : ℝ) 
  (AB1 B1C BC1 C1A CA1 A1B : ℝ) : Prop :=
  -- Conditions
  (AB1 / B1C = (c^n) / (a^n)) ∧
  (BC1 / C1A = (a^n) / (b^n)) ∧
  (CA1 / A1B = (b^n) / (c^n)) ∧
  -- Proof Goal
  (x / (a^(n-1)) + y / (b^(n-1)) + z / (c^(n-1)) = 0)

theorem solve_proof_problem (a b c : ℝ) (n : ℕ) (x y z : ℝ) 
  (AB1 B1C BC1 C1A CA1 A1B : ℝ) :
  proof_problem a b c n x y z AB1 B1C BC1 C1A CA1 A1B :=
by {
  -- sorry provides a placeholder for the unproven theorem
  sorry,
}

end solve_proof_problem_l349_349921


namespace transformation_correct_l349_349753

theorem transformation_correct (a b : ℝ) (h : a > b) : 2 * a + 1 > 2 * b + 1 :=
by
  sorry

end transformation_correct_l349_349753


namespace triangle_side_BC_length_l349_349844

noncomputable def triangle_side_length
  (AB : ℝ) (angle_a : ℝ) (angle_c : ℝ) : ℝ := 
  let sin_a := Real.sin angle_a
  let sin_c := Real.sin angle_c
  (AB * sin_a) / sin_c

theorem triangle_side_BC_length (AB : ℝ) (angle_a angle_c : ℝ) :
  AB = (Real.sqrt 6) / 2 →
  angle_a = (45 * Real.pi / 180) →
  angle_c = (60 * Real.pi / 180) →
  triangle_side_length AB angle_a angle_c = 1 :=
sorry

end triangle_side_BC_length_l349_349844


namespace quadratic_polynomial_P_l349_349118

noncomputable def P : Polynomial := {
  to_fun := λ x : ℝ, x^2 - x + 2,
  degree := 2
}

theorem quadratic_polynomial_P (P : ℝ → ℝ) 
  (h : ∀ x, P (P x) = x^4 - 2 * x^3 + 4 * x^2 - 3 * x + 4) : 
  P 8 = 58 := 
by
  sorry

end quadratic_polynomial_P_l349_349118


namespace perpendicular_bisector_intersects_at_A_B_l349_349216

noncomputable def circle_with_diameter (O A B : Point) (circ : Circle O) :=
  is_diameter O A B circ

noncomputable def chord_perpendicular (C D : Point) (circ : Circle O) (AB : Line) :=
  is_perpendicular CD AB ∧ C ∈ circ ∧ D ∈ circ ∧ midpoint CD = D

theorem perpendicular_bisector_intersects_at_A_B
  (O A B C D : Point)
  (circ : Circle O)
  (AB : Line)
  (h₁ : is_diameter O A B circ)
  (h₂ : C ∈ circ)
  (h₃ : chord_perpendicular C D circ AB)
  (h₄ : C ≠ A ∧ C ≠ B):
  let BD := perpendicular_bisector CD
  in intersection_points BD circ = {A, B} := 
sorry

end perpendicular_bisector_intersects_at_A_B_l349_349216


namespace region_area_l349_349309

theorem region_area :
  let side_length := 2 in
  side_length * side_length = 4 :=
by
  -- The proof can be filled in here
  sorry

end region_area_l349_349309


namespace binom_10_2_eq_45_l349_349686

theorem binom_10_2_eq_45 : Nat.binomial 10 2 = 45 := by
  sorry

end binom_10_2_eq_45_l349_349686


namespace custom_operator_example_l349_349831

def custom_operator (a b : ℕ) : ℕ := a^2 - 2*a*b + b^2

theorem custom_operator_example : custom_operator 5 3 = 4 := by
  sorry

end custom_operator_example_l349_349831


namespace perfect_square_trinomial_l349_349828

theorem perfect_square_trinomial (m : ℤ) : 
  (∃ x y : ℝ, 16 * x^2 + m * x * y + 25 * y^2 = (4 * x + 5 * y)^2 ∨ 16 * x^2 + m * x * y + 25 * y^2 = (4 * x - 5 * y)^2) ↔ (m = 40 ∨ m = -40) :=
by
  sorry

end perfect_square_trinomial_l349_349828


namespace cassinis_identity_l349_349673

-- Definition of Fibonacci numbers
def Fibonacci : ℕ → ℕ
| 0       := 0
| 1       := 1
| (n + 2) := Fibonacci (n + 1) + Fibonacci n

theorem cassinis_identity (n : ℕ) (h : n > 0) :
  Fibonacci (n + 1) * Fibonacci (n - 1) - Fibonacci n * Fibonacci n = (-1)^n :=
sorry

end cassinis_identity_l349_349673


namespace infinite_planar_graph_coloring_l349_349569

-- Define what it means for a finite planar graph to be 3-colored without odd monochromatic cycles.
def finite_coloring_property (G : Type) [graph G] : Prop :=
  ∃ (C : vertex G -> color), (∀ cycle, cycle_length cycle % 2 = 1 → ¬monochromatic cycle C)

-- Define the theorem for countably infinite planar graphs.
theorem infinite_planar_graph_coloring
  (G : Type) [graph G] (h : fin_colorable_property G) : 
  ∃ (C : vertex G -> color), (∀ cycle, cycle_length cycle % 2 = 1 → ¬monochromatic cycle C) :=
sorry

end infinite_planar_graph_coloring_l349_349569


namespace distance_focus_directrix_parabola_l349_349945

theorem distance_focus_directrix_parabola (p : ℝ) (h : y^2 = 20 * x) : 
  2 * p = 10 :=
by
  -- h represents the given condition y^2 = 20x.
  sorry

end distance_focus_directrix_parabola_l349_349945


namespace count_primes_between_30_and_50_l349_349426

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between_30_and_50 : List ℕ :=
  [31, 37, 41, 43, 47]

theorem count_primes_between_30_and_50 : 
  (primes_between_30_and_50.filter is_prime).length = 5 :=
by
  sorry

end count_primes_between_30_and_50_l349_349426


namespace length_CF_equals_l349_349871

-- Definitions for points and lengths
variable (A B C D E F : Type)
variable [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F]
variable (AB BC CA : ℝ)
variable [fact (AB = 13)] [fact (BC = 26)] [fact (CA = 24)]

-- Angles and intersection points
variable (angleBAC : angle A B C)
variable (bisectorBAC D : Type) [IsBisector angleBAC (lineSegment A C) (lineSegment D C)]
variable [OnCircumcircle E (triangle A B C)]
variable [Intersects E (triangle A B C) (angleBisector A B C) (lineSegment B E)]
variable [IsOnLine F (lineSegment A B)]
variable [fact (F ≠ B)]

theorem length_CF_equals :
  ∃ (CF : ℝ), CF = 769/37 :=
by
  sorry

end length_CF_equals_l349_349871


namespace simplify_large_exp_division_l349_349136

theorem simplify_large_exp_division :
  (27 * 10^12) / (9 * 10^5) = 30_000_000 := by
  sorry

end simplify_large_exp_division_l349_349136


namespace primes_between_30_and_50_l349_349393

theorem primes_between_30_and_50 : (Finset.card (Finset.filter Nat.Prime (Finset.Ico 30 51))) = 5 :=
by
  sorry

end primes_between_30_and_50_l349_349393


namespace P_eight_value_l349_349096

def quadratic_polynomial (P : ℤ → ℤ) : Prop :=
  ∃ a b c : ℤ, ∀ x : ℤ, P(x) = a * x^2 + b * x + c

theorem P_eight_value (P : ℤ → ℤ)
  (H : ∀ x : ℤ, P(P(x)) = x^4 - 2 * x^3 + 4 * x^2 - 3 * x + 4)
  (H_quad : quadratic_polynomial P) :
  P 8 = 58 :=
sorry

end P_eight_value_l349_349096


namespace tau_phi_inequality_l349_349494

variable (n : ℕ)

-- Definition of the number of divisors function
noncomputable def tau (n : ℕ) : ℕ := 
  Nat.divisors n |>.length

-- Definition of Euler's totient function
noncomputable def phi (n : ℕ) : ℕ := 
  if n = 0 then 0 else n * ((Finset.filter (fun x => Nat.coprime x n) (Finset.range n)).card) / n

theorem tau_phi_inequality (n : ℕ) : phi n * tau n ≥ n := by
  sorry

end tau_phi_inequality_l349_349494


namespace volume_of_regular_triangular_pyramid_l349_349960

theorem volume_of_regular_triangular_pyramid (l h : ℝ) : 
  (V = (sqrt 3 * h * (l^2 - h^2)) / 4) :=
sorry

end volume_of_regular_triangular_pyramid_l349_349960


namespace binom_10_2_eq_45_l349_349687

theorem binom_10_2_eq_45 : Nat.binomial 10 2 = 45 := by
  sorry

end binom_10_2_eq_45_l349_349687


namespace binomial_10_2_l349_349694

noncomputable def binom (n k : ℕ) : ℕ := n! / (k! * (n - k)!)

theorem binomial_10_2 : binom 10 2 = 45 := by
  sorry

end binomial_10_2_l349_349694


namespace quadratic_polynomial_P_l349_349119

noncomputable def P : Polynomial := {
  to_fun := λ x : ℝ, x^2 - x + 2,
  degree := 2
}

theorem quadratic_polynomial_P (P : ℝ → ℝ) 
  (h : ∀ x, P (P x) = x^4 - 2 * x^3 + 4 * x^2 - 3 * x + 4) : 
  P 8 = 58 := 
by
  sorry

end quadratic_polynomial_P_l349_349119


namespace find_k_l349_349543

theorem find_k : 
  (∃ y, -3 * (-15) + y = k ∧ 0.3 * (-15) + y = 10) → k = 59.5 :=
by
  sorry

end find_k_l349_349543


namespace range_of_a_l349_349370

noncomputable def f (a x : ℝ) : ℝ := (x^2 - a * x + 1) * exp x

theorem range_of_a (a : ℝ) (h₀ : 0 ≤ a) :
  (∀ x, 0 ≤ x ∧ x ≤ 1 → f a x ≥ 1) ↔ 0 ≤ a ∧ a ≤ 1 :=
by
  sorry

end range_of_a_l349_349370


namespace factor_expression_correct_l349_349730

variable (y : ℝ)

def expression := 4 * y * (y + 2) + 6 * (y + 2)

theorem factor_expression_correct : expression y = (y + 2) * (2 * (2 * y + 3)) :=
by
  sorry

end factor_expression_correct_l349_349730


namespace problem_area_of_triangle_ABC_l349_349145

noncomputable def area_of_triangle_ABC : ℚ :=
  let r := 3
  let AB := 2 * r
  let BD := 4
  let AD := AB + BD
  let DE := 6
  let EA := Real.sqrt (AD^2 + DE^2)
  let R := 3
  let EC := 76 / EA
  let AC := EA - EC
  let BC := Real.sqrt (R^2 - AC^2)
  (BC * AC) / 2

theorem problem_area_of_triangle_ABC :
  area_of_triangle_ABC = 270 / 34 :=
sorry

end problem_area_of_triangle_ABC_l349_349145


namespace perimeter_of_triangle_l349_349955

theorem perimeter_of_triangle (r a : ℝ) (p : ℝ) (h1 : r = 3.5) (h2 : a = 56) :
  p = 32 :=
by
  sorry

end perimeter_of_triangle_l349_349955


namespace contribution_of_B_l349_349601

noncomputable def B_contribution : ℝ :=
  let investment_A := 4500
  let time_A := 12
  let profit_ratio_A := 2
  let time_B := 5
  let profit_ratio_B := 3
  let x := (investment_A * time_A * profit_ratio_B) / (time_B * profit_ratio_A)
  x

theorem contribution_of_B : B_contribution = 16200 := 
  by
    have h : B_contribution = (4500 * 12 * 3) / (5 * 2) := rfl
    have t : (4500 * 12 * 3) / (5 * 2) = 16200 := by norm_num
    rw [h, t]
    rfl

end contribution_of_B_l349_349601


namespace area_of_shaded_region_l349_349735

theorem area_of_shaded_region:
  let line1 := (0, 5, 10, 2)
  let line2 := (2, 6, 9, 1) in
  ∃ (area : ℚ), area = 39/35 :=
sorry

end area_of_shaded_region_l349_349735


namespace problem_statement_l349_349777

open Real

theorem problem_statement (α : ℝ) 
  (h1 : cos (α + π / 4) = (7 * sqrt 2) / 10)
  (h2 : cos (2 * α) = 7 / 25) :
  sin α + cos α = 1 / 5 :=
sorry

end problem_statement_l349_349777


namespace binomial_arithmetic_series_l349_349303

theorem binomial_arithmetic_series {n k : ℕ} (h1 : 2 < k) (h2 : k < n)
  (h3 : nat.choose n (k-1) + nat.choose n (k+1) = 2 * nat.choose n k) :
  ∃ p : ℤ, n = 4 * p^2 - 2 ∧ (k = 2 * p^2 - 1 + p ∨ k = 2 * p^2 - 1 - p) :=
sorry

end binomial_arithmetic_series_l349_349303


namespace P_eight_value_l349_349095

def quadratic_polynomial (P : ℤ → ℤ) : Prop :=
  ∃ a b c : ℤ, ∀ x : ℤ, P(x) = a * x^2 + b * x + c

theorem P_eight_value (P : ℤ → ℤ)
  (H : ∀ x : ℤ, P(P(x)) = x^4 - 2 * x^3 + 4 * x^2 - 3 * x + 4)
  (H_quad : quadratic_polynomial P) :
  P 8 = 58 :=
sorry

end P_eight_value_l349_349095


namespace option_A_option_B_option_D_l349_349042

-- Definitions to represent the problem conditions
variables {a b c : ℝ} {A B C : ℝ}
variables {ΔABC : Prop} -- Assume ΔABC implies angles are A, B, C and sides are a, b, c
variables (acute_triangle : ΔABC → A < π / 2 ∧ B < π / 2 ∧ C < π / 2)

-- Proof goal 1: If a > b, then sin A > sin B
theorem option_A (h : ΔABC) (h1 : a > b) : sin A > sin B := 
sorry

-- Proof goal 2: If sin A > sin B, then A > B
theorem option_B (h : ΔABC) (h1 : sin A > sin B) : A > B := 
sorry

-- Proof goal 3: If triangle ABC is acute, then sin A > cos B
theorem option_D (h : ΔABC) (h1 : acute_triangle h) : sin A > cos B := 
sorry

end option_A_option_B_option_D_l349_349042


namespace polynomial_value_at_8_l349_349107

noncomputable def P : ℝ → ℝ := λ x, x^2 - x + 2

theorem polynomial_value_at_8 :
  (P (P x) = x^4 - 2*x^3 + 4*x^2 - 3*x + 4) ∧ (P = λ x, x^2 - x + 2) → P 8 = 58 :=
by
  sorry

end polynomial_value_at_8_l349_349107


namespace am_gm_inequality_l349_349338

-- Definitions of the variables and hypotheses
variables {a b : ℝ}

-- The theorem statement
theorem am_gm_inequality (h : a * b > 0) : a / b + b / a ≥ 2 :=
sorry

end am_gm_inequality_l349_349338


namespace molecular_weight_compound_l349_349186

def atomic_weight_H : ℝ := 1.01
def atomic_weight_Br : ℝ := 79.90
def atomic_weight_O : ℝ := 16.00

def molecular_weight (n_H n_Br n_O : ℕ) : ℝ :=
  n_H * atomic_weight_H + n_Br * atomic_weight_Br + n_O * atomic_weight_O

theorem molecular_weight_compound : 
  molecular_weight 1 1 3 = 128.91 :=
by
  -- This is where the proof would go
  sorry

end molecular_weight_compound_l349_349186


namespace time_to_walk_against_walkway_150_l349_349247

def v_p := 4 / 3
def v_w := 2 - v_p
def distance := 100
def time_against_walkway := distance / (v_p - v_w)

theorem time_to_walk_against_walkway_150 :
  time_against_walkway = 150 := by
  -- Note: Proof goes here (not required)
  sorry

end time_to_walk_against_walkway_150_l349_349247


namespace number_of_valid_b_l349_349747

theorem number_of_valid_b : ∃ (bs : Finset ℂ), bs.card = 2 ∧ ∀ b ∈ bs, ∃ (x : ℂ), (x + b = b^2) :=
by
  sorry

end number_of_valid_b_l349_349747


namespace sues_answer_l349_349259

theorem sues_answer (x : ℕ) (hx : x = 6) : 
  let b := 2 * (x + 1)
  let s := 2 * (b - 1)
  s = 26 :=
by
  sorry

end sues_answer_l349_349259


namespace number_of_primes_between_30_and_50_l349_349419

/-- 
  Prove that there are exactly 5 prime numbers in the range from 30 to 50. 
  These primes are 31, 37, 41, 43, and 47.
-/
theorem number_of_primes_between_30_and_50 : 
  (Finset.filter Nat.Prime (Finset.range 51)).card - 
  (Finset.filter Nat.Prime (Finset.range 31)).card = 5 := 
by 
  sorry

end number_of_primes_between_30_and_50_l349_349419


namespace area_of_triangle_COB_l349_349079

-- Define the points C and B
variables (q a : ℝ) (h1 : q ∈ set.Icc 0 15) (h2 : a > 0)

-- Define the area function and the proof statement
def area_triangle_COB (q a : ℝ) : ℝ := (1 / 2) * a * q

theorem area_of_triangle_COB : area_triangle_COB q a = (1 / 2) * a * q :=
by sorry

end area_of_triangle_COB_l349_349079


namespace train_length_correct_l349_349602

-- Definitions based on the conditions
def speed_kmh : ℝ := 60  -- speed of train in km/hr
def time_sec : ℝ := 6    -- time to cross the pole in seconds

-- Convert speed to m/s
def speed_ms : ℝ := speed_kmh * (1000 / 3600)

-- Length of the train
def length_train : ℝ := speed_ms * time_sec

-- The theorem to prove
theorem train_length_correct : length_train = 100.02 := sorry

end train_length_correct_l349_349602


namespace set_diff_N_M_l349_349720

universe u

def set_difference {α : Type u} (A B : Set α) : Set α :=
  { x | x ∈ A ∧ x ∉ B }

def M : Set ℕ := { 1, 2, 3, 4, 5 }
def N : Set ℕ := { 1, 2, 3, 7 }

theorem set_diff_N_M : set_difference N M = { 7 } :=
  by
    sorry

end set_diff_N_M_l349_349720


namespace exists_ell_and_N_l349_349904

noncomputable def a (r : ℕ) (a₀ : Fin r → ℝ) (n : ℕ) : ℝ :=
if h : n ≤ r then a₀ ⟨n - 1, Nat.pred_lt_pred h⟩
else ⨆ k in Finset.range n, a (r) (a₀) ((n - k) - 1) + a (r) (a₀) (k - 1)

theorem exists_ell_and_N (r : ℕ) (a₀ : Fin r → ℝ) (h_pos : ∀ i, 0 < a₀ i) :
  ∃ (ℓ N : ℕ), ℓ ≤ r ∧ (∀ n ≥ N, a r a₀ n = a r a₀ (n - ℓ) + a r a₀ ℓ) :=
sorry

end exists_ell_and_N_l349_349904


namespace tan_product_of_angles_l349_349965

theorem tan_product_of_angles (tan_21 tan_24 : ℝ) :
  (1 + tan 21) * (1 + tan 24) = 2 :=
by
  -- Given conditions
  have h1: tan 45 = 1 := by sorry
  have h2: tan (21 + 24) = (tan 21 + tan 24) / (1 - tan 21 * tan 24) := by sorry

  -- Use the identities and prove the final expression
  sorry

end tan_product_of_angles_l349_349965


namespace binomial_10_2_l349_349696

noncomputable def binom (n k : ℕ) : ℕ := n! / (k! * (n - k)!)

theorem binomial_10_2 : binom 10 2 = 45 := by
  sorry

end binomial_10_2_l349_349696


namespace domain_logarithm_l349_349946

theorem domain_logarithm : 
  {x : ℝ | log 2 (x^2 - 4)} = {x : ℝ | x < -2 ∨ x > 2} :=
by sorry

end domain_logarithm_l349_349946


namespace distance_to_center_l349_349002

def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 = 8 * x - 4 * y + 16

def point := (3 : ℝ, -1 : ℝ)

def circle_center := (4 : ℝ, -2 : ℝ)

theorem distance_to_center :
  let d := (λ (p1 p2: ℝ × ℝ), real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2))
  in d point circle_center = real.sqrt 2 :=
by
  sorry

end distance_to_center_l349_349002


namespace proof_problem_l349_349793

variables (a b x y : ℝ)

-- Condition 1: Ellipse passes through point M(2, 1)
def ellipse_eq (a b : ℝ) : Prop :=
  (2^2 / a^2 + 1^2 / b^2 = 1)

-- Condition 2: Eccentricity
def eccentricity (a b : ℝ) : Prop :=
  (Real.sqrt (1 - b^2 / a^2) = Real.sqrt 3 / 2)

-- Question 1: Equation of the ellipse
def equation_of_ellipse : Prop :=
  (a^2 = 8 ∧ b^2 = 2 ∧ ∀ x y : ℝ, (x^2 / 8 + y^2 / 2 = 1))

-- Condition 3: Line passing through origin intersects ellipse
def line_through_origin (l1_eq : Prop) : Prop :=
  ∃ P Q : ℝ × ℝ, P ≠ Q ∧ (P.fst^2 / 8 + P.snd^2 / 2 = 1 ∧ Q.fst^2 / 8 + Q.snd^2 / 2 = 1) ∧ l1_eq

-- Condition 4: Point M on line l2
def point_on_line_l2 (k : ℝ) : Prop :=
  ∃ (M : ℝ × ℝ), M.1 - M.2 + 2 * Real.sqrt 6 = 0 → 
  (∃ P Q : ℝ × ℝ, P ≠ Q ∧ equation_of_ellipse ∧ 
                   ∃ l1_eq : Prop, is_eq_triangle P Q M ∧ line_through_origin l1_eq) 

-- Question 2: Equation of the line l1
def equation_of_line_l1 : Prop :=
  (∀ k : ℝ, k = 0 ∨ k = 2 / 7)

theorem proof_problem :
  (ellipse_eq a b) →
  (eccentricity a b) →
  equation_of_ellipse →
  (∀ k : ℝ, point_on_line_l2 k → equation_of_line_l1) :=
sorry

end proof_problem_l349_349793


namespace toys_between_l349_349220

theorem toys_between (n : ℕ) (a b : ℕ) 
  (h_n : n = 19) 
  (h_a : a = 9) 
  (h_b : b = 15) : 
  b - a - 1 = 5 := 
by 
  rw [h_a, h_b] 
  norm_num 
  sorry

end toys_between_l349_349220


namespace factory_toys_production_each_day_l349_349238

theorem factory_toys_production_each_day 
  (weekly_production : ℕ)
  (days_worked_per_week : ℕ)
  (h1 : weekly_production = 4560)
  (h2 : days_worked_per_week = 4) : 
  (weekly_production / days_worked_per_week) = 1140 :=
  sorry

end factory_toys_production_each_day_l349_349238


namespace books_received_l349_349518

theorem books_received (initial_books : ℕ) (total_books : ℕ) (h1 : initial_books = 54) (h2 : total_books = 77) : (total_books - initial_books) = 23 :=
by
  sorry

end books_received_l349_349518


namespace max_length_OB_l349_349979

theorem max_length_OB :
  ∀ (O A B : Type) [metric_space O] [metric_space A] [metric_space B]
    (OA OB : metric O) (angle_O : real) (AB : real),
  angle_O = pi / 4 →  -- $\angle AOB = 45^\circ$
  AB = real.sqrt 2 →
  (∃ (OB_max : real), OB ≤ OB_max) →
  OB_max = 2 :=
begin
  intros O A B _ _ OA OB angle_O AB h_angle h_AB h_OB_max,
  sorry,
end

end max_length_OB_l349_349979


namespace bottle_caps_division_l349_349558

theorem bottle_caps_division (total_bottle_caps : ℕ) (num_groups : ℕ) (num_each_group : ℕ)
  (h1 : total_bottle_caps = 35)
  (h2 : num_groups = 7) :
  num_each_group = total_bottle_caps / num_groups :=
by
  have h3 : num_each_group = 35 / 7,
  { -- This is the part where the specific calculation would go,
    -- but we'll just add sorry to skip the proof.
    sorry },
  exact h3

end bottle_caps_division_l349_349558


namespace average_of_numbers_divisible_by_4_between_6_and_30_l349_349736

theorem average_of_numbers_divisible_by_4_between_6_and_30 : 
  let S := {n ∈ Finset.range 31 | n ≥ 6 ∧ n % 4 = 0} in 
  (Finset.sum S id) / (S.card : ℝ) = 18 :=
by
  sorry

end average_of_numbers_divisible_by_4_between_6_and_30_l349_349736


namespace car_mileage_city_l349_349227

theorem car_mileage_city (h c t : ℕ) 
  (h_eq_tank_mileage : 462 = h * t) 
  (c_eq_tank_mileage : 336 = c * t) 
  (mileage_diff : c = h - 3) : 
  c = 8 := 
by
  sorry

end car_mileage_city_l349_349227


namespace polynomial_value_at_8_l349_349109

noncomputable def P : ℝ → ℝ := λ x, x^2 - x + 2

theorem polynomial_value_at_8 :
  (P (P x) = x^4 - 2*x^3 + 4*x^2 - 3*x + 4) ∧ (P = λ x, x^2 - x + 2) → P 8 = 58 :=
by
  sorry

end polynomial_value_at_8_l349_349109


namespace eval_expression_at_neg_one_l349_349194

theorem eval_expression_at_neg_one : 
  let x := -1 in 
  x^2 + 5 * x - 6 = -10 :=
by
  let x := -1
  sorry

end eval_expression_at_neg_one_l349_349194


namespace math_problem_l349_349235

variable (f : ℝ → ℝ)
variable (x : ℝ)

-- Define conditions
-- Condition 1: f is an even function
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- Condition 2: f has a period of 2
def is_periodic (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 2) = f x

-- Condition 3: f is monotonically increasing on [0,1]
def is_monotonically_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x ∧ x ≤ y ∧ y ≤ 1 → f x ≤ f y

-- Define variables a, b, c
def a : ℝ := f 3
def b : ℝ := f (Real.sqrt 2)
def c : ℝ := f 2

-- Proof Problem Statement
theorem math_problem (hf_even : is_even_function f) (hf_periodic : is_periodic f) (hf_mono_inc : is_monotonically_increasing f) :
  a f = f 3 ∧ b f = f (Real.sqrt 2) ∧ c f = f 2 →
  a f = f 1 ∧ b f = f (2 - Real.sqrt 2) ∧ c f = f 0 →
  a f > b f ∧ b f > c f := by
  sorry

end math_problem_l349_349235


namespace proof_problem_l349_349039

-- Definitions and conditions
variables {A B C : ℝ} {a b c : ℝ}

-- This condition describes a triangle with sides opposite to angles A, B, and C
axiom triangle_ABC (h₁ : 0 < A) (h₂ : 0 < B) (h₃ : 0 < C) 
                   (h₁₂ : A + B + C = π) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
                   (h_sine_rule : ∀ a b c A B C, a / sin A = b / sin B = c / sin C) :

-- Option A
def option_A (h : a > b) : Prop :=
  sin A > sin B

-- Option B
def option_B (h : sin A > sin B) : Prop :=
  A > B

-- Option C
def option_C (h : a * cos A = b * cos B) : Prop :=
  a = b ∧ A = B

-- Option D
def triangle_is_acute (h : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2) : Prop :=
  A + B + C = π

def option_D (h : triangle_is_acute)
            (h_sine_cosine : ∀ A B, sin A = cos B) : Prop :=
  sin A > cos B

-- The theorem that encapsulates all conditions and options
theorem proof_problem :
  (triangle_ABC h₁ h₂ h₃ h₁₂ ha hb hc h_sine_rule) →
  (∀ h, option_A h) →
  (∀ h, option_B h) →
  (∀ h, ¬ option_C h) →
  (∀ hₐ, option_D hₐ h_sine_cosine) :=
sorry

end proof_problem_l349_349039


namespace foldable_positions_l349_349249

def isFoldable (n : ℕ) : Prop := (1 ≤ n ∧ n ≤ 4)

theorem foldable_positions : ∃ p : ℕ, p = 12 ∧ (∑ i in finset.range p, if isFoldable (i+1) then 1 else 0) = 4 :=
by
  sorry

end foldable_positions_l349_349249


namespace parabola_directrix_l349_349313

theorem parabola_directrix (a : ℝ) (x y : ℝ) : 
  y = -(1/4) * x^2 → ∃ d : ℝ, d = 1/16 ∧ directrix_eq y d :=
by
  sorry

-- Definition of the directrix for a given parabola equation
def directrix_eq (y : ℝ) (d : ℝ) : Prop := y = d

end parabola_directrix_l349_349313


namespace ellipse_equation_and_triangle_area_l349_349773

-- Defining the problem conditions
theorem ellipse_equation_and_triangle_area :
  ∀ (a b x y : ℝ) (h1 : a > b > 0)
    (h2 : 6 / a^2 + 2 / b^2 = 1)
    (h3 : (a^2 = b^2 + (√6)^2))
    (h4 : M : ℝ × ℝ) (h5 : M = (√6, √2))
    (h6 : e : ℝ) (h7 : e = √6 / 3)
    (P : ℝ × ℝ) (hP : P = (-3, 2))
    (l_slope : ℝ) (hl_slope : l_slope = 1),
    
    -- First statement: the equation of the ellipse
    (a^2 = 12 ∧ b^2 = 4 ∧ h2 ∧ h3) →
    (
      -- Second statement: the area of the triangle PAB
      -- If a line l with slope 1 intersects the ellipse at points A and B
      ∃ (x1 x2 y1 y2 : ℝ),
        (
          -- Points A and B lie on both the line and the ellipse
          y1 = x1 + 2 ∧ y2 = x2 + 2 ∧
          (x1^2 / 12 + y1^2 / 4 = 1) ∧
          (x2^2 / 12 + y2^2 / 4 = 1) ∧
          -- Calculate length of AB and distance from P to AB
          (∀ (AB : ℝ), AB = √2[((x1 + x2)^2 / 4) + ((y1 + y2)^2 / 4)] ∧
          ∀ (dist_P_AB : ℝ), dist_P_AB = |(-3 + 0 + 2)| / √2) ∧
          -- Calculate the area of the triangle PAB
          (∀ (area_PAB : ℝ), area_PAB = 1 / 2 * dist_P_AB * AB ∧ area_PAB = 9 / 2)
        )
    ) :=
sorry

end ellipse_equation_and_triangle_area_l349_349773


namespace num_comics_liked_by_males_l349_349077

-- Define the problem conditions
def num_comics : ℕ := 300
def percent_liked_by_females : ℕ := 30
def percent_disliked_by_both : ℕ := 30

-- Define the main theorem to prove
theorem num_comics_liked_by_males :
  let percent_liked_by_at_least_one_gender := 100 - percent_disliked_by_both
  let num_comics_liked_by_females := percent_liked_by_females * num_comics / 100
  let num_comics_liked_by_at_least_one_gender := percent_liked_by_at_least_one_gender * num_comics / 100
  num_comics_liked_by_at_least_one_gender - num_comics_liked_by_females = 120 :=
by
  sorry

end num_comics_liked_by_males_l349_349077


namespace number_of_integers_satisfying_inequality_l349_349742

theorem number_of_integers_satisfying_inequality :
  {n : ℤ | 50 < n^2 ∧ n^2 < 200}.finite.card = 14 :=
by
  sorry

end number_of_integers_satisfying_inequality_l349_349742


namespace orvin_max_balloons_l349_349920

variable (C : ℕ) (P : ℕ)

noncomputable def max_balloons (C P : ℕ) : ℕ :=
  let pair_cost := P + P / 2  -- Cost for two balloons
  let pairs := C / pair_cost  -- Maximum number of pairs
  pairs * 2 + (if C % pair_cost >= P then 1 else 0) -- Total balloons considering the leftover money

theorem orvin_max_balloons (hC : C = 120) (hP : P = 3) : max_balloons C P = 53 :=
by
  sorry

end orvin_max_balloons_l349_349920


namespace positive_difference_l349_349580

noncomputable def calculate_diff : ℕ :=
  let first_term := (8^2 - 8^2) / 8
  let second_term := (8^2 * 8^2) / 8
  second_term - first_term

theorem positive_difference : calculate_diff = 512 := by
  sorry

end positive_difference_l349_349580


namespace number_of_yellow_balls_l349_349617

-- Definitions based on conditions
def number_of_red_balls : ℕ := 10
def probability_red_ball := (1 : ℚ) / 3

-- Theorem stating the number of yellow balls
theorem number_of_yellow_balls :
  ∃ (y : ℕ), (number_of_red_balls : ℚ) / (number_of_red_balls + y) = probability_red_ball ∧ y = 20 :=
by
  sorry

end number_of_yellow_balls_l349_349617


namespace part1_part2_l349_349348

variables {ABC : Type*} [triangle ABC]

def circumcenter (ABC : Type*) : Type* := sorry
def incenter (ABC : Type*) : Type* := sorry
def centroid (ABC : Type*) : Type* := sorry
def distance_to_side (p : Point) (side : Line) : Real := sorry
def side_a (ABC : Type*) : Real := sorry
def side_b (ABC : Type*) : Real := sorry
def side_c (ABC : Type*) : Real := sorry
def angle_A (ABC : Type*) : Real := sorry
def angle_C (ABC : Type*) : Real := sorry

theorem part1 (ABC : Type*) [triangle ABC] (O : Point) (I : Point) (G : Point)
    (r0 r1 : Real) (h1 : O = circumcenter ABC) (h2 : I = incenter ABC)
    (h3 : distance_to_side O (side_AC ABC) = r0)
    (h4 : distance_to_side I (side_AC ABC) = r1) (h5 : r0 = r1) :
  cos (angle_A ABC) + cos (angle_C ABC) = 1 := 
sorry

theorem part2 (ABC : Type*) [triangle ABC] (O : Point) (I : Point) (G : Point)
    (r1 r2 : Real) (h1 : I = incenter ABC) (h2 : G = centroid ABC)
    (h3 : distance_to_side I (side_AC ABC) = r1)
    (h4 : distance_to_side G (side_AC ABC) = r2) (h5 : r1 = r2) :
  (side_a ABC) + (side_c ABC) = 2 * (side_b ABC) := 
sorry

end part1_part2_l349_349348


namespace cos_x_plus_pi_over_6_l349_349335

theorem cos_x_plus_pi_over_6 (x : ℝ) (h : real.sin (π / 3 - x) = 3 / 5) : 
  real.cos (x + π / 6) = 3 / 5 := 
by sorry

end cos_x_plus_pi_over_6_l349_349335


namespace matrix_N_satisfies_l349_349320

open Matrix

variable {α : Type*} [Fintype α] [DecidableEq α]
variable {R : Type*} [CommRing R]

def N : Matrix (Fin 2) (Fin 2) ℝ := !![2, 2; 1, 2]

theorem matrix_N_satisfies (N : Matrix (Fin 2) (Fin 2) ℝ)
  (h : N^3 - 3•N^2 + 4•N = !![6, 12; 3, 6]) :
  N = !![2, 2; 1, 2] := 
sorry

end matrix_N_satisfies_l349_349320


namespace count_quadruples_ad_bc_even_l349_349056

theorem count_quadruples_ad_bc_even : 
  let S := {0, 1, 2, 3, 4} in
  {p : S × S × S × S | let (a, b, c, d) := p in (a * d - b * c) % 2 = 0 }.to_finset.card = 97 := 
by
  sorry

end count_quadruples_ad_bc_even_l349_349056


namespace largest_int_less_than_100_rem_5_by_7_l349_349316

theorem largest_int_less_than_100_rem_5_by_7 :
  ∃ k : ℤ, (7 * k + 5 = 96) ∧ ∀ n : ℤ, (7 * n + 5 < 100) → (n ≤ k) :=
sorry

end largest_int_less_than_100_rem_5_by_7_l349_349316


namespace maggie_bought_yellow_packs_l349_349512

theorem maggie_bought_yellow_packs :
  ∃ y_packs : ℕ, y_packs = 8 ∧
  let r_packs := 4,
      g_packs := 4,
      n_balls := 10,
      B_total := 160,
      red_balls := r_packs * n_balls,
      green_balls := g_packs * n_balls,
      total_rg_balls := red_balls + green_balls,
      yellow_balls := B_total - total_rg_balls in
  yellow_balls / n_balls = y_packs :=
by sorry

end maggie_bought_yellow_packs_l349_349512


namespace hike_total_distance_and_daily_walks_l349_349974

theorem hike_total_distance_and_daily_walks (D d1 d2 d3 : ℝ) (h1 : d1 = D / 3) (h2 : d2 = (2 * (D - d1)) / 9) (h3 : d3 = (D - d1 - d2) / 4) (h4 : D / 3 = 24) : 
  D = 72 ∧ d1 = 24 ∧ d2 = 16 ∧ d3 = 8 :=
begin
  sorry
end

end hike_total_distance_and_daily_walks_l349_349974


namespace sin_cos_fourth_power_l349_349495

theorem sin_cos_fourth_power (θ : ℝ) (h : Real.sin (2 * θ) = 1 / 4) : Real.sin θ ^ 4 + Real.cos θ ^ 4 = 63 / 64 :=
by
  sorry

end sin_cos_fourth_power_l349_349495


namespace number_of_primes_between_30_and_50_l349_349437

-- Define what it means to be a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the interval condition
def in_interval (n : ℕ) : Prop :=
  30 ≤ n ∧ n ≤ 50

-- Define the count of prime numbers in the interval
def prime_count_in_interval : ℕ :=
  (List.range' 30 21).countp (λ n, is_prime n)

-- We state that the above count is equal to 5
theorem number_of_primes_between_30_and_50 : prime_count_in_interval = 5 :=
  sorry

end number_of_primes_between_30_and_50_l349_349437


namespace triangle_inequality_l349_349503

theorem triangle_inequality 
(a b c : ℝ) (α β γ : ℝ)
(h_t : a + b > c ∧ a + c > b ∧ b + c > a)
(h_opposite : 0 < α ∧ α < π ∧ 0 < β ∧ β < π ∧ 0 < γ ∧ γ < π ∧ α + β + γ = π) :
  a * α + b * β + c * γ ≥ a * β + b * γ + c * α :=
sorry

end triangle_inequality_l349_349503


namespace distance_from_center_to_chord_l349_349342

theorem distance_from_center_to_chord (O A : Point) (center : Point) 
  (C : Circle) (h₀ : O = (0, 0)) (h₁ : A = (4, 2)) 
  (h₂ : C.center lies_on Line.mk (1, 2, -1)) 
  (h₃ : C.passes_through O ∧ C.passes_through A) 
  : distance_to_chord center O A = 2 / (5 * sqrt 5) :=
sorry

end distance_from_center_to_chord_l349_349342


namespace volume_of_revolved_region_l349_349493

theorem volume_of_revolved_region :
  let R := {p : ℝ × ℝ | |8 - p.1| + p.2 ≤ 10 ∧ 3 * p.2 - p.1 ≥ 15}
  let volume := (1 / 3) * Real.pi * (7 / Real.sqrt 10)^2 * (7 * Real.sqrt 10 / 4)
  let m := 343
  let n := 12
  let p := 10
  m + n + p = 365 := by
  sorry

end volume_of_revolved_region_l349_349493


namespace graphs_symmetric_origin_l349_349542

theorem graphs_symmetric_origin (x y : ℝ) :
  (y = 3^x → ∃ x' y', (x' = -x ∧ y' = -y ∧ y' = -3^(-x'))) →
  symmetric_with_respect_to_origin (λ x, 3^x) (λ x, -3^(-x)) :=
by
  sorry  -- Proof to be filled in

end graphs_symmetric_origin_l349_349542


namespace infinite_triangular_pairs_l349_349964

theorem infinite_triangular_pairs : ∃ (a_i b_i : ℕ → ℕ), (∀ m : ℕ, ∃ n : ℕ, m = n * (n + 1) / 2 ↔ ∃ k : ℕ, a_i k * m + b_i k = k * (k + 1) / 2) ∧ ∀ j : ℕ, ∃ k : ℕ, k > j :=
by {
  sorry
}

end infinite_triangular_pairs_l349_349964


namespace solve_system_of_equations_l349_349934

theorem solve_system_of_equations (x y : ℝ) (h1 : x + y = 5) (h2 : 2 * x - y = 1) : x = 2 ∧ y = 3 := 
sorry

end solve_system_of_equations_l349_349934


namespace blue_bird_arrangements_l349_349938

theorem blue_bird_arrangements: 
  let boys := {b1, b2} in
  let girls := {g1, g2, g3} in
  ∃ arrangements : Finset (List (boys ∪ girls)), 
    arrangements.card = 12 ∧ 
    ∀ arrangement ∈ arrangements, 
       arrangement.head ∈ boys ∧ 
       arrangement.last ∈ boys ∧ 
       Multiset.inter (arrangement.tail.init.to_multiset) girls = girls.to_multiset :=
by
  sorry

end blue_bird_arrangements_l349_349938


namespace circle_center_radius_l349_349367

theorem circle_center_radius :
  ∃ (h k r : ℝ), (∀ x y : ℝ, (x - 2)^2 + (y + 1)^2 = 4 ↔ (x - h)^2 + (y - k)^2 = r^2) ∧ h = 2 ∧ k = -1 ∧ r = 2 :=
by
  sorry

end circle_center_radius_l349_349367


namespace q_compound_l349_349907

def q (x y : ℤ) : ℤ :=
  if x ≥ 1 ∧ y ≥ 1 then 2 * x + 3 * y
  else if x < 0 ∧ y < 0 then x + y^2
  else 4 * x - 2 * y

theorem q_compound : q (q 2 (-2)) (q 0 0) = 48 := 
by 
  sorry

end q_compound_l349_349907


namespace ratio_blue_to_gold_l349_349179

-- Define the number of brown stripes
def brown_stripes : Nat := 4

-- Given condition: There are three times as many gold stripes as brown stripes
def gold_stripes : Nat := 3 * brown_stripes

-- Given condition: There are 60 blue stripes
def blue_stripes : Nat := 60

-- The actual statement to prove
theorem ratio_blue_to_gold : blue_stripes / gold_stripes = 5 := by
  -- Proof would go here
  sorry

end ratio_blue_to_gold_l349_349179


namespace car_avg_speed_is_correct_l349_349620

def avg_speed (Speed1 Speed2 Speed3 Speed4 Distance1 Distance2 : ℝ) (Time3 Time4 : ℝ) : ℝ :=
  let Distance3 := Speed3 * Time3
  let Distance4 := Speed4 * Time4
  let TotalDistance := Distance1 + Distance2 + Distance3 + Distance4
  let TotalTime := (Distance1 / Speed1) + (Distance2 / Speed2) + Time3 + Time4
  TotalDistance / TotalTime

theorem car_avg_speed_is_correct :
  avg_speed 30 45 55 50 15 30 0.5 (1/3) = 44.585 := by
  sorry

end car_avg_speed_is_correct_l349_349620


namespace find_m_values_l349_349781

noncomputable def find_m (a b c m : ℝ) : Prop :=
  (a + b = 4) ∧
  (ab = m) ∧
  (b + c = 8) ∧
  (bc = 5m)

theorem find_m_values (a b c m : ℝ) (h : find_m a b c m) : m = 0 ∨ m = 3 :=
by
  sorry

end find_m_values_l349_349781


namespace max_moves_21x21_square_proof_max_moves_20x21_rectangle_proof_l349_349863

namespace MaxMoves

-- Definition for the 21x21 grid problem
def max_moves_21x21_square : Prop :=
  ∀ (M : Type) (is_square : M = (fin 21 × fin 21)), 
  maximum_moves_to_turn_on_all_lights M = 3

-- Definition for the 20x21 grid problem
def max_moves_20x21_rectangle : Prop :=
  ∀ (M : Type) (is_rectangle : M = (fin 20 × fin 21)), 
  maximum_moves_to_turn_on_all_lights M = 4

/-- Proof for the maximum moves required to turn on all lamps in a 21x21 grid --/
theorem max_moves_21x21_square_proof : max_moves_21x21_square :=
  sorry

/-- Proof for the maximum moves required to turn on all lamps in a 20x21 grid --/
theorem max_moves_20x21_rectangle_proof : max_moves_20x21_rectangle :=
  sorry

end MaxMoves

end max_moves_21x21_square_proof_max_moves_20x21_rectangle_proof_l349_349863


namespace difference_is_24_l349_349461

namespace BuffaloesAndDucks

def numLegs (B D : ℕ) : ℕ := 4 * B + 2 * D

def numHeads (B D : ℕ) : ℕ := B + D

def diffLegsAndHeads (B D : ℕ) : ℕ := numLegs B D - 2 * numHeads B D

theorem difference_is_24 (D : ℕ) : diffLegsAndHeads 12 D = 24 := by
  sorry

end BuffaloesAndDucks

end difference_is_24_l349_349461


namespace soda_difference_l349_349884

theorem soda_difference :
  let Julio_orange_bottles := 4
  let Julio_grape_bottles := 7
  let Mateo_orange_bottles := 1
  let Mateo_grape_bottles := 3
  let liters_per_bottle := 2
  let Julio_total_liters := Julio_orange_bottles * liters_per_bottle + Julio_grape_bottles * liters_per_bottle
  let Mateo_total_liters := Mateo_orange_bottles * liters_per_bottle + Mateo_grape_bottles * liters_per_bottle
  Julio_total_liters - Mateo_total_liters = 14 := by
    sorry

end soda_difference_l349_349884


namespace number_of_students_l349_349853

theorem number_of_students (n_last_year : ℕ) (increase_rate : ℝ) (this_year_students : ℕ) :
  n_last_year = 800 → increase_rate = 0.20 → this_year_students = n_last_year + increase_rate * (n_last_year : ℝ) → this_year_students = 960 := 
by
  intros h1 h2 h3
  subst h1
  subst h2
  simp at h3
  exact h3

end number_of_students_l349_349853


namespace binom_10_2_eq_45_l349_349701

-- Definitions used in the conditions
def binom (n k : ℕ) := n.choose k

-- The statement that needs to be proven
theorem binom_10_2_eq_45 : binom 10 2 = 45 :=
by
  sorry

end binom_10_2_eq_45_l349_349701


namespace jars_needed_l349_349045

def hives : ℕ := 5
def honey_per_hive : ℕ := 20
def jar_capacity : ℝ := 0.5
def friend_ratio : ℝ := 0.5

theorem jars_needed : (hives * honey_per_hive) / 2 / jar_capacity = 100 := 
by sorry

end jars_needed_l349_349045


namespace percentage_commute_l349_349440

variable (x : Real)
variable (h : 0.20 * 0.10 * x = 12)

theorem percentage_commute :
  0.10 * 0.20 * x = 12 :=
by
  sorry

end percentage_commute_l349_349440


namespace maximize_S_n_decreasing_arithmetic_sequence_l349_349961

theorem maximize_S_n_decreasing_arithmetic_sequence
  (a : ℕ → ℝ) (S : ℕ → ℝ)
  (d : ℝ)
  (h1 : ∀ n, a (n + 1) = a n + d)
  (h2 : d < 0)
  (h3 : ∀ n, S n = (n * (2 * a 1 + (n - 1) * d)) / 2)
  (h4 : S 5 = S 10) :
  S 7 = S 8 :=
sorry

end maximize_S_n_decreasing_arithmetic_sequence_l349_349961


namespace minimum_value_S_l349_349260

/-- A thin sheet of an equilateral triangle with side length 1 is cut into
    two parts along a line parallel to its base. One of the resulting parts is a trapezoid.
    Define S as the ratio of the perimeter of the trapezoid to its area and prove that
    the minimum value of S is \( \frac{4\sqrt{6}}{3} + 2\sqrt{3} \). -/
theorem minimum_value_S :
  ∃ S, let x := 1 in
          (0 < x ∧ x < 1) ∧
          S = (∃ t, t = 3 - x ∧ 2 < t ∧ t < 3 ∧
              S = (4 / sqrt 3) * (3 - x) / (1 - x^2)) ∧
          ∀ x, (0 < x ∧ x < 1) →
            (S = (4 / sqrt 3) * (3 - x) / (1 - x^2)) →
            S ≥ (4 * sqrt 6 / 3 + 2 * sqrt 3) :=
begin
  sorry
end

end minimum_value_S_l349_349260


namespace solve_fraction_x_l349_349992

theorem solve_fraction_x (a b c d : ℤ) (hb : b ≠ 0) (hdc : d + c ≠ 0) 
: (2 * a + (bc - 2 * a * d) / (d + c)) / (b - (bc - 2 * a * d) / (d + c)) = c / d := 
sorry

end solve_fraction_x_l349_349992


namespace base_seven_product_correct_l349_349183

def base_seven_to_base_ten (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | _ => let d := (n % 10); let r := (n / 10);
         d * 7 ^ (Nat.logBase 10 n) + base_seven_to_base_ten r

def base_ten_to_base_seven (n : ℕ) : ℕ :=
  if n < 7 then n else
  let r := n / 7;
  let d := n % 7;
  10 * base_ten_to_base_seven r + d

def base_seven_product (a b : ℕ) : ℕ :=
  let a_base10 := base_seven_to_base_ten a;
  let b_base10 := base_seven_to_base_ten b;
  let product_base10 := a_base10 * b_base10;
  base_ten_to_base_seven product_base10

-- Statement of the theorem
theorem base_seven_product_correct : base_seven_product 1324 23 = 31415 := sorry

end base_seven_product_correct_l349_349183


namespace num_possible_k_l349_349467

def lattice_points (x y : ℤ) : Prop := true

def line1 (x : ℤ) : ℤ := x - 3
def line2 (x k : ℤ) : ℤ := k * x + k

noncomputable def intersection_x (k : ℤ) : ℚ :=
  (↑k + 3) / (1 - ↑k)

noncomputable def intersection_y (k : ℤ) : ℚ :=
  (4 * ↑k) / (1 - ↑k)

theorem num_possible_k :
  { k : ℤ | lattice_points (⌊intersection_x k⌋) (⌊intersection_y k⌋) }.to_finset.card = 6 :=
sorry

end num_possible_k_l349_349467


namespace min_x_sqrt_floor_l349_349890

theorem min_x_sqrt_floor (n : ℕ) (h : n > 1):
  ∃ i, (i ≤ n ∧ x_seq n (i + 1) = intRoot n)
  where 
    x_seq : ℕ → ℕ → ℕ
    | n, 0 => n
    | n, i + 1 => (x_seq n i + y_seq n i) / 2
    
    y_seq : ℕ → ℕ → ℕ
    | n, i => n / x_seq n i

    intRoot : ℕ → ℕ
    | n => let k := int.sqrt n in if k * k = n then k else k - 1 :=
sorry

end min_x_sqrt_floor_l349_349890


namespace part_I_proof_part_II_proof_l349_349360

variables {a b : ℝ × ℝ} (k : ℝ)

-- Define the magnitudes of vectors a and b
def magnitude_a : ℝ := 3
def magnitude_b : ℝ := 4
def angle_ab : ℝ := real.pi / 3 -- 60 degrees in radians

-- Define the magnitudes condition
axiom magnitude_a_def : |a| = magnitude_a
axiom magnitude_b_def : |b| = magnitude_b

-- Define the dot product condition
axiom dot_product_ab : a.1 * b.1 + a.2 * b.2 = magnitude_a * magnitude_b * real.cos angle_ab

-- Questions (I) and (II)
theorem part_I_proof : |a + b| = real.sqrt 37 :=
sorry

theorem part_II_proof : (a + k • b = ⟂ (a - k • b)) → (k = 3 / 4 ∨ k = -3 / 4) :=
sorry

end part_I_proof_part_II_proof_l349_349360


namespace two_numbers_as_difference_of_two_primes_in_set_l349_349277

def is_prime (n : ℕ) : Prop := nat.prime n

def is_difference_of_two_primes (n : ℕ) : Prop :=
  ∃ (p : ℕ), is_prime p ∧ n = p - 2

def original_set := {x | ∃ (k : ℕ), x = 10 * k + 7}

def count_numbers_in_set_as_difference_of_two_primes : ℕ :=
  (set.univ.filter (λ x, x ∈ original_set ∧ is_difference_of_two_primes x)).card

theorem two_numbers_as_difference_of_two_primes_in_set : count_numbers_in_set_as_difference_of_two_primes = 2 :=
sorry

end two_numbers_as_difference_of_two_primes_in_set_l349_349277


namespace exists_positive_integer_with_2020_close_divisors_l349_349211

def is_close_divisor (n d : ℕ) : Prop := 
  (nat.sqrt n < d) ∧ (d < 2 * nat.sqrt n)

theorem exists_positive_integer_with_2020_close_divisors : 
  ∃ n : ℕ, (n > 0) ∧ (∃ S : finset ℕ, S.card = 2020 ∧ ∀ d ∈ S, is_close_divisor n d) := 
sorry

end exists_positive_integer_with_2020_close_divisors_l349_349211


namespace correct_answer_l349_349256

theorem correct_answer (x : ℝ) (h : sqrt x = 9) : x^2 = 6561 :=
sorry

end correct_answer_l349_349256


namespace age_of_first_man_replaced_l349_349538

theorem age_of_first_man_replaced (x : ℕ) (avg_before : ℝ) : avg_before * 15 + 30 = avg_before * 15 + 74 - (x + 23) → (37 * 2 - (x + 23) = 30) → x = 21 :=
sorry

end age_of_first_man_replaced_l349_349538


namespace binomial_alternating_sum_zero_l349_349300

theorem binomial_alternating_sum_zero :
  ∑ k in Finset.range 102, binomial 101 k * (-1)^k = 0 :=
sorry

end binomial_alternating_sum_zero_l349_349300


namespace monic_quadratic_has_root_l349_349741

theorem monic_quadratic_has_root {p : ℂ[X]} 
  (h_monic : p.monic) 
  (h_real_coeff : ∀ x : ℂ, p.coeff x ∈ ℝ) 
  (h_root : p.eval (-3 - complex.I * real.sqrt 7) = 0) :
  p = polynomial.X ^ 2 + 6 * polynomial.X + 16 :=
sorry

end monic_quadratic_has_root_l349_349741


namespace right_triangle_count_l349_349825

theorem right_triangle_count : ∃ n : ℕ, n = 16 ∧
  (∀ (a b : ℕ), b < 200 ∧ b > 0 ∧ a > 0 ∧ a * a = 6 * b + 9 → true) :=
begin
  -- We define the conditions here
  let n := 16,
  use n,
  split,
  { refl, },
  { intros a b hb hpositive_b hpositive_a ha,
    sorry, -- Skipping the proof steps after defining the conditions
  }
end

end right_triangle_count_l349_349825


namespace a_plus_b_l349_349465

noncomputable def vector_3d := (ℝ × ℝ × ℝ)
noncomputable def start_pt : vector_3d := (1, 1, 1)
noncomputable def end_pt : vector_3d := (4, 2, 2)
noncomputable def param_line (t : ℝ) : vector_3d := (1 + 3 * t, 1 + t, 1 + t)
noncomputable def sphere_radius : ℝ := 2

def intersects_sphere (t : ℝ) : Prop :=
  let pt := param_line t in
  (pt.1)^2 + (pt.2)^2 + (pt.3)^2 = (sphere_radius)^2

theorem a_plus_b :
  ∃ (a b : ℕ), b % (2 * 2) ≠ 0 ∧ 
  (∀ t : ℝ, intersects_sphere t → (1 + 3 * t, 1 + t, 1 + t)) → 
  a + b = 21 :=
by
  sorry

end a_plus_b_l349_349465


namespace function_exists_iff_l349_349080

open Function

noncomputable def satisfies_condition {f : ℕ → ℕ} (k n a : ℕ) : Prop :=
  iterate f k n = n + a

theorem function_exists_iff (f : ℕ → ℕ) (k a : ℕ) :
  (∃ f : ℕ → ℕ, ∀ n : ℕ, satisfies_condition f k n a) ↔ (a ∈ ℕ ∧ k ∣ a) :=
by
  split
  case mp =>
    intro h
    rcases h with ⟨f, hf⟩
    sorry -- proof that if f exists, then a is in ℕ and k divides a
  case mpr =>
    rintro ⟨a_nat, k_div_a⟩
    use (fun n => n + a / k)
    intro n
    sorry -- proof that such f satisfies the conditions

end function_exists_iff_l349_349080


namespace area_of_triangle_ABC_l349_349148

-- Define the setup for the problem
noncomputable def circle_radius : ℝ := 3
noncomputable def BD : ℝ := 4
noncomputable def ED : ℝ := 6

-- Define points A, B, D, E, and C
structure Point :=
(x : ℝ)
(y : ℝ)

def A : Point := ⟨0, 0⟩
def B : Point := ⟨2 * circle_radius, 0⟩
def D : Point := ⟨2 * circle_radius + BD, 0⟩
def E : Point := ⟨2 * circle_radius + BD, ED⟩

-- Define intersection point C
axiom C_between_AE (C : Point) : Prop
axiom AE_intersects_circle_at_C (C : Point) : Prop

-- Define the conditions in the problem
axiom perpendicular_ED_AD : E.y - D.y = 6

-- Define the theorem we want to prove
theorem area_of_triangle_ABC : 
  ∃ (C : Point), C_between_AE C ∧ AE_intersects_circle_at_C C →
  (1 / 2) * abs ((D.x - A.x) * (E.y - A.y) - (D.y - A.y) * (E.x - A.x)) = 36 * sqrt 30.24 / 25 := 
sorry

end area_of_triangle_ABC_l349_349148


namespace find_m_values_l349_349782

noncomputable def find_m (a b c m : ℝ) : Prop :=
  (a + b = 4) ∧
  (ab = m) ∧
  (b + c = 8) ∧
  (bc = 5m)

theorem find_m_values (a b c m : ℝ) (h : find_m a b c m) : m = 0 ∨ m = 3 :=
by
  sorry

end find_m_values_l349_349782


namespace reciprocal_eq_self_l349_349841

theorem reciprocal_eq_self {x : ℝ} (h : x ≠ 0) : (1 / x = x) → (x = 1 ∨ x = -1) :=
by
  intro h1
  sorry

end reciprocal_eq_self_l349_349841


namespace windows_needed_l349_349482

theorem windows_needed
  (cost_sheers : ℕ := 40)
  (cost_drapes : ℕ := 60)
  (total_cost : ℕ := 300) :
  let cost_per_window := cost_sheers + cost_drapes
  in ∃ x : ℕ, cost_per_window * x = total_cost ∧ x = 3 := 
by 
  let cost_per_window := cost_sheers + cost_drapes
  existsi (total_cost / cost_per_window)
  split
  . calc cost_per_window * (total_cost / cost_per_window) = total_cost : by ring
  . sorry

end windows_needed_l349_349482


namespace number_of_primes_between_30_and_50_l349_349421

/-- 
  Prove that there are exactly 5 prime numbers in the range from 30 to 50. 
  These primes are 31, 37, 41, 43, and 47.
-/
theorem number_of_primes_between_30_and_50 : 
  (Finset.filter Nat.Prime (Finset.range 51)).card - 
  (Finset.filter Nat.Prime (Finset.range 31)).card = 5 := 
by 
  sorry

end number_of_primes_between_30_and_50_l349_349421


namespace max_nine_multiple_l349_349806

theorem max_nine_multiple {a b c n : ℕ} (h1 : Prime a) (h2 : Prime b) (h3 : Prime c) (h4 : 3 < a) (h5 : 3 < b) (h6 : 3 < c) (h7 : 2 * a + 5 * b = c) : 9 ∣ (a + b + c) :=
sorry

end max_nine_multiple_l349_349806


namespace systematic_sampling_example_l349_349258

def sample_interval (total_students sample_size : ℕ) : ℕ := total_students / sample_size

def number_drawn (first_draw interval group_number : ℕ) : ℕ := first_draw + (group_number - 1) * interval

theorem systematic_sampling_example :
  let total_students := 600 in
  let sample_size := 20 in
  let first_draw := 2 in
  let group_number := 4 in
  number_drawn first_draw (sample_interval total_students sample_size) group_number = 92 :=
by
  sorry

end systematic_sampling_example_l349_349258


namespace a2_value_l349_349550

-- Definitions based on the identified conditions
def seq (n : ℕ) : ℝ := ∑ i in finset.range (n^2 + 1), if i ≥ n then 1 / (i : ℝ) else 0

-- The math proof statement based on the correct answer
theorem a2_value : seq 2 = 1 / 2 + 1 / 3 + 1 / 4 :=
by
  sorry

end a2_value_l349_349550


namespace P_eight_value_l349_349094

def quadratic_polynomial (P : ℤ → ℤ) : Prop :=
  ∃ a b c : ℤ, ∀ x : ℤ, P(x) = a * x^2 + b * x + c

theorem P_eight_value (P : ℤ → ℤ)
  (H : ∀ x : ℤ, P(P(x)) = x^4 - 2 * x^3 + 4 * x^2 - 3 * x + 4)
  (H_quad : quadratic_polynomial P) :
  P 8 = 58 :=
sorry

end P_eight_value_l349_349094


namespace trapezoid_rotated_quadrilateral_equal_diagonals_l349_349959

-- Definition of quadrilateral, midpoint rotation, and trapezoid
structure Quadrilateral (A B C D : Type) :=
  (AD_parallel_BC : Parallel AD BC)

structure RotatedQuadrilateral (A B C D M N P Q : Type) :=
  (M_rotated: Rotated90AroundMidpoint A D M)
  (N_rotated: Rotated90AroundMidpoint D B N)
  (P_rotated: Rotated90AroundMidpoint B C P)
  (Q_rotated: Rotated90AroundMidpoint C A Q)
  (equal_diagonals : Diagonal M P = Diagonal N Q)

-- Proving the theorem
theorem trapezoid_rotated_quadrilateral_equal_diagonals
  {A B C D M N P Q : Type}
  (quad : Quadrilateral A B C D)
  (rotated_quad : RotatedQuadrilateral A B C D M N P Q) :
  rotated_quad.equal_diagonals := by sorry

end trapezoid_rotated_quadrilateral_equal_diagonals_l349_349959


namespace directrix_of_given_parabola_l349_349312

-- Define the equation of the parabola
def parabola (x : ℝ) : ℝ := -1 / 4 * x ^ 2

-- Define what it means for a point (x, y) to be on the parabola
def on_parabola (x y : ℝ) : Prop := y = parabola x

-- Define what it means to be the directrix of the parabola
def is_directrix (d : ℝ) : Prop :=
  ∀ (x : ℝ), (parabola x - (d * √((x - 0) ^ 2 + (parabola x - (-d)) ^ 2))) = 0

theorem directrix_of_given_parabola : is_directrix 1 :=
sorry

end directrix_of_given_parabola_l349_349312


namespace find_f_e_l349_349358

-- Given conditions
def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def specific_f (f : ℝ → ℝ) := ∀ x, x < 0 → f x = Real.exp x

-- The proof statement
theorem find_f_e (f : ℝ → ℝ) (h_odd : odd_function f)
  (h_specific : specific_f f) : f Real.exp = -Real.exp (-Real.exp) :=
by
  sorry  -- Proof is skipped

end find_f_e_l349_349358


namespace quadratic_polynomial_P8_l349_349127

theorem quadratic_polynomial_P8 :
  ∃ (a b c : ℝ), 
  (∀ x : ℝ, P x = a * x^2 + b * x + c) ∧ 
  (∀ x : ℝ, P (P x) = x^4 - 2 * x^3 + 4 * x^2 - 3 * x + 4) ∧ 
  P 8 = 58 :=
begin
  sorry
end

end quadratic_polynomial_P8_l349_349127


namespace continuous_numbers_up_to_100_l349_349835

def is_continuous_number (n : ℕ) : Prop :=
  (n + (n + 1) + (n + 2) < 10)

def count_continuous_numbers (limit : ℕ) : ℕ :=
  (Finset.range limit).filter is_continuous_number |>.card

theorem continuous_numbers_up_to_100 : count_continuous_numbers 100 = 12 := 
  by
    sorry

end continuous_numbers_up_to_100_l349_349835


namespace sin_of_smallest_angle_l349_349366

theorem sin_of_smallest_angle
  (a b c : ℝ) (h_seq : a - b = 2 ∧ b - c = 2)
  (h_triangle : a > b ∧ b > c ∧ c > 0)
  (h_sin_largest : Real.sin (Real.acos (-1/2)) = Real.sqrt 3 / 2) : 
  Real.sin (Real.acos (13/14)) = 3 * Real.sqrt 3 / 14 :=
sorry

end sin_of_smallest_angle_l349_349366


namespace largest_subset_no_member_is_4_times_another_l349_349257

theorem largest_subset_no_member_is_4_times_another :
  ∃ (S : set ℕ), S ⊆ {n | 1 ≤ n ∧ n ≤ 50} ∧ (∀ a ∈ S, ∀ b ∈ S, a ≠ b → a ≠ 4 * b) ∧ S.card = 44 := 
sorry

end largest_subset_no_member_is_4_times_another_l349_349257


namespace binomial_10_2_equals_45_l349_349678

open Nat

theorem binomial_10_2_equals_45 : Nat.choose 10 2 = 45 := 
by
  sorry

end binomial_10_2_equals_45_l349_349678


namespace problem_d_l349_349898

noncomputable def f (n : ℕ) : ℝ :=
∑ i in (finset.range (3 * n + 1) \ finset.range (n + 1)), (1 : ℝ) / (i + 1)

theorem problem_d (k : ℕ) (h : 0 < k) :
  f (k + 1) - f k = (1 : ℝ) / (3 * k + 1) + (1 : ℝ) / (3 * k + 2) - (2 : ℝ) / (3 * k + 3) :=
sorry

end problem_d_l349_349898


namespace petya_exchange_correct_l349_349663

/-- Exchange options and their corresponding conversions -/
def exchange_20 : list ℕ := [15, 2, 2, 1]
def exchange_15 : list ℕ := [10, 2, 2, 1]
def exchange_10 : list ℕ := [3, 3, 2, 2]

/-- Petya's total amount in kopecks -/
def total_kopecks : ℕ := 125

/-- The result of Petya's exchange -/
def petya_coins : list ℕ := [15] ++ list.repeat 10 11

theorem petya_exchange_correct :
  (sum petya_coins = total_kopecks) ∧
  (∀ coin ∈ petya_coins, coin = 15 ∨ coin = 10) :=
by sorry

end petya_exchange_correct_l349_349663


namespace monotonicity_F_range_of_a_l349_349371

def f (a x : ℝ) : ℝ := (1 / 3) * x^3 - (1 / 2) * (a + 4) * x^2 + (3 * a + 5) * x - (2 * a + 2) * log x
def F (a x : ℝ) : ℝ := f a x - (1 / 3) * x^3 + (1 / 2) * (a + 5) * x^2 - (2 * a + 6) * x
def g (a x : ℝ) : ℝ := (f a x) + (2 * a + 2) / x

theorem monotonicity_F (a : ℝ) (h1 : a < -1) : 
  ((-3 < a ∧ ((0 < x ∧ x < -a - 1) ∨ (x > 2 → F' a x > 0)) ∧ ((-a - 1 < x ∧ x < 2) → F' a x < 0)) ∨ 
  (a = -3 ∧ (x > 0 → F' a x ≥ 0)) ∨ 
  (a < -3 ∧ ((0 < x ∧ x < 2) ∨ (x > -a - 1) → F' a x > 0) ∧ ((2 < x ∧ x < -a - 1) → F' a x < 0))) :=
sorry

theorem range_of_a (a : ℝ) (h2 : ∀ (x : ℝ), x > 0 → g a x ≥ (2 / 3) * log x + 3 * a + (14 / 3)) : 
  a ≤ (-4 - 4 * log 2) / 3 :=
sorry

end monotonicity_F_range_of_a_l349_349371


namespace evaluate_expression_l349_349669

theorem evaluate_expression :
  -1^2008 + 3*(-1)^2007 + 1^2008 - 2*(-1)^2009 = -5 := 
by
  sorry

end evaluate_expression_l349_349669


namespace add_base6_l349_349263

theorem add_base6 : ∀ (x y z : ℕ), 
  nat.of_digits 6 [1, 5, 2] = x → 
  nat.of_digits 6 [3, 5] = y → 
  nat.of_digits 6 [2, 1, 3] = z → 
  x + y = z :=
by
  intros x y z h1 h2 h3
  rw [h1, h2, h3]
  sorry

end add_base6_l349_349263


namespace primes_between_30_and_50_l349_349390

theorem primes_between_30_and_50 : (Finset.card (Finset.filter Nat.Prime (Finset.Ico 30 51))) = 5 :=
by
  sorry

end primes_between_30_and_50_l349_349390


namespace sum_of_values_l349_349836

variable {R : ℝ}

noncomputable def equation (N : ℝ) := N - 5/N = R

theorem sum_of_values (h : equation N) : (Σ N, equation N N) = R := 
sorry

end sum_of_values_l349_349836


namespace prove_sin_cos_equations_l349_349757

variables (α : ℝ)
noncomputable def sin_cos_sum_eq (α : ℝ) : Prop := sin α + cos α = (sqrt 3) / 3

theorem prove_sin_cos_equations (h : sin_cos_sum_eq α) :
  (sin α) ^ 4 + (cos α) ^ 4 = 7 / 9 ∧
  (tan α) / (1 + (tan α) ^ 2) = -1 / 3 :=
by 
  sorry

end prove_sin_cos_equations_l349_349757


namespace first_discount_percentage_l349_349547

theorem first_discount_percentage (x : ℝ) (h : 450 * (1 - x / 100) * 0.85 = 306) : x = 20 :=
sorry

end first_discount_percentage_l349_349547


namespace minimum_value_of_a_l349_349377

theorem minimum_value_of_a :
  ∀ (x : ℝ), (2 * x + 2 / (x - 1) ≥ 7) ↔ (3 ≤ x) :=
sorry

end minimum_value_of_a_l349_349377


namespace smallest_multiple_17_7_more_53_l349_349192

theorem smallest_multiple_17_7_more_53 : 
  ∃ a : ℕ, (17 * a ≡ 7 [MOD 53]) ∧ 17 * a = 187 := 
by
  have h : 17* 11 = 187 := by norm_num
  use 11
  constructor
  · norm_num
  · exact h
  sorry

end smallest_multiple_17_7_more_53_l349_349192


namespace parallel_implies_ratio_perpendicular_implies_difference_l349_349339

open Real

variable (x : ℝ)

def a := (1 : ℝ, cos x)
def b := (1 / 3 : ℝ, sin x)

def parallel := (1 / 3) * cos x = sin x
def perpendicular := (1 : ℝ) * (1 / 3) + cos x * sin x = 0

theorem parallel_implies_ratio (h : parallel x) (hx : 0 < x ∧ x < π) : 
  (sin x + cos x) / (sin x - cos x) = -2 :=
sorry

theorem perpendicular_implies_difference (h : perpendicular x) (hx : 0 < x ∧ x < π) : 
  (sin x - cos x) = (sqrt 15) / 3 :=
sorry

end parallel_implies_ratio_perpendicular_implies_difference_l349_349339


namespace quadratic_P_value_l349_349122

noncomputable def P (x : ℝ) : ℝ :=
  x^2 - x + 2

theorem quadratic_P_value :
  P (P 8) = 8^4 - 2 * 8^3 + 4 * 8^2 - 3 * 8 + 4 ∧ P 8 = 58 := 
by
  have h1 : P(P(8)) = 8^4 - 2 * 8^3 + 4 * 8^2 - 3 * 8 + 4 := sorry
  have h2 : P(8) = 58 := sorry
  exact ⟨h1, h2⟩  

end quadratic_P_value_l349_349122


namespace find_a_l349_349939

-- Define Fibonacci sequence
def fib : ℕ → ℕ
| 0       := 0
| 1       := 1
| (n + 2) := fib n + fib (n + 1)

-- Define conditions in Lean 4

def is_arith_seq (a b c : ℕ) :=
  b - a = c - b

theorem find_a:
  ∃ a b c : ℕ, fib a < fib b ∧ fib b < fib c ∧ is_arith_seq (fib a) (fib b) (fib c) ∧ a + b + c = 3000 ∧ a = 998 :=
by
  sorry

end find_a_l349_349939


namespace f_periodic_l349_349505

noncomputable def f : ℝ → ℝ := sorry

variable (a : ℝ) (h_a : 0 < a)
variable (h_cond : ∀ x : ℝ, f (x + a) = 1 / 2 + sqrt (f x - (f x)^2))

theorem f_periodic : ∀ x : ℝ, f (x + 2 * a) = f x := sorry

end f_periodic_l349_349505


namespace number_of_solutions_l349_349389

-- Define the conditions
def condition1 (a b : ℤ) : Prop := a ^ 2 + b ^ 2 < 25
def condition2 (a b : ℤ) : Prop := a ^ 2 + b ^ 2 < 10 * a
def condition3 (a b : ℤ) : Prop := a ^ 2 + b ^ 2 < 10 * b

-- Main theorem statement
theorem number_of_solutions : 
  {p : ℤ × ℤ // condition1 p.1 p.2 ∧ condition2 p.1 p.2 ∧ condition3 p.1 p.2}.toFinset.card = 8 :=
by
  sorry

end number_of_solutions_l349_349389


namespace hibiscus_plants_solution_l349_349073

theorem hibiscus_plants_solution :
  ∃ (n : ℕ), 
      let flowers_1 := 2 in
      let flowers_2 := 2 * flowers_1 in
      let flowers_3 := 4 * flowers_2 in
      flowers_1 + flowers_2 + flowers_3 = 22 ∧
      n = 3 :=
begin
  sorry
end

end hibiscus_plants_solution_l349_349073


namespace find_ellipse_equation_find_k_value_l349_349768

noncomputable def ellipse_params : Prop :=
  ∃ a b : ℝ, a > b ∧ b > 0 ∧
  (∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1 →
    (real.sqrt (a^2 - b^2) / a = real.sqrt 3 / 3)) ∧
  (∀ x y : ℝ, (x^2 + y^2 = b^2 → (abs 2 / real.sqrt (1^2 + (-1)^2) = b)) ∧
  (a^2 = 3 * (a^2 - b^2)))

theorem find_ellipse_equation (a b : ℝ) (h : ellipse_params) :
  ∀ x y : ℝ, x^2 / 3 + y^2 / 2 = 1 :=
sorry

noncomputable def point_conditions (k : ℝ) (x₀ y₀ : ℝ) : Prop :=
  k > 1 ∧ y₀ = k * x₀ ∧
  (λ x y : ℝ, y = k * x ∧ 2 * x^2 + 3 * (k * x)^2 = 6) ∧
  (x₀ = real.sqrt 6 / real.sqrt (2 + 3 * k^2) ∧ 
   y₀ = (real.sqrt 6 * k) / real.sqrt (2 + 3 * k^2)) ∧
  ((x₀ * real.sqrt 2 / real.sqrt (2 + 3 * k^2)) + (y₀ * real.sqrt 2 * k / real.sqrt (2 + 3 * k^2)) = real.sqrt 6) →
  k = real.sqrt 2

theorem find_k_value (k : ℝ) (x₀ y₀ : ℝ) (h : point_conditions k x₀ y₀) :
  k = real.sqrt 2 :=
sorry

end find_ellipse_equation_find_k_value_l349_349768


namespace sequence_not_arithmetic_nor_geometric_l349_349765

def sum_first_n_terms (a q : ℝ) (n : ℕ) : ℝ :=
  a * q ^ n

def sequence_term (a q : ℝ) (n : ℕ) : ℝ :=
  sum_first_n_terms a q n - sum_first_n_terms a q (n - 1)

theorem sequence_not_arithmetic_nor_geometric (a q : ℝ) (h₀ : a ≠ 0) (h₁ : q ≠ 1) (h₂ : q ≠ 0) :
  ∀ n, ¬ (∃ d : ℝ, ∀ n, sequence_term a q n = sequence_term a q (n-1) + d) ∧
       ¬ (∃ r : ℝ, ∀ n, sequence_term a q (n+1) = r * sequence_term a q n) :=
by
  sorry

end sequence_not_arithmetic_nor_geometric_l349_349765


namespace perpendicular_diagonals_iff_cyclic_midpoints_l349_349082

noncomputable def quadrilateral_is_cyclic (A B C D : Point)
  (E F G H : Point)
  (AB BC CD DA : Line)
  (AC BD EH EF FG GH EG FH : Line)
  (midpoint_E : is_midpoint E A B)
  (midpoint_F : is_midpoint F B C)
  (midpoint_G : is_midpoint G C D)
  (midpoint_H : is_midpoint H D A)
  (intersection_EH : intersects EH AC)
  (intersection_EF : intersects EF BD)
  (intersection_FG : intersects FG AC)
  (intersection_GH : intersects GH BD)
  (intersection_AC_BD : intersects AC BD)
  (intersection_EG_FH : intersects EG FH) 
  : Prop :=
  (perpendicular AC BD) ↔ (cyclic_quadrilateral E F G H)

-- Variables Definition
variable (A B C D E F G H P I J K L O : Point)
variable (AB BC CD DA AC BD EH EF FG GH EG FH : Line)

-- Hypothesis Definition
variable (h1 : is_midpoint E A B)
variable (h2 : is_midpoint F B C)
variable (h3 : is_midpoint G C D)
variable (h4 : is_midpoint H D A)
variable (h5 : intersects EH AC)
variable (h6 : intersects EF BD)
variable (h7 : intersects FG AC)
variable (h8 : intersects GH BD)
variable (h9 : intersects AC BD)
variable (h10 : intersects EG FH)

-- Problem Statement
theorem perpendicular_diagonals_iff_cyclic_midpoints :
  (quadrilateral_is_cyclic A B C D E F G H AB BC CD DA AC BD EH EF FG GH EG FH h1 h2 h3 h4 h5 h6 h7 h8 h9 h10) :=
sorry

end perpendicular_diagonals_iff_cyclic_midpoints_l349_349082


namespace tetrahedron_projection_l349_349162

theorem tetrahedron_projection:
  (exists (tetrahedron : Type) (face1 face2 : face tetrahedron) (proj1 : face1 → ℝ) (proj2 : face2 → ℝ)
    (area_trap : ∀ P Q R S, is_trapezoid P Q R S → area P Q R S = 1),
    ∀ {P Q R S : face2},
    (is_square P Q R S → area P Q R S ≠ 1) ∧ 
    (is_square P Q R S → 1 / 2019 = area P Q R S)) := sorry

end tetrahedron_projection_l349_349162


namespace positive_difference_of_expressions_l349_349578

theorem positive_difference_of_expressions :
  let a := 8
  let expr1 := (a^2 - a^2) / a
  let expr2 := (a^2 * a^2) / a
  expr1 = 0 → expr2 = 512 → 512 - 0 = 512 := 
by
  introv h_expr1 h_expr2
  rw [h_expr1, h_expr2]
  norm_num
  exact rfl

end positive_difference_of_expressions_l349_349578


namespace balance_difference_is_89_l349_349047

noncomputable def jasmine_balance (P: ℕ) (r: ℚ) (n: ℕ) : ℚ :=
  P * (1 + r) ^ n

noncomputable def lucas_balance (P: ℕ) (r: ℚ) (n: ℕ) : ℚ :=
  P * (1 + n * r)

theorem balance_difference_is_89 :
  let P := 10000 in
  let r_J := 0.04 in
  let r_L := 0.06 in
  let n := 20 in
  let A_J := jasmine_balance P r_J n in
  let A_L := lucas_balance P r_L n in
  |A_L - A_J| = 89 :=
by
  sorry

end balance_difference_is_89_l349_349047


namespace quotient_independence_l349_349929

-- Definitions
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def partial_sum (a : ℕ → ℝ) (s : ℕ → ℝ) : Prop :=
  ∀ k : ℕ, s k = ∑ i in finset.range (k + 1), a i

-- Theorem statement
theorem quotient_independence (a : ℕ → ℝ) (s : ℕ → ℝ) (k r : ℕ) (hk : k > 0) (hr : r < k) :
  arithmetic_sequence a →
  partial_sum a s →
  (s (k + r) - s (k - r)) / r = a k + a (k + 1) :=
by sorry

end quotient_independence_l349_349929


namespace angle_difference_l349_349506

theorem angle_difference (A B C P Q : Point) (α : ℝ) 
  (hABC : ∃ (circumcircle : Circle), A ∈ circumcircle ∧ B ∈ circumcircle ∧ C ∈ circumcircle)
  (hP : P ∉ {A, B, C} ∧ ∃ circumcircle, P ∈ circumcircle)
  (hQ : Q ∉ {A, B, C} ∧ ∃ circumcircle, Q ∈ circumcircle)
  (hPA_sq_eq : dist P A ^ 2 = dist P B * dist P C)
  (hQA_sq_eq : dist Q A ^ 2 = dist Q B * dist Q C)
  (hP_arc_AB : P on arc from A to B)
  (hQ_arc_AC : Q on arc from A to C)
  (h_angle_diff : angle B - angle C = α) :
  angle P A B - angle Q A C = (π - α) / 2 := 
sorry

end angle_difference_l349_349506


namespace triangle_inequality_sides_l349_349488

theorem triangle_inequality_sides {a b c : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (triangle_ineq1 : a + b > c) (triangle_ineq2 : b + c > a) (triangle_ineq3 : c + a > b) : 
  |(a / b) + (b / c) + (c / a) - (b / a) - (c / b) - (a / c)| < 1 :=
  sorry

end triangle_inequality_sides_l349_349488


namespace binom_10_2_eq_45_l349_349710

theorem binom_10_2_eq_45 :
  binom 10 2 = 45 := by
  sorry

end binom_10_2_eq_45_l349_349710


namespace find_number_of_books_l349_349917

-- Define the constants and equation based on the conditions
def price_paid_per_book : ℕ := 11
def price_sold_per_book : ℕ := 25
def total_difference : ℕ := 210

def books_equation (x : ℕ) : Prop :=
  (price_sold_per_book * x) - (price_paid_per_book * x) = total_difference

-- The theorem statement that needs to be proved
theorem find_number_of_books (x : ℕ) (h : books_equation x) : 
  x = 15 :=
sorry

end find_number_of_books_l349_349917


namespace length_of_arc_within_triangle_l349_349142

variable (a α β : ℝ)

theorem length_of_arc_within_triangle :
  ∃ (H : ℝ), (H = a * (Real.sin α * Real.sin β) / Real.sin (α + β)) →
  (a * (Real.pi - α - β) * Real.sin α * Real.sin β) / Real.sin (α + β) = 
  (λ R, (Real.pi - α - β) * R) H :=
begin
  sorry
end

end length_of_arc_within_triangle_l349_349142


namespace other_root_l349_349838

theorem other_root (m n : ℝ) (h : (3 : ℂ) + (1 : ℂ) * Complex.I ∈ {x : ℂ | x^2 + ↑m * x + ↑n = 0}) : 
    (3 : ℂ) - (1 : ℂ) * Complex.I ∈ {x : ℂ | x^2 + ↑m * x + ↑n = 0} :=
sorry

end other_root_l349_349838


namespace primes_between_30_and_50_l349_349391

theorem primes_between_30_and_50 : (Finset.card (Finset.filter Nat.Prime (Finset.Ico 30 51))) = 5 :=
by
  sorry

end primes_between_30_and_50_l349_349391


namespace range_of_a_l349_349447

theorem range_of_a {a : ℝ} :
  (∀ x : ℝ, (a-2)*x^2 + 2*(a-2)*x - 4 < 0) ↔ (-2 < a ∧ a ≤ 2) :=
sorry

end range_of_a_l349_349447


namespace fraction_unchanged_when_increased_by_ten_l349_349837

variable {x y : ℝ}

theorem fraction_unchanged_when_increased_by_ten (x y : ℝ) :
  (5 * (10 * x)) / (10 * x + 10 * y) = 5 * x / (x + y) :=
by
  sorry

end fraction_unchanged_when_increased_by_ten_l349_349837


namespace increasing_interval_of_y_l349_349157

noncomputable def y (x : ℝ) : ℝ := 4 * x^2 + 1 / x

def y' (x : ℝ) : ℝ := 8 * x - 1 / x^2

theorem increasing_interval_of_y :
  ∀ x : ℝ, (y' x > 0) ↔ (x > 1/2) := by
  sorry -- Proof is omitted.

end increasing_interval_of_y_l349_349157


namespace binomial_10_2_equals_45_l349_349677

open Nat

theorem binomial_10_2_equals_45 : Nat.choose 10 2 = 45 := 
by
  sorry

end binomial_10_2_equals_45_l349_349677


namespace find_product_of_variables_l349_349204

variables (a b c d : ℚ)

def system_of_equations (a b c d : ℚ) :=
  3 * a + 4 * b + 6 * c + 9 * d = 45 ∧
  4 * (d + c) = b + 1 ∧
  4 * b + 2 * c = a ∧
  2 * c - 2 = d

theorem find_product_of_variables :
  system_of_equations a b c d → a * b * c * d = 162 / 185 :=
by sorry

end find_product_of_variables_l349_349204


namespace decrease_in_area_of_equilateral_triangle_l349_349266

noncomputable def equilateral_triangle_area (s : ℝ) : ℝ := (s^2 * Real.sqrt 3) / 4

theorem decrease_in_area_of_equilateral_triangle :
  (equilateral_triangle_area 20 - equilateral_triangle_area 14) = 51 * Real.sqrt 3 := by
  sorry

end decrease_in_area_of_equilateral_triangle_l349_349266


namespace count_primes_between_30_and_50_l349_349410

-- Define the range of numbers from 30 to 50
def range_30_to_50 := Set.of_list (List.range' 30 (51 - 30))

-- Define a predicate to check if a number is prime
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

-- Extract all prime numbers in the specified range
def primes_between_30_and_50 : List ℕ :=
  List.filter is_prime (Set.toList range_30_to_50)

theorem count_primes_between_30_and_50 : primes_between_30_and_50.length = 5 :=
by
  -- The proof goes here
  sorry

end count_primes_between_30_and_50_l349_349410


namespace max_sides_three_obtuse_l349_349301

theorem max_sides_three_obtuse (n : ℕ) (convex : Prop) (obtuse_angles : ℕ) :
  (convex = true ∧ obtuse_angles = 3) → n ≤ 6 :=
by
  sorry

end max_sides_three_obtuse_l349_349301


namespace parabola_vertex_and_point_l349_349948

theorem parabola_vertex_and_point (a b c : ℝ) (h_vertex : (1, -2) = (1, a * 1^2 + b * 1 + c))
  (h_point : (3, 7) = (3, a * 3^2 + b * 3 + c)) : a = 3 := 
by {
  sorry
}

end parabola_vertex_and_point_l349_349948


namespace mag_a_minus_b_l349_349340

variables (a b : EuclideanSpace) -- Define the vectors a and b

-- Define the magnitudes using the given conditions
def mag_a : ℝ := ‖a‖ = 2
def mag_b : ℝ := ‖b‖ = 1
def mag_a_plus_b : ℝ := ‖a + b‖ = √3

-- Theorem statement
theorem mag_a_minus_b (ha : mag_a) (hb : mag_b) (hab : mag_a_plus_b) : ‖a - b‖ = √7 := by
  sorry

end mag_a_minus_b_l349_349340


namespace archer_prob_6_or_less_l349_349785

noncomputable def prob_event_D (P_A P_B P_C : ℝ) : ℝ :=
  1 - (P_A + P_B + P_C)

theorem archer_prob_6_or_less :
  let P_A := 0.5
  let P_B := 0.2
  let P_C := 0.1
  prob_event_D P_A P_B P_C = 0.2 :=
by
  sorry

end archer_prob_6_or_less_l349_349785


namespace triangle_area_proof_l349_349808

noncomputable def area_of_triangle (a b c : ℝ) (A B C : ℝ) : ℝ := 
  1 / 2 * a * c * Real.sin B

theorem triangle_area_proof (a b c : ℝ) (A B C : ℝ) (h1 : b = 3) 
  (h2 : Real.cos B = 1 / 4) 
  (h3 : Real.sin C = 2 * Real.sin A) 
  (h4 : c = 2 * a) 
  (h5 : 9 = 5 * a ^ 2 - 4 * a ^ 2 * Real.cos B): 
  area_of_triangle a b c A B C = 9 * Real.sqrt 15 / 16 :=
by 
  sorry

end triangle_area_proof_l349_349808


namespace a5_a6_value_l349_349764

def S (n : ℕ) : ℕ := n^3

theorem a5_a6_value : S 6 - S 4 = 152 :=
by
  sorry

end a5_a6_value_l349_349764


namespace length_of_hypotenuse_l349_349711

noncomputable def length_hypotenuse (a b : ℝ) (α : ℝ) (sin cos : ℝ) : ℝ :=
  let sin_sq := sin ^ 2
  let cos_sq := cos ^ 2
  let trisection_eqn1 := (2/3 * a) ^ 2 + (1/3 * b) ^ 2 = sin_sq
  let trisection_eqn2 := (1/3 * a) ^ 2 + (2/3 * b) ^ 2 = cos_sq
  if trisection_eqn1 ∧ trisection_eqn2 ∧ 0 < α ∧ α < real.pi / 2 then
    real.sqrt (9 / 5)
  else
    0

theorem length_of_hypotenuse 
  (a b : ℝ) (α : ℝ) (sin cos : ℝ)
  (h1 : (2/3 * a) ^ 2 + (1/3 * b) ^ 2 = sin ^ 2)
  (h2 : (1/3 * a) ^ 2 + (2/3 * b) ^ 2 = cos ^ 2)
  (h3 : 0 < α)
  (h4 : α < real.pi / 2) :
  length_hypotenuse a b α sin cos = 3 / real.sqrt 5 := 
sorry

end length_of_hypotenuse_l349_349711


namespace birth_year_1957_l349_349474

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem birth_year_1957 (y x : ℕ) (h : y = 2023) (h1 : sum_of_digits x = y - x) : x = 1957 :=
by
  sorry

end birth_year_1957_l349_349474


namespace find_m_l349_349779

variable {a b c m : ℝ}

theorem find_m (h1 : a + b = 4)
               (h2 : a * b = m)
               (h3 : b + c = 8)
               (h4 : b * c = 5 * m) : m = 0 ∨ m = 3 :=
by {
  sorry
}

end find_m_l349_349779


namespace rocky_miles_total_l349_349855

-- Defining the conditions
def m1 : ℕ := 4
def m2 : ℕ := 2 * m1
def m3 : ℕ := 3 * m2

-- The statement to be proven
theorem rocky_miles_total : m1 + m2 + m3 = 36 := by
  sorry

end rocky_miles_total_l349_349855


namespace P_eight_value_l349_349098

def quadratic_polynomial (P : ℤ → ℤ) : Prop :=
  ∃ a b c : ℤ, ∀ x : ℤ, P(x) = a * x^2 + b * x + c

theorem P_eight_value (P : ℤ → ℤ)
  (H : ∀ x : ℤ, P(P(x)) = x^4 - 2 * x^3 + 4 * x^2 - 3 * x + 4)
  (H_quad : quadratic_polynomial P) :
  P 8 = 58 :=
sorry

end P_eight_value_l349_349098


namespace factorize_polynomial_l349_349200

noncomputable def polynomial_factorization : Prop :=
  ∀ x : ℤ, (x^12 + x^9 + 1) = (x^4 + x^3 + x^2 + x + 1) * (x^8 - x^7 + x^6 - x^5 + x^3 - x^2 + x - 1)

theorem factorize_polynomial : polynomial_factorization :=
by
  sorry

end factorize_polynomial_l349_349200


namespace circle_geometry_l349_349865

theorem circle_geometry (BC BD DA x : ℝ) (hBC : BC = Real.sqrt 901)
  (hBD : BD = 1) (hDA : DA = 16)
  (h_angle_BDC : ∠ B D C = π / 2)
  : x = 26 :=
sorry

end circle_geometry_l349_349865


namespace sin_condition_for_angle_A_l349_349473

theorem sin_condition_for_angle_A (ABC : Triangle) (A : ℝ) (h1 : sin (2 * A) = (sqrt 3) / 2) : 
  (∃ A : ℝ, A = π/6 ∨ A = π/3) ∧ (A = π/6 → sin (2 * A) = (sqrt 3) / 2) ∧ ((sin (2 * A) = (sqrt 3) / 2) → A = π/6) :=
begin
  sorry
end

end sin_condition_for_angle_A_l349_349473


namespace alan_needs_more_wings_l349_349481

theorem alan_needs_more_wings 
  (kevin_wings : ℕ) (kevin_time : ℕ) (alan_rate : ℕ) (target_wings : ℕ) : 
  kevin_wings = 64 → kevin_time = 8 → alan_rate = 5 → target_wings = 3 → 
  (kevin_wings / kevin_time < alan_rate + target_wings) :=
by
  intros kevin_eq time_eq rate_eq target_eq
  sorry

end alan_needs_more_wings_l349_349481


namespace product_of_invertible_function_labels_l349_349153

def function2 (x : ℝ) : ℝ := x^2 - 2 * x
def function3_domain := {-4, -3, -2, -1, 0, 1, 2, 3}
def function3 (x : ℝ) : ℝ := Real.sin x
def function4 (x : ℝ) : ℝ := -Real.atan x
def function5 (x : ℝ) : ℝ := 4 / x

theorem product_of_invertible_function_labels
: ∀ f2 f3_domain f3 f4 f5,
  (f2 = function2) →
  (f3_domain = function3_domain) →
  (f3 = function3) →
  (f4 = function4) →
  (f5 = function5) →
  (∃ invertible_labels : Set ℕ, 
    (¬ (∃ f_inv : ℝ → ℝ, ∀ y, (∃ x, f2 x = y) → (f_inv y = x ∧ ∀ x₁ x₂, f2 x₁ = f2 x₂ → x₁ = x₂))
    ∧ (∃ f_inv : ℝ → ℝ, ∀ y ∈ f3_domain, (∃ x, f3 x = y) → (f_inv y = x ∧ ∀ x₁ x₂, x₁ ∈ f3_domain → x₂ ∈ f3_domain → f3 x₁ = f3 x₂ → x₁ = x₂))
    ∧ (∃ f_inv : ℝ → ℝ, ∀ y, (∃ x, f4 x = y) → (f_inv y = x ∧ ∀ x₁ x₂, f4 x₁ = f4 x₂ → x₁ = x₂))
    ∧ (∃ f_inv : ℝ → ℝ, ∀ y, (∃ x, f5 x = y) → (f_inv y = x ∧ ∀ x₁ x₂, f5 x₁ = f5 x₂ → x₁ = x₂))
    → ∀ p ∈ invertible_labels, p ∈ {3, 4, 5} ∧ ∀ q ∈ {3, 4, 5}, q ∈ invertible_labels)
  → (∏ p in invertible_labels, p = 60))
:= by
  intros
  sorry

end product_of_invertible_function_labels_l349_349153


namespace P_P_eq_P_eight_equals_58_l349_349088

open Polynomial

noncomputable def P(x : ℚ) : ℚ := x^2 - x + 2

theorem P_P_eq :
  (P ∘ P)(x) = x^4 - 2 * x^3 + 4 * x^2 - 3 * x + 4 := sorry

theorem P_eight_equals_58 :
  P 8 = 58 := sorry

end P_P_eq_P_eight_equals_58_l349_349088


namespace circle_line_intersection_condition_l349_349724

theorem circle_line_intersection_condition (k : ℝ) : 
  has_common_point : Prop :=
  let d := 3 / Real.sqrt (1 + k^2) in
  d ≤ 1 ↔ (-Real.sqrt 8 ≤ k ∧ k ≤ Real.sqrt 8) :=
sorry

end circle_line_intersection_condition_l349_349724


namespace question1_question2_l349_349528

-- Problem 1
theorem question1 (a : ℝ) (h : a^(1/2) + a^(-1/2) = 3) : a + a⁻¹ = 7 := by
  sorry

-- Problem 2
theorem question2 : (log 5)^2 + log 2 * log 50 = 1 := by
  sorry

end question1_question2_l349_349528


namespace area_of_triangle_DEF_is_42_l349_349032

-- Given conditions
def DE : ℝ := 12
def height_from_D_to_EF : ℝ := 7

-- Definition of the area of triangle DEF
def area_triangle_DEF (DE : ℝ) (height : ℝ) : ℝ :=
  1/2 * DE * height

-- The proposition to prove
theorem area_of_triangle_DEF_is_42 :
  area_triangle_DEF DE height_from_D_to_EF = 42 :=
by
  sorry

end area_of_triangle_DEF_is_42_l349_349032


namespace collinear_probability_l349_349869

-- Define the rectangular array
def rows : ℕ := 4
def cols : ℕ := 5
def total_dots : ℕ := rows * cols
def chosen_dots : ℕ := 4

-- Define the collinear sets
def horizontal_lines : ℕ := rows
def vertical_lines : ℕ := cols
def collinear_sets : ℕ := horizontal_lines + vertical_lines

-- Define the total combinations of choosing 4 dots out of 20
def total_combinations : ℕ := Nat.choose total_dots chosen_dots

-- Define the probability
def probability : ℚ := collinear_sets / total_combinations

theorem collinear_probability : probability = 9 / 4845 := by
  sorry

end collinear_probability_l349_349869


namespace eval_expr_l349_349299

theorem eval_expr : (2:ℝ) ^ (-3) * (7:ℝ) ^ 0 / (2:ℝ) ^ (-5) = 4 := by
  sorry

end eval_expr_l349_349299


namespace number_of_four_digit_numbers_satisfying_conditions_l349_349388

theorem number_of_four_digit_numbers_satisfying_conditions :
  let valid_numbers := {N : ℕ | 4000 ≤ N ∧ N < 6000 ∧ 5 ∣ N ∧ 
                              ∃ a b c d : ℕ, 
                                N = 1000 * a + 100 * b + 10 * c + d ∧ 
                                4 ≤ b ∧ b < c ∧ c ≤ 7 ∧ 
                                (a = 4 ∨ a = 5) ∧ 
                                (d = 0 ∨ d = 5) ∧ 
                                a ≠ d} in
  valid_numbers.card = 12 :=
by
  sorry

end number_of_four_digit_numbers_satisfying_conditions_l349_349388


namespace product_sequence_l349_349013

theorem product_sequence :
  (∀ n : ℕ, 0 < n → a n = (n^2 + n - 2 - Real.sqrt 2) / (n^2 - 2)) →
  ∏ n in Finset.range (2016 + 1) \ {0}, a n = 2016 * Real.sqrt 2 - 2015 := 
sorry

noncomputable def a (n : ℕ) : ℝ :=
  if h: n > 0 then (n^2 + n - 2 - Real.sqrt 2) / (n^2 - 2) else 0

end product_sequence_l349_349013
