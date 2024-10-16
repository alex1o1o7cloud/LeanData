import Mathlib

namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l2265_226532

/-- Given a triangle ABC with side lengths a, b, c and angles A, B, C (in radians),
    prove that if b = 2, C = π/3, and c = √3, then B = π/2 -/
theorem triangle_angle_calculation (a b c : ℝ) (A B C : ℝ) :
  b = 2 → C = π/3 → c = Real.sqrt 3 → B = π/2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l2265_226532


namespace NUMINAMATH_CALUDE_limit_sequence_equals_e_to_four_thirds_l2265_226594

open Real

theorem limit_sequence_equals_e_to_four_thirds :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N,
    |((3 * n + 1) / (3 * n - 1)) ^ (2 * n + 3) - Real.exp (4 / 3)| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_limit_sequence_equals_e_to_four_thirds_l2265_226594


namespace NUMINAMATH_CALUDE_max_revenue_price_l2265_226571

def revenue (p : ℝ) : ℝ := p * (150 - 4 * p)

theorem max_revenue_price :
  ∃ (p : ℝ), p ≤ 30 ∧ 
  ∀ (q : ℝ), q ≤ 30 → revenue p ≥ revenue q ∧
  p = 18.75 :=
sorry

end NUMINAMATH_CALUDE_max_revenue_price_l2265_226571


namespace NUMINAMATH_CALUDE_flag_arrangement_theorem_l2265_226545

/-- The number of distinguishable arrangements of flags on two flagpoles -/
def N : ℕ := 858

/-- The number of blue flags -/
def blue_flags : ℕ := 12

/-- The number of green flags -/
def green_flags : ℕ := 11

/-- The total number of flags -/
def total_flags : ℕ := blue_flags + green_flags

/-- The number of flagpoles -/
def flagpoles : ℕ := 2

theorem flag_arrangement_theorem :
  (∀ (arrangement : Fin total_flags → Fin flagpoles),
    (∀ pole : Fin flagpoles, ∃ flag : Fin total_flags, arrangement flag = pole) ∧
    (∀ i j : Fin total_flags, i.val + 1 = j.val →
      (i.val < green_flags ∧ j.val < green_flags → arrangement i ≠ arrangement j) ∧
      (i.val ≥ green_flags ∧ j.val ≥ green_flags → arrangement i ≠ arrangement j)) →
    Fintype.card {arrangement : Fin total_flags → Fin flagpoles //
      (∀ pole : Fin flagpoles, ∃ flag : Fin total_flags, arrangement flag = pole) ∧
      (∀ i j : Fin total_flags, i.val + 1 = j.val →
        (i.val < green_flags ∧ j.val < green_flags → arrangement i ≠ arrangement j) ∧
        (i.val ≥ green_flags ∧ j.val ≥ green_flags → arrangement i ≠ arrangement j))} = N) :=
by sorry

end NUMINAMATH_CALUDE_flag_arrangement_theorem_l2265_226545


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2265_226529

def set_A : Set ℤ := {x | |x| < 3}
def set_B : Set ℤ := {x | |x| > 1}

theorem intersection_of_A_and_B :
  set_A ∩ set_B = {-2, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2265_226529


namespace NUMINAMATH_CALUDE_object_speed_approximation_l2265_226567

/-- Given an object traveling 80 feet in 4 seconds, prove that its speed is approximately 13.64 miles per hour, given that 1 mile equals 5280 feet. -/
theorem object_speed_approximation : 
  let distance_feet : ℝ := 80
  let time_seconds : ℝ := 4
  let feet_per_mile : ℝ := 5280
  let seconds_per_hour : ℝ := 3600
  let speed_mph := (distance_feet / feet_per_mile) / (time_seconds / seconds_per_hour)
  ∃ ε > 0, |speed_mph - 13.64| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_object_speed_approximation_l2265_226567


namespace NUMINAMATH_CALUDE_additional_discount_percentage_l2265_226509

/-- Proves that the additional discount percentage for mothers with 3 or more children is 4% -/
theorem additional_discount_percentage : 
  let original_price : ℚ := 125
  let mothers_day_discount : ℚ := 10 / 100
  let final_price : ℚ := 108
  let price_after_initial_discount : ℚ := original_price * (1 - mothers_day_discount)
  let additional_discount_amount : ℚ := price_after_initial_discount - final_price
  let additional_discount_percentage : ℚ := additional_discount_amount / price_after_initial_discount * 100
  additional_discount_percentage = 4 := by sorry

end NUMINAMATH_CALUDE_additional_discount_percentage_l2265_226509


namespace NUMINAMATH_CALUDE_valid_numbers_l2265_226575

def is_valid_number (n : ℕ) : Prop :=
  ∃ x y : ℕ, 
    x ≤ 9 ∧ y ≤ 9 ∧
    n = 3000000 + x * 10000 + y * 100 + 3 ∧
    n % 13 = 0

theorem valid_numbers : 
  {n : ℕ | is_valid_number n} = 
  {3020303, 3050203, 3080103, 3090503, 3060603, 3030703, 3000803} := by
sorry

end NUMINAMATH_CALUDE_valid_numbers_l2265_226575


namespace NUMINAMATH_CALUDE_tens_digit_of_special_two_digit_number_l2265_226553

/-- The product of the digits of a two-digit number -/
def P (n : ℕ) : ℕ :=
  let tens := n / 10
  let ones := n % 10
  tens * ones

/-- The sum of the digits of a two-digit number -/
def S (n : ℕ) : ℕ :=
  let tens := n / 10
  let ones := n % 10
  tens + ones

/-- A two-digit number N satisfying N = P(N)^2 + S(N) has 1 as its tens digit -/
theorem tens_digit_of_special_two_digit_number :
  ∃ N : ℕ, 
    10 ≤ N ∧ N < 100 ∧ 
    N = (P N)^2 + S N ∧
    N / 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_special_two_digit_number_l2265_226553


namespace NUMINAMATH_CALUDE_mean_equality_implies_y_value_l2265_226538

theorem mean_equality_implies_y_value : 
  let nums : List ℝ := [4, 6, 10, 14]
  let mean_nums := (nums.sum) / (nums.length : ℝ)
  mean_nums = (y + 18) / 2 → y = -1 :=
by
  sorry

end NUMINAMATH_CALUDE_mean_equality_implies_y_value_l2265_226538


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2265_226520

/-- Given a quadratic equation and an isosceles triangle, prove the perimeter is 5 -/
theorem isosceles_triangle_perimeter (k : ℝ) : 
  let equation := fun x : ℝ => x^2 - (k+2)*x + 2*k
  ∃ (b c : ℝ), 
    equation b = 0 ∧ 
    equation c = 0 ∧ 
    b = c ∧ 
    b + c + 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2265_226520


namespace NUMINAMATH_CALUDE_complex_sum_problem_l2265_226525

theorem complex_sum_problem (p r t u : ℝ) :
  let q : ℝ := 5
  let s : ℝ := 2 * q
  t = -p - r →
  Complex.I * (q + s + u) = Complex.I * 7 →
  Complex.I * u + Complex.I = Complex.I * (-8) + Complex.I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_sum_problem_l2265_226525


namespace NUMINAMATH_CALUDE_percentage_not_working_l2265_226550

/-- Represents the employment status of a group --/
structure EmploymentStatus where
  fullTime : Rat
  partTime : Rat

/-- Represents the survey data --/
structure SurveyData where
  mothers : EmploymentStatus
  fathers : EmploymentStatus
  grandparents : EmploymentStatus
  womenPercentage : Rat
  menPercentage : Rat
  grandparentsPercentage : Rat

/-- Calculates the percentage of individuals not working in a given group --/
def notWorkingPercentage (status : EmploymentStatus) : Rat :=
  1 - status.fullTime - status.partTime

/-- Theorem stating the percentage of surveyed individuals not holding a job --/
theorem percentage_not_working (data : SurveyData) :
  data.mothers = { fullTime := 5/6, partTime := 1/6 } →
  data.fathers = { fullTime := 3/4, partTime := 1/8 } →
  data.grandparents = { fullTime := 1/2, partTime := 1/4 } →
  data.womenPercentage = 55/100 →
  data.menPercentage = 35/100 →
  data.grandparentsPercentage = 1/10 →
  (notWorkingPercentage data.mothers) * data.womenPercentage +
  (notWorkingPercentage data.fathers) * data.menPercentage +
  (notWorkingPercentage data.grandparents) * data.grandparentsPercentage =
  6875/100000 := by
  sorry

end NUMINAMATH_CALUDE_percentage_not_working_l2265_226550


namespace NUMINAMATH_CALUDE_triangle_isosceles_l2265_226551

/-- If in triangle ABC, a = 2b cos C, then triangle ABC is isosceles -/
theorem triangle_isosceles (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = π ∧  -- Sum of angles in a triangle
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- Sides are positive
  a = 2 * b * Real.cos C →  -- Given condition
  b = c  -- Definition of isosceles triangle
  := by sorry

end NUMINAMATH_CALUDE_triangle_isosceles_l2265_226551


namespace NUMINAMATH_CALUDE_unknown_number_is_ten_l2265_226542

-- Define the € operation
def euro (x y : ℝ) : ℝ := 2 * x * y

-- Theorem statement
theorem unknown_number_is_ten :
  ∀ n : ℝ, euro 8 (euro n 5) = 640 → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_is_ten_l2265_226542


namespace NUMINAMATH_CALUDE_waiter_tips_fraction_l2265_226501

theorem waiter_tips_fraction (salary : ℝ) (tips : ℝ) (income : ℝ) : 
  tips = (5/3) * salary → 
  income = salary + tips → 
  tips / income = 5/8 := by
sorry

end NUMINAMATH_CALUDE_waiter_tips_fraction_l2265_226501


namespace NUMINAMATH_CALUDE_intersection_nonempty_iff_a_in_range_l2265_226583

def set_A (a : ℝ) : Set ℝ := {y | y > a^2 + 1 ∨ y < a}
def set_B : Set ℝ := {y | 2 ≤ y ∧ y ≤ 4}

theorem intersection_nonempty_iff_a_in_range (a : ℝ) :
  (set_A a ∩ set_B).Nonempty ↔ (-Real.sqrt 3 < a ∧ a < Real.sqrt 3) ∨ a > 2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_nonempty_iff_a_in_range_l2265_226583


namespace NUMINAMATH_CALUDE_square_sum_of_xy_l2265_226512

theorem square_sum_of_xy (x y : ℕ+) 
  (h1 : x * y + x + y = 83)
  (h2 : x^2 * y + x * y^2 = 1056) : 
  x^2 + y^2 = 458 := by sorry

end NUMINAMATH_CALUDE_square_sum_of_xy_l2265_226512


namespace NUMINAMATH_CALUDE_unique_solution_l2265_226531

-- Define the functions f and p
def f (x : ℝ) : ℝ := |x + 1|

def p (x a : ℝ) : ℝ := |2*x + 5| + a

-- Define the set of values for 'a'
def A : Set ℝ := {-6.5, -5, 1.5}

-- State the theorem
theorem unique_solution (a : ℝ) :
  a ∈ A ↔ ∃! x : ℝ, x ≠ 1 ∧ x ≠ 2.5 ∧ f x = p x a :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l2265_226531


namespace NUMINAMATH_CALUDE_unique_solution_trigonometric_equation_l2265_226584

theorem unique_solution_trigonometric_equation :
  ∃! x : ℝ, 0 < x ∧ x < 180 ∧
  Real.tan ((150 - x) * π / 180) = 
    (Real.sin (150 * π / 180) - Real.sin (x * π / 180)) /
    (Real.cos (150 * π / 180) - Real.cos (x * π / 180)) ∧
  x = 110 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_trigonometric_equation_l2265_226584


namespace NUMINAMATH_CALUDE_triangle_area_l2265_226522

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop := True

-- Define the length of a side
def length (p q : ℝ × ℝ) : ℝ := sorry

-- Define the angle between two sides
def angle (p q r : ℝ × ℝ) : ℝ := sorry

-- Define the area of a triangle
def area (A B C : ℝ × ℝ) : ℝ := sorry

theorem triangle_area (A B C : ℝ × ℝ) :
  Triangle A B C →
  length A B = 6 →
  angle B A C = 30 * π / 180 →
  angle A B C = 120 * π / 180 →
  area A B C = 9 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l2265_226522


namespace NUMINAMATH_CALUDE_f_two_roots_range_l2265_226559

/-- The cubic function f(x) = x^3 - 3x + 5 -/
def f (x : ℝ) : ℝ := x^3 - 3*x + 5

/-- Theorem stating the range of a for which f(x) = a has at least two distinct real roots -/
theorem f_two_roots_range :
  ∀ a : ℝ, (∃ x y : ℝ, x ≠ y ∧ f x = a ∧ f y = a) ↔ 3 ≤ a ∧ a ≤ 7 := by
  sorry

end NUMINAMATH_CALUDE_f_two_roots_range_l2265_226559


namespace NUMINAMATH_CALUDE_necklace_cost_proof_l2265_226582

/-- The cost of a single necklace -/
def necklace_cost : ℝ := 40000

/-- The total cost of the purchase -/
def total_cost : ℝ := 240000

/-- The number of necklaces purchased -/
def num_necklaces : ℕ := 3

theorem necklace_cost_proof :
  (num_necklaces : ℝ) * necklace_cost + 3 * necklace_cost = total_cost :=
by sorry

end NUMINAMATH_CALUDE_necklace_cost_proof_l2265_226582


namespace NUMINAMATH_CALUDE_scientific_notation_of_1300000000_l2265_226526

/-- Expresses 1300000000 in scientific notation -/
theorem scientific_notation_of_1300000000 :
  (1300000000 : ℝ) = 1.3 * (10 ^ 9) := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_1300000000_l2265_226526


namespace NUMINAMATH_CALUDE_original_number_proof_l2265_226514

theorem original_number_proof :
  ∃ (n : ℕ), n = 3830 ∧ (∃ (k : ℕ), n - 5 = 15 * k) ∧
  (∀ (m : ℕ), m < 5 → ¬(∃ (j : ℕ), n - m = 15 * j)) :=
by sorry

end NUMINAMATH_CALUDE_original_number_proof_l2265_226514


namespace NUMINAMATH_CALUDE_units_digit_of_L_L15_l2265_226516

/-- Lucas numbers sequence -/
def Lucas : ℕ → ℕ
  | 0 => 2
  | 1 => 1
  | (n + 2) => Lucas (n + 1) + Lucas n

/-- The period of the units digit in the Lucas sequence -/
def LucasPeriod : ℕ := 12

theorem units_digit_of_L_L15 : 
  (Lucas (Lucas 15)) % 10 = 7 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_L_L15_l2265_226516


namespace NUMINAMATH_CALUDE_zero_in_interval_l2265_226506

def f (a x : ℝ) : ℝ := 3 * a * x - 1 - 2 * a

theorem zero_in_interval (a : ℝ) : 
  (∃ x ∈ Set.Ioo (-1) 1, f a x = 0) → 
  (a < -1/5 ∨ a > 1) := by
sorry

end NUMINAMATH_CALUDE_zero_in_interval_l2265_226506


namespace NUMINAMATH_CALUDE_unique_solution_2m_minus_1_eq_3n_l2265_226598

theorem unique_solution_2m_minus_1_eq_3n :
  ∀ m n : ℕ+, 2^(m : ℕ) - 1 = 3^(n : ℕ) ↔ m = 2 ∧ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_2m_minus_1_eq_3n_l2265_226598


namespace NUMINAMATH_CALUDE_part1_min_max_part2_t_range_l2265_226580

-- Define the function f
def f (t : ℝ) (x : ℝ) : ℝ := x^2 + 2*t*x + t - 1

-- Part 1
theorem part1_min_max :
  let t := 2
  ∃ (min max : ℝ),
    (∀ x ∈ Set.Icc (-3) 1, f t x ≥ min) ∧
    (∃ x ∈ Set.Icc (-3) 1, f t x = min) ∧
    (∀ x ∈ Set.Icc (-3) 1, f t x ≤ max) ∧
    (∃ x ∈ Set.Icc (-3) 1, f t x = max) ∧
    min = -3 ∧ max = 6 :=
sorry

-- Part 2
theorem part2_t_range :
  {t : ℝ | ∀ x ∈ Set.Icc 1 2, f t x > 0} = Set.Ioi 0 :=
sorry

end NUMINAMATH_CALUDE_part1_min_max_part2_t_range_l2265_226580


namespace NUMINAMATH_CALUDE_neg_one_pow_2022_eq_one_neg_one_pow_2022_and_one_are_opposite_l2265_226561

/-- Two real numbers are opposite if their sum is zero -/
def are_opposite (a b : ℝ) : Prop := a + b = 0

/-- -1^2022 equals 1 -/
theorem neg_one_pow_2022_eq_one : (-1 : ℝ)^2022 = 1 := by sorry

theorem neg_one_pow_2022_and_one_are_opposite :
  are_opposite ((-1 : ℝ)^2022) 1 := by sorry

end NUMINAMATH_CALUDE_neg_one_pow_2022_eq_one_neg_one_pow_2022_and_one_are_opposite_l2265_226561


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2265_226557

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 5*x + 4) * (x^2 + 3*x + 2) + (x^2 + 4*x - 3) = 
  (x^2 + 4*x + 2) * (x^2 + 2*x + 4) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2265_226557


namespace NUMINAMATH_CALUDE_probability_at_least_six_sevens_l2265_226585

-- Define the number of sides on the die
def die_sides : ℕ := 8

-- Define the number of rolls
def num_rolls : ℕ := 7

-- Define the minimum number of successful rolls required
def min_successes : ℕ := 6

-- Define the minimum value considered a success
def success_value : ℕ := 7

-- Function to calculate the probability of a single success
def single_success_prob : ℚ := (die_sides - success_value + 1) / die_sides

-- Function to calculate the probability of exactly k successes in n rolls
def exact_success_prob (n k : ℕ) : ℚ :=
  Nat.choose n k * single_success_prob^k * (1 - single_success_prob)^(n - k)

-- Theorem statement
theorem probability_at_least_six_sevens :
  (exact_success_prob num_rolls min_successes + exact_success_prob num_rolls (num_rolls)) = 11 / 2048 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_six_sevens_l2265_226585


namespace NUMINAMATH_CALUDE_divisors_of_180_l2265_226549

theorem divisors_of_180 : Nat.card (Nat.divisors 180) = 18 := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_180_l2265_226549


namespace NUMINAMATH_CALUDE_product_remainder_l2265_226543

theorem product_remainder (a b m : ℕ) (ha : a = 1488) (hb : b = 1977) (hm : m = 500) :
  (a * b) % m = 276 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_l2265_226543


namespace NUMINAMATH_CALUDE_high_school_ten_games_l2265_226568

def league_size : ℕ := 10
def non_league_games_per_team : ℕ := 6

def intra_league_games (n : ℕ) : ℕ :=
  n * (n - 1)

def total_games (n : ℕ) (m : ℕ) : ℕ :=
  (intra_league_games n) + (n * m)

theorem high_school_ten_games :
  total_games league_size non_league_games_per_team = 150 := by
  sorry

end NUMINAMATH_CALUDE_high_school_ten_games_l2265_226568


namespace NUMINAMATH_CALUDE_exists_unprovable_by_induction_l2265_226508

-- Define a proposition that represents a mathematical statement
def MathStatement : Type := Prop

-- Define a function that represents the ability to prove a statement by induction
def ProvableByInduction (s : MathStatement) : Prop := sorry

-- Theorem: There exists a true mathematical statement that cannot be proven by induction
theorem exists_unprovable_by_induction : 
  ∃ (s : MathStatement), s ∧ ¬(ProvableByInduction s) := by sorry

end NUMINAMATH_CALUDE_exists_unprovable_by_induction_l2265_226508


namespace NUMINAMATH_CALUDE_equal_sum_sequence_2011_sum_l2265_226597

/-- Definition of an equal sum sequence -/
def IsEqualSumSequence (a : ℕ → ℤ) (sum : ℤ) : Prop :=
  ∀ n : ℕ, a n + a (n + 1) = sum

/-- Definition of the sum of the first n terms of a sequence -/
def SequenceSum (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  (Finset.range n).sum a

theorem equal_sum_sequence_2011_sum
  (a : ℕ → ℤ)
  (h_equal_sum : IsEqualSumSequence a 1)
  (h_first_term : a 1 = -1) :
  SequenceSum a 2011 = 1004 := by
sorry

end NUMINAMATH_CALUDE_equal_sum_sequence_2011_sum_l2265_226597


namespace NUMINAMATH_CALUDE_perimeter_of_square_d_l2265_226548

/-- Given two squares C and D, where C has a side length of 10 cm and D has an area
    that is half the area of C, the perimeter of D is 20√2 cm. -/
theorem perimeter_of_square_d (c d : Real) : 
  c = 10 →  -- side length of square C
  d ^ 2 = (c ^ 2) / 2 →  -- area of D is half the area of C
  4 * d = 20 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_perimeter_of_square_d_l2265_226548


namespace NUMINAMATH_CALUDE_intersection_point_theorem_l2265_226596

/-- A parabola that intersects the coordinate axes at three distinct points -/
structure Parabola where
  p : ℝ
  q : ℝ
  distinct_intersections : ∃ (a b : ℝ), a ≠ b ∧ a ≠ 0 ∧ b ≠ 0 ∧ q ≠ 0 ∧ a^2 + p*a + q = 0 ∧ b^2 + p*b + q = 0

/-- The circle passing through the three intersection points of the parabola with the coordinate axes -/
def intersection_circle (par : Parabola) : Set (ℝ × ℝ) :=
  {(x, y) | ∃ (a b : ℝ), a ≠ b ∧ a ≠ 0 ∧ b ≠ 0 ∧ par.q ≠ 0 ∧
            a^2 + par.p*a + par.q = 0 ∧ b^2 + par.p*b + par.q = 0 ∧
            (x^2 + y^2) * (a*b) + x * (par.q*(a+b)) + y * (par.q*par.p) + par.q^2 = 0}

/-- Theorem: All intersection circles pass through the point (0, 1) -/
theorem intersection_point_theorem (par : Parabola) :
  (0, 1) ∈ intersection_circle par :=
sorry

end NUMINAMATH_CALUDE_intersection_point_theorem_l2265_226596


namespace NUMINAMATH_CALUDE_min_value_expression_l2265_226573

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  a^2 + 8*a*b + 24*b^2 + 16*b*c + 6*c^2 ≥ 18 ∧
  ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ a₀ * b₀ * c₀ = 1 ∧
    a₀^2 + 8*a₀*b₀ + 24*b₀^2 + 16*b₀*c₀ + 6*c₀^2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l2265_226573


namespace NUMINAMATH_CALUDE_fifteen_percent_of_thousand_is_150_l2265_226591

theorem fifteen_percent_of_thousand_is_150 :
  (15 / 100) * 1000 = 150 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_percent_of_thousand_is_150_l2265_226591


namespace NUMINAMATH_CALUDE_quadratic_function_bound_l2265_226521

def f (p q : ℝ) (x : ℝ) : ℝ := x^2 + p*x + q

theorem quadratic_function_bound (p q : ℝ) :
  (max (|f p q 1|) (max (|f p q 2|) (|f p q 3|))) ≥ (1/2 : ℝ) := by sorry

end NUMINAMATH_CALUDE_quadratic_function_bound_l2265_226521


namespace NUMINAMATH_CALUDE_cubic_root_sum_squares_over_product_l2265_226500

theorem cubic_root_sum_squares_over_product (k : ℤ) (hk : k ≠ 0) 
  (a b c : ℂ) (h : ∀ x : ℂ, x^3 + 10*x^2 + 5*x - k = 0 ↔ x = a ∨ x = b ∨ x = c) : 
  (a^2 + b^2 + c^2) / (a * b * c) = 90 / k := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_squares_over_product_l2265_226500


namespace NUMINAMATH_CALUDE_triangle_side_length_l2265_226555

-- Define the triangle PQR
structure Triangle :=
  (P Q R : ℝ × ℝ)

-- Define the properties of the triangle
def is_valid_triangle (t : Triangle) (pq pr pn : ℝ) : Prop :=
  let (px, py) := t.P
  let (qx, qy) := t.Q
  let (rx, ry) := t.R
  let (nx, ny) := ((qx + rx) / 2, (qy + ry) / 2)  -- N is midpoint of QR
  (px - qx)^2 + (py - qy)^2 = pq^2 ∧  -- PQ = 6
  (px - rx)^2 + (py - ry)^2 = pr^2 ∧  -- PR = 10
  (px - nx)^2 + (py - ny)^2 = pn^2    -- PN = 5

-- Theorem statement
theorem triangle_side_length (t : Triangle) :
  is_valid_triangle t 6 10 5 →
  let (qx, qy) := t.Q
  let (rx, ry) := t.R
  (qx - rx)^2 + (qy - ry)^2 = 4 * 43 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2265_226555


namespace NUMINAMATH_CALUDE_sqrt_fraction_eval_l2265_226524

theorem sqrt_fraction_eval (x : ℝ) (h : x < -1) :
  Real.sqrt (x / (1 - (2 * x - 3) / (x + 1))) = Complex.I * Real.sqrt (x^2 - 3*x - 4) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fraction_eval_l2265_226524


namespace NUMINAMATH_CALUDE_f_is_quadratic_l2265_226590

/-- A function f : ℝ → ℝ is quadratic if there exist real numbers a, b, and c
    with a ≠ 0 such that f(x) = ax^2 + bx + c for all x ∈ ℝ. -/
def IsQuadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function f(x) = (x + 1)^2 - 5 -/
def f (x : ℝ) : ℝ := (x + 1)^2 - 5

/-- Theorem: The function f(x) = (x + 1)^2 - 5 is a quadratic function -/
theorem f_is_quadratic : IsQuadratic f := by
  sorry


end NUMINAMATH_CALUDE_f_is_quadratic_l2265_226590


namespace NUMINAMATH_CALUDE_simplify_expression_l2265_226546

theorem simplify_expression (x : ℝ) : 7*x + 8 - 3*x - 4 + 5 = 4*x + 9 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2265_226546


namespace NUMINAMATH_CALUDE_equal_intercept_line_equation_l2265_226537

/-- A line passing through point (1, 1) with equal intercepts on both coordinate axes -/
def equal_intercept_line (x y : ℝ) : Prop :=
  (x = 1 ∧ y = 1) ∨ 
  (∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ a * x + b * y = a * b ∧ a = b)

/-- The equation of the line is x - y = 0 or x + y - 2 = 0 -/
theorem equal_intercept_line_equation (x y : ℝ) :
  equal_intercept_line x y ↔ (x - y = 0 ∨ x + y - 2 = 0) := by
  sorry

end NUMINAMATH_CALUDE_equal_intercept_line_equation_l2265_226537


namespace NUMINAMATH_CALUDE_expression_value_l2265_226587

theorem expression_value : (5^2 - 5 - 12) / (5 - 4) = 8 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2265_226587


namespace NUMINAMATH_CALUDE_collatz_eighth_term_one_l2265_226599

def collatz (n : ℕ) : ℕ :=
  if n % 2 = 0 then n / 2 else 3 * n + 1

def collatzSequence (n : ℕ) : ℕ → ℕ
  | 0 => n
  | k + 1 => collatz (collatzSequence n k)

def validStartingNumbers : Set ℕ :=
  {n | n > 0 ∧ collatzSequence n 7 = 1}

theorem collatz_eighth_term_one :
  validStartingNumbers = {2, 3, 16, 20, 21, 128} :=
sorry

end NUMINAMATH_CALUDE_collatz_eighth_term_one_l2265_226599


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l2265_226593

theorem arithmetic_mean_problem (a b c d : ℝ) 
  (h1 : (a + d) / 2 = 40)
  (h2 : (b + d) / 2 = 60)
  (h3 : (a + b) / 2 = 50)
  (h4 : (b + c) / 2 = 70) :
  c - a = 40 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l2265_226593


namespace NUMINAMATH_CALUDE_product_125_sum_31_l2265_226556

theorem product_125_sum_31 :
  ∃ (a b c : ℕ+), 
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (a * b * c : ℕ) = 125 →
    (a + b + c : ℕ) = 31 := by
sorry

end NUMINAMATH_CALUDE_product_125_sum_31_l2265_226556


namespace NUMINAMATH_CALUDE_cricket_team_age_difference_l2265_226540

/-- Prove that the difference between the average age of the whole cricket team
and the average age of the remaining players is 3 years. -/
theorem cricket_team_age_difference 
  (team_size : ℕ) 
  (team_avg_age : ℝ) 
  (wicket_keeper_age_diff : ℝ) 
  (remaining_avg_age : ℝ) 
  (h1 : team_size = 11)
  (h2 : team_avg_age = 26)
  (h3 : wicket_keeper_age_diff = 3)
  (h4 : remaining_avg_age = 23) :
  team_avg_age - remaining_avg_age = 3 := by
sorry

end NUMINAMATH_CALUDE_cricket_team_age_difference_l2265_226540


namespace NUMINAMATH_CALUDE_total_tickets_won_l2265_226581

/-- Represents the number of tickets Dave used for toys. -/
def tickets_for_toys : ℕ := 8

/-- Represents the number of tickets Dave used for clothes. -/
def tickets_for_clothes : ℕ := 18

/-- Represents the difference in tickets used for clothes versus toys. -/
def difference_clothes_toys : ℕ := 10

/-- Theorem stating that the total number of tickets Dave won is the sum of
    tickets used for toys and clothes. -/
theorem total_tickets_won (hw : tickets_for_clothes = tickets_for_toys + difference_clothes_toys) :
  tickets_for_toys + tickets_for_clothes = 26 := by
  sorry

#check total_tickets_won

end NUMINAMATH_CALUDE_total_tickets_won_l2265_226581


namespace NUMINAMATH_CALUDE_no_solution_for_a_l2265_226592

theorem no_solution_for_a (x a : ℝ) : x = 4 → 1 / (x + a) + 1 / (x - a) ≠ 1 / (x - a) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_a_l2265_226592


namespace NUMINAMATH_CALUDE_max_three_digit_operation_l2265_226503

theorem max_three_digit_operation :
  ∀ n : ℕ, 100 ≤ n ∧ n ≤ 999 → 2 * (200 + n) ≤ 2398 :=
by sorry

end NUMINAMATH_CALUDE_max_three_digit_operation_l2265_226503


namespace NUMINAMATH_CALUDE_intersection_A_B_l2265_226505

-- Define set A
def A : Set ℝ := {x | |x - 1| < 2}

-- Define set B
def B : Set ℝ := {y | ∃ x ∈ Set.Icc 0 2, y = 2^x}

-- Theorem statement
theorem intersection_A_B : A ∩ B = Set.Ici 1 ∩ Set.Iio 3 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l2265_226505


namespace NUMINAMATH_CALUDE_chef_fries_problem_l2265_226576

/-- Given a chef making fries, prove the number of fries needed. -/
theorem chef_fries_problem (fries_per_potato : ℕ) (total_potatoes : ℕ) (leftover_potatoes : ℕ) :
  fries_per_potato = 25 →
  total_potatoes = 15 →
  leftover_potatoes = 7 →
  fries_per_potato * (total_potatoes - leftover_potatoes) = 200 := by
  sorry

end NUMINAMATH_CALUDE_chef_fries_problem_l2265_226576


namespace NUMINAMATH_CALUDE_pizza_cost_l2265_226544

theorem pizza_cost (total_cost : ℝ) (num_pizzas : ℕ) (h1 : total_cost = 24) (h2 : num_pizzas = 3) :
  total_cost / num_pizzas = 8 := by
  sorry

end NUMINAMATH_CALUDE_pizza_cost_l2265_226544


namespace NUMINAMATH_CALUDE_problem_statement_l2265_226570

theorem problem_statement (a b : ℝ) (h1 : a - b = 1) (h2 : a * b = -2) :
  (a + 1) * (b - 1) = -4 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2265_226570


namespace NUMINAMATH_CALUDE_complement_union_theorem_l2265_226554

open Set

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4}

theorem complement_union_theorem : (U \ A) ∪ B = {0, 2, 4} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l2265_226554


namespace NUMINAMATH_CALUDE_x_one_minus_f_equals_seven_to_500_l2265_226562

theorem x_one_minus_f_equals_seven_to_500 :
  let x : ℝ := (3 + Real.sqrt 2) ^ 500
  let n : ℤ := ⌊x⌋
  let f : ℝ := x - n
  x * (1 - f) = 7 ^ 500 := by
sorry

end NUMINAMATH_CALUDE_x_one_minus_f_equals_seven_to_500_l2265_226562


namespace NUMINAMATH_CALUDE_paco_cookies_problem_l2265_226533

/-- Calculates the initial number of salty cookies Paco had --/
def initial_salty_cookies (initial_sweet : ℕ) (sweet_eaten : ℕ) (salty_eaten : ℕ) (difference : ℕ) : ℕ :=
  initial_sweet - difference

theorem paco_cookies_problem (initial_sweet : ℕ) (sweet_eaten : ℕ) (salty_eaten : ℕ) (difference : ℕ)
  (h1 : initial_sweet = 39)
  (h2 : sweet_eaten = 32)
  (h3 : salty_eaten = 23)
  (h4 : difference = sweet_eaten - salty_eaten)
  (h5 : difference = 9) :
  initial_salty_cookies initial_sweet sweet_eaten salty_eaten difference = 30 := by
  sorry

#eval initial_salty_cookies 39 32 23 9

end NUMINAMATH_CALUDE_paco_cookies_problem_l2265_226533


namespace NUMINAMATH_CALUDE_clothes_fraction_l2265_226534

def incentive : ℚ := 240
def food_fraction : ℚ := 1/3
def savings_fraction : ℚ := 3/4
def savings_amount : ℚ := 84

theorem clothes_fraction (clothes_amount : ℚ) 
  (h1 : clothes_amount = incentive - food_fraction * incentive - savings_amount / savings_fraction) 
  (h2 : clothes_amount / incentive = 1/5) : 
  clothes_amount / incentive = 1/5 := by
sorry

end NUMINAMATH_CALUDE_clothes_fraction_l2265_226534


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l2265_226569

theorem sum_of_solutions_quadratic (z : ℂ) : 
  (z^2 = 16*z - 10) → (∃ (z1 z2 : ℂ), z1^2 = 16*z1 - 10 ∧ z2^2 = 16*z2 - 10 ∧ z1 + z2 = 16) :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l2265_226569


namespace NUMINAMATH_CALUDE_roxanne_lemonade_purchase_l2265_226579

/-- Represents the purchase of lemonade and sandwiches -/
structure Purchase where
  lemonade_price : ℚ
  sandwich_price : ℚ
  sandwich_count : ℕ
  paid_amount : ℚ
  change_received : ℚ

/-- Calculates the number of lemonade cups bought -/
def lemonade_cups (p : Purchase) : ℚ :=
  (p.paid_amount - p.change_received - p.sandwich_price * p.sandwich_count) / p.lemonade_price

/-- Theorem stating that Roxanne bought 2 cups of lemonade -/
theorem roxanne_lemonade_purchase :
  let p : Purchase := {
    lemonade_price := 2,
    sandwich_price := 5/2,
    sandwich_count := 2,
    paid_amount := 20,
    change_received := 11
  }
  lemonade_cups p = 2 := by sorry

end NUMINAMATH_CALUDE_roxanne_lemonade_purchase_l2265_226579


namespace NUMINAMATH_CALUDE_orchid_rose_difference_l2265_226589

-- Define the initial and final counts of roses and orchids
def initial_roses : ℕ := 7
def initial_orchids : ℕ := 12
def final_roses : ℕ := 11
def final_orchids : ℕ := 20

-- Theorem to prove
theorem orchid_rose_difference :
  final_orchids - final_roses = 9 :=
by sorry

end NUMINAMATH_CALUDE_orchid_rose_difference_l2265_226589


namespace NUMINAMATH_CALUDE_exists_solution_for_prime_l2265_226535

theorem exists_solution_for_prime (p : ℕ) (hp : Prime p) :
  ∃ (x y z w : ℤ), x^2 + y^2 + z^2 - w * ↑p = 0 ∧ 0 < w ∧ w < ↑p :=
by sorry

end NUMINAMATH_CALUDE_exists_solution_for_prime_l2265_226535


namespace NUMINAMATH_CALUDE_larger_integer_problem_l2265_226502

theorem larger_integer_problem (x y : ℤ) (h1 : x + y = -9) (h2 : x - y = 1) : x = -4 := by
  sorry

end NUMINAMATH_CALUDE_larger_integer_problem_l2265_226502


namespace NUMINAMATH_CALUDE_overtake_time_l2265_226523

/-- The time when b starts relative to a's start time. -/
def b_start_time : ℝ := 15

/-- The speed of person a in km/hr. -/
def speed_a : ℝ := 30

/-- The speed of person b in km/hr. -/
def speed_b : ℝ := 40

/-- The speed of person k in km/hr. -/
def speed_k : ℝ := 60

/-- The time when k starts relative to a's start time. -/
def k_start_time : ℝ := 10

theorem overtake_time (t : ℝ) : 
  speed_a * t = speed_b * (t - b_start_time) ∧ 
  speed_a * t = speed_k * (t - k_start_time) → 
  b_start_time = 15 := by sorry

end NUMINAMATH_CALUDE_overtake_time_l2265_226523


namespace NUMINAMATH_CALUDE_divisibility_of_sum_of_squares_l2265_226536

theorem divisibility_of_sum_of_squares (p a b : ℕ) : 
  Prime p → 
  (∃ n : ℕ, p = 4 * n + 3) → 
  p ∣ (a^2 + b^2) → 
  (p ∣ a ∧ p ∣ b) := by
sorry

end NUMINAMATH_CALUDE_divisibility_of_sum_of_squares_l2265_226536


namespace NUMINAMATH_CALUDE_property_set_characterization_l2265_226560

/-- The property that a^(n+1) ≡ a (mod n) holds for all integers a -/
def has_property (n : ℕ) : Prop :=
  ∀ a : ℤ, (a^(n+1) : ℤ) ≡ a [ZMOD n]

/-- The set of integers satisfying the property -/
def property_set : Set ℕ := {n | has_property n}

/-- Theorem stating that the set of integers satisfying the property is exactly {1, 2, 6, 42, 1806} -/
theorem property_set_characterization :
  property_set = {1, 2, 6, 42, 1806} := by sorry

end NUMINAMATH_CALUDE_property_set_characterization_l2265_226560


namespace NUMINAMATH_CALUDE_container_volume_ratio_l2265_226504

theorem container_volume_ratio (A B : ℝ) (h1 : A > 0) (h2 : B > 0) : 
  (2/3 * A = 1/2 * B) → (A / B = 3/4) := by
  sorry

end NUMINAMATH_CALUDE_container_volume_ratio_l2265_226504


namespace NUMINAMATH_CALUDE_tan_alpha_value_l2265_226574

theorem tan_alpha_value (α : Real) (h : Real.tan (α + π/4) = 1/2) : Real.tan α = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l2265_226574


namespace NUMINAMATH_CALUDE_age_ratio_theorem_l2265_226566

/-- Represents the ages of two people A and B -/
structure Ages where
  a : ℕ
  b : ℕ

/-- The ratio between A's and B's present ages is 6:3 -/
def present_ratio (ages : Ages) : Prop :=
  2 * ages.b = ages.a

/-- The ratio between A's age 4 years ago and B's age 4 years hence is 1:1 -/
def past_future_ratio (ages : Ages) : Prop :=
  ages.a - 4 = ages.b + 4

/-- The ratio between A's age 4 years hence and B's age 4 years ago is 5:1 -/
def future_past_ratio (ages : Ages) : Prop :=
  5 * (ages.b - 4) = ages.a + 4

/-- Theorem stating the relationship between the given conditions and the result -/
theorem age_ratio_theorem (ages : Ages) :
  present_ratio ages → past_future_ratio ages → future_past_ratio ages :=
by
  sorry

end NUMINAMATH_CALUDE_age_ratio_theorem_l2265_226566


namespace NUMINAMATH_CALUDE_even_function_derivative_odd_function_derivative_l2265_226552

variable (f : ℝ → ℝ)

-- Define even and odd functions
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Theorem for even functions
theorem even_function_derivative (h : IsEven f) (h' : Differentiable ℝ f) :
  IsOdd (deriv f) := by sorry

-- Theorem for odd functions
theorem odd_function_derivative (h : IsOdd f) (h' : Differentiable ℝ f) :
  IsEven (deriv f) := by sorry

end NUMINAMATH_CALUDE_even_function_derivative_odd_function_derivative_l2265_226552


namespace NUMINAMATH_CALUDE_sarahs_brother_apples_l2265_226588

theorem sarahs_brother_apples (sarah_apples : ℝ) (ratio : ℝ) (brother_apples : ℝ) : 
  sarah_apples = 45.0 →
  sarah_apples = ratio * brother_apples →
  ratio = 5 →
  brother_apples = 9.0 := by
sorry

end NUMINAMATH_CALUDE_sarahs_brother_apples_l2265_226588


namespace NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l2265_226507

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

/-- The theorem states that in a geometric sequence of positive numbers where the third term is 16 and the seventh term is 2, the fifth term is 2. -/
theorem geometric_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_positive : ∀ n, a n > 0)
  (h_geometric : IsGeometricSequence a)
  (h_third_term : a 3 = 16)
  (h_seventh_term : a 7 = 2) :
  a 5 = 2 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l2265_226507


namespace NUMINAMATH_CALUDE_four_digit_sum_11990_l2265_226547

def is_valid_digit (d : ℕ) : Prop := d > 0 ∧ d < 10

def distinct_digits (a b c d : ℕ) : Prop :=
  is_valid_digit a ∧ is_valid_digit b ∧ is_valid_digit c ∧ is_valid_digit d ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

def smallest_number (a b c d : ℕ) : ℕ := 1000 * a + 100 * b + 10 * c + d
def largest_number (a b c d : ℕ) : ℕ := 1000 * d + 100 * c + 10 * b + a

theorem four_digit_sum_11990 (a b c d : ℕ) :
  distinct_digits a b c d →
  (smallest_number a b c d + largest_number a b c d = 11990 ↔
   ((a = 1 ∧ b = 9 ∧ c = 9 ∧ d = 9) ∨ (a = 9 ∧ b = 9 ∧ c = 9 ∧ d = 1))) :=
by sorry

end NUMINAMATH_CALUDE_four_digit_sum_11990_l2265_226547


namespace NUMINAMATH_CALUDE_midpoint_trajectory_l2265_226513

/-- The trajectory of the midpoint of a line segment with one fixed endpoint and the other moving on a circle -/
theorem midpoint_trajectory (x y : ℝ) : 
  (∃ x₀ y₀ : ℝ, (x₀ + 1)^2 + y₀^2 = 4 ∧ x = (x₀ + 4)/2 ∧ y = (y₀ + 3)/2) → 
  (x - 3/2)^2 + (y - 3/2)^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_midpoint_trajectory_l2265_226513


namespace NUMINAMATH_CALUDE_equation_solution_l2265_226511

theorem equation_solution :
  ∃! y : ℝ, (7 : ℝ) ^ (y + 6) = 343 ^ y :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2265_226511


namespace NUMINAMATH_CALUDE_white_white_overlapping_pairs_l2265_226517

/-- Represents the number of triangles of each color in each half of the figure -/
structure TriangleCounts where
  red : ℕ
  blue : ℕ
  white : ℕ

/-- Represents the number of overlapping pairs of each type when the figure is folded -/
structure OverlappingPairs where
  redRed : ℕ
  blueBlue : ℕ
  redWhite : ℕ

/-- The main theorem stating the number of white-white overlapping pairs -/
theorem white_white_overlapping_pairs
  (counts : TriangleCounts)
  (overlaps : OverlappingPairs)
  (h1 : counts.red = 4)
  (h2 : counts.blue = 6)
  (h3 : counts.white = 9)
  (h4 : overlaps.redRed = 3)
  (h5 : overlaps.blueBlue = 4)
  (h6 : overlaps.redWhite = 3) :
  counts.white - overlaps.redWhite = 6 :=
sorry

end NUMINAMATH_CALUDE_white_white_overlapping_pairs_l2265_226517


namespace NUMINAMATH_CALUDE_jills_work_days_month3_l2265_226586

def daily_rate_month1 : ℕ := 10
def daily_rate_month2 : ℕ := 2 * daily_rate_month1
def daily_rate_month3 : ℕ := daily_rate_month2
def days_per_month : ℕ := 30
def total_earnings : ℕ := 1200

def earnings_month1 : ℕ := daily_rate_month1 * days_per_month
def earnings_month2 : ℕ := daily_rate_month2 * days_per_month

theorem jills_work_days_month3 :
  (total_earnings - earnings_month1 - earnings_month2) / daily_rate_month3 = 15 := by
  sorry

end NUMINAMATH_CALUDE_jills_work_days_month3_l2265_226586


namespace NUMINAMATH_CALUDE_right_triangle_area_right_triangle_area_proof_l2265_226564

theorem right_triangle_area : ℕ → ℕ → ℕ → Prop :=
  fun a b c =>
    (a * a + b * b = c * c) →  -- Pythagorean theorem
    (2 * b * b - 23 * b + 11 = 0) →  -- One leg satisfies the equation
    (a > 0 ∧ b > 0 ∧ c > 0) →  -- All sides are positive
    ((a * b) / 2 = 330)  -- Area of the triangle

-- The proof of this theorem
theorem right_triangle_area_proof :
  ∃ (a b c : ℕ), right_triangle_area a b c :=
sorry

end NUMINAMATH_CALUDE_right_triangle_area_right_triangle_area_proof_l2265_226564


namespace NUMINAMATH_CALUDE_parabola_tangent_constant_l2265_226572

-- Define the points and parameters
variable (p q : ℝ)
variable (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ)

-- Define the conditions
def on_parabola (x y : ℝ) : Prop := y^2 = 2 * p * x

def is_tangent (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  ∃ (xt yt : ℝ), (yt - y₁) * (x₂ - x₁) = (y₂ - y₁) * (xt - x₁) ∧
                 xt^2 = 2 * q * yt

-- State the theorem
theorem parabola_tangent_constant 
  (h_p : p > 0)
  (h_q : q > 0)
  (h_distinct : (x₁, y₁) ≠ (x₂, y₂) ∧ (x₁, y₁) ≠ (x₃, y₃) ∧ (x₂, y₂) ≠ (x₃, y₃))
  (h_on_parabola₁ : on_parabola p x₁ y₁)
  (h_on_parabola₂ : on_parabola p x₂ y₂)
  (h_on_parabola₃ : on_parabola p x₃ y₃)
  (h_tangent₁₂ : is_tangent q x₁ y₁ x₂ y₂)
  (h_tangent₁₃ : is_tangent q x₁ y₁ x₃ y₃) :
  y₁ * y₂ * (y₁ + y₂) = -2 * p^2 * q ∧
  y₁ * y₃ * (y₁ + y₃) = -2 * p^2 * q ∧
  y₂ * y₃ * (y₂ + y₃) = -2 * p^2 * q :=
sorry

end NUMINAMATH_CALUDE_parabola_tangent_constant_l2265_226572


namespace NUMINAMATH_CALUDE_total_capsules_sold_l2265_226518

def weekly_earnings_100mg : ℕ := 80
def weekly_earnings_500mg : ℕ := 60
def cost_per_capsule_100mg : ℕ := 5
def cost_per_capsule_500mg : ℕ := 2
def weeks : ℕ := 2

theorem total_capsules_sold (weekly_earnings_100mg weekly_earnings_500mg 
                             cost_per_capsule_100mg cost_per_capsule_500mg weeks : ℕ) : 
  weekly_earnings_100mg / cost_per_capsule_100mg * weeks + 
  weekly_earnings_500mg / cost_per_capsule_500mg * weeks = 92 :=
by sorry

end NUMINAMATH_CALUDE_total_capsules_sold_l2265_226518


namespace NUMINAMATH_CALUDE_ace_of_hearts_probability_l2265_226577

/-- A standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Number of cards drawn simultaneously -/
def CardsDrawn : ℕ := 2

/-- Number of Ace of Hearts in a standard deck -/
def AceOfHearts : ℕ := 1

/-- Probability of drawing the Ace of Hearts when 2 cards are drawn simultaneously from a standard 52-card deck -/
theorem ace_of_hearts_probability :
  (AceOfHearts * (StandardDeck - CardsDrawn)) / (StandardDeck.choose CardsDrawn) = 1 / 26 := by
  sorry

end NUMINAMATH_CALUDE_ace_of_hearts_probability_l2265_226577


namespace NUMINAMATH_CALUDE_height_difference_l2265_226519

def elm_height : ℚ := 35 / 3
def oak_height : ℚ := 107 / 6

theorem height_difference : oak_height - elm_height = 37 / 6 := by
  sorry

end NUMINAMATH_CALUDE_height_difference_l2265_226519


namespace NUMINAMATH_CALUDE_f_recursion_l2265_226515

/-- A function that computes the sum of binomial coefficients (n choose i) where k divides (n-2i) -/
def f (k : ℕ) (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).sum (λ i => if k ∣ (n - 2*i) then Nat.choose n i else 0)

/-- Theorem stating the recursion relation for f_n -/
theorem f_recursion (k : ℕ) (n : ℕ) (h : k > 1) (h_odd : Odd k) :
  (f k n)^2 = (Finset.range (n + 1)).sum (λ i => Nat.choose n i * f k i * f k (n - i)) := by
  sorry

end NUMINAMATH_CALUDE_f_recursion_l2265_226515


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l2265_226558

theorem cube_root_equation_solution :
  ∃ x : ℝ, (x - 5)^3 = (1/27)⁻¹ ∧ x = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l2265_226558


namespace NUMINAMATH_CALUDE_max_sum_of_squares_l2265_226563

theorem max_sum_of_squares (x y : ℤ) : 3 * x^2 + 5 * y^2 = 345 → (x + y ≤ 13) ∧ ∃ (a b : ℤ), 3 * a^2 + 5 * b^2 = 345 ∧ a + b = 13 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_squares_l2265_226563


namespace NUMINAMATH_CALUDE_class_size_l2265_226539

theorem class_size (hockey : ℕ) (basketball : ℕ) (both : ℕ) (neither : ℕ)
  (h1 : hockey = 15)
  (h2 : basketball = 16)
  (h3 : both = 10)
  (h4 : neither = 4) :
  hockey + basketball - both + neither = 25 := by
  sorry

end NUMINAMATH_CALUDE_class_size_l2265_226539


namespace NUMINAMATH_CALUDE_train_crossing_time_l2265_226530

/-- Proves the time it takes for a train to cross a signal post given its length and the time it takes to cross a bridge -/
theorem train_crossing_time (train_length : ℝ) (bridge_length : ℝ) (bridge_crossing_time : ℝ) :
  train_length = 600 →
  bridge_length = 18000 →
  bridge_crossing_time = 1200 →
  (train_length / (bridge_length / bridge_crossing_time)) = 40 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l2265_226530


namespace NUMINAMATH_CALUDE_janes_age_problem_l2265_226578

theorem janes_age_problem :
  ∃ n : ℕ+, 
    (∃ x : ℕ+, n - 1 = x^3) ∧ 
    (∃ y : ℕ+, n + 4 = y^2) ∧ 
    n = 1332 := by
  sorry

end NUMINAMATH_CALUDE_janes_age_problem_l2265_226578


namespace NUMINAMATH_CALUDE_range_of_linear_function_l2265_226565

def g (c d x : ℝ) : ℝ := c * x + d

theorem range_of_linear_function (c d : ℝ) (hc : c < 0) :
  ∀ y ∈ Set.range (g c d),
    ∃ x ∈ Set.Icc (-1 : ℝ) 1,
      y = g c d x ∧ c + d ≤ y ∧ y ≤ -c + d :=
by sorry

end NUMINAMATH_CALUDE_range_of_linear_function_l2265_226565


namespace NUMINAMATH_CALUDE_fishing_problem_solution_l2265_226510

/-- Represents the fishing problem scenario -/
structure FishingProblem where
  totalCatch : ℝ
  plannedDays : ℝ
  dailyCatch : ℝ
  stormDuration : ℝ
  stormCatchReduction : ℝ
  normalCatchIncrease : ℝ
  daysAheadOfSchedule : ℝ

/-- Theorem stating the solution to the fishing problem -/
theorem fishing_problem_solution (p : FishingProblem) 
  (h1 : p.totalCatch = 1800)
  (h2 : p.stormDuration = p.plannedDays / 3)
  (h3 : p.stormCatchReduction = 20)
  (h4 : p.normalCatchIncrease = 20)
  (h5 : p.daysAheadOfSchedule = 1)
  (h6 : p.plannedDays * p.dailyCatch = p.totalCatch)
  (h7 : p.stormDuration * (p.dailyCatch - p.stormCatchReduction) + 
        (p.plannedDays - p.stormDuration - p.daysAheadOfSchedule) * 
        (p.dailyCatch + p.normalCatchIncrease) = p.totalCatch) :
  p.dailyCatch = 100 := by
  sorry


end NUMINAMATH_CALUDE_fishing_problem_solution_l2265_226510


namespace NUMINAMATH_CALUDE_vector_magnitude_l2265_226527

/-- Given two vectors a and b in ℝ², prove that |2a + b| = 2√21 -/
theorem vector_magnitude (a b : ℝ × ℝ) : 
  a = (3, -4) → 
  ‖b‖ = 2 → 
  a • b = -5 → 
  ‖2 • a + b‖ = 2 * Real.sqrt 21 :=
by sorry

end NUMINAMATH_CALUDE_vector_magnitude_l2265_226527


namespace NUMINAMATH_CALUDE_ellipse_parallelogram_condition_l2265_226541

/-- The condition for an ellipse to have a parallelogram with a vertex on the ellipse, 
    tangent to the unit circle externally, and inscribed in the ellipse for any point on the ellipse -/
theorem ellipse_parallelogram_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  (∀ (x y : ℝ), x^2/a^2 + y^2/b^2 = 1 → 
    ∃ (p q r s : ℝ × ℝ), 
      (p.1^2 + p.2^2 = 1) ∧ 
      (q.1^2 + q.2^2 = 1) ∧ 
      (r.1^2 + r.2^2 = 1) ∧ 
      (s.1^2 + s.2^2 = 1) ∧
      (x^2/a^2 + y^2/b^2 = 1) ∧
      (p.1 - x = s.1 - r.1) ∧ 
      (p.2 - y = s.2 - r.2) ∧
      (q.1 - x = r.1 - s.1) ∧ 
      (q.2 - y = r.2 - s.2)) ↔ 
  (1/a^2 + 1/b^2 = 1) := by
sorry

end NUMINAMATH_CALUDE_ellipse_parallelogram_condition_l2265_226541


namespace NUMINAMATH_CALUDE_distinct_real_pairs_l2265_226595

theorem distinct_real_pairs (x y : ℝ) (hxy : x ≠ y) :
  x^100 - y^100 = 2^99 * (x - y) ∧
  x^200 - y^200 = 2^199 * (x - y) →
  (x = 2 ∧ y = 0) ∨ (x = 0 ∧ y = 2) := by
  sorry

end NUMINAMATH_CALUDE_distinct_real_pairs_l2265_226595


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2265_226528

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ c : ℝ, c > 0 ∧
    c / a = Real.sqrt 6 / 2 ∧
    b * c / Real.sqrt (a^2 + b^2) = 1 ∧
    c^2 = a^2 + b^2) →
  a^2 = 2 ∧ b^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2265_226528
