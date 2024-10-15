import Mathlib

namespace NUMINAMATH_CALUDE_existence_of_special_numbers_l246_24689

theorem existence_of_special_numbers : ∃ (a : Fin 15 → ℕ),
  (∀ i, ∃ k, a i = 35 * k) ∧
  (∀ i j, i ≠ j → ¬(a i ∣ a j)) ∧
  (∀ i j, (a i)^6 ∣ (a j)^5) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_special_numbers_l246_24689


namespace NUMINAMATH_CALUDE_percentage_women_non_union_l246_24620

-- Define the total number of employees
variable (E : ℝ)
-- Assume E is positive
variable (hE : E > 0)

-- Define the percentage of unionized employees
def unionized_percent : ℝ := 0.60

-- Define the percentage of men among unionized employees
def men_in_union_percent : ℝ := 0.70

-- Define the percentage of women among non-union employees
def women_non_union_percent : ℝ := 0.65

-- Theorem to prove
theorem percentage_women_non_union :
  women_non_union_percent = 0.65 := by
  sorry

end NUMINAMATH_CALUDE_percentage_women_non_union_l246_24620


namespace NUMINAMATH_CALUDE_area_of_triangle_formed_by_tangents_l246_24639

/-- Given two circles with radii R and r, where their common internal tangents
    are perpendicular to each other, the area of the triangle formed by these
    tangents and the common external tangent is equal to R * r. -/
theorem area_of_triangle_formed_by_tangents (R r : ℝ) (R_pos : R > 0) (r_pos : r > 0) :
  ∃ (S : ℝ), S = R * r ∧ S > 0 :=
by sorry

end NUMINAMATH_CALUDE_area_of_triangle_formed_by_tangents_l246_24639


namespace NUMINAMATH_CALUDE_two_number_difference_l246_24688

theorem two_number_difference (x y : ℝ) : 
  x + y = 40 → 3 * y - 4 * x = 10 → |y - x| = 60 / 7 := by
  sorry

end NUMINAMATH_CALUDE_two_number_difference_l246_24688


namespace NUMINAMATH_CALUDE_consecutive_even_sum_46_l246_24605

theorem consecutive_even_sum_46 (n m : ℤ) : 
  (Even n) → (Even m) → (m = n + 2) → (n + m = 46) → (m = 24) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_even_sum_46_l246_24605


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l246_24600

theorem least_subtraction_for_divisibility : ∃! n : ℕ, n ≤ 12 ∧ (427398 - n) % 13 = 0 ∧ ∀ m : ℕ, m < n → (427398 - m) % 13 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l246_24600


namespace NUMINAMATH_CALUDE_compound_interest_10_years_l246_24643

/-- Calculates the total amount of principal and interest after a given number of years
    with compound interest. -/
def compoundInterest (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate) ^ years

/-- Theorem stating that the total amount after 10 years of compound interest
    is equal to the initial principal multiplied by (1 + rate) raised to the power of 10. -/
theorem compound_interest_10_years
  (a : ℝ) -- initial deposit
  (r : ℝ) -- annual interest rate
  (h1 : a > 0) -- assumption that initial deposit is positive
  (h2 : r > 0) -- assumption that interest rate is positive
  : compoundInterest a r 10 = a * (1 + r)^10 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_10_years_l246_24643


namespace NUMINAMATH_CALUDE_value_of_a_minus_2b_l246_24657

theorem value_of_a_minus_2b (a b : ℝ) (h : |a + b + 2| + |b - 3| = 0) : a - 2*b = -11 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_minus_2b_l246_24657


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l246_24616

/-- 
Proves that in an arithmetic sequence with given conditions, 
the common difference is 3.
-/
theorem arithmetic_sequence_common_difference 
  (a : ℝ) (an : ℝ) (n : ℕ) (sum : ℝ) :
  a = 2 →
  an = 50 →
  sum = 442 →
  an = a + (n - 1) * (3 : ℝ) →
  sum = (n / 2) * (a + an) →
  (3 : ℝ) = (an - a) / (n - 1) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l246_24616


namespace NUMINAMATH_CALUDE_largest_lucky_number_is_499_l246_24612

def lucky_number (a b : ℕ) : ℕ := a + b + a * b

def largest_lucky_number_after_three_operations : ℕ :=
  let n1 := lucky_number 1 4
  let n2 := lucky_number 4 n1
  let n3 := lucky_number n1 n2
  max n1 (max n2 n3)

theorem largest_lucky_number_is_499 :
  largest_lucky_number_after_three_operations = 499 := by sorry

end NUMINAMATH_CALUDE_largest_lucky_number_is_499_l246_24612


namespace NUMINAMATH_CALUDE_running_yardage_difference_l246_24611

def player_yardage (total_yards pass_yards : ℕ) : ℕ :=
  total_yards - pass_yards

theorem running_yardage_difference (
  player_a_total player_a_pass player_b_total player_b_pass : ℕ
) (h1 : player_a_total = 150)
  (h2 : player_a_pass = 60)
  (h3 : player_b_total = 180)
  (h4 : player_b_pass = 80) :
  (player_yardage player_a_total player_a_pass : ℤ) - 
  (player_yardage player_b_total player_b_pass : ℤ) = -10 :=
by
  sorry

end NUMINAMATH_CALUDE_running_yardage_difference_l246_24611


namespace NUMINAMATH_CALUDE_quadratic_polynomial_existence_l246_24669

/-- A quadratic polynomial with real coefficients -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Evaluate a quadratic polynomial at a complex number -/
def evaluate (p : QuadraticPolynomial) (z : ℂ) : ℂ :=
  p.a * z^2 + p.b * z + p.c

theorem quadratic_polynomial_existence : ∃ (p : QuadraticPolynomial),
  (evaluate p (-3 - 4*I) = 0) ∧ 
  (p.b = -10) ∧
  (p.a = -5/3) ∧ 
  (p.c = -125/3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_existence_l246_24669


namespace NUMINAMATH_CALUDE_tan_theta_right_triangle_l246_24624

theorem tan_theta_right_triangle (BC AC BA : ℝ) (h1 : BC = 25) (h2 : AC = 20) 
  (h3 : BA^2 + AC^2 = BC^2) : 
  Real.tan (Real.arcsin (BA / BC)) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_right_triangle_l246_24624


namespace NUMINAMATH_CALUDE_discount_age_limit_l246_24648

/-- Represents the age limit for the discount at an amusement park. -/
def age_limit : ℕ := 10

/-- Represents the regular ticket cost. -/
def regular_ticket_cost : ℕ := 109

/-- Represents the discount amount for children. -/
def child_discount : ℕ := 5

/-- Represents the number of adults in the family. -/
def num_adults : ℕ := 2

/-- Represents the number of children in the family. -/
def num_children : ℕ := 2

/-- Represents the ages of the children in the family. -/
def children_ages : List ℕ := [6, 10]

/-- Represents the amount paid by the family. -/
def amount_paid : ℕ := 500

/-- Represents the change received by the family. -/
def change_received : ℕ := 74

/-- Theorem stating that the age limit for the discount is 10 years old. -/
theorem discount_age_limit : 
  (∀ (age : ℕ), age ∈ children_ages → age ≤ age_limit) ∧
  (amount_paid - change_received = 
    num_adults * regular_ticket_cost + 
    num_children * (regular_ticket_cost - child_discount)) →
  age_limit = 10 := by
  sorry

end NUMINAMATH_CALUDE_discount_age_limit_l246_24648


namespace NUMINAMATH_CALUDE_egyptian_fraction_for_odd_n_l246_24632

theorem egyptian_fraction_for_odd_n (n : ℕ) 
  (h_odd : Odd n) 
  (h_gt3 : n > 3) 
  (h_not_div3 : ¬(3 ∣ n)) : 
  ∃ (a b c : ℕ), 
    Odd a ∧ Odd b ∧ Odd c ∧ 
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    (3 : ℚ) / n = 1 / a + 1 / b + 1 / c := by
  sorry

end NUMINAMATH_CALUDE_egyptian_fraction_for_odd_n_l246_24632


namespace NUMINAMATH_CALUDE_urn_probability_l246_24609

/-- Represents the state of the urn -/
structure UrnState :=
  (red : ℕ)
  (blue : ℕ)

/-- Represents one operation on the urn -/
inductive Operation
  | Red
  | Blue

/-- Calculates the probability of a specific sequence of operations -/
def sequenceProbability (ops : List Operation) : ℚ :=
  sorry

/-- Calculates the number of sequences with 3 red and 2 blue operations -/
def validSequences : ℕ :=
  sorry

/-- The main theorem stating the probability of having 4 balls of each color -/
theorem urn_probability : 
  let initialState : UrnState := ⟨1, 1⟩
  let finalState : UrnState := ⟨4, 4⟩
  let totalOperations : ℕ := 5
  let probability : ℚ := (validSequences : ℚ) * sequenceProbability (List.replicate 3 Operation.Red ++ List.replicate 2 Operation.Blue)
  probability = 1 / 6 :=
sorry

end NUMINAMATH_CALUDE_urn_probability_l246_24609


namespace NUMINAMATH_CALUDE_square_sum_from_difference_and_product_l246_24604

theorem square_sum_from_difference_and_product (x y : ℝ) 
  (h1 : x - y = 10) 
  (h2 : x * y = 9) : 
  x^2 + y^2 = 118 := by
sorry

end NUMINAMATH_CALUDE_square_sum_from_difference_and_product_l246_24604


namespace NUMINAMATH_CALUDE_sqrt_1600_minus_24_form_l246_24630

theorem sqrt_1600_minus_24_form (a b : ℕ+) :
  (Real.sqrt 1600 - 24 : ℝ) = ((Real.sqrt a.val - b.val) : ℝ)^2 →
  a.val + b.val = 102 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_1600_minus_24_form_l246_24630


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l246_24661

theorem arithmetic_calculation : (-1) * (-3) + 3^2 / (8 - 5) = 6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l246_24661


namespace NUMINAMATH_CALUDE_seashell_ratio_l246_24621

theorem seashell_ratio : 
  let henry_shells : ℕ := 11
  let paul_shells : ℕ := 24
  let initial_total : ℕ := 59
  let final_total : ℕ := 53
  let leo_initial : ℕ := initial_total - henry_shells - paul_shells
  let leo_gave_away : ℕ := initial_total - final_total
  (leo_gave_away : ℚ) / leo_initial = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_seashell_ratio_l246_24621


namespace NUMINAMATH_CALUDE_sum_of_integers_l246_24667

theorem sum_of_integers (p q r s : ℤ) 
  (eq1 : p - q + r = 7)
  (eq2 : q - r + s = 8)
  (eq3 : r - s + p = 4)
  (eq4 : s - p + q = 1) :
  p + q + r + s = 20 := by sorry

end NUMINAMATH_CALUDE_sum_of_integers_l246_24667


namespace NUMINAMATH_CALUDE_planet_surface_area_unchanged_l246_24637

theorem planet_surface_area_unchanged 
  (planet_diameter : ℝ) 
  (explosion_radius : ℝ) 
  (h1 : planet_diameter = 10000) 
  (h2 : explosion_radius = 5000) :
  let planet_radius : ℝ := planet_diameter / 2
  let initial_surface_area : ℝ := 4 * Real.pi * planet_radius ^ 2
  let new_surface_area : ℝ := initial_surface_area
  new_surface_area = 100000000 * Real.pi := by sorry

end NUMINAMATH_CALUDE_planet_surface_area_unchanged_l246_24637


namespace NUMINAMATH_CALUDE_triangle_inequality_l246_24646

theorem triangle_inequality (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) (h7 : a + b + c = 2) :
  a^2 + b^2 + c^2 < 2*(1 - a*b*c) := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l246_24646


namespace NUMINAMATH_CALUDE_triangle_inequality_l246_24650

theorem triangle_inequality (a b c : ℝ) 
  (triangle_sides : a > 0 ∧ b > 0 ∧ c > 0)
  (triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b)
  (perimeter : a + b + c = 2) :
  a^2 + b^2 + c^2 < 2*(1 - a*b*c) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l246_24650


namespace NUMINAMATH_CALUDE_smallest_prime_ten_less_than_perfect_square_l246_24680

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem smallest_prime_ten_less_than_perfect_square :
  (∀ p : ℕ, p < 71 → (is_prime p → ¬∃ n : ℕ, is_perfect_square (p + 10))) ∧
  (is_prime 71 ∧ ∃ n : ℕ, is_perfect_square (71 + 10)) :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_ten_less_than_perfect_square_l246_24680


namespace NUMINAMATH_CALUDE_num_factors_41040_eq_80_l246_24699

/-- The number of positive factors of 41040 -/
def num_factors_41040 : ℕ :=
  (Finset.filter (· ∣ 41040) (Finset.range 41041)).card

/-- Theorem stating that the number of positive factors of 41040 is 80 -/
theorem num_factors_41040_eq_80 : num_factors_41040 = 80 := by
  sorry

end NUMINAMATH_CALUDE_num_factors_41040_eq_80_l246_24699


namespace NUMINAMATH_CALUDE_greatest_digit_sum_base_nine_l246_24693

/-- 
Given a positive integer n less than 5000, returns the sum of digits
in its base-nine representation.
-/
def sumOfDigitsBaseNine (n : ℕ) : ℕ := sorry

/-- 
The greatest possible sum of the digits in the base-nine representation
of a positive integer less than 5000.
-/
def maxDigitSum : ℕ := 26

theorem greatest_digit_sum_base_nine :
  ∀ n : ℕ, n < 5000 → sumOfDigitsBaseNine n ≤ maxDigitSum :=
sorry

end NUMINAMATH_CALUDE_greatest_digit_sum_base_nine_l246_24693


namespace NUMINAMATH_CALUDE_parabola_directrix_l246_24601

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 8*y

-- Define the directrix
def directrix (y : ℝ) : Prop := y = -2

-- Theorem statement
theorem parabola_directrix : 
  ∀ (x y : ℝ), parabola x y → directrix y :=
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l246_24601


namespace NUMINAMATH_CALUDE_geometric_arithmetic_progression_ratio_l246_24660

/-- Given a decreasing geometric progression a, b, c with common ratio q,
    if 19a, 124b/13, c/13 form an arithmetic progression, then q = 247. -/
theorem geometric_arithmetic_progression_ratio 
  (a b c : ℝ) (q : ℝ) (h_pos : a > 0) (h_decr : q > 1) :
  b = a * q ∧ c = a * q^2 ∧ 
  2 * (124 * b / 13) = 19 * a + c / 13 →
  q = 247 := by
sorry


end NUMINAMATH_CALUDE_geometric_arithmetic_progression_ratio_l246_24660


namespace NUMINAMATH_CALUDE_range_of_a_l246_24668

theorem range_of_a (a : ℝ) : 
  let M : Set ℝ := {a}
  let P : Set ℝ := {x | -1 < x ∧ x < 1}
  M ⊆ P → a ∈ P := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l246_24668


namespace NUMINAMATH_CALUDE_green_balls_count_l246_24653

theorem green_balls_count (blue_count : ℕ) (ratio_blue : ℕ) (ratio_green : ℕ) (green_count : ℕ) : 
  blue_count = 20 → 
  ratio_blue = 5 → 
  ratio_green = 3 → 
  blue_count * ratio_green = green_count * ratio_blue → 
  green_count = 12 := by
sorry

end NUMINAMATH_CALUDE_green_balls_count_l246_24653


namespace NUMINAMATH_CALUDE_birthday_stickers_proof_l246_24671

/-- The number of stickers James had initially -/
def initial_stickers : ℕ := 39

/-- The number of stickers James had after his birthday -/
def final_stickers : ℕ := 61

/-- The number of stickers James got for his birthday -/
def birthday_stickers : ℕ := final_stickers - initial_stickers

theorem birthday_stickers_proof :
  birthday_stickers = final_stickers - initial_stickers :=
by sorry

end NUMINAMATH_CALUDE_birthday_stickers_proof_l246_24671


namespace NUMINAMATH_CALUDE_system_equation_ratio_l246_24649

theorem system_equation_ratio (x y a b : ℝ) : 
  x ≠ 0 → 
  y ≠ 0 → 
  b ≠ 0 → 
  8 * x - 6 * y = a → 
  12 * y - 18 * x = b → 
  a / b = -1 / 2 := by
sorry

end NUMINAMATH_CALUDE_system_equation_ratio_l246_24649


namespace NUMINAMATH_CALUDE_steve_take_home_pay_l246_24670

/-- Calculates the take-home pay given annual salary and deductions --/
def takeHomePay (annualSalary : ℝ) (taxRate : ℝ) (healthcareRate : ℝ) (unionDues : ℝ) : ℝ :=
  annualSalary - (annualSalary * taxRate + annualSalary * healthcareRate + unionDues)

/-- Proves that Steve's take-home pay is $27,200 --/
theorem steve_take_home_pay :
  takeHomePay 40000 0.20 0.10 800 = 27200 := by
  sorry

end NUMINAMATH_CALUDE_steve_take_home_pay_l246_24670


namespace NUMINAMATH_CALUDE_trig_problem_l246_24681

theorem trig_problem (A : Real) (h1 : 0 < A ∧ A < π) (h2 : Real.sin A + Real.cos A = 1/5) : 
  Real.sin A * Real.cos A = -12/25 ∧ Real.tan A = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_trig_problem_l246_24681


namespace NUMINAMATH_CALUDE_fixed_points_subset_stable_points_l246_24697

-- Define the function type
def RealFunction := ℝ → ℝ

-- Define the set of fixed points
def FixedPoints (f : RealFunction) : Set ℝ :=
  {x : ℝ | f x = x}

-- Define the set of stable points
def StablePoints (f : RealFunction) : Set ℝ :=
  {x : ℝ | f (f x) = x}

-- Theorem statement
theorem fixed_points_subset_stable_points (f : RealFunction) :
  FixedPoints f ⊆ StablePoints f := by
  sorry


end NUMINAMATH_CALUDE_fixed_points_subset_stable_points_l246_24697


namespace NUMINAMATH_CALUDE_square_root_squared_specific_case_l246_24615

theorem square_root_squared (n : ℝ) (h : 0 ≤ n) : (Real.sqrt n)^2 = n := by sorry

theorem specific_case : (Real.sqrt 987654)^2 = 987654 := by sorry

end NUMINAMATH_CALUDE_square_root_squared_specific_case_l246_24615


namespace NUMINAMATH_CALUDE_two_p_plus_q_l246_24613

theorem two_p_plus_q (p q : ℚ) (h : p / q = 5 / 4) : 2 * p + q = 7 * q / 2 := by
  sorry

end NUMINAMATH_CALUDE_two_p_plus_q_l246_24613


namespace NUMINAMATH_CALUDE_largest_quantity_l246_24626

def X : ℚ := 2010 / 2009 + 2010 / 2011
def Y : ℚ := 2010 / 2011 + 2012 / 2011
def Z : ℚ := 2011 / 2010 + 2011 / 2012

theorem largest_quantity : X > Y ∧ X > Z := by
  sorry

end NUMINAMATH_CALUDE_largest_quantity_l246_24626


namespace NUMINAMATH_CALUDE_sum_and_product_implications_l246_24633

theorem sum_and_product_implications (a b : ℝ) 
  (h1 : a + b = 2) (h2 : a * b = -1) : 
  a^2 + b^2 = 6 ∧ (a - b)^2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_product_implications_l246_24633


namespace NUMINAMATH_CALUDE_lcm_sum_bound_l246_24674

theorem lcm_sum_bound (a b c d e : ℕ) (h1 : a > b) (h2 : b > c) (h3 : c > d) (h4 : d > e) (h5 : e > 1) :
  (1 : ℚ) / Nat.lcm a b + (1 : ℚ) / Nat.lcm b c + (1 : ℚ) / Nat.lcm c d + (1 : ℚ) / Nat.lcm d e ≤ 15 / 16 := by
  sorry

end NUMINAMATH_CALUDE_lcm_sum_bound_l246_24674


namespace NUMINAMATH_CALUDE_jacket_price_reduction_l246_24638

theorem jacket_price_reduction (P : ℝ) (x : ℝ) : 
  P > 0 → 
  0 ≤ x → x ≤ 100 →
  P * (1 - x / 100) * (1 - 0.25) * (1 + 0.7778) = P →
  x = 25 := by
sorry

end NUMINAMATH_CALUDE_jacket_price_reduction_l246_24638


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_expression_l246_24635

theorem largest_prime_factor_of_expression : 
  let expr := 15^4 + 2*15^2 + 1 - 14^4
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ expr ∧ p = 211 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ expr → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_expression_l246_24635


namespace NUMINAMATH_CALUDE_units_digit_of_m_squared_plus_two_to_m_l246_24677

def m : ℕ := 2016^2 + 2^2016

theorem units_digit_of_m_squared_plus_two_to_m (m : ℕ) : 
  m = 2016^2 + 2^2016 → (m^2 + 2^m) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_m_squared_plus_two_to_m_l246_24677


namespace NUMINAMATH_CALUDE_shaded_area_in_divided_square_l246_24662

/-- The area of shaded regions in a square with specific divisions -/
theorem shaded_area_in_divided_square (side_length : ℝ) (h_side : side_length = 4) :
  let square_area := side_length ^ 2
  let num_rectangles := 4
  let num_triangles_per_rectangle := 2
  let num_shaded_triangles := num_rectangles
  let rectangle_area := square_area / num_rectangles
  let triangle_area := rectangle_area / num_triangles_per_rectangle
  let total_shaded_area := num_shaded_triangles * triangle_area
  total_shaded_area = 8 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_in_divided_square_l246_24662


namespace NUMINAMATH_CALUDE_phase_without_chromatids_is_interkinesis_l246_24682

-- Define the phases of meiosis
inductive MeiosisPhase
  | prophaseI
  | interkinesis
  | prophaseII
  | lateProphaseII

-- Define a property for the presence of chromatids
def hasChromatids (phase : MeiosisPhase) : Prop :=
  match phase with
  | MeiosisPhase.interkinesis => False
  | _ => True

-- Define a property for DNA replication
def hasDNAReplication (phase : MeiosisPhase) : Prop :=
  match phase with
  | MeiosisPhase.interkinesis => False
  | _ => True

-- Theorem statement
theorem phase_without_chromatids_is_interkinesis :
  ∀ phase : MeiosisPhase, ¬(hasChromatids phase) → phase = MeiosisPhase.interkinesis :=
by
  sorry

end NUMINAMATH_CALUDE_phase_without_chromatids_is_interkinesis_l246_24682


namespace NUMINAMATH_CALUDE_right_trapezoid_diagonals_bases_squares_diff_l246_24641

/-- A right trapezoid with given properties -/
structure RightTrapezoid where
  b₁ : ℝ  -- length of smaller base BC
  b₂ : ℝ  -- length of larger base AD
  h : ℝ   -- height (length of legs AB and CD)
  h_pos : h > 0
  b₁_pos : b₁ > 0
  b₂_pos : b₂ > 0
  b₁_lt_b₂ : b₁ < b₂

/-- The theorem stating that the difference of squares of diagonals equals
    the difference of squares of bases in a right trapezoid -/
theorem right_trapezoid_diagonals_bases_squares_diff
  (t : RightTrapezoid) :
  (t.h^2 + t.b₂^2) - (t.h^2 + t.b₁^2) = t.b₂^2 - t.b₁^2 := by
  sorry

end NUMINAMATH_CALUDE_right_trapezoid_diagonals_bases_squares_diff_l246_24641


namespace NUMINAMATH_CALUDE_max_production_in_seven_days_l246_24685

/-- Represents the daily production capacity of a group -/
structure ProductionCapacity where
  shirts : ℕ
  trousers : ℕ

/-- Represents the production assignment for a group -/
structure ProductionAssignment where
  shirtDays : ℕ
  trouserDays : ℕ

/-- Calculates the total production of a group given its capacity and assignment -/
def totalProduction (capacity : ProductionCapacity) (assignment : ProductionAssignment) : ℕ × ℕ :=
  (capacity.shirts * assignment.shirtDays, capacity.trousers * assignment.trouserDays)

/-- Theorem: Maximum production of matching sets in 7 days -/
theorem max_production_in_seven_days 
  (groupA groupB groupC groupD : ProductionCapacity)
  (assignmentA assignmentB assignmentC assignmentD : ProductionAssignment)
  (h1 : assignmentA.shirtDays + assignmentA.trouserDays = 7)
  (h2 : assignmentB.shirtDays + assignmentB.trouserDays = 7)
  (h3 : assignmentC.shirtDays + assignmentC.trouserDays = 7)
  (h4 : assignmentD.shirtDays + assignmentD.trouserDays = 7)
  (h5 : groupA.shirts = 8 ∧ groupA.trousers = 10)
  (h6 : groupB.shirts = 9 ∧ groupB.trousers = 12)
  (h7 : groupC.shirts = 7 ∧ groupC.trousers = 11)
  (h8 : groupD.shirts = 6 ∧ groupD.trousers = 7) :
  (∃ (assignmentA assignmentB assignmentC assignmentD : ProductionAssignment),
    let (shirtsTotalA, trousersTotalA) := totalProduction groupA assignmentA
    let (shirtsTotalB, trousersTotalB) := totalProduction groupB assignmentB
    let (shirtsTotalC, trousersTotalC) := totalProduction groupC assignmentC
    let (shirtsTotalD, trousersTotalD) := totalProduction groupD assignmentD
    let shirtsTotal := shirtsTotalA + shirtsTotalB + shirtsTotalC + shirtsTotalD
    let trousersTotal := trousersTotalA + trousersTotalB + trousersTotalC + trousersTotalD
    min shirtsTotal trousersTotal = 125) :=
by sorry

end NUMINAMATH_CALUDE_max_production_in_seven_days_l246_24685


namespace NUMINAMATH_CALUDE_box_velvet_problem_l246_24656

theorem box_velvet_problem (long_side_length long_side_width short_side_length short_side_width total_velvet : ℕ) 
  (h1 : long_side_length = 8)
  (h2 : long_side_width = 6)
  (h3 : short_side_length = 5)
  (h4 : short_side_width = 6)
  (h5 : total_velvet = 236) :
  let side_area := 2 * (long_side_length * long_side_width + short_side_length * short_side_width)
  let remaining_area := total_velvet - side_area
  (remaining_area / 2 : ℕ) = 40 := by
  sorry

end NUMINAMATH_CALUDE_box_velvet_problem_l246_24656


namespace NUMINAMATH_CALUDE_garage_sale_pricing_l246_24683

theorem garage_sale_pricing (total_items : ℕ) (highest_rank : ℕ) (lowest_rank : ℕ) 
  (h1 : total_items = 36)
  (h2 : highest_rank = 15)
  (h3 : lowest_rank + highest_rank = total_items + 1) : 
  lowest_rank = 22 := by
sorry

end NUMINAMATH_CALUDE_garage_sale_pricing_l246_24683


namespace NUMINAMATH_CALUDE_tan_sum_special_l246_24627

theorem tan_sum_special : Real.tan (10 * π / 180) + Real.tan (50 * π / 180) + Real.sqrt 3 * Real.tan (10 * π / 180) * Real.tan (50 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_special_l246_24627


namespace NUMINAMATH_CALUDE_complex_expression_evaluation_l246_24608

theorem complex_expression_evaluation : 
  let expr := (((32400 * 4^3) / (3 * Real.sqrt 343)) / 18 / (7^3 * 10)) / 
              ((2 * Real.sqrt ((49^2 * 11)^4)) / 25^3)
  ∃ ε > 0, abs (expr - 0.00005366) < ε := by
sorry

end NUMINAMATH_CALUDE_complex_expression_evaluation_l246_24608


namespace NUMINAMATH_CALUDE_complex_power_4_l246_24698

theorem complex_power_4 : 
  (3 * (Complex.cos (30 * π / 180) + Complex.I * Complex.sin (30 * π / 180)))^4 = 
  Complex.ofReal (-40.5) + Complex.I * Complex.ofReal (40.5 * Real.sqrt 3) := by
sorry

end NUMINAMATH_CALUDE_complex_power_4_l246_24698


namespace NUMINAMATH_CALUDE_toby_breakfast_pb_servings_l246_24664

/-- Calculates the number of peanut butter servings needed for a target calorie count -/
def peanut_butter_servings (target_calories : ℕ) (bread_calories : ℕ) (pb_calories_per_serving : ℕ) : ℕ :=
  ((target_calories - bread_calories) / pb_calories_per_serving)

/-- Proves that 2 servings of peanut butter are needed for Toby's breakfast -/
theorem toby_breakfast_pb_servings :
  peanut_butter_servings 500 100 200 = 2 := by
  sorry

#eval peanut_butter_servings 500 100 200

end NUMINAMATH_CALUDE_toby_breakfast_pb_servings_l246_24664


namespace NUMINAMATH_CALUDE_number_of_observations_l246_24690

theorem number_of_observations (initial_mean : ℝ) (wrong_obs : ℝ) (correct_obs : ℝ) (new_mean : ℝ)
  (h1 : initial_mean = 36)
  (h2 : wrong_obs = 23)
  (h3 : correct_obs = 45)
  (h4 : new_mean = 36.5) :
  ∃ (n : ℕ), n * initial_mean + (correct_obs - wrong_obs) = n * new_mean ∧ n = 44 := by
sorry

end NUMINAMATH_CALUDE_number_of_observations_l246_24690


namespace NUMINAMATH_CALUDE_total_viewing_time_l246_24696

/-- The viewing times for the original animal types -/
def original_times : List Nat := [4, 6, 7, 5, 9]

/-- The viewing times for the new animal types -/
def new_times : List Nat := [3, 7, 8, 10]

/-- The total number of animal types -/
def total_types : Nat := original_times.length + new_times.length

theorem total_viewing_time :
  (List.sum original_times) + (List.sum new_times) = 59 := by
  sorry

end NUMINAMATH_CALUDE_total_viewing_time_l246_24696


namespace NUMINAMATH_CALUDE_paint_cost_per_quart_l246_24663

/-- The cost of paint per quart given specific conditions -/
theorem paint_cost_per_quart : 
  ∀ (coverage_per_quart : ℝ) (cube_edge_length : ℝ) (cost_to_paint_cube : ℝ),
  coverage_per_quart = 1200 →
  cube_edge_length = 10 →
  cost_to_paint_cube = 1.6 →
  (∃ (cost_per_quart : ℝ),
    cost_per_quart = 3.2 ∧
    cost_per_quart * (6 * cube_edge_length^2 / coverage_per_quart) = cost_to_paint_cube) :=
by sorry


end NUMINAMATH_CALUDE_paint_cost_per_quart_l246_24663


namespace NUMINAMATH_CALUDE_train_speed_calculation_l246_24691

/-- Proves that a train with given length, crossing a platform of given length in a specific time, has a specific speed. -/
theorem train_speed_calculation (train_length platform_length : ℝ) (crossing_time : ℝ) :
  train_length = 480 ∧ 
  platform_length = 620 ∧ 
  crossing_time = 71.99424046076314 →
  ∃ (speed : ℝ), abs (speed - 54.964) < 0.001 ∧ 
  speed = (train_length + platform_length) / crossing_time * 3.6 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l246_24691


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_of_quadratic_roots_l246_24655

theorem sum_of_reciprocals_of_quadratic_roots :
  ∀ (p q : ℝ), 
    p^2 - 11*p + 6 = 0 →
    q^2 - 11*q + 6 = 0 →
    p ≠ 0 →
    q ≠ 0 →
    1/p + 1/q = 11/6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_of_quadratic_roots_l246_24655


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l246_24614

-- Define the quadratic function
def f (a b x : ℝ) := a * x^2 - x - b

-- Define the solution set condition
def solution_set_condition (a b : ℝ) :=
  ∀ x, f a b x > 0 ↔ (x > 2 ∨ x < -1)

-- Theorem statement
theorem quadratic_inequality_solution :
  ∀ a b : ℝ, solution_set_condition a b →
    (a = 1 ∧ b = 2) ∧
    (∀ c : ℝ,
      (c > 1 → ∀ x, x^2 - (c+1)*x + c < 0 ↔ 1 < x ∧ x < c) ∧
      (c = 1 → ∀ x, ¬(x^2 - (c+1)*x + c < 0)) ∧
      (c < 1 → ∀ x, x^2 - (c+1)*x + c < 0 ↔ c < x ∧ x < 1)) :=
by sorry


end NUMINAMATH_CALUDE_quadratic_inequality_solution_l246_24614


namespace NUMINAMATH_CALUDE_line_l_prime_equation_l246_24645

-- Define the fixed point P
def P : ℝ × ℝ := (-1, 1)

-- Define the direction vector of line l'
def direction_vector : ℝ × ℝ := (3, 2)

-- Define the equation of line l
def line_l (m : ℝ) (x y : ℝ) : Prop :=
  (2 * m + 1) * x + (m + 1) * y + m = 0

-- State the theorem
theorem line_l_prime_equation :
  ∀ (m : ℝ),
  (∃ (x y : ℝ), line_l m x y ∧ (x, y) = P) →
  (∃ (k : ℝ), 2 * P.1 - 3 * P.2 + 5 = 0 ∧
              ∀ (t : ℝ), 2 * (P.1 + t * direction_vector.1) - 3 * (P.2 + t * direction_vector.2) + 5 = 0) :=
by sorry

end NUMINAMATH_CALUDE_line_l_prime_equation_l246_24645


namespace NUMINAMATH_CALUDE_tim_gave_six_kittens_to_sara_l246_24618

/-- The number of kittens Tim gave to Sara -/
def kittens_to_sara (initial_kittens : ℕ) (kittens_to_jessica : ℕ) (remaining_kittens : ℕ) : ℕ :=
  initial_kittens - kittens_to_jessica - remaining_kittens

/-- Proof that Tim gave 6 kittens to Sara -/
theorem tim_gave_six_kittens_to_sara :
  kittens_to_sara 18 3 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_tim_gave_six_kittens_to_sara_l246_24618


namespace NUMINAMATH_CALUDE_cookies_remaining_cookies_remaining_result_l246_24654

/-- Calculates the number of cookies remaining in Cristian's jar --/
theorem cookies_remaining (initial_white : ℕ) (black_white_diff : ℕ) : ℕ :=
  let initial_black := initial_white + black_white_diff
  let remaining_white := initial_white - (3 * initial_white / 4)
  let remaining_black := initial_black - (initial_black / 2)
  remaining_white + remaining_black

/-- Proves that the number of cookies remaining is 85 --/
theorem cookies_remaining_result : cookies_remaining 80 50 = 85 := by
  sorry

end NUMINAMATH_CALUDE_cookies_remaining_cookies_remaining_result_l246_24654


namespace NUMINAMATH_CALUDE_winning_game_score_is_3_0_l246_24694

structure FootballTeam where
  games_played : ℕ
  total_goals_scored : ℕ
  total_goals_conceded : ℕ
  wins : ℕ
  draws : ℕ
  losses : ℕ

def winning_game_score (team : FootballTeam) : ℕ × ℕ := sorry

theorem winning_game_score_is_3_0 (team : FootballTeam) 
  (h1 : team.games_played = 3)
  (h2 : team.total_goals_scored = 3)
  (h3 : team.total_goals_conceded = 1)
  (h4 : team.wins = 1)
  (h5 : team.draws = 1)
  (h6 : team.losses = 1) :
  winning_game_score team = (3, 0) := by sorry

end NUMINAMATH_CALUDE_winning_game_score_is_3_0_l246_24694


namespace NUMINAMATH_CALUDE_store_price_difference_l246_24640

/-- Given the total price and quantity of shirts and sweaters, prove that the difference
    between the average price of a sweater and the average price of a shirt is $2. -/
theorem store_price_difference (shirt_price shirt_quantity sweater_price sweater_quantity : ℕ) 
  (h1 : shirt_price = 360)
  (h2 : shirt_quantity = 20)
  (h3 : sweater_price = 900)
  (h4 : sweater_quantity = 45) :
  (sweater_price / sweater_quantity : ℚ) - (shirt_price / shirt_quantity : ℚ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_store_price_difference_l246_24640


namespace NUMINAMATH_CALUDE_solve_for_z_l246_24607

theorem solve_for_z (m : ℕ) (z : ℝ) 
  (h1 : ((1 ^ (m + 1)) / (5 ^ (m + 1))) * ((1 ^ 18) / (z ^ 18)) = 1 / (2 * (10 ^ 35)))
  (h2 : m = 34) : 
  z = 4 := by
sorry

end NUMINAMATH_CALUDE_solve_for_z_l246_24607


namespace NUMINAMATH_CALUDE_correct_operation_l246_24684

theorem correct_operation (a : ℝ) : 2 * a^2 * a = 2 * a^3 := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l246_24684


namespace NUMINAMATH_CALUDE_oldest_sibling_age_l246_24652

/-- Represents the ages and relationships in Kay's family --/
structure KayFamily where
  kay_age : ℕ
  num_siblings : ℕ
  youngest_sibling_age : ℕ
  oldest_sibling_age : ℕ

/-- The conditions given in the problem --/
def kay_family_conditions (f : KayFamily) : Prop :=
  f.kay_age = 32 ∧
  f.num_siblings = 14 ∧
  f.youngest_sibling_age = f.kay_age / 2 - 5 ∧
  f.oldest_sibling_age = 4 * f.youngest_sibling_age

/-- Theorem stating that the oldest sibling's age is 44 given the conditions --/
theorem oldest_sibling_age (f : KayFamily) 
  (h : kay_family_conditions f) : f.oldest_sibling_age = 44 := by
  sorry


end NUMINAMATH_CALUDE_oldest_sibling_age_l246_24652


namespace NUMINAMATH_CALUDE_blackboard_numbers_l246_24629

def blackboard_rule (a b : ℕ) : ℕ := a * b + a + b

def is_generable (n : ℕ) : Prop :=
  ∃ k m : ℕ, n = 2^k * 3^m - 1

theorem blackboard_numbers :
  (is_generable 13121) ∧ (¬ is_generable 12131) := by sorry

end NUMINAMATH_CALUDE_blackboard_numbers_l246_24629


namespace NUMINAMATH_CALUDE_max_negative_integers_l246_24666

theorem max_negative_integers
  (a b c d e f : ℤ)
  (h : a * b + c * d * e * f < 0) :
  ∃ (neg_count : ℕ),
    neg_count ≤ 4 ∧
    ∀ (n : ℕ),
      (∃ (neg_set : Finset ℤ),
        neg_set.card = n ∧
        neg_set ⊆ {a, b, c, d, e, f} ∧
        (∀ x ∈ neg_set, x < 0) ∧
        (∀ x ∈ {a, b, c, d, e, f} \ neg_set, x ≥ 0)) →
      n ≤ neg_count :=
sorry

end NUMINAMATH_CALUDE_max_negative_integers_l246_24666


namespace NUMINAMATH_CALUDE_no_real_solution_log_equation_l246_24658

theorem no_real_solution_log_equation :
  ¬∃ (x : ℝ), (Real.log (x + 5) + Real.log (x - 3) = Real.log (x^2 - 5*x + 6)) ∧
              (x + 5 > 0) ∧ (x - 3 > 0) ∧ (x^2 - 5*x + 6 > 0) := by
  sorry

end NUMINAMATH_CALUDE_no_real_solution_log_equation_l246_24658


namespace NUMINAMATH_CALUDE_only_setC_is_pythagorean_triple_l246_24651

-- Define the sets of numbers
def setA : List ℕ := [1, 2, 2]
def setB : List ℕ := [3^2, 4^2, 5^2]
def setC : List ℕ := [5, 12, 13]
def setD : List ℕ := [6, 6, 6]

-- Define a function to check if a list of three numbers forms a Pythagorean triple
def isPythagoreanTriple (triple : List ℕ) : Prop :=
  match triple with
  | [a, b, c] => a^2 + b^2 = c^2
  | _ => False

-- Theorem stating that only setC forms a Pythagorean triple
theorem only_setC_is_pythagorean_triple :
  ¬(isPythagoreanTriple setA) ∧
  ¬(isPythagoreanTriple setB) ∧
  (isPythagoreanTriple setC) ∧
  ¬(isPythagoreanTriple setD) :=
sorry

end NUMINAMATH_CALUDE_only_setC_is_pythagorean_triple_l246_24651


namespace NUMINAMATH_CALUDE_point_outside_circle_l246_24673

theorem point_outside_circle (m : ℝ) : 
  (1 : ℝ)^2 + (1 : ℝ)^2 + 4*m*1 - 2*1 + 5*m > 0 ∧ 
  ∃ (x y : ℝ), x^2 + y^2 + 4*m*x - 2*y + 5*m = 0 ↔ 
  m > 1 ∨ (0 < m ∧ m < 1/4) := by
sorry

end NUMINAMATH_CALUDE_point_outside_circle_l246_24673


namespace NUMINAMATH_CALUDE_unique_g_two_num_solutions_sum_solutions_final_result_l246_24634

/-- A function satisfying the given property for all real x and y -/
def SatisfiesProperty (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, g x * g y - g (x * y) = 2 * x + 2 * y

/-- The main theorem stating that g(2) = 3 is the only solution -/
theorem unique_g_two (g : ℝ → ℝ) (h : SatisfiesProperty g) : g 2 = 3 := by
  sorry

/-- The number of possible values for g(2) is 1 -/
theorem num_solutions (g : ℝ → ℝ) (h : SatisfiesProperty g) : 
  ∃! x : ℝ, g 2 = x := by
  sorry

/-- The sum of all possible values of g(2) is 3 -/
theorem sum_solutions (g : ℝ → ℝ) (h : SatisfiesProperty g) :
  ∃ x : ℝ, g 2 = x ∧ x = 3 := by
  sorry

/-- The product of the number of solutions and their sum is 3 -/
theorem final_result (g : ℝ → ℝ) (h : SatisfiesProperty g) :
  (∃! x : ℝ, g 2 = x) ∧ (∃ x : ℝ, g 2 = x ∧ x = 3) → 1 * 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_g_two_num_solutions_sum_solutions_final_result_l246_24634


namespace NUMINAMATH_CALUDE_largest_number_l246_24603

theorem largest_number (π : ℝ) (h1 : π > 3) : π = max π (max (Real.sqrt 2) (max (-2) 3)) := by
  sorry

end NUMINAMATH_CALUDE_largest_number_l246_24603


namespace NUMINAMATH_CALUDE_ball_transfer_equality_l246_24695

/-- Represents a box containing balls of different colors -/
structure Box where
  black : ℕ
  white : ℕ

/-- Transfers balls between boxes -/
def transfer (a b : Box) (n : ℕ) : Box × Box :=
  let blackToB := min n a.black
  let whiteToA := min (n - blackToB) b.white
  let blackToA := n - whiteToA
  ({ black := a.black - blackToB + blackToA,
     white := a.white + whiteToA },
   { black := b.black + blackToB - blackToA,
     white := b.white - whiteToA })

theorem ball_transfer_equality (a b : Box) (n : ℕ) :
  let (a', b') := transfer a b n
  a'.white = b'.black := by sorry

end NUMINAMATH_CALUDE_ball_transfer_equality_l246_24695


namespace NUMINAMATH_CALUDE_rhombus_side_length_l246_24672

-- Define the rhombus ABCD
def Rhombus (A B C D : Point) : Prop := sorry

-- Define the pyramid SABCD
def Pyramid (S A B C D : Point) : Prop := sorry

-- Define the inclination of lateral faces
def LateralFacesInclined (S A B C D : Point) (angle : ℝ) : Prop := sorry

-- Define midpoints
def Midpoint (M A B : Point) : Prop := sorry

-- Define the rectangular parallelepiped
def RectangularParallelepiped (M N K L F P R Q : Point) : Prop := sorry

-- Define the intersection points
def IntersectionPoints (S A B C D M N K L F P R Q : Point) : Prop := sorry

-- Define the volume of a polyhedron
def PolyhedronVolume (M N K L F P R Q : Point) : ℝ := sorry

-- Define the radius of an inscribed circle
def InscribedCircleRadius (A B C D : Point) : ℝ := sorry

-- Define the side length of a rhombus
def RhombusSideLength (A B C D : Point) : ℝ := sorry

theorem rhombus_side_length 
  (A B C D S M N K L F P R Q : Point) :
  Rhombus A B C D →
  Pyramid S A B C D →
  LateralFacesInclined S A B C D (60 * π / 180) →
  Midpoint M A B ∧ Midpoint N B C ∧ Midpoint K C D ∧ Midpoint L D A →
  RectangularParallelepiped M N K L F P R Q →
  IntersectionPoints S A B C D M N K L F P R Q →
  PolyhedronVolume M N K L F P R Q = 12 * Real.sqrt 3 →
  InscribedCircleRadius A B C D = 2.4 →
  RhombusSideLength A B C D = 5 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_side_length_l246_24672


namespace NUMINAMATH_CALUDE_not_consecutive_numbers_l246_24642

theorem not_consecutive_numbers (a b c : ℕ) (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) :
  ¬∃ (k : ℕ), ({2023 + a - b, 2023 + b - c, 2023 + c - a} : Finset ℕ) = {k - 1, k, k + 1} :=
by sorry

end NUMINAMATH_CALUDE_not_consecutive_numbers_l246_24642


namespace NUMINAMATH_CALUDE_union_when_m_is_neg_half_subset_iff_m_geq_zero_l246_24675

-- Define sets A and B
def A : Set ℝ := {x | x^2 + x - 2 < 0}
def B (m : ℝ) : Set ℝ := {x | 2*m < x ∧ x < 1 - m}

-- Theorem 1: When m = -1/2, A ∪ B = {x | -2 < x < 3/2}
theorem union_when_m_is_neg_half :
  A ∪ B (-1/2) = {x : ℝ | -2 < x ∧ x < 3/2} := by sorry

-- Theorem 2: B ⊆ A if and only if m ≥ 0
theorem subset_iff_m_geq_zero :
  ∀ m : ℝ, B m ⊆ A ↔ m ≥ 0 := by sorry

end NUMINAMATH_CALUDE_union_when_m_is_neg_half_subset_iff_m_geq_zero_l246_24675


namespace NUMINAMATH_CALUDE_days_with_parrot_l246_24628

-- Define the given conditions
def total_phrases : ℕ := 17
def phrases_per_week : ℕ := 2
def initial_phrases : ℕ := 3
def days_per_week : ℕ := 7

-- Define the theorem
theorem days_with_parrot : 
  (total_phrases - initial_phrases) / phrases_per_week * days_per_week = 49 := by
  sorry

end NUMINAMATH_CALUDE_days_with_parrot_l246_24628


namespace NUMINAMATH_CALUDE_quadratic_root_implies_k_l246_24676

/-- 
If the quadratic equation x^2 + kx - 3 = 0 has 1 as a root, 
then k = 2.
-/
theorem quadratic_root_implies_k (k : ℝ) : 
  (1 : ℝ)^2 + k*(1 : ℝ) - 3 = 0 → k = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_k_l246_24676


namespace NUMINAMATH_CALUDE_theater_group_arrangement_l246_24686

theorem theater_group_arrangement (n : ℕ) : n ≥ 1981 ∧ 
  n % 9 = 0 ∧ n % 10 = 0 ∧ n % 11 = 0 ∧ n % 12 = 1 → n = 1981 :=
by sorry

end NUMINAMATH_CALUDE_theater_group_arrangement_l246_24686


namespace NUMINAMATH_CALUDE_min_max_abs_cubic_linear_l246_24619

theorem min_max_abs_cubic_linear (y : ℝ) : 
  (∃ (x : ℝ), x ∈ Set.Icc 0 1 ∧ |x^3 - x*y| ≥ 1) ∧
  (∃ (y₀ : ℝ), ∀ (x : ℝ), x ∈ Set.Icc 0 1 → |x^3 - x*y₀| ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_min_max_abs_cubic_linear_l246_24619


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l246_24659

theorem quadratic_inequality_solution (b c : ℝ) :
  (∀ x, x^2 + b*x + c < 0 ↔ 1 < x ∧ x < 2) →
  (∀ x, x^2 + b*x + c = 0 ↔ x = 1 ∨ x = 2) →
  b + c = -1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l246_24659


namespace NUMINAMATH_CALUDE_cookie_pie_slices_left_l246_24623

theorem cookie_pie_slices_left (num_pies : ℕ) (slices_per_pie : ℕ) (num_people : ℕ) : 
  num_pies = 5 → 
  slices_per_pie = 12 → 
  num_people = 33 → 
  num_pies * slices_per_pie - num_people = 27 := by
sorry

end NUMINAMATH_CALUDE_cookie_pie_slices_left_l246_24623


namespace NUMINAMATH_CALUDE_three_digit_geometric_progression_l246_24687

theorem three_digit_geometric_progression :
  ∀ a b c : ℕ,
  100 ≤ 100 * a + 10 * b + c ∧ 100 * a + 10 * b + c < 1000 →
  (∃ r : ℚ,
    (100 * b + 10 * c + a : ℚ) = r * (100 * a + 10 * b + c : ℚ) ∧
    (100 * c + 10 * a + b : ℚ) = r * (100 * b + 10 * c + a : ℚ)) →
  ((a = b ∧ b = c ∧ 1 ≤ a ∧ a ≤ 9) ∨
   (a = 2 ∧ b = 4 ∧ c = 3) ∨
   (a = 4 ∧ b = 8 ∧ c = 6)) :=
by sorry

end NUMINAMATH_CALUDE_three_digit_geometric_progression_l246_24687


namespace NUMINAMATH_CALUDE_geometric_sequence_nth_term_l246_24644

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), ∀ (n : ℕ), a (n + 1) = r * a n

theorem geometric_sequence_nth_term
  (a : ℕ → ℝ)
  (h_geo : geometric_sequence a)
  (h_sum : a 2 + a 5 = 18)
  (h_prod : a 3 * a 4 = 32)
  (h_nth : ∃ (n : ℕ), a n = 128) :
  ∃ (n : ℕ), a n = 128 ∧ n = 8 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_nth_term_l246_24644


namespace NUMINAMATH_CALUDE_sum_of_ages_is_fifty_l246_24678

/-- The sum of ages of 5 children born at intervals of 3 years, with the youngest being 4 years old -/
def sum_of_ages : ℕ :=
  let youngest_age := 4
  let interval := 3
  let num_children := 5
  List.range num_children
    |>.map (fun i => youngest_age + i * interval)
    |>.sum

theorem sum_of_ages_is_fifty :
  sum_of_ages = 50 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ages_is_fifty_l246_24678


namespace NUMINAMATH_CALUDE_solution_satisfies_equation_l246_24625

open Real

noncomputable def y (x C₁ C₂ : ℝ) : ℝ :=
  C₁ * cos (2 * x) + C₂ * sin (2 * x) + (2 * cos (2 * x) + 8 * sin (2 * x)) * x + (1/2) * exp (2 * x)

theorem solution_satisfies_equation (x C₁ C₂ : ℝ) :
  (deriv^[2] (y C₁ C₂)) x + 4 * y C₁ C₂ x = -8 * sin (2 * x) + 32 * cos (2 * x) + 4 * exp (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_solution_satisfies_equation_l246_24625


namespace NUMINAMATH_CALUDE_product_of_seven_consecutive_divisible_by_ten_l246_24692

theorem product_of_seven_consecutive_divisible_by_ten (n : ℕ+) :
  10 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4) * (n + 5) * (n + 6)) := by
  sorry

end NUMINAMATH_CALUDE_product_of_seven_consecutive_divisible_by_ten_l246_24692


namespace NUMINAMATH_CALUDE_green_ball_probability_l246_24631

/-- Represents a container with red and green balls -/
structure Container where
  red : ℕ
  green : ℕ

/-- Calculates the probability of selecting a green ball from a container -/
def greenProbability (c : Container) : ℚ :=
  c.green / (c.red + c.green)

/-- Theorem: The probability of selecting a green ball is 26/45 -/
theorem green_ball_probability :
  let containerA : Container := ⟨5, 7⟩
  let containerB : Container := ⟨4, 5⟩
  let containerC : Container := ⟨7, 3⟩
  let totalContainers : ℕ := 3
  let probA : ℚ := 1 / totalContainers * greenProbability containerA
  let probB : ℚ := 1 / totalContainers * greenProbability containerB
  let probC : ℚ := 1 / totalContainers * greenProbability containerC
  probA + probB + probC = 26 / 45 := by
  sorry

end NUMINAMATH_CALUDE_green_ball_probability_l246_24631


namespace NUMINAMATH_CALUDE_difference_equals_negative_two_hundred_l246_24679

/-- Given that the average of (a+d) and (b+d) is 80, and the average of (b+d) and (c+d) is 180,
    prove that a - c = -200 -/
theorem difference_equals_negative_two_hundred
  (a b c d : ℝ)
  (h1 : ((a + d) + (b + d)) / 2 = 80)
  (h2 : ((b + d) + (c + d)) / 2 = 180) :
  a - c = -200 := by sorry

end NUMINAMATH_CALUDE_difference_equals_negative_two_hundred_l246_24679


namespace NUMINAMATH_CALUDE_triangle_area_5_5_6_l246_24606

/-- The area of a triangle with sides 5, 5, and 6 units is 12 square units. -/
theorem triangle_area_5_5_6 : ∃ (A : ℝ), A = 12 ∧ A = Real.sqrt (8 * (8 - 5) * (8 - 5) * (8 - 6)) := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_5_5_6_l246_24606


namespace NUMINAMATH_CALUDE_tv_purchase_time_l246_24647

/-- Calculates the number of months required to save for a television purchase. -/
def months_to_purchase_tv (monthly_income : ℕ) (food_expense : ℕ) (utilities_expense : ℕ) 
  (other_expenses : ℕ) (current_savings : ℕ) (tv_cost : ℕ) : ℕ :=
  let total_expenses := food_expense + utilities_expense + other_expenses
  let monthly_savings := monthly_income - total_expenses
  let additional_savings_needed := tv_cost - current_savings
  (additional_savings_needed + monthly_savings - 1) / monthly_savings

theorem tv_purchase_time :
  months_to_purchase_tv 30000 15000 5000 2500 10000 25000 = 2 := by
  sorry

end NUMINAMATH_CALUDE_tv_purchase_time_l246_24647


namespace NUMINAMATH_CALUDE_triangle_area_arithmetic_angles_l246_24665

/-- Given a triangle ABC with sides a and c, and angles A, B, C forming an arithmetic sequence,
    prove that its area is 3√3 when a = 4 and c = 3. -/
theorem triangle_area_arithmetic_angles (A B C : ℝ) (a c : ℝ) :
  -- Angles form an arithmetic sequence
  ∃ d : ℝ, A = B - d ∧ C = B + d
  -- Sum of angles in a triangle is π (180°)
  → A + B + C = π
  -- Given side lengths
  → a = 4
  → c = 3
  -- Area of the triangle
  → (1/2) * a * c * Real.sin B = 3 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_arithmetic_angles_l246_24665


namespace NUMINAMATH_CALUDE_duck_cow_problem_l246_24602

theorem duck_cow_problem (ducks cows : ℕ) : 
  2 * ducks + 4 * cows = 2 * (ducks + cows) + 34 → cows = 17 := by
  sorry

end NUMINAMATH_CALUDE_duck_cow_problem_l246_24602


namespace NUMINAMATH_CALUDE_a_range_l246_24617

/-- A quadratic function f(x) = x² + 2(a-1)x + 5 that is increasing on (4, +∞) -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 5

/-- The property that f is increasing on (4, +∞) -/
def f_increasing (a : ℝ) : Prop :=
  ∀ x y, x > 4 → y > x → f a y > f a x

/-- The theorem stating the range of a -/
theorem a_range (a : ℝ) (h : f_increasing a) : a ≥ -3 :=
sorry

end NUMINAMATH_CALUDE_a_range_l246_24617


namespace NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l246_24610

-- Define the angle α
def α : Real := sorry

-- Define the point P
def P : ℝ × ℝ := (-1, 2)

-- Theorem statement
theorem tan_alpha_plus_pi_fourth (h : P.fst = -Real.tan α ∧ P.snd = Real.tan α * P.fst) :
  Real.tan (α + π/4) = -1/3 := by sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l246_24610


namespace NUMINAMATH_CALUDE_excluded_students_average_mark_l246_24622

/-- Given a class of students with their exam marks, this theorem proves
    the average mark of excluded students based on the given conditions. -/
theorem excluded_students_average_mark
  (total_students : ℕ)
  (all_average : ℝ)
  (excluded_count : ℕ)
  (remaining_average : ℝ)
  (h1 : total_students = 10)
  (h2 : all_average = 70)
  (h3 : excluded_count = 5)
  (h4 : remaining_average = 90) :
  (total_students * all_average - (total_students - excluded_count) * remaining_average) / excluded_count = 50 := by
  sorry

end NUMINAMATH_CALUDE_excluded_students_average_mark_l246_24622


namespace NUMINAMATH_CALUDE_josies_remaining_money_l246_24636

/-- Given an initial amount of money and the costs of items,
    calculate the remaining amount after purchasing the items. -/
def remaining_money (initial_amount : ℕ) (item1_cost : ℕ) (item1_quantity : ℕ) (item2_cost : ℕ) : ℕ :=
  initial_amount - (item1_cost * item1_quantity + item2_cost)

/-- Prove that given an initial amount of $50, after spending $9 each on two items
    and $25 on another item, the remaining amount is $7. -/
theorem josies_remaining_money :
  remaining_money 50 9 2 25 = 7 := by
  sorry

end NUMINAMATH_CALUDE_josies_remaining_money_l246_24636
