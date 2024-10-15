import Mathlib

namespace NUMINAMATH_CALUDE_square_to_rectangle_ratio_l3111_311195

/-- Given a square of side length 4, with E and F as midpoints of opposite sides,
    and AG perpendicular to BF, prove that when dissected into four pieces and
    reassembled into a rectangle, the ratio of height to base is 4/5 -/
theorem square_to_rectangle_ratio (square_side : ℝ) (E F G : ℝ × ℝ) 
  (h1 : square_side = 4)
  (h2 : E.1 = 2 ∧ E.2 = 0)
  (h3 : F.1 = 0 ∧ F.2 = 2)
  (h4 : (G.1 - 4) * (F.2 - 0) = (G.2 - 0) * (F.1 - 4)) -- AG ⟂ BF
  : ∃ (rect_height rect_base : ℝ),
    rect_height * rect_base = square_side^2 ∧
    rect_height / rect_base = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_square_to_rectangle_ratio_l3111_311195


namespace NUMINAMATH_CALUDE_f_difference_at_five_l3111_311156

-- Define the function f
def f (x : ℝ) : ℝ := x^4 + x^2 + 5*x + 3

-- State the theorem
theorem f_difference_at_five : f 5 - f (-5) = 50 := by
  sorry

end NUMINAMATH_CALUDE_f_difference_at_five_l3111_311156


namespace NUMINAMATH_CALUDE_intersection_M_N_l3111_311159

def M : Set ℝ := {y | ∃ x : ℝ, y = x^2}
def N : Set ℝ := {y | ∃ x : ℝ, y = 2 - |x|}

theorem intersection_M_N : M ∩ N = {y : ℝ | 0 ≤ y ∧ y ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3111_311159


namespace NUMINAMATH_CALUDE_pauls_supplies_l3111_311155

/-- Given Paul's initial and final crayon counts, and initial eraser count,
    prove the difference between remaining erasers and crayons. -/
theorem pauls_supplies (initial_crayons : ℕ) (initial_erasers : ℕ) (final_crayons : ℕ)
    (h1 : initial_crayons = 601)
    (h2 : initial_erasers = 406)
    (h3 : final_crayons = 336) :
    initial_erasers - final_crayons = 70 := by
  sorry

end NUMINAMATH_CALUDE_pauls_supplies_l3111_311155


namespace NUMINAMATH_CALUDE_stock_investment_calculation_l3111_311149

/-- Given a stock with price 64 and dividend yield 1623%, prove that an investment
    earning 1900 in dividends is approximately 117.00 -/
theorem stock_investment_calculation (stock_price : ℝ) (dividend_yield : ℝ) (dividend_earned : ℝ) :
  stock_price = 64 →
  dividend_yield = 1623 →
  dividend_earned = 1900 →
  ∃ (investment : ℝ), abs (investment - 117.00) < 0.01 := by
sorry

end NUMINAMATH_CALUDE_stock_investment_calculation_l3111_311149


namespace NUMINAMATH_CALUDE_midpoint_triangle_perimeter_l3111_311111

/-- A right prism with regular pentagonal bases -/
structure RightPrism :=
  (height : ℝ)
  (base_side_length : ℝ)

/-- Midpoint of an edge -/
structure Midpoint :=
  (edge : String)

/-- Triangle formed by three midpoints -/
structure MidpointTriangle :=
  (p1 : Midpoint)
  (p2 : Midpoint)
  (p3 : Midpoint)

/-- Calculate the perimeter of the midpoint triangle -/
def perimeter (prism : RightPrism) (triangle : MidpointTriangle) : ℝ :=
  sorry

/-- Theorem stating the perimeter of the midpoint triangle -/
theorem midpoint_triangle_perimeter 
  (prism : RightPrism) 
  (triangle : MidpointTriangle) 
  (h1 : prism.height = 25) 
  (h2 : prism.base_side_length = 15) 
  (h3 : triangle.p1 = Midpoint.mk "AB") 
  (h4 : triangle.p2 = Midpoint.mk "BC") 
  (h5 : triangle.p3 = Midpoint.mk "CD") : 
  perimeter prism triangle = 15 + 2 * Real.sqrt 212.5 :=
sorry

end NUMINAMATH_CALUDE_midpoint_triangle_perimeter_l3111_311111


namespace NUMINAMATH_CALUDE_tree_planting_equation_l3111_311134

/-- Represents the relationship between the number of people planting trees and the total number of seedlings. -/
theorem tree_planting_equation (x : ℤ) (total_seedlings : ℤ) : 
  (5 * x + 3 = total_seedlings) ∧ (6 * x = total_seedlings + 4) →
  5 * x + 3 = 6 * x - 4 := by
  sorry

#check tree_planting_equation

end NUMINAMATH_CALUDE_tree_planting_equation_l3111_311134


namespace NUMINAMATH_CALUDE_last_two_digits_sum_factorials_l3111_311143

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

def last_two_digits (n : ℕ) : ℕ := n % 100

theorem last_two_digits_sum_factorials :
  last_two_digits (sum_factorials 15) = last_two_digits (sum_factorials 9) := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_sum_factorials_l3111_311143


namespace NUMINAMATH_CALUDE_chocolate_bar_problem_l3111_311181

/-- Represents the problem of calculating unsold chocolate bars -/
theorem chocolate_bar_problem (cost_per_bar : ℕ) (total_bars : ℕ) (revenue : ℕ) : 
  cost_per_bar = 3 → 
  total_bars = 9 → 
  revenue = 18 → 
  total_bars - (revenue / cost_per_bar) = 3 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bar_problem_l3111_311181


namespace NUMINAMATH_CALUDE_range_of_c_l3111_311194

/-- Given c > 0, if the function y = c^x is monotonically decreasing on ℝ or 
    the function g(x) = lg(2cx^2 + 2x + 1) has domain ℝ, but not both, 
    then c ≥ 1 or 0 < c ≤ 1/2 -/
theorem range_of_c (c : ℝ) (h_c : c > 0) : 
  (∀ x y : ℝ, x < y → c^x > c^y) ∨ 
  (∀ x : ℝ, 2*c*x^2 + 2*x + 1 > 0) ∧ 
  ¬((∀ x y : ℝ, x < y → c^x > c^y) ∧ 
    (∀ x : ℝ, 2*c*x^2 + 2*x + 1 > 0)) → 
  c ≥ 1 ∨ (0 < c ∧ c ≤ 1/2) := by
  sorry

end NUMINAMATH_CALUDE_range_of_c_l3111_311194


namespace NUMINAMATH_CALUDE_all_b_k_divisible_by_six_l3111_311101

/-- The number obtained by writing the integers from 1 to n from left to right -/
def b (n : ℕ) : ℕ := sorry

/-- The sum of the squares of the digits of b_n -/
def g (n : ℕ) : ℕ := (n * (n + 1) * (2 * n + 1)) / 6

theorem all_b_k_divisible_by_six (k : ℕ) (h : 1 ≤ k ∧ k ≤ 50) : 
  6 ∣ g k := by sorry

end NUMINAMATH_CALUDE_all_b_k_divisible_by_six_l3111_311101


namespace NUMINAMATH_CALUDE_negation_of_existence_power_of_two_exceeds_1000_l3111_311161

theorem negation_of_existence (p : ℕ → Prop) : 
  (¬ ∃ n, p n) ↔ (∀ n, ¬ p n) := by sorry

theorem power_of_two_exceeds_1000 : 
  (¬ ∃ n : ℕ, 2^n > 1000) ↔ (∀ n : ℕ, 2^n ≤ 1000) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_power_of_two_exceeds_1000_l3111_311161


namespace NUMINAMATH_CALUDE_basket_total_is_40_l3111_311151

/-- A basket containing apples and oranges -/
structure Basket where
  oranges : ℕ
  apples : ℕ

/-- The total number of fruit in the basket -/
def Basket.total (b : Basket) : ℕ := b.oranges + b.apples

theorem basket_total_is_40 (b : Basket) 
  (h1 : b.apples = 3 * b.oranges) 
  (h2 : b.oranges = 10) : 
  b.total = 40 := by
sorry

end NUMINAMATH_CALUDE_basket_total_is_40_l3111_311151


namespace NUMINAMATH_CALUDE_log_problem_l3111_311103

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_problem (x : ℝ) (h : log 3 (5 * x) = 3) : log x 125 = 3/2 :=
by
  sorry

end NUMINAMATH_CALUDE_log_problem_l3111_311103


namespace NUMINAMATH_CALUDE_edward_summer_earnings_l3111_311129

/-- Edward's lawn mowing business earnings --/
def lawn_mowing_problem (spring_earnings summer_earnings supplies_cost final_amount : ℕ) : Prop :=
  spring_earnings + summer_earnings = supplies_cost + final_amount

theorem edward_summer_earnings :
  ∃ (summer_earnings : ℕ),
    lawn_mowing_problem 2 summer_earnings 5 24 ∧ summer_earnings = 27 :=
by
  sorry

end NUMINAMATH_CALUDE_edward_summer_earnings_l3111_311129


namespace NUMINAMATH_CALUDE_ratio_transitivity_l3111_311104

theorem ratio_transitivity (a b c : ℚ) 
  (hab : a / b = 4 / 3) 
  (hbc : b / c = 1 / 5) : 
  a / c = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_transitivity_l3111_311104


namespace NUMINAMATH_CALUDE_eighth_grade_gpa_l3111_311170

/-- Proves that the average GPA for 8th graders is 91 given the specified conditions -/
theorem eighth_grade_gpa (sixth_grade_gpa seventh_grade_gpa eighth_grade_gpa school_avg_gpa : ℝ) :
  sixth_grade_gpa = 93 →
  seventh_grade_gpa = sixth_grade_gpa + 2 →
  school_avg_gpa = 93 →
  school_avg_gpa = (sixth_grade_gpa + seventh_grade_gpa + eighth_grade_gpa) / 3 →
  eighth_grade_gpa = 91 := by
  sorry

end NUMINAMATH_CALUDE_eighth_grade_gpa_l3111_311170


namespace NUMINAMATH_CALUDE_jaymee_shara_age_difference_l3111_311190

theorem jaymee_shara_age_difference (shara_age jaymee_age : ℕ) 
  (h1 : shara_age = 10) 
  (h2 : jaymee_age = 22) : 
  jaymee_age - 2 * shara_age = 2 := by
  sorry

end NUMINAMATH_CALUDE_jaymee_shara_age_difference_l3111_311190


namespace NUMINAMATH_CALUDE_students_liking_both_desserts_l3111_311184

theorem students_liking_both_desserts
  (total : ℕ)
  (like_brownies : ℕ)
  (like_ice_cream : ℕ)
  (like_neither : ℕ)
  (h1 : total = 45)
  (h2 : like_brownies = 22)
  (h3 : like_ice_cream = 17)
  (h4 : like_neither = 13) :
  (like_brownies + like_ice_cream) - (total - like_neither) = 7 := by
  sorry

end NUMINAMATH_CALUDE_students_liking_both_desserts_l3111_311184


namespace NUMINAMATH_CALUDE_omega_squared_plus_four_omega_plus_forty_modulus_l3111_311132

theorem omega_squared_plus_four_omega_plus_forty_modulus (ω : ℂ) (h : ω = 5 + 3*I) : 
  Complex.abs (ω^2 + 4*ω + 40) = 2 * Real.sqrt 1885 := by sorry

end NUMINAMATH_CALUDE_omega_squared_plus_four_omega_plus_forty_modulus_l3111_311132


namespace NUMINAMATH_CALUDE_min_value_of_sum_min_value_is_four_fifths_min_value_equality_l3111_311166

theorem min_value_of_sum (a b : ℝ) : 
  a > 0 → b > 0 → a + 2*b = 2 → 
  ∀ x y : ℝ, x > 0 → y > 0 → x + 2*y = 2 → 
  1/(1+a) + 1/(2+2*b) ≤ 1/(1+x) + 1/(2+2*y) :=
by
  sorry

theorem min_value_is_four_fifths (a b : ℝ) :
  a > 0 → b > 0 → a + 2*b = 2 → 
  1/(1+a) + 1/(2+2*b) ≥ 4/5 :=
by
  sorry

theorem min_value_equality (a b : ℝ) :
  a > 0 → b > 0 → a + 2*b = 2 → 
  (1/(1+a) + 1/(2+2*b) = 4/5) ↔ (a = 3/2 ∧ b = 1/4) :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_of_sum_min_value_is_four_fifths_min_value_equality_l3111_311166


namespace NUMINAMATH_CALUDE_average_speed_of_planets_l3111_311117

/-- Calculates the average speed of Venus, Earth, and Mars in miles per hour -/
theorem average_speed_of_planets (venus_speed earth_speed mars_speed : ℝ) 
  (h1 : venus_speed = 21.9)
  (h2 : earth_speed = 18.5)
  (h3 : mars_speed = 15) :
  (venus_speed * 3600 + earth_speed * 3600 + mars_speed * 3600) / 3 = 66480 := by
  sorry

#eval (21.9 * 3600 + 18.5 * 3600 + 15 * 3600) / 3

end NUMINAMATH_CALUDE_average_speed_of_planets_l3111_311117


namespace NUMINAMATH_CALUDE_wen_family_science_fair_cost_l3111_311123

theorem wen_family_science_fair_cost : ∀ (x : ℝ),
  x > 0 →
  0.7 * x = 7 →
  let student_ticket := 0.6 * x
  let regular_ticket := x
  let senior_ticket := 0.7 * x
  3 * student_ticket + regular_ticket + senior_ticket = 35 :=
by
  sorry

end NUMINAMATH_CALUDE_wen_family_science_fair_cost_l3111_311123


namespace NUMINAMATH_CALUDE_cost_price_calculation_l3111_311112

theorem cost_price_calculation (selling_price : ℚ) (profit_percentage : ℚ) 
  (h1 : selling_price = 48)
  (h2 : profit_percentage = 20 / 100) :
  ∃ (cost_price : ℚ), 
    cost_price * (1 + profit_percentage) = selling_price ∧ 
    cost_price = 40 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l3111_311112


namespace NUMINAMATH_CALUDE_sqrt_expression_equals_three_halves_l3111_311109

theorem sqrt_expression_equals_three_halves :
  (Real.sqrt 8 - Real.sqrt (1/2)) / Real.sqrt 2 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equals_three_halves_l3111_311109


namespace NUMINAMATH_CALUDE_rectangle_area_equals_perimeter_l3111_311164

theorem rectangle_area_equals_perimeter (x : ℝ) : 
  (3 * x) * (x + 5) = 2 * (3 * x) + 2 * (x + 5) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_equals_perimeter_l3111_311164


namespace NUMINAMATH_CALUDE_cricket_bat_selling_price_l3111_311182

-- Define the profit
def profit : ℝ := 150

-- Define the profit percentage
def profitPercentage : ℝ := 20

-- Define the selling price
def sellingPrice : ℝ := 900

-- Theorem to prove
theorem cricket_bat_selling_price :
  let costPrice := profit / (profitPercentage / 100)
  sellingPrice = costPrice + profit := by
  sorry

end NUMINAMATH_CALUDE_cricket_bat_selling_price_l3111_311182


namespace NUMINAMATH_CALUDE_prime_sum_problem_l3111_311176

theorem prime_sum_problem (m n : ℕ) (hm : Nat.Prime m) (hn : Nat.Prime n) 
  (h : 5 * m + 7 * n = 129) : m + n = 19 ∨ m + n = 25 := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_problem_l3111_311176


namespace NUMINAMATH_CALUDE_max_b_for_integer_solution_l3111_311196

theorem max_b_for_integer_solution : ∃ (b : ℤ), b = 9599 ∧
  (∀ (b' : ℤ), (∃ (x : ℤ), x^2 + b'*x - 9600 = 0 ∧ 10 ∣ x ∧ 12 ∣ x) → b' ≤ b) ∧
  (∃ (x : ℤ), x^2 + b*x - 9600 = 0 ∧ 10 ∣ x ∧ 12 ∣ x) := by
  sorry

end NUMINAMATH_CALUDE_max_b_for_integer_solution_l3111_311196


namespace NUMINAMATH_CALUDE_monotonic_decreasing_condition_l3111_311192

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 3 * x + 4

-- State the theorem
theorem monotonic_decreasing_condition (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ < 6 → f a x₁ > f a x₂) ↔ 0 ≤ a ∧ a ≤ 1/4 :=
by sorry

end NUMINAMATH_CALUDE_monotonic_decreasing_condition_l3111_311192


namespace NUMINAMATH_CALUDE_max_sum_consecutive_triples_l3111_311139

/-- Represents a permutation of the digits 1 to 9 -/
def Permutation := Fin 9 → Fin 9

/-- Calculates the sum of seven consecutive three-digit numbers formed from a permutation -/
def sumConsecutiveTriples (p : Permutation) : ℕ :=
  (100 * p 0 + 110 * p 1 + 111 * p 2 + 111 * p 3 + 111 * p 4 + 111 * p 5 + 111 * p 6 + 11 * p 7 + p 8).val

/-- The maximum possible sum of consecutive triples -/
def maxSum : ℕ := 4648

/-- Theorem stating that the maximum sum of consecutive triples is 4648 -/
theorem max_sum_consecutive_triples :
  ∀ p : Permutation, sumConsecutiveTriples p ≤ maxSum :=
sorry

end NUMINAMATH_CALUDE_max_sum_consecutive_triples_l3111_311139


namespace NUMINAMATH_CALUDE_vacation_cost_l3111_311174

theorem vacation_cost (C : ℝ) : 
  (C / 5 - C / 8 = 60) → C = 800 := by sorry

end NUMINAMATH_CALUDE_vacation_cost_l3111_311174


namespace NUMINAMATH_CALUDE_distance_from_origin_l3111_311157

theorem distance_from_origin (x y : ℝ) (h1 : x > 2) (h2 : x = 15) 
  (h3 : (x - 2)^2 + (y - 7)^2 = 13^2) : 
  Real.sqrt (x^2 + y^2) = Real.sqrt 274 := by
  sorry

end NUMINAMATH_CALUDE_distance_from_origin_l3111_311157


namespace NUMINAMATH_CALUDE_gcd_228_2008_l3111_311140

theorem gcd_228_2008 : Nat.gcd 228 2008 = 4 := by
  sorry

end NUMINAMATH_CALUDE_gcd_228_2008_l3111_311140


namespace NUMINAMATH_CALUDE_quadratic_solution_l3111_311179

theorem quadratic_solution (b : ℝ) : 
  ((-9 : ℝ)^2 + b * (-9) - 36 = 0) → b = 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_l3111_311179


namespace NUMINAMATH_CALUDE_circle_equation_specific_l3111_311154

/-- The standard equation of a circle with center (h, k) and radius r is (x - h)² + (y - k)² = r² -/
def circle_equation (h k r : ℝ) (x y : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

/-- Theorem: The standard equation of a circle with center (1, -1) and radius √3 is (x-1)² + (y+1)² = 3 -/
theorem circle_equation_specific :
  let h : ℝ := 1
  let k : ℝ := -1
  let r : ℝ := Real.sqrt 3
  ∀ x y : ℝ, circle_equation h k r x y ↔ (x - 1)^2 + (y + 1)^2 = 3 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_specific_l3111_311154


namespace NUMINAMATH_CALUDE_smallest_perfect_square_multiple_l3111_311136

def n : ℕ := 2023

-- Define 2023 as 7 * 17^2
axiom n_factorization : n = 7 * 17^2

-- Define the function to check if a number is a perfect square
def is_perfect_square (x : ℕ) : Prop := ∃ y : ℕ, x = y^2

-- Define the function to check if a number is a multiple of 2023
def is_multiple_of_2023 (x : ℕ) : Prop := ∃ k : ℕ, x = k * n

-- Theorem statement
theorem smallest_perfect_square_multiple :
  (7 * n = (7 * 17)^2) ∧
  is_perfect_square (7 * n) ∧
  is_multiple_of_2023 (7 * n) ∧
  (∀ m : ℕ, m < 7 * n → ¬(is_perfect_square m ∧ is_multiple_of_2023 m)) :=
sorry

end NUMINAMATH_CALUDE_smallest_perfect_square_multiple_l3111_311136


namespace NUMINAMATH_CALUDE_sin_minus_cos_value_l3111_311146

theorem sin_minus_cos_value (x : ℝ) (h : Real.sin x ^ 3 - Real.cos x ^ 3 = -1) : 
  Real.sin x - Real.cos x = -1 := by
sorry

end NUMINAMATH_CALUDE_sin_minus_cos_value_l3111_311146


namespace NUMINAMATH_CALUDE_smallest_even_cube_ending_392_l3111_311189

theorem smallest_even_cube_ending_392 :
  ∀ n : ℕ, n > 0 → Even n → n^3 ≡ 392 [ZMOD 1000] → n ≥ 892 :=
by sorry

end NUMINAMATH_CALUDE_smallest_even_cube_ending_392_l3111_311189


namespace NUMINAMATH_CALUDE_triangle_area_proof_l3111_311125

/-- Given a triangle DEF with inradius r, circumradius R, and angles D, E, F,
    prove that if r = 10, R = 25, and 2cos(E) = cos(D) + cos(F),
    then the area of the triangle is 225√51/5 -/
theorem triangle_area_proof (D E F : ℝ) (r R : ℝ) :
  r = 10 →
  R = 25 →
  2 * Real.cos E = Real.cos D + Real.cos F →
  ∃ (d e f : ℝ),
    d > 0 ∧ e > 0 ∧ f > 0 ∧
    e^2 = d^2 + f^2 - 2*d*f*(Real.cos E) ∧
    Real.cos D = (f^2 + e^2 - d^2) / (2*f*e) ∧
    Real.cos F = (d^2 + e^2 - f^2) / (2*d*e) ∧
    (d + e + f) / 2 * r = 225 * Real.sqrt 51 / 5 :=
sorry

end NUMINAMATH_CALUDE_triangle_area_proof_l3111_311125


namespace NUMINAMATH_CALUDE_condition1_condition2_degree_in_x_l3111_311168

/-- A polynomial in three variables -/
def f (x y z : ℝ) : ℝ := (x^2 - y^3) * (y^3 - z^6) * (z^6 - x^2)

/-- The first condition of the polynomial -/
theorem condition1 (x y z : ℝ) : f x (z^2) y + f x (y^2) z = 0 := by sorry

/-- The second condition of the polynomial -/
theorem condition2 (x y z : ℝ) : f (z^3) y x + f (x^3) y z = 0 := by sorry

/-- The polynomial is of 4th degree in x -/
theorem degree_in_x : ∃ (a b c d e : ℝ → ℝ → ℝ), ∀ x y z, 
  f x y z = a y z * x^4 + b y z * x^3 + c y z * x^2 + d y z * x + e y z := by sorry

end NUMINAMATH_CALUDE_condition1_condition2_degree_in_x_l3111_311168


namespace NUMINAMATH_CALUDE_sin_2x_given_cos_l3111_311142

theorem sin_2x_given_cos (x : ℝ) (h : Real.cos (π / 4 - x) = 3 / 5) : 
  Real.sin (2 * x) = -7 / 25 := by
  sorry

end NUMINAMATH_CALUDE_sin_2x_given_cos_l3111_311142


namespace NUMINAMATH_CALUDE_intersection_of_sets_l3111_311187

theorem intersection_of_sets : 
  let A : Set ℕ := {1, 2, 3, 4, 5}
  let B : Set ℕ := {1, 2, 4, 6}
  A ∩ B = {1, 2, 4} := by
sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l3111_311187


namespace NUMINAMATH_CALUDE_seating_arrangements_l3111_311127

/-- The number of ways to choose k items from n items without replacement and where order matters. -/
def A (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

/-- The number of ways to choose k items from n items without replacement and where order doesn't matter. -/
def C (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- Given a bench with 9 seats and 3 people to be seated with at least 2 empty seats between any two people,
    the number of different seating arrangements is 60. -/
theorem seating_arrangements (total_seats people : ℕ) (h1 : total_seats = 9) (h2 : people = 3) :
  A people people * (C 4 2 + C 4 1) = 60 := by
  sorry


end NUMINAMATH_CALUDE_seating_arrangements_l3111_311127


namespace NUMINAMATH_CALUDE_first_discount_percentage_l3111_311116

/-- Proves that the first discount percentage is 10% given the initial price, 
    second discount percentage, and final price after both discounts. -/
theorem first_discount_percentage (initial_price : ℝ) (second_discount : ℝ) (final_price : ℝ) :
  initial_price = 200 →
  second_discount = 5 →
  final_price = 171 →
  ∃ (x : ℝ), 
    (initial_price * (1 - x / 100) * (1 - second_discount / 100) = final_price) ∧
    x = 10 := by
  sorry

end NUMINAMATH_CALUDE_first_discount_percentage_l3111_311116


namespace NUMINAMATH_CALUDE_ratio_chain_l3111_311145

theorem ratio_chain (a b c d e f : ℝ) 
  (h1 : a / b = 1 / 4)
  (h2 : b / c = 13 / 9)
  (h3 : c / d = 5 / 13)
  (h4 : d / e = 2 / 3)
  (h5 : e / f = 7 / 5) :
  a / f = 7 / 6 := by
sorry

end NUMINAMATH_CALUDE_ratio_chain_l3111_311145


namespace NUMINAMATH_CALUDE_claire_pets_male_hamster_fraction_l3111_311171

theorem claire_pets_male_hamster_fraction :
  ∀ (total_pets gerbils hamsters male_pets male_gerbils male_hamsters : ℕ),
    total_pets = 90 →
    gerbils = 66 →
    total_pets = gerbils + hamsters →
    male_pets = 25 →
    male_gerbils = 16 →
    male_pets = male_gerbils + male_hamsters →
    (male_hamsters : ℚ) / (hamsters : ℚ) = 3/8 :=
by
  sorry

end NUMINAMATH_CALUDE_claire_pets_male_hamster_fraction_l3111_311171


namespace NUMINAMATH_CALUDE_banquet_food_consumption_l3111_311185

/-- The total food consumed at a banquet is at least the product of the minimum number of guests and the maximum food consumed per guest. -/
theorem banquet_food_consumption 
  (max_food_per_guest : ℝ) 
  (min_guests : ℕ) 
  (h1 : max_food_per_guest = 2) 
  (h2 : min_guests = 162) : 
  ℝ := by
  sorry

#eval (2 : ℝ) * 162  -- Expected output: 324

end NUMINAMATH_CALUDE_banquet_food_consumption_l3111_311185


namespace NUMINAMATH_CALUDE_coefficient_of_y_in_equation3_l3111_311180

-- Define the system of equations
def equation1 (x y z : ℝ) : Prop := 6*x - 5*y + 3*z = 22
def equation2 (x y z : ℝ) : Prop := 4*x + 8*y - 11*z = 7
def equation3 (x y z : ℝ) : Prop := 5*x - y + 2*z = 12/6

-- Define the sum condition
def sum_condition (x y z : ℝ) : Prop := x + y + z = 10

-- Theorem statement
theorem coefficient_of_y_in_equation3 (x y z : ℝ) 
  (eq1 : equation1 x y z) 
  (eq2 : equation2 x y z) 
  (eq3 : equation3 x y z) 
  (sum : sum_condition x y z) : 
  ∃ (a b c : ℝ), equation3 x y z ↔ a*x + (-1)*y + c*z = b :=
sorry

end NUMINAMATH_CALUDE_coefficient_of_y_in_equation3_l3111_311180


namespace NUMINAMATH_CALUDE_jason_oranges_l3111_311197

theorem jason_oranges (mary_oranges total_oranges : ℕ)
  (h1 : mary_oranges = 14)
  (h2 : total_oranges = 55) :
  total_oranges - mary_oranges = 41 := by
  sorry

end NUMINAMATH_CALUDE_jason_oranges_l3111_311197


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3111_311131

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The common difference of an arithmetic sequence -/
def CommonDifference (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_a7 : a 7 = 25)
  (h_a4 : a 4 = 13) :
  ∃ d : ℝ, CommonDifference a d ∧ d = 4 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3111_311131


namespace NUMINAMATH_CALUDE_both_are_liars_l3111_311105

-- Define the possible types of islanders
inductive IslanderType
  | Knight
  | Liar

-- Define the islanders
def A : IslanderType := sorry
def B : IslanderType := sorry

-- Define A's statement
def A_statement : Prop := (A = IslanderType.Liar) ∧ (B ≠ IslanderType.Liar)

-- Define the truth-telling property of knights and liars
def tells_truth (i : IslanderType) (p : Prop) : Prop :=
  (i = IslanderType.Knight ∧ p) ∨ (i = IslanderType.Liar ∧ ¬p)

-- Theorem to prove
theorem both_are_liars :
  tells_truth A A_statement →
  A = IslanderType.Liar ∧ B = IslanderType.Liar :=
by sorry

end NUMINAMATH_CALUDE_both_are_liars_l3111_311105


namespace NUMINAMATH_CALUDE_meteorologist_more_reliable_l3111_311191

/-- Probability of a clear day -/
def p_clear : ℝ := 0.74

/-- Accuracy of a senator's forecast -/
def p_senator_accuracy : ℝ := sorry

/-- Accuracy of the meteorologist's forecast -/
def p_meteorologist_accuracy : ℝ := 1.5 * p_senator_accuracy

/-- Event that the day is clear -/
def G : Prop := sorry

/-- Event that the first senator predicts a clear day -/
def M₁ : Prop := sorry

/-- Event that the second senator predicts a clear day -/
def M₂ : Prop := sorry

/-- Event that the meteorologist predicts a rainy day -/
def S : Prop := sorry

/-- Probability of an event -/
noncomputable def P : Prop → ℝ := sorry

/-- Conditional probability -/
noncomputable def P_cond (A B : Prop) : ℝ := P (A ∧ B) / P B

theorem meteorologist_more_reliable :
  P_cond (¬G) (S ∧ M₁ ∧ M₂) > P_cond G (S ∧ M₁ ∧ M₂) :=
sorry

end NUMINAMATH_CALUDE_meteorologist_more_reliable_l3111_311191


namespace NUMINAMATH_CALUDE_circle_radius_constant_l3111_311163

theorem circle_radius_constant (c : ℝ) : 
  (∀ x y : ℝ, x^2 + 8*x + y^2 + 2*y + c = 0 ↔ (x + 4)^2 + (y + 1)^2 = 5^2) → 
  c = 42 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_constant_l3111_311163


namespace NUMINAMATH_CALUDE_students_without_A_l3111_311120

theorem students_without_A (total_students : ℕ) (chemistry_A : ℕ) (physics_A : ℕ) (both_A : ℕ) :
  total_students = 40 →
  chemistry_A = 10 →
  physics_A = 18 →
  both_A = 5 →
  total_students - (chemistry_A + physics_A - both_A) = 17 :=
by
  sorry

end NUMINAMATH_CALUDE_students_without_A_l3111_311120


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l3111_311169

/-- Triangle ABC with vertices A(0,2), B(2,0), and C(-2,-1) -/
structure Triangle where
  A : Prod ℝ ℝ := (0, 2)
  B : Prod ℝ ℝ := (2, 0)
  C : Prod ℝ ℝ := (-2, -1)

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Theorem about the properties of triangle ABC -/
theorem triangle_abc_properties (t : Triangle) :
  ∃ (l : LineEquation) (area : ℝ),
    -- The line equation of height AH
    (l.a = 4 ∧ l.b = 1 ∧ l.c = -2) ∧
    -- The area of triangle ABC
    area = 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l3111_311169


namespace NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l3111_311178

theorem smallest_three_digit_multiple_of_17 :
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n → 102 ≤ n :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l3111_311178


namespace NUMINAMATH_CALUDE_probability_of_white_ball_l3111_311186

/-- Given a box with white and black balls, calculate the probability of drawing a white ball -/
theorem probability_of_white_ball (white_balls black_balls : ℕ) : 
  white_balls = 5 → black_balls = 6 → 
  (white_balls : ℚ) / (white_balls + black_balls : ℚ) = 5 / 11 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_white_ball_l3111_311186


namespace NUMINAMATH_CALUDE_football_team_progress_l3111_311102

theorem football_team_progress (lost_yards gained_yards : ℤ) (h1 : lost_yards = 5) (h2 : gained_yards = 7) :
  gained_yards - lost_yards = 2 := by
  sorry

end NUMINAMATH_CALUDE_football_team_progress_l3111_311102


namespace NUMINAMATH_CALUDE_triangle_area_proof_l3111_311175

/-- The area of a triangular region bounded by the x-axis, y-axis, and the line 3x + 2y = 12 -/
def triangleArea : ℝ := 12

/-- The equation of the line bounding the triangular region -/
def lineEquation (x y : ℝ) : Prop := 3 * x + 2 * y = 12

/-- The x-intercept of the line -/
def xIntercept : ℝ := 4

/-- The y-intercept of the line -/
def yIntercept : ℝ := 6

theorem triangle_area_proof :
  lineEquation xIntercept 0 ∧
  lineEquation 0 yIntercept ∧
  triangleArea = (1 / 2) * xIntercept * yIntercept :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_area_proof_l3111_311175


namespace NUMINAMATH_CALUDE_election_winner_votes_l3111_311130

theorem election_winner_votes (total_votes : ℕ) 
  (h1 : total_votes > 0)
  (h2 : (3 * total_votes) / 4 - total_votes / 4 = 500) : 
  (3 * total_votes) / 4 = 750 := by
  sorry

end NUMINAMATH_CALUDE_election_winner_votes_l3111_311130


namespace NUMINAMATH_CALUDE_cone_base_radius_l3111_311126

/-- Given a sector paper with radius 30 cm and central angle 120°,
    when used to form the lateral surface of a cone,
    the radius of the base of the cone is 10 cm. -/
theorem cone_base_radius (R : ℝ) (θ : ℝ) (r : ℝ) : 
  R = 30 → θ = 120 → 2 * π * r = (θ / 360) * 2 * π * R → r = 10 := by
  sorry

end NUMINAMATH_CALUDE_cone_base_radius_l3111_311126


namespace NUMINAMATH_CALUDE_geometric_sequence_characterization_l3111_311173

def is_geometric_sequence (a : ℕ → ℚ) : Prop :=
  ∃ r : ℚ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_characterization (a : ℕ → ℚ) :
  is_geometric_sequence a ↔ 
  (∃ r : ℚ, ∀ n : ℕ, a (n + 1) = r * a n) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_characterization_l3111_311173


namespace NUMINAMATH_CALUDE_mike_toy_expenses_l3111_311121

theorem mike_toy_expenses : 
  let marbles_cost : ℚ := 9.05
  let football_cost : ℚ := 4.95
  let baseball_cost : ℚ := 6.52
  marbles_cost + football_cost + baseball_cost = 20.52 := by
sorry

end NUMINAMATH_CALUDE_mike_toy_expenses_l3111_311121


namespace NUMINAMATH_CALUDE_division_problem_l3111_311137

theorem division_problem (L S Q : ℕ) : 
  L - S = 1500 → 
  L = 1782 → 
  L = S * Q + 15 → 
  Q = 6 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l3111_311137


namespace NUMINAMATH_CALUDE_no_savings_on_joint_purchase_l3111_311193

/-- Calculates the cost of windows under a "buy 3, get 1 free" offer -/
def windowCost (regularPrice : ℕ) (quantity : ℕ) : ℕ :=
  ((quantity + 3) / 4 * 3) * regularPrice

/-- Proves that buying windows together does not save money under the given offer -/
theorem no_savings_on_joint_purchase (regularPrice : ℕ) :
  windowCost regularPrice 19 = windowCost regularPrice 9 + windowCost regularPrice 10 :=
by sorry

end NUMINAMATH_CALUDE_no_savings_on_joint_purchase_l3111_311193


namespace NUMINAMATH_CALUDE_abs_equation_one_l3111_311199

theorem abs_equation_one (x : ℝ) : |3*x - 5| + 4 = 8 ↔ x = 3 ∨ x = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_abs_equation_one_l3111_311199


namespace NUMINAMATH_CALUDE_cody_marbles_l3111_311162

theorem cody_marbles (initial_marbles : ℕ) : 
  (initial_marbles - initial_marbles / 3 - 5 = 7) → initial_marbles = 18 := by
  sorry

end NUMINAMATH_CALUDE_cody_marbles_l3111_311162


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3111_311158

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence with a_3 = 2 and a_6 = 5, prove a_9 = 8 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
    (h_arith : arithmetic_sequence a) 
    (h_3 : a 3 = 2) 
    (h_6 : a 6 = 5) : 
  a 9 = 8 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3111_311158


namespace NUMINAMATH_CALUDE_car_speed_problem_l3111_311110

/-- Proves that given a 6-hour trip where the average speed for the first 4 hours is 35 mph
    and the average speed for the entire trip is 38 mph, the average speed for the remaining 2 hours is 44 mph. -/
theorem car_speed_problem (total_time : ℝ) (initial_time : ℝ) (initial_speed : ℝ) (total_avg_speed : ℝ) :
  total_time = 6 →
  initial_time = 4 →
  initial_speed = 35 →
  total_avg_speed = 38 →
  let remaining_time := total_time - initial_time
  let total_distance := total_avg_speed * total_time
  let initial_distance := initial_speed * initial_time
  let remaining_distance := total_distance - initial_distance
  remaining_distance / remaining_time = 44 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l3111_311110


namespace NUMINAMATH_CALUDE_prime_sum_product_l3111_311188

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem prime_sum_product :
  ∃ p q : ℕ,
    is_prime p ∧
    is_prime q ∧
    p + q = 102 ∧
    (p > 30 ∨ q > 30) ∧
    p * q = 2201 :=
by sorry

end NUMINAMATH_CALUDE_prime_sum_product_l3111_311188


namespace NUMINAMATH_CALUDE_diamond_2_3_4_eq_zero_l3111_311135

/-- Definition of the diamond operation for real numbers -/
def diamond (a b c : ℝ) : ℝ := (b + 1)^2 - 4 * (a - 1) * c

/-- Theorem stating that diamond(2, 3, 4) equals 0 -/
theorem diamond_2_3_4_eq_zero : diamond 2 3 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_diamond_2_3_4_eq_zero_l3111_311135


namespace NUMINAMATH_CALUDE_negation_of_existence_proposition_l3111_311148

theorem negation_of_existence_proposition :
  (¬ ∃ x : ℝ, |x| + x^2 < 0) ↔ (∀ x : ℝ, |x| + x^2 ≥ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_proposition_l3111_311148


namespace NUMINAMATH_CALUDE_base_seven_54321_equals_13539_l3111_311177

def base_seven_to_ten (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

theorem base_seven_54321_equals_13539 :
  base_seven_to_ten [1, 2, 3, 4, 5] = 13539 := by
  sorry

end NUMINAMATH_CALUDE_base_seven_54321_equals_13539_l3111_311177


namespace NUMINAMATH_CALUDE_range_of_r_l3111_311198

noncomputable def r (x : ℝ) : ℝ := 1 / (1 - x)^3

theorem range_of_r :
  Set.range r = {y : ℝ | y < 0 ∨ y > 0} :=
by sorry

end NUMINAMATH_CALUDE_range_of_r_l3111_311198


namespace NUMINAMATH_CALUDE_solution_set_x_squared_less_than_one_l3111_311152

theorem solution_set_x_squared_less_than_one :
  {x : ℝ | x^2 < 1} = {x : ℝ | -1 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_solution_set_x_squared_less_than_one_l3111_311152


namespace NUMINAMATH_CALUDE_course_size_l3111_311106

theorem course_size (total : ℕ) 
  (h1 : total / 5 + total / 4 + total / 2 + 30 = total) : total = 600 := by
  sorry

end NUMINAMATH_CALUDE_course_size_l3111_311106


namespace NUMINAMATH_CALUDE_third_term_coefficient_equals_4860_l3111_311118

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Define the coefficient of the third term in the expansion of (3a+2b)^6
def third_term_coefficient : ℕ := 
  binomial 6 2 * (3^4) * (2^2)

-- Theorem statement
theorem third_term_coefficient_equals_4860 : third_term_coefficient = 4860 := by
  sorry

end NUMINAMATH_CALUDE_third_term_coefficient_equals_4860_l3111_311118


namespace NUMINAMATH_CALUDE_sum_squares_of_roots_l3111_311128

theorem sum_squares_of_roots (x₁ x₂ : ℝ) : 
  6 * x₁^2 + 11 * x₁ - 35 = 0 →
  6 * x₂^2 + 11 * x₂ - 35 = 0 →
  x₁ > 2 →
  x₂ > 2 →
  x₁^2 + x₂^2 = 541 / 36 := by
sorry

end NUMINAMATH_CALUDE_sum_squares_of_roots_l3111_311128


namespace NUMINAMATH_CALUDE_beth_candy_counts_l3111_311172

def possible_candy_counts (total : ℕ) (anne_min : ℕ) (beth_min : ℕ) (chris_min : ℕ) (chris_max : ℕ) : Set ℕ :=
  {b | ∃ (a c : ℕ), 
    a + b + c = total ∧ 
    a ≥ anne_min ∧ 
    b ≥ beth_min ∧ 
    c ≥ chris_min ∧ 
    c ≤ chris_max}

theorem beth_candy_counts : 
  possible_candy_counts 10 3 2 2 3 = {2, 3, 4, 5} := by
  sorry

end NUMINAMATH_CALUDE_beth_candy_counts_l3111_311172


namespace NUMINAMATH_CALUDE_max_value_cubic_expression_l3111_311144

theorem max_value_cubic_expression (a b : ℝ) (h : a^2 + b^2 = 1) :
  ∃ (max : ℝ), max = 1/4 ∧ ∀ (x y : ℝ), x^2 + y^2 = 1 → x^3 * y - y^3 * x ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_cubic_expression_l3111_311144


namespace NUMINAMATH_CALUDE_spherical_coordinates_reflection_l3111_311133

theorem spherical_coordinates_reflection :
  ∀ (x y z : ℝ),
  (∃ (ρ θ φ : ℝ),
    ρ = 4 ∧ θ = 5 * π / 6 ∧ φ = π / 4 ∧
    x = ρ * Real.sin φ * Real.cos θ ∧
    y = ρ * Real.sin φ * Real.sin θ ∧
    z = ρ * Real.cos φ) →
  (∃ (ρ' θ' φ' : ℝ),
    ρ' = 2 * Real.sqrt 10 ∧ θ' = 5 * π / 6 ∧ φ' = 3 * π / 4 ∧
    x = ρ' * Real.sin φ' * Real.cos θ' ∧
    y = ρ' * Real.sin φ' * Real.sin θ' ∧
    -z = ρ' * Real.cos φ' ∧
    ρ' > 0 ∧ 0 ≤ θ' ∧ θ' < 2 * π ∧ 0 ≤ φ' ∧ φ' ≤ π) :=
by sorry

end NUMINAMATH_CALUDE_spherical_coordinates_reflection_l3111_311133


namespace NUMINAMATH_CALUDE_pencils_purchased_correct_l3111_311115

/-- The number of pencils purchased given the conditions of the problem -/
def num_pencils : ℕ := 75

/-- The number of pens purchased -/
def num_pens : ℕ := 30

/-- The price of each pencil -/
def pencil_price : ℚ := 2

/-- The price of each pen -/
def pen_price : ℚ := 10

/-- The total cost of the purchase -/
def total_cost : ℚ := 450

/-- Theorem stating that the number of pencils purchased is correct given the conditions -/
theorem pencils_purchased_correct :
  (num_pens : ℚ) * pen_price + (num_pencils : ℚ) * pencil_price = total_cost :=
sorry

end NUMINAMATH_CALUDE_pencils_purchased_correct_l3111_311115


namespace NUMINAMATH_CALUDE_sum_of_numbers_with_lcm_and_ratio_l3111_311122

theorem sum_of_numbers_with_lcm_and_ratio 
  (a b : ℕ+) 
  (h_lcm : Nat.lcm a b = 60)
  (h_ratio : a * 3 = b * 2) : 
  a + b = 50 := by
sorry

end NUMINAMATH_CALUDE_sum_of_numbers_with_lcm_and_ratio_l3111_311122


namespace NUMINAMATH_CALUDE_a_range_l3111_311183

-- Define the propositions and variables
variable (p q : Prop)
variable (x a : ℝ)

-- Define the conditions
axiom x_range : 1/2 ≤ x ∧ x ≤ 1
axiom q_def : q ↔ (x - a) * (x - a - 1) ≤ 0
axiom not_p_necessary : ¬q → ¬p
axiom not_p_not_sufficient : ¬(¬p → ¬q)

-- State the theorem
theorem a_range : 0 ≤ a ∧ a ≤ 1/2 := by sorry

end NUMINAMATH_CALUDE_a_range_l3111_311183


namespace NUMINAMATH_CALUDE_smallest_with_200_divisors_l3111_311138

/-- The number of distinct positive divisors of n -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- Checks if n can be written as m * 10^k where 10 is not a divisor of m -/
def has_form (n m k : ℕ) : Prop :=
  n = m * (10 ^ k) ∧ ¬(10 ∣ m)

theorem smallest_with_200_divisors :
  ∃ (n m k : ℕ),
    (∀ i < n, num_divisors i < 200) ∧
    num_divisors n = 200 ∧
    has_form n m k ∧
    m + k = 18 := by sorry

end NUMINAMATH_CALUDE_smallest_with_200_divisors_l3111_311138


namespace NUMINAMATH_CALUDE_trees_to_plant_l3111_311153

theorem trees_to_plant (current_trees final_trees : ℕ) : 
  current_trees = 25 → final_trees = 98 → final_trees - current_trees = 73 := by
  sorry

end NUMINAMATH_CALUDE_trees_to_plant_l3111_311153


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l3111_311141

/-- The function f(x) = a^(x-1) + 2 has (1, 3) as a fixed point, where a > 0 and a ≠ 1 -/
theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (ha1 : a ≠ 1) :
  let f : ℝ → ℝ := λ x => a^(x - 1) + 2
  f 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l3111_311141


namespace NUMINAMATH_CALUDE_choose_four_from_eight_l3111_311114

theorem choose_four_from_eight : Nat.choose 8 4 = 70 := by
  sorry

end NUMINAMATH_CALUDE_choose_four_from_eight_l3111_311114


namespace NUMINAMATH_CALUDE_prob_art_second_given_pe_first_l3111_311107

def total_courses : ℕ := 6
def pe_courses : ℕ := 4
def art_courses : ℕ := 2

def prob_pe_first : ℚ := pe_courses / total_courses
def prob_art_second : ℚ := art_courses / (total_courses - 1)

theorem prob_art_second_given_pe_first :
  (prob_pe_first * prob_art_second) / prob_pe_first = 2 / 5 :=
by sorry

end NUMINAMATH_CALUDE_prob_art_second_given_pe_first_l3111_311107


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3111_311165

theorem inequality_solution_set : 
  {x : ℝ | 3 ≤ |2 - x| ∧ |2 - x| < 9} = {x : ℝ | -7 < x ∧ x ≤ -1 ∨ 5 ≤ x ∧ x < 11} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3111_311165


namespace NUMINAMATH_CALUDE_minimum_value_x_plus_reciprocal_l3111_311113

theorem minimum_value_x_plus_reciprocal (x : ℝ) (h : x > 1) :
  x + 1 / (x - 1) ≥ 3 ∧ ∃ x₀ > 1, x₀ + 1 / (x₀ - 1) = 3 :=
sorry


end NUMINAMATH_CALUDE_minimum_value_x_plus_reciprocal_l3111_311113


namespace NUMINAMATH_CALUDE_fruit_salad_cherries_l3111_311124

/-- Represents the composition of a fruit salad --/
structure FruitSalad where
  blueberries : ℕ
  raspberries : ℕ
  grapes : ℕ
  cherries : ℕ

/-- Checks if the fruit salad satisfies the given conditions --/
def isValidFruitSalad (fs : FruitSalad) : Prop :=
  fs.blueberries + fs.raspberries + fs.grapes + fs.cherries = 280 ∧
  fs.raspberries = 2 * fs.blueberries ∧
  fs.grapes = 3 * fs.cherries ∧
  fs.cherries = 4 * fs.raspberries

/-- Theorem stating that a valid fruit salad has 64 cherries --/
theorem fruit_salad_cherries (fs : FruitSalad) :
  isValidFruitSalad fs → fs.cherries = 64 := by
  sorry

#check fruit_salad_cherries

end NUMINAMATH_CALUDE_fruit_salad_cherries_l3111_311124


namespace NUMINAMATH_CALUDE_triangle_part1_triangle_part2_l3111_311108

noncomputable section

-- Define the triangle ABC
variable (A B C : ℝ) -- Angles
variable (a b c : ℝ) -- Sides

-- Part 1
theorem triangle_part1 (h1 : a = Real.sqrt 3) (h2 : b = Real.sqrt 2) (h3 : B = π / 4) :
  ((A = π / 3 ∧ C = 5 * π / 12 ∧ c = (Real.sqrt 6 + Real.sqrt 2) / 2) ∨
   (A = 2 * π / 3 ∧ C = π / 12 ∧ c = (Real.sqrt 6 - Real.sqrt 2) / 2)) :=
sorry

-- Part 2
theorem triangle_part2 (h1 : Real.cos B / Real.cos C = -b / (2 * a + c)) 
                       (h2 : b = Real.sqrt 13) (h3 : a + c = 4) :
  (1 / 2) * a * c * Real.sin B = 3 * Real.sqrt 3 / 4 :=
sorry

end NUMINAMATH_CALUDE_triangle_part1_triangle_part2_l3111_311108


namespace NUMINAMATH_CALUDE_square_sum_identity_l3111_311100

theorem square_sum_identity (x : ℝ) : (x + 1)^2 + 2*(x + 1)*(3 - x) + (3 - x)^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_identity_l3111_311100


namespace NUMINAMATH_CALUDE_tangent_line_and_range_of_a_l3111_311119

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x + 1) * Real.log x - a * (x - 1)

theorem tangent_line_and_range_of_a :
  (∃ (m b : ℝ), ∀ x, (f 4) x = m * (x - 1) + (f 4) 1 → m = -2 ∧ b = 2) ∧
  (∀ a, (∀ x, x > 1 → f a x > 0) ↔ a ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_and_range_of_a_l3111_311119


namespace NUMINAMATH_CALUDE_expression_simplification_l3111_311147

theorem expression_simplification (m n : ℚ) 
  (hm : m = 2) 
  (hn : n = -1/2) : 
  3 * (m^2 - m + n^2) - 2 * (1/2 * m^2 - m*n + 3/2 * n^2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3111_311147


namespace NUMINAMATH_CALUDE_congruence_problem_l3111_311160

theorem congruence_problem (x : ℤ) : 
  (4 * x + 9) % 19 = 3 → (3 * x + 8) % 19 = 13 := by
sorry

end NUMINAMATH_CALUDE_congruence_problem_l3111_311160


namespace NUMINAMATH_CALUDE_jackson_gpa_probability_l3111_311167

-- Define the point values for each grade
def pointA : ℚ := 5
def pointB : ℚ := 4
def pointC : ℚ := 2
def pointD : ℚ := 1

-- Define the probabilities for Literature grades
def litProbA : ℚ := 1/5
def litProbB : ℚ := 2/5
def litProbC : ℚ := 2/5

-- Define the probabilities for Sociology grades
def socProbA : ℚ := 1/3
def socProbB : ℚ := 1/2
def socProbC : ℚ := 1/6

-- Define the number of classes
def numClasses : ℕ := 5

-- Define the minimum GPA required
def minGPA : ℚ := 4

-- Define the function to calculate GPA
def calculateGPA (points : ℚ) : ℚ := points / numClasses

-- Theorem statement
theorem jackson_gpa_probability :
  let confirmedPoints : ℚ := pointA + pointA  -- Calculus and Physics
  let minRemainingPoints : ℚ := minGPA * numClasses - confirmedPoints
  let probTwoAs : ℚ := litProbA * socProbA
  let probALitBSoc : ℚ := litProbA * socProbB
  let probASocBLit : ℚ := socProbA * litProbB
  (probTwoAs + probALitBSoc + probASocBLit) = 2/5 := by sorry

end NUMINAMATH_CALUDE_jackson_gpa_probability_l3111_311167


namespace NUMINAMATH_CALUDE_no_integer_solution_for_x2_plus_y2_eq_3z2_l3111_311150

theorem no_integer_solution_for_x2_plus_y2_eq_3z2 :
  ¬ ∃ (x y z : ℤ), x^2 + y^2 = 3 * z^2 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_for_x2_plus_y2_eq_3z2_l3111_311150
