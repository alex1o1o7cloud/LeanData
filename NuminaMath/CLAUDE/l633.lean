import Mathlib

namespace NUMINAMATH_CALUDE_arctan_equation_solution_l633_63305

theorem arctan_equation_solution :
  ∃ x : ℝ, 2 * Real.arctan (1/3) + Real.arctan (1/5) + Real.arctan (1/x) = π/4 ∧ x = 21/2 := by
  sorry

end NUMINAMATH_CALUDE_arctan_equation_solution_l633_63305


namespace NUMINAMATH_CALUDE_evaluate_expression_l633_63335

theorem evaluate_expression : 103^4 - 4 * 103^3 + 6 * 103^2 - 4 * 103 + 1 = 108243216 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l633_63335


namespace NUMINAMATH_CALUDE_special_polynomial_value_at_one_l633_63303

/-- A non-constant quadratic polynomial satisfying the given equation -/
def SpecialPolynomial (g : ℝ → ℝ) : Prop :=
  (∃ a b c : ℝ, a ≠ 0 ∧ ∀ x : ℝ, g x = a * x^2 + b * x + c) ∧
  (∀ x : ℝ, x ≠ 0 → g (x - 1) + g x + g (x + 1) = (g x)^2 / (2023 * x))

theorem special_polynomial_value_at_one
  (g : ℝ → ℝ) (h : SpecialPolynomial g) : g 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_special_polynomial_value_at_one_l633_63303


namespace NUMINAMATH_CALUDE_equation_solution_l633_63319

theorem equation_solution : 
  ∃ x : ℚ, (2*x - 30) / 3 = (5 - 3*x) / 4 + 1 ∧ x = 147 / 17 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l633_63319


namespace NUMINAMATH_CALUDE_four_collinear_points_l633_63373

open Real

-- Define the curve
def curve (α : ℝ) (x : ℝ) : ℝ := x^4 + 9*x^3 + α*x^2 + 9*x + 4

-- Define the second derivative of the curve
def second_derivative (α : ℝ) (x : ℝ) : ℝ := 12*x^2 + 54*x + 2*α

-- Theorem statement
theorem four_collinear_points (α : ℝ) :
  (∃ x₁ x₂ x₃ x₄ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    (∃ a b : ℝ, curve α x₁ = a*x₁ + b ∧ 
                curve α x₂ = a*x₂ + b ∧ 
                curve α x₃ = a*x₃ + b ∧ 
                curve α x₄ = a*x₄ + b)) ↔
  α < 30.375 :=
by sorry

end NUMINAMATH_CALUDE_four_collinear_points_l633_63373


namespace NUMINAMATH_CALUDE_complement_of_union_l633_63391

def U : Set Int := {-2, -1, 0, 1, 2, 3}
def A : Set Int := {-1, 0, 1}
def B : Set Int := {1, 2}

theorem complement_of_union :
  (U \ (A ∪ B)) = {-2, 3} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_l633_63391


namespace NUMINAMATH_CALUDE_mary_cut_ten_roses_l633_63378

/-- The number of roses Mary cut from her garden -/
def roses_cut : ℕ := 16 - 6

/-- Theorem stating that Mary cut 10 roses -/
theorem mary_cut_ten_roses : roses_cut = 10 := by
  sorry

end NUMINAMATH_CALUDE_mary_cut_ten_roses_l633_63378


namespace NUMINAMATH_CALUDE_total_interest_is_860_l633_63344

def inheritance : ℝ := 12000
def investment1 : ℝ := 5000
def rate1 : ℝ := 0.06
def rate2 : ℝ := 0.08

def total_interest : ℝ :=
  investment1 * rate1 + (inheritance - investment1) * rate2

theorem total_interest_is_860 : total_interest = 860 := by sorry

end NUMINAMATH_CALUDE_total_interest_is_860_l633_63344


namespace NUMINAMATH_CALUDE_sequence_relations_l633_63301

theorem sequence_relations (x y : ℕ → ℝ) 
  (h1 : ∀ k, y k = (x (k + 1) - 3 * x k) / 2)
  (h2 : ∀ k, x k = (y (k + 1) - 3 * y k) / 4) :
  (∀ k, x k = 6 * x (k - 1) - x (k - 2)) ∧
  (∀ k, x k = 34 * x (k - 2) - x (k - 4)) ∧
  (∀ k, x k = 198 * x (k - 3) - x (k - 6)) ∧
  (∀ k, y k = 6 * y (k - 1) - y (k - 2)) ∧
  (∀ k, y k = 34 * y (k - 2) - y (k - 4)) ∧
  (∀ k, y k = 198 * y (k - 3) - y (k - 6)) := by
  sorry

end NUMINAMATH_CALUDE_sequence_relations_l633_63301


namespace NUMINAMATH_CALUDE_complex_equation_solution_l633_63368

theorem complex_equation_solution (a : ℝ) : (a + Complex.I)^2 = 2 * Complex.I → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l633_63368


namespace NUMINAMATH_CALUDE_square_last_two_digits_averages_l633_63384

def last_two_digits (n : ℕ) : ℕ := n % 100

def valid_pair (a b : ℕ) : Prop :=
  a ≠ b ∧ 0 < a ∧ a < 50 ∧ 0 < b ∧ b < 50 ∧ last_two_digits (a^2) = last_two_digits (b^2)

def average (a b : ℕ) : ℚ := (a + b : ℚ) / 2

theorem square_last_two_digits_averages :
  {x : ℚ | ∃ a b : ℕ, valid_pair a b ∧ average a b = x} = {10, 15, 20, 25, 30, 35, 40} := by sorry

end NUMINAMATH_CALUDE_square_last_two_digits_averages_l633_63384


namespace NUMINAMATH_CALUDE_cosine_function_phi_range_l633_63377

/-- The cosine function -/
noncomputable def f (ω : ℝ) (φ : ℝ) (x : ℝ) : ℝ := 2 * Real.cos (ω * x + φ) + 1

/-- The theorem statement -/
theorem cosine_function_phi_range 
  (ω : ℝ) 
  (φ : ℝ) 
  (h_ω : ω > 0) 
  (h_φ : |φ| < π/2) 
  (h_period : ∃ (x₁ x₂ : ℝ), x₂ - x₁ = 2*π/3 ∧ f ω φ x₁ = 3 ∧ f ω φ x₂ = 3)
  (h_range : ∀ x ∈ Set.Ioo (-π/12) (π/6), f ω φ x > 1) :
  φ ∈ Set.Icc (-π/4) 0 :=
sorry

end NUMINAMATH_CALUDE_cosine_function_phi_range_l633_63377


namespace NUMINAMATH_CALUDE_orthocenter_centroid_angle_tangent_sum_l633_63358

/-- A triangle with vertices A, B, and C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The orthocenter of a triangle -/
def orthocenter (t : Triangle) : ℝ × ℝ := sorry

/-- The centroid of a triangle -/
def centroid (t : Triangle) : ℝ × ℝ := sorry

/-- Angle between three points -/
def angle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

/-- Tangent of an angle -/
def tan (θ : ℝ) : ℝ := sorry

/-- Predicate to check if a triangle is right-angled -/
def isRightTriangle (t : Triangle) : Prop := sorry

/-- Predicate to check if a triangle is isosceles -/
def isIsoscelesTriangle (t : Triangle) : Prop := sorry

theorem orthocenter_centroid_angle_tangent_sum (t : Triangle) :
  ¬(isRightTriangle t) → ¬(isIsoscelesTriangle t) →
  let M := orthocenter t
  let S := centroid t
  let θA := angle t.A M S
  let θB := angle t.B M S
  let θC := angle t.C M S
  (tan θA = tan θB + tan θC) ∨ (tan θB = tan θA + tan θC) ∨ (tan θC = tan θA + tan θB) := by
  sorry

end NUMINAMATH_CALUDE_orthocenter_centroid_angle_tangent_sum_l633_63358


namespace NUMINAMATH_CALUDE_a_1995_equals_3_l633_63357

def units_digit (n : ℕ) : ℕ := n % 10

def a (n : ℕ) : ℕ := units_digit (7^n)

theorem a_1995_equals_3 : a 1995 = 3 := by sorry

end NUMINAMATH_CALUDE_a_1995_equals_3_l633_63357


namespace NUMINAMATH_CALUDE_cos_two_alpha_l633_63343

theorem cos_two_alpha (α : ℝ) (h : Real.sin (π / 6 - α) = 1 / 3) :
  Real.cos (2 * π / 3 + 2 * α) = -7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cos_two_alpha_l633_63343


namespace NUMINAMATH_CALUDE_max_value_of_exponential_difference_l633_63317

theorem max_value_of_exponential_difference :
  ∃ (max : ℝ), max = 2/3 ∧ ∀ (x : ℝ), 2^x - 8^x ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_exponential_difference_l633_63317


namespace NUMINAMATH_CALUDE_valid_seating_arrangements_l633_63371

/-- The number of Democrats in the Senate committee -/
def num_democrats : ℕ := 6

/-- The number of Republicans in the Senate committee -/
def num_republicans : ℕ := 4

/-- The total number of politicians in the Senate committee -/
def total_politicians : ℕ := num_democrats + num_republicans

/-- The number of gaps between Democrats where Republicans can be placed -/
def num_gaps : ℕ := num_democrats

/-- Function to calculate the number of valid seating arrangements -/
def seating_arrangements (d r : ℕ) : ℕ :=
  (Nat.factorial (d - 1)) * (Nat.choose d r) * (Nat.factorial r)

/-- Theorem stating the number of valid seating arrangements -/
theorem valid_seating_arrangements :
  seating_arrangements num_democrats num_republicans = 43200 := by
  sorry

end NUMINAMATH_CALUDE_valid_seating_arrangements_l633_63371


namespace NUMINAMATH_CALUDE_coin_value_difference_l633_63381

/-- Represents the number of coins of each type -/
structure CoinCount where
  pennies : Nat
  nickels : Nat
  dimes : Nat

/-- Calculates the total value in cents for a given coin count -/
def totalValue (coins : CoinCount) : Nat :=
  coins.pennies + 5 * coins.nickels + 10 * coins.dimes

/-- Represents the constraint that the total number of coins is 3030 -/
def totalCoins (coins : CoinCount) : Prop :=
  coins.pennies + coins.nickels + coins.dimes = 3030

/-- Represents the constraint that there are at least 10 of each coin type -/
def atLeastTenEach (coins : CoinCount) : Prop :=
  coins.pennies ≥ 10 ∧ coins.nickels ≥ 10 ∧ coins.dimes ≥ 10

/-- The main theorem stating the difference between max and min possible values -/
theorem coin_value_difference :
  ∃ (max min : CoinCount),
    totalCoins max ∧ totalCoins min ∧
    atLeastTenEach max ∧ atLeastTenEach min ∧
    (∀ c, totalCoins c → atLeastTenEach c → totalValue c ≤ totalValue max) ∧
    (∀ c, totalCoins c → atLeastTenEach c → totalValue c ≥ totalValue min) ∧
    totalValue max - totalValue min = 27000 := by
  sorry

end NUMINAMATH_CALUDE_coin_value_difference_l633_63381


namespace NUMINAMATH_CALUDE_tan_equality_with_range_l633_63365

theorem tan_equality_with_range (n : ℤ) : 
  -90 < n ∧ n < 90 ∧ Real.tan (n * π / 180) = Real.tan (850 * π / 180) → n = -50 := by
  sorry

end NUMINAMATH_CALUDE_tan_equality_with_range_l633_63365


namespace NUMINAMATH_CALUDE_inequality_of_five_variables_l633_63369

theorem inequality_of_five_variables (a b c d e : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) : 
  Real.sqrt (a / (b + c + d + e)) + 
  Real.sqrt (b / (a + c + d + e)) + 
  Real.sqrt (c / (a + b + d + e)) + 
  Real.sqrt (d / (a + b + c + e)) + 
  Real.sqrt (e / (a + b + c + d)) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_of_five_variables_l633_63369


namespace NUMINAMATH_CALUDE_division_problem_l633_63348

theorem division_problem (x y : ℕ+) 
  (h1 : (x : ℝ) / (y : ℝ) = 96.12)
  (h2 : (x : ℝ) % (y : ℝ) = 1.44) : 
  y = 12 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l633_63348


namespace NUMINAMATH_CALUDE_scholarship_fund_scientific_notation_l633_63328

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a positive real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scholarship_fund_scientific_notation :
  toScientificNotation 445800000 = ScientificNotation.mk 4.458 8 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_scholarship_fund_scientific_notation_l633_63328


namespace NUMINAMATH_CALUDE_fiftieth_term_is_247_l633_63394

/-- The nth term of an arithmetic sequence -/
def arithmetic_sequence (a : ℤ) (d : ℤ) (n : ℕ) : ℤ := a + (n - 1 : ℤ) * d

theorem fiftieth_term_is_247 : 
  arithmetic_sequence 2 5 50 = 247 := by sorry

end NUMINAMATH_CALUDE_fiftieth_term_is_247_l633_63394


namespace NUMINAMATH_CALUDE_movie_shelf_distribution_l633_63311

theorem movie_shelf_distribution (n : ℕ) : 
  (∃ k : ℕ, n + 1 = 2 * k) → Odd n := by
  sorry

end NUMINAMATH_CALUDE_movie_shelf_distribution_l633_63311


namespace NUMINAMATH_CALUDE_money_distribution_l633_63359

theorem money_distribution (a b c d : ℚ) : 
  a = (1 : ℚ) / 3 * (b + c + d) →
  b = (2 : ℚ) / 7 * (a + c + d) →
  c = (3 : ℚ) / 11 * (a + b + d) →
  a = b + 20 →
  b = c + 15 →
  a + b + c + d = 720 := by
  sorry

end NUMINAMATH_CALUDE_money_distribution_l633_63359


namespace NUMINAMATH_CALUDE_triangle_property_l633_63379

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if (a+c)/b = cos(C) + √3*sin(C), then B = 60° and when b = 2, the max area is √3 -/
theorem triangle_property (a b c : ℝ) (A B C : Real) : 
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  (a + c) / b = Real.cos C + Real.sqrt 3 * Real.sin C →
  B = π / 3 ∧ 
  (b = 2 → ∃ (area : ℝ), area ≤ Real.sqrt 3 ∧ 
    ∀ (other_area : ℝ), (∃ (a' c' : ℝ), a' > 0 ∧ c' > 0 ∧ 
      other_area = 1/2 * a' * 2 * Real.sin (π/3)) → other_area ≤ area) := by
  sorry

end NUMINAMATH_CALUDE_triangle_property_l633_63379


namespace NUMINAMATH_CALUDE_summer_program_students_l633_63315

theorem summer_program_students : ∃! n : ℕ, 0 < n ∧ n < 500 ∧ n % 25 = 24 ∧ n % 21 = 14 ∧ n = 449 := by
  sorry

end NUMINAMATH_CALUDE_summer_program_students_l633_63315


namespace NUMINAMATH_CALUDE_square_area_side_perimeter_l633_63350

theorem square_area_side_perimeter :
  ∀ (s p : ℝ),
  s > 0 →
  s^2 = 450 →
  p = 4 * s →
  s = 15 * Real.sqrt 2 ∧ p = 60 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_square_area_side_perimeter_l633_63350


namespace NUMINAMATH_CALUDE_relationship_abc_l633_63310

theorem relationship_abc : 
  let a : ℝ := Real.sqrt 2
  let b : ℝ := Real.sqrt 7 - Real.sqrt 3
  let c : ℝ := Real.sqrt 6 - Real.sqrt 2
  a > c ∧ c > b := by sorry

end NUMINAMATH_CALUDE_relationship_abc_l633_63310


namespace NUMINAMATH_CALUDE_rebecca_checkerboard_black_squares_l633_63393

/-- Represents a checkerboard with alternating colors -/
structure Checkerboard where
  size : ℕ
  is_black : ℕ → ℕ → Prop

/-- Defines the properties of Rebecca's checkerboard -/
def rebecca_checkerboard : Checkerboard where
  size := 29
  is_black := fun i j => (i + j) % 2 = 0

/-- Counts the number of black squares in a row -/
def black_squares_in_row (c : Checkerboard) (row : ℕ) : ℕ :=
  (c.size + 1) / 2

/-- Counts the total number of black squares on the checkerboard -/
def total_black_squares (c : Checkerboard) : ℕ :=
  c.size * ((c.size + 1) / 2)

/-- Theorem stating that Rebecca's checkerboard has 435 black squares -/
theorem rebecca_checkerboard_black_squares :
  total_black_squares rebecca_checkerboard = 435 := by
  sorry


end NUMINAMATH_CALUDE_rebecca_checkerboard_black_squares_l633_63393


namespace NUMINAMATH_CALUDE_probability_quarter_or_dime_l633_63399

/-- Represents the types of coins in the jar -/
inductive Coin
  | Quarter
  | Dime
  | Nickel

/-- The value of each coin type in cents -/
def coinValue : Coin → ℕ
  | Coin.Quarter => 25
  | Coin.Dime => 10
  | Coin.Nickel => 5

/-- The total value of each coin type in the jar in cents -/
def totalValue : Coin → ℕ
  | Coin.Quarter => 500
  | Coin.Dime => 600
  | Coin.Nickel => 200

/-- The number of coins of each type in the jar -/
def coinCount (c : Coin) : ℕ := totalValue c / coinValue c

/-- The total number of coins in the jar -/
def totalCoins : ℕ := coinCount Coin.Quarter + coinCount Coin.Dime + coinCount Coin.Nickel

/-- The probability of selecting either a quarter or a dime from the jar -/
def probQuarterOrDime : ℚ :=
  (coinCount Coin.Quarter + coinCount Coin.Dime : ℚ) / totalCoins

theorem probability_quarter_or_dime :
  probQuarterOrDime = 2 / 3 := by
  sorry


end NUMINAMATH_CALUDE_probability_quarter_or_dime_l633_63399


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l633_63349

theorem triangle_angle_measure (D E F : ℝ) : 
  D = 3 * E →  -- Angle D is thrice as large as angle E
  E = 18 →     -- Angle E measures 18°
  D + E + F = 180 →  -- Sum of angles in a triangle is 180°
  F = 108 :=   -- Angle F measures 108°
by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l633_63349


namespace NUMINAMATH_CALUDE_line_intercepts_sum_l633_63331

/-- Given a line with equation y - 5 = 3(x - 9), the sum of its x-intercept and y-intercept is -44/3 -/
theorem line_intercepts_sum (x y : ℝ) : 
  (y - 5 = 3 * (x - 9)) → 
  (∃ x_int y_int : ℝ, 
    (y_int - 5 = 3 * (x_int - 9)) ∧ 
    (0 - 5 = 3 * (x_int - 9)) ∧ 
    (y_int - 5 = 3 * (0 - 9)) ∧ 
    (x_int + y_int = -44/3)) := by
  sorry

end NUMINAMATH_CALUDE_line_intercepts_sum_l633_63331


namespace NUMINAMATH_CALUDE_stack_toppled_plates_l633_63352

/-- The number of plates in Alice's stack when it toppled over --/
def total_plates (initial_plates added_plates : ℕ) : ℕ :=
  initial_plates + added_plates

/-- Theorem: The total number of plates when the stack toppled is 64 --/
theorem stack_toppled_plates :
  total_plates 27 37 = 64 := by
  sorry

end NUMINAMATH_CALUDE_stack_toppled_plates_l633_63352


namespace NUMINAMATH_CALUDE_set_equality_implies_a_value_l633_63308

theorem set_equality_implies_a_value (A B : Set ℤ) (a : ℤ) : 
  A = {2, 3} → B = {2, 2*a - 1} → A = B → a = 2 := by sorry

end NUMINAMATH_CALUDE_set_equality_implies_a_value_l633_63308


namespace NUMINAMATH_CALUDE_library_books_loaned_l633_63336

theorem library_books_loaned (initial_books : ℕ) (return_rate : ℚ) (final_books : ℕ) 
  (h1 : initial_books = 75)
  (h2 : return_rate = 70 / 100)
  (h3 : final_books = 57) :
  ∃ (loaned_books : ℕ), loaned_books = 60 ∧ 
    initial_books - (↑loaned_books * (1 - return_rate)).floor = final_books :=
by sorry

end NUMINAMATH_CALUDE_library_books_loaned_l633_63336


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l633_63341

/-- An isosceles triangle with sides 4 and 9 has a perimeter of 22 -/
theorem isosceles_triangle_perimeter : ∀ a b c : ℝ,
  a = 4 ∧ b = 9 ∧ c = 9 →  -- Two sides are 9, one side is 4
  a < b + c ∧ b < a + c ∧ c < a + b →  -- Triangle inequality
  a + b + c = 22 :=  -- Perimeter is 22
by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l633_63341


namespace NUMINAMATH_CALUDE_distributive_analogy_l633_63322

theorem distributive_analogy (a b c : ℝ) (h : c ≠ 0) :
  ((a + b) * c = a * c + b * c) ↔ ((a + b) / c = a / c + b / c) :=
sorry

end NUMINAMATH_CALUDE_distributive_analogy_l633_63322


namespace NUMINAMATH_CALUDE_expand_staircase_4_to_7_l633_63395

/-- Calculates the number of toothpicks needed for a staircase with n steps -/
def toothpicks (n : ℕ) : ℕ :=
  if n = 0 then 0
  else 2 * n * n + 2 * n

/-- The number of additional toothpicks needed to expand from m steps to n steps -/
def additional_toothpicks (m n : ℕ) : ℕ :=
  toothpicks n - toothpicks m

theorem expand_staircase_4_to_7 :
  additional_toothpicks 4 7 = 48 := by
  sorry

#eval additional_toothpicks 4 7

end NUMINAMATH_CALUDE_expand_staircase_4_to_7_l633_63395


namespace NUMINAMATH_CALUDE_m_range_for_third_quadrant_l633_63392

/-- A complex number z is in the third quadrant if its real and imaginary parts are both negative -/
def in_third_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im < 0

/-- The theorem stating that if z = (m+4) + (m-2)i is in the third quadrant, 
    then m is in the interval (-∞, -4) -/
theorem m_range_for_third_quadrant (m : ℝ) :
  let z : ℂ := Complex.mk (m + 4) (m - 2)
  in_third_quadrant z → m < -4 := by
  sorry

#check m_range_for_third_quadrant

end NUMINAMATH_CALUDE_m_range_for_third_quadrant_l633_63392


namespace NUMINAMATH_CALUDE_banquet_solution_l633_63347

def banquet_problem (total_attendees : ℕ) (resident_price : ℚ) (total_revenue : ℚ) (num_residents : ℕ) : ℚ :=
  let non_residents := total_attendees - num_residents
  let resident_revenue := num_residents * resident_price
  let non_resident_revenue := total_revenue - resident_revenue
  non_resident_revenue / non_residents

theorem banquet_solution :
  banquet_problem 586 12.95 9423.70 219 = 17.95 := by
  sorry

end NUMINAMATH_CALUDE_banquet_solution_l633_63347


namespace NUMINAMATH_CALUDE_inequality_solution_l633_63383

theorem inequality_solution (n k : ℤ) :
  let x : ℝ := (-1)^n * π/6 + 2*π*n
  let y : ℝ := π/2 + π*k
  4 * Real.sin x - Real.sqrt (Real.cos y) - Real.sqrt (Real.cos y - 16 * (Real.cos x)^2 + 12) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l633_63383


namespace NUMINAMATH_CALUDE_not_prime_1000000027_l633_63366

theorem not_prime_1000000027 : ¬ Nat.Prime 1000000027 := by
  sorry

end NUMINAMATH_CALUDE_not_prime_1000000027_l633_63366


namespace NUMINAMATH_CALUDE_multiplicative_inverse_600_mod_4901_l633_63398

theorem multiplicative_inverse_600_mod_4901 :
  ∃ n : ℕ, n < 4901 ∧ (600 * n) % 4901 = 1 ∧ n = 3196 := by sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_600_mod_4901_l633_63398


namespace NUMINAMATH_CALUDE_complex_product_l633_63300

theorem complex_product (z₁ z₂ : ℂ) 
  (h1 : Complex.abs z₁ = 2) 
  (h2 : Complex.abs z₂ = 3) 
  (h3 : 3 * z₁ - 2 * z₂ = 2 - Complex.I) : 
  z₁ * z₂ = -18/5 + 24/5 * Complex.I := by
sorry

end NUMINAMATH_CALUDE_complex_product_l633_63300


namespace NUMINAMATH_CALUDE_y_equals_sixteen_l633_63339

/-- The star operation defined as a ★ b = 4a - b -/
def star (a b : ℝ) : ℝ := 4 * a - b

/-- Theorem stating that y = 16 satisfies the equation 3 ★ (6 ★ y) = 4 -/
theorem y_equals_sixteen : ∃ y : ℝ, star 3 (star 6 y) = 4 ∧ y = 16 := by
  sorry

end NUMINAMATH_CALUDE_y_equals_sixteen_l633_63339


namespace NUMINAMATH_CALUDE_parallel_planes_false_l633_63323

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (belongs_to : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the planes and lines
variable (α β : Plane)
variable (m n : Line)

-- State the theorem
theorem parallel_planes_false :
  (¬ (α = β)) →  -- α and β are non-coincident planes
  (¬ (m = n)) →  -- m and n are non-coincident lines
  ¬ (
    (belongs_to m α ∧ belongs_to n α ∧ 
     parallel_line_plane m β ∧ parallel_line_plane n β) → 
    (parallel α β)
  ) := by sorry

end NUMINAMATH_CALUDE_parallel_planes_false_l633_63323


namespace NUMINAMATH_CALUDE_remy_water_usage_l633_63337

theorem remy_water_usage (roman : ℕ) (remy : ℕ) : 
  remy = 3 * roman + 1 →
  roman + remy = 33 →
  remy = 25 := by
sorry

end NUMINAMATH_CALUDE_remy_water_usage_l633_63337


namespace NUMINAMATH_CALUDE_remainder_3_100_plus_4_mod_5_l633_63364

theorem remainder_3_100_plus_4_mod_5 : (3^100 + 4) % 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3_100_plus_4_mod_5_l633_63364


namespace NUMINAMATH_CALUDE_composite_divisor_inequality_l633_63367

def d (k : ℕ) : ℕ := (Nat.divisors k).card

theorem composite_divisor_inequality (n : ℕ) (h_composite : ¬ Nat.Prime n) :
  ∃ m : ℕ, m > 0 ∧ m ∣ n ∧ m * m ≤ n ∧ d n ≤ d m * d m * d m := by
  sorry

end NUMINAMATH_CALUDE_composite_divisor_inequality_l633_63367


namespace NUMINAMATH_CALUDE_right_trapezoid_diagonals_l633_63362

/-- A right trapezoid with specific properties -/
structure RightTrapezoid where
  /-- Length of the smaller base -/
  small_base : ℝ
  /-- Length of the larger base -/
  large_base : ℝ
  /-- Angle at one vertex of the smaller base (in radians) -/
  angle_at_small_base : ℝ

/-- The diagonals of the trapezoid -/
def diagonals (t : RightTrapezoid) : ℝ × ℝ :=
  sorry

theorem right_trapezoid_diagonals :
  let t : RightTrapezoid := {
    small_base := 6,
    large_base := 8,
    angle_at_small_base := 2 * Real.pi / 3  -- 120° in radians
  }
  diagonals t = (4 * Real.sqrt 3, 2 * Real.sqrt 19) := by
  sorry

end NUMINAMATH_CALUDE_right_trapezoid_diagonals_l633_63362


namespace NUMINAMATH_CALUDE_min_ratio_two_digit_integers_l633_63354

theorem min_ratio_two_digit_integers (x y : ℕ) : 
  (10 ≤ x ∧ x ≤ 99) →  -- x is a two-digit positive integer
  (10 ≤ y ∧ y ≤ 99) →  -- y is a two-digit positive integer
  (x + y) / 2 = 75 →   -- mean of x and y is 75
  (∀ a b : ℕ, (10 ≤ a ∧ a ≤ 99) → (10 ≤ b ∧ b ≤ 99) → (a + b) / 2 = 75 → 
    x / (3 * y + 4 : ℚ) ≤ a / (3 * b + 4 : ℚ)) →
  x / (3 * y + 4 : ℚ) = 70 / 17 := by
sorry

end NUMINAMATH_CALUDE_min_ratio_two_digit_integers_l633_63354


namespace NUMINAMATH_CALUDE_equation_solution_l633_63330

theorem equation_solution : ∃! x : ℚ, (3 - x) / (x + 2) + (3*x - 9) / (3 - x) = 2 ∧ x = -7/6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l633_63330


namespace NUMINAMATH_CALUDE_function_range_l633_63397

noncomputable def f (x : ℝ) : ℝ := 1 / Real.exp x - Real.exp x + 2 * x - (1 / 3) * x^3

theorem function_range (a : ℝ) : f (3 * a^2) + f (2 * a - 1) ≥ 0 → a ∈ Set.Icc (-1) (1/3) :=
by sorry

end NUMINAMATH_CALUDE_function_range_l633_63397


namespace NUMINAMATH_CALUDE_string_average_length_l633_63316

theorem string_average_length : 
  let string1 : ℝ := 1.5
  let string2 : ℝ := 4.5
  let average := (string1 + string2) / 2
  average = 3 := by
sorry

end NUMINAMATH_CALUDE_string_average_length_l633_63316


namespace NUMINAMATH_CALUDE_girls_in_circle_l633_63306

theorem girls_in_circle (total : ℕ) (holding_boys_hand : ℕ) (holding_girls_hand : ℕ) 
  (h1 : total = 40)
  (h2 : holding_boys_hand = 22)
  (h3 : holding_girls_hand = 30) :
  ∃ (girls : ℕ), girls = 24 ∧ 
    girls * 2 = holding_girls_hand * 2 + holding_boys_hand + holding_girls_hand - total :=
by
  sorry

end NUMINAMATH_CALUDE_girls_in_circle_l633_63306


namespace NUMINAMATH_CALUDE_hendrix_class_size_l633_63387

theorem hendrix_class_size (initial_students : ℕ) (new_students : ℕ) (transfer_fraction : ℚ) : 
  initial_students = 160 → 
  new_students = 20 → 
  transfer_fraction = 1/3 →
  (initial_students + new_students) - ((initial_students + new_students : ℚ) * transfer_fraction).floor = 120 := by
  sorry

end NUMINAMATH_CALUDE_hendrix_class_size_l633_63387


namespace NUMINAMATH_CALUDE_two_unique_intersection_lines_l633_63346

/-- A parabola defined by the equation y^2 = 8x -/
def parabola (x y : ℝ) : Prop := y^2 = 8*x

/-- A point on the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line on the plane -/
structure Line where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- Predicate to check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  p.y = l.m * p.x + l.b

/-- Predicate to check if a line intersects the parabola at only one point -/
def uniqueIntersection (l : Line) : Prop :=
  ∃! p : Point, pointOnLine p l ∧ parabola p.x p.y

/-- The point (2, 4) -/
def givenPoint : Point := ⟨2, 4⟩

/-- The theorem stating that there are exactly two lines passing through (2, 4) 
    that intersect the parabola y^2 = 8x at only one point -/
theorem two_unique_intersection_lines : 
  ∃! (l1 l2 : Line), 
    pointOnLine givenPoint l1 ∧ 
    pointOnLine givenPoint l2 ∧ 
    uniqueIntersection l1 ∧ 
    uniqueIntersection l2 ∧ 
    l1 ≠ l2 :=
sorry

end NUMINAMATH_CALUDE_two_unique_intersection_lines_l633_63346


namespace NUMINAMATH_CALUDE_distance_between_sets_l633_63314

/-- The distance between two sets A and B, where
    A = {y | y = 2x - 1, x ∈ ℝ} and
    B = {y | y = x² + 1, x ∈ ℝ},
    is defined as the minimum value of |a - b|, where a ∈ A and b ∈ B. -/
theorem distance_between_sets :
  ∃ (x y : ℝ), |((2 * x) - 1) - (y^2 + 1)| = 0 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_sets_l633_63314


namespace NUMINAMATH_CALUDE_chess_tournament_impossibility_l633_63361

theorem chess_tournament_impossibility (n : ℕ) (g : ℕ) (x : ℕ) : 
  n = 50 →  -- Total number of players
  g = 61 →  -- Total number of games played
  x ≤ n →   -- Number of players who played 3 games
  (3 * x + 2 * (n - x)) / 2 = g →  -- Total games calculation
  x * 3 > g →  -- Contradiction: games played by 3-game players exceed total games
  False :=
by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_impossibility_l633_63361


namespace NUMINAMATH_CALUDE_root_implies_a_value_l633_63380

theorem root_implies_a_value (a : ℝ) : 
  ((3 : ℝ) = 3 ∧ (a - 2) / 3 - 1 / (3 - 2) = 0) → a = 5 := by
  sorry

end NUMINAMATH_CALUDE_root_implies_a_value_l633_63380


namespace NUMINAMATH_CALUDE_f_second_derivative_at_zero_l633_63312

noncomputable def f (x : ℝ) : ℝ := Real.exp x + Real.sin x - Real.cos x

theorem f_second_derivative_at_zero :
  (deriv (deriv f)) 0 = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_second_derivative_at_zero_l633_63312


namespace NUMINAMATH_CALUDE_equal_remainders_divisor_l633_63355

theorem equal_remainders_divisor : ∃ (n : ℕ), n > 0 ∧ 
  n ∣ (2287 - 2028) ∧ 
  n ∣ (2028 - 1806) ∧ 
  n ∣ (2287 - 1806) ∧
  ∀ (m : ℕ), m > n → ¬(m ∣ (2287 - 2028) ∧ m ∣ (2028 - 1806) ∧ m ∣ (2287 - 1806)) :=
by sorry

end NUMINAMATH_CALUDE_equal_remainders_divisor_l633_63355


namespace NUMINAMATH_CALUDE_inequality_system_solution_l633_63327

theorem inequality_system_solution (x : ℝ) :
  (x - 1 > 0 ∧ (2 * x + 1) / 3 ≤ 3) ↔ (1 < x ∧ x ≤ 4) :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l633_63327


namespace NUMINAMATH_CALUDE_factorial_fraction_equality_l633_63353

theorem factorial_fraction_equality : (5 * Nat.factorial 7 + 35 * Nat.factorial 6) / Nat.factorial 8 = 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_factorial_fraction_equality_l633_63353


namespace NUMINAMATH_CALUDE_gcf_of_154_308_462_l633_63324

theorem gcf_of_154_308_462 : Nat.gcd 154 (Nat.gcd 308 462) = 154 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_154_308_462_l633_63324


namespace NUMINAMATH_CALUDE_total_campers_l633_63382

def basketball_campers : ℕ := 24
def football_campers : ℕ := 32
def soccer_campers : ℕ := 32

theorem total_campers : basketball_campers + football_campers + soccer_campers = 88 := by
  sorry

end NUMINAMATH_CALUDE_total_campers_l633_63382


namespace NUMINAMATH_CALUDE_smallest_next_divisor_after_221_l633_63334

theorem smallest_next_divisor_after_221 (m : ℕ) (h1 : 1000 ≤ m ∧ m ≤ 9999) 
  (h2 : Even m) (h3 : m % 221 = 0) : 
  ∃ (d : ℕ), d > 221 ∧ m % d = 0 ∧ d = 247 ∧ 
  ∀ (x : ℕ), 221 < x ∧ x < 247 → m % x ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_smallest_next_divisor_after_221_l633_63334


namespace NUMINAMATH_CALUDE_transformation_matrix_correct_l633_63385

def rotation_matrix : Matrix (Fin 2) (Fin 2) ℝ := !![0, -1; 1, 0]
def scaling_matrix (s : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![s, 0; 0, s]

/-- The transformation matrix M represents a 90° counter-clockwise rotation followed by a scaling of factor 3 -/
theorem transformation_matrix_correct :
  let M : Matrix (Fin 2) (Fin 2) ℝ := !![0, -3; 3, 0]
  M = scaling_matrix 3 * rotation_matrix := by sorry

end NUMINAMATH_CALUDE_transformation_matrix_correct_l633_63385


namespace NUMINAMATH_CALUDE_sales_equation_l633_63320

/-- Represents the salesman's total sales -/
def S : ℝ := sorry

/-- Old commission rate -/
def old_rate : ℝ := 0.05

/-- New fixed salary -/
def new_fixed_salary : ℝ := 1300

/-- New commission rate -/
def new_rate : ℝ := 0.025

/-- Sales threshold for new commission -/
def threshold : ℝ := 4000

/-- Difference in remuneration between new and old schemes -/
def remuneration_difference : ℝ := 600

/-- Theorem stating the equation that the salesman's total sales must satisfy -/
theorem sales_equation : 
  new_fixed_salary + new_rate * (S - threshold) = old_rate * S + remuneration_difference :=
sorry

end NUMINAMATH_CALUDE_sales_equation_l633_63320


namespace NUMINAMATH_CALUDE_line_slope_intercept_sum_l633_63396

/-- Given a line passing through points (1, 2) and (3, 0), 
    prove that the sum of its slope and y-intercept is 2. -/
theorem line_slope_intercept_sum (m b : ℝ) : 
  (2 = m * 1 + b) → (0 = m * 3 + b) → m + b = 2 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_intercept_sum_l633_63396


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l633_63389

-- Define the sets M and N
def M : Set ℝ := {x | 1 - 2/x < 0}
def N : Set ℝ := {x | -1 ≤ x}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | 0 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l633_63389


namespace NUMINAMATH_CALUDE_game_ends_after_46_rounds_game_doesnt_end_before_46_rounds_l633_63321

/-- Represents the state of the game at any point --/
structure GameState where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Represents a single round of the game --/
def play_round (state : GameState) : GameState :=
  if state.a ≥ state.b ∧ state.a ≥ state.c then
    { a := state.a - 3, b := state.b + 1, c := state.c + 1 }
  else if state.b ≥ state.a ∧ state.b ≥ state.c then
    { a := state.a + 1, b := state.b - 3, c := state.c + 1 }
  else
    { a := state.a + 1, b := state.b + 1, c := state.c - 3 }

/-- Plays the game for a given number of rounds --/
def play_game (initial_state : GameState) (rounds : ℕ) : GameState :=
  match rounds with
  | 0 => initial_state
  | n + 1 => play_round (play_game initial_state n)

/-- The main theorem stating that the game ends after 46 rounds --/
theorem game_ends_after_46_rounds :
  let initial_state := { a := 18, b := 17, c := 16 : GameState }
  let final_state := play_game initial_state 46
  final_state.a = 0 ∨ final_state.b = 0 ∨ final_state.c = 0 :=
by sorry

/-- The game doesn't end before 46 rounds --/
theorem game_doesnt_end_before_46_rounds :
  let initial_state := { a := 18, b := 17, c := 16 : GameState }
  ∀ n < 46, let state := play_game initial_state n
    state.a > 0 ∧ state.b > 0 ∧ state.c > 0 :=
by sorry

end NUMINAMATH_CALUDE_game_ends_after_46_rounds_game_doesnt_end_before_46_rounds_l633_63321


namespace NUMINAMATH_CALUDE_overtime_hours_calculation_l633_63318

theorem overtime_hours_calculation (regular_rate : ℝ) (regular_hours : ℝ) (total_pay : ℝ) :
  regular_rate = 3 →
  regular_hours = 40 →
  total_pay = 180 →
  (total_pay - regular_rate * regular_hours) / (2 * regular_rate) = 10 := by
  sorry

end NUMINAMATH_CALUDE_overtime_hours_calculation_l633_63318


namespace NUMINAMATH_CALUDE_hallie_net_earnings_l633_63342

def hourly_rate : ℝ := 10

def monday_hours : ℝ := 7
def monday_tips : ℝ := 18

def tuesday_hours : ℝ := 5
def tuesday_tips : ℝ := 12

def wednesday_hours : ℝ := 7
def wednesday_tips : ℝ := 20

def thursday_hours : ℝ := 8
def thursday_tips : ℝ := 25

def friday_hours : ℝ := 6
def friday_tips : ℝ := 15

def discount_rate : ℝ := 0.05

def total_earnings : ℝ := 
  (monday_hours * hourly_rate + monday_tips) +
  (tuesday_hours * hourly_rate + tuesday_tips) +
  (wednesday_hours * hourly_rate + wednesday_tips) +
  (thursday_hours * hourly_rate + thursday_tips) +
  (friday_hours * hourly_rate + friday_tips)

def discount_amount : ℝ := total_earnings * discount_rate

def net_earnings : ℝ := total_earnings - discount_amount

theorem hallie_net_earnings : net_earnings = 399 := by
  sorry

end NUMINAMATH_CALUDE_hallie_net_earnings_l633_63342


namespace NUMINAMATH_CALUDE_second_sum_calculation_l633_63338

theorem second_sum_calculation (total : ℚ) (x : ℚ) 
  (h1 : total = 2678)
  (h2 : x * (3 / 100) * 8 = (total - x) * (5 / 100) * 3) :
  total - x = 2401 := by
  sorry

end NUMINAMATH_CALUDE_second_sum_calculation_l633_63338


namespace NUMINAMATH_CALUDE_coefficient_of_x_cubed_is_39_l633_63329

def expression (x : ℝ) : ℝ :=
  2 * (x^2 - 2*x^3 + 2*x) + 4 * (x + 3*x^3 - 2*x^2 + 2*x^5 - x^3) - 7 * (2 + 2*x - 5*x^3 - x^2)

theorem coefficient_of_x_cubed_is_39 :
  (deriv (deriv (deriv expression))) 0 / 6 = 39 := by sorry

end NUMINAMATH_CALUDE_coefficient_of_x_cubed_is_39_l633_63329


namespace NUMINAMATH_CALUDE_opposite_of_negative_abs_two_l633_63345

theorem opposite_of_negative_abs_two : -(- |(-2)|) = 2 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_abs_two_l633_63345


namespace NUMINAMATH_CALUDE_max_sides_convex_polygon_four_obtuse_l633_63313

/-- Represents a convex polygon with n sides and exactly four obtuse angles -/
structure ConvexPolygon (n : ℕ) where
  sides : n > 0
  obtuse_angles : ℕ
  obtuse_count : obtuse_angles = 4

/-- The sum of interior angles of an n-sided polygon is (n-2) * 180 degrees -/
def interior_angle_sum (n : ℕ) : ℝ := (n - 2) * 180

/-- An obtuse angle is greater than 90 degrees and less than 180 degrees -/
def is_obtuse (angle : ℝ) : Prop := 90 < angle ∧ angle < 180

/-- An acute angle is greater than 0 degrees and less than 90 degrees -/
def is_acute (angle : ℝ) : Prop := 0 < angle ∧ angle < 90

/-- The maximum number of sides for a convex polygon with exactly four obtuse angles is 7 -/
theorem max_sides_convex_polygon_four_obtuse :
  ∀ n : ℕ, ConvexPolygon n → n ≤ 7 :=
sorry

end NUMINAMATH_CALUDE_max_sides_convex_polygon_four_obtuse_l633_63313


namespace NUMINAMATH_CALUDE_geometric_sequence_207th_term_l633_63375

def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a₁ * r ^ (n - 1)

theorem geometric_sequence_207th_term :
  let a₁ := 8
  let a₂ := -8
  let r := a₂ / a₁
  geometric_sequence a₁ r 207 = 8 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_207th_term_l633_63375


namespace NUMINAMATH_CALUDE_parallelogram_area_is_fifteen_l633_63374

/-- Represents a parallelogram EFGH with base FG and height FH -/
structure Parallelogram where
  base : ℝ
  height : ℝ

/-- Calculates the area of a parallelogram -/
def area (p : Parallelogram) : ℝ := p.base * p.height

/-- The theorem stating that the area of the given parallelogram EFGH is 15 -/
theorem parallelogram_area_is_fifteen : ∃ (p : Parallelogram), p.base = 5 ∧ p.height = 3 ∧ area p = 15 := by
  sorry


end NUMINAMATH_CALUDE_parallelogram_area_is_fifteen_l633_63374


namespace NUMINAMATH_CALUDE_art_museum_picture_distribution_l633_63386

theorem art_museum_picture_distribution (total_pictures : ℕ) (num_exhibits : ℕ) : 
  total_pictures = 154 → num_exhibits = 9 → 
  (∃ (additional_pictures : ℕ), 
    (total_pictures + additional_pictures) % num_exhibits = 0 ∧
    additional_pictures = 8) := by
  sorry

end NUMINAMATH_CALUDE_art_museum_picture_distribution_l633_63386


namespace NUMINAMATH_CALUDE_additional_money_needed_mrs_smith_shopping_l633_63307

/-- Calculates the additional money needed for Mrs. Smith's shopping --/
theorem additional_money_needed (total_budget dress_budget shoe_budget accessory_budget : ℚ)
  (dress_discount shoe_discount accessory_discount : ℚ) : ℚ :=
  let dress_needed := dress_budget * (1 + 2/5)
  let shoe_needed := shoe_budget * (1 + 2/5)
  let accessory_needed := accessory_budget * (1 + 2/5)
  let dress_discounted := dress_needed * (1 - dress_discount)
  let shoe_discounted := shoe_needed * (1 - shoe_discount)
  let accessory_discounted := accessory_needed * (1 - accessory_discount)
  let total_needed := dress_discounted + shoe_discounted + accessory_discounted
  total_needed - total_budget

/-- Proves that Mrs. Smith needs $84.50 more to complete her shopping --/
theorem mrs_smith_shopping :
  additional_money_needed 500 300 150 50 (20/100) (10/100) (15/100) = 169/2 := by
  sorry

end NUMINAMATH_CALUDE_additional_money_needed_mrs_smith_shopping_l633_63307


namespace NUMINAMATH_CALUDE_expression_simplification_l633_63326

theorem expression_simplification (a : ℚ) (h : a = -1/2) :
  (4 - 3*a)*(1 + 2*a) - 3*a*(1 - 2*a) = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l633_63326


namespace NUMINAMATH_CALUDE_inequality_proof_l633_63351

theorem inequality_proof (a₁ a₂ a₃ b₁ b₂ b₃ : ℝ) 
  (ha₁ : 0 < a₁) (ha₂ : 0 < a₂) (ha₃ : 0 < a₃) 
  (hb₁ : 0 < b₁) (hb₂ : 0 < b₂) (hb₃ : 0 < b₃) : 
  (a₁*b₂ + a₂*b₁ + a₂*b₃ + a₃*b₂ + a₃*b₁ + a₁*b₃)^2 ≥ 
  4*(a₁*a₂ + a₂*a₃ + a₃*a₁)*(b₁*b₂ + b₂*b₃ + b₃*b₁) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l633_63351


namespace NUMINAMATH_CALUDE_g_2187_equals_343_l633_63333

-- Define the properties of function g
def satisfies_property (g : ℕ → ℝ) : Prop :=
  ∀ (x y m : ℕ), x > 0 → y > 0 → m > 0 → x + y = 3^m → g x + g y = m^3

-- Theorem statement
theorem g_2187_equals_343 (g : ℕ → ℝ) (h : satisfies_property g) : g 2187 = 343 := by
  sorry

end NUMINAMATH_CALUDE_g_2187_equals_343_l633_63333


namespace NUMINAMATH_CALUDE_no_sequence_satisfying_condition_l633_63340

theorem no_sequence_satisfying_condition :
  ¬ (∃ (a : ℝ) (a_n : ℕ → ℝ), 
    (0 < a ∧ a < 1) ∧
    (∀ n : ℕ, n > 0 → a_n n > 0) ∧
    (∀ n : ℕ, n > 0 → 1 + a_n (n + 1) ≤ a_n n + (a / n) * a_n n)) :=
by sorry

end NUMINAMATH_CALUDE_no_sequence_satisfying_condition_l633_63340


namespace NUMINAMATH_CALUDE_linear_functions_properties_l633_63325

/-- Two linear functions -/
def y₁ (x : ℝ) : ℝ := -x + 1
def y₂ (x : ℝ) : ℝ := -3*x + 2

theorem linear_functions_properties :
  (∃ a : ℝ, ∀ x > 0, y₁ x = a + y₂ x → a > -1) ∧
  (∀ x y : ℝ, y₁ x = y ∧ y₂ x = y → 12*x^2 + 12*x*y + 3*y^2 = 27/4) ∧
  (∃ A B : ℝ, ∀ x : ℝ, x ≠ 1 ∧ 3*x ≠ 2 →
    (4-2*x)/((3*x-2)*(x-1)) = A/(y₁ x) + B/(y₂ x) →
    A/B + B/A = -4.25) := by sorry

end NUMINAMATH_CALUDE_linear_functions_properties_l633_63325


namespace NUMINAMATH_CALUDE_constant_function_operation_l633_63376

-- Define the function g
def g : ℝ → ℝ := fun _ ↦ 5

-- State the theorem
theorem constant_function_operation (x : ℝ) : 3 * g (x - 3) + 1 = 16 := by
  sorry

end NUMINAMATH_CALUDE_constant_function_operation_l633_63376


namespace NUMINAMATH_CALUDE_geometric_sequence_propositions_l633_63390

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- An increasing sequence -/
def IncreasingSequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, n < m → a n < a m

theorem geometric_sequence_propositions (a : ℕ → ℝ) (h : GeometricSequence a) :
  (((a 1 < a 2) ∧ (a 2 < a 3)) → IncreasingSequence a) ∧
  (IncreasingSequence a → ((a 1 < a 2) ∧ (a 2 < a 3))) ∧
  (¬((a 1 < a 2) ∧ (a 2 < a 3)) → ¬IncreasingSequence a) ∧
  (¬IncreasingSequence a → ¬((a 1 < a 2) ∧ (a 2 < a 3))) :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_propositions_l633_63390


namespace NUMINAMATH_CALUDE_digit_sum_of_squared_palindrome_l633_63370

theorem digit_sum_of_squared_palindrome (r : ℕ) (x : ℕ) (p q : ℕ) :
  r ≤ 400 →
  x = p * r^3 + p * r^2 + q * r + q →
  7 * q = 17 * p →
  ∃ (a b c : ℕ),
    x^2 = a * r^6 + b * r^5 + c * r^4 + 0 * r^3 + c * r^2 + b * r + a →
  2 * (a + b + c) = 400 :=
by sorry

end NUMINAMATH_CALUDE_digit_sum_of_squared_palindrome_l633_63370


namespace NUMINAMATH_CALUDE_james_purchase_cost_l633_63332

theorem james_purchase_cost : 
  let num_shirts : ℕ := 10
  let num_pants : ℕ := num_shirts / 2
  let shirt_cost : ℕ := 6
  let pants_cost : ℕ := 8
  let total_cost : ℕ := num_shirts * shirt_cost + num_pants * pants_cost
  total_cost = 100 := by sorry

end NUMINAMATH_CALUDE_james_purchase_cost_l633_63332


namespace NUMINAMATH_CALUDE_min_value_fraction_l633_63304

theorem min_value_fraction (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_sum : a + 2*b = 2) :
  (a + b) / (a * b) ≥ (3 + 2 * Real.sqrt 2) / 2 ∧
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 2*b₀ = 2 ∧ (a₀ + b₀) / (a₀ * b₀) = (3 + 2 * Real.sqrt 2) / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_fraction_l633_63304


namespace NUMINAMATH_CALUDE_subset_implies_m_values_l633_63309

-- Define the sets P and Q
def P : Set ℝ := {x | x^2 = 1}
def Q (m : ℝ) : Set ℝ := {x | m * x = 1}

-- State the theorem
theorem subset_implies_m_values (m : ℝ) : Q m ⊆ P → m = 0 ∨ m = 1 ∨ m = -1 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_m_values_l633_63309


namespace NUMINAMATH_CALUDE_tangent_line_equation_l633_63360

-- Define the line l
def line_l (x y : ℝ) : Prop := y = (1/2) * x

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 1

-- Define the line ST
def line_ST (x y : ℝ) : Prop := y = -2 * x + 11/2

-- Theorem statement
theorem tangent_line_equation 
  (h1 : line_l 2 1)
  (h2 : line_l 6 3)
  (h3 : ∃ (x₀ y₀ : ℝ), line_l x₀ y₀ ∧ circle_C x₀ y₀)
  (h4 : circle_C 2 0) :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
    line_ST x₁ y₁ ∧ line_ST x₂ y₂ ∧
    line_ST 6 3 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l633_63360


namespace NUMINAMATH_CALUDE_union_complement_equality_complement_intersection_equality_l633_63302

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5, 6, 7}

-- Define set A
def A : Set Nat := {2, 4, 5}

-- Define set B
def B : Set Nat := {1, 3, 5, 7}

-- Theorem for part (1)
theorem union_complement_equality :
  A ∪ (U \ B) = {2, 4, 5, 6} := by sorry

-- Theorem for part (2)
theorem complement_intersection_equality :
  U \ (A ∩ B) = {1, 2, 3, 4, 6, 7} := by sorry

end NUMINAMATH_CALUDE_union_complement_equality_complement_intersection_equality_l633_63302


namespace NUMINAMATH_CALUDE_musical_chairs_theorem_l633_63356

/-- A function is a derangement if it has no fixed points -/
def IsDerangement {α : Type*} (f : α → α) : Prop :=
  ∀ x, f x ≠ x

/-- A positive integer is a prime power if it's of the form p^k where p is prime and k > 0 -/
def IsPrimePower (n : ℕ) : Prop :=
  ∃ (p k : ℕ), Prime p ∧ k > 0 ∧ n = p^k

theorem musical_chairs_theorem (n m : ℕ) 
    (h1 : m > 1) 
    (h2 : m ≤ n) 
    (h3 : ¬ IsPrimePower m) : 
    ∃ (f : Fin n → Fin n), 
      Function.Bijective f ∧ 
      IsDerangement f ∧ 
      (∀ x, (f^[m]) x = x) ∧ 
      (∀ (k : ℕ) (hk : k < m), ∃ x, (f^[k]) x ≠ x) := by
  sorry

end NUMINAMATH_CALUDE_musical_chairs_theorem_l633_63356


namespace NUMINAMATH_CALUDE_remainder_9876543210_mod_140_l633_63388

theorem remainder_9876543210_mod_140 : 9876543210 % 140 = 70 := by
  sorry

end NUMINAMATH_CALUDE_remainder_9876543210_mod_140_l633_63388


namespace NUMINAMATH_CALUDE_sum_of_xyz_equals_695_l633_63372

theorem sum_of_xyz_equals_695 (a b : ℝ) (x y z : ℕ+) :
  a^2 = 9/25 →
  b^2 = (3 + Real.sqrt 2)^2 / 14 →
  a < 0 →
  b > 0 →
  (a + b)^3 = (x.val : ℝ) * Real.sqrt y.val / z.val →
  x.val + y.val + z.val = 695 := by
sorry

end NUMINAMATH_CALUDE_sum_of_xyz_equals_695_l633_63372


namespace NUMINAMATH_CALUDE_bus_driver_regular_rate_l633_63363

/-- Represents the compensation structure and work hours of a bus driver --/
structure BusDriverCompensation where
  regularRate : ℝ
  overtimeRate : ℝ
  regularHours : ℝ
  overtimeHours : ℝ
  totalCompensation : ℝ

/-- Theorem stating that given the conditions, the regular rate is $18 per hour --/
theorem bus_driver_regular_rate 
  (comp : BusDriverCompensation)
  (h1 : comp.regularHours = 40)
  (h2 : comp.overtimeHours = 48.12698412698413 - 40)
  (h3 : comp.overtimeRate = comp.regularRate * 1.75)
  (h4 : comp.totalCompensation = 976)
  (h5 : comp.totalCompensation = comp.regularRate * comp.regularHours + 
                                 comp.overtimeRate * comp.overtimeHours) :
  comp.regularRate = 18 := by
  sorry


end NUMINAMATH_CALUDE_bus_driver_regular_rate_l633_63363
