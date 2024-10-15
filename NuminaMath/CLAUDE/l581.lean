import Mathlib

namespace NUMINAMATH_CALUDE_total_books_count_l581_58126

/-- Given Benny's initial book count, the number of books he gave to Sandy, and Tim's book count,
    prove that the total number of books Benny and Tim have together is 47. -/
theorem total_books_count (benny_initial : ℕ) (given_to_sandy : ℕ) (tim_books : ℕ)
    (h1 : benny_initial = 24)
    (h2 : given_to_sandy = 10)
    (h3 : tim_books = 33) :
    benny_initial - given_to_sandy + tim_books = 47 := by
  sorry

end NUMINAMATH_CALUDE_total_books_count_l581_58126


namespace NUMINAMATH_CALUDE_crickets_collected_l581_58172

theorem crickets_collected (total : ℕ) (more_needed : ℕ) (h : total = 11 ∧ more_needed = 4) : 
  total - more_needed = 7 := by
  sorry

end NUMINAMATH_CALUDE_crickets_collected_l581_58172


namespace NUMINAMATH_CALUDE_problem_solid_surface_area_l581_58166

/-- Represents a solid formed by unit cubes -/
structure CubeSolid where
  length : ℕ
  width : ℕ
  height : ℕ
  additional_cubes : ℕ

/-- Calculates the surface area of the CubeSolid -/
def surface_area (solid : CubeSolid) : ℕ :=
  2 * (solid.length * solid.height + solid.additional_cubes) + -- front and back
  (solid.length * solid.width + (solid.length * solid.width - solid.additional_cubes)) + -- top and bottom
  2 * (solid.width * solid.height) -- left and right

/-- The specific solid described in the problem -/
def problem_solid : CubeSolid :=
  { length := 4
    width := 3
    height := 1
    additional_cubes := 2 }

theorem problem_solid_surface_area :
  surface_area problem_solid = 42 := by
  sorry

end NUMINAMATH_CALUDE_problem_solid_surface_area_l581_58166


namespace NUMINAMATH_CALUDE_double_earnings_days_theorem_l581_58114

/-- Calculate the number of additional days needed to double earnings -/
def daysToDoubleEarnings (daysSoFar : ℕ) (earningsSoFar : ℚ) : ℕ :=
  daysSoFar

/-- Theorem: The number of additional days needed to double earnings
    is equal to the number of days already worked -/
theorem double_earnings_days_theorem (daysSoFar : ℕ) (earningsSoFar : ℚ) 
    (hDays : daysSoFar > 0) (hEarnings : earningsSoFar > 0) :
  daysToDoubleEarnings daysSoFar earningsSoFar = daysSoFar := by
  sorry

#eval daysToDoubleEarnings 10 250  -- Should output 10

end NUMINAMATH_CALUDE_double_earnings_days_theorem_l581_58114


namespace NUMINAMATH_CALUDE_sum_of_primes_divisible_by_12_l581_58113

theorem sum_of_primes_divisible_by_12 (p q : ℕ) : 
  Prime p → Prime q → p - q = 2 → q > 3 → ∃ k : ℕ, p + q = 12 * k := by
sorry

end NUMINAMATH_CALUDE_sum_of_primes_divisible_by_12_l581_58113


namespace NUMINAMATH_CALUDE_coaching_start_date_l581_58199

/-- Represents a date in a year -/
structure Date :=
  (month : Nat)
  (day : Nat)

/-- Calculates the number of days from the start of the year to a given date in a non-leap year -/
def daysFromYearStart (d : Date) : Nat :=
  sorry

/-- Calculates the date that is a given number of days before another date in a non-leap year -/
def dateBeforeDays (d : Date) (days : Nat) : Date :=
  sorry

theorem coaching_start_date :
  let end_date : Date := ⟨9, 4⟩  -- September 4
  let coaching_duration : Nat := 245
  let start_date := dateBeforeDays end_date coaching_duration
  start_date = ⟨1, 2⟩  -- January 2
  :=
sorry

end NUMINAMATH_CALUDE_coaching_start_date_l581_58199


namespace NUMINAMATH_CALUDE_doubled_factorial_30_trailing_zeros_l581_58109

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ := sorry

/-- The number of trailing zeros in 2 * n! -/
def trailingZerosDoubled (n : ℕ) : ℕ := sorry

theorem doubled_factorial_30_trailing_zeros :
  trailingZerosDoubled 30 = 7 := by sorry

end NUMINAMATH_CALUDE_doubled_factorial_30_trailing_zeros_l581_58109


namespace NUMINAMATH_CALUDE_no_formula_matches_all_points_l581_58106

-- Define the table of values
def table : List (ℤ × ℤ) := [(0, 200), (1, 160), (2, 100), (3, 20), (4, -80)]

-- Define the formulas
def formula_A (x : ℤ) : ℤ := 200 - 30*x
def formula_B (x : ℤ) : ℤ := 200 - 20*x - 10*x^2
def formula_C (x : ℤ) : ℤ := 200 - 40*x + 10*x^2
def formula_D (x : ℤ) : ℤ := 200 - 10*x - 20*x^2

-- Theorem statement
theorem no_formula_matches_all_points :
  ¬(∀ (x y : ℤ), (x, y) ∈ table → 
    (y = formula_A x ∨ y = formula_B x ∨ y = formula_C x ∨ y = formula_D x)) :=
by sorry

end NUMINAMATH_CALUDE_no_formula_matches_all_points_l581_58106


namespace NUMINAMATH_CALUDE_z_range_in_parallelogram_l581_58155

-- Define the parallelogram ABCD
def A : ℝ × ℝ := (-1, 2)
def B : ℝ × ℝ := (3, 4)
def C : ℝ × ℝ := (4, -2)
def D : ℝ × ℝ := (4, 0)

-- Define a function to check if a point is within or on the boundary of the parallelogram
def isInParallelogram (p : ℝ × ℝ) : Prop := sorry

-- Define the function z
def z (p : ℝ × ℝ) : ℝ := 2 * p.1 - 5 * p.2

-- State the theorem
theorem z_range_in_parallelogram :
  ∀ p : ℝ × ℝ, isInParallelogram p → -14 ≤ z p ∧ z p ≤ 18 := by sorry

end NUMINAMATH_CALUDE_z_range_in_parallelogram_l581_58155


namespace NUMINAMATH_CALUDE_pascal_triangle_element_l581_58122

/-- The number of elements in the row of Pascal's triangle we're considering -/
def row_length : ℕ := 31

/-- The position of the number we're looking for (1-indexed) -/
def target_position : ℕ := 25

/-- The row number in Pascal's triangle (0-indexed) -/
def row_number : ℕ := row_length - 1

/-- The column number in Pascal's triangle (0-indexed) -/
def column_number : ℕ := target_position - 1

theorem pascal_triangle_element :
  Nat.choose row_number column_number = 593775 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_element_l581_58122


namespace NUMINAMATH_CALUDE_tangent_line_slope_l581_58170

/-- The slope of a line tangent to the circle x^2 + y^2 - 4x + 2 = 0 is either 1 or -1 -/
theorem tangent_line_slope (m : ℝ) :
  (∀ x y : ℝ, y = m * x → x^2 + y^2 - 4*x + 2 = 0 → 
    ∀ ε > 0, ∃ δ > 0, ∀ x' y' : ℝ, 
      (x' - x)^2 + (y' - y)^2 < δ^2 → x'^2 + y'^2 - 4*x' + 2 > 0) →
  m = 1 ∨ m = -1 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_slope_l581_58170


namespace NUMINAMATH_CALUDE_matrix_determinant_l581_58138

theorem matrix_determinant : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![8, 4; -2, 3]
  Matrix.det A = 32 := by
  sorry

end NUMINAMATH_CALUDE_matrix_determinant_l581_58138


namespace NUMINAMATH_CALUDE_house_size_ratio_l581_58184

/-- The size of Kennedy's house in square feet -/
def kennedy_house_size : ℝ := 10000

/-- The size of Benedict's house in square feet -/
def benedict_house_size : ℝ := 2350

/-- The additional size in square feet added to the ratio of house sizes -/
def additional_size : ℝ := 600

/-- Theorem stating that the ratio of (Kennedy's house size - additional size) to Benedict's house size is 4 -/
theorem house_size_ratio : 
  (kennedy_house_size - additional_size) / benedict_house_size = 4 := by
  sorry

end NUMINAMATH_CALUDE_house_size_ratio_l581_58184


namespace NUMINAMATH_CALUDE_lee_makes_27_cookies_l581_58134

/-- Given that Lee can make 18 cookies with 2 cups of flour, 
    this function calculates how many cookies he can make with any amount of flour. -/
def cookies_from_flour (cups : ℚ) : ℚ :=
  18 * cups / 2

/-- Theorem stating that Lee can make 27 cookies with 3 cups of flour. -/
theorem lee_makes_27_cookies : cookies_from_flour 3 = 27 := by
  sorry

end NUMINAMATH_CALUDE_lee_makes_27_cookies_l581_58134


namespace NUMINAMATH_CALUDE_correct_green_pens_l581_58132

-- Define the ratio of blue pens to green pens
def blue_to_green_ratio : ℚ := 5 / 3

-- Define the number of blue pens
def blue_pens : ℕ := 20

-- Define the number of green pens
def green_pens : ℕ := 12

-- Theorem to prove
theorem correct_green_pens : 
  (blue_pens : ℚ) / green_pens = blue_to_green_ratio := by
  sorry

end NUMINAMATH_CALUDE_correct_green_pens_l581_58132


namespace NUMINAMATH_CALUDE_total_spent_is_108_l581_58140

/-- The total amount spent by Robert and Teddy on snacks -/
def total_spent (pizza_boxes : ℕ) (pizza_price : ℕ) (robert_drinks : ℕ) (drink_price : ℕ)
                (hamburgers : ℕ) (hamburger_price : ℕ) (teddy_drinks : ℕ) : ℕ :=
  pizza_boxes * pizza_price + robert_drinks * drink_price +
  hamburgers * hamburger_price + teddy_drinks * drink_price

/-- Theorem stating that the total amount spent is $108 -/
theorem total_spent_is_108 :
  total_spent 5 10 10 2 6 3 10 = 108 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_is_108_l581_58140


namespace NUMINAMATH_CALUDE_ellipse_locus_theorem_l581_58149

-- Define the ellipse
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define point A
def point_A (a : ℝ) : ℝ × ℝ := (a, 0)

-- Define point B
def point_B (a : ℝ) : ℝ × ℝ := (-a, 0)

-- Define a point P on the ellipse
def point_P (a b x y : ℝ) : Prop :=
  ellipse a b x y ∧ (x, y) ≠ point_A a ∧ (x, y) ≠ point_B a

-- Define the locus of M
def locus_M (a b x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / (a^2 / b)^2 = 1

-- Theorem statement
theorem ellipse_locus_theorem (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  ∀ x y : ℝ, point_P a b x y → ∃ m_x m_y : ℝ, locus_M a b m_x m_y :=
sorry

end NUMINAMATH_CALUDE_ellipse_locus_theorem_l581_58149


namespace NUMINAMATH_CALUDE_power_comparison_l581_58146

theorem power_comparison : 3^15 < 10^9 ∧ 10^9 < 5^13 := by
  sorry

end NUMINAMATH_CALUDE_power_comparison_l581_58146


namespace NUMINAMATH_CALUDE_function_value_at_negative_five_hundred_l581_58176

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * y) + 2 * x = x * f y + 2 * f x

theorem function_value_at_negative_five_hundred
  (f : ℝ → ℝ)
  (h1 : FunctionalEquation f)
  (h2 : f (-2) = 11) :
  f (-500) = -487 := by
sorry

end NUMINAMATH_CALUDE_function_value_at_negative_five_hundred_l581_58176


namespace NUMINAMATH_CALUDE_high_correlation_implies_r_close_to_one_l581_58169

-- Define a type for variables
def Variable : Type := ℝ

-- Define a correlation coefficient
def correlation_coefficient (x y : Variable) : ℝ := sorry

-- Define what it means for the degree of linear correlation to be very high
def high_linear_correlation (x y : Variable) : Prop :=
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ abs (correlation_coefficient x y) > 1 - ε

-- The theorem to prove
theorem high_correlation_implies_r_close_to_one (x y : Variable) :
  high_linear_correlation x y → ∃ (δ : ℝ), δ > 0 ∧ δ < 0.1 ∧ abs (correlation_coefficient x y) > 1 - δ :=
sorry

end NUMINAMATH_CALUDE_high_correlation_implies_r_close_to_one_l581_58169


namespace NUMINAMATH_CALUDE_probability_red_in_middle_l581_58167

/- Define the types of rosebushes -/
inductive Rosebush
| Red
| White

/- Define a row of rosebushes -/
def Row := List Rosebush

/- Define a function to check if the middle two rosebushes are red -/
def middleTwoAreRed (row : Row) : Bool :=
  match row with
  | [_, Rosebush.Red, Rosebush.Red, _] => true
  | _ => false

/- Define a function to generate all possible arrangements -/
def allArrangements : List Row :=
  sorry

/- Define a function to count arrangements with red rosebushes in the middle -/
def countRedInMiddle (arrangements : List Row) : Nat :=
  sorry

/- Theorem statement -/
theorem probability_red_in_middle :
  let arrangements := allArrangements
  let total := arrangements.length
  let favorable := countRedInMiddle arrangements
  (favorable : ℚ) / total = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_probability_red_in_middle_l581_58167


namespace NUMINAMATH_CALUDE_solution_set_implies_a_value_solution_set_all_reals_implies_a_range_l581_58121

-- Part 1
theorem solution_set_implies_a_value (a : ℝ) :
  (∀ x : ℝ, ax^2 - 2*a*x + 3 > 0 ↔ -1 < x ∧ x < 3) →
  a = -1 :=
sorry

-- Part 2
theorem solution_set_all_reals_implies_a_range (a : ℝ) :
  (∀ x : ℝ, ax^2 - 2*a*x + 3 > 0) →
  0 ≤ a ∧ a < 3 :=
sorry

end NUMINAMATH_CALUDE_solution_set_implies_a_value_solution_set_all_reals_implies_a_range_l581_58121


namespace NUMINAMATH_CALUDE_fraction_simplification_l581_58127

theorem fraction_simplification :
  (2 * (Real.sqrt 2 + Real.sqrt 6)) / (3 * Real.sqrt (2 + Real.sqrt 3)) = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l581_58127


namespace NUMINAMATH_CALUDE_magnitude_of_vector_combination_l581_58103

-- Define the vectors a and b
def a : Fin 2 → ℝ := ![1, 2]
def b : Fin 2 → ℝ := ![-1, 2]

-- State the theorem
theorem magnitude_of_vector_combination :
  ‖(3 • a) - b‖ = 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_vector_combination_l581_58103


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l581_58136

theorem binomial_coefficient_equality (n : ℕ) (h1 : n ≥ 6) :
  (3^5 * Nat.choose n 5 = 3^6 * Nat.choose n 6) → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l581_58136


namespace NUMINAMATH_CALUDE_imo_2007_problem_5_l581_58100

theorem imo_2007_problem_5 (k : ℕ+) :
  (∃ (n : ℕ+), (8 * k * n - 1) ∣ (4 * k^2 - 1)^2) ↔ Even k := by
  sorry

end NUMINAMATH_CALUDE_imo_2007_problem_5_l581_58100


namespace NUMINAMATH_CALUDE_algebraic_simplification_l581_58129

theorem algebraic_simplification (a b : ℝ) : 2 * a^2 * b - a^2 * b = a^2 * b := by
  sorry

end NUMINAMATH_CALUDE_algebraic_simplification_l581_58129


namespace NUMINAMATH_CALUDE_frog_final_position_l581_58159

-- Define the circle points
inductive CirclePoint
| One
| Two
| Three
| Four
| Five

-- Define the jump function
def jump (p : CirclePoint) : CirclePoint :=
  match p with
  | CirclePoint.One => CirclePoint.Two
  | CirclePoint.Two => CirclePoint.Four
  | CirclePoint.Three => CirclePoint.Four
  | CirclePoint.Four => CirclePoint.One
  | CirclePoint.Five => CirclePoint.One

-- Define the function to perform multiple jumps
def multiJump (start : CirclePoint) (n : Nat) : CirclePoint :=
  match n with
  | 0 => start
  | Nat.succ m => jump (multiJump start m)

-- Theorem statement
theorem frog_final_position :
  multiJump CirclePoint.Five 1995 = CirclePoint.Four := by
  sorry

end NUMINAMATH_CALUDE_frog_final_position_l581_58159


namespace NUMINAMATH_CALUDE_fencing_calculation_l581_58197

/-- The total fencing length for a square playground and a rectangular garden -/
def total_fencing (playground_side : ℝ) (garden_length garden_width : ℝ) : ℝ :=
  4 * playground_side + 2 * (garden_length + garden_width)

/-- Theorem: The total fencing for a square playground with side length 27 yards
    and a rectangular garden of 12 yards by 9 yards is equal to 150 yards -/
theorem fencing_calculation :
  total_fencing 27 12 9 = 150 := by
  sorry

end NUMINAMATH_CALUDE_fencing_calculation_l581_58197


namespace NUMINAMATH_CALUDE_tan_2x_and_sin_x_plus_pi_4_l581_58135

theorem tan_2x_and_sin_x_plus_pi_4 (x : ℝ) 
  (h1 : |Real.tan x| = 2) 
  (h2 : x ∈ Set.Ioo (π / 2) π) : 
  Real.tan (2 * x) = 4 / 3 ∧ 
  Real.sin (x + π / 4) = Real.sqrt 10 / 10 := by
  sorry

end NUMINAMATH_CALUDE_tan_2x_and_sin_x_plus_pi_4_l581_58135


namespace NUMINAMATH_CALUDE_max_product_sum_2020_l581_58144

theorem max_product_sum_2020 : 
  (∃ (a b : ℤ), a + b = 2020 ∧ a * b = 1020100) ∧ 
  (∀ (x y : ℤ), x + y = 2020 → x * y ≤ 1020100) := by
sorry

end NUMINAMATH_CALUDE_max_product_sum_2020_l581_58144


namespace NUMINAMATH_CALUDE_average_age_after_leaving_l581_58180

theorem average_age_after_leaving (initial_people : ℕ) (initial_avg : ℚ) (leaving_age : ℕ) (remaining_people : ℕ) :
  initial_people = 7 →
  initial_avg = 32 →
  leaving_age = 22 →
  remaining_people = 6 →
  (initial_people * initial_avg - leaving_age) / remaining_people = 34 := by
sorry

end NUMINAMATH_CALUDE_average_age_after_leaving_l581_58180


namespace NUMINAMATH_CALUDE_solution_count_theorem_l581_58156

/-- The number of solutions to the equation 2x + 3y + z + x^2 = n for positive integers x, y, z -/
def num_solutions (n : ℕ+) : ℕ := 
  (Finset.filter (fun t : ℕ × ℕ × ℕ => 
    2 * t.1 + 3 * t.2.1 + t.2.2 + t.1 * t.1 = n.val ∧ 
    t.1 > 0 ∧ t.2.1 > 0 ∧ t.2.2 > 0) (Finset.product (Finset.range n.val) (Finset.product (Finset.range n.val) (Finset.range n.val)))).card

theorem solution_count_theorem (n : ℕ+) : 
  num_solutions n = 25 → n = 32 ∨ n = 33 := by
  sorry

end NUMINAMATH_CALUDE_solution_count_theorem_l581_58156


namespace NUMINAMATH_CALUDE_least_five_digit_square_cube_l581_58116

theorem least_five_digit_square_cube : ∃ n : ℕ,
  (n ≥ 10000 ∧ n < 100000) ∧  -- five-digit number
  (∃ a : ℕ, n = a^2) ∧        -- perfect square
  (∃ b : ℕ, n = b^3) ∧        -- perfect cube
  (∀ m : ℕ, m < n →
    (m < 10000 ∨ m ≥ 100000) ∨
    (∀ a : ℕ, m ≠ a^2) ∨
    (∀ b : ℕ, m ≠ b^3)) ∧
  n = 15625 :=
by sorry

end NUMINAMATH_CALUDE_least_five_digit_square_cube_l581_58116


namespace NUMINAMATH_CALUDE_bakers_earnings_l581_58185

/-- The baker's earnings problem -/
theorem bakers_earnings (cakes_sold : ℕ) (cake_price : ℕ) (pies_sold : ℕ) (pie_price : ℕ) 
  (h1 : cakes_sold = 453)
  (h2 : cake_price = 12)
  (h3 : pies_sold = 126)
  (h4 : pie_price = 7) :
  cakes_sold * cake_price + pies_sold * pie_price = 6318 := by
  sorry


end NUMINAMATH_CALUDE_bakers_earnings_l581_58185


namespace NUMINAMATH_CALUDE_original_price_calculation_l581_58160

theorem original_price_calculation (P : ℝ) : 
  (P * (1 - 0.06) * (1 + 0.10) = 6876.1) → P = 6650 := by
  sorry

end NUMINAMATH_CALUDE_original_price_calculation_l581_58160


namespace NUMINAMATH_CALUDE_sum_of_percentages_l581_58133

theorem sum_of_percentages (X Y Z : ℝ) : 
  X = 0.2 * 50 →
  40 = 0.2 * Y →
  40 = (Z / 100) * 50 →
  X + Y + Z = 290 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_percentages_l581_58133


namespace NUMINAMATH_CALUDE_rectangle_dimension_change_l581_58124

theorem rectangle_dimension_change (L B : ℝ) (L' B' : ℝ) (h1 : L' = 1.4 * L) (h2 : B' * L' = 1.05 * B * L) :
  B' = 0.75 * B :=
sorry

end NUMINAMATH_CALUDE_rectangle_dimension_change_l581_58124


namespace NUMINAMATH_CALUDE_inequality_equivalence_l581_58139

open Real

theorem inequality_equivalence (k : ℝ) : 
  (∀ x ∈ Set.Ioo 0 2, x / (exp x) < 1 / (k + 2 * x - x^2)) ↔ k ∈ Set.Icc 0 (exp 1 - 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l581_58139


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_real_roots_l581_58151

theorem quadratic_two_distinct_real_roots :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁^2 - 2*x₁ - 1 = 0 ∧ x₂^2 - 2*x₂ - 1 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_real_roots_l581_58151


namespace NUMINAMATH_CALUDE_custom_deck_probability_l581_58163

theorem custom_deck_probability : 
  let total_cards : ℕ := 65
  let spades : ℕ := 14
  let other_suits : ℕ := 13
  let aces : ℕ := 4
  let kings : ℕ := 4
  (aces : ℚ) / total_cards * kings / (total_cards - 1) = 1 / 260 :=
by sorry

end NUMINAMATH_CALUDE_custom_deck_probability_l581_58163


namespace NUMINAMATH_CALUDE_sum_of_cubes_l581_58194

theorem sum_of_cubes (x y z c d : ℝ) 
  (h1 : x * y * z = c)
  (h2 : 1 / x^3 + 1 / y^3 + 1 / z^3 = d) :
  (x + y + z)^3 = d * c^3 + 3 * c - 3 * c * d :=
by sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l581_58194


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l581_58130

theorem pure_imaginary_complex_number (m : ℝ) : 
  let z : ℂ := Complex.mk (m^2 - m - 2) (m + 1)
  (z.re = 0 ∧ z.im ≠ 0) → m = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l581_58130


namespace NUMINAMATH_CALUDE_calculation_result_l581_58110

theorem calculation_result : (25 * 8 + 1 / (5/7)) / (2014 - 201.4 * 2) = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_calculation_result_l581_58110


namespace NUMINAMATH_CALUDE_k_increasing_range_l581_58142

-- Define the function f(x) = x^2
def f (x : ℝ) : ℝ := x^2

-- Define the domain of f
def D : Set ℝ := { x | x ≥ -1 }

-- Define the property of being k-increasing on a set
def is_k_increasing (f : ℝ → ℝ) (k : ℝ) (S : Set ℝ) : Prop :=
  k ≠ 0 ∧ ∀ x ∈ S, (x + k) ∈ S → f (x + k) ≥ f x

-- State the theorem
theorem k_increasing_range (k : ℝ) :
  is_k_increasing f k D → k ≥ 2 := by sorry

end NUMINAMATH_CALUDE_k_increasing_range_l581_58142


namespace NUMINAMATH_CALUDE_aquarium_feeding_ratio_l581_58174

/-- The ratio of buckets fed to other sea animals to buckets fed to sharks -/
def ratio_other_to_sharks : ℚ := 5

theorem aquarium_feeding_ratio : 
  let sharks_buckets : ℕ := 4
  let dolphins_buckets : ℕ := sharks_buckets / 2
  let total_buckets : ℕ := 546
  let days : ℕ := 21
  
  ∃ (other_buckets : ℚ),
    other_buckets = ratio_other_to_sharks * sharks_buckets ∧
    total_buckets = (sharks_buckets + dolphins_buckets + other_buckets) * days :=
by sorry

end NUMINAMATH_CALUDE_aquarium_feeding_ratio_l581_58174


namespace NUMINAMATH_CALUDE_product_primitive_roots_congruent_one_l581_58112

/-- Given a prime p > 3, the product of all primitive roots modulo p is congruent to 1 modulo p -/
theorem product_primitive_roots_congruent_one (p : Nat) (hp : p.Prime) (hp3 : p > 3) :
  ∃ (S : Finset Nat), 
    (∀ s ∈ S, 1 ≤ s ∧ s < p ∧ IsPrimitiveRoot s p) ∧ 
    (∀ x, 1 ≤ x ∧ x < p ∧ IsPrimitiveRoot x p → x ∈ S) ∧
    (S.prod id) % p = 1 := by
  sorry


end NUMINAMATH_CALUDE_product_primitive_roots_congruent_one_l581_58112


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l581_58118

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 1 + a 6 + a 11 = 3 →
  a 3 + a 9 = 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l581_58118


namespace NUMINAMATH_CALUDE_negation_of_implication_negation_of_x_squared_positive_l581_58187

theorem negation_of_implication (P Q : Prop) :
  ¬(P → Q) ↔ (P ∧ ¬Q) :=
by sorry

theorem negation_of_x_squared_positive :
  ¬(∀ x : ℝ, x > 0 → x^2 > 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_implication_negation_of_x_squared_positive_l581_58187


namespace NUMINAMATH_CALUDE_emily_cleaning_time_l581_58120

/-- Represents the cleaning time distribution among four people -/
structure CleaningTime where
  total : ℝ
  lillyAndFiona : ℝ
  jack : ℝ
  emily : ℝ

/-- Theorem stating Emily's cleaning time in minutes -/
theorem emily_cleaning_time (ct : CleaningTime) : 
  ct.total = 8 ∧ 
  ct.lillyAndFiona = 1/4 * ct.total ∧ 
  ct.jack = 1/3 * ct.total ∧ 
  ct.emily = ct.total - ct.lillyAndFiona - ct.jack → 
  ct.emily * 60 = 200 := by
  sorry

#check emily_cleaning_time

end NUMINAMATH_CALUDE_emily_cleaning_time_l581_58120


namespace NUMINAMATH_CALUDE_max_value_theorem_l581_58190

theorem max_value_theorem (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_constraint : a^2 + b^2 + 4*c^2 = 1) :
  ∃ (max : ℝ), max = Real.sqrt 2 ∧ ∀ x, x = a*b + 2*a*c + 3*Real.sqrt 2*b*c → x ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l581_58190


namespace NUMINAMATH_CALUDE_distance_to_nearest_town_l581_58125

theorem distance_to_nearest_town (d : ℝ) : 
  (¬ (d ≥ 8)) → (¬ (d ≤ 7)) → (¬ (d ≤ 6)) → (7 < d ∧ d < 8) := by
  sorry

end NUMINAMATH_CALUDE_distance_to_nearest_town_l581_58125


namespace NUMINAMATH_CALUDE_special_triangle_smallest_angle_cos_l581_58107

/-- A triangle with sides of three consecutive odd numbers where the largest angle is thrice the smallest angle -/
structure SpecialTriangle where
  n : ℕ
  side1 : ℕ := n + 2
  side2 : ℕ := n + 3
  side3 : ℕ := n + 4
  is_valid : side1 + side2 > side3 ∧ side1 + side3 > side2 ∧ side2 + side3 > side1
  largest_angle_triple : Real.cos ((n + 1) / (2 * (n + 2))) = 
    4 * ((n + 5) / (2 * (n + 4))) ^ 3 - 3 * ((n + 5) / (2 * (n + 4)))

/-- The cosine of the smallest angle in a SpecialTriangle is 6/11 -/
theorem special_triangle_smallest_angle_cos (t : SpecialTriangle) : 
  Real.cos ((t.n + 5) / (2 * (t.n + 4))) = 6 / 11 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_smallest_angle_cos_l581_58107


namespace NUMINAMATH_CALUDE_intersection_complement_when_a_2_range_of_a_for_proper_superset_l581_58111

-- Define sets P and Q
def P (a : ℝ) : Set ℝ := {x | 3*a - 10 ≤ x ∧ x < 2*a + 1}
def Q : Set ℝ := {x | |2*x - 3| ≤ 7}

-- Part 1
theorem intersection_complement_when_a_2 : 
  P 2 ∩ (Set.univ \ Q) = {x | -4 ≤ x ∧ x < -2} := by sorry

-- Part 2
theorem range_of_a_for_proper_superset : 
  {a : ℝ | P a ⊃ Q ∧ P a ≠ Q} = Set.Ioo 2 (8/3) := by sorry

end NUMINAMATH_CALUDE_intersection_complement_when_a_2_range_of_a_for_proper_superset_l581_58111


namespace NUMINAMATH_CALUDE_gcd_of_powers_of_101_plus_one_l581_58168

theorem gcd_of_powers_of_101_plus_one (h : Nat.Prime 101) :
  Nat.gcd (101^5 + 1) (101^5 + 101^3 + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_powers_of_101_plus_one_l581_58168


namespace NUMINAMATH_CALUDE_heather_walk_distance_l581_58145

/-- The distance Heather walked from the carnival rides back to the car -/
def carnival_to_car : ℝ := 0.08333333333333333

/-- The total distance Heather walked -/
def total_distance : ℝ := 0.75

/-- The distance from the car to the entrance (and from the entrance to the carnival rides) -/
def car_to_entrance : ℝ := 0.33333333333333335

theorem heather_walk_distance :
  2 * car_to_entrance + carnival_to_car = total_distance :=
by sorry

end NUMINAMATH_CALUDE_heather_walk_distance_l581_58145


namespace NUMINAMATH_CALUDE_volume_formula_l581_58119

/-- A right rectangular prism with edge lengths 2, 3, and 5 -/
structure Prism where
  length : ℝ := 2
  width : ℝ := 3
  height : ℝ := 5

/-- The set of points within distance r of the prism -/
def S (B : Prism) (r : ℝ) : Set (ℝ × ℝ × ℝ) := sorry

/-- The volume of S(r) -/
noncomputable def volume (B : Prism) (r : ℝ) : ℝ := sorry

/-- Coefficients of the volume polynomial -/
structure VolumeCoeffs where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

theorem volume_formula (B : Prism) (coeffs : VolumeCoeffs) :
  (∀ r : ℝ, volume B r = coeffs.a * r^3 + coeffs.b * r^2 + coeffs.c * r + coeffs.d) →
  coeffs.a > 0 ∧ coeffs.b > 0 ∧ coeffs.c > 0 ∧ coeffs.d > 0 →
  coeffs.b * coeffs.c / (coeffs.a * coeffs.d) = 20.67 := by
  sorry

end NUMINAMATH_CALUDE_volume_formula_l581_58119


namespace NUMINAMATH_CALUDE_uncle_ben_chickens_l581_58164

/-- Represents Uncle Ben's farm --/
structure Farm where
  roosters : Nat
  nonLayingHens : Nat
  eggsPerLayingHen : Nat
  totalEggs : Nat

/-- Calculates the total number of chickens on the farm --/
def totalChickens (f : Farm) : Nat :=
  let layingHens := f.totalEggs / f.eggsPerLayingHen
  f.roosters + f.nonLayingHens + layingHens

/-- Theorem stating that Uncle Ben has 440 chickens --/
theorem uncle_ben_chickens :
  ∀ (f : Farm),
    f.roosters = 39 →
    f.nonLayingHens = 15 →
    f.eggsPerLayingHen = 3 →
    f.totalEggs = 1158 →
    totalChickens f = 440 := by
  sorry

end NUMINAMATH_CALUDE_uncle_ben_chickens_l581_58164


namespace NUMINAMATH_CALUDE_eighth_root_of_5487587353601_l581_58128

theorem eighth_root_of_5487587353601 : ∃ n : ℕ, n ^ 8 = 5487587353601 ∧ n = 101 := by
  sorry

end NUMINAMATH_CALUDE_eighth_root_of_5487587353601_l581_58128


namespace NUMINAMATH_CALUDE_correct_calculation_l581_58108

theorem correct_calculation (a b : ℝ) : 2 * a * b + 3 * b * a = 5 * a * b := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l581_58108


namespace NUMINAMATH_CALUDE_mn_inequality_l581_58101

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

-- Define the set P
def P : Set ℝ := {x | f x > 4}

-- State the theorem
theorem mn_inequality (m n : ℝ) (hm : m ∈ P) (hn : n ∈ P) :
  |m * n + 4| > 2 * |m + n| := by
  sorry

end NUMINAMATH_CALUDE_mn_inequality_l581_58101


namespace NUMINAMATH_CALUDE_median_sum_bounds_l581_58182

/-- Given a triangle ABC with medians m_a, m_b, m_c, and perimeter p,
    prove that the sum of the medians is between 3/2 and 2 times the perimeter. -/
theorem median_sum_bounds (m_a m_b m_c p : ℝ) (h_positive : m_a > 0 ∧ m_b > 0 ∧ m_c > 0 ∧ p > 0)
  (h_triangle : ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = p ∧
    m_a^2 = (2*b^2 + 2*c^2 - a^2) / 4 ∧
    m_b^2 = (2*a^2 + 2*c^2 - b^2) / 4 ∧
    m_c^2 = (2*a^2 + 2*b^2 - c^2) / 4) :
  (3/2) * p < m_a + m_b + m_c ∧ m_a + m_b + m_c < 2 * p := by
  sorry

end NUMINAMATH_CALUDE_median_sum_bounds_l581_58182


namespace NUMINAMATH_CALUDE_max_value_cube_plus_one_l581_58192

/-- Given that x + y = 1, prove that (x³+1)(y³+1) achieves its maximum value
    when x = (1 ± √5)/2 and y = (1 ∓ √5)/2 -/
theorem max_value_cube_plus_one (x y : ℝ) (h : x + y = 1) :
  ∃ (max_x max_y : ℝ), 
    (max_x = (1 + Real.sqrt 5) / 2 ∧ max_y = (1 - Real.sqrt 5) / 2) ∨
    (max_x = (1 - Real.sqrt 5) / 2 ∧ max_y = (1 + Real.sqrt 5) / 2) ∧
    ∀ (a b : ℝ), a + b = 1 → 
      (x^3 + 1) * (y^3 + 1) ≤ (max_x^3 + 1) * (max_y^3 + 1) :=
sorry

end NUMINAMATH_CALUDE_max_value_cube_plus_one_l581_58192


namespace NUMINAMATH_CALUDE_min_a_value_l581_58152

-- Define the functions f and g
def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

-- Define the properties of f and g
axiom f_odd : ∀ x, f (-x) = -f x
axiom g_even : ∀ x, g (-x) = g x

-- Define the relationship between f, g, and 2^x
axiom f_g_sum : ∀ x ∈ Set.Icc 1 2, f x + g x = 2^x

-- Define the inequality condition
def inequality_holds (a : ℝ) : Prop :=
  ∀ x ∈ Set.Icc 1 2, a * f x + g (2*x) ≥ 0

-- State the theorem
theorem min_a_value :
  ∃ a_min : ℝ, a_min = -17/6 ∧
  (∀ a, inequality_holds a ↔ a ≥ a_min) :=
sorry

end NUMINAMATH_CALUDE_min_a_value_l581_58152


namespace NUMINAMATH_CALUDE_commuter_distance_commuter_distance_is_12_sqrt_2_l581_58188

/-- The distance from the starting point after a commuter drives 21 miles east, 
    15 miles south, 9 miles west, and 3 miles north. -/
theorem commuter_distance : ℝ :=
  let east : ℝ := 21
  let south : ℝ := 15
  let west : ℝ := 9
  let north : ℝ := 3
  let net_east_west : ℝ := east - west
  let net_south_north : ℝ := south - north
  Real.sqrt (net_east_west ^ 2 + net_south_north ^ 2)

/-- Proof that the commuter's distance from the starting point is 12√2 miles. -/
theorem commuter_distance_is_12_sqrt_2 : commuter_distance = 12 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_commuter_distance_commuter_distance_is_12_sqrt_2_l581_58188


namespace NUMINAMATH_CALUDE_g_of_three_equals_twentyone_l581_58173

-- Define the function g
noncomputable def g : ℝ → ℝ := sorry

-- State the theorem
theorem g_of_three_equals_twentyone :
  (∀ x : ℝ, g (2 * x - 5) = 3 * x + 9) →
  g 3 = 21 := by
  sorry

end NUMINAMATH_CALUDE_g_of_three_equals_twentyone_l581_58173


namespace NUMINAMATH_CALUDE_sixteen_tourists_remain_l581_58189

/-- Calculates the number of tourists remaining after a dangerous rainforest tour --/
def tourists_remaining (initial : ℕ) : ℕ :=
  let after_anaconda := initial - 3
  let poisoned := (2 * after_anaconda) / 3
  let recovered := (2 * poisoned) / 9
  let after_poison := after_anaconda - poisoned + recovered
  let snake_bitten := after_poison / 4
  let saved_from_snakes := (3 * snake_bitten) / 5
  after_poison - snake_bitten + saved_from_snakes

/-- Theorem stating that 16 tourists remain at the end of the tour --/
theorem sixteen_tourists_remain : tourists_remaining 42 = 16 := by
  sorry


end NUMINAMATH_CALUDE_sixteen_tourists_remain_l581_58189


namespace NUMINAMATH_CALUDE_pizza_difference_l581_58181

/-- Given that Seung-hyeon gave Su-yeon 2 pieces of pizza and then had 5 more pieces than Su-yeon,
    prove that Seung-hyeon had 9 more pieces than Su-yeon before giving. -/
theorem pizza_difference (s y : ℕ) : 
  s - 2 = y + 2 + 5 → s - y = 9 := by
  sorry

end NUMINAMATH_CALUDE_pizza_difference_l581_58181


namespace NUMINAMATH_CALUDE_rohits_walk_l581_58102

/-- Given a right triangle with one leg of length 20 and hypotenuse of length 35,
    the length of the other leg is √825. -/
theorem rohits_walk (a b c : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : a = 20) (h3 : c = 35) :
  b = Real.sqrt 825 := by
  sorry

end NUMINAMATH_CALUDE_rohits_walk_l581_58102


namespace NUMINAMATH_CALUDE_batsman_highest_score_l581_58198

theorem batsman_highest_score 
  (total_innings : ℕ) 
  (average : ℚ) 
  (score_difference : ℕ) 
  (average_excluding_extremes : ℚ) 
  (h : total_innings = 46)
  (h1 : average = 60)
  (h2 : score_difference = 180)
  (h3 : average_excluding_extremes = 58) :
  ∃ (highest_score lowest_score : ℕ),
    (highest_score : ℚ) - lowest_score = score_difference ∧
    (highest_score + lowest_score : ℚ) = 
      total_innings * average - (total_innings - 2) * average_excluding_extremes ∧
    highest_score = 194 := by
  sorry

end NUMINAMATH_CALUDE_batsman_highest_score_l581_58198


namespace NUMINAMATH_CALUDE_ocean_area_ratio_l581_58117

theorem ocean_area_ratio (total_area land_area ocean_area : ℝ)
  (land_ratio : land_area / total_area = 29 / 100)
  (ocean_ratio : ocean_area / total_area = 71 / 100)
  (northern_land : ℝ) (southern_land : ℝ)
  (northern_land_ratio : northern_land / land_area = 3 / 4)
  (southern_land_ratio : southern_land / land_area = 1 / 4)
  (northern_ocean southern_ocean : ℝ)
  (northern_hemisphere : northern_land + northern_ocean = total_area / 2)
  (southern_hemisphere : southern_land + southern_ocean = total_area / 2) :
  southern_ocean / northern_ocean = 171 / 113 :=
sorry

end NUMINAMATH_CALUDE_ocean_area_ratio_l581_58117


namespace NUMINAMATH_CALUDE_three_Y_two_equals_one_l581_58162

-- Define the Y operation
def Y (a b : ℝ) : ℝ := a^2 - 2*a*b + b^2

-- Theorem statement
theorem three_Y_two_equals_one : Y 3 2 = 1 := by sorry

end NUMINAMATH_CALUDE_three_Y_two_equals_one_l581_58162


namespace NUMINAMATH_CALUDE_matrix_determinant_l581_58150

theorem matrix_determinant : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![5, 3/2; 2, 6]
  Matrix.det A = 27 := by
sorry

end NUMINAMATH_CALUDE_matrix_determinant_l581_58150


namespace NUMINAMATH_CALUDE_value_of_P_l581_58143

theorem value_of_P (x y : ℝ) 
  (h1 : |x| + x + y = 10) 
  (h2 : |y| + x - y = 10) : 
  x + y = 4 := by
sorry

end NUMINAMATH_CALUDE_value_of_P_l581_58143


namespace NUMINAMATH_CALUDE_extreme_points_count_l581_58175

-- Define the derivative of f(x)
def f_prime (x : ℝ) : ℝ := (x + 1)^2 * (x - 1) * (x - 2)

-- Define what an extreme point is
def is_extreme_point (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∃ ε > 0, ∀ y ∈ Set.Ioo (x - ε) (x + ε), y ≠ x → f y ≠ f x

-- State the theorem
theorem extreme_points_count :
  ∃ (f : ℝ → ℝ), (∀ x, deriv f x = f_prime x) ∧ 
  (∃ (a b : ℝ), a ≠ b ∧ 
    is_extreme_point f a ∧ 
    is_extreme_point f b ∧ 
    ∀ c, is_extreme_point f c → (c = a ∨ c = b)) :=
sorry

end NUMINAMATH_CALUDE_extreme_points_count_l581_58175


namespace NUMINAMATH_CALUDE_all_statements_imply_theorem_l581_58183

theorem all_statements_imply_theorem (p q r : Prop) : 
  ((p ∧ ¬q ∧ r) ∨ (¬p ∧ ¬q ∧ r) ∨ (p ∧ ¬q ∧ ¬r) ∨ (¬p ∧ q ∧ r)) → ((p → q) → r) := by
  sorry

#check all_statements_imply_theorem

end NUMINAMATH_CALUDE_all_statements_imply_theorem_l581_58183


namespace NUMINAMATH_CALUDE_complex_division_equality_l581_58137

theorem complex_division_equality : (2 : ℂ) / (2 - I) = 4/5 + 2/5 * I := by sorry

end NUMINAMATH_CALUDE_complex_division_equality_l581_58137


namespace NUMINAMATH_CALUDE_max_added_value_l581_58115

/-- The added value function for the car manufacturer's production line renovation --/
def f (a : ℝ) (x : ℝ) : ℝ := 8 * (a - x) * x^2

/-- The theorem stating the maximum value of the added value function --/
theorem max_added_value (a : ℝ) (h_a : a > 0) :
  ∃ (x : ℝ), x ∈ Set.Ioo 0 (4*a/5) ∧ 
    (∀ (y : ℝ), y ∈ Set.Ioo 0 (4*a/5) → f a y ≤ f a x) ∧
    f a x = 32 * a^3 / 27 ∧
    x = 2*a/3 := by
  sorry

end NUMINAMATH_CALUDE_max_added_value_l581_58115


namespace NUMINAMATH_CALUDE_fraction_equality_l581_58195

theorem fraction_equality (x y z : ℝ) 
  (h : (x + 4) / 2 = (y + 9) / (z - 3) ∧ (x + 4) / 2 = (x + 5) / (z - 5)) : 
  x / y = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l581_58195


namespace NUMINAMATH_CALUDE_circle_intersection_chord_length_l581_58153

/-- A circle in the xy-plane -/
structure Circle where
  a : ℝ
  equation : ℝ → ℝ → Prop :=
    fun x y ↦ x^2 + y^2 + 2*x - 2*y + a = 0

/-- A line in the xy-plane -/
def Line : ℝ → ℝ → Prop :=
  fun x y ↦ x + y + 2 = 0

/-- The length of a chord formed by the intersection of a circle and a line -/
def ChordLength (c : Circle) : ℝ :=
  4 -- Given in the problem

/-- The main theorem -/
theorem circle_intersection_chord_length (c : Circle) :
  (∀ x y, Line x y → c.equation x y) →
  ChordLength c = 4 →
  c.a = -4 := by
  sorry

end NUMINAMATH_CALUDE_circle_intersection_chord_length_l581_58153


namespace NUMINAMATH_CALUDE_min_stamps_for_30_cents_l581_58141

/-- Represents the number of stamps needed to make a certain value -/
structure StampCombination :=
  (threes : ℕ)
  (fours : ℕ)

/-- Calculates the total value of stamps in cents -/
def value (s : StampCombination) : ℕ := 3 * s.threes + 4 * s.fours

/-- Calculates the total number of stamps -/
def total_stamps (s : StampCombination) : ℕ := s.threes + s.fours

/-- Checks if a StampCombination is valid for the given target value -/
def is_valid (s : StampCombination) (target : ℕ) : Prop :=
  value s = target

/-- Theorem: The minimum number of stamps needed to make 30 cents is 8 -/
theorem min_stamps_for_30_cents :
  ∃ (s : StampCombination), is_valid s 30 ∧
    total_stamps s = 8 ∧
    (∀ (t : StampCombination), is_valid t 30 → total_stamps s ≤ total_stamps t) :=
sorry

end NUMINAMATH_CALUDE_min_stamps_for_30_cents_l581_58141


namespace NUMINAMATH_CALUDE_sqrt_two_thirds_times_sqrt_six_equals_two_l581_58196

theorem sqrt_two_thirds_times_sqrt_six_equals_two :
  Real.sqrt (2 / 3) * Real.sqrt 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_thirds_times_sqrt_six_equals_two_l581_58196


namespace NUMINAMATH_CALUDE_three_buildings_height_l581_58147

/-- The height of three buildings given specific conditions -/
theorem three_buildings_height 
  (h1 : ℕ) -- Height of the first building
  (h2_eq : h2 = 2 * h1) -- Second building is twice as tall as the first
  (h3_eq : h3 = 3 * (h1 + h2)) -- Third building is three times as tall as the first two combined
  (h1_val : h1 = 600) -- First building is 600 feet tall
  : h1 + h2 + h3 = 7200 := by
  sorry

#check three_buildings_height

end NUMINAMATH_CALUDE_three_buildings_height_l581_58147


namespace NUMINAMATH_CALUDE_family_ages_solution_l581_58123

structure Family where
  father_age : ℕ
  mother_age : ℕ
  john_age : ℕ
  ben_age : ℕ
  mary_age : ℕ

def age_difference (f : Family) : ℕ :=
  f.father_age - f.mother_age

theorem family_ages_solution (f : Family) 
  (h1 : age_difference f = f.john_age - f.ben_age)
  (h2 : age_difference f = f.ben_age - f.mary_age)
  (h3 : f.john_age * f.ben_age = f.father_age)
  (h4 : f.ben_age * f.mary_age = f.mother_age)
  (h5 : f.father_age + f.mother_age + f.john_age + f.ben_age + f.mary_age = 90)
  : f.father_age = 36 ∧ f.mother_age = 36 ∧ f.john_age = 6 ∧ f.ben_age = 6 ∧ f.mary_age = 6 := by
  sorry

end NUMINAMATH_CALUDE_family_ages_solution_l581_58123


namespace NUMINAMATH_CALUDE_smallest_k_for_inequality_l581_58179

theorem smallest_k_for_inequality : ∃ k : ℕ, k = 8 ∧ 
  (∀ m : ℕ, 64^m > 4^22 → m ≥ k) ∧ 64^k > 4^22 := by
  sorry

end NUMINAMATH_CALUDE_smallest_k_for_inequality_l581_58179


namespace NUMINAMATH_CALUDE_trapezoid_cd_length_l581_58161

/-- Represents a trapezoid ABCD with specific properties -/
structure Trapezoid where
  -- Length of segment AB
  ab : ℝ
  -- Length of segment CD
  cd : ℝ
  -- The ratio of the area of triangle ABC to the area of triangle ADC is 5:3
  area_ratio : ab / cd = 5 / 3
  -- The sum of AB and CD is 192 cm
  sum_sides : ab + cd = 192

/-- 
Theorem: In a trapezoid ABCD, if the ratio of the area of triangle ABC to the area of triangle ADC 
is 5:3, and AB + CD = 192 cm, then the length of segment CD is 72 cm.
-/
theorem trapezoid_cd_length (t : Trapezoid) : t.cd = 72 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_cd_length_l581_58161


namespace NUMINAMATH_CALUDE_tangent_point_coordinates_l581_58131

/-- A point on the curve y = x^3 - 3x with a tangent line parallel to the x-axis -/
structure TangentPoint where
  x : ℝ
  y : ℝ
  on_curve : y = x^3 - 3*x
  parallel_tangent : 3*x^2 - 3 = 0

theorem tangent_point_coordinates (P : TangentPoint) : 
  (P.x = 1 ∧ P.y = -2) ∨ (P.x = -1 ∧ P.y = 2) := by
  sorry

end NUMINAMATH_CALUDE_tangent_point_coordinates_l581_58131


namespace NUMINAMATH_CALUDE_remainder_of_98_times_102_mod_9_l581_58148

theorem remainder_of_98_times_102_mod_9 : (98 * 102) % 9 = 8 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_98_times_102_mod_9_l581_58148


namespace NUMINAMATH_CALUDE_unique_number_exists_l581_58165

def is_valid_number (n : ℕ) : Prop :=
  ∀ k ∈ Finset.range 10, n % (k + 3) = k + 2

theorem unique_number_exists : 
  ∃! n : ℕ, is_valid_number n ∧ n = 27719 := by sorry

end NUMINAMATH_CALUDE_unique_number_exists_l581_58165


namespace NUMINAMATH_CALUDE_flour_to_baking_soda_ratio_l581_58105

/-- Prove that the ratio of flour to baking soda is 10 to 1 given the specified conditions -/
theorem flour_to_baking_soda_ratio 
  (sugar : ℕ) 
  (flour : ℕ) 
  (baking_soda : ℕ) 
  (h1 : sugar * 6 = flour * 5) 
  (h2 : sugar = 2000) 
  (h3 : flour = 8 * (baking_soda + 60)) : 
  flour / baking_soda = 10 := by
  sorry

end NUMINAMATH_CALUDE_flour_to_baking_soda_ratio_l581_58105


namespace NUMINAMATH_CALUDE_distance_between_ports_l581_58157

/-- The distance between ports A and B in kilometers -/
def distance_AB : ℝ := 40

/-- The speed of the ship in still water in km/h -/
def ship_speed : ℝ := 26

/-- The speed of the river current in km/h -/
def current_speed : ℝ := 6

/-- The number of round trips made by the ship -/
def round_trips : ℕ := 4

/-- The total time taken for all round trips in hours -/
def total_time : ℝ := 13

theorem distance_between_ports :
  let downstream_speed := ship_speed + current_speed
  let upstream_speed := ship_speed - current_speed
  let time_per_round_trip := total_time / round_trips
  let downstream_time := (upstream_speed * time_per_round_trip) / (downstream_speed + upstream_speed)
  distance_AB = downstream_speed * downstream_time :=
by sorry

end NUMINAMATH_CALUDE_distance_between_ports_l581_58157


namespace NUMINAMATH_CALUDE_friends_hiking_distance_l581_58178

-- Define the hiking scenario
structure HikingScenario where
  total_time : Real
  birgit_time_diff : Real
  birgit_time : Real
  birgit_distance : Real

-- Define the theorem
theorem friends_hiking_distance (h : HikingScenario) 
  (h_total_time : h.total_time = 3.5) 
  (h_birgit_time_diff : h.birgit_time_diff = 4) 
  (h_birgit_time : h.birgit_time = 48) 
  (h_birgit_distance : h.birgit_distance = 8) : 
  (h.total_time * 60) / (h.birgit_time / h.birgit_distance + h.birgit_time_diff) = 21 := by
  sorry


end NUMINAMATH_CALUDE_friends_hiking_distance_l581_58178


namespace NUMINAMATH_CALUDE_remaining_coin_value_l581_58191

/-- Represents the number and type of coins --/
structure Coins where
  quarters : Nat
  dimes : Nat
  nickels : Nat

/-- Calculates the total value of coins in cents --/
def coinValue (c : Coins) : Nat :=
  c.quarters * 25 + c.dimes * 10 + c.nickels * 5

/-- Represents Olivia's initial coins --/
def initialCoins : Coins :=
  { quarters := 11, dimes := 15, nickels := 7 }

/-- Represents the coins spent on purchases --/
def purchasedCoins : Coins :=
  { quarters := 1, dimes := 8, nickels := 3 }

/-- Calculates the remaining coins after purchases --/
def remainingCoins (initial : Coins) (purchased : Coins) : Coins :=
  { quarters := initial.quarters - purchased.quarters,
    dimes := initial.dimes - purchased.dimes,
    nickels := initial.nickels - purchased.nickels }

theorem remaining_coin_value :
  coinValue (remainingCoins initialCoins purchasedCoins) = 340 := by
  sorry


end NUMINAMATH_CALUDE_remaining_coin_value_l581_58191


namespace NUMINAMATH_CALUDE_course_selection_combinations_l581_58193

theorem course_selection_combinations :
  let total_courses : ℕ := 7
  let required_courses : ℕ := 2
  let math_courses : ℕ := 2
  let program_size : ℕ := 5
  let remaining_courses : ℕ := total_courses - required_courses
  let remaining_selections : ℕ := program_size - required_courses

  (Nat.choose remaining_courses remaining_selections) -
  (Nat.choose (remaining_courses - math_courses) remaining_selections) = 9 :=
by sorry

end NUMINAMATH_CALUDE_course_selection_combinations_l581_58193


namespace NUMINAMATH_CALUDE_xyz_value_l581_58154

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 40)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) + 2 * x * y * z = 20) :
  x * y * z = 20 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l581_58154


namespace NUMINAMATH_CALUDE_some_number_value_l581_58104

theorem some_number_value (a n : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 25 * n * 49) : n = 9 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l581_58104


namespace NUMINAMATH_CALUDE_cousins_ages_sum_l581_58177

theorem cousins_ages_sum : 
  ∀ (a b c d : ℕ),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
    0 < a ∧ a < 10 →
    0 < b ∧ b < 10 →
    0 < c ∧ c < 10 →
    0 < d ∧ d < 10 →
    (a * b = 40 ∨ a * c = 40 ∨ a * d = 40 ∨ b * c = 40 ∨ b * d = 40 ∨ c * d = 40) →
    (a * b = 36 ∨ a * c = 36 ∨ a * d = 36 ∨ b * c = 36 ∨ b * d = 36 ∨ c * d = 36) →
    a + b + c + d = 26 :=
by sorry

end NUMINAMATH_CALUDE_cousins_ages_sum_l581_58177


namespace NUMINAMATH_CALUDE_unique_positive_solution_l581_58158

theorem unique_positive_solution : ∃! (x : ℝ), x > 0 ∧ (x - 6) / 12 = 6 / (x - 12) := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l581_58158


namespace NUMINAMATH_CALUDE_divisibility_implies_p_q_values_l581_58171

/-- A polynomial is divisible by (x + 2)(x - 2) if and only if it equals zero when x = 2 and x = -2 -/
def is_divisible_by_x2_minus4 (f : ℝ → ℝ) : Prop :=
  f 2 = 0 ∧ f (-2) = 0

/-- The polynomial x^5 - x^4 + x^3 - px^2 + qx - 8 -/
def polynomial (p q : ℝ) (x : ℝ) : ℝ :=
  x^5 - x^4 + x^3 - p*x^2 + q*x - 8

theorem divisibility_implies_p_q_values :
  ∀ p q : ℝ, is_divisible_by_x2_minus4 (polynomial p q) → p = -2 ∧ q = -12 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implies_p_q_values_l581_58171


namespace NUMINAMATH_CALUDE_sum_of_solutions_is_four_l581_58186

theorem sum_of_solutions_is_four :
  let f : ℝ → ℝ := λ N ↦ N * (N - 4) - 12
  ∃ N₁ N₂ : ℝ, (f N₁ = 0 ∧ f N₂ = 0) ∧ N₁ + N₂ = 4 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_is_four_l581_58186
