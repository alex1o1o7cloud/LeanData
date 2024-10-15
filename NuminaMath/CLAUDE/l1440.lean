import Mathlib

namespace NUMINAMATH_CALUDE_floor_abs_negative_l1440_144009

theorem floor_abs_negative : ⌊|(-57.8 : ℝ)|⌋ = 57 := by sorry

end NUMINAMATH_CALUDE_floor_abs_negative_l1440_144009


namespace NUMINAMATH_CALUDE_two_numbers_difference_l1440_144050

theorem two_numbers_difference (a b : ℕ) 
  (h1 : a + b = 12390)
  (h2 : b = 2 * a + 18) : 
  b - a = 4142 := by
sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l1440_144050


namespace NUMINAMATH_CALUDE_chicken_count_l1440_144063

/-- Given a farm with chickens and buffalos, prove the number of chickens. -/
theorem chicken_count (total_animals : ℕ) (total_legs : ℕ) (chickens : ℕ) (buffalos : ℕ) : 
  total_animals = 9 →
  total_legs = 26 →
  chickens + buffalos = total_animals →
  2 * chickens + 4 * buffalos = total_legs →
  chickens = 5 := by
sorry

end NUMINAMATH_CALUDE_chicken_count_l1440_144063


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1440_144018

theorem quadratic_factorization (a : ℝ) : a^2 + 4*a - 21 = (a - 3) * (a + 7) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1440_144018


namespace NUMINAMATH_CALUDE_square_to_circle_ratio_l1440_144076

-- Define the sector and its properties
structure RectangularSector where
  R : ℝ  -- Radius of the sector
  a : ℝ  -- Side length of the inscribed square

-- Define the circle touching the chord, arc, and square side
def TouchingCircle (sector : RectangularSector) :=
  { r : ℝ // r > 0 }

-- State the theorem
theorem square_to_circle_ratio
  (sector : RectangularSector)
  (circle : TouchingCircle sector) :
  sector.a / circle.val =
    ((Real.sqrt 5 + Real.sqrt 2) * (3 + Real.sqrt 5)) / (6 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_square_to_circle_ratio_l1440_144076


namespace NUMINAMATH_CALUDE_sum_of_absolute_values_zero_l1440_144057

theorem sum_of_absolute_values_zero (a b : ℝ) : 
  |a + 2| + |b - 7| = 0 → a + b = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_absolute_values_zero_l1440_144057


namespace NUMINAMATH_CALUDE_modular_inverse_of_seven_mod_2003_l1440_144053

theorem modular_inverse_of_seven_mod_2003 : ∃ x : ℕ, x < 2003 ∧ (7 * x) % 2003 = 1 :=
by
  use 1717
  sorry

end NUMINAMATH_CALUDE_modular_inverse_of_seven_mod_2003_l1440_144053


namespace NUMINAMATH_CALUDE_multiple_of_smaller_integer_l1440_144048

theorem multiple_of_smaller_integer (s l : ℤ) (k : ℚ) : 
  s + l = 30 → 
  s = 10 → 
  2 * l = k * s - 10 → 
  k = 5 := by sorry

end NUMINAMATH_CALUDE_multiple_of_smaller_integer_l1440_144048


namespace NUMINAMATH_CALUDE_discounted_price_calculation_l1440_144078

def television_price : ℝ := 650
def number_of_televisions : ℕ := 2
def discount_percentage : ℝ := 0.25

theorem discounted_price_calculation :
  let total_price := television_price * number_of_televisions
  let discount_amount := total_price * discount_percentage
  let final_price := total_price - discount_amount
  final_price = 975 := by
  sorry

end NUMINAMATH_CALUDE_discounted_price_calculation_l1440_144078


namespace NUMINAMATH_CALUDE_pieces_present_l1440_144083

/-- The number of pieces in a standard chess set -/
def standard_chess_pieces : ℕ := 32

/-- The number of missing pieces -/
def missing_pieces : ℕ := 4

/-- Theorem: The number of pieces present in an incomplete chess set -/
theorem pieces_present (standard : ℕ) (missing : ℕ) 
  (h1 : standard = standard_chess_pieces) 
  (h2 : missing = missing_pieces) : 
  standard - missing = 28 := by
  sorry

end NUMINAMATH_CALUDE_pieces_present_l1440_144083


namespace NUMINAMATH_CALUDE_value_calculation_l1440_144059

theorem value_calculation (a b : ℝ) (h1 : a + b = 5) (h2 : a * b = 1) :
  a + a^3 / b^2 + b^3 / a^2 + b = 2535 := by
  sorry

end NUMINAMATH_CALUDE_value_calculation_l1440_144059


namespace NUMINAMATH_CALUDE_power_of_product_equality_l1440_144086

theorem power_of_product_equality (x y : ℝ) : (x^2 * y)^3 = x^6 * y^3 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_equality_l1440_144086


namespace NUMINAMATH_CALUDE_james_lifting_ratio_l1440_144023

def initial_total : ℝ := 2200
def initial_weight : ℝ := 245
def total_gain_percentage : ℝ := 0.15
def weight_gain : ℝ := 8

def new_total : ℝ := initial_total * (1 + total_gain_percentage)
def new_weight : ℝ := initial_weight + weight_gain

theorem james_lifting_ratio :
  new_total / new_weight = 10 := by sorry

end NUMINAMATH_CALUDE_james_lifting_ratio_l1440_144023


namespace NUMINAMATH_CALUDE_triangle_longest_side_l1440_144069

theorem triangle_longest_side (x : ℝ) : 
  5 + (x + 3) + (3 * x - 2) = 40 → 
  max 5 (max (x + 3) (3 * x - 2)) = 23.5 := by
sorry

end NUMINAMATH_CALUDE_triangle_longest_side_l1440_144069


namespace NUMINAMATH_CALUDE_real_number_inequalities_l1440_144099

theorem real_number_inequalities (a b c : ℝ) :
  (∀ (c : ℝ), c ≠ 0 → (a * c^2 > b * c^2 → a > b)) ∧
  (a < b ∧ b < 0 → a^2 > a * b) ∧
  (∃ (a b c : ℝ), c > a ∧ a > b ∧ b > 0 ∧ a / (c - a) ≥ b / (c - b)) ∧
  (a > b ∧ b > 1 → a - 1 / b > b - 1 / a) :=
by sorry

end NUMINAMATH_CALUDE_real_number_inequalities_l1440_144099


namespace NUMINAMATH_CALUDE_concert_ticket_price_l1440_144003

/-- Proves that the cost of each ticket is $30 given the concert conditions --/
theorem concert_ticket_price :
  ∀ (ticket_price : ℝ),
    (500 : ℝ) * ticket_price * 0.7 = (4 : ℝ) * 2625 →
    ticket_price = 30 := by
  sorry

end NUMINAMATH_CALUDE_concert_ticket_price_l1440_144003


namespace NUMINAMATH_CALUDE_fraction_simplification_l1440_144092

theorem fraction_simplification (x y : ℚ) 
  (hx : x = 4/7) 
  (hy : y = 5/8) : 
  (6*x - 4*y) / (36*x*y) = 13/180 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1440_144092


namespace NUMINAMATH_CALUDE_figure_b_impossible_l1440_144043

-- Define the shape of a square
structure Square :=
  (side : ℝ)
  (area : ℝ := side * side)

-- Define the set of available squares
def available_squares : Finset Square := sorry

-- Define the shapes of the five figures
inductive Figure
| A
| B
| C
| D
| E

-- Function to check if a figure can be formed from the available squares
def can_form_figure (f : Figure) (squares : Finset Square) : Prop := sorry

-- Theorem stating that figure B cannot be formed while others can
theorem figure_b_impossible :
  (∀ s ∈ available_squares, s.side = 1) →
  (available_squares.card = 17) →
  (¬ can_form_figure Figure.B available_squares) ∧
  (can_form_figure Figure.A available_squares) ∧
  (can_form_figure Figure.C available_squares) ∧
  (can_form_figure Figure.D available_squares) ∧
  (can_form_figure Figure.E available_squares) :=
by sorry

end NUMINAMATH_CALUDE_figure_b_impossible_l1440_144043


namespace NUMINAMATH_CALUDE_range_of_a_for_full_range_l1440_144001

/-- Piecewise function f(x) defined by a real parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then a * x - 1 else x^2 - 2 * a * x

/-- The range of f(x) is all real numbers -/
def has_full_range (a : ℝ) : Prop :=
  ∀ y : ℝ, ∃ x : ℝ, f a x = y

/-- The range of a for which f(x) has a full range is [2/3, +∞) -/
theorem range_of_a_for_full_range :
  {a : ℝ | has_full_range a} = {a : ℝ | a ≥ 2/3} :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_full_range_l1440_144001


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_area_isosceles_trapezoid_area_is_768_l1440_144029

/-- An isosceles trapezoid with the given properties has an area of 768 sq cm. -/
theorem isosceles_trapezoid_area : ℝ → ℝ → ℝ → ℝ → Prop :=
  fun leg_length diagonal_length longer_base area =>
    leg_length = 30 ∧
    diagonal_length = 40 ∧
    longer_base = 50 ∧
    area = 768 ∧
    ∃ (height shorter_base : ℝ),
      height > 0 ∧
      shorter_base > 0 ∧
      shorter_base < longer_base ∧
      leg_length^2 = height^2 + ((longer_base - shorter_base) / 2)^2 ∧
      diagonal_length^2 = height^2 + (longer_base^2 / 4) ∧
      area = (longer_base + shorter_base) * height / 2

/-- The isosceles trapezoid with the given properties has an area of 768 sq cm. -/
theorem isosceles_trapezoid_area_is_768 : isosceles_trapezoid_area 30 40 50 768 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_area_isosceles_trapezoid_area_is_768_l1440_144029


namespace NUMINAMATH_CALUDE_max_value_of_f_l1440_144074

-- Define the function f(x)
def f (b : ℝ) (x : ℝ) : ℝ := x^2 - 2*b*x + 3

-- State the theorem
theorem max_value_of_f (b : ℝ) :
  (∀ x ∈ Set.Icc (-1) 2, f b x ≥ 1) →
  (∃ x ∈ Set.Icc (-1) 2, f b x = 1) →
  (∃ x ∈ Set.Icc (-1) 2, f b x = max 13 (4 + 2*Real.sqrt 2)) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l1440_144074


namespace NUMINAMATH_CALUDE_salary_savings_percentage_l1440_144097

theorem salary_savings_percentage (last_year_salary : ℝ) (last_year_savings_percentage : ℝ) : 
  last_year_savings_percentage > 0 →
  (0.15 * (1.1 * last_year_salary) = 1.65 * (last_year_savings_percentage / 100 * last_year_salary)) →
  last_year_savings_percentage = 10 := by
sorry

end NUMINAMATH_CALUDE_salary_savings_percentage_l1440_144097


namespace NUMINAMATH_CALUDE_sin_40_tan_10_minus_sqrt_3_l1440_144024

/-- Prove that sin 40° * (tan 10° - √3) = -1 -/
theorem sin_40_tan_10_minus_sqrt_3 :
  Real.sin (40 * π / 180) * (Real.tan (10 * π / 180) - Real.sqrt 3) = -1 := by
  sorry

end NUMINAMATH_CALUDE_sin_40_tan_10_minus_sqrt_3_l1440_144024


namespace NUMINAMATH_CALUDE_candies_given_to_stephanie_l1440_144051

theorem candies_given_to_stephanie (initial_candies remaining_candies : ℕ) 
  (h1 : initial_candies = 95)
  (h2 : remaining_candies = 92) :
  initial_candies - remaining_candies = 3 := by
  sorry

end NUMINAMATH_CALUDE_candies_given_to_stephanie_l1440_144051


namespace NUMINAMATH_CALUDE_orange_bucket_difference_l1440_144008

theorem orange_bucket_difference (bucket1 bucket2 bucket3 total : ℕ) : 
  bucket1 = 22 →
  bucket2 = bucket1 + 17 →
  bucket3 < bucket2 →
  total = bucket1 + bucket2 + bucket3 →
  total = 89 →
  bucket2 - bucket3 = 11 := by
sorry

end NUMINAMATH_CALUDE_orange_bucket_difference_l1440_144008


namespace NUMINAMATH_CALUDE_frank_skee_ball_tickets_proof_l1440_144090

def frank_skee_ball_tickets (whack_a_mole_tickets : ℕ) (candy_cost : ℕ) (candies_bought : ℕ) : ℕ :=
  candies_bought * candy_cost - whack_a_mole_tickets

theorem frank_skee_ball_tickets_proof :
  frank_skee_ball_tickets 33 6 7 = 9 := by
  sorry

end NUMINAMATH_CALUDE_frank_skee_ball_tickets_proof_l1440_144090


namespace NUMINAMATH_CALUDE_equality_of_two_numbers_l1440_144081

theorem equality_of_two_numbers (x y z : ℝ) 
  (h : x * y + z = y * z + x ∧ y * z + x = z * x + y) : 
  x = y ∨ y = z ∨ z = x := by
  sorry

end NUMINAMATH_CALUDE_equality_of_two_numbers_l1440_144081


namespace NUMINAMATH_CALUDE_revenue_decrease_percent_l1440_144061

/-- Calculates the percentage decrease in revenue when tax is reduced and consumption is increased -/
theorem revenue_decrease_percent (tax_reduction : Real) (consumption_increase : Real)
  (h1 : tax_reduction = 0.20)
  (h2 : consumption_increase = 0.05) :
  1 - (1 - tax_reduction) * (1 + consumption_increase) = 0.16 := by
  sorry

end NUMINAMATH_CALUDE_revenue_decrease_percent_l1440_144061


namespace NUMINAMATH_CALUDE_triangle_properties_l1440_144058

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove the area and cosine of angle ADC where D is the midpoint of BC. -/
theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  c = 4 →
  b = 3 →
  A = π / 3 →
  let S := (1 / 2) * b * c * Real.sin A
  let cos_ADC := (7 * Real.sqrt 481) / 481
  (S = 3 * Real.sqrt 3 ∧ cos_ADC = (7 * Real.sqrt 481) / 481) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l1440_144058


namespace NUMINAMATH_CALUDE_smallest_k_for_three_reals_l1440_144025

theorem smallest_k_for_three_reals : ∃ (k : ℝ),
  (∀ (x y z : ℝ), x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 →
    (|x - y| ≤ k ∨ |1/x - 1/y| ≤ k) ∨
    (|y - z| ≤ k ∨ |1/y - 1/z| ≤ k) ∨
    (|x - z| ≤ k ∨ |1/x - 1/z| ≤ k)) ∧
  (∀ (k' : ℝ), k' < k →
    ∃ (x y z : ℝ), x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧
      (|x - y| > k' ∧ |1/x - 1/y| > k') ∧
      (|y - z| > k' ∧ |1/y - 1/z| > k') ∧
      (|x - z| > k' ∧ |1/x - 1/z| > k')) ∧
  k = 1.5 :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_for_three_reals_l1440_144025


namespace NUMINAMATH_CALUDE_factories_unchecked_l1440_144052

theorem factories_unchecked (total : ℕ) (first_group : ℕ) (second_group : ℕ)
  (h1 : total = 169)
  (h2 : first_group = 69)
  (h3 : second_group = 52) :
  total - (first_group + second_group) = 48 := by
  sorry

end NUMINAMATH_CALUDE_factories_unchecked_l1440_144052


namespace NUMINAMATH_CALUDE_denny_followers_after_one_year_l1440_144082

/-- Calculates the number of followers after one year --/
def followers_after_one_year (initial_followers : ℕ) (daily_new_followers : ℕ) (unfollows_per_year : ℕ) : ℕ :=
  initial_followers + daily_new_followers * 365 - unfollows_per_year

/-- Theorem stating that Denny will have 445,000 followers after one year --/
theorem denny_followers_after_one_year :
  followers_after_one_year 100000 1000 20000 = 445000 := by
  sorry

#eval followers_after_one_year 100000 1000 20000

end NUMINAMATH_CALUDE_denny_followers_after_one_year_l1440_144082


namespace NUMINAMATH_CALUDE_fraction_power_equality_l1440_144016

theorem fraction_power_equality (x y : ℝ) (h1 : x > y) (h2 : y > 0) :
  (x^y * y^x) / (y^y * x^x) = (x/y)^(y-x) := by sorry

end NUMINAMATH_CALUDE_fraction_power_equality_l1440_144016


namespace NUMINAMATH_CALUDE_pizza_cost_l1440_144042

/-- The cost of purchasing pizzas with special pricing -/
theorem pizza_cost (standard_price : ℕ) (triple_cheese_count : ℕ) (meat_lovers_count : ℕ) :
  standard_price = 5 →
  triple_cheese_count = 10 →
  meat_lovers_count = 9 →
  (standard_price * (triple_cheese_count / 2 + 2 * meat_lovers_count / 3) : ℕ) = 55 := by
  sorry

#check pizza_cost

end NUMINAMATH_CALUDE_pizza_cost_l1440_144042


namespace NUMINAMATH_CALUDE_sum_solution_equations_find_a_value_l1440_144017

/-- Definition of a "sum solution equation" -/
def is_sum_solution_equation (a b : ℚ) : Prop :=
  (b / a) = b + a

/-- Theorem for the given equations -/
theorem sum_solution_equations :
  is_sum_solution_equation (-3) (9/4) ∧
  ¬is_sum_solution_equation (2/3) (-2/3) ∧
  ¬is_sum_solution_equation 5 (-2) :=
sorry

/-- Theorem for finding the value of a -/
theorem find_a_value (a : ℚ) :
  is_sum_solution_equation 3 (2*a - 10) → a = 11/4 :=
sorry

end NUMINAMATH_CALUDE_sum_solution_equations_find_a_value_l1440_144017


namespace NUMINAMATH_CALUDE_intersection_point_and_lines_l1440_144041

/-- Given two lines that intersect at point P, this theorem proves:
    1. The equation of a line passing through P and parallel to a given line
    2. The equation of a line passing through P that maximizes the distance from the origin --/
theorem intersection_point_and_lines (x y : ℝ) :
  (2 * x + y = 8) →
  (x - 2 * y = -1) →
  (∃ P : ℝ × ℝ, P.1 = x ∧ P.2 = y) →
  (∃ l₁ : ℝ → ℝ → Prop, l₁ x y ↔ 4 * x - 3 * y = 6) ∧
  (∃ l₂ : ℝ → ℝ → Prop, l₂ x y ↔ 3 * x + 2 * y = 13) :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_and_lines_l1440_144041


namespace NUMINAMATH_CALUDE_vector_translation_result_l1440_144035

def vector_translation (a : ℝ × ℝ) (right : ℝ) (down : ℝ) : ℝ × ℝ :=
  (a.1 + right, a.2 - down)

theorem vector_translation_result :
  let a : ℝ × ℝ := (1, 1)
  let b : ℝ × ℝ := vector_translation a 2 1
  b = (3, 0) := by sorry

end NUMINAMATH_CALUDE_vector_translation_result_l1440_144035


namespace NUMINAMATH_CALUDE_star_removal_theorem_l1440_144068

/-- Represents a 2n × 2n table with stars -/
structure StarTable (n : ℕ) where
  stars : Finset ((Fin (2*n)) × (Fin (2*n)))
  star_count : stars.card = 3*n

/-- Represents a selection of rows and columns -/
structure Selection (n : ℕ) where
  rows : Finset (Fin (2*n))
  columns : Finset (Fin (2*n))
  row_count : rows.card = n
  column_count : columns.card = n

/-- Predicate to check if a star is removed by a selection -/
def is_removed (star : (Fin (2*n)) × (Fin (2*n))) (sel : Selection n) : Prop :=
  star.1 ∈ sel.rows ∨ star.2 ∈ sel.columns

/-- Theorem: For any 2n × 2n table with 3n stars, there exists a selection
    of n rows and n columns that removes all stars -/
theorem star_removal_theorem (n : ℕ) (table : StarTable n) :
  ∃ (sel : Selection n), ∀ star ∈ table.stars, is_removed star sel :=
sorry

end NUMINAMATH_CALUDE_star_removal_theorem_l1440_144068


namespace NUMINAMATH_CALUDE_pages_ratio_l1440_144026

theorem pages_ratio (lana_initial : ℕ) (duane_initial : ℕ) (lana_final : ℕ)
  (h1 : lana_initial = 8)
  (h2 : duane_initial = 42)
  (h3 : lana_final = 29) :
  (lana_final - lana_initial) * 2 = duane_initial :=
by sorry

end NUMINAMATH_CALUDE_pages_ratio_l1440_144026


namespace NUMINAMATH_CALUDE_kevin_koala_leaves_kevin_koala_leaves_min_l1440_144087

theorem kevin_koala_leaves (n : ℕ) : n > 1 ∧ ∃ k : ℕ, n^2 = k^6 → n ≥ 8 :=
by sorry

theorem kevin_koala_leaves_min : ∃ k : ℕ, 8^2 = k^6 :=
by sorry

end NUMINAMATH_CALUDE_kevin_koala_leaves_kevin_koala_leaves_min_l1440_144087


namespace NUMINAMATH_CALUDE_seven_twelfths_decimal_l1440_144002

theorem seven_twelfths_decimal : 7 / 12 = 0.5833333333333333 := by sorry

end NUMINAMATH_CALUDE_seven_twelfths_decimal_l1440_144002


namespace NUMINAMATH_CALUDE_line_and_circle_problem_l1440_144015

-- Define the lines
def line1 (x y : ℝ) : Prop := 2 * x - y - 3 = 0
def line2 (x y : ℝ) : Prop := 4 * x - 3 * y - 5 = 0
def line3 (x y : ℝ) : Prop := x + y - 2 = 0

-- Define line l
def line_l (x y : ℝ) : Prop := y = x - 1

-- Define circle C
def circle_C (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 4

-- Theorem statement
theorem line_and_circle_problem :
  ∀ (x_int y_int : ℝ),
    (line1 x_int y_int ∧ line2 x_int y_int) →  -- Intersection point condition
    (∀ (x y : ℝ), line_l x y → (x + y - 2 ≠ 0)) →  -- Perpendicularity condition
    circle_C 1 0 →  -- Circle passes through (1,0)
    (∃ (a : ℝ), a > 0 ∧ circle_C a 0) →  -- Center on positive x-axis
    (∃ (x1 y1 x2 y2 : ℝ),
      line_l x1 y1 ∧ line_l x2 y2 ∧
      circle_C x1 y1 ∧ circle_C x2 y2 ∧
      (x2 - x1)^2 + (y2 - y1)^2 = 8) →  -- Chord length condition
    (∀ (x y : ℝ), line_l x y ↔ y = x - 1) ∧
    (∀ (x y : ℝ), circle_C x y ↔ (x - 3)^2 + y^2 = 4) :=
by sorry

end NUMINAMATH_CALUDE_line_and_circle_problem_l1440_144015


namespace NUMINAMATH_CALUDE_polygon_diagonals_minus_sides_l1440_144085

theorem polygon_diagonals_minus_sides (n : ℕ) (h : n = 105) : 
  (n * (n - 3)) / 2 - n = 5250 := by
  sorry

end NUMINAMATH_CALUDE_polygon_diagonals_minus_sides_l1440_144085


namespace NUMINAMATH_CALUDE_inner_triangle_perimeter_is_330_75_l1440_144028

/-- Triangle ABC with given side lengths and parallel lines forming a new triangle -/
structure TriangleWithParallelLines where
  -- Side lengths of triangle ABC
  AB : ℝ
  BC : ℝ
  AC : ℝ
  -- Lengths of segments formed by parallel lines
  ℓA_length : ℝ
  ℓB_length : ℝ
  ℓC_length : ℝ
  -- Conditions
  AB_positive : AB > 0
  BC_positive : BC > 0
  AC_positive : AC > 0
  ℓA_positive : ℓA_length > 0
  ℓB_positive : ℓB_length > 0
  ℓC_positive : ℓC_length > 0
  triangle_inequality : AB + BC > AC ∧ BC + AC > AB ∧ AC + AB > BC
  ℓA_inside : ℓA_length < BC
  ℓB_inside : ℓB_length < AC
  ℓC_inside : ℓC_length < AB

/-- The perimeter of the triangle formed by parallel lines -/
def innerTrianglePerimeter (t : TriangleWithParallelLines) : ℝ :=
  sorry

/-- Theorem stating that for the given triangle and parallel lines, the inner triangle perimeter is 330.75 -/
theorem inner_triangle_perimeter_is_330_75 
  (t : TriangleWithParallelLines) 
  (h1 : t.AB = 150) 
  (h2 : t.BC = 270) 
  (h3 : t.AC = 210) 
  (h4 : t.ℓA_length = 65) 
  (h5 : t.ℓB_length = 60) 
  (h6 : t.ℓC_length = 20) : 
  innerTrianglePerimeter t = 330.75 := by
  sorry

end NUMINAMATH_CALUDE_inner_triangle_perimeter_is_330_75_l1440_144028


namespace NUMINAMATH_CALUDE_u_value_l1440_144071

theorem u_value : 
  let u : ℝ := 1 / (2 - Real.rpow 3 (1/3))
  u = 2 + Real.rpow 3 (1/3) := by
  sorry

end NUMINAMATH_CALUDE_u_value_l1440_144071


namespace NUMINAMATH_CALUDE_solve_linear_equation_l1440_144065

theorem solve_linear_equation :
  ∃! x : ℚ, 3 * x - 5 = 8 ∧ x = 13 / 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l1440_144065


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_geometric_sequence_l1440_144032

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem arithmetic_mean_of_geometric_sequence
  (a : ℕ → ℝ) (q : ℝ)
  (h_geom : geometric_sequence a q)
  (h_q : q = -2)
  (h_condition : a 3 * a 7 = 4 * a 4) :
  (a 8 + a 11) / 2 = -56 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_geometric_sequence_l1440_144032


namespace NUMINAMATH_CALUDE_quadratic_properties_l1440_144095

/-- A quadratic function with specific properties -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a ≠ 0
  hpos : ∃ x₁ x₂, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0
  hsym : -b / (2 * a) = 2
  hintercept : ∃ x, x > 0 ∧ a * x^2 + b * x + c = 0 ∧ |c| = x

/-- Theorem stating properties of the quadratic function -/
theorem quadratic_properties (f : QuadraticFunction) :
  f.c > -1 ∧ f.a * (-f.c)^2 + f.b * (-f.c) + f.c = 0 := by
  sorry


end NUMINAMATH_CALUDE_quadratic_properties_l1440_144095


namespace NUMINAMATH_CALUDE_fourth_root_equivalence_l1440_144045

theorem fourth_root_equivalence (x : ℝ) (hx : x > 0) : (x * x^(1/3))^(1/4) = x^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_equivalence_l1440_144045


namespace NUMINAMATH_CALUDE_not_necessary_nor_sufficient_condition_l1440_144067

theorem not_necessary_nor_sufficient_condition (x : ℝ) :
  ¬((-2 < x ∧ x < 1) → (|x| > 1)) ∧ ¬((|x| > 1) → (-2 < x ∧ x < 1)) := by
  sorry

end NUMINAMATH_CALUDE_not_necessary_nor_sufficient_condition_l1440_144067


namespace NUMINAMATH_CALUDE_alice_bushes_theorem_l1440_144054

/-- The number of bushes Alice needs to buy for her yard -/
def bushes_needed (sides : ℕ) (side_length : ℕ) (bush_length : ℕ) : ℕ :=
  (sides * side_length) / bush_length

theorem alice_bushes_theorem :
  bushes_needed 3 16 4 = 12 := by
  sorry

end NUMINAMATH_CALUDE_alice_bushes_theorem_l1440_144054


namespace NUMINAMATH_CALUDE_max_value_three_power_minus_nine_power_l1440_144096

theorem max_value_three_power_minus_nine_power (x : ℝ) :
  ∃ (max : ℝ), max = (1 : ℝ) / 4 ∧ ∀ y : ℝ, 3^y - 9^y ≤ max :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_three_power_minus_nine_power_l1440_144096


namespace NUMINAMATH_CALUDE_fraction_unchanged_l1440_144046

theorem fraction_unchanged (x y : ℝ) (h : y ≠ 2*x) : 
  (3*(3*x)) / (2*(3*x) - 3*y) = (3*x) / (2*x - y) :=
by sorry

end NUMINAMATH_CALUDE_fraction_unchanged_l1440_144046


namespace NUMINAMATH_CALUDE_tv_show_duration_l1440_144075

theorem tv_show_duration (seasons_15 seasons_20 seasons_12 : ℕ)
  (episodes_15 episodes_20 episodes_12 : ℕ)
  (avg_episodes_per_year : ℕ) :
  seasons_15 = 8 →
  seasons_20 = 4 →
  seasons_12 = 2 →
  episodes_15 = 15 →
  episodes_20 = 20 →
  episodes_12 = 12 →
  avg_episodes_per_year = 16 →
  (seasons_15 * episodes_15 + seasons_20 * episodes_20 + seasons_12 * episodes_12) /
    avg_episodes_per_year = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_tv_show_duration_l1440_144075


namespace NUMINAMATH_CALUDE_A_sufficient_not_necessary_for_B_l1440_144013

/-- Proposition A: 0 < x < 5 -/
def prop_A (x : ℝ) : Prop := 0 < x ∧ x < 5

/-- Proposition B: |x - 2| < 3 -/
def prop_B (x : ℝ) : Prop := |x - 2| < 3

theorem A_sufficient_not_necessary_for_B :
  (∀ x : ℝ, prop_A x → prop_B x) ∧
  (∃ x : ℝ, prop_B x ∧ ¬prop_A x) := by sorry

end NUMINAMATH_CALUDE_A_sufficient_not_necessary_for_B_l1440_144013


namespace NUMINAMATH_CALUDE_min_value_abs_sum_l1440_144091

theorem min_value_abs_sum (x : ℚ) : 
  |x - 1| + |x + 3| ≥ 4 ∧ ∃ y : ℚ, |y - 1| + |y + 3| = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_abs_sum_l1440_144091


namespace NUMINAMATH_CALUDE_two_tetrahedra_in_cube_l1440_144030

/-- A cube with edge length a -/
structure Cube (a : ℝ) where
  edge_length : a > 0

/-- A regular tetrahedron with edge length a -/
structure RegularTetrahedron (a : ℝ) where
  edge_length : a > 0

/-- Represents the placement of a tetrahedron within a cube -/
def TetrahedronPlacement (a : ℝ) := Cube a → RegularTetrahedron a → Prop

/-- Two tetrahedra do not overlap -/
def NonOverlapping (a : ℝ) (t1 t2 : RegularTetrahedron a) : Prop := sorry

/-- Theorem stating that two non-overlapping regular tetrahedra can be inscribed in a cube -/
theorem two_tetrahedra_in_cube (a : ℝ) (h : a > 0) :
  ∃ (c : Cube a) (t1 t2 : RegularTetrahedron a) (p1 p2 : TetrahedronPlacement a),
    p1 c t1 ∧ p2 c t2 ∧ NonOverlapping a t1 t2 :=
  sorry

end NUMINAMATH_CALUDE_two_tetrahedra_in_cube_l1440_144030


namespace NUMINAMATH_CALUDE_wall_width_l1440_144094

/-- Given a rectangular wall with specific proportions and volume, prove its width --/
theorem wall_width (w h l : ℝ) (h_height : h = 6 * w) (h_length : l = 7 * h) 
  (h_volume : w * h * l = 86436) : w = 7 := by
  sorry

end NUMINAMATH_CALUDE_wall_width_l1440_144094


namespace NUMINAMATH_CALUDE_eggs_used_for_crepes_l1440_144031

theorem eggs_used_for_crepes 
  (total_eggs : ℕ) 
  (eggs_left : ℕ) 
  (h1 : total_eggs = 3 * 12)
  (h2 : eggs_left = 9)
  (h3 : ∃ remaining_after_crepes : ℕ, 
    remaining_after_crepes ≤ total_eggs ∧ 
    eggs_left = remaining_after_crepes - (2 * remaining_after_crepes / 3)) :
  (total_eggs - (total_eggs - eggs_left * 3)) / total_eggs = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_eggs_used_for_crepes_l1440_144031


namespace NUMINAMATH_CALUDE_caroline_lassis_l1440_144021

/-- The number of lassis Caroline can make from a given number of mangoes -/
def lassis_from_mangoes (mangoes : ℕ) : ℕ :=
  8 * mangoes / 3

/-- Theorem stating that Caroline can make 40 lassis from 15 mangoes -/
theorem caroline_lassis : lassis_from_mangoes 15 = 40 := by
  sorry

end NUMINAMATH_CALUDE_caroline_lassis_l1440_144021


namespace NUMINAMATH_CALUDE_triangle_problem_l1440_144040

/-- Given a triangle ABC with sides a and b that are roots of x^2 - 2√3x + 2 = 0,
    and 2cos(A+B) = 1, prove that angle C is 120° and side AB has length √10 -/
theorem triangle_problem (a b : ℝ) (A B C : ℝ) :
  a^2 - 2 * Real.sqrt 3 * a + 2 = 0 →
  b^2 - 2 * Real.sqrt 3 * b + 2 = 0 →
  2 * Real.cos (A + B) = 1 →
  C = 2 * π / 3 ∧
  (a^2 + b^2 - 2 * a * b * Real.cos C) = 10 :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l1440_144040


namespace NUMINAMATH_CALUDE_henley_candy_problem_l1440_144039

theorem henley_candy_problem :
  ∀ (total_candies : ℕ),
    (total_candies : ℚ) * (60 : ℚ) / 100 = 3 * 60 →
    total_candies = 300 :=
by
  sorry

end NUMINAMATH_CALUDE_henley_candy_problem_l1440_144039


namespace NUMINAMATH_CALUDE_rectangle_triangle_area_ratio_l1440_144004

/-- 
Given a rectangle with length L and width W, and a triangle with one side of the rectangle as its base 
and a vertex on the opposite side of the rectangle, the ratio of the area of the rectangle to the area 
of the triangle is 2:1.
-/
theorem rectangle_triangle_area_ratio 
  (L W : ℝ) 
  (hL : L > 0) 
  (hW : W > 0) : 
  (L * W) / ((1/2) * L * W) = 2 := by
sorry

end NUMINAMATH_CALUDE_rectangle_triangle_area_ratio_l1440_144004


namespace NUMINAMATH_CALUDE_symmetric_line_y_axis_correct_l1440_144060

/-- Given a line with equation ax + by + c = 0, return the equation of the line symmetric to it with respect to the y-axis -/
def symmetricLineYAxis (a b c : ℝ) : ℝ × ℝ × ℝ := (-a, b, c)

theorem symmetric_line_y_axis_correct :
  let original_line := (2, 1, -4)
  let symmetric_line := symmetricLineYAxis 2 1 (-4)
  symmetric_line = (-2, 1, -4) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_line_y_axis_correct_l1440_144060


namespace NUMINAMATH_CALUDE_right_triangle_inequality_l1440_144080

theorem right_triangle_inequality (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a^2 + b^2 = c^2) : 
  3 < (c^3 - a^3 - b^3) / (c * (c - a) * (c - b)) ∧ 
  (c^3 - a^3 - b^3) / (c * (c - a) * (c - b)) < Real.sqrt 2 + 2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_inequality_l1440_144080


namespace NUMINAMATH_CALUDE_x_range_for_inequality_l1440_144038

theorem x_range_for_inequality (x : ℝ) :
  (∀ a : ℝ, a ∈ Set.Icc 0 2 → a * x^2 + (a + 1) * x + 1 - (3/2) * a < 0) →
  x ∈ Set.Ioo (-2) (-1) := by
  sorry

end NUMINAMATH_CALUDE_x_range_for_inequality_l1440_144038


namespace NUMINAMATH_CALUDE_cost_of_pancakes_l1440_144070

/-- The cost of pancakes given initial order, tax, payment, and change --/
theorem cost_of_pancakes 
  (eggs_cost : ℕ)
  (cocoa_cost : ℕ)
  (cocoa_quantity : ℕ)
  (tax : ℕ)
  (payment : ℕ)
  (change : ℕ)
  (h1 : eggs_cost = 3)
  (h2 : cocoa_cost = 2)
  (h3 : cocoa_quantity = 2)
  (h4 : tax = 1)
  (h5 : payment = 15)
  (h6 : change = 1)
  : ℕ := by
  sorry

#check cost_of_pancakes

end NUMINAMATH_CALUDE_cost_of_pancakes_l1440_144070


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1440_144055

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 + 4*x - 5 < 0} = Set.Ioo (-5) 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1440_144055


namespace NUMINAMATH_CALUDE_square_on_circle_radius_l1440_144027

theorem square_on_circle_radius (S : ℝ) (x : ℝ) (R : ℝ) : 
  S = 256 → -- Square area is 256 cm²
  x^2 = S → -- Side length of the square
  (x - R)^2 = R^2 - (x/2)^2 → -- Pythagoras theorem application
  R = 10 := by
  sorry

end NUMINAMATH_CALUDE_square_on_circle_radius_l1440_144027


namespace NUMINAMATH_CALUDE_fifth_sphere_radius_l1440_144089

/-- Represents a cone with height and base radius 7 -/
structure Cone :=
  (height : ℝ := 7)
  (base_radius : ℝ := 7)

/-- Represents a sphere with a center and radius -/
structure Sphere :=
  (center : ℝ × ℝ × ℝ)
  (radius : ℝ)

/-- Represents the configuration of spheres in the cone -/
structure SphereConfiguration :=
  (cone : Cone)
  (base_spheres : Fin 4 → Sphere)
  (top_sphere : Sphere)

/-- Checks if two spheres are externally touching -/
def externally_touching (s1 s2 : Sphere) : Prop :=
  let (x1, y1, z1) := s1.center
  let (x2, y2, z2) := s2.center
  (x1 - x2)^2 + (y1 - y2)^2 + (z1 - z2)^2 = (s1.radius + s2.radius)^2

/-- Checks if a sphere touches the lateral surface of the cone -/
def touches_lateral_surface (s : Sphere) (c : Cone) : Prop :=
  sorry -- Definition omitted for brevity

/-- Checks if a sphere touches the base of the cone -/
def touches_base (s : Sphere) (c : Cone) : Prop :=
  sorry -- Definition omitted for brevity

/-- Theorem stating the radius of the fifth sphere -/
theorem fifth_sphere_radius (config : SphereConfiguration) :
  (∀ i j : Fin 4, i ≠ j → externally_touching (config.base_spheres i) (config.base_spheres j)) →
  (∀ i : Fin 4, touches_lateral_surface (config.base_spheres i) config.cone) →
  (∀ i : Fin 4, touches_base (config.base_spheres i) config.cone) →
  (∀ i : Fin 4, externally_touching (config.base_spheres i) config.top_sphere) →
  touches_lateral_surface config.top_sphere config.cone →
  config.top_sphere.radius = 2 * Real.sqrt 2 - 1 :=
by sorry

end NUMINAMATH_CALUDE_fifth_sphere_radius_l1440_144089


namespace NUMINAMATH_CALUDE_real_solution_exists_l1440_144006

theorem real_solution_exists (x : ℝ) : ∃ y : ℝ, 9 * y^2 + 3 * x * y + x - 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_real_solution_exists_l1440_144006


namespace NUMINAMATH_CALUDE_tree_distance_l1440_144047

/-- The distance between consecutive trees in a yard with an obstacle -/
theorem tree_distance (yard_length : ℝ) (num_trees : ℕ) (obstacle_gap : ℝ) :
  yard_length = 600 →
  num_trees = 36 →
  obstacle_gap = 10 →
  (yard_length - obstacle_gap) / (num_trees - 1 : ℝ) = 590 / 35 := by
  sorry

end NUMINAMATH_CALUDE_tree_distance_l1440_144047


namespace NUMINAMATH_CALUDE_sqrt_inequality_l1440_144064

theorem sqrt_inequality (x : ℝ) (h : x ≥ 4) :
  Real.sqrt (x - 3) + Real.sqrt (x - 2) > Real.sqrt (x - 4) + Real.sqrt (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l1440_144064


namespace NUMINAMATH_CALUDE_fifth_monday_in_leap_year_l1440_144077

/-- Represents a date in February of a leap year -/
structure FebruaryDate :=
  (day : ℕ)
  (is_leap_year : Bool)

/-- Represents a day of the week -/
inductive Weekday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Returns the weekday of a given February date -/
def weekday_of_date (d : FebruaryDate) : Weekday :=
  sorry

/-- Returns the number of Mondays up to and including a given date in February -/
def mondays_up_to (d : FebruaryDate) : ℕ :=
  sorry

/-- Theorem: In a leap year where February 7 is a Tuesday, 
    the fifth Monday in February falls on February 27 -/
theorem fifth_monday_in_leap_year :
  let feb7 : FebruaryDate := ⟨7, true⟩
  let feb27 : FebruaryDate := ⟨27, true⟩
  weekday_of_date feb7 = Weekday.Tuesday →
  mondays_up_to feb27 = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_fifth_monday_in_leap_year_l1440_144077


namespace NUMINAMATH_CALUDE_tetrahedron_angle_difference_l1440_144062

open Real

/-- Represents a tetrahedron -/
structure Tetrahedron where
  /-- The sum of all dihedral angles in the tetrahedron -/
  dihedral_sum : ℝ
  /-- The sum of all trihedral angles in the tetrahedron -/
  trihedral_sum : ℝ

/-- 
Theorem: For any tetrahedron, the difference between the sum of its dihedral angles 
and the sum of its trihedral angles is equal to 4π.
-/
theorem tetrahedron_angle_difference (t : Tetrahedron) : 
  t.dihedral_sum - t.trihedral_sum = 4 * π :=
sorry

end NUMINAMATH_CALUDE_tetrahedron_angle_difference_l1440_144062


namespace NUMINAMATH_CALUDE_total_cost_price_is_60_2_l1440_144044

/-- Calculates the cost price given the selling price and loss ratio -/
def costPrice (sellingPrice : ℚ) (lossRatio : ℚ) : ℚ :=
  sellingPrice / (1 - lossRatio)

/-- The total cost price of an apple, an orange, and a banana -/
def totalCostPrice : ℚ :=
  costPrice 16 (1/6) + costPrice 20 (1/5) + costPrice 12 (1/4)

theorem total_cost_price_is_60_2 :
  totalCostPrice = 60.2 := by sorry

end NUMINAMATH_CALUDE_total_cost_price_is_60_2_l1440_144044


namespace NUMINAMATH_CALUDE_grade11_sample_count_l1440_144011

/-- Represents a systematic sampling of students. -/
structure SystematicSampling where
  totalStudents : ℕ
  sampleSize : ℕ
  grade10Students : ℕ
  grade11Students : ℕ
  grade12Students : ℕ
  firstDrawn : ℕ

/-- Calculates the number of grade 11 students in the sample. -/
def grade11InSample (s : SystematicSampling) : ℕ :=
  let interval := s.totalStudents / s.sampleSize
  let grade11Start := s.grade10Students + 1
  let grade11End := grade11Start + s.grade11Students - 1
  let firstSampleInGrade11 := (((grade11Start - 1) / interval) * interval + s.firstDrawn - 1) / interval + 1
  let lastSampleInGrade11 := (((grade11End - 1) / interval) * interval + s.firstDrawn - 1) / interval + 1
  lastSampleInGrade11 - firstSampleInGrade11 + 1

/-- Theorem stating that for the given conditions, the number of grade 11 students in the sample is 17. -/
theorem grade11_sample_count (s : SystematicSampling) 
    (h1 : s.totalStudents = 1470)
    (h2 : s.sampleSize = 49)
    (h3 : s.grade10Students = 495)
    (h4 : s.grade11Students = 493)
    (h5 : s.grade12Students = 482)
    (h6 : s.firstDrawn = 23) :
    grade11InSample s = 17 := by
  sorry

end NUMINAMATH_CALUDE_grade11_sample_count_l1440_144011


namespace NUMINAMATH_CALUDE_max_quarters_sasha_l1440_144019

/-- Represents the value of a coin in cents -/
def coin_value (coin_type : String) : ℕ :=
  match coin_type with
  | "quarter" => 25
  | "dime" => 10
  | "nickel" => 5
  | _ => 0

/-- The total amount Sasha has in cents -/
def total_amount : ℕ := 480

/-- Theorem stating the maximum number of quarters Sasha can have -/
theorem max_quarters_sasha : 
  ∀ (quarters nickels dimes : ℕ),
  quarters = nickels →
  dimes = 4 * nickels →
  quarters * coin_value "quarter" + 
  nickels * coin_value "nickel" + 
  dimes * coin_value "dime" ≤ total_amount →
  quarters ≤ 6 := by
sorry

end NUMINAMATH_CALUDE_max_quarters_sasha_l1440_144019


namespace NUMINAMATH_CALUDE_candy_packing_problem_l1440_144014

theorem candy_packing_problem :
  ∃! (s : Finset ℕ),
    (∀ a ∈ s, 200 ≤ a ∧ a ≤ 250) ∧
    (∀ a ∈ s, a % 10 = 6) ∧
    (∀ a ∈ s, a % 15 = 11) ∧
    s.card = 2 :=
by sorry

end NUMINAMATH_CALUDE_candy_packing_problem_l1440_144014


namespace NUMINAMATH_CALUDE_a_must_be_negative_l1440_144066

theorem a_must_be_negative (a b c d e : ℝ) 
  (h1 : a / b < -(c / d))
  (h2 : b > 0)
  (h3 : d > 0)
  (h4 : e > 0)
  (h5 : a + e > 0) :
  a < 0 := by
  sorry

end NUMINAMATH_CALUDE_a_must_be_negative_l1440_144066


namespace NUMINAMATH_CALUDE_diagonal_intersection_probability_is_six_thirteenths_l1440_144037

/-- A regular nonagon is a 9-sided regular polygon -/
structure RegularNonagon where
  -- We don't need to define the structure explicitly for this problem

/-- The number of diagonals in a regular nonagon -/
def num_diagonals (n : RegularNonagon) : ℕ := 27

/-- The total number of pairs of diagonals in a regular nonagon -/
def total_diagonal_pairs (n : RegularNonagon) : ℕ := 351

/-- The number of pairs of intersecting diagonals in a regular nonagon -/
def intersecting_diagonal_pairs (n : RegularNonagon) : ℕ := 126

/-- The probability that two randomly chosen diagonals in a regular nonagon intersect -/
def diagonal_intersection_probability (n : RegularNonagon) : ℚ :=
  intersecting_diagonal_pairs n / total_diagonal_pairs n

theorem diagonal_intersection_probability_is_six_thirteenths (n : RegularNonagon) :
  diagonal_intersection_probability n = 6/13 := by
  sorry

end NUMINAMATH_CALUDE_diagonal_intersection_probability_is_six_thirteenths_l1440_144037


namespace NUMINAMATH_CALUDE_coprime_divisibility_implies_one_l1440_144079

theorem coprime_divisibility_implies_one (a b c : ℕ+) :
  Nat.Coprime a.val b.val →
  Nat.Coprime a.val c.val →
  Nat.Coprime b.val c.val →
  a.val^2 ∣ (b.val^3 + c.val^3) →
  b.val^2 ∣ (a.val^3 + c.val^3) →
  c.val^2 ∣ (a.val^3 + b.val^3) →
  a = 1 ∧ b = 1 ∧ c = 1 := by
sorry

end NUMINAMATH_CALUDE_coprime_divisibility_implies_one_l1440_144079


namespace NUMINAMATH_CALUDE_additional_money_needed_per_twin_l1440_144012

def initial_amount : ℝ := 50
def toilet_paper_cost : ℝ := 12
def groceries_cost : ℝ := 2 * toilet_paper_cost
def remaining_after_groceries : ℝ := initial_amount - toilet_paper_cost - groceries_cost
def boot_cost : ℝ := 3 * remaining_after_groceries
def total_boot_cost : ℝ := 2 * boot_cost

theorem additional_money_needed_per_twin : 
  (total_boot_cost - remaining_after_groceries) / 2 = 35 := by sorry

end NUMINAMATH_CALUDE_additional_money_needed_per_twin_l1440_144012


namespace NUMINAMATH_CALUDE_power_of_81_five_sixths_l1440_144088

theorem power_of_81_five_sixths :
  (81 : ℝ) ^ (5/6) = 27 * (3 : ℝ) ^ (1/3) := by sorry

end NUMINAMATH_CALUDE_power_of_81_five_sixths_l1440_144088


namespace NUMINAMATH_CALUDE_equation_solution_l1440_144034

theorem equation_solution : ∃ x : ℝ, (144 / 0.144 = x / 0.0144) ∧ (x = 14.4) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1440_144034


namespace NUMINAMATH_CALUDE_min_strikes_to_defeat_dragon_l1440_144084

/-- Represents the state of the dragon -/
structure DragonState where
  heads : Nat
  tails : Nat

/-- Represents a strike against the dragon -/
inductive Strike
  | CutOneHead
  | CutOneTail
  | CutTwoHeads
  | CutTwoTails

/-- Applies a strike to the dragon state -/
def applyStrike (state : DragonState) (strike : Strike) : DragonState :=
  match strike with
  | Strike.CutOneHead => ⟨state.heads, state.tails⟩
  | Strike.CutOneTail => ⟨state.heads, state.tails + 1⟩
  | Strike.CutTwoHeads => ⟨state.heads - 2, state.tails⟩
  | Strike.CutTwoTails => ⟨state.heads + 1, state.tails - 2⟩

/-- Checks if the dragon is defeated (no heads and tails) -/
def isDragonDefeated (state : DragonState) : Prop :=
  state.heads = 0 ∧ state.tails = 0

/-- Theorem: The minimum number of strikes to defeat the dragon is 9 -/
theorem min_strikes_to_defeat_dragon :
  ∃ (strikes : List Strike),
    strikes.length = 9 ∧
    isDragonDefeated (strikes.foldl applyStrike ⟨3, 3⟩) ∧
    ∀ (otherStrikes : List Strike),
      otherStrikes.length < 9 →
      ¬isDragonDefeated (otherStrikes.foldl applyStrike ⟨3, 3⟩) :=
by
  sorry

end NUMINAMATH_CALUDE_min_strikes_to_defeat_dragon_l1440_144084


namespace NUMINAMATH_CALUDE_function_inequality_l1440_144005

noncomputable def f (x : ℝ) : ℝ := (Real.exp 2 * x^2 + 1) / x

noncomputable def g (x : ℝ) : ℝ := (Real.exp 2 * x^2) / Real.exp x

theorem function_inequality (k : ℝ) (hk : k > 0) :
  (∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → g x₁ / k ≤ f x₂ / (k + 1)) →
  k ≥ 4 / (2 * Real.exp 1 - 4) :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_l1440_144005


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l1440_144072

/-- Given an arithmetic sequence with common ratio q ≠ 0, where S_n is the sum of first n terms,
    if S_3, S_9, and S_6 form an arithmetic sequence, then q^3 = 3/2 -/
theorem arithmetic_sequence_ratio (q : ℝ) (a₁ : ℝ) (S : ℕ → ℝ) : 
  q ≠ 0 ∧ 
  (∀ n, S n = a₁ * (1 - q^n) / (1 - q)) ∧ 
  (2 * S 9 = S 3 + S 6) →
  q^3 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l1440_144072


namespace NUMINAMATH_CALUDE_copperfield_numbers_l1440_144093

theorem copperfield_numbers : ∃ (x₁ x₂ x₃ : ℕ), 
  x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
  (∃ (k₁ k₂ k₃ : ℕ+), 
    x₁ * (3 ^ k₁.val) = x₁ + 2500 * k₁.val ∧
    x₂ * (3 ^ k₂.val) = x₂ + 2500 * k₂.val ∧
    x₃ * (3 ^ k₃.val) = x₃ + 2500 * k₃.val) :=
by sorry

end NUMINAMATH_CALUDE_copperfield_numbers_l1440_144093


namespace NUMINAMATH_CALUDE_expression_values_l1440_144000

theorem expression_values (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  let expr := a / abs a + b / abs b + c / abs c + (a * b * c) / abs (a * b * c)
  expr = -4 ∨ expr = 0 ∨ expr = 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_values_l1440_144000


namespace NUMINAMATH_CALUDE_base_7_65234_equals_16244_l1440_144056

def base_7_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

theorem base_7_65234_equals_16244 :
  base_7_to_10 [4, 3, 2, 5, 6] = 16244 := by
  sorry

end NUMINAMATH_CALUDE_base_7_65234_equals_16244_l1440_144056


namespace NUMINAMATH_CALUDE_birds_count_l1440_144073

/-- The number of fish-eater birds Cohen saw over three days -/
def total_birds (initial : ℕ) : ℕ :=
  let day1 := initial
  let day2 := 2 * day1
  let day3 := day2 - 200
  day1 + day2 + day3

/-- Theorem stating that the total number of birds seen over three days is 1300 -/
theorem birds_count : total_birds 300 = 1300 := by
  sorry

end NUMINAMATH_CALUDE_birds_count_l1440_144073


namespace NUMINAMATH_CALUDE_abraham_shower_gels_l1440_144022

def shower_gel_problem (budget : ℕ) (shower_gel_cost : ℕ) (toothpaste_cost : ℕ) (detergent_cost : ℕ) (remaining : ℕ) : Prop :=
  let total_spent : ℕ := budget - remaining
  let non_gel_cost : ℕ := toothpaste_cost + detergent_cost
  let gel_cost : ℕ := total_spent - non_gel_cost
  gel_cost / shower_gel_cost = 4

theorem abraham_shower_gels :
  shower_gel_problem 60 4 3 11 30 := by
  sorry

end NUMINAMATH_CALUDE_abraham_shower_gels_l1440_144022


namespace NUMINAMATH_CALUDE_approximate_root_l1440_144098

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem approximate_root (hcont : Continuous f) 
  (h1 : f 0.64 < 0) (h2 : f 0.72 > 0) (h3 : f 0.68 < 0) :
  ∃ (x : ℝ), f x = 0 ∧ |x - 0.7| ≤ 0.1 := by
  sorry

end NUMINAMATH_CALUDE_approximate_root_l1440_144098


namespace NUMINAMATH_CALUDE_projection_of_a_onto_b_l1440_144010

def a : Fin 2 → ℚ := ![1, 2]
def b : Fin 2 → ℚ := ![-2, 4]

def dot_product (v w : Fin 2 → ℚ) : ℚ :=
  (v 0) * (w 0) + (v 1) * (w 1)

def magnitude_squared (v : Fin 2 → ℚ) : ℚ :=
  dot_product v v

def scalar_mult (c : ℚ) (v : Fin 2 → ℚ) : Fin 2 → ℚ :=
  fun i => c * (v i)

def projection (v w : Fin 2 → ℚ) : Fin 2 → ℚ :=
  scalar_mult ((dot_product v w) / (magnitude_squared w)) w

theorem projection_of_a_onto_b :
  projection a b = ![-(3/5), 6/5] := by
  sorry

end NUMINAMATH_CALUDE_projection_of_a_onto_b_l1440_144010


namespace NUMINAMATH_CALUDE_square_side_length_equal_perimeter_l1440_144007

theorem square_side_length_equal_perimeter (rectangle_length rectangle_width : ℝ) 
  (h1 : rectangle_length = 10)
  (h2 : rectangle_width = 8) : 
  let rectangle_perimeter := 2 * (rectangle_length + rectangle_width)
  let square_side_length := rectangle_perimeter / 4
  square_side_length = 9 := by
sorry

end NUMINAMATH_CALUDE_square_side_length_equal_perimeter_l1440_144007


namespace NUMINAMATH_CALUDE_merchant_profit_percentage_l1440_144020

/-- Calculates the profit percentage for a merchant who marks up goods by 50%
    and then offers a 10% discount on the marked price. -/
theorem merchant_profit_percentage 
  (cost_price : ℝ) 
  (markup_percentage : ℝ) 
  (discount_percentage : ℝ) 
  (hp_markup : markup_percentage = 50) 
  (hp_discount : discount_percentage = 10) : 
  let marked_price := cost_price * (1 + markup_percentage / 100)
  let discounted_price := marked_price * (1 - discount_percentage / 100)
  let profit := discounted_price - cost_price
  let profit_percentage := (profit / cost_price) * 100
  profit_percentage = 35 := by
sorry

end NUMINAMATH_CALUDE_merchant_profit_percentage_l1440_144020


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1440_144036

theorem quadratic_equation_solution :
  let a : ℝ := 1
  let b : ℝ := 3
  let c : ℝ := -1
  let x₁ : ℝ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ : ℝ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  x₁^2 + 3*x₁ - 1 = 0 ∧ x₂^2 + 3*x₂ - 1 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1440_144036


namespace NUMINAMATH_CALUDE_x_value_theorem_l1440_144049

theorem x_value_theorem (x n : ℕ) : 
  x = 2^n - 32 →
  (∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ p ≠ 3 ∧ q ≠ 3 ∧
    x = 3 * p * q) →
  x = 480 ∨ x = 2016 := by
sorry

end NUMINAMATH_CALUDE_x_value_theorem_l1440_144049


namespace NUMINAMATH_CALUDE_bicycle_journey_initial_time_l1440_144033

theorem bicycle_journey_initial_time 
  (speed : ℝ) 
  (additional_distance : ℝ) 
  (rest_time : ℝ) 
  (final_distance : ℝ) 
  (total_time : ℝ) :
  speed = 10 →
  additional_distance = 15 →
  rest_time = 30 →
  final_distance = 20 →
  total_time = 270 →
  ∃ (initial_time : ℝ), 
    initial_time * 60 + additional_distance / speed * 60 + rest_time + final_distance / speed * 60 = total_time ∧ 
    initial_time * 60 = 30 :=
by sorry

end NUMINAMATH_CALUDE_bicycle_journey_initial_time_l1440_144033
