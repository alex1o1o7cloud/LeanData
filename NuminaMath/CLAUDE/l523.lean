import Mathlib

namespace NUMINAMATH_CALUDE_equal_savings_time_l523_52382

/-- Proves that Jim and Sara will have saved the same amount after 820 weeks -/
theorem equal_savings_time (sara_initial : ℕ) (sara_weekly : ℕ) (jim_initial : ℕ) (jim_weekly : ℕ)
  (h1 : sara_initial = 4100)
  (h2 : sara_weekly = 10)
  (h3 : jim_initial = 0)
  (h4 : jim_weekly = 15) :
  ∃ w : ℕ, w = 820 ∧ sara_initial + w * sara_weekly = jim_initial + w * jim_weekly :=
by
  sorry

end NUMINAMATH_CALUDE_equal_savings_time_l523_52382


namespace NUMINAMATH_CALUDE_last_two_digits_sum_of_squares_l523_52327

theorem last_two_digits_sum_of_squares :
  ∀ (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℕ),
  a₁ % 100 = 11 →
  a₂ % 100 = 12 →
  a₃ % 100 = 13 →
  a₄ % 100 = 14 →
  a₅ % 100 = 15 →
  a₆ % 100 = 16 →
  a₇ % 100 = 17 →
  a₈ % 100 = 18 →
  a₉ % 100 = 19 →
  (a₁^2 + a₂^2 + a₃^2 + a₄^2 + a₅^2 + a₆^2 + a₇^2 + a₈^2 + a₉^2) % 100 = 85 :=
by sorry

end NUMINAMATH_CALUDE_last_two_digits_sum_of_squares_l523_52327


namespace NUMINAMATH_CALUDE_floor_of_sum_l523_52323

theorem floor_of_sum (x : ℝ) (h : x = -3.7 + 1.5) : ⌊x⌋ = -3 := by
  sorry

end NUMINAMATH_CALUDE_floor_of_sum_l523_52323


namespace NUMINAMATH_CALUDE_janice_working_days_l523_52387

-- Define the problem parameters
def regular_pay : ℕ := 30
def overtime_pay : ℕ := 15
def overtime_shifts : ℕ := 3
def total_earnings : ℕ := 195

-- Define the function to calculate the number of working days
def calculate_working_days (regular_pay overtime_pay overtime_shifts total_earnings : ℕ) : ℕ :=
  (total_earnings - overtime_pay * overtime_shifts) / regular_pay

-- Theorem statement
theorem janice_working_days :
  calculate_working_days regular_pay overtime_pay overtime_shifts total_earnings = 5 := by
  sorry


end NUMINAMATH_CALUDE_janice_working_days_l523_52387


namespace NUMINAMATH_CALUDE_roots_sum_less_than_two_l523_52361

-- Define the function f(x) = x³ - 3x
def f (x : ℝ) := x^3 - 3*x

-- Define the theorem
theorem roots_sum_less_than_two (x₁ x₂ m : ℝ) 
  (h1 : 0 < x₁) (h2 : x₁ < x₂) 
  (h3 : f x₁ = m) (h4 : f x₂ = m) : 
  x₁ + x₂ < 2 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_less_than_two_l523_52361


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l523_52385

/-- The speed of a boat in still water, given its travel times with varying current and wind conditions. -/
theorem boat_speed_in_still_water 
  (downstream_time : ℝ) 
  (upstream_time : ℝ) 
  (current_start : ℝ) 
  (current_end : ℝ) 
  (wind_slowdown : ℝ) 
  (h1 : downstream_time = 3)
  (h2 : upstream_time = 4.5)
  (h3 : current_start = 2)
  (h4 : current_end = 4)
  (h5 : wind_slowdown = 1) :
  ∃ (boat_speed : ℝ), boat_speed = 16 ∧ 
  (boat_speed + (current_start + current_end) / 2 - wind_slowdown) * downstream_time = 
  (boat_speed - (current_start + current_end) / 2 - wind_slowdown) * upstream_time :=
by sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l523_52385


namespace NUMINAMATH_CALUDE_sin_addition_equality_l523_52334

theorem sin_addition_equality (y : Real) : 
  (y ∈ Set.Icc 0 (Real.pi / 2)) → 
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), Real.sin (x + y) = Real.sin x + Real.sin y) → 
  y = 0 :=
by sorry

end NUMINAMATH_CALUDE_sin_addition_equality_l523_52334


namespace NUMINAMATH_CALUDE_smallest_perfect_square_sum_l523_52335

/-- The sum of 20 consecutive positive integers starting from n -/
def sum_20_consecutive (n : ℕ) : ℕ := 10 * (2 * n + 19)

/-- Predicate to check if a number is a perfect square -/
def is_perfect_square (m : ℕ) : Prop := ∃ k : ℕ, m = k * k

theorem smallest_perfect_square_sum :
  (∃ n : ℕ, sum_20_consecutive n = 250) ∧
  (∀ m : ℕ, m < 250 → ¬∃ n : ℕ, sum_20_consecutive n = m ∧ is_perfect_square m) :=
sorry

end NUMINAMATH_CALUDE_smallest_perfect_square_sum_l523_52335


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l523_52304

theorem geometric_sequence_sum : 
  let a : ℚ := 1/3  -- first term
  let r : ℚ := 1/3  -- common ratio
  let n : ℕ := 5    -- number of terms
  let S : ℚ := (a * (1 - r^n)) / (1 - r)  -- sum formula
  S = 121/243 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l523_52304


namespace NUMINAMATH_CALUDE_tan_x_is_zero_l523_52318

theorem tan_x_is_zero (x : Real) 
  (h1 : 0 ≤ x ∧ x ≤ π) 
  (h2 : 3 * Real.sin (x / 2) = Real.sqrt (1 + Real.sin x) - Real.sqrt (1 - Real.sin x)) : 
  Real.tan x = 0 := by
  sorry

end NUMINAMATH_CALUDE_tan_x_is_zero_l523_52318


namespace NUMINAMATH_CALUDE_revenue_is_90_dollars_l523_52360

def total_bags : ℕ := 10
def oranges_per_bag : ℕ := 30
def bags_with_10_percent_rotten : ℕ := 4
def bags_with_20_percent_rotten : ℕ := 3
def bags_with_5_percent_rotten : ℕ := 3
def oranges_for_juice : ℕ := 70
def oranges_for_jams : ℕ := 15
def selling_price_per_orange : ℚ := 0.50

def total_oranges : ℕ := total_bags * oranges_per_bag

def rotten_oranges : ℕ := 
  bags_with_10_percent_rotten * oranges_per_bag / 10 +
  bags_with_20_percent_rotten * oranges_per_bag / 5 +
  bags_with_5_percent_rotten * oranges_per_bag / 20

def good_oranges : ℕ := total_oranges - rotten_oranges

def oranges_for_sale : ℕ := good_oranges - oranges_for_juice - oranges_for_jams

def total_revenue : ℚ := oranges_for_sale * selling_price_per_orange

theorem revenue_is_90_dollars : total_revenue = 90 := by
  sorry

end NUMINAMATH_CALUDE_revenue_is_90_dollars_l523_52360


namespace NUMINAMATH_CALUDE_cow_problem_l523_52346

theorem cow_problem (purchase_price daily_food_cost additional_costs selling_price profit : ℕ) 
  (h1 : purchase_price = 600)
  (h2 : daily_food_cost = 20)
  (h3 : additional_costs = 500)
  (h4 : selling_price = 2500)
  (h5 : profit = 600) :
  ∃ days : ℕ, days = (selling_price - profit - purchase_price - additional_costs) / daily_food_cost ∧ days = 40 :=
sorry

end NUMINAMATH_CALUDE_cow_problem_l523_52346


namespace NUMINAMATH_CALUDE_projection_vector_l523_52309

def a : Fin 3 → ℝ := ![0, 1, 1]
def b : Fin 3 → ℝ := ![1, 1, 0]

theorem projection_vector :
  let proj_a_b := (a • b) / (a • a) • a
  proj_a_b = ![0, 1/2, 1/2] := by
sorry

end NUMINAMATH_CALUDE_projection_vector_l523_52309


namespace NUMINAMATH_CALUDE_tan_3_negative_l523_52376

theorem tan_3_negative : Real.tan 3 < 0 := by
  sorry

end NUMINAMATH_CALUDE_tan_3_negative_l523_52376


namespace NUMINAMATH_CALUDE_visual_range_increase_l523_52333

theorem visual_range_increase (original_range new_range : ℝ) 
  (h1 : original_range = 100)
  (h2 : new_range = 150) :
  (new_range - original_range) / original_range * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_visual_range_increase_l523_52333


namespace NUMINAMATH_CALUDE_horseback_trip_distance_l523_52373

/-- Calculates the total distance traveled during a horseback riding trip -/
def total_distance : ℝ :=
  let day1_segment1 := 5 * 7
  let day1_segment2 := 3 * 2
  let day2_segment1 := 6 * 6
  let day2_segment2 := 3 * 3
  let day3_segment1 := 4 * 3
  let day3_segment2 := 7 * 5
  day1_segment1 + day1_segment2 + day2_segment1 + day2_segment2 + day3_segment1 + day3_segment2

theorem horseback_trip_distance : total_distance = 133 := by
  sorry

end NUMINAMATH_CALUDE_horseback_trip_distance_l523_52373


namespace NUMINAMATH_CALUDE_money_division_l523_52303

/-- Proves that the total amount of money divided amongst a, b, and c is $400 --/
theorem money_division (a b c : ℝ) : 
  a = (2/3) * (b + c) →   -- a gets 2/3 as much as b and c together
  b = (6/9) * (a + c) →   -- b gets 6/9 as much as a and c together
  a = 160 →               -- The share of a is $160
  a + b + c = 400 := by   -- The total amount is $400
sorry


end NUMINAMATH_CALUDE_money_division_l523_52303


namespace NUMINAMATH_CALUDE_triangle_area_is_seven_l523_52364

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line by two points
structure Line2D where
  p1 : Point2D
  p2 : Point2D

-- Define the three lines
def line1 : Line2D := { p1 := { x := 0, y := 5 }, p2 := { x := 10, y := 2 } }
def line2 : Line2D := { p1 := { x := 2, y := 6 }, p2 := { x := 8, y := 1 } }
def line3 : Line2D := { p1 := { x := 0, y := 3 }, p2 := { x := 5, y := 0 } }

-- Function to calculate the area of a triangle formed by three lines
def triangleArea (l1 l2 l3 : Line2D) : ℝ :=
  sorry

-- Theorem stating that the area of the triangle is 7
theorem triangle_area_is_seven :
  triangleArea line1 line2 line3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_is_seven_l523_52364


namespace NUMINAMATH_CALUDE_clock_hands_straight_period_l523_52374

/-- Represents the number of times clock hands are straight in a given period -/
def straight_hands (period : ℝ) : ℕ := sorry

/-- Represents the number of times clock hands coincide in a given period -/
def coinciding_hands (period : ℝ) : ℕ := sorry

/-- Represents the number of times clock hands are opposite in a given period -/
def opposite_hands (period : ℝ) : ℕ := sorry

theorem clock_hands_straight_period :
  straight_hands 12 = 22 ∧
  (∀ period : ℝ, straight_hands period = coinciding_hands period + opposite_hands period) ∧
  coinciding_hands 12 = 11 ∧
  opposite_hands 12 = 11 :=
by sorry

end NUMINAMATH_CALUDE_clock_hands_straight_period_l523_52374


namespace NUMINAMATH_CALUDE_student_average_grade_l523_52314

theorem student_average_grade 
  (courses_last_year : ℕ)
  (courses_year_before : ℕ)
  (avg_grade_year_before : ℚ)
  (avg_grade_two_years : ℚ)
  (h1 : courses_last_year = 6)
  (h2 : courses_year_before = 5)
  (h3 : avg_grade_year_before = 50)
  (h4 : avg_grade_two_years = 77)
  : ∃ (avg_grade_last_year : ℚ), avg_grade_last_year = 99.5 := by
  sorry

end NUMINAMATH_CALUDE_student_average_grade_l523_52314


namespace NUMINAMATH_CALUDE_correct_calculation_l523_52365

theorem correct_calculation (x : ℤ) : 
  x + 238 = 637 → x - 382 = 17 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l523_52365


namespace NUMINAMATH_CALUDE_parabola_equation_l523_52379

/-- A parabola with vertex at the origin and axis along the y-axis passing through (30, -40) with focus at (0, -45/4) has the equation x^2 = -45/2 * y -/
theorem parabola_equation (p : ℝ × ℝ) (f : ℝ × ℝ) :
  p.1 = 30 ∧ p.2 = -40 ∧ f.1 = 0 ∧ f.2 = -45/4 →
  ∀ x y : ℝ, (x^2 = -45/2 * y ↔ (x - f.1)^2 + (y - f.2)^2 = (y - p.2)^2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_equation_l523_52379


namespace NUMINAMATH_CALUDE_function_from_derivative_and_point_l523_52325

/-- Given a function f: ℝ → ℝ, if its derivative is 4x³ for all x
and f(1) = -1, then f(x) = x⁴ - 2 for all x. -/
theorem function_from_derivative_and_point (f : ℝ → ℝ) 
    (h1 : ∀ x, deriv f x = 4 * x^3)
    (h2 : f 1 = -1) :
    ∀ x, f x = x^4 - 2 := by
  sorry

end NUMINAMATH_CALUDE_function_from_derivative_and_point_l523_52325


namespace NUMINAMATH_CALUDE_sum_of_largest_and_smallest_l523_52317

def digits : List Nat := [0, 1, 3, 5]

def is_valid_number (n : Nat) : Prop :=
  n ≥ 100 ∧ n ≤ 999 ∧ (∀ d, d ∈ digits → (n / 100 = d ∨ (n / 10) % 10 = d ∨ n % 10 = d))

def largest_number : Nat :=
  531

def smallest_number : Nat :=
  103

theorem sum_of_largest_and_smallest :
  is_valid_number largest_number ∧
  is_valid_number smallest_number ∧
  (∀ n, is_valid_number n → n ≤ largest_number) ∧
  (∀ n, is_valid_number n → n ≥ smallest_number) ∧
  largest_number + smallest_number = 634 :=
sorry

end NUMINAMATH_CALUDE_sum_of_largest_and_smallest_l523_52317


namespace NUMINAMATH_CALUDE_money_distribution_l523_52301

-- Define the variables
variable (A B C : ℕ)

-- State the theorem
theorem money_distribution (h1 : A + B + C = 500) (h2 : A + C = 200) (h3 : B + C = 320) : C = 20 := by
  sorry

end NUMINAMATH_CALUDE_money_distribution_l523_52301


namespace NUMINAMATH_CALUDE_point_on_h_graph_l523_52392

-- Define the function g
def g : ℝ → ℝ := sorry

-- Define the function h in terms of g
def h (x : ℝ) : ℝ := (g x)^3

-- State the theorem
theorem point_on_h_graph :
  ∃ (x y : ℝ), g 2 = -5 ∧ h x = y ∧ x + y = -123 := by sorry

end NUMINAMATH_CALUDE_point_on_h_graph_l523_52392


namespace NUMINAMATH_CALUDE_equal_triplet_solution_l523_52367

theorem equal_triplet_solution {a b c : ℝ} (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0)
  (h1 : a * (a + b) = b * (b + c)) (h2 : b * (b + c) = c * (c + a)) :
  a = b ∧ b = c := by
sorry

end NUMINAMATH_CALUDE_equal_triplet_solution_l523_52367


namespace NUMINAMATH_CALUDE_quadratic_opens_downward_l523_52353

def f (x : ℝ) := -x^2 + 3

theorem quadratic_opens_downward :
  ∃ (a : ℝ), ∀ (x : ℝ), x > a → f x < f a :=
sorry

end NUMINAMATH_CALUDE_quadratic_opens_downward_l523_52353


namespace NUMINAMATH_CALUDE_cubic_minus_linear_factorization_l523_52320

theorem cubic_minus_linear_factorization (a : ℝ) : a^3 - a = a * (a + 1) * (a - 1) := by
  sorry

end NUMINAMATH_CALUDE_cubic_minus_linear_factorization_l523_52320


namespace NUMINAMATH_CALUDE_square_area_from_adjacent_points_l523_52302

/-- The area of a square with adjacent vertices at (1,2) and (5,6) is 32 -/
theorem square_area_from_adjacent_points : 
  let p1 : ℝ × ℝ := (1, 2)
  let p2 : ℝ × ℝ := (5, 6)
  let side_length := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  let area := side_length^2
  area = 32 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_adjacent_points_l523_52302


namespace NUMINAMATH_CALUDE_janettes_remaining_jerky_l523_52352

def camping_days : ℕ := 5
def initial_jerky : ℕ := 40
def breakfast_jerky : ℕ := 1
def lunch_jerky : ℕ := 1
def dinner_jerky : ℕ := 2

def daily_consumption : ℕ := breakfast_jerky + lunch_jerky + dinner_jerky

def total_consumed : ℕ := daily_consumption * camping_days

def remaining_after_trip : ℕ := initial_jerky - total_consumed

def given_to_brother : ℕ := remaining_after_trip / 2

theorem janettes_remaining_jerky :
  initial_jerky - total_consumed - given_to_brother = 10 := by
  sorry

end NUMINAMATH_CALUDE_janettes_remaining_jerky_l523_52352


namespace NUMINAMATH_CALUDE_total_viewing_time_l523_52357

-- Define the viewing segments
def segment1 : ℕ := 35
def segment2 : ℕ := 45
def segment3 : ℕ := 20

-- Define the rewind times
def rewind1 : ℕ := 5
def rewind2 : ℕ := 15

-- Theorem to prove
theorem total_viewing_time :
  segment1 + segment2 + segment3 + rewind1 + rewind2 = 120 := by
  sorry

end NUMINAMATH_CALUDE_total_viewing_time_l523_52357


namespace NUMINAMATH_CALUDE_product_sum_relation_l523_52344

theorem product_sum_relation (a b x : ℤ) : 
  b = 9 → b - a = 5 → a * b = 2 * (a + b) + x → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_relation_l523_52344


namespace NUMINAMATH_CALUDE_cos_alpha_for_point_on_terminal_side_l523_52321

/-- Given a point P(-3, -4) on the terminal side of angle α, prove that cos α = -3/5 -/
theorem cos_alpha_for_point_on_terminal_side (α : Real) :
  let P : Prod Real Real := (-3, -4)
  ∃ (r : Real), r > 0 ∧ P = (r * Real.cos α, r * Real.sin α) →
  Real.cos α = -3/5 := by
sorry

end NUMINAMATH_CALUDE_cos_alpha_for_point_on_terminal_side_l523_52321


namespace NUMINAMATH_CALUDE_jack_classic_authors_l523_52370

/-- The number of books each classic author has in Jack's collection -/
def books_per_author : ℕ := 33

/-- The total number of books in Jack's classics section -/
def total_books : ℕ := 198

/-- The number of classic authors in Jack's collection -/
def num_authors : ℕ := total_books / books_per_author

theorem jack_classic_authors :
  num_authors = 6 :=
by sorry

end NUMINAMATH_CALUDE_jack_classic_authors_l523_52370


namespace NUMINAMATH_CALUDE_sum_in_base6_l523_52363

/-- Represents a number in base 6 -/
def Base6 : Type := List Nat

/-- Converts a base 6 number to its decimal representation -/
def to_decimal (n : Base6) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

/-- Converts a decimal number to its base 6 representation -/
def to_base6 (n : Nat) : Base6 :=
  sorry

/-- Adds two base 6 numbers -/
def add_base6 (a b : Base6) : Base6 :=
  to_base6 (to_decimal a + to_decimal b)

theorem sum_in_base6 (a b c : Base6) :
  a = [0, 5, 6] ∧ b = [5, 0, 1] ∧ c = [2] →
  add_base6 (add_base6 a b) c = [1, 1, 1, 1] :=
sorry

end NUMINAMATH_CALUDE_sum_in_base6_l523_52363


namespace NUMINAMATH_CALUDE_three_a_in_S_implies_a_in_S_l523_52383

theorem three_a_in_S_implies_a_in_S (a : ℤ) : 
  (∃ x y : ℤ, 3 * a = x^2 + 2 * y^2) → 
  (∃ u v : ℤ, a = u^2 + 2 * v^2) := by
sorry

end NUMINAMATH_CALUDE_three_a_in_S_implies_a_in_S_l523_52383


namespace NUMINAMATH_CALUDE_smallest_x_for_cube_1680x_l523_52356

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m^3

theorem smallest_x_for_cube_1680x : 
  (∀ x : ℕ, x > 0 ∧ x < 44100 → ¬ is_perfect_cube (1680 * x)) ∧ 
  is_perfect_cube (1680 * 44100) := by
sorry

end NUMINAMATH_CALUDE_smallest_x_for_cube_1680x_l523_52356


namespace NUMINAMATH_CALUDE_four_digit_sum_divisible_by_nine_l523_52339

theorem four_digit_sum_divisible_by_nine 
  (a b c d e f g h i j : Nat) 
  (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧ a ≠ j ∧
              b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧ b ≠ j ∧
              c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧ c ≠ j ∧
              d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧ d ≠ j ∧
              e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧ e ≠ j ∧
              f ≠ g ∧ f ≠ h ∧ f ≠ i ∧ f ≠ j ∧
              g ≠ h ∧ g ≠ i ∧ g ≠ j ∧
              h ≠ i ∧ h ≠ j ∧
              i ≠ j)
  (digits : a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧ f < 10 ∧ g < 10 ∧ h < 10 ∧ i < 10 ∧ j < 10)
  (sum_equality : 100 * a + 10 * b + c + 100 * d + 10 * e + f = 1000 * g + 100 * h + 10 * i + j) :
  (1000 * g + 100 * h + 10 * i + j) % 9 = 0 := by
  sorry


end NUMINAMATH_CALUDE_four_digit_sum_divisible_by_nine_l523_52339


namespace NUMINAMATH_CALUDE_line_parameterization_l523_52396

/-- Given a line y = (2/3)x + 5 parameterized as [x; y] = [-3; s] + t[l; -6],
    prove that s = 3 and l = -9 -/
theorem line_parameterization (s l : ℝ) : 
  (∀ x y t : ℝ, y = (2/3) * x + 5 ↔ 
    ∃ t, (x, y) = (-3 + t * l, s + t * (-6))) →
  s = 3 ∧ l = -9 := by
  sorry

end NUMINAMATH_CALUDE_line_parameterization_l523_52396


namespace NUMINAMATH_CALUDE_chocolate_bars_per_small_box_l523_52326

theorem chocolate_bars_per_small_box 
  (total_bars : ℕ) 
  (small_boxes : ℕ) 
  (h1 : total_bars = 525) 
  (h2 : small_boxes = 21) : 
  total_bars / small_boxes = 25 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bars_per_small_box_l523_52326


namespace NUMINAMATH_CALUDE_contrapositive_example_l523_52337

theorem contrapositive_example :
  (∀ x : ℝ, x > 1 → x^2 + x > 2) ↔ (∀ x : ℝ, x^2 + x ≤ 2 → x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_example_l523_52337


namespace NUMINAMATH_CALUDE_five_balls_three_boxes_count_l523_52393

/-- The number of ways to distribute n indistinguishable balls into k distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 5 indistinguishable balls into 3 distinguishable boxes -/
def five_balls_three_boxes : ℕ := distribute_balls 5 3

theorem five_balls_three_boxes_count : five_balls_three_boxes = 21 := by sorry

end NUMINAMATH_CALUDE_five_balls_three_boxes_count_l523_52393


namespace NUMINAMATH_CALUDE_e_power_necessary_not_sufficient_for_ln_l523_52308

theorem e_power_necessary_not_sufficient_for_ln (x : ℝ) :
  (∃ y, (Real.exp y > 1 ∧ Real.log y ≥ 0)) ∧
  (∀ z, Real.log z < 0 → Real.exp z > 1) :=
sorry

end NUMINAMATH_CALUDE_e_power_necessary_not_sufficient_for_ln_l523_52308


namespace NUMINAMATH_CALUDE_two_face_painted_count_l523_52347

/-- Represents a cube that has been painted on all faces and cut into smaller cubes --/
structure PaintedCube where
  /-- The number of smaller cubes along each edge of the original cube --/
  edge_count : Nat
  /-- Assumption that the cube is fully painted before cutting --/
  is_fully_painted : Bool

/-- Counts the number of smaller cubes painted on exactly two faces --/
def count_two_face_painted_cubes (cube : PaintedCube) : Nat :=
  sorry

/-- Theorem stating that a cube cut into 27 smaller cubes has 12 cubes painted on two faces --/
theorem two_face_painted_count (cube : PaintedCube) 
  (h1 : cube.edge_count = 3)
  (h2 : cube.is_fully_painted = true) : 
  count_two_face_painted_cubes cube = 12 :=
sorry

end NUMINAMATH_CALUDE_two_face_painted_count_l523_52347


namespace NUMINAMATH_CALUDE_ninth_group_number_l523_52340

/-- Represents a systematic sampling scenario -/
structure SystematicSampling where
  totalWorkers : ℕ
  sampleSize : ℕ
  samplingInterval : ℕ
  fifthGroupNumber : ℕ

/-- Calculates the number drawn from a specific group given the sampling parameters -/
def groupNumber (s : SystematicSampling) (groupIndex : ℕ) : ℕ :=
  s.fifthGroupNumber + (groupIndex - 5) * s.samplingInterval

/-- Theorem stating that for the given systematic sampling scenario, 
    the number drawn from the 9th group is 43 -/
theorem ninth_group_number (s : SystematicSampling) 
  (h1 : s.totalWorkers = 100)
  (h2 : s.sampleSize = 20)
  (h3 : s.samplingInterval = 5)
  (h4 : s.fifthGroupNumber = 23) :
  groupNumber s 9 = 43 := by
  sorry

end NUMINAMATH_CALUDE_ninth_group_number_l523_52340


namespace NUMINAMATH_CALUDE_equal_probabilities_decreasing_probabilities_l523_52328

/-- Represents the probability of finding a specific item -/
def item_probability : ℝ := 0.1

/-- Represents the total number of items in the collection -/
def total_items : ℕ := 10

/-- Represents the probability that the second collection is missing exactly k items when the first collection is completed -/
noncomputable def p (k : ℕ) : ℝ := sorry

/-- The probability of missing 1 item equals the probability of missing 2 items -/
theorem equal_probabilities : p 1 = p 2 := by sorry

/-- The probabilities form a strictly decreasing sequence for k from 2 to 10 -/
theorem decreasing_probabilities : ∀ k ∈ Finset.range 9, p (k + 2) > p (k + 3) := by sorry

end NUMINAMATH_CALUDE_equal_probabilities_decreasing_probabilities_l523_52328


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l523_52350

def A : Set ℤ := {x | x^2 - 2*x - 3 < 0}
def B : Set ℤ := {-2, -1, 0, 1}

theorem union_of_A_and_B : A ∪ B = {-2, -1, 0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l523_52350


namespace NUMINAMATH_CALUDE_specific_trapezoid_ratio_l523_52389

/-- Represents a trapezoid with extended legs -/
structure ExtendedTrapezoid where
  -- Base lengths
  ab : ℝ
  cd : ℝ
  -- Height
  h : ℝ
  -- Condition that it's a valid trapezoid (cd > ab)
  h_valid : cd > ab

/-- The ratio of the area of triangle EAB to the area of trapezoid ABCD -/
def area_ratio (t : ExtendedTrapezoid) : ℝ :=
  -- Definition to be filled
  sorry

/-- Theorem stating the ratio for the specific trapezoid in the problem -/
theorem specific_trapezoid_ratio :
  let t : ExtendedTrapezoid := ⟨5, 20, 12, by norm_num⟩
  area_ratio t = 1 / 15 := by
  sorry

end NUMINAMATH_CALUDE_specific_trapezoid_ratio_l523_52389


namespace NUMINAMATH_CALUDE_fraction_product_simplification_l523_52358

theorem fraction_product_simplification :
  (2 / 3) * (3 / 7) * (7 / 4) * (4 / 5) * (5 / 6) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_simplification_l523_52358


namespace NUMINAMATH_CALUDE_sixth_is_wednesday_l523_52342

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Calculates the day of the week for a given date in a month starting with Friday -/
def dayOfWeek (date : Nat) : DayOfWeek :=
  match (date - 1) % 7 with
  | 0 => DayOfWeek.Friday
  | 1 => DayOfWeek.Saturday
  | 2 => DayOfWeek.Sunday
  | 3 => DayOfWeek.Monday
  | 4 => DayOfWeek.Tuesday
  | 5 => DayOfWeek.Wednesday
  | _ => DayOfWeek.Thursday

theorem sixth_is_wednesday 
  (h1 : ∃ (x : Nat), x + (x + 7) + (x + 14) + (x + 21) + (x + 28) = 75) 
  : dayOfWeek 6 = DayOfWeek.Wednesday := by
  sorry

end NUMINAMATH_CALUDE_sixth_is_wednesday_l523_52342


namespace NUMINAMATH_CALUDE_mitchell_chews_145_pieces_l523_52313

/-- The number of pieces of gum Mitchell chews -/
def chewed_pieces (packets : ℕ) (pieces_per_packet : ℕ) (unchewed : ℕ) : ℕ :=
  packets * pieces_per_packet - unchewed

/-- Proof that Mitchell chews 145 pieces of gum -/
theorem mitchell_chews_145_pieces :
  chewed_pieces 15 10 5 = 145 := by
  sorry

end NUMINAMATH_CALUDE_mitchell_chews_145_pieces_l523_52313


namespace NUMINAMATH_CALUDE_triangle_problem_l523_52345

open Real

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  0 < A ∧ A < π/2 →
  0 < B ∧ B < π/2 →
  0 < C ∧ C < π/2 →
  a > 0 ∧ b > 0 ∧ c > 0 →
  b = 2 * a * sin A / sin B →
  c = 2 * a * sin C / sin B →
  b / 2 = (2 * a * sin A * cos C + c * sin (2 * A)) / 2 →
  (A = π/6 ∧
   (a = 2 →
    ∀ (b' c' : ℝ),
      b' > 0 ∧ c' > 0 →
      b' = 2 * sin A / sin B →
      c' = 2 * sin C / sin B →
      1/2 * b' * c' * sin A ≤ 2 + sqrt 3)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l523_52345


namespace NUMINAMATH_CALUDE_unique_valid_number_l523_52316

/-- Represents a four-digit integer as a tuple of its digits -/
def FourDigitInt := (Fin 10 × Fin 10 × Fin 10 × Fin 10)

/-- Converts a pair of digits to a two-digit integer -/
def twoDigitInt (a b : Fin 10) : Nat := 10 * a.val + b.val

/-- Checks if three numbers form a geometric sequence -/
def isGeometricSequence (x y z : Nat) : Prop := ∃ r : ℚ, r > 1 ∧ y = r * x ∧ z = r * y

/-- Predicate for a valid four-digit integer satisfying the problem conditions -/
def isValidNumber (n : FourDigitInt) : Prop :=
  let (a, b, c, d) := n
  a ≠ 0 ∧
  isGeometricSequence (twoDigitInt a b) (twoDigitInt b c) (twoDigitInt c d)

theorem unique_valid_number :
  ∃! n : FourDigitInt, isValidNumber n :=
sorry

end NUMINAMATH_CALUDE_unique_valid_number_l523_52316


namespace NUMINAMATH_CALUDE_fraction_equality_l523_52390

theorem fraction_equality (x y : ℚ) (hx : x = 4/7) (hy : y = 5/11) : 
  (7*x + 11*y) / (77*x*y) = 9/20 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l523_52390


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l523_52329

theorem quadratic_roots_range (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > 1 ∧ x₂ > 1 ∧ x₁^2 - 4*x₁ + a = 0 ∧ x₂^2 - 4*x₂ + a = 0) 
  ↔ 
  (3 < a ∧ a ≤ 4) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l523_52329


namespace NUMINAMATH_CALUDE_darwin_remaining_money_l523_52305

/-- Calculates the remaining money after Darwin's expenditures --/
def remaining_money (initial : ℝ) : ℝ :=
  let after_gas := initial * (1 - 0.35)
  let after_food := after_gas * (1 - 0.2)
  let after_clothing := after_food * (1 - 0.25)
  after_clothing * (1 - 0.15)

/-- Theorem stating that Darwin's remaining money is $4,972.50 --/
theorem darwin_remaining_money :
  remaining_money 15000 = 4972.50 := by
  sorry

end NUMINAMATH_CALUDE_darwin_remaining_money_l523_52305


namespace NUMINAMATH_CALUDE_matchstick_ratio_is_half_l523_52397

/-- The ratio of matchsticks used to matchsticks originally had -/
def matchstick_ratio (houses : ℕ) (sticks_per_house : ℕ) (original_sticks : ℕ) : ℚ :=
  (houses * sticks_per_house : ℚ) / original_sticks

/-- Proof that the ratio of matchsticks used to matchsticks originally had is 1/2 -/
theorem matchstick_ratio_is_half :
  matchstick_ratio 30 10 600 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_matchstick_ratio_is_half_l523_52397


namespace NUMINAMATH_CALUDE_regular_polygon_properties_l523_52300

/-- A regular polygon with an exterior angle of 18 degrees has 20 sides and interior angles of 162 degrees. -/
theorem regular_polygon_properties :
  ∀ (n : ℕ) (exterior_angle interior_angle : ℝ),
  exterior_angle = 18 →
  n * exterior_angle = 360 →
  interior_angle = (n - 2 : ℝ) * 180 / n →
  n = 20 ∧ interior_angle = 162 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_properties_l523_52300


namespace NUMINAMATH_CALUDE_patty_score_proof_l523_52398

def june_score : ℝ := 97
def josh_score : ℝ := 100
def henry_score : ℝ := 94
def average_score : ℝ := 94

theorem patty_score_proof (patty_score : ℝ) : 
  (june_score + josh_score + henry_score + patty_score) / 4 = average_score →
  patty_score = 85 := by
sorry

end NUMINAMATH_CALUDE_patty_score_proof_l523_52398


namespace NUMINAMATH_CALUDE_max_leftover_apples_l523_52380

theorem max_leftover_apples (n : ℕ) (students : ℕ) (h : students = 8) :
  ∃ (apples_per_student : ℕ) (leftover : ℕ),
    n = students * apples_per_student + leftover ∧
    leftover < students ∧
    leftover ≤ 7 ∧
    (∀ k, k > leftover → ¬(∃ m, n = students * m + k)) :=
by sorry

end NUMINAMATH_CALUDE_max_leftover_apples_l523_52380


namespace NUMINAMATH_CALUDE_cannot_reach_54_from_12_l523_52391

/-- Represents the possible operations that can be performed on the number -/
inductive Operation
  | MultiplyBy2
  | DivideBy2
  | MultiplyBy3
  | DivideBy3

/-- Applies a single operation to a number -/
def applyOperation (n : ℕ) (op : Operation) : ℕ :=
  match op with
  | Operation.MultiplyBy2 => n * 2
  | Operation.DivideBy2 => n / 2
  | Operation.MultiplyBy3 => n * 3
  | Operation.DivideBy3 => n / 3

/-- Applies a sequence of operations to a number -/
def applyOperations (initial : ℕ) (ops : List Operation) : ℕ :=
  ops.foldl applyOperation initial

/-- Theorem stating that it's impossible to reach 54 from 12 after 60 operations -/
theorem cannot_reach_54_from_12 (ops : List Operation) :
  ops.length = 60 → applyOperations 12 ops ≠ 54 := by
  sorry


end NUMINAMATH_CALUDE_cannot_reach_54_from_12_l523_52391


namespace NUMINAMATH_CALUDE_sum_of_x_coordinates_l523_52343

-- Define the points
def O : ℝ × ℝ := (0, 0)
def P : ℝ → ℝ × ℝ := λ t ↦ (5*t, 12*t)
def Q : ℝ → ℝ × ℝ := λ t ↦ (8*t, 6*t)

-- State the theorem
theorem sum_of_x_coordinates (t : ℝ) : 
  (P t).1 + (Q t).1 = 13*t := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_coordinates_l523_52343


namespace NUMINAMATH_CALUDE_jogger_count_difference_l523_52378

/-- Proves the difference in jogger counts between Christopher and Alexander --/
theorem jogger_count_difference :
  ∀ (christopher_count tyson_count alexander_count : ℕ),
  christopher_count = 80 →
  christopher_count = 20 * tyson_count →
  alexander_count = tyson_count + 22 →
  christopher_count - alexander_count = 54 := by
sorry

end NUMINAMATH_CALUDE_jogger_count_difference_l523_52378


namespace NUMINAMATH_CALUDE_yogurt_price_is_2_5_l523_52372

/-- The price of a pack of yogurt in yuan -/
def yogurt_price : ℝ := 2.5

/-- The price of a pack of fresh milk in yuan -/
def milk_price : ℝ := 1

/-- The total cost of 4 packs of yogurt and 4 packs of fresh milk is 14 yuan -/
axiom first_purchase : 4 * yogurt_price + 4 * milk_price = 14

/-- The total cost of 2 packs of yogurt and 8 packs of fresh milk is 13 yuan -/
axiom second_purchase : 2 * yogurt_price + 8 * milk_price = 13

/-- The price of each pack of yogurt is 2.5 yuan -/
theorem yogurt_price_is_2_5 : yogurt_price = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_yogurt_price_is_2_5_l523_52372


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l523_52348

theorem arithmetic_sequence_problem (x : ℚ) (n : ℕ) : 
  let a₁ := 3 * x - 2
  let a₂ := 7 * x - 15
  let a₃ := 4 * x + 3
  let d := a₂ - a₁
  let aₙ := a₁ + (n - 1) * d
  (a₂ - a₁ = a₃ - a₂) ∧ (aₙ = 4020) → n = 851 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l523_52348


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l523_52354

-- Problem 1
theorem simplify_expression_1 (x : ℝ) : (2*x + 1) * (2*x - 1) = 4*x^2 - 1 := by
  sorry

-- Problem 2
theorem simplify_expression_2 (x y : ℝ) : (x - 2*y)^2 - x*y = x^2 - 5*x*y + 4*y^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l523_52354


namespace NUMINAMATH_CALUDE_total_apples_picked_l523_52338

-- Define the number of apples picked by each person
def benny_apples : ℕ := 2 * 4
def dan_apples : ℕ := 9 * 5
def sarah_apples : ℕ := (dan_apples + 1) / 2  -- Rounding up
def lisa_apples : ℕ := ((3 * (benny_apples + dan_apples) + 4) / 5)  -- Rounding up

-- Theorem to prove
theorem total_apples_picked : 
  benny_apples + dan_apples + sarah_apples + lisa_apples = 108 := by
  sorry


end NUMINAMATH_CALUDE_total_apples_picked_l523_52338


namespace NUMINAMATH_CALUDE_triangle_max_height_l523_52331

/-- In a triangle ABC with sides a, b, c corresponding to angles A, B, C respectively,
    given that c = 1 and a*cos(B) + b*cos(A) = 2*cos(C),
    the maximum value of the height h on side AB is √3/2 -/
theorem triangle_max_height (a b c : ℝ) (A B C : ℝ) (h : ℝ) :
  c = 1 →
  a * Real.cos B + b * Real.cos A = 2 * Real.cos C →
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  a / Real.sin A = b / Real.sin B →
  b / Real.sin B = c / Real.sin C →
  h ≤ Real.sqrt 3 / 2 ∧ ∃ (a' b' : ℝ), h = Real.sqrt 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_height_l523_52331


namespace NUMINAMATH_CALUDE_total_cars_on_train_l523_52395

/-- The number of cars Rita counted in the first 15 seconds -/
def initial_cars : ℕ := 9

/-- The time in seconds during which Rita counted the initial cars -/
def initial_time : ℕ := 15

/-- The total time in seconds for the train to pass -/
def total_time : ℕ := 195

/-- The rate of cars passing per second -/
def rate : ℚ := initial_cars / initial_time

/-- The theorem stating the total number of cars on the train -/
theorem total_cars_on_train : ⌊rate * total_time⌋ = 117 := by sorry

end NUMINAMATH_CALUDE_total_cars_on_train_l523_52395


namespace NUMINAMATH_CALUDE_lords_partition_l523_52394

/-- A graph with vertices of type α -/
structure Graph (α : Type) where
  adj : α → α → Prop

/-- The degree of a vertex in a graph -/
def degree {α : Type} (G : Graph α) (v : α) : ℕ := 
  sorry

/-- A partition of a set into two subsets -/
def Partition (α : Type) := (α → Bool)

/-- The number of adjacent vertices in the same partition -/
def samePartitionDegree {α : Type} (G : Graph α) (p : Partition α) (v : α) : ℕ := 
  sorry

theorem lords_partition {α : Type} (G : Graph α) :
  (∀ v : α, degree G v ≤ 3) →
  ∃ p : Partition α, ∀ v : α, samePartitionDegree G p v ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_lords_partition_l523_52394


namespace NUMINAMATH_CALUDE_sphere_radius_in_unit_cube_l523_52381

/-- The radius of a sphere satisfying specific conditions in a unit cube -/
theorem sphere_radius_in_unit_cube : ∃ r : ℝ,
  (r > 0) ∧ 
  (r^4 - 4*r^3 + 6*r^2 - 8*r + 4 = 0) ∧
  ((0 - r)^2 + (0 - r)^2 + (0 - (1 - r))^2 = r^2) ∧ -- Sphere passes through A(0,0,0)
  ((1 - r)^2 + (1 - r)^2 + (0 - (1 - r))^2 = r^2) ∧ -- Sphere passes through C(1,1,0)
  ((1 - r)^2 + (0 - r)^2 = r^2) ∧                   -- Sphere touches edge through B(1,0,0)
  (1 - (1 - r) = r)                                 -- Sphere touches top face (z=1)
  := by sorry

end NUMINAMATH_CALUDE_sphere_radius_in_unit_cube_l523_52381


namespace NUMINAMATH_CALUDE_max_value_on_parabola_l523_52349

/-- The maximum value of m + n for a point (m, n) on the graph of y = -x^2 + 3 is 13/4 -/
theorem max_value_on_parabola : 
  ∃ (max : ℝ), max = 13/4 ∧ 
  ∀ (m n : ℝ), n = -m^2 + 3 → m + n ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_on_parabola_l523_52349


namespace NUMINAMATH_CALUDE_min_value_of_complex_expression_l523_52315

theorem min_value_of_complex_expression (Z : ℂ) (h : Complex.abs Z = 1) :
  ∃ (min_val : ℝ), min_val = 0 ∧ ∀ (W : ℂ), Complex.abs W = 1 → Complex.abs (W^2 - 2*W + 1) ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_min_value_of_complex_expression_l523_52315


namespace NUMINAMATH_CALUDE_correct_substitution_l523_52355

theorem correct_substitution (x y : ℝ) :
  (5 * x + 3 * y = 22) ∧ (y = x - 2) →
  5 * x + 3 * (x - 2) = 22 :=
by sorry

end NUMINAMATH_CALUDE_correct_substitution_l523_52355


namespace NUMINAMATH_CALUDE_company_uses_systematic_sampling_l523_52366

/-- Represents a sampling method -/
inductive SamplingMethod
| LotteryMethod
| RandomNumberTableMethod
| SystematicSampling
| StratifiedSampling

/-- Represents a production line -/
structure ProductionLine :=
  (uniform : Bool)

/-- Represents a sampling process -/
structure SamplingProcess :=
  (line : ProductionLine)
  (interval : ℕ)

/-- Determines if a sampling process is systematic -/
def is_systematic (process : SamplingProcess) : Prop :=
  process.line.uniform ∧ process.interval > 0

/-- The company's sampling method -/
def company_sampling : SamplingProcess :=
  { line := { uniform := true },
    interval := 10 }

/-- Theorem stating that the company's sampling method is systematic sampling -/
theorem company_uses_systematic_sampling :
  is_systematic company_sampling ∧ 
  SamplingMethod.SystematicSampling = 
    (match company_sampling with
     | { line := { uniform := true }, interval := 10 } => SamplingMethod.SystematicSampling
     | _ => SamplingMethod.LotteryMethod) :=
sorry

end NUMINAMATH_CALUDE_company_uses_systematic_sampling_l523_52366


namespace NUMINAMATH_CALUDE_tan_two_implications_l523_52359

theorem tan_two_implications (θ : Real) (h : Real.tan θ = 2) : 
  (Real.cos θ)^2 = 1/5 ∧ (Real.sin θ)^2 = 4/5 ∧ 
  (4 * Real.sin θ - 3 * Real.cos θ) / (6 * Real.cos θ + 2 * Real.sin θ) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_tan_two_implications_l523_52359


namespace NUMINAMATH_CALUDE_triangle_50_40_l523_52324

-- Define the triangle operation
def triangle (a b : ℤ) : ℤ := a * b + (a - b) + 6

-- Theorem statement
theorem triangle_50_40 : triangle 50 40 = 2016 := by
  sorry

end NUMINAMATH_CALUDE_triangle_50_40_l523_52324


namespace NUMINAMATH_CALUDE_max_sum_on_circle_l523_52377

def is_on_circle (x y : ℤ) : Prop :=
  (x - 1)^2 + (y + 2)^2 = 16

theorem max_sum_on_circle :
  ∃ (a b : ℤ), is_on_circle a b ∧
  ∀ (x y : ℤ), is_on_circle x y → x + y ≤ a + b ∧
  a + b = 3 :=
sorry

end NUMINAMATH_CALUDE_max_sum_on_circle_l523_52377


namespace NUMINAMATH_CALUDE_dana_soda_consumption_l523_52368

/-- The number of milliliters in one liter -/
def ml_per_liter : ℕ := 1000

/-- The size of the soda bottle in liters -/
def bottle_size : ℕ := 2

/-- The number of days the bottle lasts -/
def days_lasted : ℕ := 4

/-- Dana's daily soda consumption in milliliters -/
def daily_consumption : ℕ := (bottle_size * ml_per_liter) / days_lasted

theorem dana_soda_consumption :
  daily_consumption = 500 :=
sorry

end NUMINAMATH_CALUDE_dana_soda_consumption_l523_52368


namespace NUMINAMATH_CALUDE_rain_probability_tel_aviv_l523_52362

def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

theorem rain_probability_tel_aviv : 
  let n : ℕ := 6
  let k : ℕ := 4
  let p : ℝ := 0.5
  binomial_probability n k p = 0.234375 := by
sorry

end NUMINAMATH_CALUDE_rain_probability_tel_aviv_l523_52362


namespace NUMINAMATH_CALUDE_hannah_savings_l523_52351

theorem hannah_savings (a₁ : ℕ) (r : ℕ) (n : ℕ) (last_term : ℕ) :
  a₁ = 4 → r = 2 → n = 4 → last_term = 20 →
  (a₁ * (r^n - 1) / (r - 1)) + last_term = 80 := by
  sorry

end NUMINAMATH_CALUDE_hannah_savings_l523_52351


namespace NUMINAMATH_CALUDE_remainder_product_mod_75_l523_52310

theorem remainder_product_mod_75 : (3203 * 4507 * 9929) % 75 = 34 := by
  sorry

end NUMINAMATH_CALUDE_remainder_product_mod_75_l523_52310


namespace NUMINAMATH_CALUDE_a_to_b_equals_negative_one_l523_52369

theorem a_to_b_equals_negative_one (a b : ℝ) (h : |a + 1| = -(b - 3)^2) : a^b = -1 := by
  sorry

end NUMINAMATH_CALUDE_a_to_b_equals_negative_one_l523_52369


namespace NUMINAMATH_CALUDE_soccer_field_kids_l523_52336

/-- The number of kids on a soccer field after more kids join -/
def total_kids (initial : ℕ) (joined : ℕ) : ℕ :=
  initial + joined

/-- Theorem: The total number of kids on the soccer field is 36 -/
theorem soccer_field_kids : total_kids 14 22 = 36 := by
  sorry

end NUMINAMATH_CALUDE_soccer_field_kids_l523_52336


namespace NUMINAMATH_CALUDE_complex_number_fourth_quadrant_range_l523_52371

theorem complex_number_fourth_quadrant_range (a : ℝ) : 
  let z : ℂ := (2 + Complex.I) * (a + 2 * Complex.I^3)
  (z.re > 0 ∧ z.im < 0) → -1 < a ∧ a < 4 :=
by sorry

end NUMINAMATH_CALUDE_complex_number_fourth_quadrant_range_l523_52371


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l523_52330

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Line structure -/
structure Line where
  m : ℝ
  b : ℝ

/-- Circle structure -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Theorem statement -/
theorem parabola_line_intersection (C : Parabola) (l : Line) (M N : ℝ × ℝ) :
  l.m = -Real.sqrt 3 ∧ l.b = Real.sqrt 3 →  -- Line equation: y = -√3(x-1)
  (C.p / 2, 0) ∈ {(x, y) | y = l.m * x + l.b} →  -- Line passes through focus
  M ∈ {(x, y) | y^2 = 2 * C.p * x} ∧ N ∈ {(x, y) | y^2 = 2 * C.p * x} →  -- M and N on parabola
  M ∈ {(x, y) | y = l.m * x + l.b} ∧ N ∈ {(x, y) | y = l.m * x + l.b} →  -- M and N on line
  ∃ (circ : Circle), circ.center = ((M.1 + N.1) / 2, (M.2 + N.2) / 2) ∧
                     circ.radius = Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2) / 2 →
  C.p = 2 ∧  -- First conclusion
  circ.center.1 - circ.radius = -C.p / 2  -- Second conclusion: circle tangent to directrix
  := by sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l523_52330


namespace NUMINAMATH_CALUDE_binary_sum_equals_decimal_l523_52375

/-- Converts a binary number represented as a sum of powers of 2 to its decimal equivalent -/
def binary_to_decimal (powers : List Nat) : Nat :=
  powers.foldl (fun acc p => acc + 2^p) 0

theorem binary_sum_equals_decimal : 
  let a := binary_to_decimal [0, 1, 2, 3, 4, 5, 6, 7, 8]  -- 111111111₂
  let b := binary_to_decimal [2, 3, 4, 5]                 -- 110110₂
  a + b = 571 := by sorry

end NUMINAMATH_CALUDE_binary_sum_equals_decimal_l523_52375


namespace NUMINAMATH_CALUDE_cube_sum_divided_by_quadratic_minus_product_plus_square_l523_52311

theorem cube_sum_divided_by_quadratic_minus_product_plus_square (a b : ℝ) :
  a = 6 ∧ b = 3 → (a^3 + b^3) / (a^2 - a*b + b^2) = 9 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_divided_by_quadratic_minus_product_plus_square_l523_52311


namespace NUMINAMATH_CALUDE_unique_number_l523_52386

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_product (n : ℕ) : ℕ :=
  (n / 10) * (n % 10)

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem unique_number : ∃! n : ℕ, 
  is_two_digit n ∧ 
  n % 2 = 1 ∧ 
  n % 13 = 0 ∧ 
  is_perfect_square (digit_product n) ∧
  n = 91 := by
sorry

end NUMINAMATH_CALUDE_unique_number_l523_52386


namespace NUMINAMATH_CALUDE_count_tuples_divisible_sum_l523_52307

theorem count_tuples_divisible_sum : 
  let n := 2012
  let f : Fin n → ℕ → ℕ := fun i x => (i.val + 1) * x
  (Finset.univ.filter (fun t : Fin n → Fin n => 
    (Finset.sum Finset.univ (fun i => f i (t i).val)) % n = 0)).card = n^(n-1) := by
  sorry

end NUMINAMATH_CALUDE_count_tuples_divisible_sum_l523_52307


namespace NUMINAMATH_CALUDE_three_over_x_plus_one_is_fraction_l523_52399

/-- A fraction is an expression where the denominator includes a variable. -/
def is_fraction (n d : ℝ → ℝ) : Prop :=
  ∃ x, d x ≠ d 0

/-- The expression 3/(x+1) is a fraction. -/
theorem three_over_x_plus_one_is_fraction :
  is_fraction (λ _ ↦ 3) (λ x ↦ x + 1) := by
sorry

end NUMINAMATH_CALUDE_three_over_x_plus_one_is_fraction_l523_52399


namespace NUMINAMATH_CALUDE_notebook_cost_l523_52384

theorem notebook_cost (total_cost cover_cost notebook_cost : ℚ) : 
  total_cost = 3.5 →
  notebook_cost = cover_cost + 2 →
  total_cost = notebook_cost + cover_cost →
  notebook_cost = 2.75 := by
sorry

end NUMINAMATH_CALUDE_notebook_cost_l523_52384


namespace NUMINAMATH_CALUDE_euler_6_years_or_more_percentage_l523_52332

/-- Represents the number of units for each tenure range in the bar graph --/
structure EmployeeDistribution where
  less_than_2_years : ℕ
  two_to_4_years : ℕ
  four_to_6_years : ℕ
  six_to_8_years : ℕ
  eight_to_10_years : ℕ
  more_than_10_years : ℕ

/-- Calculates the percentage of employees who have worked for 6 years or more --/
def percentage_6_years_or_more (d : EmployeeDistribution) : ℚ :=
  let total := d.less_than_2_years + d.two_to_4_years + d.four_to_6_years +
                d.six_to_8_years + d.eight_to_10_years + d.more_than_10_years
  let six_plus := d.six_to_8_years + d.eight_to_10_years + d.more_than_10_years
  (six_plus : ℚ) / (total : ℚ) * 100

/-- The actual distribution of employees at Euler Company --/
def euler_distribution : EmployeeDistribution :=
  { less_than_2_years := 4
  , two_to_4_years := 6
  , four_to_6_years := 7
  , six_to_8_years := 3
  , eight_to_10_years := 2
  , more_than_10_years := 1 }

theorem euler_6_years_or_more_percentage :
  percentage_6_years_or_more euler_distribution = 26 := by
  sorry

end NUMINAMATH_CALUDE_euler_6_years_or_more_percentage_l523_52332


namespace NUMINAMATH_CALUDE_max_value_quadratic_expression_l523_52341

theorem max_value_quadratic_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^2 - 2*x*y + 3*y^2 = 10) : 
  x^2 + 2*x*y + 3*y^2 ≤ 10*(45 + 42*Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_max_value_quadratic_expression_l523_52341


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l523_52322

/-- Given a hyperbola with equation x² - y²/b² = 1 where b > 0,
    if one of its asymptotes has the equation y = 2x, then b = 2 -/
theorem hyperbola_asymptote (b : ℝ) (hb : b > 0) :
  (∀ x y : ℝ, x^2 - y^2 / b^2 = 1) →
  (∃ x y : ℝ, y = 2*x ∧ x^2 - y^2 / b^2 = 1) →
  b = 2 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l523_52322


namespace NUMINAMATH_CALUDE_total_money_divided_l523_52306

def money_division (maya annie saiji : ℕ) : Prop :=
  maya = annie / 2 ∧ annie = saiji / 2 ∧ saiji = 400

theorem total_money_divided : 
  ∀ maya annie saiji : ℕ, 
  money_division maya annie saiji → 
  maya + annie + saiji = 700 := by
sorry

end NUMINAMATH_CALUDE_total_money_divided_l523_52306


namespace NUMINAMATH_CALUDE_inequality_range_l523_52312

theorem inequality_range (m : ℝ) : 
  (∀ x : ℝ, 3 * x^2 + 1 ≥ m * x * (x - 1)) → 
  -6 ≤ m ∧ m ≤ 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_range_l523_52312


namespace NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l523_52319

theorem coefficient_x_squared_in_expansion (x : ℝ) : 
  (Finset.range 6).sum (λ k => (Nat.choose 5 k) * (2^(5-k)) * x^(5-k)) = 
  40 * x^2 + (Finset.range 6).sum (λ k => if k ≠ 3 then (Nat.choose 5 k) * (2^(5-k)) * x^(5-k) else 0) := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l523_52319


namespace NUMINAMATH_CALUDE_surface_area_ratio_l523_52388

/-- The ratio of the total surface area of n³ unit cubes to the surface area of a cube with edge length n is equal to n. -/
theorem surface_area_ratio (n : ℕ) (h : n > 0) :
  (n^3 * (6 : ℝ)) / (6 * n^2) = n :=
sorry

end NUMINAMATH_CALUDE_surface_area_ratio_l523_52388
