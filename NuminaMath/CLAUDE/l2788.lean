import Mathlib

namespace NUMINAMATH_CALUDE_two_digit_number_sum_l2788_278838

theorem two_digit_number_sum (a b : ℕ) : 
  1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 →
  (10 * a + b) - (10 * b + a) = 5 * (a + b) →
  (10 * a + b) + (10 * b + a) = 99 := by
sorry

end NUMINAMATH_CALUDE_two_digit_number_sum_l2788_278838


namespace NUMINAMATH_CALUDE_peaches_picked_l2788_278899

def initial_peaches : ℕ := 34
def current_peaches : ℕ := 86

theorem peaches_picked (initial : ℕ) (current : ℕ) :
  current ≥ initial → current - initial = current - initial :=
by sorry

end NUMINAMATH_CALUDE_peaches_picked_l2788_278899


namespace NUMINAMATH_CALUDE_ellipse_equation_l2788_278850

/-- Given an ellipse with foci on the y-axis, sum of distances from any point to the foci equal to 8,
    and focal length 2√15, prove that its standard equation is (y²/16) + x² = 1 -/
theorem ellipse_equation (a b c : ℝ) (h1 : 2 * a = 8) (h2 : 2 * c = 2 * Real.sqrt 15)
    (h3 : a ^ 2 = b ^ 2 + c ^ 2) :
  ∀ (x y : ℝ), (y ^ 2 / 16 + x ^ 2 = 1) ↔ (y ^ 2 / a ^ 2 + x ^ 2 / b ^ 2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l2788_278850


namespace NUMINAMATH_CALUDE_correct_managers_in_sample_l2788_278808

/-- Calculates the number of managers to be drawn in a stratified sample -/
def managers_in_sample (total_employees : ℕ) (total_managers : ℕ) (sample_size : ℕ) : ℕ :=
  (total_managers * sample_size) / total_employees

theorem correct_managers_in_sample :
  managers_in_sample 160 32 20 = 4 := by
  sorry

end NUMINAMATH_CALUDE_correct_managers_in_sample_l2788_278808


namespace NUMINAMATH_CALUDE_no_real_solution_for_sqrt_equation_l2788_278810

theorem no_real_solution_for_sqrt_equation :
  ¬∃ t : ℝ, Real.sqrt (49 - t^2) + 7 = 0 := by
sorry

end NUMINAMATH_CALUDE_no_real_solution_for_sqrt_equation_l2788_278810


namespace NUMINAMATH_CALUDE_square_circumcircle_integer_points_l2788_278826

/-- The circumcircle of a square with side length 1978 contains no integer points other than the vertices of the square. -/
theorem square_circumcircle_integer_points :
  ∀ x y : ℤ,
  (x - 989)^2 + (y - 989)^2 = 2 * 989^2 →
  (x = 0 ∧ y = 0) ∨ (x = 0 ∧ y = 1978) ∨ (x = 1978 ∧ y = 0) ∨ (x = 1978 ∧ y = 1978) :=
by sorry


end NUMINAMATH_CALUDE_square_circumcircle_integer_points_l2788_278826


namespace NUMINAMATH_CALUDE_sum_of_real_and_imag_parts_of_z_l2788_278849

theorem sum_of_real_and_imag_parts_of_z (i : ℂ) (h : i^2 = -1) : 
  let z : ℂ := (1 + 2*i) / i
  (z.re + z.im : ℝ) = 1 := by
sorry

end NUMINAMATH_CALUDE_sum_of_real_and_imag_parts_of_z_l2788_278849


namespace NUMINAMATH_CALUDE_books_read_during_travel_l2788_278877

theorem books_read_during_travel (total_distance : ℕ) (reading_rate : ℕ) (books_finished : ℕ) : 
  total_distance = 6760 → reading_rate = 450 → books_finished = total_distance / reading_rate → books_finished = 15 := by
  sorry

end NUMINAMATH_CALUDE_books_read_during_travel_l2788_278877


namespace NUMINAMATH_CALUDE_price_change_l2788_278890

theorem price_change (r s : ℝ) (h : r ≠ -100) (h2 : s ≠ 100) : 
  let initial_price := (10000 : ℝ) / (10000 + 100 * (r - s) - r * s)
  let price_after_increase := initial_price * (1 + r / 100)
  let final_price := price_after_increase * (1 - s / 100)
  final_price = 1 :=
by sorry

end NUMINAMATH_CALUDE_price_change_l2788_278890


namespace NUMINAMATH_CALUDE_profit_function_correct_max_profit_production_break_even_range_l2788_278829

/-- Represents the production and profit model of a company -/
structure CompanyModel where
  fixedCost : ℝ
  variableCost : ℝ
  annualDemand : ℝ
  revenueFunction : ℝ → ℝ

/-- The company's specific model -/
def company : CompanyModel :=
  { fixedCost := 0.5,  -- In ten thousand yuan
    variableCost := 0.025,  -- In ten thousand yuan per hundred units
    annualDemand := 5,  -- In hundreds of units
    revenueFunction := λ x => 5 * x - x^2 }  -- In ten thousand yuan

/-- The profit function for the company -/
def profitFunction (x : ℝ) : ℝ :=
  company.revenueFunction x - (company.variableCost * x + company.fixedCost)

/-- Theorem stating the correctness of the profit function -/
theorem profit_function_correct (x : ℝ) (h : 0 ≤ x ∧ x ≤ 5) :
  profitFunction x = 5 * x - x^2 - (0.025 * x + 0.5) := by sorry

/-- Theorem stating the maximum profit production -/
theorem max_profit_production :
  ∃ x, x = 4.75 ∧ ∀ y, 0 ≤ y ∧ y ≤ 5 → profitFunction x ≥ profitFunction y := by sorry

/-- Theorem stating the break-even production range -/
theorem break_even_range :
  ∃ a b, a = 0.1 ∧ b = 48 ∧ 
  ∀ x, (a ≤ x ∧ x ≤ 5) ∨ (5 < x ∧ x < b) → profitFunction x ≥ 0 := by sorry

end NUMINAMATH_CALUDE_profit_function_correct_max_profit_production_break_even_range_l2788_278829


namespace NUMINAMATH_CALUDE_distance_table_1_to_3_l2788_278887

/-- Calculates the distance between the first and third table in a relay race. -/
def distance_between_tables_1_and_3 (race_length : ℕ) (num_tables : ℕ) : ℕ :=
  2 * (race_length / num_tables)

/-- Proves that in a 1200-meter race with 6 equally spaced tables, 
    the distance between the first and third table is 400 meters. -/
theorem distance_table_1_to_3 : 
  distance_between_tables_1_and_3 1200 6 = 400 := by
  sorry

end NUMINAMATH_CALUDE_distance_table_1_to_3_l2788_278887


namespace NUMINAMATH_CALUDE_distribute_teachers_count_l2788_278878

/-- The number of ways to distribute teachers to schools -/
def distribute_teachers : ℕ :=
  let chinese_teachers := 2
  let math_teachers := 4
  let total_teachers := chinese_teachers + math_teachers
  let schools := 2
  let teachers_per_school := 3
  let ways_to_choose_math := Nat.choose math_teachers (teachers_per_school - 1)
  ways_to_choose_math * schools

/-- Theorem stating that the number of ways to distribute teachers is 12 -/
theorem distribute_teachers_count : distribute_teachers = 12 := by
  sorry

end NUMINAMATH_CALUDE_distribute_teachers_count_l2788_278878


namespace NUMINAMATH_CALUDE_toothbrush_ratio_l2788_278858

theorem toothbrush_ratio (total brushes_jan brushes_feb brushes_mar : ℕ)
  (busiest_slowest_diff : ℕ) :
  total = 330 →
  brushes_jan = 53 →
  brushes_feb = 67 →
  brushes_mar = 46 →
  busiest_slowest_diff = 36 →
  ∃ (brushes_apr brushes_may : ℕ),
    brushes_apr + brushes_may = total - (brushes_jan + brushes_feb + brushes_mar) ∧
    brushes_apr - brushes_may = busiest_slowest_diff ∧
    brushes_apr * 16 = brushes_may * 25 :=
by sorry

end NUMINAMATH_CALUDE_toothbrush_ratio_l2788_278858


namespace NUMINAMATH_CALUDE_complex_exponential_to_rectangular_l2788_278876

theorem complex_exponential_to_rectangular : 
  ∃ (z : ℂ), z = Real.sqrt 2 * Complex.exp (Complex.I * (13 * Real.pi / 6)) ∧ 
             z = (Real.sqrt 6 / 2 : ℂ) + Complex.I * (Real.sqrt 2 / 2 : ℂ) := by
  sorry

end NUMINAMATH_CALUDE_complex_exponential_to_rectangular_l2788_278876


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l2788_278848

theorem absolute_value_inequality (x : ℝ) : 
  3 < |x + 2| ∧ |x + 2| ≤ 6 ↔ (-8 ≤ x ∧ x < -5) ∨ (1 < x ∧ x ≤ 4) := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l2788_278848


namespace NUMINAMATH_CALUDE_max_value_fraction_sum_l2788_278875

theorem max_value_fraction_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x / (2 * x + y)) + (y / (x + 2 * y)) ≤ 2 / 3 ∧
  ∃ x y, x > 0 ∧ y > 0 ∧ (x / (2 * x + y)) + (y / (x + 2 * y)) = 2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_max_value_fraction_sum_l2788_278875


namespace NUMINAMATH_CALUDE_truck_capacity_l2788_278860

theorem truck_capacity (large small : ℝ) 
  (h1 : 2 * large + 3 * small = 15.5)
  (h2 : 5 * large + 6 * small = 35) :
  3 * large + 2 * small = 17 := by
sorry

end NUMINAMATH_CALUDE_truck_capacity_l2788_278860


namespace NUMINAMATH_CALUDE_edge_coloring_theorem_l2788_278820

/-- Given a complete graph K_n with n vertices, this theorem states that:
    1) If we color the edges with at least n colors, there will be a triangle with all edges in different colors.
    2) If we color the edges with at most n-3 colors, there will be a cycle of length 3 or 4 with all edges in the same color.
    3) For n = 2023, it's possible to color the edges using 2022 colors without violating the conditions,
       and it's also possible using 2020 colors without violating the conditions.
    4) The difference between the maximum and minimum number of colors that satisfy the conditions is 2. -/
theorem edge_coloring_theorem (n : ℕ) (h : n = 2023) :
  (∀ (coloring : Fin n → Fin n → Fin n), 
    (∃ (i j k : Fin n), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
      coloring i j ≠ coloring j k ∧ coloring j k ≠ coloring i k ∧ coloring i j ≠ coloring i k)) ∧
  (∀ (coloring : Fin n → Fin n → Fin (n-3)), 
    (∃ (i j k l : Fin n), (i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ l ≠ i) ∧ 
      ((coloring i j = coloring j k ∧ coloring j k = coloring k i) ∨
       (coloring i j = coloring j k ∧ coloring j k = coloring k l ∧ coloring k l = coloring l i)))) ∧
  (∃ (coloring : Fin n → Fin n → Fin 2022), 
    (∀ (i j k : Fin n), i ≠ j ∧ j ≠ k ∧ i ≠ k → 
      ¬(coloring i j = coloring j k ∧ coloring j k = coloring i k) ∧
      ¬(coloring i j ≠ coloring j k ∧ coloring j k ≠ coloring i k ∧ coloring i j ≠ coloring i k)) ∧
    (∀ (i j k l : Fin n), i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ l ≠ i → 
      ¬(coloring i j = coloring j k ∧ coloring j k = coloring k l ∧ coloring k l = coloring l i))) ∧
  (∃ (coloring : Fin n → Fin n → Fin 2020), 
    (∀ (i j k : Fin n), i ≠ j ∧ j ≠ k ∧ i ≠ k → 
      ¬(coloring i j = coloring j k ∧ coloring j k = coloring i k) ∧
      ¬(coloring i j ≠ coloring j k ∧ coloring j k ≠ coloring i k ∧ coloring i j ≠ coloring i k)) ∧
    (∀ (i j k l : Fin n), i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ l ≠ i → 
      ¬(coloring i j = coloring j k ∧ coloring j k = coloring k l ∧ coloring k l = coloring l i))) ∧
  (2022 - 2020 = 2) :=
by sorry


end NUMINAMATH_CALUDE_edge_coloring_theorem_l2788_278820


namespace NUMINAMATH_CALUDE_max_area_of_three_rectangles_l2788_278821

/-- Given two rectangles with dimensions 9x12 and 10x15, 
    prove that the maximum area of a rectangle that can be formed 
    by arranging these two rectangles along with a third rectangle is 330. -/
theorem max_area_of_three_rectangles : 
  let rect1_width : ℝ := 9
  let rect1_height : ℝ := 12
  let rect2_width : ℝ := 10
  let rect2_height : ℝ := 15
  ∃ (rect3_width rect3_height : ℝ),
    (max 
      (max rect1_width rect2_width * (rect1_height + rect2_height))
      (max rect1_height rect2_height * (rect1_width + rect2_width))
    ) = 330 := by
  sorry

end NUMINAMATH_CALUDE_max_area_of_three_rectangles_l2788_278821


namespace NUMINAMATH_CALUDE_similar_triangles_side_length_l2788_278817

theorem similar_triangles_side_length 
  (A₁ A₂ : ℕ) (k : ℕ) (side_small : ℝ) :
  A₁ > A₂ →
  A₁ - A₂ = 32 →
  A₁ = k^2 * A₂ →
  side_small = 4 →
  ∃ (side_large : ℝ), side_large = 12 :=
by sorry

end NUMINAMATH_CALUDE_similar_triangles_side_length_l2788_278817


namespace NUMINAMATH_CALUDE_simplification_and_exponent_sum_l2788_278807

-- Define the original expression
def original_expression (x y z : ℝ) : ℝ := (40 * x^5 * y^3 * z^8) ^ (1/3)

-- Define the simplified expression
def simplified_expression (x y z : ℝ) : ℝ := 2 * x * y * z * (5 * x^2 * z^5) ^ (1/3)

-- Theorem statement
theorem simplification_and_exponent_sum :
  ∀ x y z : ℝ, x > 0 → y > 0 → z > 0 →
  (original_expression x y z = simplified_expression x y z) ∧
  (1 + 1 + 1 = 3) := by sorry

end NUMINAMATH_CALUDE_simplification_and_exponent_sum_l2788_278807


namespace NUMINAMATH_CALUDE_expression_value_l2788_278833

theorem expression_value (x y z : ℤ) (hx : x = 25) (hy : y = 30) (hz : z = 7) :
  (x - (y - z)) - ((x - y) - (z - 1)) = 13 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2788_278833


namespace NUMINAMATH_CALUDE_calculate_ampersand_composition_l2788_278828

-- Define the operations
def ampersand_right (x : ℝ) : ℝ := 10 - x
def ampersand_left (x : ℝ) : ℝ := x - 10

-- State the theorem
theorem calculate_ampersand_composition : ampersand_left (ampersand_right 15) = -15 := by
  sorry

end NUMINAMATH_CALUDE_calculate_ampersand_composition_l2788_278828


namespace NUMINAMATH_CALUDE_consecutive_good_numbers_l2788_278801

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Predicate for a number being "good" (not divisible by the sum of its digits) -/
def is_good (n : ℕ) : Prop := ¬(sum_of_digits n ∣ n)

/-- Main theorem -/
theorem consecutive_good_numbers (n : ℕ) (hn : n > 0) :
  ∃ (start : ℕ), ∀ (i : ℕ), i < n → is_good (start + i) := by sorry

end NUMINAMATH_CALUDE_consecutive_good_numbers_l2788_278801


namespace NUMINAMATH_CALUDE_meeting_distance_meeting_distance_is_correct_l2788_278818

/-- The distance between two people moving towards each other -/
theorem meeting_distance (a b : ℝ) (ha : a > 0) (hb : b > 0) : ℝ :=
  2 * (a + b)
where
  /-- Person A's speed in km/h -/
  speed_a : ℝ := a
  /-- Person B's speed in km/h -/
  speed_b : ℝ := b
  /-- Time taken for them to meet in hours -/
  meeting_time : ℝ := 2
  /-- The two people start from different locations -/
  different_start_locations : Prop := True
  /-- The two people start at the same time -/
  same_start_time : Prop := True
  /-- The two people move towards each other -/
  move_towards_each_other : Prop := True

theorem meeting_distance_is_correct (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  meeting_distance a b ha hb = 2 * (a + b) := by sorry

end NUMINAMATH_CALUDE_meeting_distance_meeting_distance_is_correct_l2788_278818


namespace NUMINAMATH_CALUDE_no_solution_to_inequality_system_l2788_278824

theorem no_solution_to_inequality_system :
  ¬ ∃ x : ℝ, (2 * x + 3 ≥ x + 11) ∧ ((2 * x + 5) / 3 - 1 < 2 - x) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_to_inequality_system_l2788_278824


namespace NUMINAMATH_CALUDE_cube_square_third_prime_times_fourth_prime_l2788_278888

def third_smallest_prime : Nat := 5

def fourth_smallest_prime : Nat := 7

theorem cube_square_third_prime_times_fourth_prime :
  (third_smallest_prime ^ 2) ^ 3 * fourth_smallest_prime = 109375 := by
  sorry

end NUMINAMATH_CALUDE_cube_square_third_prime_times_fourth_prime_l2788_278888


namespace NUMINAMATH_CALUDE_complex_square_problem_l2788_278844

theorem complex_square_problem (z : ℂ) (h : z⁻¹ = 1 + Complex.I) : z^2 = -Complex.I / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_square_problem_l2788_278844


namespace NUMINAMATH_CALUDE_crayons_remaining_l2788_278882

theorem crayons_remaining (initial : ℕ) (given_away : ℕ) (lost : ℕ) : 
  initial = 440 → given_away = 111 → lost = 106 → initial - given_away - lost = 223 := by
  sorry

end NUMINAMATH_CALUDE_crayons_remaining_l2788_278882


namespace NUMINAMATH_CALUDE_remainder_theorem_polynomial_remainder_l2788_278814

def f (x : ℝ) : ℝ := 5*x^4 - 12*x^3 + 3*x^2 - 8*x + 15

theorem remainder_theorem (f : ℝ → ℝ) (a : ℝ) :
  ∃ q : ℝ → ℝ, ∀ x, f x = (x - a) * q x + f a := sorry

theorem polynomial_remainder : f 4 = 543 := by sorry

end NUMINAMATH_CALUDE_remainder_theorem_polynomial_remainder_l2788_278814


namespace NUMINAMATH_CALUDE_lucky_larry_coincidence_l2788_278866

theorem lucky_larry_coincidence (a b c d e : ℤ) : 
  a = 1 → b = 2 → c = 3 → d = 4 → 
  (a - b - c - d + e = a - (b - (c - (d + e)))) → e = 3 := by
sorry

end NUMINAMATH_CALUDE_lucky_larry_coincidence_l2788_278866


namespace NUMINAMATH_CALUDE_complementary_event_l2788_278885

-- Define the sample space
def SampleSpace := List Bool

-- Define the event of missing both times
def MissBoth (outcome : SampleSpace) : Prop := outcome = [false, false]

-- Define the event of at least one hit
def AtLeastOneHit (outcome : SampleSpace) : Prop := outcome ≠ [false, false]

-- Theorem statement
theorem complementary_event : 
  ∀ (outcome : SampleSpace), MissBoth outcome ↔ ¬(AtLeastOneHit outcome) :=
sorry

end NUMINAMATH_CALUDE_complementary_event_l2788_278885


namespace NUMINAMATH_CALUDE_tree_distance_l2788_278816

theorem tree_distance (n : ℕ) (d : ℝ) (h1 : n = 6) (h2 : d = 60) :
  (n - 1) * (d / 3) = 100 := by
  sorry

end NUMINAMATH_CALUDE_tree_distance_l2788_278816


namespace NUMINAMATH_CALUDE_three_tangent_lines_m_values_l2788_278840

-- Define the function f
def f (x : ℝ) : ℝ := -x^3 + 2*x^2 - 3*x

-- Define the derivative of f
def f' (x : ℝ) : ℝ := -3*x^2 + 4*x - 3

-- Define the tangent line equation
def tangent_line (x₀ : ℝ) (x : ℝ) : ℝ := 
  f x₀ + f' x₀ * (x - x₀)

-- Define the condition for a point to be on the tangent line
def on_tangent_line (x₀ m : ℝ) : Prop :=
  m = tangent_line x₀ (-1)

-- Theorem statement
theorem three_tangent_lines_m_values :
  ∀ m : ℤ, (∃ x₁ x₂ x₃ : ℝ, 
    x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    on_tangent_line x₁ m ∧
    on_tangent_line x₂ m ∧
    on_tangent_line x₃ m) →
  m = 4 ∨ m = 5 :=
sorry

end NUMINAMATH_CALUDE_three_tangent_lines_m_values_l2788_278840


namespace NUMINAMATH_CALUDE_largest_expression_l2788_278864

theorem largest_expression : 
  let a := 3 + 1 + 2 + 9
  let b := 3 * 1 + 2 + 9
  let c := 3 + 1 * 2 + 9
  let d := 3 + 1 + 2 * 9
  let e := 3 * 1 * 2 * 9
  (e > a) ∧ (e > b) ∧ (e > c) ∧ (e > d) := by
  sorry

end NUMINAMATH_CALUDE_largest_expression_l2788_278864


namespace NUMINAMATH_CALUDE_contractor_labor_problem_l2788_278847

theorem contractor_labor_problem (planned_days : ℕ) (absent_workers : ℕ) (actual_days : ℕ) 
  (h1 : planned_days = 9)
  (h2 : absent_workers = 6)
  (h3 : actual_days = 15) :
  ∃ (original_workers : ℕ), 
    original_workers * planned_days = (original_workers - absent_workers) * actual_days ∧ 
    original_workers = 15 := by
  sorry


end NUMINAMATH_CALUDE_contractor_labor_problem_l2788_278847


namespace NUMINAMATH_CALUDE_complex_subtraction_l2788_278881

theorem complex_subtraction : (7 - 3*I) - (2 + 4*I) = 5 - 7*I := by sorry

end NUMINAMATH_CALUDE_complex_subtraction_l2788_278881


namespace NUMINAMATH_CALUDE_square_minus_self_sum_l2788_278857

theorem square_minus_self_sum : (2^2 - 2) - (3^2 - 3) + (4^2 - 4) = 8 := by
  sorry

end NUMINAMATH_CALUDE_square_minus_self_sum_l2788_278857


namespace NUMINAMATH_CALUDE_apple_problem_l2788_278803

theorem apple_problem (initial_apples : ℕ) (sold_to_jill_percent : ℚ) (sold_to_june_percent : ℚ) (given_to_teacher : ℕ) : 
  initial_apples = 150 →
  sold_to_jill_percent = 30 / 100 →
  sold_to_june_percent = 20 / 100 →
  given_to_teacher = 2 →
  initial_apples - 
    (↑initial_apples * sold_to_jill_percent).floor - 
    ((↑initial_apples - (↑initial_apples * sold_to_jill_percent).floor) * sold_to_june_percent).floor - 
    given_to_teacher = 82 :=
by sorry

end NUMINAMATH_CALUDE_apple_problem_l2788_278803


namespace NUMINAMATH_CALUDE_determine_key_lock_pairs_l2788_278837

/-- Represents a lock -/
structure Lock :=
  (id : Nat)

/-- Represents a key -/
structure Key :=
  (id : Nat)

/-- Represents a pair of locks that a key can open -/
structure LockPair :=
  (lock1 : Lock)
  (lock2 : Lock)

/-- Represents the result of testing a key on a lock -/
inductive TestResult
  | Opens
  | DoesNotOpen

/-- Represents the state of knowledge about which keys open which locks -/
structure KeyLockState :=
  (locks : Finset Lock)
  (keys : Finset Key)
  (openPairs : Finset (Key × LockPair))

/-- Represents a single test of a key on a lock -/
def test (k : Key) (l : Lock) : TestResult := sorry

/-- The main theorem to prove -/
theorem determine_key_lock_pairs 
  (locks : Finset Lock) 
  (keys : Finset Key) 
  (h1 : locks.card = 4) 
  (h2 : keys.card = 6) 
  (h3 : ∀ k : Key, k ∈ keys → (∃! p : LockPair, p.lock1 ∈ locks ∧ p.lock2 ∈ locks ∧ 
    test k p.lock1 = TestResult.Opens ∧ test k p.lock2 = TestResult.Opens))
  (h4 : ∀ k1 k2 : Key, k1 ∈ keys → k2 ∈ keys → k1 ≠ k2 → 
    ¬∃ p : LockPair, p.lock1 ∈ locks ∧ p.lock2 ∈ locks ∧ 
    test k1 p.lock1 = TestResult.Opens ∧ test k1 p.lock2 = TestResult.Opens ∧
    test k2 p.lock1 = TestResult.Opens ∧ test k2 p.lock2 = TestResult.Opens) :
  ∃ (final_state : KeyLockState) (test_count : Nat),
    test_count ≤ 13 ∧
    final_state.locks = locks ∧
    final_state.keys = keys ∧
    (∀ k : Key, k ∈ keys → 
      ∃! p : LockPair, (k, p) ∈ final_state.openPairs ∧ 
        p.lock1 ∈ locks ∧ p.lock2 ∈ locks ∧
        test k p.lock1 = TestResult.Opens ∧ 
        test k p.lock2 = TestResult.Opens) :=
by
  sorry

end NUMINAMATH_CALUDE_determine_key_lock_pairs_l2788_278837


namespace NUMINAMATH_CALUDE_parallel_lines_angle_measure_l2788_278869

-- Define the angle measures as real numbers
variable (angle1 angle2 angle5 : ℝ)

-- State the theorem
theorem parallel_lines_angle_measure :
  -- Conditions
  angle1 = (1 / 4) * angle2 →  -- ∠1 is 1/4 of ∠2
  angle1 = angle5 →            -- ∠1 and ∠5 are alternate angles (implied by parallel lines)
  angle2 + angle5 = 180 →      -- ∠2 and ∠5 form a straight line
  -- Conclusion
  angle5 = 36 := by
sorry

end NUMINAMATH_CALUDE_parallel_lines_angle_measure_l2788_278869


namespace NUMINAMATH_CALUDE_percentage_of_c_grades_l2788_278886

def grading_scale : List (String × (Int × Int)) :=
  [("A", (95, 100)), ("B", (88, 94)), ("C", (78, 87)), ("D", (65, 77)), ("F", (0, 64))]

def scores : List Int :=
  [94, 65, 59, 99, 82, 89, 90, 68, 79, 62, 85, 81, 64, 83, 91]

def is_grade_c (score : Int) : Bool :=
  78 ≤ score ∧ score ≤ 87

def count_grade_c (scores : List Int) : Nat :=
  (scores.filter is_grade_c).length

theorem percentage_of_c_grades :
  (count_grade_c scores : Rat) / (scores.length : Rat) * 100 = 100/3 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_c_grades_l2788_278886


namespace NUMINAMATH_CALUDE_square_remainders_l2788_278893

theorem square_remainders (n : ℤ) : 
  (∃ r : ℤ, r ∈ ({0, 1} : Set ℤ) ∧ n^2 ≡ r [ZMOD 3]) ∧
  (∃ r : ℤ, r ∈ ({0, 1} : Set ℤ) ∧ n^2 ≡ r [ZMOD 4]) ∧
  (∃ r : ℤ, r ∈ ({0, 1, 4} : Set ℤ) ∧ n^2 ≡ r [ZMOD 5]) ∧
  (∃ r : ℤ, r ∈ ({0, 1, 4} : Set ℤ) ∧ n^2 ≡ r [ZMOD 8]) :=
by sorry

end NUMINAMATH_CALUDE_square_remainders_l2788_278893


namespace NUMINAMATH_CALUDE_roger_candies_left_l2788_278812

/-- The number of candies Roger has left after giving some away -/
def candies_left (initial : ℕ) (given_to_stephanie : ℕ) (given_to_john : ℕ) (given_to_emily : ℕ) : ℕ :=
  initial - (given_to_stephanie + given_to_john + given_to_emily)

/-- Theorem stating that Roger has 262 candies left -/
theorem roger_candies_left :
  candies_left 350 45 25 18 = 262 := by
  sorry

end NUMINAMATH_CALUDE_roger_candies_left_l2788_278812


namespace NUMINAMATH_CALUDE_teacherStudentArrangements_eq_144_l2788_278873

/-- The number of ways to arrange 2 teachers and 4 students in a row
    with exactly 2 students between the teachers -/
def teacherStudentArrangements : ℕ :=
  3 * 2 * 24

/-- Proof that the number of arrangements is 144 -/
theorem teacherStudentArrangements_eq_144 :
  teacherStudentArrangements = 144 := by
  sorry

end NUMINAMATH_CALUDE_teacherStudentArrangements_eq_144_l2788_278873


namespace NUMINAMATH_CALUDE_calculate_total_exports_l2788_278855

/-- Calculates the total yearly exports of a country given the percentage of fruit exports,
    the percentage of orange exports within fruit exports, and the revenue from orange exports. -/
theorem calculate_total_exports (fruit_export_percent : ℝ) (orange_export_fraction : ℝ) (orange_export_revenue : ℝ) :
  fruit_export_percent = 0.20 →
  orange_export_fraction = 1 / 6 →
  orange_export_revenue = 4.25 →
  (orange_export_revenue / orange_export_fraction) / fruit_export_percent = 127.5 := by
  sorry

end NUMINAMATH_CALUDE_calculate_total_exports_l2788_278855


namespace NUMINAMATH_CALUDE_tan_quadruple_angle_l2788_278851

theorem tan_quadruple_angle (θ : Real) (h : Real.tan θ = 3) : 
  Real.tan (4 * θ) = -24 / 7 := by
  sorry

end NUMINAMATH_CALUDE_tan_quadruple_angle_l2788_278851


namespace NUMINAMATH_CALUDE_vectors_not_collinear_l2788_278872

/-- Given two vectors a and b in ℝ³, we define c₁ and c₂ as linear combinations of a and b.
    This theorem states that c₁ and c₂ are not collinear. -/
theorem vectors_not_collinear :
  let a : Fin 3 → ℝ := ![1, 0, 1]
  let b : Fin 3 → ℝ := ![-2, 3, 5]
  let c₁ : Fin 3 → ℝ := a + 2 • b
  let c₂ : Fin 3 → ℝ := 3 • a - b
  ¬ ∃ (k : ℝ), c₁ = k • c₂ := by
  sorry

end NUMINAMATH_CALUDE_vectors_not_collinear_l2788_278872


namespace NUMINAMATH_CALUDE_total_sharks_l2788_278874

/-- The total number of sharks on three beaches given specific ratios -/
theorem total_sharks (newport : ℕ) (dana_point : ℕ) (huntington : ℕ) 
  (h1 : newport = 22)
  (h2 : dana_point = 4 * newport)
  (h3 : huntington = dana_point / 2) :
  newport + dana_point + huntington = 154 := by
  sorry

end NUMINAMATH_CALUDE_total_sharks_l2788_278874


namespace NUMINAMATH_CALUDE_infinitely_many_lcm_greater_than_ck_l2788_278811

theorem infinitely_many_lcm_greater_than_ck
  (a : ℕ → ℕ)  -- Sequence of positive integers
  (h_distinct : ∀ i j, i ≠ j → a i ≠ a j)  -- Distinct elements
  (c : ℝ)  -- Real number c
  (h_c_pos : 0 < c)  -- c is positive
  (h_c_bound : c < 3/2)  -- c is less than 3/2
  : ∃ S : Set ℕ, Set.Infinite S ∧ ∀ k ∈ S, Nat.lcm (a k) (a (k+1)) > ↑k * c :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_lcm_greater_than_ck_l2788_278811


namespace NUMINAMATH_CALUDE_sum_remainder_thirteen_l2788_278863

theorem sum_remainder_thirteen : ∃ k : ℕ, (8930 + 8931 + 8932 + 8933 + 8934) = 13 * k + 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_thirteen_l2788_278863


namespace NUMINAMATH_CALUDE_boys_camp_total_l2788_278806

theorem boys_camp_total (total : ℕ) 
  (h1 : (total : ℝ) * 0.2 * 0.7 = 42) : total = 300 := by
  sorry

#check boys_camp_total

end NUMINAMATH_CALUDE_boys_camp_total_l2788_278806


namespace NUMINAMATH_CALUDE_triangle_problem_l2788_278880

theorem triangle_problem (A B C : Real) (a b c : Real) :
  -- Triangle ABC is acute
  0 < A ∧ A < π/2 ∧
  0 < B ∧ B < π/2 ∧
  0 < C ∧ C < π/2 ∧
  -- a, b, c are sides opposite to angles A, B, C
  a = 4 ∧
  b = 5 ∧
  -- Area of triangle ABC is 5√3
  (1/2) * a * b * Real.sin C = 5 * Real.sqrt 3 →
  c = Real.sqrt 21 ∧
  Real.sin A = (2 * Real.sqrt 7) / 7 := by
sorry

end NUMINAMATH_CALUDE_triangle_problem_l2788_278880


namespace NUMINAMATH_CALUDE_parabola_vertex_l2788_278896

/-- Represents a parabola in the xy-plane -/
structure Parabola where
  /-- The equation of the parabola in the form y^2 + ay + bx + c = 0 -/
  equation : ℝ → ℝ → ℝ → ℝ

/-- The vertex of a parabola -/
def vertex (p : Parabola) : ℝ × ℝ := sorry

theorem parabola_vertex :
  let p : Parabola := { equation := fun y x _ => y^2 - 4*y + 3*x + 7 }
  vertex p = (-1, 2) := by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l2788_278896


namespace NUMINAMATH_CALUDE_vitya_catches_up_in_5_minutes_l2788_278868

-- Define the initial walking speed
def initial_speed : ℝ := 1

-- Define the time they walk before Vitya turns back
def initial_time : ℝ := 10

-- Define Vitya's speed multiplier when he starts chasing
def speed_multiplier : ℝ := 5

-- Define the theorem
theorem vitya_catches_up_in_5_minutes :
  let distance := 2 * initial_speed * initial_time
  let relative_speed := speed_multiplier * initial_speed - initial_speed
  distance / relative_speed = 5 := by
sorry

end NUMINAMATH_CALUDE_vitya_catches_up_in_5_minutes_l2788_278868


namespace NUMINAMATH_CALUDE_max_value_in_region_D_l2788_278804

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 1

-- Define the asymptotes
def asymptote1 (x y : ℝ) : Prop := y = x
def asymptote2 (x y : ℝ) : Prop := y = -x

-- Define the bounding line
def boundingLine (x : ℝ) : Prop := x = 3

-- Define the region D
def regionD (x y : ℝ) : Prop :=
  x ≤ 3 ∧ y ≤ x ∧ y ≥ -x

-- Define the objective function
def objectiveFunction (x y : ℝ) : ℝ := x + 4*y

-- Theorem statement
theorem max_value_in_region_D :
  ∃ (x y : ℝ), regionD x y ∧
  ∀ (x' y' : ℝ), regionD x' y' →
  objectiveFunction x y ≥ objectiveFunction x' y' ∧
  objectiveFunction x y = 15 :=
sorry

end NUMINAMATH_CALUDE_max_value_in_region_D_l2788_278804


namespace NUMINAMATH_CALUDE_markup_percentage_of_selling_price_l2788_278894

theorem markup_percentage_of_selling_price 
  (cost selling_price markup : ℝ) 
  (h1 : markup = 0.1 * cost) 
  (h2 : selling_price = cost + markup) :
  markup / selling_price = 100 / 11 / 100 := by
sorry

end NUMINAMATH_CALUDE_markup_percentage_of_selling_price_l2788_278894


namespace NUMINAMATH_CALUDE_angies_age_problem_l2788_278845

theorem angies_age_problem (angie_age : ℕ) (certain_number : ℕ) : 
  angie_age = 8 → 2 * angie_age + certain_number = 20 → certain_number = 4 := by
  sorry

end NUMINAMATH_CALUDE_angies_age_problem_l2788_278845


namespace NUMINAMATH_CALUDE_root_range_implies_m_range_l2788_278884

theorem root_range_implies_m_range (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ < -1 ∧ x₂ > 1 ∧ 
   x₁^2 + (m^2 - 1)*x₁ + m - 2 = 0 ∧
   x₂^2 + (m^2 - 1)*x₂ + m - 2 = 0) →
  -2 < m ∧ m < 0 := by
sorry

end NUMINAMATH_CALUDE_root_range_implies_m_range_l2788_278884


namespace NUMINAMATH_CALUDE_arithmetic_sequence_s_value_l2788_278835

/-- An arithmetic sequence with 7 terms -/
structure ArithmeticSequence :=
  (a : Fin 7 → ℚ)
  (is_arithmetic : ∀ i j k : Fin 7, i.val + 1 = j.val ∧ j.val + 1 = k.val → 
                   a j - a i = a k - a j)

/-- The theorem statement -/
theorem arithmetic_sequence_s_value 
  (seq : ArithmeticSequence)
  (first_term : seq.a 0 = 20)
  (last_term : seq.a 6 = 40)
  (second_to_last : seq.a 5 = seq.a 4 + 10) :
  seq.a 4 = 30 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_s_value_l2788_278835


namespace NUMINAMATH_CALUDE_ram_bicycle_sale_loss_percentage_l2788_278846

/-- Calculates the percentage loss on the second bicycle sold by Ram -/
theorem ram_bicycle_sale_loss_percentage :
  let selling_price : ℚ := 990
  let total_cost : ℚ := 1980
  let profit_percentage_first : ℚ := 10 / 100

  let cost_price_first : ℚ := selling_price / (1 + profit_percentage_first)
  let cost_price_second : ℚ := total_cost - cost_price_first
  let loss_second : ℚ := cost_price_second - selling_price
  let loss_percentage_second : ℚ := (loss_second / cost_price_second) * 100

  loss_percentage_second = 25 / 3 := by sorry

end NUMINAMATH_CALUDE_ram_bicycle_sale_loss_percentage_l2788_278846


namespace NUMINAMATH_CALUDE_piano_harmonies_count_l2788_278834

theorem piano_harmonies_count : 
  (Nat.choose 7 3) + (Nat.choose 7 4) + (Nat.choose 7 5) + (Nat.choose 7 6) + (Nat.choose 7 7) = 99 := by
  sorry

end NUMINAMATH_CALUDE_piano_harmonies_count_l2788_278834


namespace NUMINAMATH_CALUDE_smallest_pencil_count_l2788_278819

theorem smallest_pencil_count (p : ℕ) : 
  (p > 0) →
  (p % 6 = 5) → 
  (p % 7 = 3) → 
  (p % 8 = 7) → 
  (∀ q : ℕ, q > 0 → q % 6 = 5 → q % 7 = 3 → q % 8 = 7 → p ≤ q) →
  p = 35 := by
sorry

end NUMINAMATH_CALUDE_smallest_pencil_count_l2788_278819


namespace NUMINAMATH_CALUDE_abes_age_l2788_278802

theorem abes_age (present_age : ℕ) : 
  (present_age + (present_age - 7) = 31) → present_age = 19 := by
  sorry

end NUMINAMATH_CALUDE_abes_age_l2788_278802


namespace NUMINAMATH_CALUDE_rose_apple_sharing_l2788_278831

theorem rose_apple_sharing (total_apples : ℕ) (num_friends : ℕ) (apples_per_friend : ℕ) :
  total_apples = 9 →
  num_friends = 3 →
  total_apples = num_friends * apples_per_friend →
  apples_per_friend = 3 :=
by sorry

end NUMINAMATH_CALUDE_rose_apple_sharing_l2788_278831


namespace NUMINAMATH_CALUDE_line_quadrants_l2788_278883

/-- Given a line ax + by + c = 0 where ab < 0 and bc < 0, 
    the line passes through the first, second, and third quadrants -/
theorem line_quadrants (a b c : ℝ) (hab : a * b < 0) (hbc : b * c < 0) :
  ∃ (x1 y1 x2 y2 x3 y3 : ℝ),
    (x1 > 0 ∧ y1 > 0) ∧  -- First quadrant
    (x2 < 0 ∧ y2 > 0) ∧  -- Second quadrant
    (x3 < 0 ∧ y3 < 0) ∧  -- Third quadrant
    (a * x1 + b * y1 + c = 0) ∧
    (a * x2 + b * y2 + c = 0) ∧
    (a * x3 + b * y3 + c = 0) :=
by sorry

end NUMINAMATH_CALUDE_line_quadrants_l2788_278883


namespace NUMINAMATH_CALUDE_range_of_p_l2788_278815

def h (x : ℝ) : ℝ := 4 * x - 3

def p (x : ℝ) : ℝ := h (h (h x))

theorem range_of_p :
  ∀ y ∈ Set.range (fun x => p x), 1 ≤ y ∧ y ≤ 129 ∧
  ∀ y, 1 ≤ y ∧ y ≤ 129 → ∃ x, 1 ≤ x ∧ x ≤ 3 ∧ p x = y :=
by sorry

end NUMINAMATH_CALUDE_range_of_p_l2788_278815


namespace NUMINAMATH_CALUDE_quadratic_roots_in_sixth_degree_l2788_278856

theorem quadratic_roots_in_sixth_degree (p q : ℝ) : 
  (∀ x : ℝ, x^2 - x - 1 = 0 → x^6 - p*x^2 + q = 0) → 
  p = 8 ∧ q = 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_in_sixth_degree_l2788_278856


namespace NUMINAMATH_CALUDE_mexican_food_pricing_l2788_278897

/-- Given the pricing conditions for Mexican food items, prove the cost of a specific combination. -/
theorem mexican_food_pricing
  (enchilada_price taco_price burrito_price : ℚ)
  (h1 : 2 * enchilada_price + 3 * taco_price + burrito_price = 5)
  (h2 : 3 * enchilada_price + 2 * taco_price + 2 * burrito_price = 15/2) :
  2 * enchilada_price + 2 * taco_price + 3 * burrito_price = 85/8 := by
  sorry

end NUMINAMATH_CALUDE_mexican_food_pricing_l2788_278897


namespace NUMINAMATH_CALUDE_min_adventurers_l2788_278809

/-- Represents a group of adventurers with their gem possessions -/
structure AdventurerGroup where
  rubies : Finset Nat
  emeralds : Finset Nat
  sapphires : Finset Nat
  diamonds : Finset Nat

/-- The conditions for the adventurer group -/
def validGroup (g : AdventurerGroup) : Prop :=
  (g.rubies.card = 4) ∧
  (g.emeralds.card = 10) ∧
  (g.sapphires.card = 6) ∧
  (g.diamonds.card = 14) ∧
  (∀ a ∈ g.rubies, (a ∈ g.emeralds ∨ a ∈ g.diamonds) ∧ ¬(a ∈ g.emeralds ∧ a ∈ g.diamonds)) ∧
  (∀ a ∈ g.emeralds, (a ∈ g.rubies ∨ a ∈ g.sapphires) ∧ ¬(a ∈ g.rubies ∧ a ∈ g.sapphires))

/-- The theorem stating the minimum number of adventurers -/
theorem min_adventurers (g : AdventurerGroup) (h : validGroup g) :
  (g.rubies ∪ g.emeralds ∪ g.sapphires ∪ g.diamonds).card ≥ 18 := by
  sorry


end NUMINAMATH_CALUDE_min_adventurers_l2788_278809


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2788_278854

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The line l: x + y - 1 = 0 -/
def line_l (p : Point) : Prop := p.x + p.y - 1 = 0

/-- The specific condition x=2 and y=-1 -/
def specific_condition (p : Point) : Prop := p.x = 2 ∧ p.y = -1

/-- Theorem stating that the specific condition is sufficient but not necessary -/
theorem sufficient_not_necessary :
  (∀ p : Point, specific_condition p → line_l p) ∧
  ¬(∀ p : Point, line_l p → specific_condition p) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2788_278854


namespace NUMINAMATH_CALUDE_additional_money_needed_l2788_278865

/-- Given a football team, budget, and cost per football, calculate the additional money needed --/
theorem additional_money_needed 
  (num_players : ℕ) 
  (budget : ℕ) 
  (cost_per_football : ℕ) 
  (h1 : num_players = 22)
  (h2 : budget = 1500)
  (h3 : cost_per_football = 69) : 
  (num_players * cost_per_football - budget : ℤ) = 18 := by
  sorry

end NUMINAMATH_CALUDE_additional_money_needed_l2788_278865


namespace NUMINAMATH_CALUDE_cylindrical_bar_length_l2788_278841

/-- The length of a cylindrical steel bar formed from a rectangular billet -/
theorem cylindrical_bar_length 
  (billet_length : ℝ) 
  (billet_width : ℝ) 
  (billet_height : ℝ) 
  (cylinder_diameter : ℝ) 
  (h1 : billet_length = 12.56)
  (h2 : billet_width = 5)
  (h3 : billet_height = 4)
  (h4 : cylinder_diameter = 4) : 
  (billet_length * billet_width * billet_height) / (π * (cylinder_diameter / 2)^2) = 20 := by
  sorry

#check cylindrical_bar_length

end NUMINAMATH_CALUDE_cylindrical_bar_length_l2788_278841


namespace NUMINAMATH_CALUDE_solution_satisfies_equations_l2788_278859

theorem solution_satisfies_equations :
  let x : ℚ := 599 / 204
  let y : ℚ := 65 / 136
  (7 * x - 50 * y = -3) ∧ (3 * x - 2 * y = 8) := by
  sorry

end NUMINAMATH_CALUDE_solution_satisfies_equations_l2788_278859


namespace NUMINAMATH_CALUDE_elvis_song_writing_time_l2788_278852

/-- Proves that Elvis spent 15 minutes writing each song given the conditions of his album production. -/
theorem elvis_song_writing_time :
  let total_songs : ℕ := 10
  let total_studio_time : ℕ := 5 * 60  -- in minutes
  let recording_time_per_song : ℕ := 12
  let editing_time_all_songs : ℕ := 30
  let total_recording_time := total_songs * recording_time_per_song
  let remaining_time := total_studio_time - total_recording_time - editing_time_all_songs
  let writing_time_per_song := remaining_time / total_songs
  writing_time_per_song = 15 := by
    sorry

#check elvis_song_writing_time

end NUMINAMATH_CALUDE_elvis_song_writing_time_l2788_278852


namespace NUMINAMATH_CALUDE_least_clock_equivalent_hour_l2788_278822

def is_clock_equivalent (t : ℕ) : Prop :=
  24 ∣ (t^2 - t)

theorem least_clock_equivalent_hour : 
  ∀ t : ℕ, t > 5 → t < 9 → ¬(is_clock_equivalent t) ∧ is_clock_equivalent 9 :=
by sorry

end NUMINAMATH_CALUDE_least_clock_equivalent_hour_l2788_278822


namespace NUMINAMATH_CALUDE_inequality_reciprocal_l2788_278800

theorem inequality_reciprocal (a b : ℝ) (h : a * b > 0) :
  a > b ↔ 1 / a < 1 / b := by
sorry

end NUMINAMATH_CALUDE_inequality_reciprocal_l2788_278800


namespace NUMINAMATH_CALUDE_distance_ratio_proof_l2788_278892

/-- Proves that the ratio of distances covered at different speeds is 1:1 given specific conditions -/
theorem distance_ratio_proof (total_distance : ℝ) (speed1 speed2 : ℝ) (total_time : ℝ) 
  (h1 : total_distance = 3600)
  (h2 : speed1 = 90)
  (h3 : speed2 = 180)
  (h4 : total_time = 30)
  (h5 : ∃ (d1 d2 : ℝ), d1 + d2 = total_distance ∧ d1 / speed1 + d2 / speed2 = total_time) :
  ∃ (d1 d2 : ℝ), d1 + d2 = total_distance ∧ d1 / speed1 + d2 / speed2 = total_time ∧ d1 = d2 := by
  sorry

#check distance_ratio_proof

end NUMINAMATH_CALUDE_distance_ratio_proof_l2788_278892


namespace NUMINAMATH_CALUDE_expression_evaluation_l2788_278889

theorem expression_evaluation (a : ℝ) (h : a = -6) :
  (1 - a / (a - 3)) / ((a^2 + 3*a) / (a^2 - 9)) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2788_278889


namespace NUMINAMATH_CALUDE_enrollment_difference_l2788_278898

/-- Represents the enrollment of a school --/
structure School where
  name : String
  enrollment : ℕ

/-- Theorem: The positive difference between the maximum and minimum enrollments is 750 --/
theorem enrollment_difference (schools : List School) 
  (h1 : schools.length = 5)
  (h2 : ∃ s ∈ schools, s.name = "Varsity" ∧ s.enrollment = 1680)
  (h3 : ∃ s ∈ schools, s.name = "Northwest" ∧ s.enrollment = 1170)
  (h4 : ∃ s ∈ schools, s.name = "Central" ∧ s.enrollment = 1840)
  (h5 : ∃ s ∈ schools, s.name = "Greenbriar" ∧ s.enrollment = 1090)
  (h6 : ∃ s ∈ schools, s.name = "Eastside" ∧ s.enrollment = 1450) :
  (schools.map (·.enrollment)).maximum?.get! - (schools.map (·.enrollment)).minimum?.get! = 750 := by
  sorry


end NUMINAMATH_CALUDE_enrollment_difference_l2788_278898


namespace NUMINAMATH_CALUDE_multiplication_fraction_problem_l2788_278830

theorem multiplication_fraction_problem : 8 * (1 / 15) * 30 * 3 = 48 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_fraction_problem_l2788_278830


namespace NUMINAMATH_CALUDE_kenya_has_more_peanuts_l2788_278867

/-- The number of peanuts Jose has -/
def jose_peanuts : ℕ := 85

/-- The number of peanuts Kenya has -/
def kenya_peanuts : ℕ := 133

/-- The difference in peanuts between Kenya and Jose -/
def peanut_difference : ℕ := kenya_peanuts - jose_peanuts

theorem kenya_has_more_peanuts : peanut_difference = 48 := by
  sorry

end NUMINAMATH_CALUDE_kenya_has_more_peanuts_l2788_278867


namespace NUMINAMATH_CALUDE_smallest_k_for_triangular_l2788_278879

/-- A positive integer T is triangular if there exists a positive integer n such that T = n * (n + 1) / 2 -/
def IsTriangular (T : ℕ) : Prop :=
  ∃ n : ℕ, n > 0 ∧ T = n * (n + 1) / 2

/-- The smallest positive integer k such that for any triangular number T, 81T + k is also triangular -/
theorem smallest_k_for_triangular : ∃! k : ℕ, 
  k > 0 ∧ 
  (∀ T : ℕ, IsTriangular T → IsTriangular (81 * T + k)) ∧
  (∀ k' : ℕ, k' > 0 → k' < k → 
    ∃ T : ℕ, IsTriangular T ∧ ¬IsTriangular (81 * T + k')) :=
sorry

end NUMINAMATH_CALUDE_smallest_k_for_triangular_l2788_278879


namespace NUMINAMATH_CALUDE_lemons_for_twenty_gallons_l2788_278895

/-- Calculates the number of lemons needed for a given volume of lemonade -/
def lemons_needed (base_lemons : ℕ) (base_gallons : ℕ) (target_gallons : ℕ) : ℕ :=
  let base_ratio := base_lemons / base_gallons
  let base_lemons_needed := base_ratio * target_gallons
  let additional_lemons := target_gallons / 10
  base_lemons_needed + additional_lemons

theorem lemons_for_twenty_gallons :
  lemons_needed 40 50 20 = 18 := by
  sorry

#eval lemons_needed 40 50 20

end NUMINAMATH_CALUDE_lemons_for_twenty_gallons_l2788_278895


namespace NUMINAMATH_CALUDE_f_properties_l2788_278823

def f (x : ℝ) := |x - 2|

theorem f_properties :
  (∀ x, f (x + 2) = f (-x + 2)) ∧ 
  (∀ x y, x < y → x < 2 → y < 2 → f x > f y) ∧
  (∀ x y, x < y → x > 2 → y > 2 → f x < f y) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l2788_278823


namespace NUMINAMATH_CALUDE_peggy_stickers_count_l2788_278839

/-- The number of folders Peggy buys -/
def num_folders : Nat := 3

/-- The number of sheets in each folder -/
def sheets_per_folder : Nat := 10

/-- The number of stickers on each sheet in the red folder -/
def red_stickers_per_sheet : Nat := 3

/-- The number of stickers on each sheet in the green folder -/
def green_stickers_per_sheet : Nat := 2

/-- The number of stickers on each sheet in the blue folder -/
def blue_stickers_per_sheet : Nat := 1

/-- The total number of stickers Peggy uses -/
def total_stickers : Nat := 
  sheets_per_folder * red_stickers_per_sheet +
  sheets_per_folder * green_stickers_per_sheet +
  sheets_per_folder * blue_stickers_per_sheet

theorem peggy_stickers_count : total_stickers = 60 := by
  sorry

end NUMINAMATH_CALUDE_peggy_stickers_count_l2788_278839


namespace NUMINAMATH_CALUDE_geometric_sequence_properties_l2788_278871

-- Define the geometric sequence
def geometric_sequence (n : ℕ) : ℚ :=
  (-1/2) ^ (n - 1)

-- Define the sum of the first n terms
def geometric_sum (n : ℕ) : ℚ :=
  (2/3) * (1 - (-1/2)^n)

-- Theorem statement
theorem geometric_sequence_properties :
  (geometric_sequence 3 = 1/4) ∧
  (∀ n : ℕ, geometric_sequence n = (-1/2)^(n-1)) ∧
  (∀ n : ℕ, geometric_sum n = (2/3) * (1 - (-1/2)^n)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_properties_l2788_278871


namespace NUMINAMATH_CALUDE_anne_travel_distance_l2788_278825

/-- The distance traveled given time and speed -/
def distance (time : ℝ) (speed : ℝ) : ℝ := time * speed

/-- Theorem: Anne's travel distance -/
theorem anne_travel_distance :
  let time : ℝ := 3
  let speed : ℝ := 2
  distance time speed = 6 := by sorry

end NUMINAMATH_CALUDE_anne_travel_distance_l2788_278825


namespace NUMINAMATH_CALUDE_square_root_sum_l2788_278813

theorem square_root_sum (x : ℝ) 
  (h : Real.sqrt (64 - x^2) - Real.sqrt (36 - x^2) = 4) : 
  Real.sqrt (64 - x^2) + Real.sqrt (36 - x^2) = 7 := by
  sorry

end NUMINAMATH_CALUDE_square_root_sum_l2788_278813


namespace NUMINAMATH_CALUDE_quadratic_minimum_l2788_278870

-- Define the quadratic function
def f (x : ℝ) : ℝ := 2 * x^2 - 8 * x + 15

-- State the theorem
theorem quadratic_minimum :
  (∃ (x_min : ℝ), ∀ (x : ℝ), f x ≥ f x_min) ∧
  (∃ (y_min : ℝ), y_min = f 2 ∧ y_min = 7) ∧
  (∀ (x : ℝ), f x ≥ 7) :=
sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l2788_278870


namespace NUMINAMATH_CALUDE_pyramid_volume_in_unit_cube_l2788_278891

/-- The volume of a pyramid within a unit cube, where the pyramid's vertex is at one corner of the cube
    and its base is a triangle formed by the midpoints of three adjacent edges meeting at the opposite corner -/
theorem pyramid_volume_in_unit_cube : ∃ V : ℝ, V = Real.sqrt 3 / 24 :=
  sorry

end NUMINAMATH_CALUDE_pyramid_volume_in_unit_cube_l2788_278891


namespace NUMINAMATH_CALUDE_hyperbola_a_minus_h_l2788_278832

/-- The standard form equation of a hyperbola -/
def is_hyperbola (a b h k x y : ℝ) : Prop :=
  (y - k)^2 / a^2 - (x - h)^2 / b^2 = 1

/-- The equation of an asymptote -/
def is_asymptote (m c x y : ℝ) : Prop :=
  y = m * x + c

theorem hyperbola_a_minus_h (a b h k : ℝ) :
  a > 0 →
  b > 0 →
  is_asymptote 3 4 h k →
  is_asymptote (-3) 6 h k →
  is_hyperbola a b h k 1 9 →
  a - h = 2 * Real.sqrt 3 - 1/3 := by sorry

end NUMINAMATH_CALUDE_hyperbola_a_minus_h_l2788_278832


namespace NUMINAMATH_CALUDE_jordan_rectangle_length_l2788_278827

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.width

theorem jordan_rectangle_length : 
  ∀ (carol jordan : Rectangle),
    carol.length = 5 →
    carol.width = 24 →
    jordan.width = 60 →
    area carol = area jordan →
    jordan.length = 2 := by
  sorry

end NUMINAMATH_CALUDE_jordan_rectangle_length_l2788_278827


namespace NUMINAMATH_CALUDE_f_equals_g_l2788_278861

def f (x : ℝ) : ℝ := x^2 - 2*x - 1
def g (t : ℝ) : ℝ := t^2 - 2*t - 1

theorem f_equals_g : ∀ x : ℝ, f x = g x := by sorry

end NUMINAMATH_CALUDE_f_equals_g_l2788_278861


namespace NUMINAMATH_CALUDE_girls_combined_average_score_l2788_278836

theorem girls_combined_average_score 
  (f1 l1 f2 l2 : ℕ) 
  (h1 : (71 * f1 + 76 * l1) / (f1 + l1) = 74)
  (h2 : (81 * f2 + 90 * l2) / (f2 + l2) = 84)
  (h3 : (71 * f1 + 81 * f2) / (f1 + f2) = 79)
  : (76 * l1 + 90 * l2) / (l1 + l2) = 84 := by
  sorry


end NUMINAMATH_CALUDE_girls_combined_average_score_l2788_278836


namespace NUMINAMATH_CALUDE_right_angled_triangle_exists_l2788_278805

/-- A color type with exactly three colors -/
inductive Color
  | Red
  | Green
  | Blue

/-- A point in the cartesian grid -/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- A coloring function that assigns a color to each grid point -/
def Coloring := GridPoint → Color

/-- Predicate to check if a triangle is right-angled -/
def isRightAngled (p1 p2 p3 : GridPoint) : Prop :=
  (p2.x - p1.x) * (p3.x - p1.x) + (p2.y - p1.y) * (p3.y - p1.y) = 0 ∨
  (p1.x - p2.x) * (p3.x - p2.x) + (p1.y - p2.y) * (p3.y - p2.y) = 0 ∨
  (p1.x - p3.x) * (p2.x - p3.x) + (p1.y - p3.y) * (p2.y - p3.y) = 0

/-- Main theorem: There always exists a right-angled triangle with vertices of different colors -/
theorem right_angled_triangle_exists (f : Coloring)
  (h1 : ∃ p : GridPoint, f p = Color.Red)
  (h2 : ∃ p : GridPoint, f p = Color.Green)
  (h3 : ∃ p : GridPoint, f p = Color.Blue) :
  ∃ p1 p2 p3 : GridPoint,
    isRightAngled p1 p2 p3 ∧
    f p1 ≠ f p2 ∧ f p2 ≠ f p3 ∧ f p1 ≠ f p3 :=
by
  sorry


end NUMINAMATH_CALUDE_right_angled_triangle_exists_l2788_278805


namespace NUMINAMATH_CALUDE_sin_360_minus_alpha_eq_sin_alpha_l2788_278842

theorem sin_360_minus_alpha_eq_sin_alpha (α : ℝ) : 
  Real.sin (2 * Real.pi - α) = Real.sin α := by sorry

end NUMINAMATH_CALUDE_sin_360_minus_alpha_eq_sin_alpha_l2788_278842


namespace NUMINAMATH_CALUDE_train_speed_l2788_278862

/-- Proves that the speed of a train is 72 km/hr, given its length and time to cross a pole -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 180) (h2 : time = 9) :
  (length / time) * 3.6 = 72 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l2788_278862


namespace NUMINAMATH_CALUDE_y_derivative_at_zero_l2788_278843

noncomputable def y (x : ℝ) : ℝ := Real.exp (Real.sin x) * Real.cos (Real.sin x)

theorem y_derivative_at_zero : 
  deriv y 0 = 1 := by sorry

end NUMINAMATH_CALUDE_y_derivative_at_zero_l2788_278843


namespace NUMINAMATH_CALUDE_melinda_paid_759_l2788_278853

-- Define the cost of items
def doughnut_cost : ℚ := 0.45
def coffee_cost : ℚ := (4.91 - 3 * doughnut_cost) / 4

-- Define Melinda's purchase
def melinda_doughnuts : ℕ := 5
def melinda_coffees : ℕ := 6

-- Define Melinda's total cost
def melinda_total_cost : ℚ := melinda_doughnuts * doughnut_cost + melinda_coffees * coffee_cost

-- Theorem to prove
theorem melinda_paid_759 : melinda_total_cost = 7.59 := by
  sorry

end NUMINAMATH_CALUDE_melinda_paid_759_l2788_278853
