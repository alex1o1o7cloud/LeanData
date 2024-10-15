import Mathlib

namespace NUMINAMATH_CALUDE_solution_set_equality_l3907_390749

theorem solution_set_equality (m : ℝ) : 
  (Set.Iio m = {x : ℝ | 2 * x + 1 < 5}) → m ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_equality_l3907_390749


namespace NUMINAMATH_CALUDE_largest_integer_y_f_42_is_integer_largest_y_is_42_l3907_390746

def is_integer (x : ℚ) : Prop := ∃ n : ℤ, x = n

def f (y : ℤ) : ℚ := (y^2 + 3*y + 10) / (y - 4)

theorem largest_integer_y : ∀ y : ℤ, is_integer (f y) → y ≤ 42 :=
by sorry

theorem f_42_is_integer : is_integer (f 42) :=
by sorry

theorem largest_y_is_42 : 
  (∃ y : ℤ, is_integer (f y)) ∧ 
  (∀ y : ℤ, is_integer (f y) → y ≤ 42) ∧ 
  is_integer (f 42) :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_y_f_42_is_integer_largest_y_is_42_l3907_390746


namespace NUMINAMATH_CALUDE_lcm_18_42_l3907_390733

theorem lcm_18_42 : Nat.lcm 18 42 = 126 := by sorry

end NUMINAMATH_CALUDE_lcm_18_42_l3907_390733


namespace NUMINAMATH_CALUDE_multiplication_by_hundred_l3907_390708

theorem multiplication_by_hundred : 38 * 100 = 3800 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_by_hundred_l3907_390708


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l3907_390785

/-- 
An arithmetic sequence starting at 5, with a common difference of 3, 
and ending at 140, contains 46 terms.
-/
theorem arithmetic_sequence_length : 
  ∀ (a : ℕ → ℕ), 
    a 0 = 5 → 
    (∀ n, a (n + 1) = a n + 3) → 
    (∃ m, a m = 140) → 
    (∃ n, a n = 140 ∧ n = 45) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l3907_390785


namespace NUMINAMATH_CALUDE_log_inequality_l3907_390726

theorem log_inequality (x : ℝ) :
  (Real.log x / Real.log (1/2) - Real.sqrt (2 - Real.log x / Real.log 4) + 1 ≤ 0) ↔
  (1 / Real.sqrt 2 ≤ x ∧ x ≤ 16) :=
by sorry

end NUMINAMATH_CALUDE_log_inequality_l3907_390726


namespace NUMINAMATH_CALUDE_min_cost_theorem_l3907_390789

/-- Represents the cost of tickets in rubles -/
def N : ℝ := sorry

/-- The number of southern cities -/
def num_southern_cities : ℕ := 4

/-- The number of northern cities -/
def num_northern_cities : ℕ := 5

/-- The cost of a one-way ticket between any two connected cities -/
def one_way_cost : ℝ := N

/-- The cost of a round-trip ticket between any two connected cities -/
def round_trip_cost : ℝ := 1.6 * N

/-- A route represents a sequence of city visits -/
def Route := List ℕ

/-- Predicate to check if a route is valid according to the problem constraints -/
def is_valid_route (r : Route) : Prop := sorry

/-- The cost of a given route -/
def route_cost (r : Route) : ℝ := sorry

/-- Theorem stating the minimum cost to visit all southern cities and return to the start -/
theorem min_cost_theorem :
  ∀ (r : Route), is_valid_route r →
    route_cost r ≥ 6.4 * N ∧
    ∃ (optimal_route : Route), 
      is_valid_route optimal_route ∧ 
      route_cost optimal_route = 6.4 * N :=
by sorry

end NUMINAMATH_CALUDE_min_cost_theorem_l3907_390789


namespace NUMINAMATH_CALUDE_percentage_reduction_price_increase_l3907_390744

-- Define the original price
def original_price : ℝ := 50

-- Define the final price after two reductions
def final_price : ℝ := 32

-- Define the initial profit per kilogram
def initial_profit : ℝ := 10

-- Define the initial daily sales
def initial_sales : ℝ := 500

-- Define the sales decrease rate
def sales_decrease_rate : ℝ := 20

-- Define the target daily profit
def target_profit : ℝ := 6000

-- Theorem for the percentage reduction
theorem percentage_reduction (x : ℝ) : 
  original_price * (1 - x)^2 = final_price → x = 0.2 := by sorry

-- Theorem for the price increase
theorem price_increase (x : ℝ) :
  (initial_profit + x) * (initial_sales - sales_decrease_rate * x) = target_profit →
  x > 0 →
  ∀ y, y > 0 → 
  (initial_profit + y) * (initial_sales - sales_decrease_rate * y) = target_profit →
  x ≤ y →
  x = 5 := by sorry

end NUMINAMATH_CALUDE_percentage_reduction_price_increase_l3907_390744


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l3907_390794

/-- A point in a 2D plane. -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Defines symmetry with respect to the x-axis. -/
def symmetricXAxis (p p' : Point2D) : Prop :=
  p'.x = p.x ∧ p'.y = -p.y

/-- Theorem: If P(-3, 2) is symmetric to P' with respect to the x-axis,
    then P' has coordinates (-3, -2). -/
theorem symmetric_point_coordinates :
  let P : Point2D := ⟨-3, 2⟩
  let P' : Point2D := ⟨-3, -2⟩
  symmetricXAxis P P' → P' = ⟨-3, -2⟩ :=
by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_coordinates_l3907_390794


namespace NUMINAMATH_CALUDE_midpoint_square_sum_l3907_390730

def A : ℝ × ℝ := (2, 6)
def C : ℝ × ℝ := (4, 1)

theorem midpoint_square_sum (x y : ℝ) : 
  (∀ (p : ℝ × ℝ), p = ((A.1 + x) / 2, (A.2 + y) / 2) → p = C) →
  x^2 + y^2 = 52 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_square_sum_l3907_390730


namespace NUMINAMATH_CALUDE_max_product_with_geometric_mean_l3907_390719

theorem max_product_with_geometric_mean (a b : ℝ) : 
  a > 0 → b > 0 → (3 : ℝ)^((a + b) / 2) = Real.sqrt 3 → ab ≤ (1 / 4 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_max_product_with_geometric_mean_l3907_390719


namespace NUMINAMATH_CALUDE_zibo_barbecue_analysis_l3907_390781

/-- Contingency table data --/
structure ContingencyData where
  male_very_like : ℕ
  male_average : ℕ
  female_very_like : ℕ
  female_average : ℕ

/-- Chi-square test result --/
inductive ChiSquareResult
  | Significant
  | NotSignificant

/-- Distribution of ξ --/
def DistributionXi := List (ℕ × ℚ)

/-- Theorem statement --/
theorem zibo_barbecue_analysis 
  (data : ContingencyData)
  (total_sample : ℕ)
  (chi_square_formula : ContingencyData → ℝ)
  (chi_square_critical : ℝ)
  (calculate_distribution : ContingencyData → DistributionXi)
  (calculate_expectation : DistributionXi → ℚ)
  (h_total : data.male_very_like + data.male_average + data.female_very_like + data.female_average = total_sample)
  (h_female_total : data.female_very_like + data.female_average = 100)
  (h_average_total : data.male_average + data.female_average = 70)
  (h_female_very_like : data.female_very_like = 2 * data.male_average)
  : 
  let chi_square_value := chi_square_formula data
  let result := if chi_square_value < chi_square_critical then ChiSquareResult.NotSignificant else ChiSquareResult.Significant
  let distribution := calculate_distribution data
  let expectation := calculate_expectation distribution
  result = ChiSquareResult.NotSignificant ∧ expectation = 17 / 6 := by
  sorry

end NUMINAMATH_CALUDE_zibo_barbecue_analysis_l3907_390781


namespace NUMINAMATH_CALUDE_nadine_pebbles_l3907_390770

def white_pebbles : ℕ := 20

def red_pebbles : ℕ := white_pebbles / 2

def total_pebbles : ℕ := white_pebbles + red_pebbles

theorem nadine_pebbles : total_pebbles = 30 := by
  sorry

end NUMINAMATH_CALUDE_nadine_pebbles_l3907_390770


namespace NUMINAMATH_CALUDE_building_shadow_length_l3907_390706

/-- Given a flagpole and a building under similar conditions, 
    calculate the length of the shadow cast by the building. -/
theorem building_shadow_length 
  (flagpole_height : ℝ) 
  (flagpole_shadow : ℝ) 
  (building_height : ℝ) 
  (h1 : flagpole_height = 18)
  (h2 : flagpole_shadow = 45)
  (h3 : building_height = 22) :
  (building_height * flagpole_shadow) / flagpole_height = 55 := by
sorry

end NUMINAMATH_CALUDE_building_shadow_length_l3907_390706


namespace NUMINAMATH_CALUDE_root_product_expression_l3907_390724

theorem root_product_expression (p q : ℝ) 
  (α β γ δ : ℂ) 
  (hαβ : α^2 + p*α = 1 ∧ β^2 + p*β = 1) 
  (hγδ : γ^2 + q*γ = -1 ∧ δ^2 + q*δ = -1) : 
  (α - γ) * (β - γ) * (α - δ) * (β - δ) = p^2 - q^2 := by
  sorry

end NUMINAMATH_CALUDE_root_product_expression_l3907_390724


namespace NUMINAMATH_CALUDE_combination_ratio_problem_l3907_390765

theorem combination_ratio_problem (m n : ℕ) : 
  (Nat.choose (n + 1) (m + 1) : ℚ) / (Nat.choose (n + 1) m : ℚ) = 5 / 5 ∧
  (Nat.choose (n + 1) m : ℚ) / (Nat.choose (n + 1) m : ℚ) = 5 / 5 ∧
  (Nat.choose (n + 1) m : ℚ) / (Nat.choose (n + 1) (m - 1) : ℚ) = 5 / 3 →
  m = 3 ∧ n = 6 := by
sorry

end NUMINAMATH_CALUDE_combination_ratio_problem_l3907_390765


namespace NUMINAMATH_CALUDE_fraction_zero_solution_l3907_390703

theorem fraction_zero_solution (x : ℝ) : 
  (x^2 - 4) / (x + 2) = 0 ∧ x + 2 ≠ 0 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_solution_l3907_390703


namespace NUMINAMATH_CALUDE_last_card_in_box_three_l3907_390776

/-- The number of boxes -/
def num_boxes : ℕ := 7

/-- The total number of cards -/
def total_cards : ℕ := 2015

/-- The length of a complete cycle -/
def cycle_length : ℕ := 12

/-- Function to determine the box number for a given card number -/
def box_number (card : ℕ) : ℕ :=
  let cycle_position := card % cycle_length
  if cycle_position ≤ num_boxes
  then cycle_position
  else 2 * num_boxes - cycle_position

/-- Theorem stating that the 2015th card will be placed in box 3 -/
theorem last_card_in_box_three :
  box_number total_cards = 3 := by
  sorry


end NUMINAMATH_CALUDE_last_card_in_box_three_l3907_390776


namespace NUMINAMATH_CALUDE_polygon_exterior_angle_72_l3907_390743

theorem polygon_exterior_angle_72 (n : ℕ) (exterior_angle : ℝ) :
  exterior_angle = 72 →
  (360 : ℝ) / exterior_angle = n →
  n = 5 ∧ (n - 2) * 180 = 540 := by
  sorry

end NUMINAMATH_CALUDE_polygon_exterior_angle_72_l3907_390743


namespace NUMINAMATH_CALUDE_ab_neq_zero_sufficient_not_necessary_for_a_neq_zero_l3907_390700

theorem ab_neq_zero_sufficient_not_necessary_for_a_neq_zero :
  (∀ a b : ℝ, ab ≠ 0 → a ≠ 0) ∧
  (∃ a b : ℝ, a ≠ 0 ∧ ab = 0) :=
by sorry

end NUMINAMATH_CALUDE_ab_neq_zero_sufficient_not_necessary_for_a_neq_zero_l3907_390700


namespace NUMINAMATH_CALUDE_triangle_theorem_l3907_390778

/-- Triangle ABC with sides a, b, c and angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_theorem (t : Triangle) (h1 : t.a * (Real.cos (t.C / 2))^2 + t.c * (Real.cos (t.A / 2))^2 = (3/2) * t.b)
  (h2 : t.B = π/3) (h3 : (1/2) * t.a * t.c * Real.sin t.B = 8 * Real.sqrt 3) :
  (2 * t.b = t.a + t.c) ∧ (t.b = 4 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l3907_390778


namespace NUMINAMATH_CALUDE_smallest_b_in_arithmetic_sequence_l3907_390786

theorem smallest_b_in_arithmetic_sequence (a b c : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →  -- a, b, c are positive
  ∃ (d : ℝ), a = b - d ∧ c = b + d →  -- a, b, c form an arithmetic sequence
  a * b * c = 216 →  -- product is 216
  ∀ (x : ℝ), (∃ (y z : ℝ), 0 < y ∧ 0 < x ∧ 0 < z ∧  -- for any positive x, y, z in arithmetic sequence
    ∃ (e : ℝ), y = x - e ∧ z = x + e ∧  
    y * x * z = 216) →  -- with product 216
  x ≥ b →  -- if x is greater than or equal to b
  b = 6  -- then b must be 6
  := by sorry

end NUMINAMATH_CALUDE_smallest_b_in_arithmetic_sequence_l3907_390786


namespace NUMINAMATH_CALUDE_circle_area_tripled_l3907_390718

theorem circle_area_tripled (r m : ℝ) : 
  (r > 0) → (m > 0) → (π * (r + m)^2 = 3 * π * r^2) → (r = m * (Real.sqrt 3 - 1) / 2) := by
  sorry

end NUMINAMATH_CALUDE_circle_area_tripled_l3907_390718


namespace NUMINAMATH_CALUDE_second_place_prize_l3907_390764

theorem second_place_prize (total_prize : ℕ) (num_winners : ℕ) (first_prize : ℕ) (third_prize : ℕ) (other_prize : ℕ) :
  total_prize = 800 →
  num_winners = 18 →
  first_prize = 200 →
  third_prize = 120 →
  other_prize = 22 →
  (num_winners - 3) * other_prize + first_prize + third_prize + 150 = total_prize :=
by sorry

end NUMINAMATH_CALUDE_second_place_prize_l3907_390764


namespace NUMINAMATH_CALUDE_biology_enrollment_percentage_l3907_390795

theorem biology_enrollment_percentage (total_students : ℕ) (not_enrolled : ℕ) : 
  total_students = 880 →
  not_enrolled = 528 →
  (((total_students - not_enrolled) : ℚ) / total_students) * 100 = 40 := by
sorry

end NUMINAMATH_CALUDE_biology_enrollment_percentage_l3907_390795


namespace NUMINAMATH_CALUDE_algebraic_expression_solution_l3907_390748

theorem algebraic_expression_solution (m : ℚ) : 
  (5 * (2 - 1) + 3 * m * 2 = -7) → 
  (∃ x : ℚ, 5 * (x - 1) + 3 * m * x = -1 ∧ x = -4) :=
by sorry

end NUMINAMATH_CALUDE_algebraic_expression_solution_l3907_390748


namespace NUMINAMATH_CALUDE_common_difference_is_two_l3907_390716

/-- An arithmetic sequence with specified terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m
  fifth_term : a 5 = 6
  third_term : a 3 = 2

/-- The common difference of an arithmetic sequence -/
def commonDifference (seq : ArithmeticSequence) : ℝ :=
  seq.a 2 - seq.a 1

theorem common_difference_is_two (seq : ArithmeticSequence) :
  commonDifference seq = 2 := by
  sorry

end NUMINAMATH_CALUDE_common_difference_is_two_l3907_390716


namespace NUMINAMATH_CALUDE_factorial_ratio_l3907_390727

theorem factorial_ratio : Nat.factorial 45 / Nat.factorial 42 = 85140 := by sorry

end NUMINAMATH_CALUDE_factorial_ratio_l3907_390727


namespace NUMINAMATH_CALUDE_max_yellow_balls_l3907_390752

/-- Represents the total number of balls -/
def n : ℕ := 91

/-- Represents the number of yellow balls in the first 70 picked -/
def initial_yellow : ℕ := 63

/-- Represents the total number of balls initially picked -/
def initial_total : ℕ := 70

/-- Represents the number of yellow balls in each subsequent batch of 7 -/
def batch_yellow : ℕ := 5

/-- Represents the total number of balls in each subsequent batch -/
def batch_total : ℕ := 7

/-- The minimum percentage of yellow balls required -/
def min_percentage : ℚ := 85 / 100

theorem max_yellow_balls :
  n = initial_total + batch_total * ((n - initial_total) / batch_total) ∧
  (initial_yellow + batch_yellow * ((n - initial_total) / batch_total)) / n ≥ min_percentage ∧
  ∀ m : ℕ, m > n →
    (initial_yellow + batch_yellow * ((m - initial_total) / batch_total)) / m < min_percentage :=
by sorry

end NUMINAMATH_CALUDE_max_yellow_balls_l3907_390752


namespace NUMINAMATH_CALUDE_intersection_point_of_lines_l3907_390738

theorem intersection_point_of_lines (x y : ℝ) :
  y = x ∧ y = -x + 2 → (x = 1 ∧ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_of_lines_l3907_390738


namespace NUMINAMATH_CALUDE_scientific_notation_of_161000_l3907_390735

/-- The scientific notation representation of 161,000 -/
theorem scientific_notation_of_161000 : 161000 = 1.61 * (10 ^ 5) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_161000_l3907_390735


namespace NUMINAMATH_CALUDE_tribe_leadership_arrangements_l3907_390721

def tribe_size : ℕ := 15
def num_chiefs : ℕ := 1
def num_supporting_chiefs : ℕ := 2
def inferior_officers_per_chief : ℕ := 3

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

theorem tribe_leadership_arrangements :
  tribe_size * (tribe_size - 1) * (tribe_size - 2) *
  (choose (tribe_size - 3) inferior_officers_per_chief) *
  (choose (tribe_size - 3 - inferior_officers_per_chief) inferior_officers_per_chief) = 3243240 :=
by sorry

end NUMINAMATH_CALUDE_tribe_leadership_arrangements_l3907_390721


namespace NUMINAMATH_CALUDE_square_sum_equality_l3907_390725

theorem square_sum_equality (x y P Q : ℝ) :
  x^2 + y^2 = (x + y)^2 + P ∧ x^2 + y^2 = (x - y)^2 + Q →
  P = -2*x*y ∧ Q = 2*x*y := by
sorry

end NUMINAMATH_CALUDE_square_sum_equality_l3907_390725


namespace NUMINAMATH_CALUDE_sqrt_2023_bound_l3907_390763

theorem sqrt_2023_bound (n : ℤ) : n < Real.sqrt 2023 ∧ Real.sqrt 2023 < n + 1 → n = 44 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_2023_bound_l3907_390763


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3907_390753

-- Define the quadratic function
def f (x : ℝ) := x^2 - 6*x + 8

-- Define the solution set
def solution_set : Set ℝ := {x | x < 2 ∨ x > 4}

-- Theorem statement
theorem quadratic_inequality_solution :
  {x : ℝ | f x > 0} = solution_set :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3907_390753


namespace NUMINAMATH_CALUDE_min_tiles_for_floor_l3907_390798

/-- Calculates the minimum number of rectangular tiles needed to cover a rectangular floor -/
def min_tiles_needed (tile_length inch_per_foot tile_width floor_length floor_width : ℕ) : ℕ :=
  let floor_area := (floor_length * inch_per_foot) * (floor_width * inch_per_foot)
  let tile_area := tile_length * tile_width
  (floor_area + tile_area - 1) / tile_area

/-- The minimum number of 5x6 inch tiles needed to cover a 3x4 foot floor is 58 -/
theorem min_tiles_for_floor :
  min_tiles_needed 5 12 6 3 4 = 58 := by
  sorry

end NUMINAMATH_CALUDE_min_tiles_for_floor_l3907_390798


namespace NUMINAMATH_CALUDE_gcd_of_squares_l3907_390742

theorem gcd_of_squares : Nat.gcd (111^2 + 222^2 + 333^2) (110^2 + 221^2 + 334^2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_squares_l3907_390742


namespace NUMINAMATH_CALUDE_sugar_solution_volume_l3907_390773

/-- Given a sugar solution, prove that the initial volume was 3 liters -/
theorem sugar_solution_volume (V : ℝ) : 
  V > 0 → -- Initial volume is positive
  (0.4 * V) / (V + 1) = 0.30000000000000004 → -- New concentration after adding 1 liter of water
  V = 3 := by
sorry

end NUMINAMATH_CALUDE_sugar_solution_volume_l3907_390773


namespace NUMINAMATH_CALUDE_paco_initial_salty_cookies_l3907_390702

/-- Represents the number of cookies Paco has --/
structure CookieCount where
  salty : ℕ
  sweet : ℕ

/-- The problem of determining Paco's initial salty cookie count --/
theorem paco_initial_salty_cookies 
  (initial : CookieCount) 
  (eaten : CookieCount) 
  (final : CookieCount) : 
  (initial.sweet = 17) →
  (eaten.sweet = 14) →
  (eaten.salty = 9) →
  (final.salty = 17) →
  (initial.salty = final.salty + eaten.salty) →
  (initial.salty = 26) := by
sorry


end NUMINAMATH_CALUDE_paco_initial_salty_cookies_l3907_390702


namespace NUMINAMATH_CALUDE_jelly_bean_count_l3907_390717

/-- The number of jelly beans in jar X -/
def jarX (total : ℕ) (y : ℕ) : ℕ := 3 * y - 400

/-- The number of jelly beans in jar Y -/
def jarY (total : ℕ) (x : ℕ) : ℕ := total - x

theorem jelly_bean_count (total : ℕ) (h : total = 1200) :
  ∃ y : ℕ, jarX total y + jarY total (jarX total y) = total ∧ jarX total y = 800 := by
sorry

end NUMINAMATH_CALUDE_jelly_bean_count_l3907_390717


namespace NUMINAMATH_CALUDE_line_equations_l3907_390731

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point (x, y) lies on a line -/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y + l.c = 0

/-- Check if two lines are perpendicular -/
def Line.perpendicular (l₁ l₂ : Line) : Prop :=
  l₁.a * l₂.a + l₁.b * l₂.b = 0

/-- Check if a line has equal intercepts on both axes -/
def Line.equalIntercepts (l : Line) : Prop :=
  l.a = l.b ∧ l.a ≠ 0

theorem line_equations (l₁ : Line) :
  (l₁.contains 2 3) →
  (∃ l₂ : Line, l₂.a = 1 ∧ l₂.b = 2 ∧ l₂.c = 4 ∧ l₁.perpendicular l₂) →
  (l₁.a = 2 ∧ l₁.b = -1 ∧ l₁.c = -1) ∨
  (l₁.equalIntercepts → (l₁.a = 1 ∧ l₁.b = 1 ∧ l₁.c = -5) ∨ (l₁.a = 3 ∧ l₁.b = -2 ∧ l₁.c = 0)) :=
by sorry

end NUMINAMATH_CALUDE_line_equations_l3907_390731


namespace NUMINAMATH_CALUDE_f_pi_half_value_l3907_390755

theorem f_pi_half_value : 
  let f : ℝ → ℝ := fun x ↦ x * Real.sin x + Real.cos x
  f (Real.pi / 2) = Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_f_pi_half_value_l3907_390755


namespace NUMINAMATH_CALUDE_x_coord_difference_at_y_10_l3907_390758

/-- Represents a line in 2D space -/
structure Line where
  slope : ℚ
  intercept : ℚ

/-- Calculates the x-coordinate for a given y-coordinate on a line -/
def xCoordAtY (l : Line) (y : ℚ) : ℚ :=
  (y - l.intercept) / l.slope

/-- Creates a line from two points -/
def lineFromPoints (x1 y1 x2 y2 : ℚ) : Line where
  slope := (y2 - y1) / (x2 - x1)
  intercept := y1 - (y2 - y1) / (x2 - x1) * x1

theorem x_coord_difference_at_y_10 : 
  let p := lineFromPoints 0 3 4 0
  let q := lineFromPoints 0 1 8 0
  let xp := xCoordAtY p 10
  let xq := xCoordAtY q 10
  |xp - xq| = 188 / 3 := by
    sorry

end NUMINAMATH_CALUDE_x_coord_difference_at_y_10_l3907_390758


namespace NUMINAMATH_CALUDE_maximal_cross_section_area_l3907_390775

/-- A triangular prism with vertical edges parallel to the z-axis -/
structure TriangularPrism where
  base : Set (ℝ × ℝ)
  height : ℝ → ℝ

/-- The cross-section of the prism is an equilateral triangle with side length 8 -/
def equilateralBase (p : TriangularPrism) : Prop :=
  ∃ (A B C : ℝ × ℝ),
    A ∈ p.base ∧ B ∈ p.base ∧ C ∈ p.base ∧
    dist A B = 8 ∧ dist B C = 8 ∧ dist C A = 8

/-- The plane that intersects the prism -/
def intersectingPlane (x y z : ℝ) : Prop :=
  3 * x - 5 * y + 2 * z = 30

/-- The cross-section formed by the intersection of the prism and the plane -/
def crossSection (p : TriangularPrism) : Set (ℝ × ℝ × ℝ) :=
  {(x, y, z) | (x, y) ∈ p.base ∧ z = p.height x ∧ intersectingPlane x y z}

/-- The area of the cross-section -/
noncomputable def crossSectionArea (p : TriangularPrism) : ℝ :=
  sorry

/-- The main theorem stating that the maximal area of the cross-section is 92 -/
theorem maximal_cross_section_area (p : TriangularPrism) 
  (h : equilateralBase p) : 
  crossSectionArea p ≤ 92 ∧ ∃ (p' : TriangularPrism), equilateralBase p' ∧ crossSectionArea p' = 92 :=
sorry

end NUMINAMATH_CALUDE_maximal_cross_section_area_l3907_390775


namespace NUMINAMATH_CALUDE_brendas_age_is_real_l3907_390799

/-- Represents the ages of individuals --/
structure Ages where
  addison : ℝ
  brenda : ℝ
  carlos : ℝ
  janet : ℝ

/-- The conditions given in the problem --/
def age_conditions (ages : Ages) : Prop :=
  ages.addison = 4 * ages.brenda ∧
  ages.carlos = 2 * ages.brenda ∧
  ages.addison = ages.janet

/-- Theorem stating that Brenda's age is a positive real number --/
theorem brendas_age_is_real (ages : Ages) (h : age_conditions ages) :
  ∃ (B : ℝ), B > 0 ∧ ages.brenda = B :=
sorry

end NUMINAMATH_CALUDE_brendas_age_is_real_l3907_390799


namespace NUMINAMATH_CALUDE_division_of_fractions_l3907_390720

theorem division_of_fractions : (5 : ℚ) / 6 / ((2 : ℚ) / 3) = (5 : ℚ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_division_of_fractions_l3907_390720


namespace NUMINAMATH_CALUDE_ryan_solution_unique_l3907_390771

/-- Represents the solution to Ryan's grocery purchase --/
structure GrocerySolution where
  corn : ℝ
  beans : ℝ
  rice : ℝ

/-- Checks if a given solution satisfies all the problem conditions --/
def is_valid_solution (s : GrocerySolution) : Prop :=
  s.corn + s.beans + s.rice = 30 ∧
  1.20 * s.corn + 0.60 * s.beans + 0.80 * s.rice = 24 ∧
  s.beans = s.rice

/-- The unique solution to the problem --/
def ryan_solution : GrocerySolution :=
  { corn := 6, beans := 12, rice := 12 }

/-- Theorem stating that ryan_solution is the only valid solution --/
theorem ryan_solution_unique :
  is_valid_solution ryan_solution ∧
  ∀ s : GrocerySolution, is_valid_solution s → s = ryan_solution :=
sorry

end NUMINAMATH_CALUDE_ryan_solution_unique_l3907_390771


namespace NUMINAMATH_CALUDE_inscribed_square_area_l3907_390767

/-- A square inscribed in a semicircle with radius 1 -/
structure InscribedSquare where
  /-- The side length of the square -/
  side : ℝ
  /-- One side of the square is flush with the diameter of the semicircle -/
  flush_with_diameter : True
  /-- The square is inscribed in the semicircle -/
  inscribed : side^2 + (side/2)^2 = 1

/-- The area of an inscribed square is 4/5 -/
theorem inscribed_square_area (s : InscribedSquare) : s.side^2 = 4/5 := by
  sorry

#check inscribed_square_area

end NUMINAMATH_CALUDE_inscribed_square_area_l3907_390767


namespace NUMINAMATH_CALUDE_cos_shift_odd_condition_l3907_390734

open Real

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem cos_shift_odd_condition (φ : ℝ) :
  (φ = π / 2 → is_odd_function (λ x => cos (x + φ))) ∧
  (∃ φ', φ' ≠ π / 2 ∧ is_odd_function (λ x => cos (x + φ'))) :=
sorry

end NUMINAMATH_CALUDE_cos_shift_odd_condition_l3907_390734


namespace NUMINAMATH_CALUDE_remainder_11_power_1995_mod_5_l3907_390797

theorem remainder_11_power_1995_mod_5 : 11^1995 % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_11_power_1995_mod_5_l3907_390797


namespace NUMINAMATH_CALUDE_functional_equation_implies_ge_l3907_390740

/-- A function f: ℝ⁺ → ℝ⁺ satisfying f(f(x)) + x = f(2x) for all x > 0 -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x > 0, f x > 0 ∧ f (f x) + x = f (2 * x)

/-- Theorem: If f satisfies the functional equation, then f(x) ≥ x for all x > 0 -/
theorem functional_equation_implies_ge (f : ℝ → ℝ) (h : FunctionalEquation f) :
  ∀ x > 0, f x ≥ x := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_implies_ge_l3907_390740


namespace NUMINAMATH_CALUDE_exactly_one_survives_l3907_390761

theorem exactly_one_survives (p q : ℝ) (hp : 0 ≤ p ∧ p ≤ 1) (hq : 0 ≤ q ∧ q ≤ 1) :
  (p * (1 - q)) + ((1 - p) * q) = p + q - p * q := by
  sorry

end NUMINAMATH_CALUDE_exactly_one_survives_l3907_390761


namespace NUMINAMATH_CALUDE_cindy_calculation_l3907_390768

def original_number : ℝ := (23 * 5) + 7

theorem cindy_calculation : 
  Int.floor ((original_number + 7) / 5) = 26 := by
  sorry

end NUMINAMATH_CALUDE_cindy_calculation_l3907_390768


namespace NUMINAMATH_CALUDE_fourth_term_of_geometric_sequence_l3907_390741

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

/-- Given a geometric sequence satisfying certain conditions, its 4th term equals 8. -/
theorem fourth_term_of_geometric_sequence (a : ℕ → ℝ) 
    (h_geo : IsGeometricSequence a) 
    (h_sum : a 6 + a 2 = 34) 
    (h_diff : a 6 - a 2 = 30) : 
  a 4 = 8 := by
  sorry


end NUMINAMATH_CALUDE_fourth_term_of_geometric_sequence_l3907_390741


namespace NUMINAMATH_CALUDE_original_amount_calculation_l3907_390715

theorem original_amount_calculation (total : ℚ) : 
  (3/4 : ℚ) * total - (1/5 : ℚ) * total = 132 → total = 240 := by
  sorry

end NUMINAMATH_CALUDE_original_amount_calculation_l3907_390715


namespace NUMINAMATH_CALUDE_perpendicular_bisector_b_value_l3907_390701

/-- The line x + y = b is a perpendicular bisector of the line segment from (2, 5) to (8, 11) -/
def is_perpendicular_bisector (b : ℝ) : Prop :=
  let midpoint := ((2 + 8) / 2, (5 + 11) / 2)
  midpoint.1 + midpoint.2 = b

/-- The value of b for which x + y = b is a perpendicular bisector of the line segment from (2, 5) to (8, 11) is 13 -/
theorem perpendicular_bisector_b_value :
  ∃ b : ℝ, is_perpendicular_bisector b ∧ b = 13 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_b_value_l3907_390701


namespace NUMINAMATH_CALUDE_complex_sum_problem_l3907_390705

theorem complex_sum_problem (x y u v w z : ℝ) : 
  y = 5 →
  w = -x - u →
  Complex.I * (x + y + u + v + w + z) = 4 * Complex.I →
  v + z = -1 := by
sorry

end NUMINAMATH_CALUDE_complex_sum_problem_l3907_390705


namespace NUMINAMATH_CALUDE_find_a_l3907_390769

theorem find_a (x y a : ℤ) 
  (eq1 : 3 * x + y = 40)
  (eq2 : a * x - y = 20)
  (eq3 : 3 * y^2 = 48) :
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_a_l3907_390769


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_2_to_20_l3907_390736

def arithmetic_sequence_sum (a : ℕ) (l : ℕ) (d : ℕ) : ℕ :=
  let n : ℕ := (l - a) / d + 1
  n * (a + l) / 2

theorem arithmetic_sequence_sum_2_to_20 :
  arithmetic_sequence_sum 2 20 2 = 110 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_2_to_20_l3907_390736


namespace NUMINAMATH_CALUDE_largest_guaranteed_divisor_l3907_390714

def die_numbers : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

def is_valid_roll (roll : Finset ℕ) : Prop :=
  roll ⊆ die_numbers ∧ roll.card = 7

def roll_product (roll : Finset ℕ) : ℕ :=
  roll.prod id

theorem largest_guaranteed_divisor :
  ∀ roll : Finset ℕ, is_valid_roll roll →
    ∃ m : ℕ, m = 192 ∧ 
      (∀ n : ℕ, n > 192 → ¬(∀ r : Finset ℕ, is_valid_roll r → n ∣ roll_product r)) ∧
      (192 ∣ roll_product roll) :=
by sorry

end NUMINAMATH_CALUDE_largest_guaranteed_divisor_l3907_390714


namespace NUMINAMATH_CALUDE_triangle_side_length_l3907_390777

theorem triangle_side_length (a b c : ℝ) (A B C : Real) :
  a = 5 →
  b = 7 →
  B = π / 3 →  -- 60° in radians
  b^2 = a^2 + c^2 - 2*a*c*(Real.cos B) →
  c = 8 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3907_390777


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3907_390779

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (3 + Real.sqrt x) = 4 → x = 169 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3907_390779


namespace NUMINAMATH_CALUDE_elizabeth_ate_four_bananas_l3907_390782

/-- The number of bananas Elizabeth ate -/
def bananas_eaten (initial : ℕ) (remaining : ℕ) : ℕ :=
  initial - remaining

/-- Proof that Elizabeth ate 4 bananas -/
theorem elizabeth_ate_four_bananas :
  let initial := 12
  let remaining := 8
  bananas_eaten initial remaining = 4 := by
  sorry

end NUMINAMATH_CALUDE_elizabeth_ate_four_bananas_l3907_390782


namespace NUMINAMATH_CALUDE_tutor_schedule_lcm_l3907_390788

theorem tutor_schedule_lcm : Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 8 (Nat.lcm 10 11))) = 1320 := by
  sorry

end NUMINAMATH_CALUDE_tutor_schedule_lcm_l3907_390788


namespace NUMINAMATH_CALUDE_ellipse_sum_coordinates_and_axes_l3907_390750

/-- Represents an ellipse on a coordinate plane -/
structure Ellipse where
  center : ℝ × ℝ
  semiMajorAxis : ℝ
  semiMinorAxis : ℝ

/-- Theorem: For the given ellipse, h + k + a + b = 9 -/
theorem ellipse_sum_coordinates_and_axes (e : Ellipse) 
  (h_center : e.center = (1, -3))
  (h_major : e.semiMajorAxis = 7)
  (h_minor : e.semiMinorAxis = 4) :
  e.center.1 + e.center.2 + e.semiMajorAxis + e.semiMinorAxis = 9 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_sum_coordinates_and_axes_l3907_390750


namespace NUMINAMATH_CALUDE_hyperbola_through_C_l3907_390723

/-- Given a point A on the parabola y = x^2 and a point B such that OB is perpendicular to OA,
    prove that the point C formed by the rectangle AOBC lies on the hyperbola y = -2/x -/
theorem hyperbola_through_C (A B C : ℝ × ℝ) : 
  A.1 = -1/2 ∧ A.2 = 1/4 ∧                          -- A is (-1/2, 1/4)
  A.2 = A.1^2 ∧                                     -- A is on the parabola y = x^2
  B.1 = 2 ∧ B.2 = 4 ∧                               -- B is (2, 4)
  (B.2 - 0) / (B.1 - 0) = -(A.2 - 0) / (A.1 - 0) ∧  -- OB ⟂ OA
  C.1 = A.1 ∧ C.2 = B.2                             -- C forms rectangle AOBC
  →
  C.2 = -2 / C.1                                    -- C is on the hyperbola y = -2/x
:= by sorry

end NUMINAMATH_CALUDE_hyperbola_through_C_l3907_390723


namespace NUMINAMATH_CALUDE_ball_probability_l3907_390729

theorem ball_probability (total : Nat) (white green yellow red purple blue black : Nat)
  (h_total : total = 200)
  (h_white : white = 50)
  (h_green : green = 40)
  (h_yellow : yellow = 20)
  (h_red : red = 30)
  (h_purple : purple = 30)
  (h_blue : blue = 10)
  (h_black : black = 20)
  (h_sum : total = white + green + yellow + red + purple + blue + black) :
  (white + green + yellow + blue : ℚ) / total = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_ball_probability_l3907_390729


namespace NUMINAMATH_CALUDE_sum_of_cubes_of_roots_l3907_390780

theorem sum_of_cubes_of_roots (a b : ℝ) (α β : ℝ) : 
  (α^2 + a*α + b = 0) → (β^2 + a*β + b = 0) → α^3 + β^3 = -(a^3 - 3*a*b) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_of_roots_l3907_390780


namespace NUMINAMATH_CALUDE_regular_polygon_perimeter_l3907_390707

/-- Given a regular polygon with central angle 45° and side length 5, its perimeter is 40. -/
theorem regular_polygon_perimeter (central_angle : ℝ) (side_length : ℝ) :
  central_angle = 45 →
  side_length = 5 →
  (360 / central_angle) * side_length = 40 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_perimeter_l3907_390707


namespace NUMINAMATH_CALUDE_tractors_count_l3907_390745

/-- Represents the number of tractors initially ploughing the field -/
def T : ℕ := sorry

/-- The area of the field in hectares -/
def field_area : ℕ := sorry

/-- Each tractor ploughs this many hectares per day -/
def hectares_per_tractor_per_day : ℕ := 120

/-- The number of days it takes all tractors to plough the field -/
def days_all_tractors : ℕ := 4

/-- The number of tractors remaining after two are removed -/
def remaining_tractors : ℕ := 4

/-- The number of days it takes the remaining tractors to plough the field -/
def days_remaining_tractors : ℕ := 5

theorem tractors_count :
  (T * hectares_per_tractor_per_day * days_all_tractors = field_area) ∧
  (remaining_tractors * hectares_per_tractor_per_day * days_remaining_tractors = field_area) ∧
  (T = remaining_tractors + 2) →
  T = 10 := by sorry

end NUMINAMATH_CALUDE_tractors_count_l3907_390745


namespace NUMINAMATH_CALUDE_brother_ages_l3907_390796

theorem brother_ages (B E : ℕ) (h1 : B = E + 4) (h2 : B + E = 28) : E = 12 := by
  sorry

end NUMINAMATH_CALUDE_brother_ages_l3907_390796


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3907_390759

-- Define sets A and B
def A : Set ℝ := {x | x - 1 ≤ 0}
def B : Set ℝ := {x | x^2 - 4*x ≤ 0}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {x | 0 ≤ x ∧ x ≤ 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3907_390759


namespace NUMINAMATH_CALUDE_abc_perfect_cube_l3907_390737

theorem abc_perfect_cube (a b c : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) 
  (h4 : ∃ (n : ℤ), (a : ℚ) / b + (b : ℚ) / c + (c : ℚ) / a = n) :
  ∃ (k : ℤ), a * b * c = k^3 := by
sorry

end NUMINAMATH_CALUDE_abc_perfect_cube_l3907_390737


namespace NUMINAMATH_CALUDE_prime_square_sum_l3907_390751

theorem prime_square_sum (p q r : ℕ) : 
  Prime p ∧ Prime q ∧ Prime r ∧ ∃ (n : ℕ), p^q + p^r = n^2 ↔ 
  ((p = 2 ∧ q = 2 ∧ r = 5) ∨ 
   (p = 2 ∧ q = 5 ∧ r = 2) ∨ 
   (p = 3 ∧ q = 2 ∧ r = 3) ∨ 
   (p = 3 ∧ q = 3 ∧ r = 2) ∨ 
   (p = 2 ∧ q = r ∧ q ≥ 3 ∧ Prime q)) :=
by sorry

end NUMINAMATH_CALUDE_prime_square_sum_l3907_390751


namespace NUMINAMATH_CALUDE_sale_price_calculation_l3907_390762

def original_price : ℝ := 100
def discount_percentage : ℝ := 25

theorem sale_price_calculation :
  let discount_amount := (discount_percentage / 100) * original_price
  let sale_price := original_price - discount_amount
  sale_price = 75 := by sorry

end NUMINAMATH_CALUDE_sale_price_calculation_l3907_390762


namespace NUMINAMATH_CALUDE_probability_two_successes_four_trials_l3907_390722

/-- The binomial coefficient (n choose k) -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The probability of exactly k successes in n trials with probability p of success per trial -/
def binomialProbability (n k : ℕ) (p : ℝ) : ℝ :=
  (binomial n k : ℝ) * p^k * (1 - p)^(n - k)

/-- The theorem stating the probability of 2 successes in 4 trials with 0.3 probability of success -/
theorem probability_two_successes_four_trials :
  binomialProbability 4 2 0.3 = 0.2646 := by sorry

end NUMINAMATH_CALUDE_probability_two_successes_four_trials_l3907_390722


namespace NUMINAMATH_CALUDE_cylinder_height_comparison_l3907_390760

-- Define the cylinders
structure Cylinder where
  radius : ℝ
  height : ℝ

-- Define the theorem
theorem cylinder_height_comparison (c1 c2 : Cylinder) 
  (h_volume : π * c1.radius^2 * c1.height = π * c2.radius^2 * c2.height)
  (h_radius : c2.radius = 1.2 * c1.radius) :
  c1.height = 1.44 * c2.height := by
  sorry

end NUMINAMATH_CALUDE_cylinder_height_comparison_l3907_390760


namespace NUMINAMATH_CALUDE_z_squared_and_modulus_l3907_390713

-- Define the complex number z
def z : ℂ := 5 + 3 * Complex.I

-- Theorem statement
theorem z_squared_and_modulus :
  z ^ 2 = 16 + 30 * Complex.I ∧ Complex.abs (z ^ 2) = 34 := by
  sorry

end NUMINAMATH_CALUDE_z_squared_and_modulus_l3907_390713


namespace NUMINAMATH_CALUDE_rectangle_areas_l3907_390757

theorem rectangle_areas (square_area : ℝ) (ratio1_width ratio1_length ratio2_width ratio2_length : ℕ) :
  square_area = 98 →
  ratio1_width = 2 →
  ratio1_length = 3 →
  ratio2_width = 3 →
  ratio2_length = 8 →
  ∃ (rect1_perim rect2_perim : ℝ),
    4 * Real.sqrt square_area = rect1_perim + rect2_perim ∧
    (rect1_perim * ratio1_width * rect1_perim * ratio1_length) / ((ratio1_width + ratio1_length) ^ 2) =
    (rect2_perim * ratio2_width * rect2_perim * ratio2_length) / ((ratio2_width + ratio2_length) ^ 2) →
  (rect1_perim * ratio1_width * rect1_perim * ratio1_length) / ((ratio1_width + ratio1_length) ^ 2) = 64 / 3 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_areas_l3907_390757


namespace NUMINAMATH_CALUDE_f_monotone_condition_l3907_390756

-- Define the piecewise function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 - 2 else Real.log (|x - m|)

-- Define monotonically increasing property
def MonotonicallyIncreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x ≤ f y

-- State the theorem
theorem f_monotone_condition (m : ℝ) :
  (∀ x y, 0 ≤ x ∧ x < y → f m x ≤ f m y) ↔ m ≤ 9/10 :=
sorry

end NUMINAMATH_CALUDE_f_monotone_condition_l3907_390756


namespace NUMINAMATH_CALUDE_rectangle_area_l3907_390739

/-- Given a rectangle where the length is four times the width and the perimeter is 200 cm,
    prove that its area is 1600 square centimeters. -/
theorem rectangle_area (w : ℝ) (h1 : w > 0) : 
  let l := 4 * w
  2 * l + 2 * w = 200 →
  l * w = 1600 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_l3907_390739


namespace NUMINAMATH_CALUDE_perimeter_inscribable_equivalence_l3907_390790

/-- Triangle represented by its side lengths -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a line segment intersecting two sides of a triangle -/
structure IntersectingLine (T : Triangle) where
  A' : ℝ  -- Distance from A to A' on AC
  B' : ℝ  -- Distance from B to B' on BC

/-- Condition for the perimeter of the inner triangle -/
def perimeterCondition (T : Triangle) (L : IntersectingLine T) : Prop :=
  L.A' + L.B' + (T.c - L.A' - L.B') = T.a + T.b - T.c

/-- Condition for the quadrilateral to be inscribable -/
def inscribableCondition (T : Triangle) (L : IntersectingLine T) : Prop :=
  T.c + (T.a + T.b - T.c - (L.A' + L.B')) = (T.a - L.A') + (T.b - L.B')

theorem perimeter_inscribable_equivalence (T : Triangle) (L : IntersectingLine T) :
  perimeterCondition T L ↔ inscribableCondition T L := by sorry

end NUMINAMATH_CALUDE_perimeter_inscribable_equivalence_l3907_390790


namespace NUMINAMATH_CALUDE_balloon_arrangements_l3907_390793

/-- The number of distinct arrangements of letters in a word -/
def distinctArrangements (totalLetters : Nat) (repeatedLetters : List (Nat)) : Nat :=
  Nat.factorial totalLetters / (repeatedLetters.map Nat.factorial).prod

theorem balloon_arrangements :
  distinctArrangements 7 [2, 2] = 1260 := by
  sorry

end NUMINAMATH_CALUDE_balloon_arrangements_l3907_390793


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l3907_390783

theorem cubic_equation_roots (p q : ℝ) : 
  (∃ (r₁ r₂ r₃ : ℕ+), r₁ ≠ r₂ ∧ r₂ ≠ r₃ ∧ r₁ ≠ r₃ ∧
    (∀ (x : ℝ), x^3 - 6*x^2 + p*x - q = 0 ↔ (x = r₁ ∨ x = r₂ ∨ x = r₃))) →
  p + q = 17 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l3907_390783


namespace NUMINAMATH_CALUDE_harry_monday_speed_l3907_390704

/-- Harry's marathon running speeds throughout the week -/
def harry_speeds (monday_speed : ℝ) : Fin 5 → ℝ
  | 0 => monday_speed  -- Monday
  | 1 => 1.5 * monday_speed  -- Tuesday
  | 2 => 1.5 * monday_speed  -- Wednesday
  | 3 => 1.5 * monday_speed  -- Thursday
  | 4 => 1.6 * 1.5 * monday_speed  -- Friday

theorem harry_monday_speed :
  ∃ (monday_speed : ℝ), 
    (harry_speeds monday_speed 4 = 24) ∧ 
    (monday_speed = 10) := by
  sorry

end NUMINAMATH_CALUDE_harry_monday_speed_l3907_390704


namespace NUMINAMATH_CALUDE_officer_average_salary_l3907_390774

/-- Proves that the average salary of officers is 420 Rs/month given the specified conditions -/
theorem officer_average_salary
  (total_employees : ℕ)
  (officers : ℕ)
  (non_officers : ℕ)
  (average_salary : ℚ)
  (non_officer_salary : ℚ)
  (h1 : total_employees = officers + non_officers)
  (h2 : total_employees = 465)
  (h3 : officers = 15)
  (h4 : non_officers = 450)
  (h5 : average_salary = 120)
  (h6 : non_officer_salary = 110) :
  (total_employees * average_salary - non_officers * non_officer_salary) / officers = 420 :=
by sorry

end NUMINAMATH_CALUDE_officer_average_salary_l3907_390774


namespace NUMINAMATH_CALUDE_cosine_of_specific_line_l3907_390712

/-- A line in 2D space represented by parametric equations -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- The inclination angle of a line -/
def inclinationAngle (l : ParametricLine) : ℝ := sorry

/-- Cosine of the inclination angle of a line -/
def cosInclinationAngle (l : ParametricLine) : ℝ := sorry

theorem cosine_of_specific_line :
  let l : ParametricLine := {
    x := λ t => 1 + 3 * t,
    y := λ t => 2 - 4 * t
  }
  cosInclinationAngle l = -3/5 := by sorry

end NUMINAMATH_CALUDE_cosine_of_specific_line_l3907_390712


namespace NUMINAMATH_CALUDE_tissue_paper_count_l3907_390784

theorem tissue_paper_count (remaining : ℕ) (used : ℕ) (initial : ℕ) : 
  remaining = 93 → used = 4 → initial = remaining + used :=
by sorry

end NUMINAMATH_CALUDE_tissue_paper_count_l3907_390784


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3907_390732

theorem sufficient_not_necessary (m n : ℝ) :
  (∀ m n : ℝ, m / n - 1 = 0 → m - n = 0) ∧
  (∃ m n : ℝ, m - n = 0 ∧ ¬(m / n - 1 = 0)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3907_390732


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3907_390710

/-- An arithmetic sequence -/
def arithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n, a n = a₁ + (n - 1) * d

/-- The theorem to prove -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
    (h : arithmeticSequence a) 
    (h_sum : a 5 + a 10 = 12) : 
  3 * a 7 + a 9 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3907_390710


namespace NUMINAMATH_CALUDE_quadratic_has_two_distinct_roots_l3907_390772

theorem quadratic_has_two_distinct_roots (a : ℝ) :
  let f : ℝ → ℝ := fun x ↦ x^2 - (2*a - 1)*x + a^2 - a
  (∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_has_two_distinct_roots_l3907_390772


namespace NUMINAMATH_CALUDE_cube_edge_length_l3907_390709

theorem cube_edge_length (material_volume : ℕ) (num_cubes : ℕ) (edge_length : ℕ) : 
  material_volume = 12 * 18 * 6 →
  num_cubes = 48 →
  material_volume = num_cubes * edge_length * edge_length * edge_length →
  edge_length = 3 := by
sorry

end NUMINAMATH_CALUDE_cube_edge_length_l3907_390709


namespace NUMINAMATH_CALUDE_oranges_per_group_l3907_390791

def total_oranges : ℕ := 356
def orange_groups : ℕ := 178

theorem oranges_per_group : total_oranges / orange_groups = 2 := by
  sorry

end NUMINAMATH_CALUDE_oranges_per_group_l3907_390791


namespace NUMINAMATH_CALUDE_stock_trading_profit_l3907_390754

/-- Represents the stock trading scenario described in the problem -/
def stock_trading (initial_investment : ℝ) (profit_rate : ℝ) (loss_rate : ℝ) (final_sale_rate : ℝ) : ℝ :=
  let first_sale := initial_investment * (1 + profit_rate)
  let second_sale := first_sale * (1 - loss_rate)
  let third_sale := second_sale * final_sale_rate
  let first_profit := first_sale - initial_investment
  let final_loss := second_sale - third_sale
  first_profit - final_loss

/-- Theorem stating that given the conditions in the problem, A's overall profit is 10 yuan -/
theorem stock_trading_profit :
  stock_trading 10000 0.1 0.1 0.9 = 10 := by
  sorry

end NUMINAMATH_CALUDE_stock_trading_profit_l3907_390754


namespace NUMINAMATH_CALUDE_hiking_team_selection_l3907_390728

theorem hiking_team_selection (n : ℕ) (k : ℕ) : n = 10 → k = 3 → Nat.choose n k = 120 := by
  sorry

end NUMINAMATH_CALUDE_hiking_team_selection_l3907_390728


namespace NUMINAMATH_CALUDE_arithmetic_problems_l3907_390787

theorem arithmetic_problems :
  (270 * 9 = 2430) ∧
  (735 / 7 = 105) ∧
  (99 * 9 = 891) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_problems_l3907_390787


namespace NUMINAMATH_CALUDE_no_integer_solution_l3907_390711

theorem no_integer_solution : ¬ ∃ (x : ℤ), x + (2*x + 33) + (3*x - 24) = 100 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l3907_390711


namespace NUMINAMATH_CALUDE_second_player_winning_strategy_l3907_390766

/-- Represents the n-gon coloring game -/
structure ColoringGame where
  n : ℕ  -- number of sides in the n-gon

/-- Defines when the second player has a winning strategy -/
def second_player_wins (game : ColoringGame) : Prop :=
  ∃ k : ℕ, game.n = 4 + 3 * k

/-- Theorem: The second player has a winning strategy if and only if n = 4 + 3k, where k ≥ 0 -/
theorem second_player_winning_strategy (game : ColoringGame) :
  second_player_wins game ↔ ∃ k : ℕ, game.n = 4 + 3 * k :=
by sorry

end NUMINAMATH_CALUDE_second_player_winning_strategy_l3907_390766


namespace NUMINAMATH_CALUDE_intersection_solution_set_l3907_390792

theorem intersection_solution_set (a b : ℝ) : 
  (∀ x : ℝ, x^2 + a*x + b < 0 ↔ (x^2 - 2*x - 3 < 0 ∧ x^2 + x - 6 < 0)) → 
  a + b = -3 := by
sorry

end NUMINAMATH_CALUDE_intersection_solution_set_l3907_390792


namespace NUMINAMATH_CALUDE_factor_sum_l3907_390747

theorem factor_sum (R S : ℝ) : 
  (∃ d e : ℝ, (X^4 + R*X^2 + S) = (X^2 - 3*X + 7) * (X^2 + d*X + e)) → 
  R + S = 54 :=
sorry

end NUMINAMATH_CALUDE_factor_sum_l3907_390747
