import Mathlib

namespace NUMINAMATH_CALUDE_parabola_point_x_coordinate_l2400_240079

theorem parabola_point_x_coordinate 
  (P : ℝ × ℝ) 
  (h1 : (P.2)^2 = 4 * P.1) 
  (h2 : Real.sqrt ((P.1 - 1)^2 + P.2^2) = 10) : 
  P.1 = 9 := by
sorry

end NUMINAMATH_CALUDE_parabola_point_x_coordinate_l2400_240079


namespace NUMINAMATH_CALUDE_circle_radii_sum_l2400_240034

theorem circle_radii_sum (r₁ r₂ : ℝ) : 
  (∃ (a : ℝ), (2 - a)^2 + (5 - a)^2 = a^2 ∧ 
               r₁ = a ∧ 
               (∃ (b : ℝ), b^2 - 14*b + 29 = 0 ∧ r₂ = b)) →
  r₁ + r₂ = 14 := by
sorry


end NUMINAMATH_CALUDE_circle_radii_sum_l2400_240034


namespace NUMINAMATH_CALUDE_gym_equipment_cost_l2400_240071

/-- Calculates the total cost in dollars including sales tax for gym equipment purchase -/
def total_cost_with_tax (squat_rack_cost : ℝ) (barbell_fraction : ℝ) (weights_cost : ℝ) 
  (exchange_rate : ℝ) (tax_rate : ℝ) : ℝ :=
  let barbell_cost := squat_rack_cost * barbell_fraction
  let total_euro := squat_rack_cost + barbell_cost + weights_cost
  let total_dollar := total_euro * exchange_rate
  let tax := total_dollar * tax_rate
  total_dollar + tax

/-- Theorem stating the total cost of gym equipment including tax -/
theorem gym_equipment_cost : 
  total_cost_with_tax 2500 0.1 750 1.15 0.06 = 4266.50 := by
  sorry

end NUMINAMATH_CALUDE_gym_equipment_cost_l2400_240071


namespace NUMINAMATH_CALUDE_exponent_division_l2400_240005

theorem exponent_division (a : ℝ) : a^5 / a^2 = a^3 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l2400_240005


namespace NUMINAMATH_CALUDE_problem_solution_l2400_240055

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

noncomputable def F (a : ℝ) (x : ℝ) : ℝ := f a x + f a (2 * x)

noncomputable def g (x : ℝ) : ℝ := f 2 x - f 2 (-x)

theorem problem_solution :
  (∀ a : ℝ, a > 0 → (∃ x : ℝ, F a x = 3) → (∀ y : ℝ, F a y ≥ 3) → a = 6) ∧
  (∀ m n : ℝ, m > 0 → n > 0 → 2 * m + 3 * n = (⨆ x, g x) →
    (∀ p q : ℝ, p > 0 → q > 0 → 1 / p + 2 / (3 * q) ≥ 2) ∧
    (∃ r s : ℝ, r > 0 ∧ s > 0 ∧ 1 / r + 2 / (3 * s) = 2)) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2400_240055


namespace NUMINAMATH_CALUDE_exists_number_with_digit_sum_decrease_l2400_240059

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem stating the existence of a number satisfying the conditions -/
theorem exists_number_with_digit_sum_decrease : 
  ∃ (N : ℕ), (∃ (M : ℕ), M = (11 * N) / 10) ∧ 
  (sum_of_digits ((11 * N) / 10) = (9 * sum_of_digits N) / 10) := by sorry

end NUMINAMATH_CALUDE_exists_number_with_digit_sum_decrease_l2400_240059


namespace NUMINAMATH_CALUDE_smallest_valid_fourth_number_l2400_240015

def is_valid_fourth_number (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100 ∧
  let sum_of_numbers := 42 + 25 + 56 + n
  let sum_of_digits := (4 + 2 + 2 + 5 + 5 + 6 + (n / 10) + (n % 10))
  4 * sum_of_digits = sum_of_numbers

theorem smallest_valid_fourth_number :
  ∀ n : ℕ, is_valid_fourth_number n → n ≥ 79 :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_fourth_number_l2400_240015


namespace NUMINAMATH_CALUDE_divisible_by_thirty_l2400_240093

theorem divisible_by_thirty (a b : ℤ) : 
  30 ∣ (a * b * (a^4 - b^4)) := by sorry

end NUMINAMATH_CALUDE_divisible_by_thirty_l2400_240093


namespace NUMINAMATH_CALUDE_max_x_on_circle_l2400_240099

/-- The maximum x-coordinate of a point on the circle (x-10)^2 + (y-30)^2 = 100 is 20. -/
theorem max_x_on_circle : 
  ∀ x y : ℝ, (x - 10)^2 + (y - 30)^2 = 100 → x ≤ 20 :=
by sorry

end NUMINAMATH_CALUDE_max_x_on_circle_l2400_240099


namespace NUMINAMATH_CALUDE_min_value_expression_l2400_240084

theorem min_value_expression (n : ℕ+) : 
  (n : ℝ) / 2 + 24 / (n : ℝ) ≥ 7 ∧ ∃ m : ℕ+, (m : ℝ) / 2 + 24 / (m : ℝ) = 7 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l2400_240084


namespace NUMINAMATH_CALUDE_eunji_exam_result_l2400_240037

def exam_problem (exam_a_total exam_b_total exam_a_wrong exam_b_extra_wrong : ℕ) : Prop :=
  let exam_a_right := exam_a_total - exam_a_wrong
  let exam_b_wrong := exam_a_wrong + exam_b_extra_wrong
  let exam_b_right := exam_b_total - exam_b_wrong
  exam_a_right + exam_b_right = 9

theorem eunji_exam_result :
  exam_problem 12 15 8 2 := by
  sorry

end NUMINAMATH_CALUDE_eunji_exam_result_l2400_240037


namespace NUMINAMATH_CALUDE_parabola_vertex_l2400_240096

/-- The vertex of the parabola y = 2x^2 - 4x - 7 is at the point (1, -9). -/
theorem parabola_vertex (x y : ℝ) : y = 2 * x^2 - 4 * x - 7 → (1, -9) = (x, y) := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_l2400_240096


namespace NUMINAMATH_CALUDE_rectangles_cover_interior_l2400_240047

-- Define the basic structures
structure Triangle where
  vertices : Fin 3 → ℝ × ℝ

structure Rectangle where
  corners : Fin 4 → ℝ × ℝ

-- Define the given line
def given_line : Set (ℝ × ℝ) := sorry

-- Define the property of covering the sides of a triangle
def covers_sides (rectangles : Fin 3 → Rectangle) (triangle : Triangle) : Prop := sorry

-- Define the property of having a side parallel to the given line
def has_parallel_side (rectangle : Rectangle) : Prop := sorry

-- Define the property of covering the interior of a triangle
def covers_interior (rectangles : Fin 3 → Rectangle) (triangle : Triangle) : Prop := sorry

-- The main theorem
theorem rectangles_cover_interior 
  (triangle : Triangle) 
  (rectangles : Fin 3 → Rectangle) 
  (h1 : covers_sides rectangles triangle)
  (h2 : ∀ i : Fin 3, has_parallel_side (rectangles i)) :
  covers_interior rectangles triangle := by sorry

end NUMINAMATH_CALUDE_rectangles_cover_interior_l2400_240047


namespace NUMINAMATH_CALUDE_rectangle_max_area_l2400_240038

theorem rectangle_max_area (l w : ℕ) : 
  (2 * l + 2 * w = 40) → 
  (∀ a b : ℕ, 2 * a + 2 * b = 40 → l * w ≥ a * b) →
  l * w = 100 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l2400_240038


namespace NUMINAMATH_CALUDE_linear_equation_and_expression_l2400_240078

theorem linear_equation_and_expression (a : ℝ) : 
  (∀ x, (a - 1) * x^(|a|) - 3 = 0 → (a - 1) * x - 3 = 0) ∧ (a - 1 ≠ 0) →
  a = -1 ∧ -4 * a^2 - 2 * (a - (2 * a^2 - a + 2)) = 8 := by
sorry

end NUMINAMATH_CALUDE_linear_equation_and_expression_l2400_240078


namespace NUMINAMATH_CALUDE_exists_divisible_by_15_with_sqrt_between_30_and_30_5_l2400_240013

theorem exists_divisible_by_15_with_sqrt_between_30_and_30_5 :
  ∃ n : ℕ+, 15 ∣ n ∧ 30 < (n : ℝ).sqrt ∧ (n : ℝ).sqrt < 30.5 :=
by
  use 900
  sorry

end NUMINAMATH_CALUDE_exists_divisible_by_15_with_sqrt_between_30_and_30_5_l2400_240013


namespace NUMINAMATH_CALUDE_complex_subtraction_l2400_240054

theorem complex_subtraction (z₁ z₂ : ℂ) (h₁ : z₁ = 3 + 4*I) (h₂ : z₂ = 1 + I) : 
  z₁ - z₂ = 2 + 3*I := by
  sorry

end NUMINAMATH_CALUDE_complex_subtraction_l2400_240054


namespace NUMINAMATH_CALUDE_rectangular_park_length_l2400_240076

theorem rectangular_park_length 
  (perimeter : ℝ) 
  (breadth : ℝ) 
  (h1 : perimeter = 1000) 
  (h2 : breadth = 200) : 
  2 * (breadth + (perimeter / 2 - breadth)) = perimeter ∧ 
  perimeter / 2 - breadth = 300 := by
sorry

end NUMINAMATH_CALUDE_rectangular_park_length_l2400_240076


namespace NUMINAMATH_CALUDE_inequality_solution_l2400_240017

theorem inequality_solution (x : ℝ) : 3 - 1 / (3 * x + 4) < 5 ↔ x > -3/2 ∧ 3 * x + 4 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l2400_240017


namespace NUMINAMATH_CALUDE_hexagons_in_100th_ring_hexagons_in_nth_ring_formula_l2400_240064

/-- Represents the number of hexagons in the nth ring of a hexagonal array -/
def hexagons_in_nth_ring (n : ℕ) : ℕ := 6 * n

/-- The hexagonal array satisfies the initial conditions -/
axiom first_ring : hexagons_in_nth_ring 1 = 6
axiom second_ring : hexagons_in_nth_ring 2 = 12

/-- Theorem: The number of hexagons in the 100th ring is 600 -/
theorem hexagons_in_100th_ring : hexagons_in_nth_ring 100 = 600 := by
  sorry

/-- Theorem: The number of hexagons in the nth ring is 6n -/
theorem hexagons_in_nth_ring_formula (n : ℕ) : hexagons_in_nth_ring n = 6 * n := by
  sorry

end NUMINAMATH_CALUDE_hexagons_in_100th_ring_hexagons_in_nth_ring_formula_l2400_240064


namespace NUMINAMATH_CALUDE_suv_max_distance_l2400_240068

-- Define the parameters
def highway_mpg : ℝ := 12.2
def city_mpg : ℝ := 7.6
def total_gallons : ℝ := 20
def highway_city_split : ℝ := 0.5  -- Equal split between highway and city

-- Theorem statement
theorem suv_max_distance :
  let highway_distance := highway_mpg * (highway_city_split * total_gallons)
  let city_distance := city_mpg * (highway_city_split * total_gallons)
  let max_distance := highway_distance + city_distance
  max_distance = 198 := by
  sorry

end NUMINAMATH_CALUDE_suv_max_distance_l2400_240068


namespace NUMINAMATH_CALUDE_arithmetic_sequence_2016_l2400_240080

def arithmetic_sequence (a : ℕ → ℕ) (d : ℕ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_2016 (a : ℕ → ℕ) (d : ℕ) :
  arithmetic_sequence a d →
  d = 2 →
  a 2007 = 2007 →
  a 2016 = 2025 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_2016_l2400_240080


namespace NUMINAMATH_CALUDE_max_cookies_buyable_l2400_240074

theorem max_cookies_buyable (total_money : ℚ) (pack_price : ℚ) (cookies_per_pack : ℕ) : 
  total_money = 20.75 ∧ pack_price = 1.75 ∧ cookies_per_pack = 2 →
  ⌊total_money / pack_price⌋ * cookies_per_pack = 22 := by
sorry

end NUMINAMATH_CALUDE_max_cookies_buyable_l2400_240074


namespace NUMINAMATH_CALUDE_robot_center_movement_l2400_240020

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circular robot -/
structure CircularRobot where
  center : Point
  radius : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Predicate to check if a point remains on a line -/
def remainsOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Predicate to check if a point is on the boundary of a circular robot -/
def isOnBoundary (p : Point) (r : CircularRobot) : Prop :=
  (p.x - r.center.x)^2 + (p.y - r.center.y)^2 = r.radius^2

/-- The main theorem -/
theorem robot_center_movement
  (r : CircularRobot)
  (h : ∀ (p : Point), isOnBoundary p r → ∃ (l : Line), ∀ (t : ℝ), remainsOnLine p l) :
  ¬ (∀ (t : ℝ), ∃ (l : Line), remainsOnLine r.center l) :=
sorry

end NUMINAMATH_CALUDE_robot_center_movement_l2400_240020


namespace NUMINAMATH_CALUDE_units_digit_of_p_l2400_240007

def units_digit (n : ℤ) : ℕ := n.natAbs % 10

theorem units_digit_of_p (p : ℤ) : 
  (0 < units_digit p) → 
  (units_digit (p^3) - units_digit (p^2) = 0) →
  (units_digit (p + 5) = 1) →
  units_digit p = 6 :=
by sorry

end NUMINAMATH_CALUDE_units_digit_of_p_l2400_240007


namespace NUMINAMATH_CALUDE_john_games_l2400_240065

/-- Calculates the number of unique working games John ended up with -/
def unique_working_games (friend_games : ℕ) (friend_nonworking : ℕ) (garage_games : ℕ) (garage_nonworking : ℕ) (garage_duplicates : ℕ) : ℕ :=
  (friend_games - friend_nonworking) + (garage_games - garage_nonworking - garage_duplicates)

/-- Theorem stating that John ended up with 17 unique working games -/
theorem john_games : unique_working_games 25 12 15 8 3 = 17 := by
  sorry

end NUMINAMATH_CALUDE_john_games_l2400_240065


namespace NUMINAMATH_CALUDE_z_modulus_l2400_240028

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation for z
def z_equation (z : ℂ) : Prop := z + 2 * i = (3 - i^3) / (1 + i)

-- Theorem statement
theorem z_modulus (z : ℂ) (h : z_equation z) : Complex.abs z = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_z_modulus_l2400_240028


namespace NUMINAMATH_CALUDE_benny_turnips_l2400_240012

theorem benny_turnips (melanie_turnips benny_turnips total_turnips : ℕ) 
  (h1 : melanie_turnips = 139)
  (h2 : total_turnips = 252)
  (h3 : melanie_turnips + benny_turnips = total_turnips) : 
  benny_turnips = 113 := by
  sorry

end NUMINAMATH_CALUDE_benny_turnips_l2400_240012


namespace NUMINAMATH_CALUDE_blocks_used_for_tower_l2400_240031

theorem blocks_used_for_tower (initial_blocks : ℕ) (remaining_blocks : ℕ) : 
  initial_blocks = 97 → remaining_blocks = 72 → initial_blocks - remaining_blocks = 25 := by
  sorry

end NUMINAMATH_CALUDE_blocks_used_for_tower_l2400_240031


namespace NUMINAMATH_CALUDE_correct_calculation_l2400_240057

theorem correct_calculation (x : ℤ) (h : x + 35 = 77) : x - 35 = 7 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2400_240057


namespace NUMINAMATH_CALUDE_minimum_training_months_l2400_240073

/-- The distance of a marathon in miles -/
def marathonDistance : ℝ := 26.3

/-- The initial running distance in miles -/
def initialDistance : ℝ := 3

/-- The function that calculates the running distance after a given number of months -/
def runningDistance (months : ℕ) : ℝ :=
  initialDistance * (2 ^ months)

/-- The theorem stating that 5 months is the minimum number of months needed to run a marathon -/
theorem minimum_training_months :
  (∀ m : ℕ, m < 5 → runningDistance m < marathonDistance) ∧
  (runningDistance 5 ≥ marathonDistance) := by
  sorry

#check minimum_training_months

end NUMINAMATH_CALUDE_minimum_training_months_l2400_240073


namespace NUMINAMATH_CALUDE_equation_solutions_l2400_240000

theorem equation_solutions :
  let f : ℝ → ℝ := fun x ↦ x * (x - 3)^2 * (5 - x)
  {x : ℝ | f x = 0} = {0, 3, 5} := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l2400_240000


namespace NUMINAMATH_CALUDE_biology_group_size_l2400_240090

theorem biology_group_size : 
  ∃ (n : ℕ), n > 0 ∧ n * (n - 1) = 210 ∧ ∀ m : ℕ, m > 0 ∧ m * (m - 1) = 210 → m = n :=
by sorry

end NUMINAMATH_CALUDE_biology_group_size_l2400_240090


namespace NUMINAMATH_CALUDE_find_y_value_l2400_240067

theorem find_y_value (x y : ℝ) (h1 : x + 3 * y = 33) (h2 : x = 3) : y = 10 := by
  sorry

end NUMINAMATH_CALUDE_find_y_value_l2400_240067


namespace NUMINAMATH_CALUDE_color_film_fraction_l2400_240092

/-- Given a film festival selection process, prove that the fraction of selected films that are in color is 30/31. -/
theorem color_film_fraction (x y : ℚ) (h1 : x > 0) (h2 : y > 0) : 
  let total_bw : ℚ := 20 * x
  let total_color : ℚ := 6 * y
  let selected_bw : ℚ := (y / x) * total_bw / 100
  let selected_color : ℚ := total_color
  (selected_color) / (selected_bw + selected_color) = 30 / 31 := by
sorry


end NUMINAMATH_CALUDE_color_film_fraction_l2400_240092


namespace NUMINAMATH_CALUDE_ellipse_k_range_l2400_240098

/-- The equation of an ellipse with parameter k -/
def ellipse_equation (x y k : ℝ) : Prop :=
  x^2 / (5 - k) + y^2 / (k - 3) = 1

/-- Conditions for the equation to represent an ellipse -/
def is_ellipse (k : ℝ) : Prop :=
  5 - k > 0 ∧ k - 3 > 0 ∧ 5 - k ≠ k - 3

/-- The range of k for which the equation represents an ellipse -/
theorem ellipse_k_range :
  ∀ k : ℝ, is_ellipse k ↔ (3 < k ∧ k < 5 ∧ k ≠ 4) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_k_range_l2400_240098


namespace NUMINAMATH_CALUDE_gdp_scientific_notation_correct_l2400_240014

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The GDP value in ten thousand yuan -/
def gdp : ℕ := 84300000

/-- The GDP expressed in scientific notation -/
def gdp_scientific : ScientificNotation where
  coefficient := 8.43
  exponent := 7
  is_valid := by sorry

/-- Theorem stating that the GDP value is correctly expressed in scientific notation -/
theorem gdp_scientific_notation_correct : 
  (gdp_scientific.coefficient * (10 : ℝ) ^ gdp_scientific.exponent) = gdp := by sorry

end NUMINAMATH_CALUDE_gdp_scientific_notation_correct_l2400_240014


namespace NUMINAMATH_CALUDE_short_hair_dog_count_is_six_l2400_240083

/-- Represents the dog grooming scenario -/
structure DogGrooming where
  shortHairDryTime : ℕ
  fullHairDryTime : ℕ
  fullHairDogCount : ℕ
  totalDryTime : ℕ

/-- The number of short-haired dogs in the grooming scenario -/
def shortHairDogCount (dg : DogGrooming) : ℕ :=
  (dg.totalDryTime - dg.fullHairDogCount * dg.fullHairDryTime) / dg.shortHairDryTime

/-- Theorem stating the number of short-haired dogs in the given scenario -/
theorem short_hair_dog_count_is_six :
  let dg : DogGrooming := {
    shortHairDryTime := 10,
    fullHairDryTime := 20,
    fullHairDogCount := 9,
    totalDryTime := 240
  }
  shortHairDogCount dg = 6 := by sorry

end NUMINAMATH_CALUDE_short_hair_dog_count_is_six_l2400_240083


namespace NUMINAMATH_CALUDE_james_friends_count_l2400_240051

/-- The number of pages James writes per letter -/
def pages_per_letter : ℕ := 3

/-- The number of times James writes letters per week -/
def times_per_week : ℕ := 2

/-- The total number of pages James writes in a year -/
def total_pages_per_year : ℕ := 624

/-- The number of weeks in a year -/
def weeks_per_year : ℕ := 52

theorem james_friends_count :
  (total_pages_per_year / weeks_per_year / times_per_week) / pages_per_letter = 2 := by
  sorry

end NUMINAMATH_CALUDE_james_friends_count_l2400_240051


namespace NUMINAMATH_CALUDE_product_of_y_coordinates_l2400_240040

-- Define the point (-4, -3)
def point_A : ℝ × ℝ := (-4, -3)

-- Define the x-coordinate of point Q
def x_Q : ℝ := 1

-- Define the distance between Q and point_A
def distance : ℝ := 8

-- Theorem statement
theorem product_of_y_coordinates :
  ∃ (y₁ y₂ : ℝ), 
    (x_Q - point_A.1)^2 + (y₁ - point_A.2)^2 = distance^2 ∧
    (x_Q - point_A.1)^2 + (y₂ - point_A.2)^2 = distance^2 ∧
    y₁ * y₂ = -30 :=
by sorry

end NUMINAMATH_CALUDE_product_of_y_coordinates_l2400_240040


namespace NUMINAMATH_CALUDE_rightmost_three_digits_of_3_to_2023_l2400_240046

theorem rightmost_three_digits_of_3_to_2023 : 3^2023 % 1000 = 787 := by
  sorry

end NUMINAMATH_CALUDE_rightmost_three_digits_of_3_to_2023_l2400_240046


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l2400_240029

-- Define the condition for a meaningful square root
def meaningful_sqrt (x : ℝ) : Prop := x - 3 ≥ 0

-- Theorem statement
theorem sqrt_meaningful_range (x : ℝ) :
  meaningful_sqrt x ↔ x ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l2400_240029


namespace NUMINAMATH_CALUDE_solution_set_l2400_240060

/-- A linear function passing through first, second, and third quadrants -/
structure LinearFunction where
  a : ℝ
  b : ℝ
  a_pos : a > 0
  intersect_x_axis : 0 = a * (-2) + b

/-- The solution set of ax > b for the given linear function -/
theorem solution_set (f : LinearFunction) : 
  ∀ x : ℝ, f.a * x > f.b ↔ x > -2 := by
sorry

end NUMINAMATH_CALUDE_solution_set_l2400_240060


namespace NUMINAMATH_CALUDE_profit_calculation_l2400_240042

/-- Given that the cost price of 55 articles equals the selling price of n articles,
    and the percent profit is 10.000000000000004%, prove that n equals 50. -/
theorem profit_calculation (C S : ℝ) (n : ℕ) 
    (h1 : 55 * C = n * S)
    (h2 : (S - C) / C * 100 = 10.000000000000004) :
    n = 50 := by
  sorry

end NUMINAMATH_CALUDE_profit_calculation_l2400_240042


namespace NUMINAMATH_CALUDE_equidistant_is_circumcenter_l2400_240023

/-- Triangle represented by complex coordinates of its vertices -/
structure ComplexTriangle where
  z₁ : ℂ
  z₂ : ℂ
  z₃ : ℂ

/-- A point is equidistant from all vertices of the triangle -/
def isEquidistant (z : ℂ) (t : ComplexTriangle) : Prop :=
  Complex.abs (z - t.z₁) = Complex.abs (z - t.z₂) ∧
  Complex.abs (z - t.z₂) = Complex.abs (z - t.z₃)

/-- The circumcenter of a triangle -/
def isCircumcenter (z : ℂ) (t : ComplexTriangle) : Prop :=
  -- Definition of circumcenter (placeholder)
  True

theorem equidistant_is_circumcenter (t : ComplexTriangle) (z : ℂ) :
  isEquidistant z t → isCircumcenter z t := by
  sorry

end NUMINAMATH_CALUDE_equidistant_is_circumcenter_l2400_240023


namespace NUMINAMATH_CALUDE_last_score_is_90_l2400_240069

def scores : List Nat := [72, 77, 85, 90, 94]

def isValidOrder (order : List Nat) : Prop :=
  order.length = 5 ∧
  order.toFinset = scores.toFinset ∧
  ∀ k : Fin 5, (order.take k.val.succ).sum % k.val.succ = 0

theorem last_score_is_90 :
  ∀ order : List Nat, isValidOrder order → order.getLast? = some 90 := by
  sorry

end NUMINAMATH_CALUDE_last_score_is_90_l2400_240069


namespace NUMINAMATH_CALUDE_triangle_area_is_63_l2400_240085

/-- The area of a triangle formed by three lines -/
def triangleArea (m1 m2 : ℚ) : ℚ :=
  let x1 : ℚ := 1
  let y1 : ℚ := 1
  let x2 : ℚ := (14/5)
  let y2 : ℚ := (23/5)
  let x3 : ℚ := (11/2)
  let y3 : ℚ := (5/2)
  (1/2) * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

/-- The theorem stating that the area of the triangle is 6.3 -/
theorem triangle_area_is_63 :
  triangleArea (3/2) (1/3) = 63/10 := by sorry

end NUMINAMATH_CALUDE_triangle_area_is_63_l2400_240085


namespace NUMINAMATH_CALUDE_mini_crossword_probability_l2400_240056

/-- Represents a crossword puzzle -/
structure Crossword :=
  (size : Nat)
  (num_clues : Nat)
  (prob_know_clue : ℚ)

/-- Calculates the probability of filling in all unshaded squares in a crossword -/
def probability_fill_crossword (c : Crossword) : ℚ :=
  sorry

/-- The specific crossword from the problem -/
def mini_crossword : Crossword :=
  { size := 5
  , num_clues := 10
  , prob_know_clue := 1/2
  }

/-- Theorem stating the probability of filling in all unshaded squares in the mini crossword -/
theorem mini_crossword_probability :
  probability_fill_crossword mini_crossword = 11/128 :=
sorry

end NUMINAMATH_CALUDE_mini_crossword_probability_l2400_240056


namespace NUMINAMATH_CALUDE_tangent_line_equation_l2400_240004

noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 - 2 * Real.log x - 1/2

def a : ℝ := 2

theorem tangent_line_equation :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := (deriv f) x₀
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) ↔ (x + y - 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l2400_240004


namespace NUMINAMATH_CALUDE_quadratic_equation_completion_square_l2400_240001

theorem quadratic_equation_completion_square (x : ℝ) :
  (16 * x^2 - 32 * x - 512 = 0) →
  ∃ (k m : ℝ), ((x + k)^2 = m) ∧ (m = 65) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_completion_square_l2400_240001


namespace NUMINAMATH_CALUDE_election_votes_calculation_l2400_240053

theorem election_votes_calculation (winning_percentage : ℚ) (majority : ℕ) (total_votes : ℕ) : 
  winning_percentage = 60 / 100 →
  majority = 1300 →
  winning_percentage * total_votes = (total_votes / 2 + majority : ℚ) →
  total_votes = 6500 := by
  sorry

end NUMINAMATH_CALUDE_election_votes_calculation_l2400_240053


namespace NUMINAMATH_CALUDE_triangle_tangent_solution_l2400_240009

theorem triangle_tangent_solution (x y z : ℝ) :
  x * (1 - y^2) * (1 - z^2) + y * (1 - z^2) * (1 - x^2) + z * (1 - x^2) * (1 - y^2) = 4 * x * y * z →
  4 * x * y * z = 4 * (x + y + z) →
  ∃ (A B C : ℝ), 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π ∧ x = Real.tan A ∧ y = Real.tan B ∧ z = Real.tan C :=
by sorry

end NUMINAMATH_CALUDE_triangle_tangent_solution_l2400_240009


namespace NUMINAMATH_CALUDE_sock_knitting_time_l2400_240081

theorem sock_knitting_time (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (1 / a = 1 / 3) → 
  (1 / a + 1 / b = 1 / 2) → 
  (b = 6) := by
sorry

end NUMINAMATH_CALUDE_sock_knitting_time_l2400_240081


namespace NUMINAMATH_CALUDE_xy_divides_x2_plus_y2_plus_1_l2400_240045

theorem xy_divides_x2_plus_y2_plus_1 (x y : ℕ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : (x * y) ∣ (x^2 + y^2 + 1)) : 
  (x^2 + y^2 + 1) / (x * y) = 3 := by
sorry

end NUMINAMATH_CALUDE_xy_divides_x2_plus_y2_plus_1_l2400_240045


namespace NUMINAMATH_CALUDE_quadratic_equation_proof_l2400_240077

theorem quadratic_equation_proof (m : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ 
    x1^2 - 4*x1 - 2*m + 5 = 0 ∧ 
    x2^2 - 4*x2 - 2*m + 5 = 0 ∧
    x1*x2 + x1 + x2 = m^2 + 6) →
  m = 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_proof_l2400_240077


namespace NUMINAMATH_CALUDE_floor_sqrt_equality_l2400_240039

theorem floor_sqrt_equality (n : ℕ+) :
  ⌊Real.sqrt (4 * n + 1)⌋ = ⌊Real.sqrt (4 * n + 2)⌋ ∧
  ⌊Real.sqrt (4 * n + 2)⌋ = ⌊Real.sqrt (4 * n + 3)⌋ ∧
  ⌊Real.sqrt (4 * n + 3)⌋ = ⌊Real.sqrt n + Real.sqrt (n + 1)⌋ :=
by sorry

end NUMINAMATH_CALUDE_floor_sqrt_equality_l2400_240039


namespace NUMINAMATH_CALUDE_faster_train_speed_l2400_240026

theorem faster_train_speed 
  (train_length : ℝ) 
  (slower_speed : ℝ) 
  (passing_time : ℝ) 
  (h1 : train_length = 80) 
  (h2 : slower_speed = 36) 
  (h3 : passing_time = 36 / 3600) : 
  ∃ (faster_speed : ℝ), faster_speed = 52 := by
  sorry

end NUMINAMATH_CALUDE_faster_train_speed_l2400_240026


namespace NUMINAMATH_CALUDE_candle_height_ratio_l2400_240027

theorem candle_height_ratio (h1 h2 : ℝ) : 
  h1 > 0 → h2 > 0 → 
  (h1 / 6 : ℝ) * 3 = h1 / 2 →
  (h2 / 8 : ℝ) * 3 = 3 * h2 / 8 →
  h1 / 2 = 5 * h2 / 8 →
  h1 / h2 = 5 / 4 := by
sorry

end NUMINAMATH_CALUDE_candle_height_ratio_l2400_240027


namespace NUMINAMATH_CALUDE_no_intersection_points_l2400_240041

theorem no_intersection_points : 
  ¬∃ (x y : ℝ), (9 * x^2 + y^2 = 9) ∧ (x^2 + 16 * y^2 = 16) := by
  sorry

end NUMINAMATH_CALUDE_no_intersection_points_l2400_240041


namespace NUMINAMATH_CALUDE_intersection_A_B_l2400_240002

-- Define set A
def A : Set ℝ := {x | ∃ y, y = Real.sqrt (2 - 2^x)}

-- Define set B
def B : Set ℝ := {x | x^2 - 3*x ≤ 0}

-- Theorem statement
theorem intersection_A_B : A ∩ B = Set.Icc 0 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l2400_240002


namespace NUMINAMATH_CALUDE_onion_harvest_weight_l2400_240075

/-- Calculates the total weight of onions harvested given the number of trips, 
    initial number of bags, increase in bags per trip, and weight per bag. -/
def totalOnionWeight (trips : ℕ) (initialBags : ℕ) (increase : ℕ) (weightPerBag : ℕ) : ℕ :=
  let finalBags := initialBags + (trips - 1) * increase
  let totalBags := trips * (initialBags + finalBags) / 2
  totalBags * weightPerBag

/-- Theorem stating that the total weight of onions harvested is 29,000 kilograms
    given the specific conditions of the problem. -/
theorem onion_harvest_weight :
  totalOnionWeight 20 10 2 50 = 29000 := by
  sorry

end NUMINAMATH_CALUDE_onion_harvest_weight_l2400_240075


namespace NUMINAMATH_CALUDE_sum_of_differences_l2400_240058

def S : Finset ℕ := Finset.range 11

def pairDifference (i j : ℕ) : ℕ := 
  if i < j then 2^j - 2^i else 2^i - 2^j

def N : ℕ := Finset.sum (S.product S) (fun (p : ℕ × ℕ) => pairDifference p.1 p.2)

theorem sum_of_differences : N = 16398 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_differences_l2400_240058


namespace NUMINAMATH_CALUDE_pentagon_area_is_8_5_l2400_240087

-- Define the pentagon vertices
def pentagon_vertices : List (ℤ × ℤ) := [(0, 0), (1, 2), (3, 3), (4, 1), (2, 0)]

-- Define the function to calculate the area of the pentagon
def pentagon_area (vertices : List (ℤ × ℤ)) : ℚ :=
  sorry

-- Theorem statement
theorem pentagon_area_is_8_5 :
  pentagon_area pentagon_vertices = 17/2 := by sorry

end NUMINAMATH_CALUDE_pentagon_area_is_8_5_l2400_240087


namespace NUMINAMATH_CALUDE_expand_expression_l2400_240050

theorem expand_expression (x : ℝ) : (11 * x + 17) * (3 * x) + 5 = 33 * x^2 + 51 * x + 5 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l2400_240050


namespace NUMINAMATH_CALUDE_allocation_schemes_l2400_240094

theorem allocation_schemes (n : ℕ) (k : ℕ) (h1 : n = 12) (h2 : k = 9) :
  (Nat.choose (n + k - 1) (k - 1)) = 165 := by
  sorry

end NUMINAMATH_CALUDE_allocation_schemes_l2400_240094


namespace NUMINAMATH_CALUDE_daniels_age_l2400_240086

theorem daniels_age (emily_age : ℕ) (brianna_age : ℕ) (daniel_age : ℕ) : 
  emily_age = 48 →
  brianna_age = emily_age / 3 →
  daniel_age = brianna_age - 3 →
  daniel_age * 2 = brianna_age →
  daniel_age = 13 := by
sorry

end NUMINAMATH_CALUDE_daniels_age_l2400_240086


namespace NUMINAMATH_CALUDE_ana_win_probability_l2400_240088

/-- Represents a player in the coin flipping game -/
inductive Player
| Juan
| Carlos
| Manu
| Ana

/-- The coin flipping game with four players -/
def CoinFlipGame :=
  {players : List Player // players = [Player.Juan, Player.Carlos, Player.Manu, Player.Ana]}

/-- The probability of flipping heads on a single flip -/
def headsProbability : ℚ := 1/2

/-- The probability of Ana winning the game -/
def anaProbability (game : CoinFlipGame) : ℚ := 1/31

/-- Theorem stating that the probability of Ana winning is 1/31 -/
theorem ana_win_probability (game : CoinFlipGame) :
  anaProbability game = 1/31 := by
  sorry

end NUMINAMATH_CALUDE_ana_win_probability_l2400_240088


namespace NUMINAMATH_CALUDE_bobs_spending_l2400_240008

def notebook_price : ℝ := 2
def magazine_price : ℝ := 5
def book_price : ℝ := 15
def notebook_quantity : ℕ := 4
def magazine_quantity : ℕ := 3
def book_quantity : ℕ := 2
def book_discount : ℝ := 0.2
def coupon_value : ℝ := 10
def coupon_threshold : ℝ := 50

def total_spending : ℝ := 
  notebook_price * notebook_quantity +
  magazine_price * magazine_quantity +
  book_price * (1 - book_discount) * book_quantity

theorem bobs_spending (spending : ℝ) :
  spending = total_spending ∧ 
  spending < coupon_threshold →
  spending = 47 :=
by sorry

end NUMINAMATH_CALUDE_bobs_spending_l2400_240008


namespace NUMINAMATH_CALUDE_temperature_conversion_l2400_240061

theorem temperature_conversion (t k : ℝ) : 
  t = 5 / 9 * (k - 32) → t = 35 → k = 95 := by
  sorry

end NUMINAMATH_CALUDE_temperature_conversion_l2400_240061


namespace NUMINAMATH_CALUDE_no_periodic_difference_with_3_and_pi_periods_l2400_240025

-- Define a periodic function
def isPeriodic (f : ℝ → ℝ) :=
  (∃ x y, f x ≠ f y) ∧ (∃ p > 0, ∀ x, f (x + p) = f x)

-- Define the period of a function
def isPeriodOf (p : ℝ) (f : ℝ → ℝ) :=
  p > 0 ∧ ∀ x, f (x + p) = f x

-- Theorem statement
theorem no_periodic_difference_with_3_and_pi_periods :
  ¬ ∃ (g h : ℝ → ℝ),
    isPeriodic g ∧ isPeriodic h ∧
    isPeriodOf 3 g ∧ isPeriodOf π h ∧
    isPeriodic (g - h) :=
sorry

end NUMINAMATH_CALUDE_no_periodic_difference_with_3_and_pi_periods_l2400_240025


namespace NUMINAMATH_CALUDE_triangle_special_sequence_l2400_240052

theorem triangle_special_sequence (A B C : ℝ) (a b c : ℝ) : 
  -- Angles form an arithmetic sequence
  ∃ (α d : ℝ), A = α ∧ B = α + d ∧ C = α + 2*d ∧
  -- Sum of angles is π
  A + B + C = π ∧
  -- Reciprocals of sides form an arithmetic sequence
  2 * (1/b) = 1/a + 1/c ∧
  -- Law of sines
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C →
  -- Conclusion: all angles are π/3
  A = π/3 ∧ B = π/3 ∧ C = π/3 := by
sorry

end NUMINAMATH_CALUDE_triangle_special_sequence_l2400_240052


namespace NUMINAMATH_CALUDE_fraction_equality_l2400_240049

theorem fraction_equality : 
  (3 / 7 + 5 / 8) / (5 / 12 + 2 / 15) = 295 / 154 := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l2400_240049


namespace NUMINAMATH_CALUDE_decagon_triangles_l2400_240095

theorem decagon_triangles : ∀ n : ℕ, n = 10 → (n.choose 3) = 120 := by sorry

end NUMINAMATH_CALUDE_decagon_triangles_l2400_240095


namespace NUMINAMATH_CALUDE_det_B_squared_minus_3B_l2400_240063

def B : Matrix (Fin 2) (Fin 2) ℝ := !![2, 4; 3, 2]

theorem det_B_squared_minus_3B : 
  Matrix.det ((B ^ 2) - 3 • B) = 88 := by sorry

end NUMINAMATH_CALUDE_det_B_squared_minus_3B_l2400_240063


namespace NUMINAMATH_CALUDE_sqrt_plus_square_zero_implies_diff_l2400_240066

theorem sqrt_plus_square_zero_implies_diff (a b : ℝ) : 
  Real.sqrt (a - 3) + (b + 1)^2 = 0 → a - b = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_plus_square_zero_implies_diff_l2400_240066


namespace NUMINAMATH_CALUDE_fraction_reducibility_l2400_240021

def is_reducible (a : ℕ) : Prop :=
  ∃ d : ℕ, d > 1 ∧ d ∣ (2 * a + 5) ∧ d ∣ (3 * a + 4)

theorem fraction_reducibility (a : ℕ) :
  is_reducible a ↔ ∃ k : ℕ, a = 7 * k + 1 :=
by sorry

end NUMINAMATH_CALUDE_fraction_reducibility_l2400_240021


namespace NUMINAMATH_CALUDE_triangle_is_right_angled_l2400_240044

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively,
    if b² = c² + a² - ca and sin A = 2 sin C, then the triangle is right-angled. -/
theorem triangle_is_right_angled 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h1 : b^2 = c^2 + a^2 - c*a) 
  (h2 : Real.sin A = 2 * Real.sin C) 
  (h3 : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h4 : 0 < A ∧ 0 < B ∧ 0 < C) 
  (h5 : A + B + C = Real.pi) : 
  ∃ (X : ℝ), (X = A ∨ X = B ∨ X = C) ∧ X = Real.pi / 2 := by
  sorry


end NUMINAMATH_CALUDE_triangle_is_right_angled_l2400_240044


namespace NUMINAMATH_CALUDE_saltwater_animals_per_aquarium_l2400_240082

theorem saltwater_animals_per_aquarium :
  ∀ (num_aquariums : ℕ) (total_animals : ℕ) (animals_per_aquarium : ℕ),
    num_aquariums = 26 →
    total_animals = 52 →
    total_animals = num_aquariums * animals_per_aquarium →
    animals_per_aquarium = 2 := by
  sorry

end NUMINAMATH_CALUDE_saltwater_animals_per_aquarium_l2400_240082


namespace NUMINAMATH_CALUDE_abs_value_difference_l2400_240006

theorem abs_value_difference (a b : ℝ) (ha : |a| = 3) (hb : |b| = 5) (hab : a > b) :
  a - b = 8 := by
  sorry

end NUMINAMATH_CALUDE_abs_value_difference_l2400_240006


namespace NUMINAMATH_CALUDE_heracles_age_l2400_240033

/-- Proves that Heracles' current age is 10 years, given the conditions stated in the problem. -/
theorem heracles_age : ∃ (H : ℕ), 
  (∀ (A : ℕ), A = H + 7 → A + 3 = 2 * H) → H = 10 := by
  sorry

end NUMINAMATH_CALUDE_heracles_age_l2400_240033


namespace NUMINAMATH_CALUDE_spoons_to_knives_ratio_l2400_240010

/-- Given a silverware set where the number of spoons is three times the number of knives,
    and the number of knives is 6, prove that the ratio of spoons to knives is 3:1. -/
theorem spoons_to_knives_ratio (knives : ℕ) (spoons : ℕ) : 
  knives = 6 → spoons = 3 * knives → spoons / knives = 3 :=
by
  sorry

#check spoons_to_knives_ratio

end NUMINAMATH_CALUDE_spoons_to_knives_ratio_l2400_240010


namespace NUMINAMATH_CALUDE_car_rental_cost_per_mile_l2400_240089

theorem car_rental_cost_per_mile 
  (rental_cost : ℝ) 
  (gas_price : ℝ) 
  (gas_amount : ℝ) 
  (miles_driven : ℝ) 
  (total_expense : ℝ) 
  (h1 : rental_cost = 150) 
  (h2 : gas_price = 3.5) 
  (h3 : gas_amount = 8) 
  (h4 : miles_driven = 320) 
  (h5 : total_expense = 338) :
  (total_expense - (rental_cost + gas_price * gas_amount)) / miles_driven = 0.5 := by
sorry


end NUMINAMATH_CALUDE_car_rental_cost_per_mile_l2400_240089


namespace NUMINAMATH_CALUDE_apples_for_juice_apples_for_juice_proof_l2400_240032

/-- Calculates the amount of apples used for fruit juice given the harvest and sales information -/
theorem apples_for_juice (total_harvest : ℕ) (restaurant_amount : ℕ) (bag_size : ℕ) 
  (total_sales : ℕ) (price_per_bag : ℕ) : ℕ :=
  let bags_sold := total_sales / price_per_bag
  let apples_sold := bags_sold * bag_size
  total_harvest - (restaurant_amount + apples_sold)

/-- Proves that 90 kg of apples were used for fruit juice given the specific values -/
theorem apples_for_juice_proof : 
  apples_for_juice 405 60 5 408 8 = 90 := by
  sorry

end NUMINAMATH_CALUDE_apples_for_juice_apples_for_juice_proof_l2400_240032


namespace NUMINAMATH_CALUDE_ollie_final_fraction_l2400_240003

-- Define the colleagues
structure Colleague where
  name : String
  initial_money : ℚ
  fraction_given : ℚ

-- Define the problem setup
def setup : List Colleague := [
  { name := "Max", initial_money := 6, fraction_given := 1/6 },
  { name := "Leevi", initial_money := 3, fraction_given := 1/3 },
  { name := "Nolan", initial_money := 2, fraction_given := 1/2 },
  { name := "Ollie", initial_money := 0, fraction_given := 0 }
]

-- Define the amount given by each colleague
def amount_given (c : Colleague) : ℚ :=
  c.initial_money * c.fraction_given

-- Define the total initial money
def total_initial_money : ℚ :=
  setup.foldl (fun acc c => acc + c.initial_money) 0

-- Define the amount Ollie receives
def ollie_receives : ℚ :=
  setup.foldl (fun acc c => acc + amount_given c) 0

-- Theorem statement
theorem ollie_final_fraction :
  ollie_receives / total_initial_money = 3 / 11 := by
  sorry

end NUMINAMATH_CALUDE_ollie_final_fraction_l2400_240003


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l2400_240043

theorem simplify_trig_expression :
  Real.sqrt (1 - 2 * Real.sin (π - 2) * Real.cos (π - 2)) = Real.sin 2 + Real.cos 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l2400_240043


namespace NUMINAMATH_CALUDE_cyclist_journey_solution_l2400_240062

/-- Represents the cyclist's journey with flat, uphill, and downhill segments -/
structure CyclistJourney where
  flat : ℝ
  uphill : ℝ
  downhill : ℝ

/-- Checks if the given journey satisfies all conditions -/
def is_valid_journey (j : CyclistJourney) : Prop :=
  -- Total distance is 80 km
  j.flat + j.uphill + j.downhill = 80 ∧
  -- Forward journey time (47/12 hours)
  j.flat / 21 + j.uphill / 12 + j.downhill / 30 = 47 / 12 ∧
  -- Return journey time (14/3 hours)
  j.flat / 21 + j.uphill / 30 + j.downhill / 12 = 14 / 3

/-- The theorem stating the correct lengths of the journey segments -/
theorem cyclist_journey_solution :
  ∃ (j : CyclistJourney), is_valid_journey j ∧ j.flat = 35 ∧ j.uphill = 15 ∧ j.downhill = 30 :=
by
  sorry


end NUMINAMATH_CALUDE_cyclist_journey_solution_l2400_240062


namespace NUMINAMATH_CALUDE_line_problem_l2400_240016

-- Define the lines
def l1 (x y : ℝ) : Prop := 2*x + y + 2 = 0
def l2 (m n x y : ℝ) : Prop := m*x + 4*y + n = 0

-- Define parallel lines
def parallel (m : ℝ) : Prop := m / 4 = 2

-- Define the distance between lines
def distance (m n : ℝ) : Prop := |2 + n/4| / Real.sqrt 5 = Real.sqrt 5

theorem line_problem (m n : ℝ) :
  parallel m → distance m n → (m + n = 36 ∨ m + n = -4) := by sorry

end NUMINAMATH_CALUDE_line_problem_l2400_240016


namespace NUMINAMATH_CALUDE_olivia_total_time_l2400_240048

/-- The total time Olivia spent on her math problems -/
def total_time (
  num_problems : ℕ)
  (time_first_three : ℕ)
  (time_next_three : ℕ)
  (time_last : ℕ)
  (break_time : ℕ)
  (checking_time : ℕ) : ℕ :=
  3 * time_first_three + 3 * time_next_three + time_last + break_time + checking_time

/-- Theorem stating that Olivia spent 43 minutes in total on her math problems -/
theorem olivia_total_time :
  total_time 7 4 6 8 2 3 = 43 :=
by sorry

end NUMINAMATH_CALUDE_olivia_total_time_l2400_240048


namespace NUMINAMATH_CALUDE_problem_statement_l2400_240022

theorem problem_statement (a b c : ℝ) 
  (h1 : a * c / (a + b) + b * a / (b + c) + c * b / (c + a) = -8)
  (h2 : b * c / (a + b) + c * a / (b + c) + a * b / (c + a) = 12)
  (h3 : a * b * c = 1) :
  b / (a + b) + c / (b + c) + a / (c + a) = -8.5 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2400_240022


namespace NUMINAMATH_CALUDE_sentence_A_most_appropriate_l2400_240024

/-- Represents a sentence to be evaluated for appropriateness --/
inductive Sentence
| A
| B
| C
| D

/-- Criteria for evaluating the appropriateness of a sentence --/
structure EvaluationCriteria :=
  (identity : Bool)
  (status : Bool)
  (occasion : Bool)
  (audience : Bool)
  (purpose : Bool)
  (respectfulLanguage : Bool)
  (toneOfDiscourse : Bool)

/-- Evaluates a sentence based on the given criteria --/
def evaluateSentence (s : Sentence) (c : EvaluationCriteria) : Bool :=
  match s with
  | Sentence.A => c.identity ∧ c.status ∧ c.occasion ∧ c.audience ∧ c.purpose ∧ c.respectfulLanguage ∧ c.toneOfDiscourse
  | Sentence.B => false
  | Sentence.C => false
  | Sentence.D => false

/-- The criteria used for evaluation --/
def criteria : EvaluationCriteria :=
  { identity := true
  , status := true
  , occasion := true
  , audience := true
  , purpose := true
  , respectfulLanguage := true
  , toneOfDiscourse := true }

/-- Theorem stating that sentence A is the most appropriate --/
theorem sentence_A_most_appropriate :
  ∀ s : Sentence, s ≠ Sentence.A → ¬(evaluateSentence s criteria) ∧ evaluateSentence Sentence.A criteria :=
sorry

end NUMINAMATH_CALUDE_sentence_A_most_appropriate_l2400_240024


namespace NUMINAMATH_CALUDE_trapezoid_construction_possible_l2400_240018

/-- Represents a trapezoid with sides a, b, c, d and diagonals d₁, d₂ -/
structure Trapezoid where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  d₁ : ℝ
  d₂ : ℝ
  h_parallel : c = d
  h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ d₁ > 0 ∧ d₂ > 0
  h_inequality₁ : d₁ - d₂ < a + b
  h_inequality₂ : a + b < d₁ + d₂

/-- A trapezoid can be constructed given parallel sides and diagonals satisfying certain conditions -/
theorem trapezoid_construction_possible (a b c d d₁ d₂ : ℝ) 
  (h_parallel : c = d)
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ d₁ > 0 ∧ d₂ > 0)
  (h_inequality₁ : d₁ - d₂ < a + b)
  (h_inequality₂ : a + b < d₁ + d₂) :
  ∃ t : Trapezoid, t.a = a ∧ t.b = b ∧ t.c = c ∧ t.d = d ∧ t.d₁ = d₁ ∧ t.d₂ = d₂ :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_construction_possible_l2400_240018


namespace NUMINAMATH_CALUDE_savings_difference_l2400_240011

/-- Represents the price of a book in dollars -/
def book_price : ℝ := 25

/-- Represents the discount percentage for Discount A -/
def discount_a_percentage : ℝ := 0.4

/-- Represents the fixed discount amount for Discount B in dollars -/
def discount_b_amount : ℝ := 5

/-- Calculates the total cost with Discount A -/
def total_cost_a : ℝ := book_price + (book_price * (1 - discount_a_percentage))

/-- Calculates the total cost with Discount B -/
def total_cost_b : ℝ := book_price + (book_price - discount_b_amount)

/-- Theorem stating the difference in savings between Discount A and Discount B -/
theorem savings_difference : total_cost_b - total_cost_a = 5 := by
  sorry

end NUMINAMATH_CALUDE_savings_difference_l2400_240011


namespace NUMINAMATH_CALUDE_students_not_enrolled_l2400_240019

theorem students_not_enrolled (total : ℕ) (french : ℕ) (german : ℕ) (both : ℕ)
  (h_total : total = 78)
  (h_french : french = 41)
  (h_german : german = 22)
  (h_both : both = 9) :
  total - (french + german - both) = 24 := by
  sorry

end NUMINAMATH_CALUDE_students_not_enrolled_l2400_240019


namespace NUMINAMATH_CALUDE_tens_digit_of_9_to_1024_l2400_240030

theorem tens_digit_of_9_to_1024 : ∃ n : ℕ, n ≥ 10 ∧ n < 100 ∧ 9^1024 ≡ n [ZMOD 100] ∧ (n / 10) % 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_9_to_1024_l2400_240030


namespace NUMINAMATH_CALUDE_cupcakes_sold_l2400_240097

theorem cupcakes_sold (initial : ℕ) (additional : ℕ) (final : ℕ) : 
  initial = 19 → additional = 10 → final = 24 → 
  initial + additional - final = 5 := by
sorry

end NUMINAMATH_CALUDE_cupcakes_sold_l2400_240097


namespace NUMINAMATH_CALUDE_binomial_15_13_l2400_240072

theorem binomial_15_13 : Nat.choose 15 13 = 105 := by
  sorry

end NUMINAMATH_CALUDE_binomial_15_13_l2400_240072


namespace NUMINAMATH_CALUDE_inner_circle_radius_is_one_l2400_240091

/-- A square with side length 4 containing 16 tangent semicircles along its perimeter --/
structure TangentCirclesSquare where
  side_length : ℝ
  num_semicircles : ℕ
  semicircle_radius : ℝ
  h_side_length : side_length = 4
  h_num_semicircles : num_semicircles = 16
  h_semicircle_radius : semicircle_radius = 1

/-- The radius of a circle tangent to all semicircles in a TangentCirclesSquare --/
def inner_circle_radius (s : TangentCirclesSquare) : ℝ := 1

/-- Theorem stating that the radius of the inner tangent circle is 1 --/
theorem inner_circle_radius_is_one (s : TangentCirclesSquare) :
  inner_circle_radius s = 1 := by sorry

end NUMINAMATH_CALUDE_inner_circle_radius_is_one_l2400_240091


namespace NUMINAMATH_CALUDE_worker_count_l2400_240070

theorem worker_count (work_amount : ℝ) : ∃ (workers : ℕ), 
  (workers : ℝ) * 75 = work_amount ∧ 
  (workers + 10 : ℝ) * 65 = work_amount ∧ 
  workers = 65 := by
sorry

end NUMINAMATH_CALUDE_worker_count_l2400_240070


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l2400_240036

theorem trigonometric_simplification (x : ℝ) : 
  Real.sin (x + π / 3) + 2 * Real.sin (x - π / 3) - Real.sqrt 3 * Real.cos (2 * π / 3 - x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l2400_240036


namespace NUMINAMATH_CALUDE_negation_of_implication_l2400_240035

theorem negation_of_implication (x y : ℝ) :
  ¬(xy = 0 → x = 0 ∨ y = 0) ↔ (xy ≠ 0 → x ≠ 0 ∧ y ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_implication_l2400_240035
