import Mathlib

namespace NUMINAMATH_CALUDE_maria_piggy_bank_theorem_l64_6410

/-- Represents the value of a coin in cents -/
def coin_value (coin : String) : ℕ :=
  match coin with
  | "dime" => 10
  | "quarter" => 25
  | "nickel" => 5
  | _ => 0

/-- Calculates the total value of coins in dollars -/
def total_value (dimes quarters nickels additional_quarters : ℕ) : ℚ :=
  (dimes * coin_value "dime" +
   (quarters + additional_quarters) * coin_value "quarter" +
   nickels * coin_value "nickel") / 100

theorem maria_piggy_bank_theorem (dimes quarters nickels additional_quarters : ℕ)
  (h1 : dimes = 4)
  (h2 : quarters = 4)
  (h3 : nickels = 7)
  (h4 : additional_quarters = 5) :
  total_value dimes quarters nickels additional_quarters = 3 :=
sorry

end NUMINAMATH_CALUDE_maria_piggy_bank_theorem_l64_6410


namespace NUMINAMATH_CALUDE_bags_difference_l64_6428

def bags_on_monday : ℕ := 7
def bags_on_next_day : ℕ := 12

theorem bags_difference : bags_on_next_day - bags_on_monday = 5 := by
  sorry

end NUMINAMATH_CALUDE_bags_difference_l64_6428


namespace NUMINAMATH_CALUDE_cooler_capacity_ratio_l64_6488

/-- Given three coolers with specific capacities, prove the ratio of the third to the second is 1/2. -/
theorem cooler_capacity_ratio :
  ∀ (c₁ c₂ c₃ : ℝ),
  c₁ = 100 →
  c₂ = c₁ + 0.5 * c₁ →
  c₁ + c₂ + c₃ = 325 →
  c₃ / c₂ = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_cooler_capacity_ratio_l64_6488


namespace NUMINAMATH_CALUDE_ticket_price_increase_l64_6430

/-- Represents the percentage increase in ticket price for each round -/
def x : ℝ := sorry

/-- The initial ticket price in yuan -/
def initial_price : ℝ := 108

/-- The final ticket price in yuan -/
def final_price : ℝ := 168

/-- Theorem stating that the equation 108(1+x)^2 = 168 correctly represents 
    the ticket price increase over two rounds -/
theorem ticket_price_increase : initial_price * (1 + x)^2 = final_price := by
  sorry

end NUMINAMATH_CALUDE_ticket_price_increase_l64_6430


namespace NUMINAMATH_CALUDE_evaluate_expression_l64_6400

theorem evaluate_expression : (64 / 0.08) - 2.5 = 797.5 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l64_6400


namespace NUMINAMATH_CALUDE_inscribed_sphere_volume_l64_6499

/-- The volume of a sphere inscribed in a cube with edge length 8 inches -/
theorem inscribed_sphere_volume :
  let cube_edge : ℝ := 8
  let sphere_radius : ℝ := cube_edge / 2
  let sphere_volume : ℝ := (4 / 3) * Real.pi * sphere_radius ^ 3
  sphere_volume = (256 / 3) * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_inscribed_sphere_volume_l64_6499


namespace NUMINAMATH_CALUDE_linear_system_solution_l64_6444

/-- Given a system of linear equations and a condition on its solution, prove the value of k. -/
theorem linear_system_solution (x y k : ℝ) : 
  3 * x + 2 * y = k + 1 →
  2 * x + 3 * y = k →
  x + y = 3 →
  k = 7 := by sorry

end NUMINAMATH_CALUDE_linear_system_solution_l64_6444


namespace NUMINAMATH_CALUDE_area_relation_l64_6491

/-- A square with vertices O, P, Q, R where O is the origin and Q is at (3,3) -/
structure Square :=
  (O : ℝ × ℝ)
  (P : ℝ × ℝ)
  (Q : ℝ × ℝ)
  (R : ℝ × ℝ)
  (is_origin : O = (0, 0))
  (is_square : Q = (3, 3))

/-- The area of a square -/
def area_square (s : Square) : ℝ := sorry

/-- The area of a triangle given three points -/
def area_triangle (A B C : ℝ × ℝ) : ℝ := sorry

/-- The theorem stating that T(3, -12) makes the area of triangle PQT twice the area of square OPQR -/
theorem area_relation (s : Square) : 
  let T : ℝ × ℝ := (3, -12)
  area_triangle s.P s.Q T = 2 * area_square s := by sorry

end NUMINAMATH_CALUDE_area_relation_l64_6491


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l64_6417

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 138) 
  (h2 : a*b + b*c + c*a = 131) : 
  a + b + c = 20 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l64_6417


namespace NUMINAMATH_CALUDE_jack_emails_l64_6421

theorem jack_emails (morning_emails : ℕ) (difference : ℕ) (afternoon_emails : ℕ) : 
  morning_emails = 6 → 
  difference = 4 → 
  morning_emails = afternoon_emails + difference → 
  afternoon_emails = 2 := by
sorry

end NUMINAMATH_CALUDE_jack_emails_l64_6421


namespace NUMINAMATH_CALUDE_maggie_goldfish_fraction_l64_6478

theorem maggie_goldfish_fraction (total : ℕ) (caught_fraction : ℚ) (remaining : ℕ) :
  total = 100 →
  caught_fraction = 3 / 5 →
  remaining = 20 →
  (total : ℚ) / 2 = (caught_fraction * ((caught_fraction * (total : ℚ) + remaining) / caught_fraction) + remaining) / caught_fraction :=
by sorry

end NUMINAMATH_CALUDE_maggie_goldfish_fraction_l64_6478


namespace NUMINAMATH_CALUDE_min_distance_sum_l64_6475

theorem min_distance_sum (x a b : ℚ) : 
  x ≠ a ∧ x ≠ b ∧ a ≠ b →
  a > b →
  (∀ y : ℚ, |y - a| + |y - b| ≥ 2) ∧ (∃ z : ℚ, |z - a| + |z - b| = 2) →
  2022 + a - b = 2024 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_sum_l64_6475


namespace NUMINAMATH_CALUDE_intersection_of_powers_of_two_and_three_l64_6441

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def middle_digit (n : ℕ) : ℕ := (n / 10) % 10

theorem intersection_of_powers_of_two_and_three :
  ∃! d : ℕ, 
    (∃ m : ℕ, is_three_digit (2^m) ∧ middle_digit (2^m) = d) ∧
    (∃ n : ℕ, is_three_digit (3^n) ∧ middle_digit (3^n) = d) :=
sorry

end NUMINAMATH_CALUDE_intersection_of_powers_of_two_and_three_l64_6441


namespace NUMINAMATH_CALUDE_sum_remainder_of_arithmetic_sequence_l64_6408

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (aₙ : ℕ) : List ℕ :=
  let n := (aₙ - a₁) / d + 1
  List.range n |>.map (λ i => a₁ + i * d)

theorem sum_remainder_of_arithmetic_sequence : 
  (arithmetic_sequence 3 8 283).sum % 8 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_of_arithmetic_sequence_l64_6408


namespace NUMINAMATH_CALUDE_degree_to_seconds_one_point_four_five_deg_to_seconds_l64_6415

theorem degree_to_seconds (deg : Real) (min_per_deg : Nat) (sec_per_min : Nat) 
  (h1 : min_per_deg = 60) (h2 : sec_per_min = 60) :
  deg * (min_per_deg * sec_per_min) = deg * 3600 := by
  sorry

theorem one_point_four_five_deg_to_seconds :
  1.45 * 3600 = 5220 := by
  sorry

end NUMINAMATH_CALUDE_degree_to_seconds_one_point_four_five_deg_to_seconds_l64_6415


namespace NUMINAMATH_CALUDE_range_of_a_l64_6469

-- Define proposition p
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 - 2*a*x + 4 > 0

-- Define proposition q
def q (a : ℝ) : Prop := ∃ x y : ℝ, (y + (a-1)*x + 2*a - 1 = 0) ∧ 
  ((x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0))

-- Theorem statement
theorem range_of_a (a : ℝ) : 
  (p a ∨ q a) ∧ ¬(p a ∧ q a) → a ≤ -2 ∨ (1 ≤ a ∧ a < 2) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l64_6469


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l64_6470

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if the perimeter of the quadrilateral formed by lines parallel to its asymptotes
    drawn from its left and right foci is 8b, then the equation of its asymptotes is y = ±x -/
theorem hyperbola_asymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (∃ c : ℝ, 2 * b = Real.sqrt ((b^2 * c^2) / a^2 + c^2)) →
  (∀ x y : ℝ, (y = x ∨ y = -x) ↔ y^2 = x^2) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l64_6470


namespace NUMINAMATH_CALUDE_subtract_inequality_from_less_than_l64_6476

theorem subtract_inequality_from_less_than (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a < b) : 
  a - b < 0 := by
  sorry

end NUMINAMATH_CALUDE_subtract_inequality_from_less_than_l64_6476


namespace NUMINAMATH_CALUDE_max_value_of_a_l64_6404

def determinant (a b c d : ℝ) : ℝ := a * d - b * c

theorem max_value_of_a :
  (∀ x : ℝ, determinant (x - 1) (a - 2) (a + 1) x ≥ 1) →
  a ≤ 3/2 ∧ ∀ b : ℝ, (∀ x : ℝ, determinant (x - 1) (b - 2) (b + 1) x ≥ 1) → b ≤ a :=
sorry

end NUMINAMATH_CALUDE_max_value_of_a_l64_6404


namespace NUMINAMATH_CALUDE_area_ratio_theorem_l64_6412

/-- A right triangle with a point on its hypotenuse and parallel lines dividing it -/
structure DividedRightTriangle where
  -- The rectangle formed by the parallel lines
  rectangle_area : ℝ
  -- The area of one of the smaller right triangles
  small_triangle_area : ℝ
  -- The condition that the area of one small triangle is n times the rectangle area
  area_condition : ∃ n : ℝ, small_triangle_area = n * rectangle_area

/-- The theorem stating the ratio of areas -/
theorem area_ratio_theorem (t : DividedRightTriangle) : 
  ∃ n : ℝ, t.small_triangle_area = n * t.rectangle_area → 
  ∃ other_triangle_area : ℝ, other_triangle_area / t.rectangle_area = 1 / (4 * n) :=
sorry

end NUMINAMATH_CALUDE_area_ratio_theorem_l64_6412


namespace NUMINAMATH_CALUDE_eventual_shot_probability_l64_6484

def basketball_game (make_probability : ℝ) (get_ball_back_probability : ℝ) : Prop :=
  (0 ≤ make_probability ∧ make_probability ≤ 1) ∧
  (0 ≤ get_ball_back_probability ∧ get_ball_back_probability ≤ 1)

theorem eventual_shot_probability
  (make_prob : ℝ)
  (get_ball_back_prob : ℝ)
  (h_game : basketball_game make_prob get_ball_back_prob)
  (h_make_prob : make_prob = 1/10)
  (h_get_ball_back_prob : get_ball_back_prob = 9/10) :
  (1 - (1 - make_prob) * get_ball_back_prob / (1 - (1 - make_prob) * (1 - get_ball_back_prob))) = 10/19 :=
by sorry


end NUMINAMATH_CALUDE_eventual_shot_probability_l64_6484


namespace NUMINAMATH_CALUDE_map_distance_to_actual_distance_l64_6422

/-- Given a map scale and a distance on the map, calculate the actual distance -/
theorem map_distance_to_actual_distance 
  (scale : ℚ) 
  (map_distance : ℚ) 
  (h_scale : scale = 1 / 10000) 
  (h_map_distance : map_distance = 16) : 
  let actual_distance := map_distance / scale
  actual_distance = 1600 := by sorry

end NUMINAMATH_CALUDE_map_distance_to_actual_distance_l64_6422


namespace NUMINAMATH_CALUDE_area_of_four_squares_l64_6437

/-- The area of a shape composed of four identical squares with side length 3 cm -/
theorem area_of_four_squares : 
  ∀ (side_length : ℝ) (num_squares : ℕ),
    side_length = 3 →
    num_squares = 4 →
    (num_squares : ℝ) * (side_length^2) = 36 := by
  sorry

end NUMINAMATH_CALUDE_area_of_four_squares_l64_6437


namespace NUMINAMATH_CALUDE_log_comparison_l64_6447

theorem log_comparison : Real.log 675 / Real.log 135 > Real.log 75 / Real.log 45 := by
  sorry

end NUMINAMATH_CALUDE_log_comparison_l64_6447


namespace NUMINAMATH_CALUDE_magnitude_of_complex_number_l64_6427

theorem magnitude_of_complex_number (i : ℂ) : i^2 = -1 → Complex.abs ((1 + i) - 2 / i) = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_number_l64_6427


namespace NUMINAMATH_CALUDE_triangle_longest_side_l64_6494

theorem triangle_longest_side (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (ratio : a / 5 = b / 6 ∧ b / 6 = c / 7)
  (perimeter : a + b + c = 720) :
  c = 280 := by
sorry

end NUMINAMATH_CALUDE_triangle_longest_side_l64_6494


namespace NUMINAMATH_CALUDE_max_triangle_side_length_l64_6431

theorem max_triangle_side_length (a b c : ℕ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →  -- Three different side lengths
  a + b + c = 24 →         -- Perimeter is 24
  a < b + c ∧ b < a + c ∧ c < a + b →  -- Triangle inequality
  a ≤ 11 ∧ b ≤ 11 ∧ c ≤ 11 :=
by sorry

end NUMINAMATH_CALUDE_max_triangle_side_length_l64_6431


namespace NUMINAMATH_CALUDE_sum_of_four_consecutive_even_integers_l64_6433

theorem sum_of_four_consecutive_even_integers (n : ℤ) : 
  (∃ k : ℤ, n = 4*k + 12 ∧ k % 2 = 0) ↔ n ∈ ({56, 80, 124, 200} : Set ℤ) := by
  sorry

#check sum_of_four_consecutive_even_integers 34
#check sum_of_four_consecutive_even_integers 56
#check sum_of_four_consecutive_even_integers 80
#check sum_of_four_consecutive_even_integers 124
#check sum_of_four_consecutive_even_integers 200

end NUMINAMATH_CALUDE_sum_of_four_consecutive_even_integers_l64_6433


namespace NUMINAMATH_CALUDE_power_of_seven_mod_twelve_l64_6432

theorem power_of_seven_mod_twelve : 7^145 % 12 = 7 := by
  sorry

end NUMINAMATH_CALUDE_power_of_seven_mod_twelve_l64_6432


namespace NUMINAMATH_CALUDE_special_number_is_24_l64_6462

/-- A two-digit number where the unit's digit exceeds the 10's digit by 2,
    and the product of the number and the sum of its digits is equal to 144. -/
def SpecialNumber (n : ℕ) : Prop :=
  ∃ (x y : ℕ),
    n = 10 * x + y ∧
    10 ≤ n ∧ n < 100 ∧
    y = x + 2 ∧
    n * (x + y) = 144

/-- The special number described is 24. -/
theorem special_number_is_24 : ∃ (n : ℕ), SpecialNumber n ∧ n = 24 := by
  sorry

end NUMINAMATH_CALUDE_special_number_is_24_l64_6462


namespace NUMINAMATH_CALUDE_range_of_a_l64_6480

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 0 1 ∧ 2 * x * (3 * x + a) < 1) ↔ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l64_6480


namespace NUMINAMATH_CALUDE_compare_fractions_l64_6413

theorem compare_fractions : (-5/6 : ℚ) > -|(-8/9 : ℚ)| := by sorry

end NUMINAMATH_CALUDE_compare_fractions_l64_6413


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l64_6414

/-- Two vectors are parallel if and only if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_m_value :
  ∀ m : ℝ, parallel (m, 4) (3, -2) → m = -6 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l64_6414


namespace NUMINAMATH_CALUDE_dessert_preference_theorem_l64_6472

/-- Represents the dessert preferences of a group of students -/
structure DessertPreferences where
  total : ℕ
  apple : ℕ
  chocolate : ℕ
  carrot : ℕ
  none : ℕ
  apple_chocolate_not_carrot : ℕ

/-- The theorem stating the number of students who like both apple pie and chocolate cake but not carrot cake -/
theorem dessert_preference_theorem (prefs : DessertPreferences) : 
  prefs.total = 50 ∧ 
  prefs.apple = 23 ∧ 
  prefs.chocolate = 20 ∧ 
  prefs.carrot = 10 ∧ 
  prefs.none = 15 → 
  prefs.apple_chocolate_not_carrot = 7 := by
  sorry

end NUMINAMATH_CALUDE_dessert_preference_theorem_l64_6472


namespace NUMINAMATH_CALUDE_linear_function_problem_l64_6495

-- Define a linear function
def linear_function (k b : ℝ) : ℝ → ℝ := λ x => k * x + b

-- Define the problem
theorem linear_function_problem :
  ∃ (k b : ℝ),
    -- The function passes through (3, 2) and (1, -2)
    (linear_function k b 3 = 2) ∧
    (linear_function k b 1 = -2) ∧
    -- The function is f(x) = 2x - 4
    (k = 2 ∧ b = -4) ∧
    -- The points (5, 6) and (-5, -14) lie on the line
    (linear_function k b 5 = 6) ∧
    (linear_function k b (-5) = -14) ∧
    -- These points are 5 units away from the y-axis
    (5 = 5 ∨ 5 = -5) :=
by
  sorry

end NUMINAMATH_CALUDE_linear_function_problem_l64_6495


namespace NUMINAMATH_CALUDE_prob_calculations_l64_6411

/-- Represents a box containing balls of two colors -/
structure Box where
  white : ℕ
  red : ℕ

/-- Calculates the probability of drawing two red balls without replacement -/
def prob_two_red (b : Box) : ℚ :=
  (b.red * (b.red - 1)) / ((b.white + b.red) * (b.white + b.red - 1))

/-- Calculates the probability of drawing a red ball after transferring two balls -/
def prob_red_after_transfer (b1 b2 : Box) : ℚ :=
  let total_ways := (b1.white + b1.red) * (b1.white + b1.red - 1) / 2
  let p_two_red := (b1.red * (b1.red - 1) / 2) / total_ways
  let p_one_each := (b1.red * b1.white) / total_ways
  let p_two_white := (b1.white * (b1.white - 1) / 2) / total_ways
  
  let p_red_given_two_red := (b2.red + 2) / (b2.white + b2.red + 2)
  let p_red_given_one_each := (b2.red + 1) / (b2.white + b2.red + 2)
  let p_red_given_two_white := b2.red / (b2.white + b2.red + 2)
  
  p_two_red * p_red_given_two_red + p_one_each * p_red_given_one_each + p_two_white * p_red_given_two_white

theorem prob_calculations (b1 b2 : Box) 
  (h1 : b1.white = 2) (h2 : b1.red = 4) (h3 : b2.white = 5) (h4 : b2.red = 3) :
  prob_two_red b1 = 2/5 ∧ prob_red_after_transfer b1 b2 = 13/30 := by
  sorry

end NUMINAMATH_CALUDE_prob_calculations_l64_6411


namespace NUMINAMATH_CALUDE_rainfall_difference_l64_6416

theorem rainfall_difference (monday_rain tuesday_rain : Real) 
  (h1 : monday_rain = 0.9)
  (h2 : tuesday_rain = 0.2) :
  monday_rain - tuesday_rain = 0.7 := by
sorry

end NUMINAMATH_CALUDE_rainfall_difference_l64_6416


namespace NUMINAMATH_CALUDE_water_pricing_l64_6453

/-- Water pricing problem -/
theorem water_pricing
  (a : ℝ) -- Previous year's water usage
  (k : ℝ) -- Proportionality coefficient
  (h_a : a > 0) -- Assumption: water usage is positive
  (h_k : k > 0) -- Assumption: coefficient is positive
  :
  -- 1. Revenue function
  let revenue (x : ℝ) := (a + k / (x - 2)) * (x - 1.8)
  -- 2. Minimum water price for 20% increase when k = 0.4a
  ∃ (x : ℝ), x = 2.4 ∧ 
    (∀ y ∈ Set.Icc 2.3 2.6, 
      revenue y ≥ 1.2 * (2.8 * a - 1.8 * a) → y ≥ x) ∧
    k = 0.4 * a →
    revenue x ≥ 1.2 * (2.8 * a - 1.8 * a)
  -- 3. Water price for minimum revenue and minimum revenue when k = 0.8a
  ∧ ∃ (x : ℝ), x = 2.4 ∧
    (∀ y ∈ Set.Icc 2.3 2.6, revenue x ≤ revenue y) ∧
    k = 0.8 * a →
    revenue x = 1.8 * a :=
by
  sorry

end NUMINAMATH_CALUDE_water_pricing_l64_6453


namespace NUMINAMATH_CALUDE_A_in_second_quadrant_l64_6460

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the second quadrant -/
def second_quadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- The point A with coordinates (-3, 4) -/
def A : Point :=
  { x := -3, y := 4 }

/-- Theorem stating that point A is in the second quadrant -/
theorem A_in_second_quadrant : second_quadrant A := by
  sorry

end NUMINAMATH_CALUDE_A_in_second_quadrant_l64_6460


namespace NUMINAMATH_CALUDE_rectangle_max_area_l64_6405

theorem rectangle_max_area (x y : ℝ) (h : x > 0 ∧ y > 0) : 
  2 * x + 2 * y = 60 → x * y ≤ 225 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l64_6405


namespace NUMINAMATH_CALUDE_jacket_cost_calculation_l64_6458

def initial_amount : ℚ := 13.99
def shirt_cost : ℚ := 12.14
def found_money : ℚ := 7.43

theorem jacket_cost_calculation : 
  let remaining_after_shirt := initial_amount - shirt_cost
  let total_remaining := remaining_after_shirt + found_money
  total_remaining = 9.28 := by sorry

end NUMINAMATH_CALUDE_jacket_cost_calculation_l64_6458


namespace NUMINAMATH_CALUDE_faulty_passed_ratio_is_one_to_eight_l64_6446

/-- Represents the ratio of two natural numbers -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- Represents the circuit board inspection results -/
structure CircuitBoardInspection where
  total : ℕ
  failed : ℕ
  faulty : ℕ

def faultyPassedRatio (inspection : CircuitBoardInspection) : Ratio :=
  { numerator := inspection.faulty - inspection.failed,
    denominator := inspection.total - inspection.failed }

theorem faulty_passed_ratio_is_one_to_eight 
  (inspection : CircuitBoardInspection) 
  (h1 : inspection.total = 3200)
  (h2 : inspection.failed = 64)
  (h3 : inspection.faulty = 456) : 
  faultyPassedRatio inspection = { numerator := 1, denominator := 8 } := by
  sorry

#check faulty_passed_ratio_is_one_to_eight

end NUMINAMATH_CALUDE_faulty_passed_ratio_is_one_to_eight_l64_6446


namespace NUMINAMATH_CALUDE_lamps_remaining_lit_l64_6452

/-- The number of lamps initially lit -/
def total_lamps : ℕ := 1997

/-- Function to count lamps that are multiples of a given number -/
def count_multiples (n : ℕ) : ℕ :=
  (total_lamps - (total_lamps % n)) / n

/-- Function to count lamps that are multiples of two given numbers -/
def count_common_multiples (a b : ℕ) : ℕ :=
  (total_lamps - (total_lamps % (a * b))) / (a * b)

/-- Function to count lamps that are multiples of three given numbers -/
def count_triple_multiples (a b c : ℕ) : ℕ :=
  (total_lamps - (total_lamps % (a * b * c))) / (a * b * c)

/-- The main theorem stating the number of lamps that remain lit -/
theorem lamps_remaining_lit : 
  total_lamps - 
  (count_multiples 2 - count_common_multiples 2 3 - count_common_multiples 2 5 + count_triple_multiples 2 3 5) -
  (count_multiples 3 - count_common_multiples 2 3 - count_common_multiples 3 5 + count_triple_multiples 2 3 5) -
  (count_multiples 5 - count_common_multiples 2 5 - count_common_multiples 3 5 + count_triple_multiples 2 3 5) = 999 := by
  sorry

end NUMINAMATH_CALUDE_lamps_remaining_lit_l64_6452


namespace NUMINAMATH_CALUDE_regular_tetradecagon_side_length_l64_6473

/-- A regular tetradecagon with perimeter 154 cm has sides of length 11 cm. -/
theorem regular_tetradecagon_side_length :
  ∀ (perimeter : ℝ) (num_sides : ℕ) (side_length : ℝ),
    perimeter = 154 →
    num_sides = 14 →
    side_length * num_sides = perimeter →
    side_length = 11 :=
by sorry

end NUMINAMATH_CALUDE_regular_tetradecagon_side_length_l64_6473


namespace NUMINAMATH_CALUDE_tangerines_most_numerous_l64_6474

/-- Represents the number of boxes for each fruit type -/
structure BoxCounts where
  tangerines : Nat
  apples : Nat
  pears : Nat

/-- Represents the number of fruits per box for each fruit type -/
structure FruitsPerBox where
  tangerines : Nat
  apples : Nat
  pears : Nat

/-- Calculates the total number of fruits for each type -/
def totalFruits (boxes : BoxCounts) (perBox : FruitsPerBox) : BoxCounts :=
  { tangerines := boxes.tangerines * perBox.tangerines
  , apples := boxes.apples * perBox.apples
  , pears := boxes.pears * perBox.pears }

/-- Proves that tangerines are the most numerous fruit -/
theorem tangerines_most_numerous (boxes : BoxCounts) (perBox : FruitsPerBox) :
  boxes.tangerines = 5 →
  boxes.apples = 3 →
  boxes.pears = 4 →
  perBox.tangerines = 30 →
  perBox.apples = 20 →
  perBox.pears = 15 →
  let totals := totalFruits boxes perBox
  totals.tangerines > totals.apples ∧ totals.tangerines > totals.pears :=
by
  sorry


end NUMINAMATH_CALUDE_tangerines_most_numerous_l64_6474


namespace NUMINAMATH_CALUDE_work_completion_time_l64_6468

/-- The time taken to complete a work given two workers with different rates and a specific work pattern. -/
theorem work_completion_time 
  (p_time q_time : ℝ) 
  (solo_time : ℝ) 
  (h1 : p_time > 0) 
  (h2 : q_time > 0) 
  (h3 : solo_time > 0) 
  (h4 : solo_time < p_time) :
  let p_rate := 1 / p_time
  let q_rate := 1 / q_time
  let work_done_solo := solo_time * p_rate
  let remaining_work := 1 - work_done_solo
  let combined_rate := p_rate + q_rate
  let remaining_time := remaining_work / combined_rate
  solo_time + remaining_time = 20 := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l64_6468


namespace NUMINAMATH_CALUDE_inequality_proof_l64_6487

theorem inequality_proof (x : ℝ) (h1 : (3/2 : ℝ) ≤ x) (h2 : x ≤ 5) :
  2 * Real.sqrt (x + 1) + Real.sqrt (2 * x - 3) + Real.sqrt (15 - 3 * x) < 2 * Real.sqrt 19 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l64_6487


namespace NUMINAMATH_CALUDE_music_store_purchase_total_l64_6457

def trumpet_price : ℝ := 149.16
def music_tool_price : ℝ := 9.98
def song_book_price : ℝ := 4.14
def accessories_price : ℝ := 21.47
def valve_oil_original_price : ℝ := 8.20
def tshirt_price : ℝ := 14.95
def valve_oil_discount_rate : ℝ := 0.20
def sales_tax_rate : ℝ := 0.065

def total_spent : ℝ := 219.67

theorem music_store_purchase_total :
  let valve_oil_price := valve_oil_original_price * (1 - valve_oil_discount_rate)
  let subtotal := trumpet_price + music_tool_price + song_book_price + 
                  accessories_price + valve_oil_price + tshirt_price
  let sales_tax := subtotal * sales_tax_rate
  subtotal + sales_tax = total_spent := by sorry

end NUMINAMATH_CALUDE_music_store_purchase_total_l64_6457


namespace NUMINAMATH_CALUDE_quadratic_roots_difference_l64_6423

theorem quadratic_roots_difference (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  a * X^2 + b * X + c = 0 → |r₁ - r₂| = 3 :=
by
  sorry

#check quadratic_roots_difference 1 (-7) 10

end NUMINAMATH_CALUDE_quadratic_roots_difference_l64_6423


namespace NUMINAMATH_CALUDE_binomial_distribution_unique_parameters_l64_6440

/-- A binomial distribution with parameters n and p -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- Expected value of a binomial distribution -/
def expectation (X : BinomialDistribution) : ℝ := X.n * X.p

/-- Variance of a binomial distribution -/
def variance (X : BinomialDistribution) : ℝ := X.n * X.p * (1 - X.p)

theorem binomial_distribution_unique_parameters
  (X : BinomialDistribution)
  (h_expectation : expectation X = 8)
  (h_variance : variance X = 1.6) :
  X.n = 10 ∧ X.p = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_binomial_distribution_unique_parameters_l64_6440


namespace NUMINAMATH_CALUDE_equation_describes_parabola_l64_6479

/-- Represents a conic section type -/
inductive ConicSection
  | Circle
  | Parabola
  | Ellipse
  | Hyperbola
  | None

/-- Determines the type of conic section based on the equation |y-3| = √((x+4)² + y²) -/
def determineConicSection : ConicSection := by sorry

/-- Theorem stating that the equation |y-3| = √((x+4)² + y²) describes a parabola -/
theorem equation_describes_parabola : determineConicSection = ConicSection.Parabola := by sorry

end NUMINAMATH_CALUDE_equation_describes_parabola_l64_6479


namespace NUMINAMATH_CALUDE_red_marbles_count_l64_6456

theorem red_marbles_count (total : ℕ) (blue : ℕ) (orange : ℕ) (red : ℕ) : 
  total = 24 →
  blue = total / 2 →
  orange = 6 →
  total = blue + orange + red →
  red = 6 := by
sorry

end NUMINAMATH_CALUDE_red_marbles_count_l64_6456


namespace NUMINAMATH_CALUDE_quadratic_max_value_l64_6429

theorem quadratic_max_value :
  let f : ℝ → ℝ := fun z ↦ -4 * z^2 + 20 * z - 6
  ∃ (max : ℝ), max = 19 ∧ ∀ z, f z ≤ max :=
by sorry

end NUMINAMATH_CALUDE_quadratic_max_value_l64_6429


namespace NUMINAMATH_CALUDE_train_platform_crossing_time_l64_6425

/-- The time required for a train to cross a platform -/
theorem train_platform_crossing_time
  (train_speed : Real)
  (man_crossing_time : Real)
  (platform_length : Real)
  (h1 : train_speed = 72 / 3.6) -- 72 kmph converted to m/s
  (h2 : man_crossing_time = 18)
  (h3 : platform_length = 340) :
  let train_length := train_speed * man_crossing_time
  let total_distance := train_length + platform_length
  total_distance / train_speed = 35 := by sorry

end NUMINAMATH_CALUDE_train_platform_crossing_time_l64_6425


namespace NUMINAMATH_CALUDE_local_minimum_of_f_l64_6486

def f (x : ℝ) := x^3 - 3*x

theorem local_minimum_of_f :
  ∃ δ > 0, ∀ x, |x - 1| < δ → f x ≥ f 1 :=
sorry

end NUMINAMATH_CALUDE_local_minimum_of_f_l64_6486


namespace NUMINAMATH_CALUDE_son_age_theorem_l64_6454

/-- Represents the ages of three generations in a family -/
structure FamilyAges where
  grandson_days : ℕ
  son_months : ℕ
  grandfather_years : ℕ

/-- Calculates the son's age in weeks given the family ages -/
def son_age_weeks (ages : FamilyAges) : ℕ :=
  ages.son_months * 4 -- Approximate weeks in a month

/-- The main theorem stating the son's age in weeks -/
theorem son_age_theorem (ages : FamilyAges) : 
  ages.grandson_days = ages.son_months ∧ 
  ages.grandson_days / 30 = ages.grandfather_years ∧ 
  ages.grandson_days / 360 + ages.son_months / 12 + ages.grandfather_years = 140 ∧ 
  ages.grandfather_years = 84 →
  son_age_weeks ages = 2548 := by
  sorry

#eval son_age_weeks { grandson_days := 2520, son_months := 588, grandfather_years := 84 }

end NUMINAMATH_CALUDE_son_age_theorem_l64_6454


namespace NUMINAMATH_CALUDE_volumes_equal_l64_6465

/-- The volume of a solid obtained by rotating a region around the y-axis -/
noncomputable def rotationVolume (f : ℝ → ℝ → Prop) : ℝ := sorry

/-- The region enclosed by the curves x² = 4y, x² = -4y, x = 4, and x = -4 -/
def region1 (x y : ℝ) : Prop :=
  (x^2 = 4*y ∨ x^2 = -4*y) ∧ -4 ≤ x ∧ x ≤ 4

/-- The region represented by points (x, y) that satisfy x² + y² ≤ 16, x² + (y - 2)² ≥ 4, and x² + (y + 2)² ≥ 4 -/
def region2 (x y : ℝ) : Prop :=
  x^2 + y^2 ≤ 16 ∧ x^2 + (y - 2)^2 ≥ 4 ∧ x^2 + (y + 2)^2 ≥ 4

/-- The theorem stating that the volumes of the two solids are equal -/
theorem volumes_equal : rotationVolume region1 = rotationVolume region2 := by
  sorry

end NUMINAMATH_CALUDE_volumes_equal_l64_6465


namespace NUMINAMATH_CALUDE_even_function_m_value_l64_6436

-- Define a function f
def f (m : ℝ) (x : ℝ) : ℝ := (x - 2) * (x - m)

-- State the theorem
theorem even_function_m_value :
  (∀ x : ℝ, f m x = f m (-x)) → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_even_function_m_value_l64_6436


namespace NUMINAMATH_CALUDE_product_base_8_units_digit_l64_6409

def base_10_product : ℕ := 123 * 57

def base_8_units_digit (n : ℕ) : ℕ := n % 8

theorem product_base_8_units_digit :
  base_8_units_digit base_10_product = 7 := by
  sorry

end NUMINAMATH_CALUDE_product_base_8_units_digit_l64_6409


namespace NUMINAMATH_CALUDE_quiz_variance_is_64_l64_6401

/-- Represents a multiple-choice quiz -/
structure Quiz where
  num_questions : ℕ
  options_per_question : ℕ
  points_per_correct : ℕ
  total_points : ℕ
  correct_probability : ℝ

/-- Calculates the variance of a student's score in the quiz -/
def quiz_score_variance (q : Quiz) : ℝ :=
  q.num_questions * q.correct_probability * (1 - q.correct_probability) * q.points_per_correct^2

/-- Theorem stating that the variance of the student's score in the given quiz is 64 -/
theorem quiz_variance_is_64 : 
  let q : Quiz := {
    num_questions := 25,
    options_per_question := 4,
    points_per_correct := 4,
    total_points := 100,
    correct_probability := 0.8
  }
  quiz_score_variance q = 64 := by
  sorry

end NUMINAMATH_CALUDE_quiz_variance_is_64_l64_6401


namespace NUMINAMATH_CALUDE_smallest_k_property_l64_6455

theorem smallest_k_property : ∃ k : ℝ, k = 2 ∧
  (∀ a b : ℝ, a ≠ 0 → b ≠ 0 → 
    (a ≤ k ∨ b ≤ k ∨ (5 / a^2 + 6 / b^3) ≤ k)) ∧
  (∀ k' : ℝ, k' < k →
    ∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧
      a > k' ∧ b > k' ∧ (5 / a^2 + 6 / b^3) > k') :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_property_l64_6455


namespace NUMINAMATH_CALUDE_range_of_a_l64_6450

-- Define the sets A and B
def A : Set ℝ := {x | (4*x - 3)^2 ≤ 1}
def B (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 1}

-- Define the propositions p and q
def p : Prop := ∀ x, x ∈ A
def q (a : ℝ) : Prop := ∀ x, x ∈ B a

-- State the theorem
theorem range_of_a : 
  (∀ a : ℝ, (¬p → ¬(q a)) ∧ ¬(¬(q a) → ¬p)) → 
  (∀ a : ℝ, 0 ≤ a ∧ a ≤ 1/2 ↔ (∀ x, x ∈ A → x ∈ B a) ∧ A ≠ B a) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l64_6450


namespace NUMINAMATH_CALUDE_min_words_to_pass_l64_6403

-- Define the exam parameters
def total_words : ℕ := 800
def passing_score : ℚ := 90 / 100
def guess_rate : ℚ := 10 / 100

-- Define the function to calculate the score based on words learned
def exam_score (words_learned : ℕ) : ℚ :=
  (words_learned : ℚ) / total_words + 
  guess_rate * ((total_words - words_learned) : ℚ) / total_words

-- Theorem statement
theorem min_words_to_pass : 
  ∀ n : ℕ, n < 712 → exam_score n < passing_score ∧ 
  exam_score 712 ≥ passing_score := by sorry

end NUMINAMATH_CALUDE_min_words_to_pass_l64_6403


namespace NUMINAMATH_CALUDE_rectangular_field_length_l64_6485

theorem rectangular_field_length (width : ℝ) (pond_side : ℝ) : 
  pond_side = 8 →
  (pond_side ^ 2) = (1 / 2) * (2 * width * width) →
  2 * width = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_length_l64_6485


namespace NUMINAMATH_CALUDE_S_intersect_T_eq_T_l64_6451

def S : Set Int := {s | ∃ n : Int, s = 2 * n + 1}
def T : Set Int := {t | ∃ n : Int, t = 4 * n + 1}

theorem S_intersect_T_eq_T : S ∩ T = T := by sorry

end NUMINAMATH_CALUDE_S_intersect_T_eq_T_l64_6451


namespace NUMINAMATH_CALUDE_shelf_filling_l64_6435

/-- Given a shelf that can be filled with books, this theorem relates the number of
    physics and chemistry books needed to fill it. -/
theorem shelf_filling (P C R B G : ℕ) : 
  (P > 0) → (C > 0) → (R > 0) → (B > 0) → (G > 0) →  -- Positive integers
  (P ≠ C) → (P ≠ R) → (P ≠ B) → (P ≠ G) →  -- Distinct values
  (C ≠ R) → (C ≠ B) → (C ≠ G) →
  (R ≠ B) → (R ≠ G) →
  (B ≠ G) →
  (∃ (x : ℚ), x > 0 ∧ P * x + 2 * C * x = G * x) →  -- Shelf filling condition
  (∃ (x : ℚ), x > 0 ∧ R * x + 2 * B * x = G * x) →  -- Alternative filling
  G = P + 2 * C :=
by sorry

end NUMINAMATH_CALUDE_shelf_filling_l64_6435


namespace NUMINAMATH_CALUDE_three_positions_from_eight_l64_6459

/-- The number of ways to choose 3 distinct positions from a group of n people. -/
def choose_three_positions (n : ℕ) : ℕ := n * (n - 1) * (n - 2)

/-- The theorem stating that choosing 3 distinct positions from 8 people results in 336 ways. -/
theorem three_positions_from_eight : choose_three_positions 8 = 336 := by
  sorry

end NUMINAMATH_CALUDE_three_positions_from_eight_l64_6459


namespace NUMINAMATH_CALUDE_gcd_45_75_l64_6445

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcd_45_75_l64_6445


namespace NUMINAMATH_CALUDE_intersection_distance_l64_6442

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 36 + y^2 / 16 = 1

-- Define the parabola (using the derived equation from the solution)
def parabola (x y : ℝ) : Prop := x = y^2 / (4 * Real.sqrt 5) + Real.sqrt 5

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | ellipse p.1 p.2 ∧ parabola p.1 p.2}

-- Theorem statement
theorem intersection_distance :
  ∃ p1 p2 : ℝ × ℝ, p1 ∈ intersection_points ∧ p2 ∈ intersection_points ∧
  p1 ≠ p2 ∧ Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2) = 2 * Real.sqrt 10 :=
sorry

end NUMINAMATH_CALUDE_intersection_distance_l64_6442


namespace NUMINAMATH_CALUDE_password_is_5949_l64_6477

def is_multiple_of_9 (n : ℕ) : Prop :=
  n % 9 = 0

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

def is_ambiguous_for_alice (n : ℕ) : Prop :=
  ∃ (x y : ℕ), x < 10 ∧ y < 10 ∧ 
    ((5000 + x * 100 + y * 10) % 9 = 0 ∨ (5000 + x * 100 + y * 10 + 9) % 9 = 0)

def is_ambiguous_for_bob (n : ℕ) : Prop :=
  ∃ (y z : ℕ), y < 10 ∧ z < 10 ∧ 
    ((5000 + y * 10 + z) % 9 = 0 ∨ (5000 + 900 + y * 10 + z) % 9 = 0)

theorem password_is_5949 :
  ∀ n : ℕ,
  5000 ≤ n ∧ n < 6000 →
  is_multiple_of_9 n →
  is_ambiguous_for_alice n →
  is_ambiguous_for_bob n →
  n ≤ 5949 :=
sorry

end NUMINAMATH_CALUDE_password_is_5949_l64_6477


namespace NUMINAMATH_CALUDE_probability_three_unused_theorem_expected_hits_nine_targets_theorem_l64_6448

/-- Represents a rocket artillery system on a missile cruiser -/
structure RocketSystem where
  total_rockets : ℕ
  hit_probability : ℝ

/-- Calculates the probability of exactly three unused rockets remaining after firing at five targets -/
def probability_three_unused (system : RocketSystem) : ℝ :=
  10 * system.hit_probability^3 * (1 - system.hit_probability)^2

/-- Calculates the expected number of targets hit when firing at nine targets -/
def expected_hits_nine_targets (system : RocketSystem) : ℝ :=
  10 * system.hit_probability - system.hit_probability^10

/-- Theorem stating the probability of exactly three unused rockets remaining after firing at five targets -/
theorem probability_three_unused_theorem (system : RocketSystem) :
  probability_three_unused system = 10 * system.hit_probability^3 * (1 - system.hit_probability)^2 := by
  sorry

/-- Theorem stating the expected number of targets hit when firing at nine targets -/
theorem expected_hits_nine_targets_theorem (system : RocketSystem) :
  expected_hits_nine_targets system = 10 * system.hit_probability - system.hit_probability^10 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_unused_theorem_expected_hits_nine_targets_theorem_l64_6448


namespace NUMINAMATH_CALUDE_unique_sum_of_squares_125_l64_6407

/-- A function that returns the number of ways to write a given number as the sum of three positive perfect squares,
    where the order doesn't matter and at least one square appears twice. -/
def countWaysToSum (n : ℕ) : ℕ :=
  sorry

/-- The theorem stating that there is exactly one way to write 125 as the sum of three positive perfect squares,
    where the order doesn't matter and at least one square appears twice. -/
theorem unique_sum_of_squares_125 : countWaysToSum 125 = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_sum_of_squares_125_l64_6407


namespace NUMINAMATH_CALUDE_plant_height_after_two_years_l64_6496

/-- The height of a plant after a given number of years -/
def plant_height (initial_height : ℝ) (years : ℕ) : ℝ :=
  initial_height * (4 ^ years)

/-- Theorem: A plant that quadruples its height every year and reaches 256 feet
    after 4 years will be 16 feet tall after 2 years -/
theorem plant_height_after_two_years
  (h : plant_height (plant_height 1 0) 4 = 256) :
  plant_height (plant_height 1 0) 2 = 16 := by
  sorry

#check plant_height_after_two_years

end NUMINAMATH_CALUDE_plant_height_after_two_years_l64_6496


namespace NUMINAMATH_CALUDE_triangle_area_ratio_origami_triangle_area_ratio_l64_6497

/-- The ratio of the areas of two triangles with the same base and different heights -/
theorem triangle_area_ratio (base : ℝ) (height1 height2 : ℝ) (h_base : base > 0) 
  (h_height1 : height1 > 0) (h_height2 : height2 > 0) :
  (1 / 2 * base * height1) / (1 / 2 * base * height2) = height1 / height2 := by
  sorry

/-- The specific ratio of triangle areas for the given problem -/
theorem origami_triangle_area_ratio :
  (1 / 2 * 3 * 6.02) / (1 / 2 * 3 * 2) = 3.01 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_ratio_origami_triangle_area_ratio_l64_6497


namespace NUMINAMATH_CALUDE_common_internal_tangent_length_l64_6449

theorem common_internal_tangent_length 
  (center_distance : ℝ) 
  (radius1 : ℝ) 
  (radius2 : ℝ) 
  (h1 : center_distance = 50)
  (h2 : radius1 = 7)
  (h3 : radius2 = 10) : 
  Real.sqrt (center_distance^2 - (radius1 + radius2)^2) = Real.sqrt 2211 :=
sorry

end NUMINAMATH_CALUDE_common_internal_tangent_length_l64_6449


namespace NUMINAMATH_CALUDE_chord_length_on_xaxis_l64_6467

/-- The length of the chord intercepted by the x-axis on the circle (x-1)^2+(y-1)^2=2 is 2 -/
theorem chord_length_on_xaxis (x y : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    ((x₁ - 1)^2 + (0 - 1)^2 = 2) ∧ 
    ((x₂ - 1)^2 + (0 - 1)^2 = 2) ∧ 
    (x₂ - x₁ = 2)) :=
by sorry

end NUMINAMATH_CALUDE_chord_length_on_xaxis_l64_6467


namespace NUMINAMATH_CALUDE_certain_number_proof_l64_6406

theorem certain_number_proof : ∃ x : ℝ, (1/4 * x + 15 = 27) ∧ (x = 48) := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l64_6406


namespace NUMINAMATH_CALUDE_spencer_jumps_l64_6471

/-- Calculates the total number of jumps Spencer will do in 5 days -/
def total_jumps (jumps_per_minute : ℕ) (minutes_per_session : ℕ) (sessions_per_day : ℕ) (days : ℕ) : ℕ :=
  jumps_per_minute * minutes_per_session * sessions_per_day * days

/-- Theorem stating that Spencer will do 400 jumps in 5 days -/
theorem spencer_jumps :
  total_jumps 4 10 2 5 = 400 := by
  sorry

end NUMINAMATH_CALUDE_spencer_jumps_l64_6471


namespace NUMINAMATH_CALUDE_magnitude_relationship_l64_6481

noncomputable def a : ℝ := Real.sqrt 5 + 2
noncomputable def b : ℝ := 2 - Real.sqrt 5
noncomputable def c : ℝ := Real.sqrt 5 - 2

theorem magnitude_relationship : a > c ∧ c > b :=
by sorry

end NUMINAMATH_CALUDE_magnitude_relationship_l64_6481


namespace NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l64_6492

/-- The complex number z = i(-2-i) is located in the third quadrant of the complex plane. -/
theorem complex_number_in_third_quadrant : 
  let z : ℂ := Complex.I * (-2 - Complex.I)
  (z.re < 0) ∧ (z.im < 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l64_6492


namespace NUMINAMATH_CALUDE_book_cost_range_l64_6466

theorem book_cost_range (p : ℝ) 
  (h1 : 11 * p < 15)
  (h2 : 12 * p > 16) : 
  4 / 3 < p ∧ p < 15 / 11 :=
by sorry

end NUMINAMATH_CALUDE_book_cost_range_l64_6466


namespace NUMINAMATH_CALUDE_expression_evaluation_l64_6443

theorem expression_evaluation : 
  (121 * (1/13 - 1/17) + 169 * (1/17 - 1/11) + 289 * (1/11 - 1/13)) / 
  (11 * (1/13 - 1/17) + 13 * (1/17 - 1/11) + 17 * (1/11 - 1/13)) = 41 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l64_6443


namespace NUMINAMATH_CALUDE_river_depth_l64_6419

/-- Proves that given a river with specified width, flow rate, and volume of water flowing into the sea per minute, the depth of the river is 5 meters. -/
theorem river_depth 
  (width : ℝ) 
  (flow_rate_kmph : ℝ) 
  (volume_per_minute : ℝ) 
  (h1 : width = 35) 
  (h2 : flow_rate_kmph = 2) 
  (h3 : volume_per_minute = 5833.333333333333) : 
  (volume_per_minute / (flow_rate_kmph * 1000 / 60 * width)) = 5 := by
  sorry

end NUMINAMATH_CALUDE_river_depth_l64_6419


namespace NUMINAMATH_CALUDE_absolute_value_complex_l64_6439

theorem absolute_value_complex : Complex.abs (-1 + (2/3) * Complex.I) = Real.sqrt 13 / 3 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_complex_l64_6439


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l64_6461

theorem quadratic_roots_condition (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 + x - m = 0 ∧ y^2 + y - m = 0) → m > -1/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l64_6461


namespace NUMINAMATH_CALUDE_cab_speed_fraction_l64_6464

/-- Proves that for a cab with a usual journey time of 40 minutes, if it's 8 minutes late at a reduced speed, then the reduced speed is 5/6 of its usual speed. -/
theorem cab_speed_fraction (usual_time : ℕ) (delay : ℕ) : 
  usual_time = 40 → delay = 8 → (usual_time : ℚ) / (usual_time + delay) = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_cab_speed_fraction_l64_6464


namespace NUMINAMATH_CALUDE_football_banquet_food_consumption_l64_6489

theorem football_banquet_food_consumption 
  (max_food_per_guest : ℝ) 
  (min_guests : ℕ) 
  (h1 : max_food_per_guest = 2) 
  (h2 : min_guests = 160) : 
  ∃ (total_food : ℝ), total_food = max_food_per_guest * min_guests ∧ total_food = 320 := by
  sorry

end NUMINAMATH_CALUDE_football_banquet_food_consumption_l64_6489


namespace NUMINAMATH_CALUDE_bacteria_growth_l64_6418

/-- The number of days it takes to fill the entire dish -/
def total_days : ℕ := 30

/-- The fraction of the dish filled after a given number of days -/
def fraction_filled (days : ℕ) : ℚ :=
  1 / 2^(total_days - days)

/-- The number of days it takes to fill one sixteenth of the dish -/
def days_to_fill_sixteenth : ℕ := 26

theorem bacteria_growth :
  fraction_filled days_to_fill_sixteenth = 1/16 :=
sorry

end NUMINAMATH_CALUDE_bacteria_growth_l64_6418


namespace NUMINAMATH_CALUDE_smallest_number_with_remainders_l64_6463

theorem smallest_number_with_remainders : ∃ n : ℕ, 
  (n % 4 = 1) ∧ (n % 3 = 2) ∧ (n % 5 = 2) ∧
  (∀ m : ℕ, m < n → ¬(m % 4 = 1 ∧ m % 3 = 2 ∧ m % 5 = 2)) ∧
  n = 17 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_with_remainders_l64_6463


namespace NUMINAMATH_CALUDE_vector_sum_proof_l64_6424

/-- Given points A, B, and C in ℝ², prove that AC + (1/3)BA = (2, -3) -/
theorem vector_sum_proof (A B C : ℝ × ℝ) 
  (hA : A = (2, 4)) 
  (hB : B = (-1, -5)) 
  (hC : C = (3, -2)) : 
  (C.1 - A.1, C.2 - A.2) + (1/3 * (A.1 - B.1), 1/3 * (A.2 - B.2)) = (2, -3) := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_proof_l64_6424


namespace NUMINAMATH_CALUDE_expression_evaluation_l64_6493

theorem expression_evaluation : 8 / 4 - 3^2 + 4 * 2 + Nat.factorial 5 = 121 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l64_6493


namespace NUMINAMATH_CALUDE_odd_tau_tau_count_l64_6482

/-- The number of positive integer divisors of n -/
def τ (n : ℕ) : ℕ := sorry

/-- Predicate to check if a natural number is odd -/
def isOdd (n : ℕ) : Prop := n % 2 = 1

/-- The count of integers n between 1 and 50 (inclusive) such that τ(τ(n)) is odd -/
def countOddTauTau : ℕ := sorry

theorem odd_tau_tau_count : countOddTauTau = 17 := by sorry

end NUMINAMATH_CALUDE_odd_tau_tau_count_l64_6482


namespace NUMINAMATH_CALUDE_triangle_equation_solution_l64_6438

theorem triangle_equation_solution (a b c : ℝ) 
  (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c > 0) (h4 : a < b + c) :
  let p := (a + b + c) / 2
  let x := a * b * c / (2 * Real.sqrt (p * (p - a) * (p - b) * (p - c)))
  b * Real.sqrt (x^2 - c^2) + c * Real.sqrt (x^2 - b^2) = a * x := by
sorry

end NUMINAMATH_CALUDE_triangle_equation_solution_l64_6438


namespace NUMINAMATH_CALUDE_coefficient_x_squared_proof_l64_6426

/-- The coefficient of x^2 in the expansion of (1-3x)^7 -/
def coefficient_x_squared : ℕ := 7

/-- The binomial coefficient (n choose k) -/
def binomial (n k : ℕ) : ℕ := sorry

theorem coefficient_x_squared_proof :
  coefficient_x_squared = binomial 7 6 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_proof_l64_6426


namespace NUMINAMATH_CALUDE_billy_watches_95_videos_billy_within_time_constraint_l64_6483

/-- The number of videos Billy watches in total -/
def total_videos (suggestions_per_trial : ℕ) (num_trials : ℕ) (suggestions_per_category : ℕ) (num_categories : ℕ) : ℕ :=
  suggestions_per_trial * num_trials + suggestions_per_category * num_categories

/-- Theorem stating that Billy watches 95 videos in total -/
theorem billy_watches_95_videos :
  total_videos 15 5 10 2 = 95 := by
  sorry

/-- Billy's time constraint in minutes -/
def time_constraint : ℕ := 60

/-- Time taken to watch each video in minutes -/
def time_per_video : ℕ := 4

/-- Theorem stating that Billy's total watching time does not exceed the time constraint -/
theorem billy_within_time_constraint :
  total_videos 15 5 10 2 * time_per_video ≤ time_constraint := by
  sorry

end NUMINAMATH_CALUDE_billy_watches_95_videos_billy_within_time_constraint_l64_6483


namespace NUMINAMATH_CALUDE_set_equality_from_intersection_union_equality_l64_6420

theorem set_equality_from_intersection_union_equality (A : Set α) :
  ∃ X, (X ∩ A = X ∪ A) → (X = A) := by sorry

end NUMINAMATH_CALUDE_set_equality_from_intersection_union_equality_l64_6420


namespace NUMINAMATH_CALUDE_find_n_l64_6490

theorem find_n (x y : ℝ) (h : 2 * x - y = 4) : 
  ∃ n : ℝ, 6 * x - n * y = 12 ∧ n = 3 := by
sorry

end NUMINAMATH_CALUDE_find_n_l64_6490


namespace NUMINAMATH_CALUDE_biased_coin_flip_l64_6498

theorem biased_coin_flip (h : ℝ) : 
  0 < h → h < 1 →
  (4 : ℝ) * h * (1 - h)^3 = 6 * h^2 * (1 - h)^2 →
  (6 : ℝ) * (2/5)^2 * (3/5)^2 = 216/625 :=
by sorry

end NUMINAMATH_CALUDE_biased_coin_flip_l64_6498


namespace NUMINAMATH_CALUDE_exam_maximum_marks_l64_6402

theorem exam_maximum_marks :
  ∀ (passing_threshold : ℝ) (obtained_marks : ℝ) (failing_margin : ℝ),
    passing_threshold = 0.30 →
    obtained_marks = 30 →
    failing_margin = 36 →
    ∃ (max_marks : ℝ),
      max_marks = 220 ∧
      passing_threshold * max_marks = obtained_marks + failing_margin :=
by sorry

end NUMINAMATH_CALUDE_exam_maximum_marks_l64_6402


namespace NUMINAMATH_CALUDE_definite_integral_equals_2pi_l64_6434

theorem definite_integral_equals_2pi :
  ∫ x in (-2 : ℝ)..2, (Real.sqrt (4 - x^2) - x^2017) = 2 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_definite_integral_equals_2pi_l64_6434
