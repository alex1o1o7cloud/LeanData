import Mathlib

namespace NUMINAMATH_CALUDE_spinster_cat_problem_l3602_360260

theorem spinster_cat_problem (S C : ℕ) : 
  (S : ℚ) / C = 2 / 9 →   -- Ratio of spinsters to cats
  C = S + 63 →            -- 63 more cats than spinsters
  S = 18 :=               -- Number of spinsters
by sorry

end NUMINAMATH_CALUDE_spinster_cat_problem_l3602_360260


namespace NUMINAMATH_CALUDE_birds_and_storks_l3602_360290

/-- Given a fence with birds and storks, prove that the initial number of birds
    is equal to the initial number of storks plus 3. -/
theorem birds_and_storks (initial_birds initial_storks : ℕ) : 
  initial_storks = 3 → 
  (initial_birds + initial_storks + 2 = initial_birds + 1) → 
  initial_birds = initial_storks + 3 := by
sorry

end NUMINAMATH_CALUDE_birds_and_storks_l3602_360290


namespace NUMINAMATH_CALUDE_solution_set_l3602_360272

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain : Set ℝ := {x | x > 0}

-- State the theorem
theorem solution_set (hf : StrictMono f) (hd : ∀ x ∈ domain, f x ≠ 0) :
  {x : ℝ | f x > f (8 * (x - 2))} = {x : ℝ | 2 < x ∧ x < 16/7} := by sorry

end NUMINAMATH_CALUDE_solution_set_l3602_360272


namespace NUMINAMATH_CALUDE_consecutive_square_differences_exist_l3602_360269

theorem consecutive_square_differences_exist : 
  ∃ (a b c : ℝ), 
    (a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧ 
    (a > 2022 ∨ b > 2022 ∨ c > 2022) ∧
    (∃ (k : ℤ), 
      (a^2 - b^2 = k) ∧ 
      (b^2 - c^2 = k + 1) ∧ 
      (c^2 - a^2 = k + 2)) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_square_differences_exist_l3602_360269


namespace NUMINAMATH_CALUDE_total_sand_donation_l3602_360207

-- Define the amounts of sand for each city
def city_A : ℚ := 16 + 1/2
def city_B : ℕ := 26
def city_C : ℚ := 24 + 1/2
def city_D : ℕ := 28

-- Theorem statement
theorem total_sand_donation :
  city_A + city_B + city_C + city_D = 95 := by
  sorry

end NUMINAMATH_CALUDE_total_sand_donation_l3602_360207


namespace NUMINAMATH_CALUDE_white_to_red_black_ratio_l3602_360200

/-- Represents the number of socks James has -/
structure Socks :=
  (red : ℕ)
  (black : ℕ)
  (white : ℕ)

/-- The total number of socks James has -/
def total_socks (s : Socks) : ℕ := s.red + s.black + s.white

/-- The theorem stating the ratio of white socks to red and black socks -/
theorem white_to_red_black_ratio (s : Socks) :
  s.red = 40 →
  s.black = 20 →
  s.white = s.red + s.black →
  total_socks s = 90 →
  s.white * 2 = s.red + s.black :=
by
  sorry


end NUMINAMATH_CALUDE_white_to_red_black_ratio_l3602_360200


namespace NUMINAMATH_CALUDE_geometry_problem_l3602_360217

/-- Two lines are different if they are not equal -/
def different_lines (a b : Line) : Prop := a ≠ b

/-- Two planes are different if they are not equal -/
def different_planes (α β : Plane) : Prop := α ≠ β

/-- A line is perpendicular to a plane -/
def line_perp_plane (l : Line) (p : Plane) : Prop := sorry

/-- A line is parallel to a plane -/
def line_parallel_plane (l : Line) (p : Plane) : Prop := sorry

/-- Two planes are parallel -/
def planes_parallel (p1 p2 : Plane) : Prop := sorry

/-- Two lines are parallel -/
def lines_parallel (l1 l2 : Line) : Prop := sorry

/-- Two lines are perpendicular -/
def lines_perp (l1 l2 : Line) : Prop := sorry

/-- A line is contained in a plane -/
def line_in_plane (l : Line) (p : Plane) : Prop := sorry

/-- Two planes are perpendicular -/
def planes_perp (p1 p2 : Plane) : Prop := sorry

theorem geometry_problem (a b : Line) (α β : Plane) 
  (h1 : different_lines a b) (h2 : different_planes α β) : 
  (((line_perp_plane a α ∧ line_perp_plane b β ∧ planes_parallel α β) → lines_parallel a b) ∧
   ((line_perp_plane a α ∧ line_parallel_plane b β ∧ planes_parallel α β) → lines_perp a b) ∧
   (¬((planes_parallel α β ∧ line_in_plane a α ∧ line_in_plane b β) → lines_parallel a b)) ∧
   ((line_perp_plane a α ∧ line_perp_plane b β ∧ planes_perp α β) → lines_perp a b)) := by
  sorry

end NUMINAMATH_CALUDE_geometry_problem_l3602_360217


namespace NUMINAMATH_CALUDE_milk_box_width_l3602_360210

/-- Represents a rectangular milk box -/
structure MilkBox where
  length : Real
  width : Real

/-- Calculates the volume of milk removed when lowering the level by a certain height -/
def volumeRemoved (box : MilkBox) (height : Real) : Real :=
  box.length * box.width * height

theorem milk_box_width (box : MilkBox) 
  (h1 : box.length = 50)
  (h2 : volumeRemoved box 0.5 = 4687.5 / 7.5) : 
  box.width = 25 := by
  sorry

#check milk_box_width

end NUMINAMATH_CALUDE_milk_box_width_l3602_360210


namespace NUMINAMATH_CALUDE_angle_B_measure_side_b_value_l3602_360221

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  S : ℝ
  -- Add necessary conditions
  angle_sum : A + B + C = π
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C

-- Part 1
theorem angle_B_measure (t : Triangle) (h : (Real.cos t.B) / (Real.cos t.C) = -t.b / (2 * t.a + t.c)) :
  t.B = 2 * π / 3 := by sorry

-- Part 2
theorem side_b_value (t : Triangle) (h1 : t.a = 4) (h2 : t.S = 5 * Real.sqrt 3) (h3 : t.B = 2 * π / 3) :
  t.b = Real.sqrt 61 := by sorry

end NUMINAMATH_CALUDE_angle_B_measure_side_b_value_l3602_360221


namespace NUMINAMATH_CALUDE_not_all_positive_l3602_360275

theorem not_all_positive (a b c : ℝ) 
  (sum_eq : a + b + c = 4)
  (sum_sq_eq : a^2 + b^2 + c^2 = 12)
  (prod_eq : a * b * c = 1) :
  ¬(a > 0 ∧ b > 0 ∧ c > 0) := by
  sorry

end NUMINAMATH_CALUDE_not_all_positive_l3602_360275


namespace NUMINAMATH_CALUDE_correct_factorization_l3602_360241

theorem correct_factorization (x y : ℝ) : x * (x - y) + y * (y - x) = (x - y)^2 := by
  sorry

end NUMINAMATH_CALUDE_correct_factorization_l3602_360241


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_3_seconds_l3602_360231

-- Define the motion equation
def s (t : ℝ) : ℝ := 1 - t + t^2

-- Define the instantaneous velocity (derivative of s)
def v (t : ℝ) : ℝ := -1 + 2*t

-- Theorem statement
theorem instantaneous_velocity_at_3_seconds :
  v 3 = 5 := by sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_3_seconds_l3602_360231


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3602_360244

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence where a_4 + a_8 = 16, a_6 = 8 -/
theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a) 
  (h_sum : a 4 + a 8 = 16) : 
  a 6 = 8 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3602_360244


namespace NUMINAMATH_CALUDE_square_plus_linear_plus_one_eq_square_l3602_360201

theorem square_plus_linear_plus_one_eq_square (x y : ℕ) :
  y^2 + y + 1 = x^2 ↔ x = 1 ∧ y = 0 := by sorry

end NUMINAMATH_CALUDE_square_plus_linear_plus_one_eq_square_l3602_360201


namespace NUMINAMATH_CALUDE_exponent_division_l3602_360216

theorem exponent_division (a : ℝ) : a^7 / a^3 = a^4 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l3602_360216


namespace NUMINAMATH_CALUDE_sum_of_cubes_divisible_by_three_l3602_360222

theorem sum_of_cubes_divisible_by_three (n : ℤ) : 
  ∃ k : ℤ, n^3 + (n+1)^3 + (n+2)^3 = 3 * k := by
sorry

end NUMINAMATH_CALUDE_sum_of_cubes_divisible_by_three_l3602_360222


namespace NUMINAMATH_CALUDE_x_squared_vs_two_to_x_l3602_360209

theorem x_squared_vs_two_to_x (x : ℝ) :
  ¬(∀ x, x^2 < 1 → 2^x < 1) ∧ ¬(∀ x, 2^x < 1 → x^2 < 1) :=
sorry

end NUMINAMATH_CALUDE_x_squared_vs_two_to_x_l3602_360209


namespace NUMINAMATH_CALUDE_lucy_groceries_weight_l3602_360299

/-- The total weight of groceries Lucy bought -/
def total_weight (cookies_packs noodles_packs cookie_weight noodle_weight : ℕ) : ℕ :=
  cookies_packs * cookie_weight + noodles_packs * noodle_weight

/-- Theorem stating that the total weight of Lucy's groceries is 11000g -/
theorem lucy_groceries_weight :
  total_weight 12 16 250 500 = 11000 := by
  sorry

end NUMINAMATH_CALUDE_lucy_groceries_weight_l3602_360299


namespace NUMINAMATH_CALUDE_square_root_sum_l3602_360292

theorem square_root_sum (x y : ℝ) : (x + 2)^2 + Real.sqrt (y - 18) = 0 → Real.sqrt (x + y) = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_root_sum_l3602_360292


namespace NUMINAMATH_CALUDE_original_price_calculation_l3602_360250

/-- The original price of a meal given the total amount paid and various fees and discounts -/
theorem original_price_calculation (total_paid : ℝ) (discount_rate : ℝ) (sales_tax_rate : ℝ) 
  (service_fee_rate : ℝ) (tip_rate : ℝ) (h_total : total_paid = 165) 
  (h_discount : discount_rate = 0.15) (h_sales_tax : sales_tax_rate = 0.10) 
  (h_service_fee : service_fee_rate = 0.05) (h_tip : tip_rate = 0.20) :
  ∃ (P : ℝ), P = total_paid / ((1 - discount_rate) * (1 + sales_tax_rate + service_fee_rate) * (1 + tip_rate)) := by
  sorry

#eval (165 : Float) / (0.85 * 1.15 * 1.20)

end NUMINAMATH_CALUDE_original_price_calculation_l3602_360250


namespace NUMINAMATH_CALUDE_warship_path_safe_l3602_360266

/-- Represents the distance of the reefs from the island in nautical miles -/
def reef_distance : ℝ := 3.8

/-- Represents the distance the warship travels from A to C in nautical miles -/
def travel_distance : ℝ := 8

/-- Represents the angle at which the island is seen from point A (in degrees) -/
def angle_at_A : ℝ := 75

/-- Represents the angle at which the island is seen from point C (in degrees) -/
def angle_at_C : ℝ := 60

/-- Theorem stating that the warship's path is safe from the reefs -/
theorem warship_path_safe :
  ∃ (distance_to_island : ℝ),
    distance_to_island > reef_distance ∧
    distance_to_island = travel_distance * Real.sin ((angle_at_A - angle_at_C) / 2 * π / 180) :=
by sorry

end NUMINAMATH_CALUDE_warship_path_safe_l3602_360266


namespace NUMINAMATH_CALUDE_vicente_shopping_cost_l3602_360215

/-- Calculates the total amount spent in US dollars given the following conditions:
  - 5 kg of rice at €2 per kg with a 10% discount
  - 3 pounds of meat at £5 per pound with a 5% sales tax
  - Exchange rates: €1 = $1.20 and £1 = $1.35
-/
theorem vicente_shopping_cost :
  let rice_kg : ℝ := 5
  let rice_price_euro : ℝ := 2
  let meat_lb : ℝ := 3
  let meat_price_pound : ℝ := 5
  let rice_discount : ℝ := 0.1
  let meat_tax : ℝ := 0.05
  let euro_to_usd : ℝ := 1.20
  let pound_to_usd : ℝ := 1.35
  
  let rice_cost : ℝ := rice_kg * rice_price_euro * (1 - rice_discount) * euro_to_usd
  let meat_cost : ℝ := meat_lb * meat_price_pound * (1 + meat_tax) * pound_to_usd
  let total_cost : ℝ := rice_cost + meat_cost

  total_cost = 32.06 := by sorry

end NUMINAMATH_CALUDE_vicente_shopping_cost_l3602_360215


namespace NUMINAMATH_CALUDE_number_difference_l3602_360249

theorem number_difference (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 - y^2 = 24) : |x - y| = 3 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l3602_360249


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l3602_360233

theorem quadratic_equation_properties (m : ℝ) :
  (∀ x, x^2 - (2*m - 3)*x + m^2 + 1 = 0 → x = m) →
    m = -1/3 ∧
  m < 0 →
    (2*m - 3)^2 - 4*(m^2 + 1) > 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l3602_360233


namespace NUMINAMATH_CALUDE_divisors_of_pq_divisors_of_p2q_divisors_of_p2q2_divisors_of_pmqn_l3602_360204

-- Define p and q as distinct prime numbers
variable (p q : ℕ) [Fact (Nat.Prime p)] [Fact (Nat.Prime q)] (h : p ≠ q)

-- Define m and n as natural numbers
variable (m n : ℕ)

-- Function to count divisors
noncomputable def countDivisors (n : ℕ) : ℕ := (Nat.divisors n).card

-- Theorems to prove
theorem divisors_of_pq : countDivisors (p * q) = 4 := by sorry

theorem divisors_of_p2q : countDivisors (p^2 * q) = 6 := by sorry

theorem divisors_of_p2q2 : countDivisors (p^2 * q^2) = 9 := by sorry

theorem divisors_of_pmqn : countDivisors (p^m * q^n) = (m + 1) * (n + 1) := by sorry

end NUMINAMATH_CALUDE_divisors_of_pq_divisors_of_p2q_divisors_of_p2q2_divisors_of_pmqn_l3602_360204


namespace NUMINAMATH_CALUDE_fraction_sum_inequality_l3602_360293

theorem fraction_sum_inequality (a b c : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) 
  (hb : 0 ≤ b ∧ b ≤ 1) 
  (hc : 0 ≤ c ∧ c ≤ 1) : 
  (a / (b*c + 1)) + (b / (a*c + 1)) + (c / (a*b + 1)) ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_inequality_l3602_360293


namespace NUMINAMATH_CALUDE_car_speed_first_hour_l3602_360267

/-- Proves that given specific conditions, the speed of a car in the first hour is 10 km/h -/
theorem car_speed_first_hour 
  (total_time : ℝ) 
  (second_hour_speed : ℝ) 
  (average_speed : ℝ) 
  (h1 : total_time = 2)
  (h2 : second_hour_speed = 60)
  (h3 : average_speed = 35) : 
  ∃ (first_hour_speed : ℝ), first_hour_speed = 10 ∧ 
    average_speed = (first_hour_speed + second_hour_speed) / total_time :=
by
  sorry

end NUMINAMATH_CALUDE_car_speed_first_hour_l3602_360267


namespace NUMINAMATH_CALUDE_first_class_males_count_l3602_360279

/-- Represents the number of male students in the first class -/
def first_class_males : ℕ := sorry

/-- Represents the number of female students in the first class -/
def first_class_females : ℕ := 13

/-- Represents the number of male students in the second class -/
def second_class_males : ℕ := 14

/-- Represents the number of female students in the second class -/
def second_class_females : ℕ := 18

/-- Represents the number of male students in the third class -/
def third_class_males : ℕ := 15

/-- Represents the number of female students in the third class -/
def third_class_females : ℕ := 17

/-- Represents the number of students unable to partner with the opposite gender -/
def unpartnered_students : ℕ := 2

theorem first_class_males_count : first_class_males = 21 := by
  sorry

end NUMINAMATH_CALUDE_first_class_males_count_l3602_360279


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l3602_360288

theorem algebraic_expression_value (x : ℝ) (h : x^2 + 2*x - 2 = 0) :
  x*(x+2) + (x+1)^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l3602_360288


namespace NUMINAMATH_CALUDE_rectangles_on_4x4_grid_l3602_360257

/-- The number of rectangles that can be formed on a 4x4 grid --/
def num_rectangles_4x4 : ℕ := 36

/-- The size of the grid --/
def grid_size : ℕ := 4

/-- Theorem: The number of rectangles on a 4x4 grid is 36 --/
theorem rectangles_on_4x4_grid :
  num_rectangles_4x4 = (grid_size.choose 2) * (grid_size.choose 2) :=
by sorry

end NUMINAMATH_CALUDE_rectangles_on_4x4_grid_l3602_360257


namespace NUMINAMATH_CALUDE_parabola_equation_l3602_360213

/-- A parabola with vertex at the origin and focus on the x-axis -/
structure Parabola where
  focus_x : ℝ
  focus_x_pos : focus_x > 0

/-- The line y = x -/
def line_y_eq_x (x : ℝ) : ℝ := x

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

theorem parabola_equation (C : Parabola) 
  (A B : Point) 
  (P : Point)
  (h1 : line_y_eq_x A.x = A.y ∧ line_y_eq_x B.x = B.y)  -- A and B lie on y = x
  (h2 : P.x = 2 ∧ P.y = 2)  -- P is (2,2)
  (h3 : P.x = (A.x + B.x) / 2 ∧ P.y = (A.y + B.y) / 2)  -- P is midpoint of AB
  : ∀ (x y : ℝ), (y^2 = 4*x) ↔ (∃ (t : ℝ), x = t^2 * C.focus_x ∧ y = 2*t * C.focus_x) :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_l3602_360213


namespace NUMINAMATH_CALUDE_charity_game_probability_l3602_360225

theorem charity_game_probability : 
  let p1 : ℝ := 0.9  -- Probability of correct answer for first picture
  let p2 : ℝ := 0.5  -- Probability of correct answer for second picture
  let p3 : ℝ := 0.4  -- Probability of correct answer for third picture
  let f1 : ℕ := 1000 -- Fund raised for first correct answer
  let f2 : ℕ := 2000 -- Fund raised for second correct answer
  let f3 : ℕ := 3000 -- Fund raised for third correct answer
  -- Probability of raising exactly 3000 yuan
  p1 * p2 * (1 - p3) = 0.27
  := by sorry

end NUMINAMATH_CALUDE_charity_game_probability_l3602_360225


namespace NUMINAMATH_CALUDE_geometric_quadratic_no_roots_l3602_360282

/-- A function representing a quadratic equation with coefficients forming a geometric sequence -/
def geometric_quadratic (a b c : ℝ) : ℝ → ℝ := 
  fun x => a * x^2 + b * x + c

/-- Proposition: A quadratic function with coefficients forming a geometric sequence has no real roots -/
theorem geometric_quadratic_no_roots (a b c : ℝ) (h : b^2 = a*c) :
  ∀ x : ℝ, geometric_quadratic a b c x ≠ 0 := by
  sorry

#check geometric_quadratic_no_roots

end NUMINAMATH_CALUDE_geometric_quadratic_no_roots_l3602_360282


namespace NUMINAMATH_CALUDE_distance_between_X_and_Y_l3602_360263

/-- The distance between points X and Y in miles -/
def D : ℝ := sorry

/-- Yolanda's walking rate in miles per hour -/
def yolanda_rate : ℝ := 5

/-- Bob's walking rate in miles per hour -/
def bob_rate : ℝ := 6

/-- The time in hours that Bob walks before meeting Yolanda -/
def bob_time : ℝ := sorry

/-- The distance Bob walks before meeting Yolanda in miles -/
def bob_distance : ℝ := 30

theorem distance_between_X_and_Y : D = 60 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_X_and_Y_l3602_360263


namespace NUMINAMATH_CALUDE_expression_value_l3602_360205

theorem expression_value (x : ℝ) (h : x = 5) : 3 * x + 2 = 17 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3602_360205


namespace NUMINAMATH_CALUDE_work_completion_rate_l3602_360253

/-- Given workers A and B, where A can finish a work in 6 days and B can do the same work in half the time taken by A, prove that A and B working together can finish 1/2 of the work in one day. -/
theorem work_completion_rate (days_a : ℕ) (days_b : ℕ) : 
  days_a = 6 →
  days_b = days_a / 2 →
  (1 : ℚ) / days_a + (1 : ℚ) / days_b = (1 : ℚ) / 2 :=
by sorry

end NUMINAMATH_CALUDE_work_completion_rate_l3602_360253


namespace NUMINAMATH_CALUDE_rolling_cube_path_length_l3602_360228

/-- Represents a cube with a given side length -/
structure Cube where
  sideLength : ℝ

/-- Represents the path of a point on a rolling cube -/
def RollingCubePath (c : Cube) : ℝ := sorry

/-- Theorem stating the length of the path followed by the center point on the top face of a rolling cube -/
theorem rolling_cube_path_length (c : Cube) (h : c.sideLength = 2) :
  RollingCubePath c = 4 * Real.pi := by sorry

end NUMINAMATH_CALUDE_rolling_cube_path_length_l3602_360228


namespace NUMINAMATH_CALUDE_equation_solution_l3602_360206

theorem equation_solution (x y z : ℝ) :
  (x - y - 3)^2 + (y - z)^2 + (x - z)^2 = 3 →
  x = z + 1 ∧ y = z - 1 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3602_360206


namespace NUMINAMATH_CALUDE_runners_meet_closer_than_half_diagonal_l3602_360271

/-- A point moving along a diagonal of a square -/
structure DiagonalRunner where
  position : ℝ  -- Position on the diagonal, normalized to [0, 1]
  direction : Bool  -- True if moving towards the endpoint, False if moving towards the start

/-- The state of two runners on diagonals of a square -/
structure SquareState where
  runner1 : DiagonalRunner
  runner2 : DiagonalRunner
  diagonal_length : ℝ

def distance (s : SquareState) : ℝ :=
  sorry

def update_state (s : SquareState) (t : ℝ) : SquareState :=
  sorry

theorem runners_meet_closer_than_half_diagonal
  (initial_state : SquareState)
  (h_positive_length : initial_state.diagonal_length > 0) :
  ∃ t : ℝ, distance (update_state initial_state t) < initial_state.diagonal_length / 2 := by
  sorry

end NUMINAMATH_CALUDE_runners_meet_closer_than_half_diagonal_l3602_360271


namespace NUMINAMATH_CALUDE_range_of_a_range_of_a_for_local_minimum_l3602_360245

/-- The function f(x) as defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ := (x - 2*a) * (x^2 + a^2*x + 2*a^3)

/-- Theorem stating the range of a given the conditions -/
theorem range_of_a (a : ℝ) : 
  (∀ x, x < 0 → (3*x^2 + 2*(a^2 - 2*a)*x < 0)) ↔ (a < 0 ∨ a > 2) :=
sorry

/-- Main theorem proving the range of a -/
theorem range_of_a_for_local_minimum :
  {a : ℝ | IsLocalMin (f a) 0} = {a : ℝ | a < 0 ∨ a > 2} :=
sorry

end NUMINAMATH_CALUDE_range_of_a_range_of_a_for_local_minimum_l3602_360245


namespace NUMINAMATH_CALUDE_oliver_final_balance_l3602_360203

def olivers_money (initial_amount savings chores_earnings frisbee_cost puzzle_cost stickers_cost
                   movie_ticket_cost snack_cost birthday_gift : ℤ) : ℤ :=
  initial_amount + savings + chores_earnings - frisbee_cost - puzzle_cost - stickers_cost -
  movie_ticket_cost - snack_cost + birthday_gift

theorem oliver_final_balance :
  olivers_money 9 5 6 4 3 2 7 3 8 = 9 := by
  sorry

end NUMINAMATH_CALUDE_oliver_final_balance_l3602_360203


namespace NUMINAMATH_CALUDE_length_of_segment_AB_is_10_l3602_360208

/-- Given point A with coordinates (2, -3, 5) and point B symmetrical to A with respect to the xy-plane,
    prove that the length of line segment AB is 10. -/
theorem length_of_segment_AB_is_10 :
  let A : ℝ × ℝ × ℝ := (2, -3, 5)
  let B : ℝ × ℝ × ℝ := (2, -3, -5)  -- B is symmetrical to A with respect to xy-plane
  ‖A - B‖ = 10 := by
  sorry

end NUMINAMATH_CALUDE_length_of_segment_AB_is_10_l3602_360208


namespace NUMINAMATH_CALUDE_filter_kit_solution_l3602_360296

def filter_kit_problem (last_filter_price : ℝ) : Prop :=
  let kit_price : ℝ := 72.50
  let filter1_price : ℝ := 12.45
  let filter2_price : ℝ := 14.05
  let discount_rate : ℝ := 0.1103448275862069
  let total_individual_price : ℝ := 2 * filter1_price + 2 * filter2_price + last_filter_price
  let amount_saved : ℝ := total_individual_price - kit_price
  amount_saved = discount_rate * total_individual_price

theorem filter_kit_solution :
  ∃ (last_filter_price : ℝ), filter_kit_problem last_filter_price ∧ last_filter_price = 28.50 := by
  sorry

end NUMINAMATH_CALUDE_filter_kit_solution_l3602_360296


namespace NUMINAMATH_CALUDE_select_from_m_gives_correct_probability_l3602_360278

def set_m : Finset Int := {-6, -5, -4, -3, -2}
def set_t : Finset Int := {-3, -2, -1, 0, 1, 2, 3, 4, 5}

def probability_negative_product : ℚ := 5 / 9

theorem select_from_m_gives_correct_probability :
  (set_m.card : ℚ) * (set_t.filter (λ x => x > 0)).card / set_t.card = probability_negative_product :=
sorry

end NUMINAMATH_CALUDE_select_from_m_gives_correct_probability_l3602_360278


namespace NUMINAMATH_CALUDE_x_value_l3602_360274

theorem x_value (x y : ℤ) (h1 : x + y = 20) (h2 : x - y = 10) : x = 15 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l3602_360274


namespace NUMINAMATH_CALUDE_percentage_difference_l3602_360247

theorem percentage_difference : (70 / 100 * 100) - (60 / 100 * 80) = 22 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l3602_360247


namespace NUMINAMATH_CALUDE_four_balls_four_boxes_l3602_360265

/-- The number of ways to distribute indistinguishable balls into indistinguishable boxes -/
def distribute_balls_boxes (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 5 ways to distribute 4 indistinguishable balls into 4 indistinguishable boxes -/
theorem four_balls_four_boxes : distribute_balls_boxes 4 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_four_balls_four_boxes_l3602_360265


namespace NUMINAMATH_CALUDE_intersection_A_B_l3602_360238

def A : Set ℝ := {x : ℝ | |x| < 2}
def B : Set ℝ := {-1, 0, 1, 2, 3}

theorem intersection_A_B : A ∩ B = {-1, 0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3602_360238


namespace NUMINAMATH_CALUDE_safe_round_trip_exists_l3602_360248

/-- Represents the cycle of a dragon's fire-breathing pattern -/
structure DragonCycle where
  active : ℕ
  sleep : ℕ

/-- Represents the travel times for the journey -/
structure TravelTimes where
  road : ℕ
  path : ℕ

/-- Checks if a given hour is safe from both dragons -/
def is_safe (h : ℕ) (d1 d2 : DragonCycle) : Prop :=
  h % (d1.active + d1.sleep) > d1.active ∧ 
  h % (d2.active + d2.sleep) > d2.active

/-- Checks if a round trip is possible within a given time frame -/
def round_trip_possible (start : ℕ) (t : TravelTimes) (d1 d2 : DragonCycle) : Prop :=
  ∀ h : ℕ, start ≤ h ∧ h < start + 2 * (t.road + t.path) → is_safe h d1 d2

/-- Main theorem: There exists a safe starting time for the round trip -/
theorem safe_round_trip_exists (t : TravelTimes) (d1 d2 : DragonCycle) : 
  ∃ start : ℕ, round_trip_possible start t d1 d2 :=
sorry

end NUMINAMATH_CALUDE_safe_round_trip_exists_l3602_360248


namespace NUMINAMATH_CALUDE_f_ln_2_equals_3_l3602_360243

-- Define a monotonically increasing function
def MonoIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- State the theorem
theorem f_ln_2_equals_3 
  (f : ℝ → ℝ)
  (h_mono : MonoIncreasing f)
  (h_prop : ∀ x : ℝ, f (f x - Real.exp x) = Real.exp 1 + 1) : 
  f (Real.log 2) = 3 := by
sorry


end NUMINAMATH_CALUDE_f_ln_2_equals_3_l3602_360243


namespace NUMINAMATH_CALUDE_smallest_valid_configuration_l3602_360224

/-- A configuration of lines in a plane -/
structure LineConfiguration where
  n : ℕ                   -- Total number of lines
  intersect_5 : ℕ         -- Index of line intersecting 5 others
  intersect_9 : ℕ         -- Index of line intersecting 9 others
  intersect_11 : ℕ        -- Index of line intersecting 11 others
  intersect_5_count : ℕ   -- Number of intersections for intersect_5
  intersect_9_count : ℕ   -- Number of intersections for intersect_9
  intersect_11_count : ℕ  -- Number of intersections for intersect_11

/-- Predicate to check if a line configuration is valid -/
def is_valid_configuration (config : LineConfiguration) : Prop :=
  config.n > 0 ∧
  config.intersect_5 < config.n ∧
  config.intersect_9 < config.n ∧
  config.intersect_11 < config.n ∧
  config.intersect_5_count = 5 ∧
  config.intersect_9_count = 9 ∧
  config.intersect_11_count = 11

/-- Theorem stating that 12 is the smallest number of lines satisfying the conditions -/
theorem smallest_valid_configuration :
  (∃ (config : LineConfiguration), is_valid_configuration config ∧ config.n = 12) ∧
  (∀ (config : LineConfiguration), is_valid_configuration config → config.n ≥ 12) :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_configuration_l3602_360224


namespace NUMINAMATH_CALUDE_particle_position_after_2023_minutes_l3602_360284

/-- Represents the position of a particle -/
structure Position where
  x : ℕ
  y : ℕ

/-- Calculates the time taken to complete n squares -/
def time_for_squares (n : ℕ) : ℕ :=
  n^2 + 5*n

/-- Determines the position of the particle after a given time -/
def particle_position (time : ℕ) : Position :=
  sorry

/-- The theorem to be proved -/
theorem particle_position_after_2023_minutes :
  particle_position 2023 = Position.mk 43 43 := by
  sorry

end NUMINAMATH_CALUDE_particle_position_after_2023_minutes_l3602_360284


namespace NUMINAMATH_CALUDE_k_range_for_specific_inequalities_l3602_360232

/-- Given a real number k, this theorem states that if the system of inequalities
    x^2 - x - 2 > 0 and 2x^2 + (2k+5)x + 5k < 0 has {-2} as its only integer solution,
    then k must be in the range [-3, 2). -/
theorem k_range_for_specific_inequalities (k : ℝ) :
  (∀ x : ℤ, (x^2 - x - 2 > 0 ∧ 2*x^2 + (2*k+5)*x + 5*k < 0) ↔ x = -2) →
  -3 ≤ k ∧ k < 2 :=
sorry

end NUMINAMATH_CALUDE_k_range_for_specific_inequalities_l3602_360232


namespace NUMINAMATH_CALUDE_ratio_from_mean_ratio_l3602_360240

theorem ratio_from_mean_ratio (x y : ℝ) (h : x > 0 ∧ y > 0) :
  (x + y) / 2 / Real.sqrt (x * y) = 25 / 24 →
  (x / y = 16 / 9 ∨ x / y = 9 / 16) := by
  sorry

end NUMINAMATH_CALUDE_ratio_from_mean_ratio_l3602_360240


namespace NUMINAMATH_CALUDE_pentagon_extension_l3602_360294

/-- Given a pentagon ABCDE with extended sides, prove the relation between A and A', B', C', D', E' -/
theorem pentagon_extension (A B C D E A' B' C' D' E' : ℝ × ℝ) 
  (h1 : B = (A + A') / 2)
  (h2 : C = (B + B') / 2)
  (h3 : D = (C + C') / 2)
  (h4 : E = (D + D') / 2)
  (h5 : A = (E + E') / 2) :
  A = (1/32 : ℝ) • A' + (1/16 : ℝ) • B' + (1/8 : ℝ) • C' + (1/4 : ℝ) • D' + (1/2 : ℝ) • E' :=
by sorry

end NUMINAMATH_CALUDE_pentagon_extension_l3602_360294


namespace NUMINAMATH_CALUDE_curve_properties_l3602_360298

structure Curve where
  m : ℝ
  n : ℝ
  equation : ℝ → ℝ → Prop

def isEllipse (C : Curve) : Prop := sorry

def hasYAxisFoci (C : Curve) : Prop := sorry

def isHyperbola (C : Curve) : Prop := sorry

def hasAsymptotes (C : Curve) (f : ℝ → ℝ) : Prop := sorry

def isTwoLines (C : Curve) : Prop := sorry

theorem curve_properties (C : Curve) 
  (h_eq : C.equation = fun x y ↦ C.m * x^2 + C.n * y^2 = 1) :
  (C.m > C.n ∧ C.n > 0 → isEllipse C ∧ hasYAxisFoci C) ∧
  (C.m * C.n < 0 → isHyperbola C ∧ hasAsymptotes C (fun x ↦ Real.sqrt (-C.m / C.n) * x)) ∧
  (C.m = 0 ∧ C.n > 0 → isTwoLines C) := by
  sorry

end NUMINAMATH_CALUDE_curve_properties_l3602_360298


namespace NUMINAMATH_CALUDE_expression_always_defined_l3602_360289

theorem expression_always_defined (x : ℝ) : 
  ∃ y : ℝ, y = x^2 / (2*x^2 + 1) :=
by
  sorry

end NUMINAMATH_CALUDE_expression_always_defined_l3602_360289


namespace NUMINAMATH_CALUDE_bianca_birthday_money_l3602_360276

def friend_gifts : List ℝ := [10, 15, 20, 12, 18, 22, 16, 12]
def tax_rate : ℝ := 0.1

theorem bianca_birthday_money :
  let total := friend_gifts.sum
  let tax := total * tax_rate
  total - tax = 112.50 := by sorry

end NUMINAMATH_CALUDE_bianca_birthday_money_l3602_360276


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3602_360258

open Set

def A : Set ℝ := {x | x > 1}
def B : Set ℝ := {x | x ≤ 2}

theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 1 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3602_360258


namespace NUMINAMATH_CALUDE_train_length_l3602_360297

/-- The length of a train given its passing times -/
theorem train_length (pole_time platform_time platform_length : ℝ) 
  (h1 : pole_time = 15)
  (h2 : platform_time = 40)
  (h3 : platform_length = 100) : ℝ :=
by
  -- The length of the train is 60 meters
  exact 60

end NUMINAMATH_CALUDE_train_length_l3602_360297


namespace NUMINAMATH_CALUDE_david_dogs_count_l3602_360219

/-- Given a number of boxes and dogs per box, calculates the total number of dogs. -/
def total_dogs (num_boxes : ℕ) (dogs_per_box : ℕ) : ℕ :=
  num_boxes * dogs_per_box

/-- Theorem stating that 7 boxes with 4 dogs each results in 28 dogs total. -/
theorem david_dogs_count : total_dogs 7 4 = 28 := by
  sorry

end NUMINAMATH_CALUDE_david_dogs_count_l3602_360219


namespace NUMINAMATH_CALUDE_alien_number_conversion_l3602_360255

/-- Converts a base-6 number to base-10 --/
def base6ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

/-- The base-6 representation of the number --/
def alienNumber : List Nat := [4, 5, 1, 2]

theorem alien_number_conversion :
  base6ToBase10 alienNumber = 502 := by
  sorry

#eval base6ToBase10 alienNumber

end NUMINAMATH_CALUDE_alien_number_conversion_l3602_360255


namespace NUMINAMATH_CALUDE_class_distribution_l3602_360223

theorem class_distribution (total : ℕ) (girls boys_carrots boys_apples : ℕ) : 
  total = 33 → 
  girls + boys_carrots + boys_apples = total →
  3 * boys_carrots + boys_apples = girls →
  boys_apples = girls →
  4 * boys_carrots = girls →
  girls = 15 ∧ boys_carrots = 6 ∧ boys_apples = 12 := by
  sorry

end NUMINAMATH_CALUDE_class_distribution_l3602_360223


namespace NUMINAMATH_CALUDE_seven_cupcakes_left_l3602_360234

/-- The number of cupcakes left after eating some from multiple packages -/
def cupcakes_left (packages : ℕ) (cupcakes_per_package : ℕ) (eaten : ℕ) : ℕ :=
  packages * cupcakes_per_package - eaten

/-- Proof that 7 cupcakes are left given the initial conditions -/
theorem seven_cupcakes_left :
  cupcakes_left 3 4 5 = 7 := by
  sorry

end NUMINAMATH_CALUDE_seven_cupcakes_left_l3602_360234


namespace NUMINAMATH_CALUDE_point_distance_on_line_l3602_360287

/-- Given two points on a line, prove that the horizontal distance between them is 3 -/
theorem point_distance_on_line (m n : ℝ) : 
  (m = n / 5 - 2 / 5) → 
  (m + 3 = (n + 15) / 5 - 2 / 5) := by
sorry

end NUMINAMATH_CALUDE_point_distance_on_line_l3602_360287


namespace NUMINAMATH_CALUDE_solution_for_E_l3602_360211

/-- Definition of the function E --/
def E (a b c : ℚ) : ℚ := a * b^2 + c

/-- Theorem stating that -5/8 is the solution to E(a,3,1) = E(a,5,11) --/
theorem solution_for_E : ∃ a : ℚ, E a 3 1 = E a 5 11 ∧ a = -5/8 := by
  sorry

end NUMINAMATH_CALUDE_solution_for_E_l3602_360211


namespace NUMINAMATH_CALUDE_alloy_price_example_l3602_360202

/-- The price of the alloy per kg when two metals are mixed in equal proportions -/
def alloy_price (price1 price2 : ℚ) : ℚ :=
  (price1 + price2) / 2

/-- Theorem: The price of an alloy made from two metals costing 68 and 96 per kg, 
    mixed in equal proportions, is 82 per kg -/
theorem alloy_price_example : alloy_price 68 96 = 82 := by
  sorry

end NUMINAMATH_CALUDE_alloy_price_example_l3602_360202


namespace NUMINAMATH_CALUDE_lawn_mowing_problem_l3602_360286

-- Define the rates and time
def mary_rate : ℚ := 1 / 3
def tom_rate : ℚ := 1 / 5
def work_time : ℚ := 3 / 2

-- Define the theorem
theorem lawn_mowing_problem :
  let combined_rate := mary_rate + tom_rate
  let mowed_fraction := work_time * combined_rate
  1 - mowed_fraction = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_lawn_mowing_problem_l3602_360286


namespace NUMINAMATH_CALUDE_train_speed_problem_l3602_360246

/-- Given a train that covers a distance in 3 hours at its initial speed,
    and covers the same distance in 1 hour at 450 kmph,
    prove that its initial speed is 150 kmph. -/
theorem train_speed_problem (distance : ℝ) (initial_speed : ℝ) : 
  distance = initial_speed * 3 → distance = 450 * 1 → initial_speed = 150 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_problem_l3602_360246


namespace NUMINAMATH_CALUDE_tangent_roots_sum_identity_l3602_360254

theorem tangent_roots_sum_identity (p q : ℝ) (α β : ℝ) :
  (Real.tan α + Real.tan β = -p) →
  (Real.tan α * Real.tan β = q) →
  Real.sin (α + β)^2 + p * Real.sin (α + β) * Real.cos (α + β) + q * Real.cos (α + β)^2 = q := by
  sorry

end NUMINAMATH_CALUDE_tangent_roots_sum_identity_l3602_360254


namespace NUMINAMATH_CALUDE_troll_ratio_l3602_360214

/-- Given the number of trolls in different locations, prove the ratio of trolls in the plains to trolls under the bridge -/
theorem troll_ratio (path bridge plains : ℕ) : 
  path = 6 ∧ 
  bridge = 4 * path - 6 ∧ 
  path + bridge + plains = 33 →
  plains * 2 = bridge := by
sorry

end NUMINAMATH_CALUDE_troll_ratio_l3602_360214


namespace NUMINAMATH_CALUDE_limit_of_a_is_2_l3602_360256

def a (n : ℕ) : ℚ := (4 * n - 3) / (2 * n + 1)

theorem limit_of_a_is_2 : ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a n - 2| < ε := by
  sorry

end NUMINAMATH_CALUDE_limit_of_a_is_2_l3602_360256


namespace NUMINAMATH_CALUDE_x_factor_change_l3602_360242

/-- Given a function q defined in terms of e, x, and z, prove that when e is quadrupled,
    z is tripled, and q is multiplied by 0.2222222222222222, x is doubled. -/
theorem x_factor_change (e x z : ℝ) (h : x ≠ 0) (hz : z ≠ 0) :
  let q := 5 * e / (4 * x * z^2)
  let q' := 0.2222222222222222 * (5 * (4 * e) / (4 * x * (3 * z)^2))
  ∃ x' : ℝ, x' = 2 * x ∧ q' = 5 * (4 * e) / (4 * x' * (3 * z)^2) :=
by sorry

end NUMINAMATH_CALUDE_x_factor_change_l3602_360242


namespace NUMINAMATH_CALUDE_M_equals_N_l3602_360259

def M : Set ℤ := {x | ∃ k : ℤ, x = 2 * k + 1}
def N : Set ℤ := {x | ∃ k : ℤ, x = 4 * k + 1 ∨ x = 4 * k - 1}

theorem M_equals_N : M = N := by sorry

end NUMINAMATH_CALUDE_M_equals_N_l3602_360259


namespace NUMINAMATH_CALUDE_cheese_division_theorem_l3602_360262

theorem cheese_division_theorem (a b c d e f : ℝ) 
  (h_positive : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f)
  (h_order : a < b ∧ b < c ∧ c < d ∧ d < e ∧ e < f) :
  ∃ (S₁ S₂ : Finset ℝ), 
    S₁.card = 3 ∧ 
    S₂.card = 3 ∧ 
    S₁ ∩ S₂ = ∅ ∧ 
    S₁ ∪ S₂ = {a, b, c, d, e, f} ∧
    (S₁.sum id = S₂.sum id) :=
sorry

end NUMINAMATH_CALUDE_cheese_division_theorem_l3602_360262


namespace NUMINAMATH_CALUDE_expression_equals_one_l3602_360281

theorem expression_equals_one :
  (4 * 6) / (12 * 13) * (7 * 12 * 13) / (4 * 6 * 7) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_one_l3602_360281


namespace NUMINAMATH_CALUDE_alissa_presents_l3602_360261

theorem alissa_presents (ethan_presents : ℝ) (difference : ℝ) (alissa_presents : ℝ) : 
  ethan_presents = 31.0 → 
  difference = 22.0 → 
  alissa_presents = ethan_presents - difference → 
  alissa_presents = 9.0 :=
by sorry

end NUMINAMATH_CALUDE_alissa_presents_l3602_360261


namespace NUMINAMATH_CALUDE_fraction_problem_l3602_360273

theorem fraction_problem (N : ℝ) (F : ℝ) : 
  F * (1/4 * N) = 15 ∧ (3/10) * N = 54 → F = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_fraction_problem_l3602_360273


namespace NUMINAMATH_CALUDE_first_aid_station_distance_l3602_360268

theorem first_aid_station_distance 
  (ranger_to_highway : ℝ) 
  (ranger_to_camp : ℝ) 
  (h_ranger_to_highway : ranger_to_highway = 400) 
  (h_ranger_to_camp : ranger_to_camp = 800) : 
  ∃ (first_aid_distance : ℝ), 
    first_aid_distance = 100 * Real.sqrt 28 ∧ 
    first_aid_distance^2 = ranger_to_highway^2 + (ranger_to_camp^2 - ranger_to_highway^2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_first_aid_station_distance_l3602_360268


namespace NUMINAMATH_CALUDE_f_composition_negative_four_l3602_360283

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 + x - 2 else -Real.log x

-- State the theorem
theorem f_composition_negative_four (x : ℝ) : f (f (-4)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_negative_four_l3602_360283


namespace NUMINAMATH_CALUDE_fraction_repetend_l3602_360212

/-- The repetend of the decimal representation of 7/29 -/
def repetend : Nat := 241379

/-- The length of the repetend -/
def repetend_length : Nat := 6

/-- The fraction we're considering -/
def fraction : Rat := 7 / 29

theorem fraction_repetend :
  ∃ (k : ℕ), (fraction * 10^repetend_length - fraction) * 10^k = repetend / (10^repetend_length - 1) :=
sorry

end NUMINAMATH_CALUDE_fraction_repetend_l3602_360212


namespace NUMINAMATH_CALUDE_min_queries_needed_l3602_360239

/-- Represents a quadratic polynomial ax² + bx + c -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Evaluates a quadratic polynomial at a given point -/
def evaluate (p : QuadraticPolynomial) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

/-- Represents Petia's strategy of choosing which polynomial value to return -/
def PetiaStrategy := ℕ → Bool

/-- Represents Vasya's strategy of choosing query points -/
def VasyaStrategy := ℕ → ℝ

/-- Determines if Vasya can identify one of Petia's polynomials after n queries -/
def canIdentifyPolynomial (f g : QuadraticPolynomial) (petiaStrat : PetiaStrategy) (vasyaStrat : VasyaStrategy) (n : ℕ) : Prop :=
  ∃ (i : Fin n), 
    let x := vasyaStrat i
    let y := if petiaStrat i then evaluate f x else evaluate g x
    ∀ (f' g' : QuadraticPolynomial), 
      (∀ (j : Fin n), 
        let x' := vasyaStrat j
        let y' := if petiaStrat j then evaluate f' x' else evaluate g' x'
        y' = if petiaStrat j then evaluate f x' else evaluate g x') →
      f' = f ∨ g' = g

/-- The main theorem: 8 is the smallest number of queries needed -/
theorem min_queries_needed : 
  (∃ (vasyaStrat : VasyaStrategy), ∀ (f g : QuadraticPolynomial) (petiaStrat : PetiaStrategy), 
    canIdentifyPolynomial f g petiaStrat vasyaStrat 8) ∧ 
  (∀ (n : ℕ), n < 8 → 
    ∀ (vasyaStrat : VasyaStrategy), ∃ (f g : QuadraticPolynomial) (petiaStrat : PetiaStrategy), 
      ¬canIdentifyPolynomial f g petiaStrat vasyaStrat n) := by
  sorry

end NUMINAMATH_CALUDE_min_queries_needed_l3602_360239


namespace NUMINAMATH_CALUDE_edwards_earnings_l3602_360227

/-- Edward's lawn mowing business earnings --/
theorem edwards_earnings (summer_earnings : ℕ) (supplies_cost : ℕ) (total_earnings : ℕ)
  (h1 : summer_earnings = 27)
  (h2 : supplies_cost = 5)
  (h3 : total_earnings = 24)
  : ∃ spring_earnings : ℕ,
    spring_earnings + (summer_earnings - supplies_cost) = total_earnings ∧
    spring_earnings = 2 := by
  sorry

end NUMINAMATH_CALUDE_edwards_earnings_l3602_360227


namespace NUMINAMATH_CALUDE_l_triomino_division_l3602_360277

/-- An L-triomino is a shape with 3 squares formed by removing one square from a 2x2 grid. -/
def L_triomino_area : ℕ := 3

/-- Theorem: A 1961 × 1963 grid rectangle cannot be exactly divided into L-triominoes,
    but a 1963 × 1965 rectangle can be exactly divided into L-triominoes. -/
theorem l_triomino_division :
  (¬ (1961 * 1963) % L_triomino_area = 0) ∧
  ((1963 * 1965) % L_triomino_area = 0) := by
  sorry

end NUMINAMATH_CALUDE_l_triomino_division_l3602_360277


namespace NUMINAMATH_CALUDE_natural_numbers_satisfying_condition_l3602_360252

def satisfies_condition (n : ℕ) : Prop :=
  ∃ (q r : ℕ), n = 8 * q + r ∧ r < 8 ∧ q + r = 13

theorem natural_numbers_satisfying_condition :
  {n : ℕ | satisfies_condition n} = {108, 100, 92, 84, 76, 68, 60, 52, 44} :=
by sorry

end NUMINAMATH_CALUDE_natural_numbers_satisfying_condition_l3602_360252


namespace NUMINAMATH_CALUDE_white_marble_probability_l3602_360220

theorem white_marble_probability (total_marbles : ℕ) 
  (p_green p_red_or_blue : ℝ) : 
  total_marbles = 84 →
  p_green = 2 / 7 →
  p_red_or_blue = 0.4642857142857143 →
  1 - (p_green + p_red_or_blue) = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_white_marble_probability_l3602_360220


namespace NUMINAMATH_CALUDE_smallest_angle_solution_l3602_360291

theorem smallest_angle_solution (θ : Real) : 
  (θ > 0) → 
  (θ < 360) → 
  (Real.cos (θ * π / 180) = Real.sin (70 * π / 180) + Real.cos (50 * π / 180) - Real.sin (20 * π / 180) - Real.cos (10 * π / 180)) → 
  (∀ φ, 0 < φ ∧ φ < θ → Real.cos (φ * π / 180) ≠ Real.sin (70 * π / 180) + Real.cos (50 * π / 180) - Real.sin (20 * π / 180) - Real.cos (10 * π / 180)) → 
  θ = 50 := by
sorry

end NUMINAMATH_CALUDE_smallest_angle_solution_l3602_360291


namespace NUMINAMATH_CALUDE_rectangle_area_l3602_360236

/-- The area of a rectangle with an inscribed circle of radius 5 and length-to-width ratio of 2:1 -/
theorem rectangle_area (r : ℝ) (ratio : ℝ) : 
  r = 5 → ratio = 2 → 2 * r * ratio * r = 200 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3602_360236


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l3602_360270

theorem quadratic_one_solution (k : ℝ) : 
  (∃! x : ℝ, (3 * x - 4) * (x + 6) = -53 + k * x) ↔ 
  (k = 14 + 2 * Real.sqrt 87 ∨ k = 14 - 2 * Real.sqrt 87) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l3602_360270


namespace NUMINAMATH_CALUDE_percentage_equality_l3602_360226

theorem percentage_equality : (0.2 * 4 : ℝ) = (0.8 * 1 : ℝ) := by sorry

end NUMINAMATH_CALUDE_percentage_equality_l3602_360226


namespace NUMINAMATH_CALUDE_power_of_power_l3602_360229

theorem power_of_power (a : ℝ) : (a^3)^2 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l3602_360229


namespace NUMINAMATH_CALUDE_prime_square_mod_180_l3602_360264

theorem prime_square_mod_180 (p : Nat) (h_prime : Nat.Prime p) (h_gt_5 : p > 5) :
  ∃ (r₁ r₂ : Nat), r₁ ≠ r₂ ∧ 
  (∀ (r : Nat), p^2 % 180 = r → (r = r₁ ∨ r = r₂)) :=
sorry

end NUMINAMATH_CALUDE_prime_square_mod_180_l3602_360264


namespace NUMINAMATH_CALUDE_quadratic_inequality_problem_l3602_360218

theorem quadratic_inequality_problem (p q : ℝ) :
  (∀ x, x^2 - p*x - q < 0 ↔ 2 < x ∧ x < 3) →
  (p = 5 ∧ q = -6) ∧
  (∀ x, q*x^2 - p*x - 1 > 0 ↔ -1/2 < x ∧ x < -1/3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_problem_l3602_360218


namespace NUMINAMATH_CALUDE_continuous_third_derivative_product_nonnegative_l3602_360280

/-- A real function with continuous third derivative has a point where the product of
    the function value and its first three derivatives is non-negative. -/
theorem continuous_third_derivative_product_nonnegative (f : ℝ → ℝ) 
  (hf : ContDiff ℝ 3 f) :
  ∃ a : ℝ, f a * (deriv f a) * (deriv^[2] f a) * (deriv^[3] f a) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_continuous_third_derivative_product_nonnegative_l3602_360280


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l3602_360235

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (x^2 - 6*x - 22 = 2*x + 18) → 
  (∃ x₁ x₂ : ℝ, (x₁ + x₂ = 8) ∧ (x₁ * x₂ = x₁ + x₂ - 8)) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l3602_360235


namespace NUMINAMATH_CALUDE_dozen_pens_cost_is_600_l3602_360251

/-- The cost of a pen in rupees -/
def pen_cost : ℚ := sorry

/-- The cost of a pencil in rupees -/
def pencil_cost : ℚ := sorry

/-- The cost ratio of a pen to a pencil -/
def cost_ratio : ℚ := 5

/-- The total cost of 3 pens and 5 pencils in rupees -/
def total_cost : ℚ := 200

/-- The cost of one dozen pens in rupees -/
def dozen_pens_cost : ℚ := 12 * pen_cost

theorem dozen_pens_cost_is_600 :
  pen_cost = 5 * pencil_cost ∧
  3 * pen_cost + 5 * pencil_cost = total_cost →
  dozen_pens_cost = 600 := by
  sorry

end NUMINAMATH_CALUDE_dozen_pens_cost_is_600_l3602_360251


namespace NUMINAMATH_CALUDE_remainder_theorem_l3602_360285

-- Define the polynomial
def f (x : ℝ) : ℝ := x^5 - 3*x^3 + x + 5

-- Define the divisor
def g (x : ℝ) : ℝ := (x - 3)^2

-- Theorem statement
theorem remainder_theorem :
  ∃ (q : ℝ → ℝ), ∀ x, f x = g x * q x + 65 :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3602_360285


namespace NUMINAMATH_CALUDE_solution_to_equation_l3602_360230

theorem solution_to_equation : ∃! x : ℤ, (2008 + x)^2 = x^2 ∧ x = -1004 := by sorry

end NUMINAMATH_CALUDE_solution_to_equation_l3602_360230


namespace NUMINAMATH_CALUDE_aaron_final_position_l3602_360237

/-- Represents a point on the coordinate plane -/
structure Point where
  x : Int
  y : Int

/-- Represents a direction -/
inductive Direction
  | East
  | North
  | West
  | South

/-- Represents Aaron's state -/
structure AaronState where
  position : Point
  direction : Direction
  steps : Nat

/-- Defines the rules for Aaron's movement -/
def move (state : AaronState) : AaronState :=
  sorry

/-- Theorem stating Aaron's final position after 100 steps -/
theorem aaron_final_position :
  (move^[100] { position := { x := 0, y := 0 }, direction := Direction.East, steps := 0 }).position = { x := 10, y := 0 } :=
sorry

end NUMINAMATH_CALUDE_aaron_final_position_l3602_360237


namespace NUMINAMATH_CALUDE_inverse_expression_equals_one_fifth_l3602_360295

theorem inverse_expression_equals_one_fifth :
  (2 - 3 * (2 - 3)⁻¹)⁻¹ = (1 : ℝ) / 5 := by
  sorry

end NUMINAMATH_CALUDE_inverse_expression_equals_one_fifth_l3602_360295
