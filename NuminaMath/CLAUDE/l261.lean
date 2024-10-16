import Mathlib

namespace NUMINAMATH_CALUDE_sin_cos_problem_l261_26145

theorem sin_cos_problem (x : ℝ) (h : Real.sin x = 3 * Real.cos x) :
  Real.sin x * Real.cos x = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_problem_l261_26145


namespace NUMINAMATH_CALUDE_largest_among_a_ab_aplusb_l261_26146

theorem largest_among_a_ab_aplusb (a b : ℚ) (h : b < 0) :
  (a - b) = max a (max (a - b) (a + b)) := by
  sorry

end NUMINAMATH_CALUDE_largest_among_a_ab_aplusb_l261_26146


namespace NUMINAMATH_CALUDE_smallest_common_factor_l261_26195

theorem smallest_common_factor : ∃ (n : ℕ), n > 0 ∧ n = 42 ∧ 
  (∀ (m : ℕ), m > 0 ∧ m < n → 
    (∀ (k : ℕ), k > 1 → ¬(k ∣ (11 * m - 3) ∧ k ∣ (8 * m + 4)))) ∧
  (∃ (k : ℕ), k > 1 ∧ k ∣ (11 * n - 3) ∧ k ∣ (8 * n + 4)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_common_factor_l261_26195


namespace NUMINAMATH_CALUDE_line_plane_parallelism_l261_26165

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between a line and a plane
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the parallel relation between two lines
variable (parallel_line_line : Line → Line → Prop)

-- Define the subset relation between a line and a plane
variable (subset_line_plane : Line → Plane → Prop)

-- Define the intersection operation between two planes
variable (intersection_plane_plane : Plane → Plane → Line)

-- State the theorem
theorem line_plane_parallelism 
  (m n : Line) (α β : Plane) 
  (h_diff_lines : m ≠ n) 
  (h_diff_planes : α ≠ β) 
  (h_m_parallel_α : parallel_line_plane m α) 
  (h_m_subset_β : subset_line_plane m β) 
  (h_intersection : intersection_plane_plane α β = n) : 
  parallel_line_line m n :=
sorry

end NUMINAMATH_CALUDE_line_plane_parallelism_l261_26165


namespace NUMINAMATH_CALUDE_ball_count_proof_l261_26115

/-- Given a box with balls where some are red, prove that if there are 12 red balls
    and the probability of drawing a red ball is 0.6, then the total number of balls is 20. -/
theorem ball_count_proof (total_balls : ℕ) (red_balls : ℕ) (prob_red : ℚ) 
    (h1 : red_balls = 12)
    (h2 : prob_red = 6/10)
    (h3 : (red_balls : ℚ) / total_balls = prob_red) : 
  total_balls = 20 := by
  sorry

end NUMINAMATH_CALUDE_ball_count_proof_l261_26115


namespace NUMINAMATH_CALUDE_sum_of_squares_and_squared_sum_l261_26154

theorem sum_of_squares_and_squared_sum : (5 + 9 - 3)^2 + (5^2 + 9^2 + 3^2) = 236 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_and_squared_sum_l261_26154


namespace NUMINAMATH_CALUDE_tangent_line_equation_l261_26174

/-- The equation of the tangent line to the circle x^2 + y^2 - 4x = 0 at the point (1, √3) is x - √3y + 2 = 0. -/
theorem tangent_line_equation (x y : ℝ) :
  let circle_equation := (x^2 + y^2 - 4*x = 0)
  let point_on_circle := (1, Real.sqrt 3)
  let tangent_line := (x - Real.sqrt 3 * y + 2 = 0)
  circle_equation ∧ (x, y) = point_on_circle → tangent_line := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l261_26174


namespace NUMINAMATH_CALUDE_total_birds_l261_26162

-- Define the number of geese
def geese : ℕ := 58

-- Define the number of ducks
def ducks : ℕ := 37

-- Theorem stating the total number of birds
theorem total_birds : geese + ducks = 95 := by
  sorry

end NUMINAMATH_CALUDE_total_birds_l261_26162


namespace NUMINAMATH_CALUDE_unique_seventh_digit_l261_26181

def is_seven_digit (n : ℕ) : Prop := 1000000 ≤ n ∧ n ≤ 9999999

def digit_sum (n : ℕ) : ℕ := sorry

theorem unique_seventh_digit (a b : ℕ) (h1 : is_seven_digit a) (h2 : b = digit_sum a) 
  (h3 : is_seven_digit (a - b)) (h4 : ∃ (d : Fin 7 → ℕ), 
    (∀ i, d i ∈ ({1, 2, 3, 4, 6, 7} : Set ℕ)) ∧ 
    (∃ j, d j ∉ ({1, 2, 3, 4, 6, 7} : Set ℕ)) ∧
    (a - b = d 0 * 1000000 + d 1 * 100000 + d 2 * 10000 + d 3 * 1000 + d 4 * 100 + d 5 * 10 + d 6)) :
  ∃! x, x ∉ ({1, 2, 3, 4, 6, 7} : Set ℕ) ∧ x < 10 ∧ 
    (a - b = x * 1000000 + 1 * 100000 + 2 * 10000 + 3 * 1000 + 4 * 100 + 6 * 10 + 7 ∨
     a - b = 1 * 1000000 + x * 100000 + 2 * 10000 + 3 * 1000 + 4 * 100 + 6 * 10 + 7 ∨
     a - b = 1 * 1000000 + 2 * 100000 + x * 10000 + 3 * 1000 + 4 * 100 + 6 * 10 + 7 ∨
     a - b = 1 * 1000000 + 2 * 100000 + 3 * 10000 + x * 1000 + 4 * 100 + 6 * 10 + 7 ∨
     a - b = 1 * 1000000 + 2 * 100000 + 3 * 10000 + 4 * 1000 + x * 100 + 6 * 10 + 7 ∨
     a - b = 1 * 1000000 + 2 * 100000 + 3 * 10000 + 4 * 1000 + 6 * 100 + x * 10 + 7 ∨
     a - b = 1 * 1000000 + 2 * 100000 + 3 * 10000 + 4 * 1000 + 6 * 100 + 7 * 10 + x) :=
by sorry

end NUMINAMATH_CALUDE_unique_seventh_digit_l261_26181


namespace NUMINAMATH_CALUDE_bells_lcm_l261_26100

/-- The time intervals at which the bells toll -/
def bell_intervals : List ℕ := [5, 8, 11, 15, 20]

/-- The theorem stating that the least common multiple of the bell intervals is 1320 -/
theorem bells_lcm : Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 5 8) 11) 15) 20 = 1320 := by
  sorry

end NUMINAMATH_CALUDE_bells_lcm_l261_26100


namespace NUMINAMATH_CALUDE_lastTwoDigitsOf7To2020_l261_26168

-- Define the function that gives the last two digits of 7^n
def lastTwoDigits (n : ℕ) : ℕ :=
  (7^n) % 100

-- State the periodicity of the last two digits
axiom lastTwoDigitsPeriodicity (n : ℕ) (h : n ≥ 2) : 
  lastTwoDigits n = lastTwoDigits (n % 4 + 4)

-- Define the theorem
theorem lastTwoDigitsOf7To2020 : lastTwoDigits 2020 = 01 := by
  sorry

end NUMINAMATH_CALUDE_lastTwoDigitsOf7To2020_l261_26168


namespace NUMINAMATH_CALUDE_simplify_fraction_l261_26188

theorem simplify_fraction (b : ℝ) (h : b = 5) : 15 * b^4 / (75 * b^3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l261_26188


namespace NUMINAMATH_CALUDE_chris_balls_l261_26193

/-- The number of golf balls in a dozen -/
def balls_per_dozen : ℕ := 12

/-- The number of dozens Dan buys -/
def dan_dozens : ℕ := 5

/-- The number of dozens Gus buys -/
def gus_dozens : ℕ := 2

/-- The total number of golf balls purchased -/
def total_balls : ℕ := 132

/-- Theorem: Chris buys 48 golf balls -/
theorem chris_balls : 
  total_balls - (dan_dozens * balls_per_dozen + gus_dozens * balls_per_dozen) = 48 := by
  sorry

end NUMINAMATH_CALUDE_chris_balls_l261_26193


namespace NUMINAMATH_CALUDE_water_trough_theorem_l261_26124

/-- Calculates the final amount of water in a trough after a given number of days -/
def water_trough_calculation (initial_amount : ℝ) (evaporation_rate : ℝ) (refill_rate : ℝ) (days : ℕ) : ℝ :=
  initial_amount - (evaporation_rate - refill_rate) * days

/-- Theorem stating the final amount of water in the trough after 45 days -/
theorem water_trough_theorem :
  water_trough_calculation 350 1 0.4 45 = 323 := by
  sorry

#eval water_trough_calculation 350 1 0.4 45

end NUMINAMATH_CALUDE_water_trough_theorem_l261_26124


namespace NUMINAMATH_CALUDE_sum_of_sqrt_odd_sums_equals_15_l261_26186

def odd_sum (n : ℕ) : ℕ := n^2

theorem sum_of_sqrt_odd_sums_equals_15 :
  Real.sqrt (odd_sum 1) + Real.sqrt (odd_sum 2) + Real.sqrt (odd_sum 3) + 
  Real.sqrt (odd_sum 4) + Real.sqrt (odd_sum 5) = 15 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_sqrt_odd_sums_equals_15_l261_26186


namespace NUMINAMATH_CALUDE_class_composition_l261_26119

theorem class_composition (girls boys : ℕ) : 
  girls * 6 = boys * 5 →  -- Initial ratio of girls to boys is 5:6
  (girls - 20) * 3 = boys * 2 →  -- New ratio after 20 girls leave is 2:3
  boys = 120 := by  -- The number of boys in the class is 120
sorry

end NUMINAMATH_CALUDE_class_composition_l261_26119


namespace NUMINAMATH_CALUDE_hall_width_to_length_ratio_l261_26103

/-- Represents a rectangular hall -/
structure RectangularHall where
  width : ℝ
  length : ℝ

/-- Properties of the rectangular hall -/
def HallProperties (hall : RectangularHall) : Prop :=
  hall.width > 0 ∧ 
  hall.length > 0 ∧
  hall.width * hall.length = 128 ∧ 
  hall.length - hall.width = 8

theorem hall_width_to_length_ratio 
  (hall : RectangularHall) 
  (h : HallProperties hall) : 
  hall.width / hall.length = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_hall_width_to_length_ratio_l261_26103


namespace NUMINAMATH_CALUDE_triangle_abc_is_obtuse_l261_26131

theorem triangle_abc_is_obtuse (A B C : ℝ) (h1 : A = 2 * B) (h2 : A = 3 * C) 
  (h3 : A + B + C = 180) : A > 90 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_is_obtuse_l261_26131


namespace NUMINAMATH_CALUDE_sunzi_problem_correct_l261_26163

/-- Represents the problem from "The Mathematical Classic of Sunzi" -/
structure SunziProblem where
  x : ℕ  -- Total number of people
  y : ℕ  -- Total number of carriages

/-- Checks if the given numbers satisfy the conditions of the problem -/
def is_valid_solution (p : SunziProblem) : Prop :=
  (p.x / 3 : ℚ) = p.y - 2 ∧ (p.x - 9) / 2 = p.y

/-- The system of equations correctly represents the Sunzi problem -/
theorem sunzi_problem_correct (p : SunziProblem) : 
  is_valid_solution p ↔ 
    (∃ (empty_carriages : ℕ), p.y = p.x / 3 + empty_carriages ∧ empty_carriages = 2) ∧
    (∃ (walking_people : ℕ), p.y = (p.x - walking_people) / 2 ∧ walking_people = 9) :=
sorry

end NUMINAMATH_CALUDE_sunzi_problem_correct_l261_26163


namespace NUMINAMATH_CALUDE_car_city_efficiency_l261_26121

/-- Represents the fuel efficiency of a car -/
structure CarEfficiency where
  highway : ℝ  -- Miles per gallon on highway
  city : ℝ     -- Miles per gallon in city
  tank : ℝ     -- Tank size in gallons

/-- Theorem stating the car's fuel efficiency in the city given the conditions -/
theorem car_city_efficiency (car : CarEfficiency) 
  (highway_distance : car.highway * car.tank = 900)
  (city_distance : car.city * car.tank = 600)
  (efficiency_difference : car.city = car.highway - 5) :
  car.city = 10 := by sorry

end NUMINAMATH_CALUDE_car_city_efficiency_l261_26121


namespace NUMINAMATH_CALUDE_arithmetic_sequence_60th_term_l261_26199

/-- Given an arithmetic sequence with first term 7 and fifteenth term 35,
    prove that the sixtieth term is 125. -/
theorem arithmetic_sequence_60th_term
  (a : ℕ → ℤ)  -- The arithmetic sequence
  (h1 : a 1 = 7)  -- First term is 7
  (h2 : a 15 = 35)  -- Fifteenth term is 35
  (h_arith : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1)  -- Arithmetic sequence property
  : a 60 = 125 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_60th_term_l261_26199


namespace NUMINAMATH_CALUDE_function_range_condition_l261_26177

theorem function_range_condition (a : ℝ) : 
  (∀ x₁ : ℝ, ∃ x₂ : ℝ, |2 * x₁ - a| + |2 * x₁ + 3| = |x₂ - 1| + 2) →
  (a ≥ -1 ∨ a ≤ -5) := by
sorry

end NUMINAMATH_CALUDE_function_range_condition_l261_26177


namespace NUMINAMATH_CALUDE_equidistant_point_y_coordinate_l261_26150

/-- The y-coordinate of the point on the y-axis equidistant from C(-3,0) and D(4,5) is 16/5 -/
theorem equidistant_point_y_coordinate :
  let C : ℝ × ℝ := (-3, 0)
  let D : ℝ × ℝ := (4, 5)
  let P : ℝ → ℝ × ℝ := λ y => (0, y)
  ∃ y : ℝ, (dist (P y) C = dist (P y) D) ∧ (y = 16/5)
  := by sorry

where
  dist : ℝ × ℝ → ℝ × ℝ → ℝ
  | (x₁, y₁), (x₂, y₂) => Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

end NUMINAMATH_CALUDE_equidistant_point_y_coordinate_l261_26150


namespace NUMINAMATH_CALUDE_chantal_profit_l261_26185

def sweater_profit (balls_per_sweater : ℕ) (yarn_cost : ℕ) (sell_price : ℕ) (num_sweaters : ℕ) : ℕ :=
  let total_balls := balls_per_sweater * num_sweaters
  let total_cost := total_balls * yarn_cost
  let total_revenue := sell_price * num_sweaters
  total_revenue - total_cost

theorem chantal_profit :
  sweater_profit 4 6 35 28 = 308 := by
  sorry

end NUMINAMATH_CALUDE_chantal_profit_l261_26185


namespace NUMINAMATH_CALUDE_third_to_fourth_l261_26153

/-- An angle is in the third quadrant if it's between 180° and 270° -/
def is_third_quadrant (α : ℝ) : Prop :=
  ∃ k : ℤ, 180 + 360 * k < α ∧ α < 270 + 360 * k

/-- An angle is in the fourth quadrant if it's between 270° and 360° -/
def is_fourth_quadrant (α : ℝ) : Prop :=
  ∃ k : ℤ, 270 + 360 * k < α ∧ α < 360 + 360 * k

theorem third_to_fourth (α : ℝ) (h : is_third_quadrant α) :
  is_fourth_quadrant (180 - α) :=
by sorry

end NUMINAMATH_CALUDE_third_to_fourth_l261_26153


namespace NUMINAMATH_CALUDE_initial_necklaces_count_l261_26127

theorem initial_necklaces_count (initial_earrings : ℕ) 
  (total_jewelry : ℕ) : 
  initial_earrings = 15 →
  total_jewelry = 57 →
  ∃ (initial_necklaces : ℕ),
    initial_necklaces = 15 ∧
    2 * initial_necklaces + initial_earrings + 
    (2/3 : ℚ) * initial_earrings + 
    (1/5 : ℚ) * ((2/3 : ℚ) * initial_earrings) = total_jewelry :=
by sorry

end NUMINAMATH_CALUDE_initial_necklaces_count_l261_26127


namespace NUMINAMATH_CALUDE_intersection_area_is_450_l261_26112

/-- A cube with edge length 30 units -/
structure Cube :=
  (edge_length : ℝ)
  (edge_length_eq : edge_length = 30)

/-- A point in 3D space -/
structure Point3D :=
  (x y z : ℝ)

/-- A plane in 3D space -/
structure Plane :=
  (a b c d : ℝ)
  (equation : ∀ (p : Point3D), a * p.x + b * p.y + c * p.z = d)

/-- Definition of points P, Q, R on cube edges -/
def points_on_edges (cube : Cube) : Point3D × Point3D × Point3D :=
  (Point3D.mk 10 0 0, Point3D.mk 30 20 0, Point3D.mk 30 30 5)

/-- The plane PQR -/
def plane_PQR (p q r : Point3D) : Plane :=
  sorry

/-- The polygon formed by the intersection of plane PQR with the cube -/
def intersection_polygon (cube : Cube) (plane : Plane) : Set Point3D :=
  sorry

/-- The area of a polygon -/
def polygon_area (polygon : Set Point3D) : ℝ :=
  sorry

/-- Main theorem: The area of the intersection polygon is 450 square units -/
theorem intersection_area_is_450 (cube : Cube) :
  let (p, q, r) := points_on_edges cube
  let plane := plane_PQR p q r
  let polygon := intersection_polygon cube plane
  polygon_area polygon = 450 := by
  sorry

end NUMINAMATH_CALUDE_intersection_area_is_450_l261_26112


namespace NUMINAMATH_CALUDE_calculation_proof_l261_26144

theorem calculation_proof :
  ((-1/3 : ℚ) - 15 + (-2/3) + 1 = -15) ∧
  (16 / (-2)^3 - (-1/8) * (-4 : ℚ) = -5/2) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l261_26144


namespace NUMINAMATH_CALUDE_age_ratio_sachin_rahul_l261_26128

/-- Proves that the ratio of Sachin's age to Rahul's age is 7:9 given their age difference --/
theorem age_ratio_sachin_rahul :
  ∀ (sachin_age rahul_age : ℚ),
    sachin_age = 31.5 →
    rahul_age = sachin_age + 9 →
    ∃ (a b : ℕ), a = 7 ∧ b = 9 ∧ sachin_age / rahul_age = a / b := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_sachin_rahul_l261_26128


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_2_8_plus_5_5_l261_26175

theorem greatest_prime_factor_of_2_8_plus_5_5 :
  ∃ p : ℕ, p.Prime ∧ p ∣ (2^8 + 5^5) ∧ ∀ q : ℕ, q.Prime → q ∣ (2^8 + 5^5) → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_2_8_plus_5_5_l261_26175


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l261_26189

theorem gcd_of_three_numbers : Nat.gcd 9118 (Nat.gcd 12173 33182) = 47 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l261_26189


namespace NUMINAMATH_CALUDE_homework_students_l261_26135

theorem homework_students (total : ℕ) (silent_reading : ℕ) (board_games : ℕ) : 
  total = 24 →
  silent_reading = total / 2 →
  board_games = total / 3 →
  total - (silent_reading + board_games) = 4 := by
sorry

end NUMINAMATH_CALUDE_homework_students_l261_26135


namespace NUMINAMATH_CALUDE_problem_statement_l261_26157

theorem problem_statement : (-1)^53 + 2^(3^4 + 4^3 - 6 * 7) = 2^103 - 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l261_26157


namespace NUMINAMATH_CALUDE_solution_comparison_l261_26173

theorem solution_comparison (a a' b b' : ℝ) (ha : a ≠ 0) (ha' : a' ≠ 0) :
  (-b / a < -b' / a') ↔ (b' / a' < b / a) := by
  sorry

end NUMINAMATH_CALUDE_solution_comparison_l261_26173


namespace NUMINAMATH_CALUDE_problem_solution_l261_26187

theorem problem_solution (x y : ℝ) (hx : x = 12) (hy : y = 7) : 
  (x - y) * (x + y) = 95 ∧ (x + y)^2 = 361 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l261_26187


namespace NUMINAMATH_CALUDE_quadratic_discriminant_zero_l261_26151

theorem quadratic_discriminant_zero (b : ℝ) : 
  (∀ x, 3 * x^2 + 5 * b * x + 7 = 0 → (5 * b)^2 - 4 * 3 * 7 = 0) → 
  b = 2 * Real.sqrt 21 ∨ b = -2 * Real.sqrt 21 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_zero_l261_26151


namespace NUMINAMATH_CALUDE_decreasing_interval_of_f_l261_26160

/-- The cubic function f(x) with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 5 * x^2 + 3 * x - 2

/-- The derivative of f(x) -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 - 10 * x + 3

theorem decreasing_interval_of_f (a : ℝ) :
  (f' a 3 = 0) →
  (∀ x : ℝ, x ∈ Set.Icc (1/3 : ℝ) 3 ↔ f' a x ≤ 0) :=
sorry

end NUMINAMATH_CALUDE_decreasing_interval_of_f_l261_26160


namespace NUMINAMATH_CALUDE_orange_packing_l261_26139

theorem orange_packing (total_oranges : ℕ) (oranges_per_box : ℕ) (h1 : total_oranges = 94) (h2 : oranges_per_box = 8) :
  (total_oranges + oranges_per_box - 1) / oranges_per_box = 12 := by
  sorry

end NUMINAMATH_CALUDE_orange_packing_l261_26139


namespace NUMINAMATH_CALUDE_infinitely_many_planes_through_collinear_points_lines_in_different_planes_may_be_skew_unique_plane_through_parallel_lines_l261_26158

-- Define the basic types
variable (Point Line Plane : Type)

-- Define the relations
variable (on_line : Point → Line → Prop)
variable (on_plane : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- Statement B
theorem infinitely_many_planes_through_collinear_points 
  (A B C : Point) (m : Line) 
  (h1 : on_line A m) (h2 : on_line B m) (h3 : on_line C m) :
  ∃ (P : Set Plane), Infinite P ∧ ∀ p ∈ P, on_plane m p :=
sorry

-- Statement C
theorem lines_in_different_planes_may_be_skew 
  (m n : Line) (α β : Plane) 
  (h1 : on_plane m α) (h2 : on_plane n β) :
  ∃ (skew : Line → Line → Prop), skew m n :=
sorry

-- Statement D
theorem unique_plane_through_parallel_lines 
  (m n : Line) (h : parallel m n) :
  ∃! p : Plane, on_plane m p ∧ on_plane n p :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_planes_through_collinear_points_lines_in_different_planes_may_be_skew_unique_plane_through_parallel_lines_l261_26158


namespace NUMINAMATH_CALUDE_suit_price_calculation_suit_price_proof_l261_26104

theorem suit_price_calculation (original_price : ℝ) 
  (increase_percentage : ℝ) (discount_percentage : ℝ) : ℝ :=
  let increased_price := original_price * (1 + increase_percentage)
  let final_price := increased_price * (1 - discount_percentage)
  final_price

theorem suit_price_proof :
  suit_price_calculation 200 0.3 0.3 = 182 := by
  sorry

end NUMINAMATH_CALUDE_suit_price_calculation_suit_price_proof_l261_26104


namespace NUMINAMATH_CALUDE_system_of_inequalities_solution_equation_solution_expression_evaluation_l261_26136

-- Part 1: System of inequalities
theorem system_of_inequalities_solution (x : ℝ) :
  (x - 4 < 2 * (x - 1) ∧ (1 + 2 * x) / 3 ≥ x) ↔ -2 < x ∧ x ≤ 1 := by sorry

-- Part 2: Equation solution
theorem equation_solution :
  ∃! x : ℝ, (x - 2) / (x - 3) = 2 - 1 / (3 - x) ∧ x = 3 := by sorry

-- Part 3: Expression simplification and evaluation
theorem expression_evaluation (x : ℝ) (h : x = 3) :
  (1 - 1 / (x + 2)) / ((x^2 - 1) / (x + 2)) = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_system_of_inequalities_solution_equation_solution_expression_evaluation_l261_26136


namespace NUMINAMATH_CALUDE_new_crust_flour_amount_l261_26170

/-- The amount of flour per new pie crust when changing the recipe -/
def flour_per_new_crust (original_crusts : ℕ) (original_flour_per_crust : ℚ) 
  (new_crusts : ℕ) : ℚ :=
  (original_crusts : ℚ) * original_flour_per_crust / (new_crusts : ℚ)

/-- Theorem stating that the amount of flour per new pie crust is 1/5 cup -/
theorem new_crust_flour_amount : 
  flour_per_new_crust 40 (1/8) 25 = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_new_crust_flour_amount_l261_26170


namespace NUMINAMATH_CALUDE_cistern_wet_surface_area_l261_26169

/-- Calculates the total wet surface area of a rectangular cistern -/
def totalWetSurfaceArea (length width depth : ℝ) : ℝ :=
  let bottomArea := length * width
  let longSideArea := 2 * depth * length
  let shortSideArea := 2 * depth * width
  bottomArea + longSideArea + shortSideArea

/-- Theorem stating that the total wet surface area of a specific cistern is 68.5 m² -/
theorem cistern_wet_surface_area :
  totalWetSurfaceArea 9 4 1.25 = 68.5 := by
  sorry

#eval totalWetSurfaceArea 9 4 1.25

end NUMINAMATH_CALUDE_cistern_wet_surface_area_l261_26169


namespace NUMINAMATH_CALUDE_mean_salary_proof_l261_26184

def salaries : List ℝ := [1000, 2500, 3100, 3650, 1500, 2000]

theorem mean_salary_proof :
  (salaries.sum / salaries.length : ℝ) = 2458.33 := by
  sorry

end NUMINAMATH_CALUDE_mean_salary_proof_l261_26184


namespace NUMINAMATH_CALUDE_typing_service_problem_l261_26110

/-- Typing service problem -/
theorem typing_service_problem 
  (total_pages : ℕ) 
  (pages_revised_twice : ℕ) 
  (cost_first_typing : ℕ) 
  (cost_per_revision : ℕ) 
  (total_cost : ℕ) 
  (h1 : total_pages = 200)
  (h2 : pages_revised_twice = 20)
  (h3 : cost_first_typing = 5)
  (h4 : cost_per_revision = 3)
  (h5 : total_cost = 1360) :
  ∃ (pages_revised_once : ℕ),
    pages_revised_once = 80 ∧
    total_cost = 
      total_pages * cost_first_typing + 
      pages_revised_once * cost_per_revision + 
      pages_revised_twice * cost_per_revision * 2 :=
by sorry

end NUMINAMATH_CALUDE_typing_service_problem_l261_26110


namespace NUMINAMATH_CALUDE_equation_proof_l261_26116

theorem equation_proof : 578 - 214 = 364 := by sorry

end NUMINAMATH_CALUDE_equation_proof_l261_26116


namespace NUMINAMATH_CALUDE_intersection_fixed_point_l261_26179

noncomputable section

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the line
def line (k n x y : ℝ) : Prop := y = k * x + n

-- Define the intersection points
def intersection (k n x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧ line k n x₁ y₁ ∧ line k n x₂ y₂

-- Main theorem
theorem intersection_fixed_point (k n x₁ y₁ x₂ y₂ : ℝ) 
  (hk : k ≠ 0)
  (h_int : intersection k n x₁ y₁ x₂ y₂)
  (h_slope : 3 * (y₁ / x₁ + y₂ / x₂) = 8 * k) :
  n = 1/2 ∨ n = -1/2 := by
  sorry

end

end NUMINAMATH_CALUDE_intersection_fixed_point_l261_26179


namespace NUMINAMATH_CALUDE_tile_arrangements_count_l261_26194

/-- Represents the number of ways to arrange four tiles in a row using three colors. -/
def tileArrangements : ℕ := 36

/-- The number of positions in the row of tiles. -/
def numPositions : ℕ := 4

/-- The number of available colors. -/
def numColors : ℕ := 3

/-- The number of tiles of the same color that must be used. -/
def sameColorTiles : ℕ := 2

/-- Theorem stating that the number of tile arrangements is 36. -/
theorem tile_arrangements_count :
  (numColors * (Nat.choose numPositions sameColorTiles * Nat.factorial (numPositions - sameColorTiles))) = tileArrangements :=
by sorry

end NUMINAMATH_CALUDE_tile_arrangements_count_l261_26194


namespace NUMINAMATH_CALUDE_balanced_132_l261_26138

def is_balanced (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧  -- Three-digit number
  (n / 100 ≠ (n / 10) % 10) ∧ (n / 100 ≠ n % 10) ∧ ((n / 10) % 10 ≠ n % 10) ∧  -- All digits are different
  n = (10 * (n / 100) + (n / 10) % 10) +
      (10 * (n / 100) + n % 10) +
      (10 * ((n / 10) % 10) + n / 100) +
      (10 * ((n / 10) % 10) + n % 10) +
      (10 * (n % 10) + n / 100) +
      (10 * (n % 10) + (n / 10) % 10)  -- Sum of all possible two-digit numbers

theorem balanced_132 : is_balanced 132 := by
  sorry

end NUMINAMATH_CALUDE_balanced_132_l261_26138


namespace NUMINAMATH_CALUDE_valid_arrangements_count_l261_26118

/-- Represents the number of communities --/
def num_communities : ℕ := 3

/-- Represents the number of students --/
def num_students : ℕ := 4

/-- Represents the total number of arrangements without restrictions --/
def total_arrangements : ℕ := (num_students.choose 2) * (num_communities.factorial)

/-- Represents the number of arrangements where two specific students are in the same community --/
def same_community_arrangements : ℕ := num_communities.factorial

/-- The main theorem stating the number of valid arrangements --/
theorem valid_arrangements_count : 
  total_arrangements - same_community_arrangements = 30 :=
sorry

end NUMINAMATH_CALUDE_valid_arrangements_count_l261_26118


namespace NUMINAMATH_CALUDE_same_color_plate_probability_l261_26198

/-- The probability of selecting two plates of the same color -/
theorem same_color_plate_probability 
  (total_plates : ℕ) 
  (red_plates : ℕ) 
  (blue_plates : ℕ) 
  (h1 : total_plates = red_plates + blue_plates) 
  (h2 : total_plates = 13) 
  (h3 : red_plates = 7) 
  (h4 : blue_plates = 6) : 
  (red_plates.choose 2 + blue_plates.choose 2 : ℚ) / total_plates.choose 2 = 4/9 := by
  sorry

end NUMINAMATH_CALUDE_same_color_plate_probability_l261_26198


namespace NUMINAMATH_CALUDE_expression_equals_two_l261_26125

theorem expression_equals_two : (Real.sqrt 3)^2 + (4 - Real.pi)^0 - |(-3)| + Real.sqrt 2 * Real.cos (π / 4) = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_two_l261_26125


namespace NUMINAMATH_CALUDE_ball_probability_l261_26111

theorem ball_probability (n : ℕ) : 
  (1 : ℕ) + (1 : ℕ) + n > 0 →
  (n : ℚ) / ((1 : ℚ) + (1 : ℚ) + (n : ℚ)) = (1 : ℚ) / (2 : ℚ) →
  n = 2 := by
sorry

end NUMINAMATH_CALUDE_ball_probability_l261_26111


namespace NUMINAMATH_CALUDE_tangent_lines_and_intersection_l261_26178

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 4*y + 1 = 0

-- Define point P
def point_P : ℝ × ℝ := (-3, 2)

-- Define the tangent lines
def tangent_line_1 (x : ℝ) : Prop := x = -3
def tangent_line_2 (x y : ℝ) : Prop := 3*x + 4*y + 1 = 0

-- Define the circle with diameter PC
def circle_PC (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 5

-- Define the line AB
def line_AB (x y : ℝ) : Prop := x - 2*y - 1 = 0

theorem tangent_lines_and_intersection (x y : ℝ) :
  (∀ x y, circle_C x y → (tangent_line_1 x ∨ tangent_line_2 x y)) ∧
  (∃ A B : ℝ × ℝ, 
    circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧
    circle_PC A.1 A.2 ∧ circle_PC B.1 B.2 ∧
    line_AB A.1 A.2 ∧ line_AB B.1 B.2 ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = (8*Real.sqrt 5 / 5)^2) :=
by sorry

end NUMINAMATH_CALUDE_tangent_lines_and_intersection_l261_26178


namespace NUMINAMATH_CALUDE_high_confidence_possible_no_cases_l261_26164

/-- Represents the confidence level of the relationship between smoking and lung cancer -/
def confidence_level : ℝ := 0.99

/-- Represents a sample of smokers -/
def sample_size : ℕ := 100

/-- Represents the possibility of having no lung cancer cases in a sample -/
def possible_no_cases : Prop := true

/-- Theorem stating that despite high confidence in the smoking-lung cancer relationship,
    it's possible to have a sample with no lung cancer cases -/
theorem high_confidence_possible_no_cases :
  confidence_level > 0.99 → possible_no_cases := by sorry

end NUMINAMATH_CALUDE_high_confidence_possible_no_cases_l261_26164


namespace NUMINAMATH_CALUDE_not_prime_special_expression_l261_26126

theorem not_prime_special_expression (n : ℕ) (h : n > 2) :
  ¬ Nat.Prime (n^(n^n) - 4*n^n + 3) := by
  sorry

end NUMINAMATH_CALUDE_not_prime_special_expression_l261_26126


namespace NUMINAMATH_CALUDE_square_sectors_semicircle_differences_l261_26120

/-- Given a square with side length 300 cm, containing two right-angle sectors and a semicircle,
    prove the difference in area between the two shaded regions and the difference in their perimeters. -/
theorem square_sectors_semicircle_differences (π : ℝ) (h_π : π = 3.14) :
  let square_side : ℝ := 300
  let quarter_circle_area : ℝ := 1/4 * π * square_side^2
  let semicircle_area : ℝ := 1/2 * π * (square_side/2)^2
  let quarter_circle_perimeter : ℝ := 1/2 * π * square_side
  let semicircle_perimeter : ℝ := π * square_side/2 + square_side
  let area_difference : ℝ := 2 * quarter_circle_area - square_side^2 - semicircle_area
  let perimeter_difference : ℝ := 2 * quarter_circle_perimeter - semicircle_perimeter
  area_difference = 15975 ∧ perimeter_difference = 485 :=
by sorry

end NUMINAMATH_CALUDE_square_sectors_semicircle_differences_l261_26120


namespace NUMINAMATH_CALUDE_probability_of_specific_draw_l261_26109

/-- The number of blue balls in the bag -/
def blue_balls : ℕ := 8

/-- The number of green balls in the bag -/
def green_balls : ℕ := 7

/-- The number of blue balls to be drawn -/
def blue_draw : ℕ := 3

/-- The number of green balls to be drawn -/
def green_draw : ℕ := 2

/-- The total number of balls in the bag -/
def total_balls : ℕ := blue_balls + green_balls

/-- The total number of balls to be drawn -/
def total_draw : ℕ := blue_draw + green_draw

/-- The probability of drawing 3 blue balls followed by 2 green balls without replacement -/
theorem probability_of_specific_draw :
  (Nat.choose blue_balls blue_draw * Nat.choose green_balls green_draw : ℚ) /
  (Nat.choose total_balls total_draw : ℚ) = 1176 / 3003 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_specific_draw_l261_26109


namespace NUMINAMATH_CALUDE_stock_certificate_tearing_impossible_2002_pieces_l261_26152

theorem stock_certificate_tearing (n : ℕ) : n > 0 → (∃ k : ℕ, n = 1 + 7 * k) ↔ n % 7 = 1 :=
by sorry

theorem impossible_2002_pieces : ¬(∃ k : ℕ, 2002 = 1 + 7 * k) :=
by sorry

end NUMINAMATH_CALUDE_stock_certificate_tearing_impossible_2002_pieces_l261_26152


namespace NUMINAMATH_CALUDE_vector_opposite_direction_l261_26149

/-- Given two vectors a and b in ℝ², where a = (1, -1), |b| = |a|, and b is in the opposite direction of a, prove that b = (-1, 1). -/
theorem vector_opposite_direction (a b : ℝ × ℝ) : 
  a = (1, -1) → 
  ‖b‖ = ‖a‖ → 
  ∃ (k : ℝ), k < 0 ∧ b = k • a → 
  b = (-1, 1) := by
sorry

end NUMINAMATH_CALUDE_vector_opposite_direction_l261_26149


namespace NUMINAMATH_CALUDE_max_value_implies_m_l261_26155

/-- The function f(x) = -x³ + 3x² + 9x + m has a maximum value of 20 on the interval [-2, 2] -/
def f (x m : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x + m

/-- The maximum value of f(x) on the interval [-2, 2] is 20 -/
def has_max_20 (m : ℝ) : Prop :=
  ∃ (x : ℝ), x ∈ Set.Icc (-2) 2 ∧
  f x m = 20 ∧
  ∀ (y : ℝ), y ∈ Set.Icc (-2) 2 → f y m ≤ 20

theorem max_value_implies_m (m : ℝ) :
  has_max_20 m → m = -2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_implies_m_l261_26155


namespace NUMINAMATH_CALUDE_tv_sales_decrease_l261_26102

/-- Proves that a 70% price increase and 36% revenue increase results in a 20% sales decrease -/
theorem tv_sales_decrease (initial_price initial_quantity : ℝ) 
  (initial_price_positive : initial_price > 0)
  (initial_quantity_positive : initial_quantity > 0) : 
  let new_price := 1.7 * initial_price
  let new_revenue := 1.36 * (initial_price * initial_quantity)
  let new_quantity := new_revenue / new_price
  (initial_quantity - new_quantity) / initial_quantity = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_tv_sales_decrease_l261_26102


namespace NUMINAMATH_CALUDE_intersection_y_coordinate_l261_26167

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 2*y

-- Define points P and Q on the parabola
def P : ℝ × ℝ := (4, 8)
def Q : ℝ × ℝ := (-2, 2)

-- Define the tangent lines at P and Q
def tangent_P (x y : ℝ) : Prop := y = 4*x - 8
def tangent_Q (x y : ℝ) : Prop := y = -2*x - 2

-- Define the intersection point A
def A : ℝ × ℝ := (1, -4)

-- Theorem statement
theorem intersection_y_coordinate :
  parabola P.1 P.2 ∧ 
  parabola Q.1 Q.2 ∧ 
  tangent_P A.1 A.2 ∧ 
  tangent_Q A.1 A.2 →
  A.2 = -4 :=
by sorry

end NUMINAMATH_CALUDE_intersection_y_coordinate_l261_26167


namespace NUMINAMATH_CALUDE_coke_drinking_days_l261_26108

/-- Calculates the remaining days to finish drinking Coke -/
def remaining_days (total_volume : ℕ) (daily_consumption : ℕ) (days_consumed : ℕ) : ℕ :=
  (total_volume * 1000 / daily_consumption) - days_consumed

/-- Proves that it takes 7 more days to finish the Coke -/
theorem coke_drinking_days : remaining_days 2 200 3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_coke_drinking_days_l261_26108


namespace NUMINAMATH_CALUDE_chord_length_circle_line_l261_26171

/-- The chord length cut by a circle on a line --/
theorem chord_length_circle_line (x y : ℝ) : 
  let circle := fun x y => x^2 + y^2 - 8*x - 2*y + 1 = 0
  let line := fun x => Real.sqrt 3 * x + 1
  let center := (4, 1)
  let radius := 4
  let distance_center_to_line := 2 * Real.sqrt 3
  true → -- placeholder for the circle and line equations
  2 * Real.sqrt (radius^2 - distance_center_to_line^2) = 4 :=
by sorry

end NUMINAMATH_CALUDE_chord_length_circle_line_l261_26171


namespace NUMINAMATH_CALUDE_sqrt_division_equality_l261_26196

theorem sqrt_division_equality : Real.sqrt 3 / Real.sqrt 5 = Real.sqrt 15 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_division_equality_l261_26196


namespace NUMINAMATH_CALUDE_water_for_chickens_l261_26113

/-- Calculates the amount of water needed for chickens given the total water needed and the water needed for pigs and horses. -/
theorem water_for_chickens 
  (num_pigs : ℕ) 
  (num_horses : ℕ) 
  (water_per_pig : ℕ) 
  (total_water : ℕ) 
  (h1 : num_pigs = 8)
  (h2 : num_horses = 10)
  (h3 : water_per_pig = 3)
  (h4 : total_water = 114) :
  total_water - (num_pigs * water_per_pig + num_horses * (2 * water_per_pig)) = 30 := by
  sorry

#check water_for_chickens

end NUMINAMATH_CALUDE_water_for_chickens_l261_26113


namespace NUMINAMATH_CALUDE_percentage_of_students_passed_l261_26107

/-- Given an examination where 700 students appeared and 455 failed,
    prove that 35% of students passed the examination. -/
theorem percentage_of_students_passed (total : ℕ) (failed : ℕ) (h1 : total = 700) (h2 : failed = 455) :
  (total - failed : ℚ) / total * 100 = 35 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_students_passed_l261_26107


namespace NUMINAMATH_CALUDE_hyperbola_standard_equation_l261_26143

/-- Given a hyperbola with asymptotic equations y = ±√3x and passing through the point (-1, 3),
    its standard equation is y²/6 - x²/2 = 1 -/
theorem hyperbola_standard_equation 
  (asymptote_slope : ℝ) 
  (point : ℝ × ℝ) 
  (h1 : asymptote_slope = Real.sqrt 3) 
  (h2 : point = (-1, 3)) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
    (∀ (x y : ℝ), y^2 / b^2 - x^2 / a^2 = 1 ↔ 
      (y = asymptote_slope * x ∨ y = -asymptote_slope * x) ∧ 
      (point.1^2 / a^2 - point.2^2 / b^2 = -1)) ∧
    a^2 = 2 ∧ b^2 = 6 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_standard_equation_l261_26143


namespace NUMINAMATH_CALUDE_cake_recipe_difference_l261_26141

theorem cake_recipe_difference (total_flour total_sugar flour_added : ℕ) :
  total_flour = 9 →
  total_sugar = 6 →
  flour_added = 2 →
  (total_flour - flour_added) - total_sugar = 1 := by
sorry

end NUMINAMATH_CALUDE_cake_recipe_difference_l261_26141


namespace NUMINAMATH_CALUDE_team_order_l261_26172

/-- Represents the points of a team in a sports league. -/
structure TeamPoints where
  points : ℕ

/-- Represents the points of all teams in the sports league. -/
structure LeaguePoints where
  A : TeamPoints
  B : TeamPoints
  C : TeamPoints
  D : TeamPoints

/-- Defines the conditions given in the problem. -/
def satisfiesConditions (lp : LeaguePoints) : Prop :=
  (lp.A.points + lp.C.points = lp.B.points + lp.D.points) ∧
  (lp.B.points + lp.A.points + 5 ≤ lp.D.points + lp.C.points) ∧
  (lp.B.points + lp.C.points ≥ lp.A.points + lp.D.points + 3)

/-- Defines the correct order of teams based on their points. -/
def correctOrder (lp : LeaguePoints) : Prop :=
  lp.C.points > lp.D.points ∧ lp.D.points > lp.B.points ∧ lp.B.points > lp.A.points

/-- Theorem stating that if the conditions are satisfied, the correct order of teams is C, D, B, A. -/
theorem team_order (lp : LeaguePoints) :
  satisfiesConditions lp → correctOrder lp := by
  sorry


end NUMINAMATH_CALUDE_team_order_l261_26172


namespace NUMINAMATH_CALUDE_computers_waiting_for_parts_l261_26197

theorem computers_waiting_for_parts (total : ℕ) (unfixable_percent : ℚ) (fixed_immediately : ℕ) : 
  total = 20 →
  unfixable_percent = 1/5 →
  fixed_immediately = 8 →
  (total - (unfixable_percent * total).num - fixed_immediately : ℚ) / total = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_computers_waiting_for_parts_l261_26197


namespace NUMINAMATH_CALUDE_vampire_consumption_l261_26130

/-- Represents the number of people consumed by the vampire and werewolf -/
structure Consumption where
  vampire : ℕ
  werewolf : ℕ

/-- The total consumption over a given number of weeks -/
def total_consumption (c : Consumption) (weeks : ℕ) : ℕ :=
  weeks * (c.vampire + c.werewolf)

theorem vampire_consumption (village_population : ℕ) (duration_weeks : ℕ) (c : Consumption) :
  village_population = 72 →
  duration_weeks = 9 →
  c.werewolf = 5 →
  total_consumption c duration_weeks = village_population →
  c.vampire = 3 := by
  sorry

end NUMINAMATH_CALUDE_vampire_consumption_l261_26130


namespace NUMINAMATH_CALUDE_license_plate_count_l261_26134

/-- Represents the number of possible digits (0-9) -/
def num_digits : ℕ := 10

/-- Represents the number of possible letters (A-Z) -/
def num_letters : ℕ := 26

/-- Represents the number of digits in the license plate -/
def digit_count : ℕ := 5

/-- Represents the number of letters in the license plate -/
def letter_count : ℕ := 3

/-- Represents the total number of characters in the license plate -/
def total_chars : ℕ := digit_count + letter_count

/-- Calculates the number of ways to arrange the letter block within the license plate -/
def letter_block_positions : ℕ := total_chars - letter_count + 1

/-- Calculates the number of valid letter combinations (at least one 'A') -/
def valid_letter_combinations : ℕ := 3 * num_letters^2

/-- The main theorem stating the total number of distinct license plates -/
theorem license_plate_count : 
  letter_block_positions * num_digits^digit_count * valid_letter_combinations = 1216800000 := by
  sorry


end NUMINAMATH_CALUDE_license_plate_count_l261_26134


namespace NUMINAMATH_CALUDE_map_scale_calculation_l261_26123

/-- Given a map where 15 cm represents 90 km, prove that 20 cm represents 120 km. -/
theorem map_scale_calculation (map_cm : ℝ) (real_km : ℝ) (h : map_cm = 15 ∧ real_km = 90) :
  (20 * real_km) / map_cm = 120 :=
by sorry

end NUMINAMATH_CALUDE_map_scale_calculation_l261_26123


namespace NUMINAMATH_CALUDE_marksman_hit_rate_l261_26101

theorem marksman_hit_rate (p : ℝ) : 
  (0 ≤ p ∧ p ≤ 1) →  -- p is a probability
  (1 - (1 - p)^4 = 80/81) →  -- probability of hitting at least once in 4 shots
  p = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_marksman_hit_rate_l261_26101


namespace NUMINAMATH_CALUDE_kite_coefficient_sum_l261_26129

/-- Represents a parabola in the form y = ax^2 + c -/
structure Parabola where
  a : ℝ
  c : ℝ

/-- Represents a kite formed by the intersection of two parabolas with the coordinate axes -/
structure Kite where
  p1 : Parabola
  p2 : Parabola
  area : ℝ

/-- The sum of coefficients a and b for two parabolas forming a kite with area 12 -/
def coefficient_sum (k : Kite) : ℝ := k.p1.a + (-k.p2.a)

/-- Theorem stating that the sum of coefficients a and b is 1.5 for the given conditions -/
theorem kite_coefficient_sum :
  ∀ (k : Kite),
    k.p1.c = -2 ∧ 
    k.p2.c = 4 ∧ 
    k.area = 12 →
    coefficient_sum k = 1.5 := by
  sorry


end NUMINAMATH_CALUDE_kite_coefficient_sum_l261_26129


namespace NUMINAMATH_CALUDE_woman_birth_year_l261_26182

theorem woman_birth_year (x : ℕ) (h1 : x > 0) (h2 : x^2 - x ≥ 1950) (h3 : x^2 - x < 2000) (h4 : x^2 ≥ 2000) : x^2 - x = 1980 := by
  sorry

end NUMINAMATH_CALUDE_woman_birth_year_l261_26182


namespace NUMINAMATH_CALUDE_coloring_books_distribution_l261_26106

theorem coloring_books_distribution (initial_stock : ℕ) (books_sold : ℕ) (num_shelves : ℕ) 
  (h1 : initial_stock = 27)
  (h2 : books_sold = 6)
  (h3 : num_shelves = 3) :
  (initial_stock - books_sold) / num_shelves = 7 := by
  sorry

end NUMINAMATH_CALUDE_coloring_books_distribution_l261_26106


namespace NUMINAMATH_CALUDE_puzzle_pieces_l261_26159

theorem puzzle_pieces (total_pieces : ℕ) (edge_difference : ℕ) (non_red_decrease : ℕ) : 
  total_pieces = 91 → 
  edge_difference = 24 → 
  non_red_decrease = 2 → 
  ∃ (red_pieces : ℕ) (non_red_pieces : ℕ), 
    red_pieces + non_red_pieces = total_pieces ∧ 
    non_red_pieces * non_red_decrease = edge_difference ∧
    red_pieces = 79 :=
by sorry

end NUMINAMATH_CALUDE_puzzle_pieces_l261_26159


namespace NUMINAMATH_CALUDE_equation_solutions_l261_26176

theorem equation_solutions : 
  (∃ (s₁ : Set ℝ), s₁ = {x : ℝ | x^2 - 4*x = 0} ∧ s₁ = {0, 4}) ∧
  (∃ (s₂ : Set ℝ), s₂ = {x : ℝ | x^2 = -2*x + 3} ∧ s₂ = {-3, 1}) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l261_26176


namespace NUMINAMATH_CALUDE_calligraphy_class_equation_l261_26142

theorem calligraphy_class_equation (x : ℕ+) 
  (h1 : 6 * x = planned + 7)
  (h2 : 5 * x = planned - 13)
  : 6 * x - 7 = 5 * x + 13 := by
  sorry

end NUMINAMATH_CALUDE_calligraphy_class_equation_l261_26142


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l261_26122

theorem quadratic_inequality_solution_set :
  {x : ℝ | 2 * x^2 - 3 * x - 2 > 0} = {x : ℝ | x < -1/2 ∨ x > 2} := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l261_26122


namespace NUMINAMATH_CALUDE_common_difference_unique_l261_26140

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_correct : ∀ n, S n = (n : ℝ) * (a 1 + a n) / 2

/-- The common difference of an arithmetic sequence -/
def common_difference (seq : ArithmeticSequence) : ℝ :=
  seq.a 2 - seq.a 1

theorem common_difference_unique
  (seq : ArithmeticSequence)
  (h1 : seq.a 3 = 3)
  (h2 : seq.S 4 = 14) :
  common_difference seq = -1 := by
  sorry

end NUMINAMATH_CALUDE_common_difference_unique_l261_26140


namespace NUMINAMATH_CALUDE_increase_amount_l261_26132

theorem increase_amount (x : ℝ) (amount : ℝ) (h : 15 * x + amount = 14) :
  amount = 14 - 14/15 := by
  sorry

end NUMINAMATH_CALUDE_increase_amount_l261_26132


namespace NUMINAMATH_CALUDE_fat_per_cup_of_rice_l261_26117

/-- Amount of rice eaten in the morning -/
def morning_rice : ℕ := 3

/-- Amount of rice eaten in the afternoon -/
def afternoon_rice : ℕ := 2

/-- Amount of rice eaten in the evening -/
def evening_rice : ℕ := 5

/-- Number of days in a week -/
def days_in_week : ℕ := 7

/-- Total fat intake from rice in a week (in grams) -/
def weekly_fat_intake : ℕ := 700

/-- Calculate the amount of fat in a cup of rice -/
theorem fat_per_cup_of_rice : 
  (weekly_fat_intake : ℚ) / ((morning_rice + afternoon_rice + evening_rice) * days_in_week) = 10 := by
  sorry

end NUMINAMATH_CALUDE_fat_per_cup_of_rice_l261_26117


namespace NUMINAMATH_CALUDE_linear_equation_solution_l261_26133

theorem linear_equation_solution (m : ℝ) : 
  (∃ x : ℝ, 2 * x + m = 5 ∧ x = 1) → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l261_26133


namespace NUMINAMATH_CALUDE_factor_expression_l261_26183

theorem factor_expression (x : ℝ) : 63 * x + 28 = 7 * (9 * x + 4) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l261_26183


namespace NUMINAMATH_CALUDE_system_solutions_l261_26161

/-- The system of equations has only two solutions -/
theorem system_solutions (x y z : ℝ) : 
  (2 * x^2 / (1 + x^2) = y ∧ 
   2 * y^2 / (1 + y^2) = z ∧ 
   2 * z^2 / (1 + z^2) = x) → 
  ((x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1 ∧ y = 1 ∧ z = 1)) :=
by sorry

end NUMINAMATH_CALUDE_system_solutions_l261_26161


namespace NUMINAMATH_CALUDE_radish_distribution_l261_26156

theorem radish_distribution (total : ℕ) (difference : ℕ) : 
  total = 88 → difference = 14 → ∃ (first second : ℕ), 
    first + second = total ∧ 
    second = first + difference ∧ 
    first = 37 := by
  sorry

end NUMINAMATH_CALUDE_radish_distribution_l261_26156


namespace NUMINAMATH_CALUDE_min_value_shifted_l261_26191

/-- The function f(x) = x^2 + 4x + 5 - c -/
def f (c : ℝ) (x : ℝ) : ℝ := x^2 + 4*x + 5 - c

/-- The shifted function f(x-2009) -/
def f_shifted (c : ℝ) (x : ℝ) : ℝ := f c (x - 2009)

theorem min_value_shifted (c : ℝ) :
  (∃ (m : ℝ), ∀ (x : ℝ), f c x ≥ m ∧ (∃ (x_0 : ℝ), f c x_0 = m) ∧ m = 2) →
  (∃ (m : ℝ), ∀ (x : ℝ), f_shifted c x ≥ m ∧ (∃ (x_0 : ℝ), f_shifted c x_0 = m) ∧ m = 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_shifted_l261_26191


namespace NUMINAMATH_CALUDE_weightlifter_fourth_minute_l261_26192

/-- Calculates the total weight a weightlifter can lift in the 4th minute given initial weights,
    weight increments, and fatigue factor. -/
def weightLifterFourthMinute (leftInitial rightInitial leftIncrement rightIncrement fatigueDecline : ℕ) : ℕ :=
  let leftAfterThree := leftInitial + 3 * leftIncrement
  let rightAfterThree := rightInitial + 3 * rightIncrement
  let totalAfterThree := leftAfterThree + rightAfterThree
  totalAfterThree - fatigueDecline

/-- Theorem stating that the weightlifter can lift 55 pounds in the 4th minute under given conditions. -/
theorem weightlifter_fourth_minute :
  weightLifterFourthMinute 12 18 4 6 5 = 55 := by
  sorry

end NUMINAMATH_CALUDE_weightlifter_fourth_minute_l261_26192


namespace NUMINAMATH_CALUDE_function_inequalities_l261_26148

noncomputable section

variable (a : ℝ)
variable (x : ℝ)

def f (x : ℝ) : ℝ := a^(3*x + 1)
def g (x : ℝ) : ℝ := (1/a)^(5*x - 2)

theorem function_inequalities (h1 : a > 0) (h2 : a ≠ 1) :
  (0 < a ∧ a < 1 → (f a x < 1 ↔ x > -1/3)) ∧
  ((0 < a ∧ a < 1 → (f a x ≥ g a x ↔ x ≤ 1/8)) ∧
   (a > 1 → (f a x ≥ g a x ↔ x ≥ 1/8))) := by
  sorry

end

end NUMINAMATH_CALUDE_function_inequalities_l261_26148


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l261_26180

theorem fraction_to_decimal : (7 : ℚ) / 125 = (56 : ℚ) / 1000 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l261_26180


namespace NUMINAMATH_CALUDE_diameter_is_chord_l261_26114

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a chord
def isChord (c : Circle) (p q : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2 ∧
  (q.1 - c.center.1)^2 + (q.2 - c.center.2)^2 = c.radius^2

-- Define a diameter
def isDiameter (c : Circle) (p q : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2 ∧
  (q.1 - c.center.1)^2 + (q.2 - c.center.2)^2 = c.radius^2 ∧
  (p.1 - q.1)^2 + (p.2 - q.2)^2 = 4 * c.radius^2

-- Theorem: A diameter is a chord
theorem diameter_is_chord (c : Circle) (p q : ℝ × ℝ) :
  isDiameter c p q → isChord c p q :=
by
  sorry


end NUMINAMATH_CALUDE_diameter_is_chord_l261_26114


namespace NUMINAMATH_CALUDE_water_depth_in_specific_tank_l261_26137

/-- Represents a horizontal cylindrical water tank -/
structure CylindricalTank where
  length : ℝ
  diameter : ℝ

/-- Calculates the possible depths of water in a cylindrical tank given the surface area -/
def water_depth (tank : CylindricalTank) (surface_area : ℝ) : Set ℝ :=
  { h : ℝ | ∃ (c : ℝ), 
    c = surface_area / tank.length ∧ 
    c = 2 * Real.sqrt (tank.diameter * h - h^2) ∧ 
    0 < h ∧ 
    h < tank.diameter }

/-- Theorem stating the possible depths of water in a specific cylindrical tank -/
theorem water_depth_in_specific_tank : 
  let tank : CylindricalTank := { length := 12, diameter := 8 }
  let surface_area := 48
  water_depth tank surface_area = {4 - 2 * Real.sqrt 3, 4 + 2 * Real.sqrt 3} :=
by
  sorry

end NUMINAMATH_CALUDE_water_depth_in_specific_tank_l261_26137


namespace NUMINAMATH_CALUDE_symmetric_points_values_l261_26147

/-- Two points are symmetric about the y-axis if their x-coordinates are opposite and their y-coordinates are equal -/
def symmetric_about_y_axis (x1 y1 x2 y2 : ℝ) : Prop :=
  x1 = -x2 ∧ y1 = y2

theorem symmetric_points_values :
  ∀ m n : ℝ,
  symmetric_about_y_axis (-3) (2*m - 1) (n + 1) 4 →
  m = 2.5 ∧ n = 2 :=
by sorry

end NUMINAMATH_CALUDE_symmetric_points_values_l261_26147


namespace NUMINAMATH_CALUDE_mary_james_seating_probability_l261_26190

-- Define the number of chairs
def total_chairs : ℕ := 10

-- Define the number of available chairs (excluding first and last)
def available_chairs : ℕ := total_chairs - 2

-- Define the probability of not sitting next to each other
def prob_not_adjacent : ℚ := 3/4

theorem mary_james_seating_probability :
  (1 : ℚ) - (available_chairs - 1 : ℚ) / (available_chairs.choose 2 : ℚ) = prob_not_adjacent :=
by sorry

end NUMINAMATH_CALUDE_mary_james_seating_probability_l261_26190


namespace NUMINAMATH_CALUDE_angle_sum_in_hexagon_with_triangles_l261_26105

/-- Represents a hexagon with two connected triangles -/
structure HexagonWithTriangles where
  /-- Angle A of the hexagon -/
  angle_A : ℝ
  /-- Angle B of the hexagon -/
  angle_B : ℝ
  /-- Angle C of one of the connected triangles -/
  angle_C : ℝ
  /-- An angle x in the figure -/
  x : ℝ
  /-- An angle y in the figure -/
  y : ℝ
  /-- The sum of angles in a hexagon is 720° -/
  hexagon_sum : angle_A + angle_B + (360 - x) + 90 + (114 - y) = 720

/-- Theorem stating that x + y = 50° in the given hexagon with triangles -/
theorem angle_sum_in_hexagon_with_triangles (h : HexagonWithTriangles)
    (h_A : h.angle_A = 30)
    (h_B : h.angle_B = 76)
    (h_C : h.angle_C = 24) :
    h.x + h.y = 50 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_in_hexagon_with_triangles_l261_26105


namespace NUMINAMATH_CALUDE_a_range_characterization_l261_26166

/-- Proposition p: The domain of the logarithm function is all real numbers -/
def prop_p (a : ℝ) : Prop := ∀ x : ℝ, x^2 - 2*a*x + 7*a - 6 > 0

/-- Proposition q: There exists a real x satisfying the quadratic inequality -/
def prop_q (a : ℝ) : Prop := ∃ x : ℝ, x^2 - a*x + 4 < 0

/-- The set of real numbers a where either p is true and q is false, or p is false and q is true -/
def a_range : Set ℝ := {a | (prop_p a ∧ ¬prop_q a) ∨ (¬prop_p a ∧ prop_q a)}

theorem a_range_characterization : 
  a_range = {a | a < -4 ∨ (1 < a ∧ a ≤ 4) ∨ 6 ≤ a} :=
sorry

end NUMINAMATH_CALUDE_a_range_characterization_l261_26166
