import Mathlib

namespace NUMINAMATH_CALUDE_difference_between_point_eight_and_half_l3601_360168

theorem difference_between_point_eight_and_half : 0.8 - (1/2 : ℚ) = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_difference_between_point_eight_and_half_l3601_360168


namespace NUMINAMATH_CALUDE_complex_multiplication_l3601_360197

theorem complex_multiplication (i : ℂ) : i * i = -1 → i * (2 * i - 1) = -2 - i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l3601_360197


namespace NUMINAMATH_CALUDE_increasing_f_implies_k_leq_one_l3601_360119

-- Define the function f
def f (k : ℝ) (x : ℝ) : ℝ := x^2 - 2*k*x + 1

-- State the theorem
theorem increasing_f_implies_k_leq_one :
  ∀ k : ℝ, (∀ x y : ℝ, 1 ≤ x ∧ x < y ∧ y ≤ 3 → f k x < f k y) → k ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_increasing_f_implies_k_leq_one_l3601_360119


namespace NUMINAMATH_CALUDE_f_derivative_at_2_l3601_360134

-- Define the function f
def f (f'2 : ℝ) : ℝ → ℝ := λ x ↦ 3 * x^2 - 2 * x * f'2

-- State the theorem
theorem f_derivative_at_2 : 
  ∃ f'2 : ℝ, (deriv (f f'2)) 2 = 4 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_2_l3601_360134


namespace NUMINAMATH_CALUDE_steven_route_count_l3601_360192

def central_park_routes : ℕ := 
  let home_to_sw_corner := (Nat.choose 5 2)
  let ne_corner_to_office := (Nat.choose 6 3)
  let park_diagonals := 2
  home_to_sw_corner * park_diagonals * ne_corner_to_office

theorem steven_route_count : central_park_routes = 400 := by
  sorry

end NUMINAMATH_CALUDE_steven_route_count_l3601_360192


namespace NUMINAMATH_CALUDE_total_money_problem_l3601_360184

theorem total_money_problem (brad : ℝ) (josh : ℝ) (doug : ℝ) 
  (h1 : brad = 12.000000000000002)
  (h2 : josh = 2 * brad)
  (h3 : josh = (3/4) * doug) : 
  brad + josh + doug = 68.00000000000001 := by
  sorry

end NUMINAMATH_CALUDE_total_money_problem_l3601_360184


namespace NUMINAMATH_CALUDE_benny_pie_price_l3601_360175

/-- Calculates the price per pie needed to achieve a desired profit given the number and cost of pumpkin and cherry pies -/
def price_per_pie (Np Nc : ℕ) (Cp Cc Pr : ℚ) : ℚ :=
  (Np * Cp + Nc * Cc + Pr) / (Np + Nc)

theorem benny_pie_price :
  let Np : ℕ := 10  -- Number of pumpkin pies
  let Nc : ℕ := 12  -- Number of cherry pies
  let Cp : ℚ := 3   -- Cost to make each pumpkin pie
  let Cc : ℚ := 5   -- Cost to make each cherry pie
  let Pr : ℚ := 20  -- Desired profit
  price_per_pie Np Nc Cp Cc Pr = 5 := by
sorry

end NUMINAMATH_CALUDE_benny_pie_price_l3601_360175


namespace NUMINAMATH_CALUDE_distribute_six_balls_three_boxes_l3601_360158

/-- The number of ways to distribute n distinguishable balls into k distinguishable boxes 
    such that each box contains at least one ball -/
def distribute_balls (n k : ℕ) : ℕ :=
  sorry

/-- The main theorem stating that there are 540 ways to distribute 6 distinguishable balls
    into 3 distinguishable boxes such that each box contains at least one ball -/
theorem distribute_six_balls_three_boxes : distribute_balls 6 3 = 540 := by
  sorry

end NUMINAMATH_CALUDE_distribute_six_balls_three_boxes_l3601_360158


namespace NUMINAMATH_CALUDE_tarantula_perimeter_is_16_l3601_360140

/-- Represents a rectangle with width and height in inches -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents the tarantula-shaped figure -/
structure TarantulaShape where
  body : Rectangle
  legs : Rectangle

/-- Calculates the perimeter of the tarantula-shaped figure -/
def tarantulaPerimeter (t : TarantulaShape) : ℝ :=
  2 * (t.body.width + t.body.height)

theorem tarantula_perimeter_is_16 :
  ∀ t : TarantulaShape,
    t.body.width = 3 ∧
    t.body.height = 10 ∧
    t.legs.width = 5 ∧
    t.legs.height = 3 →
    tarantulaPerimeter t = 16 := by
  sorry

#check tarantula_perimeter_is_16

end NUMINAMATH_CALUDE_tarantula_perimeter_is_16_l3601_360140


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l3601_360152

theorem simplify_trig_expression :
  let tan60 : ℝ := Real.sqrt 3
  let cot60 : ℝ := 1 / Real.sqrt 3
  (tan60^3 + cot60^3) / (tan60 + cot60) = 7/3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l3601_360152


namespace NUMINAMATH_CALUDE_curve_classification_l3601_360115

-- Define the curve equation
def curve_equation (x y m : ℝ) : Prop := 3 * x^2 + m * y^2 = 1

-- Define the possible curve types
inductive CurveType
  | TwoLines
  | Ellipse
  | Circle
  | Hyperbola

-- Theorem statement
theorem curve_classification (m : ℝ) : 
  ∃ (t : CurveType), ∀ (x y : ℝ), curve_equation x y m → 
    (t = CurveType.TwoLines ∨ 
     t = CurveType.Ellipse ∨ 
     t = CurveType.Circle ∨ 
     t = CurveType.Hyperbola) :=
sorry

end NUMINAMATH_CALUDE_curve_classification_l3601_360115


namespace NUMINAMATH_CALUDE_trumpet_section_fraction_l3601_360146

/-- The fraction of students in the trumpet section -/
def trumpet_fraction : ℝ := sorry

/-- The fraction of students in the trombone section -/
def trombone_fraction : ℝ := 0.12

/-- The fraction of students in either the trumpet or trombone section -/
def trumpet_or_trombone_fraction : ℝ := 0.63

theorem trumpet_section_fraction :
  trumpet_fraction = 0.51 :=
by
  sorry

end NUMINAMATH_CALUDE_trumpet_section_fraction_l3601_360146


namespace NUMINAMATH_CALUDE_reciprocal_sum_contains_two_l3601_360137

theorem reciprocal_sum_contains_two (a b c d : ℕ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 →
  (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c + (1 : ℚ) / d = 1 →
  a = 2 ∨ b = 2 ∨ c = 2 ∨ d = 2 := by
sorry

end NUMINAMATH_CALUDE_reciprocal_sum_contains_two_l3601_360137


namespace NUMINAMATH_CALUDE_daily_rental_cost_l3601_360127

/-- Represents the daily car rental cost problem -/
theorem daily_rental_cost (total_cost : ℝ) (miles_driven : ℝ) (per_mile_rate : ℝ) :
  total_cost = 46.12 ∧
  miles_driven = 214.0 ∧
  per_mile_rate = 0.08 →
  ∃ (daily_rate : ℝ), daily_rate = 29.00 ∧ total_cost = daily_rate + miles_driven * per_mile_rate :=
by sorry

end NUMINAMATH_CALUDE_daily_rental_cost_l3601_360127


namespace NUMINAMATH_CALUDE_lcm_gcd_problem_l3601_360104

theorem lcm_gcd_problem (a b : ℕ+) : 
  Nat.lcm a b = 5040 → 
  Nat.gcd a b = 24 → 
  a = 240 → 
  b = 504 := by sorry

end NUMINAMATH_CALUDE_lcm_gcd_problem_l3601_360104


namespace NUMINAMATH_CALUDE_four_points_no_obtuse_triangle_l3601_360195

noncomputable def probability_no_obtuse_triangle (n : ℕ) : ℝ :=
  sorry

theorem four_points_no_obtuse_triangle :
  probability_no_obtuse_triangle 4 = 3 / 32 :=
sorry

end NUMINAMATH_CALUDE_four_points_no_obtuse_triangle_l3601_360195


namespace NUMINAMATH_CALUDE_max_value_product_l3601_360148

theorem max_value_product (x y z : ℝ) (h_pos : x > 0 ∧ y > 0 ∧ z > 0) (h_sum : x + 2*y + 3*z = 1) :
  x^2 * y^2 * z ≤ 4/16807 ∧ ∃ x y z, x > 0 ∧ y > 0 ∧ z > 0 ∧ x + 2*y + 3*z = 1 ∧ x^2 * y^2 * z = 4/16807 :=
by sorry

end NUMINAMATH_CALUDE_max_value_product_l3601_360148


namespace NUMINAMATH_CALUDE_expected_boy_girl_pairs_l3601_360124

/-- The expected number of adjacent boy-girl pairs when 10 boys and 14 girls
    are seated randomly around a circular table with 24 seats. -/
theorem expected_boy_girl_pairs :
  let num_boys : ℕ := 10
  let num_girls : ℕ := 14
  let total_seats : ℕ := 24
  let prob_boy_girl : ℚ := (num_boys : ℚ) * num_girls / (total_seats * (total_seats - 1))
  let prob_girl_boy : ℚ := (num_girls : ℚ) * num_boys / (total_seats * (total_seats - 1))
  let prob_adjacent_pair : ℚ := prob_boy_girl + prob_girl_boy
  let expected_pairs : ℚ := (total_seats : ℚ) * prob_adjacent_pair
  expected_pairs = 280 / 23 :=
by sorry

end NUMINAMATH_CALUDE_expected_boy_girl_pairs_l3601_360124


namespace NUMINAMATH_CALUDE_reading_time_calculation_l3601_360101

theorem reading_time_calculation (pages_book1 pages_book2 : ℕ) 
  (time_book1 time_book2 : ℚ) (pages_to_read1 pages_to_read2 : ℕ) :
  pages_book1 = 4 →
  pages_book2 = 5 →
  time_book1 = 2 →
  time_book2 = 3 →
  pages_to_read1 = 36 →
  pages_to_read2 = 25 →
  (pages_to_read1 * (time_book1 / pages_book1) + 
   pages_to_read2 * (time_book2 / pages_book2)) = 33 :=
by
  sorry

end NUMINAMATH_CALUDE_reading_time_calculation_l3601_360101


namespace NUMINAMATH_CALUDE_angle_b_measure_l3601_360110

/-- An isosceles triangle with specific angle relationships -/
structure IsoscelesTriangle where
  A : Real
  B : Real
  C : Real
  sum_angles : A + B + C = 180
  isosceles : B = C
  angle_relation : C = 3 * A

/-- Theorem: In the given isosceles triangle, angle B measures 540/7 degrees -/
theorem angle_b_measure (t : IsoscelesTriangle) : t.B = 540 / 7 := by
  sorry

end NUMINAMATH_CALUDE_angle_b_measure_l3601_360110


namespace NUMINAMATH_CALUDE_fifteen_people_on_boats_l3601_360139

/-- Given a lake with boats and people, calculate the total number of people on the boats. -/
def total_people_on_boats (num_boats : ℕ) (people_per_boat : ℕ) : ℕ :=
  num_boats * people_per_boat

/-- Theorem: In a lake with 5 boats and 3 people per boat, there are 15 people on boats in total. -/
theorem fifteen_people_on_boats :
  total_people_on_boats 5 3 = 15 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_people_on_boats_l3601_360139


namespace NUMINAMATH_CALUDE_ellipse_special_point_l3601_360133

def ellipse (x y : ℝ) : Prop := x^2/4 + y^2 = 1

def line_intersects_ellipse (m t x y : ℝ) : Prop :=
  ellipse x y ∧ x = t*y + m

def distance_squared (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  (x₁ - x₂)^2 + (y₁ - y₂)^2

theorem ellipse_special_point :
  ∃ (m : ℝ), 
    (∀ (t x₁ y₁ x₂ y₂ : ℝ),
      line_intersects_ellipse m t x₁ y₁ ∧ 
      line_intersects_ellipse m t x₂ y₂ ∧ 
      (x₁, y₁) ≠ (x₂, y₂) →
      ∃ (k : ℝ), 
        1 / distance_squared m 0 x₁ y₁ + 1 / distance_squared m 0 x₂ y₂ = k) ∧
    m = 2 * Real.sqrt 15 / 5 ∧
    (∀ (t x₁ y₁ x₂ y₂ : ℝ),
      line_intersects_ellipse m t x₁ y₁ ∧ 
      line_intersects_ellipse m t x₂ y₂ ∧ 
      (x₁, y₁) ≠ (x₂, y₂) →
      1 / distance_squared m 0 x₁ y₁ + 1 / distance_squared m 0 x₂ y₂ = 5) :=
sorry

end NUMINAMATH_CALUDE_ellipse_special_point_l3601_360133


namespace NUMINAMATH_CALUDE_scalene_to_right_triangle_l3601_360196

/-- 
For any scalene triangle with sides a < b < c, there exists a real number x 
such that the new triangle with sides (a+x), (b+x), and (c+x) is a right triangle.
-/
theorem scalene_to_right_triangle 
  (a b c : ℝ) 
  (h_scalene : a < b ∧ b < c) : 
  ∃ x : ℝ, (a + x)^2 + (b + x)^2 = (c + x)^2 := by
  sorry

end NUMINAMATH_CALUDE_scalene_to_right_triangle_l3601_360196


namespace NUMINAMATH_CALUDE_greatest_multiple_under_1000_l3601_360178

theorem greatest_multiple_under_1000 : 
  ∀ n : ℕ, n < 1000 → n % 5 = 0 → n % 6 = 0 → n ≤ 990 :=
by sorry

end NUMINAMATH_CALUDE_greatest_multiple_under_1000_l3601_360178


namespace NUMINAMATH_CALUDE_circle_line_intersection_l3601_360154

theorem circle_line_intersection (a b : ℝ) : 
  (a^2 + b^2 > 1) →
  (∃ (x y : ℝ), x^2 + y^2 = 1 ∧ a*x + b*y + 2 = 0) →
  (a^2 + b^2 > 1) ∧
  ¬(∀ (a b : ℝ), a^2 + b^2 > 1 → ∃ (x y : ℝ), x^2 + y^2 = 1 ∧ a*x + b*y + 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_circle_line_intersection_l3601_360154


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l3601_360143

theorem absolute_value_equation_solution :
  ∃! y : ℝ, |y - 8| + 3 * y = 12 :=
by
  -- The unique solution is y = 2
  use 2
  constructor
  · -- Prove that y = 2 satisfies the equation
    sorry
  · -- Prove that any y satisfying the equation must equal 2
    sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l3601_360143


namespace NUMINAMATH_CALUDE_jack_payback_l3601_360193

/-- The amount borrowed by Jack from Jill -/
def principal : ℝ := 1200

/-- The interest rate on the loan -/
def interest_rate : ℝ := 0.1

/-- The total amount Jack will pay back -/
def total_amount : ℝ := principal * (1 + interest_rate)

/-- Theorem stating that Jack will pay back $1320 -/
theorem jack_payback : total_amount = 1320 := by
  sorry

end NUMINAMATH_CALUDE_jack_payback_l3601_360193


namespace NUMINAMATH_CALUDE_complex_root_modulus_sqrt5_l3601_360174

theorem complex_root_modulus_sqrt5 (k : ℝ) :
  (∃ (z : ℂ), z^3 + 2*(k-1)*z^2 + 9*z + 5*(k-1) = 0 ∧ Complex.abs z = Real.sqrt 5) →
  k = -1 ∨ k = 3 := by
sorry

end NUMINAMATH_CALUDE_complex_root_modulus_sqrt5_l3601_360174


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l3601_360129

theorem greatest_divisor_with_remainders : Nat.gcd (3461 - 23) (4783 - 41) = 2 := by
  sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l3601_360129


namespace NUMINAMATH_CALUDE_bookstore_repricing_l3601_360123

theorem bookstore_repricing (n : Nat) (p₁ p₂ : Nat) (h₁ : n = 1452) (h₂ : p₁ = 42) (h₃ : p₂ = 45) :
  (n * p₁) % p₂ = 9 := by
  sorry

end NUMINAMATH_CALUDE_bookstore_repricing_l3601_360123


namespace NUMINAMATH_CALUDE_range_of_m_l3601_360149

theorem range_of_m (m : ℝ) : 
  (∀ θ : ℝ, m^2 + (Real.cos θ^2 - 5) * m + 4 * Real.sin θ^2 ≥ 0) → 
  (m ≥ 4 ∨ m ≤ 0) := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l3601_360149


namespace NUMINAMATH_CALUDE_chocolate_cost_proof_l3601_360189

/-- The cost of the chocolate -/
def chocolate_cost : ℝ := 3

/-- The cost of the candy bar -/
def candy_bar_cost : ℝ := 6

theorem chocolate_cost_proof :
  chocolate_cost = 3 ∧
  candy_bar_cost = 6 ∧
  candy_bar_cost = chocolate_cost + 3 :=
by sorry

end NUMINAMATH_CALUDE_chocolate_cost_proof_l3601_360189


namespace NUMINAMATH_CALUDE_martin_position_l3601_360170

/-- Represents a queue with the given properties -/
structure Queue where
  total : ℕ
  martin_pos : ℕ
  friend1_pos : ℕ
  friend2_pos : ℕ
  total_multiple_of_3 : total % 3 = 0
  martin_centered : martin_pos - 1 = total - martin_pos
  friend1_behind : friend1_pos > martin_pos
  friend2_behind : friend2_pos > martin_pos
  friend1_is_19th : friend1_pos = 19
  friend2_is_28th : friend2_pos = 28

/-- The theorem stating Martin's position in the queue -/
theorem martin_position (q : Queue) : q.martin_pos = 17 := by
  sorry

end NUMINAMATH_CALUDE_martin_position_l3601_360170


namespace NUMINAMATH_CALUDE_vehicle_speeds_l3601_360177

/-- Represents the initial speeds and distance of two vehicles --/
structure VehicleData where
  bus_speed : ℝ
  car_speed : ℝ
  final_distance : ℝ

/-- Calculates the total distance traveled by both vehicles --/
def total_distance (data : VehicleData) : ℝ :=
  2 * data.bus_speed + 2 * data.car_speed + 2 * data.bus_speed + 2 * (data.car_speed - 10)

/-- Theorem stating the initial speeds of the vehicles --/
theorem vehicle_speeds : ∃ (data : VehicleData),
  data.car_speed = data.bus_speed + 8 ∧
  data.final_distance = 384 ∧
  total_distance data = data.final_distance ∧
  data.bus_speed = 46.5 ∧
  data.car_speed = 54.5 := by
  sorry

end NUMINAMATH_CALUDE_vehicle_speeds_l3601_360177


namespace NUMINAMATH_CALUDE_race_graph_representation_l3601_360182

-- Define the types of contestants
inductive Contestant
| Snail
| Horse

-- Define the movement pattern
structure MovementPattern where
  contestant : Contestant
  isConsistent : Bool
  hasRest : Bool
  initialSpeed : ℕ
  finalPosition : ℕ

-- Define the graph types
inductive GraphType
| FluctuatingSpeed
| SteadySlowWinnerVsFastStartStop
| ConsistentlyIncreasing

-- Define the race outcome
def raceOutcome (snailPattern : MovementPattern) (horsePattern : MovementPattern) : GraphType :=
  if snailPattern.isConsistent ∧ 
     snailPattern.initialSpeed < horsePattern.initialSpeed ∧ 
     horsePattern.hasRest ∧ 
     snailPattern.finalPosition > horsePattern.finalPosition
  then GraphType.SteadySlowWinnerVsFastStartStop
  else GraphType.FluctuatingSpeed

-- Theorem statement
theorem race_graph_representation 
  (snail : MovementPattern) 
  (horse : MovementPattern) 
  (h_snail_contestant : snail.contestant = Contestant.Snail)
  (h_horse_contestant : horse.contestant = Contestant.Horse)
  (h_snail_consistent : snail.isConsistent = true)
  (h_snail_slow : snail.initialSpeed < horse.initialSpeed)
  (h_horse_rest : horse.hasRest = true)
  (h_snail_wins : snail.finalPosition > horse.finalPosition) :
  raceOutcome snail horse = GraphType.SteadySlowWinnerVsFastStartStop :=
by sorry

end NUMINAMATH_CALUDE_race_graph_representation_l3601_360182


namespace NUMINAMATH_CALUDE_factor_cubic_expression_l3601_360117

theorem factor_cubic_expression (a b c : ℝ) :
  ((a^3 - b^3)^3 + (b^3 - c^3)^3 + (c^3 - a^3)^3) / ((a - b)^3 + (b - c)^3 + (c - a)^3) 
  = (a^2 + a*b + b^2) * (b^2 + b*c + c^2) * (c^2 + c*a + a^2) :=
by sorry

end NUMINAMATH_CALUDE_factor_cubic_expression_l3601_360117


namespace NUMINAMATH_CALUDE_family_age_theorem_l3601_360199

/-- Calculates the average age of a family given initial conditions --/
def average_family_age (initial_average_age : ℚ) (years_passed : ℕ) (child_age : ℕ) : ℚ :=
  let initial_total_age := initial_average_age * 2
  let current_total_age := initial_total_age + years_passed * 2 + child_age
  current_total_age / 3

/-- Proves that the average age of the family is 19 years --/
theorem family_age_theorem (initial_average_age : ℚ) (years_passed : ℕ) (child_age : ℕ)
  (h1 : initial_average_age = 23)
  (h2 : years_passed = 5)
  (h3 : child_age = 1) :
  average_family_age initial_average_age years_passed child_age = 19 := by
  sorry

#eval average_family_age 23 5 1

end NUMINAMATH_CALUDE_family_age_theorem_l3601_360199


namespace NUMINAMATH_CALUDE_fraction_equality_l3601_360108

theorem fraction_equality (x : ℝ) : (3 + x) / (5 + x) = (1 + x) / (2 + x) ↔ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3601_360108


namespace NUMINAMATH_CALUDE_quadratic_function_a_range_l3601_360185

/-- Given a quadratic function y = (ax - 1)(x - a), this theorem proves that the range of a
    satisfying specific conditions about its roots and axis of symmetry is (0, 1). -/
theorem quadratic_function_a_range :
  ∀ a : ℝ,
  (∀ x : ℝ, (a * x - 1) * (x - a) > 0 ↔ x < a ∨ x > 1/a) ∧
  ((a^2 + 1) / (2 * a) > 0) ∧
  ¬(∀ x : ℝ, (a * x - 1) * (x - a) < 0 ↔ x < a ∨ x > 1/a)
  ↔ 0 < a ∧ a < 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_a_range_l3601_360185


namespace NUMINAMATH_CALUDE_polynomial_product_l3601_360136

variables (a b : ℚ)

theorem polynomial_product (a b : ℚ) :
  (-3 * a^2 * b) * (-2 * a * b + b - 3) = 6 * a^3 * b^2 - 3 * a^2 * b^2 + 9 * a^2 * b :=
by sorry

end NUMINAMATH_CALUDE_polynomial_product_l3601_360136


namespace NUMINAMATH_CALUDE_equal_numbers_from_equal_powers_l3601_360100

theorem equal_numbers_from_equal_powers (a : Fin 17 → ℕ) 
  (h : ∀ i : Fin 16, (a i) ^ (a (i + 1)) = (a (i + 1)) ^ (a ((i + 2) % 17))) : 
  ∀ i j : Fin 17, a i = a j := by
  sorry

end NUMINAMATH_CALUDE_equal_numbers_from_equal_powers_l3601_360100


namespace NUMINAMATH_CALUDE_largest_integer_with_remainder_l3601_360125

theorem largest_integer_with_remainder (n : ℕ) : 
  n < 100 ∧ n % 5 = 2 ∧ ∀ m : ℕ, m < 100 ∧ m % 5 = 2 → m ≤ n → n = 97 := by
  sorry

end NUMINAMATH_CALUDE_largest_integer_with_remainder_l3601_360125


namespace NUMINAMATH_CALUDE_f_of_two_equals_two_l3601_360190

theorem f_of_two_equals_two (f : ℝ → ℝ) (h : ∀ x ≥ 0, f (1 + Real.sqrt x) = x + 1) : f 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_of_two_equals_two_l3601_360190


namespace NUMINAMATH_CALUDE_balls_distribution_theorem_l3601_360194

def distribute_balls (n : ℕ) (k : ℕ) (min : ℕ) (max : ℕ) : ℕ :=
  sorry

theorem balls_distribution_theorem :
  distribute_balls 6 2 1 4 = 35 := by
  sorry

end NUMINAMATH_CALUDE_balls_distribution_theorem_l3601_360194


namespace NUMINAMATH_CALUDE_water_displaced_volume_squared_l3601_360109

/-- The square of the volume of water displaced by a cube in a cylindrical barrel -/
theorem water_displaced_volume_squared
  (barrel_radius : ℝ)
  (barrel_height : ℝ)
  (cube_side_length : ℝ)
  (h_radius : barrel_radius = 5)
  (h_height : barrel_height = 10)
  (h_side : cube_side_length = 6) :
  let diagonal := cube_side_length * Real.sqrt 3
  let triangle_side := barrel_radius * Real.sqrt 3
  let tetrahedron_leg := (5 * Real.sqrt 6) / 2
  let volume := (375 * Real.sqrt 6) / 8
  volume ^ 2 = 843750 / 64 := by
  sorry

#eval (843750 / 64 : Float)  -- Should output approximately 13141.855

end NUMINAMATH_CALUDE_water_displaced_volume_squared_l3601_360109


namespace NUMINAMATH_CALUDE_tie_record_score_difference_l3601_360183

/-- The league record average score per player per round -/
def league_record : ℕ := 287

/-- The number of players in a team -/
def players_per_team : ℕ := 4

/-- The number of rounds in a season -/
def rounds_per_season : ℕ := 10

/-- The total score of George's team after 9 rounds -/
def team_score_9_rounds : ℕ := 10440

/-- The minimum average score needed per player in the final round to tie the record -/
def min_avg_score_final_round : ℕ := (league_record * players_per_team * rounds_per_season - team_score_9_rounds) / players_per_team

/-- The difference between the league record average and the minimum average score needed -/
def score_difference : ℕ := league_record - min_avg_score_final_round

theorem tie_record_score_difference : score_difference = 27 := by
  sorry

end NUMINAMATH_CALUDE_tie_record_score_difference_l3601_360183


namespace NUMINAMATH_CALUDE_adult_ticket_cost_l3601_360128

/-- Proves that the cost of an adult ticket is $5.50 given the specified conditions -/
theorem adult_ticket_cost : 
  let child_ticket_cost : ℝ := 3.50
  let total_tickets : ℕ := 21
  let total_cost : ℝ := 83.50
  let child_tickets : ℕ := 16
  let adult_tickets : ℕ := total_tickets - child_tickets
  let adult_ticket_cost : ℝ := (total_cost - child_ticket_cost * child_tickets) / adult_tickets
  adult_ticket_cost = 5.50 := by sorry

end NUMINAMATH_CALUDE_adult_ticket_cost_l3601_360128


namespace NUMINAMATH_CALUDE_white_balls_count_l3601_360198

theorem white_balls_count (red blue white : ℕ) : 
  red = 80 → blue = 40 → red = blue + white - 12 → white = 52 := by
  sorry

end NUMINAMATH_CALUDE_white_balls_count_l3601_360198


namespace NUMINAMATH_CALUDE_unique_remainder_mod_10_l3601_360164

theorem unique_remainder_mod_10 : ∃! n : ℕ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ 123456 [MOD 10] := by
  sorry

end NUMINAMATH_CALUDE_unique_remainder_mod_10_l3601_360164


namespace NUMINAMATH_CALUDE_billy_crayons_l3601_360156

theorem billy_crayons (initial remaining eaten : ℕ) 
  (h1 : eaten = 52)
  (h2 : remaining = 10)
  (h3 : initial = remaining + eaten) :
  initial = 62 := by
  sorry

end NUMINAMATH_CALUDE_billy_crayons_l3601_360156


namespace NUMINAMATH_CALUDE_lily_lottery_tickets_l3601_360126

/-- Represents the number of lottery tickets sold -/
def n : ℕ := 5

/-- The price of the i-th ticket -/
def ticket_price (i : ℕ) : ℕ := i

/-- The total amount collected from selling n tickets -/
def total_collected (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The profit Lily keeps -/
def profit : ℕ := 4

/-- The prize money for the lottery winner -/
def prize : ℕ := 11

theorem lily_lottery_tickets :
  (total_collected n = prize + profit) ∧
  (∀ m : ℕ, m ≠ n → total_collected m ≠ prize + profit) :=
by sorry

end NUMINAMATH_CALUDE_lily_lottery_tickets_l3601_360126


namespace NUMINAMATH_CALUDE_roots_sum_of_powers_l3601_360113

theorem roots_sum_of_powers (α β : ℝ) : 
  α^2 - 2*α - 1 = 0 → β^2 - 2*β - 1 = 0 → 5*α^4 + 12*β^3 = 169 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_of_powers_l3601_360113


namespace NUMINAMATH_CALUDE_two_digit_number_property_l3601_360163

theorem two_digit_number_property (c d h : ℕ) (m : ℕ) (y : ℤ) :
  c < 10 →
  d < 10 →
  m = 10 * c + d →
  m = h * (c + d) →
  (10 * d + c : ℤ) = y * (c + d) →
  y = 12 - h :=
by sorry

end NUMINAMATH_CALUDE_two_digit_number_property_l3601_360163


namespace NUMINAMATH_CALUDE_college_students_count_l3601_360186

theorem college_students_count (boys girls : ℕ) (h1 : boys * 5 = girls * 8) (h2 : girls = 210) :
  boys + girls = 546 := by
  sorry

end NUMINAMATH_CALUDE_college_students_count_l3601_360186


namespace NUMINAMATH_CALUDE_sum_of_ratios_l3601_360179

theorem sum_of_ratios (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x + 1/y = 5) (h2 : y + 1/z = 2) (h3 : z + 1/x = 3) :
  x/y + y/z + z/x = Real.sqrt ((x/y)^2 + (y/z)^2 + (z/x)^2 + 3) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ratios_l3601_360179


namespace NUMINAMATH_CALUDE_circle_line_intersection_sum_l3601_360166

/-- Given a circle with radius 4 centered at the origin and a line y = 4 - (2 - √3)x
    intersecting the circle at points A and B, the sum of the length of segment AB
    and the length of the shorter arc AB is 4√(2 - √3) + (2π/3) -/
theorem circle_line_intersection_sum (A B : ℝ × ℝ) : 
  let r : ℝ := 4
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = r^2}
  let line := {(x, y) : ℝ × ℝ | y = 4 - (2 - Real.sqrt 3) * x}
  A ∈ circle ∧ A ∈ line ∧ B ∈ circle ∧ B ∈ line ∧ A ≠ B →
  let segment_length := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let angle := Real.arccos ((2 * r^2 - segment_length^2) / (2 * r^2))
  let arc_length := angle * r
  segment_length + min arc_length (2 * π * r - arc_length) = 4 * Real.sqrt (2 - Real.sqrt 3) + (2 * π / 3) := by
  sorry

end NUMINAMATH_CALUDE_circle_line_intersection_sum_l3601_360166


namespace NUMINAMATH_CALUDE_system_solution_ratio_l3601_360131

theorem system_solution_ratio (x y c d : ℝ) : 
  x ≠ 0 → y ≠ 0 → d ≠ 0 →
  8 * x - 6 * y = c →
  12 * y - 18 * x = d →
  c / d = -4 / 9 := by
sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l3601_360131


namespace NUMINAMATH_CALUDE_sugar_calculation_l3601_360112

/-- Proves that the total amount of sugar the owner started with is 14100 grams --/
theorem sugar_calculation (total_packs : ℕ) (pack_weight : ℝ) (remaining_sugar : ℝ) :
  total_packs = 35 →
  pack_weight = 400 →
  remaining_sugar = 100 →
  (total_packs : ℝ) * pack_weight + remaining_sugar = 14100 := by
  sorry

end NUMINAMATH_CALUDE_sugar_calculation_l3601_360112


namespace NUMINAMATH_CALUDE_amy_game_score_l3601_360173

theorem amy_game_score (points_per_treasure : ℕ) (treasures_level1 : ℕ) (treasures_level2 : ℕ)
  (h1 : points_per_treasure = 4)
  (h2 : treasures_level1 = 6)
  (h3 : treasures_level2 = 2) :
  points_per_treasure * treasures_level1 + points_per_treasure * treasures_level2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_amy_game_score_l3601_360173


namespace NUMINAMATH_CALUDE_sphere_volume_calculation_l3601_360121

-- Define the sphere and plane
def Sphere : Type := Unit
def Plane : Type := Unit

-- Define the properties of the intersection
def intersection_diameter (s : Sphere) (p : Plane) : ℝ := 6

-- Define the distance from the center of the sphere to the plane
def center_to_plane_distance (s : Sphere) (p : Plane) : ℝ := 4

-- Define the volume of a sphere
def sphere_volume (s : Sphere) : ℝ := sorry

-- Theorem statement
theorem sphere_volume_calculation (s : Sphere) (p : Plane) :
  sphere_volume s = (500 * Real.pi) / 3 :=
by sorry

end NUMINAMATH_CALUDE_sphere_volume_calculation_l3601_360121


namespace NUMINAMATH_CALUDE_max_value_of_f_l3601_360180

noncomputable def f (x : ℝ) : ℝ := 
  (Real.sqrt 2 * Real.sin x + Real.cos x) / (Real.sin x + Real.sqrt (1 - Real.sin x))

theorem max_value_of_f :
  ∃ (M : ℝ), M = Real.sqrt 2 ∧ 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi → f x ≤ M) ∧
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi ∧ f x = M) :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l3601_360180


namespace NUMINAMATH_CALUDE_max_cube_path_length_l3601_360161

/-- Represents a cube with edges of a given length -/
structure Cube where
  edgeLength : ℝ
  edgeCount : ℕ

/-- Represents a path on the cube -/
structure CubePath where
  length : ℝ
  edgeCount : ℕ

/-- The maximum path length on a cube without retracing -/
def maxPathLength (c : Cube) : ℝ := sorry

theorem max_cube_path_length 
  (c : Cube) 
  (h1 : c.edgeLength = 3)
  (h2 : c.edgeCount = 12) :
  maxPathLength c = 24 := by sorry

end NUMINAMATH_CALUDE_max_cube_path_length_l3601_360161


namespace NUMINAMATH_CALUDE_find_liar_in_17_questions_l3601_360142

/-- Represents a person who can be either a knight or a liar -/
inductive Person
  | knight : Person
  | liar : Person

/-- Represents the response to a question -/
inductive Response
  | yes : Response
  | no : Response

/-- A function that simulates asking a question to a person -/
def ask (p : Person) (cardNumber : Nat) (askedNumber : Nat) : Response :=
  match p with
  | Person.knight => if cardNumber = askedNumber then Response.yes else Response.no
  | Person.liar => if cardNumber ≠ askedNumber then Response.yes else Response.no

/-- The main theorem statement -/
theorem find_liar_in_17_questions 
  (people : Fin 10 → Person) 
  (cards : Fin 10 → Nat) 
  (h1 : ∃! i, people i = Person.liar) 
  (h2 : ∀ i j, i ≠ j → cards i ≠ cards j) 
  (h3 : ∀ i, cards i ∈ Set.range (fun n : Nat => n + 1) ∩ Set.range (fun n : Nat => 11 - n)) :
  ∃ (strategy : Nat → Fin 10 × Nat), 
    (∀ n, n < 17 → (strategy n).2 ∈ Set.range (fun n : Nat => n + 1) ∩ Set.range (fun n : Nat => 11 - n)) →
    ∃ (result : Fin 10), 
      (∀ i, i ≠ result → people i = Person.knight) ∧ 
      (people result = Person.liar) :=
sorry

end NUMINAMATH_CALUDE_find_liar_in_17_questions_l3601_360142


namespace NUMINAMATH_CALUDE_binomial_12_choose_10_l3601_360167

theorem binomial_12_choose_10 : Nat.choose 12 10 = 66 := by sorry

end NUMINAMATH_CALUDE_binomial_12_choose_10_l3601_360167


namespace NUMINAMATH_CALUDE_p_and_q_true_l3601_360130

theorem p_and_q_true (a b c : ℝ) : 
  ((a > b → a + c > b + c) ∧ ((a > b ∧ b > 0) → a * c > b * c)) := by
  sorry

end NUMINAMATH_CALUDE_p_and_q_true_l3601_360130


namespace NUMINAMATH_CALUDE_tank_inflow_rate_l3601_360103

/-- Given two tanks with equal capacity, prove the inflow rate of the slower-filling tank -/
theorem tank_inflow_rate (capacity : ℝ) (fast_rate slow_rate : ℝ) (time_diff : ℝ) :
  capacity > 0 →
  fast_rate > 0 →
  slow_rate > 0 →
  time_diff > 0 →
  capacity / fast_rate + time_diff = capacity / slow_rate →
  capacity = 20 →
  fast_rate = 4 →
  time_diff = 5 →
  slow_rate = 2 := by sorry

end NUMINAMATH_CALUDE_tank_inflow_rate_l3601_360103


namespace NUMINAMATH_CALUDE_complex_square_eq_neg45_neg48i_l3601_360187

theorem complex_square_eq_neg45_neg48i (z : ℂ) : 
  z^2 = -45 - 48*I ↔ z = 3 - 8*I ∨ z = -3 + 8*I := by
  sorry

end NUMINAMATH_CALUDE_complex_square_eq_neg45_neg48i_l3601_360187


namespace NUMINAMATH_CALUDE_smallest_cube_box_volume_l3601_360122

/-- Represents the dimensions of a pyramid -/
structure PyramidDimensions where
  height : ℝ
  baseSide : ℝ

/-- Calculates the volume of a cube given its side length -/
def cubeVolume (side : ℝ) : ℝ := side ^ 3

/-- Theorem: The smallest cube-shaped box that can contain a pyramid with given dimensions has a volume of 3375 cubic inches -/
theorem smallest_cube_box_volume
  (pyramid : PyramidDimensions)
  (h_height : pyramid.height = 15)
  (h_base : pyramid.baseSide = 14) :
  cubeVolume (max pyramid.height pyramid.baseSide) = 3375 := by
  sorry

#eval cubeVolume 15  -- Should output 3375

end NUMINAMATH_CALUDE_smallest_cube_box_volume_l3601_360122


namespace NUMINAMATH_CALUDE_trapezoid_area_equality_l3601_360105

/-- Represents a trapezoid divided into triangles and a pentagon as described in the problem -/
structure DividedTrapezoid where
  /-- Area of the central pentagon -/
  Q : ℝ
  /-- Area of the triangle adjacent to one lateral side -/
  s₁ : ℝ
  /-- Area of the triangle adjacent to the shorter base -/
  s₂ : ℝ
  /-- Area of the triangle adjacent to the other lateral side -/
  s₃ : ℝ
  /-- Area of the triangle between s₁ and s₂ -/
  x : ℝ
  /-- Area of the triangle between s₂ and s₃ -/
  y : ℝ
  /-- The sum of areas of triangles adjacent to one side and the shorter base equals half the sum of x, y, s₂, and Q -/
  h₁ : s₁ + x + s₂ = (x + y + s₂ + Q) / 2
  /-- The sum of areas of triangles adjacent to the shorter base and the other side equals half the sum of x, y, s₂, and Q -/
  h₂ : s₂ + y + s₃ = (x + y + s₂ + Q) / 2

/-- The sum of the areas of the three triangles adjacent to the lateral sides and the shorter base 
    of the trapezoid is equal to the area of the pentagon -/
theorem trapezoid_area_equality (t : DividedTrapezoid) : t.s₁ + t.s₂ + t.s₃ = t.Q := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_equality_l3601_360105


namespace NUMINAMATH_CALUDE_dan_has_five_marbles_l3601_360169

/-- The number of blue marbles Dan has -/
def dans_marbles : ℕ := sorry

/-- The number of blue marbles Mary has -/
def marys_marbles : ℕ := 10

/-- Mary has 2 times more blue marbles than Dan -/
axiom mary_double_dan : marys_marbles = 2 * dans_marbles

theorem dan_has_five_marbles : dans_marbles = 5 := by
  sorry

end NUMINAMATH_CALUDE_dan_has_five_marbles_l3601_360169


namespace NUMINAMATH_CALUDE_fraction_comparison_and_differences_l3601_360138

theorem fraction_comparison_and_differences :
  (1 / 3 : ℚ) < (1 / 2 : ℚ) ∧ (1 / 2 : ℚ) < (3 / 5 : ℚ) ∧
  (1 / 2 : ℚ) - (1 / 3 : ℚ) = (1 / 6 : ℚ) ∧
  (3 / 5 : ℚ) - (1 / 2 : ℚ) = (1 / 10 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_fraction_comparison_and_differences_l3601_360138


namespace NUMINAMATH_CALUDE_min_side_arithmetic_angles_l3601_360171

/-- Given a triangle ABC where the internal angles form an arithmetic sequence and the area is 2√3,
    the minimum value of side AB is 2√2. -/
theorem min_side_arithmetic_angles (A B C : ℝ) (a b c : ℝ) :
  -- Angles form an arithmetic sequence
  2 * C = A + B →
  -- Sum of angles in a triangle is π
  A + B + C = π →
  -- Area of the triangle is 2√3
  (1 / 2) * a * b * Real.sin C = 2 * Real.sqrt 3 →
  -- AB is the side opposite to angle C
  c = (a^2 + b^2 - 2*a*b*(Real.cos C))^(1/2) →
  -- Minimum value of AB (c) is 2√2
  c ≥ 2 * Real.sqrt 2 ∧ ∃ (a' b' : ℝ), c = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_side_arithmetic_angles_l3601_360171


namespace NUMINAMATH_CALUDE_intersection_M_N_l3601_360172

def M : Set ℝ := {x | -4 < x ∧ x < 2}
def N : Set ℝ := {x | x^2 - x - 6 < 0}

theorem intersection_M_N : M ∩ N = {x | -2 < x ∧ x < 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3601_360172


namespace NUMINAMATH_CALUDE_square_sum_ge_mixed_products_l3601_360144

theorem square_sum_ge_mixed_products (a b c : ℝ) : a^2 + b^2 + c^2 ≥ a*b + b*c + c*a := by
  sorry

end NUMINAMATH_CALUDE_square_sum_ge_mixed_products_l3601_360144


namespace NUMINAMATH_CALUDE_point_inside_circle_l3601_360150

theorem point_inside_circle (a : ℝ) : 
  ((1 - a)^2 + (1 + a)^2 < 4) ↔ (-1 < a ∧ a < 1) := by
sorry

end NUMINAMATH_CALUDE_point_inside_circle_l3601_360150


namespace NUMINAMATH_CALUDE_loan_amount_calculation_l3601_360114

/-- Calculates the total loan amount given the loan term, down payment, and monthly payment. -/
def total_loan_amount (loan_term_years : ℕ) (down_payment : ℕ) (monthly_payment : ℕ) : ℕ :=
  down_payment + loan_term_years * 12 * monthly_payment

/-- Theorem stating that given the specific loan conditions, the total loan amount is $46,000. -/
theorem loan_amount_calculation :
  total_loan_amount 5 10000 600 = 46000 := by
  sorry

end NUMINAMATH_CALUDE_loan_amount_calculation_l3601_360114


namespace NUMINAMATH_CALUDE_min_sum_of_distances_min_sum_of_distances_achievable_l3601_360176

theorem min_sum_of_distances (x : ℝ) : 
  Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((2 - x)^2 + (2 + x)^2) ≥ Real.sqrt 20 := by
  sorry

theorem min_sum_of_distances_achievable : 
  ∃ x : ℝ, Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((2 - x)^2 + (2 + x)^2) = Real.sqrt 20 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_of_distances_min_sum_of_distances_achievable_l3601_360176


namespace NUMINAMATH_CALUDE_passing_grade_fraction_l3601_360162

theorem passing_grade_fraction (students_A students_B students_C students_D students_F : ℚ) :
  students_A = 1/4 →
  students_B = 1/2 →
  students_C = 1/8 →
  students_D = 1/12 →
  students_F = 1/24 →
  students_A + students_B + students_C = 7/8 := by
  sorry

end NUMINAMATH_CALUDE_passing_grade_fraction_l3601_360162


namespace NUMINAMATH_CALUDE_tan_half_sum_l3601_360188

theorem tan_half_sum (a b : ℝ) 
  (h1 : Real.cos a + Real.cos b = (1 : ℝ) / 2)
  (h2 : Real.sin a + Real.sin b = (3 : ℝ) / 11) : 
  Real.tan ((a + b) / 2) = (6 : ℝ) / 11 := by
  sorry

end NUMINAMATH_CALUDE_tan_half_sum_l3601_360188


namespace NUMINAMATH_CALUDE_max_product_with_sum_and_even_l3601_360155

theorem max_product_with_sum_and_even (x y : ℤ) : 
  x + y = 280 → (Even x ∨ Even y) → x * y ≤ 19600 := by
  sorry

end NUMINAMATH_CALUDE_max_product_with_sum_and_even_l3601_360155


namespace NUMINAMATH_CALUDE_real_part_of_complex_fraction_l3601_360135

theorem real_part_of_complex_fraction (θ : ℝ) :
  let z : ℂ := Complex.exp (θ * Complex.I)
  Complex.abs z = 1 →
  (1 / (2 - z)).re = (2 - Real.cos θ) / (5 - 4 * Real.cos θ) := by
sorry

end NUMINAMATH_CALUDE_real_part_of_complex_fraction_l3601_360135


namespace NUMINAMATH_CALUDE_symmetry_implies_axis_1_5_l3601_360141

/-- A function f is symmetric about the line x = 1.5 if f(x) = f(3 - x) for all x. -/
def is_symmetric_about_1_5 (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (3 - x)

/-- The line x = 1.5 is an axis of symmetry for a function f if 
    for any point (x, f(x)) on the graph, the point (3 - x, f(x)) is also on the graph. -/
def is_axis_of_symmetry_1_5 (f : ℝ → ℝ) : Prop :=
  ∀ x y, f x = y → f (3 - x) = y

theorem symmetry_implies_axis_1_5 (f : ℝ → ℝ) :
  is_symmetric_about_1_5 f → is_axis_of_symmetry_1_5 f :=
by
  sorry

end NUMINAMATH_CALUDE_symmetry_implies_axis_1_5_l3601_360141


namespace NUMINAMATH_CALUDE_mail_difference_l3601_360107

/-- Proves that the difference between mail sent on Thursday and Wednesday is 15 --/
theorem mail_difference (monday tuesday wednesday thursday : ℕ) : 
  monday = 65 →
  tuesday = monday + 10 →
  wednesday = tuesday - 5 →
  thursday > wednesday →
  monday + tuesday + wednesday + thursday = 295 →
  thursday - wednesday = 15 :=
by sorry

end NUMINAMATH_CALUDE_mail_difference_l3601_360107


namespace NUMINAMATH_CALUDE_last_two_digits_of_A_power_20_l3601_360132

theorem last_two_digits_of_A_power_20 (A : ℤ) 
  (h1 : A % 2 = 0) 
  (h2 : A % 10 ≠ 0) : 
  A^20 % 100 = 76 := by
sorry

end NUMINAMATH_CALUDE_last_two_digits_of_A_power_20_l3601_360132


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3601_360147

theorem inequality_equivalence (x : ℝ) : 4 * x - 1 < 0 ↔ x < 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3601_360147


namespace NUMINAMATH_CALUDE_homework_difference_l3601_360191

theorem homework_difference (reading_pages math_pages biology_pages : ℕ) 
  (h1 : reading_pages = 4)
  (h2 : math_pages = 7)
  (h3 : biology_pages = 19) :
  math_pages - reading_pages = 3 :=
by sorry

end NUMINAMATH_CALUDE_homework_difference_l3601_360191


namespace NUMINAMATH_CALUDE_tan_seventeen_pi_fourths_l3601_360165

theorem tan_seventeen_pi_fourths : Real.tan (17 * π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_seventeen_pi_fourths_l3601_360165


namespace NUMINAMATH_CALUDE_olivia_napkins_l3601_360106

theorem olivia_napkins (initial_napkins final_napkins : ℕ) 
  (h1 : initial_napkins = 15)
  (h2 : final_napkins = 45)
  (h3 : ∃ (o : ℕ), final_napkins = initial_napkins + o + 2*o) :
  ∃ (o : ℕ), o = 10 ∧ final_napkins = initial_napkins + o + 2*o :=
by sorry

end NUMINAMATH_CALUDE_olivia_napkins_l3601_360106


namespace NUMINAMATH_CALUDE_trajectory_equation_l3601_360120

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define point P
def point_P : ℝ × ℝ := (-1, 0)

-- Define a point Q on the circle
def point_Q (x y : ℝ) : Prop := circle_O x y

-- Define the midpoint M of PQ
def midpoint_M (x y : ℝ) : Prop :=
  ∃ (qx qy : ℝ), point_Q qx qy ∧ x = (qx + point_P.1) / 2 ∧ y = (qy + point_P.2) / 2

-- Theorem: The trajectory of M forms the equation (x + 1/2)² + y² = 1
theorem trajectory_equation :
  ∀ (x y : ℝ), midpoint_M x y ↔ (x + 1/2)^2 + y^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_trajectory_equation_l3601_360120


namespace NUMINAMATH_CALUDE_willsons_work_hours_l3601_360153

theorem willsons_work_hours : 
  let monday : ℚ := 3/4
  let tuesday : ℚ := 1/2
  let wednesday : ℚ := 2/3
  let thursday : ℚ := 5/6
  let friday : ℚ := 75/60
  monday + tuesday + wednesday + thursday + friday = 4 := by
  sorry

end NUMINAMATH_CALUDE_willsons_work_hours_l3601_360153


namespace NUMINAMATH_CALUDE_division_multiplication_equivalence_l3601_360157

theorem division_multiplication_equivalence : 
  ∀ (x : ℚ), x * (9 / 3) * (5 / 6) = x / (2 / 5) :=
by sorry

end NUMINAMATH_CALUDE_division_multiplication_equivalence_l3601_360157


namespace NUMINAMATH_CALUDE_share_ratio_l3601_360160

/-- Given a total amount divided among three people (a, b, c), prove the ratio of a's share to the sum of b's and c's shares -/
theorem share_ratio (total a b c : ℚ) : 
  total = 100 →
  a = 20 →
  b = (3 / 5) * (a + c) →
  total = a + b + c →
  a / (b + c) = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_share_ratio_l3601_360160


namespace NUMINAMATH_CALUDE_min_value_fraction_equality_condition_l3601_360118

theorem min_value_fraction (x : ℝ) (h : x > 6) : x^2 / (x - 6) ≥ 18 := by
  sorry

theorem equality_condition (x : ℝ) (h : x > 6) : x^2 / (x - 6) = 18 ↔ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_min_value_fraction_equality_condition_l3601_360118


namespace NUMINAMATH_CALUDE_doritos_distribution_l3601_360145

theorem doritos_distribution (total_bags : ℕ) (doritos_fraction : ℚ) (num_piles : ℕ) : 
  total_bags = 200 →
  doritos_fraction = 2 / 5 →
  num_piles = 5 →
  (total_bags : ℚ) * doritos_fraction / num_piles = 16 := by
  sorry

end NUMINAMATH_CALUDE_doritos_distribution_l3601_360145


namespace NUMINAMATH_CALUDE_parabola_properties_l3601_360159

-- Define the parabola
def parabola (x : ℝ) : ℝ := (x - 3)^2 + 5

-- Theorem stating the properties of the parabola
theorem parabola_properties :
  (∀ x : ℝ, parabola x ≥ 5) ∧ 
  (∀ x : ℝ, parabola (3 + x) = parabola (3 - x)) ∧
  (parabola 3 = 5) := by
  sorry


end NUMINAMATH_CALUDE_parabola_properties_l3601_360159


namespace NUMINAMATH_CALUDE_not_equivalent_to_0_0000042_l3601_360116

theorem not_equivalent_to_0_0000042 : ¬ (2.1 * 10^(-6) = 0.0000042) :=
by
  have h1 : 0.0000042 = 4.2 * 10^(-6) := by sorry
  sorry

end NUMINAMATH_CALUDE_not_equivalent_to_0_0000042_l3601_360116


namespace NUMINAMATH_CALUDE_football_season_length_l3601_360102

/-- The number of football games in one month -/
def games_per_month : ℝ := 323.0

/-- The total number of football games in the season -/
def total_games : ℕ := 5491

/-- The number of months in the football season -/
def season_months : ℕ := 17

/-- Theorem stating that the number of months in the season is 17 -/
theorem football_season_length :
  (total_games : ℝ) / games_per_month = season_months := by
  sorry

end NUMINAMATH_CALUDE_football_season_length_l3601_360102


namespace NUMINAMATH_CALUDE_smallest_result_l3601_360181

def S : Finset Nat := {3, 5, 7, 11, 13, 17}

def process (a b c : Nat) : Nat :=
  max (max ((a + b) * c) ((a + c) * b)) ((b + c) * a)

def valid_selection (a b c : Nat) : Prop :=
  a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c

theorem smallest_result :
  ∃ (a b c : Nat), valid_selection a b c ∧
    process a b c = 36 ∧
    ∀ (x y z : Nat), valid_selection x y z → process x y z ≥ 36 := by
  sorry

end NUMINAMATH_CALUDE_smallest_result_l3601_360181


namespace NUMINAMATH_CALUDE_power_sum_equality_l3601_360111

theorem power_sum_equality : -2^2005 + (-2)^2006 + 2^2007 - 2^2008 = 2^2005 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equality_l3601_360111


namespace NUMINAMATH_CALUDE_train_speed_calculation_l3601_360151

/-- Calculates the speed of a train crossing a bridge -/
theorem train_speed_calculation (train_length : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) :
  train_length = 170 →
  bridge_length = 205 →
  crossing_time = 30 →
  (train_length + bridge_length) / crossing_time * 3.6 = 45 := by
  sorry

#check train_speed_calculation

end NUMINAMATH_CALUDE_train_speed_calculation_l3601_360151
