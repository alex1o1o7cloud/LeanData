import Mathlib

namespace NUMINAMATH_CALUDE_concert_cost_theorem_l2125_212532

/-- Calculates the total cost for two people to attend a concert -/
def concert_cost (ticket_price : ℝ) (processing_fee_rate : ℝ) (parking_fee : ℝ) (entrance_fee : ℝ) : ℝ :=
  let total_ticket_cost := 2 * ticket_price
  let processing_fee := total_ticket_cost * processing_fee_rate
  let total_entrance_fee := 2 * entrance_fee
  total_ticket_cost + processing_fee + parking_fee + total_entrance_fee

/-- Theorem stating that the total cost for two people to attend the concert is $135.00 -/
theorem concert_cost_theorem : 
  concert_cost 50 0.15 10 5 = 135 := by
  sorry

end NUMINAMATH_CALUDE_concert_cost_theorem_l2125_212532


namespace NUMINAMATH_CALUDE_midpoint_line_slope_l2125_212530

/-- The slope of the line containing the midpoints of two given line segments is -3/7 -/
theorem midpoint_line_slope :
  let midpoint1 := ((1 + 3) / 2, (2 + 6) / 2)
  let midpoint2 := ((4 + 7) / 2, (1 + 4) / 2)
  let slope := (midpoint2.2 - midpoint1.2) / (midpoint2.1 - midpoint1.1)
  slope = -3 / 7 := by sorry

end NUMINAMATH_CALUDE_midpoint_line_slope_l2125_212530


namespace NUMINAMATH_CALUDE_cube_edge_length_specific_l2125_212522

/-- The edge length of a cube with the same volume as a rectangular block -/
def cube_edge_length (l w h : ℝ) : ℝ :=
  (l * w * h) ^ (1/3)

/-- Theorem: The edge length of a cube with the same volume as a 50cm × 8cm × 20cm rectangular block is 20cm -/
theorem cube_edge_length_specific : cube_edge_length 50 8 20 = 20 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_length_specific_l2125_212522


namespace NUMINAMATH_CALUDE_expression_evaluation_l2125_212587

theorem expression_evaluation : 2 - (-3) - 4 + (-5) - 6 + 7 = -3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2125_212587


namespace NUMINAMATH_CALUDE_largest_divisor_of_polynomial_l2125_212541

theorem largest_divisor_of_polynomial (n : ℤ) : 
  ∃ (k : ℕ), k > 0 ∧ (∀ (m : ℤ), (m^4 - 5*m^2 + 6) % k = 0) ∧ 
  (∀ (l : ℕ), l > k → ∃ (m : ℤ), (m^4 - 5*m^2 + 6) % l ≠ 0) → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_largest_divisor_of_polynomial_l2125_212541


namespace NUMINAMATH_CALUDE_max_product_of_radii_l2125_212523

/-- Two circles C₁ and C₂ are externally tangent -/
def externally_tangent (a b : ℝ) : Prop :=
  a + b = 3

/-- The equation of circle C₁ -/
def circle_C₁ (a : ℝ) (x y : ℝ) : Prop :=
  (x + a)^2 + (y - 2)^2 = 1

/-- The equation of circle C₂ -/
def circle_C₂ (b : ℝ) (x y : ℝ) : Prop :=
  (x - b)^2 + (y - 2)^2 = 4

theorem max_product_of_radii (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) 
  (h_tangent : externally_tangent a b) :
  a * b ≤ 9/4 ∧ ∃ (a₀ b₀ : ℝ), a₀ * b₀ = 9/4 ∧ externally_tangent a₀ b₀ := by
  sorry

end NUMINAMATH_CALUDE_max_product_of_radii_l2125_212523


namespace NUMINAMATH_CALUDE_number_and_square_sum_l2125_212592

theorem number_and_square_sum (x : ℝ) : x + x^2 = 306 → x = 17 := by
  sorry

end NUMINAMATH_CALUDE_number_and_square_sum_l2125_212592


namespace NUMINAMATH_CALUDE_sum_of_cubes_l2125_212513

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 2) (h2 : x * y = 3) : x^3 + y^3 = -10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l2125_212513


namespace NUMINAMATH_CALUDE_complex_modulus_example_l2125_212557

theorem complex_modulus_example : Complex.abs (3 - 10 * Complex.I * Real.sqrt 3) = Real.sqrt 309 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_example_l2125_212557


namespace NUMINAMATH_CALUDE_distance_difference_l2125_212503

/-- Clara's travel rate in miles per hour -/
def clara_rate : ℝ := 3.75

/-- Daniel's travel rate in miles per hour -/
def daniel_rate : ℝ := 3

/-- Time period in hours -/
def time : ℝ := 5

/-- Theorem stating the difference in distance traveled -/
theorem distance_difference : clara_rate * time - daniel_rate * time = 3.75 := by
  sorry

end NUMINAMATH_CALUDE_distance_difference_l2125_212503


namespace NUMINAMATH_CALUDE_cafe_outdoor_tables_l2125_212588

/-- The number of indoor tables -/
def indoor_tables : ℕ := 9

/-- The number of chairs per indoor table -/
def chairs_per_indoor_table : ℕ := 10

/-- The number of chairs per outdoor table -/
def chairs_per_outdoor_table : ℕ := 3

/-- The total number of chairs -/
def total_chairs : ℕ := 123

/-- The number of outdoor tables -/
def outdoor_tables : ℕ := (total_chairs - indoor_tables * chairs_per_indoor_table) / chairs_per_outdoor_table

theorem cafe_outdoor_tables : outdoor_tables = 11 := by
  sorry

end NUMINAMATH_CALUDE_cafe_outdoor_tables_l2125_212588


namespace NUMINAMATH_CALUDE_marathon_average_time_l2125_212519

/-- Given Casey's time to complete a marathon and Zendaya's relative time compared to Casey,
    calculate the average time for both to complete the race. -/
theorem marathon_average_time (casey_time : ℝ) (zendaya_relative_time : ℝ) :
  casey_time = 6 →
  zendaya_relative_time = 1/3 →
  let zendaya_time := casey_time + zendaya_relative_time * casey_time
  (casey_time + zendaya_time) / 2 = 7 := by
  sorry


end NUMINAMATH_CALUDE_marathon_average_time_l2125_212519


namespace NUMINAMATH_CALUDE_no_prime_cubic_polynomial_l2125_212582

theorem no_prime_cubic_polynomial :
  ¬ ∃ (n : ℕ), n > 0 ∧ Nat.Prime (n^3 - 9*n^2 + 27*n - 28) := by
  sorry

end NUMINAMATH_CALUDE_no_prime_cubic_polynomial_l2125_212582


namespace NUMINAMATH_CALUDE_sum_of_sqrt_products_gt_sum_of_numbers_l2125_212583

theorem sum_of_sqrt_products_gt_sum_of_numbers 
  (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hxy : |x - y| < 2) (hyz : |y - z| < 2) (hzx : |z - x| < 2) : 
  Real.sqrt (x * y + 1) + Real.sqrt (y * z + 1) + Real.sqrt (z * x + 1) > x + y + z := by
  sorry

end NUMINAMATH_CALUDE_sum_of_sqrt_products_gt_sum_of_numbers_l2125_212583


namespace NUMINAMATH_CALUDE_simplify_expression_l2125_212514

theorem simplify_expression (x : ℝ) : (3 * x + 15) + (97 * x + 45) = 100 * x + 60 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2125_212514


namespace NUMINAMATH_CALUDE_train_speed_theorem_l2125_212510

/-- The speed of the first train in km/h -/
def first_train_speed : ℝ := 70

/-- The speed of the second train in km/h -/
def second_train_speed : ℝ := 80

/-- The time difference between the starts of the two trains in hours -/
def time_difference : ℝ := 1

/-- The total travel time of the first train in hours -/
def first_train_travel_time : ℝ := 8

/-- The total travel time of the second train in hours -/
def second_train_travel_time : ℝ := 7

theorem train_speed_theorem : 
  first_train_speed * first_train_travel_time = 
  second_train_speed * second_train_travel_time :=
by sorry

end NUMINAMATH_CALUDE_train_speed_theorem_l2125_212510


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2125_212571

-- Define the complex numbers
def z1 (y : ℝ) : ℂ := 3 + y * Complex.I
def z2 : ℂ := 2 - Complex.I

-- State the theorem
theorem complex_equation_solution :
  ∃ y : ℝ, z1 y / z2 = 1 + Complex.I ∧ y = 1 := by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2125_212571


namespace NUMINAMATH_CALUDE_intersection_slope_l2125_212504

/-- Given two lines that intersect at a point, prove the slope of one line. -/
theorem intersection_slope (m : ℝ) : 
  (∀ x y, y = -2 * x + 3 → y = m * x + 4) → -- Line p: y = -2x + 3, Line q: y = mx + 4
  1 = -2 * 1 + 3 →                         -- Point (1, 1) satisfies line p
  1 = m * 1 + 4 →                          -- Point (1, 1) satisfies line q
  m = -3 := by sorry

end NUMINAMATH_CALUDE_intersection_slope_l2125_212504


namespace NUMINAMATH_CALUDE_intersection_line_of_circles_l2125_212598

/-- Represents a circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The line passing through the intersection points of two circles -/
def intersectionLine (c1 c2 : Circle) : ℝ → ℝ → Prop :=
  fun x y => x + y = 6.5

theorem intersection_line_of_circles :
  let c1 : Circle := { center := (5, -2), radius := 7 }
  let c2 : Circle := { center := (-1, 5), radius := 5 }
  ∃ (p1 p2 : ℝ × ℝ),
    (p1.1 - c1.center.1)^2 + (p1.2 - c1.center.2)^2 = c1.radius^2 ∧
    (p1.1 - c2.center.1)^2 + (p1.2 - c2.center.2)^2 = c2.radius^2 ∧
    (p2.1 - c1.center.1)^2 + (p2.2 - c1.center.2)^2 = c1.radius^2 ∧
    (p2.1 - c2.center.1)^2 + (p2.2 - c2.center.2)^2 = c2.radius^2 ∧
    p1 ≠ p2 ∧
    intersectionLine c1 c2 p1.1 p1.2 ∧
    intersectionLine c1 c2 p2.1 p2.2 :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_line_of_circles_l2125_212598


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l2125_212501

def P : Set ℝ := {0, 1, 2}
def Q : Set ℝ := {y | ∃ x : ℝ, y = 3^x}

theorem intersection_of_P_and_Q : P ∩ Q = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l2125_212501


namespace NUMINAMATH_CALUDE_quadratic_form_sum_l2125_212548

theorem quadratic_form_sum (a h k : ℝ) : 
  (∀ x, 4 * x^2 - 8 * x - 3 = a * (x - h)^2 + k) →
  a + h + k = -2 := by sorry

end NUMINAMATH_CALUDE_quadratic_form_sum_l2125_212548


namespace NUMINAMATH_CALUDE_scaled_circle_equation_l2125_212559

/-- Given a circle and a scaling transformation, prove the equation of the resulting curve -/
theorem scaled_circle_equation (x y x' y' : ℝ) :
  (x^2 + y^2 = 1) →  -- Circle equation
  (x' = 2*x) →       -- Scaling for x
  (y' = 3*y) →       -- Scaling for y
  (x'^2/4 + y'^2/9 = 1) -- Resulting curve equation
:= by sorry

end NUMINAMATH_CALUDE_scaled_circle_equation_l2125_212559


namespace NUMINAMATH_CALUDE_back_seat_capacity_is_twelve_l2125_212578

/-- Represents the seating arrangement and capacity of a bus -/
structure BusSeating where
  left_seats : ℕ
  right_seats : ℕ
  people_per_seat : ℕ
  total_capacity : ℕ

/-- Calculates the number of people who can sit in the back seat of the bus -/
def back_seat_capacity (bus : BusSeating) : ℕ :=
  bus.total_capacity - (bus.left_seats + bus.right_seats) * bus.people_per_seat

/-- Theorem stating the number of people who can sit in the back seat -/
theorem back_seat_capacity_is_twelve :
  ∀ (bus : BusSeating),
    bus.left_seats = 15 →
    bus.right_seats = bus.left_seats - 3 →
    bus.people_per_seat = 3 →
    bus.total_capacity = 93 →
    back_seat_capacity bus = 12 := by
  sorry


end NUMINAMATH_CALUDE_back_seat_capacity_is_twelve_l2125_212578


namespace NUMINAMATH_CALUDE_benny_stored_bales_l2125_212537

/-- The number of bales Benny stored in the barn -/
def bales_stored (initial_bales final_bales : ℕ) : ℕ :=
  final_bales - initial_bales

/-- Proof that Benny stored 35 bales in the barn -/
theorem benny_stored_bales : 
  let initial_bales : ℕ := 47
  let final_bales : ℕ := 82
  bales_stored initial_bales final_bales = 35 := by
  sorry

end NUMINAMATH_CALUDE_benny_stored_bales_l2125_212537


namespace NUMINAMATH_CALUDE_inequality_proof_l2125_212502

-- Define the set M
def M : Set ℝ := {x | -2 < |x - 1| - |x + 2| ∧ |x - 1| - |x + 2| < 0}

-- State the theorem
theorem inequality_proof (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) : 
  (|(1/3 : ℝ) * a + (1/6 : ℝ) * b| < (1/4 : ℝ)) ∧ 
  (|1 - 4 * a * b| > 2 * |a - b|) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2125_212502


namespace NUMINAMATH_CALUDE_quadratic_function_theorem_l2125_212586

-- Define the quadratic function f(x)
def f (a b x : ℝ) : ℝ := x^2 + a*x + b

-- State the theorem
theorem quadratic_function_theorem (a b : ℝ) :
  (∀ x, (f (f a b x + 2*x) a b) / (f a b x) = x^2 + 2023*x + 2040) →
  a = 2021 ∧ b = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_theorem_l2125_212586


namespace NUMINAMATH_CALUDE_maria_earnings_l2125_212551

/-- Calculates the total earnings of a flower saleswoman over three days --/
def flower_sales_earnings (tulip_price rose_price : ℚ) 
  (day1_tulips day1_roses : ℕ) 
  (day2_multiplier : ℚ) 
  (day3_tulip_percentage : ℚ) 
  (day3_roses : ℕ) : ℚ :=
  let day1_earnings := tulip_price * day1_tulips + rose_price * day1_roses
  let day2_earnings := day2_multiplier * day1_earnings
  let day3_tulips := day3_tulip_percentage * (day2_multiplier * day1_tulips)
  let day3_earnings := tulip_price * day3_tulips + rose_price * day3_roses
  day1_earnings + day2_earnings + day3_earnings

/-- Theorem stating that Maria's total earnings over three days is $420 --/
theorem maria_earnings : 
  flower_sales_earnings 2 3 30 20 2 (1/10) 16 = 420 := by
  sorry

end NUMINAMATH_CALUDE_maria_earnings_l2125_212551


namespace NUMINAMATH_CALUDE_quadratic_reciprocal_roots_l2125_212534

theorem quadratic_reciprocal_roots (a b c : ℤ) (x₁ x₂ : ℚ) : 
  a ≠ 0 →
  (∀ x, a * x^2 + b * x + c = 0 ↔ x = x₁ ∨ x = x₂) →
  (x₁ * x₂ = 1) →
  (x₁ + x₂ = 4) →
  (∃ n m : ℤ, x₁ = n ∧ x₂ = m) →
  (c = a ∧ b = -4*a) :=
sorry

end NUMINAMATH_CALUDE_quadratic_reciprocal_roots_l2125_212534


namespace NUMINAMATH_CALUDE_repeating_decimal_value_l2125_212542

-- Define the repeating decimal 0.454545...
def repeating_decimal : ℚ := 0.454545

-- Theorem statement
theorem repeating_decimal_value : repeating_decimal = 5 / 11 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_value_l2125_212542


namespace NUMINAMATH_CALUDE_certain_number_bound_l2125_212500

theorem certain_number_bound (n : ℝ) : 
  (∀ x : ℝ, x ≤ 2 → 6.1 * 10^x < n) → n > 610 := by sorry

end NUMINAMATH_CALUDE_certain_number_bound_l2125_212500


namespace NUMINAMATH_CALUDE_f_properties_l2125_212555

/-- The function f(x) with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x * |x - 2*a| + a^2 - 3*a

/-- Theorem stating the properties of the function f and its zeros -/
theorem f_properties (a : ℝ) (x₁ x₂ x₃ : ℝ) :
  (∃ (x₁ x₂ x₃ : ℝ), x₁ < x₂ ∧ x₂ < x₃ ∧ 
    f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0 ∧ 
    x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃) →
  (3/2 < a ∧ a < 3) ∧
  (2*(Real.sqrt 2 + 1)/3 < 1/x₁ + 1/x₂ + 1/x₃) :=
by sorry


end NUMINAMATH_CALUDE_f_properties_l2125_212555


namespace NUMINAMATH_CALUDE_travel_ways_eq_nine_l2125_212549

/-- The number of different ways to travel from location A to location B in one day -/
def travel_ways (car_departures train_departures ship_departures : ℕ) : ℕ :=
  car_departures + train_departures + ship_departures

/-- Theorem: The number of different ways to travel is 9 given the specified departures -/
theorem travel_ways_eq_nine :
  travel_ways 3 4 2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_travel_ways_eq_nine_l2125_212549


namespace NUMINAMATH_CALUDE_abc_product_is_one_l2125_212596

theorem abc_product_is_one (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a)
  (h1 : a + 1 / b^2 = b + 1 / c^2) (h2 : b + 1 / c^2 = c + 1 / a^2) :
  |a * b * c| = 1 := by
sorry

end NUMINAMATH_CALUDE_abc_product_is_one_l2125_212596


namespace NUMINAMATH_CALUDE_function_positivity_range_l2125_212521

theorem function_positivity_range (a : ℝ) (h : a ∈ Set.Icc (-1) 1) :
  let f : ℝ → ℝ := λ x => x^2 + (a - 4) * x + 4 - 2 * a
  (∀ x, f x > 0) → {x | f x > 0} = {x | x < 2 ∨ x > 2} := by
  sorry

end NUMINAMATH_CALUDE_function_positivity_range_l2125_212521


namespace NUMINAMATH_CALUDE_inequality_proof_l2125_212569

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + 1)^2 / b + (b + 1)^2 / a ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2125_212569


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l2125_212574

theorem polynomial_divisibility (m : ℚ) :
  (∀ x, (x^4 - 5*x^2 + 4*x - m) % (2*x + 1) = 0) → m = -51/16 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l2125_212574


namespace NUMINAMATH_CALUDE_smallest_sqrt_x_minus_one_l2125_212577

theorem smallest_sqrt_x_minus_one :
  ∀ x : ℝ, 
    (Real.sqrt (x - 1) ≥ 0) ∧ 
    (Real.sqrt (x - 1) = 0 ↔ x = 1) :=
by sorry

end NUMINAMATH_CALUDE_smallest_sqrt_x_minus_one_l2125_212577


namespace NUMINAMATH_CALUDE_peanuts_lost_l2125_212564

def initial_peanuts : ℕ := 74
def final_peanuts : ℕ := 15

theorem peanuts_lost : initial_peanuts - final_peanuts = 59 := by
  sorry

end NUMINAMATH_CALUDE_peanuts_lost_l2125_212564


namespace NUMINAMATH_CALUDE_sector_angle_l2125_212579

/-- Given a circular sector with circumference 4 and area 1, prove that its central angle is 2 radians -/
theorem sector_angle (r : ℝ) (l : ℝ) (α : ℝ) 
  (h_circumference : 2 * r + l = 4)
  (h_area : (1 / 2) * l * r = 1) :
  α = 2 :=
sorry

end NUMINAMATH_CALUDE_sector_angle_l2125_212579


namespace NUMINAMATH_CALUDE_number_problem_l2125_212517

theorem number_problem : ∃ x : ℝ, (0.2 * x = 0.4 * 140 + 80) ∧ (x = 680) := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l2125_212517


namespace NUMINAMATH_CALUDE_fibSeriesSum_l2125_212594

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Sum of the infinite series of Fibonacci numbers divided by powers of 5 -/
noncomputable def fibSeries : ℝ := ∑' n, (fib n : ℝ) / 5^n

/-- The sum of the infinite series of Fibonacci numbers divided by powers of 5 is 5/19 -/
theorem fibSeriesSum : fibSeries = 5 / 19 := by sorry

end NUMINAMATH_CALUDE_fibSeriesSum_l2125_212594


namespace NUMINAMATH_CALUDE_x_coordinate_range_l2125_212589

-- Define the circle M
def circle_M (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 4

-- Define the line l
def line_l (x y : ℝ) : Prop := x + y = 6

-- Define a point on the circle
def point_on_circle (x y : ℝ) : Prop := circle_M x y

-- Define a point on the line
def point_on_line (x y : ℝ) : Prop := line_l x y

-- Define the angle between three points
def angle (A B C : ℝ × ℝ) : ℝ := sorry

-- Main theorem
theorem x_coordinate_range :
  ∀ (A B C : ℝ × ℝ),
  point_on_line A.1 A.2 →
  point_on_circle B.1 B.2 →
  point_on_circle C.1 C.2 →
  angle A B C = π/3 →
  1 ≤ A.1 ∧ A.1 ≤ 5 :=
by sorry

end NUMINAMATH_CALUDE_x_coordinate_range_l2125_212589


namespace NUMINAMATH_CALUDE_lcm_gcd_product_l2125_212584

theorem lcm_gcd_product (a b : ℕ) (h1 : a > 0) (h2 : b > 0) 
  (h_lcm : Nat.lcm a b = 60) (h_gcd : Nat.gcd a b = 5) : 
  a * b = 300 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_product_l2125_212584


namespace NUMINAMATH_CALUDE_maurice_horseback_rides_l2125_212567

theorem maurice_horseback_rides (maurice_visit_rides : ℕ) 
                                (matt_with_maurice : ℕ) 
                                (matt_alone_rides : ℕ) : 
  maurice_visit_rides = 8 →
  matt_with_maurice = 8 →
  matt_alone_rides = 16 →
  matt_with_maurice + matt_alone_rides = 3 * maurice_before_visit →
  maurice_before_visit = 8 := by
  sorry

def maurice_before_visit : ℕ := 8

end NUMINAMATH_CALUDE_maurice_horseback_rides_l2125_212567


namespace NUMINAMATH_CALUDE_negation_equivalence_l2125_212570

theorem negation_equivalence (x y : ℤ) :
  ¬(Even (x + y) → Even x ∧ Even y) ↔ (¬Even (x + y) → ¬(Even x ∧ Even y)) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2125_212570


namespace NUMINAMATH_CALUDE_lucy_cookie_sales_l2125_212554

/-- Given that Robyn sold 16 packs of cookies and together with Lucy they sold 35 packs,
    prove that Lucy sold 19 packs. -/
theorem lucy_cookie_sales (robyn_sales : ℕ) (total_sales : ℕ) (h1 : robyn_sales = 16) (h2 : total_sales = 35) :
  total_sales - robyn_sales = 19 := by
  sorry

end NUMINAMATH_CALUDE_lucy_cookie_sales_l2125_212554


namespace NUMINAMATH_CALUDE_quadratic_always_nonnegative_implies_m_range_l2125_212565

theorem quadratic_always_nonnegative_implies_m_range (m : ℝ) :
  (∀ x : ℝ, x^2 + m*x + 2*m - 3 ≥ 0) → 2 ≤ m ∧ m ≤ 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_nonnegative_implies_m_range_l2125_212565


namespace NUMINAMATH_CALUDE_total_ways_to_choose_courses_l2125_212533

-- Define the number of courses of each type
def num_courses_A : ℕ := 4
def num_courses_B : ℕ := 2

-- Define the total number of courses to be chosen
def total_courses_to_choose : ℕ := 4

-- Define the function to calculate the number of ways to choose courses
def num_ways_to_choose : ℕ := 
  (num_courses_B.choose 1 * num_courses_A.choose 3) +
  (num_courses_B.choose 2 * num_courses_A.choose 2)

-- Theorem statement
theorem total_ways_to_choose_courses : num_ways_to_choose = 14 := by
  sorry


end NUMINAMATH_CALUDE_total_ways_to_choose_courses_l2125_212533


namespace NUMINAMATH_CALUDE_fraction_to_decimal_decimal_representation_main_result_l2125_212562

theorem fraction_to_decimal : (47 : ℚ) / (2 * 5^4) = (376 : ℚ) / 10000 := by sorry

theorem decimal_representation : (376 : ℚ) / 10000 = 0.0376 := by sorry

theorem main_result : (47 : ℚ) / (2 * 5^4) = 0.0376 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_decimal_representation_main_result_l2125_212562


namespace NUMINAMATH_CALUDE_louis_current_age_l2125_212520

/-- Carla's current age -/
def carla_age : ℕ := 30 - 6

/-- Louis's current age -/
def louis_age : ℕ := 55 - carla_age

theorem louis_current_age : louis_age = 31 := by
  sorry

end NUMINAMATH_CALUDE_louis_current_age_l2125_212520


namespace NUMINAMATH_CALUDE_number_line_essential_elements_l2125_212525

/-- Represents the essential elements of a number line -/
inductive NumberLineElement
  | PositiveDirection
  | Origin
  | UnitLength

/-- The set of essential elements of a number line -/
def essentialElements : Set NumberLineElement :=
  {NumberLineElement.PositiveDirection, NumberLineElement.Origin, NumberLineElement.UnitLength}

/-- Theorem stating that the essential elements of a number line are precisely
    positive direction, origin, and unit length -/
theorem number_line_essential_elements :
  ∀ (e : NumberLineElement), e ∈ essentialElements ↔
    (e = NumberLineElement.PositiveDirection ∨
     e = NumberLineElement.Origin ∨
     e = NumberLineElement.UnitLength) :=
by sorry

end NUMINAMATH_CALUDE_number_line_essential_elements_l2125_212525


namespace NUMINAMATH_CALUDE_larger_number_l2125_212591

theorem larger_number (x y : ℝ) (h_sum : x + y = 40) (h_diff : x - y = 6) : x = 23 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_l2125_212591


namespace NUMINAMATH_CALUDE_largest_negative_integer_l2125_212585

theorem largest_negative_integer :
  ∀ n : ℤ, n < 0 → n ≤ -1 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_negative_integer_l2125_212585


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l2125_212561

def U : Set ℕ := {1, 2, 3, 4}
def A : Set ℕ := {1, 3}

theorem complement_of_A_in_U :
  {x ∈ U | x ∉ A} = {2, 4} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l2125_212561


namespace NUMINAMATH_CALUDE_cosine_calculation_l2125_212572

theorem cosine_calculation : Real.cos (π/3) - 2⁻¹ + Real.sqrt ((-2)^2) - (π-3)^0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_cosine_calculation_l2125_212572


namespace NUMINAMATH_CALUDE_winning_candidate_vote_percentage_l2125_212560

/-- The percentage of votes received by the winning candidate in an election with three candidates -/
theorem winning_candidate_vote_percentage 
  (votes : Fin 3 → ℕ)
  (h1 : votes 0 = 3000)
  (h2 : votes 1 = 5000)
  (h3 : votes 2 = 15000) :
  (votes 2 : ℚ) / (votes 0 + votes 1 + votes 2) * 100 = 15000 / 23000 * 100 := by
  sorry

#eval (15000 : ℚ) / 23000 * 100 -- To display the approximate result

end NUMINAMATH_CALUDE_winning_candidate_vote_percentage_l2125_212560


namespace NUMINAMATH_CALUDE_total_bikes_l2125_212595

theorem total_bikes (jungkook_bikes : ℕ) (yoongi_bikes : ℕ) 
  (h1 : jungkook_bikes = 3) (h2 : yoongi_bikes = 4) : 
  jungkook_bikes + yoongi_bikes = 7 := by
  sorry

end NUMINAMATH_CALUDE_total_bikes_l2125_212595


namespace NUMINAMATH_CALUDE_table_area_is_175_l2125_212576

def table_area_proof (total_runner_area single_layer_area double_layer_area triple_layer_area coverage_percentage : ℝ) : Prop :=
  total_runner_area = 212 ∧
  double_layer_area = 24 ∧
  triple_layer_area = 24 ∧
  coverage_percentage = 0.80 ∧
  single_layer_area + double_layer_area + triple_layer_area = coverage_percentage * 175 ∧
  single_layer_area + 2 * double_layer_area + 3 * triple_layer_area = total_runner_area

theorem table_area_is_175 :
  ∃ (total_runner_area single_layer_area double_layer_area triple_layer_area coverage_percentage : ℝ),
    table_area_proof total_runner_area single_layer_area double_layer_area triple_layer_area coverage_percentage :=
by
  sorry

end NUMINAMATH_CALUDE_table_area_is_175_l2125_212576


namespace NUMINAMATH_CALUDE_total_samplers_percentage_l2125_212511

/-- Represents the percentage of customers for a specific candy type -/
structure CandyData where
  caught : ℝ
  notCaught : ℝ

/-- Represents the data for all candy types -/
structure CandyStore where
  A : CandyData
  B : CandyData
  C : CandyData
  D : CandyData

/-- Calculates the total percentage of customers who sample any type of candy -/
def totalSamplers (store : CandyStore) : ℝ :=
  store.A.caught + store.A.notCaught +
  store.B.caught + store.B.notCaught +
  store.C.caught + store.C.notCaught +
  store.D.caught + store.D.notCaught

/-- The candy store data -/
def candyStoreData : CandyStore :=
  { A := { caught := 12, notCaught := 7 }
    B := { caught := 5,  notCaught := 6 }
    C := { caught := 9,  notCaught := 3 }
    D := { caught := 4,  notCaught := 8 } }

theorem total_samplers_percentage :
  totalSamplers candyStoreData = 54 := by sorry

end NUMINAMATH_CALUDE_total_samplers_percentage_l2125_212511


namespace NUMINAMATH_CALUDE_quadrilateral_triangle_product_l2125_212552

/-- Represents a convex quadrilateral with its four triangles formed by diagonals -/
structure ConvexQuadrilateral where
  /-- Areas of the four triangles formed by diagonals -/
  triangle_areas : Fin 4 → ℕ

/-- Theorem stating that the product of the four triangle areas in a convex quadrilateral
    cannot be congruent to 2014 modulo 10000 -/
theorem quadrilateral_triangle_product (q : ConvexQuadrilateral) :
  (q.triangle_areas 0 * q.triangle_areas 1 * q.triangle_areas 2 * q.triangle_areas 3) % 10000 ≠ 2014 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_triangle_product_l2125_212552


namespace NUMINAMATH_CALUDE_factorial_divides_theorem_l2125_212524

def divides (a b : ℤ) : Prop := ∃ k : ℤ, b = a * k

theorem factorial_divides_theorem (a : ℤ) :
  (∃ S : Set ℕ, Set.Infinite S ∧ ∀ n ∈ S, divides (n.factorial + a) ((2*n).factorial)) →
  a = 0 := by sorry

end NUMINAMATH_CALUDE_factorial_divides_theorem_l2125_212524


namespace NUMINAMATH_CALUDE_prob_monochromatic_triangle_in_hexagon_l2125_212536

/-- A regular hexagon with randomly colored edges -/
structure ColoredHexagon where
  /-- The number of sides in a regular hexagon -/
  numSides : Nat
  /-- The number of diagonals in a regular hexagon -/
  numDiagonals : Nat
  /-- The total number of edges (sides + diagonals) -/
  numEdges : Nat
  /-- The number of possible triangles in a hexagon -/
  numTriangles : Nat
  /-- The probability of an edge being a specific color -/
  probEdgeColor : ℚ
  /-- The probability of a triangle not being monochromatic -/
  probNonMonochromatic : ℚ

/-- The probability of having at least one monochromatic triangle in a colored hexagon -/
def probMonochromaticTriangle (h : ColoredHexagon) : ℚ :=
  1 - (h.probNonMonochromatic ^ h.numTriangles)

/-- Theorem stating the probability of a monochromatic triangle in a randomly colored hexagon -/
theorem prob_monochromatic_triangle_in_hexagon :
  ∃ (h : ColoredHexagon),
    h.numSides = 6 ∧
    h.numDiagonals = 9 ∧
    h.numEdges = 15 ∧
    h.numTriangles = 20 ∧
    h.probEdgeColor = 1/2 ∧
    h.probNonMonochromatic = 3/4 ∧
    probMonochromaticTriangle h = 253/256 := by
  sorry

end NUMINAMATH_CALUDE_prob_monochromatic_triangle_in_hexagon_l2125_212536


namespace NUMINAMATH_CALUDE_givenPointInFirstQuadrant_l2125_212535

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the first quadrant -/
def isInFirstQuadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y > 0

/-- The given point -/
def givenPoint : Point :=
  { x := 2, y := 1 }

/-- Theorem stating that the given point is in the first quadrant -/
theorem givenPointInFirstQuadrant : isInFirstQuadrant givenPoint := by
  sorry

end NUMINAMATH_CALUDE_givenPointInFirstQuadrant_l2125_212535


namespace NUMINAMATH_CALUDE_complex_power_magnitude_l2125_212575

theorem complex_power_magnitude : Complex.abs ((2 : ℂ) + Complex.I) ^ 8 = 625 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_magnitude_l2125_212575


namespace NUMINAMATH_CALUDE_pizza_toppings_l2125_212506

/-- Given a pizza with the following properties:
  * Total slices: 16
  * Slices with pepperoni: 8
  * Slices with mushrooms: 12
  * Plain slices: 2
  Prove that the number of slices with both pepperoni and mushrooms is 6. -/
theorem pizza_toppings (total : Nat) (pepperoni : Nat) (mushrooms : Nat) (plain : Nat)
    (h_total : total = 16)
    (h_pepperoni : pepperoni = 8)
    (h_mushrooms : mushrooms = 12)
    (h_plain : plain = 2) :
    ∃ (both : Nat), both = 6 ∧
      pepperoni + mushrooms - both = total - plain :=
by sorry

end NUMINAMATH_CALUDE_pizza_toppings_l2125_212506


namespace NUMINAMATH_CALUDE_max_area_rectangle_l2125_212573

/-- The parabola function y = 4 - x^2 --/
def parabola (x : ℝ) : ℝ := 4 - x^2

/-- The area function of the rectangle --/
def area (x : ℝ) : ℝ := 2 * x * (4 - x^2)

/-- The theorem stating the maximum area of the rectangle --/
theorem max_area_rectangle :
  ∃ (x : ℝ), x > 0 ∧ x < 2 ∧
  (∀ (y : ℝ), y > 0 ∧ y < 2 → area x ≥ area y) ∧
  (2 * x = (4 / 3) * Real.sqrt 3) := by
  sorry

#check max_area_rectangle

end NUMINAMATH_CALUDE_max_area_rectangle_l2125_212573


namespace NUMINAMATH_CALUDE_circles_common_chord_l2125_212590

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 10*x - 10*y = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 2*y - 40 = 0

-- Define the line
def common_chord (x y : ℝ) : Prop := x + 3*y - 10 = 0

-- Theorem statement
theorem circles_common_chord :
  ∃ (p1 p2 : ℝ × ℝ), p1 ≠ p2 ∧ 
    circle1 p1.1 p1.2 ∧ circle1 p2.1 p2.2 ∧
    circle2 p1.1 p1.2 ∧ circle2 p2.1 p2.2 →
  ∀ (x y : ℝ), circle1 x y ∧ circle2 x y → common_chord x y :=
sorry

end NUMINAMATH_CALUDE_circles_common_chord_l2125_212590


namespace NUMINAMATH_CALUDE_road_repair_workers_l2125_212507

theorem road_repair_workers (group1_people : ℕ) (group1_days : ℕ) (group1_hours : ℕ)
                             (group2_days : ℕ) (group2_hours : ℕ) :
  group1_people = 69 →
  group1_days = 12 →
  group1_hours = 5 →
  group2_days = 23 →
  group2_hours = 6 →
  group1_people * group1_days * group1_hours = 
    ((group1_people * group1_days * group1_hours) / (group2_days * group2_hours) : ℕ) * group2_days * group2_hours →
  ((group1_people * group1_days * group1_hours) / (group2_days * group2_hours) : ℕ) = 30 :=
by sorry

end NUMINAMATH_CALUDE_road_repair_workers_l2125_212507


namespace NUMINAMATH_CALUDE_product_inequality_l2125_212509

theorem product_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_sum : a + b + c + 2 = a * b * c) : 
  (a + 1) * (b + 1) * (c + 1) ≥ 27 ∧ 
  ((a + 1) * (b + 1) * (c + 1) = 27 ↔ a = 2 ∧ b = 2 ∧ c = 2) := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l2125_212509


namespace NUMINAMATH_CALUDE_kamal_english_marks_l2125_212553

/-- Kamal's marks in English given his other marks and average --/
theorem kamal_english_marks (math physics chem bio : ℕ) (avg : ℚ) :
  math = 65 →
  physics = 82 →
  chem = 67 →
  bio = 85 →
  avg = 75 →
  (math + physics + chem + bio + english : ℚ) / 5 = avg →
  english = 76 := by
  sorry

end NUMINAMATH_CALUDE_kamal_english_marks_l2125_212553


namespace NUMINAMATH_CALUDE_students_playing_at_least_one_sport_l2125_212526

/-- The number of students who like to play basketball -/
def B : ℕ := 7

/-- The number of students who like to play cricket -/
def C : ℕ := 10

/-- The number of students who like to play soccer -/
def S : ℕ := 8

/-- The number of students who like to play all three sports -/
def BCS : ℕ := 2

/-- The number of students who like to play both basketball and cricket -/
def BC : ℕ := 5

/-- The number of students who like to play both basketball and soccer -/
def BS : ℕ := 4

/-- The number of students who like to play both cricket and soccer -/
def CS : ℕ := 3

/-- The theorem stating that the number of students who like to play at least one sport is 21 -/
theorem students_playing_at_least_one_sport : 
  B + C + S - ((BC - BCS) + (BS - BCS) + (CS - BCS)) + BCS = 21 := by
  sorry

end NUMINAMATH_CALUDE_students_playing_at_least_one_sport_l2125_212526


namespace NUMINAMATH_CALUDE_candy_container_count_l2125_212518

theorem candy_container_count : ℕ := by
  -- Define the number of people
  let people : ℕ := 157

  -- Define the number of candies each person receives
  let candies_per_person : ℕ := 235

  -- Define the number of candies left after distribution
  let leftover_candies : ℕ := 98

  -- Define the total number of candies
  let total_candies : ℕ := people * candies_per_person + leftover_candies

  -- Prove that the total number of candies is 36,993
  have h : total_candies = 36993 := by sorry

  -- Return the result
  exact 36993

end NUMINAMATH_CALUDE_candy_container_count_l2125_212518


namespace NUMINAMATH_CALUDE_austin_friday_hours_l2125_212538

/-- Represents the problem of Austin saving for a bicycle --/
def bicycle_savings (hourly_rate : ℚ) (monday_hours : ℚ) (wednesday_hours : ℚ) (total_weeks : ℕ) (bicycle_cost : ℚ) : Prop :=
  let monday_earnings := hourly_rate * monday_hours
  let wednesday_earnings := hourly_rate * wednesday_hours
  let weekly_earnings := monday_earnings + wednesday_earnings
  let total_earnings_without_friday := weekly_earnings * total_weeks
  let remaining_earnings_needed := bicycle_cost - total_earnings_without_friday
  let friday_hours := remaining_earnings_needed / (hourly_rate * total_weeks)
  friday_hours = 3

/-- Theorem stating that Austin needs to work 3 hours on Fridays --/
theorem austin_friday_hours : 
  bicycle_savings 5 2 1 6 180 := by sorry

end NUMINAMATH_CALUDE_austin_friday_hours_l2125_212538


namespace NUMINAMATH_CALUDE_min_cost_to_1981_l2125_212558

/-- Cost of multiplying by 3 -/
def mult_cost : ℕ := 5

/-- Cost of adding 4 -/
def add_cost : ℕ := 2

/-- The target number to reach -/
def target : ℕ := 1981

/-- A step in the calculation process -/
inductive Step
| Mult : Step  -- Multiply by 3
| Add : Step   -- Add 4

/-- A sequence of steps -/
def Sequence := List Step

/-- Calculate the result of applying a sequence of steps starting from 1 -/
def apply_sequence (s : Sequence) : ℕ :=
  s.foldl (λ n step => match step with
    | Step.Mult => n * 3
    | Step.Add => n + 4) 1

/-- Calculate the cost of a sequence of steps -/
def sequence_cost (s : Sequence) : ℕ :=
  s.foldl (λ cost step => cost + match step with
    | Step.Mult => mult_cost
    | Step.Add => add_cost) 0

/-- Theorem: The minimum cost to reach 1981 is 42 kopecks -/
theorem min_cost_to_1981 :
  ∃ (s : Sequence), apply_sequence s = target ∧
    sequence_cost s = 42 ∧
    ∀ (s' : Sequence), apply_sequence s' = target →
      sequence_cost s' ≥ sequence_cost s :=
by sorry

end NUMINAMATH_CALUDE_min_cost_to_1981_l2125_212558


namespace NUMINAMATH_CALUDE_homework_theorem_l2125_212545

/-- The number of possible homework situations for a given number of teachers and students -/
def homework_situations (num_teachers : ℕ) (num_students : ℕ) : ℕ :=
  num_teachers ^ num_students

/-- Theorem: With 3 teachers and 4 students, there are 3^4 possible homework situations -/
theorem homework_theorem :
  homework_situations 3 4 = 3^4 := by
  sorry

end NUMINAMATH_CALUDE_homework_theorem_l2125_212545


namespace NUMINAMATH_CALUDE_palindrome_count_ratio_l2125_212550

/-- A palindrome is a natural number whose decimal representation reads the same from left to right and right to left. -/
def isPalindrome (n : ℕ) : Prop := sorry

/-- The sum of digits of a natural number. -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Count of palindromes with even sum of digits between 10,000 and 999,999. -/
def evenSumPalindromeCount : ℕ := sorry

/-- Count of palindromes with odd sum of digits between 10,000 and 999,999. -/
def oddSumPalindromeCount : ℕ := sorry

theorem palindrome_count_ratio :
  evenSumPalindromeCount = 3 * oddSumPalindromeCount := by sorry

end NUMINAMATH_CALUDE_palindrome_count_ratio_l2125_212550


namespace NUMINAMATH_CALUDE_strip_width_problem_l2125_212556

theorem strip_width_problem (width1 width2 : ℕ) 
  (h1 : width1 = 44) (h2 : width2 = 33) : 
  Nat.gcd width1 width2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_strip_width_problem_l2125_212556


namespace NUMINAMATH_CALUDE_trapezoid_area_l2125_212512

/-- The area of a trapezoid with given base lengths and leg lengths -/
theorem trapezoid_area (b1 b2 l1 l2 : ℝ) (h : ℝ) 
  (hb1 : b1 = 10) 
  (hb2 : b2 = 21) 
  (hl1 : l1 = Real.sqrt 34) 
  (hl2 : l2 = 3 * Real.sqrt 5) 
  (hh : h^2 + 5^2 = 34) : 
  (b1 + b2) * h / 2 = 93 / 2 := by
  sorry

#check trapezoid_area

end NUMINAMATH_CALUDE_trapezoid_area_l2125_212512


namespace NUMINAMATH_CALUDE_reflection_sum_l2125_212531

/-- Given a line y = mx + b, if the reflection of point (-3, -1) across this line is (5, 3), then m + b = 1 -/
theorem reflection_sum (m b : ℝ) : 
  (∃ (x y : ℝ), 
    (x = ((-3) + 5) / 2 ∧ y = ((-1) + 3) / 2) ∧ 
    (y = m * x + b) ∧
    (m = -(5 - (-3)) / (3 - (-1))))
  → m + b = 1 := by sorry

end NUMINAMATH_CALUDE_reflection_sum_l2125_212531


namespace NUMINAMATH_CALUDE_solution_set_inequality_1_solution_set_inequality_2_l2125_212597

-- Problem 1
theorem solution_set_inequality_1 (x : ℝ) :
  (x + 2) / (x - 3) ≤ 0 ↔ -2 ≤ x ∧ x < 3 :=
sorry

-- Problem 2
theorem solution_set_inequality_2 (x a : ℝ) :
  (x + a) * (x - 1) > 0 ↔
    (a > -1 ∧ (x < -a ∨ x > 1)) ∨
    (a = -1 ∧ x ≠ 1) ∨
    (a < -1 ∧ (x < 1 ∨ x > -a)) :=
sorry

end NUMINAMATH_CALUDE_solution_set_inequality_1_solution_set_inequality_2_l2125_212597


namespace NUMINAMATH_CALUDE_absolute_difference_simplification_l2125_212547

theorem absolute_difference_simplification (a b : ℝ) 
  (ha : a < 0) (hab : a * b < 0) : 
  |a - b - 3| - |4 + b - a| = -1 := by
sorry

end NUMINAMATH_CALUDE_absolute_difference_simplification_l2125_212547


namespace NUMINAMATH_CALUDE_prime_sum_and_product_l2125_212563

def smallest_one_digit_prime : ℕ := 2
def second_smallest_two_digit_prime : ℕ := 13
def smallest_three_digit_prime : ℕ := 101

theorem prime_sum_and_product :
  (smallest_one_digit_prime + second_smallest_two_digit_prime + smallest_three_digit_prime = 116) ∧
  (smallest_one_digit_prime * second_smallest_two_digit_prime * smallest_three_digit_prime = 2626) := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_and_product_l2125_212563


namespace NUMINAMATH_CALUDE_favorite_season_fall_l2125_212581

theorem favorite_season_fall (total_students : ℕ) (winter_angle spring_angle : ℝ) :
  total_students = 600 →
  winter_angle = 90 →
  spring_angle = 60 →
  (total_students : ℝ) * (360 - winter_angle - spring_angle - 180) / 360 = 50 :=
by sorry

end NUMINAMATH_CALUDE_favorite_season_fall_l2125_212581


namespace NUMINAMATH_CALUDE_amithab_january_expenditure_l2125_212540

/-- Amithab's monthly expenditure problem -/
theorem amithab_january_expenditure
  (avg_jan_to_jun : ℝ)
  (july_expenditure : ℝ)
  (avg_feb_to_jul : ℝ)
  (h1 : avg_jan_to_jun = 4200)
  (h2 : july_expenditure = 1500)
  (h3 : avg_feb_to_jul = 4250) :
  6 * avg_jan_to_jun + july_expenditure = 6 * avg_feb_to_jul + 1800 :=
by sorry

end NUMINAMATH_CALUDE_amithab_january_expenditure_l2125_212540


namespace NUMINAMATH_CALUDE_chess_tournament_attendance_l2125_212599

theorem chess_tournament_attendance (total_students : ℕ) 
  (h1 : total_students = 24) 
  (h2 : ∃ chess_students : ℕ, chess_students = total_students / 3)
  (h3 : ∃ tournament_students : ℕ, tournament_students = (total_students / 3) / 2) :
  ∃ tournament_students : ℕ, tournament_students = 4 := by
sorry

end NUMINAMATH_CALUDE_chess_tournament_attendance_l2125_212599


namespace NUMINAMATH_CALUDE_rational_function_property_l2125_212593

theorem rational_function_property (f : ℚ → ℝ) 
  (add_prop : ∀ x y : ℚ, f (x + y) = f x + f y)
  (mul_prop : ∀ x y : ℚ, f (x * y) = f x * f y) :
  (∀ x : ℚ, f x = x) ∨ (∀ x : ℚ, f x = 0) := by
sorry

end NUMINAMATH_CALUDE_rational_function_property_l2125_212593


namespace NUMINAMATH_CALUDE_soap_cost_l2125_212543

/-- The total cost of soap given the number of bars, weight per bar, and price per pound -/
theorem soap_cost (num_bars : ℕ) (weight_per_bar : ℝ) (price_per_pound : ℝ) :
  num_bars = 20 →
  weight_per_bar = 1.5 →
  price_per_pound = 0.5 →
  num_bars * weight_per_bar * price_per_pound = 15 := by
  sorry

end NUMINAMATH_CALUDE_soap_cost_l2125_212543


namespace NUMINAMATH_CALUDE_fraction_simplification_l2125_212515

theorem fraction_simplification : (5 * 6) / 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2125_212515


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2125_212528

/-- Given a geometric sequence {a_n} with the specified conditions, 
    prove that the sum of the 6th, 7th, and 8th terms is 32. -/
theorem geometric_sequence_sum (a : ℕ → ℝ) : 
  (∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)) →  -- geometric sequence condition
  a 1 + a 2 + a 3 = 1 →                             -- first given condition
  a 2 + a 3 + a 4 = 2 →                             -- second given condition
  a 6 + a 7 + a 8 = 32 :=                           -- conclusion to prove
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2125_212528


namespace NUMINAMATH_CALUDE_fraction_equality_l2125_212527

theorem fraction_equality (a b : ℕ) (h1 : a + b = 1210) (h2 : b = 484) :
  (4 / 15 : ℚ) * a = (2 / 5 : ℚ) * b :=
sorry

end NUMINAMATH_CALUDE_fraction_equality_l2125_212527


namespace NUMINAMATH_CALUDE_opposite_of_seven_l2125_212505

-- Define the concept of opposite
def opposite (a : ℤ) : ℤ := -a

-- Theorem statement
theorem opposite_of_seven : opposite 7 = -7 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_opposite_of_seven_l2125_212505


namespace NUMINAMATH_CALUDE_chip_price_reduction_l2125_212516

/-- Represents the price reduction process for a chip -/
theorem chip_price_reduction (initial_price final_price : ℝ) 
  (h1 : initial_price = 256) 
  (h2 : final_price = 196) 
  (x : ℝ) -- x represents the percentage of each price reduction
  (h3 : 0 ≤ x ∧ x < 1) -- ensure x is a valid percentage
  : initial_price * (1 - x)^2 = final_price ↔ 
    initial_price * (1 - x)^2 = 196 ∧ initial_price = 256 :=
by sorry

end NUMINAMATH_CALUDE_chip_price_reduction_l2125_212516


namespace NUMINAMATH_CALUDE_divisibility_property_l2125_212568

theorem divisibility_property (a b c d : ℤ) (h : (a - c) ∣ (a * b + c * d)) :
  (a - c) ∣ (a * d + b * c) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l2125_212568


namespace NUMINAMATH_CALUDE_modulo_equivalence_unique_solution_l2125_212539

theorem modulo_equivalence_unique_solution :
  ∃! n : ℤ, 0 ≤ n ∧ n ≤ 8 ∧ n ≡ 123456 [ZMOD 9] ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_modulo_equivalence_unique_solution_l2125_212539


namespace NUMINAMATH_CALUDE_gcd_lcm_product_150_180_l2125_212566

theorem gcd_lcm_product_150_180 : Nat.gcd 150 180 * Nat.lcm 150 180 = 27000 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_150_180_l2125_212566


namespace NUMINAMATH_CALUDE_expression_simplification_l2125_212546

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 2 - 1) :
  (x - 2) / (x^2 - 1) / (1 - 1 / (x - 1)) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2125_212546


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l2125_212544

theorem fractional_equation_solution :
  ∀ x : ℝ, (4 / x = 2 / (x + 1)) ↔ (x = -2) :=
by sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l2125_212544


namespace NUMINAMATH_CALUDE_parallel_lines_k_values_l2125_212529

/-- Two lines are parallel if their slopes are equal -/
def are_parallel (a₁ b₁ a₂ b₂ : ℝ) : Prop :=
  a₁ * b₂ = a₂ * b₁

/-- Definition of line l₁ -/
def l₁ (k : ℝ) (x y : ℝ) : Prop :=
  (k - 3) * x + (4 - k) * y + 1 = 0

/-- Definition of line l₂ -/
def l₂ (k : ℝ) (x y : ℝ) : Prop :=
  2 * (k - 3) * x - 2 * y + 3 = 0

theorem parallel_lines_k_values :
  ∀ k : ℝ, (∀ x y : ℝ, are_parallel (k - 3) (4 - k) (2 * (k - 3)) (-2)) →
  k = 3 ∨ k = 5 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_k_values_l2125_212529


namespace NUMINAMATH_CALUDE_black_balls_count_l2125_212508

theorem black_balls_count (total : ℕ) (red : ℕ) (white_prob : ℚ) 
  (h_total : total = 100)
  (h_red : red = 30)
  (h_white_prob : white_prob = 47/100)
  (h_sum : red + (white_prob * total).floor + (total - red - (white_prob * total).floor) = total) :
  total - red - (white_prob * total).floor = 23 := by
  sorry

end NUMINAMATH_CALUDE_black_balls_count_l2125_212508


namespace NUMINAMATH_CALUDE_kaleb_fair_expense_l2125_212580

/-- Calculates the total cost of rides given the number of tickets used and the cost per ticket -/
def total_cost (tickets_used : ℕ) (cost_per_ticket : ℕ) : ℕ :=
  tickets_used * cost_per_ticket

theorem kaleb_fair_expense :
  let initial_tickets : ℕ := 6
  let ferris_wheel_cost : ℕ := 2
  let bumper_cars_cost : ℕ := 1
  let roller_coaster_cost : ℕ := 2
  let ticket_price : ℕ := 9
  let total_tickets_used : ℕ := ferris_wheel_cost + bumper_cars_cost + roller_coaster_cost
  total_cost total_tickets_used ticket_price = 45 := by
  sorry

#eval total_cost 5 9

end NUMINAMATH_CALUDE_kaleb_fair_expense_l2125_212580
