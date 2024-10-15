import Mathlib

namespace NUMINAMATH_CALUDE_max_value_abcd_l3449_344923

def S : Finset ℕ := {1, 3, 5, 7}

theorem max_value_abcd (a b c d : ℕ) (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) (hd : d ∈ S) 
  (hdistinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) : 
  (∀ (w x y z : ℕ), w ∈ S → x ∈ S → y ∈ S → z ∈ S → 
    w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z → 
    w * x + x * y + y * z + z * w ≤ a * b + b * c + c * d + d * a) →
  a * b + b * c + c * d + d * a = 64 :=
sorry

end NUMINAMATH_CALUDE_max_value_abcd_l3449_344923


namespace NUMINAMATH_CALUDE_perpendicular_planes_parallel_l3449_344992

-- Define the necessary structures
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

structure Line3D where
  point : Point3D
  direction : Point3D

structure Plane3D where
  point : Point3D
  normal : Point3D

-- Define perpendicularity between a line and a plane
def perpendicular (l : Line3D) (p : Plane3D) : Prop :=
  l.direction.x * p.normal.x + l.direction.y * p.normal.y + l.direction.z * p.normal.z = 0

-- Define parallelism between two planes
def parallel (p1 p2 : Plane3D) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ 
    p1.normal.x = k * p2.normal.x ∧
    p1.normal.y = k * p2.normal.y ∧
    p1.normal.z = k * p2.normal.z

-- State the theorem
theorem perpendicular_planes_parallel (l : Line3D) (p1 p2 : Plane3D) :
  perpendicular l p1 → perpendicular l p2 → parallel p1 p2 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_planes_parallel_l3449_344992


namespace NUMINAMATH_CALUDE_jake_and_kendra_weight_l3449_344909

/-- Jake's current weight in pounds -/
def jake_weight : ℕ := 198

/-- The weight Jake would lose in pounds -/
def weight_loss : ℕ := 8

/-- Kendra's weight in pounds -/
def kendra_weight : ℕ := (jake_weight - weight_loss) / 2

/-- The combined weight of Jake and Kendra in pounds -/
def combined_weight : ℕ := jake_weight + kendra_weight

/-- Theorem stating the combined weight of Jake and Kendra -/
theorem jake_and_kendra_weight : combined_weight = 293 := by
  sorry

end NUMINAMATH_CALUDE_jake_and_kendra_weight_l3449_344909


namespace NUMINAMATH_CALUDE_cos_300_degrees_l3449_344996

theorem cos_300_degrees : Real.cos (300 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_300_degrees_l3449_344996


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3449_344995

theorem inequality_equivalence (x : ℝ) (h : x ≠ 2) :
  (x - 1) / (x - 2) ≤ 0 ↔ (x^3 - x^2 + x - 1) / (x - 2) ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3449_344995


namespace NUMINAMATH_CALUDE_machine_purchase_price_l3449_344920

/-- Proves that given the specified conditions, the original purchase price of the machine was Rs 9000 -/
theorem machine_purchase_price 
  (repair_cost : ℕ) 
  (transport_cost : ℕ) 
  (profit_percentage : ℚ) 
  (selling_price : ℕ) 
  (h1 : repair_cost = 5000)
  (h2 : transport_cost = 1000)
  (h3 : profit_percentage = 50 / 100)
  (h4 : selling_price = 22500) :
  ∃ (purchase_price : ℕ), 
    selling_price = (1 + profit_percentage) * (purchase_price + repair_cost + transport_cost) ∧
    purchase_price = 9000 := by
  sorry


end NUMINAMATH_CALUDE_machine_purchase_price_l3449_344920


namespace NUMINAMATH_CALUDE_base_conversion_equality_l3449_344975

def base_five_to_decimal (n : ℕ) : ℕ := 
  (n / 100) * 25 + ((n / 10) % 10) * 5 + (n % 10)

def base_b_to_decimal (n : ℕ) (b : ℕ) : ℕ := 
  (n / 100) * b^2 + ((n / 10) % 10) * b + (n % 10)

theorem base_conversion_equality :
  ∃ (b : ℕ), b > 0 ∧ base_five_to_decimal 132 = base_b_to_decimal 210 b ∧ b = 4 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_equality_l3449_344975


namespace NUMINAMATH_CALUDE_total_candy_count_l3449_344906

theorem total_candy_count (brother_candy : ℕ) (wendy_boxes : ℕ) (pieces_per_box : ℕ) : 
  brother_candy = 6 → wendy_boxes = 2 → pieces_per_box = 3 →
  brother_candy + wendy_boxes * pieces_per_box = 12 :=
by sorry

end NUMINAMATH_CALUDE_total_candy_count_l3449_344906


namespace NUMINAMATH_CALUDE_tangent_line_x_ln_x_l3449_344966

/-- The equation of the tangent line to y = x ln x at (1, 0) is x - y - 1 = 0 -/
theorem tangent_line_x_ln_x (x y : ℝ) : 
  (∀ t, t > 0 → y = t * Real.log t) →  -- Definition of the curve
  (x = 1 ∧ y = 0) →                    -- Point of tangency
  (x - y - 1 = 0) :=                   -- Equation of the tangent line
by sorry

end NUMINAMATH_CALUDE_tangent_line_x_ln_x_l3449_344966


namespace NUMINAMATH_CALUDE_student_walking_distance_l3449_344910

theorem student_walking_distance 
  (total_distance : ℝ)
  (walking_speed : ℝ)
  (bus_speed_with_students : ℝ)
  (empty_bus_speed : ℝ)
  (h1 : total_distance = 1)
  (h2 : walking_speed = 4)
  (h3 : bus_speed_with_students = 40)
  (h4 : empty_bus_speed = 60)
  (h5 : ∀ x : ℝ, 0 < x ∧ x < 1 → 
    x / walking_speed = 
    (1 - x) / bus_speed_with_students + 
    (1 - 2*x) / empty_bus_speed) :
  ∃ x : ℝ, x = 5 / 37 ∧ 
    x / walking_speed = 
    (1 - x) / bus_speed_with_students + 
    (1 - 2*x) / empty_bus_speed :=
sorry

end NUMINAMATH_CALUDE_student_walking_distance_l3449_344910


namespace NUMINAMATH_CALUDE_train_speed_calculation_l3449_344962

/-- Proves that a train with given length, crossing a bridge of given length in a given time, has a specific speed in km/h -/
theorem train_speed_calculation (train_length bridge_length : ℝ) (crossing_time : ℝ) :
  train_length = 120 →
  bridge_length = 200 →
  crossing_time = 31.99744020478362 →
  (((train_length + bridge_length) / crossing_time) * 3.6) = 36 := by
  sorry

#check train_speed_calculation

end NUMINAMATH_CALUDE_train_speed_calculation_l3449_344962


namespace NUMINAMATH_CALUDE_no_squares_in_range_l3449_344964

theorem no_squares_in_range : ¬ ∃ (x y a b : ℕ),
  988 ≤ x ∧ x < y ∧ y ≤ 1991 ∧
  x * y + x = a^2 ∧
  x * y + y = b^2 :=
sorry

end NUMINAMATH_CALUDE_no_squares_in_range_l3449_344964


namespace NUMINAMATH_CALUDE_solve_system_l3449_344927

theorem solve_system (x y : ℚ) (h1 : 3 * x - y = 9) (h2 : x + 4 * y = 11) : x = 47 / 13 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l3449_344927


namespace NUMINAMATH_CALUDE_complementary_angles_difference_l3449_344952

theorem complementary_angles_difference (a b : ℝ) : 
  a + b = 90 → -- angles are complementary
  a = 5 * b → -- ratio of angles is 5:1
  |a - b| = 60 := by sorry

end NUMINAMATH_CALUDE_complementary_angles_difference_l3449_344952


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l3449_344953

theorem fraction_sum_equality (n : ℕ) (hn : n > 1) :
  ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ x ≤ y ∧ (1 : ℚ) / n = 1 / x - 1 / (y + 1) := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l3449_344953


namespace NUMINAMATH_CALUDE_x_over_y_value_l3449_344960

theorem x_over_y_value (x y : ℝ) (h1 : x ≠ y) 
  (h2 : x / y + (x + 5 * y) / (y + 5 * x) = 2) : 
  x / y = 0.6 := by
sorry

end NUMINAMATH_CALUDE_x_over_y_value_l3449_344960


namespace NUMINAMATH_CALUDE_first_hour_distance_l3449_344955

/-- A structure representing a family road trip -/
structure RoadTrip where
  firstHourDistance : ℝ
  remainingDistance : ℝ
  totalTime : ℝ
  speed : ℝ

/-- Theorem: Given the conditions of the road trip, the distance traveled in the first hour is 100 miles -/
theorem first_hour_distance (trip : RoadTrip) 
  (h1 : trip.remainingDistance = 300)
  (h2 : trip.totalTime = 4)
  (h3 : trip.speed * 1 = trip.firstHourDistance)
  (h4 : trip.speed * 3 = trip.remainingDistance) : 
  trip.firstHourDistance = 100 := by
  sorry

#check first_hour_distance

end NUMINAMATH_CALUDE_first_hour_distance_l3449_344955


namespace NUMINAMATH_CALUDE_symmetric_points_existence_l3449_344977

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then a * Real.exp (-x) else Real.log (x / a)

theorem symmetric_points_existence (a : ℝ) (h : a > 0) :
  (∃ x₀ : ℝ, x₀ > 1 ∧ f a (-x₀) = f a x₀) ↔ 0 < a ∧ a < Real.exp (-1) :=
sorry

end NUMINAMATH_CALUDE_symmetric_points_existence_l3449_344977


namespace NUMINAMATH_CALUDE_not_all_pairs_perfect_square_l3449_344963

theorem not_all_pairs_perfect_square (d : ℕ) (h1 : d > 0) (h2 : d ≠ 2) (h3 : d ≠ 5) (h4 : d ≠ 13) :
  ∃ (a b : ℕ), a ∈ ({2, 5, 13, d} : Set ℕ) ∧ b ∈ ({2, 5, 13, d} : Set ℕ) ∧ a ≠ b ∧ ¬∃ (k : ℕ), a * b - 1 = k^2 :=
by sorry

end NUMINAMATH_CALUDE_not_all_pairs_perfect_square_l3449_344963


namespace NUMINAMATH_CALUDE_root_square_minus_three_x_minus_one_l3449_344990

theorem root_square_minus_three_x_minus_one (m : ℝ) : 
  m^2 - 3*m - 1 = 0 → 2*m^2 - 6*m = 2 := by
  sorry

end NUMINAMATH_CALUDE_root_square_minus_three_x_minus_one_l3449_344990


namespace NUMINAMATH_CALUDE_square_difference_l3449_344944

theorem square_difference (a b : ℝ) 
  (h1 : 3 * a + 3 * b = 18) 
  (h2 : a - b = 4) : 
  a^2 - b^2 = 24 := by
sorry

end NUMINAMATH_CALUDE_square_difference_l3449_344944


namespace NUMINAMATH_CALUDE_emma_walk_distance_l3449_344999

theorem emma_walk_distance
  (total_time : ℝ)
  (bike_speed : ℝ)
  (walk_speed : ℝ)
  (bike_fraction : ℝ)
  (walk_fraction : ℝ)
  (h_total_time : total_time = 1)
  (h_bike_speed : bike_speed = 20)
  (h_walk_speed : walk_speed = 6)
  (h_bike_fraction : bike_fraction = 1/3)
  (h_walk_fraction : walk_fraction = 2/3)
  (h_fractions : bike_fraction + walk_fraction = 1) :
  let total_distance := (bike_speed * bike_fraction + walk_speed * walk_fraction) * total_time
  let walk_distance := total_distance * walk_fraction
  ∃ (ε : ℝ), abs (walk_distance - 5.2) < ε ∧ ε < 0.1 :=
by sorry

end NUMINAMATH_CALUDE_emma_walk_distance_l3449_344999


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l3449_344979

theorem arithmetic_calculation : (4 + 4 + 6) / 3 - 2 / 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l3449_344979


namespace NUMINAMATH_CALUDE_trigonometric_equation_l3449_344991

theorem trigonometric_equation (α β : Real) 
  (h : (Real.cos α)^3 / Real.cos β + (Real.sin α)^3 / Real.sin β = 2) :
  (Real.sin β)^3 / Real.sin α + (Real.cos β)^3 / Real.cos α = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equation_l3449_344991


namespace NUMINAMATH_CALUDE_fib_last_digit_periodic_l3449_344973

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Period of Fibonacci sequence modulo 10 -/
def fibPeriod : ℕ := 60

/-- Theorem: The last digit of Fibonacci numbers repeats with period 60 -/
theorem fib_last_digit_periodic (n : ℕ) : fib n % 10 = fib (n + fibPeriod) % 10 := by
  sorry

end NUMINAMATH_CALUDE_fib_last_digit_periodic_l3449_344973


namespace NUMINAMATH_CALUDE_log_sqrt8_512sqrt8_equals_7_l3449_344907

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_sqrt8_512sqrt8_equals_7 :
  log (Real.sqrt 8) (512 * Real.sqrt 8) = 7 := by sorry

end NUMINAMATH_CALUDE_log_sqrt8_512sqrt8_equals_7_l3449_344907


namespace NUMINAMATH_CALUDE_cos_pi_eighth_times_cos_five_pi_eighth_l3449_344919

theorem cos_pi_eighth_times_cos_five_pi_eighth :
  Real.cos (π / 8) * Real.cos (5 * π / 8) = -Real.sqrt 2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_eighth_times_cos_five_pi_eighth_l3449_344919


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l3449_344947

theorem quadratic_coefficient (b : ℝ) (m : ℝ) : 
  b < 0 → 
  (∀ x, x^2 + b*x + 1/4 = (x + m)^2 + 1/8) → 
  b = -Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l3449_344947


namespace NUMINAMATH_CALUDE_derivative_at_one_l3449_344942

-- Define the function
def f (x : ℝ) : ℝ := (2*x + 1)^2

-- State the theorem
theorem derivative_at_one :
  deriv f 1 = 12 := by sorry

end NUMINAMATH_CALUDE_derivative_at_one_l3449_344942


namespace NUMINAMATH_CALUDE_forty_third_digit_of_one_thirteenth_l3449_344957

/-- The decimal representation of 1/13 as a sequence of digits after the decimal point -/
def decimalRep : ℕ → Fin 10
  | n => sorry

/-- The length of the repeating sequence in the decimal representation of 1/13 -/
def repeatLength : ℕ := 6

/-- The 43rd digit after the decimal point in the decimal representation of 1/13 is 0 -/
theorem forty_third_digit_of_one_thirteenth : decimalRep 42 = 0 := by sorry

end NUMINAMATH_CALUDE_forty_third_digit_of_one_thirteenth_l3449_344957


namespace NUMINAMATH_CALUDE_prob_A_not_in_A_is_two_thirds_l3449_344941

-- Define the number of volunteers and communities
def num_volunteers : ℕ := 4
def num_communities : ℕ := 3

-- Define a type for volunteers and communities
inductive Volunteer : Type
| A | B | C | D

inductive Community : Type
| A | B | C

-- Define an assignment as a function from Volunteer to Community
def Assignment := Volunteer → Community

-- Define a valid assignment
def valid_assignment (a : Assignment) : Prop :=
  ∀ c : Community, ∃ v : Volunteer, a v = c

-- Define the probability that volunteer A is not in community A
def prob_A_not_in_A (total_assignments : ℕ) (valid_assignments : ℕ) : ℚ :=
  (valid_assignments - (total_assignments / num_communities)) / valid_assignments

-- State the theorem
theorem prob_A_not_in_A_is_two_thirds :
  ∃ (total_assignments valid_assignments : ℕ),
    total_assignments > 0 ∧
    valid_assignments > 0 ∧
    valid_assignments ≤ total_assignments ∧
    prob_A_not_in_A total_assignments valid_assignments = 2/3 :=
sorry

end NUMINAMATH_CALUDE_prob_A_not_in_A_is_two_thirds_l3449_344941


namespace NUMINAMATH_CALUDE_unique_number_count_l3449_344950

/-- The number of unique 5-digit numbers that can be formed by rearranging
    the digits 3, 7, 3, 2, 2, 0, where the number doesn't start with 0. -/
def unique_numbers : ℕ := 24

/-- The set of digits available for forming the numbers. -/
def digits : Finset ℕ := {3, 7, 2, 0}

/-- The total number of digits to be used. -/
def total_digits : ℕ := 5

/-- The number of positions where 0 can be placed (not in the first position). -/
def zero_positions : ℕ := 4

/-- The number of times 3 appears in the original number. -/
def count_three : ℕ := 2

/-- The number of times 2 appears in the original number. -/
def count_two : ℕ := 2

theorem unique_number_count :
  unique_numbers = (zero_positions * Nat.factorial (total_digits - 1)) /
                   (Nat.factorial count_three * Nat.factorial count_two) :=
sorry

end NUMINAMATH_CALUDE_unique_number_count_l3449_344950


namespace NUMINAMATH_CALUDE_inscribed_sphere_radius_tetrahedron_l3449_344989

/-- Given a tetrahedron with volume V, face areas S₁, S₂, S₃, S₄, and an inscribed sphere of radius R,
    prove that R = 3V / (S₁ + S₂ + S₃ + S₄) -/
theorem inscribed_sphere_radius_tetrahedron (V : ℝ) (S₁ S₂ S₃ S₄ : ℝ) (R : ℝ) 
    (h_volume : V > 0)
    (h_areas : S₁ > 0 ∧ S₂ > 0 ∧ S₃ > 0 ∧ S₄ > 0)
    (h_inscribed : R > 0) :
  R = 3 * V / (S₁ + S₂ + S₃ + S₄) :=
sorry

end NUMINAMATH_CALUDE_inscribed_sphere_radius_tetrahedron_l3449_344989


namespace NUMINAMATH_CALUDE_quartic_polynomial_sum_l3449_344985

/-- A quartic polynomial with specific properties -/
structure QuarticPolynomial (k : ℝ) where
  P : ℝ → ℝ
  is_quartic : ∃ (a b c d : ℝ), ∀ x, P x = a * x^4 + b * x^3 + c * x^2 + d * x + k
  at_zero : P 0 = k
  at_one : P 1 = 3 * k
  at_neg_one : P (-1) = 4 * k

/-- The sum of the polynomial evaluated at 2 and -2 equals 82k -/
theorem quartic_polynomial_sum (k : ℝ) (p : QuarticPolynomial k) :
  p.P 2 + p.P (-2) = 82 * k := by
  sorry

end NUMINAMATH_CALUDE_quartic_polynomial_sum_l3449_344985


namespace NUMINAMATH_CALUDE_prime_square_mod_twelve_l3449_344965

theorem prime_square_mod_twelve (p : ℕ) (hp : Nat.Prime p) (hp_gt_3 : p > 3) :
  p^2 % 12 = 1 := by
  sorry

end NUMINAMATH_CALUDE_prime_square_mod_twelve_l3449_344965


namespace NUMINAMATH_CALUDE_units_digit_7_pow_2023_l3449_344969

def units_digit (n : ℕ) : ℕ := n % 10

def power_7_units_digit_pattern : List ℕ := [7, 9, 3, 1]

theorem units_digit_7_pow_2023 :
  units_digit (7^2023) = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_units_digit_7_pow_2023_l3449_344969


namespace NUMINAMATH_CALUDE_jim_shopping_cost_l3449_344986

theorem jim_shopping_cost (lamp_cost : ℕ) (bulb_cost : ℕ) (num_lamps : ℕ) (num_bulbs : ℕ) 
  (h1 : lamp_cost = 7)
  (h2 : bulb_cost = lamp_cost - 4)
  (h3 : num_lamps = 2)
  (h4 : num_bulbs = 6) :
  num_lamps * lamp_cost + num_bulbs * bulb_cost = 32 := by
sorry

end NUMINAMATH_CALUDE_jim_shopping_cost_l3449_344986


namespace NUMINAMATH_CALUDE_max_det_bound_l3449_344935

theorem max_det_bound (M : Matrix (Fin 17) (Fin 17) ℤ) 
  (h : ∀ i j, M i j = 1 ∨ M i j = -1) :
  |M.det| ≤ 327680 * 2^16 := by
  sorry

end NUMINAMATH_CALUDE_max_det_bound_l3449_344935


namespace NUMINAMATH_CALUDE_wire_cutting_l3449_344961

theorem wire_cutting (total_length : ℝ) (difference : ℝ) (shorter_part : ℝ) : 
  total_length = 180 ∧ difference = 32 → 
  shorter_part + (shorter_part + difference) = total_length →
  shorter_part = 74 := by sorry

end NUMINAMATH_CALUDE_wire_cutting_l3449_344961


namespace NUMINAMATH_CALUDE_quadratic_roots_reciprocal_sum_l3449_344970

theorem quadratic_roots_reciprocal_sum (x₁ x₂ : ℝ) :
  x₁^2 - 4*x₁ - 2 = 0 →
  x₂^2 - 4*x₂ - 2 = 0 →
  x₁ ≠ x₂ →
  (1/x₁) + (1/x₂) = -2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_reciprocal_sum_l3449_344970


namespace NUMINAMATH_CALUDE_rowing_speed_in_still_water_l3449_344981

/-- The speed of a man rowing a boat in still water, given his downstream performance and current speed -/
theorem rowing_speed_in_still_water 
  (distance : Real) 
  (time : Real) 
  (current_speed : Real) : 
  (distance / 1000) / (time / 3600) - current_speed = 22 :=
by
  -- Assuming:
  -- distance = 80 (meters)
  -- time = 11.519078473722104 (seconds)
  -- current_speed = 3 (km/h)
  sorry

#check rowing_speed_in_still_water


end NUMINAMATH_CALUDE_rowing_speed_in_still_water_l3449_344981


namespace NUMINAMATH_CALUDE_trig_identity_l3449_344914

theorem trig_identity (α : ℝ) :
  (Real.sin (6 * α) + Real.sin (7 * α) + Real.sin (8 * α) + Real.sin (9 * α)) /
  (Real.cos (6 * α) + Real.cos (7 * α) + Real.cos (8 * α) + Real.cos (9 * α)) =
  Real.tan (15 * α / 2) := by
sorry

end NUMINAMATH_CALUDE_trig_identity_l3449_344914


namespace NUMINAMATH_CALUDE_original_average_calc_l3449_344974

/-- Given a set of 10 numbers, if increasing one number by 6 changes the average to 6.8,
    then the original average was 6.2 -/
theorem original_average_calc (S : Finset ℝ) (original_sum : ℝ) :
  Finset.card S = 10 →
  (original_sum + 6) / 10 = 6.8 →
  original_sum / 10 = 6.2 :=
by sorry

end NUMINAMATH_CALUDE_original_average_calc_l3449_344974


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l3449_344915

theorem simplify_and_evaluate (m : ℝ) (h : m = -2) :
  m / (m^2 - 9) / (1 + 3 / (m - 3)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l3449_344915


namespace NUMINAMATH_CALUDE_det_E_l3449_344988

/-- A 2×2 matrix representing a dilation centered at the origin with scale factor 9 -/
def E : Matrix (Fin 2) (Fin 2) ℝ := !![9, 0; 0, 9]

/-- The determinant of E is 81 -/
theorem det_E : Matrix.det E = 81 := by sorry

end NUMINAMATH_CALUDE_det_E_l3449_344988


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_for_all_real_solution_l3449_344930

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |a*x - 1| + |a*x - 3*a|

-- Part 1
theorem solution_set_when_a_is_one :
  {x : ℝ | |x - 1| + |x - 3| ≥ 5} = {x : ℝ | x ≥ 9/2 ∨ x ≤ -1/2} := by sorry

-- Part 2
theorem range_of_a_for_all_real_solution :
  {a : ℝ | a > 0 ∧ ∀ x, f a x ≥ 5} = {a : ℝ | a ≥ 2} := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_for_all_real_solution_l3449_344930


namespace NUMINAMATH_CALUDE_vector_opposite_directions_x_value_l3449_344939

/-- Two vectors are in opposite directions if one is a negative scalar multiple of the other -/
def opposite_directions (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k < 0 ∧ a = (-k • b)

theorem vector_opposite_directions_x_value :
  ∀ x : ℝ,
  let a : ℝ × ℝ := (1, -x)
  let b : ℝ × ℝ := (x, -6)
  opposite_directions a b → x = -Real.sqrt 6 :=
by
  sorry

end NUMINAMATH_CALUDE_vector_opposite_directions_x_value_l3449_344939


namespace NUMINAMATH_CALUDE_two_valid_antonyms_exist_l3449_344922

/-- A word is represented as a string of characters. -/
def Word := String

/-- The maximum allowed length for an antonym. -/
def MaxLength : Nat := 10

/-- Predicate to check if a word is an antonym of "seldom". -/
def IsAntonymOfSeldom (w : Word) : Prop := sorry

/-- Predicate to check if two words have distinct meanings. -/
def HasDistinctMeaning (w1 w2 : Word) : Prop := sorry

/-- Theorem stating the existence of two valid antonyms for "seldom". -/
theorem two_valid_antonyms_exist : 
  ∃ (w1 w2 : Word), 
    IsAntonymOfSeldom w1 ∧ 
    IsAntonymOfSeldom w2 ∧ 
    w1.length ≤ MaxLength ∧ 
    w2.length ≤ MaxLength ∧ 
    w1.front = 'o' ∧ 
    w2.front = 'u' ∧ 
    HasDistinctMeaning w1 w2 :=
  sorry

end NUMINAMATH_CALUDE_two_valid_antonyms_exist_l3449_344922


namespace NUMINAMATH_CALUDE_hotel_fee_proof_l3449_344925

/-- The flat fee for the first night in a hotel -/
def flat_fee : ℝ := 87.5

/-- The nightly fee for each subsequent night -/
def nightly_fee : ℝ := 52.5

/-- Alice's total cost for a 4-night stay -/
def alice_cost : ℝ := 245

/-- Bob's total cost for a 6-night stay -/
def bob_cost : ℝ := 350

/-- The number of nights in Alice's stay -/
def alice_nights : ℕ := 4

/-- The number of nights in Bob's stay -/
def bob_nights : ℕ := 6

theorem hotel_fee_proof :
  (flat_fee + (alice_nights - 1 : ℝ) * nightly_fee = alice_cost) ∧
  (flat_fee + (bob_nights - 1 : ℝ) * nightly_fee = bob_cost) :=
by sorry

#check hotel_fee_proof

end NUMINAMATH_CALUDE_hotel_fee_proof_l3449_344925


namespace NUMINAMATH_CALUDE_ratio_of_numbers_l3449_344954

theorem ratio_of_numbers (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (sum_diff : a + b = 7 * (a - b)) (product : a * b = 50) :
  max a b / min a b = 4 / 3 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_numbers_l3449_344954


namespace NUMINAMATH_CALUDE_smallest_separating_degree_l3449_344945

/-- A point on the coordinate plane with a color -/
structure ColoredPoint where
  x : ℝ
  y : ℝ
  color : Bool  -- True for red, False for blue

/-- A set of N points is permissible if their x-coordinates are distinct -/
def isPermissible (points : Finset ColoredPoint) : Prop :=
  ∀ p q : ColoredPoint, p ∈ points → q ∈ points → p ≠ q → p.x ≠ q.x

/-- A polynomial P separates a set of points if no red points are above
    and no blue points below its graph, or vice versa -/
def separates (P : ℝ → ℝ) (points : Finset ColoredPoint) : Prop :=
  (∀ p ∈ points, p.color = true → P p.x ≥ p.y) ∧
  (∀ p ∈ points, p.color = false → P p.x ≤ p.y) ∨
  (∀ p ∈ points, p.color = true → P p.x ≤ p.y) ∧
  (∀ p ∈ points, p.color = false → P p.x ≥ p.y)

/-- The main theorem: For any N ≥ 3, the smallest degree k of a polynomial
    that can separate any permissible set of N points is N-2 -/
theorem smallest_separating_degree (N : ℕ) (h : N ≥ 3) :
  ∃ k : ℕ, (∀ points : Finset ColoredPoint, points.card = N → isPermissible points →
    ∃ P : ℝ → ℝ, (∃ coeffs : Finset ℝ, coeffs.card ≤ k + 1 ∧
      P = fun x ↦ (coeffs.toList.enum.map (fun (i, a) ↦ a * x ^ i)).sum) ∧
    separates P points) ∧
  (∀ k' : ℕ, k' < k →
    ∃ points : Finset ColoredPoint, points.card = N ∧ isPermissible points ∧
    ∀ P : ℝ → ℝ, (∃ coeffs : Finset ℝ, coeffs.card ≤ k' + 1 ∧
      P = fun x ↦ (coeffs.toList.enum.map (fun (i, a) ↦ a * x ^ i)).sum) →
    ¬separates P points) ∧
  k = N - 2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_separating_degree_l3449_344945


namespace NUMINAMATH_CALUDE_consecutive_naturals_integer_quotient_l3449_344908

theorem consecutive_naturals_integer_quotient :
  ∃! (n : ℕ), (n + 1 : ℚ) / n = ⌊(n + 1 : ℚ) / n⌋ ∧ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_naturals_integer_quotient_l3449_344908


namespace NUMINAMATH_CALUDE_function_properties_l3449_344982

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem function_properties (f : ℝ → ℝ) :
  (∀ x, f (1 + x) = f (x - 1)) →
  (∀ x, f (1 - x) = -f (x - 1)) →
  (is_periodic f 2 ∧ is_odd f) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l3449_344982


namespace NUMINAMATH_CALUDE_right_triangle_area_l3449_344994

theorem right_triangle_area (a b c : ℝ) (h1 : a^2 = 64) (h2 : b^2 = 49) (h3 : c^2 = 113) 
  (h4 : a^2 + b^2 = c^2) : (1/2) * a * b = 28 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l3449_344994


namespace NUMINAMATH_CALUDE_point_on_angle_terminal_side_l3449_344940

theorem point_on_angle_terminal_side (y : ℝ) :
  let P : ℝ × ℝ := (-1, y)
  let θ : ℝ := 2 * Real.pi / 3
  (P.1 = -1) →   -- x-coordinate is -1
  (Real.tan θ = y / P.1) →  -- point is on terminal side of angle θ
  y = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_point_on_angle_terminal_side_l3449_344940


namespace NUMINAMATH_CALUDE_income_scientific_notation_l3449_344926

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  one_le_coeff_lt_ten : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

/-- Rounds a ScientificNotation to a specified number of significant figures -/
def roundToSignificantFigures (sn : ScientificNotation) (figures : ℕ) : ScientificNotation :=
  sorry

theorem income_scientific_notation :
  let income : ℝ := 31.534 * 1000000000
  let scientific_form := toScientificNotation income
  let rounded_form := roundToSignificantFigures scientific_form 2
  rounded_form = ScientificNotation.mk 3.2 10 sorry :=
sorry

end NUMINAMATH_CALUDE_income_scientific_notation_l3449_344926


namespace NUMINAMATH_CALUDE_schedule_count_eq_42_l3449_344987

/-- The number of employees -/
def n : ℕ := 6

/-- The number of days -/
def d : ℕ := 3

/-- The number of employees working each day -/
def k : ℕ := 2

/-- Calculates the number of ways to schedule employees with given restrictions -/
def schedule_count : ℕ :=
  Nat.choose n (2 * k) * Nat.choose (n - 2 * k) k - 
  2 * (Nat.choose (n - 1) k * Nat.choose (n - 1 - k) k) +
  Nat.choose (n - 2) k * Nat.choose (n - 2 - k) k

theorem schedule_count_eq_42 : schedule_count = 42 := by
  sorry

end NUMINAMATH_CALUDE_schedule_count_eq_42_l3449_344987


namespace NUMINAMATH_CALUDE_fraction_equality_l3449_344978

theorem fraction_equality : (1012^2 - 1003^2) / (1019^2 - 996^2) = 9 / 23 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3449_344978


namespace NUMINAMATH_CALUDE_arthurs_fitness_routine_l3449_344921

/-- The expected number of chocolate balls eaten during Arthur's fitness routine -/
def expected_chocolate_balls (n : ℕ) : ℝ :=
  if n < 2 then 0 else 1

/-- Arthur's fitness routine on Édes Street -/
theorem arthurs_fitness_routine (n : ℕ) (h : n ≥ 2) :
  expected_chocolate_balls n = 1 := by
  sorry

#check arthurs_fitness_routine

end NUMINAMATH_CALUDE_arthurs_fitness_routine_l3449_344921


namespace NUMINAMATH_CALUDE_gcd_840_1764_gcd_561_255_l3449_344971

-- Part 1: GCD of 840 and 1764
theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by sorry

-- Part 2: GCD of 561 and 255
theorem gcd_561_255 : Nat.gcd 561 255 = 51 := by sorry

end NUMINAMATH_CALUDE_gcd_840_1764_gcd_561_255_l3449_344971


namespace NUMINAMATH_CALUDE_archimedes_schools_l3449_344905

/-- The number of students in Euclid's contest -/
def euclid_participants : ℕ := 69

/-- The number of students per school team -/
def team_size : ℕ := 4

/-- The total number of participants in Archimedes' contest -/
def total_participants : ℕ := euclid_participants + 100

/-- Beth's rank in the contest -/
def beth_rank : ℕ := 45

/-- Carla's rank in the contest -/
def carla_rank : ℕ := 80

/-- Andrea's teammates with lower scores -/
def andreas_lower_teammates : ℕ := 2

theorem archimedes_schools :
  ∃ (num_schools : ℕ), 
    num_schools * team_size = total_participants ∧
    num_schools = 43 :=
sorry

end NUMINAMATH_CALUDE_archimedes_schools_l3449_344905


namespace NUMINAMATH_CALUDE_boxes_with_neither_markers_nor_crayons_l3449_344959

/-- The number of boxes containing neither markers nor crayons -/
def empty_boxes (total boxes_with_markers boxes_with_crayons boxes_with_both : ℕ) : ℕ :=
  total - (boxes_with_markers + boxes_with_crayons - boxes_with_both)

/-- Theorem: Given the conditions of the problem, there are 5 boxes with neither markers nor crayons -/
theorem boxes_with_neither_markers_nor_crayons :
  empty_boxes 15 9 5 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_boxes_with_neither_markers_nor_crayons_l3449_344959


namespace NUMINAMATH_CALUDE_frisbee_price_problem_l3449_344936

theorem frisbee_price_problem (total_frisbees : ℕ) (total_receipts : ℕ) (price_a : ℕ) (min_frisbees_b : ℕ) :
  total_frisbees = 60 →
  price_a = 3 →
  total_receipts = 204 →
  min_frisbees_b = 24 →
  ∃ (frisbees_a frisbees_b price_b : ℕ),
    frisbees_a + frisbees_b = total_frisbees ∧
    frisbees_b ≥ min_frisbees_b ∧
    price_a * frisbees_a + price_b * frisbees_b = total_receipts ∧
    price_b = 4 := by
  sorry

#check frisbee_price_problem

end NUMINAMATH_CALUDE_frisbee_price_problem_l3449_344936


namespace NUMINAMATH_CALUDE_total_wheels_calculation_l3449_344937

/-- The number of wheels on a four-wheeler -/
def wheels_per_four_wheeler : ℕ := 4

/-- The number of four-wheelers parked -/
def num_four_wheelers : ℕ := 13

/-- The total number of wheels for all four-wheelers -/
def total_wheels : ℕ := num_four_wheelers * wheels_per_four_wheeler

theorem total_wheels_calculation :
  total_wheels = 52 := by sorry

end NUMINAMATH_CALUDE_total_wheels_calculation_l3449_344937


namespace NUMINAMATH_CALUDE_toby_camera_roll_photos_l3449_344900

/-- The number of photos on Toby's camera roll initially -/
def initial_photos : ℕ := 79

/-- The number of photos Toby deleted initially -/
def deleted_initially : ℕ := 7

/-- The number of photos Toby added of his cat -/
def added_photos : ℕ := 15

/-- The number of photos Toby deleted after editing -/
def deleted_after_editing : ℕ := 3

/-- The final number of photos on Toby's camera roll -/
def final_photos : ℕ := 84

theorem toby_camera_roll_photos :
  initial_photos - deleted_initially + added_photos - deleted_after_editing = final_photos :=
by sorry

end NUMINAMATH_CALUDE_toby_camera_roll_photos_l3449_344900


namespace NUMINAMATH_CALUDE_shaded_area_of_divided_triangle_l3449_344932

theorem shaded_area_of_divided_triangle (leg_length : ℝ) (total_divisions : ℕ) (shaded_divisions : ℕ) : 
  leg_length = 10 → 
  total_divisions = 20 → 
  shaded_divisions = 12 → 
  (1/2 * leg_length * leg_length * (shaded_divisions / total_divisions : ℝ)) = 30 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_of_divided_triangle_l3449_344932


namespace NUMINAMATH_CALUDE_capital_city_free_after_year_l3449_344968

/-- Represents the state of a city (under spell or not) -/
inductive SpellState
| Free
| UnderSpell

/-- Represents the kingdom with 12 cities -/
structure Kingdom where
  cities : Fin 12 → SpellState

/-- Represents the magician's action on a city -/
def magicianAction (s : SpellState) : SpellState :=
  match s with
  | SpellState.Free => SpellState.UnderSpell
  | SpellState.UnderSpell => SpellState.Free

/-- Applies the magician's transformation to the kingdom -/
def monthlyTransformation (k : Kingdom) (startCity : Fin 12) : Kingdom :=
  { cities := λ i => 
      if i.val < startCity.val 
      then k.cities i 
      else magicianAction (k.cities i) }

/-- The state of the kingdom after 12 months -/
def afterTwelveMonths (k : Kingdom) : Kingdom :=
  (List.range 12).foldl (λ acc i => monthlyTransformation acc i) k

/-- The theorem to be proved -/
theorem capital_city_free_after_year (k : Kingdom) (capitalCity : Fin 12) :
  k.cities capitalCity = SpellState.Free →
  (afterTwelveMonths k).cities capitalCity = SpellState.Free :=
sorry

end NUMINAMATH_CALUDE_capital_city_free_after_year_l3449_344968


namespace NUMINAMATH_CALUDE_division_remainder_proof_l3449_344917

theorem division_remainder_proof (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) : 
  dividend = 140 → divisor = 15 → quotient = 9 → 
  dividend = divisor * quotient + remainder → remainder = 5 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_proof_l3449_344917


namespace NUMINAMATH_CALUDE_intersection_M_N_l3449_344934

def M : Set ℕ := {1, 2}
def N : Set ℕ := {x | ∃ a ∈ M, x = 2*a - 1}

theorem intersection_M_N : M ∩ N = {1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3449_344934


namespace NUMINAMATH_CALUDE_drinks_left_calculation_l3449_344998

-- Define the initial amounts of drinks
def initial_coke : ℝ := 35.5
def initial_cider : ℝ := 27.2

-- Define the amount of coke drunk
def coke_drunk : ℝ := 1.75

-- Theorem statement
theorem drinks_left_calculation :
  initial_coke + initial_cider - coke_drunk = 60.95 := by
  sorry

end NUMINAMATH_CALUDE_drinks_left_calculation_l3449_344998


namespace NUMINAMATH_CALUDE_plant_branches_problem_l3449_344918

theorem plant_branches_problem :
  ∃ (x : ℕ),
    (1 : ℕ) + x + x * x = 91 ∧
    (∀ y : ℕ, (1 : ℕ) + y + y * y = 91 → y ≤ x) ∧
    x = 9 :=
by sorry

end NUMINAMATH_CALUDE_plant_branches_problem_l3449_344918


namespace NUMINAMATH_CALUDE_even_digits_in_base_7_of_789_l3449_344983

def base_7_representation (n : ℕ) : List ℕ :=
  sorry

def count_even_digits (digits : List ℕ) : ℕ :=
  sorry

theorem even_digits_in_base_7_of_789 :
  count_even_digits (base_7_representation 789) = 3 := by
  sorry

end NUMINAMATH_CALUDE_even_digits_in_base_7_of_789_l3449_344983


namespace NUMINAMATH_CALUDE_oven_capacity_correct_l3449_344929

/-- The number of pies Marcus can fit in his oven at once. -/
def oven_capacity : ℕ := 5

/-- The number of batches Marcus bakes. -/
def batches : ℕ := 7

/-- The number of pies Marcus drops. -/
def dropped_pies : ℕ := 8

/-- The number of pies left after dropping. -/
def remaining_pies : ℕ := 27

/-- Theorem stating that the oven capacity is correct given the conditions. -/
theorem oven_capacity_correct : 
  batches * oven_capacity - dropped_pies = remaining_pies :=
by sorry

end NUMINAMATH_CALUDE_oven_capacity_correct_l3449_344929


namespace NUMINAMATH_CALUDE_fraction_equality_l3449_344903

theorem fraction_equality (P Q M N X : ℚ) 
  (hM : M = 0.4 * Q)
  (hQ : Q = 0.3 * P)
  (hN : N = 0.6 * P)
  (hX : X = 0.25 * M)
  (hP : P ≠ 0) : 
  X / N = 1 / 20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3449_344903


namespace NUMINAMATH_CALUDE_meaningful_fraction_range_l3449_344938

theorem meaningful_fraction_range (x : ℝ) :
  (∃ y : ℝ, y = 1 / (x - 3)) ↔ x ≠ 3 := by sorry

end NUMINAMATH_CALUDE_meaningful_fraction_range_l3449_344938


namespace NUMINAMATH_CALUDE_toothpick_pattern_l3449_344901

/-- 
Given a sequence where:
- The first term is 6
- Each successive term increases by 5 more than the previous increase
Prove that the 150th term is equal to 751
-/
theorem toothpick_pattern (n : ℕ) (a : ℕ → ℕ) : 
  a 1 = 6 ∧ 
  (∀ k, k ≥ 1 → a (k + 1) - a k = a k - a (k - 1) + 5) →
  a 150 = 751 :=
sorry

end NUMINAMATH_CALUDE_toothpick_pattern_l3449_344901


namespace NUMINAMATH_CALUDE_total_jelly_beans_l3449_344997

/-- The number of vanilla jelly beans -/
def vanilla_jb : ℕ := 120

/-- The number of grape jelly beans -/
def grape_jb : ℕ := 5 * vanilla_jb + 50

/-- The total number of jelly beans -/
def total_jb : ℕ := vanilla_jb + grape_jb

theorem total_jelly_beans : total_jb = 770 := by
  sorry

end NUMINAMATH_CALUDE_total_jelly_beans_l3449_344997


namespace NUMINAMATH_CALUDE_symmetric_line_wrt_y_axis_symmetric_line_example_l3449_344984

/-- Given a line with equation y = mx + b, the line symmetric to it
    with respect to the y-axis has equation y = -mx + b -/
theorem symmetric_line_wrt_y_axis (m b : ℝ) :
  let original_line := fun (x : ℝ) => m * x + b
  let symmetric_line := fun (x : ℝ) => -m * x + b
  ∀ x y : ℝ, symmetric_line x = y ↔ original_line (-x) = y := by sorry

/-- The equation of the line symmetric to y = 2x + 1 with respect to the y-axis is y = -2x + 1 -/
theorem symmetric_line_example :
  let original_line := fun (x : ℝ) => 2 * x + 1
  let symmetric_line := fun (x : ℝ) => -2 * x + 1
  ∀ x y : ℝ, symmetric_line x = y ↔ original_line (-x) = y := by sorry

end NUMINAMATH_CALUDE_symmetric_line_wrt_y_axis_symmetric_line_example_l3449_344984


namespace NUMINAMATH_CALUDE_language_interview_probability_l3449_344948

theorem language_interview_probability 
  (total_students : ℕ) 
  (french_students : ℕ) 
  (spanish_students : ℕ) 
  (both_languages : ℕ) 
  (h1 : total_students = 28)
  (h2 : french_students = 20)
  (h3 : spanish_students = 23)
  (h4 : both_languages = 17)
  (h5 : both_languages ≤ french_students)
  (h6 : both_languages ≤ spanish_students)
  (h7 : french_students ≤ total_students)
  (h8 : spanish_students ≤ total_students) :
  (1 : ℚ) - (Nat.choose (french_students - both_languages + (spanish_students - both_languages)) 2 : ℚ) / (Nat.choose total_students 2) = 20 / 21 :=
sorry

end NUMINAMATH_CALUDE_language_interview_probability_l3449_344948


namespace NUMINAMATH_CALUDE_second_diff_is_arithmetic_sequence_l3449_344949

-- Define the cube function
def cube (n : ℕ) : ℕ := n^3

-- Define the first difference of cubes
def first_diff (n : ℕ) : ℕ := cube (n + 1) - cube n

-- Define the second difference of cubes
def second_diff (n : ℕ) : ℕ := first_diff (n + 1) - first_diff n

-- Theorem stating that the second difference is 6n + 6
theorem second_diff_is_arithmetic_sequence (n : ℕ) : second_diff n = 6 * n + 6 := by
  sorry

end NUMINAMATH_CALUDE_second_diff_is_arithmetic_sequence_l3449_344949


namespace NUMINAMATH_CALUDE_new_solutions_introduced_l3449_344928

variables {α : Type*} [LinearOrder α]
variable (x : α)
variable (F₁ F₂ f : α → ℝ)

theorem new_solutions_introduced (h : F₁ x > F₂ x) :
  (f x < 0 ∧ F₁ x < F₂ x) ↔ (f x * F₁ x < f x * F₂ x ∧ ¬(F₁ x > F₂ x)) :=
by sorry

end NUMINAMATH_CALUDE_new_solutions_introduced_l3449_344928


namespace NUMINAMATH_CALUDE_cd_store_problem_l3449_344913

/-- Represents the total number of CDs in the store -/
def total_cds : ℕ := sorry

/-- Represents the price of expensive CDs -/
def expensive_price : ℕ := 10

/-- Represents the price of cheap CDs -/
def cheap_price : ℕ := 5

/-- Represents the proportion of expensive CDs -/
def expensive_proportion : ℚ := 2/5

/-- Represents the proportion of cheap CDs -/
def cheap_proportion : ℚ := 3/5

/-- Represents the proportion of expensive CDs bought by Prince -/
def expensive_bought_proportion : ℚ := 1/2

/-- Represents the total amount spent by Prince -/
def total_spent : ℕ := 1000

theorem cd_store_problem :
  (expensive_proportion * expensive_bought_proportion * (total_cds : ℚ) * expensive_price) +
  (cheap_proportion * (total_cds : ℚ) * cheap_price) = total_spent ∧
  total_cds = 200 := by sorry

end NUMINAMATH_CALUDE_cd_store_problem_l3449_344913


namespace NUMINAMATH_CALUDE_motion_rate_of_change_l3449_344943

-- Define the law of motion
def s (t : ℝ) : ℝ := 2 * t^2 + 1

-- Define the rate of change function
def rate_of_change (d : ℝ) : ℝ := 4 + 2 * d

-- Theorem statement
theorem motion_rate_of_change (d : ℝ) :
  let t₁ := 1
  let t₂ := 1 + d
  (s t₂ - s t₁) / (t₂ - t₁) = rate_of_change d :=
by sorry

end NUMINAMATH_CALUDE_motion_rate_of_change_l3449_344943


namespace NUMINAMATH_CALUDE_inequalities_for_negative_fractions_l3449_344976

theorem inequalities_for_negative_fractions (a b : ℝ) 
  (h1 : -1 < a) (h2 : a < b) (h3 : b < 0) : 
  (1 / a > 1 / b) ∧ 
  (a^2 + b^2 > 2*a*b) ∧ 
  (a + 1/a > b + 1/b) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_for_negative_fractions_l3449_344976


namespace NUMINAMATH_CALUDE_probability_not_greater_than_four_l3449_344951

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1 : ℝ)

theorem probability_not_greater_than_four 
  (a₁ : ℝ) 
  (d : ℝ) 
  (n : ℕ) 
  (h₁ : a₁ = 12) 
  (h₂ : d = -2) 
  (h₃ : n = 16) : 
  (Finset.filter (fun i => arithmetic_sequence a₁ d i ≤ 4) (Finset.range n)).card / n = 3/4 := by
sorry

end NUMINAMATH_CALUDE_probability_not_greater_than_four_l3449_344951


namespace NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l3449_344924

theorem repeating_decimal_to_fraction :
  ∃ (x : ℚ), (∀ (n : ℕ), x * 10^(2*n + 2) - (x * 10^(2*n + 2)).floor = 0.36) ∧ x = 4/11 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l3449_344924


namespace NUMINAMATH_CALUDE_marble_probability_difference_l3449_344933

/-- The number of red marbles in the box -/
def red_marbles : ℕ := 1500

/-- The number of black marbles in the box -/
def black_marbles : ℕ := 1500

/-- The total number of marbles in the box -/
def total_marbles : ℕ := red_marbles + black_marbles

/-- The probability of drawing two marbles of the same color -/
def P_s : ℚ := (Nat.choose red_marbles 2 + Nat.choose black_marbles 2) / Nat.choose total_marbles 2

/-- The probability of drawing two marbles of different colors -/
def P_d : ℚ := (red_marbles * black_marbles) / Nat.choose total_marbles 2

/-- Theorem stating the absolute difference between P_s and P_d -/
theorem marble_probability_difference : |P_s - P_d| = 15 / 44985 := by sorry

end NUMINAMATH_CALUDE_marble_probability_difference_l3449_344933


namespace NUMINAMATH_CALUDE_symmetric_second_quadrant_condition_l3449_344904

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Definition of symmetry about the origin -/
def symmetricAboutOrigin (p : Point2D) : Prop :=
  ∃ q : Point2D, q.x = -p.x ∧ q.y = -p.y

/-- Definition of a point being in the second quadrant -/
def inSecondQuadrant (p : Point2D) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Theorem stating the condition for m -/
theorem symmetric_second_quadrant_condition (m : ℝ) :
  symmetricAboutOrigin ⟨-m, m-3⟩ ∧ inSecondQuadrant ⟨m, 3-m⟩ → m < 0 :=
sorry

end NUMINAMATH_CALUDE_symmetric_second_quadrant_condition_l3449_344904


namespace NUMINAMATH_CALUDE_matthews_friends_l3449_344946

theorem matthews_friends (total_crackers : ℕ) (crackers_per_friend : ℕ) 
  (h1 : total_crackers = 36) (h2 : crackers_per_friend = 6) : 
  total_crackers / crackers_per_friend = 6 := by
  sorry

end NUMINAMATH_CALUDE_matthews_friends_l3449_344946


namespace NUMINAMATH_CALUDE_prime_ap_difference_greater_than_30000_l3449_344993

/-- An arithmetic progression of prime numbers -/
structure PrimeArithmeticProgression where
  terms : Fin 15 → ℕ
  is_prime : ∀ i, Nat.Prime (terms i)
  is_increasing : ∀ i j, i < j → terms i < terms j
  is_arithmetic : ∀ i j k, terms j - terms i = terms k - terms j ↔ j - i = k - j

/-- The common difference of an arithmetic progression -/
def common_difference (ap : PrimeArithmeticProgression) : ℕ :=
  ap.terms 1 - ap.terms 0

/-- Theorem: The common difference of an arithmetic progression of 15 primes is greater than 30000 -/
theorem prime_ap_difference_greater_than_30000 (ap : PrimeArithmeticProgression) :
  common_difference ap > 30000 := by
  sorry

end NUMINAMATH_CALUDE_prime_ap_difference_greater_than_30000_l3449_344993


namespace NUMINAMATH_CALUDE_elective_subjects_theorem_l3449_344902

def subjects := 6
def chosen := 3

def choose (n : ℕ) (r : ℕ) : ℕ := Nat.choose n r

theorem elective_subjects_theorem :
  -- Statement A
  (choose 5 3 = choose 5 2) ∧
  -- Statement C
  (choose subjects chosen - choose 4 1 = choose subjects chosen - choose (subjects - 2) (chosen - 2)) :=
sorry

end NUMINAMATH_CALUDE_elective_subjects_theorem_l3449_344902


namespace NUMINAMATH_CALUDE_frog_jumps_equivalence_l3449_344956

/-- Represents a frog's position on an integer line -/
def FrogPosition := ℤ

/-- Represents a configuration of frogs on the line -/
def FrogConfiguration := List FrogPosition

/-- Represents a direction of movement -/
inductive Direction
| Left : Direction
| Right : Direction

/-- Represents a sequence of n moves -/
def MoveSequence (n : ℕ) := Vector Direction n

/-- Predicate to check if a configuration has distinct positions -/
def HasDistinctPositions (config : FrogConfiguration) : Prop :=
  config.Nodup

/-- Function to count valid move sequences -/
def CountValidMoveSequences (n : ℕ) (initialConfig : FrogConfiguration) (dir : Direction) : ℕ :=
  sorry  -- Implementation details omitted

theorem frog_jumps_equivalence 
  (n : ℕ) 
  (initialConfig : FrogConfiguration) 
  (h : HasDistinctPositions initialConfig) :
  CountValidMoveSequences n initialConfig Direction.Right = 
  CountValidMoveSequences n initialConfig Direction.Left :=
sorry

end NUMINAMATH_CALUDE_frog_jumps_equivalence_l3449_344956


namespace NUMINAMATH_CALUDE_sqrt_sum_problem_l3449_344916

theorem sqrt_sum_problem (y : ℝ) (h : Real.sqrt (64 - y^2) * Real.sqrt (36 - y^2) = 12) :
  Real.sqrt (64 - y^2) + Real.sqrt (36 - y^2) = 7.8 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_problem_l3449_344916


namespace NUMINAMATH_CALUDE_unique_solution_log_equation_l3449_344958

theorem unique_solution_log_equation :
  ∃! x : ℝ, (Real.log (2 * x + 1) = Real.log (x^2 - 2)) ∧ (2 * x + 1 > 0) ∧ (x^2 - 2 > 0) ∧ x = 3 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_log_equation_l3449_344958


namespace NUMINAMATH_CALUDE_vowel_sequences_count_l3449_344912

/-- The number of vowels in the alphabet -/
def num_vowels : ℕ := 5

/-- The length of each sequence -/
def sequence_length : ℕ := 5

/-- Calculates the number of five-letter sequences containing at least one of each vowel -/
def vowel_sequences : ℕ :=
  sequence_length^num_vowels - 
  (Nat.choose num_vowels 1) * (num_vowels - 1)^sequence_length +
  (Nat.choose num_vowels 2) * (num_vowels - 2)^sequence_length -
  (Nat.choose num_vowels 3) * (num_vowels - 3)^sequence_length +
  (Nat.choose num_vowels 4) * (num_vowels - 4)^sequence_length

theorem vowel_sequences_count : vowel_sequences = 120 := by
  sorry

end NUMINAMATH_CALUDE_vowel_sequences_count_l3449_344912


namespace NUMINAMATH_CALUDE_geometric_progression_integers_l3449_344980

/-- A geometric progression with first term b and common ratio r -/
def GeometricProgression (b : ℤ) (r : ℚ) : ℕ → ℚ :=
  fun n => b * r ^ (n - 1)

/-- An arithmetic progression with first term a and common difference d -/
def ArithmeticProgression (a d : ℚ) : ℕ → ℚ :=
  fun n => a + (n - 1) * d

theorem geometric_progression_integers
  (b : ℤ) (r : ℚ) (a d : ℚ)
  (h_subset : ∀ n : ℕ, ∃ m : ℕ, GeometricProgression b r n = ArithmeticProgression a d m) :
  ∀ n : ℕ, ∃ k : ℤ, GeometricProgression b r n = k :=
sorry

end NUMINAMATH_CALUDE_geometric_progression_integers_l3449_344980


namespace NUMINAMATH_CALUDE_smallest_n_for_geometric_sum_l3449_344972

/-- The sum of the first n terms of a geometric sequence with first term a and common ratio r -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The statement to prove -/
theorem smallest_n_for_geometric_sum : 
  ∀ n : ℕ, n > 0 → 
    (geometric_sum (1/3) (1/3) n = 80/243 ↔ n ≥ 5) ∧ 
    (geometric_sum (1/3) (1/3) 5 = 80/243) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_geometric_sum_l3449_344972


namespace NUMINAMATH_CALUDE_min_value_expression_l3449_344931

theorem min_value_expression (a b c : ℝ) (h1 : b > a) (h2 : a > c) (h3 : b ≠ 0) :
  ((a + b)^2 + (b + c)^2 + (c + a)^2) / b^2 ≥ 5.5 ∧
  ∃ (a' b' c' : ℝ), b' > a' ∧ a' > c' ∧ b' ≠ 0 ∧
    ((a' + b')^2 + (b' + c')^2 + (c' + a')^2) / b'^2 = 5.5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3449_344931


namespace NUMINAMATH_CALUDE_kevins_cards_l3449_344911

theorem kevins_cards (x y : ℕ) : x + y = 8 * x → y = 7 * x := by
  sorry

end NUMINAMATH_CALUDE_kevins_cards_l3449_344911


namespace NUMINAMATH_CALUDE_ellipse_tangent_collinearity_and_min_area_l3449_344967

-- Define the ellipse C
def C (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the right focus F
def F : ℝ × ℝ := (1, 0)

-- Define the point P on the line x = 4
def P : ℝ → ℝ × ℝ := λ t => (4, t)

-- Define the tangent points A and B
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- Define the area of triangle PAB
def area_PAB (t : ℝ) : ℝ := sorry

theorem ellipse_tangent_collinearity_and_min_area :
  -- Part 1: A, F, and B are collinear
  ∃ k : ℝ, (1 - k) * A.1 + k * B.1 = F.1 ∧ (1 - k) * A.2 + k * B.2 = F.2 ∧
  -- Part 2: The minimum area of triangle PAB is 9/2
  ∃ t : ℝ, area_PAB t = 9/2 ∧ ∀ s : ℝ, area_PAB s ≥ area_PAB t := by
sorry

end NUMINAMATH_CALUDE_ellipse_tangent_collinearity_and_min_area_l3449_344967
