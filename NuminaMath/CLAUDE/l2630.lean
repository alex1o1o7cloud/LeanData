import Mathlib

namespace NUMINAMATH_CALUDE_power_difference_in_set_l2630_263034

theorem power_difference_in_set (m n : ℕ) :
  (3 ^ m - 2 ^ n ∈ ({-1, 5, 7} : Set ℤ)) ↔ 
  ((m, n) ∈ ({(0, 1), (1, 2), (2, 2), (2, 1)} : Set (ℕ × ℕ))) := by
  sorry

end NUMINAMATH_CALUDE_power_difference_in_set_l2630_263034


namespace NUMINAMATH_CALUDE_first_thrilling_thursday_l2630_263053

/-- Represents a date with a day and a month -/
structure Date where
  day : Nat
  month : Nat

/-- Represents a day of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Function to determine if a given date is a Thursday -/
def isThursday (d : Date) : Bool := sorry

/-- Function to determine if a given date is a Thrilling Thursday -/
def isThrillingThursday (d : Date) : Bool := sorry

/-- The number of days in November -/
def novemberDays : Nat := 30

/-- The start date of the school -/
def schoolStartDate : Date := ⟨2, 11⟩

/-- Theorem stating that the first Thrilling Thursday after school starts is November 30 -/
theorem first_thrilling_thursday :
  let firstThrillingThursday := Date.mk 30 11
  isThursday schoolStartDate ∧
  isThrillingThursday firstThrillingThursday ∧
  (∀ d : Date, schoolStartDate.day ≤ d.day ∧ d.day < firstThrillingThursday.day →
    ¬isThrillingThursday d) := by
  sorry

end NUMINAMATH_CALUDE_first_thrilling_thursday_l2630_263053


namespace NUMINAMATH_CALUDE_juvy_chives_count_l2630_263083

/-- Calculates the number of chives planted in Juvy's garden. -/
def chives_count (total_rows : ℕ) (plants_per_row : ℕ) (parsley_rows : ℕ) (rosemary_rows : ℕ) : ℕ :=
  (total_rows - (parsley_rows + rosemary_rows)) * plants_per_row

/-- Theorem stating that the number of chives Juvy will plant is 150. -/
theorem juvy_chives_count :
  chives_count 20 10 3 2 = 150 := by
  sorry

end NUMINAMATH_CALUDE_juvy_chives_count_l2630_263083


namespace NUMINAMATH_CALUDE_wheel_rotation_on_moving_car_l2630_263077

/-- A wheel is a circular object that can rotate. -/
structure Wheel :=
  (radius : ℝ)
  (center : ℝ × ℝ)

/-- A car is a vehicle with wheels. -/
structure Car :=
  (wheels : List Wheel)

/-- Motion types that an object can exhibit. -/
inductive MotionType
  | Rotation
  | Translation
  | Other

/-- A moving car is a car with a velocity. -/
structure MovingCar extends Car :=
  (velocity : ℝ × ℝ)

/-- The motion type exhibited by a wheel on a moving car. -/
def wheelMotionType (mc : MovingCar) (w : Wheel) : MotionType :=
  sorry

/-- Theorem: The wheels of a moving car exhibit rotational motion. -/
theorem wheel_rotation_on_moving_car (mc : MovingCar) (w : Wheel) 
  (h : w ∈ mc.wheels) : 
  wheelMotionType mc w = MotionType.Rotation :=
sorry

end NUMINAMATH_CALUDE_wheel_rotation_on_moving_car_l2630_263077


namespace NUMINAMATH_CALUDE_correlation_of_product_l2630_263029

-- Define a random function type
def RandomFunction := ℝ → ℝ

-- Define the expectation operator
noncomputable def expectation (X : RandomFunction) : ℝ := sorry

-- Define the correlation function
noncomputable def correlation (X Y : RandomFunction) : ℝ := sorry

-- Define what it means for a random function to be centered
def is_centered (X : RandomFunction) : Prop :=
  expectation X = 0

-- Define what it means for two random functions to be uncorrelated
def are_uncorrelated (X Y : RandomFunction) : Prop :=
  expectation (fun t => X t * Y t) = expectation X * expectation Y

-- State the theorem
theorem correlation_of_product (X Y : RandomFunction) 
  (h1 : is_centered X) (h2 : is_centered Y) (h3 : are_uncorrelated X Y) :
  correlation (fun t => X t * Y t) (fun t => X t * Y t) = 
  correlation X X * correlation Y Y := by sorry

end NUMINAMATH_CALUDE_correlation_of_product_l2630_263029


namespace NUMINAMATH_CALUDE_geometric_sequence_formula_l2630_263019

/-- Given a geometric sequence {a_n} where the first three terms are a-1, a+1, and a+4 respectively,
    prove that the general formula for the nth term is a_n = 4 · (3/2)^(n-1) -/
theorem geometric_sequence_formula (a : ℝ) (a_n : ℕ → ℝ) :
  a_n 1 = a - 1 ∧ a_n 2 = a + 1 ∧ a_n 3 = a + 4 →
  ∀ n : ℕ, a_n n = 4 * (3/2) ^ (n - 1) := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_formula_l2630_263019


namespace NUMINAMATH_CALUDE_seats_between_17_and_39_l2630_263020

/-- The number of seats in the row -/
def total_seats : ℕ := 50

/-- The seat number of the first person -/
def seat1 : ℕ := 17

/-- The seat number of the second person -/
def seat2 : ℕ := 39

/-- The number of seats between two given seat numbers (exclusive) -/
def seats_between (a b : ℕ) : ℕ := 
  if a < b then b - a - 1 else a - b - 1

theorem seats_between_17_and_39 : 
  seats_between seat1 seat2 = 21 := by sorry

end NUMINAMATH_CALUDE_seats_between_17_and_39_l2630_263020


namespace NUMINAMATH_CALUDE_workshop_employees_l2630_263058

theorem workshop_employees :
  ∃ (n k1 k2 : ℕ),
    0 < n ∧ n < 60 ∧
    n = 8 * k1 + 5 ∧
    n = 6 * k2 + 3 ∧
    (n = 21 ∨ n = 45) :=
by sorry

end NUMINAMATH_CALUDE_workshop_employees_l2630_263058


namespace NUMINAMATH_CALUDE_revenue_is_99_l2630_263054

/-- Calculates the total revenue from selling cookies and brownies --/
def calculate_revenue (initial_cookies : ℕ) (initial_brownies : ℕ) 
                      (kyle_cookies_eaten : ℕ) (kyle_brownies_eaten : ℕ)
                      (mom_cookies_eaten : ℕ) (mom_brownies_eaten : ℕ)
                      (cookie_price : ℚ) (brownie_price : ℚ) : ℚ :=
  let remaining_cookies := initial_cookies - (kyle_cookies_eaten + mom_cookies_eaten)
  let remaining_brownies := initial_brownies - (kyle_brownies_eaten + mom_brownies_eaten)
  let cookie_revenue := remaining_cookies * cookie_price
  let brownie_revenue := remaining_brownies * brownie_price
  cookie_revenue + brownie_revenue

/-- Theorem: The total revenue from selling all remaining baked goods is $99 --/
theorem revenue_is_99 : 
  calculate_revenue 60 32 2 2 1 2 1 (3/2) = 99 := by
  sorry

end NUMINAMATH_CALUDE_revenue_is_99_l2630_263054


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l2630_263093

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) : 
  (∀ n, a (n + 1) = a n + 2) →  -- arithmetic sequence with common difference 2
  (a 3)^2 = a 1 * a 4 →  -- a_1, a_3, and a_4 form a geometric sequence
  a 1 = -8 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l2630_263093


namespace NUMINAMATH_CALUDE_sphere_radius_from_shadows_l2630_263048

/-- The radius of a sphere given its shadow and a reference post's shadow. -/
theorem sphere_radius_from_shadows
  (sphere_shadow : ℝ)
  (post_height : ℝ)
  (post_shadow : ℝ)
  (h1 : sphere_shadow = 15)
  (h2 : post_height = 1.5)
  (h3 : post_shadow = 3)
  (h4 : post_shadow > 0) -- Ensure division is valid
  : ∃ (r : ℝ), r = sphere_shadow * (post_height / post_shadow) ∧ r = 7.5 :=
by
  sorry


end NUMINAMATH_CALUDE_sphere_radius_from_shadows_l2630_263048


namespace NUMINAMATH_CALUDE_remaining_value_probability_theorem_l2630_263001

/-- Represents a bag of bills -/
structure Bag where
  tens : ℕ
  fives : ℕ
  ones : ℕ

/-- Calculates the total value of bills in a bag -/
def bagValue (b : Bag) : ℕ := 10 * b.tens + 5 * b.fives + b.ones

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- Calculates the probability of the remaining value in Bag A being greater than Bag B -/
def remainingValueProbability (bagA bagB : Bag) : ℚ :=
  let totalA := choose (bagA.tens + bagA.fives + bagA.ones) 2
  let totalB := choose (bagB.tens + bagB.fives + bagB.ones) 2
  let favorableA := choose bagA.ones 2
  let favorableB := totalB - choose bagB.ones 2
  (favorableA * favorableB : ℚ) / (totalA * totalB : ℚ)

theorem remaining_value_probability_theorem :
  let bagA : Bag := { tens := 2, fives := 0, ones := 3 }
  let bagB : Bag := { tens := 0, fives := 4, ones := 3 }
  remainingValueProbability bagA bagB = 9 / 35 := by
  sorry

end NUMINAMATH_CALUDE_remaining_value_probability_theorem_l2630_263001


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l2630_263017

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

-- Define the theorem
theorem geometric_sequence_product (a : ℕ → ℝ) :
  geometric_sequence a →
  (∀ n : ℕ, a n > 0) →
  a 1 + a 9 = 10 →
  a 1 * a 9 = 16 →
  a 2 * a 5 * a 8 = 64 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l2630_263017


namespace NUMINAMATH_CALUDE_tangents_divide_plane_l2630_263042

/-- The number of regions created by n lines in a plane --/
def num_regions (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | k + 1 => num_regions k + (k + 1)

/-- Theorem: 7 distinct tangents to a circle divide the plane into 29 regions --/
theorem tangents_divide_plane : num_regions 7 = 29 := by
  sorry

/-- Lemma: The number of regions for n tangents follows the recursive formula R(n) = R(n-1) + n --/
lemma regions_recursive_formula (n : ℕ) : num_regions (n + 1) = num_regions n + (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_tangents_divide_plane_l2630_263042


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2630_263043

theorem quadratic_factorization (c d : ℤ) : 
  (∀ x, 25 * x^2 - 160 * x - 144 = (5 * x + c) * (5 * x + d)) → 
  c + 2 * d = -2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2630_263043


namespace NUMINAMATH_CALUDE_chocolate_boxes_l2630_263075

theorem chocolate_boxes (pieces_per_box : ℕ) (total_pieces : ℕ) (h1 : pieces_per_box = 500) (h2 : total_pieces = 3000) :
  total_pieces / pieces_per_box = 6 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_boxes_l2630_263075


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l2630_263087

theorem smallest_prime_divisor_of_sum : ∃ k : ℕ, 4^15 + 6^17 = 2 * k := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l2630_263087


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l2630_263064

/-- An ellipse with the given properties has eccentricity √2 - 1 -/
theorem ellipse_eccentricity (a b c : ℝ) (P : ℝ × ℝ) :
  a > b ∧ b > 0 →  -- Standard ellipse condition
  P.1^2 / a^2 + P.2^2 / b^2 = 1 →  -- P is on the ellipse
  P.1 = c →  -- F₂ is on the x-axis
  abs P.2 = b^2 / a →  -- P is on the perpendicular line through F₂
  P.2^2 = (a^2 - c^2) / 2 →  -- Triangle F₁PF₂ is isosceles right
  c / a = Real.sqrt 2 - 1 :=  -- Eccentricity is √2 - 1
by sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l2630_263064


namespace NUMINAMATH_CALUDE_regular_pay_limit_l2630_263092

/-- The problem of finding the limit for regular pay. -/
theorem regular_pay_limit (regular_rate : ℝ) (overtime_rate : ℝ) (total_pay : ℝ) (overtime_hours : ℝ) :
  regular_rate = 3 →
  overtime_rate = 2 * regular_rate →
  total_pay = 186 →
  overtime_hours = 11 →
  ∃ (regular_hours : ℝ),
    regular_hours * regular_rate + overtime_hours * overtime_rate = total_pay ∧
    regular_hours = 40 :=
by sorry

end NUMINAMATH_CALUDE_regular_pay_limit_l2630_263092


namespace NUMINAMATH_CALUDE_rectangle_width_l2630_263096

/-- Given a rectangular area with a known area and length, prove that its width is 7 feet. -/
theorem rectangle_width (area : ℝ) (length : ℝ) (h1 : area = 35) (h2 : length = 5) :
  area / length = 7 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_width_l2630_263096


namespace NUMINAMATH_CALUDE_missing_shirts_is_eight_l2630_263089

/-- Represents the laundry problem with given conditions -/
structure LaundryProblem where
  trousers_count : ℕ
  total_bill : ℕ
  shirt_cost : ℕ
  trouser_cost : ℕ
  claimed_shirts : ℕ

/-- Calculates the number of missing shirts -/
def missing_shirts (p : LaundryProblem) : ℕ :=
  let total_trouser_cost := p.trousers_count * p.trouser_cost
  let total_shirt_cost := p.total_bill - total_trouser_cost
  let actual_shirts := total_shirt_cost / p.shirt_cost
  actual_shirts - p.claimed_shirts

/-- Theorem stating that the number of missing shirts is 8 -/
theorem missing_shirts_is_eight :
  ∃ (p : LaundryProblem),
    p.trousers_count = 10 ∧
    p.total_bill = 140 ∧
    p.shirt_cost = 5 ∧
    p.trouser_cost = 9 ∧
    p.claimed_shirts = 2 ∧
    missing_shirts p = 8 := by
  sorry

end NUMINAMATH_CALUDE_missing_shirts_is_eight_l2630_263089


namespace NUMINAMATH_CALUDE_existence_of_multiple_factorizations_l2630_263097

/-- The set V_n of integers of the form 1 + kn where k ≥ 1 -/
def V_n (n : ℕ) : Set ℕ := {m | ∃ k : ℕ, k ≥ 1 ∧ m = 1 + k * n}

/-- A number is indecomposable in V_n if it can't be expressed as a product of two numbers from V_n -/
def Indecomposable (n : ℕ) (m : ℕ) : Prop :=
  m ∈ V_n n ∧ ∀ p q : ℕ, p ∈ V_n n → q ∈ V_n n → m ≠ p * q

/-- Two lists of natural numbers are considered different if they are not permutations of each other -/
def DifferentFactorizations (l1 l2 : List ℕ) : Prop :=
  ¬(l1.Perm l2)

theorem existence_of_multiple_factorizations (n : ℕ) (h : n > 2) :
  ∃ r : ℕ, r ∈ V_n n ∧
    ∃ l1 l2 : List ℕ,
      (∀ x ∈ l1, Indecomposable n x) ∧
      (∀ x ∈ l2, Indecomposable n x) ∧
      (r = l1.prod) ∧
      (r = l2.prod) ∧
      DifferentFactorizations l1 l2 :=
sorry


end NUMINAMATH_CALUDE_existence_of_multiple_factorizations_l2630_263097


namespace NUMINAMATH_CALUDE_parallel_lines_a_value_l2630_263007

/-- Two lines are parallel if and only if they have the same slope -/
axiom parallel_lines_same_slope (m₁ m₂ : ℝ) : 
  (∃ b₁ b₂ : ℝ, ∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The value of a when two given lines are parallel -/
theorem parallel_lines_a_value : 
  (∃ b₁ b₂ : ℝ, ∀ x y : ℝ, y = 3 * x + a / 3 ↔ y = (a - 3) * x + 2) → a = 6 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_a_value_l2630_263007


namespace NUMINAMATH_CALUDE_point_inside_circle_l2630_263023

theorem point_inside_circle (a : ℝ) : 
  (1 - a)^2 + (1 + a)^2 < 4 ↔ -1 < a ∧ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_point_inside_circle_l2630_263023


namespace NUMINAMATH_CALUDE_function_characterization_l2630_263028

theorem function_characterization :
  ∀ f : ℕ → ℤ,
  (∀ k l : ℕ, (f k - f l) ∣ (k^2 - l^2)) →
  ∃ (c : ℤ) (g : ℕ → Fin 2),
    (∀ x : ℕ, f x = (-1)^(g x).val * x + c) ∨
    (∀ x : ℕ, f x = x^2 + c) ∨
    (∀ x : ℕ, f x = -x^2 + c) :=
by sorry

end NUMINAMATH_CALUDE_function_characterization_l2630_263028


namespace NUMINAMATH_CALUDE_triangle_area_234_l2630_263085

theorem triangle_area_234 : 
  let a := 2
  let b := 3
  let c := 4
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  area = (3 * Real.sqrt 15) / 4 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_234_l2630_263085


namespace NUMINAMATH_CALUDE_tank_filling_time_l2630_263080

theorem tank_filling_time : ∃ T : ℝ,
  T > 0 ∧
  (T / 2) * (1 / 40) + (T / 2) * (1 / 60 + 1 / 40) = 1 ∧
  T = 30 := by
  sorry

end NUMINAMATH_CALUDE_tank_filling_time_l2630_263080


namespace NUMINAMATH_CALUDE_cube_with_tunnel_surface_area_l2630_263078

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube with a tunnel drilled through it -/
structure CubeWithTunnel where
  sideLength : ℝ
  tunnelVertices : Fin 3 → Point3D

/-- Calculates the surface area of a cube with a tunnel drilled through it -/
def surfaceArea (cube : CubeWithTunnel) : ℝ := sorry

/-- The main theorem stating that the surface area of the cube with tunnel is 864 -/
theorem cube_with_tunnel_surface_area :
  ∃ (cube : CubeWithTunnel),
    cube.sideLength = 12 ∧
    (cube.tunnelVertices 0).x = 3 ∧ (cube.tunnelVertices 0).y = 0 ∧ (cube.tunnelVertices 0).z = 0 ∧
    (cube.tunnelVertices 1).x = 0 ∧ (cube.tunnelVertices 1).y = 12 ∧ (cube.tunnelVertices 1).z = 0 ∧
    (cube.tunnelVertices 2).x = 0 ∧ (cube.tunnelVertices 2).y = 0 ∧ (cube.tunnelVertices 2).z = 3 ∧
    surfaceArea cube = 864 := by
  sorry

end NUMINAMATH_CALUDE_cube_with_tunnel_surface_area_l2630_263078


namespace NUMINAMATH_CALUDE_remainder_67_power_67_plus_67_mod_68_l2630_263041

theorem remainder_67_power_67_plus_67_mod_68 : (67^67 + 67) % 68 = 66 := by
  sorry

end NUMINAMATH_CALUDE_remainder_67_power_67_plus_67_mod_68_l2630_263041


namespace NUMINAMATH_CALUDE_correct_calculation_l2630_263069

theorem correct_calculation (a b : ℝ) : 3 * a^2 * b - 4 * b * a^2 = -a^2 * b := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2630_263069


namespace NUMINAMATH_CALUDE_infinitely_many_non_sum_of_three_cubes_l2630_263012

theorem infinitely_many_non_sum_of_three_cubes :
  ∀ n : ℤ, (n % 9 = 4 ∨ n % 9 = 5) → ¬∃ a b c : ℤ, n = a^3 + b^3 + c^3 :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_non_sum_of_three_cubes_l2630_263012


namespace NUMINAMATH_CALUDE_jogging_no_rain_probability_l2630_263073

theorem jogging_no_rain_probability 
  (p_jog : ℚ) 
  (p_rain : ℚ) 
  (h1 : p_jog = 6 / 10) 
  (h2 : p_rain = 1 / 5) : 
  p_jog * (1 - p_rain) = 12 / 25 := by
sorry

end NUMINAMATH_CALUDE_jogging_no_rain_probability_l2630_263073


namespace NUMINAMATH_CALUDE_weight_replacement_l2630_263000

theorem weight_replacement (n : ℕ) (old_avg new_avg new_weight : ℝ) : 
  n = 8 → 
  new_avg - old_avg = 2.5 →
  new_weight = 70 →
  (n * new_avg - new_weight + (n * old_avg - n * new_avg)) / (n - 1) = 50 :=
by sorry

end NUMINAMATH_CALUDE_weight_replacement_l2630_263000


namespace NUMINAMATH_CALUDE_square_EFGH_area_l2630_263086

theorem square_EFGH_area : 
  ∀ (original_side_length : ℝ) (EFGH_side_length : ℝ),
  original_side_length = 8 →
  EFGH_side_length = original_side_length + 2 * (original_side_length / 2) →
  EFGH_side_length^2 = 256 :=
by sorry

end NUMINAMATH_CALUDE_square_EFGH_area_l2630_263086


namespace NUMINAMATH_CALUDE_correct_num_raised_beds_l2630_263004

/-- The number of raised beds Abby is building -/
def num_raised_beds : ℕ := 2

/-- The length of each raised bed in feet -/
def bed_length : ℕ := 8

/-- The width of each raised bed in feet -/
def bed_width : ℕ := 4

/-- The height of each raised bed in feet -/
def bed_height : ℕ := 1

/-- The volume of soil in each bag in cubic feet -/
def soil_per_bag : ℕ := 4

/-- The total number of soil bags needed -/
def total_soil_bags : ℕ := 16

/-- Theorem stating that the number of raised beds Abby is building is correct -/
theorem correct_num_raised_beds :
  num_raised_beds * (bed_length * bed_width * bed_height) = total_soil_bags * soil_per_bag :=
by sorry

end NUMINAMATH_CALUDE_correct_num_raised_beds_l2630_263004


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2630_263003

def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {x | x > 1}

theorem intersection_of_A_and_B : A ∩ B = {2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2630_263003


namespace NUMINAMATH_CALUDE_symmetric_point_x_axis_l2630_263009

/-- Given a point M(3, -4), its symmetric point with respect to the x-axis has coordinates (3, 4) -/
theorem symmetric_point_x_axis : 
  let M : ℝ × ℝ := (3, -4)
  let symmetric_point (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)
  symmetric_point M = (3, 4) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_x_axis_l2630_263009


namespace NUMINAMATH_CALUDE_f_solutions_l2630_263070

def f (x : ℝ) : ℝ := x^2 + 2*x + 4

theorem f_solutions : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ = 12 ∧ f x₂ = 12 ∧ 
  (∀ x : ℝ, f x = 12 → x = x₁ ∨ x = x₂) :=
sorry

end NUMINAMATH_CALUDE_f_solutions_l2630_263070


namespace NUMINAMATH_CALUDE_equilateral_triangle_on_parabola_l2630_263072

/-- An equilateral triangle with one vertex at the origin and two vertices on y^2 = 4x has side length 8√3 -/
theorem equilateral_triangle_on_parabola :
  ∀ (A B C : ℝ × ℝ),
    A = (0, 0) →
    B.1 ≥ 0 →
    C.1 ≥ 0 →
    B.2^2 = 4 * B.1 →
    C.2^2 = 4 * C.1 →
    (B.1 - A.1)^2 + (B.2 - A.2)^2 = (C.1 - A.1)^2 + (C.2 - A.2)^2 →
    (B.1 - C.1)^2 + (B.2 - C.2)^2 = (B.1 - A.1)^2 + (B.2 - A.2)^2 →
    (C.1 - A.1)^2 + (C.2 - A.2)^2 = (C.1 - B.1)^2 + (C.2 - B.2)^2 →
    (B.1 - A.1)^2 + (B.2 - A.2)^2 = 192 :=
by sorry


end NUMINAMATH_CALUDE_equilateral_triangle_on_parabola_l2630_263072


namespace NUMINAMATH_CALUDE_condition_property_l2630_263026

theorem condition_property (a b : ℝ) :
  (∀ a b, a > b ∧ b > 0 → a + a^2 > b + b^2) ∧
  (∃ a b, a + a^2 > b + b^2 ∧ ¬(a > b ∧ b > 0)) :=
by sorry

end NUMINAMATH_CALUDE_condition_property_l2630_263026


namespace NUMINAMATH_CALUDE_two_real_roots_for_radical_equation_l2630_263033

-- Define the function f(x) derived from the original equation
def f (a b c x : ℝ) : ℝ :=
  3 * x^2 - 2 * (a + b + c) * x - (a^2 + b^2 + c^2) + 2 * (a * b + b * c + c * a)

-- Main theorem statement
theorem two_real_roots_for_radical_equation (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧
  (∀ x : ℝ, f a b c x = 0 ↔ x = x₁ ∨ x = x₂) :=
sorry

end NUMINAMATH_CALUDE_two_real_roots_for_radical_equation_l2630_263033


namespace NUMINAMATH_CALUDE_dave_initial_apps_l2630_263091

/-- The number of apps Dave initially had on his phone -/
def initial_apps : ℕ := sorry

/-- The number of apps Dave deleted -/
def deleted_apps : ℕ := 18

/-- The number of apps remaining after deletion -/
def remaining_apps : ℕ := 5

/-- Theorem stating that Dave initially had 23 apps -/
theorem dave_initial_apps : initial_apps = 23 := by
  sorry

end NUMINAMATH_CALUDE_dave_initial_apps_l2630_263091


namespace NUMINAMATH_CALUDE_brick_surface_area_l2630_263027

/-- The surface area of a rectangular prism -/
def surface_area (length width height : ℝ) : ℝ :=
  2 * (length * width + length * height + width * height)

/-- Theorem: The surface area of a 10 cm x 4 cm x 3 cm rectangular prism is 164 square centimeters -/
theorem brick_surface_area :
  surface_area 10 4 3 = 164 := by
  sorry

end NUMINAMATH_CALUDE_brick_surface_area_l2630_263027


namespace NUMINAMATH_CALUDE_no_x_squared_term_l2630_263013

theorem no_x_squared_term (a : ℝ) : 
  (∀ x : ℝ, (x + 1) * (x^2 - 5*a*x + a) = x^3 + (-4*a)*x + a) → a = 1/5 := by
sorry

end NUMINAMATH_CALUDE_no_x_squared_term_l2630_263013


namespace NUMINAMATH_CALUDE_passing_marks_calculation_l2630_263066

theorem passing_marks_calculation (T : ℝ) (P : ℝ) : 
  (0.35 * T = P - 40) → 
  (0.60 * T = P + 25) → 
  P = 131 := by
  sorry

end NUMINAMATH_CALUDE_passing_marks_calculation_l2630_263066


namespace NUMINAMATH_CALUDE_three_integers_with_difference_and_quotient_l2630_263021

theorem three_integers_with_difference_and_quotient :
  ∃ (a b c : ℤ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a = b - c ∧ b = c / a := by
  sorry

end NUMINAMATH_CALUDE_three_integers_with_difference_and_quotient_l2630_263021


namespace NUMINAMATH_CALUDE_pizza_area_increase_l2630_263037

theorem pizza_area_increase (d₁ d₂ d₃ : ℝ) (h₁ : d₁ = 8) (h₂ : d₂ = 10) (h₃ : d₃ = 14) :
  let area (d : ℝ) := Real.pi * (d / 2)^2
  let percent_increase (a₁ a₂ : ℝ) := (a₂ - a₁) / a₁ * 100
  (percent_increase (area d₁) (area d₂) = 56.25) ∧
  (percent_increase (area d₂) (area d₃) = 96) := by
  sorry

end NUMINAMATH_CALUDE_pizza_area_increase_l2630_263037


namespace NUMINAMATH_CALUDE_unique_base_solution_l2630_263082

/-- Converts a number from base b to base 10 -/
def toBase10 (n : ℕ) (b : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds * b^2 + tens * b + ones

/-- The main theorem statement -/
theorem unique_base_solution :
  ∃! b : ℕ, b > 7 ∧ (toBase10 276 b) * 2 + (toBase10 145 b) = (toBase10 697 b) :=
by sorry

end NUMINAMATH_CALUDE_unique_base_solution_l2630_263082


namespace NUMINAMATH_CALUDE_inscribed_prism_properties_l2630_263036

/-- Regular triangular pyramid with inscribed regular triangular prism -/
structure PyramidWithPrism where
  pyramid_height : ℝ
  pyramid_base_side : ℝ
  prism_lateral_area : ℝ

/-- Possible solutions for the inscribed prism -/
structure PrismSolution where
  prism_height : ℝ
  lateral_area_ratio : ℝ

/-- Theorem stating the properties of the inscribed prism -/
theorem inscribed_prism_properties (p : PyramidWithPrism) 
  (h1 : p.pyramid_height = 15)
  (h2 : p.pyramid_base_side = 12)
  (h3 : p.prism_lateral_area = 120) :
  ∃ (s1 s2 : PrismSolution),
    (s1.prism_height = 10 ∧ s1.lateral_area_ratio = 1/9) ∧
    (s2.prism_height = 5 ∧ s2.lateral_area_ratio = 4/9) :=
sorry

end NUMINAMATH_CALUDE_inscribed_prism_properties_l2630_263036


namespace NUMINAMATH_CALUDE_existence_of_smaller_value_l2630_263079

open Set
open Function
open Real

theorem existence_of_smaller_value (f : ℝ → ℝ) (hf : ∀ x, 0 < x → 0 < f x) :
  ∃ x y, 0 < x ∧ 0 < y ∧ f (x + y) < f x + y * f (f x) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_smaller_value_l2630_263079


namespace NUMINAMATH_CALUDE_hundred_power_ten_as_sum_of_tens_l2630_263055

theorem hundred_power_ten_as_sum_of_tens (n : ℕ) :
  (100 ^ 10 : ℕ) = n * 10 → n = 10^19 := by
  sorry

end NUMINAMATH_CALUDE_hundred_power_ten_as_sum_of_tens_l2630_263055


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2630_263047

theorem isosceles_triangle_perimeter (a b c : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Sides are positive
  (a = 2 ∧ b = 5) ∨ (a = 5 ∧ b = 2) →  -- Two sides measure 2 and 5
  a + b > c ∧ b + c > a ∧ c + a > b →  -- Triangle inequality
  (a = b ∨ b = c ∨ c = a) →  -- Isosceles condition
  a + b + c = 12 :=  -- Perimeter is 12
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2630_263047


namespace NUMINAMATH_CALUDE_cubic_roots_problem_l2630_263051

theorem cubic_roots_problem (u v c d : ℝ) : 
  (∃ w, {u, v, w} = {x | x^3 + c*x + d = 0}) ∧
  (∃ w', {u+5, v-4, w'} = {x | x^3 + c*x + (d+300) = 0}) →
  d = -4 ∨ d = 6 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_problem_l2630_263051


namespace NUMINAMATH_CALUDE_lcm_of_fractions_l2630_263040

theorem lcm_of_fractions (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  lcm (1 / x) (lcm (1 / (x * y)) (1 / (x * y * z))) = 1 / (x * y * z) := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_fractions_l2630_263040


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_quadratic_l2630_263067

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  r₁ + r₂ = -b / a := by sorry

theorem sum_of_roots_specific_quadratic :
  let r₁ := (5 + Real.sqrt 1) / 2
  let r₂ := (5 - Real.sqrt 1) / 2
  r₁ + r₂ = 5 := by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_quadratic_l2630_263067


namespace NUMINAMATH_CALUDE_inverse_trig_sum_equals_pi_l2630_263030

theorem inverse_trig_sum_equals_pi : 
  let arctan_sqrt3 := π / 3
  let arcsin_neg_half := -π / 6
  let arccos_zero := π / 2
  arctan_sqrt3 - arcsin_neg_half + arccos_zero = π := by sorry

end NUMINAMATH_CALUDE_inverse_trig_sum_equals_pi_l2630_263030


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l2630_263056

/-- The line mx-y+2m+1=0 passes through the fixed point (-2, 1) for all real m -/
theorem line_passes_through_fixed_point :
  ∀ (m : ℝ), m * (-2) - 1 + 2 * m + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l2630_263056


namespace NUMINAMATH_CALUDE_no_gar_is_tren_l2630_263006

-- Define the types
variable (U : Type) -- Universe set
variable (Gar Plin Tren : Set U) -- Subsets of U

-- Define the hypotheses
variable (h1 : Gar ⊆ Plin) -- All Gars are Plins
variable (h2 : Plin ∩ Tren = ∅) -- No Plins are Trens

-- State the theorem
theorem no_gar_is_tren : Gar ∩ Tren = ∅ := by
  sorry

end NUMINAMATH_CALUDE_no_gar_is_tren_l2630_263006


namespace NUMINAMATH_CALUDE_responder_is_liar_responder_is_trulalala_l2630_263016

-- Define the brothers
inductive Brother
| Tweedledee
| Trulalala

-- Define the possible responses
inductive Response
| Circle
| Square

-- Define the property of being truthful or a liar
def isTruthful (b : Brother) : Prop :=
  match b with
  | Brother.Tweedledee => true
  | Brother.Trulalala => false

-- Define the question asked
def questionAsked (actual : Response) : Prop :=
  actual = Response.Square

-- Define the response given
def responseGiven : Response := Response.Circle

-- Theorem to prove
theorem responder_is_liar :
  ∀ (responder asker : Brother),
  responder ≠ asker →
  (isTruthful responder ↔ ¬isTruthful asker) →
  responseGiven = Response.Circle →
  ¬isTruthful responder := by
  sorry

-- Corollary: The responder is Trulalala
theorem responder_is_trulalala :
  ∀ (responder asker : Brother),
  responder ≠ asker →
  (isTruthful responder ↔ ¬isTruthful asker) →
  responseGiven = Response.Circle →
  responder = Brother.Trulalala := by
  sorry

end NUMINAMATH_CALUDE_responder_is_liar_responder_is_trulalala_l2630_263016


namespace NUMINAMATH_CALUDE_union_M_N_l2630_263015

def M : Set ℤ := {x | |x| < 2}
def N : Set ℤ := {-2, -1, 0}

theorem union_M_N : M ∪ N = {-2, -1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_union_M_N_l2630_263015


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l2630_263005

theorem line_segment_endpoint (y : ℝ) : y > 0 →
  Real.sqrt ((3 - (-5))^2 + (7 - y)^2) = 12 →
  y = 7 + 4 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l2630_263005


namespace NUMINAMATH_CALUDE_project_hours_difference_l2630_263095

theorem project_hours_difference (total_hours : ℕ) (kate_hours : ℕ) : 
  total_hours = 117 →
  2 * kate_hours + kate_hours + 6 * kate_hours = total_hours →
  6 * kate_hours - kate_hours = 65 := by
  sorry

end NUMINAMATH_CALUDE_project_hours_difference_l2630_263095


namespace NUMINAMATH_CALUDE_arithmetic_sequence_product_l2630_263014

/-- An arithmetic sequence of integers -/
def ArithmeticSequence (b : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, b (n + 1) = b n + d

theorem arithmetic_sequence_product (b : ℕ → ℤ) :
  ArithmeticSequence b →
  (∀ n : ℕ, b (n + 1) > b n) →
  b 5 * b 6 = 21 →
  b 4 * b 7 = -779 ∨ b 4 * b 7 = -11 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_product_l2630_263014


namespace NUMINAMATH_CALUDE_hypotenuse_length_16_l2630_263038

/-- A right triangle with one angle of 30 degrees -/
structure RightTriangle30 where
  /-- The length of the side opposite to the 30° angle -/
  short_side : ℝ
  /-- The short side is positive -/
  short_side_pos : 0 < short_side

/-- The length of the hypotenuse in a right triangle with a 30° angle -/
def hypotenuse (t : RightTriangle30) : ℝ := 2 * t.short_side

/-- Theorem: In a right triangle with a 30° angle, if the short side is 8, then the hypotenuse is 16 -/
theorem hypotenuse_length_16 (t : RightTriangle30) (h : t.short_side = 8) : 
  hypotenuse t = 16 := by sorry

end NUMINAMATH_CALUDE_hypotenuse_length_16_l2630_263038


namespace NUMINAMATH_CALUDE_odd_square_minus_one_div_eight_l2630_263063

theorem odd_square_minus_one_div_eight (a : ℤ) (h : ∃ k : ℤ, a = 2 * k + 1) :
  ∃ m : ℤ, a^2 - 1 = 8 * m :=
by sorry

end NUMINAMATH_CALUDE_odd_square_minus_one_div_eight_l2630_263063


namespace NUMINAMATH_CALUDE_average_expenditure_feb_to_july_l2630_263088

/-- Calculates the average expenditure for February to July given the conditions -/
theorem average_expenditure_feb_to_july 
  (avg_jan_to_june : ℝ) 
  (expenditure_jan : ℝ) 
  (expenditure_july : ℝ) 
  (h1 : avg_jan_to_june = 4200)
  (h2 : expenditure_jan = 1200)
  (h3 : expenditure_july = 1500) :
  (6 * avg_jan_to_june - expenditure_jan + expenditure_july) / 6 = 4250 := by
  sorry

#check average_expenditure_feb_to_july

end NUMINAMATH_CALUDE_average_expenditure_feb_to_july_l2630_263088


namespace NUMINAMATH_CALUDE_mickey_mounts_98_horses_per_week_l2630_263011

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of horses Minnie mounts per day -/
def minnie_horses_per_day : ℕ := days_in_week + 3

/-- The number of horses Mickey mounts per day -/
def mickey_horses_per_day : ℕ := 2 * minnie_horses_per_day - 6

/-- The number of horses Mickey mounts per week -/
def mickey_horses_per_week : ℕ := mickey_horses_per_day * days_in_week

theorem mickey_mounts_98_horses_per_week :
  mickey_horses_per_week = 98 := by
  sorry

end NUMINAMATH_CALUDE_mickey_mounts_98_horses_per_week_l2630_263011


namespace NUMINAMATH_CALUDE_imaginary_number_theorem_l2630_263099

theorem imaginary_number_theorem (z : ℂ) :
  (∃ a : ℝ, z = a * I) →
  ((z + 2) / (1 - I)).im = 0 →
  z = -2 * I :=
by sorry

end NUMINAMATH_CALUDE_imaginary_number_theorem_l2630_263099


namespace NUMINAMATH_CALUDE_geometric_sum_first_8_terms_l2630_263084

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sum_first_8_terms :
  let a : ℚ := 1/2
  let r : ℚ := 1/3
  let n : ℕ := 8
  geometric_sum a r n = 9840/6561 := by
sorry

end NUMINAMATH_CALUDE_geometric_sum_first_8_terms_l2630_263084


namespace NUMINAMATH_CALUDE_abc_sum_equals_36_l2630_263065

theorem abc_sum_equals_36 (a b c : ℕ+) 
  (h : (4 : ℕ)^(a.val) * (5 : ℕ)^(b.val) * (6 : ℕ)^(c.val) = (8 : ℕ)^8 * (9 : ℕ)^9 * (10 : ℕ)^10) : 
  a.val + b.val + c.val = 36 := by
sorry

end NUMINAMATH_CALUDE_abc_sum_equals_36_l2630_263065


namespace NUMINAMATH_CALUDE_original_ratio_l2630_263050

theorem original_ratio (x y : ℕ) (h1 : x = y + 5) (h2 : (x - 5) / (y - 5) = 5 / 4) : x / y = 6 / 5 := by
  sorry

end NUMINAMATH_CALUDE_original_ratio_l2630_263050


namespace NUMINAMATH_CALUDE_equation_D_is_correct_l2630_263062

theorem equation_D_is_correct (x : ℝ) : 2 * x^2 * (3 * x)^2 = 18 * x^4 := by
  sorry

end NUMINAMATH_CALUDE_equation_D_is_correct_l2630_263062


namespace NUMINAMATH_CALUDE_equation_solution_l2630_263032

theorem equation_solution (a b : ℝ) :
  b ≠ 0 →
  (∀ x, (4 * a * x + 1) / b - 5 = 3 * x / b) ↔
  (b = 0 ∧ False) ∨
  (a = 3/4 ∧ b = 1/5) ∨
  (4 * a - 3 ≠ 0 ∧ ∃! x, x = (5 * b - 1) / (4 * a - 3)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2630_263032


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l2630_263052

theorem complex_fraction_equality (a b : ℂ) 
  (h1 : (a + b) / (a - b) - (a - b) / (a + b) = 2)
  (h2 : a^2 + b^2 ≠ 0) :
  (a^4 + b^4) / (a^4 - b^4) - (a^4 - b^4) / (a^4 + b^4) = ((a^2 + b^2) - 3) / 3 :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l2630_263052


namespace NUMINAMATH_CALUDE_book_pages_theorem_l2630_263049

/-- Represents a reading pattern of a book -/
structure ReadingPattern where
  first_day : ℕ
  daily_increase : ℕ
  pages_left : ℕ

/-- Calculates the total number of pages in a book based on two reading patterns -/
def calculate_total_pages (r1 r2 : ReadingPattern) : ℕ :=
  sorry

/-- The theorem stating the total number of pages in the book -/
theorem book_pages_theorem (r1 r2 : ReadingPattern) 
  (h1 : r1.first_day = 35 ∧ r1.daily_increase = 5 ∧ r1.pages_left = 35)
  (h2 : r2.first_day = 45 ∧ r2.daily_increase = 5 ∧ r2.pages_left = 40) :
  calculate_total_pages r1 r2 = 385 :=
sorry

end NUMINAMATH_CALUDE_book_pages_theorem_l2630_263049


namespace NUMINAMATH_CALUDE_quadrilateral_equal_area_implies_midpoint_l2630_263061

/-- A quadrilateral in 2D space -/
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

/-- A point in 2D space -/
def Point := ℝ × ℝ

/-- The area of a triangle given its vertices -/
def triangleArea (p q r : Point) : ℝ := sorry

/-- Check if a point is inside a quadrilateral -/
def isInside (E : Point) (quad : Quadrilateral) : Prop := sorry

/-- Check if a point is the midpoint of a line segment -/
def isMidpoint (M : Point) (A B : Point) : Prop := sorry

theorem quadrilateral_equal_area_implies_midpoint 
  (quad : Quadrilateral) (E : Point) :
  isInside E quad →
  (triangleArea E quad.A quad.B = triangleArea E quad.B quad.C) ∧
  (triangleArea E quad.B quad.C = triangleArea E quad.C quad.D) ∧
  (triangleArea E quad.C quad.D = triangleArea E quad.D quad.A) →
  (isMidpoint E quad.A quad.C) ∨ (isMidpoint E quad.B quad.D) := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_equal_area_implies_midpoint_l2630_263061


namespace NUMINAMATH_CALUDE_integer_equation_solution_l2630_263076

theorem integer_equation_solution (x y : ℤ) : 
  x^2 = y^2 + 2*y + 13 ↔ (x = 4 ∧ y = 1) ∨ (x = -4 ∧ y = 1) ∨ (x = 4 ∧ y = -3) ∨ (x = -4 ∧ y = -3) := by
  sorry

end NUMINAMATH_CALUDE_integer_equation_solution_l2630_263076


namespace NUMINAMATH_CALUDE_lyon_marseille_distance_l2630_263046

/-- Given a map distance and scale, calculates the real distance between two points. -/
def real_distance (map_distance : ℝ) (scale : ℝ) : ℝ :=
  map_distance * scale

/-- Proves that the real distance between Lyon and Marseille is 1200 km. -/
theorem lyon_marseille_distance :
  let map_distance : ℝ := 120
  let scale : ℝ := 10
  real_distance map_distance scale = 1200 := by
  sorry

end NUMINAMATH_CALUDE_lyon_marseille_distance_l2630_263046


namespace NUMINAMATH_CALUDE_corresponding_time_l2630_263031

-- Define the ratio
def ratio : ℚ := 8 / 4

-- Define the conversion factor from seconds to minutes
def seconds_to_minutes : ℚ := 1 / 60

-- State the theorem
theorem corresponding_time (t : ℚ) : 
  ratio = 8 / t → t = 4 * seconds_to_minutes :=
by sorry

end NUMINAMATH_CALUDE_corresponding_time_l2630_263031


namespace NUMINAMATH_CALUDE_base_eight_47_equals_39_l2630_263039

/-- Converts a two-digit base-eight number to base-ten -/
def base_eight_to_ten (a b : Nat) : Nat :=
  a * 8 + b

/-- The base-eight number 47 is equal to the base-ten number 39 -/
theorem base_eight_47_equals_39 : base_eight_to_ten 4 7 = 39 := by
  sorry

end NUMINAMATH_CALUDE_base_eight_47_equals_39_l2630_263039


namespace NUMINAMATH_CALUDE_jack_walking_time_l2630_263002

/-- Proves that given a distance of 9 miles and a rate of 7.2 miles per hour, the time taken is 1.25 hours. -/
theorem jack_walking_time (distance : ℝ) (rate : ℝ) (time : ℝ) :
  distance = 9 →
  rate = 7.2 →
  time = distance / rate →
  time = 1.25 := by
sorry

end NUMINAMATH_CALUDE_jack_walking_time_l2630_263002


namespace NUMINAMATH_CALUDE_james_purchase_cost_l2630_263057

/-- Calculates the total cost of James' purchase --/
def totalCost (bedFramePrice bedPrice bedsideTablePrice bedFrameDiscount bedDiscount bedsideTableDiscount salesTax : ℝ) : ℝ :=
  let discountedBedFramePrice := bedFramePrice * (1 - bedFrameDiscount)
  let discountedBedPrice := bedPrice * (1 - bedDiscount)
  let discountedBedsideTablePrice := bedsideTablePrice * (1 - bedsideTableDiscount)
  let totalDiscountedPrice := discountedBedFramePrice + discountedBedPrice + discountedBedsideTablePrice
  totalDiscountedPrice * (1 + salesTax)

/-- Theorem stating the total cost of James' purchase --/
theorem james_purchase_cost :
  totalCost 75 750 120 0.20 0.20 0.15 0.085 = 826.77 := by
  sorry

end NUMINAMATH_CALUDE_james_purchase_cost_l2630_263057


namespace NUMINAMATH_CALUDE_ognev_phone_number_l2630_263035

/-- Represents a surname -/
structure Surname :=
  (name : String)

/-- Calculates the length of a surname -/
def surname_length (s : Surname) : Nat :=
  s.name.length

/-- Gets the position of a character in the alphabet (A=1, B=2, etc.) -/
def char_position (c : Char) : Nat :=
  if 'A' ≤ c ∧ c ≤ 'Z' then c.toNat - 'A'.toNat + 1
  else if 'a' ≤ c ∧ c ≤ 'z' then c.toNat - 'a'.toNat + 1
  else 0

/-- Calculates the phone number for a given surname -/
def phone_number (s : Surname) : Nat :=
  let len := surname_length s
  let first_pos := char_position s.name.front
  let last_pos := char_position s.name.back
  len * 1000 + first_pos * 100 + last_pos

/-- The theorem to be proved -/
theorem ognev_phone_number :
  phone_number { name := "Ognev" } = 5163 := by
  sorry

end NUMINAMATH_CALUDE_ognev_phone_number_l2630_263035


namespace NUMINAMATH_CALUDE_equality_check_l2630_263094

theorem equality_check : 
  ((-2 : ℤ)^3 ≠ -2 * 3) ∧ 
  (2^3 ≠ 3^2) ∧ 
  ((-2 : ℤ)^3 = -2^3) ∧ 
  (-3^2 ≠ (-3)^2) := by
  sorry

end NUMINAMATH_CALUDE_equality_check_l2630_263094


namespace NUMINAMATH_CALUDE_percentage_value_in_quarters_l2630_263045

/-- Represents the number of nickels --/
def num_nickels : ℕ := 80

/-- Represents the number of quarters --/
def num_quarters : ℕ := 40

/-- Represents the value of a nickel in cents --/
def nickel_value : ℕ := 5

/-- Represents the value of a quarter in cents --/
def quarter_value : ℕ := 25

/-- Theorem stating that the percentage of total value in quarters is 5/7 --/
theorem percentage_value_in_quarters :
  (num_quarters * quarter_value : ℚ) / (num_nickels * nickel_value + num_quarters * quarter_value) = 5 / 7 := by
  sorry

end NUMINAMATH_CALUDE_percentage_value_in_quarters_l2630_263045


namespace NUMINAMATH_CALUDE_fish_per_person_l2630_263018

/-- Represents the number of fish eyes Oomyapeck eats in a day -/
def eyes_eaten : ℕ := 22

/-- Represents the number of fish eyes Oomyapeck gives to his dog -/
def eyes_to_dog : ℕ := 2

/-- Represents the number of eyes each fish has -/
def eyes_per_fish : ℕ := 2

/-- Represents the number of family members -/
def family_members : ℕ := 3

theorem fish_per_person (eyes_eaten : ℕ) (eyes_to_dog : ℕ) (eyes_per_fish : ℕ) (family_members : ℕ) :
  eyes_eaten = 22 →
  eyes_to_dog = 2 →
  eyes_per_fish = 2 →
  family_members = 3 →
  (eyes_eaten - eyes_to_dog) / eyes_per_fish = 10 :=
by sorry

end NUMINAMATH_CALUDE_fish_per_person_l2630_263018


namespace NUMINAMATH_CALUDE_system_integer_solutions_l2630_263060

theorem system_integer_solutions (a b c d : ℤ) :
  (∀ m n : ℤ, ∃ x y : ℤ, a * x + b * y = m ∧ c * x + d * y = n) →
  (a * d - b * c = 1 ∨ a * d - b * c = -1) :=
sorry

end NUMINAMATH_CALUDE_system_integer_solutions_l2630_263060


namespace NUMINAMATH_CALUDE_koala_fiber_consumption_l2630_263081

/-- Proves that if a koala absorbs 30% of the fiber it eats and it absorbed 12 ounces of fiber in one day, then the total amount of fiber eaten by the koala that day was 40 ounces. -/
theorem koala_fiber_consumption (absorbed_percentage : ℝ) (absorbed_amount : ℝ) (total_amount : ℝ) : 
  absorbed_percentage = 0.30 →
  absorbed_amount = 12 →
  absorbed_amount = absorbed_percentage * total_amount →
  total_amount = 40 := by
sorry

end NUMINAMATH_CALUDE_koala_fiber_consumption_l2630_263081


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2630_263022

def A : Set ℝ := {x | -1 < x ∧ x < 2}
def B : Set ℝ := {x | ∃ y, y = Real.log x}

theorem intersection_of_A_and_B : A ∩ B = {x | 0 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2630_263022


namespace NUMINAMATH_CALUDE_magnitude_of_complex_number_l2630_263010

theorem magnitude_of_complex_number (z : ℂ) : z = (2 * Complex.I) / (1 - Complex.I) → Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_number_l2630_263010


namespace NUMINAMATH_CALUDE_geometric_sequence_theorem_l2630_263090

/-- Represents a geometric sequence -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a 2 / a 1

/-- Given conditions for the geometric sequence -/
def satisfies_conditions (seq : GeometricSequence) : Prop :=
  seq.a 3 + seq.a 6 = 36 ∧ seq.a 4 + seq.a 7 = 18

theorem geometric_sequence_theorem (seq : GeometricSequence) 
  (h : satisfies_conditions seq) : 
  ∃ n : ℕ, seq.a n = 1/2 ∧ n = 9 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_theorem_l2630_263090


namespace NUMINAMATH_CALUDE_possibly_six_l2630_263044

/-- Represents the possible outcomes of a dice throw -/
inductive DiceOutcome
  | one
  | two
  | three
  | four
  | five
  | six

/-- A fair six-sided dice -/
structure FairDice :=
  (outcomes : Finset DiceOutcome)
  (fair : outcomes.card = 6)
  (complete : ∀ o : DiceOutcome, o ∈ outcomes)

/-- The result of a single throw of a fair dice -/
def singleThrow (d : FairDice) : Set DiceOutcome :=
  d.outcomes

theorem possibly_six (d : FairDice) : 
  DiceOutcome.six ∈ singleThrow d :=
sorry

end NUMINAMATH_CALUDE_possibly_six_l2630_263044


namespace NUMINAMATH_CALUDE_cab_driver_income_l2630_263024

theorem cab_driver_income (income2 income3 income4 income5 avg_income : ℕ)
  (h1 : income2 = 150)
  (h2 : income3 = 750)
  (h3 : income4 = 200)
  (h4 : income5 = 600)
  (h5 : avg_income = 400)
  (h6 : ∃ income1 : ℕ, (income1 + income2 + income3 + income4 + income5) / 5 = avg_income) :
  ∃ income1 : ℕ, income1 = 300 ∧ (income1 + income2 + income3 + income4 + income5) / 5 = avg_income :=
by
  sorry

end NUMINAMATH_CALUDE_cab_driver_income_l2630_263024


namespace NUMINAMATH_CALUDE_triangle_area_l2630_263059

/-- The area of a triangle with vertices at (-3, 7), (-7, 3), and (0, 0) in a coordinate plane is 50 square units. -/
theorem triangle_area : Real := by
  -- Define the vertices of the triangle
  let v1 : Prod Real Real := (-3, 7)
  let v2 : Prod Real Real := (-7, 3)
  let v3 : Prod Real Real := (0, 0)

  -- Calculate the area of the triangle
  let area : Real := sorry

  -- Prove that the calculated area is equal to 50
  have h : area = 50 := by sorry

  -- Return the area
  exact 50


end NUMINAMATH_CALUDE_triangle_area_l2630_263059


namespace NUMINAMATH_CALUDE_speech_competition_selection_l2630_263068

def total_students : Nat := 9
def num_boys : Nat := 5
def num_girls : Nat := 4
def students_to_select : Nat := 4

def selection_methods : Nat := sorry

theorem speech_competition_selection :
  (total_students = num_boys + num_girls) →
  (students_to_select ≤ total_students) →
  (selection_methods = 86) := by sorry

end NUMINAMATH_CALUDE_speech_competition_selection_l2630_263068


namespace NUMINAMATH_CALUDE_arithmetic_series_sum_1_to_21_l2630_263074

/-- The sum of an arithmetic series with first term a, last term l, and n terms -/
def arithmetic_series_sum (a l n : ℕ) : ℕ := n * (a + l) / 2

/-- The number of terms in an arithmetic series with first term a, last term l, and common difference d -/
def arithmetic_series_length (a l d : ℕ) : ℕ := (l - a) / d + 1

theorem arithmetic_series_sum_1_to_21 : 
  let a := 1  -- first term
  let l := 21 -- last term
  let d := 2  -- common difference
  let n := arithmetic_series_length a l d
  arithmetic_series_sum a l n = 121 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_series_sum_1_to_21_l2630_263074


namespace NUMINAMATH_CALUDE_tomato_yield_per_plant_l2630_263071

theorem tomato_yield_per_plant 
  (rows : ℕ) 
  (plants_per_row : ℕ) 
  (total_yield : ℕ) 
  (h1 : rows = 30)
  (h2 : plants_per_row = 10)
  (h3 : total_yield = 6000) :
  total_yield / (rows * plants_per_row) = 20 := by
sorry

end NUMINAMATH_CALUDE_tomato_yield_per_plant_l2630_263071


namespace NUMINAMATH_CALUDE_smallest_base_perfect_square_l2630_263098

/-- 
Given a base b > 4, returns the value of 45 in base b expressed in decimal.
-/
def base_b_to_decimal (b : ℕ) : ℕ := 4 * b + 5

/-- 
Proposition: 5 is the smallest integer b > 4 for which 45_b is a perfect square.
-/
theorem smallest_base_perfect_square : 
  (∀ b : ℕ, b > 4 ∧ b < 5 → ¬ ∃ k : ℕ, base_b_to_decimal b = k ^ 2) ∧
  (∃ k : ℕ, base_b_to_decimal 5 = k ^ 2) := by
  sorry

end NUMINAMATH_CALUDE_smallest_base_perfect_square_l2630_263098


namespace NUMINAMATH_CALUDE_circle_and_line_intersection_l2630_263008

-- Define the parabola
def parabola (x y : ℝ) : Prop := y = x^2 - 4*x + 3

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = 5

-- Define the line
def line (x y : ℝ) : Prop := 2*x - y + 2 = 0

-- Theorem statement
theorem circle_and_line_intersection :
  -- Circle C passes through intersection points of parabola and coordinate axes
  (∃ x₁ x₂ x₃ y₃ : ℝ, 
    parabola x₁ 0 ∧ parabola x₂ 0 ∧ parabola 0 y₃ ∧
    circle_C x₁ 0 ∧ circle_C x₂ 0 ∧ circle_C 0 y₃) →
  -- Line intersects circle C at two points
  (∃ A B : ℝ × ℝ, 
    line A.1 A.2 ∧ line B.1 B.2 ∧
    circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧ A ≠ B) →
  -- Distance between intersection points is 6√5/5
  ∃ A B : ℝ × ℝ, line A.1 A.2 ∧ line B.1 B.2 ∧
    circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = (6 * Real.sqrt 5) / 5 :=
by
  sorry


end NUMINAMATH_CALUDE_circle_and_line_intersection_l2630_263008


namespace NUMINAMATH_CALUDE_baseball_cards_cost_l2630_263025

theorem baseball_cards_cost (football_pack_cost : ℝ) (pokemon_pack_cost : ℝ) (total_spent : ℝ)
  (h1 : football_pack_cost = 2.73)
  (h2 : pokemon_pack_cost = 4.01)
  (h3 : total_spent = 18.42) :
  total_spent - (2 * football_pack_cost + pokemon_pack_cost) = 8.95 := by
  sorry

end NUMINAMATH_CALUDE_baseball_cards_cost_l2630_263025
