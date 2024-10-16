import Mathlib

namespace NUMINAMATH_CALUDE_square_root_problem_l249_24979

theorem square_root_problem (m : ℝ) (h1 : m > 0) (h2 : ∃ a : ℝ, (3 - a)^2 = m ∧ (2*a + 1)^2 = m) : m = 49 := by
  sorry

end NUMINAMATH_CALUDE_square_root_problem_l249_24979


namespace NUMINAMATH_CALUDE_original_cube_volume_l249_24924

theorem original_cube_volume (s : ℝ) : 
  (2 * s)^3 = 2744 → s^3 = 343 := by
  sorry

end NUMINAMATH_CALUDE_original_cube_volume_l249_24924


namespace NUMINAMATH_CALUDE_seventh_row_cans_l249_24985

/-- Represents a triangular display of cans -/
structure CanDisplay where
  rows : Nat
  first_row_cans : Nat
  increment : Nat

/-- Calculate the number of cans in a specific row -/
def cans_in_row (d : CanDisplay) (row : Nat) : Nat :=
  d.first_row_cans + d.increment * (row - 1)

/-- Calculate the total number of cans in the display -/
def total_cans (d : CanDisplay) : Nat :=
  (d.rows * (2 * d.first_row_cans + (d.rows - 1) * d.increment)) / 2

/-- The main theorem -/
theorem seventh_row_cans (d : CanDisplay) :
  d.rows = 10 ∧ d.increment = 3 ∧ total_cans d < 150 →
  cans_in_row d 7 = 19 := by
  sorry

#eval cans_in_row { rows := 10, first_row_cans := 1, increment := 3 } 7

end NUMINAMATH_CALUDE_seventh_row_cans_l249_24985


namespace NUMINAMATH_CALUDE_third_boy_age_l249_24998

theorem third_boy_age (total_age : ℕ) (age_two_boys : ℕ) (num_boys : ℕ) :
  total_age = 29 →
  age_two_boys = 9 →
  num_boys = 3 →
  ∃ (third_boy_age : ℕ), third_boy_age = total_age - 2 * age_two_boys :=
by
  sorry

end NUMINAMATH_CALUDE_third_boy_age_l249_24998


namespace NUMINAMATH_CALUDE_euler_polynomial_consecutive_composites_l249_24976

theorem euler_polynomial_consecutive_composites :
  ∃ k : ℤ, ∀ j ∈ Finset.range 40,
    ∃ d : ℤ, d ∣ ((k + j)^2 + (k + j) + 41) ∧ d ≠ 1 ∧ d ≠ ((k + j)^2 + (k + j) + 41) := by
  sorry

end NUMINAMATH_CALUDE_euler_polynomial_consecutive_composites_l249_24976


namespace NUMINAMATH_CALUDE_cube_properties_l249_24902

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a line in 3D space -/
structure Line3D where
  point : Point3D
  direction : Point3D

/-- Represents a cube in 3D space -/
structure Cube where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  A₁ : Point3D
  B₁ : Point3D
  C₁ : Point3D
  D₁ : Point3D
  edgeLength : ℝ

/-- Returns true if two lines are skew -/
def areSkewLines (l1 l2 : Line3D) : Prop := sorry

/-- Returns true if a line is perpendicular to two other lines -/
def isPerpendicularToLines (l : Line3D) (l1 l2 : Line3D) : Prop := sorry

/-- Calculates the distance between two skew lines -/
def distanceBetweenSkewLines (l1 l2 : Line3D) : ℝ := sorry

theorem cube_properties (cube : Cube) :
  let AA₁ : Line3D := { point := cube.A, direction := { x := 0, y := 0, z := 1 } }
  let BC : Line3D := { point := cube.B, direction := { x := 1, y := 0, z := 0 } }
  let AB : Line3D := { point := cube.A, direction := { x := 1, y := 0, z := 0 } }
  areSkewLines AA₁ BC ∧
  isPerpendicularToLines AB AA₁ BC ∧
  distanceBetweenSkewLines AA₁ BC = cube.edgeLength := by
  sorry

end NUMINAMATH_CALUDE_cube_properties_l249_24902


namespace NUMINAMATH_CALUDE_quadratic_function_negative_on_interval_l249_24994

-- Define the function f
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_function_negative_on_interval
  (a b c : ℝ)
  (h1 : a > b)
  (h2 : b > c)
  (h3 : a + b + c = 0) :
  ∀ x ∈ Set.Ioo 0 1, f a b c x < 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_negative_on_interval_l249_24994


namespace NUMINAMATH_CALUDE_circle_symmetry_about_y_axis_l249_24928

/-- Given two circles in the xy-plane, this theorem states that they are symmetric about the y-axis
    if and only if their equations are identical when x is replaced by -x in one of them. -/
theorem circle_symmetry_about_y_axis (a b : ℝ) :
  (∀ x y, x^2 + y^2 + a*x = 0 ↔ (-x)^2 + y^2 + b*(-x) = 0) ↔
  a = -b :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_about_y_axis_l249_24928


namespace NUMINAMATH_CALUDE_circle_differences_l249_24926

theorem circle_differences (n : ℕ) (a : ℕ → ℝ) 
  (h : ∀ i, |a i - a ((i + 1) % n)| ≥ 2 * |a i - a ((i + 2) % n)|) :
  ∀ i, |a i - a ((i + 3) % n)| ≥ |a i - a ((i + 2) % n)| :=
by sorry

end NUMINAMATH_CALUDE_circle_differences_l249_24926


namespace NUMINAMATH_CALUDE_complex_modulus_range_l249_24946

theorem complex_modulus_range (a : ℝ) : 
  (∀ θ : ℝ, Complex.abs ((a + Real.cos θ) + (2 * a - Real.sin θ) * Complex.I) ≤ 2) ↔ 
  a ∈ Set.Icc (-1/2) (1/2) := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_range_l249_24946


namespace NUMINAMATH_CALUDE_min_positive_temperatures_l249_24968

theorem min_positive_temperatures (n : ℕ) (pos_products neg_products : ℕ) : 
  n = 12 → 
  pos_products = 78 → 
  neg_products = 54 → 
  pos_products + neg_products = n * (n - 1) →
  ∃ y : ℕ, y ≥ 3 ∧ y * (y - 1) + (n - y) * (n - 1 - y) = pos_products ∧
  ∀ z : ℕ, z < 3 → z * (z - 1) + (n - z) * (n - 1 - z) ≠ pos_products :=
by sorry

end NUMINAMATH_CALUDE_min_positive_temperatures_l249_24968


namespace NUMINAMATH_CALUDE_train_speed_problem_l249_24917

/-- Proves that the speed of the faster train is 31.25 km/hr given the problem conditions. -/
theorem train_speed_problem (v : ℝ) (h1 : v > 25) : 
  v = 31.25 ∧ 
  ∃ (t : ℝ), t > 0 ∧ 
    v * t + 25 * t = 630 ∧ 
    v * t = 25 * t + 70 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_problem_l249_24917


namespace NUMINAMATH_CALUDE_sum_of_even_indexed_coefficients_l249_24953

theorem sum_of_even_indexed_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) :
  (∀ x : ℝ, x^10 = a + a₁*(1-x) + a₂*(1-x)^2 + a₃*(1-x)^3 + a₄*(1-x)^4 + 
            a₅*(1-x)^5 + a₆*(1-x)^6 + a₇*(1-x)^7 + a₈*(1-x)^8 + a₉*(1-x)^9 + a₁₀*(1-x)^10) →
  a + a₂ + a₄ + a₆ + a₈ + a₁₀ = 2^9 := by
sorry

end NUMINAMATH_CALUDE_sum_of_even_indexed_coefficients_l249_24953


namespace NUMINAMATH_CALUDE_maggie_bouncy_balls_l249_24960

/-- The number of bouncy balls Maggie kept -/
def total_bouncy_balls : ℝ :=
  let yellow_packs : ℝ := 8.0
  let green_packs_given : ℝ := 4.0
  let green_packs_bought : ℝ := 4.0
  let balls_per_pack : ℝ := 10.0
  yellow_packs * balls_per_pack + (green_packs_bought - green_packs_given) * balls_per_pack

/-- Theorem stating that Maggie kept 80.0 bouncy balls -/
theorem maggie_bouncy_balls : total_bouncy_balls = 80.0 := by
  sorry

end NUMINAMATH_CALUDE_maggie_bouncy_balls_l249_24960


namespace NUMINAMATH_CALUDE_anna_truck_meet_once_l249_24997

/-- Represents the movement of Anna and the garbage truck on a path with trash pails. -/
structure TrashCollection where
  annaSpeed : ℝ
  truckSpeed : ℝ
  pailDistance : ℝ
  truckStopTime : ℝ

/-- Calculates the number of times Anna and the truck meet. -/
def meetingCount (tc : TrashCollection) : ℕ :=
  sorry

/-- The theorem states that Anna and the truck meet exactly once under the given conditions. -/
theorem anna_truck_meet_once :
  ∀ (tc : TrashCollection),
    tc.annaSpeed = 5 ∧
    tc.truckSpeed = 15 ∧
    tc.pailDistance = 300 ∧
    tc.truckStopTime = 40 →
    meetingCount tc = 1 :=
  sorry

end NUMINAMATH_CALUDE_anna_truck_meet_once_l249_24997


namespace NUMINAMATH_CALUDE_q_div_p_eq_fifty_l249_24990

/-- The number of cards in the box -/
def total_cards : ℕ := 30

/-- The number of different numbers on the cards -/
def num_types : ℕ := 6

/-- The number of cards for each number -/
def cards_per_num : ℕ := 5

/-- The number of cards drawn -/
def drawn_cards : ℕ := 4

/-- The probability of drawing four cards with the same number -/
def p : ℚ := (num_types * (cards_per_num.choose drawn_cards)) / (total_cards.choose drawn_cards)

/-- The probability of drawing two pairs of cards with different numbers -/
def q : ℚ := (num_types.choose 2 * (cards_per_num.choose 2)^2) / (total_cards.choose drawn_cards)

/-- The theorem stating that the ratio of q to p is 50 -/
theorem q_div_p_eq_fifty : q / p = 50 := by sorry

end NUMINAMATH_CALUDE_q_div_p_eq_fifty_l249_24990


namespace NUMINAMATH_CALUDE_equation_solution_l249_24927

theorem equation_solution : 
  ∀ x : ℝ, x ≠ 1 ∧ x ≠ (1/2) → 
  (((x^2 - 5*x + 4) / (x - 1)) + ((2*x^2 + 7*x - 4) / (2*x - 1)) = 4) → 
  x = 2 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l249_24927


namespace NUMINAMATH_CALUDE_integer_solutions_yk_eq_x2_plus_x_l249_24955

theorem integer_solutions_yk_eq_x2_plus_x (k : ℕ) (hk : k > 1) :
  ∀ x y : ℤ, y^k = x^2 + x ↔ (x = 0 ∧ y = 0) ∨ (x = -1 ∧ y = 0) := by
  sorry

end NUMINAMATH_CALUDE_integer_solutions_yk_eq_x2_plus_x_l249_24955


namespace NUMINAMATH_CALUDE_largest_package_size_l249_24938

theorem largest_package_size (alex_markers jordan_markers : ℕ) 
  (h_alex : alex_markers = 56) (h_jordan : jordan_markers = 42) :
  Nat.gcd alex_markers jordan_markers = 14 := by
  sorry

end NUMINAMATH_CALUDE_largest_package_size_l249_24938


namespace NUMINAMATH_CALUDE_min_lcm_ac_l249_24975

theorem min_lcm_ac (a b c : ℕ+) (h1 : Nat.lcm a b = 18) (h2 : Nat.lcm b c = 28) :
  ∃ (a' c' : ℕ+), Nat.lcm a' c' = 126 ∧ 
    (∀ (x y : ℕ+), Nat.lcm x b = 18 → Nat.lcm b y = 28 → Nat.lcm a' c' ≤ Nat.lcm x y) :=
by sorry

end NUMINAMATH_CALUDE_min_lcm_ac_l249_24975


namespace NUMINAMATH_CALUDE_sum_of_roots_absolute_value_equation_l249_24970

theorem sum_of_roots_absolute_value_equation : 
  ∃ (r₁ r₂ r₃ : ℝ), 
    (∀ x : ℝ, (|x + 3| - |x - 1| = x + 1) ↔ (x = r₁ ∨ x = r₂ ∨ x = r₃)) ∧ 
    r₁ + r₂ + r₃ = -3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_absolute_value_equation_l249_24970


namespace NUMINAMATH_CALUDE_polygon_and_calendar_problem_l249_24982

theorem polygon_and_calendar_problem :
  ∀ (n k : ℕ),
  -- Regular polygon with interior angles of 160°
  (180 - 160 : ℝ) * n = 360 →
  -- The n-th day of May is Friday
  n % 7 = 5 →
  -- The k-th day of May is Tuesday
  k % 7 = 2 →
  -- 20 < k < 26
  20 < k ∧ k < 26 →
  -- Prove n = 18 and k = 22
  n = 18 ∧ k = 22 :=
by sorry

end NUMINAMATH_CALUDE_polygon_and_calendar_problem_l249_24982


namespace NUMINAMATH_CALUDE_train_crossing_time_l249_24905

/-- Proves that a train of given length, passing a platform of given length in a given time,
    will take a specific time to cross a tree. -/
theorem train_crossing_time
  (train_length : ℝ)
  (platform_length : ℝ)
  (platform_crossing_time : ℝ)
  (h1 : train_length = 1200)
  (h2 : platform_length = 700)
  (h3 : platform_crossing_time = 190)
  : (train_length / ((train_length + platform_length) / platform_crossing_time)) = 120 :=
by sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l249_24905


namespace NUMINAMATH_CALUDE_markup_calculation_l249_24947

theorem markup_calculation (purchase_price overhead_percentage net_profit : ℝ) 
  (h1 : purchase_price = 48)
  (h2 : overhead_percentage = 0.35)
  (h3 : net_profit = 18) :
  purchase_price + purchase_price * overhead_percentage + net_profit - purchase_price = 34.80 := by
  sorry

end NUMINAMATH_CALUDE_markup_calculation_l249_24947


namespace NUMINAMATH_CALUDE_specific_grid_area_l249_24935

/-- A rectangular grid formed by perpendicular lines -/
structure RectangularGrid where
  num_boundary_lines : ℕ
  perimeter : ℝ
  is_rectangular : Bool
  has_perpendicular_lines : Bool

/-- The area of a rectangular grid -/
def grid_area (grid : RectangularGrid) : ℝ :=
  sorry

/-- Theorem stating the area of the specific rectangular grid -/
theorem specific_grid_area :
  ∀ (grid : RectangularGrid),
    grid.num_boundary_lines = 36 ∧
    grid.perimeter = 72 ∧
    grid.is_rectangular = true ∧
    grid.has_perpendicular_lines = true →
    grid_area grid = 84 :=
  sorry

end NUMINAMATH_CALUDE_specific_grid_area_l249_24935


namespace NUMINAMATH_CALUDE_function_periodicity_l249_24914

/-- Given a > 0 and a function f satisfying the specified condition, 
    prove that f is periodic with period 2a -/
theorem function_periodicity 
  (a : ℝ) 
  (ha : a > 0) 
  (f : ℝ → ℝ) 
  (hf : ∀ x : ℝ, f (x + a) = 1/2 + Real.sqrt (f x - (f x)^2)) : 
  ∃ b : ℝ, b > 0 ∧ ∀ x : ℝ, f (x + b) = f x :=
sorry

end NUMINAMATH_CALUDE_function_periodicity_l249_24914


namespace NUMINAMATH_CALUDE_ab_equals_zero_l249_24949

theorem ab_equals_zero (a b : ℝ) 
  (h1 : (4 : ℝ) ^ a = 256 ^ (b + 1))
  (h2 : (27 : ℝ) ^ b = 3 ^ (a - 2)) : 
  a * b = 0 := by sorry

end NUMINAMATH_CALUDE_ab_equals_zero_l249_24949


namespace NUMINAMATH_CALUDE_difference_of_squares_fraction_l249_24900

theorem difference_of_squares_fraction : (235^2 - 221^2) / 14 = 456 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_fraction_l249_24900


namespace NUMINAMATH_CALUDE_intersection_point_on_line_and_x_axis_l249_24913

/-- The line equation 5y - 3x = 15 -/
def line_equation (x y : ℝ) : Prop := 5 * y - 3 * x = 15

/-- A point is on the x-axis if its y-coordinate is 0 -/
def on_x_axis (x y : ℝ) : Prop := y = 0

/-- The intersection point of the line and the x-axis -/
def intersection_point : ℝ × ℝ := (-5, 0)

theorem intersection_point_on_line_and_x_axis :
  line_equation intersection_point.1 intersection_point.2 ∧
  on_x_axis intersection_point.1 intersection_point.2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_on_line_and_x_axis_l249_24913


namespace NUMINAMATH_CALUDE_lucy_mother_age_relation_l249_24969

/-- Lucy's age in 2010 -/
def lucy_age_2010 : ℕ := 10

/-- Lucy's mother's age in 2010 -/
def mother_age_2010 : ℕ := 5 * lucy_age_2010

/-- The year when Lucy's mother's age will be twice Lucy's age -/
def target_year : ℕ := 2040

/-- The number of years from 2010 to the target year -/
def years_passed : ℕ := target_year - 2010

theorem lucy_mother_age_relation :
  mother_age_2010 + years_passed = 2 * (lucy_age_2010 + years_passed) :=
by sorry

end NUMINAMATH_CALUDE_lucy_mother_age_relation_l249_24969


namespace NUMINAMATH_CALUDE_rectangle_longer_side_l249_24965

/-- Given a square with two pairs of identical isosceles triangles cut off, leaving a rectangle,
    if the total area cut off is 250 m² and one side of the rectangle is 1.5 times the length of the other,
    then the length of the longer side of the rectangle is 7.5√5 meters. -/
theorem rectangle_longer_side (x y : ℝ) : 
  x^2 + y^2 = 250 →  -- Total area cut off
  x = y →            -- Isosceles triangles condition
  1.5 * y = max x (1.5 * y) →  -- One side is 1.5 times the other
  max x (1.5 * y) = 7.5 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_longer_side_l249_24965


namespace NUMINAMATH_CALUDE_special_function_properties_l249_24964

/-- An increasing function f defined on (-1, +∞) with the property f(xy) = f(x) + f(y) -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y, x > -1 ∧ y > -1 → f (x * y) = f x + f y) ∧
  (∀ x y, x > -1 ∧ y > -1 ∧ x < y → f x < f y)

theorem special_function_properties
    (f : ℝ → ℝ)
    (hf : SpecialFunction f)
    (h3 : f 3 = 1) :
  (f 9 = 2) ∧
  (∀ a, a > -1 → (f a > f (a - 1) + 2 ↔ 0 < a ∧ a < 9/8)) :=
by sorry

end NUMINAMATH_CALUDE_special_function_properties_l249_24964


namespace NUMINAMATH_CALUDE_first_place_beats_joe_by_two_l249_24911

/-- Calculates the total points for a team based on their match results -/
def calculate_points (wins : ℕ) (ties : ℕ) : ℕ :=
  3 * wins + ties

/-- Represents the scoring system and match results for the soccer tournament -/
structure TournamentResults where
  win_points : ℕ := 3
  tie_points : ℕ := 1
  joe_wins : ℕ := 1
  joe_ties : ℕ := 3
  first_place_wins : ℕ := 2
  first_place_ties : ℕ := 2

/-- Theorem stating that the first-place team beat Joe's team by 2 points -/
theorem first_place_beats_joe_by_two (results : TournamentResults) :
  calculate_points results.first_place_wins results.first_place_ties -
  calculate_points results.joe_wins results.joe_ties = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_first_place_beats_joe_by_two_l249_24911


namespace NUMINAMATH_CALUDE_sum_abcd_equals_negative_twenty_six_thirds_l249_24962

theorem sum_abcd_equals_negative_twenty_six_thirds 
  (a b c d : ℝ) 
  (h : a + 2 = b + 3 ∧ b + 3 = c + 4 ∧ c + 4 = d + 5 ∧ d + 5 = a + b + c + d + 10) : 
  a + b + c + d = -26/3 := by
  sorry

end NUMINAMATH_CALUDE_sum_abcd_equals_negative_twenty_six_thirds_l249_24962


namespace NUMINAMATH_CALUDE_fraction_equality_l249_24939

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (4 * x + y) / (x - 4 * y) = -3) : 
  (x + 4 * y) / (4 * x - y) = 39 / 37 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l249_24939


namespace NUMINAMATH_CALUDE_angle_A_is_60_degrees_triangle_is_equilateral_l249_24943

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the properties of the triangle
def isValidTriangle (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.A + t.B + t.C = Real.pi

-- Define the given condition
def satisfiesCondition (t : Triangle) : Prop :=
  t.b^2 + t.c^2 = t.a^2 + t.b * t.c

-- Define the law of cosines
def lawOfCosines (t : Triangle) : Prop :=
  t.a^2 = t.b^2 + t.c^2 - 2 * t.b * t.c * Real.cos t.A

-- Define the law of sines
def lawOfSines (t : Triangle) : Prop :=
  t.a / Real.sin t.A = t.b / Real.sin t.B ∧
  t.b / Real.sin t.B = t.c / Real.sin t.C

-- Define the equilateral triangle condition
def isEquilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

-- Theorem 1: Prove that angle A is 60 degrees
theorem angle_A_is_60_degrees (t : Triangle) 
  (h1 : isValidTriangle t) 
  (h2 : satisfiesCondition t) 
  (h3 : lawOfCosines t) : 
  t.A = Real.pi / 3 :=
sorry

-- Theorem 2: Prove that the triangle is equilateral
theorem triangle_is_equilateral (t : Triangle)
  (h1 : isValidTriangle t)
  (h2 : satisfiesCondition t)
  (h3 : lawOfSines t)
  (h4 : Real.sin t.B * Real.sin t.C = Real.sin t.A ^ 2) :
  isEquilateral t :=
sorry

end NUMINAMATH_CALUDE_angle_A_is_60_degrees_triangle_is_equilateral_l249_24943


namespace NUMINAMATH_CALUDE_divisibility_circle_l249_24920

/-- Given seven natural numbers in a circle where each adjacent pair has a divisibility relation,
    there exists a non-adjacent pair with the same property. -/
theorem divisibility_circle (a : Fin 7 → ℕ) 
  (h : ∀ i : Fin 7, (a i ∣ a (i + 1)) ∨ (a (i + 1) ∣ a i)) :
  ∃ i j : Fin 7, i ≠ j ∧ (j ≠ i + 1) ∧ (j ≠ i - 1) ∧ ((a i ∣ a j) ∨ (a j ∣ a i)) :=
sorry

end NUMINAMATH_CALUDE_divisibility_circle_l249_24920


namespace NUMINAMATH_CALUDE_roots_quadratic_sum_l249_24932

theorem roots_quadratic_sum (a b : ℝ) : 
  (a^2 + 3*a - 4 = 0) → 
  (b^2 + 3*b - 4 = 0) → 
  (a^2 + 4*a + b - 3 = -2) := by
  sorry

end NUMINAMATH_CALUDE_roots_quadratic_sum_l249_24932


namespace NUMINAMATH_CALUDE_complex_purely_imaginary_solution_l249_24981

-- Define a complex number to be purely imaginary if its real part is zero
def is_purely_imaginary (z : ℂ) : Prop := z.re = 0

theorem complex_purely_imaginary_solution (z : ℂ) 
  (h1 : is_purely_imaginary z) 
  (h2 : is_purely_imaginary ((z - 3)^2 + 5*I)) : 
  z = 3*I ∨ z = -3*I := by
  sorry

end NUMINAMATH_CALUDE_complex_purely_imaginary_solution_l249_24981


namespace NUMINAMATH_CALUDE_reflection_theorem_l249_24971

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Reflect a point with respect to another point -/
def reflect (p : Point3D) (center : Point3D) : Point3D :=
  { x := 2 * center.x - p.x,
    y := 2 * center.y - p.y,
    z := 2 * center.z - p.z }

/-- Perform a sequence of reflections -/
def reflectSequence (p : Point3D) (centers : List Point3D) : Point3D :=
  centers.foldl reflect p

theorem reflection_theorem (A O₁ O₂ O₃ : Point3D) :
  reflectSequence (reflectSequence A [O₁, O₂, O₃]) [O₁, O₂, O₃] = A := by
  sorry


end NUMINAMATH_CALUDE_reflection_theorem_l249_24971


namespace NUMINAMATH_CALUDE_total_students_l249_24988

/-- Represents the number of students in each grade --/
def Students := Fin 8 → ℕ

/-- The total number of students in grades I-IV is 130 --/
def sum_I_to_IV (s : Students) : Prop :=
  s 0 + s 1 + s 2 + s 3 = 130

/-- Grade V has 7 more students than grade II --/
def grade_V_condition (s : Students) : Prop :=
  s 4 = s 1 + 7

/-- Grade VI has 5 fewer students than grade I --/
def grade_VI_condition (s : Students) : Prop :=
  s 5 = s 0 - 5

/-- Grade VII has 10 more students than grade IV --/
def grade_VII_condition (s : Students) : Prop :=
  s 6 = s 3 + 10

/-- Grade VIII has 4 fewer students than grade I --/
def grade_VIII_condition (s : Students) : Prop :=
  s 7 = s 0 - 4

/-- The theorem stating that the total number of students is 268 --/
theorem total_students (s : Students)
  (h1 : sum_I_to_IV s)
  (h2 : grade_V_condition s)
  (h3 : grade_VI_condition s)
  (h4 : grade_VII_condition s)
  (h5 : grade_VIII_condition s) :
  s 0 + s 1 + s 2 + s 3 + s 4 + s 5 + s 6 + s 7 = 268 := by
  sorry

end NUMINAMATH_CALUDE_total_students_l249_24988


namespace NUMINAMATH_CALUDE_zinc_copper_mixture_weight_l249_24923

/-- Proves that the weight of a zinc-copper mixture is 74 kg given the specified conditions -/
theorem zinc_copper_mixture_weight :
  ∀ (zinc copper total : ℝ),
  zinc = 33.3 →
  zinc / copper = 9 / 11 →
  total = zinc + copper →
  total = 74 := by
sorry

end NUMINAMATH_CALUDE_zinc_copper_mixture_weight_l249_24923


namespace NUMINAMATH_CALUDE_inequality_proof_l249_24993

def f (a x : ℝ) : ℝ := |x - a|

theorem inequality_proof (a s t : ℝ) (h1 : ∀ x, f a x ≤ 4 ↔ -1 ≤ x ∧ x ≤ 7) 
    (h2 : s > 0) (h3 : t > 0) (h4 : 2*s + t = a) : 
    1/s + 8/t ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l249_24993


namespace NUMINAMATH_CALUDE_negative_sixty_four_to_four_thirds_l249_24956

theorem negative_sixty_four_to_four_thirds : (-64 : ℝ) ^ (4/3) = 256 := by
  sorry

end NUMINAMATH_CALUDE_negative_sixty_four_to_four_thirds_l249_24956


namespace NUMINAMATH_CALUDE_interior_nodes_line_property_l249_24918

/-- A point with integer coordinates -/
structure Node where
  x : ℤ
  y : ℤ

/-- A triangle with vertices at nodes -/
structure Triangle where
  a : Node
  b : Node
  c : Node

/-- Checks if a node is inside a triangle -/
def Node.isInside (n : Node) (t : Triangle) : Prop :=
  sorry

/-- Checks if a line through two nodes passes through a vertex of a triangle -/
def Line.passesThroughVertex (p q : Node) (t : Triangle) : Prop :=
  sorry

/-- Checks if a line through two nodes is parallel to a side of a triangle -/
def Line.isParallelToSide (p q : Node) (t : Triangle) : Prop :=
  sorry

/-- Main theorem -/
theorem interior_nodes_line_property (t : Triangle) (p q : Node) :
  p.isInside t ∧ q.isInside t →
  (∀ r : Node, r.isInside t → r = p ∨ r = q) →
  Line.passesThroughVertex p q t ∨ Line.isParallelToSide p q t :=
sorry

end NUMINAMATH_CALUDE_interior_nodes_line_property_l249_24918


namespace NUMINAMATH_CALUDE_balloon_height_per_ounce_l249_24941

/-- Calculates the height increase per ounce of helium for a balloon flight --/
theorem balloon_height_per_ounce 
  (total_money : ℚ)
  (sheet_cost : ℚ)
  (rope_cost : ℚ)
  (propane_cost : ℚ)
  (helium_price_per_ounce : ℚ)
  (max_height : ℚ)
  (h1 : total_money = 200)
  (h2 : sheet_cost = 42)
  (h3 : rope_cost = 18)
  (h4 : propane_cost = 14)
  (h5 : helium_price_per_ounce = 3/2)
  (h6 : max_height = 9492) :
  (max_height / ((total_money - (sheet_cost + rope_cost + propane_cost)) / helium_price_per_ounce)) = 113 := by
  sorry

end NUMINAMATH_CALUDE_balloon_height_per_ounce_l249_24941


namespace NUMINAMATH_CALUDE_work_hours_calculation_l249_24987

/-- Calculates the number of hours spent at work given the total hours in a day and the percentage of time spent working. -/
def hours_at_work (total_hours : ℝ) (work_percentage : ℝ) : ℝ :=
  total_hours * work_percentage

/-- Proves that given a 16-hour day where 50% is spent at work, the number of hours spent at work is 8. -/
theorem work_hours_calculation (total_hours : ℝ) (work_percentage : ℝ) 
    (h1 : total_hours = 16) 
    (h2 : work_percentage = 0.5) : 
  hours_at_work total_hours work_percentage = 8 := by
  sorry

#eval hours_at_work 16 0.5

end NUMINAMATH_CALUDE_work_hours_calculation_l249_24987


namespace NUMINAMATH_CALUDE_modular_arithmetic_problem_l249_24972

theorem modular_arithmetic_problem :
  ∃ (a b : ℤ), 
    (7 * a) % 80 = 1 ∧ 
    (13 * b) % 80 = 1 ∧ 
    ((3 * a + 9 * b) % 80) = 2 := by
  sorry

end NUMINAMATH_CALUDE_modular_arithmetic_problem_l249_24972


namespace NUMINAMATH_CALUDE_even_polynomial_iff_product_with_negation_l249_24958

-- Define the complex polynomials
variable (P Q : ℂ → ℂ)

-- Define what it means for a function to be a polynomial
def IsPolynomial (f : ℂ → ℂ) : Prop := sorry

-- Define what it means for a function to be even
def IsEven (f : ℂ → ℂ) : Prop := ∀ z, f (-z) = f z

-- State the theorem
theorem even_polynomial_iff_product_with_negation :
  (IsPolynomial P ∧ IsEven P) ↔ 
  (∃ Q, IsPolynomial Q ∧ ∀ z, P z = Q z * Q (-z)) := by sorry

end NUMINAMATH_CALUDE_even_polynomial_iff_product_with_negation_l249_24958


namespace NUMINAMATH_CALUDE_monic_quadratic_with_complex_root_l249_24945

/-- A monic quadratic polynomial with real coefficients -/
def MonicQuadratic (a b : ℝ) : ℂ → ℂ := fun x ↦ x^2 + a*x + b

/-- The given complex number that is a root of the polynomial -/
def givenRoot : ℂ := 2 - 3*Complex.I

theorem monic_quadratic_with_complex_root :
  ∃! (a b : ℝ), (MonicQuadratic a b givenRoot = 0) ∧ (a = -4 ∧ b = 13) := by
  sorry

end NUMINAMATH_CALUDE_monic_quadratic_with_complex_root_l249_24945


namespace NUMINAMATH_CALUDE_min_distance_exp_curve_to_line_l249_24966

/-- The minimum distance from a point on the curve y = e^x to the line y = x is √2/2 -/
theorem min_distance_exp_curve_to_line : 
  ∃ (d : ℝ), d = Real.sqrt 2 / 2 ∧ 
  ∀ (x y : ℝ), y = Real.exp x → 
  d ≤ Real.sqrt ((x - y)^2 + (y - x)^2) / 2 :=
sorry

end NUMINAMATH_CALUDE_min_distance_exp_curve_to_line_l249_24966


namespace NUMINAMATH_CALUDE_ara_height_is_60_l249_24986

/-- Represents the heights of Shea and Ara --/
structure Heights where
  initial : ℝ  -- Initial height of both Shea and Ara
  shea_current : ℝ  -- Shea's current height
  shea_growth_rate : ℝ  -- Shea's growth rate as a decimal
  ara_growth_difference : ℝ  -- Difference between Shea and Ara's growth in inches

/-- Calculates Ara's current height given the initial conditions --/
def ara_current_height (h : Heights) : ℝ :=
  h.initial + (h.shea_current - h.initial) - h.ara_growth_difference

/-- Theorem stating that Ara's current height is 60 inches --/
theorem ara_height_is_60 (h : Heights)
  (h_shea_current : h.shea_current = 65)
  (h_shea_growth : h.shea_growth_rate = 0.3)
  (h_ara_diff : h.ara_growth_difference = 5) :
  ara_current_height h = 60 := by
  sorry

#eval ara_current_height { initial := 50, shea_current := 65, shea_growth_rate := 0.3, ara_growth_difference := 5 }

end NUMINAMATH_CALUDE_ara_height_is_60_l249_24986


namespace NUMINAMATH_CALUDE_square_has_perpendicular_diagonals_but_parallelogram_not_l249_24944

-- Define a square
structure Square :=
  (side : ℝ)
  (side_positive : side > 0)

-- Define a parallelogram
structure Parallelogram :=
  (base : ℝ)
  (height : ℝ)
  (base_positive : base > 0)
  (height_positive : height > 0)

-- Define the property of perpendicular diagonals
def has_perpendicular_diagonals (S : Type) : Prop :=
  ∀ s : S, ∃ d₁ d₂ : ℝ × ℝ, d₁.1 * d₂.1 + d₁.2 * d₂.2 = 0

-- Theorem statement
theorem square_has_perpendicular_diagonals_but_parallelogram_not :
  (has_perpendicular_diagonals Square) ∧ ¬(has_perpendicular_diagonals Parallelogram) :=
sorry

end NUMINAMATH_CALUDE_square_has_perpendicular_diagonals_but_parallelogram_not_l249_24944


namespace NUMINAMATH_CALUDE_stephanie_store_visits_l249_24930

/-- The number of times Stephanie went to the store last month -/
def store_visits : ℕ := 16 / 2

/-- The number of oranges Stephanie buys each time she goes to the store -/
def oranges_per_visit : ℕ := 2

/-- The total number of oranges Stephanie bought last month -/
def total_oranges : ℕ := 16

theorem stephanie_store_visits : store_visits = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_stephanie_store_visits_l249_24930


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l249_24919

/-- Given a triangle ABC with the following properties:
  - b = √2
  - c = 3
  - B + C = 3A
  Prove the following:
  1. a = √5
  2. sin(B + 3π/4) = √10/10
-/
theorem triangle_abc_properties (A B C : Real) (a b c : Real) :
  b = Real.sqrt 2 →
  c = 3 →
  B + C = 3 * A →
  a = Real.sqrt 5 ∧ Real.sin (B + 3 * Real.pi / 4) = Real.sqrt 10 / 10 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l249_24919


namespace NUMINAMATH_CALUDE_intersection_M_N_l249_24978

def M : Set ℝ := { x | -1 ≤ x ∧ x < 3 }

def N : Set ℝ := {-1, 0, 1, 2, 3}

theorem intersection_M_N : M ∩ N = {-1, 0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l249_24978


namespace NUMINAMATH_CALUDE_oxygen_weight_in_compound_l249_24992

/-- The atomic weight of hydrogen -/
def hydrogen_weight : ℝ := 1

/-- The atomic weight of chlorine -/
def chlorine_weight : ℝ := 35.5

/-- The total molecular weight of the compound -/
def total_weight : ℝ := 68

/-- The number of hydrogen atoms in the compound -/
def hydrogen_count : ℕ := 1

/-- The number of chlorine atoms in the compound -/
def chlorine_count : ℕ := 1

/-- The number of oxygen atoms in the compound -/
def oxygen_count : ℕ := 2

/-- Theorem: The atomic weight of oxygen in the given compound is 15.75 -/
theorem oxygen_weight_in_compound : 
  ∃ (oxygen_weight : ℝ), 
    (hydrogen_count : ℝ) * hydrogen_weight + 
    (chlorine_count : ℝ) * chlorine_weight + 
    (oxygen_count : ℝ) * oxygen_weight = total_weight ∧ 
    oxygen_weight = 15.75 := by sorry

end NUMINAMATH_CALUDE_oxygen_weight_in_compound_l249_24992


namespace NUMINAMATH_CALUDE_distance_is_correct_l249_24991

def point : ℝ × ℝ × ℝ := (2, 3, 4)
def line_point : ℝ × ℝ × ℝ := (4, 5, 6)
def line_direction : ℝ × ℝ × ℝ := (4, 1, -1)

def distance_to_line (p : ℝ × ℝ × ℝ) (l_point : ℝ × ℝ × ℝ) (l_direction : ℝ × ℝ × ℝ) : ℝ :=
  sorry

theorem distance_is_correct : 
  distance_to_line point line_point line_direction = (9 * Real.sqrt 2) / 4 :=
sorry

end NUMINAMATH_CALUDE_distance_is_correct_l249_24991


namespace NUMINAMATH_CALUDE_exactly_one_red_two_green_mutually_exclusive_not_opposition_l249_24922

/-- Represents a ball color -/
inductive Color
  | Red
  | Green

/-- Represents the content of a pocket -/
structure Pocket where
  red_balls : ℕ
  green_balls : ℕ
  red_more_than_two : red_balls > 2
  green_more_than_two : green_balls > 2

/-- Represents the outcome of drawing two balls -/
inductive TwoBalldraw
  | TwoRed
  | TwoGreen
  | OneRedOneGreen

/-- Defines what it means for two events to be mutually exclusive -/
def mutually_exclusive (e1 e2 : Set TwoBalldraw) : Prop :=
  e1 ∩ e2 = ∅

/-- Defines what it means for two events to be in opposition -/
def in_opposition (e1 e2 : Set TwoBalldraw) : Prop :=
  e1 ∪ e2 = Set.univ

/-- The theorem to be proved -/
theorem exactly_one_red_two_green_mutually_exclusive_not_opposition (p : Pocket) :
  let exactly_one_red := {TwoBalldraw.OneRedOneGreen}
  let exactly_two_green := {TwoBalldraw.TwoGreen}
  mutually_exclusive exactly_one_red exactly_two_green ∧
  ¬in_opposition exactly_one_red exactly_two_green := by
  sorry

end NUMINAMATH_CALUDE_exactly_one_red_two_green_mutually_exclusive_not_opposition_l249_24922


namespace NUMINAMATH_CALUDE_ocean_area_scientific_notation_l249_24907

-- Define the original number
def original_number : ℝ := 2997000

-- Define the scientific notation components
def scientific_base : ℝ := 2.997
def scientific_exponent : ℤ := 6

-- Theorem statement
theorem ocean_area_scientific_notation :
  original_number = scientific_base * (10 : ℝ) ^ scientific_exponent :=
by sorry

end NUMINAMATH_CALUDE_ocean_area_scientific_notation_l249_24907


namespace NUMINAMATH_CALUDE_geometric_series_ratio_l249_24973

theorem geometric_series_ratio (a r : ℝ) (h : r ≠ 1) :
  (a * r^4 / (1 - r)) = (1 / 64) * (a / (1 - r)) →
  r = 1/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_ratio_l249_24973


namespace NUMINAMATH_CALUDE_different_color_probability_l249_24934

def total_balls : ℕ := 5
def red_balls : ℕ := 3
def yellow_balls : ℕ := 2
def drawn_balls : ℕ := 2

theorem different_color_probability :
  (Nat.choose red_balls 1 * Nat.choose yellow_balls 1) / Nat.choose total_balls drawn_balls = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_different_color_probability_l249_24934


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l249_24931

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 1 = 2 →
  a 2 + a 3 = 13 →
  a 4 + a 5 + a 6 = 42 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l249_24931


namespace NUMINAMATH_CALUDE_complex_equation_solution_l249_24957

theorem complex_equation_solution (z : ℂ) :
  z * (2 - Complex.I) = 10 + 5 * Complex.I → z = 3 + 4 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l249_24957


namespace NUMINAMATH_CALUDE_company_managers_count_l249_24974

/-- Proves that the number of managers is 15 given the conditions in the problem --/
theorem company_managers_count :
  ∀ (num_managers : ℕ) (num_associates : ℕ) (avg_salary_managers : ℚ) (avg_salary_associates : ℚ) (avg_salary_company : ℚ),
  num_associates = 75 →
  avg_salary_managers = 90000 →
  avg_salary_associates = 30000 →
  avg_salary_company = 40000 →
  (num_managers * avg_salary_managers + num_associates * avg_salary_associates) / (num_managers + num_associates : ℚ) = avg_salary_company →
  num_managers = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_company_managers_count_l249_24974


namespace NUMINAMATH_CALUDE_peculiar_animal_farm_l249_24984

theorem peculiar_animal_farm (cats dogs : ℕ) : 
  dogs = cats + 180 →
  (cats + (dogs / 5 : ℚ)) / (cats + dogs : ℚ) = 32 / 100 →
  cats + dogs = 240 := by
sorry

end NUMINAMATH_CALUDE_peculiar_animal_farm_l249_24984


namespace NUMINAMATH_CALUDE_power_of_two_ge_square_l249_24908

theorem power_of_two_ge_square (n : ℕ) (h : n ≥ 4) : 2^n ≥ n^2 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_ge_square_l249_24908


namespace NUMINAMATH_CALUDE_no_solution_iff_m_eq_neg_four_l249_24921

theorem no_solution_iff_m_eq_neg_four :
  ∀ m : ℝ, (∀ x : ℝ, (x ≠ 2 ∧ x ≠ -2) → 
    ((x - 2) / (x + 2) - m * x / (x^2 - 4) ≠ 1)) ↔ m = -4 :=
by sorry

end NUMINAMATH_CALUDE_no_solution_iff_m_eq_neg_four_l249_24921


namespace NUMINAMATH_CALUDE_smallest_x_y_sum_l249_24940

/-- The smallest positive integer x such that 720x is a square number -/
def x : ℕ+ := sorry

/-- The smallest positive integer y such that 720y is a fourth power -/
def y : ℕ+ := sorry

theorem smallest_x_y_sum : 
  (∀ x' : ℕ+, x' < x → ¬∃ n : ℕ+, 720 * x' = n^2) ∧
  (∀ y' : ℕ+, y' < y → ¬∃ n : ℕ+, 720 * y' = n^4) ∧
  (∃ n : ℕ+, 720 * x = n^2) ∧
  (∃ n : ℕ+, 720 * y = n^4) ∧
  (x : ℕ) + (y : ℕ) = 1130 := by sorry

end NUMINAMATH_CALUDE_smallest_x_y_sum_l249_24940


namespace NUMINAMATH_CALUDE_initial_books_on_shelf_l249_24961

theorem initial_books_on_shelf (books_taken : ℕ) (books_left : ℕ) : 
  books_taken = 10 → books_left = 28 → books_taken + books_left = 38 := by
  sorry

end NUMINAMATH_CALUDE_initial_books_on_shelf_l249_24961


namespace NUMINAMATH_CALUDE_fraction_subtraction_l249_24903

theorem fraction_subtraction : 
  (1 + 4 + 7) / (2 + 5 + 8) - (2 + 5 + 8) / (1 + 4 + 7) = -9 / 20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l249_24903


namespace NUMINAMATH_CALUDE_correlation_relationships_l249_24906

/-- A relationship between two variables -/
inductive Relationship
| AgeWealth
| CurveCoordinates
| AppleProductionClimate
| TreeDiameterHeight
| StudentSchool

/-- Predicate to determine if a relationship involves correlation -/
def involves_correlation (r : Relationship) : Prop :=
  match r with
  | Relationship.AgeWealth => true
  | Relationship.CurveCoordinates => false
  | Relationship.AppleProductionClimate => true
  | Relationship.TreeDiameterHeight => true
  | Relationship.StudentSchool => false

/-- The set of all relationships -/
def all_relationships : Set Relationship :=
  {Relationship.AgeWealth, Relationship.CurveCoordinates, Relationship.AppleProductionClimate,
   Relationship.TreeDiameterHeight, Relationship.StudentSchool}

/-- The theorem stating which relationships involve correlation -/
theorem correlation_relationships :
  {r ∈ all_relationships | involves_correlation r} =
  {Relationship.AgeWealth, Relationship.AppleProductionClimate, Relationship.TreeDiameterHeight} :=
by sorry

end NUMINAMATH_CALUDE_correlation_relationships_l249_24906


namespace NUMINAMATH_CALUDE_hostel_problem_l249_24950

/-- Calculates the number of men who left a hostel given the initial conditions and the new duration of provisions. -/
def men_who_left (initial_men : ℕ) (initial_days : ℕ) (new_days : ℕ) : ℕ :=
  initial_men - (initial_men * initial_days) / new_days

/-- Proves that 50 men left the hostel under the given conditions. -/
theorem hostel_problem : men_who_left 250 48 60 = 50 := by
  sorry

end NUMINAMATH_CALUDE_hostel_problem_l249_24950


namespace NUMINAMATH_CALUDE_wedge_volume_l249_24910

/-- The volume of a wedge cut from a cylindrical log --/
theorem wedge_volume (d h : ℝ) (α : ℝ) : 
  d = 10 → α = 60 → (d / 2)^2 * h * π / 6 = 250 * π / 6 := by sorry

end NUMINAMATH_CALUDE_wedge_volume_l249_24910


namespace NUMINAMATH_CALUDE_sin_two_alpha_value_l249_24995

theorem sin_two_alpha_value (α : ℝ) (h : Real.sin α - Real.cos α = 4/3) : 
  Real.sin (2 * α) = -7/9 := by
  sorry

end NUMINAMATH_CALUDE_sin_two_alpha_value_l249_24995


namespace NUMINAMATH_CALUDE_pen_calculation_l249_24983

theorem pen_calculation (x y z : ℕ) (hx : x = 5) (hy : y = 20) (hz : z = 19) :
  2 * (x + y) - z = 31 := by
  sorry

end NUMINAMATH_CALUDE_pen_calculation_l249_24983


namespace NUMINAMATH_CALUDE_largest_divisor_of_odd_product_l249_24916

theorem largest_divisor_of_odd_product (n : ℕ) (h : Even n) (h_pos : 0 < n) :
  ∃ (k : ℕ), k = 105 ∧ 
  (∀ (m : ℕ), m ∣ ((n+1)*(n+3)*(n+5)*(n+7)*(n+9)*(n+11)*(n+13)) → m ≤ k) ∧
  k ∣ ((n+1)*(n+3)*(n+5)*(n+7)*(n+9)*(n+11)*(n+13)) :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_odd_product_l249_24916


namespace NUMINAMATH_CALUDE_unique_solution_is_zero_l249_24933

theorem unique_solution_is_zero :
  ∃! y : ℝ, y = 3 * (1 / y * (-y)) + 3 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_is_zero_l249_24933


namespace NUMINAMATH_CALUDE_better_to_answer_B_first_l249_24951

-- Define the probabilities and point values
def prob_correct_A : Real := 0.8
def prob_correct_B : Real := 0.6
def points_A : ℕ := 20
def points_B : ℕ := 80

-- Define the expected score functions
def expected_score_A_first : Real :=
  0 * (1 - prob_correct_A) +
  points_A * (prob_correct_A * (1 - prob_correct_B)) +
  (points_A + points_B) * (prob_correct_A * prob_correct_B)

def expected_score_B_first : Real :=
  0 * (1 - prob_correct_B) +
  points_B * (prob_correct_B * (1 - prob_correct_A)) +
  (points_A + points_B) * (prob_correct_B * prob_correct_A)

-- Theorem statement
theorem better_to_answer_B_first :
  expected_score_B_first > expected_score_A_first := by
  sorry


end NUMINAMATH_CALUDE_better_to_answer_B_first_l249_24951


namespace NUMINAMATH_CALUDE_sum_nine_equals_27_l249_24937

/-- An arithmetic sequence with special properties -/
structure ArithmeticSequence where
  a : ℕ+ → ℝ
  is_arithmetic : ∀ n m : ℕ+, a (n + 1) - a n = a (m + 1) - a m
  on_line : ∀ n : ℕ+, ∃ k b : ℝ, a n = k * n + b ∧ 3 = k * 5 + b

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ+) : ℝ :=
  (n : ℝ) * seq.a n

/-- The main theorem -/
theorem sum_nine_equals_27 (seq : ArithmeticSequence) : sum_n seq 9 = 27 := by
  sorry

end NUMINAMATH_CALUDE_sum_nine_equals_27_l249_24937


namespace NUMINAMATH_CALUDE_common_point_theorem_l249_24967

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point lies on a line -/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y = l.c

/-- Represents a geometric progression -/
def IsGeometricProgression (a c b : ℝ) : Prop :=
  ∃ r : ℝ, c = a * r ∧ b = a * r^2

theorem common_point_theorem :
  ∀ (l : Line), 
    IsGeometricProgression l.a l.c l.b →
    l.contains 0 0 :=
by sorry

end NUMINAMATH_CALUDE_common_point_theorem_l249_24967


namespace NUMINAMATH_CALUDE_collinear_points_right_triangle_l249_24963

/-- Given that point O is the origin, this function defines vectors OA, OB, and OC -/
def vectors (m : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) :=
  ((3, -4), (6, -3), (5 - m, -3 - m))

/-- Theorem stating that if A, B, and C are collinear, then m = 1/2 -/
theorem collinear_points (m : ℝ) :
  let (oa, ob, oc) := vectors m
  (∃ (k : ℝ), (ob.1 - oa.1, ob.2 - oa.2) = k • (oc.1 - oa.1, oc.2 - oa.2)) →
  m = 1/2 := by sorry

/-- Theorem stating that if ABC is a right triangle with A as the right angle, then m = 7/4 -/
theorem right_triangle (m : ℝ) :
  let (oa, ob, oc) := vectors m
  let ab := (ob.1 - oa.1, ob.2 - oa.2)
  let ac := (oc.1 - oa.1, oc.2 - oa.2)
  (ab.1 * ac.1 + ab.2 * ac.2 = 0) →
  m = 7/4 := by sorry

end NUMINAMATH_CALUDE_collinear_points_right_triangle_l249_24963


namespace NUMINAMATH_CALUDE_store_a_highest_capacity_l249_24912

/-- Represents a store with its CD storage capacity -/
structure Store where
  shelves : ℕ
  racks_per_shelf : ℕ
  cds_per_rack : ℕ

/-- Calculates the total CD capacity of a store -/
def total_capacity (s : Store) : ℕ :=
  s.shelves * s.racks_per_shelf * s.cds_per_rack

/-- The three stores with their respective capacities -/
def store_a : Store := ⟨5, 6, 9⟩
def store_b : Store := ⟨8, 4, 7⟩
def store_c : Store := ⟨10, 3, 8⟩

/-- Theorem stating that Store A has the highest total CD capacity -/
theorem store_a_highest_capacity :
  total_capacity store_a > total_capacity store_b ∧
  total_capacity store_a > total_capacity store_c :=
by
  sorry


end NUMINAMATH_CALUDE_store_a_highest_capacity_l249_24912


namespace NUMINAMATH_CALUDE_gasoline_price_increase_l249_24909

theorem gasoline_price_increase (original_price original_quantity : ℝ) 
  (h1 : original_price > 0) (h2 : original_quantity > 0) : 
  ∃ (price_increase : ℝ),
    (original_price * (1 + price_increase / 100) * (original_quantity * 0.95) = 
     original_price * original_quantity * 1.14) ∧ 
    (price_increase = 20) := by
  sorry

end NUMINAMATH_CALUDE_gasoline_price_increase_l249_24909


namespace NUMINAMATH_CALUDE_largest_two_three_digit_multiples_sum_l249_24954

theorem largest_two_three_digit_multiples_sum : ∃ (a b : ℕ), 
  (a > 0 ∧ a < 100 ∧ a % 5 = 0 ∧ ∀ x : ℕ, x > 0 ∧ x < 100 ∧ x % 5 = 0 → x ≤ a) ∧
  (b > 0 ∧ b < 1000 ∧ b % 7 = 0 ∧ ∀ y : ℕ, y > 0 ∧ y < 1000 ∧ y % 7 = 0 → y ≤ b) ∧
  a + b = 1089 := by
sorry

end NUMINAMATH_CALUDE_largest_two_three_digit_multiples_sum_l249_24954


namespace NUMINAMATH_CALUDE_number_of_elements_in_set_l249_24942

theorem number_of_elements_in_set (initial_avg : ℚ) (incorrect_num : ℚ) (correct_num : ℚ) (correct_avg : ℚ) (n : ℕ) : 
  initial_avg = 18 →
  incorrect_num = 26 →
  correct_num = 66 →
  correct_avg = 22 →
  n * initial_avg + (correct_num - incorrect_num) = n * correct_avg →
  n = 10 := by
sorry

end NUMINAMATH_CALUDE_number_of_elements_in_set_l249_24942


namespace NUMINAMATH_CALUDE_prime_pairs_sum_50_l249_24952

/-- A function that returns the number of unordered pairs of prime numbers that sum to a given natural number. -/
def count_prime_pairs (n : ℕ) : ℕ :=
  (Finset.filter (fun p => Nat.Prime p ∧ Nat.Prime (n - p) ∧ 2 * p ≤ n) (Finset.range (n / 2 + 1))).card

/-- The theorem stating that there are exactly 4 unordered pairs of prime numbers that sum to 50. -/
theorem prime_pairs_sum_50 : count_prime_pairs 50 = 4 := by
  sorry

end NUMINAMATH_CALUDE_prime_pairs_sum_50_l249_24952


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l249_24936

/-- An arithmetic sequence with the given properties -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_arithmetic : ArithmeticSequence a)
  (h_sum : a 1 + a 9 = 10)
  (h_second : a 2 = -1) :
  ∃ d : ℝ, d = 2 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l249_24936


namespace NUMINAMATH_CALUDE_f_max_min_values_l249_24999

-- Define the function
def f (x : ℝ) : ℝ := 3 * x - x^3

-- State the theorem
theorem f_max_min_values :
  (∃ x : ℝ, f x = 2 ∧ ∀ y : ℝ, f y ≤ 2) ∧
  (∃ x : ℝ, f x = -2 ∧ ∀ y : ℝ, f y ≥ -2) := by
  sorry

end NUMINAMATH_CALUDE_f_max_min_values_l249_24999


namespace NUMINAMATH_CALUDE_gcd_difference_theorem_l249_24989

theorem gcd_difference_theorem : Nat.gcd 5610 210 - 10 = 20 := by
  sorry

end NUMINAMATH_CALUDE_gcd_difference_theorem_l249_24989


namespace NUMINAMATH_CALUDE_arccos_of_one_eq_zero_l249_24901

theorem arccos_of_one_eq_zero : Real.arccos 1 = 0 := by sorry

end NUMINAMATH_CALUDE_arccos_of_one_eq_zero_l249_24901


namespace NUMINAMATH_CALUDE_translated_segment_endpoint_l249_24959

/-- Given a segment AB with endpoints A(-4, -1) and B(1, 1), when translated to segment A'B' where A' has coordinates (-2, 2), prove that the coordinates of B' are (3, 4). -/
theorem translated_segment_endpoint (A B A' B' : ℝ × ℝ) : 
  A = (-4, -1) → 
  B = (1, 1) → 
  A' = (-2, 2) → 
  (A'.1 - A.1 = B'.1 - B.1 ∧ A'.2 - A.2 = B'.2 - B.2) → 
  B' = (3, 4) := by
  sorry

end NUMINAMATH_CALUDE_translated_segment_endpoint_l249_24959


namespace NUMINAMATH_CALUDE_problem_statement_l249_24980

-- Define the propositions
def p : Prop := ∃ x : ℝ, Real.exp x = 0.1

def l₁ (a : ℝ) : ℝ → ℝ → Prop := λ x y ↦ x - a * y = 0

def l₂ (a : ℝ) : ℝ → ℝ → Prop := λ x y ↦ 2 * x + a * y - 1 = 0

def q : Prop := ∀ a : ℝ, (∀ x y : ℝ, l₁ a x y ∧ l₂ a x y → a = Real.sqrt 2)

theorem problem_statement : p ∧ ¬q := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l249_24980


namespace NUMINAMATH_CALUDE_sine_sum_problem_l249_24929

theorem sine_sum_problem (α : ℝ) (h : Real.sin (π / 3 + α) + Real.sin α = (4 * Real.sqrt 3) / 5) :
  Real.sin (α + 7 * π / 6) = -4 / 5 := by sorry

end NUMINAMATH_CALUDE_sine_sum_problem_l249_24929


namespace NUMINAMATH_CALUDE_milk_ratio_l249_24915

/-- Given a cafeteria that sells two types of milk (regular and chocolate),
    this theorem proves the ratio of chocolate to regular milk cartons sold. -/
theorem milk_ratio (total : ℕ) (regular : ℕ) 
    (h1 : total = 24) 
    (h2 : regular = 3) : 
    (total - regular) / regular = 7 := by
  sorry

#check milk_ratio

end NUMINAMATH_CALUDE_milk_ratio_l249_24915


namespace NUMINAMATH_CALUDE_keith_total_spent_l249_24925

-- Define the amounts spent on each item
def speakers_cost : ℚ := 136.01
def cd_player_cost : ℚ := 139.38
def tires_cost : ℚ := 112.46

-- Define the total amount spent
def total_spent : ℚ := speakers_cost + cd_player_cost + tires_cost

-- Theorem to prove
theorem keith_total_spent :
  total_spent = 387.85 :=
by sorry

end NUMINAMATH_CALUDE_keith_total_spent_l249_24925


namespace NUMINAMATH_CALUDE_printer_task_time_l249_24904

/-- Given two printers A and B, this theorem proves the time taken to complete a task together -/
theorem printer_task_time (pages : ℕ) (time_A : ℕ) (rate_diff : ℕ) : 
  pages = 480 → 
  time_A = 60 → 
  rate_diff = 4 → 
  (pages : ℚ) / ((pages : ℚ) / time_A + ((pages : ℚ) / time_A + rate_diff)) = 24 := by
  sorry

#check printer_task_time

end NUMINAMATH_CALUDE_printer_task_time_l249_24904


namespace NUMINAMATH_CALUDE_two_x_equals_two_l249_24948

theorem two_x_equals_two (h : 1 = x) : 2 * x = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_x_equals_two_l249_24948


namespace NUMINAMATH_CALUDE_hyperbola_condition_l249_24996

/-- Represents the equation of a conic section --/
def is_hyperbola (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (2 + m) + y^2 / (m + 1) = 1 ∧ 
  ((2 + m > 0 ∧ m + 1 < 0) ∨ (2 + m < 0 ∧ m + 1 > 0))

/-- The main theorem stating the condition for the equation to represent a hyperbola --/
theorem hyperbola_condition (m : ℝ) : 
  is_hyperbola m ↔ -2 < m ∧ m < -1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_condition_l249_24996


namespace NUMINAMATH_CALUDE_parabola_point_ordering_l249_24977

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = -2 * (x - 1)^2 + 3

/-- Point A lies on the parabola -/
def point_A (y₁ : ℝ) : Prop := parabola (-3) y₁

/-- Point B lies on the parabola -/
def point_B (y₂ : ℝ) : Prop := parabola 2 y₂

/-- Theorem stating the ordering of y₁, y₂, and 3 -/
theorem parabola_point_ordering (y₁ y₂ : ℝ) 
  (hA : point_A y₁) (hB : point_B y₂) : y₁ < y₂ ∧ y₂ < 3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_ordering_l249_24977
