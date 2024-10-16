import Mathlib

namespace NUMINAMATH_CALUDE_extended_equilateral_area_ratio_l3339_333907

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Calculates the area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- Checks if a triangle is equilateral -/
def isEquilateral (t : Triangle) : Prop := sorry

/-- Extends a line segment by a factor -/
def extendSegment (A B : Point) (factor : ℝ) : Point := sorry

theorem extended_equilateral_area_ratio 
  (P Q R : Point) 
  (t : Triangle)
  (h_equilateral : isEquilateral t)
  (h_t : t = Triangle.mk P Q R)
  (Q' : Point)
  (h_Q' : Q' = extendSegment P Q 3)
  (R' : Point)
  (h_R' : R' = extendSegment Q R 3)
  (P' : Point)
  (h_P' : P' = extendSegment R P 3)
  (t_extended : Triangle)
  (h_t_extended : t_extended = Triangle.mk P' Q' R') :
  triangleArea t_extended / triangleArea t = 9 := by sorry

end NUMINAMATH_CALUDE_extended_equilateral_area_ratio_l3339_333907


namespace NUMINAMATH_CALUDE_smallest_enclosing_sphere_radius_l3339_333926

/-- The radius of the smallest sphere centered at the origin that contains
    ten spheres of radius 2 positioned at the corners of a cube with side length 4 -/
theorem smallest_enclosing_sphere_radius (r : ℝ) (s : ℝ) : r = 2 ∧ s = 4 →
  (2 * Real.sqrt 3 + 2 : ℝ) = 
    (s * Real.sqrt 3 / 2 + r : ℝ) := by sorry

end NUMINAMATH_CALUDE_smallest_enclosing_sphere_radius_l3339_333926


namespace NUMINAMATH_CALUDE_race_outcomes_count_l3339_333935

-- Define the number of participants
def num_participants : ℕ := 7

-- Define a function to calculate the number of permutations
def permutations (n : ℕ) (r : ℕ) : ℕ :=
  Nat.factorial n / Nat.factorial (n - r)

-- Define a function to calculate the number of combinations
def combinations (n : ℕ) (r : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Theorem statement
theorem race_outcomes_count :
  (3 * combinations (num_participants - 1) 2 * permutations 2 2) = 90 := by
  sorry

end NUMINAMATH_CALUDE_race_outcomes_count_l3339_333935


namespace NUMINAMATH_CALUDE_cone_sphere_ratio_l3339_333986

/-- Proof that the ratio of cone height to base radius is 4/3 when cone volume is 1/3 of sphere volume --/
theorem cone_sphere_ratio (r h : ℝ) (hr : r > 0) :
  (1 / 3) * ((4 / 3) * Real.pi * r^3) = (1 / 3) * Real.pi * r^2 * h → h / r = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cone_sphere_ratio_l3339_333986


namespace NUMINAMATH_CALUDE_savings_ratio_is_three_fifths_l3339_333927

/-- Represents the savings scenario of Thomas and Joseph -/
structure SavingsScenario where
  thomas_monthly_savings : ℕ
  total_savings : ℕ
  saving_period_months : ℕ

/-- Calculates the ratio of Joseph's monthly savings to Thomas's monthly savings -/
def savings_ratio (scenario : SavingsScenario) : Rat :=
  let thomas_total := scenario.thomas_monthly_savings * scenario.saving_period_months
  let joseph_total := scenario.total_savings - thomas_total
  let joseph_monthly := joseph_total / scenario.saving_period_months
  joseph_monthly / scenario.thomas_monthly_savings

/-- The main theorem stating the ratio of Joseph's to Thomas's monthly savings -/
theorem savings_ratio_is_three_fifths (scenario : SavingsScenario)
  (h1 : scenario.thomas_monthly_savings = 40)
  (h2 : scenario.total_savings = 4608)
  (h3 : scenario.saving_period_months = 72) :
  savings_ratio scenario = 3 / 5 := by
  sorry

#eval savings_ratio { thomas_monthly_savings := 40, total_savings := 4608, saving_period_months := 72 }

end NUMINAMATH_CALUDE_savings_ratio_is_three_fifths_l3339_333927


namespace NUMINAMATH_CALUDE_limit_polynomial_at_2_l3339_333976

theorem limit_polynomial_at_2 :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x - 2| < δ → |3*x^2 - 2*x + 7 - 15| < ε := by
  sorry

end NUMINAMATH_CALUDE_limit_polynomial_at_2_l3339_333976


namespace NUMINAMATH_CALUDE_minutes_worked_yesterday_l3339_333951

/-- The number of shirts made by the machine yesterday -/
def shirts_made_yesterday : ℕ := 9

/-- The number of shirts the machine can make per minute -/
def shirts_per_minute : ℕ := 3

/-- Theorem: The number of minutes the machine worked yesterday is 3 -/
theorem minutes_worked_yesterday : 
  shirts_made_yesterday / shirts_per_minute = 3 := by
  sorry

end NUMINAMATH_CALUDE_minutes_worked_yesterday_l3339_333951


namespace NUMINAMATH_CALUDE_area_ratio_AMK_ABC_l3339_333989

/-- Triangle ABC with points M on AB and K on AC -/
structure TriangleWithPoints where
  /-- The area of triangle ABC -/
  area_ABC : ℝ
  /-- The ratio of AM to MB -/
  ratio_AM_MB : ℝ × ℝ
  /-- The ratio of AK to KC -/
  ratio_AK_KC : ℝ × ℝ

/-- The theorem stating the area ratio of triangle AMK to triangle ABC -/
theorem area_ratio_AMK_ABC (t : TriangleWithPoints) (h1 : t.area_ABC = 50) 
  (h2 : t.ratio_AM_MB = (1, 5)) (h3 : t.ratio_AK_KC = (3, 2)) : 
  (∃ (area_AMK : ℝ), area_AMK / t.area_ABC = 1 / 10) :=
sorry

end NUMINAMATH_CALUDE_area_ratio_AMK_ABC_l3339_333989


namespace NUMINAMATH_CALUDE_fraction_equality_l3339_333952

theorem fraction_equality (a b : ℚ) (h1 : 2 * a = 3 * b) (h2 : b ≠ 0) : a / b = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3339_333952


namespace NUMINAMATH_CALUDE_cube_surface_area_l3339_333972

theorem cube_surface_area (volume : ℝ) (side : ℝ) (surface_area : ℝ) : 
  volume = 729 → 
  volume = side^3 → 
  surface_area = 6 * side^2 → 
  surface_area = 486 :=
by sorry

end NUMINAMATH_CALUDE_cube_surface_area_l3339_333972


namespace NUMINAMATH_CALUDE_cookies_per_bag_l3339_333973

theorem cookies_per_bag (chocolate_chip : ℕ) (oatmeal : ℕ) (bags : ℕ) :
  chocolate_chip = 13 →
  oatmeal = 41 →
  bags = 6 →
  (chocolate_chip + oatmeal) / bags = 9 :=
by sorry

end NUMINAMATH_CALUDE_cookies_per_bag_l3339_333973


namespace NUMINAMATH_CALUDE_probability_less_than_5_is_17_18_l3339_333991

/-- A square in the 2D plane --/
structure Square where
  bottomLeft : ℝ × ℝ
  sideLength : ℝ

/-- The probability that a randomly chosen point (x,y) in the given square satisfies x + y < 5 --/
def probabilityLessThan5 (s : Square) : ℝ :=
  sorry

/-- The specific square with vertices (0,0), (0,3), (3,3), and (3,0) --/
def specificSquare : Square :=
  { bottomLeft := (0, 0), sideLength := 3 }

theorem probability_less_than_5_is_17_18 :
  probabilityLessThan5 specificSquare = 17 / 18 :=
sorry

end NUMINAMATH_CALUDE_probability_less_than_5_is_17_18_l3339_333991


namespace NUMINAMATH_CALUDE_sphere_radius_ratio_l3339_333947

theorem sphere_radius_ratio (V_L V_S r_L r_S : ℝ) : 
  V_L = 432 * Real.pi ∧ 
  V_S = 0.08 * V_L ∧ 
  V_L = (4/3) * Real.pi * r_L^3 ∧ 
  V_S = (4/3) * Real.pi * r_S^3 →
  r_S / r_L = 1/2 := by
sorry

end NUMINAMATH_CALUDE_sphere_radius_ratio_l3339_333947


namespace NUMINAMATH_CALUDE_percentage_increase_proof_l3339_333985

def old_cost : ℝ := 150
def new_cost : ℝ := 195

theorem percentage_increase_proof :
  (new_cost - old_cost) / old_cost * 100 = 30 := by sorry

end NUMINAMATH_CALUDE_percentage_increase_proof_l3339_333985


namespace NUMINAMATH_CALUDE_sixteen_solutions_l3339_333990

/-- The function f(x) = x^2 - 3x --/
def f (x : ℝ) : ℝ := x^2 - 3*x

/-- The fourth composition of f --/
def f_4 (x : ℝ) : ℝ := f (f (f (f x)))

/-- There are exactly 16 distinct real solutions to f(f(f(f(c)))) = 6 --/
theorem sixteen_solutions : ∃! (s : Finset ℝ), s.card = 16 ∧ ∀ c, c ∈ s ↔ f_4 c = 6 := by sorry

end NUMINAMATH_CALUDE_sixteen_solutions_l3339_333990


namespace NUMINAMATH_CALUDE_intersection_parallel_perpendicular_l3339_333917

-- Define the lines
def line1 (x y : ℝ) : Prop := 2 * x + y - 5 = 0
def line2 (x y : ℝ) : Prop := x - 2 * y = 0
def line_l (x y : ℝ) : Prop := 3 * x - y - 7 = 0

-- Define point P as the intersection of line1 and line2
def point_p : ℝ × ℝ := (2, 1)

-- Define the parallel and perpendicular lines
def parallel_line (x y : ℝ) : Prop := 3 * x - y - 5 = 0
def perpendicular_line (x y : ℝ) : Prop := x + 3 * y - 5 = 0

theorem intersection_parallel_perpendicular :
  (∃ (x y : ℝ), line1 x y ∧ line2 x y ∧ (x, y) = point_p) →
  (parallel_line (point_p.1) (point_p.2)) ∧
  (∀ (x y : ℝ), parallel_line x y → (y - point_p.2) = 3 * (x - point_p.1)) ∧
  (perpendicular_line (point_p.1) (point_p.2)) ∧
  (∀ (x y : ℝ), perpendicular_line x y → (y - point_p.2) = -(1/3) * (x - point_p.1)) :=
by sorry


end NUMINAMATH_CALUDE_intersection_parallel_perpendicular_l3339_333917


namespace NUMINAMATH_CALUDE_inequality_proof_l3339_333909

theorem inequality_proof (x y z : ℝ) : x^4 + y^4 + z^2 + 1 ≥ 2*x*(x*y^2 - x + z + 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3339_333909


namespace NUMINAMATH_CALUDE_reflected_quadrilateral_area_l3339_333919

/-- Represents a convex quadrilateral -/
structure ConvexQuadrilateral where
  area : ℝ
  is_convex : Bool

/-- Represents a point inside a convex quadrilateral -/
structure PointInQuadrilateral where
  quad : ConvexQuadrilateral
  is_inside : Bool

/-- Represents the quadrilateral formed by reflecting a point with respect to the midpoints of a quadrilateral's sides -/
def ReflectedQuadrilateral (p : PointInQuadrilateral) : ConvexQuadrilateral :=
  sorry

/-- The theorem stating that the area of the reflected quadrilateral is twice the area of the original quadrilateral -/
theorem reflected_quadrilateral_area 
  (q : ConvexQuadrilateral) 
  (p : PointInQuadrilateral) 
  (h1 : p.quad = q) 
  (h2 : p.is_inside = true) 
  (h3 : q.is_convex = true) :
  (ReflectedQuadrilateral p).area = 2 * q.area :=
sorry

end NUMINAMATH_CALUDE_reflected_quadrilateral_area_l3339_333919


namespace NUMINAMATH_CALUDE_david_boxes_l3339_333999

/-- Given a total number of dogs and the number of dogs per box, 
    calculate the number of boxes needed. -/
def calculate_boxes (total_dogs : ℕ) (dogs_per_box : ℕ) : ℕ :=
  total_dogs / dogs_per_box

/-- Theorem stating that given 28 total dogs and 4 dogs per box, 
    the number of boxes is 7. -/
theorem david_boxes : calculate_boxes 28 4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_david_boxes_l3339_333999


namespace NUMINAMATH_CALUDE_base_b_square_l3339_333949

theorem base_b_square (b : ℕ) (hb : b > 1) : 
  (∃ n : ℕ, b^2 + 4*b + 4 = n^2) ↔ b > 4 := by
  sorry

end NUMINAMATH_CALUDE_base_b_square_l3339_333949


namespace NUMINAMATH_CALUDE_simplify_expression_l3339_333912

theorem simplify_expression :
  ∀ x y : ℝ, (5 - 6*x) - (9 + 5*x - 2*y) = -4 - 11*x + 2*y :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l3339_333912


namespace NUMINAMATH_CALUDE_train_length_calculation_l3339_333978

/-- Calculates the length of a train given its speed, the speed of a person moving in the opposite direction, and the time it takes for the train to pass the person. -/
theorem train_length_calculation (train_speed : ℝ) (person_speed : ℝ) (passing_time : ℝ) :
  train_speed = 60 →
  person_speed = 6 →
  passing_time = 13.090909090909092 →
  ∃ (train_length : ℝ), abs (train_length - 240) < 0.01 :=
by sorry

end NUMINAMATH_CALUDE_train_length_calculation_l3339_333978


namespace NUMINAMATH_CALUDE_average_of_remaining_numbers_l3339_333900

theorem average_of_remaining_numbers
  (total_count : Nat)
  (subset_count : Nat)
  (total_average : ℝ)
  (subset_average : ℝ)
  (h_total_count : total_count = 15)
  (h_subset_count : subset_count = 9)
  (h_total_average : total_average = 30.5)
  (h_subset_average : subset_average = 17.75) :
  let remaining_count := total_count - subset_count
  let remaining_sum := total_count * total_average - subset_count * subset_average
  remaining_sum / remaining_count = 49.625 := by
sorry


end NUMINAMATH_CALUDE_average_of_remaining_numbers_l3339_333900


namespace NUMINAMATH_CALUDE_two_digit_sum_with_reverse_is_square_l3339_333911

def reverse (n : Nat) : Nat :=
  10 * (n % 10) + (n / 10)

def is_two_digit (n : Nat) : Prop :=
  10 ≤ n ∧ n ≤ 99

def is_perfect_square (n : Nat) : Prop :=
  ∃ k : Nat, n = k * k

theorem two_digit_sum_with_reverse_is_square :
  {n : Nat | is_two_digit n ∧ is_perfect_square (n + reverse n)} =
  {29, 38, 47, 56, 65, 74, 83, 92} := by
sorry

end NUMINAMATH_CALUDE_two_digit_sum_with_reverse_is_square_l3339_333911


namespace NUMINAMATH_CALUDE_horizontal_line_inclination_l3339_333950

def line (x y : ℝ) : Prop := y + 3 = 0

def angle_of_inclination (f : ℝ → ℝ → Prop) : ℝ := sorry

theorem horizontal_line_inclination :
  angle_of_inclination line = 0 := by sorry

end NUMINAMATH_CALUDE_horizontal_line_inclination_l3339_333950


namespace NUMINAMATH_CALUDE_weight_of_six_meter_rod_l3339_333969

/-- Given a uniform steel rod with specified properties, this theorem proves
    the weight of a 6 m piece of the same rod. -/
theorem weight_of_six_meter_rod (r : ℝ) (ρ : ℝ) : 
  let rod_length : ℝ := 11.25
  let rod_weight : ℝ := 42.75
  let piece_length : ℝ := 6
  let rod_volume := π * r^2 * rod_length
  let piece_volume := π * r^2 * piece_length
  let density := rod_weight / rod_volume
  piece_volume * density = 22.8 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_six_meter_rod_l3339_333969


namespace NUMINAMATH_CALUDE_factorial_sum_equality_l3339_333964

theorem factorial_sum_equality : 7 * Nat.factorial 7 + 5 * Nat.factorial 5 - 3 * Nat.factorial 3 + 2 * Nat.factorial 2 = 35866 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_equality_l3339_333964


namespace NUMINAMATH_CALUDE_doughnut_profit_l3339_333979

/-- Calculate the profit from selling doughnuts -/
theorem doughnut_profit 
  (expenses : ℕ) 
  (num_doughnuts : ℕ) 
  (price_per_doughnut : ℕ) 
  (h1 : expenses = 53)
  (h2 : num_doughnuts = 25)
  (h3 : price_per_doughnut = 3) : 
  num_doughnuts * price_per_doughnut - expenses = 22 := by
  sorry

end NUMINAMATH_CALUDE_doughnut_profit_l3339_333979


namespace NUMINAMATH_CALUDE_stream_speed_l3339_333958

/-- Given a boat traveling downstream, prove the speed of the stream. -/
theorem stream_speed 
  (boat_speed : ℝ) 
  (downstream_distance : ℝ) 
  (downstream_time : ℝ) 
  (h1 : boat_speed = 25) 
  (h2 : downstream_distance = 90) 
  (h3 : downstream_time = 3) : 
  ∃ stream_speed : ℝ, 
    downstream_distance = (boat_speed + stream_speed) * downstream_time ∧ 
    stream_speed = 5 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l3339_333958


namespace NUMINAMATH_CALUDE_annas_remaining_money_l3339_333939

/-- Given Anna's initial amount and her purchases, calculate the amount left --/
theorem annas_remaining_money (initial_amount : ℚ) 
  (gum_price choc_price cane_price : ℚ)
  (gum_quantity choc_quantity cane_quantity : ℕ) : 
  initial_amount = 10 →
  gum_price = 1 →
  choc_price = 1 →
  cane_price = 1/2 →
  gum_quantity = 3 →
  choc_quantity = 5 →
  cane_quantity = 2 →
  initial_amount - (gum_price * gum_quantity + choc_price * choc_quantity + cane_price * cane_quantity) = 1 := by
  sorry

end NUMINAMATH_CALUDE_annas_remaining_money_l3339_333939


namespace NUMINAMATH_CALUDE_stack_height_is_3_meters_l3339_333987

/-- The number of packages in a stack -/
def packages_per_stack : ℕ := 60

/-- The number of sheets in a package -/
def sheets_per_package : ℕ := 500

/-- The thickness of a single sheet in millimeters -/
def sheet_thickness : ℚ := 1/10

/-- The height of a stack in meters -/
def stack_height : ℚ := 3

/-- Theorem stating that the height of a stack of packages is 3 meters -/
theorem stack_height_is_3_meters :
  (packages_per_stack : ℚ) * sheets_per_package * sheet_thickness / 1000 = stack_height :=
by sorry

end NUMINAMATH_CALUDE_stack_height_is_3_meters_l3339_333987


namespace NUMINAMATH_CALUDE_geometry_number_theory_arrangement_l3339_333963

theorem geometry_number_theory_arrangement (n_geometry : ℕ) (n_number_theory : ℕ) :
  n_geometry = 4 →
  n_number_theory = 5 →
  (number_of_arrangements : ℕ) =
    Nat.choose (n_number_theory + 1) n_geometry :=
by sorry

end NUMINAMATH_CALUDE_geometry_number_theory_arrangement_l3339_333963


namespace NUMINAMATH_CALUDE_batsman_final_average_l3339_333983

/-- Represents a batsman's performance -/
structure Batsman where
  innings : Nat
  totalRuns : Nat
  lastInningRuns : Nat
  averageIncrease : Nat

/-- Calculates the average runs of a batsman after their last inning -/
def finalAverage (b : Batsman) : Nat :=
  (b.totalRuns + b.lastInningRuns) / (b.innings + 1)

/-- Theorem stating the final average of the batsman -/
theorem batsman_final_average (b : Batsman) 
  (h1 : b.innings = 10)
  (h2 : b.lastInningRuns = 80)
  (h3 : b.averageIncrease = 5)
  (h4 : finalAverage b = (b.totalRuns / b.innings) + b.averageIncrease) :
  finalAverage b = 30 := by
  sorry

#check batsman_final_average

end NUMINAMATH_CALUDE_batsman_final_average_l3339_333983


namespace NUMINAMATH_CALUDE_graveling_cost_theorem_l3339_333918

def lawn_length : ℝ := 75
def lawn_width : ℝ := 45

def road_width_1 : ℝ := 6
def road_width_2 : ℝ := 5
def road_width_3 : ℝ := 4
def road_width_4 : ℝ := 3

def graveling_cost_1 : ℝ := 0.90
def graveling_cost_2 : ℝ := 0.85
def graveling_cost_3 : ℝ := 0.80
def graveling_cost_4 : ℝ := 0.75

def total_graveling_cost : ℝ :=
  road_width_1 * lawn_length * graveling_cost_1 +
  road_width_2 * lawn_length * graveling_cost_2 +
  road_width_3 * lawn_width * graveling_cost_3 +
  road_width_4 * lawn_width * graveling_cost_4

theorem graveling_cost_theorem : total_graveling_cost = 969 := by
  sorry

end NUMINAMATH_CALUDE_graveling_cost_theorem_l3339_333918


namespace NUMINAMATH_CALUDE_value_of_c_l3339_333997

theorem value_of_c (k a b c : ℝ) (hk : k ≠ 0) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : 1 / (k * a) - 1 / (k * b) = 1 / c) : c = k * a * b / (b - a) := by
  sorry

end NUMINAMATH_CALUDE_value_of_c_l3339_333997


namespace NUMINAMATH_CALUDE_sqrt_equality_implies_specific_values_l3339_333901

theorem sqrt_equality_implies_specific_values :
  ∀ a b : ℕ+,
  a < b →
  Real.sqrt (4 + Real.sqrt (76 + 40 * Real.sqrt 3)) = Real.sqrt a + Real.sqrt b →
  a = 1 ∧ b = 10 := by
sorry

end NUMINAMATH_CALUDE_sqrt_equality_implies_specific_values_l3339_333901


namespace NUMINAMATH_CALUDE_subtraction_of_negatives_l3339_333944

theorem subtraction_of_negatives : -14 - (-26) = 12 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_negatives_l3339_333944


namespace NUMINAMATH_CALUDE_seconds_in_minutes_l3339_333975

theorem seconds_in_minutes (minutes : ℚ) (seconds_per_minute : ℕ) :
  minutes = 11 / 3 →
  seconds_per_minute = 60 →
  (minutes * seconds_per_minute : ℚ) = 220 := by
sorry

end NUMINAMATH_CALUDE_seconds_in_minutes_l3339_333975


namespace NUMINAMATH_CALUDE_area_of_ABCM_l3339_333953

structure Polygon where
  sides : ℕ
  sideLength : ℝ
  rightAngles : Bool

def intersectionPoint (p1 p2 p3 p4 : Point) : Point := sorry

def quadrilateralArea (p1 p2 p3 p4 : Point) : ℝ := sorry

theorem area_of_ABCM (poly : Polygon) (A B C G J M : Point) :
  poly.sides = 14 ∧
  poly.sideLength = 3 ∧
  poly.rightAngles = true ∧
  M = intersectionPoint A G C J →
  quadrilateralArea A B C M = 24.75 := by
  sorry

end NUMINAMATH_CALUDE_area_of_ABCM_l3339_333953


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l3339_333982

theorem bridge_length_calculation (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 250 →
  train_speed_kmh = 72 →
  crossing_time = 20 →
  ∃ bridge_length : ℝ,
    bridge_length = 150 ∧
    train_length + bridge_length = (train_speed_kmh * 1000 / 3600) * crossing_time :=
by sorry

end NUMINAMATH_CALUDE_bridge_length_calculation_l3339_333982


namespace NUMINAMATH_CALUDE_matching_pair_probability_is_0_5226_l3339_333965

/-- Represents the types of shoes in the warehouse -/
inductive ShoeType
  | Sneaker
  | Boot
  | DressShoe

/-- Represents the shoe warehouse inventory -/
structure ShoeWarehouse where
  sneakers : ℕ
  boots : ℕ
  dressShoes : ℕ
  sneakerProb : ℝ
  bootProb : ℝ
  dressShoeProb : ℝ

/-- Calculates the probability of selecting a matching pair of shoes -/
def matchingPairProbability (warehouse : ShoeWarehouse) : ℝ :=
  let sneakerProb := warehouse.sneakers * warehouse.sneakerProb * (warehouse.sneakers - 1) * warehouse.sneakerProb
  let bootProb := warehouse.boots * warehouse.bootProb * (warehouse.boots - 1) * warehouse.bootProb
  let dressShoeProb := warehouse.dressShoes * warehouse.dressShoeProb * (warehouse.dressShoes - 1) * warehouse.dressShoeProb
  sneakerProb + bootProb + dressShoeProb

/-- Theorem stating the probability of selecting a matching pair of shoes -/
theorem matching_pair_probability_is_0_5226 :
  let warehouse : ShoeWarehouse := {
    sneakers := 12,
    boots := 15,
    dressShoes := 18,
    sneakerProb := 0.04,
    bootProb := 0.03,
    dressShoeProb := 0.02
  }
  matchingPairProbability warehouse = 0.5226 := by sorry

end NUMINAMATH_CALUDE_matching_pair_probability_is_0_5226_l3339_333965


namespace NUMINAMATH_CALUDE_tennis_balls_order_l3339_333928

theorem tennis_balls_order (white yellow : ℕ) : 
  white = yellow →
  white / (yellow + 90) = 8 / 13 →
  white + yellow = 288 :=
by sorry

end NUMINAMATH_CALUDE_tennis_balls_order_l3339_333928


namespace NUMINAMATH_CALUDE_max_m_value_l3339_333955

theorem max_m_value (x y m : ℝ) : 
  (4 * x + 3 * y = 4 * m + 5) →
  (3 * x - y = m - 1) →
  (x + 4 * y ≤ 3) →
  (∀ m' : ℝ, m' > m → ¬(∃ x' y' : ℝ, 
    (4 * x' + 3 * y' = 4 * m' + 5) ∧
    (3 * x' - y' = m' - 1) ∧
    (x' + 4 * y' ≤ 3))) →
  m = -1 := by
sorry

end NUMINAMATH_CALUDE_max_m_value_l3339_333955


namespace NUMINAMATH_CALUDE_not_all_vertices_on_same_branch_coordinates_of_Q_and_R_l3339_333940

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x * y = 1

-- Define the branches of the hyperbola
def branch1 (x y : ℝ) : Prop := hyperbola x y ∧ x > 0 ∧ y > 0
def branch2 (x y : ℝ) : Prop := hyperbola x y ∧ x < 0 ∧ y < 0

-- Define an equilateral triangle
def is_equilateral_triangle (P Q R : ℝ × ℝ) : Prop :=
  let (px, py) := P
  let (qx, qy) := Q
  let (rx, ry) := R
  (px - qx)^2 + (py - qy)^2 = (qx - rx)^2 + (qy - ry)^2 ∧
  (qx - rx)^2 + (qy - ry)^2 = (rx - px)^2 + (ry - py)^2

-- Theorem 1: Not all vertices can lie on the same branch
theorem not_all_vertices_on_same_branch 
  (P Q R : ℝ × ℝ) 
  (h_triangle : is_equilateral_triangle P Q R)
  (h_on_hyperbola : hyperbola P.1 P.2 ∧ hyperbola Q.1 Q.2 ∧ hyperbola R.1 R.2) :
  ¬(branch1 P.1 P.2 ∧ branch1 Q.1 Q.2 ∧ branch1 R.1 R.2) ∧
  ¬(branch2 P.1 P.2 ∧ branch2 Q.1 Q.2 ∧ branch2 R.1 R.2) :=
sorry

-- Theorem 2: Coordinates of Q and R given P(-1, -1)
theorem coordinates_of_Q_and_R
  (P Q R : ℝ × ℝ)
  (h_triangle : is_equilateral_triangle P Q R)
  (h_on_hyperbola : hyperbola P.1 P.2 ∧ hyperbola Q.1 Q.2 ∧ hyperbola R.1 R.2)
  (h_P : P = (-1, -1))
  (h_Q_R_branch1 : branch1 Q.1 Q.2 ∧ branch1 R.1 R.2) :
  (Q = (2 - Real.sqrt 3, 2 + Real.sqrt 3) ∧ R = (2 + Real.sqrt 3, 2 - Real.sqrt 3)) ∨
  (Q = (2 + Real.sqrt 3, 2 - Real.sqrt 3) ∧ R = (2 - Real.sqrt 3, 2 + Real.sqrt 3)) :=
sorry

end NUMINAMATH_CALUDE_not_all_vertices_on_same_branch_coordinates_of_Q_and_R_l3339_333940


namespace NUMINAMATH_CALUDE_prob_a_not_on_jan1_is_two_thirds_l3339_333908

/-- Represents the number of employees -/
def num_employees : ℕ := 6

/-- Represents the number of days -/
def num_days : ℕ := 3

/-- Represents the number of employees scheduled each day -/
def employees_per_day : ℕ := 2

/-- Calculates the total number of possible duty arrangements -/
def total_arrangements : ℕ := (num_employees.choose employees_per_day) * 
  ((num_employees - employees_per_day).choose employees_per_day) * 
  ((num_employees - 2*employees_per_day).choose employees_per_day)

/-- Calculates the number of arrangements where employee A is not on duty on January 1 -/
def arrangements_a_not_on_jan1 : ℕ := (num_employees - 1) * 
  ((num_employees - employees_per_day).choose employees_per_day) * 
  ((num_employees - 2*employees_per_day).choose employees_per_day)

/-- The probability that employee A is not on duty on January 1 -/
def prob_a_not_on_jan1 : ℚ := arrangements_a_not_on_jan1 / total_arrangements

theorem prob_a_not_on_jan1_is_two_thirds : 
  prob_a_not_on_jan1 = 2/3 := by sorry

end NUMINAMATH_CALUDE_prob_a_not_on_jan1_is_two_thirds_l3339_333908


namespace NUMINAMATH_CALUDE_ratio_percent_problem_l3339_333910

theorem ratio_percent_problem (ratio_percent : ℝ) (second_part : ℝ) (first_part : ℝ) :
  ratio_percent = 50 →
  second_part = 20 →
  first_part = ratio_percent / 100 * second_part →
  first_part = 10 :=
by sorry

end NUMINAMATH_CALUDE_ratio_percent_problem_l3339_333910


namespace NUMINAMATH_CALUDE_fourth_power_subset_exists_l3339_333956

/-- The set of prime numbers less than or equal to 26 -/
def primes_le_26 : Finset ℕ := sorry

/-- A function that represents a number as a tuple of exponents of primes <= 26 -/
def exponent_tuple (n : ℕ) : Fin 9 → ℕ := sorry

/-- The set M of 1985 different positive integers with prime factors <= 26 -/
def M : Finset ℕ := sorry

/-- The cardinality of M is 1985 -/
axiom M_card : Finset.card M = 1985

/-- All elements in M have prime factors <= 26 -/
axiom M_primes (n : ℕ) : n ∈ M → ∀ p : ℕ, p.Prime → p ∣ n → p ≤ 26

/-- All elements in M are different -/
axiom M_distinct : ∀ a b : ℕ, a ∈ M → b ∈ M → a ≠ b

/-- Main theorem: There exists a subset of 4 elements from M whose product is a fourth power -/
theorem fourth_power_subset_exists : 
  ∃ (a b c d : ℕ) (k : ℕ), a ∈ M ∧ b ∈ M ∧ c ∈ M ∧ d ∈ M ∧ 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  a * b * c * d = k^4 := by sorry

end NUMINAMATH_CALUDE_fourth_power_subset_exists_l3339_333956


namespace NUMINAMATH_CALUDE_peters_savings_l3339_333916

/-- Peter's vacation savings problem -/
theorem peters_savings (total_needed : ℕ) (monthly_savings : ℕ) (months_to_goal : ℕ) 
  (h1 : total_needed = 5000)
  (h2 : monthly_savings = 700)
  (h3 : months_to_goal = 3)
  (h4 : total_needed = monthly_savings * months_to_goal + current_savings) :
  current_savings = 2900 :=
by
  sorry

end NUMINAMATH_CALUDE_peters_savings_l3339_333916


namespace NUMINAMATH_CALUDE_common_chord_circle_equation_l3339_333980

-- Define the two circles
def circle_C1 (x y : ℝ) : Prop := x^2 + y^2 - 12*x - 2*y - 13 = 0
def circle_C2 (x y : ℝ) : Prop := x^2 + y^2 + 12*x + 16*y - 25 = 0

-- Define the result circle
def result_circle (x y : ℝ) : Prop := (x - 2)^2 + (y + 2)^2 = 25

-- Theorem statement
theorem common_chord_circle_equation :
  ∀ x y : ℝ,
  (∃ x₁ y₁ x₂ y₂ : ℝ,
    circle_C1 x₁ y₁ ∧ circle_C2 x₂ y₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 > 0 ∧
    (x - (x₁ + x₂)/2)^2 + (y - (y₁ + y₂)/2)^2 = ((x₁ - x₂)^2 + (y₁ - y₂)^2) / 4) →
  result_circle x y :=
by sorry

end NUMINAMATH_CALUDE_common_chord_circle_equation_l3339_333980


namespace NUMINAMATH_CALUDE_expand_product_l3339_333925

theorem expand_product (x : ℝ) : (x + 3) * (x - 2) * (x + 4) = x^3 + 5*x^2 - 2*x - 24 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l3339_333925


namespace NUMINAMATH_CALUDE_no_real_roots_third_polynomial_l3339_333962

/-- Given two quadratic polynomials with integer roots, prove the third has no real roots -/
theorem no_real_roots_third_polynomial (a b : ℝ) :
  (∃ x : ℤ, x^2 + a*x + b = 0) →
  (∃ y : ℤ, y^2 + a*y + (b+1) = 0) →
  ¬∃ z : ℝ, z^2 + a*z + (b+2) = 0 :=
by sorry

end NUMINAMATH_CALUDE_no_real_roots_third_polynomial_l3339_333962


namespace NUMINAMATH_CALUDE_largest_number_is_sqrt5_l3339_333922

theorem largest_number_is_sqrt5 (x y z : ℝ) 
  (sum_eq : x + y + z = 3)
  (sum_prod_eq : x*y + x*z + y*z = -11)
  (prod_eq : x*y*z = 15) :
  max x (max y z) = Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_largest_number_is_sqrt5_l3339_333922


namespace NUMINAMATH_CALUDE_cans_collected_l3339_333970

theorem cans_collected (monday_cans tuesday_cans : ℕ) 
  (h1 : monday_cans = 71) 
  (h2 : tuesday_cans = 27) : 
  monday_cans + tuesday_cans = 98 := by
  sorry

end NUMINAMATH_CALUDE_cans_collected_l3339_333970


namespace NUMINAMATH_CALUDE_delivery_driver_stops_l3339_333957

theorem delivery_driver_stops (initial_stops total_stops : ℕ) 
  (h1 : initial_stops = 3)
  (h2 : total_stops = 7) :
  total_stops - initial_stops = 4 := by
  sorry

end NUMINAMATH_CALUDE_delivery_driver_stops_l3339_333957


namespace NUMINAMATH_CALUDE_charles_share_l3339_333920

/-- The number of sheep in the inheritance problem -/
structure SheepInheritance where
  john : ℕ
  alfred : ℕ
  charles : ℕ
  alfred_more_than_john : alfred = (120 * john) / 100
  alfred_more_than_charles : alfred = (125 * charles) / 100
  john_share : john = 3600

/-- Theorem stating that Charles receives 3456 sheep -/
theorem charles_share (s : SheepInheritance) : s.charles = 3456 := by
  sorry

end NUMINAMATH_CALUDE_charles_share_l3339_333920


namespace NUMINAMATH_CALUDE_simplify_expression_l3339_333968

variable (R : Type*) [Ring R]
variable (a b : R)

theorem simplify_expression : (a - b) - (a + b) = -2 * b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3339_333968


namespace NUMINAMATH_CALUDE_smallest_dual_palindrome_proof_l3339_333998

/-- Checks if a natural number is a palindrome when represented in the given base. -/
def is_palindrome (n : ℕ) (base : ℕ) : Prop :=
  sorry

/-- Converts a natural number to its representation in the given base. -/
def to_base (n : ℕ) (base : ℕ) : List ℕ :=
  sorry

/-- The smallest 5-digit palindrome in base 2 that is also a 3-digit palindrome in base 5 -/
def smallest_dual_palindrome : ℕ := 27

theorem smallest_dual_palindrome_proof :
  (is_palindrome smallest_dual_palindrome 2) ∧
  (is_palindrome smallest_dual_palindrome 5) ∧
  (to_base smallest_dual_palindrome 2).length = 5 ∧
  (to_base smallest_dual_palindrome 5).length = 3 ∧
  (∀ m : ℕ, m < smallest_dual_palindrome →
    ¬(is_palindrome m 2 ∧ is_palindrome m 5 ∧
      (to_base m 2).length = 5 ∧ (to_base m 5).length = 3)) :=
by
  sorry

#eval smallest_dual_palindrome

end NUMINAMATH_CALUDE_smallest_dual_palindrome_proof_l3339_333998


namespace NUMINAMATH_CALUDE_shoes_alteration_problem_l3339_333931

theorem shoes_alteration_problem (cost_per_shoe : ℕ) (total_cost : ℕ) (num_pairs : ℕ) :
  cost_per_shoe = 29 →
  total_cost = 986 →
  num_pairs = total_cost / (2 * cost_per_shoe) →
  num_pairs = 17 :=
by sorry

end NUMINAMATH_CALUDE_shoes_alteration_problem_l3339_333931


namespace NUMINAMATH_CALUDE_simplified_fraction_sum_l3339_333938

theorem simplified_fraction_sum (a b : ℕ) (h : a = 54 ∧ b = 81) :
  let g := Nat.gcd a b
  (a / g) + (b / g) = 5 := by
sorry

end NUMINAMATH_CALUDE_simplified_fraction_sum_l3339_333938


namespace NUMINAMATH_CALUDE_hexagon_centers_square_area_ratio_l3339_333966

/-- Square represents a square in 2D space -/
structure Square where
  side : ℝ
  center : ℝ × ℝ

/-- RegularHexagon represents a regular hexagon in 2D space -/
structure RegularHexagon where
  side : ℝ
  center : ℝ × ℝ

/-- Configuration represents the problem setup -/
structure Configuration where
  square : Square
  hexagons : Fin 4 → RegularHexagon

/-- Defines the specific configuration described in the problem -/
def problem_configuration : Configuration :=
  sorry

/-- Calculate the area of a square given its side length -/
def square_area (s : Square) : ℝ :=
  s.side * s.side

/-- Calculate the area of the square formed by the centers of the hexagons -/
def hexagon_centers_square_area (c : Configuration) : ℝ :=
  sorry

/-- The main theorem to be proved -/
theorem hexagon_centers_square_area_ratio (c : Configuration) :
  hexagon_centers_square_area c / square_area c.square = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_centers_square_area_ratio_l3339_333966


namespace NUMINAMATH_CALUDE_monotonic_difference_increasing_decreasing_l3339_333948

-- Define monotonic functions on ℝ
def Monotonic (f : ℝ → ℝ) : Prop := 
  ∀ x y, x ≤ y → f x ≤ f y ∨ f x ≥ f y

-- Define increasing function
def Increasing (f : ℝ → ℝ) : Prop := 
  ∀ x y, x < y → f x < f y

-- Define decreasing function
def Decreasing (f : ℝ → ℝ) : Prop := 
  ∀ x y, x < y → f x > f y

-- Theorem statement
theorem monotonic_difference_increasing_decreasing 
  (f g : ℝ → ℝ) 
  (hf : Monotonic f) (hg : Monotonic g) :
  (Increasing f ∧ Decreasing g → Increasing (fun x ↦ f x - g x)) ∧
  (Decreasing f ∧ Increasing g → Decreasing (fun x ↦ f x - g x)) := by
  sorry


end NUMINAMATH_CALUDE_monotonic_difference_increasing_decreasing_l3339_333948


namespace NUMINAMATH_CALUDE_max_m_value_l3339_333932

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 9

-- Define the points A and B
def point_A (m : ℝ) : ℝ × ℝ := (-m, 0)
def point_B (m : ℝ) : ℝ × ℝ := (m, 0)

-- Define the condition for point P
def point_P_condition (P : ℝ × ℝ) (m : ℝ) : Prop :=
  circle_C P.1 P.2 ∧
  let AP := (P.1 + m, P.2)
  let BP := (P.1 - m, P.2)
  AP.1 * BP.1 + AP.2 * BP.2 = 0

theorem max_m_value :
  ∀ m : ℝ, m > 0 →
  (∃ P : ℝ × ℝ, point_P_condition P m) →
  m ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_max_m_value_l3339_333932


namespace NUMINAMATH_CALUDE_A_intersect_B_equals_zero_two_l3339_333905

def A : Set ℤ := {-2, 0, 2}

-- Define the absolute value function
def f (x : ℤ) : ℤ := abs x

-- Define B as the image of A under f
def B : Set ℤ := f '' A

-- State the theorem
theorem A_intersect_B_equals_zero_two : A ∩ B = {0, 2} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_equals_zero_two_l3339_333905


namespace NUMINAMATH_CALUDE_total_hours_worked_l3339_333946

def hours_per_day : ℕ := 3
def number_of_days : ℕ := 6

theorem total_hours_worked : hours_per_day * number_of_days = 18 := by
  sorry

end NUMINAMATH_CALUDE_total_hours_worked_l3339_333946


namespace NUMINAMATH_CALUDE_triangle_inequality_l3339_333961

theorem triangle_inequality (a b c A B C : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  3 * a^2 + 3 * b^2 = c^2 + 4 * a * b →
  Real.tan (Real.sin A) ≤ Real.tan (Real.cos B) :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3339_333961


namespace NUMINAMATH_CALUDE_reconstruct_numbers_l3339_333903

/-- Given 10 real numbers representing pairwise sums of 5 unknown numbers,
    prove that the 5 original numbers can be uniquely reconstructed. -/
theorem reconstruct_numbers (a : Fin 10 → ℝ) :
  ∃! (x : Fin 5 → ℝ), ∀ (i j : Fin 5), i < j →
    ∃ (k : Fin 10), a k = x i + x j :=
by sorry

end NUMINAMATH_CALUDE_reconstruct_numbers_l3339_333903


namespace NUMINAMATH_CALUDE_tangent_product_equals_two_l3339_333915

theorem tangent_product_equals_two (α β : Real) (h : α + β = π / 4) :
  (1 + Real.tan α) * (1 + Real.tan β) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_product_equals_two_l3339_333915


namespace NUMINAMATH_CALUDE_complement_of_M_l3339_333924

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2, 4}

theorem complement_of_M : (U \ M) = {3, 5} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_l3339_333924


namespace NUMINAMATH_CALUDE_sum_and_count_theorem_l3339_333992

def sum_of_integers (a b : ℕ) : ℕ := (b - a + 1) * (a + b) / 2

def count_even_integers (a b : ℕ) : ℕ := (b - a) / 2 + 1

theorem sum_and_count_theorem :
  let x := sum_of_integers 10 20
  let y := count_even_integers 10 20
  x + y = 171 := by sorry

end NUMINAMATH_CALUDE_sum_and_count_theorem_l3339_333992


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l3339_333974

theorem least_subtraction_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃ (x : ℕ), x ≤ d - 1 ∧ (n - x) % d = 0 ∧ ∀ (y : ℕ), y < x → (n - y) % d ≠ 0 :=
by sorry

theorem problem_solution :
  let n : ℕ := 102932847
  let d : ℕ := 25
  ∃ (x : ℕ), x = 22 ∧ x ≤ d - 1 ∧ (n - x) % d = 0 ∧ ∀ (y : ℕ), y < x → (n - y) % d ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l3339_333974


namespace NUMINAMATH_CALUDE_equation_proof_l3339_333937

theorem equation_proof : 361 + 2 * 19 * 6 + 36 = 625 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l3339_333937


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l3339_333923

theorem regular_polygon_sides (n : ℕ) : n ≥ 3 →
  (n : ℝ) - (n * (n - 3) / 2) = 2 → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l3339_333923


namespace NUMINAMATH_CALUDE_x_neq_zero_necessary_not_sufficient_l3339_333929

theorem x_neq_zero_necessary_not_sufficient :
  (∀ x : ℝ, x > 0 → x ≠ 0) ∧
  ¬(∀ x : ℝ, x ≠ 0 → x > 0) :=
by sorry

end NUMINAMATH_CALUDE_x_neq_zero_necessary_not_sufficient_l3339_333929


namespace NUMINAMATH_CALUDE_floor_plus_self_unique_solution_l3339_333941

theorem floor_plus_self_unique_solution (r : ℝ) : 
  (⌊r⌋ : ℝ) + r = 18.2 ↔ r = 9.2 := by sorry

end NUMINAMATH_CALUDE_floor_plus_self_unique_solution_l3339_333941


namespace NUMINAMATH_CALUDE_sqrt_sum_fourth_power_l3339_333902

theorem sqrt_sum_fourth_power : (Real.sqrt (Real.sqrt 9 + Real.sqrt 1))^4 = 16 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_fourth_power_l3339_333902


namespace NUMINAMATH_CALUDE_circle_point_range_l3339_333930

theorem circle_point_range (a : ℝ) : 
  ((-1 + a)^2 + (-1 - a)^2 < 4) → (-1 < a ∧ a < 1) :=
by sorry

end NUMINAMATH_CALUDE_circle_point_range_l3339_333930


namespace NUMINAMATH_CALUDE_solution_set_inequality_l3339_333914

theorem solution_set_inequality (x : ℝ) :
  Set.Icc (-1/2 : ℝ) 1 ∪ Set.Ioo 1 3 =
  {x | (x + 5) / ((x - 1)^2) ≥ 2 ∧ x ≠ 1} :=
by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l3339_333914


namespace NUMINAMATH_CALUDE_books_per_box_is_fifteen_l3339_333934

/-- Represents the number of books in Henry's collection at different stages --/
structure BookCollection where
  initial : Nat
  room : Nat
  coffeeTable : Nat
  kitchen : Nat
  final : Nat
  pickedUp : Nat

/-- Calculates the number of books in each donation box --/
def booksPerBox (collection : BookCollection) : Nat :=
  let totalDonated := collection.initial - collection.final + collection.pickedUp
  let outsideBoxes := collection.room + collection.coffeeTable + collection.kitchen
  let inBoxes := totalDonated - outsideBoxes
  inBoxes / 3

/-- Theorem stating that the number of books in each box is 15 --/
theorem books_per_box_is_fifteen (collection : BookCollection)
  (h1 : collection.initial = 99)
  (h2 : collection.room = 21)
  (h3 : collection.coffeeTable = 4)
  (h4 : collection.kitchen = 18)
  (h5 : collection.final = 23)
  (h6 : collection.pickedUp = 12) :
  booksPerBox collection = 15 := by
  sorry

end NUMINAMATH_CALUDE_books_per_box_is_fifteen_l3339_333934


namespace NUMINAMATH_CALUDE_six_houses_configurations_l3339_333967

/-- Represents the material of a house -/
inductive Material
  | Brick
  | Wood

/-- A configuration of houses is a list of their materials -/
def Configuration := List Material

/-- Checks if a configuration is valid (no adjacent wooden houses) -/
def isValidConfiguration (config : Configuration) : Bool :=
  match config with
  | [] => true
  | [_] => true
  | Material.Wood :: Material.Wood :: _ => false
  | _ :: rest => isValidConfiguration rest

/-- Generates all possible configurations of n houses -/
def allConfigurations (n : Nat) : List Configuration :=
  match n with
  | 0 => [[]]
  | m + 1 => 
    let prev := allConfigurations m
    (prev.map (λ c => Material.Brick :: c)) ++ (prev.map (λ c => Material.Wood :: c))

/-- Counts the number of valid configurations for n houses -/
def countValidConfigurations (n : Nat) : Nat :=
  (allConfigurations n).filter isValidConfiguration |>.length

/-- The main theorem: there are 21 valid configurations for 6 houses -/
theorem six_houses_configurations :
  countValidConfigurations 6 = 21 := by
  sorry


end NUMINAMATH_CALUDE_six_houses_configurations_l3339_333967


namespace NUMINAMATH_CALUDE_driver_net_rate_of_pay_l3339_333995

/-- Calculate the net rate of pay for a driver given specific conditions --/
theorem driver_net_rate_of_pay
  (travel_time : ℝ)
  (speed : ℝ)
  (fuel_efficiency : ℝ)
  (pay_rate : ℝ)
  (gas_price : ℝ)
  (h1 : travel_time = 3)
  (h2 : speed = 60)
  (h3 : fuel_efficiency = 25)
  (h4 : pay_rate = 0.60)
  (h5 : gas_price = 2.50)
  : ∃ (net_rate : ℝ), net_rate = 30 :=
by
  sorry


end NUMINAMATH_CALUDE_driver_net_rate_of_pay_l3339_333995


namespace NUMINAMATH_CALUDE_greatest_integer_quadratic_inequality_l3339_333977

theorem greatest_integer_quadratic_inequality :
  ∃ (n : ℤ), n^2 - 13*n + 40 ≤ 0 ∧
  ∀ (m : ℤ), m^2 - 13*m + 40 ≤ 0 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_quadratic_inequality_l3339_333977


namespace NUMINAMATH_CALUDE_cube_root_equality_l3339_333945

theorem cube_root_equality (x : ℝ) :
  (x * (x^5)^(1/4))^(1/3) = 4 → x = 2^(8/3) := by sorry

end NUMINAMATH_CALUDE_cube_root_equality_l3339_333945


namespace NUMINAMATH_CALUDE_topsoil_cost_theorem_l3339_333959

/-- The cost of topsoil in dollars per cubic foot -/
def cost_per_cubic_foot : ℝ := 8

/-- The number of cubic feet in one cubic yard -/
def cubic_feet_per_cubic_yard : ℝ := 27

/-- The number of cubic yards of topsoil -/
def cubic_yards : ℝ := 3

/-- The cost of topsoil for a given number of cubic yards -/
def topsoil_cost (yards : ℝ) : ℝ :=
  yards * cubic_feet_per_cubic_yard * cost_per_cubic_foot

theorem topsoil_cost_theorem : topsoil_cost cubic_yards = 648 := by
  sorry

end NUMINAMATH_CALUDE_topsoil_cost_theorem_l3339_333959


namespace NUMINAMATH_CALUDE_fifth_graders_count_l3339_333906

def sixth_graders : ℕ := 115
def seventh_graders : ℕ := 118
def teachers_per_grade : ℕ := 4
def parents_per_grade : ℕ := 2
def num_buses : ℕ := 5
def seats_per_bus : ℕ := 72

def total_seats : ℕ := num_buses * seats_per_bus
def total_chaperones : ℕ := (teachers_per_grade + parents_per_grade) * 3
def sixth_and_seventh : ℕ := sixth_graders + seventh_graders
def seats_taken : ℕ := sixth_and_seventh + total_chaperones

theorem fifth_graders_count : 
  total_seats - seats_taken = 109 := by sorry

end NUMINAMATH_CALUDE_fifth_graders_count_l3339_333906


namespace NUMINAMATH_CALUDE_prob_TT_after_second_H_l3339_333993

/-- A fair coin flip sequence that stops when two consecutive flips are the same -/
inductive CoinFlipSequence
  | HH
  | TT
  | HTH : CoinFlipSequence → CoinFlipSequence
  | HTT : CoinFlipSequence

/-- The probability of a coin flip sequence -/
def prob : CoinFlipSequence → ℚ
  | CoinFlipSequence.HH => 1/4
  | CoinFlipSequence.TT => 1/4
  | CoinFlipSequence.HTH s => (1/8) * prob s
  | CoinFlipSequence.HTT => 1/8

/-- The probability of getting two tails in a row but seeing a second head before seeing a second tail -/
def probTTAfterSecondH : ℚ := prob CoinFlipSequence.HTT

theorem prob_TT_after_second_H : probTTAfterSecondH = 1/24 := by
  sorry

end NUMINAMATH_CALUDE_prob_TT_after_second_H_l3339_333993


namespace NUMINAMATH_CALUDE_smallest_overlap_percentage_l3339_333921

/-- The smallest possible percentage of a population playing both football and basketball,
    given that 85% play football and 75% play basketball. -/
theorem smallest_overlap_percentage (total population_football population_basketball : ℝ) :
  population_football = 0.85 * total →
  population_basketball = 0.75 * total →
  total > 0 →
  ∃ (overlap : ℝ), 
    overlap ≥ 0.60 * total ∧
    overlap ≤ population_football ∧
    overlap ≤ population_basketball ∧
    ∀ (x : ℝ), 
      x ≥ 0 ∧ 
      x ≤ population_football ∧ 
      x ≤ population_basketball ∧ 
      population_football + population_basketball - x ≤ total → 
      x ≥ overlap :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_overlap_percentage_l3339_333921


namespace NUMINAMATH_CALUDE_function_equality_implies_k_range_l3339_333988

open Real

/-- Given a function f(x) = 1 + ln x + kx where k is a real number,
    if there exists a positive x such that e^x = f(x)/x, then k ≥ 1 -/
theorem function_equality_implies_k_range (k : ℝ) :
  (∃ x > 0, exp x = (1 + log x + k * x) / x) → k ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_function_equality_implies_k_range_l3339_333988


namespace NUMINAMATH_CALUDE_angle_implies_x_min_value_of_f_l3339_333943

noncomputable section

def a : ℝ × ℝ := (Real.sqrt 2 / 2, -Real.sqrt 2 / 2)
def b (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.cos x)

def angle_between (u v : ℝ × ℝ) : ℝ := Real.arccos ((u.1 * v.1 + u.2 * v.2) / (Real.sqrt (u.1^2 + u.2^2) * Real.sqrt (v.1^2 + v.2^2)))

def f (x : ℝ) : ℝ := a.1 * (b x).1 + a.2 * (b x).2

theorem angle_implies_x (x : ℝ) (h : x ∈ Set.Ioo 0 (Real.pi / 2)) :
  angle_between a (b x) = Real.pi / 3 → x = 5 * Real.pi / 12 := by sorry

theorem min_value_of_f :
  ∃ (x : ℝ), x ∈ Set.Ioo 0 (Real.pi / 2) ∧ ∀ (y : ℝ), y ∈ Set.Ioo 0 (Real.pi / 2) → f x ≤ f y ∧ f x = -Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_angle_implies_x_min_value_of_f_l3339_333943


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_no_real_roots_iff_negative_discriminant_l3339_333984

theorem quadratic_no_real_roots :
  ∀ (x : ℝ), x^2 + x + 2 ≠ 0 :=
by
  sorry

-- Auxiliary definitions and theorems
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

theorem no_real_roots_iff_negative_discriminant (a b c : ℝ) (ha : a ≠ 0) :
  (∀ x : ℝ, a*x^2 + b*x + c ≠ 0) ↔ discriminant a b c < 0 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_no_real_roots_iff_negative_discriminant_l3339_333984


namespace NUMINAMATH_CALUDE_survey_result_survey_result_proof_l3339_333996

theorem survey_result (total_surveyed : ℕ) 
  (electrical_fire_believers : ℕ) 
  (hantavirus_believers : ℕ) : Prop :=
  let electrical_fire_percentage : ℚ := 754 / 1000
  let hantavirus_percentage : ℚ := 523 / 1000
  (electrical_fire_believers : ℚ) / (total_surveyed : ℚ) = electrical_fire_percentage ∧
  (hantavirus_believers : ℚ) / (electrical_fire_believers : ℚ) = hantavirus_percentage ∧
  hantavirus_believers = 31 →
  total_surveyed = 78

theorem survey_result_proof : survey_result 78 59 31 :=
sorry

end NUMINAMATH_CALUDE_survey_result_survey_result_proof_l3339_333996


namespace NUMINAMATH_CALUDE_polynomial_inequality_l3339_333904

-- Define the polynomial P(x) = (x - x₁) ⋯ (x - xₙ)
def P (x : ℝ) (roots : List ℝ) : ℝ :=
  roots.foldl (fun acc r => acc * (x - r)) 1

-- State the theorem
theorem polynomial_inequality (roots : List ℝ) :
  ∀ x : ℝ, (deriv (P · roots) x)^2 ≥ (P x roots) * (deriv^[2] (P · roots) x) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_inequality_l3339_333904


namespace NUMINAMATH_CALUDE_train_length_l3339_333913

/-- The length of a train given its speed, platform length, and time to cross the platform -/
theorem train_length (train_speed : ℝ) (platform_length : ℝ) (crossing_time : ℝ) :
  train_speed = 72 * (5 / 18) →
  platform_length = 250 →
  crossing_time = 36 →
  train_speed * crossing_time - platform_length = 470 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l3339_333913


namespace NUMINAMATH_CALUDE_roses_unchanged_l3339_333994

def initial_roses : ℕ := 12
def initial_orchids : ℕ := 2
def final_orchids : ℕ := 21
def cut_orchids : ℕ := 19

theorem roses_unchanged (h : final_orchids - cut_orchids = initial_orchids) :
  initial_roses = initial_roses := by sorry

end NUMINAMATH_CALUDE_roses_unchanged_l3339_333994


namespace NUMINAMATH_CALUDE_quadratic_roots_preservation_l3339_333960

theorem quadratic_roots_preservation (p q α : ℝ) 
  (h1 : ∃ x : ℝ, x^2 + p*x + q = 0) 
  (h2 : 0 < α) (h3 : α ≤ 1) : 
  ∃ y : ℝ, α*y^2 + p*y + q = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_preservation_l3339_333960


namespace NUMINAMATH_CALUDE_arc_label_sum_bounds_l3339_333933

/-- Represents the color of a point on the circle -/
inductive Color
  | Red
  | Blue
  | Green

/-- Calculates the label for an arc based on endpoint colors -/
def arcLabel (c1 c2 : Color) : Nat :=
  match c1, c2 with
  | Color.Red, Color.Blue | Color.Blue, Color.Red => 1
  | Color.Red, Color.Green | Color.Green, Color.Red => 2
  | Color.Blue, Color.Green | Color.Green, Color.Blue => 3
  | _, _ => 0

/-- Represents the configuration of points on the circle -/
structure CircleConfig where
  points : List Color
  red_count : Nat
  blue_count : Nat
  green_count : Nat

/-- Calculates the sum of arc labels for a given configuration -/
def sumArcLabels (config : CircleConfig) : Nat :=
  let arcs := List.zip config.points (List.rotateLeft config.points 1)
  List.sum (List.map (fun (c1, c2) => arcLabel c1 c2) arcs)

/-- The main theorem statement -/
theorem arc_label_sum_bounds 
  (config : CircleConfig)
  (h_red : config.red_count = 40)
  (h_blue : config.blue_count = 30)
  (h_green : config.green_count = 20)
  (h_total : config.points.length = 90) :
  6 ≤ sumArcLabels config ∧ sumArcLabels config ≤ 140 := by
  sorry

end NUMINAMATH_CALUDE_arc_label_sum_bounds_l3339_333933


namespace NUMINAMATH_CALUDE_cars_without_ac_l3339_333936

theorem cars_without_ac (total : ℕ) (min_racing : ℕ) (max_ac_no_racing : ℕ)
  (h_total : total = 100)
  (h_min_racing : min_racing = 51)
  (h_max_ac_no_racing : max_ac_no_racing = 49) :
  total - (max_ac_no_racing + (min_racing - max_ac_no_racing)) = 49 := by
  sorry

end NUMINAMATH_CALUDE_cars_without_ac_l3339_333936


namespace NUMINAMATH_CALUDE_ball_ground_hit_time_l3339_333954

/-- The time at which a ball hits the ground when thrown downward -/
theorem ball_ground_hit_time :
  let h (t : ℝ) := -16 * t^2 - 30 * t + 200
  ∃ t : ℝ, h t = 0 ∧ t = (-15 + Real.sqrt 3425) / 16 :=
by sorry

end NUMINAMATH_CALUDE_ball_ground_hit_time_l3339_333954


namespace NUMINAMATH_CALUDE_specific_cube_unpainted_count_l3339_333942

/-- Represents a cube with painted strips on its faces -/
structure PaintedCube where
  size : Nat
  totalUnitCubes : Nat
  verticalStripWidth : Nat
  horizontalStripHeight : Nat

/-- Calculates the number of unpainted unit cubes in the painted cube -/
def unpaintedUnitCubes (cube : PaintedCube) : Nat :=
  sorry

/-- Theorem stating that a 6x6x6 cube with specific painted strips has 160 unpainted unit cubes -/
theorem specific_cube_unpainted_count :
  let cube : PaintedCube := {
    size := 6,
    totalUnitCubes := 216,
    verticalStripWidth := 2,
    horizontalStripHeight := 2
  }
  unpaintedUnitCubes cube = 160 := by
  sorry

end NUMINAMATH_CALUDE_specific_cube_unpainted_count_l3339_333942


namespace NUMINAMATH_CALUDE_garrett_peanut_granola_bars_l3339_333981

/-- The number of granola bars Garrett bought in total -/
def total_granola_bars : ℕ := 14

/-- The number of oatmeal raisin granola bars Garrett bought -/
def oatmeal_raisin_bars : ℕ := 6

/-- The number of peanut granola bars Garrett bought -/
def peanut_granola_bars : ℕ := total_granola_bars - oatmeal_raisin_bars

theorem garrett_peanut_granola_bars : peanut_granola_bars = 8 := by
  sorry

end NUMINAMATH_CALUDE_garrett_peanut_granola_bars_l3339_333981


namespace NUMINAMATH_CALUDE_anns_skating_speed_l3339_333971

/-- Proves that Ann's skating speed is 6 miles per hour given the problem conditions. -/
theorem anns_skating_speed :
  ∀ (ann_speed : ℝ),
  let glenda_speed : ℝ := 8
  let time : ℝ := 3
  let total_distance : ℝ := 42
  (ann_speed * time + glenda_speed * time = total_distance) →
  ann_speed = 6 := by
sorry

end NUMINAMATH_CALUDE_anns_skating_speed_l3339_333971
