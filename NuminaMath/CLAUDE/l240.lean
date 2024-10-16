import Mathlib

namespace NUMINAMATH_CALUDE_inequality_equivalence_l240_24055

def inequality_solution (x : ℝ) : Prop :=
  x ∈ Set.Iio 0 ∪ Set.Ioo 0 5 ∪ Set.Ioi 5

theorem inequality_equivalence :
  ∀ x : ℝ, (x^2 / (x - 5)^2 > 0) ↔ inequality_solution x :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l240_24055


namespace NUMINAMATH_CALUDE_basketball_season_games_l240_24051

/-- The number of teams in the basketball conference -/
def num_teams : ℕ := 10

/-- The number of times each team plays every other team -/
def intra_conference_games : ℕ := 2

/-- The number of non-conference games each team plays -/
def non_conference_games : ℕ := 6

/-- The total number of games in a season -/
def total_games : ℕ := (num_teams.choose 2 * intra_conference_games) + (num_teams * non_conference_games)

theorem basketball_season_games :
  total_games = 150 := by
sorry

end NUMINAMATH_CALUDE_basketball_season_games_l240_24051


namespace NUMINAMATH_CALUDE_point_on_line_k_l240_24007

/-- A line passing through the origin with slope 1/5 -/
def line_k (x y : ℝ) : Prop := y = (1/5) * x

theorem point_on_line_k (x : ℝ) : 
  line_k x 1 → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_k_l240_24007


namespace NUMINAMATH_CALUDE_sum_nonnegative_implies_one_nonnegative_l240_24092

theorem sum_nonnegative_implies_one_nonnegative (a b : ℝ) :
  a + b ≥ 0 → (a ≥ 0 ∨ b ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_sum_nonnegative_implies_one_nonnegative_l240_24092


namespace NUMINAMATH_CALUDE_cube_packing_percentage_l240_24052

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents the dimensions of a cube -/
structure CubeDimensions where
  side : ℕ

/-- Calculates the number of cubes that fit along a given dimension -/
def cubesFitAlongDimension (boxDim : ℕ) (cubeSide : ℕ) : ℕ :=
  boxDim / cubeSide

/-- Calculates the total number of cubes that fit in the box -/
def totalCubesFit (box : BoxDimensions) (cube : CubeDimensions) : ℕ :=
  (cubesFitAlongDimension box.length cube.side) *
  (cubesFitAlongDimension box.width cube.side) *
  (cubesFitAlongDimension box.height cube.side)

/-- Calculates the volume of a rectangular box -/
def boxVolume (box : BoxDimensions) : ℕ :=
  box.length * box.width * box.height

/-- Calculates the volume of a cube -/
def cubeVolume (cube : CubeDimensions) : ℕ :=
  cube.side * cube.side * cube.side

/-- Calculates the percentage of box volume occupied by cubes -/
def percentageOccupied (box : BoxDimensions) (cube : CubeDimensions) : ℚ :=
  let totalCubes := totalCubesFit box cube
  let volumeOccupied := totalCubes * (cubeVolume cube)
  (volumeOccupied : ℚ) / (boxVolume box : ℚ) * 100

/-- Theorem stating that the percentage of volume occupied by 3-inch cubes
    in a 9x8x12 inch box is 75% -/
theorem cube_packing_percentage :
  let box := BoxDimensions.mk 9 8 12
  let cube := CubeDimensions.mk 3
  percentageOccupied box cube = 75 := by
  sorry

end NUMINAMATH_CALUDE_cube_packing_percentage_l240_24052


namespace NUMINAMATH_CALUDE_winston_gas_refill_l240_24062

/-- Calculates the amount of gas needed to refill a car's tank -/
def gas_needed_to_refill (initial_gas tank_capacity gas_used_store gas_used_doctor : ℚ) : ℚ :=
  tank_capacity - (initial_gas - gas_used_store - gas_used_doctor)

/-- Proves that given the initial conditions, the amount of gas needed to refill the tank is 10 gallons -/
theorem winston_gas_refill :
  let initial_gas : ℚ := 10
  let tank_capacity : ℚ := 12
  let gas_used_store : ℚ := 6
  let gas_used_doctor : ℚ := 2
  gas_needed_to_refill initial_gas tank_capacity gas_used_store gas_used_doctor = 10 := by
  sorry


end NUMINAMATH_CALUDE_winston_gas_refill_l240_24062


namespace NUMINAMATH_CALUDE_impossible_configuration_l240_24023

/-- Represents the sign at a vertex -/
inductive Sign
| Plus
| Minus

/-- Represents the state of the 12-gon -/
def TwelveGonState := Fin 12 → Sign

/-- Initial state of the 12-gon -/
def initialState : TwelveGonState :=
  fun i => if i = 0 then Sign.Minus else Sign.Plus

/-- Applies an operation to change signs at consecutive vertices -/
def applyOperation (state : TwelveGonState) (start : Fin 12) (count : Nat) : TwelveGonState :=
  fun i => if (i - start) % 12 < count then
    match state i with
    | Sign.Plus => Sign.Minus
    | Sign.Minus => Sign.Plus
    else state i

/-- Checks if the state matches the target configuration -/
def isTargetState (state : TwelveGonState) : Prop :=
  state 1 = Sign.Minus ∧ ∀ i : Fin 12, i ≠ 1 → state i = Sign.Plus

/-- The main theorem to be proved -/
theorem impossible_configuration
  (n : Nat)
  (h : n = 6 ∨ n = 4 ∨ n = 3)
  : ¬ ∃ (operations : List (Fin 12)), 
    let finalState := operations.foldl (fun s (start : Fin 12) => applyOperation s start n) initialState
    isTargetState finalState :=
sorry

end NUMINAMATH_CALUDE_impossible_configuration_l240_24023


namespace NUMINAMATH_CALUDE_inequality_condition_max_area_ellipse_l240_24089

-- Define the line l: y = k(x+1)
def line_l (k : ℝ) (x : ℝ) : ℝ := k * (x + 1)

-- Define the ellipse: x^2 + 4y^2 = a^2
def ellipse (a : ℝ) (x y : ℝ) : Prop := x^2 + 4*y^2 = a^2

-- Define the intersection points A and B
def intersection_points (k a : ℝ) : Prop :=
  ∃ x1 y1 x2 y2 : ℝ,
    ellipse a x1 y1 ∧ y1 = line_l k x1 ∧
    ellipse a x2 y2 ∧ y2 = line_l k x2 ∧
    (x1 ≠ x2 ∨ y1 ≠ y2)

-- Define point C as the intersection of line l with x-axis
def point_c (k : ℝ) : ℝ := -1

-- Define the condition AC = 2CB
def ac_twice_cb (k a : ℝ) : Prop :=
  ∃ x1 y1 x2 y2 : ℝ,
    ellipse a x1 y1 ∧ y1 = line_l k x1 ∧
    ellipse a x2 y2 ∧ y2 = line_l k x2 ∧
    (x1 - point_c k) = 2 * (point_c k - x2)

-- Theorem 1: a^2 > 4k^2 / (1+k^2)
theorem inequality_condition (k a : ℝ) (h1 : a > 0) (h2 : intersection_points k a) :
  a^2 > 4*k^2 / (1 + k^2) := by sorry

-- Theorem 2: When the area of triangle OAB is maximized, the equation of the ellipse is x^2 + 4y^2 = 5
theorem max_area_ellipse (k a : ℝ) (h1 : a > 0) (h2 : intersection_points k a) (h3 : ac_twice_cb k a) :
  (∀ x y : ℝ, ellipse a x y ↔ x^2 + 4*y^2 = 5) := by sorry

end NUMINAMATH_CALUDE_inequality_condition_max_area_ellipse_l240_24089


namespace NUMINAMATH_CALUDE_bridge_length_l240_24075

/-- The length of a bridge given train parameters -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 130 ∧ 
  train_speed_kmh = 54 ∧ 
  crossing_time = 30 →
  (train_speed_kmh * 1000 / 3600) * crossing_time - train_length = 320 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_l240_24075


namespace NUMINAMATH_CALUDE_expression_evaluation_l240_24024

theorem expression_evaluation :
  68 + (156 / 12) + (11 * 19) - 250 - (450 / 9) = -10 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l240_24024


namespace NUMINAMATH_CALUDE_repeating_decimal_fraction_l240_24016

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents a repeating decimal of the form 0.ẋyȧ -/
def RepeatingDecimal (x y : Digit) : ℚ :=
  (x.val * 100 + y.val * 10 + 3) / 999

/-- The theorem to be proved -/
theorem repeating_decimal_fraction (x y : Digit) (a : ℤ) 
  (h1 : x ≠ y)
  (h2 : RepeatingDecimal x y = a / 27) :
  a = 37 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_fraction_l240_24016


namespace NUMINAMATH_CALUDE_equation_equivalence_and_domain_x_domain_l240_24061

-- Define the original equation
def original_equation (x y : ℝ) : Prop :=
  x = (2 * y + 1) / (y - 2)

-- Define the inverted equation
def inverted_equation (x y : ℝ) : Prop :=
  y = (2 * x + 1) / (x - 2)

-- Theorem stating the equivalence of the equations and the domain of x
theorem equation_equivalence_and_domain :
  ∀ x y : ℝ, original_equation x y ↔ (inverted_equation x y ∧ x ≠ 2) :=
by
  sorry

-- Theorem stating the domain of x
theorem x_domain : ∀ x : ℝ, (∃ y : ℝ, original_equation x y) → x ≠ 2 :=
by
  sorry

end NUMINAMATH_CALUDE_equation_equivalence_and_domain_x_domain_l240_24061


namespace NUMINAMATH_CALUDE_line_segment_b_value_l240_24085

/-- Given a line segment with slope -3/2 from (0, b) to (8, 0), prove b = 12 -/
theorem line_segment_b_value (b : ℝ) : 
  (∀ x y, 0 ≤ x → x ≤ 8 → y = b - (3/2) * x) → 
  (b - (3/2) * 8 = 0) → 
  b = 12 := by
  sorry


end NUMINAMATH_CALUDE_line_segment_b_value_l240_24085


namespace NUMINAMATH_CALUDE_one_student_per_class_l240_24097

/-- Represents a school with a reading program -/
structure School where
  classes : ℕ
  books_per_student_per_month : ℕ
  total_books_per_year : ℕ

/-- Calculates the number of students in each class -/
def students_per_class (school : School) : ℕ :=
  school.total_books_per_year / (school.books_per_student_per_month * 12)

/-- Theorem stating that the number of students in each class is 1 -/
theorem one_student_per_class (school : School) 
  (h1 : school.classes > 0)
  (h2 : school.books_per_student_per_month = 3)
  (h3 : school.total_books_per_year = 36) : 
  students_per_class school = 1 := by
  sorry

#check one_student_per_class

end NUMINAMATH_CALUDE_one_student_per_class_l240_24097


namespace NUMINAMATH_CALUDE_right_triangular_prism_dimension_l240_24037

/-- 
Given a right triangular prism with:
- base edges a = 5 and b = 12
- height c = 13
- body diagonal d = 15
Prove that the third dimension of a rectangular face (h) is equal to 2√14
-/
theorem right_triangular_prism_dimension (a b c d h : ℝ) : 
  a = 5 → b = 12 → c = 13 → d = 15 →
  a^2 + b^2 = c^2 →
  d^2 = a^2 + b^2 + h^2 →
  h = 2 * Real.sqrt 14 := by
  sorry

end NUMINAMATH_CALUDE_right_triangular_prism_dimension_l240_24037


namespace NUMINAMATH_CALUDE_line_plane_perpendicularity_l240_24057

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perpendicularLines : Line → Line → Prop)

-- State the theorem
theorem line_plane_perpendicularity 
  (a b : Line) (α : Plane) :
  parallel a α → perpendicular b α → perpendicularLines a b :=
sorry

end NUMINAMATH_CALUDE_line_plane_perpendicularity_l240_24057


namespace NUMINAMATH_CALUDE_vertices_form_parabola_l240_24003

/-- The set of vertices of a family of parabolas forms another parabola -/
theorem vertices_form_parabola (a c : ℝ) (h_a : a = 2) (h_c : c = 6) :
  ∃ (f : ℝ → ℝ × ℝ),
    (∀ t, f t = (-(t / (2 * a)), a * (-(t / (2 * a)))^2 + t * (-(t / (2 * a))) + c)) ∧
    (∃ (g : ℝ → ℝ), ∀ x, (x, g x) ∈ Set.range f ↔ g x = -a * x^2 + c) :=
by sorry

end NUMINAMATH_CALUDE_vertices_form_parabola_l240_24003


namespace NUMINAMATH_CALUDE_circle_B_radius_l240_24087

-- Define the circles
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the problem setup
def problem_setup (A B C D : Circle) : Prop :=
  -- Circles A, B, and C are externally tangent to each other
  (A.radius + B.radius = dist A.center B.center) ∧
  (A.radius + C.radius = dist A.center C.center) ∧
  (B.radius + C.radius = dist B.center C.center) ∧
  -- Circles A, B, and C are internally tangent to circle D
  (D.radius - A.radius = dist D.center A.center) ∧
  (D.radius - B.radius = dist D.center B.center) ∧
  (D.radius - C.radius = dist D.center C.center) ∧
  -- Circles B and C are congruent
  (B.radius = C.radius) ∧
  -- Circle A has radius 1
  (A.radius = 1) ∧
  -- Circle A passes through the center of D
  (dist A.center D.center = A.radius + D.radius)

-- Theorem statement
theorem circle_B_radius (A B C D : Circle) :
  problem_setup A B C D → B.radius = 8/9 := by
  sorry

end NUMINAMATH_CALUDE_circle_B_radius_l240_24087


namespace NUMINAMATH_CALUDE_linear_function_expression_y_value_at_negative_four_l240_24068

/-- A linear function passing through two given points -/
structure LinearFunction where
  k : ℝ
  b : ℝ
  point1 : k * 1 + b = 5
  point2 : k * (-1) + b = 1

/-- The unique linear function passing through (1, 5) and (-1, 1) -/
def uniqueLinearFunction : LinearFunction where
  k := 2
  b := 3
  point1 := by sorry
  point2 := by sorry

theorem linear_function_expression (f : LinearFunction) :
  f.k = 2 ∧ f.b = 3 := by sorry

theorem y_value_at_negative_four (f : LinearFunction) :
  f.k * (-4) + f.b = -5 := by sorry

end NUMINAMATH_CALUDE_linear_function_expression_y_value_at_negative_four_l240_24068


namespace NUMINAMATH_CALUDE_number_ordering_l240_24086

theorem number_ordering : (5 : ℝ) / 2 < 3 ∧ 3 < Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_number_ordering_l240_24086


namespace NUMINAMATH_CALUDE_solution_set_theorem_l240_24033

-- Define the function f
def f (a b x : ℝ) : ℝ := (a * x - 1) * (x + b)

-- State the theorem
theorem solution_set_theorem (a b : ℝ) :
  (∀ x : ℝ, f a b x > 0 ↔ -1 < x ∧ x < 3) →
  (∀ x : ℝ, f a b (-2*x) < 0 ↔ x < -3/2 ∨ 1/2 < x) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_theorem_l240_24033


namespace NUMINAMATH_CALUDE_inverse_mod_101_l240_24066

theorem inverse_mod_101 (h : (7⁻¹ : ZMod 101) = 55) : (49⁻¹ : ZMod 101) = 96 := by
  sorry

end NUMINAMATH_CALUDE_inverse_mod_101_l240_24066


namespace NUMINAMATH_CALUDE_solve_equation_l240_24036

theorem solve_equation (x : ℝ) : 
  Real.sqrt ((2 / x) + 3) = 5 / 3 → x = -9 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l240_24036


namespace NUMINAMATH_CALUDE_cross_section_area_is_21_over_8_l240_24002

/-- Right prism with specific properties -/
structure RightPrism where
  -- Base triangle
  base_hypotenuse : ℝ
  base_angle_B : ℝ
  base_angle_C : ℝ
  -- Cutting plane properties
  distance_C_to_plane : ℝ

/-- The cross-section of the prism -/
def cross_section_area (prism : RightPrism) : ℝ :=
  sorry

/-- Main theorem: The area of the cross-section is 21/8 -/
theorem cross_section_area_is_21_over_8 (prism : RightPrism) 
  (h1 : prism.base_hypotenuse = Real.sqrt 14)
  (h2 : prism.base_angle_B = 90)
  (h3 : prism.base_angle_C = 30)
  (h4 : prism.distance_C_to_plane = 2) :
  cross_section_area prism = 21 / 8 :=
sorry

end NUMINAMATH_CALUDE_cross_section_area_is_21_over_8_l240_24002


namespace NUMINAMATH_CALUDE_roots_equation_l240_24096

theorem roots_equation (A B a b c d : ℝ) : 
  (a^2 + A*a + 1 = 0) → 
  (b^2 + A*b + 1 = 0) → 
  (c^2 + B*c + 1 = 0) → 
  (d^2 + B*d + 1 = 0) → 
  (a - c)*(b - c)*(a + d)*(b + d) = B^2 - A^2 := by
  sorry

end NUMINAMATH_CALUDE_roots_equation_l240_24096


namespace NUMINAMATH_CALUDE_girls_on_playground_l240_24049

theorem girls_on_playground (total_children boys : ℕ) 
  (h1 : total_children = 62) 
  (h2 : boys = 27) : 
  total_children - boys = 35 := by
sorry

end NUMINAMATH_CALUDE_girls_on_playground_l240_24049


namespace NUMINAMATH_CALUDE_lunch_sales_calculation_l240_24048

/-- Represents the number of hot dogs served by a restaurant -/
structure HotDogSales where
  total : ℕ
  dinner : ℕ
  lunch : ℕ

/-- Given the total number of hot dogs sold and the number sold during dinner,
    calculate the number of hot dogs sold during lunch -/
def lunchSales (sales : HotDogSales) : ℕ :=
  sales.total - sales.dinner

theorem lunch_sales_calculation (sales : HotDogSales) 
  (h1 : sales.total = 11)
  (h2 : sales.dinner = 2) :
  lunchSales sales = 9 := by
sorry

end NUMINAMATH_CALUDE_lunch_sales_calculation_l240_24048


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l240_24021

theorem cube_root_equation_solution :
  ∃! x : ℝ, (5 - x / 3) ^ (1/3 : ℝ) = -4 ∧ x = 207 :=
by sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l240_24021


namespace NUMINAMATH_CALUDE_cubic_root_in_interval_l240_24032

theorem cubic_root_in_interval (a b c : ℝ) 
  (h_roots : ∃ (r₁ r₂ r₃ : ℝ), ∀ x, x^3 + a*x^2 + b*x + c = (x - r₁) * (x - r₂) * (x - r₃))
  (h_sum : -2 ≤ a + b + c ∧ a + b + c ≤ 0) :
  ∃ r, (r^3 + a*r^2 + b*r + c = 0) ∧ (0 ≤ r ∧ r ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_in_interval_l240_24032


namespace NUMINAMATH_CALUDE_race_result_l240_24093

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ  -- Speed in meters per second
  time : ℝ   -- Time to complete the race in seconds

/-- The race scenario -/
def Race : Prop :=
  ∃ (A B : Runner),
    -- Total race distance is 200 meters
    A.speed * A.time = 200 ∧
    -- A's time is 33 seconds
    A.time = 33 ∧
    -- A is 35 meters ahead of B at the finish line
    A.speed * A.time - B.speed * A.time = 35 ∧
    -- B's total race time
    B.time * B.speed = 200 ∧
    -- A beats B by 7 seconds
    B.time - A.time = 7

/-- Theorem stating that given the race conditions, A beats B by 7 seconds -/
theorem race_result : Race := by sorry

end NUMINAMATH_CALUDE_race_result_l240_24093


namespace NUMINAMATH_CALUDE_percent_y_of_x_l240_24095

theorem percent_y_of_x (x y : ℝ) (h : 0.2 * (x - y) = 0.15 * (x + y)) : 
  y = (100 / 7) * x / 100 := by
  sorry

end NUMINAMATH_CALUDE_percent_y_of_x_l240_24095


namespace NUMINAMATH_CALUDE_parabola_properties_l240_24063

/-- Given a parabola y = x^2 + 2bx + b^2 - 2 where b > 0 and passing through point (0, -1) -/
theorem parabola_properties (b : ℝ) (h1 : b > 0) :
  let f (x : ℝ) := x^2 + 2*b*x + b^2 - 2
  ∃ (vertex_x vertex_y : ℝ),
    -- 1. The vertex coordinates are (-b, -2)
    (vertex_x = -b ∧ vertex_y = -2) ∧
    -- Parabola passes through (0, -1)
    (f 0 = -1) ∧
    -- 2. When -2 < x < 3, the range of y is -2 ≤ y < 14
    (∀ x, -2 < x → x < 3 → -2 ≤ f x ∧ f x < 14) ∧
    -- 3. When k ≤ x ≤ 2 and -2 ≤ y ≤ 7, the range of k is -4 ≤ k ≤ -1
    (∀ k, (∀ x, k ≤ x → x ≤ 2 → -2 ≤ f x → f x ≤ 7) → -4 ≤ k ∧ k ≤ -1) :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l240_24063


namespace NUMINAMATH_CALUDE_cubic_derivative_value_l240_24005

theorem cubic_derivative_value (f : ℝ → ℝ) (x₀ : ℝ) :
  (∀ x, f x = x^3) →
  (deriv f x₀ = 3) →
  (x₀ = 1 ∨ x₀ = -1) := by
sorry

end NUMINAMATH_CALUDE_cubic_derivative_value_l240_24005


namespace NUMINAMATH_CALUDE_min_value_problem_l240_24071

theorem min_value_problem (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : 1 / (x + 3) + 1 / (y + 3) = 1 / 4) :
  ∀ a b : ℝ, a > 0 → b > 0 → 1 / (a + 3) + 1 / (b + 3) = 1 / 4 →
  x + 3 * y ≤ a + 3 * b ∧ x + 3 * y = 18 + 21 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_min_value_problem_l240_24071


namespace NUMINAMATH_CALUDE_jar_size_proof_l240_24056

/-- Proves that the size of the second jar type is 1/2 gallon given the problem conditions -/
theorem jar_size_proof (total_water : ℚ) (total_jars : ℕ) 
  (h1 : total_water = 28)
  (h2 : total_jars = 48)
  (h3 : ∃ (n : ℕ), n * (1 + x + 1/4) = total_jars ∧ n * (1 + x + 1/4) = total_water)
  : x = 1/2 := by
  sorry

#check jar_size_proof

end NUMINAMATH_CALUDE_jar_size_proof_l240_24056


namespace NUMINAMATH_CALUDE_floor_negative_fraction_l240_24011

theorem floor_negative_fraction : ⌊(-19 : ℝ) / 3⌋ = -7 := by sorry

end NUMINAMATH_CALUDE_floor_negative_fraction_l240_24011


namespace NUMINAMATH_CALUDE_person_b_hit_six_shots_l240_24079

/-- A shooting competition between two people -/
structure ShootingCompetition where
  hits_points : ℕ     -- Points gained for each hit
  miss_points : ℕ     -- Points deducted for each miss
  total_shots : ℕ     -- Total number of shots per person
  total_score : ℕ     -- Combined score of both persons
  score_diff  : ℕ     -- Score difference between person A and B

/-- The number of shots hit by person B in the competition -/
def person_b_hits (comp : ShootingCompetition) : ℕ := 
  sorry

/-- Theorem stating that person B hit 6 shots in the given competition -/
theorem person_b_hit_six_shots 
  (comp : ShootingCompetition) 
  (h1 : comp.hits_points = 20)
  (h2 : comp.miss_points = 12)
  (h3 : comp.total_shots = 10)
  (h4 : comp.total_score = 208)
  (h5 : comp.score_diff = 64) : 
  person_b_hits comp = 6 := by
  sorry

end NUMINAMATH_CALUDE_person_b_hit_six_shots_l240_24079


namespace NUMINAMATH_CALUDE_smallest_quadratic_coefficient_l240_24004

theorem smallest_quadratic_coefficient (a b c : ℤ) :
  (∃ x y : ℝ, x ≠ y ∧ 0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 ∧ 
   a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0) →
  |a| ≥ 5 :=
sorry

end NUMINAMATH_CALUDE_smallest_quadratic_coefficient_l240_24004


namespace NUMINAMATH_CALUDE_dance_lesson_cost_l240_24080

/-- Calculates the total cost of dance lessons given the number of lessons,
    cost per lesson, and number of free lessons. -/
def total_cost (total_lessons : ℕ) (cost_per_lesson : ℕ) (free_lessons : ℕ) : ℕ :=
  (total_lessons - free_lessons) * cost_per_lesson

/-- Theorem stating that given 10 dance lessons costing $10 each,
    with 2 lessons for free, the total cost is $80. -/
theorem dance_lesson_cost :
  total_cost 10 10 2 = 80 := by
  sorry

end NUMINAMATH_CALUDE_dance_lesson_cost_l240_24080


namespace NUMINAMATH_CALUDE_max_value_of_f_l240_24078

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x + 2

-- Define the interval
def interval : Set ℝ := Set.Icc (-2) 2

-- Statement to prove
theorem max_value_of_f :
  ∃ (c : ℝ), c ∈ interval ∧ ∀ (x : ℝ), x ∈ interval → f x ≤ f c ∧ f c = 7 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l240_24078


namespace NUMINAMATH_CALUDE_probability_two_red_crayons_l240_24053

/-- The probability of selecting 2 red crayons from a jar containing 3 red, 2 blue, and 1 green crayon -/
theorem probability_two_red_crayons (total : ℕ) (red : ℕ) (blue : ℕ) (green : ℕ) :
  total = red + blue + green →
  total = 6 →
  red = 3 →
  blue = 2 →
  green = 1 →
  (Nat.choose red 2 : ℚ) / (Nat.choose total 2) = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_red_crayons_l240_24053


namespace NUMINAMATH_CALUDE_negation_equivalence_l240_24065

theorem negation_equivalence :
  (¬ ∃ x₀ : ℝ, x₀ > 0 ∧ Real.log x₀ > 3 - x₀) ↔ 
  (∀ x : ℝ, x > 0 → Real.log x ≤ 3 - x) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l240_24065


namespace NUMINAMATH_CALUDE_triangle_heights_semiperimeter_inequality_l240_24091

/-- Given a triangle with heights m_a, m_b, m_c and semiperimeter s,
    prove that the sum of squares of the heights is less than or equal to
    the square of the semiperimeter. -/
theorem triangle_heights_semiperimeter_inequality 
  (m_a m_b m_c s : ℝ) 
  (h_pos_a : 0 < m_a) (h_pos_b : 0 < m_b) (h_pos_c : 0 < m_c) (h_pos_s : 0 < s)
  (h_heights : ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    m_a = (2 * (s * (s - a) * (s - b) * (s - c))^(1/2)) / a ∧
    m_b = (2 * (s * (s - a) * (s - b) * (s - c))^(1/2)) / b ∧
    m_c = (2 * (s * (s - a) * (s - b) * (s - c))^(1/2)) / c ∧
    s = (a + b + c) / 2) :
  m_a^2 + m_b^2 + m_c^2 ≤ s^2 := by sorry

end NUMINAMATH_CALUDE_triangle_heights_semiperimeter_inequality_l240_24091


namespace NUMINAMATH_CALUDE_inequality_proof_l240_24074

theorem inequality_proof (a b c d : ℝ) (h1 : d ≥ 0) (h2 : a + b = 2) (h3 : c + d = 2) :
  (a^2 + c^2) * (a^2 + d^2) * (b^2 + c^2) * (b^2 + d^2) ≤ 25 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l240_24074


namespace NUMINAMATH_CALUDE_repeating_decimal_proof_l240_24040

/-- The repeating decimal 0.817817817... as a real number -/
def F : ℚ := 817 / 999

/-- The difference between the denominator and numerator of F when expressed as a fraction in lowest terms -/
def denominator_numerator_difference : ℕ := 999 - 817

theorem repeating_decimal_proof :
  F = 817 / 999 ∧ denominator_numerator_difference = 182 :=
sorry

end NUMINAMATH_CALUDE_repeating_decimal_proof_l240_24040


namespace NUMINAMATH_CALUDE_arithmetic_equality_l240_24084

theorem arithmetic_equality : 1 - 0.2 + 0.03 - 0.004 = 0.826 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equality_l240_24084


namespace NUMINAMATH_CALUDE_problem_solution_l240_24082

theorem problem_solution (x y : ℝ) 
  (h1 : x = 153) 
  (h2 : x^3*y - 4*x^2*y + 4*x*y = 350064) : 
  y = 40/3967 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l240_24082


namespace NUMINAMATH_CALUDE_train_speed_calculation_l240_24001

theorem train_speed_calculation (t m s : ℝ) (ht : t > 0) (hm : m > 0) (hs : s > 0) :
  let m₁ := (Real.sqrt (t * m * (4 * s + t * m)) - t * m) / (2 * t)
  ∃ (t₁ : ℝ), t₁ > 0 ∧ m₁ * t₁ = s ∧ (m₁ + m) * (t₁ - t) = s :=
by sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l240_24001


namespace NUMINAMATH_CALUDE_first_player_winning_strategy_l240_24039

/-- Represents the number of points on the circle -/
def n : Nat := 98

/-- Represents the number of moves to connect n-2 points -/
def N (n : Nat) : Nat := (n - 3) * (n - 4) / 2

/-- Represents whether a number is odd -/
def isOdd (m : Nat) : Prop := ∃ k, m = 2 * k + 1

/-- Represents the winning condition for the first player -/
def firstPlayerWins (n : Nat) : Prop := isOdd (N n)

/-- Theorem stating that the first player has a winning strategy -/
theorem first_player_winning_strategy : firstPlayerWins n := by
  sorry

end NUMINAMATH_CALUDE_first_player_winning_strategy_l240_24039


namespace NUMINAMATH_CALUDE_redistribution_theorem_l240_24067

/-- The number of trucks after redistribution of oil containers -/
def num_trucks_after_redistribution : ℕ :=
  let initial_trucks_1 : ℕ := 7
  let boxes_per_truck_1 : ℕ := 20
  let initial_trucks_2 : ℕ := 5
  let boxes_per_truck_2 : ℕ := 12
  let containers_per_box : ℕ := 8
  let containers_per_truck_after : ℕ := 160
  let total_boxes : ℕ := initial_trucks_1 * boxes_per_truck_1 + initial_trucks_2 * boxes_per_truck_2
  let total_containers : ℕ := total_boxes * containers_per_box
  total_containers / containers_per_truck_after

theorem redistribution_theorem :
  num_trucks_after_redistribution = 10 :=
by sorry

end NUMINAMATH_CALUDE_redistribution_theorem_l240_24067


namespace NUMINAMATH_CALUDE_toms_barbados_trip_cost_l240_24022

/-- The total cost for Tom's trip to Barbados -/
def total_cost (num_vaccines : ℕ) (vaccine_cost : ℚ) (doctor_visit_cost : ℚ) 
                (insurance_coverage : ℚ) (trip_cost : ℚ) : ℚ :=
  let medical_cost := num_vaccines * vaccine_cost + doctor_visit_cost
  let out_of_pocket_medical := medical_cost * (1 - insurance_coverage)
  out_of_pocket_medical + trip_cost

/-- Theorem stating the total cost for Tom's trip to Barbados -/
theorem toms_barbados_trip_cost :
  total_cost 10 45 250 0.8 1200 = 1340 := by
  sorry

end NUMINAMATH_CALUDE_toms_barbados_trip_cost_l240_24022


namespace NUMINAMATH_CALUDE_conic_is_hyperbola_l240_24070

/-- The equation of a conic section -/
def conic_equation (x y : ℝ) : Prop :=
  (x - 7)^2 = 3*(4*y + 2)^2 - 108

/-- A hyperbola is characterized by having coefficients of x^2 and y^2 with opposite signs
    when the equation is in standard form -/
def is_hyperbola (eq : (ℝ → ℝ → Prop)) : Prop :=
  ∃ (a b c d e f : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ a*b < 0 ∧
    ∀ x y, eq x y ↔ a*x^2 + b*y^2 + c*x + d*y + e*x*y + f = 0

/-- The given conic equation represents a hyperbola -/
theorem conic_is_hyperbola : is_hyperbola conic_equation := by
  sorry

end NUMINAMATH_CALUDE_conic_is_hyperbola_l240_24070


namespace NUMINAMATH_CALUDE_ratio_equivalences_l240_24028

theorem ratio_equivalences (a b c d : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : d ≠ 0) 
  (h5 : a * d = b * c) : 
  (a / b = c / d) ∧ (b / a = d / c) ∧ (a / c = b / d) := by
  sorry

end NUMINAMATH_CALUDE_ratio_equivalences_l240_24028


namespace NUMINAMATH_CALUDE_clock_hands_angle_at_8_30_clock_hands_angle_at_8_30_is_75_l240_24019

/-- The angle between clock hands at 8:30 -/
theorem clock_hands_angle_at_8_30 : ℝ :=
  let hours : ℝ := 8
  let minutes : ℝ := 30
  let degrees_per_hour : ℝ := 360 / 12
  let degrees_per_minute : ℝ := 360 / 60
  let hour_hand_angle : ℝ := hours * degrees_per_hour + (minutes / 60) * degrees_per_hour
  let minute_hand_angle : ℝ := minutes * degrees_per_minute
  let angle_diff : ℝ := |hour_hand_angle - minute_hand_angle|
  75

/-- The theorem stating that the angle between clock hands at 8:30 is 75 degrees -/
theorem clock_hands_angle_at_8_30_is_75 : clock_hands_angle_at_8_30 = 75 := by
  sorry

end NUMINAMATH_CALUDE_clock_hands_angle_at_8_30_clock_hands_angle_at_8_30_is_75_l240_24019


namespace NUMINAMATH_CALUDE_square_of_binomial_b_value_l240_24054

theorem square_of_binomial_b_value (p b : ℝ) : 
  (∃ q : ℝ, ∀ x : ℝ, x^2 + p*x + b = (x + q)^2) → 
  p = -10 → 
  b = 25 := by
sorry

end NUMINAMATH_CALUDE_square_of_binomial_b_value_l240_24054


namespace NUMINAMATH_CALUDE_number_equation_solution_l240_24076

theorem number_equation_solution : 
  ∃ x : ℝ, x - (1002 / 20.04) = 2984 ∧ x = 3034 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l240_24076


namespace NUMINAMATH_CALUDE_pirate_catch_time_l240_24015

/-- Represents the pursuit problem between a pirate ship and a trading vessel -/
structure PursuitProblem where
  initial_distance : ℝ
  pirate_speed_initial : ℝ
  trader_speed : ℝ
  pursuit_start_time : ℝ
  speed_change_time : ℝ
  new_speed_ratio : ℝ

/-- Calculates the time at which the pirate ship catches the trading vessel -/
def catch_time (p : PursuitProblem) : ℝ :=
  sorry

/-- Theorem stating that the catch time for the given problem is 4:40 p.m. (16.67 hours) -/
theorem pirate_catch_time :
  let problem := PursuitProblem.mk 12 12 9 12 3 1.2
  catch_time problem = 16 + 2/3 :=
sorry

end NUMINAMATH_CALUDE_pirate_catch_time_l240_24015


namespace NUMINAMATH_CALUDE_angle_bisector_shorter_than_median_l240_24030

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the length of a line segment
def length (p q : ℝ × ℝ) : ℝ := sorry

-- Define the angle bisector
def angle_bisector (t : Triangle) : ℝ × ℝ := sorry

-- Define the median
def median (t : Triangle) : ℝ × ℝ := sorry

-- Theorem statement
theorem angle_bisector_shorter_than_median (t : Triangle) :
  length t.A t.B ≤ length t.A t.C →
  length t.A (angle_bisector t) ≤ length t.A (median t) ∧
  (length t.A (angle_bisector t) = length t.A (median t) ↔ length t.A t.B = length t.A t.C) :=
sorry

end NUMINAMATH_CALUDE_angle_bisector_shorter_than_median_l240_24030


namespace NUMINAMATH_CALUDE_solution_set_abs_inequality_l240_24017

theorem solution_set_abs_inequality (x : ℝ) :
  |2 - x| ≥ 1 ↔ x ≤ 1 ∨ x ≥ 3 := by sorry

end NUMINAMATH_CALUDE_solution_set_abs_inequality_l240_24017


namespace NUMINAMATH_CALUDE_f_2018_equals_1_l240_24077

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

theorem f_2018_equals_1
  (f : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_f0 : f 0 = -1)
  (h_fx : ∀ x, f x = -f (2 - x)) :
  f 2018 = 1 := by
  sorry

end NUMINAMATH_CALUDE_f_2018_equals_1_l240_24077


namespace NUMINAMATH_CALUDE_total_apples_is_eleven_l240_24046

/-- The number of apples Marin has -/
def marin_apples : ℕ := 9

/-- The number of apples Donald has -/
def donald_apples : ℕ := 2

/-- The total number of apples Marin and Donald have together -/
def total_apples : ℕ := marin_apples + donald_apples

/-- Proof that the total number of apples is 11 -/
theorem total_apples_is_eleven : total_apples = 11 := by
  sorry

end NUMINAMATH_CALUDE_total_apples_is_eleven_l240_24046


namespace NUMINAMATH_CALUDE_system_solution_l240_24088

theorem system_solution :
  let S := {(x, y) : ℝ × ℝ | x + y = 3 ∧ 2*x - 3*y = 1}
  S = {(2, 1)} := by sorry

end NUMINAMATH_CALUDE_system_solution_l240_24088


namespace NUMINAMATH_CALUDE_fuel_mixture_problem_l240_24025

/-- Proves that the amount of fuel A added to the tank is 106 gallons -/
theorem fuel_mixture_problem (tank_capacity : ℝ) (ethanol_a : ℝ) (ethanol_b : ℝ) (total_ethanol : ℝ) 
  (h1 : tank_capacity = 214)
  (h2 : ethanol_a = 0.12)
  (h3 : ethanol_b = 0.16)
  (h4 : total_ethanol = 30) :
  ∃ (fuel_a : ℝ), fuel_a = 106 ∧ 
    ethanol_a * fuel_a + ethanol_b * (tank_capacity - fuel_a) = total_ethanol :=
by sorry

end NUMINAMATH_CALUDE_fuel_mixture_problem_l240_24025


namespace NUMINAMATH_CALUDE_ST_SQ_ratio_is_930_2197_l240_24094

-- Define the points
variable (P Q R S T : ℝ × ℝ)

-- Define the triangles and their properties
def triangle_PQR_right_at_R : Prop := sorry
def PR_length : ℝ := 5
def RQ_length : ℝ := 12

def triangle_PQS_right_at_P : Prop := sorry
def PS_length : ℝ := 15

-- R and S are on opposite sides of PQ
def R_S_opposite_sides : Prop := sorry

-- Line through S parallel to PR meets RQ extended at T
def S_parallel_PR_meets_RQ_at_T : Prop := sorry

-- Define the ratio ST/SQ
def ST_SQ_ratio : ℝ := sorry

-- Theorem statement
theorem ST_SQ_ratio_is_930_2197 
  (h1 : triangle_PQR_right_at_R)
  (h2 : triangle_PQS_right_at_P)
  (h3 : R_S_opposite_sides)
  (h4 : S_parallel_PR_meets_RQ_at_T) :
  ST_SQ_ratio = 930 / 2197 := by sorry

end NUMINAMATH_CALUDE_ST_SQ_ratio_is_930_2197_l240_24094


namespace NUMINAMATH_CALUDE_jennys_bottle_cap_bounce_fraction_l240_24035

theorem jennys_bottle_cap_bounce_fraction :
  ∀ (jenny_initial : ℝ) (mark_initial : ℝ) (jenny_fraction : ℝ),
    jenny_initial = 18 →
    mark_initial = 15 →
    (mark_initial + 2 * mark_initial) - (jenny_initial + jenny_initial * jenny_fraction) = 21 →
    jenny_fraction = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_jennys_bottle_cap_bounce_fraction_l240_24035


namespace NUMINAMATH_CALUDE_solution_set_absolute_value_inequality_l240_24014

theorem solution_set_absolute_value_inequality :
  {x : ℝ | |x - 3| + |x - 5| ≥ 4} = {x : ℝ | x ≤ 2 ∨ x ≥ 6} := by sorry

end NUMINAMATH_CALUDE_solution_set_absolute_value_inequality_l240_24014


namespace NUMINAMATH_CALUDE_complement_P_inter_Q_l240_24043

open Set

def P : Set ℝ := { x | x - 1 ≤ 0 }
def Q : Set ℝ := { x | 0 < x ∧ x ≤ 2 }

theorem complement_P_inter_Q : (P.compl ∩ Q) = Ioo 1 2 := by sorry

end NUMINAMATH_CALUDE_complement_P_inter_Q_l240_24043


namespace NUMINAMATH_CALUDE_unique_perpendicular_line_l240_24050

/-- A plane in Euclidean geometry -/
structure EuclideanPlane :=
  (points : Type*)
  (lines : Type*)
  (on_line : points → lines → Prop)

/-- Definition of perpendicular lines in a plane -/
def perpendicular (p : EuclideanPlane) (l1 l2 : p.lines) : Prop :=
  sorry

/-- Statement: In a plane, given a line and a point not on the line,
    there exists a unique line passing through the point
    that is perpendicular to the given line -/
theorem unique_perpendicular_line
  (p : EuclideanPlane) (l : p.lines) (pt : p.points)
  (h : ¬ p.on_line pt l) :
  ∃! l' : p.lines, p.on_line pt l' ∧ perpendicular p l l' :=
sorry

end NUMINAMATH_CALUDE_unique_perpendicular_line_l240_24050


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l240_24009

/-- The perimeter of a rhombus given its diagonals -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 24) (h2 : d2 = 10) : 
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 52 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l240_24009


namespace NUMINAMATH_CALUDE_A_power_101_l240_24027

def A : Matrix (Fin 3) (Fin 3) ℕ := !![0, 0, 1; 1, 0, 0; 0, 1, 0]

theorem A_power_101 : A ^ 101 = A ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_A_power_101_l240_24027


namespace NUMINAMATH_CALUDE_negation_of_implication_l240_24029

theorem negation_of_implication (P Q : Prop) :
  ¬(P → Q) ↔ (¬P → ¬Q) := by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l240_24029


namespace NUMINAMATH_CALUDE_center_of_symmetry_condition_l240_24069

/-- A point A(a, b) is a center of symmetry for a function f if and only if
    for all x, f(a-x) + f(a+x) = 2b -/
theorem center_of_symmetry_condition (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x y : ℝ, f x = y → f (2*a - x) = 2*b - y) ↔
  (∀ x : ℝ, f (a-x) + f (a+x) = 2*b) :=
sorry

end NUMINAMATH_CALUDE_center_of_symmetry_condition_l240_24069


namespace NUMINAMATH_CALUDE_expand_expression_l240_24072

theorem expand_expression (x y : ℝ) : 
  5 * (3 * x^2 * y - 4 * x * y^2 + 2 * y^3) = 15 * x^2 * y - 20 * x * y^2 + 10 * y^3 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l240_24072


namespace NUMINAMATH_CALUDE_max_interval_length_l240_24083

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

-- State the theorem
theorem max_interval_length 
  (a b : ℝ) 
  (h1 : a ≤ b) 
  (h2 : ∀ x ∈ Set.Icc a b, -3 ≤ f x ∧ f x ≤ 1) 
  (h3 : ∃ x ∈ Set.Icc a b, f x = -3) 
  (h4 : ∃ x ∈ Set.Icc a b, f x = 1) :
  b - a ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_max_interval_length_l240_24083


namespace NUMINAMATH_CALUDE_unique_quadratic_solution_l240_24013

theorem unique_quadratic_solution (a c : ℝ) : 
  (∃! x, a * x^2 + 20 * x + c = 0) →  -- exactly one solution
  (a + c = 29) →                      -- sum condition
  (a < c) →                           -- order condition
  (a = 4 ∧ c = 25) := by              -- conclusion
sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_l240_24013


namespace NUMINAMATH_CALUDE_common_divisors_9240_8820_l240_24018

theorem common_divisors_9240_8820 : 
  (Nat.divisors (Nat.gcd 9240 8820)).card = 24 := by sorry

end NUMINAMATH_CALUDE_common_divisors_9240_8820_l240_24018


namespace NUMINAMATH_CALUDE_triangle_side_length_l240_24008

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) :
  A = 60 * π / 180 →
  b = 1 →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 →
  a^2 = b^2 + c^2 - 2*b*c*(Real.cos A) →
  a = Real.sqrt 13 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l240_24008


namespace NUMINAMATH_CALUDE_debate_team_combinations_l240_24047

theorem debate_team_combinations (n : ℕ) (k : ℕ) : n = 7 → k = 4 → Nat.choose n k = 35 := by
  sorry

end NUMINAMATH_CALUDE_debate_team_combinations_l240_24047


namespace NUMINAMATH_CALUDE_bert_spending_l240_24058

theorem bert_spending (n : ℝ) : 
  (3/4 * n - 9) / 2 = 12 → n = 44 := by sorry

end NUMINAMATH_CALUDE_bert_spending_l240_24058


namespace NUMINAMATH_CALUDE_subtraction_of_negatives_l240_24099

theorem subtraction_of_negatives : (-2) - (-4) = 2 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_negatives_l240_24099


namespace NUMINAMATH_CALUDE_largest_integer_with_remainder_l240_24031

theorem largest_integer_with_remainder : ∃ n : ℕ, 
  (n < 100) ∧ 
  (n % 6 = 4) ∧ 
  (∀ m : ℕ, m < 100 → m % 6 = 4 → m ≤ n) ∧
  (n = 94) :=
sorry

end NUMINAMATH_CALUDE_largest_integer_with_remainder_l240_24031


namespace NUMINAMATH_CALUDE_fourth_intersection_point_l240_24041

/-- Given a hyperbola xy = 2 and three points on this curve that also lie on a circle,
    prove that the fourth intersection point has specific coordinates. -/
theorem fourth_intersection_point (P₁ P₂ P₃ P₄ : ℝ × ℝ) : 
  P₁.1 * P₁.2 = 2 ∧ P₂.1 * P₂.2 = 2 ∧ P₃.1 * P₃.2 = 2 ∧ P₄.1 * P₄.2 = 2 →
  P₁ = (3, 2/3) ∧ P₂ = (-4, -1/2) ∧ P₃ = (1/2, 4) →
  ∃ (a b r : ℝ), 
    (P₁.1 - a)^2 + (P₁.2 - b)^2 = r^2 ∧
    (P₂.1 - a)^2 + (P₂.2 - b)^2 = r^2 ∧
    (P₃.1 - a)^2 + (P₃.2 - b)^2 = r^2 ∧
    (P₄.1 - a)^2 + (P₄.2 - b)^2 = r^2 →
  P₄ = (-2/3, -3) := by
sorry

end NUMINAMATH_CALUDE_fourth_intersection_point_l240_24041


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l240_24026

theorem sufficient_not_necessary (x : ℝ) :
  (∀ x, 1 < x ∧ x < 2 → (x - 2)^2 < 1) ∧
  (∃ x, (x - 2)^2 < 1 ∧ ¬(1 < x ∧ x < 2)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l240_24026


namespace NUMINAMATH_CALUDE_additional_bags_capacity_plane_capacity_proof_l240_24064

/-- Calculates the number of additional maximum-weight bags an airplane can hold -/
theorem additional_bags_capacity 
  (num_people : ℕ) 
  (bags_per_person : ℕ) 
  (bag_weight : ℕ) 
  (plane_capacity : ℕ) : ℕ :=
  let total_bags := num_people * bags_per_person
  let total_weight := total_bags * bag_weight
  let remaining_capacity := plane_capacity - total_weight
  remaining_capacity / bag_weight

/-- Proves that given the specific conditions, the plane can hold 90 more bags -/
theorem plane_capacity_proof :
  additional_bags_capacity 6 5 50 6000 = 90 := by
  sorry

end NUMINAMATH_CALUDE_additional_bags_capacity_plane_capacity_proof_l240_24064


namespace NUMINAMATH_CALUDE_even_sum_probability_l240_24090

/-- Represents a die with faces numbered from 1 to n -/
structure Die (n : ℕ) where
  faces : Finset ℕ
  face_count : faces.card = n
  valid_faces : ∀ x ∈ faces, 1 ≤ x ∧ x ≤ n

/-- A regular die with faces numbered from 1 to 6 -/
def regular_die : Die 6 := {
  faces := Finset.range 6,
  face_count := sorry,
  valid_faces := sorry
}

/-- An odd-numbered die with faces 1, 3, 5, 7, 9, 11 -/
def odd_die : Die 6 := {
  faces := Finset.range 6,
  face_count := sorry,
  valid_faces := sorry
}

/-- The probability of an event occurring -/
def probability (event : Prop) : ℚ := sorry

/-- The sum of the top faces of three dice -/
def dice_sum (d1 d2 : Die 6) (d3 : Die 6) : ℕ := sorry

/-- The statement to be proved -/
theorem even_sum_probability :
  probability (∃ (r1 r2 : Die 6) (o : Die 6), 
    r1 = regular_die ∧ 
    r2 = regular_die ∧ 
    o = odd_die ∧ 
    Even (dice_sum r1 r2 o)) = 1/2 := sorry

end NUMINAMATH_CALUDE_even_sum_probability_l240_24090


namespace NUMINAMATH_CALUDE_triangle_side_sum_bound_l240_24081

/-- Given a triangle ABC with side lengths a, b, and c, where c = 2 and the dot product 
    of vectors AC and AB is equal to b² - (1/2)ab, prove that 2 < a + b ≤ 4 -/
theorem triangle_side_sum_bound (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  let c : ℝ := 2
  let dot_product : ℝ := b^2 - (1/2) * a * b
  2 < a + b ∧ a + b ≤ 4 := by sorry

end NUMINAMATH_CALUDE_triangle_side_sum_bound_l240_24081


namespace NUMINAMATH_CALUDE_inequality_proof_l240_24042

theorem inequality_proof (x y z : ℝ) (hx : 0 < x ∧ x < π/2) (hy : 0 < y ∧ y < π/2) (hz : 0 < z ∧ z < π/2) :
  (x * Real.cos x + y * Real.cos y + z * Real.cos z) / (x + y + z) ≤ (Real.cos x + Real.cos y + Real.cos z) / 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l240_24042


namespace NUMINAMATH_CALUDE_laptop_price_difference_l240_24073

/-- The list price of Laptop Y -/
def list_price : ℝ := 69.80

/-- The discount percentage at Tech Giant -/
def tech_giant_discount : ℝ := 0.15

/-- The fixed discount amount at EconoTech -/
def econotech_discount : ℝ := 10

/-- The sale price at Tech Giant -/
def tech_giant_price : ℝ := list_price * (1 - tech_giant_discount)

/-- The sale price at EconoTech -/
def econotech_price : ℝ := list_price - econotech_discount

/-- The price difference between EconoTech and Tech Giant in dollars -/
def price_difference : ℝ := econotech_price - tech_giant_price

theorem laptop_price_difference :
  ⌊price_difference * 100⌋ = 47 := by sorry

end NUMINAMATH_CALUDE_laptop_price_difference_l240_24073


namespace NUMINAMATH_CALUDE_expression_evaluation_l240_24006

theorem expression_evaluation : 
  let x : ℚ := -1/4
  (2*x + 1) * (2*x - 1) - (x - 2)^2 - 3*x^2 = -6 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l240_24006


namespace NUMINAMATH_CALUDE_abnormal_segregation_in_secondary_spermatocyte_l240_24038

-- Define the alleles
inductive Allele
| E  -- normal eye
| e  -- eyeless

-- Define a genotype as a list of alleles
def Genotype := List Allele

-- Define the parents' genotypes
def male_parent : Genotype := [Allele.E, Allele.e]
def female_parent : Genotype := [Allele.e, Allele.e]

-- Define the offspring's genotype
def offspring : Genotype := [Allele.E, Allele.E, Allele.e]

-- Define the possible cell types where segregation could have occurred abnormally
inductive CellType
| PrimarySpermatocyte
| PrimaryOocyte
| SecondarySpermatocyte
| SecondaryOocyte

-- Define the property of no crossing over
def no_crossing_over : Prop := sorry

-- Define the dominance of E over e
def E_dominant_over_e : Prop := sorry

-- Theorem statement
theorem abnormal_segregation_in_secondary_spermatocyte :
  E_dominant_over_e →
  no_crossing_over →
  (∃ (abnormal_cell : CellType), 
    abnormal_cell = CellType.SecondarySpermatocyte ∧
    (∀ (other_cell : CellType), 
      other_cell ≠ CellType.SecondarySpermatocyte → 
      ¬(offspring = [Allele.E, Allele.E, Allele.e]))) :=
sorry

end NUMINAMATH_CALUDE_abnormal_segregation_in_secondary_spermatocyte_l240_24038


namespace NUMINAMATH_CALUDE_franks_trivia_score_l240_24098

/-- Frank's trivia game score calculation -/
theorem franks_trivia_score :
  ∀ (first_half second_half points_per_question : ℕ),
    first_half = 3 →
    second_half = 2 →
    points_per_question = 3 →
    (first_half + second_half) * points_per_question = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_franks_trivia_score_l240_24098


namespace NUMINAMATH_CALUDE_arithmetic_sequence_cos_relation_l240_24060

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

theorem arithmetic_sequence_cos_relation (a : ℕ → ℝ) :
  arithmetic_sequence a → a 1 + a 5 + a 9 = 8 * Real.pi → Real.cos (a 3 + a 7) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_cos_relation_l240_24060


namespace NUMINAMATH_CALUDE_min_perimeter_triangle_l240_24020

/-- Represents a triangle with integer side lengths -/
structure Triangle where
  a : ℕ
  b : ℕ
  c : ℕ
  sides_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c

/-- Represents a circle -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents an excircle of a triangle -/
structure Excircle where
  center : ℝ × ℝ
  radius : ℝ

/-- The incenter of a triangle -/
def incenter (t : Triangle) : ℝ × ℝ := sorry

/-- The incircle of a triangle -/
def incircle (t : Triangle) : Circle := sorry

/-- The excircles of a triangle -/
def excircles (t : Triangle) : Fin 3 → Excircle := sorry

/-- Checks if two circles are internally tangent -/
def is_internally_tangent (c1 c2 : Circle) : Prop := sorry

/-- The main theorem -/
theorem min_perimeter_triangle (t : Triangle) 
  (h1 : is_internally_tangent (incircle t) { center := (excircles t 0).center, radius := (excircles t 0).radius })
  (h2 : ¬ is_internally_tangent (incircle t) { center := (excircles t 1).center, radius := (excircles t 1).radius })
  (h3 : ¬ is_internally_tangent (incircle t) { center := (excircles t 2).center, radius := (excircles t 2).radius }) :
  t.a + t.b + t.c ≥ 12 := by
  sorry

end NUMINAMATH_CALUDE_min_perimeter_triangle_l240_24020


namespace NUMINAMATH_CALUDE_tangent_lines_to_unit_circle_l240_24010

/-- The equation of a circle with radius 1 centered at the origin -/
def unitCircle (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- A point on the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The equation of a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A line is tangent to the unit circle at a given point -/
def isTangent (l : Line) (p : Point) : Prop :=
  unitCircle p.x p.y ∧
  l.a * p.x + l.b * p.y + l.c = 0 ∧
  ∀ (x y : ℝ), unitCircle x y → (l.a * x + l.b * y + l.c = 0 → x = p.x ∧ y = p.y)

theorem tangent_lines_to_unit_circle :
  let p1 : Point := ⟨-1, 0⟩
  let p2 : Point := ⟨-1, 2⟩
  let l1 : Line := ⟨1, 0, 1⟩  -- Represents x = -1
  let l2 : Line := ⟨3, 4, -5⟩  -- Represents 3x + 4y - 5 = 0
  isTangent l1 p1 ∧ isTangent l2 p2 := by sorry

end NUMINAMATH_CALUDE_tangent_lines_to_unit_circle_l240_24010


namespace NUMINAMATH_CALUDE_fifth_term_of_sequence_l240_24059

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

theorem fifth_term_of_sequence (a₁ d : ℤ) :
  arithmetic_sequence a₁ d 20 = 12 →
  arithmetic_sequence a₁ d 21 = 15 →
  arithmetic_sequence a₁ d 5 = -33 :=
by sorry

end NUMINAMATH_CALUDE_fifth_term_of_sequence_l240_24059


namespace NUMINAMATH_CALUDE_tank_height_problem_l240_24012

/-- Given two right circular cylinders A and B, where A has a circumference of 6 meters,
    B has a height of 6 meters and a circumference of 10 meters, and A's capacity is 60% of B's capacity,
    prove that the height of A is 10 meters. -/
theorem tank_height_problem (h_A : ℝ) : 
  let r_A : ℝ := 3 / Real.pi
  let r_B : ℝ := 5 / Real.pi
  let volume_A : ℝ := Real.pi * r_A^2 * h_A
  let volume_B : ℝ := Real.pi * r_B^2 * 6
  volume_A = 0.6 * volume_B → h_A = 10 := by
  sorry

#check tank_height_problem

end NUMINAMATH_CALUDE_tank_height_problem_l240_24012


namespace NUMINAMATH_CALUDE_slope_theorem_l240_24000

/-- Given two points R(-3,9) and S(3,y) on a coordinate plane, 
    if the slope of the line through R and S is -2, then y = -3. -/
theorem slope_theorem (y : ℝ) : 
  let R : ℝ × ℝ := (-3, 9)
  let S : ℝ × ℝ := (3, y)
  let slope := (S.2 - R.2) / (S.1 - R.1)
  slope = -2 → y = -3 := by
sorry

end NUMINAMATH_CALUDE_slope_theorem_l240_24000


namespace NUMINAMATH_CALUDE_tax_calculation_correct_l240_24044

/-- Represents the tax rate for a given income range -/
structure TaxBracket where
  lower : ℝ
  upper : Option ℝ
  rate : ℝ

/-- Calculates the tax for a given taxable income -/
def calculateTax (brackets : List TaxBracket) (taxableIncome : ℝ) : ℝ :=
  sorry

/-- Represents the tax system with its parameters -/
structure TaxSystem where
  threshold : ℝ
  brackets : List TaxBracket
  elderlyDeduction : ℝ

/-- Calculates the after-tax income given a pre-tax income and tax system -/
def afterTaxIncome (preTaxIncome : ℝ) (system : TaxSystem) : ℝ :=
  sorry

theorem tax_calculation_correct (preTaxIncome : ℝ) (system : TaxSystem) :
  let taxPaid := 180
  let afterTax := 9720
  system.threshold = 5000 ∧
  system.elderlyDeduction = 1000 ∧
  system.brackets = [
    ⟨0, some 3000, 0.03⟩,
    ⟨3000, some 12000, 0.10⟩,
    ⟨12000, some 25000, 0.20⟩,
    ⟨25000, none, 0.25⟩
  ] →
  calculateTax system.brackets (preTaxIncome - system.threshold - system.elderlyDeduction) = taxPaid ∧
  afterTaxIncome preTaxIncome system = afterTax :=
by sorry

end NUMINAMATH_CALUDE_tax_calculation_correct_l240_24044


namespace NUMINAMATH_CALUDE_cats_remaining_after_sale_l240_24045

theorem cats_remaining_after_sale (siamese : ℕ) (house : ℕ) (sold : ℕ) : 
  siamese = 13 → house = 5 → sold = 10 → siamese + house - sold = 8 := by
sorry

end NUMINAMATH_CALUDE_cats_remaining_after_sale_l240_24045


namespace NUMINAMATH_CALUDE_maximize_expression_l240_24034

def a : Set Int := {-3, -2, -1, 0, 1, 2, 3}

theorem maximize_expression (v : ℝ) :
  ∃ (x y z : Int),
    x ∈ a ∧ y ∈ a ∧ z ∈ a ∧
    (∀ (x' y' z' : Int), x' ∈ a → y' ∈ a → z' ∈ a → v * x' - y' * z' ≤ v * x - y * z) ∧
    v * x - y * z = 15 ∧
    y = -3 ∧ z = 3 :=
sorry

end NUMINAMATH_CALUDE_maximize_expression_l240_24034
