import Mathlib

namespace mandy_toys_count_l3239_323939

theorem mandy_toys_count (mandy anna amanda : ℕ) 
  (h1 : anna = 3 * mandy)
  (h2 : amanda = anna + 2)
  (h3 : mandy + anna + amanda = 142) :
  mandy = 20 := by
sorry

end mandy_toys_count_l3239_323939


namespace square_root_equation_l3239_323995

theorem square_root_equation (n : ℝ) : Real.sqrt (10 + n) = 8 → n = 54 := by
  sorry

end square_root_equation_l3239_323995


namespace sum_of_squares_l3239_323974

theorem sum_of_squares (x y z : ℝ) 
  (eq1 : x^2 + 3*y = 20)
  (eq2 : y^2 + 5*z = -20)
  (eq3 : z^2 + 7*x = -34) :
  x^2 + y^2 + z^2 = 20.75 := by
sorry

end sum_of_squares_l3239_323974


namespace integer_solution_inequality_l3239_323980

theorem integer_solution_inequality (x : ℤ) : 
  (3 * |2 * x + 1| + 6 < 24) ↔ x ∈ ({-3, -2, -1, 0, 1, 2} : Set ℤ) := by
  sorry

end integer_solution_inequality_l3239_323980


namespace total_pages_read_l3239_323909

/-- Represents the number of pages in each chapter of the book --/
def pages_per_chapter : ℕ := 40

/-- Represents the number of chapters Mitchell read before 4 o'clock --/
def chapters_before_4 : ℕ := 10

/-- Represents the number of pages Mitchell read from the 11th chapter at 4 o'clock --/
def pages_at_4 : ℕ := 20

/-- Represents the number of additional chapters Mitchell read after 4 o'clock --/
def chapters_after_4 : ℕ := 2

/-- Theorem stating that the total number of pages Mitchell read is 500 --/
theorem total_pages_read : 
  pages_per_chapter * chapters_before_4 + 
  pages_at_4 + 
  pages_per_chapter * chapters_after_4 = 500 := by
sorry

end total_pages_read_l3239_323909


namespace arithmetic_sequence_150th_term_l3239_323926

/-- The nth term of an arithmetic sequence -/
def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

/-- The 150th term of the specific arithmetic sequence -/
def term_150 : ℝ :=
  arithmetic_sequence 3 4 150

theorem arithmetic_sequence_150th_term :
  term_150 = 599 := by sorry

end arithmetic_sequence_150th_term_l3239_323926


namespace rectangle_from_isosceles_120_l3239_323901

/-- An isosceles triangle with a vertex angle of 120 degrees --/
structure IsoscelesTriangle120 where
  -- We represent the triangle by its side lengths
  base : ℝ
  leg : ℝ
  base_positive : 0 < base
  leg_positive : 0 < leg
  vertex_angle : Real.cos (120 * π / 180) = (base^2 - 2 * leg^2) / (2 * leg^2)

/-- A rectangle formed by isosceles triangles --/
structure RectangleFromTriangles where
  width : ℝ
  height : ℝ
  triangles : List IsoscelesTriangle120
  width_positive : 0 < width
  height_positive : 0 < height

/-- Theorem stating that it's possible to form a rectangle from isosceles triangles with 120° vertex angle --/
theorem rectangle_from_isosceles_120 : 
  ∃ (r : RectangleFromTriangles), r.triangles.length > 0 :=
sorry

end rectangle_from_isosceles_120_l3239_323901


namespace grass_weeds_count_l3239_323908

/-- Represents the number of weeds in different areas of the garden -/
structure GardenWeeds where
  flower_bed : ℕ
  vegetable_patch : ℕ
  grass : ℕ

/-- Represents Lucille's earnings and expenses -/
structure LucilleFinances where
  cents_per_weed : ℕ
  soda_cost : ℕ
  remaining_cents : ℕ

def calculate_grass_weeds (garden : GardenWeeds) (finances : LucilleFinances) : ℕ :=
  garden.grass

theorem grass_weeds_count 
  (garden : GardenWeeds) 
  (finances : LucilleFinances) 
  (h1 : garden.flower_bed = 11)
  (h2 : garden.vegetable_patch = 14)
  (h3 : finances.cents_per_weed = 6)
  (h4 : finances.soda_cost = 99)
  (h5 : finances.remaining_cents = 147)
  : calculate_grass_weeds garden finances = 32 := by
  sorry

#eval calculate_grass_weeds 
  { flower_bed := 11, vegetable_patch := 14, grass := 32 } 
  { cents_per_weed := 6, soda_cost := 99, remaining_cents := 147 }

end grass_weeds_count_l3239_323908


namespace fifth_derivative_y_l3239_323985

noncomputable def y (x : ℝ) : ℝ := (2 * x^2 - 7) * Real.log (x - 1)

theorem fifth_derivative_y (x : ℝ) (h : x ≠ 1) :
  (deriv^[5] y) x = 8 * (x^2 - 5*x - 11) / (x - 1)^5 :=
by sorry

end fifth_derivative_y_l3239_323985


namespace intersection_distance_to_side_l3239_323997

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square -/
structure Square where
  sideLength : ℝ
  A : Point
  B : Point
  C : Point
  D : Point

/-- Represents a circle -/
structure Circle where
  center : Point
  radius : ℝ

def intersectionPoint (c1 c2 : Circle) : Point := sorry

/-- Calculates the distance between a point and a line defined by y = k -/
def distanceToHorizontalLine (p : Point) (k : ℝ) : ℝ := sorry

theorem intersection_distance_to_side (s : Square) 
  (c1 c2 : Circle) (h1 : s.sideLength = 10) 
  (h2 : c1.center = s.A) (h3 : c2.center = s.B) 
  (h4 : c1.radius = 5) (h5 : c2.radius = 5) :
  let X := intersectionPoint c1 c2
  distanceToHorizontalLine X s.sideLength = 10 := by sorry

end intersection_distance_to_side_l3239_323997


namespace hcf_of_ratio_and_lcm_l3239_323910

theorem hcf_of_ratio_and_lcm (a b : ℕ+) : 
  (a : ℚ) / b = 4 / 5 → 
  Nat.lcm a b = 80 → 
  Nat.gcd a b = 2 := by
sorry

end hcf_of_ratio_and_lcm_l3239_323910


namespace midpoint_coordinate_sum_l3239_323976

/-- The sum of the coordinates of the midpoint of a segment with endpoints (8, 16) and (2, -8) is 9 -/
theorem midpoint_coordinate_sum : 
  let x1 : ℝ := 8
  let y1 : ℝ := 16
  let x2 : ℝ := 2
  let y2 : ℝ := -8
  let midpoint_x : ℝ := (x1 + x2) / 2
  let midpoint_y : ℝ := (y1 + y2) / 2
  midpoint_x + midpoint_y = 9 := by
sorry

end midpoint_coordinate_sum_l3239_323976


namespace calculate_difference_l3239_323943

theorem calculate_difference (x y z : ℝ) (hx : x = 40) (hy : y = 20) (hz : z = 5) :
  0.8 * (3 * (2 * x))^2 - 0.8 * Real.sqrt ((y / 4)^3 * z^3) = 45980 := by
  sorry

end calculate_difference_l3239_323943


namespace product_expansion_l3239_323968

theorem product_expansion (x : ℝ) : (2 + 3 * x) * (-2 + 3 * x) = 9 * x^2 - 4 := by
  sorry

end product_expansion_l3239_323968


namespace fraction_of_savings_used_for_bills_l3239_323941

def weekly_savings : ℚ := 25
def weeks_saved : ℕ := 6
def dad_contribution : ℚ := 70
def coat_cost : ℚ := 170

theorem fraction_of_savings_used_for_bills :
  let total_savings := weekly_savings * weeks_saved
  let remaining_cost := coat_cost - dad_contribution
  let amount_for_bills := total_savings - remaining_cost
  amount_for_bills / total_savings = 1 / 3 := by
sorry

end fraction_of_savings_used_for_bills_l3239_323941


namespace car_cyclist_problem_solution_l3239_323913

/-- Represents the speeds and meeting point of a car and cyclist problem -/
structure CarCyclistProblem where
  car_speed : ℝ
  cyclist_speed : ℝ
  meeting_distance_from_A : ℝ

/-- Checks if the given speeds and meeting point satisfy the problem conditions -/
def is_valid_solution (p : CarCyclistProblem) : Prop :=
  let total_distance := 80
  let time_to_meet := 1.5
  let distance_after_one_hour := 24
  let car_distance_one_hour := p.car_speed
  let cyclist_distance_one_hour := p.cyclist_speed
  let car_total_distance := p.car_speed * time_to_meet
  let cyclist_total_distance := p.cyclist_speed * 1.25  -- Cyclist rests for 1 hour

  -- Condition 1: After one hour, they are 24 km apart
  (total_distance - (car_distance_one_hour + cyclist_distance_one_hour) = distance_after_one_hour) ∧
  -- Condition 2: They meet after 90 minutes
  (car_total_distance + cyclist_total_distance = total_distance) ∧
  -- Condition 3: Meeting point is correct
  (p.meeting_distance_from_A = car_total_distance)

/-- The theorem stating that the given solution satisfies the problem conditions -/
theorem car_cyclist_problem_solution :
  is_valid_solution ⟨40, 16, 60⟩ := by
  sorry

end car_cyclist_problem_solution_l3239_323913


namespace remainder_of_1531_base12_div_8_l3239_323916

/-- Represents a base-12 number as a list of digits (least significant first) -/
def Base12 := List Nat

/-- Converts a base-12 number to base-10 -/
def toBase10 (n : Base12) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (12 ^ i)) 0

/-- The base-12 number 1531 -/
def num : Base12 := [1, 3, 5, 1]

theorem remainder_of_1531_base12_div_8 :
  toBase10 num % 8 = 5 := by
  sorry

end remainder_of_1531_base12_div_8_l3239_323916


namespace equivalence_sqrt_and_fraction_l3239_323929

theorem equivalence_sqrt_and_fraction (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (Real.sqrt a + 1 > Real.sqrt b) ↔ 
  (∀ x > 1, a * x + x / (x - 1) > b) :=
by sorry

end equivalence_sqrt_and_fraction_l3239_323929


namespace zoey_reading_schedule_l3239_323965

def days_to_read (n : ℕ) : ℕ := n + 1

def total_days (n : ℕ) : ℕ := 
  n * (2 * 2 + (n - 1) * 1) / 2

def weekday (start_day : ℕ) (days_passed : ℕ) : ℕ := 
  (start_day + days_passed) % 7

theorem zoey_reading_schedule :
  let number_of_books : ℕ := 20
  let friday : ℕ := 5
  (total_days number_of_books = 230) ∧ 
  (weekday friday (total_days number_of_books) = 4) := by
  sorry

#check zoey_reading_schedule

end zoey_reading_schedule_l3239_323965


namespace dilation_complex_mapping_l3239_323969

theorem dilation_complex_mapping :
  let center : ℂ := 2 - 3*I
  let scale_factor : ℝ := 3
  let original : ℂ := (5 - 4*I) / 3
  let image : ℂ := 1 + 2*I
  (image - center) = scale_factor • (original - center) := by sorry

end dilation_complex_mapping_l3239_323969


namespace range_of_m_l3239_323972

theorem range_of_m (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_eq : x + 2 * y = 4)
  (h_solution : ∃ m : ℝ, m^2 + (1/3) * m > 2/x + 1/(y+1)) :
  ∃ m : ℝ, (m < -4/3 ∨ m > 1) ∧ m^2 + (1/3) * m > 2/x + 1/(y+1) :=
by sorry

end range_of_m_l3239_323972


namespace gcd_7_factorial_5_factorial_squared_l3239_323988

theorem gcd_7_factorial_5_factorial_squared : Nat.gcd (Nat.factorial 7) ((Nat.factorial 5)^2) = 720 := by
  sorry

end gcd_7_factorial_5_factorial_squared_l3239_323988


namespace remaining_money_for_seat_and_tape_l3239_323964

def initial_amount : ℕ := 60
def frame_cost : ℕ := 15
def wheel_cost : ℕ := 25

theorem remaining_money_for_seat_and_tape :
  initial_amount - (frame_cost + wheel_cost) = 20 := by
  sorry

end remaining_money_for_seat_and_tape_l3239_323964


namespace three_tangent_lines_m_values_l3239_323986

-- Define the function f
def f (x : ℝ) : ℝ := -x^3 + 2*x^2 - 3*x

-- Define the derivative of f
def f' (x : ℝ) : ℝ := -3*x^2 + 4*x - 3

-- Define the tangent line equation
def tangent_line (x₀ : ℝ) (x : ℝ) : ℝ := 
  f x₀ + f' x₀ * (x - x₀)

-- Define the condition for a point to be on the tangent line
def on_tangent_line (x₀ m : ℝ) : Prop :=
  m = tangent_line x₀ (-1)

-- Theorem statement
theorem three_tangent_lines_m_values :
  ∀ m : ℤ, (∃ x₁ x₂ x₃ : ℝ, 
    x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    on_tangent_line x₁ m ∧
    on_tangent_line x₂ m ∧
    on_tangent_line x₃ m) →
  m = 4 ∨ m = 5 :=
sorry

end three_tangent_lines_m_values_l3239_323986


namespace complex_number_problem_l3239_323927

def i : ℂ := Complex.I

theorem complex_number_problem (z : ℂ) (h : (1 + 2*i)*z = 3 - 4*i) : 
  z.im = -2 ∧ Complex.abs z = Real.sqrt 5 := by
  sorry

end complex_number_problem_l3239_323927


namespace no_zeros_of_g_l3239_323966

/-- Given a differentiable function f: ℝ → ℝ such that f'(x) + f(x)/x > 0 for all x ≠ 0,
    the function g(x) = f(x) + 1/x has no zeros. -/
theorem no_zeros_of_g (f : ℝ → ℝ) (hf : Differentiable ℝ f)
    (h : ∀ x ≠ 0, deriv f x + f x / x > 0) :
    ∀ x ≠ 0, f x + 1 / x ≠ 0 := by
  sorry

end no_zeros_of_g_l3239_323966


namespace planes_parallel_if_perpendicular_to_same_line_l3239_323998

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (intersect : Line → Line → Prop)
variable (intersectPlanes : Plane → Plane → Prop)

-- Theorem statement
theorem planes_parallel_if_perpendicular_to_same_line
  (m n : Line) (α β : Plane)
  (h1 : ¬ intersect m n)
  (h2 : ¬ intersectPlanes α β)
  (h3 : perpendicular m α)
  (h4 : perpendicular m β) :
  parallel α β :=
sorry

end planes_parallel_if_perpendicular_to_same_line_l3239_323998


namespace odot_calculation_l3239_323920

def odot (a b : ℝ) : ℝ := a * b + (a - b)

theorem odot_calculation : odot (odot 3 2) 4 = 31 := by
  sorry

end odot_calculation_l3239_323920


namespace log_equation_solution_l3239_323973

theorem log_equation_solution (k x : ℝ) (h : k > 0) (h' : x > 0) :
  Real.log x / Real.log k * Real.log k / Real.log 5 = 3 → x = 125 := by
  sorry

end log_equation_solution_l3239_323973


namespace sum_of_solutions_l3239_323946

theorem sum_of_solutions : ∃ (S : Finset ℕ), 
  (∀ x ∈ S, x > 0 ∧ x ≤ 30 ∧ (17*(4*x - 3) % 10 = 34 % 10)) ∧
  (∀ x : ℕ, x > 0 ∧ x ≤ 30 ∧ (17*(4*x - 3) % 10 = 34 % 10) → x ∈ S) ∧
  (Finset.sum S id = 51) := by
sorry

end sum_of_solutions_l3239_323946


namespace unit_digit_of_4137_to_754_l3239_323990

theorem unit_digit_of_4137_to_754 : (4137^754) % 10 = 9 := by
  sorry

end unit_digit_of_4137_to_754_l3239_323990


namespace min_value_problem_l3239_323931

theorem min_value_problem (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : 1 / (a + 3) + 1 / (b + 3) + 1 / (c + 3) = 1 / 4) :
  22.75 ≤ a + 3 * b + 2 * c ∧ ∃ (a₀ b₀ c₀ : ℝ), 
    0 < a₀ ∧ 0 < b₀ ∧ 0 < c₀ ∧
    1 / (a₀ + 3) + 1 / (b₀ + 3) + 1 / (c₀ + 3) = 1 / 4 ∧
    a₀ + 3 * b₀ + 2 * c₀ = 22.75 :=
by sorry

end min_value_problem_l3239_323931


namespace tencent_dialectical_materialism_alignment_l3239_323983

/-- Represents the principles of dialectical materialism -/
structure DialecticalMaterialism where
  dialectical_negation : Bool
  innovation : Bool
  development : Bool
  unity_of_opposites : Bool
  unity_of_progressiveness_and_tortuosity : Bool
  unity_of_quantitative_and_qualitative_changes : Bool

/-- Represents Tencent's development characteristics -/
structure TencentDevelopment where
  technological_innovation : Bool
  continuous_growth : Bool
  overcoming_difficulties : Bool
  qualitative_leaps : Bool

/-- Given information about Tencent's development history -/
axiom tencent_history : TencentDevelopment

/-- Theorem stating that Tencent's development aligns with dialectical materialism -/
theorem tencent_dialectical_materialism_alignment :
  ∃ (dm : DialecticalMaterialism),
    dm.dialectical_negation ∧
    dm.innovation ∧
    dm.development ∧
    dm.unity_of_opposites ∧
    dm.unity_of_progressiveness_and_tortuosity ∧
    dm.unity_of_quantitative_and_qualitative_changes ∧
    tencent_history.technological_innovation ∧
    tencent_history.continuous_growth ∧
    tencent_history.overcoming_difficulties ∧
    tencent_history.qualitative_leaps :=
by
  sorry

end tencent_dialectical_materialism_alignment_l3239_323983


namespace rubber_boat_fall_time_l3239_323999

/-- Represents the speed of the ship in still water -/
def ship_speed : ℝ := sorry

/-- Represents the speed of the water flow -/
def water_flow : ℝ := sorry

/-- Represents the time (in hours) when the rubber boat fell into the water, before 5 PM -/
def fall_time : ℝ := sorry

/-- Represents the fact that the ship catches up with the rubber boat after 1 hour -/
axiom catch_up_condition : (5 - fall_time) * (ship_speed - water_flow) + (6 - fall_time) * water_flow = ship_speed + water_flow

theorem rubber_boat_fall_time : fall_time = 4 := by sorry

end rubber_boat_fall_time_l3239_323999


namespace sufficient_not_necessary_and_necessary_not_sufficient_l3239_323932

theorem sufficient_not_necessary_and_necessary_not_sufficient :
  (∀ a : ℝ, a > 1 → 1 / a < 1) ∧
  (∃ a : ℝ, ¬(a > 1) ∧ 1 / a < 1) ∧
  (∀ a b : ℝ, a * b ≠ 0 → a ≠ 0) ∧
  (∃ a b : ℝ, a ≠ 0 ∧ a * b = 0) :=
by sorry

end sufficient_not_necessary_and_necessary_not_sufficient_l3239_323932


namespace cookie_ratio_l3239_323962

def cookie_problem (initial_white : ℕ) (black_white_difference : ℕ) (remaining_total : ℕ) : Prop :=
  let initial_black : ℕ := initial_white + black_white_difference
  let eaten_black : ℕ := initial_black / 2
  let remaining_black : ℕ := initial_black - eaten_black
  let remaining_white : ℕ := remaining_total - remaining_black
  let eaten_white : ℕ := initial_white - remaining_white
  (eaten_white : ℚ) / initial_white = 3 / 4

theorem cookie_ratio :
  cookie_problem 80 50 85 := by
  sorry

end cookie_ratio_l3239_323962


namespace ellipse_product_l3239_323957

/-- Given an ellipse with center O, major axis AB, minor axis CD, and focus F,
    prove that if OF = 8 and the diameter of the inscribed circle of triangle OCF is 4,
    then (AB)(CD) = 240 -/
theorem ellipse_product (O A B C D F : ℝ × ℝ) : 
  let OA := dist O A
  let OB := dist O B
  let OC := dist O C
  let OD := dist O D
  let OF := dist O F
  let a := OA
  let b := OC
  let inscribed_diameter := 4
  (OA = OB) →  -- A and B are equidistant from O (major axis)
  (OC = OD) →  -- C and D are equidistant from O (minor axis)
  (a > b) →    -- major axis is longer than minor axis
  (OF = 8) →   -- given condition
  (b + OF - a = inscribed_diameter / 2) →  -- inradius formula for triangle OCF
  (2 * a) * (2 * b) = 240 :=
by sorry

end ellipse_product_l3239_323957


namespace triangle_angle_problem_l3239_323947

theorem triangle_angle_problem (x : ℝ) : 
  x > 0 ∧ 
  x + 2*x + 40 = 180 → 
  x = 140/3 :=
by sorry

end triangle_angle_problem_l3239_323947


namespace complex_root_ratio_l3239_323923

theorem complex_root_ratio (m n : ℝ) : 
  (Complex.I * 2 + 1) ^ 2 + m * (Complex.I * 2 + 1) + n = 0 → m / n = 2 / 5 := by
  sorry

end complex_root_ratio_l3239_323923


namespace complement_P_correct_l3239_323922

/-- The set P defined as {x | x² < 1} -/
def P : Set ℝ := {x | x^2 < 1}

/-- The complement of P in ℝ -/
def complement_P : Set ℝ := {x | x ≤ -1 ∨ x ≥ 1}

theorem complement_P_correct : 
  {x : ℝ | x ∉ P} = complement_P := by
  sorry

end complement_P_correct_l3239_323922


namespace evaluate_expression_l3239_323949

theorem evaluate_expression : (33 + 12)^2 - (12^2 + 33^2) = 792 := by
  sorry

end evaluate_expression_l3239_323949


namespace circle_exists_with_conditions_l3239_323936

-- Define a 3D point
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a plane in 3D space
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

-- Define a circle in 3D space
structure Circle3D where
  center : Point3D
  radius : ℝ
  normal : Point3D  -- Normal vector to the plane of the circle

def angle_between_planes (p1 p2 : Plane3D) : ℝ := sorry

def circle_touches_plane (c : Circle3D) (p : Plane3D) : Prop := sorry

def circle_passes_through_points (c : Circle3D) (p1 p2 : Point3D) : Prop := sorry

theorem circle_exists_with_conditions 
  (p1 p2 : Point3D) 
  (projection_plane : Plane3D) : 
  ∃ (c : Circle3D), 
    circle_passes_through_points c p1 p2 ∧ 
    angle_between_planes (Plane3D.mk c.normal.x c.normal.y c.normal.z 0) projection_plane = π/3 ∧
    circle_touches_plane c projection_plane := by
  sorry

end circle_exists_with_conditions_l3239_323936


namespace function_properties_l3239_323925

def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def IsPeriodic (f : ℝ → ℝ) (T : ℝ) : Prop := ∀ x, f (x + T) = f x

theorem function_properties (f : ℝ → ℝ) 
    (h1 : ∀ x, f (10 + x) = f (10 - x))
    (h2 : ∀ x, f (5 - x) = f (5 + x))
    (h3 : ¬ (∀ x y, f x = f y)) :
    IsEven f ∧ IsPeriodic f 10 := by
  sorry

end function_properties_l3239_323925


namespace kevins_food_spending_l3239_323981

theorem kevins_food_spending (total_budget : ℕ) (samuels_ticket : ℕ) (samuels_food_drinks : ℕ)
  (kevins_ticket : ℕ) (kevins_drinks : ℕ) (kevins_food : ℕ)
  (h1 : total_budget = 20)
  (h2 : samuels_ticket = 14)
  (h3 : samuels_food_drinks = 6)
  (h4 : kevins_ticket = 14)
  (h5 : kevins_drinks = 2)
  (h6 : samuels_ticket + samuels_food_drinks = total_budget)
  (h7 : kevins_ticket + kevins_drinks + kevins_food = total_budget) :
  kevins_food = 4 := by
  sorry

end kevins_food_spending_l3239_323981


namespace ab_value_l3239_323924

theorem ab_value (a b : ℝ) (h1 : a - b = 8) (h2 : a^2 + b^2 = 164) : a * b = 50 := by
  sorry

end ab_value_l3239_323924


namespace scientific_notation_138000_l3239_323944

theorem scientific_notation_138000 : 
  138000 = 1.38 * (10 : ℝ)^5 := by sorry

end scientific_notation_138000_l3239_323944


namespace complex_norm_problem_l3239_323953

theorem complex_norm_problem (z w : ℂ) 
  (h1 : Complex.abs (3 * z - 2 * w) = 30)
  (h2 : Complex.abs (z + 3 * w) = 6)
  (h3 : Complex.abs (z - w) = 3) :
  Complex.abs z = 3 := by
  sorry

end complex_norm_problem_l3239_323953


namespace baseball_bat_price_l3239_323930

-- Define the prices and quantities
def basketball_price : ℝ := 29
def basketball_quantity : ℕ := 10
def baseball_price : ℝ := 2.5
def baseball_quantity : ℕ := 14
def price_difference : ℝ := 237

-- Define the theorem
theorem baseball_bat_price :
  ∃ (bat_price : ℝ),
    (basketball_price * basketball_quantity) =
    (baseball_price * baseball_quantity + bat_price + price_difference) ∧
    bat_price = 18 := by
  sorry

end baseball_bat_price_l3239_323930


namespace first_degree_function_theorem_l3239_323958

/-- A first-degree function from ℝ to ℝ -/
structure FirstDegreeFunction where
  f : ℝ → ℝ
  k : ℝ
  b : ℝ
  h : ∀ x, f x = k * x + b
  k_nonzero : k ≠ 0

/-- Theorem: If f is a first-degree function satisfying f(f(x)) = 4x + 9 for all x,
    then f(x) = 2x + 3 or f(x) = -2x - 9 -/
theorem first_degree_function_theorem (f : FirstDegreeFunction) 
  (h : ∀ x, f.f (f.f x) = 4 * x + 9) :
  (∀ x, f.f x = 2 * x + 3) ∨ (∀ x, f.f x = -2 * x - 9) := by
  sorry

end first_degree_function_theorem_l3239_323958


namespace preimage_of_two_neg_one_l3239_323918

/-- A mapping f from ℝ² to ℝ² defined by f(a,b) = (a+b, a-b) -/
def f : ℝ × ℝ → ℝ × ℝ := λ (a, b) ↦ (a + b, a - b)

/-- The theorem stating that the preimage of (2, -1) under f is (1/2, 3/2) -/
theorem preimage_of_two_neg_one : 
  f (1/2, 3/2) = (2, -1) := by sorry

end preimage_of_two_neg_one_l3239_323918


namespace age_of_other_man_l3239_323937

/-- Given a group of men where two are replaced by women, prove the age of a specific man. -/
theorem age_of_other_man
  (n : ℕ)  -- Total number of people
  (m : ℕ)  -- Number of men initially
  (w : ℕ)  -- Number of women replacing men
  (age_increase : ℝ)  -- Increase in average age
  (known_man_age : ℝ)  -- Age of the known man
  (women_avg_age : ℝ)  -- Average age of the women
  (h1 : n = 8)  -- Total number of people is 8
  (h2 : m = 8)  -- Initial number of men is 8
  (h3 : w = 2)  -- Number of women replacing men is 2
  (h4 : age_increase = 2)  -- Average age increases by 2 years
  (h5 : known_man_age = 24)  -- One man is 24 years old
  (h6 : women_avg_age = 30)  -- Average age of women is 30 years
  : ∃ (other_man_age : ℝ), other_man_age = 20 :=
by sorry

end age_of_other_man_l3239_323937


namespace base3_to_base10_equality_l3239_323975

/-- Converts a base 3 number represented as a list of digits to its base 10 equivalent -/
def base3ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

/-- The base 3 representation of the number we want to convert -/
def base3Number : List Nat := [2, 1, 2, 0, 1]

/-- Theorem stating that the base 3 number 10212₃ is equal to 104 in base 10 -/
theorem base3_to_base10_equality : base3ToBase10 base3Number = 104 := by
  sorry

end base3_to_base10_equality_l3239_323975


namespace mark_has_10_fewer_cards_l3239_323970

/-- The number of Pokemon cards each person has. -/
structure CardCounts where
  lloyd : ℕ
  mark : ℕ
  michael : ℕ

/-- The conditions of the Pokemon card problem. -/
def PokemonCardProblem (c : CardCounts) : Prop :=
  c.mark = 3 * c.lloyd ∧
  c.mark < c.michael ∧
  c.michael = 100 ∧
  c.lloyd + c.mark + c.michael + 80 = 300

/-- The theorem stating that Mark has 10 fewer cards than Michael. -/
theorem mark_has_10_fewer_cards (c : CardCounts) 
  (h : PokemonCardProblem c) : c.michael - c.mark = 10 := by
  sorry

end mark_has_10_fewer_cards_l3239_323970


namespace root_sum_reciprocal_l3239_323978

-- Define the polynomial
def f (x : ℝ) := 40 * x^3 - 70 * x^2 + 32 * x - 2

-- State the theorem
theorem root_sum_reciprocal (a b c : ℝ) :
  f a = 0 → f b = 0 → f c = 0 →  -- a, b, c are roots of f
  a ≠ b → b ≠ c → a ≠ c →        -- a, b, c are distinct
  0 < a → a < 1 →                -- 0 < a < 1
  0 < b → b < 1 →                -- 0 < b < 1
  0 < c → c < 1 →                -- 0 < c < 1
  1/(1-a) + 1/(1-b) + 1/(1-c) = 11/20 :=
by sorry

end root_sum_reciprocal_l3239_323978


namespace subset_condition_l3239_323992

theorem subset_condition (a : ℝ) : 
  ({x : ℝ | 1 ≤ x ∧ x ≤ 2} ⊆ {x : ℝ | a < x}) ↔ a < 1 := by
  sorry

end subset_condition_l3239_323992


namespace yogurt_combinations_l3239_323993

/- Define the number of flavors and toppings -/
def num_flavors : ℕ := 4
def num_toppings : ℕ := 8

/- Define the function to calculate combinations -/
def choose (n k : ℕ) : ℕ := (n.factorial) / (k.factorial * (n - k).factorial)

/- Theorem statement -/
theorem yogurt_combinations :
  let no_topping := 1
  let two_toppings := choose num_toppings 2
  let combinations_per_flavor := no_topping + two_toppings
  num_flavors * combinations_per_flavor = 116 := by
  sorry

end yogurt_combinations_l3239_323993


namespace expression_equivalence_l3239_323934

theorem expression_equivalence (a b c : ℝ) (hc : c ≠ 0) :
  ((a - 0.07 * a) + 0.05 * b) / c = (0.93 * a + 0.05 * b) / c := by
  sorry

end expression_equivalence_l3239_323934


namespace baseball_card_value_decrease_l3239_323954

theorem baseball_card_value_decrease : ∀ (initial_value : ℝ),
  initial_value > 0 →
  let first_year_value := initial_value * (1 - 0.4)
  let second_year_value := first_year_value * (1 - 0.1)
  let total_decrease := (initial_value - second_year_value) / initial_value
  total_decrease = 0.46 := by
sorry

end baseball_card_value_decrease_l3239_323954


namespace quadratic_intercepts_l3239_323950

/-- A quadratic function. -/
structure QuadraticFunction where
  f : ℝ → ℝ

/-- The x-intercepts of two quadratic functions. -/
structure XIntercepts where
  x₁ : ℝ
  x₂ : ℝ
  x₃ : ℝ
  x₄ : ℝ

/-- The problem statement. -/
theorem quadratic_intercepts 
  (f g : QuadraticFunction) 
  (x : XIntercepts) 
  (h1 : ∀ x, g.f x = -f.f (120 - x))
  (h2 : ∃ v, g.f v = f.f v ∧ ∀ x, f.f x ≤ f.f v)
  (h3 : x.x₁ < x.x₂ ∧ x.x₂ < x.x₃ ∧ x.x₃ < x.x₄)
  (h4 : x.x₃ - x.x₂ = 160) :
  x.x₄ - x.x₁ = 640 + 320 * Real.sqrt 3 :=
sorry

end quadratic_intercepts_l3239_323950


namespace jessie_weight_before_jogging_l3239_323977

/-- Jessie's weight before jogging, given her current weight and weight loss -/
theorem jessie_weight_before_jogging 
  (current_weight : ℕ) 
  (weight_loss : ℕ) 
  (h1 : current_weight = 67) 
  (h2 : weight_loss = 7) : 
  current_weight + weight_loss = 74 := by
  sorry

end jessie_weight_before_jogging_l3239_323977


namespace inequality_proof_l3239_323955

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a + b + c = 1) : 
  (1 + 9*a^2)/(1 + 2*a + 2*b^2 + 2*c^2) + 
  (1 + 9*b^2)/(1 + 2*b + 2*c^2 + 2*a^2) + 
  (1 + 9*c^2)/(1 + 2*c + 2*a^2 + 2*b^2) < 4 := by
  sorry

end inequality_proof_l3239_323955


namespace river_current_speed_l3239_323942

/-- The speed of a river's current given swimmer's performance -/
theorem river_current_speed 
  (distance : ℝ) 
  (time : ℝ) 
  (still_water_speed : ℝ) 
  (h1 : distance = 7) 
  (h2 : time = 3.684210526315789) 
  (h3 : still_water_speed = 4.4) : 
  ∃ current_speed : ℝ, 
    current_speed = 2.5 ∧ 
    distance / time = still_water_speed - current_speed := by
  sorry

#check river_current_speed

end river_current_speed_l3239_323942


namespace quadratic_positive_iff_not_one_l3239_323921

/-- Given a quadratic function f(x) = x^2 + bx + 1 where f(-1) = f(3),
    prove that f(x) > 0 if and only if x ≠ 1 -/
theorem quadratic_positive_iff_not_one (b : ℝ) (f : ℝ → ℝ) 
    (h1 : ∀ x, f x = x^2 + b*x + 1)
    (h2 : f (-1) = f 3) :
    ∀ x, f x > 0 ↔ x ≠ 1 := by
  sorry

end quadratic_positive_iff_not_one_l3239_323921


namespace congruence_from_equation_l3239_323915

theorem congruence_from_equation (a b : ℕ+) (h : a^(b : ℕ) - b^(a : ℕ) = 1008) :
  a ≡ b [ZMOD 1008] := by
  sorry

end congruence_from_equation_l3239_323915


namespace only_set_A_forms_triangle_l3239_323979

-- Define a function to check if three lengths can form a triangle
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- Define the sets of line segments
def set_A : List ℝ := [5, 6, 10]
def set_B : List ℝ := [5, 2, 9]
def set_C : List ℝ := [5, 7, 12]
def set_D : List ℝ := [3, 4, 8]

-- Theorem statement
theorem only_set_A_forms_triangle :
  (can_form_triangle 5 6 10) ∧
  ¬(can_form_triangle 5 2 9) ∧
  ¬(can_form_triangle 5 7 12) ∧
  ¬(can_form_triangle 3 4 8) :=
sorry

end only_set_A_forms_triangle_l3239_323979


namespace black_area_calculation_l3239_323951

theorem black_area_calculation (large_side : ℝ) (small_side : ℝ) :
  large_side = 12 →
  small_side = 5 →
  large_side^2 - 2 * small_side^2 = 94 := by
  sorry

end black_area_calculation_l3239_323951


namespace bird_weights_solution_l3239_323912

theorem bird_weights_solution :
  ∃! (A B V G : ℕ+),
    A + B + V + G = 32 ∧
    V < G ∧
    V + G < B ∧
    A < V + B ∧
    G + B < A + V ∧
    A = 13 ∧ B = 10 ∧ V = 4 ∧ G = 5 :=
by sorry

end bird_weights_solution_l3239_323912


namespace max_distance_42000km_l3239_323935

/-- Represents the maximum distance a car can travel with tire switching -/
def maxDistanceWithTireSwitch (frontTireLife rear_tire_life : ℕ) : ℕ :=
  min frontTireLife rear_tire_life

/-- Theorem stating the maximum distance a car can travel with specific tire lifespans -/
theorem max_distance_42000km (frontTireLife rearTireLife : ℕ) 
  (h1 : frontTireLife = 42000)
  (h2 : rearTireLife = 56000) :
  maxDistanceWithTireSwitch frontTireLife rearTireLife = 42000 :=
by
  sorry

#eval maxDistanceWithTireSwitch 42000 56000

end max_distance_42000km_l3239_323935


namespace isosceles_triangle_side_length_l3239_323928

/-- Given a square with side length 2 and four congruent isosceles triangles constructed with 
    their bases on the sides of the square, if the sum of the areas of the four isosceles 
    triangles is equal to the area of the square, then the length of one of the two congruent 
    sides of one isosceles triangle is √17/2. -/
theorem isosceles_triangle_side_length (square_side : ℝ) (triangle_base : ℝ) (triangle_height : ℝ) :
  square_side = 2 →
  triangle_base = square_side →
  (4 * (1/2 * triangle_base * triangle_height)) = square_side^2 →
  ∃ (triangle_side : ℝ), 
    triangle_side^2 = (triangle_base/2)^2 + triangle_height^2 ∧ 
    triangle_side = Real.sqrt 17 / 2 :=
by sorry

end isosceles_triangle_side_length_l3239_323928


namespace dividend_percentage_calculation_l3239_323919

/-- Calculate the dividend percentage of shares -/
theorem dividend_percentage_calculation 
  (cost_price : ℝ) 
  (desired_interest_rate : ℝ) 
  (market_value : ℝ) : 
  cost_price = 60 →
  desired_interest_rate = 12 / 100 →
  market_value = 45 →
  (market_value * desired_interest_rate) / cost_price * 100 = 9 := by
  sorry

end dividend_percentage_calculation_l3239_323919


namespace line_slope_intercept_sum_l3239_323994

/-- Given a line passing through points (1, 3) and (3, 7), prove that m + b = 3 -/
theorem line_slope_intercept_sum (m b : ℝ) : 
  (3 = m * 1 + b) → (7 = m * 3 + b) → m + b = 3 := by sorry

end line_slope_intercept_sum_l3239_323994


namespace recycled_cans_count_l3239_323945

def recycle_cans (initial_cans : ℕ) : ℕ :=
  if initial_cans < 5 then 0
  else (initial_cans / 5) + recycle_cans (initial_cans / 5)

theorem recycled_cans_count :
  recycle_cans 3125 = 781 :=
by sorry

end recycled_cans_count_l3239_323945


namespace school_picnic_volunteers_l3239_323963

theorem school_picnic_volunteers (total_parents : ℕ) (supervise : ℕ) (both : ℕ) (refresh_ratio : ℚ) : 
  total_parents = 84 →
  supervise = 25 →
  both = 11 →
  refresh_ratio = 3/2 →
  ∃ (refresh : ℕ) (neither : ℕ),
    refresh = refresh_ratio * neither ∧
    total_parents = (supervise - both) + (refresh - both) + both + neither ∧
    refresh = 42 := by
  sorry

end school_picnic_volunteers_l3239_323963


namespace trees_survived_vs_died_l3239_323967

theorem trees_survived_vs_died (initial_trees : ℕ) (trees_died : ℕ) : 
  initial_trees = 13 → trees_died = 6 → (initial_trees - trees_died) - trees_died = 1 := by
  sorry

end trees_survived_vs_died_l3239_323967


namespace smallest_x_with_remainders_l3239_323982

theorem smallest_x_with_remainders : ∃ x : ℕ+, 
  (x : ℕ) % 3 = 2 ∧ 
  (x : ℕ) % 7 = 6 ∧ 
  (x : ℕ) % 8 = 7 ∧ 
  (∀ y : ℕ+, y < x → 
    (y : ℕ) % 3 ≠ 2 ∨ 
    (y : ℕ) % 7 ≠ 6 ∨ 
    (y : ℕ) % 8 ≠ 7) ∧
  x = 167 :=
by sorry

end smallest_x_with_remainders_l3239_323982


namespace small_font_words_per_page_l3239_323989

/-- Calculates the number of words per page in the small font given the article constraints -/
theorem small_font_words_per_page 
  (total_words : ℕ) 
  (total_pages : ℕ) 
  (large_font_pages : ℕ) 
  (large_font_words_per_page : ℕ) 
  (h1 : total_words = 48000)
  (h2 : total_pages = 21)
  (h3 : large_font_pages = 4)
  (h4 : large_font_words_per_page = 1800) :
  (total_words - large_font_pages * large_font_words_per_page) / (total_pages - large_font_pages) = 2400 :=
by
  sorry

#check small_font_words_per_page

end small_font_words_per_page_l3239_323989


namespace sports_participation_l3239_323917

theorem sports_participation (B C S Ba : ℕ)
  (BC BS BBa CS CBa SBa BCSL : ℕ)
  (h1 : B = 12)
  (h2 : C = 10)
  (h3 : S = 9)
  (h4 : Ba = 6)
  (h5 : BC = 5)
  (h6 : BS = 4)
  (h7 : BBa = 3)
  (h8 : CS = 2)
  (h9 : CBa = 3)
  (h10 : SBa = 2)
  (h11 : BCSL = 1) :
  B + C + S + Ba - BC - BS - BBa - CS - CBa - SBa + BCSL = 19 := by
  sorry

end sports_participation_l3239_323917


namespace quadratic_coefficient_inequalities_l3239_323952

theorem quadratic_coefficient_inequalities
  (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_real_roots : ∃ x y : ℝ, a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 ∧ x ≠ y) :
  min a (min b c) ≤ (1/4) * (a + b + c) ∧
  max a (max b c) ≥ (4/9) * (a + b + c) := by
  sorry

end quadratic_coefficient_inequalities_l3239_323952


namespace inequality_holds_iff_p_le_eight_l3239_323906

theorem inequality_holds_iff_p_le_eight (p : ℝ) : 
  (∀ x : ℝ, 0 < x ∧ x < π/2 → (1 + 1/Real.sin x)^3 ≥ p/(Real.tan x)^2) ↔ p ≤ 8 :=
sorry

end inequality_holds_iff_p_le_eight_l3239_323906


namespace medium_lights_count_l3239_323971

/-- Represents the number of medium ceiling lights -/
def M : ℕ := sorry

/-- The number of small ceiling lights -/
def small_lights : ℕ := M + 10

/-- The number of large ceiling lights -/
def large_lights : ℕ := 2 * M

/-- The total number of bulbs needed -/
def total_bulbs : ℕ := 118

/-- Theorem stating that the number of medium ceiling lights is 12 -/
theorem medium_lights_count : M = 12 := by
  have bulb_equation : small_lights * 1 + M * 2 + large_lights * 3 = total_bulbs := by sorry
  sorry

end medium_lights_count_l3239_323971


namespace algebraic_expression_value_l3239_323903

theorem algebraic_expression_value (a : ℝ) (h : 2 * a^2 - 3 * a = 1) :
  9 * a + 7 - 6 * a^2 = 4 := by
  sorry

end algebraic_expression_value_l3239_323903


namespace max_volume_open_top_box_l3239_323956

/-- Given a square sheet metal of width 60 cm, the maximum volume of an open-top box 
    with a square base that can be created from it is 16000 cm³. -/
theorem max_volume_open_top_box (sheet_width : ℝ) (h : sheet_width = 60) :
  ∃ (x : ℝ), 0 < x ∧ x < sheet_width / 2 ∧
  (∀ (y : ℝ), 0 < y → y < sheet_width / 2 → 
    x * (sheet_width - 2 * x)^2 ≥ y * (sheet_width - 2 * y)^2) ∧
  x * (sheet_width - 2 * x)^2 = 16000 :=
by sorry


end max_volume_open_top_box_l3239_323956


namespace cats_favorite_number_l3239_323991

def is_two_digit_positive (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

def has_distinct_nonzero_digits (n : ℕ) : Prop :=
  let tens := n / 10
  let ones := n % 10
  tens ≠ ones ∧ tens ≠ 0 ∧ ones ≠ 0

def digits_are_factors (n : ℕ) : Prop :=
  let tens := n / 10
  let ones := n % 10
  n % tens = 0 ∧ n % ones = 0

def satisfies_four_number_property (a b c d : ℕ) : Prop :=
  a + b - c = d ∧ b + c - a = d ∧ c + d - b = a ∧ d + a - c = b

theorem cats_favorite_number :
  ∃! n : ℕ,
    is_two_digit_positive n ∧
    has_distinct_nonzero_digits n ∧
    digits_are_factors n ∧
    ∃ a b c : ℕ,
      satisfies_four_number_property n a b c ∧
      n^2 = a * b ∧
      (a ≠ n ∧ b ≠ n ∧ c ≠ n) :=
by
  sorry

end cats_favorite_number_l3239_323991


namespace roberts_pencils_l3239_323940

-- Define the price of a pencil in cents
def pencil_price : ℕ := 20

-- Define the number of pencils Tolu wants
def tolu_pencils : ℕ := 3

-- Define the number of pencils Melissa wants
def melissa_pencils : ℕ := 2

-- Define the total amount spent by all students in cents
def total_spent : ℕ := 200

-- Theorem to prove Robert's number of pencils
theorem roberts_pencils : 
  ∃ (robert_pencils : ℕ), 
    pencil_price * (tolu_pencils + melissa_pencils + robert_pencils) = total_spent ∧
    robert_pencils = 5 := by
  sorry

end roberts_pencils_l3239_323940


namespace arithmetic_sequence_sum_l3239_323905

def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℕ) :
  is_arithmetic_sequence a →
  a 1 = 3 →
  a 2 = 8 →
  a 5 = 28 →
  a 3 + a 4 = 31 :=
by
  sorry

end arithmetic_sequence_sum_l3239_323905


namespace bird_families_flew_away_l3239_323914

/-- Given the initial number of bird families and the number of families left,
    calculate the number of families that flew away. -/
theorem bird_families_flew_away (initial : ℕ) (left : ℕ) (flew_away : ℕ) 
    (h1 : initial = 41)
    (h2 : left = 14)
    (h3 : flew_away = initial - left) :
  flew_away = 27 := by
  sorry

end bird_families_flew_away_l3239_323914


namespace cos_30_degree_calculation_l3239_323959

theorem cos_30_degree_calculation : 
  |Real.sqrt 3 - 1| - 2 * (Real.sqrt 3 / 2) = -1 := by sorry

end cos_30_degree_calculation_l3239_323959


namespace quadratic_intersects_x_axis_l3239_323948

/-- 
A quadratic function y = kx^2 + 2x + 1 intersects the x-axis at two points 
if and only if k < 1 and k ≠ 0
-/
theorem quadratic_intersects_x_axis (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ k * x₁^2 + 2 * x₁ + 1 = 0 ∧ k * x₂^2 + 2 * x₂ + 1 = 0) ↔
  (k < 1 ∧ k ≠ 0) :=
by sorry

end quadratic_intersects_x_axis_l3239_323948


namespace cylindrical_bar_length_l3239_323987

/-- The length of a cylindrical steel bar formed from a rectangular billet -/
theorem cylindrical_bar_length 
  (billet_length : ℝ) 
  (billet_width : ℝ) 
  (billet_height : ℝ) 
  (cylinder_diameter : ℝ) 
  (h1 : billet_length = 12.56)
  (h2 : billet_width = 5)
  (h3 : billet_height = 4)
  (h4 : cylinder_diameter = 4) : 
  (billet_length * billet_width * billet_height) / (π * (cylinder_diameter / 2)^2) = 20 := by
  sorry

#check cylindrical_bar_length

end cylindrical_bar_length_l3239_323987


namespace ratio_of_segments_l3239_323904

-- Define the points on a line
variable (E F G H : ℝ)

-- Define the conditions
variable (h1 : E < F)
variable (h2 : F < G)
variable (h3 : G < H)
variable (h4 : F - E = 3)
variable (h5 : G - F = 8)
variable (h6 : H - E = 23)

-- Theorem statement
theorem ratio_of_segments :
  (G - E) / (H - F) = 11 / 20 := by
  sorry

end ratio_of_segments_l3239_323904


namespace banana_basket_count_l3239_323984

theorem banana_basket_count (total_baskets : ℕ) (average_fruits : ℕ) 
  (basket_a : ℕ) (basket_b : ℕ) (basket_c : ℕ) (basket_d : ℕ) :
  total_baskets = 5 →
  average_fruits = 25 →
  basket_a = 15 →
  basket_b = 30 →
  basket_c = 20 →
  basket_d = 25 →
  (total_baskets * average_fruits) - (basket_a + basket_b + basket_c + basket_d) = 35 :=
by sorry

end banana_basket_count_l3239_323984


namespace cube_volume_surface_area_l3239_323902

/-- Given a cube with volume 8x cubic units and surface area 4x square units, x = 216 -/
theorem cube_volume_surface_area (x : ℝ) : 
  (∃ (s : ℝ), s > 0 ∧ s^3 = 8*x ∧ 6*s^2 = 4*x) → x = 216 := by
  sorry

end cube_volume_surface_area_l3239_323902


namespace true_absolute_error_example_l3239_323900

/-- The true absolute error of a₀ with respect to a -/
def trueAbsoluteError (a₀ a : ℝ) : ℝ := |a - a₀|

theorem true_absolute_error_example : 
  trueAbsoluteError 245.2 246 = 0.8 := by sorry

end true_absolute_error_example_l3239_323900


namespace max_PXQ_value_l3239_323996

def is_two_digit_with_equal_digits (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧ n % 11 = 0

def is_one_digit (n : ℕ) : Prop :=
  0 < n ∧ n ≤ 9

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def starts_with (n : ℕ) (d : ℕ) : Prop :=
  (n / 100) = d

def ends_with (n : ℕ) (d : ℕ) : Prop :=
  n % 10 = d

theorem max_PXQ_value :
  ∀ XX X PXQ : ℕ,
    is_two_digit_with_equal_digits XX →
    is_one_digit X →
    is_three_digit PXQ →
    XX * X = PXQ →
    starts_with PXQ (PXQ / 100) →
    ends_with PXQ X →
    PXQ ≤ 396 :=
sorry

end max_PXQ_value_l3239_323996


namespace geometric_sequence_sum_l3239_323911

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →
  (∀ n, a (n + 1) = q * a n) →
  a 1 = 1 →
  a 1 + a 3 + a 5 = 21 →
  a 2 + a 4 + a 6 = 42 := by
sorry

end geometric_sequence_sum_l3239_323911


namespace equation_solution_l3239_323933

open Real

theorem equation_solution (x : ℝ) :
  0 < x ∧ x < π →
  ((Real.sqrt 2014 - Real.sqrt 2013) ^ (tan x)^2 + 
   (Real.sqrt 2014 + Real.sqrt 2013) ^ (-(tan x)^2) = 
   2 * (Real.sqrt 2014 - Real.sqrt 2013)^3) ↔ 
  (x = π/3 ∨ x = 2*π/3) :=
sorry

end equation_solution_l3239_323933


namespace steve_long_letter_time_l3239_323938

/-- Represents the writing habits of Steve --/
structure WritingHabits where
  days_between_letters : ℕ
  minutes_per_regular_letter : ℕ
  minutes_per_page : ℕ
  long_letter_time_factor : ℕ
  total_pages_per_month : ℕ
  days_in_month : ℕ

/-- Calculates the time spent on the long letter at the end of the month --/
def long_letter_time (habits : WritingHabits) : ℕ :=
  let regular_letters := habits.days_in_month / habits.days_between_letters
  let pages_per_regular_letter := habits.minutes_per_regular_letter / habits.minutes_per_page
  let regular_letter_pages := regular_letters * pages_per_regular_letter
  let long_letter_pages := habits.total_pages_per_month - regular_letter_pages
  long_letter_pages * (habits.minutes_per_page * habits.long_letter_time_factor)

/-- Theorem stating that Steve spends 80 minutes writing the long letter --/
theorem steve_long_letter_time :
  ∃ (habits : WritingHabits),
    habits.days_between_letters = 3 ∧
    habits.minutes_per_regular_letter = 20 ∧
    habits.minutes_per_page = 10 ∧
    habits.long_letter_time_factor = 2 ∧
    habits.total_pages_per_month = 24 ∧
    habits.days_in_month = 30 ∧
    long_letter_time habits = 80 := by
  sorry

end steve_long_letter_time_l3239_323938


namespace unique_modular_solution_l3239_323961

theorem unique_modular_solution (m : ℤ) : 
  (5 ≤ m ∧ m ≤ 9) → (m ≡ 5023 [ZMOD 6]) → m = 7 := by
  sorry

end unique_modular_solution_l3239_323961


namespace ball_distribution_ratio_l3239_323907

/-- The number of balls -/
def n : ℕ := 25

/-- The number of bins -/
def m : ℕ := 5

/-- The probability of distributing n balls into m bins such that
    one bin has 3 balls, another has 7 balls, and the other three have 5 balls each -/
noncomputable def p : ℝ := sorry

/-- The probability of distributing n balls equally into m bins (5 balls each) -/
noncomputable def q : ℝ := sorry

/-- Theorem stating that the ratio of p to q is 12 -/
theorem ball_distribution_ratio : p / q = 12 := by sorry

end ball_distribution_ratio_l3239_323907


namespace triangle_construction_possible_l3239_323960

-- Define the necessary types and structures
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

def Midpoint (A B M : Point) : Prop :=
  M.x = (A.x + B.x) / 2 ∧ M.y = (A.y + B.y) / 2

def OnLine (P : Point) (l : Line) : Prop :=
  l.a * P.x + l.b * P.y + l.c = 0

def AngleBisector (A B C : Point) (l : Line) : Prop :=
  -- This is a simplified representation of an angle bisector
  OnLine A l ∧ ∃ (P : Point), OnLine P l ∧ P ≠ A

-- Theorem statement
theorem triangle_construction_possible (l : Line) :
  ∃ (A B C N M : Point),
    Midpoint A C N ∧
    Midpoint B C M ∧
    AngleBisector A B C l :=
sorry

end triangle_construction_possible_l3239_323960
