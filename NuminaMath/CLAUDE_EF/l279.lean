import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compare_f_values_l279_27915

noncomputable def f (x : ℝ) : ℝ := 1 / (1 + x^2)

theorem compare_f_values (a : ℝ) (ha : a > 0) :
  (f a > f (a/2) * f (a/2) ↔ a > 2 * Real.sqrt 2) ∧
  (f a = f (a/2) * f (a/2) ↔ a = 2 * Real.sqrt 2) ∧
  (f a < f (a/2) * f (a/2) ↔ 0 < a ∧ a < 2 * Real.sqrt 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_compare_f_values_l279_27915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_theorem_l279_27981

/-- Line represented by parametric equations -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- Circle represented by polar equation -/
structure PolarCircle where
  ρ : ℝ → ℝ

/-- Minimum distance from a circle to a line -/
noncomputable def minDistanceCircleToLine (l : ParametricLine) (c : PolarCircle) : ℝ :=
  sorry

/-- The given line l -/
noncomputable def line_l : ParametricLine :=
  { x := λ t => 1 - (Real.sqrt 2 / 2) * t,
    y := λ t => 2 + (Real.sqrt 2 / 2) * t }

/-- The given circle C -/
noncomputable def circle_C : PolarCircle :=
  { ρ := λ θ => 2 * Real.cos θ }

theorem min_distance_theorem :
  minDistanceCircleToLine line_l circle_C = Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_theorem_l279_27981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_inscribed_circle_radius_is_3_sqrt_5_l279_27989

/-- The radius of the largest inscribed circle in a quadrilateral with given side lengths -/
noncomputable def largest_inscribed_circle_radius (a b c d : ℝ) : ℝ :=
  let s := (a + b + c + d) / 2
  let area := Real.sqrt ((s - a) * (s - b) * (s - c) * (s - d))
  area / s

/-- Theorem stating that the largest inscribed circle in the given quadrilateral has radius 3√5 -/
theorem largest_inscribed_circle_radius_is_3_sqrt_5 :
  largest_inscribed_circle_radius 13 10 8 11 = 3 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_inscribed_circle_radius_is_3_sqrt_5_l279_27989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_continuity_condition_l279_27918

-- Define the piecewise function f(x)
noncomputable def f (b : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 4 then 4 * x^2 + 5 else b * x + 2

-- State the theorem
theorem continuity_condition (b : ℝ) :
  Continuous (f b) ↔ b = 16.75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_continuity_condition_l279_27918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_surface_area_l279_27970

/-- A cone with a cross-section that is an equilateral triangle with area √3 has a surface area of 3π. -/
theorem cone_surface_area (a : ℝ) (h_area : (Real.sqrt 3 / 4) * a^2 = Real.sqrt 3) :
  let r := (Real.sqrt 3 / 3) * a
  let h := Real.sqrt (a^2 - r^2)
  let l := Real.sqrt (h^2 + r^2)
  π * r^2 + π * r * l = 3 * π :=
by
  -- Introduce the local definitions
  have r := (Real.sqrt 3 / 3) * a
  have h := Real.sqrt (a^2 - r^2)
  have l := Real.sqrt (h^2 + r^2)

  -- The proof steps would go here
  sorry  -- We use sorry to skip the actual proof


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_surface_area_l279_27970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_slope_l279_27946

/-- The slope of the first line -/
noncomputable def m₁ : ℝ := 2

/-- The slope of the second line -/
noncomputable def m₂ : ℝ := 5

/-- The slope of the angle bisector -/
noncomputable def k : ℝ := (Real.sqrt 30 - 7) / 9

/-- Theorem stating the relationship between the slopes -/
theorem angle_bisector_slope :
  k = (m₁ + m₂ - Real.sqrt (1 + m₁^2 + m₂^2)) / (m₁ * m₂ - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_slope_l279_27946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_plane_implies_perpendicular_lines_l279_27901

/-- A line in 3D space -/
structure Line3D where
  -- Define the line structure (implementation details omitted)
  mk :: -- Add a constructor

/-- A plane in 3D space -/
structure Plane where
  -- Define the plane structure (implementation details omitted)
  mk :: -- Add a constructor

/-- Perpendicularity between a line and a plane -/
def perpendicular_line_plane (l : Line3D) (p : Plane) : Prop :=
  sorry -- Placeholder for the actual definition

/-- A line being a subset of a plane -/
def line_subset_plane (l : Line3D) (p : Plane) : Prop :=
  sorry -- Placeholder for the actual definition

/-- Perpendicularity between two lines -/
def perpendicular_lines (l1 l2 : Line3D) : Prop :=
  sorry -- Placeholder for the actual definition

theorem perpendicular_line_plane_implies_perpendicular_lines 
  (m n : Line3D) (α : Plane) : 
  perpendicular_line_plane m α → line_subset_plane n α → 
  perpendicular_lines m n :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_plane_implies_perpendicular_lines_l279_27901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truck_distance_l279_27923

/-- Calculates the distance in yards traveled by a truck in 5 minutes -/
noncomputable def distance_in_yards (b : ℝ) (t : ℝ) : ℝ :=
  let feet_per_second := b / 4
  let seconds_in_five_minutes := 5 * 60
  let feet_in_five_minutes := (feet_per_second / t) * seconds_in_five_minutes
  let yards_in_five_minutes := feet_in_five_minutes / 3
  yards_in_five_minutes

/-- Theorem stating that a truck traveling b/4 feet every t seconds will cover 25b/t yards in 5 minutes -/
theorem truck_distance (b : ℝ) (t : ℝ) (h : t ≠ 0) :
  distance_in_yards b t = 25 * b / t :=
by
  -- Expand the definition of distance_in_yards
  unfold distance_in_yards
  -- Perform algebraic simplifications
  simp [mul_div_assoc, mul_comm, mul_assoc]
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_truck_distance_l279_27923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_40_l279_27924

/-- The angle of the minute hand at a given time -/
noncomputable def minuteHandAngle (minutes : ℕ) : ℝ :=
  (minutes % 60 : ℝ) * 6

/-- The angle of the hour hand at a given time -/
noncomputable def hourHandAngle (hours minutes : ℕ) : ℝ :=
  (hours % 12 : ℝ) * 30 + (minutes : ℝ) * 0.5

/-- The smaller angle between two angles on a circle -/
noncomputable def smallerAngle (angle1 angle2 : ℝ) : ℝ :=
  min (abs (angle1 - angle2)) (360 - abs (angle1 - angle2))

theorem clock_angle_at_3_40 :
  smallerAngle (minuteHandAngle 40) (hourHandAngle 3 40) = 130 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_40_l279_27924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tamika_always_wins_l279_27961

def tamika_set : Finset ℕ := {7, 8, 11, 13}
def carlos_set : Finset ℕ := {2, 4, 9}

def tamika_products : Finset ℕ := 
  (tamika_set.powerset.filter (λ s => s.card = 2)).image (λ s => s.prod id)

def carlos_products : Finset ℕ := 
  (carlos_set.powerset.filter (λ s => s.card = 2)).image (λ s => s.prod id)

theorem tamika_always_wins : 
  (tamika_products.card * carlos_products.card : ℚ)⁻¹ * 
  (tamika_products.card * carlos_products.card) = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tamika_always_wins_l279_27961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_u_properties_l279_27914

def u : ℕ → ℕ
  | 0 => 1  -- Added case for 0
  | 1 => 1
  | 2 => 2
  | 3 => 24
  | (n + 4) => (6 * (u (n + 3))^2 * u (n + 2) - 8 * u (n + 3) * (u (n + 3))^2) / (u (n + 3) * u (n + 2))

theorem u_properties :
  (∀ n : ℕ, n ∣ u n) := by
  sorry

#check u_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_u_properties_l279_27914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_moves_solution_l279_27962

structure Board :=
  (rows : Fin 4 → Fin 4 → Bool)

def initial_board : Board :=
  { rows := λ i j => 
    match i.val, j.val with
    | 0, 0 | 0, 1 | 1, 0 | 1, 2 | 2, 2 | 2, 3 | 3, 1 | 3, 3 => true
    | _, _ => false }

def is_adjacent (i1 j1 i2 j2 : Fin 4) : Bool :=
  (i1 = i2 ∧ (j1.val + 1 = j2.val ∨ j2.val + 1 = j1.val)) ∨
  (j1 = j2 ∧ (i1.val + 1 = i2.val ∨ i2.val + 1 = i1.val)) ∨
  ((i1.val + 1 = i2.val ∨ i2.val + 1 = i1.val) ∧ (j1.val + 1 = j2.val ∨ j2.val + 1 = j1.val))

def is_valid_move (b : Board) (i1 j1 i2 j2 : Fin 4) : Prop :=
  b.rows i1 j1 ∧ ¬b.rows i2 j2 ∧ is_adjacent i1 j1 i2 j2

def move (b : Board) (i1 j1 i2 j2 : Fin 4) : Board :=
  { rows := λ i j => 
    if i = i1 ∧ j = j1 then false
    else if i = i2 ∧ j = j2 then true
    else b.rows i j }

def is_final_state (b : Board) : Prop :=
  (∀ i : Fin 4, (Finset.filter (λ j => b.rows i j) Finset.univ).card = 2) ∧
  (∀ j : Fin 4, (Finset.filter (λ i => b.rows i j) Finset.univ).card = 2)

theorem two_moves_solution :
  ∃ (i1 j1 i2 j2 i3 j3 i4 j4 : Fin 4),
    is_valid_move initial_board i1 j1 i2 j2 ∧
    let b1 := move initial_board i1 j1 i2 j2
    is_valid_move b1 i3 j3 i4 j4 ∧
    is_final_state (move b1 i3 j3 i4 j4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_moves_solution_l279_27962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_purely_imaginary_quotient_l279_27992

def z1 (a : ℝ) : ℂ := a - 2 * Complex.I
def z2 : ℂ := 2 + Complex.I

theorem purely_imaginary_quotient (a : ℝ) : 
  (z1 a / z2).re = 0 ∧ (z1 a / z2).im ≠ 0 → a = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_purely_imaginary_quotient_l279_27992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_pi_12_eq_sqrt_2_l279_27956

/-- The function f(x) = √3 * sin(x) + cos(x) -/
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x + Real.cos x

/-- Theorem stating that f(π/12) = √2 -/
theorem f_pi_12_eq_sqrt_2 : f (π / 12) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_pi_12_eq_sqrt_2_l279_27956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l279_27943

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sin x, -1)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos x, -1/2)

noncomputable def f (x : ℝ) : ℝ :=
  let a_vec := a x
  let b_vec := b x
  (a_vec.1 + b_vec.1) * a_vec.1 + (a_vec.2 + b_vec.2) * a_vec.2 - 2

theorem problem_solution 
  (A : ℝ)
  (h_acute : 0 < A ∧ A < π/2) 
  (a b c : ℝ)
  (h_a : a = 2 * Real.sqrt 3) 
  (h_c : c = 4) 
  (h_fA : f A = 1) :
  (∃ T > 0, (∀ x, f (x + T) = f x) ∧ ∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T') ∧ 
  A = π/3 ∧ 
  b = 2 ∧ 
  (1/2 * b * c * Real.sin A = 2 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l279_27943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_sets_gcd_l279_27938

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

-- Define the greatest common divisor function
def gcd (a b : ℤ) : ℕ := Int.gcd a b

-- Define the sets S₁ and S₂
def S₁ : Set ℕ := {n : ℕ | n > 0 ∧ gcd n (floor (Real.sqrt 2 * ↑n)) = 1}
def S₂ : Set ℕ := {n : ℕ | n > 0 ∧ gcd n (floor (Real.sqrt 2 * ↑n)) ≠ 1}

-- Theorem statement
theorem infinite_sets_gcd :
  (∀ n : ℕ, ∃ m > n, m ∈ S₁) ∧ (∀ n : ℕ, ∃ m > n, m ∈ S₂) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_sets_gcd_l279_27938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_grid_point_distance_l279_27996

/-- The distance from the closest grid point to the line y = (5/3)x + 4/5 is √34/85 -/
theorem closest_grid_point_distance :
  ∃ (x₀ y₀ : ℤ), ∀ (x y : ℤ),
    let line_eq := λ (x : ℝ) => (5/3) * x + 4/5
    let grid_point_distance := λ (x y : ℝ) => |5*x - 3*y + 12| / Real.sqrt 34
    grid_point_distance (x : ℝ) (y : ℝ) ≥ grid_point_distance (x₀ : ℝ) (y₀ : ℝ) ∧
    grid_point_distance (x₀ : ℝ) (y₀ : ℝ) = Real.sqrt 34 / 85 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_grid_point_distance_l279_27996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l279_27911

noncomputable def f (x : ℝ) : ℝ := x / (x^2 - x + 1)

theorem range_of_f :
  Set.range f = Set.Icc (-1/3 : ℝ) 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l279_27911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_with_special_sequences_l279_27983

-- Define a triangle with sides a, b, c and angles A, B, C
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the properties of the triangle
def is_valid_triangle (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.A + t.B + t.C = Real.pi

-- Define arithmetic sequence property for sides
def sides_form_arithmetic_sequence (t : Triangle) : Prop :=
  2 * t.b = t.a + t.c

-- Define geometric sequence property for sines of angles
def sines_form_geometric_sequence (t : Triangle) : Prop :=
  (Real.sin t.B) ^ 2 = Real.sin t.A * Real.sin t.C

-- Theorem statement
theorem triangle_with_special_sequences (t : Triangle)
  (h1 : is_valid_triangle t)
  (h2 : sides_form_arithmetic_sequence t)
  (h3 : sines_form_geometric_sequence t) :
  t.B = Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_with_special_sequences_l279_27983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_B_l279_27980

-- Define the function
noncomputable def f (B C : ℝ) (x : ℝ) : ℝ := Real.sin (B * x + C) + 1

-- State the theorem
theorem find_B : 
  ∃ (B C : ℝ), (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 * Real.pi → 
    f B C x = f B C (x + Real.pi)) ∧ B = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_B_l279_27980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_payment_is_correct_l279_27969

/-- Represents the internet service parameters and user's situation -/
structure InternetService where
  daily_cost : ℚ
  max_debt : ℚ
  days_connected : ℕ
  initial_balance : ℚ

/-- Calculates the minimum payment required for the given internet service -/
def minimum_payment (service : InternetService) : ℚ :=
  service.daily_cost * service.days_connected

/-- Theorem: The minimum payment for the given scenario is $12.5 -/
theorem minimum_payment_is_correct (service : InternetService) 
  (h1 : service.daily_cost = 1/2)
  (h2 : service.max_debt = 5)
  (h3 : service.days_connected = 25)
  (h4 : service.initial_balance = 0) :
  minimum_payment service = 25/2 := by
  sorry

#eval minimum_payment { daily_cost := 1/2, max_debt := 5, days_connected := 25, initial_balance := 0 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_payment_is_correct_l279_27969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_product_probability_l279_27952

def m : Finset Int := {-6, -5, -4, -3, -2}
def t : Finset Int := {-3, -2, -1, 0, 1, 2, 3, 4}

theorem negative_product_probability :
  let total_pairs := (m.card * t.card : ℚ)
  let negative_product_pairs := (m.card * (t.filter (λ x => x > 0)).card : ℚ)
  negative_product_pairs / total_pairs = 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_product_probability_l279_27952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_T_l279_27912

/-- The solid T in ℝ³ -/
def T : Set (ℝ × ℝ × ℝ) :=
  {p : ℝ × ℝ × ℝ | abs p.1 + abs p.2.1 ≤ 2 ∧ abs p.1 + abs p.2.2 ≤ 2 ∧ abs p.2.1 + abs p.2.2 ≤ 2}

/-- The volume of a set in ℝ³ -/
noncomputable def volume (S : Set (ℝ × ℝ × ℝ)) : ℝ := sorry

/-- The volume of T is 8/3 -/
theorem volume_of_T : volume T = 8/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_T_l279_27912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_calculation_l279_27919

-- Define the given values
noncomputable def trainSpeed : ℝ := 72  -- km/h
noncomputable def crossingTime : ℝ := 26  -- seconds
noncomputable def trainLength : ℝ := 270.0416  -- meters

-- Define the conversion factor
noncomputable def kmphToMps : ℝ := 5 / 18  -- Convert km/h to m/s

-- Define the platform length calculation
noncomputable def platformLength : ℝ :=
  trainSpeed * kmphToMps * crossingTime - trainLength

-- Theorem to prove
theorem platform_length_calculation :
  abs (platformLength - 249.9584) < 0.0001 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_calculation_l279_27919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2023_4_eq_neg_2_l279_27990

noncomputable def f (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0 => 2 * Real.sqrt x
  | n + 1 => 4 / (2 - f n x)

theorem f_2023_4_eq_neg_2 : f 2023 4 = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2023_4_eq_neg_2_l279_27990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l279_27984

/-- The area of a right triangle with vertices at (-10, 0), (0, 10), and (0, 0) is 50 square units -/
theorem triangle_area : ℝ := by
  -- Define the vertices of the triangle
  let v1 : ℝ × ℝ := (-10, 0)
  let v2 : ℝ × ℝ := (0, 10)
  let v3 : ℝ × ℝ := (0, 0)

  -- Define the function to calculate the area of a right triangle
  let area (base height : ℝ) : ℝ := (1/2) * base * height

  -- Calculate the base and height of the triangle
  let base : ℝ := v1.1 - v3.1  -- x-coordinate difference
  let height : ℝ := v2.2 - v3.2  -- y-coordinate difference

  -- Calculate the area
  let triangleArea : ℝ := area (abs base) (abs height)

  -- Assert that the area is 50 square units
  have h : triangleArea = 50 := by sorry

  -- Return the calculated area
  exact triangleArea


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l279_27984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_water_amount_l279_27941

/-- Represents the amount of water in the tank at each hour -/
def water_amount : Fin 5 → ℝ := sorry

/-- The rate at which water is lost per hour -/
def water_loss_rate : ℝ := 2

/-- The amount of water added at each hour -/
def water_added : Fin 5 → ℝ
  | 0 => 0  -- Initial state
  | 1 => 0  -- First hour
  | 2 => 0  -- Second hour
  | 3 => 1  -- Third hour
  | 4 => 3  -- Fourth hour

theorem initial_water_amount (h : water_amount 4 = 36) :
  water_amount 0 = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_water_amount_l279_27941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_is_line_and_circle_l279_27939

-- Define the polar equation
def polar_equation (ρ θ : ℝ) : Prop := ρ * Real.cos θ = 2 * Real.sin (2 * θ)

-- Define the curve
def curve : Set (ℝ × ℝ) :=
  {(x, y) | ∃ (ρ θ : ℝ), polar_equation ρ θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ}

-- Define what it means to be a line
def IsLine (s : Set (ℝ × ℝ)) : Prop :=
  ∃ (a b c : ℝ), s = {(x, y) | a * x + b * y + c = 0}

-- Define what it means to be a circle
def IsCircle (s : Set (ℝ × ℝ)) : Prop :=
  ∃ (x₀ y₀ r : ℝ), s = {(x, y) | (x - x₀)^2 + (y - y₀)^2 = r^2}

-- Statement of the theorem
theorem curve_is_line_and_circle :
  ∃ (l : Set (ℝ × ℝ)) (c : Set (ℝ × ℝ)),
    IsLine l ∧ IsCircle c ∧ curve = l ∪ c :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_is_line_and_circle_l279_27939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_on_real_line_l279_27964

open Set Real

/-- Given sets A and B on the real number line, prove the complement and intersection properties. -/
theorem set_operations_on_real_line :
  let A : Set ℝ := {x : ℝ | 3 ≤ x ∧ x < 7}
  let B : Set ℝ := {x : ℝ | 2 < x ∧ x < 6}
  (((A ∪ B)ᶜ) = {x : ℝ | x ≤ 2 ∨ x ≥ 7}) ∧
  ((Aᶜ ∩ B) = {x : ℝ | 2 < x ∧ x < 3}) :=
by
  intro A B
  constructor
  · sorry  -- Proof for (A ∪ B)ᶜ = {x : ℝ | x ≤ 2 ∨ x ≥ 7}
  · sorry  -- Proof for Aᶜ ∩ B = {x : ℝ | 2 < x ∧ x < 3}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_on_real_line_l279_27964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_instantaneous_velocity_at_2_l279_27982

-- Define the displacement function
noncomputable def S (t : ℝ) : ℝ := 2 * (1 - t)^2

-- Define the velocity function as the derivative of displacement
noncomputable def v (t : ℝ) : ℝ := deriv S t

-- Theorem stating that the instantaneous velocity at t = 2 is 4
theorem instantaneous_velocity_at_2 : v 2 = 4 := by
  -- Unfold the definitions of v and S
  unfold v S
  -- Calculate the derivative
  simp [deriv]
  -- Simplify the result
  norm_num
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_instantaneous_velocity_at_2_l279_27982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_circumference_approx_l279_27913

/-- The width of the circular race track in meters -/
def track_width : ℝ := 14

/-- The radius of the outer circle of the race track in meters -/
def outer_radius : ℝ := 84.02817496043394

/-- The radius of the inner circle of the race track in meters -/
def inner_radius : ℝ := outer_radius - track_width

/-- The inner circumference of the race track in meters -/
noncomputable def inner_circumference : ℝ := 2 * Real.pi * inner_radius

/-- Theorem stating that the inner circumference of the race track is approximately 440.12 meters -/
theorem inner_circumference_approx :
  abs (inner_circumference - 440.12) < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_circumference_approx_l279_27913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inradius_smaller_for_inner_triangle_l279_27995

-- Define triangles and their properties
structure Triangle where
  points : Fin 3 → ℝ × ℝ
  is_nondegenerate : Prop

-- Define the concept of one triangle being inside another
def triangle_inside (t1 t2 : Triangle) : Prop := sorry

-- Define the inradius of a triangle
noncomputable def inradius (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem inradius_smaller_for_inner_triangle {A B C A' B' C' : Triangle} :
  triangle_inside A A' →
  triangle_inside B A' →
  triangle_inside C A' →
  inradius A < inradius A' := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inradius_smaller_for_inner_triangle_l279_27995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_equality_implies_z_value_l279_27907

theorem mean_equality_implies_z_value :
  ∀ (x₁ x₂ x₃ y₁ z : ℚ),
  x₁ = 8 →
  x₂ = 7 →
  x₃ = 28 →
  y₁ = 14 →
  (x₁ + x₂ + x₃) / 3 = (y₁ + z) / 2 →
  z = 44 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_equality_implies_z_value_l279_27907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_tim_speed_ratio_l279_27955

-- Define Tim's and Tom's typing speeds
noncomputable def tim_speed : ℝ := sorry
noncomputable def tom_speed : ℝ := sorry

-- Define the conditions
axiom combined_normal_speed : tim_speed + tom_speed = 12
axiom combined_increased_speed : tim_speed + (1.25 * tom_speed) = 14

-- Define the ratio we want to prove
noncomputable def speed_ratio : ℝ := tom_speed / tim_speed

-- Theorem statement
theorem tom_tim_speed_ratio : speed_ratio = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_tim_speed_ratio_l279_27955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_expression_equality_second_expression_equality_l279_27934

-- Define the logarithm base 10
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- First expression
theorem first_expression_equality :
  1 / (Real.sqrt 2 - 1) - (3 / 5)^(0 : ℝ) + (9 / 4)^(-(1/2) : ℝ) + ((Real.sqrt 2 - Real.exp 1)^4)^(1/4) = 2/3 + Real.exp 1 :=
by sorry

-- Second expression
theorem second_expression_equality :
  lg 500 + lg (8/5) - 1/2 * lg 64 + 50 * (lg 2 + lg 5)^2 = 52 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_expression_equality_second_expression_equality_l279_27934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_number_on_blackboard_l279_27906

theorem final_number_on_blackboard : 
  let initial_numbers : List ℚ := List.range 2001 |>.map (λ i => 1 / (i + 1))
  let operation (x y : ℚ) := x + y + x * y
  ∃ (sequence : List (List ℚ)), 
    sequence.length = 2001 ∧ 
    sequence.head! = initial_numbers ∧
    (∀ i < 2000, 
      ∃ x y, x ∈ sequence[i]! ∧ y ∈ sequence[i]! ∧ x ≠ y ∧
      sequence[i+1]! = (sequence[i]!.filter (λ z => z ≠ x ∧ z ≠ y)) ++ [operation x y]) ∧
    sequence.getLast!.length = 1 ∧
    sequence.getLast!.head! = 2001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_number_on_blackboard_l279_27906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leftmost_vertex_is_nine_l279_27910

/-- The area of a pentagon with vertices on y = ln(x) and x-coordinates being consecutive integers --/
noncomputable def pentagon_area (n : ℕ) : ℝ :=
  Real.log ((n + 1 : ℝ) * (n + 2) * (n + 3) / (n * (n + 4)))

/-- Theorem stating that if the area of the pentagon is ln(23/21), then the leftmost x-coordinate is 9 --/
theorem leftmost_vertex_is_nine :
  ∃ (n : ℕ), n > 0 ∧ pentagon_area n = Real.log (23 / 21) → n = 9 := by
  sorry

#check leftmost_vertex_is_nine

end NUMINAMATH_CALUDE_ERRORFEEDBACK_leftmost_vertex_is_nine_l279_27910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_rotation_intersection_lengths_l279_27997

noncomputable def parabola (x : ℝ) : ℝ := x^2

noncomputable def θ : ℝ := Real.pi / 4

noncomputable def rotated_parabola (x : ℝ) : ℝ := 
  x * Real.cos θ + (parabola x) * Real.sin θ

noncomputable def x_intersections : Set ℝ := {x : ℝ | rotated_parabola x = 0}

noncomputable def y_intersections : Set ℝ := {y : ℝ | ∃ x, rotated_parabola x = y ∧ x * Real.cos θ + y * Real.sin θ = 0}

theorem parabola_rotation_intersection_lengths :
  (∃ a b, a ∈ x_intersections ∧ b ∈ x_intersections ∧ a ≠ b ∧ |b - a| = 1) →
  (∃ c d, c ∈ y_intersections ∧ d ∈ y_intersections ∧ c ≠ d ∧ |d - c| = 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_rotation_intersection_lengths_l279_27997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l279_27926

/-- A parabola with equation y = (1/2)x^2 - bx + c -/
structure Parabola where
  b : ℝ
  c : ℝ

/-- The y-coordinate for a given x on the parabola -/
noncomputable def y_coord (p : Parabola) (x : ℝ) : ℝ := (1/2) * x^2 - p.b * x + p.c

/-- A parabola satisfying the given conditions -/
def satisfying_parabola (p : Parabola) : Prop :=
  y_coord p 1 < 0 ∧ y_coord p 2 < 0

theorem parabola_properties (p : Parabola) (h : satisfying_parabola p) :
  (p.b^2 > 2 * p.c) ∧
  (p.c > 1 → p.b > 3/2) ∧
  (∀ m₁ m₂ n₁ n₂ : ℝ,
    y_coord p m₁ = n₁ →
    y_coord p m₂ = n₂ →
    m₁ < m₂ →
    m₂ < p.b →
    n₁ > n₂) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l279_27926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_proof_l279_27935

/-- Given α and β in (0,π), cos α = √5/5, and sin(α+β) = -√2/10, prove that 3α + β = 7π/4 -/
theorem angle_sum_proof (α β : ℝ) 
  (h1 : 0 < α ∧ α < Real.pi) 
  (h2 : 0 < β ∧ β < Real.pi) 
  (h3 : Real.cos α = Real.sqrt 5 / 5) 
  (h4 : Real.sin (α + β) = -(Real.sqrt 2) / 10) : 
  3 * α + β = 7 * Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_proof_l279_27935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_integer_solutions_of_equation_l279_27922

theorem positive_integer_solutions_of_equation :
  ∀ x y z : ℕ,
  x > 0 → y > 0 → z > 0 →
  x^2 + y^2 + 1 = 2^z →
  ((x = 1 ∧ y = 1 ∧ z = 1) ∨ (x = 1 ∧ y = 1 ∧ z = 1)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_integer_solutions_of_equation_l279_27922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_place_is_unnamed_l279_27993

-- Define the type for racers
inductive Racer
| Victor
| Elise
| Jane
| Kumar
| Lucas
| Henry
| Unnamed (n : Nat)

-- Define the function for finishing position
def finishPosition : Racer → Nat
| _ => 0  -- Default implementation, will be axiomatized later

-- Define the total number of racers
def totalRacers : Nat := 12

-- State the theorem
theorem fifth_place_is_unnamed :
  (finishPosition Racer.Lucas = finishPosition Racer.Kumar - 5) →
  (finishPosition Racer.Elise = finishPosition Racer.Jane + 1) →
  (finishPosition Racer.Victor = finishPosition Racer.Kumar + 3) →
  (finishPosition Racer.Jane = finishPosition Racer.Henry + 3) →
  (finishPosition Racer.Henry = finishPosition Racer.Lucas + 2) →
  (finishPosition Racer.Victor = 9) →
  (∃ (n : Nat), finishPosition (Racer.Unnamed n) = 5) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_place_is_unnamed_l279_27993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_m_values_l279_27951

/-- A line with equation x - 2y + m = 0 -/
structure TangentLine (m : ℝ) where
  eq : ℝ → ℝ → Prop
  eq_def : ∀ x y : ℝ, eq x y ↔ x - 2*y + m = 0

/-- A circle with equation x^2 + y^2 - 2y - 4 = 0 -/
structure Circle where
  eq : ℝ → ℝ → Prop
  eq_def : ∀ x y : ℝ, eq x y ↔ x^2 + y^2 - 2*y - 4 = 0

/-- The line is tangent to the circle -/
def is_tangent (l : TangentLine m) (c : Circle) : Prop :=
  ∃ x y : ℝ, l.eq x y ∧ c.eq x y ∧
    ∀ x' y' : ℝ, l.eq x' y' ∧ c.eq x' y' → (x' = x ∧ y' = y)

/-- Main theorem: If the line is tangent to the circle, then m = 7 or m = -3 -/
theorem tangent_line_m_values (m : ℝ) (l : TangentLine m) (c : Circle) :
  is_tangent l c → m = 7 ∨ m = -3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_m_values_l279_27951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parameterization_correctness_l279_27998

-- Define the line equation
def line_equation (x y : ℝ) : Prop := y = 3 * x - 5

-- Define the parameterization function
def parameterization (p : ℝ × ℝ) (v : ℝ × ℝ) (t : ℝ) : ℝ × ℝ :=
  (p.1 + t * v.1, p.2 + t * v.2)

-- Define the correctness of a parameterization
def correct_parameterization (p v : ℝ × ℝ) : Prop :=
  ∀ t : ℝ, let (x, y) := parameterization p v t; line_equation x y

-- Theorem statement
theorem parameterization_correctness :
  correct_parameterization (1, -2) (1/3, 1) ∧
  correct_parameterization (-2, -11) (1/3, 1) ∧
  ¬ correct_parameterization (5/3, 0) (1, 3) ∧
  ¬ correct_parameterization (0, -5) (3, 1) ∧
  ¬ correct_parameterization (-5/3, 0) (-1, -3) := by
  sorry

#check parameterization_correctness

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parameterization_correctness_l279_27998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l279_27954

noncomputable section

def f (x : ℝ) := Real.sqrt 2 * Real.sin (2 * x + Real.pi / 4) + 1

theorem f_properties :
  ∃ (T : ℝ) (min_val max_val : ℝ),
    (∀ x, f (x + T) = f x) ∧  -- smallest positive period
    (T = Real.pi) ∧
    (∀ k : ℤ, StrictMono (fun x => f x)) ∧  -- monotonically increasing interval
    (∀ x ∈ Set.Icc (-Real.pi / 4) (Real.pi / 4), f x ≥ min_val ∧ f x ≤ max_val) ∧  -- min and max on [-π/4, π/4]
    (f (-Real.pi / 4) = min_val) ∧
    (f (Real.pi / 8) = max_val) ∧
    (min_val = 0) ∧
    (max_val = Real.sqrt 2 + 1) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l279_27954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_comparison_l279_27966

noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℝ) (n : ℝ) : ℝ :=
  principal * (1 + rate / n) ^ (n * time)

noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

theorem loan_comparison : 
  let principal := (12000 : ℝ)
  let compound_rate := (0.08 : ℝ)
  let simple_rate := (0.1 : ℝ)
  let time := (12 : ℝ)
  let compound_freq := (2 : ℝ)

  let compound_balance_6years := compound_interest principal compound_rate 6 compound_freq
  let half_payment := compound_balance_6years / 2
  let remaining_balance := compound_balance_6years - half_payment
  let final_compound_payment := compound_interest remaining_balance compound_rate 6 compound_freq
  let total_compound_payment := half_payment + final_compound_payment

  let total_simple_payment := simple_interest principal simple_rate time

  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ |total_simple_payment - total_compound_payment - 1896| < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_comparison_l279_27966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersection_points_l279_27932

/-- A fourth degree polynomial function -/
def fourth_degree_poly (a b c d e : ℝ) : ℝ → ℝ := λ x ↦ a*x^4 + b*x^3 + c*x^2 + d*x + e

/-- A third degree polynomial function -/
def third_degree_poly (a b c d : ℝ) : ℝ → ℝ := λ x ↦ a*x^3 + b*x^2 + c*x + d

/-- The maximum number of real intersection points between a fourth degree polynomial
    and a third degree polynomial is 4 -/
theorem max_intersection_points (p : ℝ → ℝ) (q : ℝ → ℝ)
  (hp : ∃ a b c d e, p = fourth_degree_poly a b c d e)
  (hq : ∃ a b c d, q = third_degree_poly a b c d) :
  ∃ S : Finset ℝ, (∀ x ∈ S, p x = q x) ∧ S.card ≤ 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersection_points_l279_27932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_axis_of_f_l279_27953

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (x + 5 * Real.pi / 6) * Real.sin (x + Real.pi / 3)

theorem symmetric_axis_of_f :
  ∃ (k : ℤ), (∀ (x : ℝ), f (5 * Real.pi / 12 + x) = f (5 * Real.pi / 12 - x)) ∧
  (∀ (c : ℝ), c ≠ 5 * Real.pi / 12 → ¬(∀ (x : ℝ), f (c + x) = f (c - x))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_axis_of_f_l279_27953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_waiting_probability_l279_27902

/-- The probability of waiting no more than 10 minutes for a train that departs on average once per hour is 1/6. -/
theorem train_waiting_probability :
  (10 : ℝ) / 60 = 1 / 6 := by
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_waiting_probability_l279_27902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_equality_l279_27985

theorem exponent_equality (x : ℝ) : (3 : ℝ)^x + (3 : ℝ)^x + (3 : ℝ)^x + (3 : ℝ)^x = 729 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_equality_l279_27985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_root_sum_k_sum_possible_k_l279_27929

theorem common_root_sum_k : ∀ k : ℝ,
  (∃ x : ℝ, x^2 - 3*x + 2 = 0 ∧ x^2 - 5*x + k = 0) →
  (k = 4 ∨ k = 6) :=
by sorry

theorem sum_possible_k : 
  Finset.sum {4, 6} id = 10 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_root_sum_k_sum_possible_k_l279_27929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_l279_27974

-- Define the total revenue function
noncomputable def R (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 400 then 400 * x - (1/2) * x^2
  else 80000

-- Define the profit function
noncomputable def f (x : ℝ) : ℝ :=
  R x - (100 * x + 20000)

-- Theorem statement
theorem max_profit :
  ∃ (x : ℝ), x = 300 ∧ f x = 25000 ∧ ∀ (y : ℝ), f y ≤ f x := by
  sorry

#check max_profit

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_l279_27974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_cardinality_l279_27987

def A : Finset Int := {-1, 1, 2}
def B : Finset Int := {0, 1, 2, 7}

theorem union_cardinality : Finset.card (A ∪ B) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_cardinality_l279_27987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_price_is_60_l279_27976

/-- The cost price of a ball, given the conditions of the problem -/
noncomputable def cost_price_of_ball : ℚ :=
  let total_balls : ℕ := 17
  let selling_price : ℚ := 720
  let loss_balls : ℕ := 5
  let discount_rate : ℚ := 1/10
  let tax_rate : ℚ := 1/20
  let total_cost := selling_price + (selling_price / total_balls) * loss_balls
  total_cost / total_balls

/-- Theorem: The cost price of a ball is 60 -/
theorem cost_price_is_60 : cost_price_of_ball = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_price_is_60_l279_27976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_to_quaternary_l279_27940

-- Define the binary number
def binary_num : ℕ := 28

-- Define the quaternary number
def quaternary_num : List ℕ := [1, 3, 0]

-- Theorem statement
theorem binary_to_quaternary : 
  (binary_num = 1 * 2^4 + 1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 0 * 2^0) ∧ 
  (quaternary_num = [1, 3, 0]) ∧
  (binary_num = quaternary_num.foldl (fun acc d ↦ acc * 4 + d) 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_to_quaternary_l279_27940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_increase_l279_27942

/-- The increase in surface area when cutting a cube into 27 smaller cubes -/
theorem surface_area_increase (a : ℝ) (h : a > 0) : 
  6 * a^2 + 12 * a^2 = 48 * (6 * (a / 3)^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_increase_l279_27942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_prime_305_20B_l279_27944

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem unique_prime_305_20B : 
  ∃! B : ℕ, B ∈ ({1, 3, 5, 7, 9} : Set ℕ) ∧ is_prime (305200 + B) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_prime_305_20B_l279_27944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_purchase_price_value_price_equation_l279_27968

/-- The purchase price of a commodity -/
def purchase_price : ℝ := 800

/-- The retail price of the commodity -/
def retail_price : ℝ := 1100

/-- The discount rate applied to the retail price -/
def discount_rate : ℝ := 0.8

/-- The profit rate relative to the purchase price -/
def profit_rate : ℝ := 0.1

theorem purchase_price_value : 
  purchase_price = 800 :=
by
  -- Unfold the definition of purchase_price
  unfold purchase_price
  -- The definition directly gives us the result
  rfl

theorem price_equation :
  purchase_price * (1 + profit_rate) = retail_price * discount_rate :=
by
  -- Replace the variables with their values
  simp [purchase_price, retail_price, discount_rate, profit_rate]
  -- Evaluate the arithmetic
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_purchase_price_value_price_equation_l279_27968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_with_dual_base_angle_l279_27933

/-- A triangle with vertices A, B, and C -/
structure Triangle where
  A : Real
  B : Real
  C : Real
  sum_to_pi : A + B + C = Real.pi

/-- The dual triangle of a given triangle -/
structure DualTriangle (t : Triangle) where
  A₁ : Real
  B₁ : Real
  C₁ : Real
  dual_condition : Real.cos t.A / Real.sin A₁ = Real.cos t.B / Real.sin B₁ ∧ 
                   Real.cos t.B / Real.sin B₁ = Real.cos t.C / Real.sin C₁ ∧ 
                   Real.cos t.C / Real.sin C₁ = 1

/-- Theorem stating that an isosceles triangle with a dual triangle has a base angle of 3π/8 -/
theorem isosceles_with_dual_base_angle 
  (t : Triangle) 
  (d : DualTriangle t) 
  (h_isosceles : t.A = t.B) : 
  t.A = 3 * Real.pi / 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_with_dual_base_angle_l279_27933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_proof_l279_27920

/-- The volume of a pyramid with an equilateral triangular base of side length √2/2 and height 1 -/
noncomputable def pyramid_volume : ℝ := Real.sqrt 3 / 24

theorem pyramid_volume_proof :
  let base_side : ℝ := Real.sqrt 2 / 2
  let height : ℝ := 1
  let base_area : ℝ := (Real.sqrt 3 / 4) * base_side^2
  pyramid_volume = (1/3) * base_area * height := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_proof_l279_27920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l279_27900

theorem simplify_expression (x : ℝ) (h : x ≠ 0) :
  (6 / (5 * x^4)) * ((5 * x^3) / 4) = (3 / 2) * x^7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l279_27900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_implies_a_ge_one_l279_27949

/-- A cubic function with a parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 + x^2 + a*x - 5

/-- The derivative of f with respect to x -/
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*x + a

/-- f is monotonic on ℝ -/
def is_monotonic (a : ℝ) : Prop := ∀ x : ℝ, f' a x ≥ 0

theorem monotonic_implies_a_ge_one (a : ℝ) (h : is_monotonic a) : a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_implies_a_ge_one_l279_27949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_two_primes_six_dice_l279_27977

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def count_primes_on_die : ℕ := 4  -- 2, 3, 5, 7 are prime on a 10-sided die

def total_outcomes : ℕ := 10^6  -- 10 possibilities for each of 6 dice

def favorable_outcomes : ℕ := Nat.choose 6 2 * (count_primes_on_die^2) * ((10 - count_primes_on_die)^4)

theorem probability_two_primes_six_dice :
  (favorable_outcomes : ℚ) / (total_outcomes : ℚ) = 4860 / 15625 := by
  -- Proof steps would go here
  sorry

#eval favorable_outcomes
#eval total_outcomes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_two_primes_six_dice_l279_27977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_isosceles_points_l279_27975

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A square in a 2D plane -/
structure Square where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Check if a triangle is isosceles -/
def isIsosceles (P Q R : Point) : Prop :=
  (P.x - Q.x)^2 + (P.y - Q.y)^2 = (P.x - R.x)^2 + (P.y - R.y)^2

/-- The set of points P that form isosceles triangles with each side of the square -/
def isoscelesPoints (s : Square) : Set Point :=
  {P : Point | isIsosceles P s.A s.B ∧ isIsosceles P s.B s.C ∧ 
               isIsosceles P s.C s.D ∧ isIsosceles P s.D s.A}

/-- Theorem stating that there are exactly 9 points P that satisfy the conditions -/
theorem square_isosceles_points (s : Square) : 
  ∃ (points : Finset Point), points.card = 9 ∧ ∀ P, P ∈ points ↔ P ∈ isoscelesPoints s :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_isosceles_points_l279_27975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_passage_conclusions_l279_27916

/-- Represents a writer -/
structure MyWriter where
  skill : ℕ

/-- Represents a poet -/
structure Poet extends MyWriter

/-- Represents a word -/
structure Word where
  association : ℕ

/-- Represents the main idea of a passage -/
inductive MainIdea
| ChoosingWords
| PowerOfWords
| GreatWriters
| CarefulWordChoice

/-- The passage discussed in the problem -/
def passage : MainIdea := MainIdea.ChoosingWords

/-- A real poet is a master of words -/
axiom real_poet_mastery (p : Poet) : p.toMyWriter.skill ≥ 100

/-- The power of a word comes from its association -/
axiom word_power (w : Word) : w.association > 0

/-- The main idea of the passage is about choosing words -/
axiom passage_main_idea : passage = MainIdea.ChoosingWords

/-- Theorem combining all the statements -/
theorem passage_conclusions :
  (∀ p : Poet, p.toMyWriter.skill ≥ 100) ∧
  (∀ w : Word, w.association > 0) ∧
  (passage = MainIdea.ChoosingWords) := by
  constructor
  · intro p
    exact real_poet_mastery p
  constructor
  · intro w
    exact word_power w
  · exact passage_main_idea


end NUMINAMATH_CALUDE_ERRORFEEDBACK_passage_conclusions_l279_27916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_and_closeness_l279_27928

-- Define the power function f(x) passing through (√2, 2)
noncomputable def f (x : ℝ) : ℝ := x^2

-- Define g(x) = 2mx + 1/2
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := 2 * m * x + 1/2

-- Define the closeness property
def are_close (f g : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, a ≤ x ∧ x ≤ b → |f x - g x| ≤ 1

-- Main theorem
theorem power_function_and_closeness :
  (f (Real.sqrt 2) = 2) ∧
  (∃ x : ℝ, -π ≤ x ∧ x ≤ π ∧ |Real.sin x - x| > 1) ∧
  (∀ m : ℝ, are_close f (g m) 1 2 ↔ 7/8 ≤ m ∧ m ≤ Real.sqrt 6 / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_and_closeness_l279_27928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_point_in_sphere_l279_27903

theorem probability_point_in_sphere : 
  let cube_volume := (2 - (-2))^3
  let sphere_volume := (4/3) * Real.pi * 2^3
  let point_in_cube := {p : ℝ × ℝ × ℝ | -2 ≤ p.fst ∧ p.fst ≤ 2 ∧ -2 ≤ p.snd.fst ∧ p.snd.fst ≤ 2 ∧ -2 ≤ p.snd.snd ∧ p.snd.snd ≤ 2}
  let point_in_sphere := {p : ℝ × ℝ × ℝ | p.fst^2 + p.snd.fst^2 + p.snd.snd^2 ≤ 4}
  (sphere_volume / cube_volume : ℝ) = Real.pi / 6 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_point_in_sphere_l279_27903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chip_cost_proof_l279_27971

theorem chip_cost_proof (num_friends : ℕ) (num_bags : ℕ) (discount_percent : ℚ) 
  (h_friends : num_friends = 4)
  (h_bags : num_bags = 5)
  (h_discount : discount_percent = 15/100)
  (total_cost : ℕ)
  (h_multiple : ∃ k : ℕ, total_cost = 13 * k)
  (h_whole_payment : ∃ n : ℕ, (1 - discount_percent) * (total_cost : ℚ) / num_friends = n)
  : total_cost / num_bags = 208 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chip_cost_proof_l279_27971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_angle_implies_cot_difference_l279_27930

-- Define a triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a median
noncomputable def median (t : Triangle) (vertex : ℝ × ℝ) (opposite : ℝ × ℝ) : ℝ × ℝ :=
  (vertex.1 + opposite.1 / 2, vertex.2 + opposite.2 / 2)

-- Define the angle between two vectors
noncomputable def angle (v1 v2 : ℝ × ℝ) : ℝ := sorry

-- Define cotangent
noncomputable def cot (θ : ℝ) : ℝ := sorry

theorem median_angle_implies_cot_difference (t : Triangle) :
  let AD := median t t.A (t.B.1 + t.C.1, t.B.2 + t.C.2)
  angle AD (t.B.1 - t.C.1, t.B.2 - t.C.2) = π / 3 →
  |cot (angle (t.A.1 - t.B.1, t.A.2 - t.B.2) (t.C.1 - t.B.1, t.C.2 - t.B.2)) -
   cot (angle (t.A.1 - t.C.1, t.A.2 - t.C.2) (t.B.1 - t.C.1, t.B.2 - t.C.2))| = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_angle_implies_cot_difference_l279_27930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_ellipse_to_line_l279_27965

-- Define the ellipse C₁
def ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 4 = 1

-- Define the line C
def line (x y : ℝ) : Prop := x + 2*y = 10

-- Define the distance between a point (x, y) and the line
noncomputable def distance_to_line (x y : ℝ) : ℝ := 
  |x + 2*y - 10| / Real.sqrt 5

-- Theorem statement
theorem min_distance_ellipse_to_line :
  ∃ (x y : ℝ), ellipse x y ∧ 
  (∀ (x' y' : ℝ), ellipse x' y' → distance_to_line x y ≤ distance_to_line x' y') ∧
  distance_to_line x y = Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_ellipse_to_line_l279_27965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scaling_transform_sin_2x_l279_27909

/-- The scaling transformation applied to a curve -/
noncomputable def scaling_transform (f : ℝ → ℝ) (sx sy : ℝ) : ℝ → ℝ :=
  fun x => sy * f (x / sx)

/-- The original curve -/
noncomputable def original_curve : ℝ → ℝ :=
  fun x => Real.sin (2 * x)

theorem scaling_transform_sin_2x :
  scaling_transform original_curve 2 3 = fun x => 3 * Real.sin x := by
  ext x
  simp [scaling_transform, original_curve]
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scaling_transform_sin_2x_l279_27909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_element_implies_range_l279_27925

def A : Set ℝ := {0, 1}
def B (a : ℝ) : Set ℝ := {a^2, 2*a}

def C (a : ℝ) : Set ℝ := {x | ∃ x₁ x₂, x₁ ∈ A ∧ x₂ ∈ B a ∧ x = x₁ + x₂}

theorem largest_element_implies_range (a : ℝ) :
  (∀ x ∈ C a, x ≤ 2*a + 1) ∧ (2*a + 1 ∈ C a) → 0 < a ∧ a < 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_element_implies_range_l279_27925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_z_l279_27988

theorem smallest_positive_z (x z : ℝ) 
  (h1 : Real.cos x = -1) 
  (h2 : Real.cos (x + z) = Real.sqrt 3 / 2) : 
  ∀ w, w > 0 → Real.cos (x + w) = Real.sqrt 3 / 2 → w ≥ π / 6 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_z_l279_27988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exterior_angle_square_pentagon_l279_27948

-- Define the interior angle of a regular polygon
noncomputable def interior_angle (n : ℕ) : ℝ := (180 * (n - 2 : ℝ)) / n

-- Define the theorem
theorem exterior_angle_square_pentagon (square_interior : ℝ) (pentagon_interior : ℝ) :
  square_interior = 90 ∧ 
  pentagon_interior = interior_angle 5 → 
  360 - square_interior - pentagon_interior = 162 := by
  intro h
  have h1 : square_interior = 90 := h.left
  have h2 : pentagon_interior = interior_angle 5 := h.right
  have h3 : interior_angle 5 = 108 := by
    unfold interior_angle
    norm_num
  rw [h1, h2, h3]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exterior_angle_square_pentagon_l279_27948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_adam_students_in_ten_years_l279_27999

/-- The number of students Adam teaches in a given year -/
def students (year : ℕ) : ℕ :=
  if year = 1 then 40
  else if year = 2 then 60
  else if year ≤ 4 then 70
  else 70

/-- The total number of students Adam teaches over n years -/
def total_students (n : ℕ) : ℕ :=
  (List.range n).map (fun i => students (i + 1)) |>.sum

/-- Theorem stating that Adam teaches 650 students over 10 years -/
theorem adam_students_in_ten_years :
  total_students 10 = 650 := by
  sorry

#eval total_students 10  -- This will evaluate the result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_adam_students_in_ten_years_l279_27999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_examine_mixture_sufficient_l279_27917

/- Define the types of seeds and bags -/
inductive SeedType
| Poppy
| Millet
deriving Repr, BEq, Inhabited

instance : ToString SeedType where
  toString : SeedType → String
    | SeedType.Poppy => "Poppy"
    | SeedType.Millet => "Millet"

structure Bag where
  label : String
  contents : SeedType
deriving Repr, BEq, Inhabited

/- Define the problem setup -/
def bags : List Bag := [
  { label := "Poppy", contents := SeedType.Millet },
  { label := "Millet", contents := SeedType.Poppy },
  { label := "Mixture", contents := SeedType.Poppy }
]

/- All labels are incorrect -/
axiom labels_incorrect : ∀ b : Bag, b ∈ bags → b.label ≠ toString b.contents

/- Function to determine bag contents based on a single seed from the "Mixture" bag -/
def determine_contents (seed : SeedType) : List Bag := sorry

/- Theorem: Examining a seed from the "Mixture" bag is sufficient -/
theorem examine_mixture_sufficient :
  ∀ (seed : SeedType),
  seed = (bags.find? (λ b => b.label = "Mixture")).get!.contents →
  determine_contents seed = bags := by sorry

#check examine_mixture_sufficient

end NUMINAMATH_CALUDE_ERRORFEEDBACK_examine_mixture_sufficient_l279_27917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_complex_fraction_l279_27947

/-- The imaginary part of (2*i^3)/(2+i) is -4/5, where i is the imaginary unit. -/
theorem imaginary_part_of_complex_fraction :
  let i : ℂ := Complex.I
  let z : ℂ := (2 * i^3) / (2 + i)
  z.im = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_complex_fraction_l279_27947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_point_not_on_M_exists_regular_polygon_on_M_l279_27931

-- Define the family of lines M
def M (θ : Real) : Set (Real × Real) :=
  {(x, y) | x * Real.cos θ + (y - 2) * Real.sin θ = 1 ∧ 0 ≤ θ ∧ θ ≤ 2 * Real.pi}

-- Define the circle C
def C : Set (Real × Real) :=
  {(x, y) | x^2 + (y - 2)^2 = 1}

-- Statement 1: There exists a point not on any line in M
theorem exists_point_not_on_M :
  ∃ (P : Real × Real), ∀ θ, P ∉ M θ :=
sorry

-- Helper function to represent edges of a polygon
def edges (polygon : List (Real × Real)) : List (Set (Real × Real)) :=
  sorry

-- Statement 2: For any n ≥ 3, there exists a regular n-gon inscribed in C with sides on M
theorem exists_regular_polygon_on_M (n : Nat) (hn : n ≥ 3) :
  ∃ (polygon : List (Real × Real)), 
    (polygon.length = n) ∧ 
    (∀ v ∈ polygon, v ∈ C) ∧
    (∀ e ∈ edges polygon, ∃ θ, e ⊆ M θ) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_point_not_on_M_exists_regular_polygon_on_M_l279_27931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_triangle_hypotenuse_l279_27994

noncomputable def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

def roots (a b c : ℝ) : Set ℝ := {x | quadratic_function a b c x = 0}

noncomputable def vertex (a b c : ℝ) : ℝ × ℝ := (-b / (2 * a), quadratic_function a b c (-b / (2 * a)))

def is_right_isosceles_triangle (p₁ p₂ p₃ : ℝ × ℝ) : Prop := sorry

theorem quadratic_triangle_hypotenuse (a b c : ℝ) (h₁ : a ≠ 0) :
  let r := roots a b c
  let v := vertex a b c
  (∃ (x₁ x₂ : ℝ), x₁ ∈ r ∧ x₂ ∈ r ∧ is_right_isosceles_triangle (x₁, 0) (x₂, 0) v) →
  ∃ (h : ℝ), h = 2 ∧ h^2 = (x₁ - x₂)^2 + (v.2)^2 :=
by sorry

#check quadratic_triangle_hypotenuse

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_triangle_hypotenuse_l279_27994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l279_27945

def is_valid_f (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f (x + y) + 1 = f x + f y) ∧
  (f (1/2) = 0) ∧
  (∀ x > 1/2, f x < 0)

theorem f_properties {f : ℝ → ℝ} (hf : is_valid_f f) :
  (∀ x : ℝ, f x = 1/2 + 1/2 * f (2*x)) ∧
  (∀ n : ℕ, ∀ x ∈ Set.Icc (1/(2^(n+1:ℕ):ℝ)) (1/(2^n:ℝ)), f x ≤ 1 - 1/(2^n:ℝ)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l279_27945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l279_27921

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 - 4*x else -((-x)^2 - 4*(-x))

-- State the theorem
theorem max_value_of_f :
  (∀ x, f (-x) = -f x) →  -- f is odd
  (∀ x ≥ 0, f x = x^2 - 4*x) →  -- definition of f for x ≥ 0
  (∃ x ∈ Set.Icc (-4) 1, ∀ y ∈ Set.Icc (-4) 1, f y ≤ f x) →  -- maximum exists in [-4, 1]
  (∃ x ∈ Set.Icc (-4) 1, f x = 4) :=  -- maximum value is 4
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l279_27921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_threshold_value_l279_27959

/-- Represents the tax calculation system in Country X -/
structure TaxSystem where
  x : ℚ  -- The threshold amount
  income : ℚ  -- Total income
  tax : ℚ  -- Total tax paid

/-- Calculates the tax based on the given tax system -/
def calculate_tax (ts : TaxSystem) : ℚ :=
  (15 / 100) * min ts.x ts.income + (20 / 100) * max (ts.income - ts.x) 0

/-- Theorem stating that X = 40000 satisfies the given conditions -/
theorem tax_threshold_value (ts : TaxSystem) :
  ts.income = 50000 ∧ ts.tax = 8000 ∧ calculate_tax ts = ts.tax → ts.x = 40000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_threshold_value_l279_27959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_power_four_l279_27972

noncomputable def A : Matrix (Fin 2) (Fin 2) ℝ :=
  !![1, -Real.sqrt 3; Real.sqrt 3, 1]

theorem matrix_power_four :
  A ^ 4 = !![(-8 : ℝ), -8 * Real.sqrt 3; 8 * Real.sqrt 3, (-8 : ℝ)] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_power_four_l279_27972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_330_distance_is_sum_of_travels_l279_27957

/-- The distance between two cities A and B --/
def distance_between_cities : ℝ := 330

/-- The speed of the train from A to B in km/hr --/
def speed_train_A : ℝ := 60

/-- The speed of the train from B to A in km/hr --/
def speed_train_B : ℝ := 75

/-- The time train A travels in hours --/
def time_train_A : ℝ := 3

/-- The time train B travels in hours --/
def time_train_B : ℝ := 2

/-- Theorem stating that the distance between cities A and B is 330 km --/
theorem distance_is_330 : distance_between_cities = 330 := by
  rfl

/-- Theorem proving that the distance is the sum of distances traveled by both trains --/
theorem distance_is_sum_of_travels : 
  distance_between_cities = speed_train_A * time_train_A + speed_train_B * time_train_B := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_330_distance_is_sum_of_travels_l279_27957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_increase_sufficient_l279_27950

/-- Represents the speed increase needed for Vehicle A -/
noncomputable def speed_increase : ℝ := 5

/-- Initial speed of Vehicle A in mph -/
noncomputable def speed_A : ℝ := 80

/-- Speed of Vehicle B in mph -/
noncomputable def speed_B : ℝ := 60

/-- Initial speed of Vehicle C in mph -/
noncomputable def speed_C : ℝ := 70

/-- Deceleration rate of Vehicle C in mph per minute -/
noncomputable def decel_C : ℝ := 2

/-- Initial distance between Vehicle A and B in feet -/
noncomputable def distance_AB : ℝ := 40

/-- Initial distance between Vehicle A and C in feet -/
noncomputable def distance_AC : ℝ := 260

/-- Time for A to overtake B -/
noncomputable def time_AB : ℝ := distance_AB / (speed_A + speed_increase - speed_B)

/-- Function to calculate the relative speed between A and C at time t -/
noncomputable def rel_speed_AC (t : ℝ) : ℝ := (speed_A + speed_increase) + (speed_C - decel_C * t)

/-- Theorem stating that the speed increase is sufficient -/
theorem speed_increase_sufficient :
  ∀ t : ℝ, t ≥ 0 → time_AB < distance_AC / rel_speed_AC t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_increase_sufficient_l279_27950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_zero_zero_conditions_l279_27991

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (1 + x) + a * x * Real.exp (-x)

theorem tangent_slope_at_zero (a : ℝ) (h : a = 1) :
  (fun x ↦ (1 / (1 + x)) + Real.exp (-x) - x * Real.exp (-x)) 0 = 2 := by
  sorry

theorem zero_conditions (a : ℝ) :
  (∃! x, x ∈ Set.Ioo (-1 : ℝ) 0 ∧ f a x = 0) ∧
  (∃! x, x ∈ Set.Ioi 0 ∧ f a x = 0) ↔
  a < -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_zero_zero_conditions_l279_27991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l279_27986

theorem inequality_proof (a b : ℕ+) (c : ℝ) 
  (h : (a.val + 1 : ℝ) / (b.val + c) = (b.val : ℝ) / a.val) 
  (hc : c > 0) : 
  c ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l279_27986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_square_image_l279_27908

-- Define the transformations
noncomputable def u (x y : ℝ) : ℝ := Real.sin (Real.pi * x) * Real.cos (Real.pi * y)
def v (x y : ℝ) : ℝ := x + y - x * y

-- Define the unit square vertices
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (1, 1)
def C : ℝ × ℝ := (0, 1)

-- Define the transformation function
noncomputable def transform (p : ℝ × ℝ) : ℝ × ℝ :=
  (u p.1 p.2, v p.1 p.2)

-- Define the set of points in the unit square
def unitSquare : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}

-- Theorem: The image of the unit square is a vertical line segment from (0,0) to (0,1)
theorem unit_square_image :
  {p | ∃ q ∈ unitSquare, p = transform q} =
  {p | p.1 = 0 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_square_image_l279_27908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_24_16_in_terms_of_q_l279_27905

-- Define p and q as noncomputable
noncomputable def p : ℝ := Real.log 6 / Real.log 4
noncomputable def q : ℝ := Real.log 4 / Real.log 6

-- Theorem statement
theorem log_24_16_in_terms_of_q (p q : ℝ) (hp : p = Real.log 6 / Real.log 4) (hq : q = Real.log 4 / Real.log 6) :
  Real.log 16 / Real.log 24 = 2 * q / (q + 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_24_16_in_terms_of_q_l279_27905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_removed_percentage_is_correct_l279_27904

noncomputable def box_length : ℝ := 20
noncomputable def box_width : ℝ := 12
noncomputable def box_height : ℝ := 10
noncomputable def cube_side : ℝ := 4

noncomputable def box_volume : ℝ := box_length * box_width * box_height
noncomputable def cube_volume : ℝ := cube_side ^ 3
noncomputable def total_removed_volume : ℝ := 8 * cube_volume

noncomputable def volume_removed_percentage : ℝ := (total_removed_volume / box_volume) * 100

theorem volume_removed_percentage_is_correct :
  ∃ ε > 0, |volume_removed_percentage - 21.333| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_removed_percentage_is_correct_l279_27904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_notebooks_purchasable_l279_27978

/-- Represents the price of a notebook in dollars -/
def notebook_price : ℚ := 215/100

/-- Represents the discount rate applied every 6th notebook -/
def discount_rate : ℚ := 15/100

/-- Represents Lucy's total money in dollars -/
def total_money : ℚ := 2545/100

/-- Calculates the cost of a full cycle of 6 notebooks -/
def cycle_cost : ℚ := 5 * notebook_price + notebook_price * (1 - discount_rate)

/-- Theorem stating the maximum number of notebooks Lucy can buy -/
theorem max_notebooks_purchasable : 
  ∃ (n : ℕ), n ≤ 12 ∧ 
    (n : ℚ) * notebook_price ≤ total_money ∧ 
    ((n + 1) : ℚ) * notebook_price > total_money ∧
    ((n / 6) : ℚ).floor * cycle_cost + ((n % 6) : ℚ) * notebook_price ≤ total_money ∧
    ((n / 6) : ℚ).floor * cycle_cost + (((n % 6) + 1) : ℚ) * notebook_price > total_money :=
by
  sorry

#eval (12 : ℚ) * notebook_price ≤ total_money
#eval (13 : ℚ) * notebook_price > total_money

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_notebooks_purchasable_l279_27978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_segment_length_l279_27963

open Real

-- Define the curve C
noncomputable def curve_C (ρ θ : ℝ) : Prop := ρ * (sin θ)^2 = 4 * cos θ

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (2 + (sqrt 2 / 2) * t, (sqrt 2 / 2) * t)

-- Define point P
def point_P : ℝ × ℝ := (2, 0)

-- Theorem statement
theorem intersection_segment_length :
  ∀ A B : ℝ × ℝ,
  (∃ θ_A ρ_A, curve_C ρ_A θ_A ∧ A = (ρ_A * cos θ_A, ρ_A * sin θ_A)) →
  (∃ θ_B ρ_B, curve_C ρ_B θ_B ∧ B = (ρ_B * cos θ_B, ρ_B * sin θ_B)) →
  (∃ t_A, A = line_l t_A) →
  (∃ t_B, B = line_l t_B) →
  sqrt ((A.1 - point_P.1)^2 + (A.2 - point_P.2)^2) +
  sqrt ((B.1 - point_P.1)^2 + (B.2 - point_P.2)^2) =
  4 * sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_segment_length_l279_27963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sasha_coffee_concentration_l279_27937

/-- Represents the coffee mixing process with given parameters -/
structure CoffeeMixing where
  initial_volume : ℝ
  iterations : ℕ
  pour_volume : ℝ
  water_volume : ℝ

/-- Calculates the final concentration of coffee in the thermos -/
noncomputable def final_concentration (cm : CoffeeMixing) : ℝ :=
  1 - (1 - cm.pour_volume / cm.initial_volume) ^ cm.iterations

/-- The specific coffee mixing scenario described in the problem -/
def sasha_coffee : CoffeeMixing where
  initial_volume := 300
  iterations := 6
  pour_volume := 200
  water_volume := 200

/-- Theorem stating that the final concentration in Sasha's thermos is 1/5 -/
theorem sasha_coffee_concentration :
  final_concentration sasha_coffee = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sasha_coffee_concentration_l279_27937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_9_l279_27958

/-- An arithmetic sequence with its properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence (changed to ℚ for computability)
  d : ℚ      -- Common difference
  seq_def : ∀ n, a (n + 1) = a n + d  -- Definition of arithmetic sequence

/-- Sum of first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (seq.a 1 + seq.a n) / 2

/-- Main theorem -/
theorem arithmetic_sequence_sum_9 (seq : ArithmeticSequence) :
  2 * seq.a 7 = 5 + seq.a 9 → S seq 9 = 45 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_9_l279_27958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_is_one_fourth_l279_27960

/-- Represents a particle moving along the edges of a triangle -/
structure Particle where
  position : ℝ × ℝ
  speed : ℝ

/-- Represents the triangle ABC -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The path traced by the midpoint of the line segment joining two particles -/
def midpointPath (p1 p2 : Particle) : Set (ℝ × ℝ) := sorry

/-- The area of a set in ℝ² -/
noncomputable def area (s : Set (ℝ × ℝ)) : ℝ := sorry

/-- Helper function to convert Triangle to Set (ℝ × ℝ) -/
def triangleToSet (t : Triangle) : Set (ℝ × ℝ) :=
  {p | p = t.A ∨ p = t.B ∨ p = t.C}

/-- The theorem to be proved -/
theorem area_ratio_is_one_fourth (ABC : Triangle) (p1 p2 : Particle) (v : ℝ) :
  ABC.A = (0, 0) →
  ABC.B = (1, 0) →
  ABC.C = (0, 1) →
  p1.position = ABC.A →
  p2.position = ((ABC.B.1 + ABC.A.1) / 2, (ABC.B.2 + ABC.A.2) / 2) →
  p1.speed = v →
  p2.speed = 2 * v →
  (area (midpointPath p1 p2)) / (area (triangleToSet ABC)) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_is_one_fourth_l279_27960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_line_intersection_sum_reciprocals_l279_27973

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the left focus
def left_focus : ℝ × ℝ := (-1, 0)

-- Define the line with 60° inclination passing through the left focus
noncomputable def line (x y : ℝ) : Prop := y = Real.sqrt 3 * (x + 1)

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  ellipse A.1 A.2 ∧ ellipse B.1 B.2 ∧ line A.1 A.2 ∧ line B.1 B.2

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem ellipse_focus_line_intersection_sum_reciprocals
  (A B : ℝ × ℝ) (h : intersection_points A B) :
  1 / distance A left_focus + 1 / distance B left_focus = 4/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_line_intersection_sum_reciprocals_l279_27973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_product_theorem_l279_27967

def d (x : ℕ) : ℕ := (Nat.divisors x).card

def s (x : ℕ) : ℕ := (Nat.divisors x).sum id

theorem divisor_product_theorem (x : ℕ) : s x * d x = 96 ↔ x ∈ ({14, 15, 47} : Finset ℕ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_product_theorem_l279_27967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_integers_l279_27979

theorem sum_of_integers (x y : ℤ) 
  (h1 : x > 0) (h2 : y > 0) 
  (h3 : x - y = 6) 
  (h4 : x * y = 112) : 
  x + y = 22 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_integers_l279_27979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_value_l279_27936

theorem sin_2alpha_value (α β : ℝ) 
  (h1 : π / 2 < α) 
  (h2 : α < β) 
  (h3 : β < 3 * π / 4)
  (h4 : Real.cos (α - β) = 12 / 13)
  (h5 : Real.sin (α + β) = -3 / 5) : 
  Real.sin (2 * α) = -16 / 65 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_value_l279_27936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_real_number_line_bijection_l279_27927

-- Define the number line as a type alias for real numbers
def NumberLine := ℝ

-- Define the bijective function between real numbers and the number line
def realToNumberLine : ℝ → NumberLine := id

-- State the theorem
theorem real_number_line_bijection :
  Function.Bijective realToNumberLine :=
by
  -- Prove that realToNumberLine is injective
  apply Function.bijective_id


end NUMINAMATH_CALUDE_ERRORFEEDBACK_real_number_line_bijection_l279_27927
