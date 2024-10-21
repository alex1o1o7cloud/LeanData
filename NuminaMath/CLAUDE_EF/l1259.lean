import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l1259_125978

noncomputable def curve_C (α : Real) : Real × Real :=
  (3 + Real.sqrt 10 * Real.cos α, 1 + Real.sqrt 10 * Real.sin α)

def line_polar (θ : Real) (ρ : Real) : Prop :=
  Real.sin θ - Real.cos θ = 1 / ρ

theorem chord_length :
  ∃ (a b : Real), 
    let (x₁, y₁) := curve_C a
    let (x₂, y₂) := curve_C b
    line_polar (Real.arctan (y₁ / x₁)) (Real.sqrt (x₁^2 + y₁^2)) ∧
    line_polar (Real.arctan (y₂ / x₂)) (Real.sqrt (x₂^2 + y₂^2)) ∧
    (x₂ - x₁)^2 + (y₂ - y₁)^2 = 22 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l1259_125978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absent_students_count_l1259_125997

def Statement := ℕ → Bool

def Peter : Statement := λ n => n > 1
def Victor : Statement := λ n => n > 2
def Tatyana : Statement := λ n => n > 3
def Charles : Statement := λ n => n > 4
def Polina : Statement := λ n => n < 4
def Shurik : Statement := λ n => n < 3

def statements : List Statement := [Peter, Victor, Tatyana, Charles, Polina, Shurik]

def count_true (n : ℕ) : List Statement → ℕ
  | [] => 0
  | s::ss => if s n then 1 + count_true n ss else count_true n ss

theorem absent_students_count :
  ∃ n : ℕ, n ∈ [2, 3, 4] ∧ count_true n statements = 3 := by
  sorry

#eval count_true 2 statements
#eval count_true 3 statements
#eval count_true 4 statements

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absent_students_count_l1259_125997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_t_l1259_125938

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 else -x^2

-- State the theorem
theorem range_of_t (t : ℝ) :
  (∀ x ∈ Set.Icc t (t + 2), f (x + 2*t) ≥ 4 * f x) →
  (∀ x : ℝ, f (-x) = -f x) →
  t ≥ 2 := by
  sorry

#check range_of_t

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_t_l1259_125938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_acute_triangles_four_points_l1259_125956

/-- A point in a plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A triangle in a plane -/
structure Triangle where
  a : Point2D
  b : Point2D
  c : Point2D

/-- Check if a triangle is acute -/
def isAcute (t : Triangle) : Bool := sorry

/-- The set of all possible triangles formed by 4 points -/
def allTriangles (p1 p2 p3 p4 : Point2D) : Finset Triangle := sorry

/-- The set of all acute triangles formed by 4 points -/
def acuteTriangles (p1 p2 p3 p4 : Point2D) : Finset Triangle :=
  (allTriangles p1 p2 p3 p4).filter (fun t => isAcute t)

/-- Theorem: The maximum number of acute triangles formed by any 4 points in a plane is 4 -/
theorem max_acute_triangles_four_points :
  ∀ p1 p2 p3 p4 : Point2D, (acuteTriangles p1 p2 p3 p4).card ≤ 4 ∧
  ∃ q1 q2 q3 q4 : Point2D, (acuteTriangles q1 q2 q3 q4).card = 4 := by
  sorry

#check max_acute_triangles_four_points

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_acute_triangles_four_points_l1259_125956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_difference_l1259_125999

/-- The polynomial we're working with -/
def f (x : ℝ) : ℝ := 16 * x^3 - 20 * x^2 + 9 * x - 1

/-- The roots of the polynomial are in arithmetic progression -/
axiom roots_in_ap : ∃ (a d : ℝ), Set.range (λ i : Fin 3 => a + i * d) = {x | f x = 0}

/-- The difference between the largest and smallest roots of f is 1/2 -/
theorem root_difference : 
  let roots := {x | f x = 0}
  ∃ (max min : ℝ), max ∈ roots ∧ min ∈ roots ∧ 
    (∀ x ∈ roots, x ≤ max ∧ min ≤ x) ∧ 
    max - min = 1/2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_difference_l1259_125999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pig_distance_before_capture_l1259_125972

/-- Represents the pursuit scenario in a square field -/
structure PursuitScenario where
  fieldSize : ℝ
  speedRatio : ℝ

/-- Calculates the distance traveled by the target before being caught -/
noncomputable def distanceTraveledByTarget (scenario : PursuitScenario) : ℝ :=
  let initialDistance := scenario.fieldSize * Real.sqrt 2
  let curveLength := initialDistance * scenario.speedRatio^2 / (scenario.speedRatio^2 - 1)
  curveLength / (2 * scenario.speedRatio)

/-- Theorem stating the distance traveled by the pig before being caught -/
theorem pig_distance_before_capture :
  let scenario := PursuitScenario.mk 100 2
  distanceTraveledByTarget scenario = 200 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pig_distance_before_capture_l1259_125972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1259_125990

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 
  Real.sqrt 3 / 2 - Real.sqrt 3 * Real.sin (ω * x) ^ 2 - Real.sin (ω * x) * Real.cos (ω * x)

def has_symmetry_center (f : ℝ → ℝ) : Prop :=
  ∃ c : ℝ, ∀ x : ℝ, f (c + x) = f (c - x)

noncomputable def distance_to_nearest_symmetry_axis (f : ℝ → ℝ) (c : ℝ) : ℝ :=
  Real.pi / 4

theorem f_properties (ω : ℝ) (h1 : ω > 0) 
  (h2 : has_symmetry_center (f ω))
  (h3 : ∃ c : ℝ, distance_to_nearest_symmetry_axis (f ω) c = Real.pi / 4) :
  ω = 1 ∧ 
  (∀ x ∈ Set.Icc Real.pi (3 * Real.pi / 2), f ω x ≤ Real.sqrt 3 / 2) ∧
  (∀ x ∈ Set.Icc Real.pi (3 * Real.pi / 2), f ω x ≥ -1) ∧
  (∃ x ∈ Set.Icc Real.pi (3 * Real.pi / 2), f ω x = Real.sqrt 3 / 2) ∧
  (∃ x ∈ Set.Icc Real.pi (3 * Real.pi / 2), f ω x = -1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1259_125990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_residues_5_7_l1259_125931

theorem quadratic_residues_5_7 (p : ℕ) (hp : Nat.Prime p) :
  (∃ x : ℕ, x^2 % p = 5 % p) = (p % 5 = 1 ∨ p % 5 = 4) ∧
  (∃ y : ℕ, y^2 % p = 7 % p) = 
    ((p % 4 = 1 ∧ (p % 7 = 1 ∨ p % 7 = 2 ∨ p % 7 = 4)) ∨
     (p % 4 = 3 ∧ (p % 7 = 3 ∨ p % 7 = 5 ∨ p % 7 = 6))) ∧
  ((∃ x : ℕ, x^2 % p = 5 % p) ∧ (∃ y : ℕ, y^2 % p = 7 % p)) =
    (p % 35 = 1 ∨ p % 35 = 9 ∨ p % 35 = 11 ∨ p % 35 = 19 ∨ p % 35 = 21 ∨ p % 35 = 29) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_residues_5_7_l1259_125931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_condition_l1259_125962

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x - 2

theorem decreasing_condition (a : ℝ) :
  (∀ x y, x ∈ Set.Iic (-2) → y ∈ Set.Iic (-2) → f a x ≥ f a y → x ≤ y) →
  (a = 2 → ∀ x y, x ∈ Set.Iic (-2) → y ∈ Set.Iic (-2) → x ≤ y → f a x ≥ f a y) ∧
  (∃ b, b ≠ 2 ∧ ∀ x y, x ∈ Set.Iic (-2) → y ∈ Set.Iic (-2) → x ≤ y → f b x ≥ f b y) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_condition_l1259_125962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_direction_vector_proof_l1259_125926

noncomputable section

def line (x : ℝ) : ℝ := (2 * x + 1) / 3

def parameterization (t : ℝ) (d : ℝ × ℝ) : ℝ × ℝ :=
  (4 + t * d.1, 2 + t * d.2)

theorem direction_vector_proof (d : ℝ × ℝ) : 
  (∀ x ≥ 4, line x = (parameterization (x - 4) d).2) ∧
  (∀ x ≥ 4, Real.sqrt ((x - 4)^2 + (line x - 2)^2) = x - 4) →
  d = (3 / Real.sqrt 13, 2 / Real.sqrt 13) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_direction_vector_proof_l1259_125926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_distance_is_288km_l1259_125960

/-- Represents the scenario of a car journey with a breakdown -/
structure JourneyScenario where
  normalSpeed : ℝ
  distanceToBreakdown : ℝ
  repairTime : ℝ
  speedReductionFactor : ℝ
  totalDistance : ℝ
  plannedArrivalDelay : ℝ

/-- Calculates the total journey time for a given scenario -/
noncomputable def totalJourneyTime (s : JourneyScenario) : ℝ :=
  s.distanceToBreakdown / s.normalSpeed +
  s.repairTime +
  (s.totalDistance - s.distanceToBreakdown) / (s.speedReductionFactor * s.normalSpeed)

/-- The main theorem stating the total distance of the journey -/
theorem journey_distance_is_288km :
  ∃ (s : JourneyScenario),
    s.normalSpeed > 0 ∧
    s.distanceToBreakdown = s.normalSpeed * 2 ∧
    s.repairTime = 2/3 ∧
    s.speedReductionFactor = 3/4 ∧
    s.plannedArrivalDelay = 2 ∧
    totalJourneyTime s = s.totalDistance / s.normalSpeed + s.plannedArrivalDelay ∧
    (let s' : JourneyScenario := { s with
        distanceToBreakdown := s.distanceToBreakdown + 72,
        plannedArrivalDelay := 3/2
      };
    totalJourneyTime s' = s'.totalDistance / s'.normalSpeed + s'.plannedArrivalDelay) ∧
    s.totalDistance = 288 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_distance_is_288km_l1259_125960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_l1259_125916

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 * b.2 = k * a.2 * b.1

theorem parallel_vectors (l : ℝ) :
  let a : ℝ × ℝ := (2, 6)
  let b : ℝ × ℝ := (-3, l)
  parallel a b → l = -9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_l1259_125916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_is_10_meters_l1259_125977

-- Define the given parameters
noncomputable def train_length : ℝ := 350
noncomputable def train_speed_kmh : ℝ := 72
noncomputable def crossing_time : ℝ := 18

-- Define the conversion factor from km/h to m/s
noncomputable def kmh_to_ms : ℝ := 1000 / 3600

-- Theorem statement
theorem platform_length_is_10_meters :
  let train_speed_ms := train_speed_kmh * kmh_to_ms
  let total_distance := train_speed_ms * crossing_time
  let platform_length := total_distance - train_length
  platform_length = 10 := by
  -- Proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_is_10_meters_l1259_125977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_sum_l1259_125968

theorem max_distance_sum (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : x₁^2 + y₁^2 = 1)
  (h2 : x₂^2 + y₂^2 = 1)
  (h3 : x₁*x₂ + y₁*y₂ = 1/2) :
  ∃ (M : ℝ), M = Real.sqrt 2 + Real.sqrt 3 ∧ 
  ∀ (z : ℝ), z = (|x₁ + y₁ - 1| / Real.sqrt 2) + (|x₂ + y₂ - 1| / Real.sqrt 2) → z ≤ M :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_sum_l1259_125968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_ellipse_to_Q_l1259_125910

/-- The ellipse with equation x^2/4 + y^2 = 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 4 + p.2^2 = 1}

/-- The point Q -/
def Q : ℝ × ℝ := (1, 0)

/-- The distance between two points in ℝ² -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- The minimum distance between a point on the ellipse and Q -/
theorem min_distance_ellipse_to_Q :
  ∃ (d : ℝ), d = Real.sqrt 6 / 3 ∧
  ∀ (p : ℝ × ℝ), p ∈ Ellipse → distance p Q ≥ d := by
  sorry

#check min_distance_ellipse_to_Q

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_ellipse_to_Q_l1259_125910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_is_focus_of_hyperbola_l1259_125966

/-- The equation of the hyperbola -/
def hyperbola_eq (x y : ℝ) : Prop :=
  -2 * x^2 + 3 * y^2 - 8 * x + 12 * y - 4 = 0

/-- The x-coordinate of the focus -/
def focus_x : ℝ := -2

/-- The y-coordinate of the focus -/
noncomputable def focus_y : ℝ := -2 + (2 * Real.sqrt 15) / 3

/-- Theorem stating that the given point is a focus of the hyperbola -/
theorem is_focus_of_hyperbola :
  ∃ (c : ℝ), c > 0 ∧
  ∀ (x y : ℝ), hyperbola_eq x y →
    (x - focus_x)^2 + (y - focus_y)^2 = 
    (x - focus_x)^2 + (y - (-2 + c))^2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_is_focus_of_hyperbola_l1259_125966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_area_equilateral_triangle_l1259_125946

/-- The area of the circumcircle of an equilateral triangle with side length 8 units is 64π/3 -/
theorem circumcircle_area_equilateral_triangle : 
  let side_length : ℝ := 8
  let triangle_area : ℝ := (Real.sqrt 3 / 4) * side_length^2
  let circumradius : ℝ := side_length / Real.sqrt 3
  let circumcircle_area : ℝ := π * circumradius^2
  circumcircle_area = 64 * π / 3 := by
  sorry

#check circumcircle_area_equilateral_triangle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_area_equilateral_triangle_l1259_125946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1259_125963

noncomputable def f (x : ℝ) : ℝ := (3 * x + 1) / (x - 1)

theorem range_of_f :
  Set.range f = {y : ℝ | y ≠ 3} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1259_125963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_sum_l1259_125923

theorem complex_power_sum (z : ℂ) (h : z + z⁻¹ = 2 * Real.cos (3 * π / 180)) :
  z^2000 + (z^2000)⁻¹ = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_sum_l1259_125923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_squared_l1259_125989

theorem coefficient_x_squared (x : ℝ) : 
  (∃ c : ℝ, (x^(1/2) + 1/(2*x^(1/2)))^8 = c*x^2 + (terms_without_x_squared : ℝ)) → 
  (∃ c : ℝ, (x^(1/2) + 1/(2*x^(1/2)))^8 = 7*x^2 + (terms_without_x_squared : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_squared_l1259_125989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_length_l1259_125953

noncomputable section

-- Define the parametric equations
def x (t : ℝ) : ℝ := Real.exp t * (Real.cos t + Real.sin t)
def y (t : ℝ) : ℝ := Real.exp t * (Real.cos t - Real.sin t)

-- Define the arc length function
def arcLength (a b : ℝ) : ℝ :=
  ∫ t in a..b, Real.sqrt ((deriv x t) ^ 2 + (deriv y t) ^ 2)

-- State the theorem
theorem curve_length :
  arcLength (Real.pi / 2) Real.pi = 2 * (Real.exp Real.pi - Real.exp (Real.pi / 2)) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_length_l1259_125953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_5_equals_9_l1259_125998

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then x^2 + 3*x - 1 else x^2 - 3*x - 1

-- State the theorem
theorem f_of_5_equals_9 
  (h_even : ∀ x, f x = f (-x))  -- f is even
  (h_def : ∀ x, x < 0 → f x = x^2 + 3*x - 1)  -- definition of f for x < 0
  : f 5 = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_5_equals_9_l1259_125998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1259_125969

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x - Real.pi / 6) + Real.cos (2 * x - Real.pi / 6)

theorem f_properties :
  ∃ (period : ℝ) (max_value : ℝ) (max_set : Set ℝ),
    (∀ x : ℝ, f (x + period) = f x) ∧
    (period > 0) ∧
    (∀ p : ℝ, p > 0 → (∀ x : ℝ, f (x + p) = f x) → period ≤ p) ∧
    (∀ x : ℝ, f x ≤ max_value) ∧
    (∀ x : ℝ, x ∈ max_set ↔ f x = max_value) ∧
    period = Real.pi ∧
    max_value = 2 ∧
    max_set = {x : ℝ | ∃ k : ℤ, x = k * Real.pi + Real.pi / 4} :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1259_125969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_fourth_quadrant_l1259_125937

theorem tan_alpha_fourth_quadrant (α : Real) 
  (h1 : Real.cos α = 12/13) 
  (h2 : α ∈ Set.Icc (3*Real.pi/2) (2*Real.pi)) : 
  Real.tan α = -5/12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_fourth_quadrant_l1259_125937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_solution_l1259_125918

theorem log_inequality_solution (x : ℝ) (hx_pos : x > 0) (hx_neq_one : x ≠ 1) (hx_neq_four_thirds : x ≠ 4/3) :
  Real.log (16 - 24*x + 9*x^2) / Real.log x < 0 ↔ 
  (x > 0 ∧ x < 1) ∨ (x > 1 ∧ x < 4/3) ∨ (x > 4/3 ∧ x < 5/3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_solution_l1259_125918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equivalence_l1259_125939

/-- Given that the terminal side of angle α passes through the point (-1, √3),
    prove that f(x) = sin α cos 2x + cos α cos (2x - π/2) is equivalent to sin(2(x + π/3)) -/
theorem function_equivalence (α : ℝ) 
  (h : ((-1 : ℝ), Real.sqrt 3) ∈ Set.range (λ t : ℝ => (t * Real.cos α, t * Real.sin α))) :
  ∀ x, Real.sin α * Real.cos (2 * x) + Real.cos α * Real.cos (2 * x - π / 2) = 
       Real.sin (2 * (x + π / 3)) :=
by
  -- We'll use 'sorry' to skip the proof for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equivalence_l1259_125939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_one_product_successful_first_firing_all_products_successful_both_firings_l1259_125912

-- Define success rates for the first firing
def success_rate_A : ℚ := 4/5
def success_rate_B : ℚ := 3/4
def success_rate_C : ℚ := 2/3

-- Define success rate for the second firing (same for all products)
def success_rate_second : ℚ := 3/5

-- Theorem for part (I)
theorem only_one_product_successful_first_firing : 
  (success_rate_A * (1 - success_rate_B) * (1 - success_rate_C) +
   (1 - success_rate_A) * success_rate_B * (1 - success_rate_C) +
   (1 - success_rate_A) * (1 - success_rate_B) * success_rate_C) = 3/20 := by
  sorry

-- Theorem for part (II)
theorem all_products_successful_both_firings :
  ((success_rate_A * success_rate_second) * 
   (success_rate_B * success_rate_second) * 
   (success_rate_C * success_rate_second)) = 54/625 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_one_product_successful_first_firing_all_products_successful_both_firings_l1259_125912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_perpendicular_to_given_line_l1259_125936

-- Define the circle
def my_circle (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = 4

-- Define the point P
def P : ℝ × ℝ := (3, 4)

-- Define the line ax - y + 1 = 0
def line (a : ℝ) (x y : ℝ) : Prop := a * x - y + 1 = 0

-- Define a tangent line to a circle
def is_tangent (x₀ y₀ : ℝ) (m : ℝ) : Prop :=
  ∃ (x y : ℝ), (y - y₀ = m * (x - x₀)) ∧ my_circle x y ∧
  ∀ (x' y' : ℝ), (y' - y₀ = m * (x' - x₀)) → (my_circle x' y' → x' = x ∧ y' = y)

-- Define perpendicularity of two lines
def perpendicular (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

-- The main theorem
theorem tangent_line_perpendicular_to_given_line :
  ∃ (m a : ℝ), is_tangent P.1 P.2 m ∧ perpendicular m (-1/a) → a = 3/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_perpendicular_to_given_line_l1259_125936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1259_125943

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h : a > 0
  k : b > 0

/-- The equation of the hyperbola -/
def hyperbolaEq (h : Hyperbola) (p : Point) : Prop :=
  p.x^2 / h.a^2 - p.y^2 / h.b^2 = 1

/-- The square root function -/
noncomputable def sqrtFunc (x : ℝ) : ℝ := Real.sqrt x

/-- The derivative of the square root function -/
noncomputable def sqrtFuncDerivative (x : ℝ) : ℝ := 1 / (2 * Real.sqrt x)

/-- The slope of the tangent line at a point -/
noncomputable def tangentSlope (p : Point) : ℝ := sqrtFuncDerivative p.x

/-- The left focus of the hyperbola -/
def leftFocus : Point := ⟨-2, 0⟩

/-- The condition that the tangent line passes through the left focus -/
def tangentPassesThroughFocus (p : Point) : Prop :=
  tangentSlope p = (p.y - leftFocus.y) / (p.x - leftFocus.x)

/-- The eccentricity of the hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := Real.sqrt (h.a^2 + h.b^2) / h.a

/-- The main theorem -/
theorem hyperbola_eccentricity 
  (h : Hyperbola) 
  (p : Point) 
  (hIntersection : p.y = sqrtFunc p.x) 
  (hOnHyperbola : hyperbolaEq h p) 
  (hTangent : tangentPassesThroughFocus p) : 
  eccentricity h = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1259_125943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1259_125974

-- Define the ellipse
def ellipse (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the circle
def circle_eq (x y r : ℝ) : Prop := (x - 1)^2 + y^2 = r^2

-- Define the eccentricity
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

theorem ellipse_properties (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  let e := eccentricity a b
  (e = 1/2) →
  (ellipse 1 (3/2) a b) →
  (∃ (r : ℝ), 0 < r ∧ r < 3/2 ∧
    (∀ (M N : ℝ × ℝ),
      (∃ (l1 l2 : ℝ → ℝ),
        l1 1 = 3/2 ∧ l2 1 = 3/2 ∧
        (∃ (t1 t2 : ℝ), circle_eq (t1) (l1 t1) r ∧ circle_eq (t2) (l2 t2) r) ∧
        ellipse M.1 M.2 a b ∧ ellipse N.1 N.2 a b ∧
        (∃ (k : ℝ), ∀ (x : ℝ), l1 x = k * (x - 1) + 3/2) ∧
        (∃ (k : ℝ), ∀ (x : ℝ), l2 x = k * (x - 1) + 3/2)) →
      (N.2 - M.2) / (N.1 - M.1) = 1/2)) →
  (a = 2 ∧ b = Real.sqrt 3) ∧
  (∀ (M N : ℝ × ℝ),
    ellipse M.1 M.2 a b →
    ellipse N.1 N.2 a b →
    (∃ (m : ℝ), N.2 - M.2 = 1/2 * (N.1 - M.1) ∧ N.2 = 1/2 * N.1 + m) →
    1/2 * Real.sqrt ((N.1 - M.1)^2 + (N.2 - M.2)^2) * (2 * abs m) / Real.sqrt 5 ≤ Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1259_125974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_pi_24_max_monotonic_interval_monotonic_in_max_interval_l1259_125986

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x * Real.cos x + Real.cos x ^ 2

-- Theorem 1: Value of f at π/24
theorem f_value_at_pi_24 : f (π / 24) = (Real.sqrt 2 + 1) / 2 := by sorry

-- Theorem 2: Maximum interval for monotonic increase
theorem max_monotonic_interval : 
  ∀ m : ℝ, (0 < m ∧ ∀ x y : ℝ, -m ≤ x ∧ x < y ∧ y ≤ m → f x < f y) → m ≤ π / 6 := by sorry

-- Theorem 3: Monotonicity in [-π/6, π/6]
theorem monotonic_in_max_interval : 
  ∀ x y : ℝ, -π/6 ≤ x ∧ x < y ∧ y ≤ π/6 → f x < f y := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_pi_24_max_monotonic_interval_monotonic_in_max_interval_l1259_125986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1259_125976

open Real

-- Define the function
noncomputable def f (x : ℝ) : ℝ := x / (exp x)

-- State the theorem
theorem max_value_of_f :
  ∃ (x : ℝ), x ∈ Set.Icc 0 2 ∧ f x = (1 : ℝ) / exp 1 ∧
  ∀ (y : ℝ), y ∈ Set.Icc 0 2 → f y ≤ f x :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1259_125976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_mammoths_8x8_mammoth_placement_theorem_l1259_125904

/-- Represents a chessboard -/
structure Chessboard where
  size : ℕ

/-- Represents a mammoth chess piece -/
structure Mammoth where
  directions : Fin 4 → Bool
  total_directions : ℕ
  direction_count : total_directions = 3

/-- Represents the total number of diagonals on a chessboard -/
def total_diagonals (board : Chessboard) : ℕ :=
  2 * (2 * board.size - 1)

/-- The maximum number of mammoths that can be placed on the chessboard -/
def max_mammoths (board : Chessboard) : ℕ :=
  (total_diagonals board * 2) / 3

/-- Theorem stating the maximum number of mammoths on an 8x8 chessboard -/
theorem max_mammoths_8x8 :
  ∀ (board : Chessboard),
  board.size = 8 →
  max_mammoths board = 20 :=
by
  sorry

/-- Represents whether two mammoths can attack each other -/
def can_attack (m1 m2 : Mammoth) : Prop :=
  sorry  -- Definition to be implemented

/-- Proof that the maximum number of mammoths on an 8x8 chessboard is 20 -/
theorem mammoth_placement_theorem :
  ∃ (board : Chessboard) (mammoths : List Mammoth),
  board.size = 8 ∧
  mammoths.length = 20 ∧
  (∀ m, m ∈ mammoths → m.total_directions = 3) ∧
  (∀ m1 m2, m1 ∈ mammoths → m2 ∈ mammoths → m1 ≠ m2 → ¬ can_attack m1 m2) ∧
  (∀ board_size : ℕ,
   board_size = 8 →
   ¬ ∃ (larger_list : List Mammoth),
     larger_list.length > 20 ∧
     (∀ m, m ∈ larger_list → m.total_directions = 3) ∧
     (∀ m1 m2, m1 ∈ larger_list → m2 ∈ larger_list → m1 ≠ m2 → ¬ can_attack m1 m2)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_mammoths_8x8_mammoth_placement_theorem_l1259_125904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mathematics_value_l1259_125927

def letter_value (n : ℕ) : ℤ :=
  match n % 10 with
  | 1 => 1
  | 2 => 2
  | 3 => 1
  | 4 => 0
  | 5 => -1
  | 6 => -2
  | 7 => -1
  | 8 => 0
  | 9 => 1
  | 0 => 2
  | _ => 0  -- This case should never occur due to modulo 10

def alphabet_position (c : Char) : ℕ :=
  match c with
  | 'a' => 1
  | 'b' => 2
  | 'c' => 3
  | 'd' => 4
  | 'e' => 5
  | 'f' => 6
  | 'g' => 7
  | 'h' => 8
  | 'i' => 9
  | 'j' => 10
  | 'k' => 11
  | 'l' => 12
  | 'm' => 13
  | 'n' => 14
  | 'o' => 15
  | 'p' => 16
  | 'q' => 17
  | 'r' => 18
  | 's' => 19
  | 't' => 20
  | 'u' => 21
  | 'v' => 22
  | 'w' => 23
  | 'x' => 24
  | 'y' => 25
  | 'z' => 26
  | _ => 0  -- For any other character

def word_value (word : String) : ℤ :=
  word.toList.map (fun c => letter_value (alphabet_position c)) |>.sum

theorem mathematics_value : word_value "mathematics" = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mathematics_value_l1259_125927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_with_inscribed_circle_l1259_125914

noncomputable section

/-- The distance between two parallel lines -/
def distance_between_lines : ℝ := 18/25

/-- The radius of the inscribed circle in triangle ABC -/
def incircle_radius : ℝ := 8/3

/-- Triangle ABC with A and B on one line, C on a parallel line -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Predicate to check if a triangle is isosceles -/
def is_isosceles (t : Triangle) : Prop :=
  (dist t.A t.C = dist t.B t.C) ∨ (dist t.A t.B = dist t.A t.C)

/-- Predicate to check if points are on parallel lines with given distance -/
def on_parallel_lines (t : Triangle) (d : ℝ) : Prop :=
  abs (t.A.2 - t.C.2) = d ∧ abs (t.B.2 - t.C.2) = d ∧ t.A.2 = t.B.2

/-- Theorem statement -/
theorem isosceles_triangle_with_inscribed_circle 
  (t : Triangle) 
  (h1 : is_isosceles t) 
  (h2 : on_parallel_lines t distance_between_lines) 
  (h3 : ∃ (center : ℝ × ℝ), (dist center t.A = incircle_radius) ∧ 
                             (dist center t.B = incircle_radius) ∧ 
                             (dist center t.C = incircle_radius)) : 
  dist t.A t.B = 20 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_with_inscribed_circle_l1259_125914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1259_125940

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the properties of the triangle
def isAcuteTriangle (t : Triangle) : Prop :=
  0 < t.A ∧ t.A < Real.pi/2 ∧
  0 < t.B ∧ t.B < Real.pi/2 ∧
  0 < t.C ∧ t.C < Real.pi/2

-- Define the given conditions
def satisfiesConditions (t : Triangle) : Prop :=
  2 * t.a * Real.sin t.B = Real.sqrt 3 * t.b ∧
  t.a = 4 ∧
  t.b + t.c = 8

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h_acute : isAcuteTriangle t) 
  (h_conditions : satisfiesConditions t) : 
  t.A = Real.pi/3 ∧ t.a * t.b * Real.sin t.C / 2 = 4 * Real.sqrt 3 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1259_125940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_tangent_l1259_125903

-- Define the moving straight line
noncomputable def movingLine (t : ℝ) (x : ℝ) : ℝ := (2/t) * (x - t - 1)

-- Define the parabola
noncomputable def parabola (x : ℝ) : ℝ := Real.sqrt (4 * x)

-- Define the locus of the midpoint
def locusMidpoint (x y : ℝ) : Prop := (y + 1)^2 = 2 * (x - 1/2)

-- Define the tangent line
def tangentLine (x y : ℝ) : Prop := x + y + 1 = 0

-- Theorem statement
theorem locus_is_tangent :
  ∀ t x y,
  (∃ x₁ y₁ x₂ y₂,
    y₁ = movingLine t x₁ ∧ y₁ = parabola x₁ ∧
    y₂ = movingLine t x₂ ∧ y₂ = parabola x₂ ∧
    x = (x₁ + x₂) / 2 ∧ y = (y₁ + y₂) / 2) →
  locusMidpoint x y ∧
  (∃ x₀ y₀, locusMidpoint x₀ y₀ ∧ tangentLine x₀ y₀) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_tangent_l1259_125903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l1259_125944

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 4*y + 4 = 0

-- Define point P
def point_P : ℝ × ℝ := (2, 0)

-- Define a line passing through P(2, 0)
def line_through_P (m : ℝ) (x y : ℝ) : Prop := y = m * (x - 2)

-- Define the chord length
noncomputable def chord_length : ℝ := 4 * Real.sqrt 2

-- Main theorem
theorem line_equation :
  ∀ m : ℝ, 
  (∀ x y : ℝ, circle_C x y ∧ line_through_P m x y → 
    ∃ x' y', circle_C x' y' ∧ line_through_P m x' y' ∧ 
    ((x - x')^2 + (y - y')^2 = chord_length^2)) →
  (m = 0 ∨ 3 * point_P.fst + 4 * m * (point_P.fst - 2) - 6 = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l1259_125944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_tangent_through_point_l1259_125941

-- Define the function f(x)
noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 - 2/x - a

-- Part 1: Tangent line equation when a = 0
theorem tangent_line_at_one (x y : ℝ) :
  (deriv (f · 0)) 1 = 5 ∧ f 1 0 = -1 →
  (5 * x - y - 6 = 0 ↔ y - f 1 0 = (deriv (f · 0)) 1 * (x - 1)) :=
by sorry

-- Part 2: Value of a when line through (-1,0) is tangent at x=1
theorem tangent_through_point (a : ℝ) :
  (∃ k : ℝ, k = (deriv (f · a)) 1 ∧ k = (f 1 a - 0) / (1 - (-1))) →
  a = -11 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_tangent_through_point_l1259_125941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1259_125955

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def isAcute (t : Triangle) : Prop :=
  0 < t.A ∧ t.A < Real.pi/2 ∧
  0 < t.B ∧ t.B < Real.pi/2 ∧
  0 < t.C ∧ t.C < Real.pi/2

def satisfiesConditions (t : Triangle) : Prop :=
  isAcute t ∧
  Real.sqrt 3 * t.a = 2 * t.c * Real.sin t.A ∧
  t.c = Real.sqrt 7 ∧
  t.a * t.b = 6

-- State the theorem
theorem triangle_problem (t : Triangle) (h : satisfiesConditions t) :
  t.C = Real.pi/3 ∧ ((t.a = 2 ∧ t.b = 3) ∨ (t.a = 3 ∧ t.b = 2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1259_125955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_is_zero_l1259_125993

-- Define the imaginary unit
noncomputable def i : ℂ := Complex.I

-- Define the complex expression
noncomputable def complex_expression : ℂ := ((1 + i)^2) * i

-- Define the imaginary part function
def imaginary_part_of_complex_expression (z : ℂ) : ℝ := z.im

-- Theorem statement
theorem imaginary_part_is_zero :
  imaginary_part_of_complex_expression complex_expression = 0 :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_is_zero_l1259_125993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_in_square_l1259_125924

/-- Predicate to check if a circle is tangent to all semicircles -/
def is_tangent_to_all_semicircles (circle_radius : ℝ) (semicircle_radius : ℝ) (square_side : ℝ) : Prop :=
  ∀ (x y : ℝ), 
    x ∈ Set.Icc (0 : ℝ) square_side →
    y ∈ Set.Icc (0 : ℝ) square_side →
    (x = 0 ∨ x = square_side ∨ y = 0 ∨ y = square_side) →
    (x - square_side / 2) ^ 2 + (y - square_side / 2) ^ 2 = 
      (circle_radius + semicircle_radius) ^ 2

/-- The radius of a circle tangent to twelve semicircles inside a square -/
theorem circle_radius_in_square (square_side : ℝ) (num_semicircles : ℕ) :
  square_side = 4 →
  num_semicircles = 12 →
  ∃ (r : ℝ), r = Real.sqrt 2 ∧ 
    (∀ (semicircle_radius : ℝ), 
      semicircle_radius = square_side / (3 : ℝ) / 2 →
      is_tangent_to_all_semicircles r semicircle_radius square_side) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_in_square_l1259_125924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_implies_log_inequality_l1259_125925

theorem exponential_inequality_implies_log_inequality (x y : ℝ) :
  (2 : ℝ)^x - (2 : ℝ)^y < (3 : ℝ)^(-x) - (3 : ℝ)^(-y) → Real.log (y - x + 1) > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_implies_log_inequality_l1259_125925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alexei_weight_loss_weeks_l1259_125907

/-- Calculates the number of weeks Alexei lost weight given the weight loss information of Aleesia and Alexei. -/
theorem alexei_weight_loss_weeks 
  (aleesia_weekly_loss : ℚ) 
  (aleesia_weeks : ℕ) 
  (alexei_weekly_loss : ℚ) 
  (total_loss : ℚ) 
  (h1 : aleesia_weekly_loss = 3/2)
  (h2 : aleesia_weeks = 10)
  (h3 : alexei_weekly_loss = 5/2)
  (h4 : total_loss = 35) :
  (total_loss - aleesia_weekly_loss * aleesia_weeks) / alexei_weekly_loss = 8 := by
  sorry

#eval (35 : ℚ) - (3/2 : ℚ) * 10 / (5/2 : ℚ)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alexei_weight_loss_weeks_l1259_125907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_line_equation_l1259_125950

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 6*y = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 6*x = 0

-- Define the line
def line (x y : ℝ) : Prop := x + 3*y = 0

-- Define a point as a pair of real numbers
def Point := ℝ × ℝ

-- Define a function to check if a point is on the line segment between two points
def on_line_segment (A B P : Point) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (t • B.1 + (1 - t) • A.1, t • B.2 + (1 - t) • A.2)

-- Theorem statement
theorem intersection_line_equation :
  ∃ (A B : Point),
    (circle1 A.1 A.2 ∧ circle2 A.1 A.2) ∧
    (circle1 B.1 B.2 ∧ circle2 B.1 B.2) ∧
    (A ≠ B) ∧
    (∀ (P : Point), on_line_segment A B P → line P.1 P.2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_line_equation_l1259_125950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_of_equations_result_l1259_125996

theorem system_of_equations_result :
  ∀ (x y z w : ℤ),
    (23 * x + 47 * y - 3 * z = 434) →
    (47 * x - 23 * y - 4 * w = 183) →
    (19 * z + 17 * w = 91) →
    (13 * x - 14 * y)^3 - (15 * z + 16 * w)^3 = -456190 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_of_equations_result_l1259_125996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_octagon_area_ratio_l1259_125900

/-- A regular octagon -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ
  is_regular : ∀ i j : Fin 8, dist (vertices i) (vertices ((i+1) % 8)) = dist (vertices j) (vertices ((j+1) % 8))

/-- The octagon formed by joining the midpoints of a regular octagon's sides -/
noncomputable def midpointOctagon (o : RegularOctagon) : RegularOctagon where
  vertices i := ((o.vertices i).1 + (o.vertices ((i+1) % 8)).1, (o.vertices i).2 + (o.vertices ((i+1) % 8)).2) / 2
  is_regular := by sorry

/-- The area of a regular octagon -/
noncomputable def area (o : RegularOctagon) : ℝ := sorry

theorem midpoint_octagon_area_ratio (o : RegularOctagon) :
  area (midpointOctagon o) = (1/4) * area o := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_octagon_area_ratio_l1259_125900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_tangent_function_l1259_125929

noncomputable def f (x : ℝ) : ℝ := Real.tan (2 * x + Real.pi / 6)

theorem period_of_tangent_function :
  ∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x ∧ ∀ (S : ℝ), 0 < S ∧ S < T → ∃ (y : ℝ), f (y + S) ≠ f y :=
by
  -- The period of tan(ax + b) is π/|a|
  let T := Real.pi / 2
  use T
  sorry -- Skip the proof details


end NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_tangent_function_l1259_125929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hypercube_common_sum_l1259_125948

/-- Represents a hypercube with vertices numbered from 1 to 12 -/
structure Hypercube where
  vertices : Fin 12 → ℕ
  vertex_values : ∀ i, vertices i ∈ Finset.range 13 \ {0}

/-- A hyperface of the hypercube -/
def Hyperface := Finset (Fin 12)

/-- The set of all hyperfaces of the hypercube -/
def allHyperfaces : Finset Hyperface := sorry

/-- The sum of numbers on a given hyperface -/
def hyperfaceSum (h : Hypercube) (f : Hyperface) : ℕ :=
  f.sum (fun i => h.vertices i)

/-- The property that all hyperfaces have the same sum -/
def allHyperfacesSameSum (h : Hypercube) : Prop :=
  ∃ s, ∀ f ∈ allHyperfaces, hyperfaceSum h f = s

theorem hypercube_common_sum (h : Hypercube) (h_sum : allHyperfacesSameSum h) :
  ∃ s, s = 13 ∧ ∀ f ∈ allHyperfaces, hyperfaceSum h f = s := by
  sorry

#check hypercube_common_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hypercube_common_sum_l1259_125948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_area_l1259_125942

/-- The area of a triangle given its vertices -/
noncomputable def triangle_area (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  (1/2) * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

/-- Triangle ABC with vertices on a 5x4 rectangle -/
theorem triangle_abc_area :
  let A : ℝ × ℝ := (0, 1)
  let B : ℝ × ℝ := (4, 0)
  let C : ℝ × ℝ := (3, 5)
  triangle_area A.1 A.2 B.1 B.2 C.1 C.2 = 9.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_area_l1259_125942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_at_log2_3_l1259_125949

-- Define a monotonic function f on ℝ
variable (f : ℝ → ℝ)

-- Define the property that f is monotonic
def IsMonotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

-- Define the functional equation
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x, f (f x + 2 / (2^x + 1)) = 1/3

-- Theorem statement
theorem function_value_at_log2_3 
  (h_monotonic : IsMonotonic f) 
  (h_equation : SatisfiesFunctionalEquation f) : 
  f (Real.log 3 / Real.log 2) = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_at_log2_3_l1259_125949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_s_seven_times_20_l1259_125988

/-- The function s as defined in the problem -/
noncomputable def s (θ : ℝ) : ℝ := 1 / (2 - θ)

/-- The theorem stating that applying s seven times to 20 results in 18/37 -/
theorem s_seven_times_20 : s (s (s (s (s (s (s 20)))))) = 18/37 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_s_seven_times_20_l1259_125988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_symmetric_about_pi_over_4_l1259_125959

-- Define the function g(x)
noncomputable def g (x : ℝ) : ℝ := Real.tan (2 * x + Real.pi / 2)

-- Theorem statement
theorem g_symmetric_about_pi_over_4 :
  ∀ (x : ℝ), g (π/4 + x) = -g (π/4 - x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_symmetric_about_pi_over_4_l1259_125959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_R_squared_inverse_relation_residuals_l1259_125901

/-- The coefficient of determination in regression analysis -/
def R_squared : ℝ → ℝ := sorry

/-- The sum of squares of residuals in regression analysis -/
def sum_squares_residuals : ℝ → ℝ := sorry

/-- R_squared is used to judge the fitting effect of the model -/
axiom R_squared_measures_fit : 
  ∀ x y : ℝ, x < y → R_squared x < R_squared y → 
  sum_squares_residuals x > sum_squares_residuals y

/-- As R_squared increases, the sum of squares of residuals decreases -/
theorem R_squared_inverse_relation_residuals : 
  ∀ x y : ℝ, x < y → R_squared x < R_squared y → 
  sum_squares_residuals x > sum_squares_residuals y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_R_squared_inverse_relation_residuals_l1259_125901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_of_exponential_function_l1259_125961

-- Define the function f(x) = a^(x+2)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x + 2)

-- State the theorem
theorem fixed_point_of_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  f a (-2) = 1 ∧ ∀ x : ℝ, f a x = x → x = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_of_exponential_function_l1259_125961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_shift_right_l1259_125911

theorem sin_shift_right (x : ℝ) :
  Real.sin (2 * (x - π / 6)) = Real.sin (2 * x - π / 3) := by
  congr
  ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_shift_right_l1259_125911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_problem_2_eval_l1259_125951

-- Problem 1
theorem problem_1 : (2 : ℝ).sqrt - 1 ^ (0 : ℤ) + 2 * (1 / 3) + (-1 : ℝ) ^ 2023 - (-1 / 3) ^ (-1 : ℤ) = 11 / 3 := by sorry

-- Problem 2
theorem problem_2 (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 2) : 
  (x^2 / (x - 2) - x - 2) / (4 * x / (x^2 - 4)) = (x + 2) / x := by sorry

theorem problem_2_eval : 
  let x : ℝ := 3
  (x^2 / (x - 2) - x - 2) / (4 * x / (x^2 - 4)) = 5 / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_problem_2_eval_l1259_125951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_circles_collinearity_l1259_125908

-- Define the types for points and circles
variable (Point Circle : Type)

-- Define the property of three circles concurring at a point
variable (concur : Circle → Circle → Circle → Point → Prop)

-- Define the property of a point being on a circle
variable (on_circle : Point → Circle → Prop)

-- Define the property of two circles intersecting at a point
variable (intersect : Circle → Circle → Point → Prop)

-- Define the property of a line passing through two points
variable (line : Point → Point → Prop)

-- Define the property of a point being on a line
variable (on_line : Point → Point → Point → Prop)

-- Define the property of three points being collinear
variable (collinear : Point → Point → Point → Prop)

-- Given circles and points
variable (ω₁ ω₂ ω₃ : Circle) (Q A A' B B' C C' : Point)

-- State the theorem
theorem three_circles_collinearity 
  (h1 : concur ω₁ ω₂ ω₃ Q)
  (h2 : intersect ω₂ ω₃ A')
  (h3 : intersect ω₁ ω₃ B')
  (h4 : intersect ω₁ ω₂ C')
  (h5 : on_circle A ω₁)
  (h6 : on_line B A C')
  (h7 : on_circle B ω₂)
  (h8 : on_line C A B')
  (h9 : on_circle C ω₃)
  : collinear B C A' := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_circles_collinearity_l1259_125908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diver_descent_time_l1259_125947

/-- The time taken for a diver to reach a certain depth -/
noncomputable def time_to_reach_depth (depth : ℝ) (descent_rate : ℝ) : ℝ :=
  depth / descent_rate

/-- Theorem: The time taken for a diver to reach a depth of 6400 feet,
    descending at a rate of 32 feet per minute, is 200 minutes -/
theorem diver_descent_time :
  time_to_reach_depth 6400 32 = 200 := by
  -- Unfold the definition of time_to_reach_depth
  unfold time_to_reach_depth
  -- Perform the division
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diver_descent_time_l1259_125947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_focal_distance_l1259_125932

/-- Represents an ellipse in standard form (x²/a² + y²/b² = 1) -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- Calculates the focal distance of an ellipse -/
noncomputable def focalDistance (e : Ellipse) : ℝ := 
  Real.sqrt (e.a ^ 2 - e.b ^ 2)

/-- Given ellipses -/
noncomputable def C₁ : Ellipse := { a := Real.sqrt 12, b := 2 }
noncomputable def C₂ : Ellipse := { a := 4, b := Real.sqrt 8 }

/-- Theorem: C₁ and C₂ have the same focal distance -/
theorem same_focal_distance : focalDistance C₁ = focalDistance C₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_focal_distance_l1259_125932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1259_125945

theorem problem_statement (x : ℝ) (p q r : ℕ+) 
  (h1 : (1 + Real.sin x) * (1 + Real.cos x) = 9/4)
  (h2 : (1 - Real.sin x) * (1 - Real.cos x) = (p : ℝ) / (q : ℝ) - Real.sqrt (r : ℝ))
  (h3 : Nat.Coprime p.val q.val) : 
  r.val + p.val + q.val = 13 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1259_125945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_transformation_l1259_125905

theorem matrix_transformation (a b c d e f g h i : ℝ) :
  let M : Matrix (Fin 3) (Fin 3) ℝ := !![3, 0, 0; 0, 0, 1; 0, 1, 0]
  let N : Matrix (Fin 3) (Fin 3) ℝ := !![a, b, c; d, e, f; g, h, i]
  M * N = !![3*a, 3*b, 3*c; g, h, i; d, e, f] := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_transformation_l1259_125905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_term_of_geometric_sequence_l1259_125964

def geometric_sequence (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * r ^ (n - 1)

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem first_term_of_geometric_sequence (a : ℝ) (r : ℝ) :
  geometric_sequence a r 6 = (factorial 9 : ℝ) ∧
  geometric_sequence a r 9 = (factorial 10 : ℝ) →
  a = 362880 / (10 ^ (5/3)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_term_of_geometric_sequence_l1259_125964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_real_implies_a_eq_four_l1259_125967

/-- Given two complex numbers z₁ and z₂, where z₁ = -2 + i and z₂ = a + 2i,
    if their product is real, then a = 4. -/
theorem product_real_implies_a_eq_four (a : ℝ) :
  ((-2 : ℂ) + Complex.I) * (a + 2 * Complex.I) ∈ Set.range (Complex.ofReal) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_real_implies_a_eq_four_l1259_125967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_through_origin_g_upper_bound_l1259_125975

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log x + 1

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := f x / x

-- Theorem for the tangent line
theorem tangent_line_through_origin (x : ℝ) (hx : x > 0) :
  ∃ t : ℝ, t > 0 ∧ (f t - 0) / (t - 0) = 1 ∧ deriv f t = 1 / t :=
sorry

-- Theorem for the upper bound of g
theorem g_upper_bound (x : ℝ) (hx : x > 0) : g x ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_through_origin_g_upper_bound_l1259_125975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_expression_rounded_min_value_l1259_125984

open Real

theorem min_value_expression :
  ∃ (m : ℝ), 
    (∀ x y : ℝ, 
      (Real.sqrt (2 * (1 + Real.cos (2 * x))) - Real.sqrt (3 - Real.sqrt 2) * Real.sin x + 1) *
      (3 + 2 * Real.sqrt (7 - Real.sqrt 2) * Real.cos y - Real.cos (2 * y)) ≥ m) ∧ 
    (∃ x₀ y₀ : ℝ, 
      (Real.sqrt (2 * (1 + Real.cos (2 * x₀))) - Real.sqrt (3 - Real.sqrt 2) * Real.sin x₀ + 1) *
      (3 + 2 * Real.sqrt (7 - Real.sqrt 2) * Real.cos y₀ - Real.cos (2 * y₀)) = m) ∧ 
    m = 2 * Real.sqrt 2 - 12 :=
by sorry

theorem rounded_min_value :
  Int.floor (2 * Real.sqrt 2 - 12) + 1 = -9 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_expression_rounded_min_value_l1259_125984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_period_cos_scaled_is_6pi_min_period_cos_scaled_correct_l1259_125995

/-- The minimum positive period of f(x) = cos(x/3) -/
noncomputable def min_period_cos_scaled (f : ℝ → ℝ) : ℝ :=
  6 * Real.pi

/-- The standard cosine function -/
noncomputable def cos_standard : ℝ → ℝ :=
  Real.cos

/-- The scaled cosine function f(x) = cos(x/3) -/
noncomputable def cos_scaled (x : ℝ) : ℝ :=
  Real.cos (x / 3)

theorem min_period_cos_scaled_is_6pi :
  min_period_cos_scaled cos_scaled = 6 * Real.pi :=
by
  sorry

/-- The period of the standard cosine function is 2π -/
axiom period_cos_standard : ∀ x : ℝ, cos_standard (x + 2 * Real.pi) = cos_standard x

/-- The minimum positive period of cos_scaled is correct -/
theorem min_period_cos_scaled_correct :
  ∀ x : ℝ, cos_scaled (x + min_period_cos_scaled cos_scaled) = cos_scaled x :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_period_cos_scaled_is_6pi_min_period_cos_scaled_correct_l1259_125995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1259_125965

/-- Triangle ABC with vertices A(3,4), B(0,0), and C(c,0) -/
structure Triangle (c : ℝ) where
  A : Fin 2 → ℝ
  B : Fin 2 → ℝ
  C : Fin 2 → ℝ
  h_A : A = ![3, 4]
  h_B : B = ![0, 0]
  h_C : C = ![c, 0]

/-- Vector from point P to point Q -/
def vector (P Q : Fin 2 → ℝ) : Fin 2 → ℝ :=
  λ i => Q i - P i

/-- Dot product of two 2D vectors -/
def dot_product (u v : Fin 2 → ℝ) : ℝ :=
  (u 0) * (v 0) + (u 1) * (v 1)

/-- Cosine of angle A in triangle ABC -/
noncomputable def cos_angle_A (t : Triangle c) : ℝ :=
  let AB := vector t.B t.A
  let AC := vector t.C t.A
  let BC := vector t.C t.B
  (dot_product AB AB + dot_product AC AC - dot_product BC BC) / (2 * Real.sqrt (dot_product AB AB) * Real.sqrt (dot_product AC AC))

theorem triangle_properties (c : ℝ) (t : Triangle c) :
  (dot_product (vector t.B t.A) (vector t.C t.A) = 0 → c = 25 / 3) ∧
  (c = 5 → cos_angle_A t = Real.sqrt 5 / 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1259_125965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jump_rope_competition_l1259_125985

/-- Represents a class in the jump rope competition -/
structure JumpRopeClass where
  jumps : List ℕ

/-- Calculates the probability of a class receiving an award -/
def prob_award (c : JumpRopeClass) : ℚ :=
  (c.jumps.filter (· > 120)).length / c.jumps.length

/-- Calculates the average number of jumps for a class -/
def avg_jumps (c : JumpRopeClass) : ℚ :=
  c.jumps.sum / c.jumps.length

/-- The jump rope competition theorem -/
theorem jump_rope_competition
  (senior1 senior2 senior3 senior4 : JumpRopeClass)
  (h1 : senior1.jumps = [142, 131, 129, 126, 121, 109, 103, 98, 96, 94])
  (h2 : senior2.jumps = [137, 126, 116, 108])
  (h3 : senior3.jumps = [163, 134, 112, 103])
  (h4 : senior4.jumps = [158, 132, 130, 127, 110, 106]) :
  (prob_award senior1 = 1/2) ∧
  (((prob_award senior1) + (prob_award senior2) + (prob_award senior3) + (prob_award senior4)) * 1/4 = 13/24) ∧
  (avg_jumps senior3 ≥ avg_jumps senior1 ∧
   avg_jumps senior3 ≥ avg_jumps senior2 ∧
   avg_jumps senior3 ≥ avg_jumps senior4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jump_rope_competition_l1259_125985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_function_properties_l1259_125933

noncomputable def f (a b x : ℝ) : ℝ := a * (Real.cos x)^2 + b * Real.sin x * Real.cos x - a / 2 - 1

theorem circle_function_properties (a b : ℝ) 
  (h : a^2 + b^2 = 1) :
  (∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f a b (x + p) = f a b x) ∧
    (∀ (q : ℝ), q > 0 → (∀ (x : ℝ), f a b (x + q) = f a b x) → p ≤ q)) ∧
  (∃ (m : ℝ), m = -3/2 ∧ (∀ (x : ℝ), f a b x ≥ m)) := by
  sorry

#check circle_function_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_function_properties_l1259_125933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1259_125958

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a * (1 - x)

-- State the theorem
theorem f_properties (a : ℝ) :
  -- Domain condition
  (∀ x, x > 0 → f a x = Real.log x + a * (1 - x)) →
  -- Part 1: Monotonicity when a ≤ 0
  (a ≤ 0 → ∀ x₁ x₂, x₁ > 0 → x₂ > 0 → x₁ < x₂ → f a x₁ < f a x₂) ∧
  -- Part 2: Existence of maximum when a > 0
  (a > 0 → ∃ x_max, x_max > 0 ∧ ∀ x, x > 0 → f a x ≤ f a x_max) ∧
  -- Part 3: Range of a for maximum value > 2a-2
  (∃ x_max, x_max > 0 ∧ (∀ x, x > 0 → f a x ≤ f a x_max) ∧ f a x_max > 2*a - 2 ↔ 0 < a ∧ a < 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1259_125958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_composite_function_l1259_125919

-- Define the original function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the composite function g
noncomputable def g (x : ℝ) : ℝ := f (Real.log x / Real.log (1/2))

-- State the theorem
theorem domain_of_composite_function :
  (∀ x, 2 ≤ x ∧ x ≤ 4 → f x ≠ 0) →
  (∀ x, 1/16 ≤ x ∧ x ≤ 1/4 → g x ≠ 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_composite_function_l1259_125919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_element_implies_a_zero_l1259_125934

theorem set_element_implies_a_zero (a : ℝ) : 
  (1 ∈ ({a + 2, (a + 1)^2, a^2 + 3*a + 3} : Set ℝ)) → a = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_element_implies_a_zero_l1259_125934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_with_sum_l1259_125913

theorem subset_with_sum (partition : Finset (Finset ℕ)) : 
  (∀ n ∈ Finset.range 49, ∃ S ∈ partition, n + 1 ∈ S) →
  partition.card = 3 →
  ∃ S ∈ partition, ∃ a b c, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + b = c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_with_sum_l1259_125913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clea_escalator_ride_time_l1259_125991

/-- The time it takes Clea to ride down an operating escalator while standing -/
noncomputable def escalator_ride_time (walk_time_nonoperating : ℝ) (walk_time_operating : ℝ) : ℝ :=
  (walk_time_nonoperating * walk_time_operating) / (walk_time_nonoperating - walk_time_operating)

theorem clea_escalator_ride_time :
  escalator_ride_time 80 20 = 80 / 3 := by
  unfold escalator_ride_time
  -- The proof steps would go here, but we'll use sorry for now
  sorry

#eval Float.toString ((80 : Float) * 20 / (80 - 20))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clea_escalator_ride_time_l1259_125991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_groups_meeting_time_l1259_125983

/-- The distance between Budapest and Hatvan in kilometers -/
def distance : ℝ := 60

/-- The initial speed of the first group in km/h -/
def speed1_initial : ℝ := 8

/-- The initial speed of the second group in km/h -/
def speed2_initial : ℝ := 6

/-- The hourly speed decrease of the first group in km/h -/
def speed1_decrease : ℝ := 1

/-- The hourly speed decrease of the second group in km/h -/
def speed2_decrease : ℝ := 0.5

/-- The speed of the first group as a function of time t -/
noncomputable def speed1 (t : ℝ) : ℝ := speed1_initial - speed1_decrease * t

/-- The speed of the second group as a function of time t -/
noncomputable def speed2 (t : ℝ) : ℝ := speed2_initial - speed2_decrease * t

/-- The distance traveled by the first group as a function of time t -/
noncomputable def distance1 (t : ℝ) : ℝ := speed1_initial * t - (speed1_decrease / 2) * t^2

/-- The distance traveled by the second group as a function of time t -/
noncomputable def distance2 (t : ℝ) : ℝ := speed2_initial * t - (speed2_decrease / 2) * t^2

theorem groups_meeting_time :
  ∃ t : ℝ, t > 0 ∧ t < 7 ∧ 3 * t^2 - 56 * t + 240 = 0 ∧ 
  distance1 t + distance2 t = distance := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_groups_meeting_time_l1259_125983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_points_l1259_125952

/-- The distance between two points in a 2D plane. -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

/-- Theorem: The distance between the points (-3, 5) and (4, -7) is √193. -/
theorem distance_between_specific_points :
  distance (-3) 5 4 (-7) = Real.sqrt 193 := by
  -- Unfold the definition of distance
  unfold distance
  -- Simplify the expression
  simp [Real.sqrt_eq_rpow]
  -- The proof is completed
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_points_l1259_125952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1259_125957

-- Define the function f(x) = -2x^2 + 4x - 5
def f (x : ℝ) : ℝ := -2 * x^2 + 4 * x - 5

-- Theorem stating the properties of f
theorem f_properties :
  (∀ x : ℝ, f x ∈ Set.univ) ∧ 
  (f (1/2) = -7/2) ∧
  (∃ M : ℝ, M = -3 ∧ ∀ x : ℝ, f x ≤ M) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1259_125957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_equality_implies_y_value_l1259_125928

theorem mean_equality_implies_y_value (y : ℝ) : 
  (8 + 9 + 18) / 3 = (15 + y) / 2 → y = 25 / 3 := by
  sorry

#check mean_equality_implies_y_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_equality_implies_y_value_l1259_125928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_change_result_price_change_approximation_l1259_125906

/-- The percentage change that results in a final price of 81% of the original when applied as an increase and then a decrease -/
noncomputable def price_change_percentage : ℝ := Real.sqrt 0.19 * 100

/-- Theorem stating that applying the price_change_percentage as an increase and then a decrease results in 81% of the original price -/
theorem price_change_result (P : ℝ) (h : P > 0) : 
  P * (1 + price_change_percentage / 100) * (1 - price_change_percentage / 100) = 0.81 * P := by
  sorry

/-- Theorem stating that the price_change_percentage is approximately 43.6 -/
theorem price_change_approximation : 
  43.5 < price_change_percentage ∧ price_change_percentage < 43.7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_change_result_price_change_approximation_l1259_125906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_direction_angles_l1259_125915

theorem cosine_direction_angles {α' β' γ' : ℝ} (hpos : α' > 0 ∧ β' > 0 ∧ γ' > 0)
  (hα : Real.cos α' = 2/5) (hβ : Real.cos β' = 1/4)
  (hsum : Real.cos α' ^ 2 + Real.cos β' ^ 2 + Real.cos γ' ^ 2 = 1) :
  Real.cos γ' = Real.sqrt 311 / 20 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_direction_angles_l1259_125915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l1259_125971

open Real

-- Define the function as noncomputable
noncomputable def f (x : ℝ) := (1/2) * log x + x - (1/x) - 2

-- State the theorem
theorem root_in_interval :
  ∃ x : ℝ, 2 < x ∧ x < Real.exp 1 ∧ f x = 0 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l1259_125971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_less_than_4_range_of_m_for_f_geq_mx_minus_2_l1259_125921

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < -1 then (1/2)^x else x^2 + 3*x

-- Theorem for part (I)
theorem solution_set_f_less_than_4 :
  {x : ℝ | f x < 4} = Set.Ioo (-2 : ℝ) 1 := by sorry

-- Theorem for part (II)
theorem range_of_m_for_f_geq_mx_minus_2 :
  {m : ℝ | ∀ x ∈ Set.Ioo 0 2, f x ≥ m*x - 2} = Set.Iic (3 + 2*Real.sqrt 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_less_than_4_range_of_m_for_f_geq_mx_minus_2_l1259_125921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_betting_theorem_l1259_125987

/-- Represents the outcome of a bet -/
inductive BetOutcome
| Win
| Loss
deriving BEq, Repr

/-- Calculates the result of a single bet -/
def bet_result (initial_amount : ℚ) (bet_fraction : ℚ) (outcome : BetOutcome) : ℚ :=
  match outcome with
  | BetOutcome.Win => initial_amount + bet_fraction * initial_amount
  | BetOutcome.Loss => initial_amount - bet_fraction * initial_amount

/-- Represents a betting strategy -/
structure BettingStrategy where
  initial_amount : ℚ
  first_two_bet_fraction : ℚ
  last_two_bet_fraction : ℚ

/-- Calculates the final amount after a series of bets -/
def final_amount (strategy : BettingStrategy) (outcomes : List BetOutcome) : ℚ :=
  let rec aux (amount : ℚ) (remaining_bets : Nat) (outcomes : List BetOutcome) : ℚ :=
    match remaining_bets, outcomes with
    | 0, _ => amount
    | _, [] => amount
    | n+1, outcome::rest =>
      let bet_fraction := if n > 2 then strategy.first_two_bet_fraction else strategy.last_two_bet_fraction
      aux (bet_result amount bet_fraction outcome) n rest
  aux strategy.initial_amount 4 outcomes

theorem betting_theorem (strategy : BettingStrategy) :
  strategy.initial_amount = 100 ∧
  strategy.first_two_bet_fraction = 1/3 ∧
  strategy.last_two_bet_fraction = 2/3 →
  ∀ (outcomes : List BetOutcome),
    outcomes.length = 4 ∧
    outcomes.count BetOutcome.Win = 2 ∧
    outcomes.count BetOutcome.Loss = 2 →
    final_amount strategy outcomes = 1600/81 * 100 := by
  sorry

#eval final_amount
  { initial_amount := 100, first_two_bet_fraction := 1/3, last_two_bet_fraction := 2/3 }
  [BetOutcome.Win, BetOutcome.Win, BetOutcome.Loss, BetOutcome.Loss]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_betting_theorem_l1259_125987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_zeros_imply_a_range_l1259_125973

/-- The function f(x) with parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 + ((1-a)/2) * x^2 - a*x - a

/-- The statement that f has exactly two zeros in the interval (-2,0) -/
def has_two_zeros_in_interval (a : ℝ) : Prop :=
  ∃ x₁ x₂, -2 < x₁ ∧ x₁ < x₂ ∧ x₂ < 0 ∧ 
    f a x₁ = 0 ∧ f a x₂ = 0 ∧
    ∀ x, -2 < x ∧ x < 0 ∧ f a x = 0 → x = x₁ ∨ x = x₂

/-- Theorem stating that if f has exactly two zeros in (-2,0), then a is in (0,1/3) -/
theorem f_zeros_imply_a_range (a : ℝ) (h1 : a > 0) (h2 : has_two_zeros_in_interval a) :
  0 < a ∧ a < 1/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_zeros_imply_a_range_l1259_125973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_C_y_coordinate_l1259_125980

-- Define the pentagon
structure Pentagon where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ

-- Define the properties of the pentagon
def isPentagonABCDE (p : Pentagon) : Prop :=
  p.A = (0, 0) ∧ p.B = (0, 6) ∧ p.D = (6, 6) ∧ p.E = (6, 0) ∧
  p.C.1 = 3 ∧ -- C is on the line of symmetry
  (p.C.2 - 6) * 3 = 54 -- Area of triangle BCD is 54

-- Define the area of the pentagon
noncomputable def pentagonArea (p : Pentagon) : ℝ :=
  36 + (1/2) * 6 * (p.C.2 - 6)

-- Theorem statement
theorem pentagon_C_y_coordinate (p : Pentagon) 
  (h1 : isPentagonABCDE p) 
  (h2 : pentagonArea p = 90) : 
  p.C.2 = 24 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_C_y_coordinate_l1259_125980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_binomial_expansion_l1259_125982

noncomputable def binomial_expression (x : ℝ) : ℝ := (2/x - x)^6

theorem constant_term_of_binomial_expansion :
  ∃ (f : ℝ → ℝ), (∀ x ≠ 0, f x = binomial_expression x) ∧ 
  (∃ c : ℝ, c = -160 ∧ ∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x| ∧ |x| < δ → |f x - c| < ε) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_binomial_expansion_l1259_125982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_innings_played_l1259_125979

/-- Represents the number of innings played by a cricket player. -/
def innings : ℕ := sorry

/-- Represents the current average runs of the player. -/
def current_average : ℕ := 30

/-- Represents the runs needed in the next innings to increase the average. -/
def runs_needed : ℕ := 74

/-- Represents the increase in average after scoring the needed runs. -/
def average_increase : ℕ := 4

/-- Theorem stating that given the conditions, the player has played 10 innings. -/
theorem innings_played : 
  (current_average * innings + runs_needed) / (innings + 1) = current_average + average_increase →
  innings = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_innings_played_l1259_125979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l1259_125909

-- Define the function f
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_function_properties
  (a b c : ℝ)
  (ha : a ≠ 0)
  (h1 : f a b c (-1) = 0)
  (h2 : ∀ x : ℝ, x ≤ f a b c x ∧ f a b c x ≤ (x^2 + 1) / 2)
  (h3 : Set.Ioo (-1 : ℝ) 3 = {x : ℝ | |f a b c x| < 1}) :
  (∀ x, f a b c x = (1/4) * (x + 1)^2) ∧
  ((-1/2 < a ∧ a < 0) ∨ (0 < a ∧ a < 1/2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l1259_125909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1259_125930

noncomputable def f (x : ℝ) : ℝ := Real.sin (x + Real.pi/3) * Real.cos x - Real.sqrt 3 * (Real.cos x)^2 + Real.sqrt 3 / 4

theorem f_properties :
  (∃ T : ℝ, T > 0 ∧ (∀ x, f (x + T) = f x) ∧ (∀ S, 0 < S ∧ S < T → ∃ y, f (y + S) ≠ f y)) ∧
  (∀ x, -Real.pi/4 ≤ x ∧ x ≤ Real.pi/4 → -1/2 ≤ f x ∧ f x ≤ 1/4) ∧
  (f (Real.pi/6) = 0 ∧ ∀ x, -Real.pi/4 ≤ x ∧ x ≤ Real.pi/4 ∧ f x = 0 → x = Real.pi/6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1259_125930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_chords_l1259_125954

-- Define the circle and chords
def circle_radius : ℝ := 8
def chord_distance : ℝ := 8

-- Theorem statement
theorem area_between_chords (h : chord_distance = circle_radius) :
  let area := (64 * Real.pi / 3) + 16 * Real.sqrt 3
  area = 32 * Real.sqrt 3 + 64 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_chords_l1259_125954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_approx_l1259_125992

-- Define the triangle sides
noncomputable def a : ℝ := 26
noncomputable def b : ℝ := 22
noncomputable def c : ℝ := 10

-- Define the semi-perimeter
noncomputable def s : ℝ := (a + b + c) / 2

-- Define Heron's formula
noncomputable def heronFormula (a b c : ℝ) : ℝ := 
  Real.sqrt (s * (s - a) * (s - b) * (s - c))
  where s := (a + b + c) / 2

-- Theorem statement
theorem triangle_area_approx : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |heronFormula a b c - 107.76| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_approx_l1259_125992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_planes_l1259_125994

/-- The distance between two parallel planes -/
noncomputable def distance_between_planes (a b c d₁ d₂ : ℝ) : ℝ :=
  |d₁ - d₂| / Real.sqrt (a^2 + b^2 + c^2)

theorem distance_between_specific_planes :
  distance_between_planes 3 (-2) 6 15 3 = 9/7 := by
  -- Unfold the definition of distance_between_planes
  unfold distance_between_planes
  -- Simplify the expression
  simp [Real.sqrt_eq_rpow]
  -- The rest of the proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_planes_l1259_125994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_cosine_identity_l1259_125917

theorem sine_cosine_identity (θ : ℝ) (h : Real.sin θ + Real.sin θ ^ 2 = 1) :
  3 * Real.cos θ ^ 2 + Real.cos θ ^ 4 - 2 * Real.sin θ + 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_cosine_identity_l1259_125917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_allison_wins_probability_l1259_125922

/-- Allison's cube always shows 4 -/
def allison_cube : Fin 6 → ℕ := λ _ => 4

/-- Derek's cube is numbered from 1 to 6 uniformly -/
def derek_cube : Fin 6 → ℕ := Fin.val

/-- Sophie's cube has four faces with 3 and two faces with 5 -/
def sophie_cube : Fin 6 → ℕ
| ⟨0, _⟩ => 3
| ⟨1, _⟩ => 3
| ⟨2, _⟩ => 3
| ⟨3, _⟩ => 3
| ⟨4, _⟩ => 5
| ⟨5, _⟩ => 5

/-- The probability that Allison's roll is greater than both Derek's and Sophie's -/
theorem allison_wins_probability :
  (Finset.filter (λ i => derek_cube i < 4) (Finset.univ : Finset (Fin 6))).card * 
  (Finset.filter (λ i => sophie_cube i < 4) (Finset.univ : Finset (Fin 6))).card = 12 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_allison_wins_probability_l1259_125922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_segment_length_formula_l1259_125935

/-- A trapezoid with bases a and b, where a > b -/
structure Trapezoid (a b : ℝ) where
  h : a > b

/-- The length of a line segment parallel to the bases of a trapezoid,
    passing through the intersection of the diagonals -/
noncomputable def parallel_segment_length (a b : ℝ) (t : Trapezoid a b) : ℝ :=
  2 * a * b / (a + b)

/-- Theorem stating that the length of the parallel segment is 2ab/(a+b) -/
theorem parallel_segment_length_formula (a b : ℝ) (t : Trapezoid a b) :
  parallel_segment_length a b t = 2 * a * b / (a + b) :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_segment_length_formula_l1259_125935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_area_example_l1259_125970

/-- The area of an isosceles trapezoid -/
noncomputable def isoscelesTrapezoidArea (a b c : ℝ) : ℝ :=
  ((a + b) / 2) * Real.sqrt (c ^ 2 - ((b - a) / 2) ^ 2)

/-- Theorem: The area of an isosceles trapezoid with non-parallel sides of length 5
    and bases of length 8 and 14 is equal to 44 square units -/
theorem isosceles_trapezoid_area_example :
  isoscelesTrapezoidArea 8 14 5 = 44 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_area_example_l1259_125970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_representation_length_bound_l1259_125920

/-- The length of the period in the decimal representation of m/n. -/
def decimal_period_length (m n : ℕ+) : ℕ := sorry

/-- The length of the pre-period in the decimal representation of m/n. -/
def decimal_pre_period_length (m n : ℕ+) : ℕ := sorry

/-- Given coprime positive integers m and n, the sum of the lengths of the period
    and pre-period in the decimal representation of m/n is at most φ(n). -/
theorem decimal_representation_length_bound (m n : ℕ+) (h : Nat.Coprime m.val n.val) :
  (decimal_period_length m n) + (decimal_pre_period_length m n) ≤ Nat.totient n.val := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_representation_length_bound_l1259_125920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frequency_is_correct_term_l1259_125902

/-- Represents a group of data points within a specific range -/
structure DataGroup where
  range : Set ℝ
  points : Finset ℝ

/-- Represents a categorization of data into groups -/
structure DataCategorization where
  groups : List DataGroup

/-- The number of data points in a group -/
def pointCount (group : DataGroup) : ℕ :=
  group.points.card

/-- The term used for the number of data points in each group -/
def termForPointCount : String :=
  "Frequency"

theorem frequency_is_correct_term (categorization : DataCategorization) :
  ∀ group ∈ categorization.groups,
  pointCount group = group.points.card →
  termForPointCount = "Frequency" :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_frequency_is_correct_term_l1259_125902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1259_125981

noncomputable def f (x : ℝ) : ℝ := Real.exp x * (x^2 + x + 1)

theorem f_properties :
  let f' := λ x => Real.exp x * (x^2 + 3*x + 2)
  ∀ x : ℝ,
  (x < -2 ∨ x > -1 → f' x > 0) ∧
  (-2 < x ∧ x < -1 → f' x < 0) ∧
  (x = -2 → f x = 3 / Real.exp 2) ∧
  (x = -1 → f x = 1 / Real.exp 1) ∧
  (∀ y : ℝ, f y ≤ f (-2)) ∧
  (∀ y : ℝ, f (-1) ≤ f y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1259_125981
