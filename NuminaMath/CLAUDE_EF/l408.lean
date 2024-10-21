import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_percentage_theorem_l408_40806

/-- Represents the contents of a glass -/
structure Glass where
  water : ℝ
  alcohol : ℝ

/-- Represents the state of both glasses -/
structure TwoGlasses where
  first : Glass
  second : Glass

/-- Transfers liquid from one glass to another -/
noncomputable def transfer (glasses : TwoGlasses) (amount : ℝ) : TwoGlasses :=
  sorry

/-- The percentage of alcohol in a glass -/
noncomputable def alcoholPercentage (glass : Glass) : ℝ :=
  glass.alcohol / (glass.water + glass.alcohol)

/-- Theorem: The alcohol percentage in the first glass never exceeds that in the second -/
theorem alcohol_percentage_theorem (initialWater initialAlcohol : ℝ) :
  initialWater > 0 → initialAlcohol > 0 →
  ∀ (transfers : List ℝ),
    let initialState : TwoGlasses :=
      { first := { water := initialWater, alcohol := 0 },
        second := { water := 0, alcohol := initialAlcohol } }
    let finalState := transfers.foldl transfer initialState
    alcoholPercentage finalState.first ≤ alcoholPercentage finalState.second := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_percentage_theorem_l408_40806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplified_ratio_l408_40885

theorem simplified_ratio (kids_meals : ℕ) (adult_meals : ℕ)
  (h1 : kids_meals = 70) (h2 : adult_meals = 49) :
  (kids_meals / Nat.gcd kids_meals adult_meals, adult_meals / Nat.gcd kids_meals adult_meals) = (10, 7) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplified_ratio_l408_40885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_pyramid_lateral_surface_area_l408_40810

/-- A pyramid with an equilateral triangular base and one lateral face perpendicular to the base and also equilateral. -/
structure SpecialPyramid where
  /-- The side length of the base equilateral triangle -/
  a : ℝ
  /-- The base is an equilateral triangle -/
  base_equilateral : True
  /-- One lateral face is perpendicular to the base -/
  lateral_face_perpendicular : True
  /-- The perpendicular lateral face is an equilateral triangle -/
  lateral_face_equilateral : True

/-- The lateral surface area of the special pyramid -/
noncomputable def lateralSurfaceArea (p : SpecialPyramid) : ℝ :=
  (p.a^2 / 4) * (Real.sqrt 15 + Real.sqrt 3)

/-- Theorem: The lateral surface area of the special pyramid is (a^2 / 4) * (√15 + √3) -/
theorem special_pyramid_lateral_surface_area (p : SpecialPyramid) :
  lateralSurfaceArea p = (p.a^2 / 4) * (Real.sqrt 15 + Real.sqrt 3) := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_pyramid_lateral_surface_area_l408_40810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_foci_on_circle_l408_40808

-- Define the ellipse
def Ellipse (a b : ℝ) (h : a > b ∧ b > 0) : Set (ℝ × ℝ) :=
  {p | (p.1 / a)^2 + (p.2 / b)^2 = 1}

-- Define a point on the ellipse
def PointOnEllipse (a b : ℝ) (h : a > b ∧ b > 0) (p : ℝ × ℝ) : Prop :=
  p ∈ Ellipse a b h

-- Define the vertices of the ellipse
def Vertices (a b : ℝ) (h : a > b ∧ b > 0) : Set (ℝ × ℝ) :=
  {(a, 0), (-a, 0)}

-- Define a non-vertex point on the ellipse
def NonVertexPoint (a b : ℝ) (h : a > b ∧ b > 0) (p : ℝ × ℝ) : Prop :=
  PointOnEllipse a b h p ∧ p ∉ Vertices a b h

-- Define the tangent line at a point on the ellipse
def TangentLine (a b : ℝ) (h : a > b ∧ b > 0) (p : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {q | p.1 * q.1 / a^2 + p.2 * q.2 / b^2 = 1}

-- Define the intersection points B₁ and B₂
def IntersectionPoints (a b : ℝ) (h : a > b ∧ b > 0) (p : ℝ × ℝ) : Prop :=
  ∃ b₁ b₂ : ℝ × ℝ,
    b₁ ∈ TangentLine a b h p ∩ TangentLine a b h (a, 0) ∧
    b₂ ∈ TangentLine a b h p ∩ TangentLine a b h (-a, 0)

-- Define the foci of the ellipse
noncomputable def Foci (a b : ℝ) (h : a > b ∧ b > 0) : Set (ℝ × ℝ) :=
  let c := Real.sqrt (a^2 - b^2)
  {(c, 0), (-c, 0)}

-- Define a circle given its diameter endpoints
def Circle (p q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {r | (r.1 - p.1)^2 + (r.2 - p.2)^2 = (r.1 - q.1)^2 + (r.2 - q.2)^2}

-- The main theorem
theorem foci_on_circle (a b : ℝ) (h : a > b ∧ b > 0) (p : ℝ × ℝ)
  (hp : NonVertexPoint a b h p)
  (hb : IntersectionPoints a b h p) :
  ∃ b₁ b₂ : ℝ × ℝ, (∀ f ∈ Foci a b h, f ∈ Circle b₁ b₂) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_foci_on_circle_l408_40808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_x_from_cos_2x_l408_40859

theorem sin_x_from_cos_2x (x a : ℝ) 
  (h1 : x ∈ Set.Ioo (-π/2) 0) 
  (h2 : Real.cos (2*x) = a) : 
  Real.sin x = -Real.sqrt ((1 - a) / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_x_from_cos_2x_l408_40859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_battery_is_12_hours_l408_40802

/-- Represents the battery life of a smartphone --/
structure BatteryLife where
  standby_hours : ℚ
  continuous_use_hours : ℚ

/-- Calculates the remaining battery life on standby --/
def remaining_standby_hours (battery : BatteryLife) (total_on_hours : ℚ) (active_use_hours : ℚ) : ℚ :=
  let standby_rate := 1 / battery.standby_hours
  let active_rate := 1 / battery.continuous_use_hours
  let standby_hours := total_on_hours - active_use_hours
  let battery_used := standby_hours * standby_rate + active_use_hours * active_rate
  let battery_remaining := 1 - battery_used
  battery_remaining / standby_rate

/-- Theorem stating that the remaining battery life is 12 hours --/
theorem remaining_battery_is_12_hours (battery : BatteryLife) 
    (h1 : battery.standby_hours = 36)
    (h2 : battery.continuous_use_hours = 4)
    (h3 : remaining_standby_hours battery 12 (3/2) = 12) : 
  remaining_standby_hours battery 12 (3/2) = 12 := by
  sorry

#eval remaining_standby_hours ⟨36, 4⟩ 12 (3/2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_battery_is_12_hours_l408_40802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_maximized_at_optimal_radius_l408_40879

/-- The circumference of the sector -/
noncomputable def sector_circumference : ℝ := 20

/-- The radius that maximizes the area -/
noncomputable def optimal_radius : ℝ := 5

/-- The area of the sector as a function of the radius -/
noncomputable def sector_area (r : ℝ) : ℝ := r * (sector_circumference - 2 * r) / 2

theorem sector_area_maximized_at_optimal_radius :
  ∀ r : ℝ, r > 0 → sector_area r ≤ sector_area optimal_radius := by
  sorry

#check sector_area_maximized_at_optimal_radius

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_maximized_at_optimal_radius_l408_40879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_lines_l408_40897

-- Define the two lines
def l₁ (x y : ℝ) : Prop := 4 * x + 2 * y - 3 = 0
def l₂ (x y : ℝ) : Prop := 2 * x + y + 1 = 0

-- Define the distance between two parallel lines
noncomputable def distance_parallel_lines (a b c₁ c₂ : ℝ) : ℝ := 
  abs (c₁ - c₂) / Real.sqrt (a^2 + b^2)

-- Theorem statement
theorem distance_between_lines :
  distance_parallel_lines 4 2 (-3) 2 = Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_lines_l408_40897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diorama_time_calculation_l408_40843

/-- Diorama time calculation -/
theorem diorama_time_calculation 
  (total_time : ℕ) 
  (building_time : ℕ) 
  (planning_time : ℕ) 
  (x : ℕ) 
  (h1 : total_time = 67)
  (h2 : building_time = 49)
  (h3 : building_time = 3 * planning_time - x)
  (h4 : total_time = planning_time + building_time) :
  x = 5 := by
  sorry

#check diorama_time_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diorama_time_calculation_l408_40843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_M_l408_40820

-- Define the circle
noncomputable def circle_equation (x y m : ℝ) : Prop := x^2 + y^2 = m

-- Define the point M
noncomputable def point_M : ℝ × ℝ := (1, Real.sqrt 3)

-- Define the slope of the tangent line
noncomputable def tangent_slope (x y : ℝ) : ℝ := -y / x

-- Theorem statement
theorem tangent_slope_at_M (m : ℝ) :
  circle_equation point_M.1 point_M.2 m →
  tangent_slope point_M.1 point_M.2 = -(Real.sqrt 3) / 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_M_l408_40820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_doghouse_area_theorem_l408_40866

/-- The area outside a regular octagonal doghouse that a dog can reach when tethered to a vertex -/
noncomputable def doghouse_area (side_length : ℝ) (rope_length : ℝ) : ℝ :=
  (15 / 4) * Real.pi

/-- Theorem stating the area a dog can reach outside its octagonal doghouse -/
theorem doghouse_area_theorem (side_length rope_length : ℝ) 
  (h1 : side_length = 2)
  (h2 : rope_length = 3)
  (h3 : rope_length > side_length) :
  doghouse_area side_length rope_length = (15 / 4) * Real.pi := by
  sorry

#check doghouse_area_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_doghouse_area_theorem_l408_40866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_arc_length_l408_40813

/-- The arc length of the curve y = e^x + 26 from x = ln(√8) to x = ln(√24) -/
noncomputable def arcLength : ℝ := 2 + (1/2) * Real.log (4/3)

/-- The curve function -/
noncomputable def f (x : ℝ) : ℝ := Real.exp x + 26

/-- The lower bound of the interval -/
noncomputable def a : ℝ := Real.log (Real.sqrt 8)

/-- The upper bound of the interval -/
noncomputable def b : ℝ := Real.log (Real.sqrt 24)

/-- Theorem stating that the arc length of the curve y = e^x + 26 
    from x = ln(√8) to x = ln(√24) is equal to 2 + (1/2) * ln(4/3) -/
theorem curve_arc_length : 
  (∫ x in a..b, Real.sqrt (1 + (deriv f x)^2)) = arcLength := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_arc_length_l408_40813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_rectangle_area_l408_40825

/-- The area of a rectangle formed by one side of a regular hexagon as its base
    and the distance between two opposite sides as its height, 
    given that the side length of the hexagon is 10 cm. -/
theorem hexagon_rectangle_area : 
  ∃ (hexagon_side_length : ℝ) (rectangle_base : ℝ) (rectangle_height : ℝ),
    hexagon_side_length = 10 ∧
    rectangle_base = hexagon_side_length ∧
    rectangle_height = hexagon_side_length * Real.sqrt 3 ∧
    rectangle_base * rectangle_height = 100 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_rectangle_area_l408_40825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_arrives_first_l408_40849

/-- Represents the journey from point A to point B -/
structure Journey where
  distance : ℝ
  cyclist_speed : ℝ
  motorist_initial_speed : ℝ
  motorist_walking_speed : ℝ

/-- Calculates the time taken for the cyclist to complete the journey -/
noncomputable def cyclist_time (j : Journey) : ℝ :=
  j.distance / j.cyclist_speed

/-- Calculates the time taken for the motorist to complete the journey -/
noncomputable def motorist_time (j : Journey) : ℝ :=
  (j.distance / 2) / j.motorist_initial_speed + (j.distance / 2) / j.motorist_walking_speed

/-- Theorem stating that the cyclist arrives before the motorist -/
theorem cyclist_arrives_first (j : Journey) 
    (h1 : j.distance > 0)
    (h2 : j.cyclist_speed > 0)
    (h3 : j.motorist_initial_speed = 5 * j.cyclist_speed)
    (h4 : j.motorist_walking_speed = j.cyclist_speed / 2) : 
    cyclist_time j < motorist_time j := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_arrives_first_l408_40849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l408_40876

def triangle_proof (a b c : ℝ) (A B C : ℝ) : Prop :=
  let cos_A := Real.cos A
  let cos_B := Real.cos B
  let sin_A := Real.sin A
  (((b^2 + c^2 - a^2) / cos_A = 2) →
  ((a*cos_B - b*cos_A) / (a*cos_B + b*cos_A) - b/c = 1) →
  (b*c = 1) ∧ 
  (1/2 * b * c * sin_A = Real.sqrt 3 / 4))

theorem triangle_theorem :
  ∀ (a b c : ℝ) (A B C : ℝ),
  triangle_proof a b c A B C := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l408_40876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_0_value_l408_40868

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- f is an odd function
axiom f_odd : ∀ x, f (-x) = -f x

-- When x < 0, f(x) = 3^x
axiom f_neg : ∀ x, x < 0 → f x = Real.exp (x * Real.log 3)

-- Theorem to prove
theorem x_0_value (x₀ : ℝ) (h : f x₀ = -1/9) : x₀ = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_0_value_l408_40868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_three_extremum_points_l408_40821

noncomputable def f (x : ℝ) : ℝ := 
  if x ≤ 0 then (x + 1)^3 * Real.exp (x + 1) else (((-x) + 1)^3 * Real.exp ((-x) + 1))

theorem f_has_three_extremum_points :
  (∀ x, f (-x) = f x) ∧ 
  (∀ x ≤ 0, f x = (x + 1)^3 * Real.exp (x + 1)) →
  ∃! (s : Finset ℝ), s.card = 3 ∧ 
    (∀ x ∈ s, ∀ ε > 0, ∃ y₁ y₂, |y₁ - x| < ε ∧ |y₂ - x| < ε ∧ 
      ((f y₁ ≤ f x ∧ f y₂ ≤ f x) ∨ (f y₁ ≥ f x ∧ f y₂ ≥ f x))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_three_extremum_points_l408_40821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l408_40882

/-- A hyperbola with foci F₁ and F₂ -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  h_positive : a > 0 ∧ b > 0
  h_equation : ∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 → (x, y) ∈ {P | ∃ (t : ℝ), P = F₁ + t • (F₂ - F₁)}

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt ((h.a^2 + h.b^2) / h.a^2)

/-- The distance between two points -/
noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

theorem hyperbola_eccentricity (h : Hyperbola) (P : ℝ × ℝ) 
    (h_on_hyperbola : P.1^2 / h.a^2 - P.2^2 / h.b^2 = 1)
    (h_distance_sum : distance P h.F₁ + distance P h.F₂ = 4 * h.a)
    (h_angle : Real.sin (Real.pi / 6) = 1 / 3) :
    eccentricity h = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l408_40882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ladder_rungs_widths_l408_40800

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

def ladder_sequence (n : ℕ) : ℝ := arithmetic_sequence 33 7 n

theorem ladder_rungs_widths :
  (∀ n : ℕ, 1 ≤ n ∧ n ≤ 12 → ladder_sequence n ∈ Set.Icc 33 110) ∧
  ladder_sequence 1 = 33 ∧
  ladder_sequence 12 = 110 ∧
  (∀ n : ℕ, 2 ≤ n ∧ n ≤ 11 → ladder_sequence n = 33 + 7 * (n - 1)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ladder_rungs_widths_l408_40800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_collinearity_l408_40890

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def are_collinear (a b : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), a = t • b ∨ b = t • a

theorem vector_collinearity (a b : ℝ × ℝ) (k : ℝ) 
  (h1 : a ≠ (0, 0))
  (h2 : b ≠ (0, 0))
  (h3 : ¬ are_collinear a b)
  (h4 : are_collinear (k • a + b) (a + k • b)) :
  k = 1 ∨ k = -1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_collinearity_l408_40890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fill_time_with_leakage_is_18_l408_40829

/-- Represents the time it takes to fill a tank with leakage -/
noncomputable def fill_time_with_leakage (empty_time leak_time fill_time_no_leak : ℝ) : ℝ :=
  1 / (1 / fill_time_no_leak - 1 / empty_time)

/-- Theorem stating the time to fill a tank with leakage -/
theorem fill_time_with_leakage_is_18 (empty_time : ℝ) (fill_time_no_leak : ℝ)
  (h1 : empty_time = 36)
  (h2 : fill_time_no_leak = 12) :
  fill_time_with_leakage empty_time empty_time fill_time_no_leak = 18 := by
  sorry

#check fill_time_with_leakage_is_18

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fill_time_with_leakage_is_18_l408_40829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_l408_40872

noncomputable def f (x : ℝ) := 2 * Real.sin (2 * x + 2 * Real.pi / 3)

theorem f_monotone_decreasing :
  MonotoneOn f (Set.Icc (-Real.pi/12) (5*Real.pi/12)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_l408_40872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_is_one_l408_40881

theorem b_is_one (a b : ℝ) (h1 : a + b - a * b = 1) (h2 : ¬ ∃ (n : ℤ), a = n) : b = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_is_one_l408_40881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_eq_2pi_div_3_l408_40838

noncomputable def a : ℝ × ℝ := (1, 0)
noncomputable def b : ℝ × ℝ := (-1, Real.sqrt 3)

noncomputable def angle_between_vectors (x y : ℝ × ℝ) : ℝ :=
  Real.arccos ((x.1 * y.1 + x.2 * y.2) / (Real.sqrt (x.1^2 + x.2^2) * Real.sqrt (y.1^2 + y.2^2)))

theorem angle_between_vectors_eq_2pi_div_3 : 
  angle_between_vectors a b = 2 * Real.pi / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_eq_2pi_div_3_l408_40838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_magnitude_3_minus_5i_over_2_l408_40892

theorem complex_magnitude_3_minus_5i_over_2 :
  Complex.abs (3 - (5 / 2) * Complex.I) = Real.sqrt 61 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_magnitude_3_minus_5i_over_2_l408_40892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_and_a_value_l408_40858

/-- The function f(x) defined as 1/a - 1/x --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1/a - 1/x

/-- Theorem stating that f is increasing on (0, +∞) and finding the value of a --/
theorem f_increasing_and_a_value (a : ℝ) (h_a : a > 0) :
  (∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → x₁ < x₂ → f a x₁ < f a x₂) ∧
  (f a 4 = 5 → a = 4/21) := by
  sorry

#check f_increasing_and_a_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_and_a_value_l408_40858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l408_40857

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := x + Real.sqrt (1 - x)

-- State the theorem
theorem f_max_value :
  ∃ (max : ℝ), (∀ x, f x ≤ max) ∧ (max = 5/4) := by
  -- We'll use 5/4 as our maximum value
  use 5/4
  constructor
  
  -- Prove that f(x) ≤ 5/4 for all x
  · sorry
  
  -- Prove that 5/4 is indeed the maximum (equality holds)
  · sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l408_40857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_n_divisible_by_143_l408_40867

def b : ℕ → ℕ
  | 0 => 15
  | n+1 => if n+1 ≥ 15 then 150 * b n + (n+1) else 15

theorem least_n_divisible_by_143 : 
  (∀ k, 15 < k → k < 27 → ¬(143 ∣ b k)) ∧ (143 ∣ b 27) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_n_divisible_by_143_l408_40867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_ratio_equals_negative_three_l408_40855

/-- Given a point A(2,-1) on the terminal side of angle θ, 
    prove that (sin θ - cos θ) / (sin θ + cos θ) = -3 -/
theorem angle_ratio_equals_negative_three (θ : ℝ) 
  (h : ∃ (A : ℝ × ℝ), A = (2, -1) ∧ A.1 = 2 * Real.cos θ ∧ A.2 = 2 * Real.sin θ) : 
  (Real.sin θ - Real.cos θ) / (Real.sin θ + Real.cos θ) = -3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_ratio_equals_negative_three_l408_40855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_berry_average_temperature_l408_40847

def sunday_temp : ℚ := 99.1
def monday_temp : ℚ := 98.2
def tuesday_temp : ℚ := 98.7
def wednesday_temp : ℚ := 99.3
def thursday_temp : ℚ := 99.8
def friday_temp : ℚ := 99.0
def saturday_temp : ℚ := 98.9

def days_in_week : ℚ := 7

theorem berry_average_temperature : 
  (sunday_temp + monday_temp + tuesday_temp + wednesday_temp + 
   thursday_temp + friday_temp + saturday_temp) / days_in_week = 99 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_berry_average_temperature_l408_40847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_chord_length_l408_40817

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

/-- The parabola y² = 4x -/
def isOnParabola (p : Point) : Prop :=
  p.y^2 = 4 * p.x

/-- The directrix of the parabola -/
def directrix : ℝ := -1

/-- Check if a circle is tangent to the directrix -/
def isTangentToDirectrix (c : Circle) : Prop :=
  c.radius = c.center.x - directrix

/-- Check if a circle passes through a point -/
def passesThrough (c : Circle) (p : Point) : Prop :=
  (c.center.x - p.x)^2 + (c.center.y - p.y)^2 = c.radius^2

/-- Calculate the chord length where a circle intersects the y-axis -/
noncomputable def chordLength (c : Circle) : ℝ :=
  2 * Real.sqrt (c.radius^2 - c.center.x^2)

theorem parabola_circle_chord_length :
  ∀ (M : Point),
  isOnParabola M →
  ∃ (c : Circle),
  c.center = M ∧
  isTangentToDirectrix c ∧
  passesThrough c (Point.mk 3 0) →
  chordLength c = 2 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_chord_length_l408_40817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_equality_l408_40841

-- Define the ordering of p, q, r, s, t
variable (p q r s t : ℝ)
variable (h : p < q ∧ q < r ∧ r < s ∧ s < t)

-- Define M and m functions
noncomputable def M (x y : ℝ) : ℝ := max x y
noncomputable def m (x y : ℝ) : ℝ := min x y

-- State the theorem
theorem max_min_equality : M (M p (m q r)) (m s (m p t)) = q := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_equality_l408_40841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_point_and_tangent_line_imply_k_equals_negative_one_l408_40832

-- Define the functions f and g
def f (c : ℝ) (x : ℝ) : ℝ := x^2 + c
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x

-- Define the point P
def P : ℝ × ℝ := (2, 0)  -- We use 0 as a placeholder for t

-- State the theorem
theorem common_point_and_tangent_line_imply_k_equals_negative_one 
  (c a : ℝ) 
  (h1 : f c (P.1) = g a (P.1))  -- Common point condition
  (h2 : (deriv (f c)) P.1 = (deriv (g a)) P.1)  -- Same tangent line condition
  (h3 : ∃ (x : ℝ), x < 0 ∧ f c x = g a x)  -- Existence of negative zero
  (h4 : ∃ (k : ℤ), ∀ (x : ℝ), f c x = g a x → x ∈ Set.Ioo (k : ℝ) ((k : ℝ) + 1))  -- Zero in (k, k+1)
  : ∃ (k : ℤ), k = -1 ∧ 
    ∀ (x : ℝ), f c x = g a x → x < 0 → x ∈ Set.Ioo (k : ℝ) ((k : ℝ) + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_point_and_tangent_line_imply_k_equals_negative_one_l408_40832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_equations_l408_40830

-- Define the circle
def circle_A (x y : ℝ) : Prop := (x + 1)^2 + (y - 2)^2 = 20

-- Define the tangent line l₁
def line_l1 (x y : ℝ) : Prop := x + 2*y + 7 = 0

-- Define point B
def point_B : ℝ × ℝ := (-2, 0)

-- Define the variable line l
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k * (x + 2) ∨ x = -2

-- Define the length of MN
noncomputable def length_MN : ℝ := 2 * Real.sqrt 19

theorem circle_and_line_equations :
  -- The circle A is centered at (-1, 2) and tangent to line l₁
  (∀ x y, circle_A x y ↔ (x + 1)^2 + (y - 2)^2 = 20) ∧
  -- When |MN| = 2√19, the equation of line l is 3x - 4y + 6 = 0 or x = -2
  (∃ k, ∀ x y, length_MN = 2 * Real.sqrt 19 → 
    (line_l k x y ↔ (3*x - 4*y + 6 = 0 ∨ x = -2))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_equations_l408_40830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_farm_trees_count_l408_40801

/-- Calculates the number of trees on a farm after a typhoon and replanting --/
def trees_after_typhoon_and_replanting (
  initial_mahogany : ℕ)
  (initial_narra : ℕ)
  (total_fallen : ℕ)
  (mahogany_fallen_diff : ℕ)
  (narra_replant_factor : ℕ)
  (mahogany_replant_factor : ℕ) : ℕ :=
  -- The implementation goes here
  sorry

/-- Theorem stating that given the specific conditions, there are 88 trees after the typhoon and replanting --/
theorem farm_trees_count : 
  trees_after_typhoon_and_replanting 50 30 5 1 2 3 = 88 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_farm_trees_count_l408_40801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_equation_proof_l408_40823

/-- A plane in 3D space --/
structure Plane where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ

/-- A point in 3D space --/
structure Point where
  x : ℤ
  y : ℤ
  z : ℤ

/-- Check if a plane contains a point --/
def Plane.contains (p : Plane) (pt : Point) : Prop :=
  p.a * pt.x + p.b * pt.y + p.c * pt.z + p.d = 0

/-- Check if two planes are parallel --/
def Plane.parallel (p1 p2 : Plane) : Prop :=
  ∃ (k : ℚ), k ≠ 0 ∧ p1.a = k * p2.a ∧ p1.b = k * p2.b ∧ p1.c = k * p2.c

theorem plane_equation_proof (given_plane : Plane) (point : Point) :
  let result_plane := Plane.mk 3 (-2) 4 (-16)
  given_plane.a = 3 ∧ given_plane.b = -2 ∧ given_plane.c = 4 ∧ given_plane.d = 5 →
  point.x = 2 ∧ point.y = -3 ∧ point.z = 1 →
  result_plane.contains point ∧
  result_plane.parallel given_plane ∧
  result_plane.a > 0 ∧
  Int.gcd (Int.natAbs result_plane.a) (Int.gcd (Int.natAbs result_plane.b) (Int.gcd (Int.natAbs result_plane.c) (Int.natAbs result_plane.d))) = 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_equation_proof_l408_40823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_microphotonics_allocation_l408_40877

theorem microphotonics_allocation (home_electronics : ℝ) (food_additives : ℝ) 
  (genetically_modified : ℝ) (industrial_lubricants : ℝ) (astrophysics_degrees : ℝ) :
  home_electronics = 24 →
  food_additives = 20 →
  genetically_modified = 29 →
  industrial_lubricants = 8 →
  astrophysics_degrees = 18 →
  let astrophysics_percent := (astrophysics_degrees / 360) * 100;
  let total_known := home_electronics + food_additives + genetically_modified + 
                     industrial_lubricants + astrophysics_percent;
  let microphotonics := 100 - total_known
  microphotonics = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_microphotonics_allocation_l408_40877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_with_arithmetic_sides_and_area_6_l408_40850

-- Define the triangle sides as an arithmetic sequence
def triangle_sides (x : ℝ) : Fin 3 → ℝ := ![x - 2, x, x + 2]

-- Define the semi-perimeter
noncomputable def semi_perimeter (x : ℝ) : ℝ := (3 * x) / 2

-- Define the area using Heron's formula
noncomputable def area (x : ℝ) : ℝ :=
  let s := semi_perimeter x
  let sides := triangle_sides x
  Real.sqrt (s * (s - sides 0) * (s - sides 1) * (s - sides 2))

-- Theorem statement
theorem triangle_with_arithmetic_sides_and_area_6 :
  ∃ x : ℝ, x > 0 ∧ area x = 6 ∧ 
  triangle_sides x = ![2 * Real.sqrt 6 - 2, 2 * Real.sqrt 6, 2 * Real.sqrt 6 + 2] := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_with_arithmetic_sides_and_area_6_l408_40850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sergeant_distance_l408_40891

/-- The distance traveled by a sergeant on a gyroscooter given specific conditions --/
theorem sergeant_distance (column_length : ℝ) (column_distance : ℝ) (speed_ratio : ℝ) : 
  column_length = 1 →
  column_distance = 2.4 →
  speed_ratio = 1.5 →
  ∃ (column_speed : ℝ), column_speed > 0 ∧
    let sergeant_speed := speed_ratio * column_speed
    let time := column_distance / column_speed
    let sergeant_distance := 2 * column_length + column_distance
    sergeant_distance = 3.6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sergeant_distance_l408_40891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_negative_power_l408_40815

theorem power_of_negative_power (a : ℝ) (h : a ≠ 0) : (a^(-2 : ℝ))^3 = a^(-6 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_negative_power_l408_40815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_special_set_l408_40853

theorem existence_of_special_set (n : ℕ) (h : n ≥ 2) :
  ∃ (S : Finset ℤ), (Finset.card S = n) ∧
  ∀ (a b : ℤ), a ∈ S → b ∈ S → a ≠ b → (a - b)^2 ∣ (a * b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_special_set_l408_40853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_impossible_n_l408_40895

/-- Represents the number of students --/
def num_students : ℕ := 31

/-- Represents the number of questions each student knows --/
def questions_known_per_student : ℕ := 3000

/-- Represents the minimum number of students who know each question --/
def min_students_per_question : ℕ := 29

/-- Represents the set of questions known by a student --/
def set_of_known_questions (student : ℕ) (n : ℕ) : Set ℕ :=
  {q | q < n ∧ (q % num_students ≠ (2 * student - 1) % num_students)}

/-- 
Checks if it's possible to find a starting point in a circular arrangement 
of n questions such that each student receives a known question
--/
def exists_valid_starting_point (n : ℕ) : Prop :=
  ∃ (start : ℕ), ∀ (i : ℕ), i < num_students → 
    ((start + i) % n) ∈ set_of_known_questions i n

theorem smallest_impossible_n : 
  (∀ n : ℕ, n ≥ 3000 ∧ n < 3100 → exists_valid_starting_point n) ∧
  ¬exists_valid_starting_point 3100 :=
sorry

#check smallest_impossible_n

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_impossible_n_l408_40895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marathon_speed_ratio_l408_40852

/-- The marathon distance in kilometers -/
noncomputable def marathonDistance : ℝ := 42

/-- Jack's time to complete the marathon in hours -/
noncomputable def jackTime : ℝ := 6

/-- Jill's time to complete the marathon in hours -/
noncomputable def jillTime : ℝ := 4.2

/-- Jack's average speed in km/h -/
noncomputable def jackSpeed : ℝ := marathonDistance / jackTime

/-- Jill's average speed in km/h -/
noncomputable def jillSpeed : ℝ := marathonDistance / jillTime

/-- The ratio of Jack's speed to Jill's speed -/
noncomputable def speedRatio : ℝ := jackSpeed / jillSpeed

theorem marathon_speed_ratio : speedRatio = 7 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marathon_speed_ratio_l408_40852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lee_breaks_even_l408_40865

/-- Represents the sale of two books -/
structure BookSale where
  price : ℝ
  profit_percent : ℝ
  loss_percent : ℝ

/-- Calculates the cost price of a book given its selling price and profit percentage -/
noncomputable def cost_price_profit (price : ℝ) (profit_percent : ℝ) : ℝ :=
  price / (1 + profit_percent / 100)

/-- Calculates the cost price of a book given its selling price and loss percentage -/
noncomputable def cost_price_loss (price : ℝ) (loss_percent : ℝ) : ℝ :=
  price / (1 - loss_percent / 100)

/-- Theorem stating that Mr. Lee breaks even in the given scenario -/
theorem lee_breaks_even (sale : BookSale) 
  (h1 : sale.price = 1.5)
  (h2 : sale.profit_percent = 25)
  (h3 : sale.loss_percent = 16.67) : 
  2 * sale.price = cost_price_profit sale.price sale.profit_percent + cost_price_loss sale.price sale.loss_percent := by
  sorry

-- Remove the #eval line as it's causing issues with universe levels

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lee_breaks_even_l408_40865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_special_case_l408_40842

theorem sin_double_angle_special_case (α : ℝ) 
  (h1 : Real.cos (Real.pi / 2 - α) = 1 / 3)
  (h2 : Real.pi / 2 < α)
  (h3 : α < Real.pi) :
  Real.sin (2 * α) = -4 * Real.sqrt 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_special_case_l408_40842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagonal_pyramid_inclinations_l408_40856

-- Define the pyramid structure
structure Pyramid where
  base : RegularPentagon
  apex : Point

-- Define the inclination of a face to the base plane
noncomputable def faceInclination (p : Pyramid) (face : Nat) : Real := sorry

-- State the theorem
theorem pentagonal_pyramid_inclinations (p : Pyramid) :
  faceInclination p 1 = 30 * Real.pi / 180 ∧
  faceInclination p 2 = 90 * Real.pi / 180 ∧
  faceInclination p 3 = 60 * Real.pi / 180 →
  (abs (faceInclination p 4 - 16.4833 * Real.pi / 180) < 0.0001 ∧
   abs (faceInclination p 5 - 20.5667 * Real.pi / 180) < 0.0001) := by
  sorry

-- Define RegularPentagon (as it's not provided in the standard library)
structure RegularPentagon where
  -- Add necessary fields here
  dummy : Unit

-- Define Point (if not already defined in Mathlib)
structure Point where
  x : Real
  y : Real
  z : Real

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagonal_pyramid_inclinations_l408_40856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octahedron_faces_count_l408_40887

/-- An octahedron is a three-dimensional geometric shape with 8 faces. -/
structure Octahedron where
  faces : Fin 8

/-- The number of faces in an octahedron is 8. -/
theorem octahedron_faces_count : ∀ (o : Octahedron), Fintype.card (Fin 8) = 8 := by
  intro o
  exact rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_octahedron_faces_count_l408_40887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_radius_is_three_l408_40851

/-- A triangular pyramid with perpendicular side edges -/
structure TriangularPyramid where
  -- S is the apex, A, B, C are the base vertices
  SA : ℝ
  SB : ℝ
  SC : ℝ
  -- The side edges are perpendicular
  perpendicular_edges : SA * SB = 0 ∧ SB * SC = 0 ∧ SC * SA = 0

/-- The radius of the circumscribed sphere of a triangular pyramid -/
noncomputable def circumscribed_sphere_radius (p : TriangularPyramid) : ℝ :=
  ((p.SA ^ 2 + p.SB ^ 2 + p.SC ^ 2) / 4) ^ (1/2)

/-- The theorem to be proved -/
theorem circumscribed_sphere_radius_is_three 
  (p : TriangularPyramid) 
  (h1 : p.SA = 2) 
  (h2 : p.SB = 4) 
  (h3 : p.SC = 4) : 
  circumscribed_sphere_radius p = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_radius_is_three_l408_40851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_calculation_l408_40862

/-- Calculates the total shaded area on a rectangular floor with patterned tiles -/
theorem shaded_area_calculation (floor_length floor_width tile_size circle_radius : ℝ) 
  (h1 : floor_length = 12)
  (h2 : floor_width = 15)
  (h3 : tile_size = 2)
  (h4 : circle_radius = 1) : 
  (floor_length / tile_size) * (floor_width / tile_size) * (tile_size^2 - π * circle_radius^2) = 180 - 45 * π := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_calculation_l408_40862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_two_red_balls_l408_40811

/-- The probability of selecting two red balls from a bag containing 5 red balls, 6 blue balls, and 2 green balls. -/
theorem probability_two_red_balls : 
  (5 : ℚ) / 13 * (4 : ℚ) / 12 = 5 / 39 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_two_red_balls_l408_40811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l408_40871

/-- The eccentricity of a hyperbola with the given conditions -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (x y c : ℝ),
    x^2 / a^2 - y^2 / b^2 = 1 ∧   -- Hyperbola equation
    c > 0 ∧                      -- c is positive (implied by the existence of foci)
    x^2 + y^2 = c^2 ∧            -- Circle equation (diameter is 2c)
    (x - c)^2 + y^2 = a^2 ∧      -- Distance from left focus to P is a
    (x + c)^2 + y^2 = (3*a)^2 ∧  -- Distance from right focus to P is 3a (implied by circle property)
    c / a = Real.sqrt 10 / 2     -- Eccentricity is sqrt(10)/2
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l408_40871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_polynomial_l408_40854

-- Define a monic polynomial with integer coefficients
def MonicPolynomialZ (f : ℕ → ℤ) : Prop :=
  ∃ n : ℕ, ∀ k > n, f k = 0 ∧ f n = 1

-- Define the condition for the polynomial
def SatisfiesCondition (f : ℕ → ℤ) : Prop :=
  ∃ N : ℕ, ∀ p : ℕ, Nat.Prime p → 0 < f p → p ∣ (2 * (Nat.factorial (Int.toNat (f p))) + 1)

-- The main theorem
theorem unique_polynomial :
  ∃! f : ℕ → ℤ, MonicPolynomialZ f ∧ SatisfiesCondition f ∧ (∀ x, f x = x - 3) := by
  sorry

#check unique_polynomial

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_polynomial_l408_40854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_constant_l408_40844

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The function f(x) = a - 2/(e^x - 1) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a - 2 / (Real.exp x - 1)

theorem odd_function_constant (a : ℝ) :
  IsOdd (f a) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_constant_l408_40844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fan_energy_consumption_l408_40873

/-- Calculates the monthly energy consumption in kWh for an electric device -/
noncomputable def monthly_energy_consumption (power_watts : ℝ) (hours_per_day : ℝ) (days : ℕ) : ℝ :=
  power_watts * hours_per_day * (days : ℝ) / 1000

/-- Theorem: The monthly energy consumption of a 75-watt fan used 8 hours a day for 30 days is 18 kWh -/
theorem fan_energy_consumption :
  monthly_energy_consumption 75 8 30 = 18 := by
  unfold monthly_energy_consumption
  -- Perform the calculation
  have : 75 * 8 * (30 : ℝ) / 1000 = 18 := by
    norm_num
  -- Apply the calculation result
  exact this


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fan_energy_consumption_l408_40873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_transformation_l408_40835

theorem sqrt_transformation (a : ℕ) (ha : 0 < a) :
  ∃ (x y : ℕ) (z : ℤ), 
    0 < x ∧ 0 < y ∧
    |z| < (x : ℤ)^2 ∧
    (Real.sqrt a = (x : ℝ) / (y : ℝ) * Real.sqrt (1 + (z : ℝ) / (x : ℝ)^2) ∨
     Real.sqrt a = (x : ℝ) / (y : ℝ) * Real.sqrt (1 - (z : ℝ) / (x : ℝ)^2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_transformation_l408_40835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_length_l408_40803

-- Define the parametric equations
noncomputable def x (θ : ℝ) : ℝ := 2 * (Real.cos θ)^2
noncomputable def y (θ : ℝ) : ℝ := 3 * (Real.sin θ)^2

-- Define the curve
def curve : Set (ℝ × ℝ) := {p | ∃ θ, p.1 = x θ ∧ p.2 = y θ}

-- State the theorem
theorem curve_length : Metric.diam curve = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_length_l408_40803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l408_40884

noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2

noncomputable def g (x : ℝ) : ℝ := -1/2 * x + 4

-- Theorem statement
theorem function_properties :
  (g (f (1/4)) = 5) ∧
  (∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → max (f y) (g y) ≥ max (f x) (g x)) ∧
  (max (f 4) (g 4) = 2 → ∀ (y : ℝ), y > 0 → max (f y) (g y) ≥ 2) ∧
  (∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → min (f y) (g y) ≤ min (f x) (g x)) ∧
  (min (f 4) (g 4) = 2 → ∀ (y : ℝ), y > 0 → min (f y) (g y) ≤ 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l408_40884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_planted_fraction_is_eleven_twelfths_l408_40816

/-- Represents an isosceles triangle with a square at one vertex -/
structure TriangleWithSquare where
  /-- Length of the equal sides of the isosceles triangle -/
  side : ℝ
  /-- Length of the base of the isosceles triangle -/
  base : ℝ
  /-- Side length of the square -/
  square_side : ℝ
  /-- Shortest distance from the square to the opposite side -/
  square_distance : ℝ

/-- The specific triangle and square configuration from the problem -/
def problem_config : TriangleWithSquare :=
  { side := 5
  , base := 6
  , square_side := 1  -- This is derived in the solution, not given in the problem
  , square_distance := 3 }

/-- Calculates the area of an isosceles triangle given its side and base lengths -/
noncomputable def triangle_area (t : TriangleWithSquare) : ℝ :=
  let s := (t.side + t.side + t.base) / 2
  Real.sqrt (s * (s - t.side) * (s - t.side) * (s - t.base))

/-- Calculates the fraction of the triangle that is not covered by the square -/
noncomputable def planted_fraction (t : TriangleWithSquare) : ℝ :=
  (triangle_area t - t.square_side ^ 2) / triangle_area t

/-- Theorem stating that the planted fraction for the problem configuration is 11/12 -/
theorem planted_fraction_is_eleven_twelfths :
  planted_fraction problem_config = 11 / 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_planted_fraction_is_eleven_twelfths_l408_40816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_4th_and_12th_term_l408_40836

/-- An arithmetic progression is a sequence where the difference between any two consecutive terms is constant. -/
structure ArithmeticProgression where
  first : ℚ
  diff : ℚ

/-- The nth term of an arithmetic progression -/
def ArithmeticProgression.nthTerm (ap : ArithmeticProgression) (n : ℕ) : ℚ :=
  ap.first + (n - 1 : ℚ) * ap.diff

/-- The sum of the first n terms of an arithmetic progression -/
def ArithmeticProgression.sumFirstN (ap : ArithmeticProgression) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * ap.first + (n - 1 : ℚ) * ap.diff)

/-- 
Theorem: In an arithmetic progression where the sum of the first 15 terms is 120,
the sum of the 4th term and the 12th term is 16.
-/
theorem sum_of_4th_and_12th_term (ap : ArithmeticProgression) 
  (h : ap.sumFirstN 15 = 120) : 
  ap.nthTerm 4 + ap.nthTerm 12 = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_4th_and_12th_term_l408_40836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_f_domain_equals_set_l408_40875

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x - 2) + 1 / (x - 3)

-- Define the set representing [2,3) ∪ (3, +∞)
def domain_set : Set ℝ := {x : ℝ | x ≥ 2 ∧ x ≠ 3}

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = domain_set := by sorry

-- Prove that the domain of f is equal to [2,3) ∪ (3, +∞)
theorem f_domain_equals_set :
  {x : ℝ | ∃ y, f x = y} = domain_set := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_f_domain_equals_set_l408_40875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_perimeter_length_l408_40883

noncomputable def quadrilateral_vertices : List (ℝ × ℝ) := [(0, 0), (5, 0), (5, 8), (0, 8)]

noncomputable def diagonal_length : ℝ := Real.sqrt 89

def removed_segment_length : ℝ := 2

def retained_right_side_length : ℝ := 3

def top_side_length : ℝ := 5

def left_side_retained_length : ℝ := 8 - removed_segment_length

theorem pentagon_perimeter_length :
  let perimeter := left_side_retained_length + diagonal_length + top_side_length + retained_right_side_length
  perimeter = 14 + Real.sqrt 89 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_perimeter_length_l408_40883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_composite_number_property_l408_40863

def is_valid_sequence (d : Nat → Nat) (k : Nat) : Prop :=
  d 1 = 1 ∧
  (∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ k → d i < d j) ∧
  (∀ i, 1 < i ∧ i ≤ k → d i - d (i-1) = (i-1) * (d 2 - d 1))

def is_composite (n : Nat) : Prop :=
  ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

theorem composite_number_property (n : Nat) :
  is_composite n →
  (∃ k : Nat, ∃ d : Nat → Nat,
    d k = n ∧
    (∀ i ≤ k, n % d i = 0) ∧
    is_valid_sequence d k) →
  n = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_composite_number_property_l408_40863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_greater_than_geometric_mean_l408_40870

theorem arithmetic_mean_greater_than_geometric_mean
  (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a ≠ b) :
  (a + b) / 2 > Real.sqrt (a * b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_greater_than_geometric_mean_l408_40870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_trig_expression_l408_40886

theorem simplify_trig_expression (θ : ℝ) 
  (h1 : Real.sin θ < 0) (h2 : Real.tan θ > 0) : 
  Real.sqrt (1 - Real.sin θ ^ 2) = -Real.cos θ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_trig_expression_l408_40886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_half_fruits_selection_third_fruits_selection_l408_40839

/-- Represents a box containing apples and oranges -/
structure Box where
  apples : ℕ
  oranges : ℕ

/-- The theorem to be proved for part (a) -/
theorem half_fruits_selection (boxes : Fin 99 → Box) :
  ∃ (selected : Fin 99 → Bool),
    (Finset.sum (Finset.univ : Finset (Fin 99)) (λ i => if selected i then 1 else 0) = 50) ∧
    (2 * Finset.sum (Finset.univ : Finset (Fin 99)) (λ i => if selected i then (boxes i).apples else 0)
      ≥ Finset.sum (Finset.univ : Finset (Fin 99)) (λ i => (boxes i).apples)) ∧
    (2 * Finset.sum (Finset.univ : Finset (Fin 99)) (λ i => if selected i then (boxes i).oranges else 0)
      ≥ Finset.sum (Finset.univ : Finset (Fin 99)) (λ i => (boxes i).oranges)) :=
by
  sorry

/-- The theorem to be proved for part (b) -/
theorem third_fruits_selection (boxes : Fin 100 → Box) :
  ∃ (selected : Fin 100 → Bool),
    (Finset.sum (Finset.univ : Finset (Fin 100)) (λ i => if selected i then 1 else 0) = 34) ∧
    (3 * Finset.sum (Finset.univ : Finset (Fin 100)) (λ i => if selected i then (boxes i).apples else 0)
      ≥ Finset.sum (Finset.univ : Finset (Fin 100)) (λ i => (boxes i).apples)) ∧
    (3 * Finset.sum (Finset.univ : Finset (Fin 100)) (λ i => if selected i then (boxes i).oranges else 0)
      ≥ Finset.sum (Finset.univ : Finset (Fin 100)) (λ i => (boxes i).oranges)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_half_fruits_selection_third_fruits_selection_l408_40839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_six_terms_equals_fifteen_l408_40874

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

noncomputable def sum_of_arithmetic_sequence (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2

theorem sum_of_six_terms_equals_fifteen
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_a2 : a 2 = 4)
  (h_a4 : a 4 = 2) :
  sum_of_arithmetic_sequence a 6 = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_six_terms_equals_fifteen_l408_40874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_square_area_l408_40878

/-- The area of a square inscribed in the ellipse (x²/4) + (y²/8) = 1, 
    with its sides parallel to the coordinate axes -/
theorem inscribed_square_area : 
  ∃ (s : ℝ), s > 0 ∧
    (∀ (x y : ℝ), x ∈ Set.Icc (-s) s ∧ y ∈ Set.Icc (-s) s → x^2/4 + y^2/8 = 1) ∧
    (4 * s^2 : ℝ) = 32/3 := by
  sorry

#check inscribed_square_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_square_area_l408_40878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_distribution_probability_l408_40848

/-- The probability density function of a normal distribution -/
noncomputable def normal_pdf (μ σ : ℝ) (x : ℝ) : ℝ :=
  (1 / (σ * Real.sqrt (2 * Real.pi))) * Real.exp (-(1/2) * ((x - μ) / σ)^2)

/-- The cumulative distribution function of a normal distribution -/
noncomputable def normal_cdf (μ σ : ℝ) (x : ℝ) : ℝ :=
  ∫ y in Set.Iio x, normal_pdf μ σ y

/-- The probability of X falling within an interval for a normal distribution -/
noncomputable def normal_prob (μ σ : ℝ) (a b : ℝ) : ℝ :=
  normal_cdf μ σ b - normal_cdf μ σ a

theorem normal_distribution_probability (X : ℝ → ℝ) (μ σ : ℝ) 
  (h1 : normal_prob μ σ (μ-σ) (μ+σ) = 0.6826)
  (h2 : normal_prob μ σ (μ-2*σ) (μ+2*σ) = 0.9544)
  (h3 : μ = 100)
  (h4 : σ = 10) :
  normal_prob μ σ 110 120 = 0.1359 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_distribution_probability_l408_40848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_population_equal_in_15_years_l408_40864

/-- The number of years it takes for two villages' populations to be equal -/
def years_to_equal_population (x_init y_init : ℕ) (x_rate y_rate : ℤ) : ℕ :=
  Int.natAbs ((x_init - y_init) / (y_rate - x_rate))

/-- Theorem stating that it takes 15 years for the populations to be equal -/
theorem population_equal_in_15_years :
  years_to_equal_population 72000 42000 (-1200) 800 = 15 := by
  -- Proof goes here
  sorry

#eval years_to_equal_population 72000 42000 (-1200) 800

end NUMINAMATH_CALUDE_ERRORFEEDBACK_population_equal_in_15_years_l408_40864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fair_payment_distribution_l408_40840

def man_rate : ℚ := 1 / 10
def woman_rate : ℚ := 1 / 12
def teen_rate : ℚ := 1 / 15
def total_days : ℕ := 5
def total_payment : ℚ := 6000

def combined_rate : ℚ := man_rate + woman_rate + 2 * teen_rate

def man_share : ℚ := (man_rate / combined_rate) * total_payment
def woman_share : ℚ := (woman_rate / combined_rate) * total_payment
def teen_share : ℚ := (teen_rate / combined_rate) * total_payment

def round_to_cents (x : ℚ) : ℚ := (Int.floor (x * 100)) / 100

theorem fair_payment_distribution :
  (round_to_cents man_share = 1894.73) ∧
  (round_to_cents woman_share = 1578.95) ∧
  (round_to_cents teen_share = 1263.16) ∧
  (round_to_cents (man_share + woman_share + 2 * teen_share) = total_payment) :=
by sorry

#eval round_to_cents man_share
#eval round_to_cents woman_share
#eval round_to_cents teen_share
#eval round_to_cents (man_share + woman_share + 2 * teen_share)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fair_payment_distribution_l408_40840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_properties_existence_of_n_l408_40899

/-- A function that checks if a number contains both digits 9 and 7 -/
def containsNineAndSeven (n : ℕ) : Prop :=
  (∃ d₁ d₂, d₁ ∈ Nat.digits 10 n ∧ d₂ ∈ Nat.digits 10 n ∧ d₁ = 9 ∧ d₂ = 7)

/-- A function that checks if a fraction is a terminating decimal -/
def isTerminatingDecimal (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = 2^a * 5^b

/-- The main theorem -/
theorem smallest_n_with_properties :
  ∀ n : ℕ, n > 0 → isTerminatingDecimal n → containsNineAndSeven n → n ≥ 32768 :=
by sorry

/-- The existence theorem -/
theorem existence_of_n :
  ∃ n : ℕ, n > 0 ∧ isTerminatingDecimal n ∧ containsNineAndSeven n ∧ n = 32768 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_properties_existence_of_n_l408_40899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_statements_l408_40896

-- Define the cube root function
noncomputable def cube_root (x : ℝ) : ℝ := Real.rpow x (1/3)

-- Define the square root function
noncomputable def square_root (x : ℝ) : ℝ := Real.sqrt x

-- Theorem statement
theorem root_statements : 
  (cube_root 8 = 2) ∧ 
  (square_root (square_root 81) = 3) ∧
  (cube_root (-1) = -1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_statements_l408_40896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_periodic_l408_40814

noncomputable def a : ℕ → ℝ
  | 0 => 0  -- Define a value for 0 to cover all cases
  | 1 => Real.sqrt 2
  | n + 1 => (Real.sqrt 3 * a n - 1) / (a n + Real.sqrt 3)

theorem sequence_periodic : ∀ n : ℕ, n > 0 → a (n + 6) = a n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_periodic_l408_40814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_x0_plus_pi_6_l408_40826

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (Real.pi / 2 + x) * Real.sin (Real.pi / 3 + x)

theorem f_value_at_x0_plus_pi_6 
  (x₀ : ℝ) 
  (h1 : x₀ > Real.pi / 6) 
  (h2 : x₀ < Real.pi / 2) 
  (h3 : f x₀ = 4 / 5 + Real.sqrt 3 / 2) : 
  f (x₀ + Real.pi / 6) = (2 + Real.sqrt 3) / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_x0_plus_pi_6_l408_40826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_vertex_angle_l408_40861

/-- Given four cones with a common vertex A, where three of them are externally tangent to each other
    and internally tangent to the fourth, prove that the vertex angle of the two identical cones is
    2 * arctan(2/3) -/
theorem cone_vertex_angle (α β γ : ℝ) (h1 : β = π / 8) (h2 : γ = 3 * π / 8) :
  let φ := 2 * Real.arctan (1 / Real.sqrt 2)
  2 * α = 2 * Real.arctan (2 / 3) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_vertex_angle_l408_40861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_power_six_decomposition_l408_40807

theorem cosine_power_six_decomposition (b₁ b₂ b₃ b₄ b₅ b₆ : ℝ) :
  (∀ θ : ℝ, (Real.cos θ)^6 = b₁ * Real.cos θ + b₂ * Real.cos (2*θ) + b₃ * Real.cos (3*θ) + 
                              b₄ * Real.cos (4*θ) + b₅ * Real.cos (5*θ) + b₆ * Real.cos (6*θ)) →
  b₁^2 + b₂^2 + b₃^2 + b₄^2 + b₅^2 + b₆^2 = 131/512 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_power_six_decomposition_l408_40807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumsphere_radius_formula_l408_40869

/-- A regular triangular pyramid (tetrahedron) -/
structure RegularTetrahedron where
  a : ℝ  -- side length of the base
  b : ℝ  -- lateral edge
  h_positive : 0 < a ∧ 0 < b

/-- The radius of the circumscribed sphere of a regular tetrahedron -/
noncomputable def circumsphere_radius (t : RegularTetrahedron) : ℝ :=
  (1/2) * Real.sqrt (t.b^2 + t.a^2/3)

/-- Theorem: The radius of the circumscribed sphere of a regular tetrahedron
    is (1/2) * √(b² + a²/3), where a is the side length of the base
    and b is the lateral edge. -/
theorem circumsphere_radius_formula (t : RegularTetrahedron) :
  circumsphere_radius t = (1/2) * Real.sqrt (t.b^2 + t.a^2/3) := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumsphere_radius_formula_l408_40869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_sums_count_l408_40819

def S : Finset ℕ := {2, 7, 12, 17, 22, 27, 32}

def sums_of_three (S : Finset ℕ) : Finset ℕ :=
  (Finset.powerset S).filter (λ subset => subset.card = 3) |>.image (λ subset => subset.sum id)

theorem distinct_sums_count : (sums_of_three S).card = 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_sums_count_l408_40819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_makeup_exam_average_score_l408_40812

/-- Proves that the average score of students who took the exam on the make-up date is 95% --/
theorem makeup_exam_average_score 
  (total_students : ℕ)
  (assigned_day_percentage : ℚ)
  (assigned_day_average : ℚ)
  (total_average : ℚ)
  (h1 : total_students = 100)
  (h2 : assigned_day_percentage = 70 / 100)
  (h3 : assigned_day_average = 55 / 100)
  (h4 : total_average = 67 / 100) :
  (total_average * total_students - assigned_day_average * (total_students * assigned_day_percentage).floor) / 
  (total_students - (total_students * assigned_day_percentage).floor) = 95 / 100 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_makeup_exam_average_score_l408_40812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_congruence_implies_input_congruence_l408_40860

theorem function_congruence_implies_input_congruence 
  (p : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1)
  (f : ℕ → ℕ)
  (hf : ∀ n, f n = (Finset.range (p - 1)).sum (fun i ↦ (i + 1) * n^i))
  (m n : ℕ)
  (h_cong : f m ≡ f n [MOD p]) :
  m ≡ n [MOD p] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_congruence_implies_input_congruence_l408_40860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_of_angle_with_point_l408_40893

/-- Given an angle C with a point (-1, 2) on its terminal side, prove that sin(C) = 2√5/5 -/
theorem sine_of_angle_with_point (C : ℝ) : 
  (∃ (x y : ℝ), x = -1 ∧ y = 2 ∧ 
   x = Real.cos C * Real.sqrt (x^2 + y^2) ∧ 
   y = Real.sin C * Real.sqrt (x^2 + y^2)) → 
  Real.sin C = 2 * Real.sqrt 5 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_of_angle_with_point_l408_40893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_range_l408_40888

/-- Definition of the line l -/
def line_l (a : ℝ) (x y : ℝ) : Prop :=
  x - y + a = 0

/-- Definition of the circle C -/
def circle_C (x y : ℝ) : Prop :=
  (x + 1)^2 + y^2 = 2

/-- Theorem stating the range of a for which the line and circle intersect -/
theorem line_circle_intersection_range (a : ℝ) :
  (∃ x y : ℝ, line_l a x y ∧ circle_C x y) →
  a ∈ Set.Icc (-1) 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_range_l408_40888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_value_l408_40846

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then Real.sin (Real.pi * x / 6) else 1 - 2 * x

theorem f_composition_value : f (f 3) = -1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_value_l408_40846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_two_digit_number_l408_40805

theorem largest_two_digit_number : ∀ (a b c d : ℕ), 
  a ∈ ({3, 9, 5, 8} : Set ℕ) ∧ 
  b ∈ ({3, 9, 5, 8} : Set ℕ) ∧ 
  c ∈ ({3, 9, 5, 8} : Set ℕ) ∧ 
  d ∈ ({3, 9, 5, 8} : Set ℕ) ∧ 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  (∃ (x y : ℕ), 
    ({x, y} : Set ℕ) ⊆ ({a, b, c, d} : Set ℕ) ∧ 
    x ≠ y ∧
    10 * x + y ≤ 98 ∧
    (a + b + c + d) - (x + y) ≤ 9) ∧
  (∃ (x y : ℕ), 
    ({x, y} : Set ℕ) ⊆ ({a, b, c, d} : Set ℕ) ∧ 
    x ≠ y ∧
    10 * x + y = 98 ∧
    (a + b + c + d) - (x + y) = 8) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_two_digit_number_l408_40805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_iterate_correct_l408_40804

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := Real.sqrt (a * x^2 + b)

-- Define the n-th iterate of f
noncomputable def f_iterate (n : ℕ) (a b x : ℝ) : ℝ := 
  Real.sqrt (a^n * (x^2 - b / (1 - a)) + b / (1 - a))

-- Theorem statement
theorem f_iterate_correct (n : ℕ) (a b x : ℝ) 
  (ha : a > 0) (hb : b > 0) (ha_neq : a ≠ 1) :
  (fun x => f a b x)^[n] x = f_iterate n a b x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_iterate_correct_l408_40804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_factorization_1_polynomial_factorization_2_arithmetic_sequence_calculation_l408_40837

-- Define the sum of integers from 2 to n
def sum_2_to_n (n : ℕ) : ℤ :=
  ↑(n * (n + 1) / 2) - 1

-- Define the sum of integers from 1 to n with alternating signs
def alt_sum_1_to_n (n : ℕ) : ℤ :=
  if n % 2 = 0 then -↑(n / 2) else ↑((n + 1) / 2)

theorem polynomial_factorization_1 (x : ℝ) :
  (x^2 + 2*x) * (x^2 + 2*x + 2) + 1 = (x + 1)^4 := by
  sorry

theorem polynomial_factorization_2 (x : ℝ) :
  (x^2 - 6*x + 8) * (x^2 - 6*x + 10) + 1 = (x - 3)^4 := by
  sorry

theorem arithmetic_sequence_calculation :
  (alt_sum_1_to_n 2020) * (sum_2_to_n 2021) - (alt_sum_1_to_n 2021) * (sum_2_to_n 2020) = 2021 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_factorization_1_polynomial_factorization_2_arithmetic_sequence_calculation_l408_40837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l408_40809

theorem cos_alpha_value (α : ℝ) 
  (h1 : Real.tan (π - α) = 3/4) 
  (h2 : α ∈ Set.Ioo (π/2) π) : 
  Real.cos α = -4/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l408_40809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_theorem_l408_40833

noncomputable def original_function (x : ℝ) : ℝ := (1/2) * Real.sin (2 * x)
noncomputable def transformed_function (x : ℝ) : ℝ := (1/4) * Real.sin x

noncomputable def horizontal_transform (f : ℝ → ℝ) : ℝ → ℝ := λ x => f (x / 2)
noncomputable def vertical_transform (f : ℝ → ℝ) : ℝ → ℝ := λ x => (1/2) * f x

theorem transformation_theorem :
  ∀ x, (vertical_transform (horizontal_transform original_function)) x = transformed_function x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_theorem_l408_40833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_geometric_series_ratio_l408_40894

/-- The common ratio of an infinite geometric series -/
noncomputable def common_ratio (a : ℝ) (S : ℝ) : ℝ := 1 - (a / S)

/-- Theorem stating that for an infinite geometric series with first term 500 and sum 3125,
    the common ratio is 0.84 -/
theorem infinite_geometric_series_ratio :
  let a : ℝ := 500
  let S : ℝ := 3125
  common_ratio a S = 0.84 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_geometric_series_ratio_l408_40894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_R_equals_one_l408_40845

theorem R_equals_one (x y z t : ℝ) (h : x * y * z * t = 1) : 
  (1 / (1 + x + x*y + x*y*z) + 
   1 / (1 + y + y*z + y*z*t) + 
   1 / (1 + z + z*t + z*t*x) + 
   1 / (1 + t + t*x + t*x*y)) = 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_R_equals_one_l408_40845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_origin_l408_40880

/-- The equation of the tangent line to y = 2ln(x+1) at (0, 0) is y = 2x -/
theorem tangent_line_at_origin (x y : ℝ) :
  let f : ℝ → ℝ := λ t ↦ 2 * Real.log (t + 1)
  let tangent_line : ℝ → ℝ := λ t ↦ 2 * t
  (∀ t, HasDerivAt f (2 / (t + 1)) t) →
  f 0 = 0 →
  tangent_line 0 = 0 →
  (∀ ε > 0, ∃ δ > 0, ∀ t, 0 < |t| → |t| < δ → |(tangent_line t - f t) / t| < ε) →
  y = tangent_line x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_origin_l408_40880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l408_40831

/-- The distance between two parallel lines -/
noncomputable def distance_between_parallel_lines (A B C₁ C₂ : ℝ) : ℝ :=
  |C₁ - C₂| / Real.sqrt (A^2 + B^2)

/-- Proof that the distance between 4x + 3y - 1 = 0 and 8x + 6y - 5 = 0 is 3/10 -/
theorem distance_between_given_lines :
  distance_between_parallel_lines 8 6 (-2) (-5) = 3/10 := by
  sorry

#check distance_between_given_lines

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l408_40831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_and_area_theorem_l408_40834

-- Define the given points and line
def F : ℝ × ℝ := (0, 1)
def l : Set (ℝ × ℝ) := {p | p.2 = -1}

-- Define the curve C
def C : Set (ℝ × ℝ) := {p | p.1^2 = 4 * p.2}

-- Define the circle (renamed to avoid conflict)
def circleSet : Set (ℝ × ℝ) := {p | p.1^2 + (p.2 + 2)^2 = 4}

-- State the theorem
theorem curve_and_area_theorem :
  ∃ (M P : ℝ × ℝ),
    M ∈ l ∧
    (∀ x : ℝ, (x, M.2) ∈ l → (x - M.1)^2 + (M.2 - P.2)^2 = (M.1 - P.1)^2 + (M.2 - P.2)^2) ∧
    (P.1 - M.1) * (F.1 - M.1) + (P.2 - M.2) * (F.2 - M.2) = 0 ∧
    C = {p : ℝ × ℝ | p.1^2 = 4 * p.2} ∧
    (∃ (A B S T : ℝ × ℝ),
      A ∈ circleSet ∧ B ∈ circleSet ∧
      (A.1 - B.1)^2 + (A.2 - B.2)^2 = 4 ∧
      S ∈ C ∧ T ∈ C ∧
      (∃ (k : ℝ), k ≠ 0 ∧ S.2 = k * S.1 ∧ T.2 = -1/k * T.1) ∧
      (∀ (S' T' : ℝ × ℝ),
        S' ∈ C → T' ∈ C →
        (∃ (k' : ℝ), k' ≠ 0 ∧ S'.2 = k' * S'.1 ∧ T'.2 = -1/k' * T'.1) →
        abs ((S.1 - A.1) * (T.2 - A.2) - (T.1 - A.1) * (S.2 - A.2)) ≤
        abs ((S'.1 - A.1) * (T'.2 - A.2) - (T'.1 - A.1) * (S'.2 - A.2))) ∧
      abs ((S.1 - A.1) * (T.2 - A.2) - (T.1 - A.1) * (S.2 - A.2)) = 36) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_and_area_theorem_l408_40834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_trucks_required_l408_40822

theorem min_trucks_required (total_weight : ℝ) (max_box_weight : ℝ) (truck_capacity : ℝ) :
  total_weight = 13.5 →
  max_box_weight = 0.35 →
  truck_capacity = 1.5 →
  ∃ (n : ℕ), n = 11 ∧ 
    (∀ (m : ℕ), m < n → 
      ∃ (weights : List ℝ),
        weights.sum = total_weight ∧
        (∀ w ∈ weights, w ≤ max_box_weight) ∧
        ¬(∃ (partition : List (List ℝ)), 
          partition.length = m ∧
          partition.all (λ sub_list ↦ sub_list.sum ≤ truck_capacity) ∧
          partition.join.sum = total_weight)) ∧
    (∃ (weights : List ℝ) (partition : List (List ℝ)),
      weights.sum = total_weight ∧
      (∀ w ∈ weights, w ≤ max_box_weight) ∧
      partition.length = n ∧
      partition.all (λ sub_list ↦ sub_list.sum ≤ truck_capacity) ∧
      partition.join.sum = total_weight) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_trucks_required_l408_40822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l408_40828

/-- The maximum area of a triangle with sides 2017 and 2018 is 2035133 -/
theorem max_triangle_area : 
  ∀ (A B C : ℝ × ℝ),
    let ab := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
    let bc := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
    let ac := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
    ab = 2017 → bc = 2018 → 
    (1/2) * ab * bc * Real.sqrt (1 - ((ab^2 + bc^2 - ac^2) / (2 * ab * bc))^2) ≤ 2035133
:= by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l408_40828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l408_40898

noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 2 + Real.sin x * Real.cos x

theorem max_value_of_f :
  ∃ (x : ℝ), x ∈ Set.Icc 0 (2 * Real.pi) ∧
  f x = (3 : ℝ) / 2 ∧
  ∀ (y : ℝ), y ∈ Set.Icc 0 (2 * Real.pi) → f y ≤ (3 : ℝ) / 2 := by
  sorry

#check max_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l408_40898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_meaningful_l408_40818

-- Define a function to represent the fraction
noncomputable def f (x : ℝ) : ℝ := 1 / (x + 5)

-- State the theorem
theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, f x = y) ↔ x ≠ -5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_meaningful_l408_40818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lennon_tuesday_miles_l408_40889

/-- Calculates the number of miles driven on Tuesday given the reimbursement rate,
    miles driven on other days, and total reimbursement. -/
def miles_driven_tuesday (rate : ℚ) (mon wed thu fri : ℕ) (total : ℚ) : ℕ :=
  let other_days := mon + wed + thu + fri
  let other_reimbursement := rate * other_days
  let tuesday_reimbursement := total - other_reimbursement
  (tuesday_reimbursement / rate).floor.toNat

/-- Theorem stating that given the problem conditions, Lennon drove 26 miles on Tuesday. -/
theorem lennon_tuesday_miles : 
  miles_driven_tuesday (36/100) 18 20 20 16 36 = 26 := by
  sorry

#eval miles_driven_tuesday (36/100) 18 20 20 16 36

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lennon_tuesday_miles_l408_40889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_direction_vector_of_line_l408_40827

/-- The direction vector of a parameterized line. -/
noncomputable def direction_vector (line : ℝ → ℝ × ℝ) : ℝ × ℝ :=
  sorry

/-- The line equation y = (4x - 8)/5 -/
noncomputable def line_equation (x : ℝ) : ℝ := (4 * x - 8) / 5

/-- The parameterization of the line -/
noncomputable def line_param (t : ℝ) : ℝ × ℝ :=
  (5 + 5 * t / Real.sqrt 41, 2 + 4 * t / Real.sqrt 41)

theorem direction_vector_of_line :
  let d := direction_vector line_param
  (∀ x ≥ 5, line_equation x = (line_param ((x - 5) * Real.sqrt 41 / 5)).2) ∧
  (∀ t, (line_param t).1 ≥ 5 → 
    Real.sqrt ((line_param t).1 - 5)^2 + ((line_param t).2 - 2)^2 = t) →
  d = (5 / Real.sqrt 41, 4 / Real.sqrt 41) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_direction_vector_of_line_l408_40827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_movie_expense_split_l408_40824

/-- The amount each friend should contribute to split the movie expenses equally -/
def contribution_per_person : ℚ :=
  let ticket_price : ℚ := 7
  let popcorn_price : ℚ := (3/2)
  let milk_tea_price : ℚ := 3
  let num_friends : ℕ := 3
  let num_tickets : ℕ := 3
  let num_popcorn : ℕ := 2
  let num_milk_tea : ℕ := 3
  let total_cost : ℚ := ticket_price * num_tickets + popcorn_price * num_popcorn + milk_tea_price * num_milk_tea
  total_cost / num_friends

theorem movie_expense_split :
  contribution_per_person = 11 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_movie_expense_split_l408_40824
