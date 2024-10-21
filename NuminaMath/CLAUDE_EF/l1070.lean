import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_is_projective_l1070_107069

-- Define the basic structures
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

structure Line where
  a : Point
  b : Point

structure Plane where
  a : Point
  b : Point
  c : Point

structure Lens where
  O : Point -- center of the lens
  a : Line  -- optical axis
  f : Line  -- focal line
  π : Plane -- plane passing through optical axis

-- Define the transformation (noncomputable as it involves geometric operations)
noncomputable def T (lens : Lens) (M : Point) : Point :=
  sorry

-- Define what it means for a transformation to be projective
def IsProjective (f : Point → Point) : Prop :=
  sorry

-- State the theorem
theorem transformation_is_projective (lens : Lens) :
  IsProjective (T lens) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_is_projective_l1070_107069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_to_excircle_touch_distance_intersection_to_B_distance_l1070_107003

/-- Triangle ABC with sides a, b, c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_pos_a : 0 < a
  h_pos_b : 0 < b
  h_pos_c : 0 < c
  h_triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b
  h_b_geq_a : b ≥ a

/-- The incenter of a triangle -/
noncomputable def incenter (t : Triangle) : ℝ × ℝ := sorry

/-- The point where the excircle opposite to vertex C touches AB -/
noncomputable def excircle_touch_point (t : Triangle) : ℝ := sorry

/-- The midpoint of AB -/
noncomputable def midpoint_AB (t : Triangle) : ℝ := t.c / 2

/-- The intersection of the line connecting the midpoint of AB and the incenter with BC -/
noncomputable def intersection_point (t : Triangle) : ℝ := sorry

theorem midpoint_to_excircle_touch_distance (t : Triangle) :
  |midpoint_AB t - excircle_touch_point t| = (t.b - t.a) / 2 := by sorry

theorem intersection_to_B_distance (t : Triangle) :
  |t.a - intersection_point t| = t.a * t.c / (t.b + t.c - t.a) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_to_excircle_touch_distance_intersection_to_B_distance_l1070_107003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l1070_107076

/-- Represents a parabola with focus on positive x-axis and vertex at origin -/
structure Parabola where
  p : ℝ
  eq : ℝ → ℝ → Prop := fun x y => y^2 = 2 * p * x

/-- Represents a line passing through a point -/
structure Line where
  k : ℝ
  b : ℝ
  eq : ℝ → ℝ → Prop := fun x y => y = k * x + b

/-- Calculates the area of a triangle given three points -/
noncomputable def triangleArea (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  (1/2) * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

theorem parabola_line_intersection (p : Parabola) (l : Line) :
  p.eq 1 (-2) →
  (∃ x1 y1 x2 y2, p.eq x1 y1 ∧ p.eq x2 y2 ∧ l.eq x1 y1 ∧ l.eq x2 y2) →
  triangleArea 0 0 1 (-2) 1 2 = 2 * Real.sqrt 2 →
  (l.eq = fun x y => y = -x + 1 ∨ l.eq = fun x y => y = x - 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l1070_107076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_intersection_point_l1070_107022

/-- The hyperbola equation xy = 2 -/
def hyperbola (x y : ℝ) : Prop := x * y = 2

/-- The circle equation (x - 3)^2 + (y + 1)^2 = 25 -/
def circle_eq (x y : ℝ) : Prop := (x - 3)^2 + (y + 1)^2 = 25

/-- The set of known intersection points -/
noncomputable def known_points : Set (ℝ × ℝ) := {(4, 1/2), (-2, -1), (2/3, 3)}

/-- The fourth intersection point -/
noncomputable def fourth_point : ℝ × ℝ := (-3/4, -8/3)

theorem fourth_intersection_point :
  (∀ (p : ℝ × ℝ), p ∈ known_points → hyperbola p.1 p.2 ∧ circle_eq p.1 p.2) →
  hyperbola fourth_point.1 fourth_point.2 ∧
  circle_eq fourth_point.1 fourth_point.2 ∧
  fourth_point ∉ known_points :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_intersection_point_l1070_107022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equality_proof_l1070_107058

theorem log_equality_proof : ∃ y : ℝ, y > 0 ∧ Real.log 125 / Real.log y = Real.log 27 / Real.log 3 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equality_proof_l1070_107058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coplanar_vectors_lambda_l1070_107004

def a (l : ℝ) : Fin 3 → ℝ := ![1, l, 2]
def b : Fin 3 → ℝ := ![2, -1, 2]
def c : Fin 3 → ℝ := ![1, 4, 4]

def are_coplanar (v1 v2 v3 : Fin 3 → ℝ) : Prop :=
  ∃ (m n : ℝ), v3 = m • v1 + n • v2

theorem coplanar_vectors_lambda (l : ℝ) :
  are_coplanar (a l) b c → l = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coplanar_vectors_lambda_l1070_107004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_four_least_satisfying_l1070_107026

-- Define τ(n) as the number of positive integer divisors of n
def τ (n : ℕ) : ℕ := (Nat.divisors n).card

-- Define a function to check if a number satisfies the condition
def satisfiesCondition (n : ℕ) : Bool := τ n + τ (n + 1) = 8

-- Define a function to get the four least positive integers satisfying the condition
def fourLeastSatisfying : List ℕ :=
  (List.range 1000).filter satisfiesCondition |>.take 4

-- Theorem statement
theorem sum_of_four_least_satisfying :
  (fourLeastSatisfying.sum) = 80 := by sorry

#eval fourLeastSatisfying
#eval fourLeastSatisfying.sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_four_least_satisfying_l1070_107026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_theorem_l1070_107064

theorem max_value_theorem (x : ℝ) (hx : x > 0) :
  (x^2 + 3 - Real.sqrt (x^4 + 9)) / x ≤ 6 * Real.sqrt 6 / (6 + 3 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_theorem_l1070_107064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_convergence_l1070_107023

def x : ℕ → ℚ
  | 0 => 7
  | n + 1 => (x n ^ 2 + 6 * x n + 8) / (x n + 7)

theorem sequence_convergence :
  ∃ m : ℕ, m ∈ Set.Icc 69 205 ∧
    x m ≤ 3 + 1 / (2^18) ∧
    ∀ k : ℕ, k > 0 → k < m → x k > 3 + 1 / (2^18) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_convergence_l1070_107023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_cone_volume_ratio_l1070_107034

theorem inscribed_sphere_cone_volume_ratio (α : Real) (α_pos : 0 < α) (α_lt_pi_div_2 : α < π / 2) :
  let cone_volume := (1/3) * π * (Real.tan α)^2 * (1 / Real.sin α)
  let sphere_volume := (4/3) * π * (Real.cos α / (1 + Real.sin α))^3
  sphere_volume / cone_volume = (4 * Real.cos α^2 * Real.sin α) / (1 + Real.sin α)^3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_cone_volume_ratio_l1070_107034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_two_roots_derivative_positive_l1070_107057

/-- The function f(x) = x^2 - (a-2)x - a*ln(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - (a-2)*x - a * Real.log x

/-- The derivative of f(x) -/
noncomputable def f_prime (a : ℝ) (x : ℝ) : ℝ := 2*x - (a-2) - a/x

theorem f_two_roots_derivative_positive (a : ℝ) (c : ℝ) (x₁ x₂ : ℝ) :
  a > 0 →
  0 < x₁ →
  0 < x₂ →
  x₁ < x₂ →
  f a x₁ = c →
  f a x₂ = c →
  f_prime a ((x₁ + x₂) / 2) > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_two_roots_derivative_positive_l1070_107057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_c_days_l1070_107078

/-- Proves that given the conditions of the problem, c worked for 4 days. -/
theorem worker_c_days : ∀ (days_c : ℕ),
  let wage_c : ℕ := 110
  let wage_b : ℕ := (4 * wage_c) / 5
  let wage_a : ℕ := (3 * wage_c) / 5
  let days_a : ℕ := 6
  let days_b : ℕ := 9
  let total_earning : ℕ := 1628
  (wage_a * days_a + wage_b * days_b + wage_c * days_c = total_earning) →
  days_c = 4 := by
  intro days_c
  intro h
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_c_days_l1070_107078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kite_circle_radius_l1070_107091

/-- A kite is a quadrilateral with two pairs of adjacent sides of equal length. -/
structure Kite where
  a : ℝ
  b : ℝ
  h : a ≠ b

/-- The radius of the circle touching the extensions of all four sides of a kite. -/
noncomputable def circle_radius (k : Kite) : ℝ := (k.a * k.b) / (k.a - k.b)

/-- 
Theorem: For a kite with sides a and b (a ≠ b), where sides of different lengths 
enclose a right angle, the radius of the circle that touches the extensions of 
all four sides is given by ab / (a - b).
-/
theorem kite_circle_radius (k : Kite) : 
  circle_radius k = (k.a * k.b) / (k.a - k.b) := by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_kite_circle_radius_l1070_107091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_ten_equals_twenty_l1070_107041

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  let y := (x - 1) / 3
  y^2 + 3*y + 2

-- State the theorem
theorem f_of_ten_equals_twenty : f 10 = 20 := by
  -- Unfold the definition of f
  unfold f
  -- Simplify the let expression
  simp
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_ten_equals_twenty_l1070_107041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_tetrahedron_volume_l1070_107029

/-- Tetrahedron with given properties -/
structure Tetrahedron where
  AB : ℝ
  CD : ℝ
  distance : ℝ
  angle : ℝ

/-- Volume of a tetrahedron with given properties -/
noncomputable def tetrahedron_volume (t : Tetrahedron) : ℝ :=
  (1/6) * t.AB * t.CD * Real.sin t.angle * t.distance

/-- Theorem stating the volume of the specific tetrahedron -/
theorem specific_tetrahedron_volume :
  let t : Tetrahedron := {
    AB := 3,
    CD := 2,
    distance := 2,
    angle := π/6
  }
  tetrahedron_volume t = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_tetrahedron_volume_l1070_107029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_equals_positive_integers_l1070_107075

def A : Set ℕ := sorry

axiom A_has_at_least_3_elements : ∃ (x y z : ℕ), x ∈ A ∧ y ∈ A ∧ z ∈ A ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z

axiom A_closed_under_divisors : ∀ (a : ℕ), a ∈ A → ∀ (d : ℕ), d ∣ a → d ∈ A

axiom A_closure_property : ∀ (a b : ℕ), a ∈ A → b ∈ A → 1 < a → a < b → (1 + a * b) ∈ A

theorem A_equals_positive_integers : A = {n : ℕ | n > 0} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_equals_positive_integers_l1070_107075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_middle_to_bottom_segment_ratio_l1070_107066

/-- A right circular cone with given dimensions and segments -/
structure Cone where
  height : ℝ
  baseRadius : ℝ
  topSegmentHeight : ℝ
  middleSegmentHeight : ℝ
  bottomSegmentHeight : ℝ

/-- The volume of a cone segment given its height and radius -/
noncomputable def segmentVolume (h : ℝ) (r : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

/-- The radius at a given height in the cone -/
noncomputable def radiusAtHeight (c : Cone) (h : ℝ) : ℝ := 
  c.baseRadius * (c.height - h) / c.height

/-- Theorem: The ratio of middle to bottom segment volumes is 13/97 -/
theorem middle_to_bottom_segment_ratio (c : Cone) 
    (h_total : c.height = 100)
    (h_base : c.baseRadius = 50)
    (h_top : c.topSegmentHeight = 20)
    (h_middle : c.middleSegmentHeight = 30)
    (h_bottom : c.bottomSegmentHeight = 50) :
    let r_middle := radiusAtHeight c (c.topSegmentHeight + c.middleSegmentHeight)
    let r_bottom := c.baseRadius
    let v_middle := segmentVolume c.middleSegmentHeight r_middle - 
                    segmentVolume c.topSegmentHeight (radiusAtHeight c c.topSegmentHeight)
    let v_bottom := segmentVolume c.bottomSegmentHeight r_bottom - 
                    segmentVolume (c.topSegmentHeight + c.middleSegmentHeight) r_middle
    v_middle / v_bottom = 13 / 97 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_middle_to_bottom_segment_ratio_l1070_107066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_plus_two_beta_l1070_107063

theorem sin_alpha_plus_two_beta 
  (α β : ℝ) 
  (h1 : 0 < α ∧ α < π / 2) 
  (h2 : 0 < β ∧ β < π / 2) 
  (h3 : Real.cos (α + β) = -5/13) 
  (h4 : Real.sin β = 3/5) : 
  Real.sin (α + 2*β) = 33/65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_plus_two_beta_l1070_107063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_intersects_x_axis_l1070_107087

-- Define the logarithm function with base 1/2
noncomputable def log_half (x : ℝ) : ℝ := Real.log x / Real.log (1/2)

-- Define our function
noncomputable def f (x : ℝ) : ℝ := log_half (-x)

-- State the theorem
theorem f_intersects_x_axis :
  ∃ (x : ℝ), x < 0 ∧ f x = 0 ∧ x = -1 := by
  -- Proof goes here
  sorry

-- Example to show that f(-1) = 0
example : f (-1) = 0 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_intersects_x_axis_l1070_107087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_volume_percentage_l1070_107012

/-- Represents a cylindrical tank with given height and circumference -/
structure Tank where
  height : ℝ
  circumference : ℝ

/-- Calculates the volume of a cylindrical tank -/
noncomputable def tankVolume (t : Tank) : ℝ :=
  (t.circumference ^ 2 * t.height) / (4 * Real.pi)

/-- Theorem stating the combined volume of tanks A and C as a percentage of tank B -/
theorem combined_volume_percentage (tankA tankB tankC : Tank)
  (hA : tankA.height = 10 ∧ tankA.circumference = 7)
  (hB : tankB.height = 7 ∧ tankB.circumference = 10)
  (hC : tankC.height = 5 ∧ tankC.circumference = 14) :
  (tankVolume tankA + tankVolume tankC) / tankVolume tankB * 100 = 210 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_volume_percentage_l1070_107012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l1070_107032

-- Define a, b, and c
noncomputable def a : ℝ := Real.log 10 / Real.log 5
noncomputable def b : ℝ := Real.log 12 / Real.log 6
noncomputable def c : ℝ := Real.log 14 / Real.log 7

-- Theorem statement
theorem log_inequality : a > b ∧ b > c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l1070_107032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_305207_is_prime_l1070_107077

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def candidate_numbers : List ℕ := [305201, 305203, 305205, 305207, 305209]

theorem only_305207_is_prime :
  ∃! n, n ∈ candidate_numbers ∧ is_prime n ∧ n = 305207 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_305207_is_prime_l1070_107077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_of_right_isosceles_triangle_l1070_107092

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := Real.sqrt (1 - e.b^2 / e.a^2)

/-- The focal distance of an ellipse -/
noncomputable def focal_distance (e : Ellipse) : ℝ := Real.sqrt (e.a^2 - e.b^2)

theorem ellipse_eccentricity_of_right_isosceles_triangle (e : Ellipse) :
  ∃ (M : ℝ × ℝ),
    M.1^2 / e.a^2 + M.2^2 / e.b^2 = 1 ∧  -- M is on the ellipse
    M.1 = (e.a - focal_distance e) / 2 ∧  -- x-coordinate of M
    M.2 = (e.a + focal_distance e) / 2  -- y-coordinate of M
    →
    eccentricity e = 2 - Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_of_right_isosceles_triangle_l1070_107092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_phi_value_l1070_107097

-- Define the original function f(x)
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 2 * (Real.cos x)^2 - 1

-- Define the translated function g(x)
noncomputable def g (φ : ℝ) (x : ℝ) : ℝ := f (x - φ)

-- Define the symmetry condition
def is_symmetric_about_y_axis (h : ℝ → ℝ) : Prop :=
  ∀ x, h x = h (-x)

-- Theorem statement
theorem min_phi_value :
  ∃ φ : ℝ, φ > 0 ∧ 
  is_symmetric_about_y_axis (g φ) ∧
  (∀ ψ : ℝ, ψ > 0 ∧ is_symmetric_about_y_axis (g ψ) → φ ≤ ψ) ∧
  φ = Real.pi / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_phi_value_l1070_107097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_quadratic_l1070_107050

theorem divisors_of_quadratic (a b : ℤ) (h : 4 * b = 9 - 3 * a) :
  (Finset.filter (λ i : ℕ => (b^2 + 12*b + 15).natAbs % i = 0)
    (Finset.range 7)).card = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_quadratic_l1070_107050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_opposite_side_equation_l1070_107061

/-- Given a parallelogram ABCD with center at (0, 3) and side AB on the line 3x + 4y - 2 = 0,
    the equation of the line on which side CD lies is 3x + 4y - 22 = 0 -/
theorem parallelogram_opposite_side_equation (A B C D : ℝ × ℝ) :
  let center := (0, 3)
  let line_AB (x y : ℝ) := 3 * x + 4 * y - 2 = 0
  (∀ p ∈ ({A, B} : Set (ℝ × ℝ)), line_AB p.1 p.2) →
  (A + C) / 2 = center ∧ (B + D) / 2 = center →
  (∀ p ∈ ({C, D} : Set (ℝ × ℝ)), 3 * p.1 + 4 * p.2 - 22 = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_opposite_side_equation_l1070_107061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_battery_life_is_four_hours_l1070_107027

/-- Represents the battery life of Jamie's smartphone --/
structure BatteryLife where
  standbyTime : ℚ  -- Time the battery lasts in standby mode
  usageTime : ℚ    -- Time the battery lasts in constant use mode
  totalOnTime : ℚ  -- Total time the phone has been on
  usedTime : ℚ     -- Time the phone has been actively used

/-- Calculates the remaining battery life --/
def remainingBatteryLife (b : BatteryLife) : ℚ :=
  let standbyRate := 1 / b.standbyTime
  let usageRate := 1 / b.usageTime
  let standbyConsumption := (b.totalOnTime - b.usedTime) * standbyRate
  let usageConsumption := b.usedTime * usageRate
  let remainingBattery := 1 - (standbyConsumption + usageConsumption)
  remainingBattery / standbyRate

/-- Theorem stating that the remaining battery life is 4 hours --/
theorem remaining_battery_life_is_four_hours (b : BatteryLife) 
    (h1 : b.standbyTime = 20)
    (h2 : b.usageTime = 4)
    (h3 : b.totalOnTime = 10)
    (h4 : b.usedTime = 3/2) :
  remainingBatteryLife b = 4 := by
  sorry

#eval remainingBatteryLife { standbyTime := 20, usageTime := 4, totalOnTime := 10, usedTime := 3/2 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_battery_life_is_four_hours_l1070_107027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_simplification_l1070_107033

theorem trigonometric_equation_simplification :
  ∃ (a b c : ℕ+), 
    (∀ x : ℝ, (Real.sin x)^2 + (Real.sin (3*x))^2 + (Real.sin (5*x))^2 + (Real.sin (7*x))^2 = 2 ↔ 
      Real.cos (a.val * x) * Real.cos (b.val * x) * Real.cos (c.val * x) = 0) ∧
    a.val + b.val + c.val = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_simplification_l1070_107033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_given_equal_perimeter_and_triangle_area_l1070_107072

/-- Given a regular triangle and a regular hexagon with the same perimeter,
    if the area of the triangle is 2, then the area of the hexagon is 3. -/
theorem hexagon_area_given_equal_perimeter_and_triangle_area (s t : ℝ) :
  s > 0 ∧ t > 0 ∧  -- side lengths are positive
  3 * s = 6 * t ∧  -- perimeters are equal
  (Real.sqrt 3 / 4) * s^2 = 2  -- area of triangle is 2
  →
  6 * ((Real.sqrt 3 / 4) * t^2) = 3  -- area of hexagon is 3
  := by
    intro h
    sorry

#check hexagon_area_given_equal_perimeter_and_triangle_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_given_equal_perimeter_and_triangle_area_l1070_107072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beta_quadrants_l1070_107016

theorem beta_quadrants (α β : Real) : 
  0 < α ∧ α < Real.pi / 2 →  -- α is acute
  Real.cos α = 3 / 5 →
  Real.cos (α + β) = -5 / 13 →
  (Real.cos β = 5 / 13 ∧ Real.sin β = 12 / 13) ∨ (Real.cos β = -5 / 13 ∧ Real.sin β = -12 / 13) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_beta_quadrants_l1070_107016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_black_to_white_area_ratio_l1070_107009

noncomputable def circle_area (r : ℝ) : ℝ := Real.pi * r^2

theorem black_to_white_area_ratio :
  let r₁ := (3 : ℝ)
  let r₂ := (5 : ℝ)
  let r₃ := (7 : ℝ)
  let r₄ := (9 : ℝ)
  let r₅ := (11 : ℝ)
  let black_area := circle_area r₁ + (circle_area r₃ - circle_area r₂) + (circle_area r₅ - circle_area r₄)
  let white_area := (circle_area r₂ - circle_area r₁) + (circle_area r₄ - circle_area r₃)
  black_area / white_area = 73 / 48 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_black_to_white_area_ratio_l1070_107009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_valid_n_l1070_107093

def is_valid (n : ℕ+) : Prop :=
  ∀ m : ℕ, Nat.Coprime m n.val → (m^6 : ℤ) ≡ 1 [ZMOD n.val]

theorem max_valid_n :
  (∃ n : ℕ+, n.val = 504 ∧ is_valid n) ∧
  (∀ n : ℕ+, n.val > 504 → ¬is_valid n) := by
  sorry

#check max_valid_n

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_valid_n_l1070_107093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_l1070_107085

theorem expression_value : 
  (81 : ℝ) ^ (-1/4 : ℝ) + (27 / 8 : ℝ) ^ (-1/3 : ℝ) ^ (1/2 : ℝ) + (1/2 : ℝ) * Real.log 4 - Real.log (1/5 : ℝ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_l1070_107085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_hexagon_area_l1070_107090

/-- The area of a regular hexagon inscribed in a circle, which is itself inscribed in an equilateral triangle with side length a -/
noncomputable def hexagon_area (a : ℝ) : ℝ :=
  (a^2 * Real.sqrt 3) / 8

/-- Theorem stating that the area of the inscribed hexagon is (a^2 * √3) / 8 -/
theorem inscribed_hexagon_area (a : ℝ) (h : a > 0) :
  let r := a * Real.sqrt 3 / 6  -- radius of the inscribed circle
  let s := r  -- side length of the hexagon
  (3 * Real.sqrt 3 / 2) * s^2 = hexagon_area a := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_hexagon_area_l1070_107090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_first_and_last_l1070_107054

def numbers : List ℤ := [-2, 4, 6, 9, 12]

def is_valid_arrangement (arr : List ℤ) : Prop :=
  arr.length = 5 ∧
  arr.toFinset = numbers.toFinset ∧
  arr.get! 0 ≠ 12 ∧
  (arr.get! 1 = 12 ∨ arr.get! 2 = 12 ∨ arr.get! 3 = 12) ∧
  arr.get! 4 ≠ -2 ∧
  (arr.get! 1 = -2 ∨ arr.get! 2 = -2 ∨ arr.get! 3 = -2) ∧
  arr.get! 0 ≠ 6 ∧
  arr.get! 4 ≠ 6

theorem average_of_first_and_last (arr : List ℤ) :
  is_valid_arrangement arr →
  (arr.get! 0 : ℚ) + (arr.get! 4 : ℚ) / 2 = 13 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_first_and_last_l1070_107054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_between_tara_and_uma_l1070_107073

-- Define the type for people in the line
inductive Person : Type
  | Pat | Qasim | Roman | Sam | Tara | Uma

-- Define a type for the position in the line (1 to 6)
def Position : Type := Fin 6

-- Define a function type that represents the arrangement of people in the line
def Arrangement := Person → Position

-- Define the conditions of the problem
def ValidArrangement (arr : Arrangement) : Prop :=
  ∃ (p q r s : Position),
    -- Pat and Qasim have 3 people between them
    (arr Person.Pat).val + 4 = (arr Person.Qasim).val ∨ (arr Person.Qasim).val + 4 = (arr Person.Pat).val ∧
    -- Qasim and Roman have 2 people between them
    (arr Person.Qasim).val + 3 = (arr Person.Roman).val ∨ (arr Person.Roman).val + 3 = (arr Person.Qasim).val ∧
    -- Roman and Sam have 1 person between them
    (arr Person.Roman).val + 2 = (arr Person.Sam).val ∨ (arr Person.Sam).val + 2 = (arr Person.Roman).val ∧
    -- Sam is not at either end of the line
    (arr Person.Sam).val ≠ 0 ∧ (arr Person.Sam).val ≠ 5

-- Theorem statement
theorem two_between_tara_and_uma (arr : Arrangement) 
  (h : ValidArrangement arr) : 
  (arr Person.Tara).val + 3 = (arr Person.Uma).val ∨ (arr Person.Uma).val + 3 = (arr Person.Tara).val := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_between_tara_and_uma_l1070_107073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_digits_divisible_l1070_107049

/-- An infinite sequence of digits -/
def DigitSequence := ℕ → Fin 10

theorem consecutive_digits_divisible
  (seq : DigitSequence)
  (n : ℕ)
  (h : Nat.Coprime n 10) :
  ∃ (start finish : ℕ),
    start ≤ finish ∧
    (∃ k : ℕ+, k * n = (List.range (finish - start + 1)).foldl (λ acc i => acc * 10 + (seq (start + i)).val) 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_digits_divisible_l1070_107049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_calculation_l1070_107060

/-- Simple interest calculation -/
noncomputable def simple_interest (principal rate time : ℝ) : ℝ :=
  principal * rate * time / 100

theorem time_calculation (principal interest rate : ℝ) 
  (h_principal : principal = 400)
  (h_interest : interest = 180)
  (h_rate : rate = 22.5) :
  ∃ time : ℝ, simple_interest principal rate time = interest ∧ time = 2 :=
by
  -- We'll use 2 as our witness for the existential quantifier
  use 2
  constructor
  · -- Prove that simple_interest principal rate 2 = interest
    calc
      simple_interest principal rate 2
        = principal * rate * 2 / 100 := by rfl
      _ = 400 * 22.5 * 2 / 100 := by rw [h_principal, h_rate]
      _ = 180 := by norm_num
    rw [h_interest]
  · -- Prove that time = 2
    rfl

#check time_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_calculation_l1070_107060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_prime_at_1_l1070_107018

-- Define the function f
noncomputable def f (θ : Real) (x : Real) : Real :=
  (Real.sin θ / 3) * x^3 + (Real.sqrt 3 * Real.cos θ / 2) * x^2 + Real.tan θ

-- Define the derivative of f at x = 1
noncomputable def f_prime_at_1 (θ : Real) : Real :=
  Real.sin θ + Real.sqrt 3 * Real.cos θ

-- Theorem statement
theorem range_of_f_prime_at_1 :
  ∀ θ : Real, 0 ≤ θ ∧ θ ≤ 5 * Real.pi / 12 →
  ∃ y : Real, f_prime_at_1 θ = y ∧ Real.sqrt 2 ≤ y ∧ y ≤ 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_prime_at_1_l1070_107018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_equals_total_withdrawal_l1070_107038

/-- Calculates the total amount withdrawn after 8 years of fixed deposits -/
noncomputable def total_withdrawal (a : ℝ) (p : ℝ) : ℝ :=
  (a / p) * ((1 + p)^8 - (1 + p))

/-- Theorem: The sum of the geometric series matches the total withdrawal formula -/
theorem geometric_sum_equals_total_withdrawal (a : ℝ) (p : ℝ) (hp : p ≠ 0) :
  (Finset.range 8).sum (λ i => a * (1 + p)^(8 - i)) = total_withdrawal a p :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_equals_total_withdrawal_l1070_107038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_equals_pi_over_four_l1070_107068

theorem angle_sum_equals_pi_over_four (α β : Real) (h1 : 0 < α ∧ α < Real.pi / 2) 
  (h2 : 0 < β ∧ β < Real.pi / 2) (h3 : Real.sin α = Real.sqrt 5 / 5) (h4 : Real.tan β = 1 / 3) : 
  α + β = Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_equals_pi_over_four_l1070_107068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_diagonals_l1070_107067

-- Define the structure for a parallelogram
structure Parallelogram where
  -- Add necessary fields here
  mk :: -- Constructor

-- Define the properties as functions
def diagonals_bisect_each_other (p : Parallelogram) : Prop :=
  sorry -- Add the actual definition here

def diagonals_are_equal (p : Parallelogram) : Prop :=
  sorry -- Add the actual definition here

-- The main theorem
theorem parallelogram_diagonals (p q : Prop) : 
  (p ↔ (∀ parallelogram : Parallelogram, diagonals_bisect_each_other parallelogram)) →
  (q ↔ (∀ parallelogram : Parallelogram, diagonals_are_equal parallelogram)) →
  ¬p ∨ ¬q :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_diagonals_l1070_107067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_equals_three_l1070_107010

/-- Polar coordinates of a point -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- The area of a triangle given two points in polar coordinates and the pole -/
noncomputable def triangleArea (a b : PolarPoint) : ℝ :=
  (1/2) * |a.r| * |b.r| * Real.sin |a.θ - b.θ|

theorem triangle_area_equals_three : 
  let a : PolarPoint := ⟨3, π/3⟩
  let b : PolarPoint := ⟨-4, 7*π/6⟩
  triangleArea a b = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_equals_three_l1070_107010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_expression_l1070_107081

-- Define the sine function
noncomputable def sine_function (A ω φ : ℝ) (x : ℝ) : ℝ := A * Real.sin (ω * x + φ)

-- Define the theorem
theorem sine_function_expression 
  (A ω φ : ℝ) 
  (h1 : A > 0) 
  (h2 : ω > 0) 
  (h3 : 0 < φ ∧ φ < 2 * Real.pi) 
  (h4 : sine_function A ω φ 0 = 1) 
  (h5 : sine_function A ω φ 2 = Real.sqrt 2) 
  (h6 : ∀ x, x ∈ Set.Icc 0 2 → sine_function A ω φ x ≤ Real.sqrt 2) :
  (A = Real.sqrt 2 ∧ ω = Real.pi / 8 ∧ φ = Real.pi / 4) ∨
  (A = Real.sqrt 2 ∧ ω = 7 * Real.pi / 8 ∧ φ = 3 * Real.pi / 4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_expression_l1070_107081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_reflections_l1070_107002

/-- Represents a line in 2D space --/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Represents a reflection line --/
structure ReflectionLine where
  slope : ℝ
  intercept : ℝ

/-- Calculates the minimum distance between two parallel lines --/
noncomputable def minDistance (l1 l2 : Line) : ℝ :=
  |l2.intercept - l1.intercept| / Real.sqrt (1 + l1.slope^2)

/-- Reflects a line across a given reflection line --/
noncomputable def reflect (l : Line) (r : ReflectionLine) : Line :=
  sorry

/-- Theorem: The minimum distance between images formed by two reflections of a line --/
theorem min_distance_between_reflections 
  (original : Line) 
  (reflection1 reflection2 : ReflectionLine) : 
  ∃ (image1 image2 : Line), 
    image1 = reflect original reflection1 ∧ 
    image2 = reflect (reflect original reflection1) reflection2 ∧
    image1.slope = image2.slope ∧
    minDistance image1 image2 = |image2.intercept - image1.intercept| / Real.sqrt (1 + image1.slope^2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_reflections_l1070_107002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rex_roaming_area_difference_l1070_107043

/-- The radius of the circular playground in feet -/
def playground_radius : ℝ := 20

/-- The length of the rope used to tether Rex in feet -/
def rope_length : ℝ := 30

/-- The distance from the tether point to the fence in Arrangement II in feet -/
def tether_distance : ℝ := 12

/-- The area Rex can roam in Arrangement I -/
noncomputable def area_arrangement_I : ℝ := Real.pi * rope_length^2

/-- The area Rex can roam in Arrangement II -/
noncomputable def area_arrangement_II : ℝ := Real.pi * (rope_length - tether_distance)^2 + (Real.pi * tether_distance^2) / 4

/-- The difference in roaming area between Arrangement I and Arrangement II -/
noncomputable def area_difference : ℝ := area_arrangement_I - area_arrangement_II

theorem rex_roaming_area_difference :
  area_difference = 540 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rex_roaming_area_difference_l1070_107043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_revenue_change_l1070_107056

theorem revenue_change 
  (original_price : ℝ) 
  (original_quantity : ℝ) 
  (price_decrease_percent : ℝ) 
  (ratio_quantity_increase_to_price_decrease : ℝ) 
  (h1 : original_price > 0) 
  (h2 : original_quantity > 0) 
  (h3 : price_decrease_percent = 0.2) 
  (h4 : ratio_quantity_increase_to_price_decrease = 5) : 
  let new_price := original_price * (1 - price_decrease_percent)
  let quantity_increase_percent := price_decrease_percent * ratio_quantity_increase_to_price_decrease
  let new_quantity := original_quantity * (1 + quantity_increase_percent)
  let original_revenue := original_price * original_quantity
  let new_revenue := new_price * new_quantity
  (new_revenue - original_revenue) / original_revenue = 0.6 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_revenue_change_l1070_107056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_f_implies_a_range_l1070_107059

/-- A function f is monotonically increasing on ℝ if for all x, y ∈ ℝ, x < y implies f(x) < f(y) -/
def MonotonicallyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

/-- The function f(x) = x - (1/3) * sin(2x) + a * sin(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  x - (1/3) * Real.sin (2*x) + a * Real.sin x

theorem monotone_f_implies_a_range :
  ∀ a : ℝ, MonotonicallyIncreasing (f a) → a ∈ Set.Icc (-1/3) (1/3) := by
  sorry

#check monotone_f_implies_a_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_f_implies_a_range_l1070_107059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_region_l1070_107014

-- Define the surfaces
def surface1 (x y : ℝ) : Prop := x^2 / 27 + y^2 = 1
def surface2 (y z : ℝ) : Prop := z = y / Real.sqrt 3
def surface3 (z : ℝ) : Prop := z = 0

-- Define the region
def region (x y z : ℝ) : Prop :=
  surface1 x y ∧ surface2 y z ∧ surface3 z ∧ y ≥ 0

-- Define the volume function
noncomputable def volume : ℝ :=
  ∫ x in Set.Icc (-3 * Real.sqrt 3) (3 * Real.sqrt 3),
    ∫ y in Set.Icc 0 (Real.sqrt (1 - x^2 / 27)),
      ∫ z in Set.Icc 0 (y / Real.sqrt 3), 1

-- State the theorem
theorem volume_of_region : volume = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_region_l1070_107014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_100th_term_l1070_107037

def sequenceA (n : ℕ) : ℕ :=
  match n with
  | 0 => 2
  | n + 1 => sequenceA n + 2 * (n + 1)

theorem sequence_100th_term : sequenceA 99 = 9902 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_100th_term_l1070_107037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_C_equation_and_min_area_l1070_107011

noncomputable section

-- Define the circle C
def circle_C (t : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - t/2)^2 + (p.2 - 1)^2 = (t/2)^2 + 1}

-- Define the points
def point_A : ℝ × ℝ := (0, 2)
def point_O : ℝ × ℝ := (0, 0)
def point_D (t : ℝ) : ℝ × ℝ := (t, 0)
def point_B : ℝ × ℝ := (1, 0)

-- Define the line l₂
def line_l2 (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = k * (p.1 - 1)}

-- Define the condition AM ≤ 2BM
def AM_le_2BM (t : ℝ) (x y : ℝ) : Prop :=
  (x - 4/3)^2 + (y + 2/3)^2 ≥ 20/9

-- Define the area of triangle EPQ
noncomputable def area_EPQ (k : ℝ) : ℝ :=
  Real.sqrt ((4 / k^2) - (2 / k) + 4)

theorem circle_C_equation_and_min_area :
  ∃ (t : ℝ), t > 0 ∧
  (∀ x y : ℝ, 2*x + t*y = 2*t → AM_le_2BM t x y) ∧
  (∃ k : ℝ, (k = 0 ∨ k = 4/3) ∧ 
    (∀ p : ℝ × ℝ, p ∈ circle_C 6 ↔ p ∈ line_l2 k)) ∧
  (∀ k : ℝ, k ≠ 0 → area_EPQ k ≥ Real.sqrt 15 / 2) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_C_equation_and_min_area_l1070_107011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_true_propositions_l1070_107021

theorem three_true_propositions : 
  (∃ x : ℝ, x ≤ 0) ∧ 
  (∃ n : ℕ, ¬(Nat.Prime n) ∧ n > 1 ∧ ∀ m : ℕ, 1 < m → m < n → ¬(n % m = 0)) ∧ 
  (∃ x : ℝ, Irrational x ∧ Irrational (x^2)) := by
  sorry

#check three_true_propositions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_true_propositions_l1070_107021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_solution_of_equation_l1070_107006

theorem smallest_solution_of_equation :
  ∃ (x : ℝ), x^4 - 16*x^2 + 63 = 0 ∧ 
  (∀ (y : ℝ), y^4 - 16*y^2 + 63 = 0 → x ≤ y) ∧
  x = -3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_solution_of_equation_l1070_107006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_slope_l1070_107000

-- Define the curve function
noncomputable def f (x : ℝ) : ℝ := x - Real.exp x

-- Define the tangent line function
def tangent_line (k : ℝ) (x : ℝ) : ℝ := k * x

-- Theorem statement
theorem tangent_line_slope (k : ℝ) :
  (∃ x₀ : ℝ, f x₀ = tangent_line k x₀ ∧ 
    (deriv f) x₀ = k) →
  k = 1 - Real.exp 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_slope_l1070_107000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cube_volume_l1070_107031

/-- A pyramid with a square base and equilateral triangle lateral faces -/
structure Pyramid where
  base_side : ℝ
  is_square_base : base_side > 0
  is_equilateral_lateral : True

/-- A cube inscribed in a pyramid -/
structure InscribedCube where
  side : ℝ
  is_positive : side > 0

/-- The volume of a cube -/
def cube_volume (c : InscribedCube) : ℝ := c.side ^ 3

/-- The theorem stating the volume of the inscribed cube in the given pyramid -/
theorem inscribed_cube_volume (p : Pyramid) (c : InscribedCube) 
  (h1 : p.base_side = 2)
  (h2 : c.side > 0)
  (h3 : c.side = Real.sqrt 3 / 2) :
  cube_volume c = 3 * Real.sqrt 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cube_volume_l1070_107031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spinner_probability_divisible_by_five_l1070_107024

def spinner : Finset Nat := {0, 5, 7}

def is_divisible_by_five (n : Nat) : Bool :=
  n % 5 = 0

def three_digit_number (a b c : Nat) : Nat :=
  100 * a + 10 * b + c

theorem spinner_probability_divisible_by_five :
  let outcomes := spinner.product (spinner.product spinner)
  let favorable_outcomes := outcomes.filter (fun abc => 
    let (a, (b, c)) := abc
    is_divisible_by_five (three_digit_number a b c))
  (favorable_outcomes.card : Rat) / outcomes.card = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spinner_probability_divisible_by_five_l1070_107024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kanul_total_amount_l1070_107044

/-- The total amount Kanul had, given his spending on raw materials, machinery, and cash. -/
noncomputable def total_amount (raw_materials : ℝ) (machinery : ℝ) (cash_percentage : ℝ) : ℝ :=
  (raw_materials + machinery) / (1 - cash_percentage)

/-- Theorem stating that given Kanul's spending, his total amount was 5000 / 0.9 -/
theorem kanul_total_amount :
  total_amount 3000 2000 0.1 = 5000 / 0.9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_kanul_total_amount_l1070_107044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_for_integer_solutions_sum_zero_l1070_107095

/-- Given an inequality 2 < 2x - m < 8 where the sum of all integer solutions for x is 0,
    prove that the range of m is -6 < m < -4 -/
theorem m_range_for_integer_solutions_sum_zero 
  (h : ∀ x : ℤ, (2 < 2*x - m ∧ 2*x - m < 8) → x ∈ ({-1, 0, 1} : Set ℤ))
  (sum_zero : (-1) + 0 + 1 = 0) :
  -6 < m ∧ m < -4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_for_integer_solutions_sum_zero_l1070_107095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_revenue_change_l1070_107005

theorem tax_revenue_change 
  (original_tax : ℝ) 
  (original_consumption : ℝ) 
  (tax_reduction_rate : ℝ) 
  (consumption_increase_rate : ℝ) 
  (h1 : tax_reduction_rate = 0.25) 
  (h2 : consumption_increase_rate = 0.20) : 
  (original_tax * (1 - tax_reduction_rate) * (original_consumption * (1 + consumption_increase_rate))) / 
  (original_tax * original_consumption) = 0.90 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_revenue_change_l1070_107005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_problem_l1070_107001

noncomputable def q (x : ℝ) : ℝ := (x^2011 - 1) / (x - 1)

noncomputable def s (x : ℝ) : ℝ := -x^2 - 2*x

theorem remainder_problem : Int.mod (Int.natAbs (Int.floor (s 2011))) 500 = 357 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_problem_l1070_107001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_stays_in_R_l1070_107089

-- Define the region R
def R : Set ℂ := {z | -2 ≤ z.re ∧ z.re ≤ 2 ∧ -1 ≤ z.im ∧ z.im ≤ 1}

-- Define the transformation
noncomputable def transform (z : ℂ) : ℂ := (1/2 + 1/2*Complex.I) * z

-- Theorem statement
theorem transform_stays_in_R : ∀ z ∈ R, transform z ∈ R := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_stays_in_R_l1070_107089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_sum_comparison_l1070_107028

def isPrime (n : ℕ) : Bool :=
  if n ≤ 1 then false
  else (List.range (n - 2)).all (fun k => n % (k + 2) ≠ 0)

def primesInRange (a b : ℕ) : List ℕ :=
  (List.range (b - a + 1)).map (· + a) |>.filter isPrime

def firstNPrimes (n : ℕ) : List ℕ :=
  (List.range 100).filter isPrime |>.take n

theorem prime_sum_comparison :
  (primesInRange 10 20).sum = 60 ∧
  (firstNPrimes 4).sum = 17 ∧
  60 > 17 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_sum_comparison_l1070_107028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_algebra_problem_distribution_l1070_107094

theorem algebra_problem_distribution (total_problems : ℕ) 
  (algebra_percentage : ℚ) (linear_percentage : ℚ) (quadratic_percentage : ℚ) 
  (systems_percentage : ℚ) (polynomial_percentage : ℚ) :
  total_problems = 250 →
  algebra_percentage = 1/2 →
  linear_percentage = 7/20 →
  quadratic_percentage = 1/4 →
  systems_percentage = 1/5 →
  polynomial_percentage = 1/5 →
  ∃ (linear quadratic systems polynomial : ℕ),
    linear = 44 ∧
    quadratic = 31 ∧
    systems = 25 ∧
    polynomial = 25 ∧
    linear + quadratic + systems + polynomial = 
      (algebra_percentage * ↑total_problems).floor := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_algebra_problem_distribution_l1070_107094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_h_functions_l1070_107025

-- Define the concept of an "H function"
def is_h_function (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → x₁ * f x₁ + x₂ * f x₂ ≥ x₁ * f x₂ + x₂ * f x₁

-- Define the given functions
def f₁ (x : ℝ) : ℝ := -x^3 + x + 1
noncomputable def f₂ (x : ℝ) : ℝ := 3*x - 2*(Real.sin x - Real.cos x)
noncomputable def f₃ (x : ℝ) : ℝ := Real.exp x + 1
noncomputable def f₄ (x : ℝ) : ℝ := if x ≥ 1 then Real.log x else 0

-- State the theorem
theorem three_h_functions : 
  (¬is_h_function f₁ ∧ is_h_function f₂ ∧ is_h_function f₃ ∧ is_h_function f₄) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_h_functions_l1070_107025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_distance_l1070_107084

/-- The distance from a point (x, y) to the line 4x - 3y + 25 = 0 --/
noncomputable def distToLine (x y : ℝ) : ℝ :=
  abs (4 * x - 3 * y + 25) / Real.sqrt 25

theorem circle_line_distance (r : ℝ) :
  r > 0 →
  (∃! (p₁ p₂ : ℝ × ℝ), p₁ ≠ p₂ ∧
    p₁.1^2 + p₁.2^2 = r^2 ∧
    p₂.1^2 + p₂.2^2 = r^2 ∧
    distToLine p₁.1 p₁.2 = 1 ∧
    distToLine p₂.1 p₂.2 = 1) →
  4 < r ∧ r < 6 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_distance_l1070_107084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1070_107013

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  m : ℝ

-- Define the conditions
def is_valid_triangle (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.A + t.B + t.C = Real.pi ∧
  t.m > 0

def arithmetic_sequence (t : Triangle) : Prop :=
  t.a + t.c = 2 * t.b

def angle_relation (t : Triangle) : Prop :=
  t.C = 2 * t.A

def side_a_formula (t : Triangle) : Prop :=
  t.a = (4 * t.m^2 + 4 * t.m + 9) / (t.m + 1)

theorem triangle_properties (t : Triangle)
  (h1 : is_valid_triangle t)
  (h2 : arithmetic_sequence t)
  (h3 : angle_relation t)
  (h4 : side_a_formula t) :
  Real.cos t.A = 3/4 ∧
  ∃ (area : ℝ), area = 15 * Real.sqrt 7 ∧
  ∀ (other_area : ℝ), other_area ≥ area :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1070_107013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_asymptote_of_f_l1070_107079

/-- The function f(x) = (6x^2 + 4) / (4x^2 + 3x + 1) -/
noncomputable def f (x : ℝ) : ℝ := (6 * x^2 + 4) / (4 * x^2 + 3 * x + 1)

/-- The horizontal asymptote of f(x) is 1.5 -/
theorem horizontal_asymptote_of_f :
  ∀ ε > 0, ∃ N, ∀ x, x > N → |f x - 1.5| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_asymptote_of_f_l1070_107079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_condition_for_inequality_l1070_107045

theorem necessary_condition_for_inequality :
  (∀ y : ℝ, (1 - y) * (1 + abs y) > 0 → y < 2) ∧ 
  (∃ z : ℝ, z < 2 ∧ (1 - z) * (1 + abs z) ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_condition_for_inequality_l1070_107045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_difference_ratio_l1070_107030

noncomputable def root_difference (a b c : ℝ) : ℝ := Real.sqrt (b^2 - 4*a*c) / a

theorem quadratic_root_difference_ratio (a b : ℝ) : 
  let f₁ := fun x : ℝ => x^2 - x - a
  let f₂ := fun x : ℝ => x^2 + b*x + 2
  let f₃ := fun x : ℝ => 4*x^2 + (b-3)*x - 3*a + 2
  let f₄ := fun x : ℝ => 4*x^2 + (3*b-1)*x + 6 - a
  let A := root_difference 1 (-1) (-a)
  let B := root_difference 1 b 2
  let C := root_difference 4 (b-3) (-3*a + 2)
  let D := root_difference 4 (3*b-1) (6 - a)
  A ≠ 0 → B ≠ 0 → A ≠ B → (C^2 - D^2) / (A^2 - B^2) = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_difference_ratio_l1070_107030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_factorial_ratio_l1070_107071

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem floor_factorial_ratio : 
  ⌊(factorial 2010 + factorial 2007 : ℚ) / (factorial 2009 + factorial 2008)⌋ = 2010 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_factorial_ratio_l1070_107071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_s_l1070_107088

-- Define the function s(x) as noncomputable
noncomputable def s (x : ℝ) : ℝ := 1 / (1 - x)^3

-- State the theorem about the range of s(x)
theorem range_of_s :
  ∀ y : ℝ, (∃ x : ℝ, x ≠ 1 ∧ s x = y) ↔ y > 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_s_l1070_107088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_sqrt_count_l1070_107086

theorem ceiling_sqrt_count : ∃ (S : Finset ℤ), (∀ x ∈ S, ⌈Real.sqrt (x : ℝ)⌉ = 20) ∧ S.card = 39 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_sqrt_count_l1070_107086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shooting_probabilities_l1070_107082

/-- Represents a person with a given shooting accuracy -/
structure Person where
  accuracy : ℝ
  accuracy_valid : 0 ≤ accuracy ∧ accuracy ≤ 1

/-- Calculates the probability of hitting exactly k balloons out of n shots -/
def prob_hit (p : Person) (k n : ℕ) : ℝ :=
  (n.choose k : ℝ) * p.accuracy ^ k * (1 - p.accuracy) ^ (n - k)

/-- The main theorem to prove -/
theorem shooting_probabilities 
  (person_a person_b : Person)
  (ha : person_a.accuracy = 0.7)
  (hb : person_b.accuracy = 0.4)
  (n : ℕ)
  (hn : n = 2) :
  let prob_a_1_b_2 := prob_hit person_a 1 n * prob_hit person_b 2 n
  let prob_equal := (prob_hit person_a 0 n * prob_hit person_b 0 n) +
                    (prob_hit person_a 1 n * prob_hit person_b 1 n) +
                    (prob_hit person_a 2 n * prob_hit person_b 2 n)
  prob_a_1_b_2 = 0.0672 ∧ prob_equal = 0.3124 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shooting_probabilities_l1070_107082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_with_abc_partition_l1070_107020

def T (m : ℕ) : Set ℕ := {n | 2 ≤ n ∧ n ≤ m}

def has_abc (S : Set ℕ) : Prop :=
  ∃ a b c, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a * b = c

theorem smallest_m_with_abc_partition : 
  (∀ m : ℕ, m ≥ 2 → m < 256 → 
    ∃ A B : Set ℕ, A ∪ B = T m ∧ A ∩ B = ∅ ∧ ¬has_abc A ∧ ¬has_abc B) ∧
  (∀ A B : Set ℕ, A ∪ B = T 256 ∧ A ∩ B = ∅ → has_abc A ∨ has_abc B) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_with_abc_partition_l1070_107020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eiffel_tower_scale_factor_l1070_107008

/-- The scale factor between the Eiffel Tower and its model -/
noncomputable def scaleFactor (eiffelTowerHeight : ℝ) (modelHeight : ℝ) : ℝ :=
  (eiffelTowerHeight / modelHeight) / 100

/-- Theorem: The scale factor between the Eiffel Tower and its model is 18 meters per centimeter -/
theorem eiffel_tower_scale_factor :
  let eiffelTowerHeight : ℝ := 324
  let modelHeightCm : ℝ := 18
  scaleFactor eiffelTowerHeight (modelHeightCm / 100) = 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eiffel_tower_scale_factor_l1070_107008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_terminating_decimal_with_eight_l1070_107062

/-- A number is a terminating decimal if it can be expressed as a/b where b is of the form 2^m * 5^n for some non-negative integers m and n. -/
def IsTerminatingDecimal (x : ℚ) : Prop :=
  ∃ (a b : ℕ) (m n : ℕ), x = a / b ∧ b = 2^m * 5^n

/-- Check if a natural number contains the digit 8. -/
def ContainsEight (n : ℕ) : Prop :=
  ∃ (k : ℕ), n / 10^k % 10 = 8

/-- The smallest positive integer n such that 1/n is a terminating decimal and n contains the digit 8 is 8. -/
theorem smallest_terminating_decimal_with_eight :
  (∀ k : ℕ, k < 8 → ¬(IsTerminatingDecimal (1 / (k : ℚ)) ∧ ContainsEight k)) ∧
  (IsTerminatingDecimal (1 / 8) ∧ ContainsEight 8) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_terminating_decimal_with_eight_l1070_107062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_slope_l1070_107065

-- Define the slopes of two lines
noncomputable def slope_l1 (a : ℝ) := a
noncomputable def slope_l2 : ℝ := -1/2

-- Define the condition for parallel lines
def parallel_lines (a : ℝ) : Prop := slope_l1 a = slope_l2

-- Theorem statement
theorem parallel_lines_slope (a : ℝ) : 
  parallel_lines a → a = -1/2 := by
  intro h
  exact h

#check parallel_lines_slope

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_slope_l1070_107065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l1070_107053

/-- The parabola equation -/
def f (x : ℝ) : ℝ := -5 * x^2 - 40 * x - 95

/-- The y-coordinate of the vertex -/
def vertex_y : ℝ := -15

/-- The x-coordinate of one x-intercept -/
noncomputable def x_intercept : ℝ := -4 + Real.sqrt 3

theorem parabola_properties :
  (∃ x : ℝ, ∀ y : ℝ, f y ≤ f x) ∧  -- vertex exists
  (f vertex_y = f (-4)) ∧         -- y-coordinate of vertex
  (f x_intercept = 0)             -- x-intercept
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l1070_107053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_intersection_point_l1070_107070

/-- Represents a circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if a point is tangent to a circle --/
def is_tangent_to (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

theorem tangent_intersection_point (c1 c2 : Circle) (x : ℝ) : 
  c1.center = (0, 0) →
  c1.radius = 3 →
  c2.center = (18, 0) →
  c2.radius = 8 →
  0 < x →
  (∃ (y : ℝ), is_tangent_to c1 (x, y) ∧ is_tangent_to c2 (x, y)) →
  x = 54 / 11 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_intersection_point_l1070_107070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_eight_l1070_107042

theorem cube_root_of_eight : (8 : ℝ) ^ (1/3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_eight_l1070_107042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wrapping_time_proof_l1070_107015

/-- Represents the time Emma and Troy worked together -/
noncomputable def t : ℝ := 2

/-- Emma's rate of wrapping presents (portion of task per hour) -/
noncomputable def emma_rate : ℝ := 1/6

/-- Troy's rate of wrapping presents (portion of task per hour) -/
noncomputable def troy_rate : ℝ := 1/8

/-- The combined rate of Emma and Troy working together -/
noncomputable def combined_rate : ℝ := emma_rate + troy_rate

theorem wrapping_time_proof :
  t * combined_rate + 2.5 * emma_rate = 1 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wrapping_time_proof_l1070_107015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_enclosing_circle_radius_is_half_length_smallest_enclosing_circle_radius_closed_is_quarter_l1070_107051

/-- A polygonal chain in a plane -/
structure PolygonalChain where
  vertices : List (ℝ × ℝ)
  length : ℝ

/-- A closed polygonal chain in a plane -/
structure ClosedPolygonalChain where
  vertices : List (ℝ × ℝ)
  length : ℝ
  is_closed : vertices.head? = vertices.getLast?

/-- The radius of the smallest enclosing circle for a polygonal chain -/
noncomputable def smallest_enclosing_circle_radius (chain : PolygonalChain) : ℝ := sorry

/-- The radius of the smallest enclosing circle for a closed polygonal chain -/
noncomputable def smallest_enclosing_circle_radius_closed (chain : ClosedPolygonalChain) : ℝ := sorry

/-- Theorem: The radius of the smallest circle that can enclose any polygonal chain of length l is l/2 -/
theorem smallest_enclosing_circle_radius_is_half_length (chain : PolygonalChain) :
  smallest_enclosing_circle_radius chain = chain.length / 2 := by sorry

/-- Theorem: The radius of the smallest circle that can enclose any closed polygonal chain of length 1 is 1/4 -/
theorem smallest_enclosing_circle_radius_closed_is_quarter (chain : ClosedPolygonalChain) 
  (h : chain.length = 1) :
  smallest_enclosing_circle_radius_closed chain = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_enclosing_circle_radius_is_half_length_smallest_enclosing_circle_radius_closed_is_quarter_l1070_107051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_equals_one_l1070_107098

theorem tan_alpha_equals_one (α : Real) 
  (h1 : Real.sin α = Real.cos α) 
  (h2 : α ∈ Set.Icc Real.pi (3 * Real.pi / 2)) : 
  Real.tan α = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_equals_one_l1070_107098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_house_rent_calculation_l1070_107035

/-- Calculates the monthly rent for a house given the purchase price, repair percentage, annual taxes, and desired return on investment. -/
noncomputable def calculate_monthly_rent (purchase_price : ℝ) (repair_percentage : ℝ) (annual_taxes : ℝ) (roi_percentage : ℝ) : ℝ :=
  let annual_roi := purchase_price * roi_percentage
  let total_annual_income := annual_roi + annual_taxes
  let monthly_income := total_annual_income / 12
  monthly_income / (1 - repair_percentage)

/-- Theorem stating that the monthly rent for the given conditions is approximately $83.33 -/
theorem house_rent_calculation :
  let purchase_price : ℝ := 10000
  let repair_percentage : ℝ := 0.125
  let annual_taxes : ℝ := 325
  let roi_percentage : ℝ := 0.055
  abs (calculate_monthly_rent purchase_price repair_percentage annual_taxes roi_percentage - 83.33) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_house_rent_calculation_l1070_107035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcf_of_m_and_n_l1070_107047

theorem gcf_of_m_and_n (m n : ℕ) 
  (hm : (Nat.factors m).toFinset.card = 4)
  (hn : (Nat.factors n).toFinset.card = 3)
  (hgcf : ∃ g : ℕ, g = Nat.gcd m n ∧ g > 1)
  (hmn : (Nat.factors (m * n)).toFinset.card = 5) :
  ∃ p : ℕ, Nat.Prime p ∧ Nat.gcd m n = p :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcf_of_m_and_n_l1070_107047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solutions_l1070_107019

def equation1 (a x y : ℤ) : Prop :=
  y = |a - 3| * x + |x + 3| + 3 * |x + 1|

noncomputable def equation2 (a x y : ℤ) : Prop :=
  (2 : ℝ)^(2 - y) * Real.log ((x + |a + 2*x|)^2 - 6*(x + 1 + |a + 2*x|) + 16) / Real.log 3 / 2 + 
  (2 : ℝ)^(x + |a + 2*x|) * Real.log (y^2 + 1) / Real.log 3 = 0

def equation3 (a x : ℤ) : Prop :=
  x + |a + 2*x| ≤ 3

def has_unique_solution (a : ℤ) : Prop :=
  ∃! x y : ℤ, equation1 a x y ∧ equation2 a x y ∧ equation3 a x

theorem system_solutions :
  (has_unique_solution (-2) ∧ 
   has_unique_solution (-1) ∧ 
   has_unique_solution 1 ∧ 
   has_unique_solution 3 ∧ 
   has_unique_solution 4 ∧ 
   has_unique_solution 6) ∧
  (∀ a : ℤ, has_unique_solution a → 
    a = -2 ∨ a = -1 ∨ a = 1 ∨ a = 3 ∨ a = 4 ∨ a = 6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solutions_l1070_107019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_config_count_l1070_107040

/-- Represents a single dial in the lock -/
def Dial := Fin 8

/-- Represents the entire lock configuration -/
def LockConfig := Fin 4 → Dial

/-- Checks if two numbers are consecutive -/
def isConsecutive (a b : Dial) : Prop :=
  (a.val + 1 = b.val) ∨ (b.val + 1 = a.val)

/-- Checks if a lock configuration is valid -/
def isValidConfig (config : LockConfig) : Prop :=
  ∀ i : Fin 3, ¬ isConsecutive (config i) (config (Fin.succ i))

/-- The set of all valid lock configurations -/
def ValidConfigs : Set LockConfig :=
  { config | isValidConfig config }

/-- Proof that ValidConfigs is finite -/
instance : Fintype ValidConfigs := by
  sorry

/-- The main theorem: there are 1728 valid lock configurations -/
theorem valid_config_count : Fintype.card ValidConfigs = 1728 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_config_count_l1070_107040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_path_ratio_l1070_107055

theorem rectangle_path_ratio (n m k : ℕ) (h : n = k * m) :
  (Nat.choose (n + m - 1) m) = k * (Nat.choose (n + m - 1) (m - 1)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_path_ratio_l1070_107055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_sum_of_sum_of_digits_eq_five_l1070_107083

/-- Sum of digits function -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Theorem: There are exactly 12 two-digit positive integers x such that sumOfDigits(sumOfDigits(x)) = 5 -/
theorem two_digit_sum_of_sum_of_digits_eq_five :
  (Finset.filter (fun x => sumOfDigits (sumOfDigits x) = 5) (Finset.range 90 \ Finset.range 10)).card = 12 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_sum_of_sum_of_digits_eq_five_l1070_107083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_air_conditioner_price_X_l1070_107017

/-- The price of an air conditioner offered by Company X -/
def price_X : ℝ := 575

/-- The price of an air conditioner offered by Company Y -/
def price_Y : ℝ := 530

/-- The total amount saved by dealing with the company that offers the lower total charge -/
def savings : ℝ := 41.60

/-- The total charge for Company X -/
def total_charge_X (p i : ℝ) : ℝ := price_X + (p * price_X) + i

/-- The total charge for Company Y -/
def total_charge_Y (p i : ℝ) : ℝ := price_Y + (p * price_Y) + i

/-- Theorem stating that the price of the air conditioner offered by Company X is $575 -/
theorem air_conditioner_price_X : 
  ∀ (p i : ℝ), total_charge_X p i - total_charge_Y p i = savings → price_X = 575 := by
  intro p i h
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_air_conditioner_price_X_l1070_107017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1070_107046

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3) + Real.cos (2 * x + Real.pi / 6) + 2 * Real.sin x * Real.cos x

theorem f_properties :
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ ∀ S, 0 < S → S < T → ∃ y, f (y + S) ≠ f y) ∧
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≤ 2) ∧
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≥ -Real.sqrt 3) ∧
  (∃ x ∈ Set.Icc 0 (Real.pi / 2), f x = 2) ∧
  (∃ x ∈ Set.Icc 0 (Real.pi / 2), f x = -Real.sqrt 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1070_107046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_real_sum_zeroes_l1070_107048

noncomputable def zeroes (n : ℕ) : List (ℂ) := 
  List.map (λ k => 8 * (Complex.exp (2 * Real.pi * Complex.I * (k : ℝ) / 6 : ℂ))) (List.range n)

noncomputable def w_sum (zs : List ℂ) : ℂ :=
  zs.foldl (λ acc z => acc + if (z.re ≥ 0) then z else (Complex.I * z)) 0

theorem max_real_sum_zeroes : 
  let zs := zeroes 6
  Complex.re (w_sum zs) = 8 + 8 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_real_sum_zeroes_l1070_107048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_twelve_draws_for_ten_red_l1070_107036

def total_balls : ℕ := 8
def red_balls : ℕ := 3
def white_balls : ℕ := 5
def target_red_balls : ℕ := 10
def total_draws : ℕ := 12

theorem probability_of_twelve_draws_for_ten_red (total_balls red_balls white_balls target_red_balls total_draws : ℕ) :
  total_balls = red_balls + white_balls →
  target_red_balls = 10 →
  total_draws = 12 →
  (Nat.choose 11 9 : ℚ) * (3 / 8 : ℚ) ^ 9 * (5 / 8 : ℚ) ^ 2 * (3 / 8 : ℚ) =
  (Nat.choose 11 9 * 3^10 * 5^2 : ℚ) / (8^12 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_twelve_draws_for_ten_red_l1070_107036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_irrational_sum_subset_l1070_107039

theorem irrational_sum_subset (S : Finset ℝ) (h_card : S.card = 5) 
  (h_irrational : ∀ x, x ∈ S → Irrational x) : 
  ∃ T : Finset ℝ, T ⊆ S ∧ T.card = 3 ∧ ∀ x y, x ∈ T → y ∈ T → x ≠ y → Irrational (x + y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_irrational_sum_subset_l1070_107039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l1070_107007

/-- The set of digits excluding 5 and 8 -/
def ValidDigits : Finset Nat := {0, 1, 2, 3, 4, 6, 7, 9}

/-- The count of valid digits for the first position (excluding 0) -/
def FirstDigitCount : Nat := (ValidDigits.filter (· ≠ 0)).card

/-- The count of valid digits for the second and third positions -/
def OtherDigitCount : Nat := ValidDigits.card

/-- A three-digit number with no 5's and 8's -/
structure ValidNumber where
  hundreds : Nat
  tens : Nat
  units : Nat
  hundreds_valid : hundreds ∈ ValidDigits ∧ hundreds ≠ 0
  tens_valid : tens ∈ ValidDigits
  units_valid : units ∈ ValidDigits

instance : Fintype ValidNumber := by sorry

theorem count_valid_numbers : Fintype.card ValidNumber = 448 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l1070_107007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_QXY_l1070_107080

/-- Given a rectangle PQRS with length 8 cm and width 6 cm, if the diagonal PR is divided into four
    equal segments by points X and Y, then the area of triangle QXY is 6 cm². -/
theorem area_of_triangle_QXY (P Q R S X Y : ℝ × ℝ) : 
  let length : ℝ := 8
  let width : ℝ := 6
  let diagonal_length : ℝ := Real.sqrt (length ^ 2 + width ^ 2)
  let segment_length : ℝ := diagonal_length / 4
  -- Rectangle PQRS properties
  (Q.1 - P.1 = length ∧ Q.2 = P.2) →
  (S.1 - P.1 = 0 ∧ S.2 - P.2 = width) →
  (R.1 = S.1 + length ∧ R.2 = S.2) →
  -- X and Y divide PR into four equal segments
  (X.1 - P.1 = segment_length * (R.1 - P.1) / diagonal_length ∧
   X.2 - P.2 = segment_length * (R.2 - P.2) / diagonal_length) →
  (Y.1 - P.1 = 3 * segment_length * (R.1 - P.1) / diagonal_length ∧
   Y.2 - P.2 = 3 * segment_length * (R.2 - P.2) / diagonal_length) →
  -- The area of triangle QXY is 6 cm²
  abs ((Q.1 * (Y.2 - X.2) + X.1 * (Q.2 - Y.2) + Y.1 * (X.2 - Q.2)) / 2) = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_QXY_l1070_107080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_a_part_b_l1070_107096

/-- A directed graph with 101 vertices -/
structure DirectedGraph :=
  (V : Finset Nat)
  (E : Finset (Nat × Nat))
  (vertex_count : V.card = 101)

/-- The in-degree of a vertex in a directed graph -/
def in_degree (G : DirectedGraph) (v : Nat) : Nat :=
  (G.E.filter (fun e => e.2 = v)).card

/-- The out-degree of a vertex in a directed graph -/
def out_degree (G : DirectedGraph) (v : Nat) : Nat :=
  (G.E.filter (fun e => e.1 = v)).card

/-- A path in a directed graph -/
def path (G : DirectedGraph) (start finish : Nat) (length : Nat) : Prop :=
  ∃ (p : List Nat), p.length = length + 1 ∧ p.head? = some start ∧ p.getLast? = some finish ∧
    ∀ i : Fin length, ((p.get? i).get!, (p.get? (i + 1)).get!) ∈ G.E

/-- Theorem for part (a) -/
theorem part_a (G : DirectedGraph) 
  (h1 : ∀ v, v ∈ G.V → in_degree G v = 50 ∧ out_degree G v = 50) :
  ∀ u v, u ∈ G.V → v ∈ G.V → ∃ length, length ≤ 2 ∧ path G u v length := by
  sorry

/-- Theorem for part (b) -/
theorem part_b (G : DirectedGraph) 
  (h1 : ∀ v, v ∈ G.V → in_degree G v = 40 ∧ out_degree G v = 40) :
  ∀ u v, u ∈ G.V → v ∈ G.V → ∃ length, length ≤ 3 ∧ path G u v length := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_a_part_b_l1070_107096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_A_l1070_107074

def U : Set ℕ := {1, 2, 3, 4, 5}

def A : Set ℕ := {x ∈ U | (x : ℤ) - 3 < 2 ∧ 3 - (x : ℤ) < 2}

theorem complement_of_A (x : ℕ) : x ∈ (U \ A) ↔ x = 1 ∨ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_A_l1070_107074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equality_l1070_107052

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := x + 1 + Real.log x

-- Define the quadratic function
def g (a x : ℝ) : ℝ := a * x^2 + (a + 2) * x + 1

-- Define the tangent line at (1, 2)
def tangent_line (x : ℝ) : ℝ := 2 * x

theorem tangent_line_equality (a : ℝ) : 
  (∀ x, tangent_line x = g a x) → a = 4 := by
  sorry

#check tangent_line_equality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equality_l1070_107052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_factorial_representation_of_1729_l1070_107099

theorem min_factorial_representation_of_1729 :
  ∃ (a₁ a₂ b₁ b₂ : ℕ),
    (a₁ > 0) ∧ (a₂ > 0) ∧ (b₁ > 0) ∧ (b₂ > 0) ∧
    (a₁ ≥ a₂) ∧ (b₁ ≥ b₂) ∧
    (1729 * Nat.factorial b₁ * Nat.factorial b₂ = Nat.factorial a₁ * Nat.factorial a₂) ∧
    (∀ (c₁ c₂ d₁ d₂ : ℕ),
      (c₁ > 0) ∧ (c₂ > 0) ∧ (d₁ > 0) ∧ (d₂ > 0) ∧
      (c₁ ≥ c₂) ∧ (d₁ ≥ d₂) ∧
      (1729 * Nat.factorial d₁ * Nat.factorial d₂ = Nat.factorial c₁ * Nat.factorial c₂) →
      (a₁ + a₂ + b₁ + b₂ ≤ c₁ + c₂ + d₁ + d₂)) ∧
    (Int.natAbs (a₁ - b₁) = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_factorial_representation_of_1729_l1070_107099
