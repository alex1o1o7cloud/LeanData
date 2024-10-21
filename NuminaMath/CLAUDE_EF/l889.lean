import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l889_88983

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x - 1 / x

theorem f_properties (a : ℝ) :
  -- Part I: Tangent line perpendicular to x + 2y = 0 at (1, f(1))
  (∃ (a : ℝ), (deriv (f a) 1 = 2) ∧ (a = 1)) ∧
  -- Part II: Monotonicity intervals
  ((a ≥ 0 → ∀ x > 0, deriv (f a) x > 0) ∧
   (a < 0 → ∃ x > 0, deriv (f a) x = 0 ∧
     (∀ y ∈ Set.Ioo 0 x, deriv (f a) y > 0) ∧
     (∀ y > x, deriv (f a) y < 0))) ∧
  -- Part III: Inequality when a = 1 and x ≥ 2
  (∀ x ≥ 2, f 1 (x - 1) ≤ 2 * x - 5) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l889_88983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_min_length_l889_88999

/-- The ellipse C with right focus at (√2, 0) -/
def ellipse_C (x y : ℝ) : Prop := x^2 / 4 + y^2 / 2 = 1

/-- The line passing through the right focus and endpoint of minor axis -/
def focus_minor_axis_line (x y : ℝ) : Prop := y = x - Real.sqrt 2

/-- The line y = 2x -/
def line_y_2x (x y : ℝ) : Prop := y = 2 * x

/-- Point A on the line y = 2x -/
def point_A : ℝ × ℝ → Prop := λ p => line_y_2x p.1 p.2

/-- Point B on the ellipse C -/
def point_B : ℝ × ℝ → Prop := λ p => ellipse_C p.1 p.2

/-- OA perpendicular to OB -/
def OA_perp_OB (a b : ℝ × ℝ) : Prop := a.1 * b.1 + a.2 * b.2 = 0

theorem ellipse_and_min_length :
  (∀ x y, focus_minor_axis_line x y → y / x = 1) →
  (∀ a b, point_A a ∧ point_B b ∧ OA_perp_OB a b →
    (a.1 - b.1)^2 + (a.2 - b.2)^2 ≥ 8) ∧
  (∃ a b, point_A a ∧ point_B b ∧ OA_perp_OB a b ∧
    (a.1 - b.1)^2 + (a.2 - b.2)^2 = 8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_min_length_l889_88999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_min_value_f_three_zeros_l889_88941

-- Define the functions f and g
noncomputable def f (a x : ℝ) : ℝ := Real.log x + (4 * a) / (x + a^2) - 2
noncomputable def g (a : ℝ) : ℝ := f a (a^2)

-- Theorem for the minimum value of g
theorem g_min_value (a : ℝ) (h : a > 0) :
  ∃ (min : ℝ), min = 0 ∧ ∀ x > 0, g x ≥ min := by
  sorry

-- Theorem for the range of a when f has three distinct zero points
theorem f_three_zeros (a : ℝ) :
  (∃ x y z : ℝ, x < y ∧ y < z ∧ f a x = 0 ∧ f a y = 0 ∧ f a z = 0) ↔ 0 < a ∧ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_min_value_f_three_zeros_l889_88941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_l889_88908

/-- The radius of the inscribed circle in a triangle with two sides of length 8 and one side of length 10 -/
theorem inscribed_circle_radius (A B C : EuclideanSpace ℝ (Fin 2)) 
  (h1 : dist A B = 8) (h2 : dist B C = 8) (h3 : dist A C = 10) :
  let s := (dist A B + dist B C + dist A C) / 2
  let area := Real.sqrt (s * (s - dist A B) * (s - dist B C) * (s - dist A C))
  let r := area / s
  r = 5 * Real.sqrt 39 / 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_l889_88908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_characterization_l889_88979

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then -x^2 + 2*x else Real.log (x + 1)

theorem f_inequality_characterization :
  (∀ x, |f x| ≥ x * a) ↔ a ∈ Set.Icc (-2 : ℝ) 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_characterization_l889_88979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_hexagonal_pyramid_volume_specific_l889_88942

/-- The volume of a regular hexagonal pyramid -/
noncomputable def regular_hexagonal_pyramid_volume (base_edge : ℝ) (side_edge : ℝ) : ℝ :=
  let base_area := 3 * base_edge^2 * Real.sqrt 3 / 2
  let height := Real.sqrt (side_edge^2 - base_edge^2)
  (1 / 3) * base_area * height

/-- Theorem: The volume of a regular hexagonal pyramid with base edge length 3 and side edge length 5 is 18√3 -/
theorem regular_hexagonal_pyramid_volume_specific : 
  regular_hexagonal_pyramid_volume 3 5 = 18 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_hexagonal_pyramid_volume_specific_l889_88942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_recipe_count_l889_88923

/-- Calculates the number of full recipes needed for a school play -/
def cookies_for_school_play (total_students : ℕ) (attendance_rate : ℚ) 
  (cookies_per_student : ℕ) (cookies_per_recipe : ℕ) : ℕ :=
  let expected_students := (total_students : ℚ) * attendance_rate
  let total_cookies_needed := expected_students * (cookies_per_student : ℚ)
  let recipes_needed := total_cookies_needed / (cookies_per_recipe : ℚ)
  (Int.ceil recipes_needed).toNat

/-- Proves that the calculation for the given problem yields 11 recipes -/
theorem correct_recipe_count : 
  cookies_for_school_play 108 (3/5) 3 18 = 11 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_recipe_count_l889_88923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_dot_product_result_l889_88964

def unit_vector (v : ℝ × ℝ) : Prop := v.1^2 + v.2^2 = 1

theorem vector_dot_product_result 
  (e₁ e₂ : ℝ × ℝ) 
  (h₁ : unit_vector e₁) 
  (h₂ : unit_vector e₂) 
  (h₃ : e₁.1 * e₂.1 + e₁.2 * e₂.2 = 1/2) : 
  (e₁.1 - e₂.1) * (-3*e₁.1 + 2*e₂.1) + (e₁.2 - e₂.2) * (-3*e₁.2 + 2*e₂.2) = -5/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_dot_product_result_l889_88964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_cross_bridge_time_l889_88915

/-- Calculates the time (in seconds) required for a train to cross a bridge -/
noncomputable def timeToCrossBridge (trainLength : ℝ) (bridgeLength : ℝ) (trainSpeed : ℝ) : ℝ :=
  let totalDistance : ℝ := trainLength + bridgeLength
  let speedInMetersPerSecond : ℝ := trainSpeed * 1000 / 3600
  totalDistance / speedInMetersPerSecond

/-- Proves that a train of length 165 meters traveling at 36 kmph takes 82.5 seconds to cross a bridge of length 660 meters -/
theorem train_cross_bridge_time :
  timeToCrossBridge 165 660 36 = 82.5 := by
  sorry

-- Use #eval only for computable functions
def approxTimeToCrossBridge (trainLength : Float) (bridgeLength : Float) (trainSpeed : Float) : Float :=
  let totalDistance : Float := trainLength + bridgeLength
  let speedInMetersPerSecond : Float := trainSpeed * 1000 / 3600
  totalDistance / speedInMetersPerSecond

#eval approxTimeToCrossBridge 165 660 36

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_cross_bridge_time_l889_88915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l889_88962

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.sin (Real.pi / 4 + x))^2 - Real.sqrt 3 * Real.cos (2 * x)

theorem f_properties :
  -- Smallest positive period is π
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x ∧ ∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  -- Monotonically decreasing interval
  (∀ (k : ℤ), ∀ (x y : ℝ), 5 * Real.pi / 12 + k * Real.pi ≤ x ∧ x < y ∧ y ≤ 11 * Real.pi / 12 + k * Real.pi → f y < f x) ∧
  -- Maximum value on [π/4, π/2]
  (∀ (x : ℝ), Real.pi / 4 ≤ x ∧ x ≤ Real.pi / 2 → f x ≤ 3) ∧
  (∃ (x : ℝ), Real.pi / 4 ≤ x ∧ x ≤ Real.pi / 2 ∧ f x = 3) ∧
  -- Minimum value on [π/4, π/2]
  (∀ (x : ℝ), Real.pi / 4 ≤ x ∧ x ≤ Real.pi / 2 → f x ≥ 2) ∧
  (∃ (x : ℝ), Real.pi / 4 ≤ x ∧ x ≤ Real.pi / 2 ∧ f x = 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l889_88962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_twelfth_square_side_length_l889_88945

def spiral_sequence : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | 2 => 1
  | 3 => 3
  | 4 => 4
  | 5 => 7
  | n+6 => spiral_sequence (n+4) + spiral_sequence (n+5)

theorem twelfth_square_side_length :
  spiral_sequence 11 = 123 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_twelfth_square_side_length_l889_88945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l889_88976

theorem cos_alpha_value (α β γ : ℝ) : 
  (β = 2 * α ∧ γ = 2 * β) →  -- geometric progression condition
  (∃ r : ℝ, Real.sin β = r * Real.sin α ∧ Real.sin γ = r * Real.sin β) →  -- sin values form geometric progression
  Real.cos α = -1/2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l889_88976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sin_cos_ratio_l889_88950

theorem max_sin_cos_ratio (α β γ : Real) 
  (h_acute : 0 < α ∧ α < Real.pi/2 ∧ 0 < β ∧ β < Real.pi/2 ∧ 0 < γ ∧ γ < Real.pi/2)
  (h_sin_sum : Real.sin α ^ 2 + Real.sin β ^ 2 + Real.sin γ ^ 2 = 1) :
  (Real.sin α + Real.sin β + Real.sin γ) / (Real.cos α + Real.cos β + Real.cos γ) ≤ Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sin_cos_ratio_l889_88950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coeff_x_cubed_value_l889_88948

/-- Represents the polynomial (ax - 3/(4x) + 2/3)(x - 3/x)^6 --/
noncomputable def P (a x : ℝ) : ℝ := (a * x - 3 / (4 * x) + 2 / 3) * (x - 3 / x)^6

/-- The sum of coefficients of P when x = 1 --/
noncomputable def sum_of_coefficients (a : ℝ) : ℝ := P a 1

/-- The coefficient of x^3 in the expansion of P --/
noncomputable def coeff_x_cubed (a : ℝ) : ℝ := sorry

theorem coeff_x_cubed_value :
  ∃ a : ℝ, sum_of_coefficients a = 16 ∧ coeff_x_cubed a = 117 / 2 := by
  sorry

#eval "Theorem stated successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coeff_x_cubed_value_l889_88948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_not_in_fourth_quadrant_power_function_decreasing_in_first_quadrant_l889_88970

-- Define a power function
noncomputable def power_function (n : ℝ) : ℝ → ℝ := λ x => x^n

-- Statement 1: The graph of a power function cannot be in the fourth quadrant
theorem power_function_not_in_fourth_quadrant (n : ℝ) :
  ∀ x > 0, power_function n x > 0 := by
  sorry

-- Statement 2: When n<0, in the first quadrant, the function decreases as x increases
theorem power_function_decreasing_in_first_quadrant (n : ℝ) (hn : n < 0) :
  ∀ x₁ x₂, 0 < x₁ → x₁ < x₂ → power_function n x₂ < power_function n x₁ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_not_in_fourth_quadrant_power_function_decreasing_in_first_quadrant_l889_88970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_local_min_and_inequality_l889_88912

def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x

theorem local_min_and_inequality (a : ℝ) : 
  (∃ δ > 0, ∀ x ∈ Set.Ioo (1 - δ) (1 + δ), f a x ≥ f a 1) →
  (a = 3 ∧ ∀ x > 0, max (-2 * f a x / x^2) ((3 * x^2 - a) / x) ≥ Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_local_min_and_inequality_l889_88912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l889_88922

theorem triangle_inequality (a b c : ℝ) (A B C : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  a = b * Real.sin A / Real.sin B →
  b = c * Real.sin B / Real.sin C →
  c = a * Real.sin C / Real.sin A →
  2 * b * Real.cos (C / 2) + 2 * c * Real.cos (B / 2) > a + b + c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l889_88922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_a_value_l889_88965

/-- An ellipse with foci on the x-axis -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_a_pos : 0 < a
  h_b_pos : 0 < b
  h_a_ge_b : a ≥ b

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (e.a^2 - e.b^2) / e.a

theorem ellipse_a_value (e : Ellipse) 
  (h_eq : e.b^2 = 5)
  (h_ecc : eccentricity e = 2/3) : 
  e.a = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_a_value_l889_88965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_arrangements_l889_88901

theorem student_arrangements (n : ℕ) (m : ℕ) : 
  n = 7 → m = 2 → (Nat.factorial (n - m + 1)) * (Nat.factorial m) = 1440 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_arrangements_l889_88901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_approximations_are_close_l889_88992

/-- Linear approximation of the fourth root of 17 -/
noncomputable def fourth_root_17_approx : ℝ := 2 + 1 / 32

/-- Linear approximation of arctangent of 0.98 -/
noncomputable def arctg_098_approx : ℝ := Real.pi / 4 - 0.01

/-- Linear approximation of sine of 29 degrees -/
noncomputable def sin_29_deg_approx : ℝ := 1 / 2 - (Real.sqrt 3 * Real.pi) / 360

/-- Theorem stating that the linear approximations are close to the actual values -/
theorem linear_approximations_are_close :
  (|fourth_root_17_approx - 17^(1/4)| < 0.001) ∧
  (|arctg_098_approx - Real.arctan 0.98| < 0.001) ∧
  (|sin_29_deg_approx - Real.sin (29 * Real.pi / 180)| < 0.001) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_approximations_are_close_l889_88992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_is_scalene_triangle_l889_88977

-- Define the lines
noncomputable def line1 (x : ℝ) : ℝ := 3 * x + 2
noncomputable def line2 (x : ℝ) : ℝ := -4 * x + 2
noncomputable def line3 : ℝ := -2

-- Define the intersection points
noncomputable def point1 : ℝ × ℝ := (0, 2)
noncomputable def point2 : ℝ × ℝ := (-4/3, -2)
noncomputable def point3 : ℝ × ℝ := (1, -2)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem polygon_is_scalene_triangle :
  let side1 := distance point1 point2
  let side2 := distance point1 point3
  let side3 := distance point2 point3
  (side1 ≠ side2) ∧ (side1 ≠ side3) ∧ (side2 ≠ side3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_is_scalene_triangle_l889_88977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l889_88929

-- Define the function f as noncomputable
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.sqrt (2 * Real.log (x + 1) + x - m)

-- State the theorem
theorem range_of_m (m : ℝ) :
  (∃ x₀ y₀ : ℝ, y₀ = Real.cos x₀ ∧ f m (f m y₀) = y₀) →
  0 ≤ m ∧ m ≤ 2 * Real.log 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l889_88929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_sum_divisible_by_6_l889_88930

def S : Finset ℕ := {1, 2, 3, 4}

def is_sum_divisible_by_6 (a b c : ℕ) : Bool :=
  (a + b + c) % 6 = 0

def valid_selection (a b c : ℕ) : Bool :=
  a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c

theorem probability_sum_divisible_by_6 :
  (Finset.filter (λ (t : ℕ × ℕ × ℕ) => valid_selection t.1 t.2.1 t.2.2 ∧ 
    is_sum_divisible_by_6 t.1 t.2.1 t.2.2) (S.product (S.product S))).card /
  (Finset.filter (λ (t : ℕ × ℕ × ℕ) => valid_selection t.1 t.2.1 t.2.2) (S.product (S.product S))).card
  = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_sum_divisible_by_6_l889_88930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_length_60N_on_20cm_globe_l889_88931

/-- The length of a parallel of latitude on a globe -/
noncomputable def parallel_length (R : ℝ) (latitude : ℝ) : ℝ :=
  2 * Real.pi * R * Real.cos latitude

/-- Theorem: On a globe with radius 20 cm, the length of the parallel of latitude at 60°N is 20π cm -/
theorem parallel_length_60N_on_20cm_globe : 
  parallel_length 20 (Real.pi / 3) = 20 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_length_60N_on_20cm_globe_l889_88931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_similarity_l889_88997

/-- Predicate for points being on the same side of a line -/
def OnSameSide (P Q R : EuclideanSpace ℝ (Fin 2)) (l : Set (EuclideanSpace ℝ (Fin 2))) : Prop := sorry

/-- Predicate for triangle similarity -/
def SimilarTriangles (P Q R S T U : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

/-- Given points X, Y, Z on one side of line AB, with XAB similar to BYA and ABZ, 
    prove XYZ is similar to XAB -/
theorem triangle_similarity 
  (A B X Y Z : EuclideanSpace ℝ (Fin 2)) 
  (l : Set (EuclideanSpace ℝ (Fin 2))) :
  OnSameSide X Y Z l →
  A ∈ l →
  B ∈ l →
  SimilarTriangles X A B B Y A →
  SimilarTriangles X A B A B Z →
  SimilarTriangles X Y Z X A B :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_similarity_l889_88997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_problem_l889_88944

/-- The sum of a geometric series with first term a, common ratio r, and n terms -/
noncomputable def geometricSum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

/-- The problem statement -/
theorem geometric_sum_problem :
  let a : ℝ := 1/4
  let r : ℝ := 1/4
  let n : ℕ := 5
  geometricSum a r n = 1023/3072 := by
  sorry

#eval (1023 : ℚ) / 3072

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_problem_l889_88944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_l889_88920

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log (1 / x)
  else if x < 0 then 1 / x
  else 0  -- Undefined for x = 0, so we assign an arbitrary value

theorem solution_set_f (x : ℝ) :
  f x > -1 ↔ x ∈ Set.union (Set.Ioi (-1)) (Set.Ioo 0 (Real.exp 1)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_l889_88920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wedge_volume_problem_l889_88993

noncomputable section

/-- The volume of a wedge cut from a cylindrical log --/
def wedge_volume (d : ℝ) (θ : ℝ) : ℝ :=
  (1 / 2) * (θ / (2 * Real.pi)) * Real.pi * (d / 2)^2 * d

/-- The problem statement --/
theorem wedge_volume_problem :
  ∃ (m : ℕ), wedge_volume 20 (Real.pi / 3) = m * Real.pi ∧ m = 333 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wedge_volume_problem_l889_88993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_3pi_8_l889_88994

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ - Real.pi / 6)

theorem f_value_at_3pi_8 (ω φ : ℝ) 
  (h1 : ω > 0)
  (h2 : Real.pi / 2 < φ)
  (h3 : φ < Real.pi)
  (h4 : ∀ x, f ω φ x = f ω φ (-x))  -- even function
  (h5 : ∀ x, f ω φ (x + Real.pi / (2 * ω)) = f ω φ x)  -- distance between adjacent axes of symmetry
  : f ω φ (3 * Real.pi / 8) = -Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_3pi_8_l889_88994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l889_88996

theorem relationship_abc : 
  let a : ℝ := (1/2)^10
  let b : ℝ := (1/5)^(-(1/2 : ℝ))
  let c : ℝ := Real.log 10 / Real.log (1/5)
  b > a ∧ a > c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l889_88996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_gain_950g_l889_88975

/-- Represents the gain percentage of a shopkeeper using false weights -/
noncomputable def shopkeeper_gain (false_weight : ℝ) : ℝ :=
  (1000 - false_weight) / 1000 * 100

/-- Theorem stating that a shopkeeper using a 950g weight for 1kg has a 5% gain -/
theorem shopkeeper_gain_950g :
  shopkeeper_gain 950 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_gain_950g_l889_88975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l889_88985

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.sin x)^2 + 2 * Real.sqrt 3 * Real.sin x * Real.sin (x + Real.pi / 2)

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ T = Real.pi ∧ ∀ (x : ℝ), f (x + T) = f x ∧
    ∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  (∀ (y : ℝ), y ∈ Set.Icc 0 3 ↔ ∃ (x : ℝ), x ∈ Set.Icc 0 (2 * Real.pi / 3) ∧ f x = y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l889_88985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_rotating_curves_l889_88949

/-- The volume of the solid generated by rotating the area enclosed by two curves about the x-axis -/
noncomputable def rotationVolume (f g : ℝ → ℝ) (a b : ℝ) : ℝ :=
  Real.pi * ∫ x in a..b, (g x)^2 - (f x)^2

/-- The condition that two curves touch each other -/
def touchingCurves (f g : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, f x = g x ∧ deriv f x = deriv g x

theorem volume_of_rotating_curves (a : ℝ) :
  let f := fun x : ℝ ↦ Real.log x / x
  let g := fun x : ℝ ↦ a * x^2
  touchingCurves f g →
  rotationVolume f g 0 1 + rotationVolume g f 1 (Real.exp (1/3)) =
    Real.pi * (1 + 100 * Real.exp (1/3) - 72 * Real.exp (2/3)) / (36 * Real.exp (2/3)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_rotating_curves_l889_88949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_special_function_unique_function_is_special_l889_88926

/-- A function satisfying specific properties -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x, x ≠ 0 → f x ≠ 1) ∧
  (∀ x y, x ≠ 0 ∧ y ≠ 0 → f (x * y) = f x * f (-y) - f x + f y) ∧
  (∀ x, x ≠ 0 ∧ x ≠ 1 → f (f x) = 1 / f (1 / x))

/-- The unique function satisfying the special properties -/
noncomputable def unique_function (x : ℝ) : ℝ :=
  (x - 1) / x

/-- Theorem stating that the unique_function is the only function satisfying special_function -/
theorem unique_special_function :
  ∀ f : ℝ → ℝ, special_function f → f = unique_function := by
  sorry

/-- Theorem stating that unique_function satisfies special_function -/
theorem unique_function_is_special :
  special_function unique_function := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_special_function_unique_function_is_special_l889_88926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_ghee_quantity_l889_88969

/-- Represents the composition of a ghee mixture -/
structure GheeMixture where
  total : ℝ
  pure_percentage : ℝ
  vanaspati_percentage : ℝ

/-- Calculates the new mixture after adding pure ghee -/
noncomputable def add_pure_ghee (mixture : GheeMixture) (added_pure : ℝ) : GheeMixture :=
  { total := mixture.total + added_pure,
    pure_percentage := (mixture.pure_percentage * mixture.total + added_pure) / (mixture.total + added_pure),
    vanaspati_percentage := mixture.vanaspati_percentage * mixture.total / (mixture.total + added_pure) }

theorem original_ghee_quantity (mixture : GheeMixture) (h1 : mixture.pure_percentage = 0.6)
    (h2 : mixture.vanaspati_percentage = 0.4)
    (h3 : (add_pure_ghee mixture 10).vanaspati_percentage = 0.2) :
  mixture.total = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_ghee_quantity_l889_88969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_example_l889_88913

/-- The volume of a cone in liters given its diameter and height in centimeters -/
noncomputable def cone_volume_liters (diameter : ℝ) (height : ℝ) : ℝ :=
  (1 / 3000) * Real.pi * (diameter / 2) ^ 2 * height

/-- Theorem: The volume of a cone with diameter 12cm and height 10cm is 0.12π liters -/
theorem cone_volume_example : cone_volume_liters 12 10 = 0.12 * Real.pi := by
  sorry

-- Remove the #eval line as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_example_l889_88913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_area_is_4pi_l889_88933

/-- The equation of the ellipse -/
def ellipse_equation (x y : ℝ) : Prop :=
  4 * x^2 + 8 * x + 9 * y^2 - 36 * y + 64 = 0

/-- The area of the ellipse -/
noncomputable def ellipse_area : ℝ := 4 * Real.pi

/-- Theorem stating that the area of the ellipse is 4π -/
theorem ellipse_area_is_4pi :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  (∀ (x y : ℝ), ellipse_equation x y ↔ (x + 1)^2 / a^2 + (y - 2)^2 / b^2 = 1) ∧
  ellipse_area = Real.pi * a * b := by
  sorry

#check ellipse_area_is_4pi

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_area_is_4pi_l889_88933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_centers_collinearity_l889_88921

/-- Triangle class representing a triangle in 2D space -/
class Triangle (α : Type*) [NormedAddCommGroup α] [InnerProductSpace ℝ α] where
  A : α
  B : α
  C : α

/-- Circumcenter of a triangle -/
noncomputable def circumcenter {α : Type*} [NormedAddCommGroup α] [InnerProductSpace ℝ α] (t : Triangle α) : α :=
  sorry

/-- Centroid of a triangle -/
noncomputable def centroid {α : Type*} [NormedAddCommGroup α] [InnerProductSpace ℝ α] (t : Triangle α) : α :=
  sorry

/-- Orthocenter of a triangle -/
noncomputable def orthocenter {α : Type*} [NormedAddCommGroup α] [InnerProductSpace ℝ α] (t : Triangle α) : α :=
  sorry

/-- Collinearity of points -/
def collinear {α : Type*} [NormedAddCommGroup α] [InnerProductSpace ℝ α] (p q r : α) : Prop :=
  ∃ t : ℝ, q - p = t • (r - p)

theorem triangle_centers_collinearity {α : Type*} [NormedAddCommGroup α] [InnerProductSpace ℝ α] (t : Triangle α) :
  let O := circumcenter t
  let G := centroid t
  let H := orthocenter t
  collinear O G H ∧ dist O G = (1/2 : ℝ) * dist G H := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_centers_collinearity_l889_88921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_is_positive_reals_range_of_f_is_positive_reals_alt_l889_88981

-- Define the function f(x) = (1/2)^(2-x)
noncomputable def f (x : ℝ) : ℝ := (1/2)^(2-x)

-- Theorem stating that the range of f is (0, +∞)
theorem range_of_f_is_positive_reals :
  Set.range f = Set.Ioi 0 :=
by
  sorry

-- Alternative formulation of the theorem
theorem range_of_f_is_positive_reals_alt :
  ∀ y > 0, ∃ x, f x = y :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_is_positive_reals_range_of_f_is_positive_reals_alt_l889_88981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_finishes_in_45_days_l889_88936

/-- The time (in days) it takes A to complete the entire work -/
noncomputable def time_A : ℝ := 40

/-- The time (in days) it takes B to complete the entire work -/
noncomputable def time_B : ℝ := 60

/-- The number of days A worked before leaving -/
noncomputable def days_A_worked : ℝ := 10

/-- The fraction of work completed by A before leaving -/
noncomputable def work_completed_by_A : ℝ := days_A_worked / time_A

/-- The fraction of work remaining after A left -/
noncomputable def work_remaining : ℝ := 1 - work_completed_by_A

/-- The time (in days) it takes B to complete the remaining work -/
noncomputable def time_B_to_finish : ℝ := work_remaining * time_B

theorem b_finishes_in_45_days : time_B_to_finish = 45 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_finishes_in_45_days_l889_88936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_term_of_geometric_series_l889_88960

/-- The sum of an infinite geometric series with first term a and common ratio r -/
noncomputable def infiniteGeometricSeriesSum (a : ℝ) (r : ℝ) : ℝ := a / (1 - r)

/-- Theorem: For an infinite geometric series with common ratio -1/3 and sum 9, the first term is 12 -/
theorem first_term_of_geometric_series (r : ℝ) (S : ℝ) (a : ℝ) 
    (h_r : r = -1/3) 
    (h_S : S = 9) 
    (h_sum : S = infiniteGeometricSeriesSum a r) : 
  a = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_term_of_geometric_series_l889_88960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_BFG_l889_88995

-- Define the rectangle ABCD
noncomputable def rectangle_length : ℝ := 6
noncomputable def rectangle_width : ℝ := 4

-- Define the diagonal AC and its division
noncomputable def diagonal_length : ℝ := Real.sqrt (rectangle_length ^ 2 + rectangle_width ^ 2)
noncomputable def segment_length : ℝ := diagonal_length / 4

-- Define the theorem
theorem area_of_triangle_BFG :
  let height : ℝ := (rectangle_length * rectangle_width) / diagonal_length
  let area_BFG : ℝ := (1 / 2) * segment_length * height
  area_BFG = 6 / 13 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_BFG_l889_88995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_fourth_term_l889_88905

def IsGeometricSequence (a b c : ℝ) : Prop := b / a = c / b ∧ a ≠ 0 ∧ b ≠ 0

theorem geometric_sequence_fourth_term (a : ℝ) 
  (h1 : IsGeometricSequence a (2*a+2) (3*a+3)) : 
  let r := (2*a+2) / a
  let fourth_term := (3*a+3) * r
  fourth_term = -13.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_fourth_term_l889_88905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_measure_l889_88911

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The area of the triangle -/
noncomputable def area (t : Triangle) : ℝ := (t.b^2 + t.c^2 - t.a^2) / 4

/-- The condition b * sin(B) - c * sin(C) = a -/
def condition (t : Triangle) : Prop := t.b * Real.sin t.B - t.c * Real.sin t.C = t.a

/-- The theorem stating that if the conditions are met, angle B measures 77.5° -/
theorem angle_B_measure (t : Triangle) 
  (h1 : area t = (t.b^2 + t.c^2 - t.a^2) / 4)
  (h2 : condition t) : 
  t.B = 77.5 * π / 180 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_measure_l889_88911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_area_100_l889_88907

/-- A geometric figure consisting of three identical squares and two congruent right triangles -/
structure GeometricFigure where
  square_side : ℝ
  triangle_leg : ℝ

/-- The area of the geometric figure -/
noncomputable def area (f : GeometricFigure) : ℝ :=
  3 * f.square_side^2 + 2 * (1/2 * f.triangle_leg^2)

/-- The perimeter of the geometric figure -/
noncomputable def perimeter (f : GeometricFigure) : ℝ :=
  3 * f.square_side + 2 * (f.triangle_leg * Real.sqrt 2)

/-- Theorem stating that a geometric figure with area 100 has perimeter 15 + 10√2 -/
theorem perimeter_of_area_100 (f : GeometricFigure) :
  area f = 100 → perimeter f = 15 + 10 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_area_100_l889_88907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cards_arrangement_exists_l889_88952

/-- Represents a card with two numbers -/
structure Card where
  side1 : Nat
  side2 : Nat

/-- Theorem: Cards can be arranged to show all numbers from 1 to n -/
theorem cards_arrangement_exists (n : Nat) (cards : List Card) :
  (cards.length = n) →
  (∀ c ∈ cards, c.side1 ≤ n ∧ c.side2 ≤ n) →
  (∀ i ∈ Finset.range n, (cards.filter (λ c => c.side1 = i + 1 ∨ c.side2 = i + 1)).length = 2) →
  ∃ arrangement : List Nat, arrangement.length = n ∧ 
    (∀ i ∈ Finset.range n, i + 1 ∈ arrangement) ∧
    (∀ i ∈ Finset.range n, ∃ c ∈ cards, c.side1 = arrangement[i]? ∨ c.side2 = arrangement[i]?) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cards_arrangement_exists_l889_88952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_equality_l889_88959

-- Define the triangle ABC
variable (A B C : ℂ)

-- Define midpoints
noncomputable def A' (A B C : ℂ) : ℂ := (B + C) / 2
noncomputable def B' (A B C : ℂ) : ℂ := (C + A) / 2
noncomputable def C' (A B C : ℂ) : ℂ := (A + B) / 2

-- Define altitudes
noncomputable def B_star (A B C : ℂ) : ℂ := (A + B + C - C * A / B) / 2
noncomputable def C_star (A B C : ℂ) : ℂ := (A + B + C - A * B / C) / 2

-- Define midpoints of altitudes
noncomputable def B_hash (A B C : ℂ) : ℂ := (B + B_star A B C) / 2
noncomputable def C_hash (A B C : ℂ) : ℂ := (C + C_star A B C) / 2

-- Define intersection K
noncomputable def K (A B C : ℂ) : ℂ := 
  let num := 2 * ((A^2 * B^2 - A^2 * B * C) + (B^2 * C^2 - B^2 * C * A) + (C^2 * A^2 - C^2 * A * B))
  let den := (A^2 * B + B^2 * C + C^2 * A) - 6 * A * B * C
  num / den

-- Define intersection L (on BC)
noncomputable def L (A B C : ℂ) : ℂ := 
  let t := (K A B C - A).re / (B - C).re
  B + t * (C - B)

-- Define angle
noncomputable def angle (P Q R : ℂ) : ℝ := sorry

-- State the theorem
theorem angle_equality (A B C : ℂ) : 
  angle B A (L A B C) = angle C A (A' A B C) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_equality_l889_88959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_relations_and_a_value_l889_88925

-- Define the universal set U
def U : Set ℕ := {x | x > 0 ∧ x < 6}

-- Define set A
def A : Set ℕ := U

-- Define set B
def B : Set ℕ := {x | (x - 1) * (x - 2) = 0}

-- Define set C
def C (a : ℕ) : Set ℕ := {a, a^2 + 1}

-- State the theorem
theorem set_relations_and_a_value :
  ∃ (a : ℕ),
    (B ⊆ C a) ∧
    (C a ⊆ B) ∧
    (A ∩ (U \ B) = {3, 4, 5}) ∧
    (A ∪ B = {1, 2, 3, 4, 5}) ∧
    (a = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_relations_and_a_value_l889_88925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_cosine_region_l889_88902

theorem area_cosine_region : 
  let lowerBound : ℝ := -π/3
  let upperBound : ℝ := π/3
  let f (x : ℝ) := Real.cos x
  ∫ x in lowerBound..upperBound, f x = 2 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_cosine_region_l889_88902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_divisors_1_to_20_l889_88956

def divisors (n : ℕ) : Finset ℕ :=
  Finset.filter (λ m => n % m = 0) (Finset.range (n + 1))

def divisor_count (n : ℕ) : ℕ := (divisors n).card

theorem max_divisors_1_to_20 :
  (∀ n ∈ Finset.range 21, divisor_count n ≤ 6) ∧
  (divisor_count 12 = 6) ∧ (divisor_count 18 = 6) ∧ (divisor_count 20 = 6) ∧
  (∀ n ∈ Finset.range 21, divisor_count n = 6 → n ∈ ({12, 18, 20} : Finset ℕ)) :=
sorry

#eval divisor_count 12
#eval divisor_count 18
#eval divisor_count 20

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_divisors_1_to_20_l889_88956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_product_equality_l889_88953

def A : Matrix (Fin 2) (Fin 3) ℤ := ![![3, -1, 4], ![0, 5, -2]]
def B : Matrix (Fin 3) (Fin 3) ℤ := ![![2, 0, -1], ![1, 3, 4], ![5, -2, 3]]
def C : Matrix (Fin 2) (Fin 3) ℤ := ![![25, -11, 5], ![-5, 19, 14]]

theorem matrix_product_equality : A * B = C := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_product_equality_l889_88953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_AB_l889_88984

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 2*y + 9 = 0

-- Define the line l (axis of symmetry)
def line_l (k x y : ℝ) : Prop := k*x + y - 2 = 0

-- Define point A
def point_A (k : ℝ) : ℝ × ℝ := (0, k)

-- State the theorem
theorem length_of_AB (k : ℝ) :
  -- Given conditions
  (∃ x y, circle_C x y ∧ line_l k x y) →  -- l is the axis of symmetry of C
  (∃ B : ℝ × ℝ, circle_C B.1 B.2 ∧ 
    -- B is on the circle and AB is tangent to the circle
    (∃ t : ℝ, t * (B.1 - 0) = B.2 - k ∧ 
    (B.1 - 0) * (B.1 - 3) + (B.2 - k) * (B.2 + 1) = 0)) →
  -- Conclusion
  let A := point_A k
  let d := Real.sqrt ((A.1 - 3)^2 + (A.2 + 1)^2)
  d = 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_AB_l889_88984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_symmetry_implies_sum_not_negative_one_l889_88967

/-- Given a function f(x) = (x-1)³ - ax - b + 2, if f(2-x) = 1 - f(x) for all x, then a + b ≠ -1 -/
theorem function_symmetry_implies_sum_not_negative_one (a b : ℝ) :
  (∀ x, (2 - x - 1)^3 - a * (2 - x) - b + 2 = 1 - ((x - 1)^3 - a * x - b + 2)) →
  a + b ≠ -1 := by
  intro h
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_symmetry_implies_sum_not_negative_one_l889_88967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_endpoint_coordinates_l889_88932

def A : ℝ × ℝ := (3, -1)
def b : ℝ × ℝ := (-3, 4)

def is_unit_vector (v : ℝ × ℝ) : Prop :=
  v.1 * v.1 + v.2 * v.2 = 1

def is_parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v = (k * w.1, k * w.2)

def endpoint (start : ℝ × ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (start.1 + v.1, start.2 + v.2)

theorem vector_endpoint_coordinates :
  ∀ a : ℝ × ℝ,
  is_unit_vector a →
  is_parallel a b →
  (let e := endpoint A a;
   e = (3 - 3 / Real.sqrt 5, -1 + 4 / Real.sqrt 5) ∨
   e = (3 + 3 / Real.sqrt 5, -1 - 4 / Real.sqrt 5)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_endpoint_coordinates_l889_88932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_81_expression_is_integer_l889_88980

theorem divisible_by_81 (n : ℕ) : ∃ (k : ℤ), (10^n - 1 - 9*n : ℤ) = 81 * k := by sorry

theorem expression_is_integer (n : ℕ) : ∃ (m : ℤ), ((1 : ℚ)/81) * ((10^n - 1 : ℕ) : ℚ) - (n : ℚ)/9 = m := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_81_expression_is_integer_l889_88980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_intersection_l889_88935

/-- Curve C₁ -/
noncomputable def C₁ (t : ℝ) : ℝ × ℝ :=
  (t + 1/t, t - 1/t)

/-- Curve C₂ -/
noncomputable def C₂ (a θ : ℝ) : ℝ × ℝ :=
  (a * Real.cos θ, Real.sin θ)

/-- Focus of C₂ -/
noncomputable def focus (a : ℝ) : ℝ × ℝ :=
  (Real.sqrt (a^2 - 1), 0)

/-- Main theorem -/
theorem curve_intersection (a : ℝ) (h : a > 1) :
  (∃ t : ℝ, C₁ t = focus a) → a = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_intersection_l889_88935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l889_88971

/-- The function g(x) -/
noncomputable def g (a b x : ℝ) : ℝ := a * x^2 - 2 * a * x - 1 + b

/-- The function f(x) -/
noncomputable def f (a b x : ℝ) : ℝ := g a b x / x

/-- Theorem stating the properties of g(x) and f(x) -/
theorem function_properties (a b : ℝ) :
  a > 0 ∧
  (∀ x ∈ Set.Icc 2 3, g a b x ≤ 4) ∧
  (∃ x ∈ Set.Icc 2 3, g a b x = 4) ∧
  (∀ x ∈ Set.Icc 2 3, g a b x ≥ 1) ∧
  (∃ x ∈ Set.Icc 2 3, g a b x = 1) →
  a = 1 ∧ b = 2 ∧
  (∀ k : ℝ, (∃ x ∈ Set.Icc (-1 : ℝ) 1, f a b (2^x) - k * 2^x ≥ 0) → k ≤ 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l889_88971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_43_and_73_l889_88927

theorem sin_43_and_73 (a : ℝ) (h : Real.sin (43 * π / 180) = a) :
  (a < Real.sqrt 2 / 2) ∧
  (Real.sin (73 * π / 180) = (Real.sqrt (1 - a^2)) / 2 + (Real.sqrt 3 * a) / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_43_and_73_l889_88927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_domain_sum_l889_88973

-- Define the function
noncomputable def f (x : ℝ) : ℝ := 2^(2*x) - 2^(x+2) + 7

-- State the theorem
theorem max_domain_sum {m n : ℝ} (hm : m ≥ 0) (hn : n ≤ 2) 
  (hrange : ∀ x ∈ Set.Icc m n, f x ∈ Set.Icc 3 7) :
  n + m ≤ 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_domain_sum_l889_88973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_determination_l889_88947

theorem triangle_side_determination (a b c : ℝ) (B : ℝ) :
  c = 5 → b = 3 → B = 2 * π / 3 →
  ∃! a, a > 0 ∧ a + b > c ∧ b + c > a ∧ a + c > b ∧ 
    Real.cos B = (a^2 + c^2 - b^2) / (2 * a * c) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_determination_l889_88947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_time_formula_l889_88951

/-- Calculates the total time taken by a train traveling three sections of a journey. -/
noncomputable def totalTime (b km p : ℝ) : ℝ :=
  b / 50 + km / 75 + p / 100

/-- Theorem stating that the total time taken by the train is equal to (6b + 4km + 3p) / 300 hours. -/
theorem total_time_formula (b km p : ℝ) :
  totalTime b km p = (6 * b + 4 * km + 3 * p) / 300 := by
  -- Expand the definition of totalTime
  unfold totalTime
  -- Perform algebraic manipulations
  ring
  -- The proof is complete
  done

#check total_time_formula

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_time_formula_l889_88951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l889_88924

/-- The function f(x) as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2017 * x) + Real.cos (2017 * x)

/-- The maximum value of f(x) -/
def A : ℝ := 2  -- This is given in the problem conditions

/-- Theorem stating the minimum value of 2A|x₁ - x₂| -/
theorem min_value_theorem (x₁ x₂ : ℝ) 
  (h : ∀ x : ℝ, f x₁ ≤ f x ∧ f x ≤ f x₂) : 
  ∃ (x₁' x₂' : ℝ), (∀ x : ℝ, f x₁' ≤ f x ∧ f x ≤ f x₂') ∧ 
    2 * A * |x₁' - x₂'| = 4 * Real.pi / 2017 ∧
    ∀ x₁ x₂ : ℝ, (∀ x : ℝ, f x₁ ≤ f x ∧ f x ≤ f x₂) → 
      2 * A * |x₁ - x₂| ≥ 4 * Real.pi / 2017 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l889_88924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_neg_8_47_to_nearest_tenth_l889_88988

/-- Rounds a real number to the nearest tenth -/
noncomputable def roundToNearestTenth (x : ℝ) : ℝ := 
  (⌊x * 10 + 0.5⌋ : ℝ) / 10

/-- The statement that rounding -8.47 to the nearest tenth equals -8.5 -/
theorem round_neg_8_47_to_nearest_tenth :
  roundToNearestTenth (-8.47) = -8.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_neg_8_47_to_nearest_tenth_l889_88988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_power_sum_l889_88928

theorem tenth_power_sum (a b : ℝ) : 
  (∀ n : ℕ, a^(n+1) + b^(n+1) = 2*(n+1) - 1) → a^10 + b^10 = 19 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_power_sum_l889_88928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_divisible_by_3_or_8_l889_88987

def is_divisible_by (n m : ℕ) : Prop := ∃ k, n = m * k

def count_divisible (max divisor : ℕ) : ℕ :=
  (max / divisor : ℕ)

theorem probability_divisible_by_3_or_8 (max : ℕ) (h : max = 75) :
  (count_divisible max 3 + count_divisible max 8 - count_divisible max 24 : ℚ) / max = 31 / 75 := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_divisible_by_3_or_8_l889_88987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_seven_count_l889_88938

/-- The count of a specific digit in a range of integers -/
def digitCount (start : ℕ) (stop : ℕ) (digit : ℕ) : ℕ :=
  sorry

/-- Theorem: The count of the digit 7 in the integers from 10 through 149 inclusive is 24 -/
theorem digit_seven_count : digitCount 10 149 7 = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_seven_count_l889_88938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_needed_is_36_l889_88955

/-- Represents the recipe for fruit punch -/
structure FruitPunchRecipe where
  water : ℚ
  honey : ℚ
  apple_juice : ℚ
  water_honey_ratio : water = 3 * honey
  honey_juice_ratio : honey = 3 * apple_juice

/-- Calculates the amount of water needed for the fruit punch -/
def water_needed (recipe : FruitPunchRecipe) (juice_used : ℚ) : ℚ :=
  recipe.water * (juice_used / recipe.apple_juice)

/-- Theorem: Given the recipe ratios and 4 cups of apple juice, 36 cups of water are needed -/
theorem water_needed_is_36 (recipe : FruitPunchRecipe) :
  water_needed recipe 4 = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_needed_is_36_l889_88955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ones_digit_of_8_to_47_l889_88998

/-- The ones digit of 8^n for positive n -/
def onesDigitOf8Power (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 6
  | 1 => 8
  | 2 => 4
  | _ => 2

theorem ones_digit_of_8_to_47 :
  onesDigitOf8Power 47 = 2 := by
  rfl

#eval onesDigitOf8Power 47

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ones_digit_of_8_to_47_l889_88998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l889_88903

noncomputable def f (x : ℝ) : ℝ := 
  (Real.arcsin (x/3))^2 + Real.pi * Real.arccos (x/3) + (Real.pi^2/4) * (x^2 + 4*x + 3)

theorem f_range : 
  ∀ x, x ∈ Set.Icc (-3 : ℝ) 3 → 
    (5 * Real.pi^2) / 4 ≤ f x ∧ f x ≤ (53 * Real.pi^2) / 12 ∧
    ∃ x₁ x₂, x₁ ∈ Set.Icc (-3 : ℝ) 3 ∧ x₂ ∈ Set.Icc (-3 : ℝ) 3 ∧
      f x₁ = (5 * Real.pi^2) / 4 ∧ f x₂ = (53 * Real.pi^2) / 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l889_88903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_tournament_games_l889_88963

theorem chess_tournament_games (n : ℕ) (h1 : n ≥ 17) : 
  (n * (n - 1)) / 2 = 136 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_tournament_games_l889_88963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_perp_plane_and_line_parallel_plane_implies_lines_perp_l889_88909

/-- A plane in 3D space -/
structure Plane where

/-- A line in 3D space -/
structure Line where

/-- Perpendicularity relation between a line and a plane -/
def Line.perp_to_plane (l : Line) (α : Plane) : Prop := sorry

/-- Parallelism relation between a line and a plane -/
def Line.parallel_to_plane (l : Line) (α : Plane) : Prop := sorry

/-- Perpendicularity relation between two lines -/
def Line.perp (l₁ l₂ : Line) : Prop := sorry

/-- The theorem stating that if a line is perpendicular to a plane and another line is parallel to the same plane, then the two lines are perpendicular -/
theorem line_perp_plane_and_line_parallel_plane_implies_lines_perp
  (l m : Line) (α : Plane)
  (h1 : l.perp_to_plane α)
  (h2 : m.parallel_to_plane α) :
  l.perp m :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_perp_plane_and_line_parallel_plane_implies_lines_perp_l889_88909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_sequence_f_5_geometric_sequence_f_n_arithmetic_sequence_f_n_l889_88940

def binomial_coefficient (n k : ℕ) : ℕ := 
  Nat.choose n k

def f (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  Finset.sum (Finset.range n) (λ k => a (k + 1) * binomial_coefficient n (k + 1))

theorem constant_sequence_f_5 (a : ℕ → ℕ) (h : ∀ n, a n = 1) :
  f a 5 = 31 := by
  sorry

theorem geometric_sequence_f_n (a : ℕ → ℕ) (h : ∀ n, a n = 3^(n-1)) :
  ∀ n, f a n = (4^n - 1) / 3 := by
  sorry

theorem arithmetic_sequence_f_n (a : ℕ → ℕ) (h : ∀ n, a n = 2*n - 1) :
  ∀ n, f a n - 1 = (n - 1) * 2^n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_sequence_f_5_geometric_sequence_f_n_arithmetic_sequence_f_n_l889_88940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bill_face_value_l889_88961

/-- Calculates the face value of a bill given its true discount, interest rate, and time to maturity. -/
noncomputable def calculate_face_value (true_discount : ℝ) (interest_rate : ℝ) (time : ℝ) : ℝ :=
  (true_discount * (100 + interest_rate * time)) / (interest_rate * time)

/-- Theorem stating that the face value of a bill with the given conditions is 2520. -/
theorem bill_face_value : 
  let true_discount : ℝ := 270
  let interest_rate : ℝ := 16
  let time : ℝ := 9 / 12
  calculate_face_value true_discount interest_rate time = 2520 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval calculate_face_value 270 16 (9/12)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bill_face_value_l889_88961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_value_tangent_line_equation_l889_88914

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + x

-- Statement for the definite integral
theorem integral_value : 
  ∫ x in (-3)..3, (f x + x^2) = 18 := by sorry

-- Statement for the tangent line
theorem tangent_line_equation :
  ∃ x₀ : ℝ, 
    (f x₀ = x₀^3 + x₀) ∧ 
    ((deriv f) x₀ * 0 - 2 = f x₀) ∧
    (∀ x, (deriv f) x₀ * x - 2 = 4 * x - 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_value_tangent_line_equation_l889_88914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_F_measure_l889_88989

/-- Given a geometric configuration where:
    - Angle B measures 120°
    - Angle C is adjacent to angle B on a straight line
    - Angles C, D, and E form a triangle
    - Angle D is 45°
    - Angle E is 30°
    - Angle F is vertically opposite to angle C
    Prove that the measure of angle F is 60° -/
theorem angle_F_measure (B C D E F : ℝ) : 
  B = 120 → 
  B + C = 180 → 
  C + D + E = 180 → 
  D = 45 → 
  E = 30 → 
  F = C → 
  F = 60 := by
  intros hB hBC hCDE hD hE hF
  -- Proof steps would go here
  sorry

#check angle_F_measure

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_F_measure_l889_88989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_imaginary_part_of_roots_l889_88918

theorem max_imaginary_part_of_roots (z : ℂ) : 
  z^6 - z^4 + z^2 - 1 = 0 → 
  ∃ (root : ℂ), root^6 - root^4 + root^2 - 1 = 0 ∧ 
    ∀ (w : ℂ), w^6 - w^4 + w^2 - 1 = 0 → w.im ≤ root.im ∧
    root.im = Real.sin (π/4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_imaginary_part_of_roots_l889_88918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l889_88968

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 3 * x / (x - 4) + Real.sqrt (x + 2)

-- Define the domain of f
def domain_f : Set ℝ := {x | x ≥ -2 ∧ x ≠ 4}

-- Theorem statement
theorem domain_of_f : 
  {x : ℝ | ∃ y, f x = y} = domain_f := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l889_88968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roof_to_cube_ratio_is_golden_ratio_l889_88966

/-- A cube-shaped house with a special roof -/
structure RoofedCube where
  /-- Edge length of the cube -/
  cube_edge : ℝ
  /-- Edge length of the roof -/
  roof_edge : ℝ
  /-- The roof consists of two isosceles triangles and two symmetrical trapezoids -/
  roof_shape : Prop
  /-- All edges of the roof are equal -/
  roof_edges_equal : Prop
  /-- Any two adjacent faces of the roof form the same angle with each other -/
  roof_faces_angle : Prop

/-- The ratio of the roof edge length to the cube edge length for a special roofed cube -/
noncomputable def roof_to_cube_ratio (rc : RoofedCube) : ℝ :=
  rc.roof_edge / rc.cube_edge

/-- Theorem: The ratio of the roof edge length to the cube edge length is (√5 - 1) / 2 -/
theorem roof_to_cube_ratio_is_golden_ratio (rc : RoofedCube) :
  roof_to_cube_ratio rc = (Real.sqrt 5 - 1) / 2 := by
  sorry

#check roof_to_cube_ratio_is_golden_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roof_to_cube_ratio_is_golden_ratio_l889_88966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_square_area_l889_88991

/-- The parabola function -/
def f (x : ℝ) : ℝ := x^2 - 10*x + 21

/-- The side length of the inscribed square -/
noncomputable def s : ℝ := -1 + 2 * Real.sqrt 5

/-- The x-coordinate of the center of the inscribed square -/
def center_x : ℝ := 5

theorem inscribed_square_area :
  let square_area := (2 * s)^2
  square_area = 64 - 16 * Real.sqrt 5 ∧
  ∀ x ∈ Set.Icc (center_x - s) (center_x + s),
    0 ≤ f x ∧
    f (center_x - s) = 0 ∧
    f (center_x + s) = 0 ∧
    f (center_x + s) = 2 * s :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_square_area_l889_88991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_domain_l889_88919

open Real

-- Define the function
noncomputable def f (x : ℝ) : ℝ := tan (x - π/3)

-- Define the domain
def domain : Set ℝ := {x | ∀ k : ℤ, x ≠ k * π + 5*π/6}

-- Theorem statement
theorem tan_domain : 
  {x : ℝ | ∃ y, f x = y} = domain := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_domain_l889_88919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_single_line_worth_1000_l889_88904

/-- The value of a single line in points -/
def single_line_value : ℕ := sorry

/-- The value of a tetris in points -/
def tetris_value : ℕ := 8 * single_line_value

/-- The number of single lines Tim scored -/
def tim_singles : ℕ := 6

/-- The number of tetrises Tim scored -/
def tim_tetrises : ℕ := 4

/-- Tim's total score -/
def tim_total_score : ℕ := 38000

theorem single_line_worth_1000 :
  tim_singles * single_line_value + tim_tetrises * tetris_value = tim_total_score →
  single_line_value = 1000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_single_line_worth_1000_l889_88904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_range_l889_88954

theorem log_inequality_range (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) 
  (h3 : Real.log (3/4) / Real.log a < 1) : 
  a ∈ Set.Ioo 0 (3/4) ∪ Set.Ioi 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_range_l889_88954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_l889_88900

/-- Definition of the Focus of a parabola -/
def Focus (f : ℝ → ℝ) : ℝ × ℝ := sorry

/-- The focus of the parabola y = -1/16 * x^2 is at the point (0, -4) -/
theorem parabola_focus : 
  Focus (λ x => -1/16 * x^2) = (0, -4) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_l889_88900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_company_reduction_l889_88910

def original_employees : ℝ := 224.13793103448276
def reduction_percentage : ℝ := 0.13

theorem company_reduction :
  ∃ (new_employees : ℕ), 
    (Int.floor (original_employees * (1 - reduction_percentage))) = new_employees ∧
    new_employees = 195 :=
by
  -- We use Int.floor instead of .round
  -- The proof is skipped using sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_company_reduction_l889_88910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_formula_triangle_area_specific_case_l889_88957

/-- The area of a triangle enclosed by the x-axis, y-axis, and the line y = tx + 6 -/
noncomputable def triangleArea (t : ℝ) : ℝ :=
  |18 / t|

/-- Theorem stating that the area of the triangle is |18/t| -/
theorem triangle_area_formula (t : ℝ) (h : t ≠ 0) :
  let line := fun (x : ℝ) => t * x + 6
  triangleArea t = |18 / t| :=
by
  -- The proof goes here
  sorry

/-- Specific case when t = -4 -/
theorem triangle_area_specific_case :
  triangleArea (-4) = 9/2 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_formula_triangle_area_specific_case_l889_88957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_property_l889_88916

noncomputable def projection (v : ℝ × ℝ) (u : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := v.1 * u.1 + v.2 * u.2
  let norm_squared := u.1 * u.1 + u.2 * u.2
  (dot_product / norm_squared * u.1, dot_product / norm_squared * u.2)

theorem projection_property :
  let proj := projection (3, 3) (45/10, 9/10)
  proj.1 = 45/10 ∧ proj.2 = 9/10 →
  let result := projection (1, -1) (5, 1)
  result.1 = 10/13 ∧ result.2 = 2/13 := by
  sorry

#check projection_property

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_property_l889_88916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_positive_probability_l889_88917

-- Define the interval
def interval : Set ℝ := Set.Icc (-30) 15

-- Define a function to represent the random selection
noncomputable def randomSelect : ℝ → ℝ :=
  sorry

-- Define the probability of an event
noncomputable def probability (event : ℝ → ℝ → Prop) : ℝ :=
  sorry

-- State the theorem
theorem product_positive_probability :
  probability (fun x y => x * y > 0 ∧ x ∈ interval ∧ y ∈ interval) = 5/9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_positive_probability_l889_88917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_holes_unfilled_volume_l889_88972

theorem dog_holes_unfilled_volume (hole_volumes : Fin 8 → ℝ) (fill_percentages : Fin 8 → ℝ)
  (hv : ∀ i, hole_volumes i = [2, 3, 1.5, 4, 2.5, 1, 3.5, 2].get i)
  (fp : ∀ i, fill_percentages i = [0.6, 0.75, 0.8, 0.5, 0.9, 0.7, 0.4, 0.85].get i) :
  (Finset.univ.sum fun i => hole_volumes i * (1 - fill_percentages i)) = 6.8 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_holes_unfilled_volume_l889_88972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lathe_defect_probabilities_l889_88986

theorem lathe_defect_probabilities
  (defect_rate_1 defect_rate_2 defect_rate_3 : ℝ)
  (proportion_1 proportion_2 proportion_3 : ℝ)
  (h1 : defect_rate_1 = 0.06)
  (h2 : defect_rate_2 = 0.05)
  (h3 : defect_rate_3 = 0.05)
  (h4 : proportion_1 = 0.25)
  (h5 : proportion_2 = 0.30)
  (h6 : proportion_3 = 0.45)
  (h7 : proportion_1 + proportion_2 + proportion_3 = 1) :
  let total_defect_rate := defect_rate_1 * proportion_1 + defect_rate_2 * proportion_2 + defect_rate_3 * proportion_3
  (defect_rate_1 * proportion_1 = 0.015) ∧
  (total_defect_rate = 0.0525) ∧
  (defect_rate_2 * proportion_2 / total_defect_rate = 2/7) ∧
  (defect_rate_3 * proportion_3 / total_defect_rate = 3/7) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lathe_defect_probabilities_l889_88986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_distances_l889_88934

-- Define the line l
noncomputable def line_l (α : ℝ) (t : ℝ) : ℝ × ℝ :=
  (t * Real.cos α, -2 + t * Real.sin α)

-- Define the circle C
def circle_C : ℝ × ℝ × ℝ :=
  (0, 3, 3)  -- (center_x, center_y, radius)

-- Define the intersection points A and B
noncomputable def intersection_points (α : ℝ) : Option (ℝ × ℝ × ℝ × ℝ) :=
  sorry  -- This would compute the intersection points

-- Theorem statement
theorem product_of_distances (α : ℝ) :
  let points := intersection_points α
  ∀ xA yA xB yB, points = some (xA, yA, xB, yB) →
    let PA := Real.sqrt ((xA - 0)^2 + (yA - (-2))^2)
    let PB := Real.sqrt ((xB - 0)^2 + (yB - (-2))^2)
    PA * PB = 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_distances_l889_88934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_radius_in_cone_max_radius_value_proof_l889_88982

/-- A cone with specified dimensions containing three spheres -/
structure ConeSpheres where
  height : ℝ
  slant_height : ℝ
  sphere_radius : ℝ

/-- Predicate for valid configuration of spheres in the cone -/
def valid_configuration (cs : ConeSpheres) : Prop :=
  cs.height = 4 ∧
  cs.slant_height = 8 ∧
  cs.sphere_radius > 0 ∧
  -- Spheres touch each other externally
  -- Spheres touch the lateral surface of the cone
  -- First two spheres touch the base of the cone
  True  -- placeholder for complex geometric conditions

/-- The maximum radius of spheres in the given cone configuration -/
noncomputable def max_sphere_radius : ℝ := 12 / (5 + 2 * Real.sqrt 3)

/-- Theorem stating the maximum sphere radius in the given cone configuration -/
theorem max_radius_in_cone (cs : ConeSpheres) :
  valid_configuration cs →
  cs.sphere_radius ≤ max_sphere_radius ∧
  ∃ (cs' : ConeSpheres), valid_configuration cs' ∧ cs'.sphere_radius = max_sphere_radius := by
  sorry

/-- Proof that the maximum radius is indeed the claimed value -/
theorem max_radius_value_proof :
  max_sphere_radius = 12 / (5 + 2 * Real.sqrt 3) := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_radius_in_cone_max_radius_value_proof_l889_88982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_A_equals_negative_fifteen_l889_88939

def A (x y : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![x, 3; -4, y]

theorem det_A_equals_negative_fifteen (x y : ℝ) :
  let A := A x y
  let B := 3 • A⁻¹
  (A + B = 1) → Matrix.det A = -15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_A_equals_negative_fifteen_l889_88939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_sqrt_two_l889_88906

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

theorem unique_solution_sqrt_two :
  ∀ x : ℝ, (x^2 - floor x = 1) ↔ (x = Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_sqrt_two_l889_88906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_sum_max_triangle_cosine_sum_max_achievable_l889_88943

theorem triangle_cosine_sum_max (A B C : ℝ) : 
  A + B + C = π → Real.cos A + Real.cos B * Real.cos C ≤ 1 := by sorry

theorem triangle_cosine_sum_max_achievable : 
  ∃ A B C, A + B + C = π ∧ Real.cos A + Real.cos B * Real.cos C = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_sum_max_triangle_cosine_sum_max_achievable_l889_88943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_time_lost_l889_88946

/-- Represents the distance between points A and B in kilometers. -/
noncomputable def distance : ℝ := 75

/-- Represents the speed of the car in km/h. -/
noncomputable def car_speed : ℝ := 120

/-- Represents the speed of the train in km/h. -/
noncomputable def train_speed : ℝ := car_speed * 1.5

/-- Represents the time taken by the car to travel from A to B in hours. -/
noncomputable def car_time : ℝ := distance / car_speed

/-- Represents the actual time taken by the train to travel from A to B in hours. -/
noncomputable def train_time : ℝ := distance / train_speed

/-- Represents the time lost by the train due to stops in minutes. -/
noncomputable def time_lost : ℝ := (car_time - train_time) * 60

/-- Theorem stating that the time lost by the train due to stops is 12.5 minutes. -/
theorem train_time_lost : time_lost = 12.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_time_lost_l889_88946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equation_solution_l889_88958

theorem complex_equation_solution (z : ℂ) :
  z / (2 + Complex.I) = 2 - Complex.I * Real.sqrt 2 →
  z = (4 + Real.sqrt 2) + Complex.I * (2 * (1 - Real.sqrt 2)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equation_solution_l889_88958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_and_F_properties_l889_88978

noncomputable def f (x : ℝ) := (2 * x^2 - 2 * x + 2) / (x^2 + 1)

noncomputable def F (x : ℝ) := Real.log (f x)

theorem f_range_and_F_properties :
  (∀ x, 1 ≤ f x ∧ f x ≤ 3) ∧
  (∀ x y, x ∈ Set.Icc (-1) 1 → y ∈ Set.Icc (-1) 1 → x < y → F x < F y) ∧
  (∀ t, Real.log (7/5) ≤ F (|t - 1/6| - |t + 1/6|) ∧ F (|t - 1/6| - |t + 1/6|) ≤ Real.log (13/5)) :=
by sorry

#check f_range_and_F_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_and_F_properties_l889_88978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_z_l889_88974

-- Define the complex number z
noncomputable def z : ℂ := (1 : ℂ) / (Complex.I - 1)

-- Theorem statement
theorem modulus_of_z : Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_z_l889_88974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fc_length_l889_88990

-- Define the points
variable (A B C D F : ℝ × ℝ)

-- Define the conditions
def right_angled (p q r : ℝ × ℝ) : Prop := sorry
def angle_measure (p q r : ℝ × ℝ) : ℝ := sorry
def distance (p q : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem fc_length 
  (h1 : right_angled A F B)
  (h2 : right_angled B F C)
  (h3 : right_angled C F D)
  (h4 : angle_measure A F B = 60)
  (h5 : angle_measure F B C = 60)
  (h6 : angle_measure F C D = 60)
  (h7 : distance A F = 48) :
  distance F C = 12 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fc_length_l889_88990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_is_24_l889_88937

/-- Definition of the sequence b_k -/
def b : ℕ → ℚ
  | 0 => 2  -- We start from 0 to match Lean's natural number indexing
  | 1 => 3
  | k + 2 => (1 / 2) * b (k + 1) + (1 / 3) * b k

/-- The sum of the entire sequence -/
noncomputable def sequenceSum : ℚ := ∑' k, b k

/-- Theorem stating that the sum of the sequence is 24 -/
theorem sequence_sum_is_24 : sequenceSum = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_is_24_l889_88937
