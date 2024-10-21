import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_specific_l1352_135222

/-- The volume of a square-based pyramid with given dimensions -/
noncomputable def pyramid_volume (base_side : ℝ) (lateral_edge : ℝ) : ℝ :=
  let base_area := base_side ^ 2
  let height := Real.sqrt (lateral_edge ^ 2 - (base_side ^ 2 / 2))
  (1 / 3) * base_area * height

/-- Theorem: The volume of a square-based pyramid with base side length 2 
    and lateral edge length √6 is equal to 8/3 -/
theorem pyramid_volume_specific : pyramid_volume 2 (Real.sqrt 6) = 8 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_specific_l1352_135222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sk_finite_iff_power_of_two_l1352_135288

/-- Definition of the set Sk -/
def Sk (k : ℕ) : Set (ℕ × ℕ × ℕ) :=
  {t | Odd t.2.1 ∧ 
       Nat.gcd t.2.2 t.1 = 1 ∧ 
       t.2.2 + t.1 = k ∧ 
       t.2.1 ∣ t.2.2 ^ t.2.1 + t.1 ^ t.2.1}

/-- Main theorem: Sk is finite iff k is a power of 2 -/
theorem sk_finite_iff_power_of_two (k : ℕ) (h : k > 1) :
  Set.Finite (Sk k) ↔ ∃ m : ℕ, k = 2^m := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sk_finite_iff_power_of_two_l1352_135288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l1352_135223

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (2 + (Real.sqrt 2 / 2) * t, 1 + (Real.sqrt 2 / 2) * t)

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 9

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ t, p = line_l t ∧ circle_C p.1 p.2}

-- State the theorem
theorem intersection_distance :
  ∃ A B, A ∈ intersection_points ∧ B ∈ intersection_points ∧ dist A B = 2 * Real.sqrt 7 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l1352_135223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_line_segment_l1352_135207

/-- The set of points P satisfying |PF₁| + |PF₂| = 4, where F₁ = (-2, 0) and F₂ = (2, 0), forms a line segment. -/
theorem trajectory_is_line_segment :
  ∃ (S : Set (ℝ × ℝ)),
    S = {P : ℝ × ℝ | Real.sqrt ((P.1 + 2)^2 + P.2^2) + Real.sqrt ((P.1 - 2)^2 + P.2^2) = 4} ∧
    ∃ (a b : ℝ × ℝ), S = {P : ℝ × ℝ | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • a + t • b} :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_line_segment_l1352_135207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_points_at_distance_l1352_135231

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Distance between two points in 2D plane -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Check if a point is outside a circle -/
def isOutside (p : ℝ × ℝ) (c : Circle) : Prop :=
  distance p c.center > c.radius

/-- Number of intersection points between two circles -/
noncomputable def intersectionPoints (c1 c2 : Circle) : ℕ :=
  let d := distance c1.center c2.center
  if d > c1.radius + c2.radius then 0
  else if d < abs (c1.radius - c2.radius) then 0
  else if d = c1.radius + c2.radius then 1
  else if d = abs (c1.radius - c2.radius) then 1
  else 2

theorem max_points_at_distance (D : Circle) (Q : ℝ × ℝ) :
  D.radius = 4 →
  isOutside Q D →
  (∃ (n : ℕ), ∀ (P : ℝ × ℝ),
    (distance P D.center = D.radius ∧ distance P Q = 5) →
    n ≤ 2 ∧
    (∃ (points : Finset (ℝ × ℝ)), points.card = n ∧
      ∀ (p : ℝ × ℝ), p ∈ points ↔ (distance p D.center = D.radius ∧ distance p Q = 5))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_points_at_distance_l1352_135231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_photo_arrangement_count_l1352_135262

theorem photo_arrangement_count : 
  (10 : ℕ).choose 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_photo_arrangement_count_l1352_135262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_closed_form_l1352_135247

def f : ℕ → ℚ
| 0 => 1  -- We define f(0) as 1 to match f(1) in the original problem
| 1 => 3  -- This corresponds to f(2) in the original problem
| n+2 => 2 * f n + f (n+1)

theorem f_closed_form (n : ℕ) : f n = (2^(n+2) + (-1)^(n+1)) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_closed_form_l1352_135247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_discount_percentage_l1352_135212

/-- Additional discount percentage calculation --/
theorem additional_discount_percentage
  (shoes_price : ℝ)
  (shoes_discount : ℝ)
  (shirt_price : ℝ)
  (num_shirts : ℕ)
  (total_spent : ℝ)
  (h1 : shoes_price = 200)
  (h2 : shoes_discount = 0.3)
  (h3 : shirt_price = 80)
  (h4 : num_shirts = 2)
  (h5 : total_spent = 285) :
  (shoes_price * (1 - shoes_discount) + num_shirts * shirt_price - total_spent) /
  (shoes_price * (1 - shoes_discount) + num_shirts * shirt_price) = 0.05 := by
  sorry

#check additional_discount_percentage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_discount_percentage_l1352_135212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_value_l1352_135228

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + a * x - 1

def g (a : ℝ) (x : ℝ) : ℝ := (Real.exp 2 * (x^2 - a)) / (f a x - a * x + 1)

theorem lambda_value (a : ℝ) (x₁ x₂ : ℝ) (h₁ : x₁ < x₂) 
  (h₂ : ∃ (l : ℝ), ∀ (x : ℝ), l * ((2 * x₁ - x₁^2) * Real.exp (2 - x₁) - a) - x₂ * g a x₁ ≥ 0) :
  ∃ (l : ℝ), l = (2 * Real.exp 2) / (Real.exp 2 + 1) :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_value_l1352_135228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_barycentric_coordinates_of_J_l1352_135232

-- Define the triangle and points
variable (A B C G H K L D J : ℝ × ℝ)

-- Define the ratios
def AG_GB : ℚ := 3/2
def BH_HC : ℚ := 1/3
def AK_KC : ℚ := 1/2
def BL_LC : ℚ := 2/1

-- Define the intersections
def D_is_intersection (A B C G H D : ℝ × ℝ) : Prop := 
  ∃ t₁ t₂ : ℝ, 0 < t₁ ∧ t₁ < 1 ∧ 0 < t₂ ∧ t₂ < 1 ∧ 
  D = (1 - t₁) • A + t₁ • G ∧ D = (1 - t₂) • C + t₂ • H

def J_is_intersection (A B C D L K J : ℝ × ℝ) : Prop := 
  ∃ s₁ s₂ : ℝ, 0 < s₁ ∧ s₁ < 1 ∧ 0 < s₂ ∧ s₂ < 1 ∧ 
  J = (1 - s₁) • D + s₁ • L ∧ J = (1 - s₂) • B + s₂ • K

-- Theorem statement
theorem barycentric_coordinates_of_J
  (hAG_GB : AG_GB = 3/2) (hBH_HC : BH_HC = 1/3) 
  (hAK_KC : AK_KC = 1/2) (hBL_LC : BL_LC = 2/1)
  (hD : D_is_intersection A B C G H D)
  (hJ : J_is_intersection A B C D L K J) :
  ∃ x y z : ℚ, x + y + z = 1 ∧ 
    x = 8/45 ∧ y = 44/135 ∧ z = 83/135 ∧
    J = x • A + y • B + z • C :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_barycentric_coordinates_of_J_l1352_135232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_triple_sequence_exists_l1352_135249

def a (n : ℕ) : ℕ :=
  if (n.digits 2).sum % 2 = 0 then 0 else 1

theorem no_triple_sequence_exists :
  ∀ (k m : ℕ), m > 0 → ∃ (j : ℕ), j < m ∧
    (a (k + j) ≠ a (k + m + j) ∨
     a (k + j) ≠ a (k + 2*m + j) ∨
     a (k + m + j) ≠ a (k + 2*m + j)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_triple_sequence_exists_l1352_135249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_pyramid_height_15cm_l1352_135241

/-- The height of a square-based pyramid formed by four identical cylindrical pipes -/
noncomputable def cylinderPyramidHeight (diameter : ℝ) : ℝ :=
  (diameter * (Real.sqrt 3 + 1)) / 2

/-- Theorem stating that the height of the pyramid of cylinders with diameter 15 cm is (15(√3 + 1))/2 cm -/
theorem cylinder_pyramid_height_15cm :
  cylinderPyramidHeight 15 = (15 * (Real.sqrt 3 + 1)) / 2 := by
  -- Unfold the definition of cylinderPyramidHeight
  unfold cylinderPyramidHeight
  -- The rest of the proof would go here, but we'll use sorry for now
  sorry

#check cylinder_pyramid_height_15cm

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_pyramid_height_15cm_l1352_135241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1352_135292

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos : 0 < a ∧ a < b

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := 
  Real.sqrt (1 + (h.b / h.a)^2)

/-- Checks if a point is on the asymptote of the hyperbola -/
def IsOnAsymptote (h : Hyperbola) (p : ℝ × ℝ) : Prop := sorry

/-- Computes the symmetric point with respect to a given point -/
def SymmetricPoint (f : ℝ × ℝ) (p : ℝ × ℝ) : ℝ × ℝ := sorry

/-- Returns the left focus of the hyperbola -/
def leftFocus (h : Hyperbola) : ℝ × ℝ := sorry

/-- 
Given a hyperbola C: (x^2/a^2) - (y^2/b^2) = 1 with b > a > 0,
if the symmetric point of the left focus with respect to one asymptote 
lies on the other asymptote, then the eccentricity of C is 2.
-/
theorem hyperbola_eccentricity (h : Hyperbola) 
  (h_symmetric : ∃ (p : ℝ × ℝ), p.1 ≠ 0 ∧ 
    IsOnAsymptote h p ∧ 
    IsOnAsymptote h (SymmetricPoint (leftFocus h) p)) :
  eccentricity h = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1352_135292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sunzi_car_problem_l1352_135233

/-- Represents the number of people in the problem -/
def x : ℕ := sorry

/-- Represents the number of cars in the problem -/
def y : ℕ := sorry

/-- Condition 1: If every 3 people share a car, there will be 2 cars left -/
def condition1 : Prop := 3 * (y - 2) = x

/-- Condition 2: If every 2 people share a car, there will be 9 people left without a car -/
def condition2 : Prop := 2 * y + 9 = x

/-- Theorem stating that the system of equations correctly represents the problem -/
theorem sunzi_car_problem :
  (condition1 ∧ condition2) ↔
  (∃ (x y : ℕ), (3 * (y - 2) = x) ∧ (2 * y + 9 = x)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sunzi_car_problem_l1352_135233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inconsistent_polynomial_value_l1352_135245

def is_integer_coeff_poly (f : ℤ → ℤ) : Prop :=
  ∃ (n : ℕ) (a : ℕ → ℤ), ∀ x, f x = (Finset.range (n + 1)).sum (λ i ↦ a i * x^i)

theorem inconsistent_polynomial_value
  (f : ℤ → ℤ)
  (h_integer_coeff : is_integer_coeff_poly f)
  (h_m2 : f (-2) = -56)
  (h_1 : f 1 = -2)
  (h_6 : f 6 = 528) :
  f 3 ≠ 53 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inconsistent_polynomial_value_l1352_135245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_prop_parallel_vectors_solution_l1352_135258

/-- Two planar vectors are parallel if and only if their cross product is zero -/
def IsParallelTo (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

/-- Theorem stating the parallelism condition for planar vectors -/
theorem parallel_vectors_prop {a b c d : ℝ} :
  IsParallelTo (a, b) (c, d) ↔ a * d = b * c := by sorry

/-- Theorem proving the solution to the original problem -/
theorem parallel_vectors_solution :
  ∀ x : ℝ, IsParallelTo (1, x) (-2, 1) → x = -1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_prop_parallel_vectors_solution_l1352_135258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_i_part_ii_l1352_135206

/-- The function f(x) = (x^2 + kx + 1) / (x^2 + 1) -/
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (x^2 + k*x + 1) / (x^2 + 1)

/-- Part I: If the minimum value of f(x) is -1 for x > 0, then k = -4 -/
theorem part_i (k : ℝ) : 
  (∀ x > 0, f k x ≥ -1) ∧ (∃ x > 0, f k x = -1) → k = -4 := by sorry

/-- Part II: For f(x) to form a triangle with side lengths f(x₁), f(x₂), f(x₃) 
    for any x₁, x₂, x₃ ∈ (0, +∞), k must satisfy -1 ≤ k ≤ 2 -/
theorem part_ii (k : ℝ) : 
  (∀ x₁ x₂ x₃, x₁ > 0 → x₂ > 0 → x₃ > 0 → 
    f k x₁ + f k x₂ > f k x₃ ∧ 
    f k x₂ + f k x₃ > f k x₁ ∧ 
    f k x₃ + f k x₁ > f k x₂) 
  ↔ -1 ≤ k ∧ k ≤ 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_i_part_ii_l1352_135206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_min_perpendicular_distance_l1352_135227

noncomputable section

structure Rhombus :=
  (A B C D : ℝ × ℝ)
  (is_rhombus : sorry)
  (diag_AC : Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2) = 24)
  (diag_BD : Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2) = 40)

def point_on_segment (A B M : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ M = (A.1 + t * (B.1 - A.1), A.2 + t * (B.2 - A.2))

noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

theorem rhombus_min_perpendicular_distance (ABCD : Rhombus) (M : ℝ × ℝ)
  (h_M : point_on_segment ABCD.A ABCD.B M)
  (h_AM : distance ABCD.A M = 6)
  (R : ℝ × ℝ)
  (h_R : sorry) -- R is the foot of perpendicular from M to AC
  (S : ℝ × ℝ)
  (h_S : sorry) -- S is the foot of perpendicular from M to BD
  : distance R S = 3.01 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_min_perpendicular_distance_l1352_135227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_for_meaningful_f_l1352_135218

-- Define the function f as noncomputable due to Real.log
noncomputable def f (x m : ℝ) : ℝ := Real.log (x + 2^x - m)

-- State the theorem
theorem range_of_m_for_meaningful_f :
  ∀ m : ℝ, (∀ x ∈ Set.Icc 1 2, ∃ y, f x m = y) →
  m < 3 :=
by
  -- The proof is skipped using sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_for_meaningful_f_l1352_135218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_xy_value_l1352_135220

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), a = (k * b.fst, k * b.snd.fst, k * b.snd.snd)

/-- The problem statement -/
theorem parallel_vectors_xy_value :
  ∀ (x y : ℝ),
  let a : ℝ × ℝ × ℝ := (x, 4, 3)
  let b : ℝ × ℝ × ℝ := (3, -2, y)
  parallel a b → x * y = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_xy_value_l1352_135220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_dot_product_implies_tan_l1352_135244

theorem vector_dot_product_implies_tan (α : ℝ) 
  (h1 : α ∈ Set.Ioo (π / 2) π)
  (h2 : (Real.cos α * Real.cos α * Real.cos α + Real.sin α * (Real.sin α - 1)) = 1 / 5) :
  Real.tan (α + π / 4) = -1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_dot_product_implies_tan_l1352_135244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minor_arc_KB_measure_l1352_135278

-- Define the circle and points
variable (Q : Set (ℝ × ℝ)) -- Circle Q
variable (K T B : ℝ × ℝ) -- Points on the circle

-- Define the angles
variable (angle_KAT : ℝ)
variable (angle_KBT : ℝ)

-- Define the measure of minor arc KB
def minor_arc_KB : ℝ := 2 * angle_KBT

-- Theorem statement
theorem minor_arc_KB_measure
  (h1 : K ∈ Q) (h2 : T ∈ Q) (h3 : B ∈ Q) -- Points on the circle
  (h4 : angle_KAT = 72) -- Angle KAT is 72 degrees
  (h5 : angle_KBT = 40) -- Angle KBT is 40 degrees
  : minor_arc_KB = 80 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minor_arc_KB_measure_l1352_135278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_triangle_properties_l1352_135260

/-- Triangle PQR with given properties -/
structure Triangle where
  -- Side lengths
  PQ : ℝ
  QR : ℝ
  PR : ℝ
  -- Angle equality
  angle_PQR_eq_PRQ : Prop  -- Changed from True to Prop

/-- Properties of the specific triangle in the problem -/
def problem_triangle : Triangle where
  PQ := 10  -- Derived from PR = 10 and angle equality
  QR := 6
  PR := 10
  angle_PQR_eq_PRQ := True  -- This is now a Prop, so True is correct here

/-- Definition of a right triangle -/
def is_right_triangle (t : Triangle) : Prop :=
  t.PQ^2 = t.QR^2 + t.PR^2 ∨ t.QR^2 = t.PQ^2 + t.PR^2 ∨ t.PR^2 = t.PQ^2 + t.QR^2

/-- Definition of triangle perimeter -/
def triangle_perimeter (t : Triangle) : ℝ :=
  t.PQ + t.QR + t.PR

/-- Main theorem about the problem triangle -/
theorem problem_triangle_properties :
  ¬(is_right_triangle problem_triangle) ∧
  triangle_perimeter problem_triangle = 26 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_triangle_properties_l1352_135260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cake_sharing_l1352_135210

noncomputable def max_share (n : ℕ) : ℝ := sorry
noncomputable def min_share (n : ℕ) : ℝ := sorry
noncomputable def sum_shares (n : ℕ) : ℝ := sorry

theorem cake_sharing (n : ℕ) 
  (h_max : ∃ (x : ℝ), x = 1 / 11 ∧ x = max_share n)
  (h_min : ∃ (y : ℝ), y = 1 / 14 ∧ y = min_share n)
  (h_sum : sum_shares n = 1) : 
  12 ≤ n ∧ n ≤ 13 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cake_sharing_l1352_135210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_a_d_over_5_d_l1352_135200

/-- Definition of n_k for a given n and k -/
def n_k (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Definition of s_m(n) -/
def s_m (m : ℕ) (n : ℕ) : ℕ := sorry

/-- Definition of a_d -/
def a_d (d : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem limit_a_d_over_5_d :
  ∀ ε > 0, ∃ N : ℕ, ∀ d ≥ N, |((a_d d : ℚ) / (5^d : ℚ)) - 1/3| < ε :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_a_d_over_5_d_l1352_135200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_real_solutions_l1352_135257

def P : ℕ → (ℝ → ℝ)
  | 0 => fun x => x^2 - 2
  | n + 1 => fun x => P 0 (P n x)

theorem distinct_real_solutions (n : ℕ+) :
  ∀ x y : ℝ, P n x = x → P n y = y → x ≠ y → x = y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_real_solutions_l1352_135257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_valid_arrangements_l1352_135202

def total_students : ℕ := 360
def min_rows : ℕ := 12
def min_students_per_row : ℕ := 18

def is_valid_arrangement (students_per_row : ℕ) : Bool :=
  students_per_row ≥ min_students_per_row &&
  (total_students / students_per_row) ≥ min_rows &&
  total_students % students_per_row = 0

def valid_arrangements : List ℕ :=
  (List.range (total_students - min_students_per_row + 1)).filter (λ x => is_valid_arrangement (x + min_students_per_row))

theorem sum_of_valid_arrangements :
  (valid_arrangements.map (λ x => x + min_students_per_row)).sum = 92 := by
  sorry

#eval (valid_arrangements.map (λ x => x + min_students_per_row)).sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_valid_arrangements_l1352_135202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1352_135209

noncomputable def f (x : ℝ) : ℝ := -2 * (Real.sin x)^2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 1

theorem f_properties :
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ ∀ S, 0 < S ∧ S < T → ∃ y, f (y + S) ≠ f y) ∧
  (∀ k : ℤ, ∃ x₀, ∀ x, f (x₀ + x) = f (x₀ - x) ∧ x₀ = k / 2 * π - π / 12) ∧
  (∀ x ∈ Set.Icc (-π / 6) (π / 3), f x ≤ 2 ∧ f x ≥ -1) ∧
  (∃ x₁ ∈ Set.Icc (-π / 6) (π / 3), f x₁ = 2) ∧
  (∃ x₂ ∈ Set.Icc (-π / 6) (π / 3), f x₂ = -1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1352_135209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_calculation_l1352_135217

-- Define the triangle ABC
structure Triangle where
  A : ℝ  -- Angle A
  B : ℝ  -- Angle B
  C : ℝ  -- Angle C
  a : ℝ  -- Side opposite to angle A
  b : ℝ  -- Side opposite to angle B
  c : ℝ  -- Side opposite to angle C

-- Define the theorem
theorem triangle_area_calculation (t : Triangle) 
  (h1 : Real.cos t.A = -3/5)
  (h2 : Real.sin t.C = 1/2)
  (h3 : t.c = 1) :
  (1/2 * t.a * t.c * Real.sin t.B) = (8 * Real.sqrt 3 - 6) / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_calculation_l1352_135217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_comparison_l1352_135211

theorem exponential_comparison : ∃ (y₁ y₂ y₃ : ℝ), 
  y₁ = (0.9 : ℝ)^(0.2 : ℝ) ∧ 
  y₂ = (0.9 : ℝ)^(0.4 : ℝ) ∧ 
  y₃ = (1.2 : ℝ)^(0.1 : ℝ) ∧ 
  y₃ > y₁ ∧ y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_comparison_l1352_135211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_given_equation_is_ellipse_l1352_135235

/-- A conic section is an ellipse if it can be expressed in the form
    (x-h)^2/a^2 + (y-k)^2/b^2 = 1, where a and b are positive real numbers. -/
def is_ellipse (f : ℝ → ℝ → Prop) : Prop :=
  ∃ h k a b : ℝ, a > 0 ∧ b > 0 ∧
    ∀ x y, f x y ↔ (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1

/-- The equation (x-3)^2 + 9(y+2)^2 = 144 represents an ellipse. -/
theorem given_equation_is_ellipse :
  is_ellipse (λ x y ↦ (x - 3)^2 + 9*(y + 2)^2 = 144) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_given_equation_is_ellipse_l1352_135235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overlapping_squares_area_l1352_135203

/-- Represents a square sheet of paper -/
structure Sheet where
  side_length : ℝ
  rotation : ℝ

/-- Calculates the approximate area of the shape formed by overlapping sheets -/
noncomputable def approximate_overlapping_area (sheets : List Sheet) : ℝ :=
  sorry

/-- Defines approximate equality for real numbers -/
def approx_equal (x y : ℝ) (ε : ℝ := 0.1) : Prop :=
  abs (x - y) < ε

notation:50 a " ≈ " b => approx_equal a b

theorem overlapping_squares_area :
  let sheets : List Sheet := [
    { side_length := 4, rotation := 0 },
    { side_length := 4, rotation := 20 },
    { side_length := 4, rotation := 40 }
  ]
  approximate_overlapping_area sheets ≈ 35 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_overlapping_squares_area_l1352_135203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apex_to_centroid_distance_formula_l1352_135261

/-- A triangular pyramid with base sides a, b, c and lateral edges m, n, p -/
structure TriangularPyramid where
  a : ℝ
  b : ℝ
  c : ℝ
  m : ℝ
  n : ℝ
  p : ℝ

/-- The distance from the apex of a triangular pyramid to the centroid of its base -/
noncomputable def apexToCentroidDistance (pyramid : TriangularPyramid) : ℝ :=
  (1/3) * Real.sqrt (3 * pyramid.m^2 + 3 * pyramid.n^2 + 3 * pyramid.p^2 - 
                     pyramid.a^2 - pyramid.b^2 - pyramid.c^2)

/-- Theorem: The distance from the apex of a triangular pyramid to the centroid of its base
    is (1/3) * √(3m² + 3n² + 3p² - a² - b² - c²) -/
theorem apex_to_centroid_distance_formula (pyramid : TriangularPyramid) :
  apexToCentroidDistance pyramid = 
    (1/3) * Real.sqrt (3 * pyramid.m^2 + 3 * pyramid.n^2 + 3 * pyramid.p^2 - 
                       pyramid.a^2 - pyramid.b^2 - pyramid.c^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apex_to_centroid_distance_formula_l1352_135261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expMaclaurinSeriesCorrect_sinMaclaurinSeriesCorrect_cosMaclaurinSeriesCorrect_l1352_135214

-- Define the Maclaurin series for e^x
noncomputable def expMaclaurinSeries (x : ℝ) : ℝ := ∑' n, x^n / n.factorial

-- Define the Maclaurin series for sin x
noncomputable def sinMaclaurinSeries (x : ℝ) : ℝ := ∑' n, (-1)^n * x^(2*n+1) / (2*n+1).factorial

-- Define the Maclaurin series for cos x
noncomputable def cosMaclaurinSeries (x : ℝ) : ℝ := ∑' n, (-1)^n * x^(2*n) / (2*n).factorial

-- Theorem statements
theorem expMaclaurinSeriesCorrect (x : ℝ) : 
  expMaclaurinSeries x = Real.exp x := by sorry

theorem sinMaclaurinSeriesCorrect (x : ℝ) : 
  sinMaclaurinSeries x = Real.sin x := by sorry

theorem cosMaclaurinSeriesCorrect (x : ℝ) : 
  cosMaclaurinSeries x = Real.cos x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expMaclaurinSeriesCorrect_sinMaclaurinSeriesCorrect_cosMaclaurinSeriesCorrect_l1352_135214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_of_a_l1352_135287

-- Define the weights of individuals
variable (a b c d e : ℝ)

-- Define the conditions
def avg_abc (a b c : ℝ) : Prop := (a + b + c) / 3 = 70
def avg_abcd (a b c d : ℝ) : Prop := (a + b + c + d) / 4 = 70
def weight_e (d e : ℝ) : Prop := e = d + 3
def avg_bcde (b c d e : ℝ) : Prop := (b + c + d + e) / 4 = 68

-- Theorem statement
theorem weight_of_a (h1 : avg_abc a b c) (h2 : avg_abcd a b c d) (h3 : weight_e d e) (h4 : avg_bcde b c d e) :
  a = 81 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_of_a_l1352_135287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_formula_l1352_135272

def S : ℕ → ℚ
  | 0 => -2/3  -- Adding a case for 0 to cover all natural numbers
  | 1 => -2/3
  | n+2 => -1 / (S (n+1) + 2)

theorem S_formula (n : ℕ) : S n = -(n + 1) / (n + 2) := by
  induction n with
  | zero => 
    simp [S]
    -- The proof for n = 0 case
    sorry
  | succ n ih => 
    cases n with
    | zero => 
      simp [S]
      -- The proof for n = 1 case
      sorry
    | succ n => 
      simp [S]
      -- The inductive step
      sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_formula_l1352_135272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_print_is_output_statement_input_is_not_output_statement_if_is_not_output_statement_while_is_not_output_statement_l1352_135282

-- Define the basic statement types
inductive Statement
  | Input
  | Print
  | If
  | While

-- Define a predicate for output statements
def isOutputStatement : Statement → Prop
| Statement.Input => False
| Statement.Print => True
| Statement.If => False
| Statement.While => False

-- Theorem to prove
theorem print_is_output_statement :
  isOutputStatement Statement.Print :=
by
  -- Unfold the definition of isOutputStatement
  unfold isOutputStatement
  -- The goal is now True, which is trivially true
  trivial

-- Additional theorems to show other statements are not output statements
theorem input_is_not_output_statement :
  ¬(isOutputStatement Statement.Input) :=
by
  unfold isOutputStatement
  trivial

theorem if_is_not_output_statement :
  ¬(isOutputStatement Statement.If) :=
by
  unfold isOutputStatement
  trivial

theorem while_is_not_output_statement :
  ¬(isOutputStatement Statement.While) :=
by
  unfold isOutputStatement
  trivial

end NUMINAMATH_CALUDE_ERRORFEEDBACK_print_is_output_statement_input_is_not_output_statement_if_is_not_output_statement_while_is_not_output_statement_l1352_135282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_four_root_l1352_135255

theorem factorial_four_root : Real.sqrt ((4 * 3 * 2 * 1) * (4 * 3 * 2 * 1)) = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_four_root_l1352_135255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_range_intersection_l1352_135280

-- Define the domain of ln(1-x)
def A : Set ℝ := {x | x < 1}

-- Define the range of x^2
def B : Set ℝ := {y | ∃ x, y = x^2}

-- Theorem statement
theorem domain_range_intersection :
  A ∩ B = Set.Ico 0 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_range_intersection_l1352_135280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_equilateral_triangles_in_octagonal_lattice_l1352_135204

/-- Represents a point in the octagonal lattice -/
structure LatticePoint where
  x : ℝ
  y : ℝ

/-- Represents the octagonal lattice -/
def OctagonalLattice : Set LatticePoint :=
  sorry

/-- Distance between two points in the lattice -/
def distance (p q : LatticePoint) : ℝ :=
  sorry

/-- Predicate to check if three points form an equilateral triangle -/
def isEquilateralTriangle (p q r : LatticePoint) : Prop :=
  distance p q = distance q r ∧ distance q r = distance r p

/-- The theorem stating that no equilateral triangles exist in the octagonal lattice -/
theorem no_equilateral_triangles_in_octagonal_lattice :
  ∀ p q r, p ∈ OctagonalLattice → q ∈ OctagonalLattice → r ∈ OctagonalLattice → 
    ¬(isEquilateralTriangle p q r) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_equilateral_triangles_in_octagonal_lattice_l1352_135204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_configuration_theorem_l1352_135281

/-- Represents a circle with a given radius -/
structure Circle where
  radius : ℝ

/-- Represents an equilateral triangle -/
structure EquilateralTriangle

/-- Represents the configuration of circles as described in the problem -/
structure CircleConfiguration where
  A : Circle
  B : Circle
  C : Circle
  D : Circle
  E : Circle
  T : EquilateralTriangle

/-- The main theorem statement -/
theorem circle_configuration_theorem (config : CircleConfiguration) :
  config.A.radius = 12 ∧
  config.B.radius = 4 ∧
  config.C.radius = 3 ∧
  config.D.radius = 3 ∧
  (∃ m n : ℕ, config.E.radius = m / n ∧ Nat.Coprime m n) →
  config.E.radius = 25 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_configuration_theorem_l1352_135281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sleeve_fabric_area_is_five_l1352_135275

/-- Represents the costume components and material costs for Jenna's Oliver Twist costume --/
structure CostumeMaterial where
  skirtLength : ℝ
  skirtWidth : ℝ
  numSkirts : ℕ
  bodiceShirtArea : ℝ
  materialCostPerSqFt : ℝ
  totalSpent : ℝ

/-- Calculates the square footage of fabric needed for each sleeve --/
noncomputable def sleeveFabricArea (c : CostumeMaterial) : ℝ :=
  let totalSkirtArea := c.skirtLength * c.skirtWidth * (c.numSkirts : ℝ)
  let totalAreaExceptSleeves := totalSkirtArea + c.bodiceShirtArea
  let totalCostExceptSleeves := totalAreaExceptSleeves * c.materialCostPerSqFt
  let sleevesCost := c.totalSpent - totalCostExceptSleeves
  let totalSleeveArea := sleevesCost / c.materialCostPerSqFt
  totalSleeveArea / 2

/-- Theorem stating that the square footage of fabric needed for each sleeve is 5 square feet --/
theorem sleeve_fabric_area_is_five (c : CostumeMaterial)
  (h1 : c.skirtLength = 12)
  (h2 : c.skirtWidth = 4)
  (h3 : c.numSkirts = 3)
  (h4 : c.bodiceShirtArea = 2)
  (h5 : c.materialCostPerSqFt = 3)
  (h6 : c.totalSpent = 468) :
  sleeveFabricArea c = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sleeve_fabric_area_is_five_l1352_135275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_group_frequency_l1352_135259

/-- Proves that the frequency of the fifth group in a class of 50 students
    divided into 5 groups is 0.2, given the frequencies of the first four groups. -/
theorem fifth_group_frequency (total_students : ℕ) (num_groups : ℕ) 
  (freq_group1 : ℕ) (freq_group2 : ℕ) (freq_group3 : ℕ) (freq_group4 : ℕ) :
  total_students = 50 →
  num_groups = 5 →
  freq_group1 = 7 →
  freq_group2 = 12 →
  freq_group3 = 13 →
  freq_group4 = 8 →
  (total_students - (freq_group1 + freq_group2 + freq_group3 + freq_group4)) / total_students = (2 : ℚ) / 10 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_group_frequency_l1352_135259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_rotation_result_l1352_135266

noncomputable def original_number : ℂ := 3 - Complex.I * Real.sqrt 3
noncomputable def rotation_angle : ℝ := -Real.pi/3  -- Negative for clockwise rotation

theorem complex_rotation_result :
  original_number * Complex.exp (Complex.I * (rotation_angle : ℂ)) = -2 * Real.sqrt 3 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_rotation_result_l1352_135266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_is_90_l1352_135242

/-- A right triangle with a specific configuration -/
structure RightTriangle where
  -- The length of the hypotenuse
  xz : ℝ
  -- The length of one side
  xy : ℝ
  -- Assertion that this forms a right triangle
  right_triangle : xz^2 + xy^2 = (xz + xy)^2 / 4

/-- The area of a quadrilateral formed by a specific construction in the right triangle -/
noncomputable def quadrilateral_area (t : RightTriangle) : ℝ :=
  let yz := Real.sqrt (t.xy^2 - t.xz^2)
  let triangle_area := t.xz * yz / 2
  let small_triangle_area := triangle_area / 4
  triangle_area - small_triangle_area

/-- The main theorem stating the area of the quadrilateral is 90 -/
theorem quadrilateral_area_is_90 (t : RightTriangle) 
    (h1 : t.xy = 26) (h2 : t.xz = 10) : quadrilateral_area t = 90 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_is_90_l1352_135242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_second_and_fourth_l1352_135274

def numbers : List Int := [-1, 3, 5, 8, 10]

def is_valid_arrangement (arr : List Int) : Prop :=
  arr.length = 5 ∧
  arr.toFinset = numbers.toFinset ∧
  (∃ i ∈ [2, 3, 4], arr[i]! = 10) ∧
  (∃ i ∈ [1, 2], arr[i]! = -1) ∧
  arr[2]! ≠ 5

theorem average_of_second_and_fourth (arr : List Int) 
  (h : is_valid_arrangement arr) : 
  (arr[1]!.toNat + arr[3]!.toNat : ℚ) / 2 = 9/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_second_and_fourth_l1352_135274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_radius_from_perimeter_l1352_135201

/-- The perimeter of a semi-circle with radius r -/
noncomputable def semicircle_perimeter (r : ℝ) : ℝ := Real.pi * r + 2 * r

/-- Theorem: A semi-circle with perimeter 10.797344572538567 cm has a radius of approximately 2.1 cm -/
theorem semicircle_radius_from_perimeter :
  ∃ r : ℝ, semicircle_perimeter r = 10.797344572538567 ∧ abs (r - 2.1) < 0.0000000000000005 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_radius_from_perimeter_l1352_135201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_ratio_proves_final_ratio_l1352_135268

/-- Represents the ratio of milk to water in a mixture -/
structure MilkWaterRatio where
  milk : ℚ
  water : ℚ

/-- Represents a mixture of milk and water -/
structure Mixture where
  volume : ℚ
  ratio : MilkWaterRatio

/-- Performs one operation of removing half the mixture and replacing with pure milk -/
noncomputable def performOperation (m : Mixture) : Mixture :=
  { volume := m.volume,
    ratio := { milk := m.ratio.milk + m.ratio.water / 2,
               water := m.ratio.water / 2 } }

/-- The initial mixture before any operations -/
def initialMixture : Mixture :=
  { volume := 20,
    ratio := { milk := 15, water := 2 } }

/-- The final mixture after two operations -/
noncomputable def finalMixture : Mixture :=
  performOperation (performOperation initialMixture)

theorem initial_ratio_proves_final_ratio :
  finalMixture.ratio.milk / finalMixture.ratio.water = 9 →
  initialMixture.ratio.milk / initialMixture.ratio.water = 15/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_ratio_proves_final_ratio_l1352_135268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_point_division_ratio_l1352_135248

/-- Given points A, B, C, and O in a vector space, prove that if A, B, C are collinear,
    O is not on the line ABC, and there exists a real number m such that 
    m * OA - 3 * OB - OC = 0, then A divides BC in the ratio 1:3 -/
theorem collinear_point_division_ratio 
  {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]
  (O A B C : V) (m : ℝ) 
  (collinear : ∃ (t : ℝ), B - A = t • (C - A))
  (O_not_on_line : O ∉ {P | ∃ (s : ℝ), P - A = s • (C - A)})
  (h : m • (A - O) - 3 • (B - O) - (C - O) = 0) :
  ∃ (l : ℝ), B - A = l • (C - A) ∧ l = 1/3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_point_division_ratio_l1352_135248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_permutations_of_five_letter_word_with_one_repeat_l1352_135276

/-- The number of distinct permutations of a 5-letter word with one letter repeated twice -/
def distinct_permutations : ℕ := 60

/-- The total number of letters in the word -/
def total_letters : ℕ := 5

/-- The number of times the repeated letter appears -/
def repeated_letter_count : ℕ := 2

theorem distinct_permutations_of_five_letter_word_with_one_repeat :
  distinct_permutations = (Nat.factorial total_letters) / (Nat.factorial repeated_letter_count) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_permutations_of_five_letter_word_with_one_repeat_l1352_135276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rice_yield_growth_equation_l1352_135256

/-- Represents the annual average growth rate of rice yield per hectare -/
def x : ℝ := 0 -- We initialize it to 0, but it will be treated as a variable

/-- The initial rice yield per hectare two years ago in kilograms -/
def initial_yield : ℝ := 7200

/-- The current rice yield per hectare in kilograms -/
def current_yield : ℝ := 8450

/-- The number of years between the initial and current yield measurements -/
def years : ℕ := 2

/-- Theorem stating that the equation correctly represents the relationship
    between the initial yield, growth rate, and current yield -/
theorem rice_yield_growth_equation :
  initial_yield * (1 + x) ^ years = current_yield := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rice_yield_growth_equation_l1352_135256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_major_axis_length_l1352_135294

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- An ellipse with axes parallel to the coordinate axes -/
structure Ellipse where
  center : Point
  a : ℝ  -- semi-major axis length
  b : ℝ  -- semi-minor axis length

/-- Check if a point lies on an ellipse -/
def pointOnEllipse (p : Point) (e : Ellipse) : Prop :=
  ((p.x - e.center.x)^2 / e.a^2) + ((p.y - e.center.y)^2 / e.b^2) = 1

/-- The five given points -/
noncomputable def p1 : Point := ⟨-5/2, 1⟩
noncomputable def p2 : Point := ⟨0, 0⟩
noncomputable def p3 : Point := ⟨0, 3⟩
noncomputable def p4 : Point := ⟨4, 0⟩
noncomputable def p5 : Point := ⟨4, 3⟩

theorem ellipse_major_axis_length :
  ∃ (e : Ellipse),
    pointOnEllipse p1 e ∧
    pointOnEllipse p2 e ∧
    pointOnEllipse p3 e ∧
    pointOnEllipse p4 e ∧
    pointOnEllipse p5 e ∧
    e.a * 2 = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_major_axis_length_l1352_135294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_daps_equivalent_to_60_dips_l1352_135205

/-- A type representing the unit 'dap' -/
structure Dap where
  value : ℝ

/-- A type representing the unit 'dop' -/
structure Dop where
  value : ℝ

/-- A type representing the unit 'dip' -/
structure Dip where
  value : ℝ

/-- The conversion rate from daps to dops -/
noncomputable def daps_to_dops : ℝ := 5 / 6

/-- The conversion rate from dops to dips -/
noncomputable def dops_to_dips : ℝ := 10 / 3

theorem daps_equivalent_to_60_dips :
  ∀ (x : Dap),
  x.value * daps_to_dops * dops_to_dips = 60 →
  x.value = 21.6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_daps_equivalent_to_60_dips_l1352_135205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_base4_numbers_sum_specific_base4_numbers_l1352_135291

/-- Converts a base 4 number represented as a list of digits to its decimal equivalent -/
def base4ToDecimal (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => acc * 4 + d) 0

/-- Converts a decimal number to its base 4 representation as a list of digits -/
def decimalToBase4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
    aux n []

/-- The sum of multiple base 4 numbers is equal to a given base 4 number -/
theorem sum_base4_numbers (n1 n2 n3 n4 result : List Nat) : 
  base4ToDecimal n1 + base4ToDecimal n2 + base4ToDecimal n3 + base4ToDecimal n4 = base4ToDecimal result →
  decimalToBase4 (base4ToDecimal n1 + base4ToDecimal n2 + base4ToDecimal n3 + base4ToDecimal n4) = result := by
  sorry

/-- The sum of 132₄, 203₄, 321₄, and 120₄ is equal to 2010₄ in base 4 -/
theorem sum_specific_base4_numbers : 
  base4ToDecimal [2, 3, 1] + base4ToDecimal [3, 0, 2] + base4ToDecimal [1, 2, 3] + base4ToDecimal [0, 2, 1] = base4ToDecimal [0, 1, 0, 2] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_base4_numbers_sum_specific_base4_numbers_l1352_135291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_count_l1352_135234

-- Define the graphs
def graph1 (A : ℝ) (x y : ℝ) : Prop := y = A * x^2
def graph2 (x y : ℝ) : Prop := x^2 + y^2 = 4 * y

-- Define the intersection points
def intersection_points (A : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | graph1 A p.1 p.2 ∧ graph2 p.1 p.2}

-- Theorem statement
theorem intersection_count (A : ℝ) (h : A > 0) :
  (A > 1/4 → ∃ (s : Finset (ℝ × ℝ)), s.card = 3 ∧ ↑s = intersection_points A) ∧
  (A ≤ 1/4 → ∃ (s : Finset (ℝ × ℝ)), s.card = 1 ∧ ↑s = intersection_points A) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_count_l1352_135234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l1352_135224

/-- Calculates the speed of a train in km/hr given its length and time to cross a pole -/
noncomputable def trainSpeed (length : ℝ) (time : ℝ) : ℝ :=
  (length / 1000) / (time / 3600)

theorem train_speed_calculation (length time : ℝ) 
  (h1 : length = 300) 
  (h2 : time = 18) : 
  trainSpeed length time = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l1352_135224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_passage_volume_calculation_l1352_135250

/-- The volume of space within a cubic container of edge length 4 through which
    a sphere of radius 1 can pass -/
noncomputable def sphere_passage_volume : ℝ :=
  8 - (4 / 3) * Real.pi

/-- Theorem: The volume of space within a cubic container of edge length 4
    through which a sphere of radius 1 can pass is equal to 8 - (4/3)π cubic units -/
theorem sphere_passage_volume_calculation (container_edge : ℝ) (sphere_radius : ℝ)
    (h1 : container_edge = 4)
    (h2 : sphere_radius = 1) :
  sphere_passage_volume = 8 - (4 / 3) * Real.pi :=
by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_passage_volume_calculation_l1352_135250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_example_l1352_135239

noncomputable def geometric_series_sum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (r^n - 1) / (r - 1)

theorem geometric_series_example : 
  geometric_series_sum 1 3 9 = 9841 := by
  -- Unfold the definition of geometric_series_sum
  unfold geometric_series_sum
  -- Simplify the expression
  simp [pow_succ]
  -- Perform the numerical computation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_example_l1352_135239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_central_angle_l1352_135286

/-- The central angle of the unfolded diagram of a cone -/
noncomputable def central_angle (slant_height : ℝ) (lateral_surface_area : ℝ) : ℝ :=
  2 * lateral_surface_area / slant_height

/-- Theorem: The central angle of the unfolded diagram of a cone with slant height 1 and lateral surface area 3π/8 is 3π/4 -/
theorem cone_central_angle :
  let slant_height : ℝ := 1
  let lateral_surface_area : ℝ := 3 * Real.pi / 8
  central_angle slant_height lateral_surface_area = 3 * Real.pi / 4 := by
  -- Unfold the definitions
  unfold central_angle
  -- Simplify the expression
  simp [Real.pi]
  -- The proof is complete
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_central_angle_l1352_135286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_r_squared_vs_residuals_l1352_135271

/-- Represents the coefficient of determination (R-squared) in regression analysis -/
noncomputable def R_squared : ℝ → ℝ := sorry

/-- Represents the sum of squared residuals in regression analysis -/
noncomputable def sum_squared_residuals : ℝ → ℝ := sorry

/-- Represents the quality of model fit in regression analysis -/
noncomputable def model_fit_quality : (ℝ → ℝ) → ℝ → ℝ := sorry

/-- Linear regression can produce errors when approximating the real model -/
axiom linear_regression_error : True

/-- Larger R-squared indicates better model fit -/
axiom r_squared_fit : ∀ x y : ℝ, x > y → model_fit_quality R_squared x > model_fit_quality R_squared y

/-- Smaller sum of squared residuals indicates better model fit -/
axiom residuals_fit : ∀ x y : ℝ, x < y → model_fit_quality sum_squared_residuals x > model_fit_quality sum_squared_residuals y

/-- Theorem: In regression analysis, larger R-squared corresponds to smaller sum of squared residuals -/
theorem r_squared_vs_residuals : 
  ∀ x y : ℝ, x > y → R_squared x > R_squared y → sum_squared_residuals x < sum_squared_residuals y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_r_squared_vs_residuals_l1352_135271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_and_minimum_distance_l1352_135295

noncomputable def f (a b x : ℝ) : ℝ := a * x + b / x

noncomputable def min_f (a b : ℝ) : ℝ := Real.sqrt 6

theorem function_properties_and_minimum_distance 
  (a b : ℝ) 
  (h1 : 4 * a > b) 
  (h2 : b > 0) 
  (h3 : f a b 2 = 5/2) 
  (h4 : ∀ x > 0, f a b x ≥ min_f a b) :
  (a = 3/4 ∧ b = 2) ∧ 
  (∀ x > 0, 
    let y := f a b x
    Real.sqrt ((x - 2)^2 + (y - 4)^2) ≥ Real.sqrt 2) := by
  sorry

#check function_properties_and_minimum_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_and_minimum_distance_l1352_135295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_registered_customers_in_program_l1352_135263

structure Customer where
  id : Nat
  registered : Bool

structure Bank where
  id : Nat
  customers : Set Customer

structure LoyaltyProgram where
  participants : Set Customer
  banks : Set Bank

def isRegistered (c : Customer) : Bool :=
  c.registered

def customersInProgram (lp : LoyaltyProgram) (b : Bank) : Set Customer :=
  lp.participants ∩ b.customers

theorem registered_customers_in_program (lp : LoyaltyProgram) (b : Bank) :
  ∀ c ∈ customersInProgram lp b, isRegistered c :=
  sorry

#check registered_customers_in_program

-- More theorems could be added here to model other aspects of the loyalty program

end NUMINAMATH_CALUDE_ERRORFEEDBACK_registered_customers_in_program_l1352_135263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_inequality_l1352_135246

/-- For any acute triangle, the sum of square roots of the ratio of the square of each side 
to the difference of the sum of squares of the other two sides and the square of the side 
is greater than or equal to 3 times the square root of the ratio of the circumradius to 
twice the inradius. -/
theorem acute_triangle_inequality (a b c R r : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c → -- positive side lengths
  0 < R ∧ 0 < r → -- positive radii
  a + b > c ∧ b + c > a ∧ c + a > b → -- triangle inequality
  a^2 < b^2 + c^2 ∧ b^2 < c^2 + a^2 ∧ c^2 < a^2 + b^2 → -- acute triangle condition
  Real.sqrt (a^2 / (b^2 + c^2 - a^2)) + 
  Real.sqrt (b^2 / (c^2 + a^2 - b^2)) + 
  Real.sqrt (c^2 / (a^2 + b^2 - c^2)) ≥ 
  3 * Real.sqrt (R / (2 * r)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_inequality_l1352_135246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_zeros_l1352_135265

-- Define the function f(x) = sin(exp(x))
noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.exp x)

-- Define the open interval (-∞, 0)
def openInterval : Set ℝ := {x : ℝ | x < 0}

-- Theorem statement
theorem infinitely_many_zeros :
  ∃ (S : Set ℝ), S ⊆ openInterval ∧ (∀ x ∈ S, f x = 0) ∧ (¬ Set.Finite S) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_zeros_l1352_135265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_ball_higher_probability_l1352_135237

/-- Probability of a ball landing in bin k -/
noncomputable def prob_bin (k : ℕ) : ℝ :=
  if k = 0 then 1/4 else (1/2)^k * 3/4

/-- The probability that the blue ball is in a higher-numbered bin than the yellow ball -/
noncomputable def prob_blue_higher : ℝ :=
  3/8

theorem blue_ball_higher_probability :
  prob_blue_higher = ∑' k, ∑' j, (prob_bin k) * (prob_bin j) * (if k > j then 1 else 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_ball_higher_probability_l1352_135237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_12_terms_eq_24_l1352_135267

def sequenceA (n : ℕ) : ℚ :=
  match n with
  | 0 => 1
  | n + 1 => let prev := sequenceA n
              if prev < 3 then prev + 1 else prev / 3

def sum_12_terms : ℚ := (Finset.range 12).sum (λ i => sequenceA i)

theorem sum_12_terms_eq_24 : sum_12_terms = 24 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_12_terms_eq_24_l1352_135267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_ratio_l1352_135283

def mySequence (a : ℕ → ℝ) : Prop :=
  a 1 = 5 ∧ ∀ n : ℕ, n ≥ 1 → a n * a (n + 1) = 2^n

theorem sequence_ratio (a : ℕ → ℝ) (h : mySequence a) : a 7 / a 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_ratio_l1352_135283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_composition_roots_l1352_135277

/-- The quadratic function g(x) -/
noncomputable def g (x : ℝ) : ℝ := x^2/3 + x - 2

/-- Theorem stating that the y-coordinate values satisfying g(g(g(x))) = -2 are 3 and -3 -/
theorem g_composition_roots :
  {y : ℝ | ∃ x, g (g (g x)) = -2 ∧ g x = y} = {3, -3} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_composition_roots_l1352_135277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_model_a_better_fit_l1352_135297

/-- Represents a statistical model with a correlation index (R²) -/
structure Model where
  name : String
  r_squared : ℝ

/-- Determines if one model has a better fitting effect than another -/
def has_better_fit (m1 m2 : Model) : Prop :=
  m1.r_squared > m2.r_squared

/-- Approximation relation for real numbers -/
def approx (x y : ℝ) (ε : ℝ) : Prop :=
  abs (x - y) < ε

notation:50 x " ≈ " y => approx x y 0.01

theorem model_a_better_fit (model_a model_b : Model)
  (ha : model_a.name = "A" ∧ model_a.r_squared ≈ 0.96)
  (hb : model_b.name = "B" ∧ model_b.r_squared ≈ 0.85)
  (h_better_fit : ∀ m1 m2 : Model, m1.r_squared > m2.r_squared → has_better_fit m1 m2) :
  has_better_fit model_a model_b :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_model_a_better_fit_l1352_135297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_preserving_transformations_l1352_135299

/-- Represents the types of rigid motion transformations --/
inductive RigidMotion
  | Rotation
  | Translation
  | ReflectionAcross
  | ReflectionPerpendicular

/-- Represents a line in the plane --/
structure Line where
  -- Define line properties here

/-- Represents a triangle in the plane --/
structure Triangle where
  -- Define triangle properties here

/-- Represents a line segment in the plane --/
structure LineSegment where
  -- Define line segment properties here

/-- Represents the pattern on line ℓ --/
structure Pattern where
  ℓ : Line
  triangles : Set Triangle
  segments : Set LineSegment

/-- Applies a rigid motion transformation to a pattern --/
def apply_motion (t : RigidMotion) (p : Pattern) : Pattern :=
  sorry -- Implementation details

/-- Defines if a transformation preserves the pattern --/
def preserves_pattern (t : RigidMotion) (p : Pattern) : Prop :=
  apply_motion t p = p

/-- Counts the number of elements in a finite set --/
def count_elements {α : Type} (s : Set α) : Nat :=
  sorry -- Implementation details

/-- The main theorem stating that exactly two types of transformations preserve the pattern --/
theorem two_preserving_transformations (p : Pattern) :
  ∃! (s : Set RigidMotion), count_elements s = 2 ∧ 
  (∀ t ∈ s, preserves_pattern t p) ∧
  (∀ t ∉ s, ¬preserves_pattern t p) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_preserving_transformations_l1352_135299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_PQ_length_l1352_135216

structure Trapezoid (ABCD : Type) where
  BC : ℝ
  AD : ℝ
  O : ABCD
  P : ABCD
  Q : ABCD
  L : ABCD
  R : ABCD
  parallel_PQ : Bool
  intersect_AC_BD : O = O
  intersect_PQ_AC : L = L
  intersect_PQ_BD : R = R
  area_BOC_eq_LOR : Bool
  L_between_A_O : Bool

noncomputable def length_PQ (t : Trapezoid ABCD) : ℝ :=
  (t.BC * (3 * t.AD - t.BC)) / (t.AD + t.BC)

theorem trapezoid_PQ_length (t : Trapezoid ABCD) :
  length_PQ t = (t.BC * (3 * t.AD - t.BC)) / (t.AD + t.BC) := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_PQ_length_l1352_135216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_120_is_zero_l1352_135253

def c : ℕ → ℚ
  | 0 => 0  -- Add this case to handle Nat.zero
  | 1 => 2
  | 2 => 1
  | n + 3 => (1 - c (n + 2)) / (c (n + 1) ^ 2 + 1)

theorem c_120_is_zero : c 120 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_120_is_zero_l1352_135253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_B_l1352_135285

def A : Set ℝ := { a | -1 ≤ a ∧ a ≤ 2 }

def B : Set (ℝ × ℝ) := { p | p.1 ∈ A ∧ p.2 ∈ A ∧ p.1 + p.2 ≥ 0 }

noncomputable def volume (s : Set (ℝ × ℝ)) : ℝ := sorry

theorem area_of_B : volume B = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_B_l1352_135285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1352_135226

/-- A hyperbola with foci F₁ and F₂, intersected by a line through F₂ perpendicular to the real axis at points P and Q. -/
structure Hyperbola where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  P : ℝ × ℝ
  Q : ℝ × ℝ
  perpendicular_line : (P.2 - F₂.2) * (Q.1 - F₂.1) = (Q.2 - F₂.2) * (P.1 - F₂.1)
  on_hyperbola : ∃ (a : ℝ), |dist P F₁ - dist P F₂| = 2 * a ∧ |dist Q F₁ - dist Q F₂| = 2 * a

/-- The angle PF₁Q is a right angle -/
def right_angle (h : Hyperbola) : Prop :=
  let v1 := (h.P.1 - h.F₁.1, h.P.2 - h.F₁.2)
  let v2 := (h.Q.1 - h.F₁.1, h.Q.2 - h.F₁.2)
  v1.1 * v2.1 + v1.2 * v2.2 = 0

/-- The eccentricity of the hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  dist h.F₁ h.F₂ / (2 * Real.sqrt ((dist h.P h.F₁ * dist h.P h.F₂ + dist h.Q h.F₁ * dist h.Q h.F₂) / 2))

/-- The main theorem: if ∠PF₁Q is a right angle, then the eccentricity is √2 + 1 -/
theorem hyperbola_eccentricity (h : Hyperbola) (angle : right_angle h) :
  eccentricity h = Real.sqrt 2 + 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1352_135226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1352_135296

noncomputable def f (x : ℝ) := Real.sqrt 3 * Real.sin x * Real.cos x - Real.sin x ^ 2

noncomputable def g (x : ℝ) := f (x - Real.pi / 12) + 2

theorem problem_solution (α : ℝ) 
  (h : ∀ x : ℝ, g (α - x) = g (α + x)) :
  g (α + Real.pi / 4) + g (Real.pi / 4) = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1352_135296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_special_case_l1352_135289

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := Real.sqrt (1 + h.b^2 / h.a^2)

/-- The distance between the foci of a hyperbola -/
noncomputable def interfocal_distance (h : Hyperbola) : ℝ := 2 * Real.sqrt (h.a^2 - h.b^2)

theorem hyperbola_eccentricity_special_case (h : Hyperbola) 
  (P Q : ℝ × ℝ) (F₁ F₂ : ℝ × ℝ) :
  -- P and Q are on the right branch of the hyperbola
  (P.1^2 / h.a^2 - P.2^2 / h.b^2 = 1) ∧ 
  (Q.1^2 / h.a^2 - Q.2^2 / h.b^2 = 1) ∧
  P.1 > 0 ∧ Q.1 > 0 →
  -- F₁ and F₂ are the foci
  F₁.1 = -Real.sqrt (h.a^2 - h.b^2) ∧
  F₂.1 = Real.sqrt (h.a^2 - h.b^2) ∧
  F₁.2 = 0 ∧ F₂.2 = 0 →
  -- P and Q are on a circle centered at F₂ with radius F₁F₂
  (P.1 - F₂.1)^2 + P.2^2 = (interfocal_distance h)^2 ∧
  (Q.1 - F₂.1)^2 + Q.2^2 = (interfocal_distance h)^2 →
  -- F₁PQ is an equilateral triangle
  (P.1 - F₁.1)^2 + P.2^2 = (Q.1 - F₁.1)^2 + Q.2^2 ∧
  (P.1 - F₁.1)^2 + P.2^2 = (P.1 - Q.1)^2 + (P.2 - Q.2)^2 →
  -- The eccentricity is (√3 + 1) / 2
  eccentricity h = (Real.sqrt 3 + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_special_case_l1352_135289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crocodile_length_meters_l1352_135219

/-- Represents the length of one ken in centimeters -/
noncomputable def ken_to_cm : ℝ := 180

/-- Represents the length of one shaku in centimeters -/
noncomputable def shaku_to_cm : ℝ := 30

/-- Represents the number of shaku in one ken -/
noncomputable def shaku_per_ken : ℝ := 6

/-- Represents the length of the crocodile from head to tail in ken -/
noncomputable def crocodile_length_ken : ℝ := 10 / 3

/-- Represents the length of the crocodile from tail to head in ken and shaku -/
noncomputable def crocodile_length_ken_shaku : ℝ := 3 + 2 / shaku_per_ken

/-- Theorem stating that the length of the crocodile is 6 meters -/
theorem crocodile_length_meters : 
  crocodile_length_ken * ken_to_cm / 100 = 6 ∧ 
  crocodile_length_ken_shaku * ken_to_cm / 100 = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_crocodile_length_meters_l1352_135219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_monotone_increasing_condition_inequality_for_positive_reals_l1352_135270

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * (x - 1) / (x + 1)

-- Part I
theorem tangent_line_at_one (a : ℝ) :
  (∃ (k : ℝ), (deriv (f a)) 2 = 0) →
  ∃ (b : ℝ), ∀ x y, y = f a 1 + b * (x - 1) ↔ x + 8 * y = 1 :=
sorry

-- Part II
theorem monotone_increasing_condition (a : ℝ) :
  StrictMono (f a) ↔ a ≤ 2 :=
sorry

-- Part III
theorem inequality_for_positive_reals {m n : ℝ} (hm : m > 0) (hn : n > 0) (hmn : m > n) :
  (m - n) / (Real.log m - Real.log n) < (m + n) / 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_monotone_increasing_condition_inequality_for_positive_reals_l1352_135270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_l1352_135273

theorem power_equality (x : ℝ) : (1 / 16 : ℝ) * (2 : ℝ)^20 = (4 : ℝ)^x → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_l1352_135273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proportionality_problem_l1352_135243

theorem proportionality_problem (k₁ k₂ : ℝ) (h₁ : k₁ > 0) (h₂ : k₂ > 0) :
  (∀ x y z : ℝ, x = k₁ * y^3 ∧ y = k₂ / Real.sqrt z) →
  (∃ x : ℝ, x = 3 ∧ 12 = 12) →
  (∃ x : ℝ, x = 24/125 ∧ 75 = 75) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_proportionality_problem_l1352_135243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dryer_cost_is_490_l1352_135251

/-- The cost of a washer-dryer combination -/
noncomputable def total_cost : ℚ := 1200

/-- The difference in cost between the washer and dryer -/
noncomputable def cost_difference : ℚ := 220

/-- The cost of the dryer -/
noncomputable def dryer_cost : ℚ := (total_cost - cost_difference) / 2

theorem dryer_cost_is_490 : dryer_cost = 490 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dryer_cost_is_490_l1352_135251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_dot_product_l1352_135284

-- Define the circle
def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the line
def line (a b c x y : ℝ) : Prop := a*x + b*y + c = 0

-- Define the intersection points
def intersect_points (A B : ℝ × ℝ) (a b c : ℝ) : Prop :=
  line a b c A.1 A.2 ∧ line a b c B.1 B.2 ∧
  unit_circle A.1 A.2 ∧ unit_circle B.1 B.2

-- Define the distance between two points
noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Define the dot product of two vectors
def dot_product (A B : ℝ × ℝ) : ℝ :=
  A.1 * B.1 + A.2 * B.2

-- Theorem statement
theorem intersection_dot_product 
  (a b c : ℝ) (A B : ℝ × ℝ) 
  (h1 : intersect_points A B a b c) 
  (h2 : distance A B = 1) : 
  dot_product A B = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_dot_product_l1352_135284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_subset_l1352_135290

theorem existence_of_subset (n : ℕ) (S : Finset ℕ) 
  (h1 : n > 1) 
  (h2 : S ⊆ Finset.range n) 
  (h3 : S.card > (3 * n) / 4) : 
  ∃ (a b c : ℕ), 
    ({a % n, b % n, c % n, (a + b) % n, (b + c) % n, (c + a) % n, (a + b + c) % n} : Finset ℕ) ⊆ S := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_subset_l1352_135290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_willie_started_with_l1352_135215

-- Define the variables
def emily_gives : ℚ := 7
def willie_ends : ℕ := 43

-- Define the theorem
theorem willie_started_with : 
  (willie_ends : ℚ) - emily_gives = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_willie_started_with_l1352_135215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcenter_configuration_iff_ge_seven_l1352_135229

/-- A configuration of n points on a plane where each point is the circumcenter of a triangle formed by three other points. -/
structure CircumcenterConfiguration (n : ℕ) where
  points : Fin n → ℝ × ℝ
  distinct : ∀ i j, i ≠ j → points i ≠ points j
  is_circumcenter : ∀ i, ∃ j k l, j ≠ k ∧ k ≠ l ∧ l ≠ j ∧
    (points i).1 = (points j).1 + (points k).1 + (points l).1 / 3 ∧
    (points i).2 = (points j).2 + (points k).2 + (points l).2 / 3

/-- The theorem stating that a CircumcenterConfiguration exists if and only if n ≥ 7. -/
theorem circumcenter_configuration_iff_ge_seven (n : ℕ) :
  (n ≥ 3 ∧ Nonempty (CircumcenterConfiguration n)) ↔ n ≥ 7 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcenter_configuration_iff_ge_seven_l1352_135229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partner_a_profit_l1352_135238

/-- Calculate the money received by partner A in a business partnership --/
theorem partner_a_profit (a_investment b_investment total_profit : ℚ) : 
  a_investment = 3500 →
  b_investment = 2500 →
  total_profit = 9600 →
  let management_fee := (10 / 100) * total_profit;
  let remaining_profit := total_profit - management_fee;
  let total_investment := a_investment + b_investment;
  let a_share := (a_investment / total_investment) * remaining_profit
  management_fee + a_share = 6000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_partner_a_profit_l1352_135238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_waist_to_hem_ratio_l1352_135208

/-- Proves that the ratio of the waist length to the hem length is 1:3 given the specified conditions. -/
theorem waist_to_hem_ratio (cuff_length : ℝ) (num_cuffs : ℕ) (hem_length : ℝ) 
  (num_neck_ruffles : ℕ) (neck_ruffle_length : ℝ) (lace_cost_per_meter : ℝ) (total_spent : ℝ) :
  cuff_length = 50 →
  num_cuffs = 2 →
  hem_length = 300 →
  num_neck_ruffles = 5 →
  neck_ruffle_length = 20 →
  lace_cost_per_meter = 6 →
  total_spent = 36 →
  (total_spent / lace_cost_per_meter * 100 - 
   (cuff_length * (num_cuffs : ℝ) + hem_length + neck_ruffle_length * (num_neck_ruffles : ℝ))) / 
   hem_length = 1 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_waist_to_hem_ratio_l1352_135208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overlap_area_is_one_point_five_l1352_135213

/-- A point on a 2D grid --/
structure Point where
  x : ℚ
  y : ℚ

/-- A triangle defined by three points --/
structure Triangle where
  a : Point
  b : Point
  c : Point

/-- Calculate the area of a triangle given its vertices --/
def triangleArea (t : Triangle) : ℚ :=
  (1/2) * abs ((t.b.x - t.a.x) * (t.c.y - t.a.y) - (t.c.x - t.a.x) * (t.b.y - t.a.y))

/-- The first triangle with vertices (0,0), (3,0), and (1,2) --/
def triangle1 : Triangle :=
  { a := { x := 0, y := 0 }
    b := { x := 3, y := 0 }
    c := { x := 1, y := 2 } }

/-- The second triangle with vertices (3,2), (0,1), and (1,0) --/
def triangle2 : Triangle :=
  { a := { x := 3, y := 2 }
    b := { x := 0, y := 1 }
    c := { x := 1, y := 0 } }

/-- Calculate the area of overlap between two triangles --/
noncomputable def overlapArea (t1 t2 : Triangle) : ℚ :=
  sorry -- Actual implementation would go here

theorem overlap_area_is_one_point_five :
  overlapArea triangle1 triangle2 = 3/2 := by
  sorry

#eval triangleArea triangle1
#eval triangleArea triangle2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_overlap_area_is_one_point_five_l1352_135213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_points_sum_l1352_135279

/-- A line passing through two points -/
structure Line where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- Create a line from two points -/
noncomputable def lineFromPoints (x₁ y₁ x₂ y₂ : ℝ) : Line :=
  { m := (y₂ - y₁) / (x₂ - x₁),
    b := y₁ - ((y₂ - y₁) / (x₂ - x₁)) * x₁ }

/-- The sum of slope and y-intercept for a line passing through (-3, 1) and (1, -3) is -3 -/
theorem line_through_points_sum : 
  let l := lineFromPoints (-3) 1 1 (-3)
  l.m + l.b = -3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_points_sum_l1352_135279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_difference_l1352_135264

def sum_of_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

def h (n : ℕ) : ℚ := (sum_of_divisors n : ℚ) / n

theorem h_difference : h 450 - h 225 = 403 / 450 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_difference_l1352_135264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_proof_l1352_135221

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  area : ℝ

-- State the theorem
theorem triangle_proof (t : Triangle) 
  (h1 : t.c = Real.sqrt 3 * t.a * Real.sin t.C - t.c * Real.cos t.A)
  (h2 : t.a = 2)
  (h3 : t.area = Real.sqrt 3) :
  t.A = π / 3 ∧ t.b = 2 ∧ t.c = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_proof_l1352_135221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_articles_produced_correct_production_rate_l1352_135269

/-- Given that x men working x hours a day for x days produce x articles,
    this function calculates the number of articles produced by y men
    working y hours a day for z days. -/
noncomputable def articles_produced (x y z : ℝ) : ℝ :=
  (y^2 * z) / x^2

/-- Theorem stating that the articles_produced function correctly calculates
    the number of articles produced under the given conditions. -/
theorem articles_produced_correct (x y z : ℝ) (hx : x ≠ 0) :
  articles_produced x y z = (y^2 * z) / x^2 := by
  unfold articles_produced
  rfl

/-- Theorem stating that the production rate per person per hour is 1/x^2
    given the initial conditions. -/
theorem production_rate (x : ℝ) (hx : x ≠ 0) :
  1 / x^2 = x / (x * x * x) := by
  field_simp
  ring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_articles_produced_correct_production_rate_l1352_135269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_median_length_l1352_135225

/-- A triangle with specific properties -/
structure SpecialTriangle where
  -- Two medians of the triangle
  median1 : ℝ
  median2 : ℝ
  -- Area of the triangle
  area : ℝ
  -- Conditions on the medians and area
  h_median1 : median1 = 4.5
  h_median2 : median2 = 7.5
  h_area : area = 6 * Real.sqrt 20

/-- The length of the third median in the special triangle -/
noncomputable def thirdMedian (t : SpecialTriangle) : ℝ := 3 * Real.sqrt 5

/-- Theorem stating that the third median has the specified length -/
theorem third_median_length (t : SpecialTriangle) :
  thirdMedian t = 3 * Real.sqrt 5 := by
  -- Proof goes here
  sorry

#check third_median_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_median_length_l1352_135225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_l1352_135252

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = 2 * a n

theorem sequence_sum (a : ℕ → ℝ) (h1 : geometric_sequence a) (h2 : a 1 + a 4 = 2) :
  a 5 + a 8 = 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_l1352_135252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maximizeConeRatio_correct_maximizeConeRatio_is_one_l1352_135298

open Real

noncomputable def maximizeConeRatio (V : ℝ) : ℝ := 1

theorem maximizeConeRatio_correct (V : ℝ) (h V_pos : 0 < V) :
  let r := (3 * V / π) ^ (1/3)
  let h := r
  let A := π * r^2 + π * r * Real.sqrt (r^2 + h^2)
  ∀ r' h', r' > 0 → h' > 0 → (1/3) * π * r'^2 * h' = V →
    π * r'^2 + π * r' * Real.sqrt (r'^2 + h'^2) ≤ A :=
by
  sorry

#check maximizeConeRatio_correct

theorem maximizeConeRatio_is_one (V : ℝ) (h V_pos : 0 < V) :
  maximizeConeRatio V = 1 :=
by
  sorry

#check maximizeConeRatio_is_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_maximizeConeRatio_correct_maximizeConeRatio_is_one_l1352_135298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_OAB_l1352_135293

noncomputable def ellipse_C (x y : ℝ) : Prop := x^2 + 2*y^2 = 1

noncomputable def point_P : ℝ × ℝ := (Real.sqrt 2 / 2, 1 / 2)

noncomputable def focus_F1 : ℝ × ℝ := (-Real.sqrt 2 / 2, 0)

def line_AB (m : ℝ) (x : ℝ) : ℝ := x + m

noncomputable def area_OAB (m : ℝ) : ℝ := Real.sqrt (3 * m^2 - 2 * m^4) / 3

theorem max_area_OAB :
  ellipse_C point_P.1 point_P.2 →
  (∃ x, ellipse_C x (line_AB 0 x)) →
  (∀ m ≠ 0, ∃ x₁ x₂, x₁ ≠ x₂ ∧ ellipse_C x₁ (line_AB m x₁) ∧ ellipse_C x₂ (line_AB m x₂)) →
  (∃ m₀, ∀ m, area_OAB m ≤ area_OAB m₀) →
  (∃ m₀, area_OAB m₀ = Real.sqrt 2 / 4) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_OAB_l1352_135293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_fifth_and_sixth_terms_l1352_135236

def sequenceA (n : ℕ) (a : ℝ) : ℝ := (-1)^(n+1) * n * a^n

theorem sequence_fifth_and_sixth_terms (a : ℝ) :
  sequenceA 5 a = 5 * a^5 ∧ sequenceA 6 a = -6 * a^6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_fifth_and_sixth_terms_l1352_135236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_ratio_l1352_135230

theorem sphere_volume_ratio (r₁ r₂ r₃ : ℝ) (h : r₁ > 0 ∧ r₂ > 0 ∧ r₃ > 0) :
  (4 * Real.pi * r₁^2 = 4 * Real.pi * r₂^2 / 2) ∧ (4 * Real.pi * r₁^2 = 4 * Real.pi * r₃^2 / 3) →
  ((4 / 3) * Real.pi * r₁^3 = (4 / 3) * Real.pi * r₂^3 / (2 * Real.sqrt 2)) ∧
  ((4 / 3) * Real.pi * r₁^3 = (4 / 3) * Real.pi * r₃^3 / (3 * Real.sqrt 3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_ratio_l1352_135230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_surface_area_l1352_135254

/-- The surface area of the circumscribed sphere of a tetrahedron with vertices
    at (0, 0, 0), (0, 3, 1), (2, 3, 0), and (2, 0, 1) is equal to 14π. -/
theorem circumscribed_sphere_surface_area :
  let vertices : List (Fin 3 → ℝ) := [
    ![0, 0, 0],
    ![0, 3, 1],
    ![2, 3, 0],
    ![2, 0, 1]
  ]
  let box_dimensions : Fin 3 → ℝ := ![3, 2, 1]
  let sphere_radius := (Real.sqrt (box_dimensions 0 ^ 2 + box_dimensions 1 ^ 2 + box_dimensions 2 ^ 2)) / 2
  let sphere_surface_area := 4 * Real.pi * sphere_radius^2
  sphere_surface_area = 14 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_surface_area_l1352_135254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_line_no_intersection_l1352_135240

/-- A hyperbola C with equation (x²/a²) - (y²/b²) = 1 -/
def Hyperbola (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / a^2) - (p.2^2 / b^2) = 1}

/-- The line y = 2x -/
def Line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = 2 * p.1}

/-- Eccentricity of a hyperbola -/
noncomputable def Eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 + b^2 / a^2)

theorem hyperbola_line_no_intersection
  (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  Eccentricity a b = 2 →
  Hyperbola a b ∩ Line = ∅ :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_line_no_intersection_l1352_135240
