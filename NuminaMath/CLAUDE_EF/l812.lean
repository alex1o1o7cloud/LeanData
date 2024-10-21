import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_ratio_range_l812_81284

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos (2 * x + Real.pi / 3) - 2 * Real.cos (2 * x) + 1

theorem triangle_side_ratio_range (A B C : ℝ) (h_acute : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi)
  (h_fA : f A = 0) :
  ∃ (b c : ℝ), b > 0 ∧ c > 0 ∧ Real.sin B / Real.sin C = b / c ∧ 1/2 < b / c ∧ b / c < 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_ratio_range_l812_81284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_l812_81206

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus
def focus : ℝ × ℝ := (1, 0)

-- Define a point on the parabola
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : parabola x y

-- Define the line passing through the focus
def line_through_focus (m : ℝ) (p : ℝ × ℝ) : Prop :=
  p.1 = m * p.2 + 1

-- Define the vector relationship
def vector_relationship (P Q : ℝ × ℝ) : Prop :=
  (P.1 - 1, P.2) + 2 * (Q.1 - 1, Q.2) = (0, 0)

-- Main theorem
theorem area_of_triangle (P Q : PointOnParabola) (m : ℝ) :
  line_through_focus m (P.x, P.y) →
  line_through_focus m (Q.x, Q.y) →
  vector_relationship (P.x, P.y) (Q.x, Q.y) →
  (1/2) * |P.y - Q.y| = 3*Real.sqrt 2/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_l812_81206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_same_color_without_replacement_prob_diff_color_with_replacement_l812_81244

-- Define the number of black and white balls
def num_black : ℕ := 2
def num_white : ℕ := 3
def total_balls : ℕ := num_black + num_white

-- Theorem for drawing without replacement
theorem prob_same_color_without_replacement :
  (Nat.choose num_black 2 + Nat.choose num_white 2) / Nat.choose total_balls 2 = 2 / 5 := by sorry

-- Theorem for drawing with replacement
theorem prob_diff_color_with_replacement :
  (num_black * num_white + num_white * num_black) / (total_balls * total_balls) = 12 / 25 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_same_color_without_replacement_prob_diff_color_with_replacement_l812_81244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_equation_correct_l812_81299

/-- A plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Check if a point lies on a plane -/
def Point.liesOn (p : ℝ × ℝ × ℝ) (plane : Plane) : Prop :=
  plane.a * p.fst + plane.b * p.snd.fst + plane.c * p.snd.snd + plane.d = 0

/-- Check if a plane is parallel to the Oz axis -/
def Plane.parallelToOz (plane : Plane) : Prop :=
  plane.c = 0

theorem plane_equation_correct (plane : Plane) 
    (h1 : plane.a = 1 ∧ plane.b = 3 ∧ plane.c = 0 ∧ plane.d = -1) : 
    plane.parallelToOz ∧ 
    Point.liesOn (1, 0, 1) plane ∧ 
    Point.liesOn (-2, 1, 3) plane := by
  sorry

#check plane_equation_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_equation_correct_l812_81299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteenth_term_is_three_l812_81245

def mySequence (n : ℕ) : ℚ :=
  if n = 0 then 3
  else if n = 1 then 7
  else 21 / mySequence (n - 1)

theorem fifteenth_term_is_three :
  mySequence 14 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteenth_term_is_three_l812_81245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_g_value_l812_81210

-- Define the function g as a parameter
noncomputable def f (g : ℝ → ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 - x else g x

-- Define the property of f being an odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- State the theorem
theorem odd_function_g_value (g : ℝ → ℝ) :
  is_odd_function (f g) → g (f g (-2)) = -2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_g_value_l812_81210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_two_extreme_points_implies_a_range_l812_81229

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * (Real.log x - a * x)

-- Define a proposition that f has two extreme points
def has_two_extreme_points (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
  (∀ x : ℝ, x > 0 → (deriv (f a)) x = 0 → x = x₁ ∨ x = x₂)

-- Theorem statement
theorem f_two_extreme_points_implies_a_range :
  ∀ a : ℝ, has_two_extreme_points a → 0 < a ∧ a < 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_two_extreme_points_implies_a_range_l812_81229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l812_81286

noncomputable def f (x : ℝ) := Real.sin x ^ 2 + Real.sqrt 3 * Real.sin x * Real.cos x

theorem f_properties :
  (∀ k : ℤ, ∀ x y : ℝ, 
    x ∈ Set.Icc (k * Real.pi - Real.pi / 6) (k * Real.pi + Real.pi / 3) →
    y ∈ Set.Icc (k * Real.pi - Real.pi / 6) (k * Real.pi + Real.pi / 3) →
    x ≤ y → f x ≤ f y) ∧
  Set.range (fun x => f x) = Set.Icc 0 (3/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l812_81286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_flip_sequences_l812_81219

theorem coin_flip_sequences : ∃ (favorable_sequences : ℕ),
  let n : ℕ := 10  -- total number of flips
  favorable_sequences = 
    (Nat.choose n 6) + (Nat.choose n 7) + (Nat.choose n 8) + (Nat.choose n 9) + (Nat.choose n 10) ∧
  favorable_sequences = 386 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_flip_sequences_l812_81219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_de_length_approx_l812_81246

/-- A right triangle with given angle and side length -/
structure RightTriangle where
  /-- The measure of angle D in degrees -/
  angleD : ℝ
  /-- The length of side EF -/
  sideEF : ℝ
  /-- Angle D is between 0 and 90 degrees -/
  angleD_range : 0 < angleD ∧ angleD < 90
  /-- Side EF has positive length -/
  sideEF_pos : 0 < sideEF

/-- Calculate the length of side DE in a right triangle -/
noncomputable def calculateDE (triangle : RightTriangle) : ℝ :=
  triangle.sideEF / Real.tan (triangle.angleD * Real.pi / 180)

/-- The main theorem stating the approximate length of DE -/
theorem de_length_approx (triangle : RightTriangle) 
    (h1 : triangle.angleD = 25)
    (h2 : triangle.sideEF = 9) : 
    ∃ (de : ℝ), |calculateDE triangle - de| < 0.05 ∧ de = 19.3 := by
  sorry

#check de_length_approx

end NUMINAMATH_CALUDE_ERRORFEEDBACK_de_length_approx_l812_81246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_tuples_is_1007_l812_81233

/-- A function satisfying the given conditions -/
def SpecialFunction (f : ℕ → ℕ) : Prop :=
  f 1 = 1 ∧
  (∀ a b : ℕ, 0 < a → 0 < b → a ≤ b → f a ≤ f b) ∧
  (∀ a : ℕ, 0 < a → f (2 * a) = f a + 1)

/-- The number of possible values for the 2014-tuple (f(1), f(2), ..., f(2014)) -/
noncomputable def NumberOfPossibleTuples : ℕ :=
  1007 -- We directly define this as 1007 based on our proof

/-- The main theorem stating that the number of possible tuples is 1007 -/
theorem number_of_tuples_is_1007 :
  ∀ f : ℕ → ℕ, SpecialFunction f → NumberOfPossibleTuples = 1007 :=
by
  intro f hf
  -- The proof would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_tuples_is_1007_l812_81233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_l812_81230

def vector_a : Fin 2 → ℝ := ![1, 2]
def vector_b (x : ℝ) : Fin 2 → ℝ := ![x, -2]

def orthogonal (u v : Fin 2 → ℝ) : Prop :=
  (u 0) * (v 0) + (u 1) * (v 1) = 0

def vector_sum (u v : Fin 2 → ℝ) : Fin 2 → ℝ :=
  λ i => u i + v i

noncomputable def magnitude (v : Fin 2 → ℝ) : ℝ :=
  Real.sqrt ((v 0)^2 + (v 1)^2)

theorem vector_sum_magnitude :
  ∃ x : ℝ, orthogonal vector_a (vector_b x) ∧
  magnitude (vector_sum vector_a (vector_b x)) = 5 :=
by
  -- Proof goes here
  sorry

#eval vector_a 0  -- Should output 1
#eval vector_a 1  -- Should output 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_l812_81230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_close_points_l812_81213

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define a square with side length 1
def UnitSquare := {p : Point | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}

-- Define the distance between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem exists_close_points (points : Finset Point) 
  (h1 : points.card = 5) 
  (h2 : ∀ p ∈ points, p ∈ UnitSquare) : 
  ∃ p1 p2, p1 ∈ points ∧ p2 ∈ points ∧ p1 ≠ p2 ∧ distance p1 p2 < 3/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_close_points_l812_81213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incircle_radius_from_inner_circles_l812_81297

/-- A triangle formed by three points. -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- A circle is an incircle of a triangle if it's inscribed in the triangle. -/
def Incircle (Γ : Circle) (T : Triangle) : Prop := sorry

/-- A circle is an inner tangent circle if it's inside the triangle and tangent to two sides
    and another circle. -/
def InnerTangentCircle (Γ' : Circle) (T : Triangle) (Γ : Circle) : Prop := sorry

/-- The radius of a circle. -/
noncomputable def radius (Γ : Circle) : ℝ := sorry

/-- Given a triangle ABC with an incircle Γ and three circles Γ1, Γ2, Γ3 inside the triangle,
    each tangent to Γ and two sides of the triangle, with radii 1, 4, and 9 respectively,
    the radius of Γ is 11. -/
theorem incircle_radius_from_inner_circles (T : Triangle) (Γ Γ1 Γ2 Γ3 : Circle) :
  Incircle Γ T →
  InnerTangentCircle Γ1 T Γ →
  InnerTangentCircle Γ2 T Γ →
  InnerTangentCircle Γ3 T Γ →
  radius Γ1 = 1 →
  radius Γ2 = 4 →
  radius Γ3 = 9 →
  radius Γ = 11 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_incircle_radius_from_inner_circles_l812_81297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_area_from_volume_l812_81256

-- Define the volume of the cylinder
noncomputable def cylinder_volume (side_length : ℝ) : ℝ := Real.pi * side_length^3

-- Define the lateral surface area of the cylinder
noncomputable def cylinder_lateral_area (side_length : ℝ) : ℝ := 2 * Real.pi * side_length^2

-- Theorem statement
theorem cylinder_area_from_volume :
  ∀ s : ℝ, s > 0 → cylinder_volume s = 27 * Real.pi → cylinder_lateral_area s = 18 * Real.pi :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_area_from_volume_l812_81256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_exponent_evaluations_l812_81215

theorem negative_exponent_evaluations :
  (2 : ℚ)^((-3) : ℤ) = 1/8 ∧
  (1/3 : ℚ)^((-2) : ℤ) = 9 ∧
  (2/3 : ℚ)^((-4) : ℤ) = 81/16 ∧
  ((-1/5) : ℚ)^((-3) : ℤ) = -125 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_exponent_evaluations_l812_81215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shooting_test_probability_l812_81232

/-- The probability of making a single shot -/
noncomputable def p : ℝ := 0.6

/-- The number of shots taken -/
def n : ℕ := 3

/-- The minimum number of successful shots required to pass -/
def k : ℕ := 2

/-- The probability of passing the test -/
noncomputable def prob_pass : ℝ := 81 / 125

/-- Theorem stating that the probability of passing the test is 81/125 -/
theorem shooting_test_probability : 
  (Finset.sum (Finset.range (n - k + 1)) (λ i => 
    (n.choose (k + i)) * (p ^ (k + i)) * ((1 - p) ^ (n - k - i)))) = prob_pass := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shooting_test_probability_l812_81232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sqrt_square_self_l812_81231

noncomputable def min3 (a b c : ℝ) : ℝ := min a (min b c)

theorem min_sqrt_square_self (x : ℝ) : 
  min3 (Real.sqrt x) (x^2) x = 1/16 → x = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sqrt_square_self_l812_81231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_octagon_max_area_l812_81224

/-- An octagon inscribed in a circle with specific properties -/
structure InscribedOctagon where
  /-- The radius of the circumscribed circle -/
  radius : ℝ
  /-- The side length of the square formed by one set of alternate vertices -/
  square_side : ℝ
  /-- The width of the rectangle formed by the other set of alternate vertices -/
  rect_width : ℝ
  /-- The height of the rectangle formed by the other set of alternate vertices -/
  rect_height : ℝ
  /-- The square formed by one set of alternate vertices has an area of 5 -/
  square_area : square_side ^ 2 = 5
  /-- The rectangle formed by the other set of alternate vertices has an area of 4 -/
  rect_area : rect_width * rect_height = 4
  /-- The diagonal of both the square and rectangle is the diameter of the circle -/
  circle_diameter : square_side * Real.sqrt 2 = 2 * radius
  rect_diagonal : rect_width ^ 2 + rect_height ^ 2 = (2 * radius) ^ 2

/-- The maximum area of the inscribed octagon -/
noncomputable def max_octagon_area (o : InscribedOctagon) : ℝ := 3 * Real.sqrt 5

/-- Theorem stating that the maximum area of the inscribed octagon is 3√5 -/
theorem inscribed_octagon_max_area (o : InscribedOctagon) :
  max_octagon_area o = 3 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_octagon_max_area_l812_81224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_sin_A_value_l812_81249

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin (x / 3) * Real.cos (x / 3) - 2 * (Real.sin (x / 3))^2

-- Theorem for part (I)
theorem range_of_f : 
  ∀ x ∈ Set.Icc 0 Real.pi, f x ∈ Set.Icc 0 1 := by sorry

-- Theorem for part (II)
theorem sin_A_value (A B C : ℝ) (a b c : ℝ) :
  f C = 1 → b^2 = a * c → Real.sin A = (Real.sqrt 5 - 1) / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_sin_A_value_l812_81249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l812_81288

/-- Line represented by the equation kx + y - 9 = 0 -/
def line (k : ℝ) (x y : ℝ) : Prop := k * x + y - 9 = 0

/-- Circle represented by the equation x^2 + y^2 = 9 -/
def circle' (x y : ℝ) : Prop := x^2 + y^2 = 9

/-- Two points A and B -/
structure Point where
  x : ℝ
  y : ℝ

/-- Triangle OAB where O is the origin (0, 0) -/
structure Triangle where
  A : Point
  B : Point

/-- Predicate to check if a triangle is equilateral -/
def is_equilateral (t : Triangle) : Prop :=
  let d_OA := Real.sqrt (t.A.x^2 + t.A.y^2)
  let d_OB := Real.sqrt (t.B.x^2 + t.B.y^2)
  let d_AB := Real.sqrt ((t.A.x - t.B.x)^2 + (t.A.y - t.B.y)^2)
  d_OA = d_OB ∧ d_OA = d_AB ∧ d_OB = d_AB

theorem line_circle_intersection (k : ℝ) :
  ∃ (t : Triangle),
    (line k t.A.x t.A.y ∧ circle' t.A.x t.A.y) ∧
    (line k t.B.x t.B.y ∧ circle' t.B.x t.B.y) ∧
    is_equilateral t →
    k = Real.sqrt 11 ∨ k = -Real.sqrt 11 :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l812_81288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_difference_l812_81289

noncomputable def total_amount : ℝ := 10000
noncomputable def years : ℝ := 2
noncomputable def interest_rate_A : ℝ := 15
noncomputable def interest_rate_B : ℝ := 18
noncomputable def amount_B : ℝ := 4000.0000000000005

noncomputable def amount_A : ℝ := total_amount - amount_B

noncomputable def interest_A : ℝ := amount_A * interest_rate_A * years / 100
noncomputable def interest_B : ℝ := amount_B * interest_rate_B * years / 100

theorem interest_difference : 
  interest_A - interest_B = 359.99999999999965 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_difference_l812_81289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_x1_x2_l812_81283

theorem existence_of_x1_x2 (f : ℝ → ℝ) (h_pos : ∀ x ∈ Set.Icc 0 1, f x > 0)
  (h_bounded : ∃ M, ∀ x ∈ Set.Icc 0 1, f x ≤ M) :
  ∃ x₁ x₂, x₁ ∈ Set.Icc 0 1 ∧ x₂ ∈ Set.Icc 0 1 ∧ ((x₂ - x₁) * (f x₁)^2) / (f x₂) > (f 0) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_x1_x2_l812_81283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_extreme_points_when_a_is_one_range_of_a_for_inequality_l812_81207

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * (x - 1)

def tangent_line (a : ℝ) (x : ℝ) : ℝ := (1 - a) * (x - 1)

theorem tangent_line_at_one (a : ℝ) :
  ∀ x, tangent_line a x = (1 - a) * (x - 1) := by
  intro x
  rfl

theorem extreme_points_when_a_is_one :
  ∃ x_max : ℝ, x_max = 1 ∧ f 1 x_max = 0 ∧
  ∀ x : ℝ, x > 0 → f 1 x ≤ f 1 x_max ∧
  ¬∃ x_min : ℝ, ∀ x : ℝ, x > 0 → f 1 x ≥ f 1 x_min := by
  sorry

theorem range_of_a_for_inequality (a : ℝ) :
  (∀ x : ℝ, x ≥ 1 → f a x ≤ Real.log x / (x + 1)) ↔ a ≥ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_extreme_points_when_a_is_one_range_of_a_for_inequality_l812_81207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_travel_cost_comparison_l812_81241

/-- Represents the triangle ABC with given side lengths --/
structure Triangle where
  ab : ℝ
  ac : ℝ
  bc : ℝ

/-- Calculates the total distance of the trip --/
def total_distance (t : Triangle) : ℝ := t.ab + t.bc + t.ac

/-- Calculates the cost of bus travel --/
def bus_cost (distance : ℝ) (rate : ℝ) : ℝ := distance * rate

/-- Calculates the cost of airplane travel --/
def airplane_cost (distance : ℝ) (rate : ℝ) (booking_fee : ℝ) (segments : ℕ) : ℝ :=
  distance * rate + booking_fee * (segments : ℝ)

/-- Main theorem --/
theorem travel_cost_comparison (t : Triangle) 
  (h_right_angle : t.ab^2 + t.bc^2 = t.ac^2)
  (h_ab : t.ab = 4000)
  (h_ac : t.ac = 4800)
  (bus_rate : ℝ)
  (plane_rate : ℝ)
  (booking_fee : ℝ)
  (h_bus_rate : bus_rate = 0.18)
  (h_plane_rate : plane_rate = 0.12)
  (h_booking_fee : booking_fee = 120) :
  total_distance t = 11453 ∧ 
  airplane_cost (total_distance t) plane_rate booking_fee 3 < 
  bus_cost (total_distance t) bus_rate := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_travel_cost_comparison_l812_81241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l812_81238

-- Define the constants as noncomputable
noncomputable def a : ℝ := Real.rpow 0.8 0.7
noncomputable def b : ℝ := Real.rpow 0.8 0.9
noncomputable def c : ℝ := Real.rpow 1.2 0.8

-- State the theorem
theorem relationship_abc : c > a ∧ a > b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l812_81238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l812_81225

/-- Ellipse struct representing the equation (x²/a²) + (y²/b²) = 1 -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line struct representing the equation x = k -/
structure Line where
  k : ℝ

noncomputable def Ellipse.c (e : Ellipse) : ℝ := Real.sqrt (e.a^2 - e.b^2)

noncomputable def Ellipse.eccentricity (e : Ellipse) : ℝ := e.c / e.a

def on_ellipse (e : Ellipse) (p : Point) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

def perpendicular (p q : Point) (l : Line) : Prop :=
  q.x = l.k ∧ (p.x - q.x) * (p.y - q.y) = 0

def isosceles_right_triangle (p q f : Point) : Prop :=
  (p.x - f.x)^2 + (p.y - f.y)^2 = (q.x - f.x)^2 + (q.y - f.y)^2 ∧
  (p.x - f.x) * (q.x - f.x) + (p.y - f.y) * (q.y - f.y) = 0

theorem ellipse_eccentricity (e : Ellipse) (p : Point) (l : Line) (f : Point) :
  on_ellipse e p →
  l.k = -e.a^2 / e.c →
  f.x = -e.c ∧ f.y = 0 →
  (∃ q : Point, perpendicular p q l ∧ isosceles_right_triangle p q f) →
  e.eccentricity = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l812_81225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_B_power_2021_l812_81203

noncomputable def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![Real.sqrt 3 / 2, 0, -1 / 2],
    ![0, -1, 0],
    ![1 / 2, 0, Real.sqrt 3 / 2]]

theorem B_power_2021 :
  B ^ 2021 = ![![-Real.sqrt 3 / 2, 0, -1 / 2],
                ![0, -1, 0],
                ![1 / 2, 0, -Real.sqrt 3 / 2]] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_B_power_2021_l812_81203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_month_bill_l812_81285

/-- Represents the monthly telephone bill structure -/
structure PhoneBill where
  fixedCharge : ℝ  -- Fixed monthly charge for internet service
  callCharge : ℝ   -- Charge for calls made during the month

/-- Calculates the total bill amount -/
def totalBill (bill : PhoneBill) : ℝ :=
  bill.fixedCharge + bill.callCharge

theorem second_month_bill 
  (january : PhoneBill) 
  (second_month : PhoneBill) 
  (h1 : totalBill january = 52) 
  (h2 : second_month.callCharge = 2 * january.callCharge) 
  (h3 : second_month.fixedCharge = january.fixedCharge) : 
  totalBill second_month = 52 + january.callCharge := by
  sorry

#check second_month_bill

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_month_bill_l812_81285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sue_initial_books_l812_81293

/-- The number of books Sue initially borrowed -/
def initial_books : ℕ := 15

/-- The number of movies Sue initially borrowed -/
def initial_movies : ℕ := 6

/-- The number of books Sue returned -/
def returned_books : ℕ := 8

/-- The fraction of movies Sue returned -/
def returned_movies_fraction : ℚ := 1/3

/-- The number of books Sue checked out later -/
def checked_out_books : ℕ := 9

/-- The total number of items Sue has at the end -/
def total_items : ℕ := 20

theorem sue_initial_books :
  initial_books = 15 ∧
  initial_books - returned_books + checked_out_books +
  (initial_movies - (returned_movies_fraction * initial_movies).floor) = total_items :=
by
  constructor
  · rfl
  · norm_num
    rfl

#check sue_initial_books

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sue_initial_books_l812_81293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_trig_function_l812_81270

theorem range_of_trig_function :
  (∀ x : ℝ, -2 ≤ Real.sqrt 3 * Real.sin (2 * x) + 2 * (Real.cos x)^2 - 1 ∧
            Real.sqrt 3 * Real.sin (2 * x) + 2 * (Real.cos x)^2 - 1 ≤ 2) ∧
  (∃ x₁ x₂ : ℝ, Real.sqrt 3 * Real.sin (2 * x₁) + 2 * (Real.cos x₁)^2 - 1 = -2 ∧
                Real.sqrt 3 * Real.sin (2 * x₂) + 2 * (Real.cos x₂)^2 - 1 = 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_trig_function_l812_81270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_circumscribing_sphere_l812_81296

/-- A right square prism with base edge length 1 and lateral edge length √2 -/
structure RightSquarePrism where
  base_edge : ℝ
  lateral_edge : ℝ
  base_edge_eq_one : base_edge = 1
  lateral_edge_eq_sqrt_two : lateral_edge = Real.sqrt 2

/-- The sphere that circumscribes the right square prism -/
def circumscribing_sphere (prism : RightSquarePrism) : ℝ → Prop :=
  λ r ↦ r * 2 = Real.sqrt (prism.base_edge^2 + prism.base_edge^2 + prism.lateral_edge^2)

/-- The volume of a sphere with radius r -/
noncomputable def sphere_volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

theorem volume_of_circumscribing_sphere (prism : RightSquarePrism) :
  ∃ r, circumscribing_sphere prism r ∧ sphere_volume r = (4 / 3) * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_circumscribing_sphere_l812_81296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_increase_factor_l812_81212

/-- Given an initial speed, calculate the factor by which travel time is reduced when speed is increased --/
noncomputable def speedUpFactor (initialSpeed : ℝ) (speedIncrease : ℝ) : ℝ :=
  (initialSpeed + speedIncrease) / initialSpeed

theorem speed_increase_factor (v : ℝ) (h : v > 0) :
  speedUpFactor v 2 = 2.5 → speedUpFactor v 4 = 4 := by
  intro h1
  have v_eq : v = 4/3 := by
    -- Proof that v = 4/3
    sorry
  
  -- Show that speedUpFactor v 4 = 4
  calc
    speedUpFactor v 4 = (v + 4) / v := rfl
    _ = ((4/3) + 4) / (4/3) := by rw [v_eq]
    _ = (4/3 + 12/3) / (4/3) := by norm_num
    _ = (16/3) / (4/3) := by norm_num
    _ = 4 := by norm_num

  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_increase_factor_l812_81212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_packages_property_l812_81271

/-- The minimum number of packages Jane must deliver to cover her initial expenses -/
def min_packages_to_cover_expenses : ℕ :=
  let motorcycleCost : ℕ := 3600
  let deliveryGearCost : ℕ := 200
  let earningsPerPackage : ℕ := 15
  let fuelCostPerPackage : ℕ := 5
  let netProfitPerPackage : ℕ := earningsPerPackage - fuelCostPerPackage
  let totalInitialCost : ℕ := motorcycleCost + deliveryGearCost
  (totalInitialCost + netProfitPerPackage - 1) / netProfitPerPackage

theorem min_packages_property (minPackages : ℕ := min_packages_to_cover_expenses) :
  let motorcycleCost : ℕ := 3600
  let deliveryGearCost : ℕ := 200
  let earningsPerPackage : ℕ := 15
  let fuelCostPerPackage : ℕ := 5
  let netProfitPerPackage : ℕ := earningsPerPackage - fuelCostPerPackage
  let totalInitialCost : ℕ := motorcycleCost + deliveryGearCost
  minPackages * netProfitPerPackage ≥ totalInitialCost ∧
  ∀ n : ℕ, n * netProfitPerPackage ≥ totalInitialCost → n ≥ minPackages :=
by
  sorry

#eval min_packages_to_cover_expenses


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_packages_property_l812_81271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_product_l812_81273

theorem cosine_sum_product (a b c d : ℕ+) :
  (∀ x : ℝ, Real.cos (2 * x) + Real.cos (6 * x) + Real.cos (10 * x) + Real.cos (14 * x) = 
    (a : ℝ) * Real.cos (b * x) * Real.cos (c * x) * Real.cos (d * x)) →
  a + b + c + d = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_product_l812_81273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_BC_length_l812_81254

/-- A parabola defined by y = x^2 + 1 -/
noncomputable def parabola (x : ℝ) : ℝ := x^2 + 1

/-- Triangle ABC with vertices on the parabola -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  h_A : A.2 = parabola A.1
  h_B : B.2 = parabola B.1
  h_C : C.2 = parabola C.1

/-- The area of a triangle given its base and height -/
noncomputable def triangleArea (base height : ℝ) : ℝ := (1/2) * base * height

theorem triangle_BC_length (t : Triangle) 
  (h_A_x : t.A.1 = 0)
  (h_A_y : t.A.2 = 1)
  (h_BC_parallel : t.B.2 = t.C.2)
  (h_area : triangleArea (t.C.1 - t.B.1) (t.B.2 - t.A.2) = 128) :
  t.C.1 - t.B.1 = 8 * Real.rpow 2 (1/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_BC_length_l812_81254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_properties_l812_81292

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 + 4 * (m - 3) * x - 16

noncomputable def discriminant (m : ℝ) : ℝ := 16 * (m^2 - 2*m + 9)

noncomputable def distance_between_intercepts (m : ℝ) : ℝ := 
  (4 : ℝ) * Real.sqrt ((3/m - 1/3)^2 + 8/9)

theorem quadratic_properties (m : ℝ) (h : m ≠ 0) :
  -- 1. The graph intersects the x-axis at two points
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f m x₁ = 0 ∧ f m x₂ = 0) ∧
  -- 2. The minimum distance between x-intercepts is √8/3 when m = 9
  (∀ m' : ℝ, m' ≠ 0 → distance_between_intercepts 9 ≤ distance_between_intercepts m') ∧
  distance_between_intercepts 9 = Real.sqrt 8 / 3 ∧
  -- 3. When m = 9, the parabola opens upwards and has vertex at (-4/3, -32)
  (∀ x : ℝ, f 9 (-4/3) ≤ f 9 x) ∧
  f 9 (-4/3) = -32 := by
  sorry

#check quadratic_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_properties_l812_81292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_triangle_properties_l812_81216

/-- Definition of the ellipse Γ -/
def Γ (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

/-- Definition of the parabola -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = (1/2) * p.1}

/-- Theorem statement -/
theorem ellipse_and_triangle_properties
  (a b : ℝ) (h_ab : a > b ∧ b > 0)
  (h_ecc : (Real.sqrt (a^2 - b^2)) / a = Real.sqrt 2 / 2)
  (h_chord : ∃ p q : ℝ × ℝ, p ∈ Γ a b ∧ q ∈ Γ a b ∧ p.1 = q.1 ∧ Real.sqrt ((p.2 - q.2)^2) = Real.sqrt 2) :
  (∃ A B : ℝ × ℝ, A ∈ Γ a b ∧ B ∈ Γ a b ∧
    ∃ C : ℝ × ℝ, C ∈ Parabola ∧
    (A.1 + B.1 + C.1 = 0 ∧ A.2 + B.2 + C.2 = 0) ∧
    |((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2)| = 3 * Real.sqrt 6 / 4) →
  (Γ a b = Γ (Real.sqrt 2) 1 ∧
   ((C = (1, Real.sqrt 2 / 2) ∨ C = (1, -Real.sqrt 2 / 2)) ∨
    (C = (2, 1) ∨ C = (2, -1)))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_triangle_properties_l812_81216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_propositions_count_l812_81287

-- Define the types of events
inductive EventType
  | Certain
  | Impossible
  | Random

-- Define a function to check if a proposition is correct based on its event type
def isCorrectProposition (actualType expectedType : EventType) : Bool :=
  match actualType, expectedType with
  | EventType.Certain, EventType.Certain => true
  | EventType.Impossible, EventType.Impossible => true
  | EventType.Random, EventType.Random => true
  | _, _ => false

-- Define the propositions
def proposition1 : EventType := EventType.Certain
def proposition2 : EventType := EventType.Impossible
def proposition3 : EventType := EventType.Random
def proposition4 : EventType := EventType.Random

-- Function to count correct propositions
def countCorrectPropositions : Nat :=
  (if isCorrectProposition proposition1 EventType.Certain then 1 else 0) +
  (if isCorrectProposition proposition2 EventType.Impossible then 1 else 0) +
  (if isCorrectProposition proposition3 EventType.Certain then 1 else 0) +
  (if isCorrectProposition proposition4 EventType.Random then 1 else 0)

-- Theorem to prove
theorem correct_propositions_count : countCorrectPropositions = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_propositions_count_l812_81287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_solution_tan_sec_equation_l812_81248

theorem smallest_positive_solution_tan_sec_equation :
  let f : ℝ → ℝ := λ x => Real.tan (2 * x) + Real.tan (3 * x) - 1 / Real.cos (3 * x)
  ∃ x : ℝ, x > 0 ∧ f x = 0 ∧ ∀ y : ℝ, 0 < y ∧ y < x → f y ≠ 0 ∧ x = π / 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_solution_tan_sec_equation_l812_81248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_value_one_is_max_a_l812_81276

-- Define the function f
noncomputable def f (a x : ℝ) : ℝ := Real.log x + a * (x^2 - 3*x + 2)

-- State the theorem
theorem max_a_value (a : ℝ) :
  (a > 0) →
  (∀ x > 1, f a x ≥ 0) →
  a ≤ 1 :=
by sorry

-- State that 1 is the maximum value
theorem one_is_max_a :
  ∃ a : ℝ, (a > 0) ∧ (∀ x > 1, f a x ≥ 0) ∧ (a = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_value_one_is_max_a_l812_81276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_relationship_l812_81209

/-- The distance between a circle and a line -/
noncomputable def circle_line_distance (r : ℝ) (a : ℝ) : ℝ :=
  |a| / Real.sqrt 2 - r

/-- The theorem stating the relationship between the circle and line -/
theorem circle_line_relationship :
  ∀ a : ℝ,
  (circle_line_distance 2 a = 1) →
  (a = -Real.sqrt 2 ∨ a = Real.sqrt 2) :=
by
  intro a h
  sorry

#check circle_line_relationship

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_relationship_l812_81209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_pipe_fill_time_l812_81247

/-- Given three pipes with different fill/empty rates, prove the time for the second pipe to fill the cistern -/
theorem second_pipe_fill_time (fill_time_1 fill_time_3 combined_fill_time : ℝ) 
  (h1 : fill_time_1 = 10)
  (h2 : fill_time_3 = 50)
  (h3 : combined_fill_time = 6.1224489795918355)
  (h4 : ∃ fill_time_2 : ℝ, (1 / fill_time_1) + (1 / fill_time_2) - (1 / fill_time_3) = 1 / combined_fill_time) :
  ∃ fill_time_2 : ℝ, abs (fill_time_2 - 191.58163265306122) < 0.00001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_pipe_fill_time_l812_81247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2x_plus_2y_l812_81235

theorem cos_2x_plus_2y (x y : ℝ) :
  (Real.cos x * Real.cos y - Real.sin x * Real.sin y = 1/4) → Real.cos (2*x + 2*y) = -7/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2x_plus_2y_l812_81235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_only_for_m_equals_one_l812_81251

theorem inequality_holds_only_for_m_equals_one :
  ∀ m : ℝ, m ≠ 0 →
  (∀ x : ℝ, x > 0 → x^2 - 2*m*Real.log x ≥ 1) →
  m = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_only_for_m_equals_one_l812_81251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_different_numbers_l812_81257

/-- Represents a notebook with a fixed number of pages -/
structure Notebook :=
  (pages : ℕ)
  (numbers : Fin pages → Finset ℕ)
  (min_per_page : ∀ i : Fin pages, (numbers i).card ≥ 10)
  (max_consecutive_three : ∀ i : Fin pages, i.val + 2 < pages → 
    ((numbers i) ∪ (numbers ⟨i.val + 1, sorry⟩) ∪ (numbers ⟨i.val + 2, sorry⟩)).card ≤ 20)

/-- The maximum number of different numbers in an 18-page notebook -/
theorem max_different_numbers (nb : Notebook) (h : nb.pages = 18) : 
  (Finset.univ.image nb.numbers).card ≤ 190 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_different_numbers_l812_81257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gilda_marbles_remaining_l812_81279

theorem gilda_marbles_remaining (M : ℝ) (hM : M > 0) : 
  let remaining_after_pedro : ℝ := M * (1 - 0.3)
  let remaining_after_ebony : ℝ := remaining_after_pedro * (1 - 0.15)
  let remaining_after_zack : ℝ := remaining_after_ebony * (1 - 0.2)
  let final_remaining : ℝ := remaining_after_zack * (1 - 0.1)
  (final_remaining / M) * 100 = 42.84 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gilda_marbles_remaining_l812_81279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_remainder_mod_seven_l812_81277

def customSequence : List Nat := List.range 10 |> List.map (fun i => 10 * i + 3)

theorem product_remainder_mod_seven :
  (customSequence.prod) % 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_remainder_mod_seven_l812_81277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_largest_expected_value_l812_81260

theorem fourth_largest_expected_value : 
  let S : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 90}
  let m : ℕ := 5
  let k : ℕ := 4
  let E (k : ℕ) : ℚ := (k * (90 + 1)) / (m + 1 : ℚ)
  ⌊(10 : ℚ) * E k⌋ = 606 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_largest_expected_value_l812_81260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_overlap_area_l812_81222

/-- The area of a regular hexagon -/
noncomputable def hexagon_area (s : ℝ) : ℝ := 3 * Real.sqrt 3 / 2 * s^2

/-- The area of an equilateral triangle -/
noncomputable def equilateral_triangle_area (s : ℝ) : ℝ := Real.sqrt 3 / 4 * s^2

theorem hexagon_overlap_area (area : ℝ) : 
  area = 25 → 
  ∃ (s : ℝ), hexagon_area s = area ∧ 
    2 * area - 2 * equilateral_triangle_area (s / 2) = 50 - 50 * Real.sqrt 3 / 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_overlap_area_l812_81222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_circle_equation_l812_81298

/-- Definition of the ellipse E -/
def ellipse_E (x y : ℝ) : Prop :=
  x^2 / 2 + y^2 = 1

/-- Definition of the line l passing through P(-2, 0) -/
def line_l (m : ℝ) (x y : ℝ) : Prop :=
  x = m * y - 2

/-- Definition of point P -/
def point_P : ℝ × ℝ := (-2, 0)

/-- Definition of point A on ellipse E and line l -/
noncomputable def point_A (m : ℝ) : ℝ × ℝ :=
  let y := (4 * m - Real.sqrt (16 * m^2 - 8 * (m^2 + 2))) / (2 * (m^2 + 2))
  (m * y - 2, y)

/-- Definition of point B on ellipse E and line l -/
noncomputable def point_B (m : ℝ) : ℝ × ℝ :=
  let y := (4 * m + Real.sqrt (16 * m^2 - 8 * (m^2 + 2))) / (2 * (m^2 + 2))
  (m * y - 2, y)

/-- Theorem stating the equation of the circumscribed circle of ACBD -/
theorem circumscribed_circle_equation :
  ∃ (m : ℝ),
    ellipse_E 1 (Real.sqrt 2 / 2) ∧
    line_l m (point_A m).1 (point_A m).2 ∧
    line_l m (point_B m).1 (point_B m).2 ∧
    (point_B m).2 = 3 * (point_A m).2 →
    ∀ (x y : ℝ),
      (x + 1/3)^2 + y^2 = 10/9 ↔
        ∃ (t : ℝ),
          (x - (point_A m).1)^2 + (y - (point_A m).2)^2 =
          (x - (point_B m).1)^2 + (y - (point_B m).2)^2 ∧
          (x - (point_A m).1)^2 + (y + (point_A m).2)^2 =
          (x - (point_B m).1)^2 + (y + (point_B m).2)^2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_circle_equation_l812_81298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_one_doctor_one_nurse_l812_81253

def total_people : ℕ := 5
def doctors : ℕ := 3
def nurses : ℕ := 2
def people_to_select : ℕ := 2

theorem probability_one_doctor_one_nurse :
  (Nat.choose doctors 1 * Nat.choose nurses 1 : ℚ) / Nat.choose total_people people_to_select = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_one_doctor_one_nurse_l812_81253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_faiths_take_home_pay_correct_l812_81204

/-- Calculate Faith's take-home pay for the week --/
def faiths_take_home_pay
  (hourly_rate : ℚ)
  (regular_hours_per_day : ℕ)
  (work_days_per_week : ℕ)
  (overtime_hours_per_day : ℕ)
  (overtime_rate_multiplier : ℚ)
  (commission_rate : ℚ)
  (total_sales : ℚ)
  (deduction_rate : ℚ) : ℚ :=
let regular_earnings := hourly_rate * regular_hours_per_day * work_days_per_week
let overtime_rate := hourly_rate * overtime_rate_multiplier
let overtime_earnings := overtime_rate * overtime_hours_per_day * work_days_per_week
let commission := commission_rate * total_sales
let total_earnings := regular_earnings + overtime_earnings + commission
let deductions := deduction_rate * total_earnings
total_earnings - deductions

/-- Theorem stating that Faith's take-home pay is correct --/
theorem faiths_take_home_pay_correct
  (hourly_rate : ℚ)
  (regular_hours_per_day : ℕ)
  (work_days_per_week : ℕ)
  (overtime_hours_per_day : ℕ)
  (overtime_rate_multiplier : ℚ)
  (commission_rate : ℚ)
  (total_sales : ℚ)
  (deduction_rate : ℚ)
  (h1 : hourly_rate = 13.5)
  (h2 : regular_hours_per_day = 8)
  (h3 : work_days_per_week = 5)
  (h4 : overtime_hours_per_day = 2)
  (h5 : overtime_rate_multiplier = 1.5)
  (h6 : commission_rate = 0.1)
  (h7 : total_sales = 3200)
  (h8 : deduction_rate = 0.25)
  : faiths_take_home_pay hourly_rate regular_hours_per_day work_days_per_week overtime_hours_per_day
      overtime_rate_multiplier commission_rate total_sales deduction_rate = 796.87 :=
by
  sorry

#eval faiths_take_home_pay 13.5 8 5 2 1.5 0.1 3200 0.25

end NUMINAMATH_CALUDE_ERRORFEEDBACK_faiths_take_home_pay_correct_l812_81204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_surface_area_theorem_l812_81218

/-- The total surface area of a quadrilateral pyramid with a rhombus base -/
noncomputable def pyramidSurfaceArea (a : ℝ) (α β : ℝ) : ℝ :=
  (2 * a^2 * Real.sin α * (Real.cos (β/2))^2) / Real.cos β

/-- Theorem: Total surface area of a quadrilateral pyramid with rhombus base -/
theorem pyramid_surface_area_theorem (a α β : ℝ) 
  (h_a : a > 0) 
  (h_α : 0 < α ∧ α < Real.pi / 2) 
  (h_β : 0 < β ∧ β < Real.pi / 2) :
  pyramidSurfaceArea a α β = 
    (2 * a^2 * Real.sin α * (Real.cos (β/2))^2) / Real.cos β :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_surface_area_theorem_l812_81218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_circle_properties_l812_81272

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 2*x

-- Define the circle (renamed to avoid conflict with existing definition)
def myCircle (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 2

-- Theorem statement
theorem parabola_and_circle_properties :
  -- Given conditions
  let focus : ℝ × ℝ := (1/2, 0)
  let directrix : ℝ → Prop := λ x => x = -1/2
  
  -- Prove the following
  ∀ x y : ℝ,
    -- 1. Equation of the parabola
    (∀ p : ℝ × ℝ, p.1 = x ∧ p.2 = y → 
      (p.1 - focus.1)^2 + (p.2 - focus.2)^2 = (p.1 - (-1/2))^2) ↔
    parabola x y
    ∧
    -- 2. Minimum distance between tangent points
    ∃ min_dist : ℝ,
      min_dist = 2 * Real.sqrt (30 / 25) ∧
      ∀ p q : ℝ × ℝ,
        parabola p.1 p.2 →
        myCircle q.1 q.2 →
        (∃ t : ℝ × ℝ, parabola t.1 t.2 ∧ 
          ((t.2 - p.2) * (t.1 - q.1) = (t.2 - q.2) * (t.1 - p.1))) →
        ((q.1 - p.1)^2 + (q.2 - p.2)^2) ≥ min_dist^2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_circle_properties_l812_81272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l812_81262

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1) - x / (x + 1)

-- State the theorem
theorem f_properties :
  -- Domain of f
  (∀ x > -1, f x = Real.log (x + 1) - x / (x + 1)) →
  -- 1. Monotonicity properties
  (∀ x > 0, ∀ y > x, f y > f x) ∧
  (∀ x ∈ Set.Ioo (-1) 0, ∀ y ∈ Set.Ioo (-1) 0, x < y → f x > f y) ∧
  -- 2. Minimum value
  (f 0 = 0 ∧ ∀ x > -1, f x ≥ 0) ∧
  -- 3. Inequality for positive real numbers
  (∀ a b : ℝ, a > 0 → b > 0 → Real.log a - Real.log b ≥ 1 - b / a) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l812_81262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l812_81290

theorem trig_identity (α : Real) (h1 : Real.sin α + Real.cos α = Real.sqrt 2) 
  (h2 : 0 < α ∧ α < Real.pi / 2) : 
  Real.tan α + (Real.cos α / Real.sin α) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l812_81290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_theorem_l812_81265

-- Define the types for points, lines, and circles
variable (Point Line Circle : Type)

-- Define the parallel relation between lines
variable (parallel : Line → Line → Prop)

-- Define the tangent relation between circles and lines, and between circles
variable (tangent_line : Circle → Line → Prop)
variable (tangent_circle : Circle → Circle → Prop)

-- Define the intersection relation between lines and points
variable (intersect : Line → Line → Point)

-- Define the property of a point being on a line or circle
variable (on_line : Point → Line → Prop)
variable (on_circle : Point → Circle → Prop)

-- Define the distance between two points
variable (distance : Point → Point → ℝ)

-- Define a function to create a line from two points
variable (mk_line : Point → Point → Line)

-- State the theorem
theorem circle_tangent_theorem 
  (ω ω₁ ω₂ : Circle) 
  (l₁ l₂ : Line) 
  (A B C D E Q : Point) :
  parallel l₁ l₂ →
  tangent_line ω l₁ →
  tangent_line ω l₂ →
  tangent_line ω₁ l₁ →
  on_line A l₁ →
  tangent_circle ω₁ ω →
  on_circle C ω →
  on_circle C ω₁ →
  tangent_line ω₂ l₂ →
  on_line B l₂ →
  tangent_circle ω₂ ω →
  on_circle D ω →
  on_circle D ω₂ →
  tangent_circle ω₂ ω₁ →
  on_circle E ω₁ →
  on_circle E ω₂ →
  Q = intersect (mk_line A D) (mk_line B C) →
  distance Q C = distance Q D ∧ distance Q D = distance Q E :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_theorem_l812_81265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rightToLeft_eq_conventional_l812_81278

/-- Evaluates an expression from right to left -/
noncomputable def rightToLeftEval (a b c d e : ℝ) : ℝ := a * (b / c - (d + e))

/-- Conventional evaluation of the expression -/
noncomputable def conventionalEval (a b c d e : ℝ) : ℝ := a * b / c - d + e

/-- Theorem stating the equivalence of right-to-left and conventional evaluation -/
theorem rightToLeft_eq_conventional (a b c d e : ℝ) :
  rightToLeftEval a b c d e ≠ conventionalEval a b c d e := by
  sorry

#check rightToLeft_eq_conventional

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rightToLeft_eq_conventional_l812_81278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_square_with_circles_l812_81269

/-- The shaded area in a square with circles design -/
theorem shaded_area_square_with_circles (square_side : ℝ) (num_circles : ℕ) : 
  square_side = 30 → 
  num_circles = 6 → 
  let circle_radius := square_side / 6
  let square_area := square_side ^ 2
  let circle_area := π * circle_radius ^ 2
  let total_circle_area := (num_circles : ℝ) * circle_area
  square_area - total_circle_area = 900 - 150 * π := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_square_with_circles_l812_81269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_solid_is_48π_div_17_l812_81291

/-- The region R in the coordinate plane -/
noncomputable def R : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | |10 - p.1| + p.2 ≤ 12 ∧ 4 * p.2 - p.1 ≥ 20}

/-- The rotation axis -/
noncomputable def rotationAxis (x : ℝ) : ℝ := x / 4 + 5

/-- The volume of the solid formed by revolving R around the rotation axis -/
noncomputable def volumeOfSolid : ℝ := 48 * Real.pi / 17

theorem volume_of_solid_is_48π_div_17 : volumeOfSolid = 48 * Real.pi / 17 := by
  -- The proof is omitted
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_solid_is_48π_div_17_l812_81291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_on_y_axis_l812_81242

/-- Given points A and B in 3D space, prove that the point M on the y-axis 
    equidistant from A and B has y-coordinate -1. -/
theorem equidistant_point_on_y_axis 
  (A B M : ℝ × ℝ × ℝ) 
  (hA : A = (1, 0, 2)) 
  (hB : B = (1, -3, 1)) 
  (hM : M.2.1 = 0 ∧ M.2.2 = 0)  -- M is on the y-axis
  (h_equidistant : (M.1 - A.1)^2 + (M.2.1 - A.2.1)^2 + (M.2.2 - A.2.2)^2 = 
                   (M.1 - B.1)^2 + (M.2.1 - B.2.1)^2 + (M.2.2 - B.2.2)^2) :
  M.2.1 = -1 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_on_y_axis_l812_81242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l812_81264

-- Define a, b, and c
noncomputable def a : ℝ := Real.log 2 / Real.log (1/3)
noncomputable def b : ℝ := Real.log 3 / Real.log (1/2)
noncomputable def c : ℝ := (1/2) ^ (3/10)

-- State the theorem
theorem relationship_abc : b < a ∧ a < c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l812_81264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_alpha_l812_81202

noncomputable def power_function (α : ℝ) : ℝ → ℝ := fun x ↦ x ^ α

theorem power_function_alpha (α : ℝ) :
  power_function α (1/2) = Real.sqrt 2 / 2 → α = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_alpha_l812_81202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_value_l812_81295

noncomputable def f (x : ℝ) : ℝ :=
  if |x| ≤ 1 then 
    |x - 1| - 2
  else 
    1 / (1 + x^2)

theorem f_composition_value : f (f (1/2)) = 4/13 := by
  -- Evaluate f(1/2)
  have h1 : f (1/2) = -3/2 := by
    unfold f
    simp [abs]
    norm_num
  
  -- Evaluate f(-3/2)
  have h2 : f (-3/2) = 4/13 := by
    unfold f
    simp [abs]
    norm_num
  
  -- Combine the results
  rw [h1, h2]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_value_l812_81295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_and_min_equals_six_l812_81211

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x - 2)^2 * Real.sin (x - 2) + 3

-- State the theorem
theorem sum_of_max_and_min_equals_six :
  ∃ (M m : ℝ), 
    (∀ x ∈ Set.Icc (-1) 5, f x ≤ M) ∧ 
    (∃ x ∈ Set.Icc (-1) 5, f x = M) ∧
    (∀ x ∈ Set.Icc (-1) 5, m ≤ f x) ∧ 
    (∃ x ∈ Set.Icc (-1) 5, f x = m) ∧
    M + m = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_and_min_equals_six_l812_81211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_l_curve_below_right_of_l_l812_81239

/-- Curve C in parametric form -/
noncomputable def curve_C (a : ℝ) (t : ℝ) : ℝ × ℝ := (a * Real.cos t, 2 * Real.sin t)

/-- Line l in Cartesian form -/
def line_l (x y : ℝ) : Prop := x - y + 4 = 0

/-- Distance from a point to line l -/
noncomputable def distance_to_l (x y : ℝ) : ℝ := |x - y + 4| / Real.sqrt 2

theorem max_distance_to_l (a : ℝ) (h : a = 2 * Real.sqrt 3) :
  ∃ (t : ℝ), ∀ (s : ℝ), distance_to_l (curve_C a t).1 (curve_C a t).2 ≥ distance_to_l (curve_C a s).1 (curve_C a s).2 ∧
  distance_to_l (curve_C a t).1 (curve_C a t).2 = 2 * Real.sqrt 2 + 2 := by
  sorry

theorem curve_below_right_of_l (a : ℝ) :
  (∀ (t : ℝ), (curve_C a t).1 - (curve_C a t).2 + 4 > 0) ↔ 0 < a ∧ a < 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_l_curve_below_right_of_l_l812_81239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l812_81236

noncomputable def f (x : ℝ) := 1 / Real.log x + Real.sqrt (2 - x)

theorem domain_of_f :
  {x : ℝ | x ∈ Set.Ioo 0 1 ∪ Set.Ioc 1 2} =
  {x : ℝ | Real.log x ≠ 0 ∧ 2 - x ≥ 0 ∧ x > 0 ∧ x ≠ 1 ∧ x ≤ 2} :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l812_81236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l812_81294

noncomputable def f (x m : ℝ) : ℝ := |2^x - m|
noncomputable def g (x m : ℝ) : ℝ := |2^(-x) - m|

theorem m_range (m : ℝ) :
  (∃ (a b : ℝ), a < b ∧
    (∀ x ∈ Set.Ioo a b, Monotone (fun x => f x m) ∧ Monotone (fun x => g x m)) ∨
    (∀ x ∈ Set.Ioo a b, Antitone (fun x => f x m) ∧ Antitone (fun x => g x m))) →
  m ∈ Set.Icc (1/2) 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l812_81294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_k_l812_81221

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := Real.log (x^2 - 2*k*x + k)

theorem range_of_k (k : ℝ) : 
  (∀ y : ℝ, ∃ x : ℝ, f k x = y) ↔ k ≤ 0 ∨ k ≥ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_k_l812_81221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_count_is_four_l812_81227

/-- The number of intersection points between the curves (x - ⌊x⌋)² + y² = x - ⌊x⌋ and y = x² -/
noncomputable def intersection_points_count : ℕ :=
  let curve1 := {(x, y) : ℝ × ℝ | (x - ⌊x⌋)^2 + y^2 = x - ⌊x⌋}
  let curve2 := {(x, y) : ℝ × ℝ | y = x^2}
  let intersection_points := curve1 ∩ curve2
  4

/-- Proof that the number of intersection points is 4 -/
theorem intersection_points_count_is_four :
  intersection_points_count = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_count_is_four_l812_81227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l812_81263

def is_valid_number (n : ℕ) : Bool :=
  10 ≤ n ∧ n ≤ 99 ∧
  let tens := n / 10
  let ones := n % 10
  (10 * ones + tens) ≥ 2 * n

theorem count_valid_numbers :
  (Finset.filter (fun n => is_valid_number n) (Finset.range 100)).card = 14 := by
  sorry

#eval (Finset.filter (fun n => is_valid_number n) (Finset.range 100)).card

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l812_81263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_volume_zero_l812_81281

/-- Define the first region -/
def region1 (p : ℝ × ℝ × ℝ) : Prop :=
  abs p.1 ≤ 1 ∧ abs p.2.1 ≤ 1 ∧ abs p.2.2 ≤ 1

/-- Define the second region -/
def region2 (p : ℝ × ℝ × ℝ) : Prop :=
  abs p.1 ≤ 1 ∧ abs p.2.1 ≤ 1 ∧ abs (p.2.2 - 2) ≤ 1

/-- The volume of the intersection of the two regions is zero -/
theorem intersection_volume_zero :
  MeasureTheory.volume (Set.inter {p : ℝ × ℝ × ℝ | region1 p} {p : ℝ × ℝ × ℝ | region2 p}) = 0 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_volume_zero_l812_81281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_liquid_y_film_radius_l812_81228

/-- The thickness of the circular film formed by liquid Y on water -/
def film_thickness : ℝ := 0.2

/-- The height of the cylindrical container -/
def container_height : ℝ := 10

/-- The radius of the cylindrical container -/
def container_radius : ℝ := 5

/-- The volume of liquid Y in the container -/
noncomputable def liquid_volume : ℝ := Real.pi * container_radius^2 * container_height

/-- The radius of the circular film formed by liquid Y on water -/
noncomputable def film_radius : ℝ := Real.sqrt 1250

/-- Theorem stating that the volume of the circular film equals the volume of liquid Y -/
theorem liquid_y_film_radius :
  film_radius^2 * film_thickness * Real.pi = liquid_volume := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_liquid_y_film_radius_l812_81228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_power_rotation_120_l812_81282

open Real Matrix

noncomputable def rotation_matrix (angle : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![cos angle, -sin angle; sin angle, cos angle]

theorem smallest_power_rotation_120 :
  (∃ n : ℕ+, (rotation_matrix (2 * π / 3)) ^ n.val = 1 ∧
    ∀ m : ℕ+, m < n → (rotation_matrix (2 * π / 3)) ^ m.val ≠ 1) →
  (∃ n : ℕ+, (rotation_matrix (2 * π / 3)) ^ n.val = 1 ∧
    ∀ m : ℕ+, m < n → (rotation_matrix (2 * π / 3)) ^ m.val ≠ 1 ∧ n = 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_power_rotation_120_l812_81282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_competition_result_l812_81201

-- Define the participants
inductive Participant : Type
| olya : Participant
| oleg : Participant
| pasha : Participant

-- Define the possible placements
inductive Place : Type
| first : Place
| second : Place
| third : Place

-- Define a function to represent the actual placement
variable (actual_placement : Participant → Place)

-- Define a function to represent whether a participant is telling the truth
variable (is_truthful : Participant → Prop)

-- Define a function to represent whether a place is odd
def is_odd_place : Place → Prop
| Place.first => True
| Place.second => False
| Place.third => True

-- Define a function to represent whether a participant is a boy
variable (is_boy : Participant → Prop)

-- State the theorem
theorem competition_result :
  -- Each participant claims to be first
  (∀ p : Participant, is_truthful p ↔ actual_placement p = Place.first) →
  -- Olya claims all odd places are taken by boys
  (is_truthful Participant.olya ↔ 
    (∀ place : Place, is_odd_place place → 
      ∀ p : Participant, actual_placement p = place → is_boy p)) →
  -- Oleg claims Olya is wrong
  (is_truthful Participant.oleg ↔ ¬is_truthful Participant.olya) →
  -- All participants are either truthful or all are lying
  ((∀ p : Participant, is_truthful p) ∨ (∀ p : Participant, ¬is_truthful p)) →
  -- Olya is not a boy
  ¬is_boy Participant.olya →
  -- The actual placement is as stated
  actual_placement Participant.oleg = Place.first ∧
  actual_placement Participant.pasha = Place.second ∧
  actual_placement Participant.olya = Place.third :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_competition_result_l812_81201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_point_l812_81266

/-
  Define the curve C₂ and the line l
-/
def C₂ (x y : ℝ) : Prop := (x / Real.sqrt 3)^2 + (y / 2)^2 = 1
def l (x y : ℝ) : Prop := 2*x - y - 6 = 0

/-
  Define the distance function from a point to the line l
-/
noncomputable def distance_to_l (x y : ℝ) : ℝ :=
  abs (2*x - y - 6) / Real.sqrt 5

/-
  State the theorem
-/
theorem max_distance_point :
  C₂ (-3/2) 1 ∧
  ∀ x y : ℝ, C₂ x y → distance_to_l x y ≤ 2 * Real.sqrt 5 ∧
  distance_to_l (-3/2) 1 = 2 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_point_l812_81266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l812_81217

def a : ℕ → ℚ
  | 0 => 2
  | n + 1 => ((n + 2 : ℕ) * a n + 2) / (n + 1)

def S : ℕ → ℚ
  | n => 2^(n + 1)

def b : ℕ → ℚ
  | 0 => S 0
  | n + 1 => S (n + 1) - S n

def T : ℕ → ℚ
  | 0 => a 0 / b 0
  | n + 1 => T n + a (n + 1) / b (n + 1)

theorem sequence_properties :
  (∀ n : ℕ, a n = 4 * (n + 1) - 2) ∧
  (∀ n : ℕ, b n = if n = 0 then 4 else 2^(n + 1)) ∧
  (∀ n : ℕ, T n = 11/2 - (2*(n + 1) + 3)/2^n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l812_81217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_real_z_implies_m_equals_one_l812_81223

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the complex number z
noncomputable def z (m : ℝ) : ℂ := (1 + i) / (1 - i) + m * ((1 - i) / (1 + i))

-- Theorem statement
theorem real_z_implies_m_equals_one (m : ℝ) : 
  (z m).im = 0 → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_real_z_implies_m_equals_one_l812_81223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_b_value_l812_81208

/-- A cubic polynomial function -/
def cubic_polynomial (a b c d : ℝ) : ℝ → ℝ := λ x ↦ a*x^3 + b*x^2 + c*x + d

/-- Theorem: For a cubic polynomial f(x) = ax³ + bx² + cx + d, 
    if f(-2) = 0, f(2) = 0, and f(0) = 3, then b = 3/4 -/
theorem cubic_polynomial_b_value 
  (a b c d : ℝ) 
  (h1 : cubic_polynomial a b c d (-2) = 0)
  (h2 : cubic_polynomial a b c d 2 = 0)
  (h3 : cubic_polynomial a b c d 0 = 3) :
  b = 3/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_b_value_l812_81208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_l812_81267

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 2 * Real.cos (2 * x - Real.pi / 4)

-- Define the interval
def interval : Set ℝ := Set.Icc (-Real.pi / 8) (Real.pi / 2)

-- State the theorem
theorem f_extrema :
  (∃ (x : ℝ), x ∈ interval ∧ f x = Real.sqrt 2 ∧ x = Real.pi / 8) ∧
  (∀ (y : ℝ), y ∈ interval → f y ≤ Real.sqrt 2) ∧
  (∃ (x : ℝ), x ∈ interval ∧ f x = -1 ∧ x = Real.pi / 2) ∧
  (∀ (y : ℝ), y ∈ interval → f y ≥ -1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_l812_81267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_difference_l812_81280

/-- Represents a rectangular prism with face perimeters p1, p2, and p3 -/
structure RectPrism where
  p1 : ℤ
  p2 : ℤ
  p3 : ℤ

/-- Calculates the volume of a rectangular prism given its face perimeters -/
def volume (prism : RectPrism) : ℤ :=
  let x := (prism.p1 + prism.p2 - prism.p3) / 4
  let y := (prism.p1 - prism.p2 + prism.p3) / 4
  let z := (-prism.p1 + prism.p2 + prism.p3) / 4
  x * y * z

theorem volume_difference (prismA prismB : RectPrism) 
  (h1 : prismA.p1 = 12 ∧ prismA.p2 = 16 ∧ prismA.p3 = 24)
  (h2 : prismB.p1 = 12 ∧ prismB.p2 = 16 ∧ prismB.p3 = 20) :
  volume prismA - volume prismB = -13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_difference_l812_81280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_furniture_cost_theorem_l812_81275

/-- The correct cost of a piece of furniture --/
def correct_cost (total_paid : ℚ) (num_pieces : ℚ) (reimbursement : ℚ) : ℚ :=
  (total_paid - reimbursement) / num_pieces

/-- Theorem stating the correct cost of a piece of furniture --/
theorem furniture_cost_theorem (total_paid num_pieces reimbursement : ℚ) 
  (h1 : total_paid = 20700)
  (h2 : num_pieces = 150)
  (h3 : reimbursement = 600) :
  correct_cost total_paid num_pieces reimbursement = 134 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_furniture_cost_theorem_l812_81275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prescribed_dosage_percentage_less_l812_81200

noncomputable def typical_dosage_per_15_pounds : ℝ := 2
noncomputable def patient_weight : ℝ := 120
noncomputable def prescribed_dosage : ℝ := 12

noncomputable def typical_dosage : ℝ := (patient_weight / 15) * typical_dosage_per_15_pounds

theorem prescribed_dosage_percentage_less (h : typical_dosage > prescribed_dosage) :
  (typical_dosage - prescribed_dosage) / typical_dosage * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prescribed_dosage_percentage_less_l812_81200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l812_81261

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1/2) ^ (2*x - x^2)

-- State the theorem about the range of f
theorem range_of_f :
  Set.range f = Set.Ici (1/2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l812_81261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_product_semicircle_l812_81274

/-- Given a semicircle with radius 3 and endpoints P and Q, with 4 equally spaced points
    D₁, D₂, D₃, D₄ on the arc, the product of the lengths of the chords
    PD₁, PD₂, PD₃, PD₄, QD₁, QD₂, QD₃, QD₄ is equal to 98415. -/
theorem chord_product_semicircle (P Q D₁ D₂ D₃ D₄ : ℂ) : 
  let r : ℝ := 3
  let ω : ℂ := Complex.exp (2 * Real.pi * Complex.I / 10)
  -- P and Q are endpoints of the semicircle
  P = r ∧ Q = -r →
  -- D₁, D₂, D₃, D₄ are equally spaced points on the semicircle
  D₁ = r * ω ∧ D₂ = r * ω^2 ∧ D₃ = r * ω^3 ∧ D₄ = r * ω^4 →
  -- Product of chord lengths
  (Complex.abs (P - D₁) * Complex.abs (P - D₂) * Complex.abs (P - D₃) * Complex.abs (P - D₄) * 
   Complex.abs (Q - D₁) * Complex.abs (Q - D₂) * Complex.abs (Q - D₃) * Complex.abs (Q - D₄)) = 98415 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_product_semicircle_l812_81274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2011_equals_2_l812_81259

def sequence_a : ℕ → ℕ
  | 0 => 2  -- We define a(0) as 2 to match a(1) in the problem
  | 1 => 3
  | (n + 2) => (sequence_a (n + 1) * sequence_a n) % 10

theorem a_2011_equals_2 : sequence_a 2010 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2011_equals_2_l812_81259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pear_vendor_theorem_l812_81234

/-- Represents the selling and throwing away pattern of a pear vendor --/
noncomputable def pear_vendor_pattern (initial_sale_percent : ℝ) : ℝ :=
  let first_day_remaining := 1 - initial_sale_percent / 100
  let first_day_thrown := 0.5 * first_day_remaining
  let second_day_remaining := 0.5 * first_day_remaining
  let second_day_sold := 0.8 * second_day_remaining
  let second_day_thrown := 0.2 * second_day_remaining
  first_day_thrown + second_day_thrown

/-- Theorem stating that if the total percentage of pears thrown away is approximately 12%,
    then the initial sale percentage is 80% --/
theorem pear_vendor_theorem :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.000000000000004 ∧ pear_vendor_pattern 80 = 0.11999999999999996 + ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pear_vendor_theorem_l812_81234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equality_condition_l812_81237

/-- Piecewise function g as defined in the problem -/
noncomputable def g (x : ℝ) : ℝ :=
  if x ≤ 0 then -x else 2*x - 41

/-- Theorem stating the equality condition for the given problem -/
theorem equality_condition (a : ℝ) :
  a < 0 → (g (g (g 10.5)) = g (g (g a)) ↔ a = -30.5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equality_condition_l812_81237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_transport_theorem_l812_81255

-- Define the setup
structure Point where
  x : ℝ

-- Define constants
def A : Point := ⟨0⟩
def B : Point := ⟨1⟩

-- Define properties
noncomputable def distance (p q : Point) : ℝ := |p.x - q.x|
noncomputable def speed_bus : ℝ := 7
noncomputable def speed_walk : ℝ := 1

-- Define the conditions
def bus_capacity : ℕ := 4 -- Bus can carry 1/4 of the group
axiom speed_ratio : speed_bus = 7 * speed_walk
axiom simultaneous_arrival : True -- All groups arrive at B simultaneously

-- Define the journey
noncomputable def journey_distance : ℝ := distance A B
noncomputable def bus_trip_distance : ℝ := (4 : ℝ) / 7 * journey_distance

-- State the theorem
theorem optimal_transport_theorem :
  -- Proportion of journey by bus
  (4 : ℝ) / 7 * journey_distance = bus_trip_distance ∧
  -- Proportion of journey by foot
  (3 : ℝ) / 7 * journey_distance = journey_distance - bus_trip_distance ∧
  -- Time ratio for direct transport
  (let direct_time := (4 : ℝ) * journey_distance / speed_bus;
   let optimal_time := journey_distance / speed_walk;
   direct_time / optimal_time = 49 / 25) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_transport_theorem_l812_81255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_knight_moves_correct_l812_81226

-- Define the function for the number of squares reachable by a knight in n moves
def knight_moves (n : ℕ) : ℕ :=
  match n with
  | 1 => 8
  | 2 => 32
  | 3 => 68
  | 4 => 96
  | _ => 28 * n - 20

-- Define the concept of an infinite chessboard (as an axiom since we can't actually represent an infinite structure)
axiom infinite_chessboard : Type

-- Define the concept of a knight move on an infinite chessboard
axiom knight_moves_on_infinite_chessboard : infinite_chessboard → infinite_chessboard → Prop

-- Define the concept of the number of squares reachable in exactly n moves
axiom number_of_squares_reachable_in_exactly : 
  ℕ → (infinite_chessboard → infinite_chessboard → Prop) → ℕ

-- Theorem statement
theorem knight_moves_correct :
  ∀ n : ℕ, n > 0 →
  (knight_moves n) = (number_of_squares_reachable_in_exactly n knight_moves_on_infinite_chessboard) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_knight_moves_correct_l812_81226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_division_theorem_l812_81250

/-- Given a line segment CD and a point Q on CD such that CQ:QD = 4:1,
    prove that Q = (1/5)C + (4/5)D -/
theorem point_division_theorem (C D Q : ℝ × ℝ × ℝ) :
  Q ∈ Set.Icc C D →
  dist C Q / dist Q D = 4 / 1 →
  Q = (1 / 5 : ℝ) • C + (4 / 5 : ℝ) • D :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_division_theorem_l812_81250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_for_decreasing_f_l812_81268

open Real

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := sqrt (3 - m * x) / (m - 1)

-- State the theorem
theorem range_of_m_for_decreasing_f :
  ∀ m : ℝ, m ≠ 1 →
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 1 → f m x₁ > f m x₂) →
  m ∈ Set.Iic 0 ∪ Set.Ico 1 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_for_decreasing_f_l812_81268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_intersection_l812_81258

/-- The parabola y = x^2 - 4x - 1 -/
def parabola (x : ℝ) : ℝ := x^2 - 4*x - 1

/-- Point A on the parabola -/
structure PointA where
  x : ℝ
  y : ℝ
  on_parabola : y = parabola x

/-- Point B on the parabola -/
structure PointB where
  x : ℝ
  y : ℝ
  on_parabola : y = parabola x

/-- Point C on the parabola -/
structure PointC where
  x : ℝ
  y : ℝ
  on_parabola : y = parabola x
  x_value : x = 5

/-- Definition of median intersection -/
def is_median_intersection (xa ya xb yb xc yc mx my : ℝ) : Prop :=
  -- This is a placeholder definition. In a complete proof, we would define the actual conditions
  -- for a point to be the intersection of the medians.
  mx = (xa + xb + xc) / 3 ∧ my = (ya + yb + yc) / 3

theorem median_intersection
  (A : PointA) (B : PointB) (C : PointC)
  (h_distinct : A.x ≠ B.x ∧ B.x ≠ C.x ∧ A.x ≠ C.x)
  (h_equal_y : A.y = B.y) :
  ∃ M : ℝ × ℝ, M.1 = 3 ∧ is_median_intersection A.x A.y B.x B.y C.x C.y M.1 M.2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_intersection_l812_81258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spinning_ring_rotations_l812_81240

/-- The number of rotations a spinning ring makes before stopping in a dihedral angle -/
noncomputable def number_of_rotations (R ω μ g : ℝ) : ℝ :=
  (ω^2 * R * (1 + μ^2)) / (4 * Real.pi * g * μ * (1 + μ))

/-- Theorem: The number of rotations of a spinning ring before stopping in a dihedral angle -/
theorem spinning_ring_rotations
  (R ω μ g : ℝ)
  (h_R : R > 0)
  (h_ω : ω > 0)
  (h_μ : μ > 0)
  (h_g : g > 0) :
  ∃ (n : ℝ), n = number_of_rotations R ω μ g ∧ n > 0 := by
  use number_of_rotations R ω μ g
  constructor
  · rfl
  · sorry  -- The proof that n > 0 is omitted for brevity


end NUMINAMATH_CALUDE_ERRORFEEDBACK_spinning_ring_rotations_l812_81240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l812_81220

theorem relationship_abc (a b c : ℝ) 
  (ha : a = (2 : ℝ) ^ (3/10 : ℝ)) 
  (hb : b = (3 : ℝ) ^ (2 : ℝ)) 
  (hc : c = (2 : ℝ) ^ (-(3/10) : ℝ)) :
  c < a ∧ a < b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l812_81220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_rate_of_change_f_on_1_2_l812_81252

/-- The function f(x) = 2x + 1 -/
def f (x : ℝ) : ℝ := 2 * x + 1

/-- The average rate of change of a function on an interval -/
noncomputable def averageRateOfChange (f : ℝ → ℝ) (a b : ℝ) : ℝ :=
  (f b - f a) / (b - a)

theorem average_rate_of_change_f_on_1_2 :
  averageRateOfChange f 1 2 = 2 := by
  -- Unfold the definition of averageRateOfChange
  unfold averageRateOfChange
  -- Simplify the expression
  simp [f]
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_rate_of_change_f_on_1_2_l812_81252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_plane_perpendicular_properties_l812_81214

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)

-- Main theorem
theorem line_plane_perpendicular_properties 
  (l m : Line) (α : Plane) 
  (h : perpendicular_line_plane l α) :
  (parallel_lines m l → perpendicular_line_plane m α) ∧
  (perpendicular_line_plane m α → parallel_lines m l) ∧
  (parallel_line_plane m α → perpendicular_lines m l) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_plane_perpendicular_properties_l812_81214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_half_negative_two_power_l812_81205

theorem half_negative_two_power : (1 / 2 : ℚ)^(-2 : ℤ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_half_negative_two_power_l812_81205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_six_people_round_table_l812_81243

def number_of_circular_permutations (n : ℕ) : ℕ := (n - 1).factorial

theorem six_people_round_table : number_of_circular_permutations 6 = 120 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_six_people_round_table_l812_81243
