import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_large_sum_l1290_129044

/-- The sequence generation process -/
def generateSequence (N a : ℕ) : List ℕ :=
  let rec aux (last : ℕ) (acc : List ℕ) (fuel : ℕ) : List ℕ :=
    if fuel = 0 then acc.reverse
    else if last = 0 then acc.reverse
    else aux (N % last) (last :: acc) (fuel - 1)
  aux a [a] N

/-- The theorem statement -/
theorem exists_large_sum : ∃ (N a : ℕ), a < N ∧ (generateSequence N a).sum > 100 * N := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_large_sum_l1290_129044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_subsets_friend_invitation_days_l1290_129067

theorem number_of_subsets (n : ℕ) : Finset.card (Finset.powerset (Finset.range n)) = 2^n := by
  sorry

theorem friend_invitation_days : Finset.card (Finset.powerset (Finset.range 10)) = 1024 := by
  have h : Finset.card (Finset.powerset (Finset.range 10)) = 2^10 := number_of_subsets 10
  rw [h]
  norm_num

#eval Finset.card (Finset.powerset (Finset.range 10))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_subsets_friend_invitation_days_l1290_129067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_magnitude_2a_minus_b_l1290_129052

noncomputable def a (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)

noncomputable def b : ℝ × ℝ := (Real.sqrt 3, -1)

theorem max_magnitude_2a_minus_b :
  ∃ M : ℝ, M = 4 ∧ ∀ θ : ℝ, 
    Real.sqrt ((2 * (a θ).1 - b.1)^2 + (2 * (a θ).2 - b.2)^2) ≤ M :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_magnitude_2a_minus_b_l1290_129052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_property_l1290_129082

/-- Represents a triangle with vertices X, Y, Z -/
structure Triangle (X Y Z : ℝ × ℝ) : Prop where
  xy_length : dist X Y = 29
  yz_length : dist Y Z = 31
  xz_length : dist X Z = 30

/-- Represents an inscribed triangle PQR inside triangle XYZ -/
structure InscribedTriangle (X Y Z P Q R : ℝ × ℝ) : Prop where
  p_on_yz : ∃ t : ℝ, P = (1 - t) • Y + t • Z
  q_on_xz : ∃ t : ℝ, Q = (1 - t) • X + t • Z
  r_on_xy : ∃ t : ℝ, R = (1 - t) • X + t • Y

/-- Represents the equality of arcs -/
structure ArcEquality (X Y Z P Q R : ℝ × ℝ) : Prop where
  ry_qz : dist R Y = dist Q Z
  rp_qy : dist R P = dist Q Y
  py_xr : dist P Y = dist X R

/-- The main theorem -/
theorem inscribed_triangle_property 
  (X Y Z P Q R : ℝ × ℝ) 
  (tri : Triangle X Y Z) 
  (inscribed : InscribedTriangle X Y Z P Q R) 
  (arc_eq : ArcEquality X Y Z P Q R) : 
  dist X R = 33 / 2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_property_l1290_129082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l1290_129060

/-- Given vectors in ℝ², prove that the angle between 2a-b and a is 45° --/
theorem angle_between_vectors (a b : ℝ × ℝ) (ha : a = (2, 1)) (hb : b = (1, 3)) :
  let v := (2 • a) - b
  Real.arccos ((v.1 * a.1 + v.2 * a.2) / (Real.sqrt (v.1^2 + v.2^2) * Real.sqrt (a.1^2 + a.2^2))) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l1290_129060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_abs_coeff_equals_S_neg_one_l1290_129005

/-- The polynomial R(x) -/
noncomputable def R (x : ℝ) : ℝ := 1 - (1/4) * x + (1/8) * x^2

/-- The function S(x) defined as the product of R(x) at different powers of x -/
noncomputable def S (x : ℝ) : ℝ := R x * R (x^2) * R (x^4) * R (x^6) * R (x^8)

/-- The sum of absolute values of coefficients in the expansion of S(x) -/
noncomputable def sum_abs_coeff : ℝ := S (-1)

/-- Theorem stating that the sum of absolute values of coefficients equals (5/4)^5 -/
theorem sum_abs_coeff_equals_S_neg_one : sum_abs_coeff = (5/4)^5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_abs_coeff_equals_S_neg_one_l1290_129005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_of_g_l1290_129072

noncomputable def f (x : Real) : Real := Real.sin (2 * x) + Real.sqrt 3 * Real.cos (2 * x)

noncomputable def g (x : Real) : Real := 2 * Real.sin (x + Real.pi / 6)

theorem symmetry_axis_of_g :
  ∃ (k : Int), ∀ (x : Real), g x = g (2 * (Real.pi / 3 + k * Real.pi) - x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_of_g_l1290_129072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compare_ABCD_l1290_129020

noncomputable def A : ℝ := Real.sin (Real.sin (3 * Real.pi / 8))
noncomputable def B : ℝ := Real.sin (Real.cos (3 * Real.pi / 8))
noncomputable def C : ℝ := Real.cos (Real.sin (3 * Real.pi / 8))
noncomputable def D : ℝ := Real.cos (Real.cos (3 * Real.pi / 8))

theorem compare_ABCD : B < C ∧ C < A ∧ A < D := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compare_ABCD_l1290_129020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_C_properties_l1290_129042

noncomputable def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 14*y + 45 = 0

def point_Q : ℝ × ℝ := (-2, 3)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

noncomputable def k (m n : ℝ) : ℝ := (n - 3) / (m + 2)

theorem circle_C_properties :
  (∀ M : ℝ × ℝ, circle_C M.1 M.2 → distance M point_Q ≤ 6 * Real.sqrt 2) ∧
  (∀ M : ℝ × ℝ, circle_C M.1 M.2 → distance M point_Q ≥ 2 * Real.sqrt 2) ∧
  (∃ M : ℝ × ℝ, circle_C M.1 M.2 ∧ distance M point_Q = 6 * Real.sqrt 2) ∧
  (∃ M : ℝ × ℝ, circle_C M.1 M.2 ∧ distance M point_Q = 2 * Real.sqrt 2) ∧
  (∀ m n : ℝ, circle_C m n → k m n ≤ 2 + Real.sqrt 3) ∧
  (∀ m n : ℝ, circle_C m n → k m n ≥ 2 - Real.sqrt 3) ∧
  (∃ m n : ℝ, circle_C m n ∧ k m n = 2 + Real.sqrt 3) ∧
  (∃ m n : ℝ, circle_C m n ∧ k m n = 2 - Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_C_properties_l1290_129042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_f_x_increasing_l1290_129061

noncomputable def f (x : ℝ) : ℝ := (2^x - 1) / (2^x + 1)

theorem x_f_x_increasing {x₁ x₂ : ℝ} (h : 0 < x₁ ∧ x₁ < x₂) : 
  x₁ * f x₁ < x₂ * f x₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_f_x_increasing_l1290_129061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l1290_129058

def is_valid_number (n : ℕ) : Bool :=
  100 ≤ n && n ≤ 999 &&
  let h := n / 100
  let t := (n / 10) % 10
  let u := n % 10
  h > t && t > u && h ≠ t && t ≠ u && h ≠ u

theorem count_valid_numbers : 
  (Finset.filter (λ n => is_valid_number n) (Finset.range 900 ∪ {999})).card = 120 :=
by sorry

#eval (Finset.filter (λ n => is_valid_number n) (Finset.range 900 ∪ {999})).card

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l1290_129058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_divisible_terms_l1290_129090

def mySequence (a : ℕ → ℕ) : Prop :=
  a 0 = 1 ∧ ∀ n : ℕ, n > 0 → a n = a (n - 1) + a (n / 3)

theorem infinite_divisible_terms (a : ℕ → ℕ) (h_seq : mySequence a) :
  ∀ p : ℕ, Nat.Prime p → p ≤ 13 → Set.Infinite {k : ℕ | p ∣ a k} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_divisible_terms_l1290_129090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_incline_sqrt3x_l1290_129055

/-- AngleOfIncline function that calculates the angle of incline of a line -/
noncomputable def AngleOfIncline (y x : ℝ) : ℝ :=
  Real.arctan (y / x)

/-- The angle of incline of the line y = √3x is 60°. -/
theorem angle_of_incline_sqrt3x (x y : ℝ) :
  y = Real.sqrt 3 * x → AngleOfIncline y x = 60 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_incline_sqrt3x_l1290_129055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_10_terms_ap_l1290_129054

noncomputable def arithmetic_progression (a : ℝ) (d : ℝ) (n : ℕ) : ℝ := a + (n - 1 : ℝ) * d

noncomputable def sum_arithmetic_progression (a : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) / 2 * (2 * a + (n - 1 : ℝ) * d)

theorem sum_of_first_10_terms_ap (a d : ℝ) :
  arithmetic_progression a d 4 + arithmetic_progression a d 12 = 20 →
  sum_arithmetic_progression a d 10 = 100 - 25 * d :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_10_terms_ap_l1290_129054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_special_case_l1290_129001

/-- 
Given a right triangle with legs a and b, and area S,
if (a + b)^2 = 8S, then the angles of the triangle are 45°, 45°, and 90°.
-/
theorem right_triangle_special_case (a b S : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : S > 0) :
  (a + b)^2 = 8 * S →
  ∃ (α β γ : Real),
    α = 45 ∧ β = 45 ∧ γ = 90 ∧
    α + β + γ = 180 ∧
    Real.sin (α * π / 180) * a = Real.sin (β * π / 180) * b ∧
    S = (1/2) * a * b :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_special_case_l1290_129001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_composite_l1290_129030

theorem sequence_composite (n : ℕ) : ∃ (k m : ℕ), k > 1 ∧ m > 1 ∧ (10^(3*n) + 10^(2*n) + 10^n + 1 = k * m) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_composite_l1290_129030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xiang_qing_fried_eggs_min_time_exists_optimal_schedule_l1290_129088

/-- Represents a cooking step with its duration -/
structure CookingStep where
  name : String
  duration : Rat

/-- Represents the recipe for Xiang Qing Fried Eggs -/
def xiangQingFriedEggsRecipe : List CookingStep := [
  { name := "Wash and chop scallions", duration := 1 },
  { name := "Beat eggs", duration := 1/2 },
  { name := "Stir egg mixture and scallions", duration := 1 },
  { name := "Wash pan", duration := 1/2 },
  { name := "Heat pan", duration := 1/2 },
  { name := "Heat oil", duration := 1/2 },
  { name := "Cook dish", duration := 2 }
]

/-- The minimum time to complete the Xiang Qing Fried Eggs dish -/
def minTimeToCook : Rat := 5

/-- Function to calculate the total cooking time considering overlaps -/
def calculateTotalTime (schedule : List CookingStep) : Rat :=
  sorry -- Implement the logic to calculate total time with overlaps

/-- Theorem stating that the minimum time to cook Xiang Qing Fried Eggs is 5 minutes -/
theorem xiang_qing_fried_eggs_min_time :
  ∀ (schedule : List CookingStep),
    (∀ step ∈ xiangQingFriedEggsRecipe, step ∈ schedule) →
    (∀ step ∈ schedule, step ∈ xiangQingFriedEggsRecipe) →
    calculateTotalTime schedule ≥ minTimeToCook :=
by
  intro schedule hAllStepsIncluded hNoExtraSteps
  sorry -- Add the proof here

/-- Corollary: There exists a schedule that achieves the minimum cooking time -/
theorem exists_optimal_schedule :
  ∃ (optimalSchedule : List CookingStep),
    (∀ step ∈ xiangQingFriedEggsRecipe, step ∈ optimalSchedule) ∧
    (∀ step ∈ optimalSchedule, step ∈ xiangQingFriedEggsRecipe) ∧
    calculateTotalTime optimalSchedule = minTimeToCook :=
by
  sorry -- Add the proof here

end NUMINAMATH_CALUDE_ERRORFEEDBACK_xiang_qing_fried_eggs_min_time_exists_optimal_schedule_l1290_129088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_difference_l1290_129007

theorem power_difference (m n : ℝ) (h1 : (3 : ℝ)^m = 4) (h2 : (3 : ℝ)^n = 5) : 
  (3 : ℝ)^(m-n) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_difference_l1290_129007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_and_triangle_area_l1290_129085

-- Define the hyperbola and its properties
noncomputable def hyperbola (a : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 = 1 ∧ a > 0

-- Define the eccentricity
noncomputable def eccentricity : ℝ := 2 * Real.sqrt 3 / 3

-- Define the line
def line (x y : ℝ) : Prop :=
  y = x - 2

-- Helper function for triangle area
noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ :=
  abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)) / 2

-- Define the theorem
theorem hyperbola_and_triangle_area
  (a : ℝ)
  (h_hyperbola : ∀ x y, hyperbola a x y → x^2 / a^2 - y^2 = 1)
  (h_eccentricity : eccentricity = Real.sqrt (a^2 + 1) / a)
  (h_line : ∀ x y, line x y → y = x - 2)
  : (∀ x y, hyperbola (Real.sqrt 3) x y) ∧
    (∃ A B F₁ : ℝ × ℝ,
      (line A.1 A.2 ∧ hyperbola (Real.sqrt 3) A.1 A.2) ∧
      (line B.1 B.2 ∧ hyperbola (Real.sqrt 3) B.1 B.2) ∧
      (abs (F₁.1) = 2 * Real.sqrt 3 ∧ F₁.2 = 0) ∧
      (area_triangle A B F₁ = 2 * Real.sqrt 18)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_and_triangle_area_l1290_129085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_asha_remaining_money_l1290_129047

def borrowed_from_brother : ℚ := 20
def borrowed_from_father : ℚ := 40
def borrowed_from_mother : ℚ := 30
def gift_from_granny : ℚ := 70
def savings : ℚ := 100
def spending_fraction : ℚ := 3/4

def total_money : ℚ := borrowed_from_brother + borrowed_from_father + borrowed_from_mother + gift_from_granny + savings

theorem asha_remaining_money :
  total_money - (total_money * spending_fraction) = 65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_asha_remaining_money_l1290_129047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_max_l1290_129057

open Real

-- Define the function
noncomputable def f (x : ℝ) : ℝ := 1 / (cos x)^2 + 2 * tan x + 1

-- State the theorem
theorem f_min_max :
  ∀ x ∈ Set.Icc (-π/3) (π/4),
  (∀ y ∈ Set.Icc (-π/3) (π/4), f y ≥ f (-π/4)) ∧
  (f (-π/4) = 1) ∧
  (∀ y ∈ Set.Icc (-π/3) (π/4), f y ≤ f (π/4)) ∧
  (f (π/4) = 5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_max_l1290_129057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l1290_129008

-- Define the problem statement
theorem trigonometric_identities (x : ℝ) 
  (h1 : Real.cos (x - Real.pi/4) = Real.sqrt 2 / 10)
  (h2 : Real.pi/2 < x ∧ x < 3*Real.pi/4) :
  (Real.sin x = 4/5) ∧ 
  (Real.sin (2*x + Real.pi/3) = -(24 + 7*Real.sqrt 3) / 50) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l1290_129008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_a_outside_circle_l1290_129034

def circle_equation (x y a : ℝ) : Prop :=
  x^2 + y^2 - 2*a*x - 3*y + a^2 + a = 0

def point_outside_circle (a : ℝ) : Prop :=
  ∀ x y, circle_equation x y a → (x - a)^2 + (y - 3)^2 > 0

theorem point_a_outside_circle (a : ℝ) :
  point_outside_circle a ↔ 0 < a ∧ a < 9/4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_a_outside_circle_l1290_129034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_720_deg_eq_zero_sin_period_sin_zero_l1290_129018

/-- The sine of 720 degrees is equal to 0 -/
theorem sin_720_deg_eq_zero : Real.sin (720 * Real.pi / 180) = 0 := by
  sorry

/-- Sine function has a period of 2π -/
theorem sin_period (x : ℝ) : Real.sin (x + 2 * Real.pi) = Real.sin x := by
  sorry

/-- The sine of 0 is 0 -/
theorem sin_zero : Real.sin 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_720_deg_eq_zero_sin_period_sin_zero_l1290_129018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_major_axis_length_l1290_129022

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  focus1 : Point
  focus2 : Point
  tangentToYAxis : Bool

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: The length of the major axis of the given ellipse is 85 -/
theorem ellipse_major_axis_length :
  let e : Ellipse := { focus1 := ⟨10, 5⟩, focus2 := ⟨70, 30⟩, tangentToYAxis := true }
  distance e.focus1 e.focus2 = 85 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_major_axis_length_l1290_129022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_circle_C_internal_tangency_condition_l1290_129006

noncomputable section

/-- Circle C with center (m, 2m) and radius m -/
def circle_C (m : ℝ) (x y : ℝ) : Prop :=
  (x - m)^2 + (y - 2*m)^2 = m^2

/-- Circle E with center (3, 0) and radius 4 -/
def circle_E (x y : ℝ) : Prop :=
  (x - 3)^2 + y^2 = 16

/-- Line passing through origin with slope k -/
def line_through_origin (k : ℝ) (x y : ℝ) : Prop :=
  y = k * x

/-- Distance from point (x, y) to line y = kx -/
noncomputable def distance_to_line (k x y : ℝ) : ℝ :=
  |k * x - y| / Real.sqrt (k^2 + 1)

theorem tangent_line_to_circle_C (m : ℝ) (h : m > 0) :
  m = 2 →
  (∃ k, ∀ x y, circle_C m x y → distance_to_line k x y = m) ↔
  (∃ x y, (line_through_origin (3/4) x y ∨ x = 0) ∧ circle_C m x y) := by
  sorry

theorem internal_tangency_condition (m : ℝ) (h : m > 0) :
  (∃ x y, circle_C m x y ∧ circle_E x y ∧
    ∀ x' y', circle_C m x' y' → circle_E x' y' → (x = x' ∧ y = y')) →
  m = (Real.sqrt 29 - 1) / 4 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_circle_C_internal_tangency_condition_l1290_129006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_six_circles_bounded_area_l1290_129086

/-- The area of the figure bounded by six identical circles inscribed in a circle of radius R,
    where each smaller circle is tangent to the larger circle and its two neighboring circles -/
noncomputable def bounded_area (R : ℝ) : ℝ :=
  (R^2 / 9) * (3 * Real.sqrt 3 - 2 * Real.pi)

/-- Theorem stating that the area of the figure bounded by six identical circles inscribed
    in a circle of radius R, where each smaller circle is tangent to the larger circle and
    its two neighboring circles, is equal to (R^2 / 9) * (3√3 - 2π) -/
theorem six_circles_bounded_area (R : ℝ) (h : R > 0) :
  ∃ (r : ℝ), r > 0 ∧ r = R / 3 ∧
  bounded_area R = (R^2 / 9) * (3 * Real.sqrt 3 - 2 * Real.pi) := by
  use R / 3
  constructor
  · exact div_pos h (by norm_num)
  constructor
  · rfl
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_six_circles_bounded_area_l1290_129086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_plus_sin_value_l1290_129025

theorem cos_plus_sin_value (α : Real) 
  (h1 : 0 < α ∧ α < Real.pi) -- α is an interior angle of a triangle
  (h2 : Real.sin α * Real.cos α = 1/8) : 
  Real.cos α + Real.sin α = Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_plus_sin_value_l1290_129025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_cos_plus_x_sin_l1290_129053

theorem min_value_cos_plus_x_sin (x : ℝ) (h : x ∈ Set.Icc 0 (Real.pi / 2)) :
  Real.cos x + x * Real.sin x ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_cos_plus_x_sin_l1290_129053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_relationship_l1290_129012

/-- The function g(n) as defined in the problem -/
noncomputable def g (n : ℤ) : ℝ :=
  (4 + 2 * Real.sqrt 6) / 12 * ((2 + Real.sqrt 6) / 3) ^ n +
  (4 - 2 * Real.sqrt 6) / 12 * ((2 - Real.sqrt 6) / 3) ^ n

/-- Theorem stating the relationship between g(n+1), g(n-1), and g(n) -/
theorem g_relationship (n : ℤ) : g (n + 1) - g (n - 1) = (1 / 3) * g n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_relationship_l1290_129012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tube_volume_difference_l1290_129070

/-- The volume of a cylinder given its radius and height -/
noncomputable def cylinderVolume (r h : ℝ) : ℝ := Real.pi * r^2 * h

/-- The radius of a cylinder given its circumference -/
noncomputable def cylinderRadius (circumference : ℝ) : ℝ := circumference / (2 * Real.pi)

theorem tube_volume_difference :
  let amy_height : ℝ := 9
  let amy_circumference : ℝ := 7
  let belinda_height : ℝ := 10
  let belinda_circumference : ℝ := 5
  let amy_volume := cylinderVolume (cylinderRadius amy_circumference) amy_height
  let belinda_volume := cylinderVolume (cylinderRadius belinda_circumference) belinda_height
  Real.pi * |amy_volume - belinda_volume| = 191 / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tube_volume_difference_l1290_129070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_triple_g_roots_l1290_129087

-- Define the quadratic function g as noncomputable
noncomputable def g : ℝ → ℝ := λ x => x^2/4 - x - 2

-- State the theorem
theorem sum_of_triple_g_roots : 
  ∃ (x₁ x₂ x₃ x₄ : ℝ), 
    (∀ x : ℝ, g (g (g x)) = -1 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄) ∧
    x₁ + x₂ + x₃ + x₄ = -5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_triple_g_roots_l1290_129087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_side_length_l1290_129092

-- Define a right triangle with given side lengths
def right_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2

-- Define approximate equality
def approx_eq (x y : ℝ) (ε : ℝ) : Prop :=
  abs (x - y) < ε

-- Define the theorem
theorem shortest_side_length :
  ∀ x : ℝ, right_triangle 7 x 24 → approx_eq x 22.96 0.01 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_side_length_l1290_129092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_inside_curve_C_min_distance_C_to_l_l1290_129016

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y + 4 = 0

-- Define the curve C
def curve_C (x y : ℝ) : Prop := x^2 / 3 + y^2 / 2 = 1

-- Define point P
def point_P : ℝ × ℝ := (1, 1)

-- Theorem for part (I)
theorem point_P_inside_curve_C : 
  curve_C point_P.1 point_P.2 = False := by
  sorry

-- Define a point on curve C
noncomputable def point_on_C (α : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos α, Real.sqrt 2 * Real.sin α)

-- Function to calculate distance from a point to line l
noncomputable def distance_to_line_l (x y : ℝ) : ℝ :=
  abs (x - y + 4) / Real.sqrt 2

-- Theorem for part (II)
theorem min_distance_C_to_l :
  ∃ (min_dist : ℝ), min_dist = (4 * Real.sqrt 2 - Real.sqrt 10) / 2 ∧
  ∀ (α : ℝ), distance_to_line_l (point_on_C α).1 (point_on_C α).2 ≥ min_dist := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_inside_curve_C_min_distance_C_to_l_l1290_129016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_two_speak_l1290_129049

def prob_speak : ℚ := 1 / 4
def num_babies : ℕ := 5

theorem at_least_two_speak (prob_speak : ℚ) (num_babies : ℕ) :
  prob_speak = 1 / 4 →
  num_babies = 5 →
  (1 : ℚ) - (Nat.choose num_babies 0 * (1 - prob_speak) ^ num_babies +
             Nat.choose num_babies 1 * prob_speak * (1 - prob_speak) ^ (num_babies - 1)) = 47 / 128 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_two_speak_l1290_129049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_M_equals_rangeF_l1290_129062

open Set
open Function

/-- The function f(x) = x^2 - x -/
def f (x : ℝ) : ℝ := x^2 - x

/-- The open interval (-1, 1) -/
def openInterval : Set ℝ := {x | -1 < x ∧ x < 1}

/-- The set M of all possible values of m -/
def M : Set ℝ := {m | ∃ x ∈ openInterval, f x = m}

/-- The range of f over the open interval (-1, 1) -/
noncomputable def rangeF : Set ℝ := f '' openInterval

theorem M_equals_rangeF : M = rangeF := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_M_equals_rangeF_l1290_129062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_is_constant_l1290_129037

/-- A vector on the line y = 3x - 1 -/
def VectorOnLine (v : ℝ × ℝ) : Prop :=
  v.2 = 3 * v.1 - 1

/-- The projection of v onto w -/
noncomputable def proj (v w : ℝ × ℝ) : ℝ × ℝ :=
  let dot := v.1 * w.1 + v.2 * w.2
  let norm_squared := w.1 * w.1 + w.2 * w.2
  (dot / norm_squared * w.1, dot / norm_squared * w.2)

/-- The theorem statement -/
theorem projection_is_constant (v w : ℝ × ℝ) (h : VectorOnLine v) :
  proj v w = (3/10, -1/10) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_is_constant_l1290_129037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l1290_129097

-- Define the line type
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a function to check if a point is on a line
def point_on_line (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y + l.c = 0

-- Define a function to calculate the area of a triangle formed by a line and the coordinate axes
noncomputable def triangle_area (l : Line) : ℝ :=
  abs (l.c * l.c) / (2 * abs (l.a * l.b))

-- Theorem statement
theorem line_equation (l : Line) :
  point_on_line l (-5) (-4) ∧ 
  triangle_area l = 5 →
  (l.a = 8 ∧ l.b = -5 ∧ l.c = 20) ∨ 
  (l.a = 2 ∧ l.b = -5 ∧ l.c = -10) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l1290_129097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_slope_three_implies_a_range_l1290_129038

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (2 * x) - 2 * Real.exp x + a * x - 1

-- Define the derivative of f
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := 2 * Real.exp (2 * x) - 2 * Real.exp x + a

-- Theorem statement
theorem tangent_lines_slope_three_implies_a_range (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f' a x₁ = 3 ∧ f' a x₂ = 3) →
  3 < a ∧ a < 7/2 := by
  sorry

#check tangent_lines_slope_three_implies_a_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_slope_three_implies_a_range_l1290_129038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magic_square_sum_l1290_129015

/-- Represents a 3x3 magic square -/
structure MagicSquare :=
  (a11 a12 a13 a21 a22 a23 a31 a32 a33 : ℕ)

/-- The magic constant of the square -/
def magic_constant : ℕ := 66

/-- Conditions for a valid magic square -/
def is_valid_magic_square (s : MagicSquare) : Prop :=
  s.a11 + s.a12 + s.a13 = magic_constant ∧
  s.a21 + s.a22 + s.a23 = magic_constant ∧
  s.a31 + s.a32 + s.a33 = magic_constant ∧
  s.a11 + s.a21 + s.a31 = magic_constant ∧
  s.a12 + s.a22 + s.a32 = magic_constant ∧
  s.a13 + s.a23 + s.a33 = magic_constant ∧
  s.a11 + s.a22 + s.a33 = magic_constant ∧
  s.a13 + s.a22 + s.a31 = magic_constant

/-- The specific magic square from the problem -/
def problem_square : MagicSquare :=
  { a11 := 25, a12 := 0, a13 := 21,
    a21 := 18, a22 := 0, a23 := 0,
    a31 := 0, a32 := 24, a33 := 0 }

theorem magic_square_sum (s : MagicSquare) 
  (h : is_valid_magic_square s) 
  (h_prob : s = problem_square) : 
  s.a23 + s.a12 = 46 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_magic_square_sum_l1290_129015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangles_from_construction_l1290_129069

-- Define the basic structures
structure Point where
  x : ℝ
  y : ℝ

structure Triangle where
  A : Point
  B : Point
  C : Point

structure Circle where
  center : Point
  radius : ℝ

-- Define the given conditions
def is_equilateral (t : Triangle) : Prop := sorry

def on_circle (p : Point) (c : Circle) : Prop := sorry

def measure_arc (p q : Point) (c : Circle) : ℝ := sorry

def perpendicular (p q : Point) (l : Point → Point → Prop) : Prop := sorry

def orthocenter (t : Triangle) : Point := sorry

-- Main theorem
theorem equilateral_triangles_from_construction 
  (A B C N P Q : Point) (Ω : Circle) (H₁ H₂ H₃ : Point) :
  is_equilateral ⟨A, B, C⟩ →
  on_circle N Ω →
  measure_arc N B Ω = 30 →
  perpendicular N P (λ x y ↦ x.y - y.y = (A.y - C.y) / (A.x - C.x) * (x.x - y.x)) →
  perpendicular N Q (λ x y ↦ x.y - y.y = (A.y - B.y) / (A.x - B.x) * (x.x - y.x)) →
  H₁ = orthocenter ⟨N, A, B⟩ →
  H₂ = orthocenter ⟨Q, B, C⟩ →
  H₃ = orthocenter ⟨C, A, P⟩ →
  is_equilateral ⟨N, P, Q⟩ ∧ is_equilateral ⟨H₁, H₂, H₃⟩ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangles_from_construction_l1290_129069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_difference_l1290_129011

noncomputable def sample (x y : ℝ) : List ℝ := [9, 10, 11, x, y]

noncomputable def average (xs : List ℝ) : ℝ := (xs.sum) / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  let μ := average xs
  (xs.map (λ x => (x - μ)^2)).sum / xs.length

theorem sample_difference (x y : ℝ) :
  average (sample x y) = 10 ∧ variance (sample x y) = 4 →
  x - y = 2 ∨ x - y = -2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_difference_l1290_129011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_milk_percentage_l1290_129096

theorem initial_milk_percentage
  (initial_volume : ℝ)
  (added_water : ℝ)
  (final_milk_percentage : ℝ)
  (h1 : initial_volume = 60)
  (h2 : added_water = 18.75)
  (h3 : final_milk_percentage = 64)
  (h4 : initial_volume > 0)
  (h5 : added_water > 0)
  (h6 : final_milk_percentage > 0 ∧ final_milk_percentage < 100) :
  let final_volume := initial_volume + added_water
  let initial_milk_volume := (final_milk_percentage / 100) * final_volume
  initial_milk_volume / initial_volume * 100 = 84 := by
  sorry

#check initial_milk_percentage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_milk_percentage_l1290_129096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circumcircle_l1290_129019

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := x^2 = 4*y

/-- The focus of the parabola -/
def focus : ℝ × ℝ := (0, 1)

/-- The intersection point of the axis of symmetry and y-axis -/
def Q : ℝ × ℝ := (0, -1)

/-- The line passing through Q and P -/
def line_through_Q_and_P (P : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | ∃ t : ℝ, (x, y) = (1 - t) • Q + t • P}

/-- The circumcircle equation -/
def circumcircle_equation (x y : ℝ) : Prop :=
  (x - 1)^2 + y^2 = 2 ∨ (x + 1)^2 + y^2 = 2

/-- The theorem statement -/
theorem parabola_circumcircle :
  ∀ P : ℝ × ℝ,
  parabola P.1 P.2 →
  P ∈ line_through_Q_and_P P →
  ∀ x y : ℝ,
  (x, y) ∈ {(x, y) | (x - focus.1)^2 + (y - focus.2)^2 = 
            (x - P.1)^2 + (y - P.2)^2 ∧
            (x - P.1)^2 + (y - P.2)^2 = 
            (x - Q.1)^2 + (y - Q.2)^2} →
  circumcircle_equation x y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circumcircle_l1290_129019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l1290_129033

def is_valid_digit (d : ℕ) : Prop := 1 ≤ d ∧ d ≤ 9

def digit_product (n : ℕ) : ℕ :=
  (n / 100) * ((n / 10) % 10) * (n % 10)

def is_valid_number (n : ℕ) : Bool :=
  100 ≤ n ∧ n ≤ 999 ∧ digit_product n = 30

theorem count_valid_numbers :
  (Finset.filter (fun n => is_valid_number n) (Finset.range 1000)).card = 12 := by
  sorry

#eval (Finset.filter (fun n => is_valid_number n) (Finset.range 1000)).card

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l1290_129033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_base_side_length_l1290_129071

/-- The side length of the base of a right pyramid with an equilateral triangular base, given the area of one lateral face and the slant height. -/
noncomputable def base_side_length (lateral_face_area : ℝ) (slant_height : ℝ) : ℝ :=
  2 * lateral_face_area / slant_height

theorem pyramid_base_side_length :
  let lateral_face_area : ℝ := 90
  let slant_height : ℝ := 15
  base_side_length lateral_face_area slant_height = 12 := by
  -- Unfold the definition of base_side_length
  unfold base_side_length
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_base_side_length_l1290_129071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1290_129051

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x + 2

-- Define the interval
def I : Set ℝ := Set.Icc (-1) 2

-- State the theorem
theorem f_properties :
  (StrictMono f) ∧
  (∃ x ∈ I, ∀ y ∈ I, f x ≤ f y) ∧
  (f (-1) = -1) ∧
  (∃ x ∈ I, ∀ y ∈ I, f y ≤ f x) ∧
  (f 2 = 8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1290_129051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_closer_to_origin_l1290_129000

-- Define the rectangle
def rectangle : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 3 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2}

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the set of points closer to origin than to (4,1)
def closer_to_origin : Set (ℝ × ℝ) :=
  {p ∈ rectangle | distance p (0, 0) < distance p (4, 1)}

-- State the theorem
theorem probability_closer_to_origin :
  (MeasureTheory.volume closer_to_origin) / (MeasureTheory.volume rectangle) = 3/8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_closer_to_origin_l1290_129000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_9_equals_27_l1290_129035

/-- An arithmetic sequence with a specified condition -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence (using rationals instead of reals)
  d : ℚ      -- Common difference
  h1 : ∀ n, a (n + 1) = a n + d  -- Definition of arithmetic sequence
  h2 : a 2 = 3 * a 4 - 6  -- Given condition

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (2 * seq.a 1 + (n - 1) * seq.d) / 2

/-- The sum of the first 9 terms equals 27 -/
theorem sum_9_equals_27 (seq : ArithmeticSequence) : sum_n seq 9 = 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_9_equals_27_l1290_129035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_metal_mass_l1290_129027

/-- Represents the mass of each metal in the alloy -/
structure AlloyComposition where
  m1 : ℝ  -- mass of first metal
  m2 : ℝ  -- mass of second metal
  m3 : ℝ  -- mass of third metal
  m4 : ℝ  -- mass of fourth metal

/-- Checks if the given alloy composition satisfies the problem conditions -/
def isValidComposition (a : AlloyComposition) : Prop :=
  a.m1 + a.m2 + a.m3 + a.m4 = 20 ∧  -- total mass is 20 kg
  a.m1 = 1.5 * a.m2 ∧               -- first metal is 1.5 times second
  3 * a.m3 = 4 * a.m2 ∧             -- ratio of second to third is 3:4
  5 * a.m4 = 6 * a.m3               -- ratio of third to fourth is 5:6

/-- The main theorem stating the mass of the fourth metal -/
theorem fourth_metal_mass (a : AlloyComposition) 
  (h : isValidComposition a) : 
  |a.m4 - 5.89| < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_metal_mass_l1290_129027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_length_is_4pi_l1290_129064

/-- The length of the parametric curve described by (x,y) = (2 sin t, 2 cos t) for t ∈ [0, 2π] -/
noncomputable def parametric_curve_length : ℝ := 4 * Real.pi

/-- The parametric equations of the curve -/
noncomputable def curve (t : ℝ) : ℝ × ℝ := (2 * Real.sin t, 2 * Real.cos t)

theorem curve_length_is_4pi :
  ∫ t in (0 : ℝ)..(2 * Real.pi), Real.sqrt ((2 * Real.cos t)^2 + (-(2 * Real.sin t))^2) = parametric_curve_length := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_length_is_4pi_l1290_129064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_mod_15_l1290_129002

theorem sum_remainder_mod_15 (a b c : ℕ) 
  (ha : a % 15 = 11)
  (hb : b % 15 = 12)
  (hc : c % 15 = 13) : 
  (a + b + c) % 15 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_mod_15_l1290_129002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_dot_product_l1290_129077

-- Define the circle
def myCircle (x y : ℝ) : Prop := (x - 2)^2 + (y - 3)^2 = 1

-- Define the line passing through A(0,1) with direction vector (1,k)
def myLine (k x y : ℝ) : Prop := y = k * x + 1

-- Define the intersection points M and N
def intersectionPoints (k : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    myCircle x₁ y₁ ∧ myCircle x₂ y₂ ∧ 
    myLine k x₁ y₁ ∧ myLine k x₂ y₂ ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂)

-- Theorem statement
theorem intersection_and_dot_product (k : ℝ) :
  (intersectionPoints k ↔ (4 - Real.sqrt 7) / 3 < k ∧ k < (4 + Real.sqrt 7) / 3) ∧
  (∀ x₁ y₁ x₂ y₂ : ℝ, myCircle x₁ y₁ → myCircle x₂ y₂ → myLine k x₁ y₁ → myLine k x₂ y₂ →
    (x₁ - 0) * (x₂ - 0) + (y₁ - 1) * (y₂ - 1) = 7) ∧
  (∃ x₁ y₁ x₂ y₂ : ℝ, myCircle x₁ y₁ ∧ myCircle x₂ y₂ ∧ myLine k x₁ y₁ ∧ myLine k x₂ y₂ ∧
    x₁ * x₂ + y₁ * y₂ = 12 → k = 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_dot_product_l1290_129077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_percentage_both_services_l1290_129029

theorem greatest_percentage_both_services (internet_percentage snacks_percentage : ℝ) 
  (h_internet : internet_percentage = 0.45)
  (h_snacks : snacks_percentage = 0.7) :
  max 0 (internet_percentage + snacks_percentage - 1) ≤ 
    (min internet_percentage snacks_percentage) ∧ 
  (min internet_percentage snacks_percentage) = 0.45 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_percentage_both_services_l1290_129029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alex_biking_speed_l1290_129048

/-- Calculates the average speed given distance and time -/
noncomputable def average_speed (distance : ℝ) (time : ℝ) : ℝ :=
  distance / time

/-- The problem statement -/
theorem alex_biking_speed :
  let total_distance : ℝ := 48
  let biking_time : ℝ := 6
  average_speed total_distance biking_time = 8 := by
  -- Unfold the definition of average_speed
  unfold average_speed
  -- Simplify the expression
  simp
  -- Prove the equality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alex_biking_speed_l1290_129048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_equals_interval_l1290_129040

-- Define the set S as {x | x ≤ 1}
def S : Set ℝ := {x : ℝ | x ≤ 1}

-- Define the interval (-∞, 1]
def I : Set ℝ := Set.Iic 1

-- Theorem stating that S is equal to I
theorem set_equals_interval : S = I := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_equals_interval_l1290_129040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_winning_positions_count_l1290_129041

def winning_position (k : ℕ) : Bool :=
  if k % 7 = 2 ∨ k % 7 = 0 then false else true

def count_winning_positions (n : ℕ) : ℕ :=
  (List.range n).filter winning_position |>.length

theorem winning_positions_count :
  count_winning_positions 100 = 71 := by
  -- The proof would go here, but we'll use sorry for now
  sorry

#eval count_winning_positions 100

end NUMINAMATH_CALUDE_ERRORFEEDBACK_winning_positions_count_l1290_129041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_job_completion_time_correct_l1290_129073

/-- The time it takes to complete a job given two workers with different rates and a specific work schedule. -/
def job_completion_time 
  (x_rate : ℚ) -- Rate at which x completes the job (1/20 per day)
  (y_rate : ℚ) -- Rate at which y completes the job (1/12 per day)
  (x_solo_days : ℕ) -- Number of days x works alone (4 days)
  : ℕ :=
  let x_work := x_solo_days * x_rate
  let remaining_work := 1 - x_work
  let combined_rate := x_rate + y_rate
  let combined_days := (remaining_work / combined_rate).ceil.toNat
  x_solo_days + combined_days

theorem job_completion_time_correct
  (x_rate y_rate : ℚ)
  (x_solo_days : ℕ)
  (h1 : x_rate = 1 / 20)
  (h2 : y_rate = 1 / 12)
  (h3 : x_solo_days = 4)
  : job_completion_time x_rate y_rate x_solo_days = 10 :=
by
  -- The proof would go here
  sorry

#eval job_completion_time (1/20) (1/12) 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_job_completion_time_correct_l1290_129073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_sum_parity_difference_l1290_129056

/-- The number of positive integer divisors of n, including 1 and n -/
def τ (n : ℕ) : ℕ := sorry

/-- The sum of τ(k) for k from 1 to n -/
def S (n : ℕ) : ℕ := sorry

/-- The number of positive integers n ≤ 1000 with S(n) odd -/
def a : ℕ := sorry

/-- The number of positive integers n ≤ 1000 with S(n) even -/
def b : ℕ := sorry

theorem divisor_sum_parity_difference : |Int.ofNat a - Int.ofNat b| = 54 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_sum_parity_difference_l1290_129056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_range_l1290_129079

/-- The circle C with equation x² + y² = 1 -/
def Circle (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- The hyperbola with equation x²/4 - y²/9 = 1 -/
def Hyperbola (x y : ℝ) : Prop := x^2/4 - y^2/9 = 1

/-- A point on the hyperbola -/
structure PointOnHyperbola where
  x : ℝ
  y : ℝ
  onHyperbola : Hyperbola x y

/-- A point on the circle -/
structure PointOnCircle where
  x : ℝ
  y : ℝ
  onCircle : Circle x y

/-- The dot product of two vectors -/
def dotProduct (x1 y1 x2 y2 : ℝ) : ℝ := x1 * x2 + y1 * y2

/-- IsTangent is a proposition that A and B are tangent points on the circle for P -/
def IsTangent (P : PointOnHyperbola) (A B : PointOnCircle) : Prop := sorry

/-- The theorem stating the range of the dot product PA · PB -/
theorem dot_product_range (P : PointOnHyperbola) (A B : PointOnCircle) 
  (hTangent : IsTangent P A B) : 
  ∃ (d : ℝ), d ≥ 3/2 ∧ dotProduct (A.x - P.x) (A.y - P.y) (B.x - P.x) (B.y - P.y) = d := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_range_l1290_129079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_blocks_fit_l1290_129066

/-- The dimensions of the box -/
def box_dimensions : Fin 3 → ℚ
| 0 => 4
| 1 => 6
| 2 => 3
| _ => 0

/-- The dimensions of a block -/
def block_dimensions : Fin 3 → ℚ
| 0 => 3/2
| 1 => 2
| 2 => 2
| _ => 0

/-- The volume of the box -/
def box_volume : ℚ := (box_dimensions 0) * (box_dimensions 1) * (box_dimensions 2)

/-- The volume of a block -/
def block_volume : ℚ := (block_dimensions 0) * (block_dimensions 1) * (block_dimensions 2)

/-- The maximum number of blocks that can fit in the box based on volume -/
def max_blocks_by_volume : ℕ := (box_volume / block_volume).floor.toNat

/-- Theorem stating that the maximum number of blocks that can fit in the box is 12 -/
theorem max_blocks_fit (h : max_blocks_by_volume = 12) : 
  ∃ (arrangement : Fin 3 → ℕ), 
    (∀ i, (arrangement i : ℚ) * (block_dimensions i) ≤ box_dimensions i) ∧ 
    (arrangement 0 * arrangement 1 * arrangement 2 = 12) := by
  sorry

#eval max_blocks_by_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_blocks_fit_l1290_129066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_length_l1290_129093

/-- Represents a parabola in the form y^2 = 2px --/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Represents a point in 2D space --/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line passing through two points --/
structure Line where
  p1 : Point
  p2 : Point

/-- The focus of a parabola --/
noncomputable def focus (parabola : Parabola) : Point :=
  { x := parabola.p / 2, y := 0 }

/-- Theorem: Given a parabola and a line passing through its focus and a point (4,4),
    prove that the parabola's equation is y^2 = 4x and the length of the segment
    between the intersection points is 25/4 --/
theorem parabola_intersection_length 
  (Γ : Parabola) 
  (l : Line) 
  (h1 : l.p1 = focus Γ) 
  (h2 : l.p2 = { x := 4, y := 4 }) :
  Γ.p = 2 ∧ 
  let B : Point := { x := 1/4, y := -1 }
  (B.x - l.p2.x)^2 + (B.y - l.p2.y)^2 = (25/4)^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_length_l1290_129093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_midpoint_distance_l1290_129076

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 9

-- Define the hyperbola
def hyperbola_eq (x y : ℝ) : Prop := x^2/9 - y^2/16 = 1

-- Define the left focus of the hyperbola
def left_focus : ℝ × ℝ := (-5, 0)

-- Define a point on the circle
noncomputable def T : ℝ × ℝ := sorry

-- Define a point on the right branch of the hyperbola
noncomputable def P : ℝ × ℝ := sorry

-- Define the midpoint of FP
noncomputable def M : ℝ × ℝ := ((left_focus.1 + P.1) / 2, (left_focus.2 + P.2) / 2)

-- Define the origin
def O : ℝ × ℝ := (0, 0)

-- Theorem statement
theorem tangent_midpoint_distance :
  circle_eq T.1 T.2 ∧
  hyperbola_eq P.1 P.2 ∧
  (∃ k : ℝ, P = left_focus + k • (T - left_focus)) →
  |M.1 - O.1| + |M.2 - O.2| - (|M.1 - T.1| + |M.2 - T.2|) = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_midpoint_distance_l1290_129076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_diameter_theorem_l1290_129003

-- Define the line l: y = x + m
def line (m : ℝ) (x y : ℝ) : Prop := y = x + m

-- Define the circle C: x^2 + y^2 - 2x + 4y - 4 = 0
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y - 4 = 0

-- Define the condition for intersection at two distinct points
def intersect_at_two_points (m : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂, x₁ ≠ x₂ ∧ line m x₁ y₁ ∧ line m x₂ y₂ ∧ circle_eq x₁ y₁ ∧ circle_eq x₂ y₂

-- Define the range of m
def m_range (m : ℝ) : Prop := -3 - 3*Real.sqrt 2 < m ∧ m < -3 + 3*Real.sqrt 2

-- Define the condition for the circle with diameter AB passing through the origin
def circle_through_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  (x₁*x₂ + y₁*y₂ = 0) ∧ 
  ((line (-4) x₁ y₁ ∧ line (-4) x₂ y₂ ∧ circle_eq x₁ y₁ ∧ circle_eq x₂ y₂) ∨
   (line 1 x₁ y₁ ∧ line 1 x₂ y₂ ∧ circle_eq x₁ y₁ ∧ circle_eq x₂ y₂))

theorem intersection_and_diameter_theorem :
  (∀ m, intersect_at_two_points m ↔ m_range m) ∧
  (∀ x₁ y₁ x₂ y₂, circle_through_origin x₁ y₁ x₂ y₂ → 
    (line (-4) x₁ y₁ ∧ line (-4) x₂ y₂) ∨ (line 1 x₁ y₁ ∧ line 1 x₂ y₂)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_diameter_theorem_l1290_129003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ABE_l1290_129004

-- Define the points
variable (A B C D E : ℝ × ℝ)

-- Define the conditions
axiom right_triangle_ABC : (A.1 - B.1) * (C.1 - B.1) + (A.2 - B.2) * (C.2 - B.2) = 0
axiom right_triangle_ABD : (A.1 - B.1) * (D.1 - B.1) + (A.2 - B.2) * (D.2 - B.2) = 0
axiom AB_length : ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 64
axiom AC_length : ((A.1 - C.1)^2 + (A.2 - C.2)^2) = 36
axiom BD_length : ((B.1 - D.1)^2 + (B.2 - D.2)^2) = 64
axiom E_midpoint : E = ((C.1 + D.1) / 2, (C.2 + D.2) / 2)

-- Define the theorem
theorem area_of_triangle_ABE :
  let area := ((A.1 - B.1) * (E.2 - B.2) - (A.2 - B.2) * (E.1 - B.1)) / 2
  area^2 = 368 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ABE_l1290_129004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pedestrians_collinear_at_most_twice_l1290_129059

/-- Represents a point moving at constant velocity in 2D space -/
structure MovingPoint where
  initial_position : ℝ × ℝ
  velocity : ℝ × ℝ

/-- The collinearity condition function for three moving points -/
def collinearity_function (p1 p2 p3 : MovingPoint) (t : ℝ) : ℝ :=
  let v1 := (p2.initial_position.1 - p1.initial_position.1 + t * (p2.velocity.1 - p1.velocity.1),
             p2.initial_position.2 - p1.initial_position.2 + t * (p2.velocity.2 - p1.velocity.2))
  let v2 := (p3.initial_position.1 - p1.initial_position.1 + t * (p3.velocity.1 - p1.velocity.1),
             p3.initial_position.2 - p1.initial_position.2 + t * (p3.velocity.2 - p1.velocity.2))
  v1.1 * v2.2 - v1.2 * v2.1

/-- Three pedestrians walking on straight roads at constant speeds -/
def pedestrian_problem (p1 p2 p3 : MovingPoint) : Prop :=
  -- The pedestrians are not collinear at t = 0
  collinearity_function p1 p2 p3 0 ≠ 0 ∧
  -- The collinearity function is not identically zero
  ∃ t, collinearity_function p1 p2 p3 t ≠ 0

theorem pedestrians_collinear_at_most_twice (p1 p2 p3 : MovingPoint) 
  (h : pedestrian_problem p1 p2 p3) :
  ∃ (t1 t2 : ℝ), ∀ t, collinearity_function p1 p2 p3 t = 0 → t = t1 ∨ t = t2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pedestrians_collinear_at_most_twice_l1290_129059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1290_129083

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos (2 * x)

theorem f_range : Set.range f = Set.Icc (-2) (9/8) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1290_129083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_geq_0_range_of_a_l1290_129080

-- Define the function f
def f (x : ℝ) : ℝ := |(-2 * x + 4)| - |x + 6|

-- Theorem for the solution set of f(x) ≥ 0
theorem solution_set_f_geq_0 :
  {x : ℝ | f x ≥ 0} = Set.Iic (-2/3) ∪ Set.Ici 10 :=
sorry

-- Theorem for the range of a
theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, f x > a + |x - 2|) → a < 8 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_geq_0_range_of_a_l1290_129080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_academic_performance_correlations_l1290_129021

-- Define the variables
def student_attitude : ℝ → ℝ := sorry
def teacher_level : ℝ → ℝ := sorry
def student_height : ℝ → ℝ := sorry
def academic_performance : ℝ → ℝ := sorry

-- Define correlation
def is_correlated (f g : ℝ → ℝ) : Prop :=
  ∃ (c : ℝ), c ≠ 0 ∧ ∀ x, |f x - g x| ≤ c

-- State the theorem
theorem academic_performance_correlations :
  (is_correlated student_attitude academic_performance) ∧
  (is_correlated teacher_level academic_performance) ∧
  ¬(is_correlated student_height academic_performance) := by
  sorry

#check academic_performance_correlations

end NUMINAMATH_CALUDE_ERRORFEEDBACK_academic_performance_correlations_l1290_129021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_is_equilateral_l1290_129063

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- State the theorem
theorem triangle_is_equilateral (t : Triangle) 
  (h1 : t.a^2 + t.b^2 - t.c^2 = t.a * t.b)
  (h2 : 2 * Real.cos t.A * Real.sin t.B = Real.sin t.C) :
  t.a = t.b ∧ t.b = t.c ∧ t.A = t.B ∧ t.B = t.C ∧ t.C = Real.pi / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_is_equilateral_l1290_129063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_and_zeros_l1290_129023

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - x * Real.log x

-- Define the function F
noncomputable def F (a : ℝ) (x : ℝ) : ℝ := x^2 + (x - 1) * Real.log x + f a x + a

-- Theorem statement
theorem extreme_value_and_zeros 
  (a : ℝ) 
  (h1 : ∃ (ε : ℝ), ∀ (x : ℝ), x ≠ Real.exp (-2) → |x - Real.exp (-2)| < ε → |f a x| ≤ |f a (Real.exp (-2))|)
  (h2 : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ F a x₁ = 0 ∧ F a x₂ = 0) :
  a = -1 ∧ ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ F a x₁ = 0 ∧ F a x₂ = 0 ∧ x₁ + x₂ > 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_and_zeros_l1290_129023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_books_theorem_l1290_129045

/-- The number of books read by the entire student body in one year -/
def total_books_read (c s : ℕ) : ℕ :=
  84 * c * s

/-- The number of classes in the school -/
def c : ℕ := 1  -- Placeholder value, can be changed as needed

/-- The number of students in each class -/
def s : ℕ := 1  -- Placeholder value, can be changed as needed

/-- The number of books each student reads per month -/
def books_per_month : ℕ := 7

/-- The number of months in a year -/
def months_per_year : ℕ := 12

theorem total_books_theorem (c s : ℕ) :
  total_books_read c s = books_per_month * months_per_year * c * s :=
by
  -- Expand the definition of total_books_read
  unfold total_books_read
  -- Expand the definitions of books_per_month and months_per_year
  unfold books_per_month
  unfold months_per_year
  -- The equality now holds by reflexivity
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_books_theorem_l1290_129045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_on_interval_l1290_129031

-- Define the function as noncomputable due to the use of Real.sqrt
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (-x^2 + 6*x - 5)

-- State the theorem
theorem f_monotone_increasing_on_interval :
  MonotoneOn f (Set.Icc 1 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_on_interval_l1290_129031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_valued_polynomials_with_tau_condition_l1290_129098

/-- τ(n) is the number of positive divisors of n -/
def tau (n : ℕ) : ℕ := (Nat.divisors n).card

/-- An integer-valued polynomial is a function ℕ → ℕ -/
def IntegerValuedPolynomial : Type := ℕ → ℕ

theorem integer_valued_polynomials_with_tau_condition 
  (f g : IntegerValuedPolynomial) 
  (h : ∀ x : ℕ, tau (f x) = g x) : 
  ∃ k : ℕ, (∀ x : ℕ, f x = k) ∧ (∀ x : ℕ, g x = tau k) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_valued_polynomials_with_tau_condition_l1290_129098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_properties_l1290_129028

/-- The trajectory of a point P in the Cartesian plane, where the sum of distances
    from P to (0, -√3) and (0, √3) is 4. -/
def TrajectoryC (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  (x^2 + (y + Real.sqrt 3)^2)^(1/2) + (x^2 + (y - Real.sqrt 3)^2)^(1/2) = 4

theorem trajectory_properties :
  ∀ P : ℝ × ℝ, TrajectoryC P →
    (∃ x y : ℝ, P = (x, y) ∧ x^2/4 + y^2 = 1) ∧
    (∀ x y : ℝ, P = (x, y) → (x = 2 ∨ x = -2 ∨ y = 1 ∨ y = -1)) ∧
    4 = 2 * 2 ∧  -- Major axis length
    2 = 2 * 1 ∧  -- Minor axis length
    Real.sqrt 3 / 2 = (Real.sqrt ((2 : ℝ)^2 - 1^2)) / 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_properties_l1290_129028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_partitions_l1290_129075

/-- The number of ways to partition a set of n elements into four subsets
    satisfying the given conditions. -/
def validPartitions (n : ℕ) : ℕ :=
  16^n - 3 * 12^n + 2 * 10^n + 9^n - 8^n

/-- Theorem stating that validPartitions counts the number of valid partitions. -/
theorem count_valid_partitions (n : ℕ) :
  validPartitions n = (Finset.univ.powerset.filter (λ E₂ : Finset (Fin n) ↦
    ∃ E₁ E₃ E₄ : Finset (Fin n),
      E₂.Nonempty ∧
      (E₂ ∩ E₃).Nonempty ∧
      (E₃ ∩ E₄).Nonempty ∧
      E₁ ∪ E₂ ∪ E₃ ∪ E₄ = Finset.univ)).card :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_partitions_l1290_129075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_7_l1290_129036

/-- Geometric sequence with positive common ratio -/
noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

/-- Sum of first n terms of a geometric sequence -/
noncomputable def geometric_sum (a : ℕ → ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  (a 1) * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum_7 (a : ℕ → ℝ) (q : ℝ) :
  geometric_sequence a q →
  a 1 = 1 →
  a 5 = 16 →
  geometric_sum a q 7 = 127 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_7_l1290_129036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omar_coffee_remaining_l1290_129091

/-- Calculates the remaining coffee after Omar's drinking pattern --/
noncomputable def remaining_coffee (initial_amount : ℝ) : ℝ :=
  let after_first_drink := initial_amount * (1 - 1/4)
  let after_second_drink := after_first_drink * (1 - 1/2)
  after_second_drink - 1

/-- Theorem stating that 3.5 ounces of coffee remain after Omar's drinking pattern --/
theorem omar_coffee_remaining :
  remaining_coffee 12 = 3.5 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_omar_coffee_remaining_l1290_129091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divisors_of_50_factorial_l1290_129043

/-- The number of prime divisors of 50 factorial -/
def num_prime_divisors_50_factorial : ℕ := 15

/-- 50 factorial -/
def factorial_50 : ℕ := Nat.factorial 50

theorem prime_divisors_of_50_factorial :
  (Finset.filter Nat.Prime (Nat.divisors factorial_50)).card = num_prime_divisors_50_factorial := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divisors_of_50_factorial_l1290_129043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_quadrilateral_area_l1290_129065

/-- Represents a quadrilateral with a diagonal and two adjacent sides -/
structure Quadrilateral where
  diagonal : ℝ
  side1 : ℝ
  side2 : ℝ
  angle1 : ℝ
  angle2 : ℝ

/-- Calculates the area of the quadrilateral -/
noncomputable def quadrilateralArea (q : Quadrilateral) : ℝ :=
  let area1 := 0.5 * q.side1 * q.diagonal * Real.sin (q.angle1 * Real.pi / 180)
  let area2 := 0.5 * q.side2 * q.diagonal * Real.sin (q.angle2 * Real.pi / 180)
  area1 + area2

/-- The theorem stating the area of the specific quadrilateral -/
theorem specific_quadrilateral_area :
  let q : Quadrilateral := {
    diagonal := 20,
    side1 := 9,
    side2 := 6,
    angle1 := 35,
    angle2 := 110
  }
  abs (quadrilateralArea q - 108.006) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_quadrilateral_area_l1290_129065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_l1290_129050

def a : ℕ → ℚ
| 0 => 1
| n + 1 => 2 * a n / (2 + a n)

theorem a_formula : ∀ n : ℕ, a n = 2 / (n + 1 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_l1290_129050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_C₁_polar_equation_l1290_129032

-- Define the curves and conditions
noncomputable def C₁ (t α : ℝ) : ℝ × ℝ := (1 + t * Real.cos α, 1 + t * Real.sin α)

noncomputable def C₂ (θ : ℝ) : ℝ := 4 * Real.cos θ

def P : ℝ × ℝ := (1, 1)

noncomputable def A : ℝ × ℝ := sorry
noncomputable def B : ℝ × ℝ := sorry

axiom α_range : ∀ α, 0 ≤ α ∧ α < Real.pi

axiom intersection_condition : 
  1 / ((A.1 - P.1)^2 + (A.2 - P.2)^2) + 1 / ((B.1 - P.1)^2 + (B.2 - P.2)^2) = 1

-- Theorem to prove
theorem C₁_polar_equation :
  ∃ (ρ : ℝ), C₁ (ρ * Real.cos (Real.pi/4) - 1) (Real.pi/4) = 
    (ρ * Real.cos (Real.pi/4), ρ * Real.sin (Real.pi/4)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_C₁_polar_equation_l1290_129032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1290_129081

noncomputable def f (x : ℝ) : ℝ := (3 * x + 1) / (x - 2)

theorem range_of_f :
  {y : ℝ | ∃ x : ℝ, f x = y} = {y : ℝ | y ≠ 3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1290_129081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l1290_129074

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

-- State the theorem
theorem f_inequality (a b : ℝ) (h1 : b > a) (h2 : a > 3) :
  f b < f ((a + b) / 2) ∧ f ((a + b) / 2) < f (Real.sqrt (a * b)) ∧ f (Real.sqrt (a * b)) < f a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l1290_129074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1290_129068

-- Define the constant a and function f
variable (a : ℝ) (h : a > 1)
noncomputable def f (x : ℝ) : ℝ := (a^x - 1) / (a^x + 1)

-- Theorem for the three parts of the problem
theorem f_properties (a : ℝ) (h : a > 1) :
  (∀ x, f a (-x) = -(f a x)) ∧
  StrictMono (f a) ∧
  Set.range (f a) = Set.Ioo (-1 : ℝ) 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1290_129068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_perimeter_product_l1290_129084

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Rectangle ABCD -/
structure Rectangle where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Calculate the area of a rectangle -/
noncomputable def area (r : Rectangle) : ℝ :=
  distance r.A r.B * distance r.A r.D

/-- Calculate the perimeter of a rectangle -/
noncomputable def perimeter (r : Rectangle) : ℝ :=
  2 * (distance r.A r.B + distance r.A r.D)

/-- The main theorem -/
theorem rectangle_area_perimeter_product :
  let r := Rectangle.mk
    (Point.mk 3 4) (Point.mk 4 1)
    (Point.mk 2 0) (Point.mk 1 3)
  area r * perimeter r = 20 * Real.sqrt 5 + 10 * Real.sqrt 10 := by
  sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_perimeter_product_l1290_129084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_set_is_line_l1290_129094

/-- A point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- The set of points satisfying θ = π/4 in polar coordinates -/
def LineSet : Set PolarPoint :=
  {p : PolarPoint | p.θ = Real.pi/4 ∨ p.θ = 5*Real.pi/4}

/-- Definition of a line in polar coordinates -/
def IsLine (s : Set PolarPoint) : Prop :=
  ∃ θ₀ : ℝ, ∀ p ∈ s, p.θ = θ₀ ∨ p.θ = θ₀ + Real.pi

theorem line_set_is_line : IsLine LineSet := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_set_is_line_l1290_129094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_always_perpendicular_to_plane_l1290_129099

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Line → Prop)
variable (perpendicular_plane : Plane → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (intersect : Plane → Plane → Line → Prop)
variable (contained : Line → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)

-- State the theorem
theorem not_always_perpendicular_to_plane 
  (m n : Line) (α β : Plane) 
  (h_diff_lines : m ≠ n) 
  (h_diff_planes : α ≠ β) 
  (h_perp_planes : perpendicular_plane α β) 
  (h_intersect : intersect α β n) 
  (h_perp_lines : perpendicular m n) :
  ¬ (∀ m n α β, perpendicular m n → perpendicular_plane α β → 
    intersect α β n → perpendicular m n → perpendicular_line_plane m β) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_always_perpendicular_to_plane_l1290_129099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leading_coefficient_is_five_l1290_129039

noncomputable def polynomial (x : ℝ) : ℝ := 5 * (x^4 - 2*x^3 + 3*x) - 9 * (x^4 - x^2 + 1) + 3 * (3*x^4 + x^3 - 2)

theorem leading_coefficient_is_five : 
  ∃ (p : Polynomial ℝ), ∀ x, polynomial x = (5 : ℝ) * x^4 + (p.eval x) ∧ p.degree < 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_leading_coefficient_is_five_l1290_129039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_diameter_triangle_DEF_l1290_129009

/-- The diameter of the inscribed circle in a triangle -/
noncomputable def inscribed_circle_diameter (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  2 * area / s

/-- The theorem stating the diameter of the inscribed circle in the given triangle -/
theorem inscribed_circle_diameter_triangle_DEF :
  inscribed_circle_diameter 13 8 15 = 10 * Real.sqrt 3 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_diameter_triangle_DEF_l1290_129009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangency_l1290_129014

/-- Two circles are tangent if and only if the distance between their centers
    is equal to the sum or difference of their radii -/
theorem circle_tangency (O M : EuclideanSpace ℝ (Fin 2)) (R r : ℝ) :
  (∃ (P : EuclideanSpace ℝ (Fin 2)), ‖P - O‖ = R ∧ ‖P - M‖ = r) ↔ 
  (r = ‖M - O‖ + R ∨ r = |‖M - O‖ - R|) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangency_l1290_129014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l1290_129026

/-- 
The equation of a line passing through a point (x₀, y₀) with slope m 
can be represented as y - y₀ = m(x - x₀).
-/
def point_slope_form (x₀ y₀ m : ℝ) (x y : ℝ) : Prop :=
  y - y₀ = m * (x - x₀)

/-- The slope of a line can be calculated from its slope angle α as tan(α). -/
noncomputable def slope_from_angle (α : ℝ) : ℝ := Real.tan α

theorem line_equation_proof (α : ℝ) (h : Real.tan α = 4/3) :
  ∃ (A B C : ℝ), A * 2 + B * 1 + C = 0 ∧ 
                 (∀ x y : ℝ, A * x + B * y + C = 0 ↔ 
                             point_slope_form 2 1 (slope_from_angle α) x y) ∧
                 (A, B, C) = (4, -3, -5) := by
  sorry

#check line_equation_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l1290_129026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_l1290_129013

/-- Circle C defined parametrically -/
noncomputable def circleC (θ : ℝ) : ℝ × ℝ :=
  (1 + Real.cos θ, Real.sin θ)

/-- Line l defined parametrically -/
noncomputable def lineL (t α : ℝ) : ℝ × ℝ :=
  (2 + t * Real.cos α, Real.sqrt 3 + t * Real.sin α)

/-- Theorem stating the intersection condition -/
theorem line_intersects_circle (α : ℝ) :
  (∃ t θ : ℝ, lineL t α = circleC θ) ↔ π / 6 ≤ α ∧ α ≤ π / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_l1290_129013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_x_minus_one_lt_zero_l1290_129089

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  if x ≥ 0 then x - 1 else 1 - x

-- State the theorem
theorem solution_set_f_x_minus_one_lt_zero :
  (∀ x, f (-x) = f x) →  -- f is even
  (∀ x ≥ 0, f x = x - 1) →  -- f(x) = x - 1 for x ≥ 0
  {x | f (x - 1) < 0} = Set.Ioo 0 2 :=  -- solution set is (0, 2)
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_x_minus_one_lt_zero_l1290_129089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_and_decreasing_l1290_129078

-- Define the function
noncomputable def f (x : ℝ) : ℝ := 2^(-x) - 2^x

-- Theorem statement
theorem f_is_odd_and_decreasing :
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x y, x < y → f x > f y) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_and_decreasing_l1290_129078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_b_with_integer_roots_l1290_129046

/-- A function that checks if a quadratic equation x^2 - bx + 8b = 0 has only integer roots -/
def has_only_integer_roots (b : ℝ) : Prop :=
  ∃ r s : ℤ, r + s = b ∧ r * s = 8 * b

/-- The theorem stating that there are exactly 3 real numbers b such that 
    x^2 - bx + 8b = 0 has only integer roots -/
theorem count_b_with_integer_roots :
  ∃! (S : Finset ℝ), (∀ b ∈ S, has_only_integer_roots b) ∧ S.card = 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_b_with_integer_roots_l1290_129046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_translated_sine_function_l1290_129010

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x)

-- Define the translation amount
noncomputable def h : ℝ := Real.pi / 6

-- Define the translated function
noncomputable def g (x : ℝ) : ℝ := f (x + h)

-- Theorem statement
theorem translated_sine_function :
  ∀ x : ℝ, g x = Real.sin (2 * x + Real.pi / 3) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_translated_sine_function_l1290_129010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1290_129095

-- Define the ellipse C
def ellipse (a b : ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ x^2 / a^2 + y^2 / b^2 = 1

-- Define the line l
def line (m : ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ x = m * y + 1

-- Define the area of triangle MNR
def area_MNR (y1 y2 : ℝ) : ℝ :=
  |y1 - y2|

theorem ellipse_properties :
  ∀ a b : ℝ,
  a > b ∧ b > 0 →
  ellipse a b (-1) (3/2) →
  (∃ c : ℝ, c^2 = a^2 - b^2 ∧ c/a = 1/2) →
  (∃ m : ℝ, ∀ x y : ℝ, line m x y →
    (∃ y1 y2 : ℝ, y1 ≠ y2 ∧
      ellipse a b (m * y1 + 1) y1 ∧
      ellipse a b (m * y2 + 1) y2 ∧
      area_MNR y1 y2 ≤ 3)) →
  (ellipse 2 (Real.sqrt 3) = ellipse a b) ∧
  (∃ m : ℝ, ∀ x y : ℝ, line m x y →
    (∃ y1 y2 : ℝ, y1 ≠ y2 ∧
      ellipse 2 (Real.sqrt 3) (m * y1 + 1) y1 ∧
      ellipse 2 (Real.sqrt 3) (m * y2 + 1) y2 ∧
      area_MNR y1 y2 = 3)) ∧
  (line 0 = λ x y ↦ x = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1290_129095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_theorem_l1290_129017

noncomputable section

-- Define the parabola
def parabola (a : ℝ) (x y : ℝ) : Prop := y^2 = a * x

-- Define the focus of the parabola
def focus (a : ℝ) : ℝ × ℝ := (a / 4, 0)

-- Define a line passing through a point
def line_through_point (p : ℝ × ℝ) (x y : ℝ) : Prop :=
  ∃ (m b : ℝ), y = m * x + b ∧ p.2 = m * p.1 + b

-- Define the intersection of a line with the parabola
def intersects_parabola (a : ℝ) (p : ℝ × ℝ) (x y : ℝ) : Prop :=
  parabola a x y ∧ line_through_point p x y

-- Theorem statement
theorem parabola_intersection_theorem (a : ℝ) :
  (∃ (x₁ y₁ x₂ y₂ : ℝ),
    intersects_parabola a (focus a) x₁ y₁ ∧
    intersects_parabola a (focus a) x₂ y₂ ∧
    x₁ + x₂ = 8 ∧
    (x₂ - x₁)^2 + (y₂ - y₁)^2 = 12^2) →
  a = 8 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_theorem_l1290_129017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1290_129024

def a : ℝ × ℝ := (5, 1)
def b : ℝ × ℝ := (-3, 6)

theorem triangle_area : |a.1 * b.2 - a.2 * b.1| / 2 = 16.5 := by
  -- Calculate the determinant
  have det : a.1 * b.2 - a.2 * b.1 = 33 := by
    simp [a, b]
    norm_num
  
  -- Show that the absolute value of the determinant is 33
  have abs_det : |a.1 * b.2 - a.2 * b.1| = 33 := by
    rw [det]
    norm_num

  -- Prove the final result
  calc
    |a.1 * b.2 - a.2 * b.1| / 2 = 33 / 2 := by rw [abs_det]
    _ = 16.5 := by norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1290_129024
