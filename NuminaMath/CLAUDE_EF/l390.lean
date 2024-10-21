import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cycle_gain_percentage_l390_39022

/-- Calculate the gain percentage given the total cost and total sale price -/
noncomputable def gain_percentage (total_cost : ℝ) (total_sale_price : ℝ) : ℝ :=
  ((total_sale_price - total_cost) / total_cost) * 100

/-- The overall gain percentage for selling 3 cycles -/
theorem cycle_gain_percentage : 
  let total_cost : ℝ := 18000
  let total_sale_price : ℝ := 19500
  abs (gain_percentage total_cost total_sale_price - 8.33) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cycle_gain_percentage_l390_39022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_last_three_digits_l390_39081

/-- A function that returns the units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- A predicate that checks if three numbers satisfy the given condition -/
def satisfyCondition (a b c : ℕ) : Prop :=
  unitsDigit (a + b) = unitsDigit c ∧
  unitsDigit (b + c) = unitsDigit a ∧
  unitsDigit (c + a) = unitsDigit b

/-- The main theorem -/
theorem product_last_three_digits
  (a b c : ℕ)
  (ha : a > 1000)
  (hb : b > 1000)
  (hc : c > 1000)
  (h_condition : satisfyCondition a b c) :
  (a * b * c) % 1000 ∈ ({0, 250, 500, 750} : Set ℕ) := by
  sorry

#check product_last_three_digits

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_last_three_digits_l390_39081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficients_divisible_by_five_coefficients_divisible_by_seven_l390_39046

/-- A polynomial of degree 3 with integer coefficients -/
structure Polynomial3 where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ

/-- Evaluation of the polynomial at a given integer -/
def evaluate (p : Polynomial3) (x : ℤ) : ℤ :=
  p.a * x^3 + p.b * x^2 + p.c * x + p.d

/-- The polynomial is divisible by 5 for all integer inputs -/
def divisible_by_five (p : Polynomial3) : Prop :=
  ∀ x : ℤ, (evaluate p x) % 5 = 0

/-- Main theorem: If the polynomial is divisible by 5 for all integer inputs,
    then all its coefficients are divisible by 5 -/
theorem coefficients_divisible_by_five (p : Polynomial3) 
    (h : divisible_by_five p) : 
    (p.a % 5 = 0) ∧ (p.b % 5 = 0) ∧ (p.c % 5 = 0) ∧ (p.d % 5 = 0) := by
  sorry

-- Part b: Polynomial of degree 4 divisible by 7

/-- A polynomial of degree 4 with integer coefficients -/
structure Polynomial4 where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  e : ℤ

/-- Evaluation of the polynomial at a given integer -/
def evaluate4 (p : Polynomial4) (x : ℤ) : ℤ :=
  p.a * x^4 + p.b * x^3 + p.c * x^2 + p.d * x + p.e

/-- The polynomial is divisible by 7 for all integer inputs -/
def divisible_by_seven (p : Polynomial4) : Prop :=
  ∀ x : ℤ, (evaluate4 p x) % 7 = 0

/-- Main theorem for part b: If the polynomial is divisible by 7 for all integer inputs,
    then all its coefficients are divisible by 7 -/
theorem coefficients_divisible_by_seven (p : Polynomial4) 
    (h : divisible_by_seven p) : 
    (p.a % 7 = 0) ∧ (p.b % 7 = 0) ∧ (p.c % 7 = 0) ∧ (p.d % 7 = 0) ∧ (p.e % 7 = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficients_divisible_by_five_coefficients_divisible_by_seven_l390_39046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_critical_point_derivative_zero_derivative_zero_not_always_critical_point_l390_39018

open Function Real

/-- A point is critical if it's in the domain of f and f' is zero at that point,
    or if f is not differentiable at that point. -/
def IsCriticalPoint (f : ℝ → ℝ) (x : ℝ) : Prop :=
  (∀ ε > 0, ∃ δ > 0, ∀ y, |y - x| < δ → |f y - f x| ≤ ε * |y - x|) ∨
  (HasDerivAt f 0 x)

theorem critical_point_derivative_zero
  (f : ℝ → ℝ) (x : ℝ) (hf : Differentiable ℝ f) :
  IsCriticalPoint f x → HasDerivAt f 0 x :=
by sorry

theorem derivative_zero_not_always_critical_point :
  ∃ f : ℝ → ℝ, ∃ x : ℝ, Differentiable ℝ f ∧ HasDerivAt f 0 x ∧ ¬IsCriticalPoint f x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_critical_point_derivative_zero_derivative_zero_not_always_critical_point_l390_39018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PZM_l390_39075

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a polygon -/
structure Polygon where
  vertices : List Point

/-- Represents a triangle -/
structure Triangle where
  a : Point
  b : Point
  c : Point

/-- Represents a square -/
structure Square where
  side : ℝ

/-- Calculates the area of a triangle given its base, height, and angle -/
noncomputable def triangleArea (base height : ℝ) (angle : ℝ) : ℝ :=
  0.5 * base * height * Real.sin angle

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

theorem area_of_triangle_PZM 
  (LMNOPQ : Polygon)
  (LMXY QPRG : Square)
  (MYZ : Triangle)
  (P M Z : Point) :
  (∀ (v : Point), v ∈ LMNOPQ.vertices → true) →  -- Placeholder for equiangular hexagon condition
  LMXY.side^2 = 50 →  -- Area of LMXY
  QPRG.side^2 = 8 →   -- Area of QPRG
  MYZ.b = MYZ.c →     -- Isosceles triangle MYZ
  QPRG.side = distance P M →  -- QO = MP
  triangleArea (distance P M) (distance M Z) (Real.pi / 6) = 5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PZM_l390_39075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oscillation_difference_l390_39085

/-- Represents the distance between two fixed points --/
noncomputable def total_distance : ℝ := 3

/-- The fraction of remaining distance walked in each step --/
noncomputable def walk_fraction : ℝ := 2/3

/-- The limiting point when walking towards the first destination --/
noncomputable def limit_point_1 : ℝ := total_distance * walk_fraction / (1 + walk_fraction)

/-- The limiting point when walking towards the second destination --/
noncomputable def limit_point_2 : ℝ := total_distance - limit_point_1

/-- Theorem stating that the absolute difference between the two limiting points is 1 --/
theorem oscillation_difference :
  |limit_point_1 - limit_point_2| = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oscillation_difference_l390_39085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dan_balloon_count_l390_39012

/-- The number of violet balloons Tim has -/
def tim_balloons : ℝ := 4.142857143

/-- The factor by which Dan has more balloons than Tim -/
def dan_factor : ℝ := 7.0

/-- The number of violet balloons Dan has -/
def dan_balloons : ℝ := tim_balloons * dan_factor

theorem dan_balloon_count : ⌊dan_balloons⌋ = 29 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dan_balloon_count_l390_39012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_absolute_terms_l390_39005

def Sn (n : ℕ) : ℚ := -3/2 * n^2 + 205/2 * n

def Tn (n : ℕ) : ℚ :=
  if n ≤ 34 then
    -3/2 * (n : ℚ)^2 + 205/2 * n
  else
    3/2 * (n : ℚ)^2 - 205/2 * n + 3502

theorem sum_of_absolute_terms (n : ℕ) :
  Tn n = if n ≤ 34 then -3/2 * (n : ℚ)^2 + 205/2 * n else 3/2 * (n : ℚ)^2 - 205/2 * n + 3502 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_absolute_terms_l390_39005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_conversion_correct_l390_39017

/-- Conversion factor from m/s to km/h -/
noncomputable def meters_per_second_to_km_per_hour : ℝ := 3.6

/-- Original speed in m/s -/
noncomputable def speed_in_ms : ℝ := 13 / 54

/-- Converted speed in km/h -/
noncomputable def speed_in_kmh : ℝ := speed_in_ms * meters_per_second_to_km_per_hour

theorem speed_conversion_correct : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.005 ∧ |speed_in_kmh - 0.87| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_conversion_correct_l390_39017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_a_values_l390_39088

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - (min a b)^2 / (max a b)^2)

/-- The possible values of 'a' for an ellipse with given eccentricity -/
theorem ellipse_a_values (e : ℝ) :
  e = 1/2 →
  ∃ (a : ℝ), (∀ (x y : ℝ), x^2 / (a + 8) + y^2 / 9 = 1 → 
    eccentricity (Real.sqrt (a + 8)) 3 = e) ∧
    (a = 4 ∨ a = -5/4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_a_values_l390_39088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l390_39043

noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.cos x + Real.sqrt 3 * (Real.sin x)^2 - Real.sqrt 3 / 2

theorem f_properties :
  (∀ x, f x ≥ -1) ∧ 
  (∀ x, f (π/6 + x) = f (π/6 - x)) ∧
  (∀ x, f (x + π) = f x) ∧
  (∀ x, f x = Real.sin (2 * (x - π/6))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l390_39043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_multiple_zeros_iff_a_in_range_l390_39058

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 + a*x else 2*a*x - 5

-- Define the condition for g(x) having more than one zero
def has_multiple_zeros (a : ℝ) : Prop :=
  ∀ b : ℝ, ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = b ∧ f a x₂ = b

-- State the theorem
theorem f_multiple_zeros_iff_a_in_range :
  ∀ a : ℝ, has_multiple_zeros a ↔ a < 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_multiple_zeros_iff_a_in_range_l390_39058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_my_sequence_2016_l390_39001

def my_sequence (n : ℕ) : ℚ :=
  match n with
  | 0 => 2
  | n + 1 => (n + 2 : ℚ) / (n + 1 : ℚ) * my_sequence n

theorem my_sequence_2016 : my_sequence 2015 = 4032 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_my_sequence_2016_l390_39001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_gt_sin_cos_l390_39029

theorem cos_sin_gt_sin_cos (x : Real) (h : x ∈ Set.Icc 0 Real.pi) : 
  Real.cos (Real.sin x) > Real.sin (Real.cos x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_gt_sin_cos_l390_39029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l390_39048

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in parametric form -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- Represents a curve in polar form -/
structure PolarCurve where
  ρ : ℝ → ℝ

/-- The given line l in parametric form -/
noncomputable def line_l : ParametricLine :=
  { x := λ t => -1 + (Real.sqrt 3 / 2) * t
  , y := λ t => (1 / 2) * t }

/-- The given curve C in polar form -/
noncomputable def curve_C : PolarCurve :=
  { ρ := λ θ => 4 * Real.cos θ }

/-- Converts polar coordinates to Cartesian coordinates -/
noncomputable def polar_to_cartesian (ρ θ : ℝ) : Point :=
  { x := ρ * Real.cos θ
  , y := ρ * Real.sin θ }

/-- Checks if a point is on the line l -/
def is_on_line_l (p : Point) : Prop :=
  p.x - Real.sqrt 3 * p.y + 1 = 0

/-- Checks if a point is on the curve C -/
def is_on_curve_C (p : Point) : Prop :=
  (p.x - 2)^2 + p.y^2 = 4

/-- The theorem to be proved -/
theorem intersection_distance :
  ∃ (P Q : Point),
    is_on_line_l P ∧ is_on_line_l Q ∧
    is_on_curve_C P ∧ is_on_curve_C Q ∧
    (P.x - Q.x)^2 + (P.y - Q.y)^2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l390_39048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_line_tangent_to_circle_prop_three_false_prop_four_false_l390_39086

-- Define the circle
def circleM (x y θ : ℝ) : Prop :=
  (x + Real.cos θ)^2 + (y - Real.sin θ)^2 = 1

-- Define the line
def lineL (x y k : ℝ) : Prop :=
  y = k * x

-- Theorem 1: The line and circle always have common points
theorem line_circle_intersection (k θ : ℝ) :
  ∃ (x y : ℝ), lineL x y k ∧ circleM x y θ := by sorry

-- Theorem 2: For any k, there exists a θ such that the line is tangent to the circle
theorem line_tangent_to_circle (k : ℝ) :
  ∃ (θ : ℝ), ∃! (x y : ℝ), lineL x y k ∧ circleM x y θ := by sorry

-- Theorem 3: Proposition 3 is false (counterexample)
theorem prop_three_false :
  ¬ (∀ θ : ℝ, ∃ k : ℝ, ∃! (x y : ℝ), lineL x y k ∧ circleM x y θ) := by sorry

-- Theorem 4: Proposition 4 is false
theorem prop_four_false :
  ¬ (∃ k θ : ℝ, ∃ (x y : ℝ), circleM x y θ ∧ 
    (abs (k * x - y) / Real.sqrt (1 + k^2) = 3)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_line_tangent_to_circle_prop_three_false_prop_four_false_l390_39086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_central_distance_range_l390_39036

noncomputable section

/-- The function g(x) defined as x + a/x --/
def g (a : ℝ) (x : ℝ) : ℝ := x + a / x

/-- The squared distance from a point (x, g(x)) to the origin --/
def distanceSquared (a : ℝ) (x : ℝ) : ℝ := x^2 + (g a x)^2

/-- The central distance of g is not less than √2 --/
def centralDistanceCondition (a : ℝ) : Prop :=
  ∀ x : ℝ, x > 0 → distanceSquared a x ≥ 2

/-- Theorem: If a > 0 and the central distance condition holds, then a ≥ √2 - 1 --/
theorem central_distance_range (a : ℝ) (h1 : a > 0) (h2 : centralDistanceCondition a) :
  a ≥ Real.sqrt 2 - 1 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_central_distance_range_l390_39036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_cube_angle_theorem_cone_cube_angle_correct_l390_39039

/-- The angle between the generatrix and the height of a cone with an inscribed cube -/
noncomputable def cone_cube_angle (k : ℝ) : ℝ :=
  Real.arctan (1 / (Real.sqrt 2 * (k - 1)))

/-- Theorem: The angle between the generatrix and the height of a cone with an inscribed cube -/
theorem cone_cube_angle_theorem (k : ℝ) (h : k > 1) :
  cone_cube_angle k = Real.arctan (1 / (Real.sqrt 2 * (k - 1))) := by
  -- Unfold the definition of cone_cube_angle
  unfold cone_cube_angle
  -- The equality holds by definition
  rfl

/-- Lemma: The cotangent of the angle is equal to √2(k-1) -/
lemma cone_cube_angle_cotangent (k : ℝ) (h : k > 1) :
  Real.tan (cone_cube_angle k) = 1 / (Real.sqrt 2 * (k - 1)) := by
  sorry

/-- Theorem: The angle found is indeed the correct angle between the generatrix and the height -/
theorem cone_cube_angle_correct (k : ℝ) (h : k > 1) :
  let a := 1 -- We can assume the cube edge length is 1 without loss of generality
  let cone_height := k * a
  let cube_center_to_vertex := a * Real.sqrt 2 / 2
  let generatrix_projection := a * (k - 1)
  Real.tan (cone_cube_angle k) = generatrix_projection / cube_center_to_vertex := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_cube_angle_theorem_cone_cube_angle_correct_l390_39039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_2011_l390_39003

def perfect_square (n : ℕ) : ℕ := n * n

def sign (n : ℕ) : Int :=
  if n % 4 = 1 ∨ n % 4 = 0 then 1 else -1

def sequence_term (n : ℕ) : Int :=
  (sign n) * (perfect_square n)

def sequence_sum (n : ℕ) : Int :=
  (List.range n).map (fun i => sequence_term (i + 1)) |>.sum

theorem sequence_sum_2011 :
  sequence_sum 2011 = -4046132 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_2011_l390_39003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_formula_sum_first_20_a_l390_39062

def a : ℕ → ℕ
  | 0 => 1  -- Added this case to handle Nat.zero
  | 1 => 1
  | n + 1 => if n % 2 = 0 then a n + 2 else a n + 1

def b (n : ℕ) : ℕ := a (2 * n)

theorem b_formula (n : ℕ) (h : n ≥ 1) : b n = 3 * n - 1 := by sorry

theorem sum_first_20_a : (Finset.range 20).sum a = 300 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_formula_sum_first_20_a_l390_39062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_cube_root_condition_l390_39073

theorem smallest_n_with_cube_root_condition : 
  (∃ (n : ℕ), n > 0 ∧ 
    (∃ (m : ℤ), ∃ (r : ℝ), 
      (∀ (k : ℤ), k < m → ¬∃ (s : ℝ), s > 0 ∧ s < 1/500 ∧ ∃ (j : ℕ), j > 0 ∧ (k : ℝ)^(1/3) = j + s) ∧
      r > 0 ∧ 
      r < 1/500 ∧ 
      (m : ℝ)^(1/3) = n + r)) ∧
  (∀ (k : ℕ), k > 0 → 
    (∃ (m : ℤ), ∃ (r : ℝ), 
      (∀ (l : ℤ), l < m → ¬∃ (s : ℝ), s > 0 ∧ s < 1/500 ∧ ∃ (j : ℕ), j > 0 ∧ (l : ℝ)^(1/3) = j + s) ∧
      r > 0 ∧ 
      r < 1/500 ∧ 
      (m : ℝ)^(1/3) = k + r) → 
    k ≥ 13) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_cube_root_condition_l390_39073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unbiased_variance_example_l390_39026

/-- Given a sample size and biased estimate of population variance, 
    calculate the unbiased estimate of population variance -/
noncomputable def unbiased_variance_estimate (n : ℕ) (D_b : ℝ) : ℝ :=
  (n : ℝ) / ((n : ℝ) - 1) * D_b

/-- Theorem stating that for n = 41 and D_b = 3, 
    the unbiased variance estimate is approximately 3.075 -/
theorem unbiased_variance_example :
  ‖unbiased_variance_estimate 41 3 - 3.075‖ < 0.0001 := by
  sorry

/-- Compute an approximation of the unbiased variance estimate -/
def approx_unbiased_variance_estimate (n : ℕ) (D_b : ℚ) : ℚ :=
  (n : ℚ) / ((n : ℚ) - 1) * D_b

#eval approx_unbiased_variance_estimate 41 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unbiased_variance_example_l390_39026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_inventory_theorem_l390_39082

/-- Calculates the number of good jars for sale given the initial conditions and damages --/
def calculate_good_jars (initial_cartons : ℕ) (jars_per_carton : ℕ) (cartons_reduction : ℕ) 
  (damaged_cartons : ℕ) (jars_damaged_per_carton : ℕ) (totally_damaged_cartons : ℕ) 
  (spoilage_rate : ℚ) : ℕ :=
  let received_cartons := initial_cartons - cartons_reduction
  let total_jars := received_cartons * jars_per_carton
  let damaged_jars := damaged_cartons * jars_damaged_per_carton
  let totally_damaged_jars := totally_damaged_cartons * jars_per_carton
  let good_jars_before_spoilage := total_jars - damaged_jars - totally_damaged_jars
  let spoiled_jars := (spoilage_rate * good_jars_before_spoilage).floor.toNat
  good_jars_before_spoilage - spoiled_jars

theorem milk_inventory_theorem :
  let whole_milk_jars := calculate_good_jars 30 20 15 7 3 1 (5/100)
  let skim_milk_jars := calculate_good_jars 20 25 5 5 4 1 (2/100)
  let almond_milk_jars := calculate_good_jars 15 15 0 3 2 0 (1/100)
  (whole_milk_jars = 247) ∧ (skim_milk_jars = 324) ∧ (almond_milk_jars = 217) := by
  sorry

#eval calculate_good_jars 30 20 15 7 3 1 (5/100)  -- Expected: 247
#eval calculate_good_jars 20 25 5 5 4 1 (2/100)   -- Expected: 324
#eval calculate_good_jars 15 15 0 3 2 0 (1/100)   -- Expected: 217

end NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_inventory_theorem_l390_39082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_criterion_l390_39071

def P (n : ℕ) : ℤ := (n^2 - 4) * (n^2 - 1) * (n^2 + 3)

def is_valid_form (n : ℕ) : Prop :=
  (∃ t : ℤ, n = 16 * t.natAbs + 2 ∨ n = 16 * t.natAbs - 2) ∨
  (∃ u : ℤ, n = 8 * u.natAbs + 1 ∨ n = 8 * u.natAbs - 1)

theorem divisibility_criterion (n : ℕ) :
  (2880 : ℤ) ∣ P n ↔ is_valid_form n ∧ ¬(3 : ℕ) ∣ n ∧ ¬(5 : ℕ) ∣ n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_criterion_l390_39071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_crossing_time_l390_39016

/-- The time taken for a train to cross a bridge -/
noncomputable def train_crossing_time (train_length : ℝ) (train_speed_kmph : ℝ) (bridge_length : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  total_distance / train_speed_mps

theorem train_bridge_crossing_time :
  train_crossing_time 165 36 660 = 82.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_crossing_time_l390_39016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_range_l390_39093

-- Define the sine curve
noncomputable def sine_curve (x : ℝ) : ℝ := Real.sin x

-- Define the tangent line's slope at a point on the sine curve
noncomputable def tangent_slope (x : ℝ) : ℝ := Real.cos x

-- Define the inclination angle of the tangent line
noncomputable def inclination_angle (x : ℝ) : ℝ := Real.arctan (tangent_slope x)

-- Theorem statement
theorem inclination_angle_range :
  ∀ x : ℝ, inclination_angle x ∈ Set.union (Set.Icc 0 (Real.pi / 4)) (Set.Ico (3 * Real.pi / 4) Real.pi) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_range_l390_39093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radius_from_sector_area_l390_39084

/-- The radius of a circle given the area of a sector and its central angle -/
theorem radius_from_sector_area (area : ℝ) (angle : ℝ) (r : ℝ) : 
  area = (angle / 360) * Real.pi * r^2 →
  angle = 39 →
  area = 49.02857142857143 →
  abs (r - 12.03) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_radius_from_sector_area_l390_39084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ABC_l390_39000

-- Define points
def Point : Type := ℝ × ℝ

-- Define the circle with radius 2 centered at (2, 0)
def isOnCircle (p : Point) : Prop := (p.1 - 2)^2 + p.2^2 = 4

-- Define the points
noncomputable def A : Point := (0, 0)
noncomputable def B : Point := (4, 0)
noncomputable def D : Point := (7, 0)
noncomputable def E : Point := (7, 5)

-- Define C as the intersection of AE and the circle
noncomputable def C : Point := (98/37, 70/37)

-- State the theorem
theorem area_of_triangle_ABC :
  let triangle_area := (1/2) * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))
  (∀ p, isOnCircle p → (p.1 - 2)^2 + p.2^2 = 4) →  -- Circle definition
  (B.1 - A.1 = 4) →  -- AB is diameter
  (D.1 - B.1 = 3) →  -- BD = 3
  ((E.1 - D.1) * (E.2 - D.2) = 0) →  -- ED perpendicular to AD
  ((E.1 - D.1)^2 + (E.2 - D.2)^2 = 25) →  -- ED = 5
  (∃ t : ℝ, 0 < t ∧ t < 1 ∧ C = (t * A.1 + (1 - t) * E.1, t * A.2 + (1 - t) * E.2)) →  -- C between A and E
  isOnCircle C →  -- C is on the circle
  triangle_area = 140/37 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ABC_l390_39000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_condition_l390_39042

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- The focal distance of a hyperbola -/
noncomputable def focal_distance (h : Hyperbola) : ℝ :=
  Real.sqrt (h.a^2 + h.b^2)

/-- Theorem: If the right vertex of a hyperbola is inside the circle with diameter AB,
    where A and B are the intersection points of the hyperbola with a perpendicular line
    to the x-axis through the left focus, then the eccentricity of the hyperbola is greater than 2 -/
theorem hyperbola_eccentricity_condition (h : Hyperbola) :
  h.a + focal_distance h < h.b^2 / h.a → eccentricity h > 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_condition_l390_39042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l390_39067

/-- The origin point (0, 0) -/
def O : ℝ × ℝ := (0, 0)

/-- Angle AOB in degrees -/
noncomputable def angle_AOB (A B C : ℝ × ℝ) : ℝ := sorry

/-- A line y = kx + 1 intersects a circle x^2 + y^2 = 1 at points A and B. If ∠AOB = 60°, then k = ± √3/3. -/
theorem line_circle_intersection (k : ℝ) : 
  (∃ (A B : ℝ × ℝ), 
    (A.1^2 + A.2^2 = 1) ∧ 
    (B.1^2 + B.2^2 = 1) ∧ 
    (A.2 = k * A.1 + 1) ∧ 
    (B.2 = k * B.1 + 1) ∧ 
    (angle_AOB A O B = 60)) → 
  (k = Real.sqrt 3 / 3 ∨ k = -(Real.sqrt 3 / 3)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l390_39067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sum_upper_bound_l390_39055

theorem log_sum_upper_bound (a b : ℝ) (h1 : a^2 ≥ b^2) (h2 : b^2 > 1) :
  Real.log (a^2 / b^2) / Real.log a^2 + Real.log (b^2 / a^2) / Real.log b^2 ≤ 0 ∧
  (Real.log (a^2 / b^2) / Real.log a^2 + Real.log (b^2 / a^2) / Real.log b^2 = 0 ↔ a = b) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sum_upper_bound_l390_39055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_in_set_l390_39089

theorem smallest_integer_in_set :
  let A : Set ℝ := {x | |x - 2| ≤ 5}
  ∃ n : ℤ, (n : ℝ) ∈ A ∧ ∀ m : ℤ, (m : ℝ) ∈ A → n ≤ m ∧ n = -3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_in_set_l390_39089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_implies_a_range_l390_39027

-- Define the function as noncomputable due to the use of Real.sqrt
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sqrt ((2 - a) * x^2 + 2 * (2 - a) * x + 4)

-- State the theorem
theorem domain_implies_a_range (a : ℝ) :
  (∀ x : ℝ, ∃ y : ℝ, f a x = y) ↔ -2 ≤ a ∧ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_implies_a_range_l390_39027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_solutions_l390_39097

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 3 * Real.cos (Real.pi * x)

-- Define the domain
def domain (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 3

-- Define the equation to be satisfied
def satisfies_equation (x : ℝ) : Prop := f (f (f x)) = f x

-- Theorem statement
theorem number_of_solutions :
  ∃ (S : Finset ℝ), (∀ x ∈ S, domain x ∧ satisfies_equation x) ∧ S.card = 61 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_solutions_l390_39097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_2010_coordinates_l390_39032

noncomputable def rotate_point (x y : ℝ) (angle : ℝ) : ℝ × ℝ :=
  (x * Real.cos angle - y * Real.sin angle, x * Real.sin angle + y * Real.cos angle)

def extend_point (x y : ℝ) : ℝ × ℝ :=
  (2 * x, 2 * y)

noncomputable def next_point (p : ℝ × ℝ) (n : ℕ) : ℝ × ℝ :=
  if n % 2 = 1 then
    rotate_point p.1 p.2 (30 * Real.pi / 180)
  else
    extend_point p.1 p.2

noncomputable def nth_point : ℕ → ℝ × ℝ
  | 0 => (1, 0)
  | n + 1 => next_point (nth_point n) (n + 1)

theorem point_2010_coordinates :
  nth_point 2010 = (0, -2^1004) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_2010_coordinates_l390_39032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_equals_three_fourths_l390_39037

/-- The sum of the infinite series 1/(3^1) + 2/(3^2) + 3/(3^3) + ... + k/(3^k) + ... -/
noncomputable def infinite_series_sum : ℝ := ∑' k, (k : ℝ) / (3 : ℝ) ^ k

/-- Theorem: The sum of the infinite series is equal to 3/4 -/
theorem infinite_series_sum_equals_three_fourths : infinite_series_sum = 3/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_equals_three_fourths_l390_39037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_sum_l390_39041

theorem sin_cos_sum (θ : ℝ) 
  (h1 : Real.sin θ * Real.cos θ = 2/5)
  (h2 : Real.sqrt ((Real.cos θ) ^ 2) = -Real.cos θ) : 
  Real.sin θ + Real.cos θ = -(3 * Real.sqrt 5) / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_sum_l390_39041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_positive_solution_l390_39030

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.cos (Real.arcsin (Real.tan (Real.arccos x)))

-- State the theorem
theorem unique_positive_solution :
  ∃! (x : ℝ), x > 0 ∧ x ∈ Set.Icc (-1 : ℝ) 1 ∧ f x = x ∧ x = Real.sqrt ((3 - Real.sqrt 5) / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_positive_solution_l390_39030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_function_inequality_l390_39047

/-- A function f: ℝ → ℝ is increasing -/
def IncreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

theorem increasing_function_inequality (f : ℝ → ℝ) (h : IncreasingFunction f) :
  {x : ℝ | f x < f (2*x - 3)} = Set.Ioi 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_function_inequality_l390_39047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_f_at_zero_l390_39074

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (Real.sin (4 * x) / x) ^ (2 / (x + 2))

-- State the theorem
theorem limit_of_f_at_zero :
  Filter.Tendsto f (nhds 0) (nhds 4) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_f_at_zero_l390_39074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eighth_triangle_shading_l390_39068

def triangle_sequence (n : ℕ) : ℕ := n^2

def pascal_triangle : ℕ → ℕ → ℕ
  | 0, _ => 1
  | n+1, 0 => 1
  | n+1, k+1 => pascal_triangle n k + pascal_triangle n (k+1)

def is_shaded (layer : ℕ) : Bool :=
  (layer - 1) / 2 % 2 = 1

def shaded_count (n : ℕ) : ℕ :=
  (List.range n).foldl (fun acc i => 
    acc + if is_shaded (i + 1) then pascal_triangle i 0 else 0) 0

theorem eighth_triangle_shading :
  (shaded_count 8 : ℚ) / (triangle_sequence 8 : ℚ) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eighth_triangle_shading_l390_39068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l390_39034

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (7 + 6*x - x^2)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | -1 ≤ x ∧ x ≤ 7} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l390_39034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jill_food_spending_l390_39076

/-- Represents the percentage spent on each category and the tax rates --/
structure ShoppingBreakdown where
  clothing_percent : ℚ
  other_percent : ℚ
  clothing_tax_rate : ℚ
  other_tax_rate : ℚ
  total_tax_rate : ℚ

/-- Calculates the percentage spent on food given a shopping breakdown --/
def food_percent (s : ShoppingBreakdown) : ℚ :=
  100 - s.clothing_percent - s.other_percent

/-- Calculates the total tax rate based on the spending breakdown --/
def calculated_tax_rate (s : ShoppingBreakdown) : ℚ :=
  (s.clothing_percent * s.clothing_tax_rate + s.other_percent * s.other_tax_rate) / 100

/-- Theorem stating that given the shopping conditions, the percentage spent on food is 25% --/
theorem jill_food_spending (s : ShoppingBreakdown) 
  (h1 : s.clothing_percent = 50)
  (h2 : s.other_percent = 25)
  (h3 : s.clothing_tax_rate = 10)
  (h4 : s.other_tax_rate = 20)
  (h5 : s.total_tax_rate = 10)
  (h6 : calculated_tax_rate s = s.total_tax_rate) :
  food_percent s = 25 := by
  sorry

#eval food_percent { clothing_percent := 50, other_percent := 25, clothing_tax_rate := 10, other_tax_rate := 20, total_tax_rate := 10 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jill_food_spending_l390_39076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_congruence_in_range_l390_39078

theorem unique_congruence_in_range (a b : ℤ) 
  (ha : a ≡ 25 [ZMOD 42])
  (hb : b ≡ 63 [ZMOD 42]) :
  ∃! n : ℤ, n ∈ Set.Icc 200 241 ∧ a - b ≡ n [ZMOD 42] ∧ n = 214 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_congruence_in_range_l390_39078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_square_problem_l390_39095

/-- A function representing the inverse square relationship between x and y -/
noncomputable def inverse_square (k : ℝ) (y : ℝ) : ℝ := k / (y ^ 2)

/-- Theorem stating that given the inverse square relationship and initial condition,
    x equals 2.25 when y equals 2 -/
theorem inverse_square_problem (k : ℝ) :
  (inverse_square k 3 = 1) →
  (inverse_square k 2 = 2.25) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_square_problem_l390_39095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equation_solution_l390_39002

theorem vector_equation_solution : ∃ (u v : ℝ), 
  u = 5 / 17 ∧ 
  v = -31 / 17 ∧ 
  (![3, -2] : Fin 2 → ℝ) + u • ![5, 8] = (![(-1), 4] : Fin 2 → ℝ) + v • ![(-3), 2] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equation_solution_l390_39002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_population_is_1950_l390_39077

-- Define the initial population
def initial_population : ℕ := 1200

-- Define the growth rates
def first_year_growth_rate : ℚ := 25 / 100
def second_year_growth_rate : ℚ := 30 / 100

-- Define the function to calculate population after growth
def population_after_growth (initial : ℕ) (rate : ℚ) : ℕ :=
  (↑initial * (1 + rate)).floor.toNat

-- Theorem statement
theorem final_population_is_1950 :
  let first_year := population_after_growth initial_population first_year_growth_rate
  let second_year := population_after_growth first_year second_year_growth_rate
  second_year = 1950 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_population_is_1950_l390_39077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_sqrt_difference_l390_39090

theorem smallest_integer_sqrt_difference (n : ℕ) : 
  (n ≥ 250001) ↔ (Real.sqrt (n : ℝ) - Real.sqrt ((n - 1) : ℝ) < 0.001) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_sqrt_difference_l390_39090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_polar_points_l390_39049

noncomputable def distance_polar (r₁ r₂ : ℝ) (φ₁ φ₂ : ℝ) : ℝ :=
  Real.sqrt ((r₁ * Real.cos φ₁ - r₂ * Real.cos φ₂)^2 + (r₁ * Real.sin φ₁ - r₂ * Real.sin φ₂)^2)

theorem distance_between_polar_points (φ₁ φ₂ : ℝ) 
  (h : φ₁ - φ₂ = π / 3) : 
  distance_polar 5 12 φ₁ φ₂ = 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_polar_points_l390_39049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_l_properties_l_l390_39099

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := 2 * x - y - 2 = 0
def l₂ (x y : ℝ) : Prop := x + y + 3 = 0

-- Define the intersection point of l₁ and l₂
noncomputable def intersection : ℝ × ℝ := (-1/3, -8/3)

-- Define the line l
def l (x y : ℝ) : Prop := x - y - 7/3 = 0

-- Define point P
def P : ℝ × ℝ := (3, 0)

-- Define the line l'
def l' (x y : ℝ) : Prop := 8 * x - y - 24 = 0

-- Theorem 1: l passes through the intersection of l₁ and l₂ and is perpendicular to l₂
theorem l_properties : 
  l (intersection.1) (intersection.2) ∧ 
  (∀ x y : ℝ, l₂ x y → (x - intersection.1) * 1 + (y - intersection.2) * 1 = 0) := by
  sorry

-- Theorem 2: l' passes through P and bisects AB where A is on l₁ and B is on l₂
theorem l'_properties : 
  l' P.1 P.2 ∧ 
  (∃ A B : ℝ × ℝ, 
    l₁ A.1 A.2 ∧ 
    l₂ B.1 B.2 ∧ 
    P.1 = (A.1 + B.1) / 2 ∧ 
    P.2 = (A.2 + B.2) / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_l_properties_l_l390_39099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_range_of_a_l390_39033

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*(a+1)*x + 2*a*(Real.log x)

-- Part 1: Tangent line when a = 1
theorem tangent_line_at_one :
  let a : ℝ := 1
  let f₁ (x : ℝ) : ℝ := f a x
  (∀ x : ℝ, x > 0 → HasDerivAt f₁ ((2*x^2 - 4*x + 2) / x) x) →
  f₁ 1 = -3 →
  (∀ x : ℝ, x > 0 → HasDerivAt f₁ 0 1) →
  ∀ x : ℝ, x - 1 = -3 :=
by
  sorry

-- Part 2: Range of a
theorem range_of_a (a : ℝ) :
  a > 0 →
  (∀ x : ℝ, x ∈ Set.Icc 1 (Real.exp 1) → f a x ≤ 0) →
  a ≥ (Real.exp 2 - 2*Real.exp 1) / (2*Real.exp 1 - 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_range_of_a_l390_39033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_wire_square_resistance_l390_39045

/-- Represents a wire structure in the shape of a square with midpoints marked. -/
structure WireSquare where
  /-- The resistance of a wire segment that is half the length of a side of the square. -/
  R₀ : ℝ
  /-- Assertion that the wire is homogeneous with constant cross-section. -/
  homogeneous : Prop
  /-- Assertion that points B, D, F, and H are at the midpoints of the square's sides. -/
  midpoints : Prop

/-- 
Theorem stating that the resistance between opposite corners of the wire square
is equal to twice the resistance of a half-side segment.
-/
theorem wire_square_resistance (w : WireSquare) (h : w.R₀ = 1) :
  ∃ (R : ℝ), R = 2 * w.R₀ ∧ 
  (R = 2 → R = 2 * w.R₀) := by
  sorry

#check wire_square_resistance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_wire_square_resistance_l390_39045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_approximation_l390_39051

/-- The length of each identical rectangle --/
noncomputable def x : ℝ := sorry

/-- The width of each identical rectangle --/
noncomputable def w : ℝ := sorry

/-- The area of the larger rectangle PQRS --/
def area_PQRS : ℝ := 4000

/-- The relationship between width and length --/
axiom width_length_relation : w = (2/3) * x

/-- The area of PQRS is equal to the sum of areas of five identical rectangles --/
axiom area_equality : 5 * (x * w) = area_PQRS

/-- The theorem stating that the length x is approximately 34.6 --/
theorem length_approximation : ∃ (ε : ℝ), ε > 0 ∧ |x - 34.6| < ε := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_approximation_l390_39051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_commute_time_difference_l390_39023

noncomputable def commute_times (x y : ℝ) : List ℝ := [x, y, 10, 11, 9]

noncomputable def average (l : List ℝ) : ℝ := (l.sum) / l.length

noncomputable def variance (l : List ℝ) : ℝ :=
  let μ := average l
  (l.map (λ x => (x - μ)^2)).sum / l.length

theorem commute_time_difference (x y : ℝ) 
  (h1 : average (commute_times x y) = 10)
  (h2 : variance (commute_times x y) = 2) :
  |x - y| = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_commute_time_difference_l390_39023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coffee_shop_revenue_l390_39080

/-- Calculate the total revenue with tax included for a coffee shop --/
theorem coffee_shop_revenue : 
  (5 : ℚ) * 7 * (1 + 0.08) +
  (6 : ℚ) * 3 * (1 + 0.08) +
  (4 : ℚ) * 4 * (1 + 0.06) +
  (5 : ℚ) * 4 * (1 + 0.06) = 95.40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coffee_shop_revenue_l390_39080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l390_39031

noncomputable def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

noncomputable def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l390_39031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_red_sunday_minimum_time_l390_39094

/-- Represents the duration of an activity in minutes -/
def Duration := ℕ

instance : OfNat Duration n where
  ofNat := n

/-- Represents a Sunday activity -/
structure Activity where
  name : String
  duration : Duration

/-- Red's Sunday schedule -/
def sunday_schedule : List Activity := [
  ⟨"Computer", 30⟩,
  ⟨"Listening to music", 20⟩,
  ⟨"Cooking", 35⟩,
  ⟨"Tidying up", 15⟩,
  ⟨"Practicing piano", 25⟩
]

/-- The minimum time required for the most reasonable arrangement of activities -/
def minimum_time (schedule : List Activity) : Duration := sorry

/-- Theorem stating that the minimum time for Red's Sunday schedule is 90 minutes -/
theorem red_sunday_minimum_time :
  minimum_time sunday_schedule = 90 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_red_sunday_minimum_time_l390_39094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_and_sum_proof_l390_39091

noncomputable def arithmetic_sequence (a₁ : ℝ) (d : ℝ) : ℕ → ℝ :=
  fun n ↦ a₁ + (n - 1) * d

noncomputable def geometric_sequence (b₁ : ℝ) (q : ℝ) : ℕ → ℝ :=
  fun n ↦ b₁ * q^(n - 1)

noncomputable def sum_c_even (n : ℕ) : ℝ :=
  6 * (8^(n/2) - 1) / 7

noncomputable def sum_c_odd (n : ℕ) : ℝ :=
  20 * (8^((n-1)/2) - 1) / 7 + 2

theorem sequence_and_sum_proof
  (a : ℕ → ℝ)
  (b : ℕ → ℝ)
  (d : ℝ)
  (q : ℝ)
  (h_a : a = arithmetic_sequence 1 d)
  (h_b : b = geometric_sequence 1 q)
  (h_q_pos : q > 0)
  (h_eq1 : a 2 + b 3 = 8)
  (h_eq2 : a 3 + b 2 = 9) :
  (∀ n, a n = 3*n - 2) ∧
  (∀ n, b n = 2^(n-1)) ∧
  (∀ n, Even n → sum_c_even n = 6 * (8^(n/2) - 1) / 7) ∧
  (∀ n, Odd n → sum_c_odd n = 20 * (8^((n-1)/2) - 1) / 7 + 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_and_sum_proof_l390_39091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l390_39060

theorem trigonometric_identities (α : ℝ) (h : Real.sin (π + α) = -1/3) :
  (Real.cos (α - 3*π/2) = -1/3) ∧
  ((Real.sin (π/2 + α) = 2*Real.sqrt 2/3) ∨ (Real.sin (π/2 + α) = -2*Real.sqrt 2/3)) ∧
  ((Real.tan (5*π - α) = -Real.sqrt 2/4) ∨ (Real.tan (5*π - α) = Real.sqrt 2/4)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l390_39060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_BB_l390_39014

-- Define the coordinates of point B
def B : ℝ × ℝ := (-2, 0)

-- Define the reflection of a point over the y-axis
def reflect_over_y_axis (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

-- Define B' as the reflection of B over the y-axis
def B' : ℝ × ℝ := reflect_over_y_axis B

-- Define the distance between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem statement
theorem length_of_BB' : distance B B' = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_BB_l390_39014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exp_and_abs_exp_properties_l390_39013

-- Define the exponential function
noncomputable def exp (a : ℝ) (x : ℝ) : ℝ := a^x

-- Define the absolute value function
noncomputable def abs_exp (x : ℝ) : ℝ := 3^(|x|)

-- Statement of the theorem
theorem exp_and_abs_exp_properties :
  (∀ (a : ℝ), a > 0 → a ≠ 1 → (Set.range (exp a) = Set.univ)) ∧
  (Set.range abs_exp = Set.Ici 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exp_and_abs_exp_properties_l390_39013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_setB_is_pythagorean_triple_l390_39040

-- Define a function to check if three numbers form a Pythagorean triple
def isPythagoreanTriple (a b c : ℕ) : Prop := a^2 + b^2 = c^2

-- Define the sets of numbers given in the problem
def setA : List ℚ := [3/10, 4/10, 5/10]
def setB : List ℕ := [9, 40, 41]
def setC : List ℕ := [2, 3, 4]
noncomputable def setD : List ℝ := [1, Real.sqrt 2, Real.sqrt 3]

-- Theorem stating that only setB is a Pythagorean triple
theorem only_setB_is_pythagorean_triple :
  (∀ (a b c : ℚ), a ∈ setA → b ∈ setA → c ∈ setA → ¬isPythagoreanTriple (Int.natAbs a.num) (Int.natAbs b.num) (Int.natAbs c.num)) ∧
  (∃ (a b c : ℕ), a ∈ setB ∧ b ∈ setB ∧ c ∈ setB ∧ isPythagoreanTriple a b c) ∧
  (∀ (a b c : ℕ), a ∈ setC → b ∈ setC → c ∈ setC → ¬isPythagoreanTriple a b c) ∧
  (∀ (a b c : ℝ), a ∈ setD → b ∈ setD → c ∈ setD → ¬isPythagoreanTriple (Int.natAbs (Int.floor a)) (Int.natAbs (Int.floor b)) (Int.natAbs (Int.floor c))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_setB_is_pythagorean_triple_l390_39040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expGenFunc_is_exp_l390_39011

/-- The exponential generating function for the sequence r^n, where r is a positive real number -/
noncomputable def expGenFunc (r : ℝ) (t : ℝ) : ℝ := Real.exp (r * t)

/-- Theorem stating that the exponential generating function for the sequence r^n is e^(rt) -/
theorem expGenFunc_is_exp (r : ℝ) (hr : r > 0) : 
  expGenFunc r = λ t ↦ ∑' n, r^n * t^n / n.factorial := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expGenFunc_is_exp_l390_39011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_elementary_school_coats_approx_l390_39096

def total_coats : ℕ := 9437

def high_school_fraction : ℚ := 3/5
def middle_school_fraction : ℚ := 1/4

def coats_from_high_schools (total : ℕ) : ℕ := 
  Int.toNat ⌊(high_school_fraction * total : ℚ)⌋

def coats_from_middle_schools (total : ℕ) : ℕ := 
  Int.toNat ⌊(middle_school_fraction * total : ℚ)⌋

def coats_from_elementary_schools (total high middle : ℕ) : ℕ :=
  total - high - middle

theorem elementary_school_coats_approx :
  (coats_from_elementary_schools total_coats 
    (coats_from_high_schools total_coats) 
    (coats_from_middle_schools total_coats)) = 1416 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_elementary_school_coats_approx_l390_39096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycle_wheel_revolutions_l390_39059

/-- Calculates the number of revolutions for the back wheel of a bicycle to cover the same distance as the front wheel, given specific conditions. -/
theorem bicycle_wheel_revolutions 
  (front_perimeter : ℝ) 
  (back_perimeter : ℝ) 
  (front_slip_ratio : ℝ) 
  (back_slip_ratio : ℝ) 
  (front_revolutions : ℕ) 
  (h1 : front_perimeter = 30) 
  (h2 : back_perimeter = 20) 
  (h3 : front_slip_ratio = 0.05) 
  (h4 : back_slip_ratio = 0.07) 
  (h5 : front_revolutions = 240) :
  ∃ (back_revolutions : ℕ), 
    (abs ((back_revolutions : ℝ) - (front_perimeter * (1 - front_slip_ratio) * front_revolutions) / (back_perimeter * (1 - back_slip_ratio))) < 1) ∧ 
    back_revolutions = 368 :=
by
  sorry

#check bicycle_wheel_revolutions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycle_wheel_revolutions_l390_39059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_trapezoid_from_identical_right_triangles_l390_39006

/-- A right-angled triangle --/
structure RightTriangle where
  base : ℝ
  height : ℝ
  hypotenuse : ℝ
  right_angle : hypotenuse^2 = base^2 + height^2

/-- A quadrilateral formed by two right-angled triangles --/
inductive Quadrilateral
  | fromTriangles (t1 t2 : RightTriangle) : Quadrilateral

/-- A trapezoid is a quadrilateral with exactly one pair of parallel sides --/
def isTrapezoid (q : Quadrilateral) : Prop :=
  ∃ (s1 s2 p1 p2 : ℝ), s1 ≠ s2 ∧ p1 = p2 ∧
    (∀ (o1 o2 : ℝ), o1 ≠ p1 → o2 ≠ p2 → o1 ≠ o2)

theorem no_trapezoid_from_identical_right_triangles :
  ∀ (t : RightTriangle), ¬∃ (q : Quadrilateral), 
    (q = Quadrilateral.fromTriangles t t) ∧ isTrapezoid q := by
  sorry

#check no_trapezoid_from_identical_right_triangles

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_trapezoid_from_identical_right_triangles_l390_39006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_simplification_l390_39069

theorem trig_simplification (y z : ℝ) : 
  Real.sin y ^ 2 + Real.sin (y + z) ^ 2 - 2 * Real.sin y * Real.sin z * Real.sin (y + z) = 
  Real.sin y ^ 2 + Real.sin z ^ 2 * Real.cos y ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_simplification_l390_39069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_no_axis_intersection_l390_39054

-- Define the power function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m^2 + 2*m - 7) * x^m

-- State the theorem
theorem power_function_no_axis_intersection :
  ∃! m : ℝ, ∀ x : ℝ, x ≠ 0 → f m x ≠ 0 ∧ m = -4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_no_axis_intersection_l390_39054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_center_of_mass_l390_39008

/-- An isosceles right triangle with hypotenuse 2a -/
structure IsoscelesRightTriangle (a : ℝ) where
  vertex_A : ℝ × ℝ := (-a, 0)
  vertex_B : ℝ × ℝ := (a, 0)
  vertex_C : ℝ × ℝ := (0, a)

/-- Surface density function proportional to distance from hypotenuse -/
def surfaceDensity (a k : ℝ) (p : ℝ × ℝ) : ℝ :=
  k * p.2

/-- Center of mass of the triangle -/
noncomputable def centerOfMass (a : ℝ) : ℝ × ℝ := (0, a / 2)

/-- Theorem stating that the center of mass is at (0, a/2) -/
theorem isosceles_right_triangle_center_of_mass (a k : ℝ) (h : a > 0) :
  let triangle := IsoscelesRightTriangle a
  let ρ := surfaceDensity a k
  centerOfMass a = (0, a / 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_center_of_mass_l390_39008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_distribution_probability_l390_39007

-- Define the normal distribution density function
noncomputable def normal_density (x : ℝ) : ℝ :=
  (1 / Real.sqrt (2 * Real.pi)) * Real.exp (-((x - 2)^2) / 2)

-- State the theorem
theorem normal_distribution_probability 
  (h1 : ∫ x in Set.Icc 0 2, normal_density x = 1/3) : 
  ∫ x in Set.Ioi 4, normal_density x = 1/6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_distribution_probability_l390_39007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_expressions_l390_39057

theorem order_of_expressions : 
  (3 : ℝ)^(3/10) > Real.log 3 / Real.log π ∧ Real.log 3 / Real.log π > Real.log ℯ / Real.log (3/10) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_expressions_l390_39057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_in_circle_probability_l390_39087

noncomputable section

-- Define the rectangle
def rectangle_length : ℝ := 4
def rectangle_width : ℝ := 3

-- Define the circle (radius is half the diagonal of the rectangle)
def circle_radius : ℝ := Real.sqrt (rectangle_length ^ 2 + rectangle_width ^ 2) / 2

-- Define the probability
def probability : ℝ := (rectangle_length * rectangle_width) / (Real.pi * circle_radius ^ 2)

-- Theorem statement
theorem rectangle_in_circle_probability :
  probability = 48 / (25 * Real.pi) := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_in_circle_probability_l390_39087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proposition_is_correct_l390_39064

/-- Definition of a triangle -/
structure Triangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ

/-- Definition of an isosceles triangle -/
def is_isosceles (t : Triangle) : Prop :=
  t.side1 = t.side2 ∨ t.side2 = t.side3 ∨ t.side3 = t.side1

/-- Definition of base angles being equal -/
def base_angles_equal (t : Triangle) : Prop :=
  t.angle1 = t.angle2 ∨ t.angle2 = t.angle3 ∨ t.angle3 = t.angle1

/-- Original proposition -/
def original_proposition (t : Triangle) : Prop :=
  is_isosceles t → base_angles_equal t

/-- Inverse proposition -/
def inverse_proposition (t : Triangle) : Prop :=
  base_angles_equal t → is_isosceles t

/-- Theorem stating that the inverse proposition is correct -/
theorem inverse_proposition_is_correct :
  ∀ t : Triangle, inverse_proposition t = (base_angles_equal t → is_isosceles t) :=
by
  intro t
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proposition_is_correct_l390_39064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_m_value_and_k_range_l390_39092

noncomputable def f (x : ℝ) (m : ℝ) := 0.5 * (Real.exp (x * Real.log 2) + m - Real.exp (-x * Real.log 2))

theorem odd_function_m_value_and_k_range :
  (∀ x, f x m = -f (-x) m) →
  (m = 1 ∧ ∀ k, (∃ x < 0, f x m = k) → -1 < k ∧ k < 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_m_value_and_k_range_l390_39092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pine_cones_problem_l390_39021

def pine_cones_per_tree (num_trees : ℕ) (roof_percentage : ℚ) 
  (pine_cone_weight : ℚ) (roof_weight : ℚ) : ℕ :=
  let total_weight : ℚ := roof_weight / roof_percentage
  let total_pine_cones : ℚ := total_weight / pine_cone_weight
  (total_pine_cones / num_trees).floor.toNat

theorem pine_cones_problem :
  pine_cones_per_tree 8 (30/100) 4 1920 = 200 := by
  sorry

#eval pine_cones_per_tree 8 (30/100) 4 1920

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pine_cones_problem_l390_39021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_translation_for_odd_symmetry_l390_39061

noncomputable def f (x : ℝ) := Real.cos (x + 4 * Real.pi / 3)

noncomputable def translated_f (x φ : ℝ) := Real.cos (x + 4 * Real.pi / 3 + φ)

def is_odd (g : ℝ → ℝ) := ∀ x, g (-x) = -g x

theorem min_translation_for_odd_symmetry :
  ∃ φ : ℝ, (is_odd (translated_f · φ)) ∧
    (∀ ψ : ℝ, is_odd (translated_f · ψ) → abs φ ≤ abs ψ) ∧
    abs φ = Real.pi / 6 := by
  sorry

#check min_translation_for_odd_symmetry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_translation_for_odd_symmetry_l390_39061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circles_area_sum_l390_39079

/-- CircleTangent r₁ r₂ means two circles with radii r₁ and r₂ are externally tangent -/
def CircleTangent (r₁ r₂ : ℝ) : Prop :=
  ∃ (d : ℝ), d = r₁ + r₂ ∧ d > 0

/-- Given a sequence of circles Γᵢ with radii rᵢ inscribed in an angle 2α,
    where each Γᵢ is externally tangent to Γᵢ₊₁ and rᵢ₊₁ < rᵢ,
    prove that the sum of their areas equals the area of a circle with radius
    (1/2) * r₀ * (√(sin α) + √(csc α)) -/
theorem inscribed_circles_area_sum (α : ℝ) (r : ℕ → ℝ) 
  (h_angle : 0 < α ∧ α < π/2)
  (h_tangent : ∀ i : ℕ, CircleTangent (r i) (r (i+1)))
  (h_decreasing : ∀ i : ℕ, r (i+1) < r i) :
  (∑' i, π * (r i)^2) = π * ((1/2 : ℝ) * r 0 * (Real.sqrt (Real.sin α) + Real.sqrt (1 / Real.sin α)))^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circles_area_sum_l390_39079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l390_39004

-- Define the expression as noncomputable due to Real.sqrt
noncomputable def expression : ℝ := 2^2 + ((-1 : ℝ)^(1/3)) - |(-Real.sqrt 2)| + Real.sqrt 9

-- State the theorem
theorem expression_evaluation : expression = 6 - Real.sqrt 2 := by
  -- Proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l390_39004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_contest_order_l390_39066

/-- Represents a contestant in the beauty contest. -/
inductive Contestant : Type
  | A | B | C | D | E

/-- Represents the order of contestants in the beauty contest. -/
def ContestOrder := List Contestant

/-- Checks if two contestants are consecutive in a given order. -/
def are_consecutive (c1 c2 : Contestant) (order : ContestOrder) : Prop :=
  ∃ i, order.get? i = some c1 ∧ order.get? (i + 1) = some c2

/-- Checks if a given order satisfies the conditions of the problem. -/
def satisfies_conditions (order : ContestOrder) : Prop :=
  (order ≠ [Contestant.A, Contestant.B, Contestant.C, Contestant.D, Contestant.E]) ∧
  (∃ (i j : Nat) (c1 c2 : Contestant),
    i ≠ j ∧
    order.get? i = (([Contestant.D, Contestant.A, Contestant.E, Contestant.C, Contestant.B] : ContestOrder).get? i) ∧
    order.get? j = (([Contestant.D, Contestant.A, Contestant.E, Contestant.C, Contestant.B] : ContestOrder).get? j)) ∧
  (∃ (c1 c2 : Contestant),
    are_consecutive c1 c2 order ∧
    are_consecutive c1 c2 [Contestant.D, Contestant.A, Contestant.E, Contestant.C, Contestant.B])

/-- The main theorem stating that the only order satisfying the conditions is E-D-A-C-B. -/
theorem unique_contest_order :
  ∀ (order : ContestOrder),
    satisfies_conditions order →
    order = [Contestant.E, Contestant.D, Contestant.A, Contestant.C, Contestant.B] := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_contest_order_l390_39066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_50_digits_l390_39065

/-- The decimal representation of 1/1234 -/
def decimal_rep : ℚ := 1 / 1234

/-- The sequence of digits in the decimal representation of 1/1234 -/
def digit_sequence : ℕ → ℕ :=
  sorry

/-- The sum of the first n digits after the decimal point -/
def sum_of_digits (n : ℕ) : ℕ :=
  (Finset.range n).sum (λ i => digit_sequence i)

/-- The theorem stating that the sum of the first 50 digits after the decimal point in 1/1234 is 207 -/
theorem sum_of_first_50_digits :
  sum_of_digits 50 = 207 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_50_digits_l390_39065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l390_39025

/-- A quadratic function satisfying certain conditions -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 4

/-- The minimum value function for f(x) when x ∈ [t, t+1] and t > 0 -/
noncomputable def g (t : ℝ) : ℝ :=
  if t > 0 ∧ t < 1 then 3 else t^2 - 2*t + 4

/-- Theorem stating the properties of f and g -/
theorem quadratic_function_properties :
  (∀ x, f x = f (2 - x)) ∧
  (∃ r, ∀ x, f x - 3/4 * 4 = 0 → x = r) ∧
  (∀ t, t > 0 → ∀ x ∈ Set.Icc t (t + 1), f x ≥ g t) := by
  sorry

#check quadratic_function_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l390_39025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_exponent_sum_l390_39052

theorem smallest_exponent_sum (p q : ℕ) : 
  (∃ (k : ℕ), (7^(p+4) * 5^q * 2^3 = k^3)) → 
  (∀ (r s : ℕ), (∃ (m : ℕ), (7^(r+4) * 5^s * 2^3 = m^3)) → p + q ≤ r + s) → 
  p + q = 5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_exponent_sum_l390_39052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_PAMB_l390_39083

-- Define the circle M
def circle_M : Set (ℝ × ℝ) := {p | (p.1 - 1)^2 + (p.2 - 1)^2 = 4}

-- Define points A and B
def point_A : ℝ × ℝ := (1, -1)
def point_B : ℝ × ℝ := (-1, 1)

-- Define the line containing the center of circle M
def center_line (x y : ℝ) : Prop := x + y - 2 = 0

-- Define the line containing point P
def P_line (x y : ℝ) : Prop := 3*x + 4*y + 8 = 0

-- Define the tangency condition
def is_tangent (p : ℝ × ℝ) (c : Set (ℝ × ℝ)) : Prop := sorry

-- Helper function to calculate the area of a quadrilateral
noncomputable def area_quadrilateral (P A M B : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem min_area_PAMB :
  ∃ (M_center : ℝ × ℝ),
    M_center ∈ circle_M ∧
    center_line M_center.1 M_center.2 ∧
    point_A ∈ circle_M ∧
    point_B ∈ circle_M ∧
    (∀ P : ℝ × ℝ, P_line P.1 P.2 →
      is_tangent P circle_M →
      area_quadrilateral P point_A M_center point_B ≥ 2 * Real.sqrt 5) ∧
    (∃ P : ℝ × ℝ, P_line P.1 P.2 ∧
      is_tangent P circle_M ∧
      area_quadrilateral P point_A M_center point_B = 2 * Real.sqrt 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_PAMB_l390_39083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l390_39050

-- Define the vectors
def a : Fin 2 → ℝ := ![1, 2]
def b : Fin 2 → ℝ := ![-2, 2]

-- Define dot product
def dot_product (v w : Fin 2 → ℝ) : ℝ := (v 0) * (w 0) + (v 1) * (w 1)

-- Define vector magnitude
noncomputable def magnitude (v : Fin 2 → ℝ) : ℝ := Real.sqrt ((v 0)^2 + (v 1)^2)

-- Define vector projection
noncomputable def proj (v w : Fin 2 → ℝ) : Fin 2 → ℝ := 
  fun i => (dot_product v w / (magnitude w)^2) * (w i)

theorem vector_properties :
  (dot_product a b = 2) ∧ 
  (proj a b 0 = -1/2 ∧ proj a b 1 = 1/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l390_39050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_triangle_l390_39038

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y^2 = 2x -/
def isOnParabola (p : Point) : Prop :=
  p.y^2 = 2 * p.x

/-- Represents the circle (x-1)^2 + y^2 = 1 -/
def isOnCircle (p : Point) : Prop :=
  (p.x - 1)^2 + p.y^2 = 1

/-- Calculate the area of a triangle given three points -/
noncomputable def triangleArea (a b c : Point) : ℝ :=
  abs ((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y)) / 2

/-- Main theorem -/
theorem min_area_triangle (A : Point) (hA : isOnParabola A) (hx : A.x > 2) :
  ∃ (B C : Point),
    B.x = 0 ∧ C.x = 0 ∧
    (∃ (T₁ T₂ : Point), isOnCircle T₁ ∧ isOnCircle T₂ ∧
      (T₁.x - A.x) * (T₁.y - B.y) = (T₁.y - A.y) * (T₁.x - B.x) ∧
      (T₂.x - A.x) * (T₂.y - C.y) = (T₂.y - A.y) * (T₂.x - C.x)) ∧
    (∀ (B' C' : Point),
      B'.x = 0 ∧ C'.x = 0 →
      (∃ (T₁' T₂' : Point), isOnCircle T₁' ∧ isOnCircle T₂' ∧
        (T₁'.x - A.x) * (T₁'.y - B'.y) = (T₁'.y - A.y) * (T₁'.x - B'.x) ∧
        (T₂'.x - A.x) * (T₂'.y - C'.y) = (T₂'.y - A.y) * (T₂'.x - C'.x)) →
      triangleArea A B C ≤ triangleArea A B' C') ∧
    triangleArea A B C = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_triangle_l390_39038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_inclusion_and_intersection_l390_39035

-- Define the sets A and B
def A : Set ℝ := {x | 2 ≤ x ∧ x ≤ 7}
def B (m : ℝ) : Set ℝ := {x | -3*m + 4 ≤ x ∧ x ≤ 2*m - 1}

-- State the theorem
theorem set_inclusion_and_intersection (m : ℝ) 
  (h_nonempty : Set.Nonempty (B m)) : 
  ((∀ x ∈ A, x ∈ B m) ↔ m ≥ 4) ∧ 
  ((∃ x ∈ B m, x ∈ A) ↔ m ≥ 3/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_inclusion_and_intersection_l390_39035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_zero_l390_39056

theorem sum_remainder_zero (a b c : ℕ) 
  (ha : a % 30 = 15)
  (hb : b % 30 = 5)
  (hc : c % 30 = 10) : 
  (a + b + c) % 30 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_zero_l390_39056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_lengths_in_prism_l390_39024

/-- Regular prism with base side 2a and lateral edge a -/
structure RegularPrism (a : ℝ) where
  base_side : ℝ := 2 * a
  lateral_edge : ℝ := a

/-- Point on diagonal AD₁ -/
noncomputable def point_on_diagonal (a : ℝ) : ℝ × ℝ × ℝ := (4*a/3, 0, 2*a/3)

/-- Theorem about segment length and minimum length in regular prism -/
theorem segment_lengths_in_prism (a : ℝ) (ha : a > 0) :
  let prism := RegularPrism a
  let M := point_on_diagonal a
  ∃ (segment_length min_length : ℝ),
    segment_length = a * Real.sqrt 5 / 3 ∧
    min_length = a / Real.sqrt 2 ∧
    (∀ other_length, other_length ≥ min_length) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_lengths_in_prism_l390_39024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_sin_leq_one_l390_39098

theorem negation_of_sin_leq_one :
  (¬ ∀ x : ℝ, Real.sin x ≤ 1) ↔ (∃ x : ℝ, Real.sin x ≥ 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_sin_leq_one_l390_39098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_girls_in_class_l390_39070

theorem girls_in_class (total : Nat) (prob : Rat) 
  (h1 : total = 25) (h2 : prob = 3/25) :
  ∃ (boys : Nat), boys * (boys - 1) = 72 ∧ 
  total - boys = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_girls_in_class_l390_39070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_cryptarithmetic_l390_39063

-- Define the variables as natural numbers
variable (F W O I V E : ℕ)

-- Define the constraints
def different_digits (F W O I V E : ℕ) : Prop :=
  F ≠ W ∧ F ≠ O ∧ F ≠ I ∧ F ≠ V ∧ F ≠ E ∧
  W ≠ O ∧ W ≠ I ∧ W ≠ V ∧ W ≠ E ∧
  O ≠ I ∧ O ≠ V ∧ O ≠ E ∧
  I ≠ V ∧ I ≠ E ∧
  V ≠ E

def T_is_8 : Prop := True  -- T is already fixed as 8

def O_is_odd (O : ℕ) : Prop := O % 2 = 1

def equation_holds (F W O I V E : ℕ) : Prop :=
  100 * 8 + 10 * W + O +
  100 * 8 + 10 * W + O =
  1000 * F + 100 * I + 10 * V + E

-- Theorem statement
theorem solve_cryptarithmetic :
  ∀ F W O I V E : ℕ,
  different_digits F W O I V E ∧ T_is_8 ∧ O_is_odd O ∧ equation_holds F W O I V E →
  W = 2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_cryptarithmetic_l390_39063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_ECD_is_central_angle_l390_39015

noncomputable section

/-- The radius of the Earth in kilometers. -/
def earth_radius : ℝ := 6371

/-- Convert degrees to radians. -/
def deg_to_rad (deg : ℝ) : ℝ := deg * (Real.pi / 180)

/-- Caroline's longitude in degrees. -/
def caroline_lon : ℝ := 9

/-- Caroline's latitude in degrees. -/
def caroline_lat : ℝ := 0

/-- David's longitude in degrees. -/
def david_lon : ℝ := 106.92

/-- David's latitude in degrees. -/
def david_lat : ℝ := 47.92

/-- Calculate the x-coordinate of a point on Earth's surface. -/
def x_coord (lat deg : ℝ) : ℝ := earth_radius * (deg_to_rad lat).cos * (deg_to_rad deg).cos

/-- Calculate the y-coordinate of a point on Earth's surface. -/
def y_coord (lat deg : ℝ) : ℝ := earth_radius * (deg_to_rad lat).cos * (deg_to_rad deg).sin

/-- Calculate the z-coordinate of a point on Earth's surface. -/
def z_coord (lat : ℝ) : ℝ := earth_radius * (deg_to_rad lat).sin

/-- The angle ECD in radians. -/
def angle_ECD : ℝ :=
  let c_x := x_coord caroline_lat caroline_lon
  let c_y := y_coord caroline_lat caroline_lon
  let c_z := z_coord caroline_lat
  let d_x := x_coord david_lat david_lon
  let d_y := y_coord david_lat david_lon
  let d_z := z_coord david_lat
  let dot_product := c_x * d_x + c_y * d_y + c_z * d_z
  let magnitude_product := earth_radius * earth_radius
  Real.arccos (dot_product / magnitude_product)

theorem angle_ECD_is_central_angle :
  angle_ECD = Real.arccos (
    (x_coord caroline_lat caroline_lon * x_coord david_lat david_lon +
     y_coord caroline_lat caroline_lon * y_coord david_lat david_lon +
     z_coord caroline_lat * z_coord david_lat) /
    (earth_radius * earth_radius)
  ) := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_ECD_is_central_angle_l390_39015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_count_in_range_l390_39009

theorem integer_count_in_range : ∃ (S : Finset ℤ), 
  (∀ x ∈ S, (|x| : ℝ) ≤ 2 * Real.pi ∧ x % 3 = 0) ∧ 
  (∀ x : ℤ, (|x| : ℝ) ≤ 2 * Real.pi ∧ x % 3 = 0 → x ∈ S) ∧ 
  S.card = 5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_count_in_range_l390_39009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_equality_proof_l390_39028

open Set

theorem set_equality_proof (U M N : Set ℕ) (h1 : U = {1,2,3,4,5,6}) 
  (h2 : M = {1,4}) (h3 : N = {2,3}) : 
  {5,6} = (U \ M) ∩ (U \ N) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_equality_proof_l390_39028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_T1_acute_angled_exists_right_triangle_count_similar_T3_count_similar_Tn_l390_39072

-- Define a triangle
structure Triangle where
  α : Real
  β : Real
  γ : Real
  sum_angles : α + β + γ = π

-- Define the sequence of triangles
def triangle_sequence (T : Triangle) : ℕ → Triangle
  | 0 => T
  | n + 1 => sorry  -- Definition of the next triangle in the sequence

-- Theorem 1: Conditions for T₁ to be acute-angled
theorem T1_acute_angled (T : Triangle) : 
  let T1 := triangle_sequence T 1
  (π/4 < T1.α ∧ T1.α < π/2 ∧ π/4 < T1.β ∧ T1.β < π/2 ∧ π/4 < T1.γ ∧ T1.γ < π/2) ∨
  (((T1.α < π/4 ∧ T1.β < π/4) ∨ (T1.α < π/4 ∧ T1.γ < π/4) ∨ (T1.β < π/4 ∧ T1.γ < π/4)) ∧
   (T1.α < 3*π/4 ∧ T1.β < 3*π/4 ∧ T1.γ < 3*π/4)) := by
  sorry

-- Theorem 2: Existence of a right triangle in the sequence
theorem exists_right_triangle (T : Triangle) : 
  ∃ (n : ℕ) (s : ℤ), T.α = π * s / (2^n) ∨ T.β = π * s / (2^n) ∨ T.γ = π * s / (2^n) := by
  sorry

-- Theorem 3: Number of non-similar triangles T such that T₃ is similar to T
theorem count_similar_T3 : 
  ∃ (n : ℕ), n = 56 := by
  sorry

-- Theorem 4: Number of non-similar triangles T such that Tₙ is similar to T
theorem count_similar_Tn (n : ℕ) : 
  ∃ (m : ℕ), m = 2^(2*n) - 2^n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_T1_acute_angled_exists_right_triangle_count_similar_T3_count_similar_Tn_l390_39072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_f_500_plus_1_l390_39019

def f (n : ℕ) : ℕ := 2^n

theorem divisors_of_f_500_plus_1 (h : Nat.Prime (f 500 + 1)) :
  (Finset.filter (fun d => (f 500 + 1) % d = 0) (Finset.range (f 500 + 2))).card = 2 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_f_500_plus_1_l390_39019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l390_39020

-- Define set A
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = Real.log (x^2 + 1)}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = Set.Icc 0 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l390_39020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_equals_36_l390_39044

/-- The length of wire used to make a regular hexagon with side length 4 cm -/
noncomputable def hexagon_perimeter : ℝ := 6 * 4

/-- The side length of a square made with the same amount of wire as the hexagon -/
noncomputable def square_side : ℝ := hexagon_perimeter / 4

/-- The area of the square -/
noncomputable def square_area : ℝ := square_side ^ 2

theorem square_area_equals_36 : square_area = 36 := by
  -- Unfold definitions
  unfold square_area square_side hexagon_perimeter
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_equals_36_l390_39044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_of_sum_divisors_180_l390_39010

def sum_of_divisors (n : ℕ) : ℕ := (Nat.divisors n).sum id

theorem largest_prime_factor_of_sum_divisors_180 :
  (Nat.factors (sum_of_divisors 180)).maximum? = some 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_of_sum_divisors_180_l390_39010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_square_area_ratio_l390_39053

/-- The ratio of the area of a rhombus inscribed in a square to the area of the square,
    where the rhombus vertices are at quarter points of the square's sides. -/
theorem rhombus_square_area_ratio : ℝ := by
  -- Let's define our square and rhombus
  let square_side : ℝ := 1
  
  -- Calculate the diagonals of the rhombus
  let diagonal : ℝ := Real.sqrt ((1/2)^2 + (1/2)^2)

  -- Calculate areas
  let rhombus_area : ℝ := (1/2) * diagonal * diagonal
  let square_area : ℝ := square_side^2

  -- Calculate and return the ratio
  have h : rhombus_area / square_area = 1/4 := by
    -- Proof goes here
    sorry

  exact 1/4


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_square_area_ratio_l390_39053
