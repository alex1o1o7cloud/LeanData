import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_unshaded_is_correct_l181_18121

/-- Represents a rectangle in the 2 by 2011 grid -/
structure Rectangle where
  left : ℕ
  right : ℕ
  top : ℕ
  bottom : ℕ

/-- The total number of vertical segments in the rectangle -/
def total_vertical_segments : ℕ := 2012

/-- The position of the shaded square in each row -/
def shaded_position : ℕ := 1006

/-- The set of all possible rectangles in the grid -/
def all_rectangles : Finset Rectangle := sorry

/-- The set of rectangles that do not include a shaded square -/
def unshaded_rectangles : Finset Rectangle := sorry

/-- The probability of choosing a rectangle that does not include a shaded square -/
noncomputable def probability_unshaded : ℚ :=
  (unshaded_rectangles.card : ℚ) / (all_rectangles.card : ℚ)

theorem probability_unshaded_is_correct :
  probability_unshaded = 1005 / 2011 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_unshaded_is_correct_l181_18121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_area_squared_l181_18132

/-- An equilateral triangle with vertices on the hyperbola xy = 1 and centroid at (1, 1) -/
structure SpecialTriangle where
  /-- The x-coordinates of the triangle vertices -/
  x : Fin 3 → ℝ
  /-- The y-coordinates of the triangle vertices -/
  y : Fin 3 → ℝ
  /-- The vertices lie on the hyperbola xy = 1 -/
  on_hyperbola : ∀ i, x i * y i = 1
  /-- The triangle is equilateral -/
  equilateral : ∀ i j, i ≠ j → (x i - x j)^2 + (y i - y j)^2 = (x 0 - x 1)^2 + (y 0 - y 1)^2
  /-- The centroid of the triangle is at (1, 1) -/
  centroid : (x 0 + x 1 + x 2) / 3 = 1 ∧ (y 0 + y 1 + y 2) / 3 = 1

/-- The theorem stating that the square of the area of the special triangle is 108 -/
theorem special_triangle_area_squared (t : SpecialTriangle) : 
  (((t.x 1 - t.x 0) * (t.y 2 - t.y 0) - (t.x 2 - t.x 0) * (t.y 1 - t.y 0)) / 2)^2 = 108 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_area_squared_l181_18132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_final_distance_l181_18193

-- Define the conversion factor from meters to feet
def meters_to_feet : ℝ := 3.28084

-- Define Alice's movements
def north_distance : ℝ := 12  -- in meters
def east_distance : ℝ := 40   -- in feet
def south_distance : ℝ := 12 * meters_to_feet + 18  -- in feet

-- Define the theorem
theorem alice_final_distance :
  let x := east_distance
  let y := south_distance
  abs (Real.sqrt (x^2 + y^2) - 43.874) < 0.001 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_final_distance_l181_18193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l181_18125

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1 / (x - 23)

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≠ 23} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l181_18125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_properties_l181_18160

-- Define a triangle ABC
structure Triangle where
  A : ℝ  -- Angle A
  B : ℝ  -- Angle B
  C : ℝ  -- Angle C
  a : ℝ  -- Side opposite to angle A
  b : ℝ  -- Side opposite to angle B
  c : ℝ  -- Side opposite to angle C
  angle_sum : A + B + C = π
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  sine_law : a / (Real.sin A) = b / (Real.sin B)
  cosine_law : a^2 = b^2 + c^2 - 2*b*c*(Real.cos A)

-- Define the specific conditions for our triangle
def SpecialTriangle (t : Triangle) : Prop :=
  (2*t.b - t.c) * Real.cos t.A = t.a * Real.cos t.C ∧ t.a = 3

-- State the theorem
theorem special_triangle_properties (t : Triangle) (h : SpecialTriangle t) :
  t.A = π/3 ∧ (∃ p : ℝ, p ≤ 9 ∧ ∀ q : ℝ, q = t.a + t.b + t.c → q ≤ p) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_properties_l181_18160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_day_snow_amount_l181_18108

/-- Represents the depth of a snowdrift over four days -/
structure Snowdrift where
  initial_depth : ℕ
  third_day_snow : ℕ
  fourth_day_snow : ℕ
  final_depth : ℕ

/-- Calculates the depth of the snowdrift after four days -/
def calculate_final_depth (s : Snowdrift) : ℕ :=
  (s.initial_depth / 2) + s.third_day_snow + s.fourth_day_snow

/-- Theorem stating that the amount of snow added on the third day is 6 inches -/
theorem third_day_snow_amount (s : Snowdrift) 
  (h1 : s.initial_depth = 20)
  (h2 : s.fourth_day_snow = 18)
  (h3 : s.final_depth = 34)
  (h4 : calculate_final_depth s = s.final_depth) : 
  s.third_day_snow = 6 := by
  sorry

#eval calculate_final_depth { initial_depth := 20, third_day_snow := 6, fourth_day_snow := 18, final_depth := 34 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_day_snow_amount_l181_18108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tangent_ratio_l181_18166

theorem triangle_tangent_ratio (A B C : ℝ) (a b c : ℝ) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi →
  -- a, b, c are positive
  0 < a ∧ 0 < b ∧ 0 < c →
  -- a, b, c are sides opposite to angles A, B, C respectively
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C →
  -- Given condition
  a * Real.cos B - b * Real.cos A = (3/5) * c →
  -- Conclusion
  Real.tan A / Real.tan B = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tangent_ratio_l181_18166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_not_arithmetic_triangle_area_l181_18158

open Real

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition from the problem -/
def triangle_condition (t : Triangle) : Prop :=
  4 * t.c * sin t.C = (t.b + t.a) * (sin t.B - sin t.A)

/-- Arithmetic sequence property -/
def is_arithmetic_sequence (x y z : ℝ) : Prop :=
  y = (x + z) / 2

theorem triangle_not_arithmetic (t : Triangle) 
  (h : triangle_condition t) : 
  ¬ (is_arithmetic_sequence t.a t.b t.c) :=
sorry

theorem triangle_area (t : Triangle) 
  (h1 : triangle_condition t)
  (h2 : t.b = 3 * t.c)
  (h3 : t.a + t.b + t.c = 4 + Real.sqrt 5) :
  (1/2) * t.b * t.c * sin t.A = Real.sqrt 11 / 4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_not_arithmetic_triangle_area_l181_18158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_length_sum_l181_18117

-- Define the length of an integer
def length (n : ℕ) : ℕ :=
  if n ≤ 1 then 0 else (Nat.factorization n).sum (fun _ v => v)

-- Theorem statement
theorem max_length_sum :
  ∃ (max_sum : ℕ), max_sum = 15 ∧ 
    ∀ (x y : ℕ), x > 1 → y > 1 → x + 3 * y < 1000 →
      length x + length y ≤ max_sum :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_length_sum_l181_18117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_integers_sum_l181_18188

theorem two_integers_sum (x y : ℕ+) : 
  x * y + x + y = 154 →
  Nat.Coprime x.val y.val →
  x.val < 30 →
  y.val < 30 →
  x.val + y.val = 34 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_integers_sum_l181_18188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cow_grazing_area_is_155π_l181_18100

/-- The area a cow can reach when tied to a square shed --/
noncomputable def cowGrazingArea (shedSideLength : ℝ) (ropeAttachmentDistance : ℝ) (ropeLength : ℝ) : ℝ :=
  let fullCircleArea := Real.pi * ropeLength^2
  let threeQuartersCircleArea := 3/4 * fullCircleArea
  let quarterCircleRadius := shedSideLength - ropeAttachmentDistance
  let twoQuarterCirclesArea := 2 * (1/4 * Real.pi * quarterCircleRadius^2)
  threeQuartersCircleArea + twoQuarterCirclesArea

/-- The theorem stating the area a cow can reach is 155π square meters --/
theorem cow_grazing_area_is_155π :
  cowGrazingArea 10 2 14 = 155 * Real.pi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cow_grazing_area_is_155π_l181_18100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l181_18175

-- Define the function f as noncomputable due to its dependency on Real.exp
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then Real.exp (x - 2) - 2 * Real.exp 1 else Real.exp (-x - 2) - 2 * Real.exp 1

-- State the theorem
theorem f_properties :
  (∀ x : ℝ, f x = f (-x)) ∧  -- f is even
  (∀ a : ℝ, f a + f 3 < 0 → -3 - Real.log 3 < a ∧ a < 3 + Real.log 3) :=
by
  sorry  -- Skip the proof for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l181_18175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_red_tetrahedron_volume_l181_18165

/-- A cube with alternately colored vertices --/
structure ColoredCube where
  side_length : ℝ
  is_alternately_colored : Bool

/-- The volume of a tetrahedron formed by vertices of one color in a colored cube --/
noncomputable def volume_colored_tetrahedron (cube : ColoredCube) : ℝ :=
  cube.side_length^3 - 4 * (1/3) * (1/2) * cube.side_length^2 * cube.side_length

/-- Theorem: The volume of the red tetrahedron in a cube with side length 10 --/
theorem red_tetrahedron_volume (cube : ColoredCube) 
  (h1 : cube.side_length = 10)
  (h2 : cube.is_alternately_colored = true) :
  volume_colored_tetrahedron cube = 1000 - 2000/3 := by
  sorry

/-- Compute the volume of the red tetrahedron --/
def compute_red_tetrahedron_volume : ℚ :=
  1000 - 2000/3

#eval compute_red_tetrahedron_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_red_tetrahedron_volume_l181_18165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_one_third_l181_18184

-- Define the ellipse
noncomputable def ellipse (a b : ℝ) (h : a > b ∧ b > 0) : Set (ℝ × ℝ) :=
  {p | (p.1 / a)^2 + (p.2 / b)^2 = 1}

-- Define the eccentricity
noncomputable def eccentricity (a b : ℝ) (h : a > b ∧ b > 0) : ℝ :=
  Real.sqrt (1 - (b / a)^2)

theorem ellipse_eccentricity_one_third
  (a b : ℝ) (h : a > b ∧ b > 0)
  (O : ℝ × ℝ) (hO : O = (0, 0))
  (F : ℝ × ℝ) (hF : F.1 < 0 ∧ F.2 = 0)
  (A B : ℝ × ℝ) (hA : A = (-a, 0)) (hB : B = (a, 0))
  (P : ℝ × ℝ) (hP : P ∈ ellipse a b h)
  (hPF_perp : P.1 = F.1)
  (M : ℝ × ℝ) (hM : ∃ k, M.2 = k * (M.1 + a))
  (E : ℝ × ℝ) (hE : E.1 = 0 ∧ ∃ k, E.2 = k * a)
  (hBM_midpoint : ∃ t, B.1 + t * (M.1 - B.1) = 0 ∧ B.2 + t * (M.2 - B.2) = E.2 / 2) :
  eccentricity a b h = 1/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_one_third_l181_18184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_properties_l181_18151

noncomputable def interior_angle (n : ℕ) : ℝ := (n - 2 : ℝ) * 180 / n

def can_tile_plane (n : ℕ) : Prop := ∃ k : ℕ, k > 0 ∧ k * interior_angle n = 360

theorem regular_polygon_properties :
  (interior_angle 5 = 108) ∧
  (interior_angle 6 = 120) ∧
  (interior_angle 8 = 135) ∧
  can_tile_plane 3 ∧
  can_tile_plane 4 ∧
  can_tile_plane 6 ∧
  ¬(can_tile_plane 5) ∧
  ¬(can_tile_plane 8) ∧
  (∀ m n : ℕ, m * 90 + n * 135 = 360 → m = 1 ∧ n = 2) :=
by sorry

#check regular_polygon_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_properties_l181_18151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_points_theorem_l181_18196

theorem circle_points_theorem (C : Set (ℝ × ℝ)) (P : Finset (ℝ × ℝ)) : 
  (∃ center : ℝ × ℝ, ∀ p : ℝ × ℝ, p ∈ C ↔ (p.1 - center.1)^2 + (p.2 - center.2)^2 ≤ 16) →
  (P.card = 251) →
  (∀ p ∈ P, p ∈ C) →
  (∃ small_circle_center : ℝ × ℝ, (P.filter (fun p ↦ (p.1 - small_circle_center.1)^2 + (p.2 - small_circle_center.2)^2 ≤ 1)).card ≥ 11) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_points_theorem_l181_18196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_l181_18189

/-- The length of a train given its speed, time to cross a bridge, and the bridge length. -/
theorem train_length (speed time bridge_length : ℝ) :
  speed = 72 * (1000 / 3600) → 
  time = 13.598912087033037 → 
  bridge_length = 132 → 
  speed * time - bridge_length = 140 := by
  sorry

-- Remove the #eval line as it's causing issues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_l181_18189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_rates_theorem_l181_18174

/-- Represents the tax system for Mork and Mindy -/
structure TaxSystem where
  mork_base_income : ℚ
  mork_base_tax_rate : ℚ
  mork_additional_tax_rate : ℚ
  mindy_base_tax_rate : ℚ
  mindy_additional_tax_rate : ℚ
  additional_income_percentage : ℚ

/-- Calculates the effective tax rates for Mork, Mindy, and their combined rate -/
def calculate_tax_rates (ts : TaxSystem) : ℚ × ℚ × ℚ :=
  let mork_total_income := ts.mork_base_income * (1 + ts.additional_income_percentage)
  let mindy_base_income := 4 * ts.mork_base_income
  let mindy_total_income := mindy_base_income * (1 + ts.additional_income_percentage)
  
  let mork_tax := ts.mork_base_income * ts.mork_base_tax_rate + 
                  (mork_total_income - ts.mork_base_income) * ts.mork_additional_tax_rate
  let mindy_tax := mindy_base_income * ts.mindy_base_tax_rate + 
                   (mindy_total_income - mindy_base_income) * ts.mindy_additional_tax_rate
  
  let mork_rate := mork_tax / mork_total_income
  let mindy_rate := mindy_tax / mindy_total_income
  let combined_rate := (mork_tax + mindy_tax) / (mork_total_income + mindy_total_income)
  
  (mork_rate, mindy_rate, combined_rate)

/-- Theorem stating the effective tax rates for Mork, Mindy, and their combined rate -/
theorem tax_rates_theorem (ts : TaxSystem) 
  (h1 : ts.mork_base_tax_rate = 2/5)
  (h2 : ts.mork_additional_tax_rate = 1/2)
  (h3 : ts.mindy_base_tax_rate = 3/10)
  (h4 : ts.mindy_additional_tax_rate = 7/20)
  (h5 : ts.additional_income_percentage = 1/2) :
  let (mork_rate, mindy_rate, combined_rate) := calculate_tax_rates ts
  mork_rate = 13/30 ∧ mindy_rate = 19/60 ∧ combined_rate = 17/50 := by
  sorry

#eval calculate_tax_rates { 
  mork_base_income := 100, 
  mork_base_tax_rate := 2/5, 
  mork_additional_tax_rate := 1/2, 
  mindy_base_tax_rate := 3/10, 
  mindy_additional_tax_rate := 7/20, 
  additional_income_percentage := 1/2 
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_rates_theorem_l181_18174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_on_circle_and_line_l181_18152

/-- Circle C in the xy-plane -/
def circleC (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 4

/-- Line l in the xy-plane -/
def lineL (y : ℝ) : Prop := y = 2

/-- The intersection point of circle C and line l -/
def intersection_point : ℝ × ℝ := (1, 2)

/-- Theorem stating that the intersection_point lies on both the circle and the line -/
theorem intersection_point_on_circle_and_line :
  circleC intersection_point.1 intersection_point.2 ∧
  lineL intersection_point.2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_on_circle_and_line_l181_18152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chair_permutations_l181_18124

/-- Represents a circular arrangement of n chairs -/
def CircularArrangement (n : ℕ) := Fin n

/-- Checks if two positions in a circular arrangement are adjacent -/
def isAdjacent (n : ℕ) (a b : Fin n) : Prop :=
  (a.val + 1 ≡ b.val [MOD n]) ∨ (b.val + 1 ≡ a.val [MOD n])

/-- A permutation of people in chairs -/
def Permutation (n : ℕ) := Fin n → Fin n

/-- Checks if a permutation is valid according to the problem rules -/
def isValidPermutation (n : ℕ) (p : Permutation n) : Prop :=
  Function.Bijective p ∧
  ∀ i : Fin n, p i ≠ i ∧ ¬isAdjacent n (p i) i

/-- The main theorem to be proved -/
theorem chair_permutations :
  ∃ (validPerms : Finset (Permutation 8)),
    (∀ p ∈ validPerms, isValidPermutation 8 p) ∧
    validPerms.card = 80 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chair_permutations_l181_18124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_n_formula_l181_18102

noncomputable section

/-- The function f(x) = ln(1+x) -/
def f (x : ℝ) : ℝ := Real.log (1 + x)

/-- The derivative of f(x) -/
def f' : ℝ → ℝ := deriv f

/-- The function g(x) = x * f'(x) -/
def g (x : ℝ) : ℝ := x * f' x

/-- The recursive definition of gₙ(x) -/
def g_n : ℕ → (ℝ → ℝ)
  | 0 => g  -- Added case for 0
  | n + 1 => λ x => g (g_n n x)

/-- The main theorem: For all n ∈ ℕ₊ and x ≥ 0, gₙ(x) = x / (1 + nx) -/
theorem g_n_formula (n : ℕ) (x : ℝ) (h1 : n > 0) (h2 : x ≥ 0) :
  g_n n x = x / (1 + n * x) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_n_formula_l181_18102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_finite_orbit_closed_set_l181_18109

open Set
open Function
open Topology

def is_closed_set {α : Type*} [TopologicalSpace α] [MetricSpace α] (S : Set α) : Prop :=
  ∀ x, x ∉ S → ∃ δ > 0, ∀ x' ∈ S, dist x' x ≥ δ

theorem finite_orbit_closed_set (a b : ℝ) (f : ℝ → ℝ) (p : ℝ) :
  a ≤ b →
  ContinuousOn f (Icc a b) →
  p ∈ Icc a b →
  (∀ x ∈ Icc a b, f x ∈ Icc a b) →
  let T : Set ℝ := {x | ∃ n : ℕ, x = (f^[n]) p}
  is_closed_set T →
  Finite T :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_finite_orbit_closed_set_l181_18109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_age_ratio_problem_l181_18143

/-- Proves that the ratio of Collete's age to Rona's age is 1:2 given the problem conditions -/
theorem age_ratio_problem (rachel rona collete : ℚ) : 
  rachel = 2 * rona →  -- Rachel is twice as old as Rona
  rona = 8 →  -- Rona is 8 years old
  rachel - collete = 12 →  -- The difference between the age of Collete and Rachel is 12 years
  collete / rona = 1 / 2 := by  -- The ratio of Collete's age to Rona's age is 1:2
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_age_ratio_problem_l181_18143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_perpendicular_condition_l181_18154

/-- Represents an ellipse with center at origin -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The ellipse passes through point A(-2,0) -/
def passes_through_A (e : Ellipse) : Prop :=
  ((-2)^2 / e.a^2) + (0^2 / e.b^2) = 1

/-- The right focus of the ellipse is at F(1,0) -/
def right_focus_at_F (e : Ellipse) : Prop :=
  e.a^2 - e.b^2 = 1

/-- Point P is on the ellipse -/
def on_ellipse (e : Ellipse) (p : Point) : Prop :=
  (p.x^2 / e.a^2) + (p.y^2 / e.b^2) = 1

/-- M is the midpoint of AP -/
noncomputable def midpoint_M (p : Point) : Point :=
  { x := (p.x - 2) / 2, y := p.y / 2 }

/-- Q is the intersection point -/
noncomputable def intersection_Q (p : Point) (t : ℝ) : Point :=
  { x := t, y := (t * p.y) / (p.x + 2) }

/-- OM is perpendicular to FQ -/
def OM_perp_FQ (p : Point) (t : ℝ) : Prop :=
  let m := midpoint_M p
  let q := intersection_Q p t
  (m.y / m.x) * ((q.y) / (q.x - 1)) = -1

theorem ellipse_perpendicular_condition (e : Ellipse) :
  passes_through_A e →
  right_focus_at_F e →
  ∃ t : ℝ, t > 1 ∧
    ∀ p : Point, p.y ≠ 0 →
      on_ellipse e p →
      (OM_perp_FQ p t ↔ t = 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_perpendicular_condition_l181_18154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_p_l181_18133

/-- The weighted sum of the sequence {a_n} -/
def A (n : ℕ) : ℝ := n * 2^(n+1)

/-- The n-th term of the sequence {a_n} -/
def a (n : ℕ) : ℝ := 2*n + 2

/-- The sum of the first n terms of the sequence {a_n + pn} -/
noncomputable def T (n : ℕ) (p : ℝ) : ℝ := ((2+p)*n^2 + (p+6)*n) / 2

/-- The theorem stating the range of p -/
theorem range_of_p :
  ∀ p : ℝ, (∀ n : ℕ+, T n p ≤ T 6 p) ↔ -7/3 ≤ p ∧ p ≤ -16/7 :=
by sorry

#check range_of_p

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_p_l181_18133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_theorem_l181_18176

-- Define the ellipse
def C (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the line
def L (k b : ℝ) (x y : ℝ) : Prop := y = k * x + b

-- Define the intersection points
noncomputable def A : ℝ × ℝ := ⟨0, 0⟩  -- Placeholder values
noncomputable def B : ℝ × ℝ := ⟨0, 0⟩  -- Placeholder values

-- Define the midpoint
noncomputable def M (m : ℝ) : ℝ × ℝ := ⟨1, m⟩

-- Define the focus
def F : ℝ × ℝ := ⟨1, 0⟩

-- Define point P
noncomputable def P : ℝ × ℝ := ⟨0, 0⟩  -- Placeholder value

-- Theorem statement
theorem ellipse_intersection_theorem 
  (k : ℝ)  -- Slope of the line
  (m : ℝ)  -- y-coordinate of midpoint M
  (h1 : C A.1 A.2)  -- A is on the ellipse
  (h2 : C B.1 B.2)  -- B is on the ellipse
  (h3 : ∃ b, L k b A.1 A.2)  -- A is on the line
  (h4 : ∃ b, L k b B.1 B.2)  -- B is on the line
  (h5 : M m = ⟨(A.1 + B.1) / 2, (A.2 + B.2) / 2⟩)  -- M is the midpoint of AB
  (h6 : m > 0)  -- m > 0
  (h7 : C P.1 P.2)  -- P is on the ellipse
  (h8 : (P.1 - F.1, P.2 - F.2) + (A.1 - F.1, A.2 - F.2) + (B.1 - F.1, B.2 - F.2) = (0, 0))  -- FP + FA + FB = 0
  : 
  (k < -1/2) ∧  -- Slope is less than -1/2
  (∃ d : ℝ, (Real.sqrt ((A.1 - F.1)^2 + (A.2 - F.2)^2) = Real.sqrt ((P.1 - F.1)^2 + (P.2 - F.2)^2) - d ∧ 
             Real.sqrt ((B.1 - F.1)^2 + (B.2 - F.2)^2) = Real.sqrt ((P.1 - F.1)^2 + (P.2 - F.2)^2) + d)) ∧  -- Arithmetic sequence
  (∃ d : ℝ, d = 3 * Real.sqrt 21 / 28 ∨ d = -(3 * Real.sqrt 21 / 28))  -- Common difference
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_theorem_l181_18176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_factors_is_81_l181_18173

/-- The number of positive factors of a positive integer -/
def num_factors (m : ℕ) : ℕ := sorry

/-- The maximum number of positive factors for b^n where b and n are positive integers ≤ 20 -/
def max_factors : ℕ :=
  Finset.sup (Finset.range 20 ×ˢ Finset.range 20) (fun (b, n) ↦ num_factors ((b + 1) ^ (n + 1)))

theorem max_factors_is_81 : max_factors = 81 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_factors_is_81_l181_18173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l181_18144

-- Define the inequality and its solution set
def inequality (m x : ℝ) : Prop := m - |x - 2| ≥ 1
def solution_set (m : ℝ) : Set ℝ := {x | inequality m x}

-- Define the theorem
theorem inequality_solution (m : ℝ) :
  (solution_set m = Set.Icc 0 4) → 
  (m = 3) ∧ 
  (∀ a b : ℝ, a > 0 → b > 0 → a + b = m → a^2 + b^2 ≥ 9/2) ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a + b = m ∧ a^2 + b^2 = 9/2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l181_18144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_volume_of_intersection_l181_18103

/-- S is a region in 3D space where -1 ≤ z ≤ 1 -/
def S : Set (ℝ × ℝ × ℝ) :=
  {p : ℝ × ℝ × ℝ | -1 ≤ p.2.2 ∧ p.2.2 ≤ 1}

/-- A random rotation in 3D space -/
def RandomRotation : Type :=
  (ℝ × ℝ × ℝ) → (ℝ × ℝ × ℝ)

/-- S₁, S₂, ..., S₂₀₂₂ are 2022 independent random rotations of S about the origin (0,0,0) -/
noncomputable def rotatedSets : Fin 2022 → Set (ℝ × ℝ × ℝ) :=
  fun i => { p | ∃ (r : RandomRotation) (s : S), r s = p }

/-- The intersection of all rotated sets -/
noncomputable def intersectionOfRotatedSets : Set (ℝ × ℝ × ℝ) :=
  ⋂ i : Fin 2022, rotatedSets i

/-- The expected volume of the intersection -/
noncomputable def expectedVolume : ℝ := sorry

theorem expected_volume_of_intersection :
  expectedVolume = (2692 * Real.pi) / 2019 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_volume_of_intersection_l181_18103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l181_18127

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  h1 : 0 < a ∧ 0 < b ∧ 0 < c
  h2 : 0 < A ∧ 0 < B ∧ 0 < C
  h3 : A + B + C = Real.pi

-- Main theorem
theorem triangle_properties (t : Triangle) 
  (h : t.b * (Real.cos t.A - 2 * Real.cos t.C) = (2 * t.c - t.a) * Real.cos t.B) :
  -- Part 1
  Real.sin t.A / Real.sin t.C = 1 / 2 ∧
  -- Part 2
  (t.b = 2 ∧ Real.cos t.B = 1 / 4 → 
    let S := 1 / 2 * t.a * t.c * Real.sin t.B
    S = Real.sqrt 15 / 4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l181_18127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_less_f2_less_f3_l181_18111

-- Define the functions f and g
def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

-- State the conditions
axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom g_even : ∀ x : ℝ, g (-x) = g x
axiom f_minus_g : ∀ x : ℝ, f x - g x = 2^x

-- State the theorem to be proved
theorem g_less_f2_less_f3 : g 0 < f 2 ∧ f 2 < f 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_less_f2_less_f3_l181_18111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l181_18191

-- Define the ⊕ operation
noncomputable def oplus (a b : ℝ) : ℝ :=
  if a ≥ b then a else b^2

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  (oplus 1 x) * x - (oplus 2 x)

-- Theorem statement
theorem max_value_of_f :
  ∃ (M : ℝ), M = 6 ∧ ∀ x ∈ Set.Icc (-2 : ℝ) 2, f x ≤ M := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l181_18191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_symmetry_center_l181_18142

-- Define the tangent function
noncomputable def f (x : ℝ) : ℝ := Real.tan (Real.pi * x + Real.pi / 4)

-- Define the symmetry center
noncomputable def symmetry_center (k : ℤ) : ℝ × ℝ := ((2 * k - 1) / 4, 0)

-- Theorem statement
theorem tangent_symmetry_center :
  ∀ k : ℤ, ∃ (center : ℝ × ℝ), 
    center = symmetry_center k ∧
    ∀ x : ℝ, f (center.1 + x) = -f (center.1 - x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_symmetry_center_l181_18142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_expansion_equality_l181_18150

def A : Matrix (Fin 3) (Fin 3) ℤ := !![3, 1, 2; -1, 2, 5; 0, -4, 2]

theorem det_expansion_equality : 
  Matrix.det A = 
    (A 0 0 * Matrix.det !![A 1 1, A 1 2; A 2 1, A 2 2] - 
     A 0 1 * Matrix.det !![A 1 0, A 1 2; A 2 0, A 2 2] + 
     A 0 2 * Matrix.det !![A 1 0, A 1 1; A 2 0, A 2 1]) ∧
  Matrix.det A = 
    (-A 1 0 * Matrix.det !![A 0 1, A 0 2; A 2 1, A 2 2] + 
     A 1 1 * Matrix.det !![A 0 0, A 0 2; A 2 0, A 2 2] - 
     A 1 2 * Matrix.det !![A 0 0, A 0 1; A 2 0, A 2 1]) ∧
  Matrix.det A = 82 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_expansion_equality_l181_18150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_difference_angles_l181_18183

-- Define the angles α and β
noncomputable def α : ℝ := Real.arctan 2
noncomputable def β : ℝ := Real.arctan (1/5)

-- State the theorem
theorem tan_difference_angles : Real.tan (α - β) = 9/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_difference_angles_l181_18183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_real_coefficients_2010_is_zero_l181_18163

noncomputable section

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the function for the sum of real coefficients
noncomputable def sum_real_coefficients (n : ℕ) : ℝ :=
  (((1 + i)^n + (1 - i)^n) / 2).re

-- The theorem to prove
theorem sum_real_coefficients_2010_is_zero :
  sum_real_coefficients 2010 = 0 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_real_coefficients_2010_is_zero_l181_18163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_squares_in_H_l181_18112

-- Define the set H
def H : Set (ℤ × ℤ) := {p | -8 ≤ p.1 ∧ p.1 ≤ 8 ∧ -8 ≤ p.2 ∧ p.2 ≤ 8}

-- Define a square with vertices in H
def is_square_in_H (a b c d : ℤ × ℤ) : Prop :=
  a ∈ H ∧ b ∈ H ∧ c ∈ H ∧ d ∈ H ∧
  ∃ (side : ℤ), side ≥ 8 ∧
  (a.1 - b.1)^2 + (a.2 - b.2)^2 = side^2 ∧
  (b.1 - c.1)^2 + (b.2 - c.2)^2 = side^2 ∧
  (c.1 - d.1)^2 + (c.2 - d.2)^2 = side^2 ∧
  (d.1 - a.1)^2 + (d.2 - a.2)^2 = side^2

-- Theorem statement
theorem count_squares_in_H :
  ∃ (squares : Finset ((ℤ × ℤ) × (ℤ × ℤ) × (ℤ × ℤ) × (ℤ × ℤ))),
    squares.card = 4 ∧
    (∀ s ∈ squares, is_square_in_H s.1 s.2.1 s.2.2.1 s.2.2.2) ∧
    (∀ a b c d, is_square_in_H a b c d → ∃ s ∈ squares, s = (a, b, c, d)) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_squares_in_H_l181_18112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_range_l181_18140

/-- The angle B in a triangle ABC -/
noncomputable def angle_B (a b c : ℝ) : ℝ := Real.arccos ((a^2 + c^2 - b^2) / (2*a*c))

/-- In a triangle ABC where b^2 = ac, the range of sin B + cos B is (1, √2] -/
theorem sin_plus_cos_range (a b c : ℝ) (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : b^2 = a*c) :
  ∃ (x : ℝ), 1 < x ∧ x ≤ Real.sqrt 2 ∧ Real.sin (angle_B a b c) + Real.cos (angle_B a b c) = x :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_range_l181_18140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_co2_production_volume_l181_18161

-- Define the given constants and variables
noncomputable def total_volume : ℝ := 20
noncomputable def nitrogen_fraction : ℝ := 0.3
noncomputable def molar_volume : ℝ := 22.4

-- Define the balanced equation ratios
noncomputable def o2_ch4_ratio : ℝ := 2
noncomputable def co2_ch4_ratio : ℝ := 1

-- Define the function to calculate the volume of CO₂ produced
noncomputable def co2_volume (total_vol nitrogen_frac molar_vol o2_ch4_rat co2_ch4_rat : ℝ) : ℝ :=
  let oxygen_volume := total_vol * (1 - nitrogen_frac)
  let ch4_volume := oxygen_volume / o2_ch4_rat
  ch4_volume * co2_ch4_rat

-- State the theorem
theorem co2_production_volume :
  co2_volume total_volume nitrogen_fraction molar_volume o2_ch4_ratio co2_ch4_ratio = 7 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_co2_production_volume_l181_18161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dodecagon_in_triangle_l181_18139

/-- Given an equilateral triangle with side length 20 cm, prove the maximum side length of an inscribed regular dodecagon and the waste fraction. -/
theorem dodecagon_in_triangle :
  let triangle_side : ℝ := 20
  let inscribed_circle_radius : ℝ := triangle_side / 2
  let dodecagon_side : ℝ := 2 * inscribed_circle_radius * Real.sin (π / 12)
  let triangle_area : ℝ := (Real.sqrt 3 / 4) * triangle_side ^ 2
  let dodecagon_area : ℝ := 3 * inscribed_circle_radius ^ 2
  let waste_fraction : ℝ := 1 - dodecagon_area / triangle_area
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.0001 ∧ 
    (abs (dodecagon_side - 5.1764) < ε) ∧ 
    (abs (waste_fraction - 0.381) < ε) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dodecagon_in_triangle_l181_18139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_road_trip_average_speed_l181_18119

/-- Calculates the average speed for a road trip with multiple segments and breaks -/
noncomputable def averageSpeed (
  totalDistance : ℝ)
  (segment1Distance : ℝ) (segment1Speed : ℝ)
  (segment2Distance : ℝ) (segment2Speed : ℝ)
  (break1Duration : ℝ)
  (segment3Distance : ℝ) (segment3Speed : ℝ)
  (segment4Distance : ℝ) (segment4Speed : ℝ)
  (break2Duration : ℝ) : ℝ :=
  let totalTime := segment1Distance / segment1Speed +
                   segment2Distance / segment2Speed +
                   break1Duration +
                   segment3Distance / segment3Speed +
                   segment4Distance / segment4Speed +
                   break2Duration
  totalDistance / totalTime

/-- Theorem stating that the average speed for the given road trip is approximately 40.92 mph -/
theorem road_trip_average_speed :
  let result := averageSpeed 300 50 30 100 60 0.5 100 50 50 40 0.25
  ∃ ε > 0, |result - 40.92| < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_road_trip_average_speed_l181_18119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_monotone_and_bounded_l181_18164

def a : ℕ → ℚ
  | 0 => 1/2
  | n + 1 => 1/2 + (a n)^2 / 2

theorem a_monotone_and_bounded :
  (∀ n : ℕ, a (n + 1) > a n) ∧ (∀ n : ℕ, a n < 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_monotone_and_bounded_l181_18164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_thingamajig_production_l181_18197

-- Define the production rates for gadgets and thingamajigs
noncomputable def gadget_rate : ℝ := 450 / (150 * 1)
noncomputable def thingamajig_rate : ℝ := 300 / (150 * 1)

-- Define the production scenario functions
noncomputable def scenario1 (workers : ℝ) (hours : ℝ) : ℝ × ℝ :=
  (workers * hours * gadget_rate, workers * hours * thingamajig_rate)

def scenario2 (workers : ℝ) (hours : ℝ) : ℝ × ℝ :=
  (360, 450)

noncomputable def scenario3 (workers : ℝ) (hours : ℝ) : ℝ × ℝ :=
  (300, workers * hours * thingamajig_rate)

-- State the theorem
theorem thingamajig_production :
  scenario3 75 4 = (300, 600) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_thingamajig_production_l181_18197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_zero_domain_of_f_range_of_f_on_interval_l181_18153

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 3 * Real.sin (2 * x) + Real.cos (2 * x) + 1) / (2 * Real.cos x)

-- Theorem for f(0)
theorem f_at_zero : f 0 = 1 := by sorry

-- Define the domain of f
def domain_f : Set ℝ := {x | ∀ k : ℤ, x ≠ Real.pi / 2 + k * Real.pi}

-- Theorem for the domain of f
theorem domain_of_f : {x : ℝ | f x ≠ 0} = domain_f := by sorry

-- Define the range of f on (0, π/2)
def range_f : Set ℝ := Set.Ioo 1 2

-- Theorem for the range of f on (0, π/2)
theorem range_of_f_on_interval : 
  {y | ∃ x ∈ Set.Ioo 0 (Real.pi / 2), f x = y} = range_f := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_zero_domain_of_f_range_of_f_on_interval_l181_18153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_extreme_values_implies_m_equals_3_l181_18115

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 - (1/2) * (m+1) * x^2 + 2*(m-1) * x

-- Define the derivative of f
noncomputable def f' (m : ℝ) (x : ℝ) : ℝ := x^2 - (m+1) * x + 2*(m-1)

-- Theorem statement
theorem no_extreme_values_implies_m_equals_3 :
  (∀ x ∈ Set.Ioo 0 4, ∀ y ∈ Set.Ioo 0 4, f' m x * f' m y > 0) →
  m = 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_extreme_values_implies_m_equals_3_l181_18115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l181_18194

/-- Curve C in the Cartesian coordinate system -/
noncomputable def curve_C (θ : ℝ) : ℝ × ℝ := (1 + 4 * Real.cos θ, 2 + 4 * Real.sin θ)

/-- Point P -/
def point_P : ℝ × ℝ := (3, 5)

/-- Line l passing through point P with inclination angle π/3 -/
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (3 + t / 2, 5 + t * Real.sqrt 3 / 2)

/-- Theorem stating the product of distances from P to intersection points is 3 -/
theorem intersection_distance_product :
  ∃ (t₁ t₂ : ℝ), 
    (∃ (θ₁ θ₂ : ℝ), curve_C θ₁ = line_l t₁ ∧ curve_C θ₂ = line_l t₂) →
    (t₁ - t₂ ≠ 0) →
    Real.sqrt ((line_l t₁).1 - point_P.1)^2 + ((line_l t₁).2 - point_P.2)^2 *
    Real.sqrt ((line_l t₂).1 - point_P.1)^2 + ((line_l t₂).2 - point_P.2)^2 = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l181_18194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crosswalk_stripe_distance_approx_l181_18145

/-- The distance between stripes in a crosswalk --/
noncomputable def crosswalk_stripe_distance (street_width : ℝ) (crossing_angle : ℝ) (curb_length : ℝ) (stripe_length : ℝ) : ℝ :=
  let area := street_width * curb_length
  let base := curb_length / Real.cos crossing_angle
  area / stripe_length

/-- Theorem stating the distance between crosswalk stripes --/
theorem crosswalk_stripe_distance_approx :
  let street_width := (60 : ℝ)
  let crossing_angle := 30 * Real.pi / 180
  let curb_length := (20 : ℝ)
  let stripe_length := (65 : ℝ)
  abs (crosswalk_stripe_distance street_width crossing_angle curb_length stripe_length - 18.462) < 0.001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_crosswalk_stripe_distance_approx_l181_18145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_for_tangent_circle_main_theorem_l181_18187

/-- A circle with center (h, k) and radius r has the equation (x - h)² + (y - k)² = r² --/
def is_circle_equation (h k r : ℝ) (f : ℝ → ℝ → Prop) :=
  ∀ x y, f x y ↔ (x - h)^2 + (y - k)^2 = r^2

/-- A point (x, y) is on the x-axis if y = 0 --/
def on_x_axis (x y : ℝ) : Prop := y = 0

/-- A circle is tangent to the x-axis if there exists exactly one point that is both on the circle and on the x-axis --/
def circle_tangent_to_x_axis (h k r : ℝ) : Prop :=
  ∃! x, ∃ y, on_x_axis x y ∧ (x - h)^2 + (y - k)^2 = r^2

theorem circle_equation_for_tangent_circle (h k : ℝ) (hk : k > 0) :
  is_circle_equation h k k (λ x y ↦ (x - h)^2 + (y - k)^2 = k^2) ∧
  circle_tangent_to_x_axis h k k :=
sorry

theorem main_theorem :
  is_circle_equation (-3) 4 4 (λ x y ↦ (x + 3)^2 + (y - 4)^2 = 16) ∧
  circle_tangent_to_x_axis (-3) 4 4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_for_tangent_circle_main_theorem_l181_18187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l181_18180

noncomputable def f (x : ℝ) : ℝ := (x - 2) * (x - 3) * (x - 4) * (x - 7) / ((x - 1) * (x - 5) * (x - 6) * (x - 8))

def solution_set : Set ℝ := Set.Iic 1 ∪ Set.Ioo 3 4 ∪ Set.Ioo 6 7 ∪ Set.Ioi 8

theorem inequality_solution :
  {x : ℝ | f x > 0} = solution_set := by
  sorry

#check inequality_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l181_18180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_factors_60_l181_18136

/-- The sum of the positive factors of 60 is 168. -/
theorem sum_of_factors_60 : (Finset.filter (λ x => 60 % x = 0) (Finset.range 61)).sum id = 168 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_factors_60_l181_18136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_circle_intersection_l181_18156

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : 0 < a
  h_pos_b : 0 < b

/-- Represents a circle with center (h, k) and radius r -/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ
  h_pos_r : 0 < r

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := Real.sqrt (1 + h.b^2 / h.a^2)

/-- The distance between two points -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

/-- Theorem: The length of the chord formed by the intersection of a hyperbola's asymptote and a circle -/
theorem hyperbola_asymptote_circle_intersection
  (h : Hyperbola)
  (c : Circle)
  (h_ecc : eccentricity h = Real.sqrt 5)
  (h_circle : c.h = 2 ∧ c.k = 3 ∧ c.r = 1)
  : ∃ (x₁ y₁ x₂ y₂ : ℝ), distance x₁ y₁ x₂ y₂ = 4 * Real.sqrt 5 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_circle_intersection_l181_18156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l181_18122

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := 2017 * x + Real.sin (x / 2018) + (2019^x - 1) / (2019^x + 1)

-- State the theorem
theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : f (2 * a) + f (b - 4) = 0) :
  (1 / a + 2 / b) ≥ 2 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ f (2 * a₀) + f (b₀ - 4) = 0 ∧ 1 / a₀ + 2 / b₀ = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l181_18122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l181_18128

/-- Helper function to calculate the area of a triangle given its three vertices -/
noncomputable def area_triangle (v1 v2 v3 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := v1
  let (x2, y2) := v2
  let (x3, y3) := v3
  (1/2) * abs (x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))

/-- Helper function to calculate the area of a quadrilateral given its four vertices -/
noncomputable def area_quadrilateral (v1 v2 v3 v4 : ℝ × ℝ) : ℝ :=
  area_triangle v1 v2 v3 + area_triangle v1 v3 v4

/-- A quadrilateral with vertices at (6,4), (0,0), (-15,0), and (0,y) where y > 0 has an area of 60 square units if and only if y = 4 -/
theorem quadrilateral_area (y : ℝ) (h : y > 0) : 
  area_quadrilateral (6,4) (0,0) (-15,0) (0,y) = 60 ↔ y = 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l181_18128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_container_cost_minimization_l181_18155

/-- The minimum cost of a rectangular container -/
noncomputable def min_container_cost (volume : ℝ) (height : ℝ) (base_cost : ℝ) (side_cost : ℝ) : ℝ :=
  let base_area := volume / height
  let base_dim := Real.sqrt base_area
  (base_cost * base_area) + (side_cost * 2 * (base_dim * 2 + base_dim * 2))

/-- The optimal base dimensions of a rectangular container -/
noncomputable def optimal_base_dim (volume : ℝ) (height : ℝ) : ℝ :=
  Real.sqrt (volume / height)

theorem container_cost_minimization :
  let volume : ℝ := 4
  let height : ℝ := 1
  let base_cost : ℝ := 20
  let side_cost : ℝ := 10
  (min_container_cost volume height base_cost side_cost = 160) ∧
  (optimal_base_dim volume height = 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_container_cost_minimization_l181_18155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_property_l181_18131

/-- A cubic polynomial P(x) = x^3 + ax^2 + bx + c -/
structure CubicPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The sum of zeros of a cubic polynomial -/
noncomputable def sumOfZeros (p : CubicPolynomial) : ℝ := -p.a

/-- The product of zeros of a cubic polynomial -/
noncomputable def productOfZeros (p : CubicPolynomial) : ℝ := p.c

/-- The sum of coefficients of a cubic polynomial -/
noncomputable def sumOfCoefficients (p : CubicPolynomial) : ℝ := 1 + p.a + p.b + p.c

/-- The mean of zeros of a cubic polynomial -/
noncomputable def meanOfZeros (p : CubicPolynomial) : ℝ := -(p.a / 3)

theorem cubic_polynomial_property (p : CubicPolynomial) :
  meanOfZeros p = productOfZeros p ∧
  meanOfZeros p = sumOfCoefficients p ∧
  p.c = 3 →
  p.b = -16 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_property_l181_18131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_and_ratio_l181_18105

-- Define the circles C1 and C2
def C1 (x y : ℝ) : Prop := x^2 + y^2 = 2
def C2 (x y : ℝ) : Prop := x^2/2 + y^2 = 1

-- Define the relationship between P, Q, and M
def PQM_relation (xp yp xm ym : ℝ) : Prop :=
  C1 xp yp ∧ Real.sqrt 2 * ym = yp ∧ xm = xp

-- Define the tangent points A and B
def tangent_points (xt yt xa ya xb yb : ℝ) : Prop :=
  xt = 2 ∧ C1 xa ya ∧ C1 xb yb ∧
  (xt - xa) * xa + (yt - ya) * ya = 0 ∧
  (xt - xb) * xb + (yt - yb) * yb = 0

-- Define the intersection points C and D
def intersection_points (xa ya xb yb xc yc xd yd : ℝ) : Prop :=
  C2 xc yc ∧ C2 xd yd ∧
  (yc - ya) * (xb - xa) = (yb - ya) * (xc - xa) ∧
  (yd - ya) * (xb - xa) = (yb - ya) * (xd - xa)

theorem locus_and_ratio :
  (∀ x y, PQM_relation x y x y → C2 x y) ∧
  (∀ xt yt xa ya xb yb xc yc xd yd,
    tangent_points xt yt xa ya xb yb →
    intersection_points xa ya xb yb xc yc xd yd →
    Real.sqrt 2/2 ≤ (Real.sqrt ((xc - xd)^2 + (yc - yd)^2)) / (Real.sqrt ((xa - xb)^2 + (ya - yb)^2)) ∧
    (Real.sqrt ((xc - xd)^2 + (yc - yd)^2)) / (Real.sqrt ((xa - xb)^2 + (ya - yb)^2)) < 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_and_ratio_l181_18105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_proof_l181_18118

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y = 0

/-- The area of the circle -/
noncomputable def circle_area : ℝ := 5 * Real.pi

theorem circle_area_proof :
  ∃ (center_x center_y radius : ℝ),
    (∀ x y, circle_equation x y ↔ (x - center_x)^2 + (y - center_y)^2 = radius^2) ∧
    circle_area = Real.pi * radius^2 := by
  -- Provide the center coordinates and radius
  let center_x := 1
  let center_y := -2
  let radius := Real.sqrt 5
  
  -- Prove the existence of these values
  use center_x, center_y, radius
  
  constructor
  
  · -- Prove the equivalence of the circle equations
    intro x y
    sorry  -- The actual proof would go here
  
  · -- Prove that the area matches our definition
    simp [circle_area]
    sorry  -- The actual proof would go here

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_proof_l181_18118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_conversion_126kmh_l181_18199

/-- Converts kilometers per hour to meters per second -/
def kmh_to_ms (speed_kmh : ℝ) : ℝ :=
  speed_kmh * 0.277778

/-- Rounds a real number to the nearest integer -/
noncomputable def round_to_nearest (x : ℝ) : ℤ :=
  ⌊x + 0.5⌋

theorem speed_conversion_126kmh :
  round_to_nearest (kmh_to_ms 126) = 35 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_conversion_126kmh_l181_18199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_change_percentage_l181_18130

theorem price_change_percentage (initial_price : ℝ) (price_increase : ℝ) (discount1 : ℝ) (discount2 : ℝ) :
  let increased_price := initial_price * (1 + price_increase)
  let price_after_discount1 := increased_price * (1 - discount1)
  let final_price := price_after_discount1 * (1 - discount2)
  let percentage_change := (final_price - initial_price) / initial_price * 100
  price_increase = 0.32 ∧ discount1 = 0.10 ∧ discount2 = 0.15 →
  ∃ ε > 0, |percentage_change - 0.98| < ε :=
by
  intro h
  sorry

#check price_change_percentage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_change_percentage_l181_18130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_decreasing_interval_l181_18104

noncomputable def P : ℝ × ℝ := (1, -1)

noncomputable def φ : ℝ := -Real.pi/4

noncomputable def y (x : ℝ) : ℝ := 3 * Real.cos (x + φ)

theorem function_decreasing_interval (h1 : -Real.pi < φ ∧ φ < 0) 
  (h2 : P = (Real.cos φ, Real.sin φ)) :
  ∀ x₁ x₂, Real.pi/4 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ Real.pi → y x₁ > y x₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_decreasing_interval_l181_18104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overlap_length_for_paper_tape_ring_l181_18185

/-- Calculates the length of the overlapped part of tape when forming a ring --/
noncomputable def overlapLength (pieceLength : ℝ) (numPieces : ℕ) (ringCircumference : ℝ) : ℝ :=
  ((pieceLength * (numPieces : ℝ) - ringCircumference) / (numPieces : ℝ)) * 10

/-- Theorem stating the overlap length for the given problem --/
theorem overlap_length_for_paper_tape_ring :
  overlapLength 18 12 210 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_overlap_length_for_paper_tape_ring_l181_18185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_divisibility_l181_18178

theorem polynomial_divisibility (n : ℕ) :
  (∃ q : Polynomial ℤ, X^(2*n) + X^n + 1 = (X^2 + X + 1) * q) ↔ ¬(3 ∣ n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_divisibility_l181_18178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_n_max_min_points_l181_18126

-- Define the function f_1
def f_1 (x : ℝ) : ℝ := 4 * (x - x^2)

-- Define the recursive function f_n
def f_n : ℕ → ℝ → ℝ 
  | 0, x => f_1 x  -- Handle the zero case
  | 1, x => f_1 x
  | n + 1, x => f_n n (f_1 x)

-- Define a_n as the number of maximum points of f_n on [0,1]
noncomputable def a_n (n : ℕ) : ℕ := 
  if n = 0 then 1 else 2^(n-1)

-- Define b_n as the number of minimum points of f_n on [0,1]
noncomputable def b_n (n : ℕ) : ℕ := 
  if n = 0 then 2 else 2^(n-1) + 1

-- Theorem to prove
theorem f_n_max_min_points (n : ℕ) : 
  a_n n = 2^(n-1) ∧ b_n n = 2^(n-1) + 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_n_max_min_points_l181_18126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l181_18138

noncomputable section

def f (a b x : ℝ) : ℝ := (a + b * Real.log x) / (x - 1)

def g (k x : ℝ) : ℝ := k / x

theorem function_properties 
  (a b : ℝ) 
  (h1 : (deriv (f a b)) 2 = -(1/2) * Real.log 2) 
  (h2 : f a b 4 = (1 + 2 * Real.log 2) / 3) :
  (a = 1 ∧ b = 1) ∧
  (∀ x, x ∈ Set.Ioo 0 1 → (deriv (f a b)) x < 0) ∧
  (∀ x, x > 1 → (deriv (f a b)) x < 0) ∧
  (∀ k : ℕ+, k ≤ 3 ↔ 
    ∀ x₀ > 1, ∃ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < x₀ ∧ 
      f a b x₀ = f a b x₁ ∧ f a b x₀ = f a b x₂ ∧
      g ↑k x₀ = f a b x₀) :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l181_18138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existential_cosine_l181_18123

theorem negation_of_existential_cosine :
  (¬ ∃ x : ℝ, Real.cos x ≥ -1) ↔ (∀ x : ℝ, Real.cos x < -1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existential_cosine_l181_18123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_survey_chi_square_significant_l181_18162

/-- Represents a 2x2 contingency table for a survey on hometown visits during a festival --/
structure ContingencyTable :=
  (total : ℕ)
  (young_total : ℕ)
  (young_went_back : ℕ)
  (old_did_not_go : ℕ)

/-- Calculates the chi-square value for a given contingency table --/
noncomputable def calculate_chi_square (table : ContingencyTable) : ℝ :=
  let a := table.young_went_back
  let b := table.young_total - a
  let c := table.total - table.young_total - table.old_did_not_go
  let d := table.old_did_not_go
  let n := table.total
  (n * (a * d - b * c)^2 : ℝ) / ((a + b) * (c + d) * (a + c) * (b + d))

/-- Theorem stating that the chi-square value for the given survey data is greater than 10.828 --/
theorem survey_chi_square_significant (table : ContingencyTable)
  (h1 : table.total = 100)
  (h2 : table.young_total = 60)
  (h3 : table.young_went_back = 5)
  (h4 : table.old_did_not_go = 25) :
  calculate_chi_square table > 10.828 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_survey_chi_square_significant_l181_18162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_minimized_l181_18171

def population (a b : ℕ) : List ℕ := [2, 3, 3, 7, a, b, 12, 15, 18, 20]

def is_median (x : ℕ) (l : List ℕ) : Prop :=
  2 * (l.filter (· ≤ x)).length ≥ l.length ∧
  2 * (l.filter (· ≥ x)).length ≥ l.length

def variance (l : List ℕ) : ℚ :=
  let mean := (l.sum : ℚ) / l.length
  (l.map (fun x => ((x : ℚ) - mean) ^ 2)).sum / l.length

theorem variance_minimized (a b : ℕ) :
  is_median 10 (population a b) →
  (∀ (x y : ℕ), variance (population x y) ≥ variance (population 10 10)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_minimized_l181_18171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l181_18169

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- State the main theorem
theorem tangent_line_at_one (h : ∀ x : ℝ, f x = 2 * f (2 - x) - x^2 + 8*x - 8) :
  let tangent_line := λ x : ℝ ↦ 2*x - 1
  ∀ x : ℝ, f x + (tangent_line 1 - f 1) = tangent_line x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l181_18169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rocket_reaches_75km_and_exceeds_70km_l181_18141

/-- Represents the motion of a rocket launched vertically upwards -/
noncomputable def rocket_motion (a g : ℝ) (τ : ℝ) : ℝ → ℝ :=
  λ t => if t ≤ τ then
           (1/2) * a * t^2
         else
           (1/2) * a * τ^2 + a * τ * (t - τ) - (1/2) * g * (t - τ)^2

/-- The time at which the rocket reaches its maximum height -/
noncomputable def max_height_time (a g : ℝ) (τ : ℝ) : ℝ :=
  τ + a * τ / g

/-- The maximum height reached by the rocket -/
noncomputable def max_height (a g : ℝ) (τ : ℝ) : ℝ :=
  rocket_motion a g τ (max_height_time a g τ)

theorem rocket_reaches_75km_and_exceeds_70km 
  (a : ℝ) (g : ℝ) (τ : ℝ) 
  (ha : a = 20) (hg : g = 10) (hτ : τ = 50) :
  max_height a g τ = 75000 ∧ max_height a g τ > 70000 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rocket_reaches_75km_and_exceeds_70km_l181_18141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_hands_bisection_time_l181_18172

/-- The angle (in degrees) traveled by the second hand in x minutes -/
noncomputable def second_hand_angle (x : ℝ) : ℝ := 6 * x

/-- The angle (in degrees) traveled by the minute hand in x minutes -/
noncomputable def minute_hand_angle (x : ℝ) : ℝ := x / 2

/-- The angle (in degrees) traveled by the hour hand in x minutes -/
noncomputable def hour_hand_angle (x : ℝ) : ℝ := x / 24

theorem clock_hands_bisection_time :
  ∃ x : ℝ, x > 0 ∧ x < 60 ∧
  second_hand_angle x = (minute_hand_angle x + hour_hand_angle x) / 2 ∧
  x = 1440 / 1427 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_hands_bisection_time_l181_18172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonically_decreasing_condition_l181_18177

open Real

/-- The function f(x) = ln x - (1/2)ax² - x --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log x - (1/2) * a * x^2 - x

/-- The derivative of f(x) --/
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := 1/x - a*x - 1

theorem monotonically_decreasing_condition (a : ℝ) : 
  (∀ x > 0, f_derivative a x < 0) ↔ a > -1/4 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonically_decreasing_condition_l181_18177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_negative_two_equals_sqrt_three_l181_18107

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2^(-x) - 1 else Real.sqrt x

-- State the theorem
theorem f_composition_negative_two_equals_sqrt_three :
  f (f (-2)) = Real.sqrt 3 := by
  -- Evaluate f(-2)
  have h1 : f (-2) = 3 := by
    simp [f]
    norm_num
  
  -- Evaluate f(3)
  have h2 : f 3 = Real.sqrt 3 := by
    simp [f]
  
  -- Combine the steps
  calc f (f (-2))
    = f 3 := by rw [h1]
    _ = Real.sqrt 3 := by rw [h2]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_negative_two_equals_sqrt_three_l181_18107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_traces_diameter_l181_18110

/-- Represents a circle with a center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a point on a circle -/
structure PointOnCircle where
  circle : Circle
  angle : ℝ  -- Angle from the positive x-axis

/-- Represents the configuration of two circles where one rolls inside the other -/
structure RollingCircles where
  largeCircle : Circle
  smallCircle : Circle
  contactPoint : ℝ × ℝ
  rollAngle : ℝ

/-- The path traced by a point on the smaller circle as it rolls inside the larger circle -/
noncomputable def tracedPath (rc : RollingCircles) (p : PointOnCircle) : Set (ℝ × ℝ) :=
  sorry

/-- States that the radius of the smaller circle is half that of the larger circle -/
def radiusRelation (rc : RollingCircles) : Prop :=
  rc.smallCircle.radius = rc.largeCircle.radius / 2

/-- States that the smaller circle rolls without slipping inside the larger circle -/
def rollsWithoutSlipping (rc : RollingCircles) : Prop :=
  sorry

/-- Checks if a set of points forms a diameter of a circle -/
def IsDiameter (s : Set (ℝ × ℝ)) (c : Circle) : Prop :=
  sorry

/-- Theorem: The path traced by a point on the smaller circle is along the diameter of the larger circle -/
theorem point_traces_diameter
  (rc : RollingCircles)
  (p : PointOnCircle)
  (h1 : p.circle = rc.smallCircle)
  (h2 : radiusRelation rc)
  (h3 : rollsWithoutSlipping rc) :
  ∃ (d : Set (ℝ × ℝ)), d = tracedPath rc p ∧ IsDiameter d rc.largeCircle :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_traces_diameter_l181_18110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_l181_18192

-- Define the function f
noncomputable def f (a b c x : ℝ) : ℝ :=
  a * (Real.tan x)^2017 + b * x^2017 + c * Real.log (x + Real.sqrt (x^2 + 1)) + 20

-- State the theorem
theorem function_property (a b c : ℝ) :
  f a b c (Real.log (Real.log 21 / Real.log 2)) = 17 →
  f a b c (Real.log (Real.log 5 / Real.log 21)) = 23 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_l181_18192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l181_18159

open Real

-- Define the triangle ABC
def triangle_ABC (A B C : ℝ) : Prop :=
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π

-- Define the given conditions
def given_conditions (A B C : ℝ) : Prop :=
  triangle_ABC A B C ∧
  sin B = Real.sqrt 7 / 4 ∧
  cos A / sin A + cos C / sin C = 4 * Real.sqrt 7 / 7 ∧
  2 * (sin A * sin C * cos B) = 3 / 2

theorem triangle_properties (A B C : ℝ) (h : given_conditions A B C) :
  sin A * sin C = sin B * sin B ∧
  0 < B ∧ B ≤ π / 3 ∧
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    sin A / a = sin B / b ∧ sin B / b = sin C / c ∧
    (b * b + c * c - 2 * b * c * cos A) = 8 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l181_18159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_monotonicity_condition_l181_18147

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/x + a) * Real.log (1 + x)

-- Part 1: Tangent line equation
theorem tangent_line_equation (a : ℝ) (h : a = -1) :
  ∃ (m b : ℝ), m * 1 + b = f a 1 ∧
  ∀ x, m * x + b = (Real.log 2) * x + f a 1 - (Real.log 2) := by
  sorry

-- Part 2: Monotonicity condition
theorem monotonicity_condition (a : ℝ) :
  (∀ x > 0, Monotone (fun x => f a x)) ↔ a ≥ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_monotonicity_condition_l181_18147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_arrangement_swap_l181_18195

/-- Represents a circular arrangement of numbers from 1 to 2006 -/
def CircularArrangement := Fin 2006 → Fin 2006

/-- A swap operation on a circular arrangement -/
def swap (arr : CircularArrangement) (i j : Fin 2006) : CircularArrangement :=
  fun k => if k = i then arr j else if k = j then arr i else arr k

/-- Predicate to check if two positions are diametrically opposite -/
def isDiametricallyOpposite (i j : Fin 2006) : Prop :=
  (j.val - i.val) % 2006 = 1003

/-- Predicate to check if a sequence of swaps results in each number
    being diametrically opposite to its initial position -/
def resultsDiametricallyOpposite (initial final : CircularArrangement) : Prop :=
  ∀ i : Fin 2006, isDiametricallyOpposite i (final i)

/-- Predicate to check if two numbers sum to 2007 -/
def sumsTo2007 (i j : Fin 2006) : Prop :=
  i.val + j.val = 2007

theorem circular_arrangement_swap :
  ∀ (initial final : CircularArrangement),
  (∃ (swaps : List (Fin 2006 × Fin 2006)),
    (swaps.foldl (fun arr (i, j) => swap arr i j) initial = final) ∧
    (resultsDiametricallyOpposite initial final)) →
  ∃ (i j : Fin 2006), ∃ (swaps : List (Fin 2006 × Fin 2006)), (i, j) ∈ swaps ∧ sumsTo2007 i j :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_arrangement_swap_l181_18195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_statement_correct_statement_B_correct_statement_C_correct_statement_D_l181_18179

-- Define the types for plane and line
variable (α : Type) (m n : Type)

-- Define class instances for plane and line
class Plane (α : Type)
class Line (l : Type)

-- Define the parallel and perpendicular relations
def parallel (x y : Type) : Prop := sorry
def perpendicular (x y : Type) : Prop := sorry

-- The statement to be proven false
theorem incorrect_statement 
  [Plane α] [Line m] [Line n] :
  (parallel α m ∧ ¬parallel m n) → ¬(parallel α n) := by
  sorry

-- Other statements (correct ones) for reference
theorem correct_statement_B 
  [Plane α] [Line m] [Line n] :
  (parallel α m ∧ ¬perpendicular m n) → ¬(perpendicular α n) := by
  sorry

theorem correct_statement_C 
  [Plane α] [Line m] [Line n] :
  (perpendicular α m ∧ ¬parallel m n) → ¬(perpendicular α n) := by
  sorry

theorem correct_statement_D 
  [Plane α] [Line m] [Line n] :
  (perpendicular α m ∧ ¬perpendicular m n) → ¬(parallel α n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_statement_correct_statement_B_correct_statement_C_correct_statement_D_l181_18179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cinema_ticket_prices_l181_18157

theorem cinema_ticket_prices : 
  ∃ (n : ℕ), n = (Finset.filter (λ y : ℕ => 15 * y = 90 ∧ 20 * y = 120) (Finset.range 121)).card ∧ 
  n = 8 :=
by
  -- We'll use 'sorry' to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cinema_ticket_prices_l181_18157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_T_occurs_at_4_l181_18146

/-- Geometric sequence with common ratio √2 -/
noncomputable def a (n : ℕ) : ℝ := sorry

/-- Sum of first n terms of the geometric sequence -/
noncomputable def S (n : ℕ) : ℝ := sorry

/-- Definition of T_n -/
noncomputable def T (n : ℕ+) : ℝ := (17 * S n - S (2 * n)) / a (n + 1)

/-- T_m is the maximum term of the sequence {T_n} -/
def is_max_T (m : ℕ+) : Prop :=
  ∀ n : ℕ+, T n ≤ T m

/-- The maximum of T_n occurs at n = 4 -/
theorem max_T_occurs_at_4 :
  is_max_T 4 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_T_occurs_at_4_l181_18146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_chi_square_decreases_credibility_l181_18149

/- Represents the chi-square statistic -/
def chi_square : ℝ → ℝ := sorry

/- Represents the credibility of the relationship between variables -/
def relationship_credibility : ℝ → ℝ := sorry

/- The chi-square statistic measures the relationship between variables -/
axiom chi_square_measures_relationship :
  ∀ (x y : ℝ), chi_square x < chi_square y → relationship_credibility x < relationship_credibility y

/- Theorem: As the chi-square statistic decreases, the credibility of the relationship decreases -/
theorem decreasing_chi_square_decreases_credibility :
  ∀ (x y : ℝ), x < y → relationship_credibility (chi_square x) < relationship_credibility (chi_square y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_chi_square_decreases_credibility_l181_18149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_hypotenuse_l181_18168

/-- Given a right triangle with one leg of length 15 and the angle opposite that leg being 45°,
    the length of the hypotenuse is 15√2. -/
theorem right_triangle_hypotenuse (a b c : ℝ) 
  (right_triangle : a^2 + b^2 = c^2)
  (leg_length : a = 15)
  (opposite_angle : Real.cos (45 * π / 180) = b / c) :
  c = 15 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_hypotenuse_l181_18168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_workshop_equation_l181_18167

/-- Represents the daily processing capacity of Workshop A -/
noncomputable def workshop_a_capacity : ℝ → ℝ := λ x => x

/-- Represents the daily processing capacity of Workshop B -/
noncomputable def workshop_b_capacity : ℝ → ℝ := λ x => 1.5 * x

/-- Represents the time taken for Workshop A to process 4000 more pieces than Workshop B -/
noncomputable def time_difference (x : ℝ) : ℝ := 4000 / (workshop_a_capacity x) - 4200 / (workshop_b_capacity x)

/-- Theorem stating the equation for the workshop problem -/
theorem workshop_equation (x : ℝ) (h1 : x > 0) :
  time_difference x = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_workshop_equation_l181_18167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l181_18114

-- Define the parabola
structure Parabola where
  vertex : ℝ × ℝ
  focus : ℝ × ℝ

-- Define the standard form of a parabola equation
def standard_equation (p : Parabola) : (ℝ → ℝ → Prop) :=
  fun x y => y^2 = 4 * (p.focus.1 - p.vertex.1) * (x - p.vertex.1)

-- Theorem statement
theorem parabola_equation (p : Parabola) 
  (h1 : p.vertex = (0, 0)) 
  (h2 : p.focus = (2, 0)) : 
  standard_equation p = fun x y => y^2 = 8*x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l181_18114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_cube_volume_ratio_l181_18186

/-- A right circular cone containing a cube with specific properties -/
structure ConeWithCube where
  R : ℝ  -- radius of the base of the cone
  h : ℝ  -- height of the cone
  a : ℝ  -- edge length of the cube
  cube_edge_on_base : a ≤ 2 * R
  cube_vertices_on_surface : a * Real.sqrt 2 ≤ h
  cube_center_on_height : h = (a * R * Real.sqrt 2 + 2 * R * a) / (2 * R)

/-- The ratio of the volume of the cone to the volume of the cube -/
noncomputable def volume_ratio (c : ConeWithCube) : ℝ :=
  (Real.pi * (53 - 7 * Real.sqrt 3) * Real.sqrt 2) / 48

/-- Theorem stating the volume ratio for a cone with a cube satisfying specific conditions -/
theorem cone_cube_volume_ratio (c : ConeWithCube) :
  (1/3 * Real.pi * c.R^2 * c.h) / c.a^3 = volume_ratio c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_cube_volume_ratio_l181_18186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_amusing_odd_l181_18198

/-- The number of divisors of a positive integer -/
def num_divisors (k : ℕ) : ℕ := 
  (Finset.filter (· ∣ k) (Finset.range k.succ)).card

/-- The digit sum of a positive integer -/
def digit_sum : ℕ → ℕ
  | 0 => 0
  | n + 1 => (n + 1) % 10 + digit_sum (n / 10)

/-- A positive integer is amusing if there exists a k such that
    its number of divisors and digit sum are equal to the integer itself -/
def is_amusing (n : ℕ) : Prop :=
  ∃ k : ℕ, k > 0 ∧ num_divisors k = n ∧ digit_sum k = n

theorem smallest_amusing_odd : 
  (∀ n : ℕ, 0 < n → n < 9 → n % 2 = 1 → ¬is_amusing n) ∧ is_amusing 9 := by
  sorry

#eval num_divisors 36
#eval digit_sum 36

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_amusing_odd_l181_18198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_properties_l181_18129

-- Define the pyramid structure
structure Pyramid where
  base_side_length : ℝ
  base_acute_angle : ℝ
  sphere_radius : ℝ

-- Define the given pyramid
noncomputable def given_pyramid : Pyramid :=
  { base_side_length := 2
  , base_acute_angle := Real.pi/4  -- 45° in radians
  , sphere_radius := Real.sqrt 2 }

-- Theorem statement
theorem pyramid_properties (p : Pyramid) (h : p = given_pyramid) :
  (∃ (height : ℝ → ℝ → ℝ → ℝ), height p.base_side_length p.base_acute_angle p.sphere_radius = 
    (Real.sqrt 6) / 3 ∧ 
    (∀ (x y : ℝ), height x y p.sphere_radius = (Real.sqrt 6) / 3 → 
      x = p.base_side_length ∧ y = p.base_acute_angle)) ∧
  (∃ (volume : ℝ → ℝ → ℝ → ℝ), volume p.base_side_length p.base_acute_angle p.sphere_radius = 
    (2 * Real.sqrt 3) / 9) := by
  sorry

#check pyramid_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_properties_l181_18129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_divisibility_l181_18135

theorem root_divisibility (f : Polynomial ℤ) (p q k : ℤ) :
  (q ≠ 0) →
  (Nat.gcd p.natAbs q.natAbs = 1) →
  (f.eval (↑p / ↑q) = 0) →
  (p - k * q) ∣ f.eval ↑k :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_divisibility_l181_18135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_adjacent_pair_arrangement_l181_18116

/-- Given 6 distinct objects where two specific objects must be adjacent and in a fixed order,
    the number of possible arrangements is equal to 5! -/
theorem adjacent_pair_arrangement (n : ℕ) (h : n = 6) :
  Nat.factorial (n - 1) = 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_adjacent_pair_arrangement_l181_18116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_pizza_pieces_four_cuts_l181_18137

/-- Represents the number of pieces a square pizza is cut into after n cuts -/
def pizzaPieces (n : ℕ) : ℕ := sorry

/-- The maximum number of pieces a square pizza can be cut into with n linear cuts -/
def maxPizzaPieces (n : ℕ) : ℕ := pizzaPieces n

/-- Theorem: The maximum number of pieces a square pizza can be cut into with 4 linear cuts is 14 -/
theorem max_pizza_pieces_four_cuts : maxPizzaPieces 4 = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_pizza_pieces_four_cuts_l181_18137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_puzzle_solution_l181_18181

/-- Represents the dimensions of the rectangular prism -/
structure PrismDimensions where
  n : ℕ
  h_n_gt_2 : n > 2

/-- Counts the number of unpainted unit cubes in the prism -/
def count_unpainted (d : PrismDimensions) : ℕ :=
  (d.n - 2) * (d.n - 1) * d.n

/-- Counts the number of unit cubes with exactly one face painted -/
def count_one_face_painted (d : PrismDimensions) : ℕ :=
  2 * ((d.n - 2) * (d.n - 1) + (d.n - 2) * d.n + (d.n - 1) * d.n)

/-- Theorem stating that when the counts are equal, n must be 7 -/
theorem prism_puzzle_solution (d : PrismDimensions) :
  count_one_face_painted d = count_unpainted d → d.n = 7 := by
  sorry

/-- Example demonstrating the solution -/
def solution_example : PrismDimensions := {
  n := 7,
  h_n_gt_2 := by norm_num
}

#eval count_unpainted solution_example
#eval count_one_face_painted solution_example

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_puzzle_solution_l181_18181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_min_value_l181_18101

noncomputable def g (x : ℝ) : ℝ := (5 * x^2 - 10 * x + 24) / (7 * (1 + 2 * x))

theorem g_min_value (x : ℝ) (hx : x ≥ 1) : g x ≥ 29 / 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_min_value_l181_18101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithm_properties_l181_18148

-- Define the logarithm function
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem logarithm_properties :
  (f (10 ^ (-2 : ℝ)) = -2) ∧
  (∀ b : ℝ, b > 0 → f (b^3) / f b = 2) ∧
  (∀ a b c : ℝ, f 3 = a → f 7 = b → f 0.63 = c → 2*a + b - c = 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithm_properties_l181_18148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_distance_l181_18106

/-- Given two parallel lines in the form ax + by + c = 0, calculate the distance between them -/
noncomputable def distance_between_parallel_lines (a b c₁ c₂ : ℝ) : ℝ :=
  |c₁ - c₂| / Real.sqrt (a^2 + b^2)

/-- The value of m that makes the lines parallel -/
def m : ℝ := 4

theorem parallel_lines_distance :
  let line1 : ℝ → ℝ → ℝ := λ x y => 3*x + 2*y - 3
  let line2 : ℝ → ℝ → ℝ := λ x y => 6*x + m*y + 1
  (∀ x y, line1 x y = 0 → line2 x y ≠ 0) →  -- Lines are distinct
  (∀ x₁ y₁ x₂ y₂, line1 x₁ y₁ = 0 → line1 x₂ y₂ = 0 → (x₁ - x₂) * 2 = (y₂ - y₁) * 3) →  -- Line1 has slope -3/2
  (∀ x₁ y₁ x₂ y₂, line2 x₁ y₁ = 0 → line2 x₂ y₂ = 0 → (x₁ - x₂) * m = (y₂ - y₁) * 6) →  -- Line2 has slope -6/m
  distance_between_parallel_lines 6 4 1 (-6) = 7 * Real.sqrt 13 / 26 := by
  sorry

#check parallel_lines_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_distance_l181_18106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_odd_integers_with_even_product_l181_18113

theorem max_odd_integers_with_even_product (integers : Finset ℕ) : 
  integers.card = 5 → 
  Even (integers.prod id) → 
  (integers.filter (fun x => ¬Even x)).card ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_odd_integers_with_even_product_l181_18113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l181_18182

open Real

theorem function_inequality (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x > 0, f x ≥ 0) →
  (∀ x > 0, DifferentiableAt ℝ f x) →
  (∀ x > 0, x * deriv f x + f x ≤ 0) →
  0 < a →
  a < b →
  a * f b ≤ b * f a :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l181_18182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_sum_l181_18190

/-- The parabola 4y^2 = x with focus F, and points A and B on the parabola on opposite sides of the x-axis satisfying OA · OB = 15 -/
structure Parabola where
  F : ℝ × ℝ
  A : ℝ × ℝ
  B : ℝ × ℝ
  hF : F.1 = 1/16 ∧ F.2 = 0
  hA : 4 * A.2^2 = A.1 ∧ A.2 > 0
  hB : 4 * B.2^2 = B.1 ∧ B.2 < 0
  hDot : A.1 * B.1 + A.2 * B.2 = 15

/-- The sum of the areas of triangle ABO and triangle AFO -/
noncomputable def areaSum (p : Parabola) : ℝ :=
  1/2 * abs (p.A.2 - p.B.2) + 1/2 * 1/16 * p.A.2

/-- The minimum value of the sum of the areas is √65/2 -/
theorem min_area_sum (p : Parabola) : 
  ∃ (min : ℝ), min = Real.sqrt 65 / 2 ∧ ∀ (q : Parabola), areaSum q ≥ min := by
  sorry

#check min_area_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_sum_l181_18190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_prime_not_dividing_power_minus_prime_l181_18170

theorem exists_prime_not_dividing_power_minus_prime (p : ℕ) (hp : Prime p) :
  ∃ q : ℕ, Prime q ∧ ∀ n : ℕ, ¬(q ∣ n^p - p) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_prime_not_dividing_power_minus_prime_l181_18170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l181_18120

-- Define a, b, and c as in the problem
noncomputable def a : ℝ := (1/2)^(1/3)
noncomputable def b : ℝ := Real.log 2 / Real.log (1/3)
noncomputable def c : ℝ := Real.log 3 / Real.log (1/2)

-- State the theorem
theorem relationship_abc : a > b ∧ b > c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l181_18120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_chord_length_l181_18134

/-- The length of the chord formed by the intersection of a line and a circle -/
noncomputable def chord_length (a b c : ℝ) (x₀ y₀ r : ℝ) : ℝ :=
  2 * Real.sqrt (r^2 - (abs (a * x₀ + b * y₀ + c) / Real.sqrt (a^2 + b^2))^2)

/-- Theorem stating that the length of the chord formed by the intersection of
    the line x - y + 3 = 0 and the circle (x + 2)² + (y - 2)² = 2 is √6 -/
theorem intersection_chord_length :
  chord_length 1 (-1) 3 (-2) 2 (Real.sqrt 2) = Real.sqrt 6 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_chord_length_l181_18134
