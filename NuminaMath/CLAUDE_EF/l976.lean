import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l976_97647

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + (3 + a) * x + 3

-- Define the function g
def g (a k : ℝ) (x : ℝ) : ℝ := f a x - k * x

theorem problem_solution :
  ∀ a : ℝ, a ≠ 0 →
  (∀ x : ℝ, f a 2 = 3 → f a x = -x^2 + 2*x + 3) ∧
  (∀ k : ℝ, (∀ x : ℝ, x ∈ Set.Icc (-2) 2 → Monotone (g a k) ∨ StrictMonoOn (g a k) (Set.Icc (-2) 2)) →
    k ∈ Set.Iic (-2) ∪ Set.Ici 6) ∧
  (∃ x : ℝ, x ∈ Set.Icc (-1) 4 ∧ f a x = 4 ∧ (∀ y : ℝ, y ∈ Set.Icc (-1) 4 → f a y ≤ 4) ↔ a = -1 ∨ a = -9) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l976_97647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_on_line_l976_97654

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- The line represented by parametric equations x = 2 + 3t, y = 2 + t -/
def parametricLine (t : ℝ) : Point2D :=
  { x := 2 + 3 * t,
    y := 2 + t }

/-- Distance between two points in 2D space -/
noncomputable def distance (p1 p2 : Point2D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

theorem distance_between_points_on_line :
  distance (parametricLine 0) (parametricLine 1) = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_on_line_l976_97654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_PQ_is_two_l976_97636

noncomputable section

/-- Curve C in parametric form -/
def curve_C (θ : Real) : Real × Real :=
  (1 + Real.sqrt 7 * Real.cos θ, Real.sqrt 7 * Real.sin θ)

/-- Line l₁ in polar form -/
def line_l₁ (ρ θ : Real) : Prop :=
  2 * ρ * Real.sin (θ + Real.pi / 3) - Real.sqrt 3 = 0

/-- Ray l₂ in polar form -/
def ray_l₂ (ρ θ : Real) : Prop :=
  θ = Real.pi / 3 ∧ ρ > 0

/-- Point P: intersection of curve C and ray l₂ -/
def point_P : Real × Real :=
  (3, Real.pi / 3)

/-- Point Q: intersection of line l₁ and ray l₂ -/
def point_Q : Real × Real :=
  (1, Real.pi / 3)

/-- Distance between two points in polar coordinates -/
def polar_distance (p₁ p₂ : Real × Real) : Real :=
  |p₁.1 - p₂.1|

theorem distance_PQ_is_two :
  polar_distance point_P point_Q = 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_PQ_is_two_l976_97636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_and_increasing_l976_97639

-- Define the function f(x) = -2ln|x|
noncomputable def f (x : ℝ) : ℝ := -2 * Real.log (abs x)

-- State the theorem
theorem f_even_and_increasing :
  (∀ x : ℝ, f (-x) = f x) ∧ 
  (∀ x y : ℝ, x < y ∧ y < 0 → f x < f y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_and_increasing_l976_97639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_triangular_pyramid_volume_l976_97694

/-- The volume of a regular triangular pyramid, given the distances from its height midpoint to a lateral face and a lateral edge -/
theorem regular_triangular_pyramid_volume 
  (d_face : ℝ) 
  (d_edge : ℝ) 
  (h_face : d_face = 2) 
  (h_edge : d_edge = Real.sqrt 6) : 
  ∃ (v : ℝ), abs (v - 33.4) < 0.1 ∧ 
  (∃ (a h : ℝ), 
    v = (1/3) * ((Real.sqrt 3 / 4) * a^2) * h ∧
    h^2 / 4 + d_edge^2 = d_face^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_triangular_pyramid_volume_l976_97694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_floor_cube_root_29_l976_97638

theorem cube_floor_cube_root_29 : ⌊(29 : ℝ)^(1/3 : ℝ)⌋^3 = 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_floor_cube_root_29_l976_97638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_sequence_2006th_is_rhombus_l976_97603

/-- A sequence of quadrilaterals where each subsequent quadrilateral is formed by
    connecting the midpoints of the sides of the previous quadrilateral -/
def QuadrilateralSequence (ABCD : Quadrilateral) : ℕ → Quadrilateral :=
  sorry

/-- The property that AC is perpendicular to BD in a quadrilateral ABCD -/
def HasPerpendicularDiagonals (ABCD : Quadrilateral) : Prop :=
  sorry

/-- The property of being a rhombus -/
def IsRhombus (Q : Quadrilateral) : Prop :=
  sorry

theorem quadrilateral_sequence_2006th_is_rhombus 
  (ABCD : Quadrilateral) 
  (h : HasPerpendicularDiagonals ABCD) : 
  IsRhombus (QuadrilateralSequence ABCD 2006) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_sequence_2006th_is_rhombus_l976_97603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l976_97601

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * (Real.sin x)^2 + Real.sin x * Real.cos x

-- State the theorem
theorem f_properties :
  -- The smallest positive period of f is π
  (∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
    (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T')) ∧
  -- For a triangle ABC with given properties, its area is (3√3)/4
  (∀ (A B C a b c : ℝ),
    f (A / 2) = Real.sqrt 3 / 2 →
    a = 4 →
    b + c = 5 →
    -- Assuming A, B, C form a valid triangle
    0 < A ∧ A < Real.pi →
    0 < B ∧ B < Real.pi →
    0 < C ∧ C < Real.pi →
    A + B + C = Real.pi →
    -- a, b, c are opposite sides to A, B, C
    a = Real.sqrt (b^2 + c^2 - 2*b*c*Real.cos A) →
    b = Real.sqrt (a^2 + c^2 - 2*a*c*Real.cos B) →
    c = Real.sqrt (a^2 + b^2 - 2*a*b*Real.cos C) →
    -- The area of the triangle is (3√3)/4
    (1/2) * a * b * Real.sin C = (3 * Real.sqrt 3) / 4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l976_97601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l976_97689

noncomputable def g (x : ℝ) : ℝ := (7^x - 1) / (7^x + 1)

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x := by
  intro x
  simp [g]
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l976_97689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l976_97617

noncomputable def a : ℝ := ∫ (x : ℝ) in Set.Icc (Real.exp 1) (Real.exp 2), 1 / x

theorem constant_term_expansion (x : ℝ) : 
  (Finset.filter (fun k => 2 * k = 6 ∧ (Nat.choose 6 k : ℝ) * (a ^ (6 - k)) * ((-1 / x) ^ k) = (a * x^2 - 1/x)^6) (Finset.range 7)).card = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l976_97617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_line_properties_l976_97608

/-- Represents a point on a scatter plot -/
structure ScatterPoint where
  x : ℝ
  y : ℝ

/-- Represents a linear regression model -/
structure LinearModel where
  slope : ℝ
  intercept : ℝ

/-- Checks if a point lies on a given line -/
def pointOnLine (p : ScatterPoint) (m : LinearModel) : Prop :=
  p.y = m.slope * p.x + m.intercept

/-- Calculates the residual for a point given a model -/
def residualCalc (p : ScatterPoint) (m : LinearModel) : ℝ :=
  p.y - (m.slope * p.x + m.intercept)

/-- Calculates the sum of squared residuals -/
def sumSquaredResiduals (points : List ScatterPoint) (m : LinearModel) : ℝ :=
  (points.map (λ p => (residualCalc p m)^2)).sum

/-- Calculates the correlation coefficient -/
noncomputable def correlationCoefficient (points : List ScatterPoint) : ℝ :=
  sorry  -- Definition of correlation coefficient

theorem perfect_line_properties 
  (points : List ScatterPoint) 
  (m : LinearModel) 
  (h : ∀ p ∈ points, pointOnLine p m) : 
  (∀ p ∈ points, residualCalc p m = 0) ∧ 
  sumSquaredResiduals points m = 0 ∧ 
  correlationCoefficient points = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_line_properties_l976_97608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_on_AB_l976_97641

-- Define the necessary structures
structure Point := (x y : ℝ)
structure Line := (a b c : ℝ)

-- Define the given points and lines
variable (A B D E F G H I : Point)
variable (AB FI GH : Line)

-- Define auxiliary functions
def on_line (P : Point) (L : Line) : Prop := sorry
def is_equilateral_triangle (P Q R : Point) : Prop := sorry
def same_side_of_line (L : Line) (P Q R S : Point) : Prop := sorry
def parallel (L1 L2 : Line) : Prop := sorry

-- Define the conditions
axiom D_on_AB : on_line D AB
axiom E_on_AB : on_line E AB
axiom triangle_ADF_equilateral : is_equilateral_triangle A D F
axiom triangle_DBG_equilateral : is_equilateral_triangle D B G
axiom triangle_AEH_equilateral : is_equilateral_triangle A E H
axiom triangle_EBI_equilateral : is_equilateral_triangle E B I
axiom same_halfplane : same_side_of_line AB F G H I
axiom FI_GH_not_parallel : ¬ parallel FI GH

-- Define the theorem
theorem intersection_on_AB :
  ∃ P : Point, on_line P AB ∧ on_line P FI ∧ on_line P GH :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_on_AB_l976_97641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_shape_l976_97610

-- Define a triangle
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the condition
def satisfies_condition (t : Triangle) : Prop :=
  t.c - t.a * Real.cos t.B = (2 * t.a - t.b) * Real.cos t.A

-- Define isosceles triangle
def is_isosceles (t : Triangle) : Prop :=
  t.A = t.B ∨ t.B = t.C ∨ t.C = t.A

-- Define right-angled triangle
def is_right_angled (t : Triangle) : Prop :=
  t.A = Real.pi / 2 ∨ t.B = Real.pi / 2 ∨ t.C = Real.pi / 2

-- Theorem statement
theorem triangle_shape (t : Triangle) :
  satisfies_condition t → is_isosceles t ∨ is_right_angled t :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_shape_l976_97610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_difference_is_twenty_percent_l976_97629

/-- Represents the heights and reaches in the problem -/
structure Heights where
  barry_reach : ℝ
  larry_height : ℝ
  combined_reach : ℝ

/-- Calculates the percentage difference between Larry's full height and shoulder height -/
noncomputable def percentage_difference (h : Heights) : ℝ :=
  let larry_shoulder_height := h.combined_reach - h.barry_reach
  let height_difference := h.larry_height - larry_shoulder_height
  (height_difference / h.larry_height) * 100

/-- Theorem stating that the percentage difference is 20% given the problem conditions -/
theorem percentage_difference_is_twenty_percent (h : Heights) 
  (h_barry_reach : h.barry_reach = 5)
  (h_larry_height : h.larry_height = 5)
  (h_combined_reach : h.combined_reach = 9) :
  percentage_difference h = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_difference_is_twenty_percent_l976_97629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_sum_l976_97635

theorem solution_sum (a b c d : ℕ+) :
  (∃ x y : ℝ, x + y = 6 ∧ 4 * x * y = 6 ∧
   ((x = (a : ℝ) + (b : ℝ) * Real.sqrt (c : ℝ) / (d : ℝ)) ∨
    (x = (a : ℝ) - (b : ℝ) * Real.sqrt (c : ℝ) / (d : ℝ))) ∧
   (∀ a' b' c' d' : ℕ+,
     ((x = (a' : ℝ) + (b' : ℝ) * Real.sqrt (c' : ℝ) / (d' : ℝ)) ∨
      (x = (a' : ℝ) - (b' : ℝ) * Real.sqrt (c' : ℝ) / (d' : ℝ))) →
     a' + b' + c' + d' ≥ a + b + c + d)) →
  a + b + c + d = 31 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_sum_l976_97635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_calculation_l976_97632

noncomputable section

-- Define the total savings
def total_savings : ℝ := 2750

-- Define the interest rate (calculated from simple interest)
def interest_rate : ℝ := 550 / (1375 * 2)

-- Define the time period
def years : ℕ := 2

-- Define the simple interest received
def simple_interest : ℝ := 550

-- Function to calculate compound interest
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time - principal

-- Theorem statement
theorem compound_interest_calculation :
  compound_interest (total_savings / 2) interest_rate years = 605 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_calculation_l976_97632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_white_circle_area_is_72_l976_97613

/-- Represents the properties of the painted cube -/
structure PaintedCube where
  edge_length : ℝ
  green_paint_area : ℝ
  num_faces : ℕ

/-- Calculates the area of one white circle on a face of the painted cube -/
noncomputable def white_circle_area (cube : PaintedCube) : ℝ :=
  let total_surface_area := cube.num_faces * (cube.edge_length ^ 2)
  let green_area_per_face := cube.green_paint_area / cube.num_faces
  let white_area_per_face := (cube.edge_length ^ 2) - green_area_per_face
  white_area_per_face

/-- Theorem stating that the area of one white circle is 72 square feet -/
theorem white_circle_area_is_72 (cube : PaintedCube) 
  (h1 : cube.edge_length = 12)
  (h2 : cube.green_paint_area = 432)
  (h3 : cube.num_faces = 6) :
  white_circle_area cube = 72 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_white_circle_area_is_72_l976_97613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_x_equals_one_l976_97696

-- Define the curve
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

-- Define the point of tangency
def x₀ : ℝ := 1

-- State the theorem
theorem tangent_line_at_x_equals_one :
  ∃ (m b : ℝ), ∀ x : ℝ, 
    (∀ ε > 0, ∃ δ > 0, ∀ h : ℝ, |h| < δ → |f (x₀ + h) - f x₀ - m * h| ≤ ε * |h|) ∧
    (m * x₀ + b = f x₀) ∧
    (m = 1 ∧ b = -1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_x_equals_one_l976_97696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_transformation_l976_97686

/-- The matrix transformation that we're considering -/
def matrix_transform (a b : ℝ) (x y : ℝ) : ℝ × ℝ := (x + a*y, b*x + y)

/-- The original curve -/
def original_curve (x y : ℝ) : Prop := x^2 + 4*x*y + 2*y^2 = 1

/-- The transformed curve -/
def transformed_curve (x y : ℝ) : Prop := x^2 - 2*y^2 = 1

/-- The main theorem -/
theorem curve_transformation (a b : ℝ) : 
  (∀ x y : ℝ, original_curve x y → 
    let (x', y') := matrix_transform a b x y;
    transformed_curve x' y') → 
  a + b = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_transformation_l976_97686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_sum_eleven_l976_97666

def die_faces : Finset ℕ := {1, 2, 3, 4, 5, 6}

def total_outcomes : ℕ := die_faces.card * die_faces.card

def favorable_outcomes : ℕ := (die_faces.filter (λ x => (die_faces.filter (λ y => x + y = 11)).card > 0)).card

theorem probability_sum_eleven (p : ℚ) : 
  p = (favorable_outcomes : ℚ) / (total_outcomes : ℚ) → p = 1 / 18 := by
  sorry

#eval favorable_outcomes
#eval total_outcomes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_sum_eleven_l976_97666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l976_97680

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  a = 2 →
  Real.sin (B / 2) = Real.sqrt 5 / 5 →
  (1 / 2) * a * b * Real.sin C = 4 →
  Real.cos B = 3 / 5 ∧ b = Real.sqrt 17 ∧ c = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l976_97680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_of_A_is_40_sqrt_3_l976_97606

/-- The speed of person A in kilometers per hour -/
noncomputable def speed_A : ℝ := 40 * Real.sqrt 3

/-- The speed of person B in kilometers per hour -/
def speed_B : ℝ := 40

/-- The time (in hours) from start to reaching T junction for A -/
def time_to_T_A : ℝ := 2

/-- The time (in hours) from T junction to end for both A and B -/
def time_after_T : ℝ := 2

/-- The final distance between A and B in kilometers -/
def final_distance : ℝ := 160

theorem speed_of_A_is_40_sqrt_3 :
  let distance_A_after_T := speed_A * time_after_T
  let distance_B_after_T := speed_B * time_after_T
  distance_A_after_T ^ 2 + distance_B_after_T ^ 2 = final_distance ^ 2 := by
  sorry

#eval speed_B

end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_of_A_is_40_sqrt_3_l976_97606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_floor_eq_two_cos_squared_l976_97672

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

-- State the theorem
theorem tan_floor_eq_two_cos_squared (x : ℝ) : 
  floor (Real.tan x) = ⌊2 * (Real.cos x)^2⌋ ↔ ∃ k : ℤ, x = k * Real.pi + Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_floor_eq_two_cos_squared_l976_97672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_example_l976_97663

/-- The area of a trapezium given the lengths of its parallel sides and the distance between them. -/
noncomputable def trapeziumArea (a b h : ℝ) : ℝ := (1/2) * (a + b) * h

/-- Theorem: The area of a trapezium with parallel sides of lengths 20 cm and 18 cm, 
    and a distance of 16 cm between them, is equal to 304 square centimeters. -/
theorem trapezium_area_example : trapeziumArea 20 18 16 = 304 := by
  -- Unfold the definition of trapeziumArea
  unfold trapeziumArea
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_example_l976_97663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_b_more_stable_l976_97642

-- Define the Class structure
structure SchoolClass where
  students : ℕ
  average_score : ℝ
  variance : ℝ

-- Define the more_stable relation
def more_stable (a b : SchoolClass) : Prop :=
  a.variance < b.variance

-- Theorem stating that Class B is more stable than Class A
theorem class_b_more_stable (class_a class_b : SchoolClass) 
  (h1 : class_a.students = class_b.students)
  (h2 : class_a.average_score = 85)
  (h3 : class_b.average_score = 85)
  (h4 : class_a.variance = 120)
  (h5 : class_b.variance = 90) :
  more_stable class_b class_a :=
by
  -- Unfold the definition of more_stable
  unfold more_stable
  -- Use the given hypotheses
  rw [h4, h5]
  -- Check that 90 < 120
  exact lt_of_lt_of_le (by norm_num) (le_refl 120)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_b_more_stable_l976_97642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_economical_driving_problem_l976_97649

/-- Represents the problem of finding the most economical driving speed and total cost --/
theorem economical_driving_problem (distance : ℝ) (min_speed max_speed : ℝ) (gas_price : ℝ) (wage : ℝ) :
  distance = 120 →
  min_speed = 50 →
  max_speed = 100 →
  gas_price = 6 →
  wage = 36 →
  ∃ (optimal_speed : ℝ) (total_cost : ℝ),
    optimal_speed ∈ Set.Icc min_speed max_speed ∧
    (∀ x ∈ Set.Icc min_speed max_speed,
      (wage + gas_price * (4 + x^2 / 360)) * (distance / x) ≥ total_cost) ∧
    optimal_speed = 60 ∧
    total_cost = 240 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_economical_driving_problem_l976_97649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eleven_not_expressible_l976_97620

def expressible (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ 
    n * (2^c - 2^d) = 2^a - 2^b

theorem eleven_not_expressible :
  ¬ expressible 11 ∧ ∀ m : ℕ, m > 0 ∧ m < 11 → expressible m :=
sorry

#check eleven_not_expressible

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eleven_not_expressible_l976_97620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l976_97626

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 1/2
  | n+1 => sequence_a n + (1 / (n+1)^2) * (sequence_a n)^2

theorem sequence_properties :
  ∀ n : ℕ, n > 0 →
    (1 / sequence_a (n-1) - 1 / sequence_a n < 1 / n^2) ∧
    (sequence_a n < n) ∧
    (1 / sequence_a n ≤ 5/6 + 1 / (n+1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l976_97626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_arrangements_count_l976_97623

/-- Represents the four types of crops -/
inductive Crop
  | Lettuce
  | Carrots
  | Tomatoes
  | Radishes

/-- Represents a position in the 2x2 grid -/
inductive Position
  | TopLeft
  | TopRight
  | BottomLeft
  | BottomRight

/-- Represents a planting arrangement -/
def Arrangement := Position → Crop

/-- Checks if two positions are adjacent -/
def are_adjacent (p1 p2 : Position) : Prop :=
  match p1, p2 with
  | Position.TopLeft, Position.TopRight => true
  | Position.TopLeft, Position.BottomLeft => true
  | Position.TopRight, Position.TopLeft => true
  | Position.TopRight, Position.BottomRight => true
  | Position.BottomLeft, Position.TopLeft => true
  | Position.BottomLeft, Position.BottomRight => true
  | Position.BottomRight, Position.TopRight => true
  | Position.BottomRight, Position.BottomLeft => true
  | _, _ => false

/-- Checks if an arrangement is valid -/
def is_valid_arrangement (arr : Arrangement) : Prop :=
  (∀ p1 p2, are_adjacent p1 p2 →
    ¬(arr p1 = Crop.Lettuce ∧ arr p2 = Crop.Carrots) ∧
    ¬(arr p1 = Crop.Carrots ∧ arr p2 = Crop.Lettuce)) ∧
  (∀ p1 p2, are_adjacent p1 p2 →
    ¬(arr p1 = Crop.Tomatoes ∧ arr p2 = Crop.Radishes) ∧
    ¬(arr p1 = Crop.Radishes ∧ arr p2 = Crop.Tomatoes)) ∧
  ((arr Position.TopLeft = arr Position.BottomRight) ∨
   (arr Position.TopRight = arr Position.BottomLeft))

/-- The main theorem stating that there are exactly 8 valid arrangements -/
theorem valid_arrangements_count :
  ∃ (arrangements : Finset Arrangement),
    (∀ arr ∈ arrangements, is_valid_arrangement arr) ∧
    (∀ arr, is_valid_arrangement arr → arr ∈ arrangements) ∧
    arrangements.card = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_arrangements_count_l976_97623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_line_l976_97679

open Real

/-- The slope of the tangent line to y = e^x at x = 1 -/
noncomputable def tangent_slope : ℝ := exp 1

/-- The slope of the line 2x + my + 1 = 0 -/
noncomputable def line_slope (m : ℝ) : ℝ := -2 / m

/-- The condition for perpendicular lines -/
def perpendicular (m : ℝ) : Prop := tangent_slope * line_slope m = -1

theorem tangent_perpendicular_line (m : ℝ) :
  perpendicular m → m = 2 * exp 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_line_l976_97679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_perimeter_in_square_configuration_l976_97644

/-- Given a configuration of three identical squares and a rectangle forming a larger square,
    this theorem proves that the perimeter of the rectangle is 8 times the side length of a small square. -/
theorem rectangle_perimeter_in_square_configuration (x : ℝ) :
  x > 0 →
  ∃ (large_square small_square rectangle : Set (ℝ × ℝ)),
    -- The large square is formed by three small squares and the rectangle
    (∃ (s1 s2 s3 : Set (ℝ × ℝ)), 
      small_square = s1 ∧ small_square = s2 ∧ small_square = s3 ∧
      large_square = s1 ∪ s2 ∪ s3 ∪ rectangle) →
    -- Each small square has side length x
    (∀ p ∈ small_square, p.1 ≤ x ∧ p.2 ≤ x) →
    -- The rectangle's width equals the side of the small square
    (∃ w h : ℝ, w = x ∧ h = 3*x ∧ 
      ∀ p ∈ rectangle, p.1 ≤ w ∧ p.2 ≤ h) →
    -- The perimeter of the rectangle is 8x
    2 * (x + 3*x) = 8*x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_perimeter_in_square_configuration_l976_97644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_ranking_l976_97609

-- Define the set of students
inductive Student : Type
  | Hannah : Student
  | Cassie : Student
  | Bridget : Student
  | David : Student

-- Define a function to represent the test scores
def score : Student → ℕ := sorry

-- Define the conditions
axiom hannah_shows_all : ∀ s : Student, s ≠ Student.Hannah → score Student.Hannah < score s
axiom david_shows_bridget : score Student.Bridget < score Student.David
axiom cassie_not_lowest : ∃ s : Student, s ≠ Student.Cassie ∧ score s < score Student.Cassie
axiom bridget_not_highest : ∃ s : Student, s ≠ Student.Bridget ∧ score Student.Bridget < score s

-- The theorem to prove
theorem correct_ranking :
  score Student.Hannah < score Student.Cassie ∧
  score Student.Cassie < score Student.Bridget ∧
  score Student.Bridget < score Student.David := by
  sorry

#check correct_ranking

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_ranking_l976_97609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l976_97699

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def triangleConditions (t : Triangle) : Prop :=
  t.c = 4 ∧ t.C = Real.pi/3

-- Theorem for part 1
theorem part1 (t : Triangle) (h : triangleConditions t) :
  (1/2 * t.a * t.b * Real.sin t.C = 4 * Real.sqrt 3) → t.a = 4 ∧ t.b = 4 := by
  sorry

-- Theorem for part 2
theorem part2 (t : Triangle) (h : triangleConditions t) :
  Real.sin t.B = 2 * Real.sin t.A →
  1/2 * t.a * t.b * Real.sin t.C = 8 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l976_97699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_A_equals_negative_eleven_l976_97619

theorem det_A_equals_negative_eleven (a d : ℝ) :
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![a, 2; -3, d]
  (A + A⁻¹ = 0) → Matrix.det A = -11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_A_equals_negative_eleven_l976_97619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_h_l976_97676

noncomputable def h (x : ℝ) : ℝ := (5 * x + 3) / (x - 4)

theorem domain_of_h : 
  {x : ℝ | ∃ y, h x = y} = {x : ℝ | x < 4 ∨ x > 4} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_h_l976_97676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_value_given_dot_product_range_of_g_l976_97687

noncomputable def m (x : ℝ) : ℝ × ℝ := (Real.sin x, -Real.sqrt 3)
noncomputable def n (x : ℝ) : ℝ × ℝ := (1, Real.cos x)
noncomputable def f (x : ℝ) : ℝ := (m x).1 * (n x).1 + (m x).2 * (n x).2

theorem sin_value_given_dot_product (x : ℝ) 
  (h1 : x ∈ Set.Icc 0 (Real.pi / 2))
  (h2 : f x = 2 / 3) :
  Real.sin x = (1 + 2 * Real.sqrt 6) / 6 := by
  sorry

-- Part 2
noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin (2 * (x + Real.pi / 4) - Real.pi / 3)

theorem range_of_g (x : ℝ) (h : x ∈ Set.Icc 0 (Real.pi / 2)) :
  g x ∈ Set.Icc (-2) 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_value_given_dot_product_range_of_g_l976_97687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_magnitude_l976_97695

theorem complex_magnitude (m n : ℝ) : 
    (∃ i : ℂ, i * i = -1 ∧ m / (1 + i) = 1 - n * i) → Complex.abs (m + n * Complex.I) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_magnitude_l976_97695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_5_fourth_power_l976_97618

-- Define the functions f and g
noncomputable section
  variable (f g : ℝ → ℝ)

  -- State the conditions
  axiom fg_condition : ∀ x ≥ 1, f (g x) = x^2
  axiom gf_condition : ∀ x ≥ 1, g (f x) = x^4
  axiom g_25 : g 25 = 25

  -- Theorem to prove
  theorem g_5_fourth_power : (g 5)^4 = 25 := by
    sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_5_fourth_power_l976_97618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_z_equals_one_l976_97673

-- Define the complex number z
noncomputable def z : ℂ := (Complex.I * Real.sqrt 3 + Complex.I) / (1 + Complex.I)^2

-- Theorem statement
theorem abs_z_equals_one : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_z_equals_one_l976_97673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_area_in_specific_pyramid_l976_97684

/-- A pyramid with a triangular base -/
structure Pyramid where
  base_side1 : ℝ
  base_side2 : ℝ
  base_side3 : ℝ
  apex_distance : ℝ

/-- Calculate the surface area of an inscribed sphere in a pyramid -/
noncomputable def inscribed_sphere_surface_area (p : Pyramid) : ℝ :=
  sorry

/-- The theorem to be proved -/
theorem inscribed_sphere_area_in_specific_pyramid :
  let p : Pyramid := { base_side1 := 13, base_side2 := 14, base_side3 := 15, apex_distance := 5 }
  inscribed_sphere_surface_area p = (64 * Real.pi) / 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_area_in_specific_pyramid_l976_97684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_center_of_sine_l976_97631

noncomputable def f (x : ℝ) : ℝ := (1/2) * Real.sin (2*x - Real.pi/3)

/-- A function is symmetric about a point if f(c+x) = f(c-x) for all x -/
def IsSymmetricCenter (f : ℝ → ℝ) (c : ℝ × ℝ) : Prop :=
  ∀ x, f (c.1 + x) = f (c.1 - x)

theorem symmetric_center_of_sine (k : ℤ) :
  IsSymmetricCenter f (((k : ℝ) * Real.pi/2 + Real.pi/6), 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_center_of_sine_l976_97631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_proposition_l976_97661

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, (3 : ℝ)^x > 2) ↔ (∃ x : ℝ, (3 : ℝ)^x ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_proposition_l976_97661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_similar_rectangle_l976_97611

/-- Represents a rectangle -/
structure Rectangle where
  side1 : ℝ
  side2 : ℝ
  area : ℝ
  diagonal : ℝ

/-- Represents similarity between two rectangles -/
def Similar (R1 R2 : Rectangle) : Prop :=
  ∃ (k : ℝ), k > 0 ∧ R2.side1 = k * R1.side1 ∧ R2.side2 = k * R1.side2

/-- Given two similar rectangles R1 and R2, where R1 has one side of 3 inches and an area of 21 square inches,
    and R2 has a diagonal of 20 inches, the area of R2 is 60900/841 square inches. -/
theorem area_of_similar_rectangle (R1 R2 : Rectangle) : 
  (R1.side1 = 3) → 
  (R1.area = 21) → 
  (R2.diagonal = 20) → 
  Similar R1 R2 → 
  R2.area = 60900 / 841 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_similar_rectangle_l976_97611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_size_sets_l976_97637

def N (n : Nat) : Nat := n^9 % 10000

theorem equal_size_sets :
  let S₁ := {n : Nat | n < 10000 ∧ n % 2 = 1 ∧ N n > n}
  let S₂ := {n : Nat | n < 10000 ∧ n % 2 = 1 ∧ N n < n}
  Finset.card (Finset.filter (fun n => n < 10000 ∧ n % 2 = 1 ∧ N n > n) (Finset.range 10000)) =
  Finset.card (Finset.filter (fun n => n < 10000 ∧ n % 2 = 1 ∧ N n < n) (Finset.range 10000)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_size_sets_l976_97637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_operations_theorem_l976_97625

-- Define the new operations
noncomputable def newMinus (x y : ℝ) : ℝ := x * y - x / 2

noncomputable def newOplus (x y : ℝ) : ℝ := x + y / 2

-- State the theorem
theorem new_operations_theorem :
  (newMinus 3.6 2 = 5.4) ∧
  (0.12 - newOplus 7.5 4.8 = -9.78) := by
  -- Split the conjunction into two goals
  constructor
  -- Prove the first part
  · simp [newMinus]
    norm_num
  -- Prove the second part
  · simp [newOplus]
    norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_operations_theorem_l976_97625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_PQRS_area_l976_97692

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a trapezoid defined by four points -/
structure Trapezoid where
  P : Point
  Q : Point
  R : Point
  S : Point

/-- Calculates the area of a trapezoid -/
noncomputable def trapezoidArea (t : Trapezoid) : ℝ :=
  let base1 := abs (t.Q.y - t.P.y)
  let base2 := abs (t.R.y - t.S.y)
  let height := abs (t.R.x - t.P.x)
  (base1 + base2) * height / 2

/-- The specific trapezoid PQRS from the problem -/
def PQRS : Trapezoid := {
  P := { x := 1, y := 0 }
  Q := { x := 1, y := 3 }
  R := { x := 5, y := 9 }
  S := { x := 5, y := 3 }
}

theorem trapezoid_PQRS_area : trapezoidArea PQRS = 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_PQRS_area_l976_97692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l976_97671

theorem quadratic_function_properties (a b c : ℝ) : 
  a < 0 → 
  0 < -b / (2 * a) → 
  -b / (2 * a) < 1 → 
  c < 0 → 
  a + b + c > 0 → 
  (Finset.filter (λ x => x > 0) {a*b, a*c, a+b+c, a-b+c, 2*a+b, 2*a-b}).card = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l976_97671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_johnny_peanut_butter_cans_l976_97697

/-- Given:
  * Johnny bought 6 peanut butter cans.
  * The average price of all cans was 36.5¢.
  * He returned some cans.
  * The average price of remaining cans was 30¢.
  * The average price of returned cans was 49.5¢.
Prove: The number of cans Johnny returned is 2. -/
theorem johnny_peanut_butter_cans :
  ∀ (x : ℕ),
  (6 : ℚ) * (365/10) = (6 - x) * 30 + x * (495/10) →
  x = 2 :=
by
  intro x h
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_johnny_peanut_butter_cans_l976_97697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_disney_park_visitors_l976_97604

noncomputable def f (n : ℕ) : ℝ :=
  if 1 ≤ n ∧ n ≤ 8 then 200 * n + 2000
  else if 9 ≤ n ∧ n ≤ 32 then 360 * (3 ^ ((n - 8 : ℝ) / 12)) + 3000
  else if 33 ≤ n ∧ n ≤ 45 then 32400 - 720 * n
  else 0

noncomputable def g (n : ℕ) : ℝ :=
  if 1 ≤ n ∧ n ≤ 18 then 0
  else if 19 ≤ n ∧ n ≤ 32 then 500 * n - 9000
  else if 33 ≤ n ∧ n ≤ 45 then 8800
  else 0

theorem disney_park_visitors :
  (f 21 + f 22 + f 23 + f 24 = 17460) ∧
  (g 21 + g 22 + g 23 + g 24 = 9000) ∧
  (f 28 - g 28 < 80000) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_disney_park_visitors_l976_97604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_x1_x2_min_value_is_two_pi_thirds_l976_97651

noncomputable def f (x : ℝ) := Real.sin (x - Real.pi / 3)

theorem min_sum_x1_x2 (x₁ x₂ : ℝ) :
  f x₁ + f x₂ = 0 →
  (∀ x ∈ Set.Ioo x₁ x₂, StrictMono f) →
  ∃ k : ℤ, |x₁ + x₂| = 2 * |k * Real.pi + Real.pi / 3| ∧ 
  (∀ m : ℤ, 2 * |m * Real.pi + Real.pi / 3| ≥ 2 * |k * Real.pi + Real.pi / 3|) :=
by
  sorry

-- Theorem stating that the minimum value is indeed 2π/3
theorem min_value_is_two_pi_thirds :
  ∃ x₁ x₂ : ℝ, f x₁ + f x₂ = 0 ∧
  (∀ x ∈ Set.Ioo x₁ x₂, StrictMono f) ∧
  |x₁ + x₂| = 2 * Real.pi / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_x1_x2_min_value_is_two_pi_thirds_l976_97651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_value_l976_97645

-- Define the triangle ABC
structure Triangle :=
  (A B C : Real)  -- Angles
  (a b c : Real)  -- Sides

-- Define the problem conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a = 4 ∧ t.b = 4 * Real.sqrt 3 ∧ t.A = 30 * (Real.pi / 180)

-- Theorem statement
theorem angle_B_value (t : Triangle) (h : triangle_conditions t) :
  t.B = 60 * (Real.pi / 180) ∨ t.B = 120 * (Real.pi / 180) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_value_l976_97645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_implies_a_value_l976_97659

-- Define the curve function
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - Real.log (x + 1)

-- Define the derivative of the curve function
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := a - 1 / (x + 1)

-- Theorem statement
theorem tangent_line_implies_a_value (a : ℝ) :
  (f' a 0 = 2) → a = 3 := by
  intro h
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_implies_a_value_l976_97659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_remaining_bullets_value_l976_97643

/-- A shooter fires at a target until the first hit, with a hit rate of 0.6 for each shot.
    There are 4 bullets in total. -/
structure ShootingProblem where
  hitRate : ℝ
  totalBullets : ℕ
  hitRateValue : hitRate = 0.6
  totalBulletsValue : totalBullets = 4

/-- The expected number of remaining bullets -/
def expectedRemainingBullets (p : ShootingProblem) : ℝ :=
  1 * (1 - p.hitRate) ^ 2 * p.hitRate +
  2 * (1 - p.hitRate) * p.hitRate +
  3 * p.hitRate

/-- The expected number of remaining bullets is 2.376 -/
theorem expected_remaining_bullets_value (p : ShootingProblem) :
  expectedRemainingBullets p = 2.376 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_remaining_bullets_value_l976_97643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_theorem_l976_97633

/-- A tetrahedron with given edge lengths -/
structure Tetrahedron where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c

/-- The sum of distances from a point to the vertices of the tetrahedron -/
def distance_sum (t : Tetrahedron) (x : ℝ × ℝ × ℝ) : ℝ :=
  sorry

/-- The minimum sum of distances from any point to the vertices of the tetrahedron -/
noncomputable def min_distance_sum (t : Tetrahedron) : ℝ :=
  Real.sqrt (4 * t.a^2 + 2 * t.b * t.c)

/-- Theorem: The minimum sum of distances is achieved and equals √(4a² + 2bc) -/
theorem min_distance_sum_theorem (t : Tetrahedron) :
  ∀ x : ℝ × ℝ × ℝ, min_distance_sum t ≤ distance_sum t x :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_theorem_l976_97633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_cosine_inequality_l976_97640

theorem negation_of_cosine_inequality :
  (¬ ∀ x : ℝ, x > 0 → Real.cos x ≥ -1) ↔ (∃ x : ℝ, x > 0 ∧ Real.cos x < -1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_cosine_inequality_l976_97640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_exponentials_l976_97681

/-- Given a = 0.6^0.6, b = 0.6^1.5, and c = 1.5^0.6, prove that b < a < c -/
theorem order_of_exponentials :
  let a := Real.rpow 0.6 0.6
  let b := Real.rpow 0.6 1.5
  let c := Real.rpow 1.5 0.6
  b < a ∧ a < c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_exponentials_l976_97681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_repeating_decimals_l976_97652

/-- Given two repeating decimals, 0.overline{03} and 0.overline{8}, 
    their product is equal to 8/297 -/
theorem product_of_repeating_decimals : 
  ∃ (a b : ℚ), (a = 1/33) ∧ (b = 8/9) → a * b = 8 / 297 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_repeating_decimals_l976_97652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_equivalence_l976_97630

theorem proposition_equivalence (m : ℝ) : 
  (¬ ∃ x : ℝ, x^2 - m*x - m ≤ 0) ↔ 
  ((∀ x : ℝ, x^2 - m*x - m > 0) ∧ (-4 < m ∧ m < 0)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_equivalence_l976_97630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_dot_product_implies_cos_l976_97602

noncomputable def m (θ : ℝ) : ℝ × ℝ := (2 * Real.sqrt 3, Real.cos θ)
noncomputable def n (θ : ℝ) : ℝ × ℝ := (Real.sin θ, 2)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem vector_dot_product_implies_cos (θ : ℝ) :
  dot_product (m θ) (n θ) = 1 → Real.cos (2 * θ - 2 * π / 3) = -7/8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_dot_product_implies_cos_l976_97602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_sum_equals_one_fifth_l976_97627

-- Define the function g as noncomputable
noncomputable def g (x y : ℝ) : ℝ :=
  if x + y ≤ 5 then
    (x * y - x + 3) / (3 * x)
  else
    (x * y - y - 3) / (-3 * y)

-- State the theorem
theorem g_sum_equals_one_fifth : g 3 2 + g 3 5 = 1/5 := by
  -- Proof steps will go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_sum_equals_one_fifth_l976_97627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l976_97677

noncomputable def sequence_a : ℕ → ℝ := sorry
noncomputable def S : ℕ → ℝ := sorry
noncomputable def sequence_b : ℕ → ℝ := sorry
noncomputable def T : ℕ → ℝ := sorry

axiom a_1 : sequence_a 1 = 1
axiom S_relation : ∀ n : ℕ, 3 * S n = sequence_a (n + 1) - 1
axiom b_is_arithmetic : ∃ d : ℝ, ∀ n : ℕ, sequence_b (n + 1) = sequence_b n + d
axiom a_2_eq_b_2 : sequence_a 2 = sequence_b 2
axiom T_4_eq_S_3_plus_1 : T 4 = 1 + S 3

theorem min_value_theorem :
  ∃ min_value : ℝ, min_value = 23 ∧
  ∀ n : ℕ, n ≥ 1 → (2 * T n + 48) / n ≥ min_value := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l976_97677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_expression_l976_97682

theorem power_expression (c d : ℝ) (h1 : (120 : ℝ)^c = 2) (h2 : (120 : ℝ)^d = 3) :
  (24 : ℝ)^((1 - c - d)/(2*(1 - d))) = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_expression_l976_97682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_farm_arrangements_l976_97615

def num_chickens : Nat := 5
def num_dogs : Nat := 1
def num_cats : Nat := 6
def total_cages : Nat := 12

def num_arrangements : Nat := Nat.factorial 3 * Nat.factorial num_chickens * Nat.factorial num_dogs * Nat.factorial num_cats

theorem farm_arrangements :
  num_chickens + num_dogs + num_cats = total_cages →
  num_arrangements = 518400 :=
by
  intro h
  rfl

#eval num_arrangements

end NUMINAMATH_CALUDE_ERRORFEEDBACK_farm_arrangements_l976_97615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_Q₃_l976_97653

/-- Represents a sequence of polyhedra --/
inductive PolyhedraSequence
| mk : ℕ → PolyhedraSequence

/-- Volume of a polyhedron in the sequence --/
noncomputable def volume : PolyhedraSequence → ℝ
| PolyhedraSequence.mk n => sorry

/-- Constructs the next polyhedron in the sequence --/
def next_polyhedron : PolyhedraSequence → PolyhedraSequence
| PolyhedraSequence.mk n => PolyhedraSequence.mk (n + 1)

/-- Initial octahedron Q₀ --/
def Q₀ : PolyhedraSequence := PolyhedraSequence.mk 0

/-- The n-th polyhedron in the sequence --/
def Qₙ (n : ℕ) : PolyhedraSequence := PolyhedraSequence.mk n

theorem volume_Q₃ :
  volume (Qₙ 3) = 157520 / 19683 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_Q₃_l976_97653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_g_l976_97648

noncomputable def f : ℝ → ℝ := fun x => if x ≤ 0 then x^2 + 2*x else x^2 - 2*x

noncomputable def g (a : ℝ) : ℝ → ℝ := fun x => f x + (4 - 2*a)*x + 2

noncomputable def h : ℝ → ℝ := fun a => 
  if a ≤ 2 then 5 - 2*a
  else if a < 3 then -a^2 + 2*a + 1
  else 10 - 4*a

theorem min_value_of_g (a : ℝ) : 
  ∀ x ∈ Set.Icc 1 2, g a x ≥ h a := by
  sorry

#check min_value_of_g

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_g_l976_97648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_30_l976_97678

/-- The angle between clock hands at a given time -/
noncomputable def clock_angle (hours minutes : ℕ) : ℝ :=
  let hour_angle := (hours % 12 + minutes / 60 : ℝ) * 30
  let minute_angle := minutes * 6
  let angle := abs (hour_angle - minute_angle)
  min angle (360 - angle)

/-- Theorem: The smaller angle between clock hands at 3:30 is 75 degrees -/
theorem clock_angle_at_3_30 : clock_angle 3 30 = 75 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_30_l976_97678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_day_division_count_l976_97669

/-- The number of seconds in a day -/
def seconds_per_day : ℕ := 86400

/-- The number of seconds in a minute -/
def seconds_per_minute : ℕ := 60

/-- The number of minutes in a day -/
def minutes_per_day : ℕ := seconds_per_day / seconds_per_minute

/-- A function that returns the number of ways to divide a day into n periods of m minutes -/
noncomputable def count_divisions : ℕ :=
  (Finset.filter (fun d => d * (minutes_per_day / d) ≤ 60) (Nat.divisors minutes_per_day)).card

theorem day_division_count : count_divisions = 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_day_division_count_l976_97669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l976_97634

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1 / (x + 2) + 1 / (y + 2) = 1 / 4) :
  2 * x + 3 * y ≥ 5 + 4 * Real.sqrt 3 ∧
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 1 / (a + 2) + 1 / (b + 2) = 1 / 4 ∧
  2 * a + 3 * b = 5 + 4 * Real.sqrt 3 := by
  sorry

#check min_value_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l976_97634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_winter_sales_calculation_l976_97664

/-- The number of hamburgers sold in millions for each season -/
structure SeasonalSales where
  spring : ℚ
  summer : ℚ
  fall : ℚ
  winter : ℚ

/-- The total number of hamburgers sold in millions -/
def total_sales (s : SeasonalSales) : ℚ := s.spring + s.summer + s.fall + s.winter

/-- The percentage of total sales that occur in fall -/
def fall_percentage (s : SeasonalSales) : ℚ := s.fall / total_sales s

theorem winter_sales_calculation (s : SeasonalSales) 
  (h1 : s.spring = 5)
  (h2 : s.summer = 6)
  (h3 : fall_percentage s = 1/5) :
  s.winter = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_winter_sales_calculation_l976_97664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_digit_numbers_count_five_digit_numbers_ge_30000_count_rank_of_50124_l976_97624

/-- The number of digits available -/
def num_digits : ℕ := 10

/-- A five-digit number -/
def five_digit_number : Type := {n : ℕ // n ≥ 10000 ∧ n < 100000}

/-- A five-digit number greater than or equal to 30000 -/
def five_digit_number_ge_30000 : Type := {n : ℕ // n ≥ 30000 ∧ n < 100000}

/-- A five-digit number with distinct digits -/
def five_digit_number_distinct_digits : Type := 
  {n : five_digit_number // ∀ i j, i ≠ j → (n.val / 10^i) % 10 ≠ (n.val / 10^j) % 10}

/-- The rank of a number in descending order among five-digit numbers with distinct digits -/
noncomputable def rank_descending (n : five_digit_number_distinct_digits) : ℕ := sorry

instance : Fintype five_digit_number := sorry

instance : Fintype five_digit_number_ge_30000 := sorry

theorem five_digit_numbers_count : 
  Fintype.card five_digit_number = 27216 := sorry

theorem five_digit_numbers_ge_30000_count : 
  Fintype.card five_digit_number_ge_30000 = 21168 := sorry

theorem rank_of_50124 : 
  rank_descending ⟨⟨50124, by norm_num⟩, sorry⟩ = 15119 := sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_digit_numbers_count_five_digit_numbers_ge_30000_count_rank_of_50124_l976_97624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_lines_distance_range_l976_97667

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Represents a circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

/-- Checks if two circles are externally tangent -/
def areExternallyTangent (c1 c2 : Circle) : Prop :=
  distance c1.center c2.center = c1.radius + c2.radius

/-- Main theorem -/
theorem four_lines_distance_range :
  let A : Point := ⟨1, 2⟩
  let B : Point := ⟨5, 5⟩
  ∀ d : ℝ,
    (∃! (lines : Set Point),
      (∀ p ∈ lines, distance p A = 1 ∧ distance p B = d) ∧
      (Set.ncard lines = 4)) →
    0 < d ∧ d < 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_lines_distance_range_l976_97667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l976_97614

/-- The intersection points of a line and a circle --/
structure IntersectionPoints where
  A : ℝ × ℝ
  B : ℝ × ℝ

/-- The line equation y = x + 1 --/
def line_equation (x y : ℝ) : Prop := y = x + 1

/-- The circle equation x^2 + y^2 + 2y - 3 = 0 --/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 + 2*y - 3 = 0

/-- The distance between two points --/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- The main theorem --/
theorem intersection_distance :
  ∃ (points : IntersectionPoints),
    (line_equation points.A.1 points.A.2 ∧ circle_equation points.A.1 points.A.2) ∧
    (line_equation points.B.1 points.B.2 ∧ circle_equation points.B.1 points.B.2) ∧
    distance points.A points.B = 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l976_97614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_volume_after_change_l976_97646

/-- Represents a cylinder with radius and height -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- Calculate the volume of a cylinder -/
noncomputable def volume (c : Cylinder) : ℝ := Real.pi * c.radius^2 * c.height

/-- Theorem: New volume after tripling radius and halving height -/
theorem new_volume_after_change (c : Cylinder) 
  (h_vol : volume c = 15) : 
  volume { radius := 3 * c.radius, height := c.height / 2 } = 67.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_volume_after_change_l976_97646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_center_l976_97662

/-- The function f(z) representing a rotation in the complex plane -/
noncomputable def f (z : ℂ) : ℂ := ((-2 + 2*Complex.I)*z + (-3*Real.sqrt 3 - 15*Complex.I)) / 3

/-- The complex number c around which f(z) rotates -/
noncomputable def c : ℂ := ((-15*Real.sqrt 3 + 30) / 29) + ((-6*Real.sqrt 3 - 75) / 29)*Complex.I

/-- Theorem stating that f(c) = c, proving c is the center of rotation -/
theorem rotation_center : f c = c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_center_l976_97662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_x_coord_l976_97675

/-- A point in the coordinate plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance from a point to the x-axis -/
def distToXAxis (p : Point) : ℝ := |p.y|

/-- Distance from a point to the y-axis -/
def distToYAxis (p : Point) : ℝ := |p.x|

/-- Distance from a point to the line x + y = 3 -/
noncomputable def distToLine (p : Point) : ℝ := |p.x + p.y - 3| / Real.sqrt 2

/-- The point is equally distant from x-axis, y-axis, and the line x + y = 3 -/
def isEquidistant (p : Point) : Prop :=
  distToXAxis p = distToYAxis p ∧ distToXAxis p = distToLine p

theorem equidistant_point_x_coord :
  ∀ p : Point, isEquidistant p → p.x = 3/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_x_coord_l976_97675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_for_f_m_eq_4_range_of_a_for_f_a_lt_neg_6_l976_97660

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 - 5 else 1 / (x + 1)

-- Theorem for part (1)
theorem solutions_for_f_m_eq_4 :
  ∃ m₁ m₂ : ℝ, m₁ = 3 ∧ m₂ = -3/4 ∧ f m₁ = 4 ∧ f m₂ = 4 ∧
  ∀ m : ℝ, f m = 4 → m = m₁ ∨ m = m₂ := by
  sorry

-- Theorem for part (2)
theorem range_of_a_for_f_a_lt_neg_6 :
  ∀ a : ℝ, f a < -6 ↔ -7/6 < a ∧ a < -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_for_f_m_eq_4_range_of_a_for_f_a_lt_neg_6_l976_97660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a_value_l976_97607

theorem smallest_a_value (a b : ℕ) (h1 : b - a = 2013) 
  (h2 : ∃ x : ℕ, x^2 - a*x + b = 0) : 
  (∀ a' : ℕ, (∃ b' : ℕ, b' - a' = 2013 ∧ ∃ x : ℕ, x^2 - a'*x + b' = 0) → a' ≥ 93) ∧ 
  (∃ b'' : ℕ, b'' - 93 = 2013 ∧ ∃ x : ℕ, x^2 - 93*x + b'' = 0) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a_value_l976_97607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lindas_cafe_cost_l976_97685

/-- Represents Linda's Cafe pricing structure and calculates total cost -/
structure LindaCafe where
  sandwichPrice : ℕ := 4
  smoothiePrice : ℕ := 3
  discountThreshold : ℕ := 4
  discountAmount : ℕ := 1
  calculateCost : ℕ → ℕ → ℕ
    := fun numSandwiches numSmoothies =>
         let sandwichCost := 
           if numSandwiches > 4 
           then (4 - 1) * numSandwiches
           else 4 * numSandwiches
         let smoothieCost := 3 * numSmoothies
         sandwichCost + smoothieCost

/-- The instance of Linda's Cafe -/
def lindaCafe : LindaCafe := {}

/-- Theorem: The cost of 6 sandwiches and 7 smoothies at Linda's Cafe is $39 -/
theorem lindas_cafe_cost : lindaCafe.calculateCost 6 7 = 39 := by
  sorry

#eval lindaCafe.calculateCost 6 7

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lindas_cafe_cost_l976_97685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_l976_97605

theorem range_of_x (a₁ a₂ a₃ : ℝ) (h₁ : a₁ > a₂) (h₂ : a₂ > a₃) (h₃ : a₃ > 0) :
  {x : ℝ | (1 - a₁ * x)^2 < 1 ∧ (1 - a₂ * x)^2 < 1 ∧ (1 - a₃ * x)^2 < 1} =
  Set.Ioo 0 (2 / a₁) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_l976_97605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ABF_l976_97698

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/3 = 1

-- Define the right vertex A
def right_vertex : ℝ × ℝ := (1, 0)

-- Define the right focus F
def right_focus : ℝ × ℝ := (2, 0)

-- Define the asymptotes
noncomputable def asymptote_pos (x y : ℝ) : Prop := y = Real.sqrt 3 * x
noncomputable def asymptote_neg (x y : ℝ) : Prop := y = -Real.sqrt 3 * x

-- Define the line l
noncomputable def line_l (x y : ℝ) : Prop := y = Real.sqrt 3 * (x - 2)

-- Define point B as the intersection of line_l and asymptote_neg
noncomputable def point_B : ℝ × ℝ := (1, -Real.sqrt 3)

-- Theorem statement
theorem area_of_triangle_ABF :
  let A := right_vertex
  let F := right_focus
  let B := point_B
  (1/2 : ℝ) * ‖F.1 - A.1‖ * ‖B.2‖ = Real.sqrt 3 / 2 := by
  sorry

#eval "Compilation successful!"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ABF_l976_97698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_expression_l976_97691

theorem max_value_of_expression (y : ℝ) :
  y^6 / (y^12 + 4*y^9 - 6*y^6 + 16*y^3 + 64) ≤ 1/26 ∧
  (y^6 / (y^12 + 4*y^9 - 6*y^6 + 16*y^3 + 64) = 1/26 ↔ y = (4 : ℝ)^(1/3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_expression_l976_97691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2017_value_l976_97658

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- The fractional part function -/
noncomputable def frac (x : ℝ) : ℝ :=
  x - floor x

/-- The sequence a_n -/
noncomputable def a : ℕ → ℝ
  | 0 => Real.sqrt 3
  | n + 1 => floor (a n) + 1 / frac (a n)

/-- The main theorem -/
theorem a_2017_value : a 2016 = 3024 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2017_value_l976_97658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l976_97650

/-- Given a hyperbola L with foci at (±4, 0) and vertices at (±√7, 0),
    prove that its equation is x²/7 - y²/9 = 1 -/
theorem hyperbola_equation (L : Set (ℝ × ℝ)) :
  (∃ (x y : ℝ), (x, y) ∈ L ↔ x^2 / 7 - y^2 / 9 = 1) ↔
  (∀ (x y : ℝ), (x, y) ∈ L →
    ((x = 4 ∨ x = -4) ∧ y = 0) ∨
    ((x = Real.sqrt 7 ∨ x = -Real.sqrt 7) ∧ y = 0)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l976_97650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_implies_cos_pi_4_minus_alpha_l976_97622

theorem sin_2alpha_implies_cos_pi_4_minus_alpha
  (α : ℝ)
  (h1 : Real.sin (2 * α) = 24 / 25)
  (h2 : 0 < α)
  (h3 : α < Real.pi / 2) :
  Real.sqrt 2 * Real.cos (Real.pi / 4 - α) = 7 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_implies_cos_pi_4_minus_alpha_l976_97622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_is_integer_l976_97657

-- Define natural numbers as a subset of integers
def NaturalNumbers : Set ℤ := {n : ℤ | n ≥ 0}

-- Define the property that all natural numbers are integers
axiom natural_are_integers : ∀ n : ℤ, n ∈ NaturalNumbers → n ∈ Set.univ

-- State that 4 is a natural number
axiom four_is_natural : (4 : ℤ) ∈ NaturalNumbers

-- Theorem to prove
theorem four_is_integer : (4 : ℤ) ∈ Set.univ := by
  apply natural_are_integers
  exact four_is_natural


end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_is_integer_l976_97657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exsphere_implies_marked_points_on_sphere_l976_97628

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a tetrahedron -/
structure Tetrahedron where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Represents a sphere -/
structure Sphere where
  center : Point3D
  radius : ℝ

/-- Calculates the semi-perimeter of a triangle given its side lengths -/
noncomputable def semiPerimeter (a b c : ℝ) : ℝ := (a + b + c) / 2

/-- Checks if a sphere touches all edges of a tetrahedron -/
def isExsphere (s : Sphere) (t : Tetrahedron) : Prop := sorry

/-- Marks 12 points on the continuations of tetrahedron edges -/
noncomputable def markPoints (t : Tetrahedron) : List Point3D := sorry

/-- Checks if all points in a list lie on the same sphere -/
def lieOnSameSphere (points : List Point3D) : Prop := sorry

/-- Main theorem -/
theorem exsphere_implies_marked_points_on_sphere (t : Tetrahedron) (s : Sphere) :
  isExsphere s t → lieOnSameSphere (markPoints t) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exsphere_implies_marked_points_on_sphere_l976_97628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_integer_coordinates_l976_97612

theorem midpoint_integer_coordinates (points : Finset (ℤ × ℤ)) : 
  points.card = 5 → ∃ A B : ℤ × ℤ, A ∈ points ∧ B ∈ points ∧ A ≠ B ∧ 
  (∃ x y : ℤ, (x, y) = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_integer_coordinates_l976_97612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_zero_l976_97621

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (2*x - 5) / (x^2 + 1)

-- State the theorem
theorem tangent_slope_at_zero : 
  deriv f 0 = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_zero_l976_97621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_implies_a_range_l976_97600

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 1 then Real.log x / Real.log a
  else if x > 1 then (4 - a) * x^2 - a * x + 1
  else 0  -- undefined for x ≤ 0

-- State the theorem
theorem f_monotone_implies_a_range (a : ℝ) :
  (∀ x y : ℝ, 0 < x ∧ 0 < y ∧ x < y → f a x < f a y) →
  1 < a ∧ a ≤ 5/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_implies_a_range_l976_97600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_regions_l976_97665

-- Define the regions Ω₁ and Ω₂
def Ω₁ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 ≤ p.1 ∧ 3 * p.2 ≥ p.1 ∧ p.1 + p.2 ≤ 4}

def Ω₂ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 + 2)^2 + (p.2 - 2)^2 ≤ 2}

-- Define the distance function between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- State the theorem
theorem min_distance_between_regions :
  ∃ (d : ℝ), d = Real.sqrt 2 ∧
  ∀ (p q : ℝ × ℝ), p ∈ Ω₁ → q ∈ Ω₂ → distance p q ≥ d :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_regions_l976_97665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_theorem_l976_97670

/-- Represents the rotation of two points on circles -/
structure CircleRotation where
  /-- Time for the faster point to complete one revolution (in seconds) -/
  faster_time : ℝ
  /-- Time difference between the slower and faster point (in seconds) -/
  time_difference : ℝ
  /-- Number of additional revolutions the faster point makes in one minute -/
  additional_revolutions : ℝ

/-- Calculates the number of revolutions per minute for a given rotation time -/
noncomputable def revolutions_per_minute (rotation_time : ℝ) : ℝ :=
  60 / rotation_time

/-- Theorem stating the relationship between rotation times and revolutions per minute -/
theorem rotation_theorem (cr : CircleRotation) 
  (h1 : cr.time_difference = 5)
  (h2 : cr.additional_revolutions = 2)
  (h3 : cr.faster_time > 0) :
  revolutions_per_minute cr.faster_time = 6 ∧ 
  revolutions_per_minute (cr.faster_time + cr.time_difference) = 4 := by
  sorry

#check rotation_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_theorem_l976_97670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l976_97690

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x + 2) / (x - 3)

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | x ≥ -2 ∧ x ≠ 3} = {x | ∃ y, f x = y} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l976_97690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_problem_l976_97655

noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (periods : ℝ) : ℝ :=
  principal * (1 + rate / periods) ^ periods

theorem investment_problem (r : ℝ) : 
  let initial_investment := (12000 : ℝ)
  let first_rate := (0.10 : ℝ)
  let first_time := (0.5 : ℝ)
  let first_value := simple_interest initial_investment first_rate first_time
  let second_periods := (2 : ℝ)
  let final_value := (13260 : ℝ)
  compound_interest first_value (r / 100) second_periods = final_value →
  r = 10.476 := by
  sorry

#check investment_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_problem_l976_97655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_growth_equation_l976_97693

/-- Represents the growth rate of investment over two years -/
def x : ℝ := sorry

/-- The initial investment in 2021 (in units of 10 billion yuan) -/
def initial_investment : ℝ := 10

/-- The planned investment in 2023 (in units of 10 billion yuan) -/
def planned_investment : ℝ := 40

/-- Theorem stating the relationship between the initial investment, 
    planned investment, and the growth rate -/
theorem investment_growth_equation : 
  initial_investment * (1 + x)^2 = planned_investment := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_growth_equation_l976_97693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marbles_arrangement_l976_97616

/-- The number of green marbles --/
def green_marbles : ℕ := 7

/-- The maximum number of red marbles for which an arrangement is possible --/
def m : ℕ := 18

/-- The number of ways to arrange m + green_marbles marbles satisfying the condition --/
def N : ℕ := 50388

/-- Predicate to check if an arrangement is valid --/
def valid_arrangement (arrangement : List (Fin 2)) : Prop :=
  let same_color_neighbors := (arrangement.zip arrangement.tail).filter (fun (a, b) => a = b) |>.length
  let different_color_neighbors := (arrangement.zip arrangement.tail).filter (fun (a, b) => a ≠ b) |>.length
  same_color_neighbors > different_color_neighbors

theorem marbles_arrangement :
  (∃ (arrangement : List (Fin 2)), arrangement.length = m + green_marbles ∧
                                   arrangement.count 0 = green_marbles ∧
                                   arrangement.count 1 = m ∧
                                   valid_arrangement arrangement) ∧
  (∀ k > m, ¬∃ (arrangement : List (Fin 2)), arrangement.length = k + green_marbles ∧
                                             arrangement.count 0 = green_marbles ∧
                                             arrangement.count 1 = k ∧
                                             valid_arrangement arrangement) ∧
  N % 1000 = 388 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_marbles_arrangement_l976_97616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l976_97668

theorem equation_solution (b c : ℝ) (h : b > c) :
  ((b^2 - c^2) / (2*b))^2 + c^2 = (b - (b^2 - c^2) / (2*b))^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l976_97668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_1_423_round_3_2387_round_1_996_l976_97674

-- Define a function to round a number to the nearest hundredth
noncomputable def roundToHundredth (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

-- Theorem for the first number
theorem round_1_423 : roundToHundredth 1.423 = 1.42 := by sorry

-- Theorem for the second number
theorem round_3_2387 : roundToHundredth 3.2387 = 3.24 := by sorry

-- Theorem for the third number
theorem round_1_996 : roundToHundredth 1.996 = 2.00 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_1_423_round_3_2387_round_1_996_l976_97674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_store_problem_l976_97656

/-- Proves that the cost per candy is $2.50 given the conditions of the candy store problem -/
theorem candy_store_problem (grape_count : ℕ) (total_cost : ℚ) : 
  grape_count = 24 →
  total_cost = 200 →
  (let cherry_count := grape_count / 3
   let apple_count := grape_count * 2
   let total_count := cherry_count + grape_count + apple_count
   total_cost / total_count) = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_store_problem_l976_97656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_problem_l976_97683

/-- The area of the shaded region formed by semicircles in a given pattern -/
noncomputable def shaded_area (diameter : ℝ) (pattern_length : ℝ) : ℝ :=
  let num_circles := pattern_length / diameter
  let circle_area := Real.pi * (diameter / 2)^2
  num_circles * circle_area

/-- Theorem stating the area of the shaded region for the given problem -/
theorem shaded_area_problem :
  shaded_area 4 (2 * 12) = 24 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_problem_l976_97683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_segment_length_l976_97688

/-- A line passing through a point with a given inclination angle -/
structure InclinedLine where
  point : ℝ × ℝ
  angle : ℝ

/-- A circle with a given radius -/
structure Circle where
  radius : ℝ

/-- Intersection points of a line and a circle -/
structure Intersection where
  pointA : ℝ × ℝ
  pointB : ℝ × ℝ

/-- Calculate the length of a line segment given two points -/
noncomputable def segmentLength (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Main theorem: The length of the intersection segment is √14 -/
theorem intersection_segment_length 
  (l : InclinedLine) 
  (c : Circle) 
  (i : Intersection) : 
  l.point = (2, 1) → 
  l.angle = π/4 → 
  c.radius = 2 → 
  segmentLength i.pointA i.pointB = Real.sqrt 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_segment_length_l976_97688
