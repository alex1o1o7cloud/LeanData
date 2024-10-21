import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_area_l1203_120352

/-- Given a hyperbola with equation x²/4 - y²/9 = 1, 
    the area enclosed by one asymptote and the two asymptotes is 24/13 -/
theorem hyperbola_asymptote_area : 
  ∃ (x y : ℝ), 
    (x^2 / 4 - y^2 / 9 = 1) → 
    (∃ (A : Set (ℝ × ℝ)), 
      (MeasureTheory.volume A = 24 / 13 ∧ 
       ∃ (f g h : ℝ → ℝ), 
         (∀ t, (f t, g t) ∈ A) ∧ 
         (∀ t, (f t, h t) ∈ A) ∧ 
         (∃ k, ∀ t, f t = k) ∧
         (∃ m n, ∀ t, g t = m * t + n) ∧
         (∃ p q, ∀ t, h t = p * t + q) ∧
         (∀ t, f t^2 / 4 - g t^2 / 9 = 1) ∧
         (∀ t, f t^2 / 4 - h t^2 / 9 = 1))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_area_l1203_120352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_50721_between_consecutive_integers_sum_l1203_120353

theorem log_50721_between_consecutive_integers_sum :
  ∃ (c d : ℤ), c + 1 = d ∧ (c : ℝ) < Real.log 50721 / Real.log 10 ∧ Real.log 50721 / Real.log 10 < (d : ℝ) ∧ c + d = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_50721_between_consecutive_integers_sum_l1203_120353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_symmetry_l1203_120336

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

noncomputable def g (x : ℝ) : ℝ := |floor (x + 0.5)| - |floor (1.5 - x)|

theorem g_symmetry (x : ℝ) : g x = g (1.5 - x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_symmetry_l1203_120336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_students_speaking_both_languages_l1203_120371

theorem students_speaking_both_languages (total : ℕ) (english : ℕ) (telugu : ℕ) (neither : ℕ) (both : ℕ)
  (h1 : total = 150)
  (h2 : english = 55)
  (h3 : telugu = 85)
  (h4 : neither = 30) :
  total = english + telugu - both + neither ∧ both = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_students_speaking_both_languages_l1203_120371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_in_first_quadrant_l1203_120316

theorem complex_number_in_first_quadrant : ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ Complex.I * (2 - Complex.I) = Complex.mk a b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_in_first_quadrant_l1203_120316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_intervals_when_m_zero_range_of_m_for_nonnegative_f_l1203_120301

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.exp x - (m * x^2 + x + 1)

-- Part 1: Monotonicity intervals when m = 0
theorem monotonicity_intervals_when_m_zero :
  (∀ x y : ℝ, x < y ∧ y < 0 → f 0 x > f 0 y) ∧
  (∀ x y : ℝ, 0 < x ∧ x < y → f 0 x < f 0 y) := by
  sorry

-- Part 2: Range of m for which f(x) ≥ 0 when x ≥ 0
theorem range_of_m_for_nonnegative_f :
  {m : ℝ | ∀ x : ℝ, x ≥ 0 → f m x ≥ 0} = Set.Iic (1/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_intervals_when_m_zero_range_of_m_for_nonnegative_f_l1203_120301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_from_cosine_trig_expression_l1203_120309

/-- Given cosine of an angle in the third quadrant, prove the value of sine -/
theorem sine_from_cosine (α : Real) (h1 : Real.cos α = -4/5) (h2 : α ∈ Set.Icc Real.pi (3*Real.pi/2)) :
  Real.sin α = -3/5 := by sorry

/-- Given tangent of an angle, prove the value of a trigonometric expression -/
theorem trig_expression (α : Real) (h : Real.tan α = 3) :
  (4 * Real.sin α - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 5/7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_from_cosine_trig_expression_l1203_120309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_root_16_div_cube_root_27_l1203_120318

theorem fourth_root_16_div_cube_root_27 : (16 : ℝ) ^ (1/4) / (27 : ℝ) ^ (1/3) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_root_16_div_cube_root_27_l1203_120318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l1203_120398

open Set Real

def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
def B : Set ℝ := {x | Real.exp (Real.log 3 * x) < 1}

theorem intersection_of_A_and_B : A ∩ B = Icc (-1) 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l1203_120398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base17_divisibility_theorem_l1203_120347

/-- Represents a base-17 digit -/
def Base17Digit := Fin 17

/-- Converts a base-17 number to its decimal representation -/
def toDecimal (digits : List Base17Digit) : ℕ :=
  digits.foldl (fun acc d => 17 * acc + d.val) 0

/-- Checks if a natural number is divisible by 16 -/
def isDivisibleBy16 (n : ℕ) : Prop := n % 16 = 0

/-- The main theorem -/
theorem base17_divisibility_theorem (a : Base17Digit) :
  let n := toDecimal [⟨8, by norm_num⟩, ⟨3, by norm_num⟩, ⟨2, by norm_num⟩, ⟨3, by norm_num⟩, 
                      a, ⟨0, by norm_num⟩, ⟨2, by norm_num⟩, ⟨4, by norm_num⟩, ⟨2, by norm_num⟩, ⟨1, by norm_num⟩]
  isDivisibleBy16 n → a.val = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base17_divisibility_theorem_l1203_120347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_density_integral_l1203_120343

-- Define the normal distribution density function
noncomputable def normal_density (a σ x : ℝ) : ℝ := 
  (1 / (σ * Real.sqrt (2 * Real.pi))) * Real.exp (-(x - a)^2 / (2 * σ^2))

-- State the theorem
theorem normal_density_integral (a : ℝ) (σ : ℝ) (h : σ > 0) :
  ∫ (x : ℝ), normal_density a σ x = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_density_integral_l1203_120343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_and_perpendicularity_l1203_120351

/-- Defines an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Defines a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines a line in slope-intercept form -/
structure Line where
  k : ℝ
  m : ℝ

/-- Defines membership of a point in an ellipse -/
def Point.mem (p : Point) (E : Ellipse) : Prop :=
  p.x^2 / E.a^2 + p.y^2 / E.b^2 = 1

theorem ellipse_equation_and_perpendicularity 
  (E : Ellipse) 
  (F₁ F₂ : Point) 
  (h_foci : F₁.x < F₂.x ∧ F₁.y = 0 ∧ F₂.y = 0) 
  (h_ecc : (F₂.x - F₁.x) / (2 * E.a) = Real.sqrt 2 / 2) 
  (h_perim : ∀ (R : Point), Point.mem R E → 
    Real.sqrt ((R.x - F₁.x)^2 + (R.y - F₁.y)^2) + 
    Real.sqrt ((R.x - F₂.x)^2 + (R.y - F₂.y)^2) = 2 * (Real.sqrt 2 + 1)) :
  (∀ (x y : ℝ), (x^2 / 2 + y^2 = 1) ↔ (x^2 / E.a^2 + y^2 / E.b^2 = 1)) ∧ 
  (∀ (l : Line) (R N : Point), 
    Point.mem R E → 
    R.y = l.k * R.x + l.m → 
    N.x = 2 → 
    N.y = l.k * N.x + l.m → 
    (R.x - F₂.x) * (N.x - F₂.x) + (R.y - F₂.y) * (N.y - F₂.y) = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_and_perpendicularity_l1203_120351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_for_point_one_neg_two_l1203_120322

/-- Given a point (1, -2) on the terminal side of angle α, prove that sin(α) = -2√5/5 -/
theorem sin_alpha_for_point_one_neg_two (α : ℝ) :
  (∃ (r : ℝ), r > 0 ∧ r^2 = 5 ∧ r * Real.cos α = 1 ∧ r * Real.sin α = -2) →
  Real.sin α = -2 * Real.sqrt 5 / 5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_for_point_one_neg_two_l1203_120322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_ratio_equals_eccentricity_constant_ratio_set_is_ellipse_l1203_120310

/-- Definition of an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ
  e : ℝ
  h_ab : 0 < b ∧ b < a
  h_e : 0 < e ∧ e < 1
  h_e_def : e = Real.sqrt (1 - b^2 / a^2)

/-- A point on an ellipse -/
def PointOnEllipse (E : Ellipse) : Type := { p : ℝ × ℝ // (p.1 / E.a)^2 + (p.2 / E.b)^2 = 1 }

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Distance from a point to a line -/
noncomputable def distanceToLine (p : ℝ × ℝ) (l : ℝ → ℝ) : ℝ := sorry

/-- Focus of an ellipse -/
noncomputable def focus (E : Ellipse) : ℝ × ℝ := (E.a * E.e, 0)

/-- Directrix of an ellipse -/
noncomputable def directrix (E : Ellipse) : ℝ → ℝ := fun x ↦ E.a / E.e

/-- Theorem: The ratio of distances from a point on an ellipse to a focus and to a directrix equals the eccentricity -/
theorem distance_ratio_equals_eccentricity (E : Ellipse) (p : PointOnEllipse E) :
  distance p.val (focus E) / distanceToLine p.val (directrix E) = E.e := by sorry

/-- Definition of a set of points with constant distance ratio to a point and a line -/
def ConstantRatioSet (F : ℝ × ℝ) (l : ℝ → ℝ) (e : ℝ) : Set (ℝ × ℝ) :=
  { p : ℝ × ℝ | distance p F / distanceToLine p l = e }

/-- Theorem: The set of points with constant distance ratio e < 1 to a fixed point and a fixed line forms an ellipse -/
theorem constant_ratio_set_is_ellipse (F : ℝ × ℝ) (l : ℝ → ℝ) (e : ℝ) (h_e : 0 < e ∧ e < 1) :
  ∃ E : Ellipse, ConstantRatioSet F l e = { p | ∃ q : PointOnEllipse E, p = q.val } := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_ratio_equals_eccentricity_constant_ratio_set_is_ellipse_l1203_120310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_externally_tangent_l1203_120307

/-- The distance from the origin to the line ax + by + 1 = 0 is 1/2 -/
def distance_condition (a b : ℝ) : Prop :=
  1 / Real.sqrt (a^2 + b^2) = 1/2

/-- The first circle: (x-a)² + y² = 1 -/
def circle1 (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - a)^2 + p.2^2 = 1}

/-- The second circle: x² + (y-b)² = 1 -/
def circle2 (b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + (p.2 - b)^2 = 1}

/-- Two circles are externally tangent if the distance between their centers
    is equal to the sum of their radii -/
def externally_tangent (S T : Set (ℝ × ℝ)) : Prop :=
  ∃ (c₁ c₂ : ℝ × ℝ) (r₁ r₂ : ℝ),
    (∀ p ∈ S, (p.1 - c₁.1)^2 + (p.2 - c₁.2)^2 = r₁^2) ∧
    (∀ p ∈ T, (p.1 - c₂.1)^2 + (p.2 - c₂.2)^2 = r₂^2) ∧
    (c₁.1 - c₂.1)^2 + (c₁.2 - c₂.2)^2 = (r₁ + r₂)^2

theorem circles_externally_tangent (a b : ℝ) :
  distance_condition a b → externally_tangent (circle1 a) (circle2 b) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_externally_tangent_l1203_120307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_terminal_side_theorem_l1203_120381

-- Define the angle α as a function of a
noncomputable def α (a : ℝ) : ℝ := Real.arctan (3/4)

-- Define the point P
def P (a : ℝ) : ℝ × ℝ := (-4*a, 3*a)

-- Define the theorem
theorem angle_terminal_side_theorem (a : ℝ) (h : a ≠ 0) :
  (a > 0 → Real.sin (α a) + Real.cos (α a) - Real.tan (α a) = 11/20) ∧
  (a < 0 → Real.sin (α a) + Real.cos (α a) - Real.tan (α a) = 19/20) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_terminal_side_theorem_l1203_120381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_correct_l1203_120368

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then x else x^2 - 2*x - 5

-- Define the solution set
def solution_set : Set ℝ :=
  {x | -2 ≤ x ∧ x < 1} ∪ {x | x ≥ 3}

-- Theorem statement
theorem solution_set_correct :
  solution_set = {x | f x ≥ -2} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_correct_l1203_120368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_project_profitability_and_min_cost_l1203_120337

noncomputable def cost (x : ℝ) : ℝ :=
  if 120 ≤ x ∧ x < 144 then
    (1/3) * x^3 - 80 * x^2 + 5040 * x
  else if 144 ≤ x ∧ x < 500 then
    (1/2) * x^2 - 200 * x + 80000
  else
    0

noncomputable def revenue (x : ℝ) : ℝ := 200 * x

noncomputable def profit (x : ℝ) : ℝ := revenue x - cost x

noncomputable def avg_cost (x : ℝ) : ℝ := cost x / x

theorem project_profitability_and_min_cost :
  (∀ x ∈ Set.Icc 200 300, profit x < 0) ∧
  (∃ s : ℝ, s = 5000 ∧ ∀ x ∈ Set.Icc 200 300, profit x + s ≥ 0) ∧
  (∃ x_min : ℝ, x_min = 400 ∧ ∀ y ∈ Set.Icc 120 500, avg_cost x_min ≤ avg_cost y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_project_profitability_and_min_cost_l1203_120337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_conditions_l1203_120323

-- Define two intersecting lines
def Line : Type := ℝ → ℝ → Prop

-- Define the angle between two lines
noncomputable def Angle (l1 l2 : Line) : ℝ := sorry

-- Define perpendicularity of two lines
def Perpendicular (l1 l2 : Line) : Prop := Angle l1 l2 = Real.pi / 2

-- Define the four angles formed by the intersection of two lines
noncomputable def IntersectionAngles (l1 l2 : Line) : Fin 4 → ℝ := sorry

-- Define adjacent angles
def Adjacent (a1 a2 : ℝ) : Prop := sorry

-- Define supplementary angles
def Supplementary (a1 a2 : ℝ) : Prop := a1 + a2 = Real.pi

-- Define vertically opposite angles
def VerticallyOpposite (a1 a2 : ℝ) (l1 l2 : Line) : Prop := sorry

theorem perpendicular_conditions (l1 l2 : Line) :
  (∃ i : Fin 4, IntersectionAngles l1 l2 i = Real.pi / 2) →
  Perpendicular l1 l2 ∧
  (∀ i j : Fin 4, IntersectionAngles l1 l2 i = IntersectionAngles l1 l2 j) →
  Perpendicular l1 l2 ∧
  (∃ a1 a2, Adjacent a1 a2 ∧ Supplementary a1 a2 ∧ a1 = a2) →
  Perpendicular l1 l2 ∧
  (∃ a1 a2, VerticallyOpposite a1 a2 l1 l2 ∧ Supplementary a1 a2) →
  Perpendicular l1 l2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_conditions_l1203_120323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_parabola_l1203_120330

-- Define the circle
def circleO (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define points A and B on the y-axis
def pointA : ℝ × ℝ := (0, 2)
def pointB : ℝ × ℝ := (0, -2)

-- Define the tangent line l
def lineL (y : ℝ) : Prop := y = -2

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the distance from a point to a line
def distanceToLine (p : ℝ × ℝ) (y : ℝ) : ℝ := |p.2 - y|

-- Theorem statement
theorem trajectory_is_parabola :
  ∀ (p : ℝ × ℝ),
    (distance p pointA = distanceToLine p (-2)) →
    (p.1^2 = 8 * p.2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_parabola_l1203_120330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_power_dividing_factorial_l1203_120329

theorem largest_power_dividing_factorial :
  ∃ (n : ℕ), n = 7 ∧ 
  (∀ m : ℕ, (18^m : ℕ) ∣ Nat.factorial 30 → m ≤ n) ∧
  ((18^n : ℕ) ∣ Nat.factorial 30) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_power_dividing_factorial_l1203_120329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_of_f_l1203_120373

/-- The function resulting from stretching the horizontal coordinates of sin(x + π/6) to twice their original length -/
noncomputable def f (x : ℝ) : ℝ := Real.sin ((1/2) * x + Real.pi/6)

/-- Theorem stating that x = 2π/3 is a symmetry axis for the function f -/
theorem symmetry_axis_of_f :
  ∀ x : ℝ, f ((2 * Real.pi)/3 + x) = f ((2 * Real.pi)/3 - x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_of_f_l1203_120373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cup_height_ordering_l1203_120389

/-- Represents the shape of a cup -/
inductive CupShape
  | Cylindrical
  | Conical
  | Hemispherical

/-- Represents a cup with a given shape and height -/
structure Cup where
  shape : CupShape
  height : ℝ

/-- Calculate the volume of a cup given its shape and height -/
noncomputable def volume (c : Cup) : ℝ :=
  match c.shape with
  | CupShape.Cylindrical => Real.pi * c.height
  | CupShape.Conical => (1/3) * Real.pi * c.height
  | CupShape.Hemispherical => (2/3) * Real.pi

theorem cup_height_ordering (cupA cupB cupC : Cup)
  (hA : cupA.shape = CupShape.Cylindrical)
  (hB : cupB.shape = CupShape.Conical)
  (hC : cupC.shape = CupShape.Hemispherical)
  (hVol : volume cupA = volume cupB ∧ volume cupB = volume cupC)
  : cupA.height < cupC.height ∧ cupC.height < cupB.height := by
  sorry

#check cup_height_ordering

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cup_height_ordering_l1203_120389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_triangle_AOB_area_minimize_sum_OA_OB_l1203_120338

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line in 2D space represented by equation x/a + y/b = 1 -/
structure Line where
  a : ℝ
  b : ℝ

/-- Triangle formed by points O, A, and B -/
structure Triangle where
  O : Point
  A : Point
  B : Point

def P : Point := { x := 4, y := 1 }

def line_passes_through_point (l : Line) (p : Point) : Prop :=
  p.x / l.a + p.y / l.b = 1

def line_intersects_axes (l : Line) : Prop :=
  l.a > 0 ∧ l.b > 0

def triangle_AOB (l : Line) : Triangle :=
  { O := { x := 0, y := 0 },
    A := { x := l.a, y := 0 },
    B := { x := 0, y := l.b } }

noncomputable def triangle_area (t : Triangle) : ℝ :=
  (t.A.x - t.O.x) * (t.B.y - t.O.y) / 2

noncomputable def sum_OA_OB (l : Line) : ℝ :=
  l.a + l.b

theorem minimize_triangle_AOB_area (l : Line) :
  line_passes_through_point l P →
  line_intersects_axes l →
  (∀ l' : Line, line_passes_through_point l' P → line_intersects_axes l' →
    triangle_area (triangle_AOB l) ≤ triangle_area (triangle_AOB l')) →
  l.a = 8 ∧ l.b = 2 := by
  sorry

theorem minimize_sum_OA_OB (l : Line) :
  line_passes_through_point l P →
  line_intersects_axes l →
  (∀ l' : Line, line_passes_through_point l' P → line_intersects_axes l' →
    sum_OA_OB l ≤ sum_OA_OB l') →
  l.a = 6 ∧ l.b = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_triangle_AOB_area_minimize_sum_OA_OB_l1203_120338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_expression_equality_l1203_120375

theorem trigonometric_expression_equality : 
  (Real.sin (30 * π / 180) * Real.cos (24 * π / 180) + Real.cos (150 * π / 180) * Real.cos (90 * π / 180)) / 
  (Real.sin (34 * π / 180) * Real.cos (16 * π / 180) + Real.cos (146 * π / 180) * Real.cos (106 * π / 180)) = 
  Real.cos (24 * π / 180) / (2 * Real.sin (18 * π / 180)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_expression_equality_l1203_120375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_plot_length_l1203_120335

theorem rectangular_plot_length 
  (breadth : ℝ) 
  (length_diff : ℝ) 
  (fence_cost_per_meter : ℝ) 
  (total_fence_cost : ℝ) 
  (h1 : length_diff = 16)
  (h2 : fence_cost_per_meter = 26.50)
  (h3 : total_fence_cost = 5300)
  (h4 : breadth + length_diff = 58) : ℝ := by
  let length := breadth + length_diff
  let perimeter := 2 * (length + breadth)
  have h5 : perimeter = total_fence_cost / fence_cost_per_meter := by sorry
  exact 58

#check rectangular_plot_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_plot_length_l1203_120335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_expression_l1203_120393

theorem calculate_expression : |Real.sqrt 25 - 3| + (-27 : ℝ) ^ (1/3) - (-2)^3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_expression_l1203_120393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_beta_minus_alpha_l1203_120308

theorem cos_beta_minus_alpha (α β : ℝ) 
  (h1 : Real.cos α = -3/5)
  (h2 : π/2 < α ∧ α < π)
  (h3 : Real.sin β = -12/13)
  (h4 : π < β ∧ β < 3*π/2) :
  Real.cos (β - α) = -33/65 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_beta_minus_alpha_l1203_120308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_number_l1203_120380

noncomputable def a : ℝ := 98765 + 1 / 4321
noncomputable def b : ℝ := 98765 - 1 / 4321
noncomputable def c : ℝ := 98765 * (1 / 4321)
noncomputable def d : ℝ := 98765 / (1 / 4321)
noncomputable def e : ℝ := 98765.4321

theorem largest_number : d > a ∧ d > b ∧ d > c ∧ d > e := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_number_l1203_120380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_sin_l1203_120305

-- Define IsAxisOfSymmetry as a function
def IsAxisOfSymmetry (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a + x) = f (a - x)

theorem axis_of_symmetry_sin (x : ℝ) : 
  x = -π/2 → IsAxisOfSymmetry (λ y => Real.sin (2*y + 5*π/2)) x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_sin_l1203_120305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_proof_transformation_properties_l1203_120374

def rotation_matrix : Matrix (Fin 2) (Fin 2) ℝ := !![0, 1; -1, 0]

def scaling_factor : ℝ := 3

def transformation_matrix : Matrix (Fin 2) (Fin 2) ℝ := !![0, 3; -3, 0]

theorem transformation_proof :
  transformation_matrix = scaling_factor • rotation_matrix := by
  sorry

theorem transformation_properties (v : Fin 2 → ℝ) :
  transformation_matrix.vecMul v = scaling_factor • (rotation_matrix.vecMul v) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_proof_transformation_properties_l1203_120374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_w5_approx_l1203_120387

def w : ℕ → ℂ
  | 0 => 1
  | (n + 1) => w n ^ 2 + 1 + Complex.I

theorem magnitude_w5_approx : 
  Complex.abs (w 5 - (-2138 - 1899 * Complex.I)) < 0.1 ∧ 
  |Complex.abs (w 5) - 2861.8| < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_w5_approx_l1203_120387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_properties_l1203_120302

/-- Represents a repeating decimal with a non-repeating part and a repeating part. -/
structure RepeatingDecimal where
  R : ℕ  -- non-repeating part
  K : ℕ  -- repeating part
  t : ℕ  -- number of digits in R
  u : ℕ  -- number of digits in K

/-- The value of the repeating decimal as a real number. -/
noncomputable def value (d : RepeatingDecimal) : ℝ :=
  d.R / (10 ^ d.t : ℝ) + (d.K / (10 ^ d.u - 1 : ℝ)) / (10 ^ d.t : ℝ)

theorem repeating_decimal_properties (d : RepeatingDecimal) :
  let M := value d
  (∃ (x : ℝ), 10^d.t * M = d.R + x ∧ x < 1 ∧ x ≥ 0) ∧ 
  (∃ (y : ℝ), 10^(d.t + d.u) * M = d.R * 10^d.u + d.K + y ∧ y < 1 ∧ y ≥ 0) ∧
  (∃ (z : ℝ), 10^d.t * 10^(2*d.u) * M = d.R * 10^(2*d.u) + d.K * 10^d.u + d.K + z ∧ z < 1 ∧ z ≥ 0) ∧
  ¬(10^d.t * (10^d.u - 1) * M = d.K * (d.R - 1)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_properties_l1203_120302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_double_symmetry_l1203_120339

/-- Given a quadratic function f(x) = ax^2 + bx + c with a ≠ 0,
    applying symmetry with respect to the y-axis followed by
    symmetry with respect to the x-axis results in the function
    g(x) = -ax^2 + bx - c -/
theorem quadratic_double_symmetry (a b c : ℝ) (ha : a ≠ 0) :
  ∀ x y : ℝ, (a * x^2 + b * x + c = y) ↔ (-a * (-x)^2 + b * (-x) - c = -y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_double_symmetry_l1203_120339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_always_possible_to_form_special_triangle_l1203_120350

-- Define a type for a set of 6 positive real numbers
def SixSticks := { s : Finset ℝ // s.card = 6 ∧ ∀ x ∈ s, x > 0 }

-- Define a predicate to check if three numbers can form a triangle
def CanFormTriangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a < b + c ∧ b < a + c ∧ c < a + b

-- Define a predicate to check if a set of 6 numbers can form two triangles
def CanFormTwoTriangles (s : SixSticks) : Prop :=
  ∃ (a b c d e f : ℝ), a ∈ s.val ∧ b ∈ s.val ∧ c ∈ s.val ∧ d ∈ s.val ∧ e ∈ s.val ∧ f ∈ s.val ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
    d ≠ e ∧ d ≠ f ∧
    e ≠ f ∧
    CanFormTriangle a b c ∧ CanFormTriangle d e f

-- The main theorem
theorem always_possible_to_form_special_triangle (s : SixSticks) 
  (h : CanFormTwoTriangles s) :
  ∃ (a b c d e f : ℝ), a ∈ s.val ∧ b ∈ s.val ∧ c ∈ s.val ∧ d ∈ s.val ∧ e ∈ s.val ∧ f ∈ s.val ∧
    CanFormTriangle a (b + c) (d + e + f) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_always_possible_to_form_special_triangle_l1203_120350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lattice_points_count_l1203_120364

/-- The number of lattice points on a line segment --/
def latticePointCount (x1 y1 x2 y2 : ℤ) : ℕ :=
  let dx := x2 - x1
  let dy := y2 - y1
  let gcd := Int.gcd dx.natAbs dy.natAbs
  gcd + 1

/-- Theorem: The number of lattice points on the line segment from (15, 35) to (75, 515) is 61 --/
theorem lattice_points_count :
  latticePointCount 15 35 75 515 = 61 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lattice_points_count_l1203_120364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_and_solution_l1203_120363

noncomputable section

-- Define the function y
def y (x m : ℝ) : ℝ := x^2 + m*x - 4

-- Define the domain of x
def x_domain (x : ℝ) : Prop := 2 ≤ x ∧ x ≤ 4

-- Define the minimum value function g(m)
noncomputable def g (m : ℝ) : ℝ :=
  if m ≥ -4 then 2*m
  else if m > -8 then -m^2/4 - 4
  else 4*m + 12

theorem min_value_and_solution :
  (∀ x m : ℝ, x_domain x → y x m ≥ g m) ∧
  (g 5 = 10 ∧ ∀ m : ℝ, g m = 10 → m = 5) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_and_solution_l1203_120363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial15_trailing_zeroes_base27_l1203_120397

/-- The number of trailing zeroes in n! when written in base b -/
def trailingZeroes (n : ℕ) (b : ℕ) : ℕ := sorry

/-- 15 factorial -/
def factorial15 : ℕ := Nat.factorial 15

/-- Base 27 -/
def base27 : ℕ := 27

theorem factorial15_trailing_zeroes_base27 : 
  trailingZeroes factorial15 base27 = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial15_trailing_zeroes_base27_l1203_120397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_x_plus_one_ln_squared_x_plus_one_l1203_120358

theorem integral_x_plus_one_ln_squared_x_plus_one :
  ∫ x in (0:ℝ)..(1:ℝ), (x + 1) * (Real.log (x + 1))^2 = 2 * (Real.log 2)^2 - 2 * Real.log 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_x_plus_one_ln_squared_x_plus_one_l1203_120358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_of_L_shape_l1203_120370

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram in 2D space -/
structure Parallelogram where
  vertices : Fin 4 → Point2D

/-- Represents an L-shaped polygon -/
structure LShape where
  vertices : Fin 6 → Point2D

/-- Calculates the centroid of a parallelogram -/
noncomputable def centroidParallelogram (p : Parallelogram) : Point2D :=
  { x := (p.vertices 0).x + (p.vertices 2).x / 2,
    y := (p.vertices 0).y + (p.vertices 2).y / 2 }

/-- Calculates the line passing through two points -/
noncomputable def lineThroughPoints (p1 p2 : Point2D) : ℝ → Point2D :=
  λ t => { x := p1.x + t * (p2.x - p1.x),
           y := p1.y + t * (p2.y - p1.y) }

/-- Theorem: The centroid of an L-shaped polygon is the intersection of two lines -/
theorem centroid_of_L_shape (l : LShape)
  (p1 p2 p3 p4 : Parallelogram)
  (partition1 : Set (Fin 4 → Point2D))
  (partition2 : Set (Fin 4 → Point2D))
  (h1 : partition1 = {p1.vertices, p2.vertices})
  (h2 : partition2 = {p3.vertices, p4.vertices})
  : ∃ (c : Point2D),
    (∃ t : ℝ, c = lineThroughPoints (centroidParallelogram p1) (centroidParallelogram p2) t) ∧
    (∃ t : ℝ, c = lineThroughPoints (centroidParallelogram p3) (centroidParallelogram p4) t) ∧
    (∀ (i : Fin 6), (l.vertices i).x - c.x = 0 ∨ (l.vertices i).y - c.y = 0) :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_of_L_shape_l1203_120370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l1203_120327

/-- Two 2D vectors are parallel if their cross product is zero -/
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem parallel_vectors_lambda (l : ℝ) :
  let m : ℝ × ℝ := (l + 1, 1)
  let n : ℝ × ℝ := (4, -2)
  parallel m n → l = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l1203_120327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_hatchlings_l1203_120325

/-- Calculates the number of hatchlings for a turtle species -/
def hatchlings (eggs : ℕ) (rate : ℚ) (turtles : ℕ) : ℕ :=
  (↑eggs * rate * ↑turtles).floor.toNat

/-- Proves that the total number of hatchlings from all species is 112 -/
theorem total_hatchlings : 
  (hatchlings 25 (2/5) 3) + (hatchlings 20 (3/10) 6) +
  (hatchlings 10 (1/2) 4) + (hatchlings 15 (7/20) 5) = 112 := by
  sorry

#eval (hatchlings 25 (2/5) 3) + (hatchlings 20 (3/10) 6) +
      (hatchlings 10 (1/2) 4) + (hatchlings 15 (7/20) 5)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_hatchlings_l1203_120325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_specific_case_l1203_120313

/-- The chord length cut by a line intersecting a circle -/
noncomputable def chord_length (a b c : ℝ) : ℝ :=
  2 * Real.sqrt (1 - (|c| / Real.sqrt (a^2 + b^2))^2)

/-- The problem statement -/
theorem chord_length_specific_case :
  chord_length 3 (-4) 3 = 8/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_specific_case_l1203_120313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_robin_tournament_l1203_120342

/-- Represents the number of teams in a tournament. -/
def x : ℕ := sorry

/-- Represents the number of games in a single round-robin tournament. -/
def number_of_games (n : ℕ) : ℚ :=
  1/2 * n * (n - 1)

/-- Theorem stating that the number of games in a single round-robin tournament
    with x teams is equal to 21. -/
theorem round_robin_tournament :
  number_of_games x = 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_robin_tournament_l1203_120342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bromine_weight_calculation_l1203_120376

/-- The atomic weight of aluminum in atomic mass units (amu) -/
noncomputable def aluminum_weight : ℝ := 26.98

/-- The number of aluminum atoms in the compound -/
def num_aluminum : ℕ := 1

/-- The number of bromine atoms in the compound -/
def num_bromine : ℕ := 3

/-- The total molecular weight of the compound in atomic mass units (amu) -/
noncomputable def total_weight : ℝ := 267

/-- The atomic weight of bromine in atomic mass units (amu) -/
noncomputable def bromine_weight : ℝ := (total_weight - num_aluminum * aluminum_weight) / num_bromine

/-- Theorem stating that the calculated bromine weight is approximately 80.007 amu -/
theorem bromine_weight_calculation :
  ‖bromine_weight - 80.007‖ < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bromine_weight_calculation_l1203_120376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_chain_g_maximum_g_equal_implies_sum_greater_l1203_120362

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := (Real.log x) / x

-- State the theorems
theorem inequality_chain (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + b) / 2 > (a - b) / (Real.log a - Real.log b) ∧
  (a - b) / (Real.log a - Real.log b) > Real.sqrt (a * b) :=
by sorry

theorem g_maximum :
  ∃ (x : ℝ), x > 0 ∧ g x = (1 : ℝ) / Real.exp 1 ∧
  ∀ (y : ℝ), y > 0 → g y ≤ g x :=
by sorry

theorem g_equal_implies_sum_greater (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  g a = g b → a + b > 2 * Real.exp 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_chain_g_maximum_g_equal_implies_sum_greater_l1203_120362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_architecture_better_than_logistics_l1203_120390

-- Define the industry type
inductive Industry
| Mechanical
| Chemistry
| Computer
| Architecture
| Logistics

-- Define functions for applicants and openings
def applicants : Industry → ℕ := sorry
def openings : Industry → ℕ := sorry

-- Define the employment situation ratio
def employment_ratio (i : Industry) : ℚ :=
  (applicants i : ℚ) / (openings i : ℚ)

-- State the theorem
theorem architecture_better_than_logistics :
  (applicants Industry.Logistics > applicants Industry.Architecture) →
  (openings Industry.Architecture > openings Industry.Logistics) →
  employment_ratio Industry.Architecture < employment_ratio Industry.Logistics :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_architecture_better_than_logistics_l1203_120390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_slopes_constant_l1203_120331

noncomputable section

-- Define the ellipse C
def C (x y : ℝ) : Prop := x^2/8 + y^2/4 = 1

-- Define point N
def N : ℝ × ℝ := (0, 2)

-- Define point P
def P : ℝ × ℝ := (-1, -2)

-- Define a line passing through P
noncomputable def line_through_P (k : ℝ) (x : ℝ) : ℝ := k * (x + 1) - 2

-- Define the slope of a line passing through N and another point
noncomputable def slope_from_N (x y : ℝ) : ℝ := (y - N.2) / (x - N.1)

-- Theorem statement
theorem sum_of_slopes_constant 
  (k : ℝ) 
  (A B : ℝ × ℝ) 
  (hA : C A.1 A.2) 
  (hB : C B.1 B.2) 
  (hAN : A ≠ N) 
  (hBN : B ≠ N) 
  (hAl : A.2 = line_through_P k A.1) 
  (hBl : B.2 = line_through_P k B.1) :
  slope_from_N A.1 A.2 + slope_from_N B.1 B.2 = 4 :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_slopes_constant_l1203_120331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_acute_angles_l1203_120356

theorem triangle_acute_angles : 
  ∀ (a b c : Real), a > 0 → b > 0 → c > 0 → a + b + c = 180 →
  (if a < 90 then 1 else 0) + (if b < 90 then 1 else 0) + (if c < 90 then 1 else 0) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_acute_angles_l1203_120356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_greater_than_kx_implies_k_bound_l1203_120379

theorem sin_greater_than_kx_implies_k_bound (k : ℝ) :
  (∀ x : ℝ, 0 < x → x < π / 2 → Real.sin x > k * x) → k ≤ 2 / π := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_greater_than_kx_implies_k_bound_l1203_120379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_value_implies_m_l1203_120355

-- Define the function f(x) as noncomputable
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.exp x - m / x

-- State the theorem
theorem minimum_value_implies_m (m : ℝ) :
  (∀ x ∈ Set.Icc 1 2, f m x ≥ 1) ∧ (∃ x ∈ Set.Icc 1 2, f m x = 1) →
  m = Real.exp 1 - 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_value_implies_m_l1203_120355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_person_takes_twelve_days_l1203_120391

/-- The number of days it takes for the second person to finish the job -/
noncomputable def second_person_days (total_days : ℝ) (first_person_days : ℝ) : ℝ :=
  1 / (1 / total_days - 1 / first_person_days)

/-- Theorem stating that given the conditions, the second person takes 12 days to finish the job -/
theorem second_person_takes_twelve_days :
  second_person_days 8 24 = 12 := by
  -- Unfold the definition of second_person_days
  unfold second_person_days
  -- Simplify the expression
  simp
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_person_takes_twelve_days_l1203_120391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beautiful_ellipse_slope_product_l1203_120385

noncomputable def golden_ratio : ℝ := (Real.sqrt 5 - 1) / 2

noncomputable def beautiful_ellipse (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 ^ 2 / a ^ 2) + (p.2 ^ 2 / b ^ 2) = 1}

noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - (b ^ 2 / a ^ 2))

theorem beautiful_ellipse_slope_product (a b : ℝ) (h_ab : a > b ∧ b > 0) :
  let C := beautiful_ellipse a b
  let e := eccentricity a b
  ∀ P ∈ C, P ≠ (-a, 0) ∧ P ≠ (a, 0) →
    let k₁ := (P.2 / (P.1 + a))
    let k₂ := (P.2 / (P.1 - a))
    e = golden_ratio →
    k₁ * k₂ = (1 - Real.sqrt 5) / 2 := by
  sorry

#check beautiful_ellipse_slope_product

end NUMINAMATH_CALUDE_ERRORFEEDBACK_beautiful_ellipse_slope_product_l1203_120385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_in_terms_of_cos_2theta_l1203_120341

theorem sin_plus_cos_in_terms_of_cos_2theta (θ b : ℝ) 
  (h1 : 0 < θ ∧ θ < π/2)  -- θ is an acute angle
  (h2 : Real.cos (2*θ) = b) :  -- cos(2θ) = b
  Real.sin θ + Real.cos θ = Real.sqrt ((1 - b)/2) + Real.sqrt ((1 + b)/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_in_terms_of_cos_2theta_l1203_120341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_day_collection_l1203_120349

/-- Represents the number of cans collected on a given day. -/
def cans_collected (day : ℕ) : ℕ := sorry

/-- The sequence of can collections follows a linear pattern. -/
axiom linear_pattern : ∃ (a b : ℕ), ∀ (day : ℕ), cans_collected day = a * day + b

/-- Zachary collected 4 cans on the first day. -/
axiom first_day : cans_collected 1 = 4

/-- Zachary collected 14 cans on the third day. -/
axiom third_day : cans_collected 3 = 14

/-- Zachary will collect 34 cans on the seventh day. -/
axiom seventh_day : cans_collected 7 = 34

/-- Theorem: Given the conditions, Zachary must have collected 9 cans on the second day. -/
theorem second_day_collection :
  cans_collected 2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_day_collection_l1203_120349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptote_f_l1203_120315

/-- The function f(x) = (2x - 3) / (4x + 5) -/
noncomputable def f (x : ℝ) : ℝ := (2 * x - 3) / (4 * x + 5)

/-- The vertical asymptote of f occurs at x = -5/4 -/
theorem vertical_asymptote_f :
  Set.InjOn f {x : ℝ | x ≠ (-5/4)} ∧
  ∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x - (-5/4)| ∧ |x - (-5/4)| < δ → |f x| > 1/ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptote_f_l1203_120315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_expansion_l1203_120304

-- Define the binomial expansion function
def binomial_expansion (a b : ℚ → ℚ) (n : ℕ) : ℚ → ℚ := 
  fun x => (a x + b x)^n

-- Define the constant term function
def constant_term (expansion : ℚ → ℚ) : ℚ :=
  expansion 0

-- Define the specific terms in our expansion
def a (x : ℚ) : ℚ := 3 * x
def b (x : ℚ) : ℚ := 2 / (5 * x)

-- State the theorem
theorem constant_term_of_expansion :
  constant_term (binomial_expansion a b 8) = 5670000 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_expansion_l1203_120304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_one_fourth_of_sixteen_equals_negative_two_l1203_120324

-- Define the logarithm function for base 1/4
noncomputable def log_base_one_fourth (x : ℝ) : ℝ := 
  Real.log x / Real.log (1/4)

-- State the theorem
theorem log_one_fourth_of_sixteen_equals_negative_two :
  log_base_one_fourth 16 = -2 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_one_fourth_of_sixteen_equals_negative_two_l1203_120324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_length_EC_l1203_120348

theorem min_length_EC (a b : ℝ) (ρ₁ ρ₂ ρ₃ : ℝ) :
  let P : ℝ → ℝ := λ x ↦ x^3 + a*x^2 + b*x + 1
  (P ρ₁ = 0) ∧ (P ρ₂ = 0) ∧ (P ρ₃ = 0) →
  |ρ₁| < |ρ₂| ∧ |ρ₂| < |ρ₃| →
  let A : ℝ × ℝ := (0, 1)
  let B : ℝ × ℝ := (ρ₁, 0)
  let C : ℝ × ℝ := (ρ₂, 0)
  let D : ℝ × ℝ := (ρ₃, 0)
  let E : ℝ × ℝ := (0, -1/ρ₂)
  let EC : ℝ := Real.sqrt ((ρ₂ - 0)^2 + (0 - (-1/ρ₂))^2)
  (∀ ρ₁' ρ₂' ρ₃' a' b',
    let P' : ℝ → ℝ := λ x ↦ x^3 + a'*x^2 + b'*x + 1
    let A' : ℝ × ℝ := (0, 1)
    let B' : ℝ × ℝ := (ρ₁', 0)
    let C' : ℝ × ℝ := (ρ₂', 0)
    let D' : ℝ × ℝ := (ρ₃', 0)
    let E' : ℝ × ℝ := (0, -1/ρ₂')
    let EC' : ℝ := Real.sqrt ((ρ₂' - 0)^2 + (0 - (-1/ρ₂'))^2)
    (P' ρ₁' = 0) ∧ (P' ρ₂' = 0) ∧ (P' ρ₃' = 0) →
    |ρ₁'| < |ρ₂'| ∧ |ρ₂'| < |ρ₃'| →
    EC ≤ EC') →
  EC = Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_length_EC_l1203_120348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l1203_120334

-- Define the sets A and B
def A : Set ℝ := {y | ∃ x, y = |x - 1| + |x - 3|}
def B : Set ℝ := {x | ∃ y, y = Real.log (3*x - x^2)}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = Set.Icc 2 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l1203_120334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1203_120346

-- Define the piecewise function
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 - a else x - a + Real.log x

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∃ x y, -2 < x ∧ x < y ∧ y < 2 ∧ f a x = 0 ∧ f a y = 0 ∧
    ∀ z, -2 < z ∧ z < 2 ∧ f a z = 0 → z = x ∨ z = y) →
  0 ≤ a ∧ a < 2 + Real.log 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1203_120346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_section_length_l1203_120321

/-- Golden ratio -/
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

/-- Golden section point -/
def isGoldenSectionPoint (a b c : ℝ) : Prop :=
  (b - a) / (c - a) = φ

theorem golden_section_length (a b c : ℝ) :
  isGoldenSectionPoint a c b →
  b - a = 8 →
  c - a > b - c →
  c - a = 4 * (Real.sqrt 5 - 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_section_length_l1203_120321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_A_in_triangle_l1203_120312

theorem sin_A_in_triangle (a b c : ℝ) (A B C : ℝ) : 
  a = 2 → b = 3 → C = 2 * π / 3 → 
  a * Real.sin C = c * Real.sin A →
  c^2 = a^2 + b^2 - 2*a*b*(Real.cos C) →
  Real.sin A = Real.sqrt 57 / 19 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_A_in_triangle_l1203_120312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_inverse_proportion_l1203_120367

/-- Definition of an inverse proportion function -/
noncomputable def is_inverse_proportion (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x, x ≠ 0 → f x = k / x

/-- The function we want to prove is an inverse proportion -/
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (k^2 + 1) / x

/-- Theorem: f is an inverse proportion function -/
theorem f_is_inverse_proportion (k : ℝ) :
  is_inverse_proportion (f k) := by
  use k^2 + 1
  constructor
  · -- Prove k^2 + 1 ≠ 0
    sorry
  · -- Prove ∀ x, x ≠ 0 → f k x = (k^2 + 1) / x
    intro x hx
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_inverse_proportion_l1203_120367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_n_factorial_l1203_120386

/-- 
For a positive integer n, n! can be expressed as the product of n-4 consecutive positive integers 
if and only if there exists a positive integer a such that (n-4+a)!/a! = n!
-/
def can_be_expressed (n : ℕ) : Prop :=
  ∃ a : ℕ, a > 0 ∧ (n - 4 + a).factorial / a.factorial = n.factorial

/-- 4 is the largest positive integer satisfying the condition -/
theorem largest_n_factorial : 
  (can_be_expressed 4) ∧ (∀ m : ℕ, m > 4 → ¬(can_be_expressed m)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_n_factorial_l1203_120386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_for_point_l1203_120366

/-- Given an angle α whose terminal side passes through the point (-4, 3), prove that cos α = -4/5 -/
theorem cos_alpha_for_point (α : ℝ) : 
  (∃ (r : ℝ), r > 0 ∧ r * Real.cos α = -4 ∧ r * Real.sin α = 3) → 
  Real.cos α = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_for_point_l1203_120366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equation_solution_l1203_120360

theorem vector_equation_solution (x y : ℕ) (a b : ℝ × ℝ) :
  (a ≠ (0, 0) ∧ b ≠ (0, 0) ∧ ¬ ∃ (k : ℝ), a = k • b) →
  ((x + y - 1) • a + (x - y) • b = (0, 0)) →
  x + y = 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equation_solution_l1203_120360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_section_area_of_right_prism_l1203_120396

/-- Right prism with triangular base -/
structure RightPrism where
  base : Fin 3 → ℝ × ℝ
  height : ℝ

/-- The cross-section of a right prism by a plane -/
noncomputable def CrossSection (prism : RightPrism) (plane : Set (ℝ × ℝ × ℝ)) : Set (ℝ × ℝ × ℝ) :=
  sorry

/-- The area of a set in 3D space -/
noncomputable def Area (s : Set (ℝ × ℝ × ℝ)) : ℝ :=
  sorry

/-- The distance from a point to a plane in 3D space -/
noncomputable def DistanceToPlane (point : ℝ × ℝ × ℝ) (plane : Set (ℝ × ℝ × ℝ)) : ℝ :=
  sorry

/-- Angle between three points -/
noncomputable def Angle (a b c : ℝ × ℝ) : ℝ :=
  sorry

/-- Face of the prism -/
def FaceAA₁C₁C (prism : RightPrism) : Set (ℝ × ℝ × ℝ) :=
  sorry

/-- Vertex of the prism -/
def VertexB (prism : RightPrism) : ℝ × ℝ × ℝ :=
  sorry

/-- Vertex of the prism -/
def VertexC (prism : RightPrism) : ℝ × ℝ × ℝ :=
  sorry

/-- Diagonal of the prism -/
def DiagonalAB₁ (prism : RightPrism) : Set (ℝ × ℝ × ℝ) :=
  sorry

/-- Plane passes through the center of a face -/
def PassesThroughCenterOf (plane : Set (ℝ × ℝ × ℝ)) (face : Set (ℝ × ℝ × ℝ)) : Prop :=
  sorry

/-- Plane passes through a vertex -/
def PassesThroughVertex (plane : Set (ℝ × ℝ × ℝ)) (vertex : ℝ × ℝ × ℝ) : Prop :=
  sorry

/-- Plane is parallel to a line -/
def IsParallelTo (plane : Set (ℝ × ℝ × ℝ)) (line : Set (ℝ × ℝ × ℝ)) : Prop :=
  sorry

theorem cross_section_area_of_right_prism
  (prism : RightPrism)
  (plane : Set (ℝ × ℝ × ℝ))
  (h1 : Angle (prism.base 1) (prism.base 0) (prism.base 2) = π/2)
  (h2 : Angle (prism.base 2) (prism.base 0) (prism.base 1) = π/6)
  (h3 : PassesThroughCenterOf plane (FaceAA₁C₁C prism))
  (h4 : PassesThroughVertex plane (VertexB prism))
  (h5 : IsParallelTo plane (DiagonalAB₁ prism))
  (h6 : DistanceToPlane (VertexC prism) plane = 2)
  (h7 : Real.sqrt ((prism.base 1).1 - (prism.base 2).1)^2 + ((prism.base 1).2 - (prism.base 2).2)^2 = 4) :
  Area (CrossSection prism plane) = 6 / Real.sqrt 5 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_section_area_of_right_prism_l1203_120396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_of_given_hyperbola_l1203_120354

/-- The distance between the vertices of a hyperbola -/
noncomputable def distance_between_vertices (a b c d e f : ℝ) : ℝ :=
  let standard_form := λ (x y : ℝ) ↦ a * x^2 + b * x + c * y^2 + d * y + e
  Real.sqrt (2 * Real.sqrt (-c / a) * Real.sqrt ((b^2 / (4 * a) - f / a) / (1 - c / a)))

/-- Theorem: The distance between the vertices of the given hyperbola is √23 -/
theorem distance_of_given_hyperbola :
  distance_between_vertices 16 (-32) (-4) 8 (-11) 0 = Real.sqrt 23 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_of_given_hyperbola_l1203_120354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1203_120384

noncomputable def f (x : ℝ) : ℝ := (x^5 - 4*x^3 + 6*x^2 - 4*x + 1) / (x^2 - 9)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x < -3 ∨ (-3 < x ∧ x < 3) ∨ 3 < x} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1203_120384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l1203_120369

/-- The function f(x) = sin(2x) - 2cos²(x) - 1 -/
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) - 2 * (Real.cos x) ^ 2 - 1

/-- The smallest positive period of f(x) is 2π -/
theorem smallest_positive_period_of_f :
  ∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧
  (∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  p = 2 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l1203_120369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_power_of_two_l1203_120395

/-- A type representing a permutation of the numbers 1 to 10 -/
def Permutation := Fin 10 → Fin 10

/-- The product formula as described in the problem -/
def product (p : Permutation) : ℕ :=
  (p 0).val^(p 1).val * (p 1).val^(p 2).val * (p 2).val^(p 3).val * (p 3).val^(p 4).val * 
  (p 4).val^(p 5).val * (p 5).val^(p 6).val * (p 6).val^(p 7).val * 
  (p 7).val^(p 8).val * (p 8).val^(p 9).val * (p 9).val^(p 0).val

/-- The highest power of 2 that divides a natural number -/
def highestPowerOfTwo : ℕ → ℕ
  | 0 => 0
  | n + 1 => if (n + 1) % 2 = 0 then highestPowerOfTwo ((n + 1) / 2) + 1 else 0

/-- The main theorem -/
theorem max_power_of_two (p : Permutation) : 
  (∀ i j : Fin 10, i ≠ j → p i ≠ p j) →  -- p is injective (and thus bijective)
  highestPowerOfTwo (product p) ≤ 69 := by
  sorry

#eval highestPowerOfTwo 1024  -- Should output 10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_power_of_two_l1203_120395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_f_to_g_l1203_120340

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := Real.sin (3 * x + Real.pi / 6)

-- Define the target function
noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x)

-- Define the transformation
noncomputable def transform (x : ℝ) : ℝ := (2 / 3) * (x + Real.pi / 12)

-- State the theorem
theorem transform_f_to_g : ∀ x : ℝ, f (transform x) = g x := by
  intro x
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_f_to_g_l1203_120340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_primes_l1203_120333

/-- Represents a person traveling between locations A and B -/
structure Traveler where
  position : ℝ
  speed : ℝ

/-- The problem setup -/
structure TravelProblem where
  totalDistance : ℝ
  personA : Traveler
  personB : Traveler
  personC : Traveler
  p : ℕ
  q : ℕ
  r : ℕ

/-- Conditions of the problem -/
def validProblem (prob : TravelProblem) : Prop :=
  prob.totalDistance = 291 ∧
  prob.personA.position = 0 ∧
  prob.personB.position = 0 ∧
  prob.personC.position = prob.totalDistance ∧
  prob.personA.speed > 0 ∧
  prob.personB.speed > 0 ∧
  prob.personC.speed < 0 ∧
  Nat.Prime prob.p ∧
  Nat.Prime prob.q ∧
  Nat.Prime prob.r ∧
  (prob.p : ℝ) / prob.personB.speed = (prob.totalDistance - (prob.p : ℝ)) / (-prob.personC.speed) ∧
  (prob.q : ℝ) / prob.personA.speed = (prob.p : ℝ) / prob.personB.speed ∧
  (prob.r : ℝ) / prob.personB.speed = prob.totalDistance / (prob.personA.speed - prob.personC.speed)

/-- The theorem to prove -/
theorem sum_of_primes (prob : TravelProblem) (h : validProblem prob) : 
  prob.p + prob.q + prob.r = 221 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_primes_l1203_120333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l1203_120345

/-- Calculates the length of a train given the speeds of two trains, the length of one train,
    and the time it takes for them to cross each other. -/
noncomputable def calculate_train_length (speed1 speed2 : ℝ) (length1 time : ℝ) : ℝ :=
  let relative_speed := (speed1 + speed2) * (5 / 18)  -- Convert km/hr to m/s
  relative_speed * time - length1

/-- Theorem stating that under given conditions, the length of the second train
    is approximately 189.97 meters. -/
theorem train_length_calculation (speed1 speed2 length1 time : ℝ) 
  (h1 : speed1 = 60) 
  (h2 : speed2 = 40) 
  (h3 : length1 = 140) 
  (h4 : time = 11.879049676025918) :
  ∃ ε > 0, |calculate_train_length speed1 speed2 length1 time - 189.97| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l1203_120345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_calculation_l1203_120377

/-- Calculates simple interest -/
noncomputable def simple_interest (P : ℝ) (R : ℝ) (T : ℝ) : ℝ := P * R * T / 100

/-- Calculates compound interest -/
noncomputable def compound_interest (P : ℝ) (R : ℝ) (T : ℝ) : ℝ := P * (1 + R / 100) ^ T - P

/-- Theorem: If the difference between compound and simple interest
    for 2 years at 10% per annum is 20, then the principal is 2000 -/
theorem principal_calculation (P : ℝ) :
  compound_interest P 10 2 - simple_interest P 10 2 = 20 →
  P = 2000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_calculation_l1203_120377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_terminal_side_point_trig_expression_l1203_120357

theorem terminal_side_point_trig_expression (α : ℝ) : 
  1^2 + 3^2 = (1 / Real.cos α)^2 + (3 / Real.sin α)^2 →
  (Real.sin (π - α) - Real.sin (π/2 + α)) / (2 * Real.cos (α - 2*π)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_terminal_side_point_trig_expression_l1203_120357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_locus_and_constant_product_l1203_120317

/-- The ellipse parameters -/
def a : ℝ := 1  -- Assuming a positive real value for a

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2 / (1 + a^2) + y^2 = 1

/-- The right focus of the ellipse -/
def F : ℝ × ℝ := (a, 0)

/-- Point M on the x-axis -/
def M (m : ℝ) : ℝ × ℝ := (m, 0)

/-- Point N on the y-axis -/
def N (n : ℝ) : ℝ × ℝ := (0, n)

/-- The orthogonality condition -/
def orthogonal (m n : ℝ) : Prop := 
  let MN := (-(M m).1, (N n).2 - (M m).2)
  let NF := (F.1 - (N n).1, F.2 - (N n).2)
  MN.1 * NF.1 + MN.2 * NF.2 = 0

/-- Point P satisfying the given condition -/
def P (m n : ℝ) : ℝ × ℝ := (m, -2*n)

/-- The locus C of point P -/
def C (x y : ℝ) : Prop := y^2 = 4*a*x

/-- Line passing through F intersecting C at A and B -/
def lineF (k : ℝ) (x y : ℝ) : Prop := x = k*y + a

/-- Points S and T on the line y = -a -/
def S (y : ℝ) : ℝ × ℝ := (-a, y)
def T (y : ℝ) : ℝ × ℝ := (-a, y)

/-- The theorem to be proved -/
theorem ellipse_locus_and_constant_product (m n : ℝ) :
  ellipse (P m n).1 (P m n).2 ∧ 
  orthogonal m n ∧ 
  (∀ k y₁ y₂, lineF k (P m n).1 (P m n).2 → 
    let FS := (F.1 - (S y₂).1, F.2 - (S y₂).2)
    let FT := (F.1 - (T y₁).1, F.2 - (T y₁).2)
    FS.1 * FT.1 + FS.2 * FT.2 = 0) → 
  C (P m n).1 (P m n).2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_locus_and_constant_product_l1203_120317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crossSectionArea_eq_l1203_120383

/-- Regular quadrilateral pyramid with base side length a and lateral edge length b -/
structure RegularQuadPyramid where
  a : ℝ
  b : ℝ
  ha : a > 0
  hb : b > 0

/-- The area of the cross-section of a regular quadrilateral pyramid -/
noncomputable def crossSectionArea (p : RegularQuadPyramid) : ℝ := 
  5 * p.a * p.b * Real.sqrt 2 / 16

/-- Theorem stating the area of the cross-section -/
theorem crossSectionArea_eq (p : RegularQuadPyramid) : 
  crossSectionArea p = 5 * p.a * p.b * Real.sqrt 2 / 16 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_crossSectionArea_eq_l1203_120383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mower_team_size_l1203_120394

/-- Represents the area of the smaller meadow -/
noncomputable def S : ℝ := sorry

/-- Represents the number of mowers in the team -/
def N : ℕ := sorry

/-- The area one mower can cut in a day -/
noncomputable def mower_rate : ℝ := S / 2

/-- The area the whole team can cut in a day -/
noncomputable def team_rate : ℝ := 4 * S / 3

/-- Theorem stating that the number of mowers in the team is 8 -/
theorem mower_team_size : N = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mower_team_size_l1203_120394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1203_120311

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x - Real.pi / 6)

-- Define the interval
def interval : Set ℝ := Set.Icc (-Real.pi / 12) (25 * Real.pi / 36)

-- Theorem statement
theorem max_value_of_f :
  ∃ (max : ℝ), max = 2 ∧ ∀ x ∈ interval, f x ≤ max := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1203_120311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_is_17_l1203_120303

/-- Represents the number of minutes it takes to travel one mile on a given day -/
def minutes_per_mile (day : ℕ) : ℕ := 6 * day

/-- Calculates the distance traveled in one hour given the minutes per mile -/
def distance_traveled (minutes : ℕ) : ℚ := 60 / minutes

/-- Checks if a given distance is an integer -/
def is_integer_distance (distance : ℚ) : Prop := ∃ (n : ℕ), distance = n

/-- The set of valid days where the distance traveled is an integer -/
def valid_days : Finset ℕ := sorry

theorem total_distance_is_17 :
  (valid_days.sum (λ day => distance_traveled (minutes_per_mile day))).floor = 17 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_is_17_l1203_120303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_distance_sum_ellipse_tangent_line_l1203_120392

noncomputable def parabola (x y : ℝ) : Prop := y^2 = 2*x

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

theorem minimum_distance_sum :
  ∃ (min : ℝ), min = 6 ∧ 
  ∀ (x y : ℝ), parabola x y → 
    distance x y 3 6 + distance 0 y x y ≥ min :=
sorry

-- Theorem for the ellipse condition
theorem ellipse_tangent_line :
  ∀ (x y : ℝ), x^2/4 + y^2/3 = 1 →
  ∃ (l : ℝ → ℝ → Prop), l x y ↔ 3*x + 4*y - 7 = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_distance_sum_ellipse_tangent_line_l1203_120392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_expression_eval_l1203_120320

theorem trig_expression_eval :
  3 * Real.sin (π / 3) - 2 * Real.cos (π / 6) + 3 * Real.tan (π / 3) - 4 * (1 / Real.tan (π / 2)) = (7 * Real.sqrt 3) / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_expression_eval_l1203_120320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_15_simplest_l1203_120359

/-- A quadratic radical is considered simpler if it cannot be simplified further into a product of an integer and a square root of a prime number. -/
def IsSimplestQuadraticRadical (x : ℝ) (options : List ℝ) : Prop :=
  x ∈ options ∧ ∀ y ∈ options, ¬(∃ (a : ℤ) (b : ℕ), y = a * Real.sqrt b ∧ b.Prime ∧ y ≠ x)

/-- The given options for quadratic radicals -/
noncomputable def quadraticRadicals : List ℝ := [Real.sqrt 12, Real.sqrt 15, Real.sqrt 8, Real.sqrt (1/2)]

/-- Theorem stating that √15 is the simplest quadratic radical among the given options -/
theorem sqrt_15_simplest : IsSimplestQuadraticRadical (Real.sqrt 15) quadraticRadicals := by
  sorry

#check sqrt_15_simplest

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_15_simplest_l1203_120359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_quadrilateral_pyramid_volume_l1203_120300

/-- The volume of a regular quadrilateral pyramid with lateral edge length b and circumscribed sphere radius R -/
noncomputable def pyramid_volume (b R : ℝ) : ℝ := (1/3) * b^2 * (b^2 / (2*R))

/-- Theorem stating that the volume of a regular quadrilateral pyramid with lateral edge length b and circumscribed sphere radius R is equal to (1/3) * b^2 * (b^2 / (2R)) -/
theorem regular_quadrilateral_pyramid_volume (b R : ℝ) (h1 : b > 0) (h2 : R > 0) :
  pyramid_volume b R = (1/3) * b^2 * (b^2 / (2*R)) :=
by
  -- Unfold the definition of pyramid_volume
  unfold pyramid_volume
  -- The equality holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_quadrilateral_pyramid_volume_l1203_120300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1203_120332

/-- The number of days B takes to complete the work independently -/
noncomputable def b_days : ℝ := 18

/-- The number of days C takes to complete the work independently -/
noncomputable def c_days : ℝ := 12

/-- The work rate of B (fraction of work completed per day) -/
noncomputable def work_rate_b : ℝ := 1 / b_days

/-- The work rate of C (fraction of work completed per day) -/
noncomputable def work_rate_c : ℝ := 1 / c_days

/-- The work rate of A (fraction of work completed per day) -/
noncomputable def work_rate_a : ℝ := 2 * work_rate_b

/-- The combined work rate of A, B, and C -/
noncomputable def combined_work_rate : ℝ := work_rate_a + work_rate_b + work_rate_c

/-- Theorem stating that A, B, and C can complete the work together in 4 days -/
theorem work_completion_time : (1 / combined_work_rate) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1203_120332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l1203_120382

noncomputable def f (x : ℝ) := Real.sin (x / 2)

theorem f_increasing_on_interval : 
  StrictMonoOn f (Set.Icc (-Real.pi) Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l1203_120382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_collinear_k_l1203_120388

-- Define the ellipse
noncomputable def ellipse (x y : ℝ) : Prop := x^2/2 + y^2 = 1

-- Define the line
noncomputable def line (k x y : ℝ) : Prop := y = k*x + Real.sqrt 2

-- Define the intersection points
noncomputable def A : ℝ × ℝ := (Real.sqrt 2, 0)
noncomputable def B : ℝ × ℝ := (0, 1)

-- Define the vector AB
noncomputable def vecAB : ℝ × ℝ := (-Real.sqrt 2, 1)

-- Define the sum of vectors OP and OQ
def sumOPOQ (P Q : ℝ × ℝ) : ℝ × ℝ := (P.1 + Q.1, P.2 + Q.2)

-- Theorem statement
theorem no_collinear_k : 
  ¬∃ (k : ℝ), ∃ (P Q : ℝ × ℝ),
    ellipse P.1 P.2 ∧ 
    ellipse Q.1 Q.2 ∧ 
    line k P.1 P.2 ∧ 
    line k Q.1 Q.2 ∧ 
    P ≠ Q ∧
    ∃ (t : ℝ), sumOPOQ P Q = (t * vecAB.1, t * vecAB.2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_collinear_k_l1203_120388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_l1203_120326

theorem calculation_proof :
  (∃ x : ℝ, x = Real.sqrt 2 / Real.sqrt (1/2) * Real.sqrt 8 - Real.sqrt 8 ∧ x = 2 * Real.sqrt 2) ∧
  (∃ y : ℝ, y = Real.sqrt 48 - Real.sqrt 27 + (-3 * Real.sqrt 2)^2 - 3 / Real.sqrt 3 ∧ y = 18) :=
by
  constructor
  · use 2 * Real.sqrt 2
    constructor
    · sorry
    · rfl
  · use 18
    constructor
    · sorry
    · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_l1203_120326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_value_of_a_l1203_120344

def U : Set ℕ := {1, 3, 5, 7}

def M (a : ℤ) : Set ℕ := {1, (|a - 5|:ℤ).toNat}

theorem value_of_a (a : ℤ) 
  (h1 : M a ⊆ U) 
  (h2 : (U \ M a) = {5, 7}) : 
  a = 2 ∨ a = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_value_of_a_l1203_120344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_laser_path_theorem_l1203_120319

/-- The total distance traveled by a laser beam with specific bounce points -/
noncomputable def laser_path_distance : ℝ :=
  let A : ℝ × ℝ := (4, 7)
  let B : ℝ × ℝ := (-4, 7)
  let C : ℝ × ℝ := (-4, -7)
  let D : ℝ × ℝ := (4, -7)
  let E : ℝ × ℝ := (9, 7)
  let dist (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  (dist A B) + (dist B C) + (dist C D) + (dist D E)

/-- Theorem stating that the laser path distance equals 30 + √221 -/
theorem laser_path_theorem : laser_path_distance = 30 + Real.sqrt 221 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_laser_path_theorem_l1203_120319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_f_l1203_120399

noncomputable def f (a x : ℝ) : ℝ := 2 * a * Real.sqrt x - 1 / x

theorem max_value_f (a : ℝ) :
  (∀ x ∈ Set.Ioo 0 1, f a x ≤ if a > -1 then 2 * a - 1 else -3 * (a^2)^(1/3)) ∧
  (∃ x ∈ Set.Ioo 0 1, f a x = if a > -1 then 2 * a - 1 else -3 * (a^2)^(1/3)) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_f_l1203_120399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_real_condition_implies_m_equals_one_l1203_120328

noncomputable def complex_number (m : ℝ) : ℂ := (1 + m * Complex.I) / (1 + Complex.I)

theorem real_condition_implies_m_equals_one (m : ℝ) :
  (complex_number m).im = 0 → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_real_condition_implies_m_equals_one_l1203_120328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l1203_120361

/-- Given an ellipse with equation x^2/a^2 + y^2/b^2 = 1, where a > b > 0,
    left focus F(-c, 0), left vertex A(-a, 0), upper vertex B(0, b),
    and the distance from F to line AB is 2b/√17,
    prove that the eccentricity of the ellipse is 1/3. -/
theorem ellipse_eccentricity (a b c : ℝ) (h1 : a > b) (h2 : b > 0) : 
  let F : ℝ × ℝ := (-c, 0)
  let A : ℝ × ℝ := (-a, 0)
  let B : ℝ × ℝ := (0, b)
  let distance_F_to_AB := 2 * b / Real.sqrt 17
  let eccentricity := c / a
  (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 → 
    distance_F_to_AB = |b * (-c) - a * 0 + a * b| / Real.sqrt (a^2 + b^2)) →
  eccentricity = 1 / 3 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l1203_120361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_a_range_l1203_120372

noncomputable section

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1 / Real.sqrt (x + 2) + Real.log (3 - x)

-- Define the set A as the domain of f
def A : Set ℝ := {x : ℝ | x + 2 > 0 ∧ 3 - x > 0}

-- Define the set B
def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 - a < 0}

-- Statement 1: The complement of A in ℝ
theorem complement_A : 
  (Set.univ \ A : Set ℝ) = Set.Iic (-2) ∪ Set.Ici 3 := by sorry

-- Statement 2: If A ∪ B = A, then a ≤ 4
theorem a_range (a : ℝ) : 
  A ∪ B a = A → a ≤ 4 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_a_range_l1203_120372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_trajectory_l1203_120314

-- Define the ellipse C
def C (x y : ℝ) : Prop := x^2 + 2*y^2 = 4

-- Define the line l
def l (x y t : ℝ) : Prop := y = 2*x + t

-- Define the intersection points A and B
def intersectionPoints (A B : ℝ × ℝ) (t : ℝ) : Prop :=
  C A.1 A.2 ∧ C B.1 B.2 ∧ l A.1 A.2 t ∧ l B.1 B.2 t ∧ A ≠ B

-- Define the midpoint M of AB
def midpointAB (A B M : ℝ × ℝ) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

-- State the theorem
theorem midpoint_trajectory :
  ∀ (A B M : ℝ × ℝ) (t : ℝ),
  intersectionPoints A B t →
  midpointAB A B M →
  M.2 = -1/4 * M.1 ∧ -4*Real.sqrt 2/3 < M.1 ∧ M.1 < 4*Real.sqrt 2/3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_trajectory_l1203_120314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l1203_120306

/-- Calculates the length of a train given its speed, the time it takes to pass a bridge, and the length of the bridge. -/
theorem train_length_calculation (train_speed_kmh : ℝ) (bridge_length_m : ℝ) (time_to_pass_s : ℝ) :
  train_speed_kmh = 35 →
  bridge_length_m = 115 →
  time_to_pass_s = 42.68571428571429 →
  ∃ (train_length_m : ℝ), 
    (train_length_m ≥ 299.9 ∧ train_length_m ≤ 300.1) ∧ 
    (train_length_m + bridge_length_m ≥ (train_speed_kmh * 1000 / 3600) * time_to_pass_s - 0.1 ∧
     train_length_m + bridge_length_m ≤ (train_speed_kmh * 1000 / 3600) * time_to_pass_s + 0.1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l1203_120306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_real_solutions_composite_l1203_120365

-- Define real polynomials p and q
variable (p q : ℝ → ℝ)

-- Define the theorem
theorem no_real_solutions_composite 
  (h1 : ∀ x, p (q x) = q (p x)) 
  (h2 : ∀ x, p x ≠ q x) : 
  ∀ x, p (p x) ≠ q (q x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_real_solutions_composite_l1203_120365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_value_l1203_120378

noncomputable def z : ℝ := (Real.log 4 / Real.log 3) * (Real.log 5 / Real.log 4) * 
             (Real.log 6 / Real.log 5) * (Real.log 7 / Real.log 6) * 
             (Real.log 8 / Real.log 7) * (Real.log 9 / Real.log 8) * 
             (Real.log 10 / Real.log 9) * (Real.log 11 / Real.log 10) * 
             (Real.log 12 / Real.log 11) * (Real.log 13 / Real.log 12) * 
             (Real.log 14 / Real.log 13) * (Real.log 15 / Real.log 14) * 
             (Real.log 16 / Real.log 15) * (Real.log 17 / Real.log 16) * 
             (Real.log 18 / Real.log 17) * (Real.log 19 / Real.log 18) * 
             (Real.log 20 / Real.log 19)

theorem z_value : z = Real.log 20 / Real.log 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_value_l1203_120378
