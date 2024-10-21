import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_roots_real_roots_real_roots_y_l1276_127618

-- Define the quadratic equation
def quadratic (x y : ℝ) : ℝ := x^2 - 3*x*y + y^2 + 2*x - 9*y + 1

-- Theorem for equal roots
theorem equal_roots (y : ℝ) : 
  (∀ x, quadratic x y = 0 → (∃! x', quadratic x' y = 0)) ↔ 
  (y = 0 ∨ y = -24/5) := by sorry

-- Theorem for real roots
theorem real_roots (y : ℝ) :
  (∃ x : ℝ, quadratic x y = 0) ↔ (y ≥ 0 ∨ y ≤ -24/5) := by sorry

-- Define the quadratic equation solved for y
def quadratic_y (x : ℝ) : ℝ → Prop := λ y ↦ y^2 - (3*x + 9)*y + (x^2 + 2*x + 1) = 0

-- Theorem for real roots when solved for y
theorem real_roots_y (x : ℝ) :
  (∃ y : ℝ, quadratic_y x y) ↔ (x ≤ -7 ∨ x ≥ -11/5) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_roots_real_roots_real_roots_y_l1276_127618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_is_120_degrees_l1276_127679

-- Define a triangle with side lengths a, b, c
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_positive : 0 < a ∧ 0 < b ∧ 0 < c
  h_triangle : a + b > c ∧ b + c > a ∧ c + a > b

-- Define the angle C in radians
noncomputable def angle_C (t : Triangle) : ℝ := Real.arccos ((t.a^2 + t.b^2 - t.c^2) / (2 * t.a * t.b))

-- The main theorem
theorem angle_C_is_120_degrees {t : Triangle} 
  (h : (t.a + t.b - t.c) * (t.a + t.b + t.c) = t.a * t.b) : 
  angle_C t = 2 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_is_120_degrees_l1276_127679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_fourth_quadrant_l1276_127668

noncomputable def z : ℂ := (1 : ℂ) / (1 + Complex.I) + Complex.I^3

theorem z_in_fourth_quadrant : 
  Real.sign z.re = 1 ∧ Real.sign z.im = -1 :=
by
  -- Simplify z
  have h : z = 1/2 - (3/2)*Complex.I := by
    -- Proof of simplification goes here
    sorry
  
  -- Show that real part is positive
  have h_re : z.re = 1/2 := by
    -- Proof that real part is 1/2 goes here
    sorry
  have pos_re : Real.sign z.re = 1 := by
    -- Proof that sign of 1/2 is 1 goes here
    sorry

  -- Show that imaginary part is negative
  have h_im : z.im = -3/2 := by
    -- Proof that imaginary part is -3/2 goes here
    sorry
  have neg_im : Real.sign z.im = -1 := by
    -- Proof that sign of -3/2 is -1 goes here
    sorry

  -- Combine the results
  exact ⟨pos_re, neg_im⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_fourth_quadrant_l1276_127668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_f_range_on_interval_g_nonnegative_condition_l1276_127670

noncomputable def f (x : ℝ) : ℝ := -2^x / (2^x + 1)

noncomputable def g (a x : ℝ) : ℝ := a/2 + f x

theorem f_decreasing : ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ > f x₂ := by sorry

theorem f_range_on_interval : 
  (∀ x ∈ Set.Icc 1 2, f x ≥ -4/5 ∧ f x ≤ -2/3) ∧ 
  (∃ x₁ ∈ Set.Icc 1 2, f x₁ = -4/5) ∧ 
  (∃ x₂ ∈ Set.Icc 1 2, f x₂ = -2/3) := by sorry

theorem g_nonnegative_condition : 
  (∀ x ∈ Set.Icc 1 2, g (8/5) x ≥ 0) ∧ 
  (∀ ε > 0, ∃ x ∈ Set.Icc 1 2, g (8/5 - ε) x < 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_f_range_on_interval_g_nonnegative_condition_l1276_127670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_explicit_form_f_positive_condition_F_increasing_condition_l1276_127605

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * x + 2

-- Define the function h
noncomputable def h (a : ℝ) (x : ℝ) : ℝ := a * (Real.log x)^2 - Real.log x + 3

-- Define the function F
def F (a : ℝ) (x : ℝ) : ℝ := |f a x|

-- Theorem 1
theorem h_explicit_form (a : ℝ) :
  ∀ x, h a (10^x) = f a x + x + 1 := by
  sorry

-- Theorem 2
theorem f_positive_condition (a : ℝ) :
  (∀ x, x ∈ Set.Icc 1 2 → f a x > 0) → a > 1/2 := by
  sorry

-- Theorem 3
theorem F_increasing_condition (a : ℝ) :
  (∀ x₁ x₂, x₁ ∈ Set.Icc 1 2 → x₂ ∈ Set.Icc 1 2 → x₁ < x₂ → 
    (F a x₂ - F a x₁) / (x₂ - x₁) > 0) →
  a ∈ Set.Iic 0 ∪ Set.Ici 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_explicit_form_f_positive_condition_F_increasing_condition_l1276_127605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_product_inequality_l1276_127654

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x * Real.exp x - 1/2 * x^2 - x

theorem root_product_inequality {a x₁ x₂ : ℝ} (ha : a > 0) 
  (hx : x₁ ≠ x₂ ∧ f a x₁ = Real.log x₁ - 1/2 * x₁^2 ∧ f a x₂ = Real.log x₂ - 1/2 * x₂^2) :
  x₁ * x₂ > Real.exp (2 - x₁ - x₂) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_product_inequality_l1276_127654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_change_l1276_127697

theorem rectangle_area_change (L B : ℝ) (h1 : L > 0) (h2 : B > 0) : 
  (3 * L * B / 2 - L * B) / (L * B) * 100 = 50 := by
  -- Simplify the expression
  have h3 : L * B ≠ 0 := by
    apply ne_of_gt
    exact mul_pos h1 h2
  
  -- Perform algebraic manipulations
  calc (3 * L * B / 2 - L * B) / (L * B) * 100
     = ((3 / 2 * L * B - L * B) / (L * B)) * 100 := by ring_nf
   _ = ((3 / 2 - 1) * L * B / (L * B)) * 100 := by ring_nf
   _ = (1 / 2) * 100 := by
       field_simp [h3]
       ring
   _ = 50 := by norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_change_l1276_127697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_values_of_c_l1276_127617

theorem distinct_values_of_c : ∃ (S : Finset ℂ), (∀ c : ℂ, 
  (∃ p q : ℂ, p ≠ q ∧ ∀ z : ℂ, (z - p) * (z - q) = (z - c*p) * (z - c*q)) 
  ↔ c ∈ S) ∧ Finset.card S = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_values_of_c_l1276_127617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_cube_root_solution_l1276_127682

/-- The positive solution to the nested cube root equation. -/
theorem nested_cube_root_solution :
  ∃ (x : ℝ), x > 0 ∧
  (∃ (y z : ℝ),
    y = (3 * x * y) ^ (1/3) ∧
    z = (x + 1 + z) ^ (1/3) ∧
    y = z) ∧
  x = (4 + 2 * Real.sqrt 3) / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_cube_root_solution_l1276_127682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l1276_127672

theorem expression_evaluation (b : ℝ) (h : b = 2) :
  (3 * b^(-2 : ℝ) + b^(-2 : ℝ) / 3) / b^2 = 5 / 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l1276_127672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_and_circles_area_l1276_127658

/-- A square with four inscribed circles -/
structure SquareWithCircles where
  /-- The radius of each inscribed circle -/
  r : ℝ
  /-- The circles are tangent to each other and the sides of the square -/
  tangent : r > 0

/-- The area of the square -/
def square_area (s : SquareWithCircles) : ℝ := 16 * s.r^2

/-- The combined area of the four circles -/
noncomputable def circles_area (s : SquareWithCircles) : ℝ := 4 * Real.pi * s.r^2

theorem square_and_circles_area (s : SquareWithCircles) (h : s.r = 6) :
  square_area s = 576 ∧ circles_area s = 4 * Real.pi * 36 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_and_circles_area_l1276_127658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roof_area_l1276_127622

theorem roof_area (width length : ℝ) : 
  width > 0 →
  length > 0 →
  length = 8 * width →
  length - width = 66 →
  ∃ area : ℝ, abs (area - 711) < 1 ∧ area = width * length :=
by
  intro hw hl h1 h2
  use width * length
  constructor
  · -- Proof that the area is approximately 711
    sorry
  · -- Proof that area = width * length
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_roof_area_l1276_127622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_scores_above_90_l1276_127629

/-- Represents a normal distribution with mean μ and standard deviation σ -/
structure NormalDistribution where
  μ : ℝ
  σ : ℝ
  σ_pos : σ > 0

/-- Probability that a value from the distribution falls within the given range -/
noncomputable def probability (d : NormalDistribution) (lower upper : ℝ) : ℝ := sorry

/-- Number of samples above a given value -/
noncomputable def samplesAbove (d : NormalDistribution) (value : ℝ) (totalSamples : ℕ) : ℕ := sorry

/-- Approximation relation -/
noncomputable def approximatelyEqual (x y : ℝ) : Prop := sorry

notation:50 a " ≈ " b => approximatelyEqual a b

theorem exam_scores_above_90 (d : NormalDistribution) 
  (h_mean : d.μ = 80)
  (h_std : d.σ = 5)
  (h_prob1 : probability d (d.μ - d.σ) (d.μ + d.σ) = 0.6827)
  (h_prob2 : probability d (d.μ - 2 * d.σ) (d.μ + 2 * d.σ) = 0.9544)
  (totalStudents : ℕ)
  (h_total : totalStudents = 1000) :
  (samplesAbove d 90 totalStudents : ℝ) ≈ 23 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_scores_above_90_l1276_127629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_f_value_third_quadrant_f_value_specific_angle_l1276_127616

noncomputable def f (α : ℝ) : ℝ :=
  (Real.sin (Real.pi + α) * Real.cos (2 * Real.pi - α) * Real.tan (-α)) /
  (Real.tan (-Real.pi - α) * Real.sin (-Real.pi - α))

theorem f_simplification (α : ℝ) : f α = -Real.cos α := by sorry

theorem f_value_third_quadrant (α : ℝ)
  (h1 : α ∈ Set.Icc Real.pi (3 * Real.pi / 2))  -- Third quadrant
  (h2 : Real.sin (α - Real.pi) = 1 / 5) :
  f α = 2 * Real.sqrt 6 / 5 := by sorry

theorem f_value_specific_angle :
  f (-31 * Real.pi / 5) = -Real.cos (Real.pi / 5) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_f_value_third_quadrant_f_value_specific_angle_l1276_127616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_quadrilateral_l1276_127677

-- Define the triangle ABC
structure Triangle (A B C : ℝ × ℝ) : Prop where
  right_angle : (C.1 - A.1) * (B.1 - A.1) + (C.2 - A.2) * (B.2 - A.2) = 0

-- Define the quadrilateral ADFC
structure Quadrilateral (A D F C : ℝ × ℝ) : Prop where
  in_triangle : ∃ B : ℝ × ℝ, Triangle A B C
  perpendicular : (F.1 - D.1) * (A.1 - D.1) + (F.2 - D.2) * (A.2 - D.2) = 0
  on_side : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ F = (t * A.1 + (1 - t) * D.1, t * A.2 + (1 - t) * D.2)

-- Define the theorem
theorem area_of_quadrilateral 
  (A B C D F : ℝ × ℝ) 
  (quad : Quadrilateral A D F C) 
  (h1 : Real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2) = 2 * Real.sqrt ((D.1 - B.1)^2 + (D.2 - B.2)^2))
  (h2 : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 24)
  (h3 : Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2) = 10) :
  ∃ (area : ℝ), area = 10 * Real.sqrt 119 - 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_quadrilateral_l1276_127677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l1276_127628

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 3 * x + Real.sqrt 3 * y - 4 = 0

-- Define the slope of the line
noncomputable def line_slope : ℝ := -Real.sqrt 3

-- Define the inclination angle
noncomputable def inclination_angle : ℝ := 120 * (Real.pi / 180)

-- Theorem statement
theorem line_inclination_angle :
  ∀ x y : ℝ, line_equation x y →
  Real.tan inclination_angle = -line_slope := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l1276_127628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_basket_count_l1276_127662

/-- Represents the number of ways to choose fruits of a specific type -/
def choiceCount (fruitCount : Nat) : Nat :=
  fruitCount + 1

/-- Calculates the total number of fruit basket combinations -/
def totalCombinations (apples oranges bananas : Nat) : Nat :=
  choiceCount apples * choiceCount oranges * choiceCount bananas

/-- Theorem: The number of fruit baskets with at least one fruit -/
theorem fruit_basket_count (apples oranges bananas : Nat) 
  (h1 : apples = 3) 
  (h2 : oranges = 7) 
  (h3 : bananas = 4) :
  totalCombinations apples oranges bananas - 1 = 159 := by
  sorry

#eval totalCombinations 3 7 4 - 1

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_basket_count_l1276_127662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_from_medians_median_formula_l1276_127676

theorem right_triangle_from_medians (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 
    x^2 = (16 * b^2 - 4 * a^2) / 15 ∧
    y^2 = (16 * a^2 - 4 * b^2) / 15 :=
by
  -- We assert the existence of x and y
  use (((16 * b^2 - 4 * a^2) / 15).sqrt), (((16 * a^2 - 4 * b^2) / 15).sqrt)
  
  -- We need to prove four conditions
  apply And.intro
  · sorry  -- Proof that x > 0
  apply And.intro
  · sorry  -- Proof that y > 0
  apply And.intro
  · sorry  -- Proof that x^2 = (16 * b^2 - 4 * a^2) / 15
  · sorry  -- Proof that y^2 = (16 * a^2 - 4 * b^2) / 15

-- Additional helper theorem
theorem median_formula (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x^2 / 4 + y^2).sqrt^2 + (x^2 + y^2 / 4).sqrt^2 = (5 * (x^2 + y^2)) / 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_from_medians_median_formula_l1276_127676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_angle_intersection_exists_l1276_127614

/-- A line in a plane --/
structure Line where
  -- Define a line using two points it passes through
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ
  ne : point1 ≠ point2

/-- Two lines are parallel if they don't intersect --/
def parallel (l1 l2 : Line) : Prop :=
  ∀ p : ℝ × ℝ, (p = l1.point1 ∨ p = l1.point2) → (p = l2.point1 ∨ p = l2.point2) → l1 = l2

/-- The angle between two lines --/
noncomputable def angle_between (l1 l2 : Line) : ℝ :=
  sorry -- Definition of angle between two lines

/-- A line intersects two other lines at equal angles --/
def intersects_at_equal_angles (l : Line) (l1 l2 : Line) : Prop :=
  ∃ p1 p2 : ℝ × ℝ, p1 ≠ p2 ∧
    (p1 = l1.point1 ∨ p1 = l1.point2) ∧
    (p2 = l2.point1 ∨ p2 = l2.point2) ∧
    angle_between l l1 = angle_between l l2

/-- The angle bisector of two lines --/
noncomputable def angle_bisector (l1 l2 : Line) : Line :=
  sorry -- Definition of angle bisector

theorem equal_angle_intersection_exists (P : ℝ × ℝ) (L1 L2 : Line) :
  ∃ L : Line, (L.point1 = P ∨ L.point2 = P) ∧ intersects_at_equal_angles L L1 L2 :=
by
  sorry -- Proof of the theorem


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_angle_intersection_exists_l1276_127614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_values_count_l1276_127621

def is_valid_assignment (a b c d e : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e ∧
  a ∈ ({1, 2, 3, 4, 5} : Finset ℕ) ∧
  b ∈ ({1, 2, 3, 4, 5} : Finset ℕ) ∧
  c ∈ ({1, 2, 3, 4, 5} : Finset ℕ) ∧
  d ∈ ({1, 2, 3, 4, 5} : Finset ℕ) ∧
  e ∈ ({1, 2, 3, 4, 5} : Finset ℕ)

def expression_value (a b c d e : ℕ) : ℤ :=
  (a * b - c : ℤ) + (d * e)

theorem unique_values_count :
  ∃! (S : Finset ℤ), 
    (∀ x ∈ S, ∃ a b c d e, is_valid_assignment a b c d e ∧ x = expression_value a b c d e) ∧
    (∀ a b c d e, is_valid_assignment a b c d e → expression_value a b c d e ∈ S) ∧
    S.card = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_values_count_l1276_127621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_l1276_127601

-- Define the focus point F
def F : ℝ × ℝ := (2, 0)

-- Define the directrix line
def directrix (x : ℝ) : Prop := x + 3 = 0

-- Define the distance function between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the distance from a point to the directrix
def distToDirectrix (p : ℝ × ℝ) : ℝ := |p.1 + 3|

-- Define the condition for point P
def satisfiesCondition (p : ℝ × ℝ) : Prop :=
  distance p F + 1 = distToDirectrix p

-- State the theorem
theorem trajectory_equation :
  ∀ p : ℝ × ℝ, satisfiesCondition p → p.2^2 = 8 * p.1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_l1276_127601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_couple_pairing_l1276_127638

-- Define the set of men and women
inductive Man : Type
| Jia : Man
| Yi : Man
| Bing : Man

inductive Woman : Type
| A : Woman
| B : Woman
| C : Woman

-- Define the age relation
def older_than : Man → Man → Prop := sorry

-- Define the friendship relation
def is_friend_of : Man → Man → Prop := sorry

-- Define the marriage relation
def married_to : Man → Woman → Prop := sorry

-- State the theorem
theorem couple_pairing 
  (h1 : ∀ m : Man, ∃! w : Woman, married_to m w)
  (h2 : ∀ w : Woman, ∃! m : Man, married_to m w)
  (h3 : ∃ m : Man, married_to m Woman.A ∧ is_friend_of m Man.Yi)
  (h4 : ∀ m : Man, married_to m Woman.A → ∀ m' : Man, m ≠ m' → older_than m' m)
  (h5 : ∃ m : Man, married_to m Woman.C ∧ older_than Man.Bing m) :
  married_to Man.Jia Woman.A ∧ 
  married_to Man.Yi Woman.C ∧ 
  married_to Man.Bing Woman.B :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_couple_pairing_l1276_127638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l1276_127681

-- Define the ellipse
noncomputable def ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 12 = 1

-- Define the hyperbola E
noncomputable def hyperbola (x y a b : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Define the right focus F of the ellipse
def right_focus : ℝ × ℝ := (2, 0)

-- Define the distance from F to the asymptote of E
noncomputable def distance_to_asymptote (a b : ℝ) : ℝ := 2 * b / Real.sqrt (b^2 + a^2)

-- State the theorem
theorem hyperbola_eccentricity_range (a b : ℝ) :
  a > 0 →
  b > 0 →
  distance_to_asymptote a b < Real.sqrt 3 →
  let e := Real.sqrt (1 + b^2 / a^2)
  1 < e ∧ e < 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l1276_127681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_example_l1276_127644

/-- The area of a quadrilateral with a given diagonal and two offsets -/
noncomputable def quadrilateralArea (diagonal : ℝ) (offset1 : ℝ) (offset2 : ℝ) : ℝ :=
  (1/2) * diagonal * offset1 + (1/2) * diagonal * offset2

/-- Theorem: The area of a quadrilateral with diagonal 22 cm and offsets 9 cm and 6 cm is 165 cm² -/
theorem quadrilateral_area_example : quadrilateralArea 22 9 6 = 165 := by
  -- Unfold the definition of quadrilateralArea
  unfold quadrilateralArea
  -- Simplify the expression
  simp
  -- Perform the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_example_l1276_127644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_in_inner_hexagon_l1276_127675

/-- Represents a regular hexagon -/
structure RegularHexagon where
  center : ℝ × ℝ
  sideLength : ℝ

/-- Represents a point in 2D space -/
def Point := ℝ × ℝ

/-- Represents the vertices of a regular hexagon -/
def RegularHexagon.vertices (h : RegularHexagon) : Set Point :=
  sorry

/-- Represents the interior of a regular hexagon -/
def RegularHexagon.interior (h : RegularHexagon) : Set Point :=
  sorry

/-- Represents that one hexagon is inscribed in another -/
def isInscribed (inner outer : RegularHexagon) : Prop :=
  ∀ p, p ∈ inner.vertices → p ∈ outer.interior

/-- The main theorem stating that the center of the outer hexagon lies inside the inner hexagon -/
theorem center_in_inner_hexagon (inner outer : RegularHexagon) 
  (h_inscribed : isInscribed inner outer)
  (h_side_length : inner.sideLength = outer.sideLength / 2) :
  outer.center ∈ inner.interior := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_in_inner_hexagon_l1276_127675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solutions_l1276_127655

theorem system_solutions :
  let S := {(x, y) : ℝ × ℝ | x^3 + y = 2*x ∧ y^3 + x = 2*y}
  S = {(0, 0), (1, 1), (-1, -1), (Real.sqrt 3, -Real.sqrt 3), (-Real.sqrt 3, Real.sqrt 3)} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solutions_l1276_127655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repetend_of_four_seventeenths_l1276_127623

/-- Represents the repetend of a rational number's decimal representation -/
def repetend : ℚ → ℕ := sorry

/-- Represents the decimal representation of a fraction n/d -/
def decimal_rep (n d : ℕ) : ℚ := (n : ℚ) / d

/-- The repetend in the decimal representation of 4/17 is 235294117647058823529411764705882352941176470588 -/
theorem repetend_of_four_seventeenths :
  repetend (decimal_rep 4 17) = 235294117647058823529411764705882352941176470588 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repetend_of_four_seventeenths_l1276_127623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2015_value_l1276_127642

def sequence_a : ℕ → ℚ
  | 0 => 3  -- We define a₀ = 3 to handle the base case
  | n + 1 => (sequence_a n - 1) / sequence_a n

theorem a_2015_value : sequence_a 2015 = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2015_value_l1276_127642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1276_127661

noncomputable section

-- Define the ellipse C
def C : Set (ℝ × ℝ) := {p | (p.1^2 / 4) + (p.2^2 / 3) = 1}

-- Define the foci F₁ and F₂
def F₁ : ℝ × ℝ := (-1, 0)
def F₂ : ℝ × ℝ := (1, 0)

-- Define the top vertex A
noncomputable def A : ℝ × ℝ := (0, Real.sqrt 3)

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Define the line AF₁
noncomputable def line_AF₁ (x : ℝ) : ℝ := Real.sqrt 3 * (x + 1)

-- Theorem statement
theorem ellipse_properties :
  -- The triangle AF₁F₂ is equilateral
  (Real.sqrt ((F₁.1 - A.1)^2 + (F₁.2 - A.2)^2) = 
   Real.sqrt ((F₂.1 - A.1)^2 + (F₂.2 - A.2)^2)) ∧
  (Real.sqrt ((F₁.1 - A.1)^2 + (F₁.2 - A.2)^2) = 
   Real.sqrt ((F₂.1 - F₁.1)^2 + (F₂.2 - F₁.2)^2)) ∧
  -- The perimeter of AF₁F₂ is 6
  (Real.sqrt ((F₁.1 - A.1)^2 + (F₁.2 - A.2)^2) + 
   Real.sqrt ((F₂.1 - A.1)^2 + (F₂.2 - A.2)^2) + 
   Real.sqrt ((F₂.1 - F₁.1)^2 + (F₂.2 - F₁.2)^2) = 6) →
  -- 1. The eccentricity is 1/2
  (Real.sqrt (1 - (3 / 4)) = 1 / 2) ∧
  -- 2. The minimum value of |PF₂|+|PO| is √7
  (∃ (P : ℝ × ℝ), P.2 = line_AF₁ P.1 ∧
    ∀ (Q : ℝ × ℝ), Q.2 = line_AF₁ Q.1 →
      Real.sqrt ((Q.1 - F₂.1)^2 + (Q.2 - F₂.2)^2) +
      Real.sqrt ((Q.1 - O.1)^2 + (Q.2 - O.2)^2) ≥
      Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) +
      Real.sqrt ((P.1 - O.1)^2 + (P.2 - O.2)^2)) ∧
  -- The minimum value is √7
  (∃ (P : ℝ × ℝ), P.2 = line_AF₁ P.1 ∧
    Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) +
    Real.sqrt ((P.1 - O.1)^2 + (P.2 - O.2)^2) = Real.sqrt 7) ∧
  -- 3. The coordinates of P at this minimum are (-2/3, √3/3)
  (∃ (P : ℝ × ℝ), P.2 = line_AF₁ P.1 ∧
    P.1 = -2/3 ∧ P.2 = Real.sqrt 3 / 3 ∧
    ∀ (Q : ℝ × ℝ), Q.2 = line_AF₁ Q.1 →
      Real.sqrt ((Q.1 - F₂.1)^2 + (Q.2 - F₂.2)^2) +
      Real.sqrt ((Q.1 - O.1)^2 + (Q.2 - O.2)^2) ≥
      Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) +
      Real.sqrt ((P.1 - O.1)^2 + (P.2 - O.2)^2)) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1276_127661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_statistics_l1276_127659

/-- Represents a class of students with their scores -/
structure StudentClass where
  scores : List Nat
  deriving Repr

/-- Calculate the median of a list of natural numbers -/
def median (l : List Nat) : Rat :=
  sorry

/-- Calculate the mode of a list of natural numbers -/
def mode (l : List Nat) : Nat :=
  sorry

/-- Calculate the excellent rate of a list of scores -/
def excellentRate (l : List Nat) : Rat :=
  sorry

theorem class_statistics (class1 class2 : StudentClass) :
  class1.scores = [6, 8, 8, 8, 9, 9, 9, 9, 10, 10] →
  class2.scores = [6, 7, 8, 8, 8, 9, 10, 10, 10, 10] →
  median class2.scores = 17/2 ∧
  mode class1.scores = 9 ∧
  excellentRate class1.scores = 3/5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_statistics_l1276_127659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentages_H3O4Cl3_l1276_127649

/-- Represents a chemical element with its atomic mass -/
structure Element where
  symbol : String
  atomic_mass : Float

/-- Represents a chemical compound -/
structure Compound where
  formula : String
  elements : List (Element × Nat)

/-- Calculates the molar mass of a compound -/
def molar_mass (c : Compound) : Float :=
  c.elements.foldl (fun acc (elem, count) => acc + elem.atomic_mass * count.toFloat) 0

/-- Calculates the mass percentage of an element in a compound -/
def mass_percentage (c : Compound) (e : Element) : Float :=
  let element_count := (c.elements.find? (fun (elem, _) => elem.symbol == e.symbol)).map (·.2)
  match element_count with
  | some count => (e.atomic_mass * count.toFloat / molar_mass c) * 100
  | none => 0

/-- Approximation relation for floats -/
def approx (x y : Float) : Prop := (x - y).abs < 0.01

/-- Theorem: Mass percentages of H, O, and Cl in H3O4Cl3 -/
theorem mass_percentages_H3O4Cl3 (H O Cl : Element) (H3O4Cl3 : Compound) :
  H.symbol = "H" ∧ H.atomic_mass = 1.01 ∧
  O.symbol = "O" ∧ O.atomic_mass = 16.00 ∧
  Cl.symbol = "Cl" ∧ Cl.atomic_mass = 35.45 ∧
  H3O4Cl3.formula = "H3O4Cl3" ∧
  H3O4Cl3.elements = [(H, 3), (O, 4), (Cl, 3)] →
  approx (mass_percentage H3O4Cl3 H) 1.75 ∧
  approx (mass_percentage H3O4Cl3 O) 36.92 ∧
  approx (mass_percentage H3O4Cl3 Cl) 61.33 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentages_H3O4Cl3_l1276_127649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_score_probability_is_four_ninths_l1276_127636

-- Define the dartboard structure
structure Dartboard :=
  (inner_radius : ℝ)
  (outer_radius : ℝ)
  (inner_points : Fin 3 → ℕ)
  (outer_points : Fin 3 → ℕ)

-- Define the probability of hitting a region
noncomputable def hit_probability (d : Dartboard) (is_inner : Bool) : ℝ :=
  if is_inner then
    (d.inner_radius^2) / (d.outer_radius^2)
  else
    (d.outer_radius^2 - d.inner_radius^2) / (d.outer_radius^2)

-- Define the probability of getting an odd score
noncomputable def odd_score_probability (d : Dartboard) : ℝ :=
  sorry

-- Theorem statement
theorem odd_score_probability_is_four_ninths (d : Dartboard) :
  d.inner_radius = 4 ∧
  d.outer_radius = 8 ∧
  d.inner_points = ![3, 5, 5] ∧
  d.outer_points = ![4, 3, 3] →
  odd_score_probability d = 4/9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_score_probability_is_four_ninths_l1276_127636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_area_is_48_l1276_127688

/-- The function representing the upper bound of the figure -/
noncomputable def f (x : ℝ) : ℝ := 3 + Real.sqrt (4 - x)

/-- The tangent line to f at point x₀ -/
noncomputable def tangent_line (x₀ : ℝ) (x : ℝ) : ℝ :=
  -(1 / (2 * Real.sqrt (4 - x₀))) * (x - x₀) + f x₀

/-- The area of the figure for a given x₀ -/
noncomputable def area (x₀ : ℝ) : ℝ :=
  6 * ((5 + x₀) / Real.sqrt (4 - x₀) + 6 + 2 * Real.sqrt (4 - x₀))

/-- The theorem stating that the smallest area is 48 -/
theorem smallest_area_is_48 :
  ∃ (min_area : ℝ), min_area = 48 ∧
  ∀ x₀, -11 ≤ x₀ ∧ x₀ ≤ 1 → area x₀ ≥ min_area := by
  sorry

#check smallest_area_is_48

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_area_is_48_l1276_127688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_plus_5pi_over_4_l1276_127665

theorem cos_alpha_plus_5pi_over_4 
  (α β : ℝ) 
  (h_α : α ∈ Set.Ioo (-π/4) 0)
  (h_β : β ∈ Set.Ioo (π/2) π)
  (h_cos_sum : Real.cos (α + β) = -4/5)
  (h_cos_diff : Real.cos (β - π/4) = 5/13) :
  Real.cos (α + 5*π/4) = 16/65 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_plus_5pi_over_4_l1276_127665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_pyramid_volume_l1276_127650

/-- Represents a pyramid with a square base -/
structure SquarePyramid where
  -- Side length of the square base
  base_side : ℝ
  -- Height of the pyramid
  height : ℝ
  -- Angle between two adjacent edges from vertex to base
  vertex_angle : ℝ

/-- Volume of a pyramid -/
noncomputable def pyramid_volume (p : SquarePyramid) : ℝ :=
  (1/3) * p.base_side^2 * p.height

theorem square_pyramid_volume :
  ∀ (p : SquarePyramid),
    p.base_side = 2 →
    p.vertex_angle = Real.pi / 2 →
    pyramid_volume p = (4 * Real.sqrt 3) / 3 := by
  intro p h1 h2
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_pyramid_volume_l1276_127650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_inequality_l1276_127626

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- State the conditions
variable (hf_pos : ∀ x ∈ Set.Icc 0 1, 0 < f x)
variable (hg_pos : ∀ x ∈ Set.Icc 0 1, 0 < g x)
variable (hf_cont : ContinuousOn f (Set.Icc 0 1))
variable (hg_cont : ContinuousOn g (Set.Icc 0 1))
variable (hf_incr : MonotoneOn f (Set.Icc 0 1))
variable (hg_decr : AntitoneOn g (Set.Icc 0 1))

-- State the theorem
theorem integral_inequality :
  ∫ x in Set.Icc 0 1, f x * g x ≤ ∫ x in Set.Icc 0 1, f x * g (1 - x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_inequality_l1276_127626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_pages_l1276_127604

/-- Represents the number of pages read for each book on each day of the week --/
def pages_per_day : Fin 7 → Fin 3 → ℕ
  | 0, 0 => 10  -- Monday, Book A
  | 1, 0 => 15  -- Tuesday, Book A
  | 2, 0 => 10  -- Wednesday, Book A
  | 3, 0 => 15  -- Thursday, Book A
  | 4, 2 => 5   -- Friday, Book C
  | 5, 1 => 20  -- Saturday, Book B
  | 6, 1 => 25  -- Sunday, Book B
  | _, _ => 0   -- No reading on other days or for other books

/-- Represents whether Sally reads on a given day (1-indexed) --/
def reads_on_day (day : ℕ) : Bool :=
  day % 3 ≠ 0

/-- The total number of pages Sally reads from each book over two weeks --/
def total_pages_read : Fin 3 → ℕ :=
  λ book => (List.range 14).foldl (λ acc day =>
    if reads_on_day (day + 1)
    then acc + pages_per_day (day % 7) book
    else acc) 0

/-- Theorem stating the number of pages in each book based on Sally's reading progress --/
theorem book_pages :
  (total_pages_read 0 = 75) ∧
  (total_pages_read 1 * 2 = 180) ∧
  (total_pages_read 2 * 2 = 20) := by
  sorry

#eval total_pages_read 0
#eval total_pages_read 1
#eval total_pages_read 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_pages_l1276_127604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_sine_function_l1276_127609

-- Define the function f(x) = a sin(x) + b
noncomputable def f (a b x : ℝ) : ℝ := a * Real.sin x + b

-- Theorem statement
theorem max_value_of_sine_function (a b : ℝ) (ha : a < 0) :
  (∀ x, f a b x ≤ b - a) ∧ ∃ x, f a b x = b - a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_sine_function_l1276_127609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factor_implies_a_value_l1276_127641

theorem factor_implies_a_value (a b : ℤ) : 
  (∃ p : Polynomial ℤ, a * X^19 + b * X^18 + 1 = (X^2 - X - 1) * p) → a = 1597 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factor_implies_a_value_l1276_127641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_uncle_money_correct_l1276_127691

/-- The amount of money John received from his uncle -/
noncomputable def money_from_uncle : ℚ := 100

/-- The amount John gives to his sister -/
noncomputable def money_to_sister : ℚ := money_from_uncle / 4

/-- The amount John spends on groceries -/
noncomputable def grocery_cost : ℚ := 40

/-- The amount John has remaining -/
noncomputable def remaining_money : ℚ := 35

/-- Theorem stating that the amount John received from his uncle is correct -/
theorem uncle_money_correct : 
  money_from_uncle - money_to_sister - grocery_cost = remaining_money := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_uncle_money_correct_l1276_127691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_calculation_l1276_127694

/-- The area of a shape composed of a square and 8 right triangles -/
theorem shaded_area_calculation (square_side : ℝ) (triangle_base triangle_height : ℝ) 
  (h_square : square_side = 3)
  (h_triangle_base : triangle_base = 2)
  (h_triangle_height : triangle_height = 1) :
  square_side ^ 2 + 8 * (triangle_base * triangle_height / 2) = 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_calculation_l1276_127694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_penguin_colony_initial_size_fish_caught_per_day_l1276_127680

/-- Represents the growth of a penguin colony over three years -/
structure PenguinColony where
  current_size : ℕ
  third_year_increase : ℕ
  second_year_factor : ℕ
  first_year_factor : ℚ

/-- Calculates the initial size of the penguin colony -/
def initial_colony_size (colony : PenguinColony) : ℚ :=
  ((colony.current_size - colony.third_year_increase) / colony.second_year_factor : ℚ) / colony.first_year_factor

/-- Theorem stating that given the specific growth pattern, the initial colony size is 316 -/
theorem penguin_colony_initial_size :
  let colony := PenguinColony.mk 1077 129 2 (3/2)
  initial_colony_size colony = 316 := by
  sorry

/-- The number of fish caught per day at the beginning of the first year
    is equal to the initial colony size -/
theorem fish_caught_per_day (colony : PenguinColony) :
  initial_colony_size colony = initial_colony_size colony := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_penguin_colony_initial_size_fish_caught_per_day_l1276_127680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compromise_function_k_value_l1276_127620

-- Define the domain
def D : Set ℝ := Set.Icc 1 (2 * Real.exp 1)

-- Define the functions
def f (k : ℝ) (x : ℝ) : ℝ := (k - 1) * x - 1
def g (x : ℝ) : ℝ := 0
noncomputable def h (x : ℝ) : ℝ := (x + 1) * Real.log x

-- Define the compromise function property
def is_compromise (f g h : ℝ → ℝ) (D : Set ℝ) : Prop :=
  ∀ x ∈ D, g x ≤ f x ∧ f x ≤ h x

-- State the theorem
theorem compromise_function_k_value :
  ∃! k : ℝ, is_compromise (f k) g h D :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compromise_function_k_value_l1276_127620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_planning_committee_subcommittees_l1276_127643

theorem planning_committee_subcommittees (total : Nat) (teachers : Nat) (subcommittee_size : Nat) :
  total = 12 →
  teachers = 5 →
  subcommittee_size = 5 →
  (Nat.choose total subcommittee_size - 
   (Nat.choose (total - teachers) subcommittee_size + 
    Nat.choose teachers 1 * Nat.choose (total - teachers) (subcommittee_size - 1))) = 596 := by
  intros h_total h_teachers h_subcommittee_size
  -- The proof steps would go here
  sorry

#eval Nat.choose 12 5 - (Nat.choose 7 5 + Nat.choose 5 1 * Nat.choose 7 4)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_planning_committee_subcommittees_l1276_127643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_perimeter_with_ellipse_l1276_127635

/-- An ellipse passes through two given points --/
def ellipse_passes_through (P R Q S : ℝ × ℝ) : Prop :=
  sorry

/-- Two points are the foci of an ellipse --/
def ellipse_foci (Q S : ℝ × ℝ) : Prop :=
  sorry

/-- The perimeter of a rectangle given its four corners --/
def perimeter_rectangle (P Q R S : ℝ × ℝ) : ℝ :=
  sorry

/-- The perimeter of a rectangle given specific conditions involving an ellipse --/
theorem rectangle_perimeter_with_ellipse 
  (P Q R S : ℝ × ℝ) -- Points of the rectangle
  (area_rect : ℝ) -- Area of the rectangle
  (area_ellipse : ℝ) -- Area of the ellipse
  (h_rect_area : area_rect = 4032) -- Rectangle area condition
  (h_ellipse_area : area_ellipse = 4032 * Real.pi) -- Ellipse area condition
  (h_ellipse_passes : ellipse_passes_through P R Q S) -- Ellipse passes through P and R
  (h_ellipse_foci : ellipse_foci Q S) -- Q and S are foci of the ellipse
  : perimeter_rectangle P Q R S = 8 * Real.sqrt 2016 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_perimeter_with_ellipse_l1276_127635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ring_arrangements_count_l1276_127611

/-- The number of ways to arrange 5 rings out of 10 distinguishable rings on 4 fingers -/
def ring_arrangements : ℕ := 
  Nat.choose 10 5 * Nat.factorial 5 * 4

/-- Theorem stating that the number of ring arrangements is 120960 -/
theorem ring_arrangements_count : ring_arrangements = 120960 := by
  rfl

#eval ring_arrangements

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ring_arrangements_count_l1276_127611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_payment_difference_l1276_127603

/-- Calculates the compound interest for a given principal, rate, time, and compounding frequency -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℝ) (frequency : ℝ) : ℝ :=
  principal * (1 + rate / frequency) ^ (frequency * time)

/-- Calculates the total payment for Plan 1 -/
noncomputable def plan1_payment (principal : ℝ) (rate : ℝ) : ℝ :=
  let half_payment := compound_interest principal rate 3 12 / 2
  let remaining_debt := half_payment
  half_payment + compound_interest remaining_debt rate 7 12

/-- Calculates the total payment for Plan 2 -/
noncomputable def plan2_payment (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  compound_interest principal rate time 1

/-- The difference between Plan 2 and Plan 1 payments is approximately $6,705 -/
theorem loan_payment_difference (principal : ℝ) (rate : ℝ) (time : ℝ) :
  principal = 12000 →
  rate = 0.08 →
  time = 10 →
  ⌊plan2_payment principal rate time - plan1_payment principal rate⌋ = 6705 := by
  sorry

-- Remove the #eval line as it cannot be executed due to noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_payment_difference_l1276_127603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_B_age_l1276_127695

/-- The age of person A -/
def A : ℕ := sorry

/-- The age of person B -/
def B : ℕ := sorry

/-- A's age in 10 years equals twice B's age 10 years ago -/
axiom condition1 : A + 10 = 2 * (B - 10)

/-- A is currently 12 years older than B -/
axiom condition2 : A = B + 12

/-- B's current age is 42 years -/
theorem B_age : B = 42 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_B_age_l1276_127695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_2023_value_b_general_form_l1276_127639

def b : ℕ → ℚ
  | 0 => 2  -- Adding case for 0 to cover all natural numbers
  | 1 => 2
  | 2 => 6/13
  | (n+3) => (b (n+1) * b (n+2)) / (3 * b (n+1) - b (n+2))

theorem b_2023_value : b 2023 = 6/12137 := by sorry

-- Additional theorem to prove the general form
theorem b_general_form (n : ℕ) (h : n ≥ 2) : b n = 6 / (3 * (2 * n) - 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_2023_value_b_general_form_l1276_127639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_property_l1276_127651

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- Check if a point is on the hyperbola -/
def isOnHyperbola (h : Hyperbola) (p : Point) : Prop :=
  p.x^2 / h.a^2 - p.y^2 / h.b^2 = 1

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem statement -/
theorem hyperbola_focal_property (h : Hyperbola) (o f1 f2 p q : Point) : 
  h.a = 4 →
  h.b = 4 * Real.sqrt 3 →
  o.x = 0 ∧ o.y = 0 →
  isOnHyperbola h p →
  distance f1 q = distance q p →
  distance p f1 = 10 →
  distance o q = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_property_l1276_127651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l1276_127667

/-- An arithmetic sequence {a_n} -/
def a_n (n : ℕ) : ℚ := 2 * n + 1

/-- Sum of the first n terms of {a_n} -/
def S_n (n : ℕ) : ℚ := n^2 + 2 * n

/-- Sequence {b_n} defined as 1/S_n -/
noncomputable def b_n (n : ℕ) : ℚ := 1 / S_n n

/-- Sum of the first n terms of {b_n} -/
noncomputable def T_n (n : ℕ) : ℚ := 3/4 - (2 * n + 3) / (2 * (n + 1) * (n + 2))

theorem arithmetic_sequence_problem (n : ℕ) :
  (a_n 3 = 7) ∧ (a_n 5 + a_n 7 = 26) →
  (∀ k : ℕ, a_n k = 2 * k + 1) ∧
  (T_n n = 3/4 - (2 * n + 3) / (2 * (n + 1) * (n + 2))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l1276_127667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bug_can_return_to_start_l1276_127608

/-- Represents a cell in the grid -/
structure Cell where
  x : ℕ
  y : ℕ

/-- Represents a door between two cells -/
inductive Door
  | Closed
  | OpenForward
  | OpenBackward

/-- Represents the grid -/
def Grid := ℕ → ℕ → Cell

/-- Represents the state of doors in the grid -/
def DoorState := Cell → Cell → Door

/-- Represents the bug's movement -/
def Move := Cell → Cell

/-- The bug's current position -/
def BugPosition := Cell

/-- Function to update the door state after a move -/
def updateDoorState (ds : DoorState) (fromCell : Cell) (toCell : Cell) : DoorState :=
  sorry

/-- Function to check if a move is valid -/
def isValidMove (ds : DoorState) (fromCell : Cell) (toCell : Cell) : Prop :=
  sorry

theorem bug_can_return_to_start 
  (g : Grid) 
  (initial_ds : DoorState) 
  (initial_pos : BugPosition) 
  (moves : List Move) : 
  ∃ (return_moves : List Move), 
    (∀ m ∈ moves, isValidMove initial_ds initial_pos (m initial_pos)) →
    (∀ m ∈ return_moves, 
      isValidMove (updateDoorState initial_ds initial_pos (m initial_pos)) 
        (m initial_pos) initial_pos) ∧
    (List.foldl (λ pos move => move pos) initial_pos return_moves = initial_pos) :=
  by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bug_can_return_to_start_l1276_127608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_but_not_necessary_l1276_127602

/-- The function f(x) = x^2 + x - a * ln(x) --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 + x - a * Real.log x

/-- f(x) is monotonically increasing on [1, +∞) --/
def is_monotone_increasing (a : ℝ) : Prop :=
  ∀ x y, 1 ≤ x → x ≤ y → f a x ≤ f a y

theorem sufficient_but_not_necessary :
  (∀ a, a < 3 → is_monotone_increasing a) ∧
  (∃ a, a ≥ 3 ∧ is_monotone_increasing a) := by
  sorry

#check sufficient_but_not_necessary

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_but_not_necessary_l1276_127602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_2005_is_5_l1276_127606

/-- Represents the sequence of digits where each number n from 1 to 99 is written n times. -/
def digit_sequence (n : ℕ) : ℕ := sorry

/-- The sum of digits written up to and including number n in the sequence. -/
def S (n : ℕ) : ℕ :=
  if n ≤ 9 then
    (n * (n + 1)) / 2
  else
    45 + (n * (n + 1) - 90)

/-- The 2005th digit in the sequence. -/
def digit_2005 : ℕ := digit_sequence 2005

theorem digit_2005_is_5 : digit_2005 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_2005_is_5_l1276_127606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_biscuit_price_is_one_l1276_127631

/-- Represents the daily sales and pricing information for a bakery --/
structure BakerySales where
  cupcake_price : ℚ
  cookie_price : ℚ
  cupcakes_per_day : ℕ
  cookie_packets_per_day : ℕ
  biscuit_packets_per_day : ℕ
  total_earnings_5_days : ℚ

/-- Calculates the price of biscuits given the bakery sales information --/
def calculate_biscuit_price (sales : BakerySales) : ℚ :=
  let daily_cupcake_earnings := sales.cupcake_price * sales.cupcakes_per_day
  let daily_cookie_earnings := sales.cookie_price * sales.cookie_packets_per_day
  let daily_earnings := sales.total_earnings_5_days / 5
  let daily_biscuit_earnings := daily_earnings - daily_cupcake_earnings - daily_cookie_earnings
  daily_biscuit_earnings / sales.biscuit_packets_per_day

/-- Theorem stating that the calculated biscuit price is $1 --/
theorem biscuit_price_is_one (sales : BakerySales)
  (h1 : sales.cupcake_price = 3/2)
  (h2 : sales.cookie_price = 2)
  (h3 : sales.cupcakes_per_day = 20)
  (h4 : sales.cookie_packets_per_day = 10)
  (h5 : sales.biscuit_packets_per_day = 20)
  (h6 : sales.total_earnings_5_days = 350) :
  calculate_biscuit_price sales = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_biscuit_price_is_one_l1276_127631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_expressions_property_l1276_127698

noncomputable def log_expr1 (x : ℝ) : ℝ := Real.log (4*x + 1) / Real.log (Real.sqrt (5*x - 1))
noncomputable def log_expr2 (x : ℝ) : ℝ := Real.log ((x/2 + 2)^2) / Real.log (4*x + 1)
noncomputable def log_expr3 (x : ℝ) : ℝ := Real.log (5*x - 1) / Real.log (x/2 + 2)

theorem log_expressions_property :
  ∃ (a b c : ℝ), 
    (a = log_expr1 2 ∧ b = log_expr2 2 ∧ c = log_expr3 2) ∧
    ((a = b ∧ c = a - 1) ∨ (a = c ∧ b = a - 1) ∨ (b = c ∧ a = b - 1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_expressions_property_l1276_127698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1276_127656

/-- The function f(x) with parameters ω and a -/
noncomputable def f (ω : ℝ) (a : ℝ) (x : ℝ) : ℝ := 
  4 * Real.cos (ω * x) * Real.sin (ω * x + Real.pi / 6) + a

/-- The theorem stating the properties of the function f -/
theorem function_properties (ω : ℝ) (a : ℝ) (h_ω_pos : ω > 0) 
  (h_max : ∀ x, f ω a x ≤ 2) 
  (h_period : ∀ x, f ω a (x + Real.pi / ω) = f ω a x) : 
  a = -1 ∧ ω = 1 ∧ 
  ∀ x ∈ Set.Icc (Real.pi / 6 : ℝ) (2 * Real.pi / 3), 
    ∀ y ∈ Set.Icc x (2 * Real.pi / 3), f ω a x ≥ f ω a y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1276_127656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cars_parking_lot_l1276_127671

/-- Represents a parking space in the grid -/
inductive ParkingSpace
  | Empty
  | Car
  | Gate

/-- Represents the parking lot grid -/
def ParkingLot := Fin 7 → Fin 7 → ParkingSpace

/-- Checks if a car can exit from its position -/
def canExit (lot : ParkingLot) (row col : Fin 7) : Prop :=
  sorry

/-- Counts the number of cars in the parking lot -/
def countCars (lot : ParkingLot) : Nat :=
  sorry

/-- Checks if all cars in the lot can exit -/
def allCarsCanExit (lot : ParkingLot) : Prop :=
  sorry

/-- The maximum number of cars that can be parked -/
def maxCars : Nat := 28

/-- Theorem stating that 28 is the maximum number of cars that can be parked -/
theorem max_cars_parking_lot :
  ∀ (lot : ParkingLot),
    (∃ (i j : Fin 7), lot i j = ParkingSpace.Gate) →
    allCarsCanExit lot →
    countCars lot ≤ maxCars :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cars_parking_lot_l1276_127671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_interval_interval_length_l1276_127666

theorem no_solution_interval (a : ℝ) : 
  (∀ x : ℝ, |x| ≠ a * x - 2) ↔ -1 ≤ a ∧ a ≤ 1 :=
sorry

theorem interval_length : |1 - (-1)| = 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_interval_interval_length_l1276_127666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_intersection_conditions_l1276_127634

noncomputable def curve (m n ω x : ℝ) : ℝ := m * Real.sin (ω * x / 2) + n

theorem curve_intersection_conditions
  (m n ω : ℝ)
  (hω_pos : ω > 0)
  (h_intersect : ∃ (x₁ x₂ : ℝ), 0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 4 * Real.pi / ω ∧
    curve m n ω x₁ = 5 ∧ curve m n ω x₂ = -1)
  (h_equal_chords : ∃ (x₃ x₄ : ℝ), 0 ≤ x₃ ∧ x₃ < x₄ ∧ x₄ ≤ 4 * Real.pi / ω ∧
    curve m n ω x₃ = 5 ∧ curve m n ω x₄ = -1 ∧
    (x₄ - x₃) = (x₂ - x₁))
  (h_nonzero_chords : ∀ (x₁ x₂ : ℝ), 0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 4 * Real.pi / ω ∧
    curve m n ω x₁ = 5 ∧ curve m n ω x₂ = -1 → x₂ - x₁ ≠ 0) :
  m > 3 ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_intersection_conditions_l1276_127634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identities_l1276_127699

/-- Given a point P(-1, 2) on the terminal side of angle α, prove the following trigonometric identities. -/
theorem trig_identities (α : Real) (h : ∃ (P : Real × Real), P = (-1, 2) ∧ P.1 = -Real.cos α ∧ P.2 = Real.sin α) :
  (Real.tan α = -2) ∧
  ((Real.sin α + Real.cos α) / (Real.cos α - Real.sin α) = -1/3) ∧
  ((3 * Real.sin α * Real.cos α - 2 * Real.cos α ^ 2) / (1 - 2 * Real.cos α ^ 2) = -8/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identities_l1276_127699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l1276_127613

noncomputable def PowerFunction (a : ℝ) : ℝ → ℝ := fun x ↦ x^a

theorem power_function_through_point (a : ℝ) :
  PowerFunction a 4 = 8 → PowerFunction a = fun x ↦ x^(3/2) := by
  intro h
  ext x
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l1276_127613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_30_deg_l1276_127600

noncomputable section

-- Define the angle in radians (π/6 is equivalent to 30°)
def angle : ℝ := Real.pi / 6

-- Define the known values for cosine and sine at 30°
axiom cos_30 : Real.cos angle = Real.sqrt 3 / 2
axiom sin_30 : Real.sin angle = 1 / 2

-- Define the tangent function
noncomputable def tan (θ : ℝ) : ℝ := Real.sin θ / Real.cos θ

-- Theorem statement
theorem tan_30_deg : tan angle = Real.sqrt 3 / 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_30_deg_l1276_127600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2theta_third_quadrant_l1276_127669

theorem sin_2theta_third_quadrant (θ : ℝ) : 
  (π < θ ∧ θ < 3*π/2) →  -- θ is in the third quadrant
  (Real.sin θ)^4 + (Real.cos θ)^4 = 5/9 → 
  Real.sin (2*θ) = -2*Real.sqrt 2/3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2theta_third_quadrant_l1276_127669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_problem_l1276_127657

theorem trig_problem (α β : ℝ)
  (h1 : 0 < α) (h2 : α < π/2)
  (h3 : 0 < β) (h4 : β < π/2)
  (h5 : Real.cos α = 3/5)
  (h6 : Real.cos (β + α) = 5/13) :
  (Real.sin β = 16/65) ∧ 
  (Real.sin (2*α) / (Real.cos α^2 + Real.cos (2*α)) = 12) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_problem_l1276_127657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_reciprocal_sin_cos_l1276_127652

open Real

theorem integral_reciprocal_sin_cos (x : ℝ) (h : x ≠ π / 2 + π * ⌊x / π⌋) :
  deriv (λ x => log (abs (tan x))) x = 1 / (sin x * cos x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_reciprocal_sin_cos_l1276_127652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_calculation_correct_l1276_127674

/-- Calculates the personal income tax given the after-tax income -/
noncomputable def calculate_tax (after_tax_income : ℝ) : ℝ :=
  let tax_free_income := (3500 : ℝ)
  let taxable_income := after_tax_income - tax_free_income
  let tax_rate := (0.03 : ℝ)
  taxable_income * tax_rate / (1 - tax_rate)

/-- Theorem stating that for an after-tax income of 4761 yuan, the tax paid is 39 yuan -/
theorem tax_calculation_correct :
  calculate_tax 4761 = 39 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_calculation_correct_l1276_127674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_MQ_length_l1276_127653

/-- Point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Distance between two points in 3D space -/
noncomputable def distance (p q : Point3D) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2)

/-- Symmetric point with respect to xOy plane -/
def symm_xOy (p : Point3D) : Point3D :=
  ⟨p.x, p.y, -p.z⟩

/-- Symmetric point with respect to x-axis -/
def symm_x (p : Point3D) : Point3D :=
  ⟨p.x, -p.y, -p.z⟩

theorem segment_MQ_length :
  let N : Point3D := ⟨2, -1, 4⟩
  let P : Point3D := ⟨1, 3, 2⟩
  let M : Point3D := symm_xOy N
  let Q : Point3D := symm_x P
  distance M Q = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_MQ_length_l1276_127653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l1276_127663

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

/-- The equation of the ellipse -/
def Ellipse.equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The eccentricity of the ellipse -/
noncomputable def Ellipse.eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - e.b^2 / e.a^2)

/-- A line passing through a point with a given slope -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- The equation of a line -/
def Line.equation (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * (x - l.point.fst) + l.point.snd

theorem ellipse_theorem (e : Ellipse) 
    (h1 : e.a * e.b = 2 * Real.sqrt 3)
    (h2 : e.eccentricity = Real.sqrt 6 / 3) :
  ∃ (l1 l2 : Line),
    -- The ellipse equation is x^2/6 + y^2/2 = 1
    (∀ x y, e.equation x y ↔ x^2/6 + y^2/2 = 1) ∧
    -- There exist exactly two lines forming equilateral triangles
    (l1.point = (2, 0) ∧ l2.point = (2, 0)) ∧
    (l1.equation = fun x y ↦ x - y - 2 = 0) ∧
    (l2.equation = fun x y ↦ x + y - 2 = 0) ∧
    -- These are the only two lines that form equilateral triangles
    (∀ (l : Line), l.point = (2, 0) →
      (∃ A B : ℝ × ℝ, ∃ P : ℝ × ℝ,
        e.equation A.1 A.2 ∧ e.equation B.1 B.2 ∧
        l.equation A.1 A.2 ∧ l.equation B.1 B.2 ∧
        P.1 = 3 ∧
        ((A.1 - B.1)^2 + (A.2 - B.2)^2 =
         (A.1 - P.1)^2 + (A.2 - P.2)^2) ∧
        ((A.1 - B.1)^2 + (A.2 - B.2)^2 =
         (B.1 - P.1)^2 + (B.2 - P.2)^2)) →
      (l.equation = l1.equation ∨ l.equation = l2.equation)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l1276_127663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reaction_result_l1276_127684

-- Define the chemical species as types
inductive Species
| HCl
| AgNO3
| NH4NO3
| NaCl
| HNO3
| AgCl
| NH4Cl
| NaNO3

-- Define the reaction type
structure Reaction where
  reactants : List (Species × ℕ)
  products : List (Species × ℕ)

-- Define the initial amounts
def initial_amounts : List (Species × ℕ) :=
  [(Species.HCl, 4), (Species.AgNO3, 3), (Species.NH4NO3, 2), (Species.NaCl, 4)]

-- Define the reactions
def reactions : List Reaction :=
  [{ reactants := [(Species.AgNO3, 1), (Species.HCl, 1)],
     products := [(Species.AgCl, 1), (Species.HNO3, 1)] },
   { reactants := [(Species.NH4NO3, 1), (Species.NaCl, 1)],
     products := [(Species.NH4Cl, 1), (Species.NaNO3, 1)] }]

-- Define a function to calculate final amounts (placeholder)
def final_amounts : Species → ℕ
| Species.HNO3 => 3
| Species.HCl => 1
| Species.NaCl => 2
| _ => 0

-- Define the theorem
theorem reaction_result :
  (final_amounts Species.HNO3 = 3) ∧
  (final_amounts Species.HCl = 1) ∧
  (final_amounts Species.NaCl = 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reaction_result_l1276_127684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_problem_l1276_127687

theorem square_root_problem (n : ℝ) : 
  (∃ x : ℝ, Real.sqrt n = 2*x + 1 ∧ Real.sqrt n = x - 7) → n = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_problem_l1276_127687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonpositive_implies_a_in_zero_one_a_in_zero_one_implies_f_nonpositive_l1276_127648

/-- The function f(x) defined as 2a*ln(x) - x^2 + a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2*a*Real.log x - x^2 + a

/-- Theorem stating that if f(x) ≤ 0 for all x > 0, then a ∈ [0, 1] -/
theorem f_nonpositive_implies_a_in_zero_one (a : ℝ) :
  (∀ x > 0, f a x ≤ 0) → 0 ≤ a ∧ a ≤ 1 := by
  sorry

/-- Theorem stating that if a ∈ [0, 1], then f(x) ≤ 0 for all x > 0 -/
theorem a_in_zero_one_implies_f_nonpositive (a : ℝ) :
  0 ≤ a ∧ a ≤ 1 → ∀ x > 0, f a x ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonpositive_implies_a_in_zero_one_a_in_zero_one_implies_f_nonpositive_l1276_127648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_altitudes_is_511_div_17_l1276_127664

/-- A triangle formed by a line and coordinate axes --/
structure AxisTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  eq : a * b = c

/-- The sum of altitudes of the triangle --/
noncomputable def sumOfAltitudes (t : AxisTriangle) : ℝ :=
  t.b + t.a + t.c / (t.a^2 + t.b^2).sqrt

/-- The specific triangle formed by 15x + 8y = 120 --/
def specificTriangle : AxisTriangle :=
  { a := 15
  , b := 8
  , c := 120
  , eq := by
      norm_num
  }

theorem sum_of_altitudes_is_511_div_17 :
  sumOfAltitudes specificTriangle = 511 / 17 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_altitudes_is_511_div_17_l1276_127664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_solution_l1276_127637

theorem quadratic_equation_solution : ∃ (m n p : ℕ), 
  (∀ x : ℝ, x * (4 * x - 9) = -4 ↔ x = (m + Real.sqrt n : ℝ) / p ∨ x = (m - Real.sqrt n : ℝ) / p) ∧ 
  Nat.gcd m (Nat.gcd n p) = 1 ∧
  m + n + p = 34 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_solution_l1276_127637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1276_127690

open Real BigOperators

def sequence_sum (a : ℕ → ℝ) (n : ℕ) : ℝ := ∑ i in Finset.range n, a i

def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = r * b n

theorem sequence_properties (a : ℕ → ℝ) (t : ℝ) (h1 : t ≠ 0) (h2 : t ≠ 1)
  (h3 : ∀ n : ℕ, sequence_sum a (n + 1) = t * (sequence_sum a (n + 1) - a n + 1)) :
  (∀ n : ℕ, a n = t^n) ∧
  (geometric_sequence (λ n ↦ (a n)^2 + (sequence_sum a (n + 1)) * a n) → t = 1/2) ∧
  (∀ k : ℝ, (∀ n : ℕ+,
    12*k / (4 + n - (sequence_sum (λ i ↦ 4 * a i + 1) n)) ≥ 2*n - 7) → k ≥ 1/32) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1276_127690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ordinary_equation_C_chord_length_l1276_127660

-- Define the parametric equation of line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ :=
  (2 + (Real.sqrt 2) / 2 * t, (Real.sqrt 2) / 2 * t)

-- Define the polar equation of curve C
noncomputable def curve_C (θ : ℝ) : ℝ :=
  2 * Real.sin θ - 2 * Real.cos θ

-- Theorem for the ordinary equation of curve C
theorem ordinary_equation_C : 
  ∀ x y : ℝ, (x + 1)^2 + (y - 1)^2 = 2 ↔ 
    ∃ θ : ℝ, x = (curve_C θ) * Real.cos θ ∧ y = (curve_C θ) * Real.sin θ :=
by sorry

-- Theorem for the length of the chord
theorem chord_length : 
  ∃ A B : ℝ × ℝ, 
    (∃ t₁ : ℝ, A = line_l t₁) ∧
    (∃ t₂ : ℝ, B = line_l t₂) ∧
    (∃ θ₁ : ℝ, A.1 = (curve_C θ₁) * Real.cos θ₁ ∧ A.2 = (curve_C θ₁) * Real.sin θ₁) ∧
    (∃ θ₂ : ℝ, B.1 = (curve_C θ₂) * Real.cos θ₂ ∧ B.2 = (curve_C θ₂) * Real.sin θ₂) ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ordinary_equation_C_chord_length_l1276_127660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overall_average_speed_l1276_127619

/-- Calculates the overall average speed given cycling and walking speeds and durations --/
theorem overall_average_speed
  (cycling_speed : ℝ)
  (cycling_duration : ℝ)
  (break_duration : ℝ)
  (walking_speed : ℝ)
  (walking_duration : ℝ)
  (h1 : cycling_speed = 20)
  (h2 : cycling_duration = 45 / 60)
  (h3 : break_duration = 15 / 60)
  (h4 : walking_speed = 3)
  (h5 : walking_duration = 1) :
  let total_distance := cycling_speed * cycling_duration + walking_speed * walking_duration
  let total_time := cycling_duration + break_duration + walking_duration
  total_distance / total_time = 9 := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_overall_average_speed_l1276_127619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_book_distribution_l1276_127645

/-- Represents the number and weight distribution of books in a box. -/
structure BookDistribution where
  large : ℚ
  medium : ℚ
  small : ℚ
  total_weight : ℚ

/-- Checks if the given book distribution satisfies all conditions. -/
def is_valid_distribution (d : BookDistribution) : Prop :=
  d.large + d.medium + d.small = 187 ∧
  d.total_weight = 189 ∧
  2.75 * d.large + 1.5 * d.medium + (1/3) * d.small = d.total_weight ∧
  2.75 * d.large ≥ 1.5 * d.medium ∧
  2.75 * d.large ≥ (1/3) * d.small ∧
  1.5 * d.medium ≥ (1/3) * d.small

/-- The theorem stating that the given distribution is the only valid one. -/
theorem unique_book_distribution :
  ∀ d : BookDistribution,
    is_valid_distribution d →
    d.large = 36 ∧ d.medium = 34 ∧ d.small = 117 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_book_distribution_l1276_127645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourier_transform_of_f_l1276_127685

-- Define the piecewise function f(x)
noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then 0
  else if x < 2 then 1
  else 0

-- Define the complex Fourier transform
noncomputable def fourierTransform (f : ℝ → ℝ) (p : ℝ) : ℂ :=
  (1 / Real.sqrt (2 * Real.pi)) * ∫ (x : ℝ), f x * Complex.exp (Complex.I * ↑p * x)

-- State the theorem
theorem fourier_transform_of_f (p : ℝ) :
  fourierTransform f p = (1 / Real.sqrt (2 * Real.pi)) * (Complex.exp (2 * Complex.I * ↑p) - Complex.exp (Complex.I * ↑p)) / (Complex.I * ↑p) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourier_transform_of_f_l1276_127685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l1276_127678

-- Define the function f
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := (2 : ℝ)^x + (2 : ℝ)^(a*x + b)

-- Define the conditions
def condition1 (a b : ℝ) : Prop := f a b 1 = 5/2
def condition2 (a b : ℝ) : Prop := f a b 2 = 17/4

-- Theorem for part (1)
theorem part1 : ∃ a b : ℝ, condition1 a b ∧ condition2 a b ∧ a = -1 ∧ b = 0 := by
  sorry

-- Define the specific function after finding a and b
noncomputable def f_specific (x : ℝ) : ℝ := (2 : ℝ)^x + (2 : ℝ)^(-x)

-- Theorem for part (2)
theorem part2 : ∀ x y : ℝ, 0 ≤ x ∧ x < y → f_specific x < f_specific y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l1276_127678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_survivor_quitters_probability_l1276_127615

/-- The probability of both quitters being from the 10-member tribe in Survivor -/
theorem survivor_quitters_probability : 
  let total_contestants : ℕ := 18
  let tribe_a_size : ℕ := 10
  let tribe_b_size : ℕ := 8
  let num_quitters : ℕ := 2
  -- Assuming equal probability of quitting for each contestant
  -- and independence of quitting events
  (total_contestants = tribe_a_size + tribe_b_size) →
  (Nat.choose total_contestants num_quitters ≠ 0) →
  (Nat.choose tribe_a_size num_quitters ≠ 0) →
  (Nat.choose total_contestants num_quitters : ℚ)⁻¹ * 
    (Nat.choose tribe_a_size num_quitters : ℚ) = 5 / 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_survivor_quitters_probability_l1276_127615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_question_bounds_l1276_127640

def factorial_question (n : ℕ) : ℚ :=
  match n with
  | 0 => 1
  | n+1 => (n+1 : ℚ) / factorial_question n

theorem factorial_question_bounds :
  Real.sqrt 1992 < (factorial_question 1992 : ℝ) ∧ 
  (factorial_question 1992 : ℝ) < (4/3 : ℝ) * Real.sqrt 1992 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_question_bounds_l1276_127640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_points_A_B_D_collinear_l1276_127646

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define the vectors
variable (a b : V)

-- Define the points
variable (A B C D : V)

-- Define the given vectors
def AB (a b : V) : V := a + 5 • b
def BC (a b : V) : V := -2 • a + 8 • b
def CD (a b : V) : V := 3 • a - 3 • b

-- Define BD
def BD (a b : V) : V := BC a b + CD a b

-- Theorem statement
theorem points_A_B_D_collinear (a b : V) : 
  ∃ (t : ℝ), BD a b = t • AB a b :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_points_A_B_D_collinear_l1276_127646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_range_for_three_zeros_l1276_127625

-- Define the piecewise function f(x)
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 2 then 3 / (x - 1)
  else |2^x - 1|

-- Define g(x) in terms of f(x) and k
noncomputable def g (k : ℝ) (x : ℝ) : ℝ := f x - k

-- State the theorem
theorem k_range_for_three_zeros (k : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ g k x₁ = 0 ∧ g k x₂ = 0 ∧ g k x₃ = 0) →
  0 < k ∧ k < 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_range_for_three_zeros_l1276_127625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_equation_l1276_127624

-- Define the matrix transformation M
def M : Matrix (Fin 2) (Fin 2) ℝ := !![3, 0; 0, 3]

-- Define the transformation of a point
def transform_point (p : ℝ × ℝ) : ℝ × ℝ :=
  (3 * p.1, 3 * p.2)

-- Define the equation of curve C'
def curve_C' (p : ℝ × ℝ) : Prop :=
  p.2^2 = 4 * p.1

-- Convert a pair to a vector
def pair_to_vec (p : ℝ × ℝ) : Fin 2 → ℝ :=
  λ i => if i = 0 then p.1 else p.2

-- Convert a vector to a pair
def vec_to_pair (v : Fin 2 → ℝ) : ℝ × ℝ :=
  (v 0, v 1)

-- State the theorem
theorem curve_C_equation :
  (∀ q : ℝ × ℝ, transform_point q = vec_to_pair (M.mulVec (pair_to_vec q))) →
  (∀ q : ℝ × ℝ, curve_C' (transform_point q) ↔ q.2^2 = (4/3) * q.1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_equation_l1276_127624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_difference_l1276_127683

/-- Future value calculation for compound interest -/
noncomputable def future_value (principal : ℝ) (rate : ℝ) (periods : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate / periods) ^ (periods * time)

/-- Theorem stating the difference between semiannual and annual compounding -/
theorem compound_interest_difference (principal : ℝ) (rate : ℝ) (time : ℝ) 
  (h1 : principal = 8000)
  (h2 : rate = 0.10)
  (h3 : time = 5) :
  ∃ ε > 0, 
  |future_value principal rate 2 time - future_value principal rate 1 time - 147.04| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_difference_l1276_127683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_angle_A_l1276_127627

noncomputable def triangle_ABC (A B C : ℝ) (a b c : ℝ) : Prop :=
  0 < A ∧ A < Real.pi ∧
  0 < B ∧ B < Real.pi ∧
  0 < C ∧ C < Real.pi ∧
  A + B + C = Real.pi ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  a / (Real.sin A) = b / (Real.sin B) ∧
  b / (Real.sin B) = c / (Real.sin C) ∧
  c / (Real.sin C) = a / (Real.sin A)

theorem triangle_ABC_angle_A (A B C : ℝ) (a b c : ℝ) :
  triangle_ABC A B C a b c →
  B = Real.pi/4 →
  c = 2 * Real.sqrt 2 →
  b = 4 * Real.sqrt 3 / 3 →
  A = 5*Real.pi/12 ∨ A = Real.pi/12 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_angle_A_l1276_127627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixty_fifth_term_is_16_l1276_127632

def next_term (n : ℕ) : ℕ :=
  if n ≤ 12 then n * 7
  else if n % 2 = 0 then n - 7
  else n / 3

def sequence_custom (start : ℕ) : ℕ → ℕ
  | 0 => start
  | n + 1 => next_term (sequence_custom start n)

theorem sixty_fifth_term_is_16 : sequence_custom 65 64 = 16 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixty_fifth_term_is_16_l1276_127632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_solid_angle_difference_l1276_127607

/-- A polyhedron with faces, edges, and vertices -/
structure Polyhedron where
  faces : ℕ
  edges : ℕ
  vertices : ℕ

/-- The sum of dihedral angles at the edges of a polyhedron -/
noncomputable def sumDihedralAngles (p : Polyhedron) : ℝ := sorry

/-- The sum of solid angles at the vertices of a polyhedron -/
noncomputable def sumSolidAngles (p : Polyhedron) : ℝ := sorry

/-- Euler's formula for polyhedra -/
axiom eulers_formula (p : Polyhedron) : p.vertices - p.edges + p.faces = 2

/-- The main theorem: difference between sum of dihedral angles and sum of solid angles -/
theorem dihedral_solid_angle_difference (p : Polyhedron) :
  sumDihedralAngles p - sumSolidAngles p = 2 * Real.pi * (p.faces - 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_solid_angle_difference_l1276_127607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_congruent_faces_l1276_127633

/-- A tetrahedron with edges a, b, c, d, e, f -/
structure Tetrahedron where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  f : ℝ

/-- The area of a triangular face of the tetrahedron -/
def faceArea (t : Tetrahedron) (face : Fin 4) : ℝ := sorry

/-- The radius of the inscribed circle of a triangular face of the tetrahedron -/
def inradius (t : Tetrahedron) (face : Fin 4) : ℝ := sorry

/-- Two triangles are congruent if they have the same side lengths -/
def congruentFaces (t : Tetrahedron) : Prop :=
  t.a = t.f ∧ t.b = t.d ∧ t.c = t.e

theorem tetrahedron_congruent_faces (t : Tetrahedron) 
  (h1 : ∀ face1 face2 : Fin 4, faceArea t face1 = faceArea t face2)
  (h2 : ∀ face1 face2 : Fin 4, inradius t face1 = inradius t face2) :
  congruentFaces t := by
  sorry

#check tetrahedron_congruent_faces

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_congruent_faces_l1276_127633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1276_127647

theorem problem_statement (x y : ℝ) 
  (h1 : x > 0) 
  (h2 : y > 0) 
  (h3 : Real.log x / Real.log y + Real.log y / Real.log x = 4) 
  (h4 : x * y = 256) : 
  (x^2 + y^2) / 4 = 1028 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1276_127647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l1276_127696

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x) - 2 * Real.sqrt 3 * Real.sin x * Real.cos x

theorem smallest_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T' ≥ T) ∧
  T = Real.pi := by
  sorry

#check smallest_positive_period_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l1276_127696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_sequence_t_range_l1276_127689

noncomputable def f (t : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 3 then x^2 - 3*t*x + 18
  else (t - 13) * Real.sqrt (x - 3)

noncomputable def a (t : ℝ) (n : ℕ+) : ℝ := f t n

theorem decreasing_sequence_t_range :
  ∀ t : ℝ, (∀ n : ℕ+, a t n ≥ a t (n + 1)) →
  (5/3 < t ∧ t < 4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_sequence_t_range_l1276_127689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_theorem_l1276_127673

-- Define the function f(x) = 2^x + log_2(x)
noncomputable def f (x : ℝ) : ℝ := 2^x + Real.log x / Real.log 2

-- State the theorem
theorem m_range_theorem (m : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, 2^x + Real.log x / Real.log (1/2) + m ≤ 0) ↔ 
  m ∈ Set.Iic (-5) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_theorem_l1276_127673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_m_value_x_plus_2y_range_l1276_127610

-- Define the curve C
noncomputable def curve_C (θ : ℝ) : ℝ × ℝ := (4 * Real.cos θ * Real.cos θ, 4 * Real.cos θ * Real.sin θ)

-- Define the line l
noncomputable def line_l (m t : ℝ) : ℝ × ℝ := (m + Real.sqrt 2 / 2 * t, Real.sqrt 2 / 2 * t)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem 1
theorem intersection_points_m_value (m : ℝ) :
  (∃ t1 t2 θ1 θ2, 
    line_l m t1 = curve_C θ1 ∧ 
    line_l m t2 = curve_C θ2 ∧ 
    distance (line_l m t1) (line_l m t2) = Real.sqrt 14) →
  m = 1 ∨ m = 3 := by
  sorry

-- Theorem 2
theorem x_plus_2y_range (x y : ℝ) :
  (∃ θ, (x, y) = curve_C θ) →
  2 - 2 * Real.sqrt 5 ≤ x + 2 * y ∧ x + 2 * y ≤ 2 + 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_m_value_x_plus_2y_range_l1276_127610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2x_plus_cos_x_gt_one_l1276_127612

theorem sin_2x_plus_cos_x_gt_one (x : ℝ) (h : 0 < x ∧ x < π/3) : Real.sin (2*x) + Real.cos x > 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2x_plus_cos_x_gt_one_l1276_127612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flower_beds_fraction_l1276_127693

/-- Represents a rectangular yard with flower beds -/
structure YardWithFlowerBeds where
  yard_length : ℝ
  yard_width : ℝ
  triangle_leg : ℝ

/-- Calculates the area of an isosceles right triangle -/
noncomputable def isosceles_right_triangle_area (leg : ℝ) : ℝ :=
  (1 / 2) * leg ^ 2

/-- Calculates the total area of two congruent isosceles right triangles -/
noncomputable def two_triangles_area (leg : ℝ) : ℝ :=
  2 * isosceles_right_triangle_area leg

/-- Calculates the area of a rectangle -/
noncomputable def rectangle_area (length width : ℝ) : ℝ :=
  length * width

/-- Theorem: The fraction of the yard occupied by flower beds is 7/30 -/
theorem flower_beds_fraction (yard : YardWithFlowerBeds) 
  (h1 : yard.yard_length = 30)
  (h2 : yard.yard_width = 7)
  (h3 : yard.triangle_leg = 7) :
  (two_triangles_area yard.triangle_leg) / (rectangle_area yard.yard_length yard.yard_width) = 7 / 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_flower_beds_fraction_l1276_127693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_subset_P_l1276_127692

-- Define set P as a subset of ℝ instead of ℕ
def P : Set ℝ := {1, 2, 3, 4, 5, 6}

-- Define set Q
def Q : Set ℝ := {x : ℝ | 2 ≤ x ∧ x ≤ 6}

-- Theorem statement
theorem intersection_subset_P : P ∩ Q ⊂ P := by
  -- We use 'sorry' to skip the proof as requested
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_subset_P_l1276_127692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_washington_high_ratio_l1276_127630

/-- Given the number of students and teachers at Washington High,
    prove that the teacher-student ratio is 27.5. -/
theorem washington_high_ratio : 
  let students : ℕ := 1155
  let teachers : ℕ := 42
  let ratio : ℚ := students / teachers
  ratio = 27.5 := by
  -- Define the given values
  let students : ℕ := 1155
  let teachers : ℕ := 42
  
  -- Calculate the ratio
  let ratio : ℚ := students / teachers
  
  -- Prove that the ratio is equal to 27.5
  sorry  -- We use sorry to skip the proof for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_washington_high_ratio_l1276_127630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_negative_1050_degrees_l1276_127686

theorem sin_negative_1050_degrees (h : ∀ θ : ℝ, Real.sin (θ + 2 * Real.pi) = Real.sin θ) :
  Real.sin ((-1050 * Real.pi) / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_negative_1050_degrees_l1276_127686
