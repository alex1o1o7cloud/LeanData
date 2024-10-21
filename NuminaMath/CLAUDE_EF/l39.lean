import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trapezoid_distance_l39_3902

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Checks if a quadrilateral is a trapezoid -/
def isTrapezoid (A B C D : Point) : Prop :=
  (A.y = D.y ∧ B.y = C.y) ∨ (A.x - D.x) / (A.y - D.y) = (B.x - C.x) / (B.y - C.y)

theorem triangle_trapezoid_distance (ABC : Triangle) (P D E : Point) :
  distance ABC.A ABC.B = 13 →
  distance ABC.B ABC.C = 14 →
  distance ABC.A ABC.C = 15 →
  P.x = (2/3) * ABC.A.x + (1/3) * ABC.C.x →
  P.y = (2/3) * ABC.A.y + (1/3) * ABC.C.y →
  D.y = ABC.A.y →
  D.x = D.y →
  E.x = 24 →
  E.y = 24 →
  isTrapezoid ABC.A ABC.B ABC.C D →
  isTrapezoid ABC.A ABC.B ABC.C E →
  distance D E = 12 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trapezoid_distance_l39_3902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crease_length_in_folded_equilateral_triangle_l39_3961

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Checks if a triangle is equilateral -/
def isEquilateral (t : Triangle) : Prop :=
  let AB := ((t.B.x - t.A.x)^2 + (t.B.y - t.A.y)^2).sqrt
  let BC := ((t.C.x - t.B.x)^2 + (t.C.y - t.B.y)^2).sqrt
  let CA := ((t.A.x - t.C.x)^2 + (t.A.y - t.C.y)^2).sqrt
  AB = BC ∧ BC = CA

/-- Represents the folded triangle -/
structure FoldedTriangle where
  original : Triangle
  A' : Point

/-- Checks if A' is on BC -/
def isA'OnBC (ft : FoldedTriangle) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧
    ft.A'.x = ft.original.B.x + t * (ft.original.C.x - ft.original.B.x) ∧
    ft.A'.y = ft.original.B.y + t * (ft.original.C.y - ft.original.B.y)

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  ((p2.x - p1.x)^2 + (p2.y - p1.y)^2).sqrt

/-- Theorem: Length of crease in folded equilateral triangle -/
theorem crease_length_in_folded_equilateral_triangle
  (ft : FoldedTriangle)
  (h_equilateral : isEquilateral ft.original)
  (h_A'_on_BC : isA'OnBC ft)
  (h_BA' : distance ft.original.B ft.A' = 2)
  (h_A'C : distance ft.A' ft.original.C = 1) :
  ∃ P Q : Point,
    distance P Q = 15 * Real.sqrt 7 / 8 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_crease_length_in_folded_equilateral_triangle_l39_3961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_csc_function_minimum_l39_3937

theorem csc_function_minimum (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x, a * (1 / Real.sin (b * x)) ≥ 2) ∧ (∃ x, a * (1 / Real.sin (b * x)) = 2) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_csc_function_minimum_l39_3937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l39_3935

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x^2 - Real.sqrt 3 * |x| + 1) + Real.sqrt (x^2 + Real.sqrt 3 * |x| + 3)

theorem min_value_of_f :
  (∀ x : ℝ, f x ≥ Real.sqrt 7) ∧
  (f (Real.sqrt 3 / 4) = Real.sqrt 7) ∧
  (f (-Real.sqrt 3 / 4) = Real.sqrt 7) := by
  sorry

#check min_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l39_3935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_sin_sum_l39_3990

theorem triangle_max_sin_sum (A B : ℝ) : 
  0 ≤ A ∧ 0 ≤ B ∧ A + B = 120 * Real.pi / 180 →
  Real.sin A * Real.sin B + Real.sin (60 * Real.pi / 180) ≤ (3 + 2 * Real.sqrt 3) / 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_sin_sum_l39_3990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_division_equality_l39_3914

theorem floor_division_equality (α : ℝ) (d : ℕ+) (h1 : α > 0) :
  ⌊α / d⌋ = ⌊(↑⌊α⌋ : ℝ) / d⌋ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_division_equality_l39_3914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_max_area_l39_3908

/-- The equation satisfied by the vertex coordinates of the rectangle --/
def vertex_equation (x y : ℝ) : Prop :=
  |y - x| = (y + x + 1) * (5 - x - y)

/-- The maximum area of the rectangle --/
noncomputable def max_area : ℝ := 12 * Real.sqrt 3

/-- Theorem stating that the maximum area of the rectangle is 12√3 --/
theorem rectangle_max_area :
  ∃ (x y : ℝ), vertex_equation x y ∧
    ∀ (x' y' : ℝ), vertex_equation x' y' →
      2 * |x' - y'| * |x' + y'| ≤ max_area :=
by
  sorry

#check rectangle_max_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_max_area_l39_3908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fitness_enthusiasts_gender_independent_expected_reward_female_fitness_experts_l39_3928

-- Define the survey data
def total_surveyed : ℕ := 100
def male_fitness_enthusiasts : ℕ := 40
def female_fitness_enthusiasts : ℕ := 35
def male_fitness_experts : ℕ := 30
def female_fitness_experts : ℕ := 20

-- Define the chi-square test parameters
def significance_level : ℚ := 0.05
def critical_value : ℚ := 3.841

-- Define the reward amount
def reward_amount : ℚ := 1000

-- Theorem for the independence of fitness enthusiasts and gender
theorem fitness_enthusiasts_gender_independent :
  let a : ℚ := male_fitness_enthusiasts
  let b : ℚ := total_surveyed / 2 - male_fitness_enthusiasts
  let c : ℚ := female_fitness_enthusiasts
  let d : ℚ := total_surveyed / 2 - female_fitness_enthusiasts
  let chi_square : ℚ := (total_surveyed * (a * d - b * c)^2) / ((a + b) * (c + d) * (a + c) * (b + d))
  chi_square < critical_value := by sorry

-- Theorem for the expected reward of female fitness experts
theorem expected_reward_female_fitness_experts :
  let p_female : ℚ := female_fitness_experts / (male_fitness_experts + female_fitness_experts)
  let expected_value : ℚ := 4 * reward_amount * p_female
  expected_value = 1600 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fitness_enthusiasts_gender_independent_expected_reward_female_fitness_experts_l39_3928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_nonnegative_l39_3927

-- Define the function f(x) = x^(1/2)
noncomputable def f (x : ℝ) : ℝ := Real.sqrt x

-- Theorem statement
theorem sqrt_nonnegative : ∀ x : ℝ, x ≥ 0 → f x ≥ 0 := by
  intro x hx
  unfold f
  exact Real.sqrt_nonneg x


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_nonnegative_l39_3927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_on_parabola_to_line_l39_3911

-- Define the parabola
noncomputable def parabola (x : ℝ) : ℝ := x^2

-- Define the line
noncomputable def line (x y : ℝ) : ℝ := 2*x - y - 4

-- Define the distance function from a point to the line
noncomputable def distance_to_line (x y : ℝ) : ℝ := 
  |line x y| / Real.sqrt 5

theorem closest_point_on_parabola_to_line :
  ∀ x : ℝ, distance_to_line x (parabola x) ≥ distance_to_line 1 (parabola 1) := by
  sorry

#check closest_point_on_parabola_to_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_on_parabola_to_line_l39_3911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_3AB_l39_3926

open Matrix

variable {n : ℕ}
variable (A B : Matrix (Fin n) (Fin n) ℝ)

theorem det_3AB (h1 : Matrix.det A = -3) (h2 : Matrix.det B = 8) :
  Matrix.det (3 • A * B) = -8 * 3^(n + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_3AB_l39_3926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_shadows_indeterminate_heights_l39_3982

/-- Represents a person with their position relative to a light source -/
structure Person where
  height : ℝ
  distance : ℝ

/-- Represents a street lamp -/
structure StreetLamp where
  height : ℝ

/-- Calculates the shadow length of a person under a street lamp -/
noncomputable def shadowLength (p : Person) (l : StreetLamp) : ℝ :=
  (l.height - p.height) * p.distance / l.height

/-- Theorem stating that equal shadow lengths do not imply equal heights -/
theorem equal_shadows_indeterminate_heights (l : StreetLamp) (p1 p2 : Person) :
  shadowLength p1 l = shadowLength p2 l →
  ∃ (h1 h2 : ℝ), h1 ≠ h2 ∧ 
    ∃ (d1 d2 : ℝ), 
      shadowLength { height := h1, distance := d1 } l = 
      shadowLength { height := h2, distance := d2 } l := by
  sorry

#check equal_shadows_indeterminate_heights

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_shadows_indeterminate_heights_l39_3982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dan_total_dimes_l39_3915

-- Define the value of a dime in dollars
def dime_value : ℚ := 1 / 10

-- Define Barry's total amount in dollars
def barry_amount : ℚ := 10

-- Calculate the number of dimes Barry has
def barry_dimes : ℕ := (barry_amount / dime_value).floor.toNat

-- Calculate the number of dimes Dan has initially (half of Barry's)
def dan_initial_dimes : ℕ := barry_dimes / 2

-- Define the number of additional dimes Dan finds
def additional_dimes : ℕ := 2

-- Theorem to prove
theorem dan_total_dimes : dan_initial_dimes + additional_dimes = 52 := by
  -- Unfold definitions
  unfold dan_initial_dimes barry_dimes
  -- Simplify expressions
  simp [dime_value, barry_amount]
  -- The proof is completed
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dan_total_dimes_l39_3915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l39_3987

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := (Real.log (x + 1)) / Real.sqrt (-x^2 - 3*x + 4)

-- State the theorem
theorem f_domain : Set.Ioo (-1 : ℝ) 1 = {x : ℝ | f x ∈ Set.univ} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l39_3987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_bisection_theorem_l39_3986

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 36 + y^2 / 9 = 1

-- Define point P
def P : ℝ × ℝ := (4, 2)

-- Define a chord AB that passes through P and is bisected by P
def chord_through_P (A B : ℝ × ℝ) : Prop :=
  A.1 + B.1 = 2 * P.1 ∧ A.2 + B.2 = 2 * P.2

-- Define the equation of line AB
def line_equation (x y : ℝ) : Prop := x + 2*y - 8 = 0

-- Helper function for linear combination of points
def linear_combination (t : ℝ) (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((1-t)*A.1 + t*B.1, (1-t)*A.2 + t*B.2)

-- Theorem statement
theorem chord_bisection_theorem :
  ∀ A B : ℝ × ℝ,
  is_on_ellipse A.1 A.2 →
  is_on_ellipse B.1 B.2 →
  chord_through_P A B →
  ∀ x y : ℝ, line_equation x y ↔ ∃ t : ℝ, (x, y) = linear_combination t A B :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_bisection_theorem_l39_3986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_PM_MQ_is_one_to_one_l39_3962

/-- Square ABCD with side length 10 -/
def square : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 10 ∧ 0 ≤ p.2 ∧ p.2 ≤ 10}

/-- Point A -/
def A : ℝ × ℝ := (0, 10)

/-- Point B -/
def B : ℝ × ℝ := (10, 10)

/-- Point C -/
def C : ℝ × ℝ := (10, 0)

/-- Point D -/
def D : ℝ × ℝ := (0, 0)

/-- Point E on DC, 3 inches from D -/
def E : ℝ × ℝ := (3, 0)

/-- Perpendicular bisector of AE -/
def perpBisectorAE (x y : ℝ) : Prop :=
  y - 5 = -3/20 * (x - 1.5)

/-- Point M: intersection of perpendicular bisector and AE -/
def M : ℝ × ℝ := (1.5, 5)

/-- Point P: intersection of perpendicular bisector and AD -/
noncomputable def P : ℝ × ℝ := sorry

/-- Point Q: intersection of perpendicular bisector and BC -/
noncomputable def Q : ℝ × ℝ := sorry

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem ratio_PM_MQ_is_one_to_one :
  distance P M = distance M Q := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_PM_MQ_is_one_to_one_l39_3962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_to_circle_l39_3932

/-- The circle C in the problem -/
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y + 4 = 0

/-- Point P in the problem -/
def point_P : ℝ × ℝ := (0, -1)

/-- A line passing through point P -/
def line_through_P (m : ℝ) (x y : ℝ) : Prop :=
  y + 1 = m * x

/-- A line is tangent to the circle C -/
def is_tangent_line (a b c : ℝ) : Prop :=
  ∃ (x y : ℝ), circle_C x y ∧ a*x + b*y + c = 0 ∧
  ∀ (x' y' : ℝ), circle_C x' y' → a*x' + b*y' + c ≥ 0

theorem tangent_lines_to_circle :
  (∀ m : ℝ, is_tangent_line m (-1) 1 → m = 0) ∧
  is_tangent_line 1 0 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_to_circle_l39_3932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_max_value_l39_3936

/-- The function f(x) = -x + 1/(4-x) has a maximum value of -6 when x > 4 -/
theorem function_max_value (x : ℝ) (h : x > 4) :
  ∃ M : ℝ, M = -6 ∧ ∀ y : ℝ, y > 4 → -y + 1 / (4 - y) ≤ M := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_max_value_l39_3936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_difference_l39_3963

theorem quadratic_root_difference (p : ℝ) : 
  let r := (p + Real.sqrt 5) / 2
  let s := (p - Real.sqrt 5) / 2
  r - s = Real.sqrt 5 ∧ 
  r ≥ s ∧
  r^2 - p*r + ((p^2 - 5) / 4) = 0 ∧
  s^2 - p*s + ((p^2 - 5) / 4) = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_difference_l39_3963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l39_3912

theorem trigonometric_problem (α β : Real)
  (h_α : α ∈ Set.Ioo 0 π)
  (h_β : β ∈ Set.Ioo 0 π)
  (h_tan_α : Real.tan α = 2)
  (h_cos_β : Real.cos β = -7 * Real.sqrt 2 / 10) :
  Real.cos (2 * α) = -3/5 ∧ 2 * α - β = -π/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l39_3912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_three_zeros_a_range_l39_3950

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2 * x - 3) * Real.exp x + a / x

-- Define the property of having three zeros
def has_three_zeros (a : ℝ) : Prop :=
  ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ 
  f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0

-- State the theorem
theorem f_three_zeros_a_range :
  ∀ a : ℝ, has_three_zeros a → -9 * Real.exp (-3/2) < a ∧ a < 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_three_zeros_a_range_l39_3950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_k_value_l39_3980

theorem quadratic_function_k_value (a b c k : ℤ) :
  let f : ℝ → ℝ := λ x => (a : ℝ) * x^2 + (b : ℝ) * x + (c : ℝ)
  (f 1 = 0) →
  (70 < f 7) →
  (f 7 < 80) →
  (90 < f 8) →
  (f 8 < 100) →
  (6000 * (k : ℝ) < f 100) →
  (f 100 < 6000 * ((k + 1) : ℝ)) →
  k = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_k_value_l39_3980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_sharing_amount_to_give_l39_3964

def friends_earnings : List ℝ := [18, 22, 30, 35, 45]

theorem equal_sharing_amount_to_give (earnings := friends_earnings) :
  let total := earnings.sum
  let equal_share := total / earnings.length
  let highest_earner := earnings.maximum?
  match highest_earner with
  | some max_earning => max_earning - equal_share = 15
  | none => False := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_sharing_amount_to_give_l39_3964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octal_arithmetic_result_l39_3940

/-- Represents a number in base 8 --/
structure OctalNumber where
  value : ℕ

/-- Addition operation for octal numbers --/
def octal_add (a b : OctalNumber) : OctalNumber :=
  ⟨a.value + b.value⟩

/-- Subtraction operation for octal numbers --/
def octal_sub (a b : OctalNumber) : OctalNumber :=
  ⟨a.value - b.value⟩

/-- Convert a natural number to an OctalNumber --/
def to_octal (n : ℕ) : OctalNumber :=
  ⟨n⟩

theorem octal_arithmetic_result :
  octal_sub (octal_add (to_octal 453) (to_octal 267)) (to_octal 512) = to_octal 232 := by
  sorry

#eval (octal_sub (octal_add (to_octal 453) (to_octal 267)) (to_octal 512)).value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octal_arithmetic_result_l39_3940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l39_3967

theorem problem_statement (a b : ℝ) (h : ({1, a + b, a} : Set ℝ) = {0, b / a, b}) : b - 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l39_3967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l39_3955

theorem inequality_solution_set : 
  {x : ℝ | 3 ≤ |5 - 2*x| ∧ |5 - 2*x| < 9} = Set.Ioo (-2) 1 ∪ Set.Ico 4 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l39_3955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l39_3949

/-- Hyperbola C with equation x²/a² - y²/b² = 1 and one asymptotic line y = 2x -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_a_pos : a > 0
  h_b_pos : b > 0
  h_asymptote : b / a = 2

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + (h.b / h.a)^2)

/-- Theorem stating that the eccentricity of the given hyperbola is √5 -/
theorem hyperbola_eccentricity (h : Hyperbola) : eccentricity h = Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l39_3949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l39_3998

-- Define an arithmetic sequence
noncomputable def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

-- Define the sum of the first n terms of an arithmetic sequence
noncomputable def arithmetic_sum (a₁ d : ℝ) (n : ℕ) : ℝ := n * (2 * a₁ + (n - 1) * d) / 2

-- Theorem statement
theorem arithmetic_sequence_problem (a₁ d : ℝ) :
  arithmetic_sum a₁ d 17 = 51 →
  arithmetic_sequence a₁ d 7 + arithmetic_sequence a₁ d 11 = 6 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l39_3998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l39_3938

noncomputable def f (x : ℝ) : ℝ := (Real.exp x - 1) / (Real.exp x + 1)

theorem f_properties :
  (∀ x, f (-x) = -f x) ∧
  (∀ x₁ x₂, x₁ ∈ Set.Ioo 0 1 → x₂ ∈ Set.Ioo 0 1 → x₁ ≠ x₂ →
    (f x₂ - f x₁) / (x₂ - x₁) > 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l39_3938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_platform_ratio_is_one_l39_3924

/-- The ratio of train length to platform length -/
noncomputable def train_platform_ratio (train_speed : ℝ) (crossing_time : ℝ) (train_length : ℝ) : ℝ := 
  let platform_length := train_speed * 1000 / 60 * crossing_time - train_length
  train_length / platform_length

/-- Theorem stating that the ratio of train length to platform length is 1 -/
theorem train_platform_ratio_is_one :
  train_platform_ratio 54 1 450 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_platform_ratio_is_one_l39_3924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_rotation_translation_l39_3905

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Rotates a line 90° counterclockwise around the origin -/
noncomputable def rotate90 (l : Line) : Line :=
  { slope := -1 / l.slope, intercept := 0 }

/-- Translates a line horizontally -/
def translateHorizontal (l : Line) (distance : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept + l.slope * distance }

/-- The main theorem -/
theorem line_rotation_translation :
  let original := Line.mk 3 0
  let rotated := rotate90 original
  let final := translateHorizontal rotated 1
  final.slope = -1/3 ∧ final.intercept = 1/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_rotation_translation_l39_3905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_angle_l39_3929

/-- Curve C₁ in Cartesian coordinates -/
def C₁ (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 4

/-- Curve C₂ in polar coordinates -/
def C₂ (ρ θ : ℝ) : Prop := ρ = 4 * Real.cos θ

/-- Curve C₃ in polar coordinates -/
def C₃ (α ρ θ : ℝ) : Prop := θ = α ∧ 0 < α ∧ α < Real.pi

/-- Convert polar coordinates to Cartesian coordinates -/
noncomputable def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ := (ρ * Real.cos θ, ρ * Real.sin θ)

/-- Distance between two points in polar coordinates -/
noncomputable def polar_distance (ρ₁ θ₁ ρ₂ θ₂ : ℝ) : ℝ :=
  Real.sqrt ((ρ₁ * Real.cos θ₁ - ρ₂ * Real.cos θ₂)^2 + (ρ₁ * Real.sin θ₁ - ρ₂ * Real.sin θ₂)^2)

theorem intersection_angle (α : ℝ) :
  (∃ ρ₁ ρ₂ : ℝ,
    C₁ (ρ₁ * Real.cos α) (ρ₁ * Real.sin α) ∧
    C₂ ρ₂ α ∧
    C₃ α ρ₁ α ∧
    C₃ α ρ₂ α ∧
    ρ₁ ≠ 0 ∧ ρ₂ ≠ 0 ∧
    polar_distance ρ₁ α ρ₂ α = 4 * Real.sqrt 2) →
  α = 3 * Real.pi / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_angle_l39_3929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tea_blending_ratio_l39_3993

/-- Represents the cost and quantity of a tea variety -/
structure TeaVariety where
  cost : ℚ
  quantity : ℚ

/-- Represents a tea blend -/
structure TeaBlend where
  variety1 : TeaVariety
  variety2 : TeaVariety
  sellingPrice : ℚ
  gainPercent : ℚ

/-- Checks if the given tea blend satisfies the problem conditions -/
def isValidBlend (blend : TeaBlend) : Prop :=
  blend.variety1.cost = 18 ∧
  blend.variety2.cost = 20 ∧
  blend.sellingPrice = 21 ∧
  blend.gainPercent = 12

/-- Calculates the blending ratio of two tea varieties -/
def blendingRatio (blend : TeaBlend) : ℚ × ℚ :=
  (blend.variety1.quantity, blend.variety2.quantity)

/-- Theorem stating that the blending ratio is 5:3 for the given conditions -/
theorem tea_blending_ratio (blend : TeaBlend) 
  (h : isValidBlend blend) : blendingRatio blend = (5, 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tea_blending_ratio_l39_3993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_m_value_l39_3974

-- Define the hyperbola equation
noncomputable def hyperbola_equation (x y m : ℝ) : Prop :=
  x^2 / m - y^2 / (m^2 + 4) = 1

-- Define the eccentricity
noncomputable def eccentricity (m : ℝ) : ℝ :=
  Real.sqrt (m + m^2 + 4) / Real.sqrt m

-- Theorem statement
theorem hyperbola_m_value :
  ∀ m : ℝ, m > 0 → (∀ x y : ℝ, hyperbola_equation x y m) → eccentricity m = Real.sqrt 5 → m = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_m_value_l39_3974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_count_and_sum_up_to_20_l39_3991

theorem prime_count_and_sum_up_to_20 : 
  (Finset.filter Nat.Prime (Finset.range 21)).card = 8 ∧ 
  (Finset.filter Nat.Prime (Finset.range 21)).sum id = 77 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_count_and_sum_up_to_20_l39_3991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_g_l39_3925

/-- The function g(x) is defined as the minimum of three linear functions -/
noncomputable def g (x : ℝ) : ℝ := min (3 * x + 3) (min ((2 / 3) * x + 2) (-(1 / 2) * x + 8))

/-- The maximum value of g(x) is 78/21 -/
theorem max_value_of_g : (⨆ x, g x) = 78 / 21 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_g_l39_3925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_l39_3976

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h : 0 < a ∧ 0 < b

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- The left focus of the hyperbola -/
noncomputable def left_focus (h : Hyperbola) : ℝ × ℝ := (-h.a * eccentricity h, 0)

/-- The right focus of the hyperbola -/
noncomputable def right_focus (h : Hyperbola) : ℝ × ℝ := (h.a * eccentricity h, 0)

/-- A point on the right branch of the hyperbola -/
structure RightBranchPoint (h : Hyperbola) where
  x : ℝ
  y : ℝ
  right_branch : h.a < x

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Theorem stating the range of eccentricity given the condition -/
theorem eccentricity_range (h : Hyperbola) (P : RightBranchPoint h)
    (condition : (distance (P.x, P.y) (left_focus h))^2 / distance (P.x, P.y) (right_focus h) = 8 * h.a) :
    1 < eccentricity h ∧ eccentricity h ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_l39_3976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_six_fifths_l39_3966

/-- Line l in parametric form -/
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (3/5 * t, 4/5 * t)

/-- Circle C in polar form -/
noncomputable def circle_C (θ : ℝ) : ℝ := 2 * Real.cos θ

/-- Intersection points of line l and circle C -/
noncomputable def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ t θ, p = line_l t ∧ Real.sqrt (p.1^2 + p.2^2) = circle_C θ}

theorem chord_length_is_six_fifths :
  ∀ (A B : ℝ × ℝ), A ∈ intersection_points → B ∈ intersection_points → ‖A - B‖ = 6/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_six_fifths_l39_3966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_permutations_divisible_by_396_l39_3903

def number_sequence : List ℕ := [5, 3, 8, 3, 8, 2, 9, 3, 6, 5, 8, 2, 0, 3, 9, 3, 7, 6]

def available_digits : List ℕ := [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

def insert_digits (seq : List ℕ) (digs : List ℕ) : List ℕ :=
  sorry

def to_number (l : List ℕ) : ℕ :=
  sorry

theorem all_permutations_divisible_by_396 :
  ∀ (perm : List ℕ), perm.Perm available_digits →
    396 ∣ to_number (insert_digits number_sequence perm) :=
by
  sorry

#check all_permutations_divisible_by_396

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_permutations_divisible_by_396_l39_3903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_for_p_half_l39_3988

def num_cards : ℕ := 52

-- Function to calculate the probability p(a)
def p (a : ℕ) : ℚ :=
  let total_combinations := (num_cards - 2).choose 2
  let lower_team_combinations := (num_cards - (a + 10)).choose 2
  let higher_team_combinations := (a - 1).choose 2
  (lower_team_combinations + higher_team_combinations : ℚ) / total_combinations

theorem min_a_for_p_half :
  ∃ a : ℕ, (∀ b < a, p b < 1/2) ∧ p a ≥ 1/2 ∧ a = 7 ∧ p 7 = 84/175 := by
  sorry

#eval p 7

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_for_p_half_l39_3988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_equals_eight_ninths_l39_3984

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 1 - x^2 else x^2 - x - 3

theorem f_composition_equals_eight_ninths :
  f (1 / f 3) = 8 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_equals_eight_ninths_l39_3984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_is_eleven_tenths_l39_3910

/-- The series defined in the problem -/
noncomputable def problem_series (n : ℕ) : ℝ :=
  (n^4 + 2*n^3 + 8*n^2 + 8*n + 8) / (2^n * (n^4 + 4))

/-- The theorem stating that the sum of the infinite series is equal to 11/10 -/
theorem series_sum_is_eleven_tenths :
  Summable (fun n => problem_series (n + 2)) ∧ 
  ∑' n, problem_series (n + 2) = 11/10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_is_eleven_tenths_l39_3910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_inscribed_sphere_in_octahedron_l39_3922

/-- The radius of the largest sphere that can be inscribed in a regular octahedron with side length 6 -/
noncomputable def largest_inscribed_sphere_radius : ℝ := Real.sqrt 6

/-- The side length of the regular octahedron -/
def octahedron_side_length : ℝ := 6

/-- Theorem stating that the radius of the largest inscribed sphere in a regular octahedron with side length 6 is √6 -/
theorem largest_inscribed_sphere_in_octahedron :
  largest_inscribed_sphere_radius = Real.sqrt (octahedron_side_length / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_inscribed_sphere_in_octahedron_l39_3922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_f_condition_l39_3931

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (|a * x^2 - x|) / Real.log a

-- State the theorem
theorem increasing_f_condition (a : ℝ) :
  (a > 0 ∧ a ≠ 1) →
  (∀ x y : ℝ, 3 ≤ x ∧ x < y ∧ y ≤ 4 → f a x < f a y) ↔
  (a > 1 ∨ (1/6 ≤ a ∧ a < 1/4)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_f_condition_l39_3931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_limit_range_l39_3939

noncomputable section

-- Define a geometric sequence
def geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := a₁ * q^(n - 1)

-- Define the sum of the first n terms of a geometric sequence
noncomputable def geometric_sum (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := 
  if q = 1 then n * a₁ else a₁ * (1 - q^n) / (1 - q)

theorem geometric_sequence_limit_range (a₁ : ℝ) (q : ℝ) :
  a₁ > 1 →
  (∀ ε > 0, ∃ N, ∀ n ≥ N, |geometric_sum a₁ q n - 1/a₁| < ε) →
  1 < a₁ ∧ a₁ < Real.sqrt 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_limit_range_l39_3939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dotProduct_range_l39_3947

/-- The ellipse with equation x^2/4 + y^2 = 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p | (p.1^2 / 4) + p.2^2 = 1}

/-- The foci of the ellipse -/
noncomputable def F1 : ℝ × ℝ := (-Real.sqrt 3, 0)
noncomputable def F2 : ℝ × ℝ := (Real.sqrt 3, 0)

/-- The dot product of vectors PF1 and PF2 for a point P on the ellipse -/
noncomputable def dotProduct (p : ℝ × ℝ) : ℝ :=
  let pf1 := (F1.1 - p.1, F1.2 - p.2)
  let pf2 := (F2.1 - p.1, F2.2 - p.2)
  pf1.1 * pf2.1 + pf1.2 * pf2.2

/-- Theorem: The dot product of PF1 and PF2 for any point P on the ellipse is in the range [-2, 1] -/
theorem dotProduct_range :
  ∀ p ∈ Ellipse, -2 ≤ dotProduct p ∧ dotProduct p ≤ 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dotProduct_range_l39_3947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_factor_condition_l39_3904

def sequence_a (n : ℕ) : ℤ :=
  match n with
  | 0 => 1
  | 1 => 3
  | k + 2 => sequence_a (k + 1) + sequence_a k

theorem common_factor_condition (n : ℕ) :
  n ≥ 1 →
  (Int.gcd (n * sequence_a (n + 1) + sequence_a n) (n * sequence_a n + sequence_a (n - 1)) > 1) ↔
  n % 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_factor_condition_l39_3904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_income_maximizes_take_home_pay_l39_3985

/-- The tax rate function: 2x% for an income of x thousand dollars -/
noncomputable def tax_rate (x : ℝ) : ℝ := 2 * x / 100

/-- The tax amount for an income of x thousand dollars -/
noncomputable def tax_amount (x : ℝ) : ℝ := tax_rate x * (1000 * x)

/-- The take-home pay for an income of x thousand dollars -/
noncomputable def take_home_pay (x : ℝ) : ℝ := 1000 * x - tax_amount x

/-- The income that maximizes take-home pay -/
def optimal_income : ℝ := 25

theorem optimal_income_maximizes_take_home_pay :
  ∀ x : ℝ, take_home_pay x ≤ take_home_pay optimal_income :=
by sorry

#eval optimal_income

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_income_maximizes_take_home_pay_l39_3985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_proof_l39_3983

def sequence_formula (n : ℕ) : ℚ :=
  n / (3 * n - 1)

theorem sequence_proof (a : ℕ → ℚ) :
  a 1 = 1 / 2 ∧
  (∀ n ≥ 2, a (n - 1) - a n = (a n * a (n - 1)) / (n * (n - 1))) →
  ∀ n : ℕ, n ≥ 1 → a n = sequence_formula n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_proof_l39_3983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_equivalence_l39_3917

noncomputable def f (x : ℝ) := 3 * Real.cos (2 * x)
noncomputable def g (x : ℝ) := 3 * Real.sin (2 * x + Real.pi / 6)

theorem shift_equivalence : ∀ x : ℝ, f x = g (x + Real.pi / 6) := by
  intro x
  simp [f, g]
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_equivalence_l39_3917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ideal_gas_min_pressure_l39_3999

/-- Represents the minimum pressure of an ideal gas in a cyclic process -/
noncomputable def min_pressure (R T₀ V₀ a b c : ℝ) : ℝ :=
  (R * T₀ / V₀) * (a * Real.sqrt (a^2 + b^2 - c^2) - b * c) /
  (b * Real.sqrt (a^2 + b^2 - c^2) + a * c)

/-- Theorem stating the minimum pressure of an ideal gas in a cyclic process -/
theorem ideal_gas_min_pressure
  (R T₀ V₀ a b c : ℝ)
  (h_positive : R > 0 ∧ T₀ > 0 ∧ V₀ > 0)
  (h_constraint : c^2 < a^2 + b^2)
  (h_cyclic : ∀ (V T : ℝ), (V / V₀ - a)^2 + (T / T₀ - b)^2 = c^2) :
  ∀ (P : ℝ), P ≥ min_pressure R T₀ V₀ a b c :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ideal_gas_min_pressure_l39_3999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spheres_centers_intersection_l39_3952

-- Define a 3D point
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a line segment
structure Segment where
  start : Point3D
  endpoint : Point3D

-- Define a sphere
structure Sphere where
  center : Point3D
  radius : ℝ

-- Define the problem setup
def setup (A₁ A₂ B₁ B₂ C₁ C₂ P : Point3D) : Prop :=
  ∃ (seg_A seg_B seg_C : Segment),
    seg_A.start = A₁ ∧ seg_A.endpoint = A₂ ∧
    seg_B.start = B₁ ∧ seg_B.endpoint = B₂ ∧
    seg_C.start = C₁ ∧ seg_C.endpoint = C₂ ∧
    ¬ (∃ (plane : Set Point3D), A₁ ∈ plane ∧ A₂ ∈ plane ∧ B₁ ∈ plane ∧ B₂ ∈ plane ∧ C₁ ∈ plane ∧ C₂ ∈ plane) ∧
    (∃ (Q : Point3D), Q = seg_A.start ∨ Q = seg_A.endpoint) ∧
    (∃ (Q : Point3D), Q = seg_B.start ∨ Q = seg_B.endpoint) ∧
    (∃ (Q : Point3D), Q = seg_C.start ∨ Q = seg_C.endpoint) ∧
    P = seg_A.start ∨ P = seg_A.endpoint

-- Define the sphere centers
noncomputable def O (i j k : Fin 2) (A₁ A₂ B₁ B₂ C₁ C₂ P : Point3D) : Point3D :=
  (Sphere.center (Sphere.mk 
    (Point3D.mk 0 0 0) -- placeholder center
    0 -- placeholder radius
  ))

-- Define the theorem
theorem spheres_centers_intersection
  (A₁ A₂ B₁ B₂ C₁ C₂ P : Point3D)
  (h : setup A₁ A₂ B₁ B₂ C₁ C₂ P) :
  ∃ (Q : Point3D),
    (O 0 0 0 A₁ A₂ B₁ B₂ C₁ C₂ P).x = Q.x ∧ (O 1 1 1 A₁ A₂ B₁ B₂ C₁ C₂ P).x = Q.x ∧
    (O 0 0 1 A₁ A₂ B₁ B₂ C₁ C₂ P).x = Q.x ∧ (O 1 1 0 A₁ A₂ B₁ B₂ C₁ C₂ P).x = Q.x ∧
    (O 0 1 0 A₁ A₂ B₁ B₂ C₁ C₂ P).x = Q.x ∧ (O 1 0 1 A₁ A₂ B₁ B₂ C₁ C₂ P).x = Q.x ∧
    (O 1 0 0 A₁ A₂ B₁ B₂ C₁ C₂ P).x = Q.x ∧ (O 0 1 1 A₁ A₂ B₁ B₂ C₁ C₂ P).x = Q.x ∧
    (O 0 0 0 A₁ A₂ B₁ B₂ C₁ C₂ P).y = Q.y ∧ (O 1 1 1 A₁ A₂ B₁ B₂ C₁ C₂ P).y = Q.y ∧
    (O 0 0 1 A₁ A₂ B₁ B₂ C₁ C₂ P).y = Q.y ∧ (O 1 1 0 A₁ A₂ B₁ B₂ C₁ C₂ P).y = Q.y ∧
    (O 0 1 0 A₁ A₂ B₁ B₂ C₁ C₂ P).y = Q.y ∧ (O 1 0 1 A₁ A₂ B₁ B₂ C₁ C₂ P).y = Q.y ∧
    (O 1 0 0 A₁ A₂ B₁ B₂ C₁ C₂ P).y = Q.y ∧ (O 0 1 1 A₁ A₂ B₁ B₂ C₁ C₂ P).y = Q.y ∧
    (O 0 0 0 A₁ A₂ B₁ B₂ C₁ C₂ P).z = Q.z ∧ (O 1 1 1 A₁ A₂ B₁ B₂ C₁ C₂ P).z = Q.z ∧
    (O 0 0 1 A₁ A₂ B₁ B₂ C₁ C₂ P).z = Q.z ∧ (O 1 1 0 A₁ A₂ B₁ B₂ C₁ C₂ P).z = Q.z ∧
    (O 0 1 0 A₁ A₂ B₁ B₂ C₁ C₂ P).z = Q.z ∧ (O 1 0 1 A₁ A₂ B₁ B₂ C₁ C₂ P).z = Q.z ∧
    (O 1 0 0 A₁ A₂ B₁ B₂ C₁ C₂ P).z = Q.z ∧ (O 0 1 1 A₁ A₂ B₁ B₂ C₁ C₂ P).z = Q.z := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spheres_centers_intersection_l39_3952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l39_3971

/-- The function f(x) as defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x + (1 - a) / x - 1

/-- The function g(x) as defined in the problem -/
def g (b : ℝ) (x : ℝ) : ℝ := x^2 - 2*b*x + 4

/-- The theorem statement -/
theorem problem_statement (b : ℝ) : 
  (∀ x₁ ∈ Set.Ioo 0 2, ∃ x₂ ∈ Set.Icc 1 2, f (1/4) x₁ ≥ g b x₂) ↔ b ∈ Set.Ici (17/8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l39_3971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_length_l39_3951

/-- 
Given an arithmetic sequence with a non-zero common difference,
where certain terms form a geometric sequence, prove that the
number of terms in the sequence is 16.
-/
theorem arithmetic_geometric_sequence_length :
  ∀ (a : ℕ → ℝ) (d : ℝ),
  d ≠ 0 →
  (∀ k : ℕ, a (k + 1) = a k + d) →
  (∃ r : ℝ, r ≠ 0 ∧ a 4 / a 3 = r ∧ a 7 / a 4 = r ∧ a 16 / a 7 = r) →
  16 = 16 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_length_l39_3951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_claire_cookies_l39_3994

/-- Calculates the number of cookies Claire can buy given her gift card usage --/
def cookies_bought (gift_card_amount : ℚ) (latte_cost : ℚ) (croissant_cost : ℚ) 
  (cookie_cost : ℚ) (days : ℕ) (remaining_amount : ℚ) : ℕ :=
  let daily_spent := latte_cost + croissant_cost
  let week_spent := daily_spent * days
  let cookie_money := gift_card_amount - remaining_amount - week_spent
  (cookie_money / cookie_cost).floor.toNat

/-- Proves that Claire can buy 5 cookies given the problem conditions --/
theorem claire_cookies : 
  cookies_bought 100 3.75 3.50 1.25 7 43 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_claire_cookies_l39_3994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a2_less_than_two_l39_3901

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
noncomputable def sumFirstN (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * seq.a 1 + (n - 1 : ℚ) * seq.d)

theorem a2_less_than_two (seq : ArithmeticSequence) 
  (h1 : 3 * seq.a 3 = seq.a 6 + 4)
  (h2 : sumFirstN seq 5 < 10) :
  seq.a 2 < 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a2_less_than_two_l39_3901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_is_18_sin_2A_value_l39_3992

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a = 3 ∧ t.B = Real.pi/3 ∧ (1/2 * t.a * t.c * Real.sin t.B) = 6 * Real.sqrt 3

-- Theorem for the perimeter
theorem perimeter_is_18 (t : Triangle) (h : triangle_conditions t) : 
  t.a + t.b + t.c = 18 := by sorry

-- Theorem for sin(2A)
theorem sin_2A_value (t : Triangle) (h : triangle_conditions t) : 
  Real.sin (2 * t.A) = (39 * Real.sqrt 3) / 98 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_is_18_sin_2A_value_l39_3992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_50_40_eq_zero_l39_3970

def f (x y : ℕ) : ℕ := (x - y).factorial % x

theorem f_50_40_eq_zero : f 50 40 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_50_40_eq_zero_l39_3970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_board_game_parity_l39_3906

/-- The sum of integers from 1 to n -/
def sum_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The parity of a natural number -/
def parity (n : ℕ) : Bool := n % 2 = 1

/-- The operation of replacing two numbers with their difference -/
def replace_operation (a b : ℤ) : ℤ := a - b

theorem board_game_parity (n : ℕ) (h : n = 2013) :
  parity (sum_to_n n) = true →
  ∀ (remaining : List ℤ),
    (∀ x ∈ remaining, ∃ y ∈ (List.range n).map (fun i => (i : ℤ) + 1), x = y ∨ ∃ a b, x = replace_operation a b) →
    parity (remaining.sum.toNat) = true :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_board_game_parity_l39_3906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_difference_pqr_l39_3930

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem smallest_difference_pqr (p q r : ℕ+) : 
  p * q * r = factorial 8 → p < q → q < r → 
  (∀ p' q' r' : ℕ+, p' * q' * r' = factorial 8 → p' < q' → q' < r' → r - q ≤ r' - q') →
  r - q = 6 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_difference_pqr_l39_3930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l39_3965

-- Define the function f(x) = 6/x
noncomputable def f (x : ℝ) : ℝ := 6 / x

-- Theorem statement
theorem function_properties :
  -- Part 1: Existence of points (3, a) and (-3, a-4) on the graph of f
  (∃ a : ℝ, f 3 = a ∧ f (-3) = a - 4) ∧
  -- Part 2: Inequality property for three points on the graph of f
  (∀ x₁ x₂ x₃ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < x₃ →
    f x₁ + f x₂ > 2 * f x₃) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l39_3965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mans_rate_is_8_l39_3944

/-- The rate of a man rowing in still water, given his speeds with and against the stream -/
noncomputable def mans_rate (speed_with_stream speed_against_stream : ℝ) : ℝ :=
  (speed_with_stream + speed_against_stream) / 2

/-- Theorem stating that the man's rate is 8 km/h given the conditions -/
theorem mans_rate_is_8 (speed_with_stream speed_against_stream : ℝ)
  (h1 : speed_with_stream = 12)
  (h2 : speed_against_stream = 4) :
  mans_rate speed_with_stream speed_against_stream = 8 := by
  unfold mans_rate
  rw [h1, h2]
  norm_num

#check mans_rate_is_8

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mans_rate_is_8_l39_3944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_numbers_count_l39_3995

def floor_list : List ℕ := (List.range 500).map (fun n => Int.toNat ⌊((n + 1)^2 : ℚ) / 500⌋)

theorem distinct_numbers_count : (floor_list.eraseDups).length = 376 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_numbers_count_l39_3995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_range_l39_3948

-- Define the power function
noncomputable def f (n : ℝ) (x : ℝ) : ℝ := x ^ n

-- Define the theorem
theorem power_function_range (n : ℝ) (a : ℝ) :
  f n 8 = 1/4 →  -- The function passes through (8, 1/4)
  f n (a + 1) < f n 2 →  -- f(a+1) < f(2)
  a < -3 ∨ a > 1 :=  -- The range for a
by
  intro h1 h2
  -- The proof steps would go here, but we'll use sorry for now
  sorry

#check power_function_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_range_l39_3948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_points_in_circle_l39_3968

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a unit square -/
def UnitSquare : Set Point :=
  {p : Point | 0 ≤ p.x ∧ p.x ≤ 1 ∧ 0 ≤ p.y ∧ p.y ≤ 1}

/-- Check if three points can fit inside a circle with radius 1/7 -/
def FitInCircle (p1 p2 p3 : Point) : Prop :=
  ∃ (center : Point), (center.x - p1.x)^2 + (center.y - p1.y)^2 ≤ (1/7)^2 ∧
                      (center.x - p2.x)^2 + (center.y - p2.y)^2 ≤ (1/7)^2 ∧
                      (center.x - p3.x)^2 + (center.y - p3.y)^2 ≤ (1/7)^2

theorem three_points_in_circle (points : Finset Point) 
    (h1 : points.card = 51) 
    (h2 : ∀ p, p ∈ points → p ∈ UnitSquare) : 
  ∃ p1 p2 p3, p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧ 
              p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ FitInCircle p1 p2 p3 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_points_in_circle_l39_3968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greg_speed_home_l39_3916

/-- Calculates the speed given total distance, initial distance, and time -/
noncomputable def calculate_speed (total_distance : ℝ) (initial_distance : ℝ) (time : ℝ) : ℝ :=
  (total_distance - initial_distance) / time

theorem greg_speed_home :
  let total_distance : ℝ := 40
  let initial_distance : ℝ := 30
  let time : ℝ := 0.5
  calculate_speed total_distance initial_distance time = 20 := by
  -- Unfold the definition of calculate_speed
  unfold calculate_speed
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greg_speed_home_l39_3916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequalities_not_always_true_l39_3919

/-- Definition of a triangle with perimeter, circumradius, and inradius -/
structure Triangle :=
  (l : ℝ)  -- perimeter
  (R : ℝ)  -- circumradius
  (r : ℝ)  -- inradius

/-- Theorem stating that none of the given inequalities always hold for all triangles -/
theorem triangle_inequalities_not_always_true : 
  ¬(∀ t : Triangle, t.l > t.R + t.r) ∧ 
  ¬(∀ t : Triangle, t.l ≤ t.R + t.r) ∧ 
  ¬(∀ t : Triangle, (1/6 : ℝ) < t.R + t.r ∧ t.R + t.r < 6*t.l) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequalities_not_always_true_l39_3919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_on_imaginary_axis_l39_3941

theorem complex_on_imaginary_axis (a : ℝ) : 
  ((a^2 - 2*a : ℂ).re = 0) → (a = 0 ∨ a = 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_on_imaginary_axis_l39_3941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_remainder_theorem_l39_3907

theorem polynomial_remainder_theorem (P : Polynomial ℝ) 
  (h1 : P.eval 49 = 61) 
  (h2 : P.eval 61 = 49) : 
  ∃ Q : Polynomial ℝ, P = (X - 49) * (X - 61) * Q + (-X + 112) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_remainder_theorem_l39_3907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_intersect_at_single_point_l39_3909

-- Define the points of the triangles
variable (a₁ a₂ b₁ b₂ c₁ c₂ : ℝ)

-- Define the triangles ABC and A₁B₁C₁
def A : ℝ × ℝ := (a₁, a₂)
def B : ℝ × ℝ := (b₁, b₂)
def C : ℝ × ℝ := (c₁, c₂)
def A₁ : ℝ × ℝ := (-a₁, a₂)
def B₁ : ℝ × ℝ := (-b₁, b₂)
def C₁ : ℝ × ℝ := (-c₁, c₂)

-- Define the lines
def line_A₁_BC (a₁ a₂ b₁ b₂ c₁ c₂ x y : ℝ) : Prop := 
  (c₂ - b₂) * x + (c₁ - b₁) * y = -(c₂ - b₂) * a₁ + (c₁ - b₁) * a₂

def line_B₁_AC (a₁ a₂ b₁ b₂ c₁ c₂ x y : ℝ) : Prop := 
  (a₂ - c₂) * x + (a₁ - c₁) * y = (a₂ - c₂) * b₁ + (a₁ - c₁) * b₂

def line_C₁_AB (a₁ a₂ b₁ b₂ c₁ c₂ x y : ℝ) : Prop := 
  (b₂ - a₂) * x + (b₁ - a₁) * y = (b₂ - a₂) * c₁ + (b₁ - a₁) * c₂

-- Theorem statement
theorem lines_intersect_at_single_point (a₁ a₂ b₁ b₂ c₁ c₂ : ℝ) :
  ∃ (x y : ℝ), 
    line_A₁_BC a₁ a₂ b₁ b₂ c₁ c₂ x y ∧ 
    line_B₁_AC a₁ a₂ b₁ b₂ c₁ c₂ x y ∧ 
    line_C₁_AB a₁ a₂ b₁ b₂ c₁ c₂ x y :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_intersect_at_single_point_l39_3909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_instantaneous_velocity_at_one_second_l39_3969

-- Define the height function
def h (t : ℝ) : ℝ := -4.9 * t^2 + 6.5 * t + 10

-- Define the instantaneous velocity function (derivative of h)
noncomputable def v (t : ℝ) : ℝ := deriv h t

-- Theorem statement
theorem instantaneous_velocity_at_one_second :
  v 1 = -3.3 := by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_instantaneous_velocity_at_one_second_l39_3969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_third_side_l39_3981

theorem triangle_third_side (a b c : ℝ) (θ : ℝ) 
  (h1 : a = 10) (h2 : b = 15) (h3 : θ = 100 * π / 180) :
  ∃ ε > 0, |c - 19.42| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_third_side_l39_3981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_ratio_regular_octagon_l39_3921

-- Define a regular octagon
structure RegularOctagon :=
  (side_length : ℝ)

-- Define the length of a diagonal spanning one side
def diagonal_one_side (o : RegularOctagon) : ℝ := o.side_length

-- Define the length of a diagonal spanning three sides
noncomputable def diagonal_three_sides (o : RegularOctagon) : ℝ := 
  o.side_length * Real.sqrt (2 + Real.sqrt 2)

-- Theorem statement
theorem diagonal_ratio_regular_octagon (o : RegularOctagon) :
  (diagonal_one_side o) / (diagonal_three_sides o) = 1 / Real.sqrt (2 + Real.sqrt 2) := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_ratio_regular_octagon_l39_3921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blackboard_transformation_l39_3956

def transform (triple : ℕ × ℕ × ℕ) : ℕ × ℕ × ℕ → Prop :=
  fun next => ∃ (i j k : ℕ), i ≠ j ∧ i ≠ k ∧ j ≠ k ∧
    ((next.1 = triple.2.1 + triple.2.2 - 1 ∧ next.2.1 = triple.2.1 ∧ next.2.2 = triple.2.2) ∨
     (next.1 = triple.1 ∧ next.2.1 = triple.1 + triple.2.2 - 1 ∧ next.2.2 = triple.2.2) ∨
     (next.1 = triple.1 ∧ next.2.1 = triple.2.1 ∧ next.2.2 = triple.1 + triple.2.1 - 1))

def reachable (start finish : ℕ × ℕ × ℕ) : Prop :=
  ∃ (n : ℕ) (sequence : Fin (n + 1) → ℕ × ℕ × ℕ),
    sequence 0 = start ∧
    sequence (Fin.last n) = finish ∧
    ∀ i : Fin n, transform (sequence i) (sequence i.succ)

theorem blackboard_transformation :
  (¬ reachable (2, 2, 2) (17, 1967, 1983)) ∧
  (reachable (3, 3, 3) (17, 1967, 1983)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_blackboard_transformation_l39_3956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_f_implies_a_range_l39_3978

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (2 - a * x) / Real.log a

-- State the theorem
theorem monotone_f_implies_a_range (a : ℝ) :
  a > 0 →
  a ≠ 1 →
  (∀ x y, 1 < x ∧ x < y ∧ y < 3 → f a x < f a y) →
  0 < a ∧ a ≤ 2/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_f_implies_a_range_l39_3978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_three_element_subsets_l39_3918

theorem sum_of_three_element_subsets (n : ℕ) (h : n ≥ 3) :
  let M := Finset.range n
  let S (k : ℕ) := (Finset.range k).card * k * (k + 1) / 2
  (∀ k ≥ 3, S k = Nat.choose (k - 1) 2 * k * (k + 1) / 2) ∧
  (Finset.sum (Finset.filter (λ i => i ≥ 3) (Finset.range (n+1))) S = 6 * Nat.choose (n + 2) 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_three_element_subsets_l39_3918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_ten_percent_l39_3946

-- Define the simple interest calculation function
noncomputable def simpleInterest (principal rate time : ℝ) : ℝ := principal * rate * time / 100

-- Define the problem parameters
def principal : ℝ := 5000
def interest : ℝ := 2500
def time : ℝ := 5

-- Theorem statement
theorem interest_rate_is_ten_percent :
  ∃ (rate : ℝ), simpleInterest principal rate time = interest ∧ rate = 10 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_ten_percent_l39_3946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hook_length_approximation_l39_3960

-- Define the circle ω
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

-- Define the square BCDE
structure Square :=
  (center : ℝ × ℝ)
  (side_length : ℝ)

-- Define the hook
structure Hook :=
  (segment_length : ℝ)
  (arc_length : ℝ)

-- State the theorem
theorem hook_length_approximation (ω : Circle) (BCDE : Square) (A : ℝ × ℝ) :
  BCDE.side_length ^ 2 = 200 →
  ω.radius = BCDE.side_length * Real.sqrt 2 / 2 →
  A = (2 * BCDE.center.1 - ω.center.1, 2 * BCDE.center.2 - ω.center.2) →
  let hook := Hook.mk (2 * ω.radius) (3 * π * ω.radius / 2)
  ⌊hook.segment_length + hook.arc_length⌋₊ = 67 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hook_length_approximation_l39_3960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_l39_3920

theorem function_composition (g : ℝ → ℝ) :
  (∀ x, g (x + 1) = 2 * x + 3) → (∀ x, g x = 2 * x + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_l39_3920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grape_lollipops_count_l39_3979

theorem grape_lollipops_count (total : ℕ) (cherry_percent : ℚ) (watermelon_percent : ℚ) (sour_apple_percent : ℚ) :
  total = 60 →
  cherry_percent = 30 / 100 →
  watermelon_percent = 20 / 100 →
  sour_apple_percent = 15 / 100 →
  ∃ (grape : ℕ), grape = 10 ∧ 
    2 * grape ≤ total - (Int.toNat ⌊cherry_percent * total⌋ + Int.toNat ⌊watermelon_percent * total⌋ + Int.toNat ⌊sour_apple_percent * total⌋) ∧
    2 * grape + 1 > total - (Int.toNat ⌊cherry_percent * total⌋ + Int.toNat ⌊watermelon_percent * total⌋ + Int.toNat ⌊sour_apple_percent * total⌋) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grape_lollipops_count_l39_3979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tap_fill_time_l39_3953

/-- Represents the time (in hours) it takes for one tap to fill the tank alone -/
noncomputable def T : ℝ := sorry

/-- Represents the time (in hours) it takes to fill the entire tank with multiple taps -/
def total_time : ℝ := 10

/-- The time it takes to fill half the tank with one tap -/
noncomputable def half_fill_time : ℝ := T / 2

/-- The time it takes to fill the remaining half of the tank with four taps -/
noncomputable def remaining_fill_time : ℝ := T / 8

theorem tap_fill_time :
  half_fill_time + remaining_fill_time = total_time →
  T = 16 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tap_fill_time_l39_3953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l39_3977

/-- The eccentricity of a hyperbola with asymptotes tangent to a specific parabola -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (∃ k : ℝ, ∀ x : ℝ, k * x = b / a * x ∧ (1/2 * x^2 + k * x + 2 = 0 ∨ 1/2 * x^2 - k * x + 2 = 0)) →
  Real.sqrt ((a^2 + b^2) / a^2) = Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l39_3977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_empty_subset_naturals_l39_3958

theorem empty_subset_naturals : ∅ ⊆ (Set.univ : Set ℕ) := by
  apply Set.empty_subset

#check empty_subset_naturals

end NUMINAMATH_CALUDE_ERRORFEEDBACK_empty_subset_naturals_l39_3958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l39_3933

-- Define the ellipse
noncomputable def on_ellipse (x y : ℝ) : Prop :=
  (x + 4)^2 / 9 + y^2 / 16 = 1

-- Define the parabola
noncomputable def on_parabola (x y : ℝ) : Prop :=
  y^2 = 4 * x

-- Define the area of triangle PAB
noncomputable def triangle_area (x₀ y₀ : ℝ) : ℝ :=
  1/2 * (y₀^2 - 4*x₀)^(3/2)

-- Main theorem
theorem max_triangle_area :
  ∀ x₀ y₀ : ℝ, on_ellipse x₀ y₀ →
  triangle_area x₀ y₀ ≤ 137 * Real.sqrt 137 / 16 ∧
  (triangle_area x₀ y₀ = 137 * Real.sqrt 137 / 16 ↔
    (x₀ = -41/8 ∧ (y₀ = Real.sqrt 55 / 2 ∨ y₀ = -Real.sqrt 55 / 2))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l39_3933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l39_3997

open Real

-- Define the function f(x) = 1/x + ln(x)
noncomputable def f (x : ℝ) : ℝ := 1/x + log x

-- Define g(x) = xf(x)
noncomputable def g (x : ℝ) : ℝ := x * f x

-- Theorem statement
theorem f_properties :
  (∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x - 1| ∧ |x - 1| < δ → f x > f 1) ∧
  (∃! x : ℝ, f x - x = 0) ∧
  (g (1/Real.exp 1) < g (Real.sqrt (Real.exp 1))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l39_3997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_irrational_iff_infinite_nonrepeating_decimal_l39_3942

/-- A number is irrational if and only if it has an infinite non-repeating decimal representation -/
theorem irrational_iff_infinite_nonrepeating_decimal (x : ℝ) : 
  Irrational x ↔ ∀ (seq : ℕ → ℕ), ¬ (∃ (period : ℕ), ∀ (n : ℕ), seq (n + period) = seq n) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_irrational_iff_infinite_nonrepeating_decimal_l39_3942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l39_3975

-- Define the function f
noncomputable def f (x a : ℝ) : ℝ := 4 * Real.cos x * Real.sin (x - Real.pi/3) + a

-- State the theorem
theorem function_properties :
  ∃ (a : ℝ),
    (∀ x, f x a ≤ 2) ∧
    (∃ x, f x a = 2) ∧
    (a = Real.sqrt 3) ∧
    (∀ x, f (x + Real.pi) a = f x a) ∧
    (∀ p, p > 0 → (∀ x, f (x + p) a = f x a) → p ≥ Real.pi) ∧
    (∀ A B, A < B → f A a = 1 → f B a = 1 →
      ∃ C, C = Real.pi - A - B ∧ Real.sin A / Real.sin B = Real.sqrt 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l39_3975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equalities_l39_3900

theorem trigonometric_equalities : 
  (Real.cos (15 * π / 180) ^ 2 - Real.sin (15 * π / 180) ^ 2 = Real.sqrt 3 / 2) ∧ 
  (Real.sin (73 * π / 180) * Real.cos (13 * π / 180) - Real.sin (17 * π / 180) * Real.sin (167 * π / 180) = Real.sqrt 3 / 2) ∧ 
  (Real.sin (-16 * π / 3) = Real.sqrt 3 / 2) := by
  sorry

#check trigonometric_equalities

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equalities_l39_3900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_fixed_point_l39_3934

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then (2/3) * x - 1 else 1/x

theorem unique_fixed_point :
  ∃! a : ℝ, f a = a :=
by
  -- We'll prove this later
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_fixed_point_l39_3934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l39_3996

theorem inequality_solution_set (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (a > 1 → ∀ x : ℝ, a^(x^2 - 3) > a^(2*x) ↔ x > 3 ∨ x < -1) ∧
  (0 < a ∧ a < 1 → ∀ x : ℝ, a^(x^2 - 3) > a^(2*x) ↔ -1 < x ∧ x < 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l39_3996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_at_distance_two_l39_3943

/-- The distance between two parallel lines with equations ax + by + c₁ = 0 and ax + by + c₂ = 0 -/
noncomputable def distance_between_parallel_lines (a b c₁ c₂ : ℝ) : ℝ :=
  abs (c₂ - c₁) / Real.sqrt (a^2 + b^2)

/-- Check if two lines are parallel -/
def are_parallel (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : Prop :=
  a₁ * b₂ = a₂ * b₁

theorem parallel_lines_at_distance_two :
  ∀ (m : ℝ),
    are_parallel 5 (-12) 6 5 (-12) m ∧
    distance_between_parallel_lines 5 (-12) 6 m = 2 →
    m = -20 ∨ m = 32 := by
  sorry

#check parallel_lines_at_distance_two

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_at_distance_two_l39_3943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_interest_calculation_l39_3989

/-- Calculate simple interest -/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  (principal * rate * time) / 100

theorem total_interest_calculation :
  let principal_B : ℝ := 5000
  let principal_C : ℝ := 3000
  let time_B : ℝ := 2
  let time_C : ℝ := 4
  let rate : ℝ := 9
  let interest_B := simple_interest principal_B rate time_B
  let interest_C := simple_interest principal_C rate time_C
  interest_B + interest_C = 1980 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_interest_calculation_l39_3989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reduced_price_theorem_l39_3972

/-- Represents the price reduction of oil -/
def price_reduction : ℚ := 1/10

/-- Represents the additional amount of oil that can be bought after the price reduction -/
def additional_oil : ℚ := 5

/-- Represents the fixed cost of oil -/
def fixed_cost : ℚ := 800

/-- Calculates the reduced price per kg of oil -/
def reduced_price (original_price : ℚ) : ℚ :=
  original_price * (1 - price_reduction)

/-- Theorem stating that given the conditions, the reduced price is approximately 15.99 -/
theorem reduced_price_theorem (original_price : ℚ) 
  (h1 : original_price > 0)
  (h2 : fixed_cost = original_price * (fixed_cost / (reduced_price original_price) - additional_oil)) :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/100 ∧ |reduced_price original_price - 1599/100| < ε := by
  sorry

#eval reduced_price (800 / 45)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reduced_price_theorem_l39_3972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_orthogonal_foci_product_l39_3973

/-- The ellipse with equation x²/9 + y²/4 = 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / 9) + (p.2^2 / 4) = 1}

/-- The foci of the ellipse -/
def Foci : (ℝ × ℝ) × (ℝ × ℝ) := sorry

/-- A point on the ellipse -/
def P : ℝ × ℝ := sorry

/-- The condition that PF₁ · PF₂ = 0 -/
def OrthogonalVectors (p f₁ f₂ : ℝ × ℝ) : Prop :=
  (p.1 - f₁.1) * (p.1 - f₂.1) + (p.2 - f₁.2) * (p.2 - f₂.2) = 0

/-- The theorem to be proved -/
theorem ellipse_orthogonal_foci_product (h₁ : P ∈ Ellipse)
    (h₂ : OrthogonalVectors P Foci.1 Foci.2) :
    let pf₁ := ((P.1 - Foci.1.1)^2 + (P.2 - Foci.1.2)^2).sqrt
    let pf₂ := ((P.1 - Foci.2.1)^2 + (P.2 - Foci.2.2)^2).sqrt
    pf₁ * pf₂ = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_orthogonal_foci_product_l39_3973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_midpoint_relation_l39_3945

/-- Given points A, B, C, D, E in a vector space, if D is the midpoint of BC
    and AB + AC = 4AE, then AD = 2AE -/
theorem vector_midpoint_relation {V : Type*} [AddCommGroup V] [Module ℚ V]
  (A B C D E : V) : 
  D = (1 / 2 : ℚ) • (B + C) → 
  (B - A) + (C - A) = (4 : ℚ) • (E - A) → 
  D - A = (2 : ℚ) • (E - A) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_midpoint_relation_l39_3945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l39_3954

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 4 * Real.sin x * (Real.sin (Real.pi / 4 + x / 2))^2 + Real.cos (2 * x)

-- Part 1
theorem part_one (ω : ℝ) (h_ω : ω > 0) :
  (∀ x ∈ Set.Icc (-Real.pi/2) (2*Real.pi/3), StrictMono (λ y ↦ f (ω * y))) → ω ∈ Set.Ioo 0 (3/4) :=
by sorry

-- Part 2
def A : Set ℝ := Set.Icc (Real.pi/6) (2*Real.pi/3)
def B (m : ℝ) : Set ℝ := {x | |f x - m| < 2}

theorem part_two (m : ℝ) : A ⊆ B m → m ∈ Set.Ioo 1 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l39_3954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_product_l39_3913

theorem cosine_sum_product (x : ℝ) : 
  Real.cos x + Real.cos (5*x) + Real.cos (9*x) + Real.cos (13*x) + Real.cos (17*x) = 
  5 * Real.cos (9*x) * Real.cos (6*x) * Real.cos (2*x) * Real.cos 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_product_l39_3913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reaction_product_moles_l39_3957

/-- Represents a chemical compound with its molar quantity -/
structure Compound where
  formula : String
  moles : ℚ
deriving Inhabited

/-- Represents a chemical reaction with reactants and products -/
structure Reaction where
  reactants : List Compound
  products : List Compound
deriving Inhabited

/-- The balanced chemical equation for the reaction -/
def balancedEquation : Reaction :=
  { reactants := [{ formula := "CaO", moles := 1 }, { formula := "H2O", moles := 1 }],
    products := [{ formula := "Ca(OH)2", moles := 1 }] }

/-- The initial quantities of reactants -/
def initialReactants : List Compound :=
  [{ formula := "CaO", moles := 1 }, { formula := "H2O", moles := 1 }]

/-- Theorem stating that the reaction produces 1 mole of Ca(OH)2 -/
theorem reaction_product_moles :
  (balancedEquation.products.filter (fun c => c.formula = "Ca(OH)2")).head!.moles = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reaction_product_moles_l39_3957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_volume_theorem_l39_3923

/-- A frustum of a regular square pyramid inscribed around a sphere -/
structure Frustum (r : ℝ) where
  -- The radius of the inscribed sphere
  radius : r > 0
  -- The diagonal of the base is 4r
  base_diagonal : ℝ := 4 * r

/-- The volume of the frustum -/
noncomputable def frustum_volume (r : ℝ) (f : Frustum r) : ℝ :=
  (28 * r^3) / 3

/-- Theorem stating the volume of the frustum -/
theorem frustum_volume_theorem (r : ℝ) (f : Frustum r) :
  frustum_volume r f = (28 * r^3) / 3 := by
  -- Proof goes here
  sorry

#check frustum_volume_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_volume_theorem_l39_3923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pirate_rick_digging_time_l39_3959

/-- Pirate Rick's treasure digging problem -/
theorem pirate_rick_digging_time 
  (initial_depth : ℝ) 
  (initial_time : ℝ) 
  (storm_factor : ℝ) 
  (tsunami_add : ℝ) 
  (h1 : initial_depth = 8)
  (h2 : initial_time = 4)
  (h3 : storm_factor = 1/2)
  (h4 : tsunami_add = 2) : 
  (initial_depth * storm_factor + tsunami_add) / (initial_depth / initial_time) = 3 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pirate_rick_digging_time_l39_3959
