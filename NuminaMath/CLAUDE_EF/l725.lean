import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_f_symmetric_about_y_axis_l725_72596

noncomputable def f (x : ℝ) : ℝ := Real.exp (x * Real.log 2) + Real.exp (-x * Real.log 2)

theorem f_is_even : ∀ x : ℝ, f x = f (-x) := by
  sorry

theorem f_symmetric_about_y_axis (h : ∀ x : ℝ, f x = f (-x)) :
  ∃ y : ℝ → ℝ, ∀ x : ℝ, f x = y (abs x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_f_symmetric_about_y_axis_l725_72596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_coordinates_l725_72562

/-- Predicate to check if a point is a focus of the ellipse -/
def is_focus (p : ℝ × ℝ) (major : (ℝ × ℝ) × (ℝ × ℝ)) (minor : (ℝ × ℝ) × (ℝ × ℝ)) : Prop :=
  sorry

/-- Given an ellipse with major axis endpoints (0,0) and (8,0), and minor axis endpoints (4,3) and (4,-3),
    the coordinates of the focus with the greater x-coordinate are (4 + √7, 0) -/
theorem ellipse_focus_coordinates :
  let major_axis_endpoints : (ℝ × ℝ) × (ℝ × ℝ) := ((0, 0), (8, 0))
  let minor_axis_endpoints : (ℝ × ℝ) × (ℝ × ℝ) := ((4, 3), (4, -3))
  let focus_coords : ℝ × ℝ := (4 + Real.sqrt 7, 0)
  (∀ p : ℝ × ℝ, is_focus p major_axis_endpoints minor_axis_endpoints → p.1 ≤ focus_coords.1) →
  is_focus focus_coords major_axis_endpoints minor_axis_endpoints :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_coordinates_l725_72562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_special_set_l725_72544

open BigOperators

theorem exists_special_set : ∃ (S : Finset ℕ), 
  (S.card = 4004) ∧ 
  (∀ (T : Finset ℕ), T ⊆ S → T.card = 2003 → 
    ¬(2003 ∣ (∑ x in T, x))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_special_set_l725_72544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_5343_between_consecutive_integers_l725_72592

theorem log_5343_between_consecutive_integers :
  ∃ (a b : ℕ), b = a + 1 ∧ (a : ℝ) < Real.log 5343 / Real.log 10 ∧ Real.log 5343 / Real.log 10 < (b : ℝ) ∧ a + b = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_5343_between_consecutive_integers_l725_72592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_operation_result_l725_72556

/-- Operation ⊙ for plane vectors -/
noncomputable def odot (α β : ℝ × ℝ) : ℝ := 
  (α.1 * β.1 + α.2 * β.2) / (β.1 * β.1 + β.2 * β.2)

theorem vector_operation_result (a b : ℝ × ℝ) :
  a ≠ (0, 0) → b ≠ (0, 0) →
  (∃ (k₁ k₂ : ℤ), odot a b = Real.sqrt 3 * k₁ / 3 ∧ odot b a = Real.sqrt 3 * k₂ / 3) →
  Real.sqrt (a.1^2 + a.2^2) ≥ Real.sqrt (b.1^2 + b.2^2) →
  (∃ (θ : ℝ), π/6 < θ ∧ θ < π/4 ∧
    a.1 * b.1 + a.2 * b.2 = Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2) * Real.cos θ) →
  odot a b * Real.sin θ = 2/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_operation_result_l725_72556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l725_72508

variable (x y z : ℝ)

def P (x y z : ℝ) : ℝ := x + y + z
def Q (x y z : ℝ) : ℝ := x - y - z

theorem simplify_expression (x y z : ℝ) (h : x ≠ 0) (h' : y + z ≠ 0) :
  (P x y z + Q x y z) / (P x y z - Q x y z) - (P x y z - Q x y z) / (P x y z + Q x y z) = 
  (x^2 - y^2 - 2*y*z - z^2) / (x*(y+z)) :=
by
  -- Expand definitions of P and Q
  simp [P, Q]
  -- Simplify the expression
  field_simp [h, h']
  -- The proof is completed
  sorry

#check simplify_expression

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l725_72508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_lower_bound_in_interval_l725_72524

-- Define the function f as noncomputable due to Real.sqrt
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * Real.cos x + 2 * Real.sqrt 3 * (Real.cos x)^2 - Real.sqrt 3

-- Statement for the smallest positive period
theorem smallest_positive_period :
  ∃ (T : ℝ), T > 0 ∧ (∀ x : ℝ, f (x + T) = f x) ∧
  (∀ S : ℝ, S > 0 ∧ (∀ x : ℝ, f (x + S) = f x) → T ≤ S) ∧
  T = Real.pi :=
sorry

-- Statement for the lower bound in the given interval
theorem lower_bound_in_interval :
  ∀ x : ℝ, x ∈ Set.Icc (-Real.pi/3) (Real.pi/12) → f x ≥ -Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_lower_bound_in_interval_l725_72524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_from_medians_l725_72571

/-- Represents a right triangle with given medians to the legs -/
structure RightTriangle where
  s₁ : ℝ
  s₂ : ℝ
  h : s₁ > 0 ∧ s₂ > 0

/-- Predicate to check if a real number represents the area of a given triangle -/
def represents_area_of (t : ℝ) (triangle : RightTriangle) : Prop :=
  t > 0 ∧ t = (2 / 15) * Real.sqrt (17 * triangle.s₁^2 * triangle.s₂^2 - 4 * (triangle.s₁^4 + triangle.s₂^4))

/-- The area of a right triangle given the medians to the legs -/
theorem right_triangle_area_from_medians (triangle : RightTriangle) :
  ∃ (t : ℝ), represents_area_of t triangle :=
by
  let t := (2 / 15) * Real.sqrt (17 * triangle.s₁^2 * triangle.s₂^2 - 4 * (triangle.s₁^4 + triangle.s₂^4))
  use t
  apply And.intro
  · sorry -- Proof that t > 0
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_from_medians_l725_72571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylindrical_to_rectangular_conversion_l725_72540

/-- Converts a point from cylindrical coordinates to rectangular coordinates -/
noncomputable def cylindrical_to_rectangular (r : ℝ) (θ : ℝ) (z : ℝ) : ℝ × ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ, z)

/-- The given point in cylindrical coordinates -/
noncomputable def cylindrical_point : ℝ × ℝ × ℝ := (6, Real.pi / 3, 2)

/-- The expected point in rectangular coordinates -/
noncomputable def rectangular_point : ℝ × ℝ × ℝ := (3, 3 * Real.sqrt 3, 2)

theorem cylindrical_to_rectangular_conversion :
  cylindrical_to_rectangular cylindrical_point.1 cylindrical_point.2.1 cylindrical_point.2.2 = rectangular_point := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylindrical_to_rectangular_conversion_l725_72540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_APB_l725_72583

/-- A square ABDF with side length 8 inches, point C as midpoint of FD, 
    and point P such that PA = PB = PC and PC ⟂ FD -/
structure SquareConfiguration where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  F : ℝ × ℝ
  P : ℝ × ℝ
  square_side : ℝ
  is_square : A = (0, 0) ∧ B = (square_side, 0) ∧
              D = (square_side, square_side) ∧ F = (0, square_side)
  C_midpoint : C = ((F.1 + D.1) / 2, (F.2 + D.2) / 2)
  P_equidistant : Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2) = 
                  Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2) ∧
                  Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2) = 
                  Real.sqrt ((P.1 - C.1)^2 + (P.2 - C.2)^2)
  PC_perpendicular_FD : (P.2 - C.2) * (F.1 - D.1) = (P.1 - C.1) * (F.2 - D.2)
  side_length : square_side = 8

/-- The area of triangle APB in the given configuration is 12 square inches -/
theorem area_triangle_APB (config : SquareConfiguration) : 
  (1/2) * config.square_side * (config.P.2 - config.A.2) = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_APB_l725_72583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_class_size_l725_72584

/-- Represents the number of students in each of the first four rows -/
def n : ℕ := sorry

/-- The total number of students in the class -/
def class_size : ℕ := 5 * n + 2

theorem smallest_class_size :
  (class_size > 50) →  -- More than 50 students
  (∃ (r : ℕ), r > 4 ∧ class_size = 4 * n + (n + 2)) →  -- 4 rows with n, 1 row with n+2
  (∀ m : ℕ, m < n → 5 * m + 2 ≤ 50) →  -- n is the smallest possible
  class_size = 52 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_class_size_l725_72584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_equation_l725_72554

/-- Given an ellipse and a parabola with specific properties, prove that the eccentricity of the ellipse satisfies a quadratic equation. -/
theorem ellipse_eccentricity_equation (a b c p : ℝ) (P F₁ F₂ : ℝ × ℝ) :
  a > b ∧ b > 0 ∧  -- Ellipse condition
  (P.1^2 / a^2 + P.2^2 / b^2 = 1) ∧  -- P is on ellipse
  (P.2^2 = 2*p*P.1) ∧  -- P is on parabola
  F₂ = (c, 0) ∧  -- Right focus position
  F₁ = (-c, 0) ∧  -- Left focus position
  ((P.1 - F₂.1)^2 + P.2^2 = 2*p*(P.1 - c)) ∧  -- Parabola focus condition
  ((P.1 - F₁.1) / (((P.1 - F₁.1)^2 + P.2^2).sqrt * ((P.1 - F₂.1)^2 + P.2^2).sqrt) = 7/9)  -- Cosine condition
  →
  ∃ e : ℝ, e = c/a ∧ 8*e^2 - 7*e + 1 = 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_equation_l725_72554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_june_1882_rainfall_mawsynram_l725_72589

/-- Represents the total rainfall in inches for a given month -/
def total_rainfall : ℚ := 450

/-- Represents the number of days in June -/
def days_in_june : ℕ := 30

/-- Represents the number of hours in a day -/
def hours_in_day : ℕ := 24

/-- Calculates the total number of hours in June -/
def total_hours : ℕ := days_in_june * hours_in_day

/-- Represents the average rainfall per hour -/
def average_rainfall_per_hour : ℚ := 5/8

theorem june_1882_rainfall_mawsynram :
  total_rainfall / (total_hours : ℚ) = average_rainfall_per_hour := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_june_1882_rainfall_mawsynram_l725_72589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_ratio_minimum_l725_72566

-- Define the ellipse
noncomputable def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the right focus
def F₂ : ℝ × ℝ := (1, 0)

-- Define a line passing through F₂
def line_through_F₂ (t : ℝ) (y : ℝ) : ℝ := t * y + 1

-- Define the ratio function
noncomputable def ratio (A B : ℝ × ℝ) : ℝ :=
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  let S_OAB := abs (y₁ - y₂) / 2
  let AF₂ := Real.sqrt ((x₁ - F₂.1)^2 + (y₁ - F₂.2)^2)
  let BF₂ := Real.sqrt ((x₂ - F₂.1)^2 + (y₂ - F₂.2)^2)
  (AF₂ * BF₂) / S_OAB

theorem ellipse_ratio_minimum :
  ∀ t : ℝ, ∀ A B : ℝ × ℝ,
    ellipse A.1 A.2 →
    ellipse B.1 B.2 →
    A.1 = line_through_F₂ t A.2 →
    B.1 = line_through_F₂ t B.2 →
    ratio A B ≥ 3/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_ratio_minimum_l725_72566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l725_72557

noncomputable def sequence_a (n : ℕ) : ℝ := 2 * n - 1

noncomputable def S (n : ℕ) : ℝ := (sequence_a n + 1)^2 / 4

noncomputable def sequence_b (n : ℕ) : ℝ := 1 / (sequence_a n * sequence_a (n + 1))

noncomputable def T (n : ℕ) : ℝ := n / (2 * n + 1)

theorem sequence_properties :
  ∀ (n : ℕ), n > 0 →
    (∀ (k : ℕ), k > 0 → sequence_a k > 0) ∧
    (∀ (k : ℕ), k > 0 → 4 * S k = (sequence_a k + 1)^2) →
    (∀ (k : ℕ), k > 0 → sequence_a k = 2 * k - 1) ∧
    (∀ (k : ℕ), k > 0 → T k = k / (2 * k + 1)) ∧
    (∀ (m : ℕ), (∀ (k : ℕ), k > 0 → T k > m / 23) → m ≤ 7) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l725_72557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_equation_l725_72506

/-- A line perpendicular to 3x + 4y - 2 = 0 that forms a triangle with the coordinate axes -/
structure PerpendicularLine where
  -- The slope of the line
  slope : ℝ
  -- The y-intercept of the line
  y_intercept : ℝ
  -- The line is perpendicular to 3x + 4y - 2 = 0
  perpendicular : slope = 4 / 3
  -- The perimeter of the triangle formed with the coordinate axes is 5
  perimeter : Real.sqrt ((y_intercept / slope)^2 + y_intercept^2) + 
              |y_intercept / slope| + |y_intercept| = 5

/-- The equation of the perpendicular line is 4x - 3y ± 5 = 0 -/
theorem perpendicular_line_equation (l : PerpendicularLine) :
  ∃ (sign : ℝ) (h : sign = 1 ∨ sign = -1), ∀ x y : ℝ, 
    4 * x - 3 * y + 5 * sign = 0 ↔ y = l.slope * x + l.y_intercept :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_equation_l725_72506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_range_l725_72519

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the derivative of f
def f' : ℝ → ℝ := sorry

-- Axioms based on the given conditions
axiom f_even : ∀ x, x ≠ 0 → f x = f (-x)
axiom f_zero_at_one : f 1 = 0
axiom f_inequality : ∀ x, x > 0 → x * (f' x) < 2 * (f x)

-- Theorem statement
theorem f_positive_range :
  {x : ℝ | f x > 0} = {x : ℝ | -1 < x ∧ x < 0 ∨ 0 < x ∧ x < 1} :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_range_l725_72519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_value_l725_72529

open BigOperators

def S (n : ℕ) : ℚ :=
  ∑ k in Finset.range (n + 1), (1002 - k : ℚ) / (3 ^ k)

theorem sum_value :
  S 1000 = 1502.25 + 3 / (2^1001) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_value_l725_72529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gold_sand_problem_l725_72503

/-- Represents the state of the balance scale -/
structure BalanceState where
  leftPan : ℝ
  rightPan : ℝ
  unusedSand : ℝ

/-- Represents an action on the balance scale -/
inductive BalanceAction
  | Balance
  | AddToBalance

/-- Performs an action on the balance state -/
def performAction (state : BalanceState) (action : BalanceAction) : BalanceState :=
  sorry

/-- Checks if the final state has the desired amount of sand -/
def hasDesiredAmount (state : BalanceState) (desiredAmount : ℝ) : Prop :=
  state.leftPan = desiredAmount ∨ state.rightPan = desiredAmount ∨ state.unusedSand = desiredAmount

theorem gold_sand_problem :
  ∃ (action1 action2 : BalanceAction),
    let initialState : BalanceState := ⟨0, 0, 37⟩
    let state1 := performAction initialState action1
    let state2 := performAction state1 action2
    hasDesiredAmount state2 26 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gold_sand_problem_l725_72503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tallest_tree_height_l725_72597

/-- The height of the tallest tree -/
noncomputable def T : ℝ := sorry

/-- The height of the middle-sized tree -/
noncomputable def M : ℝ := T / 2 - 6

/-- The height of the smallest tree -/
noncomputable def S : ℝ := M / 4

/-- Theorem stating that the tallest tree is 108 feet tall -/
theorem tallest_tree_height : T = 108 :=
by
  have h1 : S = 12 := by sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tallest_tree_height_l725_72597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_repetition_l725_72553

theorem difference_repetition (a : Fin 108 → ℕ) 
  (h1 : ∀ i j : Fin 108, i ≠ j → a i ≠ a j)
  (h2 : ∀ i : Fin 108, a i ≤ 2015) :
  ∃ k : ℕ, 4 ≤ (Finset.univ.filter (λ p : Fin 108 × Fin 108 ↦ a p.1 - a p.2 = k)).card :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_repetition_l725_72553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l725_72545

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  Real.cos (Real.pi + x) * Real.cos ((3/2) * Real.pi - x) - 
  Real.sqrt 3 * (Real.cos x)^2 + Real.sqrt 3 / 2

-- Theorem statement
theorem f_properties :
  -- 1. Smallest positive period is π
  (∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
    (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T')) ∧
  -- 2. Maximum value is 1
  (∀ (x : ℝ), f x ≤ 1) ∧ (∃ (x : ℝ), f x = 1) ∧
  -- 3. Monotonically increasing interval on [π/6, 2π/3] is [π/6, 5π/12]
  (∀ (x y : ℝ), Real.pi/6 ≤ x ∧ x < y ∧ y ≤ 5*Real.pi/12 → f x < f y) ∧
  (∀ (x y : ℝ), 5*Real.pi/12 < x ∧ x < y ∧ y ≤ 2*Real.pi/3 → f x ≥ f y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l725_72545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_variation_solution_l725_72536

-- Define the inverse variation relationship
def inverse_variation (x w : ℝ) : Prop := ∃ k : ℝ, x^2 * Real.sqrt w = k

-- State the theorem
theorem inverse_variation_solution :
  ∀ x w : ℝ,
  inverse_variation x w →
  (x = 3 ∧ w = 4) →
  (x = 6 → w = 1/4) :=
by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_variation_solution_l725_72536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_equals_ln_three_halves_l725_72527

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 / (x^2 - 1)

-- Define the bounds of integration
def lower_bound : ℝ := 2
def upper_bound : ℝ := 3

-- State the theorem
theorem area_equals_ln_three_halves :
  ∫ x in lower_bound..upper_bound, f x = Real.log (3/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_equals_ln_three_halves_l725_72527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fractional_linear_function_exists_unique_l725_72552

/-- A fractional-linear function from ℝ to ℝ. -/
noncomputable def FractionalLinearFunction (a b c d : ℝ) : ℝ → ℝ :=
  fun x => (a * x + b) / (c * x + d)

/-- The theorem stating the existence and uniqueness of a fractional-linear function
    mapping three distinct points to three other distinct points. -/
theorem fractional_linear_function_exists_unique
  (x₁ x₂ x₃ y₁ y₂ y₃ : ℝ)
  (hx : x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃)
  (hy : y₁ ≠ y₂ ∧ y₁ ≠ y₃ ∧ y₂ ≠ y₃) :
  ∃! f : ℝ → ℝ, ∃ a b c d : ℝ,
    f = FractionalLinearFunction a b c d ∧
    a * d - b * c ≠ 0 ∧
    f x₁ = y₁ ∧ f x₂ = y₂ ∧ f x₃ = y₃ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fractional_linear_function_exists_unique_l725_72552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_intersection_cardinality_l725_72518

def A : Finset ℕ := {4, 5, 7, 9}
def B : Finset ℕ := {3, 4, 7, 8, 9}
def U : Finset ℕ := A ∪ B

theorem complement_intersection_cardinality : Finset.card (U \ (A ∩ B)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_intersection_cardinality_l725_72518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l725_72580

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then Real.exp x + a else x^2 + 1 + a

theorem f_range (a : ℝ) (h : ∀ x, f a (2 - x) ≥ f a x) :
  ∀ x, x ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l725_72580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_small_square_l725_72559

-- Define the large square
def large_square_side : ℝ := 100

-- Define the number of smaller squares
def num_small_squares : ℕ := 100000

-- Define a structure for a square
structure Square where
  side : ℝ
  center : ℝ × ℝ

-- Define a property for the smaller squares
def small_squares (squares : Finset Square) : Prop :=
  squares.card = num_small_squares ∧
  ∀ s ∈ squares, s.side ≤ large_square_side ∧
  ∀ s1 s2 : Square, s1 ∈ squares → s2 ∈ squares → s1 ≠ s2 → 
    (s1.center.1 - s2.center.1)^2 + (s1.center.2 - s2.center.2)^2 ≥ (0.49)^2

-- The theorem to be proved
theorem exists_small_square (squares : Finset Square) 
  (h : small_squares squares) : 
  ∃ s ∈ squares, s.side < 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_small_square_l725_72559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_and_min_value_l725_72595

-- Define the linear function
def linear_function (k b : ℝ) (x : ℝ) : ℝ := k * x + b

-- Define the theorem
theorem area_and_min_value (k b : ℝ) (h1 : k * b ≠ 0) :
  let f := linear_function k b
  let A := (-b / k, 0)
  let B := (0, b)
  let O := (0, 0)
  let S := abs (b^2 - b) / 2
  (f 1 = k * b) →
  (S = abs (A.1 * B.2) / 2) ∧
  (b ≥ 2 → ∀ b' ≥ 2, S ≤ abs (b'^2 - b') / 2) ∧
  (b ≥ 2 → S = 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_and_min_value_l725_72595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pension_insurance_payment_l725_72569

/-- Represents the annual pension insurance payment for person A in thousand yuan -/
def personA_annual_payment : ℝ := sorry

/-- Represents the total pension insurance payment for person A in thousand yuan -/
def personA_total_payment : ℝ := 12

/-- Represents the total pension insurance payment for person B in thousand yuan -/
def personB_total_payment : ℝ := 8

/-- Represents the difference in annual payments between person A and person B in thousand yuan -/
def payment_difference : ℝ := 0.1

/-- Represents the difference in years of payment between person A and person B -/
def years_difference : ℕ := 4

/-- Represents the maximum number of years either person can pay for pension insurance -/
def max_years : ℕ := 20

theorem pension_insurance_payment :
  personA_total_payment / personA_annual_payment -
  personB_total_payment / (personA_annual_payment - payment_difference) = years_difference ∧
  personA_total_payment / personA_annual_payment ≤ max_years ∧
  personB_total_payment / (personA_annual_payment - payment_difference) ≤ max_years →
  personA_annual_payment = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pension_insurance_payment_l725_72569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_angle_formula_l725_72576

theorem tan_half_angle_formula (α : Real) 
  (h1 : α ∈ Set.Ioo π (3*π/2))  -- α is in the third quadrant
  (h2 : Real.cos α = -4/5) : 
  (1 + Real.tan (α/2)) / (1 - Real.tan (α/2)) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_angle_formula_l725_72576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_l725_72521

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 15*x^2 - 33*x + 6

-- Theorem statement
theorem monotonic_decreasing_interval :
  ∀ x y, x ∈ Set.Ioo (-1 : ℝ) 11 → y ∈ Set.Ioo (-1 : ℝ) 11 →
    x < y → f x > f y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_l725_72521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_subset_with_property_l725_72513

-- Define the set M
def M : Finset ℕ := Finset.filter (λ n => 1 ≤ n ∧ n ≤ 20) (Finset.range 21)

-- Define the function f
def f : {S : Finset ℕ // S ⊆ M ∧ S.card = 9} → ℕ :=
λ S => sorry -- We don't need to define the actual function, just its type

-- Axiom for the range of f
axiom f_range (S : {S : Finset ℕ // S ⊆ M ∧ S.card = 9}) : 1 ≤ f S ∧ f S ≤ 20

-- Theorem statement
theorem exists_subset_with_property :
  ∃ T : Finset ℕ, T ⊆ M ∧ T.card = 10 ∧
  ∀ k ∈ T, f ⟨T.erase k, sorry⟩ ≠ k :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_subset_with_property_l725_72513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_antibiotic_residual_coefficient_l725_72537

/-- Given an equation for the residual amount of antibiotics and a specific condition, 
    prove that the residual coefficient lambda is equal to 1/4. -/
theorem antibiotic_residual_coefficient :
  ∀ (y lambda : ℝ) (t : ℝ),
    lambda ≠ 0 →
    (∀ t, y = lambda * (1 - 3^(-lambda * t))) →
    (lambda * (1 - 3^(-lambda * 8)) = (8/9) * lambda) →
    lambda = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_antibiotic_residual_coefficient_l725_72537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_ngon_containing_triangle_l725_72520

/-- A convex N-gon. -/
structure ConvexPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ
  convex : Convex ℝ (Set.range vertices)

/-- The property that three sides of a polygon, when extended, form a triangle containing the polygon. -/
def HasContainingTriangle {n : ℕ} (p : ConvexPolygon n) : Prop :=
  ∃ (i j k : Fin n), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    let triangle := convexHull ℝ {p.vertices i, p.vertices j, p.vertices k}
    Set.range p.vertices ⊆ triangle

/-- Theorem: Any convex N-gon with N ≥ 5 has three sides whose extensions form a triangle containing the N-gon. -/
theorem convex_ngon_containing_triangle {n : ℕ} (hn : n ≥ 5) (p : ConvexPolygon n) :
  HasContainingTriangle p := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_ngon_containing_triangle_l725_72520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_for_odd_g_l725_72539

-- Define the determinant operation for 2x2 matrix
def det (a₁ a₂ a₃ a₄ : ℝ) : ℝ := a₁ * a₄ - a₂ * a₃

-- Define the original function f
noncomputable def f (x : ℝ) : ℝ := det (-Real.sin x) (Real.cos x) 1 (-Real.sqrt 3)

-- Define the translated function g
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := f (x + m)

-- Theorem statement
theorem min_m_for_odd_g :
  ∃ m : ℝ, m > 0 ∧ 
  (∀ x : ℝ, g m x = -g m (-x)) ∧
  (∀ m' : ℝ, m' > 0 → (∀ x : ℝ, g m' x = -g m' (-x)) → m ≤ m') ∧
  m = Real.pi / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_for_odd_g_l725_72539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_of_equation_l725_72585

theorem roots_of_equation (x : ℝ) :
  (3 * Real.sqrt x + 3 * x^(-(1/2 : ℝ)) = 7) ↔
  (x = ((7 + Real.sqrt 13) / 6)^2 ∨ x = ((7 - Real.sqrt 13) / 6)^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_of_equation_l725_72585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_isosceles_triangle_l725_72586

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola -/
structure Hyperbola where
  m : ℝ
  f1 : Point  -- Left focus
  f2 : Point  -- Right focus

/-- Checks if a point is on the hyperbola -/
def onHyperbola (h : Hyperbola) (p : Point) : Prop :=
  p.x^2 - p.y^2 / h.m = 1

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem about the hyperbola and isosceles triangle -/
theorem hyperbola_isosceles_triangle 
  (h : Hyperbola) 
  (a b : Point) 
  (ha : onHyperbola h a)
  (hb : onHyperbola h b)
  (h_isosceles : distance a h.f1 = distance b h.f1)
  (h_right_angle : distance a h.f1 * distance a h.f1 = 
                   distance a h.f2 * distance a h.f2 + 
                   distance h.f1 h.f2 * distance h.f1 h.f2) :
  h.m = 2 * (Real.sqrt 2 - 1) ∧ 
  1/2 * distance h.f1 h.f2 * distance a h.f2 = 4 - 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_isosceles_triangle_l725_72586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_cube_resistance_l725_72541

/-- Represents a wire cube with unit resistance on each edge -/
structure WireCube where
  edge_resistance : ℝ
  edge_resistance_positive : edge_resistance > 0

/-- Calculates the resistance between opposite vertices of a wire cube -/
noncomputable def opposite_vertex_resistance (cube : WireCube) : ℝ :=
  5 / 6 * cube.edge_resistance

/-- Theorem: The resistance between opposite vertices of a wire cube
    with unit resistance edges is 5/6 ohms -/
theorem wire_cube_resistance (cube : WireCube) 
  (h : cube.edge_resistance = 1) : 
  opposite_vertex_resistance cube = 5/6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_cube_resistance_l725_72541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_expression_evaluation_l725_72507

-- Define the floor function as noncomputable
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- State the theorem
theorem floor_expression_evaluation :
  (floor 6.5) * (floor (2/3 : ℝ)) + (floor 2) * (7.2 : ℝ) + (floor 8.3) - (6.6 : ℝ) = 9.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_expression_evaluation_l725_72507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l725_72500

/-- The function f(x) = (3x + 5) / (x + 4) -/
noncomputable def f (x : ℝ) : ℝ := (3 * x + 5) / (x + 4)

/-- The range of f is (-∞, 3) ∪ (3, ∞) -/
theorem range_of_f : Set.range f = {y : ℝ | y < 3 ∨ y > 3} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l725_72500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l725_72579

/-- The distance between the foci of an ellipse with equation 9x^2 + 16y^2 = 144 is 2√7 -/
theorem ellipse_foci_distance :
  ∃ (f₁ f₂ : ℝ × ℝ), 
    (∀ x y : ℝ, 9 * x^2 + 16 * y^2 = 144 → 
      ((x - f₁.1)^2 + (y - f₁.2)^2) + ((x - f₂.1)^2 + (y - f₂.2)^2) = 16) ∧
    Real.sqrt ((f₁.1 - f₂.1)^2 + (f₁.2 - f₂.2)^2) = 2 * Real.sqrt 7 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l725_72579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_slope_l725_72570

-- Define the circle C
def circleC (x y : ℝ) : Prop := x^2 + y^2 + 4*x = 0

-- Define the point through which the chord passes
noncomputable def chord_point : ℝ × ℝ := (-2, Real.sqrt 3)

-- Define the chord length
noncomputable def chord_length : ℝ := 2 * Real.sqrt 3

-- Define the slope of the line
def line_slope (k : ℝ) : Prop := True

-- Theorem statement
theorem chord_slope :
  ∃ k : ℝ, line_slope k ∧ 
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    x₁ ≠ x₂ ∧
    circleC x₁ y₁ ∧ 
    circleC x₂ y₂ ∧
    y₁ - chord_point.2 = k * (x₁ - chord_point.1) ∧
    y₂ - chord_point.2 = k * (x₂ - chord_point.1) ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = chord_length^2) →
  k = Real.sqrt 2 ∨ k = -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_slope_l725_72570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_function_classification_l725_72516

open Real

noncomputable def is_z_function (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ > 0 → x₂ > 0 → x₁ ≠ x₂ → x₁ * f x₁ + x₂ * f x₂ > x₁ * f x₂ + x₂ * f x₁

noncomputable def f₁ (x : ℝ) : ℝ := -x^3 + 1
noncomputable def f₂ (x : ℝ) : ℝ := x + sin x
noncomputable def f₃ (x : ℝ) : ℝ := exp x * (2*x - 1)
noncomputable def f₄ (x : ℝ) : ℝ := 2*(x - log x) + (2*x - 1)/x^2

theorem z_function_classification :
  ¬(is_z_function f₁) ∧ 
  (is_z_function f₂) ∧ 
  (is_z_function f₃) ∧ 
  (is_z_function f₄) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_function_classification_l725_72516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_and_correct_l725_72530

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  if x > 0 then -x^2 + x 
  else -x^2 - x

-- State the theorem
theorem f_is_even_and_correct : 
  (∀ x, f x = f (-x)) ∧ 
  (∀ x, x > 0 → f x = -x^2 + x) ∧ 
  (∀ x, x < 0 → f x = -x^2 - x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_and_correct_l725_72530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bella_ella_meeting_l725_72502

/-- The problem setup for Bella and Ella's meeting --/
structure MeetingProblem where
  total_distance : ℕ  -- Distance between houses in feet
  speed_ratio : ℕ     -- Ratio of Ella's speed to Bella's speed
  step_length : ℕ     -- Length of Bella's step in feet

/-- Calculate the number of steps Bella takes before meeting Ella --/
def steps_taken (p : MeetingProblem) : ℕ :=
  p.total_distance * 2 / (p.speed_ratio + 1) / p.step_length

/-- The main theorem to prove --/
theorem bella_ella_meeting :
  let p : MeetingProblem := {
    total_distance := 15840,
    speed_ratio := 4,
    step_length := 3
  }
  steps_taken p = 1056 := by
  sorry

#eval steps_taken { total_distance := 15840, speed_ratio := 4, step_length := 3 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bella_ella_meeting_l725_72502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_intercept_of_parallel_line_l725_72549

/-- Given two lines in a plane, where:
    1. Line b is parallel to the line y = -3x + 6
    2. Line b passes through the point (3, -2)
    This theorem proves that the y-intercept of line b is 7. -/
theorem y_intercept_of_parallel_line (b : Set (ℝ × ℝ)) :
  (∃ k, ∀ x y, (x, y) ∈ b ↔ y = -3 * x + k) →  -- b is parallel to y = -3x + 6
  (3, -2) ∈ b →                               -- b contains point (3, -2)
  (0, 7) ∈ b :=                               -- y-intercept of b is (0, 7)
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_intercept_of_parallel_line_l725_72549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l725_72587

/-- Parabola with parameter p > 0 -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- Point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line on a 2D plane -/
structure Line where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- Given a parabola y^2 = 2px, its focus F, and point D(p, 0), prove:
    1. When MD ⊥ x-axis and |MF| = 3, the parabola equation is y^2 = 4x
    2. When α - β is maximum, AB: x - √2y - 4 = 0 -/
theorem parabola_properties (C : Parabola) (F : Point) (D : Point)
    (h_D : D.x = C.p ∧ D.y = 0)
    (M N A B : Point)
    (h_M : M.y^2 = 2 * C.p * M.x)
    (h_N : N.y^2 = 2 * C.p * N.x)
    (h_A : A.y^2 = 2 * C.p * A.x)
    (h_B : B.y^2 = 2 * C.p * B.x)
    (h_MF : (M.x - F.x)^2 + (M.y - F.y)^2 = 9)
    (h_MD_perp : M.x = D.x)
    (MN AB : Line)
    (h_MN : MN.m = (N.y - M.y) / (N.x - M.x))
    (h_AB : AB.m = (B.y - A.y) / (B.x - A.x)) :
    (C.p = 2 ∧ AB.m = -1/Real.sqrt 2 ∧ AB.b = 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l725_72587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xy_is_zero_l725_72535

theorem xy_is_zero (x y : ℝ) 
  (h1 : (2 : ℝ)^x = (16 : ℝ)^(y + 1)) 
  (h2 : (64 : ℝ)^y = (4 : ℝ)^(x - 2)) : 
  x * y = 0 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_xy_is_zero_l725_72535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l725_72550

noncomputable def f (x : ℝ) : ℝ := (1 + x^2) / (1 - x^2)

theorem f_properties :
  -- f is defined for all x ≠ ±1
  ∀ x : ℝ, x ≠ 1 ∧ x ≠ -1 →
  -- 1. f is an even function
  (f x = f (-x)) ∧
  -- 2. f(1/x) = -f(x)
  (x ≠ 0 → f (1/x) = -f x) ∧
  -- 3. The range of f is (-∞, -1) ∪ [1, +∞)
  (∀ y : ℝ, (∃ x : ℝ, x ≠ 1 ∧ x ≠ -1 ∧ f x = y) ↔ (y < -1 ∨ y ≥ 1)) :=
by
  sorry

#check f_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l725_72550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l725_72515

-- Define the hyperbola and its properties
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos : a > 0 ∧ b > 0

-- Define the foci and a point on the hyperbola
def Foci (h : Hyperbola) := (ℝ × ℝ) × (ℝ × ℝ)
def Point := ℝ × ℝ

-- Define the conditions given in the problem
def satisfies_conditions (h : Hyperbola) (f : Foci h) (p : Point) : Prop :=
  let (f₁, f₂) := f
  let (x, y) := p
  -- Point P is on the right branch of the hyperbola
  x^2 / h.a^2 - y^2 / h.b^2 = 1 ∧ x > 0 ∧
  -- (OP + OF₂) ⋅ F₂P = 0
  (x + f₂.1) * (f₂.1 - x) + (y + f₂.2) * (f₂.2 - y) = 0 ∧
  -- |PF₁| = √3|PF₂|
  ((x - f₁.1)^2 + (y - f₁.2)^2) = 3 * ((x - f₂.1)^2 + (y - f₂.2)^2)

-- Define eccentricity
noncomputable def eccentricity (h : Hyperbola) : ℝ := Real.sqrt (1 + h.b^2 / h.a^2)

-- State the theorem
theorem hyperbola_eccentricity (h : Hyperbola) (f : Foci h) (p : Point) 
  (h_cond : satisfies_conditions h f p) : 
  eccentricity h = Real.sqrt 3 + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l725_72515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_of_line_segment_l725_72590

noncomputable def is_valid_midpoint (r θ : ℝ) : Prop :=
  r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi

noncomputable def endpoint1 : ℝ × ℝ := (10, Real.pi / 6)
noncomputable def endpoint2 : ℝ × ℝ := (10, 11 * Real.pi / 6)

theorem midpoint_of_line_segment :
  let midpoint : ℝ × ℝ := (10, 0)
  is_valid_midpoint midpoint.1 midpoint.2 ∧
  ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧
    midpoint = (t * endpoint1.1 + (1 - t) * endpoint2.1,
                t * endpoint1.2 + (1 - t) * endpoint2.2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_of_line_segment_l725_72590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_seven_l725_72563

-- Define the arithmetic sequence
noncomputable def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

-- Define the sum of the first n terms of an arithmetic sequence
noncomputable def arithmetic_sum (a₁ d : ℝ) (n : ℕ) : ℝ := n * (2 * a₁ + (n - 1 : ℝ) * d) / 2

-- State the theorem
theorem arithmetic_sequence_sum_seven (a₁ d : ℝ) :
  d < 0 ∧  -- Decreasing sequence
  arithmetic_sequence a₁ d 3 = -1 ∧  -- a₃ = -1
  (arithmetic_sequence a₁ d 4)^2 = a₁ * (-arithmetic_sequence a₁ d 6) →  -- a₄ is geometric mean of a₁ and -a₆
  arithmetic_sum a₁ d 7 = -14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_seven_l725_72563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l725_72547

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (3 * x - 2) ^ (-(3/4 : ℝ))

-- State the theorem about the domain of the function
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x > 2/3} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l725_72547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangents_count_l725_72532

/-- Circle represented by its general equation -/
structure Circle where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ

/-- Calculate the center and radius of a circle from its general equation -/
noncomputable def circle_center_radius (c : Circle) : (ℝ × ℝ) × ℝ :=
  let center_x := -c.a / 2
  let center_y := -c.b / 2
  let radius := Real.sqrt ((c.a^2 + c.b^2) / 4 - c.c)
  ((center_x, center_y), radius)

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Theorem: The number of common tangents to the given circles is 3 -/
theorem common_tangents_count (c1 c2 : Circle)
  (h1 : c1 = ⟨1, 1, 1, 2, 2⟩)
  (h2 : c2 = ⟨1, -1, -3, -6, -6⟩) :
  let ((x1, y1), r1) := circle_center_radius c1
  let ((x2, y2), r2) := circle_center_radius c2
  distance (x1, y1) (x2, y2) = r1 + r2 ∧ r1 > 0 ∧ r2 > 0 → 3 = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangents_count_l725_72532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_radius_l725_72517

theorem parabola_circle_radius (n : ℕ) (r : ℝ) : 
  n = 6 →  -- Six copies of the parabola
  (∀ i : Fin n, ∃ a b c : ℝ, 
    (λ (x y : ℝ) => y = a * x^2 + b * x + c) = (λ (x y : ℝ) => y = x^2)) →  -- Congruent parabolas
  (∀ i : Fin n, ∃ x y : ℝ, x^2 + y^2 = r^2 ∧ y = x^2) →  -- Each vertex tangent to circle
  (∀ i : Fin n, ∃ j : Fin n, i ≠ j ∧ 
    ∃ x y : ℝ, y = x^2 ∧ y = (x - 1)^2) →  -- Each parabola tangent to neighbors
  (∀ i : Fin n, ∃ m b : ℝ, m = 1 ∧ 
    ∃ x y : ℝ, y = x^2 ∧ y = m * x + b) →  -- Each parabola tangent to 45-degree line
  r = 1/4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_radius_l725_72517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_sine_curve_l725_72514

theorem axis_of_symmetry_sine_curve :
  ∃ k : ℤ, (2 * Real.pi * (5/12 : ℝ) - Real.pi / 3 = k * Real.pi + Real.pi / 2) := by
  sorry

#check axis_of_symmetry_sine_curve

end NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_sine_curve_l725_72514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_factors_of_M_l725_72525

/-- The number of natural-number factors of M, where M = 2^4 · 3^3 · 5^2 · 7^1 -/
def num_factors_M : ℕ := 120

/-- M is defined as 2^4 · 3^3 · 5^2 · 7^1 -/
def M : ℕ := 2^4 * 3^3 * 5^2 * 7^1

/-- Theorem stating that the number of natural-number factors of M is equal to num_factors_M -/
theorem count_factors_of_M : (Finset.filter (· ∣ M) (Finset.range (M + 1))).card = num_factors_M := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_factors_of_M_l725_72525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l725_72598

variable (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem vector_problem (a b : V) 
  (h1 : ‖2 • a - 3 • b‖ = 2)
  (h2 : ‖3 • a + 2 • b‖ = 1) :
  (∃ (k : ℝ), ‖a + 5 • b‖ ≤ k ∧ ∀ (c : ℝ), ‖a + 5 • b‖ ≤ c → k ≤ c) ∧
  ‖b‖ / ‖a‖ = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l725_72598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_convergence_geometric_series_sum_example_series_convergence_l725_72551

noncomputable def geometric_series (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * (1 - r^n) / (1 - r)

theorem geometric_series_convergence (a : ℝ) (r : ℝ) (h : |r| < 1) :
  ∃ (L : ℝ), ∀ ε > 0, ∃ N, ∀ n ≥ N, |geometric_series a r n - L| < ε :=
sorry

theorem geometric_series_sum (a : ℝ) (r : ℝ) (h : |r| < 1) :
  ∃ (L : ℝ), L = a / (1 - r) :=
sorry

theorem example_series_convergence :
  let a : ℝ := 2
  let r : ℝ := 1/2
  ∃ (L : ℝ), (L = 4) ∧
    (∀ ε > 0, ∃ N, ∀ n ≥ N, |geometric_series a r n - L| < ε) ∧
    (∀ ε > 0, ∃ N, ∀ n ≥ N, |geometric_series a r n - 4| < ε) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_convergence_geometric_series_sum_example_series_convergence_l725_72551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_gangsters_l725_72543

/-- Represents a gang as a finite set of integers -/
def Gang := Finset ℕ

/-- Represents the set of all gangs -/
def AllGangs := Finset Gang

/-- Represents a gangster as a set of gangs they belong to -/
def Gangster := Finset Gang

/-- Defines the hostility relation between gangs -/
def IsHostile (g1 g2 : Gang) : Prop := sorry

/-- The total number of gangs -/
def TotalGangs : ℕ := 36

/-- The set of all gangsters -/
noncomputable def AllGangsters : Finset Gangster := sorry

/-- Axiom: No two gangsters belong to the same set of gangs -/
axiom distinct_gangsters : ∀ g1 g2 : Gangster, g1 ∈ AllGangsters → g2 ∈ AllGangsters → g1 ≠ g2 → g1 ≠ g2

/-- Axiom: No gangster belongs to two hostile gangs -/
axiom no_hostile_membership : ∀ gangster : Gangster, ∀ g1 g2 : Gang, 
  gangster ∈ AllGangsters → g1 ∈ gangster.val → g2 ∈ gangster.val → ¬(IsHostile g1 g2)

/-- Axiom: For each gangster, every gang they don't belong to is hostile to at least one gang they do belong to -/
axiom hostility_condition : ∀ gangster : Gangster, ∀ g : Gang, 
  gangster ∈ AllGangsters → g ∉ gangster.val → ∃ g' ∈ gangster.val, IsHostile g g'

/-- The main theorem: The maximum number of gangsters is 531441 -/
theorem max_gangsters : Finset.card AllGangsters = 531441 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_gangsters_l725_72543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l725_72542

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1 + Real.log (x + 1)) / x

-- State the theorem
theorem f_properties :
  -- f is decreasing on (0, +∞)
  (∀ x y : ℝ, x > 0 → y > 0 → x < y → f x > f y) ∧
  -- The maximum integer k such that f(x) > k / (x + 1) holds for all x > 0 is 3
  (∀ x : ℝ, x > 0 → f x > 3 / (x + 1)) ∧
  (∀ k : ℤ, (∀ x : ℝ, x > 0 → f x > (k : ℝ) / (x + 1)) → k ≤ 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l725_72542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_article_sale_theorem_l725_72561

/-- Represents the number of articles sold in the first sale -/
def x : ℝ := sorry

/-- Represents the number of articles to be sold in the second sale -/
noncomputable def y : ℝ := 0.3 * x - 9

/-- The selling price of x articles in the first sale -/
noncomputable def first_sale_price : ℝ := 100 + 5 * x

/-- The cost price of x articles -/
noncomputable def cost_price : ℝ := (100 + 5 * x) / 1.25

/-- The selling price of y articles in the second sale -/
noncomputable def second_sale_price : ℝ := 150 + 10 * y

theorem article_sale_theorem :
  (first_sale_price = 1.25 * cost_price) ∧
  (second_sale_price = 0.75 * cost_price) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_article_sale_theorem_l725_72561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_neg_thirty_one_sixths_pi_l725_72511

theorem sin_neg_thirty_one_sixths_pi : Real.sin (-31 / 6 * Real.pi) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_neg_thirty_one_sixths_pi_l725_72511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l725_72534

/-- Calculates the speed of a train given its length and time to cross a point -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  length / time

/-- Theorem: The speed of a 500m long train that crosses a point in 20 seconds is 25 m/s -/
theorem train_speed_calculation :
  train_speed 500 20 = 25 := by
  -- Unfold the definition of train_speed
  unfold train_speed
  -- Simplify the division
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l725_72534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_f_well_defined_l725_72522

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (-x^2 + x + 2)

-- State the theorem
theorem f_monotone_decreasing :
  ∀ x₁ x₂ : ℝ, 1/2 ≤ x₁ → x₁ < x₂ → x₂ ≤ 2 → f x₂ < f x₁ := by
  sorry

-- Define the domain of the function
def f_domain (x : ℝ) : Prop := -1 ≤ x ∧ x ≤ 2

-- State that the function is well-defined on its domain
theorem f_well_defined :
  ∀ x : ℝ, f_domain x → -x^2 + x + 2 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_f_well_defined_l725_72522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diatomic_gas_moles_theorem_l725_72528

/-- The number of moles of diatomic gas in a mixture of ideal gases -/
noncomputable def diatomic_gas_moles (C v R : ℝ) : ℝ := (C - 2 * v * R) / (3 * R)

/-- Theorem stating the relationship between the number of moles of diatomic gas
    and the system parameters -/
theorem diatomic_gas_moles_theorem (C v R : ℝ) (hC : C > 0) (hv : v > 0) (hR : R > 0) :
  ∃ (v' : ℝ), v' = diatomic_gas_moles C v R ∧ v' > 0 := by
  sorry

/-- Approximate calculation of the result -/
def approximate_result : ℚ := 
  let C : ℚ := 120
  let v : ℚ := 3/2
  let R : ℚ := 831/100
  (C - 2 * v * R) / (3 * R)

#eval approximate_result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diatomic_gas_moles_theorem_l725_72528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_real_root_iff_b_in_range_l725_72578

/-- The polynomial in question -/
def P (b x : ℝ) : ℝ := x^4 + b*x^3 + x^2 + b*x - 1

/-- The set of b values for which the polynomial has at least one real root -/
noncomputable def B : Set ℝ := Set.Icc (-Real.sqrt Real.pi) (-2 * Real.sqrt 3) ∪ Set.Icc 0 Real.pi

/-- Theorem stating the equivalence between the existence of a real root and b being in the set B -/
theorem polynomial_real_root_iff_b_in_range (b : ℝ) :
  (∃ x : ℝ, P b x = 0) ↔ b ∈ B := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_real_root_iff_b_in_range_l725_72578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_with_digit_sum_property_l725_72504

/-- Sum of digits of a natural number in base 10 -/
def digit_sum (m : ℕ) : ℕ :=
  if m < 10 then m else m % 10 + digit_sum (m / 10)

theorem arithmetic_sequence_with_digit_sum_property (n : ℕ) (hn : n > 0) :
  ∃ (a : ℕ → ℕ) (d : ℕ), 
    (d % 10 ≠ 0) ∧ 
    (∀ k, a (k + 1) = a k + d) ∧ 
    (∀ k, (digit_sum (a k)) > n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_with_digit_sum_property_l725_72504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_l725_72526

-- Define the set of digits
def digits : Finset Nat := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define the sum of all four-digit positive integers with non-repeating digits
noncomputable def T : Nat :=
  Finset.sum (Finset.powerset digits) (fun s => 
    if s.card = 4 then
      Finset.sum s (fun d => d * 10^(Finset.filter (· < d) s).card)
    else
      0)

-- Theorem statement
theorem sum_remainder : T % 1000 = 720 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_l725_72526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_always_large_difference_l725_72594

/-- A type representing a 20x20 grid filled with integers from 1 to 400 -/
def Grid := Fin 20 → Fin 20 → Fin 400

/-- The property that a grid has at least one row or column with a difference of at least 209 -/
def HasLargeDifference (g : Grid) : Prop :=
  (∃ i : Fin 20, ∃ j k : Fin 20, (g i j : ℕ) - (g i k : ℕ) ≥ 209) ∨
  (∃ j : Fin 20, ∃ i k : Fin 20, (g i j : ℕ) - (g k j : ℕ) ≥ 209)

/-- The theorem stating that any valid grid arrangement has a large difference -/
theorem always_large_difference (g : Grid) (h : ∀ i j : Fin 20, (g i j : ℕ) < 400) :
  HasLargeDifference g :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_always_large_difference_l725_72594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_omega_value_l725_72538

-- Define the function as noncomputable
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.tan (ω * x)

-- State the theorem
theorem max_omega_value :
  ∀ ω : ℝ, 
  ω > 0 → 
  (∀ x₁ x₂ : ℝ, -π/6 < x₁ ∧ x₁ < x₂ ∧ x₂ < π/4 → f ω x₁ < f ω x₂) →
  ω ≤ 2 :=
by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_omega_value_l725_72538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_one_closed_l725_72577

-- Define the set S as a subset of real numbers
variable (S : Set ℝ)

-- Define the closure under multiplication property for S
axiom S_closed_mult : ∀ a b : ℝ, a ∈ S → b ∈ S → (a * b) ∈ S

-- Define subsets T and U of S
variable (T U : Set ℝ)

-- T and U are subsets of S
axiom T_subset_S : T ⊆ S
axiom U_subset_S : U ⊆ S

-- T and U are disjoint
axiom T_U_disjoint : T ∩ U = ∅

-- T and U form a partition of S
axiom T_U_partition : T ∪ U = S

-- Triple product property for T
axiom T_triple_product : ∀ a b c : ℝ, a ∈ T → b ∈ T → c ∈ T → (a * b * c) ∈ T

-- Triple product property for U
axiom U_triple_product : ∀ a b c : ℝ, a ∈ U → b ∈ U → c ∈ U → (a * b * c) ∈ U

-- The theorem to be proved
theorem at_least_one_closed (S T U : Set ℝ) : 
  (∀ x y : ℝ, x ∈ T → y ∈ T → (x * y) ∈ T) ∨ 
  (∀ x y : ℝ, x ∈ U → y ∈ U → (x * y) ∈ U) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_one_closed_l725_72577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_quad_pyramid_angles_l725_72581

/-- A regular quadrilateral pyramid -/
structure RegularQuadPyramid where
  /-- The cosine of the angle between two adjacent lateral faces -/
  k : ℝ

/-- The cosine of the angle between a lateral face and the base plane -/
noncomputable def lateral_base_angle (p : RegularQuadPyramid) : ℝ := Real.sqrt (-p.k)

theorem regular_quad_pyramid_angles (p : RegularQuadPyramid) :
  (lateral_base_angle p = Real.sqrt (-p.k)) ∧ (-1 < p.k ∧ p.k < 0) := by
  sorry

#check regular_quad_pyramid_angles

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_quad_pyramid_angles_l725_72581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_shaded_area_l725_72558

/-- Side length of the square and hexagon -/
noncomputable def side_length : ℝ := 3

/-- Area of the square in Figures A and B -/
noncomputable def square_area : ℝ := side_length ^ 2

/-- Area of the inscribed circle in Figure A -/
noncomputable def circle_area : ℝ := Real.pi * (side_length / 2) ^ 2

/-- Shaded area in Figure A -/
noncomputable def shaded_area_A : ℝ := square_area - circle_area

/-- Shaded area in Figure B (same as Figure A) -/
noncomputable def shaded_area_B : ℝ := shaded_area_A

/-- Area of the regular hexagon in Figure C -/
noncomputable def hexagon_area : ℝ := 3 * Real.sqrt 3 / 2 * side_length ^ 2

/-- Shaded area in Figure C -/
noncomputable def shaded_area_C : ℝ := hexagon_area - circle_area

theorem largest_shaded_area :
  shaded_area_C > shaded_area_A ∧ shaded_area_C > shaded_area_B := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_shaded_area_l725_72558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_movie_length_increase_is_16_percent_l725_72574

/-- Calculates the percentage increase in movie length -/
noncomputable def movie_length_increase (prev_length : ℝ) (prev_cost_per_min : ℝ) (new_cost_per_min : ℝ) (new_total_cost : ℝ) : ℝ :=
  let new_length := new_total_cost / new_cost_per_min
  let length_increase := new_length - prev_length
  (length_increase / prev_length) * 100

/-- Theorem stating that the movie length increase is 16% given the specified conditions -/
theorem movie_length_increase_is_16_percent :
  let prev_length : ℝ := 2 * 60  -- 2 hours in minutes
  let prev_cost_per_min : ℝ := 50
  let new_cost_per_min : ℝ := 2 * prev_cost_per_min
  let new_total_cost : ℝ := 1920
  movie_length_increase prev_length prev_cost_per_min new_cost_per_min new_total_cost = 16 := by
  sorry

-- Remove the #eval statement as it's not necessary for compilation
-- and might cause issues with noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_movie_length_increase_is_16_percent_l725_72574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_and_chord_division_l725_72531

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 5

-- Define the line
def my_line (m x y : ℝ) : Prop := m * x - y + 1 - m = 0

-- Define point P
def point_P : ℝ × ℝ := (1, 1)

-- Define the ratio condition
def ratio_condition (A B : ℝ × ℝ) : Prop :=
  let (xa, ya) := A
  let (xb, yb) := B
  let (xp, yp) := point_P
  ((xa - xp)^2 + (ya - yp)^2) / ((xb - xp)^2 + (yb - yp)^2) = 1/4

theorem circle_line_intersection_and_chord_division :
  (∀ m : ℝ, ∃ A B : ℝ × ℝ, A ≠ B ∧ my_circle A.1 A.2 ∧ my_circle B.1 B.2 ∧ my_line m A.1 A.2 ∧ my_line m B.1 B.2) ∧
  (∀ A B : ℝ × ℝ, my_circle A.1 A.2 → my_circle B.1 B.2 → ratio_condition A B →
    (A.1 - A.2 = 0 ∧ B.1 - B.2 = 0) ∨ (A.1 + A.2 - 2 = 0 ∧ B.1 + B.2 - 2 = 0)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_and_chord_division_l725_72531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_common_difference_l725_72505

-- Define an arithmetic sequence
def arithmetic_sequence (a₁ : ℝ) (d : ℝ) : ℕ → ℝ :=
  λ n ↦ a₁ + (n - 1) * d

-- Define the sum of the first n terms of an arithmetic sequence
noncomputable def sum_arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  n * (2 * a₁ + (n - 1) * d) / 2

-- Theorem statement
theorem arithmetic_sequence_common_difference :
  ∀ a₁ d : ℝ,
  let a := arithmetic_sequence a₁ d
  sum_arithmetic_sequence a₁ d 8 = 48 →
  a 3 + a 4 = 8 →
  d = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_common_difference_l725_72505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_at_4_or_5_l725_72555

/-- An increasing arithmetic sequence with a_1 + a_9 = 0 -/
structure IncreasingArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h1 : ∀ n, a (n + 1) = a n + d
  h2 : d > 0
  h3 : a 1 + a 9 = 0

/-- Sum of the first n terms of the sequence -/
def S (seq : IncreasingArithmeticSequence) (n : ℕ) : ℚ :=
  n * seq.a 1 + (n * (n - 1) / 2) * seq.d

theorem min_sum_at_4_or_5 (seq : IncreasingArithmeticSequence) :
  ∀ k : ℕ, k ≥ 1 → (S seq 4 ≤ S seq k ∧ S seq 5 ≤ S seq k) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_at_4_or_5_l725_72555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l725_72548

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (4 - x) / (x - 3)

-- Define the domain of f
def domain_f : Set ℝ := {x : ℝ | x < 4 ∧ x ≠ 3}

-- Theorem stating that domain_f is the correct domain for f
theorem domain_of_f : 
  ∀ x : ℝ, (x ∈ domain_f) ↔ (∃ y : ℝ, f x = y) := by
  sorry

#check domain_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l725_72548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_caesar_sum_extended_l725_72599

noncomputable def caesar_sum (P : List ℝ) : ℝ :=
  let n := P.length
  let S := List.scanl (·+·) 0 P
  (S.sum - S.head!) / n

theorem caesar_sum_extended (P : List ℝ) (h : P.length = 99) 
  (h_sum : caesar_sum P = 1000) :
  caesar_sum (1 :: P) = 991 := by
  sorry

#check caesar_sum_extended

end NUMINAMATH_CALUDE_ERRORFEEDBACK_caesar_sum_extended_l725_72599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_AB_l725_72573

/-- Represents a point on a line segment --/
structure Point where
  position : ℝ

/-- Represents a line segment with two endpoints --/
structure LineSegment where
  start : Point
  finish : Point

/-- The length of a line segment --/
def length (segment : LineSegment) : ℝ := segment.finish.position - segment.start.position

/-- A point divides a line segment in a given ratio --/
def divides_in_ratio (p : Point) (segment : LineSegment) (ratio_left ratio_right : ℕ) : Prop :=
  (p.position - segment.start.position) / (segment.finish.position - segment.start.position) = ratio_left / (ratio_left + ratio_right)

theorem length_of_AB (A B P Q : Point) :
  P.position < Q.position →
  divides_in_ratio P { start := A, finish := B } 3 4 →
  divides_in_ratio Q { start := A, finish := B } 4 5 →
  length { start := P, finish := Q } = 5 →
  length { start := A, finish := B } = 315 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_AB_l725_72573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_property_l725_72510

-- Define a polynomial with real coefficients
def RealPolynomial := Polynomial ℝ

-- Define the property that P(x^2) * P(x^3) = P(x)^5 for all real x
def SatisfiesProperty (P : RealPolynomial) : Prop :=
  ∀ x : ℝ, (P.eval (x^2)) * (P.eval (x^3)) = (P.eval x)^5

-- Theorem statement
theorem polynomial_property :
  ∀ P : RealPolynomial, SatisfiesProperty P →
  ∃ n : ℕ, ∀ x : ℝ, P.eval x = x^n :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_property_l725_72510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l725_72593

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the area function for a triangle
noncomputable def Triangle.area (t : Triangle) : ℝ := 
  1/2 * t.a * t.b * Real.sin t.C

-- Define the main theorem
theorem triangle_problem (abc : Triangle) 
  (h1 : 2 * Real.cos abc.C * (abc.a * Real.cos abc.C + abc.c * Real.cos abc.A) + abc.b = 0) :
  abc.C = 2 * π / 3 ∧ 
  (abc.b = 2 → abc.c = 2 * Real.sqrt 3 → abc.a = 2 → Triangle.area abc = Real.sqrt 3) :=
by sorry

-- Define the law of sines
axiom law_of_sines (t : Triangle) : 
  t.a / Real.sin t.A = t.b / Real.sin t.B

-- Define the law of cosines
axiom law_of_cosines (t : Triangle) : 
  t.c^2 = t.a^2 + t.b^2 - 2 * t.a * t.b * Real.cos t.C

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l725_72593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_joey_age_next_multiple_sum_10_l725_72501

-- Define the ages of Joey, Mia, and Tim
def joey_age : ℕ → ℕ := sorry
def mia_age : ℕ → ℕ := sorry
def tim_age : ℕ → ℕ := sorry

-- n represents the number of years passed since the initial condition
axiom age_relations (n : ℕ) : 
  joey_age n = mia_age n + 2 ∧ 
  tim_age n = 2 + n

-- Mia's age is a multiple of Tim's age for the first time today, and will be 5 more times
axiom mia_age_multiple : 
  ∃ (k : ℕ), mia_age 0 = k * tim_age 0 ∧
  (∀ m, 0 < m → m < 6 → ∃ (l : ℕ), mia_age m = l * tim_age m)

-- Function to calculate the sum of digits
def sum_of_digits (n : ℕ) : ℕ := 
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

-- The theorem to prove
theorem joey_age_next_multiple_sum_10 : 
  ∃ (n : ℕ), 
    n > 0 ∧ 
    joey_age n % tim_age n = 0 ∧
    (∀ m, 0 < m → m < n → joey_age m % tim_age m ≠ 0) ∧
    sum_of_digits (joey_age n) = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_joey_age_next_multiple_sum_10_l725_72501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_no_squares_iff_prime_l725_72523

def d (a : ℕ) : ℕ := (Nat.divisors a).card

def seq (k : ℕ) : ℕ → ℕ
  | 0 => k
  | n + 1 => d (seq k n)

theorem sequence_no_squares_iff_prime (k : ℕ) (h : k ≥ 2) :
  (∀ n, ¬ ∃ m, (seq k n) = m ^ 2) ↔ Nat.Prime k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_no_squares_iff_prime_l725_72523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_increase_closest_to_515_l725_72588

/- Define the radius of the circular pizza -/
def circular_radius : ℝ := 7

/- Define the side length of the square pizza -/
def square_side : ℝ := 5

/- Define the area of the circular pizza -/
noncomputable def circular_area : ℝ := Real.pi * circular_radius^2

/- Define the area of the square pizza -/
def square_area : ℝ := square_side^2

/- Define the percentage increase in area -/
noncomputable def percentage_increase : ℝ := ((circular_area - square_area) / square_area) * 100

/- Theorem stating that the percentage increase is closest to 515 -/
theorem percentage_increase_closest_to_515 : 
  ∃ (n : ℤ), n = 515 ∧ ∀ (m : ℤ), m ≠ n → |percentage_increase - ↑n| < |percentage_increase - ↑m| :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_increase_closest_to_515_l725_72588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_locus_area_l725_72572

-- Define the circle
noncomputable def circle_diameter (d : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 ≤ (d/2)^2}

-- Define the centroid of a triangle
noncomputable def centroid (a b c : ℝ × ℝ) : ℝ × ℝ :=
  ((a.1 + b.1 + c.1) / 3, (a.2 + b.2 + c.2) / 3)

-- Theorem statement
theorem centroid_locus_area 
  (a b : ℝ × ℝ) 
  (h_diameter : a.1^2 + a.2^2 = b.1^2 + b.2^2 ∧ (b.1 - a.1)^2 + (b.2 - a.2)^2 = 24^2) :
  let circle := circle_diameter 24
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (∀ c ∈ circle, c ≠ a ∧ c ≠ b →
      centroid a b c ∈ circle_diameter 8) ∧
    (Real.pi * radius^2 = 16 * Real.pi) := by
  sorry

#check centroid_locus_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_locus_area_l725_72572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unknown_dimension_is_15_l725_72591

/-- Represents the dimensions and cost parameters of a room to be whitewashed -/
structure RoomParameters where
  length : ℝ
  width : ℝ
  height : ℝ
  doorArea : ℝ
  windowArea : ℝ
  numWindows : ℕ
  costPerSqFt : ℝ
  totalCost : ℝ

/-- Calculates the unknown dimension of the room based on the given parameters -/
noncomputable def calculateUnknownDimension (params : RoomParameters) : ℝ :=
  let totalWallArea := 2 * (params.length * params.height + params.width * params.height)
  let areaToSubtract := params.doorArea + (params.numWindows : ℝ) * params.windowArea
  let netArea := totalWallArea - areaToSubtract
  (params.totalCost / params.costPerSqFt - (2 * params.length * params.height - areaToSubtract)) / (2 * params.height)

/-- Theorem stating that the unknown dimension of the room is 15 feet -/
theorem unknown_dimension_is_15 (params : RoomParameters)
  (h1 : params.length = 25)
  (h2 : params.height = 12)
  (h3 : params.doorArea = 18)
  (h4 : params.windowArea = 12)
  (h5 : params.numWindows = 3)
  (h6 : params.costPerSqFt = 7)
  (h7 : params.totalCost = 6342) :
  calculateUnknownDimension params = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unknown_dimension_is_15_l725_72591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_decimal_sum_l725_72564

theorem periodic_decimal_sum (a b : ℕ+) : 
  (a.val : ℚ) / b.val = 57 / 99 → 
  Nat.gcd a.val b.val = 1 → 
  a.val + b.val = 52 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_decimal_sum_l725_72564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_twice_min_distance_l725_72509

-- Define a point in a plane
def Point := ℝ × ℝ

-- Define the distance between two points
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem statement
theorem max_distance_twice_min_distance (points : Finset Point) 
  (h : points.card = 10) :
  ∃ (p q r s : Point), p ∈ points ∧ q ∈ points ∧ r ∈ points ∧ s ∈ points ∧
  distance p q ≥ 2 * distance r s ∧
  (∀ (x y : Point), x ∈ points → y ∈ points → distance x y ≤ distance p q) ∧
  (∀ (x y : Point), x ∈ points → y ∈ points → distance x y ≥ distance r s) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_twice_min_distance_l725_72509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_remainders_existence_l725_72568

theorem distinct_remainders_existence (p : ℕ) (hp : Nat.Prime p) (a : Fin p → ℤ) :
  ∃ k : ℤ, Finset.card (Finset.filter (λ i : Fin p => ∃ j : Fin p, (a i + i.val * k) % p = (a j + j.val * k) % p) Finset.univ) ≤ p / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_remainders_existence_l725_72568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l725_72560

-- Define the function f(x) = |sin x|
noncomputable def f (x : ℝ) : ℝ := |Real.sin x|

-- State the theorem
theorem f_properties :
  (∀ x y, x ∈ Set.Ioo 0 (Real.pi / 2) → y ∈ Set.Ioo 0 (Real.pi / 2) → x < y → f x < f y) ∧ 
  (∀ x, f (-x) = f x) ∧
  (∀ x, f (x + Real.pi) = f x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l725_72560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_f_l725_72512

-- Define the function
noncomputable def f (x : ℝ) : ℝ := x / (1 - Real.cos x)

-- State the theorem
theorem derivative_of_f :
  ∀ x : ℝ, x ≠ 0 → deriv f x = (1 - Real.cos x - x * Real.sin x) / ((1 - Real.cos x)^2) :=
by
  -- We use 'sorry' to skip the proof for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_f_l725_72512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_nonmultiple_l725_72565

theorem existence_of_nonmultiple (p : ℕ) (hp : Nat.Prime p) :
  ∃ a b : ℤ, ∀ m : ℤ, ¬(p : ℤ) ∣ (m^3 + 2017*a*m + b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_nonmultiple_l725_72565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_max_sum_of_sines_l725_72533

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := sin (π/4 + x) * sin (π/4 - x) + sqrt 3 * sin x * cos x

-- Theorem for the smallest positive period of f
theorem smallest_positive_period_of_f :
  ∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧
  (∀ (q : ℝ), q > 0 → (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  p = π := by
  sorry

-- Theorem for the maximum value of sin B + sin C
theorem max_sum_of_sines (A B C : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  f (A/2) = 1 →
  ∃ (M : ℝ), (∀ x y : ℝ, 0 < x ∧ 0 < y ∧ x + y = π - A →
    sin x + sin y ≤ M) ∧
  M = sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_max_sum_of_sines_l725_72533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_three_points_distance_l725_72575

/-- The hyperbola C: x²/4 - y²/3 = 1 -/
def hyperbola (x y : ℝ) : Prop := x^2/4 - y^2/3 = 1

/-- The right vertex of the hyperbola -/
def P : ℝ × ℝ := (2, 0)

/-- The line l passing through P with normal vector (1, -1) -/
def line_l (x y : ℝ) : Prop := y = x + 2

/-- A point is on the hyperbola C -/
def on_hyperbola (p : ℝ × ℝ) : Prop := hyperbola p.1 p.2

/-- The distance from a point to line l -/
noncomputable def distance_to_line (p : ℝ × ℝ) : ℝ := 
  |p.2 - p.1 - 2| / Real.sqrt 2

theorem hyperbola_three_points_distance :
  ∃ (d : ℝ), ∃ (p₁ p₂ p₃ : ℝ × ℝ),
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₂ ≠ p₃ ∧
    on_hyperbola p₁ ∧ on_hyperbola p₂ ∧ on_hyperbola p₃ ∧
    distance_to_line p₁ = d ∧ distance_to_line p₂ = d ∧ distance_to_line p₃ = d ∧
    (∀ (p : ℝ × ℝ), on_hyperbola p ∧ distance_to_line p = d → p = p₁ ∨ p = p₂ ∨ p = p₃) →
    d = Real.sqrt 2 / 2 ∨ d = 3 * Real.sqrt 2 / 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_three_points_distance_l725_72575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_side_length_l725_72567

theorem shortest_side_length (A B C : ℝ) (a b c : ℝ) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi →
  -- Given trigonometric values
  Real.sin A = 5/13 →
  Real.cos B = 3/5 →
  -- Side lengths are positive
  a > 0 ∧ b > 0 ∧ c > 0 →
  -- Longest side is 63
  max a (max b c) = 63 →
  -- Sine rule
  a / (Real.sin A) = b / (Real.sin B) ∧ b / (Real.sin B) = c / (Real.sin C) →
  -- Conclusion: shortest side is 25
  min a (min b c) = 25 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_side_length_l725_72567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_cosines_cubic_roots_l725_72582

noncomputable section

-- Define the cubic equation and its roots
def cubic_equation (p q r x : ℝ) : Prop := x^3 - p*x^2 + q*x - r = 0

-- Define the triangle inequality for the roots
def triangle_inequality (x₁ x₂ x₃ : ℝ) : Prop :=
  x₁ + x₂ > x₃ ∧ x₁ + x₃ > x₂ ∧ x₂ + x₃ > x₁

-- Define the sum of cosines of angles in a triangle
noncomputable def sum_of_cosines (x₁ x₂ x₃ : ℝ) : ℝ :=
  let s := (x₁ + x₂ + x₃) / 2
  (s - x₁) / (x₂ * x₃) + (s - x₂) / (x₁ * x₃) + (s - x₃) / (x₁ * x₂)

-- State the theorem
theorem sum_of_cosines_cubic_roots (p q r x₁ x₂ x₃ : ℝ) :
  cubic_equation p q r x₁ ∧
  cubic_equation p q r x₂ ∧
  cubic_equation p q r x₃ ∧
  x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧
  triangle_inequality x₁ x₂ x₃ →
  sum_of_cosines x₁ x₂ x₃ = (4*p*q - 6*r - p^3) / (2*r) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_cosines_cubic_roots_l725_72582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_betty_ann_speed_ratio_l725_72546

/-- Represents a point on the park's perimeter -/
inductive ParkPoint
| TopLeft
| TopRight
| BottomRight
| BottomLeft
| BottomEdge (x : ℚ)

/-- Represents a walker in the park -/
structure Walker where
  position : ParkPoint
  clockwise : Bool

def park_width : ℚ := 600
def park_height : ℚ := 400

def bottom_edge_segment : ℚ := park_width / 6

noncomputable def distance_to_point (w : Walker) (p : ParkPoint) : ℚ :=
  sorry

def betty : Walker :=
  { position := ParkPoint.TopLeft, clockwise := true }

def ann : Walker :=
  { position := ParkPoint.TopLeft, clockwise := false }

def point_q : ParkPoint := ParkPoint.BottomEdge (2 * bottom_edge_segment)
def point_r : ParkPoint := ParkPoint.BottomEdge (3 * bottom_edge_segment)

theorem betty_ann_speed_ratio :
  ∀ (meet_point : ParkPoint),
    (∃ x, meet_point = ParkPoint.BottomEdge x) →
    (∃ q r, point_q = ParkPoint.BottomEdge q ∧ 
            point_r = ParkPoint.BottomEdge r ∧ 
            meet_point = ParkPoint.BottomEdge x ∧ 
            q < x ∧ x < r) →
    (distance_to_point betty meet_point) / (distance_to_point ann meet_point) = 9 / 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_betty_ann_speed_ratio_l725_72546
