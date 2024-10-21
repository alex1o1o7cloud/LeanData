import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_solutions_to_equation_l1197_119797

theorem integer_solutions_to_equation : 
  {x : ℤ | (x - 3 : ℚ) ^ (36 - x^2) = 1} = {-6, 2, 4, 6} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_solutions_to_equation_l1197_119797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_slope_l1197_119743

/-- Proves that the slope of a line passing through (-3, 0) and tangent to the unit circle is ± √2/4 -/
theorem tangent_line_slope : 
  ∀ k : ℝ, 
  (∃ x y : ℝ, x^2 + y^2 = 1 ∧ y = k*(x + 3)) → 
  (k = Real.sqrt 2 / 4 ∨ k = -(Real.sqrt 2 / 4)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_slope_l1197_119743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_calculation_l1197_119712

/-- Calculates the future value of an investment using compound interest --/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (periods : ℕ) : ℝ :=
  principal * (1 + rate / 12) ^ (periods * 12)

/-- Represents the problem of finding the principal amount given specific conditions --/
theorem principal_calculation (P : ℝ) : 
  (compound_interest P 0.12 3 = P + (P - 5888)) → 
  ∃ ε > 0, |P - 10254.63| < ε := by
  sorry

#check principal_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_calculation_l1197_119712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_line_equation_l1197_119746

noncomputable section

/-- The projection of vector v onto vector u -/
def proj (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := u.1 * v.1 + u.2 * v.2
  let norm_squared := u.1 * u.1 + u.2 * u.2
  (dot_product / norm_squared * u.1, dot_product / norm_squared * u.2)

/-- The vector we're projecting onto -/
def u : ℝ × ℝ := (3, 4)

theorem projection_line_equation :
  ∀ (v : ℝ × ℝ), proj u v = u → v.2 = -3/4 * v.1 + 25/4 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_line_equation_l1197_119746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cups_facing_up_are_perfect_squares_l1197_119770

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def is_facing_up (n : ℕ) : Prop := 
  (Finset.filter (λ i ↦ n % i = 0) (Finset.range 13)).card % 2 = 1

theorem cups_facing_up_are_perfect_squares : 
  ∀ n : ℕ, n ≤ 12 → (is_facing_up n ↔ is_perfect_square n) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cups_facing_up_are_perfect_squares_l1197_119770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_term_is_negative_x_cubed_l1197_119771

-- Define the line equation
def line_equation (x : ℝ) : ℝ := x^2 - x^3

-- Define the property of touching x-axis in 2 places
def touches_x_axis_twice (f : ℝ → ℝ) : Prop :=
  ∃ x₁ x₂, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0

-- Theorem statement
theorem cubic_term_is_negative_x_cubed :
  touches_x_axis_twice line_equation →
  ∃ a b c d : ℝ, (∀ x, line_equation x = a*x^3 + b*x^2 + c*x + d) ∧ a = -1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_term_is_negative_x_cubed_l1197_119771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_doubling_point_properties_l1197_119721

-- Define the concept of a "doubling point"
def is_doubling_point (P Q : ℝ × ℝ) : Prop :=
  2 * (P.1 + Q.1) = P.2 + Q.2

-- Define the given point P₁
def P₁ : ℝ × ℝ := (1, 0)

-- Define the parabola
def on_parabola (P : ℝ × ℝ) : Prop :=
  P.2 = P.1^2 - 2*P.1 - 3

-- Define the distance between two points
noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- State the theorem
theorem doubling_point_properties :
  -- 1. Q₁ and Q₂ are doubling points of P₁
  is_doubling_point P₁ (3, 8) ∧ 
  is_doubling_point P₁ (-2, -2) ∧ 
  -- 2. Exactly two points on the parabola are doubling points of P₁
  ∃ (Q₁ Q₂ : ℝ × ℝ), Q₁ ≠ Q₂ ∧ 
    on_parabola Q₁ ∧ on_parabola Q₂ ∧ 
    is_doubling_point P₁ Q₁ ∧ is_doubling_point P₁ Q₂ ∧
    (∀ Q, on_parabola Q ∧ is_doubling_point P₁ Q → Q = Q₁ ∨ Q = Q₂) ∧
  -- 3. Minimum distance to any doubling point is 4√5/5
  (∀ Q, is_doubling_point P₁ Q → distance P₁ Q ≥ 4 * Real.sqrt 5 / 5) ∧
  (∃ Q, is_doubling_point P₁ Q ∧ distance P₁ Q = 4 * Real.sqrt 5 / 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_doubling_point_properties_l1197_119721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_zero_for_all_k_and_n_l1197_119768

def u (n : ℕ) : ℤ := n^4 + n^2

def Δ (f : ℕ → ℤ) (n : ℕ) : ℤ := f (n + 1) - f n

def iteratedΔ (k : ℕ) (f : ℕ → ℤ) : ℕ → ℤ :=
  match k with
  | 0 => f
  | k + 1 => Δ (iteratedΔ k f)

theorem not_zero_for_all_k_and_n :
  ∀ k : ℕ, k ≥ 1 → ∀ n : ℕ, iteratedΔ k u n ≠ 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_zero_for_all_k_and_n_l1197_119768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1197_119787

noncomputable def ω : ℝ := 2 / 3

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (ω * x) - 2 * Real.sin (ω * x / 2) ^ 2

theorem problem_solution :
  (∀ x, f (x + 3 * Real.pi) = f x) ∧
  (∀ x ∈ Set.Icc (-Real.pi) (3 * Real.pi / 4), f x ≤ 1) ∧
  (∃ x ∈ Set.Icc (-Real.pi) (3 * Real.pi / 4), f x = 1) ∧
  (∀ x ∈ Set.Icc (-Real.pi) (3 * Real.pi / 4), f x ≥ -3) ∧
  (∃ x ∈ Set.Icc (-Real.pi) (3 * Real.pi / 4), f x = -3) ∧
  (∀ a b c A B C : ℝ,
    a < b → b < c →
    Real.sqrt 3 * a = 2 * c * Real.sin A →
    A + B + C = Real.pi →
    C = 2 * Real.pi / 3) ∧
  (∀ a b c A B C : ℝ,
    a < b → b < c →
    Real.sqrt 3 * a = 2 * c * Real.sin A →
    A + B + C = Real.pi →
    f (3 / 2 * A + Real.pi / 2) = 11 / 13 →
    Real.cos B = (12 + 5 * Real.sqrt 3) / 26) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1197_119787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_independence_test_more_accurate_l1197_119779

-- Define the types for our methods and variables
inductive Method
| BarChart3D
| IndependenceTest

structure CategoricalVariable where
  dummy : Unit

-- Define a function to represent the accuracy of a method
noncomputable def accuracy (m : Method) : ℝ :=
  match m with
  | Method.BarChart3D => 0.5
  | Method.IndependenceTest => 0.8

-- Define a function to represent the relationship between two categorical variables
def relationship (x y : CategoricalVariable) : Prop :=
  True

-- State the theorem
theorem independence_test_more_accurate :
  ∀ (x y : CategoricalVariable),
    accuracy Method.IndependenceTest > accuracy Method.BarChart3D :=
by
  intros x y
  simp [accuracy]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_independence_test_more_accurate_l1197_119779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_week_travelers_l1197_119700

/-- The total number of travelers during a 7-day period given initial number and daily changes --/
theorem golden_week_travelers (a : ℝ) (changes : List ℝ) 
  (h : changes = [1.6, 0.8, 0.4, -0.4, -0.8, 0.2, -1.2]) : 
  7 * a + changes.sum = 7 * a + 13.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_week_travelers_l1197_119700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maintain_constant_product_l1197_119737

/-- Given a constant product K of three variables x, y, and z,
    if x is increased by 30% and z is decreased by 10%,
    then y must be decreased by approximately 14.53% to maintain K. -/
theorem maintain_constant_product (x y z K : ℝ) (h : K = x * y * z) :
  let x' := 1.3 * x
  let z' := 0.9 * z
  let y' := y / (1.3 * 0.9)
  x' * y' * z' = K ∧ abs ((y - y') / y - 0.1453) < 0.0001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_maintain_constant_product_l1197_119737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_range_l1197_119762

theorem inequality_range (a : ℝ) : 
  (∀ x θ : ℝ, θ ∈ Set.Icc 0 (π/2) → 
    (x + 3 + 2 * Real.sin θ * Real.cos θ)^2 + (x + a * Real.sin θ + a * Real.cos θ)^2 ≥ 1/8) ↔ 
  (a ≤ Real.sqrt 6 ∨ a ≥ 7/2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_range_l1197_119762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_inscribed_circle_area_sum_l1197_119745

-- Define the circle and triangle
def circle_radius : ℝ := 3

-- Define the extended lengths
def AD : ℝ := 17
def AE : ℝ := 15

-- Define the theorem
theorem equilateral_triangle_inscribed_circle_area_sum : 
  ∃ (p q r : ℕ), 
    p > 0 ∧ q > 0 ∧ r > 0 ∧
    Nat.Coprime p r ∧
    (∀ (prime : ℕ), Nat.Prime prime → ¬(prime^2 ∣ q)) ∧
    (∃ (area : ℝ), area = (p : ℝ) * Real.sqrt q / r) ∧
    p + q + r = 854 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_inscribed_circle_area_sum_l1197_119745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scores_variance_l1197_119755

noncomputable def scores : List ℝ := [110, 114, 121, 119, 126]

noncomputable def mean (xs : List ℝ) : ℝ := (xs.sum) / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  (xs.map (λ x => (x - mean xs)^2)).sum / xs.length

theorem scores_variance : variance scores = 30.8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scores_variance_l1197_119755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_alpha_value_l1197_119766

noncomputable def vector_a (α : ℝ) : ℝ × ℝ := (3/2, Real.sin α)
noncomputable def vector_b (α : ℝ) : ℝ × ℝ := (Real.cos α, 1/3)

def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem parallel_vectors_alpha_value (α : ℝ) 
  (h1 : 0 < α ∧ α < π/2)  -- α is an acute angle
  (h2 : parallel (vector_a α) (vector_b α)) :
  α = π/4 := by
  sorry

#check parallel_vectors_alpha_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_alpha_value_l1197_119766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_pair_l1197_119724

theorem unique_solution_pair : 
  ∃! p : ℝ × ℝ, let (x, y) := p; x = x^2 + y^2 + x ∧ y = 3*x*y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_pair_l1197_119724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l1197_119788

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then -Real.log x - x else -Real.log (-x) + x

theorem solution_set (m : ℝ) :
  f (1/m) < Real.log (1/2) - 2 ↔ m ∈ Set.Ioo (-1/2 : ℝ) 0 ∪ Set.Ioo 0 (1/2 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l1197_119788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l1197_119704

noncomputable def g (x : ℝ) : ℤ :=
  if x > -3 then
    ⌈(1 : ℝ) / (x + 3)^2⌉
  else
    ⌊(1 : ℝ) / (x + 3)^2⌋

theorem g_range (y : ℤ) : (∃ x : ℝ, x ≠ -3 ∧ g x = y) ↔ y > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l1197_119704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_points_and_meeting_times_l1197_119738

-- Define the points and their representations
noncomputable def A : ℝ := -12
noncomputable def B : ℝ := 20
def C : ℝ := 4

-- Define the speeds of points M and N
def speed_M : ℝ := 5
def speed_N : ℝ := 3

-- Define the condition for A and B
axiom condition : |A + 12| + (B - 20)^2 = 0

-- Define the theorem to prove
theorem points_and_meeting_times :
  -- Part 1: Values of A and B
  A = -12 ∧ B = 20 ∧
  -- Part 2.1: When CM = BN and M is between A and C
  (∃ t : ℝ, t = 2 ∧ C - (A + speed_M * t) = B - speed_N * t) ∧
  -- Part 2.2: Meeting times of M and N
  (∃ t₁ t₂ t₃ : ℝ, t₁ = 4 ∧ t₂ = 12 ∧ t₃ = 16 ∧
    (A + speed_M * t₁ = B - speed_N * t₁ ∨
     A + speed_M * t₂ - (B - A) = B - speed_N * t₂ ∨
     A + speed_M * t₃ - 2 * (B - A) = B - speed_N * t₃ + (C - A))) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_points_and_meeting_times_l1197_119738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1197_119741

/-- Given a hyperbola with equation (x^2 / a^2) - (y^2 / b^2) = 1, where a > 0 and b > 0,
    with left and right foci F₁ and F₂, and a point P on the right branch of the hyperbola
    such that a tangent to the circle x^2 + y^2 = a^2 passing through F₁ intersects the hyperbola at P,
    and ∠F₁PF₂ = 45°, prove that the eccentricity of the hyperbola is √3. -/
theorem hyperbola_eccentricity (a b : ℝ) (F₁ F₂ P : ℝ × ℝ) :
  a > 0 →
  b > 0 →
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ↔ (x, y) ∈ {z : ℝ × ℝ | z.1^2 / a^2 - z.2^2 / b^2 = 1}) →
  (∃ c : ℝ, F₁ = (-c, 0) ∧ F₂ = (c, 0) ∧ c = Real.sqrt (a^2 + b^2)) →
  P ∈ {z : ℝ × ℝ | z.1^2 / a^2 - z.2^2 / b^2 = 1} →
  P.1 > 0 →
  (∃ Q : ℝ × ℝ, Q ∈ {z : ℝ × ℝ | z.1^2 + z.2^2 = a^2} ∧ 
    (Q.2 - F₁.2) * (P.1 - Q.1) = (Q.1 - F₁.1) * (P.2 - Q.2)) →
  Real.arccos ((F₁.1 - P.1) * (F₂.1 - P.1) + (F₁.2 - P.2) * (F₂.2 - P.2)) / 
    (Real.sqrt ((F₁.1 - P.1)^2 + (F₁.2 - P.2)^2) * Real.sqrt ((F₂.1 - P.1)^2 + (F₂.2 - P.2)^2)) = π / 4 →
  Real.sqrt (a^2 + b^2) / a = Real.sqrt 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1197_119741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_points_collinear_l1197_119784

-- Define a point in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define collinearity for three points
def collinear (A B C : Point) : Prop :=
  (B.y - A.y) * (C.x - A.x) = (C.y - A.y) * (B.x - A.x)

-- Define the theorem
theorem all_points_collinear (E : Set Point) (h_finite : Set.Finite E) 
    (h_collinear : ∀ A B, A ∈ E → B ∈ E → ∃ C ∈ E, collinear A B C) :
    ∃ l : Set Point, (∀ P, P ∈ E → P ∈ l) ∧ 
    (∀ A B C, A ∈ l → B ∈ l → C ∈ l → collinear A B C) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_points_collinear_l1197_119784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_day_answer_l1197_119736

/-- Represents the days of the week -/
inductive Day where
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday
deriving Inhabited

/-- Represents the girl's behavior on a given day -/
def tells_truth (d : Day) : Bool :=
  match d with
  | Day.Thursday => true
  | Day.Friday => true
  | Day.Monday => false
  | _ => true || false  -- can be either true or false

/-- Represents the sequence of answers given by the girl -/
def answers : List Nat := [2010, 2011, 2012, 2013, 2002, 2011]

/-- The theorem to be proved -/
theorem seventh_day_answer :
  ∀ (week : List Day),
    week.length = 7 →
    (∀ i, i < 6 → tells_truth (week.get! i) = (answers.get! i = 2011)) →
    (week.get! 6 = Day.Thursday ∨ week.get! 0 = Day.Friday) →
    (answers.get! 6 = 2010 ∨ answers.get! 6 = 2011) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_day_answer_l1197_119736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_part_of_one_minus_sqrt_ten_l1197_119701

-- Define the integer part function
noncomputable def integerPart (x : ℝ) : ℤ :=
  ⌊x⌋

-- State the theorem
theorem integer_part_of_one_minus_sqrt_ten :
  integerPart (1 - Real.sqrt 10) = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_part_of_one_minus_sqrt_ten_l1197_119701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_all_black_value_l1197_119776

/-- Represents a 4x4 grid of squares -/
def Grid := Fin 4 → Fin 4 → Bool

/-- Probability of a single square being black -/
noncomputable def p_black : ℝ := 1/2

/-- Rotates a position 180 degrees in a 4x4 grid -/
def rotate (i j : Fin 4) : Fin 4 × Fin 4 := (3 - i, 3 - j)

/-- The probability that the entire grid is black after the rotation and recoloring process -/
noncomputable def prob_all_black : ℝ := (1/2)^8

theorem prob_all_black_value : prob_all_black = 1/65536 := by
  unfold prob_all_black
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_all_black_value_l1197_119776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_distance_to_plane_l1197_119752

/-- Given a tetrahedron with vertices A(2,3,1), B(4,1,-2), C(6,3,7), and D(-5,-4,8),
    the distance from vertex D to plane ABC is 119/√277 -/
theorem tetrahedron_distance_to_plane :
  let A : Fin 3 → ℝ := ![2, 3, 1]
  let B : Fin 3 → ℝ := ![4, 1, -2]
  let C : Fin 3 → ℝ := ![6, 3, 7]
  let D : Fin 3 → ℝ := ![-5, -4, 8]
  let AB : Fin 3 → ℝ := λ i => B i - A i
  let AC : Fin 3 → ℝ := λ i => C i - A i
  let AD : Fin 3 → ℝ := λ i => D i - A i
  let n : Fin 3 → ℝ := ![
    AB 1 * AC 2 - AB 2 * AC 1,
    AB 2 * AC 0 - AB 0 * AC 2,
    AB 0 * AC 1 - AB 1 * AC 0
  ]
  let dot_product := (n 0 * AD 0) + (n 1 * AD 1) + (n 2 * AD 2)
  let n_magnitude := Real.sqrt ((n 0)^2 + (n 1)^2 + (n 2)^2)
  abs (dot_product / n_magnitude) = 119 / Real.sqrt 277 := by
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_distance_to_plane_l1197_119752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inv_undefined_at_two_f_and_f_inv_are_inverse_l1197_119790

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (2 * x - 5) / (x - 6)

-- Define the inverse function f_inv
noncomputable def f_inv (x : ℝ) : ℝ := (6 * x - 5) / (x - 2)

-- Theorem stating that f_inv is undefined when x = 2
theorem f_inv_undefined_at_two :
  ¬ ∃ (y : ℝ), f_inv 2 = y := by
  sorry

-- Theorem stating that f and f_inv are inverse functions
theorem f_and_f_inv_are_inverse (x : ℝ) (h : x ≠ 6) (h' : x ≠ 2) :
  f (f_inv x) = x ∧ f_inv (f x) = x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inv_undefined_at_two_f_and_f_inv_are_inverse_l1197_119790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_30_l1197_119735

/-- The number of hours on a clock face -/
def clock_hours : ℕ := 12

/-- The angle between each hour marking on a clock face in degrees -/
noncomputable def hour_angle : ℝ := 360 / clock_hours

/-- The position of the hour hand at 3:30 in degrees -/
noncomputable def hour_hand_position : ℝ := 3 * hour_angle + hour_angle / 2

/-- The position of the minute hand at 3:30 in degrees -/
def minute_hand_position : ℝ := 180

/-- The acute angle between the clock hands at 3:30 -/
noncomputable def clock_angle : ℝ := min (abs (hour_hand_position - minute_hand_position)) (360 - abs (hour_hand_position - minute_hand_position))

/-- Theorem: The acute angle formed by the hands of a clock at 3:30 is 75 degrees -/
theorem clock_angle_at_3_30 : clock_angle = 75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_30_l1197_119735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l1197_119742

/-- A line passing through point A(0,1) with slope k -/
def line (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = k * p.1 + 1}

/-- Circle C: (x-2)^2 + (y-3)^2 = 1 -/
def circle_C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 - 3)^2 = 1}

/-- The origin of the coordinate system -/
def origin : ℝ × ℝ := (0, 0)

/-- The dot product of two vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

/-- The distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem line_circle_intersection (k : ℝ) :
  ∃ (M N : ℝ × ℝ),
    M ∈ line k ∧ N ∈ line k ∧
    M ∈ circle_C ∧ N ∈ circle_C ∧
    M ≠ N ∧
    dot_product (M.1 - origin.1, M.2 - origin.2) (N.1 - origin.1, N.2 - origin.2) = 12 →
    distance M N = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l1197_119742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_properties_l1197_119778

def Z (m : ℝ) : ℂ := Complex.mk (m^2 + 5*m + 6) (m^2 - 2*m - 15)

theorem complex_number_properties (m : ℝ) :
  ((Z m).im = 0 ↔ m = -3 ∨ m = 5) ∧
  (((Z m).re > 0 ∧ (Z m).im > 0) ↔ m < -3 ∨ m > 5) ∧
  (((Z m).re + (Z m).im + 5 = 0) ↔ m = 1 ∨ m = -5/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_properties_l1197_119778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l1197_119749

/-- A geometric sequence {a_n} where a_2 = √2 and a_3 = ³√3 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  a 2 = Real.sqrt 2 ∧ a 3 = Real.rpow 3 (1/3) ∧
  ∃ (q : ℝ), ∀ (n : ℕ), a (n + 1) = q * a n

/-- The ratio of (a_1 + a_2011) to (a_7 + a_2017) is 8/9 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (h : geometric_sequence a) :
  (a 1 + a 2011) / (a 7 + a 2017) = 8 / 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l1197_119749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_range_l1197_119708

-- Define the line equation
def line_equation (x y : ℝ) (α : ℝ) : Prop :=
  x * Real.sin α + y + 2 = 0

-- Define the inclination angle
noncomputable def inclination_angle (α : ℝ) : ℝ :=
  Real.arctan (- Real.sin α)

-- State the theorem
theorem inclination_angle_range :
  ∀ α : ℝ, ∃ θ : ℝ, inclination_angle α = θ ∧
  (θ ∈ Set.Icc 0 (Real.pi / 4) ∨ θ ∈ Set.Ico (3 * Real.pi / 4) Real.pi) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_range_l1197_119708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magical_pair_minimum_value_l1197_119729

/-- A pair of real numbers (x, y) is a "magical number pair" if 1/x + 1/y = 1 -/
def IsMagicalPair (x y : ℝ) : Prop := 1 / x + 1 / y = 1

theorem magical_pair_minimum_value (m n a b c : ℝ) 
  (h1 : IsMagicalPair m n) 
  (h2 : a = b + m) 
  (h3 : b = c + n) : 
  (∀ x : ℝ, (a - c)^2 - 12 * (a - b) * (b - c) ≥ -36) ∧ 
  (∃ y : ℝ, (a - c)^2 - 12 * (a - b) * (b - c) = -36) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_magical_pair_minimum_value_l1197_119729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_point_p_l1197_119761

/-- The trajectory of a point P given specific conditions -/
theorem trajectory_of_point_p 
  (A B P : ℝ × ℝ) 
  (O : ℝ × ℝ := (0, 0))
  (h1 : ‖A - B‖ = 3)
  (h2 : A.2 = 0)
  (h3 : B.1 = 0)
  (h4 : P = (1/3 : ℝ) • A + (2/3 : ℝ) • B) :
  P.1^2 + P.2^2/4 = 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_point_p_l1197_119761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_X_l1197_119709

/-- A random variable representing the outcome of two dice rolls -/
def X : ℝ → ℝ := sorry

/-- The probability that X equals 1 (same outcomes on both rolls) -/
noncomputable def p_same : ℝ := 1 / 6

/-- The probability that X equals 0 (different outcomes on both rolls) -/
noncomputable def p_diff : ℝ := 1 - p_same

/-- The expected value of X -/
noncomputable def E_X : ℝ := 1 * p_same + 0 * p_diff

/-- The variance of X -/
noncomputable def Var_X : ℝ := p_same * (1 - E_X)^2 + p_diff * (0 - E_X)^2

theorem variance_of_X : Var_X = 5 / 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_X_l1197_119709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spiral_2006_l1197_119723

/-- Represents a point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℤ
  y : ℤ

/-- The spiral numbering function that assigns a number to each point -/
def spiral_number : Point → ℕ := sorry

/-- The inverse function that gives the point for a given number -/
def spiral_point : ℕ → Point := sorry

/-- The spiral numbering is bijective -/
axiom spiral_bijective : Function.Bijective spiral_number

/-- The spiral starts at (0,0) with number 1 -/
axiom spiral_start : spiral_number ⟨0, 0⟩ = 1

/-- The spiral follows the specified pattern -/
axiom spiral_pattern : 
  spiral_number ⟨1, 0⟩ = 2 ∧
  spiral_number ⟨1, 1⟩ = 3 ∧
  spiral_number ⟨0, 1⟩ = 4 ∧
  spiral_number ⟨0, 2⟩ = 5 ∧
  spiral_number ⟨1, 2⟩ = 6 ∧
  spiral_number ⟨2, 2⟩ = 7 ∧
  spiral_number ⟨2, 1⟩ = 8 ∧
  spiral_number ⟨2, 0⟩ = 9

/-- The 2006th point in the spiral numbering has coordinates (44, 19) -/
theorem spiral_2006 : spiral_point 2006 = ⟨44, 19⟩ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spiral_2006_l1197_119723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_volume_equals_surface_area_l1197_119706

/-- Represents a right circular cylinder with integer dimensions -/
structure Cylinder where
  r : ℕ
  h : ℕ
  r_positive : r ≥ 1

/-- The volume of a cylinder -/
noncomputable def volume (c : Cylinder) : ℝ := Real.pi * c.r^2 * c.h

/-- The total surface area of a cylinder -/
noncomputable def surfaceArea (c : Cylinder) : ℝ := 2 * Real.pi * c.r * (c.h + c.r)

/-- Theorem stating that for a cylinder with integer dimensions, 
    the volume equals the surface area if and only if the radius is 3, 4, or 6 -/
theorem cylinder_volume_equals_surface_area (c : Cylinder) :
  volume c = surfaceArea c ↔ c.r = 3 ∨ c.r = 4 ∨ c.r = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_volume_equals_surface_area_l1197_119706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_points_l1197_119799

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- The polar equation of a line passing through two points -/
def polarLineEquation (A B : PolarPoint) : ℝ → ℝ → Prop :=
  fun ρ θ ↦ ρ * Real.sin (θ + Real.pi/6) = 2

theorem line_through_points (A B : PolarPoint)
  (hA : A = ⟨4, 2*Real.pi/3⟩)
  (hB : B = ⟨2, Real.pi/3⟩) :
  polarLineEquation A B = fun ρ θ ↦ ρ * Real.sin (θ + Real.pi/6) = 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_points_l1197_119799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_player_winning_strategy_l1197_119785

/-- Represents the state of the game -/
structure GameState where
  objects : List Nat
  deriving Repr

/-- Checks if the game is over (all objects removed) -/
def isGameOver (state : GameState) : Bool :=
  state.objects.sum = 0

/-- Represents a move in the game -/
structure Move where
  objectType : Nat
  count : Nat
  deriving Repr

/-- Applies a move to the game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  { objects := state.objects.mapIdx (fun i x => if i = move.objectType ∧ i < state.objects.length then x - move.count else x) }

/-- Checks if a move is valid -/
def isValidMove (state : GameState) (move : Move) : Bool :=
  move.objectType < state.objects.length ∧ move.count ≤ state.objects[move.objectType]!

/-- Checks if all object counts are even -/
def allCountsEven (state : GameState) : Bool :=
  state.objects.all (fun x => x % 2 == 0)

/-- Theorem: The second player has a winning strategy if they can always make all counts even -/
theorem second_player_winning_strategy 
  (initialState : GameState) 
  (h_initial_odd : ¬ allCountsEven initialState) :
  ∃ (strategy : GameState → Move),
    ∀ (state : GameState),
      ¬ isGameOver state →
      ¬ allCountsEven state →
      let nextState := applyMove state (strategy state)
      isValidMove state (strategy state) ∧
      allCountsEven nextState ∧
      (isGameOver nextState → True) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_player_winning_strategy_l1197_119785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_weight_of_class_l1197_119719

theorem average_weight_of_class (group1_count group2_count : ℕ) (group1_avg group2_avg : ℝ) :
  group1_count = 20 →
  group2_count = 8 →
  group1_avg = 50.25 →
  group2_avg = 45.15 →
  let total_count : ℕ := group1_count + group2_count
  let total_weight : ℝ := group1_count * group1_avg + group2_count * group2_avg
  let class_avg : ℝ := total_weight / total_count
  abs (class_avg - 48.79) < 0.01 := by
  sorry

#check average_weight_of_class

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_weight_of_class_l1197_119719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1197_119756

-- Define the function f(x)
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := Real.exp x + a * x + b

-- State the theorem
theorem function_properties (a b : ℝ) :
  (∀ x, deriv (f a b) x = Real.exp x + a) →
  f a b 0 = 4 →
  deriv (f a b) 0 = -2 →
  a = -3 ∧ b = 3 ∧
  (∀ x, x < Real.log 3 → deriv (f (-3) 3) x < 0) ∧
  (∀ x, x > Real.log 3 → deriv (f (-3) 3) x > 0) ∧
  (f (-3) 3 (Real.log 3) = 6 - 6 * Real.log 3) ∧
  (∀ x, f (-3) 3 x ≥ f (-3) 3 (Real.log 3)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1197_119756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_monthly_wage_l1197_119782

noncomputable def monthly_wage_per_employee (
  num_employees : ℕ)
  (monthly_revenue : ℝ)
  (tax_rate : ℝ)
  (marketing_rate : ℝ)
  (operational_rate : ℝ)
  (wage_rate : ℝ) : ℝ :=
  let after_tax := monthly_revenue * (1 - tax_rate)
  let after_marketing := after_tax * (1 - marketing_rate)
  let after_operational := after_marketing * (1 - operational_rate)
  let total_wages := after_operational * wage_rate
  total_wages / (num_employees : ℝ)

theorem correct_monthly_wage :
  monthly_wage_per_employee 10 400000 0.1 0.05 0.2 0.15 = 4104 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_monthly_wage_l1197_119782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_b_l1197_119781

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2^x - a) / (2^x + 1)
noncomputable def g (b : ℝ) (x : ℝ) : ℝ := Real.log (x^2 - b)

-- State the theorem
theorem range_of_b (b : ℝ) :
  (∀ x : ℝ, f 0 x = -(f 0 (-x))) →  -- f is an odd function
  (∀ x₁ x₂ : ℝ, f 0 x₁ ≤ g b x₂) →  -- f(x₁) ≤ g(x₂) for all x₁, x₂
  b ∈ Set.Iic (-Real.exp 1) :=  -- b ≤ -e
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_b_l1197_119781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_2_15_l1197_119707

/-- Calculates the angle between clock hands given hour and minute -/
noncomputable def clockAngle (hour : ℝ) (minute : ℝ) : ℝ :=
  |60 * hour - 11 * minute| / 2

theorem clock_angle_at_2_15 :
  clockAngle 2 15 = 22.5 := by
  -- Unfold the definition of clockAngle
  unfold clockAngle
  -- Simplify the expression
  simp [abs_of_nonneg]
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_2_15_l1197_119707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_simplification_and_evaluation_l1197_119763

open Real

theorem trigonometric_simplification_and_evaluation :
  (∀ α : ℝ, (sin (α + 3/2 * π) * sin (-α + π) * cos (α + π/2)) / 
            (cos (-α - π) * cos (α - π/2) * tan (α + π)) = -cos α) ∧
  (tan (675 * π/180) + sin (-330 * π/180) + cos (960 * π/180) = -1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_simplification_and_evaluation_l1197_119763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_value_at_pi_over_4_l1197_119777

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 2 / 2) * Real.sin (2 * x) + (Real.sqrt 6 / 2) * Real.cos (2 * x)

noncomputable def g (x : ℝ) : ℝ := f (x - Real.pi / 4)

theorem g_value_at_pi_over_4 : g (Real.pi / 4) = Real.sqrt 6 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_value_at_pi_over_4_l1197_119777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_year_rate_is_five_percent_l1197_119722

/-- Calculates the interest rate for the second year given the initial amount,
    first year interest rate, and final amount after two years. -/
noncomputable def second_year_interest_rate (initial_amount : ℝ) (first_year_rate : ℝ) 
                               (final_amount : ℝ) : ℝ :=
  let first_year_amount := initial_amount * (1 + first_year_rate)
  let second_year_interest := final_amount - first_year_amount
  second_year_interest / first_year_amount

/-- Theorem stating that given the problem conditions, 
    the second year interest rate is 5%. -/
theorem second_year_rate_is_five_percent :
  let initial_amount : ℝ := 5000
  let first_year_rate : ℝ := 0.04
  let final_amount : ℝ := 5460
  second_year_interest_rate initial_amount first_year_rate final_amount = 0.05 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval second_year_interest_rate 5000 0.04 5460

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_year_rate_is_five_percent_l1197_119722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_velocity_at_three_seconds_l1197_119717

-- Define the displacement function
noncomputable def displacement (t : ℝ) : ℝ := t^(1/4)

-- Define the velocity function as the derivative of displacement
noncomputable def velocity (t : ℝ) : ℝ := deriv displacement t

-- Theorem statement
theorem velocity_at_three_seconds :
  velocity 3 = 1 / (4 * (3^3)^(1/4)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_velocity_at_three_seconds_l1197_119717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_from_B_and_C_l1197_119758

/-- The distance between two points in 3D space -/
noncomputable def distance (x₁ y₁ z₁ x₂ y₂ z₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2 + (z₂ - z₁)^2)

/-- A is equidistant from B and C -/
theorem equidistant_from_B_and_C : 
  let A : ℝ × ℝ × ℝ := (0, 8, 0)
  let B : ℝ × ℝ × ℝ := (0, 5, -9)
  let C : ℝ × ℝ × ℝ := (-1, 0, 5)
  distance A.1 A.2.1 A.2.2 B.1 B.2.1 B.2.2 = distance A.1 A.2.1 A.2.2 C.1 C.2.1 C.2.2 :=
by
  -- Unfold the definitions
  unfold distance
  -- Simplify the expressions
  simp
  -- The proof is completed using sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_from_B_and_C_l1197_119758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_distribution_l1197_119764

/-- Given a jar of 550 marbles and the condition that if 5 more people join, 
    each person would receive 3 marbles less, prove that the number of people 
    in the group today is 28. -/
theorem marble_distribution (total_marbles additional_people marble_reduction : ℕ) 
    (h1 : total_marbles = 550)
    (h2 : additional_people = 5)
    (h3 : marble_reduction = 3) :
    ∃ (current_people : ℕ), 
      (total_marbles = current_people * (total_marbles / current_people)) ∧
      (total_marbles = (current_people + additional_people) * 
        (total_marbles / current_people - marble_reduction)) ∧
      current_people = 28 := by
  sorry

#check marble_distribution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_distribution_l1197_119764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_noncongruent_triangles_l1197_119733

/-- A triangle with integer side lengths -/
structure IntTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- Check if two triangles are congruent -/
def congruent (t1 t2 : IntTriangle) : Prop :=
  (t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c) ∨
  (t1.a = t2.b ∧ t1.b = t2.c ∧ t1.c = t2.a) ∨
  (t1.a = t2.c ∧ t1.b = t2.a ∧ t1.c = t2.b)

/-- The set of all non-congruent triangles with perimeter 10 -/
def nonCongruentTriangles : Set IntTriangle :=
  { t : IntTriangle | t.a + t.b + t.c = 10 ∧
    ∀ t' : IntTriangle, t'.a + t'.b + t'.c = 10 →
      congruent t t' → t = t' }

/-- Finite type instance for nonCongruentTriangles -/
instance : Fintype nonCongruentTriangles :=
  sorry

theorem count_noncongruent_triangles :
  Fintype.card nonCongruentTriangles = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_noncongruent_triangles_l1197_119733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_domain_l1197_119731

-- Define the function f(x) = 3^x
noncomputable def f (x : ℝ) : ℝ := 3^x

-- Define the domain of f
def domain_f : Set ℝ := {x | 0 < x ∧ x ≤ 2}

-- State the theorem about the domain of the inverse function
theorem inverse_function_domain :
  {y | ∃ x ∈ domain_f, f x = y} = Set.Ioc 1 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_domain_l1197_119731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bacteria_growth_time_l1197_119728

/-- The time (in hours) for bacteria to grow from initial to final population -/
noncomputable def growth_time (initial : ℝ) (final : ℝ) (rate : ℝ) : ℝ :=
  Real.log (final / initial) / Real.log (1 + rate)

/-- Theorem stating the growth time for the given bacterial population -/
theorem bacteria_growth_time :
  let initial := (600 : ℝ)
  let final := (8917 : ℝ)
  let rate := (1.5 : ℝ)
  let t := growth_time initial final rate
  ∃ ε > 0, |t - 2.945| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bacteria_growth_time_l1197_119728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_expression_squared_l1197_119714

theorem log_expression_squared (y : ℝ) (h : Real.log 3 = y) :
  (Real.log (10 * Real.log 1000))^2 = y^2 + 2*y + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_expression_squared_l1197_119714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_field_trip_calculations_l1197_119774

theorem field_trip_calculations (buses : Nat) (initial_supervisors : List Nat)
  (stops : Nat) (supervisors_per_stop : Nat) (student_supervisor_ratio : Nat)
  (initial_students : Nat) :
  buses = 5 →
  initial_supervisors = [4, 5, 3, 6, 7] →
  stops = 3 →
  supervisors_per_stop = 2 →
  student_supervisor_ratio = 10 →
  initial_students = 200 →
  let total_initial_supervisors := initial_supervisors.sum
  let total_final_supervisors := total_initial_supervisors + stops * supervisors_per_stop
  let max_students_per_stop := ((total_final_supervisors * student_supervisor_ratio - initial_students) / stops)
  let avg_supervisors_per_bus := (total_final_supervisors : Rat) / buses
  (max_students_per_stop = 36 ∧ avg_supervisors_per_bus = 31/5) := by
  sorry

#eval (31 : Rat) / 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_field_trip_calculations_l1197_119774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_point_with_correct_slope_l1197_119716

-- Define the slope of the reference line y = -4x
noncomputable def reference_slope : ℝ := -4

-- Define the point A
def point_A : ℝ × ℝ := (1, 3)

-- Define the slope of the desired line
noncomputable def desired_slope : ℝ := reference_slope / 3

-- Define the equation of the line
def line_equation (x y : ℝ) : Prop := 4 * x + 3 * y - 13 = 0

-- Theorem statement
theorem line_passes_through_point_with_correct_slope :
  line_equation point_A.1 point_A.2 ∧
  (∀ x₁ y₁ x₂ y₂ : ℝ, line_equation x₁ y₁ ∧ line_equation x₂ y₂ ∧ x₁ ≠ x₂ →
    (y₂ - y₁) / (x₂ - x₁) = desired_slope) := by
  sorry

#check line_passes_through_point_with_correct_slope

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_point_with_correct_slope_l1197_119716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l1197_119794

def A : Set ℝ := {x : ℝ | x ≤ 4}
def B : Set ℝ := {1, 3, 5, 7}

theorem intersection_of_A_and_B : A ∩ B = {1, 3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l1197_119794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_8pi_3_l1197_119750

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.tan x - 1

theorem f_value_at_8pi_3 :
  (∀ x, f (-x) = f x) →  -- f is even
  (∀ x, f (x + π) = f x) →  -- f has period π
  (∀ x ∈ Set.Icc 0 (π/2), f x = Real.sqrt 3 * Real.tan x - 1) →  -- definition for x ∈ [0, π/2)
  f (8*π/3) = 2 := by
  sorry

#check f_value_at_8pi_3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_8pi_3_l1197_119750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base9_addition_l1197_119767

/-- Converts a list of digits in base 9 to a natural number -/
def toNat9 (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => acc * 9 + d) 0

/-- Converts a natural number to a list of digits in base 9 -/
def toBase9 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 9) ((m % 9) :: acc)
  aux n []

theorem base9_addition :
  toNat9 [2, 7, 6] + toNat9 [8, 0, 3] + toNat9 [7, 2] = toNat9 [1, 2, 1, 6] :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base9_addition_l1197_119767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_grid_configuration_l1197_119769

/-- Represents a 5 × 8 grid filled with digits -/
def Grid := Fin 5 → Fin 8 → Fin 10

/-- Checks if a digit appears in exactly four rows or columns -/
def appears_four_times (g : Grid) (d : Fin 10) : Prop :=
  ((Finset.filter (λ i : Fin 5 ↦ ∃ j : Fin 8, g i j = d) (Finset.univ)).card = 4) ∨
  ((Finset.filter (λ j : Fin 8 ↦ ∃ i : Fin 5, g i j = d) (Finset.univ)).card = 4)

/-- The main theorem stating the impossibility of the grid configuration -/
theorem impossible_grid_configuration : ¬∃ (g : Grid), ∀ d : Fin 10, appears_four_times g d := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_grid_configuration_l1197_119769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_q_three_point_five_nearest_integer_l1197_119713

-- Define the function q
def q (x : ℝ) : ℝ := 
  |x - 3|^(1/3) + 2*|x - 3|^(1/5) + |x - 3|^(1/7)

-- Theorem statement
theorem q_three_point_five_nearest_integer : 
  Int.floor (q 3.5 + 0.5) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_q_three_point_five_nearest_integer_l1197_119713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1197_119727

noncomputable def f (x : ℝ) : ℝ := (x + 1) / (x^3 + 3*x^2 - 4*x)

theorem domain_of_f :
  {x : ℝ | ¬(x = -4 ∨ x = 0 ∨ x = 1)} = {x : ℝ | x ≠ -4 ∧ x ≠ 0 ∧ x ≠ 1} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1197_119727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_fv_sum_l1197_119751

/-- A parabola with vertex V, focus F, and a point A on the parabola. -/
structure Parabola where
  V : ℝ × ℝ  -- vertex
  F : ℝ × ℝ  -- focus
  A : ℝ × ℝ  -- point on the parabola

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Theorem stating the sum of possible FV values -/
theorem parabola_fv_sum (p : Parabola) 
  (h1 : distance p.A p.F = 20)
  (h2 : distance p.A p.V = 21) : 
  ∃ (fv1 fv2 : ℝ), fv1 + fv2 = 40/3 ∧ 
    (distance p.F p.V = fv1 ∨ distance p.F p.V = fv2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_fv_sum_l1197_119751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1197_119759

-- Define the sequence a_n
def a (n : ℕ+) : ℚ :=
  n / (n + 1)

-- Define the sequence T_n
def T (n : ℕ+) : ℚ :=
  if n = 1 then 1 else 1 / n

-- Define the sequence S_n
noncomputable def S (n : ℕ+) : ℚ :=
  (Finset.range n.val).sum (λ i => T ⟨i + 1, Nat.succ_pos i⟩)

theorem sequence_properties :
  (∀ n : ℕ+, (Finset.range n.val).prod (λ i => a ⟨i + 1, Nat.succ_pos i⟩) = 1 - a n) ∧
  (∀ n : ℕ+, (1 : ℚ) / (1 - a n) = n + 1) ∧
  (∀ n : ℕ+, 1/2 ≤ S (2 * n) - S n ∧ S (2 * n) - S n < 3/4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1197_119759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_lines_range_l1197_119780

/-- The distance between two points in a 2D plane -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

/-- The condition for exactly four lines satisfying the distance requirements -/
def four_lines_condition (d : ℝ) : Prop :=
  1 < 1 + d ∧ 1 + d < distance 1 2 5 5

/-- The theorem stating the range of d for which there are exactly four lines
    satisfying the distance requirements -/
theorem four_lines_range :
  ∀ d : ℝ, four_lines_condition d ↔ 0 < d ∧ d < 4 := by
  sorry

#check four_lines_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_lines_range_l1197_119780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_11_l1197_119753

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

noncomputable def sum_of_arithmetic_sequence (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (a 1 + a n) / 2

theorem arithmetic_sequence_sum_11 (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (∀ n, n > 0 → a n > 0) →
  a 5 + a 7 = (a 6) ^ 2 →
  sum_of_arithmetic_sequence a 11 = 22 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_11_l1197_119753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_lcm_gcd_ratio_l1197_119775

theorem smallest_integer_lcm_gcd_ratio (a b : ℕ) : 
  a = 60 →
  b % 5 = 0 →
  Nat.lcm a b / Nat.gcd a b = 24 →
  (∀ c : ℕ, c < b → c % 5 = 0 → Nat.lcm a c / Nat.gcd a c ≠ 24) →
  b = 160 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_lcm_gcd_ratio_l1197_119775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_g_98_99_l1197_119757

def g (x : ℤ) : ℤ := 2 * x^2 - x + 2006

theorem gcd_g_98_99 : Int.gcd (g 98) (g 99) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_g_98_99_l1197_119757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_n_positive_sum_l1197_119710

noncomputable def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

noncomputable def sum_n_terms (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := n / 2 * (2 * a₁ + (n - 1 : ℝ) * d)

theorem largest_n_positive_sum 
  (a₁ : ℝ) 
  (d : ℝ) 
  (h₁ : a₁ > 0)
  (h₂ : arithmetic_sequence a₁ d 2013 + arithmetic_sequence a₁ d 2014 > 0)
  (h₃ : arithmetic_sequence a₁ d 2013 * arithmetic_sequence a₁ d 2014 < 0) :
  (∀ n : ℕ, n ≤ 4026 → sum_n_terms a₁ d n > 0) ∧
  (∀ n : ℕ, n > 4026 → sum_n_terms a₁ d n ≤ 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_n_positive_sum_l1197_119710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_7_equals_64_l1197_119748

/-- A sequence {aₙ} is defined by a₁ = 1 and aₙ₊₁ = 2aₙ for n ∈ ℕ -/
def a : ℕ → ℕ
  | 0 => 1
  | n + 1 => 2 * a n

/-- The 7th term of the sequence {aₙ} is equal to 64 -/
theorem a_7_equals_64 : a 7 = 64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_7_equals_64_l1197_119748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unbounded_zero_set_of_polynomial_l1197_119791

/-- A two-variable polynomial with real coefficients -/
def TwoVarPolynomial := MvPolynomial (Fin 2) ℝ

/-- The sum of monomials with the highest degree in a polynomial -/
noncomputable def highest_degree_terms (P : TwoVarPolynomial) : TwoVarPolynomial :=
  sorry

/-- A set is bounded if there exists a positive number M such that 
    the distance of all its elements from the origin is less than M -/
def is_bounded (S : Set (ℝ × ℝ)) : Prop :=
  ∃ M : ℝ, M > 0 ∧ ∀ p ∈ S, Real.sqrt (p.1^2 + p.2^2) < M

/-- Evaluation of a two-variable polynomial -/
noncomputable def eval_TwoVarPolynomial (P : TwoVarPolynomial) (x y : ℝ) : ℝ :=
  (MvPolynomial.eval (![x, y]) P)

theorem unbounded_zero_set_of_polynomial 
  (P : TwoVarPolynomial) 
  (x₁ y₁ x₂ y₂ : ℝ) 
  (h₁ : eval_TwoVarPolynomial (highest_degree_terms P) x₁ y₁ > 0) 
  (h₂ : eval_TwoVarPolynomial (highest_degree_terms P) x₂ y₂ < 0) : 
  ¬ is_bounded {p : ℝ × ℝ | eval_TwoVarPolynomial P p.1 p.2 = 0} :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unbounded_zero_set_of_polynomial_l1197_119791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matt_protein_percentage_l1197_119793

/-- The percentage of protein in Matt's protein powder -/
noncomputable def protein_percentage (
  weight : ℝ
  ) (protein_per_kg : ℝ
  ) (powder_per_week : ℝ
  ) : ℝ :=
  let daily_protein := weight * protein_per_kg
  let weekly_protein := daily_protein * 7
  (weekly_protein / powder_per_week) * 100

/-- Theorem stating that the percentage of protein in Matt's protein powder is 80% -/
theorem matt_protein_percentage :
  protein_percentage 80 2 1400 = 80 := by
  -- Unfold the definition of protein_percentage
  unfold protein_percentage
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matt_protein_percentage_l1197_119793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_average_cost_at_40_l1197_119789

/-- Cost function (in 10,000 yuan) for processing x tons of carbon dioxide -/
noncomputable def cost (x : ℝ) : ℝ :=
  if x ≥ 1 ∧ x < 30 then
    (1/25) * x^3 - 640
  else if x ≥ 30 ∧ x ≤ 50 then
    x^2 - 10*x + 1600
  else
    0

/-- Earnings (in 10,000 yuan) for processing x tons of carbon dioxide -/
def earnings (x : ℝ) : ℝ := x

/-- Average cost per ton (in 10,000 yuan) for processing x tons of carbon dioxide -/
noncomputable def averageCost (x : ℝ) : ℝ := (cost x - earnings x) / x

/-- Theorem: The minimum average cost per ton occurs at x = 40 -/
theorem min_average_cost_at_40 :
  ∀ x, x ∈ Set.Icc 1 50 → averageCost 40 ≤ averageCost x :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_average_cost_at_40_l1197_119789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_local_odd_function_range_l1197_119734

/-- A function f is a local odd function if there exists an x in its domain such that f(-x) = -f(x) -/
def IsLocalOddFunction (f : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, f (-x) = -f x

/-- The function f(x) = 9^x - m * 3^x - 3 -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  (9 : ℝ)^x - m * (3 : ℝ)^x - 3

/-- The range of m for which f is a local odd function -/
def MRange : Set ℝ :=
  {m : ℝ | IsLocalOddFunction (f m)}

theorem local_odd_function_range :
  MRange = { m : ℝ | m ≥ -2 } := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_local_odd_function_range_l1197_119734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_amount_calculation_l1197_119798

theorem total_amount_calculation (share1 share2 share3 total : ℚ) : 
  share1 + share2 + share3 = total →
  share1 = 20 * (share2 / 3) →
  share3 = 2 * (share2 / 3) →
  share2 = 36 →
  total = 300 := by
  intros h1 h2 h3 h4
  have h5 : share2 / 3 = 12 := by
    rw [h4]
    norm_num
  have h6 : share1 = 240 := by
    rw [h2, h5]
    norm_num
  have h7 : share3 = 24 := by
    rw [h3, h5]
    norm_num
  calc
    total = share1 + share2 + share3 := by rw [h1]
    _     = 240 + 36 + 24 := by rw [h6, h4, h7]
    _     = 300 := by norm_num

#check total_amount_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_amount_calculation_l1197_119798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_at_sqrt_e_decreasing_condition_l1197_119725

-- Define the function f(x) with parameter a
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a * Real.log x

-- Theorem for part (I)
theorem min_value_at_sqrt_e :
  let a := -2 * Real.exp 1
  ∃ (x : ℝ), x > 0 ∧ 
    (∀ (y : ℝ), y > 0 → f a x ≤ f a y) ∧
    f a x = 0 ∧
    x = Real.sqrt (Real.exp 1) := by sorry

-- Theorem for part (II)
theorem decreasing_condition :
  ∀ (a : ℝ), (∀ (x : ℝ), x ∈ Set.Icc 1 4 → 
    (∀ (y : ℝ), y ∈ Set.Icc 1 4 → x < y → f a x > f a y)) ↔
  a ∈ Set.Iic (-32) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_at_sqrt_e_decreasing_condition_l1197_119725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_distance_theorem_l1197_119703

/-- A line in 2D space represented by the equation Ax + By + C = 0 -/
structure Line where
  A : ℝ
  B : ℝ
  C : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Point.on_line (p : Point) (l : Line) : Prop :=
  l.A * p.x + l.B * p.y + l.C = 0

/-- Calculate the distance between two parallel lines -/
noncomputable def distance_between_parallel_lines (l1 l2 : Line) : ℝ :=
  abs (l2.C - l1.C) / Real.sqrt (l1.A^2 + l1.B^2)

theorem parallel_lines_distance_theorem :
  ∀ (l : Line) (a : Line),
    -- Line l intercepts y-axis at 4
    Point.on_line ⟨0, 4⟩ l →
    -- Line l passes through (-3, 8)
    Point.on_line ⟨-3, 8⟩ l →
    -- Line a passes through (-2, 5)
    Point.on_line ⟨-2, 5⟩ a →
    -- Lines l and a are parallel
    l.A / l.B = a.A / a.B →
    -- The distance between l and a is 1
    distance_between_parallel_lines l a = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_distance_theorem_l1197_119703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_difference_l1197_119792

def sequenceA (n : ℕ) : ℚ :=
  match n with
  | 0 => 1 / 7
  | n + 1 => (7 / 2) * sequenceA n * (1 - sequenceA n)

theorem sequence_difference : sequenceA 998 - sequenceA 887 = 3 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_difference_l1197_119792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_committee_selection_with_treasurer_l1197_119765

def club_size : ℕ := 12
def committee_size : ℕ := 5
def treasurer_count : ℕ := 1

theorem committee_selection_with_treasurer :
  (Nat.choose club_size committee_size) - (Nat.choose (club_size - treasurer_count) committee_size) = 330 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_committee_selection_with_treasurer_l1197_119765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_increasing_f_derivative_is_derivative_l1197_119772

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (3 - x) * Real.exp x

-- Define the derivative of f(x)
noncomputable def f_derivative (x : ℝ) : ℝ := (2 - x) * Real.exp x

-- Theorem statement
theorem f_monotonic_increasing :
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → x₂ < 2 → f x₁ < f x₂ := by
  sorry

-- Theorem to show that f_derivative is indeed the derivative of f
theorem f_derivative_is_derivative :
  ∀ x : ℝ, deriv f x = f_derivative x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_increasing_f_derivative_is_derivative_l1197_119772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l1197_119773

/-- Represents the speed of the train in km/hr -/
noncomputable def train_speed : ℝ := 36

/-- Represents the length of the train in meters -/
noncomputable def train_length : ℝ := 100

/-- Converts km/hr to m/s -/
noncomputable def km_hr_to_m_s (speed : ℝ) : ℝ := speed * (1000 / 3600)

/-- Calculates the time (in seconds) for the train to cross an electric pole -/
noncomputable def crossing_time : ℝ := train_length / km_hr_to_m_s train_speed

theorem train_crossing_time :
  crossing_time = 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l1197_119773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1197_119795

theorem problem_solution (c d : ℝ) 
  (h1 : (5 : ℝ) ^ c = (625 : ℝ) ^ (d + 3))
  (h2 : (343 : ℝ) ^ d = (7 : ℝ) ^ (c - 4)) : 
  c * d = 160 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1197_119795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l1197_119718

/-- Parabola G with equation y^2 = 4x -/
def G : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- Point A -/
def A : ℝ × ℝ := (3, 0)

/-- Origin O -/
def O : ℝ × ℝ := (0, 0)

/-- Line with slope 1 -/
def line_slope_1 (m : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1 + m}

/-- Intersection points of the line with G -/
def intersection_points (m : ℝ) : Set (ℝ × ℝ) := G ∩ line_slope_1 m

/-- Area of triangle ABC -/
noncomputable def triangle_area (m : ℝ) : ℝ :=
  let points := intersection_points m
  let x1 := Real.sqrt (4 * (1 - m))
  let x2 := -Real.sqrt (4 * (1 - m))
  let B : ℝ × ℝ := (2 - m - x1, 2 - m - x1 + m)
  let C : ℝ × ℝ := (2 - m + x1, 2 - m + x1 + m)
  abs ((A.1 - B.1) * (C.2 - B.2) - (C.1 - B.1) * (A.2 - B.2)) / 2

/-- Theorem stating the maximum area of triangle ABC -/
theorem max_triangle_area :
  ∃ m : ℝ, m ∈ Set.Ioo (-3) 0 ∧
    (∀ n : ℝ, n ∈ Set.Ioo (-3) 0 → triangle_area n ≤ triangle_area m) ∧
    triangle_area m = 32 * Real.sqrt 3 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l1197_119718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_properties_l1197_119726

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 13

-- Define the points M and N
def point_M : ℝ × ℝ := (3, -3)
def point_N : ℝ × ℝ := (-2, 2)

-- Define the y-axis intercept length
noncomputable def y_intercept_length : ℝ := 4 * Real.sqrt 3

-- Define the line l
def line_l (m : ℝ) (x y : ℝ) : Prop := y = -x + m

-- Define the condition for A and B being on circle C
def A_B_on_C (x₁ x₂ m : ℝ) : Prop :=
  circle_C x₁ (m - x₁) ∧ circle_C x₂ (m - x₂)

-- Define the condition for AB being a diameter of a circle through the origin
def AB_diameter_through_origin (x₁ x₂ m : ℝ) : Prop :=
  (m - x₁) / x₁ * (m - x₂) / x₂ = -1

-- Main theorem
theorem circle_and_line_properties :
  (∀ x y, circle_C x y → ((x = 3 ∧ y = -3) ∨ (x = -2 ∧ y = 2))) →
  (∃ y₁ y₂, y₂ - y₁ = y_intercept_length ∧ circle_C 0 y₁ ∧ circle_C 0 y₂) →
  (∃ m, (∀ x₁ x₂, A_B_on_C x₁ x₂ m → AB_diameter_through_origin x₁ x₂ m) →
    m = 4 ∨ m = -3) := by
  sorry

#check circle_and_line_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_properties_l1197_119726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_bisects_circle_product_bound_l1197_119739

/-- The circle C defined by x^2 + y^2 + 2x - 4y + 1 = 0 -/
def circleC (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y + 1 = 0

/-- The line L defined by ax - by + 1 = 0 -/
def lineL (a b x y : ℝ) : Prop := a*x - b*y + 1 = 0

/-- The line L bisects the circumference of circle C -/
def bisects (a b : ℝ) : Prop := ∃ x y : ℝ, circleC x y ∧ lineL a b x y

theorem line_bisects_circle_product_bound :
  ∀ a b : ℝ, bisects a b → a*b ≤ 1/8 :=
by
  sorry

#check line_bisects_circle_product_bound

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_bisects_circle_product_bound_l1197_119739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_sqrt_at_two_l1197_119754

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt x

-- State the theorem
theorem inverse_of_sqrt_at_two :
  ∃ (f_inv : ℝ → ℝ), (∀ x, x ≥ 0 → f (f_inv x) = x) ∧ (f_inv 2 = 4) := by
  -- Define the inverse function
  let f_inv (x : ℝ) := x^2
  
  -- Prove the existence of f_inv
  use f_inv
  
  constructor
  
  -- Prove that f_inv is indeed the inverse of f for x ≥ 0
  · intro x hx
    simp [f, f_inv]
    rw [Real.sqrt_sq]
    exact hx
  
  -- Prove that f_inv(2) = 4
  · simp [f_inv]
    norm_num

-- The proof is complete, so we don't need 'sorry'

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_sqrt_at_two_l1197_119754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_pairs_count_l1197_119715

theorem polygon_pairs_count : 
  let valid_pair := λ (k r : ℕ) => 
    k ≥ 3 ∧ r ≥ 3 ∧ 3 * r - 5 * k = 4 ∧ (r : Int) - (k : Int) ≤ 8 ∧ (k : Int) - (r : Int) ≤ 8
  ∃! (pairs : List (ℕ × ℕ)), pairs.length = 3 ∧ ∀ (k r : ℕ), (k, r) ∈ pairs ↔ valid_pair k r := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_pairs_count_l1197_119715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_variance_change_l1197_119740

/-- Represents a set of numbers with their statistical properties -/
structure NumberSet where
  count : Nat
  sum : ℝ
  sumSquares : ℝ

/-- Calculates the average of a NumberSet -/
noncomputable def average (s : NumberSet) : ℝ :=
  s.sum / s.count

/-- Calculates the variance of a NumberSet -/
noncomputable def variance (s : NumberSet) : ℝ :=
  s.sumSquares / s.count - (average s) ^ 2

/-- Adds a new data point to a NumberSet -/
def addDataPoint (s : NumberSet) (x : ℝ) : NumberSet :=
  { count := s.count + 1
  , sum := s.sum + x
  , sumSquares := s.sumSquares + x^2 }

theorem average_variance_change 
  (s : NumberSet) 
  (h1 : s.count = 8) 
  (h2 : average s = 5) 
  (h3 : variance s = 2) : 
  let newSet := addDataPoint s (average s)
  average newSet = average s ∧ variance newSet < variance s := by
  sorry

#eval "Theorem statement compiled successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_variance_change_l1197_119740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_distinct_roots_l1197_119732

/-- The equation has exactly three distinct roots if and only if m = 2 -/
theorem three_distinct_roots (m : ℝ) : 
  (∃! (s : Finset ℝ), s.card = 3 ∧ ∀ x ∈ s, 
    (x^2 - 2*m*x - 4*(m^2 + 1))*(x^2 - 4*x - 2*m*(m^2 + 1)) = 0) ↔ 
  m = 2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_distinct_roots_l1197_119732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_aubrey_speed_l1197_119711

/-- Given a distance and time, calculate the average speed -/
noncomputable def average_speed (distance : ℝ) (time : ℝ) : ℝ :=
  distance / time

theorem aubrey_speed :
  let distance : ℝ := 88
  let time : ℝ := 4
  average_speed distance time = 22 := by
  -- Unfold the definition of average_speed
  unfold average_speed
  -- Simplify the expression
  simp
  -- Perform the division
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_aubrey_speed_l1197_119711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_pairs_l1197_119786

/-- ω is a nonreal root of z^4 = 1 -/
noncomputable def ω : ℂ := Complex.I

/-- The set of ordered pairs (a,b) of integers such that |aω + b| = 2 -/
def S : Set (ℤ × ℤ) :=
  {p : ℤ × ℤ | Complex.abs (p.1 * ω + p.2) = 2}

/-- There are exactly 4 ordered pairs (a,b) of integers such that |aω + b| = 2 -/
theorem four_pairs : Finset.card (Finset.filter (fun p => Complex.abs (p.1 * ω + p.2) = 2) 
  (Finset.product (Finset.range 5) (Finset.range 5))) = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_pairs_l1197_119786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jason_saturday_hours_l1197_119720

/-- Represents Jason's work schedule and earnings --/
structure WorkSchedule where
  afterSchoolRate : ℚ
  saturdayRate : ℚ
  totalHours : ℚ
  totalEarnings : ℚ

/-- Calculates the number of hours worked on Saturday given a work schedule --/
def saturdayHours (w : WorkSchedule) : ℚ :=
  (w.totalEarnings - w.afterSchoolRate * w.totalHours) / (w.saturdayRate - w.afterSchoolRate)

/-- Theorem stating that Jason worked 8 hours on Saturday given the problem conditions --/
theorem jason_saturday_hours :
  let w : WorkSchedule := {
    afterSchoolRate := 4,
    saturdayRate := 6,
    totalHours := 18,
    totalEarnings := 88
  }
  saturdayHours w = 8 := by
  -- Proof goes here
  sorry

#eval saturdayHours {
  afterSchoolRate := 4,
  saturdayRate := 6,
  totalHours := 18,
  totalEarnings := 88
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jason_saturday_hours_l1197_119720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clothing_removed_calculation_l1197_119702

/-- Represents the weights of items in a suitcase -/
structure SuitcaseWeights where
  books : ℚ
  clothes : ℚ
  electronics : ℚ

/-- Calculates the ratio of books to clothes -/
noncomputable def booksToClothesRatio (w : SuitcaseWeights) : ℚ :=
  w.books / w.clothes

theorem clothing_removed_calculation (initial final : SuitcaseWeights) :
  -- Initial ratio condition
  initial.books / initial.clothes / (initial.electronics / initial.clothes) = 5 / 4 / (2 / 4) →
  -- Electronics weight condition
  initial.electronics = 9 →
  -- Ratio doubling condition
  booksToClothesRatio final = 2 * booksToClothesRatio initial →
  -- Electronics weight remains the same
  final.electronics = initial.electronics →
  -- Books weight remains the same
  final.books = initial.books →
  -- Theorem statement: Amount of clothing removed
  initial.clothes - final.clothes = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_clothing_removed_calculation_l1197_119702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_has_four_digits_l1197_119730

/-- A digit is a natural number between 1 and 9 inclusive. -/
def Digit : Type := {n : ℕ // 1 ≤ n ∧ n ≤ 9}

/-- The sum of 6543, C75, and D6 for any nonzero digits C and D. -/
def sum_numbers (C D : Digit) : ℕ := 6543 + (C.val * 100 + 75) + (D.val * 10 + 6)

/-- The number of digits in a natural number. -/
def num_digits (n : ℕ) : ℕ :=
  if n = 0 then 1 else Nat.log n 10 + 1

/-- Theorem stating that the sum of the three numbers always has 4 digits. -/
theorem sum_has_four_digits (C D : Digit) : num_digits (sum_numbers C D) = 4 := by
  sorry

#eval num_digits (sum_numbers ⟨1, by norm_num⟩ ⟨1, by norm_num⟩)
#eval num_digits (sum_numbers ⟨9, by norm_num⟩ ⟨9, by norm_num⟩)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_has_four_digits_l1197_119730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_length_l1197_119796

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the focus and directrix
def focus : ℝ × ℝ := (2, 0)
def directrix : ℝ → ℝ := λ x => -2

-- Define a point on the directrix
noncomputable def P : ℝ × ℝ := sorry

-- Define Q as the intersection of PF and the parabola
noncomputable def Q : ℝ × ℝ := sorry

-- State the theorem
theorem parabola_intersection_length :
  parabola Q.1 Q.2 ∧  -- Q is on the parabola
  P.2 = directrix P.1 ∧  -- P is on the directrix
  (∃ t : ℝ, Q = focus + t • (P - focus)) ∧  -- Q is on line PF
  (P - focus) = 4 • (Q - focus) →  -- FP = 4FQ
  ‖Q - focus‖ = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_length_l1197_119796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_cardinality_bound_l1197_119747

theorem subset_cardinality_bound {α : Type*} [Fintype α] (m n : ℕ) (S : Finset α) (A : Fin m → Finset α) :
  m > 1 →
  n > 1 →
  Finset.card S = n →
  (∀ i, A i ⊆ S) →
  (∀ x y, x ∈ S → y ∈ S → x ≠ y → ∃ i : Fin m, (x ∈ A i ∧ y ∉ A i) ∨ (x ∉ A i ∧ y ∈ A i)) →
  n ≤ 2^m := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_cardinality_bound_l1197_119747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_condition_l1197_119783

theorem integer_condition (k n : ℕ) (h1 : 1 ≤ k) (h2 : k < n) :
  ∃ m : ℤ, (n - 3 * k - 2 : ℤ) * (Nat.choose n k) = (k + 2 : ℤ) * m ↔ 
  ∃ q : ℕ, n + 4 = (k + 2) * q :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_condition_l1197_119783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_translation_is_quadrangular_prism_l1197_119744

/-- A trapezoid is a quadrilateral with at least one pair of parallel sides. -/
structure Trapezoid where
  vertices : Fin 4 → ℝ × ℝ
  parallel_sides : ∃ (i j : Fin 4), i ≠ j ∧ (vertices i).1 = (vertices j).1

/-- A prism is a 3D geometric body with two congruent parallel faces (bases) 
    connected by parallelograms (lateral faces). -/
structure Prism where
  base : Set (ℝ × ℝ)
  height : ℝ

/-- A quadrilateral is a polygon with four vertices. -/
structure Quadrilateral where
  vertices : Fin 4 → ℝ × ℝ

/-- A quadrangular prism is a prism with a quadrilateral base. -/
def QuadrangularPrism (p : Prism) : Prop :=
  ∃ (q : Quadrilateral), Set.range q.vertices = p.base

/-- The geometric body formed by translating a trapezoid in a certain direction 
    is a quadrangular prism. -/
theorem trapezoid_translation_is_quadrangular_prism 
  (t : Trapezoid) (direction : ℝ × ℝ × ℝ) : 
  ∃ (p : Prism), QuadrangularPrism p :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_translation_is_quadrangular_prism_l1197_119744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_identify_max_min_in_13_comparisons_l1197_119760

theorem identify_max_min_in_13_comparisons 
  {α : Type} [LinearOrder α] (S : Finset α) (h_card : S.card = 10) 
  (h_distinct : ∀ x y, x ∈ S → y ∈ S → x ≠ y → x < y ∨ y < x) :
  ∃ (f : α → α → Bool), 
    (∀ x y, x ∈ S → y ∈ S → (f x y = true ↔ x < y)) ∧ 
    (∃ (max min : α), 
      max ∈ S ∧ min ∈ S ∧
      (∀ x, x ∈ S → x ≤ max) ∧ 
      (∀ x, x ∈ S → min ≤ x) ∧
      (Finset.sum (Finset.product S S) (λ (p : α × α) => if f p.1 p.2 ∨ f p.2 p.1 then 1 else 0) ≤ 13)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_identify_max_min_in_13_comparisons_l1197_119760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_inequality_implies_m_equals_three_l1197_119705

theorem complex_inequality_implies_m_equals_three (m : ℝ) :
  (m - 3 : ℝ) ≥ 0 ∧ (m^2 - 9 : ℝ) = 0 → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_inequality_implies_m_equals_three_l1197_119705
