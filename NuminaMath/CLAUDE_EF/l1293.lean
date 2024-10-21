import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_of_monotonic_and_root_l1293_129318

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 + x^2 - a*x

theorem f_range_of_monotonic_and_root (a : ℝ) :
  (∀ x y, 1 < x ∧ x < y → f a x < f a y) →
  (∃ r, 1 < r ∧ r < 2 ∧ f a r = 0) →
  4/3 < a ∧ a ≤ 3 :=
by
  sorry

#check f_range_of_monotonic_and_root

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_of_monotonic_and_root_l1293_129318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_product_l1293_129391

theorem parallel_vectors_product (x z : ℝ) : 
  let a : Fin 3 → ℝ := ![x, 4, 3]
  let b : Fin 3 → ℝ := ![3, 2, z]
  (∃ (k : ℝ), a = k • b) → x * z = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_product_l1293_129391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_amplitude_and_phase_shift_l1293_129302

noncomputable def f (x : ℝ) := 5 * Real.sin (3 * x - Real.pi / 3)

theorem amplitude_and_phase_shift :
  (∃ A : ℝ, ∀ x, |f x| ≤ A ∧ (∃ x₀, |f x₀| = A) ∧ A = 5) ∧
  (∃ φ : ℝ, ∀ x, f (x + φ) = 5 * Real.sin (3 * x) ∧ φ = Real.pi / 9) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_amplitude_and_phase_shift_l1293_129302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_term_count_l1293_129309

-- Define the factors
def factor1 : String := "x+y+z"
def factor2 : String := "u+v+w+x"

-- Define a function to count terms in a polynomial expression
def count_terms (expression : String) : Nat :=
  (expression.split (fun c => c = '+') ).length

-- Theorem statement
theorem expansion_term_count :
  count_terms factor1 * count_terms factor2 = 12 :=
by
  -- Evaluate count_terms for factor1 and factor2
  have h1 : count_terms factor1 = 3 := by rfl
  have h2 : count_terms factor2 = 4 := by rfl
  
  -- Multiply the results
  calc
    count_terms factor1 * count_terms factor2 = 3 * 4 := by rw [h1, h2]
    _ = 12 := by rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_term_count_l1293_129309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_workload_increase_l1293_129328

/-- Proves the increase in workload per remaining worker when 1/4 of workers are absent -/
theorem workload_increase
  (p : ℕ) -- Total number of workers
  (a b c d : ℝ) -- Workloads of tasks A, B, C, and D
  (hp : p > 0) -- Assumption that there is at least one worker
  : (a + b + c + d) / ((3 : ℝ) * p) =
    let total_workload := a + b + c + d
    let original_workload_per_worker := total_workload / p
    let remaining_workers := (3 : ℝ) * p / 4
    let new_workload_per_worker := total_workload / remaining_workers
    new_workload_per_worker - original_workload_per_worker :=
by
  sorry

#check workload_increase

end NUMINAMATH_CALUDE_ERRORFEEDBACK_workload_increase_l1293_129328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marketing_hours_calculation_l1293_129378

noncomputable def total_work_hours : ℝ := 8
noncomputable def customer_outreach_hours : ℝ := 4
noncomputable def advertisement_hours : ℝ := customer_outreach_hours / 2

theorem marketing_hours_calculation :
  total_work_hours - (customer_outreach_hours + advertisement_hours) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marketing_hours_calculation_l1293_129378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_movie_ticket_cost_l1293_129324

/-- The cost of movie tickets for a family --/
theorem movie_ticket_cost : 30 = (
  let children_ticket_cost : ℝ := 4.25
  let adult_ticket_cost : ℝ := children_ticket_cost + 3.25
  let num_adult_tickets : ℕ := 2
  let num_children_tickets : ℕ := 4
  let discount : ℝ := 2

  num_adult_tickets * adult_ticket_cost + 
  num_children_tickets * children_ticket_cost - 
  discount
) := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_movie_ticket_cost_l1293_129324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l1293_129319

/-- Given a circle C: x^2 + y^2 = 4 and a line l: y = kx + m, 
    if the minimum chord length when k changes is 2, 
    then m = ±√3 -/
theorem circle_line_intersection (k m : ℝ) : 
  (∀ x y, x^2 + y^2 = 4 → y = k*x + m) →
  (∃ min_chord : ℝ, min_chord = 2 ∧ 
    ∀ chord : ℝ, (∃ x₁ y₁ x₂ y₂, 
      x₁^2 + y₁^2 = 4 ∧ 
      x₂^2 + y₂^2 = 4 ∧ 
      y₁ = k*x₁ + m ∧ 
      y₂ = k*x₂ + m ∧ 
      chord^2 = (x₂ - x₁)^2 + (y₂ - y₁)^2) → 
    chord ≥ min_chord) →
  m = Real.sqrt 3 ∨ m = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l1293_129319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_ratio_1638_l1293_129396

/-- Represents a factorials ratio that equals 1638 -/
def FactorialsRatio (a₁ a₂ b₁ b₂ : ℕ) : Prop :=
  a₁ ≥ a₂ ∧ b₁ ≥ b₂ ∧ (a₁.factorial * a₂.factorial) / (b₁.factorial * b₂.factorial) = 1638

/-- Checks if a₁ + b₁ is minimal for all valid factorials ratios -/
def IsMinimalSum (a₁ b₁ : ℕ) : Prop :=
  ∀ a₁' b₁' a₂' b₂', FactorialsRatio a₁' a₂' b₁' b₂' → a₁ + b₁ ≤ a₁' + b₁'

/-- Checks if a₂ + b₂ is minimal given the minimality of a₁ + b₁ -/
def IsMinimalSecondSum (a₁ a₂ b₁ b₂ : ℕ) : Prop :=
  ∀ a₂' b₂', FactorialsRatio a₁ a₂' b₁ b₂' → a₂ + b₂ ≤ a₂' + b₂'

theorem factorial_ratio_1638 :
  ∃ a₁ a₂ b₁ b₂ : ℕ,
    FactorialsRatio a₁ a₂ b₁ b₂ ∧
    IsMinimalSum a₁ b₁ ∧
    IsMinimalSecondSum a₁ a₂ b₁ b₂ ∧
    Int.natAbs (a₁ - b₁) = 1 := by
  sorry

#check factorial_ratio_1638

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_ratio_1638_l1293_129396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1293_129314

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x + m / x

-- State the theorem
theorem f_properties (m : ℝ) :
  (f m 1 = 2) →
  (m = 1 ∧ ∀ x : ℝ, x ≠ 0 → f m (-x) = -(f m x)) :=
by
  intro h
  have m_eq : m = 1 := by
    -- Proof that m = 1
    sorry
  have odd_func : ∀ x : ℝ, x ≠ 0 → f m (-x) = -(f m x) := by
    -- Proof that f is an odd function
    sorry
  exact ⟨m_eq, odd_func⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1293_129314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_series_l1293_129306

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- Define the sum of the series
noncomputable def series_sum : ℂ := (i * (1 - i^2018)) / (1 - i)

-- Theorem statement
theorem sum_of_series : series_sum = -1 + i := by
  -- The proof is skipped for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_series_l1293_129306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1293_129394

noncomputable section

-- Define the set of positive real numbers
def PositiveReals := {x : ℝ | x > 0}

-- Define the function we're minimizing
def f (a b : ℝ) := 1/a + 4/b

-- Define the constraint function
def g (x : ℝ) := |2*x - 1| - |x + 1|

theorem problem_solution :
  (∃ (min : ℝ), min = 9 ∧
    ∀ (a b : ℝ), a ∈ PositiveReals → b ∈ PositiveReals → a + b = 1 →
      f a b ≥ min ∧ (f a b = min ↔ a = 1/3 ∧ b = 2/3)) ∧
  (∀ (x : ℝ), (∀ (a b : ℝ), a ∈ PositiveReals → b ∈ PositiveReals → f a b ≥ g x) →
    -7 ≤ x ∧ x ≤ 11) ∧
  (∀ (y : ℝ), -7 ≤ y ∧ y ≤ 11 →
    ∃ (a b : ℝ), a ∈ PositiveReals ∧ b ∈ PositiveReals ∧ f a b = g y) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1293_129394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rook_routes_catalan_l1293_129363

/-- The number of valid routes for a rook on an n × n chessboard -/
def validRoutes (n : ℕ) : ℚ :=
  (1 : ℚ) / n * (Nat.choose (2 * n - 2) (n - 1) : ℚ)

/-- Catalan number definition -/
def catalanNumber (n : ℕ) : ℚ :=
  (1 : ℚ) / (n + 1) * (Nat.choose (2 * n) n : ℚ)

theorem rook_routes_catalan (n : ℕ) (h : n > 0) :
  validRoutes n = catalanNumber (n - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rook_routes_catalan_l1293_129363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_centroid_incenter_relation_l1293_129356

-- Define a triangle ABC in 2D space
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the centroid of a triangle
noncomputable def centroid (t : Triangle) : ℝ × ℝ :=
  ((t.A.1 + t.B.1 + t.C.1) / 3, (t.A.2 + t.B.2 + t.C.2) / 3)

-- Define the incenter of a triangle
-- Note: This is a placeholder definition, as the actual calculation of the incenter is complex
noncomputable def incenter (t : Triangle) : ℝ × ℝ :=
  sorry

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem triangle_centroid_incenter_relation (t : Triangle) :
  let G := centroid t
  let I := incenter t
  (distance I t.A)^2 + (distance I t.B)^2 + (distance I t.C)^2 =
  3 * (distance I G)^2 + (distance G t.A)^2 + (distance G t.B)^2 + (distance G t.C)^2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_centroid_incenter_relation_l1293_129356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intercepts_sum_l1293_129348

/-- Given a line defined by the equation y + 3 = -3(x - 5), 
    the sum of its x-intercept and y-intercept is 16. -/
theorem line_intercepts_sum : ∃ (x_int y_int : ℝ),
  (0 + 3 = -3 * (x_int - 5)) ∧
  (y_int + 3 = -3 * (0 - 5)) ∧
  x_int + y_int = 16 := by
  sorry

#check line_intercepts_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intercepts_sum_l1293_129348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_theorem_l1293_129339

/-- An arithmetic progression with 12 terms -/
structure ArithmeticProgression :=
  (a : ℝ)  -- First term
  (d : ℝ)  -- Common difference

/-- The sum of the first n terms of an arithmetic progression -/
noncomputable def sum_n_terms (ap : ArithmeticProgression) (n : ℕ) : ℝ :=
  n * (2 * ap.a + (n - 1) * ap.d) / 2

/-- The sum of even-indexed terms in an arithmetic progression with 12 terms -/
noncomputable def sum_even_terms (ap : ArithmeticProgression) : ℝ :=
  6 * (ap.a + ap.d + ap.a + 11 * ap.d) / 2

/-- The sum of odd-indexed terms in an arithmetic progression with 12 terms -/
noncomputable def sum_odd_terms (ap : ArithmeticProgression) : ℝ :=
  6 * (ap.a + ap.a + 10 * ap.d) / 2

/-- The main theorem -/
theorem arithmetic_progression_theorem (ap : ArithmeticProgression) :
  sum_n_terms ap 12 = 354 ∧
  sum_even_terms ap / sum_odd_terms ap = 32 / 27 →
  ap.a = 2 ∧ ap.d = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_theorem_l1293_129339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_property_l1293_129386

noncomputable def arithmetic_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) - a n = a 1 - a 0

noncomputable def sum_of_arithmetic_sequence (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  n * (a 0 + a (n - 1)) / 2

theorem arithmetic_sequence_sum_property (a : ℕ → ℝ) (k : ℕ) :
  arithmetic_sequence a →
  k > 2 →
  sum_of_arithmetic_sequence a (k - 2) = -4 →
  sum_of_arithmetic_sequence a k = 0 →
  sum_of_arithmetic_sequence a (k + 2) = 8 →
  k = 6 := by
  sorry

#check arithmetic_sequence_sum_property

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_property_l1293_129386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_min_equation_l1293_129323

-- Define our own min function for two rational numbers
def myMin (a b : ℚ) : ℚ := if a ≤ b then a else b

-- State the theorem
theorem unique_solution_min_equation :
  ∃! x : ℚ, myMin x (-x) = 3 * x + 4 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_min_equation_l1293_129323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1293_129329

/-- Ellipse G with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0

/-- Circle C: x² + y² - 2x - 2y = 0 -/
def Circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 2*y = 0

/-- Point B is the intersection of ellipse G with positive y-axis -/
def B (G : Ellipse) : ℝ × ℝ :=
  (0, G.b)

/-- Point F is the right focus of ellipse G -/
noncomputable def F (G : Ellipse) : ℝ × ℝ :=
  (Real.sqrt (G.a^2 - G.b^2), 0)

/-- Line l passing through point C(1,1) -/
structure Line where
  m : ℝ  -- slope
  c : ℝ  -- y-intercept

/-- Theorem statement -/
theorem ellipse_properties (G : Ellipse) 
  (h_B : Circle (B G).1 (B G).2)
  (h_F : Circle (F G).1 (F G).2) :
  (G.a = 2*Real.sqrt 2 ∧ G.b = 2) ∧
  ∀ (l : Line), ∃ (M N : ℝ × ℝ),
    (M.1 + N.1 = 2 ∧ M.2 + N.2 = 2) →
    (M.1 - N.1)^2 + (M.2 - N.2)^2 = (5*Real.sqrt 6 / 3)^2 := by
  sorry

#check ellipse_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1293_129329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_value_l1293_129382

noncomputable def f (ω φ x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem min_omega_value (ω φ : ℝ) (h_ω_pos : ω > 0)
  (h_even : ∀ x, f ω φ x = f ω φ (-x))
  (h_symmetry : ∀ x, f ω φ (1 + x) = -f ω φ (1 - x)) :
  ω ≥ π / 2 ∧ ∃ ω₀ φ₀, ω₀ = π / 2 ∧ 
    (∀ x, f ω₀ φ₀ x = f ω₀ φ₀ (-x)) ∧
    (∀ x, f ω₀ φ₀ (1 + x) = -f ω₀ φ₀ (1 - x)) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_value_l1293_129382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_in_cube_l1293_129377

/-- The side length of a regular tetrahedron inscribed in a cube -/
noncomputable def tetrahedron_side_length (cube_side_length : ℝ) : ℝ :=
  4 * Real.sqrt 3

/-- A regular tetrahedron -/
def is_regular_tetrahedron (A B C D : ℝ × ℝ × ℝ) : Prop :=
  sorry

/-- A point is on the body diagonal of a cube -/
def on_body_diagonal (cube_side_length : ℝ) (P : ℝ × ℝ × ℝ) : Prop :=
  sorry

/-- A point is on a face diagonal of a cube -/
def on_face_diagonal (cube_side_length : ℝ) (P : ℝ × ℝ × ℝ) : Prop :=
  sorry

/-- The face diagonal does not intersect the body diagonal -/
def face_diagonal_not_intersect_body_diagonal (C D : ℝ × ℝ × ℝ) : Prop :=
  sorry

/-- The problem statement -/
theorem tetrahedron_in_cube (cube_side_length : ℝ) 
  (h1 : cube_side_length = 12) 
  (h2 : ∃ (A B C D : ℝ × ℝ × ℝ), 
    is_regular_tetrahedron A B C D ∧
    on_body_diagonal cube_side_length A ∧
    on_body_diagonal cube_side_length B ∧
    on_face_diagonal cube_side_length C ∧
    on_face_diagonal cube_side_length D ∧
    face_diagonal_not_intersect_body_diagonal C D) :
  tetrahedron_side_length cube_side_length = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_in_cube_l1293_129377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sign_for_a_zero_a_value_for_local_max_l1293_129374

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2 + x + a * x^2) * Real.log (1 + x) - 2 * x

-- Part 1
theorem f_sign_for_a_zero (x : ℝ) (hx : -1 < x ∧ x ≠ 0) :
  (x < 0 → f 0 x < 0) ∧ (x > 0 → f 0 x > 0) := by sorry

-- Part 2
theorem a_value_for_local_max :
  (∃ δ > 0, ∀ x, 0 < |x| ∧ |x| < δ → f (-1/6) x ≤ f (-1/6) 0) ∧
  (∀ a ≠ -1/6, ¬∃ δ > 0, ∀ x, 0 < |x| ∧ |x| < δ → f a x ≤ f a 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sign_for_a_zero_a_value_for_local_max_l1293_129374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_existence_l1293_129387

/-- Represents a circle in Euclidean space. -/
structure Circle where
  center : EuclideanSpace ℝ (Fin 2)
  radius : ℝ

/-- Represents a line in Euclidean space. -/
structure Line where
  point : EuclideanSpace ℝ (Fin 2)
  direction : EuclideanSpace ℝ (Fin 2)

/-- Represents the angle between two lines in Euclidean space. -/
def AngleBetween (l1 l2 : Line) : ℝ := sorry

/-- Defines the number of solutions (0, 1, or 2) for the inscribed triangle problem. -/
inductive NumberOfSolutions : Type
| zero : NumberOfSolutions
| one : NumberOfSolutions
| two : NumberOfSolutions

/-- Given a circle, two lines, and a point, this theorem states the conditions
    for the existence of inscribed triangles with specific properties. -/
theorem inscribed_triangle_existence
  (O : EuclideanSpace ℝ (Fin 2)) -- Center of the circle
  (r : ℝ) -- Radius of the circle
  (a b : Line) -- Given lines
  (γ : ℝ) -- Angle between lines a and b
  (P : EuclideanSpace ℝ (Fin 2)) -- Given point
  (h_circle : Circle) -- The circle with center O and radius r
  (h_angle : AngleBetween a b = γ) -- The angle between lines a and b is γ
  (h_pos : 0 < r ∧ 0 < γ ∧ γ < Real.pi) -- Positive radius and angle between 0 and π
  : ∃ n : NumberOfSolutions, True :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_existence_l1293_129387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orthocenter_ratio_two_to_one_l1293_129355

/-- Triangle ABC with orthocenter H and altitudes AD, BE, CF -/
structure OrthocenterTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  H : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ
  BC : ℝ
  AC : ℝ
  angle_C : ℝ

/-- The ratio of AH to HD in the orthocenter triangle -/
noncomputable def ah_hd_ratio (t : OrthocenterTriangle) : ℝ := 
  let (ax, ay) := t.A
  let (hx, hy) := t.H
  let (dx, dy) := t.D
  let ah := Real.sqrt ((ax - hx)^2 + (ay - hy)^2)
  let hd := Real.sqrt ((hx - dx)^2 + (hy - dy)^2)
  ah / hd

/-- Theorem stating the ratio of AH to HD is 2:1 under given conditions -/
theorem orthocenter_ratio_two_to_one (t : OrthocenterTriangle) 
  (h_BC : t.BC = 6)
  (h_AC : t.AC = 3 * Real.sqrt 3)
  (h_angle_C : t.angle_C = π / 6) : -- 30° in radians
  ah_hd_ratio t = 2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_orthocenter_ratio_two_to_one_l1293_129355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_ln_curve_l1293_129369

noncomputable def f (x : ℝ) : ℝ := Real.log x

def tangentLine (x m : ℝ) : ℝ := 2 * x + m

theorem tangent_line_to_ln_curve (x : ℝ) (h : x > 0) :
  ∃ m : ℝ, (∀ ε > 0, ∃ δ > 0, ∀ y, |y - x| < δ → |f y - tangentLine y m| < ε * |y - x|) →
  m = -1 - Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_ln_curve_l1293_129369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_statements_true_l1293_129373

-- Define the quadratic function
noncomputable def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the properties of the function
variable (a b c : ℝ)

axiom a_nonzero : a ≠ 0
axiom axis_of_symmetry : ∀ x, quadratic_function a b c (1 - x) = quadratic_function a b c (1 + x)
axiom minimum_value : quadratic_function a b c 1 = -4
axiom x_intercepts : quadratic_function a b c (-1) = 0 ∧ quadratic_function a b c 3 = 0

-- Define the statements to be proven
def statement1 : Prop := ∀ x, quadratic_function a b c x ≥ -3
def statement2 : Prop := ∀ x, -1/2 < x → x < 2 → quadratic_function a b c x < 0
def statement3 : Prop := ∃ x₁ x₂, x₁ < 0 ∧ x₂ > 0 ∧ quadratic_function a b c x₁ = 0 ∧ quadratic_function a b c x₂ = 0

-- Theorem to be proven
theorem exactly_two_statements_true :
  (¬statement1 a b c ∧ statement2 a b c ∧ statement3 a b c) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_statements_true_l1293_129373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_distance_in_range_l1293_129366

/-- Represents the car's fuel consumption and modification details -/
structure CarData where
  initial_consumption : ℝ  -- Liters per 100 km before modification
  modified_consumption : ℝ  -- Liters per 100 km after modification
  modification_cost : ℝ  -- Cost of modification in dollars
  gas_cost : ℝ  -- Cost of gas per liter in dollars

/-- Calculates the minimum distance needed to recover modification cost -/
noncomputable def minimum_distance_to_recover (car : CarData) : ℝ :=
  let savings_per_100km := car.initial_consumption - car.modified_consumption
  let equivalent_liters := car.modification_cost / car.gas_cost
  (equivalent_liters / savings_per_100km) * 100

/-- Theorem stating that the minimum distance to recover cost is between 22000 and 26000 km -/
theorem minimum_distance_in_range (car : CarData) 
  (h1 : car.initial_consumption = 8.4)
  (h2 : car.modified_consumption = 6.3)
  (h3 : car.modification_cost = 400)
  (h4 : car.gas_cost = 0.8) :
  22000 < minimum_distance_to_recover car ∧ minimum_distance_to_recover car < 26000 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval minimum_distance_to_recover ⟨8.4, 6.3, 400, 0.8⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_distance_in_range_l1293_129366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_payment_l1293_129305

/-- Represents the pricing tiers for candy bars -/
def price_tier (n : ℕ) : ℚ :=
  if n ≤ 4 then 3/2
  else if n ≤ 9 then 27/20
  else if n ≤ 14 then 6/5
  else 21/20

/-- Calculates the total cost of candy bars before tax -/
def total_cost_before_tax (n : ℕ) : ℚ := n * price_tier n

/-- Calculates the sales tax amount -/
def sales_tax (cost : ℚ) : ℚ := cost * (7/100)

/-- Calculates Dave's portion of the total cost -/
def dave_portion (total_bars : ℕ) (dave_bars : ℕ) (total_cost : ℚ) : ℚ :=
  (dave_bars : ℚ) / (total_bars : ℚ) * total_cost

theorem john_payment (total_bars : ℕ) (dave_bars : ℕ) : 
  total_bars = 20 → dave_bars = 6 →
  let total_cost := total_cost_before_tax total_bars
  let tax := sales_tax total_cost
  let dave_cost := dave_portion total_bars dave_bars (total_cost + tax)
  ⌊(total_cost + tax - dave_cost) * 100⌋ = 1573 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_payment_l1293_129305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1293_129371

-- Define the function f(x)
noncomputable def f (x : ℝ) := Real.log (x + 1) - x

-- Theorem statement
theorem f_properties :
  (∀ x > 0, (deriv f) x < 0) ∧
  (∀ x > -1, Real.log (x + 1) ≤ x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1293_129371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_distances_exist_l1293_129332

-- Define the square
def Square : Set (ℝ × ℝ) := {p | 0 ≤ p.1 ∧ p.1 ≤ 10 ∧ 0 ≤ p.2 ∧ p.2 ≤ 10}

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem equal_distances_exist (points : Finset (ℝ × ℝ)) 
  (h1 : points.card = 6)
  (h2 : ∀ p, p ∈ points → p ∈ Square)
  (h3 : ∀ p1 p2, p1 ∈ points → p2 ∈ points → p1 ≠ p2 → ∃ n : ℕ, distance p1 p2 = n) :
  ∃ p1 p2 p3 p4, p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧ p4 ∈ points ∧ 
    p1 ≠ p2 ∧ p3 ≠ p4 ∧ (p1, p2) ≠ (p3, p4) ∧ 
    distance p1 p2 = distance p3 p4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_distances_exist_l1293_129332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_idempotent_matrices_rank_l1293_129325

theorem idempotent_matrices_rank (n k : ℕ) (A : Fin k → Matrix (Fin n) (Fin n) ℂ)
  (h_idempotent : ∀ i, A i * A i = A i)
  (h_anticommute : ∀ i j, i < j → A i * A j = -(A j * A i)) :
  ∃ i, Matrix.rank (A i) ≤ n / k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_idempotent_matrices_rank_l1293_129325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_negative_sqrt3_over_2_relationship_l1293_129384

theorem cos_negative_sqrt3_over_2_relationship (α : ℝ) :
  (∃ k : ℤ, α = 2 * k * Real.pi + 5 * Real.pi / 6) →
  (Real.cos α = -Real.sqrt 3 / 2) ∧
  ¬ ((∃ k : ℤ, α = 2 * k * Real.pi + 5 * Real.pi / 6) ↔
     (Real.cos α = -Real.sqrt 3 / 2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_negative_sqrt3_over_2_relationship_l1293_129384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bernoulli_two_successes_l1293_129368

/-- Probability of first success on nth trial and second success after mth trial in Bernoulli trials -/
theorem bernoulli_two_successes (p : ℝ) (n m : ℕ) (h_p : 0 < p ∧ p < 1) (h_m : m > n) :
  let q := 1 - p
  ∃ (P : ℝ), P = p * q^(m-1) ∧
    P = ∑' (k : ℕ), if k ≥ m - n then q^(n-1) * p * q^k * p else 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bernoulli_two_successes_l1293_129368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_propositions_truth_values_l1293_129300

-- Define the propositions
def proposition1 : Prop := ∀ x y : ℝ, (x - 1)^2 + (y + 1)^2 = 0 → (x = 1 ∨ y = -1)

def proposition2 : Prop := Even 1 ∨ Odd 1

-- Define IsEquilateralTriangle ourselves since it's not in the standard library
def IsEquilateralTriangle (a b c : ℝ) : Prop := a = b ∧ b = c ∧ c = a

def proposition3 : Prop := ¬(∀ a b c : ℝ, IsEquilateralTriangle a b c → a = b ∧ b = c ∧ c = a)

def proposition4 : Prop := (∀ x : ℝ, x^2 + x + 1 > 0) ∨ (∀ x : ℝ, x^2 - x > 0)

-- Theorem statement
theorem propositions_truth_values :
  ¬proposition1 ∧
  proposition2 ∧
  ¬proposition3 ∧
  proposition4 := by
  sorry -- Skip the proof for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_propositions_truth_values_l1293_129300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_around_interior_point_l1293_129322

-- Define the Point type
def Point := ℝ × ℝ

-- Define the Quadrilateral structure
structure Quadrilateral :=
  (A B C D E : Point)

-- Define the AngleSum proposition
def AngleSum (Q : Quadrilateral) (p q r s t : ℝ) : Prop :=
  p + q + r + s + t = 360

-- Theorem statement
theorem angle_around_interior_point (Q : Quadrilateral) (p q r s t : ℝ) 
  (h : AngleSum Q p q r s t) : p = 360 - q - r - s - t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_around_interior_point_l1293_129322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sampling_teacher_count_l1293_129385

theorem stratified_sampling_teacher_count :
  ∀ (total_teachers : ℕ),
    let senior_teachers : ℕ := 20
    let intermediate_teachers : ℕ := 30
    let survey_size : ℕ := 20
    let other_selected : ℕ := 10
    let senior_intermediate_selected : ℕ := survey_size - other_selected
    (senior_intermediate_selected : ℚ) / (senior_teachers + intermediate_teachers : ℚ) =
    (survey_size : ℚ) / (total_teachers : ℚ) →
    total_teachers = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sampling_teacher_count_l1293_129385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_distance_sum_l1293_129320

/-- The circle on which point P lies -/
def circleEq (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 4

/-- The sum of squared distances from P to A and B -/
def distanceSum (x y : ℝ) : ℝ :=
  (x + 1)^2 + y^2 + (x - 1)^2 + y^2

/-- Point A coordinates -/
def A : ℝ × ℝ := (-1, 0)

/-- Point B coordinates -/
def B : ℝ × ℝ := (1, 0)

/-- Point P coordinates -/
noncomputable def P : ℝ × ℝ := (9/5, 12/5)

theorem minimize_distance_sum :
  circleEq P.1 P.2 ∧
  ∀ x y, circleEq x y → distanceSum P.1 P.2 ≤ distanceSum x y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_distance_sum_l1293_129320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l1293_129399

noncomputable def f (x : ℝ) : ℝ := (x^2 - 1) / ((x - 4)^2)

theorem inequality_solution_set :
  ∀ x : ℝ, x ≠ 4 → (f x ≥ 0 ↔ x ∈ Set.Ici (-1) ∪ Set.Icc 1 4 ∪ Set.Ioi 4) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l1293_129399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_last_score_is_99_l1293_129354

def scores : List ℕ := [65, 72, 78, 84, 90, 99]

def is_integer_average (partial_scores : List ℕ) : Prop :=
  ∀ n : ℕ, n ≤ partial_scores.length → (partial_scores.take n).sum % n = 0

def valid_entry_order (order : List ℕ) : Prop :=
  order.toFinset = scores.toFinset ∧ is_integer_average order

theorem second_last_score_is_99 :
  ∀ order : List ℕ, valid_entry_order order → 
    order.reverse.get? (order.length - 2) = some 99 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_last_score_is_99_l1293_129354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_score_correction_effect_l1293_129392

/-- Represents a class of students with their test scores -/
structure ClassData where
  size : Nat
  scores : Fin size → Real
  avg : Real
  variance : Real

/-- Calculates the new average and variance after correcting two scores -/
def correctScores (c : ClassData) (i j : Fin c.size) (newScoreI newScoreJ : Real) : Real × Real :=
  sorry

theorem score_correction_effect :
  let c : ClassData := {
    size := 48,
    scores := fun _ => 0,  -- We don't need to specify all scores
    avg := 70,
    variance := 75
  }
  let i : Fin c.size := ⟨0, by simp⟩  -- Index for student A
  let j : Fin c.size := ⟨1, by simp⟩  -- Index for student B
  let (newAvg, newVariance) := correctScores c i j 80 70
  newAvg = 70 ∧ newVariance = 50 := by sorry

#check score_correction_effect

end NUMINAMATH_CALUDE_ERRORFEEDBACK_score_correction_effect_l1293_129392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_EG_GF_ratio_l1293_129303

/-- Triangle ABC with specific properties -/
structure TriangleABC where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  M : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ
  G : ℝ × ℝ
  h_M_midpoint : M = ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  h_AB_length : Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 15
  h_AC_length : Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 20
  h_E_on_AC : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ E = (A.1 + t * (C.1 - A.1), A.2 + t * (C.2 - A.2))
  h_F_on_AB : ∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ F = (A.1 + s * (B.1 - A.1), A.2 + s * (B.2 - A.2))
  h_G_intersection : ∃ u v : ℝ, 
    G = (A.1 + u * (M.1 - A.1), A.2 + u * (M.2 - A.2)) ∧
    G = (E.1 + v * (F.1 - E.1), E.2 + v * (F.2 - E.2))
  h_AE_3AF : Real.sqrt ((E.1 - A.1)^2 + (E.2 - A.2)^2) = 3 * Real.sqrt ((F.1 - A.1)^2 + (F.2 - A.2)^2)

/-- The main theorem -/
theorem EG_GF_ratio (t : TriangleABC) : 
  let EG := Real.sqrt ((t.G.1 - t.E.1)^2 + (t.G.2 - t.E.2)^2)
  let GF := Real.sqrt ((t.F.1 - t.G.1)^2 + (t.F.2 - t.G.2)^2)
  EG / GF = 2 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_EG_GF_ratio_l1293_129303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_with_reverse_divisible_by_11_l1293_129346

/-- A two-digit number with non-zero digits -/
structure TwoDigitNumber where
  tens : Nat
  ones : Nat
  tens_nonzero : tens ≠ 0
  ones_nonzero : ones ≠ 0
  is_two_digit : tens < 10 ∧ ones < 10

/-- The reverse of a two-digit number -/
def reverse (n : TwoDigitNumber) : TwoDigitNumber where
  tens := n.ones
  ones := n.tens
  tens_nonzero := n.ones_nonzero
  ones_nonzero := n.tens_nonzero
  is_two_digit := ⟨n.is_two_digit.2, n.is_two_digit.1⟩

/-- The sum of a two-digit number and its reverse -/
def sum_with_reverse (n : TwoDigitNumber) : Nat :=
  10 * n.tens + n.ones + 10 * n.ones + n.tens

theorem sum_with_reverse_divisible_by_11 (n : TwoDigitNumber) :
  sum_with_reverse n ≠ 181 → (sum_with_reverse n) % 11 = 0 := by
  sorry

#check sum_with_reverse_divisible_by_11

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_with_reverse_divisible_by_11_l1293_129346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_right_shift_l1293_129380

/-- The minimum right shift to transform y = 2sin(2x) into y = sin(2x) - √3 * cos(2x) -/
theorem min_right_shift : ∃ (shift : ℝ), ∀ (x : ℝ),
  Real.sin (2 * x) - Real.sqrt 3 * Real.cos (2 * x) = 2 * Real.sin (2 * (x - shift)) :=
by
  -- We claim that the shift is π/6
  let shift := Real.pi / 6
  
  -- We'll prove that this shift works
  use shift
  
  -- The proof goes here, but we'll use sorry for now
  sorry

#check min_right_shift

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_right_shift_l1293_129380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l1293_129343

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := x * Real.cos x
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a * Real.sin x

-- Part 1
theorem part_one (x : ℝ) (h : x ∈ Set.Ioo 0 (Real.pi / 2)) :
  x > Real.sin x ∧ Real.sin x > f x := by
  sorry

-- Part 2
theorem part_two (a : ℝ) :
  (∀ x, x ∈ Set.Ioo (-Real.pi / 2) 0 ∪ Set.Ioo 0 (Real.pi / 2) →
    f x / g a x < Real.sin x / x) →
  a ∈ Set.Iio 0 ∪ Set.Ici 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l1293_129343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_cubes_120_280_360_l1293_129375

/-- The number of unit cubes an internal diagonal passes through in a rectangular solid -/
def diagonal_cubes (x y z : ℕ) : ℕ :=
  x + y + z - (Nat.gcd x y + Nat.gcd y z + Nat.gcd z x) + Nat.gcd x (Nat.gcd y z)

/-- Theorem: An internal diagonal of a 120 × 280 × 360 rectangular solid 
    passes through 690 unit cubes -/
theorem diagonal_cubes_120_280_360 :
  diagonal_cubes 120 280 360 = 690 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_cubes_120_280_360_l1293_129375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_tangent_circle_l1293_129351

/-- A circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The distance between two points in 2D space -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Check if a circle is tangent to a line x = a -/
def isTangentToVerticalLine (c : Circle) (a : ℝ) : Prop :=
  |c.center.1 - a| = c.radius

/-- Check if two circles are tangent -/
def areCirclesTangent (c1 c2 : Circle) : Prop :=
  distance c1.center c2.center = c1.radius + c2.radius

/-- The main theorem -/
theorem exists_tangent_circle : ∃ (c : Circle), 
  c.radius = 1 ∧ 
  isTangentToVerticalLine c (-1) ∧ 
  areCirclesTangent c ⟨(0, 0), 1⟩ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_tangent_circle_l1293_129351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_minus_sin_value_l1293_129330

theorem cos_minus_sin_value (θ : Real) 
  (h1 : θ > π/4) (h2 : θ < π/2) (h3 : Real.sin (2*θ) = 1/16) : 
  Real.cos θ - Real.sin θ = -Real.sqrt 15 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_minus_sin_value_l1293_129330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_share_is_32_l1293_129379

-- Define the total sum of money
def total_sum : ℚ := 164

-- Define the ratios of B's and C's shares relative to A's
def b_ratio : ℚ := 65 / 100
def c_ratio : ℚ := 40 / 100

-- Define A's share as a variable
variable (a_share : ℚ)

-- Define B's and C's shares in terms of A's
def b_share (a_share : ℚ) : ℚ := b_ratio * a_share
def c_share (a_share : ℚ) : ℚ := c_ratio * a_share

-- Theorem statement
theorem c_share_is_32 : 
  a_share + b_share a_share + c_share a_share = total_sum → c_share a_share = 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_share_is_32_l1293_129379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_100_value_l1293_129388

def b : ℕ → ℕ
  | 0 => 3  -- Add this case to handle Nat.zero
  | 1 => 3
  | n + 1 => b n + 3 * n

theorem b_100_value : b 100 = 14853 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_100_value_l1293_129388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_length_relation_l1293_129338

/-- Represents a triangle with medians -/
structure TriangleWithMedians where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  P : ℝ × ℝ  -- Midpoint of BC
  Q : ℝ × ℝ  -- Midpoint of AB
  O : ℝ × ℝ  -- Intersection point of medians AP and CQ

/-- The length of a line segment between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ := 
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Theorem: In a triangle with medians, if OQ = 4 and CQ = 12, then OP = 8 -/
theorem median_length_relation (t : TriangleWithMedians) 
  (h1 : distance t.O t.Q = 4)
  (h2 : distance t.C t.Q = 12) :
  distance t.O t.P = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_length_relation_l1293_129338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_dot_product_l1293_129358

-- Define the lines l1 and l2
def l1 (x y : ℝ) : Prop := x - y + 1 = 0
def l2 (x y : ℝ) : Prop := 2*x + y - 1 = 0

-- Define point A as the intersection of l1 and l2
def A : ℝ × ℝ := (0, 1)

-- Define line l passing through A with slope k
def l (k : ℝ) (x y : ℝ) : Prop := y = k*x + 1

-- Define circle C
def C (x y : ℝ) : Prop := (x - 2)^2 + (y - 3)^2 = 1

-- Define M and N as the intersection points of l and C
noncomputable def M (k : ℝ) : ℝ × ℝ := sorry
noncomputable def N (k : ℝ) : ℝ × ℝ := sorry

-- Theorem statement
theorem intersection_dot_product (k : ℝ) : 
  let am := ((M k).1 - A.1, (M k).2 - A.2)
  let an := ((N k).1 - A.1, (N k).2 - A.2)
  am.1 * an.1 + am.2 * an.2 = 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_dot_product_l1293_129358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1293_129349

noncomputable def total_work : ℝ := 1

noncomputable def micheal_rate : ℝ := total_work / 25

noncomputable def adam_rate : ℝ := total_work / 100

noncomputable def combined_rate : ℝ := micheal_rate + adam_rate

def initial_time : ℝ := 18

def adam_solo_time : ℝ := 10

theorem work_completion_time :
  (1 : ℝ) / combined_rate = 20 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1293_129349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_solution_of_equation_l1293_129397

-- Define the equation
def equation (x : ℝ) : Prop :=
  1 / (x - 5) + 1 / (x - 7) = 5 / (2 * (x - 6))

-- Define the solution
noncomputable def solution : ℝ := 7 - Real.sqrt 6

-- Theorem statement
theorem smallest_solution_of_equation :
  (∀ y, equation y → y ≥ solution) ∧ equation solution := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_solution_of_equation_l1293_129397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_zero_one_l1293_129326

-- Define the function f(x) = x - ln x
noncomputable def f (x : ℝ) : ℝ := x - Real.log x

-- State the theorem
theorem f_decreasing_on_zero_one :
  ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → x₂ < 1 → f x₂ < f x₁ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_zero_one_l1293_129326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_position_l1293_129350

def mySequence : ℕ → ℚ
  | 0 => 1
  | n + 1 => let m := n + 2
              let k := (m * (m + 1)) / 2 - n - 1
              (k : ℚ) / (m - k : ℚ)

theorem fraction_position : ∃ n : ℕ, mySequence n = 3 / 5 ∧ n + 1 = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_position_l1293_129350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_root_l1293_129381

noncomputable def f (x : ℝ) := x^7 - 14*x^6 + 21*x^5 - 70*x^4 + 35*x^3 - 42*x^2 + 7*x - 2

noncomputable def root : ℝ := 2 + Real.rpow 3 (1/7) + Real.rpow 3 (2/7) + Real.rpow 3 (3/7) + 
                 Real.rpow 3 (4/7) + Real.rpow 3 (5/7) + Real.rpow 3 (6/7)

theorem unique_root : 
  f root = 0 ∧ ∀ x : ℝ, f x = 0 → x = root := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_root_l1293_129381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_satisfy_equation_l1293_129347

noncomputable def a : ℝ × ℝ × ℝ := (1, 2, 2)
noncomputable def b : ℝ × ℝ × ℝ := (2, -3, 1)
noncomputable def c : ℝ × ℝ × ℝ := (4, 1, -5)

noncomputable def p : ℝ := 11 / 9
noncomputable def q : ℝ := 9 / 7
noncomputable def r : ℝ := -10 / 21

def dot_product (v w : ℝ × ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2.1 * w.2.1 + v.2.2 * w.2.2

def scale_vector (s : ℝ) (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (s * v.1, s * v.2.1, s * v.2.2)

def add_vectors (v w : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (v.1 + w.1, v.2.1 + w.2.1, v.2.2 + w.2.2)

theorem vectors_satisfy_equation :
  (dot_product a b = 0) →
  (dot_product a c = 0) →
  (dot_product b c = 0) →
  add_vectors (add_vectors (scale_vector p a) (scale_vector q b)) (scale_vector r c) = (3, -2, 6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_satisfy_equation_l1293_129347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_M_N_equals_60_l1293_129365

def is_permutation (xs : List ℕ) : Prop :=
  xs.Perm [1, 2, 3, 4, 6]

def product_sum (xs : List ℕ) : ℕ :=
  match xs with
  | [x₁, x₂, x₃, x₄, x₅] => x₁ * x₂ + x₂ * x₃ + x₃ * x₄ + x₄ * x₅ + x₅ * x₁
  | _ => 0

noncomputable def M : ℕ := 
  (List.permutations [1, 2, 3, 4, 6]).map product_sum |>.maximum?
    |>.getD 0

noncomputable def N : ℕ := 
  (List.permutations [1, 2, 3, 4, 6]).filter (fun xs => product_sum xs = M) |>.length

theorem sum_M_N_equals_60 : M + N = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_M_N_equals_60_l1293_129365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_composite_function_equality_l1293_129316

/-- Given functions f and g, prove that if f[g(x₀)] = 1, then x₀ = 4/3 -/
theorem composite_function_equality (f g : ℝ → ℝ) (x₀ : ℝ) 
  (hf : f = λ x ↦ 2 * x + 3)
  (hg : g = λ x ↦ 3 * x - 5)
  (h : f (g x₀) = 1) : 
  x₀ = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_composite_function_equality_l1293_129316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equals_result_l1293_129321

-- Define the integrand
noncomputable def f (x : ℝ) : ℝ := x / (2 + Real.sqrt (2 * x + 1))

-- State the theorem
theorem integral_equals_result : 
  ∫ x in (-1/2)..0, f x = 7/6 - 3 * Real.log (3/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equals_result_l1293_129321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_semi_ellipse_perimeter_approximation_l1293_129315

/-- Ramanujan's first approximation for the perimeter of a semi-ellipse -/
noncomputable def semiEllipsePerimeter (a b : ℝ) : ℝ :=
  Real.pi / 2 * (3 * (a + b) - Real.sqrt ((3 * a + b) * (a + 3 * b)))

/-- The problem statement -/
theorem semi_ellipse_perimeter_approximation :
  let a : ℝ := 14
  let b : ℝ := 10
  abs (semiEllipsePerimeter a b - 38.013) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_semi_ellipse_perimeter_approximation_l1293_129315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression1_eq_2_expression2_eq_neg_8_l1293_129340

-- Define the first expression
noncomputable def expression1 : ℝ := Real.sqrt (25 / 9) - (8 / 27) ^ (1 / 3 : ℝ) - (Real.pi + Real.exp 1) ^ (0 : ℝ) + (1 / 4) ^ (-(1 / 2) : ℝ)

-- Define the second expression
noncomputable def expression2 : ℝ := (Real.log 8 + Real.log 125 + Real.log 2 + Real.log 5) / (Real.log (Real.sqrt 10) * Real.log 0.1)

-- Theorem stating that the first expression equals 2
theorem expression1_eq_2 : expression1 = 2 := by sorry

-- Theorem stating that the second expression equals -8
theorem expression2_eq_neg_8 : expression2 = -8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression1_eq_2_expression2_eq_neg_8_l1293_129340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mikes_music_store_purchase_l1293_129337

/-- The cost of Mike's purchases at the music store -/
theorem mikes_music_store_purchase 
  (trumpet_cost : ℝ) (song_book_cost : ℝ) 
  (h1 : trumpet_cost = 145.16)
  (h2 : song_book_cost = 5.84) : 
  trumpet_cost + song_book_cost = 151.00 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mikes_music_store_purchase_l1293_129337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_and_function_extrema_l1293_129398

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.cos (3/2 * x), Real.sin (3/2 * x))
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos (x/2), -Real.sin (x/2))

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

def vector_sum (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)

noncomputable def vector_magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

noncomputable def f (x : ℝ) : ℝ := dot_product (a x) (b x) - vector_magnitude (vector_sum (a x) (b x))

theorem vector_properties_and_function_extrema :
  ∀ x ∈ Set.Icc (-π/3) (π/4),
    (dot_product (a x) (b x) = Real.cos (2*x)) ∧
    (vector_magnitude (vector_sum (a x) (b x)) = 2 * Real.cos x) ∧
    (∀ y ∈ Set.Icc (-π/3) (π/4), f y ≥ -Real.sqrt 2) ∧
    (∀ y ∈ Set.Icc (-π/3) (π/4), f y ≤ -1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_and_function_extrema_l1293_129398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marvelous_point_on_line_l1293_129361

/-- Marvelous point of a quadratic equation -/
noncomputable def marvelous_point (a b c : ℝ) : ℝ × ℝ :=
  let x1 := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x2 := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  (x1, x2)

/-- Line equation y = kx - 2(k-2) -/
def line_equation (k x : ℝ) : ℝ :=
  k * x - 2 * (k - 2)

theorem marvelous_point_on_line :
  ∀ k : ℝ, (let M := marvelous_point 1 (-6) 8; M.2 = line_equation k M.1) :=
by sorry

#check marvelous_point_on_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marvelous_point_on_line_l1293_129361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_skew_edge_midpoint_distance_formula_l1293_129367

/-- A regular octahedron with edge length a -/
structure RegularOctahedron (a : ℝ) where
  edge_length : a > 0

/-- The distance between midpoints of two skew edges in a regular octahedron -/
noncomputable def skew_edge_midpoint_distance (a : ℝ) (octahedron : RegularOctahedron a) : ℝ :=
  a * Real.sqrt 3 / 2

/-- Theorem: The distance between midpoints of two skew edges in a regular octahedron
    with edge length a is (a * √3) / 2 -/
theorem skew_edge_midpoint_distance_formula (a : ℝ) (octahedron : RegularOctahedron a) :
  skew_edge_midpoint_distance a octahedron = a * Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_skew_edge_midpoint_distance_formula_l1293_129367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_iff_a_in_range_l1293_129383

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then 2^x + a else a*x^2 + 2*x

theorem f_increasing_iff_a_in_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ a ∈ Set.Icc (-1 : ℝ) 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_iff_a_in_range_l1293_129383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bond_draw_probability_l1293_129364

/-- Represents a bond series -/
structure BondSeries where
  total_bonds : ℕ
  winning_bonds : ℕ

/-- Represents the draw results -/
structure DrawResult where
  thrice_drawn : ℕ
  twice_drawn : ℕ
  once_drawn : ℕ

/-- Calculates the probability of not winning for a given bond series and draw result -/
noncomputable def probability_not_winning (series : BondSeries) (draw : DrawResult) : ℝ :=
  let p_not_win : ℝ := (series.total_bonds - series.winning_bonds) / series.total_bonds
  (p_not_win ^ 3) ^ draw.thrice_drawn *
  (p_not_win ^ 2) ^ draw.twice_drawn *
  (p_not_win ^ 1) ^ draw.once_drawn

/-- The main theorem stating the probability of not winning in the given scenario -/
theorem bond_draw_probability (series : BondSeries) (draw : DrawResult) 
    (h1 : series.total_bonds = 1000)
    (h2 : series.winning_bonds = 100)
    (h3 : draw.thrice_drawn = 2)
    (h4 : draw.twice_drawn = 1)
    (h5 : draw.once_drawn = 2) :
  probability_not_winning series draw = (9/10 : ℝ) ^ 7 := by
  sorry

#eval (0.9 : Float) ^ 7  -- This should output approximately 0.47829

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bond_draw_probability_l1293_129364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_g_value_l1293_129389

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Distance between two points in 3D space -/
noncomputable def distance (p q : Point3D) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2)

/-- Tetrahedron with vertices P, Q, R, S -/
structure Tetrahedron where
  P : Point3D
  Q : Point3D
  R : Point3D
  S : Point3D

/-- Function g(X) for a given tetrahedron and point X -/
noncomputable def g (t : Tetrahedron) (X : Point3D) : ℝ :=
  distance t.P X + distance t.Q X + distance t.R X + distance t.S X

/-- Theorem: Minimum value of g(X) for the given tetrahedron -/
theorem min_g_value (t : Tetrahedron) 
  (h1 : distance t.P t.R = 26)
  (h2 : distance t.Q t.S = 26)
  (h3 : distance t.P t.S = 34)
  (h4 : distance t.Q t.R = 34)
  (h5 : distance t.P t.Q = 50)
  (h6 : distance t.R t.S = 50) :
  ∃ (X : Point3D), ∀ (Y : Point3D), g t X ≤ g t Y ∧ g t X = 2 * Real.sqrt 2642 :=
by sorry

#check min_g_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_g_value_l1293_129389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_equals_two_l1293_129390

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  if x ≤ 0 then x^2 + 2*x + 2 else -x^2

-- State the theorem
theorem f_composition_equals_two (a : ℝ) : f (f a) = 2 ↔ a = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_equals_two_l1293_129390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_2_equals_7_l1293_129336

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x + a^(-x)

-- State the theorem
theorem f_of_2_equals_7 
  (a : ℝ) 
  (h1 : a > 0) 
  (h2 : a ≠ 1) 
  (h3 : f a 1 = 3) : 
  f a 2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_2_equals_7_l1293_129336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_equals_radius_squared_ratio_l1293_129345

-- Define the circles and quadrilaterals
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

structure Quadrilateral where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

-- Define the problem setup
def concentricCircles (R R₁ : ℝ) (center : ℝ × ℝ) : Prop :=
  R₁ > R ∧ Circle center R ≠ ∅ ∧ Circle center R₁ ≠ ∅

def inscribedQuadrilateral (q : Quadrilateral) (c : Set (ℝ × ℝ)) : Prop :=
  q.A ∈ c ∧ q.B ∈ c ∧ q.C ∈ c ∧ q.D ∈ c

def pointsOnRays (q q₁ : Quadrilateral) : Prop :=
  ∃ t₁ t₂ t₃ t₄ : ℝ,
    t₁ > 1 ∧ t₂ > 1 ∧ t₃ > 1 ∧ t₄ > 1 ∧
    q₁.A = q.C + (t₁ - 1) • (q.D - q.C) ∧
    q₁.B = q.D + (t₂ - 1) • (q.A - q.D) ∧
    q₁.C = q.A + (t₃ - 1) • (q.B - q.A) ∧
    q₁.D = q.B + (t₄ - 1) • (q.C - q.B)

-- Define the area of a quadrilateral
noncomputable def area (q : Quadrilateral) : ℝ := sorry

-- State the theorem
theorem area_ratio_equals_radius_squared_ratio
  (center : ℝ × ℝ) (R R₁ : ℝ) (q q₁ : Quadrilateral) :
  concentricCircles R R₁ center →
  inscribedQuadrilateral q (Circle center R) →
  inscribedQuadrilateral q₁ (Circle center R₁) →
  pointsOnRays q q₁ →
  area q₁ / area q = R₁^2 / R^2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_equals_radius_squared_ratio_l1293_129345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_2_4_l1293_129359

/-- The area of the triangle formed by the intersection of three lines -/
noncomputable def triangleArea (line1 line2 line3 : ℝ → ℝ) : ℝ :=
  let x1 := (2 - 4) / 2  -- Intersection of line1 and y = 2
  let x2 := (2 - 3) / (-1/2)  -- Intersection of line2 and y = 2
  let x3 := (3 - 4) / (5/2)  -- Intersection of line1 and line2
  let base := x2 - x1
  let height := line1 x3 - 2
  (1 / 2) * base * height

theorem triangle_area_is_2_4 :
  triangleArea (λ x => 2 * x + 4) (λ x => -1/2 * x + 3) (λ _ => 2) = 2.4 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval triangleArea (λ x => 2 * x + 4) (λ x => -1/2 * x + 3) (λ _ => 2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_2_4_l1293_129359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_segment_speed_l1293_129352

/-- Proves that the average speed during the last 40 minutes of a 120-minute journey
    is 120 mph, given specific conditions. -/
theorem last_segment_speed (total_distance : ℝ) (total_time : ℝ) 
  (speed1 : ℝ) (speed2 : ℝ) (h1 : total_distance = 150) 
  (h2 : total_time = 120) (h3 : speed1 = 50) (h4 : speed2 = 55) : 
  ∃ speed3 : ℝ, speed3 = 120 := by
  let segment_time := total_time / 3
  let distance1 := speed1 * segment_time / 60
  let distance2 := speed2 * segment_time / 60
  let remaining_distance := total_distance - distance1 - distance2
  let speed3 := remaining_distance / (segment_time / 60)
  sorry -- Placeholder for the actual proof


end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_segment_speed_l1293_129352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_satisfies_conditions_l1293_129362

/-- A quadratic polynomial that satisfies the given conditions -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (3 / (2 * a)) * (x - a) * (x - 2 * a)

/-- Predicate to check if a function is a quadratic polynomial -/
def IsQuadratic (g : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x, g x = a * x^2 + b * x + c

/-- The existence of a quadratic polynomial g with the required properties -/
def exists_g (a : ℝ) : Prop :=
  ∃ g : ℝ → ℝ, 
    IsQuadratic g ∧ 
    (∀ x, g (f a x) = 0 ↔ f a x * g x = 0) ∧
    (∃ r₁ r₂ r₃, r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧
      g (f a r₁) = 0 ∧ g (f a r₂) = 0 ∧ g (f a r₃) = 0 ∧
      r₂ - r₁ = r₃ - r₂)

/-- The main theorem stating that f satisfies the required conditions for any non-zero a -/
theorem f_satisfies_conditions (a : ℝ) (ha : a ≠ 0) : exists_g a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_satisfies_conditions_l1293_129362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l1293_129301

theorem equation_solutions (x : ℝ) :
  (3 : ℝ)^(2*x) - 13 * (3 : ℝ)^x + 40 = 0 ↔ x = Real.log 8 / Real.log 3 ∨ x = Real.log 5 / Real.log 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l1293_129301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_breadth_cube_root_549_l1293_129341

noncomputable def rectangular_prism_breadth (volume : ℝ) : ℝ :=
  (volume / 6) ^ (1/3)

theorem breadth_cube_root_549 (volume : ℝ) :
  volume = 3294 →
  let length := 3 * rectangular_prism_breadth volume
  let height := 2 * rectangular_prism_breadth volume
  volume = length * rectangular_prism_breadth volume * height →
  rectangular_prism_breadth volume = 549 ^ (1/3) :=
by
  intro h_volume
  intro h_eq
  have h1 : volume / 6 = 549 := by
    rw [h_volume]
    norm_num
  have h2 : rectangular_prism_breadth volume = (volume / 6) ^ (1/3) := by rfl
  rw [h2, h1]
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_breadth_cube_root_549_l1293_129341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_earl_stuffs_36_l1293_129304

/-- Earl's envelope stuffing rate (envelopes per minute) -/
def earl_rate : ℚ := sorry

/-- Ellen's envelope stuffing rate (envelopes per minute) -/
def ellen_rate : ℚ := sorry

/-- Ellen takes 1.5 times as long as Earl to stuff the same number of envelopes -/
axiom ellen_earl_ratio : ellen_rate = (2/3) * earl_rate

/-- Earl and Ellen can stuff 60 envelopes together in 1 minute -/
axiom combined_rate : earl_rate + ellen_rate = 60

/-- Theorem: Earl can stuff 36 envelopes in one minute -/
theorem earl_stuffs_36 : earl_rate = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_earl_stuffs_36_l1293_129304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_coordinates_l1293_129393

/-- A circle passing through (1,0) and tangent to lines x=-1 and y=4 has its center at (1,2) or (9,-6) -/
theorem circle_center_coordinates : 
  ∀ (C : Set (ℝ × ℝ)) (center : ℝ × ℝ),
    (∀ (p : ℝ × ℝ), p ∈ C ↔ (p.1 - center.1)^2 + (p.2 - center.2)^2 = (center.1 + 1)^2) →  -- Circle equation
    (1, 0) ∈ C →  -- Circle passes through (1,0)
    (∀ (y : ℝ), (-1, y) ∉ C) →  -- Circle is tangent to x=-1
    (∀ (x : ℝ), (x, 4) ∉ C) →  -- Circle is tangent to y=4
    center = (1, 2) ∨ center = (9, -6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_coordinates_l1293_129393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_james_average_speed_l1293_129335

/-- Calculates the average riding speed given total distance, total trip time, and break time -/
noncomputable def averageRidingSpeed (totalDistance : ℝ) (totalTripTime : ℝ) (breakTime : ℝ) : ℝ :=
  totalDistance / (totalTripTime - breakTime)

/-- Theorem stating that given the specific conditions, the average riding speed is 5.25 mph -/
theorem james_average_speed :
  let totalDistance : ℝ := 42
  let totalTripTime : ℝ := 9
  let breakTime : ℝ := 1
  averageRidingSpeed totalDistance totalTripTime breakTime = 5.25 := by
  -- Unfold the definition of averageRidingSpeed
  unfold averageRidingSpeed
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_james_average_speed_l1293_129335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_iff_b_eq_zero_l1293_129357

/-- A function f : ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The function f(x) = 3x + b*cos(x) -/
noncomputable def f (b : ℝ) : ℝ → ℝ := λ x ↦ 3*x + b * Real.cos x

/-- Theorem: f(x) = 3x + b*cos(x) is an odd function if and only if b = 0 -/
theorem f_odd_iff_b_eq_zero (b : ℝ) : IsOdd (f b) ↔ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_iff_b_eq_zero_l1293_129357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_euler_line_l1293_129312

/-- Predicate stating that G is the centroid of triangle ABC -/
def is_centroid (G A B C : ℝ × ℝ) : Prop := sorry

/-- Predicate stating that H is the orthocenter of triangle ABC -/
def is_orthocenter (H A B C : ℝ × ℝ) : Prop := sorry

/-- Predicate stating that O is the circumcenter of triangle ABC -/
def is_circumcenter (O A B C : ℝ × ℝ) : Prop := sorry

/-- Predicate stating that points P, Q, and R are collinear -/
def collinear (P Q R : ℝ × ℝ) : Prop := sorry

/-- Given a triangle ABC, its centroid G, orthocenter H, and circumcenter O are collinear. -/
theorem euler_line (A B C G H O : ℝ × ℝ) 
  (hG : is_centroid G A B C) (hH : is_orthocenter H A B C) (hO : is_circumcenter O A B C) : 
  collinear G H O := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_euler_line_l1293_129312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_crescent_gemini_l1293_129308

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle -/
structure Circle where
  center : Point
  radius : ℝ

/-- The Crescent Gemini region -/
structure CrescentGemini where
  largeCircle : Circle
  smallCircle : Circle
  largeCircleCenter : largeCircle.center = ⟨0, 0⟩
  largeCircleRadius : largeCircle.radius = 5
  smallCircleCenter : smallCircle.center = ⟨0, 3⟩
  smallCircleRadius : smallCircle.radius = 2

/-- Calculates the area of the Crescent Gemini -/
noncomputable def areaOfCrescentGemini (cg : CrescentGemini) : ℝ := 
  17 * Real.pi / 4

/-- Theorem stating that the area of the Crescent Gemini is 17π/4 -/
theorem area_of_crescent_gemini :
  ∀ cg : CrescentGemini, areaOfCrescentGemini cg = 17 * Real.pi / 4 := by
  intro cg
  rfl

#check area_of_crescent_gemini

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_crescent_gemini_l1293_129308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sector_angle_l1293_129342

theorem smallest_sector_angle (n : ℕ) (a d : ℤ) : 
  n = 15 ∧ 
  (∀ k, 0 ≤ k ∧ k < n → a + k * d > 0) ∧
  (∀ k, 0 ≤ k ∧ k < n → (a + k * d).natAbs = a + k * d) ∧
  n * (2 * a + (n - 1) * d) = 720 →
  a ≥ 3 ∧ (∃ d', a = 3 ∧ d = d') :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sector_angle_l1293_129342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_constant_l1293_129310

/-- The parabola parameter -/
noncomputable def p : ℝ := 2

/-- The line equation -/
def line (x : ℝ) : ℝ := 2 * x

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y^2 = 2 * p * x

/-- Point O -/
def O : ℝ × ℝ := (0, 0)

/-- Point E -/
noncomputable def E : ℝ × ℝ := (p / 2, p)

/-- The distance between O and E -/
noncomputable def OE_distance : ℝ := Real.sqrt 5

/-- Point Q -/
def Q : ℝ × ℝ := (2, 0)

/-- Point P -/
def P : ℝ → ℝ × ℝ := fun y₀ ↦ (-2, y₀)

theorem parabola_intersection_constant (y₀ : ℝ) :
  ∃ (xM xN : ℝ),
    (∃ (y₁ y₂ : ℝ), 
      parabola xM y₁ ∧ parabola xN y₂ ∧
      (y₁ - y₀) / (xM + 2) = (0 - y₀) / (0 + 2) ∧
      (y₂ - y₀) / (xN + 2) = (0 - y₀) / (0 + 2)) →
    xM * xN = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_constant_l1293_129310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_sequence_ratio_l1293_129311

def sequence_a : ℕ → ℚ
| 0 => 10
| n + 1 => sequence_a n + 2 * (n + 1)

theorem min_value_of_sequence_ratio : 
  ∃ (n : ℕ), ∀ (m : ℕ), m > 0 → sequence_a n / (n + 1) ≤ sequence_a m / (m + 1) ∧ 
  sequence_a n / (n + 1) = 16 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_sequence_ratio_l1293_129311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_staff_members_count_l1293_129344

theorem staff_members_count (teachers : ℕ) (non_pizza_eaters : ℕ) :
  teachers = 30 →
  non_pizza_eaters = 19 →
  ∃ (staff : ℕ),
    (2 : ℚ) / 3 * teachers + (4 : ℚ) / 5 * staff + non_pizza_eaters = teachers + staff ∧
    staff = 45 :=
by
  intro h_teachers h_non_pizza
  use 45
  constructor
  · simp [h_teachers, h_non_pizza]
    norm_num
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_staff_members_count_l1293_129344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speech_competition_arrangements_l1293_129376

-- Define the number of contestants for each stroke count
def contestants_4_strokes : ℕ := 2
def contestants_5_strokes : ℕ := 3
def contestants_6_strokes : ℕ := 4
def contestants_7_strokes : ℕ := 1

-- Define the total number of contestants
def total_contestants : ℕ := contestants_4_strokes + contestants_5_strokes + contestants_6_strokes + contestants_7_strokes

-- Define the function to calculate the number of possible arrangements
def num_arrangements (n : ℕ) : ℕ := Nat.factorial n

-- Theorem statement
theorem speech_competition_arrangements :
  (num_arrangements contestants_4_strokes) *
  (num_arrangements contestants_5_strokes) *
  (num_arrangements contestants_6_strokes) *
  (num_arrangements contestants_7_strokes) = 288 ∧
  total_contestants = 10 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_speech_competition_arrangements_l1293_129376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_identity_l1293_129313

-- Define the function f
noncomputable def f : ℝ → ℝ := λ x ↦ x^2 + (3/2) * x

-- State the theorem
theorem f_identity : ∀ x : ℝ, f (2 * x) = 4 * x^2 + 3 * x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_identity_l1293_129313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l1293_129370

variable (a b : ℝ)

-- Define the functions f and g
noncomputable def f (a b x : ℝ) : ℝ := 2 * (a + b) * Real.exp (2 * x) + 2 * a * b

noncomputable def g (a b x : ℝ) : ℝ := 4 * Real.exp (2 * x) + a + b

-- Define t as ((a^(1/3) + b^(1/3))/2)^3
noncomputable def t (a b : ℝ) : ℝ := ((a ^ (1/3) + b ^ (1/3)) / 2) ^ 3

-- State the theorem
theorem unique_solution (h1 : a > b) (h2 : b > 0) :
  ∃! x, f a b x = t a b * g a b x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l1293_129370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_max_ratio_l1293_129395

/-- Given a sector with central angle 2 radians, this theorem proves that the maximum value of (C-1)/S is 4, where C is the perimeter and S is the area of the sector. -/
theorem sector_max_ratio (r : ℝ) (h : r > 0) : 
  (∀ x : ℝ, x > 0 → (4 * x - 1) / x^2 ≤ (4 * r - 1) / r^2) ∧ 
  (∃ x : ℝ, x > 0 ∧ (4 * x - 1) / x^2 = (4 * r - 1) / r^2) :=
by
  sorry

#check sector_max_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_max_ratio_l1293_129395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_g_max_value_l1293_129353

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sin x + (x - 1) / Real.exp x

-- Define the domain
def domain : Set ℝ := Set.Icc (-Real.pi) (Real.pi / 2)

-- Statement 1: f is monotonically increasing on the domain
theorem f_monotone_increasing : 
  StrictMonoOn f domain := by sorry

-- Define the function g for the second part
noncomputable def g (x : ℝ) : ℝ := ((f x - Real.sin x) * Real.exp x - Real.cos x) / Real.sin x

-- Define the subdomain for the second part
def subdomain : Set ℝ := Set.Icc (-Real.pi) 0

-- Statement 2: The maximum value of g on the subdomain is 1 + π/2
theorem g_max_value : 
  ∃ (x : ℝ), x ∈ subdomain ∧ g x = 1 + Real.pi / 2 ∧ 
  ∀ (y : ℝ), y ∈ subdomain → g y ≤ g x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_g_max_value_l1293_129353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_art_class_overlap_l1293_129334

theorem art_class_overlap (total dancing singing instruments : ℕ) 
  (h1 : total = 100)
  (h2 : dancing = 67)
  (h3 : singing = 45)
  (h4 : instruments = 21)
  (h5 : dancing + singing - (Finset.card (Finset.range dancing ∩ Finset.range singing)) + instruments = total) :
  Finset.card (Finset.range dancing ∩ Finset.range singing) = 33 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_art_class_overlap_l1293_129334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1293_129331

noncomputable section

-- Define the function f(x)
def f (x : ℝ) (a : ℝ) : ℝ := 2 * Real.sin (2 * x - Real.pi / 6) + a

-- State the theorem
theorem function_properties :
  -- The smallest positive period of f(x) is π
  (∃ (T : ℝ), T > 0 ∧ T = Real.pi ∧ ∀ (x : ℝ), f x a = f (x + T) a) ∧
  -- Given that the minimum value of f(x) is -2 for x ∈ [0, π/2], the value of a is -1
  (∀ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) → f x (-1) ≥ -2) ∧
  (∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) ∧ f x (-1) = -2) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1293_129331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_payment_calculation_l1293_129307

-- Define constants
def initial_price : ℚ := 600
def phone_discount_rate : ℚ := 1/5
def case_cost : ℚ := 25
def protector_cost : ℚ := 15
def first_discount_rate : ℚ := 1/20
def first_discount_threshold : ℚ := 125
def second_discount_rate : ℚ := 1/10
def second_discount_threshold : ℚ := 150
def final_discount_rate : ℚ := 3/100
def exchange_fee_rate : ℚ := 1/50

-- Define functions
def discounted_phone_price : ℚ := initial_price * phone_discount_rate

def total_before_discounts : ℚ := discounted_phone_price + case_cost + protector_cost

noncomputable def apply_first_discount (total : ℚ) : ℚ :=
  if total > first_discount_threshold then total * (1 - first_discount_rate) else total

noncomputable def apply_second_discount (total : ℚ) : ℚ :=
  if total > second_discount_threshold then total * (1 - second_discount_rate) else total

def apply_final_discount (total : ℚ) : ℚ := total * (1 - final_discount_rate)

def apply_exchange_fee (total : ℚ) : ℚ := total * (1 + exchange_fee_rate)

-- Theorem statement
theorem total_payment_calculation :
  let price_after_discounts := apply_final_discount (apply_second_discount (apply_first_discount total_before_discounts))
  let final_price := apply_exchange_fee price_after_discounts
  ∃ (n : ℕ), (n : ℚ) / 100 = final_price ∧ n = 13535 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_payment_calculation_l1293_129307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l1293_129333

/-- The function f(x) = x^2 - mx + 5 -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x^2 - m*x + 5

/-- The minimum value function g(m) -/
noncomputable def g (m : ℝ) : ℝ :=
  if m ≤ -2 then m + 6
  else if m ≤ 2 then 5 - m^2/4
  else 6 - m

/-- Theorem stating that g(m) is the minimum value of f(x) in the interval [-1, 1] -/
theorem min_value_theorem (m : ℝ) :
  IsGLB {f m x | x ∈ Set.Icc (-1 : ℝ) 1} (g m) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l1293_129333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_liquid_evaporation_l1293_129372

theorem liquid_evaporation (x : ℝ) : 
  x ∈ Set.Icc 0 1 →
  (1 - x) * (1/4 : ℝ) = 1/6 →
  x = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_liquid_evaporation_l1293_129372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_tangent_line_equation_l1293_129317

/-- Given an ellipse and a point inside it, this theorem proves the equation of a specific line. -/
theorem ellipse_tangent_line_equation (x y : ℝ) :
  (x^2 / 9 + y^2 / 5 = 1) →  -- Ellipse equation
  (5 < 9 ∧ 2 < 5) →  -- Condition for (√5, √2) to be inside the ellipse
  ∃ (E F : ℝ × ℝ), -- Points E and F exist
    (E.1^2 / 9 + E.2^2 / 5 = 1) ∧  -- E is on the ellipse
    (F.1^2 / 9 + F.2^2 / 5 = 1) ∧  -- F is on the ellipse
    ((Real.sqrt 5 / 9) * E.1 + (Real.sqrt 2 / 5) * E.2 = 1) ∧  -- E is on the line
    ((Real.sqrt 5 / 9) * F.1 + (Real.sqrt 2 / 5) * F.2 = 1)    -- F is on the line
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_tangent_line_equation_l1293_129317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_to_cone_270_deg_l1293_129327

/-- Represents a circular sector -/
structure CircularSector where
  radius : ℝ
  angle : ℝ

/-- Represents a cone -/
structure Cone where
  baseRadius : ℝ
  slantHeight : ℝ

/-- Converts a circular sector to a cone -/
noncomputable def sectorToCone (s : CircularSector) : Cone :=
  { baseRadius := s.radius * s.angle / (2 * Real.pi),
    slantHeight := s.radius }

theorem sector_to_cone_270_deg (s : CircularSector) 
  (h1 : s.radius = 12)
  (h2 : s.angle = 3 * Real.pi / 2) :
  let c := sectorToCone s
  c.baseRadius = 9 ∧ c.slantHeight = 12 := by
  sorry

#check sector_to_cone_270_deg

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_to_cone_270_deg_l1293_129327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_four_equals_eight_l1293_129360

theorem power_of_four_equals_eight (x : ℝ) : (4 : ℝ)^x = 8 → x = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_four_equals_eight_l1293_129360
