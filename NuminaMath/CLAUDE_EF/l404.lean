import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_candy_purchase_l404_40412

theorem smallest_n_for_candy_purchase : ∃ n : ℕ, 
  (∀ m : ℕ, m < n → ¬(10 ∣ 24*m ∧ 18 ∣ 24*m ∧ 20 ∣ 24*m)) ∧
  (10 ∣ 24*n ∧ 18 ∣ 24*n ∧ 20 ∣ 24*n) ∧
  n = 15 := by
  -- The proof goes here
  sorry

#check smallest_n_for_candy_purchase

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_candy_purchase_l404_40412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_valid_set_size_l404_40402

/-- A set of positive integers not exceeding 2002 where the product of any two elements is not in the set -/
def ValidSet (S : Finset ℕ) : Prop :=
  ∀ x ∈ S, x > 0 ∧ x ≤ 2002 ∧ ∀ y ∈ S, x * y ∉ S

/-- The largest size of a valid set -/
def LargestValidSetSize : ℕ := 1958

theorem largest_valid_set_size :
  (∃ S : Finset ℕ, ValidSet S ∧ S.card = LargestValidSetSize) ∧
  (∀ S : Finset ℕ, ValidSet S → S.card ≤ LargestValidSetSize) := by
  sorry

#check largest_valid_set_size

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_valid_set_size_l404_40402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_rational_l404_40491

/-- A point in 2D space -/
structure Point where
  x : ℤ
  y : ℚ

/-- Represents a triangle with three vertices -/
structure Triangle where
  v1 : Point
  v2 : Point
  v3 : Point

/-- Checks if a point has integer coordinates -/
def Point.isInteger (p : Point) : Prop := ∃ (n : ℤ), p.y = n

/-- Checks if a point has a half-integer y-coordinate -/
def Point.isHalfInteger (p : Point) : Prop := ∃ (n : ℤ), p.y = (n : ℚ) / 2

/-- Area of a triangle given its three vertices -/
def Triangle.area (t : Triangle) : ℚ :=
  let x1 := t.v1.x
  let y1 := t.v1.y
  let x2 := t.v2.x
  let y2 := t.v2.y
  let x3 := t.v3.x
  let y3 := t.v3.y
  (1 / 2 : ℚ) * abs ((x1 * y2 - x1 * y3 + x2 * y3 - x2 * y1 + x3 * y1 - x3 * y2) : ℚ)

/-- The main theorem: Area of triangle with given conditions is rational -/
theorem triangle_area_rational (t : Triangle) 
  (h1 : t.v1.isInteger)
  (h2 : t.v2.isInteger)
  (h3 : t.v3.isInteger ∨ t.v3.isHalfInteger) :
  ∃ (q : ℚ), t.area = q :=
by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_rational_l404_40491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l404_40485

def A : Set ℝ := {x | 2 ≤ x ∧ x < 4}
def B : Set ℝ := {x | x ≥ 3}

theorem set_operations :
  (A ∪ B = {x | x ≥ 2}) ∧
  (A ∩ B = {x | 3 ≤ x ∧ x < 4}) ∧
  (Set.univ \ A = {x | x < 2 ∨ x ≥ 4}) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l404_40485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_convergence_l404_40408

/-- An equilateral triangle with perpendiculars drawn as described in the problem -/
structure TriangleWithPerpendiculars where
  -- The equilateral triangle
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  is_equilateral : ((A.1 - B.1)^2 + (A.2 - B.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2) ∧
                   ((B.1 - C.1)^2 + (B.2 - C.2)^2 = (C.1 - A.1)^2 + (C.2 - A.2)^2)
  -- The sequence of points on AB
  P : ℕ → ℝ × ℝ
  -- P₁ is on AB
  P_on_AB : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P 1 = (t * A.1 + (1 - t) * B.1, t * A.2 + (1 - t) * B.2)
  -- The perpendicular property (simplified for the statement)
  perpendicular : ∀ n : ℕ, 
    ((P (n+1)).1 - (P n).1) * (B.1 - C.1) + ((P (n+1)).2 - (P n).2) * (B.2 - C.2) = 0 ∧
    ((P (n+2)).1 - (P (n+1)).1) * (C.1 - A.1) + ((P (n+2)).2 - (P (n+1)).2) * (C.2 - A.2) = 0 ∧
    ((P (n+3)).1 - (P (n+2)).1) * (A.1 - B.1) + ((P (n+3)).2 - (P (n+2)).2) * (A.2 - B.2) = 0

/-- The convergence point of the sequence Pₙ -/
noncomputable def convergence_point (t : TriangleWithPerpendiculars) : ℝ × ℝ :=
  ((2 * t.B.1 + t.A.1) / 3, (2 * t.B.2 + t.A.2) / 3)

/-- The main theorem stating that Pₙ converges to the point that divides AB in 2:1 ratio from B to A -/
theorem perpendicular_convergence (t : TriangleWithPerpendiculars) :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, 
    ((t.P n).1 - (convergence_point t).1)^2 + ((t.P n).2 - (convergence_point t).2)^2 < ε^2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_convergence_l404_40408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_isosceles_l404_40439

theorem triangle_isosceles (A B C : ℝ) (a b c : ℝ) :
  (A + B + C = π) →
  (Real.sin A / a = Real.sin B / b) →
  (Real.sin A / a = Real.sin C / c) →
  (b * Real.cos B - a * Real.cos A = 0) →
  (a = b ∨ b = c ∨ a = c) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_isosceles_l404_40439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_figure_b_length_l404_40451

/-- The total length of visible segments in Figure B -/
noncomputable def total_length (a b c d : ℝ) : ℝ :=
  a + b + (a - c) + (b - d) + Real.sqrt ((a - c)^2 + (b - d)^2)

/-- Theorem stating the total length of visible segments in Figure B -/
theorem figure_b_length : 
  ∀ (a b c d : ℝ),
  a = 10 → b = 7 → c = 3 → d = 2 →
  total_length a b c d = 29 + Real.sqrt 74 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_figure_b_length_l404_40451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_l404_40476

/-- The speed of a train given its length and time to cross a fixed point -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 140) (h2 : time = 4.666293363197611) :
  ∃ (speed : ℝ), abs ((length / 1000) / (time / 3600) - speed) < 0.5 := by
  -- We use ∃ and abs to approximate the result, as Lean doesn't have a built-in "≈" operator for reals
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_l404_40476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bounded_iff_a0_condition_l404_40482

noncomputable def a : ℕ → ℝ → ℝ
  | 0, x => x
  | n + 1, x => (a n x)^2 - 1 / (2^(2020 * 2^n) - 1)

theorem bounded_iff_a0_condition (x : ℝ) :
  (∃ M : ℝ, ∀ n : ℕ, |a n x| ≤ M) ↔ x ≤ 1 + 1 / 2^2020 := by
  sorry

#check bounded_iff_a0_condition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bounded_iff_a0_condition_l404_40482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_b_term_l404_40490

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

noncomputable def sum_of_arithmetic_sequence (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (a 1 + a n) / 2

theorem arithmetic_sequence_b_term 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_a2 : a 2 = 8) 
  (h_sum : sum_of_arithmetic_sequence a 10 = 185) :
  ∃ b : ℕ → ℝ, ∀ n : ℕ, b n = 3 * n + 1 + 2 ∧ b n = a (3 * n) := by
  sorry

#check arithmetic_sequence_b_term

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_b_term_l404_40490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l404_40423

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (Real.log (1 + x)) / x

-- Theorem statement
theorem problem_statement :
  -- Part I: f is decreasing on (0, +∞)
  (∀ x y, 0 < x ∧ x < y → f y < f x) ∧
  -- Part II: ln(1+x) < ax for all x > 0 iff a ≥ 1
  (∀ a : ℝ, (∀ x, 0 < x → Real.log (1 + x) < a * x) ↔ 1 ≤ a) ∧
  -- Part III: (1 + 1/n)^n < e for all n ∈ ℕ*
  (∀ n : ℕ, 0 < n → (1 + 1 / n : ℝ)^n < Real.exp 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l404_40423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_betting_scenario_final_amount_l404_40435

noncomputable def initial_amount : ℝ := 120
def num_bets : ℕ := 8
def num_wins : ℕ := 4
def num_losses : ℕ := 4
noncomputable def win_multiplier : ℝ := 3/2
noncomputable def loss_multiplier : ℝ := 2/3

theorem betting_scenario_final_amount :
  let final_amount := initial_amount * (win_multiplier ^ num_wins) * (loss_multiplier ^ num_losses)
  final_amount = initial_amount :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_betting_scenario_final_amount_l404_40435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_sum_implies_non_diagonal_sum_l404_40418

def Table := Fin 3 → Fin 3 → Fin 9

def is_valid_table (t : Table) : Prop :=
  (∀ i j, t i j ≠ 0) ∧
  (∀ i j k l, (i ≠ k ∨ j ≠ l) → t i j ≠ t k l)

def diagonal_sum (t : Table) (d : Bool) : Nat :=
  (Finset.sum (Finset.univ : Finset (Fin 3)) fun i => (t i (if d then i else 2 - i)).val + 1)

def non_diagonal_sum (t : Table) : Nat :=
  (Finset.sum (Finset.univ : Finset (Fin 3)) fun i =>
    Finset.sum (Finset.univ : Finset (Fin 3)) fun j =>
      if i = j ∨ i.val + j.val = 2 then 0 else (t i j).val + 1)

theorem diagonal_sum_implies_non_diagonal_sum (t : Table) :
  is_valid_table t →
  diagonal_sum t true = 7 →
  diagonal_sum t false = 21 →
  non_diagonal_sum t = 25 :=
sorry

#check diagonal_sum_implies_non_diagonal_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_sum_implies_non_diagonal_sum_l404_40418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_circumradius_inequality_l404_40437

/-- Given a triangle with side lengths a, b, c, and circumradius R, 
    the sum of squares of the sides is less than or equal to 9 times the square of the circumradius. -/
theorem triangle_side_circumradius_inequality 
  (a b c R : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ R > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_circumradius : R = (a * b * c) / (4 * Real.sqrt ((a + b + c) * (b + c - a) * (c + a - b) * (a + b - c) / 16))) :
  a^2 + b^2 + c^2 ≤ 9 * R^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_circumradius_inequality_l404_40437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_satisfying_polynomial_l404_40444

/-- A polynomial that satisfies the given equation for all real numbers -/
def satisfying_polynomial (q : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, q (x^3) - q (x^3 - 3) = (q x)^2 + 18

/-- The specific polynomial q(x) = 9x^3 - 9 -/
def q : ℝ → ℝ := λ x ↦ 9 * x^3 - 9

/-- Theorem stating that q(x) = 9x^3 - 9 is the unique polynomial satisfying the equation -/
theorem unique_satisfying_polynomial :
  satisfying_polynomial q ∧ 
  (∀ p : ℝ → ℝ, satisfying_polynomial p → p = q) := by
  sorry

#check unique_satisfying_polynomial

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_satisfying_polynomial_l404_40444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unpainted_area_of_cube_l404_40478

-- Define the cube's edge length
def cube_edge : ℝ := 12

-- Define the area of painted surface
def painted_area : ℝ := 600

-- Define the number of unpainted faces
def unpainted_faces : ℕ := 2

-- Theorem statement
theorem unpainted_area_of_cube : 
  let total_surface_area := 6 * cube_edge ^ 2
  let painted_faces := painted_area / cube_edge ^ 2
  ⌊painted_faces⌋ = 4 →
  unpainted_faces * cube_edge ^ 2 = 288 := by
  intro h
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unpainted_area_of_cube_l404_40478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_y_l404_40484

open Real

-- Define the function
noncomputable def y (x : ℝ) : ℝ := x^2 / (x + 3)

-- State the theorem
theorem derivative_of_y (x : ℝ) (h : x ≠ -3) : 
  deriv y x = (x^2 + 6*x) / (x + 3)^2 := by
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_y_l404_40484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_and_inequality_l404_40466

/-- A function f(x) with specific properties -/
noncomputable def f (x : ℝ) := 2 * Real.sin (2 * x - Real.pi / 6)

/-- The theorem statement -/
theorem function_properties_and_inequality :
  (∀ x, f (x + Real.pi / 3) = f (Real.pi / 3 - x)) ∧  -- Symmetry about x = π/3
  (f (Real.pi / 3) = 2) ∧                             -- f(π/3) = 2
  (f (-Real.pi / 6) = -2) ∧                           -- f(-π/6) = -2
  (∃ a : ℝ, a ≥ -5/2 ∧                                -- Lower bound for a
    ∃ x : ℝ, x ∈ Set.Icc (3*Real.pi/4) (7*Real.pi/6) ∧ 
      f x ≤ 2*a + 3) ∧                                -- Inequality condition
  (∀ a : ℝ, a < -5/2 →                                -- Upper bound for a
    ∀ x : ℝ, x ∈ Set.Icc (3*Real.pi/4) (7*Real.pi/6) → 
      f x > 2*a + 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_and_inequality_l404_40466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_slope_l404_40454

/-- Given a parabola y² = 2px (p > 0), prove that if a point M on the parabola
    has a distance of 3p from the focus F, then the slope of line MF is ± √5/2 -/
theorem parabola_focus_slope (p : ℝ) (hp : p > 0) :
  ∀ (x y : ℝ),
  y^2 = 2*p*x →
  (x - p/2)^2 + y^2 = (3*p)^2 →
  (abs (y / (x - p/2)) = Real.sqrt 5 / 2) :=
by
  intros x y h1 h2
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_slope_l404_40454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_grid_solution_l404_40470

/-- A structure representing the grid with variables A, B, and C --/
structure Grid where
  A : Nat
  B : Nat
  C : Nat

/-- The theorem statement --/
theorem unique_grid_solution :
  ∃! g : Grid,
    g.A ≤ 9 ∧ g.B ≤ 9 ∧ g.C ≤ 9 ∧
    (4 + g.A + 1 + g.B = g.A + 2 + g.C) ∧
    (g.A + 2 + g.C = 1 + 2 + 6) ∧
    (4 + 3 = g.A + 1) ∧
    (g.A + 1 = 1 + 6) ∧
    (1 + 6 = g.B + 2 + g.C) ∧
    g.A + g.B + g.C = 4 + 3 ∧
    g.B + 2 + g.C = 9 ∧
    4 + 1 + g.B = 12 :=
by sorry

#eval "The theorem has been stated and the proof is left as 'sorry'."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_grid_solution_l404_40470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_two_even_is_one_third_l404_40413

/-- The set of numbers from which we select -/
def S : Finset ℕ := {1, 2, 3, 4, 5, 6}

/-- A number is even if it's divisible by 2 -/
def isEven (n : ℕ) : Prop := n % 2 = 0

/-- Instance to make isEven decidable -/
instance (n : ℕ) : Decidable (isEven n) := 
  show Decidable (n % 2 = 0) from inferInstance

/-- The probability of selecting two even numbers from S -/
def probTwoEven : ℚ :=
  (Finset.filter isEven S).card.choose 2 / S.card.choose 2

/-- Theorem stating that the probability of selecting two even numbers is 1/3 -/
theorem prob_two_even_is_one_third : probTwoEven = 1 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_two_even_is_one_third_l404_40413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cf_length_l404_40497

noncomputable def shorterLeg60 (h : ℝ) : ℝ := h / 2

noncomputable def leg45 (h : ℝ) : ℝ := h / Real.sqrt 2

theorem cf_length (AF BF CF DF : ℝ) : 
  AF = 48 → 
  BF = shorterLeg60 AF → 
  CF = shorterLeg60 BF → 
  DF = leg45 (CF + DF) → 
  CF = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cf_length_l404_40497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l404_40411

noncomputable def f (a b x : ℝ) := a^x + b

theorem function_properties :
  ∀ (a b : ℝ),
    a > 0 →
    a ≠ 1 →
    f a b 0 = -2 →
    f a b 2 = 0 →
    (a = Real.sqrt 3 ∧ b = -3) ∧
    (∀ x, x ∈ Set.Icc (-2) 4 →
      f a b x ≤ 6 ∧
      f a b x ≥ -8/3 ∧
      (∃ x₁ x₂, x₁ ∈ Set.Icc (-2) 4 ∧ x₂ ∈ Set.Icc (-2) 4 ∧ f a b x₁ = 6 ∧ f a b x₂ = -8/3)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l404_40411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_satisfying_function_inequality_l404_40403

/-- A function satisfying the given conditions -/
def SatisfyingFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, DifferentiableAt ℝ f x) ∧ 
  (∀ x, f x + deriv f x > 1) ∧ 
  (f 0 = 4)

/-- The main theorem -/
theorem satisfying_function_inequality 
  (f : ℝ → ℝ) (hf : SatisfyingFunction f) :
  ∀ x > 0, f x > 3 / Real.exp x + 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_satisfying_function_inequality_l404_40403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_specific_point_to_line_l404_40467

/-- The distance from a point in polar coordinates to a line in polar form -/
noncomputable def distance_point_to_line (r : ℝ) (θ : ℝ) (a b c : ℝ) : ℝ :=
  |a * (r * Real.cos θ) + b * (r * Real.sin θ) + c| / Real.sqrt (a^2 + b^2)

/-- The specific problem statement -/
theorem distance_specific_point_to_line :
  distance_point_to_line (Real.sqrt 2) (Real.pi / 4) 1 (-1) (-1) = Real.sqrt 2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_specific_point_to_line_l404_40467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l404_40443

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos : 0 < a ∧ 0 < b

/-- Represents a circle with center M and radius r -/
structure Circle where
  M : ℝ × ℝ
  r : ℝ

/-- Represents a point on a hyperbola -/
def PointOnHyperbola (H : Hyperbola) (p : ℝ × ℝ) : Prop :=
  (p.1^2 / H.a^2) - (p.2^2 / H.b^2) = 1

/-- Represents a circle tangent to x-axis at a focus of the hyperbola -/
def CircleTangentAtFocus (H : Hyperbola) (C : Circle) : Prop :=
  ∃ (c : ℝ), C.M.1 = c ∧ C.M.2 = C.r ∧ c^2 = H.a^2 + H.b^2

/-- Represents a circle intersecting y-axis at two points -/
def CircleIntersectsYAxis (C : Circle) : Prop :=
  ∃ (p q : ℝ), p ≠ q ∧ C.r^2 = C.M.1^2 + p^2 ∧ C.r^2 = C.M.1^2 + q^2

/-- Represents an acute triangle formed by the center of the circle and its intersections with y-axis -/
def FormsAcuteTriangle (C : Circle) : Prop :=
  ∃ (p q : ℝ), p ≠ q ∧ C.r^2 = C.M.1^2 + p^2 ∧ C.r^2 = C.M.1^2 + q^2 ∧ 
    (C.M.1^2 < C.r^2) ∧ (C.M.1^2 > C.r^2 / 2)

/-- The eccentricity of a hyperbola -/
noncomputable def Eccentricity (H : Hyperbola) : ℝ :=
  Real.sqrt (1 + H.b^2 / H.a^2)

theorem hyperbola_eccentricity_range (H : Hyperbola) (C : Circle) :
  PointOnHyperbola H C.M →
  CircleTangentAtFocus H C →
  CircleIntersectsYAxis C →
  FormsAcuteTriangle C →
  (Real.sqrt 5 + 1) / 2 < Eccentricity H ∧ Eccentricity H < (Real.sqrt 6 + Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l404_40443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_temperature_theorem_l404_40492

noncomputable def fahrenheit_to_celsius (f : ℝ) : ℝ := (5 / 9) * (f - 32)

noncomputable def temperatures : List ℝ := [-12, -6, -8, -3, 7, fahrenheit_to_celsius 28, 0]

noncomputable def mean (lst : List ℝ) : ℝ := (lst.sum) / (lst.length : ℝ)

theorem mean_temperature_theorem :
  ∃ ε > 0, |mean temperatures - (-3.46)| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_temperature_theorem_l404_40492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_inequality_l404_40407

theorem trigonometric_inequality : 
  ∃ (a b c : ℝ),
    a = (1/2) * Real.cos (8 * π / 180) - (Real.sqrt 3 / 2) * Real.sin (8 * π / 180) ∧
    b = (2 * Real.tan (14 * π / 180)) / (1 - Real.tan (14 * π / 180) ^ 2) ∧
    c = Real.sqrt ((1 - Real.cos (48 * π / 180)) / 2) ∧
    a < c ∧ c < b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_inequality_l404_40407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_cardinality_divisible_by_q_l404_40415

theorem set_cardinality_divisible_by_q (p q : ℕ) (S : Finset ℕ) : 
  Prime p → Prime q → S ⊆ Finset.range p →
  q ∣ Finset.card (Finset.filter (fun x => x.toList.sum % p = 0) 
    (Finset.powersetCard q S)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_cardinality_divisible_by_q_l404_40415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_and_constant_product_l404_40464

noncomputable section

/-- Circle O with equation x^2 + y^2 = 4 -/
def circle_O : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 4}

/-- Line l1 with equation √3x + y - 2√3 = 0 -/
def line_l1 : Set (ℝ × ℝ) := {p | Real.sqrt 3 * p.1 + p.2 - 2 * Real.sqrt 3 = 0}

/-- Intersection points of circle O and line l1 -/
def intersection_points : Set (ℝ × ℝ) := circle_O ∩ line_l1

/-- Point A is in the first quadrant -/
def point_A : ℝ × ℝ := (1, Real.sqrt 3)

/-- Point B is the other intersection point -/
def point_B : ℝ × ℝ := sorry

/-- Moving point P on circle O -/
def point_P (x0 y0 : ℝ) : Prop := x0^2 + y0^2 = 4 ∧ x0 ≠ 1 ∧ x0 ≠ -1

/-- Symmetric point P1 of P with respect to the origin -/
def point_P1 (x0 y0 : ℝ) : ℝ × ℝ := (-x0, -y0)

/-- Symmetric point P2 of P with respect to the x-axis -/
def point_P2 (x0 y0 : ℝ) : ℝ × ℝ := (x0, -y0)

/-- y-coordinate of intersection of AP1 with y-axis -/
noncomputable def m (x0 y0 : ℝ) : ℝ := (Real.sqrt 3 * x0 - y0) / (1 + x0)

/-- y-coordinate of intersection of AP2 with y-axis -/
noncomputable def n (x0 y0 : ℝ) : ℝ := (-Real.sqrt 3 * x0 - y0) / (1 - x0)

theorem circle_line_intersection_and_constant_product :
  (∀ x0 y0, point_P x0 y0 → m x0 y0 * n x0 y0 = 4) ∧
  Real.sqrt ((point_A.1 - point_B.1)^2 + (point_A.2 - point_B.2)^2) = 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_and_constant_product_l404_40464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_l404_40446

def a : ℕ → ℚ
  | 0 => 3
  | n + 1 => (3 * a n - 4) / (a n - 1)

theorem a_formula (n : ℕ+) : a n = (2 * n.val + 1) / n.val := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_l404_40446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_square_with_odd_tens_digit_l404_40469

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_square_with_odd_tens_digit (a : ℕ) :
  tens_digit (a^2) ∈ ({1, 3, 5, 7, 9} : Set ℕ) → units_digit a ∈ ({4, 6} : Set ℕ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_square_with_odd_tens_digit_l404_40469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_product_sets_l404_40427

/-- A set of eight numbers where each number is the product of two others in the set -/
def ProductSet : Type := { s : Finset ℝ // s.card = 8 ∧ ∀ x ∈ s, ∃ y z, y ∈ s ∧ z ∈ s ∧ x = y * z }

/-- There exist infinitely many different product sets -/
theorem infinitely_many_product_sets : ∀ n : ℕ, ∃ (sets : Finset ProductSet), sets.card = n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_product_sets_l404_40427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_largest_number_l404_40489

def digits : List ℕ := [1, 6, 8]

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧ (∀ d : ℕ, d ∈ digits → ((n.digits 10).count d = digits.count d))

def valid_numbers : List ℕ := [168, 186, 618, 681, 816, 861]

theorem third_largest_number :
  ∀ n : ℕ, n ∈ valid_numbers →
  (valid_numbers.filter (λ m => m > n)).length = 2 →
  n = 681 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_largest_number_l404_40489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_path_count_is_twenty_l404_40401

/-- Represents a point on the grid -/
structure Point where
  x : Nat
  y : Nat

/-- Represents a direction on the grid -/
inductive Direction
  | Right
  | Up

/-- Represents a segment on the grid -/
structure Segment where
  start : Point
  direction : Direction
  hasArrow : Bool

/-- Represents the grid -/
structure Grid where
  size : Nat
  segments : List Segment

/-- Calculates the number of distinct paths from A to B -/
def countPaths (grid : Grid) : Nat :=
  sorry

/-- The main theorem -/
theorem path_count_is_twenty :
  ∀ (grid : Grid),
    grid.size = 5 ∧
    (∃ (arrows : List Segment), arrows.length ≤ 3 ∧ arrows.all (λ s => s.hasArrow)) →
    countPaths grid = 20 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_path_count_is_twenty_l404_40401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chloromethane_reaction_result_l404_40468

/-- Represents a chemical reaction with given initial reactants and yield --/
structure ChemicalReaction where
  initialMethane : ℝ
  initialChlorine : ℝ
  yield : ℝ

/-- Calculates the moles of product formed in a chemical reaction --/
noncomputable def molesOfProduct (reaction : ChemicalReaction) : ℝ :=
  min reaction.initialMethane reaction.initialChlorine * reaction.yield

/-- Calculates the moles of unreacted reactant --/
noncomputable def molesUnreacted (initial : ℝ) (reacted : ℝ) : ℝ :=
  initial - reacted

/-- Theorem stating the results of the chemical reaction --/
theorem chloromethane_reaction_result (reaction : ChemicalReaction)
    (h1 : reaction.initialMethane = 3)
    (h2 : reaction.initialChlorine = 3)
    (h3 : reaction.yield = 0.8) :
    let productsFormed := molesOfProduct reaction
    let unreactedMethane := molesUnreacted reaction.initialMethane productsFormed
    let unreactedChlorine := molesUnreacted reaction.initialChlorine productsFormed
    productsFormed = 2.4 ∧ unreactedMethane = 0.6 ∧ unreactedChlorine = 0.6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chloromethane_reaction_result_l404_40468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_division_l404_40495

/-- A rectangle in the coordinate plane -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- A line in the coordinate plane -/
structure Line where
  start : ℝ × ℝ
  finish : ℝ × ℝ

/-- The area of a triangle given its base and height -/
noncomputable def triangleArea (base height : ℝ) : ℝ := (1/2) * base * height

/-- Theorem: A line from (c,0) to (4,2) divides a 2x3 rectangle of unit squares 
    into two equal areas if and only if c = 1 -/
theorem equal_area_division (rect : Rectangle) (line : Line) (c : ℝ) :
  rect.width = 3 ∧ rect.height = 2 ∧
  line.start = (c, 0) ∧
  line.finish = (4, 2) →
  (triangleArea (4 - c) 2 = rect.width * rect.height / 2) ↔
  c = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_division_l404_40495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_f_equals_2_08_l404_40486

noncomputable def f (n : ℕ+) : ℝ := Real.log (n.val ^ 2) / Real.log 1806

theorem sum_of_f_equals_2_08 : f 17 + f 19 + f 6 = 2.08 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_f_equals_2_08_l404_40486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_satisfies_conditions_iff_cos_or_piecewise_l404_40480

/-- A function that satisfies the given conditions -/
def SatisfiesConditions (f : ℝ → ℝ) : Prop :=
  (∀ x, f (x + 1) = -f x) ∧
  (∀ x y, x ∈ Set.Icc 0 1 → y ∈ Set.Icc 0 1 → x < y → f y < f x)

/-- The cosine function scaled by π -/
noncomputable def CosPi (x : ℝ) : ℝ := Real.cos (Real.pi * x)

/-- The piecewise quadratic function -/
noncomputable def PiecewiseQuadratic (x : ℝ) : ℝ :=
  let k := ⌊(x + 1) / 2⌋
  1 - (x - 2 * ↑k) ^ 2

theorem satisfies_conditions_iff_cos_or_piecewise (f : ℝ → ℝ) :
  SatisfiesConditions f ↔ (f = CosPi ∨ f = PiecewiseQuadratic) := by
  sorry

#check satisfies_conditions_iff_cos_or_piecewise

end NUMINAMATH_CALUDE_ERRORFEEDBACK_satisfies_conditions_iff_cos_or_piecewise_l404_40480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_product_bound_l404_40455

theorem sum_product_bound (a b c d : ℝ) (h : a + b + c + d = 0) :
  ∃ (S : Set ℝ), S = Set.Iic 0 ∧ ∀ x, x ∈ S ↔ ∃ (a b c d : ℝ), a + b + c + d = 0 ∧ a * b + a * c + a * d + b * c + b * d + c * d = x :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_product_bound_l404_40455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_tan_y_inequality_l404_40420

open Real

theorem x_tan_y_inequality (x y : ℝ) (hx : x ∈ Set.Ioo 0 (π/6)) (hy : y ∈ Set.Ioo 0 (π/6))
  (h : x * tan y = 2 * (1 - cos x)) : y < x/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_tan_y_inequality_l404_40420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_lower_bound_l404_40452

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 1  -- Add a case for 0 to cover all natural numbers
  | 1 => 1
  | n + 2 => sequence_a (n + 1) / (sequence_a (n + 1) ^ 3 + 1)

theorem sequence_a_lower_bound (n : ℕ) (h : n ≥ 1) : 
  sequence_a n > 1 / Real.rpow (3 * n + Real.log (n : ℝ) + 14/9) (1/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_lower_bound_l404_40452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l404_40429

-- Define the ellipse
def ellipse (b : ℝ) := {p : ℝ × ℝ | p.1^2 / 4 + p.2^2 / b = 1}

-- Define the foci
noncomputable def left_focus (b : ℝ) : ℝ × ℝ := (-Real.sqrt (4 - b), 0)
noncomputable def right_focus (b : ℝ) : ℝ × ℝ := (Real.sqrt (4 - b), 0)

-- Define the dot product of vectors PF₁ and PF₂
def dot_product (p : ℝ × ℝ) (b : ℝ) : ℝ :=
  (p.1^2 + p.2^2) - (4 - b)

-- Main theorem
theorem ellipse_properties (b : ℝ) (h₁ : b > 0) :
  (∀ p ∈ ellipse b, dot_product p b ≤ 1) ∧
  (∃ p ∈ ellipse b, dot_product p b = 1) →
  (b = 1 ∧
   ∀ (k : ℝ),
     let A := (k * (2 * k / (4 + k^2)) - 1, 2 * k / (4 + k^2))
     let A' := (k * (2 * k / (4 + k^2)) - 1, -2 * k / (4 + k^2))
     let B := (k * (-3 / (2 * k)) - 1, -3 / (2 * k))
     (A ∈ ellipse 1 ∧ A' ∈ ellipse 1 ∧ B ∈ ellipse 1) →
     ∃ (t : ℝ), (1 - t) * A'.1 + t * B.1 = -4 ∧ (1 - t) * A'.2 + t * B.2 = 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l404_40429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meat_per_meal_is_one_point_five_l404_40462

/-- Represents the restaurant scenario with given conditions -/
structure Restaurant where
  beef_pounds : ℚ
  pork_ratio : ℚ
  meal_price : ℚ
  total_revenue : ℚ

/-- Calculates the amount of meat used per meal -/
def meat_per_meal (r : Restaurant) : ℚ :=
  let total_meat := r.beef_pounds * (1 + r.pork_ratio)
  let num_meals := r.total_revenue / r.meal_price
  total_meat / num_meals

/-- Theorem stating that under the given conditions, the amount of meat per meal is 1.5 pounds -/
theorem meat_per_meal_is_one_point_five (r : Restaurant) 
    (h1 : r.beef_pounds = 20)
    (h2 : r.pork_ratio = 1/2)
    (h3 : r.meal_price = 20)
    (h4 : r.total_revenue = 400) : 
  meat_per_meal r = 3/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_meat_per_meal_is_one_point_five_l404_40462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_both_systematic_sampling_l404_40473

/-- Represents a sampling scenario -/
structure SamplingScenario where
  numbers : List Nat

/-- Represents the school's student distribution -/
structure School where
  totalStudents : Nat
  firstGrade : Nat
  secondGrade : Nat
  thirdGrade : Nat

/-- Checks if a sampling scenario could be a result of systematic sampling -/
def couldBeSystematicSampling (s : SamplingScenario) (school : School) : Prop :=
  ∃ (k : Nat), ∀ (i j : Nat), i < j → i < s.numbers.length → j < s.numbers.length →
    s.numbers[j]! - s.numbers[i]! = k * (j - i)

/-- The given school -/
def givenSchool : School :=
  { totalStudents := 270
  , firstGrade := 108
  , secondGrade := 81
  , thirdGrade := 81 }

/-- Scenario ② -/
def scenario2 : SamplingScenario :=
  { numbers := [5, 9, 100, 107, 111, 121, 180, 195, 200, 265] }

/-- Scenario ③ -/
def scenario3 : SamplingScenario :=
  { numbers := [11, 38, 65, 92, 119, 146, 173, 200, 227, 254] }

/-- Theorem stating that both scenarios ② and ③ cannot be systematic sampling -/
theorem not_both_systematic_sampling :
  ¬(couldBeSystematicSampling scenario2 givenSchool ∧
     couldBeSystematicSampling scenario3 givenSchool) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_both_systematic_sampling_l404_40473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_slope_OQ_l404_40463

/-- Parabola type representing y² = 2px -/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Function to get the focus of a parabola -/
noncomputable def focus (e : Parabola) : Point :=
  { x := e.p / 2, y := 0 }

/-- Function to check if a point is on the parabola -/
def on_parabola (e : Parabola) (p : Point) : Prop :=
  p.y^2 = 2 * e.p * p.x

/-- Function to check if a point is in the first quadrant -/
def in_first_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y > 0

/-- Function to check if Q is on line segment PF -/
def on_line_segment (p q f : Point) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧
    q.x = t * p.x + (1 - t) * f.x ∧
    q.y = t * p.y + (1 - t) * f.y

/-- Theorem stating the maximum slope of OQ -/
theorem max_slope_OQ (e : Parabola) (p q : Point) (hq : on_line_segment p q (focus e))
    (hp : on_parabola e p) (hpq : in_first_quadrant p)
    (hv : q.x = (2/3) * p.x + (1/3) * (focus e).x ∧ q.y = (2/3) * p.y + (1/3) * (focus e).y) :
  ∃ k : ℝ, (∀ q' : Point, on_line_segment p q' (focus e) →
    q'.x = (2/3) * p.x + (1/3) * (focus e).x ∧ q'.y = (2/3) * p.y + (1/3) * (focus e).y →
    (q'.y / q'.x) ≤ k) ∧ k = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_slope_OQ_l404_40463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_in_range_l404_40406

open BigOperators Real

def probability_mass_function (k : ℕ) (a : ℝ) : ℝ :=
  if k ∈ Finset.range 6 \ {0} then a * k else 0

theorem probability_in_range (a : ℝ) :
  (∑ k in Finset.range 6 \ {0}, probability_mass_function k a) = 1 →
  (∑ k in {2, 3}, probability_mass_function k a) = 1/5 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_in_range_l404_40406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_region_R_l404_40479

-- Define the region R
def R : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 ≤ p.2 ∧ p.2 ≤ p.1^2 ∧ 0 ≤ p.1 ∧ p.1 ≤ 1}

-- Define the volume of the solid of revolution
noncomputable def volumeOfRevolution (S : Set (ℝ × ℝ)) : ℝ :=
  ∫ x in Set.Icc 0 1, Real.pi * ((x - x^2) / Real.sqrt 2)^2

-- Theorem statement
theorem volume_of_region_R : volumeOfRevolution R = Real.pi / 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_region_R_l404_40479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_average_speed_l404_40421

noncomputable def average_speed (d1 d2 d3 d4 v1 v2 v3 v4 : ℝ) : ℝ :=
  let total_distance := d1 + d2 + d3 + d4
  let total_time := d1 / v1 + d2 / v2 + d3 / v3 + d4 / v4
  total_distance / total_time

theorem car_average_speed : 
  let d1 : ℝ := 1/4
  let d2 : ℝ := 1/3
  let d3 : ℝ := 1/5
  let d4 : ℝ := 1 - (d1 + d2 + d3)
  let v1 : ℝ := 60
  let v2 : ℝ := 24
  let v3 : ℝ := 72
  let v4 : ℝ := 48
  ∃ ε : ℝ, ε > 0 ∧ |average_speed d1 d2 d3 d4 v1 v2 v3 v4 - 39.47| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_average_speed_l404_40421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_beta_value_l404_40448

theorem sin_beta_value (α β : Real) (h1 : 0 < α ∧ α < Real.pi/2) (h2 : 0 < β ∧ β < Real.pi/2)
  (h3 : Real.sin α = 4/5) (h4 : Real.cos (α + β) = 5/13) : Real.sin β = 16/65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_beta_value_l404_40448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_group_at_least_72_l404_40430

theorem product_of_group_at_least_72 (groups : List (List Nat)) : 
  groups.length = 3 →
  (∀ g ∈ groups, g.Nodup ∧ (∀ n ∈ g, 1 ≤ n ∧ n ≤ 9)) →
  (groups.join.toFinset = Finset.range 9) →
  ∃ g ∈ groups, g.prod ≥ 72 := by
  sorry

#check product_of_group_at_least_72

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_group_at_least_72_l404_40430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_omega_value_max_omega_is_eleven_l404_40457

/-- The function f(x) = sin(ωx + φ) -/
noncomputable def f (ω φ x : ℝ) : ℝ := Real.sin (ω * x + φ)

/-- The theorem stating the maximum value of ω given the conditions -/
theorem max_omega_value (ω φ : ℝ) : 
  ω > 0 → 
  abs φ ≤ π / 2 →
  (∀ x, f ω φ (x - π / 4) = -f ω φ (-x + π / 4)) →
  (∀ x, f ω φ (π / 4 + x) = f ω φ (π / 4 - x)) →
  (∀ x ∈ Set.Ioo (π / 14) (13 * π / 84), 
    (∀ y ∈ Set.Ioo (π / 14) (13 * π / 84), x < y → f ω φ x < f ω φ y) ∨
    (∀ y ∈ Set.Ioo (π / 14) (13 * π / 84), x < y → f ω φ x > f ω φ y)) →
  ω ≤ 11 :=
by sorry

/-- The maximum value of ω is 11 -/
theorem max_omega_is_eleven : ∃ ω φ : ℝ, 
  ω > 0 ∧
  abs φ ≤ π / 2 ∧
  (∀ x, f ω φ (x - π / 4) = -f ω φ (-x + π / 4)) ∧
  (∀ x, f ω φ (π / 4 + x) = f ω φ (π / 4 - x)) ∧
  (∀ x ∈ Set.Ioo (π / 14) (13 * π / 84), 
    (∀ y ∈ Set.Ioo (π / 14) (13 * π / 84), x < y → f ω φ x < f ω φ y) ∨
    (∀ y ∈ Set.Ioo (π / 14) (13 * π / 84), x < y → f ω φ x > f ω φ y)) ∧
  ω = 11 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_omega_value_max_omega_is_eleven_l404_40457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_f_piecewise_l404_40436

-- Define the original piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ -3 ∧ x < 0 then -2 - x
  else if x ≥ 0 ∧ x < 2 then Real.sqrt (4 - (x - 2)^2) - 2
  else if x ≥ 2 ∧ x ≤ 3 then 2*(x - 2)
  else 0  -- undefined outside the given intervals

-- Define the absolute value of f
noncomputable def abs_f (x : ℝ) : ℝ := |f x|

-- State the theorem
theorem abs_f_piecewise (x : ℝ) :
  abs_f x = if x ≥ -3 ∧ x < 0 then 2 + x
            else if x ≥ 0 ∧ x < 2 then 2 - Real.sqrt (4 - (x - 2)^2)
            else if x ≥ 2 ∧ x ≤ 3 then 2*(x - 2)
            else 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_f_piecewise_l404_40436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_cardinality_bound_l404_40445

-- Define M as a finite set of natural numbers
def M (n : ℕ) := Finset (Fin n)

-- Define the condition that M contains distinct positive integers
def distinct_positive (M : Finset ℕ) : Prop :=
  ∀ x y, x ∈ M → y ∈ M → x ≠ y → x > 0 ∧ y > 0

-- Define set S
def S (M : Finset ℕ) : Finset (ℕ × ℕ) :=
  M.product M |>.filter (fun p => (p.1 - p.2) ∈ M)

-- Theorem statement
theorem S_cardinality_bound {n : ℕ} (hn : n ≥ 2) (M : Finset ℕ) 
  (hM : M.card = n) (hd : distinct_positive M) : 
  (S M).card ≤ n * (n - 1) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_cardinality_bound_l404_40445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ones_divisibility_l404_40456

theorem ones_divisibility (n : ℤ) (h1 : ¬ 2 ∣ n) (h2 : ¬ 5 ∣ n) :
  ∃ k : ℕ, n ∣ ((10 : ℤ)^k - 1) / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ones_divisibility_l404_40456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_property_f_minimal_l404_40472

/-- The smallest f(n) such that any subset of {1, 2, 3, ..., n} with f(n) elements
    contains three pairwise coprime elements -/
def f (n : ℕ) : ℕ :=
  let m := n / 6
  match n % 6 with
  | 0 | 1 => 4 * m + 1
  | 2     => 4 * m + 2
  | 3     => 4 * m + 3
  | 4 | 5 => 4 * m + 4
  | _     => 0  -- This case should never occur

theorem f_property (n : ℕ) (h : n > 2) :
  ∀ (S : Finset ℕ), S ⊆ Finset.range n → S.card = f n →
    ∃ (a b c : ℕ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧
      Nat.Coprime a b ∧ Nat.Coprime b c ∧ Nat.Coprime a c :=
by sorry

theorem f_minimal (n : ℕ) (h : n > 2) :
  ∀ (g : ℕ → ℕ), (∀ (S : Finset ℕ), S ⊆ Finset.range n → S.card = g n →
    ∃ (a b c : ℕ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧
      Nat.Coprime a b ∧ Nat.Coprime b c ∧ Nat.Coprime a c) →
  g n ≥ f n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_property_f_minimal_l404_40472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_point_range_l404_40425

open Real

noncomputable def e : ℝ := Real.exp 1

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x
noncomputable def g (x : ℝ) : ℝ := exp x

theorem symmetric_point_range (a : ℝ) :
  (∃ x ∈ Set.Icc (1/e) e, f a x = log x) →
  a ∈ Set.Icc 1 (e + 1/e) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_point_range_l404_40425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_when_no_intersection_l404_40432

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- Represents a circle with center (h, k) and radius r -/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ
  h_pos_r : r > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- Condition that asymptotes of hyperbola do not intersect with circle -/
def asymptotes_do_not_intersect (h : Hyperbola) : Prop :=
  4 * h.b / Real.sqrt (h.a^2 + h.b^2) > 2 * Real.sqrt 2

/-- Main theorem: If asymptotes do not intersect, then eccentricity > √2 -/
theorem eccentricity_range_when_no_intersection 
  (h : Hyperbola) 
  (h_no_intersect : asymptotes_do_not_intersect h) : 
  eccentricity h > Real.sqrt 2 := by
  sorry

#check eccentricity_range_when_no_intersection

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_when_no_intersection_l404_40432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_decreasing_interval_l404_40494

open Real

noncomputable def f (x : ℝ) := cos x - sin x

theorem monotone_decreasing_interval (a b : ℝ) (h₁ : 0 ≤ a) (h₂ : b ≤ π) (h₃ : a < b) :
  (∀ x ∈ Set.Icc a b, StrictAntiOn f (Set.Icc a b)) ↔ a = 0 ∧ b = 3*π/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_decreasing_interval_l404_40494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_ratio_l404_40426

/-- The volume of a cone with radius r and height h is (1/3) * π * r^2 * h -/
noncomputable def coneVolume (r h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

theorem cone_volume_ratio :
  let rC : ℝ := 16.4
  let hC : ℝ := 30.5
  let rD : ℝ := 30.5
  let hD : ℝ := 16.4
  (coneVolume rC hC) / (coneVolume rD hD) = 164 / 305 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_ratio_l404_40426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_and_geometric_sequences_l404_40453

noncomputable def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

noncomputable def sum_arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  n * (2 * a₁ + (n - 1 : ℝ) * d) / 2

noncomputable def geometric_sequence (b₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := b₁ * q^(n - 1)

noncomputable def sum_geometric_sequence (b₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  b₁ * (1 - q^n) / (1 - q)

theorem arithmetic_and_geometric_sequences :
  ∀ (a₁ : ℝ) (S₃ : ℝ),
  a₁ = 1 ∧ S₃ = 0 →
  (∀ n : ℕ, arithmetic_sequence a₁ (-1) n = 2 - n) ∧
  (∀ n : ℕ, sum_geometric_sequence (2 * a₁) (-2) n = 2 * (1 + 2^n) / 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_and_geometric_sequences_l404_40453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_tickets_proof_l404_40460

/-- The price of a single movie ticket in dollars -/
def ticket_price : ℚ := 15.25

/-- Tom's budget for buying movie tickets in dollars -/
def budget : ℚ := 200

/-- The maximum number of tickets that can be purchased given the ticket price and budget -/
def max_tickets : ℕ := (budget / ticket_price).floor.toNat

theorem max_tickets_proof : max_tickets = 13 := by
  -- Unfold the definition of max_tickets
  unfold max_tickets
  -- Evaluate the expression
  norm_num
  -- QED
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_tickets_proof_l404_40460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slopes_product_l404_40471

open Real

-- Define the slopes and angles of the lines
noncomputable def m : ℝ := 1
noncomputable def n : ℝ := 2 - sqrt 3
noncomputable def θ₁ : ℝ := π / 4
noncomputable def θ₂ : ℝ := π / 12

-- State the theorem
theorem line_slopes_product (h1 : θ₁ = 3 * θ₂) (h2 : m = 3 * n) 
  (h3 : m = tan θ₁) (h4 : n = tan θ₂) : m * n = 2 - sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slopes_product_l404_40471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l404_40410

-- Define the right triangle
noncomputable def right_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2

-- Define the area of a triangle
noncomputable def triangle_area (base height : ℝ) : ℝ :=
  (1/2) * base * height

-- Theorem statement
theorem right_triangle_area :
  ∀ a b c : ℝ,
  right_triangle a b c →
  c = 13 →
  a = 5 →
  ∃ h : ℝ, triangle_area a h = 30 :=
by
  sorry

#check right_triangle_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l404_40410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sluice_closing_time_l404_40404

/-- Represents the time of day -/
inductive TimeOfDay
| AM : Nat → TimeOfDay
| PM : Nat → TimeOfDay

/-- Represents a date with time -/
structure DateTime where
  month : Nat
  day : Nat
  time : TimeOfDay

def inflow_rate : ℝ := 200000
def daytime_evaporation_rate : ℝ := 1000
def nighttime_evaporation_rate : ℝ := 250
def initial_storage : ℝ := 4000000
def discharge_rate : ℝ := 230000
def target_storage : ℝ := 120000

def start_date : DateTime := ⟨8, 8, TimeOfDay.PM 12⟩

/-- Calculates the water storage after a given number of hours -/
noncomputable def water_storage_after_hours (hours : ℝ) : ℝ :=
  initial_storage + (inflow_rate - discharge_rate) * hours - 
  (daytime_evaporation_rate * (hours / 2) + nighttime_evaporation_rate * (hours / 2))

/-- The theorem to be proved -/
theorem sluice_closing_time : 
  ∃ (h : ℝ), water_storage_after_hours h = target_storage ∧ 
  h = (4 * 24 + 11 : ℝ) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sluice_closing_time_l404_40404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_and_perpendicular_vectors_l404_40433

noncomputable def a (x : ℝ) : ℝ × ℝ := (1, Real.cos x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (1/3, Real.sin x)

theorem parallel_and_perpendicular_vectors (x : ℝ) (h : x ∈ Set.Ioo 0 Real.pi) :
  (∃ k : ℝ, a x = k • b x → (Real.sin x + Real.cos x) / (Real.sin x - Real.cos x) = -2) ∧
  (a x • b x = 0 → Real.sin x - Real.cos x = Real.sqrt 15 / 3) := by
  sorry

#check parallel_and_perpendicular_vectors

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_and_perpendicular_vectors_l404_40433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_inscribed_square_side_length_is_correct_l404_40442

/-- The side length of the largest inscribed square given a 12x12 square with two inscribed equilateral triangles -/
noncomputable def largest_inscribed_square_side_length : ℝ :=
  6 * Real.sqrt 2 - 6

/-- Theorem stating the side length of the largest inscribed square -/
theorem largest_inscribed_square_side_length_is_correct (large_square_side : ℝ)
  (triangle_side : ℝ) (inscribed_square_side : ℝ)
  (h1 : large_square_side = 12)
  (h2 : triangle_side = 6 * Real.sqrt 2)
  (h3 : inscribed_square_side = largest_inscribed_square_side_length) :
  inscribed_square_side = 6 * Real.sqrt 2 - 6 :=
by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval largest_inscribed_square_side_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_inscribed_square_side_length_is_correct_l404_40442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_face_area_ratio_l404_40424

theorem clock_face_area_ratio : ∀ (r : ℝ), r > 0 →
  let t := (r^2 * Real.sqrt 3) / 2;
  let q := π * r^2 / 3 - r^2 * Real.sqrt 3
  q / t = 2 * Real.sqrt 3 - 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_face_area_ratio_l404_40424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_k_range_l404_40440

/-- The range of k values for which the circle x^2 + y^2 + 2kx + 4y + 3k + 8 = 0 
    passes through (-1, 0) and allows two tangents to be drawn -/
theorem circle_k_range : 
  let circle_equation (x y k : ℝ) := x^2 + y^2 + 2*k*x + 4*y + 3*k + 8 = 0
  ∃ (S : Set ℝ), S = {k : ℝ | 
    (circle_equation (-1) 0 k) ∧ 
    (∃ (x y : ℝ), circle_equation x y k) ∧
    (∃ (l₁ l₂ : ℝ → ℝ), l₁ ≠ l₂ ∧ 
      (∃ (x₁ y₁ x₂ y₂ : ℝ), 
        circle_equation x₁ y₁ k ∧ 
        circle_equation x₂ y₂ k ∧
        (∀ t, l₁ t = y₁ + (t - x₁) * ((l₁ x₁ - l₁ (x₁ - 1)) / 1)) ∧
        (∀ t, l₂ t = y₂ + (t - x₂) * ((l₂ x₂ - l₂ (x₂ - 1)) / 1)) ∧
        (∀ x y, circle_equation x y k → (y - l₁ x) * (y - l₂ x) ≥ 0)))} ∧
  S = {k : ℝ | -9 < k ∧ k < -1 ∨ k > 4} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_k_range_l404_40440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_y_intercept_l404_40496

/-- Circle with equation x^2 + (y-1)^2 = 1 -/
def Circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + (p.2 - 1)^2 = 1}

/-- Tangent line to the circle -/
def TangentLine (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | b * p.1 + a * p.2 = a * b}

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Theorem: The y-intercept that minimizes AB is (3 + √5) / 2 -/
theorem min_distance_y_intercept :
  ∃ (a b : ℝ), a > 1 ∧ b > 2 ∧
  (∀ (x y : ℝ), (x, y) ∈ TangentLine a b → (x, y) ∉ Circle) ∧
  (distance (a, 0) (0, b) = Real.sqrt (a^2 + b^2)) ∧
  (∀ (a' b' : ℝ), a' > 1 → b' > 2 →
    (∀ (x y : ℝ), (x, y) ∈ TangentLine a' b' → (x, y) ∉ Circle) →
    distance (a', 0) (0, b') ≥ distance (a, 0) (0, b)) →
  b = (3 + Real.sqrt 5) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_y_intercept_l404_40496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_answer_is_correct_l404_40481

def correct_answer : String := "B"

theorem answer_is_correct : correct_answer = "B" := by
  rfl

#check answer_is_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_answer_is_correct_l404_40481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_x_production_rate_l404_40441

/-- The production rate of Machine X in widgets per hour -/
noncomputable def machine_x_rate : ℝ := 3

/-- The production rate of Machine Y in widgets per hour -/
noncomputable def machine_y_rate : ℝ := machine_x_rate * 1.2

/-- The time it takes Machine X to produce 1080 widgets -/
noncomputable def machine_x_time : ℝ := 1080 / machine_x_rate

/-- The time it takes Machine Y to produce 1080 widgets -/
noncomputable def machine_y_time : ℝ := 1080 / machine_y_rate

/-- Theorem stating that Machine X produces 3 widgets per hour -/
theorem machine_x_production_rate : 
  (machine_x_rate = 3) ∧ 
  (machine_x_time = machine_y_time + 60) ∧ 
  (machine_y_rate = machine_x_rate * 1.2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_x_production_rate_l404_40441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_bijective_l404_40428

def x : ℕ → ℕ
  | 0 => 1  -- Added case for 0
  | 1 => 1
  | n + 2 => 
    let k := (n + 1) / 2
    if (n + 2) % 2 = 0 then
      if k % 2 = 0 then 2 * x k else 2 * x k + 1
    else
      if k % 2 = 0 then 2 * x k + 1 else 2 * x k

theorem x_bijective : Function.Bijective x := by
  sorry

#check x_bijective

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_bijective_l404_40428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_usual_commute_time_is_ten_minutes_l404_40487

/-- Represents the worker's commute scenario -/
structure CommuteData where
  normal_speed : ℝ
  normal_time : ℝ
  normal_distance : ℝ
  reduced_speed_factor : ℝ
  increased_distance_factor : ℝ
  store_stop_time : ℝ
  roadblock_delay : ℝ
  total_delay : ℝ

/-- The theorem stating that the usual commute time is 10 minutes -/
theorem usual_commute_time_is_ten_minutes (c : CommuteData) : c.normal_time = 10 :=
  by
  -- Assumptions based on the problem conditions
  have h1 : c.reduced_speed_factor = 4/5 := by sorry
  have h2 : c.increased_distance_factor = 1.2 := by sorry
  have h3 : c.store_stop_time = 5 := by sorry
  have h4 : c.roadblock_delay = 10 := by sorry
  have h5 : c.total_delay = 20 := by sorry
  
  -- Relationship between speed, time, and distance
  have h6 : c.normal_distance = c.normal_speed * c.normal_time := by sorry
  
  -- New time calculation
  have h7 : c.normal_time * (c.increased_distance_factor / c.reduced_speed_factor) + 
            c.store_stop_time + c.roadblock_delay = c.normal_time + c.total_delay := by sorry
  
  -- Proof of the theorem
  sorry

#check usual_commute_time_is_ten_minutes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_usual_commute_time_is_ten_minutes_l404_40487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_water_consumption_l404_40405

def water_consumption_problem (initial_horses : ℕ) (added_horses : ℕ) (bathing_water : ℕ) (total_water : ℕ) (total_days : ℕ) : Prop :=
  let total_horses := initial_horses + added_horses
  let daily_water := total_water / total_days
  ∃ (drinking_water : ℕ),
    drinking_water = (daily_water / total_horses) - bathing_water

theorem solve_water_consumption :
  water_consumption_problem 3 5 2 1568 28 = true :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_water_consumption_l404_40405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l404_40431

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (4 - (x - 2)^2)

-- State the theorem
theorem function_properties :
  ∀ (x₁ x₂ : ℝ), 2 < x₁ → x₁ < x₂ → x₂ < 4 →
  (x₂ * f x₁ > x₁ * f x₂) ∧
  ((x₂ - x₁) * (f x₂ - f x₁) < 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l404_40431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_of_f_l404_40498

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log (6 + x - 2 * x^2) / Real.log (1/2)

-- State the theorem
theorem monotonic_increasing_interval_of_f :
  ∃ (a b : ℝ), a = 1/4 ∧ b = 2 ∧
  (∀ x, -3/2 < x ∧ x < 2 → f x ∈ Set.Icc a b) ∧
  (∀ x y, a ≤ x ∧ x < y ∧ y < b → f x < f y) := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_of_f_l404_40498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_comparison_of_powers_l404_40465

theorem comparison_of_powers : (4 : ℝ)^(0.6 : ℝ) > (8 : ℝ)^(0.34 : ℝ) ∧ (8 : ℝ)^(0.34 : ℝ) > (1/2 : ℝ)^(-(0.9 : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_comparison_of_powers_l404_40465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_bullseyes_correct_l404_40449

/-- Represents the archery tournament scenario -/
structure ArcheryTournament where
  total_shots : Nat
  lead_halfway : Nat
  chelsea_min_score : Nat
  bullseye_score : Nat

/-- Calculates the minimum number of bullseyes needed for guaranteed victory -/
def min_bullseyes_for_victory (tournament : ArcheryTournament) : Nat :=
  let remaining_shots := tournament.total_shots / 2
  let max_opponent_score := tournament.lead_halfway + remaining_shots * tournament.bullseye_score
  let n := (max_opponent_score - remaining_shots * tournament.chelsea_min_score) / 
            (tournament.bullseye_score - tournament.chelsea_min_score)
  n + 1  -- We add 1 to ensure the ceiling effect

/-- Theorem stating the minimum number of bullseyes needed for guaranteed victory -/
theorem min_bullseyes_correct (tournament : ArcheryTournament) 
  (h1 : tournament.total_shots = 120)
  (h2 : tournament.lead_halfway = 60)
  (h3 : tournament.chelsea_min_score = 5)
  (h4 : tournament.bullseye_score = 10) :
  min_bullseyes_for_victory tournament = 49 := by
  sorry

#eval min_bullseyes_for_victory { 
  total_shots := 120, 
  lead_halfway := 60, 
  chelsea_min_score := 5, 
  bullseye_score := 10 
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_bullseyes_correct_l404_40449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_makers_existence_l404_40400

/-- Two integers are square makers if their product plus one is a perfect square -/
def square_makers (a b : ℤ) : Prop := ∃ k : ℤ, a * b + 1 = k^2

/-- A function that checks if a set can be divided into pairs of square makers -/
def can_divide_into_square_makers (n : ℕ) : Prop :=
  ∃ f : Fin (2*n) → Fin n, ∀ i : Fin n,
    ∃ a b : Fin (2*n), a ≠ b ∧
      square_makers (a.val : ℤ) (b.val : ℤ) ∧
      (f⁻¹' {i} : Set (Fin (2*n))) = {a, b}

theorem square_makers_existence (n : ℕ) :
  can_divide_into_square_makers n ↔ Even n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_makers_existence_l404_40400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_side_length_ratio_l404_40419

def square_A_area : ℝ := 25
def square_B_area : ℝ := 81
def square_C_area : ℝ := 64

theorem side_length_ratio :
  let side_A := Real.sqrt square_A_area
  let side_B := Real.sqrt square_B_area
  let side_C := Real.sqrt square_C_area
  (side_A / 5 : ℝ) = (side_B / 9 : ℝ) ∧ (side_A / 5 : ℝ) = (side_C / 8 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_side_length_ratio_l404_40419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircles_intersection_l404_40422

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Represents a circle -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represents the circumcircle of a triangle -/
noncomputable def circumcircle (t : Triangle) : Circle := sorry

/-- Checks if a triangle is acute-angled -/
def is_acute_angled (t : Triangle) : Prop := sorry

/-- Checks if a point lies on the smaller arc of a circle between two other points -/
def on_smaller_arc (p : Point) (c : Circle) (a b : Point) : Prop := sorry

/-- Computes the orthocenter of a triangle -/
noncomputable def orthocenter (t : Triangle) : Point := sorry

/-- Checks if a point lies on a circle -/
def on_circle (p : Point) (c : Circle) : Prop := sorry

/-- Main theorem -/
theorem circumcircles_intersection
  (ABC : Triangle)
  (circle : Circle)
  (A1 B1 C1 : Point)
  (A2 B2 C2 : Point)
  (h1 : is_acute_angled ABC)
  (h2 : circle = circumcircle ABC)
  (h3 : on_smaller_arc A1 circle ABC.B ABC.C)
  (h4 : on_smaller_arc B1 circle ABC.A ABC.C)
  (h5 : on_smaller_arc C1 circle ABC.A ABC.B)
  (h6 : A2 = orthocenter ⟨ABC.B, A1, ABC.C⟩)
  (h7 : B2 = orthocenter ⟨ABC.A, B1, ABC.C⟩)
  (h8 : C2 = orthocenter ⟨ABC.A, C1, ABC.B⟩) :
  ∃ (M : Point),
    on_circle M (circumcircle ⟨ABC.B, A2, ABC.C⟩) ∧
    on_circle M (circumcircle ⟨ABC.A, B2, ABC.C⟩) ∧
    on_circle M (circumcircle ⟨ABC.A, C2, ABC.B⟩) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircles_intersection_l404_40422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_distance_l404_40447

-- Define the ellipse
def ellipse (x y b : ℝ) : Prop := x^2 / 36 + y^2 / b^2 = 1

-- Define the condition on b
def b_condition (b : ℝ) : Prop := 0 < b ∧ b < 6

-- Define a point on the ellipse
def point_on_ellipse (P : ℝ × ℝ) (b : ℝ) : Prop :=
  ellipse P.1 P.2 b ∧ (P.1 ≠ 6 ∧ P.1 ≠ -6) ∧ (P.2 ≠ b ∧ P.2 ≠ -b)

-- Define the left focus
noncomputable def left_focus (b : ℝ) : ℝ × ℝ := (-Real.sqrt (36 - b^2), 0)

-- Define the vector sum condition
def vector_sum_condition (P : ℝ × ℝ) (F : ℝ × ℝ) : Prop :=
  Real.sqrt ((P.1 + F.1)^2 + (P.2 + F.2)^2) = 7

-- Main theorem
theorem ellipse_focus_distance (b : ℝ) (P : ℝ × ℝ) :
  b_condition b →
  point_on_ellipse P b →
  vector_sum_condition P (left_focus b) →
  Real.sqrt ((P.1 - (left_focus b).1)^2 + (P.2 - (left_focus b).2)^2) = 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_distance_l404_40447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_statements_l404_40477

-- Define the basic concepts
variable (Point Line Plane : Type)
variable (parallel : Plane → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (in_plane : Point → Plane → Prop)

-- Statement ①
def statement1 (Point Line Plane : Type) (parallel_line_plane : Line → Plane → Prop) : Prop :=
  ∀ α : Plane, ∀ β : Line,
    (∃ S : Set Line, (∀ l ∈ S, parallel_line_plane l α) ∧ Set.Infinite S) →
    parallel_line_plane β α

-- Statement ②
def statement2 (Line Plane : Type) (parallel : Plane → Plane → Prop) (parallel_line_plane : Line → Plane → Prop) : Prop :=
  ∀ α β : Plane, ∀ l : Line,
    parallel_line_plane l α ∧ parallel_line_plane l β →
    parallel α β

-- Statement ③
def statement3 (Point Plane : Type) (parallel : Plane → Plane → Prop) (in_plane : Point → Plane → Prop) : Prop :=
  ∀ α : Plane, ∀ p q : Point,
    ¬(in_plane p α) ∧ ¬(in_plane q α) →
    ∃ β : Plane, parallel α β ∧ in_plane p β ∧ in_plane q β

-- Statement ④
def statement4 (Plane : Type) (parallel : Plane → Plane → Prop) : Prop :=
  ∀ α β γ : Plane,
    parallel α γ ∧ parallel β γ →
    parallel α β

-- Theorem stating which statements are correct
theorem parallel_statements (Point Line Plane : Type) 
                            (parallel : Plane → Plane → Prop)
                            (parallel_line_plane : Line → Plane → Prop)
                            (in_plane : Point → Plane → Prop) :
  ¬(statement1 Point Line Plane parallel_line_plane) ∧ 
  ¬(statement2 Line Plane parallel parallel_line_plane) ∧ 
  ¬(statement3 Point Plane parallel in_plane) ∧ 
  (statement4 Plane parallel) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_statements_l404_40477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l404_40488

/-- Represents a hyperbola with equation y²/a² - x²/b² = 1 -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- The distance from the center to a focus of a hyperbola -/
noncomputable def Hyperbola.c (h : Hyperbola) : ℝ := Real.sqrt (h.a^2 + h.b^2)

/-- The slope of the asymptotes of a hyperbola -/
noncomputable def Hyperbola.asymptoteSlope (h : Hyperbola) : ℝ := h.a / h.b

/-- Theorem stating the properties of the specific hyperbola -/
theorem hyperbola_equation (h : Hyperbola) :
  h.a = Real.sqrt 12 ∧ h.b = Real.sqrt 24 →
  h.c = 6 ∧ h.asymptoteSlope = Real.sqrt 2 / 2 :=
by
  intro h_eq
  sorry -- Proof skipped for now

#check hyperbola_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l404_40488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l404_40414

noncomputable def f (x : ℝ) : ℝ := |Real.sin x + Real.cos x|

theorem smallest_positive_period_of_f :
  ∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧
  (∀ (q : ℝ), q > 0 → (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  p = Real.pi :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l404_40414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_equilateral_condition_l404_40417

theorem triangle_equilateral_condition (A B C : ℝ) :
  A + B + C = π →
  0 < A ∧ 0 < B ∧ 0 < C →
  Real.cos (A - B) * Real.cos (B - C) * Real.cos (C - A) = 1 →
  A = B ∧ B = C := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_equilateral_condition_l404_40417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_tiling_doubles_l404_40438

/-- A tiling is "regular" if no sub-rectangle is tiled with corners. -/
def IsRegularTiling (m n : ℕ) (tiling : Set (ℕ × ℕ)) : Prop :=
  ∀ (x y w h : ℕ), x + w ≤ m → y + h ≤ n →
    ¬ (∀ (i j : ℕ), i < w → j < h → (x + i, y + j) ∈ tiling)

/-- The existence of a regular tiling for an m × n rectangle -/
def ExistsRegularTiling (m n : ℕ) : Prop :=
  ∃ (tiling : Set (ℕ × ℕ)), IsRegularTiling m n tiling

/-- If there exists a regular tiling for an m × n rectangle, 
    then there exists a regular tiling for a 2m × 2n rectangle -/
theorem regular_tiling_doubles {m n : ℕ} (h : ExistsRegularTiling m n) :
  ExistsRegularTiling (2 * m) (2 * n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_tiling_doubles_l404_40438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hydras_always_live_l404_40409

/-- Represents the number of new heads a hydra can grow in a week -/
inductive NewHeads
  | five : NewHeads
  | seven : NewHeads

/-- Represents the state of the hydras' heads -/
structure HydraState where
  total : Nat
  isOdd : Odd total

/-- Represents the weekly change in total heads -/
def weeklyChange (a b : NewHeads) : Nat :=
  match a, b with
  | NewHeads.five, NewHeads.five => 6
  | NewHeads.five, NewHeads.seven => 8
  | NewHeads.seven, NewHeads.five => 8
  | NewHeads.seven, NewHeads.seven => 10

/-- The initial state of the hydras -/
def initialState : HydraState where
  total := 4033
  isOdd := by sorry

/-- The state transition function -/
def nextState (state : HydraState) (a b : NewHeads) : HydraState where
  total := state.total + weeklyChange a b
  isOdd := by sorry

theorem hydras_always_live :
  ∀ (n : Nat) (growthSequence : Fin n → NewHeads × NewHeads),
    let finalState := (List.foldl (λ acc (pair : NewHeads × NewHeads) => nextState acc pair.1 pair.2) 
                                  initialState 
                                  (List.ofFn growthSequence))
    Odd finalState.total :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hydras_always_live_l404_40409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_properties_l404_40475

-- Define the general term of the expansion
def generalTerm (n : ℕ) (r : ℕ) (x : ℝ) : ℝ :=
  (-1)^r * 3^r * (Nat.choose n r) * x^((n - 2*r) / 3)

-- Define the condition for the sixth term being constant
def sixthTermConstant (n : ℕ) : Prop :=
  (n - 10) / 3 = 0

-- Define the coefficient of x² term
def coefficientX2 (n : ℕ) : ℕ :=
  9 * (Nat.choose n 2)

theorem expansion_properties :
  ∀ n : ℕ, sixthTermConstant n →
    (n = 10 ∧ coefficientX2 n = 405) := by
  intro n h
  sorry

#eval coefficientX2 10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_properties_l404_40475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_motorcycle_race_motorcycle_race_solution_l404_40434

/-- Motorcycle race problem -/
theorem motorcycle_race
  (speed_second : ℝ)
  (distance : ℝ)
  (h1 : speed_second > 0)
  (h2 : distance > 0)
  (h3 : distance / (speed_second + 15) = distance / speed_second - 1/5)
  (h4 : distance / speed_second = distance / (speed_second - 3) - 1/20) :
  speed_second = 75 ∧ distance = 90 := by
  sorry

/-- Speeds of all motorcycles -/
def speeds (speed_second : ℝ) : Fin 3 → ℝ :=
  fun i => match i with
    | 0 => speed_second + 15
    | 1 => speed_second
    | 2 => speed_second - 3

/-- Main theorem about the motorcycle race -/
theorem motorcycle_race_solution :
  ∃ (speed_second : ℝ) (distance : ℝ),
    speed_second > 0 ∧
    distance > 0 ∧
    distance / (speed_second + 15) = distance / speed_second - 1/5 ∧
    distance / speed_second = distance / (speed_second - 3) - 1/20 ∧
    speed_second = 75 ∧
    distance = 90 ∧
    (speeds speed_second 0 = 90 ∧
     speeds speed_second 1 = 75 ∧
     speeds speed_second 2 = 72) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_motorcycle_race_motorcycle_race_solution_l404_40434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_equality_difference_not_always_equal_specific_difference_result_l404_40450

-- Define two non-empty sets A and B
variable (A B : Set ℝ)

-- Define the set difference operation
def setDifference (X Y : Set ℝ) : Set ℝ := {x | x ∈ X ∧ x ∉ Y}

-- Define specific sets A' and B' as given in the problem
def A' : Set ℝ := {x : ℝ | x > 4}
def B' : Set ℝ := {x : ℝ | -6 < x ∧ x < 6}

-- Theorem to prove the equality of A' - (A' - B') and B' - (B' - A')
theorem difference_equality :
  setDifference A' (setDifference A' B') = setDifference B' (setDifference B' A') :=
by sorry

-- Theorem to prove that A - B and B - A are not always equal
theorem difference_not_always_equal :
  ∃ X Y : Set ℝ, setDifference X Y ≠ setDifference Y X :=
by sorry

-- Theorem to prove the specific result of A' - (A' - B') and B' - (B' - A')
theorem specific_difference_result :
  setDifference A' (setDifference A' B') = {x : ℝ | 4 < x ∧ x < 6} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_equality_difference_not_always_equal_specific_difference_result_l404_40450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sumOfFirst10Is124_l404_40459

def customSequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ a 2 = 3 ∧ ∀ n : ℕ, n ≥ 1 → a n * a (n + 1) = 2 * a (n - 1) * a n

def sumOfFirstN (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  (List.range n).map (fun i => a (i + 1)) |>.sum

theorem sumOfFirst10Is124 (a : ℕ → ℕ) (h : customSequence a) :
  sumOfFirstN a 10 = 124 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sumOfFirst10Is124_l404_40459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cinnamon_nutmeg_difference_l404_40458

theorem cinnamon_nutmeg_difference 
  (cinnamon : Real) (nutmeg : Real) 
  (h1 : cinnamon = 0.67)
  (h2 : nutmeg = 0.5) :
  cinnamon - nutmeg = 0.17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cinnamon_nutmeg_difference_l404_40458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nabla_calculation_l404_40416

/-- The ∇ operation for positive real numbers -/
noncomputable def nabla (a b : ℝ) : ℝ := (a + b) / (1 + a * b)

/-- Theorem stating that (2 ∇ 3) ∇ (4 ∇ 5) = 7/8 -/
theorem nabla_calculation :
  nabla (nabla 2 3) (nabla 4 5) = 7/8 :=
by
  -- Expand the definition of nabla
  unfold nabla
  -- Perform algebraic simplifications
  -- This is where the actual proof would go
  sorry

#check nabla_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nabla_calculation_l404_40416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_problem_l404_40461

/-- The interest rate problem -/
theorem interest_rate_problem (principal : ℝ) (rate_C : ℝ) (time : ℝ) (gain_B : ℝ) (rate_A : ℝ) :
  principal = 1000 →
  rate_C = 11.5 →
  time = 3 →
  gain_B = 45 →
  principal * rate_C / 100 * time - principal * rate_A / 100 * time = gain_B →
  rate_A = 10 := by
  sorry

#check interest_rate_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_problem_l404_40461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_special_case_l404_40499

theorem sin_double_angle_special_case (θ : ℝ) (h : Real.cos θ - Real.sin θ = 1/2) : 
  Real.sin (2 * θ) = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_special_case_l404_40499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_l404_40474

theorem expression_simplification :
  4 - 1 + (-6) - (-5) = 4 - 1 - 6 + 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_l404_40474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_function_value_l404_40483

-- Define the quadratic functions p and q
noncomputable def p : ℝ → ℝ := sorry
noncomputable def q : ℝ → ℝ := sorry

-- Define the rational function f
noncomputable def f (x : ℝ) : ℝ := p x / q x

-- State the properties of f
axiom vertical_asymptote : ∀ x, x ≠ -1 → q x ≠ 0
axiom horizontal_asymptote : ∀ ε > 0, ∃ M, ∀ x, abs x > M → abs (f x + 3) < ε
axiom hole_at_6 : p 6 = 0 ∧ q 6 = 0
axiom x_intercept : f 2 = 0

-- Theorem to prove
theorem rational_function_value : f 4 = -6/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_function_value_l404_40483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_upper_bound_l404_40493

noncomputable def f (x : ℝ) : ℝ := Real.cos x * Real.sin (x + Real.pi / 3) - Real.sqrt 3 * (Real.cos x)^2 + Real.sqrt 3 / 4

theorem triangle_area_upper_bound 
  (A B C : ℝ) 
  (acute_triangle : 0 < A ∧ A < Real.pi / 2 ∧ 0 < B ∧ B < Real.pi / 2 ∧ 0 < C ∧ C < Real.pi / 2)
  (angle_sum : A + B + C = Real.pi)
  (side_a : ℝ)
  (side_a_value : side_a = Real.sqrt 3)
  (f_A_value : f A = Real.sqrt 3 / 4) :
  ∀ (area : ℝ), area = 1 / 2 * side_a * (Real.sin B) * (Real.sin C) / (Real.sin (B + C)) → area ≤ 3 * Real.sqrt 3 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_upper_bound_l404_40493
