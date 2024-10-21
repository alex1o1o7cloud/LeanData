import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_rectangle_area_diagonal_ratio_l230_23027

/-- Represents a rectangle with specific properties -/
structure SpecialRectangle where
  length : ℝ
  width : ℝ
  diagonal : ℝ
  ratio_condition : length / width = 5 / 2
  perimeter_condition : 2 * (length + width) = 42

/-- The constant k for which the area of the SpecialRectangle can be expressed as k * diagonal^2 -/
noncomputable def area_diagonal_ratio (rect : SpecialRectangle) : ℝ :=
  (rect.length * rect.width) / (rect.diagonal^2)

/-- Theorem stating that the area_diagonal_ratio of a SpecialRectangle is 10/29 -/
theorem special_rectangle_area_diagonal_ratio :
  ∀ (rect : SpecialRectangle), area_diagonal_ratio rect = 10/29 := by
  intro rect
  -- The proof steps would go here, but we'll use sorry for now
  sorry

#check special_rectangle_area_diagonal_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_rectangle_area_diagonal_ratio_l230_23027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l230_23028

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 5 * x + Real.log x

-- Define the derivative of f
noncomputable def f' (x : ℝ) : ℝ := 5 + 1 / x

-- Theorem statement
theorem tangent_line_equation :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) ↔ (6 * x - y - 1 = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l230_23028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_equality_l230_23031

open Matrix

variable {n : Type*} [DecidableEq n] [Fintype n]

variable {X Y : Matrix (Fin 2) (Fin 2) ℚ}

theorem matrix_equality (h1 : X + Y = X * Y) 
  (h2 : X * Y = !![25/4, 5/4; -10/4, 10/4]) :
  Y * X = X * Y := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_equality_l230_23031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_antifreeze_replacement_percentage_l230_23063

/-- Calculates the percentage of antifreeze in the replacement mixture for a car radiator. -/
theorem antifreeze_replacement_percentage
  (initial_percentage : ℝ)
  (final_percentage : ℝ)
  (total_volume : ℝ)
  (drained_volume : ℝ)
  (h1 : initial_percentage = 0.10)
  (h2 : final_percentage = 0.50)
  (h3 : total_volume = 4)
  (h4 : drained_volume = 2.2857) :
  let initial_antifreeze := initial_percentage * total_volume
  let drained_antifreeze := initial_percentage * drained_volume
  let remaining_volume := total_volume - drained_volume
  let remaining_antifreeze := initial_antifreeze - drained_antifreeze
  let final_antifreeze := final_percentage * total_volume
  let added_antifreeze := final_antifreeze - remaining_antifreeze
  let replacement_percentage := added_antifreeze / drained_volume
  ∃ ε > 0, |replacement_percentage - 0.80| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_antifreeze_replacement_percentage_l230_23063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_intersection_l230_23084

/-- The parabola function -/
noncomputable def f (x : ℝ) : ℝ := -1/2 * x^2 + 2*x - 6

/-- The line function -/
def g (m : ℝ) : ℝ → ℝ := λ x => m

/-- The theorem stating that the line y = m intersects the parabola at exactly one point iff m = 4 -/
theorem unique_intersection (m : ℝ) : 
  (∃! x, f x = g m x) ↔ m = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_intersection_l230_23084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_egyptian_fraction_solutions_l230_23004

def is_solution (a b c : ℕ+) : Prop :=
  (a : ℚ)⁻¹ + (b : ℚ)⁻¹ + (c : ℚ)⁻¹ = 1

def is_permutation (a b c x y z : ℕ+) : Prop :=
  Multiset.ofList [a, b, c] = Multiset.ofList [x, y, z]

theorem egyptian_fraction_solutions :
  ∀ a b c : ℕ+, is_solution a b c ↔
    (is_permutation a b c 2 3 6 ∨
     is_permutation a b c 2 4 4 ∨
     is_permutation a b c 3 3 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_egyptian_fraction_solutions_l230_23004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l230_23005

theorem triangle_problem (A B C : ℝ) (h1 : Real.sin (C - A) = 1) 
  (h2 : Real.sin B = 1/3) (h3 : 0 < A ∧ A < Real.pi) (h4 : 0 < B ∧ B < Real.pi) 
  (h5 : 0 < C ∧ C < Real.pi) (h6 : A + B + C = Real.pi) (AC : ℝ) (h8 : AC = Real.sqrt 6) : 
  Real.sin A = Real.sqrt 3 / 3 ∧ 
  (1/2 : ℝ) * AC * (AC * Real.sin B / Real.sin A) * Real.sin C = 3 * Real.sqrt 2 := by
  sorry

#check triangle_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l230_23005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_integers_median_l230_23080

theorem consecutive_integers_median (n : ℕ) (seq : List ℤ) : 
  seq.length = 81 →
  (∀ i : ℕ, i < 80 → seq.get! (i + 1) = seq.get! i + 1) →
  seq.sum = 3^8 →
  seq.get! 40 = 81 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_integers_median_l230_23080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_three_halves_l230_23094

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  (a_pos : a > 0)
  (b_pos : b > 0)

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola a b) : ℝ := 
  Real.sqrt (1 + b^2 / a^2)

/-- The x-coordinate of the focus of a hyperbola -/
noncomputable def focus_x (h : Hyperbola a b) : ℝ := 
  Real.sqrt (a^2 + b^2)

theorem hyperbola_eccentricity_three_halves 
  (a b : ℝ) (h : Hyperbola a b) 
  (slope_ab : (b^2/a) / (a + focus_x h) = 1/2) : 
  eccentricity h = 3/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_three_halves_l230_23094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vector_ratio_l230_23092

theorem triangle_vector_ratio (A B C P Q : ℝ × ℝ) :
  P ≠ A ∧ P ≠ B ∧ P ≠ C ∧ Q ≠ A ∧ Q ≠ B ∧ Q ≠ C →
  (A.1 - P.1, A.2 - P.2) + 2 • (B.1 - P.1, B.2 - P.2) + 3 • (C.1 - P.1, C.2 - P.2) = (0, 0) →
  2 • (A.1 - Q.1, A.2 - Q.2) + 3 • (B.1 - Q.1, B.2 - Q.2) + 5 • (C.1 - Q.1, C.2 - Q.2) = (0, 0) →
  ‖(P.1 - Q.1, P.2 - Q.2)‖ / ‖(A.1 - B.1, A.2 - B.2)‖ = 1 / 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vector_ratio_l230_23092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_equation_l230_23016

/-- Parabola C1 with equation y^2 = 8x -/
def C1 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 8 * p.1}

/-- Hyperbola C2 -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  center : ℝ × ℝ

/-- The directrix of a parabola y^2 = 4ax is x = -a -/
def directrix (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = -a}

/-- The focus of a parabola y^2 = 4ax is (a, 0) -/
def focus (a : ℝ) : ℝ × ℝ :=
  (a, 0)

/-- The chord intercepted by a hyperbola on a vertical line -/
noncomputable def intercepted_chord (h : Hyperbola) (x : ℝ) : ℝ :=
  2 * h.b * Real.sqrt ((x - h.center.1)^2 / h.a^2 - 1) * h.b / h.a

/-- The equation of the asymptote of a hyperbola -/
def asymptote_equation (h : Hyperbola) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = h.b / h.a * (p.1 - h.center.1) + h.center.2 ∨
               p.2 = -h.b / h.a * (p.1 - h.center.1) + h.center.2}

theorem hyperbola_asymptote_equation (h : Hyperbola) :
  (∃ a : ℝ, a > 0 ∧ C1 = {p : ℝ × ℝ | p.2^2 = 4 * a * p.1}) →
  (∃ x : ℝ, directrix a = {p : ℝ × ℝ | p.1 = x} ∧ (x, 0) ∈ {p : ℝ × ℝ | p = h.center + (h.a, 0) ∨ p = h.center - (h.a, 0)}) →
  intercepted_chord h x = 6 →
  asymptote_equation h = {p : ℝ × ℝ | p.2 = Real.sqrt 3 * (p.1 - h.center.1) + h.center.2 ∨
                                      p.2 = -Real.sqrt 3 * (p.1 - h.center.1) + h.center.2} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_equation_l230_23016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_tripling_period_l230_23019

/-- The annual interest rate as a real number -/
def interest_rate : ℝ := 0.3334

/-- The function that calculates the value of an investment after t years -/
noncomputable def investment_value (initial_value : ℝ) (t : ℝ) : ℝ :=
  initial_value * (1 + interest_rate) ^ t

/-- The theorem stating that 4 is the smallest integer number of years 
    needed for an investment to more than triple in value -/
theorem smallest_tripling_period :
  ∀ (initial_value : ℝ), initial_value > 0 →
    (∀ (t : ℕ), t < 4 → investment_value initial_value (t : ℝ) ≤ 3 * initial_value) ∧
    (investment_value initial_value 4 > 3 * initial_value) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_tripling_period_l230_23019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_compositions_l230_23047

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 - 1

noncomputable def g (x : ℝ) : ℝ :=
  if x > 0 then x - 1
  else if x < 0 then 2 - x
  else 0  -- define g(0) as 0 to make it total

-- Theorem statements
theorem function_compositions :
  (f (g 2) = 0) ∧
  (g (f 2) = 2) ∧
  (g (g (g (-2))) = 2) ∧
  (∀ x : ℝ, (x < -1 ∨ x > 1 → g (f x) = x^2 - 2) ∧
            (-1 < x ∧ x < 1 → g (f x) = 4 - x^2)) ∧
  (∀ x : ℝ, (x > 0 → f (g x) = x^2 - 2*x) ∧
            (x < 0 → f (g x) = x^2 - 4*x + 3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_compositions_l230_23047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_lies_on_ac_l230_23046

/-- Given a triangle ABC with median AM, points K and L that divide AM into three equal parts
    with K between L and M, and a point P such that triangles KPL and ABC are similar,
    with P and C on the same side of line AM, prove that P lies on line AC. -/
theorem p_lies_on_ac (A B C M K L P : EuclideanSpace ℝ (Fin 2)) : 
  (∃ (t : ℝ), M = (2 / 3 : ℝ) • A + (1 / 3 : ℝ) • C) →  -- M is the midpoint of BC
  (∃ (s₁ s₂ : ℝ), K = s₁ • A + (1 - s₁) • M ∧ L = s₂ • A + (1 - s₂) • M ∧ s₂ = 2 * s₁ ∧ 0 < s₁ ∧ s₁ < s₂ ∧ s₂ < 1) →  -- K and L divide AM into three equal parts, K between L and M
  (∃ (r : ℝ), ∀ (X Y : EuclideanSpace ℝ (Fin 2)), 
    (X - K) = r • (Y - A) ∧ (P - L) = r • (C - B)) →  -- Triangles KPL and ABC are similar
  (∃ (u : ℝ), P = u • A + (1 - u) • C) :=  -- P lies on line AC
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_lies_on_ac_l230_23046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l230_23085

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the area function for the triangle
noncomputable def triangle_area (t : Triangle) : ℝ :=
  1 / 2 * t.a * t.c * Real.sin t.B

-- State the theorem
theorem triangle_problem (t : Triangle) :
  (2 * Real.sin t.A * Real.sin t.C * (1 / (Real.tan t.A * Real.tan t.C) - 1) = -1) →
  (t.B = π / 3 ∧
   ((t.a + t.c = 3 * Real.sqrt 3 / 2 ∧ t.b = Real.sqrt 3) →
    triangle_area t = 5 * Real.sqrt 3 / 16)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l230_23085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_APQ_l230_23074

/-- Two perpendicular lines intersecting at point A(8, 10) -/
structure PerpendicularLines where
  line1 : ℝ → ℝ
  line2 : ℝ → ℝ
  perpendicular : line1 0 * line2 0 = -1
  intersect_at_A : line1 8 = 10 ∧ line2 8 = 10

/-- The sum of y-intercepts is 10 -/
def sum_of_y_intercepts (lines : PerpendicularLines) : ℝ := lines.line1 0 + lines.line2 0

/-- Points P and Q are the y-intercepts of the lines -/
def point_P (lines : PerpendicularLines) : ℝ × ℝ := (0, lines.line1 0)
def point_Q (lines : PerpendicularLines) : ℝ × ℝ := (0, lines.line2 0)

/-- The area of triangle APQ -/
noncomputable def area_APQ (lines : PerpendicularLines) : ℝ :=
  (1 / 2) * 8 * 10

/-- Theorem statement -/
theorem area_of_triangle_APQ (lines : PerpendicularLines) 
  (h : sum_of_y_intercepts lines = 10) : area_APQ lines = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_APQ_l230_23074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_students_answering_both_questions_correctly_l230_23053

theorem students_answering_both_questions_correctly 
  (total_students : ℕ) 
  (students_q1_correct : ℕ) 
  (students_q2_correct : ℕ) 
  (students_absent : ℕ) 
  (h1 : total_students = 25)
  (h2 : students_q1_correct = 22)
  (h3 : students_q2_correct = 20)
  (h4 : students_absent = 3)
  (h5 : students_q1_correct + students_absent ≤ total_students) :
  students_q2_correct = Nat.min students_q1_correct students_q2_correct := by
  sorry

#check students_answering_both_questions_correctly

end NUMINAMATH_CALUDE_ERRORFEEDBACK_students_answering_both_questions_correctly_l230_23053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l230_23062

def sequence_a : ℕ → ℤ
  | 0 => 1  -- Added case for 0
  | 1 => 1
  | 2 => 2
  | (n + 3) => 2 * sequence_a (n + 2) - sequence_a (n + 1) + 2

def sequence_b (n : ℕ) : ℤ := sequence_a (n + 1) - sequence_a n

theorem sequence_properties :
  (∃ d : ℤ, ∀ n : ℕ, sequence_b (n + 1) - sequence_b n = d) ∧
  (∀ n : ℕ, sequence_a n = n^2 - 2*n + 2) := by
  sorry

#eval sequence_a 5  -- For testing purposes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l230_23062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2A_value_l230_23073

theorem sin_2A_value (A : Real) (h1 : 0 < A) (h2 : A < π / 2) (h3 : Real.cos A = 3 / 5) : 
  Real.sin (2 * A) = 24 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2A_value_l230_23073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_successive_discounts_tv_discount_problem_l230_23081

/-- Calculates the final price as a percentage of the original price after applying two successive discounts --/
theorem successive_discounts (original_price : ℝ) (discount1 discount2 : ℝ) :
  original_price > 0 →
  0 ≤ discount1 ∧ discount1 < 1 →
  0 ≤ discount2 ∧ discount2 < 1 →
  let final_price := original_price * (1 - discount1) * (1 - discount2)
  final_price / original_price = (1 - discount1) * (1 - discount2) := by
  sorry

/-- Proves that applying 25% and 10% discounts successively to $350.00 results in 67.5% of the original price --/
theorem tv_discount_problem :
  let original_price : ℝ := 350
  let discount1 : ℝ := 0.25
  let discount2 : ℝ := 0.10
  let final_percentage : ℝ := (1 - discount1) * (1 - discount2) * 100
  final_percentage = 67.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_successive_discounts_tv_discount_problem_l230_23081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_fixed_point_l230_23071

noncomputable section

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- Checks if a point lies on the ellipse -/
def onEllipse (p : Point) (e : Ellipse) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- Calculates the slope between two points -/
def slopeBetween (p1 p2 : Point) : ℝ :=
  (p2.y - p1.y) / (p2.x - p1.x)

/-- Theorem: Given an ellipse with specific properties, any line AB where A and B are on the ellipse
    and the sum of slopes of MA and MB is 8 (M being the top vertex) passes through (-1/2, -2) -/
theorem ellipse_fixed_point (e : Ellipse) (f1 f2 p q m a b : Point) :
  f1 = ⟨-2, 0⟩ →
  f2 = ⟨2, 0⟩ →
  onEllipse p e →
  onEllipse q e →
  p.x = q.x →
  (p.y - q.y)^2 = 8 →
  onEllipse m e →
  m.y > 0 →
  (∀ y, y > m.y → ¬onEllipse ⟨m.x, y⟩ e) →
  onEllipse a e →
  onEllipse b e →
  slopeBetween m a + slopeBetween m b = 8 →
  ∃ t : ℝ, a.x * (1 - t) + b.x * t = -1/2 ∧ a.y * (1 - t) + b.y * t = -2 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_fixed_point_l230_23071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l230_23011

def start_point : Fin 3 → ℝ := ![1, 1, 1]
def end_point : Fin 3 → ℝ := ![4, 2, 2]
def sphere_radius : ℝ := 2

theorem intersection_distance :
  let line := {p : Fin 3 → ℝ | ∃ t : ℝ, p = fun i => start_point i + t * (end_point i - start_point i)}
  let sphere := {p : Fin 3 → ℝ | (p 0)^2 + (p 1)^2 + (p 2)^2 = sphere_radius^2}
  let intersection := {p : Fin 3 → ℝ | p ∈ line ∩ sphere}
  ∃ p1 p2 : Fin 3 → ℝ, p1 ∈ intersection ∧ p2 ∈ intersection ∧
    p1 ≠ p2 ∧ 
    Real.sqrt (((p1 0) - (p2 0))^2 + ((p1 1) - (p2 1))^2 + ((p1 2) - (p2 2))^2) = 2 * Real.sqrt 418 / 13 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l230_23011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_satisfying_number_l230_23032

def remove_digit (n : ℕ) (pos : ℕ) : ℕ :=
  let s := n.repr
  let left := s.take (pos - 1)
  let right := s.drop pos
  (left ++ right).toNat!

def satisfies_conditions (n : ℕ) : Prop :=
  n ≥ 100000000 ∧ n < 1000000000 ∧
  (∀ i : ℕ, i ∈ Finset.range 8 → (remove_digit n (i + 2)) % (i + 2) = 0)

theorem exists_satisfying_number : ∃ n : ℕ, satisfies_conditions n := by
  -- We know that 900900000 satisfies the conditions
  use 900900000
  apply And.intro
  · norm_num
  apply And.intro
  · norm_num
  intro i hi
  simp only [remove_digit, satisfies_conditions]
  -- The proof for each case would be quite long, so we'll use sorry for now
  sorry

#eval remove_digit 900900000 2  -- Should output 90900000
#eval remove_digit 900900000 3  -- Should output 90000000
#eval remove_digit 900900000 9  -- Should output 90090000

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_satisfying_number_l230_23032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_winning_strategy_exists_l230_23042

/-- Represents a strategy for player A --/
def Strategy := List (Fin 101) → List (Fin 101)

/-- Represents the game state after each turn --/
def GameState := List (Fin 101)

/-- Simulates a single turn of the game --/
def playTurn (s : GameState) (move : List (Fin 101)) : GameState :=
  s.filter (λ x => x ∉ move)

/-- Simulates the entire game given strategies for both players --/
def playGame (strategyA : Strategy) (strategyB : Strategy) : GameState :=
  sorry

/-- Checks if the final state satisfies the winning condition for A --/
def isWinningState (s : GameState) : Prop :=
  s.length = 2 ∧ (s.maximum?.getD 0 - s.minimum?.getD 0 : Int) = 55

theorem a_winning_strategy_exists :
  ∃ (strategyA : Strategy), ∀ (strategyB : Strategy),
    isWinningState (playGame strategyA strategyB) :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_winning_strategy_exists_l230_23042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l230_23068

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log x - (x + 1)

-- State the theorem
theorem f_properties :
  -- The tangent line at (1, f(1)) is parallel to x-axis
  (deriv f 1 = 0) →
  -- f is increasing on (0,1) and decreasing on (1,+∞)
  (∀ x y, 0 < x ∧ x < y ∧ y < 1 → f x < f y) ∧
  (∀ x y, 1 < x ∧ x < y → f y < f x) ∧
  -- For any k < 1, there exists an x₀ > 1 such that for all x ∈ (1, x₀), f(x) - x²/2 + 2x + 1/2 > k(x-1)
  (∀ k, k < 1 → ∃ x₀, 1 < x₀ ∧ ∀ x, 1 < x ∧ x < x₀ → f x - x^2/2 + 2*x + 1/2 > k*(x-1)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l230_23068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l230_23051

/-- Given a projection that takes [1, -2] to [3/2, -3/2], 
    prove that it takes [3, -6] to [4.5, -4.5] -/
theorem projection_theorem (proj : ℝ × ℝ → ℝ × ℝ) 
    (h : proj (1, -2) = (3/2, -3/2)) : 
  proj (3, -6) = (4.5, -4.5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l230_23051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_f_implies_a_in_range_l230_23049

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 + (1/2)*a*x - 2
  else a^x - a

-- State the theorem
theorem increasing_f_implies_a_in_range (a : ℝ) :
  (∀ x y : ℝ, 0 < x ∧ x < y → f a x < f a y) →
  1 < a ∧ a ≤ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_f_implies_a_in_range_l230_23049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brick_debris_area_calculation_l230_23059

-- Define the points and basic measurements
noncomputable def E : ℝ × ℝ := (0, 0)  -- Assuming E is at the origin for simplicity
noncomputable def O : ℝ × ℝ := (2, 0)  -- O is 2 meters from E on the x-axis

-- Define the radii
noncomputable def r_E : ℝ := 2 * Real.sqrt 2
noncomputable def r_O : ℝ := 6

-- Define the angles
noncomputable def angle_EOD : ℝ := Real.pi / 4  -- 45 degrees in radians

-- Define the area function
noncomputable def brick_debris_area : ℝ :=
  -- Area of sector COD
  (3 * Real.pi / 4) * r_O^2 -
  -- Area of sector FEG
  (7 * Real.pi / 6) * r_E^2 +
  -- Area of two triangles OEF
  2 * (1 + Real.sqrt 3)

-- State the theorem
theorem brick_debris_area_calculation :
  brick_debris_area = (67 * Real.pi / 3) + 2 * (1 + Real.sqrt 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_brick_debris_area_calculation_l230_23059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_root_sum_is_one_l230_23067

noncomputable def z : ℂ := Complex.exp (2 * Real.pi * Complex.I / 5)

theorem fifth_root_sum_is_one :
  z / (1 + z^2) + z^2 / (1 + z^4) + z^3 / (1 + z^6) + z^4 / (1 + z^8) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_root_sum_is_one_l230_23067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_y_and_prs_over_q_l230_23003

theorem largest_y_and_prs_over_q : ∃ (p q r s : ℤ),
  let y : ℝ := (p + q * Real.sqrt r) / s
  4 * y / 5 - 2 = 10 / y ∧
  y ≤ 5 ∧
  p * r * s / q = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_y_and_prs_over_q_l230_23003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l230_23097

-- Define the points A, B, and C
def A : ℝ × ℝ := (0, 4)
def B : ℝ × ℝ := (4, 4)
def C (p : ℝ) : ℝ × ℝ := (p, p^2 - 4*p + 4)

-- Define the parabola equation
def on_parabola (x y : ℝ) : Prop := y = x^2 - 4*x + 4

-- Define the area of triangle ABC as a function of p
noncomputable def triangle_area (p : ℝ) : ℝ := 
  abs (A.1 * B.2 + B.1 * (C p).2 + (C p).1 * A.2 - B.1 * A.2 - (C p).1 * B.2 - A.1 * (C p).2) / 2

-- State the theorem
theorem max_triangle_area :
  ∃ (max_area : ℝ), 
    (∀ p : ℝ, 0 ≤ p ∧ p ≤ 4 ∧ on_parabola (C p).1 (C p).2 → triangle_area p ≤ max_area) ∧
    (∃ p : ℝ, 0 ≤ p ∧ p ≤ 4 ∧ on_parabola (C p).1 (C p).2 ∧ triangle_area p = max_area) ∧
    max_area = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l230_23097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_range_l230_23040

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 2*x + 5

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x + 2

-- Define the slope angle of the tangent line
noncomputable def slope_angle (x : ℝ) : ℝ := Real.arctan (f' x)

-- Statement of the theorem
theorem slope_angle_range :
  ∀ x : ℝ, slope_angle x ∈ Set.union (Set.Ico 0 (Real.pi / 2)) (Set.Ico (3 * Real.pi / 4) Real.pi) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_range_l230_23040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_8_l230_23021

/-- Represents the sum of the first n terms of a geometric sequence -/
def S (n : ℕ) : ℝ := sorry

/-- The common ratio of the geometric sequence -/
def q : ℝ := sorry

theorem geometric_sequence_sum_8 (h1 : S 2 = 1) (h2 : S 4 = 5) : S 8 = 85 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_8_l230_23021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cone_volume_ratio_l230_23033

/-- A right circular cone inscribed in a cube -/
structure InscribedCone where
  s : ℝ  -- side length of the cube
  h : ℝ  -- height of the cone
  r : ℝ  -- radius of the base of the cone
  height_eq_side : h = s
  radius_half_side : r = s / 2

/-- The ratio of the volume of the cone to the volume of the cube -/
noncomputable def volume_ratio (cone : InscribedCone) : ℝ :=
  (1/3 * Real.pi * cone.r^2 * cone.h) / cone.s^3

/-- Theorem: The ratio of the volume of the inscribed cone to the volume of the cube is π/12 -/
theorem inscribed_cone_volume_ratio (cone : InscribedCone) :
  volume_ratio cone = Real.pi / 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cone_volume_ratio_l230_23033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angles_l230_23070

theorem triangle_angles (A B C : ℝ) (h1 : Real.sin A + Real.cos A = Real.sqrt 2) 
  (h2 : Real.sqrt 3 * Real.cos A = -Real.sqrt 2 * Real.cos (π - B)) 
  (h3 : 0 < A ∧ A < π) (h4 : 0 < B ∧ B < π) (h5 : A + B + C = π) : 
  A = π / 4 ∧ B = π / 6 ∧ C = 7 * π / 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angles_l230_23070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candle_lighting_time_l230_23043

/-- Represents a candle with its burn time in hours -/
structure Candle where
  burn_time : ℚ

/-- Represents the state of two candles at a given time -/
structure CandleState where
  candle1 : Candle
  candle2 : Candle
  time_elapsed : ℚ

/-- Calculates the remaining length of a candle as a fraction of its original length -/
def remaining_length (c : Candle) (t : ℚ) : ℚ :=
  1 - t / c.burn_time

/-- The theorem to be proved -/
theorem candle_lighting_time (state : CandleState) :
  state.candle1.burn_time = 5 →
  state.candle2.burn_time = 6 →
  remaining_length state.candle2 state.time_elapsed = 3 * remaining_length state.candle1 state.time_elapsed →
  state.time_elapsed = 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_candle_lighting_time_l230_23043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_l230_23000

noncomputable def f (α : ℝ) (x : ℝ) : ℝ := x ^ α

def line_eq (k x y : ℝ) : Prop := k * x - y + 2 * k + 1 + Real.sqrt 3 = 0

def point_A : ℝ × ℝ := (1, 1)

noncomputable def point_B : ℝ × ℝ := (-2, 1 + Real.sqrt 3)

theorem angle_of_inclination :
  (∀ α : ℝ, f α point_A.1 = point_A.2) →
  (∀ k : ℝ, line_eq k point_B.1 point_B.2) →
  let m := (point_B.2 - point_A.2) / (point_B.1 - point_A.1)
  Real.arctan m = 5 * Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_l230_23000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l230_23082

noncomputable def a (θ : ℝ) : ℝ × ℝ := (Real.cos (3 * θ / 2), Real.sin (3 * θ / 2))

noncomputable def b (θ : ℝ) : ℝ × ℝ := (Real.cos (θ / 2), -Real.sin (θ / 2))

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

def vector_sum (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)

noncomputable def vector_norm (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

def scalar_mult (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)

def vector_sub (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 - w.1, v.2 - w.2)

theorem vector_properties (θ : ℝ) (h : 0 ≤ θ ∧ θ ≤ π / 3) :
  (∃ (max min : ℝ),
    max = 1/2 ∧
    min = -1/2 ∧
    (∀ θ' : ℝ, 0 ≤ θ' ∧ θ' ≤ π / 3 →
      min ≤ (dot_product (a θ') (b θ')) / (vector_norm (vector_sum (a θ') (b θ'))) ∧
      (dot_product (a θ') (b θ')) / (vector_norm (vector_sum (a θ') (b θ'))) ≤ max)) ∧
  (∃ k : ℝ, vector_norm (vector_sum (scalar_mult k (a θ)) (b θ)) = Real.sqrt 3 * vector_norm (vector_sub (a θ) (scalar_mult k (b θ)))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l230_23082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_l230_23077

/-- The time it takes for a train to cross a bridge -/
noncomputable def train_crossing_time (train_length : ℝ) (bridge_length : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let total_distance := train_length + bridge_length
  total_distance / train_speed_ms

/-- Theorem stating that a train of length 250 m crossing a bridge of length 150 m
    at a speed of 57.6 km/h takes 25 seconds to complete the crossing -/
theorem train_crossing_bridge :
  train_crossing_time 250 150 57.6 = 25 := by
  -- Unfold the definition of train_crossing_time
  unfold train_crossing_time
  -- Simplify the expression
  simp
  -- The proof is skipped for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_l230_23077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_integer_in_sequence_l230_23099

def geometric_sequence (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * r^n

def is_integer (q : ℚ) : Prop :=
  ∃ (z : ℤ), q = ↑z

def our_sequence (n : ℕ) : ℚ :=
  geometric_sequence 2048000 (1/2) n

theorem last_integer_in_sequence :
  (∃ (k : ℕ), is_integer (our_sequence k) ∧ ¬is_integer (our_sequence (k + 1))) →
  (∃ (k : ℕ), our_sequence k = 125 ∧ ¬is_integer (our_sequence (k + 1))) :=
sorry

#eval our_sequence 14  -- Should output 125
#eval our_sequence 15  -- Should output 62.5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_integer_in_sequence_l230_23099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_forall_not_equal_l230_23024

theorem negation_of_forall_not_equal : 
  ¬(∀ x : ℝ, x^2 ≠ x) ↔ ∃ x : ℝ, x^2 = x :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_forall_not_equal_l230_23024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_requires_more_paint_l230_23037

/-- The volume of paint required for a sphere -/
noncomputable def sphere_paint_volume (R d : ℝ) : ℝ := 4 * Real.pi * (R^2*d + R*d^2 + d^3/3)

/-- The volume of paint required for a cylinder -/
noncomputable def cylinder_paint_volume (R d : ℝ) : ℝ := 2 * Real.pi * (2*R^2*d + R*d^2)

/-- Theorem stating that a sphere requires more paint than a cylinder -/
theorem sphere_requires_more_paint (R d : ℝ) (h1 : R > 0) (h2 : d > 0) :
  sphere_paint_volume R d > cylinder_paint_volume R d :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_requires_more_paint_l230_23037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_quadratic_l230_23055

-- Define the functions
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x
noncomputable def g (b : ℝ) (x : ℝ) : ℝ := -b / x
noncomputable def h (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the interval [0, +∞)
def nonnegative_reals : Set ℝ := {x : ℝ | x ≥ 0}

-- State the theorem
theorem decreasing_quadratic 
  (a b c : ℝ) 
  (h1 : ∀ x ∈ nonnegative_reals, StrictMonoOn (f a) nonnegative_reals)
  (h2 : ∀ x ∈ nonnegative_reals, StrictMonoOn (g b) nonnegative_reals) :
  StrictMonoOn (h a b c) nonnegative_reals :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_quadratic_l230_23055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_polar_circles_l230_23091

/-- The distance between the centers of two circles in polar form -/
noncomputable def distance_between_centers (circle1 circle2 : ℝ → ℝ) : ℝ :=
  let center1 := (1, 0)
  let center2 := (0, 1/2)
  Real.sqrt ((center2.1 - center1.1)^2 + (center2.2 - center1.2)^2)

/-- The first circle defined in polar form -/
noncomputable def circle1 (θ : ℝ) : ℝ := 2 * Real.cos θ

/-- The second circle defined in polar form -/
noncomputable def circle2 (θ : ℝ) : ℝ := Real.sin θ

theorem distance_between_polar_circles :
  distance_between_centers circle1 circle2 = Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_polar_circles_l230_23091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bluegrass_percentage_in_mixture_x_l230_23006

/-- Represents the composition of a seed mixture -/
structure SeedMixture where
  ryegrass : ℚ
  bluegrass : ℚ
  fescue : ℚ

/-- The percentage of a component in a mixture -/
def percentage (part : ℚ) (whole : ℚ) : ℚ := (part / whole) * 100

theorem bluegrass_percentage_in_mixture_x (x : SeedMixture) (y : SeedMixture) 
  (mixture : SeedMixture) :
  x.ryegrass = 40 →
  y.ryegrass = 25 →
  y.fescue = 75 →
  mixture.ryegrass = 27 →
  percentage x.ryegrass (x.ryegrass + x.bluegrass + x.fescue) = 40 →
  percentage y.ryegrass (y.ryegrass + y.fescue) = 25 →
  percentage mixture.ryegrass (mixture.ryegrass + mixture.bluegrass + mixture.fescue) = 27 →
  percentage x.ryegrass (mixture.ryegrass + mixture.bluegrass + mixture.fescue) = 40 / 3 →
  x.bluegrass = 60 := by
  sorry

#eval (40 : ℚ) / 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bluegrass_percentage_in_mixture_x_l230_23006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_unit_vectors_l230_23056

/-- Given a vector a = (2, -2), prove that (√2/2, √2/2) and (-√2/2, -√2/2) are the only unit vectors perpendicular to a. -/
theorem perpendicular_unit_vectors (a : ℝ × ℝ) (h : a = (2, -2)) :
  let perp_vectors : Set (ℝ × ℝ) := {v | v.1^2 + v.2^2 = 1 ∧ v.1 * a.1 + v.2 * a.2 = 0}
  perp_vectors = {(Real.sqrt 2 / 2, Real.sqrt 2 / 2), (-Real.sqrt 2 / 2, -Real.sqrt 2 / 2)} :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_unit_vectors_l230_23056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l230_23007

noncomputable def f (x : ℝ) := Real.sin x ^ 2 + Real.sin (2 * x) + 3 * Real.cos x ^ 2

theorem f_properties :
  (∃ (k : ℤ), ∀ (x : ℝ), f x ≥ 2 - Real.sqrt 2 ∧ (f x = 2 - Real.sqrt 2 ↔ ∃ (m : ℤ), x = m * π - 3 * π / 8)) ∧
  (∀ (k : ℤ) (x y : ℝ), k * π + π / 8 ≤ x ∧ x < y ∧ y ≤ 5 * π / 8 + k * π → f x > f y) ∧
  (∀ (x : ℝ), -π / 4 ≤ x ∧ x ≤ π / 4 → 1 ≤ f x ∧ f x ≤ 2 + Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l230_23007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l230_23009

theorem circle_properties (x y m : ℝ) :
  -- Definition of the circle equation
  let circle_eq := x^2 + y^2 - 2*x - 4*y + m
  -- Definition of the line equation
  let line_eq := x + 2*y - 4

  -- Part 1: Condition for circle existence
  (∃ (x y : ℝ), circle_eq = 0) → m < 5 ∧

  -- Part 2: Intersection with specific chord length
  (∃ (x₁ y₁ x₂ y₂ : ℝ),
    circle_eq = 0 ∧
    circle_eq = 0 ∧
    line_eq = 0 ∧
    line_eq = 0 ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = (4*Real.sqrt 5/5)^2) →
  m = 4 ∧

  -- Part 3: Perpendicular chords through origin
  (∃ (x₁ y₁ x₂ y₂ : ℝ),
    circle_eq = 0 ∧
    circle_eq = 0 ∧
    line_eq = 0 ∧
    line_eq = 0 ∧
    x₁*x₂ + y₁*y₂ = 0) →
  m = 8/5 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l230_23009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_target_hit_probability_l230_23098

/-- The probability of hitting a target in a single shot. -/
noncomputable def hit_probability : ℚ := 1/2

/-- The total number of shots. -/
def total_shots : ℕ := 6

/-- The number of successful hits required. -/
def required_hits : ℕ := 3

/-- The number of consecutive hits required. -/
def consecutive_hits : ℕ := 2

/-- The probability of hitting the target 3 times and having exactly 2 consecutive hits in 6 shots. -/
theorem target_hit_probability : 
  (Nat.choose 4 2 : ℚ) * hit_probability ^ total_shots = 
  (Nat.descFactorial 4 2 : ℚ) * (1/2)^6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_target_hit_probability_l230_23098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_base_eight_digit_sum_l230_23052

/-- Function to calculate the sum of digits in base-eight representation -/
def base_eight_digit_sum (n : ℕ) : ℕ :=
  let rec sum_digits (m : ℕ) (acc : ℕ) : ℕ :=
    if m = 0 then acc
    else sum_digits (m / 8) (acc + m % 8)
  sum_digits n 0

/-- The greatest possible sum of the digits in the base-eight representation of a positive integer less than 1728 -/
theorem max_base_eight_digit_sum : ∃ (n : ℕ), n < 1728 ∧ 
  (∀ (m : ℕ), m < 1728 → base_eight_digit_sum m ≤ base_eight_digit_sum n) ∧
  base_eight_digit_sum n = 23 := by
  -- We claim that n = 1535 (which is 2777 in base 8) satisfies the conditions
  use 1535
  constructor
  · -- Prove 1535 < 1728
    norm_num
  constructor
  · -- Prove that for all m < 1728, base_eight_digit_sum m ≤ base_eight_digit_sum 1535
    sorry -- This requires a more detailed proof
  · -- Prove that base_eight_digit_sum 1535 = 23
    -- This can be computed directly
    norm_num [base_eight_digit_sum]
    rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_base_eight_digit_sum_l230_23052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l230_23001

-- Define the equation
def equation (a : ℝ) (t : ℝ) : Prop :=
  (4*a*(Real.cos t)^2 + 4*a*(2*Real.sqrt 2 - 1)*(Real.cos t) + 4*(a-1)*(Real.sin t) + a + 2) / 
  (Real.sin t + 2*Real.sqrt 2*(Real.cos t)) = 4*a

-- Define the interval
def interval (t : ℝ) : Prop := -Real.pi/2 < t ∧ t < 0

-- Define the condition for exactly two distinct solutions
def has_two_distinct_solutions (a : ℝ) : Prop :=
  ∃ t₁ t₂, t₁ ≠ t₂ ∧ interval t₁ ∧ interval t₂ ∧ equation a t₁ ∧ equation a t₂ ∧
  ∀ t₃, interval t₃ ∧ equation a t₃ → t₃ = t₁ ∨ t₃ = t₂

-- Define the set of values for a
def valid_a_values (a : ℝ) : Prop :=
  (a < -18 - 24*Real.sqrt 2) ∨ (-18 - 24*Real.sqrt 2 < a ∧ a < -6)

-- Theorem statement
theorem equation_solutions (a : ℝ) : 
  has_two_distinct_solutions a ↔ valid_a_values a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l230_23001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l230_23086

/-- Helper definition for the directrix of a parabola -/
def is_directrix (x y : ℝ) : Prop :=
  ∀ (p q : ℝ), q = p^2 → (x - p)^2 + (y - q)^2 = (x + p)^2 + (y - q)^2

/-- The directrix of a parabola y = x^2 -/
theorem parabola_directrix :
  ∀ (x y : ℝ), y = x^2 → (∃ (k : ℝ), k = -1/4 ∧ (y = k ↔ is_directrix x y)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l230_23086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_sales_at_45_l230_23036

/-- Represents the relationship between price and number of customers for a book --/
structure BookSales where
  price : ℚ
  customers : ℚ

/-- The constant of proportionality for inverse relationship between price and customers --/
def k : ℚ := 1200

/-- The promotional effect multiplier --/
def promo_effect : ℚ := 11/10

/-- Initial sales data --/
def initial_sales : BookSales := ⟨30, 40⟩

/-- Calculates the number of customers for a given price --/
def calculate_customers (price : ℚ) : ℚ :=
  promo_effect * (k / price)

/-- The theorem to be proved --/
theorem book_sales_at_45 :
  ⌊calculate_customers 45⌋ = 29 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_sales_at_45_l230_23036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_length_is_25_l230_23088

/-- Represents the dimensions and plastering cost of a tank -/
structure Tank where
  width : ℝ
  depth : ℝ
  plastering_cost : ℝ
  plastering_rate : ℝ

/-- Calculates the length of the tank given its dimensions and plastering cost -/
noncomputable def tank_length (t : Tank) : ℝ :=
  let total_area := t.plastering_cost * 100 / t.plastering_rate
  (total_area - 2 * t.width * t.depth) / (2 * t.depth + t.width)

/-- Theorem stating that for a tank with given dimensions and plastering cost, its length is 25 meters -/
theorem tank_length_is_25 (t : Tank) 
  (h_width : t.width = 12)
  (h_depth : t.depth = 6)
  (h_cost : t.plastering_cost = 409.20)
  (h_rate : t.plastering_rate = 55) :
  tank_length t = 25 := by
  sorry

#check tank_length_is_25

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_length_is_25_l230_23088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_fraction_parts_l230_23022

/-- The repeating decimal 3.71717171... -/
def x : ℚ := 3 + 71 / 99

theorem sum_of_fraction_parts : ∃ (a b : ℕ), (x = a / b) ∧ Nat.Coprime a b ∧ a + b = 467 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_fraction_parts_l230_23022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_charge_theorem_l230_23017

/-- Calculates the total charge for Jim's taxi service --/
def calculate_taxi_charge (initial_fee : ℚ) (charge_per_increment : ℚ) (increment_distance : ℚ) 
  (trip_distance : ℚ) (heavy_traffic : Bool) (luxury_car : Bool) : ℚ :=
  let base_cost := initial_fee + (trip_distance / increment_distance).floor * charge_per_increment
  let with_traffic_surcharge := if heavy_traffic then base_cost * (1 + 1/10) else base_cost
  let total_cost := if luxury_car then with_traffic_surcharge * (1 + 1/10) else with_traffic_surcharge
  (total_cost * 100).floor / 100

/-- Theorem stating that the total charge for the given trip is $6.66 --/
theorem taxi_charge_theorem : 
  calculate_taxi_charge (235/100) (35/100) (2/5) (36/10) true true = 666/100 := by
  sorry

#eval calculate_taxi_charge (235/100) (35/100) (2/5) (36/10) true true

end NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_charge_theorem_l230_23017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l230_23030

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 + x^2 - 3*x + 1

-- Define the derivative of f
noncomputable def f' (x : ℝ) : ℝ := x^2 + 2*x - 3

-- Theorem for monotonicity intervals and extrema
theorem f_properties :
  (∀ x < -3, (f' x > 0)) ∧ 
  (∀ x ∈ Set.Ioo (-3) 1, (f' x < 0)) ∧ 
  (∀ x > 1, (f' x > 0)) ∧
  (∀ x, f x ≤ f (-3)) ∧
  (∀ x, f x ≥ f 1) ∧
  (f (-3) = 10) ∧
  (f 1 = -2/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l230_23030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_k_for_fibonacci_roots_l230_23014

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

/-- Check if a number is a Fibonacci number -/
def is_fibonacci (n : ℕ) : Prop :=
  ∃ i, fib i = n

/-- The quadratic equation x^2 - 20x + k = 0 -/
def quadratic_equation (k : ℕ) (x : ℝ) : Prop :=
  x^2 - 20*x + k = 0

/-- Both roots of the quadratic equation are Fibonacci numbers -/
def both_roots_fibonacci (k : ℕ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ quadratic_equation k x ∧ quadratic_equation k y ∧
    is_fibonacci (Int.toNat (Int.floor x)) ∧ is_fibonacci (Int.toNat (Int.floor y))

/-- The main theorem -/
theorem unique_k_for_fibonacci_roots :
  ∃! k : ℕ, both_roots_fibonacci k ∧ k = 104 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_k_for_fibonacci_roots_l230_23014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_area_eq_l230_23076

/-- The combined area of a circle with diameter 12 meters and a square with side length 12 meters -/
noncomputable def combined_area : ℝ :=
  let circle_diameter : ℝ := 12
  let circle_radius : ℝ := circle_diameter / 2
  let square_side : ℝ := circle_diameter
  let circle_area : ℝ := Real.pi * circle_radius ^ 2
  let square_area : ℝ := square_side ^ 2
  circle_area + square_area

/-- Theorem stating that the combined area is equal to 36π + 144 -/
theorem combined_area_eq : combined_area = 36 * Real.pi + 144 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_area_eq_l230_23076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_negative_slope_one_l230_23064

/-- The angle of inclination of a line with equation y = -x + 1 is 135 degrees. -/
theorem angle_of_inclination_negative_slope_one : 
  ∀ (l : Real → Real → Prop),
  (∀ x y, l x y ↔ y = -x + 1) → 
  ∃ θ, (0 ≤ θ ∧ θ < π) ∧ θ = 135 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_negative_slope_one_l230_23064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_location_l230_23065

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1/2)^x - x^3

-- State the theorem
theorem root_location (a b c x₀ : ℝ) 
  (h1 : 0 < a) (h2 : a < b) (h3 : b < c)
  (h4 : f a * f b * f c < 0)
  (h5 : f x₀ = 0) : 
  x₀ ≥ a := by
  sorry

-- Note: The proof is omitted as per the instructions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_location_l230_23065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_CD_is_one_l230_23075

/-- A line passing through the origin that intersects y = e^(x-1) at two points -/
structure IntersectingLine where
  k : ℝ
  k_pos : k > 0

/-- Points of intersection between the line and the curve y = e^(x-1) -/
structure IntersectionPoints (l : IntersectingLine) where
  x₁ : ℝ
  x₂ : ℝ
  x₁_pos : x₁ > 0
  x₂_pos : x₂ > 0
  y₁_eq : l.k * x₁ = Real.exp (x₁ - 1)
  y₂_eq : l.k * x₂ = Real.exp (x₂ - 1)

/-- The slope of line CD -/
noncomputable def slope_CD (l : IntersectingLine) (p : IntersectionPoints l) : ℝ :=
  (Real.log p.x₂ - Real.log p.x₁) / (p.x₂ - p.x₁)

/-- Main theorem: The slope of line CD is 1 -/
theorem slope_CD_is_one (l : IntersectingLine) (p : IntersectionPoints l) :
  slope_CD l p = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_CD_is_one_l230_23075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_formation_l230_23090

/-- Checks if three lengths can form a triangle -/
def canFormTriangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- The given sets of line segments -/
def segmentSets : List (List ℝ) :=
  [[2, 5, 8], [3, 3, 6], [3, 4, 5], [1, 2, 3]]

theorem triangle_formation :
  ∃! set : List ℝ, set ∈ segmentSets ∧ 
    set.length = 3 ∧ 
    canFormTriangle set[0]! set[1]! set[2]! :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_formation_l230_23090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_parallel_lines_l230_23083

-- Define the circles and points
variable (Γ₁ Γ₂ : Set (Fin 2 → ℝ))
variable (A B C D E F : Fin 2 → ℝ)

-- Define the lines
def d₁ (A C : Fin 2 → ℝ) : Set (Fin 2 → ℝ) := {x | ∃ t : ℝ, x = A + t • (C - A)}
def d₂ (D B : Fin 2 → ℝ) : Set (Fin 2 → ℝ) := {x | ∃ t : ℝ, x = D + t • (B - D)}

-- State the conditions
variable (h₁ : A ∈ Γ₁ ∩ Γ₂)
variable (h₂ : D ∈ Γ₁ ∩ Γ₂)
variable (h₃ : C ∈ Γ₁ ∩ d₁ A C)
variable (h₄ : E ∈ Γ₂ ∩ d₁ A C)
variable (h₅ : B ∈ Γ₁ ∩ d₂ D B)
variable (h₆ : F ∈ Γ₂ ∩ d₂ D B)

-- Define parallel lines
def parallel (L₁ L₂ : Set (Fin 2 → ℝ)) : Prop :=
  ∃ v : Fin 2 → ℝ, v ≠ 0 ∧ ∀ x y, x ∈ L₁ → y ∈ L₂ → ∃ t : ℝ, y - x = t • v

-- State the theorem
theorem circles_intersect_parallel_lines :
  parallel {x | ∃ t : ℝ, x = C + t • (D - C)} {x | ∃ t : ℝ, x = E + t • (F - E)} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_parallel_lines_l230_23083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_l230_23013

theorem expression_simplification :
  ∃ (a b c : ℕ), 
    (c = 2) ∧
    (a = 255) ∧
    (b = 7) ∧
    (∀ (d : ℕ), 0 < d → d < c → 
      ¬∃ (x y : ℚ), (Real.sqrt 6 + 1 / Real.sqrt 6 + Real.sqrt 8 + 1 / Real.sqrt 8) = 
        (x * Real.sqrt 6 + y * Real.sqrt 8) / d) ∧
    (Real.sqrt 6 + 1 / Real.sqrt 6 + Real.sqrt 8 + 1 / Real.sqrt 8) = 
      (a * Real.sqrt 6 + b * Real.sqrt 8) / c :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_l230_23013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_v₁_v₂_l230_23069

noncomputable def v₁ : ℝ × ℝ × ℝ := (3, -2, 2)
noncomputable def v₂ : ℝ × ℝ × ℝ := (-2, 2, 1)

noncomputable def dot_product (a b : ℝ × ℝ × ℝ) : ℝ :=
  let (a₁, a₂, a₃) := a
  let (b₁, b₂, b₃) := b
  a₁ * b₁ + a₂ * b₂ + a₃ * b₃

noncomputable def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  let (x, y, z) := v
  Real.sqrt (x^2 + y^2 + z^2)

noncomputable def angle_between (a b : ℝ × ℝ × ℝ) : ℝ :=
  Real.arccos ((dot_product a b) / (magnitude a * magnitude b))

theorem angle_between_v₁_v₂ :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.5 ∧ 
  |angle_between v₁ v₂ * (180 / Real.pi) - 127| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_v₁_v₂_l230_23069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_integer_in_sequence_l230_23020

def sequenceNum (n : ℕ) : ℚ :=
  (1000000 : ℚ) / 2^n

def is_last_integer (k : ℤ) : Prop :=
  ∃ n : ℕ, (sequenceNum n).num = k ∧ (sequenceNum (n + 1)).num ≠ (sequenceNum (n + 1)).den * (sequenceNum (n + 1)).num

theorem last_integer_in_sequence :
  is_last_integer 15625 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_integer_in_sequence_l230_23020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_unbounded_l230_23002

def sequenceA (a₀ : ℕ) : ℕ → ℕ
  | 0 => a₀
  | n + 1 =>
    let aₙ := sequenceA a₀ n
    if aₙ % 2 = 1 then aₙ * aₙ - 5 else aₙ / 2

theorem sequence_unbounded (a₀ : ℕ) (h₀ : a₀ % 2 = 1) (h₁ : a₀ > 5) :
  ∀ M : ℕ, ∃ N : ℕ, sequenceA a₀ N > M :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_unbounded_l230_23002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functions_exist_l230_23048

-- Define the functions and their domains
def f1 (x : ℝ) : ℝ := 0.5 * x
def domain1 : Set ℝ := {x | x > 0}

def f2 (x : ℝ) : ℝ := -x^2
def domain2 : Set ℝ := Set.Ioo (-1) 2 -- Changed Ioc to Set.Ioo

def f3 (x : ℝ) : ℝ := x^3
def domain3 : Set ℝ := {x | |x| ≤ 1.5}

def f4 (x : ℝ) : ℝ := x^2
def domain4 : Set ℝ := {x | |x| > 1}

-- Theorem stating the existence of these functions on their respective domains
theorem functions_exist :
  (∀ x ∈ domain1, ∃ y, y = f1 x) ∧
  (∀ x ∈ domain2, ∃ y, y = f2 x) ∧
  (∀ x ∈ domain3, ∃ y, y = f3 x) ∧
  (∀ x ∈ domain4, ∃ y, y = f4 x) :=
by
  sorry -- Skipping the proof as suggested


end NUMINAMATH_CALUDE_ERRORFEEDBACK_functions_exist_l230_23048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_b_days_l230_23050

/-- Proves that given the conditions of the problem, worker b worked for 9 days -/
theorem worker_b_days : 
  ∀ (days_b : ℕ),
  (let daily_wage_c : ℕ := 110
   let daily_wage_b : ℕ := (4 * daily_wage_c) / 5
   let daily_wage_a : ℕ := (3 * daily_wage_c) / 5
   let total_earning : ℕ := 1628
   let days_a : ℕ := 6
   let days_c : ℕ := 4
   days_a * daily_wage_a + days_b * daily_wage_b + days_c * daily_wage_c = total_earning) → 
  days_b = 9 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_b_days_l230_23050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_third_quadrant_l230_23045

theorem tan_alpha_third_quadrant (α : ℝ) :
  Real.sin (π + α) = 3/5 ∧ π < α ∧ α < 3*π/2 → Real.tan α = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_third_quadrant_l230_23045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_five_power_plus_four_l230_23072

theorem perfect_square_five_power_plus_four :
  ∀ n : ℕ, n > 0 → (∃ a : ℕ, 5^n + 4 = a^2) ↔ n = 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_five_power_plus_four_l230_23072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_sum_and_ratio_l230_23023

/-- Represents a triangle with vertices A, B, and C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Calculates the perimeter of a triangle -/
def perimeter (t : Triangle) : ℝ := sorry

/-- Constructs a λ-ratio inscribed triangle -/
def inscribedTriangle (t : Triangle) (l : ℝ) : Triangle := sorry

/-- Generates the nth triangle in the sequence -/
def nthTriangle (t : Triangle) (l : ℝ) (n : ℕ) : Triangle := sorry

/-- Theorem: Sum of perimeters and ratio of perimeters -/
theorem triangle_perimeter_sum_and_ratio (H : Triangle) (l : ℝ) (h_l : l > 0) :
  (∃ (K : ℝ), K = (l + 1)^2 / (3 * l) ∧
    (∑' n, perimeter (nthTriangle H l n)) = K) ∧
  (∀ n : ℕ, 
    perimeter (nthTriangle H l n) / perimeter (inscribedTriangle (nthTriangle H l n) l) =
    perimeter H / perimeter (inscribedTriangle H l)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_sum_and_ratio_l230_23023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_speed_for_on_time_arrival_l230_23010

/-- Represents the scenario of Mrs. Ada Late's commute to work -/
structure CommuteScenario where
  distance : ℝ  -- Distance to work in miles
  ideal_time : ℝ  -- Ideal time to reach work in hours

/-- Calculates the travel time given speed and distance -/
noncomputable def travel_time (speed : ℝ) (distance : ℝ) : ℝ := distance / speed

/-- Theorem stating the correct speed for Mrs. Late to arrive exactly on time -/
theorem correct_speed_for_on_time_arrival (scenario : CommuteScenario) : 
  ∃ (speed : ℝ), 
    (travel_time 30 scenario.distance = scenario.ideal_time + 1/15) ∧ 
    (travel_time 50 scenario.distance = scenario.ideal_time - 1/30) → 
    (travel_time speed scenario.distance = scenario.ideal_time ∧ speed = 41) := by
  sorry

#check correct_speed_for_on_time_arrival

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_speed_for_on_time_arrival_l230_23010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_number_with_remainder_l230_23034

theorem least_number_with_remainder : 
  ∃! n : ℕ, (∀ d ∈ ({12, 15, 20, 54} : Set ℕ), n % d = 5) ∧
            (∀ m < n, ∃ d ∈ ({12, 15, 20, 54} : Set ℕ), m % d ≠ 5) :=
by
  -- The proof goes here
  sorry

#check least_number_with_remainder

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_number_with_remainder_l230_23034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_angle_l230_23044

/-- The hyperbola equation -/
noncomputable def hyperbola (m : ℝ) (x y : ℝ) : Prop :=
  x^2 / (m + 4) - y^2 / m = 1

/-- The eccentricity of the hyperbola -/
noncomputable def eccentricity : ℝ := 2 * Real.sqrt 3 / 3

/-- The angle between the asymptotes -/
noncomputable def angle_between_asymptotes (m : ℝ) : ℝ :=
  2 * Real.arctan (Real.sqrt (m / (m + 4)))

theorem hyperbola_asymptote_angle (m : ℝ) :
  (∀ x y, hyperbola m x y) → eccentricity = Real.sqrt (1 + m / (m + 4)) →
  angle_between_asymptotes m = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_angle_l230_23044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_continuous_midpoint_property_implies_constant_l230_23089

theorem continuous_midpoint_property_implies_constant
  (f : ℝ → ℝ) (hf : Continuous f)
  (h : ∀ a b : ℝ, f ((a + b) / 2) ∈ ({f a, f b} : Set ℝ)) :
  ∃ c : ℝ, ∀ x : ℝ, f x = c :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_continuous_midpoint_property_implies_constant_l230_23089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l230_23087

-- Define the ellipse parameters
noncomputable def a : ℝ := Real.sqrt 2
def b : ℝ := 1
noncomputable def e : ℝ := Real.sqrt 2 / 2

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the focus
def focus : ℝ × ℝ := (1, 0)

-- Define a line passing through the focus
def line_through_focus (k : ℝ) (x y : ℝ) : Prop := y = k * (x - focus.1)

theorem ellipse_properties :
  -- (I) The equation of the ellipse
  (∀ x y : ℝ, ellipse_equation x y ↔ x^2 / 2 + y^2 = 1) ∧
  -- (II) Area of triangle POQ when slope of l is 1
  (∃ P Q : ℝ × ℝ, 
    ellipse_equation P.1 P.2 ∧ 
    ellipse_equation Q.1 Q.2 ∧
    line_through_focus 1 P.1 P.2 ∧
    line_through_focus 1 Q.1 Q.2 ∧
    abs (P.2 - Q.2) / 2 = 2/3) ∧
  -- (III) Equation of line l when parallelogram OPOQ is a rectangle
  (∃ k : ℝ, k^2 = 2 ∧ 
    ∀ x y : ℝ, line_through_focus k x y ↔ y = k * (x - 1) ∨ y = -k * (x - 1)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l230_23087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_volume_of_right_circular_cylinders_l230_23096

/-- The volume of the intersection of two right circular cylinders of radius a, intersecting at a right angle. -/
noncomputable def intersection_volume (a : ℝ) : ℝ := (16 / 3) * a^3

/-- The volume of the intersection of two right circular cylinders of radius a, intersecting at a right angle. This is a placeholder function representing the actual geometric calculation. -/
noncomputable def volume_of_intersection_of_cylinders (a : ℝ) : ℝ := sorry

/-- Theorem stating that the volume of the intersection of two right circular cylinders of radius a, intersecting at a right angle, is (16/3) * a^3. -/
theorem intersection_volume_of_right_circular_cylinders (a : ℝ) (ha : a > 0) :
  ∃ (V : ℝ), V = intersection_volume a ∧ V = volume_of_intersection_of_cylinders a :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_volume_of_right_circular_cylinders_l230_23096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_from_slope_product_ellipse_focus_line_slope_range_l230_23078

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a point on the ellipse -/
structure PointOnEllipse (e : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := Real.sqrt (1 - e.b^2 / e.a^2)

/-- Theorem about the eccentricity and slope product condition of an ellipse -/
theorem ellipse_eccentricity_from_slope_product 
  (e : Ellipse) 
  (A B P : PointOnEllipse e)
  (h_distinct : P ≠ A ∧ P ≠ B)
  (h_line : ∃ (k : ℝ), A.y = k * A.x ∧ B.y = k * B.x)
  (h_slope_product : (P.y - A.y) / (P.x - A.x) * (P.y - B.y) / (P.x - B.x) = -1/4) :
  eccentricity e = Real.sqrt 3 / 2 := by
  sorry

/-- Theorem about the range of k for a line passing through the right focus -/
theorem ellipse_focus_line_slope_range 
  (e : Ellipse)
  (h_eccentricity : eccentricity e = Real.sqrt 3 / 2)
  (k : ℝ)
  (h_line : ∃ (M N : PointOnEllipse e), 
    N.y - M.y = k * (N.x - M.x) ∧ 
    M.x + N.x = 2 * Real.sqrt 3 * e.b)
  (h_left_focus_inside : 
    let c := Real.sqrt (e.a^2 - e.b^2);
    ∀ (M N : PointOnEllipse e), (M.x + c)^2 + M.y^2 < ((M.x - N.x)^2 + (M.y - N.y)^2) / 4) :
  -Real.sqrt 47 / 47 < k ∧ k < Real.sqrt 47 / 47 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_from_slope_product_ellipse_focus_line_slope_range_l230_23078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l230_23018

/-- Calculates the speed of a train in km/h given its length and time to pass a point -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  (length / 1000) / time * 3600

/-- Theorem stating the speed of the train given the conditions -/
theorem train_speed_calculation :
  let length : ℝ := 140
  let time : ℝ := 5.142857142857143
  abs (train_speed length time - 97.96) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l230_23018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_arrangement_exists_l230_23057

/-- Represents a position on the board -/
structure Position where
  x : Nat
  y : Nat

/-- Represents a spy's direction -/
inductive Direction where
  | Up
  | Down
  | Left
  | Right

/-- Represents a spy on the board -/
structure Spy where
  pos : Position
  dir : Direction

/-- Checks if a position is within the 6x6 board -/
def isValidPosition (p : Position) : Prop :=
  0 ≤ p.x ∧ p.x < 6 ∧ 0 ≤ p.y ∧ p.y < 6

/-- Returns the set of positions a spy can see -/
def visiblePositions (s : Spy) : Set Position :=
  sorry

/-- Checks if two spies can see each other -/
def canSeeEachOther (s1 s2 : Spy) : Prop :=
  s2.pos ∈ visiblePositions s1 ∨ s1.pos ∈ visiblePositions s2

/-- A valid arrangement of spies -/
def validArrangement (spies : List Spy) : Prop :=
  spies.length = 18 ∧
  (∀ s, s ∈ spies → isValidPosition s.pos) ∧
  (∀ s1 s2, s1 ∈ spies → s2 ∈ spies → s1 ≠ s2 → ¬canSeeEachOther s1 s2)

/-- Theorem stating that a valid arrangement exists -/
theorem valid_arrangement_exists : ∃ spies : List Spy, validArrangement spies :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_arrangement_exists_l230_23057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l230_23025

noncomputable def arithmeticSequence (n : ℕ) : ℝ := 4 * n - 2

noncomputable def S (n : ℕ) : ℝ := n * (arithmeticSequence 1 + arithmeticSequence n) / 2

theorem arithmetic_sequence_properties :
  -- Given conditions
  let a₁ := arithmeticSequence 1
  let a₂ := arithmeticSequence 2
  let a₅ := arithmeticSequence 5
  -- a₁ = 2
  (a₁ = 2) →
  -- a₁, a₂, a₅ form a geometric sequence
  (a₁ * a₅ = a₂^2) →
  -- Prove the general term formula
  (∀ n : ℕ, arithmeticSequence n = 4 * n - 2) ∧
  -- Prove the smallest positive integer n such that S_n > 60n + 800 is 41
  (∀ k : ℕ, k < 41 → S k ≤ 60 * k + 800) ∧ (S 41 > 60 * 41 + 800) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l230_23025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_athlete_walking_rate_l230_23060

/-- An athlete's calorie burning rates and exercise duration --/
structure AthleteExercise where
  runningRate : ℚ  -- Calories burned per minute while running
  totalTime : ℚ    -- Total exercise time in minutes
  runningTime : ℚ  -- Time spent running in minutes
  totalCalories : ℚ -- Total calories burned

/-- Calculate the calories burned per minute while walking --/
def walkingRate (a : AthleteExercise) : ℚ :=
  (a.totalCalories - a.runningRate * a.runningTime) / (a.totalTime - a.runningTime)

/-- Theorem: Given the conditions, the athlete burns 4 calories per minute while walking --/
theorem athlete_walking_rate (a : AthleteExercise) 
  (h1 : a.runningRate = 10)
  (h2 : a.totalTime = 60)
  (h3 : a.runningTime = 35)
  (h4 : a.totalCalories = 450) :
  walkingRate a = 4 := by
  sorry

def main : IO Unit := do
  let result := walkingRate { runningRate := 10, totalTime := 60, runningTime := 35, totalCalories := 450 }
  IO.println s!"Walking rate: {result}"

#eval main

end NUMINAMATH_CALUDE_ERRORFEEDBACK_athlete_walking_rate_l230_23060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_c_is_right_angle_l230_23035

theorem triangle_angle_c_is_right_angle (A B C : ℝ) 
  (h : |Real.sin B - 1/2| + (Real.tan A - Real.sqrt 3)^2 = 0) : 
  C = 90 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_c_is_right_angle_l230_23035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_locus_is_ellipse_l230_23015

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  center : Point2D
  semiMajorAxis : ℝ
  semiMinorAxis : ℝ

/-- Represents the locus of midpoints -/
noncomputable def midpointLocus (e : Ellipse) (p : Point2D) : Ellipse :=
  { center := { x := (p.x + e.center.x) / 2, y := (p.y + e.center.y) / 2 },
    semiMajorAxis := e.semiMajorAxis / 2,
    semiMinorAxis := e.semiMinorAxis / 2 }

/-- The main theorem -/
theorem midpoint_locus_is_ellipse (e : Ellipse) (p : Point2D) :
  ∃ (resultEllipse : Ellipse),
    resultEllipse = midpointLocus e p ∧
    resultEllipse.semiMajorAxis = e.semiMajorAxis / 2 ∧
    resultEllipse.semiMinorAxis = e.semiMinorAxis / 2 := by
  -- Construct the result ellipse
  let resultEllipse := midpointLocus e p
  -- Prove existence
  use resultEllipse
  -- Prove the three conditions
  constructor
  · rfl  -- reflexivity proves equality
  constructor
  · rfl
  · rfl

#check midpoint_locus_is_ellipse

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_locus_is_ellipse_l230_23015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_of_11025_l230_23066

theorem largest_prime_factor_of_11025 : 
  (Nat.factors 11025).maximum? = some 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_of_11025_l230_23066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_trinomial_condition_l230_23095

-- Define a perfect square trinomial
def is_perfect_square_trinomial (a b c : ℝ) : Prop :=
  ∃ (p q : ℝ), ∀ (x : ℝ), a * x^2 + b * x + c = (p * x + q)^2

-- State the theorem
theorem perfect_square_trinomial_condition :
  ∀ k : ℝ, is_perfect_square_trinomial 4 k 1 → k = 4 ∨ k = -4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_trinomial_condition_l230_23095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_property_l230_23029

-- Define a point in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a distance function between two points
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

-- Define a hyperbola
def isHyperbola (trajectory : Set Point) : Prop :=
  ∃ (F₁ F₂ : Point) (a : ℝ), a > 0 ∧
    ∀ M ∈ trajectory, |distance M F₁ - distance M F₂| = 2*a

-- Define the constant difference property
def hasConstantDifference (trajectory : Set Point) : Prop :=
  ∃ (F₁ F₂ : Point) (k : ℝ), ∀ M ∈ trajectory, |distance M F₁ - distance M F₂| = k

-- State the theorem
theorem hyperbola_property (trajectory : Set Point) :
  isHyperbola trajectory → hasConstantDifference trajectory ∧
  ¬(hasConstantDifference trajectory → isHyperbola trajectory) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_property_l230_23029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_geq_2_l230_23039

/-- A random variable following a normal distribution with mean 1 and variance σ² -/
noncomputable def X : ℝ → ℝ := sorry

/-- The probability density function of X -/
noncomputable def f : ℝ → ℝ := sorry

/-- X follows a normal distribution with mean 1 and some variance σ² -/
axiom normal_dist : ∃ σ : ℝ, ∀ x, f x = (1 / (σ * Real.sqrt (2 * Real.pi))) * Real.exp (-(1 / 2) * ((x - 1) / σ)^2)

/-- The probability that X is between 0 and 1 is 0.3 -/
axiom prob_between_0_and_1 : ∫ x in Set.Icc 0 1, f x = 0.3

/-- The theorem to be proved -/
theorem prob_geq_2 : ∫ x in Set.Ici 2, f x = 0.2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_geq_2_l230_23039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trihedral_angle_symmetry_l230_23054

/-- Represents a point in 3D space -/
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a line in 3D space -/
structure Line where
  point : Point
  direction : Point

/-- Represents a plane in 3D space -/
structure Plane where
  normal : Point
  d : ℝ

/-- Represents a trihedral angle with vertex S and edges a, b, c -/
structure TrihedralAngle where
  S : Point
  a : Line
  b : Line
  c : Line

/-- Defines the bisector plane of a dihedral angle -/
def bisectorPlane (edge : Line) : Plane :=
  sorry

/-- Defines the symmetric line with respect to a bisector plane -/
def symmetricLine (l : Line) (bisectorPlane : Plane) : Line :=
  sorry

/-- Checks if three lines are coplanar -/
def areCoplanar (l1 l2 l3 : Line) : Prop :=
  sorry

/-- Defines a plane passing through a line and a point -/
def plane (l : Line) (p : Point) : Plane :=
  sorry

/-- Checks if three planes intersect in a single line -/
def intersectInSingleLine (p1 p2 p3 : Plane) : Prop :=
  sorry

/-- Main theorem statement -/
theorem trihedral_angle_symmetry (T : TrihedralAngle) 
  (α β γ : Line) 
  (α' β' γ' : Line) :
  (α' = symmetricLine α (bisectorPlane T.a)) →
  (β' = symmetricLine β (bisectorPlane T.b)) →
  (γ' = symmetricLine γ (bisectorPlane T.c)) →
  (areCoplanar α β γ ↔ areCoplanar α' β' γ') ∧
  (intersectInSingleLine (plane T.a T.S) (plane T.b T.S) (plane T.c T.S) ↔
   intersectInSingleLine (plane T.a T.S) (plane T.b T.S) (plane T.c T.S)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trihedral_angle_symmetry_l230_23054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_of_geometric_progression_l230_23038

def geometric_progression (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * r ^ (n - 1)

theorem fifth_term_of_geometric_progression (a₁ a₂ a₃ : ℝ) (h₁ : a₁ = Real.sqrt 4)
    (h₂ : a₂ = (4 : ℝ) ^ (1/6)) (h₃ : a₃ = (4 : ℝ) ^ (1/12)) 
    (h₄ : ∃ r : ℝ, a₂ = a₁ * r ∧ a₃ = a₂ * r) :
  geometric_progression a₁ ((4 : ℝ) ^ (-1/12 : ℝ)) 5 = (4 : ℝ) ^ (-1/12 : ℝ) := by
  sorry

#check fifth_term_of_geometric_progression

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_of_geometric_progression_l230_23038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jo_alex_sum_difference_l230_23079

def jo_sum (n : ℕ) : ℕ := n * (n + 1) / 2

def round_to_20 (n : ℕ) : ℕ :=
  if n % 20 ≤ 10 then (n / 20) * 20 else ((n / 20) + 1) * 20

def alex_sum (n : ℕ) : ℕ :=
  (List.range n).map round_to_20
    |> List.sum

theorem jo_alex_sum_difference :
  (jo_sum 100 : ℤ) - (alex_sum 100 : ℤ) = 4050 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jo_alex_sum_difference_l230_23079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_distance_between_stations_stations_initial_distance_l230_23058

/-- 
Given two trains A and B traveling towards each other at the same speed,
prove that the initial distance between their starting stations is twice
the distance traveled by one train when they meet.
-/
theorem initial_distance_between_stations 
  (speed_A speed_B : ℝ) 
  (distance_A : ℝ) 
  (h1 : speed_A = speed_B) 
  (h2 : speed_A > 0) 
  (h3 : distance_A > 0) : 
  2 * distance_A = speed_A * (distance_A / speed_A) + speed_B * (distance_A / speed_A) :=
by
  sorry

/-- 
Specific case where Train A and Train B both travel at 20 miles per hour,
and Train A travels 100 miles before they meet.
-/
theorem stations_initial_distance : 
  (2 : ℝ) * 100 = 20 * (100 / 20) + 20 * (100 / 20) :=
by
  norm_num

#eval (2 : ℝ) * 100 -- This will evaluate to 200

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_distance_between_stations_stations_initial_distance_l230_23058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_M_inside_curve_C_rotated_M_on_curve_C_l230_23008

-- Define the curve C
noncomputable def curve_C (α : ℝ) : ℝ × ℝ := (2 * Real.cos α, 3 * Real.sin α)

-- Define point M
noncomputable def point_M : ℝ × ℝ := (Real.sqrt 2, Real.sqrt 2)

-- Define the rotation function
noncomputable def rotate (θ : ℝ) (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 * Real.cos θ - p.2 * Real.sin θ, p.1 * Real.sin θ + p.2 * Real.cos θ)

theorem point_M_inside_curve_C :
  (point_M.1)^2 / 4 + (point_M.2)^2 / 9 < 1 := by
  sorry

theorem rotated_M_on_curve_C :
  ∃ α : ℝ, rotate (3 * π / 4) (2 * Real.cos (π / 4), 2 * Real.sin (π / 4)) = curve_C α := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_M_inside_curve_C_rotated_M_on_curve_C_l230_23008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_number_l230_23026

def is_valid_number (n : Nat) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧
  (∀ d, d ∈ n.digits 10 → 2 ≤ d) ∧
  (∀ d, d ∈ n.digits 10 → d ∣ n) ∧
  (n.digits 10).Nodup

theorem smallest_valid_number :
  (∀ m, is_valid_number m → 2460 ≤ m) ∧ is_valid_number 2460 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_number_l230_23026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_2_sqrt_7_l230_23093

/-- The line passing through the circle -/
def line (x y : ℝ) : Prop := x + y - 8 = 0

/-- The circle with center (3, -1) and radius 5 -/
def circle_equation (x y : ℝ) : Prop := (x - 3)^2 + (y + 1)^2 = 25

/-- The distance from the center of the circle to the line -/
noncomputable def distance_center_to_line : ℝ := |1 * 3 + 1 * (-1) - 8| / Real.sqrt 2

/-- The length of the chord cut by the circle from the line -/
noncomputable def chord_length : ℝ := 2 * Real.sqrt (25 - distance_center_to_line^2)

theorem chord_length_is_2_sqrt_7 : chord_length = 2 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_2_sqrt_7_l230_23093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_shift_l230_23041

theorem sin_cos_shift (x : ℝ) :
  Real.cos (2*x + π/3) = Real.sin (2*(x + 5*π/12)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_shift_l230_23041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_above_g_l230_23012

-- Define the functions f and g
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.exp x / (x - m)
def g (x : ℝ) : ℝ := x^2 + x

-- State the theorem
theorem f_above_g (m : ℝ) (h1 : 0 < m) (h2 : m < 1/2) :
  ∀ x ∈ Set.Icc m (m + 1), f m x > g x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_above_g_l230_23012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l230_23061

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := 6 / x - Real.log x / Real.log 2

-- Theorem statement
theorem zero_in_interval :
  ∃ c ∈ Set.Ioo 2 4, f c = 0 :=
by
  sorry -- Proof is omitted as per instructions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l230_23061
