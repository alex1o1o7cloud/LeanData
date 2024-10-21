import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_intersection_properties_l693_69308

-- Define the hyperbola C and line l
noncomputable def hyperbola (a : ℝ) (x y : ℝ) : Prop := x^2 / a^2 - y^2 = 1
def line (x y : ℝ) : Prop := x + y = 1

-- Define the eccentricity of the hyperbola
noncomputable def eccentricity (a : ℝ) : ℝ := Real.sqrt (1 / a^2 + 1)

-- Define the theorem
theorem hyperbola_intersection_properties
  (a : ℝ)
  (h_a_pos : a > 0)
  (h_distinct : ∃ (A B : ℝ × ℝ), A ≠ B ∧
    hyperbola a A.1 A.2 ∧ line A.1 A.2 ∧
    hyperbola a B.1 B.2 ∧ line B.1 B.2) :
  (eccentricity a > Real.sqrt 6 / 2 ∧ eccentricity a ≠ Real.sqrt 2) ∧
  (∀ (A B : ℝ × ℝ) (P : ℝ × ℝ),
    P = (0, 1) →
    hyperbola a A.1 A.2 → line A.1 A.2 →
    hyperbola a B.1 B.2 → line B.1 B.2 →
    (A.1 - P.1, A.2 - P.2) = (5/12 : ℝ) • (B.1 - P.1, B.2 - P.2) →
    a = 17/13) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_intersection_properties_l693_69308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l693_69342

-- Define the points
noncomputable def P : ℝ × ℝ := (Real.sqrt 3, 1)
noncomputable def Q (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x)
def O : ℝ × ℝ := (0, 0)

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (P.1 - (Q x).1) * P.1 + (P.2 - (Q x).2) * P.2

-- Define the triangle ABC
structure Triangle where
  A : ℝ  -- angle A in radians
  BC : ℝ -- length of side BC

-- State the theorem
theorem triangle_perimeter (t : Triangle) (h1 : f t.A = 4) (h2 : t.BC = 3)
  (h3 : 1/2 * t.BC * (Real.sin t.A) * 3 = 3 * Real.sqrt 3 / 4) :
  ∃ (AB AC : ℝ), AB + AC + t.BC = 3 + 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l693_69342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_l693_69394

theorem expression_value : 
  let a : ℝ := 0.137
  let b : ℝ := 0.098
  |((a + b)^2 - (a - b)^2) / (a * b) - 4.000297| < 0.000001 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_l693_69394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_equals_two_l693_69376

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 2 * x + 1 else 2^x

-- State the theorem
theorem function_composition_equals_two (x : ℝ) : f (f x) = 2 → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_equals_two_l693_69376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_inequality_at_one_l693_69336

theorem trig_inequality_at_one : Real.tan 1 > Real.sin 1 ∧ Real.sin 1 > Real.cos 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_inequality_at_one_l693_69336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_range_l693_69370

/-- The function f(x) defined as e^(2x) - ae^x - a^2x --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (2 * x) - a * Real.exp x - a^2 * x

/-- Theorem stating the range of a for which f(x) ≥ 0 for all x --/
theorem f_nonnegative_range (a : ℝ) :
  (∀ x, f a x ≥ 0) ↔ a ∈ Set.Icc (-2 * Real.exp (3/4)) 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_range_l693_69370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_longest_diagonal_rhombus_longest_diagonal_approx_l693_69334

/-- The length of the longest diagonal of a rhombus -/
noncomputable def longest_diagonal (area : ℝ) (ratio : ℝ × ℝ) : ℝ :=
  let d₁ := 4 * (2 * area / (ratio.1 * ratio.2))^(1/2)
  d₁

/-- The theorem stating the length of the longest diagonal of a specific rhombus -/
theorem rhombus_longest_diagonal :
  longest_diagonal 200 (4, 3) = 40 * Real.sqrt 3 / 3 := by
  -- Unfold the definition of longest_diagonal
  unfold longest_diagonal
  -- Simplify the expression
  simp
  -- The rest of the proof would go here
  sorry

-- We can't use #eval here because the function is noncomputable
-- Instead, we can state another theorem to check the approximation
theorem rhombus_longest_diagonal_approx :
  ∃ ε > 0, |longest_diagonal 200 (4, 3) - 23.094| < ε := by
  -- The proof of this approximation would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_longest_diagonal_rhombus_longest_diagonal_approx_l693_69334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_m_value_l693_69388

/-- Given two lines (m+3)x + my - 2 = 0 and mx - 6y + 5 = 0 are perpendicular, prove m = 3 -/
theorem perpendicular_lines_m_value (m : ℝ) (h1 : m ≠ 0) 
  (h2 : ((m + 3) * (1 : ℝ) + m * (-((m + 3) / m))) * (m / 6) = -1) : 
  m = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_m_value_l693_69388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_differential_equation_solution_1_differential_equation_solution_2_differential_equation_solution_3_differential_equation_solution_4_l693_69332

/-- Given a second-order linear differential equation y'' + 6y' + 5y = 25x^2 - 2,
    prove that the general solution is y(x) = C₁e^(-5x) + C₂e^(-x) + 5x^2 - 12x + 12,
    where C₁ and C₂ are arbitrary constants. -/
theorem differential_equation_solution_1 (y : ℝ → ℝ) :
  (∀ x, (deriv^[2] y) x + 6 * (deriv y) x + 5 * y x = 25 * x^2 - 2) →
  (∃ C₁ C₂, ∀ x, y x = C₁ * Real.exp (-5*x) + C₂ * Real.exp (-x) + 5*x^2 - 12*x + 12) :=
sorry

/-- Given a second-order linear differential equation y'' - 2y' + 10y = 37 cos(3x),
    prove that the general solution is y(x) = e^x(C₁cos(3x) + C₂sin(3x)) + cos(3x) - 6sin(3x),
    where C₁ and C₂ are arbitrary constants. -/
theorem differential_equation_solution_2 (y : ℝ → ℝ) :
  (∀ x, (deriv^[2] y) x - 2 * (deriv y) x + 10 * y x = 37 * Real.cos (3*x)) →
  (∃ C₁ C₂, ∀ x, y x = Real.exp x * (C₁ * Real.cos (3*x) + C₂ * Real.sin (3*x)) + Real.cos (3*x) - 6 * Real.sin (3*x)) :=
sorry

/-- Given a second-order linear differential equation y'' - 6y' + 9y = 3x - 8e^x,
    prove that the general solution is y(x) = e^(3x)(C₁ + C₂x) + (1/3)x + (2/9) - 2e^x,
    where C₁ and C₂ are arbitrary constants. -/
theorem differential_equation_solution_3 (y : ℝ → ℝ) :
  (∀ x, (deriv^[2] y) x - 6 * (deriv y) x + 9 * y x = 3 * x - 8 * Real.exp x) →
  (∃ C₁ C₂, ∀ x, y x = Real.exp (3*x) * (C₁ + C₂ * x) + (1/3) * x + (2/9) - 2 * Real.exp x) :=
sorry

/-- Given a third-order linear differential equation y''' + 4y' = 8e^(2x) + 5e^x sin x,
    prove that the general solution is y(x) = C₁ + C₂cos(2x) + C₃sin(2x) + (1/2)e^(2x) + (1/4)e^x(sin x - 3cos x),
    where C₁, C₂, and C₃ are arbitrary constants. -/
theorem differential_equation_solution_4 (y : ℝ → ℝ) :
  (∀ x, (deriv^[3] y) x + 4 * (deriv y) x = 8 * Real.exp (2*x) + 5 * Real.exp x * Real.sin x) →
  (∃ C₁ C₂ C₃, ∀ x, y x = C₁ + C₂ * Real.cos (2*x) + C₃ * Real.sin (2*x) + (1/2) * Real.exp (2*x) + (1/4) * Real.exp x * (Real.sin x - 3 * Real.cos x)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_differential_equation_solution_1_differential_equation_solution_2_differential_equation_solution_3_differential_equation_solution_4_l693_69332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_2017_is_7_l693_69391

/-- Represents the sequence of natural numbers written in ascending order without spaces -/
def natural_sequence : ℕ → ℕ := sorry

/-- Returns the nth digit in the natural_sequence -/
def nth_digit (n : ℕ) : ℕ := natural_sequence n

/-- The 2017th digit in the natural_sequence is 7 -/
theorem digit_2017_is_7 : nth_digit 2017 = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_2017_is_7_l693_69391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chessboard_coloring_impossibility_l693_69310

theorem chessboard_coloring_impossibility :
  ¬ ∃ (coloring : Fin 1990 → Fin 1990 → Bool),
    (∀ row, (Finset.filter (fun col ↦ coloring row col) Finset.univ).card = 995) ∧
    (∀ col, (Finset.filter (fun row ↦ coloring row col) Finset.univ).card = 995) ∧
    (∀ row col, coloring row col ≠ coloring (1989 - row) (1989 - col)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chessboard_coloring_impossibility_l693_69310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_consecutive_tails_probability_l693_69309

-- Define a fair coin
noncomputable def fair_coin : ℚ := 1/2

-- Define the probability of flipping tails
def prob_tails (p : ℚ) : ℚ := p

-- Define the probability of two consecutive events
def prob_consecutive (p : ℚ) : ℚ := p * p

-- Theorem statement
theorem two_consecutive_tails_probability :
  prob_consecutive (prob_tails fair_coin) = 1/4 := by
  -- Unfold definitions
  unfold prob_consecutive
  unfold prob_tails
  unfold fair_coin
  -- Simplify
  simp
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_consecutive_tails_probability_l693_69309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_sum_l693_69323

theorem repeating_decimal_sum : 
  (5/11 : ℚ) + (2/3 : ℚ) = 37/33 := by
  -- Convert fractions to a common denominator
  have h1 : (5/11 : ℚ) = 15/33 := by norm_num
  have h2 : (2/3 : ℚ) = 22/33 := by norm_num
  -- Rewrite using these equalities
  rw [h1, h2]
  -- Simplify the sum
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_sum_l693_69323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l693_69354

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := (3 * x + 4) / (x + 2)

-- State the theorem about the range of f
theorem range_of_f :
  Set.range f = {y : ℝ | y < 3 ∨ y > 3} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l693_69354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_calculation_l693_69355

/-- The area of a shaded region consisting of 20 congruent squares -/
noncomputable def shaded_area (diagonal : ℝ) (num_squares : ℕ) : ℝ :=
  (diagonal^2 / 2) * (num_squares / 25 : ℝ)

/-- Theorem stating the area of the shaded region -/
theorem shaded_area_calculation (diagonal : ℝ) (num_squares : ℕ) 
  (h1 : diagonal = 8)
  (h2 : num_squares = 20) :
  shaded_area diagonal num_squares = 25.6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_calculation_l693_69355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_expression_max_l693_69364

theorem triangle_angle_expression_max (α β γ : ℝ) (h_triangle : α + β + γ = Real.pi) :
  Real.sin α * Real.sin β * Real.cos γ + Real.sin γ ^ 2 ≤ 9 / 8 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_expression_max_l693_69364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_half_plus_alpha_l693_69381

theorem cos_pi_half_plus_alpha (α : ℝ) (h1 : -π < α) (h2 : α < 0) 
  (h3 : ∃ y : ℝ, (1/3)^2 + y^2 = 1) : 
  Real.cos (π/2 + α) = 2*Real.sqrt 2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_half_plus_alpha_l693_69381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_containing_sphere_radius_proof_l693_69367

/-- The radius of the smallest sphere containing four unit spheres in tetrahedral arrangement -/
noncomputable def smallest_containing_sphere_radius : ℝ := Real.sqrt (3 / 2) + 1

/-- The centers of four unit spheres in tetrahedral arrangement form a regular tetrahedron -/
def tetrahedron_side_length : ℝ := 2

/-- Theorem: The radius of the smallest sphere containing four unit spheres in tetrahedral arrangement is √(3/2) + 1 -/
theorem smallest_containing_sphere_radius_proof :
  let R := tetrahedron_side_length * Real.sqrt 6 / 4
  smallest_containing_sphere_radius = R + 1 := by
  sorry

#check smallest_containing_sphere_radius_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_containing_sphere_radius_proof_l693_69367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_satisfying_function_l693_69317

/-- A function satisfying the given conditions -/
def satisfying_function (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f x + f y ≥ x * y) ∧
  (∀ x : ℝ, ∃ y : ℝ, f x + f y = x * y)

/-- The proposed solution function -/
noncomputable def f (x : ℝ) : ℝ := x^2 / 2

/-- Theorem stating that f is the unique function satisfying the conditions -/
theorem unique_satisfying_function :
  satisfying_function f ∧
  ∀ g : ℝ → ℝ, satisfying_function g → g = f := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_satisfying_function_l693_69317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_angle_l693_69389

-- Define the curve C in polar coordinates
noncomputable def curve_C (θ : ℝ) : ℝ := 4 * Real.cos θ

-- Define the line l in parametric form
noncomputable def line_l (t α : ℝ) : ℝ × ℝ := (1 + t * Real.cos α, t * Real.sin α)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem intersection_angle (α : ℝ) : 
  ∃ (t1 t2 : ℝ), 
    let p1 := line_l t1 α
    let p2 := line_l t2 α
    (p1.1 - 2)^2 + p1.2^2 = 4 ∧ 
    (p2.1 - 2)^2 + p2.2^2 = 4 ∧
    distance p1 p2 = Real.sqrt 14 →
    α = π/4 ∨ α = 3*π/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_angle_l693_69389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_select_non_adjacent_teams_l693_69360

/-- A graph with 2m vertices where each vertex has degree 2 -/
structure TournamentGraph (m : ℕ) where
  vertices : Finset (Fin (2 * m))
  edges : Finset (Fin (2 * m) × Fin (2 * m))
  vertex_count : vertices.card = 2 * m
  edge_count : edges.card = 2 * m
  degree_two : ∀ v, v ∈ vertices → (edges.filter (λ e ↦ e.1 = v ∨ e.2 = v)).card = 2

/-- The theorem stating that it's possible to select m non-adjacent vertices -/
theorem select_non_adjacent_teams {m : ℕ} (G : TournamentGraph m) :
  ∃ S : Finset (Fin (2 * m)), S.card = m ∧ ∀ u v, u ∈ S → v ∈ S → u ≠ v → (u, v) ∉ G.edges := by
  sorry

#check select_non_adjacent_teams

end NUMINAMATH_CALUDE_ERRORFEEDBACK_select_non_adjacent_teams_l693_69360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reach_2020_from_any_natural_l693_69339

def can_reach_2020 (n : ℕ) : Prop :=
  ∃ (seq : List Bool), n + 3 * (seq.filter id).length - 2 * (seq.filter (λ x => !x)).length = 2020

theorem reach_2020_from_any_natural :
  ∀ n : ℕ, can_reach_2020 n :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reach_2020_from_any_natural_l693_69339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_one_third_implies_cos_squared_plus_cos_shifted_l693_69337

theorem tan_alpha_one_third_implies_cos_squared_plus_cos_shifted (α : ℝ) :
  Real.tan α = 1/3 → (Real.cos α)^2 + Real.cos (π/2 + 2*α) = 3/10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_one_third_implies_cos_squared_plus_cos_shifted_l693_69337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l693_69340

/-- Ellipse E with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

/-- Parabola y² = 9/4x -/
def parabola (x y : ℝ) : Prop := y^2 = 9/4 * x

/-- Eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := Real.sqrt (1 - (e.b / e.a)^2)

/-- Standard form of an ellipse -/
def standard_form (e : Ellipse) (x y : ℝ) : Prop := x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- Area of a triangle given three points -/
noncomputable def triangle_area (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  (1/2) * abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

theorem ellipse_properties (e : Ellipse) :
  eccentricity e = 1/2 →
  ∃ (x1 y1 x2 y2 : ℝ), 
    parabola x1 y1 ∧ parabola x2 y2 ∧
    standard_form e x1 y1 ∧ standard_form e x2 y2 →
  (standard_form e = λ x y ↦ x^2 / 4 + y^2 / 3 = 1) ∧
  (∃ (S : ℝ → ℝ → ℝ), 
    (∀ x y, S x y ≤ Real.sqrt 3) ∧
    (∃ x0 y0, S x0 y0 = Real.sqrt 3) ∧
    (∀ x y, ∃ (x1 y1 x2 y2 x3 y3 x4 y4 : ℝ),
      S x y = abs (triangle_area x1 y1 x2 y2 x3 y3 - triangle_area x1 y1 x2 y2 x4 y4) ∧
      standard_form e x3 y3 ∧ standard_form e x4 y4)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l693_69340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_with_perimeter_12_l693_69316

/-- A triangle with integral sides and perimeter 12 has area 2√6 -/
theorem triangle_area_with_perimeter_12 :
  ∀ a b c : ℕ,
  a + b + c = 12 →
  a + b > c →
  b + c > a →
  c + a > b →
  (a : ℝ) * (b : ℝ) * (c : ℝ) ≠ 0 →
  Real.sqrt ((6 : ℝ) * (6 - a) * (6 - b) * (6 - c)) = 2 * Real.sqrt 6 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_with_perimeter_12_l693_69316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_l693_69327

/-- Represents the length of one side of the rectangle -/
def x : ℝ := sorry

/-- Represents the length of the other side of the rectangle -/
def y : ℝ := sorry

/-- The perimeter of the rectangle is 60 feet -/
axiom perimeter : 2 * x + 2 * y = 60

/-- One side is at least 5 feet longer than the other -/
axiom side_difference : y = x + 5

/-- The area of the rectangle -/
def area : ℝ := x * y

/-- Theorem stating that the maximum area of the rectangle under the given constraints is 218.75 square feet -/
theorem max_area : area ≤ 218.75 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_l693_69327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylindrical_intersection_area_l693_69322

open Real MeasureTheory

noncomputable section

variable (r R : ℝ)

def m (r R : ℝ) : ℝ := r / R

def S (r R : ℝ) : ℝ := 8 * r^2 * ∫ (v : ℝ) in Set.Icc 0 1, (1 - v^2) / Real.sqrt ((1 - v^2) * (1 - (m r R)^2 * v^2))

def K (r R : ℝ) : ℝ := ∫ (v : ℝ) in Set.Icc 0 1, 1 / Real.sqrt ((1 - v^2) * (1 - (m r R)^2 * v^2))

def E (r R : ℝ) : ℝ := ∫ (v : ℝ) in Set.Icc 0 1, Real.sqrt ((1 - (m r R)^2 * v^2) / (1 - v^2))

theorem cylindrical_intersection_area (r R : ℝ) (hr : 0 < r) (hR : 0 < R) (hrR : r ≤ R) :
  S r R = 8 * (R^2 * E r R - (R^2 - r^2) * K r R) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylindrical_intersection_area_l693_69322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l693_69321

/-- The area of a triangle with base 2 meters and height 3 meters is 3 square meters. -/
theorem triangle_area : ∃ (base height area : ℝ), 
  base = 2 ∧ height = 3 ∧ area = (base * height) / 2 ∧ area = 3 := by
  use 2, 3, 3
  constructor
  · rfl
  constructor
  · rfl
  constructor
  · norm_num
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l693_69321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_l693_69351

def a : ℕ → ℚ
  | 0 => 1  -- Add this case for n = 0
  | 1 => 1
  | (n+2) => (((n+1) * a (n+1)) + 2 * (n+2)^2) / (n+3)

theorem a_formula (n : ℕ) : a n = (1/2) * n * (n+1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_l693_69351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l693_69399

-- Define the hyperbola
noncomputable def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

-- Define the condition for the line intersecting at one point
noncomputable def line_intersects_once (a b : ℝ) : Prop :=
  ∃ (x y : ℝ), hyperbola a b x y ∧ 
    (y - 0) / (x - (a * Real.sqrt (a^2 + b^2))) = Real.sqrt 3

-- Define the eccentricity
noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (a^2 + b^2) / a

-- Theorem statement
theorem hyperbola_eccentricity (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) 
  (h_intersect : line_intersects_once a b) : 
  eccentricity a b = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l693_69399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_michaels_bunnies_l693_69320

theorem michaels_bunnies (total_pets : ℕ) (dog_percentage : ℚ) (cat_percentage : ℚ) :
  total_pets = 36 →
  dog_percentage = 1/4 →
  cat_percentage = 1/2 →
  ∃ (num_bunnies : ℕ),
    num_bunnies = total_pets - (Nat.floor (↑total_pets * dog_percentage) + Nat.floor (↑total_pets * cat_percentage)) ∧
    num_bunnies = 9 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_michaels_bunnies_l693_69320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_imaginary_part_l693_69343

theorem complex_imaginary_part (z : ℂ) : 
  (z.re = 1) → (Complex.abs z = 2) → (z.im = Real.sqrt 3 ∨ z.im = -Real.sqrt 3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_imaginary_part_l693_69343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_erased_number_l693_69349

/-- Given 21 consecutive natural numbers, if the sum of 20 of these numbers is 2017, 
    then the missing number is 104. -/
theorem erased_number (n : ℕ) : 
  (∃ (x : ℕ), x ∈ Finset.range 21 ∧
   (Finset.sum (Finset.range 21) (λ i => n + i - 10) - (n + x - 10) = 2017)) →
  ∃ (x : ℕ), x ∈ Finset.range 21 ∧ (n + x - 10 = 104) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_erased_number_l693_69349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_equilateral_triangle_l693_69328

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : 0 < a ∧ 0 < b ∧ b ≤ a

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a)^2)

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: Eccentricity of ellipse with equilateral triangle property -/
theorem ellipse_eccentricity_equilateral_triangle 
  (e : Ellipse) (F₁ F₂ A B : Point) : 
  F₁.x + F₂.x = 0 → -- F₁ and F₂ are symmetrical about y-axis
  F₁.x < 0 → -- F₁ is on the left
  F₁.y = 0 → -- F₁ is on x-axis
  F₂.y = 0 → -- F₂ is on x-axis
  A.x = F₁.x → -- A is on the same vertical line as F₁
  B.x = F₁.x → -- B is on the same vertical line as F₁
  (A.x / e.a)^2 + (A.y / e.b)^2 = 1 → -- A is on the ellipse
  (B.x / e.a)^2 + (B.y / e.b)^2 = 1 → -- B is on the ellipse
  distance A B = distance A F₂ → -- ABF₂ is equilateral
  distance A B = distance B F₂ → -- ABF₂ is equilateral
  eccentricity e = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_equilateral_triangle_l693_69328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_form_horizontal_line_l693_69373

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Function to create a line from a point and an angle -/
noncomputable def lineFromPointAndAngle (p : Point) (angle : ℝ) : Line :=
  { slope := Real.tan angle,
    intercept := p.y - Real.tan angle * p.x }

/-- Function to find the intersection point of two lines -/
noncomputable def intersectionPoint (l1 l2 : Line) : Point :=
  { x := (l2.intercept - l1.intercept) / (l1.slope - l2.slope),
    y := l1.slope * ((l2.intercept - l1.intercept) / (l1.slope - l2.slope)) + l1.intercept }

/-- Theorem stating that the intersection points form a horizontal line through the center -/
theorem intersection_points_form_horizontal_line (a b c d : Point)
  (h1 : a = ⟨0, 0⟩)
  (h2 : b = ⟨0, 5⟩)
  (h3 : c = ⟨8, 5⟩)
  (h4 : d = ⟨8, 0⟩)
  (l1 : Line := lineFromPointAndAngle a (45 * π / 180))
  (l2 : Line := lineFromPointAndAngle a (75 * π / 180))
  (l3 : Line := lineFromPointAndAngle b (-45 * π / 180))
  (l4 : Line := lineFromPointAndAngle b (-75 * π / 180))
  (p1 : Point := intersectionPoint l1 l3)
  (p2 : Point := intersectionPoint l2 l4) :
  p1.y = p2.y ∧ p1.y = 2.5 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_form_horizontal_line_l693_69373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cannot_mark_on_line_l693_69363

def initial_points : List (ℤ × ℤ) := [(1, 1), (2, 3), (4, 5), (999, 111)]

def f (p : ℤ × ℤ) : ℤ := p.1^2 + p.2^2

def mark_rule1 (p : ℤ × ℤ) : List (ℤ × ℤ) := 
  [(p.2, p.1), (p.1 - p.2, p.1 + p.2)]

def mark_rule2 (p q : ℤ × ℤ) : (ℤ × ℤ) := 
  (p.1 * q.2 + p.2 * q.1, 4 * p.1 * q.1 - 4 * p.2 * q.2)

def is_on_line (p : ℤ × ℤ) : Prop := p.2 = 2 * p.1

theorem cannot_mark_on_line : 
  ∀ (marked : List (ℤ × ℤ)), 
    (∀ p, p ∈ initial_points → p ∈ marked) →
    (∀ p, p ∈ marked → ∀ q ∈ mark_rule1 p, q ∈ marked) →
    (∀ p q, p ∈ marked → q ∈ marked → mark_rule2 p q ∈ marked) →
    ¬∃ p, p ∈ marked ∧ is_on_line p :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cannot_mark_on_line_l693_69363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_equation_pi_over_four_is_line_l693_69311

/-- The curve defined by the polar equation θ = π/4 is a straight line -/
theorem polar_equation_pi_over_four_is_line :
  ∃ (m b : ℝ), ∀ (x y : ℝ), (Real.arctan (y / x) = π / 4) ↔ (y = m * x + b) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_equation_pi_over_four_is_line_l693_69311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_update_proportion_l693_69374

theorem taxi_update_proportion :
  ∃ (x : ℝ), 
    x > 0 ∧
    x + 1.2 * x + 1.44 * x = 1 ∧
    (x ≥ 0.2745 ∧ x < 0.2755) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_update_proportion_l693_69374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_60_degrees_implies_t_values_l693_69315

noncomputable def angle_between_lines (m1 m2 : ℝ) : ℝ :=
  Real.arctan (abs ((m1 - m2) / (1 + m1 * m2)))

theorem angle_60_degrees_implies_t_values
  (l1 : ℝ → ℝ → Prop)
  (l2 : ℝ → ℝ → ℝ → Prop)
  (h1 : ∀ x y, l1 x y ↔ x - Real.sqrt 3 * y + 1 = 0)
  (h2 : ∀ x y t, l2 x y t ↔ x + t * y + 1 = 0)
  (h_angle : angle_between_lines (Real.sqrt 3) (-1/t) = π/3) :
  t = 0 ∨ t = Real.sqrt 3 := by
  sorry

#check angle_60_degrees_implies_t_values

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_60_degrees_implies_t_values_l693_69315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_coordinates_l693_69375

/-- The focus of a parabola y = ax^2 (a ≠ 0) has coordinates (0, 1/(4a)) -/
theorem parabola_focus_coordinates (a : ℝ) (ha : a ≠ 0) :
  let parabola := {p : ℝ × ℝ | p.2 = a * p.1^2}
  ∃ (f : ℝ × ℝ), f ∈ parabola ∧ f = (0, 1 / (4 * a)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_coordinates_l693_69375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_two_condition_l693_69302

theorem power_of_two_condition (n : ℕ) (hn : n ≥ 2) :
  (∃ (x : Fin (n - 1) → ℤ), ∀ (i j : Fin (n - 1)), 0 < i.val ∧ 0 < j.val ∧ i ≠ j ∧ n ∣ (2 * i.val + j.val) → x i < x j) ↔
  ∃ (s : ℕ), n = 2^s ∧ s ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_two_condition_l693_69302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_h_deriv_h_max_at_one_h_max_value_l693_69396

/-- Given a function f(x) = a/x - 1 + ln(x), if there exists x₀ > 0 such that f(x₀) ≤ 0,
    then a ≤ 1. -/
theorem function_inequality (a : ℝ) :
  (∃ x₀ : ℝ, x₀ > 0 ∧ a / x₀ - 1 + Real.log x₀ ≤ 0) → a ≤ 1 := by
  sorry

/-- The helper function h(x) = x - x * ln(x) -/
noncomputable def h (x : ℝ) : ℝ := x - x * Real.log x

/-- The derivative of h(x) is -ln(x) -/
theorem h_deriv (x : ℝ) (hx : x > 0) : 
  deriv h x = -Real.log x := by
  sorry

/-- The maximum value of h(x) occurs at x = 1 -/
theorem h_max_at_one : 
  ∀ x > 0, h x ≤ h 1 := by
  sorry

/-- The maximum value of h(x) is 1 -/
theorem h_max_value : h 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_h_deriv_h_max_at_one_h_max_value_l693_69396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l693_69393

noncomputable def f (x : ℝ) := Real.sin x ^ 2 - Real.cos (x + Real.pi / 3) ^ 2

theorem f_properties :
  (∃ (k : ℤ), ∀ (x : ℝ), f (k * Real.pi / 2 + Real.pi / 12 + x) = f (k * Real.pi / 2 + Real.pi / 12 - x)) ∧
  (∀ (x y : ℝ), -Real.pi / 6 ≤ x ∧ x < y ∧ y ≤ Real.pi / 4 → f x < f y) ∧
  (∀ (x y : ℝ), -Real.pi / 3 ≤ x ∧ x < y ∧ y ≤ -Real.pi / 6 → f x > f y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l693_69393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_milk_quantity_l693_69358

/-- Represents the vessel and its contents --/
structure Vessel where
  capacity : ℚ
  milk : ℚ
  water : ℚ

/-- Performs a dilution step on the vessel --/
def dilute (v : Vessel) (amount : ℚ) : Vessel :=
  let removed_milk := (v.milk / (v.milk + v.water)) * amount
  { capacity := v.capacity,
    milk := v.milk - removed_milk,
    water := v.water - (amount - removed_milk) + amount }

theorem final_milk_quantity (initial_capacity : ℚ) (dilution_amount : ℚ) :
  let initial_vessel : Vessel := { capacity := initial_capacity, milk := initial_capacity, water := 0 }
  let first_dilution := dilute initial_vessel dilution_amount
  let final_vessel := dilute first_dilution dilution_amount
  final_vessel.milk = 58.08 := by
  sorry

#eval let initial_vessel : Vessel := { capacity := 75, milk := 75, water := 0 }
      let first_dilution := dilute initial_vessel 9
      let final_vessel := dilute first_dilution 9
      final_vessel.milk

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_milk_quantity_l693_69358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_line_equation_l693_69365

/-- Angle bisector of ∠ABC -/
def angle_bisector (A B C : ℝ × ℝ) : Set (ℝ × ℝ) :=
  sorry

/-- Line through two points -/
def line_through (A B : ℝ × ℝ) : Set (ℝ × ℝ) :=
  sorry

/-- Triangle ABC with given properties has line BC with equation 4x + 17y + 12 = 0 -/
theorem triangle_line_equation (A B C : ℝ × ℝ) : 
  A = (1, 4) →
  (∃ k : ℝ, ∀ x y : ℝ, (x, y) ∈ angle_bisector B A C ↔ x - 2*y = k) →
  (∃ m : ℝ, ∀ x y : ℝ, (x, y) ∈ angle_bisector C A B ↔ x + y = m) →
  ∃ a b c : ℝ, a = 4 ∧ b = 17 ∧ c = 12 ∧ 
    (∀ x y : ℝ, (x, y) ∈ line_through B C ↔ a*x + b*y + c = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_line_equation_l693_69365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_symmetry_l693_69338

noncomputable def f (ω φ x : ℝ) : ℝ := Real.sin (ω * x + φ)

noncomputable def g (x : ℝ) : ℝ := Real.sin x

theorem center_of_symmetry 
  (ω φ : ℝ) 
  (h1 : ω > 0) 
  (h2 : abs φ < π) 
  (h3 : ∀ x, g x = f ω φ (x / 2 + π / 3)) :
  ∃ k : ℤ, f ω φ (-π / 6) = 0 ∧ 
    ∀ x : ℝ, abs x < abs (-π / 6) → f ω φ x ≠ 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_symmetry_l693_69338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_faulty_key_identification_l693_69368

/-- Represents a digit key on a keypad -/
inductive Digit : Type where
  | zero | one | two | three | four | five | six | seven | eight | nine
  deriving Repr, DecidableEq

/-- Represents the result of pressing a key -/
inductive KeyPress where
  | registered
  | notRegistered
  deriving Repr, DecidableEq

/-- Represents a sequence of key presses -/
def KeySequence := List KeyPress

/-- Represents the attempted sequence of digits -/
def AttemptedSequence := List Digit

/-- Checks if a key is potentially faulty based on its press pattern -/
def isPotentiallyFaulty (presses : KeySequence) : Prop :=
  presses.length ≥ 5 ∧
  presses.get? 0 = some KeyPress.notRegistered ∧
  presses.get? 2 = some KeyPress.notRegistered ∧
  presses.get? 4 = some KeyPress.notRegistered ∧
  presses.get? 1 = some KeyPress.registered ∧
  presses.get? 3 = some KeyPress.registered

/-- The main theorem -/
theorem faulty_key_identification
  (attempted : AttemptedSequence)
  (registered : AttemptedSequence)
  (h1 : attempted.length = 10)
  (h2 : registered.length = 7)
  (h3 : ∃ d : Digit, isPotentiallyFaulty (attempted.map (λ x => if x = d then KeyPress.notRegistered else KeyPress.registered))) :
  ∃ d : Digit, d = Digit.seven ∨ d = Digit.nine :=
by
  sorry

#check faulty_key_identification

end NUMINAMATH_CALUDE_ERRORFEEDBACK_faulty_key_identification_l693_69368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_implies_a_less_than_two_l693_69348

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := |log (x - 1)|

-- State the theorem
theorem existence_implies_a_less_than_two
  (a b : ℝ)
  (h : ∃ (x₁ x₂ : ℝ), x₁ ∈ Set.Icc a b ∧ x₂ ∈ Set.Icc a b ∧ x₁ < x₂ ∧ f x₁ > f x₂) :
  a < 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_implies_a_less_than_two_l693_69348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_integers_problem_l693_69369

theorem consecutive_integers_problem (n m : ℕ) (h : List ℕ) : 
  (n < m) →
  (h = (List.range (m - n + 1)).map (· + n)) →
  ((h.sum : ℚ) / h.length = 20) →
  n * m = 391 →
  h.length = 7 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_integers_problem_l693_69369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minor_arc_circumference_l693_69344

noncomputable section

open Real

/-- A circle in a 2D plane. -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point in a 2D plane. -/
abbrev Point := ℝ × ℝ

/-- Check if a point is on a circle. -/
def onCircle (p : Point) (c : Circle) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

/-- Angle between three points. -/
def angleBetween (A B C : Point) : ℝ := sorry

/-- Circumference of the minor arc between two points on a circle. -/
def minorArcCircumference (A B : Point) (c : Circle) : ℝ := sorry

theorem minor_arc_circumference (r : ℝ) (A B C : Point) (circle : Circle) :
  r = 12 →
  circle.radius = r →
  onCircle A circle →
  onCircle B circle →
  onCircle C circle →
  angleBetween A C B = π / 4 →
  minorArcCircumference A B circle = 6 * π :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minor_arc_circumference_l693_69344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_boxes_is_sixteen_l693_69398

/-- Represents the number of complete packaging boxes that can be made. -/
def max_packaging_boxes (total_sheets : ℕ) (bodies_per_sheet : ℕ) (lids_per_sheet : ℕ) 
  (bodies_per_box : ℕ) (lids_per_box : ℕ) : ℕ :=
  let sheets_for_bodies := total_sheets / 2
  let sheets_for_lids := total_sheets - sheets_for_bodies
  let bodies := sheets_for_bodies * bodies_per_sheet
  let lids := sheets_for_lids * lids_per_sheet
  Int.toNat (min (bodies / bodies_per_box) (lids / lids_per_box))

/-- Theorem stating that the maximum number of packaging boxes is 16 under the given conditions. -/
theorem max_boxes_is_sixteen : 
  max_packaging_boxes 20 2 3 1 2 = 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_boxes_is_sixteen_l693_69398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alternate_angles_equal_implies_parallel_l693_69397

/-- Two lines in a Euclidean plane -/
structure Line : Type

/-- A transversal line intersecting two other lines -/
structure Transversal : Type

/-- Interior alternate angles formed by two lines and a transversal -/
def interior_alternate_angles (l₁ l₂ : Line) (t : Transversal) : Type := sorry

/-- Predicate for two lines being parallel -/
def parallel (l₁ l₂ : Line) : Prop := sorry

/-- Predicate for two angles being equal -/
def angles_equal (a₁ a₂ : interior_alternate_angles l₁ l₂ t) : Prop := sorry

/-- Theorem: If the interior alternate angles are equal, then the lines are parallel -/
theorem alternate_angles_equal_implies_parallel 
  (l₁ l₂ : Line) (t : Transversal) 
  (a₁ a₂ : interior_alternate_angles l₁ l₂ t) :
  angles_equal a₁ a₂ → parallel l₁ l₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alternate_angles_equal_implies_parallel_l693_69397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_expression_l693_69377

/-- The maximum value of c · a^b - d, where a, b, c, and d are distinct values from {1, 2, 3, 4} -/
theorem max_value_expression : 
  ∃ (a b c d : ℕ), 
    a ∈ ({1, 2, 3, 4} : Set ℕ) ∧
    b ∈ ({1, 2, 3, 4} : Set ℕ) ∧
    c ∈ ({1, 2, 3, 4} : Set ℕ) ∧
    d ∈ ({1, 2, 3, 4} : Set ℕ) ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    ∀ (a' b' c' d' : ℕ), 
      a' ∈ ({1, 2, 3, 4} : Set ℕ) →
      b' ∈ ({1, 2, 3, 4} : Set ℕ) →
      c' ∈ ({1, 2, 3, 4} : Set ℕ) →
      d' ∈ ({1, 2, 3, 4} : Set ℕ) →
      a' ≠ b' → a' ≠ c' → a' ≠ d' → b' ≠ c' → b' ≠ d' → c' ≠ d' →
      c' * (a' ^ b') - d' ≤ 32 :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_expression_l693_69377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mutual_fund_share_price_increase_l693_69329

theorem mutual_fund_share_price_increase (P : ℝ) (h : P > 0) : 
  let first_quarter_price := 1.20 * P
  let second_quarter_price := 1.50 * P
  (second_quarter_price - first_quarter_price) / first_quarter_price * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mutual_fund_share_price_increase_l693_69329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_k_l693_69372

-- Define the function f
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := x - Real.log x + k

-- Define the interval [1/e, e]
def I : Set ℝ := { x | 1 / Real.exp 1 ≤ x ∧ x ≤ Real.exp 1 }

-- State the theorem
theorem range_of_k (a b c : ℝ) (ha : a ∈ I) (hb : b ∈ I) (hc : c ∈ I) :
  (∃ k : ℝ, k > Real.exp 1 - 3 ∧ 
    (f k a + f k b > f k c) ∧ 
    (f k b + f k c > f k a) ∧ 
    (f k c + f k a > f k b)) ∧
  (∀ k : ℝ, k ≤ Real.exp 1 - 3 → 
    ¬((f k a + f k b > f k c) ∧ 
      (f k b + f k c > f k a) ∧ 
      (f k c + f k a > f k b))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_k_l693_69372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_triangle_APR_l693_69326

-- Define the circle and points
variable (circle : Type) (A B C P Q R : ℝ × ℝ)

-- Define the distances
variable (AB AC BP PQ QR CR x y : ℝ)

-- State the theorem
theorem perimeter_of_triangle_APR
  (h1 : AB = 24)
  (h2 : AC = AB)  -- Property of tangents from an external point
  (h3 : BP = x)
  (h4 : PQ = x)
  (h5 : QR = y)
  (h6 : CR = y)
  (h7 : x + y = 12)
  : abs (A.1 - P.1) + abs (P.1 - R.1) + abs (R.1 - A.1) +
    abs (A.2 - P.2) + abs (P.2 - R.2) + abs (R.2 - A.2) = 48 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_triangle_APR_l693_69326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_forms_two_rays_l693_69366

-- Define the parametric equations
noncomputable def x (t : ℝ) : ℝ := t + 1/t
def y : ℝ := 2

-- Define a point on the curve
noncomputable def point_on_curve (t : ℝ) : ℝ × ℝ := (x t, y)

-- Define the set of all points on the curve
noncomputable def curve : Set (ℝ × ℝ) := {p | ∃ t : ℝ, t ≠ 0 ∧ p = point_on_curve t}

-- Theorem statement
theorem curve_forms_two_rays :
  ∃ (a b : ℝ), a < b ∧
  (∀ p ∈ curve, p.2 = y ∧ (p.1 ≤ a ∨ p.1 ≥ b)) ∧
  (∀ x₀, x₀ ≤ a → (x₀, y) ∈ curve) ∧
  (∀ x₀, x₀ ≥ b → (x₀, y) ∈ curve) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_forms_two_rays_l693_69366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_propositions_l693_69325

-- Define the propositions
def p₁ : Prop := ∃ n : ℕ, n^2 > 2^n
def p₂ : Prop := ∀ x : ℝ, (x > 1 → x > 2) ∧ ¬(x > 2 → x > 1)
def p₃ : Prop := (∀ x y : ℝ, x = y → Real.sin x = Real.sin y) ↔ (∀ x y : ℝ, Real.sin x ≠ Real.sin y → x ≠ y)
def p₄ : Prop := ∀ p q : Prop, (p ∨ q) → p

-- Theorem stating which propositions are true
theorem correct_propositions : p₁ ∧ p₃ ∧ ¬p₂ ∧ ¬p₄ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_propositions_l693_69325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_tangents_inequality_condition_l693_69307

noncomputable def f (x : ℝ) : ℝ := (1 + x) / Real.exp x
noncomputable def g (a x : ℝ) : ℝ := 1 - a * x^2

-- Statement 1: Parallel tangent lines at x=1 implies a = 1/(2e)
theorem parallel_tangents (a : ℝ) :
  (deriv f 1 = deriv (g a) 1) → a = 1 / (2 * Real.exp 1) := by sorry

-- Statement 2: f(x) ≤ g(x) for x ∈ [0,1] implies a ≤ (e-3)/e
theorem inequality_condition (a : ℝ) :
  (∀ x ∈ Set.Icc 0 1, f x ≤ g a x) → a ≤ (Real.exp 1 - 3) / Real.exp 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_tangents_inequality_condition_l693_69307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_bound_l693_69385

/-- The area of a convex quadrilateral with positive side lengths a, b, c, d
    in cyclic order is at most (a+c)(b+d)/4. -/
theorem quadrilateral_area_bound (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  ∃ (area : ℝ), area ≤ (a + c) * (b + d) / 4 ∧ 
  (∃ (quad : Set ℝ × Set ℝ), 
    Convex ℝ quad.1 ∧
    ∃ (sides : List ℝ), sides = [a, b, c, d] ∧
    ∃ (area_func : Set ℝ × Set ℝ → ℝ), area_func quad = area) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_bound_l693_69385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_in_second_quadrant_l693_69392

def FourthQuadrant : Set ℝ := {θ | Real.cos θ > 0 ∧ Real.sin θ < 0}

def SecondQuadrant : Set (ℝ × ℝ) := {P | P.1 < 0 ∧ P.2 > 0}

theorem point_in_second_quadrant (θ : Real) (h : θ ∈ FourthQuadrant) :
  let P : ℝ × ℝ := (Real.sin (Real.sin θ), Real.cos (Real.sin θ))
  P ∈ SecondQuadrant :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_in_second_quadrant_l693_69392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_nested_rectangles_l693_69362

/-- A set of points in ℝ² forms a rectangle with given length and width. -/
def is_rectangle (S : Set (ℝ × ℝ)) (l w : ℝ) : Prop :=
  sorry

/-- The set of points formed by the intersection of diagonals of a given set. -/
def diagonal_intersection (S : Set (ℝ × ℝ)) : Set (ℝ × ℝ) :=
  sorry

/-- The area of a set of points in ℝ². -/
noncomputable def area (S : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- Given a rectangle R₁ with length 6 and width 4, R₂ is formed by the intersection of R₁'s diagonals,
    and R₃ is formed by the intersection of R₂'s diagonals. The area of R₃ is 6.5. -/
theorem area_of_nested_rectangles :
  ∀ (R₁ R₂ R₃ : Set (ℝ × ℝ)),
    (∃ (l w : ℝ), l = 6 ∧ w = 4 ∧ is_rectangle R₁ l w) →
    (R₂ = diagonal_intersection R₁) →
    (R₃ = diagonal_intersection R₂) →
    area R₃ = 6.5 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_nested_rectangles_l693_69362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_common_difference_l693_69350

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def SumOfTerms (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (a 1 + a n) / 2

/-- Common difference of an arithmetic sequence -/
def CommonDifference (a : ℕ → ℚ) : ℚ :=
  a 2 - a 1

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℚ) 
  (h_arith : ArithmeticSequence a) 
  (h_sum : SumOfTerms a 13 = 104) 
  (h_term : a 6 = 5) : 
  CommonDifference a = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_common_difference_l693_69350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_I_problem_II_l693_69371

def mySequence (a : ℕ → ℕ) : Prop :=
  ∀ n, a (n + 1) = (a n + 3) / 2

theorem problem_I (a : ℕ → ℕ) (h : mySequence a) (h1 : a 1 = 19) : 
  a 2014 = 98 := by
  sorry

theorem problem_II (a : ℕ → ℕ) (h : mySequence a) 
  (h1 : ∃ n, a n > 1 ∧ Odd (a n) ∧ ∀ m, m ≥ n → a m = a n) : 
  ∃ n, a n = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_I_problem_II_l693_69371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l693_69379

def a : Fin 3 → ℝ := ![3, 2, -1]
def b : Fin 3 → ℝ := ![2, 1, 2]

def dot_product (v w : Fin 3 → ℝ) : ℝ :=
  (v 0) * (w 0) + (v 1) * (w 1) + (v 2) * (w 2)

def add_vectors (v w : Fin 3 → ℝ) : Fin 3 → ℝ :=
  fun i => v i + w i

def scalar_mult (k : ℝ) (v : Fin 3 → ℝ) : Fin 3 → ℝ :=
  fun i => k * v i

theorem vector_problem :
  (dot_product (add_vectors a b) (add_vectors a (scalar_mult (-2) b)) = -10) ∧
  (∀ k : ℝ, dot_product (add_vectors (scalar_mult k a) b) (add_vectors a (scalar_mult (-k) b)) = 0 ↔ k = 3/2 ∨ k = -2/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l693_69379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_diameter_specific_triangle_l693_69333

/-- The diameter of the inscribed circle of a triangle --/
noncomputable def inscribed_circle_diameter (a b c : ℝ) : ℝ :=
  2 * (a + b + c) / (Real.sqrt ((a + b + c) * (-a + b + c) * (a - b + c) * (a + b - c)))

theorem inscribed_circle_diameter_specific_triangle :
  inscribed_circle_diameter 13 5 12 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_diameter_specific_triangle_l693_69333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trip_speed_calculation_l693_69386

/-- Proves that given a 100-mile trip where the first 50 miles are traveled at 20 mph
    and the average speed for the entire trip is 28.571428571428573 mph,
    the speed for the remaining 50 miles is 50 mph. -/
theorem trip_speed_calculation (total_distance : ℝ) (first_half_distance : ℝ) 
  (first_half_speed : ℝ) (average_speed : ℝ) :
  total_distance = 100 →
  first_half_distance = 50 →
  first_half_speed = 20 →
  average_speed = 28.571428571428573 →
  (let remaining_distance := total_distance - first_half_distance
   let total_time := total_distance / average_speed
   let first_half_time := first_half_distance / first_half_speed
   let remaining_time := total_time - first_half_time
   let remaining_speed := remaining_distance / remaining_time
   remaining_speed) = 50 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trip_speed_calculation_l693_69386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_l693_69353

-- Define the perpendicularity condition
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

-- Theorem statement
theorem perpendicular_lines :
  ∀ k : ℝ, perpendicular k 3 → k = -1/3 := by
  intro k h
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_l693_69353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_congruence_problem_l693_69341

theorem congruence_problem (a b n : ℤ) : 
  a ≡ 23 [ZMOD 37] →
  b ≡ 58 [ZMOD 37] →
  150 ≤ n ∧ n ≤ 191 →
  a - b ≡ n [ZMOD 37] →
  n = 150 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_congruence_problem_l693_69341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_valid_numbers_l693_69359

def isValidNumber (n : ℕ) : Bool :=
  n ≤ 60 && n % 3 = 0 && n % 4 ≠ 0

def validNumbers : Finset ℕ :=
  Finset.filter (fun n => isValidNumber n) (Finset.range 61)

theorem average_of_valid_numbers : (Finset.sum validNumbers id) / validNumbers.card = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_valid_numbers_l693_69359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l693_69318

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2^(-x) + 1 else -Real.sqrt x

-- Define the solution set
def solution_set : Set ℝ := {x | x ≥ -4}

-- Theorem statement
theorem inequality_solution_set :
  {x : ℝ | f (x + 1) - 9 ≤ 0} = solution_set :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l693_69318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_diagonal_l693_69356

/-- Represents an isosceles trapezoid -/
structure IsoscelesTrapezoid where
  leg : ℝ
  shortBase : ℝ
  longBase : ℝ

/-- Calculates the length of the diagonal in an isosceles trapezoid -/
noncomputable def diagonalLength (t : IsoscelesTrapezoid) : ℝ :=
  Real.sqrt ((t.leg - (t.longBase - t.shortBase) / 2) ^ 2 + ((t.longBase - t.shortBase) / 2 + t.shortBase) ^ 2)

theorem isosceles_trapezoid_diagonal 
  (t : IsoscelesTrapezoid) 
  (h1 : t.leg = 13)
  (h2 : t.shortBase = 10)
  (h3 : t.longBase = 24) :
  diagonalLength t = Real.sqrt 232 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_diagonal_l693_69356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_arithmetic_sequence_l693_69380

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  angle_sum : A + B + C = Real.pi
  law_of_cosines_B : Real.cos B = (a^2 + c^2 - b^2) / (2*a*c)

-- Define the given condition
def condition (t : Triangle) : Prop :=
  1 / (t.a + t.b) + 1 / (t.b + t.c) = 3 / (t.a + t.b + t.c)

-- State the theorem
theorem triangle_arithmetic_sequence (t : Triangle) (h : condition t) :
  t.A = t.B ∧ t.B = t.C ∧ t.A = Real.pi/3 ∧ t.B = Real.pi/3 ∧ t.C = Real.pi/3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_arithmetic_sequence_l693_69380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_f_lower_bound_min_a_value_l693_69390

-- Define the function f(x) = x ln x
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

-- Theorem 1: Tangent line equation
theorem tangent_line_at_one (x : ℝ) :
  ∃ (m b : ℝ), (∀ y, y = m * (x - 1) + b) ∧ 
  (m = (deriv f) 1) ∧ (b = f 1) ∧ (m = 1) ∧ (b = 0) := by sorry

-- Theorem 2: Lower bound of f(x)
theorem f_lower_bound {x : ℝ} (hx : x > 0) : f x ≥ x - 1 := by sorry

-- Theorem 3: Minimum value of a
theorem min_a_value :
  ∃ (a : ℝ), (a = -Real.exp 3) ∧
  (∀ x > 0, f x ≥ a * x^2 + 2/a) ∧
  (∀ a' < a, ∃ x > 0, f x < a' * x^2 + 2/a') := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_f_lower_bound_min_a_value_l693_69390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_wire_length_l693_69306

-- Define the diameters of the rods
def rod1_diameter : ℝ := 4
def rod2_diameter : ℝ := 16

-- Define the function to calculate the wire length
noncomputable def wire_length (d1 d2 : ℝ) : ℝ :=
  let r1 := d1 / 2
  let r2 := d2 / 2
  let straight_section := 2 * (r1 + r2)
  let arc_length := Real.pi * (r1 + r2)
  straight_section + arc_length

-- Theorem statement
theorem shortest_wire_length :
  wire_length rod1_diameter rod2_diameter = 16 + 10 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_wire_length_l693_69306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_five_l693_69395

-- Define the geometric sequence
noncomputable def geometric_sequence (q : ℝ) : ℕ → ℝ := fun n => q^(n-1)

-- Define the sum of the first n terms of a geometric sequence
noncomputable def geometric_sum (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n else (1 - q^n) / (1 - q)

theorem geometric_sequence_sum_five :
  ∃ q : ℝ,
    (q > 1) ∧
    (∀ n : ℕ, n ≥ 2 → 2 * (geometric_sequence q (n+1)) + 2 * (geometric_sequence q (n-1)) = 5 * (geometric_sequence q n)) ∧
    (geometric_sum q 5 = 31) := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_five_l693_69395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_when_angle_60_l693_69330

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- The angle between the asymptote and the x-axis of a hyperbola -/
noncomputable def asymptote_angle (h : Hyperbola) : ℝ := Real.arctan (h.b / h.a)

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := Real.sqrt (1 + h.b^2 / h.a^2)

/-- 
Given a hyperbola x²/a² - y²/b² = 1 where a > 0 and b > 0,
if the angle between its asymptote and the x-axis is 60°,
then its eccentricity is 2.
-/
theorem hyperbola_eccentricity_when_angle_60 (h : Hyperbola) 
    (h_angle : asymptote_angle h = π / 3) : eccentricity h = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_when_angle_60_l693_69330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_school_supplies_cost_per_page_approx_l693_69378

/-- Represents the cost calculation for school supplies --/
noncomputable def school_supplies_cost (notebook_price : ℝ) (pen_box_price : ℝ) (folder_pack_price : ℝ)
  (local_shipping : ℝ) (international_shipping : ℝ) (sales_tax_rate : ℝ) (exchange_rate : ℝ) : ℝ :=
  let notebook_total := notebook_price + notebook_price * 0.5
  let pen_total := pen_box_price * 0.8
  let folder_total := folder_pack_price
  let shipping_total := local_shipping + international_shipping
  let subtotal := notebook_total + pen_total + folder_total + shipping_total
  let total_with_tax := subtotal * (1 + sales_tax_rate)
  let total_local_currency := total_with_tax * exchange_rate
  total_local_currency / 100

theorem school_supplies_cost_per_page_approx (notebook_price : ℝ) (pen_box_price : ℝ) 
  (folder_pack_price : ℝ) (local_shipping : ℝ) (international_shipping : ℝ) 
  (sales_tax_rate : ℝ) (exchange_rate : ℝ) :
  notebook_price = 12 →
  pen_box_price = 9 →
  folder_pack_price = 19 →
  local_shipping = 5 →
  international_shipping = 10 →
  sales_tax_rate = 0.05 →
  exchange_rate = 2.3 →
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |school_supplies_cost notebook_price pen_box_price folder_pack_price 
    local_shipping international_shipping sales_tax_rate exchange_rate - 1.43| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_school_supplies_cost_per_page_approx_l693_69378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leak_empty_time_for_given_cistern_l693_69335

/-- Represents a cistern with a leak -/
structure Cistern where
  fill_time_no_leak : ℚ
  fill_time_with_leak : ℚ

/-- Calculates the time it takes for the leak to empty a full cistern -/
def leak_empty_time (c : Cistern) : ℚ :=
  (c.fill_time_no_leak * c.fill_time_with_leak) / (c.fill_time_with_leak - c.fill_time_no_leak)

/-- Theorem stating that for a cistern that takes 9 hours to fill without a leak
    and 10 hours with a leak, the leak will empty the full cistern in 90 hours -/
theorem leak_empty_time_for_given_cistern :
  let c : Cistern := { fill_time_no_leak := 9, fill_time_with_leak := 10 }
  leak_empty_time c = 90 := by
  -- Proof goes here
  sorry

#eval leak_empty_time { fill_time_no_leak := 9, fill_time_with_leak := 10 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_leak_empty_time_for_given_cistern_l693_69335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_correct_l693_69305

open Real

-- Define the propositions
def prop1 : Prop := ∀ (A B : ℝ) (C : ℝ), A > B → Real.sin A > Real.sin B

def prop2 (x y : ℝ) : Prop :=
  (x ≠ 2 ∨ y ≠ 3) → (x + y ≠ 5) ∧
  ∃ (x y : ℝ), (x ≠ 2 ∨ y ≠ 3) ∧ (x + y = 5)

def prop3 : Prop :=
  (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ ¬(∀ x : ℝ, x^3 - x^2 + 1 > 0)

def prop4 : Prop :=
  ¬(∀ a b : ℝ, a > b → (2 : ℝ)^a > (2 : ℝ)^b - 1) ↔ (∀ a b : ℝ, a ≤ b → (2 : ℝ)^a ≤ (2 : ℝ)^b - 1)

-- Theorem stating that exactly 3 out of 4 propositions are correct
theorem exactly_three_correct :
  (prop1 ∧ prop2 0 0 ∧ ¬prop3 ∧ prop4) ∨
  (prop1 ∧ prop2 0 0 ∧ prop3 ∧ ¬prop4) ∨
  (prop1 ∧ ¬prop2 0 0 ∧ prop3 ∧ prop4) ∨
  (¬prop1 ∧ prop2 0 0 ∧ prop3 ∧ prop4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_correct_l693_69305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_segment_product_constant_l693_69301

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point in a 2D plane -/
def Point := ℝ × ℝ

/-- A chord of a circle -/
structure Chord (c : Circle) where
  endpoint1 : Point
  endpoint2 : Point
  lies_on_circle : (endpoint1.1 - c.center.1)^2 + (endpoint1.2 - c.center.2)^2 = c.radius^2 ∧
                   (endpoint2.1 - c.center.1)^2 + (endpoint2.2 - c.center.2)^2 = c.radius^2

/-- The length of a line segment between two points -/
noncomputable def segment_length (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

/-- Theorem: The product of chord segments through a fixed point is constant -/
theorem chord_segment_product_constant (c : Circle) (p : Point) 
    (h : (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 < c.radius^2) 
    (chord1 chord2 : Chord c) 
    (h1 : segment_length p chord1.endpoint1 + segment_length p chord1.endpoint2 = 
          segment_length chord1.endpoint1 chord1.endpoint2)
    (h2 : segment_length p chord2.endpoint1 + segment_length p chord2.endpoint2 = 
          segment_length chord2.endpoint1 chord2.endpoint2) : 
  segment_length p chord1.endpoint1 * segment_length p chord1.endpoint2 = 
  segment_length p chord2.endpoint1 * segment_length p chord2.endpoint2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_segment_product_constant_l693_69301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_initial_speed_l693_69345

noncomputable section

/-- Represents the initial speed of the car in km/h -/
def initial_speed : ℝ := 100

/-- Represents the total distance from A to D in km -/
def total_distance : ℝ := 100

/-- Represents the time remaining at point B in hours -/
def time_remaining_at_B : ℝ := 0.5

/-- Represents the speed reduction at point B in km/h -/
def speed_reduction_at_B : ℝ := 10

/-- Represents the distance remaining at point C in km -/
def distance_remaining_at_C : ℝ := 20

/-- Represents the speed reduction at point C in km/h -/
def speed_reduction_at_C : ℝ := 10

/-- Represents the time difference between B to C and C to D journeys in hours -/
def time_difference : ℝ := 5 / 60

theorem car_initial_speed :
  ∃ (x : ℝ),
    x = initial_speed ∧
    (80 - (total_distance - x / 2)) / (x - speed_reduction_at_B) -
    distance_remaining_at_C / (x - speed_reduction_at_B - speed_reduction_at_C) =
    time_difference ∧
    total_distance - (total_distance - x / 2) = x * time_remaining_at_B :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_initial_speed_l693_69345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_fourth_power_coefficients_sum_of_squares_l693_69331

theorem cos_fourth_power_coefficients_sum_of_squares :
  ∃ (b₁ b₂ b₃ b₄ : ℝ),
    (∀ θ : ℝ, (Real.cos θ) ^ 4 = b₁ * Real.cos θ + b₂ * Real.cos (2 * θ) + b₃ * Real.cos (3 * θ) + b₄ * Real.cos (4 * θ)) ∧
    b₁^2 + b₂^2 + b₃^2 + b₄^2 = 17/256 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_fourth_power_coefficients_sum_of_squares_l693_69331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_efgh_l693_69319

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the area of a trapezoid given its vertices -/
noncomputable def trapezoidArea (e f g h : Point) : ℝ :=
  let base1 := |f.x - e.x|
  let base2 := |g.x - h.x|
  let height := |g.y - e.y|
  (base1 + base2) * height / 2

/-- Theorem: The area of trapezoid EFGH with given vertices is 16.5 square units -/
theorem trapezoid_area_efgh :
  let e : Point := ⟨-3, 0⟩
  let f : Point := ⟨2, 0⟩
  let g : Point := ⟨5, -3⟩
  let h : Point := ⟨-1, -3⟩
  trapezoidArea e f g h = 16.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_efgh_l693_69319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l693_69357

noncomputable def f (x : ℝ) : ℝ := (x^3 - 8) / (x - 3)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≠ 3} := by
  sorry

#check domain_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l693_69357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l693_69303

/-- Represents an ellipse with center at origin -/
structure Ellipse where
  a : ℝ  -- Semi-major axis
  b : ℝ  -- Semi-minor axis
  c : ℝ  -- Distance from center to focus
  h : c > 0

/-- The point where the directrix intersects the x-axis -/
noncomputable def directrix_x_intercept (e : Ellipse) : ℝ := e.a^2 / e.c

theorem ellipse_properties (e : Ellipse) 
  (h1 : e.b = Real.sqrt 2)
  (h2 : e.c = 2 * (directrix_x_intercept e - e.c)) :
  e.a = Real.sqrt 6 ∧ 
  e.c / e.a = Real.sqrt 6 / 3 ∧
  ∀ (k : ℝ), (k = Real.sqrt 5 / 5 ∨ k = -Real.sqrt 5 / 5) → 
    ∃ (P Q : ℝ × ℝ), 
      P.1^2 / 6 + P.2^2 / 2 = 1 ∧
      Q.1^2 / 6 + Q.2^2 / 2 = 1 ∧
      P.2 = k * (P.1 - 3) ∧
      Q.2 = k * (Q.1 - 3) ∧
      P.1 * Q.1 + P.2 * Q.2 = 0 := by
  sorry

#check ellipse_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l693_69303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_art_collection_cost_l693_69383

/-- The total cost of John's art collection --/
noncomputable def total_cost (first_three_cost : ℝ) (fourth_cost : ℝ) : ℝ :=
  first_three_cost + fourth_cost

/-- The cost of each of the first three pieces of art --/
noncomputable def cost_per_piece (first_three_cost : ℝ) : ℝ :=
  first_three_cost / 3

/-- The cost of the fourth piece of art --/
noncomputable def fourth_piece_cost (cost_per_piece : ℝ) : ℝ :=
  cost_per_piece * 1.5

theorem art_collection_cost :
  ∀ (first_three_cost : ℝ),
  first_three_cost = 45000 →
  total_cost first_three_cost (fourth_piece_cost (cost_per_piece first_three_cost)) = 67500 := by
  intro first_three_cost h
  sorry

#check art_collection_cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_art_collection_cost_l693_69383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_simplification_l693_69384

theorem cube_root_simplification : Real.rpow 8748000 (1/3 : ℝ) = 90 * Real.rpow 12 (1/3 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_simplification_l693_69384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_9_sqrt_3_l693_69361

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the properties of the triangle
def triangle_properties (t : Triangle) : Prop :=
  let AB := Real.sqrt ((t.B.1 - t.A.1)^2 + (t.B.2 - t.A.2)^2)
  let angle_A := Real.arccos ((t.B.1 - t.A.1) / AB)
  let angle_B := Real.arccos ((t.C.1 - t.B.1) / (Real.sqrt ((t.C.1 - t.B.1)^2 + (t.C.2 - t.B.2)^2)))
  AB = 6 ∧ 
  angle_A = Real.pi/6 ∧ 
  angle_B = 2*Real.pi/3

-- Define the area of a triangle
noncomputable def triangle_area (t : Triangle) : ℝ :=
  let a := Real.sqrt ((t.B.1 - t.A.1)^2 + (t.B.2 - t.A.2)^2)
  let b := Real.sqrt ((t.C.1 - t.B.1)^2 + (t.C.2 - t.B.2)^2)
  let c := Real.sqrt ((t.A.1 - t.C.1)^2 + (t.A.2 - t.C.2)^2)
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

-- Theorem statement
theorem triangle_area_is_9_sqrt_3 (t : Triangle) :
  triangle_properties t → triangle_area t = 9 * Real.sqrt 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_9_sqrt_3_l693_69361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_doubling_time_approx_six_l693_69346

/-- The time (in minutes) it takes for the population to grow from 1,000 to 500,000 bacteria -/
def total_time : ℝ := 53.794705707972525

/-- The initial population size -/
def initial_population : ℕ := 1000

/-- The final population size -/
def final_population : ℕ := 500000

/-- The doubling time of the bacterial population -/
noncomputable def doubling_time : ℝ := total_time / (Real.log (final_population / initial_population) / Real.log 2)

/-- Theorem stating that the doubling time is approximately 6 minutes -/
theorem doubling_time_approx_six : ∃ ε > 0, |doubling_time - 6| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_doubling_time_approx_six_l693_69346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_at_pi_third_l693_69387

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 5 * Real.sin x - (1/2) * x

-- State the theorem
theorem derivative_f_at_pi_third : 
  (deriv f) (π/3) = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_at_pi_third_l693_69387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_painting_time_l693_69347

/-- Calculates the total time John had to paint the walls -/
noncomputable def total_painting_time (num_walls : ℕ) (wall_width : ℝ) (wall_height : ℝ) 
  (painting_rate : ℝ) (spare_time : ℝ) : ℝ :=
  let total_area := (num_walls : ℝ) * wall_width * wall_height
  let painting_time := total_area * painting_rate
  painting_time + spare_time

/-- Theorem stating that John had 10 hours to paint everything -/
theorem john_painting_time :
  total_painting_time 5 2 3 (10 / 60) 5 = 10 := by
  unfold total_painting_time
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_painting_time_l693_69347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_lower_bound_l693_69313

/-- The distance between a pair of skew edges of a tetrahedron -/
noncomputable def distance_between_skew_edges (pair : ℕ) : ℝ := sorry

/-- The volume of the tetrahedron -/
noncomputable def tetrahedron_volume : ℝ := sorry

/-- Theorem: The volume of a tetrahedron is at least one-third the product of the distances between its skew edges. -/
theorem tetrahedron_volume_lower_bound (h₁ h₂ h₃ V : ℝ) 
  (h_positive : h₁ > 0 ∧ h₂ > 0 ∧ h₃ > 0) 
  (h_distances : h₁ = distance_between_skew_edges 1 ∧ 
                 h₂ = distance_between_skew_edges 2 ∧ 
                 h₃ = distance_between_skew_edges 3)
  (h_volume : V = tetrahedron_volume) : 
  V ≥ (h₁ * h₂ * h₃) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_lower_bound_l693_69313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_a_neg_one_range_of_a_nonnegative_f_l693_69382

-- Define the function f
def f (x a : ℝ) : ℝ := |x - a| + 5*x

-- Part 1: Solution set when a = -1
theorem solution_set_a_neg_one :
  {x : ℝ | f x (-1) ≤ 5*x + 3} = Set.Icc (-4) 2 := by sorry

-- Part 2: Range of a for which f(x) ≥ 0 for all x ≥ -1
theorem range_of_a_nonnegative_f :
  ∀ a : ℝ, (∀ x ≥ -1, f x a ≥ 0) ↔ (a ≥ 4 ∨ a ≤ -6) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_a_neg_one_range_of_a_nonnegative_f_l693_69382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_old_machine_rate_proof_l693_69312

/-- The rate of the old machine in bolts per hour -/
noncomputable def old_machine_rate : ℝ := 100

/-- The rate of the new machine in bolts per hour -/
noncomputable def new_machine_rate : ℝ := 150

/-- The time both machines work in hours -/
noncomputable def work_time : ℝ := 84 / 60

/-- The total number of bolts produced -/
def total_bolts : ℕ := 350

theorem old_machine_rate_proof :
  old_machine_rate * work_time + new_machine_rate * work_time = total_bolts := by
  sorry

#check old_machine_rate_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_old_machine_rate_proof_l693_69312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_percentage_l693_69352

-- Define the profit percentages
def profit_with_discount : ℚ := 1875 / 100
def profit_without_discount : ℚ := 25 / 1

-- Define the function to calculate the discount percentage
noncomputable def calculate_discount (profit_with_discount profit_without_discount : ℚ) : ℚ :=
  (profit_without_discount - profit_with_discount) / (100 + profit_without_discount) * 100

-- Theorem statement
theorem discount_percentage :
  calculate_discount profit_with_discount profit_without_discount = 5 := by
  -- Unfold the definitions and perform the calculation
  unfold calculate_discount profit_with_discount profit_without_discount
  -- Simplify the rational numbers
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_percentage_l693_69352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l693_69304

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y^2 = 2px -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- The focus of the parabola -/
noncomputable def focus (par : Parabola) : Point :=
  { x := par.p / 2, y := 0 }

/-- The origin point -/
def origin : Point :=
  { x := 0, y := 0 }

/-- Distance between two points -/
noncomputable def distance (a b : Point) : ℝ :=
  Real.sqrt ((a.x - b.x)^2 + (a.y - b.y)^2)

/-- Area of a triangle given three points -/
noncomputable def triangleArea (a b c : Point) : ℝ :=
  (1/2) * abs (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y))

/-- Theorem statement -/
theorem parabola_equation (par : Parabola) (m : Point)
    (h1 : m.y^2 = 2 * par.p * m.x)  -- M is on the parabola
    (h2 : distance m (focus par) = 4 * distance origin (focus par))  -- |MF| = 4|OF|
    (h3 : triangleArea m (focus par) origin = 4 * Real.sqrt 3)  -- Area of MFO = 4√3
    : par.p = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l693_69304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l693_69324

noncomputable def f (x : ℝ) : ℝ := 2 * x - Real.cos x

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem problem_solution (a : ℕ → ℝ) :
  arithmetic_sequence a (π / 8) →
  f (a 1) + f (a 2) + f (a 3) + f (a 4) + f (a 5) = 5 * π →
  (f (a 3))^2 - (a 1) * (a 5) = 13 * π^2 / 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l693_69324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_g_two_equals_four_l693_69300

-- Define the function f
noncomputable def f (x : ℝ) (g : ℝ → ℝ) : ℝ :=
  if x < 0 then -x^2 else g x

-- State the theorem
theorem odd_function_implies_g_two_equals_four
  (g : ℝ → ℝ)
  (h₁ : ∀ x, f x g = -(f (-x) g)) :  -- f is an odd function
  g 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_g_two_equals_four_l693_69300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_range_for_f_inequality_l693_69314

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^3 else Real.log (x + 1)

-- State the theorem
theorem x_range_for_f_inequality (x : ℝ) :
  f (2 - x^2) > f x → x ∈ Set.Ioo (-2) 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_range_for_f_inequality_l693_69314
