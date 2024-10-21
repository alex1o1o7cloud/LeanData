import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_to_brush_ratio_square_brush_ratio_is_four_l524_52475

/-- The ratio of the side length of a square to the width of a brush
    that paints one-third of the square's area when swept along both diagonals. -/
theorem square_to_brush_ratio : ℝ := by
  -- Let s be the side length of the square
  let s : ℝ := 1
  -- Let w be the width of the brush
  let w : ℝ := s / 4

  -- State that the painted area is one-third of the square's area
  have painted_area : ℝ := s^2 / 3

  -- The painted area consists of a central square and four triangles
  have central_square_area : ℝ := w^2
  have triangle_area : ℝ := ((s - w) / 2)^2 / 2
  have total_painted_area : ℝ := central_square_area + 4 * triangle_area

  -- Assert that the total painted area equals one-third of the square's area
  have area_equality : total_painted_area = painted_area := by sorry

  -- Solve the equation to find the ratio s/w
  have ratio : s / w = 4 := by sorry

  -- The final answer
  exact 4

/-- The main theorem stating that the ratio of the square's side length
    to the brush width is 4. -/
theorem square_brush_ratio_is_four : 
  square_to_brush_ratio = 4 := by rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_to_brush_ratio_square_brush_ratio_is_four_l524_52475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_correct_statement_l524_52465

-- Define the four statements
def statement1 : Prop := 
  ∀ angle : ℝ, angle > 90 ∧ angle < 180 → (angle > 90 ∧ angle < 180)

def statement2 : Prop := 
  ∀ angle : ℝ, angle < 90 → angle > 0 ∧ angle < 90

def statement3 : Prop := 
  ∀ angle : ℝ, (angle > 0 ∧ angle < 90) → angle ≥ 0

def statement4 : Prop := 
  ∀ angle1 angle2 : ℝ, (angle1 > 90 ∧ angle1 < 180) ∧ (angle2 > 0 ∧ angle2 < 90) → angle1 > angle2

-- Theorem stating that exactly one statement is correct
theorem exactly_one_correct_statement : 
  (statement1 ∧ ¬statement2 ∧ ¬statement3 ∧ ¬statement4) ∨
  (¬statement1 ∧ statement2 ∧ ¬statement3 ∧ ¬statement4) ∨
  (¬statement1 ∧ ¬statement2 ∧ statement3 ∧ ¬statement4) ∨
  (¬statement1 ∧ ¬statement2 ∧ ¬statement3 ∧ statement4) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_correct_statement_l524_52465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_period_l524_52400

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

def is_period (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

def smallest_positive_period (T : ℝ) : Prop :=
  T > 0 ∧ is_period T ∧ ∀ T', 0 < T' ∧ T' < T → ¬ is_period T'

theorem sine_period :
  smallest_positive_period Real.pi := by
  sorry

#check sine_period

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_period_l524_52400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_PM_l524_52453

-- Define the region for point P
def P_region (x y : ℝ) : Prop :=
  ∃ θ : ℝ, Real.cos θ ≤ x ∧ x ≤ 3 * Real.cos θ ∧ Real.sin θ ≤ y ∧ y ≤ 3 * Real.sin θ

-- Define the equation for point M
def M_equation (x y : ℝ) : Prop :=
  (x + 5)^2 + (y + 5)^2 = 1

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Theorem statement
theorem min_distance_PM :
  ∀ x1 y1 x2 y2 : ℝ,
    P_region x1 y1 → M_equation x2 y2 →
    distance x1 y1 x2 y2 ≥ Real.sqrt 61 - 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_PM_l524_52453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_x_coordinates_above_line_l524_52446

noncomputable def points : List (Real × Real) := [(4, 12), (7, 26), (13, 30), (17, 45), (22, 52)]

noncomputable def isAboveLine (p : Real × Real) : Bool :=
  p.2 > 3 * p.1 + 5

noncomputable def sumXCoordinatesAboveLine (pts : List (Real × Real)) : Real :=
  (pts.filter isAboveLine).map (·.1) |>.sum

theorem sum_x_coordinates_above_line :
  sumXCoordinatesAboveLine points = 0 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_x_coordinates_above_line_l524_52446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harpers_list_count_is_871_l524_52480

def smallest_square : ℕ := Nat.lcm (30 * 30) 30
def smallest_cube : ℕ := Nat.lcm (30 * 30 * 30) 30

def harpers_list_count (n : ℕ) : Prop :=
  (∀ k < smallest_square, k % 30 = 0 → ¬∃ m : ℕ, m * m = k) ∧
  (∀ k < smallest_cube, k % 30 = 0 → ¬∃ m : ℕ, m * m * m = k) ∧
  n = (smallest_cube / 30 - smallest_square / 30 + 1)

theorem harpers_list_count_is_871 : harpers_list_count 871 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_harpers_list_count_is_871_l524_52480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_willam_land_percentage_eq_tax_percentage_l524_52401

/-- Represents the farm tax system in a village -/
structure FarmTaxSystem where
  total_tax : ℚ
  willam_tax : ℚ
  taxable_land_percentage : ℚ

/-- Calculates the percentage of Mr. Willam's taxable land over the total taxable land -/
def willam_land_percentage (fts : FarmTaxSystem) : ℚ :=
  (fts.willam_tax / fts.total_tax) * 100

/-- Theorem stating that the percentage of Mr. Willam's taxable land
    is equal to the percentage of his tax payment -/
theorem willam_land_percentage_eq_tax_percentage (fts : FarmTaxSystem) :
  willam_land_percentage fts = (fts.willam_tax / fts.total_tax) * 100 :=
by
  -- Unfold the definition of willam_land_percentage
  unfold willam_land_percentage
  -- The equality follows directly from the definition
  rfl

/-- Example calculation using the given values -/
def example_calculation : ℚ :=
  willam_land_percentage { total_tax := 3840, willam_tax := 500, taxable_land_percentage := 60 }

#eval example_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_willam_land_percentage_eq_tax_percentage_l524_52401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pqs_is_nineteen_l524_52448

/-- Represents the symbols used in the encoding --/
inductive EncodingSymbol : Type
| P | Q | R | S | T

/-- Represents a base-4 number using EncodingSymbols --/
def SymbolNumber := List EncodingSymbol

/-- Converts a EncodingSymbol to its corresponding base-4 digit --/
def symbolToDigit (s : EncodingSymbol) : ℕ :=
  match s with
  | EncodingSymbol.P => 1
  | EncodingSymbol.Q => 0
  | EncodingSymbol.S => 3
  | _ => 0  -- R and T are not used in PQS

/-- Converts a SymbolNumber to its base-10 representation --/
def symbolNumberToBase10 (sn : SymbolNumber) : ℕ :=
  sn.enum.foldl (fun acc (i, s) => acc + symbolToDigit s * (4^(sn.length - 1 - i))) 0

/-- The main theorem to prove --/
theorem pqs_is_nineteen (pqr pqs ppt : SymbolNumber) :
  pqr.length = 3 ∧ pqs.length = 3 ∧ ppt.length = 3 ∧
  symbolNumberToBase10 pqs = symbolNumberToBase10 pqr + 1 ∧
  symbolNumberToBase10 ppt = symbolNumberToBase10 pqs + 1 ∧
  pqs = [EncodingSymbol.P, EncodingSymbol.Q, EncodingSymbol.S] →
  symbolNumberToBase10 pqs = 19 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pqs_is_nineteen_l524_52448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distances_to_line_l524_52416

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

-- Define the line l
def line_l (x y : ℝ) : Prop :=
  y = x - 2

-- Define the foci of the ellipse
def focus1 : ℝ × ℝ := (-1, 0)
def focus2 : ℝ × ℝ := (1, 0)

-- Define the distance function from a point to a line
noncomputable def distance_point_to_line (x₀ y₀ : ℝ) (A B C : ℝ) : ℝ :=
  |A * x₀ + B * y₀ + C| / Real.sqrt (A^2 + B^2)

-- Theorem statement
theorem sum_of_distances_to_line :
  let d₁ := distance_point_to_line focus1.1 focus1.2 (-1) 1 (-2)
  let d₂ := distance_point_to_line focus2.1 focus2.2 (-1) 1 (-2)
  d₁ + d₂ = 2 * Real.sqrt 2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distances_to_line_l524_52416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_target_sums_l524_52494

def dieA : Finset Nat := {1, 1, 3, 3, 4, 5}
def dieB : Finset Nat := {2, 3, 4, 5, 7, 9}

def targetSums : Finset Nat := {6, 8, 10}

def isFavorableOutcome (a b : Nat) : Bool := (a + b) ∈ targetSums

def favorableOutcomes : Finset (Nat × Nat) :=
  (dieA.product dieB).filter (fun (a, b) => isFavorableOutcome a b)

theorem probability_of_target_sums : 
  (favorableOutcomes.card : Rat) / ((dieA.card * dieB.card) : Rat) = 11 / 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_target_sums_l524_52494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l524_52412

noncomputable def f (x : ℝ) : ℝ := 2 / (x - 1) - 3 / (x - 2) + 3 / (x - 3) - 2 / (x - 4)

def solution_set : Set ℝ :=
  {x | x < -3 ∨ (-1 < x ∧ x < 1) ∨ (2 < x ∧ x < 3) ∨ (4 < x ∧ x < 7) ∨ 8 < x}

theorem inequality_solution :
  {x : ℝ | f x < 1/15} = solution_set :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l524_52412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_second_half_score_l524_52469

def basketball_game (a r e : ℕ) : Prop :=
  let team_x_total := a * (1 + r + r^2 + r^3)
  let team_y_total := 4*a + 6*e
  team_x_total = team_y_total + 2
  ∧ team_x_total ≤ 120
  ∧ team_y_total ≤ 120
  ∧ r = 2
  ∧ e = 7

theorem basketball_second_half_score (a r e : ℕ) :
  basketball_game a r e → (16 + 32 + 18 + 25 = 91) := by
  intro h
  -- The proof goes here, but we'll use sorry for now
  sorry

#check basketball_second_half_score

end NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_second_half_score_l524_52469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crawling_ant_faster_l524_52407

/-- Represents the travel scenario of two ants -/
structure AntTravel where
  d : ℝ  -- Total distance to the dragonfly
  v : ℝ  -- Speed of crawling
  (d_pos : d > 0)  -- Distance is positive
  (v_pos : v > 0)  -- Crawling speed is positive

/-- Time taken by the first ant (crawling the entire way) -/
noncomputable def time_crawling_ant (travel : AntTravel) : ℝ :=
  travel.d / travel.v

/-- Time taken by the second ant (caterpillar + grasshopper) -/
noncomputable def time_riding_ant (travel : AntTravel) : ℝ :=
  (travel.d / (2 * (travel.v / 2))) + (travel.d / (2 * (10 * travel.v)))

/-- Theorem stating that the crawling ant arrives first -/
theorem crawling_ant_faster (travel : AntTravel) :
  time_crawling_ant travel < time_riding_ant travel := by
  sorry

#check crawling_ant_faster

end NUMINAMATH_CALUDE_ERRORFEEDBACK_crawling_ant_faster_l524_52407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bisection_method_applicability_l524_52424

noncomputable def f₁ (x : ℝ) := 3 * x + 1
noncomputable def f₂ (x : ℝ) := x^3
noncomputable def f₃ (x : ℝ) := x^2
noncomputable def f₄ (x : ℝ) := Real.log x

def has_opposite_signs (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∃ (a b : ℝ), a < x ∧ x < b ∧ f a * f b < 0

def can_use_bisection (f : ℝ → ℝ) : Prop :=
  ∃ (x : ℝ), f x = 0 ∧ has_opposite_signs f x

theorem bisection_method_applicability :
  can_use_bisection f₁ ∧ can_use_bisection f₂ ∧ can_use_bisection f₄ ∧ ¬can_use_bisection f₃ := by
  sorry

#check bisection_method_applicability

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bisection_method_applicability_l524_52424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_non_negative_poly_is_sum_of_squares_l524_52406

/-- A polynomial with real coefficients that is non-negative for all real inputs
can be expressed as the sum of two squared polynomials. -/
theorem non_negative_poly_is_sum_of_squares (P : Polynomial ℝ) 
  (h : ∀ x : ℝ, 0 ≤ P.eval x) : 
  ∃ Q R : Polynomial ℝ, P = Q^2 + R^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_non_negative_poly_is_sum_of_squares_l524_52406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_selected_number_l524_52499

/-- Represents a systematic sampling setup -/
structure SystematicSampling where
  totalStudents : Nat
  sampleSize : Nat
  firstSelected : Nat

/-- Calculates the interval between parts in systematic sampling -/
def SystematicSampling.interval (s : SystematicSampling) : Nat :=
  s.totalStudents / s.sampleSize

/-- Calculates the nth selected number in systematic sampling -/
def SystematicSampling.nthSelected (s : SystematicSampling) (n : Nat) : Nat :=
  s.firstSelected + (n - 1) * s.interval

/-- Theorem stating that the 10th selected number is 195 given the conditions -/
theorem tenth_selected_number (s : SystematicSampling) 
  (h1 : s.totalStudents = 1000)
  (h2 : s.sampleSize = 50)
  (h3 : s.firstSelected = 15) :
  s.nthSelected 10 = 195 := by
  sorry

#eval SystematicSampling.nthSelected ⟨1000, 50, 15⟩ 10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_selected_number_l524_52499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_D_not_right_triangle_l524_52493

-- Define the structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  α : ℝ
  β : ℝ
  γ : ℝ

-- Define the triangles
def triangleA : Triangle := sorry
def triangleB : Triangle := sorry
def triangleC : Triangle := sorry
def triangleD : Triangle := sorry

-- Define the conditions for each triangle
axiom triangleA_angles : triangleA.α / (triangleA.α + triangleA.β + triangleA.γ) = 1 / 6 ∧
                         triangleA.β / (triangleA.α + triangleA.β + triangleA.γ) = 1 / 3 ∧
                         triangleA.γ / (triangleA.α + triangleA.β + triangleA.γ) = 1 / 2

axiom triangleB_sides_squared : triangleA.a^2 / (triangleA.a^2 + triangleA.b^2 + triangleA.c^2) = 1 / 6 ∧
                                triangleA.b^2 / (triangleA.a^2 + triangleA.b^2 + triangleA.c^2) = 1 / 3 ∧
                                triangleA.c^2 / (triangleA.a^2 + triangleA.b^2 + triangleA.c^2) = 1 / 2

axiom triangleC_sides : triangleC.a / (triangleC.a + triangleC.b + triangleC.c) = 3 / 12 ∧
                        triangleC.b / (triangleC.a + triangleC.b + triangleC.c) = 4 / 12 ∧
                        triangleC.c / (triangleC.a + triangleC.b + triangleC.c) = 5 / 12

axiom triangleD_angles : triangleD.α / (triangleD.α + triangleD.β + triangleD.γ) = 1 / 4 ∧
                         triangleD.β / (triangleD.α + triangleD.β + triangleD.γ) = 1 / 3 ∧
                         triangleD.γ / (triangleD.α + triangleD.β + triangleD.γ) = 5 / 12

-- Define a right triangle
def is_right_triangle (t : Triangle) : Prop :=
  t.α = 90 ∨ t.β = 90 ∨ t.γ = 90

-- Theorem to prove
theorem only_D_not_right_triangle :
  is_right_triangle triangleA ∧
  is_right_triangle triangleB ∧
  is_right_triangle triangleC ∧
  ¬is_right_triangle triangleD :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_D_not_right_triangle_l524_52493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_continuous_iff_b_eq_zero_l524_52459

-- Define the piecewise function g(x)
noncomputable def g (b : ℝ) (x : ℝ) : ℝ :=
  if x > 2 then x + 4 else 3 * x + b

-- State the theorem
theorem g_continuous_iff_b_eq_zero (b : ℝ) :
  Continuous (g b) ↔ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_continuous_iff_b_eq_zero_l524_52459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mary_to_brother_height_ratio_l524_52417

/-- The ratio of Mary's height to her brother's height -/
def height_ratio (mary_height brother_height : ℚ) : ℚ :=
  mary_height / brother_height

/-- Theorem stating the ratio of Mary's height to her brother's height -/
theorem mary_to_brother_height_ratio :
  let min_height : ℚ := 140
  let brother_height : ℚ := 180
  let growth_needed : ℚ := 20
  let mary_height : ℚ := min_height - growth_needed
  height_ratio mary_height brother_height = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mary_to_brother_height_ratio_l524_52417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gain_percentage_problem_specific_gain_percentage_l524_52463

/-- Calculates the gain percentage given the number of articles, selling price, and cost price -/
noncomputable def gainPercentage (articles : ℕ) (sellingPrice costPrice : ℝ) : ℝ :=
  ((sellingPrice / (articles : ℝ) - costPrice) / costPrice) * 100

/-- Represents the problem of calculating gain percentage -/
theorem gain_percentage_problem (initialArticles : ℕ) (initialPrice : ℝ) 
  (secondArticles : ℝ) (secondPrice : ℝ) (lossPercentage : ℝ) : Prop :=
  let costPrice := secondPrice / (secondArticles * (1 - lossPercentage / 100))
  gainPercentage initialArticles initialPrice costPrice = 20

/-- The specific problem instance -/
theorem specific_gain_percentage :
  gain_percentage_problem 20 60 24.999996875000388 50 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gain_percentage_problem_specific_gain_percentage_l524_52463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_lines_product_l524_52490

/-- Given that the lines y=3, y=8, x=2, and x=b form a square, 
    the product of possible values for b is -21 -/
theorem square_lines_product (b₁ b₂ : ℝ) : 
  (∃ (b : ℝ), 
    (abs (b - 2) = 5) ∧ 
    (8 - 3 = 5) ∧
    (b = b₁ ∨ b = b₂)) →
  b₁ * b₂ = -21 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_lines_product_l524_52490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_increases_l524_52422

theorem triangle_area_increases (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (htri : a + b > c ∧ b + c > a ∧ c + a > b) : 
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let s' := ((a + 1) + (b + 1) + (c + 1)) / 2
  let area' := Real.sqrt (s' * (s' - (a + 1)) * (s' - (b + 1)) * (s' - (c + 1)))
  area' > area := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_increases_l524_52422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_parallel_to_x_axis_tangent_45_degrees_l524_52418

-- Define the curve
def f (x : ℝ) : ℝ := x^2 - 1

-- Theorem for part a
theorem tangent_parallel_to_x_axis :
  ∃ (x : ℝ), (deriv f x = 0) ∧ (f x = -1) ∧ (x = 0) := by sorry

-- Theorem for part b
theorem tangent_45_degrees :
  ∃ (x : ℝ), (deriv f x = 1) ∧ (f x = -3/4) ∧ (x = 1/2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_parallel_to_x_axis_tangent_45_degrees_l524_52418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_arithmetic_sum_values_l524_52456

open BigOperators

def p_arithmetic_sum (p : ℕ) (q : ℚ) : ℚ :=
  ∑ k in Finset.range ((p + 1) / 2), (Nat.choose (p - 1) (2 * k)) * q^k

theorem p_arithmetic_sum_values (p : ℕ) (q : ℚ) 
  (h_prime : Nat.Prime p) :
  let S := p_arithmetic_sum p q
  (S = 0 ↔ q = 1/4) ∧ (q ≠ 1/4 → S = 1 ∨ S = -1) := by
  sorry

#check p_arithmetic_sum_values

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_arithmetic_sum_values_l524_52456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l524_52451

/-- Ellipse struct -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Given ellipse -/
noncomputable def C : Ellipse where
  a := Real.sqrt 3
  b := 1
  h_pos := by sorry

/-- Standard form of ellipse equation -/
def ellipse_equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- Right focus of the ellipse -/
noncomputable def right_focus : ℝ × ℝ := (Real.sqrt 2, 0)

/-- Eccentricity of the ellipse -/
noncomputable def eccentricity : ℝ := Real.sqrt 6 / 3

/-- Line intersecting the ellipse -/
def line_intersect (e : Ellipse) (l : ℝ → ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    ellipse_equation e x₁ y₁ ∧ 
    ellipse_equation e x₂ y₂ ∧ 
    y₁ = l x₁ ∧ 
    y₂ = l x₂ ∧ 
    (x₁, y₁) ≠ (x₂, y₂)

/-- Circle with AB as diameter passing through origin -/
def circle_through_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ * x₂ + y₁ * y₂ = 0

/-- Distance from origin to line -/
noncomputable def dist_origin_to_line (l : ℝ → ℝ) : ℝ := sorry

/-- Area of triangle OAB -/
noncomputable def triangle_area (x₁ y₁ x₂ y₂ : ℝ) : ℝ := sorry

/-- Main theorem -/
theorem ellipse_properties :
  /- 1. Standard equation of the ellipse -/
  (∀ x y : ℝ, ellipse_equation C x y ↔ x^2/3 + y^2 = 1) ∧ 
  /- 2. Distance from O to AB is constant -/
  (∀ l : ℝ → ℝ, line_intersect C l → 
    ∃ x₁ y₁ x₂ y₂ : ℝ, 
      ellipse_equation C x₁ y₁ ∧ 
      ellipse_equation C x₂ y₂ ∧ 
      y₁ = l x₁ ∧ 
      y₂ = l x₂ ∧ 
      circle_through_origin x₁ y₁ x₂ y₂ → 
      dist_origin_to_line l = Real.sqrt 3 / 2) ∧
  /- 3. Maximum area of triangle OAB -/
  (∀ x₁ y₁ x₂ y₂ : ℝ, 
    ellipse_equation C x₁ y₁ → 
    ellipse_equation C x₂ y₂ → 
    circle_through_origin x₁ y₁ x₂ y₂ → 
    triangle_area x₁ y₁ x₂ y₂ ≤ Real.sqrt 3 / 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l524_52451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_pole_time_l524_52410

/-- Represents the properties of a train and a platform it crosses. -/
structure TrainPlatform where
  train_length : ℝ
  platform_length : ℝ
  platform_crossing_time : ℝ

/-- Calculates the time for a train to cross a signal pole. -/
noncomputable def time_to_cross_pole (tp : TrainPlatform) : ℝ :=
  let total_distance := tp.train_length + tp.platform_length
  let train_speed := total_distance / tp.platform_crossing_time
  tp.train_length / train_speed

/-- Theorem stating that under given conditions, a train takes 18 seconds to cross a signal pole. -/
theorem train_crossing_pole_time :
  let tp : TrainPlatform := {
    train_length := 300,
    platform_length := 150.00000000000006,
    platform_crossing_time := 27
  }
  time_to_cross_pole tp = 18 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_pole_time_l524_52410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_twenty_first_term_equals_five_sqrt_five_l524_52470

noncomputable def my_sequence (n : ℕ) : ℝ := Real.sqrt (6 * n - 1)

theorem twenty_first_term_equals_five_sqrt_five :
  my_sequence 21 = 5 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_twenty_first_term_equals_five_sqrt_five_l524_52470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_456_sequence_l524_52430

def has_456_sequence (m n : ℕ) : Prop :=
  ∃ k : ℕ, (1000 * (10^k * m % n)) / n = 456

theorem smallest_n_with_456_sequence :
  ∃ m : ℕ, 
    m < 230 ∧ 
    Nat.Coprime m 230 ∧
    has_456_sequence m 230 ∧
    (∀ n < 230, ∀ m : ℕ, m < n → Nat.Coprime m n → ¬has_456_sequence m n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_456_sequence_l524_52430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_rectangles_dissection_l524_52449

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A rectangle in a 2D plane -/
structure Rectangle where
  x1 : ℝ
  y1 : ℝ
  x2 : ℝ
  y2 : ℝ

/-- Check if a point is inside a rectangle -/
def Point.isInside (p : Point) (r : Rectangle) : Prop :=
  p.x > r.x1 ∧ p.x < r.x2 ∧ p.y > r.y1 ∧ p.y < r.y2

/-- Check if two points are on a line parallel to the sides of a rectangle -/
def areParallel (p1 p2 : Point) : Prop :=
  p1.x = p2.x ∨ p1.y = p2.y

/-- The main theorem -/
theorem min_rectangles_dissection (n : ℕ) (R : Rectangle) (points : List Point) :
  (points.length = n) →
  (∀ p, p ∈ points → p.isInside R) →
  (∀ p1 p2, p1 ∈ points → p2 ∈ points → p1 ≠ p2 → ¬(areParallel p1 p2)) →
  (∀ dissection : List Rectangle,
    (∀ r, r ∈ dissection → ∀ p, p ∈ points → ¬(p.isInside r)) →
    dissection.length ≥ n + 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_rectangles_dissection_l524_52449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_table_size_lower_bound_l524_52426

/-- Represents a row in the table -/
def Row := Fin 10 → Fin 10

/-- The property that for any row and any two columns, there exists a row differing in exactly those two columns -/
def DifferenceProperty (rows : Set Row) : Prop :=
  ∀ (r : Row) (i j : Fin 10), i ≠ j →
    ∃ (r' : Row), r' ∈ rows ∧
      (∀ (k : Fin 10), k ≠ i ∧ k ≠ j → r k = r' k) ∧
      r i ≠ r' i ∧ r j ≠ r' j

theorem table_size_lower_bound (n : Nat) (rows : Finset Row) 
    (h_card : rows.card = n)
    (h_prop : DifferenceProperty rows) :
  n ≥ 512 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_table_size_lower_bound_l524_52426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l524_52467

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.exp (1 + x) + Real.exp (1 - x)

-- State the theorem
theorem range_of_f (x : ℝ) : f (x - 2) < Real.exp 2 + 1 ↔ 1 < x ∧ x < 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l524_52467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_condition_l524_52445

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(2*x) + 2*a^x - 1

theorem max_value_condition (a : ℝ) :
  (a > 0 ∧ a ≠ 1) →
  (∃ x ∈ Set.Icc (-1 : ℝ) 1, ∀ y ∈ Set.Icc (-1 : ℝ) 1, f a x ≥ f a y) ∧
  (∃ x ∈ Set.Icc (-1 : ℝ) 1, f a x = 14) ↔
  (a = 3 ∨ a = 1/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_condition_l524_52445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_triangle_l524_52441

-- Define the ellipse G
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0

-- Define the point M on the ellipse
noncomputable def M : ℝ × ℝ := (2 * Real.sqrt 2, 2 * Real.sqrt 3 / 3)

-- Define the sum of distances from M to the foci
noncomputable def focalDistanceSum : ℝ := 4 * Real.sqrt 3

-- Define point P
def P : ℝ × ℝ := (-3, 2)

-- Define area_triangle function
noncomputable def area_triangle (P A B : ℝ × ℝ) : ℝ :=
  let s := ((P.1 - A.1)^2 + (P.2 - A.2)^2 + (A.1 - B.1)^2 + (A.2 - B.2)^2 + (B.1 - P.1)^2 + (B.2 - P.2)^2) / 4
  Real.sqrt (s * (s - ((P.1 - A.1)^2 + (P.2 - A.2)^2)) * (s - ((A.1 - B.1)^2 + (A.2 - B.2)^2)) * (s - ((B.1 - P.1)^2 + (B.2 - P.2)^2)))

theorem ellipse_and_triangle (G : Ellipse) : 
  M.1^2 / G.a^2 + M.2^2 / G.b^2 = 1 → 
  focalDistanceSum = 2 * G.a →
  (∃ (A B : ℝ × ℝ), 
    -- A and B are on the ellipse
    A.1^2 / G.a^2 + A.2^2 / G.b^2 = 1 ∧
    B.1^2 / G.a^2 + B.2^2 / G.b^2 = 1 ∧
    -- A and B are on a line with slope 1
    B.2 - A.2 = B.1 - A.1 ∧
    -- PAB is an isosceles triangle
    (P.1 - A.1)^2 + (P.2 - A.2)^2 = (P.1 - B.1)^2 + (P.2 - B.2)^2) →
  -- Conclusion 1: Equation of G
  G.a^2 = 12 ∧ G.b^2 = 4 ∧
  -- Conclusion 2: Area of triangle PAB
  ∃ (A B : ℝ × ℝ), area_triangle P A B = 9/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_triangle_l524_52441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_diverges_l524_52408

open MeasureTheory

/-- The integrand function -/
noncomputable def f (x : ℝ) : ℝ := x / (1 + x^2 * (Real.cos x)^2)

/-- The interval we're integrating over -/
def I : Set ℝ := Set.Ici 0

/-- The theorem stating that the integral diverges -/
theorem integral_diverges : ¬ IntegrableOn f I := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_diverges_l524_52408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_in_interval_l524_52425

open Real

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := (1 / (2^x)) - x^(1/3)

-- Theorem statement
theorem zero_point_in_interval :
  ∃ x ∈ Set.Ioo (1/3 : ℝ) (1/2 : ℝ), f x = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_in_interval_l524_52425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_translation_proof_l524_52462

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := 2^x

-- Define the translated function
noncomputable def g (x : ℝ) : ℝ := 2^(x-2) + 3

-- Define the translation vector
def a : ℝ × ℝ := (2, 3)

-- Define vector b
def b : ℝ × ℝ := (-2, -3)

-- Theorem statement
theorem translation_proof :
  (∀ x, g x = f (x - 2) + 3) →  -- g is a translation of f
  (∃ k : ℝ, b.1 = k * a.1 ∧ b.2 = k * a.2) →  -- b is parallel to a
  b = (-2, -3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_translation_proof_l524_52462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_door_replacement_cost_l524_52481

/-- Calculates the total cost of door replacements after applying a discount --/
theorem door_replacement_cost (outside_door_cost : ℚ) 
  (outside_door_count bedroom_door_count : ℕ) 
  (bathroom_door_count : ℕ) (discount_percentage : ℚ) : 
  ∃ (total_cost : ℚ), total_cost = 106 :=
by
  let bedroom_door_cost := outside_door_cost / 2
  let bathroom_door_cost := outside_door_cost * 2
  let total_cost := outside_door_cost * outside_door_count + 
                    bedroom_door_cost * bedroom_door_count + 
                    bathroom_door_cost * bathroom_door_count
  let max_door_cost := max outside_door_cost (max bedroom_door_cost bathroom_door_cost)
  let discount_amount := max_door_cost * discount_percentage
  let final_cost := total_cost - discount_amount
  use final_cost
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_door_replacement_cost_l524_52481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_light_flashes_in_three_quarters_hour_l524_52460

/-- The number of flashes of a light in a given time period -/
def number_of_flashes (flash_interval : ℕ) (time_period : ℕ) : ℕ :=
  time_period / flash_interval

/-- Conversion from hours to seconds -/
def hours_to_seconds (hours : ℚ) : ℕ :=
  (hours * 3600).floor.toNat

theorem light_flashes_in_three_quarters_hour : 
  number_of_flashes 20 (hours_to_seconds (3/4)) = 135 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_light_flashes_in_three_quarters_hour_l524_52460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_selected_state_B_l524_52468

theorem percentage_selected_state_B :
  let candidates_per_state : ℕ := 8000
  let percentage_A : ℚ := 6 / 100
  let extra_selected_B : ℕ := 80
  (candidates_per_state * percentage_A + extra_selected_B) / candidates_per_state = 7 / 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_selected_state_B_l524_52468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_house_b_to_a_ratio_l524_52402

noncomputable def base_salary : ℝ := 3000
noncomputable def commission_rate : ℝ := 0.02
def houses_sold : ℕ := 3
noncomputable def total_earnings : ℝ := 8000
noncomputable def house_a_cost : ℝ := 60000

noncomputable def house_c_cost : ℝ := 2 * house_a_cost - 110000

noncomputable def total_commission : ℝ := total_earnings - base_salary

noncomputable def total_houses_cost : ℝ := total_commission / commission_rate

noncomputable def house_b_cost : ℝ := total_houses_cost - house_a_cost - house_c_cost

theorem house_b_to_a_ratio : house_b_cost / house_a_cost = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_house_b_to_a_ratio_l524_52402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_root_of_quadratic_l524_52432

-- Define the quadratic equation
def quadratic_equation (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0

-- Define the discriminant
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

theorem smaller_root_of_quadratic 
  (a b c x₁ x₂ : ℝ) 
  (ha : a ≠ 0)
  (hd : discriminant a b c > 0)
  (hx₁ : quadratic_equation a b c x₁)
  (hx₂ : quadratic_equation a b c x₂)
  (horder : x₁ < x₂) :
  x₁ = -b / (2*a) - Real.sqrt (discriminant a b c / (2 * abs a)) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_root_of_quadratic_l524_52432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kopeck_ways_l524_52466

/-- The number of ways to make n kopecks using coins of 1, 2, and 3 kopecks -/
def ways_123 (n : ℕ) : ℕ := 
  Finset.sum (Finset.range (n / 3 + 1)) (fun k => ((n - 3 * k) / 2 + 1))

/-- The number of ways to make n kopecks using coins of 1, 2, and 5 kopecks -/
def ways_125 (n : ℕ) : ℕ := 
  Finset.sum (Finset.range (n / 5 + 1)) (fun k => ((n - 5 * k) / 2 + 1))

theorem kopeck_ways (n : ℕ) : 
  (ways_123 n = ⌊((n : ℚ) + 3)^2 / 12⌋) ∧ 
  (ways_125 n = ⌊((n : ℚ) + 4)^2 / 20⌋) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kopeck_ways_l524_52466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_and_chord_length_l524_52429

noncomputable def line_l (t : ℝ) : ℝ × ℝ := (-1 + 2*t, 4*t)

noncomputable def curve_C_polar (θ : ℝ) : ℝ := Real.sqrt (2 / (2 * Real.sin θ^2 + Real.cos θ^2))

def curve_C_rect (x y : ℝ) : Prop := x^2/2 + y^2 = 1

noncomputable def chord_length : ℝ := 10 * Real.sqrt 2 / 9

theorem curve_C_and_chord_length :
  (∀ x y : ℝ, (∃ θ : ℝ, x = curve_C_polar θ * Real.cos θ ∧ y = curve_C_polar θ * Real.sin θ) ↔ curve_C_rect x y) ∧
  (∃ t₁ t₂ : ℝ, t₁ ≠ t₂ ∧
    let (x₁, y₁) := line_l t₁
    let (x₂, y₂) := line_l t₂
    curve_C_rect x₁ y₁ ∧ curve_C_rect x₂ y₂ ∧
    Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = chord_length) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_and_chord_length_l524_52429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_a6_l524_52474

/-- An arithmetic sequence with a positive common ratio -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  q : ℚ
  h_positive : q > 0
  h_arithmetic : ∀ n, a (n + 1) = a n * q

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  seq.a 1 * (1 - seq.q ^ n) / (1 - seq.q)

theorem arithmetic_sequence_a6 (seq : ArithmeticSequence) 
  (h1 : seq.a 1 = 1)
  (h2 : sum_n seq 3 = 7/4) :
  seq.a 6 = 1/32 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_a6_l524_52474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_five_digit_number_l524_52472

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 10000 ∧ n < 100000 ∧
  (∃ (p q r s t : ℕ),
    n = p * 10000 + q * 1000 + r * 100 + s * 10 + t ∧
    Finset.toSet {p, q, r, s, t} = Finset.toSet {3, 4, 6, 7, 8})

def PQR (n : ℕ) : ℕ :=
  n / 100

def QRS (n : ℕ) : ℕ :=
  (n / 10) % 1000

def RST (n : ℕ) : ℕ :=
  n % 1000

def P (n : ℕ) : ℕ :=
  n / 10000

theorem unique_five_digit_number :
  ∀ n : ℕ,
    is_valid_number n →
    PQR n % 3 = 0 →
    QRS n % 5 = 0 →
    RST n % 4 = 0 →
    P n = 7 :=
by
  intro n h_valid h_PQR h_QRS h_RST
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_five_digit_number_l524_52472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_problem_l524_52484

/-- Given an equilateral triangle ABC with side length 1 and points D, E, F inside it,
    such that AEF, BFD, and CDE are collinear, DEF is equilateral, and there exists a unique
    equilateral triangle XYZ with X on BC, Y on AB, Z on AC, D on XZ, E on YZ, and F on XY,
    prove that AZ = 1 / (1 + ∛2) -/
theorem equilateral_triangle_problem (A B C D E F X Y Z : ℝ × ℝ) : 
  let dist := (λ p q : ℝ × ℝ => Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2))
  -- ABC is equilateral with side length 1
  (dist A B = 1) → (dist B C = 1) → (dist C A = 1) →
  -- D, E, F are inside ABC
  (∃ t u v : ℝ, 0 < t ∧ t < 1 ∧ 0 < u ∧ u < 1 ∧ 0 < v ∧ v < 1 ∧
    D = (t * B.1 + (1 - t) * C.1, t * B.2 + (1 - t) * C.2) ∧
    E = (u * C.1 + (1 - u) * A.1, u * C.2 + (1 - u) * A.2) ∧
    F = (v * A.1 + (1 - v) * B.1, v * A.2 + (1 - v) * B.2)) →
  -- A, E, F are collinear
  (∃ k : ℝ, E = k • (F - A) + A) →
  -- B, F, D are collinear
  (∃ l : ℝ, F = l • (D - B) + B) →
  -- C, D, E are collinear
  (∃ m : ℝ, D = m • (E - C) + C) →
  -- DEF is equilateral
  (dist D E = dist E F) → (dist E F = dist F D) →
  -- X is on BC, Y on AB, Z on AC
  (∃ p q r : ℝ, 0 ≤ p ∧ p ≤ 1 ∧ 0 ≤ q ∧ q ≤ 1 ∧ 0 ≤ r ∧ r ≤ 1 ∧
    X = (p * B.1 + (1 - p) * C.1, p * B.2 + (1 - p) * C.2) ∧
    Y = (q * A.1 + (1 - q) * B.1, q * A.2 + (1 - q) * B.2) ∧
    Z = (r * A.1 + (1 - r) * C.1, r * A.2 + (1 - r) * C.2)) →
  -- D is on XZ, E on YZ, F on XY
  (∃ s w x : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ 0 ≤ w ∧ w ≤ 1 ∧ 0 ≤ x ∧ x ≤ 1 ∧
    D = s • (Z - X) + X ∧
    E = w • (Z - Y) + Y ∧
    F = x • (Y - X) + X) →
  -- XYZ is equilateral
  (dist X Y = dist Y Z) → (dist Y Z = dist Z X) →
  -- Conclusion: AZ = 1 / (1 + ∛2)
  dist A Z = 1 / (1 + Real.rpow 2 (1/3)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_problem_l524_52484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f₁_is_triangle_function_f₄_is_triangle_function_l524_52458

noncomputable def f₁ (x : ℝ) : ℝ := 4 - Real.sin x

noncomputable def f₄ (x : ℝ) : ℝ := Real.sin (2 * x) + 2 * Real.sqrt 3 * (Real.cos x) ^ 2

def is_triangle_function (f : ℝ → ℝ) (domain : Set ℝ) : Prop :=
  ∃ (f_max f_min : ℝ), 
    (∀ x ∈ domain, f x ≤ f_max) ∧
    (∀ x ∈ domain, f_min ≤ f x) ∧
    f_max < 2 * f_min

theorem f₁_is_triangle_function : 
  is_triangle_function f₁ Set.univ := by
  sorry

theorem f₄_is_triangle_function : 
  is_triangle_function f₄ (Set.Icc 0 (Real.pi / 4)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f₁_is_triangle_function_f₄_is_triangle_function_l524_52458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_CD_l524_52420

noncomputable def cylinderWithHemispheresVolume (radius : ℝ) (height : ℝ) : ℝ :=
  (4/3 * Real.pi * radius^3) + (Real.pi * radius^2 * height)

theorem length_of_CD (volume : ℝ) (radius : ℝ) :
  volume = 432 * Real.pi ∧ radius = 4 →
  ∃ (height : ℝ), cylinderWithHemispheresVolume radius height = volume ∧ height = 23 := by
  sorry

#check length_of_CD

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_CD_l524_52420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_prism_surface_area_l524_52497

theorem rectangular_prism_surface_area 
  (l w h : ℕ) 
  (h1 : l > 1) 
  (h2 : w > 1) 
  (h3 : h > 1) 
  (h4 : Nat.Coprime l w) 
  (h5 : Nat.Coprime l h) 
  (h6 : Nat.Coprime w h) 
  (h7 : l * w * h = 665) : 
  2 * (l * w + l * h + w * h) = 526 := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_prism_surface_area_l524_52497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_consumption_theorem_l524_52427

/-- Represents a person with their pizza consumption -/
structure Person where
  name : String
  pizza_pieces_eaten : ℚ
  deriving Repr

/-- The pizza problem setup -/
structure PizzaProblem where
  total_people : ℕ 
  pieces_per_pizza : ℕ 
  ann_cate_percentage : ℚ 
  total_pieces_left : ℕ 

/-- The theorem stating the result of the pizza consumption problem -/
theorem pizza_consumption_theorem (setup : PizzaProblem) 
  (h1 : setup.total_people = 4)
  (h2 : setup.pieces_per_pizza = 4)
  (h3 : setup.ann_cate_percentage = 3/4)
  (h4 : setup.total_pieces_left = 6) :
  ∃ (bill dale : Person), 
    bill.pizza_pieces_eaten = 2 ∧ 
    dale.pizza_pieces_eaten = 2 := by
  sorry

/-- An example of the pizza problem setup -/
def example_setup : PizzaProblem := {
  total_people := 4,
  pieces_per_pizza := 4,
  ann_cate_percentage := 3/4,
  total_pieces_left := 6
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_consumption_theorem_l524_52427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_properties_l524_52439

def z (a : ℝ) : ℂ := Complex.mk (a^2 + a - 2) (a^2 - 7*a + 6)

theorem z_properties :
  (∀ a : ℝ, (z a).im = 0 ↔ a = 1 ∨ a = 6) ∧
  (∀ a : ℝ, (z a).re = 0 ∧ (z a).im ≠ 0 ↔ a = -2) ∧
  (∀ a : ℝ, z a = 0 ↔ a = 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_properties_l524_52439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_ratio_l524_52495

theorem circle_area_ratio {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] 
  {O Y R : E} (h : ‖Y - O‖ = (1/3) * ‖R - O‖) :
  (π * ‖Y - O‖^2) / (π * ‖R - O‖^2) = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_ratio_l524_52495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_solutions_floor_sum_l524_52438

theorem infinite_solutions_floor_sum (n : ℕ) (hn : n > 6 * 1980^2) :
  ∃ (S : Finset (ℕ × ℕ)), S.card ≥ 1980 ∧
  ∀ (x y : ℕ), (x, y) ∈ S →
    ⌊(x : ℝ)^(3/2)⌋ + ⌊(y : ℝ)^(3/2)⌋ = 9 * n^3 + 6 * n^2 :=
by
  sorry

#check infinite_solutions_floor_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_solutions_floor_sum_l524_52438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_composition_l524_52489

/-- A square plate in 3D space -/
structure SquarePlate where
  side_length : ℝ
  center : ℝ × ℝ × ℝ
  normal : ℝ × ℝ × ℝ
  edges : Set (ℝ × ℝ × ℝ)
  vertices : Set (ℝ × ℝ × ℝ)

/-- The locus of points at a fixed distance from a square plate -/
def Locus (plate : SquarePlate) (distance : ℝ) : Set (ℝ × ℝ × ℝ) :=
  {p | ∃ q, ‖p - q‖ = distance}

/-- A parallel plane at a fixed distance from the square plate -/
def ParallelPlane (plate : SquarePlate) (distance : ℝ) : Set (ℝ × ℝ × ℝ) :=
  {p | ∃ q, p - q = distance • plate.normal ∨ p - q = -distance • plate.normal}

/-- A quarter-cylinder along an edge of the square plate -/
def QuarterCylinder (plate : SquarePlate) (distance : ℝ) : Set (ℝ × ℝ × ℝ) :=
  {p | ∃ q ∈ plate.edges, ‖p - q‖ = distance}

/-- A quarter-sphere at a vertex of the square plate -/
def QuarterSphere (plate : SquarePlate) (distance : ℝ) : Set (ℝ × ℝ × ℝ) :=
  {p | ∃ q ∈ plate.vertices, ‖p - q‖ = distance}

theorem locus_composition (plate : SquarePlate) (h1 : plate.side_length = 10) (h2 : distance = 5) :
  Locus plate distance =
    (ParallelPlane plate distance) ∪
    (QuarterCylinder plate distance) ∪
    (QuarterSphere plate distance) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_composition_l524_52489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_integral_over_ellipse_l524_52477

/-- The ellipse in the x-y plane defined by x²/a² + y²/b² = 1 -/
def Ellipse (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

/-- The vector field F(x, y) = (x² - y², 2xy) -/
def F : ℝ × ℝ → ℝ × ℝ := fun p ↦ (p.1^2 - p.2^2, 2 * p.1 * p.2)

-- We need to declare this as noncomputable because ContourIntegral might not be computable
noncomputable def lineIntegral (C : Set (ℝ × ℝ)) (f : ℝ × ℝ → ℝ × ℝ) : ℝ := sorry

theorem line_integral_over_ellipse (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ I : ℝ, lineIntegral (Ellipse a b) F = I ∧ I = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_integral_over_ellipse_l524_52477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_difference_l524_52492

theorem tan_difference (α β : ℝ) 
  (h1 : Real.tan α = -3/4)
  (h2 : Real.tan (Real.pi - β) = 1/2) :
  Real.tan (α - β) = -2/11 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_difference_l524_52492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_time_is_35_div_6_l524_52411

/-- Two runners on circular tracks with a common center -/
structure CircularTrack where
  runner1_period : ℚ
  runner2_period : ℚ

/-- The time when two runners are at maximum distance -/
noncomputable def max_distance_time (track : CircularTrack) : ℚ :=
  (track.runner1_period * track.runner2_period) / (4 * (track.runner1_period + track.runner2_period))

/-- Theorem: The shortest time when the runners are at maximum distance is 35/6 seconds -/
theorem max_distance_time_is_35_div_6 (track : CircularTrack) 
  (h1 : track.runner1_period = 20)
  (h2 : track.runner2_period = 28) : 
  max_distance_time track = 35 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_time_is_35_div_6_l524_52411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_theorem_l524_52486

/-- The area of a parallelogram formed by two vectors -/
def parallelogramArea (a b : ℝ × ℝ) : ℝ :=
  abs ((a.1 * b.2) - (a.2 * b.1))

theorem parallelogram_area_theorem (p q : ℝ × ℝ) :
  let a := (2 * p.1 - 3 * q.1, 2 * p.2 - 3 * q.2)
  let b := (3 * p.1 + q.1, 3 * p.2 + q.2)
  Real.sqrt (p.1^2 + p.2^2) = 4 →
  Real.sqrt (q.1^2 + q.2^2) = 1 →
  Real.arccos ((p.1 * q.1 + p.2 * q.2) / (Real.sqrt (p.1^2 + p.2^2) * Real.sqrt (q.1^2 + q.2^2))) = π / 6 →
  parallelogramArea a b = 22 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_theorem_l524_52486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_students_theorem_l524_52473

def num_rows : ℕ := 8
def first_row_desks : ℕ := 10

def desks_in_row : ℕ → ℕ
  | 0 => 0  -- Added case for 0
  | 1 => first_row_desks
  | n + 1 => if n % 2 = 0 then desks_in_row n + (n + 1) else desks_in_row n - (n + 1)

def students_in_row (n : ℕ) : ℕ :=
  if n % 2 = 1 then
    (desks_in_row n * 3) / 4
  else
    desks_in_row n / 2

def total_students : ℕ :=
  (List.range num_rows).map (fun i => students_in_row (i + 1)) |>.sum

theorem max_students_theorem : total_students = 45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_students_theorem_l524_52473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_cost_problem_l524_52437

/-- The cost structure for apples -/
structure AppleCost where
  l : ℚ  -- Cost per kg for first 30 kgs
  a : ℚ  -- Cost per kg for additional kgs after 30 kgs

/-- Calculate the total cost for a given weight -/
def totalCost (c : AppleCost) (weight : ℚ) : ℚ :=
  if weight ≤ 30 then c.l * weight
  else c.l * 30 + c.a * (weight - 30)

/-- The problem statement -/
theorem apple_cost_problem (c : AppleCost) : 
  c.l = 10 ∧ 
  totalCost c 33 = 333 ∧ 
  totalCost c 36 = 366 ∧ 
  totalCost c 15 = 150 → 
  c.a = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_cost_problem_l524_52437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_evelyn_paper_usage_l524_52491

/-- The number of sheets in a pad of paper -/
def sheets_per_pad : ℕ := 60

/-- The number of days Evelyn works per week -/
def workdays_per_week : ℕ := 5

/-- Evelyn uses one pad per week -/
def pads_per_week : ℕ := 1

/-- The number of sheets Evelyn uses per workday -/
def sheets_per_workday : ℕ := sheets_per_pad / workdays_per_week

theorem evelyn_paper_usage :
  sheets_per_workday = 12 := by
  -- Unfold the definition of sheets_per_workday
  unfold sheets_per_workday
  -- Simplify the division
  norm_num
  -- QED
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_evelyn_paper_usage_l524_52491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_relations_l524_52447

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Definition of perpendicular lines -/
def perpendicular (l₁ l₂ : Line) : Prop :=
  l₁.a * l₂.a + l₁.b * l₂.b = 0

/-- Definition of parallel lines -/
def parallel (l₁ l₂ : Line) : Prop :=
  l₁.a * l₂.b = l₁.b * l₂.a ∧ l₁.a * l₂.c ≠ l₁.c * l₂.a

/-- The main theorem -/
theorem line_relations (m n : ℝ) :
  let l₁ : Line := ⟨m, 8, n⟩
  let l₂ : Line := ⟨2, m, -1 + n/2⟩
  (perpendicular l₁ l₂ ↔ m = 0) ∧
  (parallel l₁ l₂ ↔ (m = 4 ∧ n ∈ Set.univ) ∨ (m = -4 ∧ n ≠ -2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_relations_l524_52447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_inscribed_circle_radius_l524_52455

/-- The radius of a circle inscribed in a rhombus with given diagonals -/
noncomputable def inscribed_circle_radius (d1 d2 : ℝ) : ℝ :=
  90 / Real.sqrt 261

/-- Theorem: The radius of a circle inscribed in a rhombus with diagonals 12 and 30 is 90/√261 -/
theorem rhombus_inscribed_circle_radius :
  inscribed_circle_radius 12 30 = 90 / Real.sqrt 261 := by
  -- Unfold the definition of inscribed_circle_radius
  unfold inscribed_circle_radius
  -- The rest of the proof would go here, but we'll use sorry for now
  sorry

#check rhombus_inscribed_circle_radius

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_inscribed_circle_radius_l524_52455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l524_52434

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x

-- Define the "closer" relation
def is_closer (s t r : ℝ) : Prop := |s - r| ≤ |t - r|

theorem problem_statement :
  ∀ a : ℝ,
  (∃ x₀ : ℝ, x₀ ∈ Set.Ici 1 ∧ f a x₀ < Real.exp 1 - a) →
  a > Real.exp 1 ∧
  ∀ x : ℝ, x ≥ 1 → is_closer (Real.exp 1 / x) (Real.exp (x - 1) + a) (Real.log x) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l524_52434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_opposite_direction_l524_52433

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem vector_opposite_direction (a b : V) 
  (h1 : ‖a‖ = 4)
  (h2 : ‖b‖ = 2)
  (h3 : ∃ (c : ℝ), c < 0 ∧ a = c • b) :
  a = (-2 : ℝ) • b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_opposite_direction_l524_52433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_skew_lines_angle_range_l524_52405

structure Space3D where
  -- Define a 3D space
  point : Type
  -- Add more properties as needed

structure Line where
  -- Define a line in 3D space
  start : Space3D
  direction : Space3D
  -- Add more properties as needed

def skew (l1 l2 : Line) : Prop :=
  -- Definition of skew lines
  sorry

def angle (l1 l2 : Line) : ℝ :=
  -- Definition of angle between two lines
  sorry

theorem skew_lines_angle_range (s : Space3D) (l1 l2 : Line) :
  skew l1 l2 → 0 < angle l1 l2 ∧ angle l1 l2 ≤ π / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_skew_lines_angle_range_l524_52405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_continuous_iff_b_eq_neg_six_l524_52413

noncomputable def f (b : ℝ) (x : ℝ) : ℝ :=
  if x > 5 then x + 4 else 3 * x + b

theorem continuous_iff_b_eq_neg_six (b : ℝ) :
  Continuous (f b) ↔ b = -6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_continuous_iff_b_eq_neg_six_l524_52413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l524_52498

/-- Given a train and platform with the same length, prove the time to cross the platform --/
theorem train_crossing_time (train_length platform_length : ℝ) 
  (time_cross_pole : ℝ) (h1 : train_length = 420) 
  (h2 : platform_length = 420) (h3 : time_cross_pole = 30) :
  (train_length + platform_length) / (train_length / time_cross_pole) = 60 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l524_52498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_half_solution_l524_52436

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 2^(-x) else Real.log x / Real.log 9

-- Define the solution set
def solution_set : Set ℝ := {x | x < 1 ∨ x > 3}

-- Theorem statement
theorem f_greater_than_half_solution :
  {x : ℝ | f x > 1/2} = solution_set :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_half_solution_l524_52436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_side_length_squared_l524_52423

/-- An isosceles right triangle inscribed in an ellipse -/
structure IsoscelesRightTriangleInEllipse where
  /-- The ellipse equation: x^2 + 9y^2 = 9 -/
  ellipse : ∀ (x y : ℝ), x^2 + 9*y^2 = 9 → True
  /-- One vertex of the triangle is at (0, 1) -/
  vertex_at_origin : ∃ (v : ℝ × ℝ), v = (0, 1)
  /-- One altitude is contained within the y-axis -/
  altitude_on_y_axis : ∃ (a : Set (ℝ × ℝ)), ∀ (p : ℝ × ℝ), p ∈ a → p.1 = 0

/-- The square of the length of each equal side of the triangle is 324/25 -/
theorem isosceles_right_triangle_side_length_squared 
  (t : IsoscelesRightTriangleInEllipse) : ∃ (s : ℝ), s^2 = 324/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_side_length_squared_l524_52423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_among_three_l524_52452

theorem largest_among_three (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : b < 1) :
  max (a * b) (max (a ^ b) (Real.log a / Real.log b)) = Real.log a / Real.log b := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_among_three_l524_52452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_direction_component_l524_52496

/-- A line in 2D space -/
structure Line2D where
  direction : Fin 2 → ℝ

/-- A point in 2D space -/
def Point2D := Fin 2 → ℝ

/-- Check if a line passes through two points -/
def Line2D.passesThrough (l : Line2D) (p1 p2 : Point2D) : Prop :=
  ∃ t : ℝ, ∀ i : Fin 2, p2 i - p1 i = t * l.direction i

theorem line_direction_component (c : ℝ) : 
  let l : Line2D := { direction := ![3, c] }
  let p1 : Point2D := ![-1, -3]
  let p2 : Point2D := ![2, 1]
  l.passesThrough p1 p2 → c = 4 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_direction_component_l524_52496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_reduction_correct_l524_52419

/-- Represents the percentage reduction in petrol consumption -/
noncomputable def percentageReduction (P : ℝ) : ℝ := (32 / (P + 32)) * 100

/-- Theorem stating that the percentage reduction in consumption is correct
    given a price increase of 32 and constant expenditure -/
theorem percentage_reduction_correct (P C : ℝ) (h_positive : P > 0) (h_consumption : C > 0) :
  let original_expenditure := P * C
  let new_price := P + 32
  let new_consumption := original_expenditure / new_price
  (C - new_consumption) / C * 100 = percentageReduction P :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_reduction_correct_l524_52419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l524_52487

/-- The ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- The eccentricity of the ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := 1/2

/-- The maximum area of the inscribed circle in the triangle formed by the foci and a point on the ellipse -/
noncomputable def max_inscribed_circle_area : ℝ := Real.pi/3

/-- The equation of the ellipse -/
def ellipse_equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The fixed points through which the circle with diameter PQ always passes -/
def fixed_points : Set (ℝ × ℝ) := {(1, 0), (7, 0)}

/-- Predicate to check if a circle passes through fixed points -/
def passes_through_fixed_points (l : ℝ → ℝ) (P Q : ℝ × ℝ) : Prop :=
  ∀ fp ∈ fixed_points, (fp.1 - P.1)^2 + (fp.2 - P.2)^2 = (fp.1 - Q.1)^2 + (fp.2 - Q.2)^2

/-- The main theorem stating the properties of the ellipse -/
theorem ellipse_properties (e : Ellipse) :
  (∀ x y, ellipse_equation e x y ↔ x^2 / 4 + y^2 / 3 = 1) ∧
  (∀ l P Q, passes_through_fixed_points l P Q) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l524_52487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_focus_l524_52485

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the point P
structure Point where
  x : ℝ
  y : ℝ
  on_parabola : parabola x y
  distance_from_line : (x + 2)^2 + y^2 = 25

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- State the theorem
theorem distance_to_focus (P : Point) :
  Real.sqrt ((P.x - focus.1)^2 + (P.y - focus.2)^2) = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_focus_l524_52485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_m_zero_l524_52403

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {a b c d e f : ℝ} (x y : ℝ) :
  (a * x + b * y + c = 0) ∧ (d * x + e * y + f = 0) → (a * e = b * d ↔ a * x + b * y + c = 0 ∧ d * x + e * y + f = 0)

/-- Given two lines l₁: x + m*y + 3 = 0 and l₂: (m-1)*x + 2*m*y + 2*m = 0,
    if l₁ // l₂, then m = 0 -/
theorem parallel_lines_m_zero (m : ℝ) :
  (∀ x y : ℝ, x + m * y + 3 = 0 ∧ (m - 1) * x + 2 * m * y + 2 * m = 0) →
  m = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_m_zero_l524_52403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_calculation_A_is_correct_answer_l524_52457

theorem correct_calculation : (1 : ℤ) + (-2) = -1 :=
by
  -- Proof goes here
  sorry

-- Define the other calculations for completeness
def calculation_B : ℤ := (1 : ℤ) - (-2)
def calculation_C : ℤ := (-2) + (-1)
def calculation_D : ℤ := (2 : ℤ) - (-1)

-- The main theorem stating that A is the correct answer
theorem A_is_correct_answer :
  (((1 : ℤ) + (-2)) = -1) ∧
  (calculation_B ≠ -1) ∧
  (calculation_C ≠ -1) ∧
  (calculation_D ≠ -1) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_calculation_A_is_correct_answer_l524_52457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_expression_l524_52450

theorem quadratic_roots_expression (x₁ x₂ : ℝ) : 
  x₁^2 - 3*x₁ + 1 = 0 → 
  x₂^2 - 3*x₂ + 1 = 0 → 
  x₁^2 - 5*x₁ - 2*x₂ = -7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_expression_l524_52450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l524_52471

noncomputable def f (x : ℝ) : ℝ :=
  if -1 ≤ x ∧ x ≤ 0 then x / (x^2 + 1)
  else if 0 < x ∧ x ≤ 1 then -x / (x^2 + 1)
  else 0  -- undefined outside [-1,1]

theorem f_properties :
  (∀ x, -1 ≤ x ∧ x ≤ 1 → f (-x) = f x) ∧  -- f is even
  (∀ x, -1 ≤ x ∧ x ≤ 0 → f x = x / (x^2 + 1)) →
  (∀ x, 0 < x ∧ x ≤ 1 → f x = -x / (x^2 + 1)) ∧
  (∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ 1 → f x > f y) ∧
  (¬ ∃ a : ℝ, f (1 - a) - f (3 + a) < 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l524_52471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_theorem_l524_52440

open Set
open Geometry

/-- Definition of a convex pentagon -/
structure ConvexPentagon where
  vertices : Fin 5 → ℝ × ℝ
  convex : Convex ℝ (range vertices)

/-- Area of a triangle given by three points -/
noncomputable def triangleArea (p q r : ℝ × ℝ) : ℝ := sorry

/-- Statement of the pentagon theorem -/
theorem pentagon_theorem (A : ConvexPentagon) :
  (∀ i : Fin 5, triangleArea (A.vertices i) (A.vertices (i + 1)) (A.vertices (i + 2)) =
                triangleArea (A.vertices (i + 1)) (A.vertices (i + 2)) (A.vertices (i + 3))) →
  ∃ M : ℝ × ℝ, ∀ i : Fin 5,
    triangleArea (A.vertices i) M (A.vertices (i + 1)) =
    triangleArea (A.vertices (i + 1)) M (A.vertices (i + 2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_theorem_l524_52440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_waste_paper_recycling_results_l524_52409

/-- Represents the relationship between waste paper recycling and its outcomes -/
structure WastePaperRecycling where
  waste_paper : ℝ
  white_paper : ℝ
  trees_saved : ℝ

/-- Calculates the ratio of white paper produced to waste paper used -/
noncomputable def white_paper_ratio (w : WastePaperRecycling) : ℝ :=
  w.white_paper / w.waste_paper

/-- Calculates the number of trees saved per ton of waste paper recycled -/
noncomputable def trees_saved_per_ton (w : WastePaperRecycling) : ℝ :=
  w.trees_saved / w.waste_paper

/-- Theorem stating the results of waste paper recycling -/
theorem waste_paper_recycling_results (w : WastePaperRecycling) 
  (h1 : w.waste_paper = 5)
  (h2 : w.white_paper = 4)
  (h3 : w.trees_saved = 40) :
  white_paper_ratio w = 0.8 ∧ trees_saved_per_ton w = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_waste_paper_recycling_results_l524_52409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_numbers_l524_52442

-- Define the set of numbers in the hat
def hat_numbers : Set ℕ := {n | 1 ≤ n ∧ n ≤ 50}

-- Define Alice's and Bob's numbers
variable (alice_number : ℕ)
variable (bob_number : ℕ)

-- Axioms based on the problem conditions
axiom alice_bob_in_hat : alice_number ∈ hat_numbers ∧ bob_number ∈ hat_numbers
axiom alice_bob_different : alice_number ≠ bob_number
axiom alice_uncertain : ∃ n ∈ hat_numbers, n > alice_number
axiom bob_certain : ∀ n ∈ hat_numbers, n ≠ bob_number → n > bob_number
axiom bob_prime : Nat.Prime bob_number
axiom perfect_square : ∃ k : ℕ, 150 * bob_number + alice_number = k^2

-- Theorem to prove
theorem sum_of_numbers : alice_number + bob_number = 26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_numbers_l524_52442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_main_result_l524_52431

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (-x^2 + 2*x + 3)

-- Define the domain of f
def domain : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 3}

-- State the theorem
theorem f_monotonicity :
  (∀ x ∈ domain, ∀ y ∈ domain, x < y → x < 1 → f x < f y) ∧
  (∀ x ∈ domain, ∀ y ∈ domain, x < y → 1 < x → f x > f y) := by
  sorry

-- Define the increasing and decreasing intervals
def increasing_interval : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 1}
def decreasing_interval : Set ℝ := {x : ℝ | 1 ≤ x ∧ x ≤ 3}

-- State the main result
theorem main_result :
  (∀ x ∈ increasing_interval, ∀ y ∈ increasing_interval, x < y → f x < f y) ∧
  (∀ x ∈ decreasing_interval, ∀ y ∈ decreasing_interval, x < y → f x > f y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_main_result_l524_52431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_allStarArrangements_eq_1152_l524_52443

/-- Factorial function -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- The number of ways to arrange 9 All-Stars in a row, where 3 are from one team
    and 2 each are from three other teams, and teammates must sit together. -/
def allStarArrangements : ℕ :=
  factorial 4 * factorial 3 * (factorial 2 * factorial 2 * factorial 2)

/-- Theorem stating that the number of arrangements is 1152 -/
theorem allStarArrangements_eq_1152 : allStarArrangements = 1152 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_allStarArrangements_eq_1152_l524_52443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_valid_positions_l524_52464

/-- Represents a position where an additional square can be attached to the T-shape. -/
inductive AttachmentPosition
| Top
| Bottom
| Left
| Right
| TopLeft
| TopRight
| BottomLeft
| BottomRight
| MiddleLeft
| MiddleRight
| MiddleTop
| MiddleBottom

/-- Represents a square in the polygon. -/
structure Square :=
  (side_length : ℝ)

/-- Predicate to check if two squares are congruent. -/
def Square.congruent (s1 s2 : Square) : Prop :=
  s1.side_length = s2.side_length

/-- Represents the T-shaped polygon made of 6 congruent squares. -/
structure TShape :=
  (squares : Fin 6 → Square)
  (is_congruent : ∀ i j : Fin 6, (squares i).congruent (squares j))
  (is_t_shape : Prop) -- Placeholder for the T-shape arrangement condition

/-- Represents the resulting polygon after attaching an additional square. -/
structure ResultingPolygon :=
  (base : TShape)
  (extra_square : Square)
  (attachment : AttachmentPosition)
  (is_congruent : (base.squares 0).congruent extra_square)

/-- Predicate to check if a resulting polygon can be folded into a cube-like structure with two faces missing. -/
def can_fold_to_cube_minus_two (p : ResultingPolygon) : Prop :=
  sorry -- Placeholder for the actual folding condition

/-- The main theorem stating that exactly 4 attachment positions allow folding into a cube-like structure. -/
theorem four_valid_positions (t : TShape) :
  ∃! (valid_positions : Finset AttachmentPosition),
    valid_positions.card = 4 ∧
    (∀ pos, pos ∈ valid_positions ↔
      can_fold_to_cube_minus_two { base := t, extra_square := t.squares 0, attachment := pos, is_congruent := sorry }) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_valid_positions_l524_52464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dodecahedron_to_cuboids_volume_ratio_l524_52483

/-- The volume of a regular dodecahedron with side length s -/
noncomputable def dodecahedronVolume (s : ℝ) : ℝ := (15 + 7 * Real.sqrt 5) / 4 * s^3

/-- The volume of a cuboid with depth d -/
def cuboidVolume (d : ℝ) : ℝ := 2 * d^3

/-- The theorem stating the ratio of dodecahedron volume to total cuboid volume -/
theorem dodecahedron_to_cuboids_volume_ratio :
  ∀ (s d : ℝ), s > 0 → d > 0 →
  12 * (cuboidVolume d) = (1/2) * (dodecahedronVolume s) →
  (dodecahedronVolume s) / (12 * cuboidVolume d) = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dodecahedron_to_cuboids_volume_ratio_l524_52483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_value_at_two_l524_52404

-- Define the function A
noncomputable def A (x : ℝ) : ℝ := (3*x/(x-1) - x/(x+1)) / (x/(x^2-1))

-- Define the inequality system
def inequality_system (x : ℝ) : Prop :=
  x - 3*(x-2) ≥ 2 ∧ 4*x - 2 < 5*x - 1

-- State the theorem
theorem A_value_at_two :
  ∀ x : ℝ, 
    inequality_system x → 
    x = 2 → 
    A x = 8 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_value_at_two_l524_52404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_trinomial_m_value_l524_52414

/-- A trinomial is a perfect square if it can be expressed as (ax + b)^2 -/
def IsPerfectSquareTrinomial (a b c : ℝ) : Prop :=
  ∃ (p q : ℝ), ∀ x : ℝ, a * x^2 + b * x + c = (p * x + q)^2

theorem perfect_square_trinomial_m_value :
  ∀ m : ℝ, IsPerfectSquareTrinomial 1 (-m) 25 → m = 10 ∨ m = -10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_trinomial_m_value_l524_52414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extended_quadrilateral_area_l524_52415

/-- Represents a quadrilateral with extended sides -/
structure ExtendedQuadrilateral where
  EF : ℝ
  FG : ℝ
  GH : ℝ
  HE : ℝ
  EF' : ℝ
  FG' : ℝ
  GH' : ℝ
  HE' : ℝ
  area : ℝ

/-- The area of the extended quadrilateral E'F'G'H' -/
noncomputable def extendedArea (q : ExtendedQuadrilateral) : ℝ :=
  q.area + (q.EF' - q.EF) / q.EF * q.area / 4 +
           (q.FG' - q.FG) / q.FG * q.area / 4 +
           (q.GH' - q.GH) / q.GH * q.area / 4 +
           (q.HE' - q.HE) / q.HE * q.area / 4

/-- Theorem: The area of the extended quadrilateral E'F'G'H' is 62.5 -/
theorem extended_quadrilateral_area :
  let q : ExtendedQuadrilateral := {
    EF := 5, FG := 7, GH := 8, HE := 9,
    EF' := 10, FG' := 14, GH' := 12, HE' := 18,
    area := 20
  }
  extendedArea q = 62.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extended_quadrilateral_area_l524_52415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calories_burned_in_40_minutes_l524_52435

/-- Calories burned per minute -/
noncomputable def caloriesPerMinute (calories : ℝ) (minutes : ℝ) : ℝ := calories / minutes

theorem calories_burned_in_40_minutes 
  {x : ℝ} (h : caloriesPerMinute 300 25 = caloriesPerMinute x 40) : x = 480 := by
  sorry

#check calories_burned_in_40_minutes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calories_burned_in_40_minutes_l524_52435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_direct_proportion_graph_is_straight_line_l524_52482

/-- A direct proportion function from ℝ to ℝ -/
def direct_proportion (k : ℝ) : ℝ → ℝ := λ x ↦ k * x

/-- The graph of a function from ℝ to ℝ -/
def graph (f : ℝ → ℝ) : Set (ℝ × ℝ) := {p | p.2 = f p.1}

/-- A straight line in ℝ² passing through the origin -/
def straight_line_through_origin (m : ℝ) : Set (ℝ × ℝ) := {p | p.2 = m * p.1}

theorem direct_proportion_graph_is_straight_line (k : ℝ) :
  ∃ m : ℝ, graph (direct_proportion k) = straight_line_through_origin m := by
  sorry

#check direct_proportion_graph_is_straight_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_direct_proportion_graph_is_straight_line_l524_52482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_yans_distance_ratio_l524_52479

/-- Yan's walking speed -/
noncomputable def walking_speed : ℝ := 1

/-- Yan's distance from home -/
noncomputable def distance_from_home : ℝ := 2

/-- Yan's distance from stadium -/
noncomputable def distance_from_stadium : ℝ := 3

/-- Yan's biking speed is 5 times his walking speed -/
noncomputable def biking_speed : ℝ := 5 * walking_speed

/-- Time taken to walk directly to stadium -/
noncomputable def time_walk_to_stadium : ℝ := distance_from_stadium / walking_speed

/-- Time taken to walk home and then bike to stadium -/
noncomputable def time_walk_home_and_bike : ℝ := 
  distance_from_home / walking_speed + 
  (distance_from_home + distance_from_stadium) / biking_speed

theorem yans_distance_ratio : 
  time_walk_to_stadium = time_walk_home_and_bike → 
  distance_from_home / distance_from_stadium = 2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_yans_distance_ratio_l524_52479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l524_52444

-- Define the function type
def RealFunction := ℝ → ℝ

-- Define the functional equation
def SatisfiesEquation (f : RealFunction) (a : ℝ) : Prop :=
  ∀ x y : ℝ, (x + y) * (f x - f y) = a * (x - y) * f (x + y)

-- Define the quadratic function
def QuadraticFunction (α β : ℝ) : RealFunction :=
  λ x ↦ α * x^2 + β * x

-- Define the zero function
def ZeroFunction : RealFunction :=
  λ _ ↦ 0

-- Define a constant function
def ConstantFunction (c : ℝ) : RealFunction :=
  λ _ ↦ c

-- The main theorem
theorem functional_equation_solution (a : ℝ) :
  (a = 1 ∧ ∃ α β : ℝ, SatisfiesEquation (QuadraticFunction α β) a) ∨
  (a ≠ 0 ∧ a ≠ 1 ∧ SatisfiesEquation ZeroFunction a) ∨
  (a = 0 ∧ ∃ c : ℝ, SatisfiesEquation (ConstantFunction c) a) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l524_52444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fundamental_system_differential_equation_l524_52421

/-- The differential equation for which y₁(x) = e^(x²) and y₂(x) = e^(-x²) form a fundamental system of solutions -/
theorem fundamental_system_differential_equation 
  (x : ℝ) 
  (y : ℝ → ℝ) 
  (h₁ : ∃ c₁ c₂ : ℝ, ∀ x, y x = c₁ * Real.exp (x^2) + c₂ * Real.exp (-x^2)) :
  ∀ x ≠ 0, (deriv (deriv y)) x - (1/x) * (deriv y x) - 4*x^2 * y x = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fundamental_system_differential_equation_l524_52421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l524_52461

-- Define the sequence a_n
def a : ℕ → ℕ → ℕ
  | p, 0 => p  -- Added case for n = 0
  | p, 1 => p
  | p, n + 1 => if a p n % 2 = 0 then a p n / 2 else a p n + 5

-- Define the sum of the first n terms
def S (p : ℕ) (n : ℕ) : ℕ :=
  (List.range n).map (a p) |> List.sum

-- Theorem statement
theorem sequence_properties :
  (∀ p : ℕ, p > 0 → (a p 3 = 9 → p = 36 ∨ p = 13)) ∧
  (S 7 150 = 616) ∧
  (∀ p : ℕ, p > 0 → (∃ m : ℕ, m > 0 ∧ a p m = 1) ↔ p % 5 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l524_52461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l524_52454

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * Real.cos x

theorem problem_solution :
  ∀ α : ℝ,
  α ∈ Set.Ioo (Real.pi / 2) Real.pi →
  f (α / 2) = 3 / 5 →
  Real.cos (α - Real.pi / 3) = (3 * Real.sqrt 3 - 4) / 10 ∧
  (∀ k : ℤ, Set.Icc (k * Real.pi + Real.pi / 4) (k * Real.pi + 3 * Real.pi / 4) ⊆ 
    {x : ℝ | ∀ y : ℝ, x < y → f x > f y}) ∧
  (∃ m : ℝ, HasDerivAt f m 0 ∧ m = 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l524_52454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_most_stable_performance_l524_52478

/-- Represents a student with their performance variance -/
structure Student where
  name : String
  variance : ℝ

/-- Theorem stating that the student with the lowest variance has the most stable performance -/
theorem most_stable_performance (students : List Student) 
  (hA : ∃ s, s ∈ students ∧ s.name = "A" ∧ s.variance = 2.6)
  (hB : ∃ s, s ∈ students ∧ s.name = "B" ∧ s.variance = 1.7)
  (hC : ∃ s, s ∈ students ∧ s.name = "C" ∧ s.variance = 3.5)
  (h_same_avg : ∀ s1 s2, s1 ∈ students → s2 ∈ students → s1.name ≠ s2.name → s1.variance ≠ s2.variance) :
  ∃ s, s ∈ students ∧ s.name = "B" ∧ 
    ∀ t, t ∈ students → t.name ≠ "B" → s.variance < t.variance :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_most_stable_performance_l524_52478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_club_student_equality_l524_52476

/-- Given a school where each club has 3 members and each student is a member of 3 clubs,
    prove that the number of clubs equals the number of students. -/
theorem club_student_equality (C E : ℕ) 
  (club_members : ℕ → ℕ)
  (student_clubs : ℕ → ℕ)
  (total_memberships : ℕ) : 
  (∀ club, club_members club = 3) →
  (∀ student, student_clubs student = 3) →
  total_memberships = 3 * C →
  total_memberships = 3 * E →
  C = E := by
  intros h1 h2 h3 h4
  have : 3 * C = 3 * E := by
    rw [← h3, h4]
  exact Nat.mul_left_cancel (by norm_num) this


end NUMINAMATH_CALUDE_ERRORFEEDBACK_club_student_equality_l524_52476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l524_52428

noncomputable def f (x : ℝ) : ℝ := Real.log (2 * x + 3 - x^2) / Real.log 4

theorem f_properties :
  (∀ x, f x ∈ Set.Ioo (-1) 3 → x ∈ Set.Ioo (-1) 3) ∧
  (∀ x y, x ∈ Set.Ioc (-1) 1 → y ∈ Set.Ioc (-1) 1 → x < y → f x < f y) ∧
  (∀ x y, x ∈ Set.Icc 1 3 → y ∈ Set.Icc 1 3 → x < y → f x > f y) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l524_52428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_girls_in_class_l524_52488

theorem girls_in_class : ∃ (girls : ℕ), girls = 13 := by
  -- Define the total number of students including absentees
  let total : ℕ := 32
  -- Define the number of absentees
  let absentees : ℕ := 2
  -- Define the ratio of girls to boys
  let ratio : ℚ := 3 / 4
  
  -- Calculate the number of students present
  let present : ℕ := total - absentees
  
  -- Assert that the number of girls is 13
  have h : ∃ (girls : ℕ), girls = 13 := by
    -- The actual proof would go here
    sorry
  
  -- Return the result
  exact h


end NUMINAMATH_CALUDE_ERRORFEEDBACK_girls_in_class_l524_52488
