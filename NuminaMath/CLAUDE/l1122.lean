import Mathlib

namespace NUMINAMATH_CALUDE_negation_equivalence_l1122_112229

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) ↔ (∀ x : ℝ, x^2 + 2*x + 2 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1122_112229


namespace NUMINAMATH_CALUDE_cubic_sum_theorem_l1122_112211

theorem cubic_sum_theorem (x y : ℝ) (h1 : y^2 - 3 = (x - 3)^3) (h2 : x^2 - 3 = (y - 3)^2) (h3 : x ≠ y) :
  x^3 + y^3 = 27 + 3 * Real.sqrt 3 ∨ x^3 + y^3 = 27 - 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_theorem_l1122_112211


namespace NUMINAMATH_CALUDE_roses_in_vase_l1122_112251

theorem roses_in_vase (initial_roses final_roses : ℕ) (h1 : initial_roses = 6) (h2 : final_roses = 22) :
  final_roses - initial_roses = 16 := by
  sorry

end NUMINAMATH_CALUDE_roses_in_vase_l1122_112251


namespace NUMINAMATH_CALUDE_equation_solutions_l1122_112224

theorem equation_solutions : 
  {x : ℝ | Real.sqrt (6 * x - 5) + 12 / Real.sqrt (6 * x - 5) = 8} = {41/6, 3/2} := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l1122_112224


namespace NUMINAMATH_CALUDE_opposite_numbers_sum_property_l1122_112265

theorem opposite_numbers_sum_property (a b : ℝ) : 
  (∃ k : ℝ, a = k ∧ b = -k) → -5 * (a + b) = 0 := by
sorry

end NUMINAMATH_CALUDE_opposite_numbers_sum_property_l1122_112265


namespace NUMINAMATH_CALUDE_sugar_harvesting_solution_l1122_112262

/-- Represents the sugar harvesting problem with ants -/
def sugar_harvesting_problem (initial_sugar : ℝ) (harvest_rate : ℝ) (remaining_time : ℝ) : Prop :=
  ∃ (harvesting_time : ℝ),
    initial_sugar - harvest_rate * harvesting_time = harvest_rate * remaining_time ∧
    harvesting_time > 0

/-- Theorem stating the solution to the sugar harvesting problem -/
theorem sugar_harvesting_solution :
  sugar_harvesting_problem 24 4 3 →
  ∃ (harvesting_time : ℝ), harvesting_time = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_sugar_harvesting_solution_l1122_112262


namespace NUMINAMATH_CALUDE_planar_graph_edge_count_l1122_112294

/-- A planar graph -/
structure PlanarGraph where
  V : Type* -- Vertex set
  E : Type* -- Edge set
  n : ℕ     -- Number of vertices
  m : ℕ     -- Number of edges
  is_planar : Bool
  vertex_count : n ≥ 3

/-- A planar triangulation -/
structure PlanarTriangulation extends PlanarGraph where
  is_triangulation : Bool

/-- Theorem about the number of edges in planar graphs and planar triangulations -/
theorem planar_graph_edge_count (G : PlanarGraph) :
  G.m ≤ 3 * G.n - 6 ∧
  (∀ (T : PlanarTriangulation), T.toPlanarGraph = G → T.m = 3 * T.n - 6) :=
sorry

end NUMINAMATH_CALUDE_planar_graph_edge_count_l1122_112294


namespace NUMINAMATH_CALUDE_equal_area_rectangles_l1122_112268

/-- Proves that given two rectangles with equal area, where one rectangle measures 8 inches by 15 inches
    and the other is 30 inches wide, the length of the second rectangle is 4 inches. -/
theorem equal_area_rectangles (carol_length carol_width jordan_width : ℝ) 
    (h1 : carol_length = 8)
    (h2 : carol_width = 15)
    (h3 : jordan_width = 30)
    (h4 : carol_length * carol_width = jordan_width * jordan_length) :
    jordan_length = 4 :=
  sorry

end NUMINAMATH_CALUDE_equal_area_rectangles_l1122_112268


namespace NUMINAMATH_CALUDE_jacket_price_restoration_l1122_112281

theorem jacket_price_restoration (initial_price : ℝ) (h_pos : initial_price > 0) :
  let price_after_first_reduction := initial_price * (1 - 0.25)
  let price_after_second_reduction := price_after_first_reduction * (1 - 0.15)
  let required_increase := (initial_price / price_after_second_reduction) - 1
  abs (required_increase - 0.5686) < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_jacket_price_restoration_l1122_112281


namespace NUMINAMATH_CALUDE_normal_pdf_max_normal_pdf_max_decreases_normal_spread_increases_l1122_112274

/-- The probability density function of a normal distribution -/
noncomputable def normal_pdf (μ σ : ℝ) (x : ℝ) : ℝ :=
  (1 / (σ * Real.sqrt (2 * Real.pi))) * Real.exp (-(1/2) * ((x - μ) / σ)^2)

/-- The maximum height of the normal distribution occurs at x = μ -/
theorem normal_pdf_max (μ σ : ℝ) (h : σ > 0) :
  ∀ x, normal_pdf μ σ x ≤ normal_pdf μ σ μ :=
sorry

/-- As σ increases, the maximum height of the normal distribution decreases -/
theorem normal_pdf_max_decreases (μ : ℝ) (σ₁ σ₂ : ℝ) (h₁ : σ₁ > 0) (h₂ : σ₂ > 0) (h₃ : σ₁ < σ₂) :
  normal_pdf μ σ₂ μ < normal_pdf μ σ₁ μ :=
sorry

/-- The spread of the normal distribution increases as σ increases -/
theorem normal_spread_increases (μ : ℝ) (σ₁ σ₂ : ℝ) (h₁ : σ₁ > 0) (h₂ : σ₂ > 0) (h₃ : σ₁ < σ₂) (ε : ℝ) (hε : ε > 0) :
  ∃ x, |x - μ| > ε ∧ normal_pdf μ σ₂ x > normal_pdf μ σ₁ x :=
sorry

end NUMINAMATH_CALUDE_normal_pdf_max_normal_pdf_max_decreases_normal_spread_increases_l1122_112274


namespace NUMINAMATH_CALUDE_locus_of_midpoints_l1122_112243

/-- The locus of midpoints theorem -/
theorem locus_of_midpoints 
  (A : ℝ × ℝ) 
  (h_A : A = (4, -2))
  (B : ℝ × ℝ → Prop)
  (h_B : ∀ x y, B (x, y) ↔ x^2 + y^2 = 4)
  (P : ℝ × ℝ)
  (h_P : ∃ x y, B (x, y) ∧ P = ((A.1 + x) / 2, (A.2 + y) / 2)) :
  (P.1 - 2)^2 + (P.2 + 1)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_locus_of_midpoints_l1122_112243


namespace NUMINAMATH_CALUDE_red_shells_count_l1122_112235

theorem red_shells_count (total : ℕ) (green : ℕ) (not_red_or_green : ℕ) 
  (h1 : total = 291)
  (h2 : green = 49)
  (h3 : not_red_or_green = 166) :
  total - green - not_red_or_green = 76 := by
sorry

end NUMINAMATH_CALUDE_red_shells_count_l1122_112235


namespace NUMINAMATH_CALUDE_least_prime_angle_in_square_triangle_l1122_112295

theorem least_prime_angle_in_square_triangle (a b : ℕ) : 
  (a > b) →
  (Nat.Prime a) →
  (Nat.Prime b) →
  (a + b = 90) →
  (∀ p, Nat.Prime p → p < b → ¬(∃ q, Nat.Prime q ∧ p + q = 90)) →
  b = 7 := by
sorry

end NUMINAMATH_CALUDE_least_prime_angle_in_square_triangle_l1122_112295


namespace NUMINAMATH_CALUDE_simplify_fraction_l1122_112283

theorem simplify_fraction : (333 : ℚ) / 9999 * 99 = 37 / 101 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1122_112283


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1122_112228

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  (∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q) →  -- a_n is a geometric sequence
  a 1 + a 2 = 3 →                           -- a_1 + a_2 = 3
  a 2 + a 3 = 6 →                           -- a_2 + a_3 = 6
  a 4 + a 5 = 24 :=                         -- a_4 + a_5 = 24
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1122_112228


namespace NUMINAMATH_CALUDE_liam_bills_cost_liam_bills_proof_l1122_112220

/-- Calculates the cost of Liam's bills given his savings and remaining money. -/
theorem liam_bills_cost (monthly_savings : ℕ) (savings_duration_months : ℕ) (money_left : ℕ) : ℕ :=
  let total_savings := monthly_savings * savings_duration_months
  total_savings - money_left

/-- Proves that Liam's bills cost $3,500 given the problem conditions. -/
theorem liam_bills_proof :
  liam_bills_cost 500 24 8500 = 3500 := by
  sorry

end NUMINAMATH_CALUDE_liam_bills_cost_liam_bills_proof_l1122_112220


namespace NUMINAMATH_CALUDE_sum_equals_300_l1122_112275

theorem sum_equals_300 : 157 + 43 + 19 + 81 = 300 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_300_l1122_112275


namespace NUMINAMATH_CALUDE_intersection_distance_l1122_112246

/-- Given a linear function f(x) = ax + b, if the distance between intersection points
    of y = x^2 - 2 and y = f(x) is √26, and the distance between intersection points
    of y = x^2 and y = f(x) + 1 is 3√2, then the distance between intersection points
    of y = x^2 and y = f(x) is √10. -/
theorem intersection_distance (a b : ℝ) : 
  let f := fun (x : ℝ) => a * x + b
  (∃ x₁ x₂ : ℝ, x₁^2 - 2 = f x₁ ∧ x₂^2 - 2 = f x₂ ∧ (x₂ - x₁)^2 = 26) →
  (∃ y₁ y₂ : ℝ, y₁^2 = f y₁ + 1 ∧ y₂^2 = f y₂ + 1 ∧ (y₂ - y₁)^2 = 18) →
  ∃ z₁ z₂ : ℝ, z₁^2 = f z₁ ∧ z₂^2 = f z₂ ∧ (z₂ - z₁)^2 = 10 :=
by sorry

end NUMINAMATH_CALUDE_intersection_distance_l1122_112246


namespace NUMINAMATH_CALUDE_base_conversion_1765_l1122_112261

/-- Converts a base 10 number to its base 6 representation -/
def toBase6 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base 6 to a natural number in base 10 -/
def fromBase6 (digits : List ℕ) : ℕ :=
  sorry

theorem base_conversion_1765 :
  toBase6 1765 = [1, 2, 1, 0, 1] ∧ fromBase6 [1, 2, 1, 0, 1] = 1765 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_1765_l1122_112261


namespace NUMINAMATH_CALUDE_fair_coin_three_heads_probability_l1122_112226

theorem fair_coin_three_heads_probability :
  let p_head : ℚ := 1/2  -- Probability of getting heads on a single flip
  let p_three_heads : ℚ := p_head * p_head * p_head  -- Probability of getting heads on all three flips
  p_three_heads = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_fair_coin_three_heads_probability_l1122_112226


namespace NUMINAMATH_CALUDE_inequality_proof_l1122_112266

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a * b + b * c + c * a = 3) : 
  (a / Real.sqrt (a^3 + 5)) + (b / Real.sqrt (b^3 + 5)) + (c / Real.sqrt (c^3 + 5)) ≤ Real.sqrt 6 / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1122_112266


namespace NUMINAMATH_CALUDE_max_salary_baseball_team_l1122_112217

/-- Represents the maximum salary for a single player in a baseball team under given constraints -/
def max_player_salary (num_players : ℕ) (min_salary : ℕ) (total_budget : ℕ) : ℕ :=
  total_budget - (num_players - 1) * min_salary

/-- Theorem stating the maximum possible salary for a single player under given constraints -/
theorem max_salary_baseball_team :
  max_player_salary 18 20000 600000 = 260000 :=
by sorry

end NUMINAMATH_CALUDE_max_salary_baseball_team_l1122_112217


namespace NUMINAMATH_CALUDE_loan_repayment_proof_l1122_112239

/-- Calculates the total amount to be repaid for a loan with simple interest -/
def total_repayment (initial_loan : ℝ) (additional_loan : ℝ) (initial_period : ℝ) (total_period : ℝ) (rate : ℝ) : ℝ :=
  let initial_with_interest := initial_loan * (1 + rate * initial_period)
  let total_loan := initial_with_interest + additional_loan
  total_loan * (1 + rate * (total_period - initial_period))

/-- Proves that the total repayment for the given loan scenario is 27376 Rs -/
theorem loan_repayment_proof :
  total_repayment 10000 12000 2 5 0.06 = 27376 := by
  sorry

#eval total_repayment 10000 12000 2 5 0.06

end NUMINAMATH_CALUDE_loan_repayment_proof_l1122_112239


namespace NUMINAMATH_CALUDE_inequality_proof_l1122_112208

theorem inequality_proof (a b : ℝ) (h : a + b = 1) :
  Real.sqrt (1 + 5 * a^2) + 5 * Real.sqrt (2 + b^2) ≥ 9 := by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1122_112208


namespace NUMINAMATH_CALUDE_power_of_power_l1122_112227

theorem power_of_power : (3^2)^4 = 6561 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l1122_112227


namespace NUMINAMATH_CALUDE_specific_shaded_square_ratio_l1122_112286

/-- A square divided into smaller squares with a shading pattern -/
structure ShadedSquare where
  /-- The number of equal triangles in the shaded area of each quarter -/
  shaded_triangles : ℕ
  /-- The number of equal triangles in the white area of each quarter -/
  white_triangles : ℕ

/-- The ratio of shaded area to white area in a ShadedSquare -/
def shaded_to_white_ratio (s : ShadedSquare) : ℚ :=
  s.shaded_triangles / s.white_triangles

/-- Theorem stating the ratio of shaded to white area for a specific configuration -/
theorem specific_shaded_square_ratio :
  ∃ (s : ShadedSquare), s.shaded_triangles = 5 ∧ s.white_triangles = 3 ∧ 
  shaded_to_white_ratio s = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_specific_shaded_square_ratio_l1122_112286


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1122_112270

-- Define the sets A and B
def A : Set ℝ := {x | x / (x - 1) < 0}
def B : Set ℝ := {x | 0 < x ∧ x < 3}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1122_112270


namespace NUMINAMATH_CALUDE_smallest_n_g_greater_than_20_l1122_112212

/-- The sum of digits to the right of the decimal point in 1/(3^n) -/
def g (n : ℕ+) : ℕ := sorry

/-- Theorem stating that 4 is the smallest positive integer n such that g(n) > 20 -/
theorem smallest_n_g_greater_than_20 :
  (∀ k : ℕ+, k < 4 → g k ≤ 20) ∧ g 4 > 20 := by sorry

end NUMINAMATH_CALUDE_smallest_n_g_greater_than_20_l1122_112212


namespace NUMINAMATH_CALUDE_right_triangle_geometric_sequence_l1122_112200

theorem right_triangle_geometric_sequence (a b c q : ℝ) : 
  q > 1 →
  a > 0 →
  b > 0 →
  c > 0 →
  a * q = b →
  b * q = c →
  a^2 + b^2 = c^2 →
  q^2 = (Real.sqrt 5 + 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_geometric_sequence_l1122_112200


namespace NUMINAMATH_CALUDE_continuity_from_g_and_h_continuous_l1122_112280

open Function Set Filter Topology

/-- Given a function f: ℝ → ℝ, if g(x) = f(x) + f(2x) and h(x) = f(x) + f(4x) are continuous,
    then f is continuous. -/
theorem continuity_from_g_and_h_continuous
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (h : ℝ → ℝ)
  (hg : g = λ x => f x + f (2 * x))
  (hh : h = λ x => f x + f (4 * x))
  (hg_cont : Continuous g)
  (hh_cont : Continuous h) :
  Continuous f :=
sorry

end NUMINAMATH_CALUDE_continuity_from_g_and_h_continuous_l1122_112280


namespace NUMINAMATH_CALUDE_solution_set_implies_k_inequality_implies_k_range_l1122_112209

/-- The quadratic function f(x) = kx^2 - 2x + 6k --/
def f (k : ℝ) (x : ℝ) : ℝ := k * x^2 - 2 * x + 6 * k

/-- Theorem 1: If f(x) < 0 has solution set (2,3), then k = 2/5 --/
theorem solution_set_implies_k (k : ℝ) :
  (∀ x, f k x < 0 ↔ 2 < x ∧ x < 3) → k = 2/5 := by sorry

/-- Theorem 2: If k > 0 and f(x) < 0 for all 2 < x < 3, then 0 < k ≤ 2/5 --/
theorem inequality_implies_k_range (k : ℝ) :
  k > 0 → (∀ x, 2 < x → x < 3 → f k x < 0) → 0 < k ∧ k ≤ 2/5 := by sorry

end NUMINAMATH_CALUDE_solution_set_implies_k_inequality_implies_k_range_l1122_112209


namespace NUMINAMATH_CALUDE_twins_age_problem_l1122_112233

theorem twins_age_problem (age : ℕ) : 
  (age + 1) * (age + 1) = age * age + 15 → age = 7 := by
  sorry

end NUMINAMATH_CALUDE_twins_age_problem_l1122_112233


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_ratio_existence_l1122_112272

theorem arithmetic_geometric_mean_ratio_existence :
  ∃ (a b : ℝ), 
    (a + b) / 2 = 3 * Real.sqrt (a * b) ∧
    a > b ∧ b > 0 ∧
    round (a / b) = 28 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_ratio_existence_l1122_112272


namespace NUMINAMATH_CALUDE_vacation_savings_l1122_112273

def total_income : ℝ := 72800
def total_expenses : ℝ := 54200
def deposit_rate : ℝ := 0.1

theorem vacation_savings : 
  (total_income - total_expenses) * (1 - deposit_rate) = 16740 := by
  sorry

end NUMINAMATH_CALUDE_vacation_savings_l1122_112273


namespace NUMINAMATH_CALUDE_min_apples_in_basket_l1122_112206

theorem min_apples_in_basket : ∃ n : ℕ, n ≥ 23 ∧ 
  (∃ a b c : ℕ, 
    n + 4 = 3 * a ∧
    2 * a + 4 = 3 * b ∧
    2 * b + 4 = 3 * c) ∧
  (∀ m : ℕ, m < n → 
    ¬(∃ a b c : ℕ, 
      m + 4 = 3 * a ∧
      2 * a + 4 = 3 * b ∧
      2 * b + 4 = 3 * c)) :=
by sorry

end NUMINAMATH_CALUDE_min_apples_in_basket_l1122_112206


namespace NUMINAMATH_CALUDE_fraction_sum_equals_123_128th_l1122_112218

theorem fraction_sum_equals_123_128th : 
  (4 : ℚ) / 4 + 7 / 8 + 12 / 16 + 19 / 32 + 28 / 64 + 39 / 128 - 3 = 123 / 128 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_123_128th_l1122_112218


namespace NUMINAMATH_CALUDE_binomial_20_4_l1122_112298

theorem binomial_20_4 : Nat.choose 20 4 = 4845 := by sorry

end NUMINAMATH_CALUDE_binomial_20_4_l1122_112298


namespace NUMINAMATH_CALUDE_expression_simplification_l1122_112279

theorem expression_simplification (x : ℝ) : (3*x - 4)*(2*x + 9) - (x + 6)*(3*x + 2) = 3*x^2 - x - 48 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1122_112279


namespace NUMINAMATH_CALUDE_number_difference_l1122_112248

theorem number_difference (x y : ℤ) : 
  x + y = 50 → 
  y = 19 → 
  x < 2 * y → 
  2 * y - x = 7 := by
sorry

end NUMINAMATH_CALUDE_number_difference_l1122_112248


namespace NUMINAMATH_CALUDE_sin_15_degrees_l1122_112284

theorem sin_15_degrees : Real.sin (15 * π / 180) = (Real.sqrt 6 - Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_15_degrees_l1122_112284


namespace NUMINAMATH_CALUDE_circle_arrangement_theorem_l1122_112258

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle with a center and radius -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

def circleA : Circle := { center := { x := 0, y := -1 }, radius := 1 }
def circleB : Circle := { center := { x := 5, y := 3 }, radius := 3 }
def circleC : Circle := { center := { x := 8, y := -4 }, radius := 4 }

def line_l : Line := { a := 0, b := 1, c := 0 }

def is_below (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c < 0

def is_above (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c > 0

def is_tangent (c : Circle) (l : Line) : Prop :=
  abs (l.a * c.center.x + l.b * c.center.y + l.c) = c.radius * (l.a^2 + l.b^2).sqrt

def are_externally_tangent (c1 c2 : Circle) : Prop :=
  let dx := c1.center.x - c2.center.x
  let dy := c1.center.y - c2.center.y
  (dx^2 + dy^2).sqrt = c1.radius + c2.radius

def triangle_area (p1 p2 p3 : Point) : ℝ :=
  0.5 * abs (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))

theorem circle_arrangement_theorem :
  is_below circleA.center line_l ∧
  is_below circleC.center line_l ∧
  is_above circleB.center line_l ∧
  is_tangent circleA line_l ∧
  is_tangent circleB line_l ∧
  is_tangent circleC line_l ∧
  are_externally_tangent circleB circleA ∧
  are_externally_tangent circleB circleC →
  triangle_area circleA.center circleB.center circleC.center = 23.5 := by
  sorry

end NUMINAMATH_CALUDE_circle_arrangement_theorem_l1122_112258


namespace NUMINAMATH_CALUDE_smallest_natural_number_square_cube_seventy_two_satisfies_conditions_smallest_natural_number_is_72_l1122_112247

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

def is_perfect_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

theorem smallest_natural_number_square_cube : 
  ∀ x : ℕ, (is_perfect_square (2 * x) ∧ is_perfect_cube (3 * x)) → x ≥ 72 :=
by sorry

theorem seventy_two_satisfies_conditions : 
  is_perfect_square (2 * 72) ∧ is_perfect_cube (3 * 72) :=
by sorry

theorem smallest_natural_number_is_72 : 
  ∃! x : ℕ, x = 72 ∧ 
    (∀ y : ℕ, (is_perfect_square (2 * y) ∧ is_perfect_cube (3 * y)) → y ≥ x) :=
by sorry

end NUMINAMATH_CALUDE_smallest_natural_number_square_cube_seventy_two_satisfies_conditions_smallest_natural_number_is_72_l1122_112247


namespace NUMINAMATH_CALUDE_square_reciprocal_sum_l1122_112250

theorem square_reciprocal_sum (m : ℝ) (h : m + 1/m = 5) : 
  m^2 + 1/m^2 + 4 = 27 := by
  sorry

end NUMINAMATH_CALUDE_square_reciprocal_sum_l1122_112250


namespace NUMINAMATH_CALUDE_magnificent_monday_l1122_112232

-- Define a structure for a month
structure Month where
  days : Nat
  firstMonday : Nat

-- Define a function to calculate the date of the nth Monday
def nthMonday (m : Month) (n : Nat) : Nat :=
  m.firstMonday + 7 * (n - 1)

-- Theorem statement
theorem magnificent_monday (m : Month) 
  (h1 : m.days = 31)
  (h2 : m.firstMonday = 2) :
  nthMonday m 5 = 30 := by
  sorry

end NUMINAMATH_CALUDE_magnificent_monday_l1122_112232


namespace NUMINAMATH_CALUDE_total_shark_teeth_l1122_112296

def tiger_shark_teeth : ℕ := 180

def hammerhead_shark_teeth : ℕ := tiger_shark_teeth / 6

def great_white_shark_teeth : ℕ := 2 * (tiger_shark_teeth + hammerhead_shark_teeth)

def mako_shark_teeth : ℕ := (5 * hammerhead_shark_teeth) / 3

theorem total_shark_teeth : 
  tiger_shark_teeth + hammerhead_shark_teeth + great_white_shark_teeth + mako_shark_teeth = 680 := by
  sorry

end NUMINAMATH_CALUDE_total_shark_teeth_l1122_112296


namespace NUMINAMATH_CALUDE_expand_expression_l1122_112225

theorem expand_expression (x y z : ℝ) : 
  (x + 12) * (3 * y + 4 * z + 15) = 3 * x * y + 4 * x * z + 15 * x + 36 * y + 48 * z + 180 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l1122_112225


namespace NUMINAMATH_CALUDE_basketball_win_percentage_l1122_112256

theorem basketball_win_percentage (games_played : ℕ) (games_won : ℕ) (games_left : ℕ) 
  (target_percentage : ℚ) (h1 : games_played = 50) (h2 : games_won = 35) 
  (h3 : games_left = 25) (h4 : target_percentage = 64 / 100) : 
  ∃ (additional_wins : ℕ), 
    (games_won + additional_wins) / (games_played + games_left : ℚ) = target_percentage ∧ 
    additional_wins = 13 := by
  sorry

end NUMINAMATH_CALUDE_basketball_win_percentage_l1122_112256


namespace NUMINAMATH_CALUDE_two_roses_more_expensive_than_three_carnations_l1122_112297

/-- The price of a single rose in yuan -/
def rose_price : ℝ := sorry

/-- The price of a single carnation in yuan -/
def carnation_price : ℝ := sorry

/-- The combined price of 6 roses and 3 carnations -/
def combined_price_1 : ℝ := 6 * rose_price + 3 * carnation_price

/-- The combined price of 4 roses and 5 carnations -/
def combined_price_2 : ℝ := 4 * rose_price + 5 * carnation_price

/-- Theorem stating that the price of 2 roses is higher than the price of 3 carnations -/
theorem two_roses_more_expensive_than_three_carnations 
  (h1 : combined_price_1 > 24)
  (h2 : combined_price_2 < 22) :
  2 * rose_price > 3 * carnation_price :=
by sorry

end NUMINAMATH_CALUDE_two_roses_more_expensive_than_three_carnations_l1122_112297


namespace NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l1122_112210

/-- Given an arithmetic sequence {a_n} where a_2 + a_4 = 5, prove that a_3 = 5/2 -/
theorem arithmetic_sequence_third_term 
  (a : ℕ → ℚ) -- a is a function from natural numbers to rationals
  (h_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) -- arithmetic sequence condition
  (h_sum : a 2 + a 4 = 5) -- given condition
  : a 3 = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l1122_112210


namespace NUMINAMATH_CALUDE_pencil_box_calculation_l1122_112259

/-- Given a total number of pencils and pencils per box, calculate the number of filled boxes -/
def filled_boxes (total_pencils : ℕ) (pencils_per_box : ℕ) : ℕ :=
  total_pencils / pencils_per_box

/-- Theorem: Given 648 pencils and 4 pencils per box, the number of filled boxes is 162 -/
theorem pencil_box_calculation :
  filled_boxes 648 4 = 162 := by
  sorry

end NUMINAMATH_CALUDE_pencil_box_calculation_l1122_112259


namespace NUMINAMATH_CALUDE_special_line_equation_l1122_112282

/-- A line passing through (5,2) with y-intercept twice the x-intercept -/
structure SpecialLine where
  -- The slope-intercept form of the line: y = mx + b
  m : ℝ
  b : ℝ
  -- The line passes through (5,2)
  point_condition : 2 = m * 5 + b
  -- The y-intercept is twice the x-intercept
  intercept_condition : b = -2 * (b / m)

/-- The equation of the special line is either 2x+y-12=0 or 2x-5y=0 -/
theorem special_line_equation (l : SpecialLine) :
  (2 * l.m + 1 ≠ 0 ∧ 2 * l.m * l.b + l.b = 12) ∨
  (l.m = 2/5 ∧ l.b = 0) :=
sorry

end NUMINAMATH_CALUDE_special_line_equation_l1122_112282


namespace NUMINAMATH_CALUDE_division_remainder_proof_l1122_112249

theorem division_remainder_proof (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) : 
  dividend = 729 → divisor = 38 → quotient = 19 → 
  dividend = divisor * quotient + remainder → remainder = 7 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_proof_l1122_112249


namespace NUMINAMATH_CALUDE_tangent_line_at_one_l1122_112203

noncomputable section

variable (f : ℝ → ℝ)

-- Define the function property
def function_property (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = 2 * f (2 * x - 1) - 3 * x^2 + 2

-- Define the tangent line equation
def tangent_line_equation (f : ℝ → ℝ) : ℝ → ℝ := λ x ↦ 2 * x - 1

-- Theorem statement
theorem tangent_line_at_one (h : function_property f) :
  ∃ (f' : ℝ → ℝ), (∀ x, HasDerivAt f (f' x) x) ∧
  (∀ x, (tangent_line_equation f) x = f 1 + f' 1 * (x - 1)) :=
sorry

end

end NUMINAMATH_CALUDE_tangent_line_at_one_l1122_112203


namespace NUMINAMATH_CALUDE_eggs_in_box_l1122_112291

/-- The number of eggs in the box after adding more eggs -/
def total_eggs (initial : Float) (added : Float) : Float :=
  initial + added

/-- Theorem stating that adding 5.0 eggs to 47.0 eggs results in 52.0 eggs -/
theorem eggs_in_box : total_eggs 47.0 5.0 = 52.0 := by
  sorry

end NUMINAMATH_CALUDE_eggs_in_box_l1122_112291


namespace NUMINAMATH_CALUDE_problem_statement_l1122_112254

open Real

theorem problem_statement : 
  let p := ∃ x₀ : ℝ, Real.exp x₀ ≤ 0
  let q := ∀ x : ℝ, 2^x > x^2
  (¬p) ∨ q := by sorry

end NUMINAMATH_CALUDE_problem_statement_l1122_112254


namespace NUMINAMATH_CALUDE_volume_to_surface_area_ratio_l1122_112276

/-- A structure composed of unit cubes -/
structure CubeStructure where
  /-- The number of unit cubes in the structure -/
  num_cubes : ℕ
  /-- The number of cubes surrounding the center cube -/
  surrounding_cubes : ℕ
  /-- Assertion that there is one center cube -/
  has_center_cube : num_cubes = surrounding_cubes + 1

/-- Calculate the volume of the structure -/
def volume (s : CubeStructure) : ℕ := s.num_cubes

/-- Calculate the surface area of the structure -/
def surface_area (s : CubeStructure) : ℕ :=
  1 + (s.surrounding_cubes - 1) * 5 + 4

/-- The theorem to be proved -/
theorem volume_to_surface_area_ratio (s : CubeStructure) 
  (h1 : s.num_cubes = 10) 
  (h2 : s.surrounding_cubes = 9) : 
  (volume s : ℚ) / (surface_area s : ℚ) = 2/9 := by
  sorry

end NUMINAMATH_CALUDE_volume_to_surface_area_ratio_l1122_112276


namespace NUMINAMATH_CALUDE_percentage_problem_l1122_112299

theorem percentage_problem (a b c : ℝ) : 
  a = 0.8 * b → 
  c = 1.4 * b → 
  c - a = 72 → 
  a = 96 ∧ b = 120 ∧ c = 168 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1122_112299


namespace NUMINAMATH_CALUDE_min_average_of_four_integers_l1122_112277

theorem min_average_of_four_integers (a b c d : ℤ) : 
  a < b ∧ b < c ∧ c < d ∧  -- Four different integers
  d = 90 ∧                 -- Largest is 90
  a ≥ 29 →                 -- Smallest is at least 29
  (a + b + c + d) / 4 ≥ 45 :=
sorry

end NUMINAMATH_CALUDE_min_average_of_four_integers_l1122_112277


namespace NUMINAMATH_CALUDE_min_value_of_function_l1122_112231

theorem min_value_of_function (x : ℝ) (h : x > 0) : 
  4 * x + 25 / x ≥ 20 ∧ ∃ y > 0, 4 * y + 25 / y = 20 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_l1122_112231


namespace NUMINAMATH_CALUDE_organization_members_l1122_112237

/-- The number of committees in the organization -/
def num_committees : ℕ := 5

/-- The number of committees each member belongs to -/
def committees_per_member : ℕ := 2

/-- The number of unique members shared between each pair of committees -/
def shared_members_per_pair : ℕ := 2

/-- The total number of members in the organization -/
def total_members : ℕ := 10

/-- Theorem stating the total number of members in the organization -/
theorem organization_members :
  (num_committees = 5) →
  (committees_per_member = 2) →
  (shared_members_per_pair = 2) →
  (total_members = 10) :=
by sorry

end NUMINAMATH_CALUDE_organization_members_l1122_112237


namespace NUMINAMATH_CALUDE_ratio_of_sum_and_difference_l1122_112223

theorem ratio_of_sum_and_difference (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) 
  (h : a + b = 5 * (a - b)) : a / b = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_sum_and_difference_l1122_112223


namespace NUMINAMATH_CALUDE_polynomial_sum_of_coefficients_l1122_112285

theorem polynomial_sum_of_coefficients 
  (g : ℂ → ℂ) 
  (p q r s : ℝ) 
  (h1 : ∀ x, g x = x^4 + p*x^3 + q*x^2 + r*x + s)
  (h2 : g (3*I) = 0)
  (h3 : g (1 + 2*I) = 0) :
  p + q + r + s = 39 := by
sorry

end NUMINAMATH_CALUDE_polynomial_sum_of_coefficients_l1122_112285


namespace NUMINAMATH_CALUDE_vector_sum_zero_l1122_112292

variable {E : Type*} [NormedAddCommGroup E]

/-- Given vectors CE, AC, DE, and AD in a normed additive commutative group E,
    prove that CE + AC - DE - AD = 0 -/
theorem vector_sum_zero (CE AC DE AD : E) :
  CE + AC - DE - AD = (0 : E) := by sorry

end NUMINAMATH_CALUDE_vector_sum_zero_l1122_112292


namespace NUMINAMATH_CALUDE_characterize_solutions_l1122_112257

/-- The functional equation satisfied by f and g -/
def functional_equation (f g : ℕ → ℕ) : Prop :=
  ∀ n, f n + f (n + g n) = f (n + 1)

/-- The trivial solution where f is identically zero -/
def trivial_solution (f g : ℕ → ℕ) : Prop :=
  ∀ n, f n = 0

/-- The non-trivial solution family -/
def non_trivial_solution (f g : ℕ → ℕ) : Prop :=
  ∃ n₀ c : ℕ,
    (∀ n < n₀, f n = 0) ∧
    (∀ n ≥ n₀, f n = c * 2^(n - n₀)) ∧
    (∀ n < n₀ - 1, g n < n₀ - n) ∧
    (g (n₀ - 1) = 1) ∧
    (∀ n ≥ n₀, g n = 0)

/-- The main theorem characterizing all solutions to the functional equation -/
theorem characterize_solutions (f g : ℕ → ℕ) :
  functional_equation f g → (trivial_solution f g ∨ non_trivial_solution f g) :=
sorry

end NUMINAMATH_CALUDE_characterize_solutions_l1122_112257


namespace NUMINAMATH_CALUDE_obtuse_triangle_from_altitudes_l1122_112287

theorem obtuse_triangle_from_altitudes (h₁ h₂ h₃ : ℝ) 
  (h_pos : h₁ > 0 ∧ h₂ > 0 ∧ h₃ > 0) 
  (h_ineq : 1/h₁ + 1/h₂ > 1/h₃ ∧ 1/h₂ + 1/h₃ > 1/h₁ ∧ 1/h₃ + 1/h₁ > 1/h₂) :
  ∃ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a + b > c ∧ b + c > a ∧ c + a > b ∧
    h₁ = (2 * (a * b * c / (a + b + c))) / a ∧
    h₂ = (2 * (a * b * c / (a + b + c))) / b ∧
    h₃ = (2 * (a * b * c / (a + b + c))) / c ∧
    a^2 + b^2 < c^2 :=
sorry

end NUMINAMATH_CALUDE_obtuse_triangle_from_altitudes_l1122_112287


namespace NUMINAMATH_CALUDE_tangent_line_at_one_l1122_112263

noncomputable def f (x : ℝ) : ℝ := (x + 1) * Real.log x - 4 * (x - 1)

noncomputable def f' (x : ℝ) : ℝ := Real.log x + (x + 1) / x - 4

theorem tangent_line_at_one (x y : ℝ) :
  (f' 1 = -2) →
  (f 1 = 0) →
  (∀ ε > 0, ∃ δ > 0, ∀ h : ℝ, |h| < δ → |f (1 + h) - (f 1 + f' 1 * h)| ≤ ε * |h|) →
  (2 * x + y - 2 = 0 ↔ y = f' 1 * (x - 1) + f 1) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_l1122_112263


namespace NUMINAMATH_CALUDE_lateral_side_is_five_l1122_112271

/-- An isosceles trapezoid with given dimensions -/
structure IsoscelesTrapezoid where
  base1 : ℝ
  base2 : ℝ
  area : ℝ
  lateral : ℝ

/-- The property that the given dimensions form a valid isosceles trapezoid -/
def is_valid_trapezoid (t : IsoscelesTrapezoid) : Prop :=
  t.base1 > 0 ∧ t.base2 > 0 ∧ t.area > 0 ∧ t.lateral > 0 ∧
  t.area = (t.base1 + t.base2) * t.lateral / 2

/-- The theorem stating that the lateral side of the trapezoid is 5 -/
theorem lateral_side_is_five (t : IsoscelesTrapezoid)
  (h1 : t.base1 = 8)
  (h2 : t.base2 = 14)
  (h3 : t.area = 44)
  (h4 : is_valid_trapezoid t) :
  t.lateral = 5 :=
sorry

end NUMINAMATH_CALUDE_lateral_side_is_five_l1122_112271


namespace NUMINAMATH_CALUDE_equal_sum_of_intervals_l1122_112244

-- Define the function f on the interval [a, b]
variable (f : ℝ → ℝ)
variable (a b : ℝ)

-- Define the property of f being continuous on [a, b]
def IsContinuousOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, a ≤ x ∧ x ≤ b → ContinuousAt f x

-- Define the property of f(a) = f(b)
def HasEqualEndpoints (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  f a = f b

-- Define the sum of lengths of intervals where f is increasing
def SumOfIncreasingIntervals (f : ℝ → ℝ) (a b : ℝ) : ℝ := sorry

-- Define the sum of lengths of intervals where f is decreasing
def SumOfDecreasingIntervals (f : ℝ → ℝ) (a b : ℝ) : ℝ := sorry

-- State the theorem
theorem equal_sum_of_intervals
  (h1 : IsContinuousOn f a b)
  (h2 : HasEqualEndpoints f a b)
  (h3 : a ≤ b) :
  SumOfIncreasingIntervals f a b = SumOfDecreasingIntervals f a b :=
by sorry

end NUMINAMATH_CALUDE_equal_sum_of_intervals_l1122_112244


namespace NUMINAMATH_CALUDE_inequality_proof_l1122_112222

theorem inequality_proof (a b c d : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_pos_d : d > 0)
  (h_product : a * b * c * d = 1) :
  a^2 + b^2 + c^2 + d^2 + a*b + a*c + a*d + b*c + b*d + c*d ≥ 10 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1122_112222


namespace NUMINAMATH_CALUDE_sin_difference_monotone_increasing_l1122_112252

/-- The function f(x) = sin(2x - π/3) - sin(2x) is monotonically increasing on [π/12, 7π/12] -/
theorem sin_difference_monotone_increasing :
  let f : ℝ → ℝ := λ x ↦ Real.sin (2 * x - π / 3) - Real.sin (2 * x)
  ∀ x y, π / 12 ≤ x ∧ x < y ∧ y ≤ 7 * π / 12 → f x < f y := by
  sorry

end NUMINAMATH_CALUDE_sin_difference_monotone_increasing_l1122_112252


namespace NUMINAMATH_CALUDE_triangle_property_l1122_112213

/-- Given a triangle ABC where a^2 + c^2 = b^2 + √2*a*c, prove that:
    1. The size of angle B is π/4
    2. The maximum value of √2*cos(A) + cos(C) is 1 -/
theorem triangle_property (a b c : ℝ) (h : a^2 + c^2 = b^2 + Real.sqrt 2 * a * c) :
  let A := Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c))
  let B := Real.arccos ((a^2 + c^2 - b^2) / (2 * a * c))
  let C := Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))
  (B = π / 4) ∧
  (∃ (x : ℝ), Real.sqrt 2 * Real.cos A + Real.cos C ≤ x ∧ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_triangle_property_l1122_112213


namespace NUMINAMATH_CALUDE_sophie_total_spend_l1122_112215

def cupcake_quantity : ℕ := 5
def cupcake_price : ℚ := 2

def doughnut_quantity : ℕ := 6
def doughnut_price : ℚ := 1

def apple_pie_quantity : ℕ := 4
def apple_pie_price : ℚ := 2

def cookie_quantity : ℕ := 15
def cookie_price : ℚ := 0.6

theorem sophie_total_spend :
  (cupcake_quantity : ℚ) * cupcake_price +
  (doughnut_quantity : ℚ) * doughnut_price +
  (apple_pie_quantity : ℚ) * apple_pie_price +
  (cookie_quantity : ℚ) * cookie_price = 33 := by
  sorry

end NUMINAMATH_CALUDE_sophie_total_spend_l1122_112215


namespace NUMINAMATH_CALUDE_finsler_hadwiger_inequality_l1122_112216

/-- The Finsler-Hadwiger inequality for triangles -/
theorem finsler_hadwiger_inequality (a b c : ℝ) (S : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) (h_area : S > 0) :
  a^2 + b^2 + c^2 - (a-b)^2 - (b-c)^2 - (c-a)^2 ≥ 4 * Real.sqrt 3 * S := by
  sorry

end NUMINAMATH_CALUDE_finsler_hadwiger_inequality_l1122_112216


namespace NUMINAMATH_CALUDE_wombat_claws_l1122_112241

theorem wombat_claws (num_wombats num_rheas total_claws : ℕ) 
  (h1 : num_wombats = 9)
  (h2 : num_rheas = 3)
  (h3 : total_claws = 39) :
  ∃ (wombat_claws : ℕ), 
    wombat_claws * num_wombats + num_rheas = total_claws ∧ 
    wombat_claws = 4 := by
  sorry

end NUMINAMATH_CALUDE_wombat_claws_l1122_112241


namespace NUMINAMATH_CALUDE_g_composition_of_3_l1122_112221

def g (x : ℤ) : ℤ :=
  if x % 2 = 0 then x / 2 else 5 * x + 3

theorem g_composition_of_3 : g (g (g (g 3))) = 24 := by
  sorry

end NUMINAMATH_CALUDE_g_composition_of_3_l1122_112221


namespace NUMINAMATH_CALUDE_inhabitable_earth_surface_inhabitable_earth_surface_proof_l1122_112214

theorem inhabitable_earth_surface : Real → Prop :=
  λ x =>
    let total_surface := 1
    let land_fraction := 1 / 4
    let inhabitable_land_fraction := 1 / 2
    x = land_fraction * inhabitable_land_fraction ∧ 
    x = 1 / 8

-- Proof
theorem inhabitable_earth_surface_proof : inhabitable_earth_surface (1 / 8) := by
  sorry

end NUMINAMATH_CALUDE_inhabitable_earth_surface_inhabitable_earth_surface_proof_l1122_112214


namespace NUMINAMATH_CALUDE_quadratic_equations_common_root_l1122_112269

theorem quadratic_equations_common_root (p q r s : ℝ) 
  (hq : q ≠ -1) (hs : s ≠ -1) : 
  (∃ (a b : ℝ), (a^2 + p*a + q = 0 ∧ a^2 + r*a + s = 0) ∧ 
   (b^2 + p*b + q = 0 ∧ (1/b)^2 + r*(1/b) + s = 0)) ↔ 
  (p*r = (q+1)*(s+1) ∧ p*(q+1)*s = r*(s+1)*q) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equations_common_root_l1122_112269


namespace NUMINAMATH_CALUDE_area_ratio_theorem_l1122_112236

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a triangle -/
structure Triangle :=
  (A : Point) (B : Point) (C : Point)

/-- Represents a trapezoid -/
structure Trapezoid :=
  (H : Point) (I : Point) (J : Point) (K : Point)

/-- Checks if a triangle is equilateral -/
def isEquilateral (t : Triangle) : Prop := sorry

/-- Checks if two lines are parallel -/
def areParallel (p1 p2 q1 q2 : Point) : Prop := sorry

/-- Checks if four points are equally spaced on a line -/
def equallySpaced (p1 p2 p3 p4 : Point) : Prop := sorry

/-- Calculates the area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- Calculates the area of a trapezoid -/
def trapezoidArea (t : Trapezoid) : ℝ := sorry

theorem area_ratio_theorem (ABC : Triangle) (HIJK : Trapezoid) 
  (D E F G : Point) :
  isEquilateral ABC →
  areParallel D E B C →
  areParallel F G B C →
  areParallel HIJK.H HIJK.I B C →
  areParallel HIJK.J HIJK.K B C →
  equallySpaced ABC.A D F HIJK.H →
  equallySpaced ABC.A D F HIJK.J →
  trapezoidArea HIJK / triangleArea ABC = 9 / 25 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_theorem_l1122_112236


namespace NUMINAMATH_CALUDE_class_test_probabilities_l1122_112289

theorem class_test_probabilities (p_first p_second p_both : ℝ) 
  (h1 : p_first = 0.63)
  (h2 : p_second = 0.49)
  (h3 : p_both = 0.32) :
  1 - (p_first + p_second - p_both) = 0.20 := by
  sorry

end NUMINAMATH_CALUDE_class_test_probabilities_l1122_112289


namespace NUMINAMATH_CALUDE_increasing_function_condition_l1122_112207

/-- The function f(x) = lg(x^2 - mx - m) is increasing on (1, +∞) iff m ≤ 1/2 -/
theorem increasing_function_condition (m : ℝ) :
  (∀ x > 1, StrictMono (fun x => Real.log (x^2 - m*x - m))) ↔ m ≤ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_increasing_function_condition_l1122_112207


namespace NUMINAMATH_CALUDE_system_solution_l1122_112253

theorem system_solution (x y z w : ℝ) : 
  (x - y + z - w = 2 ∧
   x^2 - y^2 + z^2 - w^2 = 6 ∧
   x^3 - y^3 + z^3 - w^3 = 20 ∧
   x^4 - y^4 + z^4 - w^4 = 60) ↔ 
  ((x = 1 ∧ y = 2 ∧ z = 3 ∧ w = 0) ∨
   (x = 1 ∧ y = 0 ∧ z = 3 ∧ w = 2) ∨
   (x = 3 ∧ y = 2 ∧ z = 1 ∧ w = 0) ∨
   (x = 3 ∧ y = 0 ∧ z = 1 ∧ w = 2)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1122_112253


namespace NUMINAMATH_CALUDE_smallest_AAB_l1122_112201

theorem smallest_AAB : ∃ (A B : ℕ),
  A ≠ B ∧
  A ∈ Finset.range 10 ∧
  B ∈ Finset.range 10 ∧
  A ≠ 0 ∧
  (10 * A + B) = (110 * A + B) / 8 ∧
  ∀ (A' B' : ℕ),
    A' ≠ B' →
    A' ∈ Finset.range 10 →
    B' ∈ Finset.range 10 →
    A' ≠ 0 →
    (10 * A' + B') = (110 * A' + B') / 8 →
    110 * A + B ≤ 110 * A' + B' ∧
    110 * A + B = 773 :=
by sorry

end NUMINAMATH_CALUDE_smallest_AAB_l1122_112201


namespace NUMINAMATH_CALUDE_ratio_problem_l1122_112255

theorem ratio_problem (a b c : ℝ) 
  (h1 : b / c = 1 / 5)
  (h2 : a / c = 1 / 7.5) :
  a / b = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l1122_112255


namespace NUMINAMATH_CALUDE_equation_solution_l1122_112242

theorem equation_solution : 
  ∃ x : ℝ, (1 / 7 + 7 / x = 15 / x + 1 / 15) ∧ x = 105 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1122_112242


namespace NUMINAMATH_CALUDE_equilateral_triangle_not_unique_from_angles_l1122_112293

/-- An equilateral triangle -/
structure EquilateralTriangle where
  /-- The side length of the equilateral triangle -/
  side : ℝ
  /-- The side length is positive -/
  side_pos : side > 0

/-- The theorem stating that two angles do not uniquely determine an equilateral triangle -/
theorem equilateral_triangle_not_unique_from_angles :
  ∃ (t1 t2 : EquilateralTriangle), t1 ≠ t2 ∧ 
  (∀ (θ : ℝ), 0 < θ ∧ θ < π → 
    (θ = π/3 ↔ (∃ (i : Fin 3), θ = π/3))) :=
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_not_unique_from_angles_l1122_112293


namespace NUMINAMATH_CALUDE_expansion_has_six_nonzero_terms_l1122_112264

/-- The polynomial resulting from expanding (2x^3 - 4)(3x^2 + 5x - 7) + 5 (x^4 - 3x^3 + 2x^2) -/
def expanded_polynomial (x : ℝ) : ℝ :=
  6*x^5 + 15*x^4 - 29*x^3 - 2*x^2 - 20*x + 28

/-- The coefficients of the expanded polynomial -/
def coefficients : List ℝ := [6, 15, -29, -2, -20, 28]

/-- Theorem stating that the expansion has exactly 6 nonzero terms -/
theorem expansion_has_six_nonzero_terms :
  coefficients.length = 6 ∧ coefficients.all (· ≠ 0) := by sorry

end NUMINAMATH_CALUDE_expansion_has_six_nonzero_terms_l1122_112264


namespace NUMINAMATH_CALUDE_geometric_sum_111_l1122_112245

def is_geometric_progression (a b c : ℕ) : Prop :=
  ∃ r : ℚ, r > 0 ∧ b = a * r ∧ c = a * r^2

def valid_triple (a b c : ℕ) : Prop :=
  is_geometric_progression a b c ∧ a + b + c = 111

theorem geometric_sum_111 :
  ∀ a b c : ℕ, valid_triple a b c ↔ 
    ((a, b, c) = (1, 10, 100) ∨ (a, b, c) = (100, 10, 1) ∨ (a, b, c) = (37, 37, 37)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sum_111_l1122_112245


namespace NUMINAMATH_CALUDE_temperature_range_l1122_112205

theorem temperature_range (highest_temp lowest_temp t : ℝ) 
  (h_highest : highest_temp = 30)
  (h_lowest : lowest_temp = 20)
  (h_range : lowest_temp ≤ t ∧ t ≤ highest_temp) :
  20 ≤ t ∧ t ≤ 30 := by
  sorry

end NUMINAMATH_CALUDE_temperature_range_l1122_112205


namespace NUMINAMATH_CALUDE_lost_card_number_l1122_112204

theorem lost_card_number (n : ℕ) (h1 : n > 0) (h2 : (n * (n + 1)) / 2 - 101 ≤ n) : 
  ∃ (x : ℕ), x > 0 ∧ x ≤ n ∧ (n * (n + 1)) / 2 - x = 101 ∧ x = 4 := by
  sorry

#check lost_card_number

end NUMINAMATH_CALUDE_lost_card_number_l1122_112204


namespace NUMINAMATH_CALUDE_gcd_preservation_l1122_112219

theorem gcd_preservation (a b c d x y z G : ℤ) 
  (h : G = Int.gcd a (Int.gcd b (Int.gcd c d))) : 
  G = Int.gcd a (Int.gcd b (Int.gcd c (Int.gcd d (Int.gcd (G*x) (Int.gcd (G*y) (G*z)))))) :=
by sorry

end NUMINAMATH_CALUDE_gcd_preservation_l1122_112219


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_product_l1122_112260

theorem quadratic_roots_sum_product (p q : ℝ) : 
  (∃ x y : ℝ, 3 * x^2 - p * x + q = 0 ∧ 3 * y^2 - p * y + q = 0 ∧ x + y = 9 ∧ x * y = 24) →
  p + q = 99 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_product_l1122_112260


namespace NUMINAMATH_CALUDE_x_cubed_plus_two_x_squared_plus_2007_l1122_112278

theorem x_cubed_plus_two_x_squared_plus_2007 (x : ℝ) (h : x^2 + x - 1 = 0) :
  x^3 + 2*x^2 + 2007 = 2008 := by
  sorry

end NUMINAMATH_CALUDE_x_cubed_plus_two_x_squared_plus_2007_l1122_112278


namespace NUMINAMATH_CALUDE_circle_number_placement_l1122_112240

-- Define the type for circle positions
inductive CirclePosition
  | one | two | three | four | five | six | seven | eight

-- Define the neighborhood relation
def isNeighbor : CirclePosition → CirclePosition → Prop
  | CirclePosition.one, CirclePosition.two => True
  | CirclePosition.one, CirclePosition.four => True
  | CirclePosition.two, CirclePosition.three => True
  | CirclePosition.two, CirclePosition.four => True
  | CirclePosition.two, CirclePosition.six => True
  | CirclePosition.three, CirclePosition.four => True
  | CirclePosition.three, CirclePosition.seven => True
  | CirclePosition.four, CirclePosition.five => True
  | CirclePosition.five, CirclePosition.six => True
  | CirclePosition.six, CirclePosition.seven => True
  | CirclePosition.seven, CirclePosition.eight => True
  | _, _ => False

-- Define the valid numbers
def validNumbers : List Nat := [2, 3, 4, 5, 6, 7, 8, 9]

-- Define a function to check if a number is a divisor of another
def isDivisor (a b : Nat) : Prop := b % a = 0 ∧ a ≠ 1 ∧ a ≠ b

-- Define the main theorem
theorem circle_number_placement :
  ∃ (f : CirclePosition → Nat),
    (∀ p, f p ∈ validNumbers) ∧
    (∀ p₁ p₂, p₁ ≠ p₂ → f p₁ ≠ f p₂) ∧
    (∀ p₁ p₂, isNeighbor p₁ p₂ → ¬isDivisor (f p₁) (f p₂)) := by
  sorry

end NUMINAMATH_CALUDE_circle_number_placement_l1122_112240


namespace NUMINAMATH_CALUDE_simplify_expression_l1122_112267

theorem simplify_expression (a b m : ℝ) (h1 : a + b = m) (h2 : a * b = -4) :
  (a - 2) * (b - 2) = -2 * m := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l1122_112267


namespace NUMINAMATH_CALUDE_certain_number_sum_l1122_112288

theorem certain_number_sum (x : ℤ) : 47 + x = 30 → x = -17 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_sum_l1122_112288


namespace NUMINAMATH_CALUDE_largest_n_for_factorable_quadratic_l1122_112234

/-- A structure representing a quadratic expression ax^2 + bx + c -/
structure Quadratic where
  a : ℤ
  b : ℤ
  c : ℤ

/-- A structure representing a linear factor ax + b -/
structure LinearFactor where
  a : ℤ
  b : ℤ

/-- Function to check if a quadratic can be factored into two linear factors -/
def isFactorable (q : Quadratic) (l1 l2 : LinearFactor) : Prop :=
  q.a = l1.a * l2.a ∧
  q.b = l1.a * l2.b + l1.b * l2.a ∧
  q.c = l1.b * l2.b

/-- The main theorem stating the largest value of n -/
theorem largest_n_for_factorable_quadratic :
  ∃ (n : ℤ),
    n = 451 ∧
    (∀ m : ℤ, m > n → 
      ¬∃ (l1 l2 : LinearFactor), 
        isFactorable ⟨5, m, 90⟩ l1 l2) ∧
    (∃ (l1 l2 : LinearFactor), 
      isFactorable ⟨5, n, 90⟩ l1 l2) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_for_factorable_quadratic_l1122_112234


namespace NUMINAMATH_CALUDE_no_double_application_function_l1122_112230

theorem no_double_application_function : ¬∃ (f : ℕ → ℕ), ∀ n, f (f n) = n + 1 := by
  sorry

end NUMINAMATH_CALUDE_no_double_application_function_l1122_112230


namespace NUMINAMATH_CALUDE_badminton_racket_purchase_l1122_112202

theorem badminton_racket_purchase
  (total_pairs : ℕ)
  (cost_A : ℕ)
  (cost_B : ℕ)
  (total_cost : ℕ)
  (h1 : total_pairs = 30)
  (h2 : cost_A = 50)
  (h3 : cost_B = 40)
  (h4 : total_cost = 1360) :
  ∃ (pairs_A pairs_B : ℕ),
    pairs_A + pairs_B = total_pairs ∧
    pairs_A * cost_A + pairs_B * cost_B = total_cost ∧
    pairs_A = 16 ∧
    pairs_B = 14 := by
  sorry

end NUMINAMATH_CALUDE_badminton_racket_purchase_l1122_112202


namespace NUMINAMATH_CALUDE_faye_candy_eaten_l1122_112238

/-- Represents the number of candy pieces Faye ate on the first night -/
def candy_eaten (initial : ℕ) (received : ℕ) (final : ℕ) : ℕ :=
  initial + received - final

theorem faye_candy_eaten : 
  candy_eaten 47 40 62 = 25 := by
sorry

end NUMINAMATH_CALUDE_faye_candy_eaten_l1122_112238


namespace NUMINAMATH_CALUDE_hyperbola_tangent_dot_product_l1122_112290

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 = 1

/-- The asymptotes of the hyperbola -/
def asymptote (x y : ℝ) : Prop := y = x / 2 ∨ y = -x / 2

/-- A point is on the line l -/
def on_line_l (x y : ℝ) : Prop := sorry

/-- The line l is tangent to the hyperbola at point P -/
def is_tangent (P : ℝ × ℝ) : Prop := 
  hyperbola P.1 P.2 ∧ on_line_l P.1 P.2 ∧ 
  ∀ Q : ℝ × ℝ, Q ≠ P → on_line_l Q.1 Q.2 → ¬hyperbola Q.1 Q.2

theorem hyperbola_tangent_dot_product 
  (P M N : ℝ × ℝ) 
  (h_tangent : is_tangent P) 
  (h_M : on_line_l M.1 M.2 ∧ asymptote M.1 M.2) 
  (h_N : on_line_l N.1 N.2 ∧ asymptote N.1 N.2) :
  M.1 * N.1 + M.2 * N.2 = 3 := 
sorry

end NUMINAMATH_CALUDE_hyperbola_tangent_dot_product_l1122_112290
