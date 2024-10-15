import Mathlib

namespace NUMINAMATH_CALUDE_functional_equation_solution_l2791_279151

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (f x + y) = f (f x - y) + 4 * f x * y

/-- The main theorem stating that any function satisfying the equation
    must be of the form f(x) = x² + C for some constant C -/
theorem functional_equation_solution (f : ℝ → ℝ) (h : SatisfiesEquation f) :
  ∃ C : ℝ, ∀ x : ℝ, f x = x^2 + C := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2791_279151


namespace NUMINAMATH_CALUDE_gcd_75_100_l2791_279167

theorem gcd_75_100 : Nat.gcd 75 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_gcd_75_100_l2791_279167


namespace NUMINAMATH_CALUDE_solution_set_theorem_l2791_279163

-- Define the solution sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def B : Set ℝ := {x | x^2 + x - 6 < 0}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := A ∩ B

-- Theorem statement
theorem solution_set_theorem (a b : ℝ) : 
  ({x : ℝ | x^2 + a*x + b < 0} = A_intersect_B) → a + b = -3 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_theorem_l2791_279163


namespace NUMINAMATH_CALUDE_tiger_enclosure_optimizations_l2791_279182

/-- Represents a rectangular tiger enclosure -/
structure TigerEnclosure where
  length : ℝ
  width : ℝ

/-- Calculates the area of a tiger enclosure -/
def area (e : TigerEnclosure) : ℝ := e.length * e.width

/-- Calculates the wire mesh length needed for a tiger enclosure -/
def wireMeshLength (e : TigerEnclosure) : ℝ := e.length + 2 * e.width

/-- The total available wire mesh length -/
def totalWireMesh : ℝ := 36

/-- The fixed area for part 2 of the problem -/
def fixedArea : ℝ := 32

theorem tiger_enclosure_optimizations :
  (∃ (e : TigerEnclosure),
    wireMeshLength e = totalWireMesh ∧
    area e = 162 ∧
    e.length = 18 ∧
    e.width = 9 ∧
    (∀ (e' : TigerEnclosure), wireMeshLength e' ≤ totalWireMesh → area e' ≤ area e)) ∧
  (∃ (e : TigerEnclosure),
    area e = fixedArea ∧
    wireMeshLength e = 16 ∧
    e.length = 8 ∧
    e.width = 4 ∧
    (∀ (e' : TigerEnclosure), area e' = fixedArea → wireMeshLength e' ≥ wireMeshLength e)) :=
by sorry

end NUMINAMATH_CALUDE_tiger_enclosure_optimizations_l2791_279182


namespace NUMINAMATH_CALUDE_mans_age_twice_sons_l2791_279165

/-- 
Proves that the number of years it takes for a man's age to be twice his son's age is 2,
given the initial conditions.
-/
theorem mans_age_twice_sons (
  son_age : ℕ) -- Present age of the son
  (age_diff : ℕ) -- Age difference between man and son
  (h1 : son_age = 27) -- The son's present age is 27
  (h2 : age_diff = 29) -- The man is 29 years older than his son
  : ∃ (years : ℕ), years = 2 ∧ (son_age + years + age_diff = 2 * (son_age + years)) :=
by sorry

end NUMINAMATH_CALUDE_mans_age_twice_sons_l2791_279165


namespace NUMINAMATH_CALUDE_lamplighter_monkey_distance_l2791_279159

/-- Calculates the total distance traveled by a Lamplighter monkey under specific conditions. -/
theorem lamplighter_monkey_distance (initial_swing_speed initial_run_speed : ℝ)
  (wind_resistance_factor branch_weight_factor : ℝ)
  (run_time swing_time : ℝ) :
  initial_swing_speed = 10 →
  initial_run_speed = 15 →
  wind_resistance_factor = 0.9 →
  branch_weight_factor = 1.05 →
  run_time = 5 →
  swing_time = 10 →
  let adjusted_swing_speed := initial_swing_speed * wind_resistance_factor
  let adjusted_run_speed := initial_run_speed * branch_weight_factor
  let total_distance := adjusted_run_speed * run_time + adjusted_swing_speed * swing_time
  total_distance = 168.75 := by
sorry


end NUMINAMATH_CALUDE_lamplighter_monkey_distance_l2791_279159


namespace NUMINAMATH_CALUDE_circle_radius_is_three_l2791_279153

/-- Given a circle where the product of three inches and its circumference
    equals twice its area, prove that its radius is 3 inches. -/
theorem circle_radius_is_three (r : ℝ) (h : 3 * (2 * π * r) = 2 * (π * r^2)) : r = 3 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_is_three_l2791_279153


namespace NUMINAMATH_CALUDE_task_completion_time_l2791_279161

/-- The number of days needed for three people to complete the task -/
def three_people_days : ℕ := 3 * 7 + 3

/-- The number of people in the original scenario -/
def original_people : ℕ := 3

/-- The number of people in the new scenario -/
def new_people : ℕ := 4

/-- The time needed for four people to complete the task -/
def four_people_days : ℚ := 18

theorem task_completion_time :
  (three_people_days : ℚ) * original_people / new_people = four_people_days :=
sorry

end NUMINAMATH_CALUDE_task_completion_time_l2791_279161


namespace NUMINAMATH_CALUDE_total_toys_cost_l2791_279117

def toy_cars_cost : ℚ := 14.88
def skateboard_cost : ℚ := 4.88
def toy_trucks_cost : ℚ := 5.86
def pants_cost : ℚ := 14.55

theorem total_toys_cost :
  toy_cars_cost + skateboard_cost + toy_trucks_cost = 25.62 := by sorry

end NUMINAMATH_CALUDE_total_toys_cost_l2791_279117


namespace NUMINAMATH_CALUDE_range_of_a_minus_abs_b_l2791_279147

theorem range_of_a_minus_abs_b (a b : ℝ) (ha : 1 < a ∧ a < 3) (hb : -4 < b ∧ b < 2) :
  ∀ x : ℝ, x = a - |b| → -3 < x ∧ x < 3 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_minus_abs_b_l2791_279147


namespace NUMINAMATH_CALUDE_ball_probabilities_l2791_279106

/-- Given a bag of balls with the following properties:
  - There are 10 balls in total.
  - The probability of drawing a black ball is 2/5.
  - The probability of drawing at least one white ball when drawing two balls is 19/20.

  This theorem proves:
  1. The probability of drawing two black balls is 6/45.
  2. The number of white balls is 5.
-/
theorem ball_probabilities
  (total_balls : ℕ)
  (prob_black : ℚ)
  (prob_at_least_one_white : ℚ)
  (h_total : total_balls = 10)
  (h_prob_black : prob_black = 2 / 5)
  (h_prob_white : prob_at_least_one_white = 19 / 20) :
  (∃ (black_balls white_balls : ℕ),
    black_balls + white_balls ≤ total_balls ∧
    (black_balls : ℚ) / total_balls = prob_black ∧
    1 - (total_balls - white_balls) * (total_balls - white_balls - 1) / (total_balls * (total_balls - 1)) = prob_at_least_one_white ∧
    black_balls * (black_balls - 1) / (total_balls * (total_balls - 1)) = 6 / 45 ∧
    white_balls = 5) :=
by sorry

end NUMINAMATH_CALUDE_ball_probabilities_l2791_279106


namespace NUMINAMATH_CALUDE_pony_discount_rate_l2791_279140

/-- Represents the discount rates for Fox and Pony jeans -/
structure DiscountRates where
  fox : ℝ
  pony : ℝ

/-- The problem setup -/
def jeans_problem (d : DiscountRates) : Prop :=
  -- Regular prices
  let fox_price : ℝ := 15
  let pony_price : ℝ := 18
  -- Total savings condition
  3 * fox_price * (d.fox / 100) + 2 * pony_price * (d.pony / 100) = 9 ∧
  -- Sum of discount rates condition
  d.fox + d.pony = 25

/-- The theorem to prove -/
theorem pony_discount_rate : 
  ∃ (d : DiscountRates), jeans_problem d ∧ d.pony = 25 := by
  sorry

end NUMINAMATH_CALUDE_pony_discount_rate_l2791_279140


namespace NUMINAMATH_CALUDE_geometric_sequence_general_term_l2791_279134

/-- A geometric sequence {a_n} satisfying the given conditions -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n) ∧
  a 6 = 8 * a 3 ∧
  a 6 = 8 * (a 2)^2

/-- The theorem stating the general term of the geometric sequence -/
theorem geometric_sequence_general_term {a : ℕ → ℝ} (h : geometric_sequence a) :
  ∀ n : ℕ, a n = 2^(n - 1) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_general_term_l2791_279134


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2791_279137

theorem inequality_solution_set (x : ℝ) :
  x ≠ 0 →
  ((2 * x - 5) * (x - 3)) / x ≥ 0 ↔ (0 < x ∧ x ≤ 5/2) ∨ (x ≥ 3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2791_279137


namespace NUMINAMATH_CALUDE_unique_function_is_identity_l2791_279185

/-- A function from positive integers to positive integers -/
def PositiveIntFunction := ℕ+ → ℕ+

/-- The property that f(mn) = f(m)f(n) for all positive integers m and n -/
def IsMultiplicative (f : PositiveIntFunction) : Prop :=
  ∀ m n : ℕ+, f (m * n) = f m * f n

/-- The property that f^(n^k)(n) = n for all positive integers n -/
def SatisfiesExpProperty (f : PositiveIntFunction) (k : ℕ+) : Prop :=
  ∀ n : ℕ+, (f^[n^k.val]) n = n

/-- The identity function on positive integers -/
def identityFunction : PositiveIntFunction := id

theorem unique_function_is_identity (k : ℕ+) :
  ∃! f : PositiveIntFunction, IsMultiplicative f ∧ SatisfiesExpProperty f k →
  f = identityFunction :=
sorry

end NUMINAMATH_CALUDE_unique_function_is_identity_l2791_279185


namespace NUMINAMATH_CALUDE_value_added_to_half_l2791_279101

theorem value_added_to_half : ∃ (v : ℝ), (20 / 2) + v = 17 ∧ v = 7 := by sorry

end NUMINAMATH_CALUDE_value_added_to_half_l2791_279101


namespace NUMINAMATH_CALUDE_gcf_seven_eight_factorial_l2791_279145

theorem gcf_seven_eight_factorial : Nat.gcd (Nat.factorial 7) (Nat.factorial 8) = Nat.factorial 7 := by
  sorry

end NUMINAMATH_CALUDE_gcf_seven_eight_factorial_l2791_279145


namespace NUMINAMATH_CALUDE_symmetry_composition_l2791_279168

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define symmetry operations
def symmetryXAxis (p : Point2D) : Point2D :=
  { x := p.x, y := -p.y }

def symmetryYAxis (p : Point2D) : Point2D :=
  { x := -p.x, y := p.y }

-- Theorem statement
theorem symmetry_composition (a b : ℝ) :
  let M : Point2D := { x := a, y := b }
  let N : Point2D := symmetryXAxis M
  let P : Point2D := symmetryYAxis N
  let Q : Point2D := symmetryXAxis P
  let R : Point2D := symmetryYAxis Q
  R = M := by sorry

end NUMINAMATH_CALUDE_symmetry_composition_l2791_279168


namespace NUMINAMATH_CALUDE_min_value_theorem_l2791_279100

theorem min_value_theorem (a : ℝ) (h : 8 * a^2 + 6 * a + 2 = 4) :
  ∃ (m : ℝ), (3 * a + 1 ≥ m) ∧ (∀ x, 8 * x^2 + 6 * x + 2 = 4 → 3 * x + 1 ≥ m) ∧ (m = -2) :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2791_279100


namespace NUMINAMATH_CALUDE_calculate_expression_l2791_279198

-- Define the variables and their relationships
def x : ℝ := 70 * (1 + 0.11)
def y : ℝ := x * (1 + 0.15)
def z : ℝ := y * (1 - 0.20)

-- State the theorem
theorem calculate_expression : 3 * z - 2 * x + y = 148.407 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2791_279198


namespace NUMINAMATH_CALUDE_ellipse_properties_max_radius_l2791_279183

/-- The ellipse C with foci F₁(-c, 0) and F₂(c, 0), and upper vertex M satisfying F₁M ⋅ F₂M = 0 -/
structure Ellipse (a b c : ℝ) :=
  (a_pos : a > 0)
  (b_pos : b > 0)
  (a_gt_b : a > b)
  (foci_condition : c^2 = a^2 - b^2)
  (vertex_condition : -c^2 + b^2 = 0)

/-- The point N(0, 2) is the center of a circle intersecting the ellipse C -/
def N : ℝ × ℝ := (0, 2)

/-- The theorem stating properties of the ellipse C -/
theorem ellipse_properties (a b c : ℝ) (C : Ellipse a b c) :
  -- The eccentricity of C is √2/2
  (c / a = Real.sqrt 2 / 2) ∧
  -- The equation of C is x²/18 + y²/9 = 1
  (∀ x y : ℝ, x^2 / 18 + y^2 / 9 = 1 ↔ x^2 / a^2 + y^2 / b^2 = 1) ∧
  -- The range of k for symmetric points A and B on C w.r.t. y = kx - 1
  (∀ k : ℝ, (k < -1/2 ∨ k > 1/2) ↔
    ∃ A B : ℝ × ℝ,
      (A.1^2 / 18 + A.2^2 / 9 = 1) ∧
      (B.1^2 / 18 + B.2^2 / 9 = 1) ∧
      (A.2 = k * A.1 - 1) ∧
      (B.2 = k * B.1 - 1) ∧
      (A ≠ B)) :=
sorry

/-- The maximum radius of the circle centered at N intersecting C is √26 -/
theorem max_radius (a b c : ℝ) (C : Ellipse a b c) :
  ∀ P : ℝ × ℝ, P.1^2 / a^2 + P.2^2 / b^2 = 1 →
    (P.1 - N.1)^2 + (P.2 - N.2)^2 ≤ 26 :=
sorry

end NUMINAMATH_CALUDE_ellipse_properties_max_radius_l2791_279183


namespace NUMINAMATH_CALUDE_nick_hid_ten_chocolates_l2791_279157

/-- The number of chocolates Nick hid -/
def nick_chocolates : ℕ := sorry

/-- The number of chocolates Alix hid initially -/
def alix_initial_chocolates : ℕ := 3 * nick_chocolates

/-- The number of chocolates Alix has after mom took 5 -/
def alix_current_chocolates : ℕ := alix_initial_chocolates - 5

theorem nick_hid_ten_chocolates : 
  alix_current_chocolates = nick_chocolates + 15 → nick_chocolates = 10 := by
  sorry

end NUMINAMATH_CALUDE_nick_hid_ten_chocolates_l2791_279157


namespace NUMINAMATH_CALUDE_paula_twice_karl_age_l2791_279158

/-- Represents the ages of Paula and Karl -/
structure Ages where
  paula : ℕ
  karl : ℕ

/-- The conditions of the problem -/
def problem_conditions (ages : Ages) : Prop :=
  ages.paula + ages.karl = 50 ∧
  ages.paula - 7 = 3 * (ages.karl - 7)

/-- The theorem to prove -/
theorem paula_twice_karl_age (ages : Ages) :
  problem_conditions ages →
  ∃ x : ℕ, x = 2 ∧ ages.paula + x = 2 * (ages.karl + x) :=
sorry

end NUMINAMATH_CALUDE_paula_twice_karl_age_l2791_279158


namespace NUMINAMATH_CALUDE_fraction_modification_l2791_279132

theorem fraction_modification (a b c d x y : ℚ) 
  (h1 : a ≠ b) 
  (h2 : b ≠ 0) 
  (h3 : (a + x) / (b + y) = c / d) : 
  x = (b * c - a * d + y * c) / d := by
  sorry

end NUMINAMATH_CALUDE_fraction_modification_l2791_279132


namespace NUMINAMATH_CALUDE_hyperbola_sum_l2791_279124

theorem hyperbola_sum (h k a b c : ℝ) : 
  (h = -2) →
  (k = 0) →
  (c = Real.sqrt 34) →
  (a = 3) →
  (c^2 = a^2 + b^2) →
  (h + k + a + b = 6) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_sum_l2791_279124


namespace NUMINAMATH_CALUDE_positive_number_problem_l2791_279172

theorem positive_number_problem (n : ℝ) : n > 0 ∧ 5 * (n^2 + n) = 780 → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_positive_number_problem_l2791_279172


namespace NUMINAMATH_CALUDE_range_of_a_l2791_279129

theorem range_of_a (z : ℂ) (a : ℝ) : 
  z.im ≠ 0 →  -- z is imaginary
  (z + 3 / (2 * z)).im = 0 →  -- z + 3/(2z) is real
  (z + 3 / (2 * z))^2 - 2 * a * (z + 3 / (2 * z)) + 1 - 3 * a = 0 →  -- root condition
  (a ≥ (Real.sqrt 13 - 3) / 2 ∨ a ≤ -(Real.sqrt 13 + 3) / 2) := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l2791_279129


namespace NUMINAMATH_CALUDE_manuscript_pages_l2791_279179

/-- Represents the typing service cost structure and manuscript details -/
structure ManuscriptTyping where
  first_time_cost : ℕ
  revision_cost : ℕ
  pages_revised_once : ℕ
  pages_revised_twice : ℕ
  total_cost : ℕ

/-- Calculates the total number of pages in the manuscript -/
def total_pages (mt : ManuscriptTyping) : ℕ :=
  sorry

/-- Theorem stating that the total number of pages is 100 -/
theorem manuscript_pages (mt : ManuscriptTyping) 
  (h1 : mt.first_time_cost = 5)
  (h2 : mt.revision_cost = 3)
  (h3 : mt.pages_revised_once = 30)
  (h4 : mt.pages_revised_twice = 20)
  (h5 : mt.total_cost = 710) :
  total_pages mt = 100 := by
  sorry

end NUMINAMATH_CALUDE_manuscript_pages_l2791_279179


namespace NUMINAMATH_CALUDE_cube_divisibility_l2791_279138

theorem cube_divisibility (n : ℕ) (h : ∀ k : ℕ, k > 0 → k < 42 → ¬(n ∣ k^3)) : n = 74088 := by
  sorry

end NUMINAMATH_CALUDE_cube_divisibility_l2791_279138


namespace NUMINAMATH_CALUDE_inequality_proof_l2791_279121

theorem inequality_proof (a : ℝ) : (a^2 + a + 2) / Real.sqrt (a^2 + a + 1) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2791_279121


namespace NUMINAMATH_CALUDE_mice_eaten_in_decade_l2791_279143

/-- Calculates the number of mice eaten by a snake in a decade -/
theorem mice_eaten_in_decade (weeks_per_mouse : ℕ) (years_per_decade : ℕ) (weeks_per_year : ℕ) : 
  weeks_per_mouse = 4 → years_per_decade = 10 → weeks_per_year = 52 →
  (years_per_decade * weeks_per_year) / weeks_per_mouse = 130 := by
sorry

end NUMINAMATH_CALUDE_mice_eaten_in_decade_l2791_279143


namespace NUMINAMATH_CALUDE_min_distance_sum_l2791_279111

theorem min_distance_sum (x y : ℝ) :
  Real.sqrt ((x - 1)^2 + y^2) + Real.sqrt ((x + 1)^2 + y^2) + |2 - y| ≥ 2 + Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_sum_l2791_279111


namespace NUMINAMATH_CALUDE_sum_of_features_l2791_279146

/-- A pentagonal prism with a pyramid added to one pentagonal face -/
structure PentagonalPrismWithPyramid where
  /-- Number of faces of the original pentagonal prism -/
  prism_faces : ℕ
  /-- Number of vertices of the original pentagonal prism -/
  prism_vertices : ℕ
  /-- Number of edges of the original pentagonal prism -/
  prism_edges : ℕ
  /-- Number of faces added by the pyramid -/
  pyramid_faces : ℕ
  /-- Number of vertices added by the pyramid -/
  pyramid_vertices : ℕ
  /-- Number of edges added by the pyramid -/
  pyramid_edges : ℕ
  /-- The pentagonal prism has 7 faces -/
  prism_faces_eq : prism_faces = 7
  /-- The pentagonal prism has 10 vertices -/
  prism_vertices_eq : prism_vertices = 10
  /-- The pentagonal prism has 15 edges -/
  prism_edges_eq : prism_edges = 15
  /-- The pyramid adds 5 faces -/
  pyramid_faces_eq : pyramid_faces = 5
  /-- The pyramid adds 1 vertex -/
  pyramid_vertices_eq : pyramid_vertices = 1
  /-- The pyramid adds 5 edges -/
  pyramid_edges_eq : pyramid_edges = 5

/-- The sum of exterior faces, vertices, and edges of the resulting shape is 42 -/
theorem sum_of_features (shape : PentagonalPrismWithPyramid) :
  (shape.prism_faces + shape.pyramid_faces - 1) +
  (shape.prism_vertices + shape.pyramid_vertices) +
  (shape.prism_edges + shape.pyramid_edges) = 42 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_features_l2791_279146


namespace NUMINAMATH_CALUDE_tourist_growth_and_max_l2791_279141

def tourists_feb : ℕ := 16000
def tourists_apr : ℕ := 25000
def tourists_may_21 : ℕ := 21250

def monthly_growth_rate : ℝ := 0.25

def max_daily_tourists_last_10_days : ℝ := 100000

theorem tourist_growth_and_max (growth_rate : ℝ) (max_daily : ℝ) :
  growth_rate = monthly_growth_rate ∧
  max_daily = max_daily_tourists_last_10_days ∧
  tourists_feb * (1 + growth_rate)^2 = tourists_apr ∧
  tourists_may_21 + 10 * max_daily ≤ tourists_apr * (1 + growth_rate) :=
by sorry

end NUMINAMATH_CALUDE_tourist_growth_and_max_l2791_279141


namespace NUMINAMATH_CALUDE_pipe_length_theorem_l2791_279184

theorem pipe_length_theorem (shorter_piece longer_piece total_length : ℝ) :
  longer_piece = 2 * shorter_piece →
  longer_piece = 118 →
  total_length = shorter_piece + longer_piece →
  total_length = 177 := by
  sorry

end NUMINAMATH_CALUDE_pipe_length_theorem_l2791_279184


namespace NUMINAMATH_CALUDE_angle_between_skew_lines_range_l2791_279120

-- Define skew lines
structure SkewLine where
  -- We don't need to define the internal structure of a skew line for this problem

-- Define the angle between two skew lines
def angle_between_skew_lines (a b : SkewLine) : ℝ :=
  sorry -- The actual implementation is not needed for the statement

-- Theorem statement
theorem angle_between_skew_lines_range (a b : SkewLine) :
  let θ := angle_between_skew_lines a b
  0 < θ ∧ θ ≤ π/2 :=
sorry

end NUMINAMATH_CALUDE_angle_between_skew_lines_range_l2791_279120


namespace NUMINAMATH_CALUDE_triangle_side_calculation_l2791_279103

theorem triangle_side_calculation (a b c : ℝ) (A B C : ℝ) : 
  a = 10 → B = 2 * π / 3 → C = π / 6 → 
  A + B + C = π → 
  a / Real.sin A = b / Real.sin B → 
  b = 10 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_calculation_l2791_279103


namespace NUMINAMATH_CALUDE_probability_of_nine_in_three_elevenths_l2791_279112

def decimal_representation (n d : ℕ) : List ℕ := sorry

def contains_digit (l : List ℕ) (digit : ℕ) : Bool := sorry

def probability_of_digit (n d digit : ℕ) : ℚ := sorry

theorem probability_of_nine_in_three_elevenths :
  probability_of_digit 3 11 9 = 0 := by sorry

end NUMINAMATH_CALUDE_probability_of_nine_in_three_elevenths_l2791_279112


namespace NUMINAMATH_CALUDE_fruit_salad_mixture_weight_l2791_279116

theorem fruit_salad_mixture_weight 
  (apple peach grape : ℝ) 
  (h1 : apple / grape = 12 / 7)
  (h2 : peach / grape = 8 / 7)
  (h3 : apple = grape + 10) :
  apple + peach + grape = 54 := by
sorry

end NUMINAMATH_CALUDE_fruit_salad_mixture_weight_l2791_279116


namespace NUMINAMATH_CALUDE_no_real_roots_of_composition_l2791_279126

/-- A quadratic function f(x) = ax^2 + bx + c where a ≠ 0 -/
def f (a b c : ℝ) (ha : a ≠ 0) : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c

/-- Theorem: If f(x) = x has no real roots, then f(f(x)) = x has no real roots -/
theorem no_real_roots_of_composition
  (a b c : ℝ) (ha : a ≠ 0)
  (h_no_roots : ∀ x : ℝ, f a b c ha x ≠ x) :
  ∀ x : ℝ, f a b c ha (f a b c ha x) ≠ x :=
sorry

end NUMINAMATH_CALUDE_no_real_roots_of_composition_l2791_279126


namespace NUMINAMATH_CALUDE_smallest_n_with_conditions_l2791_279102

def is_sum_of_identical_digits (n : ℕ) (count : ℕ) : Prop :=
  ∃ (d : ℕ), d ≤ 9 ∧ n = count * d

theorem smallest_n_with_conditions : 
  let n := 6036
  (n > 0) ∧ 
  (n % 2010 = 0) ∧ 
  (n % 2012 = 0) ∧ 
  (n % 2013 = 0) ∧
  (is_sum_of_identical_digits n 2010) ∧
  (is_sum_of_identical_digits n 2012) ∧
  (is_sum_of_identical_digits n 2013) ∧
  (∀ m : ℕ, m > 0 ∧ 
            m % 2010 = 0 ∧ 
            m % 2012 = 0 ∧ 
            m % 2013 = 0 ∧
            is_sum_of_identical_digits m 2010 ∧
            is_sum_of_identical_digits m 2012 ∧
            is_sum_of_identical_digits m 2013 
            → m ≥ n) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_with_conditions_l2791_279102


namespace NUMINAMATH_CALUDE_expression_value_l2791_279152

theorem expression_value : ((2^2 - 3*2 - 10) / (2 - 5)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2791_279152


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l2791_279142

theorem geometric_sequence_first_term
  (a : ℕ → ℝ)  -- The sequence
  (h_geom : ∀ n, a (n + 1) = a n * (a 1 / a 0))  -- Geometric sequence condition
  (h_4th : a 3 = 81)  -- Fourth term is 81 (index starts at 0)
  (h_5th : a 4 = 162)  -- Fifth term is 162
  : a 0 = 10.125 :=  -- First term is 10.125
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l2791_279142


namespace NUMINAMATH_CALUDE_value_of_expression_l2791_279199

theorem value_of_expression (x : ℝ) (h : 3 * x + 2 = 11) : 6 * x + 4 = 22 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l2791_279199


namespace NUMINAMATH_CALUDE_midpoint_sum_equals_vertex_sum_l2791_279130

theorem midpoint_sum_equals_vertex_sum (a b c d : ℝ) :
  a + b + c + d = 15 →
  (a + b) / 2 + (b + c) / 2 + (c + d) / 2 + (d + a) / 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_sum_equals_vertex_sum_l2791_279130


namespace NUMINAMATH_CALUDE_shirts_made_over_two_days_l2791_279177

/-- Calculates the total number of shirts made by an industrial machine over two days -/
theorem shirts_made_over_two_days 
  (shirts_per_minute : ℕ) -- Number of shirts the machine can make per minute
  (minutes_worked_yesterday : ℕ) -- Number of minutes the machine worked yesterday
  (shirts_made_today : ℕ) -- Number of shirts made today
  (h1 : shirts_per_minute = 6)
  (h2 : minutes_worked_yesterday = 12)
  (h3 : shirts_made_today = 14) :
  shirts_per_minute * minutes_worked_yesterday + shirts_made_today = 86 :=
by
  sorry

#check shirts_made_over_two_days

end NUMINAMATH_CALUDE_shirts_made_over_two_days_l2791_279177


namespace NUMINAMATH_CALUDE_percentage_equation_solution_l2791_279105

theorem percentage_equation_solution :
  ∃ x : ℝ, (65 / 100) * x = (20 / 100) * 682.50 ∧ x = 210 := by
  sorry

end NUMINAMATH_CALUDE_percentage_equation_solution_l2791_279105


namespace NUMINAMATH_CALUDE_largest_integer_four_digits_base_seven_l2791_279174

def has_four_digits_base_seven (n : ℕ) : Prop :=
  7^3 ≤ n^2 ∧ n^2 < 7^4

theorem largest_integer_four_digits_base_seven :
  ∃ M : ℕ, has_four_digits_base_seven M ∧
    ∀ n : ℕ, has_four_digits_base_seven n → n ≤ M ∧
    M = 48 :=
sorry

end NUMINAMATH_CALUDE_largest_integer_four_digits_base_seven_l2791_279174


namespace NUMINAMATH_CALUDE_max_water_depth_l2791_279115

/-- The maximum water depth during a swim, given the swimmer's height,
    the ratio of water depth to height, and the wave increase percentage. -/
theorem max_water_depth
  (height : ℝ)
  (depth_ratio : ℝ)
  (wave_increase : ℝ)
  (h1 : height = 6)
  (h2 : depth_ratio = 10)
  (h3 : wave_increase = 0.25)
  : height * depth_ratio * (1 + wave_increase) = 75 := by
  sorry

#check max_water_depth

end NUMINAMATH_CALUDE_max_water_depth_l2791_279115


namespace NUMINAMATH_CALUDE_sum_inequality_l2791_279150

theorem sum_inequality (a b x y : ℝ) (ha : 0 < a) (hb : 0 < b) (hx : 0 < x) (hy : 0 < y)
  (hax : a * x ≤ 5) (hay : a * y ≤ 10) (hbx : b * x ≤ 10) (hby : b * y ≤ 10) :
  a * x + a * y + b * x + b * y ≤ 30 := by
  sorry

end NUMINAMATH_CALUDE_sum_inequality_l2791_279150


namespace NUMINAMATH_CALUDE_tangent_parallel_points_l2791_279154

/-- The function f(x) = x^3 - x + 3 -/
def f (x : ℝ) : ℝ := x^3 - x + 3

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3 * x^2 - 1

theorem tangent_parallel_points :
  ∀ x : ℝ, (f' x = 2) ↔ (x = 1 ∨ x = -1) := by sorry

end NUMINAMATH_CALUDE_tangent_parallel_points_l2791_279154


namespace NUMINAMATH_CALUDE_problem_statement_l2791_279125

theorem problem_statement (x y z : ℝ) 
  (h1 : x^2 + 1/x^2 = 7)
  (h2 : x*y = 1)
  (h3 : z^2 + 1/z^2 = 9) :
  x^4 + y^4 - z^4 = 15 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2791_279125


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l2791_279118

theorem decimal_to_fraction : (2.375 : ℚ) = 19 / 8 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l2791_279118


namespace NUMINAMATH_CALUDE_consecutive_integers_sqrt_13_l2791_279148

theorem consecutive_integers_sqrt_13 (m n : ℤ) : 
  (n = m + 1) → (m < Real.sqrt 13) → (Real.sqrt 13 < n) → m * n = 12 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sqrt_13_l2791_279148


namespace NUMINAMATH_CALUDE_roots_sum_squares_l2791_279196

theorem roots_sum_squares (p q r : ℝ) : 
  (p^3 - 24*p^2 + 50*p - 35 = 0) →
  (q^3 - 24*q^2 + 50*q - 35 = 0) →
  (r^3 - 24*r^2 + 50*r - 35 = 0) →
  (p+q)^2 + (q+r)^2 + (r+p)^2 = 1052 := by
sorry

end NUMINAMATH_CALUDE_roots_sum_squares_l2791_279196


namespace NUMINAMATH_CALUDE_apples_handed_out_correct_l2791_279108

/-- Represents the cafeteria's apple distribution problem -/
def apples_handed_out (initial_apples : ℕ) (num_pies : ℕ) (apples_per_pie : ℕ) : ℕ :=
  initial_apples - (num_pies * apples_per_pie)

/-- Proves that the number of apples handed out is correct -/
theorem apples_handed_out_correct (initial_apples : ℕ) (num_pies : ℕ) (apples_per_pie : ℕ) :
  apples_handed_out initial_apples num_pies apples_per_pie =
  initial_apples - (num_pies * apples_per_pie) :=
by
  sorry

#eval apples_handed_out 47 5 4

end NUMINAMATH_CALUDE_apples_handed_out_correct_l2791_279108


namespace NUMINAMATH_CALUDE_arccos_one_half_l2791_279128

theorem arccos_one_half : Real.arccos (1/2) = π/3 := by
  sorry

end NUMINAMATH_CALUDE_arccos_one_half_l2791_279128


namespace NUMINAMATH_CALUDE_coaches_schedule_lcm_l2791_279119

theorem coaches_schedule_lcm : Nat.lcm 5 (Nat.lcm 3 (Nat.lcm 9 8)) = 360 := by
  sorry

end NUMINAMATH_CALUDE_coaches_schedule_lcm_l2791_279119


namespace NUMINAMATH_CALUDE_largest_multiple_of_15_under_500_l2791_279107

theorem largest_multiple_of_15_under_500 : 
  ∀ n : ℕ, n * 15 < 500 → n * 15 ≤ 495 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_15_under_500_l2791_279107


namespace NUMINAMATH_CALUDE_power_of_three_mod_eight_l2791_279194

theorem power_of_three_mod_eight : 3^1988 ≡ 1 [MOD 8] := by sorry

end NUMINAMATH_CALUDE_power_of_three_mod_eight_l2791_279194


namespace NUMINAMATH_CALUDE_edward_games_boxes_l2791_279139

def number_of_boxes (initial_games : ℕ) (sold_games : ℕ) (games_per_box : ℕ) : ℕ :=
  (initial_games - sold_games) / games_per_box

theorem edward_games_boxes :
  number_of_boxes 35 19 8 = 2 := by
  sorry

end NUMINAMATH_CALUDE_edward_games_boxes_l2791_279139


namespace NUMINAMATH_CALUDE_oxygen_atom_diameter_scientific_notation_l2791_279169

theorem oxygen_atom_diameter_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 0.000000000148 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ n = -10 := by
  sorry

end NUMINAMATH_CALUDE_oxygen_atom_diameter_scientific_notation_l2791_279169


namespace NUMINAMATH_CALUDE_division_and_multiplication_l2791_279193

theorem division_and_multiplication (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (result : ℕ) : 
  dividend = 24 → 
  divisor = 3 → 
  dividend = divisor * quotient → 
  result = quotient * 5 → 
  quotient = 8 ∧ result = 40 := by
sorry

end NUMINAMATH_CALUDE_division_and_multiplication_l2791_279193


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2791_279123

theorem complex_equation_solution (z : ℂ) : z / (1 - 2*I) = I → z = 2 + I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2791_279123


namespace NUMINAMATH_CALUDE_product_of_difference_and_sum_of_squares_l2791_279195

theorem product_of_difference_and_sum_of_squares (a b : ℝ) 
  (h1 : a - b = 3) 
  (h2 : a^2 + b^2 = 29) : 
  a * b = 10 := by
sorry

end NUMINAMATH_CALUDE_product_of_difference_and_sum_of_squares_l2791_279195


namespace NUMINAMATH_CALUDE_initial_number_proof_l2791_279144

theorem initial_number_proof : ∃ n : ℕ, n ≥ 102 ∧ (n - 5) % 97 = 0 ∧ ∀ m : ℕ, m < n → (m - 5) % 97 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_initial_number_proof_l2791_279144


namespace NUMINAMATH_CALUDE_baby_guppies_count_is_36_l2791_279136

/-- The number of baby guppies Amber saw several days after buying 7 guppies,
    given that she later saw 9 more baby guppies and now has 52 guppies in total. -/
def baby_guppies_count : ℕ := by sorry

/-- The initial number of guppies Amber bought. -/
def initial_guppies : ℕ := 7

/-- The number of additional baby guppies Amber saw two days after the first group. -/
def additional_baby_guppies : ℕ := 9

/-- The total number of guppies Amber has now. -/
def total_guppies : ℕ := 52

theorem baby_guppies_count_is_36 :
  baby_guppies_count = 36 ∧
  initial_guppies + baby_guppies_count + additional_baby_guppies = total_guppies := by
  sorry

end NUMINAMATH_CALUDE_baby_guppies_count_is_36_l2791_279136


namespace NUMINAMATH_CALUDE_abs_sum_inequality_iff_range_l2791_279187

theorem abs_sum_inequality_iff_range (x : ℝ) : 
  (abs (x + 1) + abs (x - 2) ≤ 5) ↔ (-2 ≤ x ∧ x ≤ 3) := by sorry

end NUMINAMATH_CALUDE_abs_sum_inequality_iff_range_l2791_279187


namespace NUMINAMATH_CALUDE_range_of_m_l2791_279133

theorem range_of_m (x y m : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (hxy : x + 2*y - x*y = 0) 
  (h_ineq : ∀ m : ℝ, x + 2*y > m^2 + 2*m) : 
  -4 < m ∧ m < 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l2791_279133


namespace NUMINAMATH_CALUDE_problem_statement_l2791_279104

theorem problem_statement (x y : ℝ) (h1 : x + y = 3) (h2 : x * y = 5) :
  x^2 * y + x * y^2 = 15 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2791_279104


namespace NUMINAMATH_CALUDE_alternative_bases_l2791_279189

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem alternative_bases
  (a b c : V)
  (h : LinearIndependent ℝ ![a, b, c])
  (hspan : Submodule.span ℝ {a, b, c} = ⊤) :
  LinearIndependent ℝ ![a + b, a + c, a] ∧
  Submodule.span ℝ {a + b, a + c, a} = ⊤ ∧
  LinearIndependent ℝ ![a - b + c, a - b, a + c] ∧
  Submodule.span ℝ {a - b + c, a - b, a + c} = ⊤ := by
sorry

end NUMINAMATH_CALUDE_alternative_bases_l2791_279189


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2791_279188

theorem inequality_system_solution (m : ℝ) : 
  (∀ x : ℝ, (x + 6 < 4*x - 3 ∧ x > m) ↔ x > 3) → m ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2791_279188


namespace NUMINAMATH_CALUDE_sin_shift_l2791_279122

theorem sin_shift (x : ℝ) :
  let f (x : ℝ) := Real.sin (4 * x)
  let g (x : ℝ) := f (x + π / 12)
  let h (x : ℝ) := Real.sin (4 * x + π / 3)
  g = h :=
by sorry

end NUMINAMATH_CALUDE_sin_shift_l2791_279122


namespace NUMINAMATH_CALUDE_unused_types_count_l2791_279113

/-- The number of natural resources --/
def num_resources : ℕ := 6

/-- The number of developed types of nature use --/
def developed_types : ℕ := 23

/-- The number of unused types of nature use --/
def unused_types : ℕ := 2^num_resources - 1 - developed_types

theorem unused_types_count : unused_types = 40 := by
  sorry

end NUMINAMATH_CALUDE_unused_types_count_l2791_279113


namespace NUMINAMATH_CALUDE_remainder_sum_mod_21_l2791_279181

theorem remainder_sum_mod_21 (c d : ℤ) 
  (hc : c % 60 = 47) 
  (hd : d % 42 = 17) : 
  (c + d) % 21 = 1 := by
sorry

end NUMINAMATH_CALUDE_remainder_sum_mod_21_l2791_279181


namespace NUMINAMATH_CALUDE_distance_condition_l2791_279171

theorem distance_condition (a : ℝ) : 
  (∃ (x : ℝ), x > 0 ∧ 
    (∀ (y : ℝ), y > 0 → 
      (x - a)^2 + (1/x - a)^2 ≤ (y - a)^2 + (1/y - a)^2) ∧
    (x - a)^2 + (1/x - a)^2 = 8) ↔ 
  (a = -1 ∨ a = Real.sqrt 10) :=
sorry

end NUMINAMATH_CALUDE_distance_condition_l2791_279171


namespace NUMINAMATH_CALUDE_last_three_digits_of_7_to_7500_l2791_279160

theorem last_three_digits_of_7_to_7500 (h : 7^500 ≡ 1 [ZMOD 1250]) :
  7^7500 ≡ 1 [ZMOD 1000] := by
  sorry

end NUMINAMATH_CALUDE_last_three_digits_of_7_to_7500_l2791_279160


namespace NUMINAMATH_CALUDE_range_of_a_l2791_279156

theorem range_of_a (A B : Set ℝ) (a : ℝ) 
  (h1 : A = {x : ℝ | x ≤ 2})
  (h2 : B = {x : ℝ | x ≥ a})
  (h3 : A ⊆ B) : 
  a ≤ 2 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l2791_279156


namespace NUMINAMATH_CALUDE_ceiling_floor_sum_l2791_279110

theorem ceiling_floor_sum : ⌈(7 : ℝ) / 3⌉ + ⌊-(7 : ℝ) / 3⌋ = 0 := by sorry

end NUMINAMATH_CALUDE_ceiling_floor_sum_l2791_279110


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2791_279164

theorem quadratic_factorization (a x : ℝ) : a * x^2 - 2 * a * x + a = a * (x - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2791_279164


namespace NUMINAMATH_CALUDE_flight_duration_sum_l2791_279192

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  valid : minutes < 60

/-- Calculates the difference between two times in minutes -/
def timeDiffMinutes (t1 t2 : Time) : ℕ :=
  (t2.hours - t1.hours) * 60 + t2.minutes - t1.minutes

/-- Theorem: Flight duration calculation -/
theorem flight_duration_sum (departureLA : Time) (arrivalNY : Time) (h m : ℕ) :
  departureLA.hours = 9 ∧ departureLA.minutes = 15 ∧
  arrivalNY.hours = 18 ∧ arrivalNY.minutes = 25 ∧
  0 < m ∧ m < 60 ∧
  timeDiffMinutes departureLA { hours := arrivalNY.hours - 3, minutes := arrivalNY.minutes, valid := sorry } = h * 60 + m →
  h + m = 16 := by
  sorry

#check flight_duration_sum

end NUMINAMATH_CALUDE_flight_duration_sum_l2791_279192


namespace NUMINAMATH_CALUDE_simplify_expression_l2791_279186

theorem simplify_expression : (81 ^ (1/4) - Real.sqrt 12.75) ^ 2 = (87 - 12 * Real.sqrt 51) / 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2791_279186


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2791_279170

/-- An arithmetic sequence with a_5 = 3 and a_6 = -2 has common difference -5 -/
theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℤ) 
  (h_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) 
  (h_a5 : a 5 = 3) 
  (h_a6 : a 6 = -2) : 
  a 6 - a 5 = -5 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2791_279170


namespace NUMINAMATH_CALUDE_unique_solution_and_sum_l2791_279149

theorem unique_solution_and_sum : ∃! (a b c : ℕ), 
  ({a, b, c} : Set ℕ) = {0, 1, 2} ∧ 
  (((a ≠ 2) ∧ (b ≠ 2) ∧ (c = 0)) ∨ 
   ((a = 2) ∧ (b = 2) ∧ (c ≠ 0)) ∨ 
   ((a = 2) ∧ (b ≠ 2) ∧ (c ≠ 0))) ∧
  a = 2 ∧ b = 0 ∧ c = 1 ∧ 
  100 * c + 10 * b + a = 102 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_and_sum_l2791_279149


namespace NUMINAMATH_CALUDE_base6_greater_than_base8_l2791_279197

/-- Converts a number from base 6 to base 10 -/
def base6ToBase10 (n : Nat) : Nat :=
  (n / 100) * 36 + ((n / 10) % 10) * 6 + (n % 10)

/-- Converts a number from base 8 to base 10 -/
def base8ToBase10 (n : Nat) : Nat :=
  (n / 100) * 64 + ((n / 10) % 10) * 8 + (n % 10)

theorem base6_greater_than_base8 : base6ToBase10 403 > base8ToBase10 217 := by
  sorry

end NUMINAMATH_CALUDE_base6_greater_than_base8_l2791_279197


namespace NUMINAMATH_CALUDE_smallest_number_with_digit_sum_1981_l2791_279166

/-- A function that returns the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- A function that constructs a number with 1 followed by n nines -/
def oneFollowedByNines (n : ℕ) : ℕ := sorry

/-- The theorem stating that the smallest natural number whose digits sum to 1981
    is 1 followed by 220 nines -/
theorem smallest_number_with_digit_sum_1981 :
  ∀ n : ℕ, sumOfDigits n = 1981 → n ≥ oneFollowedByNines 220 :=
sorry

end NUMINAMATH_CALUDE_smallest_number_with_digit_sum_1981_l2791_279166


namespace NUMINAMATH_CALUDE_total_score_l2791_279178

theorem total_score (darius_score marius_score matt_score : ℕ) : 
  darius_score = 10 →
  marius_score = darius_score + 3 →
  matt_score = darius_score + 5 →
  darius_score + marius_score + matt_score = 38 := by
sorry

end NUMINAMATH_CALUDE_total_score_l2791_279178


namespace NUMINAMATH_CALUDE_card_rotation_result_l2791_279191

-- Define the positions on the card
inductive Position
  | TopLeft
  | TopRight
  | BottomLeft
  | BottomRight

-- Define the colors of the triangles
inductive Color
  | LightGrey
  | DarkGrey

-- Define the card as a function mapping colors to positions
def Card := Color → Position

-- Define the initial card configuration
def initialCard : Card :=
  fun c => match c with
    | Color.LightGrey => Position.BottomRight
    | Color.DarkGrey => Position.BottomLeft

-- Define the rotation about the lower edge
def rotateLowerEdge (card : Card) : Card :=
  fun c => match card c with
    | Position.BottomLeft => Position.TopLeft
    | Position.BottomRight => Position.TopRight
    | p => p

-- Define the rotation about the right-hand edge
def rotateRightEdge (card : Card) : Card :=
  fun c => match card c with
    | Position.TopRight => Position.TopLeft
    | Position.BottomRight => Position.BottomLeft
    | p => p

-- Theorem statement
theorem card_rotation_result :
  let finalCard := rotateRightEdge (rotateLowerEdge initialCard)
  finalCard Color.LightGrey = Position.TopLeft ∧
  finalCard Color.DarkGrey = Position.TopRight := by
  sorry

end NUMINAMATH_CALUDE_card_rotation_result_l2791_279191


namespace NUMINAMATH_CALUDE_good_characterization_l2791_279180

def is_good (n : ℕ) : Prop :=
  ∀ a : ℕ, a ∣ n → (a + 1) ∣ (n + 1)

theorem good_characterization :
  ∀ n : ℕ, n ≥ 1 → (is_good n ↔ n = 1 ∨ (Nat.Prime n ∧ n % 2 = 1)) :=
by sorry

end NUMINAMATH_CALUDE_good_characterization_l2791_279180


namespace NUMINAMATH_CALUDE_interest_difference_l2791_279109

/-- Calculate the difference between compound interest and simple interest -/
theorem interest_difference (P : ℝ) (r : ℝ) (t : ℝ) (n : ℝ) : 
  P = 6000.000000000128 →
  r = 0.05 →
  t = 2 →
  n = 1 →
  let CI := P * (1 + r/n)^(n*t) - P
  let SI := P * r * t
  abs (CI - SI - 15.0000000006914) < 1e-10 := by
  sorry

end NUMINAMATH_CALUDE_interest_difference_l2791_279109


namespace NUMINAMATH_CALUDE_integer_root_of_cubic_l2791_279176

/-- A cubic polynomial with rational coefficients -/
def cubic_polynomial (a b c : ℚ) (x : ℝ) : ℝ :=
  x^3 + a*x^2 + b*x + c

theorem integer_root_of_cubic (a b c : ℚ) :
  (∃ (r : ℤ), cubic_polynomial a b c r = 0) →
  cubic_polynomial a b c (3 - Real.sqrt 5) = 0 →
  ∃ (r : ℤ), cubic_polynomial a b c r = 0 ∧ r = 0 := by
  sorry

end NUMINAMATH_CALUDE_integer_root_of_cubic_l2791_279176


namespace NUMINAMATH_CALUDE_blank_value_l2791_279155

theorem blank_value : ∃ x : ℝ, x * (-2) = 1 ∧ x = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_blank_value_l2791_279155


namespace NUMINAMATH_CALUDE_soccer_substitutions_l2791_279175

theorem soccer_substitutions (total_players : ℕ) (starting_players : ℕ) (non_playing_players : ℕ) :
  total_players = 24 →
  starting_players = 11 →
  non_playing_players = 7 →
  ∃ (first_half_subs : ℕ),
    first_half_subs = 2 ∧
    total_players = starting_players + first_half_subs + 2 * first_half_subs + non_playing_players :=
by sorry

end NUMINAMATH_CALUDE_soccer_substitutions_l2791_279175


namespace NUMINAMATH_CALUDE_survey_C_most_suitable_for_census_l2791_279173

-- Define a structure for a survey
structure Survey where
  description : String
  population_size : ℕ
  resource_requirement : ℕ

-- Define the suitability for census method
def suitable_for_census (s : Survey) : Prop :=
  s.population_size ≤ 100 ∧ s.resource_requirement ≤ 50

-- Define the four survey options
def survey_A : Survey :=
  { description := "Quality and safety of local grain processing",
    population_size := 1000,
    resource_requirement := 200 }

def survey_B : Survey :=
  { description := "Viewership ratings of the 2023 CCTV Spring Festival Gala",
    population_size := 1000000,
    resource_requirement := 500000 }

def survey_C : Survey :=
  { description := "Weekly duration of physical exercise for a ninth-grade class",
    population_size := 50,
    resource_requirement := 30 }

def survey_D : Survey :=
  { description := "Household chores participation of junior high school students in the entire city",
    population_size := 100000,
    resource_requirement := 10000 }

-- Theorem stating that survey C is the most suitable for census method
theorem survey_C_most_suitable_for_census :
  suitable_for_census survey_C ∧
  (¬ suitable_for_census survey_A ∧
   ¬ suitable_for_census survey_B ∧
   ¬ suitable_for_census survey_D) :=
by sorry


end NUMINAMATH_CALUDE_survey_C_most_suitable_for_census_l2791_279173


namespace NUMINAMATH_CALUDE_min_value_theorem_l2791_279135

theorem min_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : x^4 + y^4 + z^4 = 1) :
  (x^3 / (1 - x^8)) + (y^3 / (1 - y^8)) + (z^3 / (1 - z^8)) ≥ 9 * (3^(1/4)) / 8 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2791_279135


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l2791_279190

/-- Given a cubic equation with two known roots, find the value of k and the third root -/
theorem cubic_equation_roots (k : ℝ) : 
  (∀ x : ℝ, x^3 + 5*x^2 + k*x - 12 = 0 ↔ x = 3 ∨ x = -2 ∨ x = -6) →
  k = -12 :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l2791_279190


namespace NUMINAMATH_CALUDE_length_PR_l2791_279162

-- Define the circle and points
def Circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 49}

structure PointsOnCircle where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ
  h_P_on_circle : P ∈ Circle
  h_Q_on_circle : Q ∈ Circle
  h_PQ_distance : (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 64
  h_R_midpoint : R = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- Define the theorem
theorem length_PR (points : PointsOnCircle) : 
  ((points.P.1 - points.R.1)^2 + (points.P.2 - points.R.2)^2)^(1/2) = 4 * (2^(1/2)) := by
  sorry

end NUMINAMATH_CALUDE_length_PR_l2791_279162


namespace NUMINAMATH_CALUDE_lines_parallel_l2791_279127

/-- Two lines in the xy-plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The property of two lines being parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope ∧ l1.intercept ≠ l2.intercept

theorem lines_parallel : 
  let l1 : Line := { slope := 2, intercept := 1 }
  let l2 : Line := { slope := 2, intercept := 5 }
  parallel l1 l2 := by
  sorry

end NUMINAMATH_CALUDE_lines_parallel_l2791_279127


namespace NUMINAMATH_CALUDE_phone_purchase_problem_l2791_279131

/-- Represents the purchase price of phone models -/
structure PhonePrices where
  modelA : ℕ
  modelB : ℕ

/-- Represents a purchasing plan -/
structure PurchasePlan where
  modelACount : ℕ
  modelBCount : ℕ

def totalCost (prices : PhonePrices) (plan : PurchasePlan) : ℕ :=
  prices.modelA * plan.modelACount + prices.modelB * plan.modelBCount

theorem phone_purchase_problem (prices : PhonePrices) : 
  (prices.modelA * 2 + prices.modelB = 5000) →
  (prices.modelA * 3 + prices.modelB * 2 = 8000) →
  (prices.modelA = 2000 ∧ prices.modelB = 1000) ∧
  (∃ (plans : List PurchasePlan), 
    plans.length = 3 ∧
    (∀ plan ∈ plans, 
      plan.modelACount + plan.modelBCount = 20 ∧
      24000 ≤ totalCost prices plan ∧
      totalCost prices plan ≤ 26000) ∧
    (∀ plan : PurchasePlan, 
      plan.modelACount + plan.modelBCount = 20 ∧
      24000 ≤ totalCost prices plan ∧
      totalCost prices plan ≤ 26000 →
      plan ∈ plans)) :=
by sorry

end NUMINAMATH_CALUDE_phone_purchase_problem_l2791_279131


namespace NUMINAMATH_CALUDE_park_available_spaces_l2791_279114

/-- Calculates the number of available spaces in a park given the number of benches, 
    capacity per bench, and number of people currently sitting. -/
def available_spaces (num_benches : ℕ) (capacity_per_bench : ℕ) (people_sitting : ℕ) : ℕ :=
  num_benches * capacity_per_bench - people_sitting

/-- Theorem stating that in a park with 50 benches, each with a capacity of 4 people, 
    and 80 people currently sitting, there are 120 available spaces. -/
theorem park_available_spaces : 
  available_spaces 50 4 80 = 120 := by
  sorry

end NUMINAMATH_CALUDE_park_available_spaces_l2791_279114
