import Mathlib

namespace quadratic_vertex_on_x_axis_l2499_249927

-- Define the quadratic function
def f (x m : ℝ) : ℝ := x^2 - x + m

-- Define the condition for the vertex being on the x-axis
def vertex_on_x_axis (m : ℝ) : Prop :=
  let x₀ := 1/2  -- x-coordinate of the vertex
  f x₀ m = 0

-- Theorem statement
theorem quadratic_vertex_on_x_axis (m : ℝ) :
  vertex_on_x_axis m → m = 1/4 := by sorry

end quadratic_vertex_on_x_axis_l2499_249927


namespace perfect_non_spiral_shells_l2499_249910

theorem perfect_non_spiral_shells (total_perfect : ℕ) (total_broken : ℕ) 
  (h1 : total_perfect = 17)
  (h2 : total_broken = 52)
  (h3 : total_broken / 2 = total_broken - total_broken / 2)  -- Half of broken shells are spiral
  (h4 : total_broken / 2 = (total_perfect - (total_perfect - (total_broken / 2 - 21))) + 21) :
  total_perfect - (total_perfect - (total_broken / 2 - 21)) = 12 := by
  sorry

#check perfect_non_spiral_shells

end perfect_non_spiral_shells_l2499_249910


namespace logical_reasoning_methods_correct_answer_is_C_l2499_249981

-- Define the reasoning methods
inductive ReasoningMethod
| SphereFromCircle
| TriangleAngleSum
| ClassPerformance
| PolygonAngleSum

-- Define a predicate for logical reasoning
def isLogical : ReasoningMethod → Prop
| ReasoningMethod.SphereFromCircle => True
| ReasoningMethod.TriangleAngleSum => True
| ReasoningMethod.ClassPerformance => False
| ReasoningMethod.PolygonAngleSum => True

-- Theorem stating which reasoning methods are logical
theorem logical_reasoning_methods :
  (isLogical ReasoningMethod.SphereFromCircle) ∧
  (isLogical ReasoningMethod.TriangleAngleSum) ∧
  (¬isLogical ReasoningMethod.ClassPerformance) ∧
  (isLogical ReasoningMethod.PolygonAngleSum) :=
by sorry

-- Define the answer options
inductive AnswerOption
| A
| B
| C
| D

-- Define the correct answer
def correctAnswer : AnswerOption := AnswerOption.C

-- Theorem stating that C is the correct answer
theorem correct_answer_is_C :
  correctAnswer = AnswerOption.C :=
by sorry

end logical_reasoning_methods_correct_answer_is_C_l2499_249981


namespace solution_set_theorem_l2499_249909

def f (a b x : ℝ) := (a * x - 1) * (x + b)

theorem solution_set_theorem (a b : ℝ) :
  (∀ x, f a b x > 0 ↔ -1 < x ∧ x < 3) →
  (∀ x, f a b (-2 * x) < 0 ↔ x < -3/2 ∨ 1/2 < x) :=
by sorry

end solution_set_theorem_l2499_249909


namespace complement_intersection_equals_l2499_249988

def U : Set ℕ := {1, 2, 3, 4}
def P : Set ℕ := {2, 3}
def Q : Set ℕ := {3, 4}

theorem complement_intersection_equals :
  (U \ (P ∩ Q)) = {1, 2, 4} := by sorry

end complement_intersection_equals_l2499_249988


namespace longest_diagonal_twice_side_l2499_249966

/-- Regular octagon with side length a, shortest diagonal b, and longest diagonal c -/
structure RegularOctagon where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a

/-- Theorem: In a regular octagon, the longest diagonal is twice the side length -/
theorem longest_diagonal_twice_side (octagon : RegularOctagon) : octagon.c = 2 * octagon.a := by
  sorry

#check longest_diagonal_twice_side

end longest_diagonal_twice_side_l2499_249966


namespace professor_seating_count_l2499_249908

/-- The number of chairs in a row -/
def total_chairs : ℕ := 12

/-- The number of professors -/
def num_professors : ℕ := 4

/-- The number of students -/
def num_students : ℕ := 8

/-- The number of chairs available for professors (excluding first and last) -/
def available_chairs : ℕ := total_chairs - 2

/-- The number of effective chairs after considering spacing requirements -/
def effective_chairs : ℕ := available_chairs - (num_professors - 1)

/-- The number of ways to arrange professors' seating -/
def professor_seating_arrangements : ℕ := (effective_chairs.choose num_professors) * num_professors.factorial

theorem professor_seating_count :
  professor_seating_arrangements = 1680 :=
sorry

end professor_seating_count_l2499_249908


namespace problem_solution_l2499_249918

open Real

noncomputable def f (x : ℝ) : ℝ := (log (1 + x)) / x

theorem problem_solution :
  (∀ x y, 0 < x ∧ x < y → f y < f x) ∧
  (∀ a : ℝ, (∀ x : ℝ, 0 < x → log (1 + x) < a * x) ↔ 1 ≤ a) ∧
  (∀ n : ℕ, 0 < n → (1 + 1 / n : ℝ) ^ n < exp 1) :=
by sorry

end problem_solution_l2499_249918


namespace sum_of_series_l2499_249934

def series_sum (n : ℕ) : ℕ :=
  n * (n + 1) / 2 - 1

theorem sum_of_series :
  series_sum 15 - series_sum 1 = 91 := by
  sorry

end sum_of_series_l2499_249934


namespace parabola_focus_at_triangle_centroid_l2499_249970

/-- Given a triangle ABC with vertices A(-1,2), B(3,4), and C(4,-6),
    and a parabola y^2 = ax with focus at the centroid of ABC,
    prove that a = 8. -/
theorem parabola_focus_at_triangle_centroid :
  let A : ℝ × ℝ := (-1, 2)
  let B : ℝ × ℝ := (3, 4)
  let C : ℝ × ℝ := (4, -6)
  let centroid : ℝ × ℝ := ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)
  ∀ a : ℝ, (∀ x y : ℝ, y^2 = a*x → (x = a/4 ↔ (x, y) = centroid)) → a = 8 :=
by sorry

end parabola_focus_at_triangle_centroid_l2499_249970


namespace complement_of_union_l2499_249938

-- Define the universal set U
def U : Set Int := {-2, -1, 0, 1, 2, 3}

-- Define set A
def A : Set Int := {-1, 2}

-- Define set B
def B : Set Int := {x : Int | x^2 - 4*x + 3 = 0}

-- Theorem statement
theorem complement_of_union :
  (U \ (A ∪ B)) = {-2, 0} := by sorry

end complement_of_union_l2499_249938


namespace carmen_candle_usage_l2499_249982

/-- Calculates the number of candles used given the burning time per night and total nights -/
def candles_used (burn_time_per_night : ℚ) (total_nights : ℕ) : ℚ :=
  (burn_time_per_night * total_nights) / 8

theorem carmen_candle_usage :
  candles_used 2 24 = 6 := by sorry

end carmen_candle_usage_l2499_249982


namespace distance_between_points_l2499_249912

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (2, -3)
  let p2 : ℝ × ℝ := (5, 6)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = 3 * Real.sqrt 10 := by
  sorry

end distance_between_points_l2499_249912


namespace folded_rectangle_theorem_l2499_249973

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a rectangle with vertices A, B, C, D -/
structure Rectangle :=
  (A : Point)
  (B : Point)
  (C : Point)
  (D : Point)

/-- Represents the folded state of the rectangle -/
structure FoldedRectangle :=
  (rect : Rectangle)
  (E : Point)
  (F : Point)

/-- Given a folded rectangle, returns the measure of Angle 1 in degrees -/
def angle1 (fr : FoldedRectangle) : ℝ := sorry

/-- Given a folded rectangle, returns the measure of Angle 2 in degrees -/
def angle2 (fr : FoldedRectangle) : ℝ := sorry

/-- Predicate to check if a point is on a line segment -/
def isOnSegment (P Q R : Point) : Prop := sorry

/-- Predicate to check if two triangles are congruent -/
def areCongruentTriangles (ABC DEF : Point × Point × Point) : Prop := sorry

theorem folded_rectangle_theorem (fr : FoldedRectangle) :
  isOnSegment fr.rect.A fr.E fr.rect.B ∧
  areCongruentTriangles (fr.rect.D, fr.rect.C, fr.F) (fr.rect.D, fr.E, fr.F) ∧
  angle1 fr = 22 →
  angle2 fr = 44 :=
sorry

end folded_rectangle_theorem_l2499_249973


namespace lower_limit_proof_l2499_249902

def is_prime (n : ℕ) : Prop := sorry

def count_primes_between (a b : ℝ) : ℕ := sorry

theorem lower_limit_proof : 
  ∀ x : ℕ, x ≤ 19 ↔ count_primes_between x (87/5) ≥ 2 :=
sorry

end lower_limit_proof_l2499_249902


namespace parallelogram_altitude_base_ratio_l2499_249926

/-- Given a parallelogram with area 242 sq m and base 11 m, prove its altitude to base ratio is 2 -/
theorem parallelogram_altitude_base_ratio : 
  ∀ (area base altitude : ℝ), 
  area = 242 ∧ base = 11 ∧ area = base * altitude → 
  altitude / base = 2 := by
sorry

end parallelogram_altitude_base_ratio_l2499_249926


namespace centroid_circle_area_l2499_249962

/-- Given a circle with diameter 'd', the area of the circle traced by the centroid of a triangle
    formed by the diameter and a point on the circumference is (25/900) times the area of the original circle. -/
theorem centroid_circle_area (d : ℝ) (h : d > 0) :
  ∃ (A_centroid A_circle : ℝ),
    A_circle = π * (d/2)^2 ∧
    A_centroid = π * (d/6)^2 ∧
    A_centroid = (25/900) * A_circle :=
by sorry

end centroid_circle_area_l2499_249962


namespace max_hangers_buyable_l2499_249920

def total_budget : ℝ := 60
def tissue_cost : ℝ := 34.8
def hanger_cost : ℝ := 1.6

theorem max_hangers_buyable : 
  ⌊(total_budget - tissue_cost) / hanger_cost⌋ = 15 := by sorry

end max_hangers_buyable_l2499_249920


namespace bank_transfer_problem_l2499_249953

theorem bank_transfer_problem (X : ℝ) :
  (0.8 * X = 30000) → X = 37500 := by
  sorry

end bank_transfer_problem_l2499_249953


namespace polynomial_value_at_n_plus_one_l2499_249978

theorem polynomial_value_at_n_plus_one (n : ℕ) (p : ℝ → ℝ) :
  (∀ k : ℕ, k ≤ n → p k = k / (k + 1)) →
  p (n + 1) = if n % 2 = 1 then 1 else n / (n + 2) := by
  sorry

end polynomial_value_at_n_plus_one_l2499_249978


namespace sugar_and_salt_pricing_l2499_249949

/-- Given the price of sugar and salt, prove the cost of a specific quantity -/
theorem sugar_and_salt_pricing
  (price_2kg_sugar_5kg_salt : ℝ)
  (price_1kg_sugar : ℝ)
  (h1 : price_2kg_sugar_5kg_salt = 5.50)
  (h2 : price_1kg_sugar = 1.50) :
  3 * price_1kg_sugar + (price_2kg_sugar_5kg_salt - 2 * price_1kg_sugar) / 5 = 5 :=
by sorry

end sugar_and_salt_pricing_l2499_249949


namespace negation_of_implication_l2499_249922

theorem negation_of_implication (m : ℝ) : 
  (¬(m > 0 → ∃ x : ℝ, x^2 + x - m = 0)) ↔ 
  (m ≤ 0 → ¬∃ x : ℝ, x^2 + x - m = 0) := by sorry

end negation_of_implication_l2499_249922


namespace towel_rate_proof_l2499_249956

/-- Proves that given the specified towel purchases and average price, the unknown rate must be 300. -/
theorem towel_rate_proof (price1 price2 avg_price : ℕ) (count1 count2 count_unknown : ℕ) :
  price1 = 100 →
  price2 = 150 →
  avg_price = 165 →
  count1 = 3 →
  count2 = 5 →
  count_unknown = 2 →
  ∃ (unknown_rate : ℕ),
    (count1 * price1 + count2 * price2 + count_unknown * unknown_rate) / (count1 + count2 + count_unknown) = avg_price ∧
    unknown_rate = 300 :=
by sorry

end towel_rate_proof_l2499_249956


namespace square_graph_triangles_l2499_249996

/-- A planar graph formed by a square with interior points -/
structure SquareGraph where
  /-- The number of interior points in the square -/
  interior_points : ℕ
  /-- The total number of vertices in the graph -/
  vertices : ℕ
  /-- The total number of edges in the graph -/
  edges : ℕ
  /-- The total number of faces in the graph (including the exterior face) -/
  faces : ℕ
  /-- The condition that the graph is formed by a square with interior points -/
  square_condition : vertices = interior_points + 4
  /-- The condition that all regions except the exterior are triangles -/
  triangle_condition : 2 * edges = 3 * (faces - 1) + 4
  /-- Euler's formula for planar graphs -/
  euler_formula : vertices - edges + faces = 2

/-- The theorem stating the number of triangles in the specific square graph -/
theorem square_graph_triangles (g : SquareGraph) (h : g.interior_points = 20) :
  g.faces - 1 = 42 := by
  sorry

end square_graph_triangles_l2499_249996


namespace unique_solution_l2499_249993

/-- A three-digit number represented as a tuple of its digits -/
def ThreeDigitNumber := (ℕ × ℕ × ℕ)

/-- Convert a ThreeDigitNumber to its integer representation -/
def to_int (n : ThreeDigitNumber) : ℕ :=
  100 * n.1 + 10 * n.2.1 + n.2.2

/-- Check if a ThreeDigitNumber satisfies the condition abc = (a + b + c)^3 -/
def satisfies_condition (n : ThreeDigitNumber) : Prop :=
  to_int n = (n.1 + n.2.1 + n.2.2) ^ 3

/-- The theorem stating that 512 is the only solution -/
theorem unique_solution :
  ∃! (n : ThreeDigitNumber), 
    100 ≤ to_int n ∧ 
    to_int n ≤ 999 ∧ 
    satisfies_condition n ∧
    to_int n = 512 := by sorry

end unique_solution_l2499_249993


namespace calculate_food_price_l2499_249954

/-- Given a total bill that includes tax and tip, calculate the original food price -/
theorem calculate_food_price (total : ℝ) (tax_rate : ℝ) (tip_rate : ℝ) (food_price : ℝ) : 
  total = 211.20 ∧ 
  tax_rate = 0.10 ∧ 
  tip_rate = 0.20 ∧ 
  total = food_price * (1 + tax_rate) * (1 + tip_rate) → 
  food_price = 160 := by
  sorry

end calculate_food_price_l2499_249954


namespace root_minus_one_quadratic_equation_l2499_249998

theorem root_minus_one_quadratic_equation (p : ℚ) :
  (∀ x, (2*p - 1) * x^2 + 2*(1 - p) * x + 3*p = 0 ↔ x = -1) ↔ p = 3/7 := by
  sorry

end root_minus_one_quadratic_equation_l2499_249998


namespace arithmetic_progression_equality_l2499_249967

theorem arithmetic_progression_equality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ((a + b) / 2 = (Real.sqrt (a * b) + Real.sqrt ((a^2 + b^2) / 2)) / 2) → a = b := by
  sorry

end arithmetic_progression_equality_l2499_249967


namespace sum_of_roots_zero_l2499_249984

theorem sum_of_roots_zero (p q : ℝ) : 
  (∀ x, x^2 + p*x + q = 0 ↔ x = p ∨ x = q) → 
  p = 2*q → 
  p + q = 0 := by
sorry

end sum_of_roots_zero_l2499_249984


namespace four_inch_gold_cube_value_l2499_249969

/-- The value of a cube of gold given its side length -/
def gold_value (side : ℕ) : ℚ :=
  let base_value : ℚ := 300
  let base_side : ℕ := 1
  let increase_rate : ℚ := 1.1
  let value_per_cubic_inch : ℚ := base_value * (increase_rate ^ (side - base_side))
  (side ^ 3 : ℚ) * value_per_cubic_inch

/-- Theorem stating the value of a 4-inch cube of gold -/
theorem four_inch_gold_cube_value :
  ⌊gold_value 4⌋ = 25555 :=
sorry

end four_inch_gold_cube_value_l2499_249969


namespace total_profit_is_30000_l2499_249995

/-- Represents the profit distribution problem -/
structure ProfitProblem where
  total_subscription : ℕ
  a_more_than_b : ℕ
  b_more_than_c : ℕ
  b_profit : ℕ

/-- Calculate the total profit given the problem parameters -/
def calculate_total_profit (p : ProfitProblem) : ℕ :=
  sorry

/-- Theorem stating that the total profit is 30000 given the specific problem parameters -/
theorem total_profit_is_30000 :
  let p : ProfitProblem := {
    total_subscription := 50000,
    a_more_than_b := 4000,
    b_more_than_c := 5000,
    b_profit := 10200
  }
  calculate_total_profit p = 30000 := by sorry

end total_profit_is_30000_l2499_249995


namespace revenue_change_l2499_249980

theorem revenue_change (R : ℝ) (x : ℝ) (h : R > 0) :
  R * (1 + x / 100) * (1 - x / 100) = R * 0.96 →
  x = 20 := by
sorry

end revenue_change_l2499_249980


namespace trig_identity_l2499_249947

theorem trig_identity (x y : ℝ) : 
  Real.sin (x + y) * Real.sin x + Real.cos (x + y) * Real.cos x = Real.cos y := by
  sorry

end trig_identity_l2499_249947


namespace max_chord_length_l2499_249919

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 8*y

-- Define a point on the parabola
structure Point := (x : ℝ) (y : ℝ)

-- Define a chord on the parabola
structure Chord := (A : Point) (B : Point)

-- Define the condition for the midpoint of the chord
def midpointCondition (c : Chord) : Prop := (c.A.y + c.B.y) / 2 = 4

-- Define the length of the chord
def chordLength (c : Chord) : ℝ := abs (c.A.x - c.B.x)

-- Theorem statement
theorem max_chord_length :
  ∀ c : Chord,
  parabola c.A.x c.A.y →
  parabola c.B.x c.B.y →
  midpointCondition c →
  ∃ maxLength : ℝ, maxLength = 12 ∧ ∀ otherChord : Chord,
    parabola otherChord.A.x otherChord.A.y →
    parabola otherChord.B.x otherChord.B.y →
    midpointCondition otherChord →
    chordLength otherChord ≤ maxLength :=
sorry

end max_chord_length_l2499_249919


namespace at_least_one_zero_l2499_249986

theorem at_least_one_zero (a b : ℝ) : (a ≠ 0 ∧ b ≠ 0) → False := by
  sorry

end at_least_one_zero_l2499_249986


namespace min_value_theorem_l2499_249999

theorem min_value_theorem (c a b : ℝ) (hc : c > 0) (ha : a ≠ 0) (hb : b ≠ 0)
  (heq : 4 * a^2 - 2 * a * b + b^2 - c = 0)
  (hmax : ∀ (a' b' : ℝ), a' ≠ 0 → b' ≠ 0 → 4 * a'^2 - 2 * a' * b' + b'^2 - c = 0 →
    |2 * a + b| ≥ |2 * a' + b'|) :
  (1 / a + 2 / b + 4 / c) ≥ -1 :=
sorry

end min_value_theorem_l2499_249999


namespace circle_area_difference_l2499_249961

theorem circle_area_difference (r₁ r₂ r₃ : ℝ) (h₁ : r₁ = 24) (h₂ : r₂ = 36) :
  r₃ ^ 2 * π = (r₂ ^ 2 - r₁ ^ 2) * π → r₃ = 12 * Real.sqrt 5 := by
  sorry

end circle_area_difference_l2499_249961


namespace tires_cost_calculation_l2499_249946

def speakers_cost : ℚ := 136.01
def cd_player_cost : ℚ := 139.38
def total_spent : ℚ := 387.85

theorem tires_cost_calculation :
  total_spent - (speakers_cost + cd_player_cost) = 112.46 :=
by sorry

end tires_cost_calculation_l2499_249946


namespace largest_perfect_square_factor_of_4410_l2499_249931

def largest_perfect_square_factor (n : ℕ) : ℕ := sorry

theorem largest_perfect_square_factor_of_4410 :
  largest_perfect_square_factor 4410 = 441 := by sorry

end largest_perfect_square_factor_of_4410_l2499_249931


namespace distinct_and_no_real_solutions_l2499_249906

theorem distinct_and_no_real_solutions : 
  ∀ b c : ℕ+, 
    (∃ x y : ℝ, x ≠ y ∧ x^2 + b*x + c = 0 ∧ y^2 + b*y + c = 0) ∧ 
    (∀ z : ℝ, z^2 + c*z + b ≠ 0) → 
    ((b = 3 ∧ c = 1) ∨ (b = 3 ∧ c = 2)) := by
  sorry

end distinct_and_no_real_solutions_l2499_249906


namespace min_gumballs_for_given_machine_l2499_249904

/-- Represents the number of gumballs of each color -/
structure GumballMachine :=
  (red : ℕ)
  (white : ℕ)
  (blue : ℕ)

/-- The minimum number of gumballs needed to guarantee at least 4 of the same color -/
def minGumballs (machine : GumballMachine) : ℕ := sorry

/-- Theorem stating the minimum number of gumballs needed for the given machine -/
theorem min_gumballs_for_given_machine :
  let machine : GumballMachine := ⟨8, 10, 6⟩
  minGumballs machine = 10 := by sorry

end min_gumballs_for_given_machine_l2499_249904


namespace cuboid_surface_area_formula_l2499_249964

/-- The surface area of a cuboid with edges of length a, b, and c. -/
def cuboidSurfaceArea (a b c : ℝ) : ℝ := 2 * a * b + 2 * b * c + 2 * a * c

/-- Theorem: The surface area of a cuboid with edges of length a, b, and c
    is equal to 2ab + 2bc + 2ac. -/
theorem cuboid_surface_area_formula (a b c : ℝ) :
  cuboidSurfaceArea a b c = 2 * a * b + 2 * b * c + 2 * a * c := by
  sorry

end cuboid_surface_area_formula_l2499_249964


namespace cube_values_l2499_249928

def f (n : ℕ) : ℤ := n^3 - 18*n^2 + 115*n - 391

def is_cube (x : ℤ) : Prop := ∃ y : ℤ, y^3 = x

theorem cube_values :
  {n : ℕ | is_cube (f n)} = {7, 11, 12, 25} := by sorry

end cube_values_l2499_249928


namespace solution_to_equation_l2499_249914

theorem solution_to_equation : ∃ (x y : ℝ), 2 * x - y = 5 ∧ x = 3 ∧ y = 1 := by
  sorry

end solution_to_equation_l2499_249914


namespace rope_cutting_probability_rope_cutting_probability_is_one_third_l2499_249965

/-- The probability of cutting a rope of length 3 into two segments,
    each at least 1 unit long, when cut at a random position. -/
theorem rope_cutting_probability : ℝ :=
  let rope_length : ℝ := 3
  let min_segment_length : ℝ := 1
  let favorable_cut_length : ℝ := rope_length - 2 * min_segment_length
  favorable_cut_length / rope_length

/-- The probability of cutting a rope of length 3 into two segments,
    each at least 1 unit long, when cut at a random position, is 1/3. -/
theorem rope_cutting_probability_is_one_third :
  rope_cutting_probability = 1 / 3 := by
  sorry

end rope_cutting_probability_rope_cutting_probability_is_one_third_l2499_249965


namespace gcd_lcm_sum_75_4410_l2499_249942

theorem gcd_lcm_sum_75_4410 : Nat.gcd 75 4410 + Nat.lcm 75 4410 = 22065 := by
  sorry

end gcd_lcm_sum_75_4410_l2499_249942


namespace baker_production_theorem_l2499_249977

/-- Represents the baker's bread production over a period of time. -/
structure BakerProduction where
  loaves_per_oven_hour : ℕ
  num_ovens : ℕ
  weekday_hours : ℕ
  weekend_hours : ℕ
  num_weeks : ℕ

/-- Calculates the total number of loaves baked over the given period. -/
def total_loaves (bp : BakerProduction) : ℕ :=
  let loaves_per_hour := bp.loaves_per_oven_hour * bp.num_ovens
  let weekday_loaves := loaves_per_hour * bp.weekday_hours * 5
  let weekend_loaves := loaves_per_hour * bp.weekend_hours * 2
  (weekday_loaves + weekend_loaves) * bp.num_weeks

/-- Theorem stating that given the baker's production conditions, 
    the total number of loaves baked in 3 weeks is 1740. -/
theorem baker_production_theorem (bp : BakerProduction) 
  (h1 : bp.loaves_per_oven_hour = 5)
  (h2 : bp.num_ovens = 4)
  (h3 : bp.weekday_hours = 5)
  (h4 : bp.weekend_hours = 2)
  (h5 : bp.num_weeks = 3) :
  total_loaves bp = 1740 := by
  sorry

#eval total_loaves ⟨5, 4, 5, 2, 3⟩

end baker_production_theorem_l2499_249977


namespace p_satisfies_conditions_l2499_249987

/-- A cubic polynomial p(x) satisfying specific conditions -/
def p (x : ℝ) : ℝ := -x^3 + 4*x^2 - 7*x - 3

/-- Theorem stating that p(x) satisfies the given conditions -/
theorem p_satisfies_conditions :
  p 1 = -7 ∧ p 2 = -9 ∧ p 3 = -15 ∧ p 4 = -31 := by
  sorry

#eval p 1
#eval p 2
#eval p 3
#eval p 4

end p_satisfies_conditions_l2499_249987


namespace flag_rectangle_ratio_l2499_249994

/-- Given a rectangle with side lengths in ratio 3:5, divided into four equal area rectangles,
    the ratio of the shorter side to the longer side of one of these rectangles is 4:15 -/
theorem flag_rectangle_ratio :
  ∀ (k : ℝ), k > 0 →
  let flag_width := 5 * k
  let flag_height := 3 * k
  let small_rect_area := (flag_width * flag_height) / 4
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧
    x * y = small_rect_area ∧
    3 * y = k ∧
    5 * y = x ∧
    y / x = 4 / 15 :=
by sorry

end flag_rectangle_ratio_l2499_249994


namespace bookcase_organization_l2499_249916

/-- The number of books on each shelf -/
def books_per_shelf : ℕ := 9

/-- The number of shelves containing mystery books -/
def mystery_shelves : ℕ := 3

/-- The number of shelves containing picture books -/
def picture_shelves : ℕ := 5

/-- The total number of books in the bookcase -/
def total_books : ℕ := books_per_shelf * (mystery_shelves + picture_shelves)

theorem bookcase_organization :
  total_books = 72 := by sorry

end bookcase_organization_l2499_249916


namespace lcm_of_ratio_and_hcf_l2499_249940

theorem lcm_of_ratio_and_hcf (a b c : ℕ+) : 
  (∃ (k : ℕ+), a = 2 * k ∧ b = 3 * k ∧ c = 5 * k) → 
  Nat.gcd a (Nat.gcd b c) = 6 →
  Nat.lcm a (Nat.lcm b c) = 180 := by
sorry

end lcm_of_ratio_and_hcf_l2499_249940


namespace inequality_range_l2499_249950

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 4 * x + a > 1 - 2 * x^2) ↔ a > 2 :=
by sorry

end inequality_range_l2499_249950


namespace log_2_base_10_bounds_l2499_249957

-- Define the given conditions
axiom pow_10_4 : (10 : ℝ) ^ 4 = 10000
axiom pow_10_5 : (10 : ℝ) ^ 5 = 100000
axiom pow_2_12 : (2 : ℝ) ^ 12 = 4096
axiom pow_2_15 : (2 : ℝ) ^ 15 = 32768

-- State the theorem
theorem log_2_base_10_bounds :
  0.30 < Real.log 2 / Real.log 10 ∧ Real.log 2 / Real.log 10 < 1/3 := by
  sorry

end log_2_base_10_bounds_l2499_249957


namespace hyperbola_asymptotes_l2499_249945

/-- Given a hyperbola with equation x^2 - 4y^2 = -1, its asymptotes are x ± 2y = 0 -/
theorem hyperbola_asymptotes (x y : ℝ) :
  x^2 - 4*y^2 = -1 →
  ∃ (k : ℝ), (x + 2*y = 0 ∧ x - 2*y = 0) ∨ (2*y + x = 0 ∧ 2*y - x = 0) :=
by sorry

end hyperbola_asymptotes_l2499_249945


namespace profit_calculation_l2499_249991

-- Define package prices
def basic_price : ℕ := 5
def deluxe_price : ℕ := 10
def premium_price : ℕ := 15

-- Define weekday car wash numbers
def basic_cars : ℕ := 50
def deluxe_cars : ℕ := 40
def premium_cars : ℕ := 20

-- Define employee wages
def employee_a_wage : ℕ := 110
def employee_b_wage : ℕ := 90
def employee_c_wage : ℕ := 100
def employee_d_wage : ℕ := 80

-- Define operating expenses
def weekday_expenses : ℕ := 200

-- Define the number of weekdays
def weekdays : ℕ := 5

-- Define the function to calculate total profit
def total_profit : ℕ :=
  let daily_revenue := basic_price * basic_cars + deluxe_price * deluxe_cars + premium_price * premium_cars
  let total_revenue := daily_revenue * weekdays
  let employee_expenses := employee_a_wage * 5 + employee_b_wage * 2 + employee_c_wage * 3 + employee_d_wage * 2
  let total_expenses := employee_expenses + weekday_expenses * weekdays
  total_revenue - total_expenses

-- Theorem statement
theorem profit_calculation : total_profit = 2560 := by
  sorry

end profit_calculation_l2499_249991


namespace natural_number_power_equality_l2499_249930

theorem natural_number_power_equality (p q : ℕ) (h : p^p + q^q = p^q + q^p) : p = q := by
  sorry

end natural_number_power_equality_l2499_249930


namespace cyclic_sum_extrema_l2499_249905

def cyclic_sum (a : List ℕ) : ℕ :=
  (List.zip a (a.rotate 1)).map (fun (x, y) => x * y) |>.sum

def is_permutation (a : List ℕ) (n : ℕ) : Prop :=
  a.length = n ∧ a.toFinset = Finset.range n

def max_permutation (n : ℕ) : List ℕ :=
  (List.range ((n + 1) / 2)).map (fun i => 2 * i + 1) ++
  (List.range (n / 2)).reverse.map (fun i => 2 * (i + 1))

def min_permutation (n : ℕ) : List ℕ :=
  if n % 2 = 0 then
    (List.range (n / 2)).reverse.map (fun i => n - 2 * i) ++
    (List.range (n / 2)).map (fun i => 2 * i + 1)
  else
    (List.range ((n + 1) / 2)).reverse.map (fun i => n - 2 * i) ++
    (List.range (n / 2)).map (fun i => 2 * i + 2)

theorem cyclic_sum_extrema (n : ℕ) (a : List ℕ) (h : is_permutation a n) :
  cyclic_sum a ≤ cyclic_sum (max_permutation n) ∧
  cyclic_sum (min_permutation n) ≤ cyclic_sum a := by sorry

end cyclic_sum_extrema_l2499_249905


namespace five_balls_three_boxes_l2499_249907

/-- The number of ways to place n distinguishable balls into k indistinguishable boxes -/
def placeBalls (n k : ℕ) : ℕ := sorry

/-- Theorem: There are 31 ways to place 5 distinguishable balls into 3 indistinguishable boxes -/
theorem five_balls_three_boxes : placeBalls 5 3 = 31 := by sorry

end five_balls_three_boxes_l2499_249907


namespace sin_negative_300_degrees_l2499_249948

theorem sin_negative_300_degrees : Real.sin (-(300 * π / 180)) = Real.sqrt 3 / 2 := by
  sorry

end sin_negative_300_degrees_l2499_249948


namespace representable_integers_l2499_249974

/-- Represents an arithmetic expression using only the digit 2 and basic operations -/
inductive Expr
  | two : Expr
  | add : Expr → Expr → Expr
  | sub : Expr → Expr → Expr
  | mul : Expr → Expr → Expr
  | div : Expr → Expr → Expr

/-- Evaluates an arithmetic expression -/
def eval : Expr → ℚ
  | Expr.two => 2
  | Expr.add e1 e2 => eval e1 + eval e2
  | Expr.sub e1 e2 => eval e1 - eval e2
  | Expr.mul e1 e2 => eval e1 * eval e2
  | Expr.div e1 e2 => eval e1 / eval e2

/-- Counts the number of 2's used in an expression -/
def count_twos : Expr → ℕ
  | Expr.two => 1
  | Expr.add e1 e2 => count_twos e1 + count_twos e2
  | Expr.sub e1 e2 => count_twos e1 + count_twos e2
  | Expr.mul e1 e2 => count_twos e1 + count_twos e2
  | Expr.div e1 e2 => count_twos e1 + count_twos e2

theorem representable_integers :
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 2019 →
  ∃ e : Expr, eval e = n ∧ count_twos e ≤ 17 :=
sorry

end representable_integers_l2499_249974


namespace number_puzzle_l2499_249925

theorem number_puzzle : ∃ x : ℝ, x = 280 ∧ (x / 5 + 4 = x / 4 - 10) := by
  sorry

end number_puzzle_l2499_249925


namespace divisibility_by_seven_l2499_249913

theorem divisibility_by_seven (n a b d : ℤ) 
  (h1 : 0 ≤ b ∧ b ≤ 9)
  (h2 : 0 ≤ a)
  (h3 : n = 10 * a + b)
  (h4 : d = a - 2 * b) :
  7 ∣ n ↔ 7 ∣ d := by
  sorry

end divisibility_by_seven_l2499_249913


namespace line_perpendicular_to_parallel_line_l2499_249989

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (perpendicularLines : Line → Line → Prop)

-- State the theorem
theorem line_perpendicular_to_parallel_line
  (a b : Line) (α : Plane)
  (h1 : perpendicular a α)
  (h2 : parallel b α) :
  perpendicularLines a b :=
sorry

end line_perpendicular_to_parallel_line_l2499_249989


namespace round_73_26_repeating_l2499_249968

/-- Represents a repeating decimal number -/
structure RepeatingDecimal where
  integerPart : ℕ
  nonRepeatingPart : ℕ
  repeatingPart : ℕ

/-- Rounds a RepeatingDecimal to the nearest hundredth -/
def roundToHundredth (x : RepeatingDecimal) : ℚ :=
  sorry

/-- The specific number 73.2626... -/
def number : RepeatingDecimal :=
  { integerPart := 73,
    nonRepeatingPart := 26,
    repeatingPart := 26 }

theorem round_73_26_repeating :
  roundToHundredth number = 73.26 :=
sorry

end round_73_26_repeating_l2499_249968


namespace total_seats_calculation_l2499_249915

/-- The number of students per bus -/
def students_per_bus : ℝ := 14.0

/-- The number of buses -/
def number_of_buses : ℝ := 2.0

/-- The total number of seats taken up by students -/
def total_seats : ℝ := students_per_bus * number_of_buses

theorem total_seats_calculation : total_seats = 28 := by
  sorry

end total_seats_calculation_l2499_249915


namespace largest_x_floor_ratio_l2499_249990

theorem largest_x_floor_ratio : 
  ∀ x : ℝ, (↑(Int.floor x) / x = 8 / 9) → x ≤ 63 / 8 :=
by sorry

end largest_x_floor_ratio_l2499_249990


namespace coin_grid_intersection_probability_l2499_249955

/-- Probability of a coin intersecting grid lines -/
theorem coin_grid_intersection_probability
  (grid_edge_length : ℝ)
  (coin_diameter : ℝ)
  (h_grid : grid_edge_length = 6)
  (h_coin : coin_diameter = 2) :
  (1 : ℝ) - (grid_edge_length - coin_diameter)^2 / grid_edge_length^2 = 5/9 := by
  sorry

end coin_grid_intersection_probability_l2499_249955


namespace smallest_linear_combination_2023_54321_l2499_249933

theorem smallest_linear_combination_2023_54321 :
  ∃ (k : ℕ), k > 0 ∧ (∃ (m n : ℤ), k = 2023 * m + 54321 * n) ∧
  ∀ (j : ℕ), j > 0 → (∃ (x y : ℤ), j = 2023 * x + 54321 * y) → k ≤ j :=
by sorry

end smallest_linear_combination_2023_54321_l2499_249933


namespace committee_meeting_attendance_l2499_249952

theorem committee_meeting_attendance :
  ∀ (associate_profs assistant_profs : ℕ),
    2 * associate_profs + assistant_profs = 11 →
    associate_profs + 2 * assistant_profs = 16 →
    associate_profs + assistant_profs = 9 :=
by
  sorry

end committee_meeting_attendance_l2499_249952


namespace triangle_inradius_l2499_249972

/-- Given a triangle with perimeter 28 cm and area 35 cm², prove that its inradius is 2.5 cm. -/
theorem triangle_inradius (P : ℝ) (A : ℝ) (r : ℝ) : 
  P = 28 → A = 35 → A = r * (P / 2) → r = 2.5 := by
  sorry

end triangle_inradius_l2499_249972


namespace common_difference_is_two_l2499_249929

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  /-- The first term of the sequence -/
  a₁ : ℝ
  /-- The common difference of the sequence -/
  d : ℝ
  /-- The sum of the 4th and 6th terms is 10 -/
  sum_4_6 : a₁ + 3*d + (a₁ + 5*d) = 10
  /-- The sum of the first 5 terms is 5 -/
  sum_5 : 5*a₁ + 10*d = 5

/-- The common difference of the arithmetic sequence is 2 -/
theorem common_difference_is_two (seq : ArithmeticSequence) : seq.d = 2 := by
  sorry

end common_difference_is_two_l2499_249929


namespace gas_station_candy_boxes_l2499_249935

theorem gas_station_candy_boxes : 
  let chocolate : Real := 3.5
  let sugar : Real := 5.25
  let gum : Real := 2.75
  let licorice : Real := 4.5
  let sour : Real := 7.125
  chocolate + sugar + gum + licorice + sour = 23.125 := by
  sorry

end gas_station_candy_boxes_l2499_249935


namespace bus_cost_a_to_b_l2499_249917

/-- The cost to travel by bus between two points -/
def busCost (distance : ℝ) (costPerKm : ℝ) : ℝ :=
  distance * costPerKm

/-- Theorem: The cost to travel by bus from A to B is $900 -/
theorem bus_cost_a_to_b : 
  let distanceAB : ℝ := 4500
  let busCostPerKm : ℝ := 0.20
  busCost distanceAB busCostPerKm = 900 := by sorry

end bus_cost_a_to_b_l2499_249917


namespace inequality_proof_l2499_249975

theorem inequality_proof (a b c : ℝ) 
  (ha : a = Real.tan (23 * π / 180) / (1 - Real.tan (23 * π / 180)^2))
  (hb : b = 2 * Real.sin (13 * π / 180) * Real.cos (13 * π / 180))
  (hc : c = Real.sqrt ((1 - Real.cos (50 * π / 180)) / 2)) :
  c < b ∧ b < a :=
sorry

end inequality_proof_l2499_249975


namespace sin_30_degrees_l2499_249903

theorem sin_30_degrees : Real.sin (30 * π / 180) = 1 / 2 := by sorry

end sin_30_degrees_l2499_249903


namespace alvin_marbles_lost_l2499_249923

/-- Proves that Alvin lost 18 marbles in the first game -/
theorem alvin_marbles_lost (initial_marbles : ℕ) (won_marbles : ℕ) (final_marbles : ℕ) 
  (h1 : initial_marbles = 57)
  (h2 : won_marbles = 25)
  (h3 : final_marbles = 64) :
  initial_marbles - (final_marbles - won_marbles) = 18 := by
  sorry

#check alvin_marbles_lost

end alvin_marbles_lost_l2499_249923


namespace quadratic_other_x_intercept_l2499_249951

/-- Given a quadratic function f(x) = ax² + bx + c with vertex (5,9) and
    one x-intercept at (0,0), prove that the x-coordinate of the other x-intercept is 10. -/
theorem quadratic_other_x_intercept
  (a b c : ℝ)
  (f : ℝ → ℝ)
  (h1 : ∀ x, f x = a * x^2 + b * x + c)
  (h2 : f 5 = 9 ∧ (∀ y, f 5 ≤ f y))  -- Vertex at (5,9)
  (h3 : f 0 = 0)  -- x-intercept at (0,0)
  : ∃ x, x ≠ 0 ∧ f x = 0 ∧ x = 10 :=
by sorry

end quadratic_other_x_intercept_l2499_249951


namespace circle_center_and_radius_l2499_249983

/-- Given a circle with equation x^2 + y^2 - 2x + 6y + 6 = 0, prove that its center is at (1, -3) and its radius is 2 -/
theorem circle_center_and_radius :
  let circle_eq := fun (x y : ℝ) => x^2 + y^2 - 2*x + 6*y + 6 = 0
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (1, -3) ∧ 
    radius = 2 ∧
    ∀ (x y : ℝ), circle_eq x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
by sorry

end circle_center_and_radius_l2499_249983


namespace child_b_share_l2499_249959

def total_money : ℕ := 12000
def num_children : ℕ := 5
def ratio : List ℕ := [2, 3, 4, 5, 6]

theorem child_b_share :
  let total_parts := ratio.sum
  let part_value := total_money / total_parts
  let child_b_parts := ratio[1]
  child_b_parts * part_value = 1800 := by sorry

end child_b_share_l2499_249959


namespace calculate_expression_l2499_249911

theorem calculate_expression : 5 * 7 + 9 * 4 - 36 / 3 = 59 := by
  sorry

end calculate_expression_l2499_249911


namespace f_extrema_on_3_to_5_f_extrema_on_neg1_to_3_l2499_249941

-- Define the function f
def f (x : ℝ) := x^2 - 4*x + 3

-- Theorem for the first interval [3, 5]
theorem f_extrema_on_3_to_5 :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc 3 5, f x ≤ max) ∧
    (∃ x ∈ Set.Icc 3 5, f x = max) ∧
    (∀ x ∈ Set.Icc 3 5, min ≤ f x) ∧
    (∃ x ∈ Set.Icc 3 5, f x = min) ∧
    max = 8 ∧ min = 0 :=
sorry

-- Theorem for the second interval [-1, 3]
theorem f_extrema_on_neg1_to_3 :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc (-1) 3, f x ≤ max) ∧
    (∃ x ∈ Set.Icc (-1) 3, f x = max) ∧
    (∀ x ∈ Set.Icc (-1) 3, min ≤ f x) ∧
    (∃ x ∈ Set.Icc (-1) 3, f x = min) ∧
    max = 8 ∧ min = -1 :=
sorry

end f_extrema_on_3_to_5_f_extrema_on_neg1_to_3_l2499_249941


namespace longest_segment_in_cylinder_l2499_249992

/-- The longest segment in a cylinder -/
theorem longest_segment_in_cylinder (r h : ℝ) (hr : r = 5) (hh : h = 12) :
  Real.sqrt ((2 * r) ^ 2 + h ^ 2) = Real.sqrt 244 := by
  sorry

end longest_segment_in_cylinder_l2499_249992


namespace farm_field_theorem_l2499_249937

/-- Represents the farm field ploughing problem -/
structure FarmField where
  planned_hectares_per_day : ℕ
  actual_hectares_per_day : ℕ
  technical_delay_days : ℕ
  weather_delay_days : ℕ
  remaining_hectares : ℕ

/-- The solution to the farm field problem -/
def farm_field_solution (f : FarmField) : ℕ × ℕ :=
  let total_area := 1560
  let planned_days := 13
  (total_area, planned_days)

/-- Theorem stating the correctness of the farm field solution -/
theorem farm_field_theorem (f : FarmField)
    (h1 : f.planned_hectares_per_day = 120)
    (h2 : f.actual_hectares_per_day = 85)
    (h3 : f.technical_delay_days = 3)
    (h4 : f.weather_delay_days = 2)
    (h5 : f.remaining_hectares = 40) :
    farm_field_solution f = (1560, 13) := by
  sorry

#check farm_field_theorem

end farm_field_theorem_l2499_249937


namespace division_problem_l2499_249976

theorem division_problem (dividend : ℤ) (divisor : ℤ) (remainder : ℤ) (quotient : ℤ) :
  dividend = 12 →
  divisor = 17 →
  remainder = 8 →
  dividend = divisor * quotient + remainder →
  quotient = 0 := by
sorry

end division_problem_l2499_249976


namespace cicely_hundredth_birthday_l2499_249921

def cicely_birthday_problem (birth_year : ℕ) (twenty_first_year : ℕ) (hundredth_year : ℕ) : Prop :=
  (twenty_first_year - birth_year = 21) ∧ 
  (twenty_first_year = 1939) ∧
  (hundredth_year - birth_year = 100)

theorem cicely_hundredth_birthday : 
  ∃ (birth_year : ℕ), cicely_birthday_problem birth_year 1939 2018 := by
  sorry

end cicely_hundredth_birthday_l2499_249921


namespace possible_distances_l2499_249901

/-- Represents the position of a house on a street. -/
structure House where
  position : ℝ

/-- Represents a street with four houses. -/
structure Street where
  andrey : House
  boris : House
  vova : House
  gleb : House

/-- The distance between two houses. -/
def distance (h1 h2 : House) : ℝ :=
  |h1.position - h2.position|

/-- A street is valid if it satisfies the given conditions. -/
def validStreet (s : Street) : Prop :=
  distance s.andrey s.boris = 600 ∧
  distance s.vova s.gleb = 600 ∧
  distance s.andrey s.gleb = 3 * distance s.boris s.vova

/-- The theorem stating the possible distances between Andrey's and Gleb's houses. -/
theorem possible_distances (s : Street) (h : validStreet s) :
  distance s.andrey s.gleb = 900 ∨ distance s.andrey s.gleb = 1800 :=
sorry

end possible_distances_l2499_249901


namespace solve_sleep_problem_l2499_249971

def sleep_problem (connor_sleep : ℕ) : Prop :=
  let luke_sleep := connor_sleep + 2
  let emma_sleep := connor_sleep - 1
  let puppy_sleep := luke_sleep * 2
  connor_sleep = 6 →
  connor_sleep + luke_sleep + emma_sleep + puppy_sleep = 35

theorem solve_sleep_problem :
  sleep_problem 6 := by
  sorry

end solve_sleep_problem_l2499_249971


namespace equation_is_quadratic_l2499_249944

/-- A quadratic equation is of the form ax² + bx + c = 0, where a ≠ 0 --/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x² + 2x = 0 is a quadratic equation --/
theorem equation_is_quadratic :
  is_quadratic_equation (λ x => x^2 + 2*x) :=
sorry

end equation_is_quadratic_l2499_249944


namespace thirteen_sided_polygon_property_n_sided_polygon_property_l2499_249985

-- Define a polygon type
structure Polygon :=
  (sides : ℕ)
  (vertices : Fin sides → ℝ × ℝ)

-- Define a line type
structure Line :=
  (a b c : ℝ)

-- Function to check if a line contains a side of a polygon
def line_contains_side (l : Line) (p : Polygon) (i : Fin p.sides) : Prop :=
  -- Implementation details omitted
  sorry

-- Function to count how many sides of a polygon a line contains
def count_sides_on_line (l : Line) (p : Polygon) : ℕ :=
  -- Implementation details omitted
  sorry

-- Theorem for 13-sided polygons
theorem thirteen_sided_polygon_property :
  ∀ (p : Polygon), p.sides = 13 →
  ∃ (l : Line), ∃ (i : Fin p.sides),
    line_contains_side l p i ∧
    count_sides_on_line l p = 1 :=
sorry

-- Theorem for n-sided polygons where n > 13
theorem n_sided_polygon_property :
  ∀ (n : ℕ), n > 13 →
  ∃ (p : Polygon), p.sides = n ∧
  ∀ (l : Line), ∀ (i : Fin p.sides),
    line_contains_side l p i →
    count_sides_on_line l p ≥ 2 :=
sorry

end thirteen_sided_polygon_property_n_sided_polygon_property_l2499_249985


namespace tims_weekly_water_consumption_l2499_249939

/-- Calculates Tim's weekly water consumption in ounces -/
theorem tims_weekly_water_consumption :
  let quart_to_oz : ℚ → ℚ := (· * 32)
  let daily_bottle_oz := 2 * quart_to_oz 1.5
  let daily_total_oz := daily_bottle_oz + 20
  let weekly_oz := 7 * daily_total_oz
  weekly_oz = 812 := by sorry

end tims_weekly_water_consumption_l2499_249939


namespace opposite_of_negative_two_thirds_l2499_249932

theorem opposite_of_negative_two_thirds :
  -(-(2/3)) = 2/3 := by sorry

end opposite_of_negative_two_thirds_l2499_249932


namespace arithmetic_operations_l2499_249960

theorem arithmetic_operations : 
  (100 - 54 - 46 = 0) ∧ 
  (234 - (134 + 45) = 55) ∧ 
  (125 * 7 * 8 = 7000) ∧ 
  (15 * (61 - 45) = 240) ∧ 
  (318 / 6 + 165 = 218) := by
  sorry

end arithmetic_operations_l2499_249960


namespace proposition_analysis_l2499_249958

theorem proposition_analysis (P Q : Prop) 
  (h_P : P ↔ (2 + 2 = 5))
  (h_Q : Q ↔ (3 > 2)) : 
  (¬(P ∧ Q)) ∧ (¬P) := by
  sorry

end proposition_analysis_l2499_249958


namespace half_angle_quadrants_l2499_249943

/-- An angle is in the 4th quadrant if it's between 3π/2 and 2π (exclusive) -/
def in_fourth_quadrant (α : ℝ) : Prop :=
  3 * Real.pi / 2 < α ∧ α < 2 * Real.pi

/-- An angle is in the 2nd quadrant if it's between π/2 and π (exclusive) -/
def in_second_quadrant (α : ℝ) : Prop :=
  Real.pi / 2 < α ∧ α < Real.pi

/-- An angle is in the 4th quadrant if it's between 3π/2 and 2π (exclusive) -/
def in_fourth_quadrant_half (α : ℝ) : Prop :=
  3 * Real.pi / 2 < α ∧ α < 2 * Real.pi

theorem half_angle_quadrants (α : ℝ) :
  in_fourth_quadrant α →
  (in_second_quadrant (α/2) ∨ in_fourth_quadrant_half (α/2)) :=
by sorry

end half_angle_quadrants_l2499_249943


namespace largest_fraction_equal_digit_sums_l2499_249979

/-- Sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Check if a number is four-digit -/
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

/-- The largest fraction with equal digit sums -/
theorem largest_fraction_equal_digit_sums :
  ∀ m n : ℕ, 
    is_four_digit m → 
    is_four_digit n → 
    digit_sum m = digit_sum n → 
    (m : ℚ) / n ≤ 9900 / 1089 := by
  sorry

end largest_fraction_equal_digit_sums_l2499_249979


namespace eleven_row_triangle_pieces_l2499_249900

/-- Calculates the total number of pieces in a triangle with given number of rows -/
def totalPieces (rows : ℕ) : ℕ :=
  let rodSum := (rows * (rows + 1) * 3) / 2
  let connectorSum := (rows + 1) * (rows + 2) / 2
  rodSum + connectorSum

/-- Theorem stating that an eleven-row triangle has 276 pieces -/
theorem eleven_row_triangle_pieces :
  totalPieces 11 = 276 := by sorry

end eleven_row_triangle_pieces_l2499_249900


namespace company_workforce_l2499_249924

theorem company_workforce (initial_employees : ℕ) : 
  (initial_employees * 6 / 10 : ℚ) = (initial_employees + 20) * 11 / 20 →
  initial_employees + 20 = 240 := by
  sorry

end company_workforce_l2499_249924


namespace valid_lineup_count_l2499_249963

/- Define the total number of players -/
def total_players : ℕ := 18

/- Define the number of quadruplets -/
def quadruplets : ℕ := 4

/- Define the number of starters to select -/
def starters : ℕ := 8

/- Define the function to calculate combinations -/
def combination (n k : ℕ) : ℕ := (Nat.choose n k)

/- Theorem statement -/
theorem valid_lineup_count :
  combination total_players starters - combination (total_players - quadruplets) (starters - quadruplets) =
  42757 := by sorry

end valid_lineup_count_l2499_249963


namespace henry_chore_earnings_l2499_249997

theorem henry_chore_earnings : ∃ (earned : ℕ), 
  (5 + earned + 13 = 20) ∧ earned = 2 := by
  sorry

end henry_chore_earnings_l2499_249997


namespace no_non_divisor_exists_l2499_249936

theorem no_non_divisor_exists (a : ℕ+) : ∃ (b n : ℕ+), a.val ∣ (b.val ^ n.val - n.val) := by
  sorry

end no_non_divisor_exists_l2499_249936
