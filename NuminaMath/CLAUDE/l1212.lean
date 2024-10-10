import Mathlib

namespace smallest_integer_satisfying_inequality_l1212_121208

theorem smallest_integer_satisfying_inequality : 
  ∃ x : ℤ, (2 * x : ℚ) / 5 + 3 / 4 > 7 / 5 ∧ 
  ∀ y : ℤ, y < x → (2 * y : ℚ) / 5 + 3 / 4 ≤ 7 / 5 :=
by
  use 2
  sorry

end smallest_integer_satisfying_inequality_l1212_121208


namespace smaller_number_problem_l1212_121254

theorem smaller_number_problem (x y : ℕ) 
  (h1 : y - x = 1365)
  (h2 : y = 6 * x + 15) : 
  x = 270 := by
sorry

end smaller_number_problem_l1212_121254


namespace triangle_properties_l1212_121276

/-- Triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition in the problem -/
def condition (t : Triangle) : Prop :=
  (Real.cos t.A - 2 * Real.cos t.C) / Real.cos t.B = (2 * t.c - t.a) / t.b

theorem triangle_properties (t : Triangle) (h : condition t) :
  Real.sin t.C / Real.sin t.A = 2 ∧
  (Real.cos t.B = 1/4 ∧ t.b = 2 → 
    1/2 * t.a * t.c * Real.sin t.B = Real.sqrt 15 / 4) :=
by sorry

end triangle_properties_l1212_121276


namespace range_of_m_satisfies_conditions_l1212_121211

/-- Given two functions f and g, prove that the range of m satisfies the given conditions -/
theorem range_of_m_satisfies_conditions (f g : ℝ → ℝ) (m : ℝ) : 
  (∀ x, f x = x^2 + m) →
  (∀ x, g x = 2^x - m) →
  (∀ x₁ ∈ Set.Icc (-1) 2, ∃ x₂ ∈ Set.Icc 0 3, f x₁ = g x₂) →
  m ∈ Set.Icc (1/2) 2 := by
  sorry

#check range_of_m_satisfies_conditions

end range_of_m_satisfies_conditions_l1212_121211


namespace quadratic_equation_constant_term_l1212_121290

theorem quadratic_equation_constant_term (m : ℝ) : 
  (∀ x, (m - 2) * x^2 + 3 * x + m^2 - 4 = 0) → 
  m^2 - 4 = 0 → 
  m - 2 ≠ 0 → 
  m = -2 := by sorry

end quadratic_equation_constant_term_l1212_121290


namespace alpha_value_l1212_121259

theorem alpha_value (α β γ : Real) 
  (h1 : 0 < α ∧ α < π)
  (h2 : α + β + γ = π)
  (h3 : 2 * Real.sin α + Real.tan β + Real.tan γ = 2 * Real.sin α * Real.tan β * Real.tan γ) :
  α = π / 3 := by
  sorry

end alpha_value_l1212_121259


namespace bd_equals_twelve_l1212_121273

/-- Represents a quadrilateral ABCD with given side lengths and diagonal BD --/
structure Quadrilateral where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  DA : ℝ
  BD : ℤ

/-- Theorem stating that BD = 12 is a valid solution for the given quadrilateral --/
theorem bd_equals_twelve (q : Quadrilateral) 
  (h1 : q.AB = 6)
  (h2 : q.BC = 12)
  (h3 : q.CD = 6)
  (h4 : q.DA = 8) :
  q.BD = 12 → 
  (q.AB + q.BD > q.DA) ∧ 
  (q.BC + q.CD > q.BD) ∧ 
  (q.DA + q.BD > q.AB) ∧ 
  (q.BD + q.CD > q.BC) ∧ 
  (q.BD > 6) ∧ 
  (q.BD < 14) := by
  sorry

#check bd_equals_twelve

end bd_equals_twelve_l1212_121273


namespace quadratic_function_range_l1212_121280

theorem quadratic_function_range (a : ℝ) : 
  (∃ y₁ y₂ y₃ y₄ : ℝ, 
    (y₁ = a * (-4)^2 + 4 * a * (-4) - 6) ∧
    (y₂ = a * (-3)^2 + 4 * a * (-3) - 6) ∧
    (y₃ = a * 0^2 + 4 * a * 0 - 6) ∧
    (y₄ = a * 2^2 + 4 * a * 2 - 6) ∧
    ((y₁ > 0 ∧ y₂ ≤ 0 ∧ y₃ ≤ 0 ∧ y₄ ≤ 0) ∨
     (y₁ ≤ 0 ∧ y₂ > 0 ∧ y₃ ≤ 0 ∧ y₄ ≤ 0) ∨
     (y₁ ≤ 0 ∧ y₂ ≤ 0 ∧ y₃ > 0 ∧ y₄ ≤ 0) ∨
     (y₁ ≤ 0 ∧ y₂ ≤ 0 ∧ y₃ ≤ 0 ∧ y₄ > 0))) →
  (a < -2 ∨ a > 1/2) := by
  sorry

end quadratic_function_range_l1212_121280


namespace proposition_two_l1212_121243

theorem proposition_two (a b : ℝ) : a > b → ((1 / a < 1 / b) ↔ (a * b > 0)) := by
  sorry

end proposition_two_l1212_121243


namespace charcoal_drawings_count_l1212_121270

theorem charcoal_drawings_count (total : ℕ) (colored_pencil : ℕ) (blending_marker : ℕ) 
  (h1 : total = 25)
  (h2 : colored_pencil = 14)
  (h3 : blending_marker = 7) :
  total - (colored_pencil + blending_marker) = 4 := by
  sorry

end charcoal_drawings_count_l1212_121270


namespace max_diff_reversed_digits_l1212_121219

/-- Given two three-digit positive integers with the same digits in reverse order,
    prove that their maximum difference less than 300 is 297. -/
theorem max_diff_reversed_digits (q r : ℕ) : 
  (100 ≤ q) ∧ (q < 1000) ∧  -- q is a three-digit number
  (100 ≤ r) ∧ (r < 1000) ∧  -- r is a three-digit number
  (∃ a b c : ℕ, q = 100*a + 10*b + c ∧ r = 100*c + 10*b + a) ∧  -- q and r have reversed digits
  (q > r) ∧  -- ensure q is greater than r
  (q - r < 300) →  -- difference is less than 300
  (q - r ≤ 297) ∧ (∃ q' r' : ℕ, q' - r' = 297 ∧ 
    (100 ≤ q') ∧ (q' < 1000) ∧ (100 ≤ r') ∧ (r' < 1000) ∧
    (∃ a b c : ℕ, q' = 100*a + 10*b + c ∧ r' = 100*c + 10*b + a) ∧
    (q' > r') ∧ (q' - r' < 300)) := by
  sorry

end max_diff_reversed_digits_l1212_121219


namespace l_shape_area_is_58_l1212_121245

/-- The area of an "L" shaped figure formed by removing a smaller rectangle from a larger rectangle -/
def l_shape_area (large_width large_height small_width small_height : ℕ) : ℕ :=
  large_width * large_height - small_width * small_height

/-- Theorem: The area of the "L" shaped figure is 58 square units -/
theorem l_shape_area_is_58 :
  l_shape_area 10 7 4 3 = 58 := by
  sorry

#eval l_shape_area 10 7 4 3

end l_shape_area_is_58_l1212_121245


namespace tangent_line_is_correct_l1212_121265

/-- The equation of a parabola -/
def parabola (x : ℝ) : ℝ := 4 * x^2

/-- The slope of the tangent line at a given x-coordinate -/
def tangent_slope (x : ℝ) : ℝ := 8 * x

/-- The point of tangency -/
def point_of_tangency : ℝ × ℝ := (1, 4)

/-- The proposed equation of the tangent line -/
def tangent_line (x y : ℝ) : Prop := 8 * x - y - 4 = 0

theorem tangent_line_is_correct :
  let (x₀, y₀) := point_of_tangency
  tangent_line x₀ y₀ ∧
  y₀ = parabola x₀ ∧
  (∀ x y, tangent_line x y ↔ y - y₀ = tangent_slope x₀ * (x - x₀)) :=
sorry

end tangent_line_is_correct_l1212_121265


namespace x_range_for_quadratic_inequality_l1212_121233

theorem x_range_for_quadratic_inequality :
  ∀ x : ℝ, (∀ a : ℝ, a ∈ Set.Icc (-3) 3 → x^2 - a*x + 1 ≥ 1) →
  x ≥ (3 + Real.sqrt 5) / 2 ∨ x ≤ (-3 - Real.sqrt 5) / 2 := by
sorry

end x_range_for_quadratic_inequality_l1212_121233


namespace opposite_sides_iff_in_set_l1212_121210

/-- The set of real numbers a for which points A and B lie on opposite sides of the line 3x - y = 4 -/
def opposite_sides_set : Set ℝ :=
  {a | a < -1 ∨ (-1/3 < a ∧ a < 0) ∨ a > 8/3}

/-- Point A coordinates satisfy the given equation -/
def point_A (a x y : ℝ) : Prop :=
  26 * a^2 - 22 * a * x - 20 * a * y + 5 * x^2 + 8 * x * y + 4 * y^2 = 0

/-- Parabola equation with vertex at point B -/
def parabola (a x y : ℝ) : Prop :=
  a * x^2 + 2 * a^2 * x - a * y + a^3 + 1 = 0

/-- Line equation -/
def line (x y : ℝ) : Prop :=
  3 * x - y = 4

/-- Main theorem: A and B lie on opposite sides of the line if and only if a is in the opposite_sides_set -/
theorem opposite_sides_iff_in_set (a : ℝ) :
  (∃ x_a y_a x_b y_b : ℝ,
    point_A a x_a y_a ∧
    parabola a x_b y_b ∧
    ¬line x_a y_a ∧
    ¬line x_b y_b ∧
    (3 * x_a - y_a - 4) * (3 * x_b - y_b - 4) < 0) ↔
  a ∈ opposite_sides_set := by
  sorry

end opposite_sides_iff_in_set_l1212_121210


namespace B_power_103_l1212_121287

def B : Matrix (Fin 3) (Fin 3) ℝ :=
  !![0, 1, 0;
     0, 0, 1;
     1, 0, 0]

theorem B_power_103 : B^103 = B := by sorry

end B_power_103_l1212_121287


namespace geometric_sequence_minimum_ratio_l1212_121298

theorem geometric_sequence_minimum_ratio :
  ∀ (a : ℕ → ℕ) (q : ℚ),
  (∀ n : ℕ, 1 ≤ n → n < 2016 → a (n + 1) = a n * q) →
  (1 < q ∧ q < 2) →
  (∀ r : ℚ, 1 < r ∧ r < 2 → a 2016 ≤ a 1 * r^2015) →
  q = 6/5 := by
sorry

end geometric_sequence_minimum_ratio_l1212_121298


namespace oranges_left_l1212_121232

/-- The number of oranges originally in the basket -/
def original_oranges : ℕ := 8

/-- The number of oranges taken from the basket -/
def oranges_taken : ℕ := 5

/-- Theorem: The number of oranges left in the basket is 3 -/
theorem oranges_left : original_oranges - oranges_taken = 3 := by
  sorry

end oranges_left_l1212_121232


namespace infinitely_many_linear_combinations_l1212_121261

/-- An infinite sequence of positive integers with strictly increasing terms. -/
def StrictlyIncreasingSequence (a : ℕ → ℕ) : Prop :=
  ∀ k, 0 < a k ∧ a k < a (k + 1)

/-- The property that infinitely many terms can be expressed as a linear combination of two earlier terms. -/
def InfinitelyManyLinearCombinations (a : ℕ → ℕ) : Prop :=
  ∀ N, ∃ m p q x y, N < m ∧ p ≠ q ∧ 0 < x ∧ 0 < y ∧ a m = x * a p + y * a q

/-- The main theorem: any strictly increasing sequence of positive integers has infinitely many terms
    that can be expressed as a linear combination of two earlier terms. -/
theorem infinitely_many_linear_combinations
  (a : ℕ → ℕ) (h : StrictlyIncreasingSequence a) :
  InfinitelyManyLinearCombinations a :=
sorry

end infinitely_many_linear_combinations_l1212_121261


namespace volume_cylinder_from_square_rotation_l1212_121227

/-- The volume of a cylinder formed by rotating a square about its horizontal line of symmetry -/
theorem volume_cylinder_from_square_rotation (side_length : ℝ) (volume : ℝ) :
  side_length = 16 →
  volume = π * side_length^3 / 4 →
  volume = 1024 * π :=
by sorry

end volume_cylinder_from_square_rotation_l1212_121227


namespace complement_intersection_equals_five_l1212_121260

open Set

def U : Set Nat := {1, 2, 3, 4, 5}
def M : Set Nat := {1, 4}
def N : Set Nat := {2, 3}

theorem complement_intersection_equals_five :
  (U \ M) ∩ (U \ N) = {5} := by sorry

end complement_intersection_equals_five_l1212_121260


namespace total_players_count_l1212_121204

/-- The number of players who play kabaddi -/
def kabaddi_players : ℕ := 10

/-- The number of players who play kho-kho only -/
def kho_kho_only_players : ℕ := 20

/-- The number of players who play both games -/
def both_games_players : ℕ := 5

/-- The total number of players -/
def total_players : ℕ := kabaddi_players + kho_kho_only_players - both_games_players

theorem total_players_count : total_players = 25 := by
  sorry

end total_players_count_l1212_121204


namespace ball_arrangements_count_l1212_121217

variable (n : ℕ)

/-- The number of ways to arrange 3n people into circles with ABC pattern -/
def ball_arrangements (n : ℕ) : ℕ := (3 * n).factorial

/-- Theorem: The number of ball arrangements is (3n)! -/
theorem ball_arrangements_count :
  ball_arrangements n = (3 * n).factorial := by
  sorry

end ball_arrangements_count_l1212_121217


namespace handshake_count_l1212_121289

/-- Represents a basketball game setup with two teams and referees -/
structure BasketballGame where
  team_size : Nat
  coach_per_team : Nat
  referee_count : Nat

/-- Calculates the total number of handshakes in a basketball game -/
def total_handshakes (game : BasketballGame) : Nat :=
  let inter_team_handshakes := game.team_size * game.team_size
  let total_team_members := game.team_size + game.coach_per_team
  let intra_team_handshakes := 2 * (total_team_members.choose 2)
  let team_referee_handshakes := 2 * total_team_members * game.referee_count
  let referee_handshakes := game.referee_count.choose 2
  inter_team_handshakes + intra_team_handshakes + team_referee_handshakes + referee_handshakes

/-- The main theorem stating the total number of handshakes in the given game setup -/
theorem handshake_count :
  let game : BasketballGame := {
    team_size := 6
    coach_per_team := 1
    referee_count := 2
  }
  total_handshakes game = 107 := by
  sorry


end handshake_count_l1212_121289


namespace ball_box_problem_l1212_121228

theorem ball_box_problem (num_balls : ℕ) (X : ℕ) (h1 : num_balls = 25) 
  (h2 : num_balls - 20 = X - num_balls) : X = 30 := by
  sorry

end ball_box_problem_l1212_121228


namespace only_vegetarian_count_l1212_121288

/-- Represents the number of people in different dietary categories in a family -/
structure FamilyDiet where
  total_veg : ℕ
  only_nonveg : ℕ
  both : ℕ

/-- Given the specified family diet, prove that the number of people who eat only vegetarian is 13 -/
theorem only_vegetarian_count (f : FamilyDiet) 
  (h1 : f.total_veg = 21)
  (h2 : f.only_nonveg = 7)
  (h3 : f.both = 8) :
  f.total_veg - f.both = 13 := by
  sorry

#check only_vegetarian_count

end only_vegetarian_count_l1212_121288


namespace angies_taxes_paid_l1212_121279

/-- Represents the weekly expenses for necessities, taxes, and utilities -/
structure WeeklyExpenses where
  necessities : ℕ
  taxes : ℕ
  utilities : ℕ

/-- Represents Angie's monthly finances -/
structure MonthlyFinances where
  salary : ℕ
  week1 : WeeklyExpenses
  week2 : WeeklyExpenses
  week3 : WeeklyExpenses
  week4 : WeeklyExpenses
  leftover : ℕ

/-- Calculates the total taxes paid in a month -/
def totalTaxesPaid (finances : MonthlyFinances) : ℕ :=
  finances.week1.taxes + finances.week2.taxes + finances.week3.taxes + finances.week4.taxes

/-- Theorem stating that Angie's total taxes paid for the month is $30 -/
theorem angies_taxes_paid (finances : MonthlyFinances) 
    (h1 : finances.salary = 80)
    (h2 : finances.week1 = ⟨12, 8, 5⟩)
    (h3 : finances.week2 = ⟨15, 6, 7⟩)
    (h4 : finances.week3 = ⟨10, 9, 6⟩)
    (h5 : finances.week4 = ⟨14, 7, 4⟩)
    (h6 : finances.leftover = 18) :
    totalTaxesPaid finances = 30 := by
  sorry

#eval totalTaxesPaid ⟨80, ⟨12, 8, 5⟩, ⟨15, 6, 7⟩, ⟨10, 9, 6⟩, ⟨14, 7, 4⟩, 18⟩

end angies_taxes_paid_l1212_121279


namespace stadium_entry_exit_options_l1212_121237

theorem stadium_entry_exit_options (south_gates north_gates : ℕ) 
  (h1 : south_gates = 4) 
  (h2 : north_gates = 3) : 
  (south_gates + north_gates) * (south_gates + north_gates) = 49 := by
  sorry

end stadium_entry_exit_options_l1212_121237


namespace line_mb_value_l1212_121209

/-- Proves that for a line y = mx + b passing through points (0, -2) and (1, 1), mb = -10 -/
theorem line_mb_value (m b : ℝ) : 
  (∀ x y : ℝ, y = m * x + b) → -- Line equation
  (-2 : ℝ) = b →               -- y-intercept
  (1 : ℝ) = m * 1 + b →        -- Point (1, 1) satisfies the equation
  m * b = -10 := by
  sorry

end line_mb_value_l1212_121209


namespace can_add_flights_to_5000_l1212_121231

/-- A graph representing cities and flights --/
structure CityGraph where
  vertices : Finset Nat
  edges : Finset (Nat × Nat)
  edge_symmetric : ∀ {a b}, (a, b) ∈ edges → (b, a) ∈ edges
  no_self_loops : ∀ {a}, (a, a) ∉ edges

/-- The number of cities --/
def num_cities : Nat := 998

/-- Check if the graph satisfies the flight laws --/
def satisfies_laws (g : CityGraph) : Prop :=
  (g.vertices.card = num_cities) ∧
  (∀ k : Finset Nat, k ⊆ g.vertices →
    (g.edges.filter (fun e => e.1 ∈ k ∧ e.2 ∈ k)).card ≤ 5 * k.card + 10)

/-- The theorem to be proved --/
theorem can_add_flights_to_5000 (g : CityGraph) (h : satisfies_laws g) :
  ∃ g' : CityGraph, satisfies_laws g' ∧
    g.edges ⊆ g'.edges ∧
    g'.edges.card = 5000 := by
  sorry

end can_add_flights_to_5000_l1212_121231


namespace river_crossing_drift_l1212_121213

/-- Given a river crossing scenario, calculate the drift of the boat. -/
theorem river_crossing_drift (river_width : ℝ) (boat_speed : ℝ) (crossing_time : ℝ) 
  (h1 : river_width = 400)
  (h2 : boat_speed = 10)
  (h3 : crossing_time = 50) :
  boat_speed * crossing_time - river_width = 100 := by
  sorry

#check river_crossing_drift

end river_crossing_drift_l1212_121213


namespace sqrt_sum_equals_sqrt_60_l1212_121284

theorem sqrt_sum_equals_sqrt_60 :
  Real.sqrt (25 - 10 * Real.sqrt 6) + Real.sqrt (25 + 10 * Real.sqrt 6) = Real.sqrt 60 := by
  sorry

end sqrt_sum_equals_sqrt_60_l1212_121284


namespace simplify_expression_l1212_121223

theorem simplify_expression (x : ℝ) : 120*x - 32*x + 15 - 15 = 88*x := by
  sorry

end simplify_expression_l1212_121223


namespace quadratic_equation_roots_l1212_121278

theorem quadratic_equation_roots (p : ℝ) : 
  (∃ x₁ x₂ : ℂ, x₁ ≠ x₂ ∧ 
   (∀ x : ℂ, x^2 - p*x + 1 = 0 ↔ (x = x₁ ∨ x = x₂)) ∧
   x₁.im ≠ 0 ∧ x₂.im ≠ 0 ∧
   Complex.abs (x₁ - x₂) = 1) →
  p = Real.sqrt 3 ∨ p = -Real.sqrt 3 :=
by sorry

end quadratic_equation_roots_l1212_121278


namespace range_of_x_when_a_is_1_range_of_a_when_not_p_implies_not_q_l1212_121203

-- Define the propositions p and q
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∧ x^2 + 2*x - 8 > 0

-- Part 1
theorem range_of_x_when_a_is_1 :
  ∀ x : ℝ, p x 1 ∧ q x → 2 < x ∧ x < 3 :=
sorry

-- Part 2
theorem range_of_a_when_not_p_implies_not_q :
  (∀ x a : ℝ, ¬(p x a) → ¬(q x)) ∧ 
  (∃ x a : ℝ, ¬(q x) ∧ p x a) →
  ∀ a : ℝ, 1 < a ∧ a ≤ 2 :=
sorry

end range_of_x_when_a_is_1_range_of_a_when_not_p_implies_not_q_l1212_121203


namespace number_of_sailors_l1212_121274

theorem number_of_sailors (W : ℝ) (n : ℕ) : 
  (W + 64 - 56) / n = W / n + 1 → n = 8 := by
  sorry

end number_of_sailors_l1212_121274


namespace guppy_ratio_l1212_121249

/-- The number of guppies Haylee has -/
def hayleeGuppies : ℕ := 36

/-- The number of guppies Jose has -/
def joseGuppies : ℕ := hayleeGuppies / 2

/-- The number of guppies Charliz has -/
def charlizGuppies : ℕ := 6

/-- The number of guppies Nicolai has -/
def nicolaiGuppies : ℕ := 4 * charlizGuppies

/-- The total number of guppies all four friends have -/
def totalGuppies : ℕ := 84

/-- Theorem stating that the ratio of Charliz's guppies to Jose's guppies is 1:3 -/
theorem guppy_ratio :
  charlizGuppies * 3 = joseGuppies ∧
  hayleeGuppies + joseGuppies + charlizGuppies + nicolaiGuppies = totalGuppies :=
by sorry

end guppy_ratio_l1212_121249


namespace imaginary_part_of_z_l1212_121292

theorem imaginary_part_of_z (i : ℂ) (h : i^2 = -1) : 
  Complex.im (i^2 * (1 + i)) = -1 := by
  sorry

end imaginary_part_of_z_l1212_121292


namespace algebraic_expression_evaluation_l1212_121257

theorem algebraic_expression_evaluation :
  let x : ℝ := 2 - Real.sqrt 3
  (7 + 4 * Real.sqrt 3) * x^2 - (2 + Real.sqrt 3) * x + Real.sqrt 3 = 2 + Real.sqrt 3 := by
  sorry

end algebraic_expression_evaluation_l1212_121257


namespace unique_prime_triple_l1212_121224

theorem unique_prime_triple : ∃! (I M C : ℕ),
  (Nat.Prime I ∧ Nat.Prime M ∧ Nat.Prime C) ∧
  (I ≤ M ∧ M ≤ C) ∧
  (I * M * C = I + M + C + 1007) ∧
  I = 2 ∧ M = 2 ∧ C = 337 := by
sorry

end unique_prime_triple_l1212_121224


namespace remaining_area_of_semicircle_l1212_121268

theorem remaining_area_of_semicircle (d : ℝ) (h : d > 0) :
  let r := d / 2
  let chord_length := 2 * Real.sqrt 7
  chord_length ^ 2 + r ^ 2 = d ^ 2 →
  (π * r ^ 2 / 2) - 2 * (π * (r / 2) ^ 2 / 2) = 7 * π :=
by sorry

end remaining_area_of_semicircle_l1212_121268


namespace root_difference_ratio_l1212_121262

theorem root_difference_ratio (a b : ℝ) : 
  a^4 - 7*a - 3 = 0 → 
  b^4 - 7*b - 3 = 0 → 
  a > b → 
  (a - b) / (a^4 - b^4) = 1/7 := by
sorry

end root_difference_ratio_l1212_121262


namespace angle_from_coordinates_l1212_121253

theorem angle_from_coordinates (α : Real) 
  (h1 : α > 0) (h2 : α < 2 * Real.pi)
  (h3 : ∃ (x y : Real), x = Real.sin (Real.pi / 6) ∧ 
                        y = Real.cos (5 * Real.pi / 6) ∧
                        x = Real.sin α ∧
                        y = Real.cos α) :
  α = 5 * Real.pi / 3 := by
sorry

end angle_from_coordinates_l1212_121253


namespace james_vote_percentage_l1212_121266

theorem james_vote_percentage (total_votes : ℕ) (john_votes : ℕ) (third_candidate_extra : ℕ) :
  total_votes = 1150 →
  john_votes = 150 →
  third_candidate_extra = 150 →
  let third_candidate_votes := john_votes + third_candidate_extra
  let remaining_votes := total_votes - john_votes
  let james_votes := total_votes - (john_votes + third_candidate_votes)
  james_votes / remaining_votes = 7 / 10 := by
  sorry

#check james_vote_percentage

end james_vote_percentage_l1212_121266


namespace school_camp_buses_l1212_121277

theorem school_camp_buses (B : ℕ) (S : ℕ) : 
  B ≤ 18 ∧                           -- No more than 18 buses
  S = 22 * B + 3 ∧                   -- Initial distribution with 3 left out
  ∃ (n : ℕ), n ≤ 36 ∧                -- Each bus can hold up to 36 people
  S = n * (B - 1) ∧                  -- Even distribution after one bus leaves
  n = (22 * B + 3) / (B - 1) →       -- Relationship between n, B, and S
  S = 355 :=
by sorry

end school_camp_buses_l1212_121277


namespace mean_of_scores_l1212_121296

def scores : List ℝ := [69, 68, 70, 61, 74, 62, 65, 74]

theorem mean_of_scores :
  (scores.sum / scores.length : ℝ) = 67.875 := by
  sorry

end mean_of_scores_l1212_121296


namespace sqrt_seven_to_sixth_l1212_121285

theorem sqrt_seven_to_sixth : (Real.sqrt 7) ^ 6 = 343 := by sorry

end sqrt_seven_to_sixth_l1212_121285


namespace coin_draw_probability_l1212_121206

def penny_count : ℕ := 3
def nickel_count : ℕ := 3
def quarter_count : ℕ := 6
def dime_count : ℕ := 3
def total_coins : ℕ := penny_count + nickel_count + quarter_count + dime_count
def drawn_coins : ℕ := 8
def min_value : ℚ := 175/100

def successful_outcomes : ℕ := 9
def total_outcomes : ℕ := Nat.choose total_coins drawn_coins

theorem coin_draw_probability :
  (successful_outcomes : ℚ) / total_outcomes = 9 / 6435 :=
sorry

end coin_draw_probability_l1212_121206


namespace price_restoration_l1212_121218

theorem price_restoration (original_price : ℝ) (h : original_price > 0) :
  let reduced_price := 0.8 * original_price
  (reduced_price * 1.25 = original_price) := by sorry

end price_restoration_l1212_121218


namespace intersection_implies_a_zero_l1212_121220

def A : Set ℝ := {0, 1, 3}
def B (a : ℝ) : Set ℝ := {a + 1, a^2 + 2}

theorem intersection_implies_a_zero (a : ℝ) : A ∩ B a = {1} → a = 0 := by
  sorry

end intersection_implies_a_zero_l1212_121220


namespace correct_fraction_proof_l1212_121271

theorem correct_fraction_proof (x y : ℚ) : 
  (5 : ℚ) / 6 * 384 = x / y * 384 + 200 → x / y = (5 : ℚ) / 16 := by
sorry

end correct_fraction_proof_l1212_121271


namespace eva_total_score_2019_l1212_121258

/-- Represents Eva's scores in a semester -/
structure SemesterScores where
  maths : ℕ
  arts : ℕ
  science : ℕ

/-- Calculates the total score for a semester -/
def totalScore (scores : SemesterScores) : ℕ :=
  scores.maths + scores.arts + scores.science

/-- Represents Eva's scores for the year 2019 -/
structure YearScores where
  firstSemester : SemesterScores
  secondSemester : SemesterScores

/-- Theorem stating Eva's total score for 2019 -/
theorem eva_total_score_2019 (scores : YearScores) : 
  totalScore scores.firstSemester + totalScore scores.secondSemester = 485 :=
  by
  have h1 : scores.firstSemester.maths = scores.secondSemester.maths + 10 := by sorry
  have h2 : scores.firstSemester.arts = scores.secondSemester.arts - 15 := by sorry
  have h3 : scores.firstSemester.science = scores.secondSemester.science - scores.secondSemester.science / 3 := by sorry
  have h4 : scores.secondSemester.maths = 80 := by sorry
  have h5 : scores.secondSemester.arts = 90 := by sorry
  have h6 : scores.secondSemester.science = 90 := by sorry
  sorry

end eva_total_score_2019_l1212_121258


namespace parking_lot_buses_l1212_121230

/-- The total number of buses in a parking lot after more buses arrive -/
def total_buses (initial : ℕ) (additional : ℕ) : ℕ :=
  initial + additional

/-- Theorem: Given 7 initial buses and 6 additional buses, the total is 13 -/
theorem parking_lot_buses : total_buses 7 6 = 13 := by
  sorry

end parking_lot_buses_l1212_121230


namespace average_equals_black_dots_l1212_121286

/-- Represents the types of butterflies -/
inductive ButterflyType
  | A
  | B
  | C

/-- Returns the number of black dots for a given butterfly type -/
def blackDots (t : ButterflyType) : ℕ :=
  match t with
  | .A => 545
  | .B => 780
  | .C => 1135

/-- Returns the number of butterflies for a given type -/
def butterflyCount (t : ButterflyType) : ℕ :=
  match t with
  | .A => 15
  | .B => 25
  | .C => 35

/-- Calculates the average number of black dots per butterfly for a given type -/
def averageBlackDots (t : ButterflyType) : ℚ :=
  (blackDots t : ℚ) * (butterflyCount t : ℚ) / (butterflyCount t : ℚ)

theorem average_equals_black_dots (t : ButterflyType) :
  averageBlackDots t = blackDots t := by
  sorry

#eval averageBlackDots ButterflyType.A
#eval averageBlackDots ButterflyType.B
#eval averageBlackDots ButterflyType.C

end average_equals_black_dots_l1212_121286


namespace cos_420_plus_sin_330_equals_zero_l1212_121264

theorem cos_420_plus_sin_330_equals_zero :
  Real.cos (420 * π / 180) + Real.sin (330 * π / 180) = 0 := by
  sorry

end cos_420_plus_sin_330_equals_zero_l1212_121264


namespace unique_b_value_l1212_121205

theorem unique_b_value (a b : ℕ+) (h1 : (3 ^ a.val) ^ b.val = 3 ^ 3) (h2 : 3 ^ a.val * 3 ^ b.val = 81) : b = 3 := by
  sorry

end unique_b_value_l1212_121205


namespace power_division_nineteen_l1212_121244

theorem power_division_nineteen : 19^11 / 19^8 = 6859 := by sorry

end power_division_nineteen_l1212_121244


namespace equation_solution_l1212_121225

theorem equation_solution :
  let f (x : ℂ) := (3 * x^2 - 1) / (4 * x - 4)
  ∀ x : ℂ, f x = 2/3 ↔ x = 8/18 + (Complex.I * Real.sqrt 116)/18 ∨ x = 8/18 - (Complex.I * Real.sqrt 116)/18 := by
  sorry

end equation_solution_l1212_121225


namespace partial_fraction_decomposition_l1212_121283

theorem partial_fraction_decomposition (x A B C : ℝ) :
  (x + 2) / (x^3 - 9*x^2 + 14*x + 24) = A / (x - 4) + B / (x - 3) + C / ((x + 2)^2) →
  A = 1/6 := by
  sorry

end partial_fraction_decomposition_l1212_121283


namespace shoe_size_mode_and_median_l1212_121251

def shoe_sizes : List ℝ := [25, 25, 25.5, 25.5, 25.5, 25.5, 26, 26, 26.5, 27]

def mode (list : List ℝ) : ℝ := sorry

def median (list : List ℝ) : ℝ := sorry

theorem shoe_size_mode_and_median :
  mode shoe_sizes = 25.5 ∧ median shoe_sizes = 25.5 := by sorry

end shoe_size_mode_and_median_l1212_121251


namespace algebraic_expression_equality_l1212_121267

theorem algebraic_expression_equality (a b : ℝ) (h : 5 * a + 3 * b = -4) :
  2 * (a + b) + 4 * (2 * a + b) - 10 = -18 := by sorry

end algebraic_expression_equality_l1212_121267


namespace reflected_line_x_intercept_l1212_121246

/-- The x-intercept of a line reflected in the y-axis -/
theorem reflected_line_x_intercept :
  let original_line : ℝ → ℝ := λ x => 2 * x - 6
  let reflected_line : ℝ → ℝ := λ x => -2 * x - 6
  let x_intercept : ℝ := -3
  (reflected_line x_intercept = 0) ∧ 
  (∀ y : ℝ, reflected_line y = 0 → y = x_intercept) :=
by sorry

end reflected_line_x_intercept_l1212_121246


namespace sewer_capacity_l1212_121201

/-- The amount of run-off produced per hour of rain in gallons -/
def runoff_per_hour : ℕ := 1000

/-- The number of days the sewers can handle before overflow -/
def days_before_overflow : ℕ := 10

/-- The number of hours in a day -/
def hours_per_day : ℕ := 24

/-- The total gallons of run-off the sewers can handle -/
def total_runoff_capacity : ℕ := runoff_per_hour * days_before_overflow * hours_per_day

theorem sewer_capacity :
  total_runoff_capacity = 240000 := by
  sorry

end sewer_capacity_l1212_121201


namespace probability_even_sum_l1212_121214

def card_set : Finset ℕ := {1, 2, 3, 4, 5}

def is_even_sum (pair : ℕ × ℕ) : Bool :=
  (pair.1 + pair.2) % 2 = 0

def favorable_outcomes : Finset (ℕ × ℕ) :=
  (card_set.product card_set).filter (λ pair => pair.1 < pair.2 ∧ is_even_sum pair)

def total_outcomes : Finset (ℕ × ℕ) :=
  (card_set.product card_set).filter (λ pair => pair.1 < pair.2)

theorem probability_even_sum :
  (favorable_outcomes.card : ℚ) / total_outcomes.card = 2 / 5 :=
by sorry

end probability_even_sum_l1212_121214


namespace g_of_3_eq_38_div_5_l1212_121240

noncomputable def f (x : ℝ) : ℝ := 4 / (3 - x)

noncomputable def g (x : ℝ) : ℝ := 1 / (f.invFun x) + 7

theorem g_of_3_eq_38_div_5 : g 3 = 38 / 5 := by
  sorry

end g_of_3_eq_38_div_5_l1212_121240


namespace train_crossing_time_l1212_121294

/-- Proves that a train with given length and speed takes a specific time to cross a pole -/
theorem train_crossing_time (train_length : Real) (train_speed_kmh : Real) (crossing_time : Real) :
  train_length = 900 →
  train_speed_kmh = 180 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) →
  crossing_time = 18 := by
  sorry

end train_crossing_time_l1212_121294


namespace inequality_solution_set_l1212_121241

theorem inequality_solution_set (x : ℝ) : (x - 2) * (3 - x) > 0 ↔ 2 < x ∧ x < 3 := by
  sorry

end inequality_solution_set_l1212_121241


namespace manoj_transaction_gain_l1212_121281

/-- Calculate simple interest -/
def simple_interest (principal : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  (principal * rate * time) / 100

/-- The amount Manoj borrowed from Anwar -/
def borrowed_amount : ℚ := 3900

/-- The interest rate Manoj pays to Anwar -/
def borrowing_rate : ℚ := 6

/-- The amount Manoj lent to Ramu -/
def lent_amount : ℚ := 5655

/-- The interest rate Manoj charges Ramu -/
def lending_rate : ℚ := 9

/-- The time period for both transactions in years -/
def time_period : ℚ := 3

/-- Manoj's gain from the transaction -/
def manoj_gain : ℚ :=
  simple_interest lent_amount lending_rate time_period -
  simple_interest borrowed_amount borrowing_rate time_period

theorem manoj_transaction_gain :
  manoj_gain = 824.85 := by sorry

end manoj_transaction_gain_l1212_121281


namespace card_area_theorem_l1212_121235

/-- Represents the dimensions of a rectangular card -/
structure Card where
  length : ℝ
  width : ℝ

/-- Calculates the area of a card -/
def area (c : Card) : ℝ := c.length * c.width

/-- Theorem: Given a 5x7 card, if shortening one side by 2 inches results in 
    an area of 21 square inches, then shortening the other side by 2 inches 
    will result in an area of 25 square inches -/
theorem card_area_theorem (c : Card) 
    (h1 : c.length = 5 ∧ c.width = 7)
    (h2 : ∃ (shortened_card : Card), 
      (shortened_card.length = c.length - 2 ∧ shortened_card.width = c.width) ∨
      (shortened_card.length = c.length ∧ shortened_card.width = c.width - 2))
    (h3 : ∃ (shortened_card : Card), 
      area shortened_card = 21 ∧
      ((shortened_card.length = c.length - 2 ∧ shortened_card.width = c.width) ∨
       (shortened_card.length = c.length ∧ shortened_card.width = c.width - 2)))
    : ∃ (other_shortened_card : Card), 
      area other_shortened_card = 25 ∧
      ((other_shortened_card.length = c.length - 2 ∧ other_shortened_card.width = c.width) ∨
       (other_shortened_card.length = c.length ∧ other_shortened_card.width = c.width - 2)) :=
by
  sorry


end card_area_theorem_l1212_121235


namespace longest_segment_is_BD_l1212_121212

/-- Given a triangle ABC, returns true if AC > AB > BC -/
def triangleInequalityOrder (angleA angleB angleC : ℝ) : Prop :=
  angleA > angleB ∧ angleB > angleC

theorem longest_segment_is_BD 
  (angleABD angleADB angleCBD angleBDC : ℝ)
  (h1 : angleABD = 50)
  (h2 : angleADB = 45)
  (h3 : angleCBD = 70)
  (h4 : angleBDC = 65)
  (h5 : triangleInequalityOrder (180 - angleABD - angleADB) angleABD angleADB)
  (h6 : triangleInequalityOrder angleCBD angleBDC (180 - angleCBD - angleBDC)) :
  ∃ (lengthAB lengthBC lengthCD lengthAD lengthBD : ℝ),
    lengthAD < lengthAB ∧ 
    lengthAB < lengthBC ∧ 
    lengthBC < lengthCD ∧ 
    lengthCD < lengthBD :=
by sorry

end longest_segment_is_BD_l1212_121212


namespace white_squares_20th_row_l1212_121200

/-- Represents the number of squares in a row of the stair-step figure -/
def squares_in_row (n : ℕ) : ℕ := 2 * n + 1

/-- Represents the number of white squares in a row of the stair-step figure -/
def white_squares_in_row (n : ℕ) : ℕ := (squares_in_row n - 1) / 2

theorem white_squares_20th_row :
  white_squares_in_row 20 = 20 := by
  sorry

#eval white_squares_in_row 20

end white_squares_20th_row_l1212_121200


namespace polynomial_parity_l1212_121226

/-- Represents a polynomial with integer coefficients -/
def IntPolynomial := List Int

/-- Multiplies two polynomials -/
def polyMul (p q : IntPolynomial) : IntPolynomial := sorry

/-- Checks if all coefficients of a polynomial are even -/
def allCoeffsEven (p : IntPolynomial) : Prop := sorry

/-- Checks if all coefficients of a polynomial are divisible by 4 -/
def allCoeffsDivBy4 (p : IntPolynomial) : Prop := sorry

/-- Checks if a polynomial has at least one odd coefficient -/
def hasOddCoeff (p : IntPolynomial) : Prop := sorry

theorem polynomial_parity (p q : IntPolynomial) :
  (allCoeffsEven (polyMul p q)) ∧ ¬(allCoeffsDivBy4 (polyMul p q)) →
  (allCoeffsEven p ∧ hasOddCoeff q) ∨ (allCoeffsEven q ∧ hasOddCoeff p) := by
  sorry

end polynomial_parity_l1212_121226


namespace product_of_symmetrical_complex_l1212_121291

/-- Two complex numbers are symmetrical about y = x if their real and imaginary parts are swapped -/
def symmetrical_about_y_eq_x (z₁ z₂ : ℂ) : Prop :=
  z₁.re = z₂.im ∧ z₁.im = z₂.re

theorem product_of_symmetrical_complex : ∀ z₁ z₂ : ℂ,
  symmetrical_about_y_eq_x z₁ z₂ →
  z₁ = 3 + 2*I →
  z₁ * z₂ = 13*I :=
sorry

end product_of_symmetrical_complex_l1212_121291


namespace daily_wage_of_c_l1212_121282

theorem daily_wage_of_c (a b c : ℕ) (total_earning : ℚ) : 
  a = 6 ∧ b = 9 ∧ c = 4 → 
  ∃ (x : ℚ), 
    (3 * x * a + 4 * x * b + 5 * x * c = total_earning) ∧
    (total_earning = 1480) →
    5 * x = 100 := by
  sorry

end daily_wage_of_c_l1212_121282


namespace distance_can_be_four_l1212_121250

/-- A circle with radius 3 -/
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)
  (h_radius : radius = 3)

/-- A point outside the circle -/
structure OutsidePoint (c : Circle) :=
  (point : ℝ × ℝ)
  (h_outside : dist point c.center > c.radius)

/-- The theorem stating that the distance between the center and the outside point can be 4 -/
theorem distance_can_be_four (c : Circle) (p : OutsidePoint c) : 
  ∃ (q : OutsidePoint c), dist q.point c.center = 4 :=
sorry

end distance_can_be_four_l1212_121250


namespace minimum_point_of_translated_absolute_value_l1212_121252

-- Define the function
def f (x : ℝ) : ℝ := |x - 4| + 1

-- State the theorem
theorem minimum_point_of_translated_absolute_value :
  ∃ (x₀ : ℝ), (∀ (x : ℝ), f x₀ ≤ f x) ∧ (x₀ = 4 ∧ f x₀ = 1) :=
sorry

end minimum_point_of_translated_absolute_value_l1212_121252


namespace limit_implies_a_range_l1212_121229

/-- If the limit of 3^n / (3^(n+1) + (a+1)^n) as n approaches infinity is 1/3, 
    then a is in the open interval (-4, 2) -/
theorem limit_implies_a_range (a : ℝ) :
  (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |3^n / (3^(n+1) + (a+1)^n) - 1/3| < ε) →
  a ∈ Set.Ioo (-4 : ℝ) 2 :=
by sorry

end limit_implies_a_range_l1212_121229


namespace perfect_square_concatenation_l1212_121293

theorem perfect_square_concatenation (b m : ℕ) (h_b_odd : Odd b) :
  let A : ℕ := (5^b + 1) / 2
  let B : ℕ := 2^b * A * 100^m
  let AB : ℕ := 10^(Nat.digits 10 B).length * A + B
  ∃ (n : ℕ), AB = n^2 ∧ AB = 2 * A * B := by
  sorry

end perfect_square_concatenation_l1212_121293


namespace convex_quad_polyhedron_16v_14f_l1212_121255

/-- A convex polyhedron with quadrilateral faces -/
structure ConvexQuadPolyhedron where
  vertices : ℕ
  faces : ℕ
  edges : ℕ
  convex : Bool
  all_faces_quadrilateral : Bool
  euler : vertices + faces - edges = 2
  quad_face_edge_relation : edges = 2 * faces

/-- Theorem: A convex polyhedron with 16 vertices and all quadrilateral faces has 14 faces -/
theorem convex_quad_polyhedron_16v_14f :
  ∀ (P : ConvexQuadPolyhedron), 
    P.vertices = 16 ∧ P.convex ∧ P.all_faces_quadrilateral → P.faces = 14 :=
by sorry

end convex_quad_polyhedron_16v_14f_l1212_121255


namespace solve_for_q_l1212_121247

theorem solve_for_q (k l q : ℚ) 
  (eq1 : 3/4 = k/48)
  (eq2 : 3/4 = (k + l)/56)
  (eq3 : 3/4 = (q - l)/160) : 
  q = 126 := by
sorry

end solve_for_q_l1212_121247


namespace certain_number_problem_l1212_121256

theorem certain_number_problem (x : ℝ) : 
  (0.55 * x = (4 / 5 : ℝ) * 25 + 2) → x = 40 := by
sorry

end certain_number_problem_l1212_121256


namespace jill_earnings_l1212_121269

def first_month_daily_wage : ℕ := 10
def days_per_month : ℕ := 30

def second_month_daily_wage : ℕ := 2 * first_month_daily_wage
def third_month_working_days : ℕ := days_per_month / 2

def first_month_earnings : ℕ := first_month_daily_wage * days_per_month
def second_month_earnings : ℕ := second_month_daily_wage * days_per_month
def third_month_earnings : ℕ := second_month_daily_wage * third_month_working_days

def total_earnings : ℕ := first_month_earnings + second_month_earnings + third_month_earnings

theorem jill_earnings : total_earnings = 1200 := by
  sorry

end jill_earnings_l1212_121269


namespace fraction_equality_l1212_121295

theorem fraction_equality (a b : ℝ) (h : a / b = 3 / 4) : (b - a) / b = 1 / 4 := by
  sorry

end fraction_equality_l1212_121295


namespace sqrt_nine_minus_half_inverse_equals_one_l1212_121234

theorem sqrt_nine_minus_half_inverse_equals_one :
  Real.sqrt 9 - (1/2)⁻¹ = 1 := by
  sorry

end sqrt_nine_minus_half_inverse_equals_one_l1212_121234


namespace rectangle_y_value_l1212_121222

/-- Rectangle EFGH with vertices E(0, 0), F(0, 5), G(y, 5), and H(y, 0) -/
structure Rectangle where
  y : ℝ
  h_positive : y > 0

/-- The area of the rectangle -/
def area (r : Rectangle) : ℝ := 5 * r.y

theorem rectangle_y_value (r : Rectangle) (h_area : area r = 35) : r.y = 7 := by
  sorry

end rectangle_y_value_l1212_121222


namespace alice_winning_condition_l1212_121207

/-- Game state representing the numbers on the board -/
structure GameState where
  numbers : List ℚ
  deriving Repr

/-- Player type -/
inductive Player
| Alice
| Bob
deriving Repr

/-- Result of the game -/
inductive GameResult
| AliceWins
| BobWins
deriving Repr

/-- Perform a move in the game -/
def makeMove (state : GameState) : GameState :=
  sorry

/-- Play the game with given parameters -/
def playGame (n : ℕ) (c : ℚ) (initialNumbers : List ℕ) : GameResult :=
  sorry

/-- Alice's winning condition -/
def aliceWins (c : ℚ) : Prop :=
  ∀ n₀ : ℕ, ∃ n : ℕ, n ≥ n₀ ∧ ∀ initialNumbers : List ℕ,
    initialNumbers.length = n → (∃ x y : ℕ, x ∈ initialNumbers ∧ y ∈ initialNumbers ∧ x ≠ y) →
      playGame n c initialNumbers = GameResult.AliceWins

theorem alice_winning_condition (c : ℚ) :
  aliceWins c ↔ c ≥ (1/2 : ℚ) :=
  sorry

end alice_winning_condition_l1212_121207


namespace hyperbola_equation_l1212_121239

/-- Given a hyperbola with the equation (x²/a² - y²/b² = 1) where a > 0 and b > 0,
    and a line y = 2x - 4 that passes through the right focus F and intersects
    the hyperbola at only one point, prove that the equation of the hyperbola
    is (5x²/4 - 5y²/16 = 1). -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ F : ℝ × ℝ,
    (F.1 > 0) ∧  -- F is the right focus
    (F.2 = 0) ∧  -- F is on the x-axis
    (∀ x y : ℝ, y = 2*x - 4 → (x - F.1)^2 + y^2 = (a^2 + b^2)) ∧  -- line passes through F
    (∃! P : ℝ × ℝ, P.2 = 2*P.1 - 4 ∧ P.1^2/a^2 - P.2^2/b^2 = 1))  -- line intersects hyperbola at one point
  →
  a^2 = 4/5 ∧ b^2 = 16/5 :=
by sorry

end hyperbola_equation_l1212_121239


namespace smallest_positive_integer_ending_in_3_divisible_by_11_l1212_121221

theorem smallest_positive_integer_ending_in_3_divisible_by_11 : 
  ∃ n : ℕ, n > 0 ∧ n % 10 = 3 ∧ n % 11 = 0 ∧ ∀ m : ℕ, m > 0 ∧ m % 10 = 3 ∧ m % 11 = 0 → n ≤ m :=
by
  -- The proof would go here
  sorry

#check smallest_positive_integer_ending_in_3_divisible_by_11

end smallest_positive_integer_ending_in_3_divisible_by_11_l1212_121221


namespace adam_books_theorem_l1212_121275

def initial_books : ℕ := 67
def sold_fraction : ℚ := 2/3
def reinvestment_fraction : ℚ := 3/4
def new_book_price : ℕ := 3

def books_after_transactions : ℕ := 56

theorem adam_books_theorem :
  let sold_books := (initial_books * sold_fraction).floor
  let money_earned := sold_books * new_book_price
  let money_for_new_books := (money_earned : ℚ) * reinvestment_fraction
  let new_books := (money_for_new_books / new_book_price).floor
  initial_books - sold_books + new_books = books_after_transactions := by
  sorry

end adam_books_theorem_l1212_121275


namespace alternative_rate_calculation_l1212_121299

/-- Calculates simple interest -/
def simple_interest (principal : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  principal * rate * time

theorem alternative_rate_calculation (principal : ℚ) (time : ℚ) (actual_rate : ℚ) 
  (interest_difference : ℚ) (alternative_rate : ℚ) : 
  principal = 2500 →
  time = 2 →
  actual_rate = 18 / 100 →
  interest_difference = 300 →
  simple_interest principal actual_rate time - simple_interest principal alternative_rate time = interest_difference →
  alternative_rate = 12 / 100 := by
sorry

end alternative_rate_calculation_l1212_121299


namespace initial_workers_l1212_121238

/-- Proves that the initial number of workers is 14, given the problem conditions --/
theorem initial_workers (total_toys : ℕ) (initial_days : ℕ) (added_workers : ℕ) (remaining_days : ℕ) :
  total_toys = 1400 →
  initial_days = 5 →
  added_workers = 14 →
  remaining_days = 2 →
  ∃ (initial_workers : ℕ),
    (initial_workers * initial_days + (initial_workers + added_workers) * remaining_days) * total_toys / 
    (initial_days + remaining_days) = total_toys ∧
    initial_workers = 14 := by
  sorry


end initial_workers_l1212_121238


namespace marathon_time_l1212_121248

/-- Calculates the total time to complete a marathon given specific conditions -/
theorem marathon_time (total_distance : ℝ) (initial_distance : ℝ) (initial_time : ℝ) (remaining_pace_factor : ℝ) :
  total_distance = 26 →
  initial_distance = 10 →
  initial_time = 1 →
  remaining_pace_factor = 0.8 →
  let initial_pace := initial_distance / initial_time
  let remaining_distance := total_distance - initial_distance
  let remaining_pace := initial_pace * remaining_pace_factor
  let remaining_time := remaining_distance / remaining_pace
  initial_time + remaining_time = 3 :=
by
  sorry

end marathon_time_l1212_121248


namespace unique_three_digit_square_l1212_121272

theorem unique_three_digit_square (a b c : Nat) : 
  a > 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ 
  a^2 < 10 ∧ b^2 < 10 ∧ c^2 < 10 ∧
  (100*a + 10*b + c)^2 = 1000*100*a + 1000*10*b + 1000*c + 100*a + 10*b + c →
  a = 2 ∧ b = 3 ∧ c = 3 := by
sorry

end unique_three_digit_square_l1212_121272


namespace sandy_token_difference_l1212_121263

/-- Represents the number of Safe Moon tokens Sandy bought -/
def total_tokens : ℕ := 1000000

/-- Represents the number of Sandy's siblings -/
def num_siblings : ℕ := 4

/-- Calculates the number of tokens Sandy keeps for herself -/
def sandy_tokens : ℕ := total_tokens / 2

/-- Calculates the number of tokens each sibling receives -/
def sibling_tokens : ℕ := (total_tokens - sandy_tokens) / num_siblings

/-- Proves that Sandy has 375,000 more tokens than any of her siblings -/
theorem sandy_token_difference : sandy_tokens - sibling_tokens = 375000 := by
  sorry

end sandy_token_difference_l1212_121263


namespace money_distribution_l1212_121202

theorem money_distribution (a b : ℚ) : 
  (a + b / 2 = 50) → 
  (b + 2 * a / 3 = 50) → 
  (a = 37.5 ∧ b = 25) := by sorry

end money_distribution_l1212_121202


namespace coin_weighing_strategy_exists_l1212_121297

/-- Represents the possible weights of a coin type -/
inductive CoinWeight
  | Five
  | Six
  | Seven
  | Eight

/-- Represents the result of a weighing -/
inductive WeighingResult
  | LeftHeavier
  | RightHeavier
  | Equal

/-- Represents a weighing strategy -/
structure WeighingStrategy :=
  (firstWeighing : WeighingResult → Option WeighingResult)

/-- Represents the coin set -/
structure CoinSet :=
  (doubloonWeight : CoinWeight)
  (crownWeight : CoinWeight)

/-- Determines if a weighing strategy can identify the exact weights -/
def canIdentifyWeights (strategy : WeighingStrategy) (coins : CoinSet) : Prop :=
  ∃ (result1 : WeighingResult) (result2 : Option WeighingResult),
    (result2 = strategy.firstWeighing result1) ∧
    (∀ (otherCoins : CoinSet),
      (otherCoins ≠ coins) →
      (∃ (otherResult1 : WeighingResult) (otherResult2 : Option WeighingResult),
        (otherResult2 = strategy.firstWeighing otherResult1) ∧
        ((otherResult1 ≠ result1) ∨ (otherResult2 ≠ result2))))

theorem coin_weighing_strategy_exists :
  ∃ (strategy : WeighingStrategy),
    ∀ (coins : CoinSet),
      (coins.doubloonWeight = CoinWeight.Five ∨ coins.doubloonWeight = CoinWeight.Six) →
      (coins.crownWeight = CoinWeight.Seven ∨ coins.crownWeight = CoinWeight.Eight) →
      canIdentifyWeights strategy coins :=
by sorry


end coin_weighing_strategy_exists_l1212_121297


namespace finite_solutions_factorial_square_sum_l1212_121236

theorem finite_solutions_factorial_square_sum (a : ℕ) :
  ∃ (n : ℕ), ∀ (x y : ℕ), x! = y^2 + a^2 → x ≤ n :=
sorry

end finite_solutions_factorial_square_sum_l1212_121236


namespace wheel_speed_calculation_l1212_121242

/-- Prove that given a wheel with a circumference of 8 feet, if reducing the time
    for a complete rotation by 0.5 seconds increases the speed by 6 miles per hour,
    then the original speed of the wheel is 9 miles per hour. -/
theorem wheel_speed_calculation (r : ℝ) : 
  let circumference : ℝ := 8 / 5280  -- circumference in miles
  let t : ℝ := circumference * 3600 / r  -- time for one rotation in seconds
  let new_t : ℝ := t - 0.5  -- new time after reduction
  let new_r : ℝ := r + 6  -- new speed after increase
  (new_r * new_t / 3600 = circumference) →  -- equation for new speed and time
  r = 9 := by
sorry

end wheel_speed_calculation_l1212_121242


namespace unit_segment_construction_l1212_121216

theorem unit_segment_construction (a : ℝ) (h : a > 1) : (a / a^2) * a = 1 := by
  sorry

end unit_segment_construction_l1212_121216


namespace solve_equation_l1212_121215

theorem solve_equation (y : ℝ) : 7 - y = 12 ↔ y = -5 := by sorry

end solve_equation_l1212_121215
