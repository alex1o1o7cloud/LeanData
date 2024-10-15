import Mathlib

namespace NUMINAMATH_CALUDE_triangle_area_l2602_260241

theorem triangle_area (a b c : ℝ) (h1 : a = 14) (h2 : b = 48) (h3 : c = 50) :
  (1/2) * a * b = 336 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l2602_260241


namespace NUMINAMATH_CALUDE_vector_magnitude_proof_l2602_260257

/-- Given vectors a and b, prove that the magnitude of a - 2b is √3 -/
theorem vector_magnitude_proof (a b : ℝ × ℝ) :
  a.1 = Real.cos (15 * π / 180) ∧
  a.2 = Real.sin (15 * π / 180) ∧
  b.1 = Real.cos (75 * π / 180) ∧
  b.2 = Real.sin (75 * π / 180) →
  Real.sqrt ((a.1 - 2 * b.1)^2 + (a.2 - 2 * b.2)^2) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_proof_l2602_260257


namespace NUMINAMATH_CALUDE_students_with_a_l2602_260271

theorem students_with_a (total_students : ℕ) (ratio : ℚ) 
  (h1 : total_students = 30) 
  (h2 : ratio = 2 / 3) : 
  ∃ (a_students : ℕ) (percentage : ℚ), 
    a_students = 20 ∧ 
    percentage = 200 / 3 ∧
    (a_students : ℚ) / total_students = ratio ∧
    percentage = (a_students : ℚ) / total_students * 100 := by
  sorry

end NUMINAMATH_CALUDE_students_with_a_l2602_260271


namespace NUMINAMATH_CALUDE_common_factor_is_gcf_l2602_260246

noncomputable def p1 (x y z : ℝ) : ℝ := 3 * x^2 * y^3 * z + 9 * x^3 * y^3 * z
noncomputable def p2 (x y z : ℝ) : ℝ := 6 * x^4 * y * z^2
noncomputable def common_factor (x y z : ℝ) : ℝ := 3 * x^2 * y * z

theorem common_factor_is_gcf :
  ∀ x y z : ℝ,
  (∃ k1 k2 : ℝ, p1 x y z = common_factor x y z * k1 ∧ p2 x y z = common_factor x y z * k2) ∧
  (∀ f : ℝ → ℝ → ℝ → ℝ, (∃ l1 l2 : ℝ, p1 x y z = f x y z * l1 ∧ p2 x y z = f x y z * l2) →
    ∃ m : ℝ, f x y z = common_factor x y z * m) :=
by sorry

end NUMINAMATH_CALUDE_common_factor_is_gcf_l2602_260246


namespace NUMINAMATH_CALUDE_range_of_p_l2602_260221

-- Define the set A
def A (p : ℝ) : Set ℝ := {x : ℝ | |x| * x^2 + (p + 2) * x + 1 = 0}

-- Define the theorem
theorem range_of_p (p : ℝ) : 
  (A p ∩ Set.Ici (0 : ℝ) = ∅) ↔ (-4 < p ∧ p < 0) := by sorry

end NUMINAMATH_CALUDE_range_of_p_l2602_260221


namespace NUMINAMATH_CALUDE_expression_simplification_l2602_260276

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 3) :
  (x^2 - 2*x + 1) / (x^2 - x) / (x - 1) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2602_260276


namespace NUMINAMATH_CALUDE_quadratic_points_order_l2602_260245

/-- Given a quadratic function f(x) = x² + x - 1, prove that y₂ < y₁ < y₃ 
    where y₁, y₂, and y₃ are the y-coordinates of points on the graph of f 
    with x-coordinates -2, 0, and 2 respectively. -/
theorem quadratic_points_order : 
  let f : ℝ → ℝ := λ x ↦ x^2 + x - 1
  let y₁ : ℝ := f (-2)
  let y₂ : ℝ := f 0
  let y₃ : ℝ := f 2
  y₂ < y₁ ∧ y₁ < y₃ := by
  sorry

end NUMINAMATH_CALUDE_quadratic_points_order_l2602_260245


namespace NUMINAMATH_CALUDE_inverse_fifty_l2602_260230

theorem inverse_fifty (x : ℝ) : (1 / x = 50) → (x = 1 / 50) := by
  sorry

end NUMINAMATH_CALUDE_inverse_fifty_l2602_260230


namespace NUMINAMATH_CALUDE_age_sum_is_23_l2602_260223

/-- The ages of Al, Bob, and Carl satisfy the given conditions and their sum is 23 -/
theorem age_sum_is_23 (a b c : ℕ) : 
  a = 10 * b * c ∧ 
  a^3 = 8000 + 8 * b^3 * c^3 → 
  a + b + c = 23 := by
sorry

end NUMINAMATH_CALUDE_age_sum_is_23_l2602_260223


namespace NUMINAMATH_CALUDE_sqrt_twelve_minus_sqrt_three_equals_sqrt_three_l2602_260270

theorem sqrt_twelve_minus_sqrt_three_equals_sqrt_three :
  Real.sqrt 12 - Real.sqrt 3 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_twelve_minus_sqrt_three_equals_sqrt_three_l2602_260270


namespace NUMINAMATH_CALUDE_factorization_difference_of_squares_l2602_260220

theorem factorization_difference_of_squares (a b : ℝ) : 3 * a^2 - 3 * b^2 = 3 * (a + b) * (a - b) := by
  sorry

end NUMINAMATH_CALUDE_factorization_difference_of_squares_l2602_260220


namespace NUMINAMATH_CALUDE_line_perp_to_plane_and_line_para_to_plane_implies_lines_perp_l2602_260216

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Line → Prop)
variable (perpToPlane : Line → Plane → Prop)
variable (para : Line → Plane → Prop)

-- State the theorem
theorem line_perp_to_plane_and_line_para_to_plane_implies_lines_perp
  (a b : Line) (α : Plane) :
  a ≠ b →  -- a and b are non-coincident
  perpToPlane a α →  -- a is perpendicular to α
  para b α →  -- b is parallel to α
  perp a b :=  -- then a is perpendicular to b
by sorry

end NUMINAMATH_CALUDE_line_perp_to_plane_and_line_para_to_plane_implies_lines_perp_l2602_260216


namespace NUMINAMATH_CALUDE_taxi_problem_l2602_260226

theorem taxi_problem (fans : ℕ) (company_a company_b : ℕ) : 
  fans = 56 →
  company_b = company_a + 3 →
  5 * company_a < fans →
  6 * company_a > fans →
  4 * company_b < fans →
  5 * company_b > fans →
  company_a = 10 := by
sorry

end NUMINAMATH_CALUDE_taxi_problem_l2602_260226


namespace NUMINAMATH_CALUDE_geometric_sequence_properties_l2602_260252

/-- Represents the nth term of the geometric sequence -/
def a (n : ℕ) : ℝ := sorry

/-- Represents the sum of the first n terms of the geometric sequence -/
def S (n : ℕ) : ℝ := sorry

/-- The common difference when inserting n numbers between a_n and a_{n+1} -/
def d (n : ℕ) : ℝ := sorry

/-- Main theorem encompassing both parts of the problem -/
theorem geometric_sequence_properties :
  (∀ n : ℕ, n ≥ 1 → a (n + 1) = 2 * S n + 2) →
  (∀ n : ℕ, n ≥ 1 → a n = 2 * 3^(n - 1)) ∧
  (¬ ∃ m k p : ℕ,
    m < k ∧ k < p ∧
    (k - m = p - k) ∧
    (d m * d p = d k * d k)) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_properties_l2602_260252


namespace NUMINAMATH_CALUDE_rower_distance_l2602_260217

/-- Represents the problem of calculating the distance traveled by a rower in a river --/
theorem rower_distance (man_speed : ℝ) (river_speed : ℝ) (total_time : ℝ) : 
  man_speed = 10 →
  river_speed = 1.2 →
  total_time = 1 →
  let upstream_speed := man_speed - river_speed
  let downstream_speed := man_speed + river_speed
  let one_way_distance := (upstream_speed * downstream_speed * total_time) / (2 * (upstream_speed + downstream_speed))
  2 * one_way_distance = 9.856 := by
sorry

#eval (10 - 1.2) * (10 + 1.2) / (2 * ((10 - 1.2) + (10 + 1.2)))

end NUMINAMATH_CALUDE_rower_distance_l2602_260217


namespace NUMINAMATH_CALUDE_last_number_in_first_set_l2602_260281

def first_set_mean : ℝ := 90
def second_set_mean : ℝ := 423

def first_set (x y : ℝ) : List ℝ := [28, x, 42, 78, y]
def second_set (x : ℝ) : List ℝ := [128, 255, 511, 1023, x]

theorem last_number_in_first_set (x y : ℝ) : 
  (List.sum (first_set x y) / 5 = first_set_mean) →
  (List.sum (second_set x) / 5 = second_set_mean) →
  y = 104 := by
  sorry

end NUMINAMATH_CALUDE_last_number_in_first_set_l2602_260281


namespace NUMINAMATH_CALUDE_diagonals_parity_iff_n_parity_l2602_260294

/-- The number of diagonals in a regular polygon with 2n+1 sides. -/
def num_diagonals (n : ℕ) : ℕ := (2 * n + 1).choose 2 - (2 * n + 1)

/-- Theorem: The number of diagonals in a regular polygon with 2n+1 sides is odd if and only if n is even. -/
theorem diagonals_parity_iff_n_parity (n : ℕ) (h : n > 1) :
  Odd (num_diagonals n) ↔ Even n := by
  sorry

end NUMINAMATH_CALUDE_diagonals_parity_iff_n_parity_l2602_260294


namespace NUMINAMATH_CALUDE_periodic_trig_function_l2602_260298

/-- Given a function f(x) = a*sin(πx + α) + b*cos(πx + β) where f(4) = 3,
    prove that f(2017) = -3 -/
theorem periodic_trig_function (a b α β : ℝ) :
  let f : ℝ → ℝ := λ x => a * Real.sin (π * x + α) + b * Real.cos (π * x + β)
  f 4 = 3 → f 2017 = -3 := by sorry

end NUMINAMATH_CALUDE_periodic_trig_function_l2602_260298


namespace NUMINAMATH_CALUDE_modified_prism_edge_count_l2602_260234

/-- Represents a modified rectangular prism with intersecting corner cuts -/
structure ModifiedPrism where
  original_edges : Nat
  vertex_count : Nat
  new_edges_per_vertex : Nat
  intersections_per_vertex : Nat
  additional_edges_per_intersection : Nat

/-- Calculates the total number of edges in the modified prism -/
def total_edges (p : ModifiedPrism) : Nat :=
  p.original_edges + 
  (p.vertex_count * p.new_edges_per_vertex) + 
  (p.vertex_count * p.intersections_per_vertex * p.additional_edges_per_intersection)

/-- Theorem stating that the modified prism has 52 edges -/
theorem modified_prism_edge_count :
  ∃ (p : ModifiedPrism), total_edges p = 52 :=
sorry

end NUMINAMATH_CALUDE_modified_prism_edge_count_l2602_260234


namespace NUMINAMATH_CALUDE_library_rearrangement_l2602_260200

theorem library_rearrangement (total_books : ℕ) (initial_shelves : ℕ) (books_per_new_shelf : ℕ)
  (h1 : total_books = 1500)
  (h2 : initial_shelves = 50)
  (h3 : books_per_new_shelf = 28)
  (h4 : total_books % initial_shelves = 0) : -- Ensures equally-filled initial shelves
  (total_books % books_per_new_shelf : ℕ) = 14 := by
sorry

end NUMINAMATH_CALUDE_library_rearrangement_l2602_260200


namespace NUMINAMATH_CALUDE_sigma_inequality_l2602_260211

/-- Sum of positive divisors function -/
def sigma (n : ℕ+) : ℕ := sorry

/-- Theorem: If σ(n) > 2n, then σ(mn) > 2mn for any m -/
theorem sigma_inequality (n : ℕ+) (h : sigma n > 2 * n) :
  ∀ m : ℕ+, sigma (m * n) > 2 * m * n := by
  sorry

end NUMINAMATH_CALUDE_sigma_inequality_l2602_260211


namespace NUMINAMATH_CALUDE_solve_letter_addition_puzzle_l2602_260282

theorem solve_letter_addition_puzzle (E F D : ℕ) 
  (h1 : E + F + D = 15)
  (h2 : F + E + 1 = 12)
  (h3 : E < 10 ∧ F < 10 ∧ D < 10)
  (h4 : E ≠ F ∧ F ≠ D ∧ E ≠ D) : D = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_letter_addition_puzzle_l2602_260282


namespace NUMINAMATH_CALUDE_pens_bought_proof_l2602_260202

/-- The cost of one pen in rubles -/
def pen_cost : ℕ := 21

/-- The amount Masha spent on pens in rubles -/
def masha_spent : ℕ := 357

/-- The amount Olya spent on pens in rubles -/
def olya_spent : ℕ := 441

/-- The total number of pens bought by Masha and Olya -/
def total_pens : ℕ := 38

theorem pens_bought_proof :
  pen_cost > 10 ∧
  masha_spent % pen_cost = 0 ∧
  olya_spent % pen_cost = 0 ∧
  masha_spent / pen_cost + olya_spent / pen_cost = total_pens :=
by sorry

end NUMINAMATH_CALUDE_pens_bought_proof_l2602_260202


namespace NUMINAMATH_CALUDE_curve_C_left_of_x_equals_2_l2602_260279

/-- The curve C is defined by the equation x³ + 2y² = 8 -/
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^3 + 2 * p.2^2 = 8}

/-- Theorem: All points on curve C have x-coordinate less than or equal to 2 -/
theorem curve_C_left_of_x_equals_2 : ∀ p ∈ C, p.1 ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_curve_C_left_of_x_equals_2_l2602_260279


namespace NUMINAMATH_CALUDE_percentage_of_boy_scouts_l2602_260236

theorem percentage_of_boy_scouts (total_scouts : ℝ) (boy_scouts : ℝ) (girl_scouts : ℝ)
  (h1 : boy_scouts + girl_scouts = total_scouts)
  (h2 : 0.60 * total_scouts = 0.50 * boy_scouts + 0.6818 * girl_scouts)
  : boy_scouts / total_scouts = 0.45 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_boy_scouts_l2602_260236


namespace NUMINAMATH_CALUDE_apple_box_weight_l2602_260284

theorem apple_box_weight (total_weight : ℝ) (pies : ℕ) (apples_per_pie : ℝ) : 
  total_weight / 2 = pies * apples_per_pie →
  pies = 15 →
  apples_per_pie = 4 →
  total_weight = 120 := by
sorry

end NUMINAMATH_CALUDE_apple_box_weight_l2602_260284


namespace NUMINAMATH_CALUDE_luke_coin_count_l2602_260269

def total_coins (quarter_piles dime_piles coins_per_pile : ℕ) : ℕ :=
  (quarter_piles + dime_piles) * coins_per_pile

theorem luke_coin_count :
  let quarter_piles : ℕ := 5
  let dime_piles : ℕ := 5
  let coins_per_pile : ℕ := 3
  total_coins quarter_piles dime_piles coins_per_pile = 30 := by
  sorry

end NUMINAMATH_CALUDE_luke_coin_count_l2602_260269


namespace NUMINAMATH_CALUDE_basketball_win_rate_l2602_260210

theorem basketball_win_rate (initial_wins : Nat) (initial_games : Nat) 
  (remaining_games : Nat) (target_win_rate : Rat) :
  initial_wins = 35 →
  initial_games = 45 →
  remaining_games = 55 →
  target_win_rate = 3/4 →
  ∃ (remaining_wins : Nat),
    remaining_wins = 40 ∧
    (initial_wins + remaining_wins : Rat) / (initial_games + remaining_games) = target_win_rate :=
by
  sorry

end NUMINAMATH_CALUDE_basketball_win_rate_l2602_260210


namespace NUMINAMATH_CALUDE_crayons_lost_or_given_away_l2602_260264

theorem crayons_lost_or_given_away (initial_crayons end_crayons : ℕ) 
  (h1 : initial_crayons = 479)
  (h2 : end_crayons = 134) :
  initial_crayons - end_crayons = 345 :=
by sorry

end NUMINAMATH_CALUDE_crayons_lost_or_given_away_l2602_260264


namespace NUMINAMATH_CALUDE_game_outcome_theorem_l2602_260267

/-- Represents the outcome of the game -/
inductive GameOutcome
| Draw
| BWin

/-- Defines the game rules and determines the outcome for a given n -/
def gameOutcome (n : ℕ+) : GameOutcome :=
  if n ∈ ({1, 2, 4, 6} : Finset ℕ+) then
    GameOutcome.Draw
  else
    GameOutcome.BWin

/-- Theorem stating the game outcome for all positive integers n -/
theorem game_outcome_theorem (n : ℕ+) :
  (gameOutcome n = GameOutcome.Draw ↔ n ∈ ({1, 2, 4, 6} : Finset ℕ+)) ∧
  (gameOutcome n = GameOutcome.BWin ↔ n ∉ ({1, 2, 4, 6} : Finset ℕ+)) :=
by sorry

end NUMINAMATH_CALUDE_game_outcome_theorem_l2602_260267


namespace NUMINAMATH_CALUDE_largest_multiple_of_12_less_than_350_l2602_260239

theorem largest_multiple_of_12_less_than_350 : 
  ∀ n : ℕ, n * 12 < 350 → n * 12 ≤ 348 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_12_less_than_350_l2602_260239


namespace NUMINAMATH_CALUDE_no_triangle_with_given_conditions_l2602_260253

theorem no_triangle_with_given_conditions : ¬∃ (a b c : ℝ), 
  (a > 0 ∧ b > 0 ∧ c > 0) ∧  -- positive sides
  (c = 0.2 * a) ∧            -- shortest side is 20% of longest
  (b = 0.25 * (a + b + c)) ∧ -- third side is 25% of perimeter
  (a + b > c ∧ a + c > b ∧ b + c > a) -- triangle inequality
  := by sorry

end NUMINAMATH_CALUDE_no_triangle_with_given_conditions_l2602_260253


namespace NUMINAMATH_CALUDE_addition_and_subtraction_of_integers_and_fractions_l2602_260250

theorem addition_and_subtraction_of_integers_and_fractions :
  (1 : ℤ) * 17 + (-12) = 5 ∧ -((1 : ℚ) / 7) - (-(6 / 7)) = 5 / 7 := by
  sorry

end NUMINAMATH_CALUDE_addition_and_subtraction_of_integers_and_fractions_l2602_260250


namespace NUMINAMATH_CALUDE_reimu_win_probability_l2602_260215

/-- Represents a coin with two sides that can be colored -/
structure Coin :=
  (side1 : Color)
  (side2 : Color)

/-- Possible colors for a coin side -/
inductive Color
  | White
  | Red
  | Green

/-- The game state -/
structure GameState :=
  (coins : List Coin)
  (currentPlayer : Player)

/-- The players in the game -/
inductive Player
  | Reimu
  | Sanae

/-- The result of the game -/
inductive GameResult
  | ReimuWins
  | SanaeWins
  | Tie

/-- Represents an optimal strategy for playing the game -/
def OptimalStrategy := GameState → Color

/-- The probability of a specific game result given optimal play -/
def resultProbability (strategy : OptimalStrategy) (result : GameResult) : ℚ :=
  sorry

/-- Theorem stating the probability of Reimu winning is 5/16 -/
theorem reimu_win_probability (strategy : OptimalStrategy) :
  resultProbability strategy GameResult.ReimuWins = 5 / 16 :=
sorry

end NUMINAMATH_CALUDE_reimu_win_probability_l2602_260215


namespace NUMINAMATH_CALUDE_units_digit_of_199_factorial_l2602_260258

theorem units_digit_of_199_factorial (n : ℕ) : n = 199 → (n.factorial % 10 = 0) := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_199_factorial_l2602_260258


namespace NUMINAMATH_CALUDE_sum_of_multiples_l2602_260290

theorem sum_of_multiples (x y : ℤ) (hx : 6 ∣ x) (hy : 9 ∣ y) : 3 ∣ (x + y) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_multiples_l2602_260290


namespace NUMINAMATH_CALUDE_sum_and_count_theorem_l2602_260232

def sum_integers (a b : ℕ) : ℕ :=
  (b - a + 1) * (a + b) / 2

def count_even_integers (a b : ℕ) : ℕ :=
  (b - a) / 2 + 1

theorem sum_and_count_theorem :
  sum_integers 20 40 + count_even_integers 20 40 = 641 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_count_theorem_l2602_260232


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2602_260225

theorem quadratic_equation_solution (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : ∀ x : ℝ, x^2 + b*x + a = 0 ↔ x = a ∨ x = b) : 
  a = -2 ∧ b = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2602_260225


namespace NUMINAMATH_CALUDE_smallest_sum_reciprocals_l2602_260224

theorem smallest_sum_reciprocals (x y : ℕ+) (h1 : x ≠ y) (h2 : (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 15) :
  ∃ (a b : ℕ+), a ≠ b ∧ (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 15 ∧ (a : ℕ) + b ≤ x + y ∧ (a : ℕ) + b = 64 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_reciprocals_l2602_260224


namespace NUMINAMATH_CALUDE_largest_r_for_sequence_convergence_l2602_260288

theorem largest_r_for_sequence_convergence (r : ℝ) : r > 2 →
  ∃ (a : ℕ → ℕ+), (∀ n, (a n : ℝ) ≤ a (n + 2) ∧ (a (n + 2) : ℝ) ≤ Real.sqrt ((a n : ℝ)^2 + r * (a (n + 1) : ℝ))) ∧
  ¬∃ M, ∀ n ≥ M, a (n + 2) = a n :=
by sorry

end NUMINAMATH_CALUDE_largest_r_for_sequence_convergence_l2602_260288


namespace NUMINAMATH_CALUDE_article_cost_price_l2602_260273

/-- The cost price of an article satisfying certain profit conditions -/
theorem article_cost_price (C : ℝ) (S : ℝ) : 
  S = 1.05 * C →  -- Condition 1: Selling price is 105% of cost price
  (1.05 * C - 1) = 1.1 * (0.95 * C) →  -- Condition 2: New selling price equals 110% of new cost price
  C = 200 := by
sorry

end NUMINAMATH_CALUDE_article_cost_price_l2602_260273


namespace NUMINAMATH_CALUDE_exactly_one_correct_probability_l2602_260260

theorem exactly_one_correct_probability
  (prob_A prob_B : ℝ)
  (h_A : prob_A = 0.8)
  (h_B : prob_B = 0.7)
  (h_independent : True)  -- Representing independence
  : prob_A * (1 - prob_B) + prob_B * (1 - prob_A) = 0.38 := by
  sorry

end NUMINAMATH_CALUDE_exactly_one_correct_probability_l2602_260260


namespace NUMINAMATH_CALUDE_circle_through_three_points_l2602_260275

/-- A circle in a 2D plane --/
structure Circle where
  /-- The coefficient of x^2 (always 1 for a standard form circle equation) --/
  a : ℝ := 1
  /-- The coefficient of y^2 (always 1 for a standard form circle equation) --/
  b : ℝ := 1
  /-- The coefficient of x --/
  d : ℝ
  /-- The coefficient of y --/
  e : ℝ
  /-- The constant term --/
  f : ℝ

/-- Check if a point (x, y) lies on the circle --/
def Circle.contains (c : Circle) (x y : ℝ) : Prop :=
  c.a * x^2 + c.b * y^2 + c.d * x + c.e * y + c.f = 0

/-- The theorem stating that there exists a unique circle passing through three given points --/
theorem circle_through_three_points :
  ∃! c : Circle,
    c.contains 0 0 ∧
    c.contains 1 1 ∧
    c.contains 4 2 ∧
    c.a = 1 ∧
    c.b = 1 ∧
    c.d = -8 ∧
    c.e = 6 ∧
    c.f = 0 := by
  sorry

end NUMINAMATH_CALUDE_circle_through_three_points_l2602_260275


namespace NUMINAMATH_CALUDE_point_in_region_l2602_260203

theorem point_in_region (m : ℝ) : 
  (2 * m + 3 * 1 - 5 > 0) ↔ (m > 1) := by
sorry

end NUMINAMATH_CALUDE_point_in_region_l2602_260203


namespace NUMINAMATH_CALUDE_overlap_area_l2602_260228

theorem overlap_area (total_length width : ℝ) 
  (left_length right_length : ℝ) 
  (left_only_area right_only_area : ℝ) :
  total_length = left_length + right_length →
  left_length = 9 →
  right_length = 7 →
  left_only_area = 27 →
  right_only_area = 18 →
  ∃ (overlap_area : ℝ), 
    overlap_area = 13.5 ∧
    (left_only_area + overlap_area) / (right_only_area + overlap_area) = left_length / right_length :=
by sorry

end NUMINAMATH_CALUDE_overlap_area_l2602_260228


namespace NUMINAMATH_CALUDE_length_of_CD_length_of_CD_explicit_l2602_260212

/-- Given two right triangles ABC and ABD sharing hypotenuse AB, 
    this theorem proves the length of CD. -/
theorem length_of_CD (a : ℝ) (h : a ≥ Real.sqrt 7) : ℝ :=
  let BC : ℝ := 3
  let AC : ℝ := a
  let AD : ℝ := 4
  let AB : ℝ := Real.sqrt (a^2 + 9)
  let BD : ℝ := Real.sqrt (a^2 - 7)
  |AD - BD|

/-- The length of CD is |4 - √(a² - 7)| -/
theorem length_of_CD_explicit (a : ℝ) (h : a ≥ Real.sqrt 7) :
  length_of_CD a h = |4 - Real.sqrt (a^2 - 7)| :=
sorry

end NUMINAMATH_CALUDE_length_of_CD_length_of_CD_explicit_l2602_260212


namespace NUMINAMATH_CALUDE_smallest_positive_period_l2602_260231

def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

theorem smallest_positive_period (f : ℝ → ℝ) 
  (h : ∀ x, f (3 * x) = f (3 * x - 3/2)) :
  ∃ T, T = 1/2 ∧ is_periodic f T ∧ ∀ T' > 0, is_periodic f T' → T ≤ T' :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_period_l2602_260231


namespace NUMINAMATH_CALUDE_sum_ac_equals_eight_l2602_260218

theorem sum_ac_equals_eight 
  (a b c d : ℝ) 
  (h1 : a * b + b * c + c * d + d * a = 48) 
  (h2 : b + d = 6) : 
  a + c = 8 := by
sorry

end NUMINAMATH_CALUDE_sum_ac_equals_eight_l2602_260218


namespace NUMINAMATH_CALUDE_prime_equation_unique_solution_l2602_260204

theorem prime_equation_unique_solution (p q : ℕ) :
  Prime p ∧ Prime q ∧ p^3 - q^5 = (p + q)^2 ↔ p = 7 ∧ q = 3 := by
  sorry

end NUMINAMATH_CALUDE_prime_equation_unique_solution_l2602_260204


namespace NUMINAMATH_CALUDE_max_k_value_l2602_260263

theorem max_k_value (x y k : ℝ) (pos_x : x > 0) (pos_y : y > 0) (pos_k : k > 0)
  (eq_condition : 3 = k^2 * (x^2 / y^2 + y^2 / x^2) + k * (x / y + y / x)) :
  k ≤ (-1 + Real.sqrt 7) / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_k_value_l2602_260263


namespace NUMINAMATH_CALUDE_rectangle_with_equal_adjacent_sides_is_square_l2602_260237

-- Define a rectangle
structure Rectangle :=
  (length : ℝ)
  (width : ℝ)
  (length_positive : length > 0)
  (width_positive : width > 0)

-- Define a square
structure Square :=
  (side : ℝ)
  (side_positive : side > 0)

-- Theorem: A rectangle with one pair of adjacent sides equal is a square
theorem rectangle_with_equal_adjacent_sides_is_square (r : Rectangle) 
  (h : r.length = r.width) : ∃ (s : Square), s.side = r.length :=
sorry

end NUMINAMATH_CALUDE_rectangle_with_equal_adjacent_sides_is_square_l2602_260237


namespace NUMINAMATH_CALUDE_coupon_probability_l2602_260251

theorem coupon_probability (n m k : ℕ) (h1 : n = 17) (h2 : m = 9) (h3 : k = 6) : 
  (Nat.choose k k * Nat.choose (n - k) (m - k)) / Nat.choose n m = 3 / 442 := by
  sorry

end NUMINAMATH_CALUDE_coupon_probability_l2602_260251


namespace NUMINAMATH_CALUDE_sphere_center_reciprocal_sum_l2602_260248

/-- Given a sphere with center (p,q,r) passing through the origin and three points on the axes,
    prove that the sum of reciprocals of its center coordinates equals 49/72 -/
theorem sphere_center_reciprocal_sum :
  ∀ (p q r : ℝ),
  (p^2 + q^2 + r^2 = p^2 + q^2 + r^2) ∧  -- Distance from center to origin
  (p^2 + q^2 + r^2 = (p-2)^2 + q^2 + r^2) ∧  -- Distance from center to (2,0,0)
  (p^2 + q^2 + r^2 = p^2 + (q-4)^2 + r^2) ∧  -- Distance from center to (0,4,0)
  (p^2 + q^2 + r^2 = p^2 + q^2 + (r-6)^2) →  -- Distance from center to (0,0,6)
  1/p + 1/q + 1/r = 49/72 := by
sorry

end NUMINAMATH_CALUDE_sphere_center_reciprocal_sum_l2602_260248


namespace NUMINAMATH_CALUDE_gym_class_counts_l2602_260240

/-- Given five gym classes with student counts P1, P2, P3, P4, and P5, prove that
    P2 = 5, P3 = 12.5, P4 = 25/3, and P5 = 25/3 given the following conditions:
    - P1 = 15
    - P1 = P2 + 10
    - P2 = 2 * P3 - 20
    - P3 = (P4 + P5) - 5
    - P4 = (1 / 2) * P5 + 5 -/
theorem gym_class_counts (P1 P2 P3 P4 P5 : ℚ) 
  (h1 : P1 = 15)
  (h2 : P1 = P2 + 10)
  (h3 : P2 = 2 * P3 - 20)
  (h4 : P3 = (P4 + P5) - 5)
  (h5 : P4 = (1 / 2) * P5 + 5) :
  P2 = 5 ∧ P3 = 25/2 ∧ P4 = 25/3 ∧ P5 = 25/3 := by
  sorry


end NUMINAMATH_CALUDE_gym_class_counts_l2602_260240


namespace NUMINAMATH_CALUDE_train_speed_before_accelerating_l2602_260227

/-- Calculates the average speed of a train before accelerating -/
theorem train_speed_before_accelerating
  (v : ℝ) (s : ℝ) 
  (h1 : v > 0) (h2 : s > 0) :
  ∃ (x : ℝ), x > 0 ∧ s / x = (s + 50) / (x + v) ∧ x = s * v / 50 :=
by sorry

end NUMINAMATH_CALUDE_train_speed_before_accelerating_l2602_260227


namespace NUMINAMATH_CALUDE_toy_organizer_price_correct_l2602_260268

/-- The price of a gaming chair in dollars -/
def gaming_chair_price : ℝ := 83

/-- The number of toy organizer sets ordered -/
def toy_organizer_sets : ℕ := 3

/-- The number of gaming chairs ordered -/
def gaming_chairs : ℕ := 2

/-- The delivery fee percentage -/
def delivery_fee_percent : ℝ := 0.05

/-- The total amount paid by Leon in dollars -/
def total_paid : ℝ := 420

/-- The price per set of toy organizers in dollars -/
def toy_organizer_price : ℝ := 78

theorem toy_organizer_price_correct :
  toy_organizer_price * toy_organizer_sets +
  gaming_chair_price * gaming_chairs +
  delivery_fee_percent * (toy_organizer_price * toy_organizer_sets + gaming_chair_price * gaming_chairs) =
  total_paid := by
  sorry

#check toy_organizer_price_correct

end NUMINAMATH_CALUDE_toy_organizer_price_correct_l2602_260268


namespace NUMINAMATH_CALUDE_smallest_valid_integer_N_divisible_by_1_to_28_N_not_divisible_by_29_or_30_N_is_smallest_valid_integer_l2602_260265

def N : ℕ := 2329089562800

theorem smallest_valid_integer (k : ℕ) (h : k < N) : 
  (∀ i ∈ Finset.range 28, k % (i + 1) = 0) → 
  (k % 29 ≠ 0 ∨ k % 30 ≠ 0) → 
  False :=
sorry

theorem N_divisible_by_1_to_28 : 
  ∀ i ∈ Finset.range 28, N % (i + 1) = 0 :=
sorry

theorem N_not_divisible_by_29_or_30 : 
  N % 29 ≠ 0 ∨ N % 30 ≠ 0 :=
sorry

theorem N_is_smallest_valid_integer : 
  ∀ k < N, 
  (∀ i ∈ Finset.range 28, k % (i + 1) = 0) → 
  (k % 29 ≠ 0 ∨ k % 30 ≠ 0) → 
  False :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_integer_N_divisible_by_1_to_28_N_not_divisible_by_29_or_30_N_is_smallest_valid_integer_l2602_260265


namespace NUMINAMATH_CALUDE_smith_family_seating_l2602_260295

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem smith_family_seating (boys girls : ℕ) (h : boys = 4 ∧ girls = 4) : 
  factorial (boys + girls) - (factorial boys * factorial girls) = 39744 :=
by sorry

end NUMINAMATH_CALUDE_smith_family_seating_l2602_260295


namespace NUMINAMATH_CALUDE_train_overtake_time_l2602_260255

/-- The time taken for a train to overtake a motorbike -/
theorem train_overtake_time (train_speed motorbike_speed : ℝ) (train_length : ℝ) :
  train_speed = 100 →
  motorbike_speed = 64 →
  train_length = 400.032 →
  (train_length / ((train_speed - motorbike_speed) * (1000 / 3600))) = 40.0032 := by
  sorry

end NUMINAMATH_CALUDE_train_overtake_time_l2602_260255


namespace NUMINAMATH_CALUDE_inspector_meter_count_l2602_260299

/-- Given an inspector who rejects 10% of meters as defective and finds 20 meters to be defective,
    the total number of meters examined is 200. -/
theorem inspector_meter_count (reject_rate : ℝ) (defective_count : ℕ) (total_count : ℕ) : 
  reject_rate = 0.1 →
  defective_count = 20 →
  (reject_rate : ℝ) * total_count = defective_count →
  total_count = 200 := by
  sorry

end NUMINAMATH_CALUDE_inspector_meter_count_l2602_260299


namespace NUMINAMATH_CALUDE_expression_evaluation_l2602_260291

theorem expression_evaluation : (3^2 - 3) - (4^2 - 4) + (5^2 - 5) - (6^2 - 6) = -16 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2602_260291


namespace NUMINAMATH_CALUDE_desk_lamp_profit_maximization_l2602_260266

/-- Represents the profit function for desk lamp sales -/
def profit_function (original_price cost_price initial_sales price_increase : ℝ) : ℝ → ℝ :=
  λ x => (original_price + x - cost_price) * (initial_sales - 10 * x)

theorem desk_lamp_profit_maximization 
  (original_price : ℝ) 
  (cost_price : ℝ) 
  (initial_sales : ℝ) 
  (price_range_min : ℝ) 
  (price_range_max : ℝ) 
  (h1 : original_price = 40)
  (h2 : cost_price = 30)
  (h3 : initial_sales = 600)
  (h4 : price_range_min = 40)
  (h5 : price_range_max = 60)
  (h6 : price_range_min ≤ price_range_max) :
  (∃ x : ℝ, x = 10 ∧ profit_function original_price cost_price initial_sales x = 10000) ∧
  (∀ y : ℝ, price_range_min ≤ original_price + y ∧ original_price + y ≤ price_range_max →
    profit_function original_price cost_price initial_sales y ≤ 
    profit_function original_price cost_price initial_sales (price_range_max - original_price)) :=
by sorry

end NUMINAMATH_CALUDE_desk_lamp_profit_maximization_l2602_260266


namespace NUMINAMATH_CALUDE_temperature_difference_l2602_260219

def lowest_temp_beijing : Int := -10
def lowest_temp_hangzhou : Int := -1

theorem temperature_difference :
  lowest_temp_beijing - lowest_temp_hangzhou = 9 := by
  sorry

end NUMINAMATH_CALUDE_temperature_difference_l2602_260219


namespace NUMINAMATH_CALUDE_u_less_than_v_l2602_260259

theorem u_less_than_v (u v : ℝ) 
  (hu : (u + u^2 + u^3 + u^4 + u^5 + u^6 + u^7 + u^8) + 10*u^9 = 8)
  (hv : (v + v^2 + v^3 + v^4 + v^5 + v^6 + v^7 + v^8 + v^9 + v^10) + 10*v^11 = 8) :
  u < v := by
sorry

end NUMINAMATH_CALUDE_u_less_than_v_l2602_260259


namespace NUMINAMATH_CALUDE_cats_remaining_l2602_260229

/-- The number of cats remaining after a sale at a pet store -/
theorem cats_remaining (siamese : ℕ) (house : ℕ) (sold : ℕ) : 
  siamese = 15 → house = 49 → sold = 19 → siamese + house - sold = 45 := by
  sorry

end NUMINAMATH_CALUDE_cats_remaining_l2602_260229


namespace NUMINAMATH_CALUDE_no_real_solution_l2602_260296

theorem no_real_solution : ¬∃ (x y : ℝ), x^3 + y^2 = 2 ∧ x^2 + x*y + y^2 - y = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solution_l2602_260296


namespace NUMINAMATH_CALUDE_nested_radical_value_l2602_260206

theorem nested_radical_value :
  ∃ y : ℝ, y = Real.sqrt (4 + y) → y = (1 + Real.sqrt 17) / 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_radical_value_l2602_260206


namespace NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l2602_260277

/-- An infinite geometric series with common ratio 1/4 and sum 40 has a first term of 30. -/
theorem infinite_geometric_series_first_term
  (r : ℝ) (S : ℝ) (a : ℝ)
  (h_r : r = 1/4)
  (h_S : S = 40)
  (h_sum : S = a / (1 - r))
  : a = 30 := by
  sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l2602_260277


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2602_260208

/-- The sequence a_n defined by n^2 - c*n for n ∈ ℕ+ is increasing -/
def is_increasing_sequence (c : ℝ) : Prop :=
  ∀ n : ℕ+, (n + 1)^2 - c*(n + 1) > n^2 - c*n

/-- c ≤ 2 is a sufficient but not necessary condition for the sequence to be increasing -/
theorem sufficient_not_necessary_condition :
  (∀ c : ℝ, c ≤ 2 → is_increasing_sequence c) ∧
  (∃ c : ℝ, c > 2 ∧ is_increasing_sequence c) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2602_260208


namespace NUMINAMATH_CALUDE_square_of_two_digit_number_ending_in_five_l2602_260235

theorem square_of_two_digit_number_ending_in_five (d : ℕ) 
  (h : d ≥ 1 ∧ d ≤ 9) : 
  (10 * d + 5)^2 = 100 * d * (d + 1) + 25 := by
  sorry

end NUMINAMATH_CALUDE_square_of_two_digit_number_ending_in_five_l2602_260235


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l2602_260209

theorem simplify_and_evaluate (x : ℕ) (h1 : x > 0) (h2 : 3 - x ≥ 0) :
  let expr := (1 - 1 / (x - 1)) / ((x^2 - 4*x + 4) / (x - 1))
  x = 3 → expr = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l2602_260209


namespace NUMINAMATH_CALUDE_number_of_men_l2602_260242

/-- Proves that the number of men is 15 given the specified conditions -/
theorem number_of_men (men women boys : ℕ) (total_earnings men_wage : ℕ) : 
  men = women ∧ women = 8 ∧ 
  total_earnings = 120 ∧
  men_wage = 8 ∧
  total_earnings = men_wage * men →
  men = 15 := by
sorry

end NUMINAMATH_CALUDE_number_of_men_l2602_260242


namespace NUMINAMATH_CALUDE_product_102_108_l2602_260274

theorem product_102_108 : 102 * 108 = 11016 := by
  sorry

end NUMINAMATH_CALUDE_product_102_108_l2602_260274


namespace NUMINAMATH_CALUDE_product_of_sum_and_sum_of_cubes_l2602_260287

theorem product_of_sum_and_sum_of_cubes (a b : ℝ) 
  (sum_eq : a + b = 5)
  (sum_of_cubes_eq : a^3 + b^3 = 125) : 
  a * b = 0 := by
sorry

end NUMINAMATH_CALUDE_product_of_sum_and_sum_of_cubes_l2602_260287


namespace NUMINAMATH_CALUDE_fraction_inequality_l2602_260247

theorem fraction_inequality (a b m : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : m > 0) :
  b / a < (b + m) / (a + m) := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l2602_260247


namespace NUMINAMATH_CALUDE_tom_running_distance_l2602_260280

/-- Calculates the total distance run in a week given the number of days, hours per day, and speed. -/
def weekly_distance (days : ℕ) (hours_per_day : ℝ) (speed : ℝ) : ℝ :=
  days * hours_per_day * speed

/-- Proves that running 5 days a week, 1.5 hours each day, at 8 mph results in 60 miles per week. -/
theorem tom_running_distance :
  weekly_distance 5 1.5 8 = 60 := by
  sorry

end NUMINAMATH_CALUDE_tom_running_distance_l2602_260280


namespace NUMINAMATH_CALUDE_presentation_students_l2602_260244

/-- The number of students in a presentation, given Eunjeong's position -/
def total_students (students_in_front : ℕ) (position_from_back : ℕ) : ℕ :=
  students_in_front + 1 + (position_from_back - 1)

/-- Theorem stating the total number of students in the presentation -/
theorem presentation_students : total_students 7 6 = 13 := by
  sorry

end NUMINAMATH_CALUDE_presentation_students_l2602_260244


namespace NUMINAMATH_CALUDE_polynomial_remainder_l2602_260261

/-- Given a polynomial p(x) with the specified division properties, 
    prove that its remainder when divided by (x + 1)(x + 3) is (7/2)x + 13/2 -/
theorem polynomial_remainder (p : Polynomial ℚ) 
  (h1 : (p - 3).eval (-1) = 0)
  (h2 : (p + 4).eval (-3) = 0) :
  ∃ q : Polynomial ℚ, p = q * ((X + 1) * (X + 3)) + (7/2 * X + 13/2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l2602_260261


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2602_260249

theorem complex_equation_solution (c d : ℂ) (x : ℝ) 
  (h1 : Complex.abs c = 3)
  (h2 : Complex.abs d = 5)
  (h3 : c * d = x - 6 * Complex.I) :
  x = 3 * Real.sqrt 21 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2602_260249


namespace NUMINAMATH_CALUDE_quadratic_function_unique_form_l2602_260233

/-- A quadratic function is a function of the form f(x) = ax² + bx + c where a ≠ 0 -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

theorem quadratic_function_unique_form
  (f : ℝ → ℝ)
  (hquad : QuadraticFunction f)
  (hf0 : f 0 = 1)
  (hfdiff : ∀ x, f (x + 1) - f x = 4 * x) :
  ∀ x, f x = 2 * x^2 - 2 * x + 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_unique_form_l2602_260233


namespace NUMINAMATH_CALUDE_length_of_PT_l2602_260262

/-- Given points P, Q, R, S, and T in a coordinate plane where PQ intersects RS at T,
    and the x-coordinate difference between P and Q is 6,
    and the y-coordinate difference between P and Q is 4,
    prove that the length of segment PT is 12√13/11 -/
theorem length_of_PT (P Q R S T : ℝ × ℝ) : 
  (∃ t : ℝ, 0 < t ∧ t < 1 ∧ 
    T = (1 - t) • P + t • Q ∧
    T = (1 - t) • R + t • S) →
  Q.1 - P.1 = 6 →
  Q.2 - P.2 = 4 →
  Real.sqrt ((T.1 - P.1)^2 + (T.2 - P.2)^2) = 12 * Real.sqrt 13 / 11 :=
by sorry

end NUMINAMATH_CALUDE_length_of_PT_l2602_260262


namespace NUMINAMATH_CALUDE_physics_class_size_l2602_260289

theorem physics_class_size (total_students : ℕ) 
  (math_only : ℚ) (physics_only : ℚ) (both : ℕ) :
  total_students = 100 →
  both = 10 →
  physics_only + both = 2 * (math_only + both) →
  math_only + physics_only + both = total_students →
  physics_only + both = 220 / 3 := by
  sorry

end NUMINAMATH_CALUDE_physics_class_size_l2602_260289


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2602_260254

/-- A hyperbola with given eccentricity and distance from focus to asymptote -/
structure Hyperbola where
  e : ℝ  -- eccentricity
  b : ℝ  -- distance from focus to asymptote

/-- The standard equation of a hyperbola -/
def standard_equation (h : Hyperbola) : Prop :=
  ∃ (x y : ℝ), y^2 / 2 - x^2 / 4 = 1

/-- Theorem: For a hyperbola with eccentricity √3 and distance from focus to asymptote 2,
    the standard equation is y²/2 - x²/4 = 1 -/
theorem hyperbola_equation (h : Hyperbola) 
    (h_e : h.e = Real.sqrt 3) 
    (h_b : h.b = 2) : 
    standard_equation h := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2602_260254


namespace NUMINAMATH_CALUDE_gum_cost_theorem_l2602_260205

/-- The price of one piece of gum in cents -/
def price_per_piece : ℕ := 2

/-- The number of pieces of gum being purchased -/
def num_pieces : ℕ := 5000

/-- The discount rate as a decimal -/
def discount_rate : ℚ := 1/20

/-- The minimum number of pieces required for the discount to apply -/
def discount_threshold : ℕ := 4000

/-- Calculates the final cost in dollars after applying the discount if applicable -/
def final_cost : ℚ :=
  let total_cents := price_per_piece * num_pieces
  let discounted_cents := if num_pieces > discount_threshold
                          then total_cents * (1 - discount_rate)
                          else total_cents
  discounted_cents / 100

theorem gum_cost_theorem :
  final_cost = 95 := by sorry

end NUMINAMATH_CALUDE_gum_cost_theorem_l2602_260205


namespace NUMINAMATH_CALUDE_simplify_complex_expression_l2602_260222

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem simplify_complex_expression : 3 * (2 - i) + i * (3 + 2 * i) = (4 : ℂ) := by
  sorry

end NUMINAMATH_CALUDE_simplify_complex_expression_l2602_260222


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l2602_260256

theorem cube_root_equation_solution :
  ∃! x : ℝ, (3 - x)^(1/3 : ℝ) = -(3/2) ∧ x = 51/8 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l2602_260256


namespace NUMINAMATH_CALUDE_hyperbola_intersection_x_coordinate_l2602_260278

theorem hyperbola_intersection_x_coordinate :
  ∀ x y : ℝ,
  (Real.sqrt ((x - 5)^2 + y^2) - Real.sqrt ((x + 5)^2 + y^2) = 6) →
  (y = 4) →
  (x = -3 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_intersection_x_coordinate_l2602_260278


namespace NUMINAMATH_CALUDE_inequality_proof_l2602_260297

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a + b + c = 1/a + 1/b + 1/c) : a + b + c ≥ 3/(a*b*c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2602_260297


namespace NUMINAMATH_CALUDE_speed_difference_l2602_260243

/-- The difference in average speeds between two travelers -/
theorem speed_difference (distance : ℝ) (time1 time2 : ℝ) :
  distance > 0 ∧ time1 > 0 ∧ time2 > 0 →
  distance = 15 ∧ time1 = 1/3 ∧ time2 = 1/4 →
  (distance / time2) - (distance / time1) = 15 := by
  sorry

#check speed_difference

end NUMINAMATH_CALUDE_speed_difference_l2602_260243


namespace NUMINAMATH_CALUDE_parallel_alternate_interior_false_l2602_260201

-- Define the concept of lines
structure Line :=
  (slope : ℝ)
  (intercept : ℝ)

-- Define the concept of angles
structure Angle :=
  (measure : ℝ)

-- Define parallel lines
def parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

-- Define alternate interior angles
def alternate_interior (a1 a2 : Angle) (l1 l2 : Line) : Prop :=
  -- This definition is simplified for the purpose of this statement
  true

-- The theorem to be proved
theorem parallel_alternate_interior_false :
  ∃ (l1 l2 : Line) (a1 a2 : Angle),
    parallel l1 l2 ∧ ¬(alternate_interior a1 a2 l1 l2 → a1.measure = a2.measure) :=
sorry

end NUMINAMATH_CALUDE_parallel_alternate_interior_false_l2602_260201


namespace NUMINAMATH_CALUDE_simpsons_formula_volume_l2602_260283

/-- Simpson's formula for volume calculation -/
theorem simpsons_formula_volume
  (S : ℝ → ℝ) -- Cross-sectional area function
  (x₀ x₁ : ℝ) -- Start and end coordinates
  (h : ℝ) -- Height of the figure
  (hpos : 0 < h) -- Height is positive
  (hdiff : h = x₁ - x₀) -- Height definition
  (hquad : ∃ (a b c : ℝ), ∀ x, S x = a * x^2 + b * x + c) -- S is a quadratic polynomial
  :
  (∫ (x : ℝ) in x₀..x₁, S x) = 
    (h / 6) * (S x₀ + 4 * S ((x₀ + x₁) / 2) + S x₁) :=
by sorry

end NUMINAMATH_CALUDE_simpsons_formula_volume_l2602_260283


namespace NUMINAMATH_CALUDE_crude_oil_temperature_l2602_260213

-- Define the function f(x) = x^2 - 7x + 15
def f (x : ℝ) : ℝ := x^2 - 7*x + 15

-- Define the domain of f
def domain : Set ℝ := { x | 0 ≤ x ∧ x ≤ 8 }

theorem crude_oil_temperature (x : ℝ) (h : x ∈ domain) :
  -- The derivative of f at x = 4 is 1
  deriv f 4 = 1 ∧
  -- The function is increasing at x = 4
  0 < deriv f 4 := by
  sorry

end NUMINAMATH_CALUDE_crude_oil_temperature_l2602_260213


namespace NUMINAMATH_CALUDE_complex_fraction_sum_l2602_260214

theorem complex_fraction_sum : (2 + 2 * Complex.I) / Complex.I + (1 + Complex.I) / (1 - Complex.I) = 2 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_l2602_260214


namespace NUMINAMATH_CALUDE_min_value_2x_plus_y_l2602_260292

theorem min_value_2x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x^2 + 2*x*y - 3 = 0) :
  2*x + y ≥ 3 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀^2 + 2*x₀*y₀ - 3 = 0 ∧ 2*x₀ + y₀ = 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_2x_plus_y_l2602_260292


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l2602_260285

theorem absolute_value_equation_solution :
  ∀ x : ℝ, (|x - 3| = 5 - 2*x) ↔ (x = 2 ∨ x = 8/3) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l2602_260285


namespace NUMINAMATH_CALUDE_product_negative_five_sum_options_l2602_260272

theorem product_negative_five_sum_options (a b c : ℤ) : 
  a * b * c = -5 → (a + b + c = 5 ∨ a + b + c = -3 ∨ a + b + c = -7) := by
sorry

end NUMINAMATH_CALUDE_product_negative_five_sum_options_l2602_260272


namespace NUMINAMATH_CALUDE_fencing_cost_proof_l2602_260207

/-- Calculates the total cost of fencing a rectangular plot -/
def total_fencing_cost (length : ℝ) (breadth : ℝ) (cost_per_meter : ℝ) : ℝ :=
  2 * (length + breadth) * cost_per_meter

/-- Proves that the total cost of fencing the given rectangular plot is 5300 currency units -/
theorem fencing_cost_proof :
  let length : ℝ := 66
  let breadth : ℝ := 34
  let cost_per_meter : ℝ := 26.50
  total_fencing_cost length breadth cost_per_meter = 5300 := by
  sorry

end NUMINAMATH_CALUDE_fencing_cost_proof_l2602_260207


namespace NUMINAMATH_CALUDE_girls_picked_more_l2602_260286

-- Define the number of mushrooms picked by each person
variable (N I A V : ℕ)

-- Define the conditions
def natasha_most := N > I ∧ N > A ∧ N > V
def ira_not_least := I ≤ N ∧ I ≥ A ∧ I ≥ V
def alexey_more_than_vitya := A > V

-- Theorem to prove
theorem girls_picked_more (h1 : natasha_most N I A V) 
                          (h2 : ira_not_least N I A V) 
                          (h3 : alexey_more_than_vitya A V) : 
  N + I > A + V := by
  sorry

end NUMINAMATH_CALUDE_girls_picked_more_l2602_260286


namespace NUMINAMATH_CALUDE_inequality_implies_lower_bound_l2602_260238

theorem inequality_implies_lower_bound (a : ℝ) :
  (∀ x : ℝ, |x - 2| - |x + 3| ≤ a) → a ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implies_lower_bound_l2602_260238


namespace NUMINAMATH_CALUDE_return_percentage_is_80_percent_l2602_260293

/-- Represents the library book collection and loan statistics --/
structure LibraryStats where
  initial_books : ℕ
  final_books : ℕ
  loaned_books : ℕ

/-- Calculates the percentage of loaned books that were returned --/
def return_percentage (stats : LibraryStats) : ℚ :=
  ((stats.final_books - (stats.initial_books - stats.loaned_books)) / stats.loaned_books) * 100

/-- Theorem stating that the return percentage is 80% for the given statistics --/
theorem return_percentage_is_80_percent (stats : LibraryStats) 
  (h1 : stats.initial_books = 75)
  (h2 : stats.final_books = 64)
  (h3 : stats.loaned_books = 55) : 
  return_percentage stats = 80 := by
  sorry

end NUMINAMATH_CALUDE_return_percentage_is_80_percent_l2602_260293
