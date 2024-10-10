import Mathlib

namespace factor_sum_l252_25275

theorem factor_sum (P Q : ℝ) : 
  (∃ b c : ℝ, (X^2 + 4*X + 3) * (X^2 + b*X + c) = X^4 + P*X^2 + Q) →
  P + Q = -1 :=
by sorry

end factor_sum_l252_25275


namespace population_after_10_years_l252_25295

/-- The population growth over a period of years -/
def population_growth (M : ℝ) (p : ℝ) (years : ℕ) : ℝ :=
  M * (1 + p) ^ years

/-- Theorem: The population after 10 years given initial population M and growth rate p -/
theorem population_after_10_years (M : ℝ) (p : ℝ) :
  population_growth M p 10 = M * (1 + p)^10 := by
  sorry

end population_after_10_years_l252_25295


namespace usual_time_to_catch_bus_l252_25297

/-- Proves that given a person who misses a bus by 6 minutes when walking at 4/5 of their usual speed, their usual time to catch the bus is 24 minutes. -/
theorem usual_time_to_catch_bus (usual_speed : ℝ) (usual_time : ℝ) 
  (h1 : usual_speed > 0) 
  (h2 : usual_time > 0)
  (h3 : (4/5 * usual_speed) * (usual_time + 6) = usual_speed * usual_time) : 
  usual_time = 24 := by
sorry


end usual_time_to_catch_bus_l252_25297


namespace marbles_bet_thirteen_l252_25237

/-- Calculates the number of marbles bet per game -/
def marbles_bet_per_game (friend_start : ℕ) (total_games : ℕ) (reggie_end : ℕ) (reggie_losses : ℕ) : ℕ :=
  ((friend_start - reggie_end) / (total_games - 2 * reggie_losses)).succ

/-- Proves that under the given conditions, 13 marbles were bet per game -/
theorem marbles_bet_thirteen (friend_start : ℕ) (total_games : ℕ) (reggie_end : ℕ) (reggie_losses : ℕ)
  (h1 : friend_start = 100)
  (h2 : total_games = 9)
  (h3 : reggie_end = 90)
  (h4 : reggie_losses = 1) :
  marbles_bet_per_game friend_start total_games reggie_end reggie_losses = 13 := by
  sorry

#eval marbles_bet_per_game 100 9 90 1

end marbles_bet_thirteen_l252_25237


namespace sams_shirts_l252_25255

theorem sams_shirts (short_sleeve : ℕ) (long_sleeve : ℕ) (washed : ℕ) (unwashed : ℕ) : 
  long_sleeve = 23 →
  washed = 29 →
  unwashed = 34 →
  short_sleeve + long_sleeve = washed + unwashed →
  short_sleeve = 40 := by
sorry

end sams_shirts_l252_25255


namespace new_person_weight_l252_25291

theorem new_person_weight (initial_count : ℕ) (replaced_weight : ℝ) (avg_increase : ℝ) :
  initial_count = 10 →
  replaced_weight = 50 →
  avg_increase = 2.5 →
  ∃ (new_weight : ℝ),
    new_weight = initial_count * avg_increase + replaced_weight ∧
    new_weight = 75 := by
  sorry

end new_person_weight_l252_25291


namespace quadratic_inequality_solution_set_l252_25266

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 - 6*x > 20} = {x : ℝ | x < -2 ∨ x > 10} := by
  sorry

end quadratic_inequality_solution_set_l252_25266


namespace optimal_play_result_l252_25272

/-- Represents a square on the chessboard --/
structure Square where
  x : Fin 8
  y : Fin 8

/-- Represents the state of the chessboard --/
def Chessboard := Square → Bool

/-- Checks if two squares are neighbors --/
def are_neighbors (s1 s2 : Square) : Bool :=
  (s1.x = s2.x ∧ s1.y.val + 1 = s2.y.val) ∨
  (s1.x = s2.x ∧ s1.y.val = s2.y.val + 1) ∨
  (s1.x.val + 1 = s2.x.val ∧ s1.y = s2.y) ∨
  (s1.x.val = s2.x.val + 1 ∧ s1.y = s2.y)

/-- Counts the number of black connected components on the board --/
def count_black_components (board : Chessboard) : Nat :=
  sorry

/-- Represents a move in the game --/
inductive Move
| alice : Square → Move
| bob : Option Square → Move

/-- Applies a move to the chessboard --/
def apply_move (board : Chessboard) (move : Move) : Chessboard :=
  sorry

/-- Represents the game state --/
structure GameState where
  board : Chessboard
  alice_turn : Bool

/-- Checks if the game is over --/
def is_game_over (state : GameState) : Bool :=
  sorry

/-- Returns the optimal move for the current player --/
def optimal_move (state : GameState) : Move :=
  sorry

/-- Plays the game optimally from the given state until it's over --/
def play_game (state : GameState) : Nat :=
  sorry

/-- The main theorem: optimal play results in 16 black connected components --/
theorem optimal_play_result :
  let initial_state : GameState := {
    board := λ _ => false,  -- All squares are initially white
    alice_turn := true
  }
  play_game initial_state = 16 := by
  sorry

end optimal_play_result_l252_25272


namespace equation_solution_l252_25216

theorem equation_solution : 
  {x : ℝ | x * (x - 3)^2 * (5 - x) = 0} = {0, 3, 5} := by
sorry

end equation_solution_l252_25216


namespace javier_first_throw_l252_25265

/-- Represents the distance of Javier's second throw before adjustments -/
def second_throw : ℝ := sorry

/-- Calculates the adjusted distance of a throw -/
def adjusted_distance (base : ℝ) (wind_reduction : ℝ) (incline : ℝ) : ℝ :=
  base * (1 - wind_reduction) - incline

theorem javier_first_throw :
  let first_throw := 2 * second_throw
  let third_throw := 2 * first_throw
  adjusted_distance first_throw 0.05 2 +
  adjusted_distance second_throw 0.08 4 +
  adjusted_distance third_throw 0 1 = 1050 →
  first_throw = 310 := by sorry

end javier_first_throw_l252_25265


namespace count_pairs_20_l252_25261

def count_pairs (n : ℕ) : ℕ :=
  (n - 11) * (n - 11 + 1) / 2

theorem count_pairs_20 :
  count_pairs 20 = 45 :=
sorry

end count_pairs_20_l252_25261


namespace intersection_empty_range_necessary_not_sufficient_range_l252_25278

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | |x - a| ≤ 4}
def B : Set ℝ := {x | (x - 2) * (x - 3) ≤ 0}

-- Theorem 1: If A ∩ B = ∅, then a ∈ (-∞, -2) ∪ (7, ∞)
theorem intersection_empty_range (a : ℝ) : 
  A a ∩ B = ∅ → a < -2 ∨ a > 7 := by sorry

-- Theorem 2: If B is a necessary but not sufficient condition for A, then a ∈ [1, 6]
theorem necessary_not_sufficient_range (a : ℝ) :
  (B ⊆ A a ∧ ¬(A a ⊆ B)) → 1 ≤ a ∧ a ≤ 6 := by sorry

end intersection_empty_range_necessary_not_sufficient_range_l252_25278


namespace jason_money_unchanged_l252_25245

/-- Represents the money situation of Fred and Jason -/
structure MoneySituation where
  fred_initial : ℕ
  jason_initial : ℕ
  fred_final : ℕ
  total_earned : ℕ

/-- The theorem stating that Jason's final money is equal to his initial money -/
theorem jason_money_unchanged (situation : MoneySituation) 
  (h1 : situation.fred_initial = 111)
  (h2 : situation.jason_initial = 40)
  (h3 : situation.fred_final = 115)
  (h4 : situation.total_earned = 4) :
  situation.jason_initial = 40 := by
  sorry

#check jason_money_unchanged

end jason_money_unchanged_l252_25245


namespace defective_pens_count_l252_25234

/-- The number of pens in the box -/
def total_pens : ℕ := 16

/-- The probability of selecting two non-defective pens -/
def prob_two_non_defective : ℚ := 65/100

/-- The number of defective pens in the box -/
def defective_pens : ℕ := 3

/-- Theorem stating that given the total number of pens and the probability of
    selecting two non-defective pens, the number of defective pens is 3 -/
theorem defective_pens_count (n : ℕ) (h1 : n = total_pens) 
  (h2 : (n - defective_pens : ℚ) / n * ((n - defective_pens - 1) : ℚ) / (n - 1) = prob_two_non_defective) :
  defective_pens = 3 := by
  sorry

#eval defective_pens

end defective_pens_count_l252_25234


namespace remainder_6n_mod_4_l252_25285

theorem remainder_6n_mod_4 (n : ℤ) (h : n % 4 = 1) : (6 * n) % 4 = 2 := by
  sorry

end remainder_6n_mod_4_l252_25285


namespace adams_forgotten_lawns_l252_25240

/-- Adam's lawn mowing problem -/
theorem adams_forgotten_lawns (dollars_per_lawn : ℕ) (total_lawns : ℕ) (actual_earnings : ℕ) :
  dollars_per_lawn = 9 →
  total_lawns = 12 →
  actual_earnings = 36 →
  total_lawns - (actual_earnings / dollars_per_lawn) = 8 := by
  sorry

end adams_forgotten_lawns_l252_25240


namespace factorization_equality_l252_25201

theorem factorization_equality (m n : ℝ) : 8*m*n - 2*m*n^3 = 2*m*n*(2+n)*(2-n) := by sorry

end factorization_equality_l252_25201


namespace usamo_page_count_l252_25259

theorem usamo_page_count (a₁ a₂ a₃ a₄ a₅ a₆ : ℕ+) :
  (((a₁ : ℝ) + 1) / 2 + ((a₂ : ℝ) + 1) / 2 + ((a₃ : ℝ) + 1) / 2 + 
   ((a₄ : ℝ) + 1) / 2 + ((a₅ : ℝ) + 1) / 2 + ((a₆ : ℝ) + 1) / 2) = 2017 →
  (a₁ : ℕ) + a₂ + a₃ + a₄ + a₅ + a₆ = 4028 := by
  sorry

end usamo_page_count_l252_25259


namespace hyperbola_vertex_to_asymptote_distance_l252_25246

/-- Given a hyperbola with equation x²/a² - y²/3 = 1 and eccentricity 2,
    the distance from its vertex to its asymptote is √3/2 -/
theorem hyperbola_vertex_to_asymptote_distance
  (a : ℝ) -- Semi-major axis
  (h1 : a > 0) -- a is positive
  (h2 : (a^2 + 3) / a^2 = 4) -- Eccentricity condition
  : Real.sqrt 3 / 2 = 
    abs (-Real.sqrt 3 * a) / Real.sqrt (3 + 1) := by sorry

end hyperbola_vertex_to_asymptote_distance_l252_25246


namespace oliver_final_amount_l252_25268

/-- Calculates the final amount of money Oliver has after all transactions. -/
def olivers_money (initial : ℕ) (saved : ℕ) (frisbee_cost : ℕ) (puzzle_cost : ℕ) (gift : ℕ) : ℕ :=
  initial + saved - frisbee_cost - puzzle_cost + gift

/-- Theorem stating that Oliver's final amount of money is $15. -/
theorem oliver_final_amount :
  olivers_money 9 5 4 3 8 = 15 := by
  sorry

#eval olivers_money 9 5 4 3 8

end oliver_final_amount_l252_25268


namespace diophantine_equation_solution_l252_25221

theorem diophantine_equation_solution : ∃ (a b c : ℕ), a^3 + b^4 = c^5 ∧ a = 4 ∧ b = 16 ∧ c = 18 := by
  sorry

end diophantine_equation_solution_l252_25221


namespace intersection_of_A_and_B_l252_25254

open Set

-- Define the sets A and B
def A : Set ℝ := { x | 2 < x ∧ x < 4 }
def B : Set ℝ := { x | x < 3 ∨ x > 5 }

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = { x | 2 < x ∧ x < 3 } := by
  sorry

end intersection_of_A_and_B_l252_25254


namespace only_345_is_right_triangle_pythagoras_345_l252_25242

/-- A function that checks if three numbers can form a right triangle --/
def isRightTriangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2

/-- The theorem stating that only one of the given sets forms a right triangle --/
theorem only_345_is_right_triangle :
  ¬ isRightTriangle 1 1 (Real.sqrt 3) ∧
  isRightTriangle 3 4 5 ∧
  ¬ isRightTriangle 2 3 4 ∧
  ¬ isRightTriangle 5 7 9 :=
by sorry

/-- The specific theorem for the (3, 4, 5) right triangle --/
theorem pythagoras_345 : 3^2 + 4^2 = 5^2 :=
by sorry

end only_345_is_right_triangle_pythagoras_345_l252_25242


namespace monic_quadratic_with_complex_root_l252_25219

theorem monic_quadratic_with_complex_root :
  ∃ (a b : ℝ), ∀ (x : ℂ), x^2 + a*x + b = 0 ↔ x = 2 - 3*I ∨ x = 2 + 3*I :=
by
  -- The proof would go here
  sorry

end monic_quadratic_with_complex_root_l252_25219


namespace area_of_triangle_ACF_l252_25211

/-- Right triangle with sides a, b, and c -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_angle : a^2 + b^2 = c^2

/-- Rectangle with width w and height h -/
structure Rectangle where
  w : ℝ
  h : ℝ

theorem area_of_triangle_ACF (ABC : RightTriangle) (ABD : RightTriangle) (BCEF : Rectangle) :
  ABC.a = 8 →
  ABC.c = 12 →
  ABD.a = 8 →
  ABD.b = 8 →
  BCEF.w = 8 →
  BCEF.h = 8 →
  ABC.a = ABD.a →
  (1/2) * ABC.a * ABC.c = 24 := by
  sorry

#check area_of_triangle_ACF

end area_of_triangle_ACF_l252_25211


namespace red_apples_count_l252_25252

theorem red_apples_count (total green yellow : ℕ) (h1 : total = 19) (h2 : green = 2) (h3 : yellow = 14) :
  total - (green + yellow) = 3 := by
  sorry

end red_apples_count_l252_25252


namespace frustum_volume_l252_25292

/-- The volume of a frustum obtained from a cone with specific properties -/
theorem frustum_volume (r h l : ℝ) : 
  r > 0 ∧ h > 0 ∧ l > 0 ∧  -- Positive dimensions
  π * r * l = 2 * π ∧     -- Lateral surface area condition
  l = 2 * r ∧             -- Relationship between slant height and radius
  h = Real.sqrt 3 ∧       -- Height of the cone
  r = 1 →                 -- Radius of the cone
  (7 * Real.sqrt 3 * π) / 24 = 
    (7 / 8) * ((π * r^2 * h) / 3) := by
  sorry

end frustum_volume_l252_25292


namespace parallel_transitivity_counterexample_l252_25284

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for lines and planes
variable (parallelLine : Line → Line → Prop)
variable (parallelLinePlane : Line → Plane → Prop)

-- State the theorem
theorem parallel_transitivity_counterexample 
  (l n : Line) (α : Plane) : 
  ¬(∀ l n α, parallelLinePlane l α → parallelLinePlane n α → parallelLine l n) :=
sorry

end parallel_transitivity_counterexample_l252_25284


namespace fencing_cost_calculation_l252_25217

/-- Represents a rectangular plot with given dimensions and fencing cost -/
structure RectangularPlot where
  length : ℝ
  breadth : ℝ
  fencing_cost_per_meter : ℝ

/-- Calculates the total cost of fencing for a rectangular plot -/
def total_fencing_cost (plot : RectangularPlot) : ℝ :=
  2 * (plot.length + plot.breadth) * plot.fencing_cost_per_meter

/-- Theorem stating the total fencing cost for a specific rectangular plot -/
theorem fencing_cost_calculation :
  let plot : RectangularPlot :=
    { length := 58
    , breadth := 58 - 16
    , fencing_cost_per_meter := 26.5 }
  total_fencing_cost plot = 5300 := by
  sorry


end fencing_cost_calculation_l252_25217


namespace sector_arc_length_l252_25258

/-- Given a circular sector with a central angle of 40° and a radius of 18,
    the arc length is equal to 4π. -/
theorem sector_arc_length (θ : ℝ) (r : ℝ) (h1 : θ = 40) (h2 : r = 18) :
  (θ / 360) * (2 * π * r) = 4 * π := by
  sorry

end sector_arc_length_l252_25258


namespace tournament_committee_count_l252_25203

/-- The number of teams in the league -/
def num_teams : ℕ := 4

/-- The number of members in each team -/
def team_size : ℕ := 8

/-- The size of the tournament committee -/
def committee_size : ℕ := 10

/-- The number of members selected from the host team -/
def host_selection : ℕ := 4

/-- The number of members selected from each non-host team -/
def non_host_selection : ℕ := 2

/-- The number of possible tournament committees -/
def num_committees : ℕ := 6146560

theorem tournament_committee_count :
  (num_teams : ℕ) *
  (Nat.choose team_size host_selection) *
  (Nat.choose team_size non_host_selection ^ (num_teams - 1)) =
  num_committees := by sorry

end tournament_committee_count_l252_25203


namespace sector_radius_l252_25230

theorem sector_radius (l a : ℝ) (hl : l = 10 * Real.pi) (ha : a = 60 * Real.pi) :
  ∃ r : ℝ, r = 12 ∧ a = (1 / 2) * l * r := by
  sorry

end sector_radius_l252_25230


namespace babysitting_earnings_l252_25290

def payment_cycle : List Nat := [1, 2, 3, 4, 5, 6, 7]

def total_earnings (hours : Nat) : Nat :=
  let full_cycles := hours / payment_cycle.length
  let remaining_hours := hours % payment_cycle.length
  full_cycles * payment_cycle.sum + (payment_cycle.take remaining_hours).sum

theorem babysitting_earnings :
  total_earnings 25 = 94 := by
  sorry

end babysitting_earnings_l252_25290


namespace five_volunteers_four_projects_l252_25213

/-- The number of ways to allocate volunteers to projects -/
def allocation_schemes (n : ℕ) (k : ℕ) : ℕ :=
  (n.choose 2) * (k.factorial)

/-- Theorem stating the number of allocation schemes for 5 volunteers and 4 projects -/
theorem five_volunteers_four_projects :
  allocation_schemes 5 4 = 240 :=
sorry

end five_volunteers_four_projects_l252_25213


namespace amusement_park_line_count_l252_25273

theorem amusement_park_line_count : 
  ∀ (eunji_position : ℕ) (people_behind : ℕ),
    eunji_position = 6 →
    people_behind = 7 →
    eunji_position + people_behind = 13 := by
  sorry

end amusement_park_line_count_l252_25273


namespace necessary_but_not_sufficient_l252_25296

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

theorem necessary_but_not_sufficient (a b : E) (ha : a ≠ 0) (hb : b ≠ 0) :
  (∀ a b : E, a - 2 • b = 0 → ‖a - b‖ = ‖b‖) ∧
  (∃ a b : E, ‖a - b‖ = ‖b‖ ∧ a - 2 • b ≠ 0) := by
  sorry

end necessary_but_not_sufficient_l252_25296


namespace factorization_x_squared_minus_2x_l252_25277

theorem factorization_x_squared_minus_2x (x : ℝ) : x^2 - 2*x = x*(x - 2) := by
  sorry

end factorization_x_squared_minus_2x_l252_25277


namespace toby_journey_distance_l252_25288

def unloaded_speed : ℝ := 20
def loaded_speed : ℝ := 10
def first_loaded_distance : ℝ := 180
def third_loaded_distance : ℝ := 80
def fourth_unloaded_distance : ℝ := 140
def total_time : ℝ := 39

def second_unloaded_distance : ℝ := 120

theorem toby_journey_distance :
  let first_time := first_loaded_distance / loaded_speed
  let third_time := third_loaded_distance / loaded_speed
  let fourth_time := fourth_unloaded_distance / unloaded_speed
  let second_time := second_unloaded_distance / unloaded_speed
  first_time + second_time + third_time + fourth_time = total_time :=
by sorry

end toby_journey_distance_l252_25288


namespace largest_number_with_digit_sum_20_l252_25253

def is_valid_number (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 3 ∨ d = 4

def digit_sum (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem largest_number_with_digit_sum_20 :
  ∀ n : ℕ, is_valid_number n → digit_sum n = 20 → n ≤ 443333 :=
sorry

end largest_number_with_digit_sum_20_l252_25253


namespace cosine_from_tangent_third_quadrant_l252_25282

theorem cosine_from_tangent_third_quadrant (α : Real) :
  α ∈ Set.Ioo (π) (3*π/2) →  -- α is in the third quadrant
  Real.tan α = 1/2 →         -- tan(α) = 1/2
  Real.cos α = -2*Real.sqrt 5/5 := by
sorry

end cosine_from_tangent_third_quadrant_l252_25282


namespace two_red_two_blue_probability_l252_25209

/-- The probability of selecting two red and two blue marbles from a bag -/
theorem two_red_two_blue_probability (total_marbles : ℕ) (red_marbles : ℕ) (blue_marbles : ℕ)
  (h1 : total_marbles = red_marbles + blue_marbles)
  (h2 : red_marbles = 12)
  (h3 : blue_marbles = 8) :
  (Nat.choose red_marbles 2 * Nat.choose blue_marbles 2) / Nat.choose total_marbles 4 = 3696 / 9690 := by
  sorry

end two_red_two_blue_probability_l252_25209


namespace kitchen_upgrade_cost_l252_25299

/-- The cost of a kitchen upgrade with cabinet knobs and drawer pulls -/
theorem kitchen_upgrade_cost (num_knobs : ℕ) (num_pulls : ℕ) (pull_cost : ℚ) (total_cost : ℚ) 
  (h1 : num_knobs = 18)
  (h2 : num_pulls = 8)
  (h3 : pull_cost = 4)
  (h4 : total_cost = 77) :
  (total_cost - num_pulls * pull_cost) / num_knobs = 2.5 := by
  sorry

end kitchen_upgrade_cost_l252_25299


namespace snow_volume_to_clear_l252_25214

/-- Calculates the volume of snow to be cleared from a driveway -/
theorem snow_volume_to_clear (length width : Real) (depth : Real) (melt_percentage : Real) : 
  length = 30 ∧ width = 3 ∧ depth = 0.5 ∧ melt_percentage = 0.1 → 
  (1 - melt_percentage) * (length * width * depth) / 27 = 1.5 := by
  sorry

#check snow_volume_to_clear

end snow_volume_to_clear_l252_25214


namespace line_equation_slope_intercept_l252_25249

/-- Given a line equation, prove its slope and y-intercept -/
theorem line_equation_slope_intercept :
  ∀ (x y : ℝ), 
  2 * (x - 3) + (-1) * (y - (-4)) = 0 →
  ∃ (m b : ℝ), y = m * x + b ∧ m = 2 ∧ b = -10 := by
  sorry

end line_equation_slope_intercept_l252_25249


namespace f_not_prime_l252_25280

def f (n : ℕ+) : ℤ := n.val^6 - 550 * n.val^3 + 324

theorem f_not_prime : ∀ n : ℕ+, ¬ Nat.Prime (Int.natAbs (f n)) := by
  sorry

end f_not_prime_l252_25280


namespace total_fruits_picked_l252_25212

def mike_pears : ℕ := 8
def jason_pears : ℕ := 7
def fred_apples : ℕ := 6
def sarah_apples : ℕ := 12

theorem total_fruits_picked : mike_pears + jason_pears + fred_apples + sarah_apples = 33 := by
  sorry

end total_fruits_picked_l252_25212


namespace problem_solution_l252_25281

-- Define the function f
def f (x : ℝ) : ℝ := |x| - |x - 1|

-- Theorem statement
theorem problem_solution :
  (∃ (m : ℝ), ∀ (x : ℝ), f x ≥ |m - 1| → m ≤ 2) ∧
  (∀ (a b : ℝ), a > 0 → b > 0 → a^2 + b^2 = 2 → a + b ≥ 2*a*b) :=
by sorry

end problem_solution_l252_25281


namespace circle_equation_l252_25200

-- Define the circle C
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the line l: 3x - y - 3 = 0
def Line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 3 * p.1 - p.2 - 3 = 0}

theorem circle_equation :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (center ∈ Line) ∧
    ((2, 5) ∈ Circle center radius) ∧
    ((4, 3) ∈ Circle center radius) ∧
    (Circle center radius = {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 - 3)^2 = 4}) := by
  sorry

end circle_equation_l252_25200


namespace janet_additional_money_needed_l252_25233

def janet_savings : ℕ := 2225
def monthly_rent : ℕ := 1250
def months_advance : ℕ := 2
def deposit : ℕ := 500
def utility_deposit : ℕ := 300
def moving_costs : ℕ := 150

theorem janet_additional_money_needed :
  janet_savings + (monthly_rent * months_advance + deposit + utility_deposit + moving_costs - janet_savings) = 3450 :=
by sorry

end janet_additional_money_needed_l252_25233


namespace als_original_portion_l252_25238

theorem als_original_portion (a b c : ℝ) : 
  a + b + c = 1200 →
  a - 150 + 3*b + 3*c = 1800 →
  a = 825 := by
sorry

end als_original_portion_l252_25238


namespace gcd_765432_654321_l252_25218

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 111111 := by
  sorry

end gcd_765432_654321_l252_25218


namespace fourth_square_area_l252_25204

-- Define the triangles and their properties
def triangle_ABC (AB BC AC : ℝ) : Prop :=
  AB^2 + BC^2 = AC^2 ∧ AB^2 = 25 ∧ BC^2 = 49

def triangle_ACD (AC CD AD : ℝ) : Prop :=
  AC^2 + CD^2 = AD^2 ∧ CD^2 = 64

-- Theorem statement
theorem fourth_square_area 
  (AB BC AC CD AD : ℝ) 
  (h1 : triangle_ABC AB BC AC) 
  (h2 : triangle_ACD AC CD AD) :
  AD^2 = 138 := by sorry

end fourth_square_area_l252_25204


namespace tan_40_plus_6_sin_40_l252_25283

theorem tan_40_plus_6_sin_40 : 
  Real.tan (40 * π / 180) + 6 * Real.sin (40 * π / 180) = 
    Real.sqrt 3 + Real.cos (10 * π / 180) / Real.cos (40 * π / 180) := by sorry

end tan_40_plus_6_sin_40_l252_25283


namespace problem_statement_l252_25241

open Real

noncomputable def f (x : ℝ) : ℝ := log x - (1/4) * x + (3/(4*x)) - 1

def g (b : ℝ) (x : ℝ) : ℝ := x^2 - 2*b*x + 4

theorem problem_statement (b : ℝ) :
  (∀ x₁ ∈ Set.Ioo 0 2, ∃ x₂ ∈ Set.Icc 1 2, f x₁ ≥ g b x₂) ↔ b ≥ 17/8 :=
sorry

end problem_statement_l252_25241


namespace square_of_cube_third_smallest_prime_l252_25232

-- Define the third smallest prime number
def third_smallest_prime : ℕ := 5

-- State the theorem
theorem square_of_cube_third_smallest_prime :
  (third_smallest_prime ^ 3) ^ 2 = 15625 := by sorry

end square_of_cube_third_smallest_prime_l252_25232


namespace starters_count_l252_25220

-- Define the total number of players
def total_players : ℕ := 15

-- Define the number of quadruplets
def num_quadruplets : ℕ := 4

-- Define the number of starters to choose
def num_starters : ℕ := 6

-- Define the number of quadruplets that must be in the starting lineup
def quadruplets_in_lineup : ℕ := 3

-- Define the function to calculate the number of ways to choose the starting lineup
def choose_starters : ℕ :=
  (Nat.choose num_quadruplets quadruplets_in_lineup) *
  (Nat.choose (total_players - num_quadruplets) (num_starters - quadruplets_in_lineup))

-- Theorem statement
theorem starters_count : choose_starters = 660 := by
  sorry

end starters_count_l252_25220


namespace a_squared_plus_a_negative_l252_25264

theorem a_squared_plus_a_negative (a : ℝ) (h : a^2 + a < 0) : -a > a^2 ∧ a^2 > -a^2 ∧ -a^2 > a := by
  sorry

end a_squared_plus_a_negative_l252_25264


namespace max_A_value_l252_25293

-- Define the structure of our number configuration
structure NumberConfig where
  A : ℕ
  B : ℕ
  C : ℕ
  D : ℕ
  E : ℕ
  F : ℕ
  G : ℕ
  H : ℕ
  I : ℕ
  J : ℕ

-- Define the properties of the number configuration
def validConfig (n : NumberConfig) : Prop :=
  n.A > n.B ∧ n.B > n.C ∧
  n.D > n.E ∧ n.E > n.F ∧
  n.G > n.H ∧ n.H > n.I ∧ n.I > n.J ∧
  ∃ k : ℕ, n.D = 3 * k + 3 ∧ n.E = 3 * k ∧ n.F = 3 * k - 3 ∧
  ∃ m : ℕ, n.G = 2 * m + 1 ∧ n.H = 2 * m - 1 ∧ n.I = 2 * m - 3 ∧ n.J = 2 * m - 5 ∧
  n.A + n.B + n.C = 9

-- Theorem statement
theorem max_A_value (n : NumberConfig) (h : validConfig n) :
  n.A ≤ 8 ∧ ∃ m : NumberConfig, validConfig m ∧ m.A = 8 :=
sorry

end max_A_value_l252_25293


namespace arithmetic_computation_l252_25257

theorem arithmetic_computation : 3 + 8 * 3 - 4 + 2^3 * 5 / 2 = 43 := by
  sorry

end arithmetic_computation_l252_25257


namespace probability_of_five_ones_l252_25256

def num_dice : ℕ := 15
def num_ones : ℕ := 5
def sides_on_die : ℕ := 6

theorem probability_of_five_ones :
  (Nat.choose num_dice num_ones : ℚ) * (1 / sides_on_die : ℚ)^num_ones * (1 - 1 / sides_on_die : ℚ)^(num_dice - num_ones) =
  (Nat.choose 15 5 : ℚ) * (1 / 6 : ℚ)^5 * (5 / 6 : ℚ)^10 := by
  sorry

end probability_of_five_ones_l252_25256


namespace negation_of_existence_logarithm_equality_negation_l252_25223

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x : ℝ, x > 0 ∧ P x) ↔ (∀ x : ℝ, x > 0 → ¬ P x) :=
by sorry

theorem logarithm_equality_negation :
  (¬ ∃ x : ℝ, x > 0 ∧ Real.log x = x - 1) ↔
  (∀ x : ℝ, x > 0 → Real.log x ≠ x - 1) :=
by sorry

end negation_of_existence_logarithm_equality_negation_l252_25223


namespace painted_fraction_is_three_eighths_l252_25269

/-- Represents a square plate with sides of length 4 meters -/
structure Plate :=
  (side_length : ℝ)
  (area : ℝ)
  (h_side : side_length = 4)
  (h_area : area = side_length * side_length)

/-- Represents the number of equal parts the plate is divided into -/
def total_parts : ℕ := 16

/-- Represents the number of painted parts -/
def painted_parts : ℕ := 6

/-- The theorem to be proved -/
theorem painted_fraction_is_three_eighths (plate : Plate) :
  (painted_parts : ℝ) / total_parts = 3 / 8 := by
  sorry


end painted_fraction_is_three_eighths_l252_25269


namespace second_sea_fields_medalist_from_vietnam_l252_25202

/-- Represents a mathematician -/
structure Mathematician where
  name : String
  country : String

/-- Represents the Fields Medal award -/
inductive FieldsMedal
  | recipient : Mathematician → FieldsMedal

/-- Predicate to check if a country is in South East Asia -/
def is_south_east_asian (country : String) : Prop := sorry

/-- Predicate to check if a mathematician is a Fields Medal recipient -/
def is_fields_medalist (m : Mathematician) : Prop := sorry

/-- The second South East Asian Fields Medal recipient -/
def second_sea_fields_medalist : Mathematician := sorry

/-- Theorem stating that the second South East Asian Fields Medal recipient is from Vietnam -/
theorem second_sea_fields_medalist_from_vietnam :
  second_sea_fields_medalist.country = "Vietnam" := by sorry

end second_sea_fields_medalist_from_vietnam_l252_25202


namespace normal_distribution_std_dev_l252_25226

/-- For a normal distribution with mean 14.5, if the value that is exactly 2 standard deviations
    less than the mean is 11.5, then the standard deviation is 1.5. -/
theorem normal_distribution_std_dev (μ σ : ℝ) :
  μ = 14.5 →
  μ - 2 * σ = 11.5 →
  σ = 1.5 := by
sorry

end normal_distribution_std_dev_l252_25226


namespace units_digit_of_quotient_l252_25267

theorem units_digit_of_quotient (h : 7 ∣ (4^1985 + 7^1985)) :
  (4^1985 + 7^1985) / 7 % 10 = 2 := by
sorry

end units_digit_of_quotient_l252_25267


namespace bank_checks_total_amount_l252_25262

theorem bank_checks_total_amount : 
  let million_won_checks : ℕ := 25
  let hundred_thousand_won_checks : ℕ := 8
  let million_won_value : ℕ := 1000000
  let hundred_thousand_won_value : ℕ := 100000
  (million_won_checks * million_won_value + hundred_thousand_won_checks * hundred_thousand_won_value : ℕ) = 25800000 :=
by
  sorry

end bank_checks_total_amount_l252_25262


namespace watermelon_seed_requirement_l252_25227

/-- Represents the minimum number of watermelon seeds required -/
def min_seeds : ℕ := 10041

/-- Represents the number of watermelons to be sold each year -/
def watermelons_to_sell : ℕ := 10000

/-- Represents the number of seeds produced by each watermelon -/
def seeds_per_watermelon : ℕ := 250

theorem watermelon_seed_requirement (S : ℕ) :
  S ≥ min_seeds →
  ∃ (x : ℕ), S = watermelons_to_sell + x ∧
             seeds_per_watermelon * x ≥ S ∧
             ∀ (S' : ℕ), S' < S →
               ¬∃ (x' : ℕ), S' = watermelons_to_sell + x' ∧
                             seeds_per_watermelon * x' ≥ S' :=
by sorry

#check watermelon_seed_requirement

end watermelon_seed_requirement_l252_25227


namespace student_weight_loss_l252_25270

/-- The weight the student needs to lose to weigh twice as much as his sister -/
def weight_to_lose (total_weight student_weight : ℝ) : ℝ :=
  let sister_weight := total_weight - student_weight
  student_weight - 2 * sister_weight

theorem student_weight_loss (total_weight student_weight : ℝ) 
  (h1 : total_weight = 110)
  (h2 : student_weight = 75) :
  weight_to_lose total_weight student_weight = 5 := by
  sorry

end student_weight_loss_l252_25270


namespace M_subset_N_l252_25276

def M : Set ℝ := { x | ∃ k : ℤ, x = (k * Real.pi / 2) + (Real.pi / 4) }

def N : Set ℝ := { x | ∃ k : ℤ, x = (k * Real.pi / 4) + (Real.pi / 2) }

theorem M_subset_N : M ⊆ N := by
  sorry

end M_subset_N_l252_25276


namespace max_roses_is_316_l252_25279

/-- Represents the pricing options and budget for purchasing roses -/
structure RosePurchase where
  single_price : ℚ  -- Price of a single rose
  dozen_price : ℚ   -- Price of a dozen roses
  two_dozen_price : ℚ -- Price of two dozen roses
  budget : ℚ        -- Total budget

/-- Calculates the maximum number of roses that can be purchased given the pricing options and budget -/
def max_roses (rp : RosePurchase) : ℕ :=
  sorry

/-- Theorem stating that the maximum number of roses that can be purchased is 316 -/
theorem max_roses_is_316 (rp : RosePurchase) 
  (h1 : rp.single_price = 63/10)
  (h2 : rp.dozen_price = 36)
  (h3 : rp.two_dozen_price = 50)
  (h4 : rp.budget = 680) :
  max_roses rp = 316 :=
by sorry

end max_roses_is_316_l252_25279


namespace grandson_age_l252_25222

theorem grandson_age (grandmother_age grandson_age : ℕ) : 
  grandmother_age = grandson_age * 12 →
  grandmother_age + grandson_age = 65 →
  grandson_age = 5 := by
sorry

end grandson_age_l252_25222


namespace least_exponent_for_divisibility_l252_25208

/-- The function that calculates the sum of powers for the given exponent -/
def sumOfPowers (a : ℕ+) : ℕ :=
  (1995 : ℕ) ^ a.val + (1996 : ℕ) ^ a.val + (1997 : ℕ) ^ a.val

/-- The property that the sum is divisible by 10 -/
def isDivisibleBy10 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 10 * k

/-- The main theorem statement -/
theorem least_exponent_for_divisibility :
  (∀ a : ℕ+, a < 2 → ¬(isDivisibleBy10 (sumOfPowers a))) ∧
  isDivisibleBy10 (sumOfPowers 2) := by
  sorry

#check least_exponent_for_divisibility

end least_exponent_for_divisibility_l252_25208


namespace tan_double_angle_l252_25287

theorem tan_double_angle (θ : Real) (h : Real.tan θ = 2) : Real.tan (2 * θ) = -4/3 := by
  sorry

end tan_double_angle_l252_25287


namespace circle_radius_is_one_l252_25243

/-- The equation of a circle is x^2 + y^2 + 2x + 2y + 1 = 0. This theorem proves that the radius of this circle is 1. -/
theorem circle_radius_is_one :
  ∃ (h : ℝ → ℝ → Prop),
    (∀ x y : ℝ, h x y ↔ x^2 + y^2 + 2*x + 2*y + 1 = 0) →
    (∃ c : ℝ × ℝ, ∃ r : ℝ, r = 1 ∧ ∀ x y : ℝ, h x y ↔ (x - c.1)^2 + (y - c.2)^2 = r^2) :=
by sorry

end circle_radius_is_one_l252_25243


namespace ellipse_properties_l252_25271

/-- Definition of the ellipse C -/
def ellipse_C (x y : ℝ) (a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- Definition of point M -/
def point_M : ℝ × ℝ := (0, 2)

/-- Theorem stating the properties of the ellipse and its related points -/
theorem ellipse_properties :
  ∀ (a b : ℝ), a > b ∧ b > 0 →
  (∀ x y, ellipse_C x y a b ↔ x^2 / 8 + y^2 / 4 = 1) ∧
  (∀ x y, (∃ x₁ y₁, ellipse_C x₁ y₁ a b ∧ x = (x₁ + point_M.1) / 2 ∧ y = (y₁ + point_M.2) / 2) ↔
    x^2 / 2 + (y - 1)^2 = 1) ∧
  (∀ k₁ k₂ x₁ y₁ x₂ y₂,
    ellipse_C x₁ y₁ a b ∧ ellipse_C x₂ y₂ a b ∧
    k₁ = (y₁ - point_M.2) / (x₁ - point_M.1) ∧
    k₂ = (y₂ - point_M.2) / (x₂ - point_M.1) ∧
    k₁ + k₂ = 8 →
    ∃ t, (1 - t) * x₁ + t * x₂ = -1/2 ∧ (1 - t) * y₁ + t * y₂ = -2) :=
by sorry

end ellipse_properties_l252_25271


namespace positive_difference_of_numbers_l252_25225

theorem positive_difference_of_numbers (a b : ℝ) 
  (sum_eq : a + b = 10) 
  (square_diff_eq : a^2 - b^2 = 40) : 
  |a - b| = 4 := by
sorry

end positive_difference_of_numbers_l252_25225


namespace wire_cutting_l252_25224

theorem wire_cutting (total_length : ℝ) (shorter_length : ℝ) : 
  total_length = 50 →
  shorter_length + (5/2 * shorter_length) = total_length →
  shorter_length = 100/7 :=
by sorry

end wire_cutting_l252_25224


namespace addition_is_unique_solution_l252_25229

-- Define the possible operations
inductive Operation
| Add
| Subtract
| Multiply
| Divide

-- Define a function to apply the operation
def applyOperation (op : Operation) (a b : Int) : Int :=
  match op with
  | Operation.Add => a + b
  | Operation.Subtract => a - b
  | Operation.Multiply => a * b
  | Operation.Divide => a / b

-- Theorem statement
theorem addition_is_unique_solution :
  ∃! op : Operation, applyOperation op 5 (-5) = 0 :=
sorry

end addition_is_unique_solution_l252_25229


namespace pinball_spending_l252_25210

def half_dollar : ℚ := 0.5

def wednesday_spent : ℕ := 4
def next_day_spent : ℕ := 14

def total_spent : ℚ := (wednesday_spent * half_dollar) + (next_day_spent * half_dollar)

theorem pinball_spending : total_spent = 9 := by sorry

end pinball_spending_l252_25210


namespace youngest_brother_age_l252_25228

/-- Represents the ages of Rick and his brothers -/
structure FamilyAges where
  rick : ℕ
  oldest : ℕ
  middle : ℕ
  smallest : ℕ
  youngest : ℕ

/-- Defines the relationships between the ages in the family -/
def validFamilyAges (ages : FamilyAges) : Prop :=
  ages.rick = 15 ∧
  ages.oldest = 2 * ages.rick ∧
  ages.middle = ages.oldest / 3 ∧
  ages.smallest = ages.middle / 2 ∧
  ages.youngest = ages.smallest - 2

/-- Theorem stating that given the family age relationships, the youngest brother is 3 years old -/
theorem youngest_brother_age (ages : FamilyAges) (h : validFamilyAges ages) : ages.youngest = 3 := by
  sorry

end youngest_brother_age_l252_25228


namespace committee_seating_arrangements_l252_25248

/-- The number of distinct arrangements of chairs and benches -/
def distinct_arrangements (total_positions : ℕ) (bench_count : ℕ) : ℕ :=
  Nat.choose total_positions bench_count

theorem committee_seating_arrangements :
  distinct_arrangements 14 4 = 1001 := by
  sorry

end committee_seating_arrangements_l252_25248


namespace lions_mortality_rate_l252_25206

/-- The number of lions that die each month in Londolozi -/
def lions_die_per_month : ℕ := 1

/-- The initial number of lions in Londolozi -/
def initial_lions : ℕ := 100

/-- The number of lion cubs born per month in Londolozi -/
def cubs_born_per_month : ℕ := 5

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- The number of lions after one year in Londolozi -/
def lions_after_year : ℕ := 148

theorem lions_mortality_rate :
  lions_after_year = initial_lions + (cubs_born_per_month - lions_die_per_month) * months_in_year :=
sorry

end lions_mortality_rate_l252_25206


namespace alex_money_left_l252_25250

def main_job_income : ℝ := 900
def side_job_income : ℝ := 300
def main_job_tax_rate : ℝ := 0.15
def side_job_tax_rate : ℝ := 0.20
def water_bill : ℝ := 75
def main_job_tithe_rate : ℝ := 0.10
def side_job_tithe_rate : ℝ := 0.15
def groceries : ℝ := 150
def transportation : ℝ := 50

theorem alex_money_left :
  let main_job_after_tax := main_job_income * (1 - main_job_tax_rate)
  let side_job_after_tax := side_job_income * (1 - side_job_tax_rate)
  let total_income_after_tax := main_job_after_tax + side_job_after_tax
  let total_tithe := main_job_income * main_job_tithe_rate + side_job_income * side_job_tithe_rate
  let total_deductions := water_bill + groceries + transportation + total_tithe
  total_income_after_tax - total_deductions = 595 := by
  sorry

end alex_money_left_l252_25250


namespace sqrt_two_minus_one_power_l252_25294

theorem sqrt_two_minus_one_power (n : ℕ+) :
  ∃ m : ℕ+, (Real.sqrt 2 - 1) ^ n.val = Real.sqrt m.val - Real.sqrt (m.val - 1) := by
  sorry

end sqrt_two_minus_one_power_l252_25294


namespace sector_angle_l252_25215

/-- Given a circular sector with radius 2 and area 8, prove that its central angle is 4 radians -/
theorem sector_angle (r : ℝ) (area : ℝ) (θ : ℝ) 
  (h_radius : r = 2)
  (h_area : area = 8)
  (h_sector_area : area = 1/2 * r^2 * θ) : θ = 4 := by
  sorry

end sector_angle_l252_25215


namespace abs_lt_one_sufficient_not_necessary_for_lt_one_l252_25205

theorem abs_lt_one_sufficient_not_necessary_for_lt_one :
  (∃ x : ℝ, (|x| < 1 → x < 1) ∧ ¬(x < 1 → |x| < 1)) := by
  sorry

end abs_lt_one_sufficient_not_necessary_for_lt_one_l252_25205


namespace power_difference_equality_l252_25244

theorem power_difference_equality (a b : ℕ) (h1 : a = 3) (h2 : b = 2) :
  (a^b)^a - (b^a)^b = 665 := by
  sorry

end power_difference_equality_l252_25244


namespace bonus_threshold_correct_l252_25235

/-- The sales amount that triggers the bonus commission --/
def bonus_threshold : ℝ := 10000

/-- The total sales amount --/
def total_sales : ℝ := 14000

/-- The commission rate on total sales --/
def commission_rate : ℝ := 0.09

/-- The bonus commission rate on excess sales --/
def bonus_rate : ℝ := 0.03

/-- The total commission received --/
def total_commission : ℝ := 1380

/-- The bonus commission received --/
def bonus_commission : ℝ := 120

theorem bonus_threshold_correct :
  commission_rate * total_sales = total_commission - bonus_commission ∧
  bonus_rate * (total_sales - bonus_threshold) = bonus_commission :=
by sorry

end bonus_threshold_correct_l252_25235


namespace sequence_difference_l252_25260

theorem sequence_difference (a : ℕ → ℕ) : 
  (∀ n m : ℕ, n < m → a n < a m) →  -- strictly increasing
  (∀ n : ℕ, n ≥ 1 → a n ≥ 1) →     -- a_n ≥ 1 for n ≥ 1
  (∀ n : ℕ, n ≥ 1 → a (a n) = 3 * n) →  -- a_{a_n} = 3n for n ≥ 1
  a 2021 - a 1999 = 66 := by
sorry

end sequence_difference_l252_25260


namespace rowing_distance_l252_25247

theorem rowing_distance (v_man : ℝ) (v_river : ℝ) (total_time : ℝ) :
  v_man = 8 →
  v_river = 2 →
  total_time = 1 →
  ∃ (distance : ℝ),
    distance / (v_man - v_river) + distance / (v_man + v_river) = total_time ∧
    2 * distance = 7.5 :=
by sorry

end rowing_distance_l252_25247


namespace intersection_complement_empty_l252_25263

/-- The set of all non-zero real numbers -/
def P : Set ℝ := {x : ℝ | x ≠ 0}

/-- The set of all positive real numbers -/
def Q : Set ℝ := {x : ℝ | x > 0}

/-- Theorem stating that the intersection of Q and the complement of P in ℝ is empty -/
theorem intersection_complement_empty : Q ∩ (Set.univ \ P) = ∅ := by sorry

end intersection_complement_empty_l252_25263


namespace age_sum_proof_l252_25236

theorem age_sum_proof (uncle_age : ℕ) (yuna_eunji_diff : ℕ) (uncle_eunji_diff : ℕ) 
  (h1 : uncle_age = 41)
  (h2 : yuna_eunji_diff = 3)
  (h3 : uncle_eunji_diff = 25) :
  uncle_age - uncle_eunji_diff + (uncle_age - uncle_eunji_diff + yuna_eunji_diff) = 35 := by
  sorry

end age_sum_proof_l252_25236


namespace shaded_cubes_count_shaded_cubes_count_proof_l252_25274

/-- Represents a 3x3x3 cube with a specific shading pattern -/
structure ShadedCube where
  /-- The number of smaller cubes in each dimension of the large cube -/
  size : Nat
  /-- The number of shaded squares on each face -/
  shaded_per_face : Nat
  /-- The total number of smaller cubes in the large cube -/
  total_cubes : Nat
  /-- Assertion that the cube is 3x3x3 -/
  size_is_three : size = 3
  /-- Assertion that the total number of cubes is correct -/
  total_is_correct : total_cubes = size ^ 3
  /-- Assertion that each face has 5 shaded squares -/
  five_shaded_per_face : shaded_per_face = 5

/-- Theorem stating that the number of shaded cubes is 20 -/
theorem shaded_cubes_count (c : ShadedCube) : Nat :=
  20

#check shaded_cubes_count

/-- Proof of the theorem -/
theorem shaded_cubes_count_proof (c : ShadedCube) : shaded_cubes_count c = 20 := by
  sorry

end shaded_cubes_count_shaded_cubes_count_proof_l252_25274


namespace smallest_base_perfect_cube_l252_25298

/-- Given a base b > 5, returns the value of 12_b in base 10 -/
def baseB_to_base10 (b : ℕ) : ℕ := b + 2

/-- Checks if a number is a perfect cube -/
def is_perfect_cube (n : ℕ) : Prop := ∃ m : ℕ, m^3 = n

theorem smallest_base_perfect_cube :
  (∀ b : ℕ, b > 5 ∧ b < 6 → ¬ is_perfect_cube (baseB_to_base10 b)) ∧
  is_perfect_cube (baseB_to_base10 6) := by
  sorry

end smallest_base_perfect_cube_l252_25298


namespace left_handed_fraction_is_four_ninths_l252_25207

/-- Represents the ratio of red to blue participants -/
def red_to_blue_ratio : ℚ := 2

/-- Fraction of left-handed red participants -/
def left_handed_red_fraction : ℚ := 1/3

/-- Fraction of left-handed blue participants -/
def left_handed_blue_fraction : ℚ := 2/3

/-- Theorem stating the fraction of left-handed participants -/
theorem left_handed_fraction_is_four_ninths 
  (h1 : red_to_blue_ratio = 2)
  (h2 : left_handed_red_fraction = 1/3)
  (h3 : left_handed_blue_fraction = 2/3) :
  (red_to_blue_ratio * left_handed_red_fraction + left_handed_blue_fraction) / 
  (red_to_blue_ratio + 1) = 4/9 := by
  sorry

#check left_handed_fraction_is_four_ninths

end left_handed_fraction_is_four_ninths_l252_25207


namespace total_length_l252_25286

def problem (rubber pen pencil marker ruler : ℝ) : Prop :=
  pen = rubber + 3 ∧
  pencil = pen + 2 ∧
  pencil = 12 ∧
  ruler = 3 * rubber ∧
  marker = (pen + rubber + pencil) / 3 ∧
  marker = ruler / 2

theorem total_length (rubber pen pencil marker ruler : ℝ) 
  (h : problem rubber pen pencil marker ruler) : 
  rubber + pen + pencil + marker + ruler = 60.5 := by
  sorry

end total_length_l252_25286


namespace pizza_order_l252_25239

theorem pizza_order (slices_per_pizza : ℕ) (total_slices : ℕ) (h1 : slices_per_pizza = 2) (h2 : total_slices = 28) :
  total_slices / slices_per_pizza = 14 := by
  sorry

end pizza_order_l252_25239


namespace quarter_to_fourth_power_decimal_l252_25289

theorem quarter_to_fourth_power_decimal : (1 / 4 : ℚ) ^ 4 = 0.00390625 := by
  sorry

end quarter_to_fourth_power_decimal_l252_25289


namespace max_sum_nonnegative_reals_l252_25231

theorem max_sum_nonnegative_reals (a b c : ℝ) :
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 →
  a^2 + b^2 + c^2 = 52 →
  a*b + b*c + c*a = 28 →
  a + b + c ≤ 6 * Real.sqrt 3 :=
by sorry

end max_sum_nonnegative_reals_l252_25231


namespace problem_statement_l252_25251

theorem problem_statement (x : ℝ) (h : x + 1/x = 3) : 
  (x - 3)^2 + 16/((x - 3)^2) = 7 := by
  sorry

end problem_statement_l252_25251
