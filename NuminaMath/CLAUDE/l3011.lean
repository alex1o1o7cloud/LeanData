import Mathlib

namespace NUMINAMATH_CALUDE_eliminate_denominators_l3011_301101

theorem eliminate_denominators (x : ℝ) : 
  ((x + 1) / 2 + 1 = x / 3) ↔ (3 * (x + 1) + 6 = 2 * x) := by sorry

end NUMINAMATH_CALUDE_eliminate_denominators_l3011_301101


namespace NUMINAMATH_CALUDE_church_seating_problem_l3011_301158

/-- 
Proves that the number of chairs in each row is 6, given the conditions of the church seating problem.
-/
theorem church_seating_problem (rows : ℕ) (people_per_chair : ℕ) (total_capacity : ℕ) 
  (h1 : rows = 20)
  (h2 : people_per_chair = 5)
  (h3 : total_capacity = 600) :
  (total_capacity / (rows * people_per_chair) : ℚ) = 6 := by
  sorry

end NUMINAMATH_CALUDE_church_seating_problem_l3011_301158


namespace NUMINAMATH_CALUDE_arc_length_radius_l3011_301108

/-- Given an arc length and central angle, calculate the radius of the circle -/
theorem arc_length_radius (arc_length : ℝ) (central_angle : ℝ) : 
  arc_length = 4 → central_angle = 2 → arc_length = central_angle * 2 := by sorry

end NUMINAMATH_CALUDE_arc_length_radius_l3011_301108


namespace NUMINAMATH_CALUDE_abs_decreasing_neg_l3011_301127

-- Define the function f(x) = |x|
def f (x : ℝ) : ℝ := abs x

-- State the theorem
theorem abs_decreasing_neg : ∀ x y : ℝ, x < y → y < 0 → f y < f x := by
  sorry

end NUMINAMATH_CALUDE_abs_decreasing_neg_l3011_301127


namespace NUMINAMATH_CALUDE_na2so4_formation_l3011_301117

-- Define the chemical species
structure Chemical where
  name : String
  moles : ℚ

-- Define the reaction conditions
structure ReactionConditions where
  temperature : ℚ
  pressure : ℚ

-- Define the reaction
def reaction (h2so4 : Chemical) (naoh : Chemical) (hcl : Chemical) (koh : Chemical) (conditions : ReactionConditions) : Chemical :=
  { name := "Na2SO4", moles := 1 }

-- Theorem statement
theorem na2so4_formation
  (h2so4 : Chemical)
  (naoh : Chemical)
  (hcl : Chemical)
  (koh : Chemical)
  (conditions : ReactionConditions)
  (h_h2so4_moles : h2so4.moles = 1)
  (h_naoh_moles : naoh.moles = 2)
  (h_hcl_moles : hcl.moles = 1/2)
  (h_koh_moles : koh.moles = 1/2)
  (h_temperature : conditions.temperature = 25)
  (h_pressure : conditions.pressure = 1) :
  (reaction h2so4 naoh hcl koh conditions).moles = 1 := by
  sorry

end NUMINAMATH_CALUDE_na2so4_formation_l3011_301117


namespace NUMINAMATH_CALUDE_arrange_five_and_three_books_l3011_301122

/-- The number of ways to arrange two types of indistinguishable objects in a row -/
def arrange_books (n m : ℕ) : ℕ := Nat.choose (n + m) n

/-- Theorem: Arranging 5 copies of one book and 3 copies of another book results in 56 ways -/
theorem arrange_five_and_three_books : arrange_books 5 3 = 56 := by
  sorry

end NUMINAMATH_CALUDE_arrange_five_and_three_books_l3011_301122


namespace NUMINAMATH_CALUDE_reflect_A_across_y_axis_l3011_301161

/-- Reflects a point across the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

/-- The original point A -/
def A : ℝ × ℝ := (2, -1)

theorem reflect_A_across_y_axis :
  reflect_y A = (-2, -1) := by sorry

end NUMINAMATH_CALUDE_reflect_A_across_y_axis_l3011_301161


namespace NUMINAMATH_CALUDE_square_sum_given_product_and_sum_square_l3011_301147

theorem square_sum_given_product_and_sum_square (x y : ℝ) 
  (h1 : (x + y)^2 = 4) (h2 : x * y = -1) : x^2 + y^2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_given_product_and_sum_square_l3011_301147


namespace NUMINAMATH_CALUDE_andy_wrong_answers_l3011_301170

/-- Represents the number of wrong answers for each person in a 30-question test. -/
structure TestResults where
  andy : ℕ
  beth : ℕ
  charlie : ℕ
  daniel : ℕ

/-- The conditions of the problem and the theorem to be proved. -/
theorem andy_wrong_answers (t : TestResults) : 
  t.andy + t.beth = t.charlie + t.daniel →
  t.andy + t.daniel = t.beth + t.charlie + 4 →
  t.charlie = 5 →
  t.andy = 7 := by
  sorry

end NUMINAMATH_CALUDE_andy_wrong_answers_l3011_301170


namespace NUMINAMATH_CALUDE_maria_fish_removal_l3011_301179

/-- The number of fish Maria took out of her tank -/
def fish_taken_out (initial_fish current_fish : ℕ) : ℕ :=
  initial_fish - current_fish

/-- Theorem: Maria took out 16 fish from her tank -/
theorem maria_fish_removal :
  fish_taken_out 19 3 = 16 := by
  sorry

end NUMINAMATH_CALUDE_maria_fish_removal_l3011_301179


namespace NUMINAMATH_CALUDE_symmetric_complex_numbers_l3011_301114

theorem symmetric_complex_numbers (z₁ z₂ : ℂ) :
  (z₁ = 2 - 3*I) →
  (z₁ + z₂ = 0) →
  z₂ = -2 + 3*I := by
sorry

end NUMINAMATH_CALUDE_symmetric_complex_numbers_l3011_301114


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l3011_301116

theorem fraction_to_decimal : (73 : ℚ) / 160 = (45625 : ℚ) / 100000 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l3011_301116


namespace NUMINAMATH_CALUDE_prove_equation_l3011_301184

/-- Given that (x + y) / 3 = 1.888888888888889 and 2x + y = 7, prove that x + y = 5.666666666666667 
    is the equation that, when combined with 2x + y = 7, gives the correct value for (x + y) / 3. -/
theorem prove_equation (x y : ℝ) 
  (h1 : (x + y) / 3 = 1.888888888888889)
  (h2 : 2 * x + y = 7) :
  x + y = 5.666666666666667 := by
sorry

end NUMINAMATH_CALUDE_prove_equation_l3011_301184


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l3011_301113

theorem solution_set_of_inequality (x : ℝ) :
  (2 * x - 1) / (3 * x + 1) > 0 ↔ x < -1/3 ∨ x > 1/2 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l3011_301113


namespace NUMINAMATH_CALUDE_phone_plan_charge_equality_l3011_301182

/-- Represents the per-minute charge for plan B -/
def plan_b_charge : ℝ := 0.20

/-- Represents the fixed charge for the first 9 minutes in plan A -/
def plan_a_fixed_charge : ℝ := 0.60

/-- Represents the duration (in minutes) where both plans charge the same amount -/
def equal_charge_duration : ℝ := 3

theorem phone_plan_charge_equality :
  plan_a_fixed_charge = plan_b_charge * equal_charge_duration := by
  sorry

#check phone_plan_charge_equality

end NUMINAMATH_CALUDE_phone_plan_charge_equality_l3011_301182


namespace NUMINAMATH_CALUDE_work_completion_time_l3011_301100

/-- Given that:
    - A can do the work in 3 days
    - A and B together can do the work in 2 days
    Prove that B can do the work alone in 6 days -/
theorem work_completion_time (a_time b_time ab_time : ℝ) 
    (ha : a_time = 3)
    (hab : ab_time = 2) :
    b_time = 6 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l3011_301100


namespace NUMINAMATH_CALUDE_range_of_a_l3011_301160

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x^2 ≤ a) → a ∈ Set.Ici (0 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3011_301160


namespace NUMINAMATH_CALUDE_initial_children_on_bus_l3011_301126

theorem initial_children_on_bus (children_got_on : ℕ) (total_children : ℕ) 
  (h1 : children_got_on = 38)
  (h2 : total_children = 64) :
  total_children - children_got_on = 26 := by
  sorry

end NUMINAMATH_CALUDE_initial_children_on_bus_l3011_301126


namespace NUMINAMATH_CALUDE_water_bottle_refills_l3011_301187

/-- Calculates the number of times a water bottle needs to be filled in a week -/
theorem water_bottle_refills (daily_intake : ℕ) (bottle_capacity : ℕ) : 
  daily_intake = 72 → bottle_capacity = 84 → (daily_intake * 7) / bottle_capacity = 6 := by
  sorry

#check water_bottle_refills

end NUMINAMATH_CALUDE_water_bottle_refills_l3011_301187


namespace NUMINAMATH_CALUDE_net_population_increase_in_one_day_l3011_301139

/-- Represents the number of seconds in a day -/
def seconds_per_day : ℕ := 24 * 60 * 60

/-- Represents the birth rate in people per two seconds -/
def birth_rate : ℚ := 4

/-- Represents the death rate in people per two seconds -/
def death_rate : ℚ := 3

/-- Calculates the net population increase per second -/
def net_increase_per_second : ℚ := (birth_rate - death_rate) / 2

/-- Theorem stating the net population increase in one day -/
theorem net_population_increase_in_one_day :
  ⌊net_increase_per_second * seconds_per_day⌋ = 43200 := by
  sorry

#eval ⌊net_increase_per_second * seconds_per_day⌋

end NUMINAMATH_CALUDE_net_population_increase_in_one_day_l3011_301139


namespace NUMINAMATH_CALUDE_expansion_terms_imply_n_equals_10_l3011_301148

theorem expansion_terms_imply_n_equals_10 (x a : ℝ) (n : ℕ) :
  (n.choose 1 : ℝ) * x^(n - 1) * a = 210 →
  (n.choose 2 : ℝ) * x^(n - 2) * a^2 = 840 →
  (n.choose 3 : ℝ) * x^(n - 3) * a^3 = 2520 →
  n = 10 := by
sorry

end NUMINAMATH_CALUDE_expansion_terms_imply_n_equals_10_l3011_301148


namespace NUMINAMATH_CALUDE_sum_of_roots_special_quadratic_l3011_301124

theorem sum_of_roots_special_quadratic : 
  let f : ℝ → ℝ := λ x ↦ (x - 7)^2 - 16
  ∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ + x₂ = 14 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_special_quadratic_l3011_301124


namespace NUMINAMATH_CALUDE_closest_multiple_of_18_to_3050_l3011_301104

-- Define a function to check if a number is divisible by both 2 and 9
def is_multiple_of_18 (n : ℕ) : Prop :=
  n % 2 = 0 ∧ n % 9 = 0

-- Define a function to calculate the absolute difference between two numbers
def abs_diff (a b : ℕ) : ℕ :=
  if a ≥ b then a - b else b - a

-- State the theorem
theorem closest_multiple_of_18_to_3050 :
  ∀ n : ℕ, is_multiple_of_18 n → abs_diff n 3050 ≥ abs_diff 3042 3050 :=
by sorry

end NUMINAMATH_CALUDE_closest_multiple_of_18_to_3050_l3011_301104


namespace NUMINAMATH_CALUDE_investment_problem_l3011_301142

theorem investment_problem (x y : ℝ) : 
  x + y = 24000 → 
  x * 0.045 + y * 0.06 = 24000 * 0.05 → 
  x = 16000 ∧ y = 8000 :=
by sorry

end NUMINAMATH_CALUDE_investment_problem_l3011_301142


namespace NUMINAMATH_CALUDE_inequality_proof_l3011_301135

theorem inequality_proof (a b c d : ℝ) 
  (positive_a : 0 < a) (positive_b : 0 < b) (positive_c : 0 < c) (positive_d : 0 < d)
  (sum_condition : a + b + c + d = 3) :
  1/a^2 + 1/b^2 + 1/c^2 + 1/d^2 ≤ 1/(a^2*b^2*c^2*d^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3011_301135


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l3011_301133

theorem unique_solution_for_equation : ∃! (n k : ℕ), k^5 + 5*n^4 = 81*k ∧ n = 2 ∧ k = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l3011_301133


namespace NUMINAMATH_CALUDE_percent_relation_l3011_301109

theorem percent_relation (x : ℝ) (h : 0.2 * x = 60) : 0.8 * x = 240 := by
  sorry

end NUMINAMATH_CALUDE_percent_relation_l3011_301109


namespace NUMINAMATH_CALUDE_sum_base3_equals_212002_l3011_301112

/-- Converts a base 3 number represented as a list of digits to its decimal equivalent -/
def base3ToDecimal (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => acc * 3 + d) 0

/-- Converts a decimal number to its base 3 representation as a list of digits -/
def decimalToBase3 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 3) ((m % 3) :: acc)
  aux n []

/-- The main theorem stating that the sum of the given base 3 numbers equals 212002 in base 3 -/
theorem sum_base3_equals_212002 :
  let a := base3ToDecimal [1, 0, 1, 2]
  let b := base3ToDecimal [2, 0, 2, 1]
  let c := base3ToDecimal [1, 1, 0, 2, 1]
  let d := base3ToDecimal [1, 2, 0, 1, 2]
  decimalToBase3 (a + b + c + d) = [2, 1, 2, 0, 0, 2] := by
  sorry

end NUMINAMATH_CALUDE_sum_base3_equals_212002_l3011_301112


namespace NUMINAMATH_CALUDE_blood_flow_scientific_notation_l3011_301137

/-- The amount of blood flowing through the heart of a healthy adult per minute in mL -/
def blood_flow : ℝ := 4900

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ

/-- Converts a real number to scientific notation -/
def to_scientific_notation (x : ℝ) : ScientificNotation :=
  sorry

theorem blood_flow_scientific_notation :
  to_scientific_notation blood_flow = ScientificNotation.mk 4.9 3 := by
  sorry

end NUMINAMATH_CALUDE_blood_flow_scientific_notation_l3011_301137


namespace NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l3011_301110

theorem smallest_positive_multiple_of_45 : 
  ∀ n : ℕ, n > 0 ∧ 45 ∣ n → n ≥ 45 := by
sorry

end NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l3011_301110


namespace NUMINAMATH_CALUDE_problem_1994_national_l3011_301172

theorem problem_1994_national (x y a : ℝ) 
  (h1 : x ∈ Set.Icc (-π/4) (π/4))
  (h2 : y ∈ Set.Icc (-π/4) (π/4))
  (h3 : x^3 + Real.sin x - 2*a = 0)
  (h4 : 4*y^3 + Real.sin y * Real.cos y + a = 0) :
  Real.cos (x + 2*y) = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_1994_national_l3011_301172


namespace NUMINAMATH_CALUDE_quadratic_inequality_l3011_301175

/-- A quadratic function f(x) = ax^2 + bx + c where the solution set of ax^2 + bx + c > 0 is {x | -1 < x < 3} -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

/-- The solution set condition -/
def SolutionSetCondition (a b c : ℝ) : Prop :=
  ∀ x, QuadraticFunction a b c x > 0 ↔ -1 < x ∧ x < 3

theorem quadratic_inequality (a b c : ℝ) (h : SolutionSetCondition a b c) :
  QuadraticFunction a b c 5 < QuadraticFunction a b c (-1) ∧
  QuadraticFunction a b c (-1) < QuadraticFunction a b c 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l3011_301175


namespace NUMINAMATH_CALUDE_max_points_is_700_l3011_301128

/-- A board game with the following properties:
  - The board is 7 × 8 (7 rows and 8 columns)
  - Two players take turns placing pieces
  - The second player (Gretel) earns 4 points for each piece already in the same row
  - Gretel earns 3 points for each piece already in the same column
  - The game ends when all cells are filled -/
structure BoardGame where
  rows : Nat
  cols : Nat
  row_points : Nat
  col_points : Nat

/-- The maximum number of points Gretel can earn in the game -/
def max_points (game : BoardGame) : Nat :=
  700

/-- Theorem stating that the maximum number of points Gretel can earn is 700 -/
theorem max_points_is_700 (game : BoardGame) 
  (h1 : game.rows = 7) 
  (h2 : game.cols = 8) 
  (h3 : game.row_points = 4) 
  (h4 : game.col_points = 3) : 
  max_points game = 700 := by
  sorry

#check max_points_is_700

end NUMINAMATH_CALUDE_max_points_is_700_l3011_301128


namespace NUMINAMATH_CALUDE_conference_handshakes_l3011_301111

theorem conference_handshakes (total_attendees : ℕ) (leaders : ℕ) (participants : ℕ)
  (h1 : total_attendees = 30)
  (h2 : leaders = 5)
  (h3 : participants = total_attendees - leaders)
  (h4 : leaders + participants = total_attendees) :
  (leaders * (total_attendees - 1) - leaders * (leaders - 1) / 2) = 135 := by
  sorry

end NUMINAMATH_CALUDE_conference_handshakes_l3011_301111


namespace NUMINAMATH_CALUDE_complex_zero_of_polynomial_l3011_301198

def is_valid_polynomial (P : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℤ), ∀ x, P x = x^4 + a*x^3 + b*x^2 + c*x + (P 0)

theorem complex_zero_of_polynomial 
  (P : ℝ → ℝ) 
  (h_valid : is_valid_polynomial P) 
  (h_zero1 : P 3 = 0) 
  (h_zero2 : P (-1) = 0) : 
  P (3/2) = (15 : ℝ)/4 :=
sorry

end NUMINAMATH_CALUDE_complex_zero_of_polynomial_l3011_301198


namespace NUMINAMATH_CALUDE_soccer_handshakes_l3011_301129

theorem soccer_handshakes (players_per_team : ℕ) (num_teams : ℕ) (num_referees : ℕ) : 
  players_per_team = 11 → num_teams = 2 → num_referees = 3 →
  (players_per_team * players_per_team) + (players_per_team * num_teams * num_referees) = 187 :=
by sorry

end NUMINAMATH_CALUDE_soccer_handshakes_l3011_301129


namespace NUMINAMATH_CALUDE_average_after_12th_innings_l3011_301140

/-- Represents a batsman's performance over multiple innings -/
structure BatsmanPerformance where
  innings : ℕ
  lastScore : ℕ
  averageIncrease : ℕ
  neverNotOut : Bool

/-- Calculates the average score after the last innings -/
def averageAfterLastInnings (performance : BatsmanPerformance) : ℕ :=
  sorry

/-- Theorem stating the average after the 12th innings -/
theorem average_after_12th_innings (performance : BatsmanPerformance) 
  (h1 : performance.innings = 12)
  (h2 : performance.lastScore = 75)
  (h3 : performance.averageIncrease = 1)
  (h4 : performance.neverNotOut = true) :
  averageAfterLastInnings performance = 64 :=
sorry

end NUMINAMATH_CALUDE_average_after_12th_innings_l3011_301140


namespace NUMINAMATH_CALUDE_whack_a_mole_tickets_kaleb_whack_a_mole_tickets_l3011_301134

/-- Given that Kaleb won tickets from two games and could buy a certain number of candies,
    we prove how many tickets he won from one of the games. -/
theorem whack_a_mole_tickets (skee_ball_tickets : ℕ) (candy_cost : ℕ) (candies_bought : ℕ) : ℕ :=
  let total_tickets := candy_cost * candies_bought
  total_tickets - skee_ball_tickets

/-- Proof of the specific problem where Kaleb won 8 tickets playing 'whack a mole' -/
theorem kaleb_whack_a_mole_tickets : whack_a_mole_tickets 7 5 3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_whack_a_mole_tickets_kaleb_whack_a_mole_tickets_l3011_301134


namespace NUMINAMATH_CALUDE_xiao_gang_steps_for_one_kilocalorie_l3011_301121

/-- The number of steps Xiao Gang walks for 1 kilocalorie of energy -/
def xiao_gang_steps : ℕ := 30

/-- The number of steps Xiao Qiong walks for 1 kilocalorie of energy -/
def xiao_qiong_steps : ℕ := xiao_gang_steps + 15

/-- The total steps Xiao Gang walks for a certain amount of energy -/
def xiao_gang_total_steps : ℕ := 9000

/-- The total steps Xiao Qiong walks for the same amount of energy as Xiao Gang -/
def xiao_qiong_total_steps : ℕ := 13500

theorem xiao_gang_steps_for_one_kilocalorie :
  xiao_gang_steps = 30 ∧
  xiao_qiong_steps = xiao_gang_steps + 15 ∧
  xiao_gang_total_steps * xiao_qiong_steps = xiao_qiong_total_steps * xiao_gang_steps :=
by sorry

end NUMINAMATH_CALUDE_xiao_gang_steps_for_one_kilocalorie_l3011_301121


namespace NUMINAMATH_CALUDE_largest_positive_root_bound_l3011_301130

theorem largest_positive_root_bound (a₂ a₁ a₀ : ℝ) (h1 : |a₂| ≤ 3) (h2 : |a₁| ≤ 3) (h3 : |a₀| ≤ 3) (h4 : a₂ + a₁ + a₀ = -6) :
  ∃ r : ℝ, 2 < r ∧ r < 3 ∧ r^3 + a₂*r^2 + a₁*r + a₀ = 0 ∧
  ∀ x : ℝ, x > 0 ∧ x^3 + a₂*x^2 + a₁*x + a₀ = 0 → x ≤ r :=
by sorry

end NUMINAMATH_CALUDE_largest_positive_root_bound_l3011_301130


namespace NUMINAMATH_CALUDE_bob_wins_2033_alice_wins_2034_l3011_301141

/-- Represents the possible moves for each player -/
inductive Move
| Alice : (n : Nat) → n = 2 ∨ n = 5 → Move
| Bob : (n : Nat) → n = 1 ∨ n = 3 ∨ n = 4 → Move

/-- Represents the game state -/
structure GameState where
  coins : Nat
  aliceTurn : Bool

/-- Determines if the current state is a winning position for the player whose turn it is -/
def isWinningPosition (state : GameState) : Prop := sorry

/-- The game ends when there are no valid moves or Alice wins instantly -/
def gameOver (state : GameState) : Prop :=
  (state.coins < 1) ∨ (state.aliceTurn ∧ state.coins = 5)

/-- Applies a move to the current game state -/
def applyMove (state : GameState) (move : Move) : GameState := sorry

/-- Theorem stating that Bob has a winning strategy when starting with 2033 coins -/
theorem bob_wins_2033 :
  ∃ (strategy : GameState → Move),
    ∀ (aliceFirst : Bool),
      isWinningPosition (GameState.mk 2033 (¬aliceFirst)) := sorry

/-- Theorem stating that Alice has a winning strategy when starting with 2034 coins -/
theorem alice_wins_2034 :
  ∃ (strategy : GameState → Move),
    ∀ (aliceFirst : Bool),
      isWinningPosition (GameState.mk 2034 aliceFirst) := sorry

end NUMINAMATH_CALUDE_bob_wins_2033_alice_wins_2034_l3011_301141


namespace NUMINAMATH_CALUDE_medical_staff_distribution_l3011_301105

/-- Represents the number of medical staff members -/
def num_staff : ℕ := 4

/-- Represents the number of communities -/
def num_communities : ℕ := 3

/-- Represents the constraint that A and B must be together -/
def a_and_b_together : Prop := True

/-- Represents the constraint that each community must have at least one person -/
def each_community_nonempty : Prop := True

/-- The number of ways to distribute the medical staff among communities -/
def distribution_count : ℕ := 6

/-- Theorem stating that the number of ways to distribute the medical staff
    among communities, given the constraints, is equal to 6 -/
theorem medical_staff_distribution :
  (num_staff = 4) →
  (num_communities = 3) →
  a_and_b_together →
  each_community_nonempty →
  distribution_count = 6 := by
  sorry

end NUMINAMATH_CALUDE_medical_staff_distribution_l3011_301105


namespace NUMINAMATH_CALUDE_cube_surface_area_l3011_301173

/-- Given a cube with vertices A, B, and C, prove that its surface area is 150 -/
theorem cube_surface_area (A B C : ℝ × ℝ × ℝ) : 
  A = (2, 5, 3) → B = (2, 10, 3) → C = (2, 5, 8) → 
  (let surface_area := 6 * (dist A B) ^ 2
   surface_area = 150) := by
  sorry

#check cube_surface_area

end NUMINAMATH_CALUDE_cube_surface_area_l3011_301173


namespace NUMINAMATH_CALUDE_ellipse_y_axis_iff_l3011_301131

/-- Predicate to check if an equation represents an ellipse with foci on the y-axis -/
def is_ellipse_y_axis (m n : ℝ) : Prop :=
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧ m = 1 / (a^2) ∧ n = 1 / (b^2)

/-- The condition for m and n is necessary and sufficient for representing an ellipse with foci on the y-axis -/
theorem ellipse_y_axis_iff (m n : ℝ) :
  is_ellipse_y_axis m n ↔ m > n ∧ n > 0 := by sorry

end NUMINAMATH_CALUDE_ellipse_y_axis_iff_l3011_301131


namespace NUMINAMATH_CALUDE_parallelogram_base_formula_l3011_301162

/-- Given two right-angled parallelograms with bases x and z, and heights y and u,
    this theorem proves the formula for x given certain conditions. -/
theorem parallelogram_base_formula 
  (x z y u S p s s' : ℝ) 
  (h1 : x * y + z * u = S) 
  (h2 : x + z = p) 
  (h3 : z * y = s) 
  (h4 : x * u = s') : 
  x = (p * (2 * s' + S) + Real.sqrt (p^2 * (2 * s' + S)^2 - 4 * p^2 * s' * (s + s' + S))) / (2 * (s + s' + S)) ∨ 
  x = (p * (2 * s' + S) - Real.sqrt (p^2 * (2 * s' + S)^2 - 4 * p^2 * s' * (s + s' + S))) / (2 * (s + s' + S)) :=
sorry

end NUMINAMATH_CALUDE_parallelogram_base_formula_l3011_301162


namespace NUMINAMATH_CALUDE_transfer_amount_christinas_transfer_l3011_301191

/-- The amount transferred out of a bank account is equal to the difference
    between the initial balance and the final balance. -/
theorem transfer_amount (initial_balance final_balance : ℕ) 
  (h : initial_balance ≥ final_balance) :
  initial_balance - final_balance = 
  (initial_balance : ℤ) - (final_balance : ℤ) :=
by sorry

/-- Christina's bank transfer problem -/
theorem christinas_transfer : 
  (27004 : ℕ) - (26935 : ℕ) = (69 : ℕ) :=
by sorry

end NUMINAMATH_CALUDE_transfer_amount_christinas_transfer_l3011_301191


namespace NUMINAMATH_CALUDE_billy_cherries_l3011_301178

theorem billy_cherries (cherries_eaten cherries_left : ℕ) 
  (h1 : cherries_eaten = 72)
  (h2 : cherries_left = 2) : 
  cherries_eaten + cherries_left = 74 := by
  sorry

end NUMINAMATH_CALUDE_billy_cherries_l3011_301178


namespace NUMINAMATH_CALUDE_factorization_proof_l3011_301180

theorem factorization_proof (x y : ℝ) : -3 * x^3 * y + 27 * x * y = -3 * x * y * (x + 3) * (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l3011_301180


namespace NUMINAMATH_CALUDE_hill_climbing_speed_l3011_301165

/-- Proves that given a round trip journey with specified conditions, 
    the average speed while going up is 2.25 km/h -/
theorem hill_climbing_speed 
  (time_up : ℝ) 
  (time_down : ℝ) 
  (avg_speed_total : ℝ) 
  (h1 : time_up = 4) 
  (h2 : time_down = 2) 
  (h3 : avg_speed_total = 3) : 
  (avg_speed_total * (time_up + time_down) / 2) / time_up = 2.25 := by
  sorry

end NUMINAMATH_CALUDE_hill_climbing_speed_l3011_301165


namespace NUMINAMATH_CALUDE_inverse_proportion_solution_l3011_301164

/-- Given two inversely proportional quantities, this is their constant product. -/
def InverseProportionConstant (x y : ℝ) : ℝ := x * y

theorem inverse_proportion_solution (a b c : ℝ → ℝ) :
  (∀ x y, InverseProportionConstant (a x) (b x) = InverseProportionConstant (a y) (b y)) →
  (∀ x y, InverseProportionConstant (b x) (c x) = InverseProportionConstant (b y) (c y)) →
  a 1 = 40 →
  b 1 = 5 →
  c 2 = 10 →
  b 2 = 7 →
  c 3 = 5.6 →
  a 3 = 16 :=
by sorry

end NUMINAMATH_CALUDE_inverse_proportion_solution_l3011_301164


namespace NUMINAMATH_CALUDE_garrison_size_l3011_301146

-- Define the parameters
def initial_days : ℕ := 31
def days_passed : ℕ := 27
def people_left : ℕ := 200
def remaining_days : ℕ := 8

-- Theorem statement
theorem garrison_size :
  ∀ (M : ℕ),
  (M * (initial_days - days_passed) = (M - people_left) * remaining_days) →
  M = 400 :=
by
  sorry


end NUMINAMATH_CALUDE_garrison_size_l3011_301146


namespace NUMINAMATH_CALUDE_negation_of_proposition_l3011_301194

theorem negation_of_proposition (a b c : ℝ) :
  (¬ (a + b + c = 3 → a^2 + b^2 + c^2 ≥ 3)) ↔ (a + b + c ≠ 3 → a^2 + b^2 + c^2 < 3) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l3011_301194


namespace NUMINAMATH_CALUDE_max_intersections_proof_l3011_301199

/-- The maximum number of intersection points between two circles -/
def max_circle_intersections : ℕ := 2

/-- The maximum number of intersection points between a line and a circle -/
def max_line_circle_intersections : ℕ := 2

/-- The maximum number of intersection points between two lines -/
def max_line_line_intersections : ℕ := 1

/-- The number of circles -/
def num_circles : ℕ := 2

/-- The number of lines -/
def num_lines : ℕ := 3

/-- The maximum number of intersection points between all figures -/
def max_total_intersections : ℕ := 17

theorem max_intersections_proof :
  max_total_intersections = 
    (num_circles.choose 2) * max_circle_intersections +
    num_lines * num_circles * max_line_circle_intersections +
    (num_lines.choose 2) * max_line_line_intersections :=
by
  sorry

end NUMINAMATH_CALUDE_max_intersections_proof_l3011_301199


namespace NUMINAMATH_CALUDE_unique_number_exists_l3011_301193

theorem unique_number_exists : ∃! n : ℕ, 
  let sum := 555 + 445
  let diff := 555 - 445
  n / sum = 2 * diff ∧ n % sum = 80 ∧ n = 220080 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_exists_l3011_301193


namespace NUMINAMATH_CALUDE_probability_sum_greater_than_eight_given_blue_three_or_six_l3011_301138

/-- Represents the possible outcomes of a die roll -/
inductive DieOutcome
  | one
  | two
  | three
  | four
  | five
  | six

/-- Represents the outcome of rolling two dice -/
structure TwoDiceOutcome where
  red : DieOutcome
  blue : DieOutcome

/-- The sample space of all possible outcomes when rolling two dice -/
def sampleSpace : Set TwoDiceOutcome := sorry

/-- The event where the blue die shows either 3 or 6 -/
def blueThreeOrSix : Set TwoDiceOutcome := sorry

/-- The event where the sum of the numbers on both dice is greater than 8 -/
def sumGreaterThanEight : Set TwoDiceOutcome := sorry

/-- The probability of an event given a condition -/
def conditionalProbability (event condition : Set TwoDiceOutcome) : ℚ := sorry

theorem probability_sum_greater_than_eight_given_blue_three_or_six :
  conditionalProbability sumGreaterThanEight blueThreeOrSix = 5/12 := by
  sorry

end NUMINAMATH_CALUDE_probability_sum_greater_than_eight_given_blue_three_or_six_l3011_301138


namespace NUMINAMATH_CALUDE_eight_by_eight_unfolds_to_nine_l3011_301166

/-- Represents a square grid -/
structure Grid :=
  (size : ℕ)

/-- Represents a folded and cut grid -/
structure FoldedCutGrid :=
  (original : Grid)
  (folded_size : ℕ)
  (cut : Bool)

/-- Counts the number of parts after unfolding a cut grid -/
def count_parts (g : FoldedCutGrid) : ℕ :=
  sorry

/-- Theorem stating that an 8x8 grid folded to 1x1 and cut unfolds into 9 parts -/
theorem eight_by_eight_unfolds_to_nine :
  ∀ (g : FoldedCutGrid),
    g.original.size = 8 →
    g.folded_size = 1 →
    g.cut = true →
    count_parts g = 9 :=
  sorry

end NUMINAMATH_CALUDE_eight_by_eight_unfolds_to_nine_l3011_301166


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l3011_301151

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^2 - x + 2 < 0) ↔ (∃ x : ℝ, x^2 - x + 2 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l3011_301151


namespace NUMINAMATH_CALUDE_greater_than_negative_five_by_negative_six_l3011_301155

theorem greater_than_negative_five_by_negative_six :
  ((-5) + (-6) : ℤ) = -11 :=
by sorry

end NUMINAMATH_CALUDE_greater_than_negative_five_by_negative_six_l3011_301155


namespace NUMINAMATH_CALUDE_polygon_with_equal_angle_sums_l3011_301168

theorem polygon_with_equal_angle_sums (n : ℕ) (h : n > 2) :
  (n - 2) * 180 = 360 → n = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_polygon_with_equal_angle_sums_l3011_301168


namespace NUMINAMATH_CALUDE_number_greater_than_fraction_l3011_301144

theorem number_greater_than_fraction : ∃ x : ℝ, x = 40 ∧ 0.8 * x > (2 / 5) * 25 + 22 := by
  sorry

end NUMINAMATH_CALUDE_number_greater_than_fraction_l3011_301144


namespace NUMINAMATH_CALUDE_pascal_triangle_46th_row_45th_number_l3011_301181

theorem pascal_triangle_46th_row_45th_number : 
  let n : ℕ := 46  -- The row number (0-indexed)
  let k : ℕ := 44  -- The position in the row (0-indexed)
  Nat.choose n k = 1035 := by
sorry

end NUMINAMATH_CALUDE_pascal_triangle_46th_row_45th_number_l3011_301181


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l3011_301157

-- Define conditions p and q
def condition_p (x y : ℝ) : Prop := x + y > 4 ∧ x * y > 4
def condition_q (x y : ℝ) : Prop := x > 2 ∧ y > 2

-- Theorem statement
theorem p_necessary_not_sufficient_for_q :
  (∀ x y : ℝ, condition_q x y → condition_p x y) ∧
  ¬(∀ x y : ℝ, condition_p x y → condition_q x y) := by
  sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l3011_301157


namespace NUMINAMATH_CALUDE_puppy_sale_cost_l3011_301176

theorem puppy_sale_cost (total_cost : ℕ) (non_sale_cost : ℕ) (num_puppies : ℕ) (num_non_sale : ℕ) :
  total_cost = 800 →
  non_sale_cost = 175 →
  num_puppies = 5 →
  num_non_sale = 2 →
  (total_cost - num_non_sale * non_sale_cost) / (num_puppies - num_non_sale) = 150 := by
  sorry

end NUMINAMATH_CALUDE_puppy_sale_cost_l3011_301176


namespace NUMINAMATH_CALUDE_equal_sum_blocks_iff_three_l3011_301196

/-- A function that checks if a prime number p allows the sequence of natural numbers
    from 1 to p to be divided into several consecutive blocks with identical sums -/
def has_equal_sum_blocks (p : ℕ) : Prop :=
  ∃ (k : ℕ) (m : ℕ), 
    Prime p ∧
    k > 1 ∧
    m < p ∧
    (m * (m + 1)) / 2 = k * p ∧
    p * (p + 1) / 2 = k * p

/-- Theorem stating that the only prime number p that satisfies the condition is 3 -/
theorem equal_sum_blocks_iff_three :
  ∀ p : ℕ, Prime p → (has_equal_sum_blocks p ↔ p = 3) :=
by sorry

end NUMINAMATH_CALUDE_equal_sum_blocks_iff_three_l3011_301196


namespace NUMINAMATH_CALUDE_intersection_point_property_l3011_301119

theorem intersection_point_property (x : Real) : 
  x ∈ Set.Ioo 0 (π / 2) → 
  6 * Real.cos x = 9 * Real.tan x → 
  Real.sin x = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_intersection_point_property_l3011_301119


namespace NUMINAMATH_CALUDE_sum_of_factors_360_l3011_301163

/-- The sum of positive factors of a natural number n -/
def sum_of_factors (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of positive factors of 360 is 1170 -/
theorem sum_of_factors_360 : sum_of_factors 360 = 1170 := by sorry

end NUMINAMATH_CALUDE_sum_of_factors_360_l3011_301163


namespace NUMINAMATH_CALUDE_largest_inscribed_square_side_length_l3011_301174

theorem largest_inscribed_square_side_length 
  (outer_square_side : ℝ) 
  (triangle_side : ℝ) 
  (inscribed_square_side : ℝ) :
  outer_square_side = 12 →
  triangle_side = 6 * Real.sqrt 2 * (Real.sqrt 3 - 1) →
  2 * inscribed_square_side * Real.sqrt 2 + triangle_side = 12 * Real.sqrt 2 →
  inscribed_square_side = 9 - 3 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_largest_inscribed_square_side_length_l3011_301174


namespace NUMINAMATH_CALUDE_park_trees_l3011_301177

theorem park_trees (willows : ℕ) (oaks : ℕ) : 
  willows = 36 → oaks = willows + 11 → willows + oaks = 83 := by
  sorry

end NUMINAMATH_CALUDE_park_trees_l3011_301177


namespace NUMINAMATH_CALUDE_school_gender_ratio_l3011_301189

/-- Given a school with a 5:4 ratio of boys to girls and 1500 boys, prove there are 1200 girls. -/
theorem school_gender_ratio (num_boys : ℕ) (num_girls : ℕ) : 
  (num_boys : ℚ) / num_girls = 5 / 4 →
  num_boys = 1500 →
  num_girls = 1200 := by
  sorry

end NUMINAMATH_CALUDE_school_gender_ratio_l3011_301189


namespace NUMINAMATH_CALUDE_sum_product_over_sum_squares_l3011_301185

theorem sum_product_over_sum_squares (x y z : ℝ) (h1 : x ≠ y ∧ y ≠ z ∧ x ≠ z) (h2 : x + y + z = 3) :
  (x*y + y*z + z*x) / (x^2 + y^2 + z^2) = 9 / (2*(x^2 + y^2 + z^2)) - 1/2 := by
sorry

end NUMINAMATH_CALUDE_sum_product_over_sum_squares_l3011_301185


namespace NUMINAMATH_CALUDE_cubic_polynomial_special_case_l3011_301197

-- Define a cubic polynomial
def cubic_polynomial (p : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x, p x = a*x^3 + b*x^2 + c*x + d

-- Define the theorem
theorem cubic_polynomial_special_case (p : ℝ → ℝ) 
  (h_cubic : cubic_polynomial p)
  (h1 : p 1 = 1)
  (h2 : p 2 = 1/8)
  (h3 : p 3 = 1/27) :
  p 4 = 1/576 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_special_case_l3011_301197


namespace NUMINAMATH_CALUDE_plan_d_cost_effective_l3011_301106

/-- The cost in cents for Plan C given the number of minutes used -/
def plan_c_cost (minutes : ℕ) : ℕ := 15 * minutes

/-- The cost in cents for Plan D given the number of minutes used -/
def plan_d_cost (minutes : ℕ) : ℕ := 2500 + 12 * minutes

/-- The minimum number of whole minutes for Plan D to be cost-effective -/
def min_minutes_for_plan_d : ℕ := 834

theorem plan_d_cost_effective :
  (∀ m : ℕ, m ≥ min_minutes_for_plan_d → plan_d_cost m < plan_c_cost m) ∧
  (∀ m : ℕ, m < min_minutes_for_plan_d → plan_d_cost m ≥ plan_c_cost m) :=
sorry

end NUMINAMATH_CALUDE_plan_d_cost_effective_l3011_301106


namespace NUMINAMATH_CALUDE_diamond_negative_two_three_l3011_301156

-- Define the diamond operation
def diamond (a b : ℝ) : ℝ := a + a * b^2 - b + 1

-- Theorem statement
theorem diamond_negative_two_three : diamond (-2) 3 = -22 := by sorry

end NUMINAMATH_CALUDE_diamond_negative_two_three_l3011_301156


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_l3011_301150

theorem gcd_lcm_sum : Nat.gcd 8 12 + Nat.lcm 8 12 = 28 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_l3011_301150


namespace NUMINAMATH_CALUDE_total_cards_l3011_301167

theorem total_cards (initial_cards added_cards : ℕ) 
  (h1 : initial_cards = 4)
  (h2 : added_cards = 3) :
  initial_cards + added_cards = 7 := by
sorry

end NUMINAMATH_CALUDE_total_cards_l3011_301167


namespace NUMINAMATH_CALUDE_value_range_of_f_l3011_301143

def f (x : ℝ) := x^2 - 4*x

theorem value_range_of_f :
  ∀ x ∈ Set.Icc 0 5, -4 ≤ f x ∧ f x ≤ 5 ∧
  (∃ x₁ ∈ Set.Icc 0 5, f x₁ = -4) ∧
  (∃ x₂ ∈ Set.Icc 0 5, f x₂ = 5) :=
sorry

end NUMINAMATH_CALUDE_value_range_of_f_l3011_301143


namespace NUMINAMATH_CALUDE_two_numbers_problem_l3011_301149

theorem two_numbers_problem (x y : ℝ) (h1 : x + y = 40) (h2 : x - y = 6) : x = 23 ∧ y = 17 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_problem_l3011_301149


namespace NUMINAMATH_CALUDE_cricket_team_throwers_l3011_301152

/-- Represents the number of players in different categories on a cricket team. -/
structure CricketTeam where
  total : ℕ
  throwers : ℕ
  left_handed : ℕ
  right_handed : ℕ

/-- Theorem stating the number of throwers on the cricket team given the conditions. -/
theorem cricket_team_throwers (team : CricketTeam) : 
  team.total = 55 ∧ 
  team.throwers + team.left_handed + team.right_handed = team.total ∧
  team.left_handed = (team.left_handed + team.right_handed) / 3 ∧
  team.throwers + team.right_handed = 49 →
  team.throwers = 37 := by
  sorry


end NUMINAMATH_CALUDE_cricket_team_throwers_l3011_301152


namespace NUMINAMATH_CALUDE_min_value_of_function_lower_bound_is_tight_l3011_301195

theorem min_value_of_function (x : ℝ) (h : x ≥ 0) :
  (3 * x^2 + 6 * x + 19) / (8 * (1 + x)) ≥ Real.sqrt 3 :=
sorry

theorem lower_bound_is_tight :
  ∃ x : ℝ, x ≥ 0 ∧ (3 * x^2 + 6 * x + 19) / (8 * (1 + x)) = Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_function_lower_bound_is_tight_l3011_301195


namespace NUMINAMATH_CALUDE_equation_solutions_l3011_301190

theorem equation_solutions : 
  (∀ x : ℝ, (x - 1)^2 = 25 ↔ x = 6 ∨ x = -4) ∧
  (∀ x : ℝ, x^2 - 4*x + 3 = 0 ↔ x = 1 ∨ x = 3) ∧
  (∀ x : ℝ, (2*x + 1)^2 = 2*(2*x + 1) ↔ x = -1/2 ∨ x = 1/2) ∧
  (∀ x : ℝ, 2*x^2 - 5*x + 3 = 0 ↔ x = 1 ∨ x = 3/2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3011_301190


namespace NUMINAMATH_CALUDE_comparison_abc_l3011_301154

theorem comparison_abc (a b c : ℝ) 
  (ha : a = Real.rpow 0.7 0.6)
  (hb : b = Real.rpow 0.6 (-0.6))
  (hc : c = Real.rpow 0.6 0.7) :
  b > a ∧ a > c := by sorry

end NUMINAMATH_CALUDE_comparison_abc_l3011_301154


namespace NUMINAMATH_CALUDE_animal_rescue_proof_l3011_301188

theorem animal_rescue_proof (sheep cows dogs pigs chickens rabbits ducks : ℕ) : 
  sheep = 20 ∧ cows = 10 ∧ dogs = 14 ∧ pigs = 8 ∧ chickens = 12 ∧ rabbits = 6 ∧ ducks = 15 →
  ∃ (saved_sheep saved_cows saved_dogs saved_pigs saved_chickens saved_rabbits saved_ducks : ℕ),
    saved_sheep = 14 ∧
    saved_cows = 6 ∧
    saved_dogs = 11 ∧
    saved_pigs = 6 ∧
    saved_chickens = 10 ∧
    saved_rabbits = 5 ∧
    saved_ducks = 10 ∧
    saved_sheep = sheep - (sheep * 3 / 10) ∧
    (cows - saved_cows) * (cows - saved_cows) = pigs - saved_pigs ∧
    saved_dogs = dogs * 3 / 4 ∧
    chickens - saved_chickens = (cows - saved_cows) / 2 ∧
    rabbits - saved_rabbits = 1 ∧
    saved_ducks = saved_rabbits * 2 ∧
    saved_sheep + saved_cows + saved_dogs + saved_pigs + saved_chickens + saved_rabbits + saved_ducks ≥ 50 :=
by
  sorry

end NUMINAMATH_CALUDE_animal_rescue_proof_l3011_301188


namespace NUMINAMATH_CALUDE_international_shipping_charge_l3011_301118

theorem international_shipping_charge 
  (total_letters : ℕ) 
  (standard_postage : ℚ) 
  (international_letters : ℕ) 
  (total_cost : ℚ) : 
  total_letters = 4 → 
  standard_postage = 108/100 → 
  international_letters = 2 → 
  total_cost = 460/100 → 
  (total_cost - total_letters * standard_postage) / international_letters * 100 = 14 :=
by sorry

end NUMINAMATH_CALUDE_international_shipping_charge_l3011_301118


namespace NUMINAMATH_CALUDE_events_A_B_complementary_l3011_301186

-- Define the sample space for a die throw
def DieOutcome := Fin 6

-- Define event A
def eventA (outcome : DieOutcome) : Prop :=
  outcome.val ≤ 2

-- Define event B
def eventB (outcome : DieOutcome) : Prop :=
  outcome.val ≥ 3

-- Define event C (not used in the theorem, but included for completeness)
def eventC (outcome : DieOutcome) : Prop :=
  outcome.val % 2 = 1

-- Theorem stating that events A and B are complementary
theorem events_A_B_complementary :
  ∀ (outcome : DieOutcome), eventA outcome ↔ ¬ eventB outcome :=
by
  sorry


end NUMINAMATH_CALUDE_events_A_B_complementary_l3011_301186


namespace NUMINAMATH_CALUDE_chocolate_theorem_l3011_301120

def chocolate_problem (initial_bars : ℕ) (friends : ℕ) (returned_bars : ℕ) (piper_difference : ℕ) (remaining_bars : ℕ) : Prop :=
  ∃ (x : ℚ),
    -- Thomas and friends take x bars initially
    x > 0 ∧
    -- One friend returns 5 bars
    (x - returned_bars) > 0 ∧
    -- Piper takes 5 fewer bars than Thomas and friends
    (x - returned_bars - piper_difference) > 0 ∧
    -- Total bars taken plus remaining bars equals initial bars
    (x - returned_bars) + (x - returned_bars - piper_difference) + remaining_bars = initial_bars ∧
    -- The fraction of bars Thomas and friends took initially
    x / initial_bars = 21 / 80

theorem chocolate_theorem :
  chocolate_problem 200 5 5 5 110 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_theorem_l3011_301120


namespace NUMINAMATH_CALUDE_probability_nine_heads_in_twelve_flips_l3011_301183

def n : ℕ := 12
def k : ℕ := 9

theorem probability_nine_heads_in_twelve_flips :
  (n.choose k) / (2^n : ℚ) = 220 / 4096 := by sorry

end NUMINAMATH_CALUDE_probability_nine_heads_in_twelve_flips_l3011_301183


namespace NUMINAMATH_CALUDE_complement_A_complement_A_inter_B_l3011_301123

-- Define the sets
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | -1 < x ∧ x ≤ 5}
def B : Set ℝ := {x | x < 3 ∨ x ≥ 15}

-- State the theorems
theorem complement_A : Set.compl A = {x | x ≤ -1 ∨ x > 5} := by sorry

theorem complement_A_inter_B : Set.compl (A ∩ B) = {x | x ≤ -1 ∨ x ≥ 3} := by sorry

end NUMINAMATH_CALUDE_complement_A_complement_A_inter_B_l3011_301123


namespace NUMINAMATH_CALUDE_odd_integers_equality_l3011_301153

theorem odd_integers_equality (a b c d k m : ℤ) : 
  Odd a → Odd b → Odd c → Odd d →
  0 < a → a < b → b < c → c < d →
  a * d = b * c →
  a + d = 2 * k →
  b + c = 2 * m →
  a = 1 := by sorry

end NUMINAMATH_CALUDE_odd_integers_equality_l3011_301153


namespace NUMINAMATH_CALUDE_roller_coaster_cost_l3011_301103

theorem roller_coaster_cost (total_tickets : ℕ) (ferris_wheel_cost : ℕ) 
  (h1 : total_tickets = 13)
  (h2 : ferris_wheel_cost = 5)
  (h3 : ∃ x : ℕ, total_tickets = ferris_wheel_cost + x + x) :
  ∃ roller_coaster_cost : ℕ, roller_coaster_cost = 4 ∧ 
    total_tickets = ferris_wheel_cost + roller_coaster_cost + roller_coaster_cost :=
by sorry

end NUMINAMATH_CALUDE_roller_coaster_cost_l3011_301103


namespace NUMINAMATH_CALUDE_max_sum_abs_on_circle_l3011_301132

theorem max_sum_abs_on_circle :
  ∃ (M : ℝ), M = 2 * Real.sqrt 2 ∧
  (∀ x y : ℝ, x^2 + y^2 = 4 → |x| + |y| ≤ M) ∧
  (∃ x y : ℝ, x^2 + y^2 = 4 ∧ |x| + |y| = M) := by
sorry

end NUMINAMATH_CALUDE_max_sum_abs_on_circle_l3011_301132


namespace NUMINAMATH_CALUDE_oliver_remaining_money_l3011_301169

def initial_cash : ℕ := 40
def initial_quarters : ℕ := 200
def quarter_value : ℚ := 0.25
def cash_to_sister : ℕ := 5
def quarters_to_sister : ℕ := 120

theorem oliver_remaining_money :
  let total_initial := initial_cash + initial_quarters * quarter_value
  let total_to_sister := cash_to_sister + quarters_to_sister * quarter_value
  total_initial - total_to_sister = 55 := by
sorry

end NUMINAMATH_CALUDE_oliver_remaining_money_l3011_301169


namespace NUMINAMATH_CALUDE_min_value_a_l3011_301115

theorem min_value_a (a : ℝ) : 
  (∀ x y : ℝ, x > 0 ∧ y > 0 → x^2 + 2*x*y ≤ a*(x^2 + y^2)) →
  a ≥ (Real.sqrt 5 + 1) / 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_a_l3011_301115


namespace NUMINAMATH_CALUDE_triangle_square_equal_area_l3011_301171

/-- Given a square with perimeter 32 units and a right triangle with height 40 units,
    if the square and triangle have the same area, and the triangle's base is twice
    the length of x, then x = 8/5. -/
theorem triangle_square_equal_area (x : ℝ) : 
  let square_perimeter : ℝ := 32
  let square_side : ℝ := square_perimeter / 4
  let square_area : ℝ := square_side ^ 2
  let triangle_height : ℝ := 40
  let triangle_base : ℝ := 2 * x
  let triangle_area : ℝ := (1 / 2) * triangle_base * triangle_height
  square_area = triangle_area → x = 8 / 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_square_equal_area_l3011_301171


namespace NUMINAMATH_CALUDE_basketball_win_requirement_l3011_301145

theorem basketball_win_requirement (total_games : ℕ) (first_games : ℕ) (wins_so_far : ℕ) (remaining_games : ℕ) 
  (h1 : total_games = first_games + remaining_games)
  (h2 : total_games = 110)
  (h3 : first_games = 60)
  (h4 : wins_so_far = 48)
  (h5 : remaining_games = 50) :
  ∃ (additional_wins : ℕ), 
    (wins_so_far + additional_wins : ℚ) / total_games = 3/4 ∧ 
    additional_wins = 35 := by
  sorry

#check basketball_win_requirement

end NUMINAMATH_CALUDE_basketball_win_requirement_l3011_301145


namespace NUMINAMATH_CALUDE_quadratic_a_value_l3011_301159

/-- A quadratic function f(x) = ax^2 + bx + c -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c

theorem quadratic_a_value 
  (a b c : ℝ) 
  (f : ℝ → ℝ) 
  (h1 : f = QuadraticFunction a b c) 
  (h2 : f 0 = 3) 
  (h3 : ∀ x, f x ≤ f 2) 
  (h4 : f 2 = 5) : 
  a = -1/2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_a_value_l3011_301159


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l3011_301192

theorem inequality_and_equality_condition (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) : 
  Real.sqrt ((a + c) * (b + d)) ≥ Real.sqrt (a * b) + Real.sqrt (c * d) ∧ 
  (Real.sqrt ((a + c) * (b + d)) = Real.sqrt (a * b) + Real.sqrt (c * d) ↔ a * d = b * c) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l3011_301192


namespace NUMINAMATH_CALUDE_watermelon_seeds_count_l3011_301125

/-- The number of seeds in one watermelon -/
def seeds_per_watermelon : ℕ := 100

/-- The number of watermelons -/
def number_of_watermelons : ℕ := 4

/-- The total number of seeds in all watermelons -/
def total_seeds : ℕ := seeds_per_watermelon * number_of_watermelons

theorem watermelon_seeds_count : total_seeds = 400 := by
  sorry

end NUMINAMATH_CALUDE_watermelon_seeds_count_l3011_301125


namespace NUMINAMATH_CALUDE_square_with_circles_theorem_l3011_301102

/-- A square with side length 6 and three congruent circles inside it -/
structure SquareWithCircles where
  /-- Side length of the square -/
  side_length : ℝ
  side_length_eq : side_length = 6
  /-- Radius of the congruent circles -/
  radius : ℝ
  /-- Center of circle X -/
  center_x : ℝ × ℝ
  /-- Center of circle Y -/
  center_y : ℝ × ℝ
  /-- Center of circle Z -/
  center_z : ℝ × ℝ
  /-- X is tangent to sides AB and AD -/
  x_tangent : center_x.1 = radius ∧ center_x.2 = radius
  /-- Y is tangent to sides AB and BC -/
  y_tangent : center_y.1 = side_length - radius ∧ center_y.2 = radius
  /-- Z is tangent to side CD and both circles X and Y -/
  z_tangent : center_z.1 = side_length / 2 ∧ center_z.2 = side_length - radius

/-- The theorem to be proved -/
theorem square_with_circles_theorem (s : SquareWithCircles) :
  ∃ (m n : ℕ), s.radius = m - Real.sqrt n ∧ m + n = 195 := by
  sorry

end NUMINAMATH_CALUDE_square_with_circles_theorem_l3011_301102


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l3011_301107

theorem arithmetic_geometric_mean_inequality {a b : ℝ} (ha : 0 < a) (hb : 0 < b) :
  (a + b) / 2 ≥ Real.sqrt (a * b) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l3011_301107


namespace NUMINAMATH_CALUDE_milk_production_l3011_301136

theorem milk_production (y : ℝ) : 
  (y > 0) → 
  (y * (y + 1) * (y + 10)) / ((y + 2) * (y + 4)) = 
    (y + 10) / ((y + 4) * ((y + 2) / (y * (y + 1)))) := by
  sorry

end NUMINAMATH_CALUDE_milk_production_l3011_301136
