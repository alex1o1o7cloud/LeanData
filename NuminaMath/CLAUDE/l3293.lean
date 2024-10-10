import Mathlib

namespace triangular_prism_has_nine_edges_l3293_329380

/-- The number of sides in the base polygon of a triangular prism -/
def triangular_prism_base_sides : ℕ := 3

/-- The number of edges in a prism given the number of sides in its base polygon -/
def prism_edges (n : ℕ) : ℕ := 3 * n

/-- Theorem: A triangular prism has 9 edges -/
theorem triangular_prism_has_nine_edges :
  prism_edges triangular_prism_base_sides = 9 := by
  sorry

end triangular_prism_has_nine_edges_l3293_329380


namespace percentage_calculation_l3293_329375

theorem percentage_calculation : 
  (0.60 * 4500 * 0.40 * 2800) - (0.80 * 1750 + 0.35 * 3000) = 3021550 := by
  sorry

end percentage_calculation_l3293_329375


namespace distinct_sentences_count_l3293_329383

/-- Represents the number of variations for each phrase -/
def phrase_variations : Fin 4 → ℕ
  | 0 => 3  -- Phrase I
  | 1 => 2  -- Phrase II
  | 2 => 1  -- Phrase III (mandatory)
  | 3 => 2  -- Phrase IV

/-- Calculates the total number of combinations -/
def total_combinations : ℕ := 
  (phrase_variations 0) * (phrase_variations 1) * (phrase_variations 2) * (phrase_variations 3)

/-- The number of distinct meaningful sentences -/
def distinct_sentences : ℕ := total_combinations - 1

theorem distinct_sentences_count : distinct_sentences = 23 := by
  sorry

end distinct_sentences_count_l3293_329383


namespace geometric_progression_first_term_l3293_329373

theorem geometric_progression_first_term 
  (S : ℝ) 
  (sum_first_two : ℝ) 
  (h1 : S = 6) 
  (h2 : sum_first_two = 9/2) : 
  ∃ a : ℝ, (a = 9 ∨ a = 3) ∧ 
  ∃ r : ℝ, (r = 1/2 ∨ r = -1/2) ∧ 
  S = a / (1 - r) ∧ 
  sum_first_two = a + a * r :=
sorry

end geometric_progression_first_term_l3293_329373


namespace min_value_expression_l3293_329353

theorem min_value_expression (a b c d : ℝ) (h1 : b > d) (h2 : d > c) (h3 : c > a) (h4 : b ≠ 0) :
  ((a + b)^2 + (b - c)^2 + (d - c)^2 + (c - a)^2) / b^2 ≥ 9 ∧
  ∃ (a₀ b₀ c₀ d₀ : ℝ), b₀ > d₀ ∧ d₀ > c₀ ∧ c₀ > a₀ ∧ b₀ ≠ 0 ∧
    ((a₀ + b₀)^2 + (b₀ - c₀)^2 + (d₀ - c₀)^2 + (c₀ - a₀)^2) / b₀^2 = 9 :=
by sorry

end min_value_expression_l3293_329353


namespace scientific_notation_36000_l3293_329349

/-- Proves that 36000 is equal to 3.6 * 10^4 in scientific notation -/
theorem scientific_notation_36000 :
  36000 = 3.6 * (10 : ℝ)^4 := by
  sorry

end scientific_notation_36000_l3293_329349


namespace sufficient_but_not_necessary_l3293_329363

theorem sufficient_but_not_necessary : 
  (∃ x : ℝ, x < 2 ∧ ¬(1 < x ∧ x < 2)) ∧ 
  (∀ x : ℝ, 1 < x ∧ x < 2 → x < 2) :=
by sorry

end sufficient_but_not_necessary_l3293_329363


namespace travel_time_calculation_l3293_329330

/-- Given a person traveling at a constant speed for a certain distance,
    prove that the time taken is equal to the distance divided by the speed. -/
theorem travel_time_calculation (speed : ℝ) (distance : ℝ) (h1 : speed > 0) :
  let time := distance / speed
  speed = 20 ∧ distance = 50 → time = 2.5 := by
  sorry

end travel_time_calculation_l3293_329330


namespace division_remainder_3005_95_l3293_329320

theorem division_remainder_3005_95 : 3005 % 95 = 60 := by
  sorry

end division_remainder_3005_95_l3293_329320


namespace sum_of_squares_of_roots_sum_of_squares_of_roots_specific_equation_l3293_329379

theorem sum_of_squares_of_roots (a b c : ℚ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  a * r₁^2 + b * r₁ + c = 0 ∧ a * r₂^2 + b * r₂ + c = 0 →
  r₁^2 + r₂^2 = (b/a)^2 - 2*(c/a) :=
by sorry

theorem sum_of_squares_of_roots_specific_equation :
  let a : ℚ := 5
  let b : ℚ := 6
  let c : ℚ := -15
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  r₁^2 + r₂^2 = 186 / 25 :=
by sorry

end sum_of_squares_of_roots_sum_of_squares_of_roots_specific_equation_l3293_329379


namespace pencils_removed_l3293_329370

/-- Given a jar of pencils, prove that the number of pencils removed is correct. -/
theorem pencils_removed (original : ℕ) (remaining : ℕ) (removed : ℕ) : 
  original = 87 → remaining = 83 → removed = original - remaining → removed = 4 := by
  sorry

end pencils_removed_l3293_329370


namespace pauls_weekend_homework_l3293_329325

/-- Represents Paul's homework schedule for a week -/
structure HomeworkSchedule where
  weeknight_hours : ℕ  -- Hours of homework on a regular weeknight
  practice_nights : ℕ  -- Number of nights with practice (no homework)
  total_weeknights : ℕ -- Total number of weeknights
  average_hours : ℕ   -- Required average hours on non-practice nights

/-- Calculates the weekend homework hours based on Paul's schedule -/
def weekend_homework (schedule : HomeworkSchedule) : ℕ :=
  let non_practice_nights := schedule.total_weeknights - schedule.practice_nights
  let required_hours := non_practice_nights * schedule.average_hours
  let available_weeknight_hours := (schedule.total_weeknights - schedule.practice_nights) * schedule.weeknight_hours
  required_hours - available_weeknight_hours

/-- Theorem stating that Paul's weekend homework is 3 hours -/
theorem pauls_weekend_homework :
  let pauls_schedule : HomeworkSchedule := {
    weeknight_hours := 2,
    practice_nights := 2,
    total_weeknights := 5,
    average_hours := 3
  }
  weekend_homework pauls_schedule = 3 := by sorry

end pauls_weekend_homework_l3293_329325


namespace cindy_hit_nine_l3293_329310

/-- Represents a player in the dart-throwing contest -/
inductive Player
| Alice
| Ben
| Cindy
| Dave
| Ellen

/-- Represents the score of a single dart throw -/
def DartScore := Fin 15

/-- Represents the scores of three dart throws for a player -/
def PlayerScores := Fin 3 → DartScore

/-- The total score for a player is the sum of their three dart scores -/
def totalScore (scores : PlayerScores) : Nat :=
  (scores 0).val + (scores 1).val + (scores 2).val

/-- The scores for each player -/
def playerTotalScores : Player → Nat
| Player.Alice => 24
| Player.Ben => 13
| Player.Cindy => 19
| Player.Dave => 28
| Player.Ellen => 30

/-- Predicate to check if a player's scores contain a specific value -/
def containsScore (scores : PlayerScores) (n : DartScore) : Prop :=
  ∃ i, scores i = n

/-- Statement: Cindy is the only player who hit the region worth 9 points -/
theorem cindy_hit_nine :
  ∃! p : Player, ∃ scores : PlayerScores,
    totalScore scores = playerTotalScores p ∧
    containsScore scores ⟨9, by norm_num⟩ ∧
    p = Player.Cindy :=
by
  sorry

end cindy_hit_nine_l3293_329310


namespace circles_are_disjoint_l3293_329339

-- Define the circles
def C₁ : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}
def C₂ : Set (ℝ × ℝ) := {p | (p.1 - 2)^2 + (p.2 + 2)^2 = 1}

-- Define the centers and radii
def center₁ : ℝ × ℝ := (0, 0)
def center₂ : ℝ × ℝ := (2, -2)
def radius₁ : ℝ := 1
def radius₂ : ℝ := 1

-- Theorem statement
theorem circles_are_disjoint : 
  Real.sqrt ((center₂.1 - center₁.1)^2 + (center₂.2 - center₁.2)^2) > radius₁ + radius₂ := by
  sorry

end circles_are_disjoint_l3293_329339


namespace f_of_3_equals_11_f_equiv_l3293_329386

-- Define the function f
def f (t : ℝ) : ℝ := t^2 + 2

-- State the theorem
theorem f_of_3_equals_11 : f 3 = 11 := by
  sorry

-- Define the original function property
axiom f_property (x : ℝ) (hx : x ≠ 0) : f (x - 1/x) = x^2 + 1/x^2

-- Prove the equivalence of the two function definitions
theorem f_equiv (x : ℝ) (hx : x ≠ 0) : f (x - 1/x) = x^2 + 1/x^2 := by
  sorry

end f_of_3_equals_11_f_equiv_l3293_329386


namespace initial_shoe_pairs_l3293_329342

theorem initial_shoe_pairs (remaining_pairs : ℕ) (lost_shoes : ℕ) : 
  remaining_pairs = 19 → lost_shoes = 9 → 
  (2 * remaining_pairs + lost_shoes + 1) / 2 = 23 := by
sorry

end initial_shoe_pairs_l3293_329342


namespace cubeRoot_of_negative_27_l3293_329358

-- Define the cube root function
noncomputable def cubeRoot (x : ℝ) : ℝ := Real.rpow x (1/3)

-- State the theorem
theorem cubeRoot_of_negative_27 : cubeRoot (-27) = -3 := by
  sorry

end cubeRoot_of_negative_27_l3293_329358


namespace men_with_ac_at_least_12_l3293_329326

theorem men_with_ac_at_least_12 (total : ℕ) (married : ℕ) (tv : ℕ) (radio : ℕ) (all_four : ℕ) (ac : ℕ) :
  total = 100 →
  married = 82 →
  tv = 75 →
  radio = 85 →
  all_four = 12 →
  all_four ≤ ac →
  ac ≥ 12 :=
by sorry

end men_with_ac_at_least_12_l3293_329326


namespace tiles_per_row_l3293_329390

-- Define the area of the room in square feet
def room_area : ℝ := 324

-- Define the side length of a tile in inches
def tile_side : ℝ := 9

-- Define the conversion factor from feet to inches
def feet_to_inches : ℝ := 12

-- Theorem statement
theorem tiles_per_row : 
  ⌊(feet_to_inches * Real.sqrt room_area) / tile_side⌋ = 24 := by
  sorry

end tiles_per_row_l3293_329390


namespace hemisphere_on_cone_surface_area_l3293_329361

/-- The total surface area of a solid consisting of a hemisphere on top of a cone,
    where the area of the hemisphere's base is 144π and the height of the cone is
    twice the radius of the hemisphere. -/
theorem hemisphere_on_cone_surface_area :
  ∀ (r : ℝ),
  r > 0 →
  π * r^2 = 144 * π →
  let hemisphere_area := 2 * π * r^2
  let cone_height := 2 * r
  let cone_slant_height := Real.sqrt (r^2 + cone_height^2)
  let cone_area := π * r * cone_slant_height
  hemisphere_area + cone_area = 288 * π + 144 * Real.sqrt 5 * π :=
by
  sorry

end hemisphere_on_cone_surface_area_l3293_329361


namespace ladas_isosceles_triangle_l3293_329329

theorem ladas_isosceles_triangle 
  (α β γ : ℝ) 
  (triangle_sum : α + β + γ = 180)
  (positive_angles : 0 < α ∧ 0 < β ∧ 0 < γ)
  (sum_angles_exist : ∃ δ ε : ℝ, 0 < δ ∧ 0 < ε ∧ δ + ε ≤ 180 ∧ δ = α + β ∧ ε = α + γ) :
  β = γ := by
sorry

end ladas_isosceles_triangle_l3293_329329


namespace fraction_sum_inequality_l3293_329319

theorem fraction_sum_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^3 / (x^3 + 2*y^2*z)) + (y^3 / (y^3 + 2*z^2*x)) + (z^3 / (z^3 + 2*x^2*y)) ≥ 1 := by
  sorry

end fraction_sum_inequality_l3293_329319


namespace final_score_eq_initial_minus_one_l3293_329304

/-- A scoring system where 1 point is deducted for a missing answer -/
structure ScoringSystem where
  initial_score : ℝ
  missing_answer_deduction : ℝ := 1

/-- The final score after deducting for a missing answer -/
def final_score (s : ScoringSystem) : ℝ :=
  s.initial_score - s.missing_answer_deduction

/-- Theorem stating that the final score is equal to the initial score minus 1 -/
theorem final_score_eq_initial_minus_one (s : ScoringSystem) :
  final_score s = s.initial_score - 1 := by
  sorry

#check final_score_eq_initial_minus_one

end final_score_eq_initial_minus_one_l3293_329304


namespace complement_A_union_B_l3293_329366

def A : Set Int := {x | ∃ k, x = 3*k + 1}
def B : Set Int := {x | ∃ k, x = 3*k + 2}
def U : Set Int := Set.univ

theorem complement_A_union_B : 
  (U \ (A ∪ B)) = {x : Int | ∃ k, x = 3*k} := by sorry

end complement_A_union_B_l3293_329366


namespace nested_fraction_evaluation_l3293_329369

theorem nested_fraction_evaluation :
  1 / (2 - 1 / (2 - 1 / (2 - 1 / 3))) = 5 / 7 := by sorry

end nested_fraction_evaluation_l3293_329369


namespace theater_audience_l3293_329371

/-- Proves the number of children in the audience given theater conditions -/
theorem theater_audience (total_seats : ℕ) (adult_price child_price : ℚ) (total_income : ℚ) 
  (h_seats : total_seats = 200)
  (h_adult_price : adult_price = 3)
  (h_child_price : child_price = (3/2))
  (h_total_income : total_income = 510) :
  ∃ (adults children : ℕ), 
    adults + children = total_seats ∧ 
    adult_price * adults + child_price * children = total_income ∧
    children = 60 := by
  sorry

end theater_audience_l3293_329371


namespace total_money_l3293_329337

/-- Given three people A, B, and C with some money, prove their total amount. -/
theorem total_money (A B C : ℕ) 
  (hAC : A + C = 250)
  (hBC : B + C = 450)
  (hC : C = 100) : 
  A + B + C = 600 := by
  sorry

end total_money_l3293_329337


namespace small_pizza_price_is_two_l3293_329311

/-- The price of a small pizza given the conditions of the problem -/
def small_pizza_price (large_pizza_price : ℕ) (total_sales : ℕ) (small_pizzas_sold : ℕ) (large_pizzas_sold : ℕ) : ℕ :=
  (total_sales - large_pizza_price * large_pizzas_sold) / small_pizzas_sold

/-- Theorem stating that the price of a small pizza is $2 under the given conditions -/
theorem small_pizza_price_is_two :
  small_pizza_price 8 40 8 3 = 2 := by
  sorry

end small_pizza_price_is_two_l3293_329311


namespace election_winning_margin_l3293_329306

/-- Represents an election with two candidates -/
structure Election :=
  (total_votes : ℕ)
  (winner_votes : ℕ)
  (winner_percentage : ℚ)

/-- Calculates the number of votes the winner won by -/
def winning_margin (e : Election) : ℕ :=
  e.winner_votes - (e.total_votes - e.winner_votes)

/-- Theorem stating the winning margin for the given election scenario -/
theorem election_winning_margin :
  ∃ (e : Election),
    e.winner_percentage = 62 / 100 ∧
    e.winner_votes = 899 ∧
    winning_margin e = 348 := by
  sorry

end election_winning_margin_l3293_329306


namespace election_result_l3293_329388

/-- Represents the number of votes for each candidate in an election --/
structure ElectionResults where
  total_votes : Nat
  john_votes : Nat
  james_percentage : Rat
  john_votes_le_total : john_votes ≤ total_votes

/-- Calculates the difference in votes between the third candidate and John --/
def vote_difference (e : ElectionResults) : Int :=
  e.total_votes - e.john_votes - 
  Nat.floor (e.james_percentage * (e.total_votes - e.john_votes : Rat)) - e.john_votes

/-- Theorem stating the vote difference for the given election scenario --/
theorem election_result : 
  ∀ (e : ElectionResults), 
    e.total_votes = 1150 ∧ 
    e.john_votes = 150 ∧ 
    e.james_percentage = 7/10 → 
    vote_difference e = 150 := by
  sorry


end election_result_l3293_329388


namespace polynomial_simplification_l3293_329365

theorem polynomial_simplification (s : ℝ) : 
  (2 * s^2 + 5 * s - 3) - (2 * s^2 + 9 * s - 6) = -4 * s + 3 := by
  sorry

end polynomial_simplification_l3293_329365


namespace player_A_winning_strategy_l3293_329334

-- Define the game state
structure GameState where
  board : ℕ

-- Define the possible moves for player A
inductive MoveA where 
  | half : MoveA
  | quarter : MoveA
  | triple : MoveA

-- Define the possible moves for player B
inductive MoveB where
  | increment : MoveB
  | decrement : MoveB

-- Define the game step for player A
def stepA (state : GameState) (move : MoveA) : GameState :=
  match move with
  | MoveA.half => 
      if state.board % 2 = 0 then { board := state.board / 2 } else state
  | MoveA.quarter => 
      if state.board % 4 = 0 then { board := state.board / 4 } else state
  | MoveA.triple => { board := state.board * 3 }

-- Define the game step for player B
def stepB (state : GameState) (move : MoveB) : GameState :=
  match move with
  | MoveB.increment => { board := state.board + 1 }
  | MoveB.decrement => 
      if state.board > 1 then { board := state.board - 1 } else state

-- Define the winning condition
def isWinningState (state : GameState) : Prop :=
  state.board = 3

-- Theorem statement
theorem player_A_winning_strategy (n : ℕ) (h : n > 0) : 
  ∃ (strategy : ℕ → MoveA), 
    ∀ (player_B_moves : ℕ → MoveB),
      ∃ (k : ℕ), isWinningState (
        (stepB (stepA { board := n } (strategy 0)) (player_B_moves 0))
      ) ∨ 
      isWinningState (
        (List.foldl 
          (λ state i => stepB (stepA state (strategy i)) (player_B_moves i))
          { board := n }
          (List.range k)
        )
      ) := by
  sorry

end player_A_winning_strategy_l3293_329334


namespace sum_of_three_numbers_l3293_329391

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 560) 
  (h2 : a*b + b*c + c*a = 8) : 
  a + b + c = 24 := by
sorry

end sum_of_three_numbers_l3293_329391


namespace brookes_initial_balloons_l3293_329348

theorem brookes_initial_balloons :
  ∀ (b : ℕ), -- Brooke's initial number of balloons
  let brooke_final := b + 8 -- Brooke's final number of balloons
  let tracy_initial := 6 -- Tracy's initial number of balloons
  let tracy_added := 24 -- Number of balloons Tracy adds
  let tracy_final := (tracy_initial + tracy_added) / 2 -- Tracy's final number of balloons after popping half
  brooke_final + tracy_final = 35 → b = 12 :=
by
  sorry

#check brookes_initial_balloons

end brookes_initial_balloons_l3293_329348


namespace min_values_theorem_l3293_329368

theorem min_values_theorem (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : m + n = 2 * m * n) :
  (m + n ≥ 2) ∧ (Real.sqrt (m * n) ≥ 1) ∧ (n^2 / m + m^2 / n ≥ 2) := by
  sorry

end min_values_theorem_l3293_329368


namespace square_of_105_l3293_329381

theorem square_of_105 : (105 : ℕ)^2 = 11025 := by
  sorry

end square_of_105_l3293_329381


namespace minimum_value_of_x_l3293_329331

theorem minimum_value_of_x (x : ℝ) 
  (h_pos : x > 0) 
  (h_log : Real.log x ≥ Real.log 3 + Real.log (Real.sqrt x)) : 
  x ≥ 9 := by
  sorry

end minimum_value_of_x_l3293_329331


namespace reciprocal_comparison_l3293_329393

theorem reciprocal_comparison : ∃ (S : Set ℝ), 
  S = {-3, -1/2, 0.5, 1, 3} ∧ 
  (∀ x ∈ S, x < 1 / x ↔ (x = -3 ∨ x = 0.5)) := by
  sorry

end reciprocal_comparison_l3293_329393


namespace restaurant_bill_l3293_329392

theorem restaurant_bill (num_friends : ℕ) (extra_payment : ℚ) (total_bill : ℚ) :
  num_friends = 8 →
  extra_payment = 5/2 →
  (num_friends - 1) * (total_bill / num_friends + extra_payment) = total_bill →
  total_bill = 140 := by
  sorry

end restaurant_bill_l3293_329392


namespace twelfth_even_multiple_of_5_l3293_329376

-- Define a function that represents the nth positive integer that is both even and a multiple of 5
def evenMultipleOf5 (n : ℕ) : ℕ := 10 * n

-- State the theorem
theorem twelfth_even_multiple_of_5 : evenMultipleOf5 12 = 120 := by sorry

end twelfth_even_multiple_of_5_l3293_329376


namespace equation_solution_l3293_329357

theorem equation_solution : 
  ∀ x : ℝ, x ≠ -3 → 
  ((7 * x^2 - 3) / (x + 3) - 3 / (x + 3) = 1 / (x + 3)) ↔ 
  (x = 1 ∨ x = -1) :=
by sorry

end equation_solution_l3293_329357


namespace least_integer_with_deletion_property_l3293_329303

theorem least_integer_with_deletion_property : ∃ (x : ℕ), 
  (x > 0) ∧ 
  (x = 950) ∧ 
  (∀ y : ℕ, y > 0 ∧ y < x → ¬(y / 10 = y / 19)) ∧
  (x / 10 = x / 19) := by
  sorry

end least_integer_with_deletion_property_l3293_329303


namespace sixth_term_of_sequence_l3293_329355

-- Define an arithmetic sequence
def arithmetic_sequence (a : ℝ) (d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d

-- State the theorem
theorem sixth_term_of_sequence (a d : ℝ) :
  arithmetic_sequence a d 2 = 14 ∧
  arithmetic_sequence a d 4 = 32 →
  arithmetic_sequence a d 6 = 50 := by
  sorry


end sixth_term_of_sequence_l3293_329355


namespace geometric_sequence_property_l3293_329336

theorem geometric_sequence_property (q : ℝ) (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, a n = 4 * q^(n-1)) →  -- a_n is a geometric sequence with a_1 = 4 and common ratio q
  (∀ n, S n = 4 * (1 - q^n) / (1 - q)) →  -- S_n is the sum of the first n terms
  (∃ r, ∀ n, (S (n+1) + 2) = r * (S n + 2)) →  -- {S_n + 2} is a geometric sequence
  q = 3 := by
sorry

end geometric_sequence_property_l3293_329336


namespace correct_ranking_l3293_329332

-- Define the cities
inductive City
| Dover
| Eden
| Fairview

-- Define the growth rate comparison relation
def higherGrowthRate : City → City → Prop := sorry

-- Define the statements
def statement1 : Prop := higherGrowthRate City.Dover City.Eden ∧ higherGrowthRate City.Dover City.Fairview
def statement2 : Prop := ¬(higherGrowthRate City.Eden City.Dover ∧ higherGrowthRate City.Eden City.Fairview)
def statement3 : Prop := ¬(higherGrowthRate City.Dover City.Fairview ∧ higherGrowthRate City.Eden City.Fairview)

-- Theorem stating the correct ranking
theorem correct_ranking :
  (statement1 ∨ statement2 ∨ statement3) ∧
  (statement1 → ¬statement2 ∧ ¬statement3) ∧
  (statement2 → ¬statement1 ∧ ¬statement3) ∧
  (statement3 → ¬statement1 ∧ ¬statement2) →
  higherGrowthRate City.Eden City.Dover ∧
  higherGrowthRate City.Dover City.Fairview :=
sorry

end correct_ranking_l3293_329332


namespace sum_of_other_x_coordinates_l3293_329394

/-- A rectangle in a 2D plane --/
structure Rectangle where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ
  v4 : ℝ × ℝ

/-- The property that two points are opposite vertices of a rectangle --/
def are_opposite_vertices (p1 p2 : ℝ × ℝ) (r : Rectangle) : Prop :=
  (r.v1 = p1 ∧ r.v3 = p2) ∨ (r.v1 = p2 ∧ r.v3 = p1) ∨
  (r.v2 = p1 ∧ r.v4 = p2) ∨ (r.v2 = p2 ∧ r.v4 = p1)

/-- The theorem to be proved --/
theorem sum_of_other_x_coordinates (r : Rectangle) :
  are_opposite_vertices (2, 12) (8, 3) r →
  (r.v1.1 + r.v2.1 + r.v3.1 + r.v4.1) - (2 + 8) = 10 := by
  sorry


end sum_of_other_x_coordinates_l3293_329394


namespace monotonic_increasing_condition_monotonic_decreasing_interval_condition_not_always_above_line_l3293_329346

-- Define the function f(x) = x^3 - ax - 1
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x - 1

-- Statement 1
theorem monotonic_increasing_condition (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) → a ≤ 0 :=
sorry

-- Statement 2
theorem monotonic_decreasing_interval_condition (a : ℝ) :
  (∀ x y : ℝ, -1 < x ∧ x < y ∧ y < 1 → f a x > f a y) → a ≥ 3 :=
sorry

-- Statement 3
theorem not_always_above_line (a : ℝ) :
  ∃ x : ℝ, f a x < a :=
sorry

end monotonic_increasing_condition_monotonic_decreasing_interval_condition_not_always_above_line_l3293_329346


namespace expected_value_of_strategic_die_rolling_l3293_329338

/-- Represents a 6-sided die -/
def Die := Fin 6

/-- The strategy for re-rolling -/
def rerollStrategy (roll : Die) : Bool :=
  roll.val < 4

/-- The expected value of a single roll of a 6-sided die -/
def singleRollExpectedValue : ℚ := 7/2

/-- The expected value after applying the re-roll strategy once -/
def strategicRollExpectedValue : ℚ := 17/4

/-- The final expected value after up to two re-rolls -/
def finalExpectedValue : ℚ := 17/4

theorem expected_value_of_strategic_die_rolling :
  finalExpectedValue = 17/4 := by sorry

end expected_value_of_strategic_die_rolling_l3293_329338


namespace isosceles_right_triangle_leg_length_l3293_329335

/-- Proves that in an isosceles right triangle with a hypotenuse of length 8.485281374238571, the length of one leg is 6. -/
theorem isosceles_right_triangle_leg_length : 
  ∀ (a : ℝ), 
    (a > 0) →  -- Ensure positive length
    (a * Real.sqrt 2 = 8.485281374238571) →  -- Hypotenuse length condition
    (a = 6) := by
  sorry

end isosceles_right_triangle_leg_length_l3293_329335


namespace striped_shirts_difference_l3293_329377

theorem striped_shirts_difference (total : ℕ) (striped_ratio : ℚ) (checkered_ratio : ℚ) 
  (h_total : total = 120)
  (h_striped : striped_ratio = 3/5)
  (h_checkered : checkered_ratio = 1/4)
  (h_shorts_plain : ∃ (plain : ℕ) (shorts : ℕ), 
    plain = total - (striped_ratio * total).num - (checkered_ratio * total).num ∧
    shorts + 10 = plain) :
  ∃ (striped : ℕ) (shorts : ℕ),
    striped = (striped_ratio * total).num ∧
    striped - shorts = 44 :=
sorry

end striped_shirts_difference_l3293_329377


namespace same_terminal_side_as_60_degrees_l3293_329301

def has_same_terminal_side (α : ℤ) : Prop :=
  ∃ k : ℤ, α = k * 360 + 60

theorem same_terminal_side_as_60_degrees :
  has_same_terminal_side (-300) ∧
  ¬has_same_terminal_side (-60) ∧
  ¬has_same_terminal_side 600 ∧
  ¬has_same_terminal_side 1380 :=
by sorry

end same_terminal_side_as_60_degrees_l3293_329301


namespace symmetric_difference_of_A_and_B_l3293_329300

-- Define sets A and B
def A : Set ℝ := {y | ∃ x, 0 ≤ x ∧ x ≤ 3 ∧ y = (x - 1)^2 + 1}
def B : Set ℝ := {y | ∃ x, 1 ≤ x ∧ x ≤ 3 ∧ y = x^2 + 1}

-- Define set difference
def setDifference (X Y : Set ℝ) : Set ℝ := {x | x ∈ X ∧ x ∉ Y}

-- Define symmetric difference
def symmetricDifference (X Y : Set ℝ) : Set ℝ := 
  (setDifference X Y) ∪ (setDifference Y X)

-- Theorem statement
theorem symmetric_difference_of_A_and_B :
  symmetricDifference A B = {y | (1 ≤ y ∧ y < 2) ∨ (5 < y ∧ y ≤ 10)} := by
  sorry

end symmetric_difference_of_A_and_B_l3293_329300


namespace dvd_book_capacity_l3293_329343

theorem dvd_book_capacity (total_capacity : ℕ) (current_dvds : ℕ) (h1 : total_capacity = 126) (h2 : current_dvds = 81) :
  total_capacity - current_dvds = 45 := by
  sorry

end dvd_book_capacity_l3293_329343


namespace constant_value_l3293_329344

noncomputable section

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define points
def P : ℝ × ℝ := (-2, 0)
def Q : ℝ × ℝ := (-2, -1)

-- Define the line l
def line_l (n : ℝ) (x y : ℝ) : Prop := x = n * (y + 1) - 2

-- Define the intersection points A and B
def A (n : ℝ) : ℝ × ℝ := sorry
def B (n : ℝ) : ℝ × ℝ := sorry

-- Define points C and D
def C (n : ℝ) : ℝ × ℝ := sorry
def D (n : ℝ) : ℝ × ℝ := sorry

-- Define the distances |QC| and |QD|
def QC (n : ℝ) : ℝ := sorry
def QD (n : ℝ) : ℝ := sorry

-- The main theorem
theorem constant_value (n : ℝ) :
  ellipse (A n).1 (A n).2 ∧ 
  ellipse (B n).1 (B n).2 ∧ 
  (A n).2 < 0 ∧ 
  (B n).2 < 0 ∧
  line_l n (A n).1 (A n).2 ∧
  line_l n (B n).1 (B n).2 →
  QC n + QD n - QC n * QD n = 0 :=
sorry

end

end constant_value_l3293_329344


namespace sum_digits_of_valid_hex_count_l3293_329316

/-- Represents a hexadecimal digit -/
inductive HexDigit
| Numeric (n : Fin 10)
| Alpha (a : Fin 6)

/-- Represents a hexadecimal number -/
def HexNumber := List HexDigit

/-- Converts a natural number to hexadecimal representation -/
def toHex (n : ℕ) : HexNumber :=
  sorry

/-- Checks if a hexadecimal number contains only numeric digits and doesn't start with 0 -/
def isValidHex (h : HexNumber) : Bool :=
  sorry

/-- Counts valid hexadecimal numbers in the first n positive integers -/
def countValidHex (n : ℕ) : ℕ :=
  sorry

/-- Sums the digits of a natural number -/
def sumDigits (n : ℕ) : ℕ :=
  sorry

/-- The main theorem to prove -/
theorem sum_digits_of_valid_hex_count :
  sumDigits (countValidHex 2000) = 7 :=
sorry

end sum_digits_of_valid_hex_count_l3293_329316


namespace journey_time_change_l3293_329313

/-- Given a journey that takes 5 hours at 80 miles per hour, prove that the same journey at 50 miles per hour will take 8 hours. -/
theorem journey_time_change (initial_time initial_speed new_speed : ℝ) 
  (h1 : initial_time = 5)
  (h2 : initial_speed = 80)
  (h3 : new_speed = 50) :
  (initial_time * initial_speed) / new_speed = 8 := by
  sorry

end journey_time_change_l3293_329313


namespace correct_problem_percentage_l3293_329312

/-- Given a total number of problems and the number of missed problems,
    calculate the percentage of correctly solved problems. -/
theorem correct_problem_percentage
  (x : ℕ) -- x represents the number of missed problems
  (h : x > 0) -- ensure x is positive to avoid division by zero
  : (((7 : ℚ) * x - x) / (7 * x)) * 100 = (6 : ℚ) / 7 * 100 := by
  sorry

#eval (6 : ℚ) / 7 * 100 -- To show the approximate result

end correct_problem_percentage_l3293_329312


namespace expression_equals_negative_one_l3293_329354

theorem expression_equals_negative_one :
  -5 * (2/3) + 6 * (2/7) + (1/3) * (-5) - (2/7) * (-8) = -1 := by
  sorry

end expression_equals_negative_one_l3293_329354


namespace khali_snow_volume_l3293_329307

/-- The volume of snow on a rectangular sidewalk -/
def snow_volume (length width depth : ℝ) : ℝ := length * width * depth

/-- Theorem: The volume of snow on Khali's sidewalk is 20 cubic feet -/
theorem khali_snow_volume :
  snow_volume 20 2 (1/2) = 20 := by
  sorry

end khali_snow_volume_l3293_329307


namespace unreasonable_milk_volume_l3293_329302

/-- Represents the volume of milk in liters --/
def milk_volume : ℝ := 250

/-- Represents a reasonable maximum volume of milk a person can drink in a day (in liters) --/
def max_reasonable_volume : ℝ := 10

/-- Theorem stating that the given milk volume is unreasonable for a person to drink in a day --/
theorem unreasonable_milk_volume : milk_volume > max_reasonable_volume := by
  sorry

end unreasonable_milk_volume_l3293_329302


namespace train_length_calculation_l3293_329384

/-- The length of a train given specific conditions -/
theorem train_length_calculation (crossing_time : Real) (man_speed : Real) (train_speed : Real) :
  let relative_speed := (train_speed - man_speed) * (5 / 18)
  let train_length := relative_speed * crossing_time
  crossing_time = 35.99712023038157 ∧ 
  man_speed = 3 ∧ 
  train_speed = 63 →
  ∃ ε > 0, |train_length - 600| < ε :=
by
  sorry

end train_length_calculation_l3293_329384


namespace sum_of_a_and_b_is_negative_one_l3293_329345

theorem sum_of_a_and_b_is_negative_one :
  ∀ (a b : ℝ) (S T : ℕ → ℝ),
  (∀ n, S n = 2^n + a) →  -- Sum of geometric sequence
  (∀ n, T n = n^2 - 2*n + b) →  -- Sum of arithmetic sequence
  a + b = -1 :=
by sorry

end sum_of_a_and_b_is_negative_one_l3293_329345


namespace multiply_by_eleven_l3293_329398

theorem multiply_by_eleven (x : ℝ) : 11 * x = 103.95 → x = 9.45 := by
  sorry

end multiply_by_eleven_l3293_329398


namespace two_times_three_plus_two_l3293_329395

theorem two_times_three_plus_two :
  (2 : ℕ) * 3 + 2 = 8 :=
by
  sorry

end two_times_three_plus_two_l3293_329395


namespace parallelogram_angle_equality_l3293_329323

-- Define the points
variable (A B C D P : Point)

-- Define the parallelogram property
def is_parallelogram (A B C D : Point) : Prop := sorry

-- Define the angle equality
def angle_eq (P Q R S T U : Point) : Prop := sorry

-- State the theorem
theorem parallelogram_angle_equality 
  (h_parallelogram : is_parallelogram A B C D)
  (h_angle_eq : angle_eq P A D P C D) :
  angle_eq P B C P D C := by sorry

end parallelogram_angle_equality_l3293_329323


namespace at_least_one_less_than_two_l3293_329340

theorem at_least_one_less_than_two (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y > 2) :
  min ((1 + x) / y) ((1 + y) / x) < 2 := by
sorry

end at_least_one_less_than_two_l3293_329340


namespace sin_plus_two_cos_alpha_l3293_329309

/-- Given a > 0 and the terminal side of angle α passes through point P(-3a, 4a),
    prove that sin α + 2cos α = -2/5 -/
theorem sin_plus_two_cos_alpha (a : ℝ) (α : ℝ) (h1 : a > 0) 
    (h2 : ∃ (t : ℝ), t > 0 ∧ -3 * a = t * Real.cos α ∧ 4 * a = t * Real.sin α) : 
    Real.sin α + 2 * Real.cos α = -2/5 := by
  sorry

end sin_plus_two_cos_alpha_l3293_329309


namespace solution_set_inequality_l3293_329364

theorem solution_set_inequality (x : ℝ) : 
  (Set.Icc (-3 : ℝ) 6 : Set ℝ) = {x | (x + 3) * (6 - x) ≥ 0} := by sorry

end solution_set_inequality_l3293_329364


namespace free_throw_contest_l3293_329385

theorem free_throw_contest (x : ℕ) : 
  x + 3*x + 6*x = 80 → x = 8 := by sorry

end free_throw_contest_l3293_329385


namespace quadratic_inequality_empty_solution_l3293_329352

theorem quadratic_inequality_empty_solution (a : ℝ) : 
  (∀ x : ℝ, 2 * x^2 + a * x + 2 ≥ 0) → -4 ≤ a ∧ a ≤ 4 := by
  sorry

end quadratic_inequality_empty_solution_l3293_329352


namespace local_face_value_diff_problem_l3293_329389

/-- The difference between the local value and face value of a digit in a number -/
def local_face_value_diff (digit : ℕ) (position : ℕ) : ℕ :=
  digit * (10 ^ position) - digit

theorem local_face_value_diff_problem : 
  local_face_value_diff 3 3 - local_face_value_diff 7 1 = 2934 := by
  sorry

end local_face_value_diff_problem_l3293_329389


namespace min_value_reciprocal_sum_l3293_329305

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  1 / x + 4 / y ≥ 9 ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 1 ∧ 1 / x + 4 / y = 9 := by
  sorry

end min_value_reciprocal_sum_l3293_329305


namespace linear_function_quadrants_l3293_329367

/-- A linear function f(x) = mx + b passes through a quadrant if there exists a point (x, f(x)) in that quadrant. -/
def passes_through_quadrant (m b : ℝ) (quad : ℕ) : Prop :=
  ∃ x y : ℝ, y = m * x + b ∧
  match quad with
  | 1 => x > 0 ∧ y > 0
  | 2 => x < 0 ∧ y > 0
  | 3 => x < 0 ∧ y < 0
  | 4 => x > 0 ∧ y < 0
  | _ => False

/-- The slope of the linear function -/
def m : ℝ := -5

/-- The y-intercept of the linear function -/
def b : ℝ := 3

/-- Theorem stating that the linear function f(x) = -5x + 3 passes through Quadrants I, II, and IV -/
theorem linear_function_quadrants :
  passes_through_quadrant m b 1 ∧
  passes_through_quadrant m b 2 ∧
  passes_through_quadrant m b 4 ∧
  ¬passes_through_quadrant m b 3 :=
sorry

end linear_function_quadrants_l3293_329367


namespace rectangle_longer_side_length_l3293_329327

/-- Given a circle with radius 6 cm tangent to three sides of a rectangle,
    and the area of the rectangle being three times the area of the circle,
    the length of the longer side of the rectangle is 9π cm. -/
theorem rectangle_longer_side_length (circle_radius : ℝ) (rectangle_area : ℝ) (circle_area : ℝ)
  (h1 : circle_radius = 6)
  (h2 : rectangle_area = 3 * circle_area)
  (h3 : circle_area = Real.pi * circle_radius ^ 2)
  (h4 : rectangle_area = 12 * longer_side) :
  longer_side = 9 * Real.pi := by
  sorry

end rectangle_longer_side_length_l3293_329327


namespace range_of_a_l3293_329372

theorem range_of_a (p q : ℝ → Prop) :
  (∀ x, q x → p x) ∧
  (∃ x, p x ∧ ¬q x) ∧
  (∀ x, q x ↔ -x^2 + 5*x - 6 > 0) ∧
  (∀ x a, p x ↔ |x - a| < 4) →
  ∃ a_min a_max, a_min = -1 ∧ a_max = 6 ∧ ∀ a, (a_min < a ∧ a < a_max) → 
    (∃ x, p x) ∧ (∃ x, ¬p x) := by sorry


end range_of_a_l3293_329372


namespace regions_in_circle_l3293_329387

/-- The number of regions created by radii and concentric circles in a circle -/
def num_regions (num_radii : ℕ) (num_concentric_circles : ℕ) : ℕ :=
  (num_concentric_circles + 1) * num_radii

/-- Theorem stating that 16 radii and 10 concentric circles create 176 regions -/
theorem regions_in_circle (r : ℕ) (c : ℕ) 
  (h1 : r = 16) (h2 : c = 10) : 
  num_regions r c = 176 := by
  sorry

end regions_in_circle_l3293_329387


namespace sundae_cost_theorem_l3293_329318

def sundae_cost (monday_sundaes tuesday_sundaes : ℕ)
  (monday_mms monday_gummies monday_marshmallows : ℕ)
  (tuesday_mms tuesday_gummies tuesday_marshmallows : ℕ)
  (mms_per_pack gummies_per_pack marshmallows_per_pack : ℕ)
  (mms_pack_cost gummies_pack_cost marshmallows_pack_cost : ℚ) : ℚ :=
  let total_mms := monday_sundaes * monday_mms + tuesday_sundaes * tuesday_mms
  let total_gummies := monday_sundaes * monday_gummies + tuesday_sundaes * tuesday_gummies
  let total_marshmallows := monday_sundaes * monday_marshmallows + tuesday_sundaes * tuesday_marshmallows
  let mms_packs := (total_mms + mms_per_pack - 1) / mms_per_pack
  let gummies_packs := (total_gummies + gummies_per_pack - 1) / gummies_per_pack
  let marshmallows_packs := (total_marshmallows + marshmallows_per_pack - 1) / marshmallows_per_pack
  mms_packs * mms_pack_cost + gummies_packs * gummies_pack_cost + marshmallows_packs * marshmallows_pack_cost

theorem sundae_cost_theorem :
  sundae_cost 40 20 6 4 8 10 5 12 40 30 50 2 (3/2) 1 = 95/2 :=
by sorry

end sundae_cost_theorem_l3293_329318


namespace inscribed_circle_radius_l3293_329374

/-- The radius of the inscribed circle in a triangle with sides 9, 10, and 11 is 2√2 -/
theorem inscribed_circle_radius (a b c : ℝ) (h_a : a = 9) (h_b : b = 10) (h_c : c = 11) :
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  area / s = 2 * Real.sqrt 2 := by sorry

end inscribed_circle_radius_l3293_329374


namespace quadratic_behavior_l3293_329396

-- Define the quadratic function
def f (x : ℝ) : ℝ := -x^2 + 6*x - 5

-- Define the derivative of f
def f' (x : ℝ) : ℝ := -2*x + 6

-- Theorem statement
theorem quadratic_behavior (x : ℝ) : x > 5 → f x < 0 ∧ f' x < 0 := by
  sorry

end quadratic_behavior_l3293_329396


namespace logical_equivalence_l3293_329321

theorem logical_equivalence (R S T : Prop) :
  (R → ¬S ∧ ¬T) ↔ (S ∨ T → ¬R) := by sorry

end logical_equivalence_l3293_329321


namespace steve_markers_l3293_329341

/-- Given the number of markers for Alia, Austin, and Steve, 
    where Alia has 2 times as many markers as Austin, 
    Austin has one-third as many markers as Steve, 
    and Alia has 40 markers, prove that Steve has 60 markers. -/
theorem steve_markers (alia austin steve : ℕ) 
  (h1 : alia = 2 * austin) 
  (h2 : austin = steve / 3)
  (h3 : alia = 40) : 
  steve = 60 := by sorry

end steve_markers_l3293_329341


namespace two_zeros_cubic_l3293_329322

def f (c : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + c

theorem two_zeros_cubic (c : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ f c x = 0 ∧ f c y = 0 ∧ ∀ z : ℝ, f c z = 0 → z = x ∨ z = y) ↔ c = -2 ∨ c = 2 := by
  sorry

end two_zeros_cubic_l3293_329322


namespace complement_A_intersect_B_l3293_329315

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {y | ∃ x, y = 2^x + 1}

-- Define set B
def B : Set ℝ := {x | Real.log x < 0}

-- Statement to prove
theorem complement_A_intersect_B :
  (Set.compl A) ∩ B = {x | x < 1} := by sorry

end complement_A_intersect_B_l3293_329315


namespace divisibility_by_five_l3293_329397

theorem divisibility_by_five (x y : ℕ+) (h1 : 2 * x ^ 2 - 1 = y ^ 15) (h2 : x > 1) :
  5 ∣ x.val :=
sorry

end divisibility_by_five_l3293_329397


namespace percentage_difference_l3293_329356

theorem percentage_difference (x y : ℝ) (h : x = 11 * y) :
  (x - y) / x * 100 = 10 / 11 * 100 := by
  sorry

end percentage_difference_l3293_329356


namespace pentagon_area_in_16_sided_polygon_l3293_329378

/-- Represents a regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  sideLength : ℝ

/-- Represents a pentagon in a regular polygon -/
structure Pentagon (n : ℕ) where
  polygon : RegularPolygon n
  vertices : Fin 5 → Fin n

/-- Calculates the area of a pentagon in a regular polygon -/
def pentagonArea (n : ℕ) (p : Pentagon n) : ℝ := sorry

theorem pentagon_area_in_16_sided_polygon :
  ∀ (p : Pentagon 16),
    p.polygon.sideLength = 3 →
    (∀ i : Fin 5, (p.vertices i + 4) % 16 = p.vertices ((i + 1) % 5)) →
    pentagonArea 16 p = 198 := by
  sorry

end pentagon_area_in_16_sided_polygon_l3293_329378


namespace octal_123_equals_decimal_83_l3293_329359

-- Define a function to convert octal to decimal
def octal_to_decimal (octal : List Nat) : Nat :=
  octal.enum.foldl (fun acc (i, digit) => acc + digit * (8 ^ i)) 0

-- Theorem statement
theorem octal_123_equals_decimal_83 :
  octal_to_decimal [3, 2, 1] = 83 := by
  sorry

end octal_123_equals_decimal_83_l3293_329359


namespace coffee_buyers_fraction_l3293_329314

theorem coffee_buyers_fraction (total : ℕ) (non_coffee : ℕ) 
  (h1 : total = 25) (h2 : non_coffee = 10) : 
  (total - non_coffee : ℚ) / total = 3 / 5 := by
  sorry

end coffee_buyers_fraction_l3293_329314


namespace sum_lent_proof_l3293_329350

/-- Proves that given a sum P lent at 4% per annum simple interest,
    if the interest after 4 years is Rs. 1260 less than P, then P = 1500. -/
theorem sum_lent_proof (P : ℝ) : 
  (P * (4 / 100) * 4 = P - 1260) → P = 1500 := by sorry

end sum_lent_proof_l3293_329350


namespace largest_n_multiple_of_5_l3293_329351

def is_multiple_of_5 (n : ℕ) : Prop :=
  ∃ k : ℤ, 7 * (n - 3)^7 - 2 * n^3 + 21 * n - 36 = 5 * k

theorem largest_n_multiple_of_5 :
  ∀ n : ℕ, n < 100000 → is_multiple_of_5 n → n ≤ 99998 ∧
  is_multiple_of_5 99998 ∧
  99998 < 100000 :=
sorry

end largest_n_multiple_of_5_l3293_329351


namespace ab_equals_one_l3293_329328

theorem ab_equals_one (θ : ℝ) (a b : ℝ) 
  (h1 : a * Real.sin θ + Real.cos θ = 1)
  (h2 : b * Real.sin θ - Real.cos θ = 1) :
  a * b = 1 := by
  sorry

end ab_equals_one_l3293_329328


namespace min_y_difference_parabola_l3293_329347

/-- Parabola structure -/
structure Parabola where
  a : ℝ
  vertex : ℝ × ℝ

/-- Point on a parabola -/
structure ParabolaPoint (p : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = 4 * p.a * (x - p.vertex.1)

/-- Line passing through two points -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Theorem: Minimum value of |y₁ - 4y₂| for points on parabola y² = 4x -/
theorem min_y_difference_parabola (p : Parabola) 
  (h_p : p.a = 1 ∧ p.vertex = (0, 0))
  (l : Line) 
  (h_l : l.a * 1 + l.b * 0 + l.c = 0)  -- Line passes through focus (1, 0)
  (A B : ParabolaPoint p)
  (h_A : A.x ≥ 0 ∧ A.y ≥ 0)  -- A is in the first quadrant
  (h_line : l.a * A.x + l.b * A.y + l.c = 0 ∧ 
            l.a * B.x + l.b * B.y + l.c = 0)  -- A and B are on the line
  : ∃ (y₁ y₂ : ℝ), |y₁ - 4*y₂| ≥ 8 ∧ 
    (∃ (A' B' : ParabolaPoint p), 
      |A'.y - 4*B'.y| = 8 ∧ 
      l.a * A'.x + l.b * A'.y + l.c = 0 ∧ 
      l.a * B'.x + l.b * B'.y + l.c = 0) := by
  sorry

end min_y_difference_parabola_l3293_329347


namespace midpoint_product_zero_l3293_329382

/-- Given that C = (4, 3) is the midpoint of line segment AB where A = (2, 6) and B = (x, y), prove that xy = 0 -/
theorem midpoint_product_zero (x y : ℝ) : 
  (4 : ℝ) = (2 + x) / 2 → 
  (3 : ℝ) = (6 + y) / 2 → 
  x * y = 0 := by
  sorry

end midpoint_product_zero_l3293_329382


namespace fixed_point_of_exponential_function_l3293_329324

/-- Given a > 0 and a ≠ 1, prove that (-2, 2) is a fixed point of f(x) = a^(x+2) + 1 -/
theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (ha_ne_one : a ≠ 1) :
  let f : ℝ → ℝ := λ x => a^(x + 2) + 1
  f (-2) = 2 := by sorry

end fixed_point_of_exponential_function_l3293_329324


namespace coin_distribution_l3293_329317

theorem coin_distribution (total : ℕ) (ways : ℕ) 
  (h_total : total = 1512)
  (h_ways : ways = 1512)
  (h_denominations : ∃ (c₂ c₅ c₁₀ c₂₀ c₅₀ c₁₀₀ c₂₀₀ : ℕ),
    (c₂ ≥ 1 ∧ c₅ ≥ 1 ∧ c₁₀ ≥ 1 ∧ c₂₀ ≥ 1 ∧ c₅₀ ≥ 1 ∧ c₁₀₀ ≥ 1 ∧ c₂₀₀ ≥ 1) ∧
    (2 * c₂ + 5 * c₅ + 10 * c₁₀ + 20 * c₂₀ + 50 * c₅₀ + 100 * c₁₀₀ + 200 * c₂₀₀ = total) ∧
    ((c₂ + 1) * (c₅ + 1) * (c₁₀ + 1) * (c₂₀ + 1) * (c₅₀ + 1) * (c₁₀₀ + 1) * (c₂₀₀ + 1) = ways)) :
  ∃! (c₂ c₅ c₁₀ c₂₀ c₅₀ c₁₀₀ c₂₀₀ : ℕ),
    (c₂ = 1 ∧ c₅ = 2 ∧ c₁₀ = 1 ∧ c₂₀ = 2 ∧ c₅₀ = 1 ∧ c₁₀₀ = 2 ∧ c₂₀₀ = 6) ∧
    (2 * c₂ + 5 * c₅ + 10 * c₁₀ + 20 * c₂₀ + 50 * c₅₀ + 100 * c₁₀₀ + 200 * c₂₀₀ = total) ∧
    ((c₂ + 1) * (c₅ + 1) * (c₁₀ + 1) * (c₂₀ + 1) * (c₅₀ + 1) * (c₁₀₀ + 1) * (c₂₀₀ + 1) = ways) :=
by sorry

end coin_distribution_l3293_329317


namespace consecutive_odd_integers_sum_l3293_329333

theorem consecutive_odd_integers_sum (a b c : ℤ) : 
  (∃ k : ℤ, a = 2*k + 1) →  -- a is odd
  b = a + 2 →               -- b is the next consecutive odd integer
  c = b + 2 →               -- c is the next consecutive odd integer after b
  a + c = 150 →             -- sum of first and third is 150
  a + b + c = 225 :=        -- sum of all three is 225
by sorry

end consecutive_odd_integers_sum_l3293_329333


namespace power_of_power_l3293_329399

theorem power_of_power (a : ℝ) : (a^3)^2 = a^6 := by
  sorry

end power_of_power_l3293_329399


namespace inequality_proof_l3293_329360

theorem inequality_proof (a b : ℝ) (n : ℕ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : 1/a + 1/b = 1) :
  (a + b)^n - a^n - b^n ≥ 2^(2*n) - 2^(n+1) := by
  sorry

end inequality_proof_l3293_329360


namespace matrix_sum_proof_l3293_329308

theorem matrix_sum_proof : 
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![4, -3; 2, 5]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![-6, 8; -3, 7]
  A + B = !![-2, 5; -1, 12] := by
  sorry

end matrix_sum_proof_l3293_329308


namespace jelly_beans_distribution_l3293_329362

theorem jelly_beans_distribution (initial_beans : ℕ) (remaining_beans : ℕ) 
  (h1 : initial_beans = 8000)
  (h2 : remaining_beans = 1600) :
  ∃ (x : ℕ), 
    x = 400 ∧ 
    initial_beans - remaining_beans = 6 * (2 * x) + 4 * x :=
by sorry

end jelly_beans_distribution_l3293_329362
