import Mathlib

namespace polyhedron_exists_l3346_334660

/-- A vertex of the polyhedron -/
inductive Vertex : Type
| A | B | C | D | E | F | G | H

/-- An edge of the polyhedron -/
inductive Edge : Type
| AB | AC | AH | BC | BD | CD | DE | EF | EG | FG | FH | GH

/-- A polyhedron structure -/
structure Polyhedron :=
  (vertices : List Vertex)
  (edges : List Edge)

/-- The specific polyhedron we're interested in -/
def specificPolyhedron : Polyhedron :=
  { vertices := [Vertex.A, Vertex.B, Vertex.C, Vertex.D, Vertex.E, Vertex.F, Vertex.G, Vertex.H],
    edges := [Edge.AB, Edge.AC, Edge.AH, Edge.BC, Edge.BD, Edge.CD, Edge.DE, Edge.EF, Edge.EG, Edge.FG, Edge.FH, Edge.GH] }

/-- Theorem stating the existence of the polyhedron -/
theorem polyhedron_exists : ∃ (p : Polyhedron), p = specificPolyhedron :=
sorry

end polyhedron_exists_l3346_334660


namespace second_person_work_time_l3346_334652

/-- Given two persons who can finish a job in 8 days, where the first person alone can finish the job in 24 days, prove that the second person alone will take 12 days to finish the job. -/
theorem second_person_work_time (total_time : ℝ) (first_person_time : ℝ) (second_person_time : ℝ) : 
  total_time = 8 → first_person_time = 24 → second_person_time = 12 := by
  sorry

#check second_person_work_time

end second_person_work_time_l3346_334652


namespace team_games_theorem_l3346_334648

theorem team_games_theorem (first_games : Nat) (win_rate_first : Real) 
  (win_rate_remaining : Real) (total_win_rate : Real) :
  first_games = 30 →
  win_rate_first = 0.4 →
  win_rate_remaining = 0.8 →
  total_win_rate = 0.6 →
  ∃ (total_games : Nat),
    total_games = 60 ∧
    (first_games : Real) * win_rate_first + 
    (total_games - first_games : Real) * win_rate_remaining = 
    (total_games : Real) * total_win_rate :=
by sorry

#check team_games_theorem

end team_games_theorem_l3346_334648


namespace fraction_inequality_l3346_334673

theorem fraction_inequality (a b m : ℝ) 
  (h1 : a > b) (h2 : b > 0) (h3 : m > 0) (h4 : a + m > 0) :
  (b + m) / (a + m) > b / a :=
by sorry

end fraction_inequality_l3346_334673


namespace firefighter_solution_l3346_334688

/-- Represents the problem of calculating the number of firefighters needed to put out a fire. -/
def FirefighterProblem (hose_rate : ℚ) (water_needed : ℚ) (time_taken : ℚ) : Prop :=
  ∃ (num_firefighters : ℚ),
    num_firefighters * hose_rate * time_taken = water_needed ∧
    num_firefighters = 5

/-- Theorem stating that given the specific conditions of the problem, 
    the number of firefighters required is 5. -/
theorem firefighter_solution :
  FirefighterProblem 20 4000 40 := by
  sorry

end firefighter_solution_l3346_334688


namespace min_x_value_l3346_334684

theorem min_x_value (x y : ℕ+) (h : (100 : ℚ)/151 < (y : ℚ)/(x : ℚ) ∧ (y : ℚ)/(x : ℚ) < (200 : ℚ)/251) :
  ∀ z : ℕ+, z < x → ¬∃ w : ℕ+, (100 : ℚ)/151 < (w : ℚ)/(z : ℚ) ∧ (w : ℚ)/(z : ℚ) < (200 : ℚ)/251 :=
by sorry

end min_x_value_l3346_334684


namespace symmetric_difference_of_A_and_B_l3346_334694

-- Define the sets A and B
def A : Set ℝ := {y : ℝ | ∃ x : ℝ, y = x^2 - 3*x}
def B : Set ℝ := {y : ℝ | ∃ x : ℝ, y = -2^x}

-- Define the symmetric difference operation
def symmetricDifference (X Y : Set ℝ) : Set ℝ := (X \ Y) ∪ (Y \ X)

-- State the theorem
theorem symmetric_difference_of_A_and_B :
  symmetricDifference A B = {y : ℝ | y < -9/4 ∨ y ≥ 0} := by sorry

end symmetric_difference_of_A_and_B_l3346_334694


namespace largest_reciprocal_l3346_334613

theorem largest_reciprocal (a b c d e : ℚ) : 
  a = 2/7 → b = 3/8 → c = 1 → d = 4 → e = 2000 → 
  (1/a > 1/b ∧ 1/a > 1/c ∧ 1/a > 1/d ∧ 1/a > 1/e) := by
  sorry

end largest_reciprocal_l3346_334613


namespace unique_n_divisibility_l3346_334637

theorem unique_n_divisibility : ∃! n : ℕ, 
  0 < n ∧ n < 1000 ∧ 
  (∃ k₁ k₂ k₃ : ℕ, 
    345564 - n = 13 * k₁ ∧ 
    345564 - n = 17 * k₂ ∧ 
    345564 - n = 19 * k₃) :=
by sorry

end unique_n_divisibility_l3346_334637


namespace min_variance_product_l3346_334667

theorem min_variance_product (a b : ℝ) : 
  2 ≤ 3 ∧ 3 ≤ 3 ∧ 3 ≤ 7 ∧ 7 ≤ a ∧ a ≤ b ∧ b ≤ 12 ∧ 12 ≤ 13.7 ∧ 13.7 ≤ 18.3 ∧ 18.3 ≤ 21 →
  (2 + 3 + 3 + 7 + a + b + 12 + 13.7 + 18.3 + 21) / 10 = 10 →
  a + b = 20 →
  (∀ x y : ℝ, x + y = 20 → (x - 10)^2 + (y - 10)^2 ≥ (a - 10)^2 + (b - 10)^2) →
  a * b = 100 :=
by sorry


end min_variance_product_l3346_334667


namespace brandon_application_theorem_l3346_334639

/-- The number of businesses Brandon can still apply to -/
def businesses_can_apply (total : ℕ) (fired : ℕ) (quit : ℕ) (x : ℕ) (y : ℕ) : ℕ :=
  total - (fired + quit - x) + y

theorem brandon_application_theorem (x y : ℕ) :
  businesses_can_apply 72 36 24 x y = 12 + x + y := by
  sorry

end brandon_application_theorem_l3346_334639


namespace tom_rides_l3346_334672

/-- Given the total number of tickets, tickets spent, and cost per ride,
    calculate the number of rides Tom can go on. -/
def number_of_rides (total_tickets spent_tickets cost_per_ride : ℕ) : ℕ :=
  (total_tickets - spent_tickets) / cost_per_ride

/-- Theorem stating that Tom can go on 3 rides given the specific conditions. -/
theorem tom_rides : number_of_rides 40 28 4 = 3 := by
  sorry

end tom_rides_l3346_334672


namespace train_meet_time_l3346_334615

/-- The time (in hours after midnight) when the trains meet -/
def meet_time : ℝ := 11

/-- The time (in hours after midnight) when the train from B starts -/
def start_time_B : ℝ := 8

/-- The distance between stations A and B in kilometers -/
def distance : ℝ := 155

/-- The speed of the train from A in km/h -/
def speed_A : ℝ := 20

/-- The speed of the train from B in km/h -/
def speed_B : ℝ := 25

/-- The time (in hours after midnight) when the train from A starts -/
def start_time_A : ℝ := 7

theorem train_meet_time :
  start_time_A = meet_time - (distance - speed_B * (meet_time - start_time_B)) / speed_A :=
by sorry

end train_meet_time_l3346_334615


namespace cricketer_average_score_l3346_334602

theorem cricketer_average_score 
  (initial_average : ℝ) 
  (runs_19th_inning : ℝ) 
  (average_increase : ℝ) : 
  runs_19th_inning = 96 →
  average_increase = 4 →
  (18 * initial_average + runs_19th_inning) / 19 = initial_average + average_increase →
  (18 * initial_average + runs_19th_inning) / 19 = 24 :=
by sorry

end cricketer_average_score_l3346_334602


namespace probability_of_letter_in_mathematics_l3346_334605

def alphabet : Finset Char := sorry

def mathematics : String := "MATHEMATICS"

def uniqueLetters (s : String) : Finset Char :=
  s.toList.toFinset

theorem probability_of_letter_in_mathematics :
  (uniqueLetters mathematics).card / alphabet.card = 4 / 13 := by
  sorry

end probability_of_letter_in_mathematics_l3346_334605


namespace four_digit_numbers_with_5_or_8_l3346_334635

/-- The set of digits excluding 0, 5, and 8 -/
def ValidFirstDigits : Finset ℕ := {1, 2, 3, 4, 6, 7, 9}

/-- The set of digits excluding 5 and 8 -/
def ValidOtherDigits : Finset ℕ := {0, 1, 2, 3, 4, 6, 7, 9}

/-- The number of four-digit numbers -/
def TotalFourDigitNumbers : ℕ := 9000

/-- The number of four-digit numbers without 5 or 8 -/
def NumbersWithout5Or8 : ℕ := Finset.card ValidFirstDigits * Finset.card ValidOtherDigits ^ 3

theorem four_digit_numbers_with_5_or_8 :
  TotalFourDigitNumbers - NumbersWithout5Or8 = 5416 := by
  sorry

end four_digit_numbers_with_5_or_8_l3346_334635


namespace ratio_sum_theorem_l3346_334695

theorem ratio_sum_theorem (w x y : ℝ) (hw_x : w / x = 1 / 3) (hw_y : w / y = 2 / 3) :
  (x + y) / y = 3 := by
  sorry

end ratio_sum_theorem_l3346_334695


namespace math_homework_pages_l3346_334628

theorem math_homework_pages (total_problems : ℕ) (reading_pages : ℕ) (problems_per_page : ℕ) 
  (h1 : total_problems = 30)
  (h2 : reading_pages = 4)
  (h3 : problems_per_page = 3) :
  total_problems - reading_pages * problems_per_page = 6 * problems_per_page :=
by sorry

end math_homework_pages_l3346_334628


namespace gcd_228_1995_l3346_334675

theorem gcd_228_1995 : Nat.gcd 228 1995 = 57 := by
  sorry

end gcd_228_1995_l3346_334675


namespace imaginary_part_of_complex_fraction_l3346_334686

theorem imaginary_part_of_complex_fraction (i : ℂ) :
  i * i = -1 →
  (5 * i / (2 - i)).im = 2 :=
by sorry

end imaginary_part_of_complex_fraction_l3346_334686


namespace total_peaches_l3346_334643

theorem total_peaches (num_baskets : ℕ) (red_per_basket : ℕ) (green_per_basket : ℕ) :
  num_baskets = 11 →
  red_per_basket = 10 →
  green_per_basket = 18 →
  num_baskets * (red_per_basket + green_per_basket) = 308 :=
by
  sorry

end total_peaches_l3346_334643


namespace brendans_morning_catch_l3346_334630

theorem brendans_morning_catch (total : ℕ) (thrown_back : ℕ) (afternoon_catch : ℕ) (dad_catch : ℕ)
  (h1 : total = 23)
  (h2 : thrown_back = 3)
  (h3 : afternoon_catch = 5)
  (h4 : dad_catch = 13) :
  total = (morning_catch - thrown_back + afternoon_catch + dad_catch) →
  morning_catch = 8 :=
by
  sorry

end brendans_morning_catch_l3346_334630


namespace video_game_expenditure_l3346_334636

theorem video_game_expenditure (total : ℝ) (books snacks movies video_games : ℝ) : 
  total = 50 ∧ 
  books = (1/4) * total ∧ 
  snacks = (1/5) * total ∧ 
  movies = (2/5) * total ∧ 
  total = books + snacks + movies + video_games 
  → video_games = 7.5 := by
sorry

end video_game_expenditure_l3346_334636


namespace topsoil_cost_theorem_l3346_334668

/-- The cost of topsoil in dollars per cubic foot -/
def cost_per_cubic_foot : ℝ := 8

/-- The conversion factor from cubic yards to cubic feet -/
def cubic_yards_to_cubic_feet : ℝ := 27

/-- The volume of topsoil in cubic yards -/
def volume_in_cubic_yards : ℝ := 7

/-- The cost of topsoil for a given volume in cubic yards -/
def topsoil_cost (volume : ℝ) : ℝ :=
  volume * cubic_yards_to_cubic_feet * cost_per_cubic_foot

theorem topsoil_cost_theorem :
  topsoil_cost volume_in_cubic_yards = 1512 := by
  sorry

end topsoil_cost_theorem_l3346_334668


namespace bulk_warehouse_case_size_l3346_334687

/-- Proves the number of cans in a bulk warehouse case given pricing information -/
theorem bulk_warehouse_case_size (bulk_case_price : ℚ) (grocery_price : ℚ) (grocery_cans : ℕ) (price_difference : ℚ) : 
  bulk_case_price = 12 →
  grocery_price = 6 →
  grocery_cans = 12 →
  price_difference = 1/4 →
  (bulk_case_price / ((grocery_price / grocery_cans) - price_difference) : ℚ) = 48 :=
by sorry

end bulk_warehouse_case_size_l3346_334687


namespace two_digit_numbers_count_l3346_334669

/-- The number of permutations of n elements taken r at a time -/
def permutations (n : ℕ) (r : ℕ) : ℕ := 
  if r ≤ n then Nat.factorial n / Nat.factorial (n - r) else 0

/-- The set of digits used -/
def digits : Finset ℕ := {1, 2, 3, 4, 5}

theorem two_digit_numbers_count : permutations (Finset.card digits) 2 = 20 := by
  sorry

end two_digit_numbers_count_l3346_334669


namespace game_ends_in_37_rounds_l3346_334611

/-- Represents the state of the game at any point --/
structure GameState where
  a : ℕ  -- tokens of player A
  b : ℕ  -- tokens of player B
  c : ℕ  -- tokens of player C

/-- Represents a single round of the game --/
def playRound (state : GameState) : GameState :=
  if state.a ≥ state.b ∧ state.a ≥ state.c then
    { a := state.a - 3, b := state.b + 1, c := state.c + 1 }
  else if state.b ≥ state.a ∧ state.b ≥ state.c then
    { a := state.a + 1, b := state.b - 3, c := state.c + 1 }
  else
    { a := state.a + 1, b := state.b + 1, c := state.c - 3 }

/-- Checks if the game has ended (any player has 0 tokens) --/
def gameEnded (state : GameState) : Bool :=
  state.a = 0 ∨ state.b = 0 ∨ state.c = 0

/-- Plays the game for a given number of rounds --/
def playGame (initialState : GameState) (rounds : ℕ) : GameState :=
  match rounds with
  | 0 => initialState
  | n + 1 => playRound (playGame initialState n)

/-- The main theorem to prove --/
theorem game_ends_in_37_rounds :
  let initialState : GameState := { a := 15, b := 14, c := 13 }
  let finalState := playGame initialState 37
  gameEnded finalState ∧ ¬gameEnded (playGame initialState 36) := by
  sorry

#check game_ends_in_37_rounds

end game_ends_in_37_rounds_l3346_334611


namespace roots_of_x_squared_equals_16_l3346_334681

theorem roots_of_x_squared_equals_16 :
  let f : ℝ → ℝ := λ x ↦ x^2 - 16
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ = 4 ∧ x₂ = -4 :=
by sorry

end roots_of_x_squared_equals_16_l3346_334681


namespace farmer_earnings_l3346_334608

/-- Calculates the total earnings from selling potatoes and carrots given the harvest quantities and pricing. -/
theorem farmer_earnings (potato_count : ℕ) (potato_bundle_size : ℕ) (potato_bundle_price : ℚ)
                        (carrot_count : ℕ) (carrot_bundle_size : ℕ) (carrot_bundle_price : ℚ) :
  potato_count = 250 →
  potato_bundle_size = 25 →
  potato_bundle_price = 190 / 100 →
  carrot_count = 320 →
  carrot_bundle_size = 20 →
  carrot_bundle_price = 2 →
  (potato_count / potato_bundle_size * potato_bundle_price +
   carrot_count / carrot_bundle_size * carrot_bundle_price : ℚ) = 51 := by
  sorry

#eval (250 / 25 * (190 / 100) + 320 / 20 * 2 : ℚ)

end farmer_earnings_l3346_334608


namespace gianna_savings_l3346_334670

/-- Calculates the total savings given a daily savings amount and number of days -/
def totalSavings (dailySavings : ℕ) (days : ℕ) : ℕ :=
  dailySavings * days

/-- Proves that saving $39 every day for 365 days results in $14,235 total savings -/
theorem gianna_savings : totalSavings 39 365 = 14235 := by
  sorry

end gianna_savings_l3346_334670


namespace geometric_sequence_sum_l3346_334621

/-- 
Given a geometric sequence {a_n} with positive terms, 
if a_1 = 3 and S_3 = 21, then a_3 + a_4 + a_5 = 84.
-/
theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →  -- all terms are positive
  (∀ n, a (n + 1) = a n * q) →  -- definition of geometric sequence
  a 1 = 3 →  -- first term
  (a 1 + a 2 + a 3 = 21) →  -- S_3 = 21
  (a 3 + a 4 + a 5 = 84) :=
by sorry

end geometric_sequence_sum_l3346_334621


namespace probability_both_truth_l3346_334609

theorem probability_both_truth (prob_A prob_B : ℝ) 
  (h_A : prob_A = 0.7) 
  (h_B : prob_B = 0.6) : 
  prob_A * prob_B = 0.42 := by
sorry

end probability_both_truth_l3346_334609


namespace geometric_sequence_common_ratio_l3346_334679

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) -- The geometric sequence
  (S : ℕ → ℝ) -- The sum function
  (h_geom : ∀ n, a (n + 1) = a n * (a 1 / a 0)) -- Condition for geometric sequence
  (h_sum : ∀ n, S n = (a 0) * (1 - (a 1 / a 0)^n) / (1 - (a 1 / a 0))) -- Sum formula
  (h_eq : 8 * S 6 = 7 * S 3) -- Given equation
  : a 1 / a 0 = -1/2 := by
sorry

end geometric_sequence_common_ratio_l3346_334679


namespace remainder_problem_l3346_334690

theorem remainder_problem (n : ℕ) (h : n > 0) (h1 : (n + 1) % 6 = 4) : n % 2 = 1 := by
  sorry

end remainder_problem_l3346_334690


namespace max_houses_buildable_l3346_334601

def houses_buildable (sinks doors windows toilets : ℕ) : ℕ :=
  min (sinks / 6) (min (doors / 4) (min (windows / 8) (toilets / 3)))

theorem max_houses_buildable :
  houses_buildable 266 424 608 219 = 73 := by
  sorry

end max_houses_buildable_l3346_334601


namespace teairra_shirt_count_l3346_334634

/-- The number of shirts Teairra has in her closet -/
def num_shirts : ℕ := sorry

/-- The total number of pants Teairra has -/
def total_pants : ℕ := 24

/-- The number of plaid shirts -/
def plaid_shirts : ℕ := 3

/-- The number of purple pants -/
def purple_pants : ℕ := 5

/-- The number of items (shirts and pants) that are neither plaid nor purple -/
def neither_plaid_nor_purple : ℕ := 21

theorem teairra_shirt_count : num_shirts = 5 := by
  sorry

end teairra_shirt_count_l3346_334634


namespace horner_v3_value_l3346_334663

def horner_v3 (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℤ) (x : ℤ) : ℤ :=
  let v := a₆
  let v₁ := v * x + a₅
  let v₂ := v₁ * x + a₄
  v₂ * x + a₃

theorem horner_v3_value :
  horner_v3 12 35 (-8) 79 6 5 3 (-4) = -57 := by
  sorry

end horner_v3_value_l3346_334663


namespace triangle_theorem_l3346_334624

theorem triangle_theorem (a b c A B C : ℝ) (h1 : a * Real.cos C + (1/2) * c = b)
                         (h2 : b = 4) (h3 : c = 6) : 
  A = π/3 ∧ Real.cos B = 2/Real.sqrt 7 ∧ Real.cos (A + 2*B) = -11/14 := by
  sorry

end triangle_theorem_l3346_334624


namespace car_expense_difference_l3346_334685

-- Define Alberto's expenses
def alberto_engine : ℝ := 2457
def alberto_transmission : ℝ := 374
def alberto_tires : ℝ := 520
def alberto_discount_rate : ℝ := 0.05

-- Define Samara's expenses
def samara_oil : ℝ := 25
def samara_tires : ℝ := 467
def samara_detailing : ℝ := 79
def samara_stereo : ℝ := 150
def samara_tax_rate : ℝ := 0.07

-- Theorem statement
theorem car_expense_difference : 
  let alberto_total := alberto_engine + alberto_transmission + alberto_tires
  let alberto_discount := alberto_total * alberto_discount_rate
  let alberto_final := alberto_total - alberto_discount
  let samara_total := samara_oil + samara_tires + samara_detailing + samara_stereo
  let samara_tax := samara_total * samara_tax_rate
  let samara_final := samara_total + samara_tax
  alberto_final - samara_final = 2411.98 := by
    sorry

end car_expense_difference_l3346_334685


namespace inequality_proof_l3346_334626

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (Real.sqrt (a^2 + 8*b*c))/a + (Real.sqrt (b^2 + 8*a*c))/b + (Real.sqrt (c^2 + 8*a*b))/c ≥ 9 := by
  sorry

end inequality_proof_l3346_334626


namespace medicine_price_reduction_l3346_334664

theorem medicine_price_reduction (x : ℝ) : 
  (25 : ℝ) * (1 - x)^2 = 16 ↔ 
  (∃ (price_after_first_reduction : ℝ),
    price_after_first_reduction = 25 * (1 - x) ∧
    16 = price_after_first_reduction * (1 - x)) :=
by sorry

end medicine_price_reduction_l3346_334664


namespace total_tickets_is_84_l3346_334697

-- Define the prices of items in tickets
def hat_price : ℕ := 2
def stuffed_animal_price : ℕ := 10
def yoyo_price : ℕ := 2
def keychain_price : ℕ := 3
def poster_price : ℕ := 7
def toy_car_price : ℕ := 5
def puzzle_price : ℕ := 8
def tshirt_price : ℕ := 15
def novelty_pen_price : ℕ := 4

-- Define the special offer price for two posters
def two_posters_special_price : ℕ := 10

-- Define the function to calculate the total tickets spent
def total_tickets_spent : ℕ :=
  -- First trip
  hat_price + stuffed_animal_price + yoyo_price +
  -- Second trip
  keychain_price + poster_price + toy_car_price +
  -- Third trip
  puzzle_price + tshirt_price + novelty_pen_price +
  -- Fourth trip (special offer for posters)
  two_posters_special_price + stuffed_animal_price +
  -- Fifth trip (50% off sale)
  (tshirt_price / 2) + (toy_car_price / 2)

-- Theorem to prove
theorem total_tickets_is_84 : total_tickets_spent = 84 := by
  sorry

end total_tickets_is_84_l3346_334697


namespace binomial_18_4_l3346_334638

theorem binomial_18_4 : Nat.choose 18 4 = 3060 := by
  sorry

end binomial_18_4_l3346_334638


namespace pirate_loot_sum_l3346_334603

def base_6_to_10 (n : List Nat) : Nat :=
  List.foldl (fun acc d => acc * 6 + d) 0 n.reverse

theorem pirate_loot_sum :
  let silver := base_6_to_10 [4, 5, 3, 2]
  let pearls := base_6_to_10 [1, 2, 5, 4]
  let spices := base_6_to_10 [6, 5, 4]
  silver + pearls + spices = 1636 := by
sorry

end pirate_loot_sum_l3346_334603


namespace emilys_earnings_l3346_334658

-- Define the work hours for each day
def monday_hours : Real := 1
def wednesday_start : Real := 14.17  -- 2:10 PM in 24-hour format
def wednesday_end : Real := 16.83    -- 4:50 PM in 24-hour format
def thursday_hours : Real := 0.5
def saturday_hours : Real := 0.5

-- Define the hourly rate
def hourly_rate : Real := 4

-- Define the total earnings
def total_earnings : Real :=
  (monday_hours + (wednesday_end - wednesday_start) + thursday_hours + saturday_hours) * hourly_rate

-- Theorem to prove
theorem emilys_earnings :
  total_earnings = 18.68 := by
  sorry

end emilys_earnings_l3346_334658


namespace shaded_region_circle_diameter_l3346_334696

/-- Given two concentric circles with radii 24 and 36 units, the diameter of a new circle
    whose diameter is equal to the area of the shaded region between the two circles
    is 720π units. -/
theorem shaded_region_circle_diameter :
  let r₁ : ℝ := 24
  let r₂ : ℝ := 36
  let shaded_area := π * (r₂^2 - r₁^2)
  let new_circle_diameter := shaded_area
  new_circle_diameter = 720 * π :=
by sorry

end shaded_region_circle_diameter_l3346_334696


namespace repeating_decimal_difference_l3346_334674

theorem repeating_decimal_difference : 
  (6 : ℚ) / 11 - 54 / 100 = 6 / 1100 := by sorry

end repeating_decimal_difference_l3346_334674


namespace greatest_divisor_four_consecutive_integers_l3346_334665

theorem greatest_divisor_four_consecutive_integers :
  ∀ n : ℕ, n > 0 →
  ∃ k : ℕ, k > 0 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) ∧
  ∀ m : ℕ, m > k → ¬(∀ i : ℕ, i > 0 → m ∣ (i * (i + 1) * (i + 2) * (i + 3))) →
  k = 24 :=
by sorry

end greatest_divisor_four_consecutive_integers_l3346_334665


namespace initial_sum_is_500_l3346_334617

/-- Prove that the initial sum of money is $500 given the conditions of the problem. -/
theorem initial_sum_is_500 
  (sum_after_2_years : ℝ → ℝ → ℝ → ℝ) -- Function for final amount after 2 years
  (initial_sum : ℝ)  -- Initial sum of money
  (interest_rate : ℝ) -- Original interest rate
  (h1 : sum_after_2_years initial_sum interest_rate 2 = 600) -- First condition
  (h2 : sum_after_2_years initial_sum (interest_rate + 0.1) 2 = 700) -- Second condition
  : initial_sum = 500 := by
  sorry

end initial_sum_is_500_l3346_334617


namespace pqr_value_l3346_334623

theorem pqr_value (p q r : ℤ) 
  (h1 : p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0)
  (h2 : p + q + r = 29)
  (h3 : 1 / p + 1 / q + 1 / r + 392 / (p * q * r) = 1) :
  p * q * r = 630 := by
sorry

end pqr_value_l3346_334623


namespace cube_with_holes_surface_area_l3346_334693

/-- Calculates the total surface area of a cube with square holes on each face. -/
def totalSurfaceArea (cubeEdge : ℝ) (holeEdge : ℝ) (holeDepth : ℝ) : ℝ :=
  let originalSurface := 6 * cubeEdge^2
  let holeArea := 6 * holeEdge^2
  let newSurfaceInHoles := 6 * 4 * holeEdge * holeDepth
  originalSurface - holeArea + newSurfaceInHoles

/-- Theorem: The total surface area of a cube with edge length 4 meters and
    square holes (side 1 meter, depth 1 meter) centered on each face is 114 square meters. -/
theorem cube_with_holes_surface_area :
  totalSurfaceArea 4 1 1 = 114 := by
  sorry

end cube_with_holes_surface_area_l3346_334693


namespace spheres_radius_in_cone_l3346_334625

/-- Represents a right circular cone --/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Represents a sphere --/
structure Sphere where
  radius : ℝ

/-- Represents the configuration of three spheres in a cone --/
structure SpheresInCone where
  cone : Cone
  sphere : Sphere
  spheresTangent : Bool
  spheresTangentToBase : Bool
  spheresNotTangentToSides : Bool

/-- The theorem statement --/
theorem spheres_radius_in_cone (config : SpheresInCone) : 
  config.cone.baseRadius = 6 ∧ 
  config.cone.height = 15 ∧ 
  config.spheresTangent ∧ 
  config.spheresTangentToBase ∧ 
  config.spheresNotTangentToSides →
  config.sphere.radius = 27 - 6 * Real.sqrt 3 := by
  sorry

end spheres_radius_in_cone_l3346_334625


namespace adult_ticket_price_l3346_334640

theorem adult_ticket_price (total_tickets : ℕ) (senior_price : ℕ) (total_receipts : ℕ) (senior_tickets : ℕ) :
  total_tickets = 510 →
  senior_price = 15 →
  total_receipts = 8748 →
  senior_tickets = 327 →
  ∃ (adult_price : ℕ), adult_price = 21 ∧ 
    total_receipts = senior_price * senior_tickets + adult_price * (total_tickets - senior_tickets) :=
by
  sorry

end adult_ticket_price_l3346_334640


namespace large_pizza_slices_l3346_334641

/-- Proves that a large pizza has 12 slices given the problem conditions -/
theorem large_pizza_slices :
  ∀ (small_slices medium_slices large_slices : ℕ)
    (total_pizzas small_pizzas medium_pizzas large_pizzas : ℕ)
    (total_slices : ℕ),
  small_slices = 6 →
  medium_slices = 8 →
  total_pizzas = 15 →
  small_pizzas = 4 →
  medium_pizzas = 5 →
  large_pizzas = total_pizzas - small_pizzas - medium_pizzas →
  total_slices = 136 →
  total_slices = small_slices * small_pizzas + medium_slices * medium_pizzas + large_slices * large_pizzas →
  large_slices = 12 := by
sorry

end large_pizza_slices_l3346_334641


namespace smallest_class_size_seventeen_satisfies_conditions_smallest_class_size_is_seventeen_l3346_334607

theorem smallest_class_size (n : ℕ) : 
  (n % 4 = 1) ∧ (n % 5 = 2) ∧ (n % 7 = 3) → n ≥ 17 :=
by sorry

theorem seventeen_satisfies_conditions : 
  (17 % 4 = 1) ∧ (17 % 5 = 2) ∧ (17 % 7 = 3) :=
by sorry

theorem smallest_class_size_is_seventeen : 
  ∃ (n : ℕ), n = 17 ∧ (n % 4 = 1) ∧ (n % 5 = 2) ∧ (n % 7 = 3) ∧ 
  (∀ m : ℕ, (m % 4 = 1) ∧ (m % 5 = 2) ∧ (m % 7 = 3) → m ≥ n) :=
by sorry

end smallest_class_size_seventeen_satisfies_conditions_smallest_class_size_is_seventeen_l3346_334607


namespace f_at_negative_two_l3346_334614

/-- Given a function f(x) = 2x^2 - 3x + 1, prove that f(-2) = 15 -/
theorem f_at_negative_two (f : ℝ → ℝ) (h : ∀ x, f x = 2 * x^2 - 3 * x + 1) : 
  f (-2) = 15 := by
  sorry

end f_at_negative_two_l3346_334614


namespace adams_farm_animals_l3346_334682

theorem adams_farm_animals (cows sheep pigs : ℕ) : 
  sheep = 2 * cows →
  pigs = 3 * sheep →
  cows + sheep + pigs = 108 →
  cows = 12 := by
sorry

end adams_farm_animals_l3346_334682


namespace females_band_not_orchestra_l3346_334645

/-- Represents the number of students in different groups -/
structure StudentGroups where
  bandFemales : ℕ
  bandMales : ℕ
  orchestraFemales : ℕ
  orchestraMales : ℕ
  bothFemales : ℕ
  totalStudents : ℕ

/-- Theorem stating the number of females in the band but not in the orchestra -/
theorem females_band_not_orchestra (g : StudentGroups)
  (h1 : g.bandFemales = 120)
  (h2 : g.bandMales = 70)
  (h3 : g.orchestraFemales = 70)
  (h4 : g.orchestraMales = 110)
  (h5 : g.bothFemales = 45)
  (h6 : g.totalStudents = 250) :
  g.bandFemales - g.bothFemales = 75 := by
  sorry

#check females_band_not_orchestra

end females_band_not_orchestra_l3346_334645


namespace greatest_sum_on_circle_l3346_334683

theorem greatest_sum_on_circle (x y : ℤ) (h : x^2 + y^2 = 50) : x + y ≤ 10 := by
  sorry

end greatest_sum_on_circle_l3346_334683


namespace point_on_segment_with_vector_relation_l3346_334622

/-- Given two points M and N in ℝ², and a point P on the line segment MN
    such that vector PN = -2 * vector PM, prove that P has coordinates (2,4) -/
theorem point_on_segment_with_vector_relation (M N P : ℝ × ℝ) :
  M = (-2, 7) →
  N = (10, -2) →
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • M + t • N →
  (N.1 - P.1, N.2 - P.2) = (-2 * (M.1 - P.1), -2 * (M.2 - P.2)) →
  P = (2, 4) := by
  sorry

end point_on_segment_with_vector_relation_l3346_334622


namespace log_expression_equals_one_l3346_334678

-- Define the base-10 logarithm
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Theorem statement
theorem log_expression_equals_one :
  (log10 2)^2 + log10 20 * log10 5 = 1 := by
  sorry

end log_expression_equals_one_l3346_334678


namespace sugar_in_house_l3346_334618

/-- Given the total sugar needed and additional sugar needed, prove the amount of sugar stored in the house. -/
theorem sugar_in_house (total_sugar : ℕ) (additional_sugar : ℕ) 
  (h1 : total_sugar = 450)
  (h2 : additional_sugar = 163) :
  total_sugar - additional_sugar = 287 := by
  sorry

end sugar_in_house_l3346_334618


namespace complex_product_equality_l3346_334692

theorem complex_product_equality (x : ℂ) (h : x = Complex.exp (2 * Real.pi * Complex.I / 9)) : 
  (3 * x + x^3) * (3 * x^3 + x^9) * (3 * x^6 + x^18) = 
  22 - 9 * x^5 - 9 * x^2 + 3 * x^6 + 4 * x^3 + 3 * x :=
by sorry

end complex_product_equality_l3346_334692


namespace f_major_premise_incorrect_l3346_334610

-- Define the function f(x) = x^3
def f (x : ℝ) : ℝ := x^3

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3 * x^2

-- State that f'(0) = 0
theorem f'_zero : f' 0 = 0 := by sorry

-- Define what it means for a point to be an extremum
def is_extremum (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∃ ε > 0, ∀ x ∈ Set.Ioo (x₀ - ε) (x₀ + ε), f x₀ ≥ f x ∨ f x₀ ≤ f x

-- Theorem stating that the major premise is incorrect
theorem major_premise_incorrect :
  ¬(∀ x₀ : ℝ, f' x₀ = 0 → is_extremum f x₀) := by sorry

end f_major_premise_incorrect_l3346_334610


namespace dice_probability_l3346_334600

/-- The number of sides on each die -/
def n : ℕ := 4025

/-- The threshold for the first die -/
def k : ℕ := 2012

/-- The probability that the first die is less than or equal to k,
    given that it's greater than or equal to the second die -/
def prob : ℚ :=
  (k * (k + 1)) / (n * (n + 1))

theorem dice_probability :
  prob = 1006 / 4025 := by sorry

end dice_probability_l3346_334600


namespace park_visitors_l3346_334631

/-- Represents the charging conditions for the park visit -/
structure ParkVisitConditions where
  base_fee : ℕ  -- Base fee per person
  base_limit : ℕ  -- Number of people for base fee
  discount_per_person : ℕ  -- Discount per additional person
  min_fee : ℕ  -- Minimum fee per person
  total_paid : ℕ  -- Total amount paid

/-- Calculates the fee per person based on the number of visitors -/
def fee_per_person (conditions : ParkVisitConditions) (num_visitors : ℕ) : ℕ :=
  max conditions.min_fee (conditions.base_fee - conditions.discount_per_person * (num_visitors - conditions.base_limit))

/-- Theorem: Given the charging conditions, 30 people visited the park -/
theorem park_visitors (conditions : ParkVisitConditions) 
  (h1 : conditions.base_fee = 100)
  (h2 : conditions.base_limit = 25)
  (h3 : conditions.discount_per_person = 2)
  (h4 : conditions.min_fee = 70)
  (h5 : conditions.total_paid = 2700) :
  ∃ (num_visitors : ℕ), 
    num_visitors = 30 ∧ 
    num_visitors * (fee_per_person conditions num_visitors) = conditions.total_paid :=
sorry

end park_visitors_l3346_334631


namespace lcm_problem_l3346_334606

theorem lcm_problem (m : ℕ) (h1 : m > 0) (h2 : Nat.lcm 30 m = 90) (h3 : Nat.lcm m 45 = 180) : m = 36 := by
  sorry

end lcm_problem_l3346_334606


namespace min_value_of_expression_l3346_334671

theorem min_value_of_expression (x y : ℝ) 
  (h1 : x > -1) 
  (h2 : y > 0) 
  (h3 : x + 2*y = 2) : 
  ∃ (m : ℝ), m = 3 ∧ ∀ (a b : ℝ), a > -1 → b > 0 → a + 2*b = 2 → 1/(a+1) + 2/b ≥ m :=
by
  sorry

end min_value_of_expression_l3346_334671


namespace house_painting_and_window_washing_l3346_334698

/-- Represents the number of people needed to complete a task in a given number of days -/
structure WorkForce :=
  (people : ℕ)
  (days : ℕ)

/-- Calculates the total person-days for a given workforce -/
def personDays (w : WorkForce) : ℕ := w.people * w.days

theorem house_painting_and_window_washing 
  (paint_initial : WorkForce) 
  (paint_target : WorkForce) 
  (wash_initial : WorkForce) 
  (wash_target : WorkForce) :
  paint_initial.people = 8 →
  paint_initial.days = 5 →
  paint_target.days = 3 →
  wash_initial.people = paint_initial.people →
  wash_initial.days = 4 →
  wash_target.people = wash_initial.people + 4 →
  personDays paint_initial = personDays paint_target →
  personDays wash_initial = personDays wash_target →
  paint_target.people = 14 ∧ wash_target.days = 3 := by
  sorry

#check house_painting_and_window_washing

end house_painting_and_window_washing_l3346_334698


namespace first_discount_percentage_l3346_334646

/-- Proves that given specific conditions on the original price, final price, and second discount,
    the first discount must be 12%. -/
theorem first_discount_percentage (original_price : ℝ) (final_price : ℝ) (second_discount : ℝ) :
  original_price = 400 →
  final_price = 334.4 →
  second_discount = 5 →
  ∃ (first_discount : ℝ),
    final_price = original_price * (1 - first_discount / 100) * (1 - second_discount / 100) ∧
    first_discount = 12 := by
  sorry

#check first_discount_percentage

end first_discount_percentage_l3346_334646


namespace smallest_number_with_properties_l3346_334666

def ends_with_six (n : ℕ) : Prop :=
  n % 10 = 6

def move_six_to_front (n : ℕ) : ℕ :=
  let k := (Nat.log 10 n).succ
  6 * 10^k + (n - 6) / 10

theorem smallest_number_with_properties :
  ∃ (N : ℕ), N = 153846 ∧
  ends_with_six N ∧
  move_six_to_front N = 4 * N ∧
  ∀ (m : ℕ), m < N →
    ¬(ends_with_six m ∧ move_six_to_front m = 4 * m) :=
by sorry

end smallest_number_with_properties_l3346_334666


namespace jennifer_grooming_time_l3346_334691

/-- The time it takes Jennifer to groom one dog, in minutes. -/
def grooming_time : ℕ := 20

/-- The number of dogs Jennifer has. -/
def num_dogs : ℕ := 2

/-- The number of days in the given period. -/
def num_days : ℕ := 30

/-- The total time Jennifer spends grooming her dogs in the given period, in hours. -/
def total_grooming_time : ℕ := 20

theorem jennifer_grooming_time :
  grooming_time * num_dogs * num_days = total_grooming_time * 60 :=
sorry

end jennifer_grooming_time_l3346_334691


namespace quadratic_form_sum_l3346_334680

theorem quadratic_form_sum (x : ℝ) : ∃ (a b c : ℝ),
  (6 * x^2 + 72 * x + 432 = a * (x + b)^2 + c) ∧ (a + b + c = 228) := by
  sorry

end quadratic_form_sum_l3346_334680


namespace original_price_l3346_334649

theorem original_price (p q d : ℝ) (h_d_pos : d > 0) :
  let x := d / (1 + (p - q) / 100 - p * q / 10000)
  let price_after_increase := x * (1 + p / 100)
  let final_price := price_after_increase * (1 - q / 100)
  final_price = d :=
by sorry

end original_price_l3346_334649


namespace quadratic_inequality_range_l3346_334612

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, x^2 - a*x + 1 < 0) → a ∈ Set.Iio (-2) ∪ Set.Ioi 2 := by
  sorry

end quadratic_inequality_range_l3346_334612


namespace lily_book_count_l3346_334620

/-- The number of books Lily read last month -/
def books_last_month : ℕ := 4

/-- The number of books Lily plans to read this month -/
def books_this_month : ℕ := 2 * books_last_month

/-- The total number of books Lily will read in two months -/
def total_books : ℕ := books_last_month + books_this_month

theorem lily_book_count : total_books = 12 := by
  sorry

end lily_book_count_l3346_334620


namespace regular_polygon_sides_l3346_334651

theorem regular_polygon_sides (n : ℕ) (h : n > 2) : 
  (((n - 2) * 180 : ℕ) / n = 144) → n = 10 :=
by
  sorry

end regular_polygon_sides_l3346_334651


namespace green_to_red_ratio_is_three_to_one_l3346_334619

/-- Represents the contents of a bag of mints -/
structure MintBag where
  green : ℕ
  red : ℕ

/-- The ratio of green mints to red mints -/
def mintRatio (bag : MintBag) : ℚ :=
  bag.green / bag.red

theorem green_to_red_ratio_is_three_to_one 
  (bag : MintBag) 
  (h_total : bag.green + bag.red > 0)
  (h_green_percent : (bag.green : ℚ) / (bag.green + bag.red) = 3/4) :
  mintRatio bag = 3/1 := by
sorry

end green_to_red_ratio_is_three_to_one_l3346_334619


namespace dihedral_angle_lower_bound_l3346_334662

/-- Given a regular n-sided polygon inscribed in an arbitrary great circle of a sphere,
    with tangent planes laid at each vertex, the dihedral angle φ of the resulting
    polyhedral angle satisfies φ ≥ π(1 - 2/n). -/
theorem dihedral_angle_lower_bound (n : ℕ) (φ : ℝ) 
  (h1 : n ≥ 3)  -- n is at least 3 for a polygon
  (h2 : φ > 0)  -- dihedral angle is positive
  (h3 : φ < π)  -- dihedral angle is less than π
  : φ ≥ π * (1 - 2 / n) :=
sorry

end dihedral_angle_lower_bound_l3346_334662


namespace ratio_of_Q_at_one_and_minus_one_l3346_334632

/-- The polynomial g(x) = x^2009 + 19x^2008 + 1 -/
def g (x : ℂ) : ℂ := x^2009 + 19*x^2008 + 1

/-- The set of distinct zeros of g(x) -/
def S : Finset ℂ := sorry

/-- The polynomial Q of degree 2009 -/
noncomputable def Q : Polynomial ℂ := sorry

theorem ratio_of_Q_at_one_and_minus_one 
  (h1 : ∀ s ∈ S, g s = 0)
  (h2 : Finset.card S = 2009)
  (h3 : ∀ s ∈ S, Q.eval (s + 1/s) = 0)
  (h4 : Polynomial.degree Q = 2009) :
  Q.eval 1 / Q.eval (-1) = 361 / 331 := by sorry

end ratio_of_Q_at_one_and_minus_one_l3346_334632


namespace simultaneous_equations_solution_l3346_334604

theorem simultaneous_equations_solution :
  ∀ (a x : ℝ),
    (5 * x^3 + a * x^2 + 8 = 0 ∧ 5 * x^3 + 8 * x^2 + a = 0) ↔
    ((a = -13 ∧ x = 1) ∨ (a = -3 ∧ x = -1) ∨ (a = 8 ∧ x = -2)) :=
by sorry

end simultaneous_equations_solution_l3346_334604


namespace problem_statement_l3346_334659

theorem problem_statement (a b c : ℝ) 
  (h1 : a * c / (a + b) + b * a / (b + c) + c * b / (c + a) = -15)
  (h2 : b * c / (a + b) + c * a / (b + c) + a * b / (c + a) = 20) :
  b / (a + b) + c / (b + c) + a / (c + a) = 19 := by
  sorry

end problem_statement_l3346_334659


namespace focus_after_symmetry_l3346_334657

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = -8*x

-- Define the line of symmetry
def symmetry_line (x y : ℝ) : Prop := y = x - 1

-- Define the focus of the original parabola
def original_focus : ℝ × ℝ := (-2, 0)

-- Define the symmetric point
def symmetric_point (a b : ℝ) (p : ℝ × ℝ) : Prop :=
  (b + p.2) / 2 = (p.1 + a) / 2 - 1 ∧
  (b - p.2) / (a - p.1) = -1

-- Theorem statement
theorem focus_after_symmetry :
  ∃ (a b : ℝ), symmetric_point a b original_focus ∧ a = 1 ∧ b = -3 :=
sorry

end focus_after_symmetry_l3346_334657


namespace intersection_point_inequality_l3346_334661

theorem intersection_point_inequality (a b : ℝ) :
  (∃ x₀ : ℝ, x₀ > 0 ∧ Real.exp x₀ = a * Real.sin x₀ + b * Real.sqrt x₀) →
  a^2 + b^2 > Real.exp 1 := by
  sorry

end intersection_point_inequality_l3346_334661


namespace f_lower_bound_a_range_l3346_334699

def f (x : ℝ) : ℝ := |x - 2| + |x + 1| + 2 * |x + 2|

theorem f_lower_bound : ∀ x : ℝ, f x ≥ 5 := by sorry

theorem a_range (a : ℝ) : 
  (∀ x : ℝ, 15 - 2 * (f x) < a^2 + 9 / (a^2 + 1)) → 
  a ≠ Real.sqrt 2 ∧ a ≠ -Real.sqrt 2 := by sorry

end f_lower_bound_a_range_l3346_334699


namespace perpendicular_parallel_implication_l3346_334642

-- Define a type for lines in 3D space
variable (Line : Type)

-- Define the perpendicular relation between two lines
variable (perpendicular : Line → Line → Prop)

-- Define the parallel relation between two lines
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem perpendicular_parallel_implication (l₁ l₂ l₃ : Line) :
  perpendicular l₁ l₂ → parallel l₂ l₃ → perpendicular l₁ l₃ :=
by sorry

end perpendicular_parallel_implication_l3346_334642


namespace negation_of_exists_exp_leq_zero_l3346_334676

theorem negation_of_exists_exp_leq_zero :
  (¬ ∃ x : ℝ, Real.exp x ≤ 0) ↔ (∀ x : ℝ, Real.exp x > 0) := by sorry

end negation_of_exists_exp_leq_zero_l3346_334676


namespace ellipse_a_range_l3346_334644

/-- An ellipse with equation (x^2)/(a-5) + (y^2)/2 = 1 and foci on the x-axis -/
structure Ellipse (a : ℝ) where
  eq : ∀ (x y : ℝ), x^2 / (a - 5) + y^2 / 2 = 1
  foci_on_x : True  -- This is a placeholder for the condition that foci are on the x-axis

/-- The range of values for a in the given ellipse -/
theorem ellipse_a_range (a : ℝ) (e : Ellipse a) : a > 7 := by
  sorry

end ellipse_a_range_l3346_334644


namespace race_speed_ratio_l3346_334689

/-- Proves that A runs 4 times faster than B given the race conditions --/
theorem race_speed_ratio (v_B : ℝ) (k : ℝ) : 
  (k > 0) →  -- A is faster than B
  (88 / (k * v_B) = (88 - 66) / v_B) →  -- They finish at the same time
  (k = 4) :=
by sorry

end race_speed_ratio_l3346_334689


namespace gcd_of_three_numbers_l3346_334629

theorem gcd_of_three_numbers : Nat.gcd 13926 (Nat.gcd 20031 47058) = 33 := by
  sorry

end gcd_of_three_numbers_l3346_334629


namespace no_solution_iff_m_leq_two_l3346_334677

/-- The system of inequalities has no solution if and only if m ≤ 2 -/
theorem no_solution_iff_m_leq_two (m : ℝ) :
  (∀ x : ℝ, ¬(x - 2 < 3*x - 6 ∧ x < m)) ↔ m ≤ 2 :=
by sorry

end no_solution_iff_m_leq_two_l3346_334677


namespace zinc_copper_mixture_l3346_334654

theorem zinc_copper_mixture (total_weight : ℝ) (zinc_ratio copper_ratio : ℕ) : 
  total_weight = 70 →
  zinc_ratio = 9 →
  copper_ratio = 11 →
  (zinc_ratio : ℝ) / ((zinc_ratio : ℝ) + (copper_ratio : ℝ)) * total_weight = 31.5 := by
  sorry

end zinc_copper_mixture_l3346_334654


namespace melon_amount_in_fruit_salad_l3346_334653

/-- Given a fruit salad with melon and berries, prove the amount of melon used. -/
theorem melon_amount_in_fruit_salad
  (total_fruit : ℝ)
  (berries : ℝ)
  (h_total : total_fruit = 0.63)
  (h_berries : berries = 0.38) :
  total_fruit - berries = 0.25 := by
  sorry

end melon_amount_in_fruit_salad_l3346_334653


namespace cubic_equation_integer_roots_l3346_334633

theorem cubic_equation_integer_roots :
  ∃! p : ℝ, 
    (∃ x y z : ℕ+, 
      (5 * (x : ℝ)^3 - 5*(p+1)*(x : ℝ)^2 + (71*p - 1)*(x : ℝ) + 1 = 66*p) ∧
      (5 * (y : ℝ)^3 - 5*(p+1)*(y : ℝ)^2 + (71*p - 1)*(y : ℝ) + 1 = 66*p) ∧
      (5 * (z : ℝ)^3 - 5*(p+1)*(z : ℝ)^2 + (71*p - 1)*(z : ℝ) + 1 = 66*p) ∧
      x ≠ y ∧ y ≠ z ∧ x ≠ z) ∧
    p = 76 :=
by sorry

end cubic_equation_integer_roots_l3346_334633


namespace cube_volume_surface_area_l3346_334656

/-- Given a cube with volume 8x cubic units and surface area 2x square units, prove that x = 1728 -/
theorem cube_volume_surface_area (x : ℝ) : 
  (∃ (s : ℝ), s > 0 ∧ s^3 = 8*x ∧ 6*s^2 = 2*x) → x = 1728 := by
  sorry

end cube_volume_surface_area_l3346_334656


namespace max_areas_for_n_eq_one_l3346_334627

/-- Represents a circular disk divided by radii and secant lines -/
structure DividedDisk where
  n : ℕ
  radii : Fin (3 * n)
  secant_lines : Fin 2

/-- The maximum number of non-overlapping areas in a divided disk -/
def max_areas (disk : DividedDisk) : ℕ :=
  sorry

/-- Theorem stating the maximum number of non-overlapping areas for n = 1 -/
theorem max_areas_for_n_eq_one :
  ∀ (disk : DividedDisk),
    disk.n = 1 →
    max_areas disk = 15 :=
  sorry

end max_areas_for_n_eq_one_l3346_334627


namespace opposite_of_2023_l3346_334616

theorem opposite_of_2023 : 
  ∀ x : ℤ, x + 2023 = 0 ↔ x = -2023 := by
  sorry

end opposite_of_2023_l3346_334616


namespace length_of_AB_prime_l3346_334655

/-- Given points A, B, and C, where A' and B' are on the line y=x, and AC and BC intersect at C,
    prove that the length of A'B' is 10√2/11 -/
theorem length_of_AB_prime (A B C A' B' : ℝ × ℝ) : 
  A = (0, 10) →
  B = (0, 15) →
  C = (3, 7) →
  A'.1 = A'.2 →
  B'.1 = B'.2 →
  (C.2 - A.2) / (C.1 - A.1) = (A'.2 - A.2) / (A'.1 - A.1) →
  (C.2 - B.2) / (C.1 - B.1) = (B'.2 - B.2) / (B'.1 - B.1) →
  Real.sqrt ((A'.1 - B'.1)^2 + (A'.2 - B'.2)^2) = 10 * Real.sqrt 2 / 11 := by
  sorry

#check length_of_AB_prime

end length_of_AB_prime_l3346_334655


namespace alcohol_solution_percentage_l3346_334650

/-- Given a solution with initial volume and alcohol percentage, proves that adding pure alcohol to reach a target percentage results in the correct initial alcohol percentage. -/
theorem alcohol_solution_percentage
  (initial_volume : ℝ)
  (pure_alcohol_added : ℝ)
  (target_percentage : ℝ)
  (h1 : initial_volume = 6)
  (h2 : pure_alcohol_added = 3)
  (h3 : target_percentage = 0.5)
  : ∃ (initial_percentage : ℝ),
    initial_percentage * initial_volume + pure_alcohol_added =
    target_percentage * (initial_volume + pure_alcohol_added) ∧
    initial_percentage = 0.25 := by
  sorry

end alcohol_solution_percentage_l3346_334650


namespace prove_age_difference_l3346_334647

-- Define the given information
def wayne_age_2021 : ℕ := 37
def peter_age_diff : ℕ := 3
def julia_birth_year : ℕ := 1979

-- Define the current year
def current_year : ℕ := 2021

-- Define the age difference between Julia and Peter
def julia_peter_age_diff : ℕ := 2

-- Theorem to prove
theorem prove_age_difference :
  (current_year - wayne_age_2021 - peter_age_diff) - julia_birth_year = julia_peter_age_diff :=
by sorry

end prove_age_difference_l3346_334647
