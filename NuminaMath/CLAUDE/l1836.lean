import Mathlib

namespace min_value_sum_reciprocals_l1836_183670

theorem min_value_sum_reciprocals (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) 
  (sum_eq_two : x + y + z = 2) : 
  1 / (x + y) + 1 / (y + z) + 1 / (z + x) ≥ 9 / 4 := by
  sorry

end min_value_sum_reciprocals_l1836_183670


namespace sqrt_equation_solution_l1836_183646

theorem sqrt_equation_solution :
  ∃! z : ℝ, Real.sqrt (10 + 3 * z) = 14 :=
by
  sorry

end sqrt_equation_solution_l1836_183646


namespace total_monthly_payment_l1836_183654

/-- Calculates the total monthly payment for employees after new hires --/
theorem total_monthly_payment
  (initial_employees : ℕ)
  (hourly_rate : ℚ)
  (hours_per_day : ℕ)
  (days_per_week : ℕ)
  (weeks_per_month : ℕ)
  (additional_hires : ℕ)
  (h1 : initial_employees = 500)
  (h2 : hourly_rate = 12)
  (h3 : hours_per_day = 10)
  (h4 : days_per_week = 5)
  (h5 : weeks_per_month = 4)
  (h6 : additional_hires = 200) :
  (initial_employees + additional_hires) *
  (hourly_rate * hours_per_day * days_per_week * weeks_per_month) = 1680000 := by
  sorry

#eval (500 + 200) * (12 * 10 * 5 * 4)

end total_monthly_payment_l1836_183654


namespace quadratic_equation_solution_fractional_equation_solution_l1836_183660

-- Problem 1
theorem quadratic_equation_solution (x : ℝ) :
  x^2 + 6*x - 1 = 0 ↔ x = Real.sqrt 10 - 3 ∨ x = -Real.sqrt 10 - 3 :=
sorry

-- Problem 2
theorem fractional_equation_solution (x : ℝ) :
  x ≠ -2 ∧ x ≠ 1 →
  (x / (x + 2) = 2 / (x - 1) + 1 ↔ x = -1/2) :=
sorry

end quadratic_equation_solution_fractional_equation_solution_l1836_183660


namespace pick_shoes_five_pairs_l1836_183697

/-- The number of ways to pick 4 shoes from 5 pairs such that exactly one pair is among them -/
def pick_shoes (num_pairs : ℕ) : ℕ := 
  num_pairs * (Nat.choose (num_pairs - 1) 2) * 2 * 2

/-- Theorem stating that picking 4 shoes from 5 pairs with exactly one pair among them can be done in 120 ways -/
theorem pick_shoes_five_pairs : pick_shoes 5 = 120 := by
  sorry

end pick_shoes_five_pairs_l1836_183697


namespace intersection_of_A_and_B_l1836_183684

def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
def B : Set ℝ := {-1, 0, 2}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0} := by sorry

end intersection_of_A_and_B_l1836_183684


namespace eighth_term_of_specific_geometric_sequence_l1836_183624

-- Define a geometric sequence
def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a₁ * r ^ (n - 1)

-- Theorem statement
theorem eighth_term_of_specific_geometric_sequence :
  geometric_sequence 8 2 8 = 1024 := by
  sorry

end eighth_term_of_specific_geometric_sequence_l1836_183624


namespace percentage_difference_l1836_183647

theorem percentage_difference : (40 * 0.8) - (25 * (2/5)) = 22 := by
  sorry

end percentage_difference_l1836_183647


namespace q_sufficient_not_necessary_for_p_l1836_183629

-- Define the propositions p and q
def p (x : ℝ) : Prop := |x| < 2
def q (x : ℝ) : Prop := x^2 - x - 2 < 0

-- Theorem stating that q is sufficient but not necessary for p
theorem q_sufficient_not_necessary_for_p :
  (∀ x, q x → p x) ∧ ¬(∀ x, p x → q x) := by
  sorry

end q_sufficient_not_necessary_for_p_l1836_183629


namespace emily_cleaning_time_l1836_183609

/-- Represents the cleaning time distribution among four people -/
structure CleaningTime where
  total : ℝ
  lillyAndFiona : ℝ
  jack : ℝ
  emily : ℝ

/-- Theorem stating Emily's cleaning time in minutes -/
theorem emily_cleaning_time (ct : CleaningTime) : 
  ct.total = 8 ∧ 
  ct.lillyAndFiona = 1/4 * ct.total ∧ 
  ct.jack = 1/3 * ct.total ∧ 
  ct.emily = ct.total - ct.lillyAndFiona - ct.jack → 
  ct.emily * 60 = 200 := by
  sorry

#check emily_cleaning_time

end emily_cleaning_time_l1836_183609


namespace chess_tournament_l1836_183617

theorem chess_tournament (W M : ℕ) 
  (h1 : W * (W - 1) / 2 = 45)  -- Number of games with both women
  (h2 : W * M = 200)           -- Number of games with one man and one woman
  : M * (M - 1) / 2 = 190 :=   -- Number of games with both men
by
  sorry

end chess_tournament_l1836_183617


namespace weight_of_new_person_l1836_183615

/-- Calculates the weight of a new person in a group replacement scenario. -/
def newPersonWeight (groupSize : ℕ) (avgWeightIncrease : ℝ) (replacedPersonWeight : ℝ) : ℝ :=
  replacedPersonWeight + groupSize * avgWeightIncrease

/-- Proves that the weight of the new person is 108 kg given the specified conditions. -/
theorem weight_of_new_person :
  let groupSize : ℕ := 15
  let avgWeightIncrease : ℝ := 2.2
  let replacedPersonWeight : ℝ := 75
  newPersonWeight groupSize avgWeightIncrease replacedPersonWeight = 108 := by
  sorry

#eval newPersonWeight 15 2.2 75

end weight_of_new_person_l1836_183615


namespace coefficient_x2y3_in_binomial_expansion_l1836_183627

theorem coefficient_x2y3_in_binomial_expansion :
  (Finset.range 6).sum (fun k => (Nat.choose 5 k) * x^k * y^(5-k)) =
  10 * x^2 * y^3 + (Finset.range 6).sum (fun k => if k ≠ 2 then (Nat.choose 5 k) * x^k * y^(5-k) else 0) :=
by sorry

end coefficient_x2y3_in_binomial_expansion_l1836_183627


namespace no_77_cents_combination_l1836_183698

/-- Represents the set of available coin values in cents -/
def CoinValues : Set ℕ := {1, 5, 10, 50}

/-- Represents a selection of exactly three coins -/
def CoinSelection := Fin 3 → ℕ

/-- The sum of a coin selection -/
def sum_coins (selection : CoinSelection) : ℕ :=
  (selection 0) + (selection 1) + (selection 2)

/-- Predicate to check if a selection is valid (all coins are from CoinValues) -/
def valid_selection (selection : CoinSelection) : Prop :=
  ∀ i, selection i ∈ CoinValues

theorem no_77_cents_combination :
  ¬∃ (selection : CoinSelection), valid_selection selection ∧ sum_coins selection = 77 := by
  sorry

#check no_77_cents_combination

end no_77_cents_combination_l1836_183698


namespace probability_red_is_half_l1836_183664

def bag_contents : ℕ × ℕ := (3, 3)

def probability_red (contents : ℕ × ℕ) : ℚ :=
  contents.1 / (contents.1 + contents.2)

theorem probability_red_is_half : 
  probability_red bag_contents = 1/2 := by sorry

end probability_red_is_half_l1836_183664


namespace team_e_not_played_b_l1836_183620

/-- Represents a soccer team in the tournament -/
inductive Team : Type
  | A | B | C | D | E | F

/-- The number of matches played by each team at a certain point -/
def matches_played (t : Team) : ℕ :=
  match t with
  | Team.A => 5
  | Team.B => 4
  | Team.C => 3
  | Team.D => 2
  | Team.E => 1
  | Team.F => 0

/-- Predicate to check if two teams have played against each other -/
def has_played_against (t1 t2 : Team) : Prop :=
  sorry

/-- The total number of teams in the tournament -/
def total_teams : ℕ := 6

/-- The maximum number of matches a team can play in a round-robin tournament -/
def max_matches : ℕ := total_teams - 1

theorem team_e_not_played_b :
  matches_played Team.A = max_matches ∧
  matches_played Team.E = 1 →
  ¬ has_played_against Team.E Team.B :=
by sorry

end team_e_not_played_b_l1836_183620


namespace complement_of_union_l1836_183626

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {1, 3, 5}
def B : Set Nat := {3, 4, 5}

theorem complement_of_union :
  (U \ (A ∪ B)) = {2, 6} := by sorry

end complement_of_union_l1836_183626


namespace ship_passengers_asia_fraction_l1836_183653

theorem ship_passengers_asia_fraction (total : ℕ) 
  (north_america : ℚ) (europe : ℚ) (africa : ℚ) (other : ℕ) :
  total = 108 →
  north_america = 1 / 12 →
  europe = 1 / 4 →
  africa = 1 / 9 →
  other = 42 →
  (north_america + europe + africa + (other : ℚ) / total + 1 / 6 : ℚ) = 1 :=
by sorry

end ship_passengers_asia_fraction_l1836_183653


namespace train_speed_l1836_183634

/-- The speed of a train given its length and time to pass a stationary point. -/
theorem train_speed (length time : ℝ) (h1 : length = 300) (h2 : time = 6) :
  length / time = 50 := by
  sorry

end train_speed_l1836_183634


namespace paint_usage_l1836_183613

theorem paint_usage (initial_paint : ℝ) (first_week_fraction : ℝ) (second_week_fraction : ℝ)
  (h1 : initial_paint = 360)
  (h2 : first_week_fraction = 1/6)
  (h3 : second_week_fraction = 1/5) :
  let first_week_usage := first_week_fraction * initial_paint
  let remaining_paint := initial_paint - first_week_usage
  let second_week_usage := second_week_fraction * remaining_paint
  first_week_usage + second_week_usage = 120 := by
sorry

end paint_usage_l1836_183613


namespace touching_spheres_radii_l1836_183689

/-- Given four spheres of radius r, where each sphere touches the other three,
    calculate the radii of spheres that touch all four spheres internally and externally. -/
theorem touching_spheres_radii (r : ℝ) (r_pos : r > 0) :
  ∃ (p R : ℝ),
    (p = r * (Real.sqrt 6 / 2 - 1)) ∧
    (R = r * (Real.sqrt 6 / 2 + 1)) ∧
    (p > 0) ∧ (R > 0) :=
by sorry

end touching_spheres_radii_l1836_183689


namespace total_seashells_is_fifty_l1836_183636

/-- The number of seashells Tim found -/
def tim_seashells : ℕ := 37

/-- The number of seashells Sally found -/
def sally_seashells : ℕ := 13

/-- The total number of seashells found by Tim and Sally -/
def total_seashells : ℕ := tim_seashells + sally_seashells

/-- Theorem stating that the total number of seashells found is 50 -/
theorem total_seashells_is_fifty : total_seashells = 50 := by
  sorry

end total_seashells_is_fifty_l1836_183636


namespace scientific_notation_50300_l1836_183668

theorem scientific_notation_50300 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 50300 = a * (10 : ℝ) ^ n ∧ a = 5.03 ∧ n = 4 := by
  sorry

end scientific_notation_50300_l1836_183668


namespace coefficient_of_x_squared_l1836_183686

/-- The coefficient of x^2 in the expansion of (1+x+x^2)^6 -/
def a₂ : ℕ := (6 * (6 + 1)) / 2

/-- The expansion of (1+x+x^2)^6 -/
def expansion (x : ℝ) : ℝ := (1 + x + x^2)^6

theorem coefficient_of_x_squared :
  ∃ (f : ℝ → ℝ) (g : ℝ → ℝ),
    expansion = λ x => a₂ * x^2 + f x * x^3 + g x := by
  sorry

end coefficient_of_x_squared_l1836_183686


namespace triangle_abc_properties_l1836_183644

theorem triangle_abc_properties (A B C : ℝ) (a b c : ℝ) :
  -- Triangle conditions
  0 < A ∧ A < π ∧
  0 < B ∧ B < π ∧
  0 < C ∧ C < π ∧
  A + B + C = π ∧
  -- Given conditions
  a * Real.sin B = Real.sqrt 3 * b * Real.cos A ∧
  b = 3 ∧
  c = 2 →
  -- Conclusions
  A = π / 3 ∧ a = Real.sqrt 7 := by
  sorry

end triangle_abc_properties_l1836_183644


namespace cubic_function_property_l1836_183619

theorem cubic_function_property (a b : ℝ) :
  let f : ℝ → ℝ := λ x => a * x^3 + b * x - 4
  f 2 = 6 → f (-2) = -14 := by
  sorry

end cubic_function_property_l1836_183619


namespace mango_rate_calculation_l1836_183675

theorem mango_rate_calculation (grape_quantity : ℕ) (grape_rate : ℕ) (mango_quantity : ℕ) (total_paid : ℕ) :
  grape_quantity = 3 →
  grape_rate = 70 →
  mango_quantity = 9 →
  total_paid = 705 →
  (total_paid - grape_quantity * grape_rate) / mango_quantity = 55 := by
  sorry

end mango_rate_calculation_l1836_183675


namespace negation_existence_sufficient_not_necessary_sufficient_necessary_relationship_quadratic_inequality_condition_l1836_183693

-- 1. Negation of existence statement
theorem negation_existence : 
  (¬ ∃ x : ℝ, x ≥ 1 ∧ x^2 > 1) ↔ (∀ x : ℝ, x ≥ 1 → x^2 ≤ 1) := by sorry

-- 2. Sufficient but not necessary condition
theorem sufficient_not_necessary :
  (∃ x : ℝ, x ≠ 1 ∧ x^2 + 2*x - 3 = 0) ∧
  (∀ x : ℝ, x = 1 → x^2 + 2*x - 3 = 0) := by sorry

-- 3. Relationship between sufficient and necessary conditions
theorem sufficient_necessary_relationship (p q s : Prop) :
  ((p → q) ∧ (q → s)) → (p → s) := by sorry

-- 4. Conditions for quadratic inequality
theorem quadratic_inequality_condition (m : ℝ) :
  (¬ ∃ x : ℝ, m*x^2 + m*x + 1 < 0) → (0 ≤ m ∧ m ≤ 4) := by sorry

end negation_existence_sufficient_not_necessary_sufficient_necessary_relationship_quadratic_inequality_condition_l1836_183693


namespace helmet_safety_analysis_l1836_183633

/-- Data for people not wearing helmets over 4 years -/
def helmet_data : List (Nat × Nat) := [(1, 1250), (2, 1050), (3, 1000), (4, 900)]

/-- Contingency table for helmet wearing and casualties -/
def contingency_table : Matrix (Fin 2) (Fin 2) Nat :=
  ![![7, 3],
    ![13, 27]]

/-- Calculate the regression line equation coefficients -/
def regression_line (data : List (Nat × Nat)) : ℝ × ℝ :=
  sorry

/-- Estimate the number of people not wearing helmets for a given year -/
def estimate_no_helmet (coef : ℝ × ℝ) (year : Nat) : ℝ :=
  sorry

/-- Calculate the K^2 statistic for a 2x2 contingency table -/
def k_squared (table : Matrix (Fin 2) (Fin 2) Nat) : ℝ :=
  sorry

theorem helmet_safety_analysis :
  let (b, a) := regression_line helmet_data
  (b = -110 ∧ a = 1325) ∧
  estimate_no_helmet (b, a) 5 = 775 ∧
  k_squared contingency_table > 3.841 :=
sorry

end helmet_safety_analysis_l1836_183633


namespace ellipse_locus_theorem_l1836_183600

-- Define the ellipse
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define point A
def point_A (a : ℝ) : ℝ × ℝ := (a, 0)

-- Define point B
def point_B (a : ℝ) : ℝ × ℝ := (-a, 0)

-- Define a point P on the ellipse
def point_P (a b x y : ℝ) : Prop :=
  ellipse a b x y ∧ (x, y) ≠ point_A a ∧ (x, y) ≠ point_B a

-- Define the locus of M
def locus_M (a b x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / (a^2 / b)^2 = 1

-- Theorem statement
theorem ellipse_locus_theorem (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  ∀ x y : ℝ, point_P a b x y → ∃ m_x m_y : ℝ, locus_M a b m_x m_y :=
sorry

end ellipse_locus_theorem_l1836_183600


namespace dogwood_tree_count_l1836_183667

theorem dogwood_tree_count (current_trees new_trees : ℕ) 
  (h1 : current_trees = 34)
  (h2 : new_trees = 49) :
  current_trees + new_trees = 83 := by
  sorry

end dogwood_tree_count_l1836_183667


namespace min_value_quadratic_l1836_183614

theorem min_value_quadratic (x y : ℝ) :
  y = x^2 + 16*x + 20 → ∀ z : ℝ, y ≥ -44 ∧ (∃ x₀ : ℝ, x₀^2 + 16*x₀ + 20 = -44) :=
by sorry

end min_value_quadratic_l1836_183614


namespace difference_of_squares_l1836_183645

theorem difference_of_squares (x y : ℝ) (h_sum : x + y = 10) (h_diff : x - y = 19) :
  x^2 - y^2 = 190 := by sorry

end difference_of_squares_l1836_183645


namespace coefficient_x_plus_one_squared_in_x_to_tenth_l1836_183638

theorem coefficient_x_plus_one_squared_in_x_to_tenth : ∃ (a₀ a₁ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ),
  ∀ x : ℝ, x^10 = a₀ + a₁*(x+1) + 45*(x+1)^2 + a₃*(x+1)^3 + a₄*(x+1)^4 + 
            a₅*(x+1)^5 + a₆*(x+1)^6 + a₇*(x+1)^7 + a₈*(x+1)^8 + a₉*(x+1)^9 + a₁₀*(x+1)^10 :=
by sorry

end coefficient_x_plus_one_squared_in_x_to_tenth_l1836_183638


namespace first_solution_concentration_l1836_183642

-- Define the variables
def total_volume : ℝ := 630
def final_concentration : ℝ := 50
def first_solution_volume : ℝ := 420
def second_solution_concentration : ℝ := 30

-- Define the theorem
theorem first_solution_concentration :
  ∃ (x : ℝ),
    x * first_solution_volume / 100 +
    second_solution_concentration * (total_volume - first_solution_volume) / 100 =
    final_concentration * total_volume / 100 ∧
    x = 60 := by
  sorry

end first_solution_concentration_l1836_183642


namespace shirts_washed_l1836_183692

theorem shirts_washed (short_sleeve : ℕ) (long_sleeve : ℕ) (not_washed : ℕ) : 
  short_sleeve = 9 → long_sleeve = 27 → not_washed = 16 →
  short_sleeve + long_sleeve - not_washed = 20 := by
  sorry

end shirts_washed_l1836_183692


namespace retained_pits_problem_l1836_183659

/-- The maximum number of pits that can be retained on a road --/
def max_retained_pits (road_length : ℕ) (initial_spacing : ℕ) (revised_spacing : ℕ) : ℕ :=
  2 * (road_length / (initial_spacing * revised_spacing) + 1)

/-- Theorem stating the maximum number of retained pits for the given problem --/
theorem retained_pits_problem :
  max_retained_pits 120 3 5 = 18 := by
  sorry

end retained_pits_problem_l1836_183659


namespace next_three_same_calendar_years_l1836_183623

/-- A function that determines if a given year is a leap year -/
def isLeapYear (year : ℕ) : Prop :=
  (year % 4 = 0 ∧ year % 100 ≠ 0) ∨ year % 400 = 0

/-- The number of years between consecutive years with the same calendar -/
def calendarCycle : ℕ := 28

/-- The base year from which we start calculating -/
def baseYear : ℕ := 2024

/-- A function that calculates the nth year with the same calendar as the base year -/
def nthSameCalendarYear (n : ℕ) : ℕ :=
  baseYear + n * calendarCycle

/-- Theorem stating that the next three years following 2024 with the same calendar
    are 2052, 2080, and 2108 -/
theorem next_three_same_calendar_years :
  (nthSameCalendarYear 1 = 2052) ∧
  (nthSameCalendarYear 2 = 2080) ∧
  (nthSameCalendarYear 3 = 2108) ∧
  (isLeapYear baseYear) ∧
  (∀ n : ℕ, isLeapYear (nthSameCalendarYear n)) :=
sorry

end next_three_same_calendar_years_l1836_183623


namespace second_place_score_l1836_183685

/-- Represents a player in the chess tournament -/
structure Player where
  score : ℕ

/-- Represents a chess tournament -/
structure ChessTournament where
  players : Finset Player
  secondPlace : Player
  lastFour : Finset Player

/-- The rules and conditions of the tournament -/
def TournamentRules (t : ChessTournament) : Prop :=
  -- 8 players in total
  t.players.card = 8 ∧
  -- Second place player is in the set of all players
  t.secondPlace ∈ t.players ∧
  -- Last four players are in the set of all players
  t.lastFour ⊆ t.players ∧
  -- Last four players are distinct and have 4 members
  t.lastFour.card = 4 ∧
  -- All scores are different
  ∀ p1 p2 : Player, p1 ∈ t.players → p2 ∈ t.players → p1 ≠ p2 → p1.score ≠ p2.score ∧
  -- Second place score equals sum of last four scores
  t.secondPlace.score = (t.lastFour.toList.map Player.score).sum ∧
  -- Maximum possible score is 14
  ∀ p : Player, p ∈ t.players → p.score ≤ 14

/-- The main theorem -/
theorem second_place_score (t : ChessTournament) :
  TournamentRules t → t.secondPlace.score = 12 := by sorry

end second_place_score_l1836_183685


namespace heather_walk_distance_l1836_183604

/-- The distance Heather walked from the carnival rides back to the car -/
def carnival_to_car : ℝ := 0.08333333333333333

/-- The total distance Heather walked -/
def total_distance : ℝ := 0.75

/-- The distance from the car to the entrance (and from the entrance to the carnival rides) -/
def car_to_entrance : ℝ := 0.33333333333333335

theorem heather_walk_distance :
  2 * car_to_entrance + carnival_to_car = total_distance :=
by sorry

end heather_walk_distance_l1836_183604


namespace equation_equality_l1836_183655

theorem equation_equality : ∀ x y : ℝ, 9*x*y - 6*x*y = 3*x*y := by
  sorry

end equation_equality_l1836_183655


namespace min_ops_to_500_l1836_183687

def calculator_ops (n : ℕ) : ℕ → ℕ
| 0     => n
| (k+1) => calculator_ops (min (2*n) (n+1)) k

theorem min_ops_to_500 : ∃ k, calculator_ops 1 k = 500 ∧ ∀ j, j < k → calculator_ops 1 j ≠ 500 :=
  sorry

end min_ops_to_500_l1836_183687


namespace spade_equation_solution_l1836_183682

/-- Definition of the spade operation -/
def spade (A B : ℝ) : ℝ := 4 * A + 3 * B + 6

/-- Theorem stating that 9.5 is the unique solution to A ♠ 5 = 59 -/
theorem spade_equation_solution :
  ∃! A : ℝ, spade A 5 = 59 ∧ A = 9.5 := by sorry

end spade_equation_solution_l1836_183682


namespace equilateral_triangle_not_centrally_symmetric_l1836_183649

-- Define the shape type
inductive Shape
  | Parallelogram
  | LineSegment
  | EquilateralTriangle
  | Rhombus

-- Define the property of being centrally symmetric
def is_centrally_symmetric (s : Shape) : Prop :=
  match s with
  | Shape.Parallelogram => True
  | Shape.LineSegment => True
  | Shape.EquilateralTriangle => False
  | Shape.Rhombus => True

-- Theorem statement
theorem equilateral_triangle_not_centrally_symmetric :
  ∀ s : Shape, ¬(is_centrally_symmetric s) ↔ s = Shape.EquilateralTriangle :=
by sorry

end equilateral_triangle_not_centrally_symmetric_l1836_183649


namespace march_starts_on_friday_l1836_183691

/-- Represents the days of the week -/
inductive Weekday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a month with its properties -/
structure Month where
  days : Nat
  first_day : Weekday
  monday_count : Nat
  thursday_count : Nat

/-- The specific March we're considering -/
def march : Month :=
  { days := 31
  , first_day := Weekday.Friday  -- This is what we want to prove
  , monday_count := 5
  , thursday_count := 5 }

/-- Main theorem: If March has 31 days, 5 Mondays, and 5 Thursdays, then it starts on a Friday -/
theorem march_starts_on_friday :
  march.days = 31 ∧ march.monday_count = 5 ∧ march.thursday_count = 5 →
  march.first_day = Weekday.Friday :=
sorry

end march_starts_on_friday_l1836_183691


namespace pascal_triangle_element_l1836_183612

/-- The number of elements in the row of Pascal's triangle we're considering -/
def row_length : ℕ := 31

/-- The position of the number we're looking for (1-indexed) -/
def target_position : ℕ := 25

/-- The row number in Pascal's triangle (0-indexed) -/
def row_number : ℕ := row_length - 1

/-- The column number in Pascal's triangle (0-indexed) -/
def column_number : ℕ := target_position - 1

theorem pascal_triangle_element :
  Nat.choose row_number column_number = 593775 := by
  sorry

end pascal_triangle_element_l1836_183612


namespace unique_positive_solution_l1836_183610

theorem unique_positive_solution : ∃! (x : ℝ), x > 0 ∧ (x - 6) / 12 = 6 / (x - 12) := by
  sorry

end unique_positive_solution_l1836_183610


namespace ticket_sales_l1836_183694

theorem ticket_sales (total : ℕ) (full_price : ℕ) (reduced_price : ℕ) :
  total = 25200 →
  full_price = 16500 →
  full_price = 5 * reduced_price →
  reduced_price = 3300 := by
sorry

end ticket_sales_l1836_183694


namespace isosceles_trapezoid_right_angle_point_isosceles_trapezoid_point_distances_l1836_183607

/-- An isosceles trapezoid with bases a and b, and height h -/
structure IsoscelesTrapezoid where
  a : ℝ
  b : ℝ
  h : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  h_pos : 0 < h

/-- Point P on the axis of symmetry of the trapezoid -/
structure PointP (t : IsoscelesTrapezoid) where
  x : ℝ  -- Distance from P to one base
  y : ℝ  -- Distance from P to the other base
  sum_eq_h : x + y = t.h
  product_eq_ab_div_4 : x * y = t.a * t.b / 4

theorem isosceles_trapezoid_right_angle_point 
  (t : IsoscelesTrapezoid) : 
  (∃ p : PointP t, True) ↔ t.h^2 ≥ t.a * t.b :=
sorry

theorem isosceles_trapezoid_point_distances 
  (t : IsoscelesTrapezoid) 
  (h : t.h^2 ≥ t.a * t.b) :
  ∃ p : PointP t, 
    (p.x = (t.h + Real.sqrt (t.h^2 - t.a * t.b)) / 2 ∧ 
     p.y = (t.h - Real.sqrt (t.h^2 - t.a * t.b)) / 2) ∨
    (p.x = (t.h - Real.sqrt (t.h^2 - t.a * t.b)) / 2 ∧ 
     p.y = (t.h + Real.sqrt (t.h^2 - t.a * t.b)) / 2) :=
sorry

end isosceles_trapezoid_right_angle_point_isosceles_trapezoid_point_distances_l1836_183607


namespace freshman_count_l1836_183696

theorem freshman_count (total : ℕ) (f s j r : ℕ) : 
  total = 2158 →
  5 * s = 4 * f →
  8 * s = 7 * j →
  7 * j = 9 * r →
  total = f + s + j + r →
  f = 630 := by
sorry

end freshman_count_l1836_183696


namespace dishonest_dealer_profit_percentage_l1836_183695

/-- The profit percentage of a dishonest dealer who uses a weight of 600 grams per kg while selling at the professed cost price. -/
theorem dishonest_dealer_profit_percentage :
  let actual_weight : ℝ := 600  -- grams
  let claimed_weight : ℝ := 1000  -- grams (1 kg)
  let profit_ratio := (claimed_weight - actual_weight) / actual_weight
  profit_ratio * 100 = 200 / 3 :=
by sorry

end dishonest_dealer_profit_percentage_l1836_183695


namespace force_resultant_arithmetic_mean_l1836_183663

/-- Given two forces p₁ and p₂ forming an angle α, if their resultant is equal to their arithmetic mean, 
    then the angle α is between 120° and 180°, and the ratio of the forces is between 1/3 and 3. -/
theorem force_resultant_arithmetic_mean 
  (p₁ p₂ : ℝ) 
  (α : Real) 
  (h_positive : p₁ > 0 ∧ p₂ > 0) 
  (h_resultant : Real.sqrt (p₁^2 + p₂^2 + 2*p₁*p₂*(Real.cos α)) = (p₁ + p₂)/2) : 
  (2*π/3 ≤ α ∧ α ≤ π) ∧ (1/3 ≤ p₁/p₂ ∧ p₁/p₂ ≤ 3) := by
  sorry

end force_resultant_arithmetic_mean_l1836_183663


namespace quadratic_equation_solution_l1836_183643

theorem quadratic_equation_solution :
  ∀ x : ℝ, x^2 = x ↔ x = 0 ∨ x = 1 := by
  sorry

end quadratic_equation_solution_l1836_183643


namespace quadratic_transformation_l1836_183606

theorem quadratic_transformation (a b c : ℝ) (ha : a ≠ 0) :
  ∃ (h k r : ℝ) (hr : r ≠ 0), ∀ x : ℝ,
    a * x^2 + b * x + c = r^2 * ((x / r - h)^2) + k :=
by sorry

end quadratic_transformation_l1836_183606


namespace cos_alpha_value_l1836_183637

theorem cos_alpha_value (α : Real) 
  (h1 : α ∈ Set.Ioo 0 (π / 2))
  (h2 : Real.sin (α - π / 6) = 3 / 5) : 
  Real.cos α = (4 * Real.sqrt 3 - 3) / 10 := by
  sorry

end cos_alpha_value_l1836_183637


namespace function_existence_l1836_183605

theorem function_existence (k : ℤ) (hk : k ≠ 0) :
  ∃ f : ℤ → ℤ, ∀ a b : ℤ, k * (f (a + b)) + f (a * b) = f a * f b + k :=
by sorry

end function_existence_l1836_183605


namespace high_school_total_students_l1836_183650

/-- Represents a high school with three grades -/
structure HighSchool where
  freshman : ℕ
  sophomore : ℕ
  senior : ℕ

/-- Represents a stratified sample from the high school -/
structure StratifiedSample where
  freshman : ℕ
  sophomore : ℕ
  senior : ℕ
  total : ℕ

theorem high_school_total_students (hs : HighSchool) (sample : StratifiedSample) : 
  hs.senior = 1000 →
  sample.freshman = 75 →
  sample.sophomore = 60 →
  sample.total = 185 →
  hs.freshman + hs.sophomore + hs.senior = 3700 := by
  sorry

#check high_school_total_students

end high_school_total_students_l1836_183650


namespace cube_pyramid_volume_equality_l1836_183671

theorem cube_pyramid_volume_equality (h : ℝ) : 
  let cube_edge : ℝ := 6
  let pyramid_base : ℝ := 12
  let cube_volume : ℝ := cube_edge^3
  let pyramid_volume : ℝ := (1/3) * pyramid_base^2 * h
  cube_volume = pyramid_volume → h = 4.5 := by
sorry

end cube_pyramid_volume_equality_l1836_183671


namespace sufficient_not_necessary_condition_l1836_183639

open Real

theorem sufficient_not_necessary_condition : 
  (∀ α : ℝ, ∃ k : ℤ, α = π / 6 + 2 * k * π → cos (2 * α) = 1 / 2) ∧ 
  (∃ α : ℝ, cos (2 * α) = 1 / 2 ∧ ∀ k : ℤ, α ≠ π / 6 + 2 * k * π) := by
  sorry

end sufficient_not_necessary_condition_l1836_183639


namespace trapezoid_cd_length_l1836_183601

/-- Represents a trapezoid ABCD with specific properties -/
structure Trapezoid where
  -- Length of segment AB
  ab : ℝ
  -- Length of segment CD
  cd : ℝ
  -- The ratio of the area of triangle ABC to the area of triangle ADC is 5:3
  area_ratio : ab / cd = 5 / 3
  -- The sum of AB and CD is 192 cm
  sum_sides : ab + cd = 192

/-- 
Theorem: In a trapezoid ABCD, if the ratio of the area of triangle ABC to the area of triangle ADC 
is 5:3, and AB + CD = 192 cm, then the length of segment CD is 72 cm.
-/
theorem trapezoid_cd_length (t : Trapezoid) : t.cd = 72 := by
  sorry

end trapezoid_cd_length_l1836_183601


namespace cost_price_per_meter_l1836_183674

/-- The cost price of one meter of cloth given the selling price, quantity, and profit per meter -/
theorem cost_price_per_meter
  (selling_price : ℕ)
  (quantity : ℕ)
  (profit_per_meter : ℕ)
  (h1 : selling_price = 8925)
  (h2 : quantity = 85)
  (h3 : profit_per_meter = 20) :
  (selling_price - quantity * profit_per_meter) / quantity = 85 :=
by sorry

end cost_price_per_meter_l1836_183674


namespace ballsInBoxes_eq_36_l1836_183699

/-- The number of ways to distribute n indistinguishable objects into k distinguishable boxes -/
def starsAndBars (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of ways to distribute 7 indistinguishable balls into 3 distinguishable boxes -/
def ballsInBoxes : ℕ := starsAndBars 7 3

theorem ballsInBoxes_eq_36 : ballsInBoxes = 36 := by
  sorry

end ballsInBoxes_eq_36_l1836_183699


namespace nested_bracket_equals_two_l1836_183677

/-- Define the operation [a, b, c] as (a + b) / c, where c ≠ 0 -/
def bracket (a b c : ℚ) : ℚ :=
  if c ≠ 0 then (a + b) / c else 0

/-- The main theorem to prove -/
theorem nested_bracket_equals_two :
  bracket (bracket 50 50 100) (bracket 3 6 9) (bracket 20 30 50) = 2 := by
  sorry

end nested_bracket_equals_two_l1836_183677


namespace square_circle_perimeter_l1836_183673

/-- Given a square with perimeter 28 cm and a circle with radius equal to the side of the square,
    the perimeter of the circle is 14π cm. -/
theorem square_circle_perimeter : 
  ∀ (square_side circle_radius : ℝ),
    square_side * 4 = 28 →
    circle_radius = square_side →
    2 * Real.pi * circle_radius = 14 * Real.pi := by
  sorry

end square_circle_perimeter_l1836_183673


namespace three_Y_two_equals_one_l1836_183602

-- Define the Y operation
def Y (a b : ℝ) : ℝ := a^2 - 2*a*b + b^2

-- Theorem statement
theorem three_Y_two_equals_one : Y 3 2 = 1 := by sorry

end three_Y_two_equals_one_l1836_183602


namespace factorization_difference_l1836_183648

theorem factorization_difference (y : ℝ) (a b : ℤ) : 
  (5 * y^2 + 17 * y + 6 = (5 * y + a) * (y + b)) → (a - b = -1) := by
  sorry

end factorization_difference_l1836_183648


namespace binomial_coefficient_problem_l1836_183688

theorem binomial_coefficient_problem (m : ℝ) (n : ℕ) :
  (∃ k : ℕ, (Nat.choose n 3) * m^3 = 160 ∧ n = 6) → m = 2 := by
  sorry

end binomial_coefficient_problem_l1836_183688


namespace local_politics_coverage_l1836_183652

/-- The percentage of reporters covering politics -/
def politics_coverage : ℝ := 100 - 92.85714285714286

/-- The percentage of reporters covering local politics among those covering politics -/
def local_coverage_ratio : ℝ := 100 - 30

theorem local_politics_coverage :
  (local_coverage_ratio * politics_coverage / 100) = 5 := by sorry

end local_politics_coverage_l1836_183652


namespace original_number_proof_l1836_183625

theorem original_number_proof (q : ℝ) : 
  (q + 0.125 * q) - (q - 0.25 * q) = 30 → q = 80 :=
by sorry

end original_number_proof_l1836_183625


namespace power_three_mod_eleven_l1836_183628

theorem power_three_mod_eleven : 3^221 % 11 = 3 := by
  sorry

end power_three_mod_eleven_l1836_183628


namespace line_through_point_l1836_183676

/-- The value of b for which the line bx + (b-1)y = b+3 passes through the point (3, -7) -/
theorem line_through_point (b : ℚ) : 
  (b * 3 + (b - 1) * (-7) = b + 3) → b = 4/5 := by
  sorry

end line_through_point_l1836_183676


namespace ellipse_sum_range_l1836_183641

theorem ellipse_sum_range (x y : ℝ) (h : x^2/16 + y^2/9 = 1) :
  ∃ (z : ℝ), z = x + y ∧ -5 ≤ z ∧ z ≤ 5 :=
sorry

end ellipse_sum_range_l1836_183641


namespace volume_formula_l1836_183608

/-- A right rectangular prism with edge lengths 2, 3, and 5 -/
structure Prism where
  length : ℝ := 2
  width : ℝ := 3
  height : ℝ := 5

/-- The set of points within distance r of the prism -/
def S (B : Prism) (r : ℝ) : Set (ℝ × ℝ × ℝ) := sorry

/-- The volume of S(r) -/
noncomputable def volume (B : Prism) (r : ℝ) : ℝ := sorry

/-- Coefficients of the volume polynomial -/
structure VolumeCoeffs where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

theorem volume_formula (B : Prism) (coeffs : VolumeCoeffs) :
  (∀ r : ℝ, volume B r = coeffs.a * r^3 + coeffs.b * r^2 + coeffs.c * r + coeffs.d) →
  coeffs.a > 0 ∧ coeffs.b > 0 ∧ coeffs.c > 0 ∧ coeffs.d > 0 →
  coeffs.b * coeffs.c / (coeffs.a * coeffs.d) = 20.67 := by
  sorry

end volume_formula_l1836_183608


namespace triangle_coloring_theorem_l1836_183635

/-- The number of ways to color 6 circles in a fixed triangular arrangement with 4 blue, 1 green, and 1 red circle -/
def triangle_coloring_ways : ℕ := 30

/-- Theorem stating that the number of ways to color the triangular arrangement is 30 -/
theorem triangle_coloring_theorem : triangle_coloring_ways = 30 := by
  sorry

end triangle_coloring_theorem_l1836_183635


namespace first_digit_1025_base12_l1836_183656

/-- The first digit of a number in a given base -/
def firstDigitInBase (n : ℕ) (base : ℕ) : ℕ :=
  sorry

/-- Theorem: The first digit of 1025 (base 10) in base 12 is 7 -/
theorem first_digit_1025_base12 : firstDigitInBase 1025 12 = 7 := by
  sorry

end first_digit_1025_base12_l1836_183656


namespace root_equation_problem_l1836_183618

theorem root_equation_problem (c d : ℝ) : 
  (∃! x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    ((x + c) * (x + d) * (x + 10)) / ((x + 5)^2) = 0 ∧
    ((y + c) * (y + d) * (y + 10)) / ((y + 5)^2) = 0 ∧
    ((z + c) * (z + d) * (z + 10)) / ((z + 5)^2) = 0) ∧
  (∃! w : ℝ, ((w + 3*c) * (w + 2) * (w + 4)) / ((w + d) * (w + 10)) = 0) →
  50 * c + 10 * d = 310 / 3 := by
sorry

end root_equation_problem_l1836_183618


namespace y_equals_x_cubed_l1836_183631

/-- Represents a pair of x and y values from the table -/
structure XYPair where
  x : ℕ
  y : ℕ

/-- The set of (x, y) pairs from the given table -/
def xyTable : List XYPair := [
  ⟨1, 1⟩,
  ⟨2, 8⟩,
  ⟨3, 27⟩,
  ⟨4, 64⟩,
  ⟨5, 125⟩
]

/-- Theorem stating that y = x^3 holds for all pairs in the table -/
theorem y_equals_x_cubed (pair : XYPair) (h : pair ∈ xyTable) : pair.y = pair.x ^ 3 := by
  sorry

end y_equals_x_cubed_l1836_183631


namespace divides_totient_power_two_minus_one_l1836_183679

theorem divides_totient_power_two_minus_one (n : ℕ) (hn : n > 0) : 
  n ∣ Nat.totient (2^n - 1) := by
  sorry

end divides_totient_power_two_minus_one_l1836_183679


namespace remainder_problem_l1836_183665

theorem remainder_problem (N : ℕ) (R : ℕ) (h1 : R < 100) (h2 : ∃ k : ℕ, N = 100 * k + R) (h3 : ∃ m : ℕ, N = R * m + 1) : R = 1 := by
  sorry

end remainder_problem_l1836_183665


namespace congruence_problem_l1836_183683

theorem congruence_problem (n : ℤ) : 
  0 ≤ n ∧ n < 25 ∧ -150 ≡ n [ZMOD 25] → n = 0 := by
  sorry

end congruence_problem_l1836_183683


namespace ship_passengers_heads_l1836_183672

/-- Represents the number of heads and legs on a ship with cats, crew, and a one-legged captain. -/
structure ShipPassengers where
  cats : ℕ
  crew : ℕ
  captain : ℕ := 1

/-- Calculates the total number of heads on the ship. -/
def totalHeads (p : ShipPassengers) : ℕ :=
  p.cats + p.crew + p.captain

/-- Calculates the total number of legs on the ship. -/
def totalLegs (p : ShipPassengers) : ℕ :=
  p.cats * 4 + p.crew * 2 + 1

/-- Theorem stating that given the conditions, the total number of heads on the ship is 14. -/
theorem ship_passengers_heads :
  ∃ (p : ShipPassengers),
    p.cats = 7 ∧
    totalLegs p = 41 ∧
    totalHeads p = 14 :=
sorry

end ship_passengers_heads_l1836_183672


namespace imaginary_number_theorem_l1836_183662

theorem imaginary_number_theorem (z : ℂ) :
  (∃ a : ℝ, z = a * I) →
  ((z + 2) / (1 - I)).im = 0 →
  z = -2 * I :=
by sorry

end imaginary_number_theorem_l1836_183662


namespace negative_one_to_zero_power_l1836_183678

theorem negative_one_to_zero_power : ((-1 : ℤ) ^ (0 : ℕ)) = 1 := by
  sorry

end negative_one_to_zero_power_l1836_183678


namespace smallest_base_perfect_square_l1836_183661

/-- 
Given a base b > 4, returns the value of 45 in base b expressed in decimal.
-/
def base_b_to_decimal (b : ℕ) : ℕ := 4 * b + 5

/-- 
Proposition: 5 is the smallest integer b > 4 for which 45_b is a perfect square.
-/
theorem smallest_base_perfect_square : 
  (∀ b : ℕ, b > 4 ∧ b < 5 → ¬ ∃ k : ℕ, base_b_to_decimal b = k ^ 2) ∧
  (∃ k : ℕ, base_b_to_decimal 5 = k ^ 2) := by
  sorry

end smallest_base_perfect_square_l1836_183661


namespace increase_by_percentage_increase_800_by_110_percent_l1836_183690

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) : 
  initial * (1 + percentage / 100) = initial + initial * (percentage / 100) := by sorry

theorem increase_800_by_110_percent : 
  800 * (1 + 110 / 100) = 1680 := by sorry

end increase_by_percentage_increase_800_by_110_percent_l1836_183690


namespace lattice_points_5_11_to_35_221_l1836_183622

/-- The number of lattice points on a line segment --/
def lattice_points_on_segment (x1 y1 x2 y2 : ℤ) : ℕ :=
  sorry

/-- Theorem: The number of lattice points on the line segment from (5, 11) to (35, 221) is 31 --/
theorem lattice_points_5_11_to_35_221 :
  lattice_points_on_segment 5 11 35 221 = 31 := by
  sorry

end lattice_points_5_11_to_35_221_l1836_183622


namespace triangle_properties_l1836_183651

theorem triangle_properties (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  (A + B + C = π) →
  -- Side lengths are positive
  (a > 0 ∧ b > 0 ∧ c > 0) →
  -- Law of sines
  (a / Real.sin A = b / Real.sin B) →
  -- Law of cosines
  (b^2 = a^2 + c^2 - 2*a*c*Real.cos B) →
  -- Prove the three properties
  ((A > B ↔ Real.sin A > Real.sin B) ∧
   (B = π/3 ∧ b^2 = a*c → A = π/3 ∧ B = π/3 ∧ C = π/3) ∧
   (b = a * Real.cos C + c * Real.sin A → A = π/4)) :=
by sorry

end triangle_properties_l1836_183651


namespace exists_valid_heptagon_arrangement_l1836_183657

/-- Represents a heptagon with numbers placed in its vertices -/
def Heptagon := Fin 7 → Nat

/-- Checks if a given heptagon arrangement satisfies the sum condition -/
def is_valid_arrangement (h : Heptagon) : Prop :=
  (∀ i : Fin 7, h i ∈ Finset.range 15 \ {0}) ∧
  (∀ i : Fin 7, h i + h ((i + 1) % 7) + h ((i + 2) % 7) = 19)

/-- Theorem stating the existence of a valid heptagon arrangement -/
theorem exists_valid_heptagon_arrangement : ∃ h : Heptagon, is_valid_arrangement h :=
sorry

end exists_valid_heptagon_arrangement_l1836_183657


namespace triangle_perimeter_range_l1836_183630

/-- Given a triangle ABC with side AC of length 2 and satisfying the equation
    √3 tan A tan C = tan A + tan C + √3, its perimeter is in (4, 2 + 2√3) ∪ (2 + 2√3, 6] -/
theorem triangle_perimeter_range (A C : Real) (hAC : Real) :
  hAC = 2 →
  Real.sqrt 3 * Real.tan A * Real.tan C = Real.tan A + Real.tan C + Real.sqrt 3 →
  ∃ (p : Real), p ∈ Set.union (Set.Ioo 4 (2 + 2 * Real.sqrt 3)) (Set.Ioc (2 + 2 * Real.sqrt 3) 6) ∧
                p = hAC + 2 * Real.sin (A + π / 6) + 2 * Real.sin (C + π / 6) :=
by sorry

end triangle_perimeter_range_l1836_183630


namespace total_games_is_62_l1836_183680

/-- Represents a baseball league with its characteristics and calculates the total number of games played -/
structure BaseballLeague where
  teams : Nat
  games_per_team_per_month : Nat
  season_months : Nat
  playoff_rounds : Nat
  games_per_playoff_round : Nat

/-- Calculates the total number of games played in the season, including playoffs -/
def BaseballLeague.total_games (league : BaseballLeague) : Nat :=
  let regular_season_games := (league.teams / 2) * league.games_per_team_per_month * league.season_months
  let playoff_games := league.playoff_rounds * league.games_per_playoff_round
  regular_season_games + playoff_games

/-- The specific baseball league described in the problem -/
def specific_league : BaseballLeague :=
  { teams := 8
  , games_per_team_per_month := 7
  , season_months := 2
  , playoff_rounds := 3
  , games_per_playoff_round := 2
  }

/-- Theorem stating that the total number of games in the specific league is 62 -/
theorem total_games_is_62 : specific_league.total_games = 62 := by
  sorry


end total_games_is_62_l1836_183680


namespace unique_three_digit_integer_l1836_183669

theorem unique_three_digit_integer : ∃! n : ℕ,
  100 ≤ n ∧ n < 1000 ∧
  n % 7 = 3 ∧
  n % 8 = 4 ∧
  n % 13 = 8 :=
by
  sorry

end unique_three_digit_integer_l1836_183669


namespace cubic_roots_sum_of_cubes_reciprocal_l1836_183616

theorem cubic_roots_sum_of_cubes_reciprocal (a b c d r s : ℝ) :
  a ≠ 0 →
  c ≠ 0 →
  a * r^3 + b * r^2 + c * r + d = 0 →
  a * s^3 + b * s^2 + c * s + d = 0 →
  r ≠ 0 →
  s ≠ 0 →
  (1 / r^3) + (1 / s^3) = (b^3 - 3 * a * b * c) / c^3 :=
by sorry

end cubic_roots_sum_of_cubes_reciprocal_l1836_183616


namespace exhibit_fish_count_l1836_183658

/-- The number of pufferfish in the exhibit -/
def num_pufferfish : ℕ := 15

/-- The ratio of swordfish to pufferfish -/
def swordfish_ratio : ℕ := 5

/-- The total number of fish in the exhibit -/
def total_fish : ℕ := num_pufferfish + swordfish_ratio * num_pufferfish

theorem exhibit_fish_count : total_fish = 90 := by
  sorry

end exhibit_fish_count_l1836_183658


namespace log_problem_l1836_183666

-- Define the logarithm function
noncomputable def log : ℝ → ℝ := Real.log

-- Define the conditions and the theorem
theorem log_problem (x y : ℝ) (h1 : log (x * y^5) = 1) (h2 : log (x^3 * y) = 1) :
  log (x^2 * y^2) = 6/7 := by
  sorry

end log_problem_l1836_183666


namespace thomas_total_training_hours_l1836_183640

/-- Calculates the total training hours for Thomas given his training schedule --/
def total_training_hours : ℕ :=
  let first_phase_days : ℕ := 15
  let first_phase_hours_per_day : ℕ := 5
  let second_phase_days : ℕ := 15
  let second_phase_rest_days : ℕ := 3
  let third_phase_days : ℕ := 12
  let third_phase_rest_days : ℕ := 2
  let new_schedule_morning_hours : ℕ := 4
  let new_schedule_evening_hours : ℕ := 3

  let first_phase_total := first_phase_days * first_phase_hours_per_day
  let second_phase_total := (second_phase_days - second_phase_rest_days) * (new_schedule_morning_hours + new_schedule_evening_hours)
  let third_phase_total := (third_phase_days - third_phase_rest_days) * (new_schedule_morning_hours + new_schedule_evening_hours)

  first_phase_total + second_phase_total + third_phase_total

/-- Theorem stating that Thomas' total training hours is 229 --/
theorem thomas_total_training_hours :
  total_training_hours = 229 := by
  sorry

end thomas_total_training_hours_l1836_183640


namespace vector_scalar_mult_and_add_l1836_183681

theorem vector_scalar_mult_and_add :
  (3 : ℝ) • ((-3 : ℝ), (2 : ℝ), (-5 : ℝ)) + ((1 : ℝ), (7 : ℝ), (-3 : ℝ)) = ((-8 : ℝ), (13 : ℝ), (-18 : ℝ)) := by
  sorry

end vector_scalar_mult_and_add_l1836_183681


namespace max_product_sum_2020_l1836_183603

theorem max_product_sum_2020 : 
  (∃ (a b : ℤ), a + b = 2020 ∧ a * b = 1020100) ∧ 
  (∀ (x y : ℤ), x + y = 2020 → x * y ≤ 1020100) := by
sorry

end max_product_sum_2020_l1836_183603


namespace solution_set_implies_a_value_solution_set_all_reals_implies_a_range_l1836_183611

-- Part 1
theorem solution_set_implies_a_value (a : ℝ) :
  (∀ x : ℝ, ax^2 - 2*a*x + 3 > 0 ↔ -1 < x ∧ x < 3) →
  a = -1 :=
sorry

-- Part 2
theorem solution_set_all_reals_implies_a_range (a : ℝ) :
  (∀ x : ℝ, ax^2 - 2*a*x + 3 > 0) →
  0 ≤ a ∧ a < 3 :=
sorry

end solution_set_implies_a_value_solution_set_all_reals_implies_a_range_l1836_183611


namespace leila_money_left_l1836_183621

def money_left_after_shopping (initial_money sweater_cost jewelry_cost : ℕ) : ℕ :=
  initial_money - (sweater_cost + jewelry_cost)

theorem leila_money_left :
  ∀ (sweater_cost : ℕ),
    sweater_cost = 40 →
    ∀ (initial_money : ℕ),
      initial_money = 4 * sweater_cost →
      ∀ (jewelry_cost : ℕ),
        jewelry_cost = sweater_cost + 60 →
        money_left_after_shopping initial_money sweater_cost jewelry_cost = 20 :=
by
  sorry

end leila_money_left_l1836_183621


namespace perception_arrangements_l1836_183632

def word : String := "PERCEPTION"

theorem perception_arrangements :
  (word.length = 10) →
  (word.count 'P' = 2) →
  (word.count 'E' = 2) →
  (word.count 'I' = 2) →
  (word.count 'C' = 1) →
  (word.count 'T' = 1) →
  (word.count 'O' = 1) →
  (word.count 'N' = 1) →
  (Nat.factorial 10 / (Nat.factorial 2 * Nat.factorial 2 * Nat.factorial 2) = 453600) :=
by
  sorry

end perception_arrangements_l1836_183632
