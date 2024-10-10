import Mathlib

namespace decagon_ratio_l3593_359348

/-- A decagon made up of unit squares with specific properties -/
structure Decagon where
  /-- The total number of unit squares in the decagon -/
  num_squares : ℕ
  /-- The total area of the decagon in square units -/
  total_area : ℝ
  /-- LZ is a line segment intersecting the left and right vertices of the decagon -/
  lz : ℝ × ℝ
  /-- XZ is a segment from LZ to a vertex -/
  xz : ℝ
  /-- ZY is a segment from LZ to another vertex -/
  zy : ℝ
  /-- The number of unit squares is 12 -/
  h_num_squares : num_squares = 12
  /-- The total area is 12 square units -/
  h_total_area : total_area = 12
  /-- LZ bisects the area of the decagon -/
  h_bisects : lz.1 = total_area / 2

/-- The ratio of XZ to ZY is 1 -/
theorem decagon_ratio (d : Decagon) : d.xz / d.zy = 1 := by
  sorry


end decagon_ratio_l3593_359348


namespace drawn_games_in_specific_tournament_l3593_359305

/-- Represents a chess tournament. -/
structure ChessTournament where
  participants : Nat
  total_matches : Nat
  wins_per_participant : Nat
  has_growing_lists : Bool

/-- Calculates the number of drawn games in a chess tournament. -/
def drawn_games (tournament : ChessTournament) : Nat :=
  tournament.total_matches - (tournament.participants * tournament.wins_per_participant)

/-- Theorem stating the number of drawn games in the specific tournament. -/
theorem drawn_games_in_specific_tournament :
  ∀ (t : ChessTournament),
    t.participants = 12 ∧
    t.total_matches = (12 * 11) / 2 ∧
    t.wins_per_participant = 1 ∧
    t.has_growing_lists = true →
    drawn_games t = 54 := by
  sorry

end drawn_games_in_specific_tournament_l3593_359305


namespace sin_pi_12_function_value_l3593_359302

theorem sin_pi_12_function_value
  (f : ℝ → ℝ)
  (h : ∀ x, f (Real.cos x) = Real.cos (2 * x)) :
  f (Real.sin (π / 12)) = -Real.sqrt 3 / 2 :=
sorry

end sin_pi_12_function_value_l3593_359302


namespace travis_cereal_cost_l3593_359386

/-- The amount Travis spends on cereal in a year -/
def cereal_cost (boxes_per_week : ℕ) (cost_per_box : ℚ) (weeks_per_year : ℕ) : ℚ :=
  (boxes_per_week : ℚ) * cost_per_box * (weeks_per_year : ℚ)

/-- Proof that Travis spends $312.00 on cereal in a year -/
theorem travis_cereal_cost :
  cereal_cost 2 3 52 = 312 := by
  sorry

end travis_cereal_cost_l3593_359386


namespace framing_for_enlarged_picture_l3593_359380

/-- Calculates the minimum number of linear feet of framing needed for an enlarged and bordered picture. -/
def min_framing_feet (orig_width orig_height enlarge_factor border_width : ℕ) : ℕ :=
  let enlarged_width := orig_width * enlarge_factor
  let enlarged_height := orig_height * enlarge_factor
  let framed_width := enlarged_width + 2 * border_width
  let framed_height := enlarged_height + 2 * border_width
  let perimeter_inches := 2 * (framed_width + framed_height)
  (perimeter_inches + 11) / 12  -- Round up to the nearest foot

/-- Theorem stating that for the given picture dimensions and specifications, 10 feet of framing is needed. -/
theorem framing_for_enlarged_picture :
  min_framing_feet 5 7 4 3 = 10 := by
  sorry

end framing_for_enlarged_picture_l3593_359380


namespace matrix_N_satisfies_conditions_l3593_359337

def N : Matrix (Fin 4) (Fin 4) ℝ := !![3, -1, 8, 1; 4, 6, -2, 0; -9, -3, 5, 7; 1, 2, 0, -1]

def i : Fin 4 → ℝ := ![1, 0, 0, 0]
def j : Fin 4 → ℝ := ![0, 1, 0, 0]
def k : Fin 4 → ℝ := ![0, 0, 1, 0]
def l : Fin 4 → ℝ := ![0, 0, 0, 1]

theorem matrix_N_satisfies_conditions :
  N.mulVec i = ![3, 4, -9, 1] ∧
  N.mulVec j = ![-1, 6, -3, 2] ∧
  N.mulVec k = ![8, -2, 5, 0] ∧
  N.mulVec l = ![1, 0, 7, -1] := by
  sorry

end matrix_N_satisfies_conditions_l3593_359337


namespace intersection_implies_a_zero_l3593_359379

def set_A (a : ℝ) : Set ℝ := {a^2, a+1, -1}
def set_B (a : ℝ) : Set ℝ := {2*a-1, |a-2|, 3*a^2+4}

theorem intersection_implies_a_zero (a : ℝ) :
  set_A a ∩ set_B a = {-1} → a = 0 := by
sorry

end intersection_implies_a_zero_l3593_359379


namespace original_price_calculation_l3593_359323

theorem original_price_calculation (decreased_price : ℝ) (decrease_percentage : ℝ) 
  (h1 : decreased_price = 1064)
  (h2 : decrease_percentage = 24) : 
  decreased_price / (1 - decrease_percentage / 100) = 1400 := by
  sorry

end original_price_calculation_l3593_359323


namespace lisa_likes_one_last_digit_l3593_359309

def divisible_by_5 (n : ℕ) : Prop := n % 5 = 0

def divisible_by_2 (n : ℕ) : Prop := n % 2 = 0

def last_digit (n : ℕ) : ℕ := n % 10

theorem lisa_likes_one_last_digit :
  ∃! d : ℕ, d < 10 ∧ ∀ n : ℕ, last_digit n = d → (divisible_by_5 n ∧ divisible_by_2 n) :=
by
  sorry

end lisa_likes_one_last_digit_l3593_359309


namespace average_of_four_l3593_359368

theorem average_of_four (total : ℕ) (avg_all : ℚ) (avg_two : ℚ) :
  total = 6 →
  avg_all = 8 →
  avg_two = 14 →
  (total * avg_all - 2 * avg_two) / (total - 2) = 5 :=
by
  sorry

end average_of_four_l3593_359368


namespace simplify_and_evaluate_l3593_359391

theorem simplify_and_evaluate (a b : ℝ) (h : a = -b) :
  2 * (3 * a^2 + a - 2*b) - 6 * (a^2 - b) = 0 := by
  sorry

end simplify_and_evaluate_l3593_359391


namespace intersection_of_M_and_N_l3593_359343

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | -1 < x ∧ x < 3}
def N : Set ℝ := {x : ℝ | -2 < x ∧ x < 1}

-- Theorem statement
theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | -1 < x ∧ x < 1} := by sorry

end intersection_of_M_and_N_l3593_359343


namespace probability_sum_multiple_of_three_l3593_359328

/-- The type representing the possible outcomes of rolling a standard 6-sided die. -/
inductive Die : Type
  | one | two | three | four | five | six

/-- The function that returns the numeric value of a die roll. -/
def dieValue : Die → Nat
  | Die.one => 1
  | Die.two => 2
  | Die.three => 3
  | Die.four => 4
  | Die.five => 5
  | Die.six => 6

/-- The type representing the outcome of rolling two dice. -/
def TwoDiceRoll : Type := Die × Die

/-- The function that calculates the sum of two dice rolls. -/
def rollSum (roll : TwoDiceRoll) : Nat :=
  dieValue roll.1 + dieValue roll.2

/-- The predicate that checks if a number is a multiple of 3. -/
def isMultipleOfThree (n : Nat) : Prop :=
  ∃ k, n = 3 * k

/-- The set of all possible outcomes when rolling two dice. -/
def allOutcomes : Finset TwoDiceRoll :=
  sorry

/-- The set of outcomes where the sum is a multiple of 3. -/
def favorableOutcomes : Finset TwoDiceRoll :=
  sorry

theorem probability_sum_multiple_of_three :
  (favorableOutcomes.card : ℚ) / (allOutcomes.card : ℚ) = 1 / 3 :=
sorry

end probability_sum_multiple_of_three_l3593_359328


namespace square_perimeter_l3593_359334

/-- Theorem: A square with an area of 625 cm² has a perimeter of 100 cm. -/
theorem square_perimeter (s : ℝ) (h_area : s^2 = 625) : 4 * s = 100 := by
  sorry

end square_perimeter_l3593_359334


namespace polynomial_value_theorem_l3593_359393

-- Define the polynomial function g(x)
def g (p q r s : ℝ) (x : ℝ) : ℝ := p * x^3 + q * x^2 + r * x + s

-- Theorem statement
theorem polynomial_value_theorem (p q r s : ℝ) :
  g p q r s (-1) = 4 → 6*p - 3*q + r - 2*s = -24 := by
  sorry

end polynomial_value_theorem_l3593_359393


namespace triangle_inequality_l3593_359372

/-- For a triangle with side lengths a, b, and c, area 1/4, and circumradius 1,
    √a + √b + √c < 1/a + 1/b + 1/c -/
theorem triangle_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (harea : a * b * c / 4 = 1/4) (hcircum : a * b * c = 1) :
  Real.sqrt a + Real.sqrt b + Real.sqrt c < 1/a + 1/b + 1/c := by
  sorry

end triangle_inequality_l3593_359372


namespace suit_cost_problem_l3593_359306

theorem suit_cost_problem (x : ℝ) (h1 : x + (3 * x + 200) = 1400) : x = 300 := by
  sorry

end suit_cost_problem_l3593_359306


namespace correct_calculation_l3593_359341

def correct_sum (mistaken_sum : ℕ) (original_tens : ℕ) (mistaken_tens : ℕ) 
                (original_units : ℕ) (mistaken_units : ℕ) : ℕ :=
  mistaken_sum - (mistaken_units - original_units) + (original_tens - mistaken_tens) * 10

theorem correct_calculation (mistaken_sum : ℕ) (original_tens : ℕ) (mistaken_tens : ℕ) 
                            (original_units : ℕ) (mistaken_units : ℕ) : 
  mistaken_sum = 111 ∧ 
  original_tens = 7 ∧ 
  mistaken_tens = 4 ∧ 
  original_units = 5 ∧ 
  mistaken_units = 8 → 
  correct_sum mistaken_sum original_tens mistaken_tens original_units mistaken_units = 138 := by
  sorry

#eval correct_sum 111 7 4 5 8

end correct_calculation_l3593_359341


namespace total_pages_is_62_l3593_359377

/-- The number of pages Jairus read -/
def jairus_pages : ℕ := 20

/-- The number of pages Arniel read -/
def arniel_pages : ℕ := 2 * jairus_pages + 2

/-- The total number of pages read by Jairus and Arniel -/
def total_pages : ℕ := jairus_pages + arniel_pages

/-- Theorem stating that the total number of pages read is 62 -/
theorem total_pages_is_62 : total_pages = 62 := by
  sorry

end total_pages_is_62_l3593_359377


namespace total_apples_to_pack_l3593_359397

/-- The number of apples in one dozen -/
def apples_per_dozen : ℕ := 12

/-- The number of boxes needed -/
def boxes_needed : ℕ := 90

/-- Theorem stating the total number of apples to be packed -/
theorem total_apples_to_pack : apples_per_dozen * boxes_needed = 1080 := by
  sorry

end total_apples_to_pack_l3593_359397


namespace a_plays_d_on_third_day_l3593_359387

-- Define the players
inductive Player : Type
| A : Player
| B : Player
| C : Player
| D : Player

-- Define a match as a pair of players
def Match := Player × Player

-- Define the schedule as a function from day to pair of matches
def Schedule := Nat → Match × Match

-- Define the condition that each player plays against each other exactly once
def playsAgainstEachOther (s : Schedule) : Prop :=
  ∀ p1 p2 : Player, p1 ≠ p2 → ∃ d : Nat, (s d).1 = (p1, p2) ∨ (s d).1 = (p2, p1) ∨ (s d).2 = (p1, p2) ∨ (s d).2 = (p2, p1)

-- Define the condition that each player plays only one match per day
def oneMatchPerDay (s : Schedule) : Prop :=
  ∀ d : Nat, ∀ p : Player, 
    ((s d).1.1 = p ∨ (s d).1.2 = p) → ((s d).2.1 ≠ p ∧ (s d).2.2 ≠ p)

-- Define the given conditions for the first two days
def givenConditions (s : Schedule) : Prop :=
  (s 1).1 = (Player.A, Player.C) ∨ (s 1).1 = (Player.C, Player.A) ∨ 
  (s 1).2 = (Player.A, Player.C) ∨ (s 1).2 = (Player.C, Player.A) ∧
  (s 2).1 = (Player.C, Player.D) ∨ (s 2).1 = (Player.D, Player.C) ∨ 
  (s 2).2 = (Player.C, Player.D) ∨ (s 2).2 = (Player.D, Player.C)

-- Theorem statement
theorem a_plays_d_on_third_day (s : Schedule) 
  (h1 : playsAgainstEachOther s) 
  (h2 : oneMatchPerDay s) 
  (h3 : givenConditions s) : 
  (s 3).1 = (Player.A, Player.D) ∨ (s 3).1 = (Player.D, Player.A) ∨ 
  (s 3).2 = (Player.A, Player.D) ∨ (s 3).2 = (Player.D, Player.A) :=
sorry

end a_plays_d_on_third_day_l3593_359387


namespace g_prime_symmetry_l3593_359304

open Function Real

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the derivatives of f and g
variable (f' g' : ℝ → ℝ)

-- Assume f' is the derivative of f and g' is the derivative of g
variable (hf : ∀ x, HasDerivAt f (f' x) x)
variable (hg : ∀ x, HasDerivAt g (g' x) x)

-- Define the conditions
variable (h1 : ∀ x, f x + g' x = 5)
variable (h2 : ∀ x, f (2 - x) - g' (2 + x) = 5)
variable (h3 : Odd g)

-- State the theorem
theorem g_prime_symmetry (x : ℝ) : g' (8 - x) = g' x := sorry

end g_prime_symmetry_l3593_359304


namespace equation_solutions_l3593_359357

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = (5 + Real.sqrt 21) / 2 ∧ x₂ = (5 - Real.sqrt 21) / 2 ∧
    x₁^2 - 5*x₁ + 1 = 0 ∧ x₂^2 - 5*x₂ + 1 = 0) ∧
  (∃ y₁ y₂ : ℝ, y₁ = 5 ∧ y₂ = 10/3 ∧
    2*(y₁-5)^2 + y₁*(y₁-5) = 0 ∧ 2*(y₂-5)^2 + y₂*(y₂-5) = 0) :=
by
  sorry


end equation_solutions_l3593_359357


namespace jung_age_l3593_359370

/-- Proves Jung's age given the ages of Li and Zhang and their relationships -/
theorem jung_age (li_age : ℕ) (zhang_age : ℕ) (jung_age : ℕ)
  (h1 : zhang_age = 2 * li_age)
  (h2 : li_age = 12)
  (h3 : jung_age = zhang_age + 2) :
  jung_age = 26 := by
sorry

end jung_age_l3593_359370


namespace bus_count_l3593_359327

theorem bus_count (total_students : ℕ) (students_per_bus : ℕ) (h1 : total_students = 360) (h2 : students_per_bus = 45) :
  total_students / students_per_bus = 8 :=
by sorry

end bus_count_l3593_359327


namespace recipe_sugar_amount_l3593_359347

/-- The amount of sugar in cups already added to the recipe -/
def sugar_added : ℕ := 4

/-- The amount of sugar in cups still needed to be added to the recipe -/
def sugar_needed : ℕ := 3

/-- The total amount of sugar in cups required by the recipe -/
def total_sugar : ℕ := sugar_added + sugar_needed

theorem recipe_sugar_amount : total_sugar = 7 := by sorry

end recipe_sugar_amount_l3593_359347


namespace cos_alpha_for_point_four_neg_three_l3593_359321

theorem cos_alpha_for_point_four_neg_three (α : Real) :
  (∃ (x y : Real), x = 4 ∧ y = -3 ∧ x = Real.cos α * Real.sqrt (x^2 + y^2) ∧ y = Real.sin α * Real.sqrt (x^2 + y^2)) →
  Real.cos α = 4/5 := by
sorry

end cos_alpha_for_point_four_neg_three_l3593_359321


namespace sum_greater_than_one_l3593_359355

theorem sum_greater_than_one
  (a b c d : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (hc : c > 0)
  (hd : d > 0)
  (hac : a > c)
  (hbd : b < d)
  (h1 : a + Real.sqrt b ≥ c + Real.sqrt d)
  (h2 : Real.sqrt a + b ≤ Real.sqrt c + d) :
  a + b + c + d > 1 :=
by sorry

end sum_greater_than_one_l3593_359355


namespace geometric_sequence_product_l3593_359320

/-- Given a geometric sequence {aₙ} with common ratio q = √2 and a₁ · a₃ · a₅ = 4,
    prove that a₄ · a₅ · a₆ = 32. -/
theorem geometric_sequence_product (a : ℕ → ℝ) :
  (∀ n, a (n + 1) = a n * Real.sqrt 2) →  -- geometric sequence with ratio √2
  a 1 * a 3 * a 5 = 4 →                   -- given condition
  a 4 * a 5 * a 6 = 32 := by
sorry

end geometric_sequence_product_l3593_359320


namespace oldest_babysat_age_l3593_359364

theorem oldest_babysat_age (jane_start_age : ℕ) (jane_current_age : ℕ) (years_since_stopped : ℕ) :
  jane_start_age = 18 →
  jane_current_age = 34 →
  years_since_stopped = 12 →
  (∀ (jane_age : ℕ) (child_age : ℕ),
    jane_age ≥ jane_start_age →
    jane_age ≤ jane_current_age - years_since_stopped →
    child_age ≤ jane_age / 2) →
  (jane_current_age - years_since_stopped - jane_start_age) + years_since_stopped + 
    ((jane_current_age - years_since_stopped) / 2) = 23 :=
by sorry

end oldest_babysat_age_l3593_359364


namespace arithmetic_increasing_iff_positive_difference_l3593_359394

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, d ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n + d

/-- An increasing sequence -/
def IncreasingSequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, n < m → a n < a m

/-- Theorem: An arithmetic sequence is increasing if and only if its common difference is positive -/
theorem arithmetic_increasing_iff_positive_difference (a : ℕ → ℝ) :
  ArithmeticSequence a → (IncreasingSequence a ↔ ∃ d : ℝ, d > 0 ∧ ∀ n : ℕ, a (n + 1) = a n + d) :=
sorry

end arithmetic_increasing_iff_positive_difference_l3593_359394


namespace certain_number_addition_l3593_359329

theorem certain_number_addition (x : ℝ) (h : 5 * x = 60) : x + 34 = 46 := by
  sorry

end certain_number_addition_l3593_359329


namespace sequence_formula_l3593_359333

theorem sequence_formula (a : ℕ+ → ℚ) :
  (∀ n : ℕ+, a (n + 1) / a n = (n + 2) / n) →
  a 1 = 1 →
  ∀ n : ℕ+, a n = n * (n + 1) / 2 := by
  sorry

end sequence_formula_l3593_359333


namespace lemonade_sales_difference_l3593_359374

/-- Anna's lemonade sales problem -/
theorem lemonade_sales_difference :
  let plain_glasses : ℕ := 36
  let plain_price : ℚ := 3/4  -- $0.75 represented as a rational number
  let strawberry_earnings : ℚ := 16
  let plain_earnings := plain_glasses * plain_price
  plain_earnings - strawberry_earnings = 11 := by sorry

end lemonade_sales_difference_l3593_359374


namespace bennys_work_days_l3593_359300

/-- Given that Benny worked 3 hours a day for a total of 18 hours,
    prove that he worked for 6 days. -/
theorem bennys_work_days (hours_per_day : ℕ) (total_hours : ℕ) (days : ℕ) : 
  hours_per_day = 3 → total_hours = 18 → days * hours_per_day = total_hours → days = 6 := by
  sorry


end bennys_work_days_l3593_359300


namespace negative_of_negative_five_l3593_359360

theorem negative_of_negative_five : -(- 5) = 5 := by sorry

end negative_of_negative_five_l3593_359360


namespace prob_king_then_ten_l3593_359338

/-- Represents a standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Number of Kings in a standard deck -/
def NumKings : ℕ := 4

/-- Number of 10s in a standard deck -/
def NumTens : ℕ := 4

/-- Probability of drawing a King first and then a 10 from a standard deck -/
theorem prob_king_then_ten : 
  (NumKings : ℚ) / StandardDeck * NumTens / (StandardDeck - 1) = 4 / 663 := by
  sorry

end prob_king_then_ten_l3593_359338


namespace child_ticket_price_soccer_match_l3593_359340

/-- The price of a child's ticket at a soccer match -/
def child_ticket_price (num_adults num_children : ℕ) (adult_ticket_price total_bill : ℚ) : ℚ :=
  (total_bill - num_adults * adult_ticket_price) / num_children

theorem child_ticket_price_soccer_match :
  child_ticket_price 25 32 12 450 = 469/100 :=
by sorry

end child_ticket_price_soccer_match_l3593_359340


namespace range_of_m_for_decreasing_function_l3593_359336

-- Define a decreasing function on an open interval
def DecreasingOnInterval (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x > f y

-- Main theorem
theorem range_of_m_for_decreasing_function 
  (f : ℝ → ℝ) (m : ℝ) 
  (h_decreasing : DecreasingOnInterval f (-2) 2)
  (h_inequality : f (m - 1) > f (2 * m - 1)) :
  0 < m ∧ m < 3/2 :=
sorry

end range_of_m_for_decreasing_function_l3593_359336


namespace simple_interest_rate_percent_l3593_359312

/-- Simple interest calculation -/
theorem simple_interest_rate_percent 
  (principal : ℝ) 
  (interest : ℝ) 
  (time : ℝ) 
  (h1 : principal = 1000)
  (h2 : interest = 400)
  (h3 : time = 4)
  : (interest * 100) / (principal * time) = 10 := by
  sorry

end simple_interest_rate_percent_l3593_359312


namespace max_area_rectangular_garden_l3593_359344

/-- The maximum area of a rectangular garden with integer side lengths and a perimeter of 150 feet. -/
theorem max_area_rectangular_garden : ∃ (l w : ℕ), 
  (2 * l + 2 * w = 150) ∧ 
  (∀ (a b : ℕ), (2 * a + 2 * b = 150) → (a * b ≤ l * w)) ∧
  (l * w = 1406) := by
  sorry

end max_area_rectangular_garden_l3593_359344


namespace cos_4theta_from_complex_exp_l3593_359395

theorem cos_4theta_from_complex_exp (θ : ℝ) :
  Complex.exp (θ * Complex.I) = (3 + Complex.I * Real.sqrt 8) / 4 →
  Real.cos (4 * θ) = -287 / 256 := by
  sorry

end cos_4theta_from_complex_exp_l3593_359395


namespace square_not_always_positive_l3593_359371

theorem square_not_always_positive : ¬ (∀ x : ℝ, x^2 > 0) := by
  sorry

end square_not_always_positive_l3593_359371


namespace range_of_a_l3593_359314

theorem range_of_a (a : ℝ) : 
  (∃ b : ℝ, b ∈ Set.Icc 1 2 ∧ 2^b * (b + a) ≥ 4) ↔ a ∈ Set.Ici (-1) := by
sorry

end range_of_a_l3593_359314


namespace power_negative_cube_squared_l3593_359303

theorem power_negative_cube_squared (a : ℝ) (n : ℤ) : (-a^(3*n))^2 = a^(6*n) := by
  sorry

end power_negative_cube_squared_l3593_359303


namespace original_group_size_l3593_359349

/-- Proves that the original number of men in a group is 22, given the conditions of the problem. -/
theorem original_group_size (initial_days : ℕ) (absent_men : ℕ) (final_days : ℕ) : 
  initial_days = 20 → absent_men = 2 → final_days = 22 → 
  ∃ (original_men : ℕ), 
    original_men * initial_days = (original_men - absent_men) * final_days ∧ 
    original_men = 22 := by
  sorry

end original_group_size_l3593_359349


namespace place_left_l3593_359310

/-- A two-digit number is between 10 and 99, inclusive. -/
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- A one-digit number is between 1 and 9, inclusive. -/
def is_one_digit (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 9

/-- Placing a one-digit number b to the left of a two-digit number a results in 100b + a. -/
theorem place_left (a b : ℕ) (ha : is_two_digit a) (hb : is_one_digit b) :
  100 * b + a = (100 * b + a) := by sorry

end place_left_l3593_359310


namespace trig_equation_iff_equal_l3593_359359

theorem trig_equation_iff_equal (a b : Real) 
  (ha : 0 ≤ a ∧ a ≤ π/2) (hb : 0 ≤ b ∧ b ≤ π/2) : 
  (Real.sin a)^6 + 3*(Real.sin a)^2*(Real.cos b)^2 + (Real.cos b)^6 = 1 ↔ a = b := by
  sorry

end trig_equation_iff_equal_l3593_359359


namespace system_of_equations_range_l3593_359392

theorem system_of_equations_range (x y m : ℝ) : 
  x + 2*y = 1 + m →
  2*x + y = 3 →
  x + y > 0 →
  m > -4 := by
sorry

end system_of_equations_range_l3593_359392


namespace range_of_m_l3593_359356

/-- The condition p -/
def p (x : ℝ) : Prop := -x^2 + 7*x + 8 ≥ 0

/-- The condition q -/
def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - 4*m^2 ≤ 0

/-- The statement that "not p" is sufficient but not necessary for "not q" -/
def not_p_suff_not_nec_not_q (m : ℝ) : Prop :=
  (∀ x, ¬(p x) → ¬(q x m)) ∧ ¬(∀ x, ¬(q x m) → ¬(p x))

/-- The main theorem -/
theorem range_of_m :
  ∀ m : ℝ, (m > 0 ∧ not_p_suff_not_nec_not_q m) ↔ (m > 0 ∧ m ≤ 1) :=
sorry

end range_of_m_l3593_359356


namespace system_solution_l3593_359384

theorem system_solution : ∃ (x y : ℝ), (7 * x - 3 * y = 2) ∧ (2 * x + y = 8) := by
  use 2, 4
  sorry

end system_solution_l3593_359384


namespace rational_coloring_exists_l3593_359369

theorem rational_coloring_exists : ∃ (f : ℚ → Bool), 
  (∀ x : ℚ, x ≠ 0 → f x ≠ f (-x)) ∧ 
  (∀ x : ℚ, x ≠ 1/2 → f x ≠ f (1 - x)) ∧ 
  (∀ x : ℚ, x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1 → f x ≠ f (1 / x)) := by
  sorry

end rational_coloring_exists_l3593_359369


namespace ninas_ants_l3593_359346

theorem ninas_ants (spider_count : ℕ) (spider_eyes : ℕ) (ant_eyes : ℕ) (total_eyes : ℕ) :
  spider_count = 3 →
  spider_eyes = 8 →
  ant_eyes = 2 →
  total_eyes = 124 →
  (total_eyes - spider_count * spider_eyes) / ant_eyes = 50 := by
  sorry

end ninas_ants_l3593_359346


namespace torn_sheets_count_l3593_359313

/-- Represents a book with numbered pages -/
structure Book where
  /-- The number of the first torn-out page -/
  first_torn_page : ℕ
  /-- The number of the last torn-out page -/
  last_torn_page : ℕ

/-- Calculates the number of torn-out sheets given a book -/
def torn_sheets (b : Book) : ℕ :=
  (b.last_torn_page - b.first_torn_page + 1) / 2

/-- The main theorem stating that 167 sheets were torn out -/
theorem torn_sheets_count (b : Book) 
  (h1 : b.first_torn_page = 185)
  (h2 : b.last_torn_page = 518) :
  torn_sheets b = 167 := by
  sorry

end torn_sheets_count_l3593_359313


namespace alcohol_percentage_in_first_vessel_l3593_359342

/-- Proves that the percentage of alcohol in the first vessel is 25% --/
theorem alcohol_percentage_in_first_vessel : 
  ∀ (x : ℝ),
  -- Vessel capacities and total liquid
  let vessel1_capacity : ℝ := 2
  let vessel2_capacity : ℝ := 6
  let total_liquid : ℝ := 8
  let final_vessel_capacity : ℝ := 10
  -- Alcohol percentages
  let vessel2_alcohol_percentage : ℝ := 50
  let final_mixture_percentage : ℝ := 35
  -- Condition: total alcohol in final mixture
  (x / 100) * vessel1_capacity + (vessel2_alcohol_percentage / 100) * vessel2_capacity = 
    (final_mixture_percentage / 100) * final_vessel_capacity →
  -- Conclusion: alcohol percentage in first vessel is 25%
  x = 25 := by
sorry

end alcohol_percentage_in_first_vessel_l3593_359342


namespace max_intersection_faces_l3593_359353

def W : Set (Fin 4 → ℝ) := {x | ∀ i, 0 ≤ x i ∧ x i ≤ 1}

def isParallelHyperplane (h : ℝ → ℝ → ℝ → ℝ → Prop) : Prop :=
  ∃ (k : ℝ), ∀ x₁ x₂ x₃ x₄, h x₁ x₂ x₃ x₄ ↔ x₁ + x₂ + x₃ + x₄ = k

def intersectionFaces (h : ℝ → ℝ → ℝ → ℝ → Prop) : ℕ :=
  sorry

theorem max_intersection_faces :
  ∀ h, isParallelHyperplane h →
    (∃ x ∈ W, h (x 0) (x 1) (x 2) (x 3)) →
    intersectionFaces h ≤ 8 ∧
    (∃ h', isParallelHyperplane h' ∧
      (∃ x ∈ W, h' (x 0) (x 1) (x 2) (x 3)) ∧
      intersectionFaces h' = 8) :=
by sorry

end max_intersection_faces_l3593_359353


namespace fixed_cost_calculation_l3593_359326

/-- The fixed cost to run the molding machine per week -/
def fixed_cost : ℝ := 7640

/-- The cost to mold each handle -/
def mold_cost : ℝ := 0.60

/-- The selling price per handle -/
def selling_price : ℝ := 4.60

/-- The number of handles needed to break even -/
def break_even_quantity : ℕ := 1910

/-- Theorem stating that the fixed cost is correct given the conditions -/
theorem fixed_cost_calculation :
  fixed_cost = (selling_price - mold_cost) * break_even_quantity := by
  sorry

end fixed_cost_calculation_l3593_359326


namespace smallest_multiple_l3593_359367

theorem smallest_multiple (x : ℕ) : x = 48 ↔ (
  x > 0 ∧
  (∃ k : ℕ, 600 * x = 1152 * k) ∧
  (∀ y : ℕ, y > 0 → y < x → ¬∃ k : ℕ, 600 * y = 1152 * k)
) := by sorry

end smallest_multiple_l3593_359367


namespace reciprocal_sum_inequality_l3593_359318

theorem reciprocal_sum_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + y + z ≤ 3) : 1/x + 1/y + 1/z ≥ 3 := by
  sorry

end reciprocal_sum_inequality_l3593_359318


namespace correct_subtraction_result_l3593_359389

/-- 
Given a subtraction problem where:
- The tens digit 7 was mistaken for 9
- The ones digit 3 was mistaken for 8
- The mistaken subtraction resulted in a difference of 76

Prove that the correct difference is 51.
-/
theorem correct_subtraction_result : 
  ∀ (original_tens original_ones mistaken_tens mistaken_ones mistaken_difference : ℕ),
  original_tens = 7 →
  original_ones = 3 →
  mistaken_tens = 9 →
  mistaken_ones = 8 →
  mistaken_difference = 76 →
  (mistaken_tens * 10 + mistaken_ones) - (original_tens * 10 + original_ones) = mistaken_difference →
  (original_tens * 10 + original_ones) - 
    ((mistaken_tens * 10 + mistaken_ones) - (original_tens * 10 + original_ones)) = 51 := by
  sorry

end correct_subtraction_result_l3593_359389


namespace stationery_shop_sales_l3593_359390

theorem stationery_shop_sales (total_sales percent_pens percent_pencils : ℝ) 
  (h_total : total_sales = 100)
  (h_pens : percent_pens = 38)
  (h_pencils : percent_pencils = 35) :
  total_sales - percent_pens - percent_pencils = 27 := by
  sorry

end stationery_shop_sales_l3593_359390


namespace complex_square_l3593_359365

theorem complex_square (z : ℂ) : z = 2 + 5*I → z^2 = -21 + 20*I := by
  sorry

end complex_square_l3593_359365


namespace max_xy_value_l3593_359319

theorem max_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  ∃ (m : ℝ), m = 1/4 ∧ ∀ (z : ℝ), z = x * y → z ≤ m := by
  sorry

end max_xy_value_l3593_359319


namespace quadratic_one_solution_l3593_359345

theorem quadratic_one_solution (k : ℚ) : 
  (∃! x, 2 * x^2 - 5 * x + k = 0) ↔ k = 25/8 := by
  sorry

end quadratic_one_solution_l3593_359345


namespace zeros_before_last_digit_2009_pow_2011_l3593_359363

theorem zeros_before_last_digit_2009_pow_2011 :
  ∃ n : ℕ, n > 0 ∧ (2009^2011 % 10^(n+1)) / 10^n = 0 ∧ (2009^2011 % 10^n) / 10^(n-1) ≠ 0 :=
sorry

end zeros_before_last_digit_2009_pow_2011_l3593_359363


namespace sequence_difference_equals_170000_l3593_359339

/-- The sum of an arithmetic sequence with first term a, last term l, and n terms -/
def arithmetic_sum (a l n : ℕ) : ℕ := n * (a + l) / 2

/-- The difference between two sums of arithmetic sequences -/
def sequence_difference : ℕ :=
  arithmetic_sum 2001 2100 100 - arithmetic_sum 301 400 100

theorem sequence_difference_equals_170000 : sequence_difference = 170000 := by
  sorry

end sequence_difference_equals_170000_l3593_359339


namespace area_of_square_II_l3593_359388

/-- Given a square I with diagonal 3(a+b), where a and b are positive real numbers,
    the area of a square II that is three times the area of square I is equal to 27(a+b)^2/2. -/
theorem area_of_square_II (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  let diagonal_I := 3 * (a + b)
  let area_I := (diagonal_I ^ 2) / 2
  let area_II := 3 * area_I
  area_II = 27 * (a + b)^2 / 2 := by sorry

end area_of_square_II_l3593_359388


namespace percentage_less_than_third_l3593_359332

theorem percentage_less_than_third (n1 n2 n3 : ℝ) : 
  n1 = 0.7 * n3 →  -- First number is 30% less than third number
  n2 = 0.9 * n1 →  -- Second number is 10% less than first number
  n2 = 0.63 * n3   -- Second number is 37% less than third number
:= by sorry

end percentage_less_than_third_l3593_359332


namespace inequality_solution_implies_a_value_l3593_359366

/-- Given that the solution set of the inequality (ax-1)(x+1)<0 is (-∞, -1) ∪ (-1/2, +∞),
    prove that a = -2 -/
theorem inequality_solution_implies_a_value (a : ℝ) : 
  (∀ x : ℝ, (a*x - 1)*(x + 1) < 0 ↔ x < -1 ∨ -1/2 < x) → a = -2 := by
sorry

end inequality_solution_implies_a_value_l3593_359366


namespace incorrect_permutations_of_good_l3593_359317

-- Define a structure for our word
structure Word where
  length : Nat
  repeated_letter_count : Nat

-- Define our specific word "good"
def good : Word := { length := 4, repeated_letter_count := 2 }

-- Theorem statement
theorem incorrect_permutations_of_good (w : Word) (h1 : w = good) : 
  (w.length.factorial / w.repeated_letter_count.factorial) - 1 = 11 := by
  sorry

end incorrect_permutations_of_good_l3593_359317


namespace plane_flight_distance_l3593_359385

/-- Given a plane that flies with and against the wind, prove the distance flown against the wind -/
theorem plane_flight_distance 
  (distance_with_wind : ℝ) 
  (wind_speed : ℝ) 
  (plane_speed : ℝ) 
  (h1 : distance_with_wind = 420) 
  (h2 : wind_speed = 23) 
  (h3 : plane_speed = 253) : 
  (distance_with_wind * (plane_speed - wind_speed)) / (plane_speed + wind_speed) = 350 := by
  sorry

end plane_flight_distance_l3593_359385


namespace problem_solution_l3593_359308

theorem problem_solution (x y : ℝ) (hx : x = 3) (hy : y = 4) :
  (x^4 + 3*y^3 + 10) / 7 = 283/7 := by
  sorry

end problem_solution_l3593_359308


namespace expression_equality_l3593_359396

theorem expression_equality : (19 * 19 - 12 * 12) / ((19 / 12) - (12 / 19)) = 228 := by
  sorry

end expression_equality_l3593_359396


namespace book_cost_calculation_l3593_359373

theorem book_cost_calculation (num_books : ℕ) (money_have : ℕ) (money_save : ℕ) :
  num_books = 8 ∧ money_have = 13 ∧ money_save = 27 →
  (money_have + money_save) / num_books = 5 := by
sorry

end book_cost_calculation_l3593_359373


namespace cake_mass_proof_l3593_359325

/-- The initial mass of the cake in grams -/
def initial_mass : ℝ := 750

/-- The mass of cake Karlson ate for breakfast -/
def karlson_ate : ℝ := 0.4 * initial_mass

/-- The mass of cake Malish ate for breakfast -/
def malish_ate : ℝ := 150

/-- The percentage of remaining cake Freken Bok ate for lunch -/
def freken_bok_percent : ℝ := 0.3

/-- The additional mass of cake Freken Bok ate for lunch -/
def freken_bok_additional : ℝ := 120

/-- The mass of cake crumbs Matilda licked -/
def matilda_licked : ℝ := 90

theorem cake_mass_proof :
  initial_mass = karlson_ate + malish_ate +
  (freken_bok_percent * (initial_mass - karlson_ate - malish_ate) + freken_bok_additional) +
  matilda_licked := by sorry

end cake_mass_proof_l3593_359325


namespace milk_leftover_problem_l3593_359361

/-- Calculates the amount of milk left over from yesterday given today's milk production and sales --/
def milk_leftover (morning_milk : ℕ) (evening_milk : ℕ) (sold_milk : ℕ) (total_left : ℕ) : ℕ :=
  total_left - ((morning_milk + evening_milk) - sold_milk)

/-- Theorem stating that given the problem conditions, the milk leftover from yesterday is 15 gallons --/
theorem milk_leftover_problem : milk_leftover 365 380 612 148 = 15 := by
  sorry

end milk_leftover_problem_l3593_359361


namespace only_parallelogram_not_axially_symmetric_l3593_359324

-- Define the shapes
inductive Shape
  | Rectangle
  | IsoscelesTrapezoid
  | Parallelogram
  | EquilateralTriangle

-- Define axial symmetry
def is_axially_symmetric (s : Shape) : Prop :=
  match s with
  | Shape.Rectangle => true
  | Shape.IsoscelesTrapezoid => true
  | Shape.Parallelogram => false
  | Shape.EquilateralTriangle => true

-- Theorem statement
theorem only_parallelogram_not_axially_symmetric :
  ∀ s : Shape, ¬(is_axially_symmetric s) ↔ s = Shape.Parallelogram :=
by sorry

end only_parallelogram_not_axially_symmetric_l3593_359324


namespace thirty_thousand_squared_l3593_359382

theorem thirty_thousand_squared :
  (30000 : ℕ) ^ 2 = 900000000 := by
  sorry

end thirty_thousand_squared_l3593_359382


namespace prism_volume_l3593_359381

/-- The volume of a right rectangular prism with face areas 15, 10, and 30 square inches is 30√5 cubic inches. -/
theorem prism_volume (a b c : ℝ) (h1 : a * b = 15) (h2 : b * c = 10) (h3 : c * a = 30) :
  a * b * c = 30 * Real.sqrt 5 := by
  sorry

end prism_volume_l3593_359381


namespace largest_among_three_l3593_359330

theorem largest_among_three (sin2 log132 log1213 : ℝ) 
  (h1 : 0 < sin2 ∧ sin2 < 1)
  (h2 : log132 < 0)
  (h3 : log1213 > 1) :
  log1213 = max sin2 (max log132 log1213) :=
sorry

end largest_among_three_l3593_359330


namespace equation_solutions_l3593_359311

theorem equation_solutions : 
  ∃! (s : Set ℝ), 
    (∀ x ∈ s, |x - 2| = |x - 1| + |x - 3| + |x - 4|) ∧ 
    s = {2, 2.25} := by
  sorry

end equation_solutions_l3593_359311


namespace ethan_present_count_l3593_359315

/-- The number of presents Ethan has -/
def ethan_presents : ℕ := 31

/-- The number of presents Alissa has -/
def alissa_presents : ℕ := 53

/-- The difference in presents between Alissa and Ethan -/
def present_difference : ℕ := 22

theorem ethan_present_count : ethan_presents = alissa_presents - present_difference := by
  sorry

end ethan_present_count_l3593_359315


namespace smaller_number_proof_l3593_359378

theorem smaller_number_proof (x y : ℝ) : 
  x + y = 14 → y = 3 * x → x = 3.5 := by
  sorry

end smaller_number_proof_l3593_359378


namespace servant_cash_payment_l3593_359307

-- Define the problem parameters
def annual_cash_salary : ℕ := 90
def turban_price : ℕ := 70
def months_worked : ℕ := 9
def months_per_year : ℕ := 12

-- Define the theorem
theorem servant_cash_payment :
  let total_annual_salary := annual_cash_salary + turban_price
  let proportion_worked := months_worked / months_per_year
  let earned_amount := (proportion_worked * total_annual_salary : ℚ).floor
  earned_amount - turban_price = 50 := by
  sorry

end servant_cash_payment_l3593_359307


namespace prime_condition_characterization_l3593_359351

def satisfies_condition (p : Nat) : Prop :=
  Nat.Prime p ∧
  ∀ q, Nat.Prime q → q < p →
    ∀ k r, p = k * q + r → 0 ≤ r → r < q →
      ∀ a, a > 1 → ¬(a^2 ∣ r)

theorem prime_condition_characterization :
  {p : Nat | satisfies_condition p} = {2, 3, 5, 7, 13} := by sorry

end prime_condition_characterization_l3593_359351


namespace real_roots_condition_l3593_359301

theorem real_roots_condition (a : ℝ) :
  (∃ x : ℝ, x^2 + x + |a - 1/4| + |a| = 0) ↔ 0 ≤ a ∧ a ≤ 1/4 := by
  sorry

end real_roots_condition_l3593_359301


namespace sequence_properties_l3593_359376

/-- Sequence sum function -/
def S (n : ℕ) : ℤ := -n^2 + 7*n

/-- Sequence term function -/
def a (n : ℕ) : ℤ := -2*n + 8

theorem sequence_properties :
  (∀ n : ℕ, n ≥ 1 → a n = S n - S (n-1)) ∧
  (∃ m : ℕ, m ≥ 1 ∧ ∀ n : ℕ, n ≥ 1 → S n ≤ S m) ∧
  (∀ n : ℕ, n ≥ 1 → S n ≤ 12) := by
  sorry

end sequence_properties_l3593_359376


namespace ice_cream_distribution_l3593_359352

theorem ice_cream_distribution (total_sandwiches : ℕ) (num_nieces : ℕ) 
  (h1 : total_sandwiches = 143) (h2 : num_nieces = 11) :
  ∃ (sandwiches_per_niece : ℕ), 
    sandwiches_per_niece * num_nieces = total_sandwiches ∧ 
    sandwiches_per_niece = 13 := by
  sorry

end ice_cream_distribution_l3593_359352


namespace triangle_DEF_is_right_angled_and_isosceles_l3593_359335

-- Define the basic structures
structure Point := (x y : ℝ)

structure Triangle :=
  (A B C : Point)

-- Define the properties of the given triangles
def is_midpoint (F : Point) (B C : Point) : Prop :=
  F.x = (B.x + C.x) / 2 ∧ F.y = (B.y + C.y) / 2

def is_isosceles_right_triangle (A B D : Point) : Prop :=
  (A.x - B.x)^2 + (A.y - B.y)^2 = (A.x - D.x)^2 + (A.y - D.y)^2 ∧
  (A.x - D.x) * (B.x - D.x) + (A.y - D.y) * (B.y - D.y) = 0

-- Define the theorem
theorem triangle_DEF_is_right_angled_and_isosceles 
  (ABC : Triangle) 
  (F D E : Point) 
  (h1 : is_midpoint F ABC.B ABC.C)
  (h2 : is_isosceles_right_triangle ABC.A ABC.B D)
  (h3 : is_isosceles_right_triangle ABC.A ABC.C E) :
  is_isosceles_right_triangle D E F := by
  sorry

end triangle_DEF_is_right_angled_and_isosceles_l3593_359335


namespace aaron_earnings_l3593_359350

/-- Represents the work hours for each day of the week -/
structure WorkHours :=
  (monday : Real)
  (tuesday : Real)
  (wednesday : Real)
  (friday : Real)

/-- Calculates the total earnings for the week given work hours and hourly rate -/
def calculateEarnings (hours : WorkHours) (hourlyRate : Real) : Real :=
  (hours.monday + hours.tuesday + hours.wednesday + hours.friday) * hourlyRate

/-- Theorem stating that Aaron's earnings for the week are $38.75 -/
theorem aaron_earnings :
  let hours : WorkHours := {
    monday := 2,
    tuesday := 1.25,
    wednesday := 2.833,
    friday := 0.667
  }
  let hourlyRate : Real := 5
  calculateEarnings hours hourlyRate = 38.75 := by
  sorry

#check aaron_earnings

end aaron_earnings_l3593_359350


namespace falcons_win_percentage_l3593_359322

theorem falcons_win_percentage (initial_games : ℕ) (initial_falcon_wins : ℕ) (win_percentage : ℚ) :
  let additional_games : ℕ := 42
  initial_games = 8 ∧ 
  initial_falcon_wins = 3 ∧ 
  win_percentage = 9/10 →
  (initial_falcon_wins + additional_games : ℚ) / (initial_games + additional_games) ≥ win_percentage ∧
  ∀ n : ℕ, n < additional_games → 
    (initial_falcon_wins + n : ℚ) / (initial_games + n) < win_percentage :=
by sorry

end falcons_win_percentage_l3593_359322


namespace widgets_in_shipping_box_is_300_l3593_359358

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  width : ℕ
  length : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ :=
  d.width * d.length * d.height

/-- Represents the problem setup -/
structure WidgetProblem where
  cartonDimensions : BoxDimensions
  shippingBoxDimensions : BoxDimensions
  widgetsPerCarton : ℕ

/-- Calculates the number of widgets in a shipping box -/
def widgetsInShippingBox (p : WidgetProblem) : ℕ :=
  let cartonsInBox := (boxVolume p.shippingBoxDimensions) / (boxVolume p.cartonDimensions)
  cartonsInBox * p.widgetsPerCarton

/-- The main theorem to prove -/
theorem widgets_in_shipping_box_is_300 (p : WidgetProblem) : 
  p.cartonDimensions = ⟨4, 4, 5⟩ ∧ 
  p.shippingBoxDimensions = ⟨20, 20, 20⟩ ∧ 
  p.widgetsPerCarton = 3 → 
  widgetsInShippingBox p = 300 := by
  sorry


end widgets_in_shipping_box_is_300_l3593_359358


namespace total_savings_calculation_l3593_359383

def initial_savings : ℕ := 849400
def monthly_income : ℕ := 110000
def monthly_expenses : ℕ := 58500
def months : ℕ := 5

theorem total_savings_calculation :
  initial_savings + months * monthly_income - months * monthly_expenses = 1106900 := by
  sorry

end total_savings_calculation_l3593_359383


namespace a_5_value_l3593_359316

def geometric_sequence (a : ℕ → ℝ) (r : ℝ) :=
  ∀ n : ℕ, a (n + 1) - a n = r * (a n - a (n - 1))

theorem a_5_value (a : ℕ → ℝ) :
  geometric_sequence a 2 →
  a 1 - a 0 = 1 →
  a 5 = 31 :=
by
  sorry

end a_5_value_l3593_359316


namespace no_real_solution_for_equation_l3593_359331

theorem no_real_solution_for_equation : 
  ∀ x : ℝ, ¬(5 * (2*x)^2 - 3*(2*x) + 7 = 2*(8*x^2 - 2*x + 3)) := by
  sorry

end no_real_solution_for_equation_l3593_359331


namespace max_value_x4y2z_l3593_359399

theorem max_value_x4y2z (x y z : ℝ) (hpos : x > 0 ∧ y > 0 ∧ z > 0) (hsum : x^2 + y^2 + z^2 = 1) :
  x^4 * y^2 * z ≤ 32 / (16807 * Real.sqrt 7) ∧ 
  ∃ x y z, x > 0 ∧ y > 0 ∧ z > 0 ∧ x^2 + y^2 + z^2 = 1 ∧ x^4 * y^2 * z = 32 / (16807 * Real.sqrt 7) := by
  sorry

#check max_value_x4y2z

end max_value_x4y2z_l3593_359399


namespace point_movement_l3593_359354

/-- Given a point A with coordinates (-3, -2), moving it up by 3 units
    and then left by 2 units results in a point B with coordinates (-5, 1). -/
theorem point_movement :
  let A : ℝ × ℝ := (-3, -2)
  let up_movement : ℝ := 3
  let left_movement : ℝ := 2
  let B : ℝ × ℝ := (A.1 - left_movement, A.2 + up_movement)
  B = (-5, 1) := by sorry

end point_movement_l3593_359354


namespace product_mod_six_l3593_359362

theorem product_mod_six : (2015 * 2016 * 2017 * 2018) % 6 = 0 := by
  sorry

end product_mod_six_l3593_359362


namespace log_expression_equals_twelve_l3593_359375

theorem log_expression_equals_twelve : 
  (4 - (Real.log 4 / Real.log 36) - (Real.log 18 / Real.log 6)) / (Real.log 3 / Real.log 4) * 
  ((Real.log 27 / Real.log 8) + (Real.log 9 / Real.log 2)) = 12 := by
  sorry

end log_expression_equals_twelve_l3593_359375


namespace probability_at_least_three_hits_l3593_359398

def probability_hit_single_shot : ℝ := 0.8
def number_of_shots : ℕ := 4
def minimum_hits : ℕ := 3

theorem probability_at_least_three_hits :
  let p := probability_hit_single_shot
  let n := number_of_shots
  let k := minimum_hits
  (Finset.sum (Finset.range (n - k + 1))
    (λ i => (n.choose (k + i)) * p^(k + i) * (1 - p)^(n - k - i))) = 0.8192 :=
by sorry

end probability_at_least_three_hits_l3593_359398
