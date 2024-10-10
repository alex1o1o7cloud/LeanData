import Mathlib

namespace mMobileCheaperByEleven_l3614_361494

/-- Calculates the cost of a mobile plan given the base cost for two lines, 
    the cost per additional line, and the total number of lines. -/
def mobilePlanCost (baseCost : ℕ) (additionalLineCost : ℕ) (totalLines : ℕ) : ℕ :=
  baseCost + (max (totalLines - 2) 0) * additionalLineCost

/-- Proves that M-Mobile is $11 cheaper than T-Mobile for a family plan with 5 lines. -/
theorem mMobileCheaperByEleven : 
  mobilePlanCost 50 16 5 - mobilePlanCost 45 14 5 = 11 := by
  sorry

end mMobileCheaperByEleven_l3614_361494


namespace total_crayons_l3614_361458

/-- Theorem: The total number of crayons after adding more is the sum of the initial number and the added number. -/
theorem total_crayons (initial : ℕ) (added : ℕ) : 
  initial + added = initial + added :=
by sorry

end total_crayons_l3614_361458


namespace biology_class_boys_l3614_361450

theorem biology_class_boys (girls_to_boys_ratio : ℚ) (physics_students : ℕ) (biology_to_physics_ratio : ℚ) :
  girls_to_boys_ratio = 3 →
  physics_students = 200 →
  biology_to_physics_ratio = 1/2 →
  (physics_students : ℚ) * biology_to_physics_ratio / (1 + girls_to_boys_ratio) = 25 :=
by sorry

end biology_class_boys_l3614_361450


namespace bookmark_position_l3614_361408

/-- Represents a book with pages and a bookmark --/
structure Book where
  pages : ℕ
  coverThickness : ℕ
  bookmarkPosition : ℕ

/-- Calculates the total thickness of a book in page-equivalent units --/
def bookThickness (b : Book) : ℕ := b.pages + 2 * b.coverThickness

/-- The problem setup --/
def bookshelfProblem (book1 book2 : Book) : Prop :=
  book1.pages = 250 ∧
  book2.pages = 250 ∧
  book1.coverThickness = 10 ∧
  book2.coverThickness = 10 ∧
  book1.bookmarkPosition = 125 ∧
  (bookThickness book1 + bookThickness book2) / 3 = book1.bookmarkPosition + book1.coverThickness + book2.bookmarkPosition

theorem bookmark_position (book1 book2 : Book) :
  bookshelfProblem book1 book2 → book2.bookmarkPosition = 35 :=
by sorry

end bookmark_position_l3614_361408


namespace sqrt_24_minus_3sqrt_2_3_l3614_361456

theorem sqrt_24_minus_3sqrt_2_3 : Real.sqrt 24 - 3 * Real.sqrt (2/3) = Real.sqrt 6 := by
  sorry

end sqrt_24_minus_3sqrt_2_3_l3614_361456


namespace max_value_expression_l3614_361467

theorem max_value_expression (c d : ℝ) (hc : c > 0) (hd : d > 0) :
  (∀ y : ℝ, 3 * (c - y) * (y + Real.sqrt (y^2 + d^2)) ≤ 3/2 * (c^2 + d^2)) ∧
  (∃ y : ℝ, 3 * (c - y) * (y + Real.sqrt (y^2 + d^2)) = 3/2 * (c^2 + d^2)) :=
sorry

end max_value_expression_l3614_361467


namespace arithmetic_sequence_common_difference_l3614_361433

def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℤ) (h : is_arithmetic_sequence a) (h2 : a 2 = 2) (h3 : a 3 = -4) :
  ∃ d : ℤ, d = -6 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
sorry

end arithmetic_sequence_common_difference_l3614_361433


namespace cone_lateral_surface_area_l3614_361451

/-- The lateral surface area of a cone with base radius 1 and height √3 is 2π. -/
theorem cone_lateral_surface_area : 
  let r : ℝ := 1
  let h : ℝ := Real.sqrt 3
  let l : ℝ := Real.sqrt (r^2 + h^2)
  let A : ℝ := π * r * l
  A = 2 * π :=
by sorry

end cone_lateral_surface_area_l3614_361451


namespace probability_age_21_to_30_l3614_361488

theorem probability_age_21_to_30 (total_people : ℕ) (people_21_to_30 : ℕ) 
  (h1 : total_people = 160) (h2 : people_21_to_30 = 70) : 
  (people_21_to_30 : ℚ) / total_people = 7 / 16 := by
  sorry

end probability_age_21_to_30_l3614_361488


namespace sine_squares_sum_l3614_361485

theorem sine_squares_sum (α : Real) : 
  (Real.sin (α - π/3))^2 + (Real.sin α)^2 + (Real.sin (α + π/3))^2 = 3/2 := by
  sorry

end sine_squares_sum_l3614_361485


namespace custom_op_example_l3614_361487

-- Define the custom operation
def custom_op (m n p q : ℚ) : ℚ := m * p * ((q + n) / n)

-- State the theorem
theorem custom_op_example :
  custom_op 5 9 7 4 = 455 / 9 := by
  sorry

end custom_op_example_l3614_361487


namespace max_coins_ali_baba_l3614_361407

/-- Represents the coin distribution game --/
structure CoinGame where
  totalPiles : Nat
  initialCoinsPerPile : Nat
  totalCoins : Nat
  selectablePiles : Nat
  takablePiles : Nat

/-- Defines the specific game instance --/
def aliBabaGame : CoinGame :=
  { totalPiles := 10
  , initialCoinsPerPile := 10
  , totalCoins := 100
  , selectablePiles := 4
  , takablePiles := 3 
  }

/-- Theorem stating the maximum number of coins Ali Baba can take --/
theorem max_coins_ali_baba (game : CoinGame) (h1 : game = aliBabaGame) : 
  ∃ (maxCoins : Nat), maxCoins = 72 ∧ 
  (∀ (strategy : CoinGame → Nat), strategy game ≤ maxCoins) := by
  sorry

end max_coins_ali_baba_l3614_361407


namespace intersection_implies_a_value_l3614_361481

theorem intersection_implies_a_value (A B : Set ℤ) (a : ℤ) : 
  A = {-1, 0, 1} →
  B = {0, a, 2} →
  A ∩ B = {-1, 0} →
  a = -1 := by
sorry

end intersection_implies_a_value_l3614_361481


namespace condition_sufficiency_not_necessity_l3614_361462

theorem condition_sufficiency_not_necessity :
  (∀ x : ℝ, x^2 - 4*x < 0 → 0 < x ∧ x < 5) ∧
  (∃ x : ℝ, 0 < x ∧ x < 5 ∧ x^2 - 4*x ≥ 0) := by
  sorry

end condition_sufficiency_not_necessity_l3614_361462


namespace functional_equation_solution_l3614_361470

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, (f x * f y - f (x * y)) / 5 = x + y + 3

/-- The main theorem stating that the function f(x) = x + 4 satisfies the functional equation -/
theorem functional_equation_solution :
  ∃ f : ℝ → ℝ, FunctionalEquation f ∧ ∀ x : ℝ, f x = x + 4 := by
  sorry

end functional_equation_solution_l3614_361470


namespace distribute_four_books_to_three_people_l3614_361466

/-- Represents the number of ways to distribute books to people. -/
def distribute_books (num_books : ℕ) (num_people : ℕ) : ℕ :=
  sorry

/-- Theorem stating that distributing 4 different books to 3 people,
    with each person getting at least one book, can be done in 36 ways. -/
theorem distribute_four_books_to_three_people :
  distribute_books 4 3 = 36 :=
sorry

end distribute_four_books_to_three_people_l3614_361466


namespace number_of_arrangements_l3614_361417

/-- Represents the number of students of each gender -/
def num_students : ℕ := 3

/-- Represents the total number of students -/
def total_students : ℕ := 2 * num_students

/-- Represents the number of positions where male student A can stand -/
def positions_for_A : ℕ := total_students - 2

/-- Represents the number of ways to arrange the two adjacent female students -/
def adjacent_female_arrangements : ℕ := 2

/-- Represents the number of ways to arrange the remaining students -/
def remaining_arrangements : ℕ := 3 * 2

/-- The theorem stating the number of different arrangements -/
theorem number_of_arrangements :
  positions_for_A * adjacent_female_arrangements * remaining_arrangements * Nat.factorial 2 = 288 := by
  sorry

end number_of_arrangements_l3614_361417


namespace solution_pairs_l3614_361404

theorem solution_pairs (x y : ℝ) (h1 : x^5 + y^5 = 33) (h2 : x + y = 3) :
  (x = 2 ∧ y = 1) ∨ (x = 1 ∧ y = 2) :=
sorry

end solution_pairs_l3614_361404


namespace group_size_l3614_361434

theorem group_size (average_increase : ℝ) (old_weight : ℝ) (new_weight : ℝ) : 
  average_increase = 2 →
  old_weight = 65 →
  new_weight = 81 →
  (new_weight - old_weight) / average_increase = 8 :=
by
  sorry

end group_size_l3614_361434


namespace gcd_459_357_l3614_361469

theorem gcd_459_357 : Nat.gcd 459 357 = 51 := by
  sorry

end gcd_459_357_l3614_361469


namespace billy_hike_distance_l3614_361438

theorem billy_hike_distance (east north : ℝ) (h1 : east = 7) (h2 : north = 8 * (Real.sqrt 2) / 2) :
  Real.sqrt (east^2 + north^2) = 9 := by
  sorry

end billy_hike_distance_l3614_361438


namespace initial_hamburgers_count_l3614_361416

/-- Proves that the number of hamburgers made initially equals 9 -/
theorem initial_hamburgers_count (initial : ℕ) (additional : ℕ) (total : ℕ)
  (h1 : additional = 3)
  (h2 : total = 12)
  (h3 : initial + additional = total) :
  initial = 9 := by
  sorry

end initial_hamburgers_count_l3614_361416


namespace quadratic_root_ratio_l3614_361474

theorem quadratic_root_ratio (p : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ 0 ∧ x₂ / x₁ = -4 ∧ x₁^2 + p*x₁ - 16 = 0 ∧ x₂^2 + p*x₂ - 16 = 0) → 
  (p = 6 ∨ p = -6) := by
sorry

end quadratic_root_ratio_l3614_361474


namespace sixth_term_of_arithmetic_sequence_l3614_361413

/-- Given an arithmetic sequence with first term 5/8 and eleventh term 3/4,
    the sixth term is 11/16. -/
theorem sixth_term_of_arithmetic_sequence :
  ∀ (a : ℕ → ℚ), 
    (∀ n m, a (n + m) - a n = m * (a 2 - a 1)) →  -- arithmetic sequence condition
    a 1 = 5/8 →                                   -- first term
    a 11 = 3/4 →                                  -- eleventh term
    a 6 = 11/16 :=                                -- sixth term
by
  sorry

end sixth_term_of_arithmetic_sequence_l3614_361413


namespace gcd_problem_l3614_361495

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 97 * (2 * k)) :
  Int.gcd (3 * b^2 + 41 * b + 74) (b + 19) = 1 := by
  sorry

end gcd_problem_l3614_361495


namespace magical_stack_size_157_l3614_361422

/-- A stack of cards is magical if it satisfies certain conditions --/
structure MagicalStack :=
  (n : ℕ)
  (total_cards : ℕ := 2 * n)
  (card_157_position : ℕ)
  (card_157_retains_position : card_157_position = 157)

/-- The number of cards in a magical stack where card 157 retains its position --/
def magical_stack_size (stack : MagicalStack) : ℕ := stack.total_cards

/-- Theorem: The size of a magical stack where card 157 retains its position is 470 --/
theorem magical_stack_size_157 (stack : MagicalStack) : 
  magical_stack_size stack = 470 := by sorry

end magical_stack_size_157_l3614_361422


namespace p_sufficient_not_necessary_for_q_l3614_361489

-- Define the propositions p and q
def p (x : ℝ) : Prop := 0 < x ∧ x < 2
def q (x : ℝ) : Prop := -1 < x ∧ x < 3

-- Theorem stating that p is a sufficient but not necessary condition for q
theorem p_sufficient_not_necessary_for_q : 
  (∀ x, p x → q x) ∧ (∃ x, q x ∧ ¬p x) :=
sorry

end p_sufficient_not_necessary_for_q_l3614_361489


namespace water_jars_count_l3614_361427

theorem water_jars_count (total_water : ℚ) (quart_jars half_gal_jars one_gal_jars two_gal_jars : ℕ) : 
  total_water = 56 →
  quart_jars = 16 →
  half_gal_jars = 12 →
  one_gal_jars = 8 →
  two_gal_jars = 4 →
  ∃ (three_gal_jars : ℕ), 
    (quart_jars : ℚ) * (1/4) + 
    (half_gal_jars : ℚ) * (1/2) + 
    (one_gal_jars : ℚ) + 
    (two_gal_jars : ℚ) * 2 + 
    (three_gal_jars : ℚ) * 3 = total_water ∧
    quart_jars + half_gal_jars + one_gal_jars + two_gal_jars + three_gal_jars = 50 :=
by sorry

end water_jars_count_l3614_361427


namespace equation_solutions_l3614_361479

theorem equation_solutions :
  (∀ x : ℝ, 2 * (x + 1)^2 - 49 = 1 ↔ x = 4 ∨ x = -6) ∧
  (∀ x : ℝ, (1/2) * (x - 1)^3 = -4 ↔ x = -1) := by
sorry

end equation_solutions_l3614_361479


namespace decimal_place_of_13_over_17_l3614_361483

/-- The decimal representation of 13/17 repeats every 17 digits -/
def decimal_period : ℕ := 17

/-- The repeating sequence of digits in the decimal representation of 13/17 -/
def repeating_sequence : List ℕ := [7, 6, 4, 7, 0, 5, 8, 8, 2, 3, 5, 2, 9, 4, 1, 1, 7]

/-- The position we're interested in -/
def target_position : ℕ := 250

theorem decimal_place_of_13_over_17 :
  (repeating_sequence.get! ((target_position - 1) % decimal_period)) = 3 := by
  sorry

end decimal_place_of_13_over_17_l3614_361483


namespace inequality_solution_set_l3614_361497

theorem inequality_solution_set (x : ℝ) : 
  (((1 - 2*x) / ((x - 3) * (2*x + 1))) ≥ 0) ↔ 
  (x ∈ Set.Iio (-1/2) ∪ Set.Icc (1/2) 3) :=
sorry

end inequality_solution_set_l3614_361497


namespace tennis_tournament_n_is_five_l3614_361409

/-- Represents a tennis tournament with the given conditions --/
structure TennisTournament where
  n : ℕ
  total_players : ℕ := 5 * n
  total_matches : ℕ := (total_players * (total_players - 1)) / 2
  women_wins : ℕ
  men_wins : ℕ
  no_ties : women_wins + men_wins = total_matches
  win_ratio : women_wins * 2 = men_wins * 3

/-- The theorem stating that n must be 5 for the given conditions --/
theorem tennis_tournament_n_is_five :
  ∀ t : TennisTournament, t.n = 5 := by sorry

end tennis_tournament_n_is_five_l3614_361409


namespace nantucket_meeting_attendance_l3614_361476

theorem nantucket_meeting_attendance :
  let total_population : ℕ := 300
  let females_attending : ℕ := 50
  let males_attending : ℕ := 2 * females_attending
  let total_attending : ℕ := males_attending + females_attending
  (total_attending : ℚ) / total_population = 1 / 2 := by
sorry

end nantucket_meeting_attendance_l3614_361476


namespace lcm_from_hcf_and_product_l3614_361401

theorem lcm_from_hcf_and_product (a b : ℕ+) : 
  Nat.gcd a b = 14 → a * b = 2562 → Nat.lcm a b = 183 := by
sorry

end lcm_from_hcf_and_product_l3614_361401


namespace linear_function_kb_positive_l3614_361468

/-- A linear function passing through the second, third, and fourth quadrants -/
structure LinearFunction where
  k : ℝ
  b : ℝ
  second_quadrant : ∃ x y, x < 0 ∧ y > 0 ∧ y = k * x + b
  third_quadrant : ∃ x y, x < 0 ∧ y < 0 ∧ y = k * x + b
  fourth_quadrant : ∃ x y, x > 0 ∧ y < 0 ∧ y = k * x + b

/-- Theorem: For a linear function passing through the second, third, and fourth quadrants, kb > 0 -/
theorem linear_function_kb_positive (f : LinearFunction) : f.k * f.b > 0 := by
  sorry

end linear_function_kb_positive_l3614_361468


namespace complex_fraction_simplification_l3614_361490

theorem complex_fraction_simplification :
  (5 + 7 * Complex.I) / (2 + 3 * Complex.I) = 31/13 - 1/13 * Complex.I :=
by sorry

end complex_fraction_simplification_l3614_361490


namespace m_value_l3614_361475

def A (m : ℝ) : Set ℝ := {-1, 3, m^2}
def B : Set ℝ := {3, 4}

theorem m_value (m : ℝ) (h : B ⊆ A m) : m = 2 ∨ m = -2 := by
  sorry

end m_value_l3614_361475


namespace distinct_points_on_curve_l3614_361482

theorem distinct_points_on_curve (a b : ℝ) : 
  a ≠ b →
  (a^2 + Real.sqrt π^4 = 2 * (Real.sqrt π)^2 * a + 1) →
  (b^2 + Real.sqrt π^4 = 2 * (Real.sqrt π)^2 * b + 1) →
  |a - b| = 2 := by sorry

end distinct_points_on_curve_l3614_361482


namespace ellipse_standard_equation_l3614_361410

/-- Given an ellipse with equation x²/m² + y²/n² = 1, where m > 0 and n > 0,
    whose right focus coincides with the focus of the parabola y² = 8x,
    and has an eccentricity of 1/2, prove that its standard equation is
    x²/16 + y²/12 = 1. -/
theorem ellipse_standard_equation
  (m n : ℝ)
  (h_m_pos : m > 0)
  (h_n_pos : n > 0)
  (h_focus : m^2 - n^2 = 4)  -- Right focus coincides with parabola focus (2, 0)
  (h_eccentricity : 2 / m = 1 / 2)  -- Eccentricity is 1/2
  : ∃ (x y : ℝ), x^2/16 + y^2/12 = 1 ∧ x^2/m^2 + y^2/n^2 = 1 :=
sorry

end ellipse_standard_equation_l3614_361410


namespace total_arms_collected_l3614_361452

theorem total_arms_collected (starfish_count : ℕ) (starfish_arms : ℕ) (seastar_count : ℕ) (seastar_arms : ℕ) :
  starfish_count = 7 →
  starfish_arms = 5 →
  seastar_count = 1 →
  seastar_arms = 14 →
  starfish_count * starfish_arms + seastar_count * seastar_arms = 49 :=
by
  sorry

end total_arms_collected_l3614_361452


namespace folk_song_competition_probability_l3614_361430

theorem folk_song_competition_probability : 
  ∀ (n m k : ℕ),
  n = 6 →  -- number of provinces
  m = 2 →  -- number of singers per province
  k = 4 →  -- number of winners selected
  (Nat.choose n 1 * Nat.choose (n - 1) 2 * Nat.choose m 1 * Nat.choose m 1) / 
  (Nat.choose (n * m) k) = 16 / 33 := by
  sorry

end folk_song_competition_probability_l3614_361430


namespace probability_of_two_boys_l3614_361455

theorem probability_of_two_boys (total : ℕ) (boys : ℕ) (girls : ℕ) 
  (h1 : total = 15)
  (h2 : boys = 9)
  (h3 : girls = 6)
  (h4 : total = boys + girls) :
  (Nat.choose boys 2 : ℚ) / (Nat.choose total 2 : ℚ) = 12 / 35 := by
sorry

end probability_of_two_boys_l3614_361455


namespace least_integer_with_12_factors_l3614_361425

-- Define a function to count the number of positive factors of a natural number
def countFactors (n : ℕ) : ℕ := sorry

-- Define a function to check if a number has exactly 12 factors
def has12Factors (n : ℕ) : Prop := countFactors n = 12

-- Theorem statement
theorem least_integer_with_12_factors :
  ∃ (k : ℕ), k > 0 ∧ has12Factors k ∧ ∀ (m : ℕ), m > 0 → has12Factors m → k ≤ m :=
sorry

end least_integer_with_12_factors_l3614_361425


namespace equation_solutions_l3614_361426

theorem equation_solutions : 
  (∃ (x₁ x₂ : ℝ), (x₁ = 3/5 ∧ x₂ = -3) ∧ 
    (2*x₁ - 3)^2 = 9*x₁^2 ∧ (2*x₂ - 3)^2 = 9*x₂^2) ∧
  (∃ (y₁ y₂ : ℝ), (y₁ = 2 ∧ y₂ = -1/2) ∧ 
    2*y₁*(y₁-2) + y₁ = 2 ∧ 2*y₂*(y₂-2) + y₂ = 2) :=
by sorry

end equation_solutions_l3614_361426


namespace multiplicative_inverse_144_mod_941_l3614_361432

theorem multiplicative_inverse_144_mod_941 : ∃ n : ℤ, 
  0 ≤ n ∧ n < 941 ∧ (144 * n) % 941 = 1 := by
  use 364
  sorry

end multiplicative_inverse_144_mod_941_l3614_361432


namespace base_conversion_theorem_l3614_361443

-- Define a function to convert a number from any base to base 10
def to_base_10 (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * base^i) 0

-- Define the given numbers in their respective bases
def num1 : List Nat := [2, 4, 3]
def den1 : List Nat := [1, 3]
def num2 : List Nat := [2, 0, 4]
def den2 : List Nat := [2, 3]

-- Convert the numbers to base 10
def num1_base10 : Nat := to_base_10 num1 8
def den1_base10 : Nat := to_base_10 den1 4
def num2_base10 : Nat := to_base_10 num2 7
def den2_base10 : Nat := to_base_10 den2 5

-- Define the theorem
theorem base_conversion_theorem :
  (num1_base10 : ℚ) / den1_base10 + (num2_base10 : ℚ) / den2_base10 = 31 + 51 / 91 :=
by sorry

end base_conversion_theorem_l3614_361443


namespace probability_all_players_have_initial_coins_l3614_361428

/-- Represents a player in the game -/
inductive Player : Type
| Alice : Player
| Bob : Player
| Charlie : Player
| Dana : Player

/-- Represents a ball color -/
inductive BallColor : Type
| Blue : BallColor
| Red : BallColor
| White : BallColor
| Yellow : BallColor

/-- Represents the state of the game -/
structure GameState :=
  (coins : Player → ℕ)
  (round : ℕ)

/-- Represents a single round of the game -/
def play_round (state : GameState) : GameState :=
  sorry

/-- Probability of a specific outcome in a single round -/
def round_probability : ℚ :=
  12 / 120

/-- The game consists of 5 rounds -/
def num_rounds : ℕ := 5

/-- The initial number of coins for each player -/
def initial_coins : ℕ := 5

/-- Theorem stating the probability of all players having the initial number of coins after the game -/
theorem probability_all_players_have_initial_coins :
  (round_probability ^ num_rounds : ℚ) = 1 / 10000 := by
  sorry

end probability_all_players_have_initial_coins_l3614_361428


namespace lines_non_intersecting_l3614_361439

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the property of a line being parallel to a plane
variable (parallel_to_plane : Line → Plane → Prop)

-- Define the property of a line being contained within a plane
variable (contained_in_plane : Line → Plane → Prop)

-- Define the property of two lines being non-intersecting
variable (non_intersecting : Line → Line → Prop)

-- State the theorem
theorem lines_non_intersecting
  (l a : Line) (α : Plane)
  (h1 : parallel_to_plane l α)
  (h2 : contained_in_plane a α) :
  non_intersecting l a :=
sorry

end lines_non_intersecting_l3614_361439


namespace cos_90_degrees_is_zero_l3614_361472

theorem cos_90_degrees_is_zero :
  let cos_36 : ℝ := (1 + Real.sqrt 5) / 4
  let cos_54 : ℝ := (1 - Real.sqrt 5) / 4
  let sin_36 : ℝ := Real.sqrt (10 - 2 * Real.sqrt 5) / 4
  let sin_54 : ℝ := Real.sqrt (10 + 2 * Real.sqrt 5) / 4
  let cos_sum := cos_36 * cos_54 - sin_36 * sin_54
  cos_sum = 0 := by sorry

end cos_90_degrees_is_zero_l3614_361472


namespace extreme_values_depend_on_a_consistent_monotonicity_implies_b_bound_max_ab_difference_l3614_361419

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x
def g (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x

-- Define the derivatives of f and g
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + a
def g' (b : ℝ) (x : ℝ) : ℝ := 2*x + b

-- Define consistent monotonicity
def consistent_monotonicity (a b : ℝ) (l : Set ℝ) : Prop :=
  ∀ x ∈ l, f' a x * g' b x ≥ 0

theorem extreme_values_depend_on_a (a : ℝ) : 
  (a ≥ 0 → ∀ x : ℝ, f' a x ≥ 0) ∧ 
  (a < 0 → ∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ f' a x₁ < 0 ∧ f' a x₂ > 0) :=
sorry

theorem consistent_monotonicity_implies_b_bound (a b : ℝ) :
  a > 0 → consistent_monotonicity a b { x | x ≥ -2 } → b ≥ 4 :=
sorry

theorem max_ab_difference (a b : ℝ) :
  a < 0 → a ≠ b → consistent_monotonicity a b (Set.Ioo a b) → |a - b| ≤ 1/3 :=
sorry

end extreme_values_depend_on_a_consistent_monotonicity_implies_b_bound_max_ab_difference_l3614_361419


namespace point_inside_circle_l3614_361441

theorem point_inside_circle (r d : ℝ) (hr : r = 6) (hd : d = 4) :
  d < r → ∃ (P : ℝ × ℝ) (O : ℝ × ℝ), ‖P - O‖ = d ∧ P ∈ interior {x | ‖x - O‖ ≤ r} :=
by sorry

end point_inside_circle_l3614_361441


namespace number_of_planes_l3614_361437

/-- Given an air exhibition with commercial planes, prove the number of planes
    when the total number of wings and wings per plane are known. -/
theorem number_of_planes (total_wings : ℕ) (wings_per_plane : ℕ) (h1 : total_wings = 90) (h2 : wings_per_plane = 2) :
  total_wings / wings_per_plane = 45 := by
  sorry

#check number_of_planes

end number_of_planes_l3614_361437


namespace prob_reach_edge_in_six_hops_l3614_361444

/-- Represents the 4x4 grid --/
inductive Grid
| Center : Grid
| Edge : Grid

/-- Represents the possible directions of movement --/
inductive Direction
| Up | Down | Left | Right

/-- Defines the movement rules on the grid --/
def move (g : Grid) (d : Direction) : Grid :=
  match g with
  | Grid.Center => Grid.Edge  -- Simplified for this problem
  | Grid.Edge => Grid.Edge

/-- Calculates the probability of reaching an edge square within n hops --/
def prob_reach_edge (n : ℕ) : ℚ :=
  sorry  -- Proof to be implemented

/-- Main theorem: The probability of reaching an edge square within 6 hops is 211/256 --/
theorem prob_reach_edge_in_six_hops :
  prob_reach_edge 6 = 211 / 256 :=
sorry

end prob_reach_edge_in_six_hops_l3614_361444


namespace total_dress_designs_l3614_361492

/-- The number of fabric colors available -/
def num_colors : ℕ := 5

/-- The number of patterns available -/
def num_patterns : ℕ := 4

/-- The number of sleeve length options -/
def num_sleeve_lengths : ℕ := 2

/-- Each dress design requires exactly one color, one pattern, and one sleeve length -/
axiom dress_design_requirement : True

/-- The total number of different dress designs possible -/
def total_designs : ℕ := num_colors * num_patterns * num_sleeve_lengths

/-- Theorem stating that the total number of different dress designs is 40 -/
theorem total_dress_designs : total_designs = 40 := by
  sorry

end total_dress_designs_l3614_361492


namespace complement_A_intersect_B_l3614_361486

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | 0 < x ∧ x < 2}

def B : Set ℝ := {x | abs x ≤ 1}

theorem complement_A_intersect_B :
  (Set.compl A ∩ B) = {x : ℝ | -1 ≤ x ∧ x ≤ 0} := by sorry

end complement_A_intersect_B_l3614_361486


namespace quadratic_sum_and_square_sum_l3614_361459

theorem quadratic_sum_and_square_sum (a b c d m n : ℕ+) 
  (h1 : a^2 + b^2 + c^2 + d^2 = 1989)
  (h2 : a + b + c + d = m^2)
  (h3 : max a (max b (max c d)) = n^2) :
  m = 9 ∧ n = 6 := by
sorry

end quadratic_sum_and_square_sum_l3614_361459


namespace girls_to_boys_fraction_l3614_361405

theorem girls_to_boys_fraction (total : ℕ) (girls : ℕ) (h1 : total = 35) (h2 : girls = 10) :
  (girls : ℚ) / ((total - girls) : ℚ) = 2 / 5 := by
  sorry

end girls_to_boys_fraction_l3614_361405


namespace truck_wheels_count_l3614_361465

/-- Calculates the toll for a truck given the number of axles -/
def toll (x : ℕ) : ℚ := 3.50 + 0.50 * (x - 2)

/-- Calculates the total number of wheels on a truck given the number of axles -/
def totalWheels (x : ℕ) : ℕ := 2 + 4 * (x - 1)

theorem truck_wheels_count :
  ∃ (x : ℕ), 
    x > 0 ∧
    toll x = 5 ∧
    totalWheels x = 18 :=
by sorry

end truck_wheels_count_l3614_361465


namespace lumberjack_firewood_l3614_361402

/-- Calculates the total number of firewood pieces produced by a lumberjack --/
theorem lumberjack_firewood (trees : ℕ) (logs_per_tree : ℕ) (pieces_per_log : ℕ) 
  (h1 : logs_per_tree = 4)
  (h2 : pieces_per_log = 5)
  (h3 : trees = 25) :
  trees * logs_per_tree * pieces_per_log = 500 := by
  sorry

#check lumberjack_firewood

end lumberjack_firewood_l3614_361402


namespace rice_price_fall_l3614_361445

theorem rice_price_fall (old_price new_price : ℝ) 
  (h : 40 * old_price = 50 * new_price) : 
  (old_price - new_price) / old_price = 1/5 := by
sorry

end rice_price_fall_l3614_361445


namespace fourth_game_shots_l3614_361406

/-- Given a basketball player's performance over four games, calculate the number of successful shots in the fourth game. -/
theorem fourth_game_shots (initial_shots initial_made fourth_game_shots : ℕ) 
  (h1 : initial_shots = 30)
  (h2 : initial_made = 12)
  (h3 : fourth_game_shots = 10)
  (h4 : (initial_made : ℚ) / initial_shots = 2/5)
  (h5 : ((initial_made + x) : ℚ) / (initial_shots + fourth_game_shots) = 1/2) :
  x = 8 :=
sorry

end fourth_game_shots_l3614_361406


namespace count_integer_points_l3614_361499

def point_A : ℤ × ℤ := (2, 3)
def point_B : ℤ × ℤ := (150, 903)

def is_between (p q r : ℤ × ℤ) : Prop :=
  (p.1 < q.1 ∧ q.1 < r.1) ∨ (r.1 < q.1 ∧ q.1 < p.1)

def on_line (p q r : ℤ × ℤ) : Prop :=
  (r.2 - p.2) * (q.1 - p.1) = (q.2 - p.2) * (r.1 - p.1)

def integer_points_between : Prop :=
  ∃ (S : Finset (ℤ × ℤ)),
    S.card = 4 ∧
    (∀ p ∈ S, is_between point_A p point_B ∧ on_line point_A point_B p) ∧
    (∀ p : ℤ × ℤ, is_between point_A p point_B ∧ on_line point_A point_B p → p ∈ S)

theorem count_integer_points : integer_points_between := by
  sorry

end count_integer_points_l3614_361499


namespace six_balls_three_boxes_l3614_361414

/-- The number of ways to distribute distinguishable balls into distinguishable boxes -/
def distribute_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  num_boxes ^ num_balls

/-- Theorem: There are 729 ways to distribute 6 distinguishable balls into 3 distinguishable boxes -/
theorem six_balls_three_boxes :
  distribute_balls 6 3 = 729 := by
  sorry

end six_balls_three_boxes_l3614_361414


namespace count_cow_herds_l3614_361429

/-- Given a farm with cows organized into herds, this theorem proves
    the number of herds given the total number of cows and the number
    of cows per herd. -/
theorem count_cow_herds (total_cows : ℕ) (cows_per_herd : ℕ) 
    (h1 : total_cows = 320) (h2 : cows_per_herd = 40) :
    total_cows / cows_per_herd = 8 := by
  sorry

end count_cow_herds_l3614_361429


namespace binomial_10_0_l3614_361473

theorem binomial_10_0 : (10 : ℕ).choose 0 = 1 := by
  sorry

end binomial_10_0_l3614_361473


namespace pregnant_cow_percentage_l3614_361412

theorem pregnant_cow_percentage (total_cows : ℕ) (female_percentage : ℚ) (pregnant_cows : ℕ) : 
  total_cows = 44 →
  female_percentage = 1/2 →
  pregnant_cows = 11 →
  (pregnant_cows : ℚ) / (female_percentage * total_cows) = 1/2 := by
  sorry

end pregnant_cow_percentage_l3614_361412


namespace brand_z_fraction_l3614_361493

/-- Represents the state of the fuel tank -/
structure TankState where
  z : ℚ  -- Amount of brand Z gasoline
  y : ℚ  -- Amount of brand Y gasoline

/-- Fills the tank with brand Y gasoline when it's partially empty -/
def fillWithY (s : TankState) (emptyFraction : ℚ) : TankState :=
  { z := s.z, y := s.y + emptyFraction }

/-- Fills the tank with brand Z gasoline when it's partially empty -/
def fillWithZ (s : TankState) (emptyFraction : ℚ) : TankState :=
  { z := s.z + emptyFraction, y := s.y }

/-- Empties the tank by a given fraction -/
def emptyTank (s : TankState) (emptyFraction : ℚ) : TankState :=
  { z := s.z * (1 - emptyFraction), y := s.y * (1 - emptyFraction) }

/-- The final state of the tank after the described filling process -/
def finalState : TankState :=
  let s1 := { z := 1, y := 0 }
  let s2 := fillWithY (emptyTank s1 (3/4)) (3/4)
  let s3 := fillWithZ (emptyTank s2 (1/2)) (1/2)
  fillWithY (emptyTank s3 (1/2)) (1/2)

/-- The fraction of brand Z gasoline in the final state is 5/16 -/
theorem brand_z_fraction :
  finalState.z / (finalState.z + finalState.y) = 5/16 := by sorry

end brand_z_fraction_l3614_361493


namespace inscribed_squares_ratio_l3614_361442

/-- A right triangle with sides 6, 8, and 10 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_triangle : a^2 + b^2 = c^2
  a_eq : a = 6
  b_eq : b = 8
  c_eq : c = 10

/-- Square inscribed in the triangle with one vertex at the right angle -/
def inscribed_square_right_angle (t : RightTriangle) (x : ℝ) : Prop :=
  x > 0 ∧ x ≤ t.a ∧ x ≤ t.b ∧ x / t.a = x / t.b

/-- Square inscribed in the triangle with one side along the hypotenuse -/
def inscribed_square_hypotenuse (t : RightTriangle) (y : ℝ) : Prop :=
  y > 0 ∧ y ≤ t.c ∧ y / t.c = (6/5 * y + 8/5 * y) / (t.a + t.b)

theorem inscribed_squares_ratio (t : RightTriangle) (x y : ℝ) 
  (hx : inscribed_square_right_angle t x) (hy : inscribed_square_hypotenuse t y) : 
  x / y = 111 / 175 := by
  sorry

end inscribed_squares_ratio_l3614_361442


namespace rectangular_plot_length_l3614_361448

theorem rectangular_plot_length
  (width : ℝ)
  (num_poles : ℕ)
  (pole_spacing : ℝ)
  (h1 : width = 50)
  (h2 : num_poles = 14)
  (h3 : pole_spacing = 20)
  : ∃ (length : ℝ), length = 80 ∧ 2 * (length + width) = (num_poles - 1) * pole_spacing :=
by sorry

end rectangular_plot_length_l3614_361448


namespace tan_half_angle_problem_l3614_361411

theorem tan_half_angle_problem (α : Real) (h : Real.tan (α / 2) = 2) :
  (Real.tan (α + Real.pi / 4) = -1 / 7) ∧
  ((6 * Real.sin α + Real.cos α) / (3 * Real.sin α - 2 * Real.cos α) = 7 / 6) := by
  sorry

end tan_half_angle_problem_l3614_361411


namespace arithmetic_square_root_of_sqrt_16_l3614_361421

theorem arithmetic_square_root_of_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 := by
  sorry

end arithmetic_square_root_of_sqrt_16_l3614_361421


namespace probability_ace_second_draw_l3614_361461

/-- The probability of drawing an Ace in the second draw without replacement from a deck of 52 cards, given that an Ace was drawn in the first draw. -/
theorem probability_ace_second_draw (initial_deck_size : ℕ) (initial_aces : ℕ) 
  (h1 : initial_deck_size = 52)
  (h2 : initial_aces = 4)
  (h3 : initial_aces > 0) :
  (initial_aces - 1 : ℚ) / (initial_deck_size - 1 : ℚ) = 1 / 17 := by
  sorry

end probability_ace_second_draw_l3614_361461


namespace simplify_and_rationalize_l3614_361457

theorem simplify_and_rationalize :
  (Real.sqrt 3 / Real.sqrt 4 + Real.sqrt 4 / Real.sqrt 5) * (Real.sqrt 5 / Real.sqrt 6) =
  (Real.sqrt 10 + 2 * Real.sqrt 2) / 4 := by sorry

end simplify_and_rationalize_l3614_361457


namespace no_guaranteed_win_strategy_l3614_361420

/-- Represents a game state with the current number on the board -/
structure GameState where
  number : ℕ

/-- Represents a player's move, adding a digit to the number -/
inductive Move
| PrependDigit (d : ℕ) : Move
| AppendDigit (d : ℕ) : Move
| InsertDigit (d : ℕ) (pos : ℕ) : Move

/-- Apply a move to the current game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  sorry

/-- Check if a number is a perfect square -/
def isPerfectSquare (n : ℕ) : Bool :=
  sorry

/-- Theorem stating that no player can guarantee a win -/
theorem no_guaranteed_win_strategy :
  ∀ (strategy : GameState → Move),
  ∃ (opponent_moves : List Move),
  let final_state := opponent_moves.foldl applyMove ⟨7⟩
  ¬ isPerfectSquare final_state.number :=
sorry

end no_guaranteed_win_strategy_l3614_361420


namespace complex_power_2018_l3614_361463

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_power_2018 : ((1 + i) / (1 - i)) ^ 2018 = -1 := by
  sorry

end complex_power_2018_l3614_361463


namespace original_rate_l3614_361480

/-- Given a reduction of 'a' yuan followed by a 20% reduction resulting in a final rate of 'b' yuan per minute, 
    the original rate was a + 1.25b yuan per minute. -/
theorem original_rate (a b : ℝ) : 
  (∃ x : ℝ, 0.8 * (x - a) = b) → 
  (∃ x : ℝ, x = a + 1.25 * b ∧ 0.8 * (x - a) = b) :=
by sorry

end original_rate_l3614_361480


namespace fixed_costs_calculation_l3614_361436

/-- The fixed monthly costs for a computer manufacturer producing electronic components -/
def fixed_monthly_costs : ℝ := 16699.50

/-- The production cost per component -/
def production_cost : ℝ := 80

/-- The shipping cost per component -/
def shipping_cost : ℝ := 7

/-- The number of components produced and sold per month -/
def monthly_units : ℕ := 150

/-- The lowest selling price per component for break-even -/
def selling_price : ℝ := 198.33

theorem fixed_costs_calculation :
  fixed_monthly_costs = 
    selling_price * monthly_units - 
    (production_cost + shipping_cost) * monthly_units :=
by sorry

end fixed_costs_calculation_l3614_361436


namespace tripling_radius_and_negative_quantity_l3614_361415

theorem tripling_radius_and_negative_quantity : ∀ (r : ℝ) (x : ℝ), 
  r > 0 → x < 0 → 
  (π * (3 * r)^2 ≠ 3 * (π * r^2)) ∧ (3 * x ≤ x) := by sorry

end tripling_radius_and_negative_quantity_l3614_361415


namespace product_mb_range_l3614_361424

/-- Given a line y = mx + b with slope m = 3/4 and y-intercept b = -1/3,
    the product mb satisfies -1 < mb < 0. -/
theorem product_mb_range (m b : ℚ) : 
  m = 3/4 → b = -1/3 → -1 < m * b ∧ m * b < 0 := by
  sorry

end product_mb_range_l3614_361424


namespace staircase_steps_l3614_361431

/-- The number of toothpicks used in a staircase with n steps -/
def toothpicks (n : ℕ) : ℕ := 2 * (n * (n + 1) * (2 * n + 1)) / 3

/-- Theorem stating that a staircase with 630 toothpicks has 9 steps -/
theorem staircase_steps : ∃ (n : ℕ), toothpicks n = 630 ∧ n = 9 := by
  sorry

end staircase_steps_l3614_361431


namespace total_sightings_is_280_l3614_361477

/-- Represents the data for a single month in the national park -/
structure MonthData where
  families : ℕ
  sightings : ℕ

/-- Calculates the total number of animal sightings over six months -/
def totalSightings (jan feb mar apr may jun : MonthData) : ℕ :=
  jan.sightings + feb.sightings + mar.sightings + apr.sightings + may.sightings + jun.sightings

/-- Theorem stating that the total number of animal sightings is 280 -/
theorem total_sightings_is_280 
  (jan : MonthData)
  (feb : MonthData)
  (mar : MonthData)
  (apr : MonthData)
  (may : MonthData)
  (jun : MonthData)
  (h1 : jan.families = 100 ∧ jan.sightings = 26)
  (h2 : feb.families = 150 ∧ feb.sightings = 78)
  (h3 : mar.families = 120 ∧ mar.sightings = 39)
  (h4 : apr.families = 204 ∧ apr.sightings = 55)
  (h5 : may.families = 204 ∧ may.sightings = 41)
  (h6 : jun.families = 265 ∧ jun.sightings = 41) :
  totalSightings jan feb mar apr may jun = 280 := by
  sorry

#check total_sightings_is_280

end total_sightings_is_280_l3614_361477


namespace smallest_multiple_of_3_to_7_l3614_361471

theorem smallest_multiple_of_3_to_7 : 
  ∃ (N : ℕ), N > 0 ∧ 
    (∀ (k : ℕ), k > 0 ∧ k < N → 
      ¬(3 ∣ k ∧ 4 ∣ k ∧ 5 ∣ k ∧ 6 ∣ k ∧ 7 ∣ k)) ∧
    (3 ∣ N ∧ 4 ∣ N ∧ 5 ∣ N ∧ 6 ∣ N ∧ 7 ∣ N) ∧
    N = 420 :=
by sorry

end smallest_multiple_of_3_to_7_l3614_361471


namespace equation_solution_existence_l3614_361491

theorem equation_solution_existence (a : ℝ) :
  (∃ x : ℝ, 3 * 4^(x - 2) + 27 = a + a * 4^(x - 2)) ↔ 3 < a ∧ a < 27 := by
  sorry

end equation_solution_existence_l3614_361491


namespace distinct_cube_paintings_eq_30_l3614_361449

/-- The number of faces on a cube -/
def num_faces : ℕ := 6

/-- The number of available colors -/
def num_colors : ℕ := 6

/-- The number of rotational symmetries of a cube -/
def num_rotations : ℕ := 24

/-- The number of distinct ways to paint a cube -/
def distinct_cube_paintings : ℕ := (num_colors.factorial) / num_rotations

theorem distinct_cube_paintings_eq_30 : distinct_cube_paintings = 30 := by
  sorry

end distinct_cube_paintings_eq_30_l3614_361449


namespace inequality_proof_l3614_361435

theorem inequality_proof (a b c d : ℝ) 
  (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d) 
  (h_sum : a*b + b*c + c*d + d*a = 1) : 
  (a^3 / (b + c + d)) + (b^3 / (a + c + d)) + 
  (c^3 / (a + b + d)) + (d^3 / (a + b + c)) ≥ 1/3 := by
sorry

end inequality_proof_l3614_361435


namespace root_product_plus_one_l3614_361496

theorem root_product_plus_one (r s t : ℝ) : 
  r^3 - 15*r^2 + 25*r - 10 = 0 →
  s^3 - 15*s^2 + 25*s - 10 = 0 →
  t^3 - 15*t^2 + 25*t - 10 = 0 →
  (1+r)*(1+s)*(1+t) = 51 := by
sorry

end root_product_plus_one_l3614_361496


namespace expression_approximately_equal_to_0_2436_l3614_361453

-- Define the expression
def expression : ℚ := (108 * 3 - (108 + 92)) / (92 * 7 - (45 * 3))

-- State the theorem
theorem expression_approximately_equal_to_0_2436 :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 0.00005 ∧ |expression - 0.2436| < ε := by
  sorry

end expression_approximately_equal_to_0_2436_l3614_361453


namespace probability_through_C_l3614_361403

theorem probability_through_C (total_paths : ℕ) (paths_A_to_C : ℕ) (paths_C_to_B : ℕ) :
  total_paths = Nat.choose 6 3 →
  paths_A_to_C = Nat.choose 3 2 →
  paths_C_to_B = Nat.choose 3 1 →
  (paths_A_to_C * paths_C_to_B : ℚ) / total_paths = 21 / 32 :=
by sorry

end probability_through_C_l3614_361403


namespace exists_specific_number_l3614_361460

theorem exists_specific_number : ∃ y : ℕ+, 
  (y.val % 4 = 0) ∧ 
  (y.val % 5 = 0) ∧ 
  (y.val % 7 = 0) ∧ 
  (y.val % 13 = 0) ∧ 
  (y.val % 8 ≠ 0) ∧ 
  (y.val % 15 ≠ 0) ∧ 
  (y.val % 50 ≠ 0) ∧ 
  (y.val % 10 = 0) ∧ 
  (y.val = 1820) :=
by sorry

end exists_specific_number_l3614_361460


namespace sqrt_equation_solution_l3614_361454

theorem sqrt_equation_solution :
  ∃! z : ℚ, Real.sqrt (5 - 4 * z) = 7 :=
by
  -- Proof goes here
  sorry

end sqrt_equation_solution_l3614_361454


namespace g_composition_of_three_l3614_361498

def g (x : ℝ) : ℝ := 3 * x + 2

theorem g_composition_of_three : g (g (g 3)) = 107 := by
  sorry

end g_composition_of_three_l3614_361498


namespace contradiction_assumption_l3614_361440

theorem contradiction_assumption (x y : ℝ) (h : x > y) : 
  ¬(x^3 > y^3) ↔ x^3 ≤ y^3 := by
sorry

end contradiction_assumption_l3614_361440


namespace groom_age_l3614_361478

theorem groom_age (bride_age groom_age : ℕ) : 
  bride_age = groom_age + 19 →
  bride_age + groom_age = 185 →
  groom_age = 83 := by
sorry

end groom_age_l3614_361478


namespace quadratic_inequality_solution_l3614_361446

theorem quadratic_inequality_solution (a : ℝ) : 
  (a > 0 ∧ ∃ x : ℝ, x^2 - 8*x + a < 0) ↔ (0 < a ∧ a < 16) :=
by sorry

end quadratic_inequality_solution_l3614_361446


namespace passing_marks_l3614_361423

theorem passing_marks (T : ℝ) (P : ℝ) 
  (h1 : 0.20 * T = P - 40)
  (h2 : 0.30 * T = P + 20) : 
  P = 160 := by
sorry

end passing_marks_l3614_361423


namespace jimmy_matchbooks_count_l3614_361400

/-- The number of matches in one matchbook -/
def matches_per_matchbook : ℕ := 24

/-- The number of matches equivalent to one stamp -/
def matches_per_stamp : ℕ := 12

/-- The number of stamps Tonya initially had -/
def tonya_initial_stamps : ℕ := 13

/-- The number of stamps Tonya had left after trading -/
def tonya_final_stamps : ℕ := 3

/-- The number of matchbooks Jimmy had -/
def jimmy_matchbooks : ℕ := 5

theorem jimmy_matchbooks_count :
  jimmy_matchbooks * matches_per_matchbook = 
    (tonya_initial_stamps - tonya_final_stamps) * matches_per_stamp :=
by sorry

end jimmy_matchbooks_count_l3614_361400


namespace factorial_ratio_equals_two_l3614_361447

theorem factorial_ratio_equals_two : (Nat.factorial 10 * Nat.factorial 4 * Nat.factorial 3) / (Nat.factorial 9 * Nat.factorial 6) = 2 := by
  sorry

end factorial_ratio_equals_two_l3614_361447


namespace emma_bank_balance_emma_final_balance_l3614_361484

theorem emma_bank_balance (initial_balance : ℝ) (shoe_percentage : ℝ) 
  (tuesday_deposit_percentage : ℝ) (wednesday_deposit_percentage : ℝ) 
  (final_withdrawal_percentage : ℝ) : ℝ :=
  let shoe_cost := initial_balance * shoe_percentage
  let monday_balance := initial_balance - shoe_cost
  let tuesday_deposit := shoe_cost * tuesday_deposit_percentage
  let tuesday_balance := monday_balance + tuesday_deposit
  let wednesday_deposit := shoe_cost * wednesday_deposit_percentage
  let wednesday_balance := tuesday_balance + wednesday_deposit
  let final_withdrawal := wednesday_balance * final_withdrawal_percentage
  let final_balance := wednesday_balance - final_withdrawal
  final_balance
  
theorem emma_final_balance : 
  emma_bank_balance 1200 0.08 0.25 1.5 0.05 = 1208.40 := by
  sorry

end emma_bank_balance_emma_final_balance_l3614_361484


namespace simplify_tan_cot_expression_l3614_361464

theorem simplify_tan_cot_expression :
  let tan_45 : Real := 1
  let cot_45 : Real := 1
  (tan_45^3 + cot_45^3) / (tan_45 + cot_45) = 1 := by
  sorry

end simplify_tan_cot_expression_l3614_361464


namespace fencing_cost_140m_perimeter_l3614_361418

/-- The cost of fencing a rectangular plot -/
def fencing_cost (width : ℝ) (rate : ℝ) : ℝ :=
  let length : ℝ := width + 10
  let perimeter : ℝ := 2 * (length + width)
  rate * perimeter

theorem fencing_cost_140m_perimeter :
  ∃ (width : ℝ),
    (2 * (width + (width + 10)) = 140) ∧
    (fencing_cost width 6.5 = 910) := by
  sorry

end fencing_cost_140m_perimeter_l3614_361418
