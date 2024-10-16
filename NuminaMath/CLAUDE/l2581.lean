import Mathlib

namespace NUMINAMATH_CALUDE_jerry_ring_toss_games_l2581_258119

/-- The number of games Jerry played in the ring toss game -/
def games_played (total_rings : ℕ) (rings_per_game : ℕ) : ℕ :=
  total_rings / rings_per_game

/-- Theorem: Jerry played 8 games of ring toss -/
theorem jerry_ring_toss_games : games_played 48 6 = 8 := by
  sorry

end NUMINAMATH_CALUDE_jerry_ring_toss_games_l2581_258119


namespace NUMINAMATH_CALUDE_center_is_five_l2581_258127

/-- Represents a 3x3 grid filled with numbers from 1 to 9 -/
def Grid := Fin 3 → Fin 3 → Fin 9

/-- Checks if two positions in the grid share an edge -/
def sharesEdge (p1 p2 : Fin 3 × Fin 3) : Prop :=
  (p1.1 = p2.1 ∧ (p1.2.val + 1 = p2.2.val ∨ p2.2.val + 1 = p1.2.val)) ∨
  (p1.2 = p2.2 ∧ (p1.1.val + 1 = p2.1.val ∨ p2.1.val + 1 = p1.1.val))

/-- Checks if the grid satisfies the consecutive number condition -/
def isConsecutive (g : Grid) : Prop :=
  ∀ p1 p2 : Fin 3 × Fin 3, sharesEdge p1 p2 →
    (g p1.1 p1.2).val + 1 = (g p2.1 p2.2).val ∨
    (g p2.1 p2.2).val + 1 = (g p1.1 p1.2).val

/-- The sum of corner numbers in the grid -/
def cornerSum (g : Grid) : Nat :=
  (g 0 0).val + (g 0 2).val + (g 2 0).val + (g 2 2).val

/-- All numbers from 1 to 9 are used in the grid -/
def usesAllNumbers (g : Grid) : Prop :=
  ∀ n : Fin 9, ∃ i j : Fin 3, g i j = n

theorem center_is_five (g : Grid)
  (h1 : isConsecutive g)
  (h2 : cornerSum g = 20)
  (h3 : usesAllNumbers g) :
  g 1 1 = 5 :=
sorry

end NUMINAMATH_CALUDE_center_is_five_l2581_258127


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l2581_258180

theorem smallest_integer_with_remainders : ∃! M : ℕ,
  (M > 0) ∧
  (M % 7 = 6) ∧
  (M % 8 = 7) ∧
  (M % 9 = 8) ∧
  (M % 10 = 9) ∧
  (M % 11 = 10) ∧
  (M % 12 = 11) ∧
  (∀ n : ℕ, n > 0 ∧
    n % 7 = 6 ∧
    n % 8 = 7 ∧
    n % 9 = 8 ∧
    n % 10 = 9 ∧
    n % 11 = 10 ∧
    n % 12 = 11 → n ≥ M) ∧
  M = 27719 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l2581_258180


namespace NUMINAMATH_CALUDE_garage_sale_pricing_l2581_258181

theorem garage_sale_pricing (prices : Finset ℕ) (radio_price : ℕ) (n : ℕ) :
  prices.card = 36 →
  prices.toList.Nodup →
  radio_price ∈ prices →
  (prices.filter (λ x => x > radio_price)).card = n - 1 →
  (prices.filter (λ x => x < radio_price)).card = 21 →
  n = 16 := by
  sorry

end NUMINAMATH_CALUDE_garage_sale_pricing_l2581_258181


namespace NUMINAMATH_CALUDE_mary_total_cards_l2581_258116

/-- The number of baseball cards Mary has now -/
def mary_cards : ℕ := 76

/-- Mary's initial number of baseball cards -/
def initial_cards : ℕ := 18

/-- Number of torn cards from Mary's initial set -/
def torn_cards : ℕ := 8

/-- Number of cards Fred gave to Mary -/
def fred_cards : ℕ := 26

/-- Number of cards Mary bought -/
def bought_cards : ℕ := 40

/-- Theorem stating that Mary now has 76 baseball cards -/
theorem mary_total_cards : 
  mary_cards = initial_cards - torn_cards + fred_cards + bought_cards :=
by sorry

end NUMINAMATH_CALUDE_mary_total_cards_l2581_258116


namespace NUMINAMATH_CALUDE_largest_integer_with_remainder_ninety_eight_satisfies_conditions_ninety_eight_is_largest_l2581_258125

theorem largest_integer_with_remainder : 
  ∀ n : ℕ, n < 100 ∧ n % 6 = 2 → n ≤ 98 :=
by
  sorry

theorem ninety_eight_satisfies_conditions : 
  98 < 100 ∧ 98 % 6 = 2 :=
by
  sorry

theorem ninety_eight_is_largest :
  ∀ n : ℕ, n < 100 ∧ n % 6 = 2 → n ≤ 98 ∧ 
  ∃ m : ℕ, m < 100 ∧ m % 6 = 2 ∧ m = 98 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_integer_with_remainder_ninety_eight_satisfies_conditions_ninety_eight_is_largest_l2581_258125


namespace NUMINAMATH_CALUDE_election_votes_l2581_258146

theorem election_votes (total_votes : ℕ) 
  (h1 : total_votes > 0)
  (h2 : (62 * total_votes) / 100 - (38 * total_votes) / 100 = 384) :
  (62 * total_votes) / 100 = 992 := by
  sorry

end NUMINAMATH_CALUDE_election_votes_l2581_258146


namespace NUMINAMATH_CALUDE_negative_one_is_root_l2581_258170

/-- The polynomial f(x) = x^3 + x^2 - 6x - 6 -/
def f (x : ℝ) : ℝ := x^3 + x^2 - 6*x - 6

/-- Theorem: -1 is a root of the polynomial f(x) = x^3 + x^2 - 6x - 6 -/
theorem negative_one_is_root : f (-1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_negative_one_is_root_l2581_258170


namespace NUMINAMATH_CALUDE_cubic_function_extrema_l2581_258184

theorem cubic_function_extrema (a b : ℝ) (h_a : a > 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^3 - 3 * a * x^2 + b
  (∀ x ∈ Set.Icc (-1) 2, f x ≤ 3) ∧
  (∃ x ∈ Set.Icc (-1) 2, f x = 3) ∧
  (∀ x ∈ Set.Icc (-1) 2, f x ≥ -21) ∧
  (∃ x ∈ Set.Icc (-1) 2, f x = -21) →
  a = 6 ∧ b = 3 := by
sorry

end NUMINAMATH_CALUDE_cubic_function_extrema_l2581_258184


namespace NUMINAMATH_CALUDE_equation_solution_l2581_258172

theorem equation_solution : ∃ x₁ x₂ : ℝ, 
  (x₁ * (x₁ - 2) + x₁ - 2 = 0) ∧ 
  (x₂ * (x₂ - 2) + x₂ - 2 = 0) ∧ 
  x₁ = 2 ∧ x₂ = -1 ∧ 
  ∀ x : ℝ, x * (x - 2) + x - 2 = 0 → (x = x₁ ∨ x = x₂) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2581_258172


namespace NUMINAMATH_CALUDE_train_length_l2581_258130

/-- The length of a train given its speed and time to pass a point -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) (h1 : speed_kmh = 63) (h2 : time_s = 16) :
  speed_kmh * 1000 / 3600 * time_s = 280 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l2581_258130


namespace NUMINAMATH_CALUDE_discount_is_eleven_l2581_258108

/-- The discount on the first two books when ordering four books online --/
def discount_on_first_two_books : ℝ :=
  let free_shipping_threshold : ℝ := 50
  let book1_price : ℝ := 13
  let book2_price : ℝ := 15
  let book3_price : ℝ := 10
  let book4_price : ℝ := 10
  let additional_spend_needed : ℝ := 9
  let total_without_discount : ℝ := book1_price + book2_price + book3_price + book4_price
  let total_with_discount : ℝ := free_shipping_threshold + additional_spend_needed
  total_with_discount - total_without_discount

/-- Theorem stating that the discount on the first two books is $11.00 --/
theorem discount_is_eleven : discount_on_first_two_books = 11 := by
  sorry

end NUMINAMATH_CALUDE_discount_is_eleven_l2581_258108


namespace NUMINAMATH_CALUDE_johns_number_l2581_258177

theorem johns_number : ∃! n : ℕ, 1000 < n ∧ n < 3000 ∧ 200 ∣ n ∧ 45 ∣ n ∧ n = 1800 := by
  sorry

end NUMINAMATH_CALUDE_johns_number_l2581_258177


namespace NUMINAMATH_CALUDE_fraction_simplification_l2581_258168

theorem fraction_simplification (x y : ℝ) (h : 2*x - y ≠ 0) :
  (3*x) / (2*x - y) - (x + y) / (2*x - y) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2581_258168


namespace NUMINAMATH_CALUDE_negation_of_implication_l2581_258145

theorem negation_of_implication (a b c : ℝ) :
  ¬(a > b → a + c > b + c) ↔ (a ≤ b → a + c ≤ b + c) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_implication_l2581_258145


namespace NUMINAMATH_CALUDE_simplify_fraction_l2581_258175

theorem simplify_fraction : (120 : ℚ) / 2160 = 1 / 18 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2581_258175


namespace NUMINAMATH_CALUDE_students_after_yoongi_l2581_258144

/-- Given a group of students waiting for a bus, this theorem proves
    the number of students who came after a specific student. -/
theorem students_after_yoongi 
  (total : ℕ) 
  (before_jungkook : ℕ) 
  (h1 : total = 20) 
  (h2 : before_jungkook = 11) 
  (h3 : ∃ (before_yoongi : ℕ), before_yoongi + 1 = before_jungkook) : 
  ∃ (after_yoongi : ℕ), after_yoongi = total - (before_jungkook - 1) - 1 ∧ after_yoongi = 9 :=
by sorry

end NUMINAMATH_CALUDE_students_after_yoongi_l2581_258144


namespace NUMINAMATH_CALUDE_congruence_problem_l2581_258149

theorem congruence_problem : ∃! n : ℤ, 0 ≤ n ∧ n ≤ 4 ∧ n ≡ -1458 [ZMOD 5] ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l2581_258149


namespace NUMINAMATH_CALUDE_walking_problem_l2581_258159

/-- Represents the ratio of steps taken by the good walker to the bad walker in the same time -/
def step_ratio : ℚ := 100 / 60

/-- Represents the head start of the bad walker in steps -/
def head_start : ℕ := 100

/-- Represents the walking problem described in "Nine Chapters on the Mathematical Art" -/
theorem walking_problem (x y : ℚ) :
  (x - y = head_start) ∧ (x = step_ratio * y) ↔
  x - y = head_start ∧ x = (100 : ℚ) / 60 * y :=
sorry

end NUMINAMATH_CALUDE_walking_problem_l2581_258159


namespace NUMINAMATH_CALUDE_sum_of_six_consecutive_odd_iff_l2581_258186

/-- Predicate to check if an integer is odd -/
def IsOdd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

/-- Predicate to check if an integer can be written as the sum of six consecutive odd integers -/
def IsSumOfSixConsecutiveOdd (S : ℤ) : Prop :=
  IsOdd ((S - 30) / 6)

theorem sum_of_six_consecutive_odd_iff (S : ℤ) :
  IsSumOfSixConsecutiveOdd S ↔
  ∃ n : ℤ, S = n + (n + 2) + (n + 4) + (n + 6) + (n + 8) + (n + 10) ∧ IsOdd n :=
sorry

end NUMINAMATH_CALUDE_sum_of_six_consecutive_odd_iff_l2581_258186


namespace NUMINAMATH_CALUDE_matt_trading_profit_l2581_258198

/-- Represents the profit made from trading baseball cards -/
def tradingProfit (initialCardCount : ℕ) (initialCardValue : ℕ) 
                  (tradedCardCount : ℕ) (receivedCardValues : List ℕ) : ℤ :=
  let initialValue := initialCardCount * initialCardValue
  let tradedValue := tradedCardCount * initialCardValue
  let receivedValue := receivedCardValues.sum
  (receivedValue : ℤ) - (tradedValue : ℤ)

/-- Theorem stating that Matt's trading profit is $3 -/
theorem matt_trading_profit :
  tradingProfit 8 6 2 [2, 2, 2, 9] = 3 := by
  sorry

#eval tradingProfit 8 6 2 [2, 2, 2, 9]

end NUMINAMATH_CALUDE_matt_trading_profit_l2581_258198


namespace NUMINAMATH_CALUDE_popsicle_sticks_count_l2581_258121

theorem popsicle_sticks_count (gino_sticks : ℕ) (total_sticks : ℕ) (my_sticks : ℕ) : 
  gino_sticks = 63 → total_sticks = 113 → total_sticks = gino_sticks + my_sticks → my_sticks = 50 := by
  sorry

end NUMINAMATH_CALUDE_popsicle_sticks_count_l2581_258121


namespace NUMINAMATH_CALUDE_twelve_months_probability_l2581_258166

/-- Represents the card game "Twelve Months" -/
structure TwelveMonths where
  /-- Number of columns -/
  n : Nat
  /-- Number of cards per column -/
  m : Nat
  /-- Total number of cards -/
  total_cards : Nat
  /-- Condition: total cards equals m * n -/
  h_total : total_cards = m * n

/-- The probability of all cards being flipped in the "Twelve Months" game -/
def probability_all_flipped (game : TwelveMonths) : ℚ :=
  1 / game.n

/-- Theorem stating the probability of all cards being flipped in the "Twelve Months" game -/
theorem twelve_months_probability (game : TwelveMonths) 
  (h_columns : game.n = 12) 
  (h_cards_per_column : game.m = 4) : 
  probability_all_flipped game = 1 / 12 := by
  sorry

#eval probability_all_flipped ⟨12, 4, 48, rfl⟩

end NUMINAMATH_CALUDE_twelve_months_probability_l2581_258166


namespace NUMINAMATH_CALUDE_CaBr2_molecular_weight_l2581_258161

/-- The atomic weight of calcium in g/mol -/
def atomic_weight_Ca : ℝ := 40.08

/-- The atomic weight of bromine in g/mol -/
def atomic_weight_Br : ℝ := 79.904

/-- The number of calcium atoms in CaBr2 -/
def num_Ca_atoms : ℕ := 1

/-- The number of bromine atoms in CaBr2 -/
def num_Br_atoms : ℕ := 2

/-- The molecular weight of CaBr2 in g/mol -/
def molecular_weight_CaBr2 : ℝ :=
  atomic_weight_Ca * num_Ca_atoms + atomic_weight_Br * num_Br_atoms

theorem CaBr2_molecular_weight :
  molecular_weight_CaBr2 = 199.888 := by
  sorry

end NUMINAMATH_CALUDE_CaBr2_molecular_weight_l2581_258161


namespace NUMINAMATH_CALUDE_min_value_of_f_l2581_258156

/-- The function f(x) = 2x³ - 6x² + m -/
def f (m : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 6 * x^2 + m

theorem min_value_of_f (m : ℝ) :
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f m x ≥ f m y) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f m x = 3) →
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f m x ≤ f m y) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f m x = -37) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2581_258156


namespace NUMINAMATH_CALUDE_first_hour_distance_correct_l2581_258174

/-- The distance traveled by a car in the first hour, given that its speed increases by 2 km/h every hour and it travels 492 km in 12 hours -/
def first_hour_distance : ℝ :=
  let speed_increase : ℝ := 2
  let total_hours : ℕ := 12
  let total_distance : ℝ := 492
  30

/-- Theorem stating that the first hour distance is correct -/
theorem first_hour_distance_correct :
  let speed_increase : ℝ := 2
  let total_hours : ℕ := 12
  let total_distance : ℝ := 492
  (first_hour_distance + total_hours * (total_hours - 1) / 2 * speed_increase) * total_hours / 2 = total_distance :=
by
  sorry

#eval first_hour_distance

end NUMINAMATH_CALUDE_first_hour_distance_correct_l2581_258174


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2581_258194

-- Define the condition for a hyperbola
def is_hyperbola (k : ℝ) : Prop :=
  (k - 2) * (k - 6) < 0

-- Define the condition given in the problem
def condition (k : ℝ) : Prop :=
  1 < k ∧ k < 7

-- Theorem statement
theorem necessary_but_not_sufficient :
  (∀ k, is_hyperbola k → condition k) ∧
  (∃ k, condition k ∧ ¬is_hyperbola k) :=
sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2581_258194


namespace NUMINAMATH_CALUDE_quadratic_form_minimum_l2581_258117

theorem quadratic_form_minimum (x y : ℝ) : 
  3 * x^2 + 2 * x * y + y^2 - 6 * x + 4 * y + 9 ≥ 0 ∧ 
  (3 * x^2 + 2 * x * y + y^2 - 6 * x + 4 * y + 9 = 0 ↔ x = 2 ∧ y = -3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_form_minimum_l2581_258117


namespace NUMINAMATH_CALUDE_jeremy_songs_l2581_258112

def songs_problem (songs_yesterday : ℕ) (difference : ℕ) : Prop :=
  let songs_today : ℕ := songs_yesterday + difference
  let total_songs : ℕ := songs_yesterday + songs_today
  total_songs = 23

theorem jeremy_songs :
  songs_problem 9 5 := by
  sorry

end NUMINAMATH_CALUDE_jeremy_songs_l2581_258112


namespace NUMINAMATH_CALUDE_turning_to_similar_section_is_random_event_l2581_258195

/-- Represents the event of turning to a similar section in a textbook -/
def turning_to_similar_section : Type := Unit

/-- Defines the properties of the event -/
class EventProperties (α : Type) where
  not_guaranteed : ∀ (x y : α), x ≠ y → True
  possible : ∃ (x : α), True
  not_certain : ¬ (∀ (x : α), True)
  not_impossible : ∃ (x : α), True
  not_predictable : ∀ (x : α), ¬ (∀ (y : α), x = y)

/-- Defines a random event -/
class RandomEvent (α : Type) extends EventProperties α

/-- Theorem stating that turning to a similar section is a random event -/
theorem turning_to_similar_section_is_random_event :
  RandomEvent turning_to_similar_section :=
sorry

end NUMINAMATH_CALUDE_turning_to_similar_section_is_random_event_l2581_258195


namespace NUMINAMATH_CALUDE_abs_neg_sqrt_six_l2581_258182

theorem abs_neg_sqrt_six : |(-Real.sqrt 6)| = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_sqrt_six_l2581_258182


namespace NUMINAMATH_CALUDE_numberOfWaysTheorem_l2581_258160

/-- The number of ways to choose sets S_ij satisfying the given conditions -/
def numberOfWays (n : ℕ) : ℕ :=
  (Nat.factorial (2 * n)) * (2 ^ (n ^ 2))

/-- The theorem stating the number of ways to choose sets S_ij -/
theorem numberOfWaysTheorem (n : ℕ) :
  numberOfWays n = (Nat.factorial (2 * n)) * (2 ^ (n ^ 2)) :=
by sorry

end NUMINAMATH_CALUDE_numberOfWaysTheorem_l2581_258160


namespace NUMINAMATH_CALUDE_correct_answer_is_ten_l2581_258197

theorem correct_answer_is_ten (x : ℝ) (h : 3 * x = 90) : x / 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_correct_answer_is_ten_l2581_258197


namespace NUMINAMATH_CALUDE_sum_not_equal_product_l2581_258102

theorem sum_not_equal_product : (0 + 1 + 2 + 3) ≠ (0 * 1 * 2 * 3) := by
  sorry

end NUMINAMATH_CALUDE_sum_not_equal_product_l2581_258102


namespace NUMINAMATH_CALUDE_prob_at_least_one_event_l2581_258131

/-- The probability that at least one of two independent events occurs -/
theorem prob_at_least_one_event (A B : ℝ) (hA : 0 ≤ A ∧ A ≤ 1) (hB : 0 ≤ B ∧ B ≤ 1) 
  (hAval : A = 0.9) (hBval : B = 0.8) :
  1 - (1 - A) * (1 - B) = 0.98 := by
sorry

end NUMINAMATH_CALUDE_prob_at_least_one_event_l2581_258131


namespace NUMINAMATH_CALUDE_kody_age_is_32_l2581_258140

-- Define Mohamed's current age
def mohamed_current_age : ℕ := 2 * 30

-- Define Mohamed's age four years ago
def mohamed_past_age : ℕ := mohamed_current_age - 4

-- Define Kody's age four years ago
def kody_past_age : ℕ := mohamed_past_age / 2

-- Define Kody's current age
def kody_current_age : ℕ := kody_past_age + 4

-- Theorem stating Kody's current age
theorem kody_age_is_32 : kody_current_age = 32 := by
  sorry

end NUMINAMATH_CALUDE_kody_age_is_32_l2581_258140


namespace NUMINAMATH_CALUDE_probability_of_one_out_of_four_l2581_258136

theorem probability_of_one_out_of_four (S : Finset α) (h : S.card = 4) :
  ∀ a ∈ S, (1 : ℝ) / S.card = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_one_out_of_four_l2581_258136


namespace NUMINAMATH_CALUDE_cars_distribution_after_six_days_l2581_258192

/-- Represents the number of cars at each station and the number of days passed -/
structure CarDistribution :=
  (cars_a : ℕ)
  (cars_b : ℕ)
  (days : ℕ)

/-- Updates the car distribution for one day -/
def update_distribution (d : CarDistribution) : CarDistribution :=
  { cars_a := d.cars_a - 21 + 24,
    cars_b := d.cars_b + 21 - 24,
    days := d.days + 1 }

/-- Theorem stating that after 6 days, cars at A will be 7 times cars at B -/
theorem cars_distribution_after_six_days :
  ∃ (d : CarDistribution),
    d.cars_a = 192 ∧
    d.cars_b = 48 ∧
    d.days = 0 ∧
    (update_distribution^[6] d).cars_a = 7 * (update_distribution^[6] d).cars_b :=
sorry

end NUMINAMATH_CALUDE_cars_distribution_after_six_days_l2581_258192


namespace NUMINAMATH_CALUDE_S_bounds_l2581_258143

def S : Set ℝ := {y | ∃ x : ℝ, x ≥ 0 ∧ y = (3*x + 2)/(x + 1)}

theorem S_bounds : 
  ∃ (M m : ℝ), 
    (∀ y ∈ S, y ≤ M) ∧ 
    (∀ y ∈ S, y ≥ m) ∧ 
    (M ∉ S) ∧ 
    (m ∈ S) ∧
    (M = 3) ∧ 
    (m = 2) :=
by sorry

end NUMINAMATH_CALUDE_S_bounds_l2581_258143


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2581_258101

theorem sqrt_equation_solution :
  ∃! z : ℝ, Real.sqrt (10 + 3 * z) = 8 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2581_258101


namespace NUMINAMATH_CALUDE_product_divisible_by_four_probability_l2581_258164

/-- The set of integers from 6 to 18, inclusive -/
def IntegerRange : Set ℤ := {n : ℤ | 6 ≤ n ∧ n ≤ 18}

/-- The set of integers in IntegerRange that are divisible by 4 -/
def DivisibleBy4 : Set ℤ := {n : ℤ | n ∈ IntegerRange ∧ n % 4 = 0}

/-- The set of even integers in IntegerRange -/
def EvenInRange : Set ℤ := {n : ℤ | n ∈ IntegerRange ∧ n % 2 = 0}

/-- The number of ways to choose 2 distinct integers from IntegerRange -/
def TotalChoices : ℕ := Nat.choose (Finset.card (Finset.range 13)) 2

/-- The number of ways to choose 2 distinct integers from IntegerRange 
    such that their product is divisible by 4 -/
def FavorableChoices : ℕ := 33

theorem product_divisible_by_four_probability : 
  (FavorableChoices : ℚ) / TotalChoices = 33 / 78 := by sorry

end NUMINAMATH_CALUDE_product_divisible_by_four_probability_l2581_258164


namespace NUMINAMATH_CALUDE_cookies_with_three_cups_l2581_258165

/- Define the rate of cookies per cup of flour -/
def cookies_per_cup (total_cookies : ℕ) (total_cups : ℕ) : ℚ :=
  total_cookies / total_cups

/- Define the function to calculate cookies from cups of flour -/
def cookies_from_cups (rate : ℚ) (cups : ℕ) : ℚ :=
  rate * cups

/- Theorem statement -/
theorem cookies_with_three_cups 
  (h1 : cookies_per_cup 24 4 = 6) 
  (h2 : cookies_from_cups (cookies_per_cup 24 4) 3 = 18) : 
  ℕ := by
  sorry

end NUMINAMATH_CALUDE_cookies_with_three_cups_l2581_258165


namespace NUMINAMATH_CALUDE_product_of_six_consecutive_numbers_l2581_258106

theorem product_of_six_consecutive_numbers (n : ℕ) (h : n = 3) :
  (n - 1) * n * (n + 1) * (n + 2) * (n + 3) * (n + 4) = 5040 := by
  sorry

end NUMINAMATH_CALUDE_product_of_six_consecutive_numbers_l2581_258106


namespace NUMINAMATH_CALUDE_function_properties_l2581_258109

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^4 - 4*x^3 + a*x^2 - 1

-- Define the function g
def g (b : ℝ) (x : ℝ) : ℝ := b*x^2 - 1

-- Theorem statement
theorem function_properties :
  ∃ (a : ℝ),
    (∀ x y : ℝ, 0 ≤ x ∧ x < y ∧ y ≤ 1 → f a x < f a y) ∧
    (∀ x y : ℝ, 1 ≤ x ∧ x < y ∧ y ≤ 2 → f a x > f a y) ∧
    a = 4 ∧
    ∃ b : ℝ, (b = 0 ∨ b = 4) ∧
      (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = g b x₁ ∧ f a x₂ = g b x₂) ∧
      (∀ x₃ : ℝ, x₃ ≠ x₁ ∧ x₃ ≠ x₂ → f a x₃ ≠ g b x₃) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l2581_258109


namespace NUMINAMATH_CALUDE_rectangular_box_surface_area_l2581_258190

/-- Given a rectangular box with dimensions a, b, and c, if the sum of the lengths
    of the twelve edges is 140 and the distance from one corner to the farthest
    corner is 21, then the total surface area of the box is 784. -/
theorem rectangular_box_surface_area
  (a b c : ℝ)
  (edge_sum : a + b + c = 35)
  (diagonal : a^2 + b^2 + c^2 = 441) :
  2 * (a * b + b * c + c * a) = 784 := by
sorry

end NUMINAMATH_CALUDE_rectangular_box_surface_area_l2581_258190


namespace NUMINAMATH_CALUDE_minimum_score_for_english_l2581_258183

/-- Given the average score of two subjects and a desired average for three subjects,
    calculate the minimum score needed for the third subject. -/
def minimum_third_score (avg_two : ℝ) (desired_avg : ℝ) : ℝ :=
  3 * desired_avg - 2 * avg_two

theorem minimum_score_for_english (avg_two : ℝ) (desired_avg : ℝ)
  (h1 : avg_two = 90)
  (h2 : desired_avg ≥ 92) :
  minimum_third_score avg_two desired_avg ≥ 96 :=
sorry

end NUMINAMATH_CALUDE_minimum_score_for_english_l2581_258183


namespace NUMINAMATH_CALUDE_sine_sum_constant_l2581_258150

theorem sine_sum_constant (α : Real) :
  (Real.sin α) ^ 2 + (Real.sin (α + 60 * π / 180)) ^ 2 + (Real.sin (α + 120 * π / 180)) ^ 2 =
  (Real.sin (α - 60 * π / 180)) ^ 2 + (Real.sin α) ^ 2 + (Real.sin (α + 60 * π / 180)) ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_sine_sum_constant_l2581_258150


namespace NUMINAMATH_CALUDE_car_trip_average_speed_l2581_258100

/-- Calculates the average speed of a car trip given specific conditions -/
theorem car_trip_average_speed
  (total_time : ℝ)
  (initial_time : ℝ)
  (initial_speed : ℝ)
  (remaining_speed : ℝ)
  (h_total_time : total_time = 6)
  (h_initial_time : initial_time = 4)
  (h_initial_speed : initial_speed = 55)
  (h_remaining_speed : remaining_speed = 70) :
  (initial_speed * initial_time + remaining_speed * (total_time - initial_time)) / total_time = 60 :=
by sorry

end NUMINAMATH_CALUDE_car_trip_average_speed_l2581_258100


namespace NUMINAMATH_CALUDE_G_properties_l2581_258103

/-- The curve G defined by x³ + y³ - 6xy = 0 for x > 0 and y > 0 -/
def G : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 > 0 ∧ p.2 > 0 ∧ p.1^3 + p.2^3 - 6*p.1*p.2 = 0}

/-- The line y = x -/
def line_y_eq_x : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = p.2}

/-- The line x + y - 6 = 0 -/
def line_x_plus_y_eq_6 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + p.2 = 6}

theorem G_properties :
  (∀ p : ℝ × ℝ, p ∈ G → (p.2, p.1) ∈ G) ∧ 
  (∃! p : ℝ × ℝ, p ∈ G ∩ line_x_plus_y_eq_6) ∧
  (∀ p : ℝ × ℝ, p ∈ G → Real.sqrt (p.1^2 + p.2^2) ≤ 3 * Real.sqrt 2) ∧
  (∃ p : ℝ × ℝ, p ∈ G ∧ Real.sqrt (p.1^2 + p.2^2) = 3 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_G_properties_l2581_258103


namespace NUMINAMATH_CALUDE_inequality_proof_l2581_258104

theorem inequality_proof (x y z : ℝ) 
  (sum_zero : x + y + z = 0)
  (abs_sum_le_one : |x| + |y| + |z| ≤ 1) :
  x + y / 2 + z / 3 ≤ 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2581_258104


namespace NUMINAMATH_CALUDE_min_m_value_l2581_258176

theorem min_m_value (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x + 2*y + 2/x + 1/y = 6) : 
  ∃ (m : ℝ), (∀ (x y : ℝ), x > 0 → y > 0 → x + 2*y + 2/x + 1/y = 6 → m ≥ x + 2*y) ∧ 
  (∀ (m' : ℝ), (∀ (x y : ℝ), x > 0 → y > 0 → x + 2*y + 2/x + 1/y = 6 → m' ≥ x + 2*y) → m' ≥ m) ∧
  m = 4 :=
sorry

end NUMINAMATH_CALUDE_min_m_value_l2581_258176


namespace NUMINAMATH_CALUDE_gain_percent_calculation_l2581_258123

theorem gain_percent_calculation (cost_price selling_price : ℝ) 
  (h1 : cost_price = 100)
  (h2 : selling_price = 115) :
  (selling_price - cost_price) / cost_price * 100 = 15 := by
sorry

end NUMINAMATH_CALUDE_gain_percent_calculation_l2581_258123


namespace NUMINAMATH_CALUDE_solve_for_y_l2581_258120

theorem solve_for_y (x y : ℝ) (hx : x = 99) (heq : x^3*y - 2*x^2*y + x*y = 970200) : y = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l2581_258120


namespace NUMINAMATH_CALUDE_quarters_in_jar_l2581_258196

def pennies : ℕ := 123
def nickels : ℕ := 85
def dimes : ℕ := 35
def half_dollars : ℕ := 15
def dollar_coins : ℕ := 5
def family_members : ℕ := 8
def ice_cream_cost : ℚ := 4.5
def leftover : ℚ := 0.97

def total_other_coins : ℚ := 
  pennies * 0.01 + nickels * 0.05 + dimes * 0.1 + half_dollars * 0.5 + dollar_coins * 1.0

theorem quarters_in_jar : 
  ∃ (quarters : ℕ), 
    (quarters : ℚ) * 0.25 + total_other_coins = 
      family_members * ice_cream_cost + leftover ∧ 
    quarters = 140 := by sorry

end NUMINAMATH_CALUDE_quarters_in_jar_l2581_258196


namespace NUMINAMATH_CALUDE_sum_of_segment_lengths_divisible_by_four_l2581_258189

/-- Represents a square sheet of graph paper -/
structure GraphPaper where
  sideLength : ℕ

/-- The sum of lengths of all segments in the graph paper -/
def sumOfSegmentLengths (paper : GraphPaper) : ℕ :=
  2 * paper.sideLength * (paper.sideLength + 1)

/-- Theorem stating that the sum of segment lengths is divisible by 4 -/
theorem sum_of_segment_lengths_divisible_by_four (paper : GraphPaper) :
  4 ∣ sumOfSegmentLengths paper := by
  sorry

end NUMINAMATH_CALUDE_sum_of_segment_lengths_divisible_by_four_l2581_258189


namespace NUMINAMATH_CALUDE_function_identity_l2581_258188

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
variable (h_continuous : Continuous f)
variable (h_inequality : ∀ (a b c : ℝ) (x : ℝ), f (a * x^2 + b * x + c) ≥ a * (f x)^2 + b * (f x) + c)

-- Theorem statement
theorem function_identity : f = id := by
  sorry

end NUMINAMATH_CALUDE_function_identity_l2581_258188


namespace NUMINAMATH_CALUDE_companion_pair_expression_l2581_258171

/-- Definition of a companion pair -/
def is_companion_pair (m n : ℝ) : Prop :=
  m / 2 + n / 3 = (m + n) / 5

/-- Theorem: For any companion pair (m, n), the expression 
    m - (22/3)n - [4m - 2(3n - 1)] equals -2 -/
theorem companion_pair_expression (m n : ℝ) 
  (h : is_companion_pair m n) : 
  m - (22/3) * n - (4 * m - 2 * (3 * n - 1)) = -2 := by
  sorry

end NUMINAMATH_CALUDE_companion_pair_expression_l2581_258171


namespace NUMINAMATH_CALUDE_division_problem_l2581_258173

theorem division_problem (a b c d : ℚ) 
  (h1 : a / b = 3)
  (h2 : b / c = 5/3)
  (h3 : c / d = 2) :
  d / a = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l2581_258173


namespace NUMINAMATH_CALUDE_circle_path_in_right_triangle_l2581_258158

theorem circle_path_in_right_triangle : 
  ∀ (a b c : ℝ) (r : ℝ),
    a = 6 ∧ b = 8 ∧ c = 10 →  -- Triangle side lengths
    r = 1 →                   -- Circle radius
    a^2 + b^2 = c^2 →         -- Right triangle condition
    (a + b + c) - 6*r = 12 := by  -- Path length
  sorry

end NUMINAMATH_CALUDE_circle_path_in_right_triangle_l2581_258158


namespace NUMINAMATH_CALUDE_candy_distribution_l2581_258114

/-- Given the number of candies for each type and the number of cousins,
    calculates the number of candies left after equal distribution. -/
def candies_left (apple orange lemon grape cousins : ℕ) : ℕ :=
  (apple + orange + lemon + grape) % cousins

theorem candy_distribution (apple orange lemon grape cousins : ℕ) 
    (h : cousins > 0) : 
  candies_left apple orange lemon grape cousins = 
  (apple + orange + lemon + grape) % cousins := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l2581_258114


namespace NUMINAMATH_CALUDE_matthews_cakes_equal_crackers_l2581_258132

/-- The number of friends Matthew gave crackers and cakes to -/
def num_friends : ℕ := 4

/-- The number of crackers Matthew had initially -/
def initial_crackers : ℕ := 32

/-- The number of crackers each person ate -/
def crackers_per_person : ℕ := 8

/-- The number of cakes Matthew had initially -/
def initial_cakes : ℕ := initial_crackers

theorem matthews_cakes_equal_crackers :
  initial_cakes = initial_crackers :=
by sorry

end NUMINAMATH_CALUDE_matthews_cakes_equal_crackers_l2581_258132


namespace NUMINAMATH_CALUDE_solution_set_implies_a_bound_l2581_258169

theorem solution_set_implies_a_bound (a : ℝ) :
  (∀ x : ℝ, |2 - x| + |x + 1| ≥ a) → a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_implies_a_bound_l2581_258169


namespace NUMINAMATH_CALUDE_proportion_solution_l2581_258152

theorem proportion_solution (y : ℝ) : y / 1.35 = 5 / 9 → y = 0.75 := by
  sorry

end NUMINAMATH_CALUDE_proportion_solution_l2581_258152


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l2581_258148

theorem complex_magnitude_problem (z w : ℂ) 
  (h1 : Complex.abs (2 * z - w) = 29)
  (h2 : Complex.abs (z + 2 * w) = 7)
  (h3 : Complex.abs (z + w) = 3) :
  Complex.abs z = 11 := by sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l2581_258148


namespace NUMINAMATH_CALUDE_machining_defect_probability_l2581_258162

theorem machining_defect_probability (defect_rate1 defect_rate2 : ℝ) 
  (h1 : defect_rate1 = 0.03) 
  (h2 : defect_rate2 = 0.05) 
  (h3 : 0 ≤ defect_rate1 ∧ defect_rate1 ≤ 1) 
  (h4 : 0 ≤ defect_rate2 ∧ defect_rate2 ≤ 1) :
  1 - (1 - defect_rate1) * (1 - defect_rate2) = 0.0785 := by
  sorry

#check machining_defect_probability

end NUMINAMATH_CALUDE_machining_defect_probability_l2581_258162


namespace NUMINAMATH_CALUDE_middle_card_is_six_l2581_258193

def is_valid_set (a b c : ℕ) : Prop :=
  a < b ∧ b < c ∧ a + b + c = 20 ∧ a % 2 = 0 ∧ c % 2 = 0

def possible_after_aria (a b c : ℕ) : Prop :=
  is_valid_set a b c ∧ a ≠ 6

def possible_after_cece (a b c : ℕ) : Prop :=
  possible_after_aria a b c ∧ c ≠ 13

def possible_after_bruce (a b c : ℕ) : Prop :=
  possible_after_cece a b c ∧ (b ≠ 5 ∨ a ≠ 4)

theorem middle_card_is_six :
  ∀ a b c : ℕ, possible_after_bruce a b c → b = 6 :=
sorry

end NUMINAMATH_CALUDE_middle_card_is_six_l2581_258193


namespace NUMINAMATH_CALUDE_symmetry_axis_implies_p_plus_s_zero_l2581_258128

/-- Given a curve y = (px + q)/(rx + s) with y = 2x as its axis of symmetry,
    where p, q, r, s are nonzero real numbers, prove that p + s = 0. -/
theorem symmetry_axis_implies_p_plus_s_zero
  (p q r s : ℝ)
  (hp : p ≠ 0)
  (hq : q ≠ 0)
  (hr : r ≠ 0)
  (hs : s ≠ 0)
  (h_symmetry : ∀ (x y : ℝ), y = (p * x + q) / (r * x + s) → 2 * x = (p * (2 * y) + q) / (r * (2 * y) + s)) :
  p + s = 0 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_axis_implies_p_plus_s_zero_l2581_258128


namespace NUMINAMATH_CALUDE_rectangle_intersection_sum_l2581_258124

/-- A circle in the xy-plane -/
structure Circle where
  center : ℝ × ℝ
  equation : ℝ → ℝ → ℝ → Prop

/-- A line in the xy-plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Predicate to check if four points form a rectangle -/
def form_rectangle (p1 p2 p3 p4 : ℝ × ℝ) : Prop := sorry

/-- Predicate to check if a point lies on a circle -/
def on_circle (p : ℝ × ℝ) (c : Circle) : Prop := sorry

/-- Predicate to check if a point lies on a line -/
def on_line (p : ℝ × ℝ) (l : Line) : Prop := sorry

theorem rectangle_intersection_sum (m n k : ℝ) :
  let c : Circle := ⟨(-3/2, -1/2), fun x y k => x^2 + y^2 + 3*x + y + k = 0⟩
  let l₁ : Line := ⟨3, m⟩
  let l₂ : Line := ⟨3, n⟩
  (∃ p1 p2 p3 p4 : ℝ × ℝ,
    on_circle p1 c ∧ on_circle p2 c ∧ on_circle p3 c ∧ on_circle p4 c ∧
    ((on_line p1 l₁ ∧ on_line p2 l₁) ∨ (on_line p1 l₁ ∧ on_line p3 l₁) ∨
     (on_line p1 l₁ ∧ on_line p4 l₁) ∨ (on_line p2 l₁ ∧ on_line p3 l₁) ∨
     (on_line p2 l₁ ∧ on_line p4 l₁) ∨ (on_line p3 l₁ ∧ on_line p4 l₁)) ∧
    ((on_line p1 l₂ ∧ on_line p2 l₂) ∨ (on_line p1 l₂ ∧ on_line p3 l₂) ∨
     (on_line p1 l₂ ∧ on_line p4 l₂) ∨ (on_line p2 l₂ ∧ on_line p3 l₂) ∨
     (on_line p2 l₂ ∧ on_line p4 l₂) ∨ (on_line p3 l₂ ∧ on_line p4 l₂)) ∧
    form_rectangle p1 p2 p3 p4) →
  m + n = 8 := by sorry

end NUMINAMATH_CALUDE_rectangle_intersection_sum_l2581_258124


namespace NUMINAMATH_CALUDE_second_invoice_not_23_l2581_258191

def systematic_sampling (first_invoice : ℕ) : ℕ → ℕ
  | 0 => first_invoice
  | n + 1 => systematic_sampling first_invoice n + 10

theorem second_invoice_not_23 :
  ∀ (first_invoice : ℕ), 1 ≤ first_invoice ∧ first_invoice ≤ 10 →
    systematic_sampling first_invoice 1 ≠ 23 := by
  sorry

end NUMINAMATH_CALUDE_second_invoice_not_23_l2581_258191


namespace NUMINAMATH_CALUDE_mean_equality_implies_z_l2581_258185

theorem mean_equality_implies_z (z : ℚ) : 
  (5 + 10 + 20) / 3 = (15 + z) / 2 → z = 25 / 3 :=
by sorry

end NUMINAMATH_CALUDE_mean_equality_implies_z_l2581_258185


namespace NUMINAMATH_CALUDE_line_through_point_l2581_258137

/-- Proves that the value of k is -10 for a line passing through (-1/3, -2) --/
theorem line_through_point (k : ℝ) : 
  (2 - 3 * k * (-1/3) = 4 * (-2)) → k = -10 := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_l2581_258137


namespace NUMINAMATH_CALUDE_fourth_month_sale_l2581_258163

def sale_month1 : ℕ := 5420
def sale_month2 : ℕ := 5660
def sale_month3 : ℕ := 6200
def sale_month5 : ℕ := 6500
def sale_month6 : ℕ := 6470
def average_sale : ℕ := 6100
def num_months : ℕ := 6

theorem fourth_month_sale :
  sale_month1 + sale_month2 + sale_month3 + sale_month5 + sale_month6 + 6350 = average_sale * num_months :=
by sorry

end NUMINAMATH_CALUDE_fourth_month_sale_l2581_258163


namespace NUMINAMATH_CALUDE_paper_airplane_class_composition_l2581_258155

theorem paper_airplane_class_composition 
  (total_students : ℕ) 
  (total_airplanes : ℕ) 
  (girls_airplanes : ℕ) 
  (boys_airplanes : ℕ) 
  (h1 : total_students = 21)
  (h2 : total_airplanes = 69)
  (h3 : girls_airplanes = 2)
  (h4 : boys_airplanes = 5) :
  ∃ (boys girls : ℕ), 
    boys + girls = total_students ∧ 
    boys * boys_airplanes + girls * girls_airplanes = total_airplanes ∧
    boys = 9 ∧ 
    girls = 12 := by
  sorry

end NUMINAMATH_CALUDE_paper_airplane_class_composition_l2581_258155


namespace NUMINAMATH_CALUDE_solution_set_when_m_3_range_of_m_for_nonnegative_f_l2581_258110

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - m*x + m - 1

-- Statement for Question 1
theorem solution_set_when_m_3 :
  {x : ℝ | f 3 x ≤ 0} = {x : ℝ | 1 ≤ x ∧ x ≤ 2} := by sorry

-- Statement for Question 2
theorem range_of_m_for_nonnegative_f :
  ∀ m : ℝ, (∀ x ∈ Set.Icc 2 4, f m x ≥ -1) ↔ m ≤ 4 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_m_3_range_of_m_for_nonnegative_f_l2581_258110


namespace NUMINAMATH_CALUDE_last_islander_is_knight_l2581_258142

/-- Represents the type of an islander: either a knight or a liar -/
inductive IslanderType
  | Knight
  | Liar

/-- Represents what an islander says about their neighbor -/
inductive Statement
  | Knight
  | Liar

/-- The number of islanders around the table -/
def numIslanders : Nat := 50

/-- Function that determines what an islander at a given position says -/
def statement (position : Nat) : Statement :=
  if position % 2 == 1 then Statement.Knight else Statement.Liar

/-- Function that determines the actual type of an islander based on their statement and the type of their neighbor -/
def actualType (position : Nat) (neighborType : IslanderType) : IslanderType :=
  match (statement position, neighborType) with
  | (Statement.Knight, IslanderType.Knight) => IslanderType.Knight
  | (Statement.Knight, IslanderType.Liar) => IslanderType.Liar
  | (Statement.Liar, IslanderType.Knight) => IslanderType.Liar
  | (Statement.Liar, IslanderType.Liar) => IslanderType.Knight

theorem last_islander_is_knight : 
  ∀ (first : IslanderType), actualType numIslanders first = IslanderType.Knight :=
by sorry

end NUMINAMATH_CALUDE_last_islander_is_knight_l2581_258142


namespace NUMINAMATH_CALUDE_inequality_proof_l2581_258126

theorem inequality_proof (x y z : ℝ) : x^2 + y^2 + z^2 ≥ Real.sqrt 2 * (x*y + y*z) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2581_258126


namespace NUMINAMATH_CALUDE_savings_calculation_l2581_258118

/-- Calculates savings given income and income-to-expenditure ratio --/
def calculate_savings (income : ℕ) (income_ratio : ℕ) (expenditure_ratio : ℕ) : ℕ :=
  income - (income * expenditure_ratio) / income_ratio

/-- Theorem: Given the specified conditions, the savings is 2000 --/
theorem savings_calculation :
  let income := 18000
  let income_ratio := 9
  let expenditure_ratio := 8
  calculate_savings income income_ratio expenditure_ratio = 2000 := by
  sorry

#eval calculate_savings 18000 9 8

end NUMINAMATH_CALUDE_savings_calculation_l2581_258118


namespace NUMINAMATH_CALUDE_max_z_value_l2581_258139

theorem max_z_value (x y z : ℕ) : 
  7 < x → x < 9 → 9 < y → y < 15 → 
  0 < z → 
  Nat.Prime x → Nat.Prime y → Nat.Prime z →
  (y - x) % z = 0 →
  z ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_max_z_value_l2581_258139


namespace NUMINAMATH_CALUDE_group_size_problem_l2581_258178

theorem group_size_problem (total_paise : ℕ) (h1 : total_paise = 5776) : ∃ n : ℕ, n * n = total_paise ∧ n = 76 := by
  sorry

end NUMINAMATH_CALUDE_group_size_problem_l2581_258178


namespace NUMINAMATH_CALUDE_point_in_planar_region_l2581_258157

/-- A point (m, 1) is within the planar region represented by 2x + 3y - 5 > 0 if and only if m > 1 -/
theorem point_in_planar_region (m : ℝ) : 2*m + 3*1 - 5 > 0 ↔ m > 1 := by
  sorry

end NUMINAMATH_CALUDE_point_in_planar_region_l2581_258157


namespace NUMINAMATH_CALUDE_log_expression_equals_two_l2581_258105

-- Define the logarithm function (base 10)
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_expression_equals_two :
  2 * log10 2 + log10 5 / log10 (Real.sqrt 10) = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_expression_equals_two_l2581_258105


namespace NUMINAMATH_CALUDE_total_cost_is_four_dollars_l2581_258122

/-- The cost of a single tire in dollars -/
def cost_per_tire : ℝ := 0.50

/-- The number of tires -/
def number_of_tires : ℕ := 8

/-- The total cost of all tires -/
def total_cost : ℝ := cost_per_tire * number_of_tires

theorem total_cost_is_four_dollars : total_cost = 4 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_four_dollars_l2581_258122


namespace NUMINAMATH_CALUDE_apples_per_pie_l2581_258129

theorem apples_per_pie 
  (total_apples : ℕ) 
  (unripe_apples : ℕ) 
  (num_pies : ℕ) 
  (h1 : total_apples = 34) 
  (h2 : unripe_apples = 6) 
  (h3 : num_pies = 7) 
  (h4 : unripe_apples < total_apples) :
  (total_apples - unripe_apples) / num_pies = 4 := by
  sorry

end NUMINAMATH_CALUDE_apples_per_pie_l2581_258129


namespace NUMINAMATH_CALUDE_nth_equation_holds_l2581_258179

theorem nth_equation_holds (n : ℕ) : 
  (n : ℚ) / (n + 1) = (n + 3 * 2 * n) / (n + 1 + 3 * 2 * (n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_nth_equation_holds_l2581_258179


namespace NUMINAMATH_CALUDE_system_solution_1_l2581_258115

theorem system_solution_1 (x y : ℝ) : 
  2^(x + y) = x + 7 ∧ x + y = 3 → x = 1 ∧ y = 2 := by sorry

end NUMINAMATH_CALUDE_system_solution_1_l2581_258115


namespace NUMINAMATH_CALUDE_tan_theta_value_l2581_258133

theorem tan_theta_value (θ : ℝ) (z : ℂ) : 
  z = Complex.mk (Real.sin θ - 3/5) (Real.cos θ - 4/5) → 
  z.re = 0 → 
  z.im ≠ 0 → 
  Real.tan θ = -3/4 :=
sorry

end NUMINAMATH_CALUDE_tan_theta_value_l2581_258133


namespace NUMINAMATH_CALUDE_cube_cube_squared_power_calculation_l2581_258199

theorem cube_cube_squared (a b : ℕ) : (a^3 * b^3)^2 = (a * b)^6 := by sorry

theorem power_calculation : (3^3 * 4^3)^2 = 2985984 := by sorry

end NUMINAMATH_CALUDE_cube_cube_squared_power_calculation_l2581_258199


namespace NUMINAMATH_CALUDE_no_twin_prime_legs_in_right_triangle_l2581_258107

theorem no_twin_prime_legs_in_right_triangle :
  ∀ (p k : ℕ), 
    Prime p → 
    Prime (p + 2) → 
    (∃ (h : ℕ), h * h = p * p + (p + 2) * (p + 2)) → 
    False :=
by
  sorry

end NUMINAMATH_CALUDE_no_twin_prime_legs_in_right_triangle_l2581_258107


namespace NUMINAMATH_CALUDE_tricycle_wheels_l2581_258151

theorem tricycle_wheels (num_bicycles num_tricycles bicycle_wheels total_wheels : ℕ) 
  (h1 : num_bicycles = 16)
  (h2 : num_tricycles = 7)
  (h3 : bicycle_wheels = 2)
  (h4 : total_wheels = 53)
  : (total_wheels - num_bicycles * bicycle_wheels) / num_tricycles = 3 := by
  sorry

end NUMINAMATH_CALUDE_tricycle_wheels_l2581_258151


namespace NUMINAMATH_CALUDE_john_scores_42_points_l2581_258187

/-- Calculates the total points scored by John given the specified conditions -/
def total_points_scored (shots_per_interval : ℕ) (points_per_shot : ℕ) (three_point_shots : ℕ) 
                        (interval_duration : ℕ) (num_periods : ℕ) (period_duration : ℕ) : ℕ :=
  let total_time := num_periods * period_duration
  let num_intervals := total_time / interval_duration
  let points_per_interval := shots_per_interval * points_per_shot + three_point_shots * 3
  num_intervals * points_per_interval

/-- Theorem stating that John scores 42 points under the given conditions -/
theorem john_scores_42_points : 
  total_points_scored 2 2 1 4 2 12 = 42 := by
  sorry


end NUMINAMATH_CALUDE_john_scores_42_points_l2581_258187


namespace NUMINAMATH_CALUDE_abs_negative_2022_l2581_258113

theorem abs_negative_2022 : |(-2022 : ℤ)| = 2022 := by
  sorry

end NUMINAMATH_CALUDE_abs_negative_2022_l2581_258113


namespace NUMINAMATH_CALUDE_tims_sleep_schedule_l2581_258138

/-- Tim's sleep schedule and total sleep calculation -/
theorem tims_sleep_schedule (weekday_sleep : ℕ) (weekend_sleep : ℕ) (weekdays : ℕ) (weekend_days : ℕ) :
  weekday_sleep = 6 →
  weekend_sleep = 10 →
  weekdays = 5 →
  weekend_days = 2 →
  weekday_sleep * weekdays + weekend_sleep * weekend_days = 50 := by
  sorry

#check tims_sleep_schedule

end NUMINAMATH_CALUDE_tims_sleep_schedule_l2581_258138


namespace NUMINAMATH_CALUDE_remainder_5_pow_2023_mod_6_l2581_258153

theorem remainder_5_pow_2023_mod_6 : 5^2023 % 6 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_5_pow_2023_mod_6_l2581_258153


namespace NUMINAMATH_CALUDE_jack_money_per_can_l2581_258135

def bottles_recycled : ℕ := 80
def cans_recycled : ℕ := 140
def total_money : ℚ := 15
def money_per_bottle : ℚ := 1/10

theorem jack_money_per_can :
  (total_money - (bottles_recycled : ℚ) * money_per_bottle) / (cans_recycled : ℚ) = 5/100 := by
  sorry

end NUMINAMATH_CALUDE_jack_money_per_can_l2581_258135


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l2581_258111

/-- The x-coordinate of the end point of the line segment -/
def x : ℝ := 3.4213

/-- The y-coordinate of the end point of the line segment -/
def y : ℝ := 7.8426

/-- The start point of the line segment -/
def start_point : ℝ × ℝ := (2, 2)

/-- The length of the line segment -/
def segment_length : ℝ := 6

theorem line_segment_endpoint :
  x > 0 ∧
  y = 2 * x + 1 ∧
  Real.sqrt ((x - 2)^2 + (y - 2)^2) = segment_length :=
by sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l2581_258111


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l2581_258154

theorem complex_number_quadrant (a b : ℝ) (h : (1 : ℂ) + a * I = (b + I) * (1 + I)) :
  a > 0 ∧ b > 0 :=
by sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l2581_258154


namespace NUMINAMATH_CALUDE_right_triangle_trig_l2581_258141

theorem right_triangle_trig (P Q R : ℝ × ℝ) : 
  let pq := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
  let pr := Real.sqrt ((P.1 - R.1)^2 + (P.2 - R.2)^2)
  let qr := Real.sqrt ((Q.1 - R.1)^2 + (Q.2 - R.2)^2)
  (∀ S, S ≠ R → (Real.sqrt ((R.1 - S.1)^2 + (R.2 - S.2)^2))^2 = (S.1 - R.1)^2 + (S.2 - R.2)^2) →
  pq = 15 →
  pr = 9 →
  qr^2 + pr^2 = pq^2 →
  (qr / pq) = 4/5 ∧ (pr / pq) = 3/5 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_trig_l2581_258141


namespace NUMINAMATH_CALUDE_frozen_yoghurt_cartons_l2581_258134

/-- Represents the number of cartons of ice cream Caleb bought -/
def ice_cream_cartons : ℕ := 10

/-- Represents the cost of one carton of ice cream in dollars -/
def ice_cream_cost : ℕ := 4

/-- Represents the cost of one carton of frozen yoghurt in dollars -/
def yoghurt_cost : ℕ := 1

/-- Represents the difference in dollars between ice cream and frozen yoghurt spending -/
def spending_difference : ℕ := 36

/-- Theorem stating that the number of frozen yoghurt cartons Caleb bought is 4 -/
theorem frozen_yoghurt_cartons : ℕ := by
  sorry

end NUMINAMATH_CALUDE_frozen_yoghurt_cartons_l2581_258134


namespace NUMINAMATH_CALUDE_weight_of_replaced_person_l2581_258147

theorem weight_of_replaced_person 
  (n : ℕ) 
  (avg_increase : ℝ) 
  (new_person_weight : ℝ) 
  (h1 : n = 8)
  (h2 : avg_increase = 2.5)
  (h3 : new_person_weight = 60) :
  ∃ (replaced_weight : ℝ), replaced_weight = new_person_weight - n * avg_increase :=
by sorry

end NUMINAMATH_CALUDE_weight_of_replaced_person_l2581_258147


namespace NUMINAMATH_CALUDE_function_increasing_iff_m_in_range_l2581_258167

/-- The function f(x) = (1/3)x³ - mx² - 3m²x + 1 is increasing on (1, 2) if and only if m is in [-1, 1/3] -/
theorem function_increasing_iff_m_in_range (m : ℝ) :
  (∀ x ∈ Set.Ioo 1 2, StrictMono (fun x => (1/3) * x^3 - m * x^2 - 3 * m^2 * x + 1)) ↔
  m ∈ Set.Icc (-1) (1/3) :=
sorry

end NUMINAMATH_CALUDE_function_increasing_iff_m_in_range_l2581_258167
