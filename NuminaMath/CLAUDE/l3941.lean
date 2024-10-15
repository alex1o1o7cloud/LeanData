import Mathlib

namespace NUMINAMATH_CALUDE_sum_f_mod_1000_l3941_394115

-- Define the function f
def f (n : ℕ) : ℕ := 
  (Finset.filter (fun d => d < n ∨ Nat.gcd d n ≠ 1) (Nat.divisors (2024^2024))).card

-- State the theorem
theorem sum_f_mod_1000 : 
  (Finset.sum (Finset.range (2024^2024 + 1)) f) % 1000 = 224 := by sorry

end NUMINAMATH_CALUDE_sum_f_mod_1000_l3941_394115


namespace NUMINAMATH_CALUDE_card_game_combinations_l3941_394161

theorem card_game_combinations : Nat.choose 52 13 = 635013587600 := by
  sorry

end NUMINAMATH_CALUDE_card_game_combinations_l3941_394161


namespace NUMINAMATH_CALUDE_total_games_cost_is_13800_l3941_394109

/-- Calculates the total cost of games owned by Katie and her friends -/
def totalGamesCost (katieGames : ℕ) (newFriends oldFriends : ℕ) (newFriendGames oldFriendGames : ℕ) (costPerGame : ℕ) : ℕ :=
  let totalGames := katieGames + newFriends * newFriendGames + oldFriends * oldFriendGames
  totalGames * costPerGame

/-- Theorem stating that the total cost of games is $13,800 -/
theorem total_games_cost_is_13800 :
  totalGamesCost 91 5 3 88 53 20 = 13800 := by
  sorry

#eval totalGamesCost 91 5 3 88 53 20

end NUMINAMATH_CALUDE_total_games_cost_is_13800_l3941_394109


namespace NUMINAMATH_CALUDE_f_2x_l3941_394164

/-- Given a function f(x) = x^2 - 1, prove that f(2x) = 4x^2 - 1 --/
theorem f_2x (x : ℝ) : (fun x => x^2 - 1) (2*x) = 4*x^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_f_2x_l3941_394164


namespace NUMINAMATH_CALUDE_even_quadratic_implies_m_eq_two_l3941_394138

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- The quadratic function f(x) = x^2 + (m-2)x + 1 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + (m-2)*x + 1

theorem even_quadratic_implies_m_eq_two (m : ℝ) (h : IsEven (f m)) : m = 2 := by
  sorry

end NUMINAMATH_CALUDE_even_quadratic_implies_m_eq_two_l3941_394138


namespace NUMINAMATH_CALUDE_nicky_trade_profit_l3941_394148

/-- Calculates Nicky's profit or loss in a baseball card trade with Jill --/
theorem nicky_trade_profit :
  let nicky_card1_value : ℚ := 8
  let nicky_card1_count : ℕ := 2
  let nicky_card2_value : ℚ := 5
  let nicky_card2_count : ℕ := 3
  let jill_card1_value_cad : ℚ := 21
  let jill_card1_count : ℕ := 1
  let jill_card2_value_cad : ℚ := 6
  let jill_card2_count : ℕ := 2
  let exchange_rate_usd_per_cad : ℚ := 0.8
  let tax_rate : ℚ := 0.05

  let nicky_total_value := nicky_card1_value * nicky_card1_count + nicky_card2_value * nicky_card2_count
  let jill_total_value_cad := jill_card1_value_cad * jill_card1_count + jill_card2_value_cad * jill_card2_count
  let jill_total_value_usd := jill_total_value_cad * exchange_rate_usd_per_cad
  let total_trade_value_usd := nicky_total_value + jill_total_value_usd
  let tax_amount := total_trade_value_usd * tax_rate
  let nicky_profit := jill_total_value_usd - (nicky_total_value + tax_amount)

  nicky_profit = -7.47 := by sorry

end NUMINAMATH_CALUDE_nicky_trade_profit_l3941_394148


namespace NUMINAMATH_CALUDE_fraction_equality_l3941_394198

theorem fraction_equality (x y z w : ℝ) 
  (h : (x - y) * (z - w) / ((y - z) * (w - x)) = 3/7) :
  (x - z) * (y - w) / ((x - y) * (z - w)) = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3941_394198


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l3941_394117

theorem arithmetic_expression_equality : 70 + 5 * 12 / (180 / 3) = 71 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l3941_394117


namespace NUMINAMATH_CALUDE_purely_imaginary_value_l3941_394134

-- Define a complex number z as a function of real number m
def z (m : ℝ) : ℂ := m + 2 + (m - 1) * Complex.I

-- State the theorem
theorem purely_imaginary_value (m : ℝ) : 
  (z m).re = 0 ∧ (z m).im ≠ 0 → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_value_l3941_394134


namespace NUMINAMATH_CALUDE_arc_length_proof_l3941_394197

open Real

noncomputable def curve (x : ℝ) : ℝ := Real.log (5 / (2 * x))

theorem arc_length_proof (a b : ℝ) (ha : a = Real.sqrt 3) (hb : b = Real.sqrt 8) :
  ∫ x in a..b, sqrt (1 + (deriv curve x) ^ 2) = 1 + (1 / 2) * log (3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_arc_length_proof_l3941_394197


namespace NUMINAMATH_CALUDE_existence_of_n_good_not_n_plus_1_good_l3941_394186

def sum_of_digits (k : ℕ+) : ℕ := sorry

def is_n_good (a n : ℕ+) : Prop :=
  ∃ (seq : Fin (n + 1) → ℕ+),
    seq (Fin.last n) = a ∧
    ∀ i : Fin n, seq i.succ = seq i - sum_of_digits (seq i)

theorem existence_of_n_good_not_n_plus_1_good :
  ∀ n : ℕ+, ∃ b : ℕ+, is_n_good b n ∧ ¬is_n_good b (n + 1) :=
sorry

end NUMINAMATH_CALUDE_existence_of_n_good_not_n_plus_1_good_l3941_394186


namespace NUMINAMATH_CALUDE_money_left_over_correct_l3941_394171

/-- Calculates the money left over after purchases given the specified conditions --/
def money_left_over (
  video_game_cost : ℚ)
  (video_game_discount : ℚ)
  (candy_cost : ℚ)
  (sales_tax : ℚ)
  (shipping_fee : ℚ)
  (babysitting_rate : ℚ)
  (bonus_rate : ℚ)
  (hours_worked : ℕ)
  (bonus_threshold : ℕ) : ℚ :=
  let discounted_game_cost := video_game_cost * (1 - video_game_discount)
  let total_before_tax := discounted_game_cost + shipping_fee + candy_cost
  let total_cost := total_before_tax * (1 + sales_tax)
  let regular_hours := min hours_worked bonus_threshold
  let bonus_hours := hours_worked - regular_hours
  let total_earnings := babysitting_rate * hours_worked + bonus_rate * bonus_hours
  total_earnings - total_cost

theorem money_left_over_correct :
  money_left_over 60 0.15 5 0.10 3 8 2 9 5 = 151/10 := by
  sorry

end NUMINAMATH_CALUDE_money_left_over_correct_l3941_394171


namespace NUMINAMATH_CALUDE_square_area_from_vertices_l3941_394137

/-- The area of a square with adjacent vertices at (1, 3) and (5, -1) is 32 -/
theorem square_area_from_vertices : 
  let p1 : ℝ × ℝ := (1, 3)
  let p2 : ℝ × ℝ := (5, -1)
  let side_length := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  side_length^2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_vertices_l3941_394137


namespace NUMINAMATH_CALUDE_basketball_score_problem_l3941_394182

theorem basketball_score_problem (total_points winning_margin : ℕ) 
  (h1 : total_points = 48) 
  (h2 : winning_margin = 18) : 
  ∃ (sharks_score dolphins_score : ℕ), 
    sharks_score + dolphins_score = total_points ∧ 
    sharks_score - dolphins_score = winning_margin ∧ 
    dolphins_score = 15 := by
sorry

end NUMINAMATH_CALUDE_basketball_score_problem_l3941_394182


namespace NUMINAMATH_CALUDE_air_quality_probability_l3941_394106

theorem air_quality_probability (p_good : ℝ) (p_consecutive : ℝ) :
  p_good = 0.8 →
  p_consecutive = 0.6 →
  p_good * (p_consecutive / p_good) = 0.75 :=
by sorry

end NUMINAMATH_CALUDE_air_quality_probability_l3941_394106


namespace NUMINAMATH_CALUDE_constant_remainder_iff_a_eq_neg_seven_l3941_394156

/-- The dividend polynomial -/
def dividend (a : ℝ) (x : ℝ) : ℝ := 10 * x^3 - 7 * x^2 + a * x + 6

/-- The divisor polynomial -/
def divisor (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 1

/-- The remainder of the polynomial division -/
def remainder (a : ℝ) (x : ℝ) : ℝ := (a + 7) * x + 2

theorem constant_remainder_iff_a_eq_neg_seven :
  ∀ a : ℝ, (∀ x : ℝ, ∃ q : ℝ, dividend a x = divisor x * q + remainder a x) ↔ a = -7 := by
  sorry

end NUMINAMATH_CALUDE_constant_remainder_iff_a_eq_neg_seven_l3941_394156


namespace NUMINAMATH_CALUDE_negative_six_times_negative_one_l3941_394104

theorem negative_six_times_negative_one : (-6 : ℤ) * (-1 : ℤ) = 6 := by sorry

end NUMINAMATH_CALUDE_negative_six_times_negative_one_l3941_394104


namespace NUMINAMATH_CALUDE_average_age_of_students_l3941_394193

theorem average_age_of_students (num_students : ℕ) (teacher_age : ℕ) (total_average : ℕ) 
  (h1 : num_students = 40)
  (h2 : teacher_age = 56)
  (h3 : total_average = 16) :
  (num_students * total_average - teacher_age) / num_students = 15 := by
  sorry

end NUMINAMATH_CALUDE_average_age_of_students_l3941_394193


namespace NUMINAMATH_CALUDE_parametric_to_cartesian_l3941_394124

theorem parametric_to_cartesian :
  ∀ x y θ : ℝ,
  x = Real.sin θ →
  y = Real.cos (2 * θ) →
  -1 ≤ x ∧ x ≤ 1 →
  y = 1 - 2 * x^2 :=
by
  sorry

end NUMINAMATH_CALUDE_parametric_to_cartesian_l3941_394124


namespace NUMINAMATH_CALUDE_fish_problem_solution_l3941_394111

/-- Calculates the number of fish added on day 7 given the initial conditions and daily changes --/
def fish_added_day_7 (initial : ℕ) (double : ℕ → ℕ) (remove_third : ℕ → ℕ) (remove_fourth : ℕ → ℕ) (final : ℕ) : ℕ :=
  let day1 := initial
  let day2 := double day1
  let day3 := remove_third (double day2)
  let day4 := double day3
  let day5 := remove_fourth (double day4)
  let day6 := double day5
  let day7_before_adding := double day6
  final - day7_before_adding

theorem fish_problem_solution :
  fish_added_day_7 6 (λ x => 2 * x) (λ x => x - x / 3) (λ x => x - x / 4) 207 = 15 := by
  sorry

#eval fish_added_day_7 6 (λ x => 2 * x) (λ x => x - x / 3) (λ x => x - x / 4) 207

end NUMINAMATH_CALUDE_fish_problem_solution_l3941_394111


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l3941_394142

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 267) 
  (h2 : a*b + b*c + c*a = 131) : 
  a + b + c = 23 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l3941_394142


namespace NUMINAMATH_CALUDE_line_point_x_value_l3941_394170

/-- Given a line passing through points (x, -4) and (4, 1) with a slope of 1, prove that x = -1 -/
theorem line_point_x_value (x : ℝ) : 
  let p1 : ℝ × ℝ := (x, -4)
  let p2 : ℝ × ℝ := (4, 1)
  let slope : ℝ := (p2.2 - p1.2) / (p2.1 - p1.1)
  slope = 1 → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_line_point_x_value_l3941_394170


namespace NUMINAMATH_CALUDE_age_sum_proof_l3941_394151

/-- Given the age relationship between Michael and Emily, prove that the sum of their current ages is 32. -/
theorem age_sum_proof (M E : ℚ) : 
  M = E + 9 ∧ 
  M + 5 = 3 * (E - 3) → 
  M + E = 32 := by
  sorry

end NUMINAMATH_CALUDE_age_sum_proof_l3941_394151


namespace NUMINAMATH_CALUDE_keith_digimon_packs_l3941_394114

/-- The cost of one pack of Digimon cards in dollars -/
def digimon_pack_cost : ℚ := 445/100

/-- The cost of a deck of baseball cards in dollars -/
def baseball_deck_cost : ℚ := 606/100

/-- The total amount Keith spent on cards in dollars -/
def total_spent : ℚ := 2386/100

/-- The number of Digimon card packs Keith bought -/
def digimon_packs : ℕ := 4

theorem keith_digimon_packs :
  digimon_packs * digimon_pack_cost + baseball_deck_cost = total_spent :=
sorry

end NUMINAMATH_CALUDE_keith_digimon_packs_l3941_394114


namespace NUMINAMATH_CALUDE_horseshoe_profit_is_5000_l3941_394132

/-- Calculates the profit for a horseshoe manufacturing company --/
def horseshoe_profit (initial_outlay : ℕ) (cost_per_set : ℕ) (price_per_set : ℕ) (num_sets : ℕ) : ℤ :=
  (price_per_set * num_sets : ℤ) - (initial_outlay + cost_per_set * num_sets : ℤ)

/-- Proves that the profit for the given conditions is $5,000 --/
theorem horseshoe_profit_is_5000 :
  horseshoe_profit 10000 20 50 500 = 5000 := by
  sorry

#eval horseshoe_profit 10000 20 50 500

end NUMINAMATH_CALUDE_horseshoe_profit_is_5000_l3941_394132


namespace NUMINAMATH_CALUDE_male_red_ants_percentage_l3941_394146

/-- Represents the percentage of red ants in the total population -/
def red_percentage : ℝ := 0.85

/-- Represents the percentage of female ants among red ants -/
def female_red_percentage : ℝ := 0.45

/-- Calculates the percentage of male red ants in the total population -/
def male_red_percentage : ℝ := red_percentage * (1 - female_red_percentage)

/-- Theorem stating that the percentage of male red ants in the total population is 46.75% -/
theorem male_red_ants_percentage : 
  male_red_percentage = 0.4675 := by sorry

end NUMINAMATH_CALUDE_male_red_ants_percentage_l3941_394146


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3941_394101

/-- Given an arithmetic sequence with first term 3 and last term 27,
    the sum of the two terms immediately preceding 27 is 42. -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) : 
  (∃ d : ℝ, ∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence
  a 0 = 3 →  -- first term is 3
  (∃ k : ℕ, a k = 27 ∧ ∀ n > k, a n ≠ 27) →  -- 27 is the last term
  (∃ m : ℕ, a (m - 1) + a m = 42 ∧ a (m + 1) = 27) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3941_394101


namespace NUMINAMATH_CALUDE_no_real_solution_system_l3941_394140

theorem no_real_solution_system :
  ¬∃ (x y z : ℝ), (x + y + 2 + 4*x*y = 0) ∧ 
                  (y + z + 2 + 4*y*z = 0) ∧ 
                  (z + x + 2 + 4*z*x = 0) := by
  sorry

end NUMINAMATH_CALUDE_no_real_solution_system_l3941_394140


namespace NUMINAMATH_CALUDE_cubic_sequence_with_two_squares_exists_l3941_394172

/-- A cubic sequence with integer coefficients -/
def cubic_sequence (b c d : ℤ) (n : ℤ) : ℤ :=
  n^3 + b*n^2 + c*n + d

/-- Predicate for perfect squares -/
def is_perfect_square (x : ℤ) : Prop :=
  ∃ k : ℤ, x = k^2

theorem cubic_sequence_with_two_squares_exists :
  ∃ (b c d : ℤ),
    (is_perfect_square (cubic_sequence b c d 2015)) ∧
    (is_perfect_square (cubic_sequence b c d 2016)) ∧
    (∀ n : ℤ, n ≠ 2015 → n ≠ 2016 → ¬(is_perfect_square (cubic_sequence b c d n))) ∧
    (cubic_sequence b c d 2015 * cubic_sequence b c d 2016 = 0) :=
sorry

end NUMINAMATH_CALUDE_cubic_sequence_with_two_squares_exists_l3941_394172


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l3941_394121

theorem contrapositive_equivalence (p q : Prop) :
  (p → ¬q) ↔ (q → ¬p) := by
  sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l3941_394121


namespace NUMINAMATH_CALUDE_fair_ride_cost_l3941_394139

theorem fair_ride_cost (total_tickets : ℕ) (booth_tickets : ℕ) (num_rides : ℕ) 
  (h1 : total_tickets = 79) 
  (h2 : booth_tickets = 23) 
  (h3 : num_rides = 8) : 
  (total_tickets - booth_tickets) / num_rides = 7 := by
  sorry

end NUMINAMATH_CALUDE_fair_ride_cost_l3941_394139


namespace NUMINAMATH_CALUDE_divisibility_implies_equality_l3941_394159

theorem divisibility_implies_equality (a b : ℕ) 
  (h : (a^2 + a*b + 1) % (b^2 + b*a + 1) = 0) : a = b := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implies_equality_l3941_394159


namespace NUMINAMATH_CALUDE_y₁_gt_y₂_l3941_394131

/-- A linear function y = -2x + 3 --/
def f (x : ℝ) : ℝ := -2 * x + 3

/-- Point P₁ on the graph of f --/
def P₁ : ℝ × ℝ := (-2, f (-2))

/-- Point P₂ on the graph of f --/
def P₂ : ℝ × ℝ := (3, f 3)

/-- The y-coordinate of P₁ --/
def y₁ : ℝ := P₁.2

/-- The y-coordinate of P₂ --/
def y₂ : ℝ := P₂.2

theorem y₁_gt_y₂ : y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_y₁_gt_y₂_l3941_394131


namespace NUMINAMATH_CALUDE_correct_articles_l3941_394133

/-- Represents an article in English --/
inductive Article
| None
| A
| The

/-- Represents a sentence with two blanks for articles --/
structure Sentence :=
  (first_blank : Article)
  (second_blank : Article)

/-- Checks if a given sentence has the correct articles --/
def is_correct_sentence (s : Sentence) : Prop :=
  s.first_blank = Article.None ∧ s.second_blank = Article.A

/-- The theorem stating that the correct sentence has no article in the first blank and "a" in the second blank --/
theorem correct_articles : 
  ∃ (s : Sentence), is_correct_sentence s :=
sorry

end NUMINAMATH_CALUDE_correct_articles_l3941_394133


namespace NUMINAMATH_CALUDE_train_speed_increase_l3941_394152

theorem train_speed_increase (old_time new_time : ℝ) 
  (hold : old_time = 16 ∧ new_time = 14) : 
  (1 / new_time - 1 / old_time) / (1 / old_time) = 
  (1 / 14 - 1 / 16) / (1 / 16) := by
  sorry

end NUMINAMATH_CALUDE_train_speed_increase_l3941_394152


namespace NUMINAMATH_CALUDE_max_min_f_a4_range_a_inequality_l3941_394192

-- Define the function f
def f (a x : ℝ) : ℝ := x * abs (x - a) + 2 * x - 3

-- Theorem for part 1
theorem max_min_f_a4 :
  ∃ (max min : ℝ),
    (∀ x, 2 ≤ x ∧ x ≤ 5 → f 4 x ≤ max) ∧
    (∃ x, 2 ≤ x ∧ x ≤ 5 ∧ f 4 x = max) ∧
    (∀ x, 2 ≤ x ∧ x ≤ 5 → min ≤ f 4 x) ∧
    (∃ x, 2 ≤ x ∧ x ≤ 5 ∧ f 4 x = min) ∧
    max = 12 ∧ min = 5 :=
sorry

-- Theorem for part 2
theorem range_a_inequality :
  ∀ a : ℝ,
    (∀ x, 1 ≤ x ∧ x ≤ 2 → f a x ≤ 2 * x - 2) ↔
    (3 / 2 ≤ a ∧ a ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_max_min_f_a4_range_a_inequality_l3941_394192


namespace NUMINAMATH_CALUDE_coaching_charges_calculation_l3941_394190

/-- Number of days from January 1 to November 4 in a non-leap year -/
def daysOfCoaching : Nat := 308

/-- Total payment for coaching in dollars -/
def totalPayment : Int := 7038

/-- Daily coaching charges in dollars -/
def dailyCharges : ℚ := totalPayment / daysOfCoaching

theorem coaching_charges_calculation :
  dailyCharges = 7038 / 308 := by sorry

end NUMINAMATH_CALUDE_coaching_charges_calculation_l3941_394190


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3941_394169

theorem imaginary_part_of_complex_fraction :
  let z : ℂ := (2 - I) / I
  (z.im : ℝ) = -2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3941_394169


namespace NUMINAMATH_CALUDE_inverse_proportionality_l3941_394149

theorem inverse_proportionality (x y : ℝ) (k : ℝ) (h1 : x * y = k) (h2 : 5 * 10 = k) :
  40 * (5/4 : ℝ) = k := by sorry

end NUMINAMATH_CALUDE_inverse_proportionality_l3941_394149


namespace NUMINAMATH_CALUDE_max_trig_sum_value_l3941_394153

theorem max_trig_sum_value (θ₁ θ₂ θ₃ θ₄ θ₅ θ₆ : ℝ) :
  (∀ φ₁ φ₂ φ₃ φ₄ φ₅ φ₆ : ℝ,
    (Real.cos φ₁ * Real.sin φ₂ + Real.cos φ₂ * Real.sin φ₃ + 
     Real.cos φ₃ * Real.sin φ₄ + Real.cos φ₄ * Real.sin φ₅ + 
     Real.cos φ₅ * Real.sin φ₆ + Real.cos φ₆ * Real.sin φ₁) ≤
    (Real.cos θ₁ * Real.sin θ₂ + Real.cos θ₂ * Real.sin θ₃ + 
     Real.cos θ₃ * Real.sin θ₄ + Real.cos θ₄ * Real.sin θ₅ + 
     Real.cos θ₅ * Real.sin θ₆ + Real.cos θ₆ * Real.sin θ₁)) ∧
  (Real.cos θ₁ * Real.sin θ₂ + Real.cos θ₂ * Real.sin θ₃ + 
   Real.cos θ₃ * Real.sin θ₄ + Real.cos θ₄ * Real.sin θ₅ + 
   Real.cos θ₅ * Real.sin θ₆ + Real.cos θ₆ * Real.sin θ₁) = 3 + 3 * Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_max_trig_sum_value_l3941_394153


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_relation_l3941_394120

theorem arithmetic_geometric_sequence_relation :
  ∃ (a g : ℕ → ℕ) (n : ℕ),
    (∀ k, a (k + 1) = a k + (a 2 - a 1)) ∧  -- arithmetic sequence
    (∀ k, g (k + 1) = g k * (g 2 / g 1)) ∧  -- geometric sequence
    n = 14 ∧
    a 1 = g 1 ∧ a 2 = g 2 ∧ a 5 = g 3 ∧ a n = g 4 ∧
    g 1 + g 2 + g 3 + g 4 = 80 ∧
    a 1 = 2 ∧ a 2 = 6 ∧ a 5 = 18 ∧ a n = 54 ∧
    g 1 = 2 ∧ g 2 = 6 ∧ g 3 = 18 ∧ g 4 = 54 :=
by sorry


end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_relation_l3941_394120


namespace NUMINAMATH_CALUDE_intersection_point_is_unique_solution_l3941_394141

/-- The system of equations representing two lines -/
def line_system (x y : ℚ) : Prop :=
  2 * y = -x + 3 ∧ -y = 5 * x + 1

/-- The intersection point of the two lines -/
def intersection_point : ℚ × ℚ := (-5/9, 16/9)

/-- Theorem stating that the intersection point is the unique solution to the system of equations -/
theorem intersection_point_is_unique_solution :
  line_system intersection_point.1 intersection_point.2 ∧
  ∀ x y, line_system x y → (x, y) = intersection_point := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_is_unique_solution_l3941_394141


namespace NUMINAMATH_CALUDE_f_is_even_count_f_eq_2016_l3941_394180

/-- The smallest factor of n that is not 1 -/
def smallest_factor (n : ℕ) : ℕ := sorry

/-- The function f as defined in the problem -/
def f (n : ℕ) : ℕ := n + smallest_factor n

/-- Theorem stating that f(n) is always even for n > 1 -/
theorem f_is_even (n : ℕ) (h : n > 1) : Even (f n) := by sorry

/-- Theorem stating that there are exactly 3 positive integers n such that f(n) = 2016 -/
theorem count_f_eq_2016 : ∃! (s : Finset ℕ), (∀ n ∈ s, f n = 2016) ∧ s.card = 3 := by sorry

end NUMINAMATH_CALUDE_f_is_even_count_f_eq_2016_l3941_394180


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l3941_394130

theorem polynomial_divisibility : ∃ (q : ℝ → ℝ), ∀ x : ℝ, 
  4 * x^2 - 6 * x - 18 = (x - 3) * q x := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l3941_394130


namespace NUMINAMATH_CALUDE_zero_to_zero_undefined_l3941_394113

theorem zero_to_zero_undefined : ¬ ∃ (x : ℝ), 0^(0 : ℝ) = x := by
  sorry

end NUMINAMATH_CALUDE_zero_to_zero_undefined_l3941_394113


namespace NUMINAMATH_CALUDE_pencil_count_l3941_394127

theorem pencil_count (group_size : ℕ) (num_groups : ℕ) (h1 : group_size = 11) (h2 : num_groups = 14) :
  group_size * num_groups = 154 := by
  sorry

end NUMINAMATH_CALUDE_pencil_count_l3941_394127


namespace NUMINAMATH_CALUDE_cos_odd_function_phi_l3941_394183

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem cos_odd_function_phi (φ : ℝ) 
  (h1 : 0 ≤ φ) (h2 : φ ≤ π) 
  (h3 : is_odd_function (fun x ↦ Real.cos (x + φ))) : 
  φ = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_odd_function_phi_l3941_394183


namespace NUMINAMATH_CALUDE_product_ratio_l3941_394118

def range_start : Int := -2020
def range_end : Int := 2019

theorem product_ratio :
  let smallest_product := range_start * (range_start + 1) * (range_start + 2)
  let largest_product := (range_end - 2) * (range_end - 1) * range_end
  (smallest_product : ℚ) / largest_product = -2020 / 2017 := by
sorry

end NUMINAMATH_CALUDE_product_ratio_l3941_394118


namespace NUMINAMATH_CALUDE_intersection_range_l3941_394194

-- Define the circles
def circle_O₁ (x y : ℝ) : Prop := x^2 + y^2 = 25
def circle_O₂ (x y r : ℝ) : Prop := (x - 7)^2 + y^2 = r^2

-- Define the condition for r
def r_positive (r : ℝ) : Prop := r > 0

-- Define the intersection condition
def circles_intersect (r : ℝ) : Prop :=
  ∃ x y, circle_O₁ x y ∧ circle_O₂ x y r

-- Main theorem
theorem intersection_range :
  ∀ r, r_positive r → (circles_intersect r ↔ 2 < r ∧ r < 12) :=
sorry

end NUMINAMATH_CALUDE_intersection_range_l3941_394194


namespace NUMINAMATH_CALUDE_rectangle_area_is_72_l3941_394174

/-- Represents a circle with a center point and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a rectangle with two corner points -/
structure Rectangle where
  topLeft : ℝ × ℝ
  bottomRight : ℝ × ℝ

def circleP : Circle := { center := (0, 3), radius := 3 }
def circleQ : Circle := { center := (3, 3), radius := 3 }
def circleR : Circle := { center := (6, 3), radius := 3 }
def circleS : Circle := { center := (9, 3), radius := 3 }

def rectangleABCD : Rectangle := { topLeft := (0, 6), bottomRight := (12, 0) }

theorem rectangle_area_is_72 
  (h1 : circleP.radius = circleQ.radius ∧ circleP.radius = circleR.radius ∧ circleP.radius = circleS.radius)
  (h2 : circleP.center.2 = circleQ.center.2 ∧ circleP.center.2 = circleR.center.2 ∧ circleP.center.2 = circleS.center.2)
  (h3 : circleP.center.1 + circleP.radius = circleQ.center.1 ∧ 
        circleQ.center.1 + circleQ.radius = circleR.center.1 ∧
        circleR.center.1 + circleR.radius = circleS.center.1)
  (h4 : rectangleABCD.topLeft.1 = circleP.center.1 - circleP.radius ∧
        rectangleABCD.bottomRight.1 = circleS.center.1 + circleS.radius)
  (h5 : rectangleABCD.topLeft.2 = circleP.center.2 + circleP.radius ∧
        rectangleABCD.bottomRight.2 = circleP.center.2 - circleP.radius)
  : (rectangleABCD.bottomRight.1 - rectangleABCD.topLeft.1) * 
    (rectangleABCD.topLeft.2 - rectangleABCD.bottomRight.2) = 72 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_is_72_l3941_394174


namespace NUMINAMATH_CALUDE_opposite_reciprocal_expression_l3941_394123

theorem opposite_reciprocal_expression (m n p q : ℝ) 
  (h1 : m + n = 0) 
  (h2 : p * q = 1) : 
  -2023 * m + 3 / (p * q) - 2023 * n = 3 := by
sorry

end NUMINAMATH_CALUDE_opposite_reciprocal_expression_l3941_394123


namespace NUMINAMATH_CALUDE_white_surface_area_fraction_l3941_394189

/-- Represents a cube with side length and number of smaller cubes -/
structure Cube where
  side_length : ℕ
  num_smaller_cubes : ℕ

/-- Represents the composition of a larger cube -/
structure CubeComposition where
  large_cube : Cube
  small_cube : Cube
  num_red : ℕ
  num_white : ℕ

/-- Calculate the surface area of a cube -/
def surface_area (c : Cube) : ℕ := 6 * c.side_length^2

/-- Calculate the minimum number of visible faces for white cubes -/
def min_visible_white_faces (cc : CubeComposition) : ℕ :=
  cc.num_white - 1

/-- The theorem stating the fraction of white surface area -/
theorem white_surface_area_fraction (cc : CubeComposition) 
  (h1 : cc.large_cube.side_length = 4)
  (h2 : cc.small_cube.side_length = 1)
  (h3 : cc.large_cube.num_smaller_cubes = 64)
  (h4 : cc.num_red = 56)
  (h5 : cc.num_white = 8) :
  (min_visible_white_faces cc : ℚ) / (surface_area cc.large_cube : ℚ) = 7 / 96 := by
  sorry

end NUMINAMATH_CALUDE_white_surface_area_fraction_l3941_394189


namespace NUMINAMATH_CALUDE_min_value_not_e_squared_minus_2m_l3941_394143

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x * Real.exp x - (m / 2) * x^2 - m * x

theorem min_value_not_e_squared_minus_2m (m : ℝ) :
  ∃ (x : ℝ), x ∈ Set.Icc 1 2 ∧ f m x < Real.exp 2 - 2 * m :=
sorry

end NUMINAMATH_CALUDE_min_value_not_e_squared_minus_2m_l3941_394143


namespace NUMINAMATH_CALUDE_solution_system_equations_l3941_394125

theorem solution_system_equations (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₃ > 0) (h₄ : x₄ > 0) (h₅ : x₅ > 0)
  (eq₁ : x₁ + x₂ = x₃^2)
  (eq₂ : x₂ + x₃ = x₄^2)
  (eq₃ : x₃ + x₄ = x₅^2)
  (eq₄ : x₄ + x₅ = x₁^2)
  (eq₅ : x₅ + x₁ = x₂^2) :
  x₁ = 2 ∧ x₂ = 2 ∧ x₃ = 2 ∧ x₄ = 2 ∧ x₅ = 2 :=
by sorry

end NUMINAMATH_CALUDE_solution_system_equations_l3941_394125


namespace NUMINAMATH_CALUDE_prime_sum_equality_l3941_394150

theorem prime_sum_equality (p q n : ℕ) : 
  Prime p → Prime q → 0 < n → 
  p * (p + 3) + q * (q + 3) = n * (n + 3) → 
  ((p = 2 ∧ q = 3 ∧ n = 4) ∨ 
   (p = 3 ∧ q = 2 ∧ n = 4) ∨ 
   (p = 3 ∧ q = 7 ∧ n = 8) ∨ 
   (p = 7 ∧ q = 3 ∧ n = 8)) := by
sorry

end NUMINAMATH_CALUDE_prime_sum_equality_l3941_394150


namespace NUMINAMATH_CALUDE_prove_smallest_positive_angle_l3941_394107

def smallest_positive_angle_theorem : Prop :=
  ∃ θ : Real,
    θ > 0 ∧
    θ < 2 * Real.pi ∧
    Real.cos θ = Real.sin (60 * Real.pi / 180) + Real.cos (42 * Real.pi / 180) - 
                 Real.sin (12 * Real.pi / 180) - Real.cos (6 * Real.pi / 180) ∧
    θ = 66 * Real.pi / 180 ∧
    ∀ φ : Real, 
      φ > 0 → 
      φ < 2 * Real.pi → 
      Real.cos φ = Real.sin (60 * Real.pi / 180) + Real.cos (42 * Real.pi / 180) - 
                   Real.sin (12 * Real.pi / 180) - Real.cos (6 * Real.pi / 180) → 
      φ ≥ θ

theorem prove_smallest_positive_angle : smallest_positive_angle_theorem :=
sorry

end NUMINAMATH_CALUDE_prove_smallest_positive_angle_l3941_394107


namespace NUMINAMATH_CALUDE_cubic_equation_value_l3941_394108

theorem cubic_equation_value (m : ℝ) (h : m^2 + m - 1 = 0) : 
  m^3 + 2*m^2 - 2005 = -2004 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_value_l3941_394108


namespace NUMINAMATH_CALUDE_cost_of_480_chocolates_l3941_394168

/-- The cost of buying a given number of chocolates, given the box size and box cost -/
def chocolate_cost (total_chocolates : ℕ) (box_size : ℕ) (box_cost : ℕ) : ℕ :=
  (total_chocolates / box_size) * box_cost

/-- Theorem: The cost of 480 chocolates is $96, given that a box of 40 chocolates costs $8 -/
theorem cost_of_480_chocolates :
  chocolate_cost 480 40 8 = 96 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_480_chocolates_l3941_394168


namespace NUMINAMATH_CALUDE_evaluate_expression_l3941_394122

theorem evaluate_expression (x y z w : ℚ) 
  (hx : x = 1/4)
  (hy : y = 1/3)
  (hz : z = -12)
  (hw : w = 5) :
  x^2 * y^3 * z + w = 179/36 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3941_394122


namespace NUMINAMATH_CALUDE_banana_sharing_l3941_394103

/-- Proves that sharing 21 bananas equally among 3 friends results in 7 bananas per friend -/
theorem banana_sharing (total_bananas : ℕ) (num_friends : ℕ) (bananas_per_friend : ℕ) :
  total_bananas = 21 →
  num_friends = 3 →
  bananas_per_friend = total_bananas / num_friends →
  bananas_per_friend = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_banana_sharing_l3941_394103


namespace NUMINAMATH_CALUDE_expression_simplification_l3941_394160

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 5 + 2) :
  (x + 2) / (x - 1) / (x + 1 - 3 / (x - 1)) = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3941_394160


namespace NUMINAMATH_CALUDE_q_investment_time_l3941_394129

-- Define the investment ratio
def investment_ratio : ℚ := 7 / 5

-- Define the profit ratio
def profit_ratio : ℚ := 7 / 10

-- Define P's investment time in months
def p_time : ℚ := 2

-- Define Q's investment time as a variable
variable (q_time : ℚ)

-- Theorem statement
theorem q_investment_time : 
  (investment_ratio * p_time) / (q_time / investment_ratio) = profit_ratio → q_time = 4 := by
  sorry

end NUMINAMATH_CALUDE_q_investment_time_l3941_394129


namespace NUMINAMATH_CALUDE_intersection_of_sets_l3941_394136

theorem intersection_of_sets (M N : Set ℝ) : 
  M = {x : ℝ | Real.sqrt (x + 1) ≥ 0} →
  N = {x : ℝ | x^2 + x - 2 < 0} →
  M ∩ N = {x : ℝ | -1 ≤ x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l3941_394136


namespace NUMINAMATH_CALUDE_bookshop_online_sales_l3941_394158

theorem bookshop_online_sales (initial_books : ℕ) (saturday_instore : ℕ) (sunday_instore : ℕ)
  (sunday_online_increase : ℕ) (shipment : ℕ) (final_books : ℕ) :
  initial_books = 743 →
  saturday_instore = 37 →
  sunday_instore = 2 * saturday_instore →
  sunday_online_increase = 34 →
  shipment = 160 →
  final_books = 502 →
  ∃ (saturday_online : ℕ),
    final_books = initial_books - saturday_instore - saturday_online -
      sunday_instore - (saturday_online + sunday_online_increase) + shipment ∧
    saturday_online = 128 := by
  sorry

end NUMINAMATH_CALUDE_bookshop_online_sales_l3941_394158


namespace NUMINAMATH_CALUDE_percentage_loss_l3941_394145

def cost_price : ℝ := 1800
def selling_price : ℝ := 1350

theorem percentage_loss : (cost_price - selling_price) / cost_price * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_percentage_loss_l3941_394145


namespace NUMINAMATH_CALUDE_max_product_sum_2000_l3941_394187

theorem max_product_sum_2000 :
  (∃ (a b : ℤ), a + b = 2000 ∧ ∀ (x y : ℤ), x + y = 2000 → x * y ≤ a * b) ∧
  (∀ (a b : ℤ), a + b = 2000 → a * b ≤ 1000000) ∧
  (∃ (a b : ℤ), a + b = 2000 ∧ a * b = 1000000) :=
by sorry

end NUMINAMATH_CALUDE_max_product_sum_2000_l3941_394187


namespace NUMINAMATH_CALUDE_movie_of_the_year_fraction_l3941_394178

/-- The required fraction for a film to be considered for "movie of the year" -/
def required_fraction (total_members : ℕ) (min_lists : ℚ) : ℚ :=
  min_lists / total_members

/-- Theorem stating the required fraction for the Cinematic Academy's "movie of the year" consideration -/
theorem movie_of_the_year_fraction :
  required_fraction 765 (191.25 : ℚ) = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_movie_of_the_year_fraction_l3941_394178


namespace NUMINAMATH_CALUDE_workshop_duration_is_450_l3941_394175

/-- Calculates the duration of a workshop excluding breaks -/
def workshop_duration (total_hours : ℕ) (total_minutes : ℕ) (break_minutes : ℕ) : ℕ :=
  total_hours * 60 + total_minutes - break_minutes

/-- Theorem: The workshop duration excluding breaks is 450 minutes -/
theorem workshop_duration_is_450 :
  workshop_duration 8 20 50 = 450 := by
  sorry

end NUMINAMATH_CALUDE_workshop_duration_is_450_l3941_394175


namespace NUMINAMATH_CALUDE_quadratic_equation_equivalence_l3941_394196

theorem quadratic_equation_equivalence :
  ∀ x : ℝ, (2 * x^2 - 12 * x + 1 = 0) ↔ ((x - 3)^2 = 17/2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_equivalence_l3941_394196


namespace NUMINAMATH_CALUDE_combined_squares_perimeter_l3941_394181

/-- The perimeter of the resulting figure when combining two squares -/
theorem combined_squares_perimeter (p1 p2 : ℝ) (h1 : p1 = 40) (h2 : p2 = 100) : 
  p1 + p2 - 2 * (p1 / 4) = 120 :=
by sorry

end NUMINAMATH_CALUDE_combined_squares_perimeter_l3941_394181


namespace NUMINAMATH_CALUDE_range_of_k_l3941_394128

-- Define the inequality condition
def inequality_condition (k : ℝ) : Prop :=
  ∀ x > 0, Real.exp (x + 1) - (Real.log x + 2 * k) / x - k ≥ 0

-- Theorem statement
theorem range_of_k (k : ℝ) :
  inequality_condition k → k ∈ Set.Iic 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_k_l3941_394128


namespace NUMINAMATH_CALUDE_platform_length_platform_length_is_210_l3941_394185

/-- Given a train's speed and time to pass a platform and a man, calculate the platform length -/
theorem platform_length 
  (train_speed : ℝ) 
  (time_platform : ℝ) 
  (time_man : ℝ) : ℝ :=
  let train_speed_ms := train_speed * (1000 / 3600)
  let train_length := train_speed_ms * time_man
  let platform_length := train_speed_ms * time_platform - train_length
  platform_length

/-- The length of the platform is 210 meters -/
theorem platform_length_is_210 :
  platform_length 54 34 20 = 210 := by
  sorry

end NUMINAMATH_CALUDE_platform_length_platform_length_is_210_l3941_394185


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l3941_394100

theorem geometric_sequence_problem (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →  -- All terms are positive
  (∀ n, a (n + 1) = q * a n) →  -- Definition of geometric sequence
  a 1 + a 2 = 4/9 →  -- First condition
  a 3 + a 4 + a 5 + a 6 = 40 →  -- Second condition
  (a 7 + a 8 + a 9) / 9 = 117 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l3941_394100


namespace NUMINAMATH_CALUDE_sum_10_terms_formula_l3941_394105

/-- An arithmetic progression with sum of 4th and 12th terms equal to 20 -/
structure ArithmeticProgression where
  a : ℝ  -- First term
  d : ℝ  -- Common difference
  sum_4_12 : a + 3*d + a + 11*d = 20

/-- The sum of the first 10 terms of the arithmetic progression -/
def sum_10_terms (ap : ArithmeticProgression) : ℝ :=
  5 * (2*ap.a + 9*ap.d)

/-- Theorem: The sum of the first 10 terms equals 100 - 25d -/
theorem sum_10_terms_formula (ap : ArithmeticProgression) :
  sum_10_terms ap = 100 - 25*ap.d := by
  sorry

end NUMINAMATH_CALUDE_sum_10_terms_formula_l3941_394105


namespace NUMINAMATH_CALUDE_problem_solution_l3941_394157

theorem problem_solution :
  let A : ℝ → ℝ → ℝ := λ x y => -4 * x^2 - 4 * x * y + 1
  let B : ℝ → ℝ → ℝ := λ x y => x^2 + x * y - 5
  let x : ℝ := 1
  let y : ℝ := -1
  2 * B x y - A x y = -11 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3941_394157


namespace NUMINAMATH_CALUDE_happyTails_cats_count_l3941_394126

/-- Represents the number of cats that can perform a specific combination of tricks -/
structure CatTricks where
  jump : Nat
  fetch : Nat
  spin : Nat
  jumpFetch : Nat
  fetchSpin : Nat
  jumpSpin : Nat
  allThree : Nat
  none : Nat

/-- Calculates the total number of cats at HappyTails Training Center -/
def totalCats (ct : CatTricks) : Nat :=
  ct.jump + ct.fetch + ct.spin - ct.jumpFetch - ct.fetchSpin - ct.jumpSpin + ct.allThree + ct.none

/-- Theorem stating that the total number of cats at HappyTails Training Center is 70 -/
theorem happyTails_cats_count (ct : CatTricks)
  (h1 : ct.jump = 40)
  (h2 : ct.fetch = 25)
  (h3 : ct.spin = 30)
  (h4 : ct.jumpFetch = 15)
  (h5 : ct.fetchSpin = 10)
  (h6 : ct.jumpSpin = 12)
  (h7 : ct.allThree = 5)
  (h8 : ct.none = 7) :
  totalCats ct = 70 := by
  sorry

end NUMINAMATH_CALUDE_happyTails_cats_count_l3941_394126


namespace NUMINAMATH_CALUDE_chocolate_bar_expense_l3941_394163

def chocolate_bar_cost : ℚ := 3/2  -- $1.50 represented as a rational number
def smores_per_bar : ℕ := 3
def num_scouts : ℕ := 15
def smores_per_scout : ℕ := 2

theorem chocolate_bar_expense : 
  ↑num_scouts * ↑smores_per_scout / ↑smores_per_bar * chocolate_bar_cost = 15 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bar_expense_l3941_394163


namespace NUMINAMATH_CALUDE_new_student_weight_l3941_394167

theorem new_student_weight (initial_count : ℕ) (initial_avg : ℝ) (new_count : ℕ) (new_avg : ℝ) : 
  initial_count = 19 →
  initial_avg = 15 →
  new_count = initial_count + 1 →
  new_avg = 14.9 →
  (initial_count : ℝ) * initial_avg + (new_count * new_avg - initial_count * initial_avg) = 13 := by
  sorry

end NUMINAMATH_CALUDE_new_student_weight_l3941_394167


namespace NUMINAMATH_CALUDE_race_order_l3941_394191

-- Define the participants
inductive Participant : Type
  | Jia : Participant
  | Yi : Participant
  | Bing : Participant
  | Ding : Participant
  | Wu : Participant

-- Define a relation for "finished before"
def finished_before (a b : Participant) : Prop := sorry

-- Define the conditions
axiom ding_faster_than_yi : finished_before Participant.Ding Participant.Yi
axiom wu_before_bing : finished_before Participant.Wu Participant.Bing
axiom jia_between_bing_and_ding : 
  finished_before Participant.Bing Participant.Jia ∧ 
  finished_before Participant.Jia Participant.Ding

-- State the theorem
theorem race_order : 
  finished_before Participant.Wu Participant.Bing ∧
  finished_before Participant.Bing Participant.Jia ∧
  finished_before Participant.Jia Participant.Ding ∧
  finished_before Participant.Ding Participant.Yi :=
by sorry

end NUMINAMATH_CALUDE_race_order_l3941_394191


namespace NUMINAMATH_CALUDE_spinsters_to_cats_ratio_l3941_394177

/-- Given the number of spinsters and cats, prove their ratio is 2:7 -/
theorem spinsters_to_cats_ratio :
  ∀ (spinsters cats : ℕ),
    spinsters = 14 →
    cats = spinsters + 35 →
    (spinsters : ℚ) / (cats : ℚ) = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_spinsters_to_cats_ratio_l3941_394177


namespace NUMINAMATH_CALUDE_locus_of_N_l3941_394176

/-- The circle on which point M moves -/
def Circle (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 8*y = 0

/-- Point N lies on the ray OM -/
def OnRay (x y : ℝ) : Prop := ∃ (t : ℝ), t > 0 ∧ x = t * (x/((x^2 + y^2)^(1/2))) ∧ y = t * (y/((x^2 + y^2)^(1/2)))

/-- The product of distances |OM| and |ON| is 150 -/
def DistanceProduct (x y : ℝ) : Prop := (x^2 + y^2)^(1/2) * ((x^2 + y^2)^(1/2) / (x^2 + y^2)) = 150

theorem locus_of_N (x y : ℝ) :
  (∃ (mx my : ℝ), Circle mx my ∧ OnRay x y ∧ DistanceProduct x y) →
  3*x + 4*y = 75 := by sorry

end NUMINAMATH_CALUDE_locus_of_N_l3941_394176


namespace NUMINAMATH_CALUDE_sin_cos_difference_equals_half_l3941_394188

theorem sin_cos_difference_equals_half : 
  Real.sin (43 * π / 180) * Real.cos (13 * π / 180) - 
  Real.sin (13 * π / 180) * Real.cos (43 * π / 180) = 1/2 := by sorry

end NUMINAMATH_CALUDE_sin_cos_difference_equals_half_l3941_394188


namespace NUMINAMATH_CALUDE_total_marbles_l3941_394112

/-- Given 5 bags with 5 marbles each and 1 bag with 8 marbles,
    the total number of marbles in all 6 bags is 33. -/
theorem total_marbles (bags_of_five : Nat) (marbles_per_bag : Nat) (extra_bag : Nat) :
  bags_of_five = 5 →
  marbles_per_bag = 5 →
  extra_bag = 8 →
  bags_of_five * marbles_per_bag + extra_bag = 33 := by
  sorry

end NUMINAMATH_CALUDE_total_marbles_l3941_394112


namespace NUMINAMATH_CALUDE_return_trip_amount_l3941_394165

def initial_amount : ℝ := 50
def gasoline_cost : ℝ := 8
def lunch_cost : ℝ := 15.65
def gift_cost_per_person : ℝ := 5
def number_of_people : ℕ := 2
def grandma_gift_per_person : ℝ := 10

def total_expenses : ℝ := gasoline_cost + lunch_cost + (gift_cost_per_person * number_of_people)
def remaining_after_expenses : ℝ := initial_amount - total_expenses
def total_grandma_gift : ℝ := grandma_gift_per_person * number_of_people
def final_amount : ℝ := remaining_after_expenses + total_grandma_gift

theorem return_trip_amount :
  final_amount = 36.35 := by sorry

end NUMINAMATH_CALUDE_return_trip_amount_l3941_394165


namespace NUMINAMATH_CALUDE_adjacent_sum_of_six_l3941_394144

/-- Represents a 3x3 table filled with numbers 1 to 9 --/
def Table := Fin 3 → Fin 3 → Fin 9

/-- Returns the list of adjacent positions for a given position --/
def adjacent_positions (row col : Fin 3) : List (Fin 3 × Fin 3) := sorry

/-- Returns the sum of adjacent numbers for a given number in the table --/
def adjacent_sum (t : Table) (n : Fin 9) : ℕ := sorry

/-- Checks if the table satisfies the given conditions --/
def valid_table (t : Table) : Prop :=
  (t 0 0 = 0) ∧ (t 2 0 = 1) ∧ (t 0 2 = 2) ∧ (t 2 2 = 3) ∧
  (adjacent_sum t 4 = 9) ∧
  (∀ i j : Fin 3, ∀ k : Fin 9, (t i j = k) → (∀ i' j' : Fin 3, (i ≠ i' ∨ j ≠ j') → t i' j' ≠ k))

theorem adjacent_sum_of_six (t : Table) (h : valid_table t) : adjacent_sum t 5 = 29 := by
  sorry

end NUMINAMATH_CALUDE_adjacent_sum_of_six_l3941_394144


namespace NUMINAMATH_CALUDE_total_dresses_l3941_394184

theorem total_dresses (emily melissa debora sophia : ℕ) : 
  emily = 16 ∧ 
  melissa = emily / 2 ∧ 
  debora = melissa + 12 ∧ 
  sophia = debora - 5 → 
  emily + melissa + debora + sophia = 59 := by
sorry

end NUMINAMATH_CALUDE_total_dresses_l3941_394184


namespace NUMINAMATH_CALUDE_f_monotonicity_and_extrema_l3941_394119

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 + x^2 - 3*x + 1

theorem f_monotonicity_and_extrema :
  (∀ x y, x < y ∧ y < -3 → f x < f y) ∧
  (∀ x y, -3 < x ∧ x < y ∧ y < 1 → f x > f y) ∧
  (∀ x y, 1 < x ∧ x < y → f x < f y) ∧
  (∀ x, f x ≤ 10) ∧
  (∀ x, f x ≥ -2/3) ∧
  (∃ x, f x = 10) ∧
  (∃ x, f x = -2/3) :=
sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_extrema_l3941_394119


namespace NUMINAMATH_CALUDE_factorization_proof_l3941_394173

theorem factorization_proof (c : ℝ) : 196 * c^2 + 42 * c - 14 = 14 * c * (14 * c + 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l3941_394173


namespace NUMINAMATH_CALUDE_infinite_pairs_exist_infinitely_many_pairs_l3941_394199

/-- Recursive definition of the sequence a_n -/
def a : ℕ → ℕ
  | 0 => 4
  | 1 => 11
  | (n + 2) => 3 * a (n + 1) - a n

/-- Theorem stating the existence of infinitely many pairs satisfying the given properties -/
theorem infinite_pairs_exist : ∀ n : ℕ, n ≥ 1 → 
  (a n < a (n + 1)) ∧ 
  (Nat.gcd (a n) (a (n + 1)) = 1) ∧
  (a n ∣ a (n + 1)^2 - 5) ∧
  (a (n + 1) ∣ a n^2 - 5) := by
  sorry

/-- Corollary: There exist infinitely many pairs of positive integers satisfying the properties -/
theorem infinitely_many_pairs : 
  ∃ f : ℕ → ℕ × ℕ, ∀ n : ℕ, 
    let (a, b) := f n
    (a > b) ∧
    (Nat.gcd a b = 1) ∧
    (a ∣ b^2 - 5) ∧
    (b ∣ a^2 - 5) := by
  sorry

end NUMINAMATH_CALUDE_infinite_pairs_exist_infinitely_many_pairs_l3941_394199


namespace NUMINAMATH_CALUDE_arccos_one_half_eq_pi_third_l3941_394179

theorem arccos_one_half_eq_pi_third : Real.arccos (1/2) = π/3 := by
  sorry

end NUMINAMATH_CALUDE_arccos_one_half_eq_pi_third_l3941_394179


namespace NUMINAMATH_CALUDE_hcd_7350_165_minus_15_l3941_394135

theorem hcd_7350_165_minus_15 : Nat.gcd 7350 165 - 15 = 0 := by
  sorry

end NUMINAMATH_CALUDE_hcd_7350_165_minus_15_l3941_394135


namespace NUMINAMATH_CALUDE_max_volume_rotating_cube_max_volume_is_eight_l3941_394162

/-- The maximum volume of a cube that can rotate freely inside a cube with edge length 2 -/
theorem max_volume_rotating_cube (outer_edge : ℝ) (h : outer_edge = 2) :
  ∃ (inner_edge : ℝ),
    inner_edge > 0 ∧
    inner_edge * Real.sqrt 3 ≤ outer_edge * Real.sqrt 3 ∧
    ∀ (x : ℝ), x > 0 → x * Real.sqrt 3 ≤ outer_edge * Real.sqrt 3 → x^3 ≤ inner_edge^3 :=
by
  sorry

/-- The maximum volume of the rotating cube is 8 -/
theorem max_volume_is_eight (outer_edge : ℝ) (h : outer_edge = 2) :
  ∃ (inner_edge : ℝ),
    inner_edge > 0 ∧
    inner_edge * Real.sqrt 3 ≤ outer_edge * Real.sqrt 3 ∧
    inner_edge^3 = 8 ∧
    ∀ (x : ℝ), x > 0 → x * Real.sqrt 3 ≤ outer_edge * Real.sqrt 3 → x^3 ≤ 8 :=
by
  sorry

end NUMINAMATH_CALUDE_max_volume_rotating_cube_max_volume_is_eight_l3941_394162


namespace NUMINAMATH_CALUDE_basketball_team_average_weight_l3941_394110

/-- The average weight of a basketball team after adding new players -/
theorem basketball_team_average_weight 
  (original_players : ℕ) 
  (original_average : ℝ) 
  (new_player1 : ℝ) 
  (new_player2 : ℝ) 
  (new_player3 : ℝ) 
  (new_player4 : ℝ) 
  (h1 : original_players = 8) 
  (h2 : original_average = 105.5) 
  (h3 : new_player1 = 110.3) 
  (h4 : new_player2 = 99.7) 
  (h5 : new_player3 = 103.2) 
  (h6 : new_player4 = 115.4) : 
  (original_players * original_average + new_player1 + new_player2 + new_player3 + new_player4) / (original_players + 4) = 106.05 := by
  sorry

end NUMINAMATH_CALUDE_basketball_team_average_weight_l3941_394110


namespace NUMINAMATH_CALUDE_negative_2023_times_99_l3941_394195

theorem negative_2023_times_99 (p : ℤ) (h : p = (-2023) * 100) : (-2023) * 99 = p + 2023 := by
  sorry

end NUMINAMATH_CALUDE_negative_2023_times_99_l3941_394195


namespace NUMINAMATH_CALUDE_compute_expression_l3941_394147

theorem compute_expression : 3 * ((25 + 15)^2 - (25 - 15)^2) = 4500 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l3941_394147


namespace NUMINAMATH_CALUDE_spherical_segment_surface_area_equals_circle_area_l3941_394102

/-- Given a spherical segment with radius R and height H, and a circle with radius b
    where b² = 2RH, the surface area of the spherical segment (2πRH) is equal to
    the area of the circle (πb²). -/
theorem spherical_segment_surface_area_equals_circle_area
  (R H b : ℝ) (h : b^2 = 2 * R * H) :
  2 * Real.pi * R * H = Real.pi * b^2 := by
  sorry

end NUMINAMATH_CALUDE_spherical_segment_surface_area_equals_circle_area_l3941_394102


namespace NUMINAMATH_CALUDE_sum_of_squares_zero_l3941_394166

theorem sum_of_squares_zero (a b c : ℝ) :
  (a - 6)^2 + (b - 3)^2 + (c - 2)^2 = 0 → a + b + c = 11 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_zero_l3941_394166


namespace NUMINAMATH_CALUDE_total_savings_l3941_394154

/-- The total savings from buying discounted milk and cereal with promotions -/
theorem total_savings (M C : ℝ) : ℝ := by
  -- M: original price of a gallon of milk
  -- C: original price of a box of cereal
  -- Milk discount: 25%
  -- Cereal promotion: buy two, get one 50% off
  -- Buying 3 gallons of milk and 6 boxes of cereal

  /- Define the milk discount -/
  let milk_discount_percent : ℝ := 0.25

  /- Define the cereal promotion discount -/
  let cereal_promotion_discount : ℝ := 0.5

  /- Calculate the savings on milk -/
  let milk_savings : ℝ := 3 * M * milk_discount_percent

  /- Calculate the savings on cereal -/
  let cereal_savings : ℝ := 2 * C * cereal_promotion_discount

  /- Calculate the total savings -/
  let total_savings : ℝ := milk_savings + cereal_savings

  /- Prove that the total savings equals (0.75 * M) + C -/
  have : total_savings = (0.75 * M) + C := by sorry

  /- Return the total savings -/
  exact total_savings

end NUMINAMATH_CALUDE_total_savings_l3941_394154


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3941_394116

theorem imaginary_part_of_complex_fraction (i : ℂ) (h : i^2 = -1) :
  let z : ℂ := (3 * i + 1) / (1 - i)
  Complex.im z = 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3941_394116


namespace NUMINAMATH_CALUDE_alpha_one_sufficient_not_necessary_l3941_394155

-- Define sets A and B
def A : Set ℝ := {x | 2 < x ∧ x < 3}
def B (α : ℝ) : Set ℝ := {x | (x + 2) * (x - α) < 0}

-- Statement to prove
theorem alpha_one_sufficient_not_necessary :
  (∀ x, x ∈ A ∩ B 1 → False) ∧
  (∃ α, α ≠ 1 ∧ ∀ x, x ∈ A ∩ B α → False) := by sorry

end NUMINAMATH_CALUDE_alpha_one_sufficient_not_necessary_l3941_394155
