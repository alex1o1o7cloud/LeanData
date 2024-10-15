import Mathlib

namespace NUMINAMATH_CALUDE_age_ratio_problem_l34_3435

theorem age_ratio_problem (sam sue kendra : ℕ) : 
  kendra = 3 * sam →
  kendra = 18 →
  (sam + 3) + (sue + 3) + (kendra + 3) = 36 →
  sam / sue = 2 := by
sorry

end NUMINAMATH_CALUDE_age_ratio_problem_l34_3435


namespace NUMINAMATH_CALUDE_opposite_of_neg_three_squared_l34_3425

theorem opposite_of_neg_three_squared : -(-(3^2)) = 9 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_neg_three_squared_l34_3425


namespace NUMINAMATH_CALUDE_prob_draw_heart_is_one_fourth_l34_3420

/-- A deck of cards with a specific number of cards, ranks, and suits. -/
structure Deck where
  total_cards : ℕ
  num_ranks : ℕ
  num_suits : ℕ
  cards_per_suit : ℕ
  h1 : total_cards = num_suits * cards_per_suit
  h2 : cards_per_suit = num_ranks

/-- The probability of drawing a card from a specific suit in a given deck. -/
def prob_draw_suit (d : Deck) : ℚ :=
  d.cards_per_suit / d.total_cards

/-- The special deck described in the problem. -/
def special_deck : Deck where
  total_cards := 60
  num_ranks := 15
  num_suits := 4
  cards_per_suit := 15
  h1 := by rfl
  h2 := by rfl

theorem prob_draw_heart_is_one_fourth :
  prob_draw_suit special_deck = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_prob_draw_heart_is_one_fourth_l34_3420


namespace NUMINAMATH_CALUDE_sequence_not_contains_square_l34_3450

def a : ℕ → ℚ
  | 0 => 2016
  | n + 1 => a n + 2 / (a n)

theorem sequence_not_contains_square : ∀ n : ℕ, ¬ ∃ q : ℚ, a n = q^2 := by
  sorry

end NUMINAMATH_CALUDE_sequence_not_contains_square_l34_3450


namespace NUMINAMATH_CALUDE_calculate_expression_l34_3432

theorem calculate_expression : 5 * 405 + 4 * 405 - 3 * 405 + 404 = 2834 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l34_3432


namespace NUMINAMATH_CALUDE_book_price_increase_l34_3474

theorem book_price_increase (original_price new_price : ℝ) 
  (h1 : original_price = 300)
  (h2 : new_price = 390) :
  (new_price - original_price) / original_price * 100 = 30 := by
sorry

end NUMINAMATH_CALUDE_book_price_increase_l34_3474


namespace NUMINAMATH_CALUDE_walter_zoo_time_l34_3463

theorem walter_zoo_time (total_time seals penguins elephants : ℕ) : 
  total_time = 130 ∧ 
  penguins = 8 * seals ∧ 
  elephants = 13 ∧ 
  seals + penguins + elephants = total_time → 
  seals = 13 := by
sorry

end NUMINAMATH_CALUDE_walter_zoo_time_l34_3463


namespace NUMINAMATH_CALUDE_fraction_equality_l34_3403

theorem fraction_equality (a b : ℝ) (h : a / b = 5 / 2) : (a + 2*b) / (a - b) = 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l34_3403


namespace NUMINAMATH_CALUDE_negative_squared_times_squared_l34_3422

theorem negative_squared_times_squared (a : ℝ) : -a^2 * a^2 = -a^4 := by
  sorry

end NUMINAMATH_CALUDE_negative_squared_times_squared_l34_3422


namespace NUMINAMATH_CALUDE_f_of_two_eq_two_l34_3492

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x + 3 * f (8 - x) = x

/-- Theorem stating that for any function satisfying the functional equation, f(2) = 2 -/
theorem f_of_two_eq_two (f : ℝ → ℝ) (h : FunctionalEquation f) : f 2 = 2 := by
  sorry

#check f_of_two_eq_two

end NUMINAMATH_CALUDE_f_of_two_eq_two_l34_3492


namespace NUMINAMATH_CALUDE_triangle_side_formulas_l34_3434

/-- Given a triangle ABC with sides a, b, c, altitude m from A, and midline k from A,
    where b + c = 2l, prove the expressions for sides a, b, and c. -/
theorem triangle_side_formulas (a b c l m k : ℝ) : 
  b + c = 2 * l →
  k^2 = (b^2 + c^2) / 4 + (a / 2)^2 →
  m = (b * c) / a →
  b = l + Real.sqrt ((k^2 - m^2) * (k^2 - l^2) / (k^2 - m^2 - l^2)) ∧
  c = l - Real.sqrt ((k^2 - m^2) * (k^2 - l^2) / (k^2 - m^2 - l^2)) ∧
  a = 2 * l * Real.sqrt ((k^2 - l^2) / (k^2 - m^2 - l^2)) := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_formulas_l34_3434


namespace NUMINAMATH_CALUDE_problem_2019_1981_l34_3430

theorem problem_2019_1981 : (2019 + 1981)^2 / 121 = 132231 := by
  sorry

end NUMINAMATH_CALUDE_problem_2019_1981_l34_3430


namespace NUMINAMATH_CALUDE_score_96_not_possible_l34_3415

/-- Represents the score on a test with 25 questions -/
structure TestScore where
  correct : Nat
  unanswered : Nat
  incorrect : Nat
  h_total : correct + unanswered + incorrect = 25

/-- Calculates the total score for a given TestScore -/
def totalScore (ts : TestScore) : Nat :=
  4 * ts.correct + 2 * ts.unanswered

/-- Theorem stating that a score of 96 is not achievable -/
theorem score_96_not_possible :
  ¬ ∃ (ts : TestScore), totalScore ts = 96 := by
  sorry

end NUMINAMATH_CALUDE_score_96_not_possible_l34_3415


namespace NUMINAMATH_CALUDE_age_ratio_in_two_years_l34_3481

/-- Pete's current age -/
def p : ℕ := sorry

/-- Mandy's current age -/
def m : ℕ := sorry

/-- The number of years until the ratio of their ages is 3:2 -/
def x : ℕ := sorry

/-- Pete's age two years ago was twice Mandy's age two years ago -/
axiom past_condition_1 : p - 2 = 2 * (m - 2)

/-- Pete's age four years ago was three times Mandy's age four years ago -/
axiom past_condition_2 : p - 4 = 3 * (m - 4)

/-- The ratio of their ages will be 3:2 after x years -/
axiom future_ratio : (p + x) / (m + x) = 3 / 2

theorem age_ratio_in_two_years :
  x = 2 :=
sorry

end NUMINAMATH_CALUDE_age_ratio_in_two_years_l34_3481


namespace NUMINAMATH_CALUDE_tree_growth_problem_l34_3428

/-- Tree growth problem -/
theorem tree_growth_problem (initial_height : ℝ) (yearly_growth : ℝ) (height_ratio : ℝ) :
  initial_height = 4 →
  yearly_growth = 1 →
  height_ratio = 5/4 →
  ∃ (years : ℕ), 
    (initial_height + years * yearly_growth) = 
    height_ratio * (initial_height + 4 * yearly_growth) ∧
    years = 6 :=
by sorry

end NUMINAMATH_CALUDE_tree_growth_problem_l34_3428


namespace NUMINAMATH_CALUDE_packaging_combinations_l34_3497

/-- The number of wrapping paper designs --/
def num_wrapping_paper : ℕ := 10

/-- The number of ribbon colors --/
def num_ribbons : ℕ := 4

/-- The number of gift card varieties --/
def num_gift_cards : ℕ := 5

/-- The number of decorative sticker styles --/
def num_stickers : ℕ := 6

/-- The total number of unique packaging combinations --/
def total_combinations : ℕ := num_wrapping_paper * num_ribbons * num_gift_cards * num_stickers

/-- Theorem stating that the total number of unique packaging combinations is 1200 --/
theorem packaging_combinations : total_combinations = 1200 := by
  sorry

end NUMINAMATH_CALUDE_packaging_combinations_l34_3497


namespace NUMINAMATH_CALUDE_external_tangent_chord_length_l34_3401

theorem external_tangent_chord_length (r₁ r₂ R : ℝ) (h₁ : r₁ = 4) (h₂ : r₂ = 8) (h₃ : R = 12) 
  (h₄ : r₁ + r₂ = R - r₁) (h₅ : r₁ + r₂ = R - r₂) : 
  ∃ (l : ℝ), l^2 = 518.4 ∧ 
  l^2 = 4 * ((R^2) - (((2 * r₂ + r₁) / 3)^2)) :=
by sorry

end NUMINAMATH_CALUDE_external_tangent_chord_length_l34_3401


namespace NUMINAMATH_CALUDE_coord_sum_of_point_B_l34_3447

/-- Given two points A(0, 0) and B(x, 3) where the slope of AB is 3/4,
    prove that the sum of B's coordinates is 7. -/
theorem coord_sum_of_point_B (x : ℝ) : 
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (x, 3)
  (3 - 0) / (x - 0) = 3 / 4 →
  x + 3 = 7 := by
sorry

end NUMINAMATH_CALUDE_coord_sum_of_point_B_l34_3447


namespace NUMINAMATH_CALUDE_market_equilibrium_and_subsidy_effect_l34_3448

-- Define the supply function
def supply (p : ℝ) : ℝ := 2 + 8 * p

-- Define the demand function (to be derived)
def demand (p : ℝ) : ℝ := -2 * p + 12

-- Define equilibrium
def is_equilibrium (p : ℝ) : Prop := supply p = demand p

-- Define the subsidy amount
def subsidy : ℝ := 1

-- Define the new supply function with subsidy
def supply_with_subsidy (p : ℝ) : ℝ := supply (p + subsidy)

-- Define the new equilibrium with subsidy
def is_equilibrium_with_subsidy (p : ℝ) : Prop := supply_with_subsidy p = demand p

theorem market_equilibrium_and_subsidy_effect :
  -- Original equilibrium
  (∃ p q : ℝ, p = 1 ∧ q = 10 ∧ is_equilibrium p ∧ supply p = q) ∧
  -- Effect of subsidy
  (∃ p' q' : ℝ, is_equilibrium_with_subsidy p' ∧ supply_with_subsidy p' = q' ∧ q' - 10 = 1.6) :=
by sorry

end NUMINAMATH_CALUDE_market_equilibrium_and_subsidy_effect_l34_3448


namespace NUMINAMATH_CALUDE_driveway_snow_volume_l34_3472

/-- The volume of snow on a driveway -/
def snow_volume (length width depth : ℝ) : ℝ := length * width * depth

/-- Theorem: The volume of snow on a driveway with length 30 feet, width 3 feet, 
    and snow depth 0.75 feet is equal to 67.5 cubic feet -/
theorem driveway_snow_volume :
  snow_volume 30 3 0.75 = 67.5 := by
  sorry

end NUMINAMATH_CALUDE_driveway_snow_volume_l34_3472


namespace NUMINAMATH_CALUDE_trig_identity_l34_3455

theorem trig_identity (α : Real) (h : Real.sin (π / 8 + α) = 3 / 4) :
  Real.cos (3 * π / 8 - α) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l34_3455


namespace NUMINAMATH_CALUDE_digit_721_of_3_over_11_l34_3471

theorem digit_721_of_3_over_11 : ∃ (d : ℕ), d = 2 ∧ 
  (∃ (seq : ℕ → ℕ), 
    (∀ n, seq n < 10) ∧ 
    (∀ n, (3 * 10^(n+1)) % 11 = seq n) ∧
    seq 720 = d) := by
  sorry

end NUMINAMATH_CALUDE_digit_721_of_3_over_11_l34_3471


namespace NUMINAMATH_CALUDE_coin_division_problem_l34_3467

theorem coin_division_problem : ∃ n : ℕ,
  (∀ m : ℕ, m > n → (m % 8 ≠ 6 ∨ m % 7 ≠ 5)) ∧
  n % 8 = 6 ∧
  n % 7 = 5 ∧
  n % 9 = 0 :=
sorry

end NUMINAMATH_CALUDE_coin_division_problem_l34_3467


namespace NUMINAMATH_CALUDE_average_unchanged_with_double_inclusion_l34_3482

theorem average_unchanged_with_double_inclusion (n : ℕ) (scores : Fin n → ℝ) :
  let original_avg := (Finset.sum Finset.univ (λ i => scores i)) / n
  let new_sum := (Finset.sum Finset.univ (λ i => scores i)) + 2 * original_avg
  let new_avg := new_sum / (n + 2)
  new_avg = original_avg :=
by sorry

end NUMINAMATH_CALUDE_average_unchanged_with_double_inclusion_l34_3482


namespace NUMINAMATH_CALUDE_average_income_P_and_R_l34_3491

/-- Given the average monthly incomes of different pairs of people and the income of one person,
    prove that the average monthly income of P and R is 5200. -/
theorem average_income_P_and_R (P Q R : ℕ) : 
  (P + Q) / 2 = 5050 →
  (Q + R) / 2 = 6250 →
  P = 4000 →
  (P + R) / 2 = 5200 := by
  sorry

end NUMINAMATH_CALUDE_average_income_P_and_R_l34_3491


namespace NUMINAMATH_CALUDE_white_shirt_cost_is_25_l34_3402

/-- Represents the t-shirt sale scenario -/
structure TShirtSale where
  totalShirts : ℕ
  saleTime : ℕ
  blackShirtCost : ℕ
  revenuePerMinute : ℕ

/-- Calculates the cost of white t-shirts given the sale conditions -/
def whiteShirtCost (sale : TShirtSale) : ℕ :=
  let totalRevenue := sale.revenuePerMinute * sale.saleTime
  let blackShirts := sale.totalShirts / 2
  let whiteShirts := sale.totalShirts / 2
  let blackRevenue := blackShirts * sale.blackShirtCost
  let whiteRevenue := totalRevenue - blackRevenue
  whiteRevenue / whiteShirts

/-- Theorem stating that the white t-shirt cost is $25 under the given conditions -/
theorem white_shirt_cost_is_25 (sale : TShirtSale) 
  (h1 : sale.totalShirts = 200)
  (h2 : sale.saleTime = 25)
  (h3 : sale.blackShirtCost = 30)
  (h4 : sale.revenuePerMinute = 220) :
  whiteShirtCost sale = 25 := by
  sorry

#eval whiteShirtCost { totalShirts := 200, saleTime := 25, blackShirtCost := 30, revenuePerMinute := 220 }

end NUMINAMATH_CALUDE_white_shirt_cost_is_25_l34_3402


namespace NUMINAMATH_CALUDE_lizzy_money_theorem_l34_3460

def lizzy_money_problem (mother_gave uncle_gave father_gave spent_on_candy : ℕ) : Prop :=
  let initial_amount := mother_gave + father_gave
  let amount_after_spending := initial_amount - spent_on_candy
  let final_amount := amount_after_spending + uncle_gave
  final_amount = 140

theorem lizzy_money_theorem :
  lizzy_money_problem 80 70 40 50 := by
  sorry

end NUMINAMATH_CALUDE_lizzy_money_theorem_l34_3460


namespace NUMINAMATH_CALUDE_perpendicular_vectors_second_component_l34_3489

/-- Given two 2D vectors a and b, if they are perpendicular, then the second component of b is 2. -/
theorem perpendicular_vectors_second_component (a b : ℝ × ℝ) :
  a = (1, 2) →
  b.1 = -4 →
  a.1 * b.1 + a.2 * b.2 = 0 →
  b.2 = 2 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_second_component_l34_3489


namespace NUMINAMATH_CALUDE_cubic_function_extrema_difference_l34_3418

/-- A cubic function with specific properties -/
def f (a b c : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x^2 + 3*b*x + c

/-- The derivative of f -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 6*a*x + 3*b

/-- The second derivative of f -/
def f'' (a : ℝ) (x : ℝ) : ℝ := 6*x + 6*a

theorem cubic_function_extrema_difference (a b c : ℝ) :
  (f' a b 2 = 0) →  -- Extremum at x = 2
  (f' a b 1 = -3) →  -- Tangent line at x = 1 has slope -3 (parallel to 6x + 2y + 5 = 0)
  (∃ (x_max x_min : ℝ), 
    (∀ x, f a b c x ≤ f a b c x_max) ∧ 
    (∀ x, f a b c x ≥ f a b c x_min) ∧
    (f a b c x_max - f a b c x_min = 4)) := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_extrema_difference_l34_3418


namespace NUMINAMATH_CALUDE_preceding_binary_number_l34_3451

/-- Represents a binary number as a list of bits (0 or 1) -/
def BinaryNumber := List Nat

/-- Converts a binary number to its decimal representation -/
def binary_to_decimal (b : BinaryNumber) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + bit * 2^i) 0

/-- Converts a decimal number to its binary representation -/
def decimal_to_binary (n : Nat) : BinaryNumber :=
  sorry

theorem preceding_binary_number (M : BinaryNumber) :
  M = [0, 1, 0, 1, 0, 0, 1] →
  decimal_to_binary (binary_to_decimal M - 1) = [1, 0, 0, 1, 0, 0, 1] :=
sorry

end NUMINAMATH_CALUDE_preceding_binary_number_l34_3451


namespace NUMINAMATH_CALUDE_monic_cubic_polynomial_uniqueness_l34_3485

/-- A monic cubic polynomial with real coefficients -/
def MonicCubicPolynomial (a b c : ℝ) (x : ℂ) : ℂ := x^3 + a*x^2 + b*x + c

theorem monic_cubic_polynomial_uniqueness (a b c : ℝ) :
  let q := MonicCubicPolynomial a b c
  (q (2 - I) = 0) → (q 0 = -40) →
  (∀ x, q x = x^3 - (61/4)*x^2 + (305/4)*x - 225/4) :=
sorry

end NUMINAMATH_CALUDE_monic_cubic_polynomial_uniqueness_l34_3485


namespace NUMINAMATH_CALUDE_billy_weight_l34_3464

theorem billy_weight (carl_weight brad_weight dave_weight billy_weight edgar_weight : ℝ) :
  carl_weight = 145 ∧
  brad_weight = carl_weight + 5 ∧
  dave_weight = carl_weight + 8 ∧
  dave_weight = 2 * brad_weight ∧
  edgar_weight = 3 * dave_weight - 20 ∧
  billy_weight = brad_weight + 9 →
  billy_weight = 85.5 := by
sorry

end NUMINAMATH_CALUDE_billy_weight_l34_3464


namespace NUMINAMATH_CALUDE_factor_expression_l34_3419

theorem factor_expression (a b c : ℝ) :
  a^4 * (b^3 - c^3) + b^4 * (c^3 - a^3) + c^4 * (a^3 - b^3) =
  (a - b) * (b - c) * (c - a) * (a*b^3 + a*c^3 + a*b*c^2 + b^2*c^2) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l34_3419


namespace NUMINAMATH_CALUDE_intersection_m_complement_n_l34_3454

/-- The intersection of set M and the complement of set N in the real numbers -/
theorem intersection_m_complement_n :
  let U : Set ℝ := Set.univ
  let M : Set ℝ := {x | x^2 - 2*x < 0}
  let N : Set ℝ := {x | x ≥ 1}
  M ∩ (U \ N) = {x | 0 < x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_m_complement_n_l34_3454


namespace NUMINAMATH_CALUDE_at_least_one_third_l34_3406

theorem at_least_one_third (a b c : ℝ) (h : a + b + c = 1) :
  (a ≥ 1/3) ∨ (b ≥ 1/3) ∨ (c ≥ 1/3) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_third_l34_3406


namespace NUMINAMATH_CALUDE_range_of_a_l34_3412

-- Define the propositions p and q as functions of a
def p (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (2*a - 1)^x < (2*a - 1)^y

def q (a : ℝ) : Prop := ∀ x : ℝ, 2*a*x^2 - 2*a*x + 1 > 0

-- Define the range of a
def range_a (a : ℝ) : Prop := (0 ≤ a ∧ a ≤ 1) ∨ (a ≥ 2)

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (p a ∨ q a) ∧ ¬(p a ∧ q a) → range_a a :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l34_3412


namespace NUMINAMATH_CALUDE_boy_age_theorem_l34_3480

/-- The age of the boy not included in either group -/
def X (A : ℝ) : ℝ := 606 - 11 * A

/-- Theorem stating the relationship between X and A -/
theorem boy_age_theorem (A : ℝ) :
  let first_six_total : ℝ := 6 * 49
  let last_six_total : ℝ := 6 * 52
  let total_boys : ℕ := 11
  X A = first_six_total + last_six_total - total_boys * A := by
  sorry

end NUMINAMATH_CALUDE_boy_age_theorem_l34_3480


namespace NUMINAMATH_CALUDE_polynomial_properties_l34_3461

-- Define the polynomial coefficients
variable (a : Fin 12 → ℚ)

-- Define the main equation
def main_equation (x : ℚ) : Prop :=
  (x - 2)^11 = a 0 + a 1 * (x - 1) + a 2 * (x - 1)^2 + 
               a 3 * (x - 1)^3 + a 4 * (x - 1)^4 + a 5 * (x - 1)^5 + 
               a 6 * (x - 1)^6 + a 7 * (x - 1)^7 + a 8 * (x - 1)^8 + 
               a 9 * (x - 1)^9 + a 10 * (x - 1)^10 + a 11 * (x - 1)^11

-- Theorem to prove
theorem polynomial_properties (a : Fin 12 → ℚ) 
  (h : ∀ x, main_equation a x) : 
  a 10 = -11 ∧ a 2 + a 4 + a 6 + a 8 + a 10 = -1023 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_properties_l34_3461


namespace NUMINAMATH_CALUDE_trigonometric_sum_equals_sqrt_three_l34_3457

theorem trigonometric_sum_equals_sqrt_three (x : ℝ) 
  (h : Real.tan (4 * x) = Real.sqrt 3 / 3) : 
  (Real.sin (4 * x)) / (Real.cos (8 * x) * Real.cos (4 * x)) + 
  (Real.sin (2 * x)) / (Real.cos (4 * x) * Real.cos (2 * x)) + 
  (Real.sin x) / (Real.cos (2 * x) * Real.cos x) + 
  (Real.sin x) / (Real.cos x) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_sum_equals_sqrt_three_l34_3457


namespace NUMINAMATH_CALUDE_ruffy_is_nine_l34_3477

/-- Ruffy's current age -/
def ruffy_age : ℕ := 9

/-- Orlie's current age -/
def orlie_age : ℕ := 12

/-- Relation between Ruffy's and Orlie's current ages -/
axiom current_age_relation : ruffy_age = (3 * orlie_age) / 4

/-- Relation between Ruffy's and Orlie's ages four years ago -/
axiom past_age_relation : ruffy_age - 4 = (orlie_age - 4) / 2 + 1

/-- Theorem: Ruffy's current age is 9 years -/
theorem ruffy_is_nine : ruffy_age = 9 := by sorry

end NUMINAMATH_CALUDE_ruffy_is_nine_l34_3477


namespace NUMINAMATH_CALUDE_lcm_gcd_product_l34_3465

theorem lcm_gcd_product (a b : ℕ) (ha : a = 9) (hb : b = 15) :
  Nat.lcm a b * Nat.gcd a b = 135 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_product_l34_3465


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l34_3493

/-- Given a square with perimeter 160 units divided into 4 congruent rectangles,
    prove that the perimeter of one rectangle is 120 units. -/
theorem rectangle_perimeter (square_perimeter : ℝ) (h1 : square_perimeter = 160) :
  let square_side : ℝ := square_perimeter / 4
  let rect_width : ℝ := square_side / 2
  let rect_height : ℝ := square_side
  let rect_perimeter : ℝ := 2 * (rect_width + rect_height)
  rect_perimeter = 120 := by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l34_3493


namespace NUMINAMATH_CALUDE_siblings_height_l34_3479

/-- The total height of 5 siblings -/
def total_height (h1 h2 h3 h4 h5 : ℕ) : ℕ := h1 + h2 + h3 + h4 + h5

/-- Theorem stating the total height of the 5 siblings is 330 inches -/
theorem siblings_height :
  ∃ (h5 : ℕ), 
    total_height 66 66 60 68 h5 = 330 ∧ h5 = 68 + 2 := by
  sorry

end NUMINAMATH_CALUDE_siblings_height_l34_3479


namespace NUMINAMATH_CALUDE_functional_equation_solution_l34_3417

theorem functional_equation_solution (f : ℤ → ℤ) 
  (h : ∀ (x y : ℤ), f (x + y) = f x + f y + 2) :
  ∃ (a : ℤ), ∀ (x : ℤ), f x = a * x - 2 := by
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l34_3417


namespace NUMINAMATH_CALUDE_classroom_books_count_l34_3488

theorem classroom_books_count (num_children : ℕ) (books_per_child : ℕ) (teacher_books : ℕ) : 
  num_children = 10 → books_per_child = 7 → teacher_books = 8 →
  num_children * books_per_child + teacher_books = 78 := by
sorry

end NUMINAMATH_CALUDE_classroom_books_count_l34_3488


namespace NUMINAMATH_CALUDE_ninth_day_practice_correct_l34_3499

/-- The number of minutes Jenna practices piano on the 9th day to achieve
    an average of 100 minutes per day over a 9-day period, given her
    practice times for the first 8 days. -/
def ninth_day_practice (days_type1 days_type2 : ℕ) 
                       (minutes_type1 minutes_type2 : ℕ) : ℕ :=
  let total_days := days_type1 + days_type2 + 1
  let target_total := total_days * 100
  let current_total := days_type1 * minutes_type1 + days_type2 * minutes_type2
  target_total - current_total

theorem ninth_day_practice_correct :
  ninth_day_practice 6 2 80 105 = 210 :=
by sorry

end NUMINAMATH_CALUDE_ninth_day_practice_correct_l34_3499


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l34_3452

theorem quadratic_inequality_solution_set (x : ℝ) : 
  x^2 - 2*x ≤ 0 ↔ 0 ≤ x ∧ x ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l34_3452


namespace NUMINAMATH_CALUDE_function_extrema_and_inequality_l34_3405

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x - 0.5 * x^2 + x

def g (x : ℝ) : ℝ := 0.5 * x^2 - 2 * x + 1

theorem function_extrema_and_inequality (e : ℝ) (h_e : e = Real.exp 1) :
  (∃ (max min : ℝ), 
    (∀ x ∈ Set.Icc 1 (e^2), f 2 x ≤ max) ∧ 
    (∃ x₀ ∈ Set.Icc 1 (e^2), f 2 x₀ = max) ∧
    (∀ x ∈ Set.Icc 1 (e^2), min ≤ f 2 x) ∧ 
    (∃ x₁ ∈ Set.Icc 1 (e^2), f 2 x₁ = min) ∧
    max = 2 * Real.log 2 ∧
    min = 4 + e^2 - 0.5 * e^4) ∧
  (∀ a : ℝ, (∀ x > 0, f a x + g x ≤ 0) ↔ a = 1) :=
sorry

end NUMINAMATH_CALUDE_function_extrema_and_inequality_l34_3405


namespace NUMINAMATH_CALUDE_eliot_account_balance_l34_3410

theorem eliot_account_balance 
  (al_balance : ℝ) 
  (eliot_balance : ℝ) 
  (al_more : al_balance > eliot_balance)
  (difference_sum : al_balance - eliot_balance = (1 / 12) * (al_balance + eliot_balance))
  (increased_difference : 1.1 * al_balance = 1.2 * eliot_balance + 20) :
  eliot_balance = 200 := by
sorry

end NUMINAMATH_CALUDE_eliot_account_balance_l34_3410


namespace NUMINAMATH_CALUDE_complex_absolute_value_product_l34_3469

theorem complex_absolute_value_product : 
  ∃ (z w : ℂ), z = 3 * Real.sqrt 5 - 5 * I ∧ w = 2 * Real.sqrt 2 + 4 * I ∧ 
  Complex.abs (z * w) = 8 * Real.sqrt 105 := by
  sorry

end NUMINAMATH_CALUDE_complex_absolute_value_product_l34_3469


namespace NUMINAMATH_CALUDE_first_week_rate_correct_l34_3438

/-- The daily rate for the first week in a student youth hostel. -/
def first_week_rate : ℝ := 18

/-- The daily rate for days after the first week. -/
def additional_week_rate : ℝ := 12

/-- The total number of days stayed. -/
def total_days : ℕ := 23

/-- The total cost for the stay. -/
def total_cost : ℝ := 318

/-- Theorem stating that the first week rate is correct given the conditions. -/
theorem first_week_rate_correct :
  first_week_rate * 7 + additional_week_rate * (total_days - 7) = total_cost :=
by sorry

end NUMINAMATH_CALUDE_first_week_rate_correct_l34_3438


namespace NUMINAMATH_CALUDE_inequality_proof_l34_3408

theorem inequality_proof (a b : ℝ) (n : ℕ) (x₁ y₁ x₂ y₂ A : ℝ) :
  a > 0 →
  b > 0 →
  n > 1 →
  x₁ > 0 →
  y₁ > 0 →
  x₂ > 0 →
  y₂ > 0 →
  x₁^n - a*y₁^n = b →
  x₂^n - a*y₂^n = b →
  y₁ < y₂ →
  A = (1/2) * |x₁*y₂ - x₂*y₁| →
  b*y₂ > 2*n*y₁^(n-1)*a^(1-1/n)*A :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l34_3408


namespace NUMINAMATH_CALUDE_combined_tower_height_l34_3409

/-- The combined height of four towers given specific conditions -/
theorem combined_tower_height :
  ∀ (clyde grace sarah linda : ℝ),
  grace = 8 * clyde →
  grace = 40.5 →
  sarah = 2 * clyde →
  linda = (clyde + grace + sarah) / 3 →
  clyde + grace + sarah + linda = 74.25 := by
  sorry

end NUMINAMATH_CALUDE_combined_tower_height_l34_3409


namespace NUMINAMATH_CALUDE_only_happiness_symmetrical_l34_3421

-- Define a type for Chinese characters
inductive ChineseCharacter : Type
| happiness : ChineseCharacter  -- 喜
| longevity : ChineseCharacter  -- 寿
| blessing : ChineseCharacter   -- 福
| prosperity : ChineseCharacter -- 禄

-- Define symmetry for Chinese characters
def isSymmetrical (c : ChineseCharacter) : Prop :=
  match c with
  | ChineseCharacter.happiness => true
  | _ => false

-- Theorem statement
theorem only_happiness_symmetrical :
  ∀ c : ChineseCharacter, isSymmetrical c ↔ c = ChineseCharacter.happiness :=
by sorry

end NUMINAMATH_CALUDE_only_happiness_symmetrical_l34_3421


namespace NUMINAMATH_CALUDE_absolute_value_sum_inequality_l34_3442

theorem absolute_value_sum_inequality (b : ℝ) :
  (∃ x : ℝ, |x - 5| + |x - 10| > b) ↔ (0 < b ∧ b < 5) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_sum_inequality_l34_3442


namespace NUMINAMATH_CALUDE_bubble_sort_iterations_for_given_list_l34_3407

def bubble_sort_iterations (list : List Int) : Nat :=
  sorry

theorem bubble_sort_iterations_for_given_list :
  bubble_sort_iterations [6, -3, 0, 15] = 3 := by
  sorry

end NUMINAMATH_CALUDE_bubble_sort_iterations_for_given_list_l34_3407


namespace NUMINAMATH_CALUDE_fencing_cost_approx_l34_3496

-- Define the diameter of the circular field
def diameter : ℝ := 40

-- Define the cost per meter of fencing
def cost_per_meter : ℝ := 3

-- Define pi as a constant (approximation)
def π : ℝ := 3.14159

-- Define the function to calculate the circumference of a circle
def circumference (d : ℝ) : ℝ := π * d

-- Define the function to calculate the total cost of fencing
def total_cost (c : ℝ) (rate : ℝ) : ℝ := c * rate

-- Theorem stating that the total cost is approximately 377
theorem fencing_cost_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 
  abs (total_cost (circumference diameter) cost_per_meter - 377) < ε :=
sorry

end NUMINAMATH_CALUDE_fencing_cost_approx_l34_3496


namespace NUMINAMATH_CALUDE_phil_bought_cards_for_52_weeks_l34_3458

/-- Represents the number of weeks Phil bought baseball card packs --/
def weeks_buying_cards (cards_per_pack : ℕ) (cards_after_fire : ℕ) : ℕ :=
  (2 * cards_after_fire) / cards_per_pack

/-- Theorem stating that Phil bought cards for 52 weeks --/
theorem phil_bought_cards_for_52_weeks :
  weeks_buying_cards 20 520 = 52 := by
  sorry

end NUMINAMATH_CALUDE_phil_bought_cards_for_52_weeks_l34_3458


namespace NUMINAMATH_CALUDE_tea_mixture_price_l34_3436

/-- Given three varieties of tea mixed in a 1:1:2 ratio, with the first two varieties
    costing 126 and 135 rupees per kg respectively, and the mixture worth 152 rupees per kg,
    prove that the third variety costs 173.5 rupees per kg. -/
theorem tea_mixture_price (price1 price2 mixture_price : ℚ) 
    (h1 : price1 = 126)
    (h2 : price2 = 135)
    (h3 : mixture_price = 152) : ∃ price3 : ℚ,
  price3 = 173.5 ∧ 
  (price1 + price2 + 2 * price3) / 4 = mixture_price :=
by
  sorry

end NUMINAMATH_CALUDE_tea_mixture_price_l34_3436


namespace NUMINAMATH_CALUDE_complex_modulus_l34_3437

theorem complex_modulus (z : ℂ) : z - Complex.I = 1 + Complex.I → Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l34_3437


namespace NUMINAMATH_CALUDE_intersection_equals_two_l34_3462

-- Define the sets A and B
def A : Set ℝ := {x | x ≤ 2}
def B (a : ℝ) : Set ℝ := {x | x ≥ a}

-- State the theorem
theorem intersection_equals_two (a : ℝ) :
  A ∩ B a = {2} → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_equals_two_l34_3462


namespace NUMINAMATH_CALUDE_largest_multiple_of_8_under_100_l34_3411

theorem largest_multiple_of_8_under_100 : 
  ∃ n : ℕ, n * 8 = 96 ∧ 
    ∀ m : ℕ, m * 8 < 100 → m * 8 ≤ 96 := by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_8_under_100_l34_3411


namespace NUMINAMATH_CALUDE_division_remainder_l34_3400

theorem division_remainder : ∃ q : ℤ, 3021 = 97 * q + 14 ∧ 0 ≤ 14 ∧ 14 < 97 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_l34_3400


namespace NUMINAMATH_CALUDE_no_cyclic_quadratic_trinomial_l34_3475

/-- A quadratic trinomial is a polynomial of degree 2 -/
def QuadraticTrinomial (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), ∀ x, f x = a * x^2 + b * x + c

/-- The main theorem stating that no quadratic trinomial satisfies the cyclic property -/
theorem no_cyclic_quadratic_trinomial :
  ¬ ∃ (f : ℝ → ℝ) (a b c : ℝ),
    QuadraticTrinomial f ∧ 
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    f a = b ∧ f b = c ∧ f c = a :=
sorry

end NUMINAMATH_CALUDE_no_cyclic_quadratic_trinomial_l34_3475


namespace NUMINAMATH_CALUDE_decimal_comparisons_l34_3443

theorem decimal_comparisons :
  (9.38 > 3.98) ∧
  (0.62 > 0.23) ∧
  (2.5 > 2.05) ∧
  (53.6 > 5.36) ∧
  (9.42 > 9.377) := by
  sorry

end NUMINAMATH_CALUDE_decimal_comparisons_l34_3443


namespace NUMINAMATH_CALUDE_batsman_average_l34_3478

theorem batsman_average (total_innings : ℕ) (last_score : ℕ) (average_increase : ℝ) : 
  total_innings = 20 →
  last_score = 90 →
  average_increase = 2 →
  (↑total_innings * (average_after_last_innings - average_increase) + ↑last_score) / ↑total_innings = average_after_last_innings →
  average_after_last_innings = 52 :=
by
  sorry

#check batsman_average

end NUMINAMATH_CALUDE_batsman_average_l34_3478


namespace NUMINAMATH_CALUDE_second_most_frequent_is_23_l34_3449

-- Define the function m(i) which represents the number of drawings where i appears in the second position
def m (i : ℕ) : ℕ := 
  if 2 ≤ i ∧ i ≤ 87 then
    (i - 1) * (90 - i).choose 3
  else
    0

-- Define the lottery parameters
def lotterySize : ℕ := 6
def lotteryRange : ℕ := 90

-- Theorem statement
theorem second_most_frequent_is_23 : 
  ∀ i, 2 ≤ i ∧ i ≤ 87 → m i ≤ m 23 :=
sorry

end NUMINAMATH_CALUDE_second_most_frequent_is_23_l34_3449


namespace NUMINAMATH_CALUDE_tangent_line_at_one_l34_3495

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x + x^2

theorem tangent_line_at_one :
  ∃ (m b : ℝ), ∀ (x y : ℝ),
    (y = m * (x - 1) + f 1) ↔ (y = m * x + b ∧ 4 * x - y - 3 = 0) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_l34_3495


namespace NUMINAMATH_CALUDE_interior_edge_sum_is_eight_l34_3459

/-- Represents a rectangular picture frame -/
structure PictureFrame where
  outerWidth : ℝ
  outerHeight : ℝ
  borderWidth : ℝ

/-- Calculate the area of the frame -/
def frameArea (frame : PictureFrame) : ℝ :=
  frame.outerWidth * frame.outerHeight - (frame.outerWidth - 2 * frame.borderWidth) * (frame.outerHeight - 2 * frame.borderWidth)

/-- Calculate the sum of the interior edge lengths -/
def interiorEdgeSum (frame : PictureFrame) : ℝ :=
  2 * ((frame.outerWidth - 2 * frame.borderWidth) + (frame.outerHeight - 2 * frame.borderWidth))

/-- Theorem: The sum of interior edges is 8 inches for a frame with given properties -/
theorem interior_edge_sum_is_eight (frame : PictureFrame) 
  (h1 : frame.borderWidth = 2)
  (h2 : frameArea frame = 32)
  (h3 : frame.outerWidth = 7) : 
  interiorEdgeSum frame = 8 := by
  sorry


end NUMINAMATH_CALUDE_interior_edge_sum_is_eight_l34_3459


namespace NUMINAMATH_CALUDE_f_properties_l34_3453

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + Real.log (x + 1) - a * x - Real.cos x

theorem f_properties (a : ℝ) :
  (∀ x > -1, a ≤ 1 → Monotone (f a)) ∧
  (∃ a, deriv (f a) 0 = 0) := by sorry

end NUMINAMATH_CALUDE_f_properties_l34_3453


namespace NUMINAMATH_CALUDE_modular_arithmetic_proof_l34_3439

theorem modular_arithmetic_proof : (305 * 20 - 20 * 9 + 5) % 19 = 16 := by
  sorry

end NUMINAMATH_CALUDE_modular_arithmetic_proof_l34_3439


namespace NUMINAMATH_CALUDE_correct_factorization_l34_3483

theorem correct_factorization (x : ℝ) : 1 - 2*x + x^2 = (1 - x)^2 := by
  sorry

end NUMINAMATH_CALUDE_correct_factorization_l34_3483


namespace NUMINAMATH_CALUDE_smallest_k_with_remainder_one_l34_3486

theorem smallest_k_with_remainder_one : ∃! k : ℕ,
  k > 1 ∧
  k % 19 = 1 ∧
  k % 7 = 1 ∧
  k % 3 = 1 ∧
  ∀ m : ℕ, m > 1 ∧ m % 19 = 1 ∧ m % 7 = 1 ∧ m % 3 = 1 → k ≤ m :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_k_with_remainder_one_l34_3486


namespace NUMINAMATH_CALUDE_percentage_material_B_in_solution_Y_l34_3423

/-- Given two solutions X and Y, and their mixture, this theorem proves
    the percentage of material B in solution Y. -/
theorem percentage_material_B_in_solution_Y
  (percent_A_X : ℝ) (percent_B_X : ℝ) (percent_A_Y : ℝ)
  (percent_X_in_mixture : ℝ) (percent_A_in_mixture : ℝ)
  (h1 : percent_A_X = 0.20)
  (h2 : percent_B_X = 0.80)
  (h3 : percent_A_Y = 0.30)
  (h4 : percent_X_in_mixture = 0.80)
  (h5 : percent_A_in_mixture = 0.22)
  (h6 : percent_X_in_mixture * percent_A_X + (1 - percent_X_in_mixture) * percent_A_Y = percent_A_in_mixture) :
  1 - percent_A_Y = 0.70 := by
sorry

end NUMINAMATH_CALUDE_percentage_material_B_in_solution_Y_l34_3423


namespace NUMINAMATH_CALUDE_country_z_diploma_percentage_l34_3498

theorem country_z_diploma_percentage :
  let total_population : ℝ := 100
  let no_diploma_job_choice : ℝ := 18
  let diploma_no_job_choice_ratio : ℝ := 0.25
  let job_choice : ℝ := 40

  let diploma_job_choice : ℝ := job_choice - no_diploma_job_choice
  let no_job_choice : ℝ := total_population - job_choice
  let diploma_no_job_choice : ℝ := diploma_no_job_choice_ratio * no_job_choice

  diploma_job_choice + diploma_no_job_choice = 37 :=
by sorry

end NUMINAMATH_CALUDE_country_z_diploma_percentage_l34_3498


namespace NUMINAMATH_CALUDE_triangle_shape_l34_3484

theorem triangle_shape (a b c : ℝ) (A B C : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : 0 < A) (h5 : 0 < B) (h6 : 0 < C) 
  (h7 : A + B + C = Real.pi)
  (h8 : 2 * a * Real.cos B = c)
  (h9 : a * Real.sin B = b * Real.sin A)
  (h10 : b * Real.sin C = c * Real.sin B)
  (h11 : c * Real.sin A = a * Real.sin C) :
  A = B ∨ B = C ∨ A = C :=
sorry

end NUMINAMATH_CALUDE_triangle_shape_l34_3484


namespace NUMINAMATH_CALUDE_cone_volume_from_cylinder_l34_3476

/-- Given a cylinder with volume 72π cm³, a cone with the same height and twice the radius
    of the cylinder has a volume of 96π cm³. -/
theorem cone_volume_from_cylinder (r h : ℝ) : 
  r > 0 → h > 0 → π * r^2 * h = 72 * π → 
  (1/3 : ℝ) * π * (2*r)^2 * h = 96 * π := by
sorry

end NUMINAMATH_CALUDE_cone_volume_from_cylinder_l34_3476


namespace NUMINAMATH_CALUDE_vector_equality_l34_3490

def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (2, 4)

def a (x : ℝ) : ℝ × ℝ := (2*x - 1, x^2 + 3*x - 3)

theorem vector_equality (x : ℝ) : a x = (B.1 - A.1, B.2 - A.2) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_vector_equality_l34_3490


namespace NUMINAMATH_CALUDE_min_value_product_l34_3468

theorem min_value_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 4) :
  (3 * a + b) * (2 * b + 3 * c) * (a * c + 4) ≥ 384 ∧
  ∃ (a' b' c' : ℝ), a' > 0 ∧ b' > 0 ∧ c' > 0 ∧ a' * b' * c' = 4 ∧
    (3 * a' + b') * (2 * b' + 3 * c') * (a' * c' + 4) = 384 :=
by sorry

end NUMINAMATH_CALUDE_min_value_product_l34_3468


namespace NUMINAMATH_CALUDE_magnitude_of_complex_square_root_l34_3470

theorem magnitude_of_complex_square_root (w : ℂ) (h : w^2 = 48 - 14*I) : 
  Complex.abs w = 5 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_square_root_l34_3470


namespace NUMINAMATH_CALUDE_intersection_M_N_l34_3424

def M : Set ℤ := {1, 2, 3, 4, 5, 6}
def N : Set ℤ := {x | -2 < x ∧ x < 5}

theorem intersection_M_N : M ∩ N = {1, 2, 3, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l34_3424


namespace NUMINAMATH_CALUDE_reina_kevin_marble_ratio_l34_3444

/-- Proves that the ratio of Reina's marbles to Kevin's marbles is 4:1 -/
theorem reina_kevin_marble_ratio :
  let kevin_counters : ℕ := 40
  let kevin_marbles : ℕ := 50
  let reina_counters : ℕ := 3 * kevin_counters
  let reina_total : ℕ := 320
  let reina_marbles : ℕ := reina_total - reina_counters
  (reina_marbles : ℚ) / kevin_marbles = 4 / 1 := by
  sorry

end NUMINAMATH_CALUDE_reina_kevin_marble_ratio_l34_3444


namespace NUMINAMATH_CALUDE_fraction_equality_problem_l34_3416

theorem fraction_equality_problem (x y : ℚ) :
  x / y = 12 / 5 → y = 25 → x = 60 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_problem_l34_3416


namespace NUMINAMATH_CALUDE_fraction_sum_squared_l34_3404

theorem fraction_sum_squared (a b c x y z : ℝ) 
  (h1 : x / a + y / b + z / c = 4)
  (h2 : a / x + b / y + c / z = 0)
  : x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_squared_l34_3404


namespace NUMINAMATH_CALUDE_function_inequality_l34_3413

theorem function_inequality (f : ℝ → ℝ) (h_diff : Differentiable ℝ f) 
  (h_ineq : ∀ x : ℝ, f x > deriv f x) : 
  (Real.exp 2016 * f (-2016) > f 0) ∧ (f 2016 < Real.exp 2016 * f 0) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l34_3413


namespace NUMINAMATH_CALUDE_trajectory_of_midpoint_M_l34_3427

-- Define the circle C
def circle_C (k : ℝ) (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - 4)^2 = k ∧ k > 0

-- Define the intersection points
def intersection_points (k : ℝ) : Prop :=
  ∃ (yA yB xE xF : ℝ),
    circle_C k 0 yA ∧ circle_C k 0 yB ∧ yA > yB ∧
    circle_C k xE 0 ∧ circle_C k xF 0 ∧ xE > xF

-- Define the midpoint M of AE
def midpoint_M (x y yA xE : ℝ) : Prop :=
  x = (0 + xE) / 2 ∧ y = (yA + 0) / 2

-- Theorem statement
theorem trajectory_of_midpoint_M
  (k : ℝ) (x y : ℝ) :
  circle_C k x y →
  intersection_points k →
  (∃ (yA xE : ℝ), midpoint_M x y yA xE) →
  x > 1 →
  y > 2 + Real.sqrt 3 →
  (y - 2)^2 - (x - 1)^2 = 3 :=
sorry

end NUMINAMATH_CALUDE_trajectory_of_midpoint_M_l34_3427


namespace NUMINAMATH_CALUDE_cylinder_height_in_hemisphere_l34_3456

theorem cylinder_height_in_hemisphere (r c h : ℝ) : 
  r > 0 → c > 0 → h > 0 →
  r = 7 → c = 3 →
  h^2 + c^2 = r^2 →
  h = 2 * Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_cylinder_height_in_hemisphere_l34_3456


namespace NUMINAMATH_CALUDE_wrong_number_calculation_l34_3487

theorem wrong_number_calculation (n : Nat) (initial_avg correct_avg correct_num : ℝ) 
  (h1 : n = 10)
  (h2 : initial_avg = 21)
  (h3 : correct_avg = 22)
  (h4 : correct_num = 36) :
  ∃ wrong_num : ℝ,
    n * correct_avg - n * initial_avg = correct_num - wrong_num ∧
    wrong_num = 26 := by
  sorry

end NUMINAMATH_CALUDE_wrong_number_calculation_l34_3487


namespace NUMINAMATH_CALUDE_election_win_margin_l34_3429

theorem election_win_margin :
  ∀ (total_votes : ℕ) (winner_votes loser_votes : ℕ),
    winner_votes = (52 : ℕ) * total_votes / 100 →
    winner_votes = 3744 →
    loser_votes = total_votes - winner_votes →
    winner_votes - loser_votes = 288 :=
by
  sorry

end NUMINAMATH_CALUDE_election_win_margin_l34_3429


namespace NUMINAMATH_CALUDE_average_book_width_l34_3446

theorem average_book_width :
  let book_widths : List ℝ := [3, 0.5, 1.5, 4, 2, 5, 8]
  let sum_widths : ℝ := book_widths.sum
  let num_books : ℕ := book_widths.length
  let average_width : ℝ := sum_widths / num_books
  average_width = 3.43 := by sorry

end NUMINAMATH_CALUDE_average_book_width_l34_3446


namespace NUMINAMATH_CALUDE_consecutive_odd_integers_sum_l34_3426

theorem consecutive_odd_integers_sum (x y : ℤ) : 
  (Odd x ∧ Odd y) →  -- x and y are odd
  y = x + 4 →        -- y is the next consecutive odd integer after x
  y = 5 * x →        -- y is five times x
  x + y = 6 :=       -- their sum is 6
by
  sorry

end NUMINAMATH_CALUDE_consecutive_odd_integers_sum_l34_3426


namespace NUMINAMATH_CALUDE_sally_picked_42_peaches_l34_3494

/-- The number of peaches Sally picked -/
def peaches_picked (initial : ℕ) (final : ℕ) : ℕ := final - initial

/-- Theorem: Sally picked 42 peaches -/
theorem sally_picked_42_peaches : peaches_picked 13 55 = 42 := by
  sorry

end NUMINAMATH_CALUDE_sally_picked_42_peaches_l34_3494


namespace NUMINAMATH_CALUDE_ingrid_income_calculation_l34_3466

def john_income : ℝ := 57000
def john_tax_rate : ℝ := 0.30
def ingrid_tax_rate : ℝ := 0.40
def combined_tax_rate : ℝ := 0.35581395348837205

theorem ingrid_income_calculation (ingrid_income : ℝ) : 
  (john_tax_rate * john_income + ingrid_tax_rate * ingrid_income) / (john_income + ingrid_income) = combined_tax_rate →
  ingrid_income = 72000 := by
sorry

end NUMINAMATH_CALUDE_ingrid_income_calculation_l34_3466


namespace NUMINAMATH_CALUDE_sum_of_ten_and_hundredth_l34_3445

theorem sum_of_ten_and_hundredth : 10 + 0.01 = 10.01 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ten_and_hundredth_l34_3445


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l34_3473

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 3 ∧ b = 4 ∧ c^2 = a^2 + b^2 ∨ (a = 3 ∧ b = 4 ∧ c = b) → c = 5 ∨ c = 4 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l34_3473


namespace NUMINAMATH_CALUDE_circle_equation_l34_3440

/-- Given a circle C with center (a, 0) tangent to the line y = (√3/3)x at point N(3, √3),
    prove that the equation of circle C is (x-4)² + y² = 4 -/
theorem circle_equation (a : ℝ) :
  let C : Set (ℝ × ℝ) := {p | (p.1 - a)^2 + p.2^2 = ((3 - a)^2 + 3)}
  let l : Set (ℝ × ℝ) := {p | p.2 = (Real.sqrt 3 / 3) * p.1}
  let N : ℝ × ℝ := (3, Real.sqrt 3)
  (N ∈ C) ∧ (N ∈ l) ∧ (∀ p ∈ C, p ≠ N → p ∉ l) →
  C = {p : ℝ × ℝ | (p.1 - 4)^2 + p.2^2 = 4} :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l34_3440


namespace NUMINAMATH_CALUDE_mimi_picked_24_shells_l34_3431

/-- The number of seashells Mimi picked up -/
def mimi_shells : ℕ := 24

/-- The number of seashells Kyle found -/
def kyle_shells : ℕ := 2 * mimi_shells

/-- The number of seashells Leigh grabbed -/
def leigh_shells : ℕ := 16

/-- Theorem stating that Mimi picked up 24 seashells -/
theorem mimi_picked_24_shells : mimi_shells = 24 :=
by
  have h1 : kyle_shells = 2 * mimi_shells := by rfl
  have h2 : leigh_shells = kyle_shells / 3 := by sorry
  have h3 : leigh_shells = 16 := by rfl
  sorry


end NUMINAMATH_CALUDE_mimi_picked_24_shells_l34_3431


namespace NUMINAMATH_CALUDE_translated_minimum_point_l34_3414

-- Define the original function
def f (x : ℝ) : ℝ := |x + 1| - 2

-- Define the translated function
def g (x : ℝ) : ℝ := f (x - 3) + 4

-- State the theorem
theorem translated_minimum_point :
  ∃ (x_min : ℝ), (∀ (x : ℝ), g x_min ≤ g x) ∧ x_min = 2 ∧ g x_min = 2 :=
sorry

end NUMINAMATH_CALUDE_translated_minimum_point_l34_3414


namespace NUMINAMATH_CALUDE_sqrt_eight_minus_sqrt_two_equals_sqrt_two_l34_3433

theorem sqrt_eight_minus_sqrt_two_equals_sqrt_two : 
  Real.sqrt 8 - Real.sqrt 2 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_eight_minus_sqrt_two_equals_sqrt_two_l34_3433


namespace NUMINAMATH_CALUDE_factorial_300_trailing_zeros_l34_3441

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ := sorry

/-- Theorem: The number of trailing zeros in 300! is 74 -/
theorem factorial_300_trailing_zeros :
  trailingZeros 300 = 74 := by sorry

end NUMINAMATH_CALUDE_factorial_300_trailing_zeros_l34_3441
