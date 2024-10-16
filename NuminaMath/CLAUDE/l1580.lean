import Mathlib

namespace NUMINAMATH_CALUDE_equation_solution_product_l1580_158055

theorem equation_solution_product : ∃ (r s : ℝ), 
  r ≠ s ∧ 
  (r - 3) * (3 * r + 6) = r^2 - 16 * r + 63 ∧
  (s - 3) * (3 * s + 6) = s^2 - 16 * s + 63 ∧
  (r + 2) * (s + 2) = -19.14 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_product_l1580_158055


namespace NUMINAMATH_CALUDE_sine_monotonicity_l1580_158011

open Real

theorem sine_monotonicity (k : ℤ) :
  let f : ℝ → ℝ := λ x => sin (2 * x + (5 * π) / 6)
  let interval := Set.Icc (k * π + π / 3) (k * π + 5 * π / 6)
  (∀ x, f x ≥ f (π / 3)) →
  StrictMono (interval.restrict f) :=
by sorry

end NUMINAMATH_CALUDE_sine_monotonicity_l1580_158011


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l1580_158084

theorem smallest_n_congruence : ∃ (n : ℕ), n > 0 ∧ 
  (634 * n ≡ 1275 * n [ZMOD 30]) ∧ 
  (∀ (m : ℕ), m > 0 → (634 * m ≡ 1275 * m [ZMOD 30]) → n ≤ m) ∧ 
  n = 30 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l1580_158084


namespace NUMINAMATH_CALUDE_pears_for_20_apples_is_13_l1580_158038

/-- The price of fruits in an arbitrary unit -/
structure FruitPrices where
  apple : ℚ
  orange : ℚ
  pear : ℚ

/-- Given the conditions of the problem, calculate the number of pears
    that can be bought for the price of 20 apples -/
def pears_for_20_apples (prices : FruitPrices) : ℕ :=
  sorry

/-- Theorem stating the result of the calculation -/
theorem pears_for_20_apples_is_13 (prices : FruitPrices) 
  (h1 : 10 * prices.apple = 5 * prices.orange)
  (h2 : 3 * prices.orange = 4 * prices.pear) :
  pears_for_20_apples prices = 13 := by
  sorry

end NUMINAMATH_CALUDE_pears_for_20_apples_is_13_l1580_158038


namespace NUMINAMATH_CALUDE_min_value_theorem_l1580_158092

theorem min_value_theorem (a b c : ℝ) (h : a + 2*b + 3*c = 2) :
  (∀ x y z : ℝ, x + 2*y + 3*z = 2 → a^2 + 2*b^2 + 3*c^2 ≤ x^2 + 2*y^2 + 3*z^2) →
  2*a + 4*b + 9*c = 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1580_158092


namespace NUMINAMATH_CALUDE_lcm_36_48_75_l1580_158047

theorem lcm_36_48_75 : Nat.lcm (Nat.lcm 36 48) 75 = 3600 := by
  sorry

end NUMINAMATH_CALUDE_lcm_36_48_75_l1580_158047


namespace NUMINAMATH_CALUDE_ellipse_k_value_l1580_158051

/-- The equation of an ellipse with parameter k -/
def ellipse_equation (k x y : ℝ) : Prop :=
  2 * k * x^2 + k * y^2 = 1

/-- The focus of the ellipse -/
def focus : ℝ × ℝ := (0, -4)

/-- Theorem stating that for an ellipse with the given equation and focus, k = 1/32 -/
theorem ellipse_k_value :
  ∃ (k : ℝ), k ≠ 0 ∧
  (∀ x y : ℝ, ellipse_equation k x y ↔ 2 * k * x^2 + k * y^2 = 1) ∧
  (∃ x y : ℝ, ellipse_equation k x y ∧ (x, y) = focus) ∧
  k = 1/32 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_k_value_l1580_158051


namespace NUMINAMATH_CALUDE_minimum_translation_l1580_158021

noncomputable def f (a : ℝ) (x : ℝ) := a * Real.sin x + Real.cos x

theorem minimum_translation (a : ℝ) :
  (∀ x, f a (x - π/4) = f a (π/4 + (π/4 - x))) →
  ∃ φ : ℝ, φ > 0 ∧
    (∀ x, f a (x - φ) = f a (-x)) ∧
    (∀ ψ, ψ > 0 ∧ (∀ x, f a (x - ψ) = f a (-x)) → φ ≤ ψ) ∧
    φ = 3*π/4 :=
sorry

end NUMINAMATH_CALUDE_minimum_translation_l1580_158021


namespace NUMINAMATH_CALUDE_complex_equation_sum_l1580_158009

theorem complex_equation_sum (a b : ℝ) (i : ℂ) (h : i * i = -1) :
  (1 - 2*i) * (2 + a*i) = b - 2*i → a + b = 8 :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l1580_158009


namespace NUMINAMATH_CALUDE_pyramid_base_side_length_l1580_158090

theorem pyramid_base_side_length (area : ℝ) (slant_height : ℝ) (h1 : area = 120) (h2 : slant_height = 40) :
  ∃ (side_length : ℝ), side_length = 6 ∧ (1/2) * side_length * slant_height = area :=
sorry

end NUMINAMATH_CALUDE_pyramid_base_side_length_l1580_158090


namespace NUMINAMATH_CALUDE_expand_polynomial_l1580_158069

theorem expand_polynomial (x : ℝ) : (5*x^2 + 7*x + 2) * 3*x = 15*x^3 + 21*x^2 + 6*x := by
  sorry

end NUMINAMATH_CALUDE_expand_polynomial_l1580_158069


namespace NUMINAMATH_CALUDE_inequality_not_hold_l1580_158094

theorem inequality_not_hold (a b : ℝ) (h1 : a < b) (h2 : b < 0) :
  ¬(1 / (a - b) > 1 / a) := by
sorry

end NUMINAMATH_CALUDE_inequality_not_hold_l1580_158094


namespace NUMINAMATH_CALUDE_jumble_words_count_l1580_158007

/-- The number of letters in the Jumble alphabet -/
def alphabet_size : ℕ := 21

/-- The maximum word length in the Jumble language -/
def max_word_length : ℕ := 5

/-- The number of words of length n in the Jumble language that contain at least one 'A' -/
def words_with_a (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else alphabet_size^n - (alphabet_size - 1)^n

/-- The total number of words in the Jumble language -/
def total_words : ℕ :=
  (List.range max_word_length).map (λ i => words_with_a (i + 1)) |>.sum

theorem jumble_words_count :
  total_words = 920885 := by sorry

end NUMINAMATH_CALUDE_jumble_words_count_l1580_158007


namespace NUMINAMATH_CALUDE_polar_to_cartesian_l1580_158089

theorem polar_to_cartesian :
  let ρ : ℝ := 4
  let θ : ℝ := 2 * π / 3
  let x : ℝ := ρ * Real.cos θ
  let y : ℝ := ρ * Real.sin θ
  x = -2 ∧ y = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_l1580_158089


namespace NUMINAMATH_CALUDE_raines_change_l1580_158008

/-- Calculates the change Raine receives after purchasing items from a gift shop --/
theorem raines_change (bracelet_price necklace_price mug_price : ℕ)
  (bracelet_count necklace_count mug_count : ℕ)
  (payment : ℕ)
  (h1 : bracelet_price = 15)
  (h2 : necklace_price = 10)
  (h3 : mug_price = 20)
  (h4 : bracelet_count = 3)
  (h5 : necklace_count = 2)
  (h6 : mug_count = 1)
  (h7 : payment = 100) :
  payment - (bracelet_price * bracelet_count + necklace_price * necklace_count + mug_price * mug_count) = 15 := by
  sorry

#check raines_change

end NUMINAMATH_CALUDE_raines_change_l1580_158008


namespace NUMINAMATH_CALUDE_max_consecutive_semiprimes_l1580_158070

def IsPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def IsSemiPrime (n : ℕ) : Prop :=
  n > 25 ∧ ∃ p q : ℕ, IsPrime p ∧ IsPrime q ∧ p ≠ q ∧ n = p + q

def ConsecutiveSemiPrimes (n : ℕ) : Prop :=
  ∀ k : ℕ, k < n → IsSemiPrime (k + 1)

theorem max_consecutive_semiprimes :
  ∀ n : ℕ, ConsecutiveSemiPrimes n → n ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_max_consecutive_semiprimes_l1580_158070


namespace NUMINAMATH_CALUDE_A_necessary_not_sufficient_l1580_158065

-- Define propositions A and B
def prop_A (x y : ℝ) : Prop := x ≠ 2 ∨ y ≠ 3
def prop_B (x y : ℝ) : Prop := x + y ≠ 5

-- Theorem stating that A is necessary but not sufficient for B
theorem A_necessary_not_sufficient :
  (∃ x y : ℝ, prop_A x y ∧ ¬prop_B x y) ∧
  (∀ x y : ℝ, prop_B x y → prop_A x y) :=
sorry

end NUMINAMATH_CALUDE_A_necessary_not_sufficient_l1580_158065


namespace NUMINAMATH_CALUDE_benny_baseball_gear_expense_l1580_158088

/-- The amount Benny spent on baseball gear -/
def amount_spent (initial_amount remaining_amount : ℕ) : ℕ :=
  initial_amount - remaining_amount

/-- Theorem: Benny spent $47 on baseball gear -/
theorem benny_baseball_gear_expense :
  amount_spent 79 32 = 47 := by
  sorry

end NUMINAMATH_CALUDE_benny_baseball_gear_expense_l1580_158088


namespace NUMINAMATH_CALUDE_x_value_theorem_l1580_158080

theorem x_value_theorem (x n : ℕ) (h1 : x = 2^n - 32) 
  (h2 : (Nat.factors x).card = 3) 
  (h3 : 3 ∈ Nat.factors x) : 
  x = 480 ∨ x = 2016 := by
sorry

end NUMINAMATH_CALUDE_x_value_theorem_l1580_158080


namespace NUMINAMATH_CALUDE_doubled_to_original_ratio_l1580_158073

theorem doubled_to_original_ratio (x : ℝ) (h : 3 * (2 * x + 5) = 135) : 
  (2 * x) / x = 2 := by
  sorry

end NUMINAMATH_CALUDE_doubled_to_original_ratio_l1580_158073


namespace NUMINAMATH_CALUDE_train_speed_calculation_l1580_158044

/-- The speed of a train given the lengths of two trains, the speed of one train, and the time they take to cross each other when moving in opposite directions. -/
theorem train_speed_calculation (length_train1 length_train2 speed_train2 crossing_time : ℝ) 
  (h1 : length_train1 = 150)
  (h2 : length_train2 = 350.04)
  (h3 : speed_train2 = 80)
  (h4 : crossing_time = 9)
  : ∃ (speed_train1 : ℝ), abs (speed_train1 - 120.016) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l1580_158044


namespace NUMINAMATH_CALUDE_may_day_travel_scientific_notation_l1580_158030

/-- Expresses a number in scientific notation -/
def scientific_notation (n : ℝ) : ℝ × ℤ :=
  sorry

theorem may_day_travel_scientific_notation :
  scientific_notation (56.99 * 1000000) = (5.699, 7) :=
sorry

end NUMINAMATH_CALUDE_may_day_travel_scientific_notation_l1580_158030


namespace NUMINAMATH_CALUDE_frisbee_game_probability_l1580_158017

/-- The probability that Alice has the frisbee after three turns in the frisbee game. -/
theorem frisbee_game_probability : 
  let alice_toss_prob : ℚ := 2/3
  let alice_keep_prob : ℚ := 1/3
  let bob_toss_prob : ℚ := 1/4
  let bob_keep_prob : ℚ := 3/4
  let alice_has_frisbee_after_three_turns : ℚ := 
    alice_toss_prob * bob_keep_prob * bob_keep_prob +
    alice_keep_prob * alice_keep_prob
  alice_has_frisbee_after_three_turns = 35/72 :=
by sorry

end NUMINAMATH_CALUDE_frisbee_game_probability_l1580_158017


namespace NUMINAMATH_CALUDE_average_and_variance_after_adding_datapoint_l1580_158025

def initial_average : ℝ := 4
def initial_variance : ℝ := 2
def initial_count : ℕ := 7
def new_datapoint : ℝ := 4
def new_count : ℕ := initial_count + 1

def new_average (x : ℝ) : Prop :=
  x = (initial_count * initial_average + new_datapoint) / new_count

def new_variance (s : ℝ) : Prop :=
  s = (initial_count * initial_variance + (new_datapoint - initial_average)^2) / new_count

theorem average_and_variance_after_adding_datapoint :
  ∃ (x s : ℝ), new_average x ∧ new_variance s ∧ x = initial_average ∧ s < initial_variance :=
sorry

end NUMINAMATH_CALUDE_average_and_variance_after_adding_datapoint_l1580_158025


namespace NUMINAMATH_CALUDE_lucy_total_cost_l1580_158003

/-- The total cost Lucy paid for a lamp and a table, given specific pricing conditions. -/
theorem lucy_total_cost : 
  ∀ (lamp_original_price lamp_discounted_price table_price : ℝ),
  lamp_discounted_price = 20 →
  lamp_discounted_price = (1/5) * (0.6 * lamp_original_price) →
  table_price = 2 * lamp_original_price →
  lamp_discounted_price + table_price = 353.34 := by
  sorry

#check lucy_total_cost

end NUMINAMATH_CALUDE_lucy_total_cost_l1580_158003


namespace NUMINAMATH_CALUDE_new_year_firework_boxes_l1580_158020

/-- Calculates the number of firework boxes used in a New Year's Eve display. -/
def firework_boxes_used (total_fireworks : ℕ) (fireworks_per_digit : ℕ) (fireworks_per_letter : ℕ) (fireworks_per_box : ℕ) (year_digits : ℕ) (phrase_letters : ℕ) : ℕ :=
  let year_fireworks := fireworks_per_digit * year_digits
  let phrase_fireworks := fireworks_per_letter * phrase_letters
  let remaining_fireworks := total_fireworks - (year_fireworks + phrase_fireworks)
  remaining_fireworks / fireworks_per_box

/-- The number of firework boxes used in the New Year's Eve display is 50. -/
theorem new_year_firework_boxes :
  firework_boxes_used 484 6 5 8 4 12 = 50 := by
  sorry

end NUMINAMATH_CALUDE_new_year_firework_boxes_l1580_158020


namespace NUMINAMATH_CALUDE_piggy_bank_dimes_l1580_158012

/-- Represents the contents of a piggy bank --/
structure PiggyBank where
  quarters : ℕ
  dimes : ℕ
  total_value : ℚ
  total_coins : ℕ

/-- The value of a quarter in dollars --/
def quarter_value : ℚ := 25 / 100

/-- The value of a dime in dollars --/
def dime_value : ℚ := 10 / 100

theorem piggy_bank_dimes (pb : PiggyBank) 
  (h1 : pb.total_value = 1975 / 100)
  (h2 : pb.total_coins = 100)
  (h3 : pb.total_value = quarter_value * pb.quarters + dime_value * pb.dimes)
  (h4 : pb.total_coins = pb.quarters + pb.dimes) :
  pb.dimes = 35 := by
sorry

end NUMINAMATH_CALUDE_piggy_bank_dimes_l1580_158012


namespace NUMINAMATH_CALUDE_function_decomposition_l1580_158041

theorem function_decomposition (f : ℝ → ℝ) : 
  ∃ (g h : ℝ → ℝ), 
    (∀ x, g (-x) = g x) ∧ 
    (∀ x, h (-x) = -h x) ∧ 
    (∀ x, f x = g x + h x) := by
  sorry

end NUMINAMATH_CALUDE_function_decomposition_l1580_158041


namespace NUMINAMATH_CALUDE_mixed_number_calculation_l1580_158072

theorem mixed_number_calculation :
  let a := 5 + 1 / 2
  let b := 2 + 2 / 3
  let c := 1 + 1 / 5
  let d := 3 + 1 / 4
  (a - b) / (c + d) = 170 / 267 :=
by sorry

end NUMINAMATH_CALUDE_mixed_number_calculation_l1580_158072


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l1580_158043

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  d_ne_zero : d ≠ 0
  is_arithmetic : ∀ n, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_first_term
  (seq : ArithmeticSequence)
  (h1 : seq.a 2 * seq.a 3 = seq.a 4 * seq.a 5)
  (h2 : sum_n seq 9 = 1) :
  seq.a 1 = -5/27 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l1580_158043


namespace NUMINAMATH_CALUDE_at_least_one_third_l1580_158076

theorem at_least_one_third (a b c : ℝ) (h : a + b + c = 1) :
  (a ≥ 1/3) ∨ (b ≥ 1/3) ∨ (c ≥ 1/3) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_third_l1580_158076


namespace NUMINAMATH_CALUDE_logarithm_simplification_l1580_158083

theorem logarithm_simplification 
  (m n p q x z : ℝ) 
  (hm : m > 0) (hn : n > 0) (hp : p > 0) (hq : q > 0) (hx : x > 0) (hz : z > 0) : 
  Real.log (m / n) + Real.log (n / p) + Real.log (p / q) - Real.log (m * x / (q * z)) = Real.log (z / x) := by
  sorry

end NUMINAMATH_CALUDE_logarithm_simplification_l1580_158083


namespace NUMINAMATH_CALUDE_evaluate_expression_l1580_158042

theorem evaluate_expression : (-2 : ℤ) ^ (3^2) + 2 ^ (3^2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1580_158042


namespace NUMINAMATH_CALUDE_farm_animals_l1580_158098

theorem farm_animals (total_legs : ℕ) (sheep_count : ℕ) : 
  total_legs = 60 ∧ sheep_count = 10 → 
  ∃ (chicken_count : ℕ), 
    chicken_count * 2 + sheep_count * 4 = total_legs ∧
    chicken_count + sheep_count = 20 :=
by sorry

end NUMINAMATH_CALUDE_farm_animals_l1580_158098


namespace NUMINAMATH_CALUDE_max_girls_in_ballet_l1580_158086

/-- Represents the number of boys participating in the ballet -/
def num_boys : ℕ := 5

/-- Represents the distance requirement between girls and boys -/
def distance : ℕ := 5

/-- Represents the number of boys required at the specified distance from each girl -/
def boys_per_girl : ℕ := 2

/-- Calculates the maximum number of girls that can participate in the ballet -/
def max_girls : ℕ := (num_boys.choose boys_per_girl) * 2

/-- Theorem stating the maximum number of girls that can participate in the ballet -/
theorem max_girls_in_ballet : max_girls = 20 := by
  sorry

end NUMINAMATH_CALUDE_max_girls_in_ballet_l1580_158086


namespace NUMINAMATH_CALUDE_fence_cost_per_foot_l1580_158016

theorem fence_cost_per_foot 
  (plot_area : ℝ) 
  (total_cost : ℝ) 
  (h1 : plot_area = 289) 
  (h2 : total_cost = 3808) : 
  total_cost / (4 * Real.sqrt plot_area) = 56 := by
sorry

end NUMINAMATH_CALUDE_fence_cost_per_foot_l1580_158016


namespace NUMINAMATH_CALUDE_storks_equal_other_birds_l1580_158081

/-- Represents the count of different bird species on a fence --/
structure BirdCounts where
  sparrows : ℕ
  crows : ℕ
  storks : ℕ
  egrets : ℕ

/-- Calculates the final bird counts after all arrivals and departures --/
def finalBirdCounts (initial : BirdCounts) 
  (firstArrival : BirdCounts) 
  (firstDeparture : BirdCounts)
  (secondArrival : BirdCounts) : BirdCounts :=
  { sparrows := initial.sparrows + firstArrival.sparrows - firstDeparture.sparrows,
    crows := initial.crows + firstArrival.crows + secondArrival.crows,
    storks := initial.storks + firstArrival.storks + secondArrival.storks,
    egrets := firstArrival.egrets - firstDeparture.egrets }

/-- The main theorem stating that the number of storks equals the sum of all other birds --/
theorem storks_equal_other_birds : 
  let initial := BirdCounts.mk 2 1 3 0
  let firstArrival := BirdCounts.mk 1 3 6 4
  let firstDeparture := BirdCounts.mk 2 0 0 1
  let secondArrival := BirdCounts.mk 0 4 3 0
  let final := finalBirdCounts initial firstArrival firstDeparture secondArrival
  final.storks = final.sparrows + final.crows + final.egrets := by
  sorry

end NUMINAMATH_CALUDE_storks_equal_other_birds_l1580_158081


namespace NUMINAMATH_CALUDE_exists_max_in_finite_list_l1580_158082

theorem exists_max_in_finite_list : 
  ∀ (L : List ℝ), L.length = 1000 → ∃ (m : ℝ), ∀ (x : ℝ), x ∈ L → x ≤ m :=
by sorry

end NUMINAMATH_CALUDE_exists_max_in_finite_list_l1580_158082


namespace NUMINAMATH_CALUDE_smallest_divisible_by_8_11_15_l1580_158068

theorem smallest_divisible_by_8_11_15 : ∃! n : ℕ+, 
  (∀ m : ℕ+, m < n → ¬(8 ∣ m ∧ 11 ∣ m ∧ 15 ∣ m)) ∧ 
  (8 ∣ n) ∧ (11 ∣ n) ∧ (15 ∣ n) := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_8_11_15_l1580_158068


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1580_158026

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_sum : a 4 + a 6 + a 8 + a 10 = 80) :
  a 1 + a 13 = 40 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1580_158026


namespace NUMINAMATH_CALUDE_min_value_ab_l1580_158018

/-- Given b > 0 and two perpendicular lines, prove the minimum value of ab is 2 -/
theorem min_value_ab (b : ℝ) (a : ℝ) (h1 : b > 0) 
  (h2 : ∀ x y : ℝ, (b^2 + 1) * x + a * y + 2 = 0 ↔ x - b^2 * y - 1 = 0) : 
  (∀ a' b' : ℝ, b' > 0 ∧ (∀ x y : ℝ, (b'^2 + 1) * x + a' * y + 2 = 0 ↔ x - b'^2 * y - 1 = 0) → a' * b' ≥ 2) ∧ 
  (∃ a₀ b₀ : ℝ, b₀ > 0 ∧ (∀ x y : ℝ, (b₀^2 + 1) * x + a₀ * y + 2 = 0 ↔ x - b₀^2 * y - 1 = 0) ∧ a₀ * b₀ = 2) :=
sorry

end NUMINAMATH_CALUDE_min_value_ab_l1580_158018


namespace NUMINAMATH_CALUDE_circle_radius_sqrt_61_l1580_158049

/-- Given a circle with center on the x-axis passing through points (2,5) and (3,2),
    its radius is √61. -/
theorem circle_radius_sqrt_61 :
  ∀ x : ℝ,
  (∃ r : ℝ, r > 0 ∧
    r^2 = (x - 2)^2 + 5^2 ∧
    r^2 = (x - 3)^2 + 2^2) →
  ∃ r : ℝ, r > 0 ∧ r^2 = 61 :=
by sorry


end NUMINAMATH_CALUDE_circle_radius_sqrt_61_l1580_158049


namespace NUMINAMATH_CALUDE_common_difference_is_three_l1580_158002

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem common_difference_is_three
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 3 + a 11 = 24)
  (h_a4 : a 4 = 3) :
  ∃ d : ℝ, d = 3 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
sorry

end NUMINAMATH_CALUDE_common_difference_is_three_l1580_158002


namespace NUMINAMATH_CALUDE_physics_marks_l1580_158035

/-- Represents the marks obtained in each subject --/
structure Marks where
  physics : ℝ
  chemistry : ℝ
  mathematics : ℝ
  biology : ℝ
  computerScience : ℝ

/-- The conditions of the problem --/
def ProblemConditions (m : Marks) : Prop :=
  -- Average score across all subjects is 75
  (m.physics + m.chemistry + m.mathematics + m.biology + m.computerScience) / 5 = 75 ∧
  -- Average score in Physics, Mathematics, and Biology is 85
  (m.physics + m.mathematics + m.biology) / 3 = 85 ∧
  -- Average score in Physics, Chemistry, and Computer Science is 70
  (m.physics + m.chemistry + m.computerScience) / 3 = 70 ∧
  -- Weightages sum to 100%
  0.20 + 0.25 + 0.20 + 0.15 + 0.20 = 1

theorem physics_marks (m : Marks) (h : ProblemConditions m) : m.physics = 90 := by
  sorry

end NUMINAMATH_CALUDE_physics_marks_l1580_158035


namespace NUMINAMATH_CALUDE_no_base_with_final_digit_one_l1580_158071

theorem no_base_with_final_digit_one : 
  ∀ b : ℕ, 2 ≤ b ∧ b ≤ 9 → ¬(∃ k : ℕ, 360 = k * b + 1) := by
  sorry

end NUMINAMATH_CALUDE_no_base_with_final_digit_one_l1580_158071


namespace NUMINAMATH_CALUDE_roots_sum_theorem_l1580_158024

theorem roots_sum_theorem (p q : ℝ) : 
  p^2 - 5*p + 6 = 0 → q^2 - 5*q + 6 = 0 → p^3 + p^5*q + p*q^5 + q^3 = 617 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_theorem_l1580_158024


namespace NUMINAMATH_CALUDE_h1n1_vaccine_scientific_notation_l1580_158001

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation with a specified number of significant figures -/
def toScientificNotation (x : ℝ) (sigFigs : ℕ) : ScientificNotation :=
  sorry

/-- Checks if two ScientificNotation values are equal up to a certain number of significant figures -/
def equalUpToSigFigs (a b : ScientificNotation) (sigFigs : ℕ) : Prop :=
  sorry

theorem h1n1_vaccine_scientific_notation :
  equalUpToSigFigs (toScientificNotation (25.06 * 1000000) 3) 
                   { coefficient := 2.51, exponent := 7, is_valid := by sorry } 3 := by
  sorry

end NUMINAMATH_CALUDE_h1n1_vaccine_scientific_notation_l1580_158001


namespace NUMINAMATH_CALUDE_expression_value_l1580_158015

theorem expression_value (a b c : ℝ) (h : a * b + b * c + c * a = 3) :
  (a * (b^2 + 3)) / (a + b) + (b * (c^2 + 3)) / (b + c) + (c * (a^2 + 3)) / (c + a) = 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1580_158015


namespace NUMINAMATH_CALUDE_min_value_and_y_l1580_158046

theorem min_value_and_y (x y z : ℝ) (h : 2*x - 3*y + z = 3) :
  ∃ (min_val : ℝ), 
    (∀ x' y' z' : ℝ, 2*x' - 3*y' + z' = 3 → x'^2 + (y' - 1)^2 + z'^2 ≥ min_val) ∧
    (x^2 + (y - 1)^2 + z^2 = min_val ↔ y = -2/7) ∧
    min_val = 18/7 :=
sorry

end NUMINAMATH_CALUDE_min_value_and_y_l1580_158046


namespace NUMINAMATH_CALUDE_cuboid_properties_l1580_158033

/-- Represents a cuboid with length, width, and height -/
structure Cuboid where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the sum of all edge lengths of a cuboid -/
def sumEdgeLengths (c : Cuboid) : ℝ :=
  4 * (c.length + c.width + c.height)

/-- Calculates the surface area of a cuboid -/
def surfaceArea (c : Cuboid) : ℝ :=
  2 * (c.length * c.width + c.length * c.height + c.width * c.height)

/-- Calculates the volume of a cuboid -/
def volume (c : Cuboid) : ℝ :=
  c.length * c.width * c.height

/-- Theorem about a specific cuboid's properties -/
theorem cuboid_properties :
  ∃ c : Cuboid,
    c.length = 2 * c.width ∧
    c.width = c.height ∧
    sumEdgeLengths c = 48 ∧
    surfaceArea c = 90 ∧
    volume c = 54 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_properties_l1580_158033


namespace NUMINAMATH_CALUDE_square_difference_equality_l1580_158078

theorem square_difference_equality : (15 + 12)^2 - (12^2 + 15^2) = 360 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l1580_158078


namespace NUMINAMATH_CALUDE_equation_root_implies_a_value_l1580_158037

theorem equation_root_implies_a_value (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ (3 * x - 1) / (x - 3) = a / (3 - x) - 1) →
  a = -8 := by
  sorry

end NUMINAMATH_CALUDE_equation_root_implies_a_value_l1580_158037


namespace NUMINAMATH_CALUDE_parallel_lines_k_equals_3_l1580_158099

/-- Two lines are parallel if and only if they have the same slope -/
axiom parallel_lines_same_slope (m₁ m₂ : ℝ) : 
  (∃ b₁ b₂ : ℝ, ∀ x y : ℝ, (y = m₁ * x + b₁) ↔ (y = m₂ * x + b₂)) ↔ m₁ = m₂

/-- If the line y = kx - 1 is parallel to the line y = 3x, then k = 3 -/
theorem parallel_lines_k_equals_3 (k : ℝ) :
  (∃ x y : ℝ, y = k * x - 1 ∧ y = 3 * x) → k = 3 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_k_equals_3_l1580_158099


namespace NUMINAMATH_CALUDE_birds_after_week_l1580_158023

def initial_chickens : ℕ := 300
def initial_turkeys : ℕ := 200
def initial_guinea_fowls : ℕ := 80

def daily_loss_chickens : ℕ := 20
def daily_loss_turkeys : ℕ := 8
def daily_loss_guinea_fowls : ℕ := 5

def days_in_week : ℕ := 7

def remaining_birds : ℕ := 
  (initial_chickens - daily_loss_chickens * days_in_week) +
  (initial_turkeys - daily_loss_turkeys * days_in_week) +
  (initial_guinea_fowls - daily_loss_guinea_fowls * days_in_week)

theorem birds_after_week : remaining_birds = 349 := by
  sorry

end NUMINAMATH_CALUDE_birds_after_week_l1580_158023


namespace NUMINAMATH_CALUDE_classroom_size_theorem_l1580_158087

/-- Represents the number of students in a classroom -/
def classroom_size (boys : ℕ) (girls : ℕ) : ℕ := boys + girls

/-- Represents the ratio of boys to girls -/
def ratio_boys_girls (boys : ℕ) (girls : ℕ) : Prop := 3 * girls = 5 * boys

theorem classroom_size_theorem (boys girls : ℕ) :
  ratio_boys_girls boys girls →
  girls = boys + 4 →
  classroom_size boys girls = 16 := by
sorry

end NUMINAMATH_CALUDE_classroom_size_theorem_l1580_158087


namespace NUMINAMATH_CALUDE_identity_element_is_one_zero_l1580_158067

-- Define the operation ⊕
def oplus (a b c d : ℝ) : ℝ × ℝ := (a * c + b * d, a * d + b * c)

-- State the theorem
theorem identity_element_is_one_zero :
  (∀ a b : ℝ, oplus a b x y = (a, b)) → (x, y) = (1, 0) := by
  sorry

end NUMINAMATH_CALUDE_identity_element_is_one_zero_l1580_158067


namespace NUMINAMATH_CALUDE_square_ratio_proof_l1580_158048

theorem square_ratio_proof (area_ratio : Rat) (a b c : ℕ) : 
  area_ratio = 50 / 98 →
  (a : Rat) * Real.sqrt b / c = Real.sqrt (area_ratio) →
  (a : Rat) / c = 5 / 7 →
  a + b + c = 12 :=
by sorry

end NUMINAMATH_CALUDE_square_ratio_proof_l1580_158048


namespace NUMINAMATH_CALUDE_triangle_third_angle_l1580_158053

theorem triangle_third_angle (A B C : ℝ) (h : A + B = 90) : C = 90 :=
  by
  sorry

end NUMINAMATH_CALUDE_triangle_third_angle_l1580_158053


namespace NUMINAMATH_CALUDE_expression_simplification_l1580_158039

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 2 + 3) :
  (x^2 - 1) / (x^2 - 6*x + 9) * (1 - x / (x - 1)) / ((x + 1) / (x - 3)) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1580_158039


namespace NUMINAMATH_CALUDE_martha_coffee_spending_l1580_158052

/-- The cost of an iced coffee that satisfies Martha's coffee spending reduction --/
def iced_coffee_cost : ℚ := by sorry

/-- Proves that the cost of an iced coffee is $2.00 --/
theorem martha_coffee_spending :
  let latte_cost : ℚ := 4
  let lattes_per_week : ℕ := 5
  let iced_coffees_per_week : ℕ := 3
  let weeks_per_year : ℕ := 52
  let spending_reduction_ratio : ℚ := 1 / 4
  let spending_reduction_amount : ℚ := 338

  let annual_latte_spending : ℚ := latte_cost * lattes_per_week * weeks_per_year
  let annual_iced_coffee_spending : ℚ := iced_coffee_cost * iced_coffees_per_week * weeks_per_year
  let total_annual_spending : ℚ := annual_latte_spending + annual_iced_coffee_spending

  (1 - spending_reduction_ratio) * total_annual_spending = total_annual_spending - spending_reduction_amount →
  iced_coffee_cost = 2 := by sorry

end NUMINAMATH_CALUDE_martha_coffee_spending_l1580_158052


namespace NUMINAMATH_CALUDE_sequence_property_l1580_158013

theorem sequence_property (a : ℕ → ℝ) 
  (h : ∀ m : ℕ, m > 1 → a (m + 1) * a (m - 1) = a m ^ 2 - a 1 ^ 2) :
  ∀ m n : ℕ, m > n ∧ n > 1 → a (m + n) * a (m - n) = a m ^ 2 - a n ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_sequence_property_l1580_158013


namespace NUMINAMATH_CALUDE_rain_probability_l1580_158010

theorem rain_probability (p : ℚ) (h : p = 3/4) :
  1 - (1 - p)^4 = 255/256 := by sorry

end NUMINAMATH_CALUDE_rain_probability_l1580_158010


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_leg_length_l1580_158061

/-- An isosceles trapezoid circumscribed around a circle with area S and acute base angle π/6 has leg length √(2S) -/
theorem isosceles_trapezoid_leg_length (S : ℝ) (h_pos : S > 0) :
  ∃ (x : ℝ),
    x > 0 ∧
    x = Real.sqrt (2 * S) ∧
    ∃ (a b h : ℝ),
      a > 0 ∧ b > 0 ∧ h > 0 ∧
      a + b = 2 * x ∧
      h = x * Real.sin (π / 6) ∧
      S = (a + b) * h / 2 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_leg_length_l1580_158061


namespace NUMINAMATH_CALUDE_parabola_point_D_l1580_158029

/-- A parabola passing through three given points -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  point_A : a * (-0.8)^2 + b * (-0.8) + c = 4.132
  point_B : a * 1.2^2 + b * 1.2 + c = -1.948
  point_C : a * 2.8^2 + b * 2.8 + c = -3.932

/-- The y-coordinate of a point on the parabola given its x-coordinate -/
def y_coordinate (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

theorem parabola_point_D (p : Parabola) : y_coordinate p 1.8 = -2.992 := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_D_l1580_158029


namespace NUMINAMATH_CALUDE_circle_center_correct_l1580_158059

/-- The equation of a circle in the form ax² + bx + cy² + dy + e = 0 -/
structure CircleEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ

/-- The center of a circle -/
structure CircleCenter where
  x : ℝ
  y : ℝ

/-- Given a circle equation, return its center -/
def findCircleCenter (eq : CircleEquation) : CircleCenter :=
  sorry

theorem circle_center_correct :
  let eq := CircleEquation.mk 4 8 4 (-24) 16
  let center := findCircleCenter eq
  center.x = -1 ∧ center.y = 3 := by sorry

end NUMINAMATH_CALUDE_circle_center_correct_l1580_158059


namespace NUMINAMATH_CALUDE_inequality_proof_l1580_158091

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a + b + c = 1) : 
  (a * b + b * c + a * c ≤ 1 / 3) ∧ 
  (a^2 / b + b^2 / c + c^2 / a ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1580_158091


namespace NUMINAMATH_CALUDE_calculation_proof_l1580_158004

theorem calculation_proof : (((15 - 2 + 4) / 1) / 2) * 8 = 68 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1580_158004


namespace NUMINAMATH_CALUDE_average_after_removal_l1580_158085

theorem average_after_removal (numbers : Finset ℝ) (sum : ℝ) :
  Finset.card numbers = 12 →
  sum / 12 = 90 →
  sum = Finset.sum numbers id →
  68 ∈ numbers →
  75 ∈ numbers →
  82 ∈ numbers →
  (sum - 68 - 75 - 82) / 9 = 95 := by
sorry

end NUMINAMATH_CALUDE_average_after_removal_l1580_158085


namespace NUMINAMATH_CALUDE_acacia_arrangement_probability_l1580_158005

/-- The number of fir trees -/
def num_fir : ℕ := 4

/-- The number of pine trees -/
def num_pine : ℕ := 5

/-- The number of acacia trees -/
def num_acacia : ℕ := 6

/-- The total number of trees -/
def total_trees : ℕ := num_fir + num_pine + num_acacia

/-- The probability of no two acacia trees being next to each other -/
def prob_no_adjacent_acacia : ℚ := 84 / 159

theorem acacia_arrangement_probability :
  let total_arrangements := Nat.choose total_trees num_acacia
  let valid_arrangements := Nat.choose (num_fir + num_pine + 1) num_acacia * Nat.choose (num_fir + num_pine) num_fir
  (valid_arrangements : ℚ) / total_arrangements = prob_no_adjacent_acacia := by
  sorry

end NUMINAMATH_CALUDE_acacia_arrangement_probability_l1580_158005


namespace NUMINAMATH_CALUDE_mean_salary_proof_l1580_158066

def salaries : List ℝ := [1000, 2500, 3100, 3650, 1500, 2000]

theorem mean_salary_proof :
  (salaries.sum / salaries.length : ℝ) = 2458.33 := by
  sorry

end NUMINAMATH_CALUDE_mean_salary_proof_l1580_158066


namespace NUMINAMATH_CALUDE_square_perimeter_l1580_158075

theorem square_perimeter (s : ℝ) (h1 : s > 0) (h2 : (5 * s) / 2 = 44) : 4 * s = 70.4 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l1580_158075


namespace NUMINAMATH_CALUDE_payment_cases_count_l1580_158097

/-- The number of 500-won coins available -/
def num_500_coins : ℕ := 4

/-- The number of 100-won coins available -/
def num_100_coins : ℕ := 2

/-- The number of 10-won coins available -/
def num_10_coins : ℕ := 5

/-- The total number of non-zero payment cases -/
def total_cases : ℕ := (num_500_coins + 1) * (num_100_coins + 1) * (num_10_coins + 1) - 1

theorem payment_cases_count : total_cases = 89 := by
  sorry

end NUMINAMATH_CALUDE_payment_cases_count_l1580_158097


namespace NUMINAMATH_CALUDE_largest_s_value_largest_s_value_is_121_l1580_158060

/-- The largest possible value of s for regular polygons Q1 (r-gon) and Q2 (s-gon) 
    satisfying the given conditions -/
theorem largest_s_value : ℕ :=
  let r : ℕ → ℕ := fun s => 120 * s / (122 - s)
  let interior_angle : ℕ → ℚ := fun n => (n - 2 : ℚ) * 180 / n
  let s_max := 121
  have h1 : ∀ s : ℕ, s ≥ 3 → s ≤ s_max → 
    (interior_angle (r s)) / (interior_angle s) = 61 / 60 := by sorry
  have h2 : ∀ s : ℕ, s > s_max → ¬(∃ r : ℕ, r ≥ s ∧ 
    (interior_angle r) / (interior_angle s) = 61 / 60) := by sorry
  s_max

/-- Proof that the largest possible value of s is indeed 121 -/
theorem largest_s_value_is_121 : largest_s_value = 121 := by sorry

end NUMINAMATH_CALUDE_largest_s_value_largest_s_value_is_121_l1580_158060


namespace NUMINAMATH_CALUDE_smallest_value_l1580_158045

theorem smallest_value (A B C D : ℝ) : 
  A = Real.sin (50 * π / 180) * Real.cos (39 * π / 180) - Real.sin (40 * π / 180) * Real.cos (51 * π / 180) →
  B = -2 * Real.sin (40 * π / 180)^2 + 1 →
  C = 2 * Real.sin (6 * π / 180) * Real.cos (6 * π / 180) →
  D = Real.sqrt 3 / 2 * Real.sin (43 * π / 180) - 1 / 2 * Real.cos (43 * π / 180) →
  B < A ∧ B < C ∧ B < D :=
by sorry


end NUMINAMATH_CALUDE_smallest_value_l1580_158045


namespace NUMINAMATH_CALUDE_floor_ceil_sum_l1580_158028

theorem floor_ceil_sum : ⌊(-3.01 : ℝ)⌋ + ⌈(24.99 : ℝ)⌉ = 21 := by sorry

end NUMINAMATH_CALUDE_floor_ceil_sum_l1580_158028


namespace NUMINAMATH_CALUDE_tan_roots_and_angle_sum_cosine_product_l1580_158040

theorem tan_roots_and_angle_sum_cosine_product
  (α β : Real)
  (h1 : ∀ x, x^2 + 3 * Real.sqrt 3 * x + 4 = 0 ↔ x = Real.tan α ∨ x = Real.tan β)
  (h2 : α ∈ Set.Ioo (-π/2) (π/2))
  (h3 : β ∈ Set.Ioo (-π/2) (π/2)) :
  (α + β = -2*π/3) ∧ (Real.cos α * Real.cos β = 1/6) := by
sorry

end NUMINAMATH_CALUDE_tan_roots_and_angle_sum_cosine_product_l1580_158040


namespace NUMINAMATH_CALUDE_triangular_weight_is_60_l1580_158093

/-- Given a set of weights with specific balancing conditions, prove that the triangular weight is 60 grams. -/
theorem triangular_weight_is_60 :
  ∀ (round_weight triangular_weight : ℝ),
  (round_weight + triangular_weight = 3 * round_weight) →
  (4 * round_weight + triangular_weight = triangular_weight + round_weight + 90) →
  triangular_weight = 60 := by
  sorry

end NUMINAMATH_CALUDE_triangular_weight_is_60_l1580_158093


namespace NUMINAMATH_CALUDE_problem_statement_l1580_158079

theorem problem_statement (x y : ℝ) (h1 : x - y = 2) (h2 : x^2 + y^2 = 4) :
  x^2004 + y^2004 = 2^2004 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1580_158079


namespace NUMINAMATH_CALUDE_bee_hatch_count_l1580_158050

/-- The number of bees hatching from the queen's eggs every day -/
def daily_hatch : ℕ := 3001

/-- The number of bees the queen loses every day -/
def daily_loss : ℕ := 900

/-- The number of days -/
def days : ℕ := 7

/-- The total number of bees in the hive after 7 days -/
def final_bees : ℕ := 27201

/-- The initial number of bees -/
def initial_bees : ℕ := 12500

/-- Theorem stating that the number of bees hatching daily is correct -/
theorem bee_hatch_count :
  initial_bees + days * (daily_hatch - daily_loss) = final_bees :=
by sorry

end NUMINAMATH_CALUDE_bee_hatch_count_l1580_158050


namespace NUMINAMATH_CALUDE_tangent_parabola_to_line_l1580_158096

theorem tangent_parabola_to_line (a : ℝ) : 
  (∃ x : ℝ, a * x^2 + 1 = x ∧ ∀ y : ℝ, y ≠ x → a * y^2 + 1 ≠ y) → a = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_tangent_parabola_to_line_l1580_158096


namespace NUMINAMATH_CALUDE_pirate_treasure_probability_l1580_158063

def num_islands : ℕ := 8
def num_treasure_islands : ℕ := 4

def prob_treasure_only : ℚ := 1/5
def prob_traps_only : ℚ := 1/10
def prob_both : ℚ := 1/10
def prob_neither : ℚ := 3/5

def prob_treasure : ℚ := prob_treasure_only + prob_both
def prob_no_treasure_no_traps : ℚ := prob_neither

theorem pirate_treasure_probability :
  (Nat.choose num_islands num_treasure_islands : ℚ) *
  (prob_treasure ^ num_treasure_islands) *
  (prob_no_treasure_no_traps ^ (num_islands - num_treasure_islands)) =
  91854/1250000 := by
  sorry

end NUMINAMATH_CALUDE_pirate_treasure_probability_l1580_158063


namespace NUMINAMATH_CALUDE_apartment_floors_proof_l1580_158000

/-- The number of apartment buildings -/
def num_buildings : ℕ := 2

/-- The number of apartments per floor -/
def apartments_per_floor : ℕ := 6

/-- The number of doors needed per apartment -/
def doors_per_apartment : ℕ := 7

/-- The total number of doors needed -/
def total_doors : ℕ := 1008

/-- The number of floors in each apartment building -/
def floors_per_building : ℕ := 12

theorem apartment_floors_proof :
  floors_per_building * num_buildings * apartments_per_floor * doors_per_apartment = total_doors :=
by sorry

end NUMINAMATH_CALUDE_apartment_floors_proof_l1580_158000


namespace NUMINAMATH_CALUDE_toms_books_l1580_158034

/-- Given that Joan has 10 books and together with Tom they have 48 books,
    prove that Tom has 38 books. -/
theorem toms_books (joan_books : ℕ) (total_books : ℕ) (h1 : joan_books = 10) (h2 : total_books = 48) :
  total_books - joan_books = 38 := by
  sorry

end NUMINAMATH_CALUDE_toms_books_l1580_158034


namespace NUMINAMATH_CALUDE_derivative_at_one_l1580_158058

def f (x : ℝ) (k : ℝ) : ℝ := x^3 - 2*k*x + 1

theorem derivative_at_one (f : ℝ → ℝ) (f' : ℝ → ℝ) :
  (∀ x, deriv f x = f' x) →
  (∃ k, ∀ x, f x = x^3 - 2*k*x + 1) →
  f' 1 = 1 := by
sorry

end NUMINAMATH_CALUDE_derivative_at_one_l1580_158058


namespace NUMINAMATH_CALUDE_binomial_expansion_positive_integer_powers_l1580_158006

theorem binomial_expansion_positive_integer_powers (x : ℝ) : 
  (Finset.filter (fun r : ℕ => (10 - 3*r) / 2 > 0 ∧ (10 - 3*r) % 2 = 0) (Finset.range 11)).card = 2 :=
sorry

end NUMINAMATH_CALUDE_binomial_expansion_positive_integer_powers_l1580_158006


namespace NUMINAMATH_CALUDE_fermat_prime_condition_l1580_158031

theorem fermat_prime_condition (a n : ℕ) (ha : a > 1) (hn : n > 1) :
  Nat.Prime (a^n + 1) → (Even a ∧ ∃ k : ℕ, n = 2^k) :=
by sorry

end NUMINAMATH_CALUDE_fermat_prime_condition_l1580_158031


namespace NUMINAMATH_CALUDE_pebble_splash_width_proof_l1580_158056

/-- The width of a splash made by a pebble -/
def pebble_splash_width : ℝ := 0.25

theorem pebble_splash_width_proof 
  (total_splash_width : ℝ) 
  (rock_splash_width : ℝ) 
  (boulder_splash_width : ℝ) 
  (pebble_count : ℕ) 
  (rock_count : ℕ) 
  (boulder_count : ℕ) 
  (h1 : total_splash_width = 7) 
  (h2 : rock_splash_width = 1/2) 
  (h3 : boulder_splash_width = 2) 
  (h4 : pebble_count = 6) 
  (h5 : rock_count = 3) 
  (h6 : boulder_count = 2) : 
  pebble_splash_width = (total_splash_width - rock_count * rock_splash_width - boulder_count * boulder_splash_width) / pebble_count :=
by sorry

end NUMINAMATH_CALUDE_pebble_splash_width_proof_l1580_158056


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_on_circle_l1580_158014

theorem sum_of_x_and_y_on_circle (x y : ℝ) (h : x^2 + y^2 = 16*x - 8*y - 60) : x + y = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_on_circle_l1580_158014


namespace NUMINAMATH_CALUDE_fern_bushes_needed_l1580_158022

/-- The number of bushes needed to produce a given amount of perfume -/
def bushes_needed (petals_per_ounce : ℕ) (petals_per_rose : ℕ) (roses_per_bush : ℕ) 
                  (ounces_per_bottle : ℕ) (num_bottles : ℕ) : ℕ :=
  (petals_per_ounce * ounces_per_bottle * num_bottles) / (petals_per_rose * roses_per_bush)

/-- Theorem stating the number of bushes Fern needs to harvest -/
theorem fern_bushes_needed : 
  bushes_needed 320 8 12 12 20 = 800 := by
  sorry

end NUMINAMATH_CALUDE_fern_bushes_needed_l1580_158022


namespace NUMINAMATH_CALUDE_shortest_distance_exp_to_line_l1580_158032

-- Define the function f(x) = e^x
noncomputable def f (x : ℝ) : ℝ := Real.exp x

-- Define the line g(x) = x
def g (x : ℝ) : ℝ := x

-- Statement: The shortest distance from any point on f to g is √2/2
theorem shortest_distance_exp_to_line :
  ∃ d : ℝ, d = Real.sqrt 2 / 2 ∧
  ∀ x y : ℝ, f x = y → 
  ∀ p : ℝ × ℝ, p.1 = x ∧ p.2 = y → 
  d ≤ Real.sqrt ((p.1 - p.2)^2 + 1) / Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_shortest_distance_exp_to_line_l1580_158032


namespace NUMINAMATH_CALUDE_sqrt_inequality_l1580_158062

theorem sqrt_inequality (a : ℝ) (h : a ≥ 2) : 
  Real.sqrt (a + 1) - Real.sqrt a < Real.sqrt (a - 1) - Real.sqrt (a - 2) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l1580_158062


namespace NUMINAMATH_CALUDE_power_two_eq_square_plus_one_solutions_power_two_plus_one_eq_square_solution_l1580_158054

theorem power_two_eq_square_plus_one_solutions (x n : ℕ) :
  2^n = x^2 + 1 ↔ (x = 0 ∧ n = 0) ∨ (x = 1 ∧ n = 1) := by sorry

theorem power_two_plus_one_eq_square_solution (x n : ℕ) :
  2^n + 1 = x^2 ↔ x = 3 ∧ n = 3 := by sorry

end NUMINAMATH_CALUDE_power_two_eq_square_plus_one_solutions_power_two_plus_one_eq_square_solution_l1580_158054


namespace NUMINAMATH_CALUDE_uncool_parents_count_l1580_158057

theorem uncool_parents_count (total_students : ℕ) (cool_dads : ℕ) (cool_moms : ℕ) (both_cool : ℕ) 
  (h1 : total_students = 40)
  (h2 : cool_dads = 18)
  (h3 : cool_moms = 22)
  (h4 : both_cool = 10) :
  total_students - (cool_dads - both_cool + cool_moms - both_cool + both_cool) = 10 :=
by sorry

end NUMINAMATH_CALUDE_uncool_parents_count_l1580_158057


namespace NUMINAMATH_CALUDE_tangent_slope_at_2_10_l1580_158064

/-- The function f(x) = x^2 + 3x --/
def f (x : ℝ) : ℝ := x^2 + 3*x

/-- The derivative of f(x) --/
def f' (x : ℝ) : ℝ := 2*x + 3

theorem tangent_slope_at_2_10 : 
  f' 2 = 7 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_at_2_10_l1580_158064


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l1580_158074

theorem polynomial_division_theorem (x : ℝ) :
  (x - 2) * (x^5 + 2*x^4 + 4*x^3 + 13*x^2 + 26*x + 52) + 96 = x^6 + 5*x^3 - 8 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l1580_158074


namespace NUMINAMATH_CALUDE_classroom_wall_paint_area_l1580_158077

/-- Calculates the area to be painted on a wall with two windows. -/
def areaToBePainted (wallHeight wallWidth window1Height window1Width window2Height window2Width : ℕ) : ℕ :=
  let wallArea := wallHeight * wallWidth
  let window1Area := window1Height * window1Width
  let window2Area := window2Height * window2Width
  wallArea - window1Area - window2Area

/-- Proves that the area to be painted on the classroom wall is 243 square feet. -/
theorem classroom_wall_paint_area :
  areaToBePainted 15 18 3 5 2 6 = 243 := by
  sorry

#eval areaToBePainted 15 18 3 5 2 6

end NUMINAMATH_CALUDE_classroom_wall_paint_area_l1580_158077


namespace NUMINAMATH_CALUDE_expression_evaluation_l1580_158036

theorem expression_evaluation : 
  ∃ ε > 0, |((10 * 1.8 - 2 * 1.5) / 0.3 + Real.rpow 3 (2/3) - Real.log 4) - 50.6938| < ε :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1580_158036


namespace NUMINAMATH_CALUDE_election_ratio_l1580_158095

theorem election_ratio :
  ∀ (R D : ℝ),
  R > 0 → D > 0 →
  (0.70 * R + 0.25 * D) - (0.30 * R + 0.75 * D) = 0.039999999999999853 * (R + D) →
  R / D = 1.5 := by
sorry

end NUMINAMATH_CALUDE_election_ratio_l1580_158095


namespace NUMINAMATH_CALUDE_optimal_production_plan_l1580_158019

/-- Represents the production plan for the factory -/
structure ProductionPlan where
  hoursA : ℝ  -- Hours to produce Product A
  hoursB : ℝ  -- Hours to produce Product B

/-- Calculates the total profit for a given production plan -/
def totalProfit (plan : ProductionPlan) : ℝ :=
  30 * plan.hoursA + 40 * plan.hoursB

/-- Checks if a production plan is feasible given the material constraints -/
def isFeasible (plan : ProductionPlan) : Prop :=
  3 * plan.hoursA + 2 * plan.hoursB ≤ 1200 ∧
  plan.hoursA + 2 * plan.hoursB ≤ 800 ∧
  plan.hoursA ≥ 0 ∧ plan.hoursB ≥ 0

/-- The optimal production plan -/
def optimalPlan : ProductionPlan :=
  { hoursA := 200, hoursB := 300 }

theorem optimal_production_plan :
  isFeasible optimalPlan ∧
  ∀ plan : ProductionPlan, isFeasible plan →
    totalProfit plan ≤ totalProfit optimalPlan ∧
  totalProfit optimalPlan = 18000 := by
  sorry


end NUMINAMATH_CALUDE_optimal_production_plan_l1580_158019


namespace NUMINAMATH_CALUDE_tan_22_5_degree_decomposition_l1580_158027

theorem tan_22_5_degree_decomposition :
  ∃ (a b c : ℕ+), 
    (a.val ≥ b.val ∧ b.val ≥ c.val) ∧
    (Real.tan (22.5 * π / 180) = Real.sqrt a.val - 1 + Real.sqrt b.val - Real.sqrt c.val) ∧
    (a.val + b.val + c.val = 12) := by sorry

end NUMINAMATH_CALUDE_tan_22_5_degree_decomposition_l1580_158027
