import Mathlib

namespace NUMINAMATH_CALUDE_snow_probability_in_week_l1237_123730

theorem snow_probability_in_week (p1 p2 : ℝ) : 
  p1 = 1/2 → p2 = 1/3 → 
  (1 - (1 - p1)^4 * (1 - p2)^3) = 53/54 := by
  sorry

end NUMINAMATH_CALUDE_snow_probability_in_week_l1237_123730


namespace NUMINAMATH_CALUDE_cost_of_450_candies_l1237_123753

/-- The cost of buying a given number of chocolate candies -/
def cost_of_candies (candies_per_box : ℕ) (cost_per_box : ℚ) (total_candies : ℕ) : ℚ :=
  (total_candies / candies_per_box) * cost_per_box

/-- Theorem: The cost of 450 chocolate candies is $112.50 -/
theorem cost_of_450_candies :
  cost_of_candies 30 (7.5 : ℚ) 450 = (112.5 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_cost_of_450_candies_l1237_123753


namespace NUMINAMATH_CALUDE_harry_blue_weights_l1237_123785

/-- Represents the weight configuration on a gym bar -/
structure WeightConfig where
  blue_weight : ℕ  -- Weight of each blue weight in pounds
  green_weight : ℕ  -- Weight of each green weight in pounds
  num_green : ℕ  -- Number of green weights
  bar_weight : ℕ  -- Weight of the bar in pounds
  total_weight : ℕ  -- Total weight in pounds

/-- Calculates the number of blue weights given a weight configuration -/
def num_blue_weights (config : WeightConfig) : ℕ :=
  (config.total_weight - config.bar_weight - config.num_green * config.green_weight) / config.blue_weight

/-- Theorem stating that Harry's configuration results in 4 blue weights -/
theorem harry_blue_weights :
  let config : WeightConfig := {
    blue_weight := 2,
    green_weight := 3,
    num_green := 5,
    bar_weight := 2,
    total_weight := 25
  }
  num_blue_weights config = 4 := by sorry

end NUMINAMATH_CALUDE_harry_blue_weights_l1237_123785


namespace NUMINAMATH_CALUDE_base_4_20312_equals_566_l1237_123764

def base_4_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

theorem base_4_20312_equals_566 :
  base_4_to_10 [2, 1, 3, 0, 2] = 566 := by
  sorry

end NUMINAMATH_CALUDE_base_4_20312_equals_566_l1237_123764


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1237_123710

theorem sufficient_not_necessary (a : ℝ) :
  (∀ a, a < -1 → a^2 - 5*a - 6 > 0) ∧
  (∃ a, a^2 - 5*a - 6 > 0 ∧ a ≥ -1) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1237_123710


namespace NUMINAMATH_CALUDE_quadratic_trinomial_transformation_root_l1237_123790

/-- Given a quadratic trinomial ax^2 + bx + c, if we swap b and c, 
    add the result to the original trinomial, and the resulting 
    trinomial has a single root, then that root must be either 0 or -2. -/
theorem quadratic_trinomial_transformation_root (a b c : ℝ) :
  let original := fun x => a * x^2 + b * x + c
  let swapped := fun x => a * x^2 + c * x + b
  let result := fun x => original x + swapped x
  (∃! r, result r = 0) → (result 0 = 0 ∨ result (-2) = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_trinomial_transformation_root_l1237_123790


namespace NUMINAMATH_CALUDE_rectangular_box_height_l1237_123779

theorem rectangular_box_height (wooden_box_length wooden_box_width wooden_box_height : ℕ)
  (box_length box_width : ℕ) (max_boxes : ℕ) :
  wooden_box_length = 800 ∧ wooden_box_width = 700 ∧ wooden_box_height = 600 ∧
  box_length = 8 ∧ box_width = 7 ∧ max_boxes = 1000000 →
  ∃ (box_height : ℕ), 
    (wooden_box_length * wooden_box_width * wooden_box_height) / max_boxes = 
    box_length * box_width * box_height ∧ box_height = 6 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_box_height_l1237_123779


namespace NUMINAMATH_CALUDE_unique_functional_equation_solution_l1237_123776

theorem unique_functional_equation_solution :
  ∃! f : ℝ → ℝ, ∀ x y : ℝ, f x * f y - f (x * y) = x^2 + y :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_functional_equation_solution_l1237_123776


namespace NUMINAMATH_CALUDE_complex_fraction_evaluation_l1237_123771

theorem complex_fraction_evaluation : 
  (((7 - 6.35) / 6.5 + 9.9) * (1 / 12.8)) / 
  ((1.2 / 36 + (1 + 1/5) / 0.25 - (1 + 5/6)) * (1 + 1/4)) / 0.125 = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_evaluation_l1237_123771


namespace NUMINAMATH_CALUDE_dogs_with_tags_l1237_123792

theorem dogs_with_tags (total : ℕ) (with_flea_collars : ℕ) (with_both : ℕ) (with_neither : ℕ) : 
  total = 80 → 
  with_flea_collars = 40 → 
  with_both = 6 → 
  with_neither = 1 → 
  total - with_flea_collars + with_both - with_neither = 45 := by
sorry

end NUMINAMATH_CALUDE_dogs_with_tags_l1237_123792


namespace NUMINAMATH_CALUDE_square_equation_solutions_l1237_123797

theorem square_equation_solutions (n : ℝ) :
  ∃ (x y : ℝ), x ≠ y ∧
  (n - (2 * n + 1) / 2)^2 = ((n + 1) - (2 * n + 1) / 2)^2 ∧
  (x = n - (2 * n + 1) / 2 ∧ y = (n + 1) - (2 * n + 1) / 2) ∨
  (x = n - (2 * n + 1) / 2 ∧ y = -((n + 1) - (2 * n + 1) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_square_equation_solutions_l1237_123797


namespace NUMINAMATH_CALUDE_lawrence_walking_days_l1237_123775

/-- Given Lawrence's walking data, prove the number of days he walked. -/
theorem lawrence_walking_days (daily_distance : ℝ) (total_distance : ℝ) 
  (h1 : daily_distance = 4.0)
  (h2 : total_distance = 12) : 
  total_distance / daily_distance = 3 := by
  sorry

end NUMINAMATH_CALUDE_lawrence_walking_days_l1237_123775


namespace NUMINAMATH_CALUDE_net_amount_calculation_l1237_123734

/-- Calculate the net amount received from selling puppies --/
def net_amount_from_puppies (luna_puppies stella_puppies : ℕ)
                            (luna_sold stella_sold : ℕ)
                            (luna_price stella_price : ℕ)
                            (luna_cost stella_cost : ℕ) : ℕ :=
  let luna_revenue := luna_sold * luna_price
  let stella_revenue := stella_sold * stella_price
  let luna_expenses := luna_puppies * luna_cost
  let stella_expenses := stella_puppies * stella_cost
  (luna_revenue + stella_revenue) - (luna_expenses + stella_expenses)

theorem net_amount_calculation :
  net_amount_from_puppies 10 14 8 10 200 250 80 90 = 2040 := by
  sorry

end NUMINAMATH_CALUDE_net_amount_calculation_l1237_123734


namespace NUMINAMATH_CALUDE_complex_fraction_equals_two_l1237_123727

theorem complex_fraction_equals_two (z : ℂ) (h : z = 1 - I) : z^2 / (z - 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_two_l1237_123727


namespace NUMINAMATH_CALUDE_distinct_ratios_theorem_l1237_123777

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 2 then
    1/2 - |x - 3/2|
  else
    Real.exp (x - 2) * (-x^2 + 8*x - 12)

theorem distinct_ratios_theorem (n : ℕ) (x : Fin n → ℝ) :
  n ≥ 2 →
  (∀ i : Fin n, x i > 1) →
  (∀ i j : Fin n, i ≠ j → x i ≠ x j) →
  (∀ i j : Fin n, f (x i) / (x i) = f (x j) / (x j)) →
  n ∈ ({2, 3, 4} : Set ℕ) :=
sorry

end NUMINAMATH_CALUDE_distinct_ratios_theorem_l1237_123777


namespace NUMINAMATH_CALUDE_division_remainder_l1237_123739

theorem division_remainder (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) : 
  dividend = 181 → 
  divisor = 20 → 
  quotient = 9 → 
  remainder = dividend - (divisor * quotient) → 
  remainder = 1 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_l1237_123739


namespace NUMINAMATH_CALUDE_sqrt_450_simplified_l1237_123763

theorem sqrt_450_simplified : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_450_simplified_l1237_123763


namespace NUMINAMATH_CALUDE_parallel_lines_distance_l1237_123747

def line1 (x y : ℝ) : Prop := 3 * x + 4 * y = 2
def line2 (x y : ℝ) : Prop := 3 * x + 4 * y = 7

theorem parallel_lines_distance : 
  let d := |2 - 7| / Real.sqrt (3^2 + 4^2)
  d = 5 := by sorry

end NUMINAMATH_CALUDE_parallel_lines_distance_l1237_123747


namespace NUMINAMATH_CALUDE_pond_depth_l1237_123766

/-- Proves that a rectangular pond with given dimensions has a depth of 5 meters -/
theorem pond_depth (length width volume : ℝ) (h1 : length = 20) (h2 : width = 10) (h3 : volume = 1000) :
  volume / (length * width) = 5 := by
  sorry

end NUMINAMATH_CALUDE_pond_depth_l1237_123766


namespace NUMINAMATH_CALUDE_quadratic_vertex_form_h_l1237_123701

/-- 
Given a quadratic expression 3x^2 + 9x + 20, when expressed in the form a(x - h)^2 + k,
h is equal to -3/2
-/
theorem quadratic_vertex_form_h (x : ℝ) : 
  ∃ (a k : ℝ), 3 * x^2 + 9 * x + 20 = a * (x - (-3/2))^2 + k := by
  sorry

end NUMINAMATH_CALUDE_quadratic_vertex_form_h_l1237_123701


namespace NUMINAMATH_CALUDE_sum_of_values_equals_three_l1237_123744

/-- A discrete random variable with two possible values -/
structure DiscreteRV (α : Type) where
  value : α
  prob : α → ℝ

/-- The expectation of a discrete random variable -/
def expectation {α : Type} (X : DiscreteRV α) : ℝ :=
  sorry

/-- The variance of a discrete random variable -/
def variance {α : Type} (X : DiscreteRV α) : ℝ :=
  sorry

theorem sum_of_values_equals_three
  (ξ : DiscreteRV ℝ)
  (a b : ℝ)
  (h_prob_a : ξ.prob a = 2/3)
  (h_prob_b : ξ.prob b = 1/3)
  (h_lt : a < b)
  (h_expect : expectation ξ = 4/3)
  (h_var : variance ξ = 2/9) :
  a + b = 3 :=
sorry

end NUMINAMATH_CALUDE_sum_of_values_equals_three_l1237_123744


namespace NUMINAMATH_CALUDE_puzzle_solution_l1237_123704

-- Define the types of beings
inductive Being
| Human
| Monkey

-- Define the types of speakers
inductive Speaker
| Knight
| Liar

-- Define A and B as individuals
structure Individual where
  being : Being
  speaker : Speaker

-- Define the statements made by A and B
def statement_A (a b : Individual) : Prop :=
  a.being = Being.Monkey ∨ b.being = Being.Monkey

def statement_B (a b : Individual) : Prop :=
  a.speaker = Speaker.Liar ∨ b.speaker = Speaker.Liar

-- Theorem stating the conclusion
theorem puzzle_solution :
  ∃ (a b : Individual),
    (statement_A a b ↔ a.speaker = Speaker.Liar) ∧
    (statement_B a b ↔ b.speaker = Speaker.Knight) ∧
    a.being = Being.Human ∧
    b.being = Being.Human ∧
    a.speaker = Speaker.Liar ∧
    b.speaker = Speaker.Knight :=
sorry

end NUMINAMATH_CALUDE_puzzle_solution_l1237_123704


namespace NUMINAMATH_CALUDE_ducks_drinking_order_l1237_123715

theorem ducks_drinking_order (total_ducks : ℕ) (ducks_before_a : ℕ) (ducks_after_a : ℕ) :
  total_ducks = 20 →
  ducks_before_a = 11 →
  ducks_after_a = total_ducks - (ducks_before_a + 1) →
  ducks_after_a = 8 :=
by sorry

end NUMINAMATH_CALUDE_ducks_drinking_order_l1237_123715


namespace NUMINAMATH_CALUDE_marble_distribution_l1237_123787

theorem marble_distribution (total_marbles : ℕ) (additional_people : ℕ) : 
  total_marbles = 220 →
  additional_people = 2 →
  ∃ (x : ℕ), 
    (x > 0) ∧ 
    (total_marbles / x - 1 = total_marbles / (x + additional_people)) ∧
    x = 20 :=
by sorry

end NUMINAMATH_CALUDE_marble_distribution_l1237_123787


namespace NUMINAMATH_CALUDE_factors_of_x4_minus_4_l1237_123770

theorem factors_of_x4_minus_4 (x : ℝ) : 
  (x^4 - 4 = (x^2 + 2) * (x^2 - 2)) ∧ 
  (x^4 - 4 = (x^2 - 4) * (x^2 + 4)) ∧ 
  (x^4 - 4 ≠ (x + 1) * ((x^3 - x^2 - x + 5) / (x + 1))) ∧ 
  (x^4 - 4 ≠ (x^2 - 2*x + 2) * ((x^2 + 2*x + 2) / (x^2 - 2*x + 2))) :=
by sorry

end NUMINAMATH_CALUDE_factors_of_x4_minus_4_l1237_123770


namespace NUMINAMATH_CALUDE_equal_bills_at_80_minutes_l1237_123781

/-- United Telephone's base rate in dollars -/
def united_base : ℚ := 8

/-- United Telephone's per-minute rate in dollars -/
def united_per_minute : ℚ := 1/4

/-- Atlantic Call's base rate in dollars -/
def atlantic_base : ℚ := 12

/-- Atlantic Call's per-minute rate in dollars -/
def atlantic_per_minute : ℚ := 1/5

/-- The number of minutes at which the bills are equal -/
def equal_bill_minutes : ℚ := 80

theorem equal_bills_at_80_minutes :
  united_base + united_per_minute * equal_bill_minutes =
  atlantic_base + atlantic_per_minute * equal_bill_minutes :=
by sorry

end NUMINAMATH_CALUDE_equal_bills_at_80_minutes_l1237_123781


namespace NUMINAMATH_CALUDE_min_value_of_expression_lower_bound_achievable_l1237_123714

theorem min_value_of_expression (x : ℝ) : 
  (x + 1)^2 * (x + 2)^2 * (x + 3)^2 * (x + 4)^2 + 2025 ≥ 3625 :=
by sorry

theorem lower_bound_achievable : 
  ∃ x : ℝ, (x + 1)^2 * (x + 2)^2 * (x + 3)^2 * (x + 4)^2 + 2025 = 3625 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_lower_bound_achievable_l1237_123714


namespace NUMINAMATH_CALUDE_sin_shift_l1237_123780

theorem sin_shift (x : ℝ) : Real.sin (2 * x + π / 4) = Real.sin (2 * (x - π / 8)) := by
  sorry

end NUMINAMATH_CALUDE_sin_shift_l1237_123780


namespace NUMINAMATH_CALUDE_unique_two_digit_multiple_l1237_123748

theorem unique_two_digit_multiple : ∃! t : ℕ, 10 ≤ t ∧ t < 100 ∧ 13 * t % 100 = 52 := by
  sorry

end NUMINAMATH_CALUDE_unique_two_digit_multiple_l1237_123748


namespace NUMINAMATH_CALUDE_orange_price_l1237_123707

theorem orange_price (apple_price : ℚ) (total_fruit : ℕ) (initial_avg : ℚ) 
  (oranges_removed : ℕ) (final_avg : ℚ) :
  apple_price = 40 / 100 →
  total_fruit = 10 →
  initial_avg = 54 / 100 →
  oranges_removed = 4 →
  final_avg = 50 / 100 →
  ∃ (orange_price : ℚ),
    orange_price = 60 / 100 ∧
    ∃ (apples oranges : ℕ),
      apples + oranges = total_fruit ∧
      (apple_price * apples + orange_price * oranges) / total_fruit = initial_avg ∧
      (apple_price * apples + orange_price * (oranges - oranges_removed)) / 
        (total_fruit - oranges_removed) = final_avg :=
by
  sorry

end NUMINAMATH_CALUDE_orange_price_l1237_123707


namespace NUMINAMATH_CALUDE_waiter_tips_fraction_l1237_123752

/-- Given a waiter's salary and tips, where the tips are 7/4 of the salary,
    prove that the fraction of total income from tips is 7/11. -/
theorem waiter_tips_fraction (salary : ℚ) (tips : ℚ) (total_income : ℚ) : 
  tips = (7 : ℚ) / 4 * salary →
  total_income = salary + tips →
  tips / total_income = (7 : ℚ) / 11 := by
  sorry

end NUMINAMATH_CALUDE_waiter_tips_fraction_l1237_123752


namespace NUMINAMATH_CALUDE_card_drawing_combinations_l1237_123772

-- Define the number of piles and cards per pile
def num_piles : ℕ := 3
def cards_per_pile : ℕ := 3

-- Define the total number of cards
def total_cards : ℕ := num_piles * cards_per_pile

-- Define the function to calculate the number of ways to draw the cards
def ways_to_draw_cards : ℕ := (Nat.factorial total_cards) / ((Nat.factorial cards_per_pile) ^ num_piles)

-- Theorem statement
theorem card_drawing_combinations :
  ways_to_draw_cards = 1680 :=
sorry

end NUMINAMATH_CALUDE_card_drawing_combinations_l1237_123772


namespace NUMINAMATH_CALUDE_shirt_total_price_l1237_123793

/-- The total price of 25 shirts given the conditions in the problem -/
theorem shirt_total_price : 
  ∀ (shirt_price sweater_price : ℝ),
  75 * sweater_price = 1500 →
  sweater_price = shirt_price + 4 →
  25 * shirt_price = 400 := by
    sorry

end NUMINAMATH_CALUDE_shirt_total_price_l1237_123793


namespace NUMINAMATH_CALUDE_double_time_double_discount_l1237_123705

/-- Represents the true discount calculation for a bill -/
def true_discount (face_value : ℝ) (discount : ℝ) (time : ℝ) : Prop :=
  ∃ (rate : ℝ),
    discount = (face_value - discount) * rate * time ∧
    rate > 0 ∧
    time > 0

/-- 
If the true discount on a bill of 110 is 10 for a certain time,
then the true discount on the same bill for double the time is 20.
-/
theorem double_time_double_discount :
  ∀ (time : ℝ),
    true_discount 110 10 time →
    true_discount 110 20 (2 * time) :=
by
  sorry

end NUMINAMATH_CALUDE_double_time_double_discount_l1237_123705


namespace NUMINAMATH_CALUDE_parallel_vectors_k_value_l1237_123769

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), t ≠ 0 ∧ a.1 = t * b.1 ∧ a.2 = t * b.2

/-- The theorem stating that if (1,k) is parallel to (2,1), then k = 1/2 -/
theorem parallel_vectors_k_value (k : ℝ) :
  are_parallel (1, k) (2, 1) → k = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_value_l1237_123769


namespace NUMINAMATH_CALUDE_sabrina_cookies_l1237_123741

theorem sabrina_cookies (initial_cookies : ℕ) (final_cookies : ℕ) 
  (h1 : initial_cookies = 20) 
  (h2 : final_cookies = 5) : ℕ :=
  let cookies_to_brother := 10
  let cookies_from_mother := cookies_to_brother / 2
  let total_before_sister := initial_cookies - cookies_to_brother + cookies_from_mother
  let cookies_kept := total_before_sister / 3
  by
    have h3 : cookies_kept = final_cookies := by sorry
    have h4 : total_before_sister = cookies_kept * 3 := by sorry
    have h5 : initial_cookies - cookies_to_brother + cookies_from_mother = total_before_sister := by sorry
    exact cookies_to_brother

end NUMINAMATH_CALUDE_sabrina_cookies_l1237_123741


namespace NUMINAMATH_CALUDE_max_value_of_expression_max_value_achievable_l1237_123720

theorem max_value_of_expression (x : ℝ) :
  x^6 / (x^12 + 4*x^9 - 6*x^6 + 16*x^3 + 64) ≤ 1/26 :=
by sorry

theorem max_value_achievable :
  ∃ x : ℝ, x^6 / (x^12 + 4*x^9 - 6*x^6 + 16*x^3 + 64) = 1/26 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_max_value_achievable_l1237_123720


namespace NUMINAMATH_CALUDE_probability_of_letter_selection_l1237_123706

theorem probability_of_letter_selection (total_letters : ℕ) (unique_letters : ℕ) 
  (h1 : total_letters = 26) (h2 : unique_letters = 8) :
  (unique_letters : ℚ) / total_letters = 4 / 13 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_letter_selection_l1237_123706


namespace NUMINAMATH_CALUDE_evaluate_expression_l1237_123713

theorem evaluate_expression : (1 / ((-7^3)^3)) * ((-7)^10) = -7 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1237_123713


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l1237_123762

theorem sqrt_product_equality : (Real.sqrt 12 + 2) * (Real.sqrt 3 - 1) = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l1237_123762


namespace NUMINAMATH_CALUDE_sequence_existence_and_boundedness_l1237_123749

theorem sequence_existence_and_boundedness (a : ℝ) (n : ℕ) (hn : n > 0) :
  ∃! x : Fin (n + 2) → ℝ,
    (x 0 = 0 ∧ x (Fin.last n) = 0) ∧
    (∀ i : Fin (n + 1), i.val > 0 →
      (x i + x (i + 1)) / 2 = x i + (x i)^3 - a^3) ∧
    (∀ i : Fin (n + 2), |x i| ≤ |a|) := by
  sorry

end NUMINAMATH_CALUDE_sequence_existence_and_boundedness_l1237_123749


namespace NUMINAMATH_CALUDE_relationship_abc_l1237_123795

theorem relationship_abc (x : ℝ) (h : x > 2) : (1/3)^3 < Real.log x ∧ Real.log x < x^3 := by
  sorry

end NUMINAMATH_CALUDE_relationship_abc_l1237_123795


namespace NUMINAMATH_CALUDE_log_2_3_in_terms_of_a_b_l1237_123722

theorem log_2_3_in_terms_of_a_b (a b : ℝ) (ha : a = Real.log 6) (hb : b = Real.log 20) :
  Real.log 3 / Real.log 2 = (a - b + 1) / (b - 1) := by
  sorry

end NUMINAMATH_CALUDE_log_2_3_in_terms_of_a_b_l1237_123722


namespace NUMINAMATH_CALUDE_restaurant_friends_l1237_123702

theorem restaurant_friends (initial_wings : ℕ) (cooked_wings : ℕ) (wings_per_person : ℕ) : 
  initial_wings = 9 →
  cooked_wings = 7 →
  wings_per_person = 4 →
  (initial_wings + cooked_wings) / wings_per_person = 4 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_friends_l1237_123702


namespace NUMINAMATH_CALUDE_min_xy_value_l1237_123736

theorem min_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : (x + 1) * (y + 1) = 2 * x + 2 * y + 4) : 
  x * y ≥ 9 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ (x + 1) * (y + 1) = 2 * x + 2 * y + 4 ∧ x * y = 9 := by
  sorry

end NUMINAMATH_CALUDE_min_xy_value_l1237_123736


namespace NUMINAMATH_CALUDE_student_miscalculation_l1237_123742

theorem student_miscalculation (a : ℤ) : 
  (-16 - a = -12) → (-16 + a = -20) := by
  sorry

end NUMINAMATH_CALUDE_student_miscalculation_l1237_123742


namespace NUMINAMATH_CALUDE_bouquet_composition_l1237_123754

/-- Represents a bouquet of branches -/
structure Bouquet :=
  (white : ℕ)
  (blue : ℕ)

/-- The conditions for our specific bouquet -/
def ValidBouquet (b : Bouquet) : Prop :=
  b.white + b.blue = 7 ∧
  b.white ≥ 1 ∧
  ∀ (x y : ℕ), x < y → x < 7 → y < 7 → (x = b.white → y = b.blue)

/-- The theorem to be proved -/
theorem bouquet_composition (b : Bouquet) (h : ValidBouquet b) : b.white = 1 ∧ b.blue = 6 := by
  sorry


end NUMINAMATH_CALUDE_bouquet_composition_l1237_123754


namespace NUMINAMATH_CALUDE_gcd_1908_4187_l1237_123728

theorem gcd_1908_4187 : Nat.gcd 1908 4187 = 53 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1908_4187_l1237_123728


namespace NUMINAMATH_CALUDE_sat_score_improvement_l1237_123740

theorem sat_score_improvement (first_score second_score : ℕ) 
  (h1 : first_score = 1000) 
  (h2 : second_score = 1100) : 
  (second_score - first_score) / first_score * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_sat_score_improvement_l1237_123740


namespace NUMINAMATH_CALUDE_intersection_with_complement_l1237_123798

def U : Set Int := {-1, 0, 1, 2, 3}
def A : Set Int := {0, 1, 2}
def B : Set Int := {2, 3}

theorem intersection_with_complement :
  A ∩ (U \ B) = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l1237_123798


namespace NUMINAMATH_CALUDE_max_soap_boxes_in_carton_l1237_123709

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ :=
  d.length * d.width * d.height

/-- The dimensions of the carton -/
def cartonDimensions : BoxDimensions :=
  { length := 30, width := 42, height := 60 }

/-- The dimensions of a soap box -/
def soapBoxDimensions : BoxDimensions :=
  { length := 7, width := 6, height := 5 }

/-- Theorem stating the maximum number of soap boxes that can fit in the carton -/
theorem max_soap_boxes_in_carton :
  (boxVolume cartonDimensions) / (boxVolume soapBoxDimensions) = 360 := by
  sorry

#eval (boxVolume cartonDimensions) / (boxVolume soapBoxDimensions)

end NUMINAMATH_CALUDE_max_soap_boxes_in_carton_l1237_123709


namespace NUMINAMATH_CALUDE_distinct_prime_factors_of_divisor_sum_360_l1237_123789

/-- The sum of positive divisors of a natural number n -/
def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- The number of distinct prime factors of a natural number n -/
def num_distinct_prime_factors (n : ℕ) : ℕ := sorry

/-- Theorem: The number of distinct prime factors of the sum of positive divisors of 360 is 4 -/
theorem distinct_prime_factors_of_divisor_sum_360 : 
  num_distinct_prime_factors (sum_of_divisors 360) = 4 := by sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_of_divisor_sum_360_l1237_123789


namespace NUMINAMATH_CALUDE_basketball_team_selection_l1237_123796

def team_size : ℕ := 15
def captain_count : ℕ := 2
def lineup_size : ℕ := 5

theorem basketball_team_selection :
  (team_size.choose captain_count) * 
  (team_size - captain_count).factorial / (team_size - captain_count - lineup_size).factorial = 16201200 := by
  sorry

end NUMINAMATH_CALUDE_basketball_team_selection_l1237_123796


namespace NUMINAMATH_CALUDE_min_value_of_product_l1237_123755

-- Define the quadratic function f
noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the theorem
theorem min_value_of_product (a b c : ℝ) (x₁ x₂ x₃ : ℝ) :
  a ≠ 0 →
  f a b c (-1) = 0 →
  (∀ x : ℝ, f a b c x ≥ x) →
  (∀ x : ℝ, 0 < x → x < 2 → f a b c x ≤ (x + 1)^2 / 4) →
  0 < x₁ → x₁ < 2 →
  0 < x₂ → x₂ < 2 →
  0 < x₃ → x₃ < 2 →
  1 / x₁ + 1 / x₂ + 1 / x₃ = 3 →
  ∃ (m : ℝ), m = 1 ∧ ∀ y₁ y₂ y₃ : ℝ,
    0 < y₁ → y₁ < 2 →
    0 < y₂ → y₂ < 2 →
    0 < y₃ → y₃ < 2 →
    1 / y₁ + 1 / y₂ + 1 / y₃ = 3 →
    m ≤ f a b c y₁ * f a b c y₂ * f a b c y₃ :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_of_product_l1237_123755


namespace NUMINAMATH_CALUDE_total_dolls_count_l1237_123773

/-- The number of dolls in a big box -/
def dolls_per_big_box : ℕ := 7

/-- The number of dolls in a small box -/
def dolls_per_small_box : ℕ := 4

/-- The number of big boxes -/
def num_big_boxes : ℕ := 5

/-- The number of small boxes -/
def num_small_boxes : ℕ := 9

/-- The total number of dolls in all boxes -/
def total_dolls : ℕ := dolls_per_big_box * num_big_boxes + dolls_per_small_box * num_small_boxes

theorem total_dolls_count : total_dolls = 71 := by
  sorry

end NUMINAMATH_CALUDE_total_dolls_count_l1237_123773


namespace NUMINAMATH_CALUDE_lottery_probability_l1237_123791

theorem lottery_probability (p : ℝ) (n : ℕ) (h1 : p = 1 / 10000000) (h2 : n = 5) :
  n * p = 5 / 10000000 := by sorry

end NUMINAMATH_CALUDE_lottery_probability_l1237_123791


namespace NUMINAMATH_CALUDE_stone_order_calculation_l1237_123767

theorem stone_order_calculation (total material_ordered concrete_ordered bricks_ordered stone_ordered : ℝ) :
  total_material_ordered = 0.83 ∧
  concrete_ordered = 0.17 ∧
  bricks_ordered = 0.17 ∧
  total_material_ordered = concrete_ordered + bricks_ordered + stone_ordered →
  stone_ordered = 0.49 := by
sorry

end NUMINAMATH_CALUDE_stone_order_calculation_l1237_123767


namespace NUMINAMATH_CALUDE_stratified_sampling_medium_supermarkets_l1237_123761

theorem stratified_sampling_medium_supermarkets :
  let total_supermarkets : ℕ := 200 + 400 + 1400
  let medium_supermarkets : ℕ := 400
  let sample_size : ℕ := 100
  (medium_supermarkets * sample_size) / total_supermarkets = 20 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_medium_supermarkets_l1237_123761


namespace NUMINAMATH_CALUDE_r_squared_perfect_fit_l1237_123719

/-- Linear regression model with zero error -/
structure LinearRegressionModel where
  n : ℕ
  x : Fin n → ℝ
  y : Fin n → ℝ
  a : ℝ
  b : ℝ
  h : ∀ i, y i = b * x i + a

/-- Coefficient of determination (R-squared) -/
def r_squared (model : LinearRegressionModel) : ℝ :=
  sorry

/-- Theorem: R-squared equals 1 for a perfect fit linear regression model -/
theorem r_squared_perfect_fit (model : LinearRegressionModel) :
  r_squared model = 1 :=
sorry

end NUMINAMATH_CALUDE_r_squared_perfect_fit_l1237_123719


namespace NUMINAMATH_CALUDE_least_positive_integer_divisible_by_four_primes_l1237_123786

theorem least_positive_integer_divisible_by_four_primes : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (p₁ p₂ p₃ p₄ : ℕ), Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧ 
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₃ ≠ p₄ ∧
    p₁ ∣ n ∧ p₂ ∣ n ∧ p₃ ∣ n ∧ p₄ ∣ n) ∧
  (∀ m : ℕ, m > 0 → 
    (∃ (q₁ q₂ q₃ q₄ : ℕ), Prime q₁ ∧ Prime q₂ ∧ Prime q₃ ∧ Prime q₄ ∧ 
      q₁ ≠ q₂ ∧ q₁ ≠ q₃ ∧ q₁ ≠ q₄ ∧ q₂ ≠ q₃ ∧ q₂ ≠ q₄ ∧ q₃ ≠ q₄ ∧
      q₁ ∣ m ∧ q₂ ∣ m ∧ q₃ ∣ m ∧ q₄ ∣ m) → 
    n ≤ m) ∧
  n = 210 := by
sorry

end NUMINAMATH_CALUDE_least_positive_integer_divisible_by_four_primes_l1237_123786


namespace NUMINAMATH_CALUDE_side_c_value_l1237_123717

-- Define the triangle ABC
def triangle_ABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  -- Add conditions for a valid triangle if necessary
  true

-- State the theorem
theorem side_c_value 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h_triangle : triangle_ABC a b c A B C)
  (h_b : b = 3)
  (h_a : a = Real.sqrt 3)
  (h_A : A = 30 * π / 180) -- Convert 30° to radians
  : c = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_side_c_value_l1237_123717


namespace NUMINAMATH_CALUDE_probability_all_colors_l1237_123794

/-- The probability of selecting 4 balls of all three colors from 11 balls (3 red, 3 black, 5 white) -/
theorem probability_all_colors (total : ℕ) (red : ℕ) (black : ℕ) (white : ℕ) (select : ℕ) : 
  total = 11 → red = 3 → black = 3 → white = 5 → select = 4 →
  (Nat.choose red 2 * Nat.choose black 1 * Nat.choose white 1 +
   Nat.choose black 2 * Nat.choose red 1 * Nat.choose white 1 +
   Nat.choose white 2 * Nat.choose red 1 * Nat.choose black 1) / 
  Nat.choose total select = 6 / 11 := by
sorry

end NUMINAMATH_CALUDE_probability_all_colors_l1237_123794


namespace NUMINAMATH_CALUDE_probability_even_sum_and_same_number_l1237_123783

/-- A fair six-sided die -/
def Die : Type := Fin 6

/-- The outcome of rolling two dice -/
def RollOutcome : Type := Die × Die

/-- The total number of possible outcomes when rolling two dice -/
def totalOutcomes : Nat := 36

/-- Predicate for checking if a roll outcome has an even sum -/
def hasEvenSum (roll : RollOutcome) : Prop :=
  (roll.1.val + 1 + roll.2.val + 1) % 2 = 0

/-- Predicate for checking if both dice show the same number -/
def hasSameNumber (roll : RollOutcome) : Prop :=
  roll.1 = roll.2

/-- The set of favorable outcomes (even sum and same number) -/
def favorableOutcomes : Finset RollOutcome :=
  sorry

/-- The number of favorable outcomes -/
def numFavorableOutcomes : Nat :=
  favorableOutcomes.card

theorem probability_even_sum_and_same_number :
  (numFavorableOutcomes : ℚ) / totalOutcomes = 1 / 12 :=
sorry

end NUMINAMATH_CALUDE_probability_even_sum_and_same_number_l1237_123783


namespace NUMINAMATH_CALUDE_jeanine_pencils_proof_l1237_123724

/-- The number of pencils Jeanine bought initially -/
def jeanine_pencils : ℕ := 18

/-- The number of pencils Clare bought -/
def clare_pencils : ℕ := jeanine_pencils / 2

/-- The number of pencils Jeanine has after giving some to Abby -/
def jeanine_remaining_pencils : ℕ := (2 * jeanine_pencils) / 3

theorem jeanine_pencils_proof :
  (clare_pencils = jeanine_pencils / 2) ∧
  (jeanine_remaining_pencils = (2 * jeanine_pencils) / 3) ∧
  (jeanine_remaining_pencils = clare_pencils + 3) →
  jeanine_pencils = 18 := by
sorry

end NUMINAMATH_CALUDE_jeanine_pencils_proof_l1237_123724


namespace NUMINAMATH_CALUDE_frank_weed_eating_earnings_l1237_123700

/-- Calculates the amount Frank made weed eating given his lawn mowing earnings, weekly spending, and duration of savings. -/
def weed_eating_earnings (lawn_mowing_earnings weekly_spending duration_weeks : ℕ) : ℕ :=
  weekly_spending * duration_weeks - lawn_mowing_earnings

theorem frank_weed_eating_earnings :
  weed_eating_earnings 5 7 9 = 58 := by
  sorry

end NUMINAMATH_CALUDE_frank_weed_eating_earnings_l1237_123700


namespace NUMINAMATH_CALUDE_f_is_even_f_increasing_on_nonneg_l1237_123729

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 1

-- Theorem for the parity of the function (even function)
theorem f_is_even : ∀ x : ℝ, f (-x) = f x := by sorry

-- Theorem for monotonic increase on [0, +∞)
theorem f_increasing_on_nonneg : ∀ x y : ℝ, 0 ≤ x → x < y → f x < f y := by sorry

end NUMINAMATH_CALUDE_f_is_even_f_increasing_on_nonneg_l1237_123729


namespace NUMINAMATH_CALUDE_percentage_problem_l1237_123778

theorem percentage_problem (x : ℝ) (h : 0.3 * x = 120) : 0.4 * x = 160 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1237_123778


namespace NUMINAMATH_CALUDE_find_t_value_l1237_123718

theorem find_t_value (s t : ℚ) 
  (eq1 : 12 * s + 7 * t = 154)
  (eq2 : s = 2 * t - 3) : 
  t = 190 / 31 := by
sorry

end NUMINAMATH_CALUDE_find_t_value_l1237_123718


namespace NUMINAMATH_CALUDE_no_solution_system_l1237_123760

theorem no_solution_system :
  ¬ ∃ (x y : ℝ), (3 * x - 4 * y = 8) ∧ (6 * x - 8 * y = 12) := by
sorry

end NUMINAMATH_CALUDE_no_solution_system_l1237_123760


namespace NUMINAMATH_CALUDE_cody_bill_is_99_l1237_123745

/-- Represents the cost and quantity of tickets for Cody's order -/
structure TicketOrder where
  childPrice : ℚ
  adultPrice : ℚ
  childCount : ℕ
  adultCount : ℕ

/-- Calculates the total bill for a given ticket order -/
def totalBill (order : TicketOrder) : ℚ :=
  order.childPrice * order.childCount + order.adultPrice * order.adultCount

/-- Theorem stating that Cody's total bill is $99.00 given the problem conditions -/
theorem cody_bill_is_99 : ∃ (order : TicketOrder),
  order.childPrice = 7.5 ∧
  order.adultPrice = 12 ∧
  order.childCount = order.adultCount + 8 ∧
  order.childCount + order.adultCount = 12 ∧
  totalBill order = 99 := by
  sorry

end NUMINAMATH_CALUDE_cody_bill_is_99_l1237_123745


namespace NUMINAMATH_CALUDE_unit_vector_parallel_to_a_unit_vector_perpendicular_to_a_rotated_vector_e_l1237_123738

-- Define the vector a
def a : ℝ × ℝ := (3, -4)

-- Theorem for the unit vector b parallel to a
theorem unit_vector_parallel_to_a :
  ∃ b : ℝ × ℝ, (b.1 = 3/5 ∧ b.2 = -4/5) ∨ (b.1 = -3/5 ∧ b.2 = 4/5) ∧
  (b.1 * a.1 + b.2 * a.2)^2 = (b.1^2 + b.2^2) * (a.1^2 + a.2^2) ∧
  b.1^2 + b.2^2 = 1 :=
sorry

-- Theorem for the unit vector c perpendicular to a
theorem unit_vector_perpendicular_to_a :
  ∃ c : ℝ × ℝ, (c.1 = 4/5 ∧ c.2 = 3/5) ∨ (c.1 = -4/5 ∧ c.2 = -3/5) ∧
  c.1 * a.1 + c.2 * a.2 = 0 ∧
  c.1^2 + c.2^2 = 1 :=
sorry

-- Theorem for the vector e obtained by rotating a 45° counterclockwise
theorem rotated_vector_e :
  ∃ e : ℝ × ℝ, e.1 = 7 * Real.sqrt 2 / 2 ∧ e.2 = - Real.sqrt 2 / 2 ∧
  e.1^2 + e.2^2 = a.1^2 + a.2^2 ∧
  e.1 * a.1 + e.2 * a.2 = Real.sqrt ((a.1^2 + a.2^2)^2 / 2) :=
sorry

end NUMINAMATH_CALUDE_unit_vector_parallel_to_a_unit_vector_perpendicular_to_a_rotated_vector_e_l1237_123738


namespace NUMINAMATH_CALUDE_breakfast_cost_theorem_l1237_123737

/-- The cost of each meal in Herman's breakfast purchases -/
def meal_cost (people : ℕ) (days_per_week : ℕ) (weeks : ℕ) (total_spent : ℕ) : ℚ :=
  total_spent / (people * days_per_week * weeks)

/-- Theorem stating that the meal cost is $4 given the problem conditions -/
theorem breakfast_cost_theorem :
  meal_cost 4 5 16 1280 = 4 := by
  sorry

end NUMINAMATH_CALUDE_breakfast_cost_theorem_l1237_123737


namespace NUMINAMATH_CALUDE_area_of_triangle_ABC_l1237_123759

/-- A square surrounded by four identical regular triangles -/
structure SquareWithTriangles where
  /-- Side length of the square -/
  squareSide : ℝ
  /-- The square has side length 2 -/
  squareSideIs2 : squareSide = 2
  /-- Side length of the surrounding triangles that touches the square -/
  triangleSide : ℝ
  /-- The triangle side that touches the square is equal to the square side -/
  triangleSideEqSquareSide : triangleSide = squareSide
  /-- The surrounding triangles are regular -/
  trianglesAreRegular : True
  /-- The surrounding triangles are symmetrically placed -/
  trianglesAreSymmetric : True

/-- Triangle ABC formed by connecting midpoints of outer sides of surrounding triangles -/
def TriangleABC (swt : SquareWithTriangles) : Set (ℝ × ℝ) := sorry

/-- The area of Triangle ABC -/
def areaOfTriangleABC (swt : SquareWithTriangles) : ℝ := sorry

/-- Theorem stating that the area of Triangle ABC is √3/2 -/
theorem area_of_triangle_ABC (swt : SquareWithTriangles) : 
  areaOfTriangleABC swt = Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_area_of_triangle_ABC_l1237_123759


namespace NUMINAMATH_CALUDE_sqrt_less_than_linear_approx_l1237_123723

theorem sqrt_less_than_linear_approx (x : ℝ) (hx : x > 0) : 
  Real.sqrt (1 + x) < 1 + x / 2 := by
sorry

end NUMINAMATH_CALUDE_sqrt_less_than_linear_approx_l1237_123723


namespace NUMINAMATH_CALUDE_f_properties_l1237_123735

/-- The quadratic function f(x) = x^2 + ax + 1 --/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 1

/-- Theorem stating the properties of the function f --/
theorem f_properties (a : ℝ) :
  (∃ (s : Set ℝ), ∀ x, f a x > 0 ↔ x ∈ s) ∧
  (∀ x > 0, f a x ≥ 0) ↔ a ≥ -2 :=
sorry

end NUMINAMATH_CALUDE_f_properties_l1237_123735


namespace NUMINAMATH_CALUDE_sum_of_two_numbers_l1237_123784

theorem sum_of_two_numbers (a b : ℕ) : a = 22 ∧ b = a - 10 → a + b = 34 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_numbers_l1237_123784


namespace NUMINAMATH_CALUDE_books_given_to_friend_l1237_123765

/-- Given that Paul initially had 134 books, sold 27 books, and was left with 68 books
    after giving some to his friend and selling in the garage sale,
    prove that the number of books Paul gave to his friend is 39. -/
theorem books_given_to_friend :
  ∀ (initial_books sold_books remaining_books books_to_friend : ℕ),
    initial_books = 134 →
    sold_books = 27 →
    remaining_books = 68 →
    initial_books - sold_books - books_to_friend = remaining_books →
    books_to_friend = 39 := by
  sorry

end NUMINAMATH_CALUDE_books_given_to_friend_l1237_123765


namespace NUMINAMATH_CALUDE_upstairs_vacuuming_time_l1237_123750

/-- Represents the vacuuming problem with given conditions -/
def VacuumingProblem (downstairs upstairs total : ℕ) : Prop :=
  upstairs = 2 * downstairs + 5 ∧ 
  downstairs + upstairs = total ∧
  total = 38

/-- Proves that given the conditions, the time to vacuum upstairs is 27 minutes -/
theorem upstairs_vacuuming_time :
  ∀ downstairs upstairs total, 
  VacuumingProblem downstairs upstairs total → 
  upstairs = 27 := by
  sorry

end NUMINAMATH_CALUDE_upstairs_vacuuming_time_l1237_123750


namespace NUMINAMATH_CALUDE_f_increasing_f_two_zeros_l1237_123732

/-- The function f(x) = 2|x+1| + ax -/
def f (a : ℝ) (x : ℝ) : ℝ := 2 * abs (x + 1) + a * x

/-- Theorem: f(x) is increasing on ℝ when a > 2 -/
theorem f_increasing (a : ℝ) (h : a > 2) :
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f a x₁ < f a x₂ :=
sorry

/-- Theorem: f(x) has exactly two zeros if and only if a ∈ (0,2) -/
theorem f_two_zeros (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0 ∧ ∀ x : ℝ, f a x = 0 → x = x₁ ∨ x = x₂) ↔
  0 < a ∧ a < 2 :=
sorry

end NUMINAMATH_CALUDE_f_increasing_f_two_zeros_l1237_123732


namespace NUMINAMATH_CALUDE_water_trough_problem_l1237_123703

/-- Calculates the remaining water volume in a trough after a given number of days,
    given an initial volume and a constant daily evaporation rate. -/
def remaining_water_volume (initial_volume : ℝ) (evaporation_rate : ℝ) (days : ℝ) : ℝ :=
  initial_volume - evaporation_rate * days

/-- Proves that given an initial water volume of 300 gallons, with a constant evaporation rate
    of 1 gallon per day over 45 days and no additional water added or removed,
    the final water volume will be 255 gallons. -/
theorem water_trough_problem :
  remaining_water_volume 300 1 45 = 255 := by
  sorry


end NUMINAMATH_CALUDE_water_trough_problem_l1237_123703


namespace NUMINAMATH_CALUDE_max_value_2ac_minus_abc_l1237_123768

theorem max_value_2ac_minus_abc : 
  ∀ a b c : ℕ+, 
  a ≤ 7 → b ≤ 6 → c ≤ 4 → 
  (2 * a * c - a * b * c : ℤ) ≤ 28 ∧ 
  ∃ a' b' c' : ℕ+, a' ≤ 7 ∧ b' ≤ 6 ∧ c' ≤ 4 ∧ 2 * a' * c' - a' * b' * c' = 28 :=
sorry

end NUMINAMATH_CALUDE_max_value_2ac_minus_abc_l1237_123768


namespace NUMINAMATH_CALUDE_product_equals_32_l1237_123758

theorem product_equals_32 : 
  (1 / 4) * 8 * (1 / 16) * 32 * (1 / 64) * 128 * (1 / 256) * 512 * (1 / 1024) * 2048 = 32 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_32_l1237_123758


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_relation_l1237_123712

/-- Given an ellipse and a hyperbola with coincident foci, prove that the semi-major axis of the ellipse is greater than that of the hyperbola, and the product of their eccentricities is greater than 1. -/
theorem ellipse_hyperbola_relation (m n : ℝ) (e₁ e₂ : ℝ) : 
  m > 1 →
  n > 0 →
  (∀ x y : ℝ, x^2 / m^2 + y^2 = 1 ↔ x^2 / n^2 - y^2 = 1) →
  e₁^2 = (m^2 - 1) / m^2 →
  e₂^2 = (n^2 + 1) / n^2 →
  m > n ∧ e₁ * e₂ > 1 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_relation_l1237_123712


namespace NUMINAMATH_CALUDE_triangle_heights_existence_l1237_123711

/-- Check if a triangle with given heights exists -/
def triangle_exists (h₁ h₂ h₃ : ℝ) : Prop :=
  ∃ a b c : ℝ, 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a + b > c ∧ b + c > a ∧ c + a > b ∧
    h₁ * a = h₂ * b ∧ h₂ * b = h₃ * c

theorem triangle_heights_existence :
  (¬ triangle_exists 2 3 6) ∧ (triangle_exists 2 3 5) := by
  sorry


end NUMINAMATH_CALUDE_triangle_heights_existence_l1237_123711


namespace NUMINAMATH_CALUDE_circle_to_hyperbola_l1237_123733

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 5*y + 4 = 0

-- Define the intersection points of circle C with coordinate axes
def intersection_points (C : (ℝ → ℝ → Prop)) : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), (x = 0 ∨ y = 0) ∧ C x y ∧ p = (x, y)}

-- Define the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop := (y - 1)^2 / 1 - x^2 / 15 = 1

-- Theorem statement
theorem circle_to_hyperbola :
  ∀ (focus vertex : ℝ × ℝ),
    focus ∈ intersection_points circle_C →
    vertex ∈ intersection_points circle_C →
    focus ≠ vertex →
    (∀ x y : ℝ, hyperbola_equation x y ↔
      ∃ (a b : ℝ),
        a > 0 ∧ b > 0 ∧
        (y - vertex.2)^2 / a^2 - (x - vertex.1)^2 / b^2 = 1 ∧
        (focus.1 - vertex.1)^2 + (focus.2 - vertex.2)^2 = a^2 + b^2) :=
sorry

end NUMINAMATH_CALUDE_circle_to_hyperbola_l1237_123733


namespace NUMINAMATH_CALUDE_second_number_value_l1237_123757

theorem second_number_value (A B : ℝ) (h1 : A = 200) (h2 : 0.3 * A = 0.6 * B + 30) : B = 50 := by
  sorry

end NUMINAMATH_CALUDE_second_number_value_l1237_123757


namespace NUMINAMATH_CALUDE_train_length_l1237_123731

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 60 → time = 3 → ∃ (length : ℝ), abs (length - 50.01) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_train_length_l1237_123731


namespace NUMINAMATH_CALUDE_cube_roll_no_90_degree_rotation_l1237_123725

/-- Represents a cube on a plane -/
structure Cube where
  position : ℝ × ℝ × ℝ
  top_face : Fin 6
  orientation : ℕ

/-- Represents a sequence of cube rolls -/
def RollSequence := List (Fin 4)

/-- Applies a sequence of rolls to a cube -/
def apply_rolls (c : Cube) (rolls : RollSequence) : Cube :=
  sorry

/-- Checks if the cube is in its initial position -/
def is_initial_position (initial : Cube) (final : Cube) : Prop :=
  initial.position = final.position ∧ initial.top_face = final.top_face

/-- Theorem: A cube rolled back to its initial position cannot have its top face rotated by 90 degrees -/
theorem cube_roll_no_90_degree_rotation 
  (c : Cube) (rolls : RollSequence) : 
  let c' := apply_rolls c rolls
  is_initial_position c c' → c.orientation ≠ (c'.orientation + 1) % 4 :=
sorry

end NUMINAMATH_CALUDE_cube_roll_no_90_degree_rotation_l1237_123725


namespace NUMINAMATH_CALUDE_rectangle_area_diagonal_l1237_123751

/-- Theorem: Area of a rectangle with length-to-width ratio 3:2 and diagonal d --/
theorem rectangle_area_diagonal (length width diagonal : ℝ) 
  (h_ratio : length / width = 3 / 2)
  (h_diagonal : length^2 + width^2 = diagonal^2) :
  length * width = (6/13) * diagonal^2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_diagonal_l1237_123751


namespace NUMINAMATH_CALUDE_cupcake_distribution_l1237_123708

/-- Represents the number of cupcakes in a pack --/
inductive PackSize
  | five : PackSize
  | ten : PackSize
  | fifteen : PackSize
  | twenty : PackSize

/-- Returns the number of cupcakes in a pack --/
def packSizeToInt (p : PackSize) : Nat :=
  match p with
  | PackSize.five => 5
  | PackSize.ten => 10
  | PackSize.fifteen => 15
  | PackSize.twenty => 20

/-- Calculates the total number of cupcakes from a given number of packs --/
def totalCupcakes (packSize : PackSize) (numPacks : Nat) : Nat :=
  (packSizeToInt packSize) * numPacks

/-- Represents Jean's initial purchase --/
def initialPurchase : Nat :=
  totalCupcakes PackSize.fifteen 4 + totalCupcakes PackSize.twenty 2

/-- The number of children in the orphanage --/
def numChildren : Nat := 220

/-- The theorem to prove --/
theorem cupcake_distribution :
  totalCupcakes PackSize.ten 8 + totalCupcakes PackSize.five 8 + initialPurchase = numChildren := by
  sorry

end NUMINAMATH_CALUDE_cupcake_distribution_l1237_123708


namespace NUMINAMATH_CALUDE_rectangle_area_l1237_123743

-- Define the rectangle
structure Rectangle where
  breadth : ℝ
  length : ℝ
  diagonal : ℝ

-- Define the conditions
def rectangleConditions (r : Rectangle) : Prop :=
  r.length = 3 * r.breadth ∧ r.diagonal = 20

-- Define the area function
def area (r : Rectangle) : ℝ :=
  r.length * r.breadth

-- Theorem statement
theorem rectangle_area (r : Rectangle) (h : rectangleConditions r) : area r = 120 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l1237_123743


namespace NUMINAMATH_CALUDE_sum_prod_nonzero_digits_equals_46_pow_2009_l1237_123756

/-- The number of digits in the problem -/
def n : ℕ := 2009

/-- Calculate the product of non-zero digits for a given natural number -/
def prod_nonzero_digits (k : ℕ) : ℕ := sorry

/-- Sum of products of non-zero digits for integers from 1 to 10^n -/
def sum_prod_nonzero_digits (n : ℕ) : ℕ := sorry

/-- Theorem stating the sum of products of non-zero digits for integers from 1 to 10^2009 -/
theorem sum_prod_nonzero_digits_equals_46_pow_2009 :
  sum_prod_nonzero_digits n = 46^n := by sorry

end NUMINAMATH_CALUDE_sum_prod_nonzero_digits_equals_46_pow_2009_l1237_123756


namespace NUMINAMATH_CALUDE_min_value_on_circle_l1237_123788

theorem min_value_on_circle (x y : ℝ) :
  (x - 1)^2 + (y + 2)^2 = 4 →
  ∃ (S : ℝ), S = 3*x - y ∧ S ≥ -5 - 2*Real.sqrt 10 ∧
  ∀ (S' : ℝ), (∃ (x' y' : ℝ), (x' - 1)^2 + (y' + 2)^2 = 4 ∧ S' = 3*x' - y') →
  S' ≥ -5 - 2*Real.sqrt 10 :=
sorry

end NUMINAMATH_CALUDE_min_value_on_circle_l1237_123788


namespace NUMINAMATH_CALUDE_journey_time_ratio_l1237_123746

theorem journey_time_ratio (speed_to_sf : ℝ) (avg_speed : ℝ) :
  speed_to_sf = 48 →
  avg_speed = 32 →
  (1 / avg_speed - 1 / speed_to_sf) / (1 / speed_to_sf) = 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_journey_time_ratio_l1237_123746


namespace NUMINAMATH_CALUDE_smallest_right_triangle_area_l1237_123716

theorem smallest_right_triangle_area (a b : ℝ) (ha : a = 6) (hb : b = 8) :
  ∀ (c : ℝ), c > 0 → a^2 + b^2 = c^2 → (1/2) * a * b ≤ 24 :=
by sorry

end NUMINAMATH_CALUDE_smallest_right_triangle_area_l1237_123716


namespace NUMINAMATH_CALUDE_inequality_two_integer_solutions_l1237_123782

theorem inequality_two_integer_solutions (k : ℝ) : 
  (∃ (x y : ℕ), x ≠ y ∧ 
    (k * (x : ℝ)^2 ≤ Real.log x + 1) ∧ 
    (k * (y : ℝ)^2 ≤ Real.log y + 1) ∧
    (∀ (z : ℕ), z ≠ x ∧ z ≠ y → k * (z : ℝ)^2 > Real.log z + 1)) →
  ((Real.log 3 + 1) / 9 < k ∧ k ≤ (Real.log 2 + 1) / 4) :=
by sorry

end NUMINAMATH_CALUDE_inequality_two_integer_solutions_l1237_123782


namespace NUMINAMATH_CALUDE_area_of_triangle_DBG_l1237_123721

-- Define the triangle and squares
structure RightTriangle :=
  (A B C : ℝ × ℝ)
  (is_right_angle : (A.1 - B.1) * (C.1 - B.1) + (A.2 - B.2) * (C.2 - B.2) = 0)

def square_area (side : ℝ) : ℝ := side ^ 2

-- State the theorem
theorem area_of_triangle_DBG 
  (triangle : RightTriangle)
  (area_ABDE : square_area (Real.sqrt ((triangle.A.1 - triangle.B.1)^2 + (triangle.A.2 - triangle.B.2)^2)) = 8)
  (area_BCFG : square_area (Real.sqrt ((triangle.B.1 - triangle.C.1)^2 + (triangle.B.2 - triangle.C.2)^2)) = 26) :
  let D : ℝ × ℝ := (triangle.A.1 + (triangle.B.2 - triangle.A.2), triangle.A.2 - (triangle.B.1 - triangle.A.1))
  let G : ℝ × ℝ := (triangle.B.1 + (triangle.C.2 - triangle.B.2), triangle.B.2 - (triangle.C.1 - triangle.B.1))
  (1/2) * Real.sqrt ((D.1 - triangle.B.1)^2 + (D.2 - triangle.B.2)^2) * 
         Real.sqrt ((G.1 - triangle.B.1)^2 + (G.2 - triangle.B.2)^2) = 2 * Real.sqrt 13 :=
by sorry

end NUMINAMATH_CALUDE_area_of_triangle_DBG_l1237_123721


namespace NUMINAMATH_CALUDE_six_digit_divisible_by_396_l1237_123774

theorem six_digit_divisible_by_396 : ∃ (x y z : ℕ), 
  x < 10 ∧ y < 10 ∧ z < 10 ∧ 
  (243000 + 100 * x + 10 * y + z) % 396 = 0 := by
  sorry

end NUMINAMATH_CALUDE_six_digit_divisible_by_396_l1237_123774


namespace NUMINAMATH_CALUDE_orthogonal_vectors_l1237_123799

/-- Given non-zero plane vectors a and b satisfying |a + b| = |a - b|, prove that a ⋅ b = 0 -/
theorem orthogonal_vectors (a b : ℝ × ℝ) (ha : a ≠ (0, 0)) (hb : b ≠ (0, 0)) 
  (h : ‖a + b‖ = ‖a - b‖) : a • b = 0 := by
  sorry

end NUMINAMATH_CALUDE_orthogonal_vectors_l1237_123799


namespace NUMINAMATH_CALUDE_sequence_terms_equal_twenty_l1237_123726

def a (n : ℕ) : ℤ := n^2 - 14*n + 65

theorem sequence_terms_equal_twenty :
  (∀ n : ℕ, a n = 20 ↔ n = 5 ∨ n = 9) :=
sorry

end NUMINAMATH_CALUDE_sequence_terms_equal_twenty_l1237_123726
