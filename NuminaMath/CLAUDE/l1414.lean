import Mathlib

namespace NUMINAMATH_CALUDE_probability_inner_circle_l1414_141454

theorem probability_inner_circle (R : ℝ) (r : ℝ) (h1 : R = 6) (h2 : r = 2) :
  (π * r^2) / (π * R^2) = 1 / 9 :=
sorry

end NUMINAMATH_CALUDE_probability_inner_circle_l1414_141454


namespace NUMINAMATH_CALUDE_cubic_zeros_sum_less_than_two_l1414_141439

noncomputable def f (a b c x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x

noncomputable def F (a b c x : ℝ) : ℝ := f a b c x - x * Real.exp (-x)

theorem cubic_zeros_sum_less_than_two (a b c : ℝ) (ha : a ≠ 0) 
    (h1 : 6 * a + b = 0) (h2 : f a b c 1 = 4 * a) :
    ∃ x₁ x₂ x₃ : ℝ, x₁ < x₂ ∧ x₂ < x₃ ∧ 
    0 ≤ x₁ ∧ x₃ ≤ 3 ∧
    F a b c x₁ = 0 ∧ F a b c x₂ = 0 ∧ F a b c x₃ = 0 ∧
    x₁ + x₂ + x₃ < 2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_zeros_sum_less_than_two_l1414_141439


namespace NUMINAMATH_CALUDE_initial_stock_calculation_l1414_141478

/-- The number of toys sold in the first week -/
def toys_sold_first_week : ℕ := 38

/-- The number of toys sold in the second week -/
def toys_sold_second_week : ℕ := 26

/-- The number of toys left after two weeks -/
def toys_left : ℕ := 19

/-- The initial number of toys in stock -/
def initial_stock : ℕ := toys_sold_first_week + toys_sold_second_week + toys_left

theorem initial_stock_calculation :
  initial_stock = 83 := by sorry

end NUMINAMATH_CALUDE_initial_stock_calculation_l1414_141478


namespace NUMINAMATH_CALUDE_billionth_term_is_16_l1414_141494

def sequence_term (n : ℕ) : ℕ :=
  if n = 0 then 112002
  else
    let prev := sequence_term (n - 1)
    prev + 5 * (prev % 10) - 10 * ((prev % 10) / 10)

def is_cyclic (seq : ℕ → ℕ) (cycle_length : ℕ) : Prop :=
  ∀ n : ℕ, seq (n + cycle_length) = seq n

theorem billionth_term_is_16 :
  is_cyclic sequence_term 42 →
  sequence_term (10^9 % 42) = 16 →
  sequence_term (10^9) = 16 :=
by sorry

end NUMINAMATH_CALUDE_billionth_term_is_16_l1414_141494


namespace NUMINAMATH_CALUDE_existence_of_non_dividing_sum_l1414_141418

theorem existence_of_non_dividing_sum (n : ℕ) (a : Fin n → ℕ+) (h_n : n ≥ 3) 
  (h_distinct : ∀ i j, i ≠ j → a i ≠ a j) : 
  ∃ i j, i ≠ j ∧ ∀ k, ¬((a i + a j : ℕ) ∣ (3 * (a k : ℕ))) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_non_dividing_sum_l1414_141418


namespace NUMINAMATH_CALUDE_smallest_k_with_remainder_one_k_400_satisfies_conditions_smallest_k_is_400_l1414_141412

theorem smallest_k_with_remainder_one (k : ℕ) : k > 1 ∧ 
  k % 19 = 1 ∧ k % 7 = 1 ∧ k % 3 = 1 → k ≥ 400 := by
  sorry

theorem k_400_satisfies_conditions : 
  400 > 1 ∧ 400 % 19 = 1 ∧ 400 % 7 = 1 ∧ 400 % 3 = 1 := by
  sorry

theorem smallest_k_is_400 : 
  ∃! k : ℕ, k > 1 ∧ k % 19 = 1 ∧ k % 7 = 1 ∧ k % 3 = 1 ∧ 
  ∀ m : ℕ, (m > 1 ∧ m % 19 = 1 ∧ m % 7 = 1 ∧ m % 3 = 1) → k ≤ m := by
  sorry

end NUMINAMATH_CALUDE_smallest_k_with_remainder_one_k_400_satisfies_conditions_smallest_k_is_400_l1414_141412


namespace NUMINAMATH_CALUDE_find_y_l1414_141470

theorem find_y (x : ℝ) (y : ℝ) (h1 : x^(2*y) = 16) (h2 : x = 8) : y = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_find_y_l1414_141470


namespace NUMINAMATH_CALUDE_power_of_four_l1414_141498

theorem power_of_four (k : ℝ) : (4 : ℝ) ^ (2 * k + 2) = 400 → (4 : ℝ) ^ k = 5 := by
  sorry

end NUMINAMATH_CALUDE_power_of_four_l1414_141498


namespace NUMINAMATH_CALUDE_number_of_tippers_l1414_141440

def lawn_price : ℕ := 33
def lawns_mowed : ℕ := 16
def tip_amount : ℕ := 10
def total_earnings : ℕ := 558

theorem number_of_tippers : ℕ :=
  by
    sorry

end NUMINAMATH_CALUDE_number_of_tippers_l1414_141440


namespace NUMINAMATH_CALUDE_min_packs_for_120_cans_l1414_141400

/-- Represents the number of cans in each pack size -/
inductive PackSize
  | small : PackSize
  | medium : PackSize
  | large : PackSize

/-- Returns the number of cans in a given pack size -/
def cansInPack (p : PackSize) : ℕ :=
  match p with
  | PackSize.small => 8
  | PackSize.medium => 16
  | PackSize.large => 32

/-- Represents a combination of packs -/
structure PackCombination where
  small : ℕ
  medium : ℕ
  large : ℕ

/-- Calculates the total number of cans in a pack combination -/
def totalCans (c : PackCombination) : ℕ :=
  c.small * cansInPack PackSize.small +
  c.medium * cansInPack PackSize.medium +
  c.large * cansInPack PackSize.large

/-- Calculates the total number of packs in a pack combination -/
def totalPacks (c : PackCombination) : ℕ :=
  c.small + c.medium + c.large

/-- Theorem: The minimum number of packs to buy 120 cans is 5 -/
theorem min_packs_for_120_cans :
  ∃ (c : PackCombination),
    totalCans c = 120 ∧
    totalPacks c = 5 ∧
    (∀ (c' : PackCombination), totalCans c' = 120 → totalPacks c' ≥ 5) :=
by
  sorry

end NUMINAMATH_CALUDE_min_packs_for_120_cans_l1414_141400


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1414_141413

theorem imaginary_part_of_complex_fraction (i : ℂ) (h : i^2 = -1) :
  let z := (3 - 2*i^3) / (1 + i)
  Complex.im z = -1/2 := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1414_141413


namespace NUMINAMATH_CALUDE_charles_reading_time_l1414_141497

-- Define the parameters of the problem
def total_pages : ℕ := 96
def pages_per_day : ℕ := 8

-- Define the function to calculate the number of days
def days_to_finish (total : ℕ) (per_day : ℕ) : ℕ := total / per_day

-- Theorem statement
theorem charles_reading_time : days_to_finish total_pages pages_per_day = 12 := by
  sorry

end NUMINAMATH_CALUDE_charles_reading_time_l1414_141497


namespace NUMINAMATH_CALUDE_total_interest_calculation_l1414_141467

/-- Calculate simple interest -/
def simple_interest (principal : ℕ) (rate : ℕ) (time : ℕ) : ℕ :=
  principal * rate * time / 100

/-- Problem statement -/
theorem total_interest_calculation :
  let principal_B : ℕ := 5000
  let principal_C : ℕ := 3000
  let rate : ℕ := 10
  let time_B : ℕ := 2
  let time_C : ℕ := 4
  let interest_B := simple_interest principal_B rate time_B
  let interest_C := simple_interest principal_C rate time_C
  interest_B + interest_C = 2200 := by
sorry

end NUMINAMATH_CALUDE_total_interest_calculation_l1414_141467


namespace NUMINAMATH_CALUDE_debby_text_messages_l1414_141453

theorem debby_text_messages 
  (total_messages : ℕ) 
  (before_noon_messages : ℕ) 
  (h1 : total_messages = 39) 
  (h2 : before_noon_messages = 21) : 
  total_messages - before_noon_messages = 18 := by
sorry

end NUMINAMATH_CALUDE_debby_text_messages_l1414_141453


namespace NUMINAMATH_CALUDE_shortest_distance_parabola_to_line_l1414_141484

/-- The shortest distance from a point on the parabola y = x^2 to the line 2x - y = 4 -/
theorem shortest_distance_parabola_to_line :
  let parabola := {p : ℝ × ℝ | p.2 = p.1^2}
  let line := {p : ℝ × ℝ | 2 * p.1 - p.2 = 4}
  let distance (p : ℝ × ℝ) := |2 * p.1 - p.2 - 4| / Real.sqrt 5
  (∀ p ∈ parabola, distance p ≥ 3 * Real.sqrt 5 / 5) ∧
  (∃ p ∈ parabola, distance p = 3 * Real.sqrt 5 / 5) :=
by sorry


end NUMINAMATH_CALUDE_shortest_distance_parabola_to_line_l1414_141484


namespace NUMINAMATH_CALUDE_unique_divisible_digit_l1414_141458

def is_single_digit (n : ℕ) : Prop := n < 10

def number_with_A (A : ℕ) : ℕ := 653802 * 10 + A

theorem unique_divisible_digit :
  ∃! A : ℕ, is_single_digit A ∧
    (∀ d : ℕ, d ∈ [2, 3, 4, 6, 8, 9, 25] → (number_with_A A) % d = 0) :=
by sorry

end NUMINAMATH_CALUDE_unique_divisible_digit_l1414_141458


namespace NUMINAMATH_CALUDE_M_equals_singleton_l1414_141455

def M : Set (ℝ × ℝ) := {p | 2 * p.1 + p.2 = 2 ∧ p.1 - p.2 = 1}

theorem M_equals_singleton : M = {(1, 0)} := by sorry

end NUMINAMATH_CALUDE_M_equals_singleton_l1414_141455


namespace NUMINAMATH_CALUDE_monomial_properties_l1414_141492

def monomial_coefficient (a : ℤ) (b c : ℕ) : ℤ := -2

def monomial_degree (a : ℤ) (b c : ℕ) : ℕ := 1 + b + c

theorem monomial_properties :
  let m := monomial_coefficient (-2) 2 4
  let n := monomial_degree (-2) 2 4
  m = -2 ∧ n = 7 := by sorry

end NUMINAMATH_CALUDE_monomial_properties_l1414_141492


namespace NUMINAMATH_CALUDE_log_equation_solution_l1414_141493

theorem log_equation_solution : ∃ x : ℝ, (Real.log x - Real.log 25) / 100 = -20 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l1414_141493


namespace NUMINAMATH_CALUDE_paper_distribution_l1414_141463

theorem paper_distribution (num_students : ℕ) (paper_per_student : ℕ) 
  (h1 : num_students = 230) 
  (h2 : paper_per_student = 15) : 
  num_students * paper_per_student = 3450 := by
  sorry

end NUMINAMATH_CALUDE_paper_distribution_l1414_141463


namespace NUMINAMATH_CALUDE_power_product_simplification_l1414_141428

theorem power_product_simplification :
  (10 ^ 0.4) * (10 ^ 0.6) * (10 ^ 0.3) * (10 ^ 0.2) * (10 ^ 0.5) = 100 := by
  sorry

end NUMINAMATH_CALUDE_power_product_simplification_l1414_141428


namespace NUMINAMATH_CALUDE_factorial_division_l1414_141436

theorem factorial_division :
  (10 : ℕ).factorial / (4 : ℕ).factorial = 151200 :=
by
  -- Given condition
  have h1 : (10 : ℕ).factorial = 3628800 := by sorry
  
  -- Proof
  sorry

end NUMINAMATH_CALUDE_factorial_division_l1414_141436


namespace NUMINAMATH_CALUDE_mother_triple_daughter_age_l1414_141408

/-- Represents the age difference between mother and daughter -/
def age_difference : ℕ := 42 - 8

/-- Represents the current age of the mother -/
def mother_age : ℕ := 42

/-- Represents the current age of the daughter -/
def daughter_age : ℕ := 8

/-- The number of years until the mother is three times as old as her daughter -/
def years_until_triple : ℕ := 9

theorem mother_triple_daughter_age :
  mother_age + years_until_triple = 3 * (daughter_age + years_until_triple) :=
sorry

end NUMINAMATH_CALUDE_mother_triple_daughter_age_l1414_141408


namespace NUMINAMATH_CALUDE_inequality_preserved_by_exponential_l1414_141433

theorem inequality_preserved_by_exponential (a b : ℝ) (h : a > b) :
  ∀ x : ℝ, a * (2 : ℝ)^x > b * (2 : ℝ)^x :=
by
  sorry

end NUMINAMATH_CALUDE_inequality_preserved_by_exponential_l1414_141433


namespace NUMINAMATH_CALUDE_counterexample_exists_l1414_141443

theorem counterexample_exists : ∃ n : ℕ, 15 ≤ n ∧ n ≤ 30 ∧ ¬(Nat.Prime n) ∧ Nat.Prime (n - 5) := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l1414_141443


namespace NUMINAMATH_CALUDE_sarah_final_toads_l1414_141421

-- Define the number of toads each person has
def tim_toads : ℕ := 30
def jim_toads : ℕ := tim_toads + 20
def sarah_initial_toads : ℕ := 2 * jim_toads

-- Define the number of toads Sarah gives away
def sarah_gives_away : ℕ := sarah_initial_toads / 4

-- Define the number of toads Sarah buys
def sarah_buys : ℕ := 15

-- Theorem to prove
theorem sarah_final_toads :
  sarah_initial_toads - sarah_gives_away + sarah_buys = 90 := by
  sorry

end NUMINAMATH_CALUDE_sarah_final_toads_l1414_141421


namespace NUMINAMATH_CALUDE_bug_return_probability_l1414_141404

/-- Probability of the bug being at the starting vertex after n moves -/
def Q : ℕ → ℚ
  | 0 => 1
  | n + 1 => 1/3 * (1 - Q n)

/-- The probability of the bug returning to its starting vertex on the eighth move is 547/2187 -/
theorem bug_return_probability : Q 8 = 547 / 2187 := by
  sorry

end NUMINAMATH_CALUDE_bug_return_probability_l1414_141404


namespace NUMINAMATH_CALUDE_friendly_parabola_symmetric_l1414_141451

/-- Represents a parabola of the form y = ax² + bx --/
structure Parabola where
  a : ℝ
  b : ℝ

/-- Defines the "friendly parabola" relationship between two parabolas --/
def is_friendly_parabola (L₁ L₂ : Parabola) : Prop :=
  L₂.a = -L₁.a ∧ 
  L₂.b = -2 * L₁.a * (-L₁.b / (2 * L₁.a)) + L₁.b

/-- Theorem: The "friendly parabola" relationship is symmetric --/
theorem friendly_parabola_symmetric (L₁ L₂ : Parabola) :
  is_friendly_parabola L₁ L₂ → is_friendly_parabola L₂ L₁ := by
  sorry


end NUMINAMATH_CALUDE_friendly_parabola_symmetric_l1414_141451


namespace NUMINAMATH_CALUDE_percentage_of_older_female_students_l1414_141437

/-- Represents the percentage of female students who are 25 years old or older -/
def P : ℝ := 30

theorem percentage_of_older_female_students :
  let total_students : ℝ := 100
  let male_percentage : ℝ := 40
  let female_percentage : ℝ := 100 - male_percentage
  let older_male_percentage : ℝ := 40
  let younger_probability : ℝ := 0.66
  
  (male_percentage / 100 * (100 - older_male_percentage) / 100 +
   female_percentage / 100 * (100 - P) / 100) * total_students = younger_probability * total_students :=
by sorry

end NUMINAMATH_CALUDE_percentage_of_older_female_students_l1414_141437


namespace NUMINAMATH_CALUDE_adams_initial_money_l1414_141476

/-- Adam's initial money problem -/
theorem adams_initial_money :
  ∀ (x : ℤ), (x - 2 + 5 = 8) → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_adams_initial_money_l1414_141476


namespace NUMINAMATH_CALUDE_spade_operation_l1414_141417

def spade (a b : ℝ) : ℝ := |a - b|

theorem spade_operation : (5 : ℝ) * (spade 2 (spade 6 9)) = 5 := by
  sorry

end NUMINAMATH_CALUDE_spade_operation_l1414_141417


namespace NUMINAMATH_CALUDE_jills_net_salary_l1414_141449

/-- Calculates the net monthly salary given the discretionary income ratio and remaining amount --/
def calculate_net_salary (discretionary_ratio : ℚ) (vacation_ratio : ℚ) (savings_ratio : ℚ) 
  (socializing_ratio : ℚ) (remaining_amount : ℚ) : ℚ :=
  remaining_amount / (discretionary_ratio * (1 - (vacation_ratio + savings_ratio + socializing_ratio)))

/-- Proves that given the specified conditions, Jill's net monthly salary is $3700 --/
theorem jills_net_salary :
  let discretionary_ratio : ℚ := 1/5
  let vacation_ratio : ℚ := 30/100
  let savings_ratio : ℚ := 20/100
  let socializing_ratio : ℚ := 35/100
  let remaining_amount : ℚ := 111
  calculate_net_salary discretionary_ratio vacation_ratio savings_ratio socializing_ratio remaining_amount = 3700 := by
  sorry

#eval calculate_net_salary (1/5) (30/100) (20/100) (35/100) 111

end NUMINAMATH_CALUDE_jills_net_salary_l1414_141449


namespace NUMINAMATH_CALUDE_min_value_on_circle_l1414_141407

theorem min_value_on_circle (x y : ℝ) :
  (x - 2)^2 + (y - 3)^2 = 1 →
  ∃ (z : ℝ), z = 14 - 2 * Real.sqrt 13 ∧ ∀ (a b : ℝ), (a - 2)^2 + (b - 3)^2 = 1 → x^2 + y^2 ≤ a^2 + b^2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_on_circle_l1414_141407


namespace NUMINAMATH_CALUDE_smallest_multiple_of_3_5_7_9_l1414_141419

theorem smallest_multiple_of_3_5_7_9 (n : ℕ) :
  (∀ m : ℕ, m > 0 ∧ 3 ∣ m ∧ 5 ∣ m ∧ 7 ∣ m ∧ 9 ∣ m → n ≤ m) ↔ n = 315 :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_3_5_7_9_l1414_141419


namespace NUMINAMATH_CALUDE_square_of_a_l1414_141445

theorem square_of_a (a b c d : ℕ+) 
  (h1 : a < b) (h2 : b ≤ c) (h3 : c < d)
  (h4 : a * d = b * c)
  (h5 : Real.sqrt d - Real.sqrt a ≤ 1) :
  ∃ (n : ℕ), a = n^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_a_l1414_141445


namespace NUMINAMATH_CALUDE_unique_satisfying_polynomial_l1414_141424

/-- A polynomial satisfying the given conditions -/
def SatisfyingPolynomial (P : ℝ → ℝ) : Prop :=
  (P 0 = 0) ∧ (∀ x, P (x^2 + 1) = P x^2 + 1)

/-- Theorem stating that the identity function is the only polynomial satisfying the conditions -/
theorem unique_satisfying_polynomial :
  ∀ P : ℝ → ℝ, SatisfyingPolynomial P → (∀ x, P x = x) :=
by sorry

end NUMINAMATH_CALUDE_unique_satisfying_polynomial_l1414_141424


namespace NUMINAMATH_CALUDE_monochromatic_state_reachable_final_color_independent_l1414_141473

/-- Represents the three possible colors of glass pieces -/
inductive Color
  | Red
  | Yellow
  | Blue

/-- Represents the state of glass pieces -/
structure GlassState :=
  (red : Nat)
  (yellow : Nat)
  (blue : Nat)
  (total : Nat)
  (total_eq : red + yellow + blue = total)

/-- Represents an operation on glass pieces -/
def perform_operation (state : GlassState) : GlassState :=
  sorry

/-- Theorem stating that it's always possible to reach a monochromatic state -/
theorem monochromatic_state_reachable (initial_state : GlassState) 
  (h : initial_state.total = 1987) :
  ∃ (final_state : GlassState) (c : Color), 
    (final_state.red = initial_state.total ∧ c = Color.Red) ∨
    (final_state.yellow = initial_state.total ∧ c = Color.Yellow) ∨
    (final_state.blue = initial_state.total ∧ c = Color.Blue) :=
  sorry

/-- Theorem stating that the final color is independent of operation order -/
theorem final_color_independent (initial_state : GlassState) 
  (h : initial_state.total = 1987) :
  ∀ (final_state1 final_state2 : GlassState) (c1 c2 : Color),
    ((final_state1.red = initial_state.total ∧ c1 = Color.Red) ∨
     (final_state1.yellow = initial_state.total ∧ c1 = Color.Yellow) ∨
     (final_state1.blue = initial_state.total ∧ c1 = Color.Blue)) →
    ((final_state2.red = initial_state.total ∧ c2 = Color.Red) ∨
     (final_state2.yellow = initial_state.total ∧ c2 = Color.Yellow) ∨
     (final_state2.blue = initial_state.total ∧ c2 = Color.Blue)) →
    c1 = c2 :=
  sorry

end NUMINAMATH_CALUDE_monochromatic_state_reachable_final_color_independent_l1414_141473


namespace NUMINAMATH_CALUDE_square_sum_equals_90_l1414_141426

theorem square_sum_equals_90 (x y : ℝ) (h1 : x + 3*y = 6) (h2 : x*y = -9) : 
  x^2 + 9*y^2 = 90 := by
sorry

end NUMINAMATH_CALUDE_square_sum_equals_90_l1414_141426


namespace NUMINAMATH_CALUDE_root_condition_implies_m_range_l1414_141423

theorem root_condition_implies_m_range (m : ℝ) : 
  (∃ x y : ℝ, x^2 - 2*m*x + 4 = 0 ∧ y^2 - 2*m*y + 4 = 0 ∧ x > 1 ∧ y < 1) →
  m > 5/2 :=
by sorry

end NUMINAMATH_CALUDE_root_condition_implies_m_range_l1414_141423


namespace NUMINAMATH_CALUDE_expression_value_l1414_141468

theorem expression_value : 7^3 - 4 * 7^2 + 4 * 7 - 1 = 174 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1414_141468


namespace NUMINAMATH_CALUDE_odd_digits_346_base5_l1414_141415

/-- Counts the number of odd digits in a base-5 number --/
def countOddDigitsBase5 (n : ℕ) : ℕ := sorry

/-- Converts a base-10 number to base-5 --/
def toBase5 (n : ℕ) : ℕ := sorry

theorem odd_digits_346_base5 : 
  countOddDigitsBase5 (toBase5 346) = 2 := by sorry

end NUMINAMATH_CALUDE_odd_digits_346_base5_l1414_141415


namespace NUMINAMATH_CALUDE_average_pages_per_day_l1414_141486

theorem average_pages_per_day 
  (total_pages : ℕ) 
  (pages_read : ℕ) 
  (remaining_days : ℕ) 
  (h1 : total_pages = 212) 
  (h2 : pages_read = 97) 
  (h3 : remaining_days = 5) :
  (total_pages - pages_read) / remaining_days = 23 := by
sorry

end NUMINAMATH_CALUDE_average_pages_per_day_l1414_141486


namespace NUMINAMATH_CALUDE_expression_simplification_l1414_141409

theorem expression_simplification (y : ℝ) :
  3 * y + 12 * y^2 + 18 - (6 - 3 * y - 12 * y^2) + 5 * y^3 = 5 * y^3 + 24 * y^2 + 6 * y + 12 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1414_141409


namespace NUMINAMATH_CALUDE_correct_pricing_l1414_141464

/-- A hotel's pricing structure -/
structure HotelPricing where
  flat_fee : ℝ
  additional_fee : ℝ
  discount : ℝ := 10

/-- Calculate the cost of a stay given the pricing and number of nights -/
def stay_cost (p : HotelPricing) (nights : ℕ) : ℝ :=
  if nights ≤ 4 then
    p.flat_fee + p.additional_fee * (nights - 1 : ℝ)
  else
    p.flat_fee + p.additional_fee * 3 + (p.additional_fee - p.discount) * ((nights - 4) : ℝ)

/-- The theorem stating the correct pricing structure -/
theorem correct_pricing :
  ∃ (p : HotelPricing),
    stay_cost p 4 = 180 ∧
    stay_cost p 7 = 302 ∧
    p.flat_fee = 28 ∧
    p.additional_fee = 50.67 := by
  sorry

end NUMINAMATH_CALUDE_correct_pricing_l1414_141464


namespace NUMINAMATH_CALUDE_student_selection_count_l1414_141446

theorem student_selection_count (n m k : ℕ) (hn : n = 60) (hm : m = 2) (hk : k = 5) :
  (Nat.choose n k - Nat.choose (n - m) k : ℕ) =
  (Nat.choose m 1 * Nat.choose (n - 1) (k - 1) -
   Nat.choose m 2 * Nat.choose (n - m) (k - 2) : ℕ) :=
by sorry

end NUMINAMATH_CALUDE_student_selection_count_l1414_141446


namespace NUMINAMATH_CALUDE_clothing_store_pricing_strategy_l1414_141466

/-- Clothing store pricing and sales model -/
structure ClothingStore where
  cost : ℕ             -- Cost price per piece in yuan
  price : ℕ            -- Original selling price per piece in yuan
  baseSales : ℕ        -- Original daily sales in pieces
  salesIncrease : ℕ    -- Additional sales per yuan of price reduction

/-- Calculate daily sales after price reduction -/
def dailySales (store : ClothingStore) (reduction : ℕ) : ℕ :=
  store.baseSales + store.salesIncrease * reduction

/-- Calculate profit per piece after price reduction -/
def profitPerPiece (store : ClothingStore) (reduction : ℕ) : ℤ :=
  (store.price - store.cost - reduction : ℤ)

/-- Calculate total daily profit after price reduction -/
def dailyProfit (store : ClothingStore) (reduction : ℕ) : ℤ :=
  (dailySales store reduction) * (profitPerPiece store reduction)

/-- The main theorem about the clothing store's pricing strategy -/
theorem clothing_store_pricing_strategy 
  (store : ClothingStore) 
  (h_cost : store.cost = 45) 
  (h_price : store.price = 65) 
  (h_baseSales : store.baseSales = 30) 
  (h_salesIncrease : store.salesIncrease = 5) : 
  (dailySales store 3 = 45 ∧ profitPerPiece store 3 = 17) ∧
  (∃ x : ℕ, x = 10 ∧ dailyProfit store x = 800) := by
  sorry

end NUMINAMATH_CALUDE_clothing_store_pricing_strategy_l1414_141466


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1414_141425

/-- Proves that the eccentricity of a hyperbola with specific properties is 2√3/3 -/
theorem hyperbola_eccentricity (a b c : ℝ) (ha : a > 0) (hb : b > 0) : 
  let hyperbola := fun (x y : ℝ) ↦ x^2 / a^2 - y^2 / b^2 = 1
  let asymptote := fun (x : ℝ) ↦ b / a * x
  let F := (c, 0)
  let A := (c, b^2 / a)
  let B := (c, b * c / a)
  hyperbola c (b^2 / a) ∧ 
  A.1 = (F.1 + B.1) / 2 ∧ 
  A.2 = (F.2 + B.2) / 2 →
  c / a = 2 * Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1414_141425


namespace NUMINAMATH_CALUDE_triangle_arithmetic_geometric_is_equilateral_l1414_141457

/-- A triangle with sides a, b, c and angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The property that the angles form an arithmetic sequence -/
def Triangle.angles_arithmetic_sequence (t : Triangle) : Prop :=
  ∃ d : ℝ, (t.B - t.A = d ∧ t.C - t.B = d) ∨ (t.A - t.B = d ∧ t.B - t.C = d) ∨ (t.C - t.A = d ∧ t.A - t.B = d)

/-- The property that the sides form a geometric sequence -/
def Triangle.sides_geometric_sequence (t : Triangle) : Prop :=
  (t.b^2 = t.a * t.c) ∨ (t.a^2 = t.b * t.c) ∨ (t.c^2 = t.a * t.b)

/-- A triangle is equilateral if all its sides are equal -/
def Triangle.is_equilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

/-- The main theorem -/
theorem triangle_arithmetic_geometric_is_equilateral (t : Triangle) :
  t.angles_arithmetic_sequence → t.sides_geometric_sequence → t.is_equilateral :=
sorry

end NUMINAMATH_CALUDE_triangle_arithmetic_geometric_is_equilateral_l1414_141457


namespace NUMINAMATH_CALUDE_function_determination_l1414_141441

/-- Given a function f: ℝ → ℝ satisfying f(1/x) = 1/(x+1) for x ≠ 0 and x ≠ -1,
    prove that f(x) = x/(x+1) for x ≠ 0 and x ≠ -1 -/
theorem function_determination (f : ℝ → ℝ) :
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ -1 → f (1/x) = 1/(x+1)) →
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ -1 → f x = x/(x+1)) :=
by sorry

end NUMINAMATH_CALUDE_function_determination_l1414_141441


namespace NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l1414_141444

theorem polar_to_rectangular_conversion :
  let r : ℝ := 7
  let θ : ℝ := Real.pi / 4
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  (x, y) = (7 * Real.sqrt 2 / 2, 7 * Real.sqrt 2 / 2) := by
  sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l1414_141444


namespace NUMINAMATH_CALUDE_arithmetic_sequence_terms_l1414_141411

theorem arithmetic_sequence_terms (a₁ : ℝ) (d : ℝ) (aₙ : ℝ) (n : ℕ) :
  a₁ = 2.5 →
  d = 4 →
  aₙ = 46.5 →
  aₙ = a₁ + (n - 1) * d →
  n = 12 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_terms_l1414_141411


namespace NUMINAMATH_CALUDE_max_value_theorem_l1414_141405

theorem max_value_theorem (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) (sum_eq_two : a + b + c = 2) :
  (a * b / (a + b) + a * c / (a + c) + b * c / (b + c)) ≤ 1 ∧
  ∃ (x y z : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ x + y + z = 2 ∧
    x * y / (x + y) + x * z / (x + z) + y * z / (y + z) = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l1414_141405


namespace NUMINAMATH_CALUDE_roberts_balls_theorem_l1414_141477

/-- Calculates the final number of balls Robert has -/
def robertsFinalBalls (robertsInitial : ℕ) (timsTotal : ℕ) (jennysTotal : ℕ) : ℕ :=
  robertsInitial + timsTotal / 2 + jennysTotal / 3

theorem roberts_balls_theorem :
  robertsFinalBalls 25 40 60 = 65 := by
  sorry

end NUMINAMATH_CALUDE_roberts_balls_theorem_l1414_141477


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1414_141490

theorem sufficient_not_necessary :
  (∀ x y : ℝ, x > 1 ∧ y > 1 → x + y > 2) ∧
  (∃ x y : ℝ, x + y > 2 ∧ ¬(x > 1 ∧ y > 1)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1414_141490


namespace NUMINAMATH_CALUDE_range_of_x_range_of_a_l1414_141420

-- Part 1
theorem range_of_x (x : ℝ) :
  (x^2 - 4*x + 3 < 0) → ((x - 3)^2 < 1) → (2 < x ∧ x < 3) := by sorry

-- Part 2
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (x^2 - 4*a*x + 3*a^2 ≥ 0 → (x - 3)^2 ≥ 1)) →
  (∃ x : ℝ, (x - 3)^2 ≥ 1 ∧ x^2 - 4*a*x + 3*a^2 < 0) →
  (4/3 ≤ a ∧ a ≤ 2) := by sorry

end NUMINAMATH_CALUDE_range_of_x_range_of_a_l1414_141420


namespace NUMINAMATH_CALUDE_inequality_proof_l1414_141450

theorem inequality_proof (a b c : Real) 
  (ha : 0 ≤ a ∧ a ≤ 1) 
  (hb : 0 ≤ b ∧ b ≤ 1) 
  (hc : 0 ≤ c ∧ c ≤ 1) : 
  (a / (b + c + 1)) + (b / (c + a + 1)) + (c / (a + b + 1)) + (1 - a) * (1 - b) * (1 - c) ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1414_141450


namespace NUMINAMATH_CALUDE_no_solution_condition_l1414_141430

theorem no_solution_condition (b : ℝ) : 
  (∀ x : ℝ, 4 * (3 * x - b) ≠ 3 * (4 * x + 16)) ↔ b = -12 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_condition_l1414_141430


namespace NUMINAMATH_CALUDE_abc_zero_l1414_141414

theorem abc_zero (a b c : ℝ) 
  (h1 : (a + b) * (b + c) * (c + a) = a * b * c)
  (h2 : (a^3 + b^3) * (b^3 + c^3) * (c^3 + a^3) = a^3 * b^3 * c^3) :
  a * b * c = 0 := by
sorry

end NUMINAMATH_CALUDE_abc_zero_l1414_141414


namespace NUMINAMATH_CALUDE_A_intersect_B_eq_open_interval_l1414_141434

-- Define sets A and B
def A : Set ℝ := {x | Real.sqrt (x - 1) < Real.sqrt 2}
def B : Set ℝ := {x | x^2 - 6*x + 8 < 0}

-- State the theorem
theorem A_intersect_B_eq_open_interval : A ∩ B = Set.Ioo 2 3 := by
  sorry

end NUMINAMATH_CALUDE_A_intersect_B_eq_open_interval_l1414_141434


namespace NUMINAMATH_CALUDE_smaller_factor_of_4536_l1414_141459

theorem smaller_factor_of_4536 (a b : ℕ) : 
  10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ a * b = 4536 → 
  min a b = 63 := by
sorry

end NUMINAMATH_CALUDE_smaller_factor_of_4536_l1414_141459


namespace NUMINAMATH_CALUDE_sum_of_roots_equation_l1414_141465

theorem sum_of_roots_equation (x : ℝ) : 
  (∃ x₁ x₂ : ℝ, (x₁^2 + 18*x₁ + 30 = 2 * Real.sqrt (x₁^2 + 18*x₁ + 45)) ∧ 
                (x₂^2 + 18*x₂ + 30 = 2 * Real.sqrt (x₂^2 + 18*x₂ + 45)) ∧ 
                (∀ y : ℝ, y^2 + 18*y + 30 = 2 * Real.sqrt (y^2 + 18*y + 45) → y = x₁ ∨ y = x₂)) → 
  (∃ x₁ x₂ : ℝ, (x₁^2 + 18*x₁ + 30 = 2 * Real.sqrt (x₁^2 + 18*x₁ + 45)) ∧ 
                (x₂^2 + 18*x₂ + 30 = 2 * Real.sqrt (x₂^2 + 18*x₂ + 45)) ∧ 
                (x₁ + x₂ = -18)) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_equation_l1414_141465


namespace NUMINAMATH_CALUDE_f_sum_theorem_l1414_141469

def is_increasing (f : ℕ+ → ℕ+) : Prop :=
  ∀ x y : ℕ+, x < y → f x < f y

theorem f_sum_theorem (f : ℕ+ → ℕ+) 
  (h_increasing : is_increasing f)
  (h_functional : ∀ k : ℕ+, f (f k) = 3 * k) :
  f 1 + f 9 + f 96 = 197 := by
sorry

end NUMINAMATH_CALUDE_f_sum_theorem_l1414_141469


namespace NUMINAMATH_CALUDE_quadratic_roots_modulus_l1414_141431

theorem quadratic_roots_modulus (a : ℝ) : 
  (∀ x : ℂ, (a * x^2 + x + 1 = 0) → Complex.abs x < 1) → a > 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_modulus_l1414_141431


namespace NUMINAMATH_CALUDE_cistern_leak_time_l1414_141402

/-- Represents the cistern problem -/
def CisternProblem (capacity : ℝ) (tapRate : ℝ) (timeWithTap : ℝ) : Prop :=
  let leakRate := capacity / timeWithTap + tapRate
  let timeWithoutTap := capacity / leakRate
  timeWithoutTap = 20

/-- Theorem stating the solution to the cistern problem -/
theorem cistern_leak_time :
  CisternProblem 480 4 24 := by sorry

end NUMINAMATH_CALUDE_cistern_leak_time_l1414_141402


namespace NUMINAMATH_CALUDE_coconut_crab_goat_trade_l1414_141438

theorem coconut_crab_goat_trade (coconuts_per_crab : ℕ) (total_coconuts : ℕ) (final_goats : ℕ) :
  coconuts_per_crab = 3 →
  total_coconuts = 342 →
  final_goats = 19 →
  (total_coconuts / coconuts_per_crab) / final_goats = 6 :=
by sorry

end NUMINAMATH_CALUDE_coconut_crab_goat_trade_l1414_141438


namespace NUMINAMATH_CALUDE_late_condition_early_condition_60_early_condition_50_exact_time_speed_correct_l1414_141403

/-- Represents the speed Mr. Bird needs to drive to arrive exactly on time -/
def exact_time_speed : ℚ :=
  160 / 3

/-- Represents the distance to Mr. Bird's workplace in miles -/
def distance_to_work : ℚ :=
  40 / 3

/-- Represents the ideal time to reach work on time in hours -/
def ideal_time : ℚ :=
  1 / 4

/-- Given that driving at 40 mph makes Mr. Bird 5 minutes late -/
theorem late_condition (speed : ℚ) (time : ℚ) :
  speed = 40 → time = ideal_time + 5 / 60 → speed * time = distance_to_work :=
sorry

/-- Given that driving at 60 mph makes Mr. Bird 2 minutes early -/
theorem early_condition_60 (speed : ℚ) (time : ℚ) :
  speed = 60 → time = ideal_time - 2 / 60 → speed * time = distance_to_work :=
sorry

/-- Given that driving at 50 mph makes Mr. Bird 1 minute early -/
theorem early_condition_50 (speed : ℚ) (time : ℚ) :
  speed = 50 → time = ideal_time - 1 / 60 → speed * time = distance_to_work :=
sorry

/-- Theorem stating that the exact_time_speed is the speed required to arrive exactly on time -/
theorem exact_time_speed_correct :
  exact_time_speed * ideal_time = distance_to_work :=
sorry

end NUMINAMATH_CALUDE_late_condition_early_condition_60_early_condition_50_exact_time_speed_correct_l1414_141403


namespace NUMINAMATH_CALUDE_exists_tastrophic_function_l1414_141480

/-- A function is k-tastrophic if its k-th iteration raises its input to the k-th power. -/
def IsTastrophic (k : ℕ) (f : ℕ → ℕ) : Prop :=
  k > 1 ∧ ∀ n : ℕ, n > 0 → (f^[k] n = n^k)

/-- For every integer k > 1, there exists a k-tastrophic function. -/
theorem exists_tastrophic_function :
  ∀ k : ℕ, k > 1 → ∃ f : ℕ → ℕ, IsTastrophic k f :=
sorry

end NUMINAMATH_CALUDE_exists_tastrophic_function_l1414_141480


namespace NUMINAMATH_CALUDE_ab_power_is_negative_eight_l1414_141481

theorem ab_power_is_negative_eight (a b : ℝ) (h : |a + 2| + (b - 3)^2 = 0) : a^b = -8 := by
  sorry

end NUMINAMATH_CALUDE_ab_power_is_negative_eight_l1414_141481


namespace NUMINAMATH_CALUDE_complex_number_equality_l1414_141475

theorem complex_number_equality : Complex.I * 2 / (1 - Complex.I) = -1 + Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_number_equality_l1414_141475


namespace NUMINAMATH_CALUDE_units_digit_problem_l1414_141406

theorem units_digit_problem : ∃ n : ℕ, n < 10 ∧ 
  (72^129 + 36^93 + 57^73 - 45^105) % 10 = n ∧ n = 0 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_problem_l1414_141406


namespace NUMINAMATH_CALUDE_comic_book_collections_equal_l1414_141472

/-- Kymbrea's initial comic book collection size -/
def kymbrea_initial : ℕ := 40

/-- Kymbrea's monthly comic book addition rate -/
def kymbrea_rate : ℕ := 3

/-- LaShawn's initial comic book collection size -/
def lashawn_initial : ℕ := 20

/-- LaShawn's monthly comic book addition rate -/
def lashawn_rate : ℕ := 5

/-- The number of months after which LaShawn's collection will be three times Kymbrea's -/
def months : ℕ := 25

theorem comic_book_collections_equal : 
  lashawn_initial + lashawn_rate * months = 3 * (kymbrea_initial + kymbrea_rate * months) := by
  sorry

end NUMINAMATH_CALUDE_comic_book_collections_equal_l1414_141472


namespace NUMINAMATH_CALUDE_largest_digit_divisible_by_six_l1414_141416

theorem largest_digit_divisible_by_six :
  ∃ (N : ℕ), N ≤ 9 ∧ 
  (∀ (M : ℕ), M ≤ 9 → 6 ∣ (5678 * 10 + M) → M ≤ N) ∧
  (6 ∣ (5678 * 10 + N)) :=
by sorry

end NUMINAMATH_CALUDE_largest_digit_divisible_by_six_l1414_141416


namespace NUMINAMATH_CALUDE_max_number_with_30_divisors_l1414_141452

/-- The number of positive divisors of n -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- n is divisible by m -/
def is_divisible_by (n m : ℕ) : Prop := sorry

theorem max_number_with_30_divisors :
  ∀ n : ℕ, 
    is_divisible_by n 30 → 
    num_divisors n = 30 → 
    n ≤ 11250 ∧ 
    (n = 11250 → is_divisible_by 11250 30 ∧ num_divisors 11250 = 30) :=
by sorry

end NUMINAMATH_CALUDE_max_number_with_30_divisors_l1414_141452


namespace NUMINAMATH_CALUDE_equilateral_triangle_third_vertex_y_coord_l1414_141401

/-- Given an equilateral triangle with two vertices at (2,7) and (10,7),
    and the third vertex in the first quadrant, the y-coordinate of the third vertex is 7 + 4√3 -/
theorem equilateral_triangle_third_vertex_y_coord :
  ∀ (x y : ℝ),
  (x > 0 ∧ y > 0) →  -- Third vertex is in the first quadrant
  (x - 2)^2 + (y - 7)^2 = 8^2 →  -- Distance from (2,7) is 8
  (x - 10)^2 + (y - 7)^2 = 8^2 →  -- Distance from (10,7) is 8
  y = 7 + 4 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_third_vertex_y_coord_l1414_141401


namespace NUMINAMATH_CALUDE_line_through_point_l1414_141429

/-- Given a line equation -1/2 - 2kx = 5y that passes through the point (1/4, -6),
    prove that k = 59 is the unique solution. -/
theorem line_through_point (k : ℝ) : 
  (-1/2 : ℝ) - 2 * k * (1/4 : ℝ) = 5 * (-6 : ℝ) ↔ k = 59 := by
sorry

end NUMINAMATH_CALUDE_line_through_point_l1414_141429


namespace NUMINAMATH_CALUDE_second_meeting_at_six_minutes_l1414_141462

/-- Represents a swimmer in the race -/
structure Swimmer where
  speed : ℝ
  startPosition : ℝ

/-- Represents the race scenario -/
structure RaceScenario where
  poolLength : ℝ
  swimmer1 : Swimmer
  swimmer2 : Swimmer
  firstMeetingTime : ℝ
  firstMeetingPosition : ℝ

/-- Calculates the time of the second meeting given a race scenario -/
def secondMeetingTime (race : RaceScenario) : ℝ :=
  sorry

/-- Theorem stating that under the given conditions, the second meeting occurs at 6 minutes -/
theorem second_meeting_at_six_minutes (race : RaceScenario) 
  (h1 : race.poolLength = 120)
  (h2 : race.swimmer1.startPosition = 0)
  (h3 : race.swimmer2.startPosition = 120)
  (h4 : race.firstMeetingTime = 1)
  (h5 : race.firstMeetingPosition = 40)
  (h6 : race.swimmer1.speed = race.firstMeetingPosition / race.firstMeetingTime)
  (h7 : race.swimmer2.speed = (race.poolLength - race.firstMeetingPosition) / race.firstMeetingTime) :
  secondMeetingTime race = 6 :=
sorry

end NUMINAMATH_CALUDE_second_meeting_at_six_minutes_l1414_141462


namespace NUMINAMATH_CALUDE_tax_discount_commute_l1414_141485

theorem tax_discount_commute (price : ℝ) (tax_rate discount_rate : ℝ) 
  (h1 : 0 ≤ tax_rate) (h2 : 0 ≤ discount_rate) (h3 : tax_rate < 1) (h4 : discount_rate < 1) :
  price * (1 + tax_rate) * (1 - discount_rate) = price * (1 - discount_rate) * (1 + tax_rate) :=
by sorry

end NUMINAMATH_CALUDE_tax_discount_commute_l1414_141485


namespace NUMINAMATH_CALUDE_count_integer_pairs_l1414_141479

theorem count_integer_pairs : 
  ∃! (s : Finset (ℤ × ℤ)), 
    (∀ (p : ℤ × ℤ), p ∈ s ↔ p.1^2 + p.2 = p.1 * p.2 + 1) ∧ 
    s.card = 5 := by
  sorry

end NUMINAMATH_CALUDE_count_integer_pairs_l1414_141479


namespace NUMINAMATH_CALUDE_coefficient_of_x_fourth_power_is_zero_l1414_141491

def expression (x : ℝ) : ℝ := 3 * (x^2 - x^4) - 5 * (x^4 - x^6 + x^2) + 4 * (2*x^4 - x^8)

theorem coefficient_of_x_fourth_power_is_zero :
  ∃ f : ℝ → ℝ, ∀ x : ℝ, expression x = f x + 0 * x^4 :=
sorry

end NUMINAMATH_CALUDE_coefficient_of_x_fourth_power_is_zero_l1414_141491


namespace NUMINAMATH_CALUDE_inverse_proportion_example_l1414_141460

/-- Represents an inverse proportional relationship between two variables -/
def InverseProportion (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x : ℝ, x ≠ 0 → f x = k / x

/-- The function f(x) = 3/x is inversely proportional -/
theorem inverse_proportion_example : InverseProportion (fun x => 3 / x) := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_example_l1414_141460


namespace NUMINAMATH_CALUDE_class_size_l1414_141499

theorem class_size (french : ℕ) (german : ℕ) (both : ℕ) (neither : ℕ)
  (h1 : french = 41)
  (h2 : german = 22)
  (h3 : both = 9)
  (h4 : neither = 40) :
  french + german - both + neither = 94 := by
  sorry

end NUMINAMATH_CALUDE_class_size_l1414_141499


namespace NUMINAMATH_CALUDE_oil_tank_capacity_l1414_141471

theorem oil_tank_capacity (t : ℝ) (h1 : t > 0) : 
  (1/4 : ℝ) * t + 6 = (1/3 : ℝ) * t → t = 72 := by
  sorry

end NUMINAMATH_CALUDE_oil_tank_capacity_l1414_141471


namespace NUMINAMATH_CALUDE_number_of_divisors_sum_of_divisors_l1414_141442

def n : ℕ := 120

-- Number of positive divisors
theorem number_of_divisors : (Finset.filter (· ∣ n) (Finset.range (n + 1))).card = 16 := by sorry

-- Sum of positive divisors
theorem sum_of_divisors : (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id = 3240 := by sorry

end NUMINAMATH_CALUDE_number_of_divisors_sum_of_divisors_l1414_141442


namespace NUMINAMATH_CALUDE_hvac_cost_per_vent_l1414_141410

/-- Calculates the cost per vent of an HVAC system -/
def cost_per_vent (total_cost : ℕ) (num_zones : ℕ) (vents_per_zone : ℕ) : ℕ :=
  total_cost / (num_zones * vents_per_zone)

/-- Proves that the cost per vent of the given HVAC system is $2,000 -/
theorem hvac_cost_per_vent :
  cost_per_vent 20000 2 5 = 2000 := by
  sorry

#eval cost_per_vent 20000 2 5

end NUMINAMATH_CALUDE_hvac_cost_per_vent_l1414_141410


namespace NUMINAMATH_CALUDE_doughnuts_given_away_l1414_141496

/-- Represents the bakery's doughnut sales and production --/
structure BakeryData where
  total_doughnuts : ℕ
  small_box_capacity : ℕ
  large_box_capacity : ℕ
  small_box_price : ℚ
  large_box_price : ℚ
  discount_rate : ℚ
  small_boxes_sold : ℕ
  large_boxes_sold : ℕ
  large_boxes_discounted : ℕ

/-- Theorem stating the number of doughnuts given away --/
theorem doughnuts_given_away (data : BakeryData) : 
  data.total_doughnuts = 300 ∧
  data.small_box_capacity = 6 ∧
  data.large_box_capacity = 12 ∧
  data.small_box_price = 5 ∧
  data.large_box_price = 9 ∧
  data.discount_rate = 1/10 ∧
  data.small_boxes_sold = 20 ∧
  data.large_boxes_sold = 10 ∧
  data.large_boxes_discounted = 5 →
  data.total_doughnuts - 
    (data.small_boxes_sold * data.small_box_capacity + 
     data.large_boxes_sold * data.large_box_capacity) = 60 := by
  sorry

end NUMINAMATH_CALUDE_doughnuts_given_away_l1414_141496


namespace NUMINAMATH_CALUDE_quadratic_roots_implications_l1414_141482

theorem quadratic_roots_implications (a : ℝ) :
  (∃ x₁ x₂ : ℝ, 
    (7 * x₁^2 - (a + 13) * x₁ + a^2 - a - 2 = 0) ∧
    (7 * x₂^2 - (a + 13) * x₂ + a^2 - a - 2 = 0) ∧
    (0 < x₁ ∧ x₁ < 1 ∧ 1 < x₂ ∧ x₂ < 2)) →
  (a ∈ Set.Ioo (-2 : ℝ) (-1) ∪ Set.Ioo 3 4) ∧
  (∀ a' ∈ Set.Ioo 3 4, a'^3 > a'^2 - a' + 1) ∧
  (∀ a' ∈ Set.Ioo (-2 : ℝ) (-1), a'^3 < a'^2 - a' + 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_implications_l1414_141482


namespace NUMINAMATH_CALUDE_pentagon_area_increase_l1414_141474

/-- The increase in area when expanding a convex pentagon's boundary --/
theorem pentagon_area_increase (P s : ℝ) (h : P > 0) (h' : s > 0) :
  let increase := s * P + π * s^2
  increase = s * P + π * s^2 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_area_increase_l1414_141474


namespace NUMINAMATH_CALUDE_coffee_cost_is_18_l1414_141487

/-- Represents the coffee consumption and cost parameters --/
structure CoffeeParams where
  cups_per_day : ℕ
  oz_per_cup : ℚ
  bag_cost : ℚ
  oz_per_bag : ℚ
  milk_gal_per_week : ℚ
  milk_cost_per_gal : ℚ

/-- Calculates the weekly cost of coffee given the parameters --/
def weekly_coffee_cost (params : CoffeeParams) : ℚ :=
  let beans_oz_per_week := params.cups_per_day * params.oz_per_cup * 7
  let bags_per_week := beans_oz_per_week / params.oz_per_bag
  let bean_cost := bags_per_week * params.bag_cost
  let milk_cost := params.milk_gal_per_week * params.milk_cost_per_gal
  bean_cost + milk_cost

/-- Theorem stating that the weekly coffee cost is $18 --/
theorem coffee_cost_is_18 :
  ∃ (params : CoffeeParams),
    params.cups_per_day = 2 ∧
    params.oz_per_cup = 3/2 ∧
    params.bag_cost = 8 ∧
    params.oz_per_bag = 21/2 ∧
    params.milk_gal_per_week = 1/2 ∧
    params.milk_cost_per_gal = 4 ∧
    weekly_coffee_cost params = 18 :=
  sorry

end NUMINAMATH_CALUDE_coffee_cost_is_18_l1414_141487


namespace NUMINAMATH_CALUDE_absolute_value_equation_roots_l1414_141447

theorem absolute_value_equation_roots : ∃ (x y : ℝ), 
  (x^2 - 3*|x| - 10 = 0) ∧ 
  (y^2 - 3*|y| - 10 = 0) ∧ 
  (x + y = 0) ∧ 
  (x * y = -25) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_roots_l1414_141447


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l1414_141461

theorem absolute_value_equation_solution :
  {y : ℝ | |4 * y - 5| = 39} = {11, -8.5} := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l1414_141461


namespace NUMINAMATH_CALUDE_complex_i_minus_one_in_third_quadrant_l1414_141456

theorem complex_i_minus_one_in_third_quadrant :
  let z : ℂ := Complex.I * (Complex.I - 1)
  (z.re < 0) ∧ (z.im < 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_i_minus_one_in_third_quadrant_l1414_141456


namespace NUMINAMATH_CALUDE_no_linear_term_implies_a_equals_six_l1414_141422

theorem no_linear_term_implies_a_equals_six (a : ℝ) : 
  (∀ x : ℝ, ∃ b c : ℝ, (2*x + a) * (3 - x) = b * x^2 + c) → a = 6 := by
  sorry

end NUMINAMATH_CALUDE_no_linear_term_implies_a_equals_six_l1414_141422


namespace NUMINAMATH_CALUDE_polynomial_ascending_powers_x_l1414_141432

-- Define the polynomial
def p (x y : ℝ) : ℝ := x^3 - 5*x*y^2 - 7*y^3 + 8*x^2*y

-- Define a function to extract the degree of x in a term
def degree_x (term : ℝ → ℝ → ℝ) : ℕ :=
  sorry  -- Implementation details omitted

-- Define the ascending order of terms with respect to x
def ascending_order_x (term1 term2 : ℝ → ℝ → ℝ) : Prop :=
  degree_x term1 ≤ degree_x term2

-- State the theorem
theorem polynomial_ascending_powers_x :
  ∃ (term1 term2 term3 term4 : ℝ → ℝ → ℝ),
    (∀ x y, p x y = term1 x y + term2 x y + term3 x y + term4 x y) ∧
    (ascending_order_x term1 term2) ∧
    (ascending_order_x term2 term3) ∧
    (ascending_order_x term3 term4) ∧
    (∀ x y, term1 x y = -7*y^3) ∧
    (∀ x y, term2 x y = -5*x*y^2) ∧
    (∀ x y, term3 x y = 8*x^2*y) ∧
    (∀ x y, term4 x y = x^3) :=
  sorry

end NUMINAMATH_CALUDE_polynomial_ascending_powers_x_l1414_141432


namespace NUMINAMATH_CALUDE_correct_option_is_valid_print_statement_l1414_141483

-- Define an enum for the options
inductive ProgramOption
| A
| B
| C
| D

-- Define a function to check if an option is a valid print statement
def isValidPrintStatement (option : ProgramOption) : Prop :=
  match option with
  | ProgramOption.A => True  -- PRINT 4*x is valid
  | _ => False               -- Other options are not valid print statements

-- Theorem statement
theorem correct_option_is_valid_print_statement :
  ∃ (option : ProgramOption), isValidPrintStatement option :=
by
  sorry


end NUMINAMATH_CALUDE_correct_option_is_valid_print_statement_l1414_141483


namespace NUMINAMATH_CALUDE_product_equals_reversed_product_l1414_141427

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

theorem product_equals_reversed_product 
  (a b : ℕ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : is_two_digit a) 
  (h4 : (reverse_digits a) * b = 220) : 
  a * b = 220 := by
sorry

end NUMINAMATH_CALUDE_product_equals_reversed_product_l1414_141427


namespace NUMINAMATH_CALUDE_same_theme_probability_l1414_141489

/-- The probability of two students choosing the same theme out of two options -/
theorem same_theme_probability (themes : Nat) (students : Nat) : 
  themes = 2 → students = 2 → (themes^students / 2) / themes^students = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_same_theme_probability_l1414_141489


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1414_141495

theorem inequality_system_solution (x : ℝ) :
  (3 * (x - 1) < 5 * x + 11 ∧ 2 * x > (9 - x) / 4) ↔ x > 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1414_141495


namespace NUMINAMATH_CALUDE_tree_height_problem_l1414_141448

theorem tree_height_problem (h₁ h₂ : ℝ) : 
  h₁ = h₂ + 16 →  -- One tree is 16 feet taller than the other
  h₂ / h₁ = 3 / 4 →  -- The heights are in the ratio 3:4
  h₁ = 64 :=  -- The taller tree is 64 feet tall
by
  sorry

end NUMINAMATH_CALUDE_tree_height_problem_l1414_141448


namespace NUMINAMATH_CALUDE_distance_proof_l1414_141488

/-- The distance between two locations A and B, where two buses meet under specific conditions --/
def distance_between_locations : ℝ :=
  let first_meeting_distance : ℝ := 85
  let second_meeting_distance : ℝ := 65
  3 * first_meeting_distance - second_meeting_distance

theorem distance_proof :
  let first_meeting_distance : ℝ := 85
  let second_meeting_distance : ℝ := 65
  let total_distance := distance_between_locations
  (∃ (speed_A speed_B : ℝ), speed_A > 0 ∧ speed_B > 0 ∧
    first_meeting_distance / speed_A = (total_distance - first_meeting_distance) / speed_B ∧
    (total_distance - first_meeting_distance + second_meeting_distance) / speed_A + 0.5 =
    (first_meeting_distance + (total_distance - second_meeting_distance)) / speed_B + 0.5) →
  total_distance = 190 := by
  sorry

end NUMINAMATH_CALUDE_distance_proof_l1414_141488


namespace NUMINAMATH_CALUDE_juice_bar_group_size_l1414_141435

theorem juice_bar_group_size :
  ∀ (total_spent mango_price pineapple_price pineapple_spent : ℕ),
    total_spent = 94 →
    mango_price = 5 →
    pineapple_price = 6 →
    pineapple_spent = 54 →
    ∃ (mango_count pineapple_count : ℕ),
      pineapple_count * pineapple_price = pineapple_spent ∧
      mango_count * mango_price = total_spent - pineapple_spent ∧
      mango_count + pineapple_count = 17 :=
by
  sorry


end NUMINAMATH_CALUDE_juice_bar_group_size_l1414_141435
