import Mathlib

namespace quadratic_inequality_solution_set_R_l214_21418

/-- The solution set of a quadratic inequality is R iff a < 0 and discriminant < 0 -/
theorem quadratic_inequality_solution_set_R 
  (a b c : ℝ) (h : a ≠ 0) : 
  (∀ x, a * x^2 + b * x + c < 0) ↔ (a < 0 ∧ b^2 - 4*a*c < 0) :=
sorry

end quadratic_inequality_solution_set_R_l214_21418


namespace phoenix_airport_on_time_rate_l214_21490

/-- Calculates the on-time departure rate given the number of on-time departures and total flights -/
def onTimeRate (onTime : ℕ) (total : ℕ) : ℚ :=
  onTime / total

/-- Proves that adding one more on-time flight after 3 on-time and 1 late flight 
    results in an on-time rate higher than 60% -/
theorem phoenix_airport_on_time_rate : 
  let initialOnTime : ℕ := 3
  let initialTotal : ℕ := 4
  let additionalOnTime : ℕ := 1
  onTimeRate (initialOnTime + additionalOnTime) (initialTotal + additionalOnTime) > 60 / 100 := by
  sorry

#eval onTimeRate 4 5 > 60 / 100

end phoenix_airport_on_time_rate_l214_21490


namespace investment_sum_l214_21402

/-- Given a sum invested at different interest rates, prove the sum equals 8400 --/
theorem investment_sum (P : ℝ) : 
  (P * 0.15 * 2) - (P * 0.10 * 2) = 840 → P = 8400 := by
  sorry

end investment_sum_l214_21402


namespace lcm_problem_l214_21496

theorem lcm_problem (m : ℕ+) (h1 : Nat.lcm 36 m = 180) (h2 : Nat.lcm m 50 = 300) : m = 60 := by
  sorry

end lcm_problem_l214_21496


namespace simplified_fourth_root_l214_21434

theorem simplified_fourth_root (c d : ℕ+) :
  (2^5 * 5^3 : ℝ)^(1/4) = c * d^(1/4) → c + d = 252 := by
  sorry

end simplified_fourth_root_l214_21434


namespace sum_of_digits_power_product_l214_21485

def power_product : ℕ := 2^2009 * 5^2010 * 7

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem sum_of_digits_power_product : sum_of_digits power_product = 8 := by
  sorry

end sum_of_digits_power_product_l214_21485


namespace third_number_in_ratio_l214_21457

theorem third_number_in_ratio (a b c : ℝ) : 
  a / 5 = b / 6 ∧ b / 6 = c / 8 ∧  -- numbers are in ratio 5 : 6 : 8
  a + c = b + 49 →                -- sum of longest and smallest equals sum of third and 49
  b = 42 :=                       -- prove that the third number (b) is 42
by sorry

end third_number_in_ratio_l214_21457


namespace a_equals_3_sufficient_not_necessary_l214_21472

def A (a : ℕ) : Set ℕ := {1, a}
def B : Set ℕ := {1, 2, 3}

theorem a_equals_3_sufficient_not_necessary :
  (∀ a : ℕ, a = 3 → A a ⊆ B) ∧
  (∃ a : ℕ, A a ⊆ B ∧ a ≠ 3) :=
by sorry

end a_equals_3_sufficient_not_necessary_l214_21472


namespace jungkook_has_smallest_number_l214_21477

def yoongi_number : ℕ := 7
def jungkook_number : ℕ := 6
def yuna_number : ℕ := 9

theorem jungkook_has_smallest_number :
  jungkook_number ≤ yoongi_number ∧ jungkook_number ≤ yuna_number :=
by
  sorry

end jungkook_has_smallest_number_l214_21477


namespace circle_intersection_range_l214_21408

-- Define the circles
def circle1 (a : ℝ) (x y : ℝ) : Prop := (x - a)^2 + y^2 = 1
def circle2 (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 25

-- Define the theorem
theorem circle_intersection_range (a : ℝ) :
  (a ≥ 0) →
  (∃ x y : ℝ, circle1 a x y ∧ circle2 x y) →
  2 * Real.sqrt 3 ≤ a ∧ a ≤ 4 * Real.sqrt 2 :=
by sorry

end circle_intersection_range_l214_21408


namespace no_matrix_sin_B_l214_21470

def B : Matrix (Fin 2) (Fin 2) ℝ := !![1, 1996; 0, 1]

-- Define sin(A) using power series
noncomputable def matrix_sin (A : Matrix (Fin 2) (Fin 2) ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  A - (A^3 / 6) + (A^5 / 120) - (A^7 / 5040) + (A^9 / 362880) - (A^11 / 39916800) + higher_order_terms
where
  higher_order_terms := sorry  -- Represents the rest of the infinite series

theorem no_matrix_sin_B : ¬ ∃ (A : Matrix (Fin 2) (Fin 2) ℝ), matrix_sin A = B := by
  sorry

end no_matrix_sin_B_l214_21470


namespace bowling_team_size_l214_21465

theorem bowling_team_size (original_avg : ℝ) (new_player1_weight : ℝ) (new_player2_weight : ℝ) (new_avg : ℝ) :
  original_avg = 103 →
  new_player1_weight = 110 →
  new_player2_weight = 60 →
  new_avg = 99 →
  ∃ n : ℕ, n > 0 ∧ n * original_avg + new_player1_weight + new_player2_weight = (n + 2) * new_avg ∧ n = 7 :=
by sorry

end bowling_team_size_l214_21465


namespace solution_set_part_i_solution_range_part_ii_l214_21462

-- Define the functions f and g
def f (x : ℝ) := |x - 1|
def g (a x : ℝ) := 2 * |x + a|

-- Part I
theorem solution_set_part_i :
  {x : ℝ | f x - g 1 x > 1} = {x : ℝ | -1 < x ∧ x < -1/3} :=
sorry

-- Part II
theorem solution_range_part_ii :
  ∀ a : ℝ, (∃ x : ℝ, 2 * f x + g a x ≤ (a + 1)^2) ↔ (a ≤ -3 ∨ a ≥ 1) :=
sorry

end solution_set_part_i_solution_range_part_ii_l214_21462


namespace min_value_expression_l214_21467

theorem min_value_expression (a b m n : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_m : 0 < m) (h_pos_n : 0 < n)
  (h_sum : a + b = 1) (h_prod : m * n = 2) :
  2 ≤ (a * m + b * n) * (b * m + a * n) :=
sorry

end min_value_expression_l214_21467


namespace angle_terminal_side_point_l214_21473

theorem angle_terminal_side_point (α : Real) :
  let P : ℝ × ℝ := (4, -3)
  (P.1 = 4 ∧ P.2 = -3) →
  2 * Real.sin α + Real.cos α = -2/5 := by
sorry

end angle_terminal_side_point_l214_21473


namespace percentage_of_sum_l214_21436

theorem percentage_of_sum (x y : ℝ) (P : ℝ) 
  (h1 : 0.5 * (x - y) = (P / 100) * (x + y)) 
  (h2 : y = 0.25 * x) : 
  P = 30 := by
sorry

end percentage_of_sum_l214_21436


namespace square_root_of_four_l214_21450

theorem square_root_of_four (x : ℝ) : x^2 = 4 → x = 2 := by
  sorry

end square_root_of_four_l214_21450


namespace negation_of_existence_negation_of_quadratic_equation_l214_21437

theorem negation_of_existence (p : ℝ → Prop) : 
  (¬ ∃ x : ℝ, p x) ↔ (∀ x : ℝ, ¬ p x) := by sorry

theorem negation_of_quadratic_equation : 
  (¬ ∃ x : ℝ, x^2 - 3*x + 2 = 0) ↔ (∀ x : ℝ, x^2 - 3*x + 2 ≠ 0) := by sorry

end negation_of_existence_negation_of_quadratic_equation_l214_21437


namespace matches_for_128_teams_l214_21426

/-- Represents a single-elimination tournament -/
structure Tournament where
  num_teams : ℕ
  num_teams_positive : 0 < num_teams

/-- The number of matches required to determine the championship team -/
def matches_required (t : Tournament) : ℕ :=
  t.num_teams - 1

/-- Theorem: In a tournament with 128 teams, 127 matches are required -/
theorem matches_for_128_teams :
  ∀ t : Tournament, t.num_teams = 128 → matches_required t = 127 := by
  sorry

#check matches_for_128_teams

end matches_for_128_teams_l214_21426


namespace rectangle_diagonal_l214_21448

theorem rectangle_diagonal (l w : ℝ) (h1 : l = 8) (h2 : 2 * l + 2 * w = 46) :
  Real.sqrt (l^2 + w^2) = 17 := by
  sorry

end rectangle_diagonal_l214_21448


namespace pocket_money_calculation_l214_21493

def fifty_cent_coins : ℕ := 6
def twenty_cent_coins : ℕ := 6
def fifty_cent_value : ℚ := 0.5
def twenty_cent_value : ℚ := 0.2

theorem pocket_money_calculation :
  (fifty_cent_coins : ℚ) * fifty_cent_value + (twenty_cent_coins : ℚ) * twenty_cent_value = 4.2 := by
  sorry

end pocket_money_calculation_l214_21493


namespace max_value_problem_l214_21460

theorem max_value_problem (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) 
  (h4 : x^2 + y^2 + z^2 = 1) : 
  3 * x * z * Real.sqrt 2 + 9 * y * z ≤ Real.sqrt 27 := by
sorry

end max_value_problem_l214_21460


namespace square_side_length_average_l214_21488

theorem square_side_length_average (a b c : ℝ) 
  (ha : a = 25) (hb : b = 64) (hc : c = 225) : 
  (Real.sqrt a + Real.sqrt b + Real.sqrt c) / 3 = 28 / 3 := by
  sorry

end square_side_length_average_l214_21488


namespace bread_cost_l214_21419

theorem bread_cost (initial_amount : ℕ) (amount_left : ℕ) (num_bread : ℕ) (num_milk : ℕ) 
  (h1 : initial_amount = 47)
  (h2 : amount_left = 35)
  (h3 : num_bread = 4)
  (h4 : num_milk = 2) :
  (initial_amount - amount_left) / (num_bread + num_milk) = 2 :=
by sorry

end bread_cost_l214_21419


namespace remaining_subtasks_l214_21469

def total_problems : ℝ := 72.0
def completed_problems : ℝ := 32.0
def subtasks_per_problem : ℕ := 5

theorem remaining_subtasks : 
  (total_problems - completed_problems) * subtasks_per_problem = 200 := by
  sorry

end remaining_subtasks_l214_21469


namespace flag_arrangement_problem_l214_21409

/-- Number of blue flags -/
def blue_flags : ℕ := 10

/-- Number of green flags -/
def green_flags : ℕ := 9

/-- Total number of flags -/
def total_flags : ℕ := blue_flags + green_flags

/-- Number of flagpoles -/
def flagpoles : ℕ := 2

/-- Function to calculate the number of arrangements -/
def calculate_arrangements (a b : ℕ) : ℕ :=
  (a + 1) * Nat.choose (a + 2) b - 2 * Nat.choose (a + 1) b

/-- Theorem stating the result of the flag arrangement problem -/
theorem flag_arrangement_problem :
  calculate_arrangements blue_flags green_flags % 1000 = 310 := by
  sorry

end flag_arrangement_problem_l214_21409


namespace place_value_sum_l214_21486

/-- Given place values, prove the total number -/
theorem place_value_sum (thousands hundreds tens ones : ℕ) :
  thousands = 6 →
  hundreds = 3 →
  tens = 9 →
  ones = 7 →
  thousands * 1000 + hundreds * 100 + tens * 10 + ones = 6397 := by
  sorry

end place_value_sum_l214_21486


namespace nine_sequence_sum_to_1989_l214_21447

theorem nine_sequence_sum_to_1989 : ∃ (a b c : ℕ), 
  a + b + c = 9999999 ∧ 
  a ≤ 999 ∧ b ≤ 999 ∧ c ≤ 999 ∧
  a + b - c = 1989 := by
sorry

end nine_sequence_sum_to_1989_l214_21447


namespace triangle_inequality_bound_bound_is_tight_l214_21484

theorem triangle_inequality_bound (a b c : ℝ) (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) (h_cond : a ≥ (b + c) / 3) :
  (a * c + b * c - c^2) / (a^2 + b^2 + 3 * c^2 + 2 * a * b - 4 * b * c) ≤ (2 * Real.sqrt 2 + 1) / 7 :=
sorry

theorem bound_is_tight :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b ∧
  a ≥ (b + c) / 3 ∧
  (a * c + b * c - c^2) / (a^2 + b^2 + 3 * c^2 + 2 * a * b - 4 * b * c) = (2 * Real.sqrt 2 + 1) / 7 :=
sorry

end triangle_inequality_bound_bound_is_tight_l214_21484


namespace not_fourth_ABE_l214_21431

-- Define the set of runners
inductive Runner : Type
  | A | B | C | D | E | F

-- Define the ordering relation for runners
def beats : Runner → Runner → Prop := sorry

-- Define the race result as a function from position to runner
def raceResult : Nat → Runner := sorry

-- State the given conditions
axiom A_beats_B : beats Runner.A Runner.B
axiom A_beats_C : beats Runner.A Runner.C
axiom B_beats_D : beats Runner.B Runner.D
axiom B_beats_E : beats Runner.B Runner.E
axiom C_beats_F : beats Runner.C Runner.F
axiom E_after_B_before_C : beats Runner.B Runner.E ∧ beats Runner.E Runner.C

-- Define what it means to finish in a certain position
def finishesIn (r : Runner) (pos : Nat) : Prop :=
  raceResult pos = r

-- State the theorem
theorem not_fourth_ABE :
  ¬(finishesIn Runner.A 4) ∧ ¬(finishesIn Runner.B 4) ∧ ¬(finishesIn Runner.E 4) :=
by sorry

end not_fourth_ABE_l214_21431


namespace trig_identity_l214_21494

theorem trig_identity : 
  1 / Real.cos (70 * π / 180) - Real.sqrt 3 / Real.sin (70 * π / 180) = 1 / Real.sin (20 * π / 180) := by
  sorry

end trig_identity_l214_21494


namespace prob_diamond_ace_king_l214_21443

/-- The number of cards in the modified deck -/
def deck_size : ℕ := 56

/-- The number of cards that are either diamonds, aces, or kings -/
def target_cards : ℕ := 20

/-- The probability of drawing a card that is not a diamond, ace, or king -/
def prob_not_target : ℚ := (deck_size - target_cards) / deck_size

/-- The probability of drawing at least one diamond, ace, or king in two draws with replacement -/
def prob_at_least_one_target : ℚ := 1 - prob_not_target^2

theorem prob_diamond_ace_king : prob_at_least_one_target = 115 / 196 := by
  sorry

end prob_diamond_ace_king_l214_21443


namespace quadratic_radical_problem_l214_21483

/-- A number is a simplest quadratic radical if it cannot be further simplified -/
def IsSimplestQuadraticRadical (x : ℝ) : Prop :=
  ∃ n : ℕ, x = Real.sqrt n ∧ ∀ m : ℕ, m < n → ¬∃ k : ℕ, n = k^2 * m

/-- Two quadratic radicals are of the same type if their radicands have the same squarefree part -/
def SameTypeRadical (x y : ℝ) : Prop :=
  ∃ a b : ℕ, x = Real.sqrt a ∧ y = Real.sqrt b ∧ ∃ k m n : ℕ, k ≠ 0 ∧ m.Coprime n ∧ a = k * m ∧ b = k * n

theorem quadratic_radical_problem (a : ℝ) :
  IsSimplestQuadraticRadical (Real.sqrt (2 * a + 1)) →
  SameTypeRadical (Real.sqrt (2 * a + 1)) (Real.sqrt 48) →
  a = 1 := by
  sorry

end quadratic_radical_problem_l214_21483


namespace polynomial_value_l214_21400

theorem polynomial_value (a : ℝ) (h : a^2 + 2*a = 1) : 
  2*a^5 + 7*a^4 + 5*a^3 + 2*a^2 + 5*a + 1 = 4 := by
  sorry

end polynomial_value_l214_21400


namespace average_monthly_growth_rate_correct_l214_21425

/-- The average monthly growth rate of a factory's production volume -/
def average_monthly_growth_rate (a : ℝ) : ℝ := a^(1/11) - 1

/-- Theorem stating that the average monthly growth rate is correct -/
theorem average_monthly_growth_rate_correct (a : ℝ) (h : a > 0) :
  (1 + average_monthly_growth_rate a)^11 = a :=
by sorry

end average_monthly_growth_rate_correct_l214_21425


namespace mango_price_proof_l214_21498

/-- The cost of a single lemon in dollars -/
def lemon_cost : ℚ := 2

/-- The cost of a single papaya in dollars -/
def papaya_cost : ℚ := 1

/-- The number of fruits required to get a discount -/
def fruits_for_discount : ℕ := 4

/-- The discount amount in dollars -/
def discount_amount : ℚ := 1

/-- The number of lemons Tom bought -/
def lemons_bought : ℕ := 6

/-- The number of papayas Tom bought -/
def papayas_bought : ℕ := 4

/-- The number of mangos Tom bought -/
def mangos_bought : ℕ := 2

/-- The total amount Tom paid in dollars -/
def total_paid : ℚ := 21

/-- The cost of a single mango in dollars -/
def mango_cost : ℚ := 4

theorem mango_price_proof :
  let total_fruits := lemons_bought + papayas_bought + mangos_bought
  let total_discounts := (total_fruits / fruits_for_discount : ℚ)
  let total_discount_amount := total_discounts * discount_amount
  let total_cost_before_discount := lemon_cost * lemons_bought + papaya_cost * papayas_bought + mango_cost * mangos_bought
  total_cost_before_discount - total_discount_amount = total_paid :=
sorry

end mango_price_proof_l214_21498


namespace candles_from_beehives_l214_21428

/-- Given that 3 beehives can make enough wax for 12 candles,
    prove that 24 beehives can make enough wax for 96 candles. -/
theorem candles_from_beehives :
  ∀ (beehives candles : ℕ),
    beehives = 3 →
    candles = 12 →
    (24 : ℕ) * candles / beehives = 96 :=
by
  sorry

end candles_from_beehives_l214_21428


namespace tan_alpha_value_l214_21475

theorem tan_alpha_value (α : Real) (h : Real.tan (α + π / 4) = 1 / 7) : 
  Real.tan α = -3 / 4 := by
  sorry

end tan_alpha_value_l214_21475


namespace factorize_quadratic_minimum_value_quadratic_sum_abc_l214_21422

-- Problem 1
theorem factorize_quadratic (m : ℝ) : m^2 - 6*m + 5 = (m - 1)*(m - 5) := by sorry

-- Problem 2
theorem minimum_value_quadratic (a b : ℝ) :
  a^2 + b^2 - 4*a + 10*b + 33 ≥ 4 ∧
  (a^2 + b^2 - 4*a + 10*b + 33 = 4 ↔ a = 2 ∧ b = -5) := by sorry

-- Problem 3
theorem sum_abc (a b c : ℝ) (h1 : a - b = 8) (h2 : a*b + c^2 - 4*c + 20 = 0) :
  a + b + c = 2 := by sorry

end factorize_quadratic_minimum_value_quadratic_sum_abc_l214_21422


namespace james_savings_l214_21401

/-- Proves that James saved for 4 weeks given the problem conditions --/
theorem james_savings (w : ℕ) : 
  (10 : ℚ) * w - ((10 : ℚ) * w / 2) / 4 = 15 → w = 4 := by
  sorry

end james_savings_l214_21401


namespace equal_probability_sums_l214_21491

/-- Represents a standard six-sided die -/
def Die := Fin 6

/-- The number of dice being rolled -/
def numDice : ℕ := 8

/-- The sum we're comparing to -/
def targetSum : ℕ := 12

/-- Function to calculate the complementary sum -/
def complementarySum (n : ℕ) : ℕ := 2 * (numDice * 3 + numDice) - n

/-- Theorem stating that the sum of 44 occurs with the same probability as the sum of 12 -/
theorem equal_probability_sums :
  complementarySum targetSum = 44 := by
  sorry

end equal_probability_sums_l214_21491


namespace parabola_properties_l214_21417

def parabola (x : ℝ) : ℝ := -3 * x^2

theorem parabola_properties :
  (∀ x : ℝ, parabola x ≤ parabola 0) ∧
  (parabola 0 = 0) ∧
  (∀ x y : ℝ, x > 0 → y > x → parabola y < parabola x) :=
by sorry

end parabola_properties_l214_21417


namespace invalid_deduction_from_false_premise_l214_21424

-- Define the concept of a premise
def Premise : Type := Prop

-- Define the concept of a conclusion
def Conclusion : Type := Prop

-- Define the concept of a deduction
def Deduction := Premise → Conclusion

-- Define what it means for a premise to be false
def IsFalsePremise (p : Premise) : Prop := ¬p

-- Define what it means for a conclusion to be valid
def IsValidConclusion (c : Conclusion) : Prop := c

-- Theorem: Logical deductions based on false premises cannot lead to valid conclusions
theorem invalid_deduction_from_false_premise :
  ∀ (p : Premise) (d : Deduction),
    IsFalsePremise p → ¬(IsValidConclusion (d p)) :=
by sorry

end invalid_deduction_from_false_premise_l214_21424


namespace max_ac_value_l214_21441

theorem max_ac_value (a c x y z m n : ℤ) : 
  x^2 + a*x + 48 = (x + y)*(x + z) →
  x^2 - 8*x + c = (x + m)*(x + n) →
  y ≥ -50 → y ≤ 50 →
  z ≥ -50 → z ≤ 50 →
  m ≥ -50 → m ≤ 50 →
  n ≥ -50 → n ≤ 50 →
  ∃ (a' c' : ℤ), a'*c' = 98441 ∧ ∀ (a'' c'' : ℤ), a''*c'' ≤ 98441 :=
by sorry

end max_ac_value_l214_21441


namespace least_positive_integer_divisible_by_four_primes_l214_21420

theorem least_positive_integer_divisible_by_four_primes : ∃ n : ℕ, 
  (∃ p₁ p₂ p₃ p₄ : ℕ, Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧ 
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₃ ≠ p₄ ∧
    n % p₁ = 0 ∧ n % p₂ = 0 ∧ n % p₃ = 0 ∧ n % p₄ = 0) ∧
  (∀ m : ℕ, m < n → 
    ¬(∃ q₁ q₂ q₃ q₄ : ℕ, Prime q₁ ∧ Prime q₂ ∧ Prime q₃ ∧ Prime q₄ ∧ 
      q₁ ≠ q₂ ∧ q₁ ≠ q₃ ∧ q₁ ≠ q₄ ∧ q₂ ≠ q₃ ∧ q₂ ≠ q₄ ∧ q₃ ≠ q₄ ∧
      m % q₁ = 0 ∧ m % q₂ = 0 ∧ m % q₃ = 0 ∧ m % q₄ = 0)) ∧
  n = 210 :=
by sorry

end least_positive_integer_divisible_by_four_primes_l214_21420


namespace cubic_equation_root_range_l214_21454

theorem cubic_equation_root_range (m : ℝ) :
  (∃ x : ℝ, x ∈ Set.Icc 0 1 ∧ x^3 - 3*x - m = 0) ↔ m ∈ Set.Icc (-2) 0 := by
  sorry

end cubic_equation_root_range_l214_21454


namespace chicken_cost_l214_21471

def initial_amount : Int := 55
def banana_packs : Int := 2
def banana_cost : Int := 4
def pear_cost : Int := 2
def asparagus_cost : Int := 6
def remaining_amount : Int := 28

theorem chicken_cost : 
  initial_amount - (banana_packs * banana_cost + pear_cost + asparagus_cost) - remaining_amount = 11 := by
  sorry

end chicken_cost_l214_21471


namespace sculpture_and_base_height_l214_21452

/-- Converts feet and inches to total inches -/
def feet_inches_to_inches (feet : ℕ) (inches : ℕ) : ℕ :=
  feet * 12 + inches

/-- Converts inches to feet, rounding down -/
def inches_to_feet (inches : ℕ) : ℕ :=
  inches / 12

theorem sculpture_and_base_height :
  let sculpture_height := feet_inches_to_inches 2 10
  let base_height := 2
  let total_height := sculpture_height + base_height
  inches_to_feet total_height = 3 := by
  sorry

end sculpture_and_base_height_l214_21452


namespace max_value_ln_x_over_x_l214_21492

/-- The function f(x) = ln(x) / x attains its maximum value at e^(-1) for x > 0 -/
theorem max_value_ln_x_over_x : 
  ∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → (Real.log x) / x ≥ (Real.log y) / y ∧ (Real.log x) / x = Real.exp (-1) := by
  sorry

end max_value_ln_x_over_x_l214_21492


namespace alternating_squares_sum_l214_21435

theorem alternating_squares_sum : 
  23^2 - 21^2 + 19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 4^2 - 2^2 = 272 := by
  sorry

end alternating_squares_sum_l214_21435


namespace a_1_value_l214_21444

/-- An arithmetic sequence with common difference 2 -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + 2

/-- Three terms form a geometric sequence -/
def geometric_sequence (x y z : ℝ) : Prop :=
  y ^ 2 = x * z

theorem a_1_value (a : ℕ → ℝ) :
  arithmetic_sequence a →
  geometric_sequence (a 1) (a 3) (a 4) →
  a 1 = -8 := by
sorry

end a_1_value_l214_21444


namespace inspection_decision_l214_21451

/-- Represents the probability of an item being defective -/
def p : Real := 0.1

/-- Total number of items in a box -/
def totalItems : Nat := 200

/-- Number of items in the initial sample -/
def sampleSize : Nat := 20

/-- Number of defective items found in the sample -/
def defectivesInSample : Nat := 2

/-- Cost of inspecting one item -/
def inspectionCost : Real := 2

/-- Compensation fee for one defective item -/
def compensationFee : Real := 25

/-- Expected number of defective items in the remaining items -/
def expectedDefectives : Real := (totalItems - sampleSize) * p

/-- Expected cost without further inspection -/
def expectedCostWithoutInspection : Real :=
  sampleSize * inspectionCost + expectedDefectives * compensationFee

/-- Cost of inspecting all items -/
def costOfInspectingAll : Real := totalItems * inspectionCost

theorem inspection_decision :
  expectedCostWithoutInspection > costOfInspectingAll :=
sorry

end inspection_decision_l214_21451


namespace absolute_value_square_equivalence_l214_21456

theorem absolute_value_square_equivalence (m n : ℝ) :
  (|m| > |n| → m^2 > n^2) ∧
  (m^2 > n^2 → |m| > |n|) ∧
  (|m| ≤ |n| → m^2 ≤ n^2) ∧
  (m^2 ≤ n^2 → |m| ≤ |n|) := by
  sorry

end absolute_value_square_equivalence_l214_21456


namespace deepak_age_l214_21487

theorem deepak_age (arun_age deepak_age : ℕ) : 
  (arun_age : ℚ) / deepak_age = 4 / 3 →
  arun_age + 6 = 26 →
  deepak_age = 15 := by
sorry

end deepak_age_l214_21487


namespace power_tower_mod_500_l214_21499

theorem power_tower_mod_500 : 7^(7^(7^7)) ≡ 543 [ZMOD 500] := by
  sorry

end power_tower_mod_500_l214_21499


namespace dance_team_initial_members_l214_21423

theorem dance_team_initial_members (initial_members quit_members new_members current_members : ℕ) 
  (h1 : quit_members = 8)
  (h2 : new_members = 13)
  (h3 : current_members = 30)
  (h4 : current_members = initial_members - quit_members + new_members) : 
  initial_members = 25 := by
  sorry

end dance_team_initial_members_l214_21423


namespace other_sales_percentage_l214_21446

/-- The percentage of sales for notebooks -/
def notebooks_sales : ℝ := 25

/-- The percentage of sales for markers -/
def markers_sales : ℝ := 40

/-- The total percentage of all sales -/
def total_sales : ℝ := 100

/-- Theorem: The percentage of sales that were neither notebooks nor markers is 35% -/
theorem other_sales_percentage : 
  total_sales - (notebooks_sales + markers_sales) = 35 := by
  sorry

end other_sales_percentage_l214_21446


namespace sequence_formulas_l214_21438

def S (n : ℕ) : ℝ := sorry

def a : ℕ → ℝ
  | 0 => 1
  | n + 1 => 2 * S n + 1

def b (n : ℕ) : ℝ := (3 * n - 1) * a n

def T (n : ℕ) : ℝ := sorry

theorem sequence_formulas :
  (∀ n : ℕ, a n = 3^n) ∧
  (∀ n : ℕ, T n = ((3 * n / 2) - 5 / 4) * 3^n + 5 / 4) :=
sorry

end sequence_formulas_l214_21438


namespace residue_of_negative_935_mod_24_l214_21445

theorem residue_of_negative_935_mod_24 : 
  ∃ (r : ℤ), 0 ≤ r ∧ r < 24 ∧ -935 ≡ r [ZMOD 24] ∧ r = 1 :=
sorry

end residue_of_negative_935_mod_24_l214_21445


namespace range_of_b_l214_21407

theorem range_of_b (a b : ℝ) (h : a * b^2 > a ∧ a > a * b) : b < -1 := by
  sorry

end range_of_b_l214_21407


namespace even_composition_is_even_l214_21410

/-- A function f is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem even_composition_is_even (f : ℝ → ℝ) (h : IsEven f) : IsEven (f ∘ f) := by
  sorry

end even_composition_is_even_l214_21410


namespace angle_sum_triangle_l214_21403

theorem angle_sum_triangle (A B C : ℝ) (h : A + B = 110) : C = 70 := by
  sorry

end angle_sum_triangle_l214_21403


namespace solution_set_when_a_zero_range_of_a_no_solution_l214_21429

-- Define the function f(x) = |2x+2| - |x-1|
def f (x : ℝ) : ℝ := |2*x + 2| - |x - 1|

-- Part 1: Solution set when a = 0
theorem solution_set_when_a_zero :
  {x : ℝ | f x > 0} = {x : ℝ | x < -3 ∨ x > -1/3} := by sorry

-- Part 2: Range of a when no solution in [-4, 2]
theorem range_of_a_no_solution :
  ∀ a : ℝ, (∀ x ∈ Set.Icc (-4 : ℝ) 2, f x ≤ a) → a ≤ 3 := by sorry

end solution_set_when_a_zero_range_of_a_no_solution_l214_21429


namespace all_statements_correct_l214_21474

theorem all_statements_correct :
  (∀ a b : ℕ, Odd a → Odd b → Even (a + b)) ∧
  (∀ p : ℕ, Prime p → p > 3 → ∃ k : ℕ, p^2 = 12*k + 1) ∧
  (∀ r : ℚ, ∀ i : ℝ, Irrational i → Irrational (r + i)) ∧
  (∀ n : ℕ, 2 ∣ n → 3 ∣ n → 6 ∣ n) ∧
  (∀ n : ℕ, n > 1 → Prime n ∨ ∃ (p : List ℕ), (∀ q ∈ p, Prime q) ∧ n = p.prod) :=
by sorry

end all_statements_correct_l214_21474


namespace circle_coordinates_l214_21449

theorem circle_coordinates (π : ℝ) (h : π > 0) :
  let radii : List ℝ := [2, 4, 6, 8, 10]
  let circumference (r : ℝ) : ℝ := 2 * π * r
  let area (r : ℝ) : ℝ := π * r^2
  let coordinates := radii.map (λ r => (circumference r, area r))
  coordinates = [(4*π, 4*π), (8*π, 16*π), (12*π, 36*π), (16*π, 64*π), (20*π, 100*π)] :=
by sorry

end circle_coordinates_l214_21449


namespace henrys_cd_collection_l214_21440

theorem henrys_cd_collection :
  ∀ (classical rock country : ℕ),
    classical = 10 →
    rock = 2 * classical →
    country = rock + 3 →
    country = 23 :=
by
  sorry

end henrys_cd_collection_l214_21440


namespace video_game_players_l214_21411

/-- The number of players who quit the game -/
def players_quit : ℕ := 5

/-- The number of lives each remaining player has -/
def lives_per_player : ℕ := 5

/-- The total number of lives for remaining players -/
def total_lives : ℕ := 30

/-- The initial number of players in the game -/
def initial_players : ℕ := players_quit + total_lives / lives_per_player

theorem video_game_players :
  initial_players = 11 :=
by sorry

end video_game_players_l214_21411


namespace min_value_of_function_l214_21413

theorem min_value_of_function (t : ℝ) (h : t > 0) :
  (t^2 - 4*t + 1) / t ≥ -2 ∧ 
  ∀ ε > 0, ∃ t₀ > 0, (t₀^2 - 4*t₀ + 1) / t₀ < -2 + ε :=
sorry

end min_value_of_function_l214_21413


namespace tangent_line_at_one_e_l214_21479

/-- The tangent line to y = xe^x at (1, e) -/
theorem tangent_line_at_one_e :
  let f (x : ℝ) := x * Real.exp x
  let f' (x : ℝ) := Real.exp x + x * Real.exp x
  let tangent_line (x : ℝ) := 2 * Real.exp 1 * x - Real.exp 1
  f' 1 = 2 * Real.exp 1 ∧
  tangent_line 1 = f 1 ∧
  ∀ x, tangent_line x - f x = f' 1 * (x - 1) :=
by sorry

end tangent_line_at_one_e_l214_21479


namespace speed_increase_proof_l214_21412

def distance : ℝ := 210
def forward_time : ℝ := 7
def return_time : ℝ := 5

theorem speed_increase_proof :
  let forward_speed := distance / forward_time
  let return_speed := distance / return_time
  return_speed - forward_speed = 12 := by
  sorry

end speed_increase_proof_l214_21412


namespace work_completion_theorem_l214_21442

/-- The number of men initially doing the work -/
def initial_men : ℕ := 50

/-- The number of days it takes for the initial number of men to complete the work -/
def initial_days : ℕ := 100

/-- The number of men needed to complete the work in 20 days -/
def men_for_20_days : ℕ := 250

/-- The number of days it takes for 250 men to complete the work -/
def days_for_250_men : ℕ := 20

theorem work_completion_theorem :
  initial_men * initial_days = men_for_20_days * days_for_250_men :=
by
  sorry

#check work_completion_theorem

end work_completion_theorem_l214_21442


namespace solution_eq1_solution_eq2_l214_21430

-- Define the average method for quadratic equations
def average_method (a b c : ℝ) : Set ℝ :=
  let avg := (a + b) / 2
  let diff := b - avg
  {x | (x + avg)^2 - diff^2 = c}

-- Theorem for the first equation
theorem solution_eq1 : 
  average_method 2 8 40 = {2, -12} := by sorry

-- Theorem for the second equation
theorem solution_eq2 : 
  average_method (-2) 6 4 = {-2 + 2 * Real.sqrt 5, -2 - 2 * Real.sqrt 5} := by sorry

end solution_eq1_solution_eq2_l214_21430


namespace hyperbolas_same_asymptotes_l214_21432

/-- Two hyperbolas have the same asymptotes if M = 4.5 -/
theorem hyperbolas_same_asymptotes :
  let h₁ : ℝ → ℝ → Prop := λ x y => x^2 / 9 - y^2 / 16 = 1
  let h₂ : ℝ → ℝ → ℝ → Prop := λ x y M => y^2 / 8 - x^2 / M = 1
  let asymptote₁ : ℝ → ℝ → Prop := λ x y => y = (4/3) * x ∨ y = -(4/3) * x
  let asymptote₂ : ℝ → ℝ → ℝ → Prop := λ x y M => y = Real.sqrt (8/M) * x ∨ y = -Real.sqrt (8/M) * x
  ∀ (M : ℝ), (∀ x y, asymptote₁ x y ↔ asymptote₂ x y M) → M = 4.5 :=
by sorry

end hyperbolas_same_asymptotes_l214_21432


namespace cost_per_roof_tile_is_10_l214_21421

/-- Represents the construction costs for a house. -/
structure ConstructionCosts where
  landCostPerSqMeter : ℕ
  brickCostPer1000 : ℕ
  requiredLandArea : ℕ
  requiredBricks : ℕ
  requiredRoofTiles : ℕ
  totalCost : ℕ

/-- Calculates the cost per roof tile given the construction costs. -/
def costPerRoofTile (costs : ConstructionCosts) : ℕ :=
  let landCost := costs.landCostPerSqMeter * costs.requiredLandArea
  let brickCost := (costs.requiredBricks / 1000) * costs.brickCostPer1000
  let roofTileCost := costs.totalCost - (landCost + brickCost)
  roofTileCost / costs.requiredRoofTiles

/-- Theorem stating that the cost per roof tile is $10 given the specified construction costs. -/
theorem cost_per_roof_tile_is_10 (costs : ConstructionCosts)
    (h1 : costs.landCostPerSqMeter = 50)
    (h2 : costs.brickCostPer1000 = 100)
    (h3 : costs.requiredLandArea = 2000)
    (h4 : costs.requiredBricks = 10000)
    (h5 : costs.requiredRoofTiles = 500)
    (h6 : costs.totalCost = 106000) :
    costPerRoofTile costs = 10 := by
  sorry

end cost_per_roof_tile_is_10_l214_21421


namespace fair_coin_five_tosses_l214_21405

/-- A fair coin is a coin that has an equal probability of landing on either side when tossed. -/
def FairCoin : Type := Unit

/-- The probability of a fair coin landing on one specific side in a single toss. -/
def singleTossProbability (coin : FairCoin) : ℚ := 1 / 2

/-- The number of tosses. -/
def numTosses : ℕ := 5

/-- The probability of a fair coin landing on the same side for a given number of tosses. -/
def sameSideProbability (coin : FairCoin) (n : ℕ) : ℚ :=
  (singleTossProbability coin) ^ n

theorem fair_coin_five_tosses (coin : FairCoin) :
  sameSideProbability coin numTosses = 1 / 32 := by
  sorry

end fair_coin_five_tosses_l214_21405


namespace combined_apples_l214_21476

/-- The number of apples Sara ate -/
def sara_apples : ℕ := 16

/-- The ratio of apples Ali ate compared to Sara -/
def ali_ratio : ℕ := 4

/-- The total number of apples eaten by Ali and Sara -/
def total_apples : ℕ := sara_apples + ali_ratio * sara_apples

theorem combined_apples : total_apples = 80 := by
  sorry

end combined_apples_l214_21476


namespace fred_initial_money_l214_21468

/-- Fred's money situation --/
def fred_money_problem (initial_money current_money weekend_earnings : ℕ) : Prop :=
  initial_money + weekend_earnings = current_money

theorem fred_initial_money : 
  ∃ (initial_money : ℕ), fred_money_problem initial_money 86 63 ∧ initial_money = 23 :=
sorry

end fred_initial_money_l214_21468


namespace workshop_selection_l214_21453

/-- The number of ways to select workers for a repair job. -/
def selectWorkers (totalWorkers fitters turners masterWorkers : ℕ) : ℕ :=
  let remainingWorkers := totalWorkers - turners
  let remainingFitters := fitters + masterWorkers
  let scenario1 := Nat.choose remainingWorkers 4
  let scenario2 := Nat.choose turners 3 * Nat.choose masterWorkers 1 * Nat.choose (remainingFitters - 1) 4
  let scenario3 := Nat.choose turners 2 * Nat.choose fitters 4
  scenario1 + scenario2 + scenario3

/-- Theorem stating the number of ways to select workers for the given problem. -/
theorem workshop_selection :
  selectWorkers 11 5 4 2 = 185 := by
  sorry

end workshop_selection_l214_21453


namespace triangle_squares_l214_21466

theorem triangle_squares (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  let petya_square := a * b / (a + b)
  let vasya_square := a * b * Real.sqrt (a^2 + b^2) / (a^2 + a * b + b^2)
  -- Petya's square is larger than Vasya's square
  petya_square > vasya_square ∧ 
  -- Petya's square formula is correct
  (∃ (x : ℝ), x = petya_square ∧ x * (a + b) = a * b) ∧
  -- Vasya's square formula is correct
  (∃ (y : ℝ), y = vasya_square ∧ 
    y * (a^2 / b + b + a) = Real.sqrt (a^2 + b^2) * a) :=
by sorry

end triangle_squares_l214_21466


namespace product_of_roots_l214_21414

theorem product_of_roots (x y : ℝ) : 
  x = 16^(1/4) → y = 64^(1/2) → x * y = 16 := by sorry

end product_of_roots_l214_21414


namespace water_consumption_proof_l214_21478

/-- Proves that drinking 500 milliliters every 2 hours for 12 hours results in 3 liters of water consumption. -/
theorem water_consumption_proof (liters_goal : ℝ) (ml_per_interval : ℝ) (hours_per_interval : ℝ) :
  liters_goal = 3 ∧ ml_per_interval = 500 ∧ hours_per_interval = 2 →
  (liters_goal * 1000) / ml_per_interval * hours_per_interval = 12 := by
  sorry

end water_consumption_proof_l214_21478


namespace f_derivative_at_one_is_zero_g_derivative_formula_l214_21489

noncomputable section

def f (x : ℝ) : ℝ := Real.exp x / x

def g (x : ℝ) : ℝ := f (2 * x)

theorem f_derivative_at_one_is_zero :
  deriv f 1 = 0 := by sorry

theorem g_derivative_formula (x : ℝ) (h : x ≠ 0) :
  deriv g x = (Real.exp (2 * x) * (2 * x - 1)) / (2 * x^2) := by sorry

end f_derivative_at_one_is_zero_g_derivative_formula_l214_21489


namespace committee_selection_ways_l214_21459

def club_size : ℕ := 30
def committee_size : ℕ := 5

theorem committee_selection_ways :
  Nat.choose club_size committee_size = 142506 := by
  sorry

end committee_selection_ways_l214_21459


namespace product_63_57_l214_21416

theorem product_63_57 : 63 * 57 = 3591 := by
  sorry

end product_63_57_l214_21416


namespace digits_of_expression_l214_21404

theorem digits_of_expression : ∃ n : ℕ, n = 12 ∧ n = (Nat.digits 10 (2^15 * 5^12 - 10^5)).length := by
  sorry

end digits_of_expression_l214_21404


namespace triangle_properties_l214_21482

-- Define the triangle ABC
variable (A B C : ℝ) -- Angles of the triangle
variable (a b c : ℝ) -- Sides of the triangle opposite to A, B, C respectively

-- Define the conditions
axiom bc_cos_a : b * Real.cos A = 2
axiom area : (1/2) * b * c * Real.sin A = 2
axiom sin_relation : Real.sin B = 2 * Real.cos A * Real.sin C

-- Define the theorem
theorem triangle_properties :
  (Real.tan A = 2) ∧ (c = 5) :=
sorry

end triangle_properties_l214_21482


namespace min_value_reciprocal_sum_l214_21406

theorem min_value_reciprocal_sum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 3 * b = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + 3 * y = 1 → 1 / a + 1 / b ≤ 1 / x + 1 / y) ∧
  (1 / a + 1 / b = 4 + 2 * Real.sqrt 3) :=
sorry

end min_value_reciprocal_sum_l214_21406


namespace marbles_problem_l214_21481

theorem marbles_problem (jinwoo seonghyeon cheolsu : ℕ) : 
  jinwoo = (2 * seonghyeon) / 3 →
  cheolsu = 72 →
  jinwoo + cheolsu = 2 * seonghyeon →
  jinwoo = 36 := by
  sorry

end marbles_problem_l214_21481


namespace opposite_of_one_sixth_l214_21480

theorem opposite_of_one_sixth :
  -(1 / 6 : ℚ) = -1 / 6 := by sorry

end opposite_of_one_sixth_l214_21480


namespace bolts_per_box_l214_21463

theorem bolts_per_box (bolt_boxes : ℕ) (nut_boxes : ℕ) (nuts_per_box : ℕ) 
  (bolts_left : ℕ) (nuts_left : ℕ) (bolts_and_nuts_used : ℕ) :
  bolt_boxes = 7 →
  nut_boxes = 3 →
  nuts_per_box = 15 →
  bolts_left = 3 →
  nuts_left = 6 →
  bolts_and_nuts_used = 113 →
  ∃ (bolts_per_box : ℕ),
    bolt_boxes * bolts_per_box + nut_boxes * nuts_per_box = 
    bolts_and_nuts_used + bolts_left + nuts_left ∧
    bolts_per_box = 11 :=
by sorry

end bolts_per_box_l214_21463


namespace softball_team_ratio_l214_21495

/-- Represents a co-ed softball team --/
structure Team where
  men : ℕ
  women : ℕ
  total : ℕ
  h1 : women = men + 4
  h2 : men + women = total

/-- The ratio of men to women in a team --/
def menWomenRatio (t : Team) : Rat :=
  t.men / t.women

theorem softball_team_ratio (t : Team) (h : t.total = 14) :
  menWomenRatio t = 5 / 9 := by
  sorry

end softball_team_ratio_l214_21495


namespace line_maximizing_midpoint_distance_l214_21427

/-- The equation of a line that intercepts a circle, maximizing the distance from the origin to the chord's midpoint -/
theorem line_maximizing_midpoint_distance 
  (x y a b c : ℝ) 
  (circle_eq : x^2 + y^2 = 16)
  (line_eq : a*x + b*y + c = 0)
  (condition : a + 2*b - c = 0)
  (is_max : ∀ (x' y' : ℝ), x'^2 + y'^2 ≤ (x^2 + y^2) / 4) :
  x + 2*y + 5 = 0 := by
sorry

end line_maximizing_midpoint_distance_l214_21427


namespace better_value_is_16_cents_per_ounce_l214_21415

/-- Represents a box of macaroni and cheese -/
structure MacaroniBox where
  weight : ℕ  -- weight in ounces
  price : ℕ   -- price in cents

/-- Calculates the price per ounce for a given box -/
def pricePerOunce (box : MacaroniBox) : ℚ :=
  box.price / box.weight

/-- Finds the box with the lowest price per ounce -/
def bestValue (box1 box2 : MacaroniBox) : MacaroniBox :=
  if pricePerOunce box1 ≤ pricePerOunce box2 then box1 else box2

theorem better_value_is_16_cents_per_ounce :
  let largerBox : MacaroniBox := ⟨30, 480⟩
  let smallerBox : MacaroniBox := ⟨20, 340⟩
  pricePerOunce (bestValue largerBox smallerBox) = 16 / 1 := by
  sorry

end better_value_is_16_cents_per_ounce_l214_21415


namespace multiple_power_divisibility_l214_21433

theorem multiple_power_divisibility (a n m : ℕ) (ha : a > 0) : 
  m % (a^n) = 0 → (a + 1)^m - 1 % (a^(n+1)) = 0 := by
  sorry

end multiple_power_divisibility_l214_21433


namespace supplementary_angles_ratio_l214_21458

theorem supplementary_angles_ratio (a b : ℝ) : 
  a + b = 180 →  -- angles are supplementary
  a / b = 5 / 4 →  -- angles are in ratio 5:4
  b = 80 :=  -- smaller angle is 80°
by sorry

end supplementary_angles_ratio_l214_21458


namespace circle_line_intersection_k_range_l214_21497

/-- Given a circle and a line, if there exists a point on the line such that a circle 
    with this point as its center and radius 1 has a common point with the given circle, 
    then k is within a specific range. -/
theorem circle_line_intersection_k_range :
  ∀ (k : ℝ),
  (∃ (x y : ℝ), x^2 + y^2 + 4*x + 3 = 0 ∧ y = k*x - 1 ∧
   ∃ (x₀ y₀ : ℝ), y₀ = k*x₀ - 1 ∧ 
   ∃ (x₁ y₁ : ℝ), (x₁ - x₀)^2 + (y₁ - y₀)^2 = 1 ∧ x₁^2 + y₁^2 + 4*x₁ + 3 = 0) →
  -4/3 ≤ k ∧ k ≤ 0 :=
by sorry

end circle_line_intersection_k_range_l214_21497


namespace monomial_sum_condition_l214_21455

theorem monomial_sum_condition (a b : ℕ) (m n : ℕ) : 
  (∃ k : ℕ, 2 * a^(m+2) * b^(2*n+2) + a^3 * b^8 = k * a^(m+2) * b^(2*n+2)) → 
  m = 1 ∧ n = 3 :=
by sorry

end monomial_sum_condition_l214_21455


namespace total_bike_ride_l214_21439

def morning_ride : ℝ := 2
def evening_ride_factor : ℝ := 5

theorem total_bike_ride : morning_ride + evening_ride_factor * morning_ride = 12 := by
  sorry

end total_bike_ride_l214_21439


namespace quadratic_solution_difference_squared_l214_21464

theorem quadratic_solution_difference_squared :
  ∀ p q : ℝ,
  (5 * p^2 - 8 * p - 15 = 0) →
  (5 * q^2 - 8 * q - 15 = 0) →
  (p - q)^2 = 14.5924 := by
sorry

end quadratic_solution_difference_squared_l214_21464


namespace stratified_sample_theorem_l214_21461

/-- Represents the stratified sampling problem -/
structure StratifiedSample where
  total_students : ℕ
  male_students : ℕ
  female_students : ℕ
  sample_size : ℕ
  interview_size : ℕ

/-- Calculates the number of male students in the sample -/
def male_in_sample (s : StratifiedSample) : ℕ :=
  (s.sample_size * s.male_students) / s.total_students

/-- Calculates the number of female students in the sample -/
def female_in_sample (s : StratifiedSample) : ℕ :=
  (s.sample_size * s.female_students) / s.total_students

/-- Calculates the probability of selecting exactly one female student for interview -/
def prob_one_female (s : StratifiedSample) : ℚ :=
  let male_count := male_in_sample s
  let female_count := female_in_sample s
  (male_count * female_count : ℚ) / ((s.sample_size * (s.sample_size - 1)) / 2 : ℚ)

/-- The main theorem to be proved -/
theorem stratified_sample_theorem (s : StratifiedSample) 
  (h1 : s.total_students = 50)
  (h2 : s.male_students = 30)
  (h3 : s.female_students = 20)
  (h4 : s.sample_size = 5)
  (h5 : s.interview_size = 2) :
  male_in_sample s = 3 ∧ 
  female_in_sample s = 2 ∧ 
  prob_one_female s = 3/5 := by
  sorry

end stratified_sample_theorem_l214_21461
