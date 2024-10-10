import Mathlib

namespace ice_pop_price_is_correct_l277_27751

/-- The selling price of an ice-pop that allows a school to buy pencils, given:
  * The cost to make each ice-pop
  * The cost of each pencil
  * The number of ice-pops that need to be sold
  * The number of pencils to be bought
-/
def ice_pop_selling_price (make_cost : ℚ) (pencil_cost : ℚ) (pops_sold : ℕ) (pencils_bought : ℕ) : ℚ :=
  make_cost + (pencil_cost * pencils_bought - make_cost * pops_sold) / pops_sold

/-- Theorem stating that the selling price of each ice-pop is $1.20 under the given conditions -/
theorem ice_pop_price_is_correct :
  ice_pop_selling_price 0.90 1.80 300 100 = 1.20 := by
  sorry

end ice_pop_price_is_correct_l277_27751


namespace christen_peeled_18_potatoes_l277_27711

/-- Represents the potato peeling scenario -/
structure PotatoPeeling where
  total_potatoes : ℕ
  homer_rate : ℕ
  christen_rate : ℕ
  christen_join_time : ℕ

/-- Calculates the number of potatoes Christen peeled -/
def potatoes_peeled_by_christen (scenario : PotatoPeeling) : ℕ :=
  sorry

/-- Theorem stating that Christen peeled 18 potatoes in the given scenario -/
theorem christen_peeled_18_potatoes :
  let scenario : PotatoPeeling := {
    total_potatoes := 50,
    homer_rate := 4,
    christen_rate := 6,
    christen_join_time := 5
  }
  potatoes_peeled_by_christen scenario = 18 := by
  sorry

end christen_peeled_18_potatoes_l277_27711


namespace number_divided_by_16_equals_4_l277_27787

theorem number_divided_by_16_equals_4 (x : ℤ) : x / 16 = 4 → x = 64 := by
  sorry

end number_divided_by_16_equals_4_l277_27787


namespace max_product_l277_27794

def digits : List ℕ := [1, 3, 5, 8, 9]

def is_valid_arrangement (a b c d e : ℕ) : Prop :=
  {a, b, c, d, e} = digits.toFinset

def four_digit_num (a b c d : ℕ) : ℕ := 
  1000 * a + 100 * b + 10 * c + d

def product (a b c d e : ℕ) : ℕ :=
  (four_digit_num a b c d) * e

theorem max_product :
  ∀ a b c d e,
    is_valid_arrangement a b c d e →
    product a b c d e ≤ product 8 5 3 1 9 :=
sorry

end max_product_l277_27794


namespace pencil_price_l277_27741

theorem pencil_price (x y : ℚ) 
  (eq1 : 5 * x + 4 * y = 315)
  (eq2 : 3 * x + 6 * y = 243) :
  y = 15 := by
sorry

end pencil_price_l277_27741


namespace rotation_center_l277_27744

noncomputable def f (z : ℂ) : ℂ := ((-1 - Complex.I * Real.sqrt 3) * z + (-2 * Real.sqrt 3 + 18 * Complex.I)) / 2

theorem rotation_center :
  ∃ (c : ℂ), f c = c ∧ c = -2 * Real.sqrt 3 - 4 * Complex.I :=
sorry

end rotation_center_l277_27744


namespace max_prob_with_highest_prob_second_l277_27710

/-- Represents a chess player's probability of winning against an opponent -/
structure PlayerProb where
  prob : ℝ
  pos : prob > 0

/-- Represents the probabilities of winning against three players -/
structure ThreePlayerProbs where
  p₁ : PlayerProb
  p₂ : PlayerProb
  p₃ : PlayerProb
  p₃_gt_p₂ : p₃.prob > p₂.prob
  p₂_gt_p₁ : p₂.prob > p₁.prob

/-- Calculates the probability of winning two consecutive games given the order of opponents -/
def prob_two_consecutive_wins (probs : ThreePlayerProbs) (second_player : ℕ) : ℝ :=
  match second_player with
  | 1 => 2 * (probs.p₁.prob * (probs.p₂.prob + probs.p₃.prob) - 2 * probs.p₁.prob * probs.p₂.prob * probs.p₃.prob)
  | 2 => 2 * (probs.p₂.prob * (probs.p₁.prob + probs.p₃.prob) - 2 * probs.p₁.prob * probs.p₂.prob * probs.p₃.prob)
  | _ => 2 * (probs.p₁.prob * probs.p₃.prob + probs.p₂.prob * probs.p₃.prob - 2 * probs.p₁.prob * probs.p₂.prob * probs.p₃.prob)

theorem max_prob_with_highest_prob_second (probs : ThreePlayerProbs) :
  ∀ i, prob_two_consecutive_wins probs 3 ≥ prob_two_consecutive_wins probs i :=
sorry

end max_prob_with_highest_prob_second_l277_27710


namespace negation_of_implication_l277_27748

theorem negation_of_implication (x : ℝ) :
  ¬(x > 2 → x^2 - 3*x + 2 > 0) ↔ (x ≤ 2 → x^2 - 3*x + 2 ≤ 0) :=
by sorry

end negation_of_implication_l277_27748


namespace roots_of_equation_l277_27727

theorem roots_of_equation (x : ℝ) : 
  (2 * x^2 - x = 0) ↔ (x = 0 ∨ x = 1/2) := by sorry

end roots_of_equation_l277_27727


namespace construction_time_difference_l277_27796

/-- Represents the work rate of one person per day -/
def work_rate : ℝ := 1

/-- Calculates the total work done given the number of workers, days, and work rate -/
def total_work (workers : ℕ) (days : ℕ) (rate : ℝ) : ℝ :=
  (workers : ℝ) * (days : ℝ) * rate

/-- Theorem: If 100 men work for 50 days and then 200 men work for another 50 days
    to complete a project in 100 days, it would take 150 days for 100 men to
    complete the same project working at the same rate. -/
theorem construction_time_difference :
  let initial_workers : ℕ := 100
  let additional_workers : ℕ := 100
  let initial_days : ℕ := 50
  let total_days : ℕ := 100
  let work_done_first_half := total_work initial_workers initial_days work_rate
  let work_done_second_half := total_work (initial_workers + additional_workers) initial_days work_rate
  let total_work_done := work_done_first_half + work_done_second_half
  total_work initial_workers 150 work_rate = total_work_done :=
by
  sorry

end construction_time_difference_l277_27796


namespace half_angle_quadrant_l277_27745

theorem half_angle_quadrant (α : Real) (h : ∃ k : ℤ, 2 * k * π + π / 2 < α ∧ α < 2 * k * π + π) :
  ∃ m : ℤ, (m * π < α / 2 ∧ α / 2 < m * π + π / 2) ∨ 
           (m * π + π < α / 2 ∧ α / 2 < m * π + 3 * π / 2) :=
by sorry

end half_angle_quadrant_l277_27745


namespace problem_statement_l277_27735

theorem problem_statement (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : a * b = 10) :
  (Real.log a + Real.log b > 0) ∧
  (Real.log a - Real.log b > 0) ∧
  (Real.log a * Real.log b < 1/4) ∧
  (¬ ∀ x y : ℝ, x > y ∧ y > 0 ∧ x * y = 10 → Real.log x / Real.log y > 1) :=
by sorry

end problem_statement_l277_27735


namespace i_power_sum_i_power_sum_proof_l277_27752

theorem i_power_sum : Complex → Prop :=
  fun i => i * i = -1 → i^20 + i^35 = 1 - i

-- The proof would go here, but we're skipping it as requested
theorem i_power_sum_proof : i_power_sum Complex.I :=
  sorry

end i_power_sum_i_power_sum_proof_l277_27752


namespace nancy_money_total_l277_27779

/-- Given that Nancy has 9 5-dollar bills, prove that she has $45 in total. -/
theorem nancy_money_total :
  let num_bills : ℕ := 9
  let bill_value : ℕ := 5
  num_bills * bill_value = 45 := by
sorry

end nancy_money_total_l277_27779


namespace james_chore_time_l277_27737

/-- The time James spends vacuuming, in hours -/
def vacuum_time : ℝ := 3

/-- The factor by which the time spent on other chores exceeds vacuuming time -/
def other_chores_factor : ℝ := 3

/-- The total time James spends on his chores, in hours -/
def total_chore_time : ℝ := vacuum_time + other_chores_factor * vacuum_time

theorem james_chore_time : total_chore_time = 12 := by
  sorry

end james_chore_time_l277_27737


namespace diophantine_equation_solution_l277_27732

theorem diophantine_equation_solution (n : ℕ+) (a b c : ℕ+) 
  (ha : a ≤ 3 * n ^ 2 + 4 * n) 
  (hb : b ≤ 3 * n ^ 2 + 4 * n) 
  (hc : c ≤ 3 * n ^ 2 + 4 * n) : 
  ∃ (x y z : ℤ), 
    (abs x ≤ 2 * n ∧ abs y ≤ 2 * n ∧ abs z ≤ 2 * n) ∧ 
    (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) ∧
    (a * x + b * y + c * z = 0) :=
by sorry

end diophantine_equation_solution_l277_27732


namespace monotonic_sine_phi_range_l277_27705

theorem monotonic_sine_phi_range (f : ℝ → ℝ) (φ : ℝ) :
  (∀ x, f x = -2 * Real.sin (2 * x + φ)) →
  (|φ| < π) →
  (∀ x ∈ Set.Ioo (π / 5) ((5 / 8) * π), StrictMono f) →
  φ ∈ Set.Ioo (π / 10) (π / 4) := by
sorry

end monotonic_sine_phi_range_l277_27705


namespace sqrt_3600_equals_60_l277_27746

theorem sqrt_3600_equals_60 : Real.sqrt 3600 = 60 := by
  sorry

end sqrt_3600_equals_60_l277_27746


namespace dislike_tv_and_books_l277_27786

/-- Given a population where some dislike TV and some of those also dislike books,
    calculate the number of people who dislike both TV and books. -/
theorem dislike_tv_and_books
  (total_population : ℕ)
  (tv_dislike_percent : ℚ)
  (book_dislike_percent : ℚ)
  (h_total : total_population = 1500)
  (h_tv : tv_dislike_percent = 25 / 100)
  (h_book : book_dislike_percent = 15 / 100) :
  ⌊(tv_dislike_percent * book_dislike_percent * total_population : ℚ)⌋ = 56 := by
  sorry

end dislike_tv_and_books_l277_27786


namespace dora_receives_two_packs_l277_27703

/-- The number of packs of stickers Dora receives --/
def dora_sticker_packs (allowance : ℕ) (card_price : ℕ) (sticker_price : ℕ) (num_people : ℕ) : ℕ :=
  let total_money := allowance * num_people
  let remaining_money := total_money - card_price
  let total_sticker_packs := remaining_money / sticker_price
  total_sticker_packs / num_people

/-- Theorem stating that Dora receives 2 packs of stickers --/
theorem dora_receives_two_packs :
  dora_sticker_packs 9 10 2 2 = 2 := by
  sorry

end dora_receives_two_packs_l277_27703


namespace roots_expression_value_l277_27776

theorem roots_expression_value (x₁ x₂ : ℝ) : 
  x₁^2 - 3*x₁ + 1 = 0 → x₂^2 - 3*x₂ + 1 = 0 → 
  (x₁ + x₂) / (1 + x₁ * x₂) = 3/2 := by sorry

end roots_expression_value_l277_27776


namespace colored_balls_probabilities_l277_27713

/-- Represents a bag of colored balls -/
structure ColoredBalls where
  total : ℕ
  red : ℕ
  yellow : ℕ

/-- Calculate the probability of drawing a ball of a specific color -/
def probability (bag : ColoredBalls) (color : ℕ) : ℚ :=
  color / bag.total

/-- Calculate the number of red balls to add to achieve a target probability -/
def addRedBalls (bag : ColoredBalls) (targetProb : ℚ) : ℕ :=
  let x := (targetProb * bag.total - bag.red) / (1 - targetProb)
  x.ceil.toNat

theorem colored_balls_probabilities (bag : ColoredBalls) :
  bag.total = 10 ∧ bag.red = 4 ∧ bag.yellow = 6 →
  (probability bag bag.yellow = 3/5) ∧
  (addRedBalls bag (2/3) = 8) := by
  sorry

end colored_balls_probabilities_l277_27713


namespace integer_linear_combination_sqrt2_sqrt3_l277_27792

theorem integer_linear_combination_sqrt2_sqrt3 (a b c : ℤ) :
  a * Real.sqrt 2 + b * Real.sqrt 3 + c = 0 → a = 0 ∧ b = 0 ∧ c = 0 := by
sorry

end integer_linear_combination_sqrt2_sqrt3_l277_27792


namespace fred_has_nine_dimes_l277_27720

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The amount of money Fred has in cents -/
def fred_money : ℕ := 90

/-- The number of dimes Fred has -/
def fred_dimes : ℕ := fred_money / dime_value

theorem fred_has_nine_dimes : fred_dimes = 9 := by
  sorry

end fred_has_nine_dimes_l277_27720


namespace quadratic_equation_value_l277_27715

/-- Given a quadratic equation y = ax^2 + bx + c, 
    this theorem proves that 2a - b + c = 2 
    when a = 2, b = 3, and c = 1 -/
theorem quadratic_equation_value (a b c : ℝ) : 
  a = 2 ∧ b = 3 ∧ c = 1 → 2*a - b + c = 2 := by sorry

end quadratic_equation_value_l277_27715


namespace millennium_running_time_l277_27795

/-- The running time of Millennium in minutes -/
def millennium_time : ℕ := 120

/-- The running time of Alpha Epsilon in minutes -/
def alpha_epsilon_time : ℕ := millennium_time - 30

/-- The running time of Beast of War: Armoured Command in minutes -/
def beast_of_war_time : ℕ := alpha_epsilon_time + 10

/-- Theorem stating that Millennium's running time is 120 minutes -/
theorem millennium_running_time : 
  millennium_time = 120 ∧ 
  alpha_epsilon_time = millennium_time - 30 ∧
  beast_of_war_time = alpha_epsilon_time + 10 ∧
  beast_of_war_time = 100 :=
by sorry

end millennium_running_time_l277_27795


namespace geometric_series_sum_l277_27782

theorem geometric_series_sum (a : ℕ+) (n : ℕ+) (h : (a : ℝ) / (1 - 1 / (n : ℝ)) = 3) :
  (a : ℝ) + (a : ℝ) / (n : ℝ) = 8 / 3 := by
  sorry

end geometric_series_sum_l277_27782


namespace original_equals_scientific_l277_27754

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The number we want to represent in scientific notation -/
def original_number : ℕ := 650000

/-- The scientific notation representation of the original number -/
def scientific_repr : ScientificNotation := {
  coefficient := 6.5,
  exponent := 5,
  is_valid := by sorry
}

/-- Theorem stating that the original number is equal to its scientific notation representation -/
theorem original_equals_scientific : 
  (original_number : ℝ) = scientific_repr.coefficient * (10 : ℝ) ^ scientific_repr.exponent := by
  sorry

end original_equals_scientific_l277_27754


namespace largest_non_sum_30_composite_l277_27738

def is_composite (n : ℕ) : Prop := ∃ k m : ℕ, k > 1 ∧ m > 1 ∧ n = k * m

def is_sum_of_multiple_30_and_composite (n : ℕ) : Prop :=
  ∃ k m : ℕ, k > 0 ∧ is_composite m ∧ n = 30 * k + m

theorem largest_non_sum_30_composite : 
  (∀ n : ℕ, n > 217 → is_sum_of_multiple_30_and_composite n) ∧
  ¬is_sum_of_multiple_30_and_composite 217 :=
sorry

end largest_non_sum_30_composite_l277_27738


namespace mosquito_shadow_speed_l277_27704

/-- The speed of a mosquito's shadow across the bottom of a water body -/
theorem mosquito_shadow_speed 
  (v : Real) 
  (h : Real) 
  (t : Real) 
  (cos_incidence : Real) : 
  v = 1 → 
  h = 3 → 
  t = 5 → 
  cos_incidence = 0.6 → 
  ∃ (shadow_speed : Real), 
    shadow_speed = 1.6 ∨ shadow_speed = 0 :=
by sorry

end mosquito_shadow_speed_l277_27704


namespace least_m_for_x_sequence_l277_27706

def x : ℕ → ℚ
  | 0 => 7
  | n + 1 => (x n ^ 2 + 7 * x n + 12) / (x n + 8)

theorem least_m_for_x_sequence :
  ∃ m : ℕ, (∀ k < m, x k > 6 + 1 / 2^22) ∧ x m ≤ 6 + 1 / 2^22 ∧ m = 204 :=
by sorry

end least_m_for_x_sequence_l277_27706


namespace shooting_to_total_ratio_total_time_breakdown_running_weightlifting_relation_l277_27722

/-- Represents Kyle's basketball practice schedule --/
structure BasketballPractice where
  total_time : ℕ           -- Total practice time in minutes
  weightlifting_time : ℕ   -- Time spent weightlifting in minutes
  running_time : ℕ         -- Time spent running in minutes
  shooting_time : ℕ        -- Time spent shooting in minutes

/-- Kyle's basketball practice satisfies the given conditions --/
def kyle_practice : BasketballPractice :=
  { total_time := 120,        -- 2 hours = 120 minutes
    weightlifting_time := 20, -- Given in the problem
    running_time := 40,       -- Twice the weightlifting time
    shooting_time := 60 }     -- Remaining time

/-- The ratio of shooting time to total practice time is 1:2 --/
theorem shooting_to_total_ratio :
  kyle_practice.shooting_time * 2 = kyle_practice.total_time :=
by sorry

/-- All practice activities sum up to the total time --/
theorem total_time_breakdown :
  kyle_practice.weightlifting_time + kyle_practice.running_time + kyle_practice.shooting_time = kyle_practice.total_time :=
by sorry

/-- Running time is twice the weightlifting time --/
theorem running_weightlifting_relation :
  kyle_practice.running_time = 2 * kyle_practice.weightlifting_time :=
by sorry

end shooting_to_total_ratio_total_time_breakdown_running_weightlifting_relation_l277_27722


namespace first_year_after_2000_sum_12_correct_l277_27774

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Check if a year is after 2000 -/
def is_after_2000 (year : ℕ) : Prop := year > 2000

/-- The first year after 2000 with sum of digits 12 -/
def first_year_after_2000_sum_12 : ℕ := 2019

theorem first_year_after_2000_sum_12_correct :
  (is_after_2000 first_year_after_2000_sum_12) ∧
  (sum_of_digits first_year_after_2000_sum_12 = 12) ∧
  (∀ y : ℕ, is_after_2000 y ∧ sum_of_digits y = 12 → y ≥ first_year_after_2000_sum_12) :=
by sorry

end first_year_after_2000_sum_12_correct_l277_27774


namespace student_count_l277_27730

/-- The number of storybooks distributed to the class -/
def total_books : ℕ := 60

/-- The number of students in the class -/
def num_students : ℕ := 20

theorem student_count :
  (num_students < total_books) ∧ 
  (total_books - num_students) % 2 = 0 ∧
  (total_books - num_students) / 2 = num_students :=
by sorry

end student_count_l277_27730


namespace sandy_money_left_l277_27765

/-- The amount of money Sandy has left after buying a pie -/
def money_left (initial_amount pie_cost : ℕ) : ℕ :=
  initial_amount - pie_cost

/-- Theorem: Sandy has 57 dollars left after buying the pie -/
theorem sandy_money_left :
  money_left 63 6 = 57 :=
by sorry

end sandy_money_left_l277_27765


namespace consecutive_odd_numbers_problem_l277_27773

theorem consecutive_odd_numbers_problem (x : ℤ) : 
  Odd x ∧ 
  (8 * x = 3 * (x + 4) + 2 * (x + 2) + 5) ∧ 
  (∃ p : ℕ, Prime p ∧ (x + (x + 2) + (x + 4)) % p = 0) → 
  x = 7 := by sorry

end consecutive_odd_numbers_problem_l277_27773


namespace max_pieces_with_three_cuts_l277_27700

/-- Represents a cake that can be cut -/
structure Cake :=
  (volume : ℝ)
  (height : ℝ)
  (width : ℝ)
  (depth : ℝ)

/-- Represents a cut made to the cake -/
inductive Cut
  | Horizontal
  | Vertical
  | Parallel

/-- The number of pieces resulting from a series of cuts -/
def num_pieces (cuts : List Cut) : ℕ :=
  2 ^ (cuts.length)

/-- The maximum number of identical pieces obtainable with 3 cuts -/
def max_pieces : ℕ := 8

/-- Theorem: The maximum number of identical pieces obtainable from a cake with 3 cuts is 8 -/
theorem max_pieces_with_three_cuts (c : Cake) :
  ∀ (cuts : List Cut), cuts.length = 3 → num_pieces cuts ≤ max_pieces :=
by sorry

end max_pieces_with_three_cuts_l277_27700


namespace class_average_mark_l277_27756

theorem class_average_mark (total_students : Nat) (excluded_students : Nat) 
  (excluded_avg : ℝ) (remaining_avg : ℝ) : 
  total_students = 25 → 
  excluded_students = 5 → 
  excluded_avg = 40 → 
  remaining_avg = 90 → 
  (total_students * (total_students * remaining_avg - excluded_students * excluded_avg)) / 
    (total_students * (total_students - excluded_students)) = 80 := by
  sorry

end class_average_mark_l277_27756


namespace two_abs_plus_x_nonnegative_l277_27749

theorem two_abs_plus_x_nonnegative (x : ℚ) : 2 * |x| + x ≥ 0 := by
  sorry

end two_abs_plus_x_nonnegative_l277_27749


namespace difference_squared_l277_27799

theorem difference_squared (x y a b : ℝ) 
  (h1 : x * y = b) 
  (h2 : x / y + y / x = a) : 
  (x - y)^2 = a * b - 2 * b := by
sorry

end difference_squared_l277_27799


namespace triangle_stack_impossibility_l277_27716

theorem triangle_stack_impossibility : ¬ ∃ (n : ℕ), n > 0 ∧ (n * (1 + 2 + 3)) / 3 = 1997 := by
  sorry

end triangle_stack_impossibility_l277_27716


namespace nathan_tomato_harvest_l277_27772

/-- Represents the harvest and sales data for Nathan's garden --/
structure GardenData where
  strawberry_plants : ℕ
  tomato_plants : ℕ
  strawberries_per_plant : ℕ
  fruits_per_basket : ℕ
  strawberry_basket_price : ℕ
  tomato_basket_price : ℕ
  total_revenue : ℕ

/-- Calculates the number of tomatoes harvested per plant --/
def tomatoes_per_plant (data : GardenData) : ℕ :=
  let strawberry_baskets := (data.strawberry_plants * data.strawberries_per_plant) / data.fruits_per_basket
  let strawberry_revenue := strawberry_baskets * data.strawberry_basket_price
  let tomato_revenue := data.total_revenue - strawberry_revenue
  let tomato_baskets := tomato_revenue / data.tomato_basket_price
  let total_tomatoes := tomato_baskets * data.fruits_per_basket
  total_tomatoes / data.tomato_plants

/-- Theorem stating that given Nathan's garden data, he harvested 16 tomatoes per plant --/
theorem nathan_tomato_harvest :
  let data : GardenData := {
    strawberry_plants := 5,
    tomato_plants := 7,
    strawberries_per_plant := 14,
    fruits_per_basket := 7,
    strawberry_basket_price := 9,
    tomato_basket_price := 6,
    total_revenue := 186
  }
  tomatoes_per_plant data = 16 := by
  sorry

end nathan_tomato_harvest_l277_27772


namespace y_intercept_of_line_l277_27714

/-- The y-intercept of the line 5x - 3y = 15 is (0, -5) -/
theorem y_intercept_of_line (x y : ℝ) :
  5 * x - 3 * y = 15 → y = -5 ∧ x = 0 := by
  sorry

end y_intercept_of_line_l277_27714


namespace train_distance_problem_l277_27747

/-- Proves that the distance between two stations is 540 km given the conditions of the train problem -/
theorem train_distance_problem (v1 v2 : ℝ) (d : ℝ) :
  v1 = 20 →  -- Speed of train 1 in km/hr
  v2 = 25 →  -- Speed of train 2 in km/hr
  v2 > v1 →  -- Train 2 is faster than train 1
  d = (v2 - v1) * (v1 * v2)⁻¹ * 60 →  -- Difference in distance traveled
  v1 * ((v1 + v2) * (v2 - v1)⁻¹ * 60) + 
  v2 * ((v1 + v2) * (v2 - v1)⁻¹ * 60) = 540 :=
by sorry

end train_distance_problem_l277_27747


namespace max_area_constrained_rectangle_l277_27733

/-- Represents a rectangular garden with given length and width. -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangle. -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- Calculates the area of a rectangle. -/
def area (r : Rectangle) : ℝ := r.length * r.width

/-- Checks if a rectangle satisfies the given constraints. -/
def satisfiesConstraints (r : Rectangle) : Prop :=
  perimeter r = 400 ∧ r.length ≥ 100 ∧ r.width ≥ 50

/-- States that the maximum area of a constrained rectangle is 7500. -/
theorem max_area_constrained_rectangle :
  ∀ r : Rectangle, satisfiesConstraints r → area r ≤ 7500 :=
by sorry

end max_area_constrained_rectangle_l277_27733


namespace tire_price_proof_l277_27761

/-- The regular price of a tire -/
def regular_price : ℝ := 115.71

/-- The total amount paid for four tires -/
def total_paid : ℝ := 405

/-- The promotion deal: 3 tires at regular price, 1 at half price -/
theorem tire_price_proof :
  3 * regular_price + (1/2) * regular_price = total_paid :=
by sorry

end tire_price_proof_l277_27761


namespace unique_root_implies_specific_angles_l277_27788

/-- Given α ∈ (0, π), if the equation |2x - 1/2| + |(\sqrt{6} - \sqrt{2})x| = sin α
    has exactly one real root, then α = π/12 or α = 11π/12 -/
theorem unique_root_implies_specific_angles (α : Real) 
    (h1 : α ∈ Set.Ioo 0 Real.pi)
    (h2 : ∃! x : Real, |2*x - 1/2| + |((Real.sqrt 6) - (Real.sqrt 2))*x| = Real.sin α) :
    α = Real.pi/12 ∨ α = 11*Real.pi/12 := by
  sorry

end unique_root_implies_specific_angles_l277_27788


namespace triangle_perimeter_l277_27729

/-- Given a triangle with two sides of lengths 4 and 6, and the third side length
    being a root of (x-6)(x-10)=0, prove that the perimeter of the triangle is 16. -/
theorem triangle_perimeter : ∀ x : ℝ, 
  (x - 6) * (x - 10) = 0 → 
  (4 < x ∧ x < 10) →  -- Triangle inequality
  4 + 6 + x = 16 := by
sorry

end triangle_perimeter_l277_27729


namespace unique_solution_l277_27702

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ → ℝ) : Prop :=
  ∀ a b c : ℝ, f (a^2) (f b c + 1) = a^2 * (b * c + 1)

/-- Theorem stating that the only function satisfying the equation is f(a,b) = a*b -/
theorem unique_solution {f : ℝ → ℝ → ℝ} (hf : SatisfiesEquation f) :
  ∀ a b : ℝ, f a b = a * b := by sorry

end unique_solution_l277_27702


namespace seedling_packaging_l277_27789

/-- The number of seedlings to be placed in packets -/
def total_seedlings : ℕ := 420

/-- The number of seeds required in each packet -/
def seeds_per_packet : ℕ := 7

/-- The number of packets needed to place all seedlings -/
def packets_needed : ℕ := total_seedlings / seeds_per_packet

theorem seedling_packaging : packets_needed = 60 := by
  sorry

end seedling_packaging_l277_27789


namespace same_color_probability_l277_27791

theorem same_color_probability (total : ℕ) (black white : ℕ) 
  (h1 : (black * (black - 1)) / (total * (total - 1)) = 1 / 7)
  (h2 : (white * (white - 1)) / (total * (total - 1)) = 12 / 35) :
  ((black * (black - 1)) + (white * (white - 1))) / (total * (total - 1)) = 17 / 35 := by
sorry

end same_color_probability_l277_27791


namespace events_independent_l277_27785

-- Define the sample space
def Ω : Type := (Bool × Bool)

-- Define the probability measure
noncomputable def P : Set Ω → ℝ := sorry

-- Define events A, B, and C
def A : Set Ω := {ω | ω.1 = true}
def B : Set Ω := {ω | ω.2 = true}
def C : Set Ω := {ω | ω.1 = ω.2}

-- State the theorem
theorem events_independent :
  (P (A ∩ B) = P A * P B) ∧
  (P (B ∩ C) = P B * P C) ∧
  (P (A ∩ C) = P A * P C) := by sorry

end events_independent_l277_27785


namespace lilies_bought_l277_27750

/-- Given the cost of roses and lilies, the total paid, and the change received,
    prove that the number of lilies bought is 6. -/
theorem lilies_bought (rose_cost : ℕ) (lily_cost : ℕ) (total_paid : ℕ) (change : ℕ) : 
  rose_cost = 3000 →
  lily_cost = 2800 →
  total_paid = 25000 →
  change = 2200 →
  (total_paid - change - 2 * rose_cost) / lily_cost = 6 := by
  sorry

end lilies_bought_l277_27750


namespace no_integer_solution_l277_27701

theorem no_integer_solution :
  ¬ ∃ (x y : ℤ), 8 * x + 3 * y^2 = 5 := by sorry

end no_integer_solution_l277_27701


namespace not_all_perfect_squares_l277_27783

theorem not_all_perfect_squares (a b c : ℕ+) : 
  ¬(∃ (x y z : ℕ), (a^2 + b + c : ℕ) = x^2 ∧ (b^2 + c + a : ℕ) = y^2 ∧ (c^2 + a + b : ℕ) = z^2) :=
sorry

end not_all_perfect_squares_l277_27783


namespace difference_of_squares_2x_3_l277_27724

theorem difference_of_squares_2x_3 (x : ℝ) : (2*x + 3) * (2*x - 3) = 4*x^2 - 9 := by
  sorry

end difference_of_squares_2x_3_l277_27724


namespace group_leader_selection_l277_27778

theorem group_leader_selection (n : ℕ) (k : ℕ) (h1 : n = 20) (h2 : k = 3) :
  Nat.choose n k = 1140 := by
  sorry

end group_leader_selection_l277_27778


namespace yellow_opposite_blue_l277_27731

-- Define the colors
inductive Color
  | Red
  | Blue
  | Orange
  | Yellow
  | Green
  | White

-- Define a square with colors on both sides
structure Square where
  front : Color
  back : Color

-- Define the cube
structure Cube where
  squares : Vector Square 6

-- Define the function to get the opposite face
def oppositeFace (c : Cube) (face : Color) : Color :=
  sorry

-- Theorem statement
theorem yellow_opposite_blue (c : Cube) :
  (∃ (s : Square), s ∈ c.squares.toList ∧ s.front = Color.Yellow) →
  oppositeFace c Color.Yellow = Color.Blue :=
sorry

end yellow_opposite_blue_l277_27731


namespace line_graph_best_for_daily_income_fluctuations_l277_27797

-- Define the types of statistical graphs
inductive StatGraph
| LineGraph
| BarGraph
| PieChart
| Histogram

-- Define a structure for daily income data
structure DailyIncomeData :=
  (days : Fin 7 → ℝ)

-- Define a property for showing fluctuations intuitively
def shows_fluctuations_intuitively (graph : StatGraph) : Prop :=
  match graph with
  | StatGraph.LineGraph => true
  | _ => false

-- Define the theorem
theorem line_graph_best_for_daily_income_fluctuations 
  (data : DailyIncomeData) :
  ∃ (best : StatGraph), 
    (shows_fluctuations_intuitively best ∧ 
     ∀ (g : StatGraph), shows_fluctuations_intuitively g → g = best) :=
sorry

end line_graph_best_for_daily_income_fluctuations_l277_27797


namespace quadratic_equation_coefficients_l277_27762

theorem quadratic_equation_coefficients :
  ∀ (x : ℝ), 3 * x^2 + 1 = 5 * x ↔ 3 * x^2 + (-5) * x + 1 = 0 :=
by sorry

end quadratic_equation_coefficients_l277_27762


namespace tan_theta_plus_pi_third_l277_27793

theorem tan_theta_plus_pi_third (θ : Real) (h1 : 0 ≤ θ) (h2 : θ < 2 * Real.pi) :
  (Real.sin (3 * Real.pi / 4) : Real) / Real.cos (3 * Real.pi / 4) = Real.tan θ →
  Real.tan (θ + Real.pi / 3) = 2 - Real.sqrt 3 := by
  sorry

end tan_theta_plus_pi_third_l277_27793


namespace max_xy_value_l277_27781

theorem max_xy_value (x y c : ℝ) (h : x + y = c - 195) : 
  ∃ d : ℝ, d = 4 ∧ ∀ x' y' : ℝ, x' + y' = c - 195 → x' * y' ≤ d :=
sorry

end max_xy_value_l277_27781


namespace bryan_books_count_l277_27755

/-- The number of bookshelves Bryan has -/
def num_shelves : ℕ := 2

/-- The number of books in each of Bryan's bookshelves -/
def books_per_shelf : ℕ := 17

/-- The total number of books Bryan has -/
def total_books : ℕ := num_shelves * books_per_shelf

theorem bryan_books_count : total_books = 34 := by
  sorry

end bryan_books_count_l277_27755


namespace not_divisible_by_two_l277_27719

theorem not_divisible_by_two (n : ℕ) (h_pos : n > 0) 
  (h_sum : ∃ k : ℤ, (1 : ℚ) / 2 + 1 / 3 + 1 / 5 + 1 / n = k) : 
  ¬(2 ∣ n) := by
sorry

end not_divisible_by_two_l277_27719


namespace medicine_supply_duration_l277_27798

/-- Represents the number of pills in a bottle -/
def pills_per_bottle : ℕ := 90

/-- Represents the fraction of a pill taken per dose -/
def fraction_per_dose : ℚ := 1/3

/-- Represents the number of days between doses -/
def days_between_doses : ℕ := 3

/-- Represents the average number of days in a month -/
def days_per_month : ℕ := 30

/-- Proves that the supply of medicine lasts 27 months -/
theorem medicine_supply_duration :
  (pills_per_bottle : ℚ) * days_between_doses / fraction_per_dose / days_per_month = 27 := by
  sorry


end medicine_supply_duration_l277_27798


namespace min_degree_g_l277_27736

variables (x : ℝ) (f g h : ℝ → ℝ)

def is_polynomial (p : ℝ → ℝ) : Prop := sorry

def degree (p : ℝ → ℝ) : ℕ := sorry

theorem min_degree_g 
  (eq : ∀ x, 5 * f x + 7 * g x = h x)
  (f_poly : is_polynomial f)
  (g_poly : is_polynomial g)
  (h_poly : is_polynomial h)
  (f_deg : degree f = 10)
  (h_deg : degree h = 13) :
  degree g ≥ 13 ∧ ∃ g', is_polynomial g' ∧ degree g' = 13 ∧ ∀ x, 5 * f x + 7 * g' x = h x :=
sorry

end min_degree_g_l277_27736


namespace compound_molecular_weight_l277_27742

/-- Atomic weight of Hydrogen in g/mol -/
def atomic_weight_H : ℝ := 1.01

/-- Atomic weight of Carbon in g/mol -/
def atomic_weight_C : ℝ := 12.01

/-- Atomic weight of Nitrogen in g/mol -/
def atomic_weight_N : ℝ := 14.01

/-- Atomic weight of Chlorine in g/mol -/
def atomic_weight_Cl : ℝ := 35.45

/-- Atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- Number of Hydrogen atoms in the compound -/
def num_H : ℕ := 2

/-- Number of Carbon atoms in the compound -/
def num_C : ℕ := 3

/-- Number of Nitrogen atoms in the compound -/
def num_N : ℕ := 1

/-- Number of Chlorine atoms in the compound -/
def num_Cl : ℕ := 1

/-- Number of Oxygen atoms in the compound -/
def num_O : ℕ := 3

/-- Calculates the molecular weight of the compound -/
def molecular_weight : ℝ :=
  (num_H : ℝ) * atomic_weight_H +
  (num_C : ℝ) * atomic_weight_C +
  (num_N : ℝ) * atomic_weight_N +
  (num_Cl : ℝ) * atomic_weight_Cl +
  (num_O : ℝ) * atomic_weight_O

/-- Theorem stating that the molecular weight of the compound is 135.51 g/mol -/
theorem compound_molecular_weight :
  molecular_weight = 135.51 := by sorry

end compound_molecular_weight_l277_27742


namespace storks_on_fence_l277_27712

/-- The number of storks that joined the birds on the fence -/
def num_storks_joined : ℕ := 4

theorem storks_on_fence :
  let initial_birds : ℕ := 2
  let additional_birds : ℕ := 5
  let total_birds : ℕ := initial_birds + additional_birds
  let bird_stork_difference : ℕ := 3
  num_storks_joined = total_birds - bird_stork_difference := by
  sorry

end storks_on_fence_l277_27712


namespace first_terrific_tuesday_l277_27708

/-- Represents a day of the week -/
inductive DayOfWeek
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday
| Sunday

/-- Represents a date in October -/
structure OctoberDate where
  day : Nat
  dayOfWeek : DayOfWeek

/-- Definition of a Terrific Tuesday -/
def isTerrificTuesday (date : OctoberDate) : Prop :=
  date.dayOfWeek = DayOfWeek.Tuesday ∧ 
  (∃ n : Nat, n = 5 ∧ date.day = n * 7 - 4)

/-- The company's start date -/
def startDate : OctoberDate :=
  { day := 2, dayOfWeek := DayOfWeek.Monday }

/-- The number of days in October -/
def octoberDays : Nat := 31

/-- Theorem: The first Terrific Tuesday after operations begin is October 31 -/
theorem first_terrific_tuesday : 
  ∃ (date : OctoberDate), 
    date.day = 31 ∧ 
    isTerrificTuesday date ∧ 
    ∀ (earlier : OctoberDate), 
      earlier.day > startDate.day ∧ 
      earlier.day < date.day → 
      ¬isTerrificTuesday earlier :=
by sorry

end first_terrific_tuesday_l277_27708


namespace triangle_frame_angles_l277_27707

/-- A frame consisting of congruent triangles surrounding a square --/
structure TriangleFrame where
  /-- The number of triangles in the frame --/
  num_triangles : ℕ
  /-- The angles of each triangle in the frame --/
  triangle_angles : Fin 3 → ℝ
  /-- The sum of angles in each triangle is 180° --/
  angle_sum : (triangle_angles 0) + (triangle_angles 1) + (triangle_angles 2) = 180
  /-- The triangles form a complete circle at each corner of the square --/
  corner_sum : 4 * (triangle_angles 0) + 90 = 360
  /-- The triangles along each side of the square form a straight line --/
  side_sum : (triangle_angles 1) + (triangle_angles 2) + 90 = 180

/-- The theorem stating the angles of the triangles in the frame --/
theorem triangle_frame_angles (frame : TriangleFrame) 
  (h : frame.num_triangles = 21) : 
  frame.triangle_angles 0 = 67.5 ∧ 
  frame.triangle_angles 1 = 22.5 ∧ 
  frame.triangle_angles 2 = 90 := by
  sorry

end triangle_frame_angles_l277_27707


namespace perfect_square_condition_l277_27728

theorem perfect_square_condition (x y k : ℝ) :
  (∃ (z : ℝ), x^2 + k*x*y + 49*y^2 = z^2) → (k = 14 ∨ k = -14) := by
  sorry

end perfect_square_condition_l277_27728


namespace student_speaking_probability_l277_27743

/-- The probability of a student speaking truth -/
def prob_truth : ℝ := 0.30

/-- The probability of a student speaking lie -/
def prob_lie : ℝ := 0.20

/-- The probability of a student speaking both truth and lie -/
def prob_both : ℝ := 0.10

/-- The probability of a student speaking either truth or lie -/
def prob_truth_or_lie : ℝ := prob_truth + prob_lie - prob_both

theorem student_speaking_probability :
  prob_truth_or_lie = 0.40 := by sorry

end student_speaking_probability_l277_27743


namespace t_shape_perimeter_l277_27775

/-- A figure consisting of six identical squares arranged in a "T" shape -/
structure TShapeFigure where
  /-- The side length of each square in the figure -/
  square_side : ℝ
  /-- The total area of the figure is 150 cm² -/
  total_area_eq : 6 * square_side^2 = 150

/-- The perimeter of the T-shaped figure -/
def perimeter (fig : TShapeFigure) : ℝ :=
  9 * fig.square_side

/-- Theorem stating that the perimeter of the T-shaped figure is 45 cm -/
theorem t_shape_perimeter (fig : TShapeFigure) : perimeter fig = 45 := by
  sorry

end t_shape_perimeter_l277_27775


namespace impossible_arrangement_l277_27753

/-- A grid of integers -/
def Grid := Matrix (Fin 25) (Fin 41) ℤ

/-- Predicate to check if a grid satisfies the adjacency condition -/
def SatisfiesAdjacencyCondition (g : Grid) : Prop :=
  ∀ i j i' j', (i = i' ∧ |j - j'| = 1) ∨ (j = j' ∧ |i - i'| = 1) →
    |g i j - g i' j'| ≤ 16

/-- Predicate to check if a grid contains distinct integers -/
def ContainsDistinctIntegers (g : Grid) : Prop :=
  ∀ i j i' j', g i j = g i' j' → i = i' ∧ j = j'

/-- Theorem stating the impossibility of the arrangement -/
theorem impossible_arrangement : 
  ¬∃ (g : Grid), SatisfiesAdjacencyCondition g ∧ ContainsDistinctIntegers g :=
sorry

end impossible_arrangement_l277_27753


namespace centroid_satisfies_conditions_l277_27758

/-- A point in a 2D plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A triangle defined by three points -/
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

/-- Check if a point is inside a triangle -/
def isInside (p : Point) (t : Triangle) : Prop := sorry

/-- Check if a point is on a line segment between two other points -/
def isOnSegment (p : Point) (a : Point) (b : Point) : Prop := sorry

/-- Check if two line segments are parallel -/
def areParallel (p1 : Point) (p2 : Point) (q1 : Point) (q2 : Point) : Prop := sorry

/-- Calculate the area of a triangle given three points -/
def triangleArea (p1 : Point) (p2 : Point) (p3 : Point) : ℝ := sorry

/-- The centroid of a triangle -/
def centroid (t : Triangle) : Point := sorry

theorem centroid_satisfies_conditions (t : Triangle) :
  ∃ (O L M N : Point),
    isInside O t ∧
    isOnSegment L t.A t.B ∧
    isOnSegment M t.B t.C ∧
    isOnSegment N t.C t.A ∧
    areParallel O L t.B t.C ∧
    areParallel O M t.A t.C ∧
    areParallel O N t.A t.B ∧
    triangleArea O t.B L = triangleArea O t.C M ∧
    triangleArea O t.C M = triangleArea O t.A N ∧
    O = centroid t :=
  sorry

end centroid_satisfies_conditions_l277_27758


namespace largest_multiple_of_8_less_than_120_l277_27725

theorem largest_multiple_of_8_less_than_120 :
  ∃ n : ℕ, n * 8 = 112 ∧ 
  112 < 120 ∧
  ∀ m : ℕ, m * 8 < 120 → m * 8 ≤ 112 :=
sorry

end largest_multiple_of_8_less_than_120_l277_27725


namespace train_turn_radians_l277_27790

/-- Given a circular railway arc with radius 2 km and a train moving at 30 km/h,
    the number of radians the train turns through in 10 seconds is 1/24. -/
theorem train_turn_radians (r : ℝ) (v : ℝ) (t : ℝ) :
  r = 2 →  -- radius in km
  v = 30 → -- speed in km/h
  t = 10 / 3600 → -- time in hours (10 seconds converted to hours)
  (v * t) / r = 1 / 24 := by
  sorry

end train_turn_radians_l277_27790


namespace cube_volume_from_surface_area_l277_27723

/-- Given a cube with surface area 24 square centimeters, prove its volume is 8 cubic centimeters. -/
theorem cube_volume_from_surface_area : 
  ∀ (s : ℝ), 
  (6 * s^2 = 24) →  -- surface area formula
  (s^3 = 8)         -- volume formula
:= by sorry

end cube_volume_from_surface_area_l277_27723


namespace relay_race_average_time_l277_27726

/-- Calculates the average time for a leg of a relay race given the times of two runners. -/
def average_leg_time (y_time z_time : ℕ) : ℚ :=
  (y_time + z_time : ℚ) / 2

/-- Theorem stating that for the given runner times, the average leg time is 42 seconds. -/
theorem relay_race_average_time :
  average_leg_time 58 26 = 42 := by
  sorry

end relay_race_average_time_l277_27726


namespace seed_mixture_ryegrass_percentage_l277_27734

/-- Given two seed mixtures X and Y, prove that Y contains 25% ryegrass -/
theorem seed_mixture_ryegrass_percentage :
  -- Definitions based on the problem conditions
  let x_ryegrass : ℝ := 0.40  -- 40% ryegrass in X
  let x_bluegrass : ℝ := 0.60  -- 60% bluegrass in X
  let y_fescue : ℝ := 0.75  -- 75% fescue in Y
  let final_ryegrass : ℝ := 0.32  -- 32% ryegrass in final mixture
  let x_proportion : ℝ := 0.4667  -- 46.67% of final mixture is X
  
  -- The percentage of ryegrass in Y
  ∃ y_ryegrass : ℝ,
    -- Conditions
    x_ryegrass + x_bluegrass = 1 ∧  -- X components sum to 100%
    y_ryegrass + y_fescue = 1 ∧  -- Y components sum to 100%
    x_proportion * x_ryegrass + (1 - x_proportion) * y_ryegrass = final_ryegrass →
    -- Conclusion
    y_ryegrass = 0.25 := by
  sorry

end seed_mixture_ryegrass_percentage_l277_27734


namespace S_pq_equation_l277_27709

/-- S(n) is the sum of squares of positive integers less than and coprime to n -/
def S (n : ℕ) : ℕ := sorry

/-- p is a prime number equal to 2^7 - 1 -/
def p : ℕ := 127

/-- q is a prime number equal to 2^5 - 1 -/
def q : ℕ := 31

/-- a is a positive integer -/
def a : ℕ := 7561

theorem S_pq_equation : 
  ∃ (b c : ℕ), 
    b < c ∧ 
    Nat.Coprime b c ∧
    S (p * q) = (p^2 * q^2 / 6) * (a - b / c) := by sorry

end S_pq_equation_l277_27709


namespace min_max_abs_x_squared_minus_2xy_equals_zero_l277_27718

/-- The minimum value of max_{0 ≤ x ≤ 2} |x^2 - 2xy| for y in ℝ is 0 -/
theorem min_max_abs_x_squared_minus_2xy_equals_zero :
  ∃ y : ℝ, ∀ y' : ℝ, (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → |x^2 - 2*x*y| ≤ |x^2 - 2*x*y'|) ∧
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ 2 ∧ |x^2 - 2*x*y| = 0) := by
  sorry

end min_max_abs_x_squared_minus_2xy_equals_zero_l277_27718


namespace kenneth_distance_past_finish_line_l277_27760

/-- Proves that Kenneth will be 10 yards past the finish line when Biff crosses the finish line in a 500-yard race -/
theorem kenneth_distance_past_finish_line 
  (race_distance : ℝ) 
  (biff_speed : ℝ) 
  (kenneth_speed : ℝ) 
  (h1 : race_distance = 500) 
  (h2 : biff_speed = 50) 
  (h3 : kenneth_speed = 51) : 
  kenneth_speed * (race_distance / biff_speed) - race_distance = 10 := by
  sorry

end kenneth_distance_past_finish_line_l277_27760


namespace similar_triangle_shortest_side_l277_27769

theorem similar_triangle_shortest_side
  (a b c : ℝ)
  (h1 : a > 0 ∧ b > 0 ∧ c > 0)
  (h2 : a^2 + b^2 = c^2)  -- Pythagorean theorem for the first triangle
  (h3 : a = 15)           -- Length of one leg of the first triangle
  (h4 : c = 39)           -- Length of hypotenuse of the first triangle
  (k : ℝ)
  (h5 : k > 0)
  (h6 : k * c = 78)       -- Length of hypotenuse of the second triangle
  : k * a = 30            -- Length of the shortest side of the second triangle
:= by sorry

end similar_triangle_shortest_side_l277_27769


namespace work_schedule_lcm_l277_27784

theorem work_schedule_lcm : Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 8 9)) = 360 := by
  sorry

end work_schedule_lcm_l277_27784


namespace expression_evaluation_l277_27757

theorem expression_evaluation (a b c d : ℝ) :
  (a - (b - (c + d))) - ((a + b) - (c - d)) = -2 * b + 2 * c := by
  sorry

end expression_evaluation_l277_27757


namespace ian_money_left_l277_27770

/-- Calculates the amount of money Ian has left after expenses and taxes --/
def money_left (hours_worked : ℕ) (hourly_rate : ℚ) (monthly_expense : ℚ) (tax_rate : ℚ) : ℚ :=
  let total_earnings := hours_worked * hourly_rate
  let tax := tax_rate * total_earnings
  let net_earnings := total_earnings - tax
  let amount_spent := (1/2) * net_earnings
  let remaining_after_spending := net_earnings - amount_spent
  remaining_after_spending - monthly_expense

/-- Theorem stating that Ian has $14.80 left after expenses and taxes --/
theorem ian_money_left :
  money_left 8 18 50 (1/10) = 148/10 :=
by sorry

end ian_money_left_l277_27770


namespace expression_bounds_l277_27777

theorem expression_bounds (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c)
  (ha2 : a ≤ 2) (hb2 : b ≤ 2) (hc2 : c ≤ 2) :
  let k : ℝ := 2
  let expr := Real.sqrt (k * a^2 + (2 - b)^2) + Real.sqrt (k * b^2 + (2 - c)^2) + Real.sqrt (k * c^2 + (2 - a)^2)
  6 * Real.sqrt 2 ≤ expr ∧ expr ≤ 12 * Real.sqrt 2 := by
  sorry

end expression_bounds_l277_27777


namespace family_age_relations_l277_27740

/-- Given family ages and relationships, prove age difference and Teresa's age at Michiko's birth -/
theorem family_age_relations (teresa_age morio_age : ℕ) 
  (h1 : teresa_age = 59)
  (h2 : morio_age = 71)
  (h3 : morio_age - 38 = michiko_age)
  (h4 : michiko_age - 4 = kenji_age)
  (h5 : teresa_age - 10 = emiko_age)
  (h6 : kenji_age = hideki_age)
  (h7 : morio_age = ryuji_age) :
  michiko_age - hideki_age = 4 ∧ teresa_age - michiko_age = 26 :=
by sorry


end family_age_relations_l277_27740


namespace output_value_after_five_years_l277_27717

/-- Calculates the final value after compound growth -/
def final_value (initial_value : ℝ) (growth_rate : ℝ) (years : ℕ) : ℝ :=
  initial_value * (1 + growth_rate) ^ years

/-- Theorem: The output value after 5 years with 8% annual growth -/
theorem output_value_after_five_years 
  (a : ℝ) -- initial value
  (h1 : a > 0) -- initial value is positive
  (h2 : a = 1000000) -- initial value is 1 million yuan
  : final_value a 0.08 5 = a * (1 + 0.08) ^ 5 := by
  sorry

#eval final_value 1000000 0.08 5

end output_value_after_five_years_l277_27717


namespace alphanumeric_puzzle_l277_27768

theorem alphanumeric_puzzle :
  ∃! (A B C D E F H J K L : Nat),
    (A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧ E < 10 ∧
     F < 10 ∧ H < 10 ∧ J < 10 ∧ K < 10 ∧ L < 10) ∧
    (A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧ A ≠ H ∧ A ≠ J ∧ A ≠ K ∧ A ≠ L ∧
     B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧ B ≠ H ∧ B ≠ J ∧ B ≠ K ∧ B ≠ L ∧
     C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ C ≠ H ∧ C ≠ J ∧ C ≠ K ∧ C ≠ L ∧
     D ≠ E ∧ D ≠ F ∧ D ≠ H ∧ D ≠ J ∧ D ≠ K ∧ D ≠ L ∧
     E ≠ F ∧ E ≠ H ∧ E ≠ J ∧ E ≠ K ∧ E ≠ L ∧
     F ≠ H ∧ F ≠ J ∧ F ≠ K ∧ F ≠ L ∧
     H ≠ J ∧ H ≠ K ∧ H ≠ L ∧
     J ≠ K ∧ J ≠ L ∧
     K ≠ L) ∧
    (A * B = B) ∧
    (B * C = 10 * A + C) ∧
    (C * D = 10 * B + C) ∧
    (D * E = 10 * C + H) ∧
    (E * F = 10 * D + K) ∧
    (F * H = 10 * C + J) ∧
    (H * J = 10 * K + J) ∧
    (J * K = E) ∧
    (K * L = L) ∧
    (A * L = L) ∧
    (A = 1 ∧ B = 3 ∧ C = 5 ∧ D = 7 ∧ E = 8 ∧ F = 9 ∧ H = 6 ∧ J = 4 ∧ K = 2 ∧ L = 0) :=
by sorry

end alphanumeric_puzzle_l277_27768


namespace power_of_negative_cube_l277_27739

theorem power_of_negative_cube (a : ℝ) : (-(a^3))^2 = a^6 := by
  sorry

end power_of_negative_cube_l277_27739


namespace power_of_negative_product_l277_27721

theorem power_of_negative_product (a b : ℝ) : (-a^3 * b^5)^2 = a^6 * b^10 := by
  sorry

end power_of_negative_product_l277_27721


namespace prime_pair_existence_l277_27764

theorem prime_pair_existence (n : ℕ) : 
  (∃ p q : ℕ, Prime p ∧ Prime q ∧ p + 2 = q ∧ Prime (2^n + p) ∧ Prime (2^n + q)) ↔ n = 1 ∨ n = 3 :=
by sorry

end prime_pair_existence_l277_27764


namespace small_circle_radius_l277_27780

/-- Given a large circle with radius 10 meters and seven congruent smaller circles
    arranged so that four of their diameters align with the diameter of the large circle,
    the radius of each smaller circle is 2.5 meters. -/
theorem small_circle_radius (large_radius : ℝ) (small_radius : ℝ) : 
  large_radius = 10 → 
  4 * (2 * small_radius) = 2 * large_radius → 
  small_radius = 2.5 := by
  sorry

end small_circle_radius_l277_27780


namespace rowing_problem_l277_27759

/-- Proves that given the conditions of the rowing problem, the downstream distance is 60 km -/
theorem rowing_problem (upstream_distance : ℝ) (upstream_time : ℝ) (downstream_time : ℝ) (stream_speed : ℝ)
  (h1 : upstream_distance = 30)
  (h2 : upstream_time = 3)
  (h3 : downstream_time = 3)
  (h4 : stream_speed = 5) :
  let boat_speed := upstream_distance / upstream_time + stream_speed
  let downstream_speed := boat_speed + stream_speed
  downstream_speed * downstream_time = 60 := by
  sorry

end rowing_problem_l277_27759


namespace largest_perfect_square_factor_of_1764_l277_27771

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m ^ 2

theorem largest_perfect_square_factor_of_1764 :
  ∀ k : ℕ, is_perfect_square k ∧ k ∣ 1764 → k ≤ 1764 :=
by sorry

end largest_perfect_square_factor_of_1764_l277_27771


namespace square_difference_greater_than_polynomial_l277_27763

theorem square_difference_greater_than_polynomial :
  ∀ x : ℝ, (x - 3)^2 > x^2 - 6*x + 8 := by
sorry

end square_difference_greater_than_polynomial_l277_27763


namespace num_machines_proof_l277_27767

/-- The number of machines that complete a job in 6 hours, 
    given that 2 machines complete the same job in 24 hours -/
def num_machines : ℕ :=
  let time_many : ℕ := 6  -- time taken by multiple machines
  let time_two : ℕ := 24   -- time taken by 2 machines
  let machines_two : ℕ := 2  -- number of machines in second scenario
  8  -- to be proved

theorem num_machines_proof : 
  num_machines * time_many = machines_two * time_two :=
by sorry

end num_machines_proof_l277_27767


namespace complex_product_real_implies_sum_modulus_l277_27766

theorem complex_product_real_implies_sum_modulus (a : ℝ) : 
  let z₁ : ℂ := 2 + I
  let z₂ : ℂ := a + 3*I
  (z₁ * z₂).im = 0 → Complex.abs (z₁ + z₂) = 4 * Real.sqrt 2 := by
  sorry

end complex_product_real_implies_sum_modulus_l277_27766
