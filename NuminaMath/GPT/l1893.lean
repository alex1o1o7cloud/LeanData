import Mathlib

namespace simplify_and_evaluate_l1893_189375

theorem simplify_and_evaluate (a : ℤ) (h : a = -4) :
  (4 * a ^ 2 - 3 * a) - (2 * a ^ 2 + a - 1) + (2 - a ^ 2 + 4 * a) = 19 :=
by
  rw [h]
  sorry

end simplify_and_evaluate_l1893_189375


namespace canteen_needs_bananas_l1893_189301

-- Define the given conditions
def total_bananas := 9828
def weeks := 9
def days_in_week := 7
def bananas_in_dozen := 12

-- Calculate the required value and prove the equivalence
theorem canteen_needs_bananas : 
  (total_bananas / (weeks * days_in_week)) / bananas_in_dozen = 13 :=
by
  -- This is where the proof would go
  sorry

end canteen_needs_bananas_l1893_189301


namespace all_boxcars_combined_capacity_l1893_189355

theorem all_boxcars_combined_capacity :
  let black_capacity := 4000
  let blue_capacity := 2 * black_capacity
  let red_capacity := 3 * blue_capacity
  let green_capacity := 1.5 * black_capacity
  let yellow_capacity := green_capacity + 2000
  let total_red := 3 * red_capacity
  let total_blue := 4 * blue_capacity
  let total_black := 7 * black_capacity
  let total_green := 2 * green_capacity
  let total_yellow := 5 * yellow_capacity
  total_red + total_blue + total_black + total_green + total_yellow = 184000 :=
by 
  -- Proof omitted
  sorry

end all_boxcars_combined_capacity_l1893_189355


namespace elegant_interval_solution_l1893_189368

noncomputable def elegant_interval : ℝ → ℝ × ℝ := sorry

theorem elegant_interval_solution (m : ℝ) (a b : ℕ) (s : ℝ) (p : ℕ) :
  a < m ∧ m < b ∧ a + 1 = b ∧ 3 < s + b ∧ s + b ≤ 13 ∧ s = Real.sqrt a ∧ b * b + a * s = p → p = 33 ∨ p = 127 := 
by sorry

end elegant_interval_solution_l1893_189368


namespace lottery_ticket_random_event_l1893_189385

-- Define the type of possible outcomes of buying a lottery ticket
inductive LotteryOutcome
| Win
| Lose

-- Define the random event condition
def is_random_event (outcome: LotteryOutcome) : Prop :=
  match outcome with
  | LotteryOutcome.Win => True
  | LotteryOutcome.Lose => True

-- The theorem to prove that buying 1 lottery ticket and winning is a random event
theorem lottery_ticket_random_event : is_random_event LotteryOutcome.Win :=
by
  sorry

end lottery_ticket_random_event_l1893_189385


namespace express_y_in_terms_of_x_l1893_189387

theorem express_y_in_terms_of_x (x y : ℝ) (h : 3 * x - y = 9) : y = 3 * x - 9 := 
by
  sorry

end express_y_in_terms_of_x_l1893_189387


namespace fraction_zero_implies_x_eq_two_l1893_189327

theorem fraction_zero_implies_x_eq_two (x : ℝ) (h : (x^2 - 4) / (x + 2) = 0) : x = 2 :=
sorry

end fraction_zero_implies_x_eq_two_l1893_189327


namespace share_per_person_in_dollars_l1893_189340

-- Definitions based on conditions
def total_cost_euros : ℝ := 25 * 10^9  -- 25 billion Euros
def number_of_people : ℝ := 300 * 10^6  -- 300 million people
def exchange_rate : ℝ := 1.2  -- 1 Euro = 1.2 dollars

-- To prove
theorem share_per_person_in_dollars : (total_cost_euros * exchange_rate) / number_of_people = 100 := 
by 
  sorry

end share_per_person_in_dollars_l1893_189340


namespace camden_total_legs_l1893_189334

theorem camden_total_legs 
  (num_justin_dogs : ℕ := 14)
  (num_rico_dogs := num_justin_dogs + 10)
  (num_camden_dogs := 3 * num_rico_dogs / 4)
  (camden_3_leg_dogs : ℕ := 5)
  (camden_4_leg_dogs : ℕ := 7)
  (camden_2_leg_dogs : ℕ := 2) : 
  3 * camden_3_leg_dogs + 4 * camden_4_leg_dogs + 2 * camden_2_leg_dogs = 47 :=
by sorry

end camden_total_legs_l1893_189334


namespace walking_time_l1893_189318

-- Define the conditions as Lean definitions
def minutes_in_hour : Nat := 60

def work_hours : Nat := 6
def work_minutes := work_hours * minutes_in_hour
def sitting_interval : Nat := 90
def walking_time_per_interval : Nat := 10

-- State the main theorem
theorem walking_time (h1 : 10 * 90 = 600) (h2 : 10 * (work_hours * 60) / 90 = 40) : 
  work_minutes / sitting_interval * walking_time_per_interval = 40 :=
  sorry

end walking_time_l1893_189318


namespace sum_of_cubes_inequality_l1893_189346

theorem sum_of_cubes_inequality (a b c : ℝ) (h1 : a >= -1) (h2 : b >= -1) (h3 : c >= -1) (h4 : a^3 + b^3 + c^3 = 1) :
  a + b + c + a^2 + b^2 + c^2 <= 4 := 
sorry

end sum_of_cubes_inequality_l1893_189346


namespace negation_of_exists_geq_prop_l1893_189337

open Classical

variable (P : Prop) (Q : Prop)

-- Original proposition:
def exists_geq_prop : Prop := 
  ∃ x : ℝ, x^2 + x + 1 ≥ 0

-- Its negation:
def forall_lt_neg : Prop :=
  ∀ x : ℝ, x^2 + x + 1 < 0

-- The theorem to prove:
theorem negation_of_exists_geq_prop : ¬ exists_geq_prop ↔ forall_lt_neg := 
by 
  -- The proof steps will be filled in here
  sorry

end negation_of_exists_geq_prop_l1893_189337


namespace maximize_exponential_sum_l1893_189371

theorem maximize_exponential_sum (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 ≤ 4) : 
  e^a + e^b + e^c + e^d ≤ 4 * Real.exp 1 := 
sorry

end maximize_exponential_sum_l1893_189371


namespace skirt_more_than_pants_l1893_189389

def amount_cut_off_skirt : ℝ := 0.75
def amount_cut_off_pants : ℝ := 0.5

theorem skirt_more_than_pants : 
  amount_cut_off_skirt - amount_cut_off_pants = 0.25 := 
by
  sorry

end skirt_more_than_pants_l1893_189389


namespace remainder_when_divided_by_22_l1893_189339

theorem remainder_when_divided_by_22 (y : ℤ) (k : ℤ) (h : y = 264 * k + 42) : y % 22 = 20 :=
by
  sorry

end remainder_when_divided_by_22_l1893_189339


namespace subtracting_is_adding_opposite_l1893_189379

theorem subtracting_is_adding_opposite (a b : ℚ) : a - b = a + (-b) :=
by sorry

end subtracting_is_adding_opposite_l1893_189379


namespace no_solution_eq_l1893_189336

theorem no_solution_eq (m : ℝ) : 
  (∀ x : ℝ, x ≠ 3 → ((3 - 2 * x) / (x - 3) + (2 + m * x) / (3 - x) = -1) → (m = -1)) :=
by
  sorry

end no_solution_eq_l1893_189336


namespace even_function_periodicity_l1893_189333

noncomputable def f : ℝ → ℝ :=
sorry -- The actual function definition is not provided here but assumed to exist.

theorem even_function_periodicity (x : ℝ) (h1 : 1 ≤ x ∧ x ≤ 2)
  (h2 : f (x + 2) = f x)
  (hf_even : ∀ x, f x = f (-x))
  (hf_segment : ∀ x, 1 ≤ x ∧ x ≤ 2 → f x = x^2 + 2*x - 1) :
  ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = x^2 - 6*x + 7 :=
sorry

end even_function_periodicity_l1893_189333


namespace mirror_area_l1893_189344

-- Defining the conditions as Lean functions and values
def frame_height : ℕ := 100
def frame_width : ℕ := 140
def frame_border : ℕ := 15

-- Statement to prove the area of the mirror
theorem mirror_area :
  let mirror_width := frame_width - 2 * frame_border
  let mirror_height := frame_height - 2 * frame_border
  mirror_width * mirror_height = 7700 :=
by
  sorry

end mirror_area_l1893_189344


namespace number_of_true_statements_l1893_189361

def reciprocal (n : ℕ) : ℚ := 1 / n

theorem number_of_true_statements (n : ℕ) :
  let s1 := reciprocal 4 + reciprocal 8 ≠ reciprocal 12
  let s2 := reciprocal 9 - reciprocal 3 ≠ reciprocal 6
  let s3 := reciprocal 5 * reciprocal 10 = reciprocal 50
  let s4 := reciprocal 16 / reciprocal 4 = reciprocal 4
  (cond s1 1 0) + (cond s2 1 0) + (cond s3 1 0) + (cond s4 1 0) = 2 := by
  sorry

end number_of_true_statements_l1893_189361


namespace average_first_15_nat_l1893_189367

-- Define the sequence and necessary conditions
def sum_first_n_nat (n : ℕ) : ℕ := n * (n + 1) / 2

theorem average_first_15_nat : (sum_first_n_nat 15) / 15 = 8 := 
by 
  -- Here we shall place the proof to show the above statement holds true
  sorry

end average_first_15_nat_l1893_189367


namespace find_three_digit_number_l1893_189395

theorem find_three_digit_number :
  ∃ x : ℕ, 100 ≤ x ∧ x ≤ 999 ∧
    (x - 6) % 7 = 0 ∧
    (x - 7) % 8 = 0 ∧
    (x - 8) % 9 = 0 ∧
    x = 503 :=
by
  sorry

end find_three_digit_number_l1893_189395


namespace root_expression_l1893_189364

theorem root_expression {p q x1 x2 : ℝ}
  (h1 : x1^2 + p * x1 + q = 0)
  (h2 : x2^2 + p * x2 + q = 0) :
  (x1 / x2 + x2 / x1) = (p^2 - 2 * q) / q :=
by {
  sorry
}

end root_expression_l1893_189364


namespace range_of_a_l1893_189376

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 + 2 * x - 3 > 0 → x > a) ↔ a ≥ 1 :=
by
  sorry

end range_of_a_l1893_189376


namespace hyperbola_vertex_distance_l1893_189372

theorem hyperbola_vertex_distance : 
  ∀ x y: ℝ, (x^2 / 144 - y^2 / 49 = 1) → (∃ a: ℝ, a = 12 ∧ 2 * a = 24) :=
by 
  sorry

end hyperbola_vertex_distance_l1893_189372


namespace min_value_expression_l1893_189391

theorem min_value_expression (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 9) :
  (x^2 + y^2) / (x + y) + (x^2 + z^2) / (x + z) + (y^2 + z^2) / (y + z) ≥ 9 :=
sorry

end min_value_expression_l1893_189391


namespace quadratic_example_correct_l1893_189373

-- Define the quadratic function
def quad_func (x : ℝ) : ℝ := -2 * x^2 + 12 * x - 10

-- Conditions defined
def condition1 := quad_func 1 = 0
def condition2 := quad_func 5 = 0
def condition3 := quad_func 3 = 8

-- Theorem statement combining the conditions
theorem quadratic_example_correct :
  condition1 ∧ condition2 ∧ condition3 :=
by
  -- Proof omitted as per instructions
  sorry

end quadratic_example_correct_l1893_189373


namespace expression_evaluation_l1893_189311

theorem expression_evaluation :
  2^3 + 4 * 5 - Real.sqrt 9 + (3^2 * 2) / 3 = 31 := sorry

end expression_evaluation_l1893_189311


namespace solution_set_inequality_l1893_189331

theorem solution_set_inequality (x : ℝ) : 
  ((x-2) * (3-x) > 0) ↔ (2 < x ∧ x < 3) :=
by sorry

end solution_set_inequality_l1893_189331


namespace exist_indices_for_sequences_l1893_189374

open Nat

theorem exist_indices_for_sequences 
  (a b c : ℕ → ℕ) : 
  ∃ p q, p ≠ q ∧ p > q ∧ a p ≥ a q ∧ b p ≥ b q ∧ c p ≥ c q := by
  sorry

end exist_indices_for_sequences_l1893_189374


namespace lcm_of_36_48_75_l1893_189313

-- Definitions of the numbers and their factorizations
def num1 := 36
def num2 := 48
def num3 := 75

def factor_36 := (2^2, 3^2)
def factor_48 := (2^4, 3^1)
def factor_75 := (3^1, 5^2)

def highest_power_2 := 2^4
def highest_power_3 := 3^2
def highest_power_5 := 5^2

def lcm_36_48_75 := highest_power_2 * highest_power_3 * highest_power_5

-- The theorem statement
theorem lcm_of_36_48_75 : lcm_36_48_75 = 3600 := by
  sorry

end lcm_of_36_48_75_l1893_189313


namespace eval_expression_l1893_189394

theorem eval_expression : 3 ^ 4 - 4 * 3 ^ 3 + 6 * 3 ^ 2 - 4 * 3 + 1 = 16 := 
by 
  sorry

end eval_expression_l1893_189394


namespace complement_of_union_is_singleton_five_l1893_189347

open Set

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define the set M
def M : Set ℕ := {1, 2}

-- Define the set N
def N : Set ℕ := {3, 4}

-- Define the union of M and N
def M_union_N : Set ℕ := M ∪ N

-- Define the complement of union of M and N in U
def complement_U_M_union_N : Set ℕ := U \ M_union_N

-- State the theorem to be proved
theorem complement_of_union_is_singleton_five :
  complement_U_M_union_N = {5} :=
sorry

end complement_of_union_is_singleton_five_l1893_189347


namespace kaleb_saved_initial_amount_l1893_189320

theorem kaleb_saved_initial_amount (allowance toys toy_price : ℕ) (total_savings : ℕ)
  (h1 : allowance = 15)
  (h2 : toys = 6)
  (h3 : toy_price = 6)
  (h4 : total_savings = toys * toy_price - allowance) :
  total_savings = 21 :=
  sorry

end kaleb_saved_initial_amount_l1893_189320


namespace smallest_positive_integer_l1893_189356

theorem smallest_positive_integer :
  ∃ x : ℤ, 0 < x ∧ (x % 5 = 4) ∧ (x % 7 = 6) ∧ (x % 11 = 10) ∧ x = 384 :=
by
  sorry

end smallest_positive_integer_l1893_189356


namespace measure_angle_ADC_l1893_189362

variable (A B C D : Type)
variable [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup D]

-- Definitions for the angles
variable (angle_ABC angle_BCD angle_ADC : ℝ)

-- Conditions for the problem
axiom Angle_ABC_is_4_times_Angle_BCD : angle_ABC = 4 * angle_BCD
axiom Angle_BCD_ADC_sum_to_180 : angle_BCD + angle_ADC = 180

-- The theorem that we want to prove
theorem measure_angle_ADC (Angle_ABC_is_4_times_Angle_BCD: angle_ABC = 4 * angle_BCD)
    (Angle_BCD_ADC_sum_to_180: angle_BCD + angle_ADC = 180) : 
    angle_ADC = 144 :=
by
  sorry

end measure_angle_ADC_l1893_189362


namespace equation1_solution_equation2_solution_l1893_189365

theorem equation1_solution (x : ℝ) (h : 5 / (x + 1) = 1 / (x - 3)) : x = 4 :=
sorry

theorem equation2_solution (x : ℝ) (h : (2 - x) / (x - 3) + 2 = 1 / (3 - x)) : x = 7 / 3 :=
sorry

end equation1_solution_equation2_solution_l1893_189365


namespace simplify_and_evaluate_l1893_189392

noncomputable def simplified_expression (x : ℝ) : ℝ :=
  ((1 / (x - 1)) + (1 / (x + 1))) / (x^2 / (3 * x^2 - 3))

theorem simplify_and_evaluate : simplified_expression (Real.sqrt 2) = 3 * Real.sqrt 2 :=
by 
  sorry

end simplify_and_evaluate_l1893_189392


namespace probability_of_drawing_2_red_1_white_l1893_189321

def total_balls : ℕ := 7
def red_balls : ℕ := 4
def white_balls : ℕ := 3
def draws : ℕ := 3

def combinations (n k : ℕ) : ℕ := Nat.choose n k

theorem probability_of_drawing_2_red_1_white :
  (combinations red_balls 2) * (combinations white_balls 1) / (combinations total_balls draws) = 18 / 35 := by
  sorry

end probability_of_drawing_2_red_1_white_l1893_189321


namespace arithmetic_sequence_sum_ratio_l1893_189348

variable {a : ℕ → ℚ}
variable {S : ℕ → ℚ}

-- Conditions
def is_arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

def sum_of_first_n_terms (a S : ℕ → ℚ) : Prop :=
  ∀ n, S n = n * (a 1 + a n) / 2

def condition_1 (a : ℕ → ℚ) : Prop :=
  is_arithmetic_sequence a

def condition_2 (a : ℕ → ℚ) : Prop :=
  (a 5) / (a 3) = 5 / 9

-- Proof statement
theorem arithmetic_sequence_sum_ratio (a : ℕ → ℚ) (S : ℕ → ℚ)
  (h1 : condition_1 a) (h2 : condition_2 a) (h3 : sum_of_first_n_terms a S) : 
  (S 9) / (S 5) = 1 := 
sorry

end arithmetic_sequence_sum_ratio_l1893_189348


namespace spent_on_video_games_l1893_189305

-- Defining the given amounts
def initial_amount : ℕ := 84
def grocery_spending : ℕ := 21
def final_amount : ℕ := 39

-- The proof statement: Proving Lenny spent $24 on video games.
theorem spent_on_video_games : initial_amount - final_amount - grocery_spending = 24 :=
by
  sorry

end spent_on_video_games_l1893_189305


namespace barium_atoms_in_compound_l1893_189360

noncomputable def barium_atoms (total_molecular_weight : ℝ) (weight_ba_per_atom : ℝ) (weight_br_per_atom : ℝ) (num_br_atoms : ℕ) : ℝ :=
  (total_molecular_weight - (num_br_atoms * weight_br_per_atom)) / weight_ba_per_atom

theorem barium_atoms_in_compound :
  barium_atoms 297 137.33 79.90 2 = 1 :=
by
  unfold barium_atoms
  norm_num
  sorry

end barium_atoms_in_compound_l1893_189360


namespace problem_statement_l1893_189300

def A : Prop := (∀ (x : ℝ), x^2 - 3*x + 2 = 0 → x = 2)
def B : Prop := (∃ (x : ℝ), x^2 - x + 1 < 0)
def C : Prop := (¬(∀ (x : ℝ), x > 2 → x^2 - 3*x + 2 > 0))

theorem problem_statement :
  ¬ (A ∧ ∀ (x : ℝ), (B → (x^2 - x + 1) ≥ 0) ∧ (¬(A) ∧ C)) :=
sorry

end problem_statement_l1893_189300


namespace value_of_v_over_u_l1893_189381

variable (u v : ℝ) 

theorem value_of_v_over_u (h : u - v = (u + v) / 2) : v / u = 1 / 3 :=
by
  sorry

end value_of_v_over_u_l1893_189381


namespace emily_needs_375_nickels_for_book_l1893_189332

theorem emily_needs_375_nickels_for_book
  (n : ℕ)
  (book_cost : ℝ)
  (five_dollars : ℝ)
  (one_dollars : ℝ)
  (quarters : ℝ)
  (nickel_value : ℝ)
  (total_money : ℝ)
  (h1 : book_cost = 46.25)
  (h2 : five_dollars = 4 * 5)
  (h3 : one_dollars = 5 * 1)
  (h4 : quarters = 10 * 0.25)
  (h5 : nickel_value = n * 0.05)
  (h6 : total_money = five_dollars + one_dollars + quarters + nickel_value) 
  (h7 : total_money ≥ book_cost) :
  n ≥ 375 :=
by 
  sorry

end emily_needs_375_nickels_for_book_l1893_189332


namespace kitten_food_consumption_l1893_189354

-- Definitions of the given conditions
def k : ℕ := 4  -- Number of kittens
def ac : ℕ := 3  -- Number of adult cats
def f : ℕ := 7  -- Initial cans of food
def af : ℕ := 35  -- Additional cans of food needed
def days : ℕ := 7  -- Total number of days

-- Definition of the food consumption per adult cat per day
def food_per_adult_cat_per_day : ℕ := 1

-- Definition of the correct answer: food per kitten per day
def food_per_kitten_per_day : ℚ := 0.75

-- Proof statement
theorem kitten_food_consumption (k : ℕ) (ac : ℕ) (f : ℕ) (af : ℕ) (days : ℕ) (food_per_adult_cat_per_day : ℕ) :
  (ac * food_per_adult_cat_per_day * days + k * food_per_kitten_per_day * days = f + af) → 
  food_per_kitten_per_day = 0.75 :=
sorry

end kitten_food_consumption_l1893_189354


namespace expected_worth_is_1_33_l1893_189342

noncomputable def expected_worth_of_coin_flip : ℝ :=
  let prob_heads := 2 / 3
  let profit_heads := 5
  let prob_tails := 1 / 3
  let loss_tails := -6
  (prob_heads * profit_heads + prob_tails * loss_tails)

theorem expected_worth_is_1_33 : expected_worth_of_coin_flip = 1.33 := by
  sorry

end expected_worth_is_1_33_l1893_189342


namespace find_abc_squares_l1893_189363

variable (a b c x : ℕ)

theorem find_abc_squares (h1 : 1 ≤ a) (h2 : a + b + c = 9) (h3 : 99 * (c - a) = 65 * x) (h4 : 495 = 65 * x) : a^2 + b^2 + c^2 = 53 :=
  sorry

end find_abc_squares_l1893_189363


namespace alyssa_gave_away_puppies_l1893_189319

def start_puppies : ℕ := 12
def remaining_puppies : ℕ := 5

theorem alyssa_gave_away_puppies : 
  start_puppies - remaining_puppies = 7 := 
by
  sorry

end alyssa_gave_away_puppies_l1893_189319


namespace range_of_y_l1893_189325

noncomputable def y (x : ℝ) : ℝ := Real.sin x + Real.arcsin x

theorem range_of_y : 
  ∀ x, -1 ≤ x ∧ x ≤ 1 →
  y x ∈ Set.Icc (-Real.sin 1 - (Real.pi / 2)) (Real.sin 1 + (Real.pi / 2)) :=
sorry

end range_of_y_l1893_189325


namespace algebraic_simplification_l1893_189349

variables (a b : ℝ)

theorem algebraic_simplification (h : a > b ∧ b > 0) : 
  ((a + b) / ((Real.sqrt a - Real.sqrt b)^2)) * 
  (((3 * a * b - b * Real.sqrt (a * b) + a * Real.sqrt (a * b) - 3 * b^2) / 
    (1/2 * Real.sqrt (1/4 * ((a / b + b / a)^2) - 1)) + 
   (4 * a * b * Real.sqrt a + 9 * a * b * Real.sqrt b - 9 * b^2 * Real.sqrt a) / 
   (3/2 * Real.sqrt b - 2 * Real.sqrt a))) 
  = -2 * b * (a + 3 * Real.sqrt (a * b)) :=
sorry

end algebraic_simplification_l1893_189349


namespace lisa_interest_after_10_years_l1893_189370

noncomputable def compounded_amount (P : ℕ) (r : ℚ) (n : ℕ) : ℚ :=
  P * (1 + r) ^ n

theorem lisa_interest_after_10_years :
  let P := 2000
  let r := (2 : ℚ) / 100
  let n := 10
  let A := compounded_amount P r n
  A - P = 438 := by
    let P := 2000
    let r := (2 : ℚ) / 100
    let n := 10
    let A := compounded_amount P r n
    have : A - P = 438 := sorry
    exact this

end lisa_interest_after_10_years_l1893_189370


namespace simplify_fraction_l1893_189383

-- Define the problem and conditions
def numerator : ℕ := 48
def denominator : ℕ := 72
def gcd_n_d : ℕ := Nat.gcd numerator denominator

-- The proof statement
theorem simplify_fraction : (numerator / gcd_n_d) / (denominator / gcd_n_d) = 2 / 3 :=
by
  have h_gcd : gcd_n_d = 24 := by rfl
  sorry

end simplify_fraction_l1893_189383


namespace tangent_product_l1893_189308

noncomputable section

open Real

theorem tangent_product 
  (x y k1 k2 : ℝ) :
  (x / 2) ^ 2 + y ^ 2 = 1 ∧ 
  (x, y) = (-3, -3) ∧ 
  k1 + k2 = 18 / 5 ∧
  k1 * k2 = 8 / 5 → 
  (3 * k1 - 3) * (3 * k2 - 3) = 9 := 
by
  intros 
  sorry

end tangent_product_l1893_189308


namespace point_not_in_second_quadrant_l1893_189330

-- Define the point P and the condition
def point_is_in_second_quadrant (P : ℝ × ℝ) : Prop :=
  P.1 < 0 ∧ P.2 > 0

def point (m : ℝ) : ℝ × ℝ :=
  (m + 1, m)

-- The main theorem stating that P cannot be in the second quadrant
theorem point_not_in_second_quadrant (m : ℝ) : ¬ point_is_in_second_quadrant (point m) :=
by
  sorry

end point_not_in_second_quadrant_l1893_189330


namespace units_digit_17_pow_2024_l1893_189312

theorem units_digit_17_pow_2024 : (17 ^ 2024) % 10 = 1 := 
by
  sorry

end units_digit_17_pow_2024_l1893_189312


namespace bullying_instances_l1893_189314

-- Let's denote the total number of suspension days due to bullying and serious incidents.
def total_suspension_days : ℕ := (3 * (10 + 10)) + 14

-- Each instance of bullying results in a 3-day suspension.
def days_per_instance : ℕ := 3

-- The number of instances of bullying given the total suspension days.
def instances_of_bullying := total_suspension_days / days_per_instance

-- We must prove that Kris is responsible for 24 instances of bullying.
theorem bullying_instances : instances_of_bullying = 24 := by
  sorry

end bullying_instances_l1893_189314


namespace cost_per_ticket_l1893_189358

/-- Adam bought 13 tickets and after riding the ferris wheel, he had 4 tickets left.
    He spent 81 dollars riding the ferris wheel, and we want to determine how much each ticket cost. -/
theorem cost_per_ticket (initial_tickets : ℕ) (tickets_left : ℕ) (total_cost : ℕ) (used_tickets : ℕ) 
    (ticket_cost : ℕ) (h1 : initial_tickets = 13) 
    (h2 : tickets_left = 4) 
    (h3 : total_cost = 81) 
    (h4 : used_tickets = initial_tickets - tickets_left) 
    (h5 : ticket_cost = total_cost / used_tickets) : ticket_cost = 9 :=
by {
    sorry
}

end cost_per_ticket_l1893_189358


namespace sum_zero_l1893_189396

noncomputable def f : ℝ → ℝ := sorry

theorem sum_zero :
  (∀ x : ℝ, f (-x) = -f x) →
  (∀ x : ℝ, f (x + 5) = f x) →
  f (1 / 3) = 1 →
  f (16 / 3) + f (29 / 3) + f 12 + f (-7) = 0 :=
by
  intros hodd hperiod hvalue
  sorry

end sum_zero_l1893_189396


namespace students_only_biology_students_biology_or_chemistry_but_not_both_l1893_189397

def students_enrolled_in_both : ℕ := 15
def total_biology_students : ℕ := 35
def students_only_chemistry : ℕ := 18

theorem students_only_biology (h₀ : students_enrolled_in_both ≤ total_biology_students) :
  total_biology_students - students_enrolled_in_both = 20 := by
  sorry

theorem students_biology_or_chemistry_but_not_both :
  total_biology_students - students_enrolled_in_both + students_only_chemistry = 38 := by
  sorry

end students_only_biology_students_biology_or_chemistry_but_not_both_l1893_189397


namespace sum_of_ages_53_l1893_189324

variable (B D : ℕ)

def Ben_3_years_younger_than_Dan := B + 3 = D
def Ben_is_25 := B = 25
def sum_of_their_ages (B D : ℕ) := B + D

theorem sum_of_ages_53 : ∀ (B D : ℕ), Ben_3_years_younger_than_Dan B D → Ben_is_25 B → sum_of_their_ages B D = 53 :=
by
  sorry

end sum_of_ages_53_l1893_189324


namespace forgotten_angle_measure_l1893_189378

theorem forgotten_angle_measure 
  (total_sum : ℕ) 
  (measured_sum : ℕ) 
  (sides : ℕ) 
  (n_minus_2 : ℕ)
  (polygon_has_18_sides : sides = 18)
  (interior_angle_sum : total_sum = n_minus_2 * 180)
  (n_minus : n_minus_2 = (sides - 2))
  (measured : measured_sum = 2754) :
  ∃ forgotten_angle, forgotten_angle = total_sum - measured_sum ∧ forgotten_angle = 126 :=
by
  sorry

end forgotten_angle_measure_l1893_189378


namespace speed_of_train_b_l1893_189384

-- Defining the known data
def train_a_speed := 60 -- km/h
def train_a_time_after_meeting := 9 -- hours
def train_b_time_after_meeting := 4 -- hours

-- Statement we want to prove
theorem speed_of_train_b : ∃ (V_b : ℝ), V_b = 135 :=
by
  -- Sorry placeholder, as the proof is not required
  sorry

end speed_of_train_b_l1893_189384


namespace frequency_of_fourth_group_l1893_189350

theorem frequency_of_fourth_group (f₁ f₂ f₃ f₄ f₅ f₆ : ℝ) (h1 : f₁ + f₂ + f₃ = 0.65) (h2 : f₅ + f₆ = 0.32) (h3 : f₁ + f₂ + f₃ + f₄ + f₅ + f₆ = 1) :
  f₄ = 0.03 :=
by 
  sorry

end frequency_of_fourth_group_l1893_189350


namespace molecular_weight_3_moles_l1893_189393

theorem molecular_weight_3_moles
  (C_weight : ℝ)
  (H_weight : ℝ)
  (N_weight : ℝ)
  (O_weight : ℝ)
  (Molecular_formula : ℕ → ℕ → ℕ → ℕ → Prop)
  (molecular_weight : ℝ)
  (moles : ℝ) :
  C_weight = 12.01 →
  H_weight = 1.008 →
  N_weight = 14.01 →
  O_weight = 16.00 →
  Molecular_formula 13 9 5 7 →
  molecular_weight = 156.13 + 9.072 + 70.05 + 112.00 →
  moles = 3 →
  3 * molecular_weight = 1041.756 :=
by
  sorry

end molecular_weight_3_moles_l1893_189393


namespace john_uber_profit_l1893_189359

theorem john_uber_profit
  (P0 : ℝ) (T : ℝ) (P : ℝ)
  (hP0 : P0 = 18000)
  (hT : T = 6000)
  (hP : P = 18000) :
  P + (P0 - T) = 30000 :=
by
  sorry

end john_uber_profit_l1893_189359


namespace lorelei_vase_rose_count_l1893_189322

theorem lorelei_vase_rose_count :
  let r := 12
  let p := 18
  let y := 20
  let o := 8
  let picked_r := 0.5 * r
  let picked_p := 0.5 * p
  let picked_y := 0.25 * y
  let picked_o := 0.25 * o
  picked_r + picked_p + picked_y + picked_o = 22 := 
by
  sorry

end lorelei_vase_rose_count_l1893_189322


namespace units_digit_S7890_l1893_189316

noncomputable def c : ℝ := 4 + 3 * Real.sqrt 2
noncomputable def d : ℝ := 4 - 3 * Real.sqrt 2
noncomputable def S (n : ℕ) : ℝ := (1/2:ℝ) * (c^n + d^n)

theorem units_digit_S7890 : (S 7890) % 10 = 8 :=
sorry

end units_digit_S7890_l1893_189316


namespace proof_problem_l1893_189388

-- Let P, Q, R be points on a circle of radius s
-- Given: PQ = PR, PQ > s, and minor arc QR is 2s
-- Prove: PQ / QR = sin(1)

noncomputable def point_on_circle (s : ℝ) : ℝ → ℝ × ℝ := sorry
def radius {s : ℝ} (P Q : ℝ × ℝ ) : Prop := dist P Q = s

theorem proof_problem (s : ℝ) (P Q R : ℝ × ℝ)
  (hPQ : dist P Q = dist P R)
  (hPQ_gt_s : dist P Q > s)
  (hQR_arc_len : 1 = s) :
  dist P Q / (2 * s) = Real.sin 1 := 
sorry

end proof_problem_l1893_189388


namespace coeff_x6_in_expansion_l1893_189386

theorem coeff_x6_in_expansion : 
  (Polynomial.coeff ((1 - 3 * Polynomial.X ^ 3) ^ 7 : Polynomial ℤ) 6) = 189 :=
by
  sorry

end coeff_x6_in_expansion_l1893_189386


namespace problem_demo_l1893_189310

open Set

def U : Set ℕ := {x | 0 < x ∧ x ≤ 8}
def S : Set ℕ := {1, 2, 4, 5}
def T : Set ℕ := {3, 5, 7}

theorem problem_demo : S ∩ (U \ T) = {1, 2, 4} :=
by
  sorry

end problem_demo_l1893_189310


namespace number_of_dogs_l1893_189352

theorem number_of_dogs (total_animals cats : ℕ) (probability : ℚ) (h1 : total_animals = 7) (h2 : cats = 2) (h3 : probability = 2 / 7) :
  total_animals - cats = 5 := 
by
  sorry

end number_of_dogs_l1893_189352


namespace Lulu_blueberry_pies_baked_l1893_189345

-- Definitions of conditions
def Lola_mini_cupcakes := 13
def Lola_pop_tarts := 10
def Lola_blueberry_pies := 8
def Lola_total_pastries := Lola_mini_cupcakes + Lola_pop_tarts + Lola_blueberry_pies
def Lulu_mini_cupcakes := 16
def Lulu_pop_tarts := 12
def total_pastries := 73

-- Prove that Lulu baked 14 blueberry pies
theorem Lulu_blueberry_pies_baked : 
  ∃ (Lulu_blueberry_pies : Nat), 
    Lola_total_pastries + Lulu_mini_cupcakes + Lulu_pop_tarts + Lulu_blueberry_pies = total_pastries ∧ 
    Lulu_blueberry_pies = 14 := by
  sorry

end Lulu_blueberry_pies_baked_l1893_189345


namespace smallest_solution_neg_two_l1893_189309

-- We set up the expressions and then state the smallest solution
def smallest_solution (x : ℝ) : Prop :=
  x * abs x = 3 * x + 2

theorem smallest_solution_neg_two :
  ∃ x : ℝ, smallest_solution x ∧ (∀ y : ℝ, smallest_solution y → y ≥ x) ∧ x = -2 :=
by
  sorry

end smallest_solution_neg_two_l1893_189309


namespace cookies_per_box_correct_l1893_189398

variable (cookies_per_box : ℕ)

-- Define the conditions
def morning_cookie : ℕ := 1 / 2
def bed_cookie : ℕ := 1 / 2
def day_cookies : ℕ := 2
def daily_cookies := morning_cookie + bed_cookie + day_cookies

def days : ℕ := 30
def total_cookies := days * daily_cookies

def boxes : ℕ := 2
def total_cookies_in_boxes : ℕ := cookies_per_box * boxes

-- Theorem we want to prove
theorem cookies_per_box_correct :
  total_cookies_in_boxes = 90 → cookies_per_box = 45 :=
by
  sorry

end cookies_per_box_correct_l1893_189398


namespace domain_cannot_be_0_to_3_l1893_189304

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2 * x + 2

-- Define the range of the function f
def range_f : Set ℝ := Set.Icc 1 2

-- Statement that the domain [0, 3] cannot be the domain of f given the range
theorem domain_cannot_be_0_to_3 :
  ∀ (f : ℝ → ℝ) (range_f : Set ℝ),
    (∀ x, 1 ≤ f x ∧ f x ≤ 2) →
    ¬ ∃ dom : Set ℝ, dom = Set.Icc 0 3 ∧ 
      (∀ x ∈ dom, f x ∈ range_f) :=
by
  sorry

end domain_cannot_be_0_to_3_l1893_189304


namespace square_area_ratio_l1893_189343

theorem square_area_ratio (s₁ s₂ d₂ : ℝ)
  (h1 : s₁ = 2 * d₂)
  (h2 : d₂ = s₂ * Real.sqrt 2) :
  (s₁^2) / (s₂^2) = 8 :=
by
  sorry

end square_area_ratio_l1893_189343


namespace evening_temperature_is_correct_l1893_189369

-- Define the temperatures at noon and in the evening
def T_noon : ℤ := 3
def T_evening : ℤ := -2

-- State the theorem to prove
theorem evening_temperature_is_correct : T_evening = -2 := by
  sorry

end evening_temperature_is_correct_l1893_189369


namespace cannot_finish_third_l1893_189380

-- Define the racers
inductive Racer
| P | Q | R | S | T | U
open Racer

-- Define the conditions
def beats (a b : Racer) : Prop := sorry  -- placeholder for strict order
def ties (a b : Racer) : Prop := sorry   -- placeholder for tie condition
def position (r : Racer) (p : Fin (6)) : Prop := sorry  -- placeholder for position in the race

theorem cannot_finish_third :
  (beats P Q) ∧
  (ties P R) ∧
  (beats Q S) ∧
  ∃ p₁ p₂ p₃, position P p₁ ∧ position T p₂ ∧ position Q p₃ ∧ p₁ < p₂ ∧ p₂ < p₃ ∧
  ∃ p₄ p₅, position U p₄ ∧ position S p₅ ∧ p₄ < p₅ →
  ¬ position P (3 : Fin (6)) ∧ ¬ position U (3 : Fin (6)) ∧ ¬ position S (3 : Fin (6)) :=
by sorry   -- Proof is omitted

end cannot_finish_third_l1893_189380


namespace find_triangle_height_l1893_189303

-- Define the problem conditions
def Rectangle.perimeter (l : ℕ) (w : ℕ) : ℕ := 2 * l + 2 * w
def Rectangle.area (l : ℕ) (w : ℕ) : ℕ := l * w
def Triangle.area (b : ℕ) (h : ℕ) : ℕ := (b * h) / 2

-- Conditions
namespace Conditions
  -- Perimeter of the rectangle is 60 cm
  def rect_perimeter (l w : ℕ) : Prop := Rectangle.perimeter l w = 60
  -- Base of the right triangle is 15 cm
  def tri_base : ℕ := 15
  -- Areas of the rectangle and the triangle are equal
  def equal_areas (l w h : ℕ) : Prop := Rectangle.area l w = Triangle.area tri_base h
end Conditions

-- Proof problem: Given these conditions, prove h = 30
theorem find_triangle_height (l w h : ℕ) 
  (h1 : Conditions.rect_perimeter l w)
  (h2 : Conditions.equal_areas l w h) : h = 30 :=
  sorry

end find_triangle_height_l1893_189303


namespace sugar_left_correct_l1893_189335

-- Define the total amount of sugar bought by Pamela
def total_sugar : ℝ := 9.8

-- Define the amount of sugar spilled by Pamela
def spilled_sugar : ℝ := 5.2

-- Define the amount of sugar left after spilling
def sugar_left : ℝ := total_sugar - spilled_sugar

-- State that the amount of sugar left should be equivalent to the correct answer
theorem sugar_left_correct : sugar_left = 4.6 :=
by
  sorry

end sugar_left_correct_l1893_189335


namespace find_divisor_l1893_189307

-- Define the given and calculated values in the conditions
def initial_value : ℕ := 165826
def subtracted_value : ℕ := 2
def resulting_value : ℕ := initial_value - subtracted_value

-- Define the goal: to find the smallest divisor of resulting_value other than 1
theorem find_divisor (d : ℕ) (h1 : initial_value - subtracted_value = resulting_value)
  (h2 : resulting_value % d = 0) (h3 : d > 1) : d = 2 := by
  sorry

end find_divisor_l1893_189307


namespace relative_value_ex1_max_value_of_m_plus_n_l1893_189377

-- Definition of relative relationship value
def relative_relationship_value (a b n : ℚ) : ℚ := abs (a - n) + abs (b - n)

-- First problem statement
theorem relative_value_ex1 : relative_relationship_value 2 (-5) 2 = 7 := by
  sorry

-- Second problem statement: maximum value of m + n given the relative relationship value is 2
theorem max_value_of_m_plus_n (m n : ℚ) (h : relative_relationship_value m n 2 = 2) : m + n ≤ 6 := by
  sorry

end relative_value_ex1_max_value_of_m_plus_n_l1893_189377


namespace value_of_expression_l1893_189382

theorem value_of_expression :
  3 + Real.sqrt 3 + 1 / (3 + Real.sqrt 3) + 1 / (Real.sqrt 3 - 3) = 3 + 2 * Real.sqrt 3 / 3 :=
by sorry

end value_of_expression_l1893_189382


namespace max_abs_value_l1893_189329

theorem max_abs_value (x y : ℝ) (h1 : |x - 1| ≤ 1) (h2 : |y - 2| ≤ 1) : |x - 2 * y + 1| ≤ 5 :=
by
  sorry

end max_abs_value_l1893_189329


namespace solve_k_l1893_189357

theorem solve_k (x y k : ℝ) (h1 : x + 2 * y = k - 1) (h2 : 2 * x + y = 5 * k + 4) (h3 : x + y = 5) :
  k = 2 :=
sorry

end solve_k_l1893_189357


namespace finding_a_of_geometric_sequence_l1893_189317
noncomputable def geometric_sequence_a : Prop :=
  ∃ a : ℝ, (1, a, 2) = (1, a, 2) ∧ a^2 = 2

theorem finding_a_of_geometric_sequence :
  ∃ a : ℝ, (1, a, 2) = (1, a, 2) → a = Real.sqrt 2 ∨ a = -Real.sqrt 2 :=
by
  sorry

end finding_a_of_geometric_sequence_l1893_189317


namespace modulus_of_complex_division_l1893_189353

noncomputable def complexDivisionModulus : ℂ := Complex.normSq (2 * Complex.I / (Complex.I - 1))

theorem modulus_of_complex_division : complexDivisionModulus = Real.sqrt 2 := by
  sorry

end modulus_of_complex_division_l1893_189353


namespace lambda_inequality_l1893_189366

-- Define the problem hypothesis and conclusion
theorem lambda_inequality (n : ℕ) (hn : n ≥ 4) (lambda_n : ℝ) :
  lambda_n ≥ 2 * Real.sin ((n-2) * Real.pi / (2 * n)) :=
by
  -- Placeholder for the proof
  sorry

end lambda_inequality_l1893_189366


namespace stacy_days_to_complete_paper_l1893_189306

-- Conditions as definitions
def total_pages : ℕ := 63
def pages_per_day : ℕ := 9

-- The problem statement
theorem stacy_days_to_complete_paper : total_pages / pages_per_day = 7 :=
by
  sorry

end stacy_days_to_complete_paper_l1893_189306


namespace probability_two_of_three_survive_l1893_189390

-- Let's define the necessary components
def binomial_coefficient (n k : ℕ) : ℕ := Nat.choose n k

-- Define the probability of exactly 2 out of 3 seedlings surviving
theorem probability_two_of_three_survive (p : ℝ) (h0 : 0 ≤ p) (h1 : p ≤ 1) :
  binomial_coefficient 3 2 * p^2 * (1 - p) = 3 * p^2 * (1 - p) :=
by
  sorry

end probability_two_of_three_survive_l1893_189390


namespace hyperbola_vertex_distance_l1893_189399

theorem hyperbola_vertex_distance (a b : ℝ) (h_eq : a^2 = 16) (hyperbola_eq : ∀ x y : ℝ, 
  (x^2 / 16) - (y^2 / 9) = 1) : 
  (2 * a) = 8 :=
by
  have h_a : a = 4 := by sorry
  rw [h_a]
  norm_num

end hyperbola_vertex_distance_l1893_189399


namespace shaded_area_l1893_189302

theorem shaded_area (x1 y1 x2 y2 x3 y3 : ℝ) 
  (vA vB vC vD vE vF : ℝ × ℝ)
  (h1 : vA = (0, 0))
  (h2 : vB = (0, 12))
  (h3 : vC = (12, 12))
  (h4 : vD = (12, 0))
  (h5 : vE = (24, 0))
  (h6 : vF = (18, 12))
  (h_base : 32 - 12 = 20)
  (h_height : 12 = 12) :
  (1 / 2 : ℝ) * 20 * 12 = 120 :=
by
  sorry

end shaded_area_l1893_189302


namespace anna_chocolates_l1893_189323

theorem anna_chocolates : ∃ (n : ℕ), (5 * 2^(n-1) > 200) ∧ n = 7 :=
by
  sorry

end anna_chocolates_l1893_189323


namespace positive_m_of_quadratic_has_one_real_root_l1893_189328

theorem positive_m_of_quadratic_has_one_real_root : 
  (∃ m : ℝ, m > 0 ∧ ∀ x : ℝ, x^2 + 6 * m * x + m = 0 → x = -3 * m) :=
by
  sorry

end positive_m_of_quadratic_has_one_real_root_l1893_189328


namespace picnic_total_cost_is_correct_l1893_189338

-- Define the conditions given in the problem
def number_of_people : Nat := 4
def cost_per_sandwich : Nat := 5
def cost_per_fruit_salad : Nat := 3
def sodas_per_person : Nat := 2
def cost_per_soda : Nat := 2
def number_of_snack_bags : Nat := 3
def cost_per_snack_bag : Nat := 4

-- Calculate the total cost based on the given conditions
def total_cost_sandwiches : Nat := number_of_people * cost_per_sandwich
def total_cost_fruit_salads : Nat := number_of_people * cost_per_fruit_salad
def total_cost_sodas : Nat := number_of_people * sodas_per_person * cost_per_soda
def total_cost_snack_bags : Nat := number_of_snack_bags * cost_per_snack_bag

def total_spent : Nat := total_cost_sandwiches + total_cost_fruit_salads + total_cost_sodas + total_cost_snack_bags

-- The statement we want to prove
theorem picnic_total_cost_is_correct : total_spent = 60 :=
by
  -- Proof would be written here
  sorry

end picnic_total_cost_is_correct_l1893_189338


namespace find_units_digit_l1893_189315

def is_three_digit (n : ℕ) := 100 ≤ n ∧ n < 1000

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

def is_divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + ((n % 100) / 10) + (n % 10)

theorem find_units_digit (n : ℕ) :
  is_three_digit n →
  (is_perfect_square n ∨ is_even n ∨ is_divisible_by_11 n ∨ digit_sum n = 12) ∧
  (¬is_perfect_square n ∨ ¬is_even n ∨ ¬is_divisible_by_11 n ∨ ¬(digit_sum n = 12)) →
  (n % 10 = 4) :=
sorry

end find_units_digit_l1893_189315


namespace triangle_XYZ_r_s_max_sum_l1893_189326

theorem triangle_XYZ_r_s_max_sum
  (r s : ℝ)
  (h_area : 1/2 * abs (r * (15 - 18) + 10 * (18 - s) + 20 * (s - 15)) = 90)
  (h_slope : s = -3 * r + 61.5) :
  r + s ≤ 42.91 :=
sorry

end triangle_XYZ_r_s_max_sum_l1893_189326


namespace total_coffee_cost_l1893_189351

def vacation_days : ℕ := 40
def daily_coffee : ℕ := 3
def pods_per_box : ℕ := 30
def box_cost : ℕ := 8

theorem total_coffee_cost : vacation_days * daily_coffee / pods_per_box * box_cost = 32 := by
  -- proof goes here
  sorry

end total_coffee_cost_l1893_189351


namespace sum_modulo_nine_l1893_189341

theorem sum_modulo_nine :
  (88135 + 88136 + 88137 + 88138 + 88139 + 88140) % 9 = 3 := 
by
  sorry

end sum_modulo_nine_l1893_189341
