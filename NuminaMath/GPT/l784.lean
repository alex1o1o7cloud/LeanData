import Mathlib

namespace NUMINAMATH_GPT_negation_universal_exists_l784_78402

open Classical

theorem negation_universal_exists :
  (¬ ∀ x : ℝ, x > 0 → (x^2 - x + 3 > 0)) ↔ ∃ x : ℝ, x > 0 ∧ (x^2 - x + 3 ≤ 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_universal_exists_l784_78402


namespace NUMINAMATH_GPT_certain_event_is_eventC_l784_78433

-- Definitions for the conditions:
def eventA := "A vehicle randomly arriving at an intersection encountering a red light"
def eventB := "The sun rising from the west in the morning"
def eventC := "Two out of 400 people sharing the same birthday"
def eventD := "Tossing a fair coin with the head facing up"

-- The proof goal: proving that event C is the certain event.
theorem certain_event_is_eventC : eventC = "Two out of 400 people sharing the same birthday" :=
sorry

end NUMINAMATH_GPT_certain_event_is_eventC_l784_78433


namespace NUMINAMATH_GPT_algebraic_expression_correct_l784_78401

variable (x y : ℤ)

theorem algebraic_expression_correct (h : (x - y) / (x + y) = 3) : (2 * (x - y)) / (x + y) - (x + y) / (3 * (x - y)) = 53 / 9 := 
by  
  sorry

end NUMINAMATH_GPT_algebraic_expression_correct_l784_78401


namespace NUMINAMATH_GPT_find_abc_l784_78483

theorem find_abc (a b c : ℝ) 
  (h1 : a = 0.8 * b) 
  (h2 : c = 1.4 * b) 
  (h3 : c - a = 72) : 
  a = 96 ∧ b = 120 ∧ c = 168 := 
by
  sorry

end NUMINAMATH_GPT_find_abc_l784_78483


namespace NUMINAMATH_GPT_number_less_than_neg_one_is_neg_two_l784_78438

theorem number_less_than_neg_one_is_neg_two : ∃ x : ℤ, x = -1 - 1 ∧ x = -2 := by
  sorry

end NUMINAMATH_GPT_number_less_than_neg_one_is_neg_two_l784_78438


namespace NUMINAMATH_GPT_trigonometric_expression_identity_l784_78476

open Real

theorem trigonometric_expression_identity :
  (1 - 1 / cos (35 * (pi / 180))) * 
  (1 + 1 / sin (55 * (pi / 180))) * 
  (1 - 1 / sin (35 * (pi / 180))) * 
  (1 + 1 / cos (55 * (pi / 180))) = 1 := by
  sorry

end NUMINAMATH_GPT_trigonometric_expression_identity_l784_78476


namespace NUMINAMATH_GPT_alice_prob_after_three_turns_l784_78442

/-
Definition of conditions:
 - Alice starts with the ball.
 - If Alice has the ball, there is a 1/3 chance that she will toss it to Bob and a 2/3 chance that she will keep the ball.
 - If Bob has the ball, there is a 1/4 chance that he will toss it to Alice and a 3/4 chance that he keeps the ball.
-/

def alice_to_bob : ℚ := 1/3
def alice_keeps : ℚ := 2/3
def bob_to_alice : ℚ := 1/4
def bob_keeps : ℚ := 3/4

theorem alice_prob_after_three_turns :
  alice_to_bob * bob_keeps * bob_to_alice +
  alice_keeps * alice_keeps * alice_keeps +
  alice_to_bob * bob_to_alice * alice_keeps = 179/432 :=
by
  sorry

end NUMINAMATH_GPT_alice_prob_after_three_turns_l784_78442


namespace NUMINAMATH_GPT_g_at_neg10_l784_78488

def g (x : ℤ) : ℤ := 
  if x < -3 then 3 * x + 7 else 4 - x

theorem g_at_neg10 : g (-10) = -23 := by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_g_at_neg10_l784_78488


namespace NUMINAMATH_GPT_min_value_l784_78468

theorem min_value (x : ℝ) (h : x > 1) : ∃ m : ℝ, m = 2 * Real.sqrt 5 ∧ ∀ y : ℝ, y = Real.sqrt (x - 1) → (x = y^2 + 1) → (x + 4) / y = m :=
by
  sorry

end NUMINAMATH_GPT_min_value_l784_78468


namespace NUMINAMATH_GPT_find_cake_box_width_l784_78443

-- Define the dimensions of the carton
def carton_length := 25
def carton_width := 42
def carton_height := 60
def carton_volume := carton_length * carton_width * carton_height

-- Define the dimensions of the cake box
def cake_box_length := 8
variable (cake_box_width : ℝ) -- This is the unknown width we need to find
def cake_box_height := 5
def cake_box_volume := cake_box_length * cake_box_width * cake_box_height

-- Maximum number of cake boxes that can be placed in the carton
def max_cake_boxes := 210
def total_cake_boxes_volume := max_cake_boxes * cake_box_volume cake_box_width

-- Theorem to prove
theorem find_cake_box_width : cake_box_width = 7.5 :=
by
  sorry

end NUMINAMATH_GPT_find_cake_box_width_l784_78443


namespace NUMINAMATH_GPT_rectangle_properties_l784_78408

theorem rectangle_properties :
  ∃ (length width : ℝ),
    (length / width = 3) ∧ 
    (length * width = 75) ∧
    (length = 15) ∧
    (width = 5) ∧
    ∀ (side : ℝ), 
      (side^2 = 75) → 
      (side - width > 3) :=
by
  sorry

end NUMINAMATH_GPT_rectangle_properties_l784_78408


namespace NUMINAMATH_GPT_money_left_after_purchase_l784_78489

noncomputable def initial_money : ℝ := 200
noncomputable def candy_bars : ℝ := 25
noncomputable def bags_of_chips : ℝ := 10
noncomputable def soft_drinks : ℝ := 15

noncomputable def cost_per_candy_bar : ℝ := 3
noncomputable def cost_per_bag_of_chips : ℝ := 2.5
noncomputable def cost_per_soft_drink : ℝ := 1.75

noncomputable def discount_candy_bars : ℝ := 0.10
noncomputable def discount_bags_of_chips : ℝ := 0.05
noncomputable def sales_tax : ℝ := 0.06

theorem money_left_after_purchase : initial_money - 
  ( ((candy_bars * cost_per_candy_bar * (1 - discount_candy_bars)) + 
    (bags_of_chips * cost_per_bag_of_chips * (1 - discount_bags_of_chips)) + 
    (soft_drinks * cost_per_soft_drink)) * 
    (1 + sales_tax)) = 75.45 := by
  sorry

end NUMINAMATH_GPT_money_left_after_purchase_l784_78489


namespace NUMINAMATH_GPT_find_multiple_l784_78441

theorem find_multiple (x m : ℝ) (h₁ : 10 * x = m * x - 36) (h₂ : x = -4.5) : m = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_multiple_l784_78441


namespace NUMINAMATH_GPT_answer_to_rarely_infrequently_word_l784_78484

-- Declare variables and definitions based on given conditions
-- In this context, we'll introduce a basic definition for the word "seldom".

noncomputable def is_word_meaning_rarely (w : String) : Prop :=
  w = "seldom"

-- Now state the problem in the form of a Lean theorem
theorem answer_to_rarely_infrequently_word : ∃ w, is_word_meaning_rarely w :=
by
  use "seldom"
  unfold is_word_meaning_rarely
  rfl

end NUMINAMATH_GPT_answer_to_rarely_infrequently_word_l784_78484


namespace NUMINAMATH_GPT_knicks_win_tournament_probability_l784_78415

noncomputable def knicks_win_probability : ℚ :=
  let knicks_win_proba := 2 / 5
  let heat_win_proba := 3 / 5
  let first_4_games_scenarios := 6 * (knicks_win_proba^2 * heat_win_proba^2)
  first_4_games_scenarios * knicks_win_proba

theorem knicks_win_tournament_probability :
  knicks_win_probability = 432 / 3125 :=
by
  sorry

end NUMINAMATH_GPT_knicks_win_tournament_probability_l784_78415


namespace NUMINAMATH_GPT_first_car_departure_time_l784_78497

variable (leave_time : Nat) -- in minutes past 8:00 am

def speed : Nat := 60 -- km/h
def firstCarTimeAt32 : Nat := 32 -- minutes since 8:00 am
def secondCarFactorAt32 : Nat := 3
def firstCarTimeAt39 : Nat := 39 -- minutes since 8:00 am
def secondCarFactorAt39 : Nat := 2

theorem first_car_departure_time :
  let firstCarSpeed := (60 / 60 : Nat) -- km/min
  let d1_32 := firstCarSpeed * firstCarTimeAt32
  let d2_32 := firstCarSpeed * (firstCarTimeAt32 - leave_time)
  let d1_39 := firstCarSpeed * firstCarTimeAt39
  let d2_39 := firstCarSpeed * (firstCarTimeAt39 - leave_time)
  d1_32 = secondCarFactorAt32 * d2_32 →
  d1_39 = secondCarFactorAt39 * d2_39 →
  leave_time = 11 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_first_car_departure_time_l784_78497


namespace NUMINAMATH_GPT_molecular_weight_one_mole_of_AlPO4_l784_78437

theorem molecular_weight_one_mole_of_AlPO4
  (molecular_weight_4_moles : ℝ)
  (h : molecular_weight_4_moles = 488) :
  molecular_weight_4_moles / 4 = 122 :=
by
  sorry

end NUMINAMATH_GPT_molecular_weight_one_mole_of_AlPO4_l784_78437


namespace NUMINAMATH_GPT_find_f2_l784_78446

noncomputable def f (x a b : ℝ) : ℝ := x^5 + a * x^3 + b * x - 8

theorem find_f2 (a b : ℝ) (h : f (-2) a b = 0) : f 2 a b = -16 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_f2_l784_78446


namespace NUMINAMATH_GPT_parabola_min_value_l784_78455

variable {x0 y0 : ℝ}

def isOnParabola (x0 y0 : ℝ) : Prop := x0^2 = y0

noncomputable def expression (y0 x0 : ℝ) : ℝ :=
  Real.sqrt 2 * y0 + |x0 - y0 - 2|

theorem parabola_min_value :
  isOnParabola x0 y0 → ∃ (m : ℝ), m = (9 / 4 : ℝ) - (Real.sqrt 2 / 4) ∧ 
  ∀ y0 x0, expression y0 x0 ≥ (9 / 4 : ℝ) - (Real.sqrt 2 / 4) := 
by
  sorry

end NUMINAMATH_GPT_parabola_min_value_l784_78455


namespace NUMINAMATH_GPT_solve_weights_problem_l784_78471

variable (a b c d : ℕ) 

def weights_problem := 
  a + b = 280 ∧ 
  a + d = 300 ∧ 
  c + d = 290 → 
  b + c = 270

theorem solve_weights_problem (a b c d : ℕ) : weights_problem a b c d :=
 by
  sorry

end NUMINAMATH_GPT_solve_weights_problem_l784_78471


namespace NUMINAMATH_GPT_total_pieces_of_art_l784_78458

variable (A : ℕ) (displayed : ℕ) (sculptures_on_display : ℕ) (not_on_display : ℕ) (paintings_not_on_display : ℕ) (sculptures_not_on_display : ℕ)

-- Constants and conditions from the problem
axiom H1 : displayed = 1 / 3 * A
axiom H2 : sculptures_on_display = 1 / 6 * displayed
axiom H3 : not_on_display = 2 / 3 * A
axiom H4 : paintings_not_on_display = 1 / 3 * not_on_display
axiom H5 : sculptures_not_on_display = 800
axiom H6 : sculptures_not_on_display = 2 / 3 * not_on_display

-- Prove that the total number of pieces of art is 1800
theorem total_pieces_of_art : A = 1800 :=
by
  sorry

end NUMINAMATH_GPT_total_pieces_of_art_l784_78458


namespace NUMINAMATH_GPT_sister_weight_difference_is_12_l784_78427

-- Define Antonio's weight
def antonio_weight : ℕ := 50

-- Define the combined weight of Antonio and his sister
def combined_weight : ℕ := 88

-- Define the weight of Antonio's sister
def sister_weight : ℕ := combined_weight - antonio_weight

-- Define the weight difference
def weight_difference : ℕ := antonio_weight - sister_weight

-- Theorem statement to prove the weight difference is 12 kg
theorem sister_weight_difference_is_12 : weight_difference = 12 := by
  sorry

end NUMINAMATH_GPT_sister_weight_difference_is_12_l784_78427


namespace NUMINAMATH_GPT_tom_balloons_count_l784_78418

-- Define the number of balloons Tom initially has
def balloons_initial : Nat := 30

-- Define the number of balloons Tom gave away
def balloons_given : Nat := 16

-- Define the number of balloons Tom now has
def balloons_remaining : Nat := balloons_initial - balloons_given

theorem tom_balloons_count :
  balloons_remaining = 14 := by
  sorry

end NUMINAMATH_GPT_tom_balloons_count_l784_78418


namespace NUMINAMATH_GPT_no_30_cents_l784_78480

/-- Given six coins selected from nickels (5 cents), dimes (10 cents), and quarters (25 cents),
prove that the total value of the six coins cannot be 30 cents or less. -/
theorem no_30_cents {n d q : ℕ} (h : n + d + q = 6) (hn : n * 5 + d * 10 + q * 25 <= 30) : false :=
by
  sorry

end NUMINAMATH_GPT_no_30_cents_l784_78480


namespace NUMINAMATH_GPT_solution_proof_l784_78432

noncomputable def f (n : ℕ) : ℝ := Real.logb 143 (n^2)

theorem solution_proof : f 7 + f 11 + f 13 = 2 + 2 * Real.logb 143 7 := by
  sorry

end NUMINAMATH_GPT_solution_proof_l784_78432


namespace NUMINAMATH_GPT_distinct_sequences_ten_flips_l784_78449

-- Define the problem condition and question
def flip_count : ℕ := 10

-- Define the function to calculate the number of distinct sequences
def number_of_sequences (n : ℕ) : ℕ := 2 ^ n

-- Statement to be proven
theorem distinct_sequences_ten_flips : number_of_sequences flip_count = 1024 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_distinct_sequences_ten_flips_l784_78449


namespace NUMINAMATH_GPT_joe_eggs_town_hall_l784_78495

-- Define the conditions.
def eggs_club_house : ℕ := 12
def eggs_park : ℕ := 5
def eggs_total : ℕ := 20

-- Define the desired result.
def eggs_town_hall : ℕ := eggs_total - eggs_club_house - eggs_park

-- The statement that needs to be proved.
theorem joe_eggs_town_hall : eggs_town_hall = 3 :=
by
  sorry

end NUMINAMATH_GPT_joe_eggs_town_hall_l784_78495


namespace NUMINAMATH_GPT_calendar_reuse_initial_year_l784_78491

theorem calendar_reuse_initial_year (y k : ℕ)
    (h2064 : 2052 % 4 = 0)
    (h_y: y + 28 * k = 2052) :
    y = 1912 := by
  sorry

end NUMINAMATH_GPT_calendar_reuse_initial_year_l784_78491


namespace NUMINAMATH_GPT_polly_breakfast_minutes_l784_78439
open Nat

theorem polly_breakfast_minutes (B : ℕ) 
  (lunch_minutes : ℕ)
  (dinner_4_days_minutes : ℕ)
  (dinner_3_days_minutes : ℕ)
  (total_minutes : ℕ)
  (h1 : lunch_minutes = 5 * 7)
  (h2 : dinner_4_days_minutes = 10 * 4)
  (h3 : dinner_3_days_minutes = 30 * 3)
  (h4 : total_minutes = 305) 
  (h5 : 7 * B + lunch_minutes + dinner_4_days_minutes + dinner_3_days_minutes = total_minutes) :
  B = 20 :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_polly_breakfast_minutes_l784_78439


namespace NUMINAMATH_GPT_find_f_three_l784_78477

noncomputable def f : ℝ → ℝ := sorry -- f(x) is a linear function

axiom f_linear : ∃ (a b : ℝ), ∀ x, f x = a * x + b

axiom equation : ∀ x, f x = 3 * (f⁻¹ x) + 9

axiom f_zero : f 0 = 3

axiom f_inv_three : f⁻¹ 3 = 0

theorem find_f_three : f 3 = 6 * Real.sqrt 3 := 
by sorry

end NUMINAMATH_GPT_find_f_three_l784_78477


namespace NUMINAMATH_GPT_positive_number_property_l784_78463

theorem positive_number_property (y : ℝ) (hy : 0 < y) : 
  (y^2 / 100) + 6 = 10 → y = 20 := by
  sorry

end NUMINAMATH_GPT_positive_number_property_l784_78463


namespace NUMINAMATH_GPT_terminating_decimal_l784_78431

theorem terminating_decimal : (47 : ℚ) / (2 * 5^4) = 376 / 10^4 :=
by sorry

end NUMINAMATH_GPT_terminating_decimal_l784_78431


namespace NUMINAMATH_GPT_polygon_angle_arithmetic_progression_l784_78444

theorem polygon_angle_arithmetic_progression
  (h1 : ∀ {n : ℕ}, n ≥ 3)   -- The polygon is convex and n-sided
  (h2 : ∀ (angles : Fin n → ℝ), (∀ i j, i < j → angles i + 5 = angles j))   -- The interior angles form an arithmetic progression with a common difference of 5°
  (h3 : ∀ (angles : Fin n → ℝ), (∃ i, angles i = 160))  -- The largest angle is 160°
  : n = 9 := sorry

end NUMINAMATH_GPT_polygon_angle_arithmetic_progression_l784_78444


namespace NUMINAMATH_GPT_water_saving_percentage_l784_78407

/-- 
Given:
1. The old toilet uses 5 gallons of water per flush.
2. The household flushes 15 times per day.
3. John saved 1800 gallons of water in June.

Prove that the percentage of water saved per flush by the new toilet compared 
to the old one is 80%.
-/
theorem water_saving_percentage 
  (old_toilet_usage_per_flush : ℕ)
  (flushes_per_day : ℕ)
  (savings_in_june : ℕ)
  (days_in_june : ℕ) :
  old_toilet_usage_per_flush = 5 →
  flushes_per_day = 15 →
  savings_in_june = 1800 →
  days_in_june = 30 →
  (old_toilet_usage_per_flush * flushes_per_day * days_in_june - savings_in_june)
  * 100 / (old_toilet_usage_per_flush * flushes_per_day * days_in_june) = 80 :=
by 
  sorry

end NUMINAMATH_GPT_water_saving_percentage_l784_78407


namespace NUMINAMATH_GPT_find_extra_lives_first_level_l784_78403

-- Conditions as definitions
def initial_lives : ℕ := 2
def extra_lives_second_level : ℕ := 11
def total_lives_after_second_level : ℕ := 19

-- Definition representing the extra lives in the first level
def extra_lives_first_level (x : ℕ) : Prop :=
  initial_lives + x + extra_lives_second_level = total_lives_after_second_level

-- The theorem we need to prove
theorem find_extra_lives_first_level : ∃ x : ℕ, extra_lives_first_level x ∧ x = 6 :=
by
  sorry  -- Placeholder for the proof

end NUMINAMATH_GPT_find_extra_lives_first_level_l784_78403


namespace NUMINAMATH_GPT_minimum_value_problem_l784_78453

theorem minimum_value_problem (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x^3 + 4 * x^2 + 2 * x + 1) * (y^3 + 4 * y^2 + 2 * y + 1) * (z^3 + 4 * z^2 + 2 * z + 1) / (x * y * z) ≥ 1331 :=
sorry

end NUMINAMATH_GPT_minimum_value_problem_l784_78453


namespace NUMINAMATH_GPT_geometric_sequence_product_correct_l784_78406

noncomputable def geometric_sequence_product (a_1 a_5 : ℝ) (a_2 a_3 a_4 : ℝ) :=
  a_1 = 1 / 2 ∧ a_5 = 8 ∧ a_2 * a_4 = a_1 * a_5 ∧ a_3^2 = a_1 * a_5

theorem geometric_sequence_product_correct:
  ∃ a_2 a_3 a_4 : ℝ, geometric_sequence_product (1 / 2) 8 a_2 a_3 a_4 ∧ (a_2 * a_3 * a_4 = 8) :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_product_correct_l784_78406


namespace NUMINAMATH_GPT_anton_food_cost_l784_78413

def food_cost_julie : ℝ := 10
def food_cost_letitia : ℝ := 20
def tip_per_person : ℝ := 4
def num_people : ℕ := 3
def tip_percentage : ℝ := 0.20

theorem anton_food_cost (A : ℝ) :
  tip_percentage * (food_cost_julie + food_cost_letitia + A) = tip_per_person * num_people →
  A = 30 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_anton_food_cost_l784_78413


namespace NUMINAMATH_GPT_number_of_possible_digits_to_make_divisible_by_4_l784_78469

def four_digit_number_divisible_by_4 (N : ℕ) : Prop :=
  let number := N * 1000 + 264
  number % 4 = 0

theorem number_of_possible_digits_to_make_divisible_by_4 :
  ∃ (count : ℕ), count = 10 ∧ (∀ (N : ℕ), N < 10 → four_digit_number_divisible_by_4 N) :=
by {
  sorry
}

end NUMINAMATH_GPT_number_of_possible_digits_to_make_divisible_by_4_l784_78469


namespace NUMINAMATH_GPT_percent_decrease_is_30_l784_78466

def original_price : ℝ := 100
def sale_price : ℝ := 70
def decrease_in_price : ℝ := original_price - sale_price

theorem percent_decrease_is_30 : (decrease_in_price / original_price) * 100 = 30 :=
by
  sorry

end NUMINAMATH_GPT_percent_decrease_is_30_l784_78466


namespace NUMINAMATH_GPT_find_total_amount_l784_78490

noncomputable def total_amount (A T yearly_income : ℝ) : Prop :=
  0.05 * A + 0.06 * (T - A) = yearly_income

theorem find_total_amount :
  ∃ T : ℝ, total_amount 1600 T 140 ∧ T = 2600 :=
sorry

end NUMINAMATH_GPT_find_total_amount_l784_78490


namespace NUMINAMATH_GPT_triangle_side_range_a_l784_78479

theorem triangle_side_range_a {a : ℝ} : 2 < a ∧ a < 5 ↔
  3 + (2 * a + 1) > 8 ∧ 
  8 - 3 < 2 * a + 1 ∧ 
  8 - (2 * a + 1) < 3 :=
by
  sorry

end NUMINAMATH_GPT_triangle_side_range_a_l784_78479


namespace NUMINAMATH_GPT_find_a_plus_b_l784_78409

theorem find_a_plus_b :
  let A := {x : ℝ | -1 < x ∧ x < 3}
  let B := {x : ℝ | -3 < x ∧ x < 2}
  let S := {x : ℝ | -1 < x ∧ x < 2}
  ∃ (a b : ℝ), (∀ x, S x ↔ (x^2 + a * x + b < 0)) ∧ a + b = -3 :=
by
  sorry

end NUMINAMATH_GPT_find_a_plus_b_l784_78409


namespace NUMINAMATH_GPT_find_angle_A_range_of_bc_l784_78459

-- Define the necessary conditions and prove the size of angle A
theorem find_angle_A 
  (a b c : ℝ)
  (A B C : ℝ)
  (h₁ : b * (Real.sin B + Real.sin C) = (a - c) * (Real.sin A + Real.sin C))
  (h₂ : B > Real.pi / 2)
  (h₃ : A + B + C = Real.pi)
  (h₄ : a > 0) (h₅ : b > 0) (h₆ : c > 0): 
  A = 2 * Real.pi / 3 :=
sorry

-- Define the necessary conditions and prove the range for b+c when a = sqrt(3)/2
theorem range_of_bc 
  (a b c : ℝ)
  (A : ℝ)
  (h₁ : A = 2 * Real.pi / 3)
  (h₂ : a = Real.sqrt 3 / 2)
  (h₃ : a > 0) (h₄ : b > 0) (h₅ : c > 0)
  (h₆ : A + B + C = Real.pi)
  (h₇ : B + C = Real.pi / 3) : 
  Real.sqrt 3 / 2 < b + c ∧ b + c ≤ 1 :=
sorry

end NUMINAMATH_GPT_find_angle_A_range_of_bc_l784_78459


namespace NUMINAMATH_GPT_range_of_real_number_a_l784_78457

theorem range_of_real_number_a (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 3 * x + 1 = 0 → x = a) ↔ (a = 0 ∨ a ≥ 9/4) :=
sorry

end NUMINAMATH_GPT_range_of_real_number_a_l784_78457


namespace NUMINAMATH_GPT_ratio_implies_sum_ratio_l784_78425

theorem ratio_implies_sum_ratio (x y : ℝ) (h : x / y = 3 / 4) : (x + y) / y = 7 / 4 :=
sorry

end NUMINAMATH_GPT_ratio_implies_sum_ratio_l784_78425


namespace NUMINAMATH_GPT_calculate_paintable_area_l784_78428

noncomputable def bedroom_length : ℝ := 15
noncomputable def bedroom_width : ℝ := 11
noncomputable def bedroom_height : ℝ := 9
noncomputable def door_window_area : ℝ := 70
noncomputable def num_bedrooms : ℝ := 3

theorem calculate_paintable_area :
  (num_bedrooms * ((2 * bedroom_length * bedroom_height) + (2 * bedroom_width * bedroom_height) - door_window_area)) = 1194 := 
by
  -- conditions as definitions
  let total_wall_area := (2 * bedroom_length * bedroom_height) + (2 * bedroom_width * bedroom_height)
  let paintable_wall_in_bedroom := total_wall_area - door_window_area
  let total_paintable_area := num_bedrooms * paintable_wall_in_bedroom
  show total_paintable_area = 1194
  sorry

end NUMINAMATH_GPT_calculate_paintable_area_l784_78428


namespace NUMINAMATH_GPT_complex_number_on_ray_is_specific_l784_78414

open Complex

theorem complex_number_on_ray_is_specific (a b : ℝ) (z : ℂ) (h₁ : z = a + b * I) 
  (h₂ : a = b) (h₃ : abs z = 1) : 
  z = (Real.sqrt 2 / 2) + (Real.sqrt 2 / 2) * I :=
by
  sorry

end NUMINAMATH_GPT_complex_number_on_ray_is_specific_l784_78414


namespace NUMINAMATH_GPT_compute_zeta_seventh_power_sum_l784_78472

noncomputable def complex_seventh_power_sum : Prop :=
  ∀ (ζ₁ ζ₂ ζ₃ : ℂ), 
    (ζ₁ + ζ₂ + ζ₃ = 1) ∧ 
    (ζ₁^2 + ζ₂^2 + ζ₃^2 = 3) ∧
    (ζ₁^3 + ζ₂^3 + ζ₃^3 = 7) →
    (ζ₁^7 + ζ₂^7 + ζ₃^7 = 71)

theorem compute_zeta_seventh_power_sum : complex_seventh_power_sum :=
by
  sorry

end NUMINAMATH_GPT_compute_zeta_seventh_power_sum_l784_78472


namespace NUMINAMATH_GPT_factorial_ratio_integer_l784_78493

theorem factorial_ratio_integer (m n : ℕ) : 
    (m ≥ 0) → (n ≥ 0) → ∃ k : ℤ, k = (2 * m).factorial * (2 * n).factorial / ((m.factorial * n.factorial * (m + n).factorial) : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_factorial_ratio_integer_l784_78493


namespace NUMINAMATH_GPT_value_of_x_l784_78426

theorem value_of_x (x : ℝ) (hx_pos : 0 < x) (hx_eq : x^2 = 1024) : x = 32 := 
by
  sorry

end NUMINAMATH_GPT_value_of_x_l784_78426


namespace NUMINAMATH_GPT_minimum_questionnaires_l784_78470

theorem minimum_questionnaires (p : ℝ) (r : ℝ) (n_min : ℕ) (h1 : p = 0.65) (h2 : r = 300) :
  n_min = ⌈r / p⌉ ∧ n_min = 462 := 
by
  sorry

end NUMINAMATH_GPT_minimum_questionnaires_l784_78470


namespace NUMINAMATH_GPT_find_median_of_first_twelve_positive_integers_l784_78492

def median_of_first_twelve_positive_integers : ℚ :=
  let A := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
  (A[5] + A[6]) / 2

theorem find_median_of_first_twelve_positive_integers :
  median_of_first_twelve_positive_integers = 6.5 :=
by
  sorry

end NUMINAMATH_GPT_find_median_of_first_twelve_positive_integers_l784_78492


namespace NUMINAMATH_GPT_smallest_number_l784_78405

-- Definitions of the numbers in their respective bases
def num1 := 5 * 9^0 + 8 * 9^1 -- 85_9
def num2 := 0 * 6^0 + 1 * 6^1 + 2 * 6^2 -- 210_6
def num3 := 0 * 4^0 + 0 * 4^1 + 0 * 4^2 + 1 * 4^3 -- 1000_4
def num4 := 1 * 2^0 + 1 * 2^1 + 1 * 2^2 + 1 * 2^3 + 1 * 2^4 + 1 * 2^5 -- 111111_2

-- Assert that num4 is the smallest
theorem smallest_number : num4 < num1 ∧ num4 < num2 ∧ num4 < num3 :=
by 
  sorry

end NUMINAMATH_GPT_smallest_number_l784_78405


namespace NUMINAMATH_GPT_correct_expression_l784_78465

theorem correct_expression (a b c : ℝ) : 3 * a - (2 * b - c) = 3 * a - 2 * b + c :=
sorry

end NUMINAMATH_GPT_correct_expression_l784_78465


namespace NUMINAMATH_GPT_cubic_yard_to_cubic_meter_l784_78460

theorem cubic_yard_to_cubic_meter : 
  let yard_to_foot := 3
  let foot_to_meter := 0.3048
  let side_length_in_meters := yard_to_foot * foot_to_meter
  (side_length_in_meters)^3 = 0.764554 :=
by
  sorry

end NUMINAMATH_GPT_cubic_yard_to_cubic_meter_l784_78460


namespace NUMINAMATH_GPT_students_without_pens_l784_78435

theorem students_without_pens (total_students blue_pens red_pens both_pens : ℕ)
  (h_total : total_students = 40)
  (h_blue : blue_pens = 18)
  (h_red : red_pens = 26)
  (h_both : both_pens = 10) :
  total_students - (blue_pens + red_pens - both_pens) = 6 :=
by
  sorry

end NUMINAMATH_GPT_students_without_pens_l784_78435


namespace NUMINAMATH_GPT_necessary_condition_range_l784_78461

variables {x m : ℝ}

def p (x : ℝ) : Prop := x^2 - x - 2 < 0
def q (x m : ℝ) : Prop := m ≤ x ∧ x ≤ m + 1

theorem necessary_condition_range (H : ∀ x, q x m → p x) : -1 < m ∧ m < 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_necessary_condition_range_l784_78461


namespace NUMINAMATH_GPT_calculate_Y_payment_l784_78440

theorem calculate_Y_payment (X Y : ℝ) (h1 : X + Y = 600) (h2 : X = 1.2 * Y) : Y = 600 / 2.2 :=
by
  sorry

end NUMINAMATH_GPT_calculate_Y_payment_l784_78440


namespace NUMINAMATH_GPT_find_y_l784_78498

theorem find_y (y : ℝ) (a b : ℝ × ℝ) (h_a : a = (4, 2)) (h_b : b = (6, y)) (h_parallel : 4 * y - 2 * 6 = 0) :
  y = 3 :=
sorry

end NUMINAMATH_GPT_find_y_l784_78498


namespace NUMINAMATH_GPT_geometric_sequence_a7_l784_78475

variable {a : ℕ → ℝ}
variable {r : ℝ}

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n * r

-- Given condition
axiom geom_seq_condition : a 4 * a 10 = 9

-- proving the required result
theorem geometric_sequence_a7 (h : is_geometric_sequence a r) : a 7 = 3 ∨ a 7 = -3 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_a7_l784_78475


namespace NUMINAMATH_GPT_no_four_distinct_real_roots_l784_78419

theorem no_four_distinct_real_roots (a b : ℝ) : ¬ (∃ (x1 x2 x3 x4 : ℝ), 
  x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4 ∧
  (x1^4 - 4*x1^3 + 6*x1^2 + a*x1 + b = 0) ∧ 
  (x2^4 - 4*x2^3 + 6*x2^2 + a*x2 + b = 0) ∧ 
  (x3^4 - 4*x3^3 + 6*x3^2 + a*x3 + b = 0) ∧ 
  (x4^4 - 4*x4^3 + 6*x4^2 + a*x4 + b = 0)) :=
by
  sorry

end NUMINAMATH_GPT_no_four_distinct_real_roots_l784_78419


namespace NUMINAMATH_GPT_roots_difference_l784_78451

theorem roots_difference :
  let a := 2 
  let b := 5 
  let c := -12
  let disc := b*b - 4*a*c
  let root1 := (-b + Real.sqrt disc) / (2 * a)
  let root2 := (-b - Real.sqrt disc) / (2 * a)
  let larger_root := max root1 root2
  let smaller_root := min root1 root2
  larger_root - smaller_root = 5.5 := by
  sorry

end NUMINAMATH_GPT_roots_difference_l784_78451


namespace NUMINAMATH_GPT_gcd_of_polynomial_and_linear_l784_78447

theorem gcd_of_polynomial_and_linear (b : ℤ) (h1 : b % 2 = 1) (h2 : 1019 ∣ b) : 
  Int.gcd (3 * b ^ 2 + 31 * b + 91) (b + 15) = 1 := 
by 
  sorry

end NUMINAMATH_GPT_gcd_of_polynomial_and_linear_l784_78447


namespace NUMINAMATH_GPT_algebraic_expression_value_l784_78410

theorem algebraic_expression_value 
  (θ : ℝ)
  (a := (Real.cos θ, Real.sin θ))
  (b := (1, -2))
  (parallel : ∃ k : ℝ, a = (k * 1, k * -2)) :
  (2 * Real.sin θ - Real.cos θ) / (Real.sin θ + Real.cos θ) = 5 := 
by 
  -- proof goes here 
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l784_78410


namespace NUMINAMATH_GPT_find_a_l784_78429

theorem find_a (a b d : ℕ) (h1 : a + b = d) (h2 : b + d = 7) (h3 : d = 4) : a = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l784_78429


namespace NUMINAMATH_GPT_second_alloy_amount_l784_78499

theorem second_alloy_amount (x : ℝ) :
  let chromium_first_alloy := 0.12 * 15
  let chromium_second_alloy := 0.08 * x
  let total_weight := 15 + x
  let chromium_percentage_new_alloy := (0.12 * 15 + 0.08 * x) / (15 + x)
  chromium_percentage_new_alloy = (28 / 300) →
  x = 30 := sorry

end NUMINAMATH_GPT_second_alloy_amount_l784_78499


namespace NUMINAMATH_GPT_negate_prop_l784_78400

theorem negate_prop :
  ¬ (∀ x : ℝ, x > 1 → x - 1 > Real.log x) ↔ ∃ x : ℝ, x > 1 ∧ x - 1 ≤ Real.log x :=
by
  sorry

end NUMINAMATH_GPT_negate_prop_l784_78400


namespace NUMINAMATH_GPT_max_parrots_l784_78478

-- Define the parameters and conditions for the problem
def N : ℕ := 2018
def Y : ℕ := 1009
def number_of_islanders (R L P : ℕ) := R + L + P = N

-- Define the main theorem
theorem max_parrots (R L P : ℕ) (h : number_of_islanders R L P) (hY : Y = 1009) :
  P = 1009 :=
sorry

end NUMINAMATH_GPT_max_parrots_l784_78478


namespace NUMINAMATH_GPT_analogical_reasoning_ineq_l784_78420

-- Formalization of the conditions and the theorem to be proved

def positive (a : ℕ → ℝ) (n : ℕ) := ∀ i, 1 ≤ i → i ≤ n → a i > 0

theorem analogical_reasoning_ineq {a : ℕ → ℝ} (hpos : positive a 4) (hsum : a 1 + a 2 + a 3 + a 4 = 1) : 
  (1 / a 1 + 1 / a 2 + 1 / a 3 + 1 / a 4) ≥ 16 := 
sorry

end NUMINAMATH_GPT_analogical_reasoning_ineq_l784_78420


namespace NUMINAMATH_GPT_cars_travel_same_distance_l784_78496

-- Define all the variables and conditions
def TimeR : ℝ := sorry -- the time taken by car R
def TimeP : ℝ := TimeR - 2
def SpeedR : ℝ := 58.4428877022476
def SpeedP : ℝ := SpeedR + 10

-- state the distance travelled by both cars
def DistanceR : ℝ := SpeedR * TimeR
def DistanceP : ℝ := SpeedP * TimeP

-- Prove that both distances are the same and equal to 800
theorem cars_travel_same_distance : DistanceR = 800 := by
  sorry

end NUMINAMATH_GPT_cars_travel_same_distance_l784_78496


namespace NUMINAMATH_GPT_find_k_l784_78454

theorem find_k (k : ℝ) : 
  (1 / 2) * |k| * |k / 2| = 4 → (k = 4 ∨ k = -4) := 
sorry

end NUMINAMATH_GPT_find_k_l784_78454


namespace NUMINAMATH_GPT_avg_price_six_toys_l784_78416

def avg_price_five_toys : ℝ := 10
def price_sixth_toy : ℝ := 16
def total_toys : ℕ := 5 + 1

theorem avg_price_six_toys (avg_price_five_toys price_sixth_toy : ℝ) (total_toys : ℕ) :
  (avg_price_five_toys * 5 + price_sixth_toy) / total_toys = 11 := by
  sorry

end NUMINAMATH_GPT_avg_price_six_toys_l784_78416


namespace NUMINAMATH_GPT_find_smallest_m_l784_78474

def is_in_S (z : ℂ) : Prop :=
  ∃ (x y : ℝ), ((1 / 2 : ℝ) ≤ x) ∧ (x ≤ Real.sqrt 2 / 2) ∧ (z = (x : ℂ) + (y : ℂ) * Complex.I)

def is_nth_root_of_unity (z : ℂ) (n : ℕ) : Prop :=
  z ^ n = 1

def smallest_m (m : ℕ) : Prop :=
  ∀ n : ℕ, n ≥ m → ∃ z : ℂ, is_in_S z ∧ is_nth_root_of_unity z n

theorem find_smallest_m : smallest_m 24 :=
  sorry

end NUMINAMATH_GPT_find_smallest_m_l784_78474


namespace NUMINAMATH_GPT_inequality_problem_l784_78434

open Real

theorem inequality_problem
  (a b c x y z : ℝ)
  (hx_pos : 0 < x) (hy_pos : 0 < y) (hz_pos : 0 < z)
  (ha_pos : 0 < a) (hb_pos : 0 < b) (hc_pos : 0 < c)
  (h_condition : 1 / x + 1 / y + 1 / z = 1) :
  a^x + b^y + c^z ≥ 4 * a * b * c * x * y * z / (x + y + z - 3) ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_problem_l784_78434


namespace NUMINAMATH_GPT_curve_touches_all_Ca_l784_78456

theorem curve_touches_all_Ca (a : ℝ) (h : a > 0) : ∃ C : ℝ → ℝ, ∀ x y, (y - a^2)^2 = x^2 * (a^2 - x^2) → y = C x ∧ C x = 3 * x^2 / 4 :=
sorry

end NUMINAMATH_GPT_curve_touches_all_Ca_l784_78456


namespace NUMINAMATH_GPT_current_speed_correct_l784_78436

noncomputable def boat_upstream_speed : ℝ := (1 / 20) * 60
noncomputable def boat_downstream_speed : ℝ := (1 / 9) * 60
noncomputable def speed_of_current : ℝ := (boat_downstream_speed - boat_upstream_speed) / 2

theorem current_speed_correct :
  speed_of_current = 1.835 :=
by
  sorry

end NUMINAMATH_GPT_current_speed_correct_l784_78436


namespace NUMINAMATH_GPT_train_length_l784_78430

theorem train_length (speed_km_hr : ℝ) (time_seconds : ℝ) (speed_ms : ℝ) (distance_m : ℝ)
  (h1 : speed_km_hr = 90)
  (h2 : time_seconds = 9)
  (h3 : speed_ms = speed_km_hr * (1000 / 3600))
  (h4 : distance_m = speed_ms * time_seconds) :
  distance_m = 225 :=
by
  sorry

end NUMINAMATH_GPT_train_length_l784_78430


namespace NUMINAMATH_GPT_equation_solution_unique_l784_78473

theorem equation_solution_unique (m a b : ℕ) (hm : 1 < m) (ha : 1 < a) (hb : 1 < b) :
  ((m + 1) * a = m * b + 1) ↔ m = 2 :=
sorry

end NUMINAMATH_GPT_equation_solution_unique_l784_78473


namespace NUMINAMATH_GPT_concentration_of_first_solution_l784_78487

theorem concentration_of_first_solution
  (C : ℝ)
  (h : 4 * (C / 100) + 0.2 = 0.36) :
  C = 4 :=
by
  sorry

end NUMINAMATH_GPT_concentration_of_first_solution_l784_78487


namespace NUMINAMATH_GPT_actual_distance_between_towns_l784_78422

def map_distance := 20 -- distance between towns on the map in inches
def scale := 10 -- scale: 1 inch = 10 miles

theorem actual_distance_between_towns : map_distance * scale = 200 := by
  sorry

end NUMINAMATH_GPT_actual_distance_between_towns_l784_78422


namespace NUMINAMATH_GPT_log_sum_reciprocals_of_logs_l784_78411

-- Problem (1)
theorem log_sum (log_two : Real.log 2 ≠ 0) :
    Real.log 4 / Real.log 10 + Real.log 50 / Real.log 10 - Real.log 2 / Real.log 10 = 2 := by
  sorry

-- Problem (2)
theorem reciprocals_of_logs (a b : Real) (h : 1 + Real.log a / Real.log 2 = 2 + Real.log b / Real.log 3 ∧ (1 + Real.log a / Real.log 2) = Real.log (a + b) / Real.log 6) : 
    1 / a + 1 / b = 6 := by
  sorry

end NUMINAMATH_GPT_log_sum_reciprocals_of_logs_l784_78411


namespace NUMINAMATH_GPT_lemonade_total_difference_is_1860_l784_78467

-- Define the conditions
def stanley_rate : Nat := 4
def stanley_price : Real := 1.50

def carl_rate : Nat := 7
def carl_price : Real := 1.30

def lucy_rate : Nat := 5
def lucy_price : Real := 1.80

def hours : Nat := 3

-- Compute the total amounts for each sibling
def stanley_total : Real := stanley_rate * hours * stanley_price
def carl_total : Real := carl_rate * hours * carl_price
def lucy_total : Real := lucy_rate * hours * lucy_price

-- Compute the individual differences
def diff_stanley_carl : Real := carl_total - stanley_total
def diff_stanley_lucy : Real := lucy_total - stanley_total
def diff_carl_lucy : Real := carl_total - lucy_total

-- Sum the differences
def total_difference : Real := diff_stanley_carl + diff_stanley_lucy + diff_carl_lucy

-- The proof statement
theorem lemonade_total_difference_is_1860 :
  total_difference = 18.60 :=
by
  sorry

end NUMINAMATH_GPT_lemonade_total_difference_is_1860_l784_78467


namespace NUMINAMATH_GPT_problem1_problem2_l784_78481

open Real

noncomputable def f (x a : ℝ) : ℝ := |2 * x + 3| - |2 * x - a|

-- Problem (1)
theorem problem1 {a : ℝ} (h : ∃ x, f x a ≤ -5) : a ≤ -8 ∨ a ≥ 2 :=
sorry

-- Problem (2)
theorem problem2 {a : ℝ} (h : ∀ x, f (x - 1/2) a + f (-x - 1/2) a = 0) : a = 1 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l784_78481


namespace NUMINAMATH_GPT_range_of_sum_l784_78412

theorem range_of_sum (a b : ℝ) (h : a^2 - a * b + b^2 = a + b) :
  0 ≤ a + b ∧ a + b ≤ 4 :=
by
  sorry

end NUMINAMATH_GPT_range_of_sum_l784_78412


namespace NUMINAMATH_GPT_at_least_one_female_team_l784_78421

open Classical

namespace Probability

-- Define the Problem
noncomputable def prob_at_least_one_female (females males : ℕ) (team_size : ℕ) :=
  let total_students := females + males
  let total_ways := Nat.choose total_students team_size
  let ways_all_males := Nat.choose males team_size
  1 - (ways_all_males / total_ways : ℝ)

-- Verify the given problem against the expected answer
theorem at_least_one_female_team :
  prob_at_least_one_female 1 3 2 = 1 / 2 := by
  sorry

end Probability

end NUMINAMATH_GPT_at_least_one_female_team_l784_78421


namespace NUMINAMATH_GPT_tensor_calculation_jiaqi_statement_l784_78452

def my_tensor (a b : ℝ) : ℝ := a * (1 - b)

theorem tensor_calculation :
  my_tensor (1 + Real.sqrt 2) (Real.sqrt 2) = -1 := 
by
  sorry

theorem jiaqi_statement (a b : ℝ) (h : a + b = 0) :
  my_tensor a a + my_tensor b b = 2 * a * b := 
by
  sorry

end NUMINAMATH_GPT_tensor_calculation_jiaqi_statement_l784_78452


namespace NUMINAMATH_GPT_negation_of_proposition_l784_78417

-- Definitions of the conditions
variables (a b c : ℝ) 

-- Prove the mathematically equivalent statement:
theorem negation_of_proposition :
  (a + b + c ≠ 1) → (a^2 + b^2 + c^2 > 1 / 9) :=
sorry

end NUMINAMATH_GPT_negation_of_proposition_l784_78417


namespace NUMINAMATH_GPT_length_is_62_l784_78423

noncomputable def length_of_plot (b : ℝ) := b + 24

theorem length_is_62 (b : ℝ) (h1 : length_of_plot b = b + 24) 
  (h2 : 2 * (length_of_plot b + b) = 200) : 
  length_of_plot b = 62 :=
by sorry

end NUMINAMATH_GPT_length_is_62_l784_78423


namespace NUMINAMATH_GPT_initial_capacity_of_bottle_l784_78482

theorem initial_capacity_of_bottle 
  (C : ℝ)
  (h1 : 1/3 * 3/4 * C = 1) : 
  C = 4 :=
by
  sorry

end NUMINAMATH_GPT_initial_capacity_of_bottle_l784_78482


namespace NUMINAMATH_GPT_origin_inside_circle_range_l784_78445

theorem origin_inside_circle_range (m : ℝ) :
  ((0 - m)^2 + (0 + m)^2 < 8) → (-2 < m ∧ m < 2) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_origin_inside_circle_range_l784_78445


namespace NUMINAMATH_GPT_minimum_value_of_x_plus_y_l784_78485

-- Define the conditions as a hypothesis and the goal theorem statement.
theorem minimum_value_of_x_plus_y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 1 / x + 9 / y = 1) :
  x + y = 16 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_x_plus_y_l784_78485


namespace NUMINAMATH_GPT_number_of_shelves_l784_78424

-- Define the initial conditions and required values
def initial_bears : ℕ := 6
def shipment_bears : ℕ := 18
def bears_per_shelf : ℕ := 6

-- Define the result we want to prove
theorem number_of_shelves : (initial_bears + shipment_bears) / bears_per_shelf = 4 :=
by
    -- Proof steps go here
    sorry

end NUMINAMATH_GPT_number_of_shelves_l784_78424


namespace NUMINAMATH_GPT_combined_salaries_of_A_B_C_E_is_correct_l784_78450

-- Given conditions
def D_salary : ℕ := 7000
def average_salary : ℕ := 8800
def n_individuals : ℕ := 5

-- Combined salary of A, B, C, and E
def combined_salaries : ℕ := 37000

theorem combined_salaries_of_A_B_C_E_is_correct :
  (average_salary * n_individuals - D_salary) = combined_salaries :=
by
  sorry

end NUMINAMATH_GPT_combined_salaries_of_A_B_C_E_is_correct_l784_78450


namespace NUMINAMATH_GPT_cupcake_ratio_l784_78494

theorem cupcake_ratio (C B : ℕ) (hC : C = 4) (hTotal : C + B = 12) : B / C = 2 :=
by
  sorry

end NUMINAMATH_GPT_cupcake_ratio_l784_78494


namespace NUMINAMATH_GPT_total_pens_l784_78404

theorem total_pens (r : ℕ) (h1 : r > 10)
  (h2 : 357 % r = 0)
  (h3 : 441 % r = 0) :
  357 / r + 441 / r = 38 :=
by
  sorry

end NUMINAMATH_GPT_total_pens_l784_78404


namespace NUMINAMATH_GPT_Rachel_total_score_l784_78464

theorem Rachel_total_score
    (points_per_treasure : ℕ)
    (treasures_first_level : ℕ)
    (treasures_second_level : ℕ)
    (h1 : points_per_treasure = 9)
    (h2 : treasures_first_level = 5)
    (h3 : treasures_second_level = 2) : 
    (points_per_treasure * treasures_first_level + points_per_treasure * treasures_second_level = 63) :=
by
    sorry

end NUMINAMATH_GPT_Rachel_total_score_l784_78464


namespace NUMINAMATH_GPT_circle_range_k_l784_78486

theorem circle_range_k (k : ℝ) : (∀ x y : ℝ, x^2 + y^2 - 4 * x + 4 * y + 10 - k = 0) → k > 2 :=
by
  sorry

end NUMINAMATH_GPT_circle_range_k_l784_78486


namespace NUMINAMATH_GPT_smallest_d_l784_78448

theorem smallest_d (d : ℕ) (h_pos : 0 < d) (h_square : ∃ k : ℕ, 3150 * d = k^2) : d = 14 :=
sorry

end NUMINAMATH_GPT_smallest_d_l784_78448


namespace NUMINAMATH_GPT_opposite_of_2023_l784_78462

theorem opposite_of_2023 : 2023 + (-2023) = 0 :=
by
  sorry

end NUMINAMATH_GPT_opposite_of_2023_l784_78462
