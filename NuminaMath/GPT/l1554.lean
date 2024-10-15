import Mathlib

namespace NUMINAMATH_GPT_distilled_water_required_l1554_155491

theorem distilled_water_required :
  ∀ (nutrient_concentrate distilled_water : ℝ) (total_solution prep_solution : ℝ), 
    nutrient_concentrate = 0.05 →
    distilled_water = 0.025 →
    total_solution = 0.075 → 
    prep_solution = 0.6 →
    (prep_solution * (distilled_water / total_solution)) = 0.2 :=
by
  intros nutrient_concentrate distilled_water total_solution prep_solution
  sorry

end NUMINAMATH_GPT_distilled_water_required_l1554_155491


namespace NUMINAMATH_GPT_parallel_lines_l1554_155460

noncomputable def line1 (x y : ℝ) : Prop := x - y + 1 = 0
noncomputable def line2 (a x y : ℝ) : Prop := x + a * y + 3 = 0

theorem parallel_lines (a x y : ℝ) : (∀ (x y : ℝ), line1 x y → line2 a x y → x = y ∨ (line1 x y ∧ x ≠ y)) → 
  (a = -1 ∧ ∃ d : ℝ, d = Real.sqrt 2) :=
sorry

end NUMINAMATH_GPT_parallel_lines_l1554_155460


namespace NUMINAMATH_GPT_find_prime_p_l1554_155477

open Int

theorem find_prime_p (p k m n : ℕ) (hp : Nat.Prime p) 
  (hk : 0 < k) (hm : 0 < m)
  (h_eq : (mk^2 + 2 : ℤ) * p - (m^2 + 2 * k^2 : ℤ) = n^2 * (mp + 2 : ℤ)) :
  p = 3 ∨ p = 1 := sorry

end NUMINAMATH_GPT_find_prime_p_l1554_155477


namespace NUMINAMATH_GPT_perpendicular_lines_a_l1554_155489

theorem perpendicular_lines_a {a : ℝ} :
  ((∀ x y : ℝ, (2 * a - 1) * x + a * y + a = 0) → (∀ x y : ℝ, a * x - y + 2 * a = 0) → a = 0 ∨ a = 1) :=
by
  intro h₁ h₂
  sorry

end NUMINAMATH_GPT_perpendicular_lines_a_l1554_155489


namespace NUMINAMATH_GPT_Maaza_liters_l1554_155443

theorem Maaza_liters 
  (M L : ℕ)
  (Pepsi : ℕ := 144)
  (Sprite : ℕ := 368)
  (total_liters := M + Pepsi + Sprite)
  (cans_required : ℕ := 281)
  (H : total_liters = cans_required * L)
  : M = 50 :=
by
  sorry

end NUMINAMATH_GPT_Maaza_liters_l1554_155443


namespace NUMINAMATH_GPT_alice_wrong_questions_l1554_155449

theorem alice_wrong_questions :
  ∃ a b e : ℕ,
    (a + b = 6 + 8 + e) ∧
    (a + 8 = b + 6 + 3) ∧
    a = 9 :=
by {
  sorry
}

end NUMINAMATH_GPT_alice_wrong_questions_l1554_155449


namespace NUMINAMATH_GPT_inequality_ac2_geq_bc2_l1554_155413

theorem inequality_ac2_geq_bc2 (a b c : ℝ) (h : a > b) : a * c^2 ≥ b * c^2 :=
sorry

end NUMINAMATH_GPT_inequality_ac2_geq_bc2_l1554_155413


namespace NUMINAMATH_GPT_age_relationships_l1554_155430

variables (a b c d : ℕ)

theorem age_relationships (h1 : a + b = b + c + d + 18) (h2 : 2 * a = 3 * c) :
  c = 2 * a / 3 ∧ d = a / 3 - 18 :=
by
  sorry

end NUMINAMATH_GPT_age_relationships_l1554_155430


namespace NUMINAMATH_GPT_degree_g_is_six_l1554_155458

theorem degree_g_is_six 
  (f g : Polynomial ℂ) 
  (h : Polynomial ℂ) 
  (h_def : h = f.comp g + Polynomial.X * g) 
  (deg_h : h.degree = 7) 
  (deg_f : f.degree = 3) 
  : g.degree = 6 := 
sorry

end NUMINAMATH_GPT_degree_g_is_six_l1554_155458


namespace NUMINAMATH_GPT_combined_probability_l1554_155448

-- Definitions:
def number_of_ways_to_get_3_heads_and_1_tail := Nat.choose 4 3
def probability_of_specific_sequence_of_3_heads_and_1_tail := (1/2) ^ 4
def probability_of_3_heads_and_1_tail := number_of_ways_to_get_3_heads_and_1_tail * probability_of_specific_sequence_of_3_heads_and_1_tail

def favorable_outcomes_die := 2
def total_outcomes_die := 6
def probability_of_number_greater_than_4 := favorable_outcomes_die / total_outcomes_die

-- Proof statement:
theorem combined_probability : probability_of_3_heads_and_1_tail * probability_of_number_greater_than_4 = 1/12 := by
  sorry

end NUMINAMATH_GPT_combined_probability_l1554_155448


namespace NUMINAMATH_GPT_dice_sum_prime_probability_l1554_155473

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

noncomputable def roll_dice_prob_prime : ℚ :=
  let total_outcomes := 6^7
  let prime_sums := [7, 11, 13, 17, 19, 23, 29, 31, 37, 41]
  let P := 80425 -- Assume pre-computed sum counts based on primes
  (P : ℚ) / total_outcomes

theorem dice_sum_prime_probability :
  roll_dice_prob_prime = 26875 / 93312 :=
by
  sorry

end NUMINAMATH_GPT_dice_sum_prime_probability_l1554_155473


namespace NUMINAMATH_GPT_total_trolls_l1554_155462

noncomputable def troll_count (forest bridge plains : ℕ) : ℕ := forest + bridge + plains

theorem total_trolls (forest_trolls bridge_trolls plains_trolls : ℕ)
  (h1 : forest_trolls = 6)
  (h2 : bridge_trolls = 4 * forest_trolls - 6)
  (h3 : plains_trolls = bridge_trolls / 2) :
  troll_count forest_trolls bridge_trolls plains_trolls = 33 :=
by
  -- Proof steps would be filled in here
  sorry

end NUMINAMATH_GPT_total_trolls_l1554_155462


namespace NUMINAMATH_GPT_tire_circumference_l1554_155454

/-- 
Given:
1. The tire rotates at 400 revolutions per minute.
2. The car is traveling at a speed of 168 km/h.

Prove that the circumference of the tire is 7 meters.
-/
theorem tire_circumference (rpm : ℕ) (speed_km_h : ℕ) (C : ℕ) 
  (h1 : rpm = 400) 
  (h2 : speed_km_h = 168)
  (h3 : C = 7) : 
  C = (speed_km_h * 1000 / 60) / rpm :=
by
  rw [h1, h2]
  exact h3

end NUMINAMATH_GPT_tire_circumference_l1554_155454


namespace NUMINAMATH_GPT_percentage_decrease_stock_l1554_155433

theorem percentage_decrease_stock (F J M : ℝ)
  (h1 : J = F - 0.10 * F)
  (h2 : M = J - 0.20 * J) :
  (F - M) / F * 100 = 28 := by
sorry

end NUMINAMATH_GPT_percentage_decrease_stock_l1554_155433


namespace NUMINAMATH_GPT_problem_statement_l1554_155418

theorem problem_statement (x : ℝ) (h : (x / 5) / 3 = 5 / (x / 3)) : x = 15 ∨ x = -15 := 
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1554_155418


namespace NUMINAMATH_GPT_bacteria_growth_l1554_155419

-- Defining the function for bacteria growth
def bacteria_count (t : ℕ) (initial_count : ℕ) (division_time : ℕ) : ℕ :=
  initial_count * 2 ^ (t / division_time)

-- The initial conditions given in the problem
def initial_bacteria : ℕ := 1
def division_interval : ℕ := 10
def total_time : ℕ := 2 * 60

-- Stating the hypothesis and the goal
theorem bacteria_growth : bacteria_count total_time initial_bacteria division_interval = 2 ^ 12 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_bacteria_growth_l1554_155419


namespace NUMINAMATH_GPT_largest_square_side_l1554_155427

variable (length width : ℕ)
variable (h_length : length = 54)
variable (h_width : width = 20)
variable (num_squares : ℕ)
variable (h_num_squares : num_squares = 3)

theorem largest_square_side : (length : ℝ) / num_squares = 18 := by
  sorry

end NUMINAMATH_GPT_largest_square_side_l1554_155427


namespace NUMINAMATH_GPT_rancher_unique_solution_l1554_155463

-- Defining the main problem statement
theorem rancher_unique_solution : ∃! (b h : ℕ), 30 * b + 32 * h = 1200 ∧ b > h := by
  sorry

end NUMINAMATH_GPT_rancher_unique_solution_l1554_155463


namespace NUMINAMATH_GPT_increase_in_value_l1554_155499

-- Define the conditions
def starting_weight : ℝ := 400
def weight_multiplier : ℝ := 1.5
def price_per_pound : ℝ := 3

-- Define new weight and values
def new_weight : ℝ := starting_weight * weight_multiplier
def value_at_starting_weight : ℝ := starting_weight * price_per_pound
def value_at_new_weight : ℝ := new_weight * price_per_pound

-- Theorem to prove
theorem increase_in_value : value_at_new_weight - value_at_starting_weight = 600 := by
  sorry

end NUMINAMATH_GPT_increase_in_value_l1554_155499


namespace NUMINAMATH_GPT_bob_plate_price_correct_l1554_155450

-- Assuming units and specific values for the problem
def anne_plate_area : ℕ := 20 -- in square units
def bob_clay_usage : ℕ := 600 -- total clay used by Bob in square units
def bob_number_of_plates : ℕ := 15
def anne_plate_price : ℕ := 50 -- in cents
def anne_number_of_plates : ℕ := 30
def total_anne_earnings : ℕ := anne_number_of_plates * anne_plate_price

-- Condition
def bob_plate_area : ℕ := bob_clay_usage / bob_number_of_plates

-- Prove the price of one of Bob's plates
theorem bob_plate_price_correct : bob_number_of_plates * bob_plate_area = bob_clay_usage →
                                  bob_number_of_plates * 100 = total_anne_earnings :=
by
  intros 
  sorry

end NUMINAMATH_GPT_bob_plate_price_correct_l1554_155450


namespace NUMINAMATH_GPT_line_circle_separation_l1554_155457

theorem line_circle_separation (a b : ℝ) (h : a^2 + b^2 < 1) :
    let d := 1 / (Real.sqrt (a^2 + b^2))
    d > 1 := by
    sorry

end NUMINAMATH_GPT_line_circle_separation_l1554_155457


namespace NUMINAMATH_GPT_lowest_possible_score_l1554_155461

def total_points_first_four_tests : ℕ := 82 + 90 + 78 + 85
def required_total_points_for_seven_tests : ℕ := 80 * 7
def points_needed_for_last_three_tests : ℕ :=
  required_total_points_for_seven_tests - total_points_first_four_tests

theorem lowest_possible_score 
  (max_points_per_test : ℕ)
  (points_first_four_tests : ℕ := total_points_first_four_tests)
  (required_points : ℕ := required_total_points_for_seven_tests)
  (total_points_needed_last_three : ℕ := points_needed_for_last_three_tests) :
  ∃ (lowest_score : ℕ), 
    max_points_per_test = 100 ∧
    points_first_four_tests = 335 ∧
    required_points = 560 ∧
    total_points_needed_last_three = 225 ∧
    lowest_score = 25 :=
by
  sorry

end NUMINAMATH_GPT_lowest_possible_score_l1554_155461


namespace NUMINAMATH_GPT_initial_students_count_l1554_155444

theorem initial_students_count (N T : ℕ) (h1 : T = N * 90) (h2 : (T - 120) / (N - 3) = 95) : N = 33 :=
by
  sorry

end NUMINAMATH_GPT_initial_students_count_l1554_155444


namespace NUMINAMATH_GPT_candy_cost_l1554_155442

theorem candy_cost (C : ℝ) 
  (h1 : 20 * C + 80 * 5 = 100 * 6) : 
  C = 10 := 
by
  sorry

end NUMINAMATH_GPT_candy_cost_l1554_155442


namespace NUMINAMATH_GPT_seating_arrangement_l1554_155415

theorem seating_arrangement (students : ℕ) (desks : ℕ) (empty_desks : ℕ) 
  (h_students : students = 2) (h_desks : desks = 5) 
  (h_empty : empty_desks ≥ 1) :
  ∃ ways, ways = 12 := by
  sorry

end NUMINAMATH_GPT_seating_arrangement_l1554_155415


namespace NUMINAMATH_GPT_part1_solution_part2_solution_l1554_155496

def f (x : ℝ) (a : ℝ) := |x + 1| - |a * x - 1|

-- Statement for part 1
theorem part1_solution (x : ℝ) : (f x 1 > 1) ↔ (x > 1 / 2) := sorry

-- Statement for part 2
theorem part2_solution (a : ℝ) (x : ℝ) (h : 0 < x ∧ x < 1) : 
  (f x a > x) ↔ (0 < a ∧ a ≤ 2) := sorry

end NUMINAMATH_GPT_part1_solution_part2_solution_l1554_155496


namespace NUMINAMATH_GPT_math_problem_l1554_155451

variables (a b c d m : ℤ)

theorem math_problem (h1 : a = -b) (h2 : c * d = 1) (h3 : m = -1) : c * d - a - b + m^2022 = 2 :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l1554_155451


namespace NUMINAMATH_GPT_B_finishes_work_in_54_days_l1554_155481

-- The problem statement rewritten in Lean 4.
theorem B_finishes_work_in_54_days
  (A_eff : ℕ) -- amount of work A can do in one day
  (B_eff : ℕ) -- amount of work B can do in one day
  (work_days_together : ℕ) -- number of days A and B work together to finish the work
  (h1 : A_eff = 2 * B_eff)
  (h2 : A_eff + B_eff = 3)
  (h3 : work_days_together = 18) :
  work_days_together * (A_eff + B_eff) / B_eff = 54 :=
by
  sorry

end NUMINAMATH_GPT_B_finishes_work_in_54_days_l1554_155481


namespace NUMINAMATH_GPT_fraction_comparison_l1554_155466

theorem fraction_comparison (a b c d : ℝ) (h1 : a / b < c / d) (h2 : b > d) (h3 : d > 0) :
  (a + c) / (b + d) < 1/2 * (a / b + c / d) :=
by
  sorry

end NUMINAMATH_GPT_fraction_comparison_l1554_155466


namespace NUMINAMATH_GPT_pandas_bamboo_consumption_l1554_155486

def small_pandas : ℕ := 4
def big_pandas : ℕ := 5
def daily_bamboo_small : ℕ := 25
def daily_bamboo_big : ℕ := 40
def days_in_week : ℕ := 7

theorem pandas_bamboo_consumption : 
  (small_pandas * daily_bamboo_small + big_pandas * daily_bamboo_big) * days_in_week = 2100 := by
  sorry

end NUMINAMATH_GPT_pandas_bamboo_consumption_l1554_155486


namespace NUMINAMATH_GPT_triangle_angle_ge_60_l1554_155424

theorem triangle_angle_ge_60 {A B C : ℝ} (h : A + B + C = 180) :
  A < 60 ∧ B < 60 ∧ C < 60 → false :=
by
  sorry

end NUMINAMATH_GPT_triangle_angle_ge_60_l1554_155424


namespace NUMINAMATH_GPT_jack_second_half_time_l1554_155465

def time_jack_first_half := 19
def time_between_jill_and_jack := 7
def time_jill := 32

def time_jack (time_jill time_between_jill_and_jack : ℕ) : ℕ :=
  time_jill - time_between_jill_and_jack

def time_jack_second_half (time_jack time_jack_first_half : ℕ) : ℕ :=
  time_jack - time_jack_first_half

theorem jack_second_half_time :
  time_jack_second_half (time_jack time_jill time_between_jill_and_jack) time_jack_first_half = 6 :=
by
  sorry

end NUMINAMATH_GPT_jack_second_half_time_l1554_155465


namespace NUMINAMATH_GPT_teta_beta_gamma_l1554_155470

theorem teta_beta_gamma : 
  ∃ T E T' A B E' T'' A' G A'' M M' A''' A'''' : ℕ, 
  TETA = T * 1000 + E * 100 + T' * 10 + A ∧ 
  BETA = B * 1000 + E' * 100 + T'' * 10 + A' ∧ 
  GAMMA = G * 10000 + A'' * 1000 + M * 100 + M' * 10 + A''' ∧
  TETA + BETA = GAMMA ∧ 
  A = A'''' ∧ E = E' ∧ T = T' ∧ T' = T'' ∧ A = A' ∧ A = A'' ∧ A = A''' ∧ M = M' ∧ 
  T ≠ E ∧ T ≠ A ∧ T ≠ B ∧ T ≠ G ∧ T ≠ M ∧
  E ≠ A ∧ E ≠ B ∧ E ≠ G ∧ E ≠ M ∧
  A ≠ B ∧ A ≠ G ∧ A ≠ M ∧
  B ≠ G ∧ B ≠ M ∧
  G ≠ M ∧
  TETA = 4940 ∧ BETA = 5940 ∧ GAMMA = 10880
  :=
sorry

end NUMINAMATH_GPT_teta_beta_gamma_l1554_155470


namespace NUMINAMATH_GPT_original_number_of_men_l1554_155484

/-- 
Given:
1. A group of men decided to do a work in 20 days,
2. When 2 men became absent, the remaining men did the work in 22 days,

Prove:
The original number of men in the group was 22.
-/
theorem original_number_of_men (x : ℕ) (h : 20 * x = 22 * (x - 2)) : x = 22 :=
by
  sorry

end NUMINAMATH_GPT_original_number_of_men_l1554_155484


namespace NUMINAMATH_GPT_johns_watermelon_weight_l1554_155483

theorem johns_watermelon_weight (michael_weight clay_weight john_weight : ℕ)
  (h1 : michael_weight = 8)
  (h2 : clay_weight = 3 * michael_weight)
  (h3 : john_weight = clay_weight / 2) :
  john_weight = 12 :=
by
  sorry

end NUMINAMATH_GPT_johns_watermelon_weight_l1554_155483


namespace NUMINAMATH_GPT_boxes_same_number_oranges_l1554_155469

theorem boxes_same_number_oranges 
  (total_boxes : ℕ) (min_oranges : ℕ) (max_oranges : ℕ) 
  (boxes : ℕ) (range_oranges : ℕ) :
  total_boxes = 150 →
  min_oranges = 130 →
  max_oranges = 160 →
  range_oranges = max_oranges - min_oranges + 1 →
  boxes = total_boxes / range_oranges →
  31 = range_oranges →
  4 ≤ boxes :=
by sorry

end NUMINAMATH_GPT_boxes_same_number_oranges_l1554_155469


namespace NUMINAMATH_GPT_small_bonsai_sold_eq_l1554_155408

-- Define the conditions
def small_bonsai_cost : ℕ := 30
def big_bonsai_cost : ℕ := 20
def big_bonsai_sold : ℕ := 5
def total_earnings : ℕ := 190

-- The proof problem: Prove that the number of small bonsai sold is 3
theorem small_bonsai_sold_eq : ∃ x : ℕ, 30 * x + 20 * 5 = 190 ∧ x = 3 :=
by
  sorry

end NUMINAMATH_GPT_small_bonsai_sold_eq_l1554_155408


namespace NUMINAMATH_GPT_compute_value_l1554_155480

-- Definitions based on problem conditions
def x : ℤ := (150 - 100 + 1) * (100 + 150) / 2  -- Sum of integers from 100 to 150

def y : ℤ := (150 - 100) / 2 + 1  -- Number of even integers from 100 to 150

def z : ℤ := 0  -- Product of odd integers from 100 to 150 (including even numbers makes the product 0)

-- The theorem to prove
theorem compute_value : x + y - z = 6401 :=
by
  sorry

end NUMINAMATH_GPT_compute_value_l1554_155480


namespace NUMINAMATH_GPT_min_correct_all_four_l1554_155474

def total_questions : ℕ := 15
def correct_xiaoxi : ℕ := 11
def correct_xiaofei : ℕ := 12
def correct_xiaomei : ℕ := 13
def correct_xiaoyang : ℕ := 14

theorem min_correct_all_four : 
(∀ total_questions correct_xiaoxi correct_xiaofei correct_xiaomei correct_xiaoyang, 
  total_questions = 15 → correct_xiaoxi = 11 → 
  correct_xiaofei = 12 → correct_xiaomei = 13 → 
  correct_xiaoyang = 14 → 
  ∃ k : ℕ, k = 5 ∧ 
    k = total_questions - ((total_questions - correct_xiaoxi) + 
    (total_questions - correct_xiaofei) + 
    (total_questions - correct_xiaomei) + 
    (total_questions - correct_xiaoyang)) / 4) := 
sorry

end NUMINAMATH_GPT_min_correct_all_four_l1554_155474


namespace NUMINAMATH_GPT_area_of_trapezium_l1554_155429

-- Definitions based on conditions
def length_parallel_side1 : ℝ := 20 -- length of the first parallel side
def length_parallel_side2 : ℝ := 18 -- length of the second parallel side
def distance_between_sides : ℝ := 5 -- distance between the parallel sides

-- Statement to prove
theorem area_of_trapezium (a b h : ℝ) :
  a = length_parallel_side1 → b = length_parallel_side2 → h = distance_between_sides →
  (a + b) * h / 2 = 95 :=
by
  intros ha hb hh
  rw [ha, hb, hh]
  sorry

end NUMINAMATH_GPT_area_of_trapezium_l1554_155429


namespace NUMINAMATH_GPT_simplify_exp_l1554_155439

theorem simplify_exp : (10^8 / (10 * 10^5)) = 100 := 
by
  -- The proof is omitted; we are stating the problem.
  sorry

end NUMINAMATH_GPT_simplify_exp_l1554_155439


namespace NUMINAMATH_GPT_smallest_floor_sum_l1554_155488

theorem smallest_floor_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
    ⌊(a + b) / c⌋ + ⌊(b + c) / a⌋ + ⌊(c + a) / b⌋ ≥ 4 :=
sorry

end NUMINAMATH_GPT_smallest_floor_sum_l1554_155488


namespace NUMINAMATH_GPT_calculate_expression_l1554_155432

theorem calculate_expression : (35 / (5 * 2 + 5)) * 6 = 14 :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l1554_155432


namespace NUMINAMATH_GPT_total_amount_withdrawn_l1554_155422

def principal : ℤ := 20000
def interest_rate : ℚ := 3.33 / 100
def term : ℤ := 3

theorem total_amount_withdrawn :
  principal + (principal * interest_rate * term) = 21998 := by
  sorry

end NUMINAMATH_GPT_total_amount_withdrawn_l1554_155422


namespace NUMINAMATH_GPT_width_of_room_l1554_155407

theorem width_of_room 
  (length : ℝ) 
  (cost : ℝ) 
  (rate : ℝ) 
  (h_length : length = 6.5) 
  (h_cost : cost = 10725) 
  (h_rate : rate = 600) 
  : (cost / rate) / length = 2.75 :=
by
  rw [h_length, h_cost, h_rate]
  norm_num

end NUMINAMATH_GPT_width_of_room_l1554_155407


namespace NUMINAMATH_GPT_simplify_fraction_l1554_155436

theorem simplify_fraction : (3^9 / 9^3) = 27 :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l1554_155436


namespace NUMINAMATH_GPT_solve_fraction_inequality_l1554_155468

theorem solve_fraction_inequality :
  { x : ℝ | x / (x + 5) ≥ 0 } = { x : ℝ | x < -5 } ∪ { x : ℝ | x ≥ 0 } := by
  sorry

end NUMINAMATH_GPT_solve_fraction_inequality_l1554_155468


namespace NUMINAMATH_GPT_number_of_possible_lists_l1554_155485

theorem number_of_possible_lists : 
  let balls := 15
  let draws := 4
  (balls ^ draws) = 50625 := by
  sorry

end NUMINAMATH_GPT_number_of_possible_lists_l1554_155485


namespace NUMINAMATH_GPT_stagePlayRolesAssignment_correct_l1554_155453

noncomputable def stagePlayRolesAssignment : ℕ :=
  let male_roles : ℕ := 4 * 3 -- ways to assign male roles
  let female_roles : ℕ := 5 * 4 -- ways to assign female roles
  let either_gender_roles : ℕ := 5 * 4 * 3 -- ways to assign either-gender roles
  male_roles * female_roles * either_gender_roles -- total assignments

theorem stagePlayRolesAssignment_correct : stagePlayRolesAssignment = 14400 := by
  sorry

end NUMINAMATH_GPT_stagePlayRolesAssignment_correct_l1554_155453


namespace NUMINAMATH_GPT_min_folds_to_exceed_thickness_l1554_155416

def initial_thickness : ℝ := 0.1
def desired_thickness : ℝ := 12

theorem min_folds_to_exceed_thickness : ∃ (n : ℕ), initial_thickness * 2^n > desired_thickness ∧ ∀ m < n, initial_thickness * 2^m ≤ desired_thickness := by
  sorry

end NUMINAMATH_GPT_min_folds_to_exceed_thickness_l1554_155416


namespace NUMINAMATH_GPT_part1_part2_part3_l1554_155476

theorem part1 {k b : ℝ} (h₀ : k ≠ 0) (h₁ : b = 1) (h₂ : k + b = 2) : 
  ∀ x : ℝ, y = k * x + b → y = x + 1 :=
by sorry

theorem part2 : ∃ (C : ℝ × ℝ), 
  C.1 = 3 ∧ C.2 = 4 ∧ y = x + 1 :=
by sorry

theorem part3 {n k : ℝ} (h₀ : k ≠ 0) 
  (h₁ : ∀ x : ℝ, x < 3 → (2 / 3) * x + n > x + 1 ∧ (2 / 3) * x + n < 4) 
  (h₂ : ∀ x : ℝ, y = (2 / 3) * x + n → y = 4 ∧ x = 3) :
  n = 2 :=
by sorry

end NUMINAMATH_GPT_part1_part2_part3_l1554_155476


namespace NUMINAMATH_GPT_smallest_balanced_number_l1554_155426

theorem smallest_balanced_number :
  ∃ (a b c : ℕ), 
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧
  a ≠ b ∧ b ≠ c ∧ c ≠ a ∧
  100 * a + 10 * b + c = 
  (10 * a + b) + (10 * b + a) + (10 * a + c) + (10 * c + a) + (10 * b + c) + (10 * c + b) ∧ 
  100 * a + 10 * b + c = 132 :=
sorry

end NUMINAMATH_GPT_smallest_balanced_number_l1554_155426


namespace NUMINAMATH_GPT_find_n_l1554_155447

open Nat

def is_solution_of_comb_perm (n : ℕ) : Prop :=
    3 * (factorial (n-1) / (factorial (n-5) * factorial 4)) = 5 * (n-2) * (n-3)

theorem find_n (n : ℕ) (h : is_solution_of_comb_perm n) (hn : n ≠ 0) : n = 9 :=
by
  -- will fill proof steps if required
  sorry

end NUMINAMATH_GPT_find_n_l1554_155447


namespace NUMINAMATH_GPT_ratio_of_costs_l1554_155472

theorem ratio_of_costs (R N : ℝ) (hR : 3 * R = 0.25 * (3 * R + 3 * N)) : N / R = 3 := 
sorry

end NUMINAMATH_GPT_ratio_of_costs_l1554_155472


namespace NUMINAMATH_GPT_monkey_climbing_time_l1554_155471

theorem monkey_climbing_time (tree_height : ℕ) (hop_distance : ℕ) (slip_distance : ℕ) (final_hop : ℕ) (net_gain : ℕ) :
  tree_height = 19 →
  hop_distance = 3 →
  slip_distance = 2 →
  net_gain = hop_distance - slip_distance →
  final_hop = hop_distance →
  (tree_height - final_hop) % net_gain = 0 →
  18 / net_gain + 1 = (tree_height - final_hop) / net_gain + 1 := 
by {
  sorry
}

end NUMINAMATH_GPT_monkey_climbing_time_l1554_155471


namespace NUMINAMATH_GPT_year_proof_l1554_155478

variable (n : ℕ)

def packaging_waste_exceeds_threshold (y0 : ℝ) (rate : ℝ) (threshold : ℝ) : Prop :=
  let y := y0 * (rate^n)
  y > threshold

noncomputable def year_when_waste_exceeds := 
  let initial_year := 2015
  let y0 := 4 * 10^6 -- in tons
  let rate := (3.0 / 2.0) -- growth rate per year
  let threshold := 40 * 10^6 -- threshold in tons
  ∃ n, packaging_waste_exceeds_threshold n y0 rate threshold ∧ (initial_year + n = 2021)

theorem year_proof : year_when_waste_exceeds :=
  sorry

end NUMINAMATH_GPT_year_proof_l1554_155478


namespace NUMINAMATH_GPT_a2_plus_a3_eq_40_l1554_155490

theorem a2_plus_a3_eq_40 : 
  ∀ (a a1 a2 a3 a4 a5 : ℤ), 
  (2 * x - 1)^5 = a * x^5 + a1 * x^4 + a2 * x^3 + a3 * x^2 + a4 * x + a5 → 
  a2 + a3 = 40 :=
by
  sorry

end NUMINAMATH_GPT_a2_plus_a3_eq_40_l1554_155490


namespace NUMINAMATH_GPT_commute_weeks_per_month_l1554_155434

variable (total_commute_one_way : ℕ)
variable (gas_cost_per_gallon : ℝ)
variable (car_mileage : ℝ)
variable (commute_days_per_week : ℕ)
variable (individual_monthly_payment : ℝ)
variable (number_of_people : ℕ)

theorem commute_weeks_per_month :
  total_commute_one_way = 21 →
  gas_cost_per_gallon = 2.5 →
  car_mileage = 30 →
  commute_days_per_week = 5 →
  individual_monthly_payment = 14 →
  number_of_people = 5 →
  (individual_monthly_payment * number_of_people) / 
  ((total_commute_one_way * 2 / car_mileage) * gas_cost_per_gallon * commute_days_per_week) = 4 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_commute_weeks_per_month_l1554_155434


namespace NUMINAMATH_GPT_jims_investment_l1554_155414

theorem jims_investment (total_investment : ℝ) (john_ratio : ℝ) (james_ratio : ℝ) (jim_ratio : ℝ) 
                        (h_total_investment : total_investment = 80000)
                        (h_ratio_john : john_ratio = 4)
                        (h_ratio_james : james_ratio = 7)
                        (h_ratio_jim : jim_ratio = 9) : 
    jim_ratio * (total_investment / (john_ratio + james_ratio + jim_ratio)) = 36000 :=
by 
  sorry

end NUMINAMATH_GPT_jims_investment_l1554_155414


namespace NUMINAMATH_GPT_number_of_winning_scores_l1554_155492

theorem number_of_winning_scores : 
  ∃ (scores: ℕ), scores = 19 := by
  sorry

end NUMINAMATH_GPT_number_of_winning_scores_l1554_155492


namespace NUMINAMATH_GPT_second_person_days_l1554_155487

theorem second_person_days (x : ℕ) (h1 : ∀ y : ℝ, y = 24 → 1 / y = 1 / 24)
  (h2 : ∀ z : ℝ, z = 15 → 1 / z = 1 / 15) :
  (1 / 24 + 1 / x = 1 / 15) → x = 40 :=
by
  intro h
  have h3 : 15 * (x + 24) = 24 * x := sorry
  have h4 : 15 * x + 360 = 24 * x := sorry
  have h5 : 360 = 24 * x - 15 * x := sorry
  have h6 : 360 = 9 * x := sorry
  have h7 : x = 360 / 9 := sorry
  have h8 : x = 40 := sorry
  exact h8

end NUMINAMATH_GPT_second_person_days_l1554_155487


namespace NUMINAMATH_GPT_sequence_general_term_formula_l1554_155475

-- Definitions based on conditions
def alternating_sign (n : ℕ) : ℤ := (-1) ^ n
def arithmetic_sequence (n : ℕ) : ℤ := 4 * n - 3

-- Definition for the general term formula
def general_term (n : ℕ) : ℤ := alternating_sign n * arithmetic_sequence n

-- Theorem stating that the given sequence's general term formula is a_n = (-1)^n * (4n - 3)
theorem sequence_general_term_formula (n : ℕ) : general_term n = (-1) ^ n * (4 * n - 3) :=
by
  -- Proof logic will go here
  sorry

end NUMINAMATH_GPT_sequence_general_term_formula_l1554_155475


namespace NUMINAMATH_GPT_find_x_l1554_155403

noncomputable def x_value : ℝ :=
  let x := 24
  x

theorem find_x (x : ℝ) (h : 7 * x + 3 * x + 4 * x + x = 360) : x = 24 := by
  sorry

end NUMINAMATH_GPT_find_x_l1554_155403


namespace NUMINAMATH_GPT_minimum_value_fraction_l1554_155412

theorem minimum_value_fraction (m n : ℝ) (h1 : m + 4 * n = 1) (h2 : m > 0) (h3 : n > 0): 
  (1 / m + 4 / n) ≥ 25 :=
sorry

end NUMINAMATH_GPT_minimum_value_fraction_l1554_155412


namespace NUMINAMATH_GPT_row_seat_notation_l1554_155431

-- Define that the notation (4, 5) corresponds to "Row 4, Seat 5"
def notation_row_seat := (4, 5)

-- Prove that "Row 5, Seat 4" should be denoted as (5, 4)
theorem row_seat_notation : (5, 4) = (5, 4) :=
by sorry

end NUMINAMATH_GPT_row_seat_notation_l1554_155431


namespace NUMINAMATH_GPT_female_guests_from_jays_family_l1554_155498

theorem female_guests_from_jays_family (total_guests : ℕ) (percent_females : ℝ) (percent_from_jays_family : ℝ)
    (h1 : total_guests = 240) (h2 : percent_females = 0.60) (h3 : percent_from_jays_family = 0.50) :
    total_guests * percent_females * percent_from_jays_family = 72 := by
  sorry

end NUMINAMATH_GPT_female_guests_from_jays_family_l1554_155498


namespace NUMINAMATH_GPT_min_value_change_when_2x2_added_l1554_155410

variable (f : ℝ → ℝ)
variable (a b c : ℝ)

def quadratic (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem min_value_change_when_2x2_added
  (a b : ℝ)
  (h1 : ∀ x : ℝ, f x = a * x^2 + b * x + c)
  (h2 : ∀ x : ℝ, (a + 1) * x^2 + b * x + c > a * x^2 + b * x + c + 1)
  (h3 : ∀ x : ℝ, (a - 1) * x^2 + b * x + c < a * x^2 + b * x + c - 3) :
  ∀ x : ℝ, (a + 2) * x^2 + b * x + c = a * x^2 + b * x + (c + 1.5) :=
sorry

end NUMINAMATH_GPT_min_value_change_when_2x2_added_l1554_155410


namespace NUMINAMATH_GPT_jared_popcorn_l1554_155441

-- Define the given conditions
def pieces_per_serving := 30
def number_of_friends := 3
def pieces_per_friend := 60
def servings_ordered := 9

-- Define the total pieces of popcorn
def total_pieces := servings_ordered * pieces_per_serving

-- Define the total pieces of popcorn eaten by Jared's friends
def friends_total_pieces := number_of_friends * pieces_per_friend

-- State the theorem
theorem jared_popcorn : total_pieces - friends_total_pieces = 90 :=
by 
  -- The detailed proof would go here.
  sorry

end NUMINAMATH_GPT_jared_popcorn_l1554_155441


namespace NUMINAMATH_GPT_solve_fraction_eq_l1554_155479

theorem solve_fraction_eq :
  ∀ x : ℝ, (x - 3 ≠ 0) → ((x + 6) / (x - 3) = 4) → x = 6 :=
by
  intros x h_nonzero h_eq
  sorry

end NUMINAMATH_GPT_solve_fraction_eq_l1554_155479


namespace NUMINAMATH_GPT_percentage_problem_l1554_155494

theorem percentage_problem (x : ℝ)
  (h : 0.70 * 600 = 0.40 * x) : x = 1050 :=
sorry

end NUMINAMATH_GPT_percentage_problem_l1554_155494


namespace NUMINAMATH_GPT_intersection_complement_range_m_l1554_155438

open Set

variable (A : Set ℝ) (B : ℝ → Set ℝ) (m : ℝ)

def setA : Set ℝ := Icc (-1 : ℝ) (3 : ℝ)
def setB (m : ℝ) : Set ℝ := Icc m (m + 6)

theorem intersection_complement (m : ℝ) (h : m = 2) : 
  (setA ∩ (setB 2)ᶜ) = Ico (-1 : ℝ) (2 : ℝ) :=
by
  sorry

theorem range_m (m : ℝ) : 
  A ∪ B m = B m ↔ -3 ≤ m ∧ m ≤ -1 :=
by
  sorry

end NUMINAMATH_GPT_intersection_complement_range_m_l1554_155438


namespace NUMINAMATH_GPT_ordered_triples_unique_solution_l1554_155409

theorem ordered_triples_unique_solution :
  ∃! (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ ab = c ∧ bc = a ∧ ca = b ∧ a + b + c = 2 :=
sorry

end NUMINAMATH_GPT_ordered_triples_unique_solution_l1554_155409


namespace NUMINAMATH_GPT_cyclic_sums_sine_cosine_l1554_155445

theorem cyclic_sums_sine_cosine (α β γ : ℝ) (h : α + β + γ = Real.pi) : 
  (Real.sin (2 * α) + Real.sin (2 * β) + Real.sin (2 * γ)) = 
  2 * (Real.sin α + Real.sin β + Real.sin γ) * 
      (Real.cos α + Real.cos β + Real.cos γ) - 
  2 * (Real.sin α + Real.sin β + Real.sin γ) := 
  sorry

end NUMINAMATH_GPT_cyclic_sums_sine_cosine_l1554_155445


namespace NUMINAMATH_GPT_sequence_inequality_l1554_155400

theorem sequence_inequality
  (a : ℕ → ℝ)
  (h_cond : ∀ k m : ℕ, |a (k + m) - a k - a m| ≤ 1) :
  ∀ p q : ℕ, |a p / p - a q / q| < 1 / p + 1 / q :=
by
  intros p q
  sorry

end NUMINAMATH_GPT_sequence_inequality_l1554_155400


namespace NUMINAMATH_GPT_part1_part2_l1554_155456

def f (x m : ℝ) : ℝ := |x - m| - |x + 3 * m|

theorem part1 (x : ℝ) : 
  (∀ (x : ℝ), f x 1 ≥ 1 → x ≤ -3 / 2) :=
sorry

theorem part2 (x t : ℝ) (h : ∀ (x t : ℝ), f x m < |2 + t| + |t - 1|) : 
  0 < m ∧ m < 3 / 4 :=
sorry

end NUMINAMATH_GPT_part1_part2_l1554_155456


namespace NUMINAMATH_GPT_pencil_count_l1554_155464

theorem pencil_count (a : ℕ) :
  200 ≤ a ∧ a ≤ 300 ∧ a % 10 = 7 ∧ a % 12 = 9 → (a = 237 ∨ a = 297) :=
by sorry

end NUMINAMATH_GPT_pencil_count_l1554_155464


namespace NUMINAMATH_GPT_volume_region_inequality_l1554_155482

theorem volume_region_inequality : 
  ∃ (V : ℝ), V = (20 / 3) ∧ 
    ∀ (x y z : ℝ), |x + y + z| + |x + y - z| + |x - y + z| + |-x + y + z| ≤ 4 
    → x^2 + y^2 + z^2 ≤ V :=
sorry

end NUMINAMATH_GPT_volume_region_inequality_l1554_155482


namespace NUMINAMATH_GPT_cube_difference_divisible_by_16_l1554_155401

theorem cube_difference_divisible_by_16 (a b : ℤ) : 
  16 ∣ ((2 * a + 1)^3 - (2 * b + 1)^3 + 8) :=
by
  sorry

end NUMINAMATH_GPT_cube_difference_divisible_by_16_l1554_155401


namespace NUMINAMATH_GPT_average_speed_increased_pace_l1554_155455

theorem average_speed_increased_pace 
  (speed_constant : ℝ) (time_constant : ℝ) (distance_increased : ℝ) (total_time : ℝ) 
  (h1 : speed_constant = 15) 
  (h2 : time_constant = 3) 
  (h3 : distance_increased = 190) 
  (h4 : total_time = 13) :
  (distance_increased / (total_time - time_constant)) = 19 :=
by
  sorry

end NUMINAMATH_GPT_average_speed_increased_pace_l1554_155455


namespace NUMINAMATH_GPT_motel_total_rent_l1554_155405

theorem motel_total_rent (R₅₀ R₆₀ : ℕ) 
  (h₁ : ∀ x y : ℕ, 50 * x + 60 * y = 50 * (x + 10) + 60 * (y - 10) + 100)
  (h₂ : ∀ x y : ℕ, 25 * (50 * x + 60 * y) = 10000) : 
  50 * R₅₀ + 60 * R₆₀ = 400 :=
by
  sorry

end NUMINAMATH_GPT_motel_total_rent_l1554_155405


namespace NUMINAMATH_GPT_probability_of_selecting_quarter_l1554_155411

theorem probability_of_selecting_quarter 
  (value_quarters value_nickels value_pennies total_value : ℚ)
  (coin_value_quarter coin_value_nickel coin_value_penny : ℚ) 
  (h1 : value_quarters = 10)
  (h2 : value_nickels = 10)
  (h3 : value_pennies = 10)
  (h4 : coin_value_quarter = 0.25)
  (h5 : coin_value_nickel = 0.05)
  (h6 : coin_value_penny = 0.01)
  (total_coins : ℚ) 
  (h7 : total_coins = (value_quarters / coin_value_quarter) + (value_nickels / coin_value_nickel) + (value_pennies / coin_value_penny)) : 
  (value_quarters / coin_value_quarter) / total_coins = 1 / 31 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_selecting_quarter_l1554_155411


namespace NUMINAMATH_GPT_find_k_l1554_155440

theorem find_k (k : ℤ) (h1 : ∃(a b c : ℤ), a = (36 + k) ∧ b = (300 + k) ∧ c = (596 + k) ∧ (∃ d, 
  (a = d^2) ∧ (b = (d + 1)^2) ∧ (c = (d + 2)^2)) ) : k = 925 := by
  sorry

end NUMINAMATH_GPT_find_k_l1554_155440


namespace NUMINAMATH_GPT_minimum_value_of_function_l1554_155459

theorem minimum_value_of_function (x : ℝ) (hx : x > 1) : (x + 4 / (x - 1)) ≥ 5 := by
  sorry

end NUMINAMATH_GPT_minimum_value_of_function_l1554_155459


namespace NUMINAMATH_GPT_find_integers_divisible_by_18_in_range_l1554_155452

theorem find_integers_divisible_by_18_in_range :
  ∃ n : ℕ, (n % 18 = 0) ∧ (n ≥ 900) ∧ (n ≤ 930) ∧ (n = 900 ∨ n = 918) :=
sorry

end NUMINAMATH_GPT_find_integers_divisible_by_18_in_range_l1554_155452


namespace NUMINAMATH_GPT_find_f3_l1554_155417

theorem find_f3 (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, f x + 3 * f (1 - x) = 4 * x^3) : f 3 = -25.5 :=
sorry

end NUMINAMATH_GPT_find_f3_l1554_155417


namespace NUMINAMATH_GPT_not_all_pieces_found_l1554_155425

theorem not_all_pieces_found (k p v : ℕ) (h1 : p + v > 0) (h2 : k % 2 = 1) : k + 4 * p + 8 * v ≠ 1988 :=
by
  sorry

end NUMINAMATH_GPT_not_all_pieces_found_l1554_155425


namespace NUMINAMATH_GPT_multiplication_modulo_l1554_155420

theorem multiplication_modulo :
  ∃ n : ℕ, (253 * 649 ≡ n [MOD 100]) ∧ (0 ≤ n) ∧ (n < 100) ∧ (n = 97) := 
by
  sorry

end NUMINAMATH_GPT_multiplication_modulo_l1554_155420


namespace NUMINAMATH_GPT_circle_condition_tangent_lines_right_angle_triangle_l1554_155495

-- Part (1): Range of m for the equation to represent a circle
theorem circle_condition {m : ℝ} : 
  (∀ x y : ℝ, x^2 + y^2 - 2*x + 2*m*y + m^2 - 2*m - 2 = 0 →
  (m > -3 / 2)) :=
sorry

-- Part (2): Equation of tangent line to circle C
theorem tangent_lines {m : ℝ} (h : m = -1) : 
  ∀ x y : ℝ,
  ((x - 1)^2 + (y - 1)^2 = 1 →
  ((x = 2) ∨ (4*x - 3*y + 4 = 0))) :=
sorry

-- Part (3): Value of t for the line intersecting circle at a right angle
theorem right_angle_triangle {t : ℝ} :
  (∀ x y : ℝ, 
  (x + y + t = 0) →
  (t = -3 ∨ t = -1)) :=
sorry

end NUMINAMATH_GPT_circle_condition_tangent_lines_right_angle_triangle_l1554_155495


namespace NUMINAMATH_GPT_trees_left_after_typhoon_and_growth_l1554_155493

-- Conditions
def initial_trees : ℕ := 9
def trees_died_in_typhoon : ℕ := 4
def new_trees : ℕ := 5

-- Question (Proof Problem)
theorem trees_left_after_typhoon_and_growth : 
  initial_trees - trees_died_in_typhoon + new_trees = 10 := 
by
  sorry

end NUMINAMATH_GPT_trees_left_after_typhoon_and_growth_l1554_155493


namespace NUMINAMATH_GPT_perimeter_of_first_square_l1554_155497

theorem perimeter_of_first_square
  (s1 s2 s3 : ℝ)
  (P1 P2 P3 : ℝ)
  (A1 A2 A3 : ℝ)
  (hs2 : s2 = 8)
  (hs3 : s3 = 10)
  (hP2 : P2 = 4 * s2)
  (hP3 : P3 = 4 * s3)
  (hP2_val : P2 = 32)
  (hP3_val : P3 = 40)
  (hA2 : A2 = s2^2)
  (hA3 : A3 = s3^2)
  (hA1_A2_A3 : A3 = A1 + A2)
  (hA3_val : A3 = 100)
  (hA2_val : A2 = 64) :
  P1 = 24 := by
  sorry

end NUMINAMATH_GPT_perimeter_of_first_square_l1554_155497


namespace NUMINAMATH_GPT_sum_of_decimals_l1554_155404

theorem sum_of_decimals : (1 / 10) + (9 / 100) + (9 / 1000) + (7 / 10000) = 0.1997 := 
sorry

end NUMINAMATH_GPT_sum_of_decimals_l1554_155404


namespace NUMINAMATH_GPT_solution_set_of_new_inequality_l1554_155437

-- Define the conditions
variable (a b c x : ℝ)

-- ax^2 + bx + c > 0 has solution set {-3 < x < 2}
def inequality_solution_set (a b c : ℝ) : Prop := ∀ x : ℝ, (-3 < x ∧ x < 2) → a * x^2 + b * x + c > 0

-- Prove that cx^2 + bx + a > 0 has solution set {x < -1/3 ∨ x > 1/2}
theorem solution_set_of_new_inequality
  (a b c : ℝ)
  (h : a < 0 ∧ inequality_solution_set a b c) :
  ∀ x : ℝ, (x < -1/3 ∨ x > 1/2) ↔ (c * x^2 + b * x + a > 0) := sorry

end NUMINAMATH_GPT_solution_set_of_new_inequality_l1554_155437


namespace NUMINAMATH_GPT_fraction_exists_l1554_155467

theorem fraction_exists (n d k : ℕ) (h₁ : n = k * d) (h₂ : d > 0) (h₃ : k > 0) : 
  ∃ (i j : ℕ), i < n ∧ j < n ∧ i + j = n ∧ i/j = d-1 :=
by
  sorry

end NUMINAMATH_GPT_fraction_exists_l1554_155467


namespace NUMINAMATH_GPT_friends_meeting_games_only_l1554_155423

theorem friends_meeting_games_only 
  (M P G MP MG PG MPG : ℕ) 
  (h1 : M + MP + MG + MPG = 10) 
  (h2 : P + MP + PG + MPG = 20) 
  (h3 : MP = 4) 
  (h4 : MG = 2) 
  (h5 : PG = 0) 
  (h6 : MPG = 2) 
  (h7 : M + P + G + MP + MG + PG + MPG = 31) : 
  G = 1 := 
by
  sorry

end NUMINAMATH_GPT_friends_meeting_games_only_l1554_155423


namespace NUMINAMATH_GPT_height_difference_l1554_155402

variable (H_A H_B : ℝ)

-- Conditions
axiom B_is_66_67_percent_more_than_A : H_B = H_A * 1.6667

-- Proof statement
theorem height_difference (H_A H_B : ℝ) (h : H_B = H_A * 1.6667) : 
  (H_B - H_A) / H_B * 100 = 40 := by
sorry

end NUMINAMATH_GPT_height_difference_l1554_155402


namespace NUMINAMATH_GPT_largest_integer_satisfying_conditions_l1554_155428

theorem largest_integer_satisfying_conditions (n : ℤ) (m : ℤ) :
  n^2 = (m + 1)^3 - m^3 ∧ ∃ k : ℤ, 2 * n + 103 = k^2 → n = 313 := 
by 
  sorry

end NUMINAMATH_GPT_largest_integer_satisfying_conditions_l1554_155428


namespace NUMINAMATH_GPT_problem_conditions_l1554_155421

theorem problem_conditions (x y : ℝ) (h : x^2 + y^2 - x * y = 1) :
  ¬ (x + y ≤ 1) ∧ (x + y ≥ -2) ∧ (x^2 + y^2 ≤ 2) ∧ ¬ (x^2 + y^2 ≥ 1) :=
by
  sorry

end NUMINAMATH_GPT_problem_conditions_l1554_155421


namespace NUMINAMATH_GPT_find_xyz_sum_l1554_155435

variables {x y z : ℝ}

def system_of_equations (x y z : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ z > 0 ∧
  (x^2 + x * y + y^2 = 12) ∧
  (y^2 + y * z + z^2 = 9) ∧
  (z^2 + z * x + x^2 = 21)

theorem find_xyz_sum (x y z : ℝ) (h : system_of_equations x y z) : 
  x * y + y * z + z * x = 12 :=
sorry

end NUMINAMATH_GPT_find_xyz_sum_l1554_155435


namespace NUMINAMATH_GPT_sarah_min_width_l1554_155446

noncomputable def minWidth (S : Type) [LinearOrder S] (w : S) : Prop :=
  ∃ w, w ≥ 0 ∧ w * (w + 20) ≥ 150 ∧ ∀ w', (w' ≥ 0 ∧ w' * (w' + 20) ≥ 150) → w ≤ w'

theorem sarah_min_width : minWidth ℝ 10 :=
by {
  sorry -- proof goes here
}

end NUMINAMATH_GPT_sarah_min_width_l1554_155446


namespace NUMINAMATH_GPT_negation_of_p_l1554_155406

def p : Prop := ∀ x : ℝ, x^2 - 2*x + 2 ≤ Real.sin x
def not_p : Prop := ∃ x : ℝ, x^2 - 2*x + 2 > Real.sin x

theorem negation_of_p : ¬ p ↔ not_p := by
  sorry

end NUMINAMATH_GPT_negation_of_p_l1554_155406
