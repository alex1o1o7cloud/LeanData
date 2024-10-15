import Mathlib

namespace NUMINAMATH_GPT_sum_of_cubes_is_81720_l2025_202541

-- Let n be the smallest of these consecutive even integers.
def smallest_even : Int := 28

-- Assumptions given the conditions
def sum_of_squares (n : Int) : Int := n^2 + (n + 2)^2 + (n + 4)^2

-- The condition provided is that sum of the squares is 2930
lemma sum_of_squares_is_2930 : sum_of_squares smallest_even = 2930 := by
  sorry

-- To prove that the sum of the cubes of these three integers is 81720
def sum_of_cubes (n : Int) : Int := n^3 + (n + 2)^3 + (n + 4)^3

theorem sum_of_cubes_is_81720 : sum_of_cubes smallest_even = 81720 := by
  sorry

end NUMINAMATH_GPT_sum_of_cubes_is_81720_l2025_202541


namespace NUMINAMATH_GPT_custom_mul_4_3_l2025_202500

-- Define the binary operation a*b = a^2 - ab + b^2
def custom_mul (a b : ℕ) : ℕ := a^2 - a*b + b^2

-- State the theorem to prove that 4 * 3 = 13
theorem custom_mul_4_3 : custom_mul 4 3 = 13 := by
  sorry -- Proof will be filled in here

end NUMINAMATH_GPT_custom_mul_4_3_l2025_202500


namespace NUMINAMATH_GPT_bakery_gives_away_30_doughnuts_at_end_of_day_l2025_202524

def boxes_per_day (total_doughnuts doughnuts_per_box : ℕ) : ℕ :=
  total_doughnuts / doughnuts_per_box

def leftover_boxes (total_boxes sold_boxes : ℕ) : ℕ :=
  total_boxes - sold_boxes

def doughnuts_given_away (leftover_boxes doughnuts_per_box : ℕ) : ℕ :=
  leftover_boxes * doughnuts_per_box

theorem bakery_gives_away_30_doughnuts_at_end_of_day 
  (total_doughnuts doughnuts_per_box sold_boxes : ℕ) 
  (H1 : total_doughnuts = 300) (H2 : doughnuts_per_box = 10) (H3 : sold_boxes = 27) : 
  doughnuts_given_away (leftover_boxes (boxes_per_day total_doughnuts doughnuts_per_box) sold_boxes) doughnuts_per_box = 30 :=
by
  sorry

end NUMINAMATH_GPT_bakery_gives_away_30_doughnuts_at_end_of_day_l2025_202524


namespace NUMINAMATH_GPT_monomial_combined_l2025_202529

theorem monomial_combined (n m : ℕ) (h₁ : 2 = n) (h₂ : m = 4) : n^m = 16 := by
  sorry

end NUMINAMATH_GPT_monomial_combined_l2025_202529


namespace NUMINAMATH_GPT_area_of_quadrilateral_l2025_202595

/-- The area of the quadrilateral defined by the system of inequalities is 15/7. -/
theorem area_of_quadrilateral : 
  (∃ (x y : ℝ), 3 * x + 2 * y ≤ 6 ∧ x + 3 * y ≥ 3 ∧ x ≥ 0 ∧ y ≥ 0) →
  (∃ (area : ℝ), area = 15 / 7) :=
by
  sorry

end NUMINAMATH_GPT_area_of_quadrilateral_l2025_202595


namespace NUMINAMATH_GPT_prime_square_minus_one_divisible_by_twelve_l2025_202535

theorem prime_square_minus_one_divisible_by_twelve (p : ℕ) (hp : Nat.Prime p) (hp_gt : p > 3) : 12 ∣ (p^2 - 1) :=
by
  sorry

end NUMINAMATH_GPT_prime_square_minus_one_divisible_by_twelve_l2025_202535


namespace NUMINAMATH_GPT_max_ab_l2025_202571

theorem max_ab (a b c : ℝ) (h1 : 0 < a ∧ a < 1) (h2 : 0 < b ∧ b < 1) (h3 : 0 < c ∧ c < 1) (h4 : 3 * a + 2 * b = 2) :
  ab ≤ 1 / 6 :=
sorry

end NUMINAMATH_GPT_max_ab_l2025_202571


namespace NUMINAMATH_GPT_joan_spent_on_toys_l2025_202549

theorem joan_spent_on_toys :
  let toy_cars := 14.88
  let toy_trucks := 5.86
  toy_cars + toy_trucks = 20.74 :=
by
  let toy_cars := 14.88
  let toy_trucks := 5.86
  sorry

end NUMINAMATH_GPT_joan_spent_on_toys_l2025_202549


namespace NUMINAMATH_GPT_y_coordinate_of_second_point_l2025_202557

variable {m n k : ℝ}

theorem y_coordinate_of_second_point (h1 : m = 2 * n + 5) (h2 : k = 0.5) : (n + k) = n + 0.5 := 
by
  sorry

end NUMINAMATH_GPT_y_coordinate_of_second_point_l2025_202557


namespace NUMINAMATH_GPT_recommendation_plans_count_l2025_202530

theorem recommendation_plans_count :
  let total_students := 7
  let sports_talents := 2
  let artistic_talents := 2
  let other_talents := 3
  let recommend_count := 4
  let condition_sports := recommend_count >= 1
  let condition_artistic := recommend_count >= 1
  (condition_sports ∧ condition_artistic) → 
  ∃ (n : ℕ), n = 25 := sorry

end NUMINAMATH_GPT_recommendation_plans_count_l2025_202530


namespace NUMINAMATH_GPT_muffins_in_each_pack_l2025_202513

-- Define the conditions as constants
def total_amount_needed : ℕ := 120
def price_per_muffin : ℕ := 2
def number_of_cases : ℕ := 5
def packs_per_case : ℕ := 3

-- Define the theorem to prove
theorem muffins_in_each_pack :
  (total_amount_needed / price_per_muffin) / (number_of_cases * packs_per_case) = 4 :=
by
  sorry

end NUMINAMATH_GPT_muffins_in_each_pack_l2025_202513


namespace NUMINAMATH_GPT_side_length_is_36_l2025_202583

variable (a : ℝ)

def side_length_of_largest_square (a : ℝ) := 
  2 * (a / 2) ^ 2 + 2 * (a / 4) ^ 2 = 810

theorem side_length_is_36 (h : side_length_of_largest_square a) : a = 36 :=
by
  sorry

end NUMINAMATH_GPT_side_length_is_36_l2025_202583


namespace NUMINAMATH_GPT_sum_series_a_eq_one_sum_series_b_eq_half_sum_series_c_eq_third_l2025_202556

noncomputable def sum_series_a : ℝ :=
∑' n, (1 / (n * (n + 1)))

noncomputable def sum_series_b : ℝ :=
∑' n, (1 / ((n + 1) * (n + 2)))

noncomputable def sum_series_c : ℝ :=
∑' n, (1 / ((n + 2) * (n + 3)))

theorem sum_series_a_eq_one : sum_series_a = 1 := sorry

theorem sum_series_b_eq_half : sum_series_b = 1 / 2 := sorry

theorem sum_series_c_eq_third : sum_series_c = 1 / 3 := sorry

end NUMINAMATH_GPT_sum_series_a_eq_one_sum_series_b_eq_half_sum_series_c_eq_third_l2025_202556


namespace NUMINAMATH_GPT_total_amount_l2025_202538

variable (x y z : ℝ)

def condition1 : Prop := y = 0.45 * x
def condition2 : Prop := z = 0.30 * x
def condition3 : Prop := y = 36

theorem total_amount (h1 : condition1 x y)
                     (h2 : condition2 x z)
                     (h3 : condition3 y) :
  x + y + z = 140 :=
by
  sorry

end NUMINAMATH_GPT_total_amount_l2025_202538


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l2025_202599

theorem sufficient_but_not_necessary_condition (x : ℝ) (h : x^2 - 3 * x + 2 > 0) : x > 2 ∨ x < -1 :=
by
  sorry

example (x : ℝ) (h : x^2 - 3 * x + 2 > 0) : (x > 2) ∨ (x < -1) := 
by 
  apply sufficient_but_not_necessary_condition; exact h

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l2025_202599


namespace NUMINAMATH_GPT_total_drums_l2025_202562

theorem total_drums (x y : ℕ) (hx : 30 * x + 20 * y = 160) : x + y = 7 :=
sorry

end NUMINAMATH_GPT_total_drums_l2025_202562


namespace NUMINAMATH_GPT_problem_statement_l2025_202568

def g (x : ℕ) : ℕ := x^2 - 4 * x

theorem problem_statement :
  g (g (g (g (g (g 2))))) = L := sorry

end NUMINAMATH_GPT_problem_statement_l2025_202568


namespace NUMINAMATH_GPT_bridge_length_l2025_202589

theorem bridge_length 
  (train_length : ℕ) 
  (speed_km_hr : ℕ) 
  (cross_time_sec : ℕ) 
  (conversion_factor_num : ℕ) 
  (conversion_factor_den : ℕ)
  (expected_length : ℕ) 
  (speed_m_s : ℕ := speed_km_hr * conversion_factor_num / conversion_factor_den)
  (total_distance : ℕ := speed_m_s * cross_time_sec) :
  train_length = 150 →
  speed_km_hr = 45 →
  cross_time_sec = 30 →
  conversion_factor_num = 1000 →
  conversion_factor_den = 3600 →
  expected_length = 225 →
  total_distance - train_length = expected_length :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  sorry

end NUMINAMATH_GPT_bridge_length_l2025_202589


namespace NUMINAMATH_GPT_restaurant_production_in_june_l2025_202554

def cheese_pizzas_per_day (hot_dogs_per_day : ℕ) : ℕ :=
  hot_dogs_per_day + 40

def pepperoni_pizzas_per_day (cheese_pizzas_per_day : ℕ) : ℕ :=
  2 * cheese_pizzas_per_day

def hot_dogs_per_day := 60
def beef_hot_dogs_per_day := 30
def chicken_hot_dogs_per_day := 30
def days_in_june := 30

theorem restaurant_production_in_june :
  (cheese_pizzas_per_day hot_dogs_per_day * days_in_june = 3000) ∧
  (pepperoni_pizzas_per_day (cheese_pizzas_per_day hot_dogs_per_day) * days_in_june = 6000) ∧
  (beef_hot_dogs_per_day * days_in_june = 900) ∧
  (chicken_hot_dogs_per_day * days_in_june = 900) :=
by
  sorry

end NUMINAMATH_GPT_restaurant_production_in_june_l2025_202554


namespace NUMINAMATH_GPT_total_cost_in_dollars_l2025_202547

def pencil_price := 20 -- price of one pencil in cents
def tolu_pencils := 3 -- pencils Tolu wants
def robert_pencils := 5 -- pencils Robert wants
def melissa_pencils := 2 -- pencils Melissa wants

theorem total_cost_in_dollars :
  (pencil_price * (tolu_pencils + robert_pencils + melissa_pencils)) / 100 = 2 := 
by
  sorry

end NUMINAMATH_GPT_total_cost_in_dollars_l2025_202547


namespace NUMINAMATH_GPT_total_fruit_count_l2025_202522

-- Define the conditions as variables and equations
def apples := 4 -- based on the final deduction from the solution
def pears := 6 -- calculated from the condition of bananas
def bananas := 9 -- given in the problem

-- State the conditions
axiom h1 : pears = apples + 2
axiom h2 : bananas = pears + 3
axiom h3 : bananas = 9

-- State the proof objective
theorem total_fruit_count : apples + pears + bananas = 19 :=
by
  sorry

end NUMINAMATH_GPT_total_fruit_count_l2025_202522


namespace NUMINAMATH_GPT_find_divisor_l2025_202520

theorem find_divisor :
  ∃ d : ℕ, (4499 + 1) % d = 0 ∧ d = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_divisor_l2025_202520


namespace NUMINAMATH_GPT_adam_money_given_l2025_202592

theorem adam_money_given (original_money : ℕ) (final_money : ℕ) (money_given : ℕ) :
  original_money = 79 →
  final_money = 92 →
  money_given = final_money - original_money →
  money_given = 13 := by
sorry

end NUMINAMATH_GPT_adam_money_given_l2025_202592


namespace NUMINAMATH_GPT_sum_of_remainders_l2025_202569

theorem sum_of_remainders (a b c d : ℕ)
  (ha : a % 17 = 3) (hb : b % 17 = 5) (hc : c % 17 = 7) (hd : d % 17 = 9) :
  (a + b + c + d) % 17 = 7 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_remainders_l2025_202569


namespace NUMINAMATH_GPT_original_number_is_45_l2025_202532

theorem original_number_is_45 (x : ℕ) (h : x - 30 = x / 3) : x = 45 :=
by {
  sorry
}

end NUMINAMATH_GPT_original_number_is_45_l2025_202532


namespace NUMINAMATH_GPT_problem1_problem2_l2025_202570

noncomputable def problem1_solution1 : ℝ := (2 + Real.sqrt 6) / 2
noncomputable def problem1_solution2 : ℝ := (2 - Real.sqrt 6) / 2

theorem problem1 (x : ℝ) : 
  (2 * x ^ 2 - 4 * x - 1 = 0) ↔ (x = problem1_solution1 ∨ x = problem1_solution2) :=
by
  sorry

theorem problem2 : 
  (4 * (x + 2) ^ 2 - 9 * (x - 3) ^ 2 = 0) ↔ (x = 1 ∨ x = 13) :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l2025_202570


namespace NUMINAMATH_GPT_four_circles_max_parts_l2025_202548

theorem four_circles_max_parts (n : ℕ) (h1 : ∀ n, n = 1 ∨ n = 2 ∨ n = 3 → ∃ k, k = 2^n) :
    n = 4 → ∃ k, k = 14 :=
by
  sorry

end NUMINAMATH_GPT_four_circles_max_parts_l2025_202548


namespace NUMINAMATH_GPT_find_largest_number_l2025_202508

theorem find_largest_number (a b c : ℝ) (h1 : a + b + c = 100) (h2 : c - b = 10) (h3 : b - a = 5) : c = 41.67 := 
sorry

end NUMINAMATH_GPT_find_largest_number_l2025_202508


namespace NUMINAMATH_GPT_initial_cost_of_article_l2025_202501

variable (P : ℝ)

theorem initial_cost_of_article (h1 : 0.70 * P = 2100) (h2 : 0.50 * (0.70 * P) = 1050) : P = 3000 :=
by
  sorry

end NUMINAMATH_GPT_initial_cost_of_article_l2025_202501


namespace NUMINAMATH_GPT_impossible_four_teams_tie_possible_three_teams_tie_l2025_202551

-- Definitions for the conditions
def num_teams : ℕ := 4
def num_matches : ℕ := (num_teams * (num_teams - 1)) / 2
def total_possible_outcomes : ℕ := 2^num_matches
def winning_rate : ℚ := 1 / 2

-- Problem 1: It is impossible for exactly four teams to tie for first place.
theorem impossible_four_teams_tie :
  ¬ ∃ (score : ℕ), (∀ (team : ℕ) (h : team < num_teams), team = score ∧
                     (num_teams * score = num_matches / 2 ∧
                      num_teams * score + num_matches / 2 = num_matches)) := sorry

-- Problem 2: It is possible for exactly three teams to tie for first place.
theorem possible_three_teams_tie :
  ∃ (score : ℕ), (∃ (teamA teamB teamC teamD : ℕ),
  (teamA < num_teams ∧ teamB < num_teams ∧ teamC < num_teams ∧ teamD <num_teams ∧ teamA ≠ teamB ∧ teamA ≠ teamC ∧ teamA ≠ teamD ∧ 
  teamB ≠ teamC ∧ teamB ≠ teamD ∧ teamC ≠ teamD)) ∧
  (teamA = score ∧ teamB = score ∧ teamC = score ∧ teamD = 0) := sorry

end NUMINAMATH_GPT_impossible_four_teams_tie_possible_three_teams_tie_l2025_202551


namespace NUMINAMATH_GPT_mary_initial_amount_l2025_202566

theorem mary_initial_amount (current_amount pie_cost mary_after_pie : ℕ) 
  (h1 : pie_cost = 6) 
  (h2 : mary_after_pie = 52) :
  current_amount = pie_cost + mary_after_pie → 
  current_amount = 58 :=
by
  intro h
  rw [h1, h2] at h
  exact h

end NUMINAMATH_GPT_mary_initial_amount_l2025_202566


namespace NUMINAMATH_GPT_population_net_increase_in_one_day_l2025_202586

-- Define the problem conditions
def birth_rate : ℕ := 6 / 2  -- births per second
def death_rate : ℕ := 3 / 2  -- deaths per second
def seconds_in_a_day : ℕ := 60 * 60 * 24

-- Define the assertion we want to prove
theorem population_net_increase_in_one_day : 
  ( (birth_rate - death_rate) * seconds_in_a_day ) = 259200 := by
  -- Since 6/2 = 3 and 3/2 = 1.5 is not an integer in Lean, we use ratios directly
  sorry  -- Proof is not required

end NUMINAMATH_GPT_population_net_increase_in_one_day_l2025_202586


namespace NUMINAMATH_GPT_ticket_sales_revenue_l2025_202590

theorem ticket_sales_revenue :
  let student_ticket_price := 4
  let general_admission_ticket_price := 6
  let total_tickets_sold := 525
  let general_admission_tickets_sold := 388
  let student_tickets_sold := total_tickets_sold - general_admission_tickets_sold
  let money_from_student_tickets := student_tickets_sold * student_ticket_price
  let money_from_general_admission_tickets := general_admission_tickets_sold * general_admission_ticket_price
  let total_money_collected := money_from_student_tickets + money_from_general_admission_tickets
  total_money_collected = 2876 :=
by
  sorry

end NUMINAMATH_GPT_ticket_sales_revenue_l2025_202590


namespace NUMINAMATH_GPT_max_value_a_plus_b_plus_c_plus_d_eq_34_l2025_202581

theorem max_value_a_plus_b_plus_c_plus_d_eq_34 :
  ∃ (a b c d : ℕ), (∀ (x y: ℝ), 0 < x → 0 < y → x^2 - 2 * x * y + 3 * y^2 = 10 → x^2 + 2 * x * y + 3 * y^2 = (a + b * Real.sqrt c) / d) ∧ a + b + c + d = 34 :=
sorry

end NUMINAMATH_GPT_max_value_a_plus_b_plus_c_plus_d_eq_34_l2025_202581


namespace NUMINAMATH_GPT_squirrels_more_than_nuts_l2025_202572

theorem squirrels_more_than_nuts (squirrels nuts : ℕ) (h1 : squirrels = 4) (h2 : nuts = 2) : squirrels - nuts = 2 := by
  sorry

end NUMINAMATH_GPT_squirrels_more_than_nuts_l2025_202572


namespace NUMINAMATH_GPT_John_study_time_second_exam_l2025_202579

variable (StudyTime Score : ℝ)
variable (k : ℝ) (h1 : k = Score / StudyTime)
variable (study_first : ℝ := 3) (score_first : ℝ := 60)
variable (avg_target : ℝ := 75)
variable (total_tests : ℕ := 2)

theorem John_study_time_second_exam :
  (avg_target * total_tests - score_first) / (score_first / study_first) = 4.5 :=
by
  sorry

end NUMINAMATH_GPT_John_study_time_second_exam_l2025_202579


namespace NUMINAMATH_GPT_grasshopper_total_distance_l2025_202533

theorem grasshopper_total_distance :
  let initial := 2
  let first_jump := -3
  let second_jump := 8
  let final_jump := -1
  abs (first_jump - initial) + abs (second_jump - first_jump) + abs (final_jump - second_jump) = 25 :=
by
  sorry

end NUMINAMATH_GPT_grasshopper_total_distance_l2025_202533


namespace NUMINAMATH_GPT_negation_of_exists_leq_l2025_202588

theorem negation_of_exists_leq (
  P : ∃ x : ℝ, x^2 - 2 * x + 4 ≤ 0
) : ∀ x : ℝ, x^2 - 2 * x + 4 > 0 :=
sorry

end NUMINAMATH_GPT_negation_of_exists_leq_l2025_202588


namespace NUMINAMATH_GPT_find_f_2547_l2025_202526

theorem find_f_2547 (f : ℚ → ℚ)
  (h1 : ∀ x y : ℚ, f (x + y) = f x + f y + 2547)
  (h2 : f 2004 = 2547) :
  f 2547 = 2547 :=
sorry

end NUMINAMATH_GPT_find_f_2547_l2025_202526


namespace NUMINAMATH_GPT_inequality_system_solution_l2025_202594

theorem inequality_system_solution (x : ℝ) (h1 : 5 - 2 * x ≤ 1) (h2 : x - 4 < 0) : 2 ≤ x ∧ x < 4 :=
  sorry

end NUMINAMATH_GPT_inequality_system_solution_l2025_202594


namespace NUMINAMATH_GPT_perfect_even_multiples_of_3_under_3000_l2025_202537

theorem perfect_even_multiples_of_3_under_3000 :
  ∃ n : ℕ, n = 9 ∧ ∀ (k : ℕ), (36 * k^2 < 3000) → (36 * k^2) % 2 = 0 ∧ (36 * k^2) % 3 = 0 ∧ ∃ m : ℕ, m^2 = 36 * k^2 :=
by
  sorry

end NUMINAMATH_GPT_perfect_even_multiples_of_3_under_3000_l2025_202537


namespace NUMINAMATH_GPT_length_of_bridge_l2025_202567

theorem length_of_bridge (ship_length : ℝ) (ship_speed_kmh : ℝ) (time : ℝ) (bridge_length : ℝ) :
  ship_length = 450 → ship_speed_kmh = 24 → time = 202.48 → bridge_length = (6.67 * 202.48 - 450) → bridge_length = 900.54 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_length_of_bridge_l2025_202567


namespace NUMINAMATH_GPT_q_is_false_l2025_202542

-- Given conditions
variables (p q : Prop)
axiom h1 : ¬ (p ∧ q)
axiom h2 : ¬ ¬ p

-- Proof that q is false
theorem q_is_false : q = False :=
by
  sorry

end NUMINAMATH_GPT_q_is_false_l2025_202542


namespace NUMINAMATH_GPT_min_value_f_l2025_202575

noncomputable def f (a b : ℝ) : ℝ := (1 / a^5 + a^5 - 2) * (1 / b^5 + b^5 - 2)

theorem min_value_f :
  ∀ (a b : ℝ), a > 0 → b > 0 → a + b = 1 → f a b ≥ (31^4 / 32^2) :=
by
  intros
  sorry

end NUMINAMATH_GPT_min_value_f_l2025_202575


namespace NUMINAMATH_GPT_jorges_total_yield_l2025_202558

def total_yield (good_acres clay_acres : ℕ) (good_yield clay_yield : ℕ) : ℕ :=
  good_acres * good_yield + clay_acres * clay_yield / 2

theorem jorges_total_yield :
  let acres := 60
  let good_yield_per_acre := 400
  let clay_yield_per_acre := good_yield_per_acre / 2
  let good_acres := 2 * acres / 3
  let clay_acres := acres / 3
  total_yield good_acres clay_acres good_yield_per_acre clay_yield_per_acre = 20000 :=
by
  sorry

end NUMINAMATH_GPT_jorges_total_yield_l2025_202558


namespace NUMINAMATH_GPT_f_neg_9_over_2_f_in_7_8_l2025_202540

noncomputable def f (x : ℝ) : ℝ :=
if 0 ≤ x ∧ x < 1 then x / (x + 1) else sorry

theorem f_neg_9_over_2 (h : ∀ x : ℝ, f (x + 2) = f x) (h_odd : ∀ x : ℝ, f (-x) = -f x) : 
  f (-9 / 2) = -1 / 3 :=
by
  have h_period : f (-9 / 2) = f (-1 / 2) := by
    sorry  -- Using periodicity property
  have h_odd1 : f (-1 / 2) = -f (1 / 2) := by
    sorry  -- Using odd function property
  have h_def : f (1 / 2) = 1 / 3 := by
    sorry  -- Using the definition of f(x) for x in [0, 1)
  rw [h_period, h_odd1, h_def]
  norm_num

theorem f_in_7_8 (h : ∀ x : ℝ, f (x + 2) = f x) (h_odd : ∀ x : ℝ, f (-x) = -f x) :
  ∀ x : ℝ, (7 < x ∧ x ≤ 8) → f x = - (x - 8) / (x - 9) :=
by
  intro x hx
  have h_period : f x = f (x - 8) := by
    sorry  -- Using periodicity property
  sorry  -- Apply the negative intervals and substitution to achieve the final form

end NUMINAMATH_GPT_f_neg_9_over_2_f_in_7_8_l2025_202540


namespace NUMINAMATH_GPT_calculate_fourth_quarter_shots_l2025_202553

-- Definitions based on conditions
def first_quarters_shots : ℕ := 20
def first_quarters_successful_shots : ℕ := 12
def third_quarter_shots : ℕ := 10
def overall_accuracy : ℚ := 46 / 100
def total_shots (n : ℕ) : ℕ := first_quarters_shots + third_quarter_shots + n
def total_successful_shots (n : ℕ) : ℚ := first_quarters_successful_shots + 3 + (4 / 10 * n)


-- Main theorem to prove
theorem calculate_fourth_quarter_shots (n : ℕ) (h : (total_successful_shots n) / (total_shots n) = overall_accuracy) : 
  n = 20 :=
by {
  sorry
}

end NUMINAMATH_GPT_calculate_fourth_quarter_shots_l2025_202553


namespace NUMINAMATH_GPT_problem_statement_l2025_202503

-- Definitions
def div_remainder (a b : ℕ) : ℕ × ℕ :=
  (a / b, a % b)

-- Conditions and question as Lean structures
def condition := ∀ (a b k : ℕ), k ≠ 0 → div_remainder (a * k) (b * k) = (a / b, (a % b) * k)
def question := div_remainder 4900 600 = div_remainder 49 6

-- Theorem stating the problem's conclusion
theorem problem_statement (cond : condition) : ¬question :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l2025_202503


namespace NUMINAMATH_GPT_gwen_received_more_money_from_mom_l2025_202598

theorem gwen_received_more_money_from_mom :
  let mom_money := 8
  let dad_money := 5
  mom_money - dad_money = 3 :=
by
  sorry

end NUMINAMATH_GPT_gwen_received_more_money_from_mom_l2025_202598


namespace NUMINAMATH_GPT_rugged_terrain_distance_ratio_l2025_202550

theorem rugged_terrain_distance_ratio (D k : ℝ) 
  (hD : D > 0) 
  (hk : k > 0) 
  (v_M v_P : ℝ) 
  (hm : v_M = 2 * k) 
  (hp : v_P = 3 * k)
  (v_Mr v_Pr : ℝ) 
  (hmr : v_Mr = k) 
  (hpr : v_Pr = 3 * k / 2) :
  ∀ (x y a b : ℝ), (x + y = D / 2) → (a + b = D / 2) → (y + b = 2 * D / 3) →
  (x / (2 * k) + y / k = a / (3 * k) + 2 * b / (3 * k)) → 
  (y / b = 1 / 3) := 
sorry

end NUMINAMATH_GPT_rugged_terrain_distance_ratio_l2025_202550


namespace NUMINAMATH_GPT_number_of_pieces_of_string_l2025_202593

theorem number_of_pieces_of_string (total_length piece_length : ℝ) (h1 : total_length = 60) (h2 : piece_length = 0.6) :
    total_length / piece_length = 100 := by
  sorry

end NUMINAMATH_GPT_number_of_pieces_of_string_l2025_202593


namespace NUMINAMATH_GPT_expected_lifetime_flashlight_l2025_202518

noncomputable def E (X : ℝ) : ℝ := sorry -- Define E as the expectation operator

variables (ξ η : ℝ) -- Define ξ and η as random variables representing lifetimes of blue and red bulbs
variable (h_exi : E ξ = 2) -- Given condition E ξ = 2

theorem expected_lifetime_flashlight (h_min : ∀ x y : ℝ, min x y ≤ x) :
  E (min ξ η) ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_expected_lifetime_flashlight_l2025_202518


namespace NUMINAMATH_GPT_part1_part2_l2025_202521

-- Definition of sets A and B
def A (a : ℝ) : Set ℝ := {x | a-1 < x ∧ x < a+1}
def B : Set ℝ := {x | x^2 - 4*x + 3 ≥ 0}

-- Theorem (Ⅰ)
theorem part1 (a : ℝ) : (A a ∩ B = ∅ ∧ A a ∪ B = Set.univ) → a = 2 :=
by
  sorry

-- Theorem (Ⅱ)
theorem part2 (a : ℝ) : (A a ⊆ B ∧ A a ≠ ∅) → (a ≤ 0 ∨ a ≥ 4) :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l2025_202521


namespace NUMINAMATH_GPT_aunt_masha_butter_usage_l2025_202546

theorem aunt_masha_butter_usage
  (x y : ℝ)
  (h1 : x + 10 * y = 600)
  (h2 : x = 5 * y) :
  (2 * x + 2 * y = 480) := 
by
  sorry

end NUMINAMATH_GPT_aunt_masha_butter_usage_l2025_202546


namespace NUMINAMATH_GPT_perfect_cubes_count_l2025_202585

theorem perfect_cubes_count : 
  Nat.card {n : ℕ | n^3 > 500 ∧ n^3 < 2000} = 5 :=
by
  sorry

end NUMINAMATH_GPT_perfect_cubes_count_l2025_202585


namespace NUMINAMATH_GPT_cos_angle_relation_l2025_202576

theorem cos_angle_relation (α : ℝ) (h : Real.sin (α + π / 6) = 1 / 3) : Real.cos (2 * α - 2 * π / 3) = -7 / 9 := by 
  sorry

end NUMINAMATH_GPT_cos_angle_relation_l2025_202576


namespace NUMINAMATH_GPT_potato_gun_distance_l2025_202545

noncomputable def length_of_football_field_in_yards : ℕ := 200
noncomputable def conversion_factor_yards_to_feet : ℕ := 3
noncomputable def length_of_football_field_in_feet : ℕ := length_of_football_field_in_yards * conversion_factor_yards_to_feet

noncomputable def dog_running_speed : ℕ := 400
noncomputable def time_for_dog_to_fetch_potato : ℕ := 9
noncomputable def total_distance_dog_runs : ℕ := dog_running_speed * time_for_dog_to_fetch_potato

noncomputable def actual_distance_to_potato : ℕ := total_distance_dog_runs / 2

noncomputable def distance_in_football_fields : ℕ := actual_distance_to_potato / length_of_football_field_in_feet

theorem potato_gun_distance :
  distance_in_football_fields = 3 :=
by
  sorry

end NUMINAMATH_GPT_potato_gun_distance_l2025_202545


namespace NUMINAMATH_GPT_intersection_A_B_l2025_202552

def A : Set ℝ := {y | ∃ x : ℝ, y = x^2 + 1 / (x^2 + 1) }
def B : Set ℝ := {x | 3 * x - 2 < 7}

theorem intersection_A_B : A ∩ B = Set.Ico 1 3 := 
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l2025_202552


namespace NUMINAMATH_GPT_similar_triangles_l2025_202506

theorem similar_triangles (y : ℝ) 
  (h₁ : 12 / y = 9 / 6) : y = 8 :=
by {
  -- solution here
  -- currently, we just provide the theorem statement as requested
  sorry
}

end NUMINAMATH_GPT_similar_triangles_l2025_202506


namespace NUMINAMATH_GPT_initial_weight_of_fish_l2025_202517

theorem initial_weight_of_fish (B F : ℝ) 
  (h1 : B + F = 54) 
  (h2 : B + F / 2 = 29) : 
  F = 50 := 
sorry

end NUMINAMATH_GPT_initial_weight_of_fish_l2025_202517


namespace NUMINAMATH_GPT_scientific_notation_3080000_l2025_202510

theorem scientific_notation_3080000 : (∃ (a : ℝ) (b : ℤ), 1 ≤ a ∧ a < 10 ∧ (3080000 : ℝ) = a * 10^b) ∧ (3080000 : ℝ) = 3.08 * 10^6 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_3080000_l2025_202510


namespace NUMINAMATH_GPT_total_cans_given_away_l2025_202504

noncomputable def total_cans_initial : ℕ := 2000
noncomputable def cans_taken_first_day : ℕ := 500
noncomputable def restocked_first_day : ℕ := 1500
noncomputable def people_second_day : ℕ := 1000
noncomputable def cans_per_person_second_day : ℕ := 2
noncomputable def restocked_second_day : ℕ := 3000

theorem total_cans_given_away :
  (cans_taken_first_day + (people_second_day * cans_per_person_second_day) = 2500) :=
by
  sorry

end NUMINAMATH_GPT_total_cans_given_away_l2025_202504


namespace NUMINAMATH_GPT_inequality_sin_cos_l2025_202577

theorem inequality_sin_cos 
  (a b : ℝ) (n : ℝ) (x : ℝ) 
  (ha : 0 < a) (hb : 0 < b) : 
  (a / (Real.sin x)^n) + (b / (Real.cos x)^n) ≥ (a^(2/(n+2)) + b^(2/(n+2)))^((n+2)/2) :=
sorry

end NUMINAMATH_GPT_inequality_sin_cos_l2025_202577


namespace NUMINAMATH_GPT_tangent_line_value_l2025_202543

theorem tangent_line_value {k : ℝ} 
  (h1 : ∃ x y : ℝ, x^2 + y^2 - 6*y + 8 = 0) 
  (h2 : ∃ P Q : ℝ, x^2 + y^2 - 6*y + 8 = 0 ∧ Q = k * P)
  (h3 : P * k < 0 ∧ P < 0 ∧ Q > 0) : 
  k = -2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_tangent_line_value_l2025_202543


namespace NUMINAMATH_GPT_prod_div_sum_le_square_l2025_202555

theorem prod_div_sum_le_square (m n : ℕ) (h : (m * n) ∣ (m + n)) : m + n ≤ n^2 := sorry

end NUMINAMATH_GPT_prod_div_sum_le_square_l2025_202555


namespace NUMINAMATH_GPT_find_common_chord_l2025_202516

-- Define the two circles
def C1 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 8*y - 8 = 0
def C2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y - 2 = 0

-- The common chord is the line we need to prove
def CommonChord (x y : ℝ) : Prop := x + 2*y - 1 = 0

-- The theorem stating that the common chord is the line x + 2*y - 1 = 0
theorem find_common_chord (x y : ℝ) (p : C1 x y ∧ C2 x y) : CommonChord x y :=
sorry

end NUMINAMATH_GPT_find_common_chord_l2025_202516


namespace NUMINAMATH_GPT_at_least_12_boxes_l2025_202514

theorem at_least_12_boxes (extra_boxes : Nat) : 
  let total_boxes := 12 + extra_boxes
  extra_boxes ≥ 0 → total_boxes ≥ 12 :=
by
  intros
  sorry

end NUMINAMATH_GPT_at_least_12_boxes_l2025_202514


namespace NUMINAMATH_GPT_find_abc_value_l2025_202544

noncomputable def abc_value_condition (a b c : ℝ) : Prop := 
  a + b + c = 4 ∧
  b * c + c * a + a * b = 5 ∧
  a^3 + b^3 + c^3 = 10

theorem find_abc_value (a b c : ℝ) (h : abc_value_condition a b c) : a * b * c = 2 := 
sorry

end NUMINAMATH_GPT_find_abc_value_l2025_202544


namespace NUMINAMATH_GPT_completion_time_C_l2025_202597

theorem completion_time_C (r_A r_B r_C : ℝ) 
  (h1 : r_A + r_B = 1 / 3) 
  (h2 : r_B + r_C = 1 / 3) 
  (h3 : r_A + r_C = 1 / 3) :
  1 / r_C = 6 :=
by
  sorry

end NUMINAMATH_GPT_completion_time_C_l2025_202597


namespace NUMINAMATH_GPT_rectangles_single_row_7_rectangles_grid_7_4_l2025_202502

def rectangles_in_single_row (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

theorem rectangles_single_row_7 :
  rectangles_in_single_row 7 = 28 :=
by
  -- Add the proof here
  sorry

def rectangles_in_grid (rows cols : ℕ) : ℕ :=
  ((cols + 1) * cols / 2) * ((rows + 1) * rows / 2)

theorem rectangles_grid_7_4 :
  rectangles_in_grid 4 7 = 280 :=
by
  -- Add the proof here
  sorry

end NUMINAMATH_GPT_rectangles_single_row_7_rectangles_grid_7_4_l2025_202502


namespace NUMINAMATH_GPT_minimize_cost_l2025_202507

noncomputable def cost_function (x : ℝ) : ℝ :=
  (1 / 2) * (x + 5)^2 + 1000 / (x + 5)

theorem minimize_cost :
  (∀ x, 2 ≤ x ∧ x ≤ 8 → cost_function x ≥ 150) ∧ cost_function 5 = 150 :=
by
  sorry

end NUMINAMATH_GPT_minimize_cost_l2025_202507


namespace NUMINAMATH_GPT_largest_value_of_x_l2025_202559

theorem largest_value_of_x (x : ℝ) (hx : x / 3 + 1 / (7 * x) = 1 / 2) : 
  x = (21 + Real.sqrt 105) / 28 := 
sorry

end NUMINAMATH_GPT_largest_value_of_x_l2025_202559


namespace NUMINAMATH_GPT_original_price_of_shoes_l2025_202584

theorem original_price_of_shoes (P : ℝ) (h1 : 0.25 * P = 51) : P = 204 := 
by 
  sorry

end NUMINAMATH_GPT_original_price_of_shoes_l2025_202584


namespace NUMINAMATH_GPT_james_hours_to_work_l2025_202580

theorem james_hours_to_work :
  let meat_cost := 20 * 5
  let fruits_vegetables_cost := 15 * 4
  let bread_cost := 60 * 1.5
  let janitorial_cost := 10 * (10 * 1.5)
  let total_cost := meat_cost + fruits_vegetables_cost + bread_cost + janitorial_cost
  let hourly_wage := 8
  let hours_to_work := total_cost / hourly_wage
  hours_to_work = 50 :=
by 
  sorry

end NUMINAMATH_GPT_james_hours_to_work_l2025_202580


namespace NUMINAMATH_GPT_existence_of_special_numbers_l2025_202582

theorem existence_of_special_numbers :
  ∃ (N : Finset ℕ), N.card = 1998 ∧ 
  ∀ (a b : ℕ), a ∈ N → b ∈ N → a ≠ b → a * b ∣ (a - b)^2 :=
sorry

end NUMINAMATH_GPT_existence_of_special_numbers_l2025_202582


namespace NUMINAMATH_GPT_find_prime_triplet_l2025_202519

open Nat

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, 2 ≤ m → m ≤ n / 2 → (m ∣ n) → False

theorem find_prime_triplet :
  ∃ p q r : ℕ, is_prime p ∧ is_prime q ∧ is_prime r ∧ 
  (p + q = r) ∧ 
  (∃ k : ℕ, (r - p) * (q - p) - 27 * p = k * k) ∧ 
  (p = 2 ∧ q = 29 ∧ r = 31) := by
  sorry

end NUMINAMATH_GPT_find_prime_triplet_l2025_202519


namespace NUMINAMATH_GPT_polygon_has_9_diagonals_has_6_sides_l2025_202563

theorem polygon_has_9_diagonals_has_6_sides :
  ∀ (n : ℕ), (∃ D : ℕ, D = n * (n - 3) / 2 ∧ D = 9) → n = 6 := 
by
  sorry

end NUMINAMATH_GPT_polygon_has_9_diagonals_has_6_sides_l2025_202563


namespace NUMINAMATH_GPT_variance_proof_l2025_202528

noncomputable def calculate_mean (scores : List ℝ) : ℝ :=
  (scores.sum / scores.length)

noncomputable def calculate_variance (scores : List ℝ) : ℝ :=
  let mean := calculate_mean scores
  (scores.map (λ x => (x - mean)^2)).sum / scores.length

def scores_A : List ℝ := [8, 6, 9, 5, 10, 7, 4, 7, 9, 5]
def scores_B : List ℝ := [7, 6, 5, 8, 6, 9, 6, 8, 8, 7]

noncomputable def variance_A : ℝ := calculate_variance scores_A
noncomputable def variance_B : ℝ := calculate_variance scores_B

theorem variance_proof :
  variance_A = 3.6 ∧ variance_B = 1.4 ∧ variance_B < variance_A :=
by
  -- proof steps - use sorry to skip the proof
  sorry

end NUMINAMATH_GPT_variance_proof_l2025_202528


namespace NUMINAMATH_GPT_range_of_a_l2025_202591

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = x^2 + a * x + 2)
  (h2 : ∀ y, (∃ x, y = f (f x)) ↔ (∃ x, y = f x)) : a ≥ 4 ∨ a ≤ -2 := 
sorry

end NUMINAMATH_GPT_range_of_a_l2025_202591


namespace NUMINAMATH_GPT_total_cost_of_hotel_stay_l2025_202587

-- Define the necessary conditions
def cost_per_night_per_person : ℕ := 40
def number_of_people : ℕ := 3
def number_of_nights : ℕ := 3

-- State the problem
theorem total_cost_of_hotel_stay :
  (cost_per_night_per_person * number_of_people * number_of_nights) = 360 := by
  sorry

end NUMINAMATH_GPT_total_cost_of_hotel_stay_l2025_202587


namespace NUMINAMATH_GPT_unique_common_element_l2025_202574

variable (A B : Set ℝ)
variable (a : ℝ)

theorem unique_common_element :
  A = {1, 3, a} → 
  B = {4, 5} →
  A ∩ B = {4} →
  a = 4 := 
by
  intro hA hB hAB
  sorry

end NUMINAMATH_GPT_unique_common_element_l2025_202574


namespace NUMINAMATH_GPT_minimum_value_expr_l2025_202527

variable (a b : ℝ)

theorem minimum_value_expr (h1 : 0 < a) (h2 : 0 < b) : 
  (a + 1 / b) * (a + 1 / b - 1009) + (b + 1 / a) * (b + 1 / a - 1009) ≥ -509004.5 :=
sorry

end NUMINAMATH_GPT_minimum_value_expr_l2025_202527


namespace NUMINAMATH_GPT_trail_mix_total_weight_l2025_202523

noncomputable def peanuts : ℝ := 0.16666666666666666
noncomputable def chocolate_chips : ℝ := 0.16666666666666666
noncomputable def raisins : ℝ := 0.08333333333333333

theorem trail_mix_total_weight :
  peanuts + chocolate_chips + raisins = 0.41666666666666663 :=
by
  unfold peanuts chocolate_chips raisins
  sorry

end NUMINAMATH_GPT_trail_mix_total_weight_l2025_202523


namespace NUMINAMATH_GPT_number_of_boys_l2025_202525

-- Definitions based on conditions
def students_in_class : ℕ := 30
def cups_brought_total : ℕ := 90
def cups_per_boy : ℕ := 5

-- Definition of boys and girls, with a constraint from the conditions
variable (B : ℕ)
def girls_in_class (B : ℕ) : ℕ := 2 * B

-- Properties from the conditions
axiom h1 : B + girls_in_class B = students_in_class
axiom h2 : B * cups_per_boy = cups_brought_total - (students_in_class - B) * 0 -- Assume no girl brought any cup

-- We state the question as a theorem to be proved
theorem number_of_boys (B : ℕ) : B = 10 :=
by
  sorry

end NUMINAMATH_GPT_number_of_boys_l2025_202525


namespace NUMINAMATH_GPT_import_tax_calculation_l2025_202536

def import_tax_rate : ℝ := 0.07
def excess_value_threshold : ℝ := 1000
def total_value_item : ℝ := 2610
def correct_import_tax : ℝ := 112.7

theorem import_tax_calculation :
  (total_value_item - excess_value_threshold) * import_tax_rate = correct_import_tax :=
by
  sorry

end NUMINAMATH_GPT_import_tax_calculation_l2025_202536


namespace NUMINAMATH_GPT_exists_disk_of_radius_one_containing_1009_points_l2025_202534

theorem exists_disk_of_radius_one_containing_1009_points
  (points : Fin 2017 → ℝ × ℝ)
  (h : ∀ (a b c : Fin 2017), (dist (points a) (points b) < 1) ∨ (dist (points b) (points c) < 1) ∨ (dist (points c) (points a) < 1)) :
  ∃ (center : ℝ × ℝ), ∃ (sub_points : Finset (Fin 2017)), sub_points.card ≥ 1009 ∧ ∀ p ∈ sub_points, dist (center) (points p) ≤ 1 :=
sorry

end NUMINAMATH_GPT_exists_disk_of_radius_one_containing_1009_points_l2025_202534


namespace NUMINAMATH_GPT_solution_of_equation_l2025_202596

def solve_equation (x : ℚ) : Prop := 
  (x^2 + 3 * x + 4) / (x + 5) = x + 6

theorem solution_of_equation : solve_equation (-13/4) := 
by
  sorry

end NUMINAMATH_GPT_solution_of_equation_l2025_202596


namespace NUMINAMATH_GPT_mutually_exclusive_white_ball_events_l2025_202578

-- Definitions of persons and balls
inductive Person | A | B | C
inductive Ball | red | black | white

-- Definitions of events
def eventA (dist : Person → Ball) : Prop := dist Person.A = Ball.white
def eventB (dist : Person → Ball) : Prop := dist Person.B = Ball.white

theorem mutually_exclusive_white_ball_events (dist : Person → Ball) :
  (eventA dist → ¬eventB dist) :=
by
  sorry

end NUMINAMATH_GPT_mutually_exclusive_white_ball_events_l2025_202578


namespace NUMINAMATH_GPT_min_x_squared_plus_y_squared_l2025_202515

theorem min_x_squared_plus_y_squared (x y : ℝ) (h : (x + 5) * (y - 5) = 0) : x^2 + y^2 ≥ 50 := by
  sorry

end NUMINAMATH_GPT_min_x_squared_plus_y_squared_l2025_202515


namespace NUMINAMATH_GPT_percentage_of_hindu_boys_l2025_202509

-- Define the total number of boys in the school
def total_boys := 700

-- Define the percentage of Muslim boys
def muslim_percentage := 44 / 100

-- Define the percentage of Sikh boys
def sikh_percentage := 10 / 100

-- Define the number of boys from other communities
def other_communities_boys := 126

-- State the main theorem to prove the percentage of Hindu boys
theorem percentage_of_hindu_boys (h1 : total_boys = 700)
                                 (h2 : muslim_percentage = 44 / 100)
                                 (h3 : sikh_percentage = 10 / 100)
                                 (h4 : other_communities_boys = 126) : 
                                 ((total_boys - (total_boys * muslim_percentage + total_boys * sikh_percentage + other_communities_boys)) / total_boys) * 100 = 28 :=
by {
  sorry
}

end NUMINAMATH_GPT_percentage_of_hindu_boys_l2025_202509


namespace NUMINAMATH_GPT_circles_internally_tangent_l2025_202511

theorem circles_internally_tangent :
  ∀ (x y : ℝ),
  (x^2 + y^2 - 6 * x + 4 * y + 12 = 0) ∧ (x^2 + y^2 - 14 * x - 2 * y + 14 = 0) →
  ∃ (C1 C2 : ℝ × ℝ) (r1 r2 : ℝ),
  C1 = (3, -2) ∧ r1 = 1 ∧
  C2 = (7, 1) ∧ r2 = 6 ∧
  dist C1 C2 = r2 - r1 :=
by
  sorry

end NUMINAMATH_GPT_circles_internally_tangent_l2025_202511


namespace NUMINAMATH_GPT_smallest_positive_integer_l2025_202531

theorem smallest_positive_integer (m n : ℤ) : ∃ k : ℕ, k > 0 ∧ (∃ m n : ℤ, k = 5013 * m + 111111 * n) ∧ k = 3 :=
by {
  sorry 
}

end NUMINAMATH_GPT_smallest_positive_integer_l2025_202531


namespace NUMINAMATH_GPT_months_passed_l2025_202573

-- Let's define our conditions in mathematical terms
def received_bones (months : ℕ) : ℕ := 10 * months
def buried_bones : ℕ := 42
def available_bones : ℕ := 8
def total_bones (months : ℕ) : Prop := received_bones months = buried_bones + available_bones

-- We need to prove that the number of months (x) satisfies the condition
theorem months_passed (x : ℕ) : total_bones x → x = 5 :=
by
  sorry

end NUMINAMATH_GPT_months_passed_l2025_202573


namespace NUMINAMATH_GPT_find_y_coordinate_of_Q_l2025_202512

noncomputable def y_coordinate_of_Q 
  (P R T S : ℝ × ℝ) (Q : ℝ × ℝ) (areaPentagon areaSquare : ℝ) : Prop :=
  P = (0, 0) ∧ 
  R = (0, 5) ∧ 
  T = (6, 0) ∧ 
  S = (6, 5) ∧ 
  Q.fst = 3 ∧ 
  areaSquare = 25 ∧ 
  areaPentagon = 50 ∧ 
  (1 / 2) * 6 * (Q.snd - 5) + areaSquare = areaPentagon

theorem find_y_coordinate_of_Q : 
  ∃ y_Q : ℝ, y_coordinate_of_Q (0, 0) (0, 5) (6, 0) (6, 5) (3, y_Q) 50 25 ∧ y_Q = 40 / 3 :=
sorry

end NUMINAMATH_GPT_find_y_coordinate_of_Q_l2025_202512


namespace NUMINAMATH_GPT_triangle_perimeter_l2025_202561

-- Define the given quadratic equation
def quadratic_equation (m : ℝ) (x : ℝ) : ℝ :=
  x^2 - (5 + m) * x + 5 * m

-- Define the isosceles triangle with sides given by the roots of the equation
def is_isosceles_triangle (a b c : ℝ) : Prop :=
  a = b ∨ b = c ∨ c = a

-- Defining the fact that 2 is a root of the given quadratic equation with an unknown m
lemma two_is_root (m : ℝ) : quadratic_equation m 2 = 0 := sorry

-- Prove that the perimeter of triangle ABC is 12 given the conditions
theorem triangle_perimeter (α β γ : ℝ) (m : ℝ) (h1 : quadratic_equation m α = 0) 
  (h2 : quadratic_equation m β = 0) 
  (h3 : is_isosceles_triangle α β γ) : α + β + γ = 12 := sorry

end NUMINAMATH_GPT_triangle_perimeter_l2025_202561


namespace NUMINAMATH_GPT_sequence_value_2_l2025_202565

/-- 
Given the following sequence:
1 = 6
3 = 18
4 = 24
5 = 30

The sequence follows the pattern that for all n ≠ 6, n is mapped to n * 6.
Prove that the value of the 2nd term in the sequence is 12.
-/

theorem sequence_value_2 (a : ℕ → ℕ) 
  (h1 : a 1 = 6) 
  (h3 : a 3 = 18) 
  (h4 : a 4 = 24) 
  (h5 : a 5 = 30) 
  (h_pattern : ∀ n, n ≠ 6 → a n = n * 6) :
  a 2 = 12 :=
by
  sorry

end NUMINAMATH_GPT_sequence_value_2_l2025_202565


namespace NUMINAMATH_GPT_smallest_b_of_factored_quadratic_l2025_202505

theorem smallest_b_of_factored_quadratic (r s : ℕ) (h1 : r * s = 1620) : (r + s) = 84 :=
sorry

end NUMINAMATH_GPT_smallest_b_of_factored_quadratic_l2025_202505


namespace NUMINAMATH_GPT_sum_gn_eq_one_third_l2025_202560

noncomputable def g (n : ℕ) : ℝ :=
  ∑' i : ℕ, if i ≥ 3 then 1 / (i ^ n) else 0

theorem sum_gn_eq_one_third :
  (∑' n : ℕ, if n ≥ 3 then g n else 0) = 1 / 3 := 
by sorry

end NUMINAMATH_GPT_sum_gn_eq_one_third_l2025_202560


namespace NUMINAMATH_GPT_marcella_pairs_l2025_202564

theorem marcella_pairs (pairs_initial : ℕ) (shoes_lost : ℕ) (h1 : pairs_initial = 50) (h2 : shoes_lost = 15) :
  ∃ pairs_left : ℕ, pairs_left = 35 := 
by
  existsi 35
  sorry

end NUMINAMATH_GPT_marcella_pairs_l2025_202564


namespace NUMINAMATH_GPT_cats_more_than_spinsters_l2025_202539

def ratio (a b : ℕ) := ∃ k : ℕ, a = b * k

theorem cats_more_than_spinsters (S C : ℕ) (h1 : ratio 2 9) (h2 : S = 12) (h3 : 2 * C = 108) :
  C - S = 42 := by 
  sorry

end NUMINAMATH_GPT_cats_more_than_spinsters_l2025_202539
