import Mathlib

namespace books_on_each_shelf_l2064_206479

theorem books_on_each_shelf (M P x : ℕ) (h1 : 3 * M + 5 * P = 72) (h2 : M = x) (h3 : P = x) : x = 9 :=
by
  sorry

end books_on_each_shelf_l2064_206479


namespace sum_ratio_l2064_206471

def arithmetic_sequence (a_1 d : ℚ) (n : ℕ) : ℚ :=
  a_1 + (n - 1) * d

def S_n (a_1 d : ℚ) (n : ℕ) : ℚ :=
  n * (2 * a_1 + (n - 1) * d) / 2 -- sum of first n terms of arithmetic sequence

theorem sum_ratio (a_1 d : ℚ) (h : 13 * (a_1 + 6 * d) = 7 * (a_1 + 3 * d)) :
  S_n a_1 d 13 / S_n a_1 d 7 = 1 :=
by
  -- Proof omitted
  sorry

end sum_ratio_l2064_206471


namespace inspectors_in_group_B_l2064_206434

theorem inspectors_in_group_B
  (a b : ℕ)  -- a: number of original finished products, b: daily production
  (A_inspectors := 8)  -- Number of inspectors in group A
  (total_days := 5) -- Group B inspects in 5 days
  (inspects_same_speed : (2 * a + 2 * 2 * b) * total_days/A_inspectors = (2 * a + 2 * 5 * b) * (total_days/3))
  : ∃ (B_inspectors : ℕ), B_inspectors = 12 := 
by
  sorry

end inspectors_in_group_B_l2064_206434


namespace solution_in_quadrant_I_l2064_206418

theorem solution_in_quadrant_I (k : ℝ) :
  ∃ x y : ℝ, (2 * x - y = 5 ∧ k * x + 2 * y = 4 ∧ x > 0 ∧ y > 0) ↔ -4 < k ∧ k < 8 / 5 :=
by
  sorry

end solution_in_quadrant_I_l2064_206418


namespace lcm_of_denominators_l2064_206443

theorem lcm_of_denominators (x : ℕ) [NeZero x] : Nat.lcm (Nat.lcm x (2 * x)) (3 * x^2) = 6 * x^2 :=
by
  sorry

end lcm_of_denominators_l2064_206443


namespace sum_series_eq_one_quarter_l2064_206461

noncomputable def series_term (n : ℕ) : ℝ :=
  3^n / (1 + 3^n + 3^(n + 1) + 3^(2 * n + 1))

theorem sum_series_eq_one_quarter : 
  (∑' n, series_term (n + 1)) = 1 / 4 :=
by
  sorry

end sum_series_eq_one_quarter_l2064_206461


namespace correct_option_l2064_206441

theorem correct_option (x y a b : ℝ) :
  ((x + 2 * y) ^ 2 ≠ x ^ 2 + 4 * y ^ 2) ∧
  ((-2 * (a ^ 3)) ^ 2 = 4 * (a ^ 6)) ∧
  (-6 * (a ^ 2) * (b ^ 5) + a * b ^ 2 ≠ -6 * a * (b ^ 3)) ∧
  (2 * (a ^ 2) * 3 * (a ^ 3) ≠ 6 * (a ^ 6)) :=
by
  sorry

end correct_option_l2064_206441


namespace find_x_l2064_206425

noncomputable def solution_x (m n y : ℝ) (m_gt_3n : m > 3 * n) : ℝ :=
  (n * m) / (m + n)

theorem find_x (m n y : ℝ) (m_gt_3n : m > 3 * n) :
  let initial_acid := m * (m / 100)
  let final_volume := m + (solution_x m n y m_gt_3n) + y
  let final_acid := (m - n) / 100 * final_volume
  initial_acid = final_acid → 
  solution_x m n y m_gt_3n = (n * m) / (m + n) :=
by sorry

end find_x_l2064_206425


namespace largest_expression_is_d_l2064_206464

def expr_a := 3 + 0 + 4 + 8
def expr_b := 3 * 0 + 4 + 8
def expr_c := 3 + 0 * 4 + 8
def expr_d := 3 + 0 + 4 * 8
def expr_e := 3 * 0 * 4 * 8
def expr_f := (3 + 0 + 4) / 8

theorem largest_expression_is_d : 
  expr_d = 35 ∧ 
  expr_a = 15 ∧ 
  expr_b = 12 ∧ 
  expr_c = 11 ∧ 
  expr_e = 0 ∧ 
  expr_f = 7 / 8 ∧
  35 > 15 ∧ 
  35 > 12 ∧ 
  35 > 11 ∧ 
  35 > 0 ∧ 
  35 > 7 / 8 := 
by
  sorry

end largest_expression_is_d_l2064_206464


namespace maximize_S_n_l2064_206456

def a1 : ℚ := 5
def d : ℚ := -5 / 7

def S_n (n : ℕ) : ℚ :=
  (n * (2 * a1 + (n - 1) * d)) / 2

theorem maximize_S_n :
  (∃ n : ℕ, (S_n n ≥ S_n (n - 1)) ∧ (S_n n ≥ S_n (n + 1))) →
  (n = 7 ∨ n = 8) :=
sorry

end maximize_S_n_l2064_206456


namespace transportation_cost_l2064_206436

theorem transportation_cost 
  (cost_per_kg : ℝ) 
  (weight_communication : ℝ) 
  (weight_sensor : ℝ) 
  (extra_sensor_cost_percentage : ℝ) 
  (cost_communication : ℝ)
  (basic_cost_sensor : ℝ)
  (extra_cost_sensor : ℝ)
  (total_cost : ℝ) : 
  cost_per_kg = 25000 → 
  weight_communication = 0.5 → 
  weight_sensor = 0.3 → 
  extra_sensor_cost_percentage = 0.10 →
  cost_communication = weight_communication * cost_per_kg →
  basic_cost_sensor = weight_sensor * cost_per_kg →
  extra_cost_sensor = extra_sensor_cost_percentage * basic_cost_sensor →
  total_cost = cost_communication + basic_cost_sensor + extra_cost_sensor →
  total_cost = 20750 :=
by sorry

end transportation_cost_l2064_206436


namespace probability_of_darkness_l2064_206417

theorem probability_of_darkness (rev_per_min : ℕ) (stay_in_dark_time : ℕ) (revolution_time : ℕ) (stay_fraction : ℕ → ℚ) :
  rev_per_min = 2 →
  stay_in_dark_time = 10 →
  revolution_time = 60 / rev_per_min →
  stay_fraction stay_in_dark_time / revolution_time = 1 / 3 :=
by
  sorry

end probability_of_darkness_l2064_206417


namespace problem1_coefficient_of_x_problem2_maximum_coefficient_term_l2064_206411

-- Problem 1: Coefficient of x term
theorem problem1_coefficient_of_x (n : ℕ) 
  (A : ℕ := (3 + 1)^n) 
  (B : ℕ := 2^n) 
  (h1 : A + B = 272) 
  : true :=  -- Replacing true with actual condition
by sorry

-- Problem 2: Term with maximum coefficient
theorem problem2_maximum_coefficient_term (n : ℕ)
  (h : 1 + n + (n * (n - 1)) / 2 = 79) 
  : true :=  -- Replacing true with actual condition
by sorry

end problem1_coefficient_of_x_problem2_maximum_coefficient_term_l2064_206411


namespace inequality_proof_l2064_206467

open Real

variable (a b c : ℝ)

theorem inequality_proof
  (h1 : 0 < a)
  (h2 : 0 < b)
  (h3 : 0 < c) :
  sqrt (a * b * c) * (sqrt a + sqrt b + sqrt c) + (a + b + c) ^ 2 ≥ 
  4 * sqrt (3 * a * b * c * (a + b + c)) :=
by sorry

end inequality_proof_l2064_206467


namespace unique_prime_solution_l2064_206493

theorem unique_prime_solution :
  ∃! (p : ℕ), Prime p ∧ (∃ (k : ℤ), 2 * (p ^ 4) - 7 * (p ^ 2) + 1 = k ^ 2) := 
sorry

end unique_prime_solution_l2064_206493


namespace trees_in_garden_l2064_206400

theorem trees_in_garden (yard_length : ℕ) (distance_between_trees : ℕ) (H1 : yard_length = 400) (H2 : distance_between_trees = 16) : 
  (yard_length / distance_between_trees) + 1 = 26 :=
by
  -- Adding sorry to skip the proof
  sorry

end trees_in_garden_l2064_206400


namespace part1_part2_l2064_206497

noncomputable def f (x : ℝ) : ℝ := Real.cos x * Real.cos (x - Real.pi / 3)

theorem part1 : f (2 * Real.pi / 3) = -1 / 4 :=
by
  sorry

theorem part2 : {x | f x < 1 / 4} = { x | ∃ k : ℤ, k * Real.pi + 5 * Real.pi / 12 < x ∧ x < k * Real.pi + 11 * Real.pi / 12 } :=
by
  sorry

end part1_part2_l2064_206497


namespace distribution_of_collection_items_l2064_206413

-- Declaring the collections
structure Collection where
  stickers : Nat
  baseball_cards : Nat
  keychains : Nat
  stamps : Nat

-- Defining the individual collections based on the conditions
def Karl : Collection := { stickers := 25, baseball_cards := 15, keychains := 5, stamps := 10 }
def Ryan : Collection := { stickers := Karl.stickers + 20, baseball_cards := Karl.baseball_cards - 10, keychains := Karl.keychains + 2, stamps := Karl.stamps }
def Ben_scenario1 : Collection := { stickers := Ryan.stickers - 10, baseball_cards := (Ryan.baseball_cards / 2), keychains := Karl.keychains * 2, stamps := Karl.stamps + 5 }

-- Total number of items in the collection
def total_items_scenario1 :=
  Karl.stickers + Karl.baseball_cards + Karl.keychains + Karl.stamps +
  Ryan.stickers + Ryan.baseball_cards + Ryan.keychains + Ryan.stamps +
  Ben_scenario1.stickers + Ben_scenario1.baseball_cards + Ben_scenario1.keychains + Ben_scenario1.stamps

-- The proof statement
theorem distribution_of_collection_items :
  total_items_scenario1 = 184 ∧ total_items_scenario1 % 4 = 0 → (184 / 4 = 46) := 
by
  sorry

end distribution_of_collection_items_l2064_206413


namespace ratio_of_ages_l2064_206428

theorem ratio_of_ages (F C : ℕ) (h1 : F = C) (h2 : F = 75) :
  (C + 5 * 15) / (F + 15) = 5 / 3 :=
by
  sorry

end ratio_of_ages_l2064_206428


namespace complementary_angle_l2064_206460

theorem complementary_angle (α : ℝ) (h : α = 35 + 30 / 60) : 90 - α = 54 + 30 / 60 :=
by
  sorry

end complementary_angle_l2064_206460


namespace quadratic_solution_l2064_206415

theorem quadratic_solution :
  ∀ x : ℝ, x^2 = 4 ↔ x = 2 ∨ x = -2 := sorry

end quadratic_solution_l2064_206415


namespace max_value_of_z_l2064_206421

theorem max_value_of_z (k : ℝ) (x y : ℝ)
  (h1 : x + 2 * y - 1 ≥ 0)
  (h2 : x - y ≥ 0)
  (h3 : 0 ≤ x)
  (h4 : x ≤ k)
  (h5 : ∀ x y, x + 2 * y - 1 ≥ 0 ∧ x - y ≥ 0 ∧ 0 ≤ x ∧ x ≤ k → x + k * y ≥ -2) :
  ∃ (x y : ℝ), x + k * y = 20 := 
by
  sorry

end max_value_of_z_l2064_206421


namespace eva_fruit_diet_l2064_206416

noncomputable def dietary_requirements : Prop :=
  ∃ (days_in_week : ℕ) (days_in_month : ℕ) (apples : ℕ) (bananas : ℕ) (pears : ℕ) (oranges : ℕ),
    days_in_week = 7 ∧
    days_in_month = 30 ∧
    apples = 2 * days_in_week ∧
    bananas = days_in_week / 2 ∧
    pears = 4 ∧
    oranges = days_in_month / 3 ∧
    apples = 14 ∧
    bananas = 4 ∧
    pears = 4 ∧
    oranges = 10

theorem eva_fruit_diet : dietary_requirements :=
sorry

end eva_fruit_diet_l2064_206416


namespace captain_age_l2064_206408

-- Definitions
def num_team_members : ℕ := 11
def total_team_age : ℕ := 11 * 24
def total_age_remainder := 9 * (24 - 1)
def combined_age_of_captain_and_keeper := total_team_age - total_age_remainder

-- The actual proof statement
theorem captain_age (C : ℕ) (W : ℕ) 
  (hW : W = C + 5)
  (h_total_team : total_team_age = 264)
  (h_total_remainders : total_age_remainder = 207)
  (h_combined_age : combined_age_of_captain_and_keeper = 57) :
  C = 26 :=
by sorry

end captain_age_l2064_206408


namespace sqrt_of_1024_is_32_l2064_206494

theorem sqrt_of_1024_is_32 (y : ℕ) (h1 : y > 0) (h2 : y^2 = 1024) : y = 32 :=
sorry

end sqrt_of_1024_is_32_l2064_206494


namespace find_k_l2064_206427

variable (m n k : ℚ)

def line_eq (x y : ℚ) : Prop := x - (5/2 : ℚ) * y + 1 = 0

theorem find_k (h1 : line_eq m n) (h2 : line_eq (m + 1/2) (n + 1/k)) : k = 3/5 := by
  sorry

end find_k_l2064_206427


namespace price_of_70_cans_l2064_206401

noncomputable def regular_price_per_can : ℝ := 0.55
noncomputable def discount_rate_case : ℝ := 0.25
noncomputable def bulk_discount_rate : ℝ := 0.10
noncomputable def cans_per_case : ℕ := 24
noncomputable def total_cans_purchased : ℕ := 70

theorem price_of_70_cans :
  let discounted_price_per_can := regular_price_per_can * (1 - discount_rate_case)
  let discounted_price_for_cases := 48 * discounted_price_per_can
  let bulk_discount := if 70 >= 3 * cans_per_case then discounted_price_for_cases * bulk_discount_rate else 0
  let final_price_for_cases := discounted_price_for_cases - bulk_discount
  let additional_cans := total_cans_purchased % cans_per_case
  let price_for_additional_cans := additional_cans * discounted_price_per_can
  final_price_for_cases + price_for_additional_cans = 26.895 :=
by sorry

end price_of_70_cans_l2064_206401


namespace red_sea_glass_pieces_l2064_206431

theorem red_sea_glass_pieces (R : ℕ) 
    (h_bl : ∃ g : ℕ, g = 12) 
    (h_rose_red : ∃ r_b : ℕ, r_b = 9)
    (h_rose_blue : ∃ b : ℕ, b = 11) 
    (h_dorothy_red : 2 * (R + 9) + 3 * 11 = 57) : R = 3 :=
  by
    sorry

end red_sea_glass_pieces_l2064_206431


namespace product_of_fractions_l2064_206459

theorem product_of_fractions : (2 / 9) * (5 / 11) = 10 / 99 := 
by
  sorry

end product_of_fractions_l2064_206459


namespace fourth_person_height_l2064_206403

theorem fourth_person_height (h : ℝ)
  (h2 : h + 2 = h₂)
  (h3 : h + 4 = h₃)
  (h4 : h + 10 = h₄)
  (average_height : (h + h₂ + h₃ + h₄) / 4 = 77) :
  h₄ = 83 :=
by
  sorry

end fourth_person_height_l2064_206403


namespace new_rectangle_area_l2064_206450

theorem new_rectangle_area :
  let a := 3
  let b := 4
  let diagonal := Real.sqrt (a^2 + b^2)
  let sum_of_sides := a + b
  let area := diagonal * sum_of_sides
  area = 35 :=
by
  sorry

end new_rectangle_area_l2064_206450


namespace compare_fx_l2064_206444

noncomputable def f (x : ℝ) (b : ℝ) (c : ℝ) : ℝ :=
  x^2 - b * x + c

theorem compare_fx (b c : ℝ) (x : ℝ) (h1 : ∀ x : ℝ, f (1 - x) b c = f (1 + x) b c) (h2 : f 0 b c = 3) :
  f (2^x) b c ≤ f (3^x) b c :=
by
  sorry

end compare_fx_l2064_206444


namespace rectangle_lengths_l2064_206482

theorem rectangle_lengths (side_length : ℝ) (width1 width2: ℝ) (length1 length2 : ℝ) 
  (h1 : side_length = 6) 
  (h2 : width1 = 4) 
  (h3 : width2 = 3)
  (h_area_square : side_length * side_length = 36)
  (h_area_rectangle1 : width1 * length1 = side_length * side_length)
  (h_area_rectangle2 : width2 * length2 = (1 / 2) * (side_length * side_length)) :
  length1 = 9 ∧ length2 = 6 :=
by
  sorry

end rectangle_lengths_l2064_206482


namespace two_b_leq_a_plus_c_l2064_206437

variable (t a b c : ℝ)

theorem two_b_leq_a_plus_c (ht : t > 1)
  (h : 2 / Real.log t / Real.log b = 1 / Real.log t / Real.log a + 1 / Real.log t / Real.log c) :
  2 * b ≤ a + c := by sorry

end two_b_leq_a_plus_c_l2064_206437


namespace ratio_of_a_over_b_l2064_206488

noncomputable def f (a b x : ℝ) : ℝ := x^3 + a * x^2 + b * x - a^2 - 7 * a

theorem ratio_of_a_over_b (a b : ℝ) (h_max : ∀ x : ℝ, f a b x ≤ 10)
  (h_cond1 : f a b 1 = 10) (h_cond2 : (deriv (f a b)) 1 = 0) :
  a / b = -2/3 :=
sorry

end ratio_of_a_over_b_l2064_206488


namespace problem_I_problem_II_problem_III_l2064_206439

noncomputable def f (a x : ℝ) := a * x * Real.exp x
noncomputable def f' (a x : ℝ) := a * (1 + x) * Real.exp x

theorem problem_I (a : ℝ) (h : a ≠ 0) :
  (if a > 0 then ∀ x, (f' a x > 0 ↔ x > -1) ∧ (f' a x < 0 ↔ x < -1)
  else ∀ x, (f' a x > 0 ↔ x < -1) ∧ (f' a x < 0 ↔ x > -1)) :=
sorry

theorem problem_II (h : ∃ a : ℝ, a = 1) :
  ∃ (x : ℝ) (y : ℝ), x = -1 ∧ f 1 (-1) = -1 / Real.exp 1 ∧ ¬ ∃ y, ∀ x, y = f 1 x ∧ (f' 1 x) < 0 :=
sorry

theorem problem_III (h : ∃ m : ℝ, f 1 m = e * m * Real.exp m ∧ f' 1 m = e * (1 + m) * Real.exp m) :
  ∃ a : ℝ, a = 1 / 2 :=
sorry

end problem_I_problem_II_problem_III_l2064_206439


namespace smallest_next_divisor_l2064_206426

noncomputable def is_even (n : ℕ) : Prop := n % 2 = 0

noncomputable def is_4_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

noncomputable def has_divisor_323 (n : ℕ) : Prop := 323 ∣ n

theorem smallest_next_divisor (n : ℕ) (h1 : is_even n) (h2 : is_4_digit n) (h3 : has_divisor_323 n) :
  ∃ m : ℕ, m > 323 ∧ m ∣ n ∧ (∀ k : ℕ, k > 323 ∧ k < m → ¬ k ∣ n) ∧ m = 340 :=
sorry

end smallest_next_divisor_l2064_206426


namespace profit_percentage_l2064_206474

theorem profit_percentage (SP CP : ℝ) (hs : SP = 270) (hc : CP = 225) : 
  ((SP - CP) / CP) * 100 = 20 :=
by
  rw [hs, hc]
  sorry  -- The proof will go here

end profit_percentage_l2064_206474


namespace choose_questions_l2064_206438

theorem choose_questions (q : ℕ) (last : ℕ) (total : ℕ) (chosen : ℕ) 
  (condition : q ≥ 3) 
  (n : last = 5) 
  (m : total = 10) 
  (k : chosen = 6) : 
  ∃ (ways : ℕ), ways = 155 := 
by
  sorry

end choose_questions_l2064_206438


namespace find_constants_l2064_206487

theorem find_constants
  (a_1 a_2 : ℚ)
  (h1 : 3 * a_1 - 3 * a_2 = 0)
  (h2 : 4 * a_1 + 7 * a_2 = 5) :
  a_1 = 5 / 11 ∧ a_2 = 5 / 11 :=
by
  sorry

end find_constants_l2064_206487


namespace maximum_value_of_f_l2064_206424

theorem maximum_value_of_f (x : ℝ) (h : x^4 + 36 ≤ 13 * x^2) : 
  ∃ (m : ℝ), m = 18 ∧ ∀ (x : ℝ), (x^4 + 36 ≤ 13 * x^2) → (x^3 - 3 * x ≤ m) :=
sorry

end maximum_value_of_f_l2064_206424


namespace quadratic_solution_is_unique_l2064_206489

theorem quadratic_solution_is_unique (p q : ℝ) (hp : p ≠ 0) (hq : q ≠ 0)
  (h1 : 2 * p + q / 2 = -p)
  (h2 : 2 * p * (q / 2) = q) :
  (p, q) = (1, -6) :=
by
  sorry

end quadratic_solution_is_unique_l2064_206489


namespace selling_price_of_car_l2064_206486

theorem selling_price_of_car (purchase_price repair_cost : ℝ) (profit_percent : ℝ) 
    (h1 : purchase_price = 42000) (h2 : repair_cost = 8000) (h3 : profit_percent = 29.8) :
    (purchase_price + repair_cost) * (1 + profit_percent / 100) = 64900 := 
by 
  -- The proof will go here
  sorry

end selling_price_of_car_l2064_206486


namespace avg_score_assigned_day_l2064_206498

theorem avg_score_assigned_day
  (total_students : ℕ)
  (exam_assigned_day_students_perc : ℕ)
  (exam_makeup_day_students_perc : ℕ)
  (avg_makeup_day_score : ℕ)
  (total_avg_score : ℕ)
  : exam_assigned_day_students_perc = 70 → 
    exam_makeup_day_students_perc = 30 → 
    avg_makeup_day_score = 95 → 
    total_avg_score = 74 → 
    total_students = 100 → 
    (70 * 65 + 30 * 95 = 7400) :=
by
  intros h1 h2 h3 h4 h5
  sorry

end avg_score_assigned_day_l2064_206498


namespace odd_prime_does_not_divide_odd_nat_number_increment_l2064_206433

theorem odd_prime_does_not_divide_odd_nat_number_increment (p n : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) (hn_odd : n % 2 = 1) :
  ¬ (p * n + 1 ∣ p ^ p - 1) :=
by
  sorry

end odd_prime_does_not_divide_odd_nat_number_increment_l2064_206433


namespace order_A_C_B_l2064_206448

noncomputable def A (a b : ℝ) : ℝ := Real.log ((a + b) / 2)
noncomputable def B (a b : ℝ) : ℝ := Real.sqrt (Real.log a * Real.log b)
noncomputable def C (a b : ℝ) : ℝ := (Real.log a + Real.log b) / 2

theorem order_A_C_B (a b : ℝ) (h1 : 1 < b) (h2 : b < a) :
  A a b > C a b ∧ C a b > B a b :=
by 
  sorry

end order_A_C_B_l2064_206448


namespace minimum_value_am_bn_l2064_206423

theorem minimum_value_am_bn (a b m n : ℝ) (hp_a : a > 0)
    (hp_b : b > 0) (hp_m : m > 0) (hp_n : n > 0) (ha_b : a + b = 1)
    (hm_n : m * n = 2) :
    (am + bn) * (bm + an) ≥ 3/2 := by
  sorry

end minimum_value_am_bn_l2064_206423


namespace quadratic_function_range_l2064_206422

theorem quadratic_function_range (f : ℝ → ℝ) (a : ℝ)
  (h_quad : ∃ p q r : ℝ, ∀ x, f x = p * x^2 + q * x + r)
  (h_sym : ∀ x, f (2 + x) = f (2 - x))
  (h_cond : f a ≤ f 0 ∧ f 0 < f 1) :
  a ≤ 0 ∨ a ≥ 4 :=
sorry

end quadratic_function_range_l2064_206422


namespace union_of_sets_l2064_206463

def A : Set ℤ := {1, 2}
def B : Set ℤ := {-1, 0, 1}

theorem union_of_sets : A ∪ B = {-1, 0, 1, 2} :=
by
  sorry

end union_of_sets_l2064_206463


namespace flower_problem_l2064_206414

theorem flower_problem
  (O : ℕ) 
  (total : ℕ := 105)
  (pink_purple : ℕ := 30)
  (red := 2 * O)
  (yellow := 2 * O - 5)
  (pink := pink_purple / 2)
  (purple := pink)
  (H1 : pink + purple = pink_purple)
  (H2 : pink_purple = 30)
  (H3 : pink = purple)
  (H4 : O + red + yellow + pink + purple = total)
  (H5 : total = 105):
  O = 16 := 
by 
  sorry

end flower_problem_l2064_206414


namespace rationalize_denominator_l2064_206472

-- Problem statement
theorem rationalize_denominator :
  1 / (Real.sqrt 5 - 2) = Real.sqrt 5 + 2 := by
  sorry

end rationalize_denominator_l2064_206472


namespace divisor_is_11_l2064_206419

noncomputable def least_subtracted_divisor : Nat := 11

def problem_condition (D : Nat) (x : Nat) : Prop :=
  2000 - x = 1989 ∧ (2000 - x) % D = 0

theorem divisor_is_11 (D : Nat) (x : Nat) (h : problem_condition D x) : D = least_subtracted_divisor :=
by
  sorry

end divisor_is_11_l2064_206419


namespace ducklings_distance_l2064_206462

noncomputable def ducklings_swim (r : ℝ) (n : ℕ) : Prop :=
  ∀ (ducklings : Fin n → ℝ × ℝ), (∀ i, (ducklings i).1 ^ 2 + (ducklings i).2 ^ 2 = r ^ 2) →
    ∃ (i j : Fin n), i ≠ j ∧ (ducklings i - ducklings j).1 ^ 2 + (ducklings i - ducklings j).2 ^ 2 ≤ r ^ 2

theorem ducklings_distance :
  ducklings_swim 5 6 :=
by sorry

end ducklings_distance_l2064_206462


namespace chessboard_tiling_l2064_206492

theorem chessboard_tiling (chessboard : Fin 8 × Fin 8 → Prop) (colors : Fin 8 × Fin 8 → Bool)
  (removed_squares : (Fin 8 × Fin 8) × (Fin 8 × Fin 8))
  (h_diff_colors : colors removed_squares.1 ≠ colors removed_squares.2) :
  ∃ f : (Fin 8 × Fin 8) → (Fin 8 × Fin 8), ∀ x, chessboard x → chessboard (f x) :=
by
  sorry

end chessboard_tiling_l2064_206492


namespace tangent_value_of_k_k_range_l2064_206406

noncomputable def f (x : Real) : Real := Real.exp (2 * x)
def g (k x : Real) : Real := k * x + 1

theorem tangent_value_of_k (k : Real) :
  (∃ t : Real, f t = g k t ∧ deriv f t = deriv (g k) t) → k = 2 :=
by
  sorry

theorem k_range (k : Real) (h : k > 0) :
  (∃ m : Real, m > 0 ∧ ∀ x : Real, 0 < x → x < m → |f x - g k x| > 2 * x) → 4 < k :=
by
  sorry

end tangent_value_of_k_k_range_l2064_206406


namespace total_tickets_sold_l2064_206499

theorem total_tickets_sold (A C : ℕ) (hC : C = 16) (h1 : 3 * C = 48) (h2 : 5 * A + 3 * C = 178) : 
  A + C = 42 :=
by
  sorry

end total_tickets_sold_l2064_206499


namespace minimum_value_of_expression_l2064_206432

noncomputable def min_expression_value (a b : ℝ) : ℝ :=
  1 / (1 + a) + 4 / (2 + b)

theorem minimum_value_of_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + 3 * b = 7) : 
  min_expression_value a b ≥ (13 + 4 * Real.sqrt 3) / 14 :=
by
  sorry

end minimum_value_of_expression_l2064_206432


namespace graph_passes_quadrants_l2064_206429

theorem graph_passes_quadrants {x y : ℝ} (h : y = -x - 2) :
  -- Statement that the graph passes through the second, third, and fourth quadrants.
  (∃ (x : ℝ), x > 0 ∧ (∃ (y : ℝ), y < 0 ∧ y = -x - 2)) ∧
  (∃ (x : ℝ), x < 0 ∧ (∃ (y : ℝ), y < 0 ∧ y = -x - 2)) ∧
  (∃ (x : ℝ), x > 0 ∧ (∃ (y : ℝ), y > 0 ∧ y = -x - 2)) :=
by
  sorry

end graph_passes_quadrants_l2064_206429


namespace initial_geese_count_l2064_206473

theorem initial_geese_count (G : ℕ) (h1 : G / 2 + 4 = 12) : G = 16 := by
  sorry

end initial_geese_count_l2064_206473


namespace calculate_v2_using_horner_method_l2064_206435

def f (x : ℕ) : ℕ := x^5 + 5 * x^4 + 10 * x^3 + 10 * x^2 + 5 * x + 1

def horner_step (x b a : ℕ) := a * x + b

def horner_eval (coeffs : List ℕ) (x : ℕ) : ℕ :=
coeffs.foldr (horner_step x) 0

theorem calculate_v2_using_horner_method :
  horner_eval [1, 5, 10, 10, 5, 1] 2 = 24 :=
by
  -- This is the theorem statement, the proof is not required as per instructions
  sorry

end calculate_v2_using_horner_method_l2064_206435


namespace options_not_equal_l2064_206476

theorem options_not_equal (a b c d e : ℚ)
  (ha : a = 14 / 10)
  (hb : b = 1 + 2 / 5)
  (hc : c = 1 + 7 / 25)
  (hd : d = 1 + 2 / 10)
  (he : e = 1 + 14 / 70) :
  a = 7 / 5 ∧ b = 7 / 5 ∧ c ≠ 7 / 5 ∧ d ≠ 7 / 5 ∧ e ≠ 7 / 5 :=
by sorry

end options_not_equal_l2064_206476


namespace last_student_calls_out_l2064_206420

-- Define the transformation rules as a function
def next_student (n : ℕ) : ℕ :=
  if n < 10 then n + 8 else (n % 10) + 7

-- Define the sequence generation function
noncomputable def student_number : ℕ → ℕ
| 0       => 1  -- the 1st student starts with number 1
| (n + 1) => next_student (student_number n)

-- The main theorem to prove
theorem last_student_calls_out (n : ℕ) : student_number 2013 = 12 :=
sorry

end last_student_calls_out_l2064_206420


namespace arccos_neg_one_l2064_206454

theorem arccos_neg_one : Real.arccos (-1) = Real.pi := by
  sorry

end arccos_neg_one_l2064_206454


namespace restaurant_made_correct_amount_l2064_206483

noncomputable def restaurant_revenue : ℝ := 
  let price1 := 8
  let qty1 := 10
  let price2 := 10
  let qty2 := 5
  let price3 := 4
  let qty3 := 20
  let total_sales := qty1 * price1 + qty2 * price2 + qty3 * price3
  let discount := 0.10
  let discounted_total := total_sales * (1 - discount)
  let sales_tax := 0.05
  let final_amount := discounted_total * (1 + sales_tax)
  final_amount

theorem restaurant_made_correct_amount : restaurant_revenue = 198.45 := by
  sorry

end restaurant_made_correct_amount_l2064_206483


namespace domain_is_correct_l2064_206449

def domain_of_function (x : ℝ) : Prop :=
  (3 - x ≥ 0) ∧ (x + 1 ≠ 0) ∧ (x + 2 > 0)

theorem domain_is_correct :
  { x : ℝ | domain_of_function x } = { x : ℝ | -2 < x ∧ x ≤ 3 ∧ x ≠ -1 } :=
by
  sorry

end domain_is_correct_l2064_206449


namespace radius_of_small_semicircle_l2064_206485

theorem radius_of_small_semicircle
  (radius_big_semicircle : ℝ)
  (radius_inner_circle : ℝ) 
  (pairwise_tangent : ∀ x : ℝ, x = radius_big_semicircle → x = 12 ∧ 
                                x = radius_inner_circle → x = 6 ∧ 
                                true) :
  ∃ (r : ℝ), r = 4 :=
by 
  sorry

end radius_of_small_semicircle_l2064_206485


namespace resulting_solid_vertices_l2064_206451

theorem resulting_solid_vertices (s1 s2 : ℕ) (orig_vertices removed_cubes : ℕ) :
  s1 = 5 → s2 = 2 → orig_vertices = 8 → removed_cubes = 8 → 
  (orig_vertices - removed_cubes + removed_cubes * (4 * 3 - 3)) = 40 := by
  intros h1 h2 h3 h4
  simp [h1, h2, h3, h4]
  sorry

end resulting_solid_vertices_l2064_206451


namespace trigonometric_identity_l2064_206407

variable (A B C a b c : ℝ)
variable (h_triangle : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2)
variable (h_sides : a > 0 ∧ b > 0 ∧ c > 0)
variable (h_sum_angles : A + B + C = π)
variable (h_condition : (c / b) + (b / c) = (5 * Real.cos A) / 2)

theorem trigonometric_identity 
  (h_triangle_acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2) 
  (h_sides_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sum_angles_eq : A + B + C = π) 
  (h_given : (c / b) + (b / c) = (5 * Real.cos A) / 2) : 
  (Real.tan A / Real.tan B) + (Real.tan A / Real.tan C) = 1/2 :=
by
  sorry

end trigonometric_identity_l2064_206407


namespace determine_all_functions_l2064_206478

noncomputable def functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 - y^2) = (x - y) * (f x + f y)

theorem determine_all_functions (f : ℝ → ℝ) (h : functional_equation f) :
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x :=
sorry

end determine_all_functions_l2064_206478


namespace find_b_l2064_206404

theorem find_b (b : ℝ) (y : ℝ) : (4 * 3 + 2 * y = b) ∧ (3 * 3 + 6 * y = 3 * b) → b = 27 :=
by
sorry

end find_b_l2064_206404


namespace greatest_possible_value_of_n_greatest_possible_value_of_10_l2064_206465

theorem greatest_possible_value_of_n (n : ℤ) (h : 101 * n^2 ≤ 12100) : n ≤ 10 :=
by
  sorry

theorem greatest_possible_value_of_10 (n : ℤ) (h : 101 * n^2 ≤ 12100) : n = 10 → n = 10 :=
by
  sorry

end greatest_possible_value_of_n_greatest_possible_value_of_10_l2064_206465


namespace even_n_square_mod_8_odd_n_square_mod_8_odd_n_fourth_mod_8_l2064_206453

open Int
open Nat

theorem even_n_square_mod_8 (n : ℤ) (h : n % 2 = 0) : (n^2 % 8 = 0) ∨ (n^2 % 8 = 4) := sorry

theorem odd_n_square_mod_8 (n : ℤ) (h : n % 2 = 1) : n^2 % 8 = 1 := sorry

theorem odd_n_fourth_mod_8 (n : ℤ) (h : n % 2 = 1) : n^4 % 8 = 1 := sorry

end even_n_square_mod_8_odd_n_square_mod_8_odd_n_fourth_mod_8_l2064_206453


namespace ratio_of_areas_l2064_206466

theorem ratio_of_areas (Q : Point) (r1 r2 : ℝ) (h : r1 < r2)
  (arc_length_smaller : ℝ) (arc_length_larger : ℝ)
  (h_arc_smaller : arc_length_smaller = (60 / 360) * (2 * r1 * π))
  (h_arc_larger : arc_length_larger = (30 / 360) * (2 * r2 * π))
  (h_equal_arcs : arc_length_smaller = arc_length_larger) :
  (π * r1^2) / (π * r2^2) = 1/4 :=
by
  sorry

end ratio_of_areas_l2064_206466


namespace negation_proposition_equivalence_l2064_206490

theorem negation_proposition_equivalence :
  (¬ (∃ x₀ : ℝ, x₀^2 - x₀ - 1 > 0)) ↔ (∀ x : ℝ, x^2 - x - 1 ≤ 0) :=
by sorry

end negation_proposition_equivalence_l2064_206490


namespace probability_second_marble_purple_correct_l2064_206480

/-!
  Bag A has 5 red marbles and 5 green marbles.
  Bag B has 8 purple marbles and 2 orange marbles.
  Bag C has 3 purple marbles and 7 orange marbles.
  Bag D has 4 purple marbles and 6 orange marbles.
  A marble is drawn at random from Bag A.
  If it is red, a marble is drawn at random from Bag B;
  if it is green, a marble is drawn at random from Bag C;
  but if it is neither (an impossible scenario in this setup), a marble would be drawn from Bag D.
  Prove that the probability of the second marble drawn being purple is 11/20.
-/

noncomputable def probability_second_marble_purple : ℚ :=
  let p_red_A := 5 / 10
  let p_green_A := 5 / 10
  let p_purple_B := 8 / 10
  let p_purple_C := 3 / 10
  (p_red_A * p_purple_B) + (p_green_A * p_purple_C)

theorem probability_second_marble_purple_correct :
  probability_second_marble_purple = 11 / 20 := sorry

end probability_second_marble_purple_correct_l2064_206480


namespace equation_1_solutions_equation_2_solutions_l2064_206412

-- Equation 1: Proving solutions for (x+8)(x+1) = -12
theorem equation_1_solutions (x : ℝ) :
  (x + 8) * (x + 1) = -12 ↔ x = -4 ∨ x = -5 :=
sorry

-- Equation 2: Proving solutions for (2x-3)^2 = 5(2x-3)
theorem equation_2_solutions (x : ℝ) :
  (2 * x - 3) ^ 2 = 5 * (2 * x - 3) ↔ x = 3 / 2 ∨ x = 4 :=
sorry

end equation_1_solutions_equation_2_solutions_l2064_206412


namespace number_of_plain_lemonade_sold_l2064_206458

theorem number_of_plain_lemonade_sold
  (price_per_plain_lemonade : ℝ)
  (earnings_strawberry_lemonade : ℝ)
  (earnings_more_plain_than_strawberry : ℝ)
  (P : ℝ)
  (H1 : price_per_plain_lemonade = 0.75)
  (H2 : earnings_strawberry_lemonade = 16)
  (H3 : earnings_more_plain_than_strawberry = 11)
  (H4 : price_per_plain_lemonade * P = earnings_strawberry_lemonade + earnings_more_plain_than_strawberry) :
  P = 36 :=
by
  sorry

end number_of_plain_lemonade_sold_l2064_206458


namespace total_amount_paid_correct_l2064_206470

-- Definitions of wholesale costs, retail markups, and employee discounts
def wholesale_cost_video_recorder : ℝ := 200
def retail_markup_video_recorder : ℝ := 0.20
def employee_discount_video_recorder : ℝ := 0.30

def wholesale_cost_digital_camera : ℝ := 150
def retail_markup_digital_camera : ℝ := 0.25
def employee_discount_digital_camera : ℝ := 0.20

def wholesale_cost_smart_tv : ℝ := 800
def retail_markup_smart_tv : ℝ := 0.15
def employee_discount_smart_tv : ℝ := 0.25

-- Calculation of retail prices
def retail_price (wholesale_cost : ℝ) (markup : ℝ) : ℝ :=
  wholesale_cost * (1 + markup)

-- Calculation of employee prices
def employee_price (retail_price : ℝ) (discount : ℝ) : ℝ :=
  retail_price * (1 - discount)

-- Retail prices
def retail_price_video_recorder := retail_price wholesale_cost_video_recorder retail_markup_video_recorder
def retail_price_digital_camera := retail_price wholesale_cost_digital_camera retail_markup_digital_camera
def retail_price_smart_tv := retail_price wholesale_cost_smart_tv retail_markup_smart_tv

-- Employee prices
def employee_price_video_recorder := employee_price retail_price_video_recorder employee_discount_video_recorder
def employee_price_digital_camera := employee_price retail_price_digital_camera employee_discount_digital_camera
def employee_price_smart_tv := employee_price retail_price_smart_tv employee_discount_smart_tv

-- Total amount paid by the employee
def total_amount_paid := 
  employee_price_video_recorder 
  + employee_price_digital_camera 
  + employee_price_smart_tv

theorem total_amount_paid_correct :
  total_amount_paid = 1008 := 
  by 
    sorry

end total_amount_paid_correct_l2064_206470


namespace rectangle_area_l2064_206484

noncomputable def radius : ℝ := 7
noncomputable def width : ℝ := 2 * radius
noncomputable def length : ℝ := 3 * width
noncomputable def area (l w : ℝ) : ℝ := l * w

theorem rectangle_area : area length width = 588 := sorry

end rectangle_area_l2064_206484


namespace sequence_length_l2064_206481

theorem sequence_length 
  (a₁ : ℤ) (d : ℤ) (aₙ : ℤ) (n : ℕ) 
  (h₁ : a₁ = -4) 
  (h₂ : d = 3) 
  (h₃ : aₙ = 32) 
  (h₄ : aₙ = a₁ + (n - 1) * d) : 
  n = 13 := 
by 
  sorry

end sequence_length_l2064_206481


namespace solve_eq1_solve_eq2_l2064_206402

theorem solve_eq1 : (2 * (x - 3) = 3 * x * (x - 3)) → (x = 3 ∨ x = 2 / 3) :=
by
  intro h
  sorry

theorem solve_eq2 : (2 * x ^ 2 - 3 * x + 1 = 0) → (x = 1 ∨ x = 1 / 2) :=
by
  intro h
  sorry

end solve_eq1_solve_eq2_l2064_206402


namespace cos_135_eq_neg_sqrt2_div_2_l2064_206446

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = - Real.sqrt 2 / 2 :=
by
  sorry

end cos_135_eq_neg_sqrt2_div_2_l2064_206446


namespace hcf_lcm_product_l2064_206469

theorem hcf_lcm_product (a b : ℕ) (H : a * b = 45276) (L : Nat.lcm a b = 2058) : Nat.gcd a b = 22 :=
by 
  -- The proof steps go here
  sorry

end hcf_lcm_product_l2064_206469


namespace no_natural_number_n_exists_l2064_206409

theorem no_natural_number_n_exists (n : ℕ) :
  ¬ ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ x^2 + y^2 = 2 * n * (n + 1) * (n + 2) * (n + 3) + 12 := 
sorry

end no_natural_number_n_exists_l2064_206409


namespace number_of_ways_to_distribute_balls_l2064_206430

theorem number_of_ways_to_distribute_balls : 
  ∃ n : ℕ, n = 81 ∧ n = 3^4 := 
by sorry

end number_of_ways_to_distribute_balls_l2064_206430


namespace total_work_stations_l2064_206475

theorem total_work_stations (total_students : ℕ) (stations_for_2 : ℕ) (stations_for_3 : ℕ)
  (h1 : total_students = 38)
  (h2 : stations_for_2 = 10)
  (h3 : 20 + 3 * stations_for_3 = total_students) :
  stations_for_2 + stations_for_3 = 16 :=
by
  sorry

end total_work_stations_l2064_206475


namespace find_product_of_offsets_l2064_206447

theorem find_product_of_offsets
  (a b c : ℝ)
  (h1 : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h2 : a * b + a + b = 99)
  (h3 : b * c + b + c = 99)
  (h4 : c * a + c + a = 99) :
  (a + 1) * (b + 1) * (c + 1) = 1000 := by
  sorry

end find_product_of_offsets_l2064_206447


namespace total_cost_for_tickets_l2064_206496

-- Define the known quantities
def students : Nat := 20
def teachers : Nat := 3
def ticket_cost : Nat := 5

-- Define the total number of people
def total_people : Nat := students + teachers

-- Define the total cost
def total_cost : Nat := total_people * ticket_cost

-- Prove that the total cost is $115
theorem total_cost_for_tickets : total_cost = 115 := by
  -- Sorry is used here to skip the proof
  sorry

end total_cost_for_tickets_l2064_206496


namespace pie_cost_correct_l2064_206440

-- Define the initial and final amounts of money Mary had.
def initial_amount : ℕ := 58
def final_amount : ℕ := 52

-- Define the cost of the pie as the difference between initial and final amounts.
def pie_cost : ℕ := initial_amount - final_amount

-- State the theorem that given the initial and final amounts, the cost of the pie is 6.
theorem pie_cost_correct : pie_cost = 6 := by 
  sorry

end pie_cost_correct_l2064_206440


namespace sean_days_played_is_14_l2064_206468

def total_minutes_played : Nat := 1512
def indira_minutes_played : Nat := 812
def sean_minutes_per_day : Nat := 50
def sean_total_minutes : Nat := total_minutes_played - indira_minutes_played
def sean_days_played : Nat := sean_total_minutes / sean_minutes_per_day

theorem sean_days_played_is_14 : sean_days_played = 14 :=
by
  sorry

end sean_days_played_is_14_l2064_206468


namespace negation_of_proposition_l2064_206410

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x^2 = 1 → x = 1 ∨ x = -1) ↔ ∀ x : ℝ, x^2 ≠ 1 → x ≠ 1 ∧ x ≠ -1 := by
sorry

end negation_of_proposition_l2064_206410


namespace correct_options_l2064_206455

-- Definitions of conditions in Lean 
def is_isosceles (T : Triangle) : Prop := sorry -- Define isosceles triangle
def is_right_angle (T : Triangle) : Prop := sorry -- Define right-angled triangle
def similar (T₁ T₂ : Triangle) : Prop := sorry -- Define similarity of triangles
def equal_vertex_angle (T₁ T₂ : Triangle) : Prop := sorry -- Define equal vertex angle
def equal_base_angle (T₁ T₂ : Triangle) : Prop := sorry -- Define equal base angle

-- Theorem statement to verify correct options (2) and (4)
theorem correct_options {T₁ T₂ : Triangle} :
  (is_right_angle T₁ ∧ is_right_angle T₂ ∧ is_isosceles T₁ ∧ is_isosceles T₂ → similar T₁ T₂) ∧ 
  (equal_vertex_angle T₁ T₂ ∧ is_isosceles T₁ ∧ is_isosceles T₂ → similar T₁ T₂) :=
sorry -- proof not required

end correct_options_l2064_206455


namespace alpha_value_l2064_206405

theorem alpha_value (α : ℝ) (h : (α * (α - 1) * (-1 : ℝ)^(α - 2)) = 4) : α = -4 :=
by
  sorry

end alpha_value_l2064_206405


namespace vector_add_sub_l2064_206445

open Matrix

section VectorProof

/-- Define the vectors a, b, and c. -/
def a : Matrix (Fin 2) (Fin 1) ℤ := ![![3], ![-6]]
def b : Matrix (Fin 2) (Fin 1) ℤ := ![![-1], ![5]]
def c : Matrix (Fin 2) (Fin 1) ℤ := ![![5], ![-20]]

/-- State the proof problem. -/
theorem vector_add_sub :
  2 • a + 4 • b - c = ![![-3], ![28]] :=
by
  sorry

end VectorProof

end vector_add_sub_l2064_206445


namespace incorrect_expression_l2064_206495

theorem incorrect_expression :
  ¬((|(-5 : ℤ)|)^2 = 5) :=
by
sorry

end incorrect_expression_l2064_206495


namespace vector_dot_product_identity_l2064_206491

-- Define the vectors a, b, and c in ℝ²
def a : ℝ × ℝ := (-1, 2)
def b : ℝ × ℝ := (-1, 1)
def c : ℝ × ℝ := (-3, 1)

-- Define vector addition and dot product in ℝ²
def vector_add (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Prove that c · (a + b) = 9
theorem vector_dot_product_identity : dot_product c (vector_add a b) = 9 := 
by 
sorry

end vector_dot_product_identity_l2064_206491


namespace ratio_of_side_lengths_l2064_206457

theorem ratio_of_side_lengths (t p : ℕ) (h1 : 3 * t = 30) (h2 : 5 * p = 30) : t / p = 5 / 3 :=
by
  sorry

end ratio_of_side_lengths_l2064_206457


namespace find_x_l2064_206442

theorem find_x (x : ℝ) : 0.5 * x + (0.3 * 0.2) = 0.26 ↔ x = 0.4 := by
  sorry

end find_x_l2064_206442


namespace total_time_iggy_runs_correct_l2064_206452

noncomputable def total_time_iggy_runs : ℝ :=
  let monday_time := 3 * (10 + 1 + 0.5);
  let tuesday_time := 5 * (9 + 1 + 1);
  let wednesday_time := 7 * (12 - 2 + 2);
  let thursday_time := 10 * (8 + 2 + 4);
  let friday_time := 4 * (10 + 0.25);
  monday_time + tuesday_time + wednesday_time + thursday_time + friday_time

theorem total_time_iggy_runs_correct : total_time_iggy_runs = 354.5 := by
  sorry

end total_time_iggy_runs_correct_l2064_206452


namespace cost_per_book_l2064_206477

theorem cost_per_book (a r n c : ℕ) (h : a - r = n * c) : c = 7 :=
by sorry

end cost_per_book_l2064_206477
