import Mathlib

namespace complement_union_A_B_l1424_142480

def A := {x : ℤ | ∃ k : ℤ, x = 3 * k + 1}
def B := {x : ℤ | ∃ k : ℤ, x = 3 * k + 2}
def U := {x : ℤ | True}

theorem complement_union_A_B:
  (U \ (A ∪ B) = {x : ℤ | ∃ k : ℤ, x = 3 * k}) :=
by
  sorry

end complement_union_A_B_l1424_142480


namespace diagonals_from_one_vertex_l1424_142478

theorem diagonals_from_one_vertex (x : ℕ) (h : (x - 2) * 180 = 1800) : (x - 3) = 9 :=
  by
  sorry

end diagonals_from_one_vertex_l1424_142478


namespace polar_area_enclosed_l1424_142422

theorem polar_area_enclosed :
  let θ1 := Real.pi / 3
  let θ2 := 2 * Real.pi / 3
  let ρ := 4
  let area := (1/2) * (θ2 - θ1) * ρ^2
  area = 8 * Real.pi / 3 :=
by
  let θ1 := Real.pi / 3
  let θ2 := 2 * Real.pi / 3
  let ρ := 4
  let area := (1/2) * (θ2 - θ1) * ρ^2
  show area = 8 * Real.pi / 3
  sorry

end polar_area_enclosed_l1424_142422


namespace point_A_final_position_supplement_of_beta_l1424_142479

-- Define the initial and final position of point A on the number line
def initial_position := -5
def moved_position_right := initial_position + 4
def final_position := moved_position_right - 1

theorem point_A_final_position : final_position = -2 := 
by 
-- Proof can be added here
sorry

-- Define the angles and the relationship between them
def alpha := 40
def beta := 90 - alpha
def supplement_beta := 180 - beta

theorem supplement_of_beta : supplement_beta = 130 := 
by 
-- Proof can be added here
sorry

end point_A_final_position_supplement_of_beta_l1424_142479


namespace average_p_q_l1424_142444

theorem average_p_q (p q : ℝ) 
  (h1 : (4 + 6 + 8 + 2 * p + 2 * q) / 7 = 20) : 
  (p + q) / 2 = 30.5 :=
by
  sorry

end average_p_q_l1424_142444


namespace big_al_bananas_l1424_142412

/-- Big Al ate 140 bananas from May 1 through May 6. Each day he ate five more bananas than on the previous day. On May 4, Big Al did not eat any bananas due to fasting. Prove that Big Al ate 38 bananas on May 6. -/
theorem big_al_bananas : 
  ∃ a : ℕ, (a + (a + 5) + (a + 10) + 0 + (a + 15) + (a + 20) = 140) ∧ ((a + 20) = 38) :=
by sorry

end big_al_bananas_l1424_142412


namespace tan_405_eq_1_l1424_142466

theorem tan_405_eq_1 : Real.tan (405 * Real.pi / 180) = 1 := 
by 
  sorry

end tan_405_eq_1_l1424_142466


namespace investment_in_stocks_l1424_142471

theorem investment_in_stocks (T b s : ℝ) (h1 : T = 200000) (h2 : s = 5 * b) (h3 : T = b + s) :
  s = 166666.65 :=
by sorry

end investment_in_stocks_l1424_142471


namespace value_at_2013_l1424_142474

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function : ∀ x : ℝ, f x = -f (-x)
axiom periodic_5 : ∀ x : ℝ, f (x + 5) ≥ f x
axiom periodic_1 : ∀ x : ℝ, f (x + 1) ≤ f x

theorem value_at_2013 : f 2013 = 0 :=
by
  -- Proof goes here
  sorry

end value_at_2013_l1424_142474


namespace residue_mod_13_l1424_142495

theorem residue_mod_13 :
  (250 ≡ 3 [MOD 13]) → 
  (20 ≡ 7 [MOD 13]) → 
  (5^2 ≡ 12 [MOD 13]) → 
  ((250 * 11 - 20 * 6 + 5^2) % 13 = 3) :=
by 
  sorry

end residue_mod_13_l1424_142495


namespace product_of_squares_is_perfect_square_l1424_142489

theorem product_of_squares_is_perfect_square (a b c : ℤ) (h : a * b + b * c + c * a = 1) :
    ∃ k : ℤ, (1 + a^2) * (1 + b^2) * (1 + c^2) = k^2 :=
sorry

end product_of_squares_is_perfect_square_l1424_142489


namespace scalene_triangle_third_side_l1424_142468

theorem scalene_triangle_third_side (a b c : ℕ) (h : (a - 3)^2 + (b - 2)^2 = 0) : 
  a = 3 ∧ b = 2 → c = 2 ∨ c = 3 ∨ c = 4 := 
by {
  sorry
}

end scalene_triangle_third_side_l1424_142468


namespace painting_problem_l1424_142475

theorem painting_problem
    (H_rate : ℝ := 1 / 60)
    (T_rate : ℝ := 1 / 90)
    (combined_rate : ℝ := H_rate + T_rate)
    (time_worked : ℝ := 15)
    (wall_painted : ℝ := time_worked * combined_rate):
  wall_painted = 5 / 12 := 
by
  sorry

end painting_problem_l1424_142475


namespace solve_problem_l1424_142499

-- Definitions based on conditions
def salty_cookies_eaten : ℕ := 28
def sweet_cookies_eaten : ℕ := 15

-- Problem statement
theorem solve_problem : salty_cookies_eaten - sweet_cookies_eaten = 13 := by
  sorry

end solve_problem_l1424_142499


namespace choir_group_students_l1424_142494

theorem choir_group_students : ∃ n : ℕ, (n % 5 = 0) ∧ (n % 9 = 0) ∧ (n % 12 = 0) ∧ (∃ m : ℕ, n = m * m) ∧ n ≥ 360 := 
sorry

end choir_group_students_l1424_142494


namespace quadratic_inequality_real_solutions_l1424_142407

theorem quadratic_inequality_real_solutions (c : ℝ) (h_pos : 0 < c) : 
  (∃ x : ℝ, x^2 - 10 * x + c < 0) ↔ c < 25 :=
sorry

end quadratic_inequality_real_solutions_l1424_142407


namespace range_of_a_l1424_142437

noncomputable def f (a x : ℝ) : ℝ := Real.exp x - a * x^2

theorem range_of_a {a : ℝ} : 
  (∀ x, Real.exp x - 2 * a * x ≥ 0) ↔ 0 ≤ a ∧ a ≤ Real.exp 1 / 2 :=
by
  sorry

end range_of_a_l1424_142437


namespace interest_earned_after_4_years_l1424_142411

noncomputable def calculate_total_interest (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  let A := P * (1 + r) ^ t
  A - P

theorem interest_earned_after_4_years :
  calculate_total_interest 2000 0.12 4 = 1147.04 :=
by
  sorry

end interest_earned_after_4_years_l1424_142411


namespace chord_length_l1424_142434

theorem chord_length
  (l_eq : ∀ (rho theta : ℝ), rho * (Real.sin theta - Real.cos theta) = 1)
  (gamma_eq : ∀ (rho : ℝ) (theta : ℝ), rho = 1) :
  ∃ AB : ℝ, AB = Real.sqrt 2 :=
by
  sorry

end chord_length_l1424_142434


namespace remainder_4x_mod_7_l1424_142486

theorem remainder_4x_mod_7 (x : ℤ) (k : ℤ) (h : x = 7 * k + 5) : (4 * x) % 7 = 6 :=
by
  sorry

end remainder_4x_mod_7_l1424_142486


namespace matthew_initial_crackers_l1424_142492

theorem matthew_initial_crackers :
  ∃ C : ℕ,
  (∀ (crackers_per_friend cakes_per_friend : ℕ), cakes_per_friend * 4 = 98 → crackers_per_friend = cakes_per_friend → crackers_per_friend * 4 + 8 * 4 = C) ∧ C = 128 :=
sorry

end matthew_initial_crackers_l1424_142492


namespace trigonometric_identity_l1424_142469

theorem trigonometric_identity (α : ℝ) :
  (2 * (Real.cos (2 * α))^2 + Real.sqrt 3 * Real.sin (4 * α) - 1) / 
  (2 * (Real.sin (2 * α))^2 + Real.sqrt 3 * Real.sin (4 * α) - 1) = 
  Real.sin (4 * α + Real.pi / 6) / Real.sin (4 * α - Real.pi / 6) :=
sorry

end trigonometric_identity_l1424_142469


namespace principal_sum_l1424_142402

noncomputable def diff_simple_compound_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
(P * ((1 + r / 100)^t) - P) - (P * r * t / 100)

theorem principal_sum (P : ℝ) (r : ℝ) (t : ℝ) (h : diff_simple_compound_interest P r t = 631) (hr : r = 10) (ht : t = 2) :
    P = 63100 := by
  sorry

end principal_sum_l1424_142402


namespace converse_inverse_l1424_142465

-- Define the properties
def is_parallelogram (polygon : Type) : Prop := sorry -- needs definitions about polygons
def has_two_pairs_of_parallel_sides (polygon : Type) : Prop := sorry -- needs definitions about polygons

-- The given condition
axiom parallelogram_implies_parallel_sides (polygon : Type) :
  is_parallelogram polygon → has_two_pairs_of_parallel_sides polygon

-- Proof of the converse:
theorem converse (polygon : Type) :
  has_two_pairs_of_parallel_sides polygon → is_parallelogram polygon := sorry

-- Proof of the inverse:
theorem inverse (polygon : Type) :
  ¬is_parallelogram polygon → ¬has_two_pairs_of_parallel_sides polygon := sorry

end converse_inverse_l1424_142465


namespace find_k_l1424_142427

theorem find_k (k : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 ↔ |k * x - 4| ≤ 2) → k = 2 :=
by
  sorry

end find_k_l1424_142427


namespace child_B_share_l1424_142429

theorem child_B_share (total_money : ℕ) (ratio_A ratio_B ratio_C ratio_D ratio_E total_parts : ℕ) 
  (h1 : total_money = 12000)
  (h2 : ratio_A = 2)
  (h3 : ratio_B = 3)
  (h4 : ratio_C = 4)
  (h5 : ratio_D = 5)
  (h6 : ratio_E = 6)
  (h_total_parts : total_parts = ratio_A + ratio_B + ratio_C + ratio_D + ratio_E) :
  (total_money / total_parts) * ratio_B = 1800 :=
by
  sorry

end child_B_share_l1424_142429


namespace find_fourth_number_l1424_142498

theorem find_fourth_number (a : ℕ → ℕ) 
  (h1 : ∀ n, n ≥ 2 → a n = a (n - 1) + a (n - 2)) 
  (h2 : a 6 = 42) 
  (h3 : a 8 = 110) : 
  a 3 = 10 := 
sorry

end find_fourth_number_l1424_142498


namespace triangle_ineq_l1424_142406

theorem triangle_ineq (a b c : ℝ) 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
  (h4 : a + b > c) (h5 : a + c > b) (h6 : b + c > a) : 
  (a / (b + c) + b / (a + c) + c / (a + b)) < 5/2 := 
by
  sorry

end triangle_ineq_l1424_142406


namespace solve_for_x_l1424_142464

theorem solve_for_x : ∃ x : ℤ, 25 - 7 = 3 + x ∧ x = 15 := by
  sorry

end solve_for_x_l1424_142464


namespace value_of_t_l1424_142403

noncomputable def f (x t k : ℝ) : ℝ := (1/3) * x^3 - (t/2) * x^2 + k * x

theorem value_of_t (a b t k : ℝ) (h1 : 0 < t) (h2 : 0 < k) 
  (h3 : a + b = t) (h4 : a * b = k) (h5 : 2 * a = b - 2) (h6 : (-2)^2 = a * b) : 
  t = 5 := 
  sorry

end value_of_t_l1424_142403


namespace find_t_l1424_142443

variables (t : ℝ)

def a : ℝ × ℝ := (-1, 1)
def b : ℝ × ℝ := (3, t)
def a_plus_b : ℝ × ℝ := (2, 1 + t)

def are_parallel (u v : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, u = (k * v.1, k * v.2)

theorem find_t (t : ℝ) :
  are_parallel (3, t) (2, 1 + t) ↔ t = -3 :=
sorry

end find_t_l1424_142443


namespace binomial_constant_term_l1424_142458

theorem binomial_constant_term (n : ℕ) (h : n > 0) :
  (∃ r : ℕ, n = 2 * r) ↔ (n = 6) :=
by
  sorry

end binomial_constant_term_l1424_142458


namespace total_peanuts_in_box_l1424_142445

def initial_peanuts := 4
def peanuts_taken_out := 3
def peanuts_added := 12

theorem total_peanuts_in_box : initial_peanuts - peanuts_taken_out + peanuts_added = 13 :=
by
sorry

end total_peanuts_in_box_l1424_142445


namespace citizen_income_l1424_142441

theorem citizen_income (I : ℝ) (h1 : ∀ I ≤ 40000, 0.15 * I = 8000) 
  (h2 : ∀ I > 40000, (0.15 * 40000 + 0.20 * (I - 40000)) = 8000) : 
  I = 50000 :=
by
  sorry

end citizen_income_l1424_142441


namespace total_apples_after_transactions_l1424_142420

def initial_apples : ℕ := 65
def percentage_used : ℕ := 20
def apples_bought : ℕ := 15

theorem total_apples_after_transactions :
  (initial_apples * (1 - percentage_used / 100)) + apples_bought = 67 := 
by
  sorry

end total_apples_after_transactions_l1424_142420


namespace avg_height_correct_l1424_142413

theorem avg_height_correct (h1 h2 h3 h4 : ℝ) (h_distinct: h1 ≠ h2 ∧ h2 ≠ h3 ∧ h3 ≠ h4 ∧ h1 ≠ h3 ∧ h1 ≠ h4 ∧ h2 ≠ h4)
  (h_tallest: h4 = 152) (h_shortest: h1 = 137) 
  (h4_largest: h4 > h3 ∧ h4 > h2 ∧ h4 > h1) (h1_smallest: h1 < h2 ∧ h1 < h3 ∧ h1 < h4) :
  ∃ (avg : ℝ), avg = 145 ∧ (h1 + h2 + h3 + h4) / 4 = avg := 
sorry

end avg_height_correct_l1424_142413


namespace melissa_work_hours_l1424_142435

variable (f : ℝ) (f_d : ℝ) (h_d : ℝ)

theorem melissa_work_hours (hf : f = 56) (hfd : f_d = 4) (hhd : h_d = 3) : 
  (f / f_d) * h_d = 42 := by
  sorry

end melissa_work_hours_l1424_142435


namespace trigonometric_proof_l1424_142410

noncomputable def proof_problem (α β : Real) : Prop :=
  (β = 90 - α) → (Real.sin β = Real.cos α) → 
  (Real.sqrt 3 * Real.sin α + Real.sin β) / Real.sqrt (2 - 2 * Real.cos 100) = 1

-- Statement that incorporates all conditions and concludes the proof problem.
theorem trigonometric_proof :
  proof_problem 20 70 :=
by
  intros h1 h2
  sorry

end trigonometric_proof_l1424_142410


namespace integer_divisibility_l1424_142401

open Nat

theorem integer_divisibility (n : ℕ) (h1 : ∃ m : ℕ, 2^n - 2 = n * m) : ∃ k : ℕ, 2^((2^n) - 1) - 2 = (2^n - 1) * k := by
  sorry

end integer_divisibility_l1424_142401


namespace four_digit_num_condition_l1424_142417

theorem four_digit_num_condition :
  ∃ (n : ℕ), 1000 ≤ n ∧ n < 10000 ∧
  ∃ (a x : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ x = 100*a ∧ n = 1000*a + x :=
by sorry

end four_digit_num_condition_l1424_142417


namespace midpoint_translation_l1424_142426

theorem midpoint_translation (x1 y1 x2 y2 tx ty mx my : ℤ) 
  (hx1 : x1 = 1) (hy1 : y1 = 3) (hx2 : x2 = 5) (hy2 : y2 = -7)
  (htx : tx = 3) (hty : ty = -4)
  (hmx : mx = (x1 + x2) / 2 + tx) (hmy : my = (y1 + y2) / 2 + ty) : 
  mx = 6 ∧ my = -6 :=
by
  sorry

end midpoint_translation_l1424_142426


namespace trig_relationship_l1424_142483

noncomputable def a := Real.cos 1
noncomputable def b := Real.cos 2
noncomputable def c := Real.sin 2

theorem trig_relationship : c > a ∧ a > b := by
  sorry

end trig_relationship_l1424_142483


namespace simplest_quadratic_radical_l1424_142408
  
theorem simplest_quadratic_radical (A B C D: ℝ) 
  (hA : A = Real.sqrt 0.1) 
  (hB : B = Real.sqrt (-2)) 
  (hC : C = 3 * Real.sqrt 2) 
  (hD : D = -Real.sqrt 20) : C = 3 * Real.sqrt 2 :=
by
  have h1 : ∀ (x : ℝ), Real.sqrt x = Real.sqrt x := sorry
  sorry

end simplest_quadratic_radical_l1424_142408


namespace num_ways_to_buy_three_items_l1424_142438

-- Defining the conditions based on the problem statement
def num_headphones : ℕ := 9
def num_mice : ℕ := 13
def num_keyboards : ℕ := 5
def num_kb_mouse_sets : ℕ := 4
def num_hp_mouse_sets : ℕ := 5

-- Defining the theorem statement
theorem num_ways_to_buy_three_items :
  (num_kb_mouse_sets * num_headphones) + 
  (num_hp_mouse_sets * num_keyboards) + 
  (num_headphones * num_mice * num_keyboards) = 646 := 
  by
  sorry

end num_ways_to_buy_three_items_l1424_142438


namespace probability_matching_shoes_l1424_142459

theorem probability_matching_shoes :
  let total_shoes := 24;
  let total_pairs := 12;
  let total_combinations := (total_shoes * (total_shoes - 1)) / 2;
  let matching_pairs := total_pairs;
  let probability := matching_pairs / total_combinations;
  probability = 1 / 23 :=
by
  let total_shoes := 24
  let total_pairs := 12
  let total_combinations := (total_shoes * (total_shoes - 1)) / 2
  let matching_pairs := total_pairs
  let probability := matching_pairs / total_combinations
  have : total_combinations = 276 := by norm_num
  have : matching_pairs = 12 := by norm_num
  have : probability = 1 / 23 := by norm_num
  exact this

end probability_matching_shoes_l1424_142459


namespace range_of_a_l1424_142484

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (2 * a ≤ x ∧ x ≤ a^2 + 1) → (x^2 - 3 * (a + 1) * x + 2 * (3 * a + 1) ≤ 0)) ↔ (1 ≤ a ∧ a ≤ 3 ∨ a = -1) :=
by
  sorry

end range_of_a_l1424_142484


namespace area_of_park_l1424_142461

noncomputable def length (x : ℕ) : ℕ := 3 * x
noncomputable def width (x : ℕ) : ℕ := 2 * x
noncomputable def area (x : ℕ) : ℕ := length x * width x
noncomputable def cost_per_meter : ℕ := 80
noncomputable def total_cost : ℕ := 200
noncomputable def perimeter (x : ℕ) : ℕ := 2 * (length x + width x)

theorem area_of_park : ∃ x : ℕ, area x = 3750 ∧ total_cost = (perimeter x) * cost_per_meter / 100 := by
  sorry

end area_of_park_l1424_142461


namespace age_difference_l1424_142462

theorem age_difference (sum_ages : ℕ) (eldest_age : ℕ) (age_diff : ℕ) 
(h1 : sum_ages = 50) (h2 : eldest_age = 14) :
  14 + (14 - age_diff) + (14 - 2 * age_diff) + (14 - 3 * age_diff) + (14 - 4 * age_diff) = 50 → age_diff = 2 := 
by
  intro h
  sorry

end age_difference_l1424_142462


namespace simplify_and_evaluate_l1424_142454

theorem simplify_and_evaluate
  (m : ℝ) (hm : m = 2 + Real.sqrt 2) :
  (1 - (m / (m + 2))) / ((m^2 - 4*m + 4) / (m^2 - 4)) = Real.sqrt 2 :=
by
  sorry

end simplify_and_evaluate_l1424_142454


namespace necessary_condition_for_inequality_l1424_142460

theorem necessary_condition_for_inequality 
  (m : ℝ) : (∀ x : ℝ, x^2 - 2 * x + m > 0) → m > 0 :=
by 
  sorry

end necessary_condition_for_inequality_l1424_142460


namespace driver_net_rate_of_pay_is_30_33_l1424_142490

noncomputable def driver_net_rate_of_pay : ℝ :=
  let hours := 3
  let speed_mph := 65
  let miles_per_gallon := 30
  let pay_per_mile := 0.55
  let cost_per_gallon := 2.50
  let total_distance := speed_mph * hours
  let gallons_used := total_distance / miles_per_gallon
  let gross_earnings := total_distance * pay_per_mile
  let fuel_cost := gallons_used * cost_per_gallon
  let net_earnings := gross_earnings - fuel_cost
  let net_rate_per_hour := net_earnings / hours
  net_rate_per_hour

theorem driver_net_rate_of_pay_is_30_33 :
  driver_net_rate_of_pay = 30.33 :=
by
  sorry

end driver_net_rate_of_pay_is_30_33_l1424_142490


namespace rental_plans_count_l1424_142419

-- Define the number of large buses, medium buses, and the total number of people.
def num_large_buses := 42
def num_medium_buses := 25
def total_people := 1511

-- State the theorem to prove that there are exactly 2 valid rental plans.
theorem rental_plans_count (x y : ℕ) :
  (num_large_buses * x + num_medium_buses * y = total_people) →
  (∃! (x y : ℕ), num_large_buses * x + num_medium_buses * y = total_people) :=
by
  sorry

end rental_plans_count_l1424_142419


namespace log_inequality_l1424_142433

theorem log_inequality {a x : ℝ} (h1 : 0 < x) (h2 : x < 1) (h3 : a > 0) (h4 : a ≠ 1) : 
  abs (Real.logb a (1 - x)) > abs (Real.logb a (1 + x)) :=
sorry

end log_inequality_l1424_142433


namespace abs_diff_eq_l1424_142496

theorem abs_diff_eq (a b c d : ℤ) (h1 : a = 13) (h2 : b = 3) (h3 : c = 4) (h4 : d = 10) : 
  |a - b| - |c - d| = 4 := 
  by
  -- Proof goes here
  sorry

end abs_diff_eq_l1424_142496


namespace at_least_one_genuine_l1424_142442

theorem at_least_one_genuine :
  ∀ (total_products genuine_products defective_products selected_products : ℕ),
  total_products = 12 →
  genuine_products = 10 →
  defective_products = 2 →
  selected_products = 3 →
  (∃ g d : ℕ, g + d = selected_products ∧ g = 0 ∧ d = selected_products) = false :=
by
  intros total_products genuine_products defective_products selected_products
  intros H_total H_gen H_def H_sel
  sorry

end at_least_one_genuine_l1424_142442


namespace inequality_absolute_value_l1424_142488

theorem inequality_absolute_value (a b : ℝ) (h1 : a < b) (h2 : b < 0) : |a| > -b :=
sorry

end inequality_absolute_value_l1424_142488


namespace unique_rectangles_l1424_142418

theorem unique_rectangles (a b x y : ℝ) (h_dim : a < b) 
    (h_perimeter : 2 * (x + y) = a + b)
    (h_area : x * y = (a * b) / 2) : 
    (∃ x y : ℝ, (2 * (x + y) = a + b) ∧ (x * y = (a * b) / 2) ∧ (x < a) ∧ (y < b)) → 
    (∃! z w : ℝ, (2 * (z + w) = a + b) ∧ (z * y = (a * b) / 2) ∧ (z < a) ∧ (w < b)) :=
sorry

end unique_rectangles_l1424_142418


namespace repeating_decimal_sum_l1424_142452

theorem repeating_decimal_sum (c d : ℕ) (h : 7 / 19 = (c * 10 + d) / 99) : c + d = 9 :=
sorry

end repeating_decimal_sum_l1424_142452


namespace cocktail_cans_l1424_142455

theorem cocktail_cans (prev_apple_ratio : ℝ) (prev_grape_ratio : ℝ) 
  (new_apple_cans : ℝ) : ∃ new_grape_cans : ℝ, new_grape_cans = 15 :=
by
  let prev_apple_per_can := 1 / 6
  let prev_grape_per_can := 1 / 10
  let prev_total_per_can := (1 / 6) + (1 / 10)
  let new_apple_per_can := 1 / 5
  let new_grape_per_can := prev_total_per_can - new_apple_per_can
  let result := 1 / new_grape_per_can
  use result
  sorry

end cocktail_cans_l1424_142455


namespace student_A_selection_probability_l1424_142476

def probability_student_A_selected (total_students : ℕ) (students_removed : ℕ) (representatives : ℕ) : ℚ :=
  representatives / (total_students : ℚ)

theorem student_A_selection_probability :
  probability_student_A_selected 752 2 5 = 5 / 752 :=
by
  sorry

end student_A_selection_probability_l1424_142476


namespace third_box_number_l1424_142409

def N : ℕ := 301

theorem third_box_number (N : ℕ) (h1 : N % 3 = 1) (h2 : N % 4 = 1) (h3 : N % 7 = 0) :
  ∃ x : ℕ, x > 4 ∧ x ≠ 7 ∧ N % x = 1 ∧ (∀ y > 4, y ≠ 7 → y < x → N % y ≠ 1) ∧ x = 6 :=
by
  sorry

end third_box_number_l1424_142409


namespace total_cost_of_4_sandwiches_and_6_sodas_is_37_4_l1424_142451

def sandwiches_cost (s: ℕ) : ℝ := 4 * s
def sodas_cost (d: ℕ) : ℝ := 3 * d
def total_cost_before_tax (s: ℕ) (d: ℕ) : ℝ := sandwiches_cost s + sodas_cost d
def tax (amount: ℝ) : ℝ := 0.10 * amount
def total_cost (s: ℕ) (d: ℕ) : ℝ := total_cost_before_tax s d + tax (total_cost_before_tax s d)

theorem total_cost_of_4_sandwiches_and_6_sodas_is_37_4 :
    total_cost 4 6 = 37.4 :=
sorry

end total_cost_of_4_sandwiches_and_6_sodas_is_37_4_l1424_142451


namespace sector_central_angle_l1424_142414

noncomputable def sector_radius (r l : ℝ) : Prop :=
2 * r + l = 10

noncomputable def sector_area (r l : ℝ) : Prop :=
(1 / 2) * l * r = 4

noncomputable def central_angle (α r l : ℝ) : Prop :=
α = l / r

theorem sector_central_angle (r l α : ℝ) 
  (h1 : sector_radius r l) 
  (h2 : sector_area r l) 
  (h3 : central_angle α r l) : 
  α = 1 / 2 := 
by
  sorry

end sector_central_angle_l1424_142414


namespace apples_in_each_crate_l1424_142421

theorem apples_in_each_crate
  (num_crates : ℕ) 
  (num_rotten : ℕ) 
  (num_boxes : ℕ) 
  (apples_per_box : ℕ) 
  (total_good_apples : ℕ) 
  (total_apples : ℕ)
  (h1 : num_crates = 12) 
  (h2 : num_rotten = 160) 
  (h3 : num_boxes = 100) 
  (h4 : apples_per_box = 20) 
  (h5 : total_good_apples = num_boxes * apples_per_box) 
  (h6 : total_apples = total_good_apples + num_rotten) : 
  total_apples / num_crates = 180 := 
by 
  sorry

end apples_in_each_crate_l1424_142421


namespace rectangle_area_l1424_142405

theorem rectangle_area (length_of_rectangle radius_of_circle side_of_square : ℝ)
  (h1 : length_of_rectangle = (2 / 5) * radius_of_circle)
  (h2 : radius_of_circle = side_of_square)
  (h3 : side_of_square * side_of_square = 1225)
  (breadth_of_rectangle : ℝ)
  (h4 : breadth_of_rectangle = 10) : 
  length_of_rectangle * breadth_of_rectangle = 140 := 
by 
  sorry

end rectangle_area_l1424_142405


namespace average_of_rest_of_class_l1424_142430

def class_average (n : ℕ) (avg : ℕ) := n * avg
def sub_class_average (n : ℕ) (sub_avg : ℕ) := (n / 4) * sub_avg

theorem average_of_rest_of_class (n : ℕ) (h1 : class_average n 80 = 80 * n) (h2 : sub_class_average n 92 = (n / 4) * 92) :
  let A := 76
  A * (3 * n / 4) + (n / 4) * 92 = 80 * n := by
  sorry

end average_of_rest_of_class_l1424_142430


namespace length_of_first_platform_l1424_142400

theorem length_of_first_platform 
  (train_length : ℕ) (first_time : ℕ) (second_platform_length : ℕ) (second_time : ℕ)
  (speed_first : ℕ) (speed_second : ℕ) :
  train_length = 230 → 
  first_time = 15 → 
  second_platform_length = 250 → 
  second_time = 20 → 
  speed_first = (train_length + L) / first_time →
  speed_second = (train_length + second_platform_length) / second_time →
  speed_first = speed_second →
  (L : ℕ) = 130 :=
by
  sorry

end length_of_first_platform_l1424_142400


namespace james_calories_per_minute_l1424_142453

-- Define the conditions
def bags : Nat := 3
def ounces_per_bag : Nat := 2
def calories_per_ounce : Nat := 150
def excess_calories : Nat := 420
def run_minutes : Nat := 40

-- Calculate the total consumed calories
def consumed_calories : Nat := (bags * ounces_per_bag) * calories_per_ounce

-- Calculate the calories burned during the run
def run_calories : Nat := consumed_calories - excess_calories

-- Calculate the calories burned per minute
def calories_per_minute : Nat := run_calories / run_minutes

-- The proof problem statement
theorem james_calories_per_minute : calories_per_minute = 12 := by
  -- Due to the proof not required, we use sorry to skip it.
  sorry

end james_calories_per_minute_l1424_142453


namespace m_squared_n_minus_1_l1424_142450

theorem m_squared_n_minus_1 (a b m n : ℝ)
  (h1 : a * m^2001 + b * n^2001 = 3)
  (h2 : a * m^2002 + b * n^2002 = 7)
  (h3 : a * m^2003 + b * n^2003 = 24)
  (h4 : a * m^2004 + b * n^2004 = 102) :
  m^2 * (n - 1) = 6 := by
  sorry

end m_squared_n_minus_1_l1424_142450


namespace harper_jack_distance_apart_l1424_142470

def total_distance : ℕ := 1000
def distance_jack_run : ℕ := 152
def distance_apart (total_distance : ℕ) (distance_jack_run : ℕ) : ℕ :=
  total_distance - distance_jack_run 

theorem harper_jack_distance_apart :
  distance_apart total_distance distance_jack_run = 848 :=
by
  unfold distance_apart
  sorry

end harper_jack_distance_apart_l1424_142470


namespace sym_axis_of_curve_eq_zero_b_plus_d_l1424_142424

theorem sym_axis_of_curve_eq_zero_b_plus_d
  (a b c d : ℝ)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hc : c ≠ 0)
  (hd : d ≠ 0)
  (h_symm : ∀ x : ℝ, 2 * x = (a * ((a * x + b) / (c * x + d)) + b) / (c * ((a * x + b) / (c * x + d)) + d)) :
  b + d = 0 :=
sorry

end sym_axis_of_curve_eq_zero_b_plus_d_l1424_142424


namespace g_at_1001_l1424_142432

open Function

variable (g : ℝ → ℝ)

axiom g_property : ∀ x y : ℝ, g (x * y) + 2 * x = x * g y + g x
axiom g_at_1 : g 1 = 3

theorem g_at_1001 : g 1001 = -997 :=
by
  sorry

end g_at_1001_l1424_142432


namespace right_angle_locus_l1424_142497

noncomputable def P (x y : ℝ) : Prop :=
  let M : ℝ × ℝ := (-2, 0)
  let N : ℝ × ℝ := (2, 0)
  (x + 2)^2 + y^2 + (x - 2)^2 + y^2 = 16

theorem right_angle_locus (x y : ℝ) : P x y → x^2 + y^2 = 4 ∧ x ≠ 2 ∧ x ≠ -2 :=
by
  sorry

end right_angle_locus_l1424_142497


namespace pi_bounds_l1424_142425

theorem pi_bounds :
  3 < Real.pi ∧ Real.pi < 4 :=
by
  sorry

end pi_bounds_l1424_142425


namespace find_abc_l1424_142440

def rearrangements (a b c : ℕ) : List ℕ :=
  [100 * a + 10 * b + c, 100 * a + 10 * c + b, 100 * b + 10 * a + c,
   100 * b + 10 * c + a, 100 * c + 10 * a + b, 100 * c + 10 * b + a]

theorem find_abc (a b c : ℕ) (habc : ℕ) :
  a ≠ 0 → b ≠ 0 → c ≠ 0 →
  (rearrangements a b c).sum = 2017 + habc →
  habc = 425 :=
by
  sorry

end find_abc_l1424_142440


namespace eighth_term_is_79_l1424_142457

variable (a d : ℤ)

def fourth_term_condition : Prop := a + 3 * d = 23
def sixth_term_condition : Prop := a + 5 * d = 51

theorem eighth_term_is_79 (h₁ : fourth_term_condition a d) (h₂ : sixth_term_condition a d) : a + 7 * d = 79 :=
sorry

end eighth_term_is_79_l1424_142457


namespace distinct_roots_condition_l1424_142473

theorem distinct_roots_condition (a : ℝ) : 
  (∃ (x1 x2 x3 x4 : ℝ), (x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4) ∧ 
  (|x1^2 - 4| = a * x1 + 6) ∧ (|x2^2 - 4| = a * x2 + 6) ∧ (|x3^2 - 4| = a * x3 + 6) ∧ (|x4^2 - 4| = a * x4 + 6)) ↔ 
  ((-3 < a ∧ a < -2 * Real.sqrt 2) ∨ (2 * Real.sqrt 2 < a ∧ a < 3)) := sorry

end distinct_roots_condition_l1424_142473


namespace problem_D_l1424_142436

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables {a b c : V}

def is_parallel (u v : V) : Prop := ∃ k : ℝ, u = k • v

theorem problem_D (h₁ : is_parallel a b) (h₂ : is_parallel b c) (h₃ : b ≠ 0) : is_parallel a c :=
sorry

end problem_D_l1424_142436


namespace yarn_for_second_ball_l1424_142446

variable (first_ball second_ball third_ball : ℝ) (yarn_used : ℝ)

-- Conditions
variable (h1 : first_ball = second_ball / 2)
variable (h2 : third_ball = 3 * first_ball)
variable (h3 : third_ball = 27)

-- Question: Prove that the second ball used 18 feet of yarn.
theorem yarn_for_second_ball (h1 : first_ball = second_ball / 2) (h2 : third_ball = 3 * first_ball) (h3 : third_ball = 27) :
  second_ball = 18 := by
  sorry

end yarn_for_second_ball_l1424_142446


namespace sufficient_but_not_necessary_l1424_142493

theorem sufficient_but_not_necessary (a b : ℝ) :
  ((a - b) ^ 3 * b ^ 2 > 0 → a > b) ∧ ¬(a > b → (a - b) ^ 3 * b ^ 2 > 0) :=
by
  sorry

end sufficient_but_not_necessary_l1424_142493


namespace speed_in_still_water_l1424_142439

namespace SwimmingProblem

variable (V_m V_s : ℝ)

-- Downstream condition
def downstream_condition : Prop := V_m + V_s = 18

-- Upstream condition
def upstream_condition : Prop := V_m - V_s = 13

-- The main theorem stating the problem
theorem speed_in_still_water (h_downstream : downstream_condition V_m V_s) 
                             (h_upstream : upstream_condition V_m V_s) :
    V_m = 15.5 :=
by
  sorry

end SwimmingProblem

end speed_in_still_water_l1424_142439


namespace teresa_science_marks_l1424_142467

-- Definitions for the conditions
def music_marks : ℕ := 80
def social_studies_marks : ℕ := 85
def physics_marks : ℕ := music_marks / 2
def total_marks : ℕ := 275

-- Statement to prove
theorem teresa_science_marks : ∃ S : ℕ, 
  S + music_marks + social_studies_marks + physics_marks = total_marks ∧ S = 70 :=
sorry

end teresa_science_marks_l1424_142467


namespace find_f_m_l1424_142482

noncomputable def f (x : ℝ) := x^5 + Real.tan x - 3

theorem find_f_m (m : ℝ) (h : f (-m) = -2) : f m = -4 :=
sorry

end find_f_m_l1424_142482


namespace area_of_paper_l1424_142428

-- Define the variables and conditions
variable (L W : ℝ)
variable (h1 : 2 * L + 4 * W = 34)
variable (h2 : 4 * L + 2 * W = 38)

-- Statement to prove
theorem area_of_paper : L * W = 35 := 
by
  sorry

end area_of_paper_l1424_142428


namespace mirror_area_l1424_142481

/-- The outer dimensions of the frame are given as 100 cm by 140 cm,
and the frame width is 15 cm. We aim to prove that the area of the mirror
inside the frame is 7700 cm². -/
theorem mirror_area (W H F: ℕ) (hW : W = 100) (hH : H = 140) (hF : F = 15) :
  (W - 2 * F) * (H - 2 * F) = 7700 :=
by
  sorry

end mirror_area_l1424_142481


namespace projectile_reaches_24_meters_l1424_142447

theorem projectile_reaches_24_meters (h : ℝ) (t : ℝ) (v₀ : ℝ) :
  (h = -4.9 * t^2 + 19.6 * t) ∧ (h = 24) → t = 4 :=
by
  intros
  sorry

end projectile_reaches_24_meters_l1424_142447


namespace min_value_inequality_l1424_142485

theorem min_value_inequality (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x + y + z = 2) :
  ∃ n : ℝ, n = 9 / 4 ∧ (∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 → x + y + z = 2 → (1 / (x + y) + 1 / (y + z) + 1 / (z + x)) ≥ n) :=
sorry

end min_value_inequality_l1424_142485


namespace compute_complex_power_l1424_142448

noncomputable def complex_number := Complex.exp (Complex.I * 125 * Real.pi / 180)

theorem compute_complex_power :
  (complex_number ^ 28) = Complex.ofReal (-Real.cos (40 * Real.pi / 180)) + Complex.I * Real.sin (40 * Real.pi / 180) :=
by
  sorry

end compute_complex_power_l1424_142448


namespace teds_age_l1424_142416

theorem teds_age (s t : ℕ) (h1 : t = 3 * s - 20) (h2 : t + s = 76) : t = 52 :=
by
  sorry

end teds_age_l1424_142416


namespace susana_chocolate_chips_l1424_142477

theorem susana_chocolate_chips :
  ∃ (S_c : ℕ), 
  (∃ (V_c V_v S_v : ℕ), 
    V_c = S_c + 5 ∧
    S_v = (3 * V_v) / 4 ∧
    V_v = 20 ∧
    V_c + S_c + V_v + S_v = 90) ∧
  S_c = 25 :=
by
  existsi 25
  sorry

end susana_chocolate_chips_l1424_142477


namespace least_tiles_needed_l1424_142415

-- Define the conditions
def hallway_length_ft : ℕ := 18
def hallway_width_ft : ℕ := 6
def tile_side_in : ℕ := 6
def feet_to_inches (ft : ℕ) : ℕ := ft * 12

-- Translate conditions
def hallway_length_in := feet_to_inches hallway_length_ft
def hallway_width_in := feet_to_inches hallway_width_ft

-- Define the areas
def hallway_area : ℕ := hallway_length_in * hallway_width_in
def tile_area : ℕ := tile_side_in * tile_side_in

-- State the theorem to be proved
theorem least_tiles_needed :
  hallway_area / tile_area = 432 := 
sorry

end least_tiles_needed_l1424_142415


namespace maximum_value_of_function_y_l1424_142431

noncomputable def function_y (x : ℝ) : ℝ :=
  x * (3 - 2 * x)

theorem maximum_value_of_function_y : ∃ (x : ℝ), 0 < x ∧ x ≤ 1 ∧ function_y x = 9 / 8 :=
by
  sorry

end maximum_value_of_function_y_l1424_142431


namespace find_term_number_l1424_142456

theorem find_term_number :
  ∃ n : ℕ, (2 * (5 : ℝ)^(1/2) = (3 * (n : ℝ) - 1)^(1/2)) ∧ n = 7 :=
sorry

end find_term_number_l1424_142456


namespace equal_ratios_l1424_142404

variable (x y : ℝ)

-- Conditions
def wire_split_to_form_square_and_pentagon (x y : ℝ) : Prop :=
  4 * (x / 4) = 5 * (y / 5)

-- Theorem to prove
theorem equal_ratios (x y : ℝ) (h : wire_split_to_form_square_and_pentagon x y) : x / y = 1 :=
  sorry

end equal_ratios_l1424_142404


namespace fraction_of_fliers_sent_out_l1424_142423

-- Definitions based on the conditions
def total_fliers : ℕ := 2500
def fliers_next_day : ℕ := 1500

-- Defining the fraction sent in the morning as x
variable (x : ℚ)

-- The remaining fliers after morning
def remaining_fliers_morning := (1 - x) * total_fliers

-- The remaining fliers after afternoon
def remaining_fliers_afternoon := remaining_fliers_morning - (1/4) * remaining_fliers_morning

-- The theorem statement
theorem fraction_of_fliers_sent_out :
  remaining_fliers_afternoon = fliers_next_day → x = 1/5 :=
sorry

end fraction_of_fliers_sent_out_l1424_142423


namespace probability_of_urn_contains_nine_red_and_four_blue_after_operations_l1424_142472

-- Definition of the initial urn state
def initial_red_balls : ℕ := 2
def initial_blue_balls : ℕ := 1

-- Definition of the number of operations
def num_operations : ℕ := 5

-- Definition of the final state
def final_red_balls : ℕ := 9
def final_blue_balls : ℕ := 4

-- Definition of total number of balls after five operations
def total_balls_after_operations : ℕ := 13

-- The probability we aim to prove
def target_probability : ℚ := 1920 / 10395

noncomputable def george_experiment_probability_theorem 
  (initial_red_balls initial_blue_balls num_operations final_red_balls final_blue_balls : ℕ)
  (total_balls_after_operations : ℕ) : ℚ :=
if initial_red_balls = 2 ∧ initial_blue_balls = 1 ∧ num_operations = 5 ∧ final_red_balls = 9 ∧ final_blue_balls = 4 ∧ total_balls_after_operations = 13 then
  target_probability
else
  0

-- The theorem statement, no proof provided (using sorry).
theorem probability_of_urn_contains_nine_red_and_four_blue_after_operations :
  george_experiment_probability_theorem 2 1 5 9 4 13 = target_probability := sorry

end probability_of_urn_contains_nine_red_and_four_blue_after_operations_l1424_142472


namespace roots_eq_squares_l1424_142463

theorem roots_eq_squares (p q : ℝ) (h1 : p^2 - 5 * p + 6 = 0) (h2 : q^2 - 5 * q + 6 = 0) :
  p^2 + q^2 = 13 :=
sorry

end roots_eq_squares_l1424_142463


namespace calculate_value_l1424_142491

-- Given conditions
def n : ℝ := 2.25

-- Lean statement to express the proof problem
theorem calculate_value : (n / 3) * 12 = 9 := by
  -- Proof will be supplied here
  sorry

end calculate_value_l1424_142491


namespace maximum_distance_l1424_142449

-- Given conditions for the problem.
def highway_mpg : ℝ := 12.2
def city_mpg : ℝ := 7.6
def gasoline : ℝ := 23

-- Problem statement: prove the maximum distance on highway mileage.
theorem maximum_distance : highway_mpg * gasoline = 280.6 :=
sorry

end maximum_distance_l1424_142449


namespace rectangular_field_area_l1424_142487

theorem rectangular_field_area :
  ∃ (w l : ℝ), (l = 3 * w) ∧ (2 * (l + w) = 72) ∧ (l * w = 243) :=
by {
  sorry
}

end rectangular_field_area_l1424_142487
