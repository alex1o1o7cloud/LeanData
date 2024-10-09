import Mathlib

namespace translated_upwards_2_units_l1867_186781

theorem translated_upwards_2_units (x : ℝ) : (x + 2 > 0) → (x > -2) :=
by 
  intros h
  exact sorry

end translated_upwards_2_units_l1867_186781


namespace inequality_abc_l1867_186728

theorem inequality_abc (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (1/a) + (1/b) ≥ 4/(a + b) :=
by
  sorry

end inequality_abc_l1867_186728


namespace second_discount_percentage_l1867_186701

-- Define the initial conditions.
def listed_price : ℝ := 200
def first_discount_rate : ℝ := 0.20
def final_sale_price : ℝ := 144

-- Calculate the price after the first discount.
def first_discount_amount := first_discount_rate * listed_price
def price_after_first_discount := listed_price - first_discount_amount

-- Define the second discount amount.
def second_discount_amount := price_after_first_discount - final_sale_price

-- Define the theorem to prove the second discount rate.
theorem second_discount_percentage : 
  (second_discount_amount / price_after_first_discount) * 100 = 10 :=
by 
  sorry -- Proof placeholder

end second_discount_percentage_l1867_186701


namespace crate_weight_l1867_186725

variable (C : ℝ)
variable (carton_weight : ℝ := 3)
variable (total_weight : ℝ := 96)
variable (num_crates : ℝ := 12)
variable (num_cartons : ℝ := 16)

theorem crate_weight :
  (num_crates * C + num_cartons * carton_weight = total_weight) → (C = 4) :=
by
  sorry

end crate_weight_l1867_186725


namespace simplify_and_evaluate_l1867_186765

theorem simplify_and_evaluate (x y : ℤ) (h1 : x = -1) (h2 : y = -2) :
  ((x + y) ^ 2 - (3 * x - y) * (3 * x + y) - 2 * y ^ 2) / (-2 * x) = -2 :=
by 
  sorry

end simplify_and_evaluate_l1867_186765


namespace buratino_loss_l1867_186720

def buratino_dollars_lost (x y : ℕ) : ℕ := 5 * y - 3 * x

theorem buratino_loss :
  ∃ (x y : ℕ), x + y = 50 ∧ 3 * y - 2 * x = 0 ∧ buratino_dollars_lost x y = 10 :=
by {
  sorry
}

end buratino_loss_l1867_186720


namespace corey_lowest_score_l1867_186745

theorem corey_lowest_score
  (e1 e2 e3 e4 : ℕ)
  (h1 : e1 = 84)
  (h2 : e2 = 67)
  (max_score : ∀ (e : ℕ), e ≤ 100)
  (avg_at_least_75 : (e1 + e2 + e3 + e4) / 4 ≥ 75) :
  e3 ≥ 49 ∨ e4 ≥ 49 :=
by
  sorry

end corey_lowest_score_l1867_186745


namespace max_variance_l1867_186741

theorem max_variance (p : ℝ) (h₀ : 0 < p) (h₁ : p < 1) : 
  ∃ q, p * (1 - p) ≤ q ∧ q = 1 / 4 :=
by
  existsi (1 / 4)
  sorry

end max_variance_l1867_186741


namespace mean_value_theorem_for_integrals_l1867_186768

variable {a b : ℝ} (f : ℝ → ℝ)

theorem mean_value_theorem_for_integrals (h_cont : ContinuousOn f (Set.Icc a b)) :
  ∃ ξ ∈ Set.Icc a b, ∫ x in a..b, f x = f ξ * (b - a) :=
sorry

end mean_value_theorem_for_integrals_l1867_186768


namespace square_perimeter_l1867_186787

-- We define a structure for a square with an area as a condition.
structure Square (s : ℝ) :=
(area_eq : s ^ 2 = 400)

-- The theorem states that given the area of the square is 400 square meters,
-- the perimeter of the square is 80 meters.
theorem square_perimeter (s : ℝ) (sq : Square s) : 4 * s = 80 :=
by
  -- proof omitted
  sorry

end square_perimeter_l1867_186787


namespace find_k_l1867_186762

theorem find_k (k : ℝ) :
    (1 - 7) * (k - 3) = (3 - k) * (7 - 1) → k = 6.5 :=
by
sorry

end find_k_l1867_186762


namespace parabola_x_intercepts_l1867_186724

theorem parabola_x_intercepts :
  ∃! y : ℝ, -3 * y^2 + 2 * y + 4 = y := 
by
  sorry

end parabola_x_intercepts_l1867_186724


namespace dave_winfield_home_runs_l1867_186777

theorem dave_winfield_home_runs (W : ℕ) (h : 755 = 2 * W - 175) : W = 465 :=
by
  sorry

end dave_winfield_home_runs_l1867_186777


namespace find_sum_of_a_and_c_l1867_186784

variable (a b c d : ℝ)

theorem find_sum_of_a_and_c (h1 : a * b + b * c + c * d + d * a = 48) (h2 : b + d = 6) :
  a + c = 8 := by sorry

end find_sum_of_a_and_c_l1867_186784


namespace product_of_consecutive_integers_is_square_l1867_186776

theorem product_of_consecutive_integers_is_square (x : ℤ) : 
  x * (x + 1) * (x + 2) * (x + 3) + 1 = (x^2 + 3 * x + 1) ^ 2 :=
by
  sorry

end product_of_consecutive_integers_is_square_l1867_186776


namespace probability_correct_l1867_186731

/-- 
The set of characters in "HMMT2005".
-/
def characters : List Char := ['H', 'M', 'M', 'T', '2', '0', '0', '5']

/--
The number of ways to choose 4 positions out of 8.
-/
def choose_4_from_8 : ℕ := Nat.choose 8 4

/-- 
The factorial of an integer n.
-/
def factorial : ℕ → ℕ
| 0     => 1
| (n+1) => (n+1) * factorial n

/-- 
The number of ways to arrange "HMMT".
-/
def arrangements_hmmt : ℕ := choose_4_from_8 * (factorial 4 / factorial 2)

/-- 
The number of ways to arrange "2005".
-/
def arrangements_2005 : ℕ := choose_4_from_8 * (factorial 4 / factorial 2)

/-- 
The number of arrangements where both "HMMT" and "2005" appear.
-/
def arrangements_both : ℕ := choose_4_from_8

/-- 
The total number of possible arrangements of "HMMT2005".
-/
def total_arrangements : ℕ := factorial 8 / (factorial 2 * factorial 2)

/-- 
The number of desirable arrangements using inclusion-exclusion.
-/
def desirable_arrangements : ℕ := arrangements_hmmt + arrangements_2005 - arrangements_both

/-- 
The probability of being able to read either "HMMT" or "2005" 
in a random arrangement of "HMMT2005".
-/
def probability : ℚ := (desirable_arrangements : ℚ) / (total_arrangements : ℚ)

/-- 
Prove that the computed probability is equal to 23/144.
-/
theorem probability_correct : probability = 23 / 144 := sorry

end probability_correct_l1867_186731


namespace box_upper_surface_area_l1867_186703

theorem box_upper_surface_area (L W H : ℕ) 
    (h1 : L * W = 120) 
    (h2 : L * H = 72) 
    (h3 : L * W * H = 720) : 
    L * W = 120 := 
by 
  sorry

end box_upper_surface_area_l1867_186703


namespace jordans_score_l1867_186785

theorem jordans_score 
  (N : ℕ) 
  (first_19_avg : ℚ) 
  (total_avg : ℚ)
  (total_score_19 : ℚ) 
  (total_score_20 : ℚ) 
  (jordan_score : ℚ) 
  (h1 : N = 19)
  (h2 : first_19_avg = 74)
  (h3 : total_avg = 76)
  (h4 : total_score_19 = N * first_19_avg)
  (h5 : total_score_20 = (N + 1) * total_avg)
  (h6 : jordan_score = total_score_20 - total_score_19) :
  jordan_score = 114 :=
by {
  -- the proof will be filled in, but for now we use sorry
  sorry
}

end jordans_score_l1867_186785


namespace min_value_quadratic_l1867_186757

theorem min_value_quadratic (x : ℝ) : ∃ x, x = -7 ∧ (x^2 + 14 * x + 24 = -25) := sorry

end min_value_quadratic_l1867_186757


namespace cubic_ineq_solution_l1867_186786

theorem cubic_ineq_solution (x : ℝ) :
  (4 < x ∧ x < 4 + 2 * Real.sqrt 3) ∨ (x > 4 + 2 * Real.sqrt 3) → (x^3 - 12 * x^2 + 44 * x - 16 > 0) :=
by
  sorry

end cubic_ineq_solution_l1867_186786


namespace sum_local_values_l1867_186735

theorem sum_local_values :
  let local_value_2 := 2000
  let local_value_3 := 300
  let local_value_4 := 40
  let local_value_5 := 5
  local_value_2 + local_value_3 + local_value_4 + local_value_5 = 2345 :=
by
  sorry

end sum_local_values_l1867_186735


namespace average_salary_rest_of_workers_l1867_186788

theorem average_salary_rest_of_workers
  (avg_salary_all : ℝ)
  (num_all_workers : ℕ)
  (avg_salary_techs : ℝ)
  (num_techs : ℕ)
  (avg_salary_rest : ℝ)
  (num_rest : ℕ) :
  avg_salary_all = 8000 →
  num_all_workers = 21 →
  avg_salary_techs = 12000 →
  num_techs = 7 →
  num_rest = num_all_workers - num_techs →
  avg_salary_rest = (avg_salary_all * num_all_workers - avg_salary_techs * num_techs) / num_rest →
  avg_salary_rest = 6000 :=
by
  intros h_avg_all h_num_all h_avg_techs h_num_techs h_num_rest h_avg_rest
  sorry

end average_salary_rest_of_workers_l1867_186788


namespace daniel_dolls_l1867_186727

theorem daniel_dolls (normal_price discount_price: ℕ) 
  (normal_dolls: ℕ) 
  (saved_money: ℕ := normal_dolls * normal_price):
  normal_price = 4 →
  normal_dolls = 15 →
  discount_price = 3 →
  saved_money = normal_dolls * normal_price →
  saved_money / discount_price = 20 :=
by
  sorry

end daniel_dolls_l1867_186727


namespace find_n_l1867_186730

theorem find_n (n : ℤ) : 
  50 < n ∧ n < 120 ∧ (n % 8 = 0) ∧ (n % 7 = 3) ∧ (n % 9 = 3) → n = 192 :=
by
  sorry

end find_n_l1867_186730


namespace trains_cross_time_l1867_186751

theorem trains_cross_time (length1 length2 : ℕ) (time1 time2 : ℕ) 
  (speed1 speed2 relative_speed total_length : ℚ) 
  (h1 : length1 = 120) (h2 : length2 = 150) 
  (h3 : time1 = 10) (h4 : time2 = 15) 
  (h5 : speed1 = length1 / time1) (h6 : speed2 = length2 / time2) 
  (h7 : relative_speed = speed1 - speed2) 
  (h8 : total_length = length1 + length2) : 
  (total_length / relative_speed = 135) := 
by sorry

end trains_cross_time_l1867_186751


namespace find_y_l1867_186795

theorem find_y (x y : ℝ) (hA : {2, Real.log x} = {a | a = 2 ∨ a = Real.log x})
                (hB : {x, y} = {a | a = x ∨ a = y})
                (hInt : {a | a = 2 ∨ a = Real.log x} ∩ {a | a = x ∨ a = y} = {0}) :
  y = 0 :=
  sorry

end find_y_l1867_186795


namespace ratio_proof_l1867_186749

noncomputable def ratio_of_segment_lengths (a b : ℝ) (points : Finset (ℝ × ℝ)) : Prop :=
  points.card = 5 ∧
  ∃ (dists : Finset ℝ), 
    dists = {a, a, a, a, a, b, 3 * a} ∧
    ∀ (p1 p2 : ℝ × ℝ), p1 ∈ points → p2 ∈ points → p1 ≠ p2 →
      (dist p1 p2 ∈ dists)

theorem ratio_proof (a b : ℝ) (points : Finset (ℝ × ℝ)) (h : ratio_of_segment_lengths a b points) : 
  b / a = 2.8 :=
sorry

end ratio_proof_l1867_186749


namespace calculate_expression_l1867_186783

theorem calculate_expression : 287 * 287 + 269 * 269 - (2 * 287 * 269) = 324 :=
by
  sorry

end calculate_expression_l1867_186783


namespace scientific_notation_of_750000_l1867_186748

theorem scientific_notation_of_750000 : 750000 = 7.5 * 10^5 :=
by
  sorry

end scientific_notation_of_750000_l1867_186748


namespace hyperbola_center_l1867_186794

theorem hyperbola_center (x y : ℝ) :
    9 * x^2 - 18 * x - 16 * y^2 + 64 * y - 143 = 0 →
    (x, y) = (1, 2) :=
sorry

end hyperbola_center_l1867_186794


namespace smallest_Norwegian_l1867_186764

def is_Norwegian (n : ℕ) : Prop :=
  ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ n = a * b * c ∧ a + b + c = 2022

theorem smallest_Norwegian :
  ∀ n : ℕ, is_Norwegian n → 1344 ≤ n := by
  sorry

end smallest_Norwegian_l1867_186764


namespace min_distance_to_line_l1867_186733

theorem min_distance_to_line (m n : ℝ) (h : 4 * m + 3 * n = 10)
  : m^2 + n^2 ≥ 4 :=
sorry

end min_distance_to_line_l1867_186733


namespace parkway_girls_not_playing_soccer_l1867_186796

theorem parkway_girls_not_playing_soccer (total_students boys soccer_students : ℕ) 
    (percent_boys_playing_soccer : ℕ) 
    (h1 : total_students = 420)
    (h2 : boys = 312)
    (h3 : soccer_students = 250)
    (h4 : percent_boys_playing_soccer = 86) :
   (total_students - boys - (soccer_students - soccer_students * percent_boys_playing_soccer / 100)) = 73 :=
by sorry

end parkway_girls_not_playing_soccer_l1867_186796


namespace pitbull_chihuahua_weight_ratio_l1867_186778

theorem pitbull_chihuahua_weight_ratio
  (C P G : ℕ)
  (h1 : G = 307)
  (h2 : G = 3 * P + 10)
  (h3 : C + P + G = 439) :
  P / C = 3 :=
by {
  sorry
}

end pitbull_chihuahua_weight_ratio_l1867_186778


namespace find_k_l1867_186710

theorem find_k (k r s : ℝ) 
  (h1 : r + s = -k) 
  (h2 : r * s = 12) 
  (h3 : (r + 7) + (s + 7) = k) : 
  k = 7 := by 
  sorry

end find_k_l1867_186710


namespace no_integers_with_cube_sum_l1867_186742

theorem no_integers_with_cube_sum (a b : ℤ) (h1 : a^3 + b^3 = 4099) (h2 : Prime 4099) : false :=
sorry

end no_integers_with_cube_sum_l1867_186742


namespace complex_multiply_cis_l1867_186752

open Complex

theorem complex_multiply_cis :
  (4 * (cos (25 * Real.pi / 180) + sin (25 * Real.pi / 180) * I)) *
  (-3 * (cos (48 * Real.pi / 180) + sin (48 * Real.pi / 180) * I)) =
  12 * (cos (253 * Real.pi / 180) + sin (253 * Real.pi / 180) * I) :=
sorry

end complex_multiply_cis_l1867_186752


namespace product_of_distinct_solutions_l1867_186771

theorem product_of_distinct_solutions (x y : ℝ) (h₁ : x ≠ y) (h₂ : x ≠ 0) (h₃ : y ≠ 0) (h₄ : x - 2 / x = y - 2 / y) :
  x * y = -2 :=
sorry

end product_of_distinct_solutions_l1867_186771


namespace find_x_l1867_186754

theorem find_x (x y : ℝ) (hx : 0 < x) (hy : 0 < y) 
(h_eq : 7 * x^3 + 14 * x^2 * y = x^4 + 2 * x^3 * y) :
  x = 7 :=
by
  sorry

end find_x_l1867_186754


namespace solve_for_y_l1867_186714

theorem solve_for_y (y : ℝ) (h : (2 / y) + (3 / y) / (6 / y) = 1.5) : y = 2 :=
sorry

end solve_for_y_l1867_186714


namespace second_worker_time_l1867_186789

theorem second_worker_time 
  (first_worker_rate : ℝ)
  (combined_rate : ℝ)
  (x : ℝ)
  (h1 : first_worker_rate = 1 / 6)
  (h2 : combined_rate = 1 / 2.4) :
  (1 / 6) + (1 / x) = combined_rate → x = 4 := 
by 
  intros h
  sorry

end second_worker_time_l1867_186789


namespace sum_of_arithmetic_sequence_l1867_186773

noncomputable def arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_of_arithmetic_sequence (a : ℕ → ℝ) 
  (h_seq : arithmetic_sequence a)
  (h_a5 : a 5 = 3)
  (h_a7 : a 7 = 7) :
  a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 35 :=
  sorry

end sum_of_arithmetic_sequence_l1867_186773


namespace calculate_expression_l1867_186711

theorem calculate_expression :
  -2^3 * (-3)^2 / (9 / 8) - abs (1 / 2 - 3 / 2) = -65 :=
by
  sorry

end calculate_expression_l1867_186711


namespace leak_time_l1867_186770

theorem leak_time (A L : ℝ) (PipeA_filling_rate : A = 1 / 6) (Combined_rate : A - L = 1 / 10) : 
  1 / L = 15 :=
by
  sorry

end leak_time_l1867_186770


namespace grandma_vasya_cheapest_option_l1867_186705

/-- Constants and definitions for the cost calculations --/
def train_ticket_cost : ℕ := 200
def collected_berries_kg : ℕ := 5
def market_berries_cost_per_kg : ℕ := 150
def sugar_cost_per_kg : ℕ := 54
def jam_made_per_kg_combination : ℕ := 15 / 10  -- representing 1.5 kg (as ratio 15/10)
def ready_made_jam_cost_per_kg : ℕ := 220

/-- Compute the cost per kg of jam for different methods --/
def cost_per_kg_jam_option1 : ℕ := (train_ticket_cost / collected_berries_kg + sugar_cost_per_kg)
def cost_per_kg_jam_option2 : ℕ := market_berries_cost_per_kg + sugar_cost_per_kg
def cost_per_kg_jam_option3 : ℕ := ready_made_jam_cost_per_kg

/-- Numbers converted to per 1.5 kg --/
def cost_for_1_5_kg (cost_per_kg: ℕ) : ℕ := cost_per_kg * (15 / 10)

/-- Theorem stating option 1 is the cheapest --/
theorem grandma_vasya_cheapest_option :
  cost_for_1_5_kg cost_per_kg_jam_option1 ≤ cost_for_1_5_kg cost_per_kg_jam_option2 ∧
  cost_for_1_5_kg cost_per_kg_jam_option1 ≤ cost_for_1_5_kg cost_per_kg_jam_option3 :=
by sorry

end grandma_vasya_cheapest_option_l1867_186705


namespace gauss_family_mean_age_l1867_186706

theorem gauss_family_mean_age :
  let ages := [8, 8, 8, 8, 16, 17]
  let num_children := 6
  let sum_ages := 65
  (sum_ages : ℚ) / (num_children : ℚ) = 65 / 6 :=
by
  sorry

end gauss_family_mean_age_l1867_186706


namespace two_colonies_reach_limit_same_time_l1867_186758

theorem two_colonies_reach_limit_same_time
  (doubles_in_size : ∀ (n : ℕ), n = n * 2)
  (reaches_limit_in_25_days : ∃ N : ℕ, ∀ t : ℕ, t = 25 → N = N * 2^t) :
  ∀ t : ℕ, t = 25 := sorry

end two_colonies_reach_limit_same_time_l1867_186758


namespace line_parabola_intersection_one_point_l1867_186732

theorem line_parabola_intersection_one_point (k : ℝ) :
  (∃ y : ℝ, (-3 * y^2 - 4 * y + 7 = k) ∧ ∀ y1 y2 : ℝ, ( 3 * y1^2 + 4 * y1 + (k - 7) = 0 → 3 * y2^2 + 4 * y2 + (k - 7) = 0 → y1 = y2)) ↔ (k = 25 / 3) :=
by
  sorry

end line_parabola_intersection_one_point_l1867_186732


namespace total_amount_paid_is_correct_l1867_186722

-- Define constants based on conditions
def cost_per_night_per_person : ℕ := 40
def number_of_people : ℕ := 3
def number_of_nights : ℕ := 3

-- Define the total cost calculation
def total_cost := cost_per_night_per_person * number_of_people * number_of_nights

-- The proof problem statement
theorem total_amount_paid_is_correct :
  total_cost = 360 :=
by
  sorry

end total_amount_paid_is_correct_l1867_186722


namespace largest_modulus_z_l1867_186736

open Complex

noncomputable def z_largest_value (a b c z : ℂ) : ℝ :=
  (1 + Real.sqrt 5) / 2

theorem largest_modulus_z (a b c z : ℂ) (r : ℝ) (hr_pos : 0 < r)
  (hmod_a : Complex.abs a = r) (hmod_b : Complex.abs b = r) (hmod_c : Complex.abs c = r)
  (heqn : a * z ^ 2 + b * z + c = 0) :
  Complex.abs z ≤ z_largest_value a b c z :=
sorry

end largest_modulus_z_l1867_186736


namespace find_number_l1867_186729

theorem find_number (x : ℝ) 
  (h : 0.4 * x + (0.3 * 0.2) = 0.26) : x = 0.5 := 
by
  sorry

end find_number_l1867_186729


namespace subset_condition_l1867_186700

theorem subset_condition (a : ℝ) :
  (∀ x : ℝ, |2 * x - 1| < 1 → x^2 - 2 * a * x + a^2 - 1 > 0) →
  (a ≤ -1 ∨ a ≥ 2) :=
by
  sorry

end subset_condition_l1867_186700


namespace smallest_x_value_l1867_186756

theorem smallest_x_value (x : ℤ) (h : 3 * x^2 - 4 < 20) : x = -2 :=
sorry

end smallest_x_value_l1867_186756


namespace jo_thinking_number_l1867_186719

theorem jo_thinking_number 
  (n : ℕ) 
  (h1 : n < 100) 
  (h2 : n % 8 = 7) 
  (h3 : n % 7 = 4) 
  : n = 95 :=
sorry

end jo_thinking_number_l1867_186719


namespace prove_M_squared_l1867_186737

noncomputable def M (x : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![ -1, 2], ![ (5/2:ℝ), x]]

def eigenvalue_condition (x : ℝ) : Prop :=
  let A := M x
  ∃ v : ℝ, (A - (-2) • (1 : Matrix (Fin 2) (Fin 2) ℝ)).det = 0

theorem prove_M_squared (x : ℝ) (h : eigenvalue_condition x) :
  (M x * M x) = ![![ 6, -9], ![ - (45/4:ℝ), 69/4]] :=
sorry

end prove_M_squared_l1867_186737


namespace chicken_nuggets_order_l1867_186791

theorem chicken_nuggets_order (cost_per_box : ℕ) (nuggets_per_box : ℕ) (total_amount_paid : ℕ) 
  (h1 : cost_per_box = 4) (h2 : nuggets_per_box = 20) (h3 : total_amount_paid = 20) : 
  total_amount_paid / cost_per_box * nuggets_per_box = 100 :=
by
  -- This is where the proof would go
  sorry

end chicken_nuggets_order_l1867_186791


namespace avg_velocity_2_to_2_1_l1867_186755

def motion_eq (t : ℝ) : ℝ := 3 + t^2

theorem avg_velocity_2_to_2_1 : 
  (motion_eq 2.1 - motion_eq 2) / (2.1 - 2) = 4.1 :=
by
  sorry

end avg_velocity_2_to_2_1_l1867_186755


namespace minimum_value_expression_l1867_186715

theorem minimum_value_expression :
  ∃ x y : ℝ, ∀ x y : ℝ, 3 * x ^ 2 + 4 * x * y + 2 * y ^ 2 - 6 * x + 8 * y + 9 ≥ -10 :=
by
  sorry

end minimum_value_expression_l1867_186715


namespace sin_double_angle_value_l1867_186702

open Real

theorem sin_double_angle_value (x : ℝ) (h : sin (x + π / 4) = - 5 / 13) : sin (2 * x) = - 119 / 169 := 
sorry

end sin_double_angle_value_l1867_186702


namespace steve_popsicle_sticks_l1867_186743

theorem steve_popsicle_sticks (S Sid Sam : ℕ) (h1 : Sid = 2 * S) (h2 : Sam = 3 * Sid) (h3 : S + Sid + Sam = 108) : S = 12 :=
by
  sorry

end steve_popsicle_sticks_l1867_186743


namespace Priya_driving_speed_l1867_186761

/-- Priya's driving speed calculation -/
theorem Priya_driving_speed
  (time_XZ : ℝ) (rate_back : ℝ) (time_ZY : ℝ)
  (midway_condition : time_XZ = 5)
  (speed_back_condition : rate_back = 60)
  (time_back_condition : time_ZY = 2.0833333333333335) :
  ∃ speed_XZ : ℝ, speed_XZ = 50 :=
by
  have distance_ZY : ℝ := rate_back * time_ZY
  have distance_XZ : ℝ := 2 * distance_ZY
  have speed_XZ : ℝ := distance_XZ / time_XZ
  existsi speed_XZ
  sorry

end Priya_driving_speed_l1867_186761


namespace gas_cost_per_gallon_is_4_l1867_186718

noncomputable def cost_per_gallon (miles_per_gallon : ℝ) (total_miles : ℝ) (total_cost : ℝ) : ℝ :=
  total_cost / (total_miles / miles_per_gallon)

theorem gas_cost_per_gallon_is_4 :
  cost_per_gallon 32 432 54 = 4 := by
  sorry

end gas_cost_per_gallon_is_4_l1867_186718


namespace find_m_l1867_186716

noncomputable def polynomial (x : ℝ) (m : ℝ) := 4 * x^2 - 3 * x + 5 - 2 * m * x^2 - x + 1

theorem find_m (m : ℝ) : 
  ∀ x : ℝ, (4 * x^2 - 3 * x + 5 - 2 * m * x^2 - x + 1 = (4 - 2 * m) * x^2 - 4 * x + 6)
  → (4 - 2 * m = 0) → (m = 2) :=
by
  intros x h1 h2
  sorry

end find_m_l1867_186716


namespace gcd_of_1230_and_990_l1867_186760

theorem gcd_of_1230_and_990 : Nat.gcd 1230 990 = 30 :=
by
  sorry

end gcd_of_1230_and_990_l1867_186760


namespace function_f_not_all_less_than_half_l1867_186709

theorem function_f_not_all_less_than_half (p q : ℝ) (f : ℝ → ℝ) (h1 : ∀ x : ℝ, f x = x^2 + p*x + q) :
  ¬ (|f 1| < 1 / 2 ∧ |f 2| < 1 / 2 ∧ |f 3| < 1 / 2) :=
sorry

end function_f_not_all_less_than_half_l1867_186709


namespace inequality_solution_l1867_186790

theorem inequality_solution (x : ℝ) : (4 + 2 * x > -6) → (x > -5) :=
by sorry

end inequality_solution_l1867_186790


namespace edith_books_total_l1867_186782

-- Define the conditions
def novels := 80
def writing_books := novels * 2

-- Theorem statement
theorem edith_books_total : novels + writing_books = 240 :=
by sorry

end edith_books_total_l1867_186782


namespace sum_of_possible_values_of_a_l1867_186717

theorem sum_of_possible_values_of_a :
  ∀ (a b c d : ℝ), a > b → b > c → c > d → a + b + c + d = 50 → 
  (a - b = 4 ∧ b - d = 7 ∧ a - c = 5 ∧ c - d = 6 ∧ b - c = 2 ∨
   a - b = 5 ∧ b - d = 6 ∧ a - c = 4 ∧ c - d = 7 ∧ b - c = 2) →
  (a = 17.75 ∨ a = 18.25) →
  a + 18.25 + 17.75 - a = 36 :=
by sorry

end sum_of_possible_values_of_a_l1867_186717


namespace coin_flip_sequences_count_l1867_186779

theorem coin_flip_sequences_count : 
  let total_flips := 10;
  let heads_fixed := 2;
  (2 : ℕ) ^ (total_flips - heads_fixed) = 256 := 
by 
  sorry

end coin_flip_sequences_count_l1867_186779


namespace volume_of_solid_bounded_by_planes_l1867_186708

theorem volume_of_solid_bounded_by_planes (a : ℝ) : 
  ∃ v, v = (a ^ 3) / 6 :=
by 
  sorry

end volume_of_solid_bounded_by_planes_l1867_186708


namespace terminal_side_equiv_l1867_186774

theorem terminal_side_equiv (θ : ℝ) (hθ : θ = 23 * π / 3) : 
  ∃ k : ℤ, θ = 2 * π * k + 5 * π / 3 := by
  sorry

end terminal_side_equiv_l1867_186774


namespace reward_function_conditions_l1867_186704

theorem reward_function_conditions :
  (∀ x : ℝ, (10 ≤ x ∧ x ≤ 1000) → (y = x / 150 + 2 → y ≤ 90 ∧ y ≤ x / 5) → False) ∧
  (∃ a : ℕ, (∀ x : ℝ, (10 ≤ x ∧ x ≤ 1000) → (y = (10 * x - 3 * a) / (x + 2) → y ≤ 9 ∧ y ≤ x / 5)) ∧ (a = 328)) :=
by
  sorry

end reward_function_conditions_l1867_186704


namespace functional_equation_zero_l1867_186723

open Function

theorem functional_equation_zero (f : ℕ+ → ℝ) 
  (h : ∀ (m n : ℕ+), n ≥ m → f (n + m) + f (n - m) = f (3 * n)) :
  ∀ n : ℕ+, f n = 0 := sorry

end functional_equation_zero_l1867_186723


namespace find_dividend_l1867_186793

noncomputable def dividend (divisor quotient remainder : ℕ) : ℕ :=
  (divisor * quotient) + remainder

theorem find_dividend :
  ∀ (divisor quotient remainder : ℕ), 
  divisor = 16 → 
  quotient = 8 → 
  remainder = 4 → 
  dividend divisor quotient remainder = 132 :=
by
  intros divisor quotient remainder hdiv hquo hrem
  sorry

end find_dividend_l1867_186793


namespace sin_lt_alpha_lt_tan_l1867_186766

open Real

theorem sin_lt_alpha_lt_tan {α : ℝ} (hα1 : 0 < α) (hα2 : α < π / 2) : sin α < α ∧ α < tan α := by
  sorry

end sin_lt_alpha_lt_tan_l1867_186766


namespace petri_dishes_count_l1867_186797

def germs_total : ℕ := 5400000
def germs_per_dish : ℕ := 500
def petri_dishes : ℕ := germs_total / germs_per_dish

theorem petri_dishes_count : petri_dishes = 10800 := by
  sorry

end petri_dishes_count_l1867_186797


namespace minimum_abs_phi_l1867_186707

noncomputable def f (ω φ x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ)

theorem minimum_abs_phi 
  (ω φ b : ℝ)
  (hω : ω > 0)
  (hb : 0 < b ∧ b < 2)
  (h_intersections : f ω φ (π / 6) = b ∧ f ω φ (5 * π / 6) = b ∧ f ω φ (7 * π / 6) = b)
  (h_minimum : f ω φ (3 * π / 2) = -2) : 
  |φ| = π / 2 :=
sorry

end minimum_abs_phi_l1867_186707


namespace expand_expression_l1867_186713

variable (y : ℤ)

theorem expand_expression : 12 * (3 * y - 4) = 36 * y - 48 := 
by
  sorry

end expand_expression_l1867_186713


namespace max_min_values_l1867_186738

def f (x a : ℝ) : ℝ := -x^2 + 2*x + a

theorem max_min_values (a : ℝ) (h : a ≠ 0) :
  (∀ x, 0 ≤ x ∧ x ≤ 3 → f x a ≤ 1 + a) ∧
  (∃ x, 0 ≤ x ∧ x ≤ 3 ∧ f x a = 1 + a) ∧
  (∀ x, 0 ≤ x ∧ x ≤ 3 → -3 + a ≤ f x a) ∧
  (∃ x, 0 ≤ x ∧ x ≤ 3 ∧ f x a = -3 + a) := 
sorry

end max_min_values_l1867_186738


namespace sufficient_condition_for_proposition_l1867_186792

theorem sufficient_condition_for_proposition (a : ℝ) : 
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≤ 0) ↔ (a ≥ 5) :=
by 
  sorry

end sufficient_condition_for_proposition_l1867_186792


namespace nuts_needed_for_cookies_l1867_186712

-- Given conditions
def total_cookies : Nat := 120
def fraction_nuts : Rat := 1 / 3
def fraction_chocolate : Rat := 0.25
def nuts_per_cookie : Nat := 3

-- Translated conditions as helpful functions
def cookies_with_nuts : Nat := Nat.floor (fraction_nuts * total_cookies)
def cookies_with_chocolate : Nat := Nat.floor (fraction_chocolate * total_cookies)
def cookies_with_both : Nat := total_cookies - cookies_with_nuts - cookies_with_chocolate
def total_cookies_with_nuts : Nat := cookies_with_nuts + cookies_with_both
def total_nuts_needed : Nat := total_cookies_with_nuts * nuts_per_cookie

-- Proof problem: proving that total nuts needed is 270
theorem nuts_needed_for_cookies : total_nuts_needed = 270 :=
by
  sorry

end nuts_needed_for_cookies_l1867_186712


namespace problem_r_of_3_eq_88_l1867_186798

def q (x : ℤ) : ℤ := 2 * x - 5
def r (x : ℤ) : ℤ := x^3 + 2 * x^2 - x - 4

theorem problem_r_of_3_eq_88 : r 3 = 88 :=
by
  sorry

end problem_r_of_3_eq_88_l1867_186798


namespace vertical_asymptotes_A_plus_B_plus_C_l1867_186740

noncomputable def A : ℤ := -6
noncomputable def B : ℤ := 5
noncomputable def C : ℤ := 12

theorem vertical_asymptotes_A_plus_B_plus_C :
  (x + 1) * (x - 3) * (x - 4) = x^3 + A*x^2 + B*x + C ∧ A + B + C = 11 := by
  sorry

end vertical_asymptotes_A_plus_B_plus_C_l1867_186740


namespace remainder_product_div_6_l1867_186753

theorem remainder_product_div_6 :
  (3 * 7 * 13 * 17 * 23 * 27 * 33 * 37 * 43 * 47 * 53 * 57 * 63 * 67 * 73 * 77 * 83 * 87 * 93 * 97 
   * 103 * 107 * 113 * 117 * 123 * 127 * 133 * 137 * 143 * 147 * 153 * 157 * 163 * 167 * 173 
   * 177 * 183 * 187 * 193 * 197) % 6 = 3 := 
by 
  -- basic info about modulo arithmetic and properties of sequences
  sorry

end remainder_product_div_6_l1867_186753


namespace part_I_part_II_l1867_186744

def S (n : ℕ) : ℕ := 2 ^ n - 1

def a (n : ℕ) : ℕ := 2 ^ (n - 1)

def T (n : ℕ) : ℕ := (n - 1) * 2 ^ n + 1

theorem part_I (S : ℕ → ℕ) (hS : ∀ n, S n = 2^n - 1) :
  ∀ n, ∃ a : ℕ → ℕ, a n = 2^(n-1) :=
by
  sorry

theorem part_II (a : ℕ → ℕ) (ha : ∀ n, a n = 2^(n-1)) :
  ∀ n, ∃ T : ℕ → ℕ, T n = (n - 1) * 2 ^ n + 1 :=
by
  sorry

end part_I_part_II_l1867_186744


namespace number_of_friends_l1867_186726

-- Definitions based on the given problem conditions
def total_candy := 420
def candy_per_friend := 12

-- Proof statement in Lean 4
theorem number_of_friends : total_candy / candy_per_friend = 35 := by
  sorry

end number_of_friends_l1867_186726


namespace find_a_range_l1867_186767

theorem find_a_range (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ 2 * x^2 - 2 * x + a - 3 = 0) ∧ 
  (∃ y : ℝ, y > 0 ∧ y ≠ x ∧ 2 * y^2 - 2 * y + a - 3 = 0) 
  ↔ 3 < a ∧ a < 7 / 2 := 
sorry

end find_a_range_l1867_186767


namespace Cauchy_solution_on_X_l1867_186739

section CauchyEquation

variable (f : ℝ → ℝ)

def is_morphism (f : ℝ → ℝ) := ∀ x y : ℝ, f (x + y) = f x + f y

theorem Cauchy_solution_on_X :
  (∀ a b : ℤ, ∀ c d : ℤ, a + b * Real.sqrt 2 = c + d * Real.sqrt 2 → a = c ∧ b = d) →
  is_morphism f →
  ∃ x y : ℝ, ∀ a b : ℤ,
    f (a + b * Real.sqrt 2) = a * x + b * y :=
by
  intros h1 h2
  let x := f 1
  let y := f (Real.sqrt 2)
  exists x, y
  intros a b
  sorry

end CauchyEquation

end Cauchy_solution_on_X_l1867_186739


namespace episodes_lost_per_season_l1867_186780

theorem episodes_lost_per_season (s1 s2 : ℕ) (e : ℕ) (remaining : ℕ) (total_seasons : ℕ) (total_episodes_before : ℕ) (total_episodes_lost : ℕ)
  (h1 : s1 = 12) (h2 : s2 = 14) (h3 : e = 16) (h4 : remaining = 364) 
  (h5 : total_seasons = s1 + s2) (h6 : total_episodes_before = s1 * e + s2 * e) 
  (h7 : total_episodes_lost = total_episodes_before - remaining) :
  total_episodes_lost / total_seasons = 2 := by
  sorry

end episodes_lost_per_season_l1867_186780


namespace range_of_a_l1867_186750

-- Define the conditions
def line1 (a x y : ℝ) : Prop := a * x + y - 4 = 0
def line2 (x y : ℝ) : Prop := x - y - 2 = 0
def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

-- The main theorem to state
theorem range_of_a (a : ℝ) : 
  (∃ x y : ℝ, line1 a x y ∧ line2 x y ∧ first_quadrant x y) ↔ -1 < a ∧ a < 2 :=
by
  sorry

end range_of_a_l1867_186750


namespace range_of_m_l1867_186746

-- Definitions based on conditions
def inequality_holds (m : ℝ) : Prop :=
  ∀ x : ℝ, 0 < x → (m * Real.exp x / x ≥ 6 - 4 * x)

-- The statement to be proved
theorem range_of_m (m : ℝ) : inequality_holds m → m ≥ 2 * Real.exp (-(1 / 2)) :=
by
  sorry

end range_of_m_l1867_186746


namespace area_of_path_is_675_l1867_186759

def rectangular_field_length : ℝ := 75
def rectangular_field_width : ℝ := 55
def path_width : ℝ := 2.5

def area_of_path : ℝ :=
  let new_length := rectangular_field_length + 2 * path_width
  let new_width := rectangular_field_width + 2 * path_width
  let area_with_path := new_length * new_width
  let area_of_grass_field := rectangular_field_length * rectangular_field_width
  area_with_path - area_of_grass_field

theorem area_of_path_is_675 : area_of_path = 675 := by
  sorry

end area_of_path_is_675_l1867_186759


namespace complement_union_l1867_186747

open Set Real

noncomputable def S : Set ℝ := {x | x > -2}
noncomputable def T : Set ℝ := {x | x^2 + 3*x - 4 ≤ 0}

theorem complement_union (x : ℝ): x ∈ (univ \ S) ∪ T ↔ x ≤ 1 :=
by
  sorry

end complement_union_l1867_186747


namespace base_length_of_isosceles_triangle_l1867_186775

theorem base_length_of_isosceles_triangle (a b : ℕ) (h1 : a = 8) (h2 : b + 2 * a = 26) : b = 10 :=
by
  have h3 : 2 * 8 = 16 := by norm_num
  rw [h1] at h2
  rw [h3] at h2
  linarith

end base_length_of_isosceles_triangle_l1867_186775


namespace find_a_b_find_m_l1867_186769

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := a * x^3 + b * x^2

theorem find_a_b (a b : ℝ) (h₁ : f 1 a b = 4)
  (h₂ : 3 * a + 2 * b = 9) : a = 1 ∧ b = 3 :=
by
  sorry

theorem find_m (m : ℝ) (h : ∀ x, (m ≤ x ∧ x ≤ m + 1) → (3 * x^2 + 6 * x > 0)) :
  m ≥ 0 ∨ m ≤ -3 :=
by
  sorry

end find_a_b_find_m_l1867_186769


namespace find_number_l1867_186734

theorem find_number (x : ℤ) (h : 3 * (x + 8) = 36) : x = 4 :=
by {
  sorry
}

end find_number_l1867_186734


namespace exponentiation_problem_l1867_186772

theorem exponentiation_problem (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h : 2^a * 2^b = 8) : (2^a)^b = 4 := 
sorry

end exponentiation_problem_l1867_186772


namespace correct_option_l1867_186799

theorem correct_option (a b c d : ℝ) (ha : a < 0) (hb : b > 0) (hd : d < 1) 
  (hA : 2 = (a-1)^2 - 2) (hB : 6 = (b-1)^2 - 2) (hC : d = (c-1)^2 - 2) :
  a < c ∧ c < b :=
by
  sorry

end correct_option_l1867_186799


namespace fraction_is_determined_l1867_186763

theorem fraction_is_determined (y x : ℕ) (h1 : y * 3 = x - 1) (h2 : (y + 4) * 2 = x) : 
  y = 7 ∧ x = 22 :=
by
  sorry

end fraction_is_determined_l1867_186763


namespace square_difference_division_l1867_186721

theorem square_difference_division (a b : ℕ) (h₁ : a = 121) (h₂ : b = 112) :
  (a^2 - b^2) / 9 = 233 :=
by
  sorry

end square_difference_division_l1867_186721
