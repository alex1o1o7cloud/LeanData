import Mathlib

namespace LCM_of_two_numbers_l965_96567

theorem LCM_of_two_numbers (a b : ℕ) (h_hcf : Nat.gcd a b = 11) (h_product : a * b = 1991) : Nat.lcm a b = 181 :=
by
  sorry

end LCM_of_two_numbers_l965_96567


namespace find_t_l965_96570

-- Define sets M and N
def M (t : ℝ) : Set ℝ := {1, t^2}
def N (t : ℝ) : Set ℝ := {-2, t + 2}

-- Goal: prove that t = 2 given M ∩ N ≠ ∅
theorem find_t (t : ℝ) (h : (M t ∩ N t).Nonempty) : t = 2 :=
sorry

end find_t_l965_96570


namespace seq_solution_l965_96590

theorem seq_solution {a b : ℝ} (h1 : a - b = 8) (h2 : a + b = 11) : 2 * a = 19 ∧ 2 * b = 3 := by
  sorry

end seq_solution_l965_96590


namespace symmetric_pentominoes_count_l965_96536

-- Assume we have exactly fifteen pentominoes
def num_pentominoes : ℕ := 15

-- Define the number of pentominoes with particular symmetrical properties
def num_reflectional_symmetry : ℕ := 8
def num_rotational_symmetry : ℕ := 3
def num_both_symmetries : ℕ := 2

-- The theorem we wish to prove
theorem symmetric_pentominoes_count 
  (n_p : ℕ) (n_r : ℕ) (n_b : ℕ) (n_tot : ℕ)
  (h1 : n_p = num_pentominoes)
  (h2 : n_r = num_reflectional_symmetry)
  (h3 : n_b = num_both_symmetries)
  (h4 : n_tot = n_r + num_rotational_symmetry - n_b) :
  n_tot = 9 := 
sorry

end symmetric_pentominoes_count_l965_96536


namespace first_meeting_time_of_boys_l965_96527

theorem first_meeting_time_of_boys 
  (L : ℝ) (v1_kmh : ℝ) (v2_kmh : ℝ) (v1_ms v2_ms : ℝ) (rel_speed : ℝ) (t : ℝ)
  (hv1_km_to_ms : v1_ms = v1_kmh * 1000 / 3600)
  (hv2_km_to_ms : v2_ms = v2_kmh * 1000 / 3600)
  (hrel_speed : rel_speed = v1_ms + v2_ms)
  (hl : L = 4800)
  (hv1 : v1_kmh = 60)
  (hv2 : v2_kmh = 100)
  (ht : t = L / rel_speed) :
  t = 108 := by
  -- we're providing a placeholder for the proof
  sorry

end first_meeting_time_of_boys_l965_96527


namespace number_of_keepers_l965_96534

theorem number_of_keepers (k : ℕ)
  (hens : ℕ := 50)
  (goats : ℕ := 45)
  (camels : ℕ := 8)
  (hen_feet : ℕ := 2)
  (goat_feet : ℕ := 4)
  (camel_feet : ℕ := 4)
  (keeper_feet : ℕ := 2)
  (feet_more_than_heads : ℕ := 224)
  (total_heads : ℕ := hens + goats + camels + k)
  (total_feet : ℕ := (hens * hen_feet) + (goats * goat_feet) + (camels * camel_feet) + (k * keeper_feet)):
  total_feet = total_heads + feet_more_than_heads → k = 15 :=
by
  sorry

end number_of_keepers_l965_96534


namespace population_definition_l965_96500

variable (students : Type) (weights : students → ℝ) (sample : Fin 50 → students)
variable (total_students : Fin 300 → students)
variable (is_selected : students → Prop)

theorem population_definition :
    (∀ s, is_selected s ↔ ∃ i, sample i = s) →
    (population = {w : ℝ | ∃ s, w = weights s}) ↔
    (population = {w : ℝ | ∃ s, w = weights s ∧ ∃ i, total_students i = s}) := by
  sorry

end population_definition_l965_96500


namespace problem_solution_l965_96591

noncomputable def M (a b c : ℝ) : ℝ := (1 - 1/a) * (1 - 1/b) * (1 - 1/c)

theorem problem_solution (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (habc : a + b + c = 1) :
  M a b c ≤ -8 :=
sorry

end problem_solution_l965_96591


namespace project_completion_l965_96575

theorem project_completion (a b : ℕ) (h1 : 3 * (1 / b : ℚ) + (1 / a : ℚ) + (1 / b : ℚ) = 1) : 
  a + b = 9 ∨ a + b = 10 :=
sorry

end project_completion_l965_96575


namespace yearly_return_500_correct_l965_96514

noncomputable def yearly_return_500_investment : ℝ :=
  let total_investment : ℝ := 500 + 1500
  let combined_yearly_return : ℝ := 0.10 * total_investment
  let yearly_return_1500 : ℝ := 0.11 * 1500
  let yearly_return_500 : ℝ := combined_yearly_return - yearly_return_1500
  (yearly_return_500 / 500) * 100

theorem yearly_return_500_correct : yearly_return_500_investment = 7 :=
by
  sorry

end yearly_return_500_correct_l965_96514


namespace largest_shaded_area_l965_96505

noncomputable def figureA_shaded_area : ℝ := 16 - 4 * Real.pi
noncomputable def figureB_shaded_area : ℝ := 16 - 4 * Real.pi
noncomputable def figureC_shaded_area : ℝ := 16 - 4 * Real.sqrt 3

theorem largest_shaded_area : 
  figureC_shaded_area > figureA_shaded_area ∧ figureC_shaded_area > figureB_shaded_area :=
by
  sorry

end largest_shaded_area_l965_96505


namespace math_problem_l965_96545

theorem math_problem (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a / (1 + a) + b / (1 + b) = 1) :
  a / (1 + b^2) - b / (1 + a^2) = a - b :=
sorry

end math_problem_l965_96545


namespace nancy_total_money_l965_96563

theorem nancy_total_money (n : ℕ) (d : ℕ) (h1 : n = 9) (h2 : d = 5) : n * d = 45 := 
by
  sorry

end nancy_total_money_l965_96563


namespace triangle_altitude_sum_l965_96585

-- Problem Conditions
def line_eq (x y : ℝ) : Prop := 10 * x + 8 * y = 80

-- Altitudes Length Sum
theorem triangle_altitude_sum :
  ∀ x y : ℝ, line_eq x y → 
  ∀ (a b c: ℝ), a = 8 → b = 10 → c = 40 / Real.sqrt 41 →
  a + b + c = (18 * Real.sqrt 41 + 40) / Real.sqrt 41 :=
by
  sorry

end triangle_altitude_sum_l965_96585


namespace chocolate_pieces_l965_96555

theorem chocolate_pieces (total_pieces : ℕ) (michael_portion : ℕ) (paige_portion : ℕ) (mandy_portion : ℕ) 
  (h_total : total_pieces = 60) 
  (h_michael : michael_portion = total_pieces / 2) 
  (h_paige : paige_portion = (total_pieces - michael_portion) / 2) 
  (h_mandy : mandy_portion = total_pieces - (michael_portion + paige_portion)) : 
  mandy_portion = 15 :=
by
  sorry

end chocolate_pieces_l965_96555


namespace product_not_divisible_by_201_l965_96560

theorem product_not_divisible_by_201 (a b : ℕ) (h₁ : a + b = 201) : ¬ (201 ∣ a * b) := sorry

end product_not_divisible_by_201_l965_96560


namespace tangent_line_sum_l965_96566

theorem tangent_line_sum (a b : ℝ) :
  (∃ x₀ : ℝ, (e^(x₀ - 1) = 1) ∧ (x₀ + a = e^(x₀-1) * (1 - x₀) - b + 1)) → a + b = 1 :=
by
  sorry

end tangent_line_sum_l965_96566


namespace value_of_business_l965_96593

theorem value_of_business (V : ℝ) (h₁ : (3/5) * (1/3) * V = 2000) : V = 10000 :=
by
  sorry

end value_of_business_l965_96593


namespace vacation_cost_l965_96599

theorem vacation_cost (C P : ℕ) 
    (h1 : C = 5 * P)
    (h2 : C = 7 * (P - 40))
    (h3 : C = 8 * (P - 60)) : C = 700 := 
by 
    sorry

end vacation_cost_l965_96599


namespace evaluate_expression_l965_96532

theorem evaluate_expression : 7^3 - 3 * 7^2 + 3 * 7 - 1 = 216 := by
  sorry

end evaluate_expression_l965_96532


namespace original_price_sarees_l965_96535

theorem original_price_sarees (P : ℝ) (h : 0.80 * P * 0.85 = 231.2) : P = 340 := 
by sorry

end original_price_sarees_l965_96535


namespace quadratic_roots_real_and_equal_l965_96595

open Real

theorem quadratic_roots_real_and_equal :
  ∀ (x : ℝ), x^2 - 4 * x * sqrt 2 + 8 = 0 → ∃ r : ℝ, x = r :=
by
  intro x
  sorry

end quadratic_roots_real_and_equal_l965_96595


namespace parallelepiped_length_l965_96587

theorem parallelepiped_length :
  ∃ n : ℕ, (n ≥ 7) ∧ (n * (n - 2) * (n - 4) = 3 * ((n - 2) * (n - 4) * (n - 6))) ∧ n = 18 :=
by
  sorry

end parallelepiped_length_l965_96587


namespace last_five_digits_l965_96578

theorem last_five_digits : (99 * 10101 * 111 * 1001) % 100000 = 88889 :=
by
  sorry

end last_five_digits_l965_96578


namespace cos_alpha_sub_beta_sin_alpha_l965_96540

open Real

variables (α β : ℝ)

-- Conditions:
-- 0 < α < π / 2
def alpha_in_first_quadrant := 0 < α ∧ α < π / 2

-- -π / 2 < β < 0
def beta_in_fourth_quadrant := -π / 2 < β ∧ β < 0

-- sin β = -5/13
def sin_beta := sin β = -5 / 13

-- tan(α - β) = 4/3
def tan_alpha_sub_beta := tan (α - β) = 4 / 3

-- Theorem statements (follows directly from the conditions and the equivalence):
theorem cos_alpha_sub_beta : alpha_in_first_quadrant α → beta_in_fourth_quadrant β → sin_beta β → tan_alpha_sub_beta α β → cos (α - β) = 3 / 5 := sorry

theorem sin_alpha : alpha_in_first_quadrant α → beta_in_fourth_quadrant β → sin_beta β → tan_alpha_sub_beta α β → sin α = 33 / 65 := sorry

end cos_alpha_sub_beta_sin_alpha_l965_96540


namespace problem_one_problem_two_l965_96574

theorem problem_one (α : ℝ) (h : Real.tan α = 2) : (3 * Real.sin α - 2 * Real.cos α) / (Real.sin α - Real.cos α) = 4 :=
by
  sorry

theorem problem_two (α : ℝ) (h : Real.tan α = 2) (h_quadrant : α > π ∧ α < 3 * π / 2) : Real.cos α = - (Real.sqrt 5 / 5) :=
by
  sorry

end problem_one_problem_two_l965_96574


namespace bag_weight_l965_96548

theorem bag_weight (W : ℕ) 
  (h1 : 2 * W + 82 * (2 * W) = 664) : 
  W = 4 := by
  sorry

end bag_weight_l965_96548


namespace sphere_in_cube_volume_unreachable_l965_96580

noncomputable def volume_unreachable_space (cube_side : ℝ) (sphere_radius : ℝ) : ℝ :=
  let corner_volume := 64 - (32/3) * Real.pi
  let edge_volume := 288 - 72 * Real.pi
  corner_volume + edge_volume

theorem sphere_in_cube_volume_unreachable : 
  (volume_unreachable_space 6 1 = 352 - (248 * Real.pi / 3)) :=
by
  sorry

end sphere_in_cube_volume_unreachable_l965_96580


namespace angles_equal_or_cofunctions_equal_l965_96557

def cofunction (θ : ℝ) : ℝ := sorry -- Define the co-function (e.g., sine and cosine)

theorem angles_equal_or_cofunctions_equal (θ₁ θ₂ : ℝ) :
  θ₁ = θ₂ ∨ cofunction θ₁ = cofunction θ₂ → θ₁ = θ₂ :=
sorry

end angles_equal_or_cofunctions_equal_l965_96557


namespace sum_inequality_l965_96528

theorem sum_inequality 
  {a b c : ℝ}
  (h : a + b + c = 3) : 
  (1 / (a^2 - a + 2) + 1 / (b^2 - b + 2) + 1 / (c^2 - c + 2)) ≤ 3 / 2 := 
sorry

end sum_inequality_l965_96528


namespace quadratic_non_negative_iff_a_in_range_l965_96588

theorem quadratic_non_negative_iff_a_in_range :
  (∀ x : ℝ, x^2 + (a - 2) * x + 1/4 ≥ 0) ↔ 1 ≤ a ∧ a ≤ 3 :=
sorry

end quadratic_non_negative_iff_a_in_range_l965_96588


namespace company_initial_bureaus_l965_96530

theorem company_initial_bureaus (B : ℕ) (offices : ℕ) (extra_bureaus : ℕ) 
  (h1 : offices = 14) 
  (h2 : extra_bureaus = 10) 
  (h3 : (B + extra_bureaus) % offices = 0) : 
  B = 8 := 
by
  sorry

end company_initial_bureaus_l965_96530


namespace face_value_of_share_l965_96558

-- Let FV be the face value of each share.
-- Given conditions:
-- Dividend rate is 9%
-- Market value of each share is Rs. 42
-- Desired interest rate is 12%

theorem face_value_of_share (market_value : ℝ) (dividend_rate : ℝ) (interest_rate : ℝ) (FV : ℝ) :
  market_value = 42 ∧ dividend_rate = 0.09 ∧ interest_rate = 0.12 →
  0.09 * FV = 0.12 * market_value →
  FV = 56 :=
by
  sorry

end face_value_of_share_l965_96558


namespace no_integer_n_gte_1_where_9_divides_7n_plus_n3_l965_96568

theorem no_integer_n_gte_1_where_9_divides_7n_plus_n3 :
  ∀ n : ℕ, 1 ≤ n → ¬ (7^n + n^3) % 9 = 0 := 
by
  intros n hn
  sorry

end no_integer_n_gte_1_where_9_divides_7n_plus_n3_l965_96568


namespace hotel_profit_calculation_l965_96513

theorem hotel_profit_calculation
  (operations_expenses : ℝ)
  (meetings_fraction : ℝ) (events_fraction : ℝ) (rooms_fraction : ℝ)
  (meetings_tax_rate : ℝ) (meetings_commission_rate : ℝ)
  (events_tax_rate : ℝ) (events_commission_rate : ℝ)
  (rooms_tax_rate : ℝ) (rooms_commission_rate : ℝ)
  (total_profit : ℝ) :
  operations_expenses = 5000 →
  meetings_fraction = 5/8 →
  events_fraction = 3/10 →
  rooms_fraction = 11/20 →
  meetings_tax_rate = 0.10 →
  meetings_commission_rate = 0.05 →
  events_tax_rate = 0.08 →
  events_commission_rate = 0.06 →
  rooms_tax_rate = 0.12 →
  rooms_commission_rate = 0.03 →
  total_profit = (operations_expenses * (meetings_fraction + events_fraction + rooms_fraction)
                - (operations_expenses
                  + operations_expenses * (meetings_fraction * (meetings_tax_rate + meetings_commission_rate)
                  + events_fraction * (events_tax_rate + events_commission_rate)
                  + rooms_fraction * (rooms_tax_rate + rooms_commission_rate)))) ->
  total_profit = 1283.75 :=
by sorry

end hotel_profit_calculation_l965_96513


namespace desired_salt_percentage_is_ten_percent_l965_96553

-- Define the initial conditions
def initial_pure_water_volume : ℝ := 100
def saline_solution_percentage : ℝ := 0.25
def added_saline_volume : ℝ := 66.67
def total_volume : ℝ := initial_pure_water_volume + added_saline_volume
def added_salt : ℝ := saline_solution_percentage * added_saline_volume
def desired_salt_percentage (P : ℝ) : Prop := added_salt = P * total_volume

-- State the theorem and its result
theorem desired_salt_percentage_is_ten_percent (P : ℝ) (h : desired_salt_percentage P) : P = 0.1 :=
sorry

end desired_salt_percentage_is_ten_percent_l965_96553


namespace donna_fully_loaded_truck_weight_l965_96569

-- Define conditions
def empty_truck_weight : ℕ := 12000
def soda_crate_weight : ℕ := 50
def soda_crate_count : ℕ := 20
def dryer_weight : ℕ := 3000
def dryer_count : ℕ := 3

-- Calculate derived weights
def soda_total_weight : ℕ := soda_crate_weight * soda_crate_count
def fresh_produce_weight : ℕ := 2 * soda_total_weight
def dryer_total_weight : ℕ := dryer_weight * dryer_count

-- Define target weight of fully loaded truck
def fully_loaded_truck_weight : ℕ :=
  empty_truck_weight + soda_total_weight + fresh_produce_weight + dryer_total_weight

-- State and prove the theorem
theorem donna_fully_loaded_truck_weight :
  fully_loaded_truck_weight = 24000 :=
by
  -- Provide necessary calculations and proof steps if needed
  sorry

end donna_fully_loaded_truck_weight_l965_96569


namespace arc_length_correct_l965_96510

noncomputable def chord_length := 2
noncomputable def central_angle := 2
noncomputable def half_chord_length := 1
noncomputable def radius := 1 / Real.sin 1
noncomputable def arc_length := 2 * radius

theorem arc_length_correct :
  arc_length = 2 / Real.sin 1 := by
sorry

end arc_length_correct_l965_96510


namespace number_of_marbles_l965_96523

theorem number_of_marbles (T : ℕ) (h1 : 12 ≤ T) : 
  (T - 12) * (T - 12) * 16 = 9 * T * T → T = 48 :=
by
  -- Proof omitted
  sorry

end number_of_marbles_l965_96523


namespace determine_constant_l965_96524

/-- If the function f(x) = a * sin x + 3 * cos x has a maximum value of 5,
then the constant a must be ± 4. -/
theorem determine_constant (a : ℝ) (h : ∀ x : ℝ, a * Real.sin x + 3 * Real.cos x ≤ 5) :
  a = 4 ∨ a = -4 :=
sorry

end determine_constant_l965_96524


namespace cylindrical_tank_volume_l965_96515

theorem cylindrical_tank_volume (d h : ℝ) (d_eq_20 : d = 20) (h_eq_10 : h = 10) : 
  π * ((d / 2) ^ 2) * h = 1000 * π :=
by
  sorry

end cylindrical_tank_volume_l965_96515


namespace evaluate_expression_l965_96576

theorem evaluate_expression : (1 / (5^2)^4) * 5^15 = 5^7 :=
by
  sorry

end evaluate_expression_l965_96576


namespace five_point_questions_l965_96537

-- Defining the conditions as Lean statements
def question_count (x y : ℕ) : Prop := x + y = 30
def total_points (x y : ℕ) : Prop := 5 * x + 10 * y = 200

-- The theorem statement that states x equals the number of 5-point questions
theorem five_point_questions (x y : ℕ) (h1 : question_count x y) (h2 : total_points x y) : x = 20 :=
sorry -- Proof is omitted

end five_point_questions_l965_96537


namespace avg_adults_proof_l965_96522

variable (n_total : ℕ) (n_girls : ℕ) (n_boys : ℕ) (n_adults : ℕ)
variable (avg_total : ℕ) (avg_girls : ℕ) (avg_boys : ℕ)

def avg_age_adults (n_total n_girls n_boys n_adults avg_total avg_girls avg_boys : ℕ) : ℕ :=
  let sum_total := n_total * avg_total
  let sum_girls := n_girls * avg_girls
  let sum_boys := n_boys * avg_boys
  let sum_adults := sum_total - sum_girls - sum_boys
  sum_adults / n_adults

theorem avg_adults_proof :
  avg_age_adults 50 25 20 5 21 18 20 = 40 := 
by
  -- Proof will go here
  sorry

end avg_adults_proof_l965_96522


namespace initial_clothing_count_l965_96508

theorem initial_clothing_count 
  (donated_first : ℕ) 
  (donated_second : ℕ) 
  (thrown_away : ℕ) 
  (remaining : ℕ) 
  (h1 : donated_first = 5) 
  (h2 : donated_second = 3 * donated_first) 
  (h3 : thrown_away = 15) 
  (h4 : remaining = 65) :
  donated_first + donated_second + thrown_away + remaining = 100 :=
by
  sorry

end initial_clothing_count_l965_96508


namespace students_play_neither_l965_96594

-- Define the given conditions
def total_students : ℕ := 36
def football_players : ℕ := 26
def long_tennis_players : ℕ := 20
def both_sports_players : ℕ := 17

-- The goal is to prove the number of students playing neither sport is 7
theorem students_play_neither :
  total_students - (football_players + long_tennis_players - both_sports_players) = 7 :=
by
  sorry

end students_play_neither_l965_96594


namespace score_on_fourth_board_l965_96583

theorem score_on_fourth_board 
  (score1 score2 score3 score4 : ℕ)
  (h1 : score1 = 30)
  (h2 : score2 = 38)
  (h3 : score3 = 41)
  (total_score : score1 + score2 = 2 * score4) :
  score4 = 34 := by
  sorry

end score_on_fourth_board_l965_96583


namespace complement_of_intersection_l965_96529

def A : Set ℤ := {-1, 0}
def B : Set ℤ := {0, 1}

theorem complement_of_intersection (AuB AcB : Set ℤ) :
  (A ∪ B) = AuB ∧ (A ∩ B) = AcB → 
  A ∪ B = ∅ ∨ A ∪ B = AuB → 
  (AuB \ AcB) = {-1, 1} :=
by
  -- Proof construction method placeholder.
  sorry

end complement_of_intersection_l965_96529


namespace probability_divisor_of_8_is_half_l965_96579

def divisors (n : ℕ) : List ℕ := 
  List.filter (λ x => n % x = 0) (List.range (n + 1))

def num_divisors : ℕ := (divisors 8).length
def total_outcomes : ℕ := 8

theorem probability_divisor_of_8_is_half :
  (num_divisors / total_outcomes : ℚ) = 1 / 2 :=
by
  sorry

end probability_divisor_of_8_is_half_l965_96579


namespace adjacent_zero_point_range_l965_96556

def f (x : ℝ) : ℝ := x - 1
def g (x : ℝ) (a : ℝ) : ℝ := x^2 - a*x - a + 3

theorem adjacent_zero_point_range (a : ℝ) :
  (∀ β, (∃ x, g x a = 0) → (|1 - β| ≤ 1 → (∃ x, f x = 0 → |x - β| ≤ 1))) →
  (2 ≤ a ∧ a ≤ 7 / 3) :=
sorry

end adjacent_zero_point_range_l965_96556


namespace quadratic_solution_l965_96502

theorem quadratic_solution 
  (x : ℝ)
  (h : x^2 - 2 * x - 1 = 0) : 
  x = 1 + Real.sqrt 2 ∨ x = 1 - Real.sqrt 2 :=
sorry

end quadratic_solution_l965_96502


namespace largest_a_l965_96539

theorem largest_a (a b : ℕ) (x : ℕ) (h_a_range : 2 < a ∧ a < x) (h_b_range : 4 < b ∧ b < 13) (h_fraction_range : 7 * a = 57) : a = 8 :=
sorry

end largest_a_l965_96539


namespace socks_pair_count_l965_96538

theorem socks_pair_count :
  let white := 5
  let brown := 5
  let blue := 3
  let green := 2
  (white * brown) + (white * blue) + (white * green) + (brown * blue) + (brown * green) + (blue * green) = 81 :=
by
  intros
  sorry

end socks_pair_count_l965_96538


namespace min_value_one_over_a_plus_nine_over_b_l965_96581

theorem min_value_one_over_a_plus_nine_over_b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) : 
  16 ≤ (1 / a) + (9 / b) :=
sorry

end min_value_one_over_a_plus_nine_over_b_l965_96581


namespace total_pics_uploaded_l965_96565

-- Definitions of conditions
def pic_in_first_album : Nat := 14
def albums_with_7_pics : Nat := 3
def pics_per_album : Nat := 7

-- Theorem statement
theorem total_pics_uploaded :
  pic_in_first_album + albums_with_7_pics * pics_per_album = 35 := by
  sorry

end total_pics_uploaded_l965_96565


namespace tan_neg_240_eq_neg_sqrt_3_l965_96512

theorem tan_neg_240_eq_neg_sqrt_3 : Real.tan (-4 * Real.pi / 3) = -Real.sqrt 3 :=
by
  sorry

end tan_neg_240_eq_neg_sqrt_3_l965_96512


namespace nesting_doll_height_l965_96584

variable (H₀ : ℝ) (n : ℕ)

theorem nesting_doll_height (H₀ : ℝ) (Hₙ : ℝ) (H₁ : H₀ = 243) (H₂ : ∀ n : ℕ, Hₙ = H₀ * (2 / 3) ^ n) (H₃ : Hₙ = 32) : n = 4 :=
by
  sorry

end nesting_doll_height_l965_96584


namespace person_A_misses_at_least_once_in_4_shots_person_B_stops_after_5_shots_due_to_2_consecutive_misses_l965_96521

-- Define the probability of hitting the target for Person A and Person B
def p_hit_A : ℚ := 2 / 3
def p_hit_B : ℚ := 3 / 4

-- Define the complementary probabilities (missing the target)
def p_miss_A := 1 - p_hit_A
def p_miss_B := 1 - p_hit_B

-- Prove the probability that Person A, shooting 4 times, misses the target at least once
theorem person_A_misses_at_least_once_in_4_shots :
  (1 - (p_hit_A ^ 4)) = 65 / 81 :=
by 
  sorry

-- Prove the probability that Person B stops shooting exactly after 5 shots
-- due to missing the target consecutively 2 times
theorem person_B_stops_after_5_shots_due_to_2_consecutive_misses :
  (p_hit_B * p_hit_B * p_miss_B * (p_miss_B * p_miss_B)) = 45 / 1024 :=
by
  sorry

end person_A_misses_at_least_once_in_4_shots_person_B_stops_after_5_shots_due_to_2_consecutive_misses_l965_96521


namespace general_term_of_sequence_l965_96552

noncomputable def seq (n : ℕ) : ℕ :=
  match n with
  | 1 => 6
  | 2 => 9
  | 3 => 14
  | 4 => 21
  | 5 => 30
  | _ => sorry

theorem general_term_of_sequence :
  ∀ n : ℕ, seq n = 5 + n^2 :=
by
  sorry

end general_term_of_sequence_l965_96552


namespace fraction_books_left_l965_96572

theorem fraction_books_left (initial_books sold_books remaining_books : ℕ)
  (h1 : initial_books = 9900) (h2 : sold_books = 3300) (h3 : remaining_books = initial_books - sold_books) :
  (remaining_books : ℚ) / initial_books = 2 / 3 :=
by
  sorry

end fraction_books_left_l965_96572


namespace weight_of_each_bag_is_7_l965_96598

-- Defining the conditions
def morning_bags : ℕ := 29
def afternoon_bags : ℕ := 17
def total_weight : ℕ := 322

-- Defining the question in terms of proving a specific weight per bag
def bags_sold := morning_bags + afternoon_bags
def weight_per_bag (w : ℕ) := total_weight = bags_sold * w

-- Proving the question == answer under the given conditions
theorem weight_of_each_bag_is_7 :
  ∃ w : ℕ, weight_per_bag w ∧ w = 7 :=
by
  sorry

end weight_of_each_bag_is_7_l965_96598


namespace ike_mike_total_items_l965_96551

theorem ike_mike_total_items :
  ∃ (s d : ℕ), s + d = 7 ∧ 5 * s + 3/2 * d = 35 :=
by sorry

end ike_mike_total_items_l965_96551


namespace tara_road_trip_cost_l965_96577

theorem tara_road_trip_cost :
  let tank_capacity := 12
  let price1 := 3
  let price2 := 3.50
  let price3 := 4
  let price4 := 4.50
  (price1 * tank_capacity) + (price2 * tank_capacity) + (price3 * tank_capacity) + (price4 * tank_capacity) = 180 :=
by
  sorry

end tara_road_trip_cost_l965_96577


namespace smallest_possible_n_l965_96503

theorem smallest_possible_n (n : ℕ) :
  ∃ n, 17 * n - 3 ≡ 0 [MOD 11] ∧ n = 6 :=
by
  sorry

end smallest_possible_n_l965_96503


namespace no_natural_number_such_that_n_sq_plus_6n_plus_2019_mod_100_eq_0_l965_96546

theorem no_natural_number_such_that_n_sq_plus_6n_plus_2019_mod_100_eq_0 :
  ¬ ∃ n : ℕ, (n^2 + 6 * n + 2019) % 100 = 0 :=
sorry

end no_natural_number_such_that_n_sq_plus_6n_plus_2019_mod_100_eq_0_l965_96546


namespace xy_product_l965_96544

variable {x y : ℝ}

theorem xy_product (h1 : x ≠ y) (h2 : x ≠ 0) (h3 : y ≠ 0) (h4 : x + 3/x = y + 3/y) : x * y = 3 :=
sorry

end xy_product_l965_96544


namespace inequality_proof_l965_96525

theorem inequality_proof (a b : ℝ) (ha : a < 0) (hb : b > 0) : (1 / a < 1 / b) :=
sorry

end inequality_proof_l965_96525


namespace find_smaller_number_l965_96507

theorem find_smaller_number (x y : ℝ) (h1 : x + y = 18) (h2 : x * y = 80) : y = 8 :=
by sorry

end find_smaller_number_l965_96507


namespace systematic_sampling_l965_96501

theorem systematic_sampling :
  let N := 60
  let n := 5
  let k := N / n
  let initial_sample := 5
  let samples := [initial_sample, initial_sample + k, initial_sample + 2 * k, initial_sample + 3 * k, initial_sample + 4 * k] 
  samples = [5, 17, 29, 41, 53] := sorry

end systematic_sampling_l965_96501


namespace find_a3_l965_96504

-- Define the polynomial equality
def polynomial_equality (x : ℝ) (a0 a1 a2 a3 a4 a5 a6 a7 : ℝ) :=
  (1 + x) * (2 - x)^6 = a0 + a1 * (x - 1) + a2 * (x - 1)^2 + a3 * (x - 1)^3 + a4 * (x - 1)^4 + a5 * (x - 1)^5 + a6 * (x - 1)^6 + a7 * (x - 1)^7

-- State the main theorem
theorem find_a3 (a0 a1 a2 a4 a5 a6 a7 : ℝ) :
  (∃ (x : ℝ), polynomial_equality x a0 a1 a2 (-25) a4 a5 a6 a7) :=
sorry

end find_a3_l965_96504


namespace tory_earns_more_than_bert_l965_96562

-- Define the initial prices of the toys
def initial_price_phones : ℝ := 18
def initial_price_guns : ℝ := 20

-- Define the quantities sold by Bert and Tory
def quantity_phones : ℕ := 10
def quantity_guns : ℕ := 15

-- Define the discounts
def discount_phones : ℝ := 0.15
def discounted_phones_quantity : ℕ := 3

def discount_guns : ℝ := 0.10
def discounted_guns_quantity : ℕ := 7

-- Define the tax
def tax_rate : ℝ := 0.05

noncomputable def bert_initial_earnings : ℝ := initial_price_phones * quantity_phones

noncomputable def tory_initial_earnings : ℝ := initial_price_guns * quantity_guns

noncomputable def bert_discount : ℝ := discount_phones * initial_price_phones * discounted_phones_quantity

noncomputable def tory_discount : ℝ := discount_guns * initial_price_guns * discounted_guns_quantity

noncomputable def bert_earnings_after_discount : ℝ := bert_initial_earnings - bert_discount

noncomputable def tory_earnings_after_discount : ℝ := tory_initial_earnings - tory_discount

noncomputable def bert_tax : ℝ := tax_rate * bert_earnings_after_discount

noncomputable def tory_tax : ℝ := tax_rate * tory_earnings_after_discount

noncomputable def bert_final_earnings : ℝ := bert_earnings_after_discount + bert_tax

noncomputable def tory_final_earnings : ℝ := tory_earnings_after_discount + tory_tax

noncomputable def earning_difference : ℝ := tory_final_earnings - bert_final_earnings

theorem tory_earns_more_than_bert : earning_difference = 119.805 := by
  sorry

end tory_earns_more_than_bert_l965_96562


namespace harry_has_19_apples_l965_96542

def apples_problem := 
  let A_M := 68  -- Martha's apples
  let A_T := A_M - 30  -- Tim's apples (68 - 30)
  let A_H := A_T / 2  -- Harry's apples (38 / 2)
  A_H = 19

theorem harry_has_19_apples : apples_problem :=
by
  -- prove A_H = 19 given the conditions
  sorry

end harry_has_19_apples_l965_96542


namespace projectile_height_reaches_35_l965_96516

theorem projectile_height_reaches_35 
  (t : ℝ)
  (h_eq : -4.9 * t^2 + 30 * t = 35) :
  t = 2 ∨ t = 50 / 7 ∧ t = min (2 : ℝ) (50 / 7) :=
by
  sorry

end projectile_height_reaches_35_l965_96516


namespace max_and_min_of_z_in_G_l965_96547

def z (x y : ℝ) : ℝ := x^2 + y^2 - 2*x*y - x - 2*y

def G (x y : ℝ) : Prop := x ≥ 0 ∧ y ≥ 0 ∧ x + y ≤ 4

theorem max_and_min_of_z_in_G :
  (∃ (x y : ℝ), G x y ∧ z x y = 12) ∧ (∃ (x y : ℝ), G x y ∧ z x y = -1/4) :=
sorry

end max_and_min_of_z_in_G_l965_96547


namespace percentage_reduction_in_price_l965_96592

-- Definitions based on conditions
def original_price (P : ℝ) (X : ℝ) := P * X
def reduced_price (R : ℝ) (X : ℝ) := R * (X + 5)

-- Theorem statement based on the problem to prove
theorem percentage_reduction_in_price
  (R : ℝ) (H1 : R = 55)
  (H2 : original_price P X = 1100)
  (H3 : reduced_price R X = 1100) :
  ((P - R) / P) * 100 = 25 :=
by
  sorry

end percentage_reduction_in_price_l965_96592


namespace range_of_a_l965_96573

noncomputable def f (x : ℝ) : ℝ := (2^x - 2^(-x)) * x^3

theorem range_of_a (a : ℝ) :
  f (Real.logb 2 a) + f (Real.logb 0.5 a) ≤ 2 * f 1 → (1/2 : ℝ) ≤ a ∧ a ≤ 2 :=
by
  sorry

end range_of_a_l965_96573


namespace min_value_abs_plus_one_l965_96559

theorem min_value_abs_plus_one : ∃ x : ℝ, |x| + 1 = 1 :=
by
  use 0
  sorry

end min_value_abs_plus_one_l965_96559


namespace fixed_point_sum_l965_96550

theorem fixed_point_sum (a m n : ℝ) (h1 : a > 0) (h2 : a ≠ 1)
  (h3 : (m, n) = (1, a * (1-1) + 2)) : m + n = 4 :=
by {
  sorry
}

end fixed_point_sum_l965_96550


namespace overall_effect_l965_96549
noncomputable def effect (x : ℚ) : ℚ :=
  ((x * (5 / 6)) * (1 / 10)) + (2 / 3)

theorem overall_effect (x : ℚ) : effect x = (x * (5 / 6) * (1 / 10)) + (2 / 3) :=
  by
  sorry

-- Prove for initial number 1
example : effect 1 = 3 / 4 :=
  by
  sorry

end overall_effect_l965_96549


namespace technician_round_trip_completion_l965_96531

theorem technician_round_trip_completion (D : ℝ) (h0 : D > 0) :
  let round_trip := 2 * D
  let to_center := D
  let from_center := 0.30 * D
  let traveled := to_center + from_center
  traveled / round_trip * 100 = 65 := 
by
  sorry

end technician_round_trip_completion_l965_96531


namespace simplify_expression_l965_96586

theorem simplify_expression : 2023^2 - 2022 * 2024 = 1 := by
  sorry

end simplify_expression_l965_96586


namespace largest_square_area_l965_96518

theorem largest_square_area (total_string_length : ℕ) (h : total_string_length = 32) : ∃ (area : ℕ), area = 64 := 
  by
    sorry

end largest_square_area_l965_96518


namespace find_a_l965_96517

open Real

def point_in_circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 6 * x + 4 * y + 4 = 0

def line_equation (x y : ℝ) : Prop :=
  x + 2 * y - 3 = 0

theorem find_a (a : ℝ) :
  point_in_circle 1 a →
  line_equation 1 a →
  a = -2 :=
by
  intro h1 h2
  sorry

end find_a_l965_96517


namespace emily_sixth_quiz_score_l965_96511

theorem emily_sixth_quiz_score (q1 q2 q3 q4 q5 target_mean : ℕ) (required_sum : ℕ) (current_sum : ℕ) (s6 : ℕ)
  (h1 : q1 = 94) (h2 : q2 = 97) (h3 : q3 = 88) (h4 : q4 = 91) (h5 : q5 = 102) (h_target_mean : target_mean = 95)
  (h_required_sum : required_sum = 6 * target_mean) (h_current_sum : current_sum = q1 + q2 + q3 + q4 + q5)
  (h6 : s6 = required_sum - current_sum) :
  s6 = 98 :=
by
  sorry

end emily_sixth_quiz_score_l965_96511


namespace correct_word_is_tradition_l965_96561

-- Definitions of the words according to the problem conditions
def tradition : String := "custom, traditional practice"
def balance : String := "equilibrium"
def concern : String := "worry, care about"
def relationship : String := "relation"

-- The sentence to be filled
def sentence (word : String) : String :=
"There’s a " ++ word ++ " in our office that when it’s somebody’s birthday, they bring in a cake for us all to share."

-- The proof problem statement
theorem correct_word_is_tradition :
  ∀ word, (word ≠ tradition) → (sentence word ≠ "There’s a tradition in our office that when it’s somebody’s birthday, they bring in a cake for us all to share.") :=
by sorry

end correct_word_is_tradition_l965_96561


namespace sequence_value_a_l965_96506

theorem sequence_value_a (a : ℚ) (a_n : ℕ → ℚ)
  (h1 : a_n 1 = a) (h2 : a_n 2 = a)
  (h3 : ∀ n ≥ 3, a_n n = a_n (n - 1) + a_n (n - 2))
  (h4 : a_n 8 = 34) :
  a = 34 / 21 :=
by sorry

end sequence_value_a_l965_96506


namespace woods_width_l965_96533

theorem woods_width (Area Length Width : ℝ) (hArea : Area = 24) (hLength : Length = 3) : 
  Width = 8 := 
by
  sorry

end woods_width_l965_96533


namespace simplify_fraction_l965_96564

theorem simplify_fraction : (45 / (7 - 3 / 4)) = (36 / 5) :=
by
  sorry

end simplify_fraction_l965_96564


namespace shares_proportion_l965_96554

theorem shares_proportion (C D : ℕ) (h1 : D = 1500) (h2 : C = D + 500) : C / Nat.gcd C D = 4 ∧ D / Nat.gcd C D = 3 := by
  sorry

end shares_proportion_l965_96554


namespace arithmetic_seq_a4_value_l965_96541

theorem arithmetic_seq_a4_value
  (a : ℕ → ℤ)
  (h : 4 * a 3 + a 11 - 3 * a 5 = 10) :
  a 4 = 5 := 
sorry

end arithmetic_seq_a4_value_l965_96541


namespace functional_equation_solution_l965_96543

theorem functional_equation_solution (f g : ℝ → ℝ)
  (H : ∀ x y : ℝ, f (x^2 - g y) = g x ^ 2 - y) :
  (∀ x : ℝ, f x = x) ∧ (∀ x : ℝ, g x = x) :=
by
  sorry

end functional_equation_solution_l965_96543


namespace problem_positive_l965_96509

theorem problem_positive : ∀ x : ℝ, x < 0 → -3 * x⁻¹ > 0 :=
by 
  sorry

end problem_positive_l965_96509


namespace length_QR_l965_96597

-- Let's define the given conditions and the theorem to prove

-- Define the lengths of the sides of the triangle
def PQ : ℝ := 4
def PR : ℝ := 7
def PM : ℝ := 3.5

-- Define the median formula
def median_formula (PQ PR QR PM : ℝ) := PM = 0.5 * Real.sqrt (2 * PQ^2 + 2 * PR^2 - QR^2)

-- The theorem to prove: QR = 9
theorem length_QR 
  (hPQ : PQ = 4) 
  (hPR : PR = 7) 
  (hPM : PM = 3.5) 
  (hMedian : median_formula PQ PR QR PM) : 
  QR = 9 :=
sorry  -- proof will be here

end length_QR_l965_96597


namespace point_on_x_axis_coordinates_l965_96526

-- Define the conditions
def lies_on_x_axis (M : ℝ × ℝ) : Prop := M.snd = 0

-- State the problem
theorem point_on_x_axis_coordinates (a : ℝ) :
  lies_on_x_axis (a + 3, a + 1) → (a = -1) ∧ ((a + 3, 0) = (2, 0)) :=
by
  intro h
  rw [lies_on_x_axis] at h
  sorry

end point_on_x_axis_coordinates_l965_96526


namespace geometric_sequence_constant_l965_96520

theorem geometric_sequence_constant (a : ℕ → ℝ) (q : ℝ)
    (h1 : ∀ n, a (n+1) = q * a n)
    (h2 : ∀ n, a n > 0)
    (h3 : (a 1 + a 3) * (a 5 + a 7) = 4 * (a 4) ^ 2) :
    ∀ n, a n = a 0 :=
by
  sorry

end geometric_sequence_constant_l965_96520


namespace sum_of_factors_eq_l965_96596

theorem sum_of_factors_eq :
  ∃ (d e f : ℤ), (∀ (x : ℤ), x^2 + 21 * x + 110 = (x + d) * (x + e)) ∧
                 (∀ (x : ℤ), x^2 - 19 * x + 88 = (x - e) * (x - f)) ∧
                 (d + e + f = 30) :=
sorry

end sum_of_factors_eq_l965_96596


namespace distance_between_closest_points_l965_96582

noncomputable def distance_closest_points (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) : ℝ :=
  (Real.sqrt ((c2.1 - c1.1)^2 + (c2.2 - c1.2)^2) - r1 - r2)

theorem distance_between_closest_points :
  distance_closest_points (4, 4) (20, 12) 4 12 = 4 * Real.sqrt 20 - 16 :=
by
  sorry

end distance_between_closest_points_l965_96582


namespace vegetarian_gluten_free_fraction_l965_96571

theorem vegetarian_gluten_free_fraction :
  ∀ (total_dishes meatless_dishes gluten_free_meatless_dishes : ℕ),
  meatless_dishes = 4 →
  meatless_dishes = total_dishes / 5 →
  gluten_free_meatless_dishes = meatless_dishes - 3 →
  gluten_free_meatless_dishes / total_dishes = 1 / 20 :=
by sorry

end vegetarian_gluten_free_fraction_l965_96571


namespace convert_base_five_to_ten_l965_96589

theorem convert_base_five_to_ten : ∃ n : ℕ, n = 38 ∧ (1 * 5^2 + 2 * 5^1 + 3 * 5^0 = n) :=
by
  sorry

end convert_base_five_to_ten_l965_96589


namespace terry_current_age_l965_96519

theorem terry_current_age (T : ℕ) (nora_current_age : ℕ) (h1 : nora_current_age = 10)
  (h2 : T + 10 = 4 * nora_current_age) : T = 30 :=
by
  sorry

end terry_current_age_l965_96519
