import Mathlib

namespace castle_lego_ratio_l562_56265

def total_legos : ℕ := 500
def legos_put_back : ℕ := 245
def legos_missing : ℕ := 5
def legos_used : ℕ := total_legos - legos_put_back - legos_missing
def ratio (a b : ℕ) : ℚ := a / b

theorem castle_lego_ratio : ratio legos_used total_legos = 1 / 2 :=
by
  unfold ratio legos_used total_legos legos_put_back legos_missing
  norm_num

end castle_lego_ratio_l562_56265


namespace corn_purchase_l562_56293

theorem corn_purchase : ∃ c b : ℝ, c + b = 30 ∧ 89 * c + 55 * b = 2170 ∧ c = 15.3 := 
by
  sorry

end corn_purchase_l562_56293


namespace clothing_store_profit_l562_56246

theorem clothing_store_profit 
  (cost_price selling_price : ℕ)
  (initial_items_per_day items_increment items_reduction : ℕ)
  (initial_profit_per_day : ℕ) :
  -- Conditions
  cost_price = 50 ∧
  selling_price = 90 ∧
  initial_items_per_day = 20 ∧
  items_increment = 2 ∧
  items_reduction = 1 ∧
  initial_profit_per_day = 1200 →
  -- Question
  exists x, 
  (selling_price - x - cost_price) * (initial_items_per_day + items_increment * x) = initial_profit_per_day ∧
  x = 20 := 
sorry

end clothing_store_profit_l562_56246


namespace cylinder_lateral_surface_area_l562_56268

theorem cylinder_lateral_surface_area
    (r h : ℝ) (hr : r = 3) (hh : h = 10) :
    2 * Real.pi * r * h = 60 * Real.pi :=
by
  sorry

end cylinder_lateral_surface_area_l562_56268


namespace area_of_gray_region_l562_56257

theorem area_of_gray_region (r : ℝ) (h1 : r * 3 - r = 3) : 
  π * (3 * r) ^ 2 - π * r ^ 2 = 18 * π :=
by
  sorry

end area_of_gray_region_l562_56257


namespace largest_even_of_sum_140_l562_56221

theorem largest_even_of_sum_140 :
  ∃ (n : ℕ), 2 * n + 2 * (n + 1) + 2 * (n + 2) + 2 * (n + 3) = 140 ∧ 2 * (n + 3) = 38 :=
by
  sorry

end largest_even_of_sum_140_l562_56221


namespace point_on_curve_l562_56201

-- Define the parametric curve equations
def onCurve (θ : ℝ) (x y : ℝ) : Prop :=
  x = Real.sin (2 * θ) ∧ y = Real.cos θ + Real.sin θ

-- Define the general form of the curve
def curveEquation (x y : ℝ) : Prop :=
  y^2 = 1 + x

-- The proof statement
theorem point_on_curve : 
  curveEquation (-3/4) (1/2) ∧ ∃ θ : ℝ, onCurve θ (-3/4) (1/2) :=
by
  sorry

end point_on_curve_l562_56201


namespace each_person_bids_five_times_l562_56267

noncomputable def auction_bidding : Prop :=
  let initial_price := 15
  let final_price := 65
  let price_increase_per_bid := 5
  let number_of_bidders := 2
  let total_increase := final_price - initial_price
  let total_bids := total_increase / price_increase_per_bid
  total_bids / number_of_bidders = 5

theorem each_person_bids_five_times : auction_bidding :=
by
  -- The proof will be filled in here.
  sorry

end each_person_bids_five_times_l562_56267


namespace complex_sum_abs_eq_1_or_3_l562_56210

open Complex

theorem complex_sum_abs_eq_1_or_3
  (a b c : ℂ)
  (ha : abs a = 1)
  (hb : abs b = 1)
  (hc : abs c = 1)
  (h : a^3 / (b * c) + b^3 / (a * c) + c^3 / (a * b) = 1) :
  ∃ r : ℝ, (r = 1 ∨ r = 3) ∧ abs (a + b + c) = r :=
by {
  -- Proof goes here
  sorry
}

end complex_sum_abs_eq_1_or_3_l562_56210


namespace sequence_sum_l562_56275

theorem sequence_sum (a : ℕ → ℕ) (S : ℕ → ℕ) (n : ℕ) 
  (H_n_def : H_n = (a 1 + (2:ℕ) * a 2 + (2:ℕ) ^ (n - 1) * a n) / n)
  (H_n_val : H_n = 2^n) :
  S n = n * (n + 3) / 2 :=
by
  sorry

end sequence_sum_l562_56275


namespace area_of_inscribed_rectangle_l562_56259

theorem area_of_inscribed_rectangle (r l w : ℝ) (h1 : r = 8) (h2 : l / w = 3) (h3 : w = 2 * r) : l * w = 768 :=
by
  sorry

end area_of_inscribed_rectangle_l562_56259


namespace pentagon_zero_impossible_l562_56277

theorem pentagon_zero_impossible
  (x : Fin 5 → ℝ)
  (h_sum : x 0 + x 1 + x 2 + x 3 + x 4 = 0)
  (operation : ∀ i : Fin 5, ∀ y : Fin 5 → ℝ,
    y i = (x i + x ((i + 1) % 5)) / 2 ∧ y ((i + 1) % 5) = (x i + x ((i + 1) % 5)) / 2) :
  ¬ ∃ (y : ℕ → (Fin 5 → ℝ)), ∃ N : ℕ, y N = 0 := 
sorry

end pentagon_zero_impossible_l562_56277


namespace fred_red_marbles_l562_56252

variable (R G B : ℕ)
variable (total : ℕ := 63)
variable (B_val : ℕ := 6)
variable (G_def : G = (1 / 2) * R)
variable (eq1 : R + G + B = total)
variable (eq2 : B = B_val)

theorem fred_red_marbles : R = 38 := 
by
  sorry

end fred_red_marbles_l562_56252


namespace absolute_value_equation_sum_l562_56290

theorem absolute_value_equation_sum (x1 x2 : ℝ) (h1 : 3 * x1 - 12 = 6) (h2 : 3 * x2 - 12 = -6) : x1 + x2 = 8 := 
sorry

end absolute_value_equation_sum_l562_56290


namespace simplify_and_find_ab_ratio_l562_56212

-- Given conditions
def given_expression (k : ℤ) : ℤ := 10 * k + 15

-- Simplified form
def simplified_form (k : ℤ) : ℤ := 2 * k + 3

-- Proof problem statement
theorem simplify_and_find_ab_ratio
  (k : ℤ) :
  let a := 2
  let b := 3
  (10 * k + 15) / 5 = 2 * k + 3 → 
  (a:ℚ) / (b:ℚ) = 2 / 3 := sorry

end simplify_and_find_ab_ratio_l562_56212


namespace part_time_job_pay_per_month_l562_56285

def tuition_fee : ℝ := 90
def scholarship_percent : ℝ := 0.30
def scholarship_amount := scholarship_percent * tuition_fee
def amount_after_scholarship := tuition_fee - scholarship_amount
def remaining_amount : ℝ := 18
def months_to_pay : ℝ := 3
def amount_paid_so_far := amount_after_scholarship - remaining_amount

theorem part_time_job_pay_per_month : amount_paid_so_far / months_to_pay = 15 := by
  sorry

end part_time_job_pay_per_month_l562_56285


namespace minimum_value_of_function_l562_56294

theorem minimum_value_of_function : ∀ x : ℝ, (x^2 + 9) / Real.sqrt (x^2 + 3) ≥ 2 * Real.sqrt 6 :=
by
  sorry

end minimum_value_of_function_l562_56294


namespace remaining_area_is_correct_l562_56278

-- Define the large rectangle's side lengths
def large_rectangle_length1 (x : ℝ) := x + 7
def large_rectangle_length2 (x : ℝ) := x + 5

-- Define the hole's side lengths
def hole_length1 (x : ℝ) := x + 1
def hole_length2 (x : ℝ) := x + 4

-- Calculate the areas
def large_rectangle_area (x : ℝ) := large_rectangle_length1 x * large_rectangle_length2 x
def hole_area (x : ℝ) := hole_length1 x * hole_length2 x

-- Define the remaining area after subtracting the hole area from the large rectangle area
def remaining_area (x : ℝ) := large_rectangle_area x - hole_area x

-- Problem statement: prove that the remaining area is 7x + 31
theorem remaining_area_is_correct (x : ℝ) : remaining_area x = 7 * x + 31 :=
by 
  -- The proof should be provided here, but for now we use 'sorry' to omit it
  sorry

end remaining_area_is_correct_l562_56278


namespace find_product_x_plus_1_x_minus_1_l562_56296

theorem find_product_x_plus_1_x_minus_1 (x : ℕ) (h : 2^x + 2^x + 2^x + 2^x = 128) : (x + 1) * (x - 1) = 24 := sorry

end find_product_x_plus_1_x_minus_1_l562_56296


namespace initial_apples_l562_56266

theorem initial_apples (Initially_Apples : ℕ) (Added_Apples : ℕ) (Total_Apples : ℕ)
  (h1 : Added_Apples = 8) (h2 : Total_Apples = 17) : Initially_Apples = 9 :=
by
  have h3 : Added_Apples + Initially_Apples = Total_Apples := by
    sorry
  linarith

end initial_apples_l562_56266


namespace sum_of_distances_to_focus_is_ten_l562_56232

theorem sum_of_distances_to_focus_is_ten (P : ℝ × ℝ) (A B F : ℝ × ℝ)
  (hP : P = (2, 1))
  (hA : A.1^2 = 12 * A.2)
  (hB : B.1^2 = 12 * B.2)
  (hMidpoint : P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
  (hFocus : F = (3, 0)) :
  |A.1 - F.1| + |B.1 - F.1| = 10 :=
by
  sorry

end sum_of_distances_to_focus_is_ten_l562_56232


namespace prod_72516_9999_l562_56270

theorem prod_72516_9999 : 72516 * 9999 = 724987484 :=
by
  sorry

end prod_72516_9999_l562_56270


namespace final_weight_is_correct_l562_56223

-- Define the various weights after each week
def initial_weight : ℝ := 180
def first_week_removed : ℝ := 0.28 * initial_weight
def first_week_remaining : ℝ := initial_weight - first_week_removed
def second_week_removed : ℝ := 0.18 * first_week_remaining
def second_week_remaining : ℝ := first_week_remaining - second_week_removed
def third_week_removed : ℝ := 0.20 * second_week_remaining
def final_weight : ℝ := second_week_remaining - third_week_removed

-- State the theorem to prove the final weight equals 85.0176 kg
theorem final_weight_is_correct : final_weight = 85.0176 := 
by 
  sorry

end final_weight_is_correct_l562_56223


namespace division_and_multiplication_l562_56297

theorem division_and_multiplication (x : ℝ) (h : x = 9) : (x / 6 * 12) = 18 := by
  sorry

end division_and_multiplication_l562_56297


namespace sum_first_six_terms_arithmetic_seq_l562_56219

theorem sum_first_six_terms_arithmetic_seq :
  ∃ a_1 d : ℤ, (a_1 + 3 * d = 7) ∧ (a_1 + 4 * d = 12) ∧ (a_1 + 5 * d = 17) ∧ 
  (6 * (2 * a_1 + 5 * d) / 2 = 27) :=
by
  sorry

end sum_first_six_terms_arithmetic_seq_l562_56219


namespace xy_system_sol_l562_56280

theorem xy_system_sol (x y : ℝ) (h1 : x * y = 8) (h2 : x^2 * y + x * y^2 + x + y = 80) :
  x^3 + y^3 = 416000 / 729 :=
by
  sorry

end xy_system_sol_l562_56280


namespace average_students_is_12_l562_56258

-- Definitions based on the problem's conditions
variables (a b c : Nat)

-- Given conditions
axiom condition1 : a + b + c = 30
axiom condition2 : a + c = 19
axiom condition3 : b + c = 9

-- Prove that the number of average students (c) is 12
theorem average_students_is_12 : c = 12 := by 
  sorry

end average_students_is_12_l562_56258


namespace cubes_with_all_three_faces_l562_56228

theorem cubes_with_all_three_faces (total_cubes red_cubes blue_cubes green_cubes: ℕ) 
  (h_total: total_cubes = 100)
  (h_red: red_cubes = 80)
  (h_blue: blue_cubes = 85)
  (h_green: green_cubes = 75) :
  40 ≤ total_cubes - ((total_cubes - red_cubes) + (total_cubes - blue_cubes) + (total_cubes - green_cubes)) ∧ (total_cubes - ((total_cubes - red_cubes) + (total_cubes - blue_cubes) + (total_cubes - green_cubes))) ≤ 75 :=
by {
  sorry
}

end cubes_with_all_three_faces_l562_56228


namespace arithmetic_mean_difference_l562_56204

-- Definitions and conditions
variable (p q r : ℝ)
variable (h1 : (p + q) / 2 = 10)
variable (h2 : (q + r) / 2 = 26)

-- Theorem statement
theorem arithmetic_mean_difference : r - p = 32 := by
  -- Proof goes here
  sorry

end arithmetic_mean_difference_l562_56204


namespace task1_task2_l562_56283

/-- Given conditions -/
def cost_A : Nat := 30
def cost_B : Nat := 40
def sell_A : Nat := 35
def sell_B : Nat := 50
def max_cost : Nat := 1550
def min_profit : Nat := 365
def total_cars : Nat := 40

/-- Task 1: Prove maximum B-type cars produced if 10 A-type cars are produced -/
theorem task1 (A: Nat) (B: Nat) (hA: A = 10) (hC: cost_A * A + cost_B * B ≤ max_cost) : B ≤ 31 :=
by sorry

/-- Task 2: Prove the possible production plans producing 40 cars meeting profit and cost constraints -/
theorem task2 (A: Nat) (B: Nat) (hTotal: A + B = total_cars)
(hCost: cost_A * A + cost_B * B ≤ max_cost) 
(hProfit: (sell_A - cost_A) * A + (sell_B - cost_B) * B ≥ min_profit) : 
  (A = 5 ∧ B = 35) ∨ (A = 6 ∧ B = 34) ∨ (A = 7 ∧ B = 33) 
∧ (375 ≤ (sell_A - cost_A) * 5 + (sell_B - cost_B) * 35 ∧ 375 ≥ (sell_A - cost_A) * 5 + (sell_B - cost_B) * 35) :=
by sorry

end task1_task2_l562_56283


namespace probability_of_color_change_is_1_over_6_l562_56284

noncomputable def watchColorChangeProbability : ℚ :=
  let cycleDuration := 45 + 5 + 40
  let favorableDuration := 5 + 5 + 5
  favorableDuration / cycleDuration

theorem probability_of_color_change_is_1_over_6 :
  watchColorChangeProbability = 1 / 6 :=
by
  sorry

end probability_of_color_change_is_1_over_6_l562_56284


namespace verify_ages_l562_56238

noncomputable def correct_ages (S M D W : ℝ) : Prop :=
  (M = S + 29) ∧
  (M + 2 = 2 * (S + 2)) ∧
  (D = S - 3.5) ∧
  (W = 1.5 * D) ∧
  (S = 27) ∧
  (M = 56) ∧
  (D = 23.5) ∧
  (W = 35.25)

theorem verify_ages : ∃ (S M D W : ℝ), correct_ages S M D W :=
by
  sorry

end verify_ages_l562_56238


namespace apples_remaining_l562_56226

-- Define the initial condition of the number of apples on the tree
def initial_apples : ℕ := 7

-- Define the number of apples picked by Rachel
def picked_apples : ℕ := 4

-- Proof goal: the number of apples remaining on the tree is 3
theorem apples_remaining : (initial_apples - picked_apples = 3) :=
sorry

end apples_remaining_l562_56226


namespace original_decimal_number_l562_56250

theorem original_decimal_number (I : ℤ) (d : ℝ) (h1 : 0 ≤ d) (h2 : d < 1) (h3 : I + 4 * (I + d) = 21.2) : I + d = 4.3 :=
by
  sorry

end original_decimal_number_l562_56250


namespace trains_meet_80_km_from_A_l562_56299

-- Define the speeds of the trains
def speed_train_A : ℝ := 60 
def speed_train_B : ℝ := 90 

-- Define the distance between locations A and B
def distance_AB : ℝ := 200 

-- Define the time when the trains meet
noncomputable def meeting_time : ℝ := distance_AB / (speed_train_A + speed_train_B)

-- Define the distance from location A to where the trains meet
noncomputable def distance_from_A (speed_A : ℝ) (meeting_time : ℝ) : ℝ :=
  speed_A * meeting_time

-- Prove the statement
theorem trains_meet_80_km_from_A :
  distance_from_A speed_train_A meeting_time = 80 :=
by
  -- leaving the proof out, it's just an assumption due to 'sorry'
  sorry

end trains_meet_80_km_from_A_l562_56299


namespace equalize_expenses_l562_56288

def total_expenses := 130 + 160 + 150 + 180
def per_person_share := total_expenses / 4
def tom_owes := per_person_share - 130
def dorothy_owes := per_person_share - 160
def sammy_owes := per_person_share - 150
def alice_owes := per_person_share - 180
def t := tom_owes
def d := dorothy_owes

theorem equalize_expenses : t - dorothy_owes = 30 := by
  sorry

end equalize_expenses_l562_56288


namespace decimal_representation_of_fraction_l562_56289

theorem decimal_representation_of_fraction :
  (3 / 40 : ℝ) = 0.075 :=
sorry

end decimal_representation_of_fraction_l562_56289


namespace ferry_q_more_time_l562_56203

variables (speed_ferry_p speed_ferry_q distance_ferry_p distance_ferry_q time_ferry_p time_ferry_q : ℕ)
  -- Conditions given in the problem
  (h1 : speed_ferry_p = 8)
  (h2 : time_ferry_p = 2)
  (h3 : distance_ferry_p = speed_ferry_p * time_ferry_p)
  (h4 : distance_ferry_q = 3 * distance_ferry_p)
  (h5 : speed_ferry_q = speed_ferry_p + 4)
  (h6 : time_ferry_q = distance_ferry_q / speed_ferry_q)

theorem ferry_q_more_time : time_ferry_q - time_ferry_p = 2 :=
by
  sorry

end ferry_q_more_time_l562_56203


namespace peggy_dolls_after_all_events_l562_56291

def initial_dolls : Nat := 6
def grandmother_gift : Nat := 28
def birthday_gift : Nat := grandmother_gift / 2
def lost_dolls (total : Nat) : Nat := (10 * total + 9) / 100  -- using integer division for rounding 10% up
def easter_gift : Nat := (birthday_gift + 2) / 3  -- using integer division for rounding one-third up
def friend_exchange_gain : Int := -1  -- gaining 1 doll but losing 2
def christmas_gift (easter_dolls : Nat) : Nat := (20 * easter_dolls) / 100 + easter_dolls  -- 20% more dolls
def ruined_dolls : Nat := 3

theorem peggy_dolls_after_all_events : initial_dolls + grandmother_gift + birthday_gift - lost_dolls (initial_dolls + grandmother_gift + birthday_gift) + easter_gift + friend_exchange_gain.toNat + christmas_gift easter_gift - ruined_dolls = 50 :=
by
  sorry

end peggy_dolls_after_all_events_l562_56291


namespace doodads_for_thingamabobs_l562_56224

-- Definitions for the conditions
def doodads_per_widgets : ℕ := 18
def widgets_per_thingamabobs : ℕ := 11
def widgets_count : ℕ := 5
def thingamabobs_count : ℕ := 4
def target_thingamabobs : ℕ := 80

-- Definition for the final proof statement
theorem doodads_for_thingamabobs : 
    doodads_per_widgets * (target_thingamabobs * widgets_per_thingamabobs / thingamabobs_count / widgets_count) = 792 := 
by
  sorry

end doodads_for_thingamabobs_l562_56224


namespace min_distance_l562_56236

noncomputable def curve (x : ℝ) : ℝ := (1/4) * x^2 - (1/2) * Real.log x
noncomputable def line (x : ℝ) : ℝ := (3/4) * x - 1

theorem min_distance :
  ∀ P Q : ℝ × ℝ, 
  P.2 = curve P.1 → 
  Q.2 = line Q.1 → 
  ∃ min_dist : ℝ, 
  min_dist = (2 - 2 * Real.log 2) / 5 := 
sorry

end min_distance_l562_56236


namespace negation_proposition_l562_56272

theorem negation_proposition :
  (¬ (∀ x : ℝ, ∃ n : ℕ, n > 0 ∧ n ≥ x)) ↔ (∃ x : ℝ, ∀ n : ℕ, n > 0 → n < x^2) := 
by
  sorry

end negation_proposition_l562_56272


namespace min_xy_min_x_add_y_l562_56256

open Real

theorem min_xy (x y : ℝ) (h1 : log x + log y = log (x + y + 3)) (hx : x > 0) (hy : y > 0) : xy ≥ 9 := sorry

theorem min_x_add_y (x y : ℝ) (h1 : log x + log y = log (x + y + 3)) (hx : x > 0) (hy : y > 0) : x + y ≥ 6 := sorry

end min_xy_min_x_add_y_l562_56256


namespace Alden_nephews_10_years_ago_l562_56200

noncomputable def nephews_Alden_now : ℕ := sorry
noncomputable def nephews_Alden_10_years_ago (N : ℕ) : ℕ := N / 2
noncomputable def nephews_Vihaan_now (N : ℕ) : ℕ := N + 60
noncomputable def total_nephews (N : ℕ) : ℕ := N + (nephews_Vihaan_now N)

theorem Alden_nephews_10_years_ago (N : ℕ) (h1 : total_nephews N = 260) : 
  nephews_Alden_10_years_ago N = 50 :=
by
  sorry

end Alden_nephews_10_years_ago_l562_56200


namespace add_fractions_l562_56227

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end add_fractions_l562_56227


namespace intersections_count_l562_56253

theorem intersections_count
  (c : ℕ)  -- crosswalks per intersection
  (l : ℕ)  -- lines per crosswalk
  (t : ℕ)  -- total lines
  (h_c : c = 4)
  (h_l : l = 20)
  (h_t : t = 400) :
  t / (c * l) = 5 :=
  by
    sorry

end intersections_count_l562_56253


namespace cricket_runs_product_l562_56281

theorem cricket_runs_product :
  let runs_first_10 := [11, 6, 7, 5, 12, 8, 3, 10, 9, 4]
  let total_runs_first_10 := runs_first_10.sum
  let total_runs := total_runs_first_10 + 2 + 7
  2 < 15 ∧ 7 < 15 ∧ (total_runs_first_10 + 2) % 11 = 0 ∧ (total_runs_first_10 + 2 + 7) % 12 = 0 →
  (2 * 7) = 14 :=
by
  intros h
  sorry

end cricket_runs_product_l562_56281


namespace smallest_prime_angle_l562_56222

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem smallest_prime_angle :
  ∃ (x : ℕ), is_prime x ∧ is_prime (2 * x) ∧ x + 2 * x = 90 ∧ x = 29 :=
by sorry

end smallest_prime_angle_l562_56222


namespace maritza_study_hours_l562_56239

noncomputable def time_to_study_for_citizenship_test (num_mc_questions num_fitb_questions time_mc time_fitb : ℕ) : ℕ :=
  (num_mc_questions * time_mc + num_fitb_questions * time_fitb) / 60

theorem maritza_study_hours :
  time_to_study_for_citizenship_test 30 30 15 25 = 20 :=
by
  sorry

end maritza_study_hours_l562_56239


namespace evaporation_period_length_l562_56244

theorem evaporation_period_length
  (initial_water : ℕ) (daily_evaporation : ℝ) (evaporated_percentage : ℝ) : 
  evaporated_percentage * (initial_water : ℝ) / 100 / daily_evaporation = 22 :=
by
  -- Conditions of the problem
  let initial_water := 12
  let daily_evaporation := 0.03
  let evaporated_percentage := 5.5
  -- Sorry proof placeholder
  sorry

end evaporation_period_length_l562_56244


namespace prob_sum_24_four_dice_l562_56279

-- The probability of each die landing on six
def prob_die_six : ℚ := 1 / 6

-- The probability of all four dice showing six
theorem prob_sum_24_four_dice : 
  prob_die_six ^ 4 = 1 / 1296 :=
by
  -- Equivalent Lean statement asserting the probability problem
  sorry

end prob_sum_24_four_dice_l562_56279


namespace smallest_number_divisible_conditions_l562_56260

theorem smallest_number_divisible_conditions :
  ∃ n : ℕ, n % 8 = 6 ∧ n % 7 = 5 ∧ ∀ m : ℕ, m % 8 = 6 ∧ m % 7 = 5 → n ≤ m →
  n % 9 = 0 := by
  sorry

end smallest_number_divisible_conditions_l562_56260


namespace sand_weight_proof_l562_56240

-- Definitions for the given conditions
def side_length : ℕ := 40
def bag_weight : ℕ := 30
def area_per_bag : ℕ := 80

-- Total area of the sandbox
def total_area := side_length * side_length

-- Number of bags needed
def number_of_bags := total_area / area_per_bag

-- Total weight of sand needed
def total_weight := number_of_bags * bag_weight

-- The proof statement
theorem sand_weight_proof :
  total_weight = 600 :=
by
  sorry

end sand_weight_proof_l562_56240


namespace determine_condition_l562_56235

theorem determine_condition (a b c : ℕ) (ha : 0 < a ∧ a < 12) (hb : 0 < b ∧ b < 12) (hc : 0 < c ∧ c < 12) 
    (h_eq : (12 * a + b) * (12 * a + c) = 144 * a * (a + 1) + b * c) : 
    b + c = 12 :=
by
  sorry

end determine_condition_l562_56235


namespace period_2_students_l562_56245

theorem period_2_students (x : ℕ) (h1 : 2 * x - 5 = 11) : x = 8 :=
by {
  sorry
}

end period_2_students_l562_56245


namespace probability_sum_greater_than_9_l562_56286

def num_faces := 6
def total_outcomes := num_faces * num_faces
def favorable_outcomes := 6
def probability := favorable_outcomes / total_outcomes

theorem probability_sum_greater_than_9 (h : total_outcomes = 36) :
  probability = 1 / 6 :=
by
  sorry

end probability_sum_greater_than_9_l562_56286


namespace common_ratio_geometric_sequence_l562_56231

theorem common_ratio_geometric_sequence
  (a_1 : ℝ) (n : ℕ) (S : ℕ → ℝ)
  (geom_sum : ∀ n q, q ≠ 1 → S n = a_1 * (1 - q^n) / (1 - q))
  (h_arithmetic : 2 * S 4 = S 5 + S 6)
  : (∃ q : ℝ, ∀ n : ℕ, q ≠ 1 → S n = a_1 * (1 - q^n) / (1 - q)) → q = -2 :=
by
  sorry

end common_ratio_geometric_sequence_l562_56231


namespace blue_cards_in_box_l562_56233

theorem blue_cards_in_box (x : ℕ) (h : 0.6 = (x : ℝ) / (x + 8)) : x = 12 :=
sorry

end blue_cards_in_box_l562_56233


namespace sufficient_condition_of_implications_l562_56264

variables (P1 P2 θ : Prop)

theorem sufficient_condition_of_implications
  (h1 : P1 → θ)
  (h2 : P2 → P1) :
  P2 → θ :=
by sorry

end sufficient_condition_of_implications_l562_56264


namespace evaluate_expression_l562_56282

theorem evaluate_expression : 
  (20 - 19 + 18 - 17 + 16 - 15 + 14 - 13 + 12 - 11 + 10 - 9 + 8 - 7 + 6 - 5 + 4 - 3 + 2 - 1) / 
  (2 - 3 + 4 - 5 + 6 - 7 + 8 - 9 + 10 - 11 + 12 - 13 + 14 - 15 + 16 - 17 + 18 - 19 + 20)
  = 10 / 11 := 
by
  sorry

end evaluate_expression_l562_56282


namespace not_divisible_by_q_plus_one_l562_56220

theorem not_divisible_by_q_plus_one (q : ℕ) (hq_odd : q % 2 = 1) (hq_gt_two : q > 2) :
  ¬ (q + 1) ∣ ((q + 1) ^ ((q - 1) / 2) + 2) :=
by
  sorry

end not_divisible_by_q_plus_one_l562_56220


namespace no_consecutive_squares_of_arithmetic_progression_l562_56241

theorem no_consecutive_squares_of_arithmetic_progression (d : ℕ):
  (d % 10000 = 2019) →
  (∀ a b c : ℕ, a < b ∧ b < c → b^2 - a^2 = d ∧ c^2 - b^2 = d →
  false) :=
sorry

end no_consecutive_squares_of_arithmetic_progression_l562_56241


namespace sphere_shot_radius_l562_56248

theorem sphere_shot_radius (R : ℝ) (N : ℕ) (π : ℝ) (r : ℝ) 
  (h₀ : R = 4) (h₁ : N = 64) 
  (h₂ : (4 / 3) * π * (R ^ 3) / ((4 / 3) * π * (r ^ 3)) = N) : 
  r = 1 := 
by
  sorry

end sphere_shot_radius_l562_56248


namespace evaluate_x3_minus_y3_l562_56225

theorem evaluate_x3_minus_y3 (x y : ℤ) (h1 : x + y = 12) (h2 : 3 * x + y = 20) : x^3 - y^3 = -448 :=
by
  sorry

end evaluate_x3_minus_y3_l562_56225


namespace ivan_max_13_bars_a_ivan_max_13_bars_b_l562_56209

variable (n : ℕ) (ivan_max_bags : ℕ)

-- Condition 1: initial count of bars in the chest
def initial_bars := 13

-- Condition 2: function to check if transfers are possible
def can_transfer (bars_in_chest : ℕ) (bars_in_bag : ℕ) (last_transfer : ℕ) : Prop :=
  ∃ t₁ t₂, t₁ ≠ t₂ ∧ t₁ ≠ last_transfer ∧ t₂ ≠ last_transfer ∧
           t₁ + bars_in_bag ≤ initial_bars ∧ bars_in_chest - t₁ + t₂ = bars_in_chest

-- Proof Problem (a): Given initially 13 bars, prove Ivan can secure 13 bars
theorem ivan_max_13_bars_a 
  (initial_bars : ℕ := 13) 
  (target_bars : ℕ := 13)
  (can_transfer : ∀ (bars_in_chest bars_in_bag last_transfer : ℕ), can_transfer bars_in_chest bars_in_bag last_transfer) 
  (h_initial_bars : initial_bars = 13) :
  ivan_max_bags = target_bars :=
by
  sorry

-- Proof Problem (b): Given initially 14 bars, prove Ivan can secure 13 bars
theorem ivan_max_13_bars_b 
  (initial_bars : ℕ := 14)
  (target_bars : ℕ := 13)
  (can_transfer : ∀ (bars_in_chest bars_in_bag last_transfer : ℕ), can_transfer bars_in_chest bars_in_bag last_transfer) 
  (h_initial_bars : initial_bars = 14) :
  ivan_max_bags = target_bars :=
by
  sorry

end ivan_max_13_bars_a_ivan_max_13_bars_b_l562_56209


namespace zilla_savings_l562_56295

theorem zilla_savings (earnings : ℝ) (rent : ℝ) (expenses : ℝ) (savings : ℝ) 
  (h1 : rent = 0.07 * earnings)
  (h2 : rent = 133)
  (h3 : expenses = earnings / 2)
  (h4 : savings = earnings - rent - expenses) :
  savings = 817 := 
sorry

end zilla_savings_l562_56295


namespace max_gcd_is_one_l562_56207

-- Defining the sequence a_n
def a_n (n : ℕ) : ℕ := 101 + n^3

-- Defining the gcd function for a_n and a_(n+1)
def d_n (n : ℕ) : ℕ := Nat.gcd (a_n n) (a_n (n + 1))

-- The theorem stating the maximum value of d_n is 1
theorem max_gcd_is_one : ∀ n : ℕ, d_n n = 1 := by
  -- Proof is omitted as per instructions
  sorry

end max_gcd_is_one_l562_56207


namespace box_cookies_count_l562_56205

theorem box_cookies_count (cookies_per_bag : ℕ) (cookies_per_box : ℕ) :
  cookies_per_bag = 7 →
  8 * cookies_per_box = 9 * cookies_per_bag + 33 →
  cookies_per_box = 12 :=
by
  intros h1 h2
  sorry

end box_cookies_count_l562_56205


namespace Youseff_time_difference_l562_56242

theorem Youseff_time_difference 
  (blocks : ℕ)
  (walk_time_per_block : ℕ) 
  (bike_time_per_block_sec : ℕ) 
  (sec_per_min : ℕ)
  (h_blocks : blocks = 12) 
  (h_walk_time_per_block : walk_time_per_block = 1) 
  (h_bike_time_per_block_sec : bike_time_per_block_sec = 20) 
  (h_sec_per_min : sec_per_min = 60) : 
  (blocks * walk_time_per_block) - ((blocks * bike_time_per_block_sec) / sec_per_min) = 8 :=
by 
  sorry

end Youseff_time_difference_l562_56242


namespace denomination_is_20_l562_56271

noncomputable def denomination_of_250_coins (x : ℕ) : Prop :=
  250 * x + 84 * 25 = 7100

theorem denomination_is_20 (x : ℕ) (h : denomination_of_250_coins x) : x = 20 :=
by
  sorry

end denomination_is_20_l562_56271


namespace number_of_adults_l562_56274

-- Given constants
def children : ℕ := 200
def price_child (price_adult : ℕ) : ℕ := price_adult / 2
def total_amount : ℕ := 16000

-- Based on the problem conditions
def price_adult := 32

-- The generated proof problem
theorem number_of_adults 
    (price_adult_gt_0 : price_adult > 0)
    (h_price_adult : price_adult = 32)
    (h_total_amount : total_amount = 16000) 
    (h_price_relation : ∀ price_adult, price_adult / 2 * 2 = price_adult) :
  ∃ A : ℕ, 32 * A + 16 * 200 = 16000 ∧ price_child price_adult = 16 := by
  sorry

end number_of_adults_l562_56274


namespace carmen_rope_gcd_l562_56261

/-- Carmen has three ropes with lengths 48, 64, and 80 inches respectively.
    She needs to cut these ropes into pieces of equal length for a craft project,
    ensuring no rope is left unused.
    Prove that the greatest length in inches that each piece can have is 16. -/
theorem carmen_rope_gcd :
  Nat.gcd (Nat.gcd 48 64) 80 = 16 := by
  sorry

end carmen_rope_gcd_l562_56261


namespace relay_go_match_outcomes_l562_56230

theorem relay_go_match_outcomes : (Nat.choose 14 7) = 3432 := by
  sorry

end relay_go_match_outcomes_l562_56230


namespace runners_adjacent_vertices_after_2013_l562_56229

def hexagon_run_probability (t : ℕ) : ℚ :=
  (2 / 3) + (1 / 3) * ((1 / 4) ^ t)

theorem runners_adjacent_vertices_after_2013 :
  hexagon_run_probability 2013 = (2 / 3) + (1 / 3) * ((1 / 4) ^ 2013) := 
by 
  sorry

end runners_adjacent_vertices_after_2013_l562_56229


namespace sum_due_is_363_l562_56247

/-
Conditions:
1. BD = 78
2. TD = 66
3. The formula: BD = TD + (TD^2 / PV)
This should imply that PV = 363 given the conditions.
-/

theorem sum_due_is_363 (BD TD PV : ℝ) (h1 : BD = 78) (h2 : TD = 66) (h3 : BD = TD + (TD^2 / PV)) : PV = 363 :=
by
  sorry

end sum_due_is_363_l562_56247


namespace apples_given_by_anita_l562_56208

variable (initial_apples current_apples needed_apples : ℕ)

theorem apples_given_by_anita (h1 : initial_apples = 4) 
                               (h2 : needed_apples = 10)
                               (h3 : needed_apples - current_apples = 1) : 
  current_apples - initial_apples = 5 := 
by
  sorry

end apples_given_by_anita_l562_56208


namespace tangent_line_intersection_l562_56216

theorem tangent_line_intersection
    (circle1_center : ℝ × ℝ)
    (circle1_radius : ℝ)
    (circle2_center : ℝ × ℝ)
    (circle2_radius : ℝ)
    (tangent_intersection_x : ℝ)
    (h1 : circle1_center = (0, 0))
    (h2 : circle1_radius = 3)
    (h3 : circle2_center = (12, 0))
    (h4 : circle2_radius = 5)
    (h5 : tangent_intersection_x > 0) :
    tangent_intersection_x = 9 / 2 := by
  sorry

end tangent_line_intersection_l562_56216


namespace problem1_problem2_l562_56254

theorem problem1 : (Real.sqrt 18 - Real.sqrt 8 - Real.sqrt 2) = 0 := 
by sorry

theorem problem2 : (6 * Real.sqrt 2 * Real.sqrt 3 + 3 * Real.sqrt 30 / Real.sqrt 5) = 9 * Real.sqrt 6 := 
by sorry

end problem1_problem2_l562_56254


namespace sin_sum_cos_product_tan_sum_tan_product_l562_56292

theorem sin_sum_cos_product
  (A B C : ℝ)
  (h : A + B + C = π) : 
  (Real.sin A + Real.sin B + Real.sin C = 4 * Real.cos (A / 2) * Real.cos (B / 2) * Real.cos (C / 2)) :=
sorry

theorem tan_sum_tan_product
  (A B C : ℝ)
  (h : A + B + C = π) :
  (Real.tan A + Real.tan B + Real.tan C = Real.tan A * Real.tan B * Real.tan C) := 
sorry

end sin_sum_cos_product_tan_sum_tan_product_l562_56292


namespace filled_sandbag_weight_is_correct_l562_56243

-- Define the conditions
def sandbag_weight : ℝ := 250
def fill_percent : ℝ := 0.80
def heavier_factor : ℝ := 1.40

-- Define the intermediate weights
def sand_weight : ℝ := sandbag_weight * fill_percent
def extra_weight : ℝ := sand_weight * (heavier_factor - 1)
def filled_material_weight : ℝ := sand_weight + extra_weight

-- Define the total weight including the empty sandbag
def total_weight : ℝ := sandbag_weight + filled_material_weight

-- Prove the total weight is correct
theorem filled_sandbag_weight_is_correct : total_weight = 530 := 
by sorry

end filled_sandbag_weight_is_correct_l562_56243


namespace exists_visible_point_l562_56218

open Nat -- to use natural numbers and their operations

def is_visible (x y : ℤ) : Prop :=
  Int.gcd x y = 1

theorem exists_visible_point (n : ℕ) (hn : n > 0) :
  ∃ a b : ℤ, is_visible a b ∧
  ∀ (P : ℤ × ℤ), (P ≠ (a, b) → (Int.sqrt ((P.fst - a) * (P.fst - a) + (P.snd - b) * (P.snd - b)) > n)) :=
sorry

end exists_visible_point_l562_56218


namespace max_and_min_sum_of_factors_of_2000_l562_56269

theorem max_and_min_sum_of_factors_of_2000 :
  ∃ (a b c d e : ℕ), 1 < a ∧ 1 < b ∧ 1 < c ∧ 1 < d ∧ 1 < e ∧ a * b * c * d * e = 2000
  ∧ (a + b + c + d + e = 133 ∨ a + b + c + d + e = 23) :=
by
  sorry

end max_and_min_sum_of_factors_of_2000_l562_56269


namespace probability_of_one_machine_maintenance_l562_56287

theorem probability_of_one_machine_maintenance :
  let pA := 0.1
  let pB := 0.2
  let pC := 0.4
  let qA := 1 - pA
  let qB := 1 - pB
  let qC := 1 - pC
  (pA * qB * qC) + (qA * pB * qC) + (qA * qB * pC) = 0.444 :=
by {
  let pA := 0.1
  let pB := 0.2
  let pC := 0.4
  let qA := 1 - pA
  let qB := 1 - pB
  let qC := 1 - pC
  show (pA * qB * qC) + (qA * pB * qC) + (qA * qB * pC) = 0.444
  sorry
}

end probability_of_one_machine_maintenance_l562_56287


namespace min_value_reciprocal_l562_56214

theorem min_value_reciprocal (m n : ℝ) (hmn_gt : 0 < m * n) (hmn_add : m + n = 2) :
  (∃ x : ℝ, x = (1/m + 1/n) ∧ x = 2) :=
by sorry

end min_value_reciprocal_l562_56214


namespace aluminum_phosphate_molecular_weight_l562_56234

theorem aluminum_phosphate_molecular_weight :
  let Al := 26.98
  let P := 30.97
  let O := 16.00
  (Al + P + 4 * O) = 121.95 :=
by
  let Al := 26.98
  let P := 30.97
  let O := 16.00
  sorry

end aluminum_phosphate_molecular_weight_l562_56234


namespace distinct_solutions_diff_l562_56255

theorem distinct_solutions_diff (r s : ℝ) (hr : r > s) 
  (h : ∀ x, (x - 5) * (x + 5) = 25 * x - 125 ↔ x = r ∨ x = s) : r - s = 15 :=
sorry

end distinct_solutions_diff_l562_56255


namespace min_balls_for_color_15_l562_56251

theorem min_balls_for_color_15
  (red green yellow blue white black : ℕ)
  (h_red : red = 28)
  (h_green : green = 20)
  (h_yellow : yellow = 19)
  (h_blue : blue = 13)
  (h_white : white = 11)
  (h_black : black = 9) :
  ∃ n, n = 76 ∧ ∀ balls_drawn, balls_drawn = n →
  ∃ color, 
    (color = "red" ∧ balls_drawn ≥ 15 ∧ balls_drawn <= red) ∨
    (color = "green" ∧ balls_drawn ≥ 15 ∧ balls_drawn <= green) ∨
    (color = "yellow" ∧ balls_drawn ≥ 15 ∧ balls_drawn <= yellow) ∨
    (color = "blue" ∧ balls_drawn ≥ 15 ∧ balls_drawn <= blue) ∨
    (color = "white" ∧ balls_drawn >= 15 ∧ balls_drawn <= white) ∨
    (color = "black" ∧ balls_drawn ≥ 15 ∧ balls_drawn <= black) := 
sorry

end min_balls_for_color_15_l562_56251


namespace show_linear_l562_56249

-- Define the conditions as given in the problem
variables (a b : ℤ)

-- The hypothesis that the equation is linear
def linear_equation_hypothesis : Prop :=
  (a + b = 1) ∧ (3 * a + 2 * b - 4 = 1)

-- Define the theorem we need to prove
theorem show_linear (h : linear_equation_hypothesis a b) : a + b = 1 := 
by
  sorry

end show_linear_l562_56249


namespace asymptotes_of_hyperbola_l562_56202

theorem asymptotes_of_hyperbola (a : ℝ) :
  (∃ a : ℝ, 9 + a = 13) →
  (∀ x y : ℝ, (x^2 / 9 - y^2 / a = 1) → (a = 4)) →
  (forall (x y : ℝ), (x^2 / 9 - y^2 / 4 = 0) → 
    (y = (2/3) * x) ∨ (y = -(2/3) * x)) :=
by
  sorry

end asymptotes_of_hyperbola_l562_56202


namespace min_value_of_sum_inverse_l562_56276

theorem min_value_of_sum_inverse (m n : ℝ) 
  (H1 : ∃ (x y : ℝ), (x + y - 1 = 0 ∧ 3 * x - y - 7 = 0) ∧ (mx + y + n = 0))
  (H2 : mn > 0) : 
  ∃ k : ℝ, k = 8 ∧ ∀ (m n : ℝ), mn > 0 → (2 * m + n = 1) → 1 / m + 2 / n ≥ k :=
by
  sorry

end min_value_of_sum_inverse_l562_56276


namespace terminal_side_in_third_quadrant_l562_56217

-- Define the conditions
def sin_condition (α : Real) : Prop := Real.sin α < 0
def tan_condition (α : Real) : Prop := Real.tan α > 0

-- State the theorem
theorem terminal_side_in_third_quadrant (α : Real) (h1 : sin_condition α) (h2 : tan_condition α) : α ∈ Set.Ioo (π / 2) π :=
  sorry

end terminal_side_in_third_quadrant_l562_56217


namespace evaluate_fractions_l562_56298

theorem evaluate_fractions (a b c : ℝ) (h : a / (30 - a) + b / (70 - b) + c / (75 - c) = 9) :
  6 / (30 - a) + 14 / (70 - b) + 15 / (75 - c) = 35 :=
by
  sorry

end evaluate_fractions_l562_56298


namespace son_work_rate_l562_56213

noncomputable def man_work_rate := 1/10
noncomputable def combined_work_rate := 1/5

theorem son_work_rate :
  ∃ S : ℝ, man_work_rate + S = combined_work_rate ∧ S = 1/10 := sorry

end son_work_rate_l562_56213


namespace problem_l562_56206

def is_irrational (x : ℝ) : Prop := ¬ ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

theorem problem :
  let A := 3.14159265
  let B := Real.sqrt 36
  let C := Real.sqrt 7
  let D := 4.1
  is_irrational C := by
  sorry

end problem_l562_56206


namespace area_of_triangle_l562_56273

theorem area_of_triangle (a b c : ℝ) (C : ℝ) 
  (h1 : c^2 = (a - b)^2 + 6)
  (h2 : C = Real.pi / 3) : 
  1/2 * a * b * Real.sin C = (3 * Real.sqrt 3) / 2 :=
by
  sorry

end area_of_triangle_l562_56273


namespace f_diff_l562_56237

def f (n : ℕ) : ℚ :=
  (Finset.range (3 * n)).sum (λ k => (1 : ℚ) / (k + 1))

theorem f_diff (n : ℕ) : f (n + 1) - f n = (1 / (3 * n) + 1 / (3 * n + 1) + 1 / (3 * n + 2)) :=
by
  sorry

end f_diff_l562_56237


namespace value_of_a_minus_b_l562_56262

theorem value_of_a_minus_b (a b c : ℝ) 
    (h1 : 2011 * a + 2015 * b + c = 2021)
    (h2 : 2013 * a + 2017 * b + c = 2023)
    (h3 : 2012 * a + 2016 * b + 2 * c = 2026) : 
    a - b = -2 := 
by
  sorry

end value_of_a_minus_b_l562_56262


namespace weight_difference_l562_56215

theorem weight_difference (brown black white grey : ℕ) 
  (h_brown : brown = 4)
  (h_white : white = 2 * brown)
  (h_grey : grey = black - 2)
  (avg_weight : (brown + black + white + grey) / 4 = 5): 
  (black - brown) = 1 := by
  sorry

end weight_difference_l562_56215


namespace geometric_sequence_l562_56211

open Nat

-- Define the sequence and conditions for the problem
variable {a : ℕ → ℕ}
variable {S : ℕ → ℕ}
variable {m p : ℕ}
variable (h1 : a 1 ≠ 0)
variable (h2 : ∀ n : ℕ, 2 * S (n + 1) - 3 * S n = 2 * a 1)
variable (h3 : S 0 = 0)
variable (h4 : ∀ n : ℕ, S (n + 1) = S n + a (n + 1))
variable (h5 : a 1 ≥ m^(p-1))
variable (h6 : a p ≤ (m+1)^(p-1))

-- The theorem that we need to prove
theorem geometric_sequence (n : ℕ) : 
  (exists r : ℕ → ℕ, ∀ k : ℕ, a (k + 1) = r (k + 1) * a k) ∧ 
  (∀ k : ℕ, a k = sorry) := sorry

end geometric_sequence_l562_56211


namespace sum_of_interior_angles_of_octagon_l562_56263

theorem sum_of_interior_angles_of_octagon (n : ℕ) (h : n = 8) : (n - 2) * 180 = 1080 := by
  sorry

end sum_of_interior_angles_of_octagon_l562_56263
