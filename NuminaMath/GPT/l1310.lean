import Mathlib

namespace cart_distance_traveled_l1310_131020

-- Define the problem parameters/conditions
def circumference_front : ℕ := 30
def circumference_back : ℕ := 33
def revolutions_difference : ℕ := 5

-- Define the question and the expected correct answer
theorem cart_distance_traveled :
  ∀ (R : ℕ), ((R + revolutions_difference) * circumference_front = R * circumference_back) → (R * circumference_back) = 1650 :=
by
  intro R h
  sorry

end cart_distance_traveled_l1310_131020


namespace weight_of_bowling_ball_l1310_131024

variable (b c : ℝ)

axiom h1 : 5 * b = 2 * c
axiom h2 : 3 * c = 84

theorem weight_of_bowling_ball : b = 11.2 :=
by
  sorry

end weight_of_bowling_ball_l1310_131024


namespace product_of_ninth_and_tenth_l1310_131040

def scores_first_8 := [7, 4, 3, 6, 8, 3, 1, 5]
def total_points_first_8 := scores_first_8.sum

def condition1 (ninth_game_points tenth_game_points : ℕ) : Prop :=
  ninth_game_points < 10 ∧ tenth_game_points < 10

def condition2 (ninth_game_points : ℕ) : Prop :=
  (total_points_first_8 + ninth_game_points) % 9 = 0

def condition3 (ninth_game_points tenth_game_points : ℕ) : Prop :=
  (total_points_first_8 + ninth_game_points + tenth_game_points) % 10 = 0

theorem product_of_ninth_and_tenth (ninth_game_points : ℕ) (tenth_game_points : ℕ) 
  (h1 : condition1 ninth_game_points tenth_game_points)
  (h2 : condition2 ninth_game_points)
  (h3 : condition3 ninth_game_points tenth_game_points) : 
  ninth_game_points * tenth_game_points = 40 :=
sorry

end product_of_ninth_and_tenth_l1310_131040


namespace term_in_census_is_population_l1310_131042

def term_for_entire_set_of_objects : String :=
  "population"

theorem term_in_census_is_population :
  term_for_entire_set_of_objects = "population" :=
sorry

end term_in_census_is_population_l1310_131042


namespace fruit_basket_ratio_l1310_131057

theorem fruit_basket_ratio (total_fruits : ℕ) (oranges : ℕ) (apples : ℕ) (h1 : total_fruits = 40) (h2 : oranges = 10) (h3 : apples = total_fruits - oranges) :
  (apples / oranges) = 3 := by
  sorry

end fruit_basket_ratio_l1310_131057


namespace All_Yarns_are_Zorps_and_Xings_l1310_131031

-- Define the basic properties
variables {α : Type}
variable (Zorp Xing Yarn Wit Vamp : α → Prop)

-- Given conditions
axiom all_Zorps_are_Xings : ∀ z, Zorp z → Xing z
axiom all_Yarns_are_Xings : ∀ y, Yarn y → Xing y
axiom all_Wits_are_Zorps : ∀ w, Wit w → Zorp w
axiom all_Yarns_are_Wits : ∀ y, Yarn y → Wit y
axiom all_Yarns_are_Vamps : ∀ y, Yarn y → Vamp y

-- Proof problem
theorem All_Yarns_are_Zorps_and_Xings : 
  ∀ y, Yarn y → (Zorp y ∧ Xing y) :=
sorry

end All_Yarns_are_Zorps_and_Xings_l1310_131031


namespace p_sufficient_not_necessary_for_q_l1310_131039

-- Definitions based on conditions
def p (x : ℝ) : Prop := 0 < x ∧ x < 2
def q (x : ℝ) : Prop := -1 < x ∧ x < 3

theorem p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, p x → q x) ∧ ¬ (∀ x : ℝ, q x → p x) :=
by
  sorry

end p_sufficient_not_necessary_for_q_l1310_131039


namespace expected_value_of_girls_left_of_boys_l1310_131058

def num_girls_to_left_of_all_boys (boys girls : ℕ) : ℚ :=
  (boys + girls : ℚ) / (boys + 1)

theorem expected_value_of_girls_left_of_boys :
  num_girls_to_left_of_all_boys 10 7 = 7 / 11 := by
  sorry

end expected_value_of_girls_left_of_boys_l1310_131058


namespace original_weight_of_beef_l1310_131022

theorem original_weight_of_beef (weight_after_processing : ℝ) (loss_percentage : ℝ) :
  loss_percentage = 0.5 → weight_after_processing = 750 → 
  (750 : ℝ) / (1 - 0.5) = 1500 :=
by
  intros h_loss_percent h_weight_after
  sorry

end original_weight_of_beef_l1310_131022


namespace find_original_number_l1310_131001

-- Variables and assumptions
variables (N y x : ℕ)

-- Conditions
def crossing_out_condition : Prop := N = 10 * y + x
def sum_condition : Prop := N + y = 54321
def remainder_condition : Prop := x = 54321 % 11
def y_value : Prop := y = 4938

-- The final proof problem statement
theorem find_original_number (h1 : crossing_out_condition N y x) (h2 : sum_condition N y) 
  (h3 : remainder_condition x) (h4 : y_value y) : N = 49383 :=
sorry

end find_original_number_l1310_131001


namespace num_pairs_divisible_7_l1310_131012

theorem num_pairs_divisible_7 (x y : ℕ) (hx : 1 ≤ x ∧ x ≤ 1000) (hy : 1 ≤ y ∧ y ≤ 1000)
  (divisible : (x^2 + y^2) % 7 = 0) : 
  (∃ k : ℕ, k = 20164) :=
sorry

end num_pairs_divisible_7_l1310_131012


namespace smallest_x_for_multiple_l1310_131006

theorem smallest_x_for_multiple (x : ℕ) (h₁: 450 = 2 * 3^2 * 5^2) (h₂: 800 = 2^6 * 5^2) : 
  ((450 * x) % 800 = 0) ↔ x ≥ 32 :=
by
  sorry

end smallest_x_for_multiple_l1310_131006


namespace exists_three_points_l1310_131088

theorem exists_three_points (n : ℕ) (h : 3 ≤ n) (points : Fin n → EuclideanSpace ℝ (Fin 2))
  (distinct : ∀ i j : Fin n, i ≠ j → points i ≠ points j) :
  ∃ (A B C : Fin n),
    A ≠ B ∧ A ≠ C ∧ B ≠ C ∧ 
    1 ≤ dist (points A) (points B) / dist (points A) (points C) ∧ 
    dist (points A) (points B) / dist (points A) (points C) < (n + 1) / (n - 1) := 
sorry

end exists_three_points_l1310_131088


namespace five_y_eq_45_over_7_l1310_131005

theorem five_y_eq_45_over_7 (x y : ℚ) (h1 : 3 * x + 4 * y = 0) (h2 : x = y - 3) : 5 * y = 45 / 7 := by
  sorry

end five_y_eq_45_over_7_l1310_131005


namespace teal_total_sales_l1310_131081

variable (pum_pie_slices_per_pie : ℕ) (cus_pie_slices_per_pie : ℕ)
variable (pum_pie_price_per_slice : ℕ) (cus_pie_price_per_slice : ℕ)
variable (pum_pies_sold : ℕ) (cus_pies_sold : ℕ)

def total_slices_sold (slices_per_pie pies_sold : ℕ) : ℕ :=
  slices_per_pie * pies_sold

def total_sales (slices_sold price_per_slice : ℕ) : ℕ :=
  slices_sold * price_per_slice

theorem teal_total_sales
  (h1 : pum_pie_slices_per_pie = 8)
  (h2 : cus_pie_slices_per_pie = 6)
  (h3 : pum_pie_price_per_slice = 5)
  (h4 : cus_pie_price_per_slice = 6)
  (h5 : pum_pies_sold = 4)
  (h6 : cus_pies_sold = 5) :
  (total_sales (total_slices_sold pum_pie_slices_per_pie pum_pies_sold) pum_pie_price_per_slice) +
  (total_sales (total_slices_sold cus_pie_slices_per_pie cus_pies_sold) cus_pie_price_per_slice) = 340 :=
by
  sorry

end teal_total_sales_l1310_131081


namespace odd_function_a_minus_b_l1310_131055

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_a_minus_b
  (a b : ℝ)
  (h : is_odd_function (λ x => 2 * x ^ 3 + a * x ^ 2 + b - 1)) :
  a - b = -1 :=
sorry

end odd_function_a_minus_b_l1310_131055


namespace largest_three_digit_sum_fifteen_l1310_131065

theorem largest_three_digit_sum_fifteen : ∃ (a b c : ℕ), (a = 9 ∧ b = 6 ∧ c = 0 ∧ 100 * a + 10 * b + c = 960 ∧ a + b + c = 15 ∧ a < 10 ∧ b < 10 ∧ c < 10) := by
  sorry

end largest_three_digit_sum_fifteen_l1310_131065


namespace total_age_difference_l1310_131029

noncomputable def ages_difference (A B C : ℕ) : ℕ :=
  (A + B) - (B + C)

theorem total_age_difference (A B C : ℕ) (h₁ : A + B > B + C) (h₂ : C = A - 11) : ages_difference A B C = 11 :=
by
  sorry

end total_age_difference_l1310_131029


namespace point_in_third_quadrant_l1310_131030

theorem point_in_third_quadrant
  (a b : ℝ)
  (hne : a ≠ 0)
  (y_increase : ∀ x1 x2, x1 < x2 → -5 * a * x1 + b < -5 * a * x2 + b)
  (ab_pos : a * b > 0) : 
  a < 0 ∧ b < 0 :=
by
  sorry

end point_in_third_quadrant_l1310_131030


namespace probability_all_same_color_l1310_131048

theorem probability_all_same_color :
  let red_plates := 7
  let blue_plates := 5
  let total_plates := red_plates + blue_plates
  let total_combinations := Nat.choose total_plates 3
  let red_combinations := Nat.choose red_plates 3
  let blue_combinations := Nat.choose blue_plates 3
  let favorable_combinations := red_combinations + blue_combinations
  let probability := (favorable_combinations : ℚ) / total_combinations
  probability = 9 / 44 :=
by 
  sorry

end probability_all_same_color_l1310_131048


namespace usual_walk_time_l1310_131013

theorem usual_walk_time (S T : ℝ)
  (h : S / (2/3 * S) = (T + 15) / T) : T = 30 :=
by
  sorry

end usual_walk_time_l1310_131013


namespace hyperbola_asymptotes_l1310_131025

noncomputable def hyperbola_eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 + (b^2 / a^2))

theorem hyperbola_asymptotes (a b : ℝ) (h : hyperbola_eccentricity a b = Real.sqrt 3) :
  (∀ x y : ℝ, (y = Real.sqrt 2 * x) ∨ (y = -Real.sqrt 2 * x)) :=
sorry

end hyperbola_asymptotes_l1310_131025


namespace chimps_seen_l1310_131050

-- Given conditions
def lions := 8
def lion_legs := 4
def lizards := 5
def lizard_legs := 4
def tarantulas := 125
def tarantula_legs := 8
def goal_legs := 1100

-- Required to be proved
def chimp_legs := 4

theorem chimps_seen : (goal_legs - ((lions * lion_legs) + (lizards * lizard_legs) + (tarantulas * tarantula_legs))) / chimp_legs = 25 :=
by
  -- placeholder for the proof
  sorry

end chimps_seen_l1310_131050


namespace minimum_overlap_l1310_131089

variable (U : Finset ℕ) -- This is the set of all people surveyed
variable (B V : Finset ℕ) -- These are the sets of people who like Beethoven and Vivaldi respectively.

-- Given conditions:
axiom h_total : U.card = 120
axiom h_B : B.card = 95
axiom h_V : V.card = 80
axiom h_subset_B : B ⊆ U
axiom h_subset_V : V ⊆ U

-- Question to prove:
theorem minimum_overlap : (B ∩ V).card = 95 + 80 - 120 := by
  sorry

end minimum_overlap_l1310_131089


namespace max_length_segment_l1310_131043

theorem max_length_segment (p b : ℝ) (h : b = p / 2) : (b * (p - b)) / p = p / 4 :=
by
  sorry

end max_length_segment_l1310_131043


namespace simplify_expression_l1310_131076

variable (a b c : ℝ) 

theorem simplify_expression (h1 : a ≠ 4) (h2 : b ≠ 5) (h3 : c ≠ 6) :
  (a - 4) / (6 - c) * (b - 5) / (4 - a) * (c - 6) / (5 - b) = -1 :=
by
  sorry

end simplify_expression_l1310_131076


namespace count_multiples_of_12_l1310_131036

theorem count_multiples_of_12 (a b : ℤ) (h1 : 15 < a) (h2 : b < 205) (h3 : ∃ k : ℤ, a = 12 * k) (h4 : ∃ k : ℤ, b = 12 * k) : 
  ∃ n : ℕ, n = 16 := 
by 
  sorry

end count_multiples_of_12_l1310_131036


namespace intersection_condition_l1310_131074

-- Define the lines
def line1 (x y : ℝ) := 2*x - 2*y - 3 = 0
def line2 (x y : ℝ) := 3*x - 5*y + 1 = 0
def line (a b x y : ℝ) := a*x - y + b = 0

-- Define the condition
def condition (a b : ℝ) := 17*a + 4*b = 11

-- Prove that the line l passes through the intersection point of l1 and l2 if and only if the condition holds
theorem intersection_condition (a b : ℝ) :
  (∃ x y : ℝ, line1 x y ∧ line2 x y ∧ line a b x y) ↔ condition a b :=
  sorry

end intersection_condition_l1310_131074


namespace fly_travel_distance_l1310_131072

theorem fly_travel_distance
  (carA_speed : ℕ)
  (carB_speed : ℕ)
  (initial_distance : ℕ)
  (fly_speed : ℕ)
  (relative_speed : ℕ := carB_speed - carA_speed)
  (catchup_time : ℚ := initial_distance / relative_speed)
  (fly_travel : ℚ := fly_speed * catchup_time) :
  carA_speed = 20 → carB_speed = 30 → initial_distance = 1 → fly_speed = 40 → fly_travel = 4 :=
by
  sorry

end fly_travel_distance_l1310_131072


namespace probability_not_finishing_on_time_l1310_131098

-- Definitions based on the conditions
def P_finishing_on_time : ℚ := 5 / 8

-- Theorem to prove the required probability
theorem probability_not_finishing_on_time :
  (1 - P_finishing_on_time) = 3 / 8 := by
  sorry

end probability_not_finishing_on_time_l1310_131098


namespace deepak_age_l1310_131044

variable (A D : ℕ)

theorem deepak_age (h1 : A / D = 2 / 3) (h2 : A + 5 = 25) : D = 30 :=
sorry

end deepak_age_l1310_131044


namespace find_x_plus_one_over_x_l1310_131078

variable (x : ℝ)

theorem find_x_plus_one_over_x
  (h1 : x^3 + (1/x)^3 = 110)
  (h2 : (x + 1/x)^2 - 2*x - 2*(1/x) = 38) :
  x + 1/x = 5 :=
sorry

end find_x_plus_one_over_x_l1310_131078


namespace find_amplitude_l1310_131019

noncomputable def amplitude_of_cosine (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) : ℝ :=
  a

theorem find_amplitude (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_max : amplitude_of_cosine a b h_a h_b = 3) :
  a = 3 :=
sorry

end find_amplitude_l1310_131019


namespace jessica_threw_away_4_roses_l1310_131011

def roses_thrown_away (a b c d : ℕ) : Prop :=
  (a + b) - d = c

theorem jessica_threw_away_4_roses :
  roses_thrown_away 2 25 23 4 :=
by
  -- This is where the proof would go
  sorry

end jessica_threw_away_4_roses_l1310_131011


namespace solve_for_x_l1310_131094

theorem solve_for_x (x : ℝ) (h : 9 / (1 + 4 / x) = 1) : x = 1 / 2 :=
by
  sorry

end solve_for_x_l1310_131094


namespace maximum_area_of_inscribed_rectangle_l1310_131053

theorem maximum_area_of_inscribed_rectangle (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∃ (A : ℝ), A = (a * b) / 4 :=
by
  sorry -- placeholder for the proof

end maximum_area_of_inscribed_rectangle_l1310_131053


namespace sum_xyz_l1310_131061

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem sum_xyz :
  (∀ x y z : ℝ,
  log_base 3 (log_base 4 (log_base 5 x)) = 0 ∧
  log_base 4 (log_base 5 (log_base 3 y)) = 0 ∧
  log_base 5 (log_base 3 (log_base 4 z)) = 0 →
  x + y + z = 932) := 
by
  sorry

end sum_xyz_l1310_131061


namespace sally_purchased_20_fifty_cent_items_l1310_131026

noncomputable def num_fifty_cent_items (x y z : ℕ) (h1 : x + y + z = 30) (h2 : 50 * x + 500 * y + 1000 * z = 10000) : ℕ :=
x

theorem sally_purchased_20_fifty_cent_items
  (x y z : ℕ)
  (h1 : x + y + z = 30)
  (h2 : 50 * x + 500 * y + 1000 * z = 10000)
  : num_fifty_cent_items x y z h1 h2 = 20 :=
sorry

end sally_purchased_20_fifty_cent_items_l1310_131026


namespace units_digit_first_four_composite_is_eight_l1310_131045

-- Definitions of the first four positive composite numbers
def first_four_composite_numbers : List ℕ := [4, 6, 8, 9]

-- Define the product of the first four composite numbers
def product_first_four_composite : ℕ := first_four_composite_numbers.prod

-- Define the function to extract the units digit of a number
def units_digit (n : ℕ) : ℕ := n % 10

-- The main statement to prove
theorem units_digit_first_four_composite_is_eight : units_digit product_first_four_composite = 8 := 
sorry

end units_digit_first_four_composite_is_eight_l1310_131045


namespace number_of_tires_slashed_l1310_131093

-- Definitions based on conditions
def cost_per_tire : ℤ := 250
def cost_window : ℤ := 700
def total_cost : ℤ := 1450

-- Proof statement
theorem number_of_tires_slashed : ∃ T : ℤ, cost_per_tire * T + cost_window = total_cost ∧ T = 3 := 
sorry

end number_of_tires_slashed_l1310_131093


namespace factorial_div_eq_l1310_131084
-- Import the entire math library

-- Define the entities involved in the problem
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

-- Define the given conditions
def given_expression : ℕ := factorial 10 / (factorial 7 * factorial 3)

-- State the main theorem that corresponds to the given problem and its correct answer
theorem factorial_div_eq : given_expression = 120 :=
by 
  -- Proof is omitted
  sorry

end factorial_div_eq_l1310_131084


namespace brendan_remaining_money_l1310_131063

-- Definitions given in the conditions
def weekly_pay (total_monthly_earnings : ℕ) (weeks_in_month : ℕ) : ℕ := total_monthly_earnings / weeks_in_month
def weekly_recharge_amount (weekly_pay : ℕ) : ℕ := weekly_pay / 2
def total_recharge_amount (weekly_recharge_amount : ℕ) (weeks_in_month : ℕ) : ℕ := weekly_recharge_amount * weeks_in_month
def remaining_money_after_car_purchase (total_monthly_earnings : ℕ) (car_cost : ℕ) : ℕ := total_monthly_earnings - car_cost
def total_remaining_money (remaining_money_after_car_purchase : ℕ) (total_recharge_amount : ℕ) : ℕ := remaining_money_after_car_purchase - total_recharge_amount

-- The actual statement to prove
theorem brendan_remaining_money
  (total_monthly_earnings : ℕ := 5000)
  (weeks_in_month : ℕ := 4)
  (car_cost : ℕ := 1500)
  (weekly_pay := weekly_pay total_monthly_earnings weeks_in_month)
  (weekly_recharge_amount := weekly_recharge_amount weekly_pay)
  (total_recharge_amount := total_recharge_amount weekly_recharge_amount weeks_in_month)
  (remaining_money_after_car_purchase := remaining_money_after_car_purchase total_monthly_earnings car_cost)
  (total_remaining_money := total_remaining_money remaining_money_after_car_purchase total_recharge_amount) :
  total_remaining_money = 1000 :=
sorry

end brendan_remaining_money_l1310_131063


namespace bob_start_time_l1310_131091

-- Define constants for the problem conditions
def yolandaRate : ℝ := 3 -- Yolanda's walking rate in miles per hour
def bobRate : ℝ := 4 -- Bob's walking rate in miles per hour
def distanceXY : ℝ := 10 -- Distance between point X and Y in miles
def bobDistanceWhenMet : ℝ := 4 -- Distance Bob had walked when they met in miles

-- Define the theorem statement
theorem bob_start_time : 
  ∃ T : ℝ, (yolandaRate * T + bobDistanceWhenMet = distanceXY) →
  (T = 2) →
  ∃ tB : ℝ, T - tB = 1 :=
by
  -- Insert proof here
  sorry

end bob_start_time_l1310_131091


namespace simplify_expression_l1310_131004

theorem simplify_expression (x : ℝ) : 7 * x + 9 - 3 * x + 15 * 2 = 4 * x + 39 := 
by sorry

end simplify_expression_l1310_131004


namespace value_of_g_at_3_l1310_131010

def g (x : ℝ) := x^2 - 2*x + 1

theorem value_of_g_at_3 : g 3 = 4 :=
by
  sorry

end value_of_g_at_3_l1310_131010


namespace focus_with_greatest_y_coordinate_l1310_131014

-- Define the conditions as hypotheses
def ellipse_major_axis : (ℝ × ℝ) := (0, 3)
def ellipse_minor_axis : (ℝ × ℝ) := (2, 0)
def ellipse_semi_major_axis : ℝ := 3
def ellipse_semi_minor_axis : ℝ := 2

-- Define the theorem to compute the coordinates of the focus with the greater y-coordinate
theorem focus_with_greatest_y_coordinate :
  let a := ellipse_semi_major_axis
  let b := ellipse_semi_minor_axis
  let c := Real.sqrt (a^2 - b^2)
  (0, c) = (0, (Real.sqrt 5) / 2) :=
by
  -- skipped proof
  sorry

end focus_with_greatest_y_coordinate_l1310_131014


namespace daisies_bought_l1310_131087

theorem daisies_bought (cost_per_flower roses total_cost : ℕ) 
  (h1 : cost_per_flower = 3) 
  (h2 : roses = 8) 
  (h3 : total_cost = 30) : 
  (total_cost - (roses * cost_per_flower)) / cost_per_flower = 2 :=
by
  sorry

end daisies_bought_l1310_131087


namespace smallest_divisible_by_15_11_12_l1310_131038

theorem smallest_divisible_by_15_11_12 : ∃ n : ℕ, (n > 0) ∧ (15 ∣ n) ∧ (11 ∣ n) ∧ (12 ∣ n) ∧ (∀ m : ℕ, (m > 0) ∧ (15 ∣ m) ∧ (11 ∣ m) ∧ (12 ∣ m) → n ≤ m) ∧ n = 660 :=
by
  sorry

end smallest_divisible_by_15_11_12_l1310_131038


namespace cone_height_l1310_131086

theorem cone_height (R : ℝ) (h : ℝ) (r : ℝ) : 
  R = 8 → r = 2 → h = 2 * Real.sqrt 15 :=
by
  intro hR hr
  sorry

end cone_height_l1310_131086


namespace find_nine_boxes_of_same_variety_l1310_131015

theorem find_nine_boxes_of_same_variety (boxes : ℕ) (A B C : ℕ) (h_total : boxes = 25) (h_one_variety : boxes = A + B + C) 
  (hA : A ≤ 25) (hB : B ≤ 25) (hC : C ≤ 25) :
  (A ≥ 9) ∨ (B ≥ 9) ∨ (C ≥ 9) :=
sorry

end find_nine_boxes_of_same_variety_l1310_131015


namespace range_of_a_l1310_131066

noncomputable def f (x : ℝ) : ℝ := x + Real.log x
noncomputable def g (a x : ℝ) : ℝ := a * x - 2 * Real.sin x

theorem range_of_a (a : ℝ) :
  (∀ x₁ > 0, ∃ x₂, (1 + 1 / x₁) * (a - 2 * Real.cos x₂) = -1) →
  -2 ≤ a ∧ a ≤ 1 :=
by {
  sorry
}

end range_of_a_l1310_131066


namespace find_real_medal_min_weighings_l1310_131099

axiom has_9_medals : Prop
axiom one_real_medal : Prop
axiom real_medal_heavier : Prop
axiom has_balance_scale : Prop

theorem find_real_medal_min_weighings
  (h1 : has_9_medals)
  (h2 : one_real_medal)
  (h3 : real_medal_heavier)
  (h4 : has_balance_scale) :
  ∃ (minimum_weighings : ℕ), minimum_weighings = 2 := 
  sorry

end find_real_medal_min_weighings_l1310_131099


namespace find_track_circumference_l1310_131092

noncomputable def track_circumference : ℝ := 720

theorem find_track_circumference
  (A B : ℝ)
  (uA uB : ℝ)
  (h1 : A = 0)
  (h2 : B = track_circumference / 2)
  (h3 : ∀ t : ℝ, A + uA * t = B + uB * t ↔ t = 150 / uB)
  (h4 : ∀ t : ℝ, A + uA * t = B + uB * t ↔ t = (track_circumference - 90) / uA)
  (h5 : ∀ t : ℝ, A + uA * t = B + uB * t ↔ t = 1.5 * track_circumference / uA) :
  track_circumference = 720 :=
by sorry

end find_track_circumference_l1310_131092


namespace P_inequality_l1310_131023

def P (n : ℕ) (x : ℝ) : ℝ := (Finset.range (n + 1)).sum (λ k => x^k)

theorem P_inequality (x : ℝ) (hx : 0 < x) :
  P 20 x * P 21 (x^2) ≤ P 20 (x^2) * P 22 x :=
by
  sorry

end P_inequality_l1310_131023


namespace largest_a_for_integer_solution_l1310_131071

theorem largest_a_for_integer_solution :
  ∃ a : ℝ, (∀ x y : ℤ, x - 4 * y = 1 ∧ a * x + 3 * y = 1) ∧ (∀ a' : ℝ, (∀ x y : ℤ, x - 4 * y = 1 ∧ a' * x + 3 * y = 1) → a' ≤ a) ∧ a = 1 :=
sorry

end largest_a_for_integer_solution_l1310_131071


namespace sum_first_four_terms_of_arithmetic_sequence_l1310_131064

theorem sum_first_four_terms_of_arithmetic_sequence (a₈ a₉ a₁₀ : ℤ) (d : ℤ) (a₁ a₂ a₃ a₄ : ℤ) : 
  (a₈ = 21) →
  (a₉ = 17) →
  (a₁₀ = 13) →
  (d = a₉ - a₈) →
  (a₁ = a₈ - 7 * d) →
  (a₂ = a₁ + d) →
  (a₃ = a₂ + d) →
  (a₄ = a₃ + d) →
  a₁ + a₂ + a₃ + a₄ = 172 :=
by 
  intros h₁ h₂ h₃ h₄ h₅ h₆ h₇ h₈
  sorry

end sum_first_four_terms_of_arithmetic_sequence_l1310_131064


namespace calculate_expression_l1310_131047

theorem calculate_expression : 
  2 - 1 / (2 - 1 / (2 + 2)) = 10 / 7 := 
by sorry

end calculate_expression_l1310_131047


namespace simplify_fraction_l1310_131079

theorem simplify_fraction (b : ℕ) (hb : b = 5) : (15 * b^4) / (90 * b^3 * b) = 1 / 6 := by
  sorry

end simplify_fraction_l1310_131079


namespace Emily_subtract_59_l1310_131009

theorem Emily_subtract_59 : (30 - 1) ^ 2 = 30 ^ 2 - 59 := by
  sorry

end Emily_subtract_59_l1310_131009


namespace fraction_addition_l1310_131034

theorem fraction_addition (d : ℤ) :
  (6 + 4 * d) / 9 + 3 / 2 = (39 + 8 * d) / 18 := sorry

end fraction_addition_l1310_131034


namespace find_a_when_lines_perpendicular_l1310_131008

theorem find_a_when_lines_perpendicular (a : ℝ) : 
  (∃ x y : ℝ, ax + 3 * y - 1 = 0 ∧  2 * x + (a^2 - a) * y + 3 = 0) ∧ 
  (∃ m₁ m₂ : ℝ, m₁ = -a / 3 ∧ m₂ = -2 / (a^2 - a) ∧ m₁ * m₂ = -1)
  → a = 0 ∨ a = 5 / 3 :=
by {
  sorry
}

end find_a_when_lines_perpendicular_l1310_131008


namespace line_passes_point_a_ne_zero_l1310_131097

theorem line_passes_point_a_ne_zero (a : ℝ) (h1 : ∀ (x y : ℝ), (y = 5 * x + a) → (x = a ∧ y = a^2)) (h2 : a ≠ 0) : a = 6 :=
sorry

end line_passes_point_a_ne_zero_l1310_131097


namespace total_days_needed_l1310_131051

-- Define the conditions
def project1_questions : ℕ := 518
def project2_questions : ℕ := 476
def questions_per_day : ℕ := 142

-- Define the statement to prove
theorem total_days_needed :
  (project1_questions + project2_questions) / questions_per_day = 7 := by
  sorry

end total_days_needed_l1310_131051


namespace initial_dog_cat_ratio_l1310_131062

theorem initial_dog_cat_ratio (C : ℕ) :
  75 / (C + 20) = 15 / 11 →
  (75 / C) = 15 / 7 :=
by
  sorry

end initial_dog_cat_ratio_l1310_131062


namespace gilbert_parsley_count_l1310_131095

variable (basil mint parsley : ℕ)
variable (initial_basil : ℕ := 3)
variable (extra_basil : ℕ := 1)
variable (initial_mint : ℕ := 2)
variable (herb_total : ℕ := 5)

def initial_parsley := herb_total - (initial_basil + extra_basil)

theorem gilbert_parsley_count : initial_parsley = 1 := by
  -- basil = initial_basil + extra_basil
  -- mint = 0 (since all mint plants eaten)
  -- herb_total = basil + parsley
  -- 5 = 4 + parsley
  -- parsley = 1
  sorry

end gilbert_parsley_count_l1310_131095


namespace arithmetic_sequence_2023rd_term_l1310_131035

theorem arithmetic_sequence_2023rd_term 
  (p q : ℤ)
  (h1 : 3 * p - q + 9 = 9)
  (h2 : 3 * (3 * p - q + 9) - q + 9 = 3 * p + q) :
  p + (2023 - 1) * (3 * p - q + 9) = 18189 := by
  sorry

end arithmetic_sequence_2023rd_term_l1310_131035


namespace max_items_per_cycle_l1310_131037

theorem max_items_per_cycle (shirts : Nat) (pants : Nat) (sweaters : Nat) (jeans : Nat)
  (cycle_time : Nat) (total_time : Nat) 
  (h_shirts : shirts = 18)
  (h_pants : pants = 12)
  (h_sweaters : sweaters = 17)
  (h_jeans : jeans = 13)
  (h_cycle_time : cycle_time = 45)
  (h_total_time : total_time = 3 * 60) :
  (shirts + pants + sweaters + jeans) / (total_time / cycle_time) = 15 :=
by
  -- We will provide the proof here
  sorry

end max_items_per_cycle_l1310_131037


namespace find_coefficients_sum_l1310_131016

theorem find_coefficients_sum (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ) :
  (∀ x : ℝ, (2 * x - 3)^5 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5) →
  a_1 + 2 * a_2 + 3 * a_3 + 4 * a_4 + 5 * a_5 = 10 :=
by
  intro h
  sorry

end find_coefficients_sum_l1310_131016


namespace original_price_lamp_l1310_131096

theorem original_price_lamp
  (P : ℝ)
  (discount_rate : ℝ)
  (discounted_price : ℝ)
  (discount_is_20_perc : discount_rate = 0.20)
  (new_price_is_96 : discounted_price = 96)
  (price_after_discount : discounted_price = P * (1 - discount_rate)) :
  P = 120 :=
by
  sorry

end original_price_lamp_l1310_131096


namespace apples_in_each_box_l1310_131090

theorem apples_in_each_box (x : ℕ) :
  (5 * x - (60 * 5)) = (2 * x) -> x = 100 :=
by
  sorry

end apples_in_each_box_l1310_131090


namespace sqrt_mul_neg_eq_l1310_131082

theorem sqrt_mul_neg_eq : - (Real.sqrt 2) * (Real.sqrt 7) = - (Real.sqrt 14) := sorry

end sqrt_mul_neg_eq_l1310_131082


namespace d_share_l1310_131032

theorem d_share (T : ℝ) (A B C D E : ℝ) 
  (h1 : A = 5 / 15 * T) 
  (h2 : B = 2 / 15 * T) 
  (h3 : C = 4 / 15 * T)
  (h4 : D = 3 / 15 * T)
  (h5 : E = 1 / 15 * T)
  (combined_AC : A + C = 3 / 5 * T)
  (diff_BE : B - E = 250) : 
  D = 750 :=
by
  sorry

end d_share_l1310_131032


namespace value_of_expression_l1310_131059

theorem value_of_expression :
  (3 * (3 * (3 * (3 * (3 * (3 * (3 + 2) + 2) + 2) + 2) + 2) + 2) + 2) = 4373 :=
by
  sorry

end value_of_expression_l1310_131059


namespace line_intersects_y_axis_at_0_2_l1310_131007

theorem line_intersects_y_axis_at_0_2 :
  ∃ y : ℝ, (2, 8) ≠ (4, 14) ∧ ∀ x: ℝ, (3 * x + y = 2) ∧ x = 0 → y = 2 :=
by
  sorry

end line_intersects_y_axis_at_0_2_l1310_131007


namespace arithmetic_geometric_sequence_l1310_131085

theorem arithmetic_geometric_sequence :
  ∀ (a₁ a₂ b₂ : ℝ),
    -- Conditions for arithmetic sequence: -1, a₁, a₂, 8
    2 * a₁ = -1 + a₂ ∧
    2 * a₂ = a₁ + 8 →
    -- Conditions for geometric sequence: -1, b₁, b₂, b₃, -4
    (∃ (b₁ b₃ : ℝ), b₁^2 = b₂ ∧ b₁ != 0 ∧ -4 * b₁^4 = b₂ → -1 * b₁ = b₃) →
    -- Goal: Calculate and prove the value
    (a₁ * a₂ / b₂) = -5 :=
by {
  sorry
}

end arithmetic_geometric_sequence_l1310_131085


namespace quotient_division_l1310_131067

/-- Definition of the condition that when 14 is divided by 3, the remainder is 2 --/
def division_property : Prop :=
  14 = 3 * (14 / 3) + 2

/-- Statement for finding the quotient when 14 is divided by 3 --/
theorem quotient_division (A : ℕ) (h : 14 = 3 * A + 2) : A = 4 :=
by
  have rem_2 := division_property
  sorry

end quotient_division_l1310_131067


namespace min_remainder_n_div_2005_l1310_131054

theorem min_remainder_n_div_2005 (n : ℕ) (hn_pos : 0 < n) 
  (h1 : n % 902 = 602) (h2 : n % 802 = 502) (h3 : n % 702 = 402) :
  n % 2005 = 101 :=
sorry

end min_remainder_n_div_2005_l1310_131054


namespace round_to_nearest_whole_l1310_131083

theorem round_to_nearest_whole (x : ℝ) (hx : x = 7643.498201) : Int.floor (x + 0.5) = 7643 := 
by
  -- To prove
  sorry

end round_to_nearest_whole_l1310_131083


namespace monotonic_increasing_on_interval_min_value_on_interval_max_value_on_interval_l1310_131021

noncomputable def f (x : ℝ) : ℝ := 1 - (3 / (x + 2))

theorem monotonic_increasing_on_interval :
  ∀ (x₁ x₂ : ℝ), 3 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 5 → f x₁ < f x₂ := sorry

theorem min_value_on_interval :
  ∃ (x : ℝ), x = 3 ∧ f x = 2 / 5 := sorry

theorem max_value_on_interval :
  ∃ (x : ℝ), x = 5 ∧ f x = 4 / 7 := sorry

end monotonic_increasing_on_interval_min_value_on_interval_max_value_on_interval_l1310_131021


namespace triangle_angle_sum_l1310_131073

theorem triangle_angle_sum {A B C : Type} 
  (angle_ABC : ℝ) (angle_BAC : ℝ) (angle_BCA : ℝ) (x : ℝ) 
  (h1: angle_ABC = 90) 
  (h2: angle_BAC = 3 * x) 
  (h3: angle_BCA = x + 10)
  : x = 20 :=
by
  sorry

end triangle_angle_sum_l1310_131073


namespace problem_statement_l1310_131070

variable (a b : ℝ)

open Real

noncomputable def inequality_holds (a b : ℝ) : Prop :=
  0 ≤ a ∧ 0 ≤ b ∧ a + b < 2 → (1 / (1 + a^2)) + (1 / (1 + b^2)) ≤ 2 / (1 + a * b)

noncomputable def equality_condition (a b : ℝ) : Prop :=
  0 ≤ a ∧ 0 ≤ b ∧ a + b < 2 → ((1 / (1 + a^2)) + (1 / (1 + b^2)) = 2 / (1 + a * b) ↔ a = b)

theorem problem_statement (a b : ℝ) : inequality_holds a b ∧ equality_condition a b :=
by
  sorry

end problem_statement_l1310_131070


namespace functional_linear_solution_l1310_131033

variable (f : ℝ → ℝ)

theorem functional_linear_solution (h : ∀ x y : ℝ, f (x^2 - y^2) = (x - y) * (f x + f y)) : 
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x :=
sorry

end functional_linear_solution_l1310_131033


namespace eval_expression_l1310_131046

-- Define the given expression
def given_expression : ℤ := -( (16 / 2) * 12 - 75 + 4 * (2 * 5) + 25 )

-- State the desired result in a theorem
theorem eval_expression : given_expression = -86 := by
  -- Skipping the proof as per instructions
  sorry

end eval_expression_l1310_131046


namespace calculate_initial_income_l1310_131077

noncomputable def initial_income : Float := 151173.52

theorem calculate_initial_income :
  let I := initial_income
  let children_distribution := 0.30 * I
  let eldest_child_share := (children_distribution / 6) + 0.05 * I
  let remaining_for_wife := 0.40 * I
  let remaining_after_distribution := I - (children_distribution + remaining_for_wife)
  let donation_to_orphanage := 0.10 * remaining_after_distribution
  let remaining_after_donation := remaining_after_distribution - donation_to_orphanage
  let federal_tax := 0.02 * remaining_after_donation
  let final_amount := remaining_after_donation - federal_tax
  final_amount = 40000 :=
by
  sorry

end calculate_initial_income_l1310_131077


namespace funct_eq_x_l1310_131003

theorem funct_eq_x (f : ℝ → ℝ) (h : ∀ x y : ℝ, f (x^4 + 4 * y^4) = f (x^2)^2 + 4 * y^3 * f y) : ∀ x : ℝ, f x = x := 
by 
  sorry

end funct_eq_x_l1310_131003


namespace long_sleeve_shirts_correct_l1310_131017

def total_shirts : ℕ := 9
def short_sleeve_shirts : ℕ := 4
def long_sleeve_shirts : ℕ := total_shirts - short_sleeve_shirts

theorem long_sleeve_shirts_correct : long_sleeve_shirts = 5 := by
  sorry

end long_sleeve_shirts_correct_l1310_131017


namespace find_pairs_l1310_131002

theorem find_pairs (x y : ℕ) (hx_pos : x > 0) (hy_pos : y > 0) :
  y ∣ x^2 + 1 ∧ x^2 ∣ y^3 + 1 ↔ (x = 1 ∧ y = 1) ∨ (x = 1 ∧ y = 2) ∨ (x = 3 ∧ y = 2) :=
by
  sorry

end find_pairs_l1310_131002


namespace complex_problem_l1310_131056

theorem complex_problem 
  (a : ℝ) 
  (ha : a^2 - 9 = 0) :
  (a + (Complex.I ^ 19)) / (1 + Complex.I) = 1 - 2 * Complex.I := by
  sorry

end complex_problem_l1310_131056


namespace angela_problems_l1310_131068

theorem angela_problems (M J S K A : ℕ) :
  M = 3 →
  J = (M * M - 5) + ((M * M - 5) / 3) →
  S = 50 / 10 →
  K = (J + S) / 2 →
  A = 50 - (M + J + S + K) →
  A = 32 :=
by
  intros hM hJ hS hK hA
  sorry

end angela_problems_l1310_131068


namespace sum_of_first_6033_terms_l1310_131075

noncomputable def geometric_series_sum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

theorem sum_of_first_6033_terms
  (a r : ℝ)  
  (h1 : geometric_series_sum a r 2011 = 200)
  (h2 : geometric_series_sum a r 4022 = 380) :
  geometric_series_sum a r 6033 = 542 := 
sorry

end sum_of_first_6033_terms_l1310_131075


namespace min_value_of_expr_l1310_131018

theorem min_value_of_expr (n : ℕ) (hn : n > 0) : (n / 3) + (27 / n) = 6 :=
by
  sorry

end min_value_of_expr_l1310_131018


namespace transform_parabola_l1310_131028

theorem transform_parabola (a b c : ℝ) (h : a ≠ 0) :
  ∃ (f : ℝ → ℝ), (∀ x, f (a * x^2 + b * x + c) = x^2) :=
sorry

end transform_parabola_l1310_131028


namespace gcd_9011_4403_l1310_131049

theorem gcd_9011_4403 : Nat.gcd 9011 4403 = 1 := 
by sorry

end gcd_9011_4403_l1310_131049


namespace inequality_x_y_z_l1310_131069

-- Definitions for the variables
variables {x y z : ℝ} 
variable {n : ℕ}

-- Positive numbers and summation condition
axiom h1 : 0 < x ∧ 0 < y ∧ 0 < z
axiom h2 : x + y + z = 1

-- The theorem to be proven
theorem inequality_x_y_z (h1 : 0 < x ∧ 0 < y ∧ 0 < z) (h2 : x + y + z = 1) (hn : n > 0) : 
  x^n + y^n + z^n ≥ (1 : ℝ) / (3:ℝ)^(n-1) :=
sorry

end inequality_x_y_z_l1310_131069


namespace train_passes_man_in_4_4_seconds_l1310_131060

noncomputable def train_speed_kmph : ℝ := 84
noncomputable def man_speed_kmph : ℝ := 6
noncomputable def train_length_m : ℝ := 110

noncomputable def kmph_to_mps (speed_kmph : ℝ) : ℝ :=
  speed_kmph * 1000 / 3600

noncomputable def train_speed_mps : ℝ :=
  kmph_to_mps train_speed_kmph

noncomputable def man_speed_mps : ℝ :=
  kmph_to_mps man_speed_kmph

noncomputable def relative_speed_mps : ℝ :=
  train_speed_mps + man_speed_mps

noncomputable def passing_time : ℝ :=
  train_length_m / relative_speed_mps

theorem train_passes_man_in_4_4_seconds :
  passing_time = 4.4 :=
by
  sorry -- Proof not required, skipping the proof logic

end train_passes_man_in_4_4_seconds_l1310_131060


namespace birds_more_than_storks_l1310_131080

theorem birds_more_than_storks :
  let birds := 6
  let initial_storks := 3
  let additional_storks := 2
  let total_storks := initial_storks + additional_storks
  birds - total_storks = 1 := by
  sorry

end birds_more_than_storks_l1310_131080


namespace average_age_constant_l1310_131052

theorem average_age_constant 
  (average_age_3_years_ago : ℕ) 
  (number_of_members_3_years_ago : ℕ) 
  (baby_age_today : ℕ) 
  (number_of_members_today : ℕ) 
  (H1 : average_age_3_years_ago = 17) 
  (H2 : number_of_members_3_years_ago = 5) 
  (H3 : baby_age_today = 2) 
  (H4 : number_of_members_today = 6) : 
  average_age_3_years_ago = (average_age_3_years_ago * number_of_members_3_years_ago + baby_age_today + 3 * number_of_members_3_years_ago) / number_of_members_today := 
by sorry

end average_age_constant_l1310_131052


namespace compare_store_costs_l1310_131027

-- Define the conditions mathematically
def StoreA_cost (x : ℕ) : ℝ := 5 * x + 125
def StoreB_cost (x : ℕ) : ℝ := 4.5 * x + 135

theorem compare_store_costs (x : ℕ) (h : x ≥ 5) : 
  5 * 15 + 125 = 200 ∧ 4.5 * 15 + 135 = 202.5 ∧ 200 < 202.5 := 
by
  -- Here the theorem states the claims to be proved.
  sorry

end compare_store_costs_l1310_131027


namespace wire_cut_l1310_131041

theorem wire_cut (total_length : ℝ) (ratio : ℝ) (shorter longer : ℝ) (h_total : total_length = 21) (h_ratio : ratio = 2/5)
  (h_shorter : longer = (5/2) * shorter) (h_sum : total_length = shorter + longer) : shorter = 6 := 
by
  -- total_length = 21, ratio = 2/5, longer = (5/2) * shorter, total_length = shorter + longer, prove shorter = 6
  sorry

end wire_cut_l1310_131041


namespace half_angle_third_quadrant_l1310_131000

theorem half_angle_third_quadrant (α : ℝ) (k : ℤ) (h1 : k * 360 + 180 < α) (h2 : α < k * 360 + 270) : 
  (∃ n : ℤ, n * 360 + 90 < (α / 2) ∧ (α / 2) < n * 360 + 135) ∨ 
  (∃ n : ℤ, n * 360 + 270 < (α / 2) ∧ (α / 2) < n * 360 + 315) := 
sorry

end half_angle_third_quadrant_l1310_131000
