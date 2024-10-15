import Mathlib

namespace NUMINAMATH_GPT_ratio_of_erasers_l119_11947

theorem ratio_of_erasers (a n : ℕ) (ha : a = 4) (hn : n = a + 12) :
  n / a = 4 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_erasers_l119_11947


namespace NUMINAMATH_GPT_time_2517_hours_from_now_l119_11915

-- Define the initial time and the function to calculate time after certain hours on a 12-hour clock
def current_time := 3
def hours := 2517

noncomputable def final_time_mod_12 (current_time : ℕ) (hours : ℕ) : ℕ :=
  (current_time + (hours % 12)) % 12

theorem time_2517_hours_from_now :
  final_time_mod_12 current_time hours = 12 :=
by
  sorry

end NUMINAMATH_GPT_time_2517_hours_from_now_l119_11915


namespace NUMINAMATH_GPT_biased_die_probability_l119_11963

theorem biased_die_probability (P2 : ℝ) (h1 : P2 ≠ 1 / 6) (h2 : 3 * P2 * (1 - P2) ^ 2 = 1 / 4) : 
  P2 = 0.211 :=
sorry

end NUMINAMATH_GPT_biased_die_probability_l119_11963


namespace NUMINAMATH_GPT_p_necessary_not_sufficient_q_l119_11989

-- Define the conditions p and q
def p (a : ℝ) : Prop := a < 1
def q (a : ℝ) : Prop := 0 < a ∧ a < 1

-- State the necessary but not sufficient condition theorem
theorem p_necessary_not_sufficient_q (a : ℝ) : p a → q a → p a ∧ ¬∀ (a : ℝ), p a → q a :=
by
  sorry

end NUMINAMATH_GPT_p_necessary_not_sufficient_q_l119_11989


namespace NUMINAMATH_GPT_carlos_class_number_l119_11946

theorem carlos_class_number (b : ℕ) :
  (100 < b ∧ b < 200) ∧
  (b + 2) % 4 = 0 ∧
  (b + 3) % 5 = 0 ∧
  (b + 4) % 6 = 0 →
  b = 122 ∨ b = 182 :=
by
  -- The proof implementation goes here
  sorry

end NUMINAMATH_GPT_carlos_class_number_l119_11946


namespace NUMINAMATH_GPT_skateboard_total_distance_l119_11960

theorem skateboard_total_distance :
  let a_1 := 8
  let d := 6
  let n := 40
  let distance (m : ℕ) := a_1 + (m - 1) * d
  let S_n := n / 2 * (distance 1 + distance n)
  S_n = 5000 := by
  sorry

end NUMINAMATH_GPT_skateboard_total_distance_l119_11960


namespace NUMINAMATH_GPT_fraction_of_profit_b_received_l119_11990

theorem fraction_of_profit_b_received (capital months_a_share months_b_share : ℝ) 
  (hA_contrib : capital * (1/4) * months_a_share = capital * (15/4))
  (hB_contrib : capital * (3/4) * months_b_share = capital * (30/4)) :
  (30/45) = (2/3) :=
by sorry

end NUMINAMATH_GPT_fraction_of_profit_b_received_l119_11990


namespace NUMINAMATH_GPT_correct_statement_exam_l119_11978

theorem correct_statement_exam 
  (students_participated : ℕ)
  (students_sampled : ℕ)
  (statement1 : Bool)
  (statement2 : Bool)
  (statement3 : Bool)
  (statement4 : Bool)
  (cond1 : students_participated = 70000)
  (cond2 : students_sampled = 1000)
  (cond3 : statement1 = False)
  (cond4 : statement2 = False)
  (cond5 : statement3 = False)
  (cond6 : statement4 = True) :
  statement4 = True := 
sorry

end NUMINAMATH_GPT_correct_statement_exam_l119_11978


namespace NUMINAMATH_GPT_scientific_notation_l119_11920

theorem scientific_notation (h : 0.000000007 = 7 * 10^(-9)) : 0.000000007 = 7 * 10^(-9) :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_l119_11920


namespace NUMINAMATH_GPT_yerema_can_pay_exactly_l119_11969

theorem yerema_can_pay_exactly (t k b m : ℤ) 
    (h_foma : 3 * t + 4 * k + 5 * b = 11 * m) : 
    ∃ n : ℤ, 9 * t + k + 4 * b = 11 * n := 
by 
    sorry

end NUMINAMATH_GPT_yerema_can_pay_exactly_l119_11969


namespace NUMINAMATH_GPT_speed_of_stream_l119_11961

-- Conditions
variables (b s : ℝ)

-- Downstream and upstream conditions
def downstream_speed := 150 = (b + s) * 5
def upstream_speed := 75 = (b - s) * 7

-- Goal statement
theorem speed_of_stream (h1 : downstream_speed b s) (h2 : upstream_speed b s) : s = 135/14 :=
by sorry

end NUMINAMATH_GPT_speed_of_stream_l119_11961


namespace NUMINAMATH_GPT_gcd_of_ratio_and_lcm_l119_11955

theorem gcd_of_ratio_and_lcm (A B : ℕ) (k : ℕ) (hA : A = 5 * k) (hB : B = 6 * k) (hlcm : Nat.lcm A B = 180) : Nat.gcd A B = 6 :=
by
  sorry

end NUMINAMATH_GPT_gcd_of_ratio_and_lcm_l119_11955


namespace NUMINAMATH_GPT_fruit_seller_profit_l119_11996

theorem fruit_seller_profit 
  (SP : ℝ) (Loss_Percentage : ℝ) (New_SP : ℝ) (Profit_Percentage : ℝ) 
  (h1: SP = 8) 
  (h2: Loss_Percentage = 20) 
  (h3: New_SP = 10.5) 
  (h4: Profit_Percentage = 5) :
  ((New_SP - (SP / (1 - (Loss_Percentage / 100.0))) / (SP / (1 - (Loss_Percentage / 100.0)))) * 100) = Profit_Percentage := 
sorry

end NUMINAMATH_GPT_fruit_seller_profit_l119_11996


namespace NUMINAMATH_GPT_smallest_possible_value_of_N_l119_11987

noncomputable def smallest_N (N : ℕ) : Prop :=
  ∃ l m n : ℕ, l * m * n = N ∧ (l - 1) * (m - 1) * (n - 1) = 378

theorem smallest_possible_value_of_N : smallest_N 560 :=
  by {
    sorry
  }

end NUMINAMATH_GPT_smallest_possible_value_of_N_l119_11987


namespace NUMINAMATH_GPT_probability_at_least_one_defective_item_l119_11936

def total_products : ℕ := 10
def defective_items : ℕ := 3
def selected_items : ℕ := 3
noncomputable def comb (n k : ℕ) : ℕ := Nat.choose n k

theorem probability_at_least_one_defective_item :
    let total_combinations := comb total_products selected_items
    let non_defective_combinations := comb (total_products - defective_items) selected_items
    let opposite_probability := (non_defective_combinations : ℚ) / (total_combinations : ℚ)
    let probability := 1 - opposite_probability
    probability = 17 / 24 :=
by
  sorry

end NUMINAMATH_GPT_probability_at_least_one_defective_item_l119_11936


namespace NUMINAMATH_GPT_problem_statement_l119_11953

noncomputable def universal_set : Set ℤ := {x : ℤ | x^2 - 5*x - 6 < 0 }

def A : Set ℤ := {x : ℤ | -1 < x ∧ x ≤ 2 }

def B : Set ℤ := {2, 3, 5}

def complement_U_A : Set ℤ := {x : ℤ | x ∈ universal_set ∧ ¬(x ∈ A)}

theorem problem_statement : 
  (complement_U_A ∩ B) = {3, 5} :=
by 
  sorry

end NUMINAMATH_GPT_problem_statement_l119_11953


namespace NUMINAMATH_GPT_badminton_players_l119_11966

theorem badminton_players (B T N Both Total: ℕ) 
  (h1: Total = 35)
  (h2: T = 18)
  (h3: N = 5)
  (h4: Both = 3)
  : B = 15 :=
by
  -- The proof block is intentionally left out.
  sorry

end NUMINAMATH_GPT_badminton_players_l119_11966


namespace NUMINAMATH_GPT_distance_to_destination_l119_11928

theorem distance_to_destination 
  (speed : ℝ) (time : ℝ) 
  (h_speed : speed = 100) 
  (h_time : time = 5) : 
  speed * time = 500 :=
by
  rw [h_speed, h_time]
  -- This simplifies to 100 * 5 = 500
  norm_num

end NUMINAMATH_GPT_distance_to_destination_l119_11928


namespace NUMINAMATH_GPT_intersection_A_B_l119_11931

def A := {x : ℝ | 2 < x ∧ x < 4}
def B := {x : ℝ | (x-1) * (x-3) < 0}

theorem intersection_A_B : A ∩ B = {x : ℝ | 2 < x ∧ x < 3} :=
sorry

end NUMINAMATH_GPT_intersection_A_B_l119_11931


namespace NUMINAMATH_GPT_minimum_value_fraction_l119_11970

theorem minimum_value_fraction (m n : ℝ) (h0 : 0 ≤ m) (h1 : 0 ≤ n) (h2 : m + n = 1) :
  ∃ min_val, min_val = (1 / 4) ∧ (∀ m n, 0 ≤ m → 0 ≤ n → m + n = 1 → (m^2) / (m + 2) + (n^2) / (n + 1) ≥ min_val) :=
sorry

end NUMINAMATH_GPT_minimum_value_fraction_l119_11970


namespace NUMINAMATH_GPT_train_speed_l119_11919

theorem train_speed (length_train length_platform : ℝ) (time : ℝ) 
  (h_length_train : length_train = 170.0416) 
  (h_length_platform : length_platform = 350) 
  (h_time : time = 26) : 
  (length_train + length_platform) / time * 3.6 = 72 :=
by 
  sorry

end NUMINAMATH_GPT_train_speed_l119_11919


namespace NUMINAMATH_GPT_cows_count_l119_11957

theorem cows_count (initial_cows last_year_deaths last_year_sales this_year_increase purchases gifts : ℕ)
  (h1 : initial_cows = 39)
  (h2 : last_year_deaths = 25)
  (h3 : last_year_sales = 6)
  (h4 : this_year_increase = 24)
  (h5 : purchases = 43)
  (h6 : gifts = 8) : 
  initial_cows - last_year_deaths - last_year_sales + this_year_increase + purchases + gifts = 83 := by
  sorry

end NUMINAMATH_GPT_cows_count_l119_11957


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l119_11968

def M : Set ℝ := {x | 0 < x ∧ x ≤ 3}
def N : Set ℝ := {x | 0 < x ∧ x ≤ 2}

theorem necessary_but_not_sufficient_condition (a : ℝ) : (a ∈ M → a ∈ N) ∧ ¬(a ∈ N → a ∈ M) := 
  by 
    sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l119_11968


namespace NUMINAMATH_GPT_max_value_of_sum_l119_11918

theorem max_value_of_sum 
  (a b c : ℝ) 
  (h : a^2 + 2 * b^2 + 3 * c^2 = 6) : 
  a + b + c ≤ Real.sqrt 11 := 
by 
  sorry

end NUMINAMATH_GPT_max_value_of_sum_l119_11918


namespace NUMINAMATH_GPT_angle_APB_l119_11982

-- Define the problem conditions
variables (XY : Π X Y : ℝ, XY = X + Y) -- Line XY is a straight line
          (semicircle_XAZ : Π X A Z : ℝ, semicircle_XAZ = X + Z - A) -- Semicircle XAZ
          (semicircle_ZBY : Π Z B Y : ℝ, semicircle_ZBY = Z + Y - B) -- Semicircle ZBY
          (PA_tangent_XAZ_at_A : Π P A X Z : ℝ, PA_tangent_XAZ_at_A = P + A + X - Z) -- PA tangent to XAZ at A
          (PB_tangent_ZBY_at_B : Π P B Z Y : ℝ, PB_tangent_ZBY_at_B = P + B + Z - Y) -- PB tangent to ZBY at B
          (arc_XA : ℝ := 45) -- Arc XA is 45 degrees
          (arc_BY : ℝ := 60) -- Arc BY is 60 degrees

-- Main theorem to prove
theorem angle_APB : ∀ P A B: ℝ, 
  540 - 90 - 135 - 120 - 90 = 105 := by 
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_angle_APB_l119_11982


namespace NUMINAMATH_GPT_sum_coordinates_B_l119_11902

noncomputable def A : (ℝ × ℝ) := (0, 0)
noncomputable def B (x : ℝ) : (ℝ × ℝ) := (x, 4)

theorem sum_coordinates_B 
  (x : ℝ) 
  (h_slope : (4 - 0)/(x - 0) = 3/4) : x + 4 = 28 / 3 := by
sorry

end NUMINAMATH_GPT_sum_coordinates_B_l119_11902


namespace NUMINAMATH_GPT_city_rentals_cost_per_mile_l119_11999

-- The parameters provided in the problem
def safety_base_rate : ℝ := 21.95
def safety_per_mile_rate : ℝ := 0.19
def city_base_rate : ℝ := 18.95
def miles_driven : ℝ := 150.0

-- The cost expressions based on the conditions
def safety_total_cost (miles: ℝ) : ℝ := safety_base_rate + safety_per_mile_rate * miles
def city_total_cost (miles: ℝ) (city_per_mile_rate: ℝ) : ℝ := city_base_rate + city_per_mile_rate * miles

-- The cost equality condition for 150 miles
def cost_condition : Prop :=
  safety_total_cost miles_driven = city_total_cost miles_driven 0.21

-- Prove that the cost per mile for City Rentals is 0.21 dollars
theorem city_rentals_cost_per_mile : cost_condition :=
by
  -- Start the proof
  sorry

end NUMINAMATH_GPT_city_rentals_cost_per_mile_l119_11999


namespace NUMINAMATH_GPT_problem_1_max_value_problem_2_good_sets_count_l119_11965

noncomputable def goodSetMaxValue : ℤ :=
  2012

noncomputable def goodSetCount : ℤ :=
  1006

theorem problem_1_max_value {M : Set ℤ} (hM : ∀ x, x ∈ M ↔ |x| ≤ 2014) :
  ∀ a b c : ℤ, (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) →
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  (1 / a + 1 / b = 2 / c) →
  (a + c = 2 * b) →
  a ∈ M ∧ b ∈ M ∧ c ∈ M →
  ∃ P : Set ℤ, P = {a, b, c} ∧ a ∈ P ∧ b ∈ P ∧ c ∈ P ∧
  goodSetMaxValue = 2012 :=
sorry

theorem problem_2_good_sets_count {M : Set ℤ} (hM : ∀ x, x ∈ M ↔ |x| ≤ 2014) :
  ∀ a b c : ℤ, (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) →
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  (1 / a + 1 / b = 2 / c) →
  (a + c = 2 * b) →
  a ∈ M ∧ b ∈ M ∧ c ∈ M →
  ∃ P : Set ℤ, P = {a, b, c} ∧ a ∈ P ∧ b ∈ P ∧ c ∈ P ∧
  goodSetCount = 1006 :=
sorry

end NUMINAMATH_GPT_problem_1_max_value_problem_2_good_sets_count_l119_11965


namespace NUMINAMATH_GPT_instantaneous_rate_of_change_at_x1_l119_11998

open Real

noncomputable def f (x : ℝ) : ℝ := (1/3)*x^3 - x^2 + 8

theorem instantaneous_rate_of_change_at_x1 : deriv f 1 = -1 := by
  sorry

end NUMINAMATH_GPT_instantaneous_rate_of_change_at_x1_l119_11998


namespace NUMINAMATH_GPT_recommendation_plans_count_l119_11975

def num_male : ℕ := 3
def num_female : ℕ := 2
def num_recommendations : ℕ := 5

def num_spots_russian : ℕ := 2
def num_spots_japanese : ℕ := 2
def num_spots_spanish : ℕ := 1

def condition_russian (males : ℕ) : Prop := males > 0
def condition_japanese (males : ℕ) : Prop := males > 0

theorem recommendation_plans_count : 
  (∃ (males_r : ℕ) (males_j : ℕ), condition_russian males_r ∧ condition_japanese males_j ∧ 
  num_male - males_r - males_j >= 0 ∧ males_r + males_j ≤ num_male ∧ 
  num_female + (num_male - males_r - males_j) >= num_recommendations - (num_spots_russian + num_spots_japanese + num_spots_spanish)) →
  (∃ (x : ℕ), x = 24) := by
  sorry

end NUMINAMATH_GPT_recommendation_plans_count_l119_11975


namespace NUMINAMATH_GPT_option_D_is_greater_than_reciprocal_l119_11933

theorem option_D_is_greater_than_reciprocal:
  ∀ (x : ℚ), (x = 2) → x > 1/x := by
  intro x
  intro hx
  rw [hx]
  norm_num

end NUMINAMATH_GPT_option_D_is_greater_than_reciprocal_l119_11933


namespace NUMINAMATH_GPT_adil_older_than_bav_by_732_days_l119_11974

-- Definitions based on the problem conditions
def adilBirthDate : String := "December 31, 2015"
def bavBirthDate : String := "January 1, 2018"

-- Main theorem statement 
theorem adil_older_than_bav_by_732_days :
    let daysIn2016 := 366
    let daysIn2017 := 365
    let transition := 1
    let totalDays := daysIn2016 + daysIn2017 + transition
    totalDays = 732 :=
by
    sorry

end NUMINAMATH_GPT_adil_older_than_bav_by_732_days_l119_11974


namespace NUMINAMATH_GPT_solve_for_x_l119_11926

theorem solve_for_x (x : ℝ) (h : (x^2 + 4*x - 5)^0 = 1) : x^2 - 5*x + 5 = 1 → x = 4 := 
by
  intro h2
  have : ∀ x, (x^2 + 4*x - 5 = 0) ↔ false := sorry
  exact sorry

end NUMINAMATH_GPT_solve_for_x_l119_11926


namespace NUMINAMATH_GPT_triangle_side_range_l119_11949

theorem triangle_side_range (a : ℝ) :
  1 < a ∧ a < 4 ↔ 3 + (2 * a - 1) > 4 ∧ 3 + 4 > 2 * a - 1 ∧ 4 + (2 * a - 1) > 3 :=
by
  sorry

end NUMINAMATH_GPT_triangle_side_range_l119_11949


namespace NUMINAMATH_GPT_makeup_palette_cost_l119_11935

variable (lipstick_cost : ℝ := 2.5)
variable (num_lipsticks : ℕ := 4)
variable (hair_color_cost : ℝ := 4)
variable (num_boxes_hair_color : ℕ := 3)
variable (total_cost : ℝ := 67)
variable (num_palettes : ℕ := 3)

theorem makeup_palette_cost :
  (total_cost - (num_lipsticks * lipstick_cost + num_boxes_hair_color * hair_color_cost)) / num_palettes = 15 := 
by
  sorry

end NUMINAMATH_GPT_makeup_palette_cost_l119_11935


namespace NUMINAMATH_GPT_fraction_burritos_given_away_l119_11907

noncomputable def total_burritos_bought : Nat := 3 * 20
noncomputable def burritos_eaten : Nat := 3 * 10
noncomputable def burritos_left : Nat := 10
noncomputable def burritos_before_eating : Nat := burritos_eaten + burritos_left
noncomputable def burritos_given_away : Nat := total_burritos_bought - burritos_before_eating

theorem fraction_burritos_given_away : (burritos_given_away : ℚ) / total_burritos_bought = 1 / 3 := by
  sorry

end NUMINAMATH_GPT_fraction_burritos_given_away_l119_11907


namespace NUMINAMATH_GPT_binomial_coefficient_times_two_l119_11914

theorem binomial_coefficient_times_two : 2 * Nat.choose 8 5 = 112 := 
by 
  -- The proof is omitted here
  sorry

end NUMINAMATH_GPT_binomial_coefficient_times_two_l119_11914


namespace NUMINAMATH_GPT_cos_135_eq_neg_inv_sqrt_2_l119_11900

theorem cos_135_eq_neg_inv_sqrt_2 :
  Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_cos_135_eq_neg_inv_sqrt_2_l119_11900


namespace NUMINAMATH_GPT_gcd_eight_digit_repeating_four_digit_l119_11945

theorem gcd_eight_digit_repeating_four_digit :
  ∀ n : ℕ, (1000 ≤ n ∧ n < 10000) → (∀ m : ℕ, (1000 ≤ m ∧ m < 10000) →
  Nat.gcd (10001 * n) (10001 * m) = 10001) :=
by
  intros n hn m hm
  sorry

end NUMINAMATH_GPT_gcd_eight_digit_repeating_four_digit_l119_11945


namespace NUMINAMATH_GPT_inequality_proof_l119_11977

noncomputable def a : ℝ := (1 / 2) * Real.cos (6 * Real.pi / 180) - (Real.sqrt 3 / 2) * Real.sin (6 * Real.pi / 180)
noncomputable def b : ℝ := (2 * Real.tan (13 * Real.pi / 180)) / (1 - (Real.tan (13 * Real.pi / 180))^2)
noncomputable def c : ℝ := Real.sqrt ((1 - Real.cos (50 * Real.pi / 180)) / 2)

theorem inequality_proof : a < c ∧ c < b := by
  sorry

end NUMINAMATH_GPT_inequality_proof_l119_11977


namespace NUMINAMATH_GPT_probability_same_color_correct_l119_11929

-- Defining the contents of Bag A and Bag B
def bagA : List (String × ℕ) := [("white", 1), ("red", 2), ("black", 3)]
def bagB : List (String × ℕ) := [("white", 2), ("red", 3), ("black", 1)]

-- The probability calculation
noncomputable def probability_same_color (bagA bagB : List (String × ℕ)) : ℚ :=
  let p_white := (1 / 6 : ℚ) * (1 / 3 : ℚ)
  let p_red := (1 / 3 : ℚ) * (1 / 2 : ℚ)
  let p_black := (1 / 2 : ℚ) * (1 / 6 : ℚ)
  p_white + p_red + p_black

-- Proof problem statement
theorem probability_same_color_correct :
  probability_same_color bagA bagB = 11 / 36 := 
by 
  sorry

end NUMINAMATH_GPT_probability_same_color_correct_l119_11929


namespace NUMINAMATH_GPT_quadratic_equation_in_x_l119_11979

theorem quadratic_equation_in_x (k x : ℝ) : 
  (k^2 + 1) * x^2 - (k * x - 8) - 1 = 0 := 
sorry

end NUMINAMATH_GPT_quadratic_equation_in_x_l119_11979


namespace NUMINAMATH_GPT_sequence_sum_l119_11991

theorem sequence_sum (A B C D E F G H : ℕ) (hC : C = 7) 
    (h_sum : A + B + C = 36 ∧ B + C + D = 36 ∧ C + D + E = 36 ∧ D + E + F = 36 ∧ E + F + G = 36 ∧ F + G + H = 36) :
    A + H = 29 :=
sorry

end NUMINAMATH_GPT_sequence_sum_l119_11991


namespace NUMINAMATH_GPT_parallel_lines_m_value_l119_11916

noncomputable def m_value_parallel (m : ℝ) : Prop :=
  (m-1) / 2 = 1 / -3

theorem parallel_lines_m_value :
  ∀ (m : ℝ), (m_value_parallel m) → m = 1 / 3 :=
by
  intro m
  intro h
  sorry

end NUMINAMATH_GPT_parallel_lines_m_value_l119_11916


namespace NUMINAMATH_GPT_sandy_puppies_l119_11973

theorem sandy_puppies :
  ∀ (initial_puppies puppies_given_away remaining_puppies : ℕ),
  initial_puppies = 8 →
  puppies_given_away = 4 →
  remaining_puppies = initial_puppies - puppies_given_away →
  remaining_puppies = 4 :=
by
  intros initial_puppies puppies_given_away remaining_puppies
  intro h_initial
  intro h_given_away
  intro h_remaining
  rw [h_initial, h_given_away] at h_remaining
  exact h_remaining

end NUMINAMATH_GPT_sandy_puppies_l119_11973


namespace NUMINAMATH_GPT_stratified_sampling_l119_11972

noncomputable def employees := 500
noncomputable def under_35 := 125
noncomputable def between_35_and_49 := 280
noncomputable def over_50 := 95
noncomputable def sample_size := 100

theorem stratified_sampling : 
  under_35 * sample_size / employees = 25 := by
  sorry

end NUMINAMATH_GPT_stratified_sampling_l119_11972


namespace NUMINAMATH_GPT_car_speed_l119_11997

theorem car_speed (distance time : ℝ) (h1 : distance = 300) (h2 : time = 5) : distance / time = 60 := by
  have h : distance / time = 300 / 5 := by
    rw [h1, h2]
  norm_num at h
  exact h

end NUMINAMATH_GPT_car_speed_l119_11997


namespace NUMINAMATH_GPT_average_percentage_decrease_l119_11956

theorem average_percentage_decrease (p1 p2 : ℝ) (n : ℕ) (h₀ : p1 = 2000) (h₁ : p2 = 1280) (h₂ : n = 2) :
  ((p1 - p2) / p1 * 100) / n = 18 := 
by
  sorry

end NUMINAMATH_GPT_average_percentage_decrease_l119_11956


namespace NUMINAMATH_GPT_jerry_current_average_l119_11938

-- Definitions for Jerry's first 3 tests average and conditions
variable (A : ℝ)

-- Condition details
def total_score_of_first_3_tests := 3 * A
def new_desired_average := A + 2
def total_score_needed := (A + 2) * 4
def score_on_fourth_test := 93

theorem jerry_current_average :
  (total_score_needed A = total_score_of_first_3_tests A + score_on_fourth_test) → A = 85 :=
by
  sorry

end NUMINAMATH_GPT_jerry_current_average_l119_11938


namespace NUMINAMATH_GPT_polar_to_rectangular_coordinates_l119_11940

theorem polar_to_rectangular_coordinates 
  (r θ : ℝ) 
  (hr : r = 7) 
  (hθ : θ = 7 * Real.pi / 4) : 
  (r * Real.cos θ, r * Real.sin θ) = (7 * Real.sqrt 2 / 2, -7 * Real.sqrt 2 / 2) := 
by
  sorry

end NUMINAMATH_GPT_polar_to_rectangular_coordinates_l119_11940


namespace NUMINAMATH_GPT_a_n_formula_S_n_formula_T_n_formula_l119_11903

noncomputable def a_sequence (n : ℕ) : ℕ := 2 * n
noncomputable def S (n : ℕ) : ℕ := n * (n + 1)
noncomputable def b_sequence (n : ℕ) : ℕ := a_sequence (3 ^ n)
noncomputable def T (n : ℕ) : ℕ := 3^(n + 1) - 3

theorem a_n_formula :
  ∀ {n : ℕ}, a_sequence 5 = 10 ∧ S 15 = 240 → a_sequence n = 2 * n :=
sorry

theorem S_n_formula :
  ∀ {n : ℕ}, a_sequence 5 = 10 ∧ S 15 = 240 → S n = n * (n + 1) :=
sorry

theorem T_n_formula :
  ∀ {n : ℕ}, a_sequence 5 = 10 ∧ S 15 = 240 → T n = 3^(n + 1) - 3 :=
sorry

end NUMINAMATH_GPT_a_n_formula_S_n_formula_T_n_formula_l119_11903


namespace NUMINAMATH_GPT_largest_divisor_of_product_of_three_consecutive_odd_integers_l119_11950

theorem largest_divisor_of_product_of_three_consecutive_odd_integers :
  ∀ n : ℕ, n > 0 → ∃ d : ℕ, d = 3 ∧ ∀ m : ℕ, m ∣ ((2*n-1)*(2*n+1)*(2*n+3)) → m ≤ d :=
by
  sorry

end NUMINAMATH_GPT_largest_divisor_of_product_of_three_consecutive_odd_integers_l119_11950


namespace NUMINAMATH_GPT_find_side_and_area_l119_11944

-- Conditions
variables {A B C a b c : ℝ} (S : ℝ)
axiom angle_sum : A + B + C = Real.pi
axiom side_a : a = 4
axiom side_b : b = 5
axiom angle_relation : C = 2 * A

-- Proven equalities
theorem find_side_and_area :
  c = 6 ∧ S = 5 * 6 * (Real.sqrt 7) / 4 / 2 := by
  sorry

end NUMINAMATH_GPT_find_side_and_area_l119_11944


namespace NUMINAMATH_GPT_math_problem_l119_11913

theorem math_problem :
  |(-3 : ℝ)| - Real.sqrt 8 - (1/2 : ℝ)⁻¹ + 2 * Real.cos (Real.pi / 4) = 1 - Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l119_11913


namespace NUMINAMATH_GPT_probability_neither_event_l119_11948

-- Definitions of given probabilities
def P_soccer_match : ℚ := 5 / 8
def P_science_test : ℚ := 1 / 4

-- Calculations of the complements
def P_no_soccer_match : ℚ := 1 - P_soccer_match
def P_no_science_test : ℚ := 1 - P_science_test

-- Independence of events implies the probability of neither event is the product of their complements
theorem probability_neither_event :
  (P_no_soccer_match * P_no_science_test) = 9 / 32 :=
by
  sorry

end NUMINAMATH_GPT_probability_neither_event_l119_11948


namespace NUMINAMATH_GPT_barbara_current_savings_l119_11962

def wristwatch_cost : ℕ := 100
def weekly_allowance : ℕ := 5
def initial_saving_duration : ℕ := 10
def further_saving_duration : ℕ := 16

theorem barbara_current_savings : 
  -- Given:
  -- wristwatch_cost: $100
  -- weekly_allowance: $5
  -- further_saving_duration: 16 weeks
  -- Prove:
  -- Barbara currently has $20
  wristwatch_cost - weekly_allowance * further_saving_duration = 20 :=
by
  sorry

end NUMINAMATH_GPT_barbara_current_savings_l119_11962


namespace NUMINAMATH_GPT_problem_statement_l119_11995

theorem problem_statement (x : ℝ) (h : x = Real.sqrt 3 + 1) : x^2 - 2*x + 1 = 3 :=
sorry

end NUMINAMATH_GPT_problem_statement_l119_11995


namespace NUMINAMATH_GPT_solve_for_a_l119_11985

theorem solve_for_a (a x : ℝ) (h : (1 / 2) * x + a = -1) (hx : x = 2) : a = -2 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_a_l119_11985


namespace NUMINAMATH_GPT_sector_central_angle_l119_11925

theorem sector_central_angle (r l : ℝ) (α : ℝ) 
  (h1 : l + 2 * r = 12) 
  (h2 : 1 / 2 * l * r = 8) : 
  α = 1 ∨ α = 4 :=
by
  sorry

end NUMINAMATH_GPT_sector_central_angle_l119_11925


namespace NUMINAMATH_GPT_total_cost_of_office_supplies_l119_11922

-- Define the conditions
def cost_of_pencil : ℝ := 0.5
def cost_of_folder : ℝ := 0.9
def count_of_pencils : ℕ := 24
def count_of_folders : ℕ := 20

-- Define the theorem to prove
theorem total_cost_of_office_supplies
  (cop : ℝ := cost_of_pencil)
  (cof : ℝ := cost_of_folder)
  (ncp : ℕ := count_of_pencils)
  (ncg : ℕ := count_of_folders) :
  cop * ncp + cof * ncg = 30 :=
sorry

end NUMINAMATH_GPT_total_cost_of_office_supplies_l119_11922


namespace NUMINAMATH_GPT_syrup_cost_per_week_l119_11904

theorem syrup_cost_per_week (gallons_per_week : ℕ) (gallons_per_box : ℕ) (cost_per_box : ℕ) 
  (h1 : gallons_per_week = 180) 
  (h2 : gallons_per_box = 30) 
  (h3 : cost_per_box = 40) : 
  (gallons_per_week / gallons_per_box) * cost_per_box = 240 := 
by
  sorry

end NUMINAMATH_GPT_syrup_cost_per_week_l119_11904


namespace NUMINAMATH_GPT_even_and_increasing_on_0_inf_l119_11941

noncomputable def fA (x : ℝ) : ℝ := x^(2/3)
noncomputable def fB (x : ℝ) : ℝ := (1/2)^x
noncomputable def fC (x : ℝ) : ℝ := Real.log x
noncomputable def fD (x : ℝ) : ℝ := -x^2 + 1

theorem even_and_increasing_on_0_inf (f : ℝ → ℝ) : 
  (∀ x, f x = f (-x)) ∧ (∀ a b, (0 < a ∧ a < b) → f a < f b) ↔ f = fA :=
sorry

end NUMINAMATH_GPT_even_and_increasing_on_0_inf_l119_11941


namespace NUMINAMATH_GPT_kim_earrings_l119_11937

-- Define the number of pairs of earrings on the first day E as a variable
variable (E : ℕ)

-- Define the total number of gumballs Kim receives based on the earrings she brings each day
def total_gumballs_received (E : ℕ) : ℕ :=
  9 * E + 9 * 2 * E + 9 * (2 * E - 1)

-- Define the total number of gumballs Kim eats in 42 days
def total_gumballs_eaten : ℕ :=
  3 * 42

-- Define the statement to be proved
theorem kim_earrings : 
  total_gumballs_received E = total_gumballs_eaten + 9 → E = 3 :=
by sorry

end NUMINAMATH_GPT_kim_earrings_l119_11937


namespace NUMINAMATH_GPT_trigonometric_identity_l119_11923

theorem trigonometric_identity :
  Real.sin (17 * Real.pi / 180) * Real.sin (223 * Real.pi / 180) + 
  Real.sin (253 * Real.pi / 180) * Real.sin (313 * Real.pi / 180) = 1 / 2 := 
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l119_11923


namespace NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l119_11992

-- Problem 1: Prove X = 93 given X - 12 = 81
theorem problem1 (X : ℝ) (h : X - 12 = 81) : X = 93 :=
by
  sorry

-- Problem 2: Prove X = 5.4 given 5.1 + X = 10.5
theorem problem2 (X : ℝ) (h : 5.1 + X = 10.5) : X = 5.4 :=
by
  sorry

-- Problem 3: Prove X = 0.7 given 6X = 4.2
theorem problem3 (X : ℝ) (h : 6 * X = 4.2) : X = 0.7 :=
by
  sorry

-- Problem 4: Prove X = 5 given X ÷ 0.4 = 12.5
theorem problem4 (X : ℝ) (h : X / 0.4 = 12.5) : X = 5 :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l119_11992


namespace NUMINAMATH_GPT_average_age_of_9_students_l119_11954

theorem average_age_of_9_students (avg_age_17_students : ℕ)
                                   (num_students : ℕ)
                                   (avg_age_5_students : ℕ)
                                   (num_5_students : ℕ)
                                   (age_17th_student : ℕ) :
    avg_age_17_students = 17 →
    num_students = 17 →
    avg_age_5_students = 14 →
    num_5_students = 5 →
    age_17th_student = 75 →
    (144 / 9) = 16 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_average_age_of_9_students_l119_11954


namespace NUMINAMATH_GPT_sarah_toy_cars_l119_11908

theorem sarah_toy_cars (initial_money toy_car_cost scarf_cost beanie_cost remaining_money: ℕ) 
  (h_initial: initial_money = 53) 
  (h_toy_car_cost: toy_car_cost = 11) 
  (h_scarf_cost: scarf_cost = 10) 
  (h_beanie_cost: beanie_cost = 14) 
  (h_remaining: remaining_money = 7) : 
  (initial_money - remaining_money - scarf_cost - beanie_cost) / toy_car_cost = 2 := 
by 
  sorry

end NUMINAMATH_GPT_sarah_toy_cars_l119_11908


namespace NUMINAMATH_GPT_custom_operation_example_l119_11967

def custom_operation (x y : Int) : Int :=
  x * y - 3 * x

theorem custom_operation_example : (custom_operation 7 4) - (custom_operation 4 7) = -9 := by
  sorry

end NUMINAMATH_GPT_custom_operation_example_l119_11967


namespace NUMINAMATH_GPT_smallest_number_diminished_by_8_divisible_by_9_6_12_18_l119_11921

theorem smallest_number_diminished_by_8_divisible_by_9_6_12_18 :
  ∃ x : ℕ, (x - 8) % Nat.lcm (Nat.lcm 9 6) (Nat.lcm 12 18) = 0 ∧ ∀ y : ℕ, (y - 8) % Nat.lcm (Nat.lcm 9 6) (Nat.lcm 12 18) = 0 → x ≤ y → x = 44 :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_diminished_by_8_divisible_by_9_6_12_18_l119_11921


namespace NUMINAMATH_GPT_ratio_proof_l119_11980

variable (x y z : ℝ)
variable (h1 : y / z = 1 / 2)
variable (h2 : z / x = 2 / 3)
variable (h3 : x / y = 3 / 1)

theorem ratio_proof : (x / (y * z)) / (y / (z * x)) = 4 / 1 := 
  sorry

end NUMINAMATH_GPT_ratio_proof_l119_11980


namespace NUMINAMATH_GPT_findNumberOfIntegers_l119_11958

def arithmeticSeq (a d n : ℕ) : ℕ :=
  a + d * n

def isInSeq (n : ℕ) : Prop :=
  ∃ k : ℕ, k ≤ 33 ∧ n = arithmeticSeq 1 3 k

def validInterval (n : ℕ) : Bool :=
  (n + 1) / 3 % 2 = 1

theorem findNumberOfIntegers :
  ∃ count : ℕ, count = 66 ∧ (∀ n : ℕ, 1 ≤ n ∧ n ≤ 100 ∧ ¬isInSeq n → validInterval n = true) :=
sorry

end NUMINAMATH_GPT_findNumberOfIntegers_l119_11958


namespace NUMINAMATH_GPT_sufficient_condition_implies_range_l119_11959

def setA : Set ℝ := {x | 1 ≤ x ∧ x < 3}

def setB (a : ℝ) : Set ℝ := {x | x^2 - a * x ≤ x - a}

theorem sufficient_condition_implies_range (a : ℝ) :
  (∀ x, x ∉ setA → x ∉ setB a) → (1 ≤ a ∧ a < 3) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_condition_implies_range_l119_11959


namespace NUMINAMATH_GPT_students_at_table_l119_11905

def numStudents (candies : ℕ) (first_last : ℕ) (st_len : ℕ) : Prop :=
  candies - 1 = st_len * first_last

theorem students_at_table 
  (candies : ℕ)
  (first_last : ℕ)
  (st_len : ℕ)
  (h1 : candies = 120) 
  (h2 : first_last = 1) :
  (st_len = 7 ∨ st_len = 17) :=
by
  sorry

end NUMINAMATH_GPT_students_at_table_l119_11905


namespace NUMINAMATH_GPT_pyramid_area_ratio_l119_11984

theorem pyramid_area_ratio (S S1 S2 : ℝ) (h1 : S1 = (99 / 100)^2 * S) (h2 : S2 = (1 / 100)^2 * S) :
  S1 / S2 = 9801 := by
  sorry

end NUMINAMATH_GPT_pyramid_area_ratio_l119_11984


namespace NUMINAMATH_GPT_arithmetic_sequence_example_l119_11939

theorem arithmetic_sequence_example (a : ℕ → ℝ) (h : ∀ n, a n = a 1 + (n - 1) * (a 2 - a 1)) (h₁ : a 1 + a 19 = 10) : a 10 = 5 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_example_l119_11939


namespace NUMINAMATH_GPT_ratio_Sydney_to_Sherry_l119_11910

variable (Randolph_age Sydney_age Sherry_age : ℕ)

-- Conditions
axiom Randolph_older_than_Sydney : Randolph_age = Sydney_age + 5
axiom Sherry_age_is_25 : Sherry_age = 25
axiom Randolph_age_is_55 : Randolph_age = 55

-- Theorem to prove
theorem ratio_Sydney_to_Sherry : (Sydney_age : ℝ) / (Sherry_age : ℝ) = 2 := by
  sorry

end NUMINAMATH_GPT_ratio_Sydney_to_Sherry_l119_11910


namespace NUMINAMATH_GPT_find_positive_real_x_l119_11981

noncomputable def positive_solution :=
  ∃ (x : ℝ), (1/3) * (4 * x^2 - 2) = (x^2 - 75 * x - 15) * (x^2 + 50 * x + 10) ∧ x > 0

theorem find_positive_real_x :
  positive_solution ↔ ∃ (x : ℝ), x = (75 + Real.sqrt 5693) / 2 :=
by sorry

end NUMINAMATH_GPT_find_positive_real_x_l119_11981


namespace NUMINAMATH_GPT_min_value_a_plus_b_l119_11951

theorem min_value_a_plus_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 3 * a + b = 2 * a * b) : a + b ≥ 2 + Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_min_value_a_plus_b_l119_11951


namespace NUMINAMATH_GPT_max_ab_upper_bound_l119_11943

noncomputable def circle_center_coords : ℝ × ℝ :=
  let center_x := -1
  let center_y := 2
  (center_x, center_y)

noncomputable def max_ab_value (a b : ℝ) : ℝ :=
  if a = 1 - 2 * b then a * b else 0

theorem max_ab_upper_bound :
  let circle : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 + 2*p.1 - 4*p.2 + 1 = 0}
  let line_cond : ℝ × ℝ := (-1, 2)
  (circle_center_coords = line_cond) →
  (∀ a b : ℝ, max_ab_value a b ≤ 1 / 8) :=
by
  intro circle line_cond h
  -- Proof is omitted as per instruction
  sorry

end NUMINAMATH_GPT_max_ab_upper_bound_l119_11943


namespace NUMINAMATH_GPT_overtime_hourly_rate_l119_11993

theorem overtime_hourly_rate
  (hourly_rate_first_40_hours: ℝ)
  (hours_first_40: ℝ)
  (gross_pay: ℝ)
  (overtime_hours: ℝ)
  (total_pay_first_40: ℝ := hours_first_40 * hourly_rate_first_40_hours)
  (pay_overtime: ℝ := gross_pay - total_pay_first_40)
  (hourly_rate_overtime: ℝ := pay_overtime / overtime_hours)
  (h1: hourly_rate_first_40_hours = 11.25)
  (h2: hours_first_40 = 40)
  (h3: gross_pay = 622)
  (h4: overtime_hours = 10.75) :
  hourly_rate_overtime = 16 := 
by
  sorry

end NUMINAMATH_GPT_overtime_hourly_rate_l119_11993


namespace NUMINAMATH_GPT_quotient_remainder_increase_l119_11986

theorem quotient_remainder_increase (a b q r q' r' : ℕ) (hb : b ≠ 0) 
    (h1 : a = b * q + r) (h2 : 0 ≤ r) (h3 : r < b) (h4 : 3 * a = 3 * b * q' + r') 
    (h5 : 0 ≤ r') (h6 : r' < 3 * b) :
    q' = q ∧ r' = 3 * r := by
  sorry

end NUMINAMATH_GPT_quotient_remainder_increase_l119_11986


namespace NUMINAMATH_GPT_chairs_in_fifth_row_l119_11909

theorem chairs_in_fifth_row : 
  ∀ (a : ℕ → ℕ), 
    a 1 = 14 ∧ 
    a 2 = 23 ∧ 
    a 3 = 32 ∧ 
    a 4 = 41 ∧ 
    a 6 = 59 ∧ 
    (∀ n, a (n + 1) = a n + 9) → 
  a 5 = 50 :=
by
  sorry

end NUMINAMATH_GPT_chairs_in_fifth_row_l119_11909


namespace NUMINAMATH_GPT_fishing_rod_price_l119_11942

theorem fishing_rod_price (initial_price : ℝ) 
  (price_increase_percentage : ℝ) 
  (price_decrease_percentage : ℝ) 
  (new_price : ℝ) 
  (final_price : ℝ) 
  (h1 : initial_price = 50) 
  (h2 : price_increase_percentage = 0.20) 
  (h3 : price_decrease_percentage = 0.15) 
  (h4 : new_price = initial_price * (1 + price_increase_percentage)) 
  (h5 : final_price = new_price * (1 - price_decrease_percentage)) 
  : final_price = 51 :=
sorry

end NUMINAMATH_GPT_fishing_rod_price_l119_11942


namespace NUMINAMATH_GPT_chord_midpoint_line_l119_11983

open Real 

theorem chord_midpoint_line (x y : ℝ) (P : ℝ × ℝ) 
  (hP : P = (1, 1)) (hcircle : ∀ (x y : ℝ), x^2 + y^2 = 10) :
  x + y - 2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_chord_midpoint_line_l119_11983


namespace NUMINAMATH_GPT_smallest_pos_d_l119_11952

theorem smallest_pos_d (d : ℕ) (h : d > 0) (hd : ∃ k : ℕ, 3150 * d = k * k) : d = 14 := 
by 
  sorry

end NUMINAMATH_GPT_smallest_pos_d_l119_11952


namespace NUMINAMATH_GPT_factorize_expression_l119_11906

theorem factorize_expression (a : ℝ) : a^3 - 4 * a^2 + 4 * a = a * (a - 2)^2 := 
by
  sorry

end NUMINAMATH_GPT_factorize_expression_l119_11906


namespace NUMINAMATH_GPT_find_essay_pages_l119_11971

/-
Conditions:
1. It costs $0.10 to print one page.
2. Jenny wants to print 7 copies of her essay.
3. Jenny wants to buy 7 pens that each cost $1.50.
4. Jenny pays the store with 2 twenty dollar bills and gets $12 in change.
-/

def cost_per_page : Float := 0.10
def number_of_copies : Nat := 7
def cost_per_pen : Float := 1.50
def number_of_pens : Nat := 7
def total_money_given : Float := 40.00  -- 2 twenty dollar bills
def change_received : Float := 12.00

theorem find_essay_pages :
  let total_spent := total_money_given - change_received
  let total_cost_of_pens := Float.ofNat number_of_pens * cost_per_pen
  let total_amount_spent_on_printing := total_spent - total_cost_of_pens
  let number_of_pages := total_amount_spent_on_printing / cost_per_page
  number_of_pages = 175 := by
  sorry

end NUMINAMATH_GPT_find_essay_pages_l119_11971


namespace NUMINAMATH_GPT_price_decrease_percentage_l119_11927

theorem price_decrease_percentage (original_price : ℝ) :
  let first_sale_price := (4/5) * original_price
  let second_sale_price := (1/2) * original_price
  let decrease := first_sale_price - second_sale_price
  let percentage_decrease := (decrease / first_sale_price) * 100
  percentage_decrease = 37.5 := by
  sorry

end NUMINAMATH_GPT_price_decrease_percentage_l119_11927


namespace NUMINAMATH_GPT_range_of_k_l119_11976

theorem range_of_k 
  (x1 x2 y1 y2 k : ℝ)
  (h1 : y1 = 2 * x1 - k * x1 + 1)
  (h2 : y2 = 2 * x2 - k * x2 + 1)
  (h3 : x1 ≠ x2)
  (h4 : (x1 - x2) * (y1 - y2) < 0) : k > 2 := 
sorry

end NUMINAMATH_GPT_range_of_k_l119_11976


namespace NUMINAMATH_GPT_find_A_l119_11930

theorem find_A (A : ℕ) (h : 10 * A + 2 - 23 = 549) : A = 5 :=
by sorry

end NUMINAMATH_GPT_find_A_l119_11930


namespace NUMINAMATH_GPT_infinitely_many_triples_l119_11924

theorem infinitely_many_triples (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : ∀ k : ℕ, 
  ∃ (x y z : ℕ), 
    x = 2^(k * m * n + 1) ∧ 
    y = 2^(n + n * k * (m * n + 1)) ∧ 
    z = 2^(m + m * k * (m * n + 1)) ∧ 
    x^(m * n + 1) = y^m + z^n := 
by 
  intros k
  use 2^(k * m * n + 1), 2^(n + n * k * (m * n + 1)), 2^(m + m * k * (m * n + 1))
  simp
  sorry

end NUMINAMATH_GPT_infinitely_many_triples_l119_11924


namespace NUMINAMATH_GPT_complementary_not_supplementary_l119_11934

theorem complementary_not_supplementary (α β : ℝ) (h₁ : α + β = 90) (h₂ : α + β ≠ 180) : (α + β = 180) = false :=
by 
  sorry

end NUMINAMATH_GPT_complementary_not_supplementary_l119_11934


namespace NUMINAMATH_GPT_union_of_sets_l119_11917

def A (x : ℤ) : Set ℤ := {x^2, 2*x - 1, -4}
def B (x : ℤ) : Set ℤ := {x - 5, 1 - x, 9}

theorem union_of_sets (x : ℤ) (hx : x = -3) (h_inter : A x ∩ B x = {9}) :
  A x ∪ B x = {-8, -4, 4, -7, 9} :=
by
  sorry

end NUMINAMATH_GPT_union_of_sets_l119_11917


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l119_11912

variable (a : ℕ → ℝ)

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (d : ℝ), ∀ (n : ℕ), a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h_arith : is_arithmetic_sequence a) 
  (h_a3 : a 3 = 4) (h_a101 : a 101 = 36) : 
  a 9 + a 52 + a 95 = 60 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l119_11912


namespace NUMINAMATH_GPT_james_muffins_l119_11932

theorem james_muffins (arthur_muffins : ℕ) (times : ℕ) (james_muffins : ℕ) 
  (h1 : arthur_muffins = 115) 
  (h2 : times = 12) 
  (h3 : james_muffins = arthur_muffins * times) : 
  james_muffins = 1380 := 
by 
  sorry

end NUMINAMATH_GPT_james_muffins_l119_11932


namespace NUMINAMATH_GPT_fraction_of_cream_in_cup1_l119_11911

/-
Problem statement:
Sarah places five ounces of coffee into an eight-ounce cup (Cup 1) and five ounces of cream into a second cup (Cup 2).
After pouring half the coffee from Cup 1 to Cup 2, one ounce of cream is added to Cup 2.
After stirring Cup 2 thoroughly, Sarah then pours half the liquid in Cup 2 back into Cup 1.
Prove that the fraction of the liquid in Cup 1 that is now cream is 4/9.
-/

theorem fraction_of_cream_in_cup1
  (initial_coffee_cup1 : ℝ)
  (initial_cream_cup2 : ℝ)
  (half_initial_coffee : ℝ)
  (added_cream : ℝ)
  (total_mixture : ℝ)
  (half_mixture : ℝ)
  (coffee_fraction : ℝ)
  (cream_fraction : ℝ)
  (coffee_transferred_back : ℝ)
  (cream_transferred_back : ℝ)
  (total_coffee_in_cup1 : ℝ)
  (total_cream_in_cup1 : ℝ)
  (total_liquid_in_cup1 : ℝ)
  :
  initial_coffee_cup1 = 5 →
  initial_cream_cup2 = 5 →
  half_initial_coffee = initial_coffee_cup1 / 2 →
  added_cream = 1 →
  total_mixture = initial_cream_cup2 + half_initial_coffee + added_cream →
  half_mixture = total_mixture / 2 →
  coffee_fraction = half_initial_coffee / total_mixture →
  cream_fraction = (total_mixture - half_initial_coffee) / total_mixture →
  coffee_transferred_back = half_mixture * coffee_fraction →
  cream_transferred_back = half_mixture * cream_fraction →
  total_coffee_in_cup1 = initial_coffee_cup1 - half_initial_coffee + coffee_transferred_back →
  total_cream_in_cup1 = cream_transferred_back →
  total_liquid_in_cup1 = total_coffee_in_cup1 + total_cream_in_cup1 →
  total_cream_in_cup1 / total_liquid_in_cup1 = 4 / 9 :=
by {
  sorry
}

end NUMINAMATH_GPT_fraction_of_cream_in_cup1_l119_11911


namespace NUMINAMATH_GPT_mark_current_trees_l119_11988

theorem mark_current_trees (x : ℕ) (h : x + 12 = 25) : x = 13 :=
by {
  -- proof omitted
  sorry
}

end NUMINAMATH_GPT_mark_current_trees_l119_11988


namespace NUMINAMATH_GPT_min_value_of_expression_l119_11964

variable (a b c : ℝ)
variable (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
variable (h_eq : a * (a + b + c) + b * c = 4 - 2 * Real.sqrt 3)

theorem min_value_of_expression :
  2 * a + b + c ≥ 2 * Real.sqrt 3 - 2 := by
  sorry

end NUMINAMATH_GPT_min_value_of_expression_l119_11964


namespace NUMINAMATH_GPT_part1_f_inequality_part2_a_range_l119_11901

open Real

-- Proof Problem 1
theorem part1_f_inequality (x : ℝ) : 
    (|x - 1| + |x + 1| ≥ 3 ↔ x ≤ -1.5 ∨ x ≥ 1.5) :=
sorry

-- Proof Problem 2
theorem part2_a_range (a : ℝ) : 
    (∀ x : ℝ, |x - 1| + |x - a| ≥ 2) ↔ (a = 3 ∨ a = -1) :=
sorry

end NUMINAMATH_GPT_part1_f_inequality_part2_a_range_l119_11901


namespace NUMINAMATH_GPT_problem1_question_problem1_contrapositive_problem1_negation_problem2_question_problem2_contrapositive_problem2_negation_l119_11994

-- Proof statement for problem 1

def is_odd (n : ℕ) : Prop := n % 2 = 1

theorem problem1_question (x y : ℕ) (h : ¬(is_odd x ∧ is_odd y)) : is_odd (x + y) := sorry

theorem problem1_contrapositive (x y : ℕ) (h : is_odd x ∧ is_odd y) : ¬ is_odd (x + y) := sorry

theorem problem1_negation : ∃ (x y : ℕ), ¬(is_odd x ∧ is_odd y) ∧ ¬ is_odd (x + y) := sorry

-- Proof statement for problem 2

structure Square : Type := (is_rhombus : Prop)

def all_squares_are_rhombuses : Prop := ∀ (sq : Square), sq.is_rhombus

theorem problem2_question : all_squares_are_rhombuses = true := sorry

theorem problem2_contrapositive : ¬ all_squares_are_rhombuses = false := sorry

theorem problem2_negation : ¬(∃ (sq : Square), ¬ sq.is_rhombus) = false := sorry

end NUMINAMATH_GPT_problem1_question_problem1_contrapositive_problem1_negation_problem2_question_problem2_contrapositive_problem2_negation_l119_11994
