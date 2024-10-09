import Mathlib

namespace rattlesnakes_count_l1196_119630

theorem rattlesnakes_count (total_snakes : ℕ) (boa_constrictors pythons rattlesnakes : ℕ)
  (h1 : total_snakes = 200)
  (h2 : boa_constrictors = 40)
  (h3 : pythons = 3 * boa_constrictors)
  (h4 : total_snakes = boa_constrictors + pythons + rattlesnakes) :
  rattlesnakes = 40 :=
by
  sorry

end rattlesnakes_count_l1196_119630


namespace silver_cost_l1196_119600

theorem silver_cost (S : ℝ) : 
  (1.5 * S) + (3 * 50 * S) = 3030 → S = 20 :=
by
  intro h
  sorry

end silver_cost_l1196_119600


namespace odd_n_cubed_plus_23n_divisibility_l1196_119664

theorem odd_n_cubed_plus_23n_divisibility (n : ℤ) (h1 : n % 2 = 1) : (n^3 + 23 * n) % 24 = 0 := 
by 
  sorry

end odd_n_cubed_plus_23n_divisibility_l1196_119664


namespace minimum_number_of_guests_l1196_119680

theorem minimum_number_of_guests :
  ∀ (total_food : ℝ) (max_food_per_guest : ℝ), total_food = 411 → max_food_per_guest = 2.5 →
  ⌈total_food / max_food_per_guest⌉ = 165 :=
by
  intros total_food max_food_per_guest h1 h2
  rw [h1, h2]
  norm_num
  sorry

end minimum_number_of_guests_l1196_119680


namespace part1_part2_part3_l1196_119666

section Part1

variables (a b : Real)

theorem part1 : 2 * (a + b)^2 - 8 * (a + b)^2 + 3 * (a + b)^2 = -3 * (a + b)^2 :=
by
  sorry

end Part1

section Part2

variables (x y : Real)

theorem part2 (h : x^2 + 2 * y = 4) : -3 * x^2 - 6 * y + 17 = 5 :=
by
  sorry

end Part2

section Part3

variables (a b c d : Real)

theorem part3 (h1 : a - 3 * b = 3) (h2 : 2 * b - c = -5) (h3 : c - d = 9) :
  (a - c) + (2 * b - d) - (2 * b - c) = 7 :=
by
  sorry

end Part3

end part1_part2_part3_l1196_119666


namespace area_of_triangle_l1196_119699

theorem area_of_triangle {a c : ℝ} (B : ℝ) (h1 : a = 1) (h2 : c = 2) (h3 : B = 60) :
    (1 / 2) * a * c * Real.sin (B * Real.pi / 180) = Real.sqrt 3 / 2 := by
  sorry

end area_of_triangle_l1196_119699


namespace g_g_g_25_l1196_119607

noncomputable def g (x : ℝ) : ℝ :=
  if x < 10 then x^2 - 9 else x - 18

theorem g_g_g_25 :
  g (g (g 25)) = 22 :=
by
  sorry

end g_g_g_25_l1196_119607


namespace smallest_angle_WYZ_l1196_119649

-- Define the given angle measures.
def angle_XYZ : ℝ := 40
def angle_XYW : ℝ := 15

-- The theorem statement proving the smallest possible degree measure for ∠WYZ
theorem smallest_angle_WYZ : angle_XYZ - angle_XYW = 25 :=
by
  -- Add the proof here
  sorry

end smallest_angle_WYZ_l1196_119649


namespace tanA_tanB_eq_thirteen_div_four_l1196_119675

-- Define the triangle and its properties
variables {A B C : Type}
variables (a b c : ℝ)  -- sides BC, AC, AB
variables (HF HC : ℝ)  -- segments of altitude CF
variables (tanA tanB : ℝ)

-- Given conditions
def orthocenter_divide_altitude (HF HC : ℝ) : Prop :=
  HF = 8 ∧ HC = 18

-- The result we want to prove
theorem tanA_tanB_eq_thirteen_div_four (h : orthocenter_divide_altitude HF HC) : 
  tanA * tanB = 13 / 4 :=
  sorry

end tanA_tanB_eq_thirteen_div_four_l1196_119675


namespace odd_function_decreasing_l1196_119624

theorem odd_function_decreasing (f : ℝ → ℝ) (h1 : ∀ x, f (-x) = -f x) (h2 : ∀ x y, x < y → y < 0 → f x > f y) :
  ∀ x y, 0 < x → x < y → f y < f x :=
by
  sorry

end odd_function_decreasing_l1196_119624


namespace pentagon_area_l1196_119642

/-- Given a convex pentagon ABCDE where BE and CE are angle bisectors at vertices B and C 
respectively, with ∠A = 35 degrees, ∠D = 145 degrees, and the area of triangle BCE is 11, 
prove that the area of the pentagon ABCDE is 22. -/
theorem pentagon_area (ABCDE : Type) (angle_A : ℝ) (angle_D : ℝ) (area_BCE : ℝ)
  (h_A : angle_A = 35) (h_D : angle_D = 145) (h_area_BCE : area_BCE = 11) :
  ∃ (area_ABCDE : ℝ), area_ABCDE = 22 :=
by
  sorry

end pentagon_area_l1196_119642


namespace unique_solution_h_l1196_119644

theorem unique_solution_h (h : ℝ) (hne_zero : h ≠ 0) :
  (∃! x : ℝ, (x - 3) / (h * x + 2) = x) ↔ h = 1 / 12 :=
by
  sorry

end unique_solution_h_l1196_119644


namespace stratified_sample_over_30_l1196_119620

-- Define the total number of employees and conditions
def total_employees : ℕ := 49
def employees_over_30 : ℕ := 14
def employees_30_or_younger : ℕ := 35
def sample_size : ℕ := 7

-- State the proportion and the final required count
def proportion_over_30 (total : ℕ) (over_30 : ℕ) : ℚ := (over_30 : ℚ) / (total : ℚ)
def required_count (proportion : ℚ) (sample : ℕ) : ℚ := proportion * (sample : ℚ)

theorem stratified_sample_over_30 :
  required_count (proportion_over_30 total_employees employees_over_30) sample_size = 2 := 
by sorry

end stratified_sample_over_30_l1196_119620


namespace percentage_seniors_with_cars_is_40_l1196_119619

noncomputable def percentage_of_seniors_with_cars 
  (total_students: ℕ) (seniors: ℕ) (lower_grades: ℕ) (percent_cars_all: ℚ) (percent_cars_lower_grades: ℚ) : ℚ :=
  let total_with_cars := percent_cars_all * total_students
  let lower_grades_with_cars := percent_cars_lower_grades * lower_grades
  let seniors_with_cars := total_with_cars - lower_grades_with_cars
  (seniors_with_cars / seniors) * 100

theorem percentage_seniors_with_cars_is_40
  : percentage_of_seniors_with_cars 1800 300 1500 0.15 0.10 = 40 := 
by
  -- Proof is omitted
  sorry

end percentage_seniors_with_cars_is_40_l1196_119619


namespace average_player_time_l1196_119694

theorem average_player_time:
  let pg := 130
  let sg := 145
  let sf := 85
  let pf := 60
  let c := 180
  let total_secs := pg + sg + sf + pf + c
  let total_mins := total_secs / 60
  let num_players := 5
  let avg_mins_per_player := total_mins / num_players
  avg_mins_per_player = 2 :=
by
  sorry

end average_player_time_l1196_119694


namespace max_value_of_function_l1196_119631

open Real 

theorem max_value_of_function : ∀ x : ℝ, 
  cos (2 * x) + 6 * cos (π / 2 - x) ≤ 5 ∧ 
  ∃ x' : ℝ, cos (2 * x') + 6 * cos (π / 2 - x') = 5 :=
by 
  sorry

end max_value_of_function_l1196_119631


namespace denominator_of_second_fraction_l1196_119632

theorem denominator_of_second_fraction :
  let a := 2007
  let b := 2999
  let c := 8001
  let d := 2001
  let e := 3999
  let sum := 3.0035428163476343
  let first_fraction := (2007 : ℝ) / 2999
  let third_fraction := (2001 : ℝ) / 3999
  ∃ x : ℤ, (first_fraction + (8001 : ℝ) / x + third_fraction) = 3.0035428163476343 ∧ x = 4362 := 
by
  sorry

end denominator_of_second_fraction_l1196_119632


namespace remaining_watermelons_l1196_119688

def initial_watermelons : ℕ := 4
def eaten_watermelons : ℕ := 3

theorem remaining_watermelons : initial_watermelons - eaten_watermelons = 1 :=
by sorry

end remaining_watermelons_l1196_119688


namespace solution_set_of_f_double_exp_inequality_l1196_119616

theorem solution_set_of_f_double_exp_inequality (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, -2 < x ∧ x < 1 ↔ 0 < f x) :
  {x : ℝ | f (2^x) < 0} = {x : ℝ | x > 0} :=
sorry

end solution_set_of_f_double_exp_inequality_l1196_119616


namespace negation_of_exisential_inequality_l1196_119601

open Classical

theorem negation_of_exisential_inequality :
  ¬ (∃ x : ℝ, x^2 - x + 1/4 ≤ 0) ↔ ∀ x : ℝ, x^2 - x + 1/4 > 0 := 
by 
sorry

end negation_of_exisential_inequality_l1196_119601


namespace prob_correct_last_digit_no_more_than_two_attempts_prob_correct_last_digit_no_more_than_two_attempts_if_even_l1196_119696

/-
Prove that if a person forgets the last digit of their 6-digit password, which can be any digit from 0 to 9,
the probability of pressing the correct last digit in no more than 2 attempts is 1/5.
-/

theorem prob_correct_last_digit_no_more_than_two_attempts :
  let correct_prob := 1 / 10 
  let incorrect_prob := 9 / 10 
  let second_attempt_prob := 1 / 9 
  correct_prob + (incorrect_prob * second_attempt_prob) = 1 / 5 :=
by
  sorry

/-
Prove that if a person forgets the last digit of their 6-digit password, but remembers that the last digit is an even number,
the probability of pressing the correct last digit in no more than 2 attempts is 2/5.
-/

theorem prob_correct_last_digit_no_more_than_two_attempts_if_even :
  let correct_prob := 1 / 5 
  let incorrect_prob := 4 / 5 
  let second_attempt_prob := 1 / 4 
  correct_prob + (incorrect_prob * second_attempt_prob) = 2 / 5 :=
by
  sorry

end prob_correct_last_digit_no_more_than_two_attempts_prob_correct_last_digit_no_more_than_two_attempts_if_even_l1196_119696


namespace Jason_spent_on_jacket_l1196_119606

/-
Given:
- Amount_spent_on_shorts: ℝ := 14.28
- Total_spent_on_clothing: ℝ := 19.02

Prove:
- Amount_spent_on_jacket = 4.74
-/
def Amount_spent_on_shorts : ℝ := 14.28
def Total_spent_on_clothing : ℝ := 19.02

-- We need to prove:
def Amount_spent_on_jacket : ℝ := Total_spent_on_clothing - Amount_spent_on_shorts 

theorem Jason_spent_on_jacket : Amount_spent_on_jacket = 4.74 := by
  sorry

end Jason_spent_on_jacket_l1196_119606


namespace a_b_sum_possible_values_l1196_119687

theorem a_b_sum_possible_values (a b : ℝ) 
  (h1 : a^3 - 12 * a^2 + 9 * a - 18 = 0)
  (h2 : 9 * b^3 - 135 * b^2 + 450 * b - 1650 = 0) :
  a + b = 6 ∨ a + b = 14 :=
sorry

end a_b_sum_possible_values_l1196_119687


namespace max_height_l1196_119647

noncomputable def ball_height (t : ℝ) : ℝ :=
  -4.9 * t^2 + 50 * t + 15

theorem max_height : ∃ t : ℝ, t < 50 / 4.9 ∧ ball_height t = 142.65 :=
sorry

end max_height_l1196_119647


namespace beads_to_remove_l1196_119667

-- Definitions for the conditions given in the problem
def initial_blue_beads : Nat := 49
def initial_red_bead : Nat := 1
def total_initial_beads : Nat := initial_blue_beads + initial_red_bead
def target_blue_percentage : Nat := 90 -- percentage

-- The goal to prove
theorem beads_to_remove (initial_blue_beads : Nat) (initial_red_bead : Nat)
    (target_blue_percentage : Nat) : Nat :=
    let target_total_beads := (initial_red_bead * 100) / target_blue_percentage
    total_initial_beads - target_total_beads
-- Expected: beads_to_remove 49 1 90 = 40

example : beads_to_remove initial_blue_beads initial_red_bead target_blue_percentage = 40 := by 
    sorry

end beads_to_remove_l1196_119667


namespace will_buy_toys_l1196_119605

theorem will_buy_toys : 
  ∀ (initialMoney spentMoney toyCost : ℕ), 
  initialMoney = 83 → spentMoney = 47 → toyCost = 4 → 
  (initialMoney - spentMoney) / toyCost = 9 :=
by
  intros initialMoney spentMoney toyCost hInit hSpent hCost
  sorry

end will_buy_toys_l1196_119605


namespace tens_digit_of_9_pow_1010_l1196_119686

theorem tens_digit_of_9_pow_1010 : (9 ^ 1010) % 100 = 1 :=
by sorry

end tens_digit_of_9_pow_1010_l1196_119686


namespace shaded_area_correct_l1196_119678

noncomputable def shaded_area (side_large side_small : ℝ) (pi_value : ℝ) : ℝ :=
  let area_large_square := side_large^2
  let area_large_circle := pi_value * (side_large / 2)^2
  let area_large_heart := area_large_square + area_large_circle
  let area_small_square := side_small^2
  let area_small_circle := pi_value * (side_small / 2)^2
  let area_small_heart := area_small_square + area_small_circle
  area_large_heart - area_small_heart

theorem shaded_area_correct : shaded_area 40 20 3.14 = 2142 :=
by
  -- Proof goes here
  sorry

end shaded_area_correct_l1196_119678


namespace grandson_age_l1196_119652

-- Define the ages of Markus, his son, and his grandson
variables (M S G : ℕ)

-- Conditions given in the problem
axiom h1 : M = 2 * S
axiom h2 : S = 2 * G
axiom h3 : M + S + G = 140

-- Theorem to prove that the age of Markus's grandson is 20 years
theorem grandson_age : G = 20 :=
by
  sorry

end grandson_age_l1196_119652


namespace simplify_expression_l1196_119651

variable (q : ℚ)

theorem simplify_expression :
  (2 * q^3 - 7 * q^2 + 3 * q - 4) + (5 * q^2 - 4 * q + 8) = 2 * q^3 - 2 * q^2 - q + 4 :=
by
  sorry

end simplify_expression_l1196_119651


namespace domain_of_sqrt_fn_l1196_119602

theorem domain_of_sqrt_fn : {x : ℝ | -2 ≤ x ∧ x ≤ 2} = {x : ℝ | 4 - x^2 ≥ 0} := 
by sorry

end domain_of_sqrt_fn_l1196_119602


namespace gcd_of_17934_23526_51774_l1196_119665

-- Define the three integers
def a : ℕ := 17934
def b : ℕ := 23526
def c : ℕ := 51774

-- State the theorem
theorem gcd_of_17934_23526_51774 : Int.gcd a (Int.gcd b c) = 2 := by
  sorry

end gcd_of_17934_23526_51774_l1196_119665


namespace find_N_l1196_119613

def f (N : ℕ) : ℕ :=
  if N % 2 = 0 then 5 * N else 3 * N + 2

theorem find_N (N : ℕ) :
  f (f (f (f (f N)))) = 542 ↔ N = 112500 := by
  sorry

end find_N_l1196_119613


namespace magic_grid_product_l1196_119627

theorem magic_grid_product (p q r s t x : ℕ) 
  (h1: p * 6 * 3 = q * r * s)
  (h2: p * q * t = 6 * r * 2)
  (h3: p * r * x = 6 * 2 * t)
  (h4: q * 2 * 3 = r * s * x)
  (h5: t * 2 * x = 6 * s * 3)
  (h6: 6 * q * 3 = r * s * t)
  (h7: p * r * s = 6 * 2 * q)
  : x = 36 := 
by
  sorry

end magic_grid_product_l1196_119627


namespace harmonic_point_P_3_m_harmonic_point_hyperbola_l1196_119621

-- Part (1)
theorem harmonic_point_P_3_m (t : ℝ) (m : ℝ) (P : ℝ × ℝ → Prop)
  (h₁ : P ⟨ 3, m ⟩)
  (h₂ : ∀ x y, P ⟨ x, y ⟩ ↔ (x^2 = 4*y + t ∧ y^2 = 4*x + t ∧ x ≠ y)) :
  m = -7 :=
by sorry

-- Part (2)
theorem harmonic_point_hyperbola (k : ℝ) (P : ℝ × ℝ → Prop)
  (h_hb : ∀ x, -3 < x ∧ x < -1 → P ⟨ x, k / x ⟩)
  (h₂ : ∀ x y, P ⟨ x, y ⟩ ↔ (x^2 = 4*y + t ∧ y^2 = 4*x + t ∧ x ≠ y)) :
  3 < k ∧ k < 4 :=
by sorry

end harmonic_point_P_3_m_harmonic_point_hyperbola_l1196_119621


namespace quadratic_general_form_l1196_119648

theorem quadratic_general_form (x : ℝ) :
    (x + 3)^2 = x * (3 * x - 1) →
    2 * x^2 - 7 * x - 9 = 0 :=
by
  intros h
  sorry

end quadratic_general_form_l1196_119648


namespace number_of_boys_in_class_l1196_119697

theorem number_of_boys_in_class (n : ℕ) (h : 182 * n - 166 + 106 = 180 * n) : n = 30 :=
by {
  sorry
}

end number_of_boys_in_class_l1196_119697


namespace probability_of_friends_in_same_lunch_group_l1196_119635

theorem probability_of_friends_in_same_lunch_group :
  let groups := 4
  let students := 720
  let group_size := students / groups
  let probability := (1 / groups) * (1 / groups) * (1 / groups)
  students % groups = 0 ->  -- Students can be evenly divided into groups
  groups > 0 ->             -- There is at least one group
  probability = (1 : ℝ) / 64 :=
by
  intros
  sorry

end probability_of_friends_in_same_lunch_group_l1196_119635


namespace find_t_l1196_119658

-- Definitions of the vectors involved
def vector_AB : ℝ × ℝ := (2, 3)
def vector_AC (t : ℝ) : ℝ × ℝ := (3, t)
def vector_BC (t : ℝ) : ℝ × ℝ := ((vector_AC t).1 - (vector_AB).1, (vector_AC t).2 - (vector_AB).2)

-- Condition for orthogonality
def is_perpendicular (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

-- Main statement to be proved
theorem find_t : ∃ t : ℝ, is_perpendicular vector_AB (vector_BC t) ∧ t = 7 / 3 :=
by
  sorry

end find_t_l1196_119658


namespace inequality_solution_l1196_119679

theorem inequality_solution (x : ℝ) (h : 3 * x - 5 > 11 - 2 * x) : x > 16 / 5 := 
sorry

end inequality_solution_l1196_119679


namespace subset_implies_range_of_a_l1196_119615

theorem subset_implies_range_of_a (a : ℝ) : 
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 5 → x > a) → a < -2 :=
by
  intro h
  sorry

end subset_implies_range_of_a_l1196_119615


namespace brick_weight_l1196_119634

theorem brick_weight (b s : ℕ) (h1 : 5 * b = 4 * s) (h2 : 2 * s = 80) : b = 32 :=
by {
  sorry
}

end brick_weight_l1196_119634


namespace max_consecutive_integers_sum_48_l1196_119603

-- Define the sum of consecutive integers
def sum_consecutive_integers (a N : ℤ) : ℤ :=
  (N * (2 * a + N - 1)) / 2

-- Define the main theorem
theorem max_consecutive_integers_sum_48 : 
  ∃ N a : ℤ, sum_consecutive_integers a N = 48 ∧ (∀ N' : ℤ, ((N' * (2 * a + N' - 1)) / 2 = 48) → N' ≤ N) :=
sorry

end max_consecutive_integers_sum_48_l1196_119603


namespace lcm_18_24_l1196_119673

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  have fact_18 : 18 = 2 * 3^2 := by norm_num
  have fact_24 : 24 = 2^3 * 3 := by norm_num
  sorry

end lcm_18_24_l1196_119673


namespace vector_dot_product_l1196_119646

-- Definitions
def vec_a : ℝ × ℝ := (1, 3)
def vec_b : ℝ × ℝ := (-2, -1)

-- Theorem to prove
theorem vector_dot_product : 
  ((vec_a.1 + vec_b.1, vec_a.2 + vec_b.2) : ℝ × ℝ) • (2 * vec_a.1 + vec_b.1, 2 * vec_a.2 + vec_b.2) = 10 :=
by
  sorry

end vector_dot_product_l1196_119646


namespace continuity_at_x0_l1196_119689

noncomputable def f (x : ℝ) : ℝ := -4 * x^2 - 7

theorem continuity_at_x0 :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x - 1| < δ → |f x - f 1| < ε :=
by
  sorry

end continuity_at_x0_l1196_119689


namespace hats_in_shipment_l1196_119637

theorem hats_in_shipment (H : ℝ) (h_condition : 0.75 * H = 90) : H = 120 :=
sorry

end hats_in_shipment_l1196_119637


namespace geometric_sequence_fourth_term_l1196_119660

theorem geometric_sequence_fourth_term (x : ℝ) (r : ℝ) 
  (h1 : 3 * x + 3 = r * x)
  (h2 : 6 * x + 6 = r * (3 * x + 3)) :
  x = -3 ∧ r = 2 → (x * r^3 = -24) :=
by
  sorry

end geometric_sequence_fourth_term_l1196_119660


namespace p_is_necessary_but_not_sufficient_for_q_l1196_119650

variable (x : ℝ)
def p := |x| ≤ 2
def q := 0 ≤ x ∧ x ≤ 2

theorem p_is_necessary_but_not_sufficient_for_q : (∀ x, q x → p x) ∧ ∃ x, p x ∧ ¬ q x := by
  sorry

end p_is_necessary_but_not_sufficient_for_q_l1196_119650


namespace Polynomial_has_root_l1196_119653

noncomputable def P : ℝ → ℝ := sorry

variables (a1 a2 a3 b1 b2 b3 : ℝ)

axiom h1 : a1 * a2 * a3 ≠ 0
axiom h2 : ∀ x : ℝ, P (a1 * x + b1) + P (a2 * x + b2) = P (a3 * x + b3)

theorem Polynomial_has_root : ∃ x : ℝ, P x = 0 :=
sorry

end Polynomial_has_root_l1196_119653


namespace passengers_at_station_in_an_hour_l1196_119633

-- Define the conditions
def train_interval_minutes := 5
def passengers_off_per_train := 200
def passengers_on_per_train := 320

-- Define the time period we're considering
def time_period_minutes := 60

-- Calculate the expected values based on conditions
def expected_trains_per_hour := time_period_minutes / train_interval_minutes
def expected_passengers_off_per_hour := passengers_off_per_train * expected_trains_per_hour
def expected_passengers_on_per_hour := passengers_on_per_train * expected_trains_per_hour
def expected_total_passengers := expected_passengers_off_per_hour + expected_passengers_on_per_hour

theorem passengers_at_station_in_an_hour :
  expected_total_passengers = 6240 :=
by
  -- Structure of the proof omitted. Just ensuring conditions and expected value defined.
  sorry

end passengers_at_station_in_an_hour_l1196_119633


namespace width_of_second_square_is_seven_l1196_119626

-- The conditions translated into Lean definitions
def first_square : ℕ × ℕ := (8, 5)
def third_square : ℕ × ℕ := (5, 5)
def flag_dimensions : ℕ × ℕ := (15, 9)

-- The area calculation functions
def area (dim : ℕ × ℕ) : ℕ := dim.fst * dim.snd

-- Given areas for the first and third square
def area_first_square : ℕ := area first_square
def area_third_square : ℕ := area third_square

-- Desired flag area
def flag_area : ℕ := area flag_dimensions

-- Total area of first and third squares
def total_area_first_and_third : ℕ := area_first_square + area_third_square

-- Required area for the second square
def area_needed_second_square : ℕ := flag_area - total_area_first_and_third

-- Given length of the second square
def second_square_length : ℕ := 10

-- Solve for the width of the second square
def second_square_width : ℕ := area_needed_second_square / second_square_length

-- The proof goal
theorem width_of_second_square_is_seven : second_square_width = 7 := by
  sorry

end width_of_second_square_is_seven_l1196_119626


namespace john_weekly_allowance_l1196_119611

noncomputable def weekly_allowance (A : ℝ) :=
  (3/5) * A + (1/3) * ((2/5) * A) + 0.60 <= A

theorem john_weekly_allowance : ∃ A : ℝ, (3/5) * A + (1/3) * ((2/5) * A) + 0.60 = A := by
  let A := 2.25
  sorry

end john_weekly_allowance_l1196_119611


namespace people_attend_both_reunions_l1196_119614

theorem people_attend_both_reunions (N D H x : ℕ) 
  (hN : N = 50)
  (hD : D = 50)
  (hH : H = 60)
  (h_total : N = D + H - x) : 
  x = 60 :=
by
  sorry

end people_attend_both_reunions_l1196_119614


namespace problem_solution_l1196_119608

theorem problem_solution :
  (3012 - 2933)^2 / 196 = 32 := sorry

end problem_solution_l1196_119608


namespace units_digit_G1000_l1196_119643

def units_digit (n : ℕ) : ℕ :=
  n % 10

def power_cycle : List ℕ := [3, 9, 7, 1]

def G (n : ℕ) : ℕ :=
  3^(2^n) + 2

theorem units_digit_G1000 : units_digit (G 1000) = 3 :=
by
  sorry

end units_digit_G1000_l1196_119643


namespace altered_solution_contains_correct_detergent_volume_l1196_119693

-- Define the original and altered ratios.
def original_ratio : ℝ × ℝ × ℝ := (2, 25, 100)
def altered_ratio_bleach_to_detergent : ℝ × ℝ := (6, 25)
def altered_ratio_detergent_to_water : ℝ × ℝ := (25, 200)

-- Define the given condition about the amount of water in the altered solution.
def altered_solution_water_volume : ℝ := 300

-- Define a function for the total altered solution volume and detergent volume
noncomputable def altered_solution_detergent_volume (water_volume : ℝ) : ℝ :=
  let detergent_volume := (altered_ratio_detergent_to_water.1 * water_volume) / altered_ratio_detergent_to_water.2
  detergent_volume

-- The proof statement asserting the amount of detergent in the altered solution.
theorem altered_solution_contains_correct_detergent_volume :
  altered_solution_detergent_volume altered_solution_water_volume = 37.5 :=
by
  sorry

end altered_solution_contains_correct_detergent_volume_l1196_119693


namespace selina_sold_shirts_l1196_119669

/-- Selina's selling problem -/
theorem selina_sold_shirts :
  let pants_price := 5
  let shorts_price := 3
  let shirts_price := 4
  let num_pants := 3
  let num_shorts := 5
  let remaining_money := 30 + (2 * 10)
  let money_from_pants := num_pants * pants_price
  let money_from_shorts := num_shorts * shorts_price
  let total_money_from_pants_and_shorts := money_from_pants + money_from_shorts
  let total_money_from_shirts := remaining_money - total_money_from_pants_and_shorts
  let num_shirts := total_money_from_shirts / shirts_price
  num_shirts = 5 := by
{
  sorry
}

end selina_sold_shirts_l1196_119669


namespace wrapping_paper_area_l1196_119681

variable (l w h : ℝ)
variable (l_gt_w : l > w)

def area_wrapping_paper (l w h : ℝ) : ℝ :=
  3 * (l + w) * h

theorem wrapping_paper_area :
  area_wrapping_paper l w h = 3 * (l + w) * h :=
sorry

end wrapping_paper_area_l1196_119681


namespace sum_of_elements_in_T_l1196_119617

noncomputable def digit_sum : ℕ := (0 + 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9) * 504
noncomputable def repeating_sum : ℕ := digit_sum * 1111
noncomputable def sum_T : ℚ := repeating_sum / 9999

theorem sum_of_elements_in_T : sum_T = 2523 := by
  sorry

end sum_of_elements_in_T_l1196_119617


namespace condition1_not_sufficient_nor_necessary_condition2_necessary_l1196_119698

variable (x y : ℝ)

-- ① Neither sufficient nor necessary
theorem condition1_not_sufficient_nor_necessary (h1 : x ≠ 1 ∧ y ≠ 2) : ¬ ((x ≠ 1 ∧ y ≠ 2) → x + y ≠ 3) ∧ ¬ (x + y ≠ 3 → x ≠ 1 ∧ y ≠ 2) := sorry

-- ② Necessary condition
theorem condition2_necessary (h2 : x ≠ 1 ∨ y ≠ 2) : x + y ≠ 3 → (x ≠ 1 ∨ y ≠ 2) := sorry

end condition1_not_sufficient_nor_necessary_condition2_necessary_l1196_119698


namespace student_adjustment_l1196_119661

noncomputable def binomial (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def permutation (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

theorem student_adjustment : 
  let front_row_size := 4
  let back_row_size := 8
  let total_students := 12
  let num_to_select := 2
  let ways_to_select := binomial back_row_size num_to_select
  let ways_to_permute := permutation (front_row_size + num_to_select) num_to_select
  ways_to_select * ways_to_permute = 840 :=
  by {
    let front_row_size := 4
    let back_row_size := 8
    let total_students := 12
    let num_to_select := 2
    let ways_to_select := binomial back_row_size num_to_select
    let ways_to_permute := permutation (front_row_size + num_to_select) num_to_select
    exact sorry
  }

end student_adjustment_l1196_119661


namespace radian_measure_of_240_degrees_l1196_119636

theorem radian_measure_of_240_degrees : (240 * (π / 180) = 4 * π / 3) := by
  sorry

end radian_measure_of_240_degrees_l1196_119636


namespace find_y_l1196_119672

theorem find_y (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hrem : x % y = 5) (hdiv : (x : ℝ) / y = 96.2) : y = 25 := by
  sorry

end find_y_l1196_119672


namespace find_a7_l1196_119684

variable (a : ℕ → ℝ)

def arithmetic_sequence (d : ℝ) (a1 : ℝ) :=
  ∀ n, a n = a1 + (n - 1) * d

theorem find_a7
  (a : ℕ → ℝ)
  (d : ℝ)
  (a1 : ℝ)
  (h_arith : arithmetic_sequence a d a1)
  (h_a3 : a 3 = 7)
  (h_a5 : a 5 = 13):
  a 7 = 19 :=
by
  sorry

end find_a7_l1196_119684


namespace part1_part2_l1196_119612

noncomputable def f (a x : ℝ) := a * Real.log x - x / 2

theorem part1 (a : ℝ) : (∀ x, f a x = a * Real.log x - x / 2) → (∃ x, x = 2 ∧ deriv (f a) x = 0) → a = 1 :=
by sorry

theorem part2 (k : ℝ) : (∀ x, x > 1 → f 1 x + k / x < 0) → k ≤ 1 / 2 :=
by sorry

end part1_part2_l1196_119612


namespace purely_periodic_fraction_period_length_divisible_l1196_119609

noncomputable def purely_periodic_fraction (p q n : ℕ) : Prop :=
  ∃ (r : ℕ), 10 ^ n - 1 = r * q ∧ (∃ (k : ℕ), q * (10 ^ (n * k)) ∣ p)

theorem purely_periodic_fraction_period_length_divisible
  (p q n : ℕ) (hq : ¬ (2 ∣ q) ∧ ¬ (5 ∣ q)) (hpq : p < q) (hn : 10 ^ n - 1 ∣ q) :
  purely_periodic_fraction p q n :=
by
  sorry

end purely_periodic_fraction_period_length_divisible_l1196_119609


namespace Christine_savings_l1196_119674

theorem Christine_savings 
  (commission_rate: ℝ) 
  (total_sales: ℝ) 
  (personal_needs_percentage: ℝ) 
  (savings: ℝ) 
  (h1: commission_rate = 0.12) 
  (h2: total_sales = 24000) 
  (h3: personal_needs_percentage = 0.60) 
  (h4: savings = total_sales * commission_rate * (1 - personal_needs_percentage)) : 
  savings = 1152 := by 
  sorry

end Christine_savings_l1196_119674


namespace Peter_bought_5_kilos_of_cucumbers_l1196_119657

/-- 
Peter carried $500 to the market. 
He bought 6 kilos of potatoes for $2 per kilo, 
9 kilos of tomato for $3 per kilo, 
some kilos of cucumbers for $4 per kilo, 
and 3 kilos of bananas for $5 per kilo. 
After buying all these items, Peter has $426 remaining. 
How many kilos of cucumbers did Peter buy? 
-/
theorem Peter_bought_5_kilos_of_cucumbers : 
   ∃ (kilos_cucumbers : ℕ),
   (500 - (6 * 2 + 9 * 3 + 3 * 5 + kilos_cucumbers * 4) = 426) →
   kilos_cucumbers = 5 :=
sorry

end Peter_bought_5_kilos_of_cucumbers_l1196_119657


namespace projectile_reaches_45_feet_first_time_l1196_119625

theorem projectile_reaches_45_feet_first_time :
  ∃ t : ℝ, (-20 * t^2 + 90 * t = 45) ∧ abs (t - 0.9) < 0.1 := sorry

end projectile_reaches_45_feet_first_time_l1196_119625


namespace area_shaded_quad_correct_l1196_119670

-- Define the side lengths of the squares
def side_length_small : ℕ := 3
def side_length_middle : ℕ := 5
def side_length_large : ℕ := 7

-- Define the total base length
def total_base_length : ℕ := side_length_small + side_length_middle + side_length_large

-- The height of triangle T3, equal to the side length of the largest square
def height_T3 : ℕ := side_length_large

-- The height-to-base ratio for each triangle
def height_to_base_ratio : ℚ := height_T3 / total_base_length

-- The heights of T1 and T2
def height_T1 : ℚ := side_length_small * height_to_base_ratio
def height_T2 : ℚ := (side_length_small + side_length_middle) * height_to_base_ratio

-- The height of the trapezoid, which is the side length of the middle square
def trapezoid_height : ℕ := side_length_middle

-- The bases of the trapezoid
def base1 : ℚ := height_T1
def base2 : ℚ := height_T2

-- The area of the trapezoid formula
def area_shaded_quad : ℚ := (trapezoid_height * (base1 + base2)) / 2

-- Assertion that the area of the shaded quadrilateral is equal to 77/6
theorem area_shaded_quad_correct : area_shaded_quad = 77 / 6 := by sorry

end area_shaded_quad_correct_l1196_119670


namespace middle_number_l1196_119676

theorem middle_number (a b c : ℕ) (h1 : a < b) (h2 : b < c) 
  (h3 : a + b = 18) (h4 : a + c = 23) (h5 : b + c = 27) : b = 11 := by
  sorry

end middle_number_l1196_119676


namespace math_team_combinations_l1196_119671

def numGirls : ℕ := 4
def numBoys : ℕ := 7
def girlsToChoose : ℕ := 2
def boysToChoose : ℕ := 3

def comb (n k : ℕ) : ℕ := n.choose k

theorem math_team_combinations : 
  comb numGirls girlsToChoose * comb numBoys boysToChoose = 210 := 
by
  sorry

end math_team_combinations_l1196_119671


namespace cos_240_eq_neg_half_l1196_119629

theorem cos_240_eq_neg_half : Real.cos (240 * Real.pi / 180) = -1 / 2 :=
by
  -- Sorry to skip the proof
  sorry

end cos_240_eq_neg_half_l1196_119629


namespace inverse_proposition_of_square_positive_l1196_119663

theorem inverse_proposition_of_square_positive :
  (∀ x : ℝ, x < 0 → x^2 > 0) →
  (∀ x : ℝ, ¬ (x^2 > 0) → ¬ (x < 0)) :=
by
  intro h
  intros x h₁
  sorry

end inverse_proposition_of_square_positive_l1196_119663


namespace steve_marbles_after_trans_l1196_119618

def initial_marbles (S T L H : ℕ) : Prop :=
  S = 2 * T ∧
  L = S - 5 ∧
  H = T + 3

def transactions (S T L H : ℕ) (new_S new_T new_L new_H : ℕ) : Prop :=
  new_S = S - 10 ∧
  new_L = L - 4 ∧
  new_T = T + 4 ∧
  new_H = H - 6

theorem steve_marbles_after_trans (S T L H new_S new_T new_L new_H : ℕ) :
  initial_marbles S T L H →
  transactions S T L H new_S new_T new_L new_H →
  new_S = 6 →
  new_T = 12 :=
by
  sorry

end steve_marbles_after_trans_l1196_119618


namespace ROI_difference_is_correct_l1196_119622

noncomputable def compound_interest (P : ℝ) (rates : List ℝ) : ℝ :=
rates.foldl (λ acc rate => acc * (1 + rate)) P

noncomputable def Emma_investment := compound_interest 300 [0.15, 0.12, 0.18]

noncomputable def Briana_investment := compound_interest 500 [0.10, 0.08, 0.14]

noncomputable def ROI_difference := Briana_investment - Emma_investment

theorem ROI_difference_is_correct : ROI_difference = 220.808 := 
sorry

end ROI_difference_is_correct_l1196_119622


namespace student_age_is_17_in_1960_l1196_119628

noncomputable def student's_age_in_1960 (x y : ℕ) (hx : 0 ≤ x ∧ x < 10) (hy : 0 ≤ y ∧ y < 10) : ℕ := 
  let birth_year : ℕ := 1900 + 10 * x + y
  let age_in_1960 : ℕ := 1960 - birth_year
  age_in_1960

theorem student_age_is_17_in_1960 :
  ∃ x y : ℕ, 0 ≤ x ∧ x < 10 ∧ 0 ≤ y ∧ y < 10 ∧ (1960 - (1900 + 10 * x + y) = 1 + 9 + x + y) ∧ (1960 - (1900 + 10 * x + y) = 17) :=
by {
  sorry -- Proof goes here
}

end student_age_is_17_in_1960_l1196_119628


namespace sum_geometric_seq_eq_l1196_119623

-- Defining the parameters of the geometric sequence
def a : ℚ := 1 / 5
def r : ℚ := 2 / 5
def n : ℕ := 8

-- Required to prove the sum of the first eight terms equals the given fraction
theorem sum_geometric_seq_eq :
  (a * (1 - r^n) / (1 - r)) = (390369 / 1171875) :=
by
  -- Proof to be completed
  sorry

end sum_geometric_seq_eq_l1196_119623


namespace factor_expression_l1196_119677

theorem factor_expression (y : ℝ) : 
  3 * y * (2 * y + 5) + 4 * (2 * y + 5) = (3 * y + 4) * (2 * y + 5) :=
by
  sorry

end factor_expression_l1196_119677


namespace extracurricular_books_l1196_119683

theorem extracurricular_books (a b c d : ℕ) 
  (h1 : b + c + d = 110)
  (h2 : a + c + d = 108)
  (h3 : a + b + d = 104)
  (h4 : a + b + c = 119) :
  a = 37 ∧ b = 39 ∧ c = 43 ∧ d = 28 :=
by {
  -- Proof to be done here
  sorry
}

end extracurricular_books_l1196_119683


namespace Ryan_spit_distance_correct_l1196_119690

-- Definitions of given conditions
def Billy_spit_distance : ℝ := 30
def Madison_spit_distance : ℝ := Billy_spit_distance * 1.20
def Ryan_spit_distance : ℝ := Madison_spit_distance * 0.50

-- Goal statement
theorem Ryan_spit_distance_correct : Ryan_spit_distance = 18 := by
  -- proof would go here
  sorry

end Ryan_spit_distance_correct_l1196_119690


namespace find_r_l1196_119655

noncomputable def f (r a : ℝ) (x : ℝ) : ℝ := (x - r - 1) * (x - r - 8) * (x - a)
noncomputable def g (r b : ℝ) (x : ℝ) : ℝ := (x - r - 2) * (x - r - 9) * (x - b)

theorem find_r
  (r a b : ℝ)
  (h_condition1 : ∀ x, f r a x - g r b x = r)
  (h_condition2 : f r a (r + 2) = r)
  (h_condition3 : f r a (r + 9) = r)
  : r = -264 / 7 := sorry

end find_r_l1196_119655


namespace rtl_to_conventional_notation_l1196_119654

theorem rtl_to_conventional_notation (a b c d e : ℚ) :
  (a / (b - (c * (d + e)))) = a / (b - c * (d + e)) := by
  sorry

end rtl_to_conventional_notation_l1196_119654


namespace original_sequence_polynomial_of_degree_3_l1196_119662

def is_polynomial_of_degree (u : ℕ → ℤ) (n : ℕ) :=
  ∃ a b c d : ℤ, u n = a * n^3 + b * n^2 + c * n + d

def fourth_difference_is_zero (u : ℕ → ℤ) :=
  ∀ n : ℕ, (u (n + 4) - 4 * u (n + 3) + 6 * u (n + 2) - 4 * u (n + 1) + u n) = 0

theorem original_sequence_polynomial_of_degree_3 (u : ℕ → ℤ)
  (h : fourth_difference_is_zero u) : 
  ∃ (a b c d : ℤ), ∀ n : ℕ, u n = a * n^3 + b * n^2 + c * n + d := sorry

end original_sequence_polynomial_of_degree_3_l1196_119662


namespace odd_function_of_power_l1196_119640

noncomputable def f (a b x : ℝ) : ℝ := (a - 1) * x ^ b

theorem odd_function_of_power (a b : ℝ) (h : f a b a = 1/2) : 
  ∀ x : ℝ, f a b (-x) = -f a b x := 
by
  sorry

end odd_function_of_power_l1196_119640


namespace trip_total_time_trip_average_speed_l1196_119638

structure Segment where
  distance : ℝ -- in kilometers
  speed : ℝ -- average speed in km/hr
  break_time : ℝ -- in minutes

def seg1 := Segment.mk 12 13 15
def seg2 := Segment.mk 18 16 30
def seg3 := Segment.mk 25 20 45
def seg4 := Segment.mk 35 25 60
def seg5 := Segment.mk 50 22 0

noncomputable def total_time_minutes (segs : List Segment) : ℝ :=
  segs.foldl (λ acc s => acc + (s.distance / s.speed) * 60 + s.break_time) 0

noncomputable def total_distance (segs : List Segment) : ℝ :=
  segs.foldl (λ acc s => acc + s.distance) 0

noncomputable def overall_average_speed (segs : List Segment) : ℝ :=
  total_distance segs / (total_time_minutes segs / 60)

def segments := [seg1, seg2, seg3, seg4, seg5]

theorem trip_total_time : total_time_minutes segments = 568.24 := by sorry
theorem trip_average_speed : overall_average_speed segments = 14.78 := by sorry

end trip_total_time_trip_average_speed_l1196_119638


namespace ratio_of_sums_equiv_seven_eighths_l1196_119685

variable (p q r u v w : ℝ)
variable (hp : 0 < p) (hq : 0 < q) (hr : 0 < r)
variable (hu : 0 < u) (hv : 0 < v) (hw : 0 < w)
variable (h1 : p^2 + q^2 + r^2 = 49)
variable (h2 : u^2 + v^2 + w^2 = 64)
variable (h3 : p * u + q * v + r * w = 56)

theorem ratio_of_sums_equiv_seven_eighths :
  (p + q + r) / (u + v + w) = 7 / 8 :=
by
  sorry

end ratio_of_sums_equiv_seven_eighths_l1196_119685


namespace time_A_problems_60_l1196_119682

variable (t : ℕ) -- time in minutes per type B problem

def time_per_A_problem := 2 * t
def time_per_C_problem := t / 2
def total_time_for_A_problems := 20 * time_per_A_problem

theorem time_A_problems_60 (hC : 80 * time_per_C_problem = 60) : total_time_for_A_problems = 60 := by
  sorry

end time_A_problems_60_l1196_119682


namespace bike_ride_time_l1196_119645

theorem bike_ride_time (y : ℚ) : 
  let speed_fast := 25
  let speed_slow := 10
  let total_distance := 170
  let total_time := 10
  (speed_fast * y + speed_slow * (total_time - y) = total_distance) 
  → y = 14 / 3 := 
by 
  sorry

end bike_ride_time_l1196_119645


namespace area_of_circle_segment_l1196_119691

-- Definitions for the conditions in the problem
def circle_eq (x y : ℝ) : Prop := x^2 - 10 * x + y^2 = 9
def line_eq (x y : ℝ) : Prop := y = x - 5

-- The area of the portion of the circle that lies above the x-axis and to the left of the line y = x - 5
theorem area_of_circle_segment :
  let area_of_circle := 34 * Real.pi
  let portion_fraction := 1 / 8
  portion_fraction * area_of_circle = 4.25 * Real.pi :=
by
  sorry

end area_of_circle_segment_l1196_119691


namespace sets_of_three_teams_l1196_119695

-- Definitions based on the conditions
def total_teams : ℕ := 20
def won_games : ℕ := 12
def lost_games : ℕ := 7

-- Main theorem to prove
theorem sets_of_three_teams : 
  (total_teams * (total_teams - 1) * (total_teams - 2)) / 6 / 2 = 570 := by
  sorry

end sets_of_three_teams_l1196_119695


namespace find_a1_in_geometric_sequence_l1196_119656

noncomputable def geometric_sequence_first_term (a : ℕ → ℝ) (r : ℝ) (h : ∀ n : ℕ, a (n + 1) = a n * r) : ℝ :=
  a 0

theorem find_a1_in_geometric_sequence (a : ℕ → ℝ) (h_geo : ∀ n : ℕ, a (n + 1) = a n * (1 / 2)) :
  a 2 = 16 → a 3 = 8 → geometric_sequence_first_term a (1 / 2) h_geo = 64 :=
by
  intros h2 h3
  -- Proof would go here
  sorry

end find_a1_in_geometric_sequence_l1196_119656


namespace purchase_price_of_radio_l1196_119610

theorem purchase_price_of_radio 
  (selling_price : ℚ) (loss_percentage : ℚ) (purchase_price : ℚ) 
  (h1 : selling_price = 465.50)
  (h2 : loss_percentage = 0.05):
  purchase_price = 490 :=
by 
  sorry

end purchase_price_of_radio_l1196_119610


namespace johnny_weekly_earnings_l1196_119659

-- Define the conditions mentioned in the problem.
def number_of_dogs_at_once : ℕ := 3
def thirty_minute_walk_payment : ℝ := 15
def sixty_minute_walk_payment : ℝ := 20
def work_hours_per_day : ℝ := 4
def sixty_minute_walks_needed_per_day : ℕ := 6
def work_days_per_week : ℕ := 5

-- Prove Johnny's weekly earnings given the conditions
theorem johnny_weekly_earnings :
  let sixty_minute_walks_per_day := sixty_minute_walks_needed_per_day / number_of_dogs_at_once
  let sixty_minute_earnings_per_day := sixty_minute_walks_per_day * number_of_dogs_at_once * sixty_minute_walk_payment
  let remaining_hours_per_day := work_hours_per_day - sixty_minute_walks_per_day
  let thirty_minute_walks_per_day := remaining_hours_per_day * 2 -- each 30-minute walk takes 0.5 hours
  let thirty_minute_earnings_per_day := thirty_minute_walks_per_day * number_of_dogs_at_once * thirty_minute_walk_payment
  let daily_earnings := sixty_minute_earnings_per_day + thirty_minute_earnings_per_day
  let weekly_earnings := daily_earnings * work_days_per_week
  weekly_earnings = 1500 :=
by
  sorry

end johnny_weekly_earnings_l1196_119659


namespace circles_intersect_l1196_119668

theorem circles_intersect :
  ∀ (x y : ℝ),
    ((x^2 + y^2 - 2 * x + 4 * y + 1 = 0) →
    (x^2 + y^2 - 6 * x + 2 * y + 9 = 0) →
    (∃ c1 c2 r1 r2 d : ℝ,
      (x - 1)^2 + (y + 2)^2 = r1 ∧ r1 = 4 ∧
      (x - 3)^2 + (y + 1)^2 = r2 ∧ r2 = 1 ∧
      d = Real.sqrt ((3 - 1)^2 + (-1 + 2)^2) ∧
      d > abs (r1 - r2) ∧ d < (r1 + r2))) :=
sorry

end circles_intersect_l1196_119668


namespace crayons_in_drawer_before_l1196_119641

theorem crayons_in_drawer_before (m c : ℕ) (h1 : m = 3) (h2 : c = 10) : c - m = 7 := 
  sorry

end crayons_in_drawer_before_l1196_119641


namespace pencils_left_l1196_119604

theorem pencils_left (anna_pencils : ℕ) (harry_pencils : ℕ)
  (h_anna : anna_pencils = 50) (h_harry : harry_pencils = 2 * anna_pencils)
  (lost_pencils : ℕ) (h_lost : lost_pencils = 19) :
  harry_pencils - lost_pencils = 81 :=
by
  sorry

end pencils_left_l1196_119604


namespace quadratic_rewrite_l1196_119692

theorem quadratic_rewrite (x : ℝ) (b c : ℝ) : 
  (x^2 + 1560 * x + 2400 = (x + b)^2 + c) → 
  c / b = -300 :=
by
  sorry

end quadratic_rewrite_l1196_119692


namespace plane_equation_correct_l1196_119639

def plane_equation (x y z : ℝ) : ℝ := 10 * x - 5 * y + 4 * z - 141

noncomputable def gcd (a b c d : ℤ) : ℤ := Int.gcd (Int.gcd a b) (Int.gcd c d)

theorem plane_equation_correct :
  (∀ x y z, plane_equation x y z = 0 ↔ 10 * x - 5 * y + 4 * z - 141 = 0)
  ∧ gcd 10 (-5) 4 (-141) = 1
  ∧ 10 > 0 := by
  sorry

end plane_equation_correct_l1196_119639
